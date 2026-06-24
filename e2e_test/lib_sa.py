#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


@dataclass
class _Pending:
    kind: str                     # "weight" or "ifmap"
    key: Tuple[int, int]          # (tile_idx, 0) for weight, (period_id, slice_id) for ifmap
    remaining_bytes: int


def simulate_os_sa_full(
    *,
    S_row: int,
    S_col: int,
    tile_lengths: List[int],
    dram_bandwidth_words_per_cycle: float,
    bytes_per_word: int = 1,
    filter_buffer_capacity_bytes: int = 0,
    ifmap_buffer_capacity_bytes: int = 0,
    ofmap_buffer_capacity_elems: int = 0,
    ofmap_dram_bandwidth_words_per_cycle: float = 0.0,
    buffer_mode: str = "double",               # "double" or "quadratic"
    sa_mode: int = 0,                          # 0: general, 1: colcomb, 2: eureka, 3: ctc/opf
    mux_size: int = 1,
    timing_mode: str = "full",                 # "full" or "compute_only"
    # --- new / optional ---
    tag_bits_per_entry: int = 0,               # e.g., CTC stream bit = 1
    activation_reuse_tiles: int = 1,           # gn_tiles (row tiles per activation slice)
    repeat_period_tiles: Optional[int] = None, # len(base_tile_lengths) if tile_lengths repeated for M tiles
    ifmap_cols_per_slice: Optional[int] = None,# 원래 K_tile (activation slice 길이). None이면 max(tile_lengths)로 근사
) -> Dict[str, float]:
    """
    Returns dict with at least:
      - "ofmap_drain_done_time": 마지막 타일의 output register group 점유가 끝난 시각(사이클)
      - "injection_done_time": 마지막 타일의 주입이 끝난 시각(사이클)
      - "total_stall_cycles": DRAM 또는 output-group 대기로 인한 stall 합(근사)
    """
    if not tile_lengths:
        return {
            "ofmap_drain_done_time": 0.0,
            "injection_done_time": 0.0,
            "total_stall_cycles": 0.0,
        }

    # -----------------------------
    # output register group 개수
    # -----------------------------
    if buffer_mode.lower() in ("quadratic", "quad", "4", "four"):
        num_groups = 4
    else:
        num_groups = 2

    # -----------------------------
    # 대역폭/바이트 단위 정리
    # -----------------------------
    if bytes_per_word <= 0:
        raise ValueError("bytes_per_word must be positive.")
    if dram_bandwidth_words_per_cycle <= 0:
        # compute_only처럼 동작시키기 위해 "무한대역폭" 취급
        bw_bytes_per_cycle = float("inf")
    else:
        bw_bytes_per_cycle = float(dram_bandwidth_words_per_cycle) * float(bytes_per_word)

    # activation slice의 "열 길이" (원래 K_tile)
    if ifmap_cols_per_slice is None:
        # packed tile_lengths만 넣으면 과소추정될 수 있음.
        ifmap_cols_per_slice = max(int(x) for x in tile_lengths)

    if activation_reuse_tiles <= 0:
        activation_reuse_tiles = 1

    # repeat_period_tiles가 없으면 전체를 1개의 period로 봄
    if repeat_period_tiles is None or repeat_period_tiles <= 0:
        repeat_period_tiles = len(tile_lengths)

    # -----------------------------
    # 바이트 요구량 계산
    # -----------------------------
    def weight_bytes_for(k_len: int) -> int:
        # weight 엔트리: S_row * k_len, 엔트리당 bytes_per_word
        w_bytes = int(S_row) * int(k_len) * int(bytes_per_word)
        if tag_bits_per_entry > 0:
            tag_bits_total = int(S_row) * int(k_len) * int(tag_bits_per_entry)
            w_bytes += _ceil_div(tag_bits_total, 8)
        return w_bytes

    ifmap_bytes_per_slice = int(S_col) * int(ifmap_cols_per_slice) * int(bytes_per_word)

    weight_bytes = [weight_bytes_for(int(k)) for k in tile_lengths]

    # 간단한 capacity sanity check (넘으면 경고 수준으로만 처리)
    # (실제로는 버퍼를 여러 번 refill해야 해서 stall이 늘어날 수 있음)
    # 여기서는 "한 타일을 버퍼에 올려야 실행 가능" 가정이므로,
    # 너무 작은 capacity가 들어오면 실행이 불가능한 것으로 처리.
    if filter_buffer_capacity_bytes and any(b > filter_buffer_capacity_bytes for b in weight_bytes):
        raise ValueError(
            "filter_buffer_capacity_bytes is smaller than at least one tile's weight+tag bytes. "
            "Increase capacity or model multi-refill."
        )
    if ifmap_buffer_capacity_bytes and ifmap_bytes_per_slice > ifmap_buffer_capacity_bytes:
        raise ValueError(
            "ifmap_buffer_capacity_bytes is smaller than one activation slice bytes. "
            "Increase capacity or model tiling/refill."
        )

    # -----------------------------
    # 로딩 상태(타일 weight, activation slice)
    #   - activation은 period별로 다시 로딩됨
    # -----------------------------
    loaded_weight = [False] * len(tile_lengths)
    loaded_ifmap: Dict[Tuple[int, int], bool] = {}  # (period_id, slice_id) -> bool

    # pending queue는 최대 2개(다음 타일 weight, 다음 slice ifmap)
    pending: List[_Pending] = []

    def _pending_find(kind: str, key: Tuple[int, int]) -> Optional[int]:
        for idx, p in enumerate(pending):
            if p.kind == kind and p.key == key:
                return idx
        return None

    def _push_pending_front(p: _Pending) -> None:
        """weight를 더 우선시키기 위해 front 삽입."""
        pending.insert(0, p)

    def _push_pending_back(p: _Pending) -> None:
        pending.append(p)

    def _consume_dram(cycles: float) -> None:
        """주어진 cycles 동안 DRAM이 전송할 수 있는 바이트만큼 pending을 소모."""
        if cycles <= 0:
            return
        if bw_bytes_per_cycle == float("inf"):
            # 전부 즉시 완료
            for p in pending:
                p.remaining_bytes = 0
            pending.clear()
            return

        budget = int(math.floor(cycles * bw_bytes_per_cycle))
        # cycles가 정수 사이클일 걸 가정하지만, 안전하게 floor 사용
        while budget > 0 and pending:
            p0 = pending[0]
            take = min(budget, p0.remaining_bytes)
            p0.remaining_bytes -= take
            budget -= take
            if p0.remaining_bytes <= 0:
                # 완료 처리
                if p0.kind == "weight":
                    tile_idx = p0.key[0]
                    if 0 <= tile_idx < len(loaded_weight):
                        loaded_weight[tile_idx] = True
                elif p0.kind == "ifmap":
                    loaded_ifmap[p0.key] = True
                pending.pop(0)

    # -----------------------------
    # 시뮬레이션 상태
    # -----------------------------
    group_free_time = [0.0] * num_groups  # group별 누산 컨텍스트가 비는 시각
    t = 0.0                               # 현재 시각(=injection 포트가 idle한 시각)
    total_stall = 0.0

    # period, slice 계산 helper
    def period_id(i: int) -> int:
        return i // repeat_period_tiles

    def local_i(i: int) -> int:
        return i % repeat_period_tiles

    def slice_id(i: int) -> int:
        return local_i(i) // activation_reuse_tiles

    def first_tile_of_slice(i: int) -> bool:
        return (local_i(i) % activation_reuse_tiles) == 0

    def first_tile_of_period(i: int) -> bool:
        return (i % repeat_period_tiles) == 0

    # -----------------------------
    # 초기 로딩: tile 0 weight + slice 0 ifmap
    # -----------------------------
    p0 = period_id(0)
    s0 = slice_id(0)

    # weight(0)
    if not loaded_weight[0]:
        _push_pending_front(_Pending("weight", (0, 0), weight_bytes[0]))
    # ifmap(period0, slice0)
    if not loaded_ifmap.get((p0, s0), False):
        _push_pending_back(_Pending("ifmap", (p0, s0), ifmap_bytes_per_slice))

    # 필요한 것이 다 로딩될 때까지 stall
    if pending:
        need_cycles = math.ceil(sum(p.remaining_bytes for p in pending) / bw_bytes_per_cycle) if bw_bytes_per_cycle != float("inf") else 0
        if need_cycles > 0:
            _consume_dram(need_cycles)
            t += need_cycles
            total_stall += need_cycles

    # -----------------------------
    # 타일 실행 루프
    # -----------------------------
    start_times: List[float] = [0.0] * len(tile_lengths)

    for i, k_len in enumerate(tile_lengths):
        k_len = int(k_len)
        pid = period_id(i)
        sid = slice_id(i)

        # period가 바뀌면 (새로운 M-타일 등) slice0 ifmap을 다시 로딩해야 함
        if first_tile_of_period(i) and i != 0:
            # 새 period의 slice0 로딩을 미리 걸어둠 (가능하면 prefetch로 숨겨짐)
            if not loaded_ifmap.get((pid, 0), False) and _pending_find("ifmap", (pid, 0)) is None:
                _push_pending_back(_Pending("ifmap", (pid, 0), ifmap_bytes_per_slice))

        # 현재 타일 weight가 없으면 로딩
        if not loaded_weight[i] and _pending_find("weight", (i, 0)) is None:
            _push_pending_front(_Pending("weight", (i, 0), weight_bytes[i]))

        # 현재 slice ifmap이 없으면 로딩
        if not loaded_ifmap.get((pid, sid), False) and _pending_find("ifmap", (pid, sid)) is None:
            _push_pending_back(_Pending("ifmap", (pid, sid), ifmap_bytes_per_slice))

        # output group 가용 시각까지 기다림 (그 동안 DRAM prefetch 가능)
        g = i % num_groups
        if group_free_time[g] > t:
            wait = group_free_time[g] - t
            _consume_dram(wait)
            t += wait
            total_stall += wait

        # 필요한 데이터가 준비될 때까지 기다림
        # (pending 우선순위: weight -> ifmap)
        while (not loaded_weight[i]) or (not loaded_ifmap.get((pid, sid), False)):
            if not pending:
                # 이론상 발생하면 안 됨(필요한 걸 pending에 올려야 함)
                raise RuntimeError("Required data not loaded but pending queue is empty.")
            # 다음 pending이 완료될 때까지 걸리는 최소 사이클을 계산
            need_bytes = pending[0].remaining_bytes
            need_cycles = math.ceil(need_bytes / bw_bytes_per_cycle) if bw_bytes_per_cycle != float("inf") else 0
            if need_cycles <= 0:
                break
            _consume_dram(need_cycles)
            t += need_cycles
            total_stall += need_cycles

        # 타일 i injection 시작
        start_times[i] = t
        inj_start = t
        inj_end = inj_start + k_len

        # 타일 i 시작 시점에 "다음 타일 weight" prefetch 걸기
        if i + 1 < len(tile_lengths):
            if not loaded_weight[i + 1] and _pending_find("weight", (i + 1, 0)) is None:
                _push_pending_front(_Pending("weight", (i + 1, 0), weight_bytes[i + 1]))

        # 현재 타일이 slice의 첫 타일이면, 다음 slice ifmap을 미리 prefetch
        # (동일 period 내에서만)
        if first_tile_of_slice(i):
            next_sid = sid + 1
            # 다음 slice가 period 안에 존재하는지 체크
            # period 길이 = repeat_period_tiles, local index 범위 내에서 slice 개수 계산
            max_sid_in_period = (repeat_period_tiles - 1) // activation_reuse_tiles
            if next_sid <= max_sid_in_period:
                if not loaded_ifmap.get((pid, next_sid), False) and _pending_find("ifmap", (pid, next_sid)) is None:
                    _push_pending_back(_Pending("ifmap", (pid, next_sid), ifmap_bytes_per_slice))

        # injection 동안 DRAM prefetch 진행
        _consume_dram(k_len)
        t = inj_end  # injection 포트가 다음 타일을 시작할 수 있는 가장 빠른 시각

        # output group 점유 갱신 (OS-SA row 기준)
        T_occ = k_len + 2 * int(S_col) - 1
        group_free_time[g] = max(group_free_time[g], inj_start + T_occ)

    injection_done_time = t
    ofmap_drain_done_time = max([injection_done_time] + group_free_time)

    return {
        "ofmap_drain_done_time": float(ofmap_drain_done_time),
        "injection_done_time": float(injection_done_time),
        "total_stall_cycles": float(total_stall),
        "num_output_groups": float(num_groups),
    }
