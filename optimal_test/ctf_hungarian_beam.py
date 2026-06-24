#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple

import torch

from ctf_optimality_oracle_dp_lb import exact_fixed_row_ctf
from ctf_greedy_row_pairing import CTFConfig, count_nnz, pack_tile

PAD_VALUE = -1


@dataclass
class BeamSettings:
    beam_width: int = 16
    row_option_limit: int = 32
    dense_tiebreak: bool = True
    fallback_to_first_fit: bool = True


@dataclass
class TransitionResult:
    now_after: torch.Tensor
    next_after: torch.Tensor
    moved_nnz: int
    row_order: List[int]
    next_groups: int
    next_cycles: int
    next_physical_slots: int
    used_first_fit_fallback: bool
    runtime_seconds: float
    notes: str = ""


def _row_score(now_packed: torch.Tensor, group_sources: Sequence[Sequence[int]], next_tile: torch.Tensor) -> List[List[int]]:
    rows = int(now_packed.shape[0])
    cols = int(next_tile.shape[1])
    normalized = []
    for srcs in group_sources:
        normalized.append({int(c) for c in srcs if 0 <= int(c) < cols})
    score = [[0 for _ in range(rows)] for _ in range(rows)]
    for i in range(rows):
        holes_by_source = {}
        for g, srcs in enumerate(normalized):
            if int(now_packed[i, g].item()) != PAD_VALUE:
                continue
            for c in srcs:
                holes_by_source[c] = holes_by_source.get(c, 0) + 1
        for j in range(rows):
            s = 0
            for c in range(cols):
                if int(next_tile[j, c].item()) != PAD_VALUE:
                    s += min(1, holes_by_source.get(c, 0))
            score[i][j] = s
    return score


def _hungarian_or_greedy_order(score: List[List[int]]) -> List[int]:
    n = len(score)
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore
        import numpy as np
        cost = np.array([[-score[i][j] for j in range(n)] for i in range(n)])
        row_ind, col_ind = linear_sum_assignment(cost)
        order = [0] * n
        for r, c in zip(row_ind, col_ind):
            order[int(r)] = int(c)
        return order
    except Exception:
        remaining = set(range(n))
        order = []
        for i in range(n):
            j = max(remaining, key=lambda x: (score[i][x], -x))
            remaining.remove(j)
            order.append(j)
        return order


def _pack_score(pk: Any, tile: torch.Tensor, config: CTFConfig) -> Tuple[int, int, int]:
    m = pack_tile(pk, tile, config)
    return m.groups, m.cycles, m.physical_slots


def _first_fit_transition(pk: Any, now_packed: torch.Tensor, group_sources: Sequence[Sequence[int]], next_tile: torch.Tensor, config: CTFConfig) -> Tuple[torch.Tensor, torch.Tensor, int, Tuple[int, int, int]]:
    if hasattr(pk, "cross_tile_fill_all_groups_current"):
        now_after, next_after, moved = pk.cross_tile_fill_all_groups_current(
            now_packed, group_sources, next_tile, pad_value=config.pad_value, reorder_rows=False
        )
    else:
        move = exact_fixed_row_ctf(now_packed, group_sources, next_tile, pad_value=config.pad_value)
        now_after, next_after, moved = move.now_after, move.next_after, move.moved_nnz
    return now_after, next_after, int(moved), _pack_score(pk, next_after, config)


def hungarian_beam_transition(
    pk: Any,
    now_packed: torch.Tensor,
    group_sources: Sequence[Sequence[int]],
    next_tile: torch.Tensor,
    config: CTFConfig,
    *,
    settings: BeamSettings,
) -> TransitionResult:
    start = time.perf_counter()
    before = count_nnz(now_packed, config.pad_value) + count_nnz(next_tile, config.pad_value)

    score = _row_score(now_packed, group_sources, next_tile)
    order = _hungarian_or_greedy_order(score)
    aligned_next = next_tile[order, :]

    # For the selected Hungarian row pairing, use exact maximum matching for the NNZ movement.
    # This is at least as strong as the old beam move selector for this fixed row pairing.
    move = exact_fixed_row_ctf(now_packed, group_sources, aligned_next, pad_value=config.pad_value)
    next_score = _pack_score(pk, move.next_after, config)
    used_fallback = False
    notes = "Hungarian row pairing + exact fixed-row max matching"

    if settings.fallback_to_first_fit:
        ff_now, ff_next, ff_moved, ff_score = _first_fit_transition(pk, now_packed, group_sources, next_tile, config)
        candidate_key = (next_score[0], next_score[1], next_score[2], -int(move.moved_nnz))
        fallback_key = (ff_score[0], ff_score[1], ff_score[2], -int(ff_moved))
        if fallback_key < candidate_key:
            used_fallback = True
            move_now, move_next, moved = ff_now, ff_next, ff_moved
            next_score = ff_score
            order = list(range(int(next_tile.shape[0])))
            notes = "first-fit fallback selected"
        else:
            move_now, move_next, moved = move.now_after, move.next_after, int(move.moved_nnz)
    else:
        move_now, move_next, moved = move.now_after, move.next_after, int(move.moved_nnz)

    after = count_nnz(move_now, config.pad_value) + count_nnz(move_next, config.pad_value)
    if before != after:
        raise AssertionError(f"lossless check failed: before={before} after={after}")

    return TransitionResult(
        now_after=move_now,
        next_after=move_next,
        moved_nnz=int(moved),
        row_order=[int(x) for x in order],
        next_groups=int(next_score[0]),
        next_cycles=int(next_score[1]),
        next_physical_slots=int(next_score[2]),
        used_first_fit_fallback=used_fallback,
        runtime_seconds=time.perf_counter() - start,
        notes=notes,
    )
