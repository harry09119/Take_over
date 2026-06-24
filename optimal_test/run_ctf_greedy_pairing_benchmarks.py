#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib.util
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from ctf_greedy_row_pairing import CTFConfig, count_nnz, pack_tile

PAD_VALUE = -1


@dataclass
class SequenceResult:
    method: str
    order: List[int]
    total_groups: int
    total_regular_groups: int
    total_residual_groups: int
    total_physical_slots: int
    total_cycles: int
    total_moved_nnz: int
    original_nnz: int
    processed_nnz: int
    lossless_verified: bool
    exact_certified: bool
    runtime_seconds: float
    transition_runtimes: List[float] = field(default_factory=list)
    transition_groups: List[int] = field(default_factory=list)
    transition_moved: List[int] = field(default_factory=list)
    notes: str = ""


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, os.path.abspath(path))
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def generate_tiles(tile_rows: int, tile_cols: int, tile_count: int, density: float, seed: int, *, pad_value: int = PAD_VALUE) -> List[torch.Tensor]:
    g = torch.Generator().manual_seed(int(seed))
    tiles: List[torch.Tensor] = []
    for _ in range(int(tile_count)):
        mask = torch.rand((int(tile_rows), int(tile_cols)), generator=g) < float(density)
        values = torch.arange(int(tile_cols), dtype=torch.int64).view(1, -1).expand(int(tile_rows), -1)
        tile = torch.where(mask, values, torch.full_like(values, int(pad_value)))
        tiles.append(tile.to(torch.int64))
    return tiles


def _accumulate_metrics(metrics: Any, tile_after: torch.Tensor, totals: Dict[str, int], config: CTFConfig) -> None:
    totals["groups"] += int(metrics.groups)
    totals["regular"] += int(metrics.regular_groups)
    totals["residual"] += int(metrics.residual_groups)
    totals["physical"] += int(metrics.physical_slots)
    totals["cycles"] += int(metrics.cycles)
    totals["processed"] += count_nnz(tile_after, config.pad_value)


def evaluate_first_fit_sequence(pk: Any, tiles: Sequence[torch.Tensor], order: Sequence[int], config: CTFConfig) -> SequenceResult:
    start = time.perf_counter()
    current = tiles[int(order[0])].clone()
    totals = dict(groups=0, regular=0, residual=0, physical=0, cycles=0, processed=0)
    moved_total = 0
    transition_groups: List[int] = []
    transition_moved: List[int] = []

    for next_id in order[1:]:
        metrics = pack_tile(pk, current, config)
        if hasattr(pk, "cross_tile_fill_all_groups_current"):
            now_after, next_after, moved = pk.cross_tile_fill_all_groups_current(
                metrics.result.scheduled_packed,
                metrics.result.gidx,
                tiles[int(next_id)],
                pad_value=config.pad_value,
                reorder_rows=False,
            )
        else:
            from ctf_optimality_oracle_dp_lb import exact_fixed_row_ctf
            move = exact_fixed_row_ctf(metrics.result.scheduled_packed, metrics.result.gidx, tiles[int(next_id)], pad_value=config.pad_value)
            now_after, next_after, moved = move.now_after, move.next_after, move.moved_nnz
        _accumulate_metrics(metrics, now_after, totals, config)
        moved_total += int(moved)
        transition_groups.append(metrics.groups)
        transition_moved.append(int(moved))
        current = next_after

    final = pack_tile(pk, current, config)
    _accumulate_metrics(final, current, totals, config)
    original_nnz = sum(count_nnz(t, config.pad_value) for t in tiles)
    lossless = totals["processed"] == original_nnz
    return SequenceResult(
        method="natural_order_first_fit_ctf",
        order=[int(x) for x in order],
        total_groups=totals["groups"],
        total_regular_groups=totals["regular"],
        total_residual_groups=totals["residual"],
        total_physical_slots=totals["physical"],
        total_cycles=totals["cycles"],
        total_moved_nnz=moved_total,
        original_nnz=original_nnz,
        processed_nnz=totals["processed"],
        lossless_verified=lossless,
        exact_certified=False,
        runtime_seconds=time.perf_counter() - start,
        transition_groups=transition_groups,
        transition_moved=transition_moved,
    )


def choose_order_first_fit_greedy(pk: Any, tiles: Sequence[torch.Tensor], config: CTFConfig) -> Tuple[List[int], float]:
    start = time.perf_counter()
    best = None
    for first in range(len(tiles)):
        current = tiles[first].clone()
        order = [first]
        remaining = set(range(len(tiles))) - {first}
        groups = cycles = physical = moved_total = 0
        while remaining:
            metrics = pack_tile(pk, current, config)
            candidates = []
            for nxt_id in sorted(remaining):
                if hasattr(pk, "cross_tile_fill_all_groups_current"):
                    _, next_after, moved = pk.cross_tile_fill_all_groups_current(
                        metrics.result.scheduled_packed, metrics.result.gidx, tiles[nxt_id], pad_value=config.pad_value, reorder_rows=False
                    )
                else:
                    from ctf_optimality_oracle_dp_lb import exact_fixed_row_ctf
                    move = exact_fixed_row_ctf(metrics.result.scheduled_packed, metrics.result.gidx, tiles[nxt_id], pad_value=config.pad_value)
                    next_after, moved = move.next_after, move.moved_nnz
                nm = pack_tile(pk, next_after, config)
                score = (groups + metrics.groups + nm.groups, cycles + metrics.cycles + nm.cycles, physical + metrics.physical_slots + nm.physical_slots, -(moved_total + int(moved)), nxt_id)
                candidates.append((score, nxt_id, next_after, int(moved)))
            score, selected, next_after, moved = min(candidates, key=lambda x: x[0])
            groups += metrics.groups; cycles += metrics.cycles; physical += metrics.physical_slots; moved_total += moved
            order.append(selected); remaining.remove(selected); current = next_after
        final = pack_tile(pk, current, config)
        key = (groups + final.groups, cycles + final.cycles, physical + final.physical_slots, -moved_total, tuple(order))
        if best is None or key < best[0]:
            best = (key, list(order))
    if best is None:
        raise RuntimeError("empty tile set")
    return best[1], time.perf_counter() - start


def evaluate_greedy_pairing_sequence(pk: Any, tiles: Sequence[torch.Tensor], order: Sequence[int], config: CTFConfig, *, swap_candidates: int = 64, swap_passes: int = 1) -> SequenceResult:
    # Conservative compatibility wrapper: use first-fit sequence.  The New CTP path is implemented
    # in compare_existing_exact_hungarian_beam_8tiles.py.
    r = evaluate_first_fit_sequence(pk, tiles, order, config)
    r.method = "greedy_pairing_compat_first_fit"
    return r


def safe_sequence_result(candidate: SequenceResult, fallback: SequenceResult) -> SequenceResult:
    ckey = (candidate.total_groups, candidate.total_cycles, candidate.total_physical_slots, -candidate.total_moved_nnz)
    fkey = (fallback.total_groups, fallback.total_cycles, fallback.total_physical_slots, -fallback.total_moved_nnz)
    return candidate if ckey <= fkey else fallback


def evaluate_exact_pairing_sequence(
    pk: Any,
    base_oracle: Any,
    row_exact: Any,
    tiles: Sequence[torch.Tensor],
    order: Sequence[int],
    config: CTFConfig,
    *,
    timeout_seconds: float = 300.0,
    max_leaf_states: int = 0,
    max_pair_options: int = 0,
) -> SequenceResult:
    start = time.perf_counter()
    current = tiles[int(order[0])].clone()
    totals = dict(groups=0, regular=0, residual=0, physical=0, cycles=0, processed=0)
    moved_total = 0
    exact_all = True
    transition_runtimes: List[float] = []
    transition_groups: List[int] = []
    transition_moved: List[int] = []
    notes: List[str] = []

    exact_func = getattr(row_exact, "exact_pair_transition", None) or getattr(row_exact, "exact_row_pairing_transition", None)
    if exact_func is None:
        raise AttributeError("row_exact module must expose exact_pair_transition or exact_row_pairing_transition")

    for next_id in order[1:]:
        metrics = pack_tile(pk, current, config)
        tr = exact_func(
            pk,
            metrics.result.scheduled_packed,
            metrics.result.gidx,
            tiles[int(next_id)],
            config,
            timeout_seconds=timeout_seconds,
            max_pair_options=max_pair_options,
        )
        _accumulate_metrics(metrics, tr.now_after, totals, config)
        moved_total += int(tr.moved_nnz)
        exact_all = exact_all and bool(tr.exact_certified)
        transition_runtimes.append(float(tr.runtime_seconds))
        transition_groups.append(int(tr.next_groups))
        transition_moved.append(int(tr.moved_nnz))
        if getattr(tr, "notes", ""):
            notes.append(str(tr.notes))
        current = tr.next_after

    final = pack_tile(pk, current, config)
    _accumulate_metrics(final, current, totals, config)
    original_nnz = sum(count_nnz(t, config.pad_value) for t in tiles)
    lossless = totals["processed"] == original_nnz
    return SequenceResult(
        method="exhaustive_order_local_exact_row_pairing",
        order=[int(x) for x in order],
        total_groups=totals["groups"],
        total_regular_groups=totals["regular"],
        total_residual_groups=totals["residual"],
        total_physical_slots=totals["physical"],
        total_cycles=totals["cycles"],
        total_moved_nnz=moved_total,
        original_nnz=original_nnz,
        processed_nnz=totals["processed"],
        lossless_verified=lossless,
        exact_certified=bool(exact_all and lossless),
        runtime_seconds=time.perf_counter() - start,
        transition_runtimes=transition_runtimes,
        transition_groups=transition_groups,
        transition_moved=transition_moved,
        notes="; ".join(notes),
    )
