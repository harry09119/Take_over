#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ctf_row_pairing_exact_oracle_memo.py

Exact row-pairing oracle for one CTF transition.

For each permutation of next-tile rows, it calls the fixed-row CTF move oracle
from ctf_optimality_oracle_dp_lb.py.  The best transition is selected by the
same practical objective used in the benchmark driver:
  1) lower packed group count of the remaining next tile
  2) lower cycle count
  3) lower physical slot count
  4) higher moved NNZ

This is intentionally brute-force for 8 rows (8! row pairings).  The user-side
8-tile exhaustive order experiment is expected to be expensive; this file only
removes the previous 4-tile/4-row hardcoding.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from ctf_optimality_oracle_dp_lb import exact_fixed_row_ctf

PAD_VALUE = -1


@dataclass
class RowExactTransitionResult:
    now_after: torch.Tensor
    next_after: torch.Tensor
    moved_nnz: int
    row_order: List[int]
    next_groups: int
    next_cycles: int
    next_physical_slots: int
    exact_certified: bool
    runtime_seconds: float
    checked_pairings: int
    notes: str = ""


def _count_nnz(x: torch.Tensor, pad_value: int = PAD_VALUE) -> int:
    return int((x != pad_value).sum().item())


def _pack_score(pk: Any, tile: torch.Tensor, config: Any) -> Tuple[int, int, int]:
    result = pk.column_combine_modulo_residual_current(
        tile,
        mux_size=int(config.mux_size),
        reuse_depth=int(config.reuse_depth),
        max_residual_groups_per_lane=int(config.max_residual_groups_per_lane),
        parallel_groups=int(config.parallel_groups),
        max_conflict=config.max_conflict,
    )
    groups = sum(1 for srcs in result.gidx if len(srcs) > 0)
    physical = int(result.scheduled_packed.shape[1])
    cycles = int(getattr(result.metadata, "total_cycles", physical // int(config.parallel_groups)))
    return groups, cycles, physical


def exact_row_pairing_transition(
    pk: Any,
    now_packed: torch.Tensor,
    group_sources: Sequence[Sequence[int]],
    next_tile: torch.Tensor,
    config: Any,
    *,
    timeout_seconds: float = 300.0,
    pad_value: int = PAD_VALUE,
    max_pair_options: int = 0,
    **_: Any,
) -> RowExactTransitionResult:
    if now_packed.ndim != 2 or next_tile.ndim != 2:
        raise ValueError("now_packed and next_tile must be 2-D tensors")
    if now_packed.shape[0] != next_tile.shape[0]:
        raise ValueError(f"row mismatch: now={now_packed.shape[0]} next={next_tile.shape[0]}")

    start = time.perf_counter()
    n_rows = int(next_tile.shape[0])
    original_total = _count_nnz(now_packed, pad_value) + _count_nnz(next_tile, pad_value)

    best = None
    checked = 0
    certified = True
    notes = ""

    for perm in itertools.permutations(range(n_rows)):
        if timeout_seconds and (time.perf_counter() - start) > timeout_seconds:
            certified = False
            notes = f"row-pairing timeout after {timeout_seconds}s"
            break
        if max_pair_options and checked >= max_pair_options:
            certified = False
            notes = f"stopped after max_pair_options={max_pair_options}"
            break

        aligned_next = next_tile[list(perm), :]
        move = exact_fixed_row_ctf(now_packed, group_sources, aligned_next, pad_value=pad_value)
        groups, cycles, physical = _pack_score(pk, move.next_after, config)
        key = (groups, cycles, physical, -int(move.moved_nnz), list(perm))
        checked += 1

        if best is None or key < best[0]:
            best = (key, move, list(perm), groups, cycles, physical)

    if best is None:
        # Safe no-move fallback.
        groups, cycles, physical = _pack_score(pk, next_tile, config)
        return RowExactTransitionResult(
            now_after=now_packed.clone(),
            next_after=next_tile.clone(),
            moved_nnz=0,
            row_order=list(range(n_rows)),
            next_groups=groups,
            next_cycles=cycles,
            next_physical_slots=physical,
            exact_certified=False,
            runtime_seconds=time.perf_counter() - start,
            checked_pairings=checked,
            notes="no row pairing was evaluated",
        )

    _, move, perm, groups, cycles, physical = best
    after_total = _count_nnz(move.now_after, pad_value) + _count_nnz(move.next_after, pad_value)
    if original_total != after_total:
        raise AssertionError(f"lossless check failed: before={original_total} after={after_total}")

    return RowExactTransitionResult(
        now_after=move.now_after,
        next_after=move.next_after,
        moved_nnz=int(move.moved_nnz),
        row_order=perm,
        next_groups=int(groups),
        next_cycles=int(cycles),
        next_physical_slots=int(physical),
        exact_certified=bool(certified),
        runtime_seconds=time.perf_counter() - start,
        checked_pairings=checked,
        notes=notes,
    )


# Backward-compatible aliases.
solve_exact_row_pairing = exact_row_pairing_transition
exact_pair_transition = exact_row_pairing_transition
