#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ctf_optimality_oracle_dp_lb.py

Small exact CTF move oracle for one already-packed current tile and one row-aligned
next tile.

The problem solved here is:
  given now_packed[row, group], group_sources[group], and next_tile[row, col],
  move as many next-tile nonzeros as possible into holes of now_packed without
  introducing any new MUX source.

For a fixed row alignment, this is a bipartite maximum matching problem:
  left  = holes (row, packed_group)
  right = next nonzeros (row, source_col)
  edge  = source_col in group_sources[packed_group]

This file intentionally has no dependency on scipy.  It is used by the row-pair
exact oracle and can also be imported directly for sanity checks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch

PAD_VALUE = -1
Hole = Tuple[int, int]      # (row, packed_group)
Source = Tuple[int, int]    # (row, source_col)


@dataclass
class MoveOracleResult:
    now_after: torch.Tensor
    next_after: torch.Tensor
    moved_nnz: int
    matching: List[Tuple[Hole, Source]]
    edge_count: int


def _normalize_sources(group_sources: Sequence[Sequence[int]], n_cols: int) -> List[List[int]]:
    out: List[List[int]] = []
    for srcs in group_sources:
        seen = []
        for x in srcs:
            ci = int(x)
            if 0 <= ci < n_cols and ci not in seen:
                seen.append(ci)
        out.append(seen)
    return out


def enumerate_move_edges(
    now_packed: torch.Tensor,
    group_sources: Sequence[Sequence[int]],
    next_tile: torch.Tensor,
    *,
    pad_value: int = PAD_VALUE,
) -> Tuple[List[Hole], List[Source], Dict[int, List[int]]]:
    """Return bipartite graph edges for exact fixed-row CTF movement."""
    if now_packed.ndim != 2 or next_tile.ndim != 2:
        raise ValueError("now_packed and next_tile must be 2-D tensors")
    if now_packed.shape[0] != next_tile.shape[0]:
        raise ValueError(f"row mismatch: now={now_packed.shape[0]} next={next_tile.shape[0]}")
    if now_packed.shape[1] != len(group_sources):
        raise ValueError("len(group_sources) must equal now_packed width")

    n_rows = int(now_packed.shape[0])
    n_groups = int(now_packed.shape[1])
    n_cols = int(next_tile.shape[1])
    sources = _normalize_sources(group_sources, n_cols)

    source_to_right: Dict[Source, int] = {}
    right_nodes: List[Source] = []
    for r in range(n_rows):
        for c in range(n_cols):
            if int(next_tile[r, c].item()) != pad_value:
                source_to_right[(r, c)] = len(right_nodes)
                right_nodes.append((r, c))

    left_nodes: List[Hole] = []
    edges: Dict[int, List[int]] = {}
    for r in range(n_rows):
        for g in range(n_groups):
            if int(now_packed[r, g].item()) != pad_value:
                continue
            adj: List[int] = []
            for c in sources[g]:
                rid = source_to_right.get((r, c))
                if rid is not None:
                    adj.append(rid)
            if adj:
                lid = len(left_nodes)
                left_nodes.append((r, g))
                # Deterministic order: dense columns are handled upstream; here source index tie-break.
                edges[lid] = sorted(adj, key=lambda rid: right_nodes[rid])
    return left_nodes, right_nodes, edges


def maximum_bipartite_matching(edges: Dict[int, List[int]], n_right: int) -> Dict[int, int]:
    """Kuhn DFS maximum cardinality matching. Returns left_id -> right_id."""
    match_r: List[int] = [-1] * int(n_right)

    # Visit high-degree left nodes first; this is deterministic and often faster.
    left_order = sorted(edges.keys(), key=lambda u: (-len(edges[u]), u))

    def dfs(u: int, seen: List[bool]) -> bool:
        for v in edges.get(u, []):
            if seen[v]:
                continue
            seen[v] = True
            if match_r[v] < 0 or dfs(match_r[v], seen):
                match_r[v] = u
                return True
        return False

    for u in left_order:
        seen = [False] * n_right
        dfs(u, seen)

    return {u: v for v, u in enumerate(match_r) if u >= 0}


def apply_matching(
    now_packed: torch.Tensor,
    next_tile: torch.Tensor,
    left_nodes: Sequence[Hole],
    right_nodes: Sequence[Source],
    match_l: Dict[int, int],
    *,
    pad_value: int = PAD_VALUE,
) -> MoveOracleResult:
    now = now_packed.clone()
    nxt = next_tile.clone()
    matching: List[Tuple[Hole, Source]] = []

    for lid, rid in sorted(match_l.items(), key=lambda kv: (left_nodes[kv[0]], right_nodes[kv[1]])):
        r, g = left_nodes[lid]
        rr, c = right_nodes[rid]
        if r != rr:
            raise AssertionError("CTF matching crossed rows, which is illegal after row alignment")
        if int(now[r, g].item()) != pad_value:
            raise AssertionError("CTF destination is not a hole")
        value = int(nxt[rr, c].item())
        if value == pad_value:
            raise AssertionError("CTF source was already consumed")
        now[r, g] = value
        nxt[rr, c] = pad_value
        matching.append(((r, g), (rr, c)))

    return MoveOracleResult(
        now_after=now,
        next_after=nxt,
        moved_nnz=len(matching),
        matching=matching,
        edge_count=sum(len(v) for v in enumerate_move_edges(now_packed, [], next_tile)[2].values()) if False else 0,
    )


def exact_fixed_row_ctf(
    now_packed: torch.Tensor,
    group_sources: Sequence[Sequence[int]],
    next_tile: torch.Tensor,
    *,
    pad_value: int = PAD_VALUE,
) -> MoveOracleResult:
    """Exact max-move CTF for a fixed row alignment."""
    before = int((now_packed != pad_value).sum().item()) + int((next_tile != pad_value).sum().item())
    left_nodes, right_nodes, edges = enumerate_move_edges(now_packed, group_sources, next_tile, pad_value=pad_value)
    match_l = maximum_bipartite_matching(edges, len(right_nodes))
    result = apply_matching(now_packed, next_tile, left_nodes, right_nodes, match_l, pad_value=pad_value)
    result.edge_count = sum(len(v) for v in edges.values())
    after = int((result.now_after != pad_value).sum().item()) + int((result.next_after != pad_value).sum().item())
    if before != after:
        raise AssertionError(f"lossless check failed: before={before} after={after}")
    return result


# Backward-compatible aliases often used by older drivers.
solve_fixed_row_ctf = exact_fixed_row_ctf
best_move_for_aligned_rows = exact_fixed_row_ctf
