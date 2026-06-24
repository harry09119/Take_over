#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

PAD_VALUE = -1


@dataclass
class CTFConfig:
    mux_size: int = 4
    reuse_depth: int = 2
    max_residual_groups_per_lane: int = 1
    parallel_groups: int = 4
    max_conflict: Optional[int] = 2
    pad_value: int = PAD_VALUE


@dataclass
class PackedTileMetrics:
    result: Any
    groups: int
    regular_groups: int
    residual_groups: int
    physical_slots: int
    cycles: int
    nnz: int


def count_nnz(x: torch.Tensor, pad_value: int = PAD_VALUE) -> int:
    return int((x != pad_value).sum().item())


def pack_tile(pk: Any, tile: torch.Tensor, config: CTFConfig) -> PackedTileMetrics:
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
    metadata = result.metadata
    cycles = int(getattr(metadata, "total_cycles", physical // int(config.parallel_groups)))

    # These metadata counters are cycles, not group counts.  Convert to physical slots and then
    # leave total logical groups above as the primary objective.
    regular_physical = int(getattr(metadata, "regular_cycles", 0)) * int(config.parallel_groups)
    residual_physical = int(getattr(metadata, "residual_cycles", 0)) * int(config.parallel_groups)
    regular_nonempty = 0
    residual_nonempty = 0
    for idx, srcs in enumerate(result.gidx):
        if not srcs:
            continue
        if idx < regular_physical:
            regular_nonempty += 1
        else:
            residual_nonempty += 1
    # If block interleaving makes the simple split imperfect, keep a sane total.
    if regular_nonempty + residual_nonempty != groups:
        residual_nonempty = max(0, groups - regular_nonempty)

    return PackedTileMetrics(
        result=result,
        groups=int(groups),
        regular_groups=int(regular_nonempty),
        residual_groups=int(residual_nonempty),
        physical_slots=physical,
        cycles=cycles,
        nnz=count_nnz(tile, config.pad_value),
    )
