from __future__ import annotations

import argparse
import csv
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from tqdm import tqdm


# =============================================================================
# Utility
# =============================================================================


def print_mat(matrix: torch.Tensor) -> None:
    n_rows, n_cols = matrix.shape
    header = " ".join(f"{value:3d}" for value in range(n_cols))
    print("\n", header)
    print("=" * len(header))

    for row_index, row in enumerate(matrix.tolist()):
        row_text = " ".join(
            f"{'x':>3}" if value == -1 else f"{value:3d}"
            for value in row
        )
        print(f"{row_text} ||{row_index}")


def generate_sparse_matrix(
    n_rows: int,
    n_cols: int,
    density: float,
    generator: torch.Generator,
) -> torch.Tensor:
    if n_rows <= 0 or n_cols <= 0:
        raise ValueError("n_rows and n_cols must be greater than 0")
    if not 0.0 <= density <= 1.0:
        raise ValueError("density must be between 0 and 1")

    mask = torch.rand((n_rows, n_cols), generator=generator) < density
    column_ids = torch.arange(n_cols, dtype=torch.int32).expand(n_rows, n_cols)
    return torch.where(mask, column_ids, torch.full_like(column_ids, -1))


def remove_empty_columns(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    column_mask = (matrix >= 0).any(dim=0)
    return matrix[:, column_mask]


def iter_mask_rows(mask: int) -> Iterable[int]:
    while mask:
        lowest_bit = mask & -mask
        yield lowest_bit.bit_length() - 1
        mask ^= lowest_bit


# =============================================================================
# Prepared sparse-column representation
# =============================================================================


@dataclass(frozen=True)
class PreparedColumns:
    compact: torch.Tensor
    column_masks: List[int]
    column_nnz: List[int]
    ranked_columns: List[int]
    remaining: Set[int]


def prepare_columns(matrix: torch.Tensor) -> PreparedColumns:
    """Precompute bit masks so both methods see exactly the same tile."""
    compact = remove_empty_columns(matrix)
    _, n_cols = compact.shape

    if n_cols == 0:
        return PreparedColumns(compact, [], [], [], set())

    patterns = (compact >= 0).transpose(0, 1).contiguous().cpu().numpy()
    packed_bits = np.packbits(patterns, axis=1, bitorder="little")
    nnz_array = patterns.sum(axis=1, dtype=np.int64)

    column_masks = [
        int.from_bytes(row.tobytes(), byteorder="little", signed=False)
        for row in packed_bits
    ]
    column_nnz = [int(value) for value in nnz_array.tolist()]
    ranked_columns = sorted(
        range(n_cols),
        key=lambda ci: (-column_nnz[ci], ci),
    )
    remaining = {ci for ci, count in enumerate(column_nnz) if count > 0}

    return PreparedColumns(
        compact=compact,
        column_masks=column_masks,
        column_nnz=column_nnz,
        ranked_columns=ranked_columns,
        remaining=remaining,
    )


def select_seed(remaining: Set[int], ranked_columns: Sequence[int]) -> int:
    for column_index in ranked_columns:
        if column_index in remaining:
            return column_index
    raise RuntimeError("No seed exists although remaining is nonempty")


def candidate_columns(
    remaining: Set[int],
    ranked_columns: Sequence[int],
    search_limit: Optional[int],
) -> Iterable[int]:
    yielded = 0
    for column_index in ranked_columns:
        if column_index not in remaining:
            continue
        yield column_index
        yielded += 1
        if search_limit is not None and yielded >= search_limit:
            break


# =============================================================================
# Result structures
# =============================================================================


@dataclass
class ResidualGroupState:
    mask: int = 0
    sources: Set[int] = field(default_factory=set)
    placement: Dict[int, int] = field(default_factory=dict)


@dataclass
class PackingMetadata:
    group_types: List[str] = field(default_factory=list)
    group_blocks: List[int] = field(default_factory=list)
    group_lanes: List[int] = field(default_factory=list)
    group_conflicts: List[int] = field(default_factory=list)

    regular_groups: int = 0
    residual_groups: int = 0
    padding_groups: int = 0

    regular_cycles: int = 0
    residual_cycles: int = 0
    total_cycles: int = 0
    blocks: int = 0

    residual_nnz: int = 0
    residual_source_references: int = 0


@dataclass
class PackingResult:
    # scheduled_packed includes padding columns so physical Cg % P is preserved.
    scheduled_packed: torch.Tensor
    group_len: List[int]
    gidx: List[List[int]]
    metadata: PackingMetadata

    @property
    def packed(self) -> torch.Tensor:
        """Compatibility alias: returns the scheduled tensor with padding."""
        return self.scheduled_packed


# =============================================================================
# Validation/build helpers
# =============================================================================


def build_packed_tensor(
    n_rows: int,
    placements: Sequence[Dict[int, int]],
    device: torch.device,
) -> torch.Tensor:
    packed = torch.full(
        (n_rows, len(placements)),
        -1,
        dtype=torch.int32,
        device=device,
    )

    for group_index, placement in enumerate(placements):
        for row_index, source_column in placement.items():
            packed[row_index, group_index] = source_column

    return packed


def validate_lossless(
    column_masks: Sequence[int],
    placements: Sequence[Dict[int, int]],
    mux_size: int,
    gidx: Sequence[Sequence[int]],
) -> None:
    expected = {
        (row_index, column_index)
        for column_index, column_mask in enumerate(column_masks)
        for row_index in iter_mask_rows(column_mask)
    }

    actual_list = [
        (row_index, source_column)
        for placement in placements
        for row_index, source_column in placement.items()
    ]
    actual = set(actual_list)

    if expected != actual:
        missing = sorted(expected - actual)[:20]
        invalid = sorted(actual - expected)[:20]
        raise AssertionError(
            "Packing is not lossless.\n"
            f"Missing nonzeros: {missing}\n"
            f"Invalid nonzeros: {invalid}"
        )

    if len(actual_list) != len(actual):
        raise AssertionError("A nonzero was duplicated across packed groups.")

    for group_index, source_columns in enumerate(gidx):
        if len(set(source_columns)) > mux_size:
            raise AssertionError(
                f"Group {group_index} exceeds mux_size: "
                f"{len(set(source_columns))} > {mux_size}"
            )


def append_scheduled_group(
    placements: List[Dict[int, int]],
    gidx: List[List[int]],
    metadata: PackingMetadata,
    placement: Dict[int, int],
    sources: Sequence[int],
    group_type: str,
    block_id: int,
    lane: int,
    conflicts: int,
    parallel_groups: int,
) -> None:
    physical_cg = len(placements)
    if physical_cg % parallel_groups != lane:
        raise AssertionError(
            f"Lane alignment error: Cg={physical_cg}, "
            f"Cg%P={physical_cg % parallel_groups}, expected lane={lane}"
        )

    placements.append(placement)
    gidx.append(list(sources))
    metadata.group_types.append(group_type)
    metadata.group_blocks.append(block_id)
    metadata.group_lanes.append(lane)
    metadata.group_conflicts.append(conflicts)

    if group_type.startswith("regular") and "padding" not in group_type:
        metadata.regular_groups += 1
    elif group_type.startswith("residual") and "padding" not in group_type:
        metadata.residual_groups += 1
    elif "padding" in group_type:
        metadata.padding_groups += 1


# =============================================================================
# Baseline: strict tile-wise lossless Column Combining
# =============================================================================


def column_combine_lossless_groupwise(
    matrix: Optional[torch.Tensor],
    mux_size: int,
    print_result: bool = False,
    candidate_search_limit: Optional[int] = None,
    prepared: Optional[PreparedColumns] = None,
) -> Tuple[torch.Tensor, List[List[int]]]:
    if mux_size <= 0:
        raise ValueError("mux_size must be greater than 0")

    if prepared is None:
        if matrix is None:
            raise ValueError("matrix is required when prepared is not provided")
        prepared = prepare_columns(matrix)

    compact = prepared.compact
    column_masks = prepared.column_masks
    column_nnz = prepared.column_nnz
    ranked_columns = prepared.ranked_columns
    remaining = set(prepared.remaining)
    n_rows, _ = compact.shape

    placements: List[Dict[int, int]] = []
    gidx: List[List[int]] = []

    while remaining:
        seed = select_seed(remaining, ranked_columns)
        remaining.remove(seed)

        group_mask = column_masks[seed]
        group_sources = [seed]
        group_placement = {
            row_index: seed for row_index in iter_mask_rows(group_mask)
        }

        while remaining and len(group_sources) < mux_size:
            best_column: Optional[int] = None
            best_score: Optional[Tuple[int, int]] = None

            for column_index in candidate_columns(
                remaining, ranked_columns, candidate_search_limit
            ):
                column_mask = column_masks[column_index]
                if group_mask & column_mask:
                    continue

                score = (column_nnz[column_index], -column_index)
                if best_score is None or score > best_score:
                    best_score = score
                    best_column = column_index

            if best_column is None:
                break

            selected_mask = column_masks[best_column]
            for row_index in iter_mask_rows(selected_mask):
                group_placement[row_index] = best_column

            group_mask |= selected_mask
            group_sources.append(best_column)
            remaining.remove(best_column)

        placements.append(group_placement)
        gidx.append(group_sources)

    validate_lossless(column_masks, placements, mux_size, gidx)
    packed = build_packed_tensor(n_rows, placements, compact.device)

    if print_result:
        print("\n<<Strict Lossless Column Combining>>")
        print_mat(packed)

    return packed, gidx


# =============================================================================
# Proposed: modulo-lane residual sharing
# =============================================================================


def residual_combine(
    matrix: Optional[torch.Tensor],
    mux_size: int = 8,
    reuse_depth: int = 2,
    max_residual_groups_per_lane: int = 2,
    parallel_groups: int = 4,
    max_conflict: Optional[int] = None,
    new_residual_group_penalty: int = 1,
    print_result: bool = False,
    candidate_search_limit: Optional[int] = None,
    prepared: Optional[PreparedColumns] = None,
) -> PackingResult:
    """
    Lossless Column Combining where groups with equal Cg % parallel_groups
    share lane-local residual groups.

    For parallel_groups=4 and reuse_depth=4, one regular block is:

        cycle 0: Cg  0,  1,  2,  3
        cycle 1: Cg  4,  5,  6,  7
        cycle 2: Cg  8,  9, 10, 11
        cycle 3: Cg 12, 13, 14, 15

    Lane-local sharing sets are therefore:

        lane 0: Cg 0, 4, 8, 12
        lane 1: Cg 1, 5, 9, 13
        lane 2: Cg 2, 6, 10, 14
        lane 3: Cg 3, 7, 11, 15

    After the regular cycles, residual round r executes four lane-local
    residual groups in parallel. Missing groups are emitted as padding columns
    so the next block again starts at Cg % parallel_groups == 0.

    ``max_residual_groups_per_lane`` is the number of residual rounds that one
    lane may use. Thus P=4 and R=3 allow at most 12 nonempty residual groups in
    a block, but at most three residual cycles.
    """
    if mux_size <= 0:
        raise ValueError("mux_size must be greater than 0")
    if reuse_depth <= 0:
        raise ValueError("reuse_depth must be greater than 0")
    if max_residual_groups_per_lane <= 0:
        raise ValueError("max_residual_groups_per_lane must be greater than 0")
    if parallel_groups <= 0:
        raise ValueError("parallel_groups must be greater than 0")
    if max_conflict is not None and max_conflict < 0:
        raise ValueError("max_conflict must be nonnegative or None")
    if new_residual_group_penalty < 0:
        raise ValueError("new_residual_group_penalty must be nonnegative")

    if prepared is None:
        if matrix is None:
            raise ValueError("matrix is required when prepared is not provided")
        prepared = prepare_columns(matrix)

    compact = prepared.compact
    column_masks = prepared.column_masks
    column_nnz = prepared.column_nnz
    ranked_columns = prepared.ranked_columns
    remaining = set(prepared.remaining)
    n_rows, _ = compact.shape

    placements: List[Dict[int, int]] = []
    gidx: List[List[int]] = []
    metadata = PackingMetadata()

    block_capacity = parallel_groups * reuse_depth
    block_id = 0

    while remaining:
        # Each physical lane owns an independent residual-group pool.
        lane_residual_groups: List[List[ResidualGroupState]] = [
            [] for _ in range(parallel_groups)
        ]

        regular_placements: List[Dict[int, int]] = []
        regular_sources_list: List[List[int]] = []
        regular_conflicts: List[int] = []

        regular_slots_used = 0

        # ------------------------------------------------------------------
        # Build up to reuse_depth regular cycles (P groups per cycle).
        # ------------------------------------------------------------------
        while remaining and regular_slots_used < block_capacity:
            lane = regular_slots_used % parallel_groups
            residual_pool = lane_residual_groups[lane]

            seed = select_seed(remaining, ranked_columns)
            remaining.remove(seed)

            regular_mask = column_masks[seed]
            regular_sources = [seed]
            regular_placement = {
                row_index: seed for row_index in iter_mask_rows(regular_mask)
            }
            regular_conflict_count = 0

            while remaining and len(regular_sources) < mux_size:
                best_column: Optional[int] = None
                best_accepted_mask = 0
                best_conflict_mask = 0
                best_conflict_count = 0
                best_residual_index: Optional[int] = None
                best_opens_residual = False
                best_score: Optional[
                    Tuple[int, int, int, int, int, int, int, int]
                ] = None

                for column_index in candidate_columns(
                    remaining, ranked_columns, candidate_search_limit
                ):
                    column_mask = column_masks[column_index]
                    conflict_mask = regular_mask & column_mask
                    accepted_mask = column_mask & ~regular_mask

                    conflict_count = conflict_mask.bit_count()
                    accepted_count = accepted_mask.bit_count()

                    # The source activation must first be consumed by the
                    # regular group before it can be retained for residual use.
                    if accepted_count == 0:
                        continue

                    if (
                        max_conflict is not None
                        and regular_conflict_count + conflict_count > max_conflict
                    ):
                        continue

                    residual_index: Optional[int] = None
                    opens_residual = False
                    residual_fill_score = 0
                    residual_source_fill = 0

                    if conflict_count > 0:
                        best_fit: Optional[Tuple[int, int, int]] = None

                        # Search only the residual groups owned by Cg % P.
                        for group_index, residual_group in enumerate(residual_pool):
                            if residual_group.mask & conflict_mask:
                                continue
                            if len(residual_group.sources | {column_index}) > mux_size:
                                continue

                            fit_score = (
                                residual_group.mask.bit_count(),
                                len(residual_group.sources),
                                -group_index,
                            )
                            if best_fit is None or fit_score > best_fit:
                                best_fit = fit_score
                                residual_index = group_index

                        if residual_index is None:
                            if (
                                len(residual_pool)
                                >= max_residual_groups_per_lane
                            ):
                                continue
                            residual_index = len(residual_pool)
                            opens_residual = True
                        else:
                            selected_residual = residual_pool[residual_index]
                            residual_fill_score = selected_residual.mask.bit_count()
                            residual_source_fill = len(selected_residual.sources)

                    net_gain = (
                        accepted_count
                        - conflict_count
                        - new_residual_group_penalty * int(opens_residual)
                    )
                    score = (
                        net_gain,
                        -int(opens_residual),
                        accepted_count,
                        -conflict_count,
                        residual_fill_score,
                        residual_source_fill,
                        column_nnz[column_index],
                        -column_index,
                    )

                    if best_score is None or score > best_score:
                        best_score = score
                        best_column = column_index
                        best_accepted_mask = accepted_mask
                        best_conflict_mask = conflict_mask
                        best_conflict_count = conflict_count
                        best_residual_index = residual_index
                        best_opens_residual = opens_residual

                if best_column is None:
                    break

                for row_index in iter_mask_rows(best_accepted_mask):
                    regular_placement[row_index] = best_column

                regular_mask |= best_accepted_mask
                regular_sources.append(best_column)

                if best_conflict_count > 0:
                    if best_residual_index is None:
                        raise AssertionError("Residual destination is missing")

                    if best_opens_residual:
                        if best_residual_index != len(residual_pool):
                            raise AssertionError("New residual index is inconsistent")
                        residual_pool.append(ResidualGroupState())

                    destination = residual_pool[best_residual_index]
                    if destination.mask & best_conflict_mask:
                        raise AssertionError("Unexpected residual row conflict")

                    for row_index in iter_mask_rows(best_conflict_mask):
                        destination.placement[row_index] = best_column

                    destination.mask |= best_conflict_mask
                    destination.sources.add(best_column)

                regular_conflict_count += best_conflict_count
                remaining.remove(best_column)

            regular_placements.append(regular_placement)
            regular_sources_list.append(regular_sources)
            regular_conflicts.append(regular_conflict_count)
            regular_slots_used += 1

        # ------------------------------------------------------------------
        # Emit the regular phase. Pad the last incomplete 4-wide cycle.
        # ------------------------------------------------------------------
        regular_slots_scheduled = (
            math.ceil(regular_slots_used / parallel_groups) * parallel_groups
        )

        for slot in range(regular_slots_scheduled):
            lane = slot % parallel_groups
            if slot < regular_slots_used:
                append_scheduled_group(
                    placements=placements,
                    gidx=gidx,
                    metadata=metadata,
                    placement=regular_placements[slot],
                    sources=regular_sources_list[slot],
                    group_type=f"regular_lane_{lane}",
                    block_id=block_id,
                    lane=lane,
                    conflicts=regular_conflicts[slot],
                    parallel_groups=parallel_groups,
                )
            else:
                append_scheduled_group(
                    placements=placements,
                    gidx=gidx,
                    metadata=metadata,
                    placement={},
                    sources=[],
                    group_type=f"regular_padding_lane_{lane}",
                    block_id=block_id,
                    lane=lane,
                    conflicts=0,
                    parallel_groups=parallel_groups,
                )

        block_regular_cycles = regular_slots_scheduled // parallel_groups
        metadata.regular_cycles += block_regular_cycles

        # ------------------------------------------------------------------
        # Emit residual rounds. Round r contains one group from every lane.
        # ------------------------------------------------------------------
        residual_rounds = max(
            (len(pool) for pool in lane_residual_groups),
            default=0,
        )

        for residual_round in range(residual_rounds):
            for lane in range(parallel_groups):
                pool = lane_residual_groups[lane]
                if residual_round < len(pool):
                    residual_group = pool[residual_round]
                    append_scheduled_group(
                        placements=placements,
                        gidx=gidx,
                        metadata=metadata,
                        placement=residual_group.placement,
                        sources=sorted(residual_group.sources),
                        group_type=(
                            f"residual_round_{residual_round}_lane_{lane}"
                        ),
                        block_id=block_id,
                        lane=lane,
                        conflicts=0,
                        parallel_groups=parallel_groups,
                    )
                    metadata.residual_nnz += len(residual_group.placement)
                    metadata.residual_source_references += len(
                        residual_group.sources
                    )
                else:
                    append_scheduled_group(
                        placements=placements,
                        gidx=gidx,
                        metadata=metadata,
                        placement={},
                        sources=[],
                        group_type=(
                            f"residual_padding_round_{residual_round}_lane_{lane}"
                        ),
                        block_id=block_id,
                        lane=lane,
                        conflicts=0,
                        parallel_groups=parallel_groups,
                    )

        metadata.residual_cycles += residual_rounds
        metadata.total_cycles += block_regular_cycles + residual_rounds
        metadata.blocks += 1
        block_id += 1

    validate_lossless(column_masks, placements, mux_size, gidx)
    scheduled_packed = build_packed_tensor(n_rows, placements, compact.device)

    # Every scheduled cycle must contain exactly P physical group slots.
    if scheduled_packed.shape[1] % parallel_groups != 0:
        raise AssertionError("Scheduled packed width is not P-aligned")
    if metadata.total_cycles * parallel_groups != scheduled_packed.shape[1]:
        raise AssertionError("Cycle count and scheduled width disagree")

    if print_result:
        print("\n<<Modulo-Lane Residual Column Combining>>")
        print_mat(scheduled_packed)
        for physical_cg, source_columns in enumerate(gidx):
            print(
                f"Cg {physical_cg}: "
                f"lane={metadata.group_lanes[physical_cg]}, "
                f"type={metadata.group_types[physical_cg]}, "
                f"block={metadata.group_blocks[physical_cg]}, "
                f"sources={source_columns}, "
                f"conflicts={metadata.group_conflicts[physical_cg]}"
            )
        print(
            f"cycles={metadata.total_cycles} "
            f"(regular={metadata.regular_cycles}, "
            f"residual={metadata.residual_cycles}), "
            f"regular_groups={metadata.regular_groups}, "
            f"residual_groups={metadata.residual_groups}, "
            f"padding_slots={metadata.padding_groups}"
        )

    return PackingResult(
        scheduled_packed=scheduled_packed,
        group_len=[len(source_columns) for source_columns in gidx],
        gidx=gidx,
        metadata=metadata,
    )


# =============================================================================
# Experiment
# =============================================================================


@dataclass
class TrialMetrics:
    original_nnz: int
    lossless_groups: int
    lossless_cycles: int
    proposed_regular_groups: int
    proposed_residual_groups: int
    proposed_padding_groups: int
    proposed_regular_cycles: int
    proposed_residual_cycles: int
    proposed_total_cycles: int
    proposed_residual_nnz: int
    proposed_residual_source_references: int

    @property
    def speedup(self) -> float:
        if self.proposed_total_cycles == 0:
            return 0.0
        return self.lossless_cycles / self.proposed_total_cycles

    @property
    def cycle_reduction_percent(self) -> float:
        if self.lossless_cycles == 0:
            return 0.0
        return (
            (self.lossless_cycles - self.proposed_total_cycles)
            / self.lossless_cycles
            * 100.0
        )

    @property
    def residual_nnz_percent(self) -> float:
        if self.original_nnz == 0:
            return 0.0
        return self.proposed_residual_nnz / self.original_nnz * 100.0


def compare_one_matrix(
    matrix: torch.Tensor,
    tile_size: int,
    mux_size: int,
    parallel_groups: int,
    reuse_depth: int,
    max_residual_groups_per_lane: int,
    max_conflict_ratio: Optional[float],
    new_residual_group_penalty: int,
    candidate_search_limit: Optional[int],
) -> TrialMetrics:
    n_rows, _ = matrix.shape
    tile_count = math.ceil(n_rows / tile_size)

    original_nnz = 0
    lossless_groups = 0
    lossless_cycles = 0

    proposed_regular_groups = 0
    proposed_residual_groups = 0
    proposed_padding_groups = 0
    proposed_regular_cycles = 0
    proposed_residual_cycles = 0
    proposed_total_cycles = 0
    proposed_residual_nnz = 0
    proposed_residual_source_references = 0

    for tile_index in range(tile_count):
        start = tile_index * tile_size
        end = min((tile_index + 1) * tile_size, n_rows)
        tile = matrix[start:end]
        tile_rows = end - start

        prepared = prepare_columns(tile)
        tile_nnz = int((prepared.compact >= 0).sum().item())
        original_nnz += tile_nnz

        baseline_packed, _ = column_combine_lossless_groupwise(
            None,
            mux_size=mux_size,
            candidate_search_limit=candidate_search_limit,
            prepared=prepared,
        )
        baseline_groups = baseline_packed.shape[1]
        lossless_groups += baseline_groups
        lossless_cycles += math.ceil(baseline_groups / parallel_groups)

        max_conflict = (
            None
            if max_conflict_ratio is None
            else math.floor(tile_rows * max_conflict_ratio)
        )

        proposed = column_combine_modulo_residual(
            None,
            mux_size=mux_size,
            reuse_depth=reuse_depth,
            max_residual_groups_per_lane=max_residual_groups_per_lane,
            parallel_groups=parallel_groups,
            max_conflict=max_conflict,
            new_residual_group_penalty=new_residual_group_penalty,
            candidate_search_limit=candidate_search_limit,
            prepared=prepared,
        )

        proposed_output_nnz = int((proposed.packed >= 0).sum().item())
        if proposed_output_nnz != tile_nnz:
            raise AssertionError(
                f"Tile {tile_index} NNZ mismatch: "
                f"input={tile_nnz}, output={proposed_output_nnz}"
            )

        md = proposed.metadata
        proposed_regular_groups += md.regular_groups
        proposed_residual_groups += md.residual_groups
        proposed_padding_groups += md.padding_groups
        proposed_regular_cycles += md.regular_cycles
        proposed_residual_cycles += md.residual_cycles
        proposed_total_cycles += md.total_cycles
        proposed_residual_nnz += md.residual_nnz
        proposed_residual_source_references += md.residual_source_references

    return TrialMetrics(
        original_nnz=original_nnz,
        lossless_groups=lossless_groups,
        lossless_cycles=lossless_cycles,
        proposed_regular_groups=proposed_regular_groups,
        proposed_residual_groups=proposed_residual_groups,
        proposed_padding_groups=proposed_padding_groups,
        proposed_regular_cycles=proposed_regular_cycles,
        proposed_residual_cycles=proposed_residual_cycles,
        proposed_total_cycles=proposed_total_cycles,
        proposed_residual_nnz=proposed_residual_nnz,
        proposed_residual_source_references=proposed_residual_source_references,
    )


def mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def pstdev(values: Sequence[float]) -> float:
    return statistics.pstdev(values) if len(values) >= 2 else 0.0


def summarize_trials(
    sa_size: int,
    parallel_groups: int,
    reuse_depth: int,
    max_residual_groups_per_lane: int,
    trials: Sequence[TrialMetrics],
) -> Dict[str, float]:
    def values(attribute: str) -> List[float]:
        return [float(getattr(trial, attribute)) for trial in trials]

    speedups = [trial.speedup for trial in trials]
    reductions = [trial.cycle_reduction_percent for trial in trials]
    residual_percent = [trial.residual_nnz_percent for trial in trials]

    return {
        "sa_size": float(sa_size),
        "parallel_groups": float(parallel_groups),
        "reuse_depth": float(reuse_depth),
        "max_residual_groups_per_lane": float(max_residual_groups_per_lane),
        "trials": float(len(trials)),
        "lossless_groups_mean": mean(values("lossless_groups")),
        "lossless_cycles_mean": mean(values("lossless_cycles")),
        "proposed_regular_groups_mean": mean(values("proposed_regular_groups")),
        "proposed_residual_groups_mean": mean(values("proposed_residual_groups")),
        "proposed_padding_groups_mean": mean(values("proposed_padding_groups")),
        "proposed_regular_cycles_mean": mean(values("proposed_regular_cycles")),
        "proposed_residual_cycles_mean": mean(values("proposed_residual_cycles")),
        "proposed_total_cycles_mean": mean(values("proposed_total_cycles")),
        "speedup_mean": mean(speedups),
        "speedup_std": pstdev(speedups),
        "cycle_reduction_percent_mean": mean(reductions),
        "cycle_reduction_percent_std": pstdev(reductions),
        "residual_nnz_percent_mean": mean(residual_percent),
        "residual_source_references_mean": mean(
            values("proposed_residual_source_references")
        ),
    }


def print_summary(summary: Dict[str, float]) -> None:
    print(
        f"\n>> [SA {int(summary['sa_size'])}, "
        f"P={int(summary['parallel_groups'])}, "
        f"depth={int(summary['reuse_depth'])}, "
        f"R/lane={int(summary['max_residual_groups_per_lane'])}]"
    )
    print(f"   Lossless groups          : {summary['lossless_groups_mean']:.2f}")
    print(f"   Lossless 4-wide cycles   : {summary['lossless_cycles_mean']:.2f}")
    print(f"   Proposed regular groups  : {summary['proposed_regular_groups_mean']:.2f}")
    print(f"   Proposed residual groups : {summary['proposed_residual_groups_mean']:.2f}")
    print(f"   Padding group slots      : {summary['proposed_padding_groups_mean']:.2f}")
    print(f"   Proposed regular cycles  : {summary['proposed_regular_cycles_mean']:.2f}")
    print(f"   Proposed residual cycles : {summary['proposed_residual_cycles_mean']:.2f}")
    print(f"   Proposed total cycles    : {summary['proposed_total_cycles_mean']:.2f}")
    print(
        f"   Speedup vs lossless      : {summary['speedup_mean']:.4f}x "
        f"(std {summary['speedup_std']:.4f})"
    )
    print(
        f"   Cycle reduction          : "
        f"{summary['cycle_reduction_percent_mean']:.2f}%"
    )
    print(
        f"   NNZ moved to residual    : "
        f"{summary['residual_nnz_percent_mean']:.2f}%"
    )


def write_csv(path: Path, summaries: Sequence[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(summaries[0].keys()) if summaries else []
    with path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare strict tile-wise lossless Column Combining with modulo-"
            "lane residual sharing for a P-wide Overlap SA."
        )
    )
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--sa-sizes", type=int, nargs="+", default=[256])
    parser.add_argument("--column-multiplier", type=int, default=4)
    parser.add_argument("--density", type=float, default=0.10)
    parser.add_argument("--mux-size", type=int, default=8)
    parser.add_argument(
        "--parallel-groups",
        type=int,
        default=4,
        help="Number of column groups executed per cycle.",
    )
    parser.add_argument(
        "--reuse-depth",
        type=int,
        default=4,
        help=(
            "Number of regular cycles whose same-lane groups share residuals. "
            "Depth 4 with P=4 groups Cg 0,4,8,12 together."
        ),
    )
    parser.add_argument(
        "--max-residual-groups-per-lane",
        type=int,
        nargs="+",
        default=[1],
        help=(
            "Residual rounds per lane to sweep. With P=4 and R=3, a block "
            "may contain up to 12 nonempty residual groups but only 3 cycles."
        ),
    )
    parser.add_argument("--new-residual-group-penalty", type=int, default=1)
    parser.add_argument("--max-conflict-ratio", type=float, default=0.25)
    parser.add_argument(
        "--candidate-search-limit",
        type=int,
        default=0,
        help="0 means exhaustive search.",
    )
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("cc_modulo_residual.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.rows <= 0:
        raise ValueError("rows must be greater than 0")
    if args.trials <= 0:
        raise ValueError("trials must be greater than 0")
    if args.column_multiplier <= 0:
        raise ValueError("column-multiplier must be greater than 0")
    if args.parallel_groups <= 0:
        raise ValueError("parallel-groups must be greater than 0")
    if args.reuse_depth <= 0:
        raise ValueError("reuse-depth must be greater than 0")
    if any(value <= 0 for value in args.max_residual_groups_per_lane):
        raise ValueError("Every R/lane value must be greater than 0")

    max_conflict_ratio: Optional[float] = (
        None if args.max_conflict_ratio < 0 else args.max_conflict_ratio
    )
    candidate_search_limit: Optional[int] = (
        None
        if args.candidate_search_limit <= 0
        else args.candidate_search_limit
    )

    print("\n=== Experiment configuration ===")
    print(f"Rows                         : {args.rows}")
    print(f"SA sizes                     : {args.sa_sizes}")
    print(f"Columns per SA size          : {args.column_multiplier}x")
    print(f"Density                      : {args.density}")
    print(f"MUX size                     : {args.mux_size}")
    print(f"Parallel groups/cycle (P)    : {args.parallel_groups}")
    print(f"Same-lane reuse depth        : {args.reuse_depth}")
    print(f"Residual rounds/lane sweep   : {args.max_residual_groups_per_lane}")
    print(f"Conflict ratio               : {max_conflict_ratio}")
    print(f"Candidate search limit       : {candidate_search_limit}")
    print(f"Trials                       : {args.trials}")

    summaries: List[Dict[str, float]] = []

    for sa_size in args.sa_sizes:
        n_cols = sa_size * args.column_multiplier

        for residual_groups_per_lane in args.max_residual_groups_per_lane:
            trials: List[TrialMetrics] = []
            progress = tqdm(
                range(args.trials),
                desc=(
                    f"SA {sa_size}, depth={args.reuse_depth}, "
                    f"R/lane={residual_groups_per_lane}"
                ),
                leave=False,
            )

            for trial_index in progress:
                # All R/lane settings receive the same matrix for this trial.
                generator = torch.Generator().manual_seed(
                    args.seed + sa_size * 100_000 + trial_index
                )
                matrix = generate_sparse_matrix(
                    n_rows=args.rows,
                    n_cols=n_cols,
                    density=args.density,
                    generator=generator,
                )

                metrics = compare_one_matrix(
                    matrix=matrix,
                    tile_size=sa_size,
                    mux_size=args.mux_size,
                    parallel_groups=args.parallel_groups,
                    reuse_depth=args.reuse_depth,
                    max_residual_groups_per_lane=residual_groups_per_lane,
                    max_conflict_ratio=max_conflict_ratio,
                    new_residual_group_penalty=args.new_residual_group_penalty,
                    candidate_search_limit=candidate_search_limit,
                )
                trials.append(metrics)

            summary = summarize_trials(
                sa_size=sa_size,
                parallel_groups=args.parallel_groups,
                reuse_depth=args.reuse_depth,
                max_residual_groups_per_lane=residual_groups_per_lane,
                trials=trials,
            )
            summaries.append(summary)
            print_summary(summary)

    write_csv(args.csv, summaries)
    print(f"\nCSV saved to: {args.csv.resolve()}")


if __name__ == "__main__":
    main()
