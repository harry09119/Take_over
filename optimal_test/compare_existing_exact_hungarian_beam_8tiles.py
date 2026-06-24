#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare existing CTF, optional exact reference, and New CTF on N 8x64 tiles.

This is the original four-tile script with only the hard-coded tile count/order
parts generalized. Default is eight 8x64 tiles.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

from ctf_greedy_row_pairing import CTFConfig, count_nnz, pack_tile
from ctf_hungarian_beam import BeamSettings, hungarian_beam_transition
from run_ctf_greedy_pairing_benchmarks import (
    SequenceResult,
    evaluate_first_fit_sequence,
    generate_tiles,
)


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, os.path.abspath(path))
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _accumulate(metrics, now_after: torch.Tensor, totals: Dict[str, int]) -> None:
    totals['groups'] += metrics.groups
    totals['regular'] += metrics.regular_groups
    totals['residual'] += metrics.residual_groups
    totals['physical'] += metrics.physical_slots
    totals['cycles'] += metrics.cycles
    totals['processed_nnz'] += count_nnz(now_after)


def evaluate_new_sequence(pk, tiles: Sequence[torch.Tensor], order: Sequence[int], config: CTFConfig, settings: BeamSettings) -> SequenceResult:
    start = time.perf_counter()
    current = tiles[int(order[0])].clone()
    totals = dict(groups=0, regular=0, residual=0, physical=0, cycles=0, processed_nnz=0)
    moved_total = 0
    runtimes: List[float] = []
    transition_groups: List[int] = []
    transition_moved: List[int] = []
    fallback_count = 0

    for next_id in order[1:]:
        current_metrics = pack_tile(pk, current, config)
        transition = hungarian_beam_transition(
            pk,
            current_metrics.result.scheduled_packed,
            current_metrics.result.gidx,
            tiles[int(next_id)],
            config,
            settings=settings,
        )
        _accumulate(current_metrics, transition.now_after, totals)
        moved_total += transition.moved_nnz
        runtimes.append(transition.runtime_seconds)
        transition_groups.append(transition.next_groups)
        transition_moved.append(transition.moved_nnz)
        fallback_count += int(transition.used_first_fit_fallback)
        current = transition.next_after

    final_metrics = pack_tile(pk, current, config)
    totals['groups'] += final_metrics.groups
    totals['regular'] += final_metrics.regular_groups
    totals['residual'] += final_metrics.residual_groups
    totals['physical'] += final_metrics.physical_slots
    totals['cycles'] += final_metrics.cycles
    totals['processed_nnz'] += count_nnz(current)

    original_nnz = sum(count_nnz(tile) for tile in tiles)
    if totals['processed_nnz'] != original_nnz:
        raise AssertionError(f"new CTF sequence NNZ mismatch: {totals['processed_nnz']} != {original_nnz}")

    return SequenceResult(
        method='greedy_order_hungarian_row_pairing_nnz_beam',
        order=[int(x) for x in order],
        total_groups=totals['groups'],
        total_regular_groups=totals['regular'],
        total_residual_groups=totals['residual'],
        total_physical_slots=totals['physical'],
        total_cycles=totals['cycles'],
        total_moved_nnz=moved_total,
        original_nnz=original_nnz,
        processed_nnz=totals['processed_nnz'],
        lossless_verified=True,
        exact_certified=False,
        runtime_seconds=time.perf_counter() - start,
        transition_runtimes=runtimes,
        transition_groups=transition_groups,
        transition_moved=transition_moved,
        notes=f'first-fit fallback used in {fallback_count} transitions',
    )


def choose_order_greedy_new(pk, tiles: Sequence[torch.Tensor], config: CTFConfig, settings: BeamSettings) -> Tuple[List[int], float]:
    start = time.perf_counter()
    best = None
    for first in range(len(tiles)):
        current = tiles[first].clone()
        order = [first]
        remaining = set(range(len(tiles))) - {first}
        groups = cycles = physical = moved = 0

        while remaining:
            current_metrics = pack_tile(pk, current, config)
            candidates = []
            for next_id in sorted(remaining):
                transition = hungarian_beam_transition(
                    pk,
                    current_metrics.result.scheduled_packed,
                    current_metrics.result.gidx,
                    tiles[next_id],
                    config,
                    settings=settings,
                )
                next_metrics = pack_tile(pk, transition.next_after, config)
                score = (
                    groups + current_metrics.groups + next_metrics.groups,
                    cycles + current_metrics.cycles + next_metrics.cycles,
                    physical + current_metrics.physical_slots + next_metrics.physical_slots,
                    -(moved + transition.moved_nnz),
                    next_id,
                )
                candidates.append((score, next_id, transition.next_after, transition.moved_nnz))
            score, selected, modified_next, selected_moved = min(candidates, key=lambda x: x[0])
            groups += current_metrics.groups
            cycles += current_metrics.cycles
            physical += current_metrics.physical_slots
            moved += selected_moved
            order.append(selected)
            remaining.remove(selected)
            current = modified_next

        final = pack_tile(pk, current, config)
        key = (groups + final.groups, cycles + final.cycles, physical + final.physical_slots, -moved, tuple(order))
        if best is None or key < best[0]:
            best = (key, list(order))
    if best is None:
        raise RuntimeError('tile-order search failed')
    return best[1], time.perf_counter() - start


def read_exact_reference(exact_summary: str) -> Tuple[dict, bool]:
    payload = json.loads(Path(exact_summary).read_text(encoding='utf-8'))
    status = payload.get('reference_status', {})
    all_exact = bool(status.get('all_orders_certified'))
    ref = status.get('best_certified') if all_exact else status.get('best_found')
    if not ref:
        raise RuntimeError('exact-reference summary has no usable result')
    return ref, all_exact


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--packing', default='packing.py')
    p.add_argument('--seed', type=int, default=4)
    p.add_argument('--density', type=float, default=0.20)
    p.add_argument('--tile-count', type=int, default=8)
    p.add_argument('--tile-rows', type=int, default=8)
    p.add_argument('--tile-cols', type=int, default=64)
    p.add_argument('--beam-width', type=int, default=16)
    p.add_argument('--row-option-limit', type=int, default=32)
    p.add_argument('--exact-summary', default='')
    p.add_argument('--output-dir', default='ctf_existing_exact_new_8tiles')
    args = p.parse_args()

    torch.set_num_threads(1)
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass

    pk = load_module(args.packing, 'comparison_pk')
    config = CTFConfig(mux_size=4, reuse_depth=2, max_residual_groups_per_lane=1, parallel_groups=4, max_conflict=2)
    settings = BeamSettings(beam_width=args.beam_width, row_option_limit=args.row_option_limit, dense_tiebreak=True, fallback_to_first_fit=True)
    tiles = generate_tiles(args.tile_rows, args.tile_cols, args.tile_count, args.density, args.seed)
    natural_order = list(range(args.tile_count))

    existing = evaluate_first_fit_sequence(pk, tiles, natural_order, config)
    order, order_time = choose_order_greedy_new(pk, tiles, config, settings)
    new = evaluate_new_sequence(pk, tiles, order, config, settings)

    rows: List[Dict[str, object]] = [
        {'label': 'Existing first-fit', **asdict(existing), 'order_search_seconds': 0.0, 'runtime_including_order_search': existing.runtime_seconds, 'reference_certified': False},
        {'label': 'New CTF', **asdict(new), 'order_search_seconds': order_time, 'runtime_including_order_search': new.runtime_seconds + order_time, 'reference_certified': False},
    ]

    exact_ref = None
    exact_certified = False
    if args.exact_summary:
        exact_ref, exact_certified = read_exact_reference(args.exact_summary)
        rows.insert(1, {'label': 'Exact reference' if exact_certified else 'Best found exact-search candidate', **exact_ref, 'order_search_seconds': 0.0, 'runtime_including_order_search': exact_ref.get('runtime_seconds', 0.0), 'reference_certified': exact_certified})

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    write_csv(out / 'comparison.csv', rows)
    summary = {
        'configuration': {
            'tile_size': f'{args.tile_rows}x{args.tile_cols}', 'tile_count': args.tile_count, 'density': args.density,
            'mux_size': 4, 'reuse_depth': 2, 'max_residual_groups_per_lane': 1,
            'beam_width': args.beam_width, 'row_option_limit': args.row_option_limit,
        },
        'existing': rows[0],
        'new': rows[-1],
        'exact_reference': exact_ref,
        'exact_reference_certified': exact_certified,
    }
    if exact_ref:
        summary['new_gap_to_reference_pct'] = (new.total_groups - int(exact_ref['total_groups'])) / int(exact_ref['total_groups']) * 100.0
    summary['new_improvement_over_existing_pct'] = (existing.total_groups - new.total_groups) / existing.total_groups * 100.0
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
