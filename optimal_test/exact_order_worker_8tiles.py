#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from dataclasses import asdict
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--packing', required=True)
    p.add_argument('--base-oracle', required=True)
    p.add_argument('--row-exact', required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--density', type=float, default=0.20)
    p.add_argument('--tile-count', type=int, default=8)
    p.add_argument('--tile-rows', type=int, default=8)
    p.add_argument('--tile-cols', type=int, default=64)
    p.add_argument('--order', nargs='+', type=int, required=True)
    p.add_argument('--pair-timeout', type=float, default=300.0)
    p.add_argument('--output', required=True)
    args=p.parse_args()

    if len(args.order) != args.tile_count:
        raise ValueError(f'order length {len(args.order)} != tile_count {args.tile_count}')
    if sorted(args.order) != list(range(args.tile_count)):
        raise ValueError(f'order must be a permutation of 0..{args.tile_count-1}: {args.order}')

    sys.path.insert(0,args.root)
    import torch
    torch.set_num_threads(1)
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass
    from run_ctf_greedy_pairing_benchmarks import (
        CTFConfig, evaluate_exact_pairing_sequence, generate_tiles, load_module
    )
    pk=load_module(args.packing,'worker_pk')
    base=load_module(args.base_oracle,'worker_base')
    row=load_module(args.row_exact,'worker_row')
    config=CTFConfig(mux_size=4,reuse_depth=2,max_residual_groups_per_lane=1,parallel_groups=4,max_conflict=2)
    tiles=generate_tiles(args.tile_rows,args.tile_cols,args.tile_count,args.density,args.seed)
    result=evaluate_exact_pairing_sequence(
        pk,base,row,tiles,args.order,config,
        timeout_seconds=args.pair_timeout,max_leaf_states=0,max_pair_options=0
    )
    Path(args.output).write_text(json.dumps(asdict(result),ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__': main()
