#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, itertools, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List


def write_csv(path:Path, rows:List[Dict]):
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root', default='.')
    p.add_argument('--packing', default='packing.py')
    p.add_argument('--base-oracle', default='ctf_optimality_oracle_dp_lb.py')
    p.add_argument('--row-exact', default='ctf_row_pairing_exact_oracle_memo.py')
    p.add_argument('--worker-script', default='exact_order_worker_8tiles.py')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--density', type=float, default=0.20)
    p.add_argument('--tile-count', type=int, default=8)
    p.add_argument('--tile-rows', type=int, default=8)
    p.add_argument('--tile-cols', type=int, default=64)
    p.add_argument('--workers', type=int, default=10)
    p.add_argument('--pair-timeout', type=float, default=300.0)
    p.add_argument('--order-timeout', type=float, default=1200.0)
    p.add_argument('--swap-candidates', type=int, default=64)
    p.add_argument('--swap-passes', type=int, default=1)
    p.add_argument('--output-dir', default='greedy_vs_exhaustive_local_exact_8tiles')
    p.add_argument('--resume', action='store_true')
    args=p.parse_args()
    root=Path(args.root).resolve(); sys.path.insert(0,str(root))
    import torch
    torch.set_num_threads(1)
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass
    from run_ctf_greedy_pairing_benchmarks import (
        CTFConfig, choose_order_first_fit_greedy, evaluate_first_fit_sequence,
        evaluate_greedy_pairing_sequence, generate_tiles, load_module, safe_sequence_result
    )
    out=Path(args.output_dir); order_dir=out/'orders'; order_dir.mkdir(parents=True,exist_ok=True)
    pk=load_module(str((root/args.packing).resolve()),'parent_pk')
    config=CTFConfig(mux_size=4,reuse_depth=2,max_residual_groups_per_lane=1,parallel_groups=4,max_conflict=2)
    tiles=generate_tiles(args.tile_rows,args.tile_cols,args.tile_count,args.density,args.seed)
    hs=time.perf_counter()
    greedy_order, order_search=choose_order_first_fit_greedy(pk,tiles,config)
    ordered_ff=evaluate_first_fit_sequence(pk,tiles,greedy_order,config)
    raw=evaluate_greedy_pairing_sequence(pk,tiles,greedy_order,config,swap_candidates=args.swap_candidates,swap_passes=args.swap_passes)
    heuristic=safe_sequence_result(raw,ordered_ff)
    heuristic_wall=time.perf_counter()-hs

    env=os.environ.copy()
    for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
        env[key]='1'

    orders=list(itertools.permutations(range(args.tile_count)))
    orders_total=len(orders)

    def run_order(order):
        tag='_'.join(map(str,order)); path=order_dir/f'order_{tag}.json'
        if args.resume and path.exists():
            return json.loads(path.read_text(encoding='utf-8'))
        cmd=[sys.executable,str((root/args.worker_script).resolve()),
             '--root',str(root),'--packing',str((root/args.packing).resolve()),
             '--base-oracle',str((root/args.base_oracle).resolve()),
             '--row-exact',str((root/args.row_exact).resolve()),
             '--seed',str(args.seed),'--density',str(args.density),
             '--tile-count',str(args.tile_count),'--tile-rows',str(args.tile_rows),'--tile-cols',str(args.tile_cols),
             '--order',*map(str,order),'--pair-timeout',str(args.pair_timeout),
             '--output',str(path)]
        started=time.perf_counter()
        try:
            subprocess.run(cmd,check=True,timeout=args.order_timeout,env=env,capture_output=True,text=True)
            row=json.loads(path.read_text(encoding='utf-8'))
            row['hard_timeout']=False; row['worker_wall_seconds']=time.perf_counter()-started; row['error']=''
        except subprocess.TimeoutExpired:
            row={'order':list(order),'exact_certified':False,'hard_timeout':True,'worker_wall_seconds':time.perf_counter()-started,'error':f'hard order timeout after {args.order_timeout}s'}
        except subprocess.CalledProcessError as e:
            row={'order':list(order),'exact_certified':False,'hard_timeout':False,'worker_wall_seconds':time.perf_counter()-started,'error':(e.stderr or e.stdout or str(e))[-2000:]}
        path.write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding='utf-8')
        return row

    exact_start=time.perf_counter(); rows=[]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs={ex.submit(run_order,o):o for o in orders}
        for fut in as_completed(futs):
            r=fut.result(); rows.append(r)
            print(f"order={r.get('order')} groups={r.get('total_groups')} exact={r.get('exact_certified')} timeout={r.get('hard_timeout')}", flush=True)
    exact_wall=time.perf_counter()-exact_start
    rows.sort(key=lambda r:tuple(r.get('order',[])))
    completed=[r for r in rows if not r.get('error') and r.get('total_groups') is not None]
    certified=[r for r in completed if r.get('exact_certified')]
    def q(r): return (int(r['total_groups']),int(r['total_cycles']),int(r['total_physical_slots']),-int(r['total_moved_nnz']),tuple(r['order']))
    best_cert=min(certified,key=q) if certified else None
    best_found=min(completed,key=q) if completed else None
    all_exact=len(certified)==orders_total
    ref=best_cert if all_exact else best_found
    summary={
      'definition':{
        'heuristic':'greedy tile order under first-fit + greedy row-pairing/swap CTF',
        'reference':f'all {orders_total} tile orders; sequential row-pairing-aware exact local CTF per adjacent pair',
        'important_scope':'reference is exhaustive for the hierarchical local-exact policy, not a joint sequence-wide CTF optimum'},
      'seed':args.seed,'density':args.density,
      'tile_shape':[args.tile_count,args.tile_rows,args.tile_cols],
      'heuristic':{**asdict(heuristic),'tile_order_search_seconds':order_search,'wall_seconds':heuristic_wall},
      'reference_status':{'orders_total':orders_total,'orders_completed':len(completed),'orders_certified':len(certified),'all_orders_certified':all_exact,'wall_seconds':exact_wall,'best_certified':best_cert,'best_found':best_found}
    }
    if ref:
      summary['comparison']={'reference_is_certified':all_exact,'heuristic_groups':heuristic.total_groups,'reference_groups':ref['total_groups'],'group_gap_pct':(heuristic.total_groups-int(ref['total_groups']))/int(ref['total_groups'])*100,'heuristic_cycles':heuristic.total_cycles,'reference_cycles':ref['total_cycles'],'cycle_gap_pct':(heuristic.total_cycles-int(ref['total_cycles']))/int(ref['total_cycles'])*100}
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'all_orders.csv',rows)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    comp=[{'label':'Greedy order + heuristic CTF','order':heuristic.order,'total_groups':heuristic.total_groups,'total_cycles':heuristic.total_cycles,'runtime_seconds':heuristic_wall,'certified':False}]
    if ref: comp.append({'label':'Exhaustive order + local exact CTF' if all_exact else 'Best found exhaustive-order candidate','order':ref['order'],'total_groups':ref['total_groups'],'total_cycles':ref['total_cycles'],'runtime_seconds':exact_wall,'certified':all_exact})
    write_csv(out/'comparison.csv',comp)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
