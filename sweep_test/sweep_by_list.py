#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import math
import multiprocessing as mp
from functools import partial
from typing import Any, Dict, List, Tuple, Union

import torch
from tqdm import tqdm

# Try local/parent import for packing.py
try:
    import lib_packing as pk
except ModuleNotFoundError:
    import sys

    this_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(this_dir, os.pardir))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    import packing as pk  # type: ignore

PAD_VALUE = -1


# -----------------------------------------------------------------------------
# Fast helpers (density_check monkey patch)
# -----------------------------------------------------------------------------
def density_check_fast(matrix: torch.Tensor, pad_value: int | float = PAD_VALUE):
    n, m = matrix.shape
    total = n * m
    if total == 0:
        return 0, 0, 0
    nonzero = int((matrix != pad_value).sum().item())
    density = round(nonzero / total, 3)
    return density, nonzero, m


def fast_nnz(matrix: torch.Tensor, pad_value: int | float = PAD_VALUE) -> int:
    if matrix.numel() == 0:
        return 0
    return int((matrix != pad_value).sum().item())


def _worker_init(torch_threads: int):
    try:
        torch.set_num_threads(max(1, int(torch_threads)))
    except Exception:
        pass
    pk.density_check = density_check_fast  # type: ignore[attr-defined]


# -----------------------------------------------------------------------------
# Random matrix + magnitude pruning (keep top fraction)
# -----------------------------------------------------------------------------
def topk_keep_mask(w: torch.Tensor, keep_ratio: float) -> torch.Tensor:
    """Keep top keep_ratio fraction by |w|. Returns bool mask (True=keep)."""
    if keep_ratio <= 0.0:
        return torch.zeros_like(w, dtype=torch.bool)
    if keep_ratio >= 1.0:
        return torch.ones_like(w, dtype=torch.bool)

    flat = w.abs().flatten()
    numel = flat.numel()
    k_keep = int(round(keep_ratio * numel))

    if k_keep <= 0:
        return torch.zeros_like(w, dtype=torch.bool)
    if k_keep >= numel:
        return torch.ones_like(w, dtype=torch.bool)

    # kth largest threshold = (numel - k_keep + 1)-th smallest
    kth = numel - k_keep + 1
    thr = flat.kthvalue(kth).values
    return (w.abs() >= thr)


def make_random_pruned_matrix(
    n: int,
    k: int,
    keep_ratio: float,
    pad_value: int,
    *,
    seed: int,
    dist: str = "normal",
) -> torch.Tensor:
    """
    Generate random N×K matrix, keep top keep_ratio by |value|,
    encode kept entries as 1, pruned as PAD_VALUE (int32).
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))

    if dist == "normal":
        w = torch.randn((n, k), generator=gen)
    elif dist == "uniform":
        w = (torch.rand((n, k), generator=gen) * 2.0) - 1.0
    else:
        raise ValueError(f"Unsupported dist: {dist}")

    keep = topk_keep_mask(w, keep_ratio)

    mat = torch.where(
        keep,
        torch.ones((n, k), dtype=torch.int32),
        torch.full((n, k), int(pad_value), dtype=torch.int32),
    )
    return mat


# -----------------------------------------------------------------------------
# Parse --values
# -----------------------------------------------------------------------------
def parse_values(raw_values: List[str], as_float: bool) -> List[Union[int, float]]:
    """
    Supports:
      --values 32 64 128
      --values 32,64,128
      --values "32,64,128" "256"  (mixed)
    """
    tokens: List[str] = []
    for rv in raw_values:
        parts = [p.strip() for p in str(rv).split(",")]
        for p in parts:
            if p != "":
                tokens.append(p)

    if len(tokens) == 0:
        raise ValueError("--values is empty after parsing")

    if as_float:
        out: List[Union[int, float]] = [float(t) for t in tokens]
    else:
        # allow "64.0" passed accidentally
        out = [int(round(float(t))) for t in tokens]
    return out


# -----------------------------------------------------------------------------
# Tile stats accumulation (S × sum(cols) 가정)
# -----------------------------------------------------------------------------
def _accum_tile_stats(tile: torch.Tensor) -> Tuple[int, int]:
    cols = int(tile.shape[1])
    nnz = fast_nnz(tile)
    return cols, nnz


def _final_density(s: int, total_cols: int, total_nnz: int) -> float:
    denom = int(s) * int(total_cols)
    if denom <= 0:
        return 0.0
    return float(total_nnz / denom)


# -----------------------------------------------------------------------------
# Packing per matrix -> return ONLY (tile_cols, density) per method
# -----------------------------------------------------------------------------
def _pack_summary(
    mat: torch.Tensor,
    *,
    s: int,
    b: int,
    mux_size: int,
    conflict: float,
    fairness_match_to_cc_lossy: bool,
) -> Dict[str, Dict[str, Union[int, float]]]:
    """Compute packing stats.

    Implements:
      - CC-lossy   : column_combine with allowed dropping (controlled by `conflict`)
      - CC-nodrop  : conflict-free column_combine only (allowed=0)
      - Optional Protocol-1 fairness: if `fairness_match_to_cc_lossy` is True,
        then General/Eureka/OPF are evaluated on a matrix re-pruned to match
        CC-lossy final NNZ.
    """
    # Base matrix: remove empty columns
    base_t = pk.remove_empty(mat)
    nnz_base = fast_nnz(base_t)

    gn, gm = base_t.shape
    gn_tiles = (gn + s - 1) // s
    gm_tiles = (gm + b - 1) // b

    def _run_cc_on(allowed: float) -> Tuple[int, int]:
        total_cols = 0
        total_nnz = 0
        for mi in range(gm_tiles):
            col_start = mi * b
            col_end = min((mi + 1) * b, gm)
            cols = base_t[:, col_start:col_end]
            cc_t, _, _ = pk.column_combine(cols, gn * allowed, mux_size)
            for ni in range(gn_tiles):
                row_start = ni * s
                row_end = min((ni + 1) * s, gn)
                now_t = cc_t[row_start:row_end]
                ccols, cnnz = _accum_tile_stats(now_t)
                total_cols += ccols
                total_nnz += cnnz
        return int(total_cols), int(total_nnz)

    # CC variants
    cc_lossy_cols, cc_lossy_nnz = _run_cc_on(float(conflict))
    #cc_nodrop_cols, cc_nodrop_nnz = _run_cc_on(0.0)

    # Fairness: match NNZ to CC-lossy final NNZ only when enabled
    fairness_match_to_cc_lossy = bool(fairness_match_to_cc_lossy)
    if fairness_match_to_cc_lossy:
        drop_needed = max(0, int(nnz_base - cc_lossy_nnz))
        work_t = pk.re_prune(base_t.to(torch.int32), drop_needed)
        work_t = pk.remove_empty(work_t)
    else:
        drop_needed = 0
        work_t = base_t.to(torch.int32)

    nnz_work = fast_nnz(work_t)
    gn2, gm2 = work_t.shape
    gn2_tiles = (gn2 + s - 1) // s
    gm2_tiles = (gm2 + b - 1) // b

    # General stats on work_t
    gen_total_cols = 0
    gen_total_nnz = 0
    for mi in range(gm2_tiles):
        col_start = mi * b
        col_end = min((mi + 1) * b, gm2)
        for ni in range(gn2_tiles):
            row_start = ni * s
            row_end = min((ni + 1) * s, gn2)
            now_t = work_t[row_start:row_end, col_start:col_end]
            ccols, cnnz = _accum_tile_stats(now_t)
            gen_total_cols += ccols
            gen_total_nnz += cnnz

    # Eureka stats on work_t
    eureka_total_cols = 0
    eureka_total_nnz = 0
    mux_size_ = mux_size# * 2
    en_tiles = (gn2 + s - 1) // s
    em_tiles = (gm2 + b - 1) // b

    for mi in range(em_tiles):
        col_start = mi * b
        col_end = min((mi + 1) * b, gm2)
        for ni in range(en_tiles):
            row_start = ni * s
            row_end = min((ni + 1) * s, gn2)
            tile = work_t[row_start:row_end, col_start:col_end]

            tile_w = int(tile.shape[1])
            if tile_w == 0:
                continue

            et_tiles = (tile_w + mux_size_ - 1) // mux_size_
            for ti in range(et_tiles):
                seg_start = ti * mux_size_
                seg_end = min((ti + 1) * mux_size_, tile_w)
                inner_tile = tile[:, seg_start:seg_end]

                eureka_t, _ = pk.eureka_optimal(inner_tile)
                ccols, cnnz = _accum_tile_stats(eureka_t)
                eureka_total_cols += ccols
                eureka_total_nnz += cnnz

    # CTP stats on work_t
    ctp_total_cols = 0
    ctp_total_nnz = 0
    total_t_ = work_t#pk.reorder_tensor(work_t, "a")
    on_tiles = (gn2 + s - 1) // s
    om_tiles = (gm2 + b - 1) // b

    for mi in range(om_tiles):
        col_start = mi * b
        col_end = min((mi + 1) * b, gm2)
        col_tiles = total_t_[:, col_start:col_end]

        diff = on_tiles * s - gn2
        if diff > 0:
            pad = torch.full((diff, col_tiles.size(1)), PAD_VALUE, dtype=col_tiles.dtype)
            col_tiles = torch.cat([col_tiles, pad], dim=0)

        for ni in range(on_tiles):
            now_start = ni * s
            now_end = min((ni + 1) * s, on_tiles * s)
            now_t = col_tiles[now_start:now_end]

            now_pt, _, now_g = pk.residual_combine(now_t, 0, mux_size)

            if ni < on_tiles - 1:
                next_start = now_end
                next_end = min((ni + 2) * s, on_tiles * s)
                next_t = col_tiles[next_start:next_end]
                now_pt, pruned_t = pk.ctf(now_pt, now_g, next_t, mux_size, s, ope=True)
                col_tiles[next_start:next_end].copy_(pruned_t)

            ccols, cnnz = _accum_tile_stats(now_pt)
            ctp_total_cols += ccols
            ctp_total_nnz += cnnz

    out: Dict[str, Dict[str, Union[int, float]]] = {
        "protocol": {
            "fairness_match_to_cc_lossy": bool(fairness_match_to_cc_lossy),
            "nnz_base": int(nnz_base),
            "nnz_cc_lossy": int(cc_lossy_nnz),
            "nnz_work": int(nnz_work),
            "nnz_drop_needed_for_match": int(drop_needed),
            "extra_drop_ratio_cc_lossy": float(0.0 if nnz_base == 0 else (1.0 - (cc_lossy_nnz / nnz_base))),
        },
        "general": {
            "tile_cols": int(gen_total_cols),
            "density": _final_density(s, int(gen_total_cols), int(gen_total_nnz)),
            "nnz": int(gen_total_nnz),
        },
        "column_combine": {
            "tile_cols": int(cc_lossy_cols),
            "density": _final_density(s, int(cc_lossy_cols), int(cc_lossy_nnz)),
            "nnz": int(cc_lossy_nnz),
        },
        "eureka": {
            "tile_cols": int(eureka_total_cols),
            "density": _final_density(s, int(eureka_total_cols), int(eureka_total_nnz)),
            "nnz": int(eureka_total_nnz),
        },
        "ctp": {
            "tile_cols": int(ctp_total_cols),
            "density": _final_density(s, int(ctp_total_cols), int(ctp_total_nnz)),
            "nnz": int(ctp_total_nnz),
        }
        # Alias key used in paper drafts
    }

    return out


# -----------------------------------------------------------------------------
# One sweep point executor
# -----------------------------------------------------------------------------
def _run_one_point(
    idx_and_value: Tuple[int, Union[int, float]],
    *,
    sweep: str,
    base_n: int,
    base_k: int,
    base_keep: float,
    base_s: int,
    base_mux: int,
    b: int,
    conflict: float,
    seed: int,
    dist: str,
    fix_nm: bool,
    nm_round: str,
) -> Dict[str, Any]:
    idx, v = idx_and_value

    # base params
    n = int(base_n)
    k = int(base_k)
    keep = float(base_keep)
    s = int(base_s)
    mux = int(base_mux)
    cval = float(conflict)

    if sweep in ("density", "keep"):
        keep = float(v)
        keep = max(0.0, min(1.0, keep))

    elif sweep == "ratio":
        r = float(v)
        if r <= 0:
            raise ValueError("ratio must be > 0")

        # N = base_n * r,  M chosen to keep NM near-constant
        n = max(1, int(round(base_n * r)))
        target = int(base_n) * int(base_k)
        k = max(1, int(round(target / n)))

    elif sweep == "n":
        n = max(1, int(v))
        if fix_nm:
            target = int(base_n) * int(base_k)
            if nm_round == "floor":
                k = max(1, int(math.floor(target / n)))
            elif nm_round == "ceil":
                k = max(1, int(math.ceil(target / n)))
            else:
                k = max(1, int(round(target / n)))

    elif sweep in ("k", "m"):
        k = max(1, int(v))
        if fix_nm:
            target = int(base_n) * int(base_k)
            if nm_round == "floor":
                n = max(1, int(math.floor(target / k)))
            elif nm_round == "ceil":
                n = max(1, int(math.ceil(target / k)))
            else:
                n = max(1, int(round(target / k)))

    elif sweep == "s":
        s = max(1, int(v))

    elif sweep in ("mux", "m_mux"):
        mux = max(1, int(v))

    elif sweep in ("c", "conflict"):
        cval = float(v)

    else:
        raise ValueError(f"Unsupported sweep param: {sweep}")

    point_seed = int(seed + idx)

    mat = make_random_pruned_matrix(
        n=n,
        k=k,
        keep_ratio=keep,
        pad_value=PAD_VALUE,
        seed=point_seed,
        dist=dist,
    )

    raw_nnz = fast_nnz(mat)
    raw_den = float(raw_nnz / mat.numel()) if mat.numel() > 0 else 0.0

    summary = _pack_summary(
        mat,
        s=s,
        b=b,
        mux_size=mux,
        conflict=cval,
        fairness_match_to_cc_lossy=(sweep in ("c", "conflict")),
    )

    return {
        "__idx": int(idx),
        "sweep_value": float(v) if isinstance(v, float) else int(v),
        "shape": [int(n), int(k)],
        "params": {
            "n": int(n),
            "k": int(k),
            "nm": int(n) * int(k),
            "keep": float(keep),
            "s": int(s),
            "b": int(b),
            "mux": int(mux),
            "conflict": float(cval),
        },
        "raw": {"nnz": int(raw_nnz), "density": float(raw_den)},
        "result": summary,
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Random N×M -> prune(top-|w| keep) -> pack -> tile_cols+density. Sweep by explicit --values list; save into one JSON."
    )

    # base params
    parser.add_argument("--n", type=int, default=1024, help="base rows N")
    parser.add_argument("--k", type=int, default=1024, help="base cols M (K)")
    parser.add_argument("--keep", type=float, default=0.20, help="base keep ratio (top-|w|)")

    parser.add_argument("--s", type=int, default=64, help="base SA rows S")
    parser.add_argument("--b", type=int, default=256, help="tile width b")
    parser.add_argument("--mux", type=int, default=8, help="base mux size")
    parser.add_argument("--c", type=float, default=0.25, help="conflict threshold factor")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dist", type=str, default="normal", choices=["normal", "uniform"])

    # sweep
    parser.add_argument(
        "--sweep",
        type=str,
        required=True,
        choices=["density", "keep", "ratio", "n", "k", "m", "s", "mux", "m_mux", "c", "conflict"],
        help="Which parameter to sweep. Use --values to specify the list in order.",
    )
    parser.add_argument(
        "--values",
        type=str,
        nargs="+",
        required=True,
        help='Explicit list for sweep. Examples: --values 32 64 128 256  OR  --values 32,64,128,256',
    )

    # keep N*M constant when sweeping n or k (optional)
    parser.add_argument("--fix-nm", action="store_true", help="Keep N*M constant (target=base_n*base_k) when sweeping n or k.")
    parser.add_argument("--nm-round", type=str, default="round", choices=["round", "floor", "ceil"])

    # parallel over sweep points
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--mp-start",
        type=str,
        default="fork" if hasattr(mp, "get_start_method") else "spawn",
        choices=["fork", "spawn", "forkserver"],
    )
    parser.add_argument("--torch-threads", type=int, default=1)

    # output
    parser.add_argument("--out", type=str, default="", help="output json filename (optional)")
    parser.add_argument("--compact-json", action="store_true")

    args = parser.parse_args()

    pk.density_check = density_check_fast  # type: ignore[attr-defined]

    sweep = args.sweep

    # determine value type
    float_sweeps = {"density", "keep", "ratio", "c", "conflict"}
    values = parse_values(args.values, as_float=(sweep in float_sweeps))

    # clamp density/keep to [0,1]
    if sweep in ("density", "keep"):
        values = [max(0.0, min(1.0, float(v))) for v in values]  # type: ignore[list-item]

    idx_values = list(enumerate(values))
    args.b = args.s * 4
    job = partial(
        _run_one_point,
        sweep=sweep,
        base_n=int(args.n),
        base_k=int(args.k),
        base_keep=float(args.keep),
        base_s=int(args.s),
        base_mux=int(args.mux),
        b=int(args.b),
        conflict=float(args.c),
        seed=int(args.seed),
        dist=str(args.dist),
        fix_nm=bool(args.fix_nm),
        nm_round=str(args.nm_round),
    )

    results: List[Dict[str, Any]] = []

    if args.workers <= 1:
        for iv in tqdm(idx_values, total=len(idx_values), desc=f"Sweep {sweep}", unit="pt"):
            results.append(job(iv))
    else:
        try:
            mp.set_start_method(args.mp_start, force=False)
        except RuntimeError:
            pass
        ctx = mp.get_context(args.mp_start)
        with ctx.Pool(processes=args.workers, initializer=_worker_init, initargs=(args.torch_threads,)) as pool:
            for out in tqdm(pool.imap_unordered(job, idx_values), total=len(idx_values), desc=f"Sweep {sweep}", unit="pt"):
                results.append(out)
        # keep user-given order
        results.sort(key=lambda d: int(d.get("__idx", 0)))

    # drop internal idx
    for r in results:
        r.pop("__idx", None)

    payload: Dict[str, Any] = {
        "meta": {
            "sweep": {"param": sweep, "values": values, "num_points": len(values)},
            "base": {"n": args.n, "k": args.k, "keep": args.keep, "s": args.s, "b": args.b, "mux": args.mux, "conflict": args.c},
            "flags": {"fix_nm": bool(args.fix_nm), "nm_round": args.nm_round},
            "random": {"seed": args.seed, "dist": args.dist},
            "fairness": {
                "protocol": "equal_final_nnz_to_column_combine_lossy" if sweep in ("c", "conflict") else "none",
                "note": "General/Eureka/OPF are evaluated on a matrix re-pruned to match CC-lossy final NNZ (Protocol-1)." if sweep in ("c", "conflict") else "No NNZ matching; all methods evaluated on the same pruned input.",
            },
            "pad_value": PAD_VALUE,
        },
        "points": results,
    }

    # filename
    if args.out:
        filename = args.out
    else:
        def _fmt_list(vs: List[Union[int, float]]) -> str:
            # 너무 길면 앞 3개만
            if len(vs) <= 4:
                s = "_".join(str(x).replace(".", "p").replace("-", "m") for x in vs)
            else:
                head = "_".join(str(x).replace(".", "p").replace("-", "m") for x in vs[:3])
                s = f"{head}_etc{len(vs)}"
            return s

        base_name = f"sweep_{sweep}"
        filename = base_name + ".json"
        i = 0
        while os.path.exists("./datas/"+filename):
            filename = f"{base_name}_v{i}.json"
            i += 1

    dump_kw = {"ensure_ascii": False, "separators": (",", ":")} if args.compact_json else {"ensure_ascii": False, "indent": 2}
    with open("./datas/"+filename, "w", encoding="utf-8") as f:
        json.dump(payload, f, **dump_kw)

    print("Saved:", filename)


if __name__ == "__main__":
    main()

