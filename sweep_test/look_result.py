#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""view_sweep.py

Sweep JSON viewer with fairness (Protocol-1) diagnostics.

Features
- Prints per-method tile_cols/density (and nnz if present) across points.
- Computes Protocol-1 metrics (nnz_base, nnz_cc, extra_drop_ratio) if present.
- Computes column-based speedup proxy: baseline_cols / method_cols.
- Prints a Pareto frontier table for a chosen method using
    x = extra_drop_ratio (lower is better)
    y = speedup proxy (higher is better)

Examples
  python3 view_sweep.py --json sweep_keep_v2.json
  python3 view_sweep.py --json sweep_keep_v2.json --show-shape --show-nnz
  python3 view_sweep.py --json sweep_keep_v2.json --derived
  python3 view_sweep.py --json sweep_keep_v2.json --pareto column_combine

If you want to treat x as a specific parameter (e.g., params.c):
  python3 view_sweep.py --json file.json --x params.c

You can pass multiple json files (they will be concatenated):
  python3 view_sweep.py --json run_c0p5.json run_c1p0.json --x file.base.conflict --sort
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and not (isinstance(x, float) and math.isnan(x))


def _fmt(x: Any, nd: int = 6) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _fmt_int(x: Any) -> str:
    if x is None:
        return "-"
    try:
        return str(int(x))
    except Exception:
        return str(x)


def _get_nested(d: Dict[str, Any], path: str) -> Any:
    """Get nested field by dotted path. Returns None if missing."""
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def load_payload(path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict) and "points" in payload and isinstance(payload["points"], list):
        meta = payload.get("meta", {}) if isinstance(payload.get("meta", {}), dict) else {}
        points = payload["points"]
        return meta, points

    # single summary dict
    return {}, [{"sweep_value": None, "result": payload}]


def discover_methods(points: List[Dict[str, Any]]) -> List[str]:
    for p in points:
        r = p.get("result", {})
        if isinstance(r, dict):
            preferred = ["general", "column_combine", "eureka", "opf", "ctc"]
            found = [m for m in preferred if m in r]
            others = [m for m in r.keys() if m not in found]
            return found + sorted(others)
    return []


def get_x_value(point: Dict[str, Any], x_spec: str, meta: Optional[Dict[str, Any]] = None, fallback_idx: Optional[int] = None) -> Any:
    """x_spec: 'sweep_value' or 'params.s' or 'protocol1.extra_drop_ratio' etc."""
    if x_spec == "sweep_value":
        x = point.get("sweep_value", None)
        return x if x is not None else fallback_idx

    if x_spec.startswith("meta."):
        if meta is None:
            return None
        return _get_nested(meta, x_spec[len("meta."):])

    if x_spec.startswith("file.base."):
        base = point.get("_file_base", {})
        return _get_nested(base, x_spec[len("file.base."):]) if isinstance(base, dict) else None

    if x_spec.startswith("file.sweep."):
        sw = point.get("_file_sweep", {})
        return _get_nested(sw, x_spec[len("file.sweep."):]) if isinstance(sw, dict) else None

    if x_spec.startswith("point."):
        return _get_nested(point, x_spec[len("point."):])

    # shorthand: params.xxx, protocol1.xxx
    if x_spec.startswith("params.") or x_spec.startswith("protocol1.") or x_spec.startswith("raw."):
        return _get_nested(point, x_spec)

    # allow direct key
    return point.get(x_spec, None)


def extract_method(point: Dict[str, Any], method: str) -> Dict[str, Any]:
    r = point.get("result", {})
    if not isinstance(r, dict):
        return {}
    mr = r.get(method, {})
    return mr if isinstance(mr, dict) else {}


def compute_protocol1(point: Dict[str, Any]) -> Dict[str, Optional[float]]:
    p1 = point.get("protocol1", {})
    if not isinstance(p1, dict):
        p1 = {}

    nnz_base = p1.get("nnz_base")
    nnz_cc = p1.get("nnz_cc")

    # Fallbacks if script didn't store protocol1 explicitly
    if nnz_cc is None:
        mr = extract_method(point, "column_combine")
        if "nnz" in mr:
            nnz_cc = mr.get("nnz")

    # nnz_base fallback is hard; best effort
    if nnz_base is None:
        # If stored in protocol1.nnz_matched, use it as base? (not ideal)
        nnz_base = p1.get("nnz_base")

    out: Dict[str, Optional[float]] = {
        "nnz_base": float(nnz_base) if nnz_base is not None else None,
        "nnz_cc": float(nnz_cc) if nnz_cc is not None else None,
        "nnz_drop": None,
        "extra_drop_ratio": None,
    }

    if out["nnz_base"] is not None and out["nnz_cc"] is not None:
        out["nnz_drop"] = out["nnz_base"] - out["nnz_cc"]
        if out["nnz_base"] > 0:
            out["extra_drop_ratio"] = max(0.0, min(1.0, out["nnz_drop"] / out["nnz_base"]))

    return out


def percent_change(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or b == 0:
        return None
    return (a - b) / b * 100.0


def pareto_frontier(points: List[Dict[str, Any]], method: str, meta: Dict[str, Any], x_spec: str) -> List[Dict[str, Any]]:
    """Pareto on (extra_drop_ratio, speedup_proxy)."""
    rows: List[Dict[str, Any]] = []
    for i, p in enumerate(points):
        p1 = compute_protocol1(p)
        x = get_x_value(p, x_spec, meta=meta, fallback_idx=i)
        extra = p1.get("extra_drop_ratio")
        # speedup proxy via cols ratio
        gen = extract_method(p, "general")
        base_cols = gen.get("tile_cols")
        mr = extract_method(p, method)
        cols = mr.get("tile_cols")
        if base_cols is None or cols is None:
            continue
        try:
            base_cols_f = float(base_cols)
            cols_f = float(cols)
        except Exception:
            continue
        if cols_f <= 0:
            continue
        speedup = base_cols_f / cols_f
        rows.append({
            "x": x,
            "extra_drop_ratio": extra,
            "speedup": speedup,
            "tile_cols": int(cols_f),
            "base_cols": int(base_cols_f),
        })

    # keep only rows with extra defined
    rows = [r for r in rows if r["extra_drop_ratio"] is not None]
    rows.sort(key=lambda r: (float(r["extra_drop_ratio"]), -float(r["speedup"])))

    frontier: List[Dict[str, Any]] = []
    best_y = -1.0
    for r in rows:
        y = float(r["speedup"])
        if y > best_y + 1e-12:
            frontier.append(r)
            best_y = y
    return frontier


def main():
    ap = argparse.ArgumentParser(description="View sweep JSON; print tile_cols/density and Protocol-1 diagnostics.")
    ap.add_argument("--json", nargs="+", required=True, help="one or more sweep JSON files")
    ap.add_argument("--methods", default="", help="comma-separated methods (default: auto)")
    ap.add_argument("--metrics", default="tile_cols,density", help="comma-separated metrics: tile_cols,density")
    ap.add_argument("--delta", default="prev", choices=["none", "prev", "first"], help="delta baseline")
    ap.add_argument("--sort", action="store_true", help="sort points by x (numeric) if possible")
    ap.add_argument("--x", default="sweep_value", help="x-axis field: sweep_value (default), params.s, protocol1.extra_drop_ratio, ...")
    ap.add_argument("--show-shape", action="store_true")
    ap.add_argument("--show-nnz", action="store_true", help="show method nnz if present in JSON")
    ap.add_argument("--derived", action="store_true", help="print derived Protocol-1 and compression/speedup proxy table")
    ap.add_argument("--pareto", default="", help="print Pareto frontier for method (e.g., column_combine)")

    args = ap.parse_args()

    all_points: List[Dict[str, Any]] = []
    metas: List[Tuple[str, Dict[str, Any]]] = []
    
    for path in args.json:
        path_ = "./datas/"+path
        meta, pts = load_payload(path_)
        metas.append((path_, meta))
        base = meta.get("base", {}) if isinstance(meta.get("base", {}), dict) else {}
        sweep = meta.get("sweep", {}) if isinstance(meta.get("sweep", {}), dict) else {}
        for p in pts:
            if isinstance(p, dict):
                p = dict(p)  # shallow copy
                p["_src"] = os.path.basename(path_)
                # stash per-file meta fields onto each point for multi-file aggregation
                p["_file_base"] = base
                p["_file_sweep"] = sweep
                all_points.append(p)

    # choose a meta (first) for display; keep per-point _src
    meta0 = metas[0][1] if metas else {}

    # build x values & optionally sort
    enriched: List[Tuple[Any, int, Dict[str, Any]]] = []
    for i, p in enumerate(all_points):
        # meta can differ per file; but x_spec meta.* refers to meta0 only
        x = get_x_value(p, args.x, meta=meta0, fallback_idx=i)
        enriched.append((x, i, p))

    if args.sort:
        numeric = [t for t in enriched if _is_number(t[0])]
        nonnum = [t for t in enriched if not _is_number(t[0])]
        numeric.sort(key=lambda t: float(t[0]))
        enriched = numeric + nonnum

    points = [p for _, _, p in enriched]

    # methods
    auto_methods = discover_methods(points)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()] if args.methods.strip() else auto_methods

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    want_cols = "tile_cols" in metrics
    want_den = "density" in metrics

    # header
    if len(args.json) == 1:
        print(f"# File: {args.json[0]}")
    else:
        print(f"# Files: {', '.join(args.json)}")
    if meta0:
        try:
            print("# Meta:", json.dumps(meta0, ensure_ascii=False))
        except Exception:
            pass
    print()

    # derived table
    if args.derived:
        print("== derived (Protocol-1 / CC diagnostics) ==")
        hdr = ["x", "src", "nnz_base", "nnz_cc", "extra_drop", "gen_cols", "cc_cols", "cc_speedup"]
        print(" | ".join(hdr))
        print("-" * (len(" | ".join(hdr)) + 10))

        for i, p in enumerate(points):
            x = get_x_value(p, args.x, meta=meta0, fallback_idx=i)
            src = p.get("_src", "-")
            p1 = compute_protocol1(p)
            gen = extract_method(p, "general")
            cc = extract_method(p, "column_combine")
            gen_cols = gen.get("tile_cols")
            cc_cols = cc.get("tile_cols")

            cc_speedup = None
            if gen_cols is not None and cc_cols is not None:
                try:
                    cc_speedup = float(gen_cols) / float(cc_cols) if float(cc_cols) > 0 else None
                except Exception:
                    cc_speedup = None

            row = [
                _fmt(x, 6),
                str(src),
                _fmt_int(p1.get("nnz_base")),
                _fmt_int(p1.get("nnz_cc")),
                (_fmt(p1.get("extra_drop_ratio"), 6) if p1.get("extra_drop_ratio") is not None else "-"),
                _fmt_int(gen_cols),
                _fmt_int(cc_cols),
                (_fmt(cc_speedup, 6) if cc_speedup is not None else "-"),
            ]
            print(" | ".join(row))
        print()

    # per-method tables
    for method in methods:
        # build series
        series: List[Tuple[Any, Any, Optional[int], Optional[float], Optional[int], str]] = []
        for i, p in enumerate(points):
            x = get_x_value(p, args.x, meta=meta0, fallback_idx=i)
            shape = p.get("shape", None)
            src = p.get("_src", "-")
            mr = extract_method(p, method)
            cols = mr.get("tile_cols")
            den = mr.get("density")
            nnz = mr.get("nnz")

            cols_i = int(cols) if cols is not None else None
            den_f = float(den) if den is not None else None
            nnz_i = int(nnz) if nnz is not None else None
            series.append((x, shape, cols_i, den_f, nnz_i, str(src)))

        if not any((c is not None or d is not None) for _, _, c, d, _, _ in series):
            print(f"== {method} == (no data)\n")
            continue

        # baselines
        if args.delta == "first":
            base_cols = series[0][2]
            base_den = series[0][3]
        else:
            base_cols = None
            base_den = None

        header_cols = ["x"]
        if len(args.json) > 1:
            header_cols.append("src")
        if args.show_shape:
            header_cols.append("shape")

        if want_cols:
            header_cols += ["tile_cols", "Δcols", "%Δcols"]
        if want_den:
            header_cols += ["density", "Δden", "%Δden"]
        if args.show_nnz:
            header_cols += ["nnz"]

        print(f"== {method} ==")
        print(" | ".join(header_cols))
        print("-" * (len(" | ".join(header_cols)) + 10))

        prev_cols = None
        prev_den = None

        for (x, shape, cols_i, den_f, nnz_i, src) in series:
            if args.delta == "prev":
                b_cols = prev_cols
                b_den = prev_den
            elif args.delta == "first":
                b_cols = base_cols
                b_den = base_den
            else:
                b_cols = None
                b_den = None

            row: List[str] = []
            row.append(_fmt(x, 6))
            if len(args.json) > 1:
                row.append(src)
            if args.show_shape:
                row.append(str(shape) if shape is not None else "-")

            if want_cols:
                dcols = (cols_i - b_cols) if (cols_i is not None and b_cols is not None) else None
                pcols = percent_change(float(cols_i), float(b_cols)) if (cols_i is not None and b_cols is not None) else None
                row += [_fmt_int(cols_i), _fmt_int(dcols), (_fmt(pcols, 3) + "%") if pcols is not None else "-"]

            if want_den:
                dden = (den_f - b_den) if (den_f is not None and b_den is not None) else None
                pden = percent_change(den_f, b_den)
                row += [_fmt(den_f, 6), _fmt(dden, 6), (_fmt(pden, 3) + "%") if pden is not None else "-"]

            if args.show_nnz:
                row.append(_fmt_int(nnz_i))

            print(" | ".join(row))

            prev_cols = cols_i
            prev_den = den_f

        cols_vals = [c for _, _, c, _, _, _ in series if c is not None]
        den_vals = [d for _, _, _, d, _, _ in series if d is not None]
        if cols_vals:
            print(f"  [tile_cols] min={min(cols_vals)} max={max(cols_vals)}")
        if den_vals:
            print(f"  [density]  min={min(den_vals):.6f} max={max(den_vals):.6f}")
        print()

    # pareto
    if args.pareto:
        method = args.pareto.strip()
        print(f"== pareto frontier ({method}) ==")
        front = pareto_frontier(points, method, meta0, args.x)
        hdr = ["x", "extra_drop", "speedup(cols)", "base_cols", "tile_cols"]
        print(" | ".join(hdr))
        print("-" * (len(" | ".join(hdr)) + 10))
        for r in front:
            print(
                " | ".join(
                    [
                        _fmt(r.get("x"), 6),
                        _fmt(r.get("extra_drop_ratio"), 6),
                        _fmt(r.get("speedup"), 6),
                        _fmt_int(r.get("base_cols")),
                        _fmt_int(r.get("tile_cols")),
                    ]
                )
            )
        if not front:
            print("(no pareto points; need protocol1.nnz_base/nnz_cc in JSON)")
        print()


if __name__ == "__main__":
    main()
