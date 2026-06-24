#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

METHODS = [
    ("0", "general"),
    ("1", "colcomb"),
    ("2", "eureka"),
    ("3", "ctp"),
]

CYCLE_KEYS = ("total_cycles", "sa_cycles_total", "sa_cycles", "cycles")
MS_KEYS = (
    # Preferred keys for SA+VP+CPU-fallback E2E.
    "nongemm_hybrid_ms_for_e2e",
    "nongemm_total_ms_for_e2e",
    # Historical key name from cpu_profile_shape_vector64_layout_aware_v4.py.
    # In that script this is hybrid: VP-supported ops + CPU fallback ops.
    "nongemm_vector_ms_shape_est",

    # CPU-measured/CPU-estimated fallbacks.
    "nongemm_cpu_ms_est",
    "cpu_nongemm_ms",
    "total_non_gemm_ms",
    "total_cpu_non_gemm_ms",
    "total_cpu_nongemm_ms",
    "non_gemm_ms",
)
SEC_KEYS = (
    "nongemm_cpu_sec_est",
    "cpu_nongemm_sec",
    "total_non_gemm_sec",
    "total_cpu_non_gemm_sec",
    "total_cpu_nongemm_sec",
    "non_gemm_sec",
)
TOTALS_MS_KEYS = (
    "cpu_nongemm_total_ms",
    "cpu_non_gemm_total_ms",
    "total_cpu_nongemm_ms",
    "total_non_gemm_ms",
)
TOTALS_SEC_KEYS = (
    "cpu_nongemm_total_sec",
    "cpu_non_gemm_total_sec",
    "total_cpu_nongemm_sec",
    "total_non_gemm_sec",
)


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _first_float(d: Dict[str, Any], keys: Iterable[str], *, scale: float = 1.0) -> Optional[float]:
    for k in keys:
        if k in d and d[k] is not None:
            try:
                return float(d[k]) * scale
            except (TypeError, ValueError):
                pass
    return None


def _is_gemm_record(rec: Dict[str, Any]) -> bool:
    """Best-effort GEMM flag parser for profiler detail records."""
    for key in ("is_gemm", "gemm", "is_matmul"):
        if key in rec:
            return bool(rec[key])

    name = str(rec.get("op_name") or rec.get("name") or rec.get("type") or rec.get("op_type") or "").lower()
    gemm_words = ("conv", "gemm", "linear", "matmul", "mkl", "mkldnn")
    return any(w in name for w in gemm_words)


def _record_time_ms(rec: Dict[str, Any]) -> Optional[float]:
    # run_non_gemm.py details use total_contribution_ms.
    val = _first_float(rec, ("total_contribution_ms", "mean_time_ms", "cpu_mean_ms", "time_ms"))
    if val is not None:
        return val

    val = _first_float(rec, ("mean_time_sec", "mean_sec", "time_sec"), scale=1e3)
    if val is not None:
        return val

    # Some profiler rows may store microseconds.
    val = _first_float(rec, ("self_cpu_time_total_us", "cpu_time_us"), scale=1e-3)
    if val is not None:
        return val

    return None


def _sum_nongemm_records(records: Any) -> Optional[float]:
    if not isinstance(records, list):
        return None

    total = 0.0
    seen = False
    for rec in records:
        if not isinstance(rec, dict):
            continue
        if _is_gemm_record(rec):
            continue
        t = _record_time_ms(rec)
        if t is None:
            continue
        total += t
        seen = True

    return total if seen else None


def extract_nongemm_ms_used(cpu_json: Any) -> Optional[float]:
    """Return CPU/VPU non-GEMM total in ms from known JSON schemas."""
    if not isinstance(cpu_json, dict):
        # A bare list of layer/detail records.
        return _sum_nongemm_records(cpu_json)

    # 1) Old look_result schema: datas/cpu_<model>.json
    val = _first_float(cpu_json, MS_KEYS)
    if val is not None:
        return val
    val = _first_float(cpu_json, SEC_KEYS, scale=1e3)
    if val is not None:
        return val

    # 2) run_non_gemm.py schema: {"summary": {"total_non_gemm_ms": ...}}
    summary = cpu_json.get("summary")
    if isinstance(summary, dict):
        val = _first_float(summary, MS_KEYS)
        if val is not None:
            return val
        val = _first_float(summary, SEC_KEYS, scale=1e3)
        if val is not None:
            return val

    # 3) run_total.py merged schema: {"totals": {"cpu_nongemm_total_ms": ...}}
    totals = cpu_json.get("totals")
    if isinstance(totals, dict):
        val = _first_float(totals, TOTALS_MS_KEYS)
        if val is not None:
            return val
        val = _first_float(totals, TOTALS_SEC_KEYS, scale=1e3)
        if val is not None:
            return val

    # 4) Fallback: sum non-GEMM rows from details/layers.
    for key in ("details", "layers"):
        val = _sum_nongemm_records(cpu_json.get(key))
        if val is not None:
            return val

    return None


def candidate_cpu_paths(cpu_dir: str, model: str) -> List[str]:
    """Try normalized, legacy run_non_gemm, and run_total-style file names."""
    return [
        os.path.join(cpu_dir, f"cpu_{model}.json"),
        os.path.join(cpu_dir, f"{model}_npu_ready.json"),
        f"{model}_npu_ready.json",
        os.path.join(cpu_dir, f"final_{model}_total_m0.json"),
        os.path.join(cpu_dir, "result", f"final_{model}_total_m0.json"),
    ]


def load_nongemm_ms_used(cpu_dir: str, model: str) -> Tuple[Optional[float], Optional[str]]:
    for path in candidate_cpu_paths(cpu_dir, model):
        if not os.path.exists(path):
            continue
        try:
            cpu = _load_json(path)
            val = extract_nongemm_ms_used(cpu)
        except Exception as e:
            print(f"[WARN] CPU json could not be parsed: {path} ({e})")
            continue
        if val is not None:
            return float(val), path
        print(f"[WARN] CPU json found but non-GEMM ms not recognized: {path}")
    return None, None


def _get_int_from_any_key(d: Dict[str, Any], keys: Iterable[str]) -> int:
    for k in keys:
        if k in d:
            try:
                return int(d[k])
            except (TypeError, ValueError):
                pass
    return 0


def _sum_sa_cycles(sa_json: Any) -> Tuple[int, int]:
    """Return (total_cycles_sum, num_layers)."""
    if not isinstance(sa_json, dict):
        return 0, 0

    # run_gemm.py schema: {"layers": {layer_name: {"total_cycles": ...}}}
    layers = sa_json.get("layers", {})
    if isinstance(layers, dict):
        total = 0
        num = 0
        for v in layers.values():
            if isinstance(v, dict):
                total += _get_int_from_any_key(v, CYCLE_KEYS)
                num += 1
        return total, num

    if isinstance(layers, list):
        total = 0
        num = 0
        for x in layers:
            if isinstance(x, dict):
                total += _get_int_from_any_key(x, CYCLE_KEYS)
                num += 1
        return total, num

    # Optional total-only fallback.
    total = _get_int_from_any_key(sa_json, ("total_cycles", "sa_cycles_total", "sa_cycles", "cycles"))
    return total, 1 if total else 0


@dataclass
class Row:
    model: str
    method: str
    method_idx: int
    sa_layers: int
    sa_cycles: int
    sa_ms: float
    nongemm_ms_used: Optional[float]
    e2e_ms: Optional[float]
    speedup_vs_general: Optional[float]
    nongemm_source: Optional[str]
    sa_source: str


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["resnet50", "mobilenet_v2", "inception_v3", "bert"])
    ap.add_argument("--sa_dir", type=str, default="datas")
    ap.add_argument("--cpu_dir", type=str, default="datas")
    ap.add_argument("--hz", type=float, default=500e6, help="SA clock (Hz), default 500e6")
    ap.add_argument("--out_csv", type=str, default=None, help="optional CSV path")
    ap.add_argument("--strict_cpu", action="store_true", help="raise an error if CPU non-GEMM JSON is missing/unrecognized")
    args = ap.parse_args()

    rows: List[Row] = []

    for model in args.models:
        nongemm_ms_used, nongemm_source = load_nongemm_ms_used(args.cpu_dir, model)
        if nongemm_ms_used is None:
            msg = f"CPU non-GEMM json not found or not recognized for model={model}; tried {candidate_cpu_paths(args.cpu_dir, model)}"
            if args.strict_cpu:
                raise FileNotFoundError(msg)
            print(f"[WARN] {msg} (E2E_ms will be N/A)")

        general_e2e: Optional[float] = None

        for idx, (tag, mname) in enumerate(METHODS):
            sa_path = os.path.join(args.sa_dir, f"sa_result_{model}_{mname}.json")
            if not os.path.exists(sa_path):
                print(f"[WARN] SA json not found: {sa_path} (skip)")
                continue

            sa = _load_json(sa_path)
            sa_cycles, sa_layers = _sum_sa_cycles(sa)
            sa_ms = (sa_cycles / float(args.hz)) * 1e3

            e2e_ms = None if nongemm_ms_used is None else nongemm_ms_used + sa_ms
            if idx == 0:
                general_e2e = e2e_ms

            speedup = None
            if e2e_ms is not None and general_e2e is not None and general_e2e > 0:
                speedup = general_e2e / e2e_ms

            rows.append(Row(
                model=model,
                method=mname,
                method_idx=idx,
                sa_layers=sa_layers,
                sa_cycles=sa_cycles,
                sa_ms=sa_ms,
                nongemm_ms_used=nongemm_ms_used,
                e2e_ms=e2e_ms,
                speedup_vs_general=speedup,
                nongemm_source=nongemm_source,
                sa_source=sa_path,
            ))

        model_rows = [r for r in rows if r.model == model]
        if not model_rows:
            continue

        print("\n" + "=" * 100)
        print(f"Model: {model}")
        print(f"Hybrid non-GEMM ms used for E2E: {nongemm_ms_used if nongemm_ms_used is not None else 'N/A'}")
        print(f"Non-GEMM source: {nongemm_source if nongemm_source else 'N/A'}")
        print(f"SA clock: {args.hz:g} Hz")
        print("-" * 100)
        header = f"{'Method':<14} {'SA_layers':>8} {'SA_cycles':>14} {'SA_ms':>12} {'E2E_ms':>12} {'Speedup':>9}"
        print(header)
        print("-" * 100)
        for r in sorted(model_rows, key=lambda x: x.method_idx):
            e2e = f"{r.e2e_ms:.6f}" if r.e2e_ms is not None else "N/A"
            spd = f"{r.speedup_vs_general:.3f}x" if r.speedup_vs_general is not None else "N/A"
            print(f"{r.method:<14} {r.sa_layers:>8d} {r.sa_cycles:>14d} {r.sa_ms:>12.6f} {e2e:>12} {spd:>9}")
        print("=" * 100)

    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "model", "method", "method_idx", "sa_layers", "sa_cycles", "sa_ms",
                "nongemm_ms_used", "e2e_ms", "speedup_vs_general", "nongemm_source", "sa_source",
            ])
            for r in rows:
                w.writerow([
                    r.model,
                    r.method,
                    r.method_idx,
                    r.sa_layers,
                    r.sa_cycles,
                    f"{r.sa_ms:.9f}",
                    "" if r.nongemm_ms_used is None else f"{r.nongemm_ms_used:.9f}",
                    "" if r.e2e_ms is None else f"{r.e2e_ms:.9f}",
                    "" if r.speedup_vs_general is None else f"{r.speedup_vs_general:.6f}",
                    r.nongemm_source or "",
                    r.sa_source,
                ])
        print(f"[OK] wrote CSV: {args.out_csv}")


if __name__ == "__main__":
    main()
