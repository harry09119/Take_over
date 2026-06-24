#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import multiprocessing as mp
from functools import partial
from typing import Any, Dict, List, Tuple, Optional

import torch
from tqdm import tqdm

import lib_packing as pk
from torchvision import models
from torchvision.models import MobileNet_V2_Weights, VGG16_Weights, Inception_V3_Weights

try:
    from transformers import BertModel
except ImportError:
    BertModel = None

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
# Model loading
# -----------------------------------------------------------------------------

def load_model_by_name(model_name: str, device: torch.device, pretrained: bool = True) -> torch.nn.Module:
    name = model_name.lower()

    if name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        model.to(device)
        return model

    if name in ("mobilenet_v2", "mobilenet"):
        model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
        model.to(device)
        return model

    if name == "vgg16":
        model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
        model.to(device)
        return model

    if name in ("inception_v3", "inception"):
        model = models.inception_v3(
            weights=Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None,
            aux_logits=True,
        )
        model.to(device)
        return model

    if name in ("bert", "bert-base", "bert-base-uncased"):
        if BertModel is None:
            raise ImportError("transformers 패키지가 필요합니다. `pip install transformers` 후 다시 시도하세요.")
        model = BertModel.from_pretrained("bert-base-uncased")
        model.to(device)
        return model

    raise ValueError(f"지원하지 않는 model_name 입니다: {model_name}")


# -----------------------------------------------------------------------------
# Pruning & GEMM(weight) matrix extraction
# -----------------------------------------------------------------------------

def magnitude_prune_mask(w: torch.Tensor, prune_ratio: float) -> torch.Tensor:
    """
    layer-wise unstructured magnitude pruning mask.
    keep (1-prune_ratio) fraction by |w|.

    Returns: bool mask, True means "keep".
    """
    if prune_ratio <= 0.0:
        return torch.ones_like(w, dtype=torch.bool)
    if prune_ratio >= 1.0:
        return torch.zeros_like(w, dtype=torch.bool)

    flat = w.abs().flatten()
    numel = flat.numel()
    k_keep = int(round((prune_ratio) * numel))
    if k_keep <= 0:
        return torch.zeros_like(w, dtype=torch.bool)
    if k_keep >= numel:
        return torch.ones_like(w, dtype=torch.bool)

    # kth smallest index for the keep-threshold (keep top-k by magnitude)
    # threshold = kth largest = (numel - k_keep + 1)-th smallest
    kth = numel - k_keep + 1
    thr = flat.kthvalue(kth).values  # scalar
    return (w.abs() >= thr)


def conv2d_weight_to_gemm(module: torch.nn.Conv2d) -> Tuple[torch.Tensor, Dict[str, Any]]:
    w = module.weight.detach().cpu()
    oc, icpg, kh, kw = w.shape
    gm = icpg * kh * kw
    w2d = w.reshape(oc, gm)
    meta = {
        "op": "conv2d",
        "out_channels": oc,
        "in_channels_per_group": icpg,
        "kernel": [kh, kw],
        "groups": int(getattr(module, "groups", 1)),
        "gemm_shape": [int(oc), int(gm)],  # [N, K]
    }
    return w2d, meta


def linear_weight_to_gemm(module: torch.nn.Linear) -> Tuple[torch.Tensor, Dict[str, Any]]:
    w = module.weight.detach().cpu()
    oc, ic = w.shape
    meta = {
        "op": "linear",
        "out_features": int(oc),
        "in_features": int(ic),
        "gemm_shape": [int(oc), int(ic)],  # [N, K]
    }
    return w, meta


def should_pack_bert_qkv(layer_name: str) -> bool:
    # transformers BertModel named_modules() 기준
    # e.g., "encoder.layer.0.attention.self.query"
    return (
        ".attention.self.query" in layer_name
        or ".attention.self.key" in layer_name
        or ".attention.self.value" in layer_name
    )


def extract_gemm_layers_and_infos(
    model: torch.nn.Module,
    model_name: str,
    prune_ratio: float,
    pad_value: int,
    *,
    b_tile: int,
    bert_pack_only_qkv: bool = True,
    disable_packing_for_group_conv: bool = True,
) -> Tuple[List[List[torch.Tensor]], List[Dict[str, Any]]]:
    """
    Returns:
      layers: list of [matrix_tensor] where matrix_tensor is int32 with pad_value for zeros
      infos : list of meta dicts (must include "name")
    """
    layers: List[List[torch.Tensor]] = []
    infos: List[Dict[str, Any]] = []

    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Conv2d):
            w2d, meta = conv2d_weight_to_gemm(mod)
        elif isinstance(mod, torch.nn.Linear):
            w2d, meta = linear_weight_to_gemm(mod)
        else:
            continue

        # BERT: packing 적용 범위 제어
        pack_enable = True
        if model_name.lower().startswith("bert") and bert_pack_only_qkv:
            pack_enable = should_pack_bert_qkv(name)

        # group/depthwise conv는 packing 비활성화 권장 (column combine/opf가 그룹 간 컬럼을 섞으면 의미가 깨질 수 있음)
        groups = int(meta.get("groups", 1))
        if disable_packing_for_group_conv and groups != 1:
            pack_enable = False
            meta["pack_disable_reason"] = f"group_conv(groups={groups})"

        keep = magnitude_prune_mask(w2d, prune_ratio)
        # packing 알고리즘은 값 자체보다 "pad_value 여부"만 쓰는 경우가 많으므로, nonzero는 1로 통일
        mat = torch.where(keep, torch.ones_like(w2d, dtype=torch.int32), torch.full_like(w2d, pad_value, dtype=torch.int32))

        meta.update({
            "name": name,
            "prune_ratio": float(prune_ratio),
            "pad_value": int(pad_value),
            "b_tile": int(b_tile),                 # main에서 말한 "b" 기록 (ifmap_cols_per_slice로 쓰기 좋음)
            "ifmap_cols_per_slice": int(min(b_tile, meta["gemm_shape"][1])),  # 기본: b(마지막 슬라이스 짧으면 K로 clamp)
            "pack_enable": bool(pack_enable),
        })

        layers.append([mat])
        infos.append(meta)

    return layers, infos


# -----------------------------------------------------------------------------
# Packing per layer
# -----------------------------------------------------------------------------

def _pack_one_layer(
    layer_and_info: Tuple[List[torch.Tensor], Dict[str, Any]],
    preset: Dict[str, Any],
    run_eureka: bool,
    run_opf: bool,
    verify_nnz: bool,
) -> Tuple[Dict[str, Any], List[List[int]], List[List[int]], List[List[int]], List[List[int]], List[int]]:
    s = preset["s"]
    b = preset["b"]
    conflict = preset["c"]
    mux_size = preset["m"]

    layer, info = layer_and_info
    meta = dict(info)
    compare = [0, 0, 0, 0]

    pack_enable = bool(meta.get("pack_enable", True))

    total_t = layer[0]

    # General (항상 수행)
    general_t = pk.remove_empty(total_t)
    origin_nzs = fast_nnz(general_t)
    gn, gm = general_t.shape
    gn_tiles = (gn + s - 1) // s
    gm_tiles = (gm + b - 1) // b

    general_tiles: List[List[int]] = []
    for mi in range(gm_tiles):
        col_start = mi * b
        col_end = min((mi + 1) * b, gm)
        for ni in range(gn_tiles):
            now_start = ni * s
            now_end = min((ni + 1) * s, gn)
            now_t = general_t[now_start:now_end, col_start:col_end]
            _, tm = now_t.shape
            general_tiles.append([s, tm])
            compare[0] += tm

    # packing 비활성화 레이어(BERT non-QKV, group conv 등): 모든 방식이 general과 동일
    if not pack_enable:
        meta.update(
            {
                "general_tiles": general_tiles,
                "column_combine_tiles": list(general_tiles),
                "eureka_tiles": list(general_tiles),
                "opf_tiles": list(general_tiles),
                "compare": [compare[0], compare[0], compare[0], compare[0]],
                "sa_row_tiles": gn_tiles,
                "colcom_row_tiles": gn_tiles,
                "eureka_row_tiles": gn_tiles,
                "opf_row_tiles": gn_tiles,
            }
        )
        return meta, meta["general_tiles"], meta["column_combine_tiles"], meta["eureka_tiles"], meta["opf_tiles"], meta["compare"]

    # Column combine (b-slice별로 수행)
    colcomb_tiles: List[List[int]] = []
    colcom_nzs = 0
    for mi in range(gm_tiles):
        col_start = mi * b
        col_end = min((mi + 1) * b, gm)
        cols = general_t[:, col_start:col_end]
        colcom_t, _, _ = pk.column_combine(cols, gn * conflict, mux_size)
        colcom_nzs += fast_nnz(colcom_t)

        for ni in range(gn_tiles):
            now_start = ni * s
            now_end = min((ni + 1) * s, gn)
            now_t = colcom_t[now_start:now_end]
            _, tm = now_t.shape
            colcomb_tiles.append([s, tm])
            compare[1] += tm

    """
    # re-prune (기존 코드 유지)
    pruned_nzs = origin_nzs - colcom_nzs
    general_pt = pk.re_prune(general_t.to(torch.int32), pruned_nzs)
    general_pt = pk.remove_empty(general_pt)
    gn, gm = general_pt.shape
    total_nzs = fast_nnz(general_pt) if verify_nnz else None
    """

    general_pt = general_t

    # Eureka
    eureka_tiles: List[List[int]] = []
    eureka_nzs = 0
    if run_eureka:
        mux_size_ = mux_size * 2
        en_tiles = (gn + s - 1) // s
        em_tiles = (gm + b - 1) // b
        et_tiles = (b + mux_size_ - 1) // b

        for mi in range(em_tiles):
            col_start = mi * b
            col_end = min((mi + 1) * b, gm)
            for ni in range(en_tiles):
                row_start = ni * s
                row_end = min((ni + 1) * s, gn)
                tile = general_pt[row_start:row_end, col_start:col_end]

                tile_len = 0
                for ti in range(et_tiles):
                    tile_start = ti * mux_size_
                    tile_end = min((ti + 1) * mux_size_, b)
                    inner_tile = tile[:, tile_start:tile_end]
                    eureka_t, _ = pk.eureka_optimal(inner_tile)
                    if verify_nnz:
                        eureka_nzs += fast_nnz(eureka_t)
                    etn, etm = eureka_t.shape
                    if etn > 0 and etm > 0:
                        tile_len += etm

                eureka_tiles.append([s, tile_len])
                compare[2] += tile_len
    else:
        en_tiles = gn_tiles

    # Cross Tile Pre-execution
    opf_tiles: List[List[int]] = []
    opf_nzs = 0
    if run_opf:
        total_t_ = pk.reorder_tensor(general_pt, "a")
        on_tiles = (gn + s - 1) // s
        om_tiles = (gm + b - 1) // b
        for mi in range(om_tiles):
            col_start = mi * b
            col_end = min((mi + 1) * b, gm)
            col_tiles = total_t_[:, col_start:col_end]
            diff = on_tiles * s - gn
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
                    now_pt, pruned_t = pk.opf(now_pt, now_g, next_t, mux_size, s, ope=True)
                    col_tiles[next_start:next_end].copy_(pruned_t)

                if verify_nnz:
                    opf_nzs += fast_nnz(now_pt)

                on, om = now_pt.shape
                if on > 0 and om > 0:
                    opf_tiles.append([s, om])
                    compare[3] += om
    else:
        on_tiles = gn_tiles

    if verify_nnz and total_nzs is not None:
        meta["verify"] = {
            "total_nzs": total_nzs,
            "eureka_nzs": eureka_nzs if run_eureka else None,
            "opf_nzs": opf_nzs if run_opf else None,
        }

    meta.update(
        {
            "general_tiles": general_tiles,
            "column_combine_tiles": colcomb_tiles,
            "eureka_tiles": eureka_tiles,
            "opf_tiles": opf_tiles,
            "compare": compare,
            "sa_row_tiles": gn_tiles,
            "colcom_row_tiles": gn_tiles,
            "eureka_row_tiles": en_tiles,
            "opf_row_tiles": on_tiles,
        }
    )

    return meta, general_tiles, colcomb_tiles, eureka_tiles, opf_tiles, compare


# -----------------------------------------------------------------------------
# BERT attention pseudo GEMMs: QK^T and SV
# -----------------------------------------------------------------------------

def _dense_tiles_for_weight_shape(N: int, K: int, s: int, b: int) -> Tuple[List[List[int]], int]:
    """
    weight shape [N, K] (rows=N, cols=K).
    return tiles list of [s, tm], and row_tiles count.
    """
    gn, gm = int(N), int(K)
    gn_tiles = (gn + s - 1) // s
    gm_tiles = (gm + b - 1) // b
    tiles: List[List[int]] = []
    for mi in range(gm_tiles):
        col_start = mi * b
        col_end = min((mi + 1) * b, gm)
        tm = col_end - col_start
        for _ in range(gn_tiles):
            tiles.append([s, tm])
    return tiles, gn_tiles


def append_bert_attention_pseudo_ops(
    total_layers: List[Dict[str, Any]],
    *,
    model: Any,
    seq_len: int,
    batch: int,
    preset: Dict[str, Any],
):
    """
    BERT self-attention에서 QK^T 및 (softmax(QK))V 를 pseudo-layer로 추가.
    - packing은 적용하지 않으므로 모든 tiles를 general과 동일하게 설정.
    - sa_profile에서 shape collector로 m_tiles를 못 구할 수 있으므로 m_tiles_override를 meta에 포함.
    """
    s = int(preset["s"])
    b = int(preset["b"])
    S_col = int(preset.get("S_col", 64))  # SA column dimension (m_tiles 계산용)

    cfg = getattr(model, "config", None)
    if cfg is None:
        return

    num_layers = int(getattr(cfg, "num_hidden_layers", 12))
    num_heads = int(getattr(cfg, "num_attention_heads", 12))
    hidden = int(getattr(cfg, "hidden_size", 768))
    head_dim = hidden // num_heads

    # M dimension (output positions) = batch * heads * seq
    M_total = batch * num_heads * seq_len
    m_tiles_override = (M_total + S_col - 1) // S_col

    for i in range(num_layers):
        # QK: weight = K^T  => [N=seq, K=head_dim]
        tiles_qk, row_tiles_qk = _dense_tiles_for_weight_shape(seq_len, head_dim, s, b)
        comp_qk = sum(t[1] for t in tiles_qk)
        meta_qk = {
            "name": f"encoder.layer.{i}.attention.qk_matmul",
            "op": "attn_qk",
            "gemm_shape": [int(seq_len), int(head_dim)],
            "b_tile": int(b),
            "ifmap_cols_per_slice": int(min(b, head_dim)),  # K slice length
            "m_tiles_override": int(m_tiles_override),
            "general_tiles": tiles_qk,
            "column_combine_tiles": list(tiles_qk),
            "eureka_tiles": list(tiles_qk),
            "opf_tiles": list(tiles_qk),
            "compare": [comp_qk, comp_qk, comp_qk, comp_qk],
            "sa_row_tiles": row_tiles_qk,
            "colcom_row_tiles": row_tiles_qk,
            "eureka_row_tiles": row_tiles_qk,
            "opf_row_tiles": row_tiles_qk,
            "pack_enable": False,
            "pack_disable_reason": "activation_matmul(no_packing)",
        }
        total_layers.append(meta_qk)

        # SV: weight = V => [N=head_dim, K=seq]
        tiles_sv, row_tiles_sv = _dense_tiles_for_weight_shape(head_dim, seq_len, s, b)
        comp_sv = sum(t[1] for t in tiles_sv)
        meta_sv = {
            "name": f"encoder.layer.{i}.attention.sv_matmul",
            "op": "attn_sv",
            "gemm_shape": [int(head_dim), int(seq_len)],
            "b_tile": int(b),
            "ifmap_cols_per_slice": int(min(b, seq_len)),
            "m_tiles_override": int(m_tiles_override),
            "general_tiles": tiles_sv,
            "column_combine_tiles": list(tiles_sv),
            "eureka_tiles": list(tiles_sv),
            "opf_tiles": list(tiles_sv),
            "compare": [comp_sv, comp_sv, comp_sv, comp_sv],
            "sa_row_tiles": row_tiles_sv,
            "colcom_row_tiles": row_tiles_sv,
            "eureka_row_tiles": row_tiles_sv,
            "opf_row_tiles": row_tiles_sv,
            "pack_enable": False,
            "pack_disable_reason": "activation_matmul(no_packing)",
        }
        total_layers.append(meta_sv)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Pack DNN GEMM layers (general / column-combine / eureka / opf) - multi-model")

    # 모델
    parser.add_argument("--model", type=str, default="resnet50",
                        choices=["resnet50", "mobilenet_v2", "vgg16", "inception_v3", "bert"])
    parser.add_argument("--no-pretrained", action="store_true")

    # pruning/SA params
    parser.add_argument("--d", type=float, default=0.2, help="unstructured pruning ratio")
    parser.add_argument("--s", type=int, default=64, help="SA rows (tile row size)")
    parser.add_argument("--b", type=int, default=256, help="K tile width (your 'b')")
    parser.add_argument("--m", type=int, default=8, help="mux size")
    parser.add_argument("--c", type=float, default=0.25, help="conflict threshold factor")

    # BERT extra
    parser.add_argument("--bert-seq-len", type=int, default=128)
    parser.add_argument("--bert-batch", type=int, default=1)
    parser.add_argument(
        "--bert-pack-all",
        action="store_true",
        help="BERT에서 Q/K/V뿐 아니라 모든 Linear 레이어에 packing을 적용",
    )
    parser.add_argument("--no-bert-attn-matmul", action="store_true", help="do not append QK/SV pseudo ops")

    # multiprocessing
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--mp-start", type=str,
                        default="fork" if hasattr(mp, "get_start_method") else "spawn",
                        choices=["fork", "spawn", "forkserver"])
    parser.add_argument("--torch-threads", type=int, default=1)

    # toggles
    parser.add_argument("--no-eureka", action="store_true")
    parser.add_argument("--no-opf", action="store_true")
    parser.add_argument("--verify-nnz", action="store_true")
    parser.add_argument("--compact-json", action="store_true")
    parser.add_argument("--no-separate-json", action="store_true", default=True)

    args = parser.parse_args()

    preset = {"d": args.d, "s": args.s, "b": args.b, "m": args.m, "c": args.c, "S_col": args.s}
    model_name = args.model
    pretrained = not args.no_pretrained
    run_eureka = not args.no_eureka
    run_opf = not args.no_opf

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Loading model: {model_name} (pretrained={pretrained}) on {device}")
    model = load_model_by_name(model_name, device=device, pretrained=pretrained)

    # 메인 프로세스도 동일하게 패치
    pk.density_check = density_check_fast  # type: ignore[attr-defined]

    # GEMM(weight) 레이어 추출 + pruning -> PAD_VALUE 행렬 생성
    layers, infos = extract_gemm_layers_and_infos(
        model=model,
        model_name=model_name,
        prune_ratio=float(args.d),
        pad_value=PAD_VALUE,
        b_tile=int(args.b),
        bert_pack_only_qkv=not bool(args.bert_pack_all),
        disable_packing_for_group_conv=True,
    )

    work_items: List[Tuple[List[torch.Tensor], Dict[str, Any]]] = list(zip(layers, infos))

    # 실행
    results: List[Tuple[Dict[str, Any], List[List[int]], List[List[int]], List[List[int]], List[List[int]], List[int]]] = []

    if args.workers <= 1:
        for wi in tqdm(work_items, total=len(work_items), desc="Packing", unit="layer"):
            results.append(_pack_one_layer(wi, preset, run_eureka, run_opf, args.verify_nnz))
    else:
        try:
            mp.set_start_method(args.mp_start, force=False)
        except RuntimeError:
            pass

        ctx = mp.get_context(args.mp_start)
        fn = partial(_pack_one_layer, preset=preset, run_eureka=run_eureka, run_opf=run_opf, verify_nnz=args.verify_nnz)

        with ctx.Pool(processes=args.workers, initializer=_worker_init, initargs=(args.torch_threads,)) as pool:
            for out in tqdm(pool.imap_unordered(fn, work_items), total=len(work_items), desc="Packing", unit="layer"):
                results.append(out)

        # 원래 순서로 정렬
        name_to_idx = {info["name"]: i for i, info in enumerate(infos)}
        results.sort(key=lambda x: name_to_idx.get(x[0].get("name", ""), 10**9))

    total_layers = [r[0] for r in results]
    general_layers = [r[1] for r in results]
    column_combine_layers = [r[2] for r in results]
    eureka_layers = [r[3] for r in results]
    ctp_layers = [r[4] for r in results]

    # BERT: QK/SV pseudo ops 추가
    if model_name.lower().startswith("bert") and not args.no_bert_attn_matmul:
        append_bert_attention_pseudo_ops(
            total_layers,
            model=model,
            seq_len=int(args.bert_seq_len),
            batch=int(args.bert_batch),
            preset=preset,
        )

    # 검증 출력
    if args.verify_nnz:
        for meta in total_layers:
            v = meta.get("verify")
            if not v:
                continue
            if run_eureka and v["eureka_nzs"] is not None and v["total_nzs"] != v["eureka_nzs"]:
                print("[WARN] Eureka nnz mismatch:", meta.get("name"), v)
            if run_opf and v["opf_nzs"] is not None and v["total_nzs"] != v["opf_nzs"]:
                print("[WARN] OPF nnz mismatch:", meta.get("name"), v)

    # 저장
    base_name = f"result_{model_name}_{args.d*100}%_{args.s}SA_{args.m}M"
    ext = ".json"

    filename = base_name + ext
    i = 0
    while os.path.exists(filename):
        filename = f"{base_name}_v{i}{ext}"
        i += 1

    print("저장 파일명:", filename)

    dump_kw = {"ensure_ascii": False, "separators": (",", ":")} if args.compact_json else {"ensure_ascii": False, "indent": 2}
    
    if not args.no_separate_json:
        with open("general_wgt.json", "w", encoding="utf-8") as f:
            json.dump(general_layers, f, **dump_kw)
        with open("column_combine_wgt.json", "w", encoding="utf-8") as f:
            json.dump(column_combine_layers, f, **dump_kw)
        with open("eureka_wgt.json", "w", encoding="utf-8") as f:
            json.dump(eureka_layers, f, **dump_kw)
        with open("ctp_wgt.json", "w", encoding="utf-8") as f:
            json.dump(ctp_layers, f, **dump_kw)
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(total_layers, f, **dump_kw)


if __name__ == "__main__":
    main()
