#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from math import ceil
from typing import Dict, Any, List, Tuple, Optional

import torch
import torch.nn as nn

from lib_sa import simulate_os_sa_full
from torchvision import models
from torchvision.models import MobileNet_V2_Weights, VGG16_Weights, Inception_V3_Weights

try:
    from transformers import BertModel
except ImportError:
    BertModel = None

METHOD_NAMES = ["General", "ColumnCombine", "Eureka", "OPF"]

# method별 tag bits (기존 실험 설정 유지; 필요시 CLI로 바꿔도 됨)
TAG_BITS_ALL = [0, 3, 5, 5]


def load_model_by_name(model_name: str, device: torch.device, pretrained: bool = False) -> torch.nn.Module:
    name = model_name.lower()
    if name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        return model.to(device)
    if name in ("mobilenet_v2", "mobilenet"):
        model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
        return model.to(device)
    if name == "vgg16":
        model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
        return model.to(device)
    if name in ("inception_v3", "inception"):
        model = models.inception_v3(
            weights=Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None,
            aux_logits=True,
        )
        return model.to(device)
    if name in ("bert", "bert-base", "bert-base-uncased"):
        if BertModel is None:
            raise ImportError("transformers 패키지가 필요합니다. `pip install transformers` 후 다시 시도하세요.")
        # BERT는 보통 from_pretrained를 쓰는 게 가장 안전합니다(로컬 캐시 사용 가능).
        # pretrained=False로도 호출될 수 있지만, shape 수집/프로파일링 목적이라면 가중치 유무는 크게 중요하지 않습니다.
        model = BertModel.from_pretrained("bert-base-uncased")
        return model.to(device)
    raise ValueError(f"Unsupported model: {model_name}")


class ShapeCollector:
    """leaf module(conv/linear) output shape 수집."""
    def __init__(self, model: nn.Module):
        self.records: Dict[str, Dict[str, Any]] = {}
        self.handles = []
        for name, module in model.named_modules():
            if len(list(module.children())) > 0:
                continue
            is_gemm = isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear))
            self.records[name] = {"is_gemm": is_gemm, "output_shape": None}
            h = module.register_forward_hook(self._make_hook(name))
            self.handles.append(h)

    def _make_hook(self, name: str):
        def hook(module, inputs, output):
            if self.records[name]["output_shape"] is None:
                if isinstance(output, torch.Tensor):
                    self.records[name]["output_shape"] = tuple(output.shape)
                elif isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
                    self.records[name]["output_shape"] = tuple(output[0].shape)
        return hook

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def estimate_m_tiles_from_output_shape(out_shape: Tuple[int, ...], S_col: int) -> int:
    """
    M_tile = ceil(M / S_col)
    - conv: [B,C,H,W] -> M=B*H*W
    - linear (CNN): [B,C] -> M=B
    - linear (BERT): [B,S,C] -> M=B*S
    """
    if not out_shape:
        return 1
    ndim = len(out_shape)
    if ndim == 4:
        B, C, H, W = out_shape
        M = int(B) * int(H) * int(W)
    elif ndim == 3:
        B, S, C = out_shape
        M = int(B) * int(S)
    elif ndim == 2:
        B, C = out_shape
        M = int(B)
    else:
        M = int(out_shape[0])
    return max(1, ceil(M / int(S_col)))


def load_result_info(path: str) -> Dict[str, Dict[str, Any]]:
    with open(path, "r") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        return {str(k): v for k, v in raw.items() if isinstance(v, dict)}
    if isinstance(raw, list):
        out: Dict[str, Dict[str, Any]] = {}
        for d in raw:
            if isinstance(d, dict) and "name" in d:
                out[str(d["name"])] = d
        return out
    raise TypeError(f"Unexpected JSON type: {type(raw)}")


def target_key(method_idx: int) -> str:
    return ["general_tiles", "column_combine_tiles", "eureka_tiles", "opf_tiles"][method_idx]


ROW_TILES_KEY = {0: "sa_row_tiles", 1: "colcom_row_tiles", 2: "eureka_row_tiles", 3: "opf_row_tiles"}


def extract_tile_lengths(layer_meta: Dict[str, Any], method_idx: int) -> List[int]:
    tiles = layer_meta.get(target_key(method_idx))
    if not tiles:
        return []
    out: List[int] = []
    for t in tiles:
        if isinstance(t, (list, tuple)) and len(t) >= 2:
            out.append(int(t[-1]))
        elif isinstance(t, dict):
            for kk in ("num_cols", "tile_length", "k_len", "k", "cols"):
                if kk in t:
                    out.append(int(t[kk]))
                    break
    return out


def infer_activation_reuse_tiles(layer_meta: Dict[str, Any], method_idx: int) -> int:
    k = ROW_TILES_KEY.get(method_idx, "sa_row_tiles")
    v = layer_meta.get(k, 1)
    try:
        return max(1, int(v))
    except Exception:
        return 1


def infer_ifmap_cols_per_slice(layer_meta: Dict[str, Any], b_tile_default: int) -> int:
    for k in ("ifmap_cols_per_slice", "K_tile", "k_tile", "b_tile", "b", "tile_col_size"):
        if k in layer_meta:
            try:
                return max(1, int(layer_meta[k]))
            except Exception:
                pass
    gs = layer_meta.get("gemm_shape")
    if isinstance(gs, (list, tuple)) and len(gs) >= 2:
        try:
            return max(1, min(int(b_tile_default), int(gs[1])))
        except Exception:
            pass
    return max(1, int(b_tile_default))


def run_sa(base_tile_lengths: List[int], m_tiles: int, *, S_row: int, S_col: int,
           tag_bits: int, activation_reuse_tiles: int, ifmap_cols_per_slice: int, buffer_mode: str,
           dram_bw_words_per_cycle: int, bytes_per_word: int,
           ifmap_buf_bytes: int, ofmap_buf_elems: Optional[int]) -> Tuple[int, Dict[str, float]]:
    if not base_tile_lengths or m_tiles <= 0:
        return 0, {"ofmap_drain_done_time": 0.0}

    tile_lengths = base_tile_lengths * int(m_tiles)

    sim = simulate_os_sa_full(
        S_row=S_row,
        S_col=S_col,
        tile_lengths=tile_lengths,
        dram_bandwidth_words_per_cycle=dram_bw_words_per_cycle,
        bytes_per_word=bytes_per_word,
        ifmap_buffer_capacity_bytes=ifmap_buf_bytes,
        ofmap_buffer_capacity_elems=int(ofmap_buf_elems or 0),
        buffer_mode=buffer_mode,
        timing_mode="full",
        tag_bits_per_entry=int(tag_bits),
        activation_reuse_tiles=int(activation_reuse_tiles),
        repeat_period_tiles=len(base_tile_lengths),
        ifmap_cols_per_slice=int(ifmap_cols_per_slice),
    )
    total_cycles = int(sim.get("ofmap_drain_done_time", 0.0))
    return total_cycles, sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True,
                    choices=["resnet50", "mobilenet_v2", "vgg16", "inception_v3", "bert"])
    ap.add_argument("--result_json", type=str, required=True)
    #ap.add_argument("--output_json", type=str, required=True)
    ap.add_argument("--method_idx", type=int, default=3, choices=[0, 1, 2, 3])
    ap.add_argument("--S_row", type=int, default=64)
    ap.add_argument("--S_col", type=int, default=64)
    ap.add_argument("--b_tile", type=int, default=256)
    ap.add_argument("--pretrained", action="store_true")

    # BERT dummy input
    ap.add_argument("--bert-seq-len", type=int, default=128)
    ap.add_argument("--bert-batch", type=int, default=1)
    ap.add_argument("--batch", type=int, default=32)
    # SA/mem params
    ap.add_argument("--dram_bw_words_per_cycle", type=int, default=512)
    ap.add_argument("--bytes_per_word", type=int, default=1)
    ap.add_argument("--ifmap_buf_bytes", type=int, default=1024 * 512)
    ap.add_argument("--ofmap_buf_elems", type=int, default=1024 * 512)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_by_name(args.model, device=device, pretrained=args.pretrained)

    # shape collect
    collector = ShapeCollector(model)
    model.eval()
    with torch.no_grad():
        if args.model == "bert":
            if BertModel is None:
                raise ImportError("transformers needed for bert profiling.")
            vocab = 30522
            input_ids = torch.randint(0, vocab, (args.bert_batch, args.bert_seq_len), device=device)
            attention_mask = torch.ones((args.bert_batch, args.bert_seq_len), device=device, dtype=torch.long)
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            if args.model == "inception_v3":
                x = torch.randn(args.batch, 3, 299, 299, device=device)
            else:
                x = torch.randn(args.batch, 3, 224, 224, device=device)
            _ = model(x)
    collector.close()

    m_tiles_by_layer: Dict[str, int] = {}
    for lname, rec in collector.records.items():
        if rec.get("is_gemm") and rec.get("output_shape") is not None:
            m_tiles_by_layer[lname] = estimate_m_tiles_from_output_shape(rec["output_shape"], int(args.S_col))

    # load packed result
    result_info = load_result_info(args.result_json)

    method_idx = int(args.method_idx)
    buffer_mode = "quadratic" if method_idx == 3 else "double"
    tag_bits = int(TAG_BITS_ALL[method_idx])

    out: Dict[str, Any] = {
        "model": args.model,
        "method": METHOD_NAMES[method_idx],
        "method_idx": method_idx,
        "S": [int(args.S_row), int(args.S_col)],
        "b_tile_default": int(args.b_tile),
        "layers": {},
    }

    for layer_name, meta in result_info.items():
        if not isinstance(meta, dict):
            continue
        base_tile_lengths = extract_tile_lengths(meta, method_idx)
        if not base_tile_lengths:
            continue

        # pseudo op (BERT QK/SV) 등: override 우선
        m_tiles = int(meta.get("m_tiles_override", m_tiles_by_layer.get(layer_name, 1)))
        activation_reuse_tiles = int(meta.get("activation_reuse_tiles_override", infer_activation_reuse_tiles(meta, method_idx)))
        ifmap_cols_per_slice = int(infer_ifmap_cols_per_slice(meta, b_tile_default=int(args.b_tile)))

        total_cycles, sim = run_sa(
            base_tile_lengths, m_tiles,
            S_row=int(args.S_row), S_col=int(args.S_col),
            tag_bits=tag_bits,
            activation_reuse_tiles=activation_reuse_tiles,
            ifmap_cols_per_slice=ifmap_cols_per_slice,
            buffer_mode=buffer_mode,
            dram_bw_words_per_cycle=int(args.dram_bw_words_per_cycle),
            bytes_per_word=int(args.bytes_per_word),
            ifmap_buf_bytes=int(args.ifmap_buf_bytes),
            ofmap_buf_elems=int(args.ofmap_buf_elems),
        )

        out["layers"][layer_name] = {
            "m_tiles": m_tiles,
            "activation_reuse_tiles": activation_reuse_tiles,
            "ifmap_cols_per_slice": ifmap_cols_per_slice,
            "base_num_tiles": len(base_tile_lengths),
            "total_cycles": total_cycles,
            "total_stall_cycles": float(sim.get("total_stall_cycles", 0.0)),
            "injection_done_time": float(sim.get("injection_done_time", 0.0)),
            "ofmap_drain_done_time": float(sim.get("ofmap_drain_done_time", 0.0)),
        }
    
    if args.method_idx == 0:
        method_name = "general"
    elif args.method_idx == 1:
        method_name = "colcomb"
    elif args.method_idx == 2:
        method_name = "eureka"
    elif args.method_idx == 3:
        method_name = "ctp"

    output_json = f"sa_result_{args.model}_{method_name}.json"
    with open("./datas/"+output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"[OK] saved: {output_json}")


if __name__ == "__main__":
    main()
