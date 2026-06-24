#!/usr/bin/env python3

import os
import csv
import json
import argparse
from collections import defaultdict
from typing import Dict, Any, Set, Callable, Tuple, Optional, List

import torch
import torch.nn as nn
import torch.utils.benchmark as bench

try:
    from torch.profiler import profile, ProfilerActivity
except Exception:
    profile = None
    ProfilerActivity = None

# project-local loader, if available
try:
    from loading import load_resnet50, load_vgg16  # type: ignore
except Exception:
    load_resnet50 = None
    load_vgg16 = None

# torchvision models
try:
    from torchvision import models
    from torchvision.models import (
        MobileNet_V2_Weights,
        Inception_V3_Weights,
        VGG16_Weights,
        ResNet50_Weights,
    )
except Exception:
    models = None
    MobileNet_V2_Weights = None
    Inception_V3_Weights = None
    VGG16_Weights = None
    ResNet50_Weights = None

# transformers for BERT
try:
    from transformers import BertModel, BertConfig
except Exception:
    BertModel = None
    BertConfig = None


# -----------------------------------------------------------------------------
# GEMM-like operators
# -----------------------------------------------------------------------------
GEMM_OP_KEYS: Set[str] = {
    # matmul / linear
    "aten::mm",
    "aten::addmm",
    "aten::matmul",
    "aten::bmm",
    "aten::addbmm",
    "aten::baddbmm",
    "aten::einsum",
    "aten::linear",

    # convolution backends
    "aten::_convolution",
    "aten::convolution",
    "aten::conv1d",
    "aten::conv2d",
    "aten::conv3d",
    "aten::mkldnn_convolution",
    "aten::_slow_conv2d_forward",
    "aten::slow_conv_dilated2d",
    "aten::thnn_conv2d",
    "aten::_nnpack_spatial_convolution",
}


def ceil_div(a: int, b: int) -> int:
    if b <= 0:
        raise ValueError("divisor must be positive")
    return (a + b - 1) // b


def prod(xs) -> int:
    v = 1
    for x in xs:
        if isinstance(x, int) and x > 0:
            v *= x
    return int(v)


def is_shape_list(x) -> bool:
    return (
        isinstance(x, (list, tuple))
        and len(x) > 0
        and all(isinstance(v, int) for v in x)
        and all(v >= 0 for v in x)
    )


def extract_tensor_shapes(obj) -> List[Tuple[int, ...]]:
    """
    Extract tensor-like shapes from torch.profiler input_shapes.

    Examples of possible profiler shapes:
      [[1, 64, 56, 56], [], [], ...]
      [[[1, 64, 56, 56], [1, 64, 56, 56]], 1]
    """
    shapes: List[Tuple[int, ...]] = []

    def rec(x):
        if is_shape_list(x):
            shp = tuple(int(v) for v in x if int(v) > 0)
            if len(shp) > 0:
                shapes.append(shp)
            return
        if isinstance(x, (list, tuple)):
            for y in x:
                rec(y)

    rec(obj)
    return shapes


def shape_numel(shape: Tuple[int, ...]) -> int:
    return prod(shape)


def largest_shape(shapes: List[Tuple[int, ...]]) -> Optional[Tuple[int, ...]]:
    if not shapes:
        return None
    return max(shapes, key=lambda s: shape_numel(s))


def largest_numel(shapes: List[Tuple[int, ...]]) -> int:
    s = largest_shape(shapes)
    return shape_numel(s) if s is not None else 0


def sum_numel(shapes: List[Tuple[int, ...]]) -> int:
    return sum(shape_numel(s) for s in shapes)


def rows_and_last_dim(shape: Tuple[int, ...]) -> Tuple[int, int]:
    if not shape:
        return 0, 0
    last = int(shape[-1])
    if last <= 0:
        return 0, 0
    n = shape_numel(shape)
    return max(1, n // last), last


def is_gemm_like_op(key: str) -> bool:
    k = key.lower()

    if key in GEMM_OP_KEYS:
        return True

    # Include depthwise convolution as SA-side GEMM/Conv according to the user assumption.
    if "convolution" in k or "conv2d" in k or "conv3d" in k or "conv1d" in k:
        return True

    if "matmul" in k:
        return True

    return False


def classify_detail_category(key: str) -> Tuple[str, str]:
    k = key.lower()

    if is_gemm_like_op(key):
        if "conv" in k or "convolution" in k:
            return "GEMM", "GEMM_Conv"
        if "bmm" in k or "matmul" in k or "mm" in k or "einsum" in k:
            return "GEMM", "GEMM_MatMul"
        if "linear" in k or "addmm" in k:
            return "GEMM", "GEMM_Linear"
        return "GEMM", "GEMM_Other"

    if "scaled_dot_product_attention" in k or "_native_multi_head_attention" in k:
        return "NonGEMM", "Mixed_FusedAttention"

    if (
        "relu" in k
        or "hardtanh" in k
        or "hardswish" in k
        or "hardsigmoid" in k
        or "gelu" in k
        or "sigmoid" in k
        or "tanh" in k
        or "silu" in k
        or "threshold" in k
        or "clamp" in k
    ):
        return "NonGEMM", "Activation_Nonlinear"

    if (
        "batch_norm" in k
        or "native_batch_norm" in k
        or "layer_norm" in k
        or "native_layer_norm" in k
        or "group_norm" in k
        or "instance_norm" in k
    ):
        return "NonGEMM", "Normalization"

    if "pool" in k:
        return "NonGEMM", "Pooling"

    if "softmax" in k:
        return "NonGEMM", "Softmax"

    if (
        k in {
            "aten::add",
            "aten::sub",
            "aten::mul",
            "aten::div",
            "aten::rsub",
            "aten::pow",
            "aten::sqrt",
            "aten::rsqrt",
            "aten::reciprocal",
        }
        or "aten::add_" in k
        or "aten::mul_" in k
        or "aten::sub_" in k
        or "aten::div_" in k
    ):
        return "NonGEMM", "Elementwise_Arithmetic"

    if (
        k in {
            "aten::sum",
            "aten::mean",
            "aten::amax",
            "aten::max",
            "aten::min",
            "aten::argmax",
            "aten::argmin",
        }
        or "reduction" in k
    ):
        return "NonGEMM", "Reduction"

    # Actual metadata/layout/data-movement category.
    # This is handled in a layout-aware way later: metadata-only ops become zero-cost;
    # copy/cat/contiguous-like ops are modeled as memory movement.
    if (
        "view" in k
        or "reshape" in k
        or "flatten" in k
        or "transpose" in k
        or "permute" in k
        or "contiguous" in k
        or "clone" in k
        or "copy" in k
        or "cat" in k
        or "stack" in k
        or "slice" in k
        or "select" in k
        or "narrow" in k
        or "unsqueeze" in k
        or "squeeze" in k
        or "expand" in k
        or "as_strided" in k
        or k == "aten::to"
        or "aten::_to_copy" in k
    ):
        return "NonGEMM", "Tensor_Movement"

    # Framework/tensor-object management overheads: not accelerator datapath work.
    if (
        k in {
            "aten::size",
            "aten::stride",
            "aten::dim",
            "aten::numel",
            "aten::is_contiguous",
            "aten::is_same_size",
            "aten::sym_size",
            "aten::sym_stride",
            "aten::detach",
            "aten::detach_",
            "aten::alias",
            "aten::resolve_conj",
            "aten::resolve_neg",
            "aten::lift_fresh",
        }
        or "prim::" in k
    ):
        return "NonGEMM", "Framework_Overhead"

    # Allocation/output-buffer preparation overheads in PyTorch; accelerator buffers are assumed preallocated.
    if (
        "aten::empty" in k
        or "aten::new_empty" in k
        or "aten::resize" in k
        or "aten::set_" in k
    ):
        return "NonGEMM", "Memory_Allocation"

    # Element-wise mask/fill/compare-like operations: can be handled by vector lanes or a memory-fill path.
    if (
        "masked_fill" in k
        or k in {
            "aten::where",
            "aten::fill_",
            "aten::zero_",
            "aten::eq",
            "aten::ne",
            "aten::lt",
            "aten::le",
            "aten::gt",
            "aten::ge",
        }
        or "aten::eq_" in k
        or "aten::ne_" in k
        or "aten::lt_" in k
        or "aten::le_" in k
        or "aten::gt_" in k
        or "aten::ge_" in k
    ):
        return "NonGEMM", "Fill_Mask_Elementwise"

    if (
        "embedding" in k
        or "index_select" in k
        or "gather" in k
    ):
        return "NonGEMM", "Embedding_Indexing"

    if "dropout" in k or "bernoulli" in k or "rand" in k:
        return "NonGEMM", "Dropout_Random"

    return "NonGEMM", "Other_NonGEMM"


def is_layernorm_op(key: str) -> bool:
    k = key.lower()
    return "layer_norm" in k or "native_layer_norm" in k


def is_softmax_op(key: str, detail: str) -> bool:
    return detail == "Softmax" or "softmax" in key.lower()


def estimate_vector_cycles_for_op(
    key: str,
    detail: str,
    shapes: List[Tuple[int, ...]],
    vector_lanes: int,
    mem_lanes: int,
    fold_bn: bool,
    activation_epilogue: bool,
    model_name: str,
) -> Tuple[int, str]:
    """
    Estimate vector cycles for a non-GEMM op.

    Notes:
    - This function estimates cycles only for operations actually modeled on the vector unit.
    - Softmax/LayerNorm CPU fallback is applied outside this function so that CPU-estimated ms can be preserved.
    """
    k = key.lower()
    is_bert = model_name.lower() in ("bert", "bert-base", "bert-base-uncased")

    if detail.startswith("GEMM"):
        return 0, "GEMM op handled by SA"

    n_largest = largest_numel(shapes)
    n_sum = sum_numel(shapes)

    if n_largest <= 0:
        return 0, "No tensor shape; treated as framework/bookkeeping overhead"

    # CNN BatchNorm folding
    if detail == "Normalization" and fold_bn and not is_bert and not is_layernorm_op(key):
        return 0, "CNN BatchNorm folded into preceding convolution"

    # CNN activation epilogue
    if detail == "Activation_Nonlinear" and activation_epilogue and not is_bert:
        return 0, "CNN activation handled by SA/vector epilogue"

    if detail == "Tensor_Movement":
        # Metadata-only layout ops: zero-cost in accelerator model.
        if (
            "view" in k
            or "reshape" in k
            or "flatten" in k
            or "transpose" in k
            or "permute" in k
            or "slice" in k
            or "select" in k
            or "narrow" in k
            or "unsqueeze" in k
            or "squeeze" in k
            or "expand" in k
            or "as_strided" in k
        ) and not ("copy" in k or "clone" in k or "contiguous" in k or "cat" in k):
            return 0, "View/layout metadata op without data copy"

        elems = n_sum if ("cat" in k or "stack" in k) else n_largest
        mem_cycles = ceil_div(2 * elems, mem_lanes)
        return mem_cycles, "Tensor movement/copy modeled as read+write traffic"

    if detail == "Activation_Nonlinear":
        if "gelu" in k:
            # GELU is kept on vector unit; it is modeled as a LUT/polynomial approximation.
            compute_cycles = 3 * ceil_div(n_largest, vector_lanes)
            mem_cycles = ceil_div(2 * n_largest, mem_lanes)
            return max(compute_cycles, mem_cycles), "GELU modeled as LUT/polynomial vector approximation"

        if "relu6" in k or "hardtanh" in k or "clamp" in k:
            compute_cycles = 2 * ceil_div(n_largest, vector_lanes)
            mem_cycles = ceil_div(2 * n_largest, mem_lanes)
            return max(compute_cycles, mem_cycles), "Clamp/ReLU6 modeled as two vector compare/select passes"

        compute_cycles = ceil_div(n_largest, vector_lanes)
        mem_cycles = ceil_div(2 * n_largest, mem_lanes)
        return max(compute_cycles, mem_cycles), "Activation modeled as elementwise vector op"

    if detail == "Normalization":
        # LayerNorm CPU fallback is applied outside. This is for BatchNorm/groupnorm fallback.
        compute_cycles = 2 * ceil_div(n_largest, vector_lanes)
        mem_cycles = ceil_div(2 * n_largest, mem_lanes)
        return max(compute_cycles, mem_cycles), "Normalization modeled as vector affine operation"

    if detail == "Elementwise_Arithmetic":
        compute_cycles = ceil_div(n_largest, vector_lanes)
        mem_cycles = ceil_div(3 * n_largest, mem_lanes)
        return max(compute_cycles, mem_cycles), "Elementwise arithmetic modeled as vector read/read/write"

    if detail == "Pooling":
        # Kernel/stride is not always recoverable from profiler shapes; approximate as input scan + output write.
        compute_cycles = ceil_div(n_largest, vector_lanes)
        mem_cycles = ceil_div(n_largest + max(1, n_largest // 4), mem_lanes)
        return max(compute_cycles, mem_cycles), "Pooling modeled as input scan plus output write approximation"

    if detail == "Reduction":
        compute_cycles = 2 * ceil_div(n_largest, vector_lanes)
        mem_cycles = ceil_div(n_largest + max(1, n_largest // vector_lanes), mem_lanes)
        return max(compute_cycles, mem_cycles), "Reduction modeled as vector reduction"

    if detail == "Softmax":
        # Usually not reached when CPU fallback is enabled.
        shp = largest_shape(shapes)
        if shp is None:
            return 0, "Softmax has no shape"
        rows, d = rows_and_last_dim(shp)
        compute_cycles = rows * ceil_div(d, vector_lanes) * 6
        mem_cycles = ceil_div(2 * n_largest, mem_lanes)
        return max(compute_cycles, mem_cycles), "Softmax modeled as max/exp/sum/reciprocal/multiply"

    if detail == "Embedding_Indexing":
        mem_cycles = ceil_div(2 * n_largest, mem_lanes)
        return mem_cycles, "Embedding/indexing modeled as memory gather traffic"

    if detail == "Framework_Overhead":
        return 0, "Framework bookkeeping/tensor-object metadata overhead removed from accelerator datapath model"

    if detail == "Memory_Allocation":
        return 0, "PyTorch allocation/output-buffer preparation overhead removed; accelerator buffers are assumed preallocated"

    if detail == "Fill_Mask_Elementwise":
        compute_cycles = ceil_div(n_largest, vector_lanes)
        mem_cycles = ceil_div(2 * n_largest, mem_lanes)
        return max(compute_cycles, mem_cycles), "Mask/fill/compare modeled as vector elementwise or memory-fill operation"

    if detail == "Mixed_FusedAttention":
        # Conservative fallback model for fused attention. Not CPU fallback by default because it may contain GEMM.
        shp = largest_shape(shapes)
        if shp is None:
            return 0, "Fused attention has no shape"
        rows, d = rows_and_last_dim(shp)
        compute_cycles = rows * ceil_div(d, vector_lanes) * 6
        mem_cycles = ceil_div(2 * n_largest, mem_lanes)
        return max(compute_cycles, mem_cycles), "Fused attention approximated as softmax-like vector work"

    if detail == "Dropout_Random":
        return 0, "Dropout should be inactive during inference"

    mem_cycles = ceil_div(2 * n_largest, mem_lanes)
    return mem_cycles, "Other non-GEMM modeled as memory read+write if tensor shape exists"


# -----------------------------------------------------------------------------
# Model loading and inputs
# -----------------------------------------------------------------------------
def _load_vision_model_torchvision(name: str, pretrained: bool) -> nn.Module:
    if models is None:
        raise RuntimeError("torchvision import failed. Please install torchvision.")

    if name == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V2 if (pretrained and ResNet50_Weights is not None) else None
        return models.resnet50(weights=weights)

    if name in ("mobilenet_v2", "mobilenet"):
        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if (pretrained and MobileNet_V2_Weights is not None) else None
        return models.mobilenet_v2(weights=weights)

    if name == "vgg16":
        weights = VGG16_Weights.IMAGENET1K_V1 if (pretrained and VGG16_Weights is not None) else None
        return models.vgg16(weights=weights)

    if name in ("inception_v3", "inception"):
        weights = Inception_V3_Weights.IMAGENET1K_V1 if (pretrained and Inception_V3_Weights is not None) else None
        return models.inception_v3(weights=weights, aux_logits=True)

    raise ValueError(f"Unknown torchvision model: {name}")


def load_model(name: str, device: torch.device, pretrained: bool, bert_attn_implementation: str = "eager") -> nn.Module:
    name = name.lower()

    if name == "resnet50" and load_resnet50 is not None:
        m = load_resnet50(pretrained=pretrained, device=device)
        m.to(device)
        m.eval()
        return m

    if name == "vgg16" and load_vgg16 is not None:
        m = load_vgg16(pretrained=pretrained, device=device)
        m.to(device)
        m.eval()
        return m

    if name in ("bert", "bert-base", "bert-base-uncased"):
        if BertModel is None:
            raise ImportError("transformers package is required for BERT. Install with `pip install transformers`.")

        attn_impl = bert_attn_implementation
        if attn_impl == "default":
            attn_impl = None

        def make_eager_config() -> "BertConfig":
            if BertConfig is None:
                raise RuntimeError("BertConfig not available.")
            cfg = BertConfig()
            if attn_impl is not None:
                # Transformers versions differ: some use _attn_implementation internally.
                setattr(cfg, "_attn_implementation", attn_impl)
                setattr(cfg, "attn_implementation", attn_impl)
            return cfg

        if pretrained:
            try:
                if attn_impl is None:
                    m = BertModel.from_pretrained("bert-base-uncased")
                else:
                    m = BertModel.from_pretrained("bert-base-uncased", attn_implementation=attn_impl)
            except TypeError:
                # Older Transformers: pass an explicitly configured BertConfig instead.
                if BertConfig is None:
                    raise
                try:
                    cfg = BertConfig.from_pretrained("bert-base-uncased")
                    if attn_impl is not None:
                        setattr(cfg, "_attn_implementation", attn_impl)
                        setattr(cfg, "attn_implementation", attn_impl)
                    m = BertModel.from_pretrained("bert-base-uncased", config=cfg)
                except Exception:
                    m = BertModel(make_eager_config())
            except Exception:
                m = BertModel(make_eager_config())
        else:
            m = BertModel(make_eager_config())

        m.to(device)
        m.eval()
        return m

    m = _load_vision_model_torchvision(name, pretrained=pretrained)
    m.to(device)
    m.eval()
    return m


def make_run_callable(
    model_name: str,
    model: nn.Module,
    batch: int,
    device: torch.device,
    img_size: Optional[int],
    bert_seq_len: int,
    bert_vocab: int,
) -> Tuple[Callable[[], Any], Dict[str, Any]]:
    name = model_name.lower()

    if name in ("bert", "bert-base", "bert-base-uncased"):
        input_ids = torch.randint(
            low=0,
            high=bert_vocab,
            size=(batch, bert_seq_len),
            device=device,
            dtype=torch.long,
        )
        attention_mask = torch.ones((batch, bert_seq_len), device=device, dtype=torch.long)

        def run():
            return model(input_ids=input_ids, attention_mask=attention_mask)

        return run, {
            "input_type": "bert",
            "bert_seq_len": bert_seq_len,
            "bert_vocab": bert_vocab,
        }

    default_size = 299 if name in ("inception_v3", "inception") else 224
    h = int(img_size) if img_size is not None else default_size
    x = torch.randn(batch, 3, h, h, device=device)

    def run():
        out = model(x)
        if hasattr(out, "logits"):
            return out.logits
        if isinstance(out, (tuple, list)) and len(out) > 0:
            return out[0]
        return out

    return run, {
        "input_type": "vision",
        "image_size": h,
    }


# -----------------------------------------------------------------------------
# Measurement and profiling
# -----------------------------------------------------------------------------
def make_inference_callable(run: Callable[[], Any]) -> Callable[[], Any]:
    def run_infer():
        with torch.inference_mode():
            return run()
    return run_infer


def measure_total_cpu_ms(run: Callable[[], Any], warmup: int, min_run_s: float) -> float:
    run_infer = make_inference_callable(run)

    for _ in range(warmup):
        _ = run_infer()

    timer = bench.Timer(
        stmt="run_infer()",
        globals={"run_infer": run_infer},
        num_threads=torch.get_num_threads(),
    )
    result = timer.blocked_autorange(min_run_time=min_run_s)
    return float(result.median) * 1e3


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: List[Dict[str, Any]], by: str) -> List[Dict[str, Any]]:
    acc = defaultdict(lambda: {
        "self_cpu_us_total": 0.0,
        "cpu_time_us_total": 0.0,
        "count_total": 0,
        "num_ops": 0,
        "cpu_est_ms": 0.0,
        "vector_cycles": 0,
        "vector_ms_shape_est": 0.0,
        "cpu_fallback_ms": 0.0,
    })

    total_self = sum(float(r["self_cpu_us_total"]) for r in rows)

    for r in rows:
        k = str(r[by])
        acc[k]["self_cpu_us_total"] += float(r["self_cpu_us_total"])
        acc[k]["cpu_time_us_total"] += float(r["cpu_time_us_total"])
        acc[k]["count_total"] += int(r["count_total"])
        acc[k]["num_ops"] += 1
        acc[k]["cpu_est_ms"] += float(r["cpu_est_ms"])
        acc[k]["vector_cycles"] += int(r["vector_cycles"])
        acc[k]["vector_ms_shape_est"] += float(r["vector_ms_shape_est"])
        if bool(r.get("cpu_fallback", False)):
            acc[k]["cpu_fallback_ms"] += float(r["cpu_est_ms"])

    out = []
    for k, v in acc.items():
        out.append({
            by: k,
            "self_cpu_us_total": v["self_cpu_us_total"],
            "self_fraction": v["self_cpu_us_total"] / total_self if total_self > 0 else 0.0,
            "cpu_time_us_total": v["cpu_time_us_total"],
            "count_total": v["count_total"],
            "num_ops": v["num_ops"],
            "cpu_est_ms": v["cpu_est_ms"],
            "vector_cycles": int(v["vector_cycles"]),
            "vector_ms_shape_est": v["vector_ms_shape_est"],
            "cpu_fallback_ms": v["cpu_fallback_ms"],
        })

    out.sort(key=lambda x: x["cpu_est_ms"], reverse=True)
    return out


def profile_and_estimate_vector(
    run: Callable[[], Any],
    model_name: str,
    total_cpu_ms: float,
    warmup: int,
    prof_iters: int,
    vector_lanes: int,
    mem_lanes: int,
    freq_mhz: float,
    fold_bn: bool,
    activation_epilogue: bool,
    cpu_fallback_softmax: bool,
    cpu_fallback_layernorm: bool,
    cpu_fallback_tensor_movement: bool,
    cpu_fallback_embedding_indexing: bool,
    cpu_fallback_fused_attention: bool,
    cpu_fallback_other_nongemm: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if profile is None:
        raise RuntimeError("torch.profiler is not available in this environment.")

    run_infer = make_inference_callable(run)

    for _ in range(warmup):
        _ = run_infer()

    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=False,
    ) as prof:
        for _ in range(prof_iters):
            _ = run_infer()

    # Separate same operators with different input shapes.
    ka = prof.key_averages(group_by_input_shape=True)

    total_self_us = sum(float(evt.self_cpu_time_total) for evt in ka)
    rows: List[Dict[str, Any]] = []
    warnings: List[str] = []

    for evt in ka:
        key = evt.key
        key_lower = key.lower()
        self_us = float(evt.self_cpu_time_total)
        cpu_total_us = float(evt.cpu_time_total)
        count = int(evt.count)

        coarse, detail = classify_detail_category(key)

        input_shapes_raw = getattr(evt, "input_shapes", [])
        shapes = extract_tensor_shapes(input_shapes_raw)

        cpu_est_ms = total_cpu_ms * (self_us / total_self_us) if total_self_us > 0 else 0.0

        vector_cycles = 0
        vector_ms = 0.0
        reason = "N/A"
        cpu_fallback = False

        if coarse == "NonGEMM":
            vector_cycles, reason = estimate_vector_cycles_for_op(
                key=key,
                detail=detail,
                shapes=shapes,
                vector_lanes=vector_lanes,
                mem_lanes=mem_lanes,
                fold_bn=fold_bn,
                activation_epilogue=activation_epilogue,
                model_name=model_name,
            )
            vector_ms = vector_cycles / (freq_mhz * 1e6) * 1e3

            # CPU fallback policy: preserve CPU-estimated time for Softmax and LayerNorm.
            if cpu_fallback_softmax and is_softmax_op(key, detail):
                cpu_fallback = True
                vector_cycles = 0
                vector_ms = cpu_est_ms
                reason = "CPU fallback: Softmax requires exp/reduction/reciprocal support."

            if cpu_fallback_layernorm and detail == "Normalization" and is_layernorm_op(key):
                cpu_fallback = True
                vector_cycles = 0
                vector_ms = cpu_est_ms
                reason = "CPU fallback: LayerNorm requires mean/variance reduction and rsqrt."

            if cpu_fallback_tensor_movement and detail == "Tensor_Movement":
                cpu_fallback = True
                vector_cycles = 0
                vector_ms = cpu_est_ms
                reason = (
                    "CPU fallback: Tensor movement represents layout/data movement rather than "
                    "vector arithmetic work."
                )

            if cpu_fallback_embedding_indexing and detail == "Embedding_Indexing":
                cpu_fallback = True
                vector_cycles = 0
                vector_ms = cpu_est_ms
                reason = (
                    "CPU fallback: Embedding/indexing involves gather-like memory access and is "
                    "not modeled as regular SIMD arithmetic."
                )

            if cpu_fallback_other_nongemm and detail == "Other_NonGEMM":
                cpu_fallback = True
                vector_cycles = 0
                vector_ms = cpu_est_ms
                reason = (
                    "CPU fallback: unclassified non-GEMM operations are kept as CPU-estimated "
                    "latency to avoid claiming unsupported vector-unit acceleration."
                )

            if detail == "Mixed_FusedAttention":
                warnings.append(
                    f"Fused attention op detected: {key}. It may include both matmul and softmax."
                )
                if cpu_fallback_fused_attention:
                    cpu_fallback = True
                    vector_cycles = 0
                    vector_ms = cpu_est_ms
                    reason = "CPU fallback: fused attention may include softmax/nonlinear work."

        rows.append({
            "key": key,
            "coarse_category": coarse,
            "detail_category": detail,
            "input_shapes": str(input_shapes_raw),
            "parsed_shapes": str(shapes),
            "largest_numel": largest_numel(shapes),
            "sum_numel": sum_numel(shapes),
            "self_cpu_us_total": self_us,
            "self_fraction": self_us / total_self_us if total_self_us > 0 else 0.0,
            "cpu_time_us_total": cpu_total_us,
            "count_total": count,
            "count_per_iter": count / prof_iters if prof_iters > 0 else 0.0,
            "cpu_est_ms": cpu_est_ms,
            "vector_cycles": int(vector_cycles),
            "vector_ms_shape_est": vector_ms,
            "cpu_fallback": cpu_fallback,
            "vector_model_reason": reason,
        })

    rows.sort(key=lambda x: x["cpu_est_ms"], reverse=True)

    gemm_cpu_ms = sum(float(r["cpu_est_ms"]) for r in rows if r["coarse_category"] == "GEMM")
    nongemm_cpu_ms = sum(float(r["cpu_est_ms"]) for r in rows if r["coarse_category"] == "NonGEMM")
    nongemm_vector_ms = sum(float(r["vector_ms_shape_est"]) for r in rows if r["coarse_category"] == "NonGEMM")
    nongemm_vector_cycles = sum(int(r["vector_cycles"]) for r in rows if r["coarse_category"] == "NonGEMM")
    cpu_fallback_ms = sum(float(r["cpu_est_ms"]) for r in rows if bool(r.get("cpu_fallback", False)))

    stats = {
        "prof_total_self_us": total_self_us,
        "gemm_cpu_ms_est": gemm_cpu_ms,
        "nongemm_cpu_ms_est": nongemm_cpu_ms,
        "nongemm_vector_cycles_shape_est": int(nongemm_vector_cycles),
        "nongemm_vector_ms_shape_est": nongemm_vector_ms,
        "cpu_fallback_ms": cpu_fallback_ms,
        "warnings": sorted(set(warnings)),
    }

    return rows, stats


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="resnet50",
        choices=["resnet50", "mobilenet_v2", "inception_v3", "bert", "vgg16"],
    )
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--batch", type=int, default=32)

    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--bert-seq-len", type=int, default=128)
    parser.add_argument("--bert-vocab", type=int, default=30522)

    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--interop_threads", type=int, default=1)

    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--min_run_s", type=float, default=2.0)

    parser.add_argument("--prof_warmup", type=int, default=10)
    parser.add_argument("--prof_iters", type=int, default=20)

    parser.add_argument("--vector-lanes", type=int, default=64)
    parser.add_argument("--mem-lanes", type=int, default=64)
    parser.add_argument("--freq-mhz", type=float, default=500.0)

    # Optional fusion rules. Default is no fusion/overlap except metadata-only view removal.
    parser.add_argument("--fold-bn", action="store_true",
                        help="Fold CNN BatchNorm into preceding convolution and remove it from vector non-GEMM.")
    parser.add_argument("--activation-epilogue", action="store_true",
                        help="Handle CNN activation by SA/vector epilogue and remove it from vector non-GEMM.")

    # CPU fallback policy. Enabled by default for operations that should not be claimed as vector-unit work.
    parser.add_argument("--no-cpu-fallback-softmax", action="store_true",
                        help="If set, Softmax is modeled on the vector unit instead of CPU fallback.")
    parser.add_argument("--no-cpu-fallback-layernorm", action="store_true",
                        help="If set, LayerNorm is modeled on the vector unit instead of CPU fallback.")
    parser.add_argument("--cpu-fallback-all-tensor-movement", action="store_true",
                        help="If set, keep all Tensor_Movement as CPU-estimated latency. Default is layout-aware: metadata-only ops are zero-cost and copy/cat/contiguous-like ops are memory-modeled.")
    parser.add_argument("--no-cpu-fallback-embedding-indexing", action="store_true",
                        help="If set, Embedding_Indexing is modeled by the vector/memory model instead of CPU fallback.")
    parser.add_argument("--no-cpu-fallback-fused-attention", action="store_true",
                        help="If set, fused attention ops are modeled on the vector unit instead of CPU fallback.")
    parser.add_argument("--no-cpu-fallback-other-nongemm", action="store_true",
                        help="If set, Other_NonGEMM is modeled by the vector/memory model instead of CPU fallback.")

    parser.add_argument("--bert-attn-implementation", type=str, default="eager",
                        choices=["eager", "sdpa", "default"],
                        help="BERT attention backend. Default=eager avoids fused scaled_dot_product_attention when supported.")

    parser.add_argument("--out", type=str, default=None,
                        help="Output JSON path. Default: <out_dir>/cpu_<model>.json")
    parser.add_argument("--out-dir", type=str, default="./datas",
                        help="Output directory used when --out is not provided.")
    parser.add_argument("--ops-out", type=str, default=None)
    parser.add_argument("--coarse-out", type=str, default=None)
    parser.add_argument("--detail-out", type=str, default=None)

    args = parser.parse_args()

    if args.out is None:
        args.out = os.path.join(args.out_dir, f"cpu_{args.model}.json")

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.interop_threads)

    device = torch.device("cpu")
    model = load_model(
        args.model,
        device=device,
        pretrained=(not args.no_pretrained),
        bert_attn_implementation=args.bert_attn_implementation,
    )
    run, input_meta = make_run_callable(
        model_name=args.model,
        model=model,
        batch=args.batch,
        device=device,
        img_size=args.img_size,
        bert_seq_len=args.bert_seq_len,
        bert_vocab=args.bert_vocab,
    )

    total_cpu_ms = measure_total_cpu_ms(
        run=run,
        warmup=args.warmup,
        min_run_s=args.min_run_s,
    )

    rows, stats = profile_and_estimate_vector(
        run=run,
        model_name=args.model,
        total_cpu_ms=total_cpu_ms,
        warmup=args.prof_warmup,
        prof_iters=args.prof_iters,
        vector_lanes=args.vector_lanes,
        mem_lanes=args.mem_lanes,
        freq_mhz=args.freq_mhz,
        fold_bn=args.fold_bn,
        activation_epilogue=args.activation_epilogue,
        cpu_fallback_softmax=(not args.no_cpu_fallback_softmax),
        cpu_fallback_layernorm=(not args.no_cpu_fallback_layernorm),
        cpu_fallback_tensor_movement=args.cpu_fallback_all_tensor_movement,
        cpu_fallback_embedding_indexing=(not args.no_cpu_fallback_embedding_indexing),
        cpu_fallback_fused_attention=(not args.no_cpu_fallback_fused_attention),
        cpu_fallback_other_nongemm=(not args.no_cpu_fallback_other_nongemm),
    )

    base, _ = os.path.splitext(args.out)
    ops_out = args.ops_out or f"{base}.ops.csv"
    coarse_out = args.coarse_out or f"{base}.coarse_summary.csv"
    detail_out = args.detail_out or f"{base}.detail_summary.csv"

    coarse_summary = summarize(rows, by="coarse_category")
    detail_summary = summarize(rows, by="detail_category")

    write_csv(ops_out, rows)
    write_csv(coarse_out, coarse_summary)
    write_csv(detail_out, detail_summary)

    result: Dict[str, Any] = {
        "model": args.model,
        "pretrained": (not args.no_pretrained),
        "batch": args.batch,
        **input_meta,

        "threads": args.threads,
        "interop_threads": args.interop_threads,

        "total_cpu_ms_benchmark": total_cpu_ms,

        "gemm_cpu_ms_est": stats["gemm_cpu_ms_est"],
        "nongemm_cpu_ms_est": stats["nongemm_cpu_ms_est"],

        "vector_lanes": args.vector_lanes,
        "mem_lanes": args.mem_lanes,
        "freq_mhz": args.freq_mhz,
        "fold_bn": args.fold_bn,
        "activation_epilogue": args.activation_epilogue,
        "cpu_fallback_softmax": (not args.no_cpu_fallback_softmax),
        "cpu_fallback_layernorm": (not args.no_cpu_fallback_layernorm),
        "cpu_fallback_all_tensor_movement": args.cpu_fallback_all_tensor_movement,
        "cpu_fallback_embedding_indexing": (not args.no_cpu_fallback_embedding_indexing),
        "cpu_fallback_fused_attention": (not args.no_cpu_fallback_fused_attention),
        "cpu_fallback_other_nongemm": (not args.no_cpu_fallback_other_nongemm),
        "bert_attn_implementation": args.bert_attn_implementation,

        "nongemm_vector_cycles_shape_est": stats["nongemm_vector_cycles_shape_est"],
        "nongemm_vector_ms_shape_est": stats["nongemm_vector_ms_shape_est"],
        "cpu_fallback_ms": stats["cpu_fallback_ms"],

        # Canonical non-GEMM value for the SA+VP+CPU-fallback E2E pipeline.
        # Despite the historical key name, nongemm_vector_ms_shape_est is hybrid:
        # vector-estimated supported ops + CPU-estimated fallback ops.
        "nongemm_hybrid_ms_for_e2e": stats["nongemm_vector_ms_shape_est"],
        "nongemm_total_ms_for_e2e": stats["nongemm_vector_ms_shape_est"],
        "nongemm_total_ms_for_e2e_source": "nongemm_vector_ms_shape_est (hybrid: VP-supported ops + CPU fallback)",

        "cpu_style_total_ms_est": stats["gemm_cpu_ms_est"] + stats["nongemm_cpu_ms_est"],
        "cpu_gemm_plus_vector_nongemm_total_ms_est": (
            stats["gemm_cpu_ms_est"] + stats["nongemm_vector_ms_shape_est"]
        ),
        "nongemm_vector_speedup_over_cpu_est": (
            stats["nongemm_cpu_ms_est"] / stats["nongemm_vector_ms_shape_est"]
            if stats["nongemm_vector_ms_shape_est"] > 0 else None
        ),

        "ops_csv": ops_out,
        "coarse_summary_csv": coarse_out,
        "detail_summary_csv": detail_out,

        "assumption_note": (
            "No overlap is modeled. GEMM/Conv/Linear/MatMul operators are assumed to be handled by the SA. "
            "Non-GEMM operators are modeled as serial execution on a 64-wide vector/post-processing unit. "
            "Softmax, LayerNorm, Embedding_Indexing, fused attention ops, and unknown Other_NonGEMM "
            "are kept as CPU-estimated latency by default. Tensor_Movement is layout-aware: metadata-only "
            "ops are removed and actual-copy ops are modeled as memory movement."
        ),
        "classification_note": (
            "Depthwise convolution is included in convolution-like operators and classified as GEMM/SA-side."
        ),
        "warnings": stats["warnings"],
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("=" * 90)
    print(f"[INFO] Saved JSON: {args.out}")
    print(f"[INFO] ops CSV: {ops_out}")
    print(f"[INFO] coarse summary CSV: {coarse_out}")
    print(f"[INFO] detail summary CSV: {detail_out}")
    print("-" * 90)
    print(f"[INFO] model: {args.model}")
    print(f"[INFO] total_cpu_ms_benchmark: {total_cpu_ms:.6f} ms")
    print(f"[INFO] gemm_cpu_ms_est: {stats['gemm_cpu_ms_est']:.6f} ms")
    print(f"[INFO] nongemm_cpu_ms_est: {stats['nongemm_cpu_ms_est']:.6f} ms")
    print(f"[INFO] nongemm_vector_ms_shape_est: {stats['nongemm_vector_ms_shape_est']:.6f} ms")
    print(f"[INFO] cpu_fallback_ms: {stats['cpu_fallback_ms']:.6f} ms")
    if stats["nongemm_vector_ms_shape_est"] > 0:
        print(
            f"[INFO] nonGEMM vector speedup over CPU-est: "
            f"{stats['nongemm_cpu_ms_est'] / stats['nongemm_vector_ms_shape_est']:.4f}x"
        )
    print(
        f"[INFO] vector_lanes={args.vector_lanes}, mem_lanes={args.mem_lanes}, "
        f"freq={args.freq_mhz} MHz"
    )
    print(f"[INFO] fold_bn={args.fold_bn}, activation_epilogue={args.activation_epilogue}")
    print(f"[INFO] cpu_fallback_softmax={not args.no_cpu_fallback_softmax}")
    print(f"[INFO] cpu_fallback_layernorm={not args.no_cpu_fallback_layernorm}")
    print(f"[INFO] cpu_fallback_all_tensor_movement={args.cpu_fallback_all_tensor_movement}")
    print(f"[INFO] cpu_fallback_embedding_indexing={not args.no_cpu_fallback_embedding_indexing}")
    print(f"[INFO] cpu_fallback_fused_attention={not args.no_cpu_fallback_fused_attention}")
    print(f"[INFO] cpu_fallback_other_nongemm={not args.no_cpu_fallback_other_nongemm}")
    print(f"[INFO] bert_attn_implementation={args.bert_attn_implementation}")
    if stats["warnings"]:
        print("[WARNINGS]")
        for w in stats["warnings"]:
            print(f"  - {w}")
    print("=" * 90)


if __name__ == "__main__":
    main()
