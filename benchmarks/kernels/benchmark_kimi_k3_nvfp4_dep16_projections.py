# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbenchmark Kimi-K3 NVFP4 TP1 decode projection GEMMs on GB300."""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import statistics
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.kernels.linear.cute_dsl import ll_bf16
from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import (
    SkinnyGemmConfig,
    shape_dynamic_skinny_gemm,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    _w8a8_triton_block_scaled_mm,
    deepgemm_post_process_fp8_weight_block,
    per_token_group_quant_fp8,
    per_token_group_quant_fp8_packed_for_deepgemm,
    w8a8_triton_block_scaled_mm,
)
from vllm.models.kimi_k3.nvidia.low_latency_gemm import (
    KIMI_K3_PROJECTIONS,
    _launch_kimi_k3_bf16_gemm_nt,
    try_low_latency_gemm,
)
from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops.kda import rms_norm_gated
from vllm.triton_utils import tl, triton
from vllm.utils.deep_gemm import (
    _block_size_multiple_scope,
    _import_deep_gemm,
    fp8_gemm_nt,
    per_block_cast_to_fp8,
)
from vllm.utils.flashinfer import flashinfer_fp8_blockscale_gemm


@dataclass(frozen=True)
class Shape:
    name: str
    n: int
    k: int
    calls: int
    precision: str
    residual: bool = False
    fp32_output: bool = False
    input_row_stride: int | None = None
    input_offset: int = 0
    batch: int = 1


FP8_SHAPES = (
    Shape("kda_in_proj_qkvgfab", 49408, 7168, 69, "fp8"),
    Shape(
        "kda_f_b_proj",
        12288,
        128,
        69,
        "fp8",
        input_row_stride=49408,
        input_offset=49152,
    ),
    Shape("kda_mla_o_proj", 7168, 12288, 93, "fp8"),
    Shape("mla_fused_qkv_a", 2176, 7168, 24, "fp8"),
    Shape("mla_q_b_proj", 18432, 1536, 24, "fp8"),
    Shape("mla_output_gate", 12288, 7168, 24, "fp8"),
)

BF16_SHAPES = (
    Shape("moe_router", 896, 7168, 92, "bf16", fp32_output=True),
    Shape("routed_latent_down", 3584, 7168, 92, "bf16"),
    Shape("routed_latent_up_add", 7168, 3584, 92, "bf16", residual=True),
    Shape("dense_gate_up", 67584, 7168, 1, "bf16"),
    Shape("dense_down", 7168, 33792, 1, "bf16"),
)

DRAFT_BF16_SHAPES = (
    Shape("draft_context_proj", 7168, 35840, 1, "bf16"),
    Shape("draft_context_kv_proj", 2880, 7168, 1, "bf16"),
    Shape("draft_mla_fused_qkv_a", 2112, 7168, 5, "bf16"),
    Shape("draft_mla_q_b_proj", 12288, 1536, 5, "bf16"),
    Shape("draft_mla_o_proj", 7168, 8192, 5, "bf16"),
    Shape("draft_dense_gate_up", 28672, 7168, 5, "bf16"),
    Shape("draft_dense_down", 7168, 14336, 5, "bf16"),
    Shape("draft_lm_head", 163840, 7168, 1, "bf16"),
    Shape("draft_markov_w2", 163840, 256, 4, "bf16"),
)

BMM_SHAPES = (
    Shape("mla_w_uk_absorbed_bmm", 512, 128, 24, "bmm", batch=96),
    Shape("mla_w_uv_absorbed_bmm", 128, 512, 24, "bmm", batch=96),
)

DEEPGEMM_CONSTRAINTS: tuple[tuple[int, int] | None, ...] = (
    None,
    *((block_m, 1) for block_m in range(16, 257, 16)),
    *((block_m, 64) for block_m in (64, 128, 192, 256)),
    *((block_m, 96) for block_m in (96, 192)),
    *((block_m, 128) for block_m in (128, 256)),
)

TRITON_FP8_CONFIGS = tuple(
    (block_m, block_n, num_warps, num_stages)
    for block_m in (16, 32, 64, 128)
    for block_n in (128, 256)
    for num_warps in (4, 8)
    for num_stages in (1, 2, 3, 4)
)


def benchmark(fn: Callable[[], torch.Tensor], rep: int) -> tuple[float, float, float]:
    values = triton.testing.do_bench_cudagraph(
        fn,
        rep=rep,
        quantiles=[0.5, 0.1, 0.9],
    )
    return tuple(float(value) * 1000 for value in values)


@triton.jit
def _fused_bf16_fp8_block_mm_k128(
    a_ptr,
    b_ptr,
    b_scale_ptr,
    output_ptr,
    m,
    n: tl.constexpr,
    stride_am,
    stride_bk,
    stride_bn,
    stride_bsn,
    stride_om,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
) -> None:
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offsets_m = pid_m * block_m + tl.arange(0, block_m)
    offsets_n = pid_n * block_n + tl.arange(0, block_n)
    offsets_k = tl.arange(0, 128)
    mask_m = offsets_m < m
    mask_n = offsets_n < n

    a = tl.load(
        a_ptr + offsets_m[:, None] * stride_am + offsets_k[None, :],
        mask=mask_m[:, None],
        other=0.0,
    ).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(a), axis=1) / 448.0, 1e-10)
    scale = tl.exp2(tl.ceil(tl.log2(scale)))
    a_q = tl.clamp(a / scale[:, None], -448.0, 448.0).to(b_ptr.dtype.element_ty)
    b = tl.load(
        b_ptr + offsets_k[:, None] * stride_bk + offsets_n[None, :] * stride_bn,
        mask=mask_n[None, :],
        other=0.0,
    )
    b_scale = tl.load(
        b_scale_ptr + (offsets_n // 128) * stride_bsn,
        mask=mask_n,
        other=0.0,
    )
    output = tl.dot(a_q, b).to(tl.float32)
    output *= scale[:, None] * b_scale[None, :]
    tl.store(
        output_ptr + offsets_m[:, None] * stride_om + offsets_n[None, :],
        output,
        mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.jit
def _bf16_bmm_kernel(
    a_ptr,
    b_ptr,
    output_ptr,
    m: tl.constexpr,
    n: tl.constexpr,
    k: tl.constexpr,
    stride_ab: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bb: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
) -> None:
    batch_id = tl.program_id(1)
    tile_id = tl.program_id(0)
    num_n_tiles = tl.cdiv(n, block_n)
    m_tile = tile_id // num_n_tiles
    n_tile = tile_id % num_n_tiles
    m_offsets = m_tile * block_m + tl.arange(0, block_m)
    n_offsets = n_tile * block_n + tl.arange(0, block_n)
    k_offsets = tl.arange(0, block_k)
    a_ptrs = (
        a_ptr
        + batch_id * stride_ab
        + m_offsets[:, None] * stride_am
        + k_offsets[None, :] * stride_ak
    )
    b_ptrs = (
        b_ptr
        + batch_id * stride_bb
        + k_offsets[:, None] * stride_bk
        + n_offsets[None, :] * stride_bn
    )
    accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)
    for k_start in range(0, k, block_k):
        a = tl.load(
            a_ptrs,
            mask=(m_offsets[:, None] < m) & (k_start + k_offsets[None, :] < k),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(k_start + k_offsets[:, None] < k) & (n_offsets[None, :] < n),
            other=0.0,
        )
        accumulator += tl.dot(a, b)
        a_ptrs += block_k * stride_ak
        b_ptrs += block_k * stride_bk
    output_offsets = (
        batch_id * stride_ob
        + m_offsets[:, None] * stride_om
        + n_offsets[None, :] * stride_on
    )
    tl.store(
        output_ptr + output_offsets,
        accumulator,
        mask=(m_offsets[:, None] < m) & (n_offsets[None, :] < n),
    )


@triton.jit
def _fused_kda_rms_gated_fp8_quant_kernel(
    x_ptr,
    gate_ptr,
    weight_ptr,
    output_ptr,
    scale_ptr,
    eps,
    tma_aligned_m,
    gate_stride_m,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_rows: tl.constexpr,
    launch_pdl: tl.constexpr,
) -> None:
    row_start = tl.program_id(0) * block_rows
    rows = row_start + tl.arange(0, block_rows)
    cols = tl.arange(0, head_dim)
    offsets = rows[:, None] * head_dim + cols[None, :]
    gate_offsets = (
        (rows // num_heads)[:, None] * gate_stride_m
        + (rows % num_heads)[:, None] * head_dim
        + cols[None, :]
    )
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    x = tl.load(x_ptr + offsets).to(tl.float32)
    gate = tl.load(gate_ptr + gate_offsets).to(tl.float32)
    weight = tl.load(weight_ptr + cols).to(tl.float32)

    variance = tl.sum(x * x, axis=1) / head_dim
    output = x * (1.0 / tl.sqrt(variance[:, None] + eps)) * weight[None, :]
    output *= tl.sigmoid(gate)
    # Preserve the existing BF16 norm-output boundary before quantization.
    output = output.to(tl.bfloat16).to(tl.float32)

    absmax = tl.max(tl.abs(output), axis=1)
    raw_scale = tl.maximum(absmax / 448.0, 1e-10)
    scale_exp = tl.ceil(tl.log2(raw_scale)).to(tl.int32) + 127
    scale = tl.exp2((scale_exp - 127).to(tl.float32))
    quantized = tl.clamp(output / scale[:, None], -448.0, 448.0)
    tl.store(output_ptr + offsets, quantized)

    scale_exp = tl.reshape(scale_exp, (block_rows // 4, 4))
    shifts = tl.arange(0, 4) * 8
    packed_scale = tl.sum(scale_exp << shifts[None, :], axis=1)
    token = row_start // num_heads
    scale_group_start = (row_start % num_heads) // 4
    scale_groups = scale_group_start + tl.arange(0, block_rows // 4)
    tl.store(scale_ptr + scale_groups * tma_aligned_m + token, packed_scale)


def relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denominator = expected.abs().mean().clamp_min(1e-8)
    return float((actual - expected).abs().mean() / denominator)


def load_checkpoint_fp8_weight(
    model_path: Path,
    tensor_prefix: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    from safetensors import safe_open

    index = json.loads(
        (model_path / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    weight_name = f"{tensor_prefix}.weight"
    scale_name = f"{tensor_prefix}.weight_scale"
    shard = index["weight_map"][weight_name]
    if index["weight_map"][scale_name] != shard:
        raise ValueError("Weight and scale must be stored in the same shard")
    with safe_open(model_path / shard, framework="pt", device="cpu") as handle:
        weight = handle.get_tensor(weight_name).cuda()
        scale = handle.get_tensor(scale_name).squeeze(1).squeeze(-1).cuda()
    return weight, scale


def checkpoint_layers_with_tensor(
    model_path: Path,
    tensor_suffix: str,
) -> list[int]:
    index = json.loads(
        (model_path / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    pattern = re.compile(
        rf"language_model\.model\.layers\.(\d+)\.{re.escape(tensor_suffix)}\.weight"
    )
    return sorted(
        int(match.group(1))
        for name in index["weight_map"]
        if (match := pattern.fullmatch(name)) is not None
    )


def fp8_exact_reference(
    a: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    expanded_scale = weight_scale.repeat_interleave(128, dim=0).repeat_interleave(
        128, dim=1
    )[: weight.shape[0], : weight.shape[1]]
    dequantized_weight = weight.float() * expanded_scale
    return torch.mm(a.float(), dequantized_weight.t()).to(torch.bfloat16)


def fp8_candidates(
    shape: Shape,
    m: int,
    include_pipeline: bool,
    triton_sweep: bool,
    checkpoint_path: Path | None,
    checkpoint_layer: int,
    weight_ring: int,
) -> tuple[torch.Tensor, dict[str, Callable]]:
    torch.manual_seed(0)
    if shape.input_row_stride is None:
        a = torch.randn((m, shape.k), device="cuda", dtype=torch.bfloat16)
    else:
        a_base = torch.randn(
            (m, shape.input_row_stride),
            device="cuda",
            dtype=torch.bfloat16,
        )
        a = a_base[:, shape.input_offset : shape.input_offset + shape.k]
    b = torch.randn((shape.n, shape.k), device="cuda", dtype=torch.bfloat16)
    a_dg, a_scale_dg = per_token_group_quant_fp8_packed_for_deepgemm(a, 128)
    if checkpoint_path is None:
        b_quant, b_scale = per_block_cast_to_fp8(
            b,
            [128, 128],
            use_ue8m0=True,
        )
        exact_reference = None
    else:
        if shape.name != "kda_f_b_proj":
            raise ValueError("Checkpoint input is currently supported only for f_b")
        prefix = f"language_model.model.layers.{checkpoint_layer}.self_attn.f_b_proj"
        b_quant, b_scale = load_checkpoint_fp8_weight(checkpoint_path, prefix)
        if b_quant.shape != (shape.n, shape.k):
            raise ValueError(
                f"Unexpected checkpoint weight shape: {tuple(b_quant.shape)}"
            )
        exact_reference = fp8_exact_reference(a, b_quant, b_scale)
    b_dg, b_scale_dg = deepgemm_post_process_fp8_weight_block(
        b_quant.clone(),
        b_scale.clone(),
        (128, 128),
        use_e8m0=True,
    )
    output = torch.empty((m, shape.n), device="cuda", dtype=torch.bfloat16)

    def deepgemm(constraint: tuple[int, int] | None) -> torch.Tensor:
        fp8_gemm_nt(
            (a_dg, a_scale_dg),
            (b_dg, b_scale_dg),
            output,
            block_size_multiple_of=constraint,
        )
        return output

    candidates: dict[str, Callable[[], torch.Tensor]] = {
        "deepgemm_default": lambda: deepgemm(None),
    }
    for constraint in DEEPGEMM_CONSTRAINTS[1:]:
        label = f"deepgemm_bm{constraint[0]}_bn{constraint[1]}"
        candidates[label] = lambda constraint=constraint: deepgemm(constraint)

    a_cutlass, a_scale_cutlass = per_token_group_quant_fp8(
        a,
        128,
        column_major_scales=True,
    )

    def cutlass() -> torch.Tensor:
        return ops.cutlass_scaled_mm(
            a_cutlass,
            b_quant.t(),
            scale_a=a_scale_cutlass,
            scale_b=b_scale.t(),
            out_dtype=torch.bfloat16,
        )

    candidates["cutlass"] = cutlass

    a_triton, a_scale_triton = per_token_group_quant_fp8(a, 128)

    def triton_block_fp8() -> torch.Tensor:
        return w8a8_triton_block_scaled_mm(
            a_triton,
            b_quant,
            a_scale_triton,
            b_scale,
            [128, 128],
            output_dtype=torch.bfloat16,
        )

    candidates["triton"] = triton_block_fp8

    if weight_ring > 1:
        if not include_pipeline:
            raise ValueError("Weight-ring replay requires --include-fp8-pipeline")

        weights = [b_quant]
        scales = [b_scale]
        deepgemm_weights = [b_dg]
        deepgemm_scales = [b_scale_dg]
        checkpoint_layers = None
        checkpoint_start = 0
        if checkpoint_path is not None:
            checkpoint_layers = checkpoint_layers_with_tensor(
                checkpoint_path, "self_attn.f_b_proj"
            )
            if checkpoint_layer not in checkpoint_layers:
                raise ValueError(f"Layer {checkpoint_layer} has no f_b_proj")
            if weight_ring > len(checkpoint_layers):
                raise ValueError(
                    f"weight_ring={weight_ring} exceeds the "
                    f"{len(checkpoint_layers)} checkpoint f_b weights"
                )
            checkpoint_start = checkpoint_layers.index(checkpoint_layer)
        for ring_index in range(1, weight_ring):
            if checkpoint_path is None:
                ring_weight = torch.randn(
                    (shape.n, shape.k), device="cuda", dtype=torch.bfloat16
                ).to(b_quant.dtype)
                exponent = torch.randint(
                    -4,
                    5,
                    (triton.cdiv(shape.n, 128), triton.cdiv(shape.k, 128)),
                    device="cuda",
                )
                ring_scale = torch.pow(2.0, exponent.float())
            else:
                assert checkpoint_layers is not None
                layer = checkpoint_layers[
                    (checkpoint_start + ring_index) % len(checkpoint_layers)
                ]
                prefix = f"language_model.model.layers.{layer}.self_attn.f_b_proj"
                ring_weight, ring_scale = load_checkpoint_fp8_weight(
                    checkpoint_path, prefix
                )
                if ring_weight.shape != (shape.n, shape.k):
                    raise ValueError(
                        f"Unexpected checkpoint weight shape at layer {layer}: "
                        f"{tuple(ring_weight.shape)}"
                    )
            ring_deepgemm_weight, ring_deepgemm_scale = (
                deepgemm_post_process_fp8_weight_block(
                    ring_weight.clone(),
                    ring_scale.clone(),
                    (128, 128),
                    use_e8m0=True,
                )
            )
            weights.append(ring_weight)
            scales.append(ring_scale)
            deepgemm_weights.append(ring_deepgemm_weight)
            deepgemm_scales.append(ring_deepgemm_scale)
        if checkpoint_path is not None:
            exact_reference = fp8_exact_reference(a, weights[-1], scales[-1])

        a_dynamics = [
            torch.empty(a.shape, device=a.device, dtype=b_quant.dtype)
            for _ in range(weight_ring)
        ]
        ring_outputs = [torch.empty_like(output) for _ in range(weight_ring)]

        def ring_deepgemm(constraint: tuple[int, int] | None) -> torch.Tensor:
            result = ring_outputs[-1]
            for ring_weight, ring_scale, a_dynamic, ring_output in zip(
                deepgemm_weights,
                deepgemm_scales,
                a_dynamics,
                ring_outputs,
                strict=True,
            ):
                a_q, a_s = per_token_group_quant_fp8_packed_for_deepgemm(
                    a, 128, out_q=a_dynamic
                )
                fp8_gemm_nt(
                    (a_q, a_s),
                    (ring_weight, ring_scale),
                    ring_output,
                    block_size_multiple_of=constraint,
                )
                result = ring_output
            return result

        def ring_triton(
            config: tuple[int, int, int, int] | None = None,
        ) -> torch.Tensor:
            result = ring_outputs[-1]
            for ring_weight, ring_scale, ring_output in zip(
                weights,
                scales,
                ring_outputs,
                strict=True,
            ):
                a_q, a_s = per_token_group_quant_fp8(a, 128)
                if config is None:
                    result = w8a8_triton_block_scaled_mm(
                        a_q,
                        ring_weight,
                        a_s,
                        ring_scale,
                        [128, 128],
                        output_dtype=torch.bfloat16,
                    )
                else:
                    block_m, block_n, num_warps, num_stages = config
                    grid = (triton.cdiv(m, block_m) * triton.cdiv(shape.n, block_n),)
                    _w8a8_triton_block_scaled_mm[grid](
                        a_q,
                        ring_weight,
                        ring_output,
                        a_s,
                        ring_scale,
                        m,
                        shape.n,
                        shape.k,
                        128,
                        128,
                        a_q.stride(0),
                        a_q.stride(1),
                        ring_weight.stride(1),
                        ring_weight.stride(0),
                        ring_output.stride(0),
                        ring_output.stride(1),
                        a_s.stride(0),
                        a_s.stride(1),
                        ring_scale.stride(1),
                        ring_scale.stride(0),
                        BLOCK_SIZE_M=block_m,
                        BLOCK_SIZE_N=block_n,
                        BLOCK_SIZE_K=128,
                        GROUP_SIZE_M=32,
                        num_warps=num_warps,
                        num_stages=num_stages,
                    )
                    result = ring_output
            return result

        def ring_cutlass() -> torch.Tensor:
            result = output
            for ring_weight, ring_scale in zip(weights, scales, strict=True):
                a_q, a_s = per_token_group_quant_fp8(
                    a,
                    128,
                    column_major_scales=True,
                )
                result = ops.cutlass_scaled_mm(
                    a_q,
                    ring_weight.t(),
                    scale_a=a_s,
                    scale_b=ring_scale.t(),
                    out_dtype=torch.bfloat16,
                )
            return result

        def ring_fused_k128(
            config: tuple[int, int, int, int],
        ) -> torch.Tensor:
            block_m, block_n, num_warps, num_stages = config
            grid = (triton.cdiv(m, block_m), triton.cdiv(shape.n, block_n))
            for ring_weight, ring_scale, ring_output in zip(
                weights,
                scales,
                ring_outputs,
                strict=True,
            ):
                weight_t = ring_weight.t()
                _fused_bf16_fp8_block_mm_k128[grid](
                    a,
                    weight_t,
                    ring_scale,
                    ring_output,
                    m,
                    shape.n,
                    a.stride(0),
                    weight_t.stride(0),
                    weight_t.stride(1),
                    ring_scale.stride(0),
                    ring_output.stride(0),
                    block_m,
                    block_n,
                    num_warps=num_warps,
                    num_stages=num_stages,
                )
            return ring_outputs[-1]

        ring_candidates: dict[str, Callable[[], torch.Tensor]] = {
            "projection_deepgemm_default": lambda: ring_deepgemm(None),
            "projection_cutlass": ring_cutlass,
            "projection_triton": ring_triton,
        }
        for constraint in DEEPGEMM_CONSTRAINTS[1:]:
            label = f"projection_deepgemm_bm{constraint[0]}_bn{constraint[1]}"
            ring_candidates[label] = lambda constraint=constraint: ring_deepgemm(
                constraint
            )
        if triton_sweep:
            for config in TRITON_FP8_CONFIGS:
                block_m, block_n, num_warps, num_stages = config
                label = (
                    f"projection_triton_bm{block_m}_bn{block_n}_"
                    f"w{num_warps}_s{num_stages}"
                )
                ring_candidates[label] = lambda config=config: ring_triton(config)
            if shape.k == 128:
                for block_m in (1, 2, 4, 8, 16, 32, 64):
                    for block_n in (64, 128, 256, 512):
                        for num_warps in (4, 8):
                            config = (block_m, block_n, num_warps, 2)
                            label = (
                                f"projection_fused_k128_bm{block_m}_"
                                f"bn{block_n}_w{num_warps}"
                            )
                            ring_candidates[label] = lambda config=config: (
                                ring_fused_k128(config)
                            )
                for block_m in (4, 8, 16, 32):
                    for num_warps in (2, 4, 8):
                        for num_stages in (1, 2, 3):
                            config = (block_m, 64, num_warps, num_stages)
                            label = (
                                f"projection_fused_k128_tuned_bm{block_m}_"
                                f"bn64_w{num_warps}_s{num_stages}"
                            )
                            ring_candidates[label] = lambda config=config: (
                                ring_fused_k128(config)
                            )
        reference = (
            ring_candidates["projection_deepgemm_default"]().clone()
            if exact_reference is None
            else exact_reference
        )
        return reference, ring_candidates

    if include_pipeline:
        a_dynamic = torch.empty(
            a.shape,
            device=a.device,
            dtype=b_quant.dtype,
        )

        def projection_deepgemm(
            constraint: tuple[int, int] | None,
        ) -> torch.Tensor:
            a_q, a_s = per_token_group_quant_fp8_packed_for_deepgemm(
                a,
                128,
                out_q=a_dynamic,
            )
            fp8_gemm_nt(
                (a_q, a_s),
                (b_dg, b_scale_dg),
                output,
                block_size_multiple_of=constraint,
            )
            return output

        candidates["projection_deepgemm_default"] = lambda: projection_deepgemm(None)
        for constraint in DEEPGEMM_CONSTRAINTS[1:]:
            label = f"projection_deepgemm_bm{constraint[0]}_bn{constraint[1]}"
            candidates[label] = lambda constraint=constraint: projection_deepgemm(
                constraint
            )

        def projection_cutlass() -> torch.Tensor:
            a_q, a_s = per_token_group_quant_fp8(
                a,
                128,
                column_major_scales=True,
            )
            return ops.cutlass_scaled_mm(
                a_q,
                b_quant.t(),
                scale_a=a_s,
                scale_b=b_scale.t(),
                out_dtype=torch.bfloat16,
            )

        candidates["projection_cutlass"] = projection_cutlass

        def projection_triton() -> torch.Tensor:
            a_q, a_s = per_token_group_quant_fp8(a, 128)
            return w8a8_triton_block_scaled_mm(
                a_q,
                b_quant,
                a_s,
                b_scale,
                [128, 128],
                output_dtype=torch.bfloat16,
            )

        candidates["projection_triton"] = projection_triton

        if shape.k == 128:

            def make_fused_k128(
                block_m: int,
                block_n: int,
                num_warps: int,
                num_stages: int,
            ) -> Callable[[], torch.Tensor]:
                def run() -> torch.Tensor:
                    grid = (triton.cdiv(m, block_m), triton.cdiv(shape.n, block_n))
                    _fused_bf16_fp8_block_mm_k128[grid](
                        a,
                        b_quant.t(),
                        b_scale,
                        output,
                        m,
                        shape.n,
                        a.stride(0),
                        b_quant.t().stride(0),
                        b_quant.t().stride(1),
                        b_scale.stride(0),
                        output.stride(0),
                        block_m,
                        block_n,
                        num_warps=num_warps,
                        num_stages=num_stages,
                    )
                    return output

                return run

            for block_m in (1, 2, 4, 8, 16, 32, 64):
                for block_n in (64, 128, 256, 512):
                    for num_warps in (4, 8):
                        label = (
                            f"projection_fused_k128_bm{block_m}_"
                            f"bn{block_n}_w{num_warps}"
                        )
                        candidates[label] = make_fused_k128(
                            block_m,
                            block_n,
                            num_warps,
                            2,
                        )
            for block_m in (4, 8, 16, 32):
                for num_warps in (2, 4, 8):
                    for num_stages in (1, 2, 3):
                        label = (
                            f"projection_fused_k128_tuned_bm{block_m}_"
                            f"bn64_w{num_warps}_s{num_stages}"
                        )
                        candidates[label] = make_fused_k128(
                            block_m,
                            64,
                            num_warps,
                            num_stages,
                        )

        if triton_sweep:

            def make_projection_triton(
                block_m: int,
                block_n: int,
                num_warps: int,
                num_stages: int,
            ) -> Callable[[], torch.Tensor]:
                def run() -> torch.Tensor:
                    a_q, a_s = per_token_group_quant_fp8(a, 128)
                    grid = (triton.cdiv(m, block_m) * triton.cdiv(shape.n, block_n),)
                    _w8a8_triton_block_scaled_mm[grid](
                        a_q,
                        b_quant,
                        output,
                        a_s,
                        b_scale,
                        m,
                        shape.n,
                        shape.k,
                        128,
                        128,
                        a_q.stride(0),
                        a_q.stride(1),
                        b_quant.stride(1),
                        b_quant.stride(0),
                        output.stride(0),
                        output.stride(1),
                        a_s.stride(0),
                        a_s.stride(1),
                        b_scale.stride(1),
                        b_scale.stride(0),
                        BLOCK_SIZE_M=block_m,
                        BLOCK_SIZE_N=block_n,
                        BLOCK_SIZE_K=128,
                        GROUP_SIZE_M=32,
                        num_warps=num_warps,
                        num_stages=num_stages,
                    )
                    return output

                return run

            for block_m, block_n, num_warps, num_stages in TRITON_FP8_CONFIGS:
                label = (
                    f"projection_triton_bm{block_m}_bn{block_n}_"
                    f"w{num_warps}_s{num_stages}"
                )
                candidates[label] = make_projection_triton(
                    block_m,
                    block_n,
                    num_warps,
                    num_stages,
                )

        candidates["projection_flashinfer"] = lambda: flashinfer_fp8_blockscale_gemm(
            input=a,
            weight=b_dg,
            weight_scale=b_scale_dg,
            out_dtype=torch.bfloat16,
        )
    reference = deepgemm(None).clone() if exact_reference is None else exact_reference
    return reference, candidates


def bf16_skinny_configs(shape: Shape, m: int) -> Iterator[tuple[str, SkinnyGemmConfig]]:
    for block_size in range(32, 257, 32):
        for outputs_per_block in (1, 2, 3, 4, 6, 7, 8):
            for k_unroll in (1, 2, 4):
                for vector_width in (4, 8):
                    config = SkinnyGemmConfig(
                        m,
                        block_size,
                        outputs_per_block,
                        k_unroll,
                        vector_width,
                    )
                    if shape.k % (block_size * config.vector_width):
                        continue
                    if shape.n % outputs_per_block:
                        continue
                    label = (
                        f"cute_skinny_bs{block_size}_opb{outputs_per_block}_"
                        f"ku{k_unroll}_vw{vector_width}"
                    )
                    yield label, config

    # With a compile-time K, the kernel uses a fully unrolled loop and ignores
    # k_unroll. Search the remaining dimensions once rather than compiling
    # duplicate kernels that differ only in an inactive field.
    for block_size in range(32, 257, 32):
        for outputs_per_block in (1, 2, 3, 4, 6, 7, 8):
            for vector_width in (4, 8):
                config = SkinnyGemmConfig(
                    m,
                    block_size,
                    outputs_per_block,
                    k_unroll=1,
                    vector_width=vector_width,
                    static_k=shape.k,
                )
                tile_k = block_size * config.vector_width
                if shape.k % tile_k or shape.k < 2 * tile_k:
                    continue
                if shape.n % outputs_per_block:
                    continue
                label = (
                    f"cute_static_k_bs{block_size}_opb{outputs_per_block}_"
                    f"vw{vector_width}"
                )
                yield label, config

    register_tuning_configs = {
        ("draft_context_proj", 1): (160, 1, 8),
        ("draft_context_proj", 2): (128, 4, 8),
        ("draft_context_proj", 3): (128, 4, 8),
        ("draft_context_proj", 4): (128, 4, 4),
        ("draft_context_proj", 5): (128, 4, 4),
        ("draft_context_kv_proj", 1): (256, 2, 4),
        ("draft_context_kv_proj", 2): (256, 2, 4),
        ("draft_context_kv_proj", 3): (128, 8, 8),
        ("draft_context_kv_proj", 4): (128, 8, 8),
        ("draft_context_kv_proj", 5): (64, 4, 8),
        ("draft_dense_down", 1): (64, 1, 4),
        ("draft_dense_down", 2): (64, 4, 8),
        ("draft_dense_down", 3): (64, 8, 8),
        ("draft_dense_down", 4): (32, 2, 8),
        ("draft_mla_q_b_proj", 1): (32, 4, 8),
        ("draft_mla_q_b_proj", 2): (32, 8, 8),
        ("draft_mla_q_b_proj", 3): (32, 8, 8),
        ("draft_mla_o_proj", 1): (128, 1, 4),
        ("draft_mla_o_proj", 2): (64, 8, 8),
        ("draft_mla_o_proj", 3): (64, 8, 8),
        ("draft_mla_o_proj", 4): (64, 8, 8),
        ("dense_gate_up", 1): (32, 1, 8),
        ("dense_gate_up", 2): (64, 4, 4),
        ("dense_gate_up", 3): (64, 4, 4),
        ("dense_gate_up", 4): (64, 4, 4),
        ("dense_down", 1): (192, 1, 8),
        ("dense_down", 2): (128, 4, 8),
        ("dense_down", 3): (128, 4, 4),
        ("dense_down", 4): (128, 4, 4),
        ("routed_latent_down", 1): (128, 1, 4),
        ("routed_latent_down", 2): (64, 4, 8),
        ("routed_latent_down", 3): (64, 4, 8),
        ("routed_latent_down", 4): (64, 4, 8),
        ("routed_latent_down", 5): (64, 4, 8),
        ("routed_latent_up_add", 1): (64, 2, 4),
        ("routed_latent_up_add", 2): (64, 8, 4),
        ("routed_latent_up_add", 3): (32, 4, 8),
        ("routed_latent_up_add", 4): (32, 4, 8),
    }
    tuning = register_tuning_configs.get((shape.name, m))
    if tuning is not None:
        block_size, outputs_per_block, vector_width = tuning
        for max_registers in (48, 64, 80, 96, 128, 160, 192, 255):
            config = SkinnyGemmConfig(
                m,
                block_size,
                outputs_per_block,
                k_unroll=1,
                vector_width=vector_width,
                static_k=shape.k,
                max_registers=max_registers,
            )
            yield f"cute_static_reg{max_registers}", config


def bf16_candidates(
    shape: Shape,
    m: int,
    router_sweep: bool,
    deepgemm_sweep: bool,
    skinny_sweep: bool,
    weight_ring: int,
) -> tuple[torch.Tensor, dict[str, Callable]]:
    torch.manual_seed(0)
    a = torch.randn((m, shape.k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((shape.n, shape.k), device="cuda", dtype=torch.bfloat16)
    output_dtype = torch.float32 if shape.fp32_output else torch.bfloat16
    residual = torch.randn((m, shape.n), device="cuda", dtype=output_dtype)
    torch_output = torch.empty((m, shape.n), device="cuda", dtype=output_dtype)

    def torch_linear() -> torch.Tensor:
        if shape.residual:
            return torch.addmm(residual, a, b.t(), out=torch_output)
        return torch.mm(a, b.t(), out_dtype=output_dtype)

    if weight_ring > 1:
        weights = [b]
        weights.extend(
            torch.randn((shape.n, shape.k), device="cuda", dtype=torch.bfloat16)
            for _ in range(1, weight_ring)
        )

        def torch_ring() -> torch.Tensor:
            result = torch_output
            for ring_weight in weights:
                if shape.residual:
                    torch.addmm(residual, a, ring_weight.t(), out=torch_output)
                elif shape.fp32_output:
                    result = torch.mm(a, ring_weight.t(), out_dtype=torch.float32)
                else:
                    torch.mm(a, ring_weight.t(), out=torch_output)
            return result

        ring_candidates: dict[str, Callable[[], torch.Tensor]] = {"torch": torch_ring}

        def ring_kimi_selected() -> torch.Tensor:
            result = torch_output
            for ring_weight in weights:
                selected = try_low_latency_gemm(
                    a,
                    ring_weight,
                    residual if shape.residual else None,
                )
                if selected is not None:
                    result = selected
                elif shape.residual:
                    torch.addmm(residual, a, ring_weight.t(), out=torch_output)
                    result = torch_output
                elif shape.fp32_output:
                    result = torch.mm(a, ring_weight.t(), out_dtype=torch.float32)
                else:
                    torch.mm(a, ring_weight.t(), out=torch_output)
                    result = torch_output
            return result

        ring_candidates["kimi_selected"] = ring_kimi_selected
        projection_spec = KIMI_K3_PROJECTIONS.get((shape.n, shape.k))
        selected_config = (
            projection_spec.cute_config(m) if projection_spec is not None else None
        )
        if selected_config is not None and not shape.fp32_output:
            static_config = replace(selected_config, static_k=shape.k)
            static_outputs = [torch.empty_like(torch_output) for _ in weights]
            static_cache_key = (torch.bfloat16, static_config, shape.residual)

            def ring_static_k() -> torch.Tensor:
                if static_cache_key not in shape_dynamic_skinny_gemm._compiled:
                    shape_dynamic_skinny_gemm._compile(
                        torch.bfloat16,
                        static_config,
                        shape.residual,
                    )
                compiled = shape_dynamic_skinny_gemm._compiled[static_cache_key]
                for ring_weight, ring_output in zip(
                    weights, static_outputs, strict=True
                ):
                    residual_arg = residual if shape.residual else ring_output
                    compiled(
                        a,
                        ring_weight,
                        residual_arg,
                        ring_output,
                        shape_dynamic_skinny_gemm._stream(),
                    )
                return static_outputs[-1]

            ring_candidates["kimi_selected_static_k"] = ring_static_k
        if projection_spec is not None and m in projection_spec.dsv3_tokens:
            dsv3_outputs = [torch.empty_like(torch_output) for _ in weights]

            def ring_dsv3() -> torch.Tensor:
                for ring_weight, ring_output in zip(weights, dsv3_outputs, strict=True):
                    ops.dsv3_fused_a_gemm(
                        ring_output,
                        a,
                        ring_weight.t(),
                        enable_pdl=True,
                    )
                return dsv3_outputs[-1]

            ring_candidates["dsv3_fused_a"] = ring_dsv3
        dg = _import_deep_gemm()
        if dg is not None:
            dg_outputs = [torch.empty_like(torch_output) for _ in weights]

            def ring_deepgemm(
                constraint: tuple[int, int] | None,
            ) -> torch.Tensor:
                context = (
                    _block_size_multiple_scope(constraint)
                    if constraint is not None
                    else contextlib.nullcontext()
                )
                with context:
                    for ring_weight, dg_output in zip(weights, dg_outputs, strict=True):
                        dg.bf16_gemm_nt(
                            a,
                            ring_weight,
                            dg_output,
                            c=residual if shape.residual else None,
                        )
                return dg_output

            def ring_cublaslt() -> torch.Tensor:
                dg_output = dg_outputs[-1]
                for ring_weight, dg_output in zip(weights, dg_outputs, strict=True):
                    dg.cublaslt_gemm_nt(
                        a,
                        ring_weight,
                        dg_output,
                        c=residual if shape.residual else None,
                    )
                return dg_output

            ring_candidates.update(
                {
                    "deepgemm_bf16": lambda: ring_deepgemm(None),
                    "deepgemm_cublaslt": ring_cublaslt,
                }
            )
            if deepgemm_sweep:
                for constraint in DEEPGEMM_CONSTRAINTS[1:]:
                    label = f"deepgemm_bm{constraint[0]}_bn{constraint[1]}"
                    ring_candidates[label] = lambda constraint=constraint: (
                        ring_deepgemm(constraint)
                    )
        if shape.fp32_output:
            cute_outputs = [torch.empty_like(torch_output) for _ in weights]

            def ring_cute() -> torch.Tensor:
                result = torch_output
                for ring_weight in weights:
                    result = ll_bf16.ll_bf16_gemm(
                        a, ring_weight, output_dtype=torch.float32
                    )
                return result

            ring_candidates["cute_ll_bf16"] = ring_cute
            if router_sweep:

                def make_ring_splitk(split_k: int, stages: int) -> Callable:
                    key = ll_bf16.LLBf16Gemm.CompileKey(
                        backend="splitk",
                        split_k=split_k,
                        num_stages=stages,
                    )

                    def run() -> torch.Tensor:
                        cache_key = (split_k, stages)
                        kernel = ll_bf16.ll_bf16_gemm_kernel
                        if cache_key not in kernel._splitk_cache:
                            kernel.compile(key)
                        compiled = kernel._splitk_cache[cache_key]
                        for ring_weight, ring_output in zip(
                            weights, cute_outputs, strict=True
                        ):
                            compiled(
                                a,
                                ring_weight,
                                ring_output,
                                ll_bf16._stream(),
                                1.0,
                            )
                        return cute_outputs[-1]

                    return run

                for split_k in range(2, 13):
                    for stages in range(1, 7):
                        label = f"cute_splitk{split_k}_stages{stages}"
                        ring_candidates[label] = make_ring_splitk(split_k, stages)
        if skinny_sweep and not shape.fp32_output:
            skinny_outputs = [torch.empty_like(torch_output) for _ in weights]

            def make_ring_skinny(config: SkinnyGemmConfig) -> Callable:
                cache_key = (torch.bfloat16, config, shape.residual)

                def run() -> torch.Tensor:
                    if cache_key not in shape_dynamic_skinny_gemm._compiled:
                        shape_dynamic_skinny_gemm._compile(
                            torch.bfloat16,
                            config,
                            shape.residual,
                        )
                    compiled = shape_dynamic_skinny_gemm._compiled[cache_key]
                    for ring_weight, ring_output in zip(
                        weights, skinny_outputs, strict=True
                    ):
                        residual_arg = residual if shape.residual else ring_output
                        compiled(
                            a,
                            ring_weight,
                            residual_arg,
                            ring_output,
                            shape_dynamic_skinny_gemm._stream(),
                        )
                    return skinny_outputs[-1]

                return run

            for label, config in bf16_skinny_configs(shape, m):
                ring_candidates[label] = make_ring_skinny(config)
        reference = torch_ring().clone()
        return reference, ring_candidates

    def kimi_selected() -> torch.Tensor:
        selected = try_low_latency_gemm(a, b, residual if shape.residual else None)
        return torch_linear() if selected is None else selected

    candidates: dict[str, Callable[[], torch.Tensor]] = {
        "torch": torch_linear,
        "kimi_selected": kimi_selected,
    }
    dg = _import_deep_gemm()
    if dg is not None:
        dg_output = torch.empty((m, shape.n), device="cuda", dtype=output_dtype)

        def deepgemm_bf16() -> torch.Tensor:
            dg.bf16_gemm_nt(
                a,
                b,
                dg_output,
                c=residual if shape.residual else None,
            )
            return dg_output

        candidates["deepgemm_bf16"] = deepgemm_bf16
        if deepgemm_sweep:

            def make_deepgemm(
                constraint: tuple[int, int],
            ) -> Callable[[], torch.Tensor]:
                def run() -> torch.Tensor:
                    with _block_size_multiple_scope(constraint):
                        dg.bf16_gemm_nt(
                            a,
                            b,
                            dg_output,
                            c=residual if shape.residual else None,
                        )
                    return dg_output

                return run

            for constraint in DEEPGEMM_CONSTRAINTS[1:]:
                label = f"deepgemm_bm{constraint[0]}_bn{constraint[1]}"
                candidates[label] = make_deepgemm(constraint)

        cublaslt_output = torch.empty((m, shape.n), device="cuda", dtype=output_dtype)

        def cublaslt() -> torch.Tensor:
            dg.cublaslt_gemm_nt(
                a,
                b,
                cublaslt_output,
                c=residual if shape.residual else None,
            )
            return cublaslt_output

        candidates["deepgemm_cublaslt"] = cublaslt

        if not shape.residual and not shape.fp32_output:
            kimi_output = torch.empty((m, shape.n), device="cuda", dtype=output_dtype)

            def kimi_deepgemm_bm64() -> torch.Tensor:
                _launch_kimi_k3_bf16_gemm_nt(a, b, kimi_output, 64, 64)
                return kimi_output

            candidates["kimi_deepgemm_bm64"] = kimi_deepgemm_bm64

    if shape.fp32_output:
        candidates["cute_ll_bf16"] = lambda: ll_bf16.ll_bf16_gemm(
            a, b, output_dtype=torch.float32
        )
        if router_sweep:
            output = torch.empty((m, shape.n), device="cuda", dtype=torch.float32)

            def make_splitk(split_k: int, stages: int) -> Callable:
                key = ll_bf16.LLBf16Gemm.CompileKey(
                    backend="splitk",
                    split_k=split_k,
                    num_stages=stages,
                )

                def run() -> torch.Tensor:
                    cache_key = (split_k, stages)
                    kernel = ll_bf16.ll_bf16_gemm_kernel
                    if cache_key not in kernel._splitk_cache:
                        kernel.compile(key)
                    kernel._splitk_cache[cache_key](
                        a,
                        b,
                        output,
                        ll_bf16._stream(),
                        1.0,
                    )
                    return output

                return run

            for split_k in range(2, 13):
                for stages in range(1, 7):
                    candidates[f"cute_splitk{split_k}_stages{stages}"] = make_splitk(
                        split_k, stages
                    )

    projection_spec = KIMI_K3_PROJECTIONS.get((shape.n, shape.k))
    selected_config = (
        projection_spec.cute_config(m) if projection_spec is not None else None
    )
    if selected_config is not None and not shape.fp32_output:
        static_config = replace(selected_config, static_k=shape.k)
        candidates["kimi_selected_static_k"] = lambda: shape_dynamic_skinny_gemm(
            a,
            b,
            static_config,
            residual if shape.residual else None,
        )
    if projection_spec is not None and m in projection_spec.dsv3_tokens:
        output = torch.empty((m, shape.n), device="cuda", dtype=torch.bfloat16)

        def dsv3() -> torch.Tensor:
            ops.dsv3_fused_a_gemm(output, a, b.t(), enable_pdl=True)
            if shape.residual:
                output.add_(residual)
            return output

        candidates["dsv3_fused_a"] = dsv3

    if skinny_sweep and not shape.fp32_output:
        skinny_output = torch.empty((m, shape.n), device="cuda", dtype=torch.bfloat16)

        def make_skinny(config: SkinnyGemmConfig) -> Callable[[], torch.Tensor]:
            cache_key = (torch.bfloat16, config, shape.residual)

            def run() -> torch.Tensor:
                if cache_key not in shape_dynamic_skinny_gemm._compiled:
                    shape_dynamic_skinny_gemm._compile(
                        torch.bfloat16,
                        config,
                        shape.residual,
                    )
                residual_arg = residual if shape.residual else skinny_output
                shape_dynamic_skinny_gemm._compiled[cache_key](
                    a,
                    b,
                    residual_arg,
                    skinny_output,
                    shape_dynamic_skinny_gemm._stream(),
                )
                return skinny_output

            return run

        for label, config in bf16_skinny_configs(shape, m):
            candidates[label] = make_skinny(config)
    return torch_linear().clone(), candidates


def bmm_candidates(
    shape: Shape,
    m: int,
    triton_sweep: bool,
) -> tuple[torch.Tensor, dict[str, Callable]]:
    torch.manual_seed(0)
    a = torch.randn(
        (shape.calls, shape.batch, m, shape.k),
        device="cuda",
        dtype=torch.bfloat16,
    )
    b = torch.randn(
        (shape.calls, shape.batch, shape.k, shape.n),
        device="cuda",
        dtype=torch.bfloat16,
    )
    output = torch.empty(
        (shape.calls, shape.batch, m, shape.n),
        device="cuda",
        dtype=torch.bfloat16,
    )

    def torch_bmm() -> torch.Tensor:
        for layer in range(shape.calls):
            torch.bmm(a[layer], b[layer], out=output[layer])
        return output

    candidates: dict[str, Callable[[], torch.Tensor]] = {"torch_bmm": torch_bmm}

    def make_triton(
        block_m: int,
        block_n: int,
        block_k: int,
        num_warps: int,
        num_stages: int,
    ) -> Callable[[], torch.Tensor]:
        def run() -> torch.Tensor:
            grid = (
                triton.cdiv(m, block_m) * triton.cdiv(shape.n, block_n),
                shape.batch,
            )
            for layer in range(shape.calls):
                _bf16_bmm_kernel[grid](
                    a[layer],
                    b[layer],
                    output[layer],
                    m,
                    shape.n,
                    shape.k,
                    a.stride(1),
                    a.stride(2),
                    a.stride(3),
                    b.stride(1),
                    b.stride(2),
                    b.stride(3),
                    output.stride(1),
                    output.stride(2),
                    output.stride(3),
                    block_m,
                    block_n,
                    block_k,
                    num_warps=num_warps,
                    num_stages=num_stages,
                )
            return output

        return run

    configs = (
        (16, 128, 64, 4, 2),
        (32, 128, 64, 4, 2),
    )
    if triton_sweep:
        configs = tuple(
            (block_m, block_n, block_k, num_warps, num_stages)
            for block_m in (16, 32)
            for block_n in (64, 128, 256)
            for block_k in (64, 128)
            for num_warps in (4, 8)
            for num_stages in (1, 2, 3, 4)
            if shape.n % block_n == 0 and shape.k % block_k == 0
        )
    for block_m, block_n, block_k, num_warps, num_stages in configs:
        label = f"triton_bm{block_m}_bn{block_n}_bk{block_k}_w{num_warps}_s{num_stages}"
        candidates[label] = make_triton(
            block_m,
            block_n,
            block_k,
            num_warps,
            num_stages,
        )
    return torch_bmm().clone(), candidates


def run_shape(
    shape: Shape,
    m: int,
    rep: int,
    router_sweep: bool,
    bf16_deepgemm_sweep: bool,
    skinny_sweep: bool,
    include_fp8_pipeline: bool,
    triton_sweep: bool,
    selected_candidates: set[str],
    candidate_prefixes: tuple[str, ...],
    trials: int,
    precondition_rep: int,
    checkpoint_path: Path | None,
    checkpoint_layer: int,
    bmm_triton_sweep: bool,
    weight_ring: int,
) -> dict:
    if shape.precision == "fp8":
        reference, candidates = fp8_candidates(
            shape,
            m,
            include_fp8_pipeline,
            triton_sweep,
            checkpoint_path,
            checkpoint_layer,
            weight_ring,
        )
    elif shape.precision == "bf16":
        reference, candidates = bf16_candidates(
            shape,
            m,
            router_sweep,
            bf16_deepgemm_sweep,
            skinny_sweep,
            weight_ring,
        )
    else:
        reference, candidates = bmm_candidates(shape, m, bmm_triton_sweep)
    reference_cpu = reference.cpu()
    if selected_candidates or candidate_prefixes:
        candidates = {
            name: candidate
            for name, candidate in candidates.items()
            if name in selected_candidates
            or any(name.startswith(prefix) for prefix in candidate_prefixes)
        }
    results_by_name: dict[str, dict] = {}
    candidate_items = list(candidates.items())
    if precondition_rep and candidate_items:
        benchmark(candidate_items[0][1], precondition_rep)
    for trial in range(trials):
        offset = trial % max(len(candidate_items), 1)
        trial_items = candidate_items[offset:] + candidate_items[:offset]
        for name, candidate in trial_items:
            result = results_by_name.setdefault(
                name,
                {
                    "backend": name,
                    "trial_p50_us": [],
                    "trial_p10_us": [],
                    "trial_p90_us": [],
                },
            )
            if "error" in result:
                continue
            try:
                actual = candidate()
                torch.cuda.synchronize()
                result["relative_error"] = relative_error(actual.cpu(), reference_cpu)
                p50_us, p10_us, p90_us = benchmark(candidate, rep)
                normalization = weight_ring
                p50_us /= normalization
                p10_us /= normalization
                p90_us /= normalization
                result["trial_p50_us"].append(p50_us)
                result["trial_p10_us"].append(p10_us)
                result["trial_p90_us"].append(p90_us)
                print(
                    f"{shape.name:28s} M={m:3d} trial={trial + 1:2d} "
                    f"{name:28s} p50={p50_us:9.3f}us "
                    f"p10={p10_us:9.3f}us p90={p90_us:9.3f}us "
                    f"error={result['relative_error']:.3e}",
                    flush=True,
                )
            except Exception as exc:
                result.clear()
                result.update({"backend": name, "error": repr(exc)})
                print(
                    f"{shape.name:28s} M={m:3d} {name:28s} FAILED: {exc!r}",
                    flush=True,
                )

    results = list(results_by_name.values())
    for result in results:
        if not result.get("trial_p50_us"):
            continue
        result["p50_us"] = statistics.median(result["trial_p50_us"])
        result["p10_us"] = statistics.median(result["trial_p10_us"])
        result["p90_us"] = statistics.median(result["trial_p90_us"])
        result["trial_p50_min_us"] = min(result["trial_p50_us"])
        result["trial_p50_max_us"] = max(result["trial_p50_us"])
    valid = [result for result in results if "p50_us" in result]
    if valid:
        winner = min(valid, key=lambda result: result["p50_us"])
        baseline = next(
            (
                result
                for result in valid
                if result["backend"] in ("deepgemm_default", "torch")
            ),
            None,
        )
        if baseline is not None:
            winner["speedup_vs_baseline"] = baseline["p50_us"] / winner["p50_us"]
    return {
        "shape": asdict(shape),
        "m": m,
        "weight_ring": weight_ring,
        "results": results,
    }


def benchmark_mla_shared_quantization(
    m: int,
    rep: int,
    trials: int,
    precondition_rep: int,
    weight_ring: int,
) -> dict:
    """Compare two MLA projections with separate versus shared input quantization."""
    if weight_ring < 1:
        raise ValueError("weight_ring must be positive")
    torch.manual_seed(0)
    hidden_size = 7168
    a = torch.randn((m, hidden_size), device="cuda", dtype=torch.bfloat16)
    a_dynamic = torch.empty_like(a, dtype=current_platform.fp8_dtype())
    a_dynamic_gate = torch.empty_like(a_dynamic)
    output_qkv = torch.empty((m, 2176), device="cuda", dtype=torch.bfloat16)
    output_gate = torch.empty((m, 12288), device="cuda", dtype=torch.bfloat16)
    weight_pairs = []
    for qkv_n, gate_n in [(2176, 12288)] * weight_ring:
        pair = []
        for n in (qkv_n, gate_n):
            weight = torch.randn((n, hidden_size), device="cuda", dtype=torch.bfloat16)
            weight_q, weight_s = per_block_cast_to_fp8(
                weight, [128, 128], use_ue8m0=True
            )
            pair.append(
                deepgemm_post_process_fp8_weight_block(
                    weight_q,
                    weight_s,
                    (128, 128),
                    use_e8m0=True,
                )
            )
        weight_pairs.append(pair)

    def run(
        shared: bool,
        qkv_constraint: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        for qkv, gate in weight_pairs:
            a_q, a_s = per_token_group_quant_fp8_packed_for_deepgemm(
                a, 128, out_q=a_dynamic
            )
            fp8_gemm_nt(
                (a_q, a_s),
                qkv,
                output_qkv,
                block_size_multiple_of=qkv_constraint,
            )
            if not shared:
                a_q, a_s = per_token_group_quant_fp8_packed_for_deepgemm(
                    a, 128, out_q=a_dynamic
                )
            fp8_gemm_nt((a_q, a_s), gate, output_gate)
        return output_gate

    aux_stream = torch.cuda.Stream()
    start_event = torch.cuda.Event()
    done_event = torch.cuda.Event()

    def launch_gate(gate_input, gate_weight) -> None:
        with torch.cuda.stream(aux_stream):
            start_event.wait()
            if gate_input is None:
                gate_input = per_token_group_quant_fp8_packed_for_deepgemm(
                    a, 128, out_q=a_dynamic_gate
                )
            fp8_gemm_nt(gate_input, gate_weight, output_gate)
            done_event.record()

    def run_parallel(
        shared: bool,
        qkv_constraint: tuple[int, int] | None = None,
        launch_aux_first: bool = False,
    ) -> torch.Tensor:
        for qkv, gate in weight_pairs:
            shared_input = None
            if shared:
                shared_input = per_token_group_quant_fp8_packed_for_deepgemm(
                    a, 128, out_q=a_dynamic
                )
            start_event.record()

            if launch_aux_first:
                launch_gate(shared_input, gate)
            qkv_input = shared_input
            if qkv_input is None:
                qkv_input = per_token_group_quant_fp8_packed_for_deepgemm(
                    a, 128, out_q=a_dynamic
                )
            fp8_gemm_nt(
                qkv_input,
                qkv,
                output_qkv,
                block_size_multiple_of=qkv_constraint,
            )
            if not launch_aux_first:
                launch_gate(shared_input, gate)
            done_event.wait()
        return output_gate

    candidates = {
        "separate_quantization": lambda: run(False),
        "shared_quantization": lambda: run(True),
        "separate_quantization_parallel": lambda: run_parallel(False),
        "shared_quantization_parallel": lambda: run_parallel(True),
        "shared_quantization_qkv_bm256_bn64": lambda: run(True, (256, 64)),
        "shared_quantization_parallel_qkv_bm256_bn64": lambda: run_parallel(
            True, (256, 64)
        ),
        "shared_quantization_aux_first": lambda: run_parallel(
            True, launch_aux_first=True
        ),
        "shared_quantization_aux_first_qkv_bm256_bn64": lambda: run_parallel(
            True, (256, 64), launch_aux_first=True
        ),
    }
    reference = candidates["separate_quantization"]().clone()
    if precondition_rep:
        benchmark(candidates["separate_quantization"], precondition_rep)
    results_by_name = {
        name: {
            "backend": name,
            "trial_p50_us": [],
            "trial_p10_us": [],
            "trial_p90_us": [],
            "relative_error": relative_error(candidate(), reference),
        }
        for name, candidate in candidates.items()
    }
    candidate_items = list(candidates.items())
    for trial in range(trials):
        offset = trial % len(candidate_items)
        for name, candidate in candidate_items[offset:] + candidate_items[:offset]:
            p50_us, p10_us, p90_us = benchmark(candidate, rep)
            result = results_by_name[name]
            result["trial_p50_us"].append(p50_us / weight_ring)
            result["trial_p10_us"].append(p10_us / weight_ring)
            result["trial_p90_us"].append(p90_us / weight_ring)
    results = list(results_by_name.values())
    for result in results:
        result["p50_us"] = statistics.median(result["trial_p50_us"])
        result["p10_us"] = statistics.median(result["trial_p10_us"])
        result["p90_us"] = statistics.median(result["trial_p90_us"])
    return {
        "name": "mla_fused_qkv_a_plus_output_gate",
        "m": m,
        "weight_ring": weight_ring,
        "results": results,
    }


def benchmark_kda_output_fusion(
    m: int,
    seed: int,
    rep: int,
    trials: int,
    precondition_rep: int,
    sweep: bool,
    selected_candidates: set[str],
) -> dict:
    """Compare KDA gated RMSNorm plus FP8 quantization with a fused kernel."""
    torch.manual_seed(seed)
    num_heads = 96
    head_dim = 128
    source_x = torch.randn(
        (m * num_heads, head_dim), device="cuda", dtype=torch.bfloat16
    )
    baseline_input = torch.empty_like(source_x)
    fused_input = torch.empty_like(source_x)
    packed_width = 49408
    gate_base = torch.randn((m, packed_width), device="cuda", dtype=torch.bfloat16)
    gate = gate_base[:, 3 * num_heads * head_dim : 4 * num_heads * head_dim]
    gate = gate.view(m, num_heads, head_dim)
    weight = torch.randn((head_dim,), device="cuda", dtype=torch.bfloat16)
    tma_aligned_m = triton.cdiv(m, 4) * 4

    def baseline() -> tuple[torch.Tensor, torch.Tensor]:
        baseline_input.copy_(source_x)
        normalized = rms_norm_gated(
            baseline_input,
            gate,
            weight,
            None,
            activation="sigmoid",
            eps=1e-5,
        ).view(m, num_heads * head_dim)
        return per_token_group_quant_fp8_packed_for_deepgemm(normalized, 128)

    def make_fused(
        block_rows: int,
        num_warps: int,
    ) -> Callable[[], tuple[torch.Tensor, torch.Tensor]]:
        output = torch.empty(
            (m, num_heads * head_dim),
            device=source_x.device,
            dtype=current_platform.fp8_dtype(),
        )
        scale = torch.empty_strided(
            (m, num_heads // 4),
            (1, tma_aligned_m),
            device=source_x.device,
            dtype=torch.int32,
        )

        def fused() -> tuple[torch.Tensor, torch.Tensor]:
            fused_input.copy_(source_x)
            _fused_kda_rms_gated_fp8_quant_kernel[(m * num_heads // block_rows,)](
                fused_input,
                gate,
                weight,
                output,
                scale,
                1e-5,
                tma_aligned_m,
                gate.stride(0),
                num_heads=num_heads,
                head_dim=head_dim,
                block_rows=block_rows,
                launch_pdl=current_platform.is_arch_support_pdl(),
                num_warps=num_warps,
                num_stages=1,
            )
            return output, scale

        return fused

    fused = make_fused(16, 8)

    reference_q, reference_scale = baseline()
    fused_q, fused_scale = fused()
    torch.cuda.synchronize()
    scale_match = torch.equal(fused_scale, reference_scale)
    quantized_match = torch.equal(fused_q, reference_q)
    quantized_error = relative_error(fused_q.float(), reference_q.float())

    fused_candidates = {"fused_rms_gated_quant_br16_w8": fused}
    if sweep:
        for block_rows in (4, 8, 16, 32):
            for num_warps in (2, 4, 8):
                if (block_rows, num_warps) == (16, 8):
                    continue
                candidate = make_fused(block_rows, num_warps)
                fused_candidates[
                    f"fused_rms_gated_quant_br{block_rows}_w{num_warps}"
                ] = candidate
    if selected_candidates:
        fused_candidates = {
            name: candidate
            for name, candidate in fused_candidates.items()
            if name in selected_candidates
        }
    invalid_candidates = []
    for name, candidate in tuple(fused_candidates.items()):
        candidate_q, candidate_scale = candidate()
        if not torch.equal(candidate_q, reference_q) or not torch.equal(
            candidate_scale, reference_scale
        ):
            invalid_candidates.append(name)
            del fused_candidates[name]
    candidates = {"rms_gated_then_quant": lambda: baseline()[0]}
    candidates.update(
        {
            name: (lambda candidate=candidate: candidate()[0])
            for name, candidate in fused_candidates.items()
        }
    )
    if precondition_rep:
        benchmark(candidates["rms_gated_then_quant"], precondition_rep)
    results = {
        name: {
            "backend": name,
            "trial_p50_us": [],
            "trial_p10_us": [],
            "trial_p90_us": [],
        }
        for name in candidates
    }
    candidate_items = list(candidates.items())
    for trial in range(trials):
        offset = trial % len(candidate_items)
        for name, candidate in candidate_items[offset:] + candidate_items[:offset]:
            p50_us, p10_us, p90_us = benchmark(candidate, rep)
            results[name]["trial_p50_us"].append(p50_us)
            results[name]["trial_p10_us"].append(p10_us)
            results[name]["trial_p90_us"].append(p90_us)
    for result in results.values():
        result["p50_us"] = statistics.median(result["trial_p50_us"])
        result["p10_us"] = statistics.median(result["trial_p10_us"])
        result["p90_us"] = statistics.median(result["trial_p90_us"])
    return {
        "name": "kda_rms_gated_plus_fp8_quant",
        "m": m,
        "seed": seed,
        "scale_match": scale_match,
        "quantized_match": quantized_match,
        "quantized_relative_error": quantized_error,
        "invalid_candidates": invalid_candidates,
        "results": list(results.values()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, nargs="+", default=[20])
    parser.add_argument("--seed", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--precision",
        choices=("fp8", "bf16", "draft-bf16", "bmm", "all"),
        default="all",
    )
    parser.add_argument("--shape", action="append", default=[])
    parser.add_argument("--rep", type=int, default=500)
    parser.add_argument("--router-sweep", action="store_true")
    parser.add_argument("--bf16-deepgemm-sweep", action="store_true")
    parser.add_argument("--cute-skinny-sweep", action="store_true")
    parser.add_argument("--include-fp8-pipeline", action="store_true")
    parser.add_argument("--triton-sweep", action="store_true")
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--candidate-prefix", action="append", default=[])
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument(
        "--precondition-rep",
        type=int,
        default=0,
        help="Milliseconds of graph replay used to reach sustained GPU clocks.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--checkpoint-layer", type=int, default=12)
    parser.add_argument("--bmm-triton-sweep", action="store_true")
    parser.add_argument(
        "--weight-ring",
        type=int,
        default=1,
        help="Rotate this many FP8 weights per replay; report per-projection time.",
    )
    parser.add_argument("--mla-shared-quant", action="store_true")
    parser.add_argument("--kda-output-fusion", action="store_true")
    parser.add_argument("--kda-output-fusion-sweep", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.kda_output_fusion:
        payload = {
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "measurements": [
                benchmark_kda_output_fusion(
                    m,
                    seed,
                    args.rep,
                    args.trials,
                    args.precondition_rep,
                    args.kda_output_fusion_sweep,
                    set(args.candidate),
                )
                for seed in args.seed
                for m in args.m
            ],
        }
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(payload, indent=2) + "\n", encoding="utf-8"
            )
        return
    if args.mla_shared_quant:
        payload = {
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "measurements": [
                benchmark_mla_shared_quantization(
                    m,
                    args.rep,
                    args.trials,
                    args.precondition_rep,
                    args.weight_ring,
                )
                for m in args.m
            ],
        }
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(payload, indent=2) + "\n", encoding="utf-8"
            )
        return
    shapes = []
    if args.precision in ("fp8", "all"):
        shapes.extend(FP8_SHAPES)
    if args.precision in ("bf16", "all"):
        shapes.extend(BF16_SHAPES)
    if args.precision in ("draft-bf16", "all"):
        shapes.extend(DRAFT_BF16_SHAPES)
    if args.precision in ("bmm", "all"):
        shapes.extend(BMM_SHAPES)
    if args.shape:
        selected = set(args.shape)
        shapes = [shape for shape in shapes if shape.name in selected]
        missing = selected - {shape.name for shape in shapes}
        if missing:
            raise ValueError(f"Unknown shapes: {sorted(missing)}")

    payload = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "measurements": [
            run_shape(
                shape,
                m,
                args.rep,
                args.router_sweep,
                args.bf16_deepgemm_sweep,
                args.cute_skinny_sweep,
                args.include_fp8_pipeline,
                args.triton_sweep,
                set(args.candidate),
                tuple(args.candidate_prefix),
                args.trials,
                args.precondition_rep,
                args.checkpoint_path,
                args.checkpoint_layer,
                args.bmm_triton_sweep,
                args.weight_ring,
            )
            for shape in shapes
            for m in args.m
        ],
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
