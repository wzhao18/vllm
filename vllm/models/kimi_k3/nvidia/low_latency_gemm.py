# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 decode GEMM selection on SM103.

Dispatch is purely by local ``(N, K)`` shape and token count ``M``. Unquantized
BF16 projections select between CuTe skinny, fused-A, and DeepGEMM kernels.
ModelOpt FP8_PB_WO projections can select Triton or constrain DeepGEMM's layout
candidates for measured decode shapes. Plans are installed once per module.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import (
    SkinnyGemmConfig,
    shape_dynamic_skinny_gemm,
)
from vllm.model_executor.kernels.linear.scaled_mm.deep_gemm import (
    DeepGemmFp8BlockScaledMMKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.triton import (
    TritonFp8BlockScaledMMKernel,
)
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptFp8PbWoLinearMethod,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
)
from vllm.models.kimi_k3.nvidia.ops.fused_k128_fp8_gemm import (
    fused_k128_fp8_gemm,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    _block_size_multiple_scope,
    _import_deep_gemm,
    supports_block_size_multiple_of,
)
from vllm.utils.torch_utils import direct_register_custom_op

Backend = Literal["cute", "dsv3_fused_a", "deepgemm_bf16"]
# A resolved per-token-count call and its optional backend configuration.
ResolvedCall = tuple[Backend, SkinnyGemmConfig | tuple[int, int] | None]


@dataclass(frozen=True, slots=True)
class ProjectionSpec:
    n: int
    k: int
    dsv3_tokens: frozenset[int] = frozenset()
    cute_configs: tuple[tuple[int, SkinnyGemmConfig], ...] = ()
    residual_configs: tuple[tuple[int, SkinnyGemmConfig], ...] = ()
    name: str = ""  # optional debug label; never used for dispatch
    deepgemm_configs: tuple[tuple[int, tuple[int, int]], ...] = ()

    def cute_config(self, num_tokens: int) -> SkinnyGemmConfig | None:
        return dict(self.cute_configs).get(num_tokens)

    def residual_config(self, num_tokens: int) -> SkinnyGemmConfig | None:
        return dict(self.residual_configs).get(num_tokens)

    def deepgemm_config(self, num_tokens: int) -> tuple[int, int] | None:
        return dict(self.deepgemm_configs).get(num_tokens)


def _cute(
    num_tokens: int,
    block_size: int,
    outputs_per_block: int,
    k_unroll: int,
    vector_width: int = 8,
    *,
    static_k: int | None = None,
    max_registers: int = 64,
) -> SkinnyGemmConfig:
    return SkinnyGemmConfig(
        num_tokens,
        block_size,
        outputs_per_block,
        k_unroll,
        vector_width,
        static_k,
        max_registers,
    )


_M1_TO_16 = frozenset(range(1, 17))
_M1 = frozenset({1})

# Keyed by local (N, K). Where two projections share a shape (only 1536x7168:
# shared_gate_up_proj and mla_g_proj) the entry is unified.
KIMI_K3_PROJECTIONS: dict[tuple[int, int], ProjectionSpec] = {
    (1536, 128): ProjectionSpec(1536, 128, _M1_TO_16, name="f_b_proj"),
    (3072, 128): ProjectionSpec(3072, 128, _M1_TO_16, name="f_b_proj"),
    # 1536x7168 is shared by shared_gate_up_proj and mla_g_proj. dsv3 M1..16 is
    # only crash-safe once the mla_g aux-stream/PDL capture fix lands (subtask
    # task_7388aba1); the fallback if it cannot be fixed is dsv3_tokens=_M1.
    (1536, 7168): ProjectionSpec(
        1536, 7168, _M1_TO_16, name="shared_gate_up_proj/mla_g_proj"
    ),
    (3072, 7168): ProjectionSpec(
        3072,
        7168,
        cute_configs=(
            (1, _cute(1, 224, 3, 4)),
            (2, _cute(2, 128, 3, 2)),
            (3, _cute(3, 128, 2, 1)),
            (4, _cute(4, 64, 2, 2)),
            (5, _cute(5, 128, 3, 1)),
        ),
        name="shared_gate_up_proj",
    ),
    (2112, 7168): ProjectionSpec(
        2112,
        7168,
        _M1_TO_16,
        cute_configs=(
            (1, _cute(1, 224, 3, 1, static_k=7168)),
            (2, _cute(2, 224, 2, 1, static_k=7168)),
        ),
        name="fused_qkv_a_proj",
    ),
    (2304, 1536): ProjectionSpec(2304, 1536, _M1_TO_16, name="q_b_proj"),
    (4608, 1536): ProjectionSpec(4608, 1536, _M1_TO_16, name="q_b_proj"),
    (3584, 7168): ProjectionSpec(
        3584,
        7168,
        frozenset(range(6, 9)),
        cute_configs=(
            (1, _cute(1, 128, 1, 1, 4, static_k=7168)),
            (2, _cute(2, 64, 4, 1, static_k=7168)),
            (3, _cute(3, 64, 4, 1, static_k=7168)),
            (4, _cute(4, 64, 4, 1, static_k=7168)),
            (5, _cute(5, 64, 4, 1, static_k=7168)),
        ),
        name="routed_expert_down_proj",
    ),
    (6288, 7168): ProjectionSpec(
        6288,
        7168,
        cute_configs=(
            (1, _cute(1, 224, 3, 4)),
            (2, _cute(2, 64, 3, 2)),
            (3, _cute(3, 32, 3, 4)),
            (4, _cute(4, 128, 6, 1)),
        ),
        name="in_proj_qkvgfab",
    ),
    (12448, 7168): ProjectionSpec(
        12448,
        7168,
        cute_configs=(
            (1, _cute(1, 224, 4, 2)),
            (2, _cute(2, 64, 4, 2)),
            (3, _cute(3, 64, 2, 2)),
        ),
        name="in_proj_qkvgfab",
    ),
    (7168, 768): ProjectionSpec(7168, 768, _M1_TO_16, name="shared_down_proj"),
    (7168, 1536): ProjectionSpec(
        7168, 1536, cute_configs=((1, _cute(1, 96, 4, 2)),), name="o_proj"
    ),
    (7168, 3072): ProjectionSpec(
        7168,
        3072,
        cute_configs=(
            (1, _cute(1, 96, 2, 4)),
            (2, _cute(2, 32, 4, 4)),
        ),
        name="o_proj",
    ),
    (7168, 3584): ProjectionSpec(
        7168,
        3584,
        cute_configs=(
            (1, _cute(1, 224, 4, 2)),
            (2, _cute(2, 64, 4, 2)),
        ),
        residual_configs=(
            (1, _cute(1, 64, 2, 1, 4, static_k=3584)),
            (2, _cute(2, 64, 8, 1, 4, static_k=3584)),
            (3, _cute(3, 32, 4, 1, static_k=3584)),
            (4, _cute(4, 32, 4, 1, static_k=3584)),
        ),
        name="routed_expert_up_proj",
    ),
    (7168, 4224): ProjectionSpec(
        7168,
        4224,
        cute_configs=((1, _cute(1, 96, 4, 2, 4)),),
        name="dense_down_proj",
    ),
    (7168, 8448): ProjectionSpec(
        7168,
        8448,
        cute_configs=(
            (1, _cute(1, 32, 4, 4)),
            (2, _cute(2, 96, 4, 1)),
            (3, _cute(3, 96, 4, 1)),
        ),
        name="dense_down_proj",
    ),
    (8448, 7168): ProjectionSpec(
        8448,
        7168,
        cute_configs=(
            (1, _cute(1, 224, 3, 4)),
            (2, _cute(2, 32, 4, 4)),
        ),
        name="dense_gate_up_proj",
    ),
    (16896, 7168): ProjectionSpec(
        16896,
        7168,
        cute_configs=(
            (1, _cute(1, 224, 6, 4)),
            (2, _cute(2, 32, 4, 4)),
        ),
        name="dense_gate_up_proj",
    ),
    (20480, 7168): ProjectionSpec(
        20480,
        7168,
        cute_configs=(
            (1, _cute(1, 224, 4, 2)),
            (2, _cute(2, 64, 4, 2)),
            (3, _cute(3, 64, 2, 2)),
            (4, _cute(4, 64, 4, 1)),
        ),
        name="lm_head",
    ),
    (40960, 7168): ProjectionSpec(
        40960,
        7168,
        cute_configs=(
            (1, _cute(1, 128, 4, 2)),
            (2, _cute(2, 64, 4, 2)),
            (3, _cute(3, 64, 2, 2)),
            (4, _cute(4, 64, 4, 1)),
        ),
        name="lm_head",
    ),
    # TP16. Measured on B300 over M=1..16 with the same >=5% threshold as the
    # entries above. The replicated projections (2112x7168, 3584x7168,
    # 7168x3584) keep their shapes at TP16 and reuse the entries above, and
    # o_proj lands on 7168x768, which shared_down_proj already covers.
    (3216, 7168): ProjectionSpec(
        3216,
        7168,
        # Both gaps in this range are measured, not oversights: dsv3 is only
        # 4% ahead at M6..M8, and at M16 cuBLAS switches to a faster kernel
        # (11.42us vs dsv3's 11.83us) after trailing it by 6-8% at M9..M15.
        frozenset(range(9, 16)),
        cute_configs=(
            (1, _cute(1, 224, 3, 4)),
            (2, _cute(2, 128, 4, 2)),
            (3, _cute(3, 128, 2, 1)),
            (4, _cute(4, 64, 2, 2)),
            (5, _cute(5, 128, 3, 1)),
        ),
        name="in_proj_qkvgfab",
    ),
    (768, 7168): ProjectionSpec(
        768,
        7168,
        frozenset(range(5, 17)),
        cute_configs=(
            (1, _cute(1, 224, 2, 4)),
            (2, _cute(2, 224, 2, 2)),
            (3, _cute(3, 224, 2, 2)),
            (4, _cute(4, 224, 2, 2)),
        ),
        name="mla_g_proj/shared_gate_up_proj",
    ),
    (1152, 1536): ProjectionSpec(
        1152,
        1536,
        frozenset(range(2, 17)),
        ((1, _cute(1, 192, 3, 4)),),
        name="q_b_proj",
    ),
    (768, 128): ProjectionSpec(768, 128, _M1_TO_16, name="f_b_proj"),
    # dsv3 drops under 5% from M9 on for this shape.
    (7168, 384): ProjectionSpec(
        7168, 384, frozenset(range(1, 9)), name="shared_down_proj"
    ),
    (4224, 7168): ProjectionSpec(
        4224,
        7168,
        frozenset(range(4, 9)),
        cute_configs=(
            (1, _cute(1, 224, 3, 4)),
            (2, _cute(2, 128, 2, 1)),
            (3, _cute(3, 64, 2, 2)),
        ),
        name="dense_gate_up_proj",
    ),
    (10240, 7168): ProjectionSpec(
        10240,
        7168,
        cute_configs=(
            (1, _cute(1, 224, 4, 2)),
            (2, _cute(2, 32, 2, 4)),
            (3, _cute(3, 64, 4, 1)),
            (4, _cute(4, 64, 4, 1)),
        ),
        name="lm_head",
    ),
    (67584, 7168): ProjectionSpec(
        67584,
        7168,
        cute_configs=(
            (1, _cute(1, 32, 1, 1, static_k=7168)),
            (2, _cute(2, 64, 4, 1, 4, static_k=7168)),
            (3, _cute(3, 64, 4, 1, 4, static_k=7168)),
            (4, _cute(4, 64, 4, 1, 4, static_k=7168)),
        ),
        name="dense_gate_up_proj",
        deepgemm_configs=(
            (5, (1, 1)),
            *((m, (64, 64)) for m in range(17, 49)),
            *((m, (1, 1)) for m in range(49, 65)),
        ),
    ),
    (7168, 33792): ProjectionSpec(
        7168,
        33792,
        cute_configs=(
            (1, _cute(1, 192, 1, 1, static_k=33792)),
            (2, _cute(2, 128, 4, 1, static_k=33792)),
            (3, _cute(3, 128, 4, 1, 4, static_k=33792)),
            (4, _cute(4, 128, 4, 1, 4, static_k=33792)),
        ),
        name="dense_down_proj",
    ),
    # Replicated TP1 DSpark projections. The draft model has a separate
    # installation hook; the shared full-vocabulary LM head is also exercised
    # by target verification.
    (7168, 35840): ProjectionSpec(
        7168,
        35840,
        cute_configs=(
            (1, _cute(1, 160, 1, 1, static_k=35840)),
            (2, _cute(2, 128, 4, 1, static_k=35840)),
            (3, _cute(3, 128, 4, 1, static_k=35840)),
            (4, _cute(4, 128, 4, 1, 4, static_k=35840)),
            (5, _cute(5, 128, 4, 1, 4, static_k=35840)),
        ),
        name="draft_context_proj",
    ),
    (2880, 7168): ProjectionSpec(
        2880,
        7168,
        cute_configs=(
            (1, _cute(1, 256, 2, 1, 4, static_k=7168)),
            (2, _cute(2, 256, 2, 1, 4, static_k=7168)),
            (3, _cute(3, 128, 8, 1, static_k=7168)),
            (4, _cute(4, 128, 8, 1, static_k=7168)),
            (5, _cute(5, 64, 4, 1, static_k=7168)),
        ),
        name="draft_context_kv_proj",
    ),
    (12288, 1536): ProjectionSpec(
        12288,
        1536,
        cute_configs=(
            (1, _cute(1, 32, 4, 1, static_k=1536)),
            (2, _cute(2, 32, 8, 1, static_k=1536)),
            (3, _cute(3, 32, 8, 1, static_k=1536)),
        ),
        name="draft_q_b_proj",
    ),
    (7168, 8192): ProjectionSpec(
        7168,
        8192,
        cute_configs=(
            (1, _cute(1, 128, 1, 1, 4, static_k=8192)),
            (2, _cute(2, 64, 8, 1, static_k=8192)),
            (3, _cute(3, 64, 8, 1, static_k=8192)),
            (4, _cute(4, 64, 8, 1, static_k=8192)),
        ),
        name="draft_mla_o_proj",
    ),
    (28672, 7168): ProjectionSpec(
        28672,
        7168,
        cute_configs=(
            (1, _cute(1, 224, 4, 4, 4)),
            (2, _cute(2, 64, 4, 2)),
            (3, _cute(3, 64, 2, 4, 4)),
        ),
        name="draft_dense_gate_up",
        deepgemm_configs=tuple((m, (1, 1)) for m in range(1, 65)),
    ),
    (7168, 14336): ProjectionSpec(
        7168,
        14336,
        cute_configs=(
            (
                1,
                _cute(
                    1,
                    64,
                    1,
                    1,
                    4,
                    static_k=14336,
                    max_registers=48,
                ),
            ),
            (2, _cute(2, 64, 4, 1, static_k=14336)),
            (3, _cute(3, 64, 8, 1, static_k=14336)),
            (4, _cute(4, 32, 2, 1, static_k=14336)),
        ),
        name="draft_dense_down",
    ),
    (163840, 7168): ProjectionSpec(
        163840,
        7168,
        cute_configs=(
            (1, _cute(1, 224, 4, 4, 4)),
            (2, _cute(2, 224, 4, 2)),
            (3, _cute(3, 64, 2, 4, 4)),
            (4, _cute(4, 64, 4, 1)),
        ),
        name="tp1_lm_head",
        deepgemm_configs=(
            *((m, (1, 1)) for m in range(1, 10)),
            *((m, (64, 64)) for m in range(10, 49)),
            *((m, (1, 1)) for m in range(49, 65)),
        ),
    ),
    (163840, 256): ProjectionSpec(
        163840,
        256,
        name="draft_markov_w2",
        deepgemm_configs=tuple((m, (1, 1)) for m in range(43, 49)),
    ),
    # 7168x2112 (TP16 dense down_proj) has no entry on purpose: K=2112 divides
    # none of the fused-A tile_k values, and the CuTe kernel is left with
    # vector_width=2, which measured slower than cuBLAS.
}

KIMI_K3_FP8_PB_WO_BLOCK_M_PLANS: dict[tuple[int, int], dict[int, int]] = {
    (7168, 12288): {m: 256 for m in range(11, 257)},
    (18432, 1536): {m: 256 for m in range(17, 65)},
}

KIMI_K3_FP8_PB_WO_BLOCK_SIZE_PLANS: dict[
    tuple[int, int], dict[int, tuple[int, int]]
] = {
    (2176, 7168): {m: (256, 64) for m in range(1, 65)},
}


def _backend_for(
    spec: ProjectionSpec, num_tokens: int, has_residual: bool
) -> Backend | None:
    if has_residual:
        return "cute" if spec.residual_config(num_tokens) is not None else None
    if spec.cute_config(num_tokens) is not None:
        return "cute"
    if spec.deepgemm_config(num_tokens) is not None:
        return "deepgemm_bf16"
    if num_tokens in spec.dsv3_tokens:
        return "dsv3_fused_a"
    return None


def select_kimi_k3_backend(
    num_tokens: int,
    n: int,
    k: int,
    *,
    has_residual: bool = False,
) -> Backend | None:
    """Backend for a local ``(N, K)`` at ``num_tokens``, or None to fall back."""
    spec = KIMI_K3_PROJECTIONS.get((n, k))
    return _backend_for(spec, num_tokens, has_residual) if spec is not None else None


def _build_plan(spec: ProjectionSpec) -> dict[int, ResolvedCall]:
    plan: dict[int, ResolvedCall] = {}
    token_counts = set(range(1, 17))
    token_counts.update(num_tokens for num_tokens, _ in spec.deepgemm_configs)
    for num_tokens in sorted(token_counts):
        backend = _backend_for(spec, num_tokens, has_residual=False)
        if backend == "cute":
            plan[num_tokens] = ("cute", spec.cute_config(num_tokens))
        elif backend == "dsv3_fused_a":
            plan[num_tokens] = ("dsv3_fused_a", None)
        elif backend == "deepgemm_bf16":
            plan[num_tokens] = ("deepgemm_bf16", spec.deepgemm_config(num_tokens))
    return plan


@functools.cache
def _deepgemm_bf16_available() -> bool:
    deep_gemm = _import_deep_gemm()
    return (
        deep_gemm is not None
        and hasattr(deep_gemm, "bf16_gemm_nt")
        and supports_block_size_multiple_of()
    )


def _available_plan(spec: ProjectionSpec) -> dict[int, ResolvedCall]:
    plan = _build_plan(spec)
    if not any(entry[0] == "deepgemm_bf16" for entry in plan.values()):
        return plan
    if _deepgemm_bf16_available():
        return plan
    return {
        num_tokens: entry
        for num_tokens, entry in plan.items()
        if entry[0] != "deepgemm_bf16"
    }


def _build_residual_plan(spec: ProjectionSpec) -> dict[int, SkinnyGemmConfig]:
    return {num_tokens: config for num_tokens, config in spec.residual_configs}


def _is_sm103() -> bool:
    return current_platform.is_device_capability((10, 3))


def _is_packed_row_major(tensor: torch.Tensor) -> bool:
    return tensor.dim() == 2 and tensor.stride() == (tensor.shape[1], 1)


def _runtime_ok(x: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        _is_packed_row_major(x)
        and _is_packed_row_major(weight)
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and x.is_cuda
        and weight.is_cuda
        and x.device == weight.device
        and x.shape[1] == weight.shape[1]
    )


def _residual_ok(x: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor) -> bool:
    return (
        residual.dim() == 2
        and residual.dtype == torch.bfloat16
        and residual.is_cuda
        and residual.device == x.device
        and residual.is_contiguous()
        and residual.shape == (x.shape[0], weight.shape[0])
    )


def _run_plan(
    plan: dict[int, ResolvedCall], x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor | None:
    entry = plan.get(x.shape[0])
    if entry is None:
        return None
    backend, config = entry
    if backend == "deepgemm_bf16":
        assert isinstance(config, tuple)
        output = torch.empty(
            (x.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device
        )
        _launch_kimi_k3_bf16_gemm_nt(x, weight, output, *config)
        return output
    if backend == "cute":
        assert isinstance(config, SkinnyGemmConfig)
        if not shape_dynamic_skinny_gemm.is_available():
            return None
        return shape_dynamic_skinny_gemm(x, weight, config, None)
    if not hasattr(torch.ops._C, "dsv3_fused_a_gemm"):
        return None
    output = torch.empty((x.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device)
    ops.dsv3_fused_a_gemm(output, x, weight.t(), enable_pdl=True)
    return output


def _launch_kimi_k3_bf16_gemm_nt(
    x: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    block_m_multiple: int,
    block_n_multiple: int,
) -> None:
    torch.ops.vllm.kimi_k3_bf16_gemm_nt(
        x,
        weight,
        output,
        block_m_multiple,
        block_n_multiple,
    )


def _kimi_k3_bf16_gemm_nt(
    x: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    block_m_multiple: int,
    block_n_multiple: int,
) -> None:
    deep_gemm = _import_deep_gemm()
    if deep_gemm is None or not hasattr(deep_gemm, "bf16_gemm_nt"):
        raise RuntimeError("DeepGEMM BF16 GEMM is unavailable")
    if (block_m_multiple, block_n_multiple) == (1, 1):
        deep_gemm.bf16_gemm_nt(x, weight, output)
        return
    with _block_size_multiple_scope((block_m_multiple, block_n_multiple)):
        deep_gemm.bf16_gemm_nt(x, weight, output)


def _kimi_k3_bf16_gemm_nt_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    block_m_multiple: int,
    block_n_multiple: int,
) -> None:
    return None


direct_register_custom_op(
    "kimi_k3_bf16_gemm_nt",
    _kimi_k3_bf16_gemm_nt,
    mutates_args=["output"],
    fake_impl=_kimi_k3_bf16_gemm_nt_fake,
)


def _run_residual_plan(
    residual_plan: dict[int, SkinnyGemmConfig],
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor | None:
    config = residual_plan.get(x.shape[0])
    if config is None or not shape_dynamic_skinny_gemm.is_available():
        return None
    return shape_dynamic_skinny_gemm(x, weight, config, residual)


def try_low_latency_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Run the shape-selected low-latency kernel, or None to fall back.

    Resolves the plan from the shape table on each call; production installs a
    precomputed plan (see :func:`enable_kimi_k3_low_latency_gemm`) and does not
    use this path.
    """
    if envs.VLLM_BATCH_INVARIANT or not _is_sm103() or not _runtime_ok(x, weight):
        return None
    spec = KIMI_K3_PROJECTIONS.get((weight.shape[0], weight.shape[1]))
    if spec is None:
        return None
    if residual is None:
        return _run_plan(_available_plan(spec), x, weight)
    if not _residual_ok(x, weight, residual):
        return None
    return _run_residual_plan(_build_residual_plan(spec), x, weight, residual)


class _KimiK3LowLatencyApply:
    """Mixin: try the precomputed plan, else defer to the base method."""

    def __init__(self, plan: dict[int, ResolvedCall]) -> None:
        self._plan = plan

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            bias is None
            and not envs.VLLM_BATCH_INVARIANT
            and _runtime_ok(x, layer.weight)
        ):
            output = _run_plan(self._plan, x, layer.weight)
            if output is not None:
                return output
        return super().apply(layer, x, bias)  # type: ignore[misc]


class KimiK3LowLatencyLinearMethod(_KimiK3LowLatencyApply, UnquantizedLinearMethod):
    def __init__(
        self,
        plan: dict[int, ResolvedCall],
        residual_plan: dict[int, SkinnyGemmConfig],
    ) -> None:
        super().__init__(plan)
        self._residual_plan = residual_plan

    def apply_with_residual(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        if (
            not envs.VLLM_BATCH_INVARIANT
            and _runtime_ok(x, layer.weight)
            and _residual_ok(x, layer.weight, residual)
        ):
            output = _run_residual_plan(self._residual_plan, x, layer.weight, residual)
            if output is not None:
                return output
        return torch.addmm(residual, x, layer.weight.t())


class KimiK3LowLatencyEmbeddingMethod(
    _KimiK3LowLatencyApply, UnquantizedEmbeddingMethod
):
    pass


class KimiK3FusedK128Fp8LinearKernel(TritonFp8BlockScaledMMKernel):
    def apply_weights(
        self,
        layer: nn.Module,
        x: torch.Tensor | QuantizedActivation,
        bias: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if isinstance(x, torch.Tensor) and x.shape[-1] == 128:
            input_2d = x.view(-1, x.shape[-1])
            if input_2d.shape[0] <= 64:
                params = self._get_layer_params(layer)
                weight_scale = (
                    params.weight_scale
                    if params.weight_scale_inv is None
                    else params.weight_scale_inv
                )
                assert weight_scale is not None
                output = fused_k128_fp8_gemm(
                    input_2d,
                    params.weight,
                    weight_scale,
                )
                if bias is not None:
                    output.add_(bias)
                return output.view(*x.shape[:-1], params.weight.shape[0])
        return super().apply_weights(layer, x, bias, **kwargs)


def enable_kimi_k3_low_latency_gemm(
    module: nn.Module,
    dtype: torch.dtype,
) -> None:
    """Install shape-selected low-latency GEMMs and register CuTe warmups."""
    if dtype != torch.bfloat16 or not _is_sm103():
        return

    warmup_configs: set[SkinnyGemmConfig] = set()
    residual_warmup_configs: set[SkinnyGemmConfig] = set()
    for child in module.modules():
        if (
            not envs.VLLM_BATCH_INVARIANT
            and isinstance(child, LinearBase)
            and isinstance(child.quant_method, ModelOptFp8PbWoLinearMethod)
        ):
            kernel = child.quant_method.w8a8_block_fp8_linear
            if tuple(child.weight.shape) == (12288, 128) and isinstance(
                kernel, DeepGemmFp8BlockScaledMMKernel
            ):
                child.quant_method.w8a8_block_fp8_linear = (
                    KimiK3FusedK128Fp8LinearKernel(kernel.config)
                )
                continue
            shape = tuple(child.weight.shape)
            block_size_plan = KIMI_K3_FP8_PB_WO_BLOCK_SIZE_PLANS.get(shape)
            block_m_plan = KIMI_K3_FP8_PB_WO_BLOCK_M_PLANS.get(shape)
            if (
                block_size_plan is not None
                and isinstance(kernel, DeepGemmFp8BlockScaledMMKernel)
                and supports_block_size_multiple_of()
            ):
                kernel.set_block_size_multiple_plan(block_size_plan)
            elif (
                block_m_plan is not None
                and isinstance(kernel, DeepGemmFp8BlockScaledMMKernel)
                and supports_block_size_multiple_of()
            ):
                kernel.set_block_m_multiple_plan(block_m_plan)
            continue

        is_linear = (
            isinstance(child, LinearBase)
            and type(child.quant_method) is UnquantizedLinearMethod
        )
        # ParallelLMHead is a VocabParallelEmbedding subclass; embed_tokens is
        # the parent type, so isinstance already excludes it.
        is_head = (
            isinstance(child, ParallelLMHead)
            and type(child.quant_method) is UnquantizedEmbeddingMethod
        )
        if not (is_linear or is_head):
            continue
        weight = getattr(child, "weight", None)
        if weight is None or weight.dim() != 2:
            continue
        spec = KIMI_K3_PROJECTIONS.get((weight.shape[0], weight.shape[1]))
        if spec is None:
            continue
        if is_linear:
            child.quant_method = KimiK3LowLatencyLinearMethod(
                _available_plan(spec), _build_residual_plan(spec)
            )
        else:
            child.quant_method = KimiK3LowLatencyEmbeddingMethod(_available_plan(spec))
        # Warm up only the configs measured for this module's local (N, K) so a
        # TP8 deployment does not compile TP4 configs and vice versa.
        warmup_configs.update(config for _, config in spec.cute_configs)
        residual_warmup_configs.update(config for _, config in spec.residual_configs)

    if shape_dynamic_skinny_gemm.is_available():
        if warmup_configs:
            shape_dynamic_skinny_gemm.request_warmup_configs(dtype, warmup_configs)
        if residual_warmup_configs:
            shape_dynamic_skinny_gemm.request_warmup_configs(
                dtype, residual_warmup_configs, has_residual=True
            )
