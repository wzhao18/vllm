# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for BF16 skinny GEMMs and the Kimi-K3 SM103 selector."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import regex as re
import torch
from torch import nn

from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import (
    SkinnyGemmConfig,
)
from vllm.model_executor.kernels.linear.scaled_mm import deep_gemm as deep_gemm_kernel
from vllm.models.deepseek_v32.nvidia import glm52_low_latency_gemm as glm52_gemm
from vllm.models.kimi_k3.nvidia import low_latency_gemm as k3_gemm
from vllm.models.kimi_k3.nvidia.low_latency_gemm import KIMI_K3_PROJECTIONS
from vllm.utils import deep_gemm as deep_gemm_utils

# Keyed by local (N, K): (cute token counts, dsv3 token counts). 1536x7168 is
# the unified shared_gate_up_proj/mla_g_proj entry (dsv3 M1..16).
EXPECTED_SELECTIONS = {
    (1536, 128): (set(), set(range(1, 17))),
    (3072, 128): (set(), set(range(1, 17))),
    (1536, 7168): (set(), set(range(1, 17))),
    (3072, 7168): (set(range(1, 6)), set()),
    (2112, 7168): ({1, 2}, set(range(1, 17))),
    (2304, 1536): (set(), set(range(1, 17))),
    (4608, 1536): (set(), set(range(1, 17))),
    (3584, 7168): (set(range(1, 6)), set(range(6, 9))),
    (6288, 7168): (set(range(1, 5)), set()),
    (12448, 7168): (set(range(1, 4)), set()),
    (7168, 768): (set(), set(range(1, 17))),
    (7168, 1536): ({1}, set()),
    (7168, 3072): ({1, 2}, set()),
    (7168, 3584): ({1, 2}, set()),
    (7168, 4224): ({1}, set()),
    (7168, 8448): (set(range(1, 4)), set()),
    (8448, 7168): ({1, 2}, set()),
    (16896, 7168): ({1, 2}, set()),
    (20480, 7168): (set(range(1, 5)), set()),
    (40960, 7168): (set(range(1, 5)), set()),
    # TP16.
    (3216, 7168): (set(range(1, 6)), set(range(9, 16))),
    (768, 7168): (set(range(1, 5)), set(range(5, 17))),
    (1152, 1536): ({1}, set(range(2, 17))),
    (768, 128): (set(), set(range(1, 17))),
    (7168, 384): (set(), set(range(1, 9))),
    (4224, 7168): (set(range(1, 4)), set(range(4, 9))),
    (10240, 7168): (set(range(1, 5)), set()),
    (7168, 33792): (set(range(1, 5)), set()),
    # Replicated TP1 DSpark projections.
    (7168, 35840): (set(range(1, 6)), set()),
    (2880, 7168): (set(range(1, 6)), set()),
    (12288, 1536): (set(range(1, 4)), set()),
    (7168, 8192): (set(range(1, 5)), set()),
    (7168, 14336): (set(range(1, 5)), set()),
}

CUTE_CASES = [
    (spec.n, spec.k, num_tokens)
    for spec in k3_gemm.KIMI_K3_PROJECTIONS.values()
    for num_tokens, _ in spec.cute_configs
]

RESIDUAL_CUTE_CASES = [
    (spec.n, spec.k, num_tokens)
    for spec in k3_gemm.KIMI_K3_PROJECTIONS.values()
    for num_tokens, _ in spec.residual_configs
]

GLM_CUTE_CASES = [
    (spec, config)
    for spec in glm52_gemm.GLM52_PROJECTIONS.values()
    for _, config in spec.cute_configs
]

EXPECTED_CUTE_CONFIGS = {
    (3072, 7168, 1): (224, 3, 4, 8),
    (3072, 7168, 2): (128, 3, 2, 8),
    (3072, 7168, 3): (128, 2, 1, 8),
    (3072, 7168, 4): (64, 2, 2, 8),
    (3072, 7168, 5): (128, 3, 1, 8),
    (2112, 7168, 1): (224, 3, 1, 8),
    (2112, 7168, 2): (224, 2, 1, 8),
    (3584, 7168, 1): (128, 1, 1, 4),
    (3584, 7168, 2): (64, 4, 1, 8),
    (3584, 7168, 3): (64, 4, 1, 8),
    (3584, 7168, 4): (64, 4, 1, 8),
    (3584, 7168, 5): (64, 4, 1, 8),
    (6288, 7168, 1): (224, 3, 4, 8),
    (6288, 7168, 2): (64, 3, 2, 8),
    (6288, 7168, 3): (32, 3, 4, 8),
    (6288, 7168, 4): (128, 6, 1, 8),
    (12448, 7168, 1): (224, 4, 2, 8),
    (12448, 7168, 2): (64, 4, 2, 8),
    (12448, 7168, 3): (64, 2, 2, 8),
    (7168, 1536, 1): (96, 4, 2, 8),
    (7168, 3072, 1): (96, 2, 4, 8),
    (7168, 3072, 2): (32, 4, 4, 8),
    (7168, 3584, 1): (224, 4, 2, 8),
    (7168, 3584, 2): (64, 4, 2, 8),
    (7168, 4224, 1): (96, 4, 2, 4),
    (7168, 8448, 1): (32, 4, 4, 8),
    (7168, 8448, 2): (96, 4, 1, 8),
    (7168, 8448, 3): (96, 4, 1, 8),
    (8448, 7168, 1): (224, 3, 4, 8),
    (8448, 7168, 2): (32, 4, 4, 8),
    (16896, 7168, 1): (224, 6, 4, 8),
    (16896, 7168, 2): (32, 4, 4, 8),
    (20480, 7168, 1): (224, 4, 2, 8),
    (20480, 7168, 2): (64, 4, 2, 8),
    (20480, 7168, 3): (64, 2, 2, 8),
    (20480, 7168, 4): (64, 4, 1, 8),
    (40960, 7168, 1): (128, 4, 2, 8),
    (40960, 7168, 2): (64, 4, 2, 8),
    (40960, 7168, 3): (64, 2, 2, 8),
    (40960, 7168, 4): (64, 4, 1, 8),
    # TP16.
    (3216, 7168, 1): (224, 3, 4, 8),
    (3216, 7168, 2): (128, 4, 2, 8),
    (3216, 7168, 3): (128, 2, 1, 8),
    (3216, 7168, 4): (64, 2, 2, 8),
    (3216, 7168, 5): (128, 3, 1, 8),
    (768, 7168, 1): (224, 2, 4, 8),
    (768, 7168, 2): (224, 2, 2, 8),
    (768, 7168, 3): (224, 2, 2, 8),
    (768, 7168, 4): (224, 2, 2, 8),
    (1152, 1536, 1): (192, 3, 4, 8),
    (4224, 7168, 1): (224, 3, 4, 8),
    (4224, 7168, 2): (128, 2, 1, 8),
    (4224, 7168, 3): (64, 2, 2, 8),
    (10240, 7168, 1): (224, 4, 2, 8),
    (10240, 7168, 2): (32, 2, 4, 8),
    (10240, 7168, 3): (64, 4, 1, 8),
    (10240, 7168, 4): (64, 4, 1, 8),
    (67584, 7168, 1): (32, 1, 1, 8),
    (67584, 7168, 2): (64, 4, 1, 4),
    (67584, 7168, 3): (64, 4, 1, 4),
    (67584, 7168, 4): (64, 4, 1, 4),
    (7168, 33792, 1): (192, 1, 1, 8),
    (7168, 33792, 2): (128, 4, 1, 8),
    (7168, 33792, 3): (128, 4, 1, 4),
    (7168, 33792, 4): (128, 4, 1, 4),
    # Replicated TP1 DSpark projections.
    (7168, 35840, 1): (160, 1, 1, 8),
    (7168, 35840, 2): (128, 4, 1, 8),
    (7168, 35840, 3): (128, 4, 1, 8),
    (7168, 35840, 4): (128, 4, 1, 4),
    (7168, 35840, 5): (128, 4, 1, 4),
    (2880, 7168, 1): (256, 2, 1, 4),
    (2880, 7168, 2): (256, 2, 1, 4),
    (2880, 7168, 3): (128, 8, 1, 8),
    (2880, 7168, 4): (128, 8, 1, 8),
    (2880, 7168, 5): (64, 4, 1, 8),
    (12288, 1536, 1): (32, 4, 1, 8),
    (12288, 1536, 2): (32, 8, 1, 8),
    (12288, 1536, 3): (32, 8, 1, 8),
    (7168, 8192, 1): (128, 1, 1, 4),
    (7168, 8192, 2): (64, 8, 1, 8),
    (7168, 8192, 3): (64, 8, 1, 8),
    (7168, 8192, 4): (64, 8, 1, 8),
    (28672, 7168, 1): (224, 4, 4, 4),
    (28672, 7168, 2): (64, 4, 2, 8),
    (28672, 7168, 3): (64, 2, 4, 4),
    (7168, 14336, 1): (64, 1, 1, 4),
    (7168, 14336, 2): (64, 4, 1, 8),
    (7168, 14336, 3): (64, 8, 1, 8),
    (7168, 14336, 4): (32, 2, 1, 8),
    (163840, 7168, 1): (224, 4, 4, 4),
    (163840, 7168, 2): (224, 4, 2, 8),
    (163840, 7168, 3): (64, 2, 4, 4),
    (163840, 7168, 4): (64, 4, 1, 8),
}

EXPECTED_RESIDUAL_CUTE_CONFIGS = {
    (7168, 3584, 1): (64, 2, 1, 4),
    (7168, 3584, 2): (64, 8, 1, 4),
    (7168, 3584, 3): (32, 4, 1, 8),
    (7168, 3584, 4): (32, 4, 1, 8),
}

EXPECTED_STATIC_K_CONFIGS = {
    **{(2112, 7168, m): (7168, 64) for m in range(1, 3)},
    **{(3584, 7168, m): (7168, 64) for m in range(1, 6)},
    **{(67584, 7168, m): (7168, 64) for m in range(1, 5)},
    **{(7168, 33792, m): (33792, 64) for m in range(1, 5)},
    **{(7168, 35840, m): (35840, 64) for m in range(1, 6)},
    **{(2880, 7168, m): (7168, 64) for m in range(1, 6)},
    **{(12288, 1536, m): (1536, 64) for m in range(1, 4)},
    **{(7168, 8192, m): (8192, 64) for m in range(1, 5)},
    (7168, 14336, 1): (14336, 48),
    **{(7168, 14336, m): (14336, 64) for m in range(2, 5)},
}

EXPECTED_RESIDUAL_STATIC_K_CONFIGS = {(7168, 3584, m): (3584, 64) for m in range(1, 5)}


def _config_tuple(config) -> tuple[int, int, int, int]:
    return (
        config.block_size,
        config.outputs_per_block,
        config.k_unroll,
        config.vector_width,
    )


def test_table_is_keyed_by_shape() -> None:
    for (n, k), spec in k3_gemm.KIMI_K3_PROJECTIONS.items():
        assert (spec.n, spec.k) == (n, k)


def test_every_dsv3_routed_shape_is_instantiated() -> None:
    """dsv3_fused_a_gemm specializes on (K, N); an unlisted shape raises.

    The table routes by shape while the kernel is built per shape, so a missing
    instantiation only shows up at the token counts that route to dsv3. Checking
    it here needs no GPU, which is the point -- a GPU-only check is exactly what
    let (3216, 7168) ship without its DISPATCH_DSV3_SHAPE(7168, 3216).
    """
    source = (
        Path(__file__).resolve().parents[2]
        / "csrc"
        / "libtorch_stable"
        / "dsv3_fused_a_gemm.cu"
    ).read_text(encoding="utf-8")
    # Benchmark-only shapes live behind VLLM_K3_BENCH_SHAPES and are not built
    # by default, so they must not count as available.
    production_macros = source.split("#ifdef VLLM_K3_BENCH_SHAPES")[0]
    explicit = source.split("#undef DISPATCH_DSV3_SHAPE")[1].split(
        "#ifdef VLLM_K3_BENCH_SHAPES"
    )[0]
    compiled = {
        (int(hd_in), int(hd_out))
        for hd_in, hd_out in re.findall(
            r"DISPATCH_DSV3_SHAPE\((\d+),\s*(\d+)\)", production_macros
        )
    } | {
        (int(hd_in), int(hd_out))
        for hd_in, hd_out in re.findall(
            r"hd_in == (\d+) && hd_out == (\d+)",
            production_macros + explicit,
        )
    }
    assert compiled, "failed to parse the dispatch list"

    specs = [
        *KIMI_K3_PROJECTIONS.values(),
        glm52_gemm.GLM52_QKV_A_PROJECTION,
        glm52_gemm.GLM52_Q_B_PROJECTION,
    ]
    missing = sorted(
        (spec.n, spec.k)
        for spec in specs
        if spec.dsv3_tokens and (spec.k, spec.n) not in compiled
    )
    assert not missing, (
        f"routed to dsv3 with no instantiation: {missing}; add "
        "DISPATCH_DSV3_SHAPE(K, N) for each"
    )


def test_packed_row_major_rejects_single_row_slice() -> None:
    packed = torch.empty(1, 128)
    sliced = torch.empty(1, 144)[:, :128]

    assert packed.is_contiguous()
    assert sliced.is_contiguous()
    assert k3_gemm._is_packed_row_major(packed)
    assert not k3_gemm._is_packed_row_major(sliced)


def test_cute_configs_match_measured_table() -> None:
    configs = [
        (spec.n, spec.k, num_tokens, config)
        for spec in k3_gemm.KIMI_K3_PROJECTIONS.values()
        for num_tokens, config in spec.cute_configs
    ]
    actual = {
        (n, k, num_tokens): _config_tuple(config)
        for n, k, num_tokens, config in configs
    }
    assert actual == EXPECTED_CUTE_CONFIGS
    actual_static = {
        (n, k, num_tokens): (config.static_k, config.max_registers)
        for n, k, num_tokens, config in configs
        if config.static_k is not None
    }
    assert actual_static == EXPECTED_STATIC_K_CONFIGS


def test_residual_cute_configs_match_measured_table() -> None:
    configs = [
        (spec.n, spec.k, num_tokens, config)
        for spec in k3_gemm.KIMI_K3_PROJECTIONS.values()
        for num_tokens, config in spec.residual_configs
    ]
    actual = {
        (n, k, num_tokens): _config_tuple(config)
        for n, k, num_tokens, config in configs
    }
    assert actual == EXPECTED_RESIDUAL_CUTE_CONFIGS
    actual_static = {
        (n, k, num_tokens): (config.static_k, config.max_registers)
        for n, k, num_tokens, config in configs
        if config.static_k is not None
    }
    assert actual_static == EXPECTED_RESIDUAL_STATIC_K_CONFIGS


def test_glm52_projection_plans_are_separate() -> None:
    qkv_a = glm52_gemm.GLM52_QKV_A_PROJECTION
    q_b = glm52_gemm.GLM52_Q_B_PROJECTION

    qkv_a_plan = qkv_a.build_plan()
    q_b_plan = q_b.build_plan()

    assert (qkv_a.n, qkv_a.k, set(qkv_a_plan)) == (
        2624,
        6144,
        set(range(1, 17)),
    )
    assert (q_b.n, q_b.k, set(q_b_plan)) == (
        2048,
        2048,
        set(range(1, 17)),
    )
    assert {
        num_tokens
        for num_tokens, (backend, _) in qkv_a_plan.items()
        if backend == "cute"
    } == {1, 2}
    assert {
        num_tokens
        for num_tokens, (backend, _) in qkv_a_plan.items()
        if backend == "dsv3_fused_a"
    } == set(range(3, 17))
    assert {
        num_tokens for num_tokens, (backend, _) in q_b_plan.items() if backend == "cute"
    } == {1, 2}
    assert {
        num_tokens
        for num_tokens, (backend, _) in q_b_plan.items()
        if backend == "dsv3_fused_a"
    } == set(range(3, 17))

    eh = glm52_gemm.GLM52_EH_PROJECTION
    eh_plan = eh.build_plan()
    assert (eh.n, eh.k) == (6144, 12288)
    # The MTP eh_proj has no dsv3 winners; M >= 4 falls back to cuBLAS.
    assert set(eh_plan) == {1, 2, 3}
    assert all(backend == "cute" for backend, _ in eh_plan.values())


def test_glm52_layout_rejects_nonpacked_single_row_view() -> None:
    single_row = torch.empty(1, 144)[:, :128]
    multiple_rows = torch.empty(2, 144)[:, :128]

    assert single_row.stride() == multiple_rows.stride() == (144, 1)
    assert not glm52_gemm._is_supported_row_major(single_row)
    assert not glm52_gemm._is_supported_row_major(multiple_rows)


def test_glm52_installer_maps_only_selected_unquantized_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLinearBase(nn.Module):
        def __init__(
            self,
            n: int,
            k: int,
            quant_method: object,
        ) -> None:
            super().__init__()
            self.weight = nn.Parameter(
                torch.empty(
                    n,
                    k,
                    dtype=torch.bfloat16,
                    device="meta",
                )
            )
            self.quant_method = quant_method

    qkv_a = glm52_gemm.GLM52_QKV_A_PROJECTION
    q_b = glm52_gemm.GLM52_Q_B_PROJECTION
    root = nn.Module()
    root.attn = nn.Module()
    root.attn.fused_qkv_a_proj = FakeLinearBase(
        qkv_a.n, qkv_a.k, glm52_gemm.UnquantizedLinearMethod()
    )
    root.attn.q_b_proj = FakeLinearBase(
        q_b.n, q_b.k, glm52_gemm.UnquantizedLinearMethod()
    )
    root.same_shape_other_name = FakeLinearBase(
        qkv_a.n, qkv_a.k, glm52_gemm.UnquantizedLinearMethod()
    )
    root.quantized = nn.Module()
    quantized_method = object()
    root.quantized.q_b_proj = FakeLinearBase(q_b.n, q_b.k, quantized_method)
    root.wrong_shape = nn.Module()
    root.wrong_shape.fused_qkv_a_proj = FakeLinearBase(
        qkv_a.n + 1, qkv_a.k, glm52_gemm.UnquantizedLinearMethod()
    )
    monkeypatch.setattr(glm52_gemm, "LinearBase", FakeLinearBase)
    monkeypatch.setattr(glm52_gemm, "_is_sm103", lambda: True)
    monkeypatch.setattr(
        glm52_gemm.shape_dynamic_skinny_gemm,
        "is_available",
        lambda: False,
    )

    glm52_gemm.enable_glm52_low_latency_gemm(
        root,
        torch.bfloat16,
    )

    assert isinstance(
        root.attn.fused_qkv_a_proj.quant_method,
        glm52_gemm.GLM52LowLatencyLinearMethod,
    )
    assert isinstance(
        root.attn.q_b_proj.quant_method,
        glm52_gemm.GLM52LowLatencyLinearMethod,
    )
    assert root.attn.fused_qkv_a_proj.quant_method._plan == qkv_a.build_plan()
    assert root.attn.q_b_proj.quant_method._plan == q_b.build_plan()
    assert isinstance(
        root.same_shape_other_name.quant_method,
        glm52_gemm.GLM52LowLatencyLinearMethod,
    )
    assert root.quantized.q_b_proj.quant_method is quantized_method
    assert (
        type(root.wrong_shape.fused_qkv_a_proj.quant_method)
        is glm52_gemm.UnquantizedLinearMethod
    )


@pytest.mark.parametrize("key", EXPECTED_SELECTIONS)
def test_sm103_selector_table(key: tuple[int, int]) -> None:
    n, k = key
    cute_tokens, dsv3_tokens = EXPECTED_SELECTIONS[key]
    for num_tokens in range(1, 17):
        backend = k3_gemm.select_kimi_k3_backend(num_tokens, n, k)
        if num_tokens in cute_tokens:
            assert backend == "cute"
        elif num_tokens in dsv3_tokens:
            assert backend == "dsv3_fused_a"
        else:
            assert backend is None


@pytest.mark.parametrize("key", EXPECTED_SELECTIONS)
def test_selector_requires_supported_shape_and_tokens(key: tuple[int, int]) -> None:
    n, k = key
    assert k3_gemm.select_kimi_k3_backend(0, n, k) is None
    assert k3_gemm.select_kimi_k3_backend(17, n, k) is None
    assert k3_gemm.select_kimi_k3_backend(1, n + 1, k) is None
    assert k3_gemm.select_kimi_k3_backend(1, n, k + 1) is None


def test_unlisted_shape_and_unselected_tokens_fall_back() -> None:
    # Shape absent from the table.
    assert k3_gemm.select_kimi_k3_backend(1, 1000, 1000) is None
    # o_proj (7168,1536) is CuTe M1 only; M2+ falls back.
    assert k3_gemm.select_kimi_k3_backend(2, 7168, 1536) is None


@pytest.mark.parametrize("num_tokens", range(1, 17))
def test_sm103_residual_selector_table(num_tokens: int) -> None:
    backend = k3_gemm.select_kimi_k3_backend(num_tokens, 7168, 3584, has_residual=True)
    assert backend == ("cute" if num_tokens <= 4 else None)


def test_build_plan_matches_selector() -> None:
    for spec in k3_gemm.KIMI_K3_PROJECTIONS.values():
        plan = k3_gemm._build_plan(spec)
        for num_tokens in range(1, 17):
            backend = k3_gemm.select_kimi_k3_backend(num_tokens, spec.n, spec.k)
            if backend is None:
                assert num_tokens not in plan
            else:
                assert plan[num_tokens][0] == backend


def test_dense_gate_up_uses_deepgemm_at_measured_ranges() -> None:
    for num_tokens in range(1, 5):
        assert k3_gemm.select_kimi_k3_backend(num_tokens, 67584, 7168) == "cute"
    for num_tokens in (5, *range(17, 65)):
        assert (
            k3_gemm.select_kimi_k3_backend(num_tokens, 67584, 7168) == "deepgemm_bf16"
        )
    for num_tokens in range(6, 17):
        assert k3_gemm.select_kimi_k3_backend(num_tokens, 67584, 7168) is None
    spec = k3_gemm.KIMI_K3_PROJECTIONS[(67584, 7168)]
    plan = k3_gemm._build_plan(spec)
    assert plan[1][0] == "cute"
    assert isinstance(plan[1][1], SkinnyGemmConfig)
    assert plan[5] == ("deepgemm_bf16", (1, 1))
    assert plan[17] == ("deepgemm_bf16", (64, 64))
    assert plan[48] == ("deepgemm_bf16", (64, 64))
    assert plan[49] == ("deepgemm_bf16", (1, 1))
    assert plan[64] == ("deepgemm_bf16", (1, 1))
    assert set(plan) == {*range(1, 6), *range(17, 65)}


def test_tp1_dspark_plans_use_measured_ranges() -> None:
    gate = k3_gemm._build_plan(k3_gemm.KIMI_K3_PROJECTIONS[(28672, 7168)])
    assert set(gate) == set(range(1, 65))
    assert gate[1][0] == "cute"
    assert gate[3][0] == "cute"
    assert gate[4] == ("deepgemm_bf16", (1, 1))
    assert gate[64] == ("deepgemm_bf16", (1, 1))

    head = k3_gemm._build_plan(k3_gemm.KIMI_K3_PROJECTIONS[(163840, 7168)])
    assert set(head) == set(range(1, 65))
    assert head[1][0] == "cute"
    assert head[4][0] == "cute"
    assert head[5] == ("deepgemm_bf16", (1, 1))
    assert head[10] == ("deepgemm_bf16", (64, 64))
    assert head[48] == ("deepgemm_bf16", (64, 64))
    assert head[49] == ("deepgemm_bf16", (1, 1))
    assert head[64] == ("deepgemm_bf16", (1, 1))

    markov = k3_gemm._build_plan(k3_gemm.KIMI_K3_PROJECTIONS[(163840, 256)])
    assert set(markov) == set(range(43, 49))
    assert set(markov.values()) == {("deepgemm_bf16", (1, 1))}


def test_unavailable_deepgemm_is_removed_from_installed_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = k3_gemm.KIMI_K3_PROJECTIONS[(67584, 7168)]

    monkeypatch.setattr(k3_gemm, "_deepgemm_bf16_available", lambda: False)
    assert set(k3_gemm._available_plan(spec)) == set(range(1, 5))
    assert all(
        backend == "cute" for backend, _ in k3_gemm._available_plan(spec).values()
    )

    monkeypatch.setattr(k3_gemm, "_deepgemm_bf16_available", lambda: True)
    assert k3_gemm._available_plan(spec) == k3_gemm._build_plan(spec)


def test_kimi_k3_fp8_pb_wo_plan_uses_measured_ranges() -> None:
    assert {
        (7168, 12288): {m: 256 for m in range(11, 257)},
        (18432, 1536): {m: 256 for m in range(17, 65)},
    } == k3_gemm.KIMI_K3_FP8_PB_WO_BLOCK_M_PLANS
    assert {
        (2176, 7168): {m: (256, 64) for m in range(1, 65)},
    } == k3_gemm.KIMI_K3_FP8_PB_WO_BLOCK_SIZE_PLANS


def test_installation_is_shape_and_quant_method_specific(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLinear(nn.Module):
        def __init__(self, quant_method: object, n: int, k: int) -> None:
            super().__init__()
            self.quant_method = quant_method
            self.weight = torch.empty(n, k)

    class FakeHead(nn.Module):
        def __init__(self, n: int, k: int) -> None:
            super().__init__()
            self.quant_method = k3_gemm.UnquantizedEmbeddingMethod()
            self.weight = torch.empty(n, k)

    root = nn.Module()
    # dsv3-only shape (no cute warmup contribution).
    root.dsv3_only = FakeLinear(k3_gemm.UnquantizedLinearMethod(), 2304, 1536)
    # quantized: must be left untouched.
    quantized_method = object()
    root.quantized = FakeLinear(quantized_method, 6288, 7168)
    pbwo_method = k3_gemm.ModelOptFp8PbWoLinearMethod.__new__(
        k3_gemm.ModelOptFp8PbWoLinearMethod
    )
    pbwo_kernel = k3_gemm.DeepGemmFp8BlockScaledMMKernel.__new__(
        k3_gemm.DeepGemmFp8BlockScaledMMKernel
    )
    pbwo_kernel._block_m_multiple_plan = {}
    pbwo_kernel._block_size_multiple_plan = {}
    pbwo_method.w8a8_block_fp8_linear = pbwo_kernel
    root.pbwo = FakeLinear(pbwo_method, 7168, 12288)
    pbwo_q_b_method = k3_gemm.ModelOptFp8PbWoLinearMethod.__new__(
        k3_gemm.ModelOptFp8PbWoLinearMethod
    )
    pbwo_q_b_kernel = k3_gemm.DeepGemmFp8BlockScaledMMKernel.__new__(
        k3_gemm.DeepGemmFp8BlockScaledMMKernel
    )
    pbwo_q_b_kernel._block_m_multiple_plan = {}
    pbwo_q_b_kernel._block_size_multiple_plan = {}
    pbwo_q_b_method.w8a8_block_fp8_linear = pbwo_q_b_kernel
    root.pbwo_q_b = FakeLinear(pbwo_q_b_method, 18432, 1536)
    pbwo_qkv_method = k3_gemm.ModelOptFp8PbWoLinearMethod.__new__(
        k3_gemm.ModelOptFp8PbWoLinearMethod
    )
    pbwo_qkv_kernel = k3_gemm.DeepGemmFp8BlockScaledMMKernel.__new__(
        k3_gemm.DeepGemmFp8BlockScaledMMKernel
    )
    pbwo_qkv_kernel._block_m_multiple_plan = {}
    pbwo_qkv_kernel._block_size_multiple_plan = {}
    pbwo_qkv_method.w8a8_block_fp8_linear = pbwo_qkv_kernel
    root.pbwo_qkv = FakeLinear(pbwo_qkv_method, 2176, 7168)
    pbwo_fb_method = k3_gemm.ModelOptFp8PbWoLinearMethod.__new__(
        k3_gemm.ModelOptFp8PbWoLinearMethod
    )
    pbwo_fb_kernel = k3_gemm.DeepGemmFp8BlockScaledMMKernel.__new__(
        k3_gemm.DeepGemmFp8BlockScaledMMKernel
    )
    pbwo_fb_kernel.config = object()
    pbwo_fb_method.w8a8_block_fp8_linear = pbwo_fb_kernel
    root.pbwo_fb = FakeLinear(pbwo_fb_method, 12288, 128)
    # cute shape.
    root.cute = FakeLinear(k3_gemm.UnquantizedLinearMethod(), 6288, 7168)
    # cute + residual shape.
    root.residual = FakeLinear(k3_gemm.UnquantizedLinearMethod(), 7168, 3584)
    root.dense_gate_up = FakeLinear(k3_gemm.UnquantizedLinearMethod(), 67584, 7168)
    # shape absent from the table: must be left untouched.
    root.unlisted = FakeLinear(k3_gemm.UnquantizedLinearMethod(), 1234, 5678)
    root.lm_head = FakeHead(20480, 7168)

    monkeypatch.setattr(k3_gemm, "LinearBase", FakeLinear)
    monkeypatch.setattr(k3_gemm, "ParallelLMHead", FakeHead)
    monkeypatch.setattr(k3_gemm, "_is_sm103", lambda: True)
    monkeypatch.setattr(k3_gemm, "supports_block_size_multiple_of", lambda: True)

    class FakeFusedK128Kernel:
        def __init__(self, config: object) -> None:
            self.config = config

    monkeypatch.setattr(
        k3_gemm,
        "KimiK3FusedK128Fp8LinearKernel",
        FakeFusedK128Kernel,
    )
    warmup_configs: set[SkinnyGemmConfig] = set()
    residual_warmup_configs: set[SkinnyGemmConfig] = set()
    monkeypatch.setattr(k3_gemm.shape_dynamic_skinny_gemm, "is_available", lambda: True)

    def request_warmup_configs(dtype, configs, *, has_residual=False):
        target = residual_warmup_configs if has_residual else warmup_configs
        target.update(configs)

    monkeypatch.setattr(
        k3_gemm.shape_dynamic_skinny_gemm,
        "request_warmup_configs",
        request_warmup_configs,
    )

    k3_gemm.enable_kimi_k3_low_latency_gemm(root, torch.bfloat16)

    assert isinstance(root.dsv3_only.quant_method, k3_gemm.KimiK3LowLatencyLinearMethod)
    assert isinstance(root.cute.quant_method, k3_gemm.KimiK3LowLatencyLinearMethod)
    assert isinstance(root.residual.quant_method, k3_gemm.KimiK3LowLatencyLinearMethod)
    assert isinstance(
        root.dense_gate_up.quant_method,
        k3_gemm.KimiK3LowLatencyLinearMethod,
    )
    assert root.quantized.quant_method is quantized_method
    assert (
        pbwo_kernel._block_m_multiple_plan
        == (k3_gemm.KIMI_K3_FP8_PB_WO_BLOCK_M_PLANS[(7168, 12288)])
    )
    assert (
        pbwo_q_b_kernel._block_m_multiple_plan
        == (k3_gemm.KIMI_K3_FP8_PB_WO_BLOCK_M_PLANS[(18432, 1536)])
    )
    assert (
        pbwo_qkv_kernel._block_size_multiple_plan
        == (k3_gemm.KIMI_K3_FP8_PB_WO_BLOCK_SIZE_PLANS[(2176, 7168)])
    )
    installed_fb_kernel = root.pbwo_fb.quant_method.w8a8_block_fp8_linear
    assert isinstance(installed_fb_kernel, FakeFusedK128Kernel)
    assert installed_fb_kernel.config is pbwo_fb_kernel.config
    assert type(root.unlisted.quant_method) is k3_gemm.UnquantizedLinearMethod
    assert isinstance(
        root.lm_head.quant_method, k3_gemm.KimiK3LowLatencyEmbeddingMethod
    )
    # Warmup covers only the installed modules' local (N, K).
    assert warmup_configs == {
        config
        for key in (
            (6288, 7168),
            (7168, 3584),
            (20480, 7168),
            (67584, 7168),
        )
        for _, config in k3_gemm.KIMI_K3_PROJECTIONS[key].cute_configs
    }
    assert residual_warmup_configs == {
        config
        for _, config in k3_gemm.KIMI_K3_PROJECTIONS[(7168, 3584)].residual_configs
    }

    pbwo_kernel._block_m_multiple_plan = {}
    pbwo_q_b_kernel._block_m_multiple_plan = {}
    pbwo_qkv_kernel._block_size_multiple_plan = {}
    monkeypatch.setattr(k3_gemm, "supports_block_size_multiple_of", lambda: False)
    k3_gemm.enable_kimi_k3_low_latency_gemm(root, torch.bfloat16)
    assert pbwo_kernel._block_m_multiple_plan == {}
    assert pbwo_q_b_kernel._block_m_multiple_plan == {}
    assert pbwo_qkv_kernel._block_size_multiple_plan == {}


def test_deep_gemm_layout_constraint_is_forwarded_per_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel = deep_gemm_kernel.DeepGemmFp8BlockScaledMMKernel.__new__(
        deep_gemm_kernel.DeepGemmFp8BlockScaledMMKernel
    )
    kernel.config = SimpleNamespace(out_dtype=torch.bfloat16)
    kernel.use_deep_gemm_e8m0 = True
    kernel._block_m_multiple_plan = {}
    kernel._block_size_multiple_plan = {}
    kernel.set_block_size_multiple_plan({32: (256, 64)})

    constraints: list[tuple[int, int]] = []

    def record_launch(
        _A: torch.Tensor,
        _As: torch.Tensor,
        _B: torch.Tensor,
        _Bs: torch.Tensor,
        _output: torch.Tensor,
        _use_deep_gemm_e8m0: bool,
        block_m_multiple_of: int,
        block_n_multiple_of: int,
    ) -> None:
        constraints.append((block_m_multiple_of, block_n_multiple_of))

    monkeypatch.setattr(deep_gemm_kernel, "_launch_fp8_gemm_nt", record_launch)
    A = torch.empty((32, 128), dtype=torch.float8_e4m3fn)
    B = torch.empty((128, 128), dtype=torch.float8_e4m3fn)
    As = torch.empty((32, 1), dtype=torch.int32)
    Bs = torch.empty((1, 1), dtype=torch.int32)

    kernel.apply_block_scaled_mm(A, B, As, Bs)
    kernel.apply_block_scaled_mm(A[:31], B, As[:31], Bs)

    assert constraints == [(256, 64), (1, 1)]


def test_deep_gemm_layout_constraint_is_restored_after_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constraints: list[tuple[int, int]] = []
    fake_deep_gemm = SimpleNamespace(
        set_block_size_multiple_of=lambda value: constraints.append(value)
    )
    monkeypatch.setattr(deep_gemm_utils, "_lazy_init", lambda: None)
    monkeypatch.setattr(deep_gemm_utils, "_import_deep_gemm", lambda: fake_deep_gemm)
    monkeypatch.setattr(
        deep_gemm_utils, "supports_block_size_multiple_of", lambda: True
    )

    def fail_launch(*args: object, **kwargs: object) -> None:
        raise RuntimeError("launch failed")

    monkeypatch.setattr(deep_gemm_utils, "_fp8_gemm_nt_impl", fail_launch)

    with pytest.raises(RuntimeError, match="launch failed"):
        deep_gemm_utils.fp8_gemm_nt(
            block_size_multiple_of=(256, 1),
            is_deep_gemm_e8m0_used=True,
        )

    assert constraints == [(256, 1), (1, 1)]


@pytest.mark.parametrize(
    "dtype,platform_enabled",
    [(torch.float16, True), (torch.bfloat16, False)],
)
def test_installation_requires_bf16_sm103(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
    platform_enabled: bool,
) -> None:
    class FakeLinear(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.quant_method = k3_gemm.UnquantizedLinearMethod()
            self.weight = torch.empty(2304, 1536)

    root = nn.Module()
    root.projection = FakeLinear()
    monkeypatch.setattr(k3_gemm, "LinearBase", FakeLinear)
    monkeypatch.setattr(k3_gemm, "_is_sm103", lambda: platform_enabled)

    k3_gemm.enable_kimi_k3_low_latency_gemm(root, dtype)

    assert type(root.projection.quant_method) is k3_gemm.UnquantizedLinearMethod


def _require_sm103_and_dsv3() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("Kimi-K3 production selection requires SM103")
    if not hasattr(torch.ops._C, "dsv3_fused_a_gemm"):
        pytest.skip("dsv3_fused_a_gemm was not built")


def _require_sm103_and_cute() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("Kimi-K3 production selection requires SM103")
    if not k3_gemm.shape_dynamic_skinny_gemm.is_available():
        pytest.skip("CuTe DSL is not available")


def _fused_k128_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    x_float = x.float()
    x_scale = (x_float.abs().amax(dim=1) / 448.0).clamp_min(1e-10)
    x_scale = torch.pow(2.0, torch.ceil(torch.log2(x_scale)))
    x_quant = (x_float / x_scale[:, None]).clamp(-448.0, 448.0).to(weight.dtype)
    output = x_quant.float() @ weight.float().t()
    return output * x_scale[:, None] * weight_scale[:, 0].repeat_interleave(128)


@pytest.mark.parametrize("num_tokens", [1, 2, 3, 16, 48, 63, 64])
def test_kimi_k3_fused_k128_fp8_gemm(num_tokens: int) -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("Kimi-K3 fused K=128 GEMM requires SM103")
    torch.manual_seed(42)
    storage = torch.randn(
        num_tokens,
        49408,
        dtype=torch.bfloat16,
        device="cuda",
    )
    x = storage[:, -128:]
    weight = torch.randn(12288, 128, dtype=torch.bfloat16, device="cuda").to(
        torch.float8_e4m3fn
    )
    weight_scale = torch.rand(96, 1, dtype=torch.float32, device="cuda") + 0.5

    output = k3_gemm.fused_k128_fp8_gemm(x, weight, weight_scale)
    reference = _fused_k128_reference(x, weight, weight_scale)

    assert x.stride(0) == 49408
    torch.testing.assert_close(output.float(), reference, rtol=2e-2, atol=2e-1)


def test_kimi_k3_fused_k128_fp8_gemm_cuda_graph() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("Kimi-K3 fused K=128 GEMM requires SM103")
    x = torch.randn(64, 128, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(12288, 128, dtype=torch.bfloat16, device="cuda").to(
        torch.float8_e4m3fn
    )
    weight_scale = torch.ones(96, 1, dtype=torch.float32, device="cuda")
    k3_gemm.fused_k128_fp8_gemm(x, weight, weight_scale)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = k3_gemm.fused_k128_fp8_gemm(x, weight, weight_scale)
    graph.replay()
    torch.cuda.synchronize()

    reference = _fused_k128_reference(x, weight, weight_scale)
    torch.testing.assert_close(output.float(), reference, rtol=2e-2, atol=2e-1)


@pytest.mark.parametrize("spec,config", GLM_CUTE_CASES)
def test_glm_cute_selected_shapes(
    spec: glm52_gemm.GLM52ProjectionSpec,
    config: SkinnyGemmConfig,
) -> None:
    _require_sm103_and_cute()
    torch.manual_seed(42)
    x = torch.randn(
        config.num_rows,
        spec.k,
        dtype=torch.bfloat16,
        device="cuda",
    )
    weight = torch.randn(spec.n, spec.k, dtype=torch.bfloat16, device="cuda")
    plan = spec.build_plan()

    output = glm52_gemm.run_glm52_plan(plan, x, weight)

    assert output is not None
    reference = x.float() @ weight.float().t()
    torch.testing.assert_close(output.float(), reference, rtol=2e-2, atol=2e-1)


def test_glm52_q_b_nonpacked_single_row_falls_back() -> None:
    _require_sm103_and_cute()
    spec = glm52_gemm.GLM52_Q_B_PROJECTION
    storage = torch.randn(
        1,
        glm52_gemm.GLM52_QKV_A_PROJECTION.n,
        dtype=torch.bfloat16,
        device="cuda",
    )
    x = storage[:, : spec.k]
    weight = torch.randn(spec.n, spec.k, dtype=torch.bfloat16, device="cuda")

    assert x.stride() == (glm52_gemm.GLM52_QKV_A_PROJECTION.n, 1)
    assert not glm52_gemm._runtime_ok(x, weight)
    output = glm52_gemm.run_glm52_plan(spec.build_plan(), x, weight)

    assert output is None


@pytest.mark.parametrize("spec,config", GLM_CUTE_CASES)
def test_glm_cute_selected_shapes_cuda_graph_capture(
    spec: glm52_gemm.GLM52ProjectionSpec,
    config: SkinnyGemmConfig,
) -> None:
    _require_sm103_and_cute()
    x = torch.randn(
        config.num_rows,
        spec.k,
        dtype=torch.bfloat16,
        device="cuda",
    )
    weight = torch.randn(spec.n, spec.k, dtype=torch.bfloat16, device="cuda")
    plan = spec.build_plan()
    glm52_gemm.run_glm52_plan(plan, x, weight)
    torch.accelerator.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = glm52_gemm.run_glm52_plan(plan, x, weight)
    graph.replay()
    torch.accelerator.synchronize()

    assert output is not None
    reference = x.float() @ weight.float().t()
    torch.testing.assert_close(output.float(), reference, rtol=2e-2, atol=2e-1)


@pytest.mark.parametrize("n,k,num_tokens", CUTE_CASES)
def test_cute_selected_shapes(n: int, k: int, num_tokens: int) -> None:
    _require_sm103_and_cute()
    torch.manual_seed(42)
    x = torch.randn(num_tokens, k, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")

    output = k3_gemm.try_low_latency_gemm(x, weight)

    assert output is not None
    reference = torch.nn.functional.linear(x, weight)
    cosine = torch.nn.functional.cosine_similarity(
        output.float().flatten(), reference.float().flatten(), dim=0
    ).item()
    assert cosine > 0.999


def _dsv3_probe_tokens(tokens: frozenset[int]) -> set[int]:
    """Extremes, plus both sides of the kernel's num_tokens<=8 tile_n branch."""
    if not tokens:
        return set()
    return {min(tokens), max(tokens)} | ({8, 9} & set(tokens))


# Derived from the table rather than hand-listed, so a shape routed to dsv3
# cannot be added without being exercised here.
DSV3_CASES = sorted(
    (num_tokens, spec.n, spec.k)
    for spec in KIMI_K3_PROJECTIONS.values()
    for num_tokens in _dsv3_probe_tokens(spec.dsv3_tokens)
)

GLM_DSV3_CASES = [
    (num_tokens, spec)
    for spec in (
        glm52_gemm.GLM52_QKV_A_PROJECTION,
        glm52_gemm.GLM52_Q_B_PROJECTION,
    )
    for num_tokens in sorted(_dsv3_probe_tokens(spec.dsv3_tokens))
]


@pytest.mark.parametrize("num_tokens,n,k", DSV3_CASES)
def test_dsv3_selected_shapes(num_tokens: int, n: int, k: int) -> None:
    _require_sm103_and_dsv3()
    spec = k3_gemm.KIMI_K3_PROJECTIONS[(n, k)]
    assert num_tokens in spec.dsv3_tokens
    torch.manual_seed(42)
    x = torch.randn(num_tokens, k, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")

    output = k3_gemm.try_low_latency_gemm(x, weight)

    assert output is not None
    reference = torch.nn.functional.linear(x, weight)
    cosine = torch.nn.functional.cosine_similarity(
        output.float().flatten(), reference.float().flatten(), dim=0
    ).item()
    assert cosine > 0.999


@pytest.mark.parametrize("num_tokens,spec", GLM_DSV3_CASES)
def test_glm_dsv3_selected_shapes(
    num_tokens: int,
    spec: glm52_gemm.GLM52ProjectionSpec,
) -> None:
    _require_sm103_and_dsv3()
    torch.manual_seed(42)
    x = torch.randn(num_tokens, spec.k, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(spec.n, spec.k, dtype=torch.bfloat16, device="cuda")

    output = glm52_gemm.run_glm52_plan(spec.build_plan(), x, weight)

    assert output is not None
    reference = torch.nn.functional.linear(x, weight)
    cosine = torch.nn.functional.cosine_similarity(
        output.float().flatten(), reference.float().flatten(), dim=0
    ).item()
    assert cosine > 0.999


def test_nonpacked_single_token_dsv3_falls_back() -> None:
    _require_sm103_and_dsv3()
    n, k = 1536, 128
    storage = torch.randn(1, k + 16, dtype=torch.bfloat16, device="cuda")
    x = storage[:, :k]
    weight = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    spec = k3_gemm.KIMI_K3_PROJECTIONS[(n, k)]
    method = k3_gemm.KimiK3LowLatencyLinearMethod(
        k3_gemm._build_plan(spec), k3_gemm._build_residual_plan(spec)
    )

    assert x.is_contiguous()
    assert x.stride() == (k + 16, 1)
    assert not k3_gemm._runtime_ok(x, weight)  # strict guard rejects the view
    output = method.apply(SimpleNamespace(weight=weight), x)

    reference = torch.nn.functional.linear(x, weight)
    torch.testing.assert_close(output, reference)


def test_selected_kernels_cuda_graph_capture() -> None:
    _require_sm103_and_cute()
    _require_sm103_and_dsv3()
    cute_spec = k3_gemm.KIMI_K3_PROJECTIONS[(6288, 7168)]
    dsv3_spec = k3_gemm.KIMI_K3_PROJECTIONS[(1536, 128)]
    cute_x = torch.randn(1, cute_spec.k, dtype=torch.bfloat16, device="cuda")
    cute_weight = torch.randn(
        cute_spec.n, cute_spec.k, dtype=torch.bfloat16, device="cuda"
    )
    dsv3_x = torch.randn(1, dsv3_spec.k, dtype=torch.bfloat16, device="cuda")
    dsv3_weight = torch.randn(
        dsv3_spec.n, dsv3_spec.k, dtype=torch.bfloat16, device="cuda"
    )
    k3_gemm.try_low_latency_gemm(cute_x, cute_weight)
    k3_gemm.try_low_latency_gemm(dsv3_x, dsv3_weight)
    torch.accelerator.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cute_output = k3_gemm.try_low_latency_gemm(cute_x, cute_weight)
        dsv3_output = k3_gemm.try_low_latency_gemm(dsv3_x, dsv3_weight)
    graph.replay()
    torch.accelerator.synchronize()

    assert cute_output is not None
    assert dsv3_output is not None
    for output, activation, weight in (
        (cute_output, cute_x, cute_weight),
        (dsv3_output, dsv3_x, dsv3_weight),
    ):
        reference = torch.nn.functional.linear(activation, weight)
        cosine = torch.nn.functional.cosine_similarity(
            output.float().flatten(), reference.float().flatten(), dim=0
        ).item()
        assert cosine > 0.999


@pytest.mark.parametrize("num_tokens", [1, 8, 9, 16])
def test_dsv3_cuda_graph_capture_tile_branches(num_tokens: int) -> None:
    """Capture DSV3 across the num_tokens<=8 vs >8 tile_n branch."""
    _require_sm103_and_dsv3()
    spec = k3_gemm.KIMI_K3_PROJECTIONS[(1536, 128)]
    x = torch.randn(num_tokens, spec.k, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(spec.n, spec.k, dtype=torch.bfloat16, device="cuda")
    k3_gemm.try_low_latency_gemm(x, weight)
    torch.accelerator.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = k3_gemm.try_low_latency_gemm(x, weight)
    graph.replay()
    torch.accelerator.synchronize()

    assert output is not None
    reference = torch.nn.functional.linear(x, weight)
    cosine = torch.nn.functional.cosine_similarity(
        output.float().flatten(), reference.float().flatten(), dim=0
    ).item()
    assert cosine > 0.999


@pytest.mark.parametrize("n,k,num_tokens", RESIDUAL_CUTE_CASES)
def test_cute_residual_epilogue(n: int, k: int, num_tokens: int) -> None:
    _require_sm103_and_cute()
    torch.manual_seed(42 + num_tokens)
    x = torch.randn(num_tokens, k, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    residual = torch.randn(num_tokens, n, dtype=torch.bfloat16, device="cuda")
    spec = k3_gemm.KIMI_K3_PROJECTIONS[(n, k)]
    config = spec.residual_config(num_tokens)
    assert config is not None

    output = k3_gemm.shape_dynamic_skinny_gemm(x, weight, config, residual)

    reference = x.float() @ weight.float().t() + residual.float()
    cosine = torch.nn.functional.cosine_similarity(
        output.float().flatten(), reference.flatten(), dim=0
    ).item()
    assert cosine > 0.999


@pytest.mark.parametrize("num_tokens", range(1, 17))
def test_cute_residual_epilogue_all_supported_token_counts(num_tokens: int) -> None:
    _require_sm103_and_cute()
    from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import (
        ShapeDynamicSkinnyGemm,
    )

    n, k = 64, 512
    x = torch.randn(num_tokens, k, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    residual = torch.randn(num_tokens, n, dtype=torch.bfloat16, device="cuda")
    config = ShapeDynamicSkinnyGemm._config(num_tokens, n, k)

    output = k3_gemm.shape_dynamic_skinny_gemm(x, weight, config, residual)

    reference = x.float() @ weight.float().t() + residual.float()
    torch.testing.assert_close(output.float(), reference, rtol=2e-2, atol=2e-1)


@pytest.mark.parametrize("num_tokens", range(1, 5))
def test_cute_residual_epilogue_cuda_graph_capture(num_tokens: int) -> None:
    _require_sm103_and_cute()
    spec = k3_gemm.KIMI_K3_PROJECTIONS[(7168, 3584)]
    config = spec.residual_config(num_tokens)
    assert config is not None
    x = torch.randn(num_tokens, spec.k, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(spec.n, spec.k, dtype=torch.bfloat16, device="cuda")
    residual = torch.randn(num_tokens, spec.n, dtype=torch.bfloat16, device="cuda")
    k3_gemm.shape_dynamic_skinny_gemm(x, weight, config, residual)
    torch.accelerator.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = k3_gemm.shape_dynamic_skinny_gemm(x, weight, config, residual)
    graph.replay()
    torch.accelerator.synchronize()

    reference = x.float() @ weight.float().t() + residual.float()
    cosine = torch.nn.functional.cosine_similarity(
        output.float().flatten(), reference.flatten(), dim=0
    ).item()
    assert cosine > 0.999


class _SkinnyGemmSpy:
    """Wraps the skinny-GEMM singleton to record whether CuTe was invoked."""

    def __init__(self, real: Any) -> None:
        self._real = real
        self.calls: list[int] = []

    def __call__(self, a, b, config=None, residual=None):
        self.calls.append(a.shape[0])
        return self._real(a, b, config, residual)

    def is_available(self) -> bool:
        return self._real.is_available()


@pytest.mark.parametrize("num_tokens", [1, 2, 3, 4])
def test_latent_moe_production_layout_residual(
    monkeypatch: pytest.MonkeyPatch,
    num_tokens: int,
) -> None:
    """The real Latent-MoE residual is a non-packed slice of a cat buffer.

    The strict packed-row-major guard rejects such a slice at every token count
    (a size-1 leading dim reads as contiguous but its stride is not packed), so
    the CuTe residual epilogue never fires for this production layout and the
    method falls back to addmm. Output is correct regardless of the path.
    """
    _require_sm103_and_cute()
    latent_dim, shared_dim = 3584, 7168  # routed_expert_up_proj K, N
    torch.manual_seed(7 + num_tokens)
    buf = torch.randn(
        num_tokens, latent_dim + shared_dim, dtype=torch.bfloat16, device="cuda"
    )
    latent = buf[:, :latent_dim]  # non-contiguous view (row stride = full width)
    residual = buf[:, latent_dim:]  # non-contiguous view
    weight = torch.randn(shared_dim, latent_dim, dtype=torch.bfloat16, device="cuda")

    spec = k3_gemm.KIMI_K3_PROJECTIONS[(shared_dim, latent_dim)]
    method = k3_gemm.KimiK3LowLatencyLinearMethod(
        k3_gemm._build_plan(spec), k3_gemm._build_residual_plan(spec)
    )
    spy = _SkinnyGemmSpy(k3_gemm.shape_dynamic_skinny_gemm)
    monkeypatch.setattr(k3_gemm, "shape_dynamic_skinny_gemm", spy)

    layer = SimpleNamespace(weight=weight)
    output = method.apply_with_residual(layer, latent, residual)

    reference = latent.float() @ weight.float().t() + residual.float()
    cosine = torch.nn.functional.cosine_similarity(
        output.float().flatten(), reference.flatten(), dim=0
    ).item()
    assert cosine > 0.999  # correct regardless of the path taken
    assert not spy.calls, (
        "non-packed buf-slice residual must fall back to addmm at every M"
    )


def test_residual_dispatch_falls_back_to_addmm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fallback = torch.randn(2, 3)
    residual = torch.randn(2, 3)
    x = torch.randn(2, 4)
    weight = torch.randn(3, 4)
    monkeypatch.setattr(torch, "addmm", lambda *args: fallback)
    # CPU tensors fail the runtime check, forcing the addmm fallback.
    method = k3_gemm.KimiK3LowLatencyLinearMethod({}, {})

    output = method.apply_with_residual(SimpleNamespace(weight=weight), x, residual)

    assert output is fallback


def test_fallback_preserves_default_method(monkeypatch: pytest.MonkeyPatch) -> None:
    fallback = torch.empty(2, 8)
    monkeypatch.setattr(
        k3_gemm.UnquantizedLinearMethod,
        "apply",
        lambda *args: fallback,
    )
    # 1-D input fails the runtime check, forcing the base-method fallback.
    method = k3_gemm.KimiK3LowLatencyLinearMethod({}, {})

    output = method.apply(
        SimpleNamespace(weight=torch.empty(0)),
        torch.empty(0),
    )

    assert output is fallback
