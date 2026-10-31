# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 MXFP8 activation producers (ops/mxfp8_act.py, ops/kda_norm.py):
each fused producer equals "BF16 producer, then FlashInfer mxfp8_quantize",
and an MXFP8 linear fed the produced activation equals the same linear fed the
BF16 activation."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not (current_platform.is_cuda() and current_platform.has_device_capability(100)),
    reason="MXFP8 producers target SM100+",
)

ROWS = [1, 7, 16, 64, 129, 300]


def _fi_quant(x):
    """FlashInfer CuTe DSL MXFP8 quantization, as the MXFP8 linears do it."""
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
    )

    return mxfp8_e4m3_quantize(x.contiguous(), is_sf_swizzled_layout=True)


def _rows_scales(sf, rows, k):
    from vllm.models.kimi_k3.nvidia.mxfp8 import unswizzle_mxfp8_scale

    return unswizzle_mxfp8_scale(sf, rows, k)


def _assert_same_quant(qa, ref_bf16, *, allow_ulps=0, check_pad=True):
    """``qa`` (QuantizedActivation) equals mxfp8_quantize(ref_bf16): FP8 bytes
    and the real rows' E8M0 scales; padding rows' scales are zero."""
    rows, k = ref_bf16.shape
    q_ref, sf_ref = _fi_quant(ref_bf16)
    assert qa.data.shape == (rows, k) and qa.data.dtype == torch.float8_e4m3fn
    got_q = qa.data.view(torch.uint8)
    ref_q = q_ref.view(torch.uint8)
    got_s = _rows_scales(qa.scale, rows, k)
    ref_s = _rows_scales(sf_ref, rows, k)
    if allow_ulps == 0:
        assert torch.equal(got_s, ref_s)
        assert torch.equal(got_q, ref_q)
    else:
        # The RMSNorm reduction order differs from the BF16 kernel's: allow a
        # few elements one BF16 ulp apart before quantization.
        assert (got_s.int() - ref_s.int()).abs().max() <= 1
        bad = (got_q != ref_q).float().mean().item()
        assert bad < 1e-3, bad
    if check_pad:
        pad = (rows + 127) // 128 * 128
        full = _rows_scales(qa.scale, pad, k)
        assert torch.all(full[rows:] == 0)


def _special_rows(k, device):
    """Rows that hit the E8M0 edges: block amax at 448 * 2^j (exact), zeros,
    subnormal BF16, huge values."""
    rows = [
        torch.full((k,), 448.0),
        torch.full((k,), 448.0 * 2.0**-20),
        torch.zeros(k),
        torch.full((k,), 1e-40),
        torch.full((k,), 3.0e38),
        torch.full((k,), -0.0),
    ]
    out = torch.stack(rows).to(torch.bfloat16)
    out[:, 1::7] *= -0.37
    return out.to(device)


@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("k", [1536, 3584, 7168, 12288])
def test_quantize_matches_flashinfer(rows, k):
    from vllm.models.kimi_k3.nvidia.ops import mxfp8_act

    torch.manual_seed(rows * 7 + k)
    x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16) * 3
    x[0, :32] *= 1e-3
    _assert_same_quant(mxfp8_act.quantize(x), x)


def test_quantize_edge_values():
    from vllm.models.kimi_k3.nvidia.ops import mxfp8_act

    x = _special_rows(7168, "cuda")
    _assert_same_quant(mxfp8_act.quantize(x), x)


def test_quantize_strided_rows():
    from vllm.models.kimi_k3.nvidia.ops import mxfp8_act

    base = torch.randn(33, 9728, device="cuda", dtype=torch.bfloat16)
    x = base[:, 1024 : 1024 + 7168]
    _assert_same_quant(mxfp8_act.quantize(x), x.contiguous())


@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("d", [6144, 1536, 768])
@pytest.mark.parametrize("linear_beta", [None, 10.0])
def test_situ_and_mul_quantize_matches_unfused(rows, d, linear_beta):
    from vllm.models.kimi_k3.nvidia.ops import mxfp8_act

    torch.manual_seed(rows + d)
    x = torch.randn(rows, 2 * d, device="cuda", dtype=torch.bfloat16) * 4
    ref = torch.empty(rows, d, device="cuda", dtype=torch.bfloat16)
    torch.ops._C.situ_and_mul(ref, x, 1.0, -1.0 if linear_beta is None else linear_beta)
    qa = mxfp8_act.situ_and_mul_quantize(x, 1.0, linear_beta)
    _assert_same_quant(qa, ref)


@pytest.mark.parametrize("rows", ROWS)
def test_silu_and_mul_quantize_matches_unfused(rows):
    from vllm.models.kimi_k3.nvidia.ops import mxfp8_act

    d = 6144
    x = torch.randn(rows, 2 * d, device="cuda", dtype=torch.bfloat16) * 4
    ref = torch.empty(rows, d, device="cuda", dtype=torch.bfloat16)
    torch.ops._C.silu_and_mul(ref, x)
    _assert_same_quant(mxfp8_act.silu_and_mul_quantize(x), ref)


@pytest.mark.parametrize("rows", ROWS)
def test_gate_sigmoid_mul_quantize_matches_eager(rows):
    from vllm.models.kimi_k3.nvidia.mla import _gate_sigmoid_mul
    from vllm.models.kimi_k3.nvidia.ops import mxfp8_act

    a = torch.randn(rows, 12288, device="cuda", dtype=torch.bfloat16)
    g = torch.randn(rows, 12288, device="cuda", dtype=torch.bfloat16) * 3
    # The compiled reference computes a * sigmoid(g) in FP32 and rounds once;
    # its exp may differ from expf by an FP32 ulp (rarely one BF16 ulp).
    _assert_same_quant(
        mxfp8_act.gate_sigmoid_mul_quantize(a, g), _gate_sigmoid_mul(a, g),
        allow_ulps=1,
    )


@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("k", [1536, 3584, 7168])
def test_rms_norm_quantize_matches_rmsnorm(default_vllm_config, rows, k):
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.models.kimi_k3.nvidia.ops import mxfp8_act

    torch.manual_seed(rows + k)
    norm = RMSNorm(k, eps=1e-6).cuda().to(torch.bfloat16)
    norm.weight.data.normal_(1.0, 0.2)
    x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16) * 2
    ref = norm(x)
    y = torch.empty_like(x)
    qa = mxfp8_act.rms_norm_quantize(x, norm.weight, 1e-6, bf16_out=y)
    torch.testing.assert_close(y, ref, rtol=1e-2, atol=1e-2)
    assert (y != ref).float().mean().item() < 1e-2
    _assert_same_quant(qa, y)  # exactly the quantization of its BF16 output
    _assert_same_quant(qa, ref, allow_ulps=1)


@pytest.mark.parametrize("rows", [1, 64, 200])
def test_k_concat_buffer(rows):
    """Two producers filling column windows of one activation equal the
    per-part quantizations (blocks never straddle the 128-aligned seam)."""
    from vllm.models.kimi_k3.nvidia.ops import mxfp8_act

    x = torch.randn(rows, 2 * 6144, device="cuda", dtype=torch.bfloat16)
    lat = torch.randn(rows, 3584, device="cuda", dtype=torch.bfloat16)
    w = torch.rand(3584, device="cuda", dtype=torch.bfloat16) + 0.5
    buf = mxfp8_act.Mxfp8Buffer(rows, 6144 + 3584, "cuda")
    mxfp8_act.situ_and_mul_quantize(x, 1.0, None, out=buf, col=0)
    y = torch.empty_like(lat)
    mxfp8_act.rms_norm_quantize(lat, w, 1e-6, out=buf, col=6144, bf16_out=y)
    ref_act = torch.empty(rows, 6144, device="cuda", dtype=torch.bfloat16)
    torch.ops._C.situ_and_mul(ref_act, x, 1.0, -1.0)
    _assert_same_quant(buf.activation(), torch.cat([ref_act, y], dim=-1))


def _online_linear(n, k, weight):
    from types import SimpleNamespace

    from vllm.config import get_current_vllm_config
    from vllm.config.quantization import QuantizationConfigArgs
    from vllm.model_executor.layers.linear import ReplicatedLinear
    from vllm.model_executor.layers.quantization.online.base import (
        OnlineQuantizationConfig,
    )
    from vllm.models.kimi_k3.quant_presets import K31_DEP_MXFP8_TARGETS

    config = get_current_vllm_config()
    if config.model_config is None:
        config.model_config = SimpleNamespace(
            dtype=torch.bfloat16, quantization_config=None
        )
    quant = OnlineQuantizationConfig(
        QuantizationConfigArgs(targets=K31_DEP_MXFP8_TARGETS)
    )
    layer = ReplicatedLinear(
        k,
        n,
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=quant,
        prefix="model.layers.4.block_sparse_moe.shared_experts.down_proj",
    )
    if layer.weight.is_meta:
        layer = layer.to_empty(device="cuda")
    layer = layer.to("cuda")
    layer.weight.data.copy_(weight)
    layer.quant_method.process_weights_after_loading(layer)
    from vllm.models.kimi_k3.nvidia.mxfp8 import enable_kimi_k3_mxfp8_dispatch

    enable_kimi_k3_mxfp8_dispatch(layer)
    return layer


@pytest.mark.parametrize("rows", [1, 16, 17, 64, 200])
def test_linear_consumes_produced_activation(default_vllm_config, dist_init, rows):
    """An online-MXFP8 linear (SM107 dispatch / CuTe DSL) gives the same output
    for the produced activation as for the BF16 input it quantizes itself."""
    from vllm.models.kimi_k3.nvidia.ops import mxfp8_act

    n, k = 7168, 6144
    torch.manual_seed(rows)
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
    layer = _online_linear(n, k, weight)
    x = torch.randn(rows, 2 * k, device="cuda", dtype=torch.bfloat16)
    act = torch.empty(rows, k, device="cuda", dtype=torch.bfloat16)
    torch.ops._C.situ_and_mul(act, x, 1.0, -1.0)
    ref = layer(act)[0]
    got = layer(mxfp8_act.situ_and_mul_quantize(x, 1.0, None))[0]
    assert torch.equal(got, ref)


@pytest.mark.parametrize("tokens", [1, 3, 64, 100])
@pytest.mark.parametrize("heads", [96, 24, 12])
def test_kda_gated_norm_mxfp8_matches_inplace(tokens, heads):
    from vllm.models.kimi_k3.nvidia.ops import kda_norm

    torch.manual_seed(tokens + heads)
    D = 128
    x = torch.randn(tokens * heads, D, device="cuda", dtype=torch.bfloat16)
    proj = torch.randn(tokens, heads * D + 64, device="cuda", dtype=torch.bfloat16)
    g = proj[:, : heads * D].view(tokens, heads, D)
    w = (torch.rand(D, device="cuda") + 0.5).to(torch.bfloat16)
    ref = x.clone()
    kda_norm.rmsnorm_gated_(ref, g, w, 1e-5)
    qa = kda_norm.rmsnorm_gated_mxfp8(x, g, w, 1e-5)
    _assert_same_quant(qa, ref.view(tokens, heads * D), check_pad=False)
