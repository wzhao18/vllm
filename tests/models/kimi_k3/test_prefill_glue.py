# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 prefill glue kernels (ops/prefill_glue.py): each one is bitwise the
stock kernel it replaces, at the prefill shapes and on edge inputs.

* router_topk == _moe_C.grouped_topk (sigmoid, one group): ties, non-finite
  logits, renormalize on / off;
* rms_norm == _C.rms_norm (the stock reduction tree replayed), strided rows;
* gate_sigmoid_mul == eager ``a * g.sigmoid()`` for every BF16 gate value;
* stage_rows == megamoe_rubin staging's Triton ``_stage_kernel`` (MXFP8 and
  NVFP4): special values, padding mask, int64 ids, BF16 weights, dead rows,
  route errors.
"""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not (current_platform.is_cuda() and current_platform.has_device_capability(100)),
    reason="Kimi K3 prefill glue targets SM100+",
)


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8)


@pytest.fixture(scope="module")
def pg():
    from vllm.models.kimi_k3.nvidia.ops import prefill_glue

    prefill_glue.warmup()
    return prefill_glue


@pytest.mark.parametrize("rows", [1, 7, 300, 4350])
@pytest.mark.parametrize("kind", ["rand", "ties", "nonfinite", "zeros"])
@pytest.mark.parametrize("renormalize", [True, False])
def test_router_topk_matches_grouped_topk(pg, rows, kind, renormalize):
    from vllm import _custom_ops as ops

    torch.manual_seed(rows)
    experts, topk = 896, 16
    x = torch.randn(rows, experts, device="cuda") * 2
    bias = torch.randn(experts, device="cuda") * 0.05
    if kind == "ties":
        x = (x * 4).round() / 4
        bias = (bias * 64).round() / 64
    elif kind == "nonfinite":
        x[::7, ::13] = float("inf")
        x[::5, ::11] = float("nan")
        x[1::9, 5::17] = -float("inf")
    elif kind == "zeros":
        x.zero_()
        bias.zero_()
    v0, i0 = ops.grouped_topk(x, 1, 1, topk, renormalize, 2.5, bias, 1)
    v1, i1 = pg.router_topk(x, bias, topk, renormalize, 2.5)
    torch.testing.assert_close(i1, i0, rtol=0, atol=0)
    assert torch.equal(_bits(v1), _bits(v0))


@pytest.mark.parametrize(
    "rows,width,cols",
    [
        (4350, 3584, (0, 3584)),  # latent norm, C512 n17k
        (8192, 3584, (0, 3584)),  # latent norm, C512 n32k
        (17400, 2112, (0, 1536)),  # q_a norm on the fused qkv_a output
        (17400, 2112, (1536, 512)),  # kv_a norm on the fused qkv_a output
        (2048, 1536, (0, 1536)),
    ],
)
def test_rms_norm_matches_vllm_c(pg, rows, width, cols):
    from vllm import _custom_ops as ops

    torch.manual_seed(rows + width)
    base = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16) * 3
    base[0] = 0
    base[1, :8] = 3.0e38
    x = base[:, cols[0] : cols[0] + cols[1]]
    w = (torch.rand(cols[1], device="cuda") + 0.5).to(torch.bfloat16)
    assert pg.use_rms_norm(x, w)
    ref = torch.empty(x.shape, dtype=x.dtype, device="cuda")
    ops.rms_norm(ref, x, w, 1e-6)
    assert torch.equal(_bits(pg.rms_norm(x, w, 1e-6)), _bits(ref))


def test_rows_rms_norm_module(pg):
    from vllm.config import VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig()):
        from vllm.model_executor.layers.layernorm import RMSNorm

        norm = pg.KimiRowsRMSNorm(3584, eps=1e-5).cuda().to(torch.bfloat16)
        ref = RMSNorm(3584, eps=1e-5).cuda().to(torch.bfloat16)
    with torch.no_grad():
        w = torch.rand(3584, device="cuda").to(torch.bfloat16) + 0.5
        norm.weight.copy_(w)
        ref.weight.copy_(w)
    for rows in (64, 4350):
        x = torch.randn(rows, 3584, device="cuda", dtype=torch.bfloat16)
        assert torch.equal(_bits(norm(x)), _bits(ref(x)))


def test_gate_sigmoid_mul_every_gate_value(pg):
    g = torch.arange(65536, dtype=torch.int32, device="cuda").to(torch.int16)
    g = g.view(torch.bfloat16).reshape(1024, 64)
    a = (torch.randn(1024, 64, device="cuda") * 4).to(torch.bfloat16)
    ref = a * g.sigmoid()
    out = pg.gate_sigmoid_mul(a, g)
    nan = torch.isnan(ref)
    assert torch.equal(nan, torch.isnan(out))
    assert torch.equal(_bits(out[~nan]), _bits(ref[~nan]))


@pytest.mark.parametrize("contiguous", [True, False])
def test_gate_sigmoid_mul_prefill_shape(pg, contiguous):
    rows, d = 4096, 3072
    if contiguous:
        a = torch.randn(rows, d, device="cuda", dtype=torch.bfloat16)
        g = torch.randn(rows, d, device="cuda", dtype=torch.bfloat16)
    else:
        base = torch.randn(rows, 2 * d, device="cuda", dtype=torch.bfloat16)
        a, g = base[:, :d], base[:, d:]
    assert pg.use_gate_sigmoid_mul(a, g)
    assert torch.equal(_bits(pg.eager_gate_sigmoid_mul(a, g)), _bits(a * g.sigmoid()))


@pytest.mark.parametrize("rows", [1024, 4350])
@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("w_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("with_pad", [False, True])
@pytest.mark.parametrize(
    "quant_kind,norm_const",
    [("mxfp4_mxfp8", 1.0), ("nvfp4", 1.0), ("nvfp4", 2688.0)],
)
def test_stage_rows_matches_triton(
    pg, rows, ids_dtype, w_dtype, with_pad, quant_kind, norm_const
):
    from vllm.models.kimi_k3.nvidia.megamoe_rubin import staging

    torch.manual_seed(rows)
    hidden, topk, experts, cap = 3584, 16, 896, 8192
    ids = torch.randint(0, experts, (rows, topk), device="cuda").to(ids_dtype)
    ids[7, 3] = -1
    wts = torch.rand(rows, topk, device="cuda").to(w_dtype)
    hid = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16) * 3
    hid[0, :64] = 3.0e38
    hid[1] = (torch.randn(hidden, device="cuda") * 1e-38).to(torch.bfloat16)
    hid[2, 5] = float("inf")
    hid[3, 7] = float("nan")
    hid[4, ::2] = 1e30
    hid[4, 1::2] = -1e-30
    hid[5] = -0.0
    pad = None
    if with_pad:
        pad = torch.zeros(rows, dtype=torch.bool, device="cuda")
        pad[::3] = True

    nvfp4 = quant_kind == "nvfp4"

    def run(rows_stager: bool):
        x_cols = hidden // 2 if nvfp4 else hidden
        sf_cols = hidden // (16 if nvfp4 else 32) + 4
        x = torch.full((cap, x_cols), 7, dtype=torch.uint8, device="cuda")
        sf = torch.full((cap, sf_cols), 7, dtype=torch.uint8, device="cuda")
        idx = torch.full((cap, topk), 7, dtype=torch.int32, device="cuda")
        sw = torch.full((cap, topk), 7.0, dtype=torch.float32, device="cuda")
        err = torch.zeros(2, dtype=torch.int64, device="cuda")
        old = staging.ROWS_STAGER_MIN_TOKENS
        staging.ROWS_STAGER_MIN_TOKENS = 1 if rows_stager else 1 << 40
        try:
            staging.stage_inputs(
                hid, ids, wts, x, sf, idx, sw, route_error=err,
                quant_kind=quant_kind, num_experts=experts,
                input_norm_const=norm_const, is_padding=pad,
            )  # fmt: skip
        finally:
            staging.ROWS_STAGER_MIN_TOKENS = old
        return x[:rows], sf[:rows], idx, sw[:rows], err

    for a, b in zip(run(True), run(False)):
        assert torch.equal(_bits(a), _bits(b))


def test_mxfp8_stage_route_error(pg):
    from vllm.models.kimi_k3.nvidia.megamoe_rubin import staging

    rows, hidden, topk, experts = 2048, 3584, 16, 896
    ids = torch.randint(0, experts, (rows, topk), device="cuda", dtype=torch.int32)
    ids[100, 2] = experts + 5
    hid = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    wts = torch.rand(rows, topk, device="cuda")
    x = torch.empty((rows, hidden), dtype=torch.uint8, device="cuda")
    sf = torch.empty((rows, hidden // 32), dtype=torch.uint8, device="cuda")
    idx = torch.empty((rows, topk), dtype=torch.int32, device="cuda")
    sw = torch.empty((rows, topk), dtype=torch.float32, device="cuda")
    err = torch.zeros(2, dtype=torch.int64, device="cuda")
    staging.stage_inputs(
        hid, ids, wts, x, sf, idx, sw, route_error=err,
        quant_kind="mxfp4_mxfp8", num_experts=experts,
    )  # fmt: skip
    assert err.tolist() == [1, experts + 5]
    assert int(idx[100, 2]) == -1 and float(sw[100, 2]) == 0.0


@pytest.mark.parametrize("idx_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("rows", [1, 5, 64])
def test_write_state_rows_matches_index_put(pg, idx_dtype, rows):
    slots, heads, dk, dv = 300, 24, 128, 128
    dst0 = torch.randn(slots, heads, dk, dv, device="cuda")
    src = torch.randn(rows, heads, dk, dv, device="cuda")
    idx = torch.randperm(slots - 1, device="cuda")[:rows].to(idx_dtype)
    if rows > 1:
        idx[0] = -1  # wraps to the last slot, as index_put_
    ref = dst0.clone()
    ref[idx] = src
    out = dst0.clone()
    pg.write_state_rows(out, idx, src)
    assert torch.equal(out, ref)
