# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 prefill glue kernels (ops/prefill_glue.py): each one is bitwise the
stock kernel it replaces, at the prefill shapes and on edge inputs.

* router_topk == _moe_C.grouped_topk (sigmoid, one group): ties, non-finite
  logits, renormalize on / off;
* rms_norm == _C.rms_norm (the stock reduction tree replayed), strided rows;
* gate_sigmoid_mul == eager ``a * g.sigmoid()`` for every BF16 gate value;
* the MegaMoE stager's route-error flag at a prefill-sized batch.
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


@pytest.mark.parametrize("rows", [1, 7, 300, 4350, 16384])
@pytest.mark.parametrize(
    "kind", ["rand", "ties", "near_ties", "pairs", "nonfinite", "zeros", "ep_bias"]
)
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
    elif kind == "near_ties":
        # Exact scores closer than the fast prefilter's error.
        x = (x * 4).round() / 4 + torch.randn_like(x) * 3e-6
        bias = (bias * 64).round() / 64
    elif kind == "pairs":
        # Few candidates, each top score with a sub-ulp-of-fast twin.
        x[:, 1::2] = x[:, 0::2] + torch.randn_like(x[:, 0::2]) * 1e-6
        bias[1::2] = bias[0::2]
    elif kind == "nonfinite":
        x[::7, ::13] = float("inf")
        x[::5, ::11] = float("nan")
        x[1::9, 5::17] = -float("inf")
    elif kind == "ep_bias":
        # The step bench's EP16 emulation: 56 of every 224 experts active.
        bias[torch.arange(experts, device="cuda") % 224 >= 56] = -100.0
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


@pytest.mark.parametrize("tokens,heads,head_dim,extra", [(6156, 8, 64, 512), (2048, 4, 128, 0)])
def test_head_rms_norm_matches_vllm_c(pg, tokens, heads, head_dim, extra):
    """Per-head q / k norms of a strided qkv slice (8 / 16-thread stock blocks)."""
    from vllm import _custom_ops as ops

    torch.manual_seed(tokens)
    width = heads * head_dim + extra
    base = torch.randn(tokens, width, device="cuda", dtype=torch.bfloat16) * 3
    base[0] = 0
    base[1, :8] = 3.0e38
    x = base[:, : heads * head_dim].view(tokens, heads, head_dim)
    w = (torch.rand(head_dim, device="cuda") + 0.5).to(torch.bfloat16)
    assert pg.use_head_rms_norm(x, w)
    ref = torch.empty(x.shape, dtype=x.dtype, device="cuda")
    ops.rms_norm(ref, x, w, 1e-6)
    assert torch.equal(_bits(pg.rms_norm(x, w, 1e-6)), _bits(ref))


def test_grouped_head_rms_norm_matches_vllm_c(pg):
    """The DSpark context K norm: [L, ctx, kv_heads, 64] with per-layer weights
    [L, 64] in one stock call, replayed per layer."""
    from vllm import _custom_ops as ops

    torch.manual_seed(5)
    x = torch.randn(5, 3000, 16, 64, device="cuda", dtype=torch.bfloat16) * 3
    w = (torch.rand(5, 64, device="cuda") + 0.5).to(torch.bfloat16)
    ref = torch.empty_like(x)
    ops.rms_norm(ref, x, w, 1e-6)
    out = torch.empty_like(x)
    for layer in range(5):
        assert pg.use_head_rms_norm(x[layer], w[layer])
        pg.rms_norm(x[layer], w[layer], 1e-6, out=out[layer])
    assert torch.equal(_bits(out), _bits(ref))


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


@pytest.mark.parametrize("idx_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("cols", [1, 4, 8])
@pytest.mark.parametrize("rows", [1, 2, 32])
def test_write_state_rows_strided_index(pg, idx_dtype, cols, rows):
    # The prefill KDA write-back passes block_table[:, 0]: with DSpark spec
    # decoding the mamba block table has num_spec + 1 columns, so the index is
    # a strided column view. Every row must land in ITS OWN column-0 slot.
    slots, heads, dk, dv = 600, 4, 128, 128
    perm = torch.randperm(slots, device="cuda")[: rows * cols].to(idx_dtype)
    block_table = perm.view(rows, cols)
    idx = block_table[:, 0]
    assert idx.stride(0) == cols
    dst0 = torch.randn(slots, heads, dk, dv, device="cuda")
    src = torch.randn(rows, heads, dk, dv, device="cuda")
    ref = dst0.clone()
    ref[idx] = src
    out = dst0.clone()
    pg.write_state_rows(out, idx, src)
    assert torch.equal(out, ref)


@pytest.mark.parametrize("rows", [2048, 6156])
@pytest.mark.parametrize("block_size", [16, 64])
def test_rope_store_kv_matches_rotary_and_cache(pg, rows, block_size):
    """RoPE + K/V cache insert == ops.rotary_embedding (NeoX, in place) followed
    by reshape_and_cache_flash ("auto" BF16 cache), the DSpark context step's
    layout: K / V row views of a token-major [rows, L, 2, nkv, hd] projection,
    skipped (-1) slots."""
    from vllm import _custom_ops as ops

    torch.manual_seed(rows + block_size)
    layers, nkv, hd, max_pos = 5, 16, 64, 1 << 17
    kv = torch.randn(rows, layers, 2, nkv, hd, device="cuda", dtype=torch.bfloat16)
    kv[0, 2, 0, :, :4] = 3.0e38
    k = kv[:, 2, 0].contiguous()
    v = kv[:, 2, 1]
    positions = torch.randint(0, max_pos, (rows,), device="cuda", dtype=torch.int64)
    inv = 1.0 / (50000.0 ** (torch.arange(0, hd, 2, device="cuda").float() / hd))
    freqs = torch.arange(max_pos, device="cuda").float()[:, None] * inv[None]
    cos_sin = torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(torch.bfloat16)
    blocks = rows // block_size + 8
    slots = torch.randperm(blocks * block_size, device="cuda")[:rows]
    slots[::7] = -1
    # FlashAttention's cache [blocks, nkv, block_size, 2 * hd] viewed as its
    # do_kv_cache_update does.
    ref = torch.randn(blocks, nkv, block_size, 2 * hd, device="cuda").to(torch.bfloat16)
    out = ref.clone()
    rk, rv = ref.transpose(1, 2).split(hd, dim=-1)
    ok, ov = out.transpose(1, 2).split(hd, dim=-1)
    ref_k = k.clone()
    ops.rotary_embedding(positions, ref_k.view(rows, nkv * hd), None, hd, cos_sin, True)
    scale = torch.ones((), device="cuda")
    ops.reshape_and_cache_flash(ref_k, v, rk, rv, slots, "auto", scale, scale)
    assert pg.use_rope_store_kv(k, v, positions, cos_sin, slots, ok, ov)
    pg.rope_store_kv(k, v, positions, cos_sin, slots, ok, ov)
    assert torch.equal(_bits(out), _bits(ref))


@pytest.mark.parametrize("rows", [4096, 6156])
def test_rms_norm_cta_7168_matches_vllm_c(pg, rows):
    """The DSpark hidden_norm (7168 columns: 8-warp stock blocks, 4 vectors per
    lane) on the persistent one-row-per-CTA replay, written in place."""
    from vllm import _custom_ops as ops

    torch.manual_seed(rows)
    x = torch.randn(rows, 7168, device="cuda", dtype=torch.bfloat16) * 3
    x[0] = 0
    x[1, :8] = 3.0e38
    w = (torch.rand(7168, device="cuda") + 0.5).to(torch.bfloat16)
    assert pg.use_rms_norm(x, w)
    assert not pg.use_rms_norm(x[: pg.RMS_NORM_CTA_MIN_ROWS - 1], w)
    ref = torch.empty_like(x)
    ops.rms_norm(ref, x, w, 1e-6)
    out = torch.full((rows + 3, 7168), 7.0, device="cuda", dtype=torch.bfloat16)
    pg.rms_norm(x, w, 1e-6, out=out[:rows])
    assert torch.equal(_bits(out[:rows]), _bits(ref))
    assert bool((out[rows:] == 7.0).all())


@pytest.mark.parametrize("rows,block_size", [(6156, 16), (2048, 64), (37, 16)])
def test_ctx_kv_insert_matches_per_layer(pg, rows, block_size):
    """One-launch context K/V insert of 5 draft layers == per layer: per-head K
    RMSNorm (_C.rms_norm on [rows * heads, head_dim]), NeoX ops.rotary_embedding
    and reshape_and_cache_flash; skipped (-1) slots, a layer without a cache, a
    layer with fewer slots than rows."""
    from vllm import _custom_ops as ops

    torch.manual_seed(rows + block_size)
    layers, nkv, hd, max_pos, eps = 5, 16, 64, 1 << 17, 1e-6
    kv = torch.randn(rows, layers, 2, nkv, hd, device="cuda", dtype=torch.bfloat16) * 2
    kv[0, 1, 0, :, :4] = 3.0e38
    kv[min(1, rows - 1), 3, 0, 2] = 0
    positions = torch.randint(0, max_pos, (rows,), device="cuda", dtype=torch.int64)
    inv = 1.0 / (50000.0 ** (torch.arange(0, hd, 2, device="cuda").float() / hd))
    freqs = torch.arange(max_pos, device="cuda").float()[:, None] * inv[None]
    cos_sin = torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(torch.bfloat16)
    weights = [(torch.rand(hd, device="cuda") + 0.5).to(torch.bfloat16) for _ in range(layers)]
    blocks = rows // block_size + 8
    refs, outs, slots, caches = [], [], [], []
    for i in range(layers):
        if i == 2:
            refs.append(None)
            outs.append(None)
            slots.append(None)
            caches.append(None)
            continue
        n = rows - 3 if (i == 4 and rows > 8) else rows
        s = torch.randperm(blocks * block_size, device="cuda")[:n]
        s[::7] = -1
        ref = torch.randn(blocks, nkv, block_size, 2 * hd, device="cuda").to(torch.bfloat16)
        refs.append(ref)
        outs.append(ref.clone())
        slots.append(s)
        caches.append(tuple(outs[-1].transpose(1, 2).split(hd, dim=-1)))
    scale = torch.ones((), device="cuda")
    for i in range(layers):
        if refs[i] is None:
            continue
        n = slots[i].numel()
        k = torch.empty(rows * nkv, hd, device="cuda", dtype=torch.bfloat16)
        ops.rms_norm(k, kv[:, i, 0].reshape(rows * nkv, hd), weights[i], eps)
        k = k.view(rows, nkv, hd)[:n].clone()
        ops.rotary_embedding(positions[:n], k.view(n, nkv * hd), None, hd, cos_sin, True)
        rk, rv = refs[i].transpose(1, 2).split(hd, dim=-1)
        ops.reshape_and_cache_flash(k, kv[:n, i, 1], rk, rv, slots[i], "auto", scale, scale)
    assert pg.use_ctx_kv_insert(kv, weights, eps, positions, cos_sin, slots, caches)
    pg.ctx_kv_insert(kv, weights, eps, positions, cos_sin, slots, caches)
    for i in range(layers):
        if refs[i] is not None:
            assert torch.equal(_bits(outs[i]), _bits(refs[i])), f"layer {i}"


@pytest.mark.parametrize("rows", [2048, 16384, 6157])
def test_rms_norm_pair_matches_vllm_c(pg, rows):
    """The DSA q_a (1536) and kv_a (512) norms of one fused_qkv_a output in one
    launch (KimiRowsRMSNorm.forward_pair): each bitwise the stock kernel."""
    from vllm import _custom_ops as ops

    torch.manual_seed(rows)
    qkv = torch.randn(rows, 1536 + 512 + 64, device="cuda", dtype=torch.bfloat16) * 3
    qkv[0] = 0
    qkv[1, :8] = 3.0e38
    qkv[2, 1536 : 1536 + 8] = 3.0e38
    q_c, kv_c = qkv[:, :1536], qkv[:, 1536 : 1536 + 512]
    wq = (torch.rand(1536, device="cuda") + 0.5).to(torch.bfloat16)
    wkv = (torch.rand(512, device="cuda") + 0.5).to(torch.bfloat16)
    assert pg.use_rms_norm(q_c, wq) and pg.use_rms_norm(kv_c, wkv)
    got = pg.rms_norm_pair(q_c, wq, kv_c, wkv, 1e-6)
    assert got is not None
    for out, x, w in zip(got, (q_c, kv_c), (wq, wkv)):
        ref = torch.empty(x.shape, dtype=x.dtype, device="cuda")
        ops.rms_norm(ref, x, w, 1e-6)
        assert torch.equal(_bits(out), _bits(ref))
    # Shapes without an instantiated pair launch nothing (caller runs both).
    assert pg.rms_norm_pair(kv_c, wkv, q_c, wq, 1e-6) is None


def test_rows_rms_norm_module_pair(pg):
    """KimiRowsRMSNorm.forward_pair == the two module calls, bitwise; None
    for a mismatched eps."""
    from vllm.config import VllmConfig, set_current_vllm_config

    torch.manual_seed(5)
    qkv = torch.randn(4096, 2112, device="cuda", dtype=torch.bfloat16)
    q_c, kv_c = qkv[:, :1536], qkv[:, 1536:2048]
    with set_current_vllm_config(VllmConfig()):
        qn = pg.KimiRowsRMSNorm(1536, eps=1e-6).cuda().to(torch.bfloat16)
        kvn = pg.KimiRowsRMSNorm(512, eps=1e-6).cuda().to(torch.bfloat16)
        for n in (qn, kvn):
            with torch.no_grad():
                n.weight.copy_(torch.rand_like(n.weight) + 0.5)
        got = qn.forward_pair(q_c, kvn, kv_c)
        assert got is not None
        assert torch.equal(_bits(got[0]), _bits(qn(q_c)))
        assert torch.equal(_bits(got[1]), _bits(kvn(kv_c)))
        kvn.variance_epsilon = 1e-5
        assert qn.forward_pair(q_c, kvn, kv_c) is None
