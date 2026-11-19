# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 SP decode (TP4 = DCP4): the query of every gathered head computed on one
rank (dsa.local_query.LocalQuery, 96 heads, one GPU, no collective) against the stock
decode's arithmetic: q_b (BF16 linear), the W_UK bmm, the stock decode query
quantization (flashinfer_mla_sparse.decode_query_quant_pad) and the stock KV insert
(RMSNorm + concat_and_cache_mla). The fused W_UK + pack (dsa.query_uk) sums the
128-term dot products in another order: an FP8 ulp on rare bytes (q_pe bitwise where
the q_b tactic is the stock GEMM); the KV rows bitwise, unowned slots
untouched, the padded heads zero. The 4-rank SP DSA layer:
tests/distributed/test_kimi_k3_sdec_sp_dsa.py."""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="the specialized decode kernels need SM100/SM103/SM107",
)

GATHERED, Q_HEADS, PITCH, UNIT, FP8_ROW, EPS = 96, 128, 5344, 64, 576, 1e-6
ROWS = (20, 64, 68, 128, 384, 512)


def context():
    from vllm.models.kimi_k3.nvidia.specialized_decode.shapes import Context

    sms = torch.cuda.get_device_properties(0).multi_processor_count
    return Context(
        arch="rubin", tp=4, rank=0, dcp=4, sms=sms, q=4, max_rows=512,
        max_context=131072, bf16_state=False, dp=8, ep=32, sequence_parallel=True,
        heads_per_rank=24,
    )  # fmt: skip


def stub_attention(gen):
    bf, dev = torch.bfloat16, "cuda"
    q_b = (torch.randn(GATHERED * 192, 1536, generator=gen, device=dev) / 40).to(bf)
    w_uk = (torch.randn(GATHERED, 128, 512, generator=gen, device=dev) / 12).to(bf)
    kv_norm = (torch.rand(512, generator=gen, device=dev) + 0.5).to(bf)
    layer = SimpleNamespace(
        W_UK_T=w_uk,
        _q_scale=torch.tensor(0.5, device=dev),
        _k_scale=torch.tensor(0.875, device=dev),
    )
    wrapper = SimpleNamespace(
        mla_attn=layer,
        q_b_proj=SimpleNamespace(weight=q_b),
        kv_a_layernorm=SimpleNamespace(weight=kv_norm, variance_epsilon=EPS),
    )
    return SimpleNamespace(mla_attn=wrapper)


def test_local_query_matches_stock():
    from vllm import _custom_ops as ops
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.local_query import (  # noqa: E501
        LocalQuery,
    )
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
        decode_query_quant_pad,
    )

    dev, bf = "cuda", torch.bfloat16
    gen = torch.Generator(device=dev).manual_seed(0)
    attention = stub_attention(gen)
    local = LocalQuery(attention, None, 1, ROWS, context(), Q_HEADS)
    layer, wrapper = attention.mla_attn.mla_attn, attention.mla_attn
    for rows in ROWS:
        projected = (torch.randn(rows, PITCH, generator=gen, device=dev) * 3).to(bf)
        q_c = (torch.randn(rows, 1536, generator=gen, device=dev)).to(bf)
        blocks = -(-rows // UNIT) + 2
        slots = torch.randperm(blocks * UNIT, generator=gen, device=dev)[:rows]
        slots[torch.arange(rows, device=dev) % 4 != 1] = -1  # other ranks' tokens
        cache = torch.zeros(blocks, UNIT, FP8_ROW, dtype=torch.uint8, device=dev)
        local.buffers.query.view(torch.uint8).fill_(0x7F)  # stale bytes
        local.buffers.query[:, GATHERED:].zero_()
        query = local.run(q_c, projected, cache, slots)
        torch.cuda.synchronize()

        q = q_c @ wrapper.q_b_proj.weight.T
        q = q.view(rows, GATHERED, 192)
        latent = torch.bmm(q[..., :128].transpose(0, 1), layer.W_UK_T).transpose(0, 1)
        ref = decode_query_quant_pad(latent, q[..., 128:], layer._q_scale, Q_HEADS)
        assert ref is not None
        ours = query.view(torch.uint8)
        assert (ours[:, GATHERED:] == 0).all()  # padded heads
        # The W_UK K order (and split-K q_b's) differs: an FP8 ulp at most, rarely
        exact = torch.cat([latent, q[..., 128:]], -1).float() / layer._q_scale
        torch.testing.assert_close(
            query[:, :GATHERED].float(), exact, atol=3e-2, rtol=7e-2
        )
        differ = (ours != ref.view(torch.uint8)).float().mean()
        print(f"rows {rows} ({local.projection.kinds[rows]}): {float(differ):.2e} differ")
        assert differ < 1e-3, rows
        if local.projection.kinds[rows] in ("cublas", "lt"):  # q_pe: bitwise
            assert torch.equal(ours[:, :GATHERED, 512:], ref.view(torch.uint8)[:, :GATHERED, 512:])

        stock = torch.zeros_like(cache)
        kv = torch.empty(rows, 512, dtype=bf, device=dev)
        ops.rms_norm(kv, projected[:, 1536:2048].contiguous(), local.kv_norm, EPS)
        k_pe = projected[:, 2048:2112]
        ops.concat_and_cache_mla(kv, k_pe, stock, slots, "fp8", local.k_scale)
        # Owned rows as stock (the fused insert sums the squares in another order:
        # a rare BF16 / E4M3 rounding flip), others untouched.
        owned = slots >= 0
        flat, ref_flat = cache.view(-1, FP8_ROW), stock.view(-1, FP8_ROW)
        mism = (flat != ref_flat).any(-1)
        assert not mism[~torch.isin(torch.arange(flat.shape[0], device=dev),
                                    slots[owned])].any(), rows
        got = flat[slots[owned]].view(torch.float8_e4m3fn).float()
        exp = ref_flat[slots[owned]].view(torch.float8_e4m3fn).float()
        bad = (got != exp).float().mean()
        print(f"rows {rows}: KV insert {float(bad):.2e} of bytes differ")
        assert bad < 1e-4 and torch.allclose(got, exp, rtol=0.125, atol=0), rows
