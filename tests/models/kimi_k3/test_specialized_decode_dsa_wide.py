# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 specialized decode, wide verify rows (> 64 token rows, Rubin):
PreparedDsa at the rows ``dsa.prepared.DSA_WIDE`` admits (65-96), and the fused
indexer key insert from the packed in-projection rows that seam S2-DSA feeds."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="the specialized decode kernels need SM100/SM103/SM107",
)

Q_LORA, KV_LORA, V = 1536, 512, 128
INDEX_DIM, INDEX_HEADS = 128, 32


@pytest.mark.parametrize("rows", [72, 96, 256, 768])
def test_fused_k_insert_from_packed_rows(rows):
    """Production K3.1 (LayerNorm key norm, no index rope, no PCP) inserts the
    indexer key with the fused ``indexer_k_norm_quant_insert``. Under seam S2-DSA it
    reads ``index_k`` as a column view of the packed projection rows (row stride
    3040 at TP16 instead of the stock ``wkv_index`` output's 160): the cache bytes
    and the top-k rows' -1 fill equal those from the contiguous stock rows."""
    from vllm.models.kimi_k3.nvidia.ops.dsa_indexer import (
        indexer_k_norm_quant_insert,
    )

    g = torch.Generator(device="cuda").manual_seed(rows)
    pitch = 2112 + 6 * V + INDEX_DIM + INDEX_HEADS
    projected = torch.randn(rows, pitch, device="cuda", generator=g).to(torch.bfloat16)
    index_k = projected[:, -160:-INDEX_HEADS]
    assert index_k.stride(0) == pitch == 3040
    stock_k = projected[:, -160:].contiguous()[:, :INDEX_DIM]  # wkv_index output
    weight = 1 + 0.1 * torch.randn(INDEX_DIM, device="cuda", generator=g)
    bias = 0.1 * torch.randn(INDEX_DIM, device="cuda", generator=g)
    blocks, block = -(-rows // 64) + 3, 64
    slots = torch.randperm(blocks * block, device="cuda", generator=g)[:rows]
    slots[1] = -1  # a token another DCP rank owns
    caches, fills = [], []
    for k in (index_k, stock_k):
        cache = torch.zeros(
            blocks, block, INDEX_DIM + 4, dtype=torch.uint8, device="cuda"
        )
        topk = torch.full((rows + 5, 2048), 3, dtype=torch.int32, device="cuda")
        indexer_k_norm_quant_insert(k, weight, bias, 1e-6, slots, cache, topk, rows)
        caches.append(cache)
        fills.append(topk)
    assert torch.equal(*caches)
    assert torch.equal(*fills)
    assert (fills[0][:rows] == -1).all() and (fills[0][rows:] == 3).all()
    assert caches[0].any()


def test_prepared_dsa_wide_admission(monkeypatch):
    """``dsa.prepared.choose``: GENERIC at the tuned rows; above them only Rubin DCP
    verify shapes that ``DSA_WIDE`` lists, at most ``WIDE_MAX_ROWS`` (MlaQuery's
    gathered-q poll needs a co-resident grid); else None (S2-DSA)."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa import prepared
    from vllm.models.kimi_k3.nvidia.specialized_decode.shapes import (
        Context,
        Path,
        Shape,
    )

    ctx = Context(
        arch="rubin", tp=16, rank=0, dcp=16, sms=212, q=4, max_rows=768,
        max_context=1 << 20, bf16_state=True,
    )  # fmt: skip
    assert prepared.choose(Shape(64, 4), ctx).path is Path.GENERIC
    monkeypatch.setattr(prepared, "DSA_WIDE", {"*": ""})
    assert all(prepared.choose(Shape(m, 4), ctx) is None for m in (72, 96, 512))
    monkeypatch.setattr(prepared, "DSA_WIDE", {"*": "65-768:1"})
    for m in (72, 96):
        assert prepared.choose(Shape(m, 4), ctx).tactic == "wide"
    assert prepared.choose(Shape(128, 4), ctx) is None  # above WIDE_MAX_ROWS
    gb300 = Context(**{**ctx.__dict__, "arch": "gb300"})
    assert prepared.choose(Shape(72, 4), gb300) is None
    tp_only = Context(**{**ctx.__dict__, "dcp": 1})
    assert prepared.choose(Shape(72, 4), tp_only) is None


def test_index_policies_wide_rows():
    """wq_b split-K tactics: the 64-row bins up to 64, the generic split-K up to 128,
    none above (cuBLAS)."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.prepared import (
        index_policies,
    )

    policies = index_policies([8, 64, 72, 96, 128, 256, 768])
    assert policies[64] == (64, 64, 1, 4)
    assert policies[72] == policies[96] == policies[128] == (64, 64, 2, 6)
    assert 256 not in policies and 768 not in policies


def test_dsa_wide_query_grid_resident():
    """Hazard H1: at wide rows every MlaQuery CTA polls the gathered q mailbox that
    other ranks' CTAs fill, so the whole grid must be resident at once. The compiled
    DSA query (DCP16, 6 heads, FP8, 96-row box) fits at 72 / 96 rows
    (cuOccupancyMaxActiveClusters, checked by compile_core), and the check refuses
    the 128-row grid that WIDE_MAX_ROWS keeps on S2-DSA."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        layout,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa import prepared
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.mla.query import (
        MlaQuery,
    )

    sms = torch.cuda.get_device_properties(0).multi_processor_count
    if sms <= 152:
        pytest.skip("Rubin-class wide rows")
    bf, dev, heads = torch.bfloat16, "cuda", 6
    weights = (
        torch.ones(Q_LORA, device=dev, dtype=bf),
        torch.ones(KV_LORA, device=dev, dtype=bf),
        torch.zeros(heads * 192, Q_LORA, device=dev, dtype=bf),
        torch.zeros(heads, 128, KV_LORA, device=dev, dtype=bf),
        torch.zeros(heads, 128, KV_LORA, device=dev, dtype=bf).transpose(1, 2),
    )
    caches = (
        torch.zeros(8, 64, 576, dtype=torch.uint8, device=dev),
        torch.zeros(8, 64, 132, dtype=torch.uint8, device=dev),
    )
    specs = tuple(layout(t) for t in (*weights, *caches))
    pitch = 2112 + heads * V + 160
    device = torch.accelerator.current_device_index()
    for rows in (72, 96):
        assert rows <= prepared.WIDE_MAX_ROWS
        query, _, _ = prepared.compile_core(
            rows, heads, pitch, specs, device, 4, 16, 128, sms, fp8=True, cap=96
        )
        mla_query = MlaQuery(heads, rows, dsa=True, fp8=True, dcp=16, q_heads=128)
        resident = prepared.check_query_residency(query.compiled, mla_query, device)
        assert resident >= mla_query.grid[1] * mla_query.grid[2] == 18
    wider = MlaQuery(heads, 128, dsa=True, fp8=True, dcp=16, q_heads=128)
    assert wider.grid[1] * wider.grid[2] > resident
    with pytest.raises(RuntimeError, match="cannot all be resident"):
        prepared.check_query_residency(query.compiled, wider, device)
