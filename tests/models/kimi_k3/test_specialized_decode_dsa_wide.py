# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 specialized decode, wide verify rows (> 64 token rows, Rubin):
WideDsa at the rows ``dsa.wide.DSA_WIDE`` admits (65 up to
VLLM_KIMI_K3_DSA_WIDE_MAX_ROWS: 96 by default, at most 512): its admission, index
policies, MlaQuery / DcpValue grid residency and kernels at those rows, its DCP core
and DCP4 mailboxes; and the fused indexer key insert from the packed in-projection
rows that seam S2-DSA feeds."""

import time

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="the specialized decode kernels need SM100/SM103/SM107",
)

Q_LORA, KV_LORA, V = 1536, 512, 128
INDEX_DIM, INDEX_HEADS = 128, 32
# WideDsa's kernels at its rows: test_specialized_decode_dsa.py's TP8 layout.
UNIT, MLA_ROW, INDEX_ROW = 64, 656, 132  # indexer page tokens; cache row bytes
FP8_ROW = 576  # per-tensor FP8 MLA row (FlashInfer sparse)
GAP = 36  # rows of the other layers' pages between two blocks of a packed cache
FAR = 5 << 30  # bytes a far packed cache spans: its later blocks lie past 2 GiB
HEADS = 12  # TP8
GATE = HEADS * 128
PITCH = 2112 + GATE + 128 + 32  # packed [q_a | kv_a | gate | wk | wv] row
EPS = 1e-6


def packed_cache(blocks, block, row, gap=GAP):
    """One layer's view of a packed (BLHNC) cache: ``blocks`` blocks of ``block``
    rows, ``gap`` rows of other layers' pages between two consecutive blocks
    (``FAR``: as many as spread the blocks over ``FAR`` bytes)."""
    stride = FAR // blocks // 256 * 256 if gap == FAR else (block + gap) * row
    raw = torch.zeros(blocks * stride, dtype=torch.uint8, device="cuda")
    return torch.as_strided(raw, (blocks, block, row), (stride, row, 1))


def index_blocks(index):
    """Each indexer page's bytes: its FP8 keys, then their FP32 scales."""
    return torch.as_strided(
        index, (index.shape[0], index.shape[1] * INDEX_ROW), (index.stride(0), 1)
    )


def rms_norm(x, weight):
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)
    return (x * weight.float()).to(torch.bfloat16)


def kernels(batch, mla, index, *weights, fp8=False):
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        layout,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.wide import (
        compile_core,
    )

    specs = tuple(layout(w) for w in (*weights, mla, index))
    return compile_core(
        batch, HEADS, PITCH, specs, torch.accelerator.current_device_index(), 1, fp8=fp8
    )


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
    """``dsa.wide.choose``: GENERIC at the tuned rows; above them only Rubin DCP
    verify shapes that ``DSA_WIDE`` lists, at most VLLM_KIMI_K3_DSA_WIDE_MAX_ROWS
    (``wide_max_rows``: WIDE_MAX_ROWS by default, 64 none, at most WIDE_LIMIT); else
    None (S2-DSA)."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa import wide
    from vllm.models.kimi_k3.nvidia.specialized_decode.shapes import (
        Context,
        Path,
        Shape,
    )

    ctx = Context(
        arch="rubin", tp=16, rank=0, dcp=16, sms=212, q=4, max_rows=768,
        max_context=1 << 20, bf16_state=True,
    )  # fmt: skip
    assert wide.choose(Shape(64, 4), ctx).path is Path.GENERIC
    monkeypatch.setattr(wide, "DSA_WIDE", {"*": ""})
    assert all(wide.choose(Shape(m, 4), ctx) is None for m in (72, 96, 512))
    monkeypatch.setattr(wide, "DSA_WIDE", {"*": "65-768:1"})
    for m in (72, 96):
        assert wide.choose(Shape(m, 4), ctx).tactic == "wide"
    assert wide.choose(Shape(128, 4), ctx) is None  # above WIDE_MAX_ROWS
    monkeypatch.setenv("VLLM_KIMI_K3_DSA_WIDE_MAX_ROWS", "256")
    for m in (72, 128, 256):
        assert wide.choose(Shape(m, 4), ctx).tactic == "wide"
    assert wide.choose(Shape(260, 4), ctx) is None
    monkeypatch.setenv("VLLM_KIMI_K3_DSA_WIDE_MAX_ROWS", "64")
    assert wide.choose(Shape(72, 4), ctx) is None
    for rows in ("63", str(wide.WIDE_LIMIT + 1)):
        monkeypatch.setenv("VLLM_KIMI_K3_DSA_WIDE_MAX_ROWS", rows)
        with pytest.raises(ValueError, match="VLLM_KIMI_K3_DSA_WIDE_MAX_ROWS"):
            wide.choose(Shape(72, 4), ctx)
    monkeypatch.delenv("VLLM_KIMI_K3_DSA_WIDE_MAX_ROWS")
    gb300 = Context(**{**ctx.__dict__, "arch": "gb300"})
    assert wide.choose(Shape(72, 4), gb300) is None
    tp_only = Context(**{**ctx.__dict__, "dcp": 1})
    assert wide.choose(Shape(72, 4), tp_only) is None


def test_index_policies_wide_rows():
    """wq_b split-K tactics: the 64-row bins up to 64, the generic split-K up to 128,
    none above (cuBLAS)."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.wide import (
        index_policies,
    )

    policies = index_policies([8, 64, 72, 96, 128, 256, 768])
    assert policies[64] == (64, 64, 1, 4)
    assert policies[72] == policies[96] == policies[128] == (64, 64, 2, 6)
    assert 256 not in policies and 768 not in policies


def test_dsa_wide_query_grid_resident():
    """Hazard H1: at wide rows the MlaQuery CTAs poll the gathered q mailbox that
    other ranks' CTAs fill, so every CTA that polls with them must be resident. The
    compiled DSA query (DCP16, 6 heads, FP8, 512-row box) fits whole at 72 / 96 rows
    (cuOccupancyMaxActiveClusters, checked by compile_core); a 128-row grid would not
    (the check refuses it without the tile poll), so above WIDE_MAX_ROWS each row
    tile polls its own tokens and one tile's 6 clusters must fit (104-512 rows);
    DcpValue's whole grid fits at every one of these rows (check_value_residency)."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        layout,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa import wide
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.mla.dcp import (
        compile_value,
    )
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
        assert rows <= wide.WIDE_MAX_ROWS
        query, _, _ = wide.compile_core(
            rows, heads, pitch, specs, device, 4, 16, 128, sms, fp8=True, cap=96
        )
        mla_query = MlaQuery(heads, rows, dsa=True, fp8=True, dcp=16, q_heads=128)
        resident = wide.check_query_residency(query.compiled, mla_query, device)
        assert resident >= mla_query.grid[1] * mla_query.grid[2] == 18
    wider = MlaQuery(heads, 128, dsa=True, fp8=True, dcp=16, q_heads=128)
    assert wider.grid[1] * wider.grid[2] > resident
    with pytest.raises(RuntimeError, match="cannot all be resident"):
        wide.check_query_residency(query.compiled, wider, device)
    for rows in (104, 128, 256, 512):
        query, _, value = wide.compile_core(
            rows, heads, pitch, specs, device, 4, 16, 128, sms, fp8=True, cap=512
        )
        tiled = MlaQuery(
            heads, rows, dsa=True, fp8=True, dcp=16, q_heads=128, tile_poll=True
        )
        assert tiled.tile_poll and tiled.grid[1] * tiled.grid[2] > resident
        assert wide.check_query_residency(query.compiled, tiled, device) >= heads
        grid = wide.check_value_residency(value, heads, 16, rows, 4, sms, device)
        assert grid[0] * grid[1] * grid[2] <= sms
    # 576 rows (past WIDE_LIMIT): 6 x 36 merge CTAs, more than one per SM fits.
    with pytest.raises(RuntimeError, match="cannot be co-resident"):
        value = compile_value(
            heads, 16, 576, 576, specs[4], pitch, device, sms, 4, dsa_heads=128,
            dsa_bf16=False,
        )  # fmt: skip
        wide.check_value_residency(value, heads, 16, 576, 4, sms, device)


@pytest.mark.parametrize(
    "block,gap", [(64, GAP), (1024, GAP), (64, FAR)], ids=["64", "1024", "far"]
)
@pytest.mark.parametrize("batch", [72, 96])
def test_query_and_cache_insert(batch, block, gap):
    """BF16 query rows (W_UK absorbed), normalized q_c and the fp8_ds_mla KV row
    as the stock path computes and writes them, into the packed MLA blocks."""
    import cutlass

    from vllm import _custom_ops as ops
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        stream,
        to_cute,
    )

    torch.manual_seed(batch)
    bf, dev = torch.bfloat16, "cuda"
    projected = torch.randn(batch, PITCH, device=dev, dtype=bf)
    q_norm = torch.rand(1536, device=dev, dtype=bf) + 0.5
    kv_norm = torch.rand(512, device=dev, dtype=bf) + 0.5
    q_b = torch.randn(HEADS * 192, 1536, device=dev, dtype=bf) / 40
    w_uk = torch.randn(HEADS, 128, 512, device=dev, dtype=bf) / 12
    w_uv = torch.randn(HEADS, 128, 512, device=dev, dtype=bf).transpose(1, 2)
    blocks = 4 if gap == FAR else 2
    mla = packed_cache(blocks, block, MLA_ROW, gap)
    index = packed_cache(8, UNIT, INDEX_ROW)
    slots = torch.randperm(blocks * block - 1, device=dev)[:batch]
    slots[0] = blocks * block - 1  # the last row
    if batch > 2:
        slots[1] = -1  # a padded row writes nothing
    weights = q_norm, kv_norm, q_b, w_uk, w_uv
    query_kernel, _, _ = kernels(batch, mla, index, *weights)
    query = torch.zeros(batch, 64, 576, device=dev, dtype=bf)
    q_c = torch.zeros(batch, 1536, device=dev, dtype=bf)
    unit = torch.ones(1, device=dev, dtype=torch.float32)
    args = (to_cute(projected), *(to_cute(w) for w in weights[:4]), to_cute(mla))
    args += (to_cute(slots, 8), to_cute(unit, 4), to_cute(unit, 4))
    query_kernel(*args, cutlass.Float32(EPS), to_cute(query), to_cute(q_c), stream(dev))

    ref_q_c = rms_norm(projected[:, :1536], q_norm)
    torch.testing.assert_close(q_c, ref_q_c, atol=1e-2, rtol=1e-2)
    q = (ref_q_c @ q_b.T).view(batch, HEADS, 192)
    latent = torch.bmm(q[..., :128].transpose(0, 1), w_uk).transpose(0, 1)
    torch.testing.assert_close(
        query[:, :HEADS], torch.cat([latent, q[..., 128:]], -1), atol=2e-2, rtol=2e-2
    )
    assert (query[:, HEADS:] == 0).all()

    stock = packed_cache(blocks, block, MLA_ROW, gap)
    kv = rms_norm(projected[:, 1536:2048], kv_norm)
    ops.concat_and_cache_mla(
        kv, projected[:, 2048:2112], stock, slots, "fp8_ds_mla", unit
    )
    live = slots[slots >= 0]
    ours, ref = (c.reshape(-1, MLA_ROW)[live] for c in (mla, stock))
    assert torch.equal(ours[:, 512:], ref[:, 512:])  # tile scales and k_pe
    # Latent bytes differ only where the RMSNorm rounds differently.
    assert (ours[:, :512] != ref[:, :512]).float().mean() < 0.01


@pytest.mark.parametrize("k_scale", [2.0, 0.875])
@pytest.mark.parametrize("gap", [GAP, FAR])
@pytest.mark.parametrize("batch", [72, 96])
def test_fp8_query_and_cache_insert(batch, gap, k_scale):
    """FlashInfer core: the per-tensor FP8 query rows [B, H, 576] and the FP8 KV row
    in the packed MLA blocks, as the stock FlashInfer path quantizes and writes them
    (concat_and_cache_mla "fp8"), with non-unit scales (0.875: dividing by k_scale
    and multiplying by its FP32 inverse round some k_pe values apart)."""
    import cutlass

    from vllm import _custom_ops as ops
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        stream,
        to_cute,
    )

    torch.manual_seed(batch)
    bf, dev, fp8 = torch.bfloat16, "cuda", torch.float8_e4m3fn
    projected = torch.randn(batch, PITCH, device=dev, dtype=bf)
    q_norm = torch.rand(1536, device=dev, dtype=bf) + 0.5
    kv_norm = torch.rand(512, device=dev, dtype=bf) + 0.5
    q_b = torch.randn(HEADS * 192, 1536, device=dev, dtype=bf) / 40
    w_uk = torch.randn(HEADS, 128, 512, device=dev, dtype=bf) / 12
    w_uv = torch.randn(HEADS, 128, 512, device=dev, dtype=bf).transpose(1, 2)
    blocks, block = 4, 64
    mla = packed_cache(blocks, block, FP8_ROW, gap)
    index = packed_cache(8, UNIT, INDEX_ROW)
    slots = torch.randperm(blocks * block - 1, device=dev)[:batch]
    slots[0] = blocks * block - 1  # the last row
    if batch > 2:
        slots[1] = -1  # a padded row writes nothing
    weights = q_norm, kv_norm, q_b, w_uk, w_uv
    query_kernel, _, _ = kernels(batch, mla, index, *weights, fp8=True)
    query = torch.zeros(batch, HEADS, 576, device=dev, dtype=torch.uint8)
    q_c = torch.zeros(batch, 1536, device=dev, dtype=bf)
    q_scale = torch.full((1,), 0.5, device=dev)
    k_scale = torch.full((1,), k_scale, device=dev)
    args = (to_cute(projected), *(to_cute(w) for w in weights[:4]), to_cute(mla))
    # The query multiplies by 1 / q_scale; the KV row is divided by k_scale (as
    # WideDsa passes them).
    args += (to_cute(slots, 8), to_cute(1 / q_scale, 4), to_cute(k_scale, 4))
    query_kernel(*args, cutlass.Float32(EPS), to_cute(query), to_cute(q_c), stream(dev))

    ref_q_c = rms_norm(projected[:, :1536], q_norm)
    torch.testing.assert_close(q_c, ref_q_c, atol=1e-2, rtol=1e-2)
    q = (ref_q_c @ q_b.T).view(batch, HEADS, 192)
    latent = torch.bmm(q[..., :128].transpose(0, 1), w_uk).transpose(0, 1)
    ref_query = torch.cat([latent, q[..., 128:]], -1).float() / q_scale
    torch.testing.assert_close(query.view(fp8).float(), ref_query, atol=3e-2, rtol=7e-2)

    stock = packed_cache(blocks, block, FP8_ROW, gap)
    kv = rms_norm(projected[:, 1536:2048], kv_norm)
    ops.concat_and_cache_mla(kv, projected[:, 2048:2112], stock, slots, "fp8", k_scale)
    live = slots[slots >= 0]
    ours, ref = (c[live // block, live % block] for c in (mla, stock))
    assert torch.equal(ours[:, 512:], ref[:, 512:])  # k_pe
    # Latent bytes differ only where the RMSNorm rounds differently.
    assert (ours[:, :512] != ref[:, :512]).float().mean() < 0.01


@pytest.mark.parametrize("gap", [0, GAP, FAR])
@pytest.mark.parametrize("batch", [72, 96])
def test_indexer_inputs(batch, gap):
    """The FP8 indexer query, its folded head weights and the indexer key insert
    match per_token_group_quant_fp8 (ue8m0) and the stock key writers, into 64-entry
    indexer pages, contiguous or packed between other layers' pages. The key is
    checked per written entry against both stock writers on SM100-family devices:
    indexer_k_quant_and_cache (in the MQA logits' scale format: "mxfp8", four packed
    UE8M0 bytes, where mqa_logits_use_mx_scales; an fp32 "ue8m0" scale elsewhere)
    and the fused indexer_k_norm_quant_insert, which reads the key as the strided
    column view of the packed projection rows (the production K3.1 path)."""
    import cutlass

    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8,
    )
    from vllm.models.kimi_k3.nvidia.ops.dsa_indexer import (
        indexer_k_norm_quant_insert,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        stream,
        to_cute,
    )
    from vllm.utils.deep_gemm import mqa_logits_use_mx_scales

    torch.manual_seed(batch)
    bf, dev = torch.bfloat16, "cuda"
    projected = torch.randn(batch, PITCH, device=dev, dtype=bf)
    iq = torch.randn(batch, 32 * 128, device=dev, dtype=bf)
    norm_w = torch.rand(128, device=dev) + 0.5
    norm_b = torch.randn(128, device=dev) / 10
    blocks = 8 if batch <= 64 else 16
    mla = packed_cache(8, UNIT, MLA_ROW)
    index = packed_cache(blocks, UNIT, INDEX_ROW, gap)
    slots = torch.randperm(blocks * UNIT - 1, device=dev)[:batch]
    slots[0] = blocks * UNIT - 1  # the last entry
    if batch > 2:
        slots[1] = -1  # a padded row or, under DCP, a token another rank owns
    weights = (
        torch.ones(1536, device=dev, dtype=bf),
        torch.ones(512, device=dev, dtype=bf),
        torch.zeros(HEADS * 192, 1536, device=dev, dtype=bf),
        torch.zeros(HEADS, 128, 512, device=dev, dtype=bf),
        torch.zeros(HEADS, 128, 512, device=dev, dtype=bf).transpose(1, 2),
    )
    _, index_kernel, _ = kernels(batch, mla, index, *weights)
    q_fp8 = torch.zeros(batch, 32, 128, device=dev, dtype=torch.uint8)
    head_w = torch.zeros(batch, 32, device=dev, dtype=bf)
    args = (to_cute(projected), to_cute(iq), to_cute(norm_w, 4), to_cute(norm_b, 4))
    args += (to_cute(index), to_cute(slots, 8), to_cute(q_fp8), to_cute(head_w, 4))
    index_kernel(*args, cutlass.Float32(EPS), stream(dev))

    ref_q, ref_scale = per_token_group_quant_fp8(
        iq.view(-1, 128), 128, column_major_scales=False, use_ue8m0=True
    )
    assert torch.equal(q_fp8.view(-1, 128), ref_q.view(torch.uint8))
    key = 2112 + GATE
    # Same FP32 fold as indexer_q_quant_fold_weights: wv * scale / sqrt(32 * 128),
    # rounded to BF16 once.
    wv = projected[:, key + 128 : key + 160].float()
    ref_w = (wv * ref_scale.view(batch, 32) * (32 * 128) ** -0.5).to(bf)
    torch.testing.assert_close(head_w, ref_w, atol=0, rtol=0)

    def entries(cache):
        """Each live slot's FP8 key bytes and its 4 scale bytes."""
        page, offset = slots[slots >= 0] // UNIT, slots[slots >= 0] % UNIT
        flat = index_blocks(cache)
        cols = offset.view(-1, 1) * 128 + torch.arange(128, device=dev)
        keys = flat[page.view(-1, 1), cols]
        at = UNIT * 128 + offset.view(-1, 1) * 4 + torch.arange(4, device=dev)
        return keys, flat[page.view(-1, 1), at]

    mx = mqa_logits_use_mx_scales()
    k_rows = projected[:, key : key + 128]  # strided: the packed projection's columns
    k = torch.nn.functional.layer_norm(k_rows.float(), (128,), norm_w, norm_b, EPS).to(
        bf
    )
    unfused = packed_cache(blocks, UNIT, INDEX_ROW, gap)
    ops.indexer_k_quant_and_cache(k, unfused, slots, 128, "mxfp8" if mx else "ue8m0")
    fused = packed_cache(blocks, UNIT, INDEX_ROW, gap)
    topk = torch.full((batch + 3, 2048), 7, device=dev, dtype=torch.int32)
    assert k_rows.stride(0) == PITCH
    indexer_k_norm_quant_insert(k_rows, norm_w, norm_b, EPS, slots, fused, topk, batch)
    assert (topk[:batch] == -1).all() and (topk[batch:] == 7).all()
    ours_k, ours_s = entries(index)
    for ref in (unfused, fused):
        ref_k, ref_s = entries(ref)
        assert torch.equal(ours_s, ref_s), "per-entry key scales"
        # Key codes: only LayerNorm reduction-order flips, one E4M3 code apart.
        diff = ours_k != ref_k
        assert diff.float().mean() < 1e-2, diff.float().mean()
        mag = (ours_k & 0x7F).int() - (ref_k & 0x7F).int()
        same_sign = ((ours_k ^ ref_k) & 0x80) == 0
        zero = ((ours_k & 0x7F) == 0) | ((ref_k & 0x7F) == 0)
        assert (mag.abs() <= 1).all() and (same_sign | zero).all()
    # Untouched bytes stay zero (no write outside the live entries).
    written = torch.zeros_like(index_blocks(index), dtype=torch.bool)
    page, offset = slots[slots >= 0] // UNIT, slots[slots >= 0] % UNIT
    cols = offset.view(-1, 1) * 128 + torch.arange(128, device=dev)
    written[page.view(-1, 1), cols] = True
    scales = UNIT * 128 + offset.view(-1, 1) * 4 + torch.arange(4, device=dev)
    written[page.view(-1, 1), scales] = True
    assert (index_blocks(index)[~written] == 0).all()


@pytest.mark.parametrize("batch", [8, 64, 72, 96, 128])
def test_index_projection_rows(batch):
    """WideDsa's replicated wq_b projection (N 4096, K 1536) on its split-K
    tactic at every prepared row count up to 128 (``index_policies``; wide rows 65-128
    the generic (64, 64, 2, 6)) against an FP32 GEMM of the same BF16 operands."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.wide import (
        index_policies,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.projection import (
        PreparedProjection,
    )

    torch.manual_seed(batch)
    bf, dev = torch.bfloat16, "cuda"
    wq = (torch.randn(32 * 128, 1536, device=dev) / 1536**0.5).to(bf)
    q_c = torch.randn(batch, 1536, device=dev).to(bf)
    projection = PreparedProjection(wq, index_policies([batch])).prepare()
    out = torch.full((batch, 32 * 128), float("nan"), device=dev, dtype=bf)
    projection.run(q_c, out)
    ref = q_c.float() @ wq.float().t()
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)
    assert ((out.float() - ref).norm() / ref.norm()) < 4e-3


DCP_CORE_CASES = {
    # dcp, rows, q, interleave, context, box rows
    "dcp8": (8, 4, 1, 1, 5000, 64),
    # Rubin wide verify rows (WideDsa at 65-96 rows, DSA_WIDE): TP16 / DCP16,
    # interleave 896 (the K3 recipe; kernel blocks of 64 under it), widened box.
    "dcp16_M72_q4": (16, 72, 4, 896, 2 * 16 * 896 + 517, 96),
    "dcp16_M96_q4": (16, 96, 4, 896, 2 * 16 * 896 + 517, 96),
    "dcp16_M72_q8": (16, 72, 8, 896, 2 * 16 * 896 + 517, 96),
    "dcp16_M96_q8": (16, 96, 8, 896, 2 * 16 * 896 + 517, 96),
}


@pytest.mark.parametrize("case", list(DCP_CORE_CASES))
@pytest.mark.parametrize("kernel", ["trtllm", "prims_ts"])
def test_flashinfer_dcp_core_merges_to_full_attention(
    default_vllm_config, dist_init, kernel, case
):
    """DCP on FlashInfer sparse (per-tensor FP8): each rank's dcp_core over its
    shard of a packed cache, its gathered 96-head query padded to 128 and the
    global top-k, merged by LSE in the kernel's base, is the sparse attention
    over the whole context. Request 0 lies on rank 0 only (empty shards: rows
    where some ranks own none of the top-k). The wide cases read the query from a
    widened box passed as ``box`` (rows above the 64-row box), q rows per request,
    and de-interleave the global ids at interleave 896 over 16 ranks."""
    from types import SimpleNamespace

    from vllm.model_executor.layers.linear import ColumnParallelLinear
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.wide import (
        WideDsa,
    )
    from vllm.utils.flashinfer import has_flashinfer_prims_ts_sparse_mla
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
        FlashInferMLASparseImpl,
    )

    if kernel == "prims_ts" and not has_flashinfer_prims_ts_sparse_mla():
        pytest.skip("FlashInfer has no prims_ts sparse MLA")
    if kernel == "prims_ts" and not (
        current_platform.is_device_capability(100)
        or current_platform.is_device_capability(103)
    ):
        pytest.skip("FlashInfer's prims_ts sparse MLA runs on SM100 / SM103 only")
    dcp, rows, q, interleave, context, cap = DCP_CORE_CASES[case]
    default_vllm_config.kernel_config.flashinfer_sparse_mla_kernel = kernel
    text = SimpleNamespace(kv_lora_rank=512, qk_rope_head_dim=64)
    object.__setattr__(
        default_vllm_config, "model_config", SimpleNamespace(hf_text_config=text)
    )
    heads, topk, block, reqs = 96, 2048, UNIT, rows // q
    impl = FlashInferMLASparseImpl(
        num_heads=heads,
        head_size=FP8_ROW,
        scale=192**-0.5,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="fp8_e4m3",
        logits_soft_cap=None,
        attn_type="decoder",
        kv_sharing_target_layer_name=None,
        q_lora_rank=None,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        qk_head_dim=192,
        v_head_dim=128,
        kv_b_proj=ColumnParallelLinear(512, heads * 256, bias=False),
    )
    impl.need_to_return_lse_for_decode = True
    impl._workspace_buffer = torch.zeros(1 << 30, dtype=torch.int8, device="cuda")
    layer = SimpleNamespace(_q_scale_float=1.0, _k_scale_float=1.0)
    e4m3 = torch.float8_e4m3fn

    gen = torch.Generator(device="cuda").manual_seed(0)
    # Each rank's latents are offset differently and the scores are peaked, so the
    # shards' partial outputs and LSEs differ: a wrong LSE base misweights them.
    kv = torch.rand(reqs, context, FP8_ROW, generator=gen, device="cuda") - 0.5
    position = torch.arange(context, device="cuda")
    owner = (position // interleave) % dcp
    local_slot = (position // (dcp * interleave)) * interleave + position % interleave
    kv[..., :512] += (1.6 / dcp) * owner.view(1, -1, 1)
    kv = kv.to(e4m3)
    query = torch.zeros(rows, 128, FP8_ROW, dtype=torch.uint8, device="cuda")
    query[:, :heads] = (
        (8 * (torch.rand(rows, heads, FP8_ROW, generator=gen, device="cuda") - 0.5))
        .to(e4m3)
        .view(torch.uint8)
    )
    tokens = torch.full((rows, topk), -1, dtype=torch.int32, device="cuda")
    for t in range(rows):
        r = t // q
        # Verify row t % q of request r sees the context up to its own position.
        visible = context - (q - 1 - t % q)
        pool = position[:visible]
        if r == 0:
            pool = pool[owner[:visible] == 0]
        pick = pool[torch.randperm(pool.numel(), generator=gen, device="cuda")]
        n = min(topk - 37 * (t % 5), pick.numel())
        tokens[t, :n] = pick[:n].int()

    local = int(local_slot.max()) + 1
    blocks_per_req = -(-local // block)
    outs, lses = [], []
    for rank in range(dcp):
        # FlashInfer aligns packed block strides to its 32-row pages.
        cache = packed_cache(reqs * blocks_per_req, block, FP8_ROW, gap=block)
        order = torch.randperm(reqs * blocks_per_req, generator=gen, device="cuda")
        table = order.view(reqs, blocks_per_req).int()
        mine = position[owner == rank]
        slot = local_slot[mine]
        for r in range(reqs):
            physical = table[r, slot // block].long()
            cache[physical, slot % block] = kv[r, mine].view(torch.uint8)
        md = SimpleNamespace(
            req_id_per_token=torch.arange(rows, dtype=torch.int32, device="cuda") // q,
            block_table=table,
            block_size=block,
        )
        box = torch.zeros(2, cap, 128, FP8_ROW, dtype=torch.uint8, device="cuda")
        box[1, :rows] = query  # parity 1
        core = SimpleNamespace(
            fp8=True,
            impl=impl,
            layer=layer,
            dcp=dcp,
            parity=1,
            box=SimpleNamespace(q=box),
            indexer=SimpleNamespace(
                dcp_rank=rank, cp_kv_cache_interleave_size=interleave
            ),
        )
        if cap > 64:
            # Wide rows read the widened box passed in, not the layer's own 64-row
            # box (here: garbage).
            own = torch.full_like(box[:, :64], 0x7F)
            core.box = SimpleNamespace(q=own)
            row, lse = WideDsa.dcp_core(
                core, cache, tokens, md, rows, SimpleNamespace(q=box)
            )
        else:
            row, lse = WideDsa.dcp_core(core, cache, tokens, md, rows)
        outs.append(row.view(rows, -1, 512)[:, :heads].float())
        lses.append(lse.view(rows, -1)[:, :heads].float())
        if rank:
            assert torch.isinf(lses[-1][:q]).all(), "an empty shard must not count"

    lse = torch.stack(lses)
    if not impl.lse_base_on_e:
        lse = lse * torch.log(torch.tensor(2.0, device="cuda"))
    weights = torch.softmax(lse, dim=0).unsqueeze(-1)
    merged = (weights * torch.stack(outs).nan_to_num()).sum(0)

    qf = query[:, :heads].view(e4m3).float()
    for t in range(rows):
        picked = tokens[t][tokens[t] >= 0].long()
        keys = kv[t // q, picked].float()
        probs = torch.softmax(qf[t] @ keys.T * impl.scale, dim=-1)
        ref = probs @ keys[:, :512]
        torch.testing.assert_close(merged[t], ref, atol=3e-2, rtol=3e-2)


# Rubin DCP16's per-rank geometry on one node's 4 GPUs: 6 heads per rank (TP16), the
# FP8 q mailbox padded to 128 heads (trtllm-gen), the widened box of DL3's widest step.
WIDE_HEADS, WIDE_Q_HEADS, WIDE_CAP = 6, 128, 512
WIDE_GATE = WIDE_HEADS * V
WIDE_PITCH = 2112 + WIDE_GATE + INDEX_DIM + INDEX_HEADS
# (rows, q): DL7 (q8, 32 requests at most) and DL3 (q4, 128) wide rows, one at 96 on
# the whole-grid poll, the others above it on the tile poll.
WIDE_MAILBOX_CASES = [(96, 8), (104, 8), (184, 8), (256, 8), (128, 4), (256, 4)]
WIDE_MAILBOX_CASES += [(512, 4)]


def wide_layer_inputs(batch, seed):
    """Random packed projection rows, layer weights (W_UV outputs of unit scale) and
    per-tensor FP8 caches of one DSA layer at 6 heads per rank."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    bf, dev = torch.bfloat16, "cuda"

    def randn(*shape, scale=1.0):
        return (torch.randn(*shape, generator=gen, device=dev) * scale).to(bf)

    weights = (
        (torch.rand(1536, generator=gen, device=dev) + 0.5).to(bf),
        (torch.rand(512, generator=gen, device=dev) + 0.5).to(bf),
        randn(WIDE_HEADS * 192, 1536, scale=1 / 40),
        randn(WIDE_HEADS, 128, 512, scale=1 / 12),
        randn(WIDE_HEADS, 128, 512, scale=1 / 16).transpose(1, 2),
    )
    blocks = -(-(batch + 1) // UNIT) + 1
    mla = packed_cache(blocks, UNIT, FP8_ROW)
    index = packed_cache(blocks, UNIT, INDEX_ROW)
    slots = torch.randperm(blocks * UNIT, generator=gen, device=dev)[:batch]
    slots[1::7] = -1  # padded rows (or other ranks' tokens) write nothing
    return randn(batch, WIDE_PITCH), weights, mla, index, slots


def _wide_mailbox_case(box, size, rank, batch, q, skew=0):
    """One (rows, q) case on this DCP rank: MlaQuery publishes this rank's query
    heads into every rank's q mailbox, then DcpValue merges every rank's partial rows
    of this rank's heads, checked against the all-gathered tensors. ``skew``: rank 1
    starts its query this many clock cycles late (the other ranks' resident tiles
    poll for its tiles meanwhile)."""
    import cutlass
    import torch.distributed as dist

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        layout,
        stream,
        to_cute,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa import wide

    dev, bf = box.q.device, torch.bfloat16
    heads, q_heads = WIDE_HEADS, WIDE_Q_HEADS
    unit = torch.ones(1, device=dev, dtype=torch.float32)
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    parity, s = 0, stream(dev)
    mailbox, mc = to_cute(box.q[parity]), box.mc(parity)
    ids = cutlass.Int32(parity), cutlass.Int32(rank)
    projected, weights, mla, index, slots = wide_layer_inputs(batch, 100 + rank)
    assert weights[4].stride() == (128 * 512, 1, 512)
    specs = tuple(layout(w) for w in (*weights, mla, index))

    def kernels(rows, dcp=size):
        args = (rows, heads, WIDE_PITCH, specs, rank, q, dcp, q_heads if dcp else 0)
        return wide.compile_core(*args, sms, fp8=True, cap=box.cap)

    def query_args(cache, query, q_c):
        args = (to_cute(projected), *(to_cute(w) for w in weights[:4]))
        args += (to_cute(cache), to_cute(slots, 8), to_cute(unit, 4), to_cute(unit, 4))
        return (*args, cutlass.Float32(EPS), to_cute(query), to_cute(q_c))

    # The query: every rank's box holds every rank's query rows (bit-identical).
    mq, _, value = kernels(batch)
    query = torch.zeros(batch, heads, 576, device=dev, dtype=torch.uint8)
    q_c = torch.zeros(batch, 1536, device=dev, dtype=bf)
    tp_mla = packed_cache(mla.shape[0], UNIT, FP8_ROW)  # mla's layout (the specs')
    assert tp_mla.stride() == mla.stride()
    tp_mla.copy_(mla)
    torch.accelerator.synchronize()
    dist.barrier()
    if skew and rank == 1:
        torch.cuda._sleep(skew)
    mq(*query_args(mla, query, q_c), mailbox, mc, ids[1], s)
    # The publish and its poll leave the query rows, q_c and KV rows as the
    # query without DCP computes them (bitwise).
    tp_query, tp_q_c = torch.zeros_like(query), torch.zeros_like(q_c)
    kernels(batch, 0)[0](*query_args(tp_mla, tp_query, tp_q_c), s)
    assert torch.equal(query, tp_query) and torch.equal(q_c, tp_q_c)
    assert torch.equal(mla, tp_mla)
    queries = [torch.empty_like(query) for _ in range(size)]
    dist.all_gather(queries, query)
    gathered = box.q[parity, :batch]
    assert torch.equal(gathered[:, : size * heads], torch.cat(queries, 1))
    assert (gathered[:, size * heads :] == 0).all()

    # The value: partial rows / LSEs of every rank's heads over the local shard.
    gen = torch.Generator(device=dev).manual_seed(rank)
    reqs = batch // q
    rows = torch.randn(batch, q_heads, 512, generator=gen, device=dev).to(bf)
    lse = torch.randn(batch, q_heads, generator=gen, device=dev) * 3
    lse[torch.rand(batch, q_heads, generator=gen, device=dev) < 0.05] = float("-inf")
    lens = torch.randint(0, 4, (reqs,), generator=gen, device=dev).int()
    shared = torch.Generator(device=dev).manual_seed(7)  # the same on every rank
    totals = torch.randint(0, 3, (reqs,), generator=shared, device=dev).int()
    w_uv = weights[4]

    def merged(kernel, start, n):
        out = torch.zeros(n, WIDE_GATE, device=dev, dtype=bf)
        gh = size * heads
        view = rows[start : start + n].as_strided(
            (n, 1, gh, 512), (q_heads * 512, 0, 512, 1)
        )
        lse_view = lse[start : start + n].as_strided((n, 1, gh), (q_heads, 0, 1))
        r0, r1 = start // q, (start + n) // q
        v_args = (to_cute(view), to_cute(lse_view, 4), to_cute(lens[r0:r1], 4))
        v_args += (to_cute(box.partial[parity]), to_cute(box.lse[parity], 4))
        v_args += (to_cute(box.peers, 8), to_cute(box.lse_peers, 8), mailbox)
        gate = to_cute(projected[start : start + n, 2112 : 2112 + WIDE_GATE])
        v_args += (to_cute(w_uv), gate, to_cute(totals[r0:r1], 4), to_cute(out))
        kernel(*v_args, *ids, s)
        torch.accelerator.synchronize()
        dist.barrier()
        return out

    out = merged(value, 0, batch)
    assert (box.q[parity, :batch, : size * heads].view(torch.int32) == -1).all()

    everything = [torch.empty_like(t) for t in (rows, lse, lens) for _ in range(size)]
    for i, t in enumerate((rows, lse, lens)):
        dist.all_gather(everything[i * size : (i + 1) * size], t)
    mine = slice(rank * heads, (rank + 1) * heads)
    all_rows = torch.stack(everything[:size])[:, :, mine]
    all_lse = torch.stack(everything[size : 2 * size])[:, :, mine]
    all_lens = torch.stack(everything[2 * size :]).repeat_interleave(q, 1)
    live = (all_lens > 0).unsqueeze(-1) & torch.isfinite(all_lse)
    lse2 = all_lse * 1.4426950408889634  # FlashInfer's base-e LSE in base 2
    lse2 = torch.where(live, lse2, torch.full_like(all_lse, -3.0e38))
    weight = torch.exp2(lse2 - lse2.amax(0))
    weight = torch.where(live, weight, torch.zeros_like(weight))
    weight = weight / weight.sum(0).clamp_min(1e-30)
    latent = (weight.unsqueeze(-1) * all_rows.float()).sum(0).to(bf)
    value_ref = torch.einsum("bhk,hkv->bhv", latent.float(), w_uv.float()).to(bf)
    gate = projected[:, 2112 : 2112 + WIDE_GATE].float().view(batch, heads, 128)
    ref = (value_ref.float() * torch.sigmoid(gate)).reshape(batch, WIDE_GATE)
    ref[(totals.repeat_interleave(q) == 0)] = 0
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)

    # Every row as narrower kernels compute it, bit for bit.
    rows_n = next(n for n in (64, 32, 16, 8, q) if batch % n == 0 and n % q == 0)
    _, _, chunk_value = kernels(rows_n)
    chunks = [merged(chunk_value, b, rows_n) for b in range(0, batch, rows_n)]
    assert torch.equal(out, torch.cat(chunks))


def _wide_mailbox_worker(rank, size, port, cases, skew=0):
    """One DCP rank of WideDsa's DCP kernels over every (rows, q) case."""
    import torch.distributed as dist

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.mla.dcp import DcpBox

    torch.accelerator.set_device_index(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=size,
        device_id=dev,
    )
    group = dist.group.WORLD
    box = DcpBox(group, dev, size, rank, WIDE_HEADS, cap=WIDE_CAP, q_heads=WIDE_Q_HEADS)
    for batch, q in cases:
        _wide_mailbox_case(box, size, rank, batch, q, skew)
    dist.destroy_process_group()


@pytest.mark.skipif(torch.accelerator.device_count() < 4, reason="needs 4 GPUs")
@pytest.mark.parametrize("skew", [0, 200_000_000], ids=["aligned", "rank1_late"])
def test_wide_dcp_mailboxes(skew):
    """DCP4 over one node's NVLink multicast at the wide rows of DL7 / DL3 (up to
    512), with WideDsa's per-rank geometry (6 heads, 128-head FP8 q mailbox, a
    512-row box): the gathered q mailbox, DcpValue's LSE merge, W_UV and gate, and
    the q mailbox re-armed for the next layer. ``rank1_late``: rank 1's query starts
    ~0.1 s late, so the other ranks' first tiles poll while its later tiles are not
    yet scheduled anywhere (the tile poll's progress, hazard H1); a hang fails."""
    import torch.multiprocessing as mp

    from vllm.utils.network_utils import get_open_port

    args = (4, get_open_port(), WIDE_MAILBOX_CASES, skew)
    context = mp.start_processes(
        _wide_mailbox_worker, args=args, nprocs=4, join=False, start_method="spawn"
    )
    deadline = time.monotonic() + 1800
    while not context.join(timeout=10):
        if time.monotonic() > deadline:
            for process in context.processes:
                process.kill()
            pytest.fail("DCP4 wide mailboxes: no progress in 30 min (a hang?)")
