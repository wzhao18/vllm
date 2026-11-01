# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 specialized decode, wide verify rows (> 64 token rows, Rubin):
WideDsa at the rows ``dsa.wide.DSA_WIDE`` admits (65-96): its admission, index
policies, MlaQuery grid residency and kernels at those rows, its DCP core; and the
fused indexer key insert from the packed in-projection rows that seam S2-DSA feeds."""

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
    verify shapes that ``DSA_WIDE`` lists, at most ``WIDE_MAX_ROWS`` (MlaQuery's
    gathered-q poll needs a co-resident grid); else None (S2-DSA)."""
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
    """Hazard H1: at wide rows every MlaQuery CTA polls the gathered q mailbox that
    other ranks' CTAs fill, so the whole grid must be resident at once. The compiled
    DSA query (DCP16, 6 heads, FP8, 96-row box) fits at 72 / 96 rows
    (cuOccupancyMaxActiveClusters, checked by compile_core), and the check refuses
    the 128-row grid that WIDE_MAX_ROWS keeps on S2-DSA."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        layout,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa import wide
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


@pytest.mark.parametrize("gap", [GAP, FAR])
@pytest.mark.parametrize("batch", [72, 96])
def test_fp8_query_and_cache_insert(batch, gap):
    """FlashInfer core: the per-tensor FP8 query rows [B, H, 576] and the FP8 KV row
    in the packed MLA blocks, as the stock FlashInfer path quantizes and writes them
    (concat_and_cache_mla "fp8"), with non-unit scales."""
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
    k_scale = torch.full((1,), 2.0, device=dev)
    args = (to_cute(projected), *(to_cute(w) for w in weights[:4]), to_cute(mla))
    args += (to_cute(slots, 8), to_cute(1 / q_scale, 4), to_cute(1 / k_scale, 4))
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
