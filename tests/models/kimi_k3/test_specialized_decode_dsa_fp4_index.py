# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 specialized decode on the MXFP4 DSA indexer cache (68 B/token,
indexer_kv_dtype="mxfp4"): the writers (DsaIndex, the index fronts) against the
stock MXFP4 writers, and their query in DeepGEMM's layout."""

import pytest
import torch

from tests.models.kimi_k3 import test_specialized_decode_dsa as dsa
from vllm.platforms import current_platform

SM100 = pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="the specialized decode kernels need SM100/SM103",
)

UNIT, FP4_ROW = 64, 68  # indexer page tokens; MXFP4 indexer row bytes
GAP = 36  # rows of the other layers' pages between two blocks of a packed cache
HEADS = 12  # TP8
GATE = HEADS * 128
PITCH = 2112 + GATE + 128 + 32  # packed [q_a | kv_a | gate | wk | wv] row
KEY = 2112 + GATE  # wk column offset in the packed row
EPS = 1e-6
TOPK = 2048
WEIGHT_SCALE = 128**-0.5 * 32**-0.5  # KimiK3DSAIndexer's softmax x weight scale
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def fp4_pages(index):
    """Each MXFP4 indexer page's bytes: [64 x 64 B E2M1 | 64 x 4 UE8M0]."""
    return torch.as_strided(
        index, (index.shape[0], index.shape[1] * FP4_ROW), (index.stride(0), 1)
    )


def dequant(codes, scales):
    """E2M1 bytes [..., 64] (low nibble first) and UE8M0 bytes [..., 4] ->
    FP32 [..., 128]."""
    table = torch.tensor(E2M1, device=codes.device)
    table = torch.cat([table, -table])
    nib = torch.stack([codes & 15, codes >> 4], -1).flatten(-2).long()
    values = table[nib].unflatten(-1, (4, 32))
    scale = torch.exp2(scales.float() - 127).unsqueeze(-1)
    return (values * scale).flatten(-2)


def adversarial_heads(iq):
    """Heads whose block maxima sit at and next to 6 x 2^k (where the stock
    writer's ceil(log2(amax / 6)) changes) and an all-zero head."""
    rows = iq.view(iq.shape[0], 32, 4, 32)
    for k, head in enumerate(range(1, 9)):
        base = 6.0 * 2.0 ** (k - 4)
        rows[:, head] *= 0.05 * 2.0 ** (k - 4)  # the planted value is the max
        for b, ulps in enumerate((0, 1, -1, 2)):
            value = torch.tensor(base, dtype=torch.bfloat16)
            bits = value.view(torch.int16) + ulps
            rows[:, head, b, 7] = bits.view(torch.bfloat16).item()
    rows[0, 0] = 0


def check_query_layout(q_rows, batch):
    """The FP4 query of ``batch`` rows (fp4_query_views): contiguous views of the
    buffer's E2M1 rows from its start and their UE8M0 x4 at batch x 2 KiB, and no
    other byte of the buffer written (filled with 0x77 before)."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.index import (
        fp4_query_views,
    )

    packed, scales = fp4_query_views(q_rows, batch, batch)
    assert packed.is_contiguous() and scales.is_contiguous()
    base = q_rows.data_ptr()
    assert (packed.data_ptr(), scales.data_ptr()) == (base, base + batch * 2048)
    assert bool((q_rows.view(-1)[batch * (2048 + 128) :] == 0x77).all())
    return packed, scales


@SM100
@pytest.mark.parametrize("gap", [0, GAP])
@pytest.mark.parametrize("batch", [1, 5, 16, 40, 64, 96, 320, 512])
def test_fp4_indexer_inputs(batch, gap):
    """DsaIndex on an MXFP4 indexer cache writes the stock writers' bytes:
    mxfp4_q_quant's E2M1 query and UE8M0 scales (in DeepGEMM's layout, nothing
    else of the FP8 query buffer) and its BF16 head weights, and
    mxfp4_k_norm_quant_insert's key bytes and scales (the stock writer's log2
    rounding next to 6 x 2^k included), into packed pages."""
    import cutlass

    from vllm.models.kimi_k3.nvidia.ops.dsa_indexer_mxfp4 import (
        mxfp4_k_norm_quant_insert,
        mxfp4_q_quant,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        fake,
        layout,
        stream,
        to_cute,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.query_stage import (  # noqa: E501
        compile_index,
    )

    torch.manual_seed(batch)
    bf, dev = torch.bfloat16, "cuda"
    projected = torch.randn(batch, PITCH, device=dev, dtype=bf)
    spread = torch.logspace(-3, 3, batch, device=dev).to(bf)[:, None]
    projected[:, KEY : KEY + 128] *= spread
    iq = torch.randn(batch, 32 * 128, device=dev, dtype=bf)
    adversarial_heads(iq)
    norm_w = torch.rand(128, device=dev) + 0.5
    norm_b = torch.randn(128, device=dev) / 10
    blocks = -(-batch // UNIT) + 8
    index = dsa.packed_cache(blocks, UNIT, FP4_ROW, gap)
    slots = torch.randperm(blocks * UNIT - 1, device=dev)[:batch]
    slots[0] = blocks * UNIT - 1  # the last entry
    if batch > 2:
        slots[1] = -1  # a padded row (or a token another DCP rank owns)
    kernel = compile_index(batch, HEADS, PITCH, fake(*layout(index)))
    q_rows = torch.full((batch, 32, 128), 0x77, device=dev, dtype=torch.uint8)
    head_w = torch.zeros(batch, 32, device=dev, dtype=bf)
    args = (to_cute(projected), to_cute(iq), to_cute(norm_w, 4), to_cute(norm_b, 4))
    args += (to_cute(index), to_cute(slots, 8), to_cute(q_rows), to_cute(head_w, 4))
    kernel(*args, cutlass.Float32(EPS), stream(dev))

    wv = projected[:, KEY + 128 : KEY + 160]
    ref_q, ref_s, ref_w = mxfp4_q_quant(iq.view(batch, 32, 128), wv, WEIGHT_SCALE, 32)
    packed, scales = check_query_layout(q_rows, batch)
    assert torch.equal(packed, ref_q)
    assert torch.equal(scales, ref_s)
    # The stock decode casts the FP32 weights to BF16 for SM100's MQA logits.
    assert torch.equal(head_w, ref_w.to(bf))

    stock = dsa.packed_cache(blocks, UNIT, FP4_ROW, gap)
    topk = torch.empty(batch, TOPK, device=dev, dtype=torch.int32)
    k = projected[:, KEY : KEY + 128]
    mxfp4_k_norm_quant_insert(k, norm_w, norm_b, EPS, slots, stock, topk, batch)
    assert torch.equal(fp4_pages(index), fp4_pages(stock))


FRONT_BATCHES = [1, 5, 16, 40, 64, 80, 144, 320]


@SM100
@pytest.mark.parametrize("kernel_kind", ["mma", "mma-qc", "cluster"])
@pytest.mark.parametrize("batch", FRONT_BATCHES)
def test_fp4_index_front(monkeypatch, batch, kernel_kind):
    """The fused index fronts on an MXFP4 indexer cache (IndexFrontMma, also from
    the normalized q_c rows, and the cluster IndexFront up to 64 rows): the key
    bytes exactly the stock insert's, the head weights exactly bf16(wv / 64), and
    the FP4 query (in DeepGEMM's layout, nothing else of the buffer) as
    mxfp4_q_quant of an FP32 projection of the same q_c except where the BF16
    projection rounds differently (within one E2M1 step of the block's scale)."""
    import cutlass

    from vllm.models.kimi_k3.nvidia.ops.dsa_indexer_mxfp4 import (
        mxfp4_k_norm_quant_insert,
        mxfp4_q_quant,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        layout,
        stream,
        to_cute,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa import (
        query_stage,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.projection import (  # noqa: E501
        weight_map,
    )

    if kernel_kind == "cluster" and batch > 64:
        pytest.skip("IndexFront serves at most 64 rows")
    torch.manual_seed(batch)
    bf, dev = torch.bfloat16, "cuda"
    projected = torch.randn(batch, PITCH, device=dev, dtype=bf) * 2
    q_norm = torch.rand(1536, device=dev, dtype=bf) + 0.5
    wq = torch.randn(32 * 128, 1536, device=dev, dtype=bf) / 30
    norm_w = torch.rand(128, device=dev) + 0.5
    norm_b = torch.randn(128, device=dev) / 10
    blocks = -(-batch // UNIT) + 8
    index = dsa.packed_cache(blocks, UNIT, FP4_ROW, GAP)
    slots = torch.randperm(blocks * UNIT - 1, device=dev)[:batch]
    if batch > 2:
        slots[1] = -1
    normalized = kernel_kind == "mma-qc"
    kind = "cluster" if kernel_kind == "cluster" else "mma"
    monkeypatch.setattr(query_stage, "FRONT_KERNEL", kind)
    dev_index = torch.accelerator.current_device_index()
    specs = layout(wq), layout(index)
    kernel = query_stage.compile_front(
        batch, HEADS, PITCH, *specs, dev_index, normalized
    )
    q_rows = torch.full((batch, 32, 128), 0x77, device=dev, dtype=torch.uint8)
    head_w = torch.zeros(batch, 32, device=dev, dtype=bf)
    q_c = dsa.rms_norm(projected[:, :1536], q_norm)
    args = (to_cute(projected),)
    weights = to_cute(wq)
    if kernel_kind != "cluster":  # the rows, and the weights' TMA map
        args += (to_cute(q_c) if normalized else to_cute(projected),)
        wq_map = weight_map(wq, 128)
        weights = cutlass.Int64(wq_map.data_ptr())
    args += (to_cute(q_norm), weights)
    args += (to_cute(norm_w, 4), to_cute(norm_b, 4), to_cute(index))
    args += (to_cute(slots, 8), to_cute(q_rows), to_cute(head_w, 4))
    kernel(*args, cutlass.Float32(EPS), cutlass.Float32(EPS), stream(dev))

    wv = projected[:, KEY + 128 : KEY + 160]
    assert torch.equal(head_w, (wv.float() / 64).to(bf))
    iq = (q_c.float() @ wq.float().T).to(bf)
    ref_q, ref_s, _ = mxfp4_q_quant(iq.view(batch, 32, 128), wv, 1 / 64, 32)
    packed, scales = check_query_layout(q_rows, batch)
    ours_s, ref_b = (s.view(torch.uint8).view(batch, 32, 4) for s in (scales, ref_s))
    ours, theirs = dequant(packed, ours_s), dequant(ref_q, ref_b)
    step = torch.exp2(torch.maximum(ours_s, ref_b).float() - 127)
    bound = 2 * step.repeat_interleave(32, -1)  # one E2M1 step at the top binade
    assert bool(((ours - theirs).abs() <= bound).all())
    assert (packed != ref_q).float().mean() < 0.05
    assert (scales != ref_s).float().mean() < 0.01

    stock = dsa.packed_cache(blocks, UNIT, FP4_ROW, GAP)
    topk = torch.empty(batch, TOPK, device=dev, dtype=torch.int32)
    k = projected[:, KEY : KEY + 128]
    mxfp4_k_norm_quant_insert(k, norm_w, norm_b, EPS, slots, stock, topk, batch)
    assert torch.equal(fp4_pages(index), fp4_pages(stock))
