# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Ported from the source branch's tests/models/kimi_k3/test_specialized_decode_dsa_fp4_index.py for the SM107
# specialized decode (vllm/models/kimi_k3/nvidia/specialized_decode/sm_107).
"""Kimi K3.1 specialized decode on the MXFP4 DSA indexer cache (68 B/token,
indexer_kv_dtype="mxfp4"): the writers (DsaIndex, the index fronts) against the
stock MXFP4 writers; their query in DeepGEMM's layout, which the indexer stage and
WideDsa read without copies, against the stock scoring and logits, also captured
first as the specialized decode runs them; the bind-time key format check; and the
DSA layer tests of test_k3_sm107_dsa.py on an MXFP4 indexer cache, at TP
and under a DCP16 launch (KIMI_K3_SD_DCP16_INIT), with the per-tensor FP8 MLA cache
and, in a tree with the NVFP4 specialized decode, nvfp4_ds_mla."""

import inspect
import math
from functools import partial
from types import SimpleNamespace

import pytest
import torch

from tests.models.kimi_k3.nvidia.specialized_decode.sm_107 import (
    test_k3_sm107_dsa as dsa,
)
from tests.models.kimi_k3.nvidia.specialized_decode.sm_107.test_k3_sm107_dsa import (  # noqa: F401
    dcp8_group,  # (fixture)
)
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_deep_gemm

# The SM107 specialized decode (specialized_decode/sm_107) serves VR200 only.
_SM107_ONLY = pytest.mark.skipif(
    not current_platform.is_device_capability((10, 7)),
    reason="the SM107 specialized decode runs on compute capability 10.7 only",
)
pytestmark = _SM107_ONLY

SM100 = pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="the specialized decode kernels need SM100/SM103",
)
DEEPGEMM = pytest.mark.skipif(
    not (current_platform.is_device_capability_family(100) and has_deep_gemm()),
    reason="DeepGEMM MXFP4 MQA logits need SM100",
)
DCP16 = pytest.mark.skipif(
    dsa.DCP16_INIT is None, reason="needs a 16-rank launch (DCP16)"
)
# nvfp4_ds_mla layer cases need the NVFP4 specialized decode (0650b801f6 ..
# 0af1e08b57), whose tests drop test_k3_sm107_dsa.py's NO_NVFP4.
NO_NVFP4 = "nvfp4_ds_mla: this tree has no NVFP4 specialized decode"

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
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.dsa.index import (
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
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.common.host import (
        fake,
        layout,
        stream,
        to_cute,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.dsa.query_stage import (  # noqa: E501
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
    head_w = torch.zeros(batch, 32, device=dev, dtype=torch.float32)
    args = (to_cute(projected), to_cute(iq), to_cute(norm_w, 4), to_cute(norm_b, 4))
    args += (to_cute(index), to_cute(slots, 8), to_cute(q_rows), to_cute(head_w, 4))
    kernel(*args, cutlass.Float32(EPS), stream(dev))

    wv = projected[:, KEY + 128 : KEY + 160]
    ref_q, ref_s, ref_w = mxfp4_q_quant(iq.view(batch, 32, 128), wv, WEIGHT_SCALE, 32)
    packed, scales = check_query_layout(q_rows, batch)
    assert torch.equal(packed, ref_q)
    assert torch.equal(scales, ref_s)
    # Stock Kimi K3 keeps the head weights FP32 (its MQA logits are FP32).
    assert ref_w.dtype == torch.float32 and torch.equal(head_w, ref_w)

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
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.common.host import (
        layout,
        stream,
        to_cute,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.dsa import (
        query_stage,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.gemm.projection import (  # noqa: E501
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
    head_w = torch.zeros(batch, 32, device=dev, dtype=torch.float32)
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
    assert torch.equal(head_w, wv.float() / 64)  # FP32, as stock
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


@SM100
@pytest.mark.parametrize("heads", [12, 6])
def test_mxfp4_key_insert_from_packed_rows(heads):
    """Seam S2-DSA (runtime Attention.stock_dsa) inserts the MXFP4 keys from the
    packed projection rows: KimiK3DSAIndexer._fused_k_insert hands
    mxfp4_k_norm_quant_insert the wk column slice, whose row stride is the packed
    pitch (3040 at DCP16's 6 local heads); it writes the bytes it writes from the
    same keys contiguous."""
    from vllm.models.kimi_k3.nvidia.ops.dsa_indexer_mxfp4 import (
        mxfp4_k_norm_quant_insert,
    )

    torch.manual_seed(heads)
    bf, dev = torch.bfloat16, "cuda"
    tokens = 300
    pitch, key = 2112 + heads * 128 + 160, 2112 + heads * 128
    projected = torch.randn(tokens, pitch, device=dev, dtype=bf)
    k = projected[:, key : key + 128]
    assert k.stride() == (pitch, 1)
    norm_w = torch.rand(128, device=dev) + 0.5
    norm_b = torch.randn(128, device=dev) / 10
    blocks = -(-tokens // UNIT) + 2
    slots = torch.randperm(blocks * UNIT, device=dev)[:tokens]
    slots[1] = -1
    caches = [dsa.packed_cache(blocks, UNIT, FP4_ROW, GAP) for _ in range(2)]
    topk = torch.empty(tokens, 16, device=dev, dtype=torch.int32)
    for keys, cache in zip((k, k.contiguous()), caches):
        mxfp4_k_norm_quant_insert(keys, norm_w, norm_b, EPS, slots, cache, topk, tokens)
    assert torch.equal(fp4_pages(caches[0]), fp4_pages(caches[1]))
    assert int(fp4_pages(caches[0])[:, UNIT * 64 :].count_nonzero()) > 0


@pytest.mark.parametrize(
    "width,use_fp4,fp4",
    [
        (132, False, False),  # FP8
        (68, True, True),  # MXFP4
        (68, False, None),  # FP8 readers on MXFP4 keys
        (132, True, None),  # MXFP4 readers on FP8 keys
    ],
)
def test_check_index_cache(width, use_fp4, fp4):
    """The bind-time key format check of PreparedDsa / WideDsa.prepare_cache: the
    bound cache's width is the format the readers (the scorer: FP8 or MXFP4 keys)
    take (use_fp4_cache)."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.dsa.indexer_stage import (  # noqa: E501
        check_index_cache,
    )

    cache = torch.empty(3, UNIT, width, dtype=torch.uint8)
    indexer = SimpleNamespace(
        use_fp4_cache=use_fp4, k_cache=SimpleNamespace(kv_cache=cache)
    )
    if fp4 is None:
        with pytest.raises(ValueError, match="indexer cache"):
            check_index_cache(indexer)
    else:
        assert check_index_cache(indexer) is fp4


def mxfp4_readers(monkeypatch, batch, rows):
    """A step of ``batch`` rows whose first ``rows`` are requests' (the rest
    CUDA-graph padding) over keys the stock MXFP4 writer inserted, its query
    written by DsaIndex, scored by the stock scoring (KimiK3DSAIndexer.score of the
    same projection rows, as the stock forward's warmup run before each capture
    scores it): the shared top-k buffer, the stock top-k, the readers of DsaIndex's
    MXFP4 query (no copies: index.fp4_query_views; sm_107's scorer: TP its top-k
    ids, DCP its logits in DeepGEMM's layout) and DeepGEMM's MXFP4 paged logits of
    the stock writer's query at FP32 (the logits reference)."""
    import cutlass

    import vllm.model_executor.layers.sparse_attn_indexer as sparse_indexer
    from vllm.config import CUDAGraphMode
    from vllm.model_executor.layers.sparse_attn_indexer import (
        kv_cache_as_quant_view,
    )
    from vllm.models.kimi_k3.nvidia.dsa import KimiK3DSAIndexer
    from vllm.models.kimi_k3.nvidia.ops.dsa_indexer_mxfp4 import (
        mxfp4_k_norm_quant_insert,
        mxfp4_q_quant,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.common.host import (
        fake,
        layout,
        stream,
        to_cute,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.dsa.indexer import (  # noqa: E501
        FusedIndexer,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.dsa.indexer_stage import (  # noqa: E501
        IndexerStage,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.dsa.query_stage import (  # noqa: E501
        compile_index,
    )
    from vllm.utils.deep_gemm import (
        fp8_fp4_paged_mqa_logits,
        get_num_sms,
        get_paged_mqa_logits_metadata,
        mqa_logits_mx_inputs,
    )
    from vllm.v1.attention.backends.mla.indexer import (
        DeepSeekV32IndexerDecodeMetadata,
        DeepseekV32IndexerMetadata,
    )

    torch.manual_seed(batch)
    bf, dev = torch.bfloat16, "cuda"
    # One query row per request (SM100's varlen decode), below and past the top-k.
    lens = [300 + 997 * r % 6500 for r in range(rows)]
    seq_lens = torch.tensor(lens, dtype=torch.int32, device=dev)
    max_blocks = -(-max(lens) // UNIT)
    num_blocks = rows * max_blocks + 1
    perm = torch.randperm(num_blocks, device=dev).int()
    block_table = perm[: rows * max_blocks].view(rows, max_blocks).contiguous()
    req = torch.repeat_interleave(torch.arange(rows, device=dev), seq_lens.long())
    pos = torch.cat([torch.arange(n, device=dev) for n in lens])
    slots = block_table[req, pos // UNIT].long() * UNIT + pos % UNIT
    cache = dsa.packed_cache(num_blocks, UNIT, FP4_ROW, GAP)
    keys = torch.randn(slots.shape[0], 128, dtype=bf, device=dev)
    k_w = torch.rand(128, device=dev) + 0.5
    k_b = torch.randn(128, device=dev) / 10
    filled = torch.empty(slots.shape[0], 16, dtype=torch.int32, device=dev)
    mxfp4_k_norm_quant_insert(keys, k_w, k_b, EPS, slots, cache, filled, len(slots))

    # DsaIndex writes the step's query (no key: every slot -1).
    projected = torch.randn(batch, PITCH, device=dev, dtype=bf)
    iq = torch.randn(batch, 32 * 128, device=dev, dtype=bf)
    adversarial_heads(iq)
    kernel = compile_index(batch, HEADS, PITCH, fake(*layout(cache)))
    q_rows = torch.full((batch, 32, 128), 0x77, device=dev, dtype=torch.uint8)
    head_w = torch.zeros(batch, 32, device=dev, dtype=torch.float32)
    norm = torch.ones(128, device=dev)
    none = torch.full((batch,), -1, dtype=torch.int64, device=dev)
    args = (to_cute(projected), to_cute(iq), to_cute(norm, 4), to_cute(norm, 4))
    args += (to_cute(cache), to_cute(none, 8), to_cute(q_rows), to_cute(head_w, 4))
    kernel(*args, cutlass.Float32(EPS), stream(dev))

    prefix = "model.layers.3.self_attn.indexer.k_cache"
    topk = torch.full((batch, TOPK), -1, dtype=torch.int32, device=dev)
    indexer = SimpleNamespace(
        use_fp4_cache=True,
        index_softmax_scale=128**-0.5,
        indexer_weight_scale=32**-0.5,
        padded_index_num_heads=32,
        k_cache=SimpleNamespace(prefix=prefix, kv_cache=cache),
        quant_block_size=128,
        scale_fmt="ue8m0",
        topk_tokens=TOPK,
        index_head_dim=128,
        max_model_len=max_blocks * UNIT,
        max_total_seq_len=max_blocks * UNIT,
        topk_indices_buffer=topk,
        use_fused_k_insert=True,
        use_pcp=False,
        _parallel_config=SimpleNamespace(
            pcp_shard_decode_requests=False, cp_kv_cache_interleave_size=1
        ),
        dense_mha_metadata_layer_name="",
        dcp_rank=0,
        dcp_world_size=1,
        cp_kv_cache_interleave_size=1,
        topk_backend="auto",
        shardq_layer_name="",
        dcp_local_topk=0,
        dcp_prefill_merge_tie_words=0,
    )
    context_lens = seq_lens.view(-1, 1).contiguous()
    indices = torch.arange(rows, dtype=torch.int32, device=dev)
    decode = DeepSeekV32IndexerDecodeMetadata(
        block_table=block_table,
        seq_lens=context_lens,
        decode_lens=torch.ones(rows, dtype=torch.int32, device=dev),
        requires_padding=False,
        schedule_metadata=get_paged_mqa_logits_metadata(
            context_lens, UNIT, get_num_sms(), indices
        ),
        indices=indices,
    )
    metadata = DeepseekV32IndexerMetadata(
        seq_lens=seq_lens,
        max_seq_len=max(lens),
        slot_mapping=torch.empty(0, dtype=torch.int64, device=dev),
        num_decodes=rows,
        num_decode_tokens=rows,
        num_prefills=0,
        num_prefill_tokens=0,
        decode=decode,
    )
    context = SimpleNamespace(
        attn_metadata={prefix: metadata},
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )
    monkeypatch.setattr(sparse_indexer, "get_forward_context", lambda: context)
    import vllm.forward_context

    monkeypatch.setattr(vllm.forward_context, "get_forward_context", lambda: context)
    q = iq.view(batch, 32, 128)[:rows]
    wv = projected[:rows, KEY + 128 : KEY + 160]
    KimiK3DSAIndexer.score(indexer, projected[:rows], q, wv, None)
    stock = topk[:rows].clone()
    assert bool((stock >= 0).any(-1).all())
    ref_q, ref_s, ref_w = mxfp4_q_quant(q, wv, WEIGHT_SCALE, 32)
    logits_dtype = torch.float32  # stock Kimi K3's indexer logits on main
    ref_s, ref_w = mqa_logits_mx_inputs(ref_q, ref_s, ref_w, logits_dtype)
    ref = fp8_fp4_paged_mqa_logits(
        (ref_q.view(torch.int8).unsqueeze(1), ref_s.unsqueeze(1)),
        kv_cache_as_quant_view(cache, 128, True),
        ref_w,
        context_lens,
        block_table,
        decode.schedule_metadata,
        max_model_len=indexer.max_model_len,
        logits_dtype=logits_dtype,
        indices=indices,
    )

    # TP: the scorer's FP32 logits and IndexerSelect ("compact_ids") into the
    # shared top-k buffer.
    stage = IndexerStage(indexer, batch)
    device = torch.device(dev)
    stage.fused = FusedIndexer(indexer, [(batch, 1)], device, batch, ids=True)
    # DCP's scorer (this rank's keys in DeepGEMM's logits layout; one rank).
    dcp = FusedIndexer(indexer, [(batch, 1)], device, batch, dcp=True)
    selects = {"IndexerStage.run": partial(stage.run, q_rows, head_w, batch)}

    def logits():
        return dcp.run_dcp(q_rows, head_w, decode, batch, 1)[0]

    return SimpleNamespace(
        lens=lens, topk=topk, stock=stock, ref=ref, selects=selects, logits=logits
    )


def check_readers_logits(step, logits):
    """The DCP scorer's logits (the FP32 DCP top-k's input) are DeepGEMM's MXFP4
    FP32 logits of the stock query, bit for bit."""
    assert logits.dtype == step.ref.dtype == torch.float32
    for i, n in enumerate(step.lens):
        assert torch.equal(logits[i, :n], step.ref[i, :n]), i


def check_readers_topk(step, name):
    """A reader's top-k against the stock scoring's. cooperative_topk and
    persistent_topk place the ids, and pick among logits tied at the k-th, by
    atomics, so the stock op does not repeat its own order: per row, as many
    distinct ids below the row's length as stock selects, and the same logits."""
    for i, n in enumerate(step.lens):
        got, want = (t[i][t[i] >= 0].long() for t in (step.topk, step.stock))
        assert got.numel() == want.numel() == min(n, TOPK), (name, i)
        assert got.unique().numel() == got.numel() and int(got.max()) < n, (name, i)
        logits = step.ref[i]
        same = torch.equal(logits[got].sort().values, logits[want].sort().values)
        assert same, (name, i)


@DEEPGEMM
@pytest.mark.parametrize("batch,rows", [(1, 1), (8, 6), (64, 64), (130, 128)])
@torch.inference_mode()
def test_mxfp4_readers_match_stock(monkeypatch, workspace_init, batch, rows):
    """The readers of DsaIndex's MXFP4 query (no copies: index.fp4_query_views) in
    a step of ``batch`` rows whose first ``rows`` are requests' (the rest CUDA-graph
    padding), over keys the stock MXFP4 writer inserted: IndexerStage.run (sm_107's
    MXFP4 scorer and exact select) selects the stock scoring's top-k
    (KimiK3DSAIndexer.score of the same projection rows), and the DCP scorer's
    logits (the FP32 DCP top-k's) are DeepGEMM's MXFP4 paged logits of the stock
    writer's query, bit for bit."""
    step = mxfp4_readers(monkeypatch, batch, rows)
    for name, select in step.selects.items():
        step.topk.fill_(-1)
        select()
        check_readers_topk(step, name)
    check_readers_logits(step, step.logits())


@DEEPGEMM
@pytest.mark.parametrize("batch,rows", [(8, 8), (130, 128)])
@torch.inference_mode()
def test_mxfp4_readers_capture_after_stock(monkeypatch, workspace_init, batch, rows):
    """The readers as the specialized decode first runs them: captured in CUDA
    graphs right after the stock scoring of the same step, with no eager run of
    their own (runtime.decode plans a step only under capture; the warmup run
    before each capture is the stock forward). Their kernels compiled when they
    were built (FusedIndexer), so nothing compiles under capture (CUDA error 901
    there would stop D's startup), and the replays give the stock top-k and
    DeepGEMM's MXFP4 logits."""
    step = mxfp4_readers(monkeypatch, batch, rows)
    graphs, outs = {}, {}
    for name, run in (*step.selects.items(), ("dcp_logits", step.logits)):
        graphs[name] = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graphs[name]):
            outs[name] = run()
    for name in step.selects:
        step.topk.fill_(-1)
        graphs[name].replay()
        torch.accelerator.synchronize()
        check_readers_topk(step, name)
    graphs["dcp_logits"].replay()
    torch.accelerator.synchronize()
    check_readers_logits(step, outs["dcp_logits"])


# ---------------------------------------------------------------------------------
# test_k3_sm107_dsa.py's DSA layer tests on an MXFP4 indexer cache
# ---------------------------------------------------------------------------------


def fp4_index_nan_free(index):
    """No NaN in an MXFP4 indexer cache: E2M1 has no NaN code, a UE8M0 scale of
    0xFF is NaN."""
    assert not (fp4_pages(index)[:, UNIT * 64 :] == 0xFF).any()


def fp4_exact_topk_widths(index, table, contexts, q, query, weights, ids, world=1):
    """exact_topk_widths on an MXFP4 indexer cache: the exact logits of the MXFP4
    query DsaIndex wrote (``query``: the step's [B, 32, 128] FP8 query rows) over
    the dequantized MXFP4 keys, at stock's precision (deepgemm_head_sum; the MX
    scales are inside the q . k sums)."""
    import torch.distributed as dist

    from vllm.distributed import get_dcp_group
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.dsa.index import (
        fp4_query_views,
    )

    rank = get_dcp_group().rank_in_group if world > 1 else 0
    batch = query.shape[0]
    codes, scales = fp4_query_views(query, batch, batch)
    query = dequant(codes, scales.view(torch.uint8).view(batch, 32, 4))
    topk, spare = ids.shape[1], 512
    pages = fp4_pages(index)
    vals, gids, lengths = [], [], []
    t = 0
    for r, c in enumerate(contexts):
        n = (c + q - rank + world - 1) // world
        blocks = pages[table[r, : -(-n // UNIT)].long()]
        codes = blocks[:, : UNIT * 64].reshape(-1, 64)[:n]
        keys = dequant(codes, blocks[:, UNIT * 64 :].reshape(-1, 4)[:n])
        for j in range(q):
            seen = (c + j + 1 - rank + world - 1) // world
            score = torch.relu(query[t] @ keys[:seen].T)
            logits = dsa.deepgemm_head_sum(score, weights[t])
            v = torch.full((topk + spare,), float("-inf"), device="cuda")
            g = torch.full((topk + spare,), -1, dtype=torch.int64, device="cuda")
            k = min(topk + spare, seen)
            top = torch.topk(logits, k)
            v[:k], g[:k] = top.values, top.indices * world + rank
            vals.append(v)
            gids.append(g)
            lengths.append(c + j + 1)
            t += 1
    vals, gids = torch.stack(vals), torch.stack(gids)
    if world > 1:
        group = get_dcp_group().device_group
        parts = [torch.empty_like(vals) for _ in range(world)]
        dist.all_gather(parts, vals, group=group)
        id_parts = [torch.empty_like(gids) for _ in range(world)]
        dist.all_gather(id_parts, gids, group=group)
        vals, gids = torch.cat(parts, 1), torch.cat(id_parts, 1)
    widest = 0.0
    for t, length in enumerate(lengths):
        got = set(ids[t][ids[t] >= 0].tolist())
        if length <= topk:
            assert got == set(range(length)), t
            continue
        top = torch.topk(vals[t], topk + spare)
        cand = gids[t][top.indices]
        kth = float(top.values[topk - 1])
        step = 2.0 ** (math.floor(math.log2(abs(kth))) - 7) if kth else 2.0**-133
        score = dict(zip(cand.tolist(), top.values.tolist()))
        exact = set(cand[:topk].tolist())
        for g in got ^ exact:
            far = g not in score
            widest = max(widest, math.inf if far else abs(score[g] - kth) / step)
    return widest


def check_new_mla_rows(ours, theirs, kv):
    """check_layer_step's check of the new tokens' MLA rows: fp8 codes one step
    apart where the RMSNorm rounds differently, k_pe exact; nvfp4_ds_mla k_pe exact,
    block scales one E4M3 code apart, few E2M1 codes different and the dequantized
    latents close."""
    if kv == "nvfp4_ds_mla":
        assert torch.equal(ours[:, 256:320], theirs[:, 256:320])
        scales = (ours[:, 320:].int() - theirs[:, 320:].int()).abs()
        assert bool((scales <= 1).all())
        differ = (ours[:, :256] != theirs[:, :256]).sum()
        assert differ <= 0.05 * ours[:, :256].numel()
        a, b = (dsa.nvfp4_rows(r)[:, :512] for r in (ours, theirs))
        assert (a - b).norm() <= 0.05 * b.norm() + 1e-6
        return
    assert kv == "fp8", kv
    assert torch.equal(ours[:, 512:], theirs[:, 512:])
    assert dsa.fp8_codes_adjacent(ours[:, :512], theirs[:, :512])
    assert (ours[:, :512] != theirs[:, :512]).sum() <= 0.05 * ours[:, :512].numel()


def check_fp4_layer_step(step, q, reference=True):
    """check_layer_step on an MXFP4 indexer cache: our top-k is the exact top-k of
    the FP32 logits of the same MXFP4 query and keys (fp4_exact_topk_widths; the
    scorer is DeepGEMM's FP32 kind::mxf4 MMA and head sum, so only the q . k
    order differs: far below one BF16 step, as on FP8 keys); the top-k overlap
    with stock 0.03 looser (stock's query, not ours: a query value the two paths
    round differently moves by an E2M1 step, as #130's MXFP4 DSA layer tests
    relax it), and the new tokens' indexer keys as MXFP4 rows: the UE8M0 scales
    exact, the E2M1 codes within one step where the LayerNorm rounds
    differently, every other byte as it was."""
    if step.width is not None:
        print(
            f"top-k vs exact (BF16 steps): ours {step.width:.4f} stock (vs our "
            f"query) {step.stock_width:.2f}"
        )
        assert step.width <= 0.05, step.width
    same = []
    for t in range(step.topk.shape[0]):
        a, b = set(step.stock_topk[t].tolist()), set(step.topk[t].tolist())
        wide = step.topk.shape[0] > 64 or step.world == 16
        bound = (0.99 if step.width is None or not wide else 0.98) - 0.03
        assert len(a & b) / len(a) > bound, (t, len(a), len(a & b))
        same.append(a == b)
    out, stock = step.out, step.stock
    rows = torch.tensor(same, device=out.device)
    ours, theirs = out[rows].float(), stock[rows].float()
    error = float((ours - theirs).norm() / theirs.norm().clamp(min=1e-30))
    print(f"relative error vs stock (same top-k rows): {error:.4f}")
    torch.testing.assert_close(out[rows], stock[rows], atol=2e-2, rtol=2e-2)
    if reference:
        args = step.attn, step.projected
        ours_ref = dsa.layer_reference(
            *args, step.caches[0], step.mla_table, step.topk, q, step.kv
        )
        stock_ref = dsa.layer_reference(
            *args, step.stock_caches[0], step.mla_table, step.stock_topk, q, step.kv
        )
        rel_error = lambda x, ref: float((x.float() - ref).norm() / ref.norm())  # noqa: E731
        ours_error, stock_error = rel_error(out, ours_ref), rel_error(stock, stock_ref)
        print(f"relative error vs FP32: ours {ours_error:.4f} stock {stock_error:.4f}")
        assert ours_error <= 1.1 * stock_error + 1e-3, (ours_error, stock_error)

    (ours_mla, ours_index), (stock_mla, stock_index) = step.caches, step.stock_caches
    new_mla, new_index = (s[s >= 0] for s in step.new_slots)
    at_new = new_mla // UNIT, new_mla % UNIT
    check_new_mla_rows(ours_mla[at_new], stock_mla[at_new], step.kv)
    ours_mla[at_new] = 0
    stock_mla[at_new] = 0
    assert torch.equal(ours_mla, stock_mla)
    page, entry = new_index // UNIT, new_index % UNIT
    blocks = fp4_pages(ours_index), fp4_pages(stock_index)
    arange = partial(torch.arange, device=ours_index.device)
    scale_at = UNIT * 64 + 4 * entry.view(-1, 1) + arange(4)
    ours_s, stock_s = (b[page].gather(1, scale_at) for b in blocks)
    assert torch.equal(ours_s, stock_s)
    code_at = 64 * entry.view(-1, 1) + arange(64)
    ours_codes, stock_codes = (b[page].gather(1, code_at) for b in blocks)
    diff = (dequant(ours_codes, ours_s) - dequant(stock_codes, ours_s)).abs()
    top = torch.exp2(ours_s.float() - 127).repeat_interleave(32, -1)
    assert bool((diff <= 2 * top).all())  # one E2M1 step at the top binade
    assert (ours_codes != stock_codes).sum() <= 0.05 * ours_codes.numel()
    for b in blocks:
        b[page.view(-1, 1), scale_at] = 0
        b[page.view(-1, 1), code_at] = 0
    assert torch.equal(blocks[0], blocks[1])


@pytest.fixture
def mxfp4_layers(monkeypatch):
    """test_k3_sm107_dsa.py's layer harness (run_layer_step, run_wide_step and the
    tests built on them) on an MXFP4 indexer cache:
    indexer_kv_dtype="mxfp4", 68 B indexer rows written by the stock MXFP4 writer,
    and the MXFP4 NaN check, exact top-k and step check."""
    from vllm.models.kimi_k3.nvidia.ops import dsa_indexer
    from vllm.models.kimi_k3.nvidia.ops.dsa_indexer_mxfp4 import (
        mxfp4_k_norm_quant_insert,
    )

    fp8_config = dsa.layer_config

    def layer_config(core, q, world=1):
        config = fp8_config(core, q, world)
        config.attention_config.indexer_kv_dtype = "mxfp4"
        return config

    monkeypatch.setattr(dsa, "layer_config", layer_config)
    monkeypatch.setattr(dsa, "INDEX_ROW", FP4_ROW)
    monkeypatch.setattr(dsa, "assert_index_nan_free", fp4_index_nan_free)
    monkeypatch.setattr(dsa, "exact_topk_widths", fp4_exact_topk_widths)
    monkeypatch.setattr(dsa, "check_layer_step", check_fp4_layer_step)
    # The harness writes the old keys with the stock writer it imports per step.
    monkeypatch.setattr(
        dsa_indexer, "indexer_k_norm_quant_insert", mxfp4_k_norm_quant_insert
    )


def mla_cache_ready(core):
    """Skip a case whose MLA cache this tree or device cannot run: nvfp4_ds_mla
    without the NVFP4 specialized decode, prims_ts where FlashInfer has none."""
    if dsa.LAYER_CORES[core][0] == "nvfp4_ds_mla" and hasattr(dsa, "NO_NVFP4"):
        pytest.skip(NO_NVFP4)
    if dsa.LAYER_CORES[core][2] == "prims_ts" and dsa.no_prims_ts():
        pytest.skip("FlashInfer prims_ts sparse MLA cannot run here")


@SM100
@pytest.mark.parametrize("overlap", ["serial", "fork", "front", "late"])
@pytest.mark.parametrize("batch,q", [(1, 1), (8, 1), (5, 5), (40, 5), (60, 5)])
@pytest.mark.parametrize("core", ["fi-auto", "fi-nvfp4"])
def test_prepared_dsa_mxfp4_matches_stock(
    workspace_init, dist_init, mxfp4_layers, core, batch, q, overlap
):
    """test_prepared_dsa_run_matches_stock (TP, one GPU) on an MXFP4 indexer cache,
    with the per-tensor FP8 (fi-auto) or nvfp4_ds_mla MLA cache: DsaIndex's,
    IndexFront's or (serial, front past 40 rows) IndexFrontMma's MXFP4 query and
    keys scored by sm_107's MXFP4 scorer (the fused decode indexer: its physical
    rows, the late modes too)."""
    mla_cache_ready(core)
    check_fp4_layer_step(dsa.run_layer_step(core, batch, q, overlap=overlap), q)


def run_fp8_test(test, **kwargs):
    """A layer test of test_k3_sm107_dsa.py by keyword, on BF16 partials
    and the per-tensor FP8 MLA cache (fi-trtllm) where it takes them as arguments
    (where it does not, only those)."""
    params = inspect.signature(test).parameters
    for name, value in (("partials", "bf16"), ("core", "fi-trtllm")):
        if name in params:
            kwargs.setdefault(name, value)
        else:
            assert kwargs.pop(name, value) == value, (test.__name__, name)
    test(**kwargs)


# Rubin's DL7 (q8: C1 = M8 .. C8 = M64) and DL3 (q4) rows.
DCP16_SHAPES = [(8, 8), (16, 8), (32, 8), (64, 8), (16, 4), (64, 4)]
# The FP8 MLA cache under the default FlashInfer policy and each query overlap,
# nvfp4_ds_mla (prims_ts at every row) under the default policy.
DCP16_CASES = [("fi-auto", o) for o in (None, "serial", "fork", "front")]
DCP16_CASES += [("fi-nvfp4", None)]


@DCP16
@pytest.mark.skip_global_cleanup  # the DCP16 group outlives each case
@pytest.mark.parametrize("batch,q", DCP16_SHAPES)
@pytest.mark.parametrize("core,overlap", DCP16_CASES)
def test_prepared_dsa_dcp16_mxfp4(
    dcp8_group,  # noqa: F811
    mxfp4_layers,
    core,
    overlap,
    batch,
    q,
):
    """test_prepared_dsa_run_dcp_matches_stock at DCP16 on an MXFP4 indexer cache:
    DsaIndex's or the index front's MXFP4 query and keys, the FP32 DCP top-k over
    sm_107's MXFP4 scorer's logits, against the stock DCP forward of the same step
    (MXFP4 writers and scoring)."""
    mla_cache_ready(core)
    run_fp8_test(
        dsa.test_prepared_dsa_run_dcp_matches_stock,
        dcp8_group=dcp8_group,
        core=core,
        overlap=overlap,
        batch=batch,
        q=q,
    )


@DCP16
@pytest.mark.skip_global_cleanup  # the DCP16 group outlives each case
@pytest.mark.parametrize("batch", [8, 64])
@pytest.mark.parametrize("core", ["fi-auto", "fi-nvfp4"])
def test_prepared_dsa_dcp16_mxfp4_graph(dcp8_group, mxfp4_layers, core, batch):  # noqa: F811
    """test_prepared_dsa_dcp_graph_poisoned at q8 on an MXFP4 indexer cache (FP8 or
    nvfp4_ds_mla MLA cache): two layers in a CUDA graph, 200 replays bit for bit
    the eager step."""
    mla_cache_ready(core)
    args = core, batch, 8, dcp8_group, None, "bf16"
    step = dsa.run_layer_step(*args, graph=True, replays=200)
    check_fp4_layer_step(step, 8, reference=False)


@DCP16
@pytest.mark.skip_global_cleanup  # the DCP16 group outlives each case
@pytest.mark.parametrize("batch", [8, 16])
def test_prepared_dsa_dcp16_mxfp4_core(dcp8_group, mxfp4_layers, batch):  # noqa: F811
    """test_prepared_dsa_dcp_sparse_core at DL7's rows on an MXFP4 indexer cache:
    the DSA sparse core (2x16s2) over the FP32 DCP top-k's rows of the MXFP4
    logits."""
    test = dsa.test_prepared_dsa_dcp_sparse_core
    args = dict(dcp8_group=dcp8_group, batch=batch, variant="2x16s2", overlap=None)
    run_fp8_test(test, **args)


@DCP16
@pytest.mark.skip_global_cleanup  # the DCP16 group outlives each case
@pytest.mark.parametrize("rows", ["short", "balanced"])
@pytest.mark.parametrize("batch", [8, 16])
def test_prepared_dsa_dcp16_mxfp4_core_rows(
    dcp8_group,  # noqa: F811
    mxfp4_layers,
    batch,
    rows,
):
    """test_prepared_dsa_dcp_sparse_core_rows on an MXFP4 indexer cache: the FP32
    DCP top-k of the MXFP4 logits feeding the DSA core (its skewed case shifts FP8
    key scales, so it is not run here)."""
    args = dict(dcp8_group=dcp8_group, overlap=None)
    test = dsa.test_prepared_dsa_dcp_sparse_core_rows
    run_fp8_test(test, **args, batch=batch, rows=rows)


@DCP16
@pytest.mark.skip_global_cleanup  # the DCP16 group outlives each case
@pytest.mark.parametrize("batch,q", dsa.WIDE_DCP_CASES)
@pytest.mark.parametrize("core", ["fi-trtllm", "fi-nvfp4"])
def test_wide_dsa_dcp16_mxfp4(
    dcp8_group,  # noqa: F811
    mxfp4_layers,
    core,
    batch,
    q,
):
    """test_wide_dsa_dcp on an MXFP4 indexer cache (FP8 or nvfp4_ds_mla MLA cache):
    WideDsa's DsaIndex writes the MXFP4 query and keys, sm_107's MXFP4 scorer
    scores them (the MXFP4 query views and use_fp4_cache) and the FP32 DCP top-k
    selects, at C64's and DL3's wide rows."""
    mla_cache_ready(core)
    args = dict(dcp8_group=dcp8_group, batch=batch, q=q)
    run_fp8_test(dsa.test_wide_dsa_dcp, **args, core=core)
