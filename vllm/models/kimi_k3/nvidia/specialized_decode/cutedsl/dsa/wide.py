# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 DSA layers at the Rubin wide verify rows ``DSA_WIDE`` admits (under DCP,
65 rows up to ``wide_max_rows``: 96 by default): round 1's prepared DSA core as Wei's
DSA_WIDE extended it, kept as it was while #108's ``dsa.prepared.PreparedDsa`` serves
the steps up to 64 rows.

  MlaQuery (dsa)   the query rows (per-tensor FP8 [B, H, 576] on FlashInfer, BF16
                   [B, 64 (padded heads), 576] on fp8_ds_mla), the KV insert of this
                   rank's slots, the normalized q_c rows, and the publish into every
                   rank's q mailbox of the widened DSA box
  wq_b             split-K projection of q_c -> [B, 32 * 128] (cuBLAS without a
                   tactic)
  DsaIndex         FP8 indexer query + head weights, indexer key insert
  indexer core     the stock decode scoring and top-k (sparse_attn_indexer, key
                   insert skipped), whose DCP merge gives global token ids
  sparse core      the global ids filtered to this rank's slots, then the sparse
                   decode of every rank's gathered heads over the local shard:
                   FlashInfer's (as the stock decode) or FlashMLA sparse
  DcpValue         the LSE merge of this rank's heads, W_UV and the sigmoid gate

The packed projection row is [q_a 1536 | kv_a 576 | gate H*128 | wk 128 | wv 32]."""

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_stream

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.utils.deep_gemm import mqa_logits_use_mx_scales
from vllm.utils.torch_utils import _encode_layer_name

from ...shapes import Choice, Path, rows_policy
from ..common.host import (
    check_residency,
    fake,
    layout,
    max_active_clusters,
    prepare_shared_carveout,
    stream,
    to_cute,
)
from ..gemm.projection import PreparedProjection
from ..mla.dcp import GB300_SMS, DcpValue, compile_value
from ..mla.prepared import pad_requests
from ..mla.query import MlaQuery
from ..mla.value import MlaValue
from .index import DIM, HEADS, DsaIndex

logger = init_logger(__name__)

QUERY_HEADS = 64  # FlashMLA sparse FP8 decode reads 64 or 128 query heads


# Rubin wide verify rows (> 64, ``shapes.rows_policy`` by q) whose DSA layers run this
# prepared core (``WideDsa`` on a widened DSA mailbox: MlaQuery's fused query and
# publish, DcpValue's merge + W_UV + gate) instead of the stock seam S2-DSA
# (``runtime.seams.stock_dsa``: stock DCP q gather, A2A LSE combine, W_UV, gate). 16
# ranks (TP16 / DCP16, interleave 896, fp8 KV, FlashInfer trtllm-gen, dummy weights,
# DSpark drafter included), ms/step S2-DSA -> prepared, AgentX contexts / uniform
# 192K: q4 M72 18.07 -> 17.49 / 16.79 -> 16.18, M96 19.85 -> 19.23 / 18.58 -> 18.02;
# q8 M72 18.22 -> 17.62 / 17.13 -> 16.53, M96 20.45 -> 19.84 / 19.25 -> 18.68; real
# (test) weights q4 M72 16.13 -> 15.58, M96 17.53 -> 16.96.
DSA_WIDE: dict = {"*": "65-512:1"}
# The most rows a prepared DSA step serves by default (VLLM_KIMI_K3_DSA_WIDE_MAX_ROWS,
# ``wide_max_rows``), and the most whose MlaQuery polls the whole gathered mailbox from
# every CTA: that grid (8-CTA clusters x 6 heads x 32-row tiles at DCP16) must be
# co-resident, and Rubin holds 22 such clusters (cuOccupancyMaxActiveClusters, checked
# at compile by ``check_query_residency``), i.e. 3 row tiles (96 rows: 18). Above it
# each row tile polls only its own tokens (``MlaQuery(tile_poll=True)``, Wei's
# d6347e7603): one tile's 6 clusters must be resident at once, and the tiles are
# dispatched in order, so every rank's lowest unfinished tile is resident.
WIDE_MAX_ROWS = 96
# The most rows VLLM_KIMI_K3_DSA_WIDE_MAX_ROWS admits (DL3's 128 requests x q4).
# DcpValue's merge CTAs (one per head and 16-row tile, the only CTAs from 193 rows:
# 192 at 512, 216 past 560) poll the partials every CTA pushes, so its whole grid must
# be co-resident (``check_value_residency``: one 512-thread CTA per SM, 212 SMs).
WIDE_LIMIT = 512


def wide_max_rows() -> int:
    """VLLM_KIMI_K3_DSA_WIDE_MAX_ROWS: the most wide rows WideDsa serves (64: none),
    read when the providers are built."""
    rows = envs.VLLM_KIMI_K3_DSA_WIDE_MAX_ROWS
    if not 64 <= rows <= WIDE_LIMIT:
        raise ValueError(
            f"VLLM_KIMI_K3_DSA_WIDE_MAX_ROWS={rows}: expected 64 to {WIDE_LIMIT}"
        )
    return rows


def choose(shape, ctx):
    """Prepared DSA core at a decode shape: GENERIC at the tuned rows, GENERIC "wide"
    where ``DSA_WIDE`` admits a Rubin DCP verify shape of at most ``wide_max_rows``,
    else None (seam S2-DSA)."""
    if shape.rows <= 64:
        return Choice(Path.GENERIC)
    admitted = (
        ctx.arch == "rubin"
        and ctx.dcp > 1
        and shape.q > 1
        and shape.rows <= wide_max_rows()
        and rows_policy(DSA_WIDE, shape, 0)
    )
    return Choice(Path.GENERIC, tactic="wide", note="prepared") if admitted else None


def gate_columns(heads):
    return 2112, 2112 + heads * 128


@cache
def compile_core(
    batch,
    heads,
    pitch,
    specs,
    device,
    q=1,
    dcp=0,
    q_heads=0,
    sms=148,
    fp8=False,
    lse_base_e=True,
    cap=64,
):
    """CuTe kernels (query, index, value) of one batch of rows (``q`` per request).
    ``specs``: the layouts of the weights, the MLA cache and the indexer cache.
    Under DCP (``dcp``: its size, ``q_heads``: the gathered query's padded heads) the
    query also publishes into the q mailbox and the value is DcpValue. ``fp8``: a
    per-tensor FP8 query [B, H, 576] and cache, and the value reads FlashInfer's
    [B, H, 512] latent (under DCP: an FP8 q mailbox, and an LSE in base e if
    ``lse_base_e``, else base 2). ``cap``: the DCP mailboxes' rows (64, or a widened
    box's)."""
    qnorm, kvnorm, qb, uk, uv, kv, index_kv = (
        fake(shape, stride, dtype, 4 if dtype == torch.float32 else 16)
        for shape, stride, dtype in specs
    )
    scalar = fake((1,), (1,), torch.float32, 4)
    query_heads = heads if fp8 else QUERY_HEADS
    if fp8:
        query = fake((batch, heads, 576), (heads * 576, 576, 1), torch.uint8)
    else:
        query = fake((batch, QUERY_HEADS, 576), (QUERY_HEADS * 576, 576, 1))
    qc = fake((batch, 1536), (1536, 1))
    projected = fake((batch, pitch), (pitch, 1))
    slots = fake((batch,), (1,), torch.int64, 8)
    f32 = cutlass.Float32
    s = make_fake_stream()
    mla_query = MlaQuery(
        heads,
        batch,
        q_b_cp_async=heads == 12 and batch == 1,
        dsa=True,
        fp8=fp8,
        dcp=dcp,
        q_heads=q_heads,
        tile_poll=bool(dcp) and batch > WIDE_MAX_ROWS,
    )
    q_args = (projected, qnorm, kvnorm, qb, uk, kv, slots, scalar, scalar, f32(1e-6))
    publish: tuple = ()
    if dcp:  # this parity's q mailbox (bytes), its multicast address, rank
        mailbox = fake((cap, q_heads, 576 if fp8 else 1152), None, torch.uint8)
        publish = (mailbox, cutlass.Int64(0), cutlass.Int32(0))
    _, key = gate_columns(heads)
    index = DsaIndex(key, key + DIM, mx=mqa_logits_use_mx_scales())
    iq = fake((batch, HEADS * DIM), (HEADS * DIM, 1))
    norm = fake((DIM,), (1,), torch.float32, 4)
    q_out = fake((batch, HEADS, DIM), None, torch.uint8)
    w_out = fake((batch, HEADS), None, torch.bfloat16, 4)
    i_args = (projected, iq, norm, norm, index_kv, slots, q_out, w_out, f32(1e-6))
    latent = fake((batch, heads, 512), (query_heads * 512, 512, 1))
    gate = fake((batch, heads * 128), (pitch, 1))
    lengths = fake((batch // q,), (1,), torch.int32, 4)
    output = fake((batch, heads * 128), (heads * 128, 1))
    with torch.accelerator.device_index(device):
        query_kernel = cute.compile(mla_query, *q_args, query, qc, *publish, s)
        if dcp and batch > 64:
            # Wide rows: the CTAs poll the gathered mailbox, which other ranks' CTAs
            # fill, so every CTA that polls with them must be resident (hazard H1).
            check_query_residency(query_kernel, mla_query, device)
        if batch > 1:
            query_kernel = prepare_shared_carveout(query_kernel, device, 100)
        index_kernel = cute.compile(index, *i_args, s)
        if dcp:
            value_kernel = compile_value(
                heads,
                dcp,
                batch,
                cap,
                specs[4],
                pitch,
                device,
                sms,
                q,
                dsa_heads=q_heads,
                dsa_bf16=not fp8,
                lse_base_e=lse_base_e,
            )
            if batch > 64:
                check_value_residency(value_kernel, heads, dcp, batch, q, sms, device)
        else:
            value = MlaValue(heads, batch, q)
            value_kernel = cute.compile(value, latent, uv, gate, lengths, output, s)
    return query_kernel, index_kernel, value_kernel


def check_query_residency(compiled, mla_query, device):
    """Raise unless the device holds at once every cluster of ``mla_query``'s grid
    that polls together (cuOccupancyMaxActiveClusters at its shared memory and
    carve-out): the whole grid, or under ``tile_poll`` one row tile's; returns the
    measured clusters. Rubin (VR200, 212 SMs), DCP16, FP8, 6 heads: one CTA per SM
    (227 KB shared, 208 registers), 22 clusters of 8 CTAs at once; 65-96 rows need
    18 (3 tiles of 32 rows x 6 heads), 97-128 rows would need 24 (hence the tile
    poll above WIDE_MAX_ROWS: 6)."""
    grid, cluster = mla_query.grid, mla_query.cluster
    resident = max_active_clusters(compiled, device, grid, 256, cluster, 100)
    need = grid[1] * (1 if mla_query.tile_poll else grid[2])
    if resident < need:
        raise RuntimeError(
            f"Kimi K3.1 DSA wide rows: MlaQuery's {need} clusters of {cluster} CTAs "
            f"(batch {mla_query.batch}) cannot all be resident ({resident} at once); "
            "its gathered-q poll could deadlock across ranks. Narrow DSA_WIDE."
        )
    return resident


def check_value_residency(compiled, heads, dcp, batch, q, sms, device):
    """Raise unless the device holds DcpValue's whole grid at once (its merge CTAs
    poll the partials that every CTA pushes first); returns the grid. The DSA value
    runs the GB300 schedule (``compile_value``: SMs capped at GB300_SMS): DCP16, 6
    heads, 6 x 16-row tiles x pushers, e.g. 144 CTAs at 96 rows, 96 at 256, 192 at
    512, against one 512-thread CTA per SM on Rubin."""
    grid = DcpValue(heads, dcp, batch, min(sms, GB300_SMS), q).grid
    what = f"Kimi K3.1 DSA wide rows: DcpValue (batch {batch})"
    check_residency(compiled, device, grid[0] * grid[1] * grid[2], 512, what)
    return grid


def index_policies(batches):
    """Split-K tactics of the replicated wq_b projection (N 4096, K 1536) per padded
    batch (GB300 us, previous / these: B1-8 4.9/2.9, B32 3.7/3.4, B64 4.9/3.9); wide
    rows up to 128 the in-projection's generic split-K tactic, above it none
    (``WideDsa.run_index``: cuBLAS, as the stock ``wq_index``)."""
    tactics = {8: (128, 8, 3, 4), 16: (64, 16, 2, 6), 32: (64, 32, 1, 8)}
    tactics[64] = (64, 64, 1, 4)
    tactics[128] = (64, 64, 2, 6)
    return {b: tactics[next(n for n in tactics if b <= n)] for b in batches if b <= 128}


def dsa_on_flashinfer(attention) -> bool:
    """Whether a KimiK3DSAAttention runs FLASHINFER_MLA_SPARSE (per-tensor FP8)."""
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
        FlashInferMLASparseImpl,
    )

    return isinstance(attention.mla_attn.mla_attn.impl, FlashInferMLASparseImpl)


class WideDsa:
    """One K3.1 DSA layer's wide rows under DCP (round 1's PreparedDsa). ``attention``:
    the KimiK3DSAAttention; ``pitches``: the packed projection pitch per batch;
    ``queries``: the query lengths q (1 + the draft length under DSpark verify)
    prepared for every batch divisible by q; ``dcp``: the model's DSA DcpBox, whose
    widened box (``DcpBox.widened``) serves the batches above its rows; ``core``: the
    TP sparse core of the shared provider policy (unused: DCP only)."""

    def __init__(
        self,
        attention,
        pitches,
        scratch,
        *,
        queries=(1,),
        core="auto",
        dcp=None,
        parity=0,
    ):
        from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseImpl

        wrapper = attention.mla_attn
        self.layer = layer = wrapper.mla_attn
        self.impl, self.indexer = layer.impl, attention.dsa_indexer
        self.fp8 = dsa_on_flashinfer(attention)
        if not (self.fp8 or isinstance(self.impl, FlashMLASparseImpl)):
            raise ValueError(
                "Kimi K3.1 specialized decode needs the FLASHMLA_SPARSE or "
                f"FLASHINFER_MLA_SPARSE backend, got {type(self.impl).__name__}"
            )
        assert dcp is not None, "Kimi K3.1 DSA wide rows run under DCP only"
        self.box, self.parity = dcp, parity
        self.dcp = dcp.size
        self.heads = layer.W_UK_T.shape[0]
        norms = wrapper.q_a_layernorm.weight, wrapper.kv_a_layernorm.weight
        self.weights: tuple = (*norms, wrapper.q_b_proj.weight, layer.W_UK_T)
        self.weights += (layer.W_UV,)
        self.device, self.pitches = layer.W_UK_T.device, pitches
        self.eps = cutlass.Float32(wrapper.q_a_layernorm.variance_epsilon)
        knorm = self.indexer.k_index_norm
        self.key_eps = cutlass.Float32(knorm.eps)
        self.key_norm = to_cute(knorm.weight, 4), to_cute(knorm.bias, 4)
        self.views = tuple(
            to_cute(w, 4 if w.dtype == torch.float32 else 16) for w in self.weights
        )
        self.unit = torch.ones(1, dtype=torch.float32, device=self.device)
        self.query, self.qc, self.iq, self.q_fp8, self.head_w, _ = scratch
        # Wide rows run verify steps only (SpecializedDecode.plan): q > 1.
        self.shapes = [(b, q) for b in pitches for q in queries if q > 1 and b % q == 0]
        # The FP8 query's inverse per-tensor scale and the KV row's scale: MlaQuery
        # divides the row by k_scale under DSA, as concat_and_cache_mla (FlashMLA:
        # unused).
        self.scales = self.unit, self.unit
        if self.fp8:
            q_scale_inv = (1 / layer._q_scale.float()).reshape(1).to(self.device)
            k_scale = layer._k_scale.float().reshape(1).to(self.device)
            self.scales = q_scale_inv, k_scale
            shape = (max(pitches), self.heads, 576)
            self.query = torch.zeros(shape, dtype=torch.uint8, device=self.device)
        # Batches above the box's rows publish into its widened box (one per model,
        # shared by every DSA layer; created collectively here, in layer order).
        rows, device = max(pitches), self.device
        padded = dcp.q_heads
        wide = rows > dcp.cap
        widened = dcp.widened(rows) if wide else dcp
        self.boxes = {b: widened if b > dcp.cap else dcp for b in pitches}
        # FlashMLA sparse's output rows (FlashInfer returns its own).
        self.rows = None
        if not self.fp8:
            self.rows = torch.zeros(
                (rows, 1, padded, 512), dtype=torch.bfloat16, device=device
            )
        self.dummy_table = torch.zeros((rows, 1), dtype=torch.int32, device=device)
        self.cache_lens = torch.full(
            (rows,), self.indexer.max_model_len, dtype=torch.int32, device=device
        )
        self.kernels = {}
        policies = index_policies(sorted(pitches))
        wq = self.indexer.wq_index.weight
        self.index_projection = PreparedProjection(wq, policies).prepare()
        self.k_cache_name = _encode_layer_name(self.indexer.k_cache.prefix)
        self.max_model_len = self.indexer.max_model_len
        self.max_total = self.indexer.max_total_seq_len
        self.topk = self.indexer.topk_tokens

    def prepare_cache(self, kv):
        caches = kv.view(torch.uint8), self.indexer.k_cache.kv_cache
        specs = tuple(layout(w) for w in (*self.weights, *caches))
        sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        for batch, q in self.shapes:
            args: tuple = (batch, self.heads, self.pitches[batch], specs)
            args += (self.device.index, q)
            box = self.boxes[batch]
            args += (self.dcp, box.q_heads, sms)
            cap = box.cap
            lse_base_e = not self.fp8 or self.impl.lse_base_on_e
            self.kernels[batch, q] = compile_core(
                *args, fp8=self.fp8, lse_base_e=lse_base_e, cap=cap
            )
        rows = sorted(b for b, _ in self.shapes)
        logger.info_once(
            "Kimi K3.1 WideDsa (DCP %d): rows %s, widened DSA box %d rows, MlaQuery "
            "tile poll above %d rows (VLLM_KIMI_K3_DSA_WIDE_MAX_ROWS=%d)",
            self.dcp,
            " ".join(map(str, rows)),
            max(box.cap for box in self.boxes.values()),
            WIDE_MAX_ROWS,
            wide_max_rows(),
            scope="global",
        )
        return self

    def indexer_core(self, projected, batch):
        """Top-k of the current rows into the shared buffer (stock decode scoring)."""
        indexer = self.indexer
        return torch.ops.vllm.sparse_attn_indexer(
            projected,
            self.k_cache_name,
            indexer.k_cache.kv_cache,
            self.q_fp8[:batch].view(torch.float8_e4m3fn),
            None,
            None,
            self.head_w[:batch],
            DIM,
            indexer.scale_fmt,
            self.topk,
            DIM,
            self.max_model_len,
            self.max_total,
            indexer.topk_indices_buffer,
            True,  # the key is inserted by DsaIndex
            False,
            False,
            _encode_layer_name(""),
            False,
            indexer.dcp_rank,
            indexer.dcp_world_size,
            indexer.cp_kv_cache_interleave_size,
            topk_backend=indexer.topk_backend,
            # Interleave 1: the exact one-round DCP merge (gated in the op).
            dcp_local_topk=indexer.dcp_local_topk,
        )

    def dcp_core(self, cache, topk, metadata, batch, box=None):
        """Sparse decode of every rank's query heads (the gathered q mailbox) over the
        local shard into the DcpValue rows and LSE. The global top-k ids are filtered
        to this rank's slots and compacted to a prefix whose length bounds the work
        (about 1 / DCP of the top-k). FlashMLA (natural-log LSE; a fresh scheduler
        metadata per call follows those lengths) or, on a per-tensor FP8 cache,
        FlashInfer's sparse decode over the flat rows, as the stock decode."""
        from vllm.v1.attention.backends.mla.flashmla_sparse import _fp8_kv_pages
        from vllm.v1.attention.backends.mla.sparse_utils import (
            flat_kv_row_view,
            triton_filter_and_convert_dcp_index,
        )
        from vllm.v1.attention.ops.flashmla import (
            flash_mla_with_kvcache,
            get_mla_metadata,
        )

        indexer, md = self.indexer, metadata
        box = self.box if box is None else box  # (``boxes[batch]``: widened rows)
        stride_rows: int | None
        if self.fp8:
            _, stride_rows = flat_kv_row_view(cache, md.block_size)
        else:
            pages, stride_rows = _fp8_kv_pages(cache, md.block_size)
        # The DcpExchange top-k (dsa_kernels.dcp_exchange.decode) already left this
        # rank's slots and their counts in the buffer.
        counts = self.impl.dcp_local_topk_counts
        valid = counts(batch) if counts is not None else None
        if valid is not None:
            slots = topk[:batch]
        else:
            slots, valid = triton_filter_and_convert_dcp_index(
                md.req_id_per_token[:batch],
                md.block_table,
                topk[:batch],
                dcp_size=self.dcp,
                dcp_rank=indexer.dcp_rank,
                cp_kv_cache_interleave_size=indexer.cp_kv_cache_interleave_size,
                BLOCK_SIZE=md.block_size,
                BLOCK_STRIDE_ROWS=stride_rows,
                NUM_TOPK_TOKENS=topk.shape[1],
                return_valid_counts=True,
            )
        if self.fp8:
            self.impl.record_logical_topk_ready()
            rows, lse = self.impl._forward_mqa_kernel(
                box.q[self.parity, :batch].view(torch.float8_e4m3fn),
                cache,
                slots,
                valid,
                layer=self.layer,
                block_size=md.block_size,
                is_decode=True,
            )
            assert lse is not None
            return rows, lse.masked_fill((valid == 0).view(-1, 1), float("-inf"))
        rows, lse = flash_mla_with_kvcache(
            q=box.q[self.parity, :batch].unsqueeze(1),
            k_cache=pages.view(torch.uint8).unsqueeze(-2),
            block_table=self.dummy_table[:batch],
            head_dim_v=512,
            cache_seqlens=self.cache_lens[:batch],
            tile_scheduler_metadata=get_mla_metadata()[0],
            is_fp8_kvcache=True,
            indices=slots.unsqueeze(1),
            topk_length=valid,
            softmax_scale=self.impl.softmax_scale,
            out=self.rows[:batch],
        )
        # A rank owning none of a row's entries: DcpValue drops it (LSE -inf).
        return rows, lse.masked_fill((valid == 0).view(-1, 1, 1), float("-inf"))

    def run_dcp(self, projected, out, cache, slots, index_slots, metadata, q):
        """``run`` under DCP: the query publishes into the q mailbox and DcpValue
        merges every rank's partial rows of this rank's heads (global lengths)."""
        batch, s = projected.shape[0], stream(self.device)
        reqs, box, p = batch // q, self.boxes[batch], self.parity
        _, lengths, _ = pad_requests(reqs, metadata.block_table, metadata.seq_lens)
        mq, index, value = self.kernels[batch, q]
        qnorm, kvnorm, qb, uk, uv = self.views
        x, q_slots = to_cute(projected), to_cute(slots[:batch], 8)
        kv = to_cute(cache.view(torch.uint8))
        q_scale_inv, k_scale = (to_cute(t, 4) for t in self.scales)
        mailbox = to_cute(box.q[p].view(torch.uint8))
        scales = q_scale_inv, k_scale, self.eps
        q_args = (x, qnorm, kvnorm, qb, uk, kv, q_slots, *scales)
        publish = (mailbox, box.mc(p), cutlass.Int32(box.rank))
        mq(*q_args, to_cute(self.query[:batch]), to_cute(self.qc[:batch]), *publish, s)
        self.run_index(x, index_slots, batch, index)
        topk = self.indexer_core(projected, batch)
        latent, lse = self.dcp_core(cache, topk, metadata, batch, box)
        lens = to_cute(lengths[:reqs], 4)
        boxes = to_cute(box.partial[p]), to_cute(box.lse[p], 4)
        peers = to_cute(box.peers, 8), to_cute(box.lse_peers, 8)
        gate = to_cute(projected[:, 2112:])
        # DcpValue's [B, 1, DCP * H] views of FlashMLA's padded-head rows and LSE.
        heads, padded = self.heads * self.dcp, box.q_heads
        latent = latent.as_strided((batch, 1, heads, 512), (padded * 512, 0, 512, 1))
        lse = lse.as_strided((batch, 1, heads), (padded, 0, 1))
        v_args = (to_cute(latent), to_cute(lse, 4), lens, *boxes, *peers, mailbox, uv)
        value(*v_args, gate, lens, to_cute(out), cutlass.Int32(p), publish[2], s)
        return out

    def run_index(self, x, index_slots, batch, index):
        """FP8 indexer query and head weights; the key into the indexer cache."""
        if batch in self.index_projection.kernels:
            iq = self.index_projection.run(self.qc[:batch], self.iq[:batch])
        else:  # no split-K tactic at these rows: cuBLAS, as the stock wq_index
            wq = self.indexer.wq_index.weight
            iq = torch.mm(self.qc[:batch], wq.t(), out=self.iq[:batch])
        rows = to_cute(self.q_fp8[:batch]), to_cute(self.head_w[:batch], 4)
        key = to_cute(self.indexer.k_cache.kv_cache), to_cute(index_slots[:batch], 8)
        s = stream(self.device)
        index(x, to_cute(iq), *self.key_norm, *key, *rows, self.key_eps, s)

    def run(self, projected, out, *, cache, slots, index_slots, metadata, q=1):
        """``projected``: the packed projection rows; ``out`` [B, H * 128] gets the
        gated attention output; ``slots`` / ``index_slots``: the MLA and indexer
        cache slots. Rows beyond the requests run as empty requests."""
        args = cache, slots, index_slots, metadata, q
        return self.run_dcp(projected, out, *args)
