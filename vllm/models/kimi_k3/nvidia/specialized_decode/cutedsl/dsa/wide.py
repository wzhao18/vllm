# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 DSA layers at the Rubin wide verify rows ``DSA_WIDE`` admits (under DCP,
every row from 65 on): round 1's prepared DSA core as Wei's DSA_WIDE extended it, kept
as it was while #108's ``dsa.prepared.PreparedDsa`` serves the steps up to 64 rows.

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
  DcpValue         the LSE merge of this rank's heads, W_UV and the sigmoid gate;
                   on "proj" rows (r1.AUTO_PARTIALS) the W_UV bmm of every
                   gathered head, then #108's DcpValue over its value rows

Query overlap (``WIDE_QUERY_OVERLAP`` per rows): "fork" runs QueryNorm (the q_c rows,
bit-identical to MlaQuery's) on the layer's stream and MlaQuery (publish only, no
q_c) on ``overlap.side_stream``, beside the indexer chain (wq_b -> DsaIndex ->
scoring, top-k, merge -> slot filter), which reads only q_c; the chain joins the
side stream, then QueryWait polls the gathered mailbox and the sparse decode reads
it. The chain runs on ``overlap.index_stream`` (``WIDE_INDEX_PRIORITY``). "serial":
every kernel in one stream, MlaQuery first.

The packed projection row is [q_a 1536 | kv_a 576 | gate H*128 | wk 128 | wv 32]."""

from functools import cache, partial

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_stream

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
from ..common.policy import batch_table
from ..gemm.projection import PreparedProjection
from ..mla.dcp import GB300_SMS, DcpValue, compile_value
from ..mla.prepared import pad_requests
from ..mla.query import MlaQuery
from ..mla.value import MlaValue
from . import dcp_value
from .dcp_stage import compile_dcp_value, projected_rows, run_value
from .dcp_value import PROJECTED
from .index import DIM, HEADS, DsaIndex
from .overlap import QueryWait, index_stream, side_stream
from .query_stage import compile_norm, is_rubin
from .query_stage import index_policies as query_index_policies
from .r1 import DsaBox

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
# (test) weights q4 M72 16.13 -> 15.58, M96 17.53 -> 16.96. The serving configs ran
# it up to 256 (C48 / C64) and 512 rows (C160 / C256, "proj" partials). Open-ended:
# every kernel of the core serves any rows (MlaQuery publish-only above
# ``WIDE_MAX_ROWS``; "proj" DcpValue's merging CTAs and the BF16 one's row tiles
# stride over at most one wave), each checked co-resident at compile; the widened box
# holds the largest captured row. NQ-TUNE: 16-rank step vs S2-DSA above 512 rows (q4
# M768 / M1024 / M2048, q8 M1024 / M2048, fp8 KV, AgentX contexts).
DSA_WIDE: dict = {"*": "65-:1"}
# The most rows whose MlaQuery polls the whole gathered mailbox from every CTA: that
# grid (8-CTA clusters x 6 heads x 32-row tiles at DCP16) must be co-resident, and Rubin
# holds 22 such clusters (cuOccupancyMaxActiveClusters, checked at compile by
# ``check_query_residency``), i.e. 3 row tiles (96 rows: 18). Above it MlaQuery only
# publishes (``gather=False``) and QueryWait (dsa.overlap) polls the gathered mailbox
# on the main stream right before the sparse decode, its first reader
# (``query_publishes_only``): the query's grid then waits on nothing (no residency
# need), and the q all-gather's NVLink delivery and the ranks' arrival skew run
# behind the indexer chain (wq_b -> DsaIndex -> logits -> top-k -> merge), which reads
# only q_c. Under the previous tile poll (``MlaQuery(tile_poll=True)``) every 32-row
# tile was a 16-rank barrier inside the query kernel: 22 resident clusters of 8 CTAs,
# 4 / 5 rounds at 360 / 560 rows, each paying the slowest rank plus that round's
# NVLink delivery (C256 trace, AgentX q4: 62.9 / 89.1 us per MLA layer on the
# critical path).
WIDE_MAX_ROWS = 96
# QueryWait at the publish-only rows: CTAs per SM (one pass of at most 8 polled
# 16 B chunks per thread up to 4 x SMs x 2048 chunks: 1.24M at 360 rows, DCP16).
WAIT_CTAS_PER_SM = 4
# Query overlap per rows (``OVERLAP_MODES``, the same idea as overlap.QUERY_OVERLAP
# for PreparedDsa's rows). Under "serial" the indexer chain (wq_b -> DsaIndex ->
# logits -> top-k -> merge, 15.4-17.7 us per layer at q8 M96-M176 / q4 M128-M220)
# waits for MlaQuery (15.1-21.5 us, the gathered-q poll included at 65-96 rows),
# though it reads only q_c: frontier-advance critical path of the ShardQ-off
# DCP16 step bench (hti/lowlat results dec-c48-bal-1064153-1113224545-1331131,
# dec-c160-bal-1064153-1113225937-1632543), while PreparedDsa's fork hides the chain
# at 17-64 rows. "fork" moves MlaQuery (publish only) off the chain onto the side
# stream at every wide row; QueryWait stays right before the sparse decode, after
# the join (its wide grid would take the chain's SMs while it polls).
WIDE_QUERY_OVERLAP = "65-:fork"
OVERLAP_MODES = ("serial", "fork")
# Rows whose forked indexer chain (wq_b through the sparse decode) runs on the
# high-priority index stream ("chain"; other rows: the layer's stream). MlaQuery
# holds one CTA per SM (227 KB shared) in 0.7-1.6 waves at 96-220 rows, so without
# priority its queued CTAs take the SMs ahead of the chain as they free up (the
# C512 precedent in overlap.INDEX_PRIORITY); it has slack until the join (the chain
# and the top-k merge, ~30-35 us, against its 13-18 us).
WIDE_INDEX_PRIORITY = "65-:chain"


def wide_overlap(batch, policy=None):
    """The query overlap mode of ``batch`` wide rows: ``policy`` is a mode or a
    "rows:mode,..." table (default WIDE_QUERY_OVERLAP)."""
    policy = WIDE_QUERY_OVERLAP if policy is None else policy
    mode = batch_table(policy).get(batch) if ":" in policy else policy
    if mode not in OVERLAP_MODES:
        raise ValueError(
            f"Kimi K3.1 WideDsa query overlap {mode!r} at {batch} rows: expected "
            f"one of {OVERLAP_MODES}"
        )
    return mode


def chain_priority(batch, policy=None):
    """Whether the forked indexer chain of ``batch`` rows runs on the index stream:
    ``policy`` a "rows:chain" table (default WIDE_INDEX_PRIORITY; "": never)."""
    policy = WIDE_INDEX_PRIORITY if policy is None else policy
    table = batch_table(policy) if policy else {}
    if set(table.values()) - {"chain"}:
        raise ValueError(f"Kimi K3.1 WideDsa index priority {policy!r}: not chain")
    return table.get(batch) == "chain"


def query_publishes_only(batch, dcp, overlap="serial"):
    """Whether WideDsa's MlaQuery at ``batch`` rows under DCP only publishes (the
    gathered mailbox then polled by QueryWait before the sparse decode): every
    forked query, and the serial ones above WIDE_MAX_ROWS."""
    return bool(dcp) and (overlap == "fork" or batch > WIDE_MAX_ROWS)


def choose(shape, ctx):
    """Prepared DSA core at a decode shape: GENERIC at the tuned rows, GENERIC "wide"
    where ``DSA_WIDE`` admits a Rubin DCP verify shape, else None (seam S2-DSA)."""
    if shape.rows <= 64:
        return Choice(Path.GENERIC)
    admitted = (
        ctx.arch == "rubin"
        and ctx.dcp > 1
        and shape.q > 1
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
    partials="bf16",
    fp8=False,
    lse_base_e=True,
    cap=64,
    nvfp4=False,
    overlap="serial",
):
    """CuTe kernels (query, index, value) of one batch of rows (``q`` per request).
    ``specs``: the layouts of the weights, the MLA cache and the indexer cache.
    Under DCP (``dcp``: its size, ``q_heads``: the gathered query's padded heads) the
    query also publishes into the q mailbox and the value is DcpValue (``partials``
    "proj": #108's, over the W_UV bmm's value rows, dcp_stage). ``fp8``: a
    per-tensor FP8 query [B, H, 576] and cache, and the value reads FlashInfer's
    [B, H, 512] latent (under DCP: an FP8 q mailbox, and an LSE in base e if
    ``lse_base_e``, else base 2). ``cap``: the DCP mailboxes' rows (64, or a widened
    box's). ``nvfp4``: that FP8 query over an nvfp4_ds_mla cache (FlashInfer
    prims_ts). ``overlap`` (OVERLAP_MODES): "fork" leaves the q_c rows to QueryNorm
    and, under DCP, the mailbox poll to QueryWait."""
    if overlap not in OVERLAP_MODES:
        raise ValueError(f"Kimi K3.1 WideDsa query overlap {overlap!r}")
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
        nvfp4=nvfp4,
        dcp=dcp,
        q_heads=q_heads,
        store_qc=overlap == "serial",
        gather=not query_publishes_only(batch, dcp, overlap),
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
        if dcp and batch > 64 and mla_query.gather:
            # Wide rows: the CTAs poll the gathered mailbox, which other ranks' CTAs
            # fill, so every CTA that polls with them must be resident (hazard H1).
            check_query_residency(query_kernel, mla_query, device)
        if batch > 1:
            query_kernel = prepare_shared_carveout(query_kernel, device, 100)
        index_kernel = cute.compile(index, *i_args, s)
        if dcp and partials == PROJECTED:
            args = heads, dcp, batch, specs[4], pitch, device, sms, q, q_heads
            value_kernel = compile_dcp_value(*args, fp8, lse_base_e, partials, cap)
            if batch > 64:
                check_proj_residency(value_kernel, heads, dcp, batch, sms, device)
        elif dcp:
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


@cache
def compile_query_wait(
    batch, heads, dcp, q_heads, sms, cap, fp8=True, overlap="serial"
):  # fmt: skip
    """QueryWait of ``batch`` publish-only rows (``query_publishes_only`` under the
    ``overlap`` mode; else None): every 16 B chunk of the DCP x ``heads`` gathered
    heads (rows of ``q_heads``) polled until published, over up to
    WAIT_CTAS_PER_SM x ``sms`` CTAs. Shared by every DSA layer (the mailbox is a
    launch argument)."""
    if not query_publishes_only(batch, dcp, overlap):
        return None
    q_bytes = 1 if fp8 else 2
    mailbox = fake((cap, q_heads, 576 * q_bytes), None, torch.uint8)
    ctas = WAIT_CTAS_PER_SM * sms
    wait = QueryWait(batch, dcp * heads, q_heads, q_bytes, ctas=ctas)
    return cute.compile(wait, mailbox, make_fake_stream())


def check_query_residency(compiled, mla_query, device):
    """Raise unless the device holds at once every cluster of ``mla_query``'s grid
    that polls together (cuOccupancyMaxActiveClusters at its shared memory and
    carve-out): the whole grid, or under ``tile_poll`` one row tile's; returns the
    measured clusters. Rubin (VR200, 212 SMs), DCP16, FP8, 6 heads: one CTA per SM
    (227 KB shared, 208 registers), 22 clusters of 8 CTAs at once; 65-96 rows need
    18 (3 tiles of 32 rows x 6 heads), 97-128 rows would need 24 (hence the
    publish-only query above WIDE_MAX_ROWS, which polls nothing)."""
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
    512; above one wave of the device's real SMs (Rubin: 6 x 36 tiles = 216 > 212,
    from 561 rows) each merging CTA strides over the row tiles
    (``mla.dcp.DcpValue.tile_stride``: 6 x 35 = 210 CTAs at any rows), against one
    512-thread CTA per SM on Rubin."""
    grid = DcpValue(heads, dcp, batch, min(sms, GB300_SMS), q, resident_sms=sms).grid
    what = f"Kimi K3.1 DSA wide rows: DcpValue (batch {batch})"
    check_residency(compiled, device, grid[0] * grid[1] * grid[2], 512, what)
    return grid


def check_proj_residency(compiled, heads, dcp, batch, sms, device):
    """Raise unless the device holds the "proj" DcpValue's whole grid at once (its
    merging CTAs poll the value rows every rank's projecting CTAs push); returns the
    grid. dsa.dcp_value's schedule on the real SM count: above PROJ_ROWS rows
    H x (SMs // H) CTAs that project on tcgen05 and then merge, else DCP x H
    projecting CTAs and H x min(tiles, (SMs - DCP x H) // H) merging ones; one
    512-thread CTA per SM."""
    grid = dcp_value.DcpValue(
        heads, dcp, batch, sms, per_row=True, partials=PROJECTED
    ).grid
    what = f"Kimi K3.1 DSA wide rows: DcpValue proj (batch {batch})"
    check_residency(compiled, device, grid[0] * grid[1] * grid[2], 512, what)
    return grid


def index_policies(batches, rubin=False):
    """Split-K tactics of the replicated wq_b projection (N 4096, K 1536) per padded
    batch (GB300 us, previous / these: B1-8 4.9/2.9, B32 3.7/3.4, B64 4.9/3.9); wide
    rows up to 128 the in-projection's generic split-K tactic, above it none
    (``WideDsa.run_index``: cuBLAS, as the stock ``wq_index``). Rubin: the query
    stage's table (``RUBIN_INDEX_TACTICS``, none above 128 rows either)."""
    if rubin:
        return query_index_policies(batches, rubin=True)
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
    TP sparse core of the shared provider policy (unused: DCP only); ``partials``:
    the DcpValue partials of its batches (r1.DsaBox; None: r1.AUTO_PARTIALS on
    FlashInfer sparse, else "bf16"); ``overlap``: a query overlap mode or
    "rows:mode" table (default WIDE_QUERY_OVERLAP); ``priority``: the forked
    batches whose indexer chain runs on the index stream, a "rows:chain" table
    (default WIDE_INDEX_PRIORITY)."""

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
        partials=None,
        overlap=None,
        priority=None,
    ):
        from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseImpl

        wrapper = attention.mla_attn
        self.layer = layer = wrapper.mla_attn
        self.impl, self.indexer = layer.impl, attention.dsa_indexer
        self.fp8 = dsa_on_flashinfer(attention)
        self.nvfp4 = layer.kv_cache_dtype == "nvfp4_ds_mla"
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
            # nvfp4_ds_mla: the KV row is unscaled (prims_ts: kv_descale 1)
            self.scales = q_scale_inv, self.unit if self.nvfp4 else k_scale
            shape = (max(pitches), self.heads, 576)
            self.query = torch.zeros(shape, dtype=torch.uint8, device=self.device)
        # Batches above the box's rows publish into its widened box (one per model,
        # shared by every DSA layer; created collectively here, in layer order).
        rows, device = max(pitches), self.device
        padded = dcp.q_heads
        wide = rows > dcp.cap
        # (the widened box's BF16 partial mailboxes only if a batch exchanges BF16
        # latent partials: DsaBox adds them)
        widened = dcp.widened(rows, partials=False) if wide else dcp
        self.boxes = {b: widened if b > dcp.cap else dcp for b in pitches}
        # "proj" batches exchange #108's value rows through the widened box's
        # DcpMailbox (collective, in layer order), from the layer's gathered W_UV
        # (PreparedDsa's, if it took it) and the bmm's rows.
        self.dsa_box = DsaBox(
            widened, pitches, partials, fp8=self.fp8, verify_q=max(queries)
        )
        if self.dsa_box.projected and not self.fp8:
            raise ValueError(
                f'Kimi K3.1 WideDsa: "proj" partials ({self.dsa_box.partials}) need '
                "the FLASHINFER_MLA_SPARSE backend"
            )
        self.uv_gathered = self.dsa_box.value_weights(layer)
        if self.uv_gathered is not None:
            projected_rows(device, widened.cap, self.dcp * self.heads)
        if self.fp8 and self.impl.use_prims_ts:
            # dcp_core runs the stock prims_ts decode on the padded q mailbox; the
            # profile run's reserve_workspace plans the native heads only.
            self.impl.prims_ts_extra_heads.update(
                b.q_heads for b in self.boxes.values()
            )
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
        # The query overlap per batch, and the forked batches whose indexer chain
        # runs on the high-priority index stream.
        self.overlaps = {b: wide_overlap(b, overlap) for b in pitches}
        self.prioritized = frozenset(
            b
            for b, mode in self.overlaps.items()
            if mode == "fork" and chain_priority(b, priority)
        )
        self.kernels = {}
        self.waits = {}  # QueryWait per batch (publish-only rows; else None)
        self.norms = {}  # QueryNorm per forked batch
        policies = index_policies(sorted(pitches), is_rubin(self.device))
        wq = self.indexer.wq_index.weight
        self.index_projection = PreparedProjection(wq, policies).prepare()
        self.k_cache_name = _encode_layer_name(self.indexer.k_cache.prefix)
        self.max_model_len = self.indexer.max_model_len
        self.max_total = self.indexer.max_total_seq_len
        self.topk = self.indexer.topk_tokens

    def prepare_cache(self, kv):
        from .indexer_stage import check_index_cache

        check_index_cache(self.indexer)
        caches = kv.view(torch.uint8), self.indexer.k_cache.kv_cache
        specs = tuple(layout(w) for w in (*self.weights, *caches))
        sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        for batch, q in self.shapes:
            args: tuple = (batch, self.heads, self.pitches[batch], specs)
            args += (self.device.index, q)
            box = self.boxes[batch]
            args += (self.dcp, box.q_heads, sms, self.dsa_box.mailbox(batch).format)
            cap = box.cap
            lse_base_e = not self.fp8 or self.impl.lse_base_on_e
            mode = self.overlaps[batch]
            self.kernels[batch, q] = compile_core(
                *args,
                fp8=self.fp8,
                lse_base_e=lse_base_e,
                cap=cap,
                nvfp4=self.nvfp4,
                overlap=mode,
            )
            w_args = batch, self.heads, self.dcp, box.q_heads, sms, cap, self.fp8
            self.waits[batch] = compile_query_wait(*w_args, overlap=mode)
            if mode == "fork":
                if self.waits[batch] is None:  # (WideDsa runs under DCP only)
                    raise RuntimeError(
                        f"Kimi K3.1 WideDsa: no QueryWait for the forked query at "
                        f"{batch} rows"
                    )
                with torch.accelerator.device_index(self.device.index):
                    self.norms[batch] = compile_norm(
                        batch, self.heads, self.pitches[batch]
                    )
        rows = sorted(b for b, _ in self.shapes)
        proj = [b for b in rows if self.dsa_box.mailbox(b).format == PROJECTED]
        forked = sorted(b for b in set(rows) if self.overlaps[b] == "fork")
        chain = sorted(b for b in forked if b in self.prioritized)
        logger.info_once(
            "Kimi K3.1 WideDsa (DCP %d): rows %s, widened DSA box %d rows (%s), "
            "query forked off the indexer chain (QueryNorm; MlaQuery publish-only on "
            "the side stream, QueryWait before the sparse decode) at rows %s, chain "
            "on the index stream at rows %s; serial: MlaQuery publish-only above %d "
            'rows; DcpValue partials %s ("proj" at rows %s)',
            self.dcp,
            " ".join(map(str, rows)),
            max(box.cap for box in self.boxes.values()),
            self.memory(),
            " ".join(map(str, forked)) or "none",
            " ".join(map(str, chain)) or "none",
            WIDE_MAX_ROWS,
            self.dsa_box.partials,
            " ".join(map(str, proj)) or "none",
            scope="global",
        )
        return self

    def memory(self):
        """The widened box's memory per GPU, shared by every DSA layer: its q (and,
        for "bf16" batches, BF16 partial / LSE) mailboxes, the "proj" mailbox and the
        W_UV bmm's value rows; MiB."""
        box = max(self.boxes.values(), key=lambda b: b.cap)
        symmetric, local = box.nbytes()
        parts = [f"q + BF16 partials {symmetric / 2**20:.0f} MiB symmetric"]
        if box.partial is None:
            parts[0] = f"q mailbox {symmetric / 2**20:.0f} MiB symmetric"
        if local:
            parts.append(f"rows {local / 2**20:.0f} MiB")
        if self.dsa_box.projected:
            mail = self.dsa_box.mailboxes[PROJECTED]
            proj = (mail.partial.nbytes + mail.lse.nbytes) / 2**20
            rows = projected_rows(self.device, box.cap, self.dcp * self.heads)
            parts.append(f'"proj" mailbox {proj:.0f} MiB symmetric')
            parts.append(f"value rows {rows.nbytes / 2**20:.0f} MiB")
        return ", ".join(parts)

    def stock_pieces(self, batch):
        """NQ-NOFALLBACK: the stock kernels inside this core at ``batch`` rows, for the
        coverage log. As a whole the core was measured against seam S2-DSA (DSA_WIDE);
        the pieces were not measured against PreparedDsa's #63 merge / DcpExchange
        rows path (NQ-TUNE)."""
        pieces = [
            "sparse_attn_indexer op (its DCP merge, or DcpExchange's top-k)",
            "Triton slot filter where DcpExchange does not serve the step",
            "FlashInfer _forward_mqa_kernel" if self.fp8 else "FlashMLA sparse",
        ]
        if batch not in self.index_projection.kernels:
            pieces.append("wq_index cuBLAS (no split-K tactic)")
        return "; ".join(pieces) + " (vs the merge-rows path unmeasured: NQ-TUNE)"

    def indexer_core(self, projected, batch):
        """Top-k of the current rows into the shared buffer (stock decode scoring of
        DsaIndex's FP8 query, or on an MXFP4 indexer cache its MXFP4 query and
        scales)."""
        from .index import fp4_query_views

        indexer, fp4 = self.indexer, self.indexer.use_fp4_cache
        query = self.q_fp8[:batch].view(torch.float8_e4m3fn), None
        if fp4:
            query = fp4_query_views(self.q_fp8, batch, batch)
        return torch.ops.vllm.sparse_attn_indexer(
            projected,
            self.k_cache_name,
            indexer.k_cache.kv_cache,
            *query,
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
            fp4,
            indexer.dcp_rank,
            indexer.dcp_world_size,
            indexer.cp_kv_cache_interleave_size,
            topk_backend=indexer.topk_backend,
            # Interleave 1: the exact one-round DCP merge (gated in the op).
            dcp_local_topk=indexer.dcp_local_topk,
        )

    def dcp_core(self, cache, topk, metadata, batch, box=None, wait=None):
        """Sparse decode of every rank's query heads (the gathered q mailbox) over the
        local shard into the DcpValue rows, LSE and per-row valid counts. The global
        top-k ids are filtered to this rank's slots and compacted to a prefix whose
        length bounds the work (about 1 / DCP of the top-k). FlashMLA (natural-log
        LSE; a fresh scheduler metadata per call follows those lengths) or, on a
        per-tensor FP8 cache, FlashInfer's sparse decode over the flat rows, as the
        stock decode. ``wait``: QueryWait's launch (publish-only rows), right before
        the decode reads the gathered q mailbox."""
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
        if wait is not None:
            wait()
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
            # A row without local entries must reach DcpValue with LSE -inf. On
            # trtllm-gen the stock decode's _finish_dcp_partials already wrote it (and
            # a zero latent), so the out-of-place masked_fill would only add two
            # kernels per DSA layer (CompareEq + masked_fill, 3.8 us per layer at q4
            # M512 on VR200, nsys); prims_ts keeps it.
            if self.impl.use_prims_ts:
                lse = lse.masked_fill((valid == 0).view(-1, 1), float("-inf"))
            return rows, lse, valid
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
        lse = lse.masked_fill((valid == 0).view(-1, 1, 1), float("-inf"))
        return rows, lse, valid

    def run_dcp(self, projected, out, cache, slots, index_slots, metadata, q):
        """``run`` under DCP: the query publishes into the q mailbox and DcpValue
        merges every rank's partial rows of this rank's heads (global lengths)."""
        batch = projected.shape[0]
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
        q_args += (to_cute(self.query[:batch]), to_cute(self.qc[:batch]))
        publish = (mailbox, box.mc(p), cutlass.Int32(box.rank))
        query_wait, wait = self.waits.get(batch), None
        if self.overlaps[batch] == "fork":
            args = q_args, publish, query_wait
            core_args = cache, index_slots, metadata, q
            latent, lse, valid = self.forked_core(projected, *core_args, *args)
        else:
            mq(*q_args, *publish, stream(self.device))
            self.run_index(x, index_slots, batch, index)
            topk = self.indexer_core(projected, batch)
            if query_wait is not None:  # publish-only rows: poll before the decode
                wait = partial(query_wait, mailbox, stream(self.device))
            latent, lse, valid = self.dcp_core(cache, topk, metadata, batch, box, wait)
        s = stream(self.device)
        if self.dsa_box.mailbox(batch).format == PROJECTED:
            core, (w_uv, uv_view) = (latent, lse, valid), self.uv_gathered
            args = value, self.dsa_box, p, publish, self.heads, core, lengths[:reqs]
            return run_value(*args, uv_view, projected, out, w_uv)
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

    def forked_core(
        self, projected, cache, index_slots, metadata, q, q_args, publish, query_wait
    ):  # fmt: skip
        """The "fork" query overlap up to the sparse decode's rows, LSE and counts:
        QueryNorm writes the q_c rows on the layer's stream; MlaQuery (publish only,
        no q_c) runs on the side stream; the indexer chain (wq_b, DsaIndex, the
        indexer op, the slot filter) and the sparse decode run on the index stream
        (prioritized batches) or the layer's stream, which joins the side stream
        and runs QueryWait right before the decode reads the gathered mailbox. The
        layer's stream waits for the index stream before DcpValue."""
        batch = projected.shape[0]
        mq, index = self.kernels[batch, q][:2]
        box = self.boxes[batch]
        main = torch.cuda.current_stream(self.device)
        side = side_stream(self.device.index)
        side.wait_stream(main)
        qc = to_cute(self.qc[:batch])
        norm = self.norms[batch]
        norm(q_args[0], self.views[0], qc, self.eps, stream(self.device))
        with torch.cuda.stream(side):
            mq(*q_args, *publish, stream(self.device))
        mailbox = publish[0]

        def join():
            joining = torch.cuda.current_stream(self.device)
            joining.wait_stream(side)
            query_wait(mailbox, stream(self.device))

        def chain():
            self.run_index(q_args[0], index_slots, batch, index)
            topk = self.indexer_core(projected, batch)
            return self.dcp_core(cache, topk, metadata, batch, box, join)

        if batch not in self.prioritized:
            return chain()
        hi = index_stream(self.device.index)
        hi.wait_stream(main)
        with torch.cuda.stream(hi):
            core = chain()
        main.wait_stream(hi)
        return core

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
