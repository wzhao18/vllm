# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K3.1 DSA layers of the sequence-parallel decode (C512 decode role: TP4 = DCP4 =
the SP group, 24 of 96 heads per rank, DSpark q 4, per-tensor FP8 MLA KV on
FlashInfer trtllm-gen, MXFP4 indexer). The SP block (runtime.sp_attention) gathers
the shard rows, projects every one of the R = 4 n rows (n = 1..128 requests per DP
group, 0.25..32 per GPU) and hands them here; this is PreparedDsa on one DcpBox of
512 rows:

  MlaQuery (dsa)   the 24-head tiled query (mla.query TILED_FROM[24]: 8-CTA
                   clusters x 24 heads x 16-64-request tiles), the KV insert of
                   this rank's slots, and the publish of its heads into every
                   rank's FP8 q mailbox [2, 512, 128 (96 padded), 576]. It never
                   polls (mla.query UNGATHERED_HEADS: one tile's 24 clusters exceed
                   the 22 Rubin holds at once, so a gathered poll could deadlock
                   across ranks); QueryWait polls the mailbox instead (the fork /
                   front query overlap, ``SP_QUERY_OVERLAP``), whose CTAs wait only
                   on other ranks' MlaQuery, which waits on nothing.
  index            the fused index front (front) or QueryNorm -> wq_b -> DsaIndex
                   (fork) into the MXFP4 (or FP8) indexer query, key insert
  top-k            exact over the bf16 logits (IndexerStage): DcpExchange's this-rank
                   slots on its automatic policy (the stock hook, one round up to 256
                   rows, two above) where it admits the step, else the #63 merge, else
                   the stock op's global ids filtered to this rank's slots
  sparse core      FlashInfer trtllm-gen over the gathered 128 query heads and this
                   rank's KV shard (``SP_KERNELS``; prims_ts is refused: SM107 has
                   none, and the stock C512 decode runs trtllm-gen)
  DcpValue "proj"  the W_UV bmm of every gathered head, the value rows (4x fewer
                   NVLink bytes than the latents) pushed to their head owner, the
                   LSE merge of this rank's 24 heads (FP32, one BF16 rounding) and
                   the sigmoid gate -> [R, 24 x 128], the stock o_proj input

Precision is the stock decode's: the same FP8 query and KV row (MlaQuery's
per-tensor quantization), the same trtllm-gen kernel on the same padded heads, an
exact top-k, and "proj" moves only the BF16 rounding point of the partials (the
merge is linear: sum_s w_s (O_s W) = (sum_s w_s O_s) W).

Every captured row count either compiles here or raises ``Unsupported`` at build,
naming the cause: there is no stock DSA fallback under SP."""

import os
from functools import cache
from typing import NamedTuple

import torch

from vllm.logger import init_logger

from ...shapes import Choice, Path, Shape, Unsupported
from ..gemm.decode import DecodeProjection
from ..gemm.sp_policy import sp_projection_tactics
from ..mla.dcp import DcpBox
from ..mla.query import TILED_FROM
from .attn_fi import dsa_on_flashinfer, kernel_for
from ..common.policy import batch_table
from .dcp_value import PROJECTED
from .index import DIM, HEADS
from .local_query import LocalQuery
from .overlap import INDEX_PRIORITY, overlap_for
from .prepared import PreparedDsa
from .r1 import fi_buffers

logger = init_logger(__name__)

SP_HEADS = 24  # query heads per TP rank at C512 (96 / TP4)
SP_ROWS = 512  # the most rows of a captured SP step (max_num_seqs 128 x q 4)
# Query overlap per rows (dsa.overlap MODES): every mode whose MlaQuery publishes
# without polling (``SP_MODES``; "serial" would poll in MlaQuery, "late" is TP-only).
# The Rubin DCP16 table's split ("dcp_rubin": front up to 16 rows, then fork),
# open-ended. NQ-TUNE: front vs fork vs wait at TP4 / DCP4, 24 heads, q4, rows
# 4..512 (u128k and AgentX contexts; front's IndexFrontMma from 41 rows).
SP_QUERY_OVERLAP = "1-16:front,17-:fork"
SP_MODES = ("front", "fork", "wait")
# DcpValue partials at every row: "proj". The BF16 latent partials win only where
# the DSA sparse core (dsa_kernels.dsa_core) pushes them, which SP never runs; on
# VR200 with the core off "proj" wins from the first row (1 GPU, fp8 KV, DcpValue +
# partials, AgentX contexts, us bf16 -> proj: q4 M4 10.71 -> 9.32, M8 11.47 -> 9.43,
# M12 13.53 -> 9.94, M16 14.52 -> 10.42; r1.py), and it needs no BF16 partial
# mailboxes (2 x 4 x 512 x 24 x 512 BF16: 200 MiB symmetric). NQ-TUNE: bf16 vs proj
# at DCP4 / 24 heads.
SP_PARTIALS = PROJECTED
# Rows whose query of all 96 heads is computed on every rank (local_query.LocalQuery:
# no q publish, QueryWait or q mailbox re-arm), the rest MlaQuery's publish of 24
# heads (fork rows only: QueryNorm writes q_c). In-step (gap512 D3 B80 / B92, 320 /
# 368 rows) MlaQuery ran 74-89 us per layer on the side stream and QueryWait 30-46
# us, beside the index chain whose wq_b split-K it slowed to 52-62 us. VR200, 1 GPU,
# cold L2, us, MlaQuery without its publish vs LocalQuery (runs/D-ATTN qpath1-5):
#   rows      20    32    64    96   128   256   384   512
#   MlaQuery 17.2  17.4  23.7  32.3  33.4  52.5  72.4  93.1
#   Local    26.7  26.4  30.2  25.2  29.0  30.9  36.9  44.2
# NQ-TUNE: 17-64 rows in-step (the publish and QueryWait the local query also saves).
# In-step D3-32 slice (EP32 k3 B16 = 64 rows, fork): MlaQuery (24 us) ran serialized
# after the top-k on the main stream instead of beside the index chain, then QueryWait
# (DSA layer span 136 us vs 102 us at 128 local rows); local from 17 rows: k3 B8 / B12
# / B16 (32 / 48 / 64 rows) -2.7 / -3.5 / -3.8 % per 24L step. Numerics: the stock q_b /
# W_UK / FP8 quantization in another K order (real weights: 1.2-2.1e-4 of FP8 bytes one
# ulp off, same max error vs FP32). K31_SP_LOCAL_QUERY replaces the table for A/B (the
# previous default was "65-:local").
SP_LOCAL_QUERY = "17-:local"
if os.environ.get("K31_SP_LOCAL_QUERY"):
    SP_LOCAL_QUERY = os.environ["K31_SP_LOCAL_QUERY"]
# The sparse MLA kernel per row count ("rows:kernel" table; trtllm-gen elsewhere): the
# Rubin keys-on-M kernel ("km", dsa.sparse_mla_km: softmax groups correct O^T and
# write the latent, QK and PV from separate warps, all-TMA gathers; one row per
# persistent CTA and SM, its LSE written directly) where it wins. VR200 (212 SMs),
# 1 GPU, CUDA graphs, cold L2, 512 +- 40 local keys per row, 4 rows per request
# sharing 80% of their keys (runs/D-ATTN-SK/r2xo2), km vs trtllm-gen us (trtllm-gen
# also runs ComputeLSEFromMD, ~1.1 us):
#   rows        64    96   128   160   192   212   224   256   320
#   km        15.1  15.4  16.0  16.8  18.3  18.4  28.7  29.1  29.9
#   trtllm    13.7  15.6  22.1  22.8  23.1  23.1  27.4  28.4  29.8
# km is flat up to one row per SM; trtllm-gen steps at ~106 rows, km at 213 (two
# rows on some CTAs). In-step (D3-32 slice, fastab dsk2-km65): 128 rows -29.5 us per
# step, 256 rows +11.9; 384 rows: km since dattn-kmg/kmh (step -14 us).
# The latent-Q-in-TMEM kernel ("ts", dsa.sparse_mla_ts: the prefill TS form with the
# trtllm-gen arithmetic) from 64 rows: VR200 1 GPU, CUDA graphs, cold L2, us at 128 /
# 256 / 384 rows: ts 16.0 / 28.8 / 30.1, trtllm-gen 21.5 / 28.2 / 34.9, km 16.0 / 29.2 /
# 31.5 (runs/D-ATTN/r3 e8); in-step vs int21 (trtllm-gen at 128 / 256, km at 384 rows;
# fastab dattn3-ts1/ts2, 3 build pairs) -43.3 / -25.6 / -39.2 us per step. Below 64
# rows trtllm-gen splits each row's KV across CTAs (BF16 partials), which an unsplit
# pass does not reproduce to the BF16 rounding of the output: those stay on trtllm-gen.
SP_KERNELS = "64-:ts"
SP_KV_DTYPES = ("fp8", "fp8_e4m3")  # per-tensor FP8 MLA KV


def sp_dsa_box(group, device, rank, *, heads=SP_HEADS, cap=SP_ROWS, interleave):
    """The model's DSA DcpBox under SP (one per model, every DSA layer alternating
    its two parities): ``group`` the TP (= DCP) process group, ``cap`` rows, the FP8
    q mailbox padded to trtllm-gen's query heads (96 -> 128), no BF16 partial
    mailboxes (``SP_PARTIALS`` "proj": r1.DsaBox adds its own value-row mailbox).
    Collective over ``group``."""
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
        trtllm_gen_query_heads,
    )

    size = torch.distributed.get_world_size(group)
    q_heads = trtllm_gen_query_heads(size * heads)
    if q_heads > 128:
        raise Unsupported(
            "dsa",
            None,
            f"{size} x {heads} gathered query heads: trtllm-gen sparse MLA reads at "
            "most 128",
        )
    try:
        return DcpBox(
            group,
            device,
            size,
            rank,
            heads,
            cap=cap,
            interleave=interleave,
            q_heads=q_heads,
            partials=False,
        )
    except RuntimeError as error:  # (no NVLink multicast: re-raised, never stock)
        raise Unsupported("dsa", None, f"DSA DcpBox: {error}") from error


class SpDsaScratch(NamedTuple):
    """The SP DSA layers' shared rows (they run one at a time on one stream pair):
    the normalized q_c, wq_b's indexer query, DsaIndex's FP8 / MXFP4 indexer query
    and its head weights. The FP8 query rows and FlashInfer's buffers, whose shapes
    follow the layer, are shared per (device, rows, heads) (``query_rows``,
    r1.fi_buffers)."""

    qc: torch.Tensor  # [rows, 1536] BF16
    iq: torch.Tensor  # [rows, 32 x 128] BF16
    q_fp8: torch.Tensor  # [rows, 32, 128] bytes
    head_w: torch.Tensor  # [rows, 32] BF16


def sp_dsa_scratch(device, rows_cap: int = SP_ROWS) -> SpDsaScratch:
    bf16 = dict(dtype=torch.bfloat16, device=device)
    return SpDsaScratch(
        torch.empty((rows_cap, 1536), **bf16),
        torch.empty((rows_cap, HEADS * DIM), **bf16),
        torch.empty((rows_cap, HEADS, DIM), dtype=torch.uint8, device=device),
        torch.empty((rows_cap, HEADS), **bf16),
    )


@cache
def query_rows(device, rows, heads):
    """MlaQuery's FP8 query rows [rows, heads, 576], shared by every SP DSA layer:
    under DCP nothing reads them back (the sparse core reads the gathered q mailbox),
    so one buffer serves all layers instead of PreparedDsa's per-layer one."""
    return torch.zeros((rows, heads, 576), dtype=torch.uint8, device=device)


def sp_unsupported(attention, box, ctx):
    """Why a KimiK3DSAAttention cannot run SpDsaCore on ``box`` under ``ctx``, or
    None (the checks every row count shares)."""
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
        trtllm_gen_query_heads,
    )

    wrapper, indexer = attention.mla_attn, attention.dsa_indexer
    layer = wrapper.mla_attn
    heads = layer.W_UK_T.shape[0]
    padded = trtllm_gen_query_heads(box.size * heads)
    ranks = (indexer.dcp_rank, indexer.dcp_world_size, box.rank, box.size)
    checks = (
        (not ctx.sequence_parallel, "not the sequence-parallel decode"),
        (
            not dsa_on_flashinfer(attention),
            f"the {type(layer.impl).__name__} backend (FLASHINFER_MLA_SPARSE only)",
        ),
        (
            layer.kv_cache_dtype not in SP_KV_DTYPES,
            f"kv_cache_dtype {layer.kv_cache_dtype} (per-tensor FP8 only)",
        ),
        (
            bool(getattr(layer.impl, "use_prims_ts", False)),
            "the prims_ts sparse MLA kernel (trtllm-gen only)",
        ),
        (
            getattr(indexer, "dsac", None) is not None,
            "the DSA sparse core (its build() leaves SP to trtllm-gen)",
        ),
        (heads != box.heads, f"{heads} heads per rank on a {box.heads}-head box"),
        (
            bool(ctx.heads_per_rank) and heads != ctx.heads_per_rank,
            f"{heads} heads per rank (the deployment's {ctx.heads_per_rank})",
        ),
        (heads not in TILED_FROM, f"{heads} heads (no MlaQuery geometry)"),
        (
            not (box.size == ctx.dcp == ctx.tp),
            f"DCP{box.size} box under TP{ctx.tp} / DCP{ctx.dcp} (DCP = TP only)",
        ),
        (
            (indexer.dcp_world_size, indexer.dcp_rank) != (box.size, box.rank),
            "indexer DCP rank {} of {}, box rank {} of {}".format(*ranks),
        ),
        (
            box.q_heads != padded,
            f"a {box.q_heads}-head q mailbox (trtllm-gen reads {padded})",
        ),
        (box.q.dtype != torch.uint8, "a BF16 q mailbox (FP8 for FlashInfer)"),
    )
    return next((cause for failed, cause in checks if failed), None)


class SpDsaCore:
    """One K3.1 DSA layer of the SP decode. ``attention``: the KimiK3DSAAttention
    (stock module); ``rows``: every captured row count R it serves (4..512, q per
    request); ``scratch``: sp_dsa_scratch; ``box``: sp_dsa_box (shared by every DSA
    layer); ``parity``: the DSA-layer ordinal % 2. Built collectively (the "proj"
    W_UV gather and mailboxes), in layer order on every rank."""

    def __init__(self, attention, rows, scratch, *, box, parity, ctx):
        from ...runtime.providers import dsa_seam_columns, pack_dsa_projection

        cause = sp_unsupported(attention, box, ctx)
        if cause is not None:
            raise Unsupported("dsa", None, cause)
        if parity not in (0, 1):
            raise Unsupported("dsa", None, f"parity {parity}")
        self.q, self.ctx = ctx.q, ctx
        q = ctx.q
        self.rows = rows = tuple(sorted(set(rows)))
        if not rows:
            raise Unsupported("dsa", None, "no captured row counts")
        layer = attention.mla_attn.mla_attn
        self.heads = heads = layer.W_UK_T.shape[0]
        self.device = device = layer.W_UK_T.device
        self.pitch = pitch = pack_dsa_projection(attention).shape[0]
        try:
            dsa_seam_columns(attention, pitch)
        except ValueError as error:  # (re-raised: the packed row layout differs)
            raise Unsupported("dsa", None, str(error)) from error
        expected = 2112 + heads * 128 + DIM + HEADS
        if pitch != expected:
            raise Unsupported(
                "dsa", None, f"packed in-projection pitch {pitch} (expected {expected})"
            )
        self.modes: dict[int, str] = {}
        for r in rows:
            shape = Shape(r, q)
            if r % q:
                raise Unsupported("dsa", shape, f"{r} rows are not whole q{q} requests")
            if r > box.cap:
                raise Unsupported("dsa", shape, f"above the DcpBox's {box.cap} rows")
            if r < TILED_FROM[heads]:
                raise Unsupported(
                    "dsa", shape, f"below the {heads}-head tiled MlaQuery's rows"
                )
            mode = overlap_for(r, SP_QUERY_OVERLAP, True, True)
            if mode not in SP_MODES:
                raise Unsupported(
                    "dsa", shape, f"query overlap {mode!r} (publish-only modes only)"
                )
            self.modes[r] = mode
        table = batch_table(SP_LOCAL_QUERY)
        self.local_rows = tuple(r for r in rows if table.get(r) == "local")
        for r in self.local_rows:
            if self.modes[r] != "fork":
                raise Unsupported(
                    "dsa", Shape(r, q), f"local query under {self.modes[r]!r} (fork)"
                )
        local = None
        if self.local_rows:  # (collective: the q_b / W_UK gathers)
            local = LocalQuery(
                attention, box.group, box.size, self.local_rows, ctx, box.q_heads
            )
        indexer = attention.dsa_indexer
        self.topk = indexer.topk_tokens
        fi = fi_buffers(device, max(rows), self.topk, box.q_heads)
        shared = (query_rows(device, max(rows), heads), *scratch)
        self.prepared = PreparedDsa(
            attention,
            dict.fromkeys(rows, pitch),
            (*shared, fi),
            queries=(q,),
            dcp=box,
            parity=parity,
            kernels=SP_KERNELS,
            overlap=SP_QUERY_OVERLAP,
            partials=SP_PARTIALS,
            priority=INDEX_PRIORITY["sp"],
            local=local,
        )
        prepared = self.prepared
        # The per-layer FP8 query rows PreparedDsa's QueryStage took: the shared ones
        # (``query_rows``).
        prepared.query_stage.query = shared[0]
        # wq_b at the SP rows: ``gemm.sp_policy``'s measured tactics (family "wq_b";
        # the TP16 bins' split-K tiles of 64 token columns re-read the weight per
        # tile above 64 rows) in place of QueryStage's TP16 table, same operands.
        stage = prepared.query_stage
        wq = stage.index_projection.weight
        n, k = wq.shape
        tactics = sp_projection_tactics("wq_b", n, k, rows, ctx)
        stage.index_projection = DecodeProjection(wq, rows, tactics, family="wq_b")
        stage.index_projection.check_operands(
            stage.qc[: max(rows)], stage.iq[: max(rows)]
        )
        self.box = prepared.box  # (r1.DsaBox over ``box``)
        for r in rows:  # the policy PreparedDsa resolved is the one checked above
            shape = Shape(r, q)
            core, mode = prepared.cores[r, q], prepared.overlaps[r, q]
            fmt = self.box.mailbox(r).format
            kernel = kernel_for(r, True, SP_KERNELS)
            if (core, mode, fmt) != (kernel, self.modes[r], SP_PARTIALS):
                raise Unsupported("dsa", shape, f"resolved {core} / {mode} / {fmt}")
        if prepared.dsac is not None:  # (sp_unsupported refused it)
            raise Unsupported("dsa", None, "the DSA sparse core")
        self.prefix = indexer.k_cache.prefix

    def prepare_cache(self, kv):
        """Compile every row count's kernels for the bound caches (``kv``: the MLA
        cache) and check that each persistent grid whose CTAs wait on one another is
        co-resident ("proj" DcpValue: its merging CTAs poll the value rows every
        rank's projecting CTAs push)."""
        from . import wide

        self.prepared.prepare_cache(kv)
        sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        for r in self.rows:
            value = self.prepared.kernels[r, self.q][2]
            try:
                wide.check_proj_residency(
                    value, self.heads, self.box.size, r, sms, self.device.index
                )
            except RuntimeError as error:  # (re-raised with the shape)
                raise Unsupported("dsa", Shape(r, self.q), str(error)) from error
        self.log_policy()
        return self

    def log_policy(self):
        """The SP DSA policy at startup (once, global rank 0)."""
        from vllm.model_executor.layers.sparse_attn_indexer import (
            DCP_DECODE_TOPK_HOOKS,
        )

        stage = self.prepared.indexer_stage
        if self.prefix in DCP_DECODE_TOPK_HOOKS:
            topk = "DcpExchange where its policy admits the step"
        else:
            topk = "no DcpExchange hook"
        if stage.local_topk:
            topk += f", else the #63 merge (N={stage.local_topk})"
        topk += ", else the stock op + local-slot filter (all exact)"
        modes: dict[str, list[int]] = {}
        for r in self.rows:
            modes.setdefault(self.modes[r], []).append(r)
        spans = "; ".join(
            f"{mode} {rows[0]}-{rows[-1]} ({len(rows)} sizes)"
            for mode, rows in modes.items()
        )
        logger.info_once(
            "Kimi K3.1 SP DSA (TP%d = DCP%d, %d heads, q%d, rows %d-%d): query %s; "
            "MlaQuery publishes, QueryWait polls (local query of every head at %s); "
            "top-k %s; trtllm-gen over %d query "
            'heads; DcpValue "%s"; box %d rows (%.0f MiB symmetric)',
            self.ctx.tp,
            self.box.size,
            self.heads,
            self.q,
            self.rows[0],
            self.rows[-1],
            spans,
            f"{len(self.local_rows)} sizes" if self.local_rows else "no rows",
            topk,
            self.box.q_heads,
            SP_PARTIALS,
            self.box.cap,
            self.box.box.nbytes()[0] / 2**20,
            scope="global",
        )

    def run(self, projected, out, *, cache, slots, index_slots, metadata, q):
        """``projected`` [R, pitch] BF16: the packed in-projection rows of all R rows;
        ``out`` [R, 24 x 128] BF16 gets the gated attention output (the stock o_proj
        input). ``slots`` / ``index_slots``: the MLA and indexer cache slots;
        ``metadata``: the layer's full-R uniform-verify metadata."""
        rows = projected.shape[0]
        if q != self.q or rows not in self.modes:
            raise Unsupported("dsa", Shape(rows, q), "not a prepared row count")
        return self.prepared.run(
            projected,
            out,
            cache=cache,
            slots=slots,
            index_slots=index_slots,
            metadata=metadata,
            q=q,
        )

    def choice(self, rows):
        """The coverage choice at ``rows``: GENERIC (specialized, untuned at DCP4);
        its tactic is (query overlap, sparse kernel, DcpValue partials)."""
        if rows not in self.modes:
            raise Unsupported("dsa", Shape(rows, self.q), "not a prepared row count")
        mode = self.modes[rows]
        return Choice(
            Path.GENERIC,
            tactic=(mode, kernel_for(rows, True, SP_KERNELS), SP_PARTIALS),
            note=f"sp {mode}/{kernel_for(rows, True, SP_KERNELS)}/{SP_PARTIALS}",
        )
