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

from functools import cache
from typing import NamedTuple

import torch

from vllm.logger import init_logger

from ...shapes import Choice, Path, Shape, Unsupported
from ..mla.dcp import DcpBox
from ..mla.query import TILED_FROM
from .attn_fi import dsa_on_flashinfer
from .dcp_value import PROJECTED
from .index import DIM, HEADS
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
SP_KERNELS = ("trtllm",)  # the FlashInfer sparse MLA kernel at every row
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
        )
        prepared = self.prepared
        # The per-layer FP8 query rows PreparedDsa's QueryStage took: the shared ones
        # (``query_rows``).
        prepared.query_stage.query = shared[0]
        self.box = prepared.box  # (r1.DsaBox over ``box``)
        for r in rows:  # the policy PreparedDsa resolved is the one checked above
            shape = Shape(r, q)
            core, mode = prepared.cores[r, q], prepared.overlaps[r, q]
            fmt = self.box.mailbox(r).format
            if (core, mode, fmt) != ("trtllm", self.modes[r], SP_PARTIALS):
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
            "MlaQuery publishes, QueryWait polls; top-k %s; trtllm-gen over %d query "
            'heads; DcpValue "%s"; box %d rows (%.0f MiB symmetric)',
            self.ctx.tp,
            self.box.size,
            self.heads,
            self.q,
            self.rows[0],
            self.rows[-1],
            spans,
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
            tactic=(mode, SP_KERNELS[0], SP_PARTIALS),
            note=f"sp {mode}/{SP_KERNELS[0]}/{SP_PARTIALS}",
        )
