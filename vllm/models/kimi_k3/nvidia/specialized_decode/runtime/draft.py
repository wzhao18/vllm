# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The K3 DSpark drafter's specialized pieces, enabled with the target's specialized
decode (``decode_backend: latency_specialized``) on Rubin-class devices.

``greedy_markov``: the drafter's greedy sequential Markov sampling as the LM-head shard
GEMM plus one vocabulary-sharded kernel (cutedsl.draft.markov) instead of a full-vocab
logits all-gather and a per-position replicated Markov head. Runs only inside CUDA graph
captures, so every rank takes the same path (the kernel exchanges keys with every rank).
``greedy_markov(..., sampling)``: the same for probabilistic drafting
(``draft_sample_method: probabilistic``): the kernels draw as the stock loop's
``gumbel_sample`` (bitwise its Gumbel noise and scores) and write this rank's columns
of the draft-logits cache;
``gather_draft_logits`` completes the cache rows for the stock rejection sampler when a
verification falls back to it.

The policy tables are keyed by rows (``shapes.rows_policy`` at Shape(rows, draft
length)): query rows R = requests x draft length for the backbone, the LM head and the
all-reduce sites, context rows C = requests x (1 + draft length) for the context step;
the Markov tables by requests (the kernel's width). Every decision is a pure function
of the deployment, the same on every rank. A range written "lo-" (no upper end) is
open: it covers every larger count the deployment captures (``closed``), where the
kernels are shape-generic and size their buffers from the captured rows; what a
kernel cannot hold (shared memory, co-residency) is dropped at build and logged, and
stays on the stock path.
"""

import torch

from vllm.logger import init_logger

from ..cutedsl.common.policy import batch_table
from ..shapes import (
    Shape,
    StockSteps,
    full_capture,
    log_coverage,
    refuse,
    rows_policy,
    target_topology,
)

logger = init_logger(__name__)

# Requests per drafter graph served by the fused greedy Markov kernel, per TP, on Rubin
# (compute capability 10.7); 1: plain (above 8 requests ``markov_wide.MarkovWide``, the
# two-level key exchange; drafts identical to the stock loop); 2: its key polls gated
# on one chunk per rank (cutedsl.draft.markov ``gate``; 16 ranks,
# scripts/dsdraft/markov_test.py, us
# ungated -> gated: B1 35.78 -> 31.20, B4 37.83 -> 34.30, B8 42.42 -> 41.51, drafts
# identical); 3: gated, and the previous tokens' markov_w1 rows staged in shared
# memory once per CTA (``staged``: from B3 the per-warp fragment loads touch a line per
# request; 16 ranks, us: B4 34.76 -> 30.47, B8 41.26 -> 31.69, drafts identical; B1 /
# B2 lose ~0.5 / ~1.4 to the extra barrier). GB300 keeps the stock drafter.
# (hb192: 9-128, MarkovWide's 128 slots; above them, and above the device's
# per-launch slots (markov_wide.launch_width), consecutive launches over request
# slices: the same drafts.) Markov rank 512 (128 rows per CTA, markov.CTA_ROWS) takes
# the rank-256 choices. NQ-TUNE: rank 512 gate / staged / settle per batch
# (scripts/dsdraft/markov_test.py at B1-B8) and B65-B192 launches vs the stock loop.
FUSED_MARKOV = {8: "1-8:1", 16: "1-2:2,3-8:3,9-:1"}
# FUSED_MARKOV requests whose kernel waits at a CTA barrier between each position's key
# poll and its shuffle maxima (cutedsl.draft.markov ``settle``), per TP. 16 ranks, real
# DSpark weights, us: B1 31.19 -> 30.65 (31.37 -> 30.73), B2 / B8 flat; drafts
# identical.
SETTLED_MARKOV = {16: "1:1"}
# Query rows whose LM-head shard GEMM is the BF16-logits GEMV (cutedsl.verify.lmhead,
# at most 8 rows, the shard read in place with an L2 evict-first policy), per TP, on
# Rubin. 1 GPU, cold L2, 7 rows: 15.3 vs 18.5 us (torch.mm; 16-rank serving: cuBLAS
# 18.5 + split-K reduce 3.8). An FP32 accumulation reorder of the base logits.
FAST_LM_HEAD = {16: "1-7:1"}
# Query rows whose LM-head shard GEMM runs with evict-first weight
# loads (cutedsl.verify.lmhead.PreparedWideLmHead: the projection core without a K
# split, BF16 logits bitwise equal to cuBLAS's unsplit kernels), per TP, on Rubin.
# NQ-NOFALLBACK: 9 query rows (q4 B3) stay on the stock cuBLAS shard GEMM by choice:
# the measurement note on lmhead.WIDE_ROWS ends "(9 rows: 35 us)" and starts the
# evict-first kernel at 10 rows; it does not show the kernel winning at 9 (it is no
# faster than torch.mm at 16 / 56 / 64 rows either), so 9 rows are not moved without
# a 9-row measurement (NQ-TUNE). Above 64 rows (q8 B10+, q4 B22+) the stock cuBLAS
# shard GEMM stays too, logged in the coverage (unmeasured: NQ-TUNE).
WIDE_LM_HEAD = {16: "10-64:1"}  # (B9's 63 rows: within lmhead.WIDE_ROWS)
# Query rows whose context step (sharded projection + row gather, context-KV GEMM
# and insert) the caller may run on a side stream, joined by the first MLA layer
# right before its attention core (cutedsl.draft.attention ``join_context``): the
# embedding site, layer 0's in-projection and MlaQuery overlap it. Per TP, on Rubin.
# Only where MlaQuery does not poll (draft.attention FORK_ROWS at B1, POLLED_CORE at
# B2-B5 and B8): then before the join no main-chain kernel spins on a remote producer
# that the side chain's resident CTAs could starve (the embedding site waits only on
# the SM-light embedding gather); where MlaQuery gathers q, that does not hold. 16
# ranks (scripts/dsdraft/overlap_test.py, serving tables), us serial -> overlap: B2
# 268.6 -> 258.2, B3 278.2 -> 266.4, B4 282.9 -> 269.8, B5 304.6 -> 292.2, B8 319.0 ->
# 305.5; per-rank random delays: 60-150 trials each, no mismatch (unjoined control:
# 59-150 of 60-150).
# (to 63 rows: B9's rows are polled, draft.attention POLLED_CORE)
CONTEXT_OVERLAP = {16: "1-63:1"}
# Query rows whose MLA layers (DCP = TP) exchange queries and partials through the
# specialized decode's Lamport mailboxes (cutedsl.draft.attention), per TP, on Rubin.
# (hb192: to 576 query rows, DL3 B192; open above: cutedsl.draft.attention DraftDcp
# keeps the wide rows whose DcpValue grid is co-resident, with the work list where it
# fits mla.wlist's packing.)
# NQ-TUNE: drafter wide rows above 576 (DL7 B83+, DL3 B193+) vs the stock DCP path.
FAST_DCP = {8: "1-64:1", 16: "1-:1"}
# Context rows whose context step runs the sharded context projection + Lamport row
# gather and the fused context-KV epilogue (cutedsl.draft.context; above 64 rows the
# shape-dynamic kernels and the stock shard GEMM), per TP, on Rubin. It replaces the
# replicated context_proj GEMM (514 MB of weights per rank).
# (hb192: to 768 context rows, DL3 B192; open above: cutedsl.draft.context keeps the
# rows its per-row gather CTAs can hold at once.) NQ-TUNE: above 768 context rows.
FAST_CONTEXT = {8: "1-64:1", 16: "1-:1"}
# Query rows whose forward runs the all-reduce + residual + RMSNorm sites as Lamport
# kernels (cutedsl.draft.allreduce): 1 one-shot, 2 two-shot (common.twoshot), per TP,
# on Rubin. 16 ranks, us/site one-shot / two-shot (scripts/dsdraft/ar_test.py, chain
# of 11 sites): 7 rows 8.60 / -, 14 rows 10.01 / 9.06, 28 rows 12.66 / 9.19, 56 rows
# 18.41 / 9.78 (serving FlashInfer: ~13). Bitwise equal to FlashInfer's fused
# all-reduce + RMSNorm semantics. 3: above 64 rows, two-shot with one CTA per row
# (cutedsl.draft.allreduce ArNormRows: one wave; the two-shot sites' sums and rounding,
# bitwise), on the stock GEMMs' partials. 4 ranks + 12 emulated, us per site with its
# o_proj / down_proj GEMM (cuBLAS), 112 / 168 / 224 rows: 10.66 / 11.23 / 17.22 (K 512),
# 11.51 / 11.97 / 17.98 (K 896); serving's stock GEMM + FlashInfer site ~19-25.
# (hb192: ArNormRows to 576 query rows, DL3 B192; open above, its two-shot box sized
# from the deployment's rows. Beyond one wave of 1024-thread CTAs a row's CTA waits
# only on the same row's CTAs of the other ranks, which push before they wait.)
# NQ-TUNE: ArNormRows above 576 rows (DL7 B83+) vs the stock site.
FAST_AR = {8: "1-7:1", 16: "1-7:1,8-64:2,65-:3"}
# FAST_AR one-shot batches whose sites gate their 16-source poll on one fragment per
# source (cutedsl.draft.allreduce ``gate``), per TP, on Rubin. 16 ranks, the B1
# drafter forward (scripts/dsdraft/layer_test.py, in process), us ungated -> gated:
# 210.1 -> 206.4 (209.3 -> 206.2); with the gated q gather 208.4 -> 204.5 (207.2 ->
# 204.1); outputs bitwise identical.
GATED_AR = {16: "1-7:1"}
# VALID_AR one-shot query rows whose polled sites run before their PDL wait and
# wait at the end (cutedsl.draft.allreduce ArNorm ``early``), released by ring
# o_proj / down_proj producers that trigger after their own wait, per TP, on Rubin.
# 16 ranks, the B1 drafter forward (5 layers), us: 8.8k context 181.9 -> 178.5, 64k
# 196.4 -> 195.0; outputs bitwise identical.
SITE_DEFERRED = {16: "1-7:1"}
# One-shot (FAST_AR 1) query rows whose polled sites land their TP slots with
# validity-reporting bulk copies (cutedsl.draft.allreduce ArNorm ``valid``, mla.qvalid)
# instead of gated LSU polls, per TP, on Rubin. 16 ranks, the B1 drafter forward
# (scripts/dsdraft/layer_test.py), us LSU -> validity copies: 187.4 / 186.7 / 186.8 ->
# 181.7 / 181.0 / 181.7 at local 560, 202.6 / 201.7 -> 196.2 / 196.2 at local 4096;
# bitwise identical.
VALID_AR = {16: "1-7:1"}
# Rubin TP16, drafter GEMMs of at most 8 rows (B1: 7 query rows, 8 context rows): the
# split-K projection (cutedsl.gemm.projection) instead of cuBLAS, tactic by (N, K).
# 1 GPU, cold weights, us (cuBLAS -> split-K): qkv_a 6.19 -> 3.91, q_b 3.45 -> 1.80,
# o_proj 3.00 -> 2.07, gate_up 6.04 -> 3.82, down 4.02 -> 2.47, context shard
# 6.72 -> 5.42, context KV 6.83 -> 4.68 (scripts/dsdraft/gemm_sweep.py). At 56-64 rows
# cuBLAS wins. Split-K sums FP32 partials over K slices (an accumulation reorder). The
# context shard at split 14 (clusters of 14 CTAs): 16 ranks, sharded projection +
# CtxGather 10.13 -> 8.62 us, + the context-KV GEMM 14.89 -> 13.77 vs split 8
# (scripts/dsdraft/ctx_test.py); the same distance to stock (max rel 3.3e-3).
# kimi-k3.1-pro-dspark-preview (96 heads: 6 per rank) has q_b (1152, 1536) and o_proj
# (7168, 768) (``DRAFT_96``): q_b at the 64-head q_b's split (K 1536 = 4 x 3 tiles of
# 128, 18 M tiles instead of 12), o_proj as the K 512 / 896 neighbours (split 1, all
# 6 K tiles staged). NQ-TUNE: sweep both (scripts/dsdraft/gemm_sweep.py) at <= 8 rows.
DRAFT_96 = {(1152, 1536): (64, 8, 4, 4), (7168, 768): (64, 8, 1, 6)}
PROJECTIONS = {
    16: {
        (2112, 7168): (64, 8, 4, 10),
        (768, 1536): (64, 8, 4, 4),
        (7168, 512): (64, 8, 1, 4),
        (1792, 7168): (64, 8, 4, 8),
        (7168, 896): (64, 8, 1, 8),
        (2880, 7168): (64, 8, 4, 8),
        (448, 35840): (64, 8, 14, 10),
        **DRAFT_96,
    }
}
PROJECTION_ROWS = 8
# Rubin TP16, drafter GEMMs of 9-64 rows (B2-B8 query rows, B2-B8 context rows) by row
# bin (largest rows, tactics by (N, K)): split-K (M64, N 16 / 32) up to 32 rows,
# SplitKTmemGemm64 (128, 64, split 6, x's slots, 14 TMEM tiles: the weights on chip
# before the PDL wait) for 33-64 rows; other shapes stay on cuBLAS. 1 GPU, cold
# weights, us (cuBLAS -> prepared; scripts/dsdraft/gemm_sweep.py): 14 rows qkv_a
# 6.30 -> 4.02, gate_up 6.08 -> 4.03, context KV 7.49 -> 4.74, q_b 3.58 -> 2.18, o_proj
# 2.71 -> 2.24, down 3.24 -> 2.66; 21-28 rows qkv_a 6.84 -> 5.08, gate_up 5.64 -> 4.85,
# o_proj 2.67 -> 2.31; 35-56 rows qkv_a 6.81 -> 5.36, gate_up 6.11 -> 4.29, context KV
# 7.79 -> 6.59. An accumulation reorder (split partials in rank order).
# DL7 B24 (168 query rows): qkv_a and gate_up on split-K 64.64.2.6 (wide/dg1.py: 1 GPU
# chain, spin stand-ins, cold weights, us cuBLAS -> split-K: qkv_a 9.17 -> 8.03, gate_up
# 8.96 -> 7.84; q_b, o_proj, down_proj stay on cuBLAS there). Rows 65-167 and above 168
# stay on cuBLAS (it wins at 112 / 224 rows except qkv_a / q_b / context KV tactics that
# need the tensor-memory kernels' 128-token tiles and short K).
DRAFT_168 = {(2112, 7168): (64, 64, 2, 6), (1792, 7168): (64, 64, 2, 6)}
PROJECTIONS_WIDE = {
    16: (
        (
            16,
            {
                (2112, 7168): (64, 16, 4, 8),
                (1792, 7168): (64, 16, 4, 8),
                (2880, 7168): (64, 16, 4, 8),
                (768, 1536): (64, 16, 4, 6),
                (7168, 512): (64, 16, 1, 8),
                (7168, 896): (64, 16, 1, 8),
                # (96 heads; NQ-TUNE: 9-16 rows)
                (1152, 1536): (64, 16, 4, 6),
                (7168, 768): (64, 16, 1, 8),
            },
        ),
        (
            32,
            {
                (2112, 7168): (64, 32, 4, 6),
                (1792, 7168): (64, 32, 4, 6),
                (2880, 7168): (64, 32, 4, 6),
                (7168, 512): (64, 32, 1, 4),
                (7168, 768): (64, 32, 1, 6),  # (96 heads; NQ-TUNE: 17-32 rows)
            },
        ),
        (
            64,
            {
                (2112, 7168): (128, 64, 6, 16, 14),
                (1792, 7168): (128, 64, 6, 14, 14),
                (2880, 7168): (128, 64, 6, 16, 14),
            },
        ),
        (167, {}),  # cuBLAS
        (168, DRAFT_168),
    )
}
# Rubin TP16, the all-reduce sites' rows (at most 8): o_proj and down_proj as ring
# projections (cutedsl.gemm.splitk_ring: the same split-K tiles, the BF16 partial rows
# multicast into the sites' Lamport slots instead of stored) and polled sites
# (cutedsl.draft.allreduce ``polled``), tactic by (N, K): ``PROJECTIONS``' with every K
# tile staged before the PDL wait (the chain below, o_proj 3 -> 4 and down 6 -> 7
# stages: 16 ranks 113.4 -> 111.2 us); gate_up_proj with silu_and_mul in its epilogue
# (cutedsl.gemm.splitk_situ.SplitKSiluMul, ``PROJECTIONS``' split); the sites release
# their PDL dependents right after their wait, so the next GEMM's weight loads (10
# of 14 K tiles staged for gate/up and qkv_a) overlap the exchange. Bitwise identical.
# 4 ranks, B1, the drafter forward with the qkv_a GEMM and a stand-in attention (5
# layers, scripts/dsdraft/chain_test.py): 138.3 -> 126.7 (rings) -> 110.6 (+ fused
# SiLU) -> 100.9 (+ early release) -> 99.1 us (10 stages; 11-12: same, split 7: +8).
# 96 heads: o_proj (7168, 768) with every one of its 6 K tiles staged (NQ-TUNE: the
# B1 chain at 6 heads per rank, scripts/dsdraft/chain_test.py).
RING_PROJECTIONS = {
    16: {
        (7168, 512): (64, 8, 1, 4),
        (7168, 896): (64, 8, 1, 7),
        (1792, 7168): (64, 8, 4, 10),
        (7168, 768): (64, 8, 1, 6),
    }
}
# Query rows with two-shot sites (FAST_AR 2) whose o_proj / down_proj run as
# two-shot producers (cutedsl.draft.producer: the target's tcgen05 o-projection
# producer pushing its BF16 rows to the column owners) and polled sites, per TP, on
# Rubin.
TWO_SHOT_PRODUCERS = {16: "8-64:1"}
# TWO_SHOT_PRODUCERS batches whose gate/up GEMM runs with silu_and_mul in its
# epilogue at the PROJECTIONS_WIDE tactic of the rows (cutedsl.gemm.splitk_situ
# SplitKSiluMul for the M tile 64 tactics, gemm.splitk_situ64 for the tensor-memory
# ones), bitwise the gate/up GEMM followed by silu_and_mul, per TP, on Rubin. 16
# ranks, the drafter forward with pushed two-shot sites (scripts/dsdraft/layer_test.py),
# us separate -> fused: B2 275.1 -> 262.4, B3 288.6 -> 274.6, B4 297.2 -> 283.2, B5
# 330.2 -> 317.1, B8 385.5 -> 372.0.
TWO_SHOT_SILU = {16: "8-64:1"}
# Query rows whose two-shot sites release their PDL dependents right after their
# own wait (cutedsl.draft.allreduce ArNormTwoShot ``trigger``), per TP, on Rubin: the
# in-projection and gate/up GEMMs after them (weights on chip before their wait)
# then load their weights while the site exchanges instead of after it. 16 ranks, the
# drafter forward, us at the end -> early: B2 254.8 -> 250.5, B3 259.7 -> 257.9, B4
# 266.9 -> 260.8, B5 296.6 -> 282.4, B8 321.4 -> 300.7 (bitwise). Those sites prefer
# the largest shared-memory carve-out, so that the tensor-memory GEMMs can place beside
# them (16 ranks, us: B5 284.1 -> 262.9, B8 300.8 -> 285.3, B2 -1.6, B3 / B4 flat).
TWO_SHOT_EARLY = {16: "8-64:1"}


class DraftProjectionMethod:
    """A drafter linear's quant method: the prepared split-K projection for its
    prepared row counts inside CUDA graph captures, else the original method."""

    def __init__(self, base, prepared):
        self.base, self.prepared = base, prepared

    def apply(self, layer, x, bias=None):
        if (
            bias is None
            and x.dim() == 2
            and x.shape[0] in self.prepared.kernels
            and x.is_contiguous()
            and torch.cuda.is_current_stream_capturing()
        ):
            out = x.new_empty((x.shape[0], layer.weight.shape[0]))
            return self.prepared.run(x, out)
        return self.base.apply(layer, x, bias)

    def __getattr__(self, name):
        return getattr(self.base, name)


def tabled_shapes(tp):
    """Every (N, K) some split-K table of ``tp`` lists (any row bin)."""
    shapes = set(PROJECTIONS.get(tp, {}))
    for _, table in PROJECTIONS_WIDE.get(tp, ()):
        shapes |= set(table)
    return shapes


def projection(weight, tp, rows):
    """The prepared split-K projection of ``weight`` for ``rows`` (or None)."""
    shape = tuple(weight.shape)
    policies = {}
    for r in rows:
        if r <= PROJECTION_ROWS:
            tactic = PROJECTIONS.get(tp, {}).get(shape)
        else:
            bins = PROJECTIONS_WIDE.get(tp, ())
            table = next((t for top, t in bins if r <= top), {})
            tactic = table.get(shape)
        if tactic is not None:
            policies[r] = tactic
    if not policies:
        return None
    from ..cutedsl.gemm.projection import PreparedProjection

    return PreparedProjection(weight, policies).prepare()


def install_projections(model, tp, query_rows, context_rows, rank=0):
    """Split-K projections for the drafter's per-layer linears (``query_rows``) and
    its context KV projection (``context_rows``); returns how many were installed."""
    linears = []
    for layer in model.model.layers:
        attn, mlp = layer.self_attn, layer.mlp
        linears += [(attn.fused_qkv_a_proj, query_rows), (attn.q_b_proj, query_rows)]
        linears += [(attn.o_proj, query_rows), (mlp.gate_up_proj, query_rows)]
        linears += [(mlp.down_proj, query_rows)]
    linears.append((model.model.context_kv_proj, context_rows))
    installed = 0
    # Shapes no table lists (a new drafter geometry): their rows stay on cuBLAS.
    tabled = tabled_shapes(tp)
    missing = sorted(
        {
            tuple(linear.weight.shape)
            for linear, _ in linears
            if linear is not None
            and getattr(linear, "bias", None) is None
            and tuple(linear.weight.shape) not in tabled
        }
    )
    if tabled and missing:
        logger.warning_once(
            "Kimi K3 DSpark drafter: no split-K tactic for linear shapes (N, K) %s at "
            "TP%d: they run cuBLAS",
            tuple(missing),
            tp,
        )
    stock: dict[tuple, list[int]] = {}
    for linear, rows in linears:
        if linear is None or getattr(linear, "bias", None) is not None:
            continue
        prepared = projection(linear.weight, tp, rows)
        served = set(prepared.kernels) if prepared is not None else set()
        shape = tuple(linear.weight.shape)
        for r in rows:
            if r not in served and r not in stock.setdefault(shape, []):
                stock[shape].append(r)
        if prepared is not None:
            linear.quant_method = DraftProjectionMethod(linear.quant_method, prepared)
            installed += 1
    # NQ-NOFALLBACK: the drafter linears' stock cuBLAS rows (a measured choice for
    # the 64-head drafter: cuBLAS wins at 56-64 and 112 / 224 rows, see
    # PROJECTIONS_WIDE; the 96-head shapes DRAFT_96 are unmeasured: NQ-TUNE).
    for shape, rows in sorted(stock.items()):
        if rows:
            measured = "unmeasured, NQ-TUNE" if shape in DRAFT_96 else "measured"
            log_coverage(
                rank,
                "draft linear",
                f"(N, K) {shape}: stock cuBLAS at rows {sorted(rows)} ({measured})",
            )
    return installed


def covered(entry, rows, steps):
    """A rows-keyed drafter table's value at ``rows`` of a draft length ``steps``
    deployment (None: not covered)."""
    return rows_policy(entry, Shape(rows, steps), default=None)


def closed(spec, top):
    """``spec`` with its open ranges ("lo-:value": every count from lo) closed at
    ``top``, the deployment's largest count (a ``batch_table`` spec)."""
    items = []
    for item in filter(None, spec.split(",")):
        span, value = item.split(":")
        if span.endswith("-"):
            lo = int(span[:-1])
            if lo > top:
                continue
            span = f"{lo}-{top}"
        items.append(f"{span}:{value}")
    return ",".join(items)


def draft_batches(vllm_config, steps):
    """Requests per drafter FULL graph: round_up(s, steps) / steps for each capture
    size whose rounded size fits the largest capture size and whose requests fit
    max_num_seqs (Model Runner V2 captures no graph above either). NQ-NOFALLBACK:
    capped at max_num_seqs, so the build refuses only batches the runner captures."""
    compilation = vllm_config.compilation_config
    sizes = compilation.cudagraph_capture_sizes or ()
    cap = compilation.max_cudagraph_capture_size or max(sizes, default=0)
    seqs = vllm_config.scheduler_config.max_num_seqs
    rounded = {-(-s // steps) * steps for s in sizes}
    return sorted(r // steps for r in rounded if r <= cap and r // steps <= seqs)


def enabled(vllm_config):
    extra = vllm_config.additional_config
    backend = extra.get("decode_backend", "auto") if isinstance(extra, dict) else "auto"
    return backend == "latency_specialized"


class DraftFastPath:
    """Per-rank prepared kernels of the drafter; built after the draft weights load."""

    def __init__(
        self, markov=None, batches=(), steps=7, tp=1, all_batches=(), sampled=False,
        strict=False, rank=0,
    ):  # fmt: skip
        self.markov, self.batches, self.steps = markov, frozenset(batches), steps
        # NQ-NOFALLBACK: the target topology (``shapes.target_topology``) refuses
        # instead of drafting a captured graph on the stock loop.
        self.strict = strict
        # Probabilistic drafting: the prepared kernels are the ``sampled`` ones.
        self.sampled = sampled
        # The LM-head GEMV with BF16 logits (cutedsl.verify.lmhead) for FAST_LM_HEAD
        # rows: prepared on the first eager call that sees the (shared) LM head.
        rows = lambda table: frozenset(
            b for b in all_batches if covered(table.get(tp, ""), b * steps, steps)
        )
        self.lm_batches = rows(FAST_LM_HEAD) & self.batches
        self.lm_logits = None
        self.wide_batches = rows(WIDE_LM_HEAD) & self.batches
        self.lm_wide = None
        self.overlap_batches = rows(CONTEXT_OVERLAP)
        # NQ-NOFALLBACK: eager drafter calls (no captured graph) on the stock Markov
        # loop, counted per cause.
        self.stock_steps = StockSteps("DSpark drafter", rank)

    def context_overlap(self, model, num_reqs):
        """CONTEXT_OVERLAP: ``num_reqs`` requests' context step may run on a side
        stream (the sharded context step and the first layer's fast DCP attention,
        which takes the join, are prepared, and its MlaQuery does not poll: fork
        rows, or a polled core's rows)."""
        inner = model.model
        dcp = getattr(inner.layers[0].self_attn, "draft_dcp", None)
        rows = num_reqs * self.steps
        return (
            num_reqs in self.overlap_batches
            and dcp is not None
            and getattr(inner, "_fast_context", None) is not None
            and (rows in dcp.shared.fork_rows or rows in dcp.shared.polled)
        )

    def defer_context_join(self, model, ready):
        """The first layer's attention waits for ``ready`` before its core."""
        model.model.layers[0].self_attn.draft_dcp.shared.context_ready = ready

    @classmethod
    def build(cls, vllm_config, model):
        """None unless the specialized decode is on and this device and deployment
        are covered. ``model``: the K3DSparkForCausalLM (weights loaded)."""
        if not enabled(vllm_config):
            return None
        spec = vllm_config.speculative_config
        if spec is None or spec.method != "dspark":
            return None
        device = torch.device(torch.cuda.current_device())
        if torch.cuda.get_device_capability(device) != (10, 7):
            return None
        from vllm.distributed import get_tp_group

        group = get_tp_group()
        tp, rank = group.world_size, group.rank_in_group
        steps = spec.num_speculative_tokens
        all_batches = draft_batches(vllm_config, steps)
        dcp = vllm_config.parallel_config.decode_context_parallel_size
        # NQ-NOFALLBACK: on the target topology (Rubin TP16 = DCP16) every drafter
        # piece a captured graph runs is covered or the load is refused; elsewhere
        # (out of scope) uncovered pieces keep the stock drafter path, logged.
        strict = target_topology(True, tp, dcp, steps + 1)

        def fail(site, cause):
            if strict:
                refuse(site, cause)
            logger.warning_once(
                "Kimi K3 DSpark drafter: %s keeps stock (%s)", site, cause
            )

        # The open tables' ranges end at the deployment's largest drafter graph.
        top = max(all_batches, default=0)
        table = batch_table(closed(FUSED_MARKOV.get(tp, ""), top))
        settle = batch_table(SETTLED_MARKOV.get(tp, ""))
        head = model.model.markov_head
        processor = model.logits_processor
        # Probabilistic drafting (the stock loop's gumbel_sample) unless the deployment
        # draws FP64 noise or watermarks drafts (the stock loop keeps those).
        sampled = spec.draft_sample_method == "probabilistic"
        batches = []
        shapes = tuple(head.markov_w1.weight.shape), tuple(head.markov_w2.weight.shape)
        # NQ-NOFALLBACK: a drafter the fused Markov kernels do not cover would draft
        # every captured batch on the stock Markov loop: refuse it at load.
        checks = (
            (processor.scale != 1.0, "a logit scale"),
            (processor.soft_cap is not None, "a logit soft cap"),
            (shapes[0] != shapes[1], f"markov_w1 {shapes[0]} != markov_w2 {shapes[1]}"),
            (
                model.config.draft_vocab_size != model.config.vocab_size,
                "a draft vocabulary other than the target's",
            ),
            (
                sampled and vllm_config.model_config.use_fp64_gumbel,
                "FP64 Gumbel draws",
            ),
            (
                sampled and getattr(vllm_config, "watermark_config", None) is not None,
                "watermarked drafts",
            ),
        )
        reason = next((why for failed, why in checks if failed), None)
        batches = [b for b in all_batches if table.get(b)]
        if reason is not None:
            fail("DSpark drafter Markov sampling", reason)
            batches = []
        missing = [b for b in all_batches if b not in batches]
        if missing and reason is None:
            fail("DSpark drafter Markov sampling", f"FUSED_MARKOV misses {missing}")
        if batches:
            from ..cutedsl.draft.markov import device_limits, supports

            if not supports(*shapes, tp, device_limits(device)):
                fail(
                    "DSpark drafter Markov sampling",
                    f"Markov head {shapes[1]} at TP{tp} (rank, vocabulary split or "
                    "co-residency of the fused kernels)",
                )
                batches = []
        markov = None
        if batches:
            from ..cutedsl.draft.markov import MAX_WIDTH, PreparedMarkov

            markov = PreparedMarkov(
                head.markov_w1.weight,
                head.markov_w2.weight,
                tp,
                rank,
                group.device_group,
                steps,
                gated=[b for b in batches if table.get(b) in (2, 3)],
                staged=[b for b in batches if table.get(b) == 3],
                settled=[b for b in batches if settle.get(b)],
                wide=max(batches) > MAX_WIDTH,
            )
            # (wide batches need one launch of at least 8 slots: the device's shared
            # memory at this Markov rank; markov_wide.launch_width)
            uncovered = [b for b in batches if not markov.covers(b, sampled)]
            if uncovered:  # NQ-NOFALLBACK
                fail(
                    "DSpark drafter Markov sampling",
                    f"no MarkovWide launch fits this device at Markov rank "
                    f"{markov.markov_rank} for draft batches {tuple(uncovered)}",
                )
                batches = [b for b in batches if b not in uncovered]
            for batch in batches:
                markov.prepare(batch, steps, steps, sampled)
            if not batches:
                markov = None
        context = [
            b * (steps + 1)
            for b in all_batches
            if covered(
                closed(FAST_CONTEXT.get(tp, ""), top * (steps + 1)),
                b * (steps + 1),
                steps,
            )
        ]
        if context and model.model.context_proj.weight.shape[0] % tp:
            # NQ-NOFALLBACK: the stock replicated context step for every row.
            fail(
                "DSpark drafter context step",
                f"context_proj rows {model.model.context_proj.weight.shape[0]} not "
                f"divisible by TP{tp}",
            )
            context = []
        if context:
            from ..cutedsl.draft.context import PreparedContext

            prepared = PreparedContext(
                model.model, group.device_group, tp, rank, context
            )
            rotary = model.model.layers[0].self_attn.rotary_emb
            prepared.prepare_kv(model.model, rotary.cos_sin_cache.shape[0])
            model.model._fast_context = prepared
            prepared.strict = strict  # NQ-NOFALLBACK
            logger.info_once(
                "Kimi K3 DSpark drafter: sharded context step for %s rows "
                "(persistent row gather at %s)",
                (*prepared.rows, *prepared.wide),
                prepared.persistent,
            )
            log_coverage(
                rank,
                "draft context",
                f"sharded context step at {len(context)} row counts; context shard "
                "GEMM stock cuBLAS above 64 rows (no split-K tactic; split-K loses at "
                "56-64 rows, scripts/dsdraft/gemm_sweep.py)",
            )
        query_rows = [b * steps for b in all_batches]
        context_rows = [b * (steps + 1) for b in all_batches]
        installed = install_projections(model, tp, query_rows, context_rows, rank)
        if installed:
            logger.info_once(
                "Kimi K3 DSpark drafter: split-K projections on %d linears", installed
            )
        top_rows = top * steps  # (the open query-row ranges end here)
        policy = lambda table, r: covered(closed(table.get(tp, ""), top_rows), r, steps)
        ar_rows = [r for r in query_rows if policy(FAST_AR, r)]
        two_shot = {r for r in ar_rows if policy(FAST_AR, r) == 2}
        gated = {r for r in ar_rows if policy(GATED_AR, r)}
        early = {r for r in ar_rows if policy(TWO_SHOT_EARLY, r)}
        valid = {r for r in ar_rows if policy(VALID_AR, r)}
        wide = {r for r in ar_rows if policy(FAST_AR, r) == 3}
        if strict and query_rows and not ar_rows:  # NQ-NOFALLBACK
            fail("DSpark drafter all-reduce sites", "FAST_AR lists no query rows")
        if ar_rows:
            from ..cutedsl.draft.allreduce import PreparedDraftAr, plain_layer

            vocab = model.config.vocab_size
            prepared_ar = PreparedDraftAr(
                group.device_group,
                device,
                tp,
                rank,
                ar_rows,
                vocab,
                two_shot,
                gated,
                early,
                valid,
                wide,
            )
            prepared_ar.strict = strict  # NQ-NOFALLBACK
            unserved = sorted(set(query_rows) - set(prepared_ar.rows))
            if strict and unserved:
                fail(
                    "DSpark drafter all-reduce sites",
                    f"query rows {unserved} have no Lamport sites (FAST_AR / CAP)",
                )
            if prepared_ar.rows:
                rings = RING_PROJECTIONS.get(tp, {})
                tactics = {r: rings for r in prepared_ar.rows if r <= PROJECTION_ROWS}
                layers = model.model.layers
                # The one-shot rows' ring producers need both o_proj's and down_proj's
                # shapes in RING_PROJECTIONS (else their sites keep stored partials).
                ring_shapes = {
                    tuple(layers[0].self_attn.o_proj.weight.shape),
                    tuple(layers[0].mlp.down_proj.weight.shape),
                }
                if tactics and not ring_shapes <= set(rings):
                    logger.warning_once(
                        "Kimi K3 DSpark drafter: no ring projection for %s at TP%d: "
                        "rows %s keep stored partials (no polled sites)",
                        tuple(sorted(ring_shapes - set(rings))),
                        tp,
                        tuple(sorted(tactics)),
                    )
                deferred = {r for r in prepared_ar.valid_ar if policy(SITE_DEFERRED, r)}
                prepared_ar.prepare_producers(layers, tactics, deferred, trigger=True)
                from ..cutedsl.draft.producer import ROWS, TwoShotProducer

                # Rows outside the producers' (e.g. 6 or 8 rows of a DL3 / DL4 B2 graph)
                # keep the sites' own two-shot push.
                pushed = sorted(
                    r
                    for r in prepared_ar.two_rows
                    if policy(TWO_SHOT_PRODUCERS, r) and r in ROWS
                )
                if pushed and all(plain_layer(layer) for layer in layers):
                    producers = [
                        tuple(
                            TwoShotProducer(weight, pushed, tp)
                            for weight in (
                                layer.self_attn.o_proj.weight,
                                layer.mlp.down_proj.weight,
                            )
                        )
                        for layer in layers
                    ]
                    prepared_ar.set_two_shot_producers(producers, pushed)
                    shape = tuple(layers[0].mlp.gate_up_proj.weight.shape)
                    fused = {}
                    for r in pushed:
                        top = next(
                            (t for t, _ in PROJECTIONS_WIDE.get(tp, ()) if r <= t), 0
                        )
                        tactic = (
                            dict(PROJECTIONS_WIDE.get(tp, ())).get(top, {}).get(shape)
                        )
                        if policy(TWO_SHOT_SILU, r) and tactic is not None:
                            fused[r] = tactic
                    prepared_ar.prepare_two_gate_up(layers, fused)
                model.model._fast_ar = prepared_ar
                logger.info_once(
                    "Kimi K3 DSpark drafter: Lamport all-reduce sites for %s rows "
                    "(o_proj / down_proj publishing their partials for %s; sites "
                    "deferred past their reads for %s)",
                    tuple(prepared_ar.rows),
                    tuple(sorted(prepared_ar.polled)),
                    tuple(sorted(r for r in deferred if r in prepared_ar.polled)),
                )
        attentions = [layer.self_attn for layer in model.model.layers]
        rows = [r for r in query_rows if policy(FAST_DCP, r)]
        if strict and rows != query_rows:
            fail("DSpark drafter attention", "FAST_DCP misses query rows")
        if dcp == tp > 1 and rows and any(a.dcp_world_size != tp for a in attentions):
            # NQ-NOFALLBACK: the drafter's MLA layers would keep the stock DCP
            # attention at every row.
            fail(
                "DSpark drafter attention",
                f"drafter MLA layers with DCP sizes "
                f"{sorted({a.dcp_world_size for a in attentions})} under DCP{dcp}",
            )
        if dcp == tp > 1 and rows and all(a.dcp_world_size == tp for a in attentions):
            from ..cutedsl.draft.attention import DraftDcp, query_heads

            heads = attentions[0].num_local_heads
            if not query_heads(heads):
                # (MlaQuery's drafter tile is keyed by heads per rank: mla.query
                # DRAFT_TILED_FROM) NQ-NOFALLBACK
                fail(
                    "DSpark drafter attention",
                    f"no fused drafter query for {heads} heads per rank",
                )
            else:
                # The drafter's mask as its MLA layers run it: the K3 DSpark layers
                # are built non_causal_multi_token_decode (dspark_mla), which the
                # stock MLA metadata builder reads (any layer non-causal: the batch
                # is), whatever the checkpoint's dflash_config says (both
                # kimi-k3-dspark and kimi-k3.1-pro-dspark-preview run non-causal).
                causal = not any(
                    getattr(a, "non_causal_multi_token_decode", False)
                    for a in attentions
                )
                prepared_dcp = DraftDcp(
                    attentions, group.device_group, tp, rank, rows, steps,
                    vllm_config.model_config.max_model_len,
                    vllm_config.parallel_config.cp_kv_cache_interleave_size,
                    causal,
                )  # fmt: skip
                prepared_dcp.strict = strict  # NQ-NOFALLBACK
                served = (*prepared_dcp.rows, *prepared_dcp.wide)
                logger.info_once(
                    "Kimi K3 DSpark drafter: Lamport DCP attention (%s) for %s rows",
                    "causal" if causal else "non-causal",
                    tuple(served),
                )
                # (per cause: DraftDcp.dropped) NQ-NOFALLBACK: every drafter row the
                # deployment captures (capped at max_num_seqs) is served or refused.
                reasons: dict[str, list[int]] = {}
                for r in rows:
                    if r not in served:
                        reason = prepared_dcp.dropped.get(r, "not covered")
                        reasons.setdefault(reason, []).append(r)
                for reason, dropped in reasons.items():
                    fail(
                        "DSpark drafter attention",
                        f"rows {tuple(dropped)} would keep the stock DCP attention "
                        f"({reason})",
                    )
                log_coverage(rank, "draft attention", prepared_dcp.coverage(rows))
        logger.info_once(
            "Kimi K3 DSpark drafter: fused %s Markov sampling for draft batches %s",
            "probabilistic" if sampled else "greedy",
            tuple(batches),  # info_once keys on its (hashable) arguments
        )
        fast = cls(markov, batches, steps, tp, all_batches, sampled, strict, rank)
        lm = sorted(b * steps for b in fast.lm_batches)
        wide_lm = sorted(b * steps for b in fast.wide_batches)
        stock_lm = sorted(
            b * steps
            for b in batches
            if b not in fast.lm_batches and b not in fast.wide_batches
        )
        log_coverage(
            rank,
            "draft LM head",
            f"GEMV rows {lm}, evict-first GEMM rows {wide_lm}, stock cuBLAS shard "
            f"GEMM rows {stock_lm} (deliberate: rows between the 1-7 GEMV and the "
            "evict-first 10-64 kernel (q4 B3: 9), whose lmhead.WIDE_ROWS note reads "
            "'9 rows: 35 us' and shows no 9-row win; above 64 rows unmeasured vs "
            "the evict-first GEMM; NQ-TUNE)",
        )
        return fast

    def greedy_markov(self, model, hidden, anchors, draft, sampling=None):
        """Draft tokens of every position into ``draft`` [batch, steps] from the
        backbone rows ``hidden`` [batch * steps, H] (request-major) and the anchor ids
        [batch]; False (nothing launched) when this call is not covered. ``sampling``:
        probabilistic drafting's ``(temperature, seeds, slots, positions, cache)``
        (``PreparedMarkov.sampling_args``), only with the ``sampled`` kernels."""
        markov = self.markov
        batch = anchors.shape[0]
        sampled = sampling is not None
        if not torch.cuda.is_current_stream_capturing():
            self.prepare_lm_head(model)
            # An eager drafter call (no captured graph: warmup, a step above the
            # captured sizes): the stock Markov loop, a run-time property counted
            # per cause (NQ-NOFALLBACK).
            self.stock_steps.count("eager drafter call", batch)
            return False
        checks = (
            (markov is None, "no fused Markov kernels"),
            (sampled != self.sampled, f"a {'sampled' if sampled else 'greedy'} call"),
            (batch not in self.batches, "a batch without prepared Markov kernels"),
            (hidden.shape[0] != batch * self.steps, f"{hidden.shape[0]} hidden rows"),
            (draft.shape != (batch, self.steps), f"draft {tuple(draft.shape)}"),
            (anchors.dtype != torch.int32, f"anchors {anchors.dtype}"),
            (draft.dtype != torch.int64, f"draft {draft.dtype}"),
            (draft.stride(1) != 1, "a strided draft row"),
        )
        reason = next((why for failed, why in checks if failed), None)
        if reason is None and not markov.ready(
            batch, anchors.stride(0), draft.stride(0), sampled
        ):
            reason = "Markov kernels not ready for these strides"
        if reason is not None:
            # NQ-NOFALLBACK: a captured drafter graph never captures the stock loop.
            if self.strict and full_capture():
                refuse("DSpark drafter Markov sampling", reason, Shape(batch))
            return False
        if sampled:
            try:
                sampling = markov.sampling_args(*sampling)
            except ValueError as error:  # NQ-NOFALLBACK: no swallowed fallback
                if self.strict:
                    refuse("DSpark drafter Markov sampling", str(error), Shape(batch))
                return False
        processor, lm_head = model.logits_processor, model.lm_head
        lm = self.lm_logits
        rows = hidden if lm is None or lm.takes(hidden) else hidden.contiguous()
        wide = self.lm_wide
        if lm is not None and batch in self.lm_batches and lm.takes(rows):
            base = lm.run(rows)
        elif wide is not None and batch in self.wide_batches and wide.takes(rows):
            base = wide.run(rows)
        elif self.strict and (batch in self.lm_batches or batch in self.wide_batches):
            # NQ-NOFALLBACK: a batch a fast LM head is listed for, not prepared.
            refuse(
                "DSpark drafter LM head",
                "its fast LM-head kernel is not prepared or does not take the rows",
                Shape(batch * self.steps, self.steps),
            )
        else:
            # The stock LM-head GEMM of the drafter, without its all-gather: this
            # rank's vocabulary shard of the base logits, bitwise as stock.
            base = processor._get_logits(hidden, lm_head, None, skip_gather=True)
        markov.run(base, anchors, draft, sampling)
        return True

    def gather_draft_logits(self, cache, slots):
        """Complete the draft-logits cache rows of ``slots`` [n] (request-state indices)
        from every rank's columns (the ``sampled`` kernels write only their own), for
        the stock rejection sampler. Eager; every rank calls it together."""
        from vllm.distributed import get_tp_group

        markov = self.markov
        if markov is None or not self.sampled or slots.numel() == 0:
            return
        lo = markov.rank * markov.vocab_rank
        slots = slots.long()
        own = cache[slots, :, lo : lo + markov.vocab_rank].contiguous()
        cache[slots] = get_tp_group().all_gather(own, dim=-1)

    def prepare_lm_head(self, model):
        """The LM-head GEMV of FAST_LM_HEAD batches (it reads the shard in place),
        once the LM head (shared with the target after loading) is there; outside
        graph captures (it compiles)."""
        if (
            self.lm_logits is not None
            or self.lm_wide is not None
            or self.markov is None
        ):
            return
        if not self.lm_batches and not self.wide_batches:
            return
        lm_head = getattr(model, "lm_head", None)
        weight = getattr(lm_head, "weight", None)
        if weight is None or weight.dim() != 2:
            return
        from ..cutedsl.verify import lmhead

        weight = weight[: self.markov.vocab_rank]
        if not lmhead.supports(weight):  # NQ-NOFALLBACK
            if self.strict:
                refuse(
                    "DSpark drafter LM head",
                    f"shard {tuple(weight.shape)} {weight.dtype} is not the BF16, "
                    "16-row-aligned contiguous shard the LM-head kernels read",
                )
            self.lm_batches = self.wide_batches = frozenset()
            return
        if self.lm_batches:
            self.lm_logits = lmhead.PreparedLmHeadLogits(weight, weight.device)
        if self.wide_batches:
            self.lm_wide = lmhead.PreparedWideLmHead(weight, weight.device)
        logger.info_once(
            "Kimi K3 DSpark drafter: LM-head GEMV for draft batches %s, evict-first "
            "LM-head GEMM for %s",
            tuple(sorted(self.lm_batches)),
            tuple(sorted(self.wide_batches)),
        )
