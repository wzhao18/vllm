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
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import gumbel_noised_argmax, spec_step_seed

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
# split, BF16 logits bitwise equal to cuBLAS's unsplit kernels), per TP, on Rubin:
# 17-32 rows, where cuBLAS splits K (lmhead.WIDE_ROWS' note). Every other row count
# above FAST_LM_HEAD's runs the stock cuBLAS shard GEMM, chosen per shape here and
# logged in the drafter's coverage (``DraftFastPath.build``), from the VR200 kernel
# bench (nqbench kern-draft gemm/lm_head, 1 GPU, [vocab / 16, 7168] shard, us,
# evict-first kernel vs cuBLAS):
#   9-16 rows: 12 / 14 / 15 rows (q4 B4, q8 B2, q4 B5) 14.84 / 14.90 / 14.91 vs 14.60 /
#     14.59 / 14.60 (1.016-1.021x; was the evict-first kernel at 10-16); 9 rows (q4 B3)
#     cuBLAS 14.55 as before;
#   17-32 rows: 18-30 rows 14.52-14.92 vs 16.12-16.37 (0.887-0.916x; kept);
#   33-64 rows: 35-63 rows (q8 B5-B9, q4 B12 / B16 / B20) 14.93-15.21 vs 13.51-13.70
#     (1.099-1.115x: cuBLAS stops splitting K above 32 rows; was the evict-first
#     kernel); the split-K sweep's best there (64.64.1.7, 15.04-15.31) loses too.
# Above 64 rows cuBLAS as before. NQ-TUNE: 9-16 rows, the sweep's split-K 64.16.1.8 is
# 14.23-14.30 us vs cuBLAS 14.55-14.60 (-2%, an FP32 K-order reorder: no K split): a
# drafter LM head on that projection tactic is not wired.
WIDE_LM_HEAD = {16: "17-32:1"}
# Query rows whose LM-head shard GEMM is the die-local persistent stream-K token-wide
# GEMM (cutedsl.gemm.wide_sk: the shard's two halves homed on the two dies, one CTA
# pair per SM pair, all rows on the MMA's N), per TP, for the release drafter
# (GqaDraftFastPath; the SP decode declares the preview drafter's LM head stock,
# sp_attention.SP_DRAFTER_TABLES). TP4 (D3-32, [40960, 7168]
# shard), VR200 1 GPU, cold weights, us, wide_sk (two TMEM accumulators) vs the
# pinned cuBLASLt algorithm: 96 rows 39.3 vs 47.0, 192 rows 46.2 vs 49.9, 288 rows
# 65.3 vs 66.8 (bench_kimi_k3_sdec_wide_sk.py).
GQA_SK_LM_HEAD = {4: "96-96:1,192-192:1,288-288:1"}
# Batches whose Markov bias GEMM ([B, 512] x [vocab / TP, 512]^T, three per step) is
# that kernel too, per TP. TP4 [40960, 512] shard, VR200 1 GPU, cold weights, us,
# wide_sk vs F.linear: B32 6.4 vs 11.8 (B64 7.0 vs 7.1, B96 7.5 vs 7.0: cuBLAS).
GQA_SK_MARKOV = {4: (32,)}
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
# TP4 = DCP4 (C512's sequence-parallel decode role, ``runtime.sp_topology``): 24 heads
# per rank, publish-only MlaQuery and polled cores at every row; the drafter's only
# specialized piece there (the rest stays the stock module: sp_attention
# SP_DRAFTER_TABLES), strict (a served row runs it or the load is refused). Draft eff
# sweep (bench_draft_eff.py, 4 x VR200, 24 heads per rank, AgentX lengths, DCP4 local
# = total / 4; dreff7 pw4, arms interleaved, mean of two), us per drafter attention
# layer, DL3, stock DCP path -> DraftDcp (its DCP4 cake rule, mla.dcp
# DRAFT_LONG_SPLIT): C512's 120 / 144 / 168 rows 1027 / 1109 / 1148 -> 320 / 385 / 488
# (its uniform-length time 343 / 408 / 470); the former stock-core rows 54 / 57 / 63
# 842 / 844 / 848 -> 211 / 217 / 184. The whole layer (norms, q_b, W_UK, RoPE, FP8,
# KV insert, attention, W_UV; bench_c512_layer.py, dreff7): 1088 / 1160 / 1250 ->
# 341 / 407 / 503.
FAST_DCP = {4: "1-:1", 8: "1-64:1", 16: "1-:1"}
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


def check_draft_formats(model):
    """The drafter weights the specialized drafter reads raw: the Markov head's two
    BF16 tables (the fused Markov kernels). (Its split-K projections refuse non-BF16
    weights in ``PreparedProjection``; the context projection in ``PreparedContext``.)
    Refuses at build instead of programming other bytes as BF16."""
    head = getattr(model.model, "markov_head", None)
    for name in ("markov_w1", "markov_w2"):
        table = getattr(getattr(head, name, None), "weight", None)
        if table is not None and table.dtype != torch.bfloat16:
            raise ValueError(
                f"Kimi K3 specialized drafter needs a BF16 Markov head, got {name} "
                f"{table.dtype}"
            )


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
        # The die-local stream-K GEMM's rows (the release drafter's GQA_SK_LM_HEAD),
        # prepared with the LM head.
        self.sk_batches = frozenset()
        self.lm_sk = None
        # The stock LM-head GEMM's rows on pinned cuBLASLt algorithms (lt_pin), for
        # the batches no fast LM head serves: prepared with the LM head.
        self.lm_pins = None
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

        check_draft_formats(model)
        group = get_tp_group()
        tp, rank = group.world_size, group.rank_in_group
        steps = spec.num_speculative_tokens
        all_batches = draft_batches(vllm_config, steps)
        dcp = vllm_config.parallel_config.decode_context_parallel_size
        # NQ-NOFALLBACK: on the target topology (Rubin TP16 = DCP16) every drafter
        # piece a captured graph runs is covered or the load is refused; elsewhere
        # (out of scope) uncovered pieces keep the stock drafter path, logged.
        strict = target_topology(True, tp, dcp, steps + 1)
        # C512's SP decode role (TP4 = DCP4): the drafter's attention is specialized
        # and strict; its other pieces are the declared stock module (sp_attention).
        from . import sp_topology

        attention_strict = strict or sp_topology(vllm_config)

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

        def fail_attention(cause):
            if attention_strict:
                refuse("DSpark drafter attention", cause)
            fail("DSpark drafter attention", cause)

        if attention_strict and rows != query_rows:
            fail_attention("FAST_DCP misses query rows")
        if dcp == tp > 1 and rows and any(a.dcp_world_size != tp for a in attentions):
            # NQ-NOFALLBACK: the drafter's MLA layers would keep the stock DCP
            # attention at every row.
            fail_attention(
                f"drafter MLA layers with DCP sizes "
                f"{sorted({a.dcp_world_size for a in attentions})} under DCP{dcp}",
            )
        if dcp == tp > 1 and rows and all(a.dcp_world_size == tp for a in attentions):
            from ..cutedsl.draft.attention import DraftDcp, query_heads

            heads = attentions[0].num_local_heads
            if not query_heads(heads):
                # (MlaQuery's drafter tile is keyed by heads per rank: mla.query
                # DRAFT_TILED_FROM) NQ-NOFALLBACK
                fail_attention(f"no fused drafter query for {heads} heads per rank")
            else:
                # The drafter's mask exactly as stock decides it: the DSpark speculator
                # builds its metadata causal unless the checkpoint asks for non-causal
                # (dflash_has_any_non_causal over the draft hf config: is_causal /
                # dflash_config.causal / layer_types). The layers' own
                # non_causal_multi_token_decode flag only permits a non-causal batch,
                # it does not make one (kimi-k3.1-pro-dspark-preview: causal true).
                from vllm.model_executor.models.qwen3_dflash import (
                    dflash_has_any_non_causal,
                )

                draft_cfg = vllm_config.speculative_config.draft_model_config
                causal = not dflash_has_any_non_causal(draft_cfg.hf_config)
                prepared_dcp = DraftDcp(
                    attentions, group.device_group, tp, rank, rows, steps,
                    vllm_config.model_config.max_model_len,
                    vllm_config.parallel_config.cp_kv_cache_interleave_size,
                    causal,
                )  # fmt: skip
                prepared_dcp.strict = attention_strict  # NQ-NOFALLBACK
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
                    fail_attention(
                        f"rows {tuple(dropped)} would keep the stock DCP attention "
                        f"({reason})"
                    )
                # NQ-NOFALLBACK: a served row whose core would be the stock tokenspeed
                # core inside the fast path (no fold, wide or rows-mode core).
                seams = prepared_dcp.stock_core_rows()
                if seams and attention_strict:
                    fail_attention(
                        f"rows {seams} would run the stock tokenspeed core ("
                        + "; ".join(prepared_dcp.core_of(r) for r in seams[:2])
                        + ")"
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
        sk_lm = sorted(b * steps for b in fast.sk_batches)
        stock_lm = sorted(
            b * steps
            for b in batches
            if b not in fast.lm_batches | fast.wide_batches | fast.sk_batches
        )
        log_coverage(
            rank,
            "draft LM head",
            f"GEMV rows {lm}, evict-first GEMM rows {wide_lm}, die-local stream-K "
            f"GEMM rows {sk_lm}, stock cuBLAS shard "
            f"GEMM rows {stock_lm} (policy WIDE_LM_HEAD: cuBLAS is measured faster "
            "than the evict-first GEMM at 9-16 rows (1.016-1.021x) and 33-64 rows "
            "(1.099-1.115x, no K split there); the evict-first GEMM only at 17-32 "
            "rows, where cuBLAS splits K; above 64 rows cuBLAS)",
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
            self.prepare_lm_sk(model)
            self.prepare_lm_pins(model)
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
        elif batch in self.sk_batches and self.lm_sk is not None and self.lm_sk.takes(rows):
            # The stock LM-head GEMM on the die-local stream-K kernel: FP32
            # accumulation (K order may differ), one BF16 rounding.
            base = rows.new_empty((rows.shape[0], self.lm_sk.n))
            self.lm_sk.run(rows, base)
        elif self.strict and batch in self.sk_batches:
            # NQ-NOFALLBACK: a GQA_SK_LM_HEAD batch whose kernel is not prepared.
            refuse(
                "DSpark drafter LM head",
                "its die-local stream-K kernel is not prepared or does not take the rows",
                Shape(batch * self.steps, self.steps),
            )
        elif self.lm_pins is not None and self.lm_pins.takes(hidden):
            # The stock LM-head GEMM (this rank's vocabulary shard of the base
            # logits) on a pinned cuBLASLt algorithm: the same FP32 accumulation.
            base = hidden.new_empty((hidden.shape[0], self.lm_pins.weight.shape[0]))
            self.lm_pins.run(hidden, base)
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

    def prepare_lm_pins(self, model):
        """Pinned cuBLASLt algorithms for the stock LM-head GEMM's rows (the batches
        without a fast LM head), once the (shared) LM head is there."""
        if self.lm_pins is not None or self.markov is None:
            return
        weight = getattr(getattr(model, "lm_head", None), "weight", None)
        vocab_rank = getattr(self.markov, "vocab_rank", None)
        if (
            weight is None
            or weight.dim() != 2
            or weight.shape[0] != vocab_rank
            or weight.dtype != torch.bfloat16
            or not weight.is_contiguous()
            or getattr(model.lm_head, "bias", None) is not None
            or model.logits_processor.scale != 1.0
            or model.logits_processor.soft_cap is not None
        ):
            return
        from ..lt_pin import pinned

        fast = self.lm_batches | self.wide_batches | self.sk_batches
        rows = [b * self.steps for b in sorted(self.batches - fast)]
        self.lm_pins = pinned(weight, rows)

    def prepare_lm_sk(self, model):
        """The die-local stream-K LM-head GEMM of GQA_SK_LM_HEAD batches, once the
        (shared) LM head is there; outside graph captures (it compiles and places the
        shard)."""
        if self.lm_sk is not None or self.markov is None or not self.sk_batches:
            return
        weight = getattr(getattr(model, "lm_head", None), "weight", None)
        vocab_rank = getattr(self.markov, "vocab_rank", None)
        from ..cutedsl.gemm import wide_sk

        rows = sorted(b * self.steps for b in self.sk_batches)
        ok = (
            weight is not None
            and weight.dim() == 2
            and weight.shape[0] >= vocab_rank
            and weight.dtype == torch.bfloat16
            and getattr(model.lm_head, "bias", None) is None
            and model.logits_processor.scale == 1.0
            and model.logits_processor.soft_cap is None
        )
        if ok:
            _, pairs = wide_sk.die_pairs(weight.device)
            ok = all(wide_sk.supports(r, vocab_rank, weight.shape[1], pairs) for r in rows)
        if not ok:  # NQ-NOFALLBACK
            if self.strict:
                refuse(
                    "DSpark drafter LM head",
                    "the shard is not the BF16 [vocab / TP, H] shard (no bias, scale 1, "
                    "no soft cap) at rows the die-local stream-K kernel takes",
                )
            self.sk_batches = frozenset()
            return
        self.lm_sk = wide_sk.PreparedWideSK(weight[:vocab_rank], rows).prepare()
        logger.info_once(
            "Kimi K3 DSpark drafter: die-local stream-K LM-head GEMM for draft rows %s",
            tuple(rows),
        )

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


# ---------------------------------------------------------------------------------
# The release DSpark drafter (Inferact/Kimi-K3.1-Pro-DSpark: dspark_gqa
# K3GqaDSparkForCausalLM, five sliding-window GQA layers of 96 query / 16 KV heads of
# 64, a 35840 -> 7168 context fc, Markov rank 512, the target's embedding and LM
# head; draft KV TP-sharded, not DCP-sharded). Its pieces are the preview drafter's
# kernels on the GQA chain: the fused Markov sampling and LM head (unchanged shapes),
# the sharded context step (fc shard + CtxGather with gamma = hidden_norm; the same
# (448, 35840) shard at TP16), split-K projections for the GQA shapes, the Lamport
# all-reduce + norm sites with ring / two-shot o_proj and down_proj producers, and
# the fused GQA attention (cutedsl.draft.gqa_*, ``install_gqa_attention``).
#
# Strict (NQ-NOFALLBACK) on the target topology (Rubin TP16 = DCP16) and on C512's
# SP decode role (TP4 = DCP4, ``runtime.sp_topology``): every captured drafter batch
# runs every piece or the build refuses. Per-rank linear shapes (N, K):
#
#     linear     TP16           TP8            TP4
#     qkv        (512, 7168)    (1024, 7168)   (2048, 7168)
#     o_proj     (7168, 384)    (7168, 768)    (7168, 1536)
#     gate_up    (1792, 7168)   (3584, 7168)   (7168, 7168)
#     down       (7168, 896)    (7168, 1792)   (7168, 3584)
#     ctx shard  (448, 35840)   (896, 35840)   (1792, 35840)
#     ctx K/V    (640, 7168)    (1280, 7168)   (2560, 7168)
#
# At TP16 the row-keyed Markov / LM-head / context / site tables are the preview
# drafter's (the same kernels at the same rows; NQ-TUNE: re-time the chain-level
# site variants, SITE_DEFERRED / TWO_SHOT_EARLY, on the GQA chain at 16 ranks).

GQA_HEADS, GQA_KV_HEADS, GQA_HEAD_DIM = 96, 16, 64
GQA_HIDDEN, GQA_FFN, GQA_LAYERS, GQA_AUX = 7168, 14336, 5, 5


def gqa_shapes(tp):
    """The release drafter's per-rank (N, K) by linear at ``tp``."""
    q = GQA_HEADS // tp * GQA_HEAD_DIM
    kv = max(1, GQA_KV_HEADS // tp) * GQA_HEAD_DIM
    return {
        "qkv": (q + 2 * kv, GQA_HIDDEN),
        "o_proj": (GQA_HIDDEN, q),
        "gate_up": (2 * GQA_FFN // tp, GQA_HIDDEN),
        "down": (GQA_HIDDEN, GQA_FFN // tp),
        "ctx_shard": (GQA_HIDDEN // tp, GQA_AUX * GQA_HIDDEN),
        "ctx_kv": (GQA_LAYERS * 2 * kv, GQA_HIDDEN),
    }


# Drafter GEMMs of at most PROJECTION_ROWS rows (B1 at k7, B2 at k3): split-K tactics
# by (N, K) (cutedsl.gemm.projection; an FP32 accumulation reorder), per TP, on Rubin.
# The (1792, 7168) / (7168, 896) / (448, 35840) entries are the preview drafter's
# measured ones (PROJECTIONS: the same shapes); the GQA shapes from the release GEMM
# bench (benchmarks/kernels/k31_vr/dsrel/bench_draft_gemm.py, 1 GPU, cold weights).
# 1 GPU, cold weights, geometric mean over the bin's drafter rows, us (cuBLAS ->
# split-K): TP16 qkv (512, 7168) 6-7 rows 4.34 -> 2.19, o_proj (7168, 384) 2.72 ->
# 1.99, context K/V (640, 7168) 8 rows 5.07 -> 2.19.
GQA_PROJECTIONS = {
    16: {
        (512, 7168): (64, 8, 7, 8),
        (7168, 384): (64, 8, 1, 3),
        (640, 7168): (64, 8, 8, 7),
        (1792, 7168): PROJECTIONS[16][(1792, 7168)],
        (7168, 896): PROJECTIONS[16][(7168, 896)],
        (448, 35840): PROJECTIONS[16][(448, 35840)],
    },
    # TP4 (C512's SP decode role, DL3: 6 query / 8 context rows): qkv 6.21 -> 3.80,
    # o_proj 3.91 -> 3.41, gate_up 11.87 -> 10.16, down 7.47 -> 5.99, context K/V
    # 6.42 -> 4.21.
    4: {
        (2048, 7168): (64, 8, 4, 8),
        (7168, 1536): (64, 8, 1, 8),
        (7168, 7168): (64, 8, 1, 8),
        (7168, 3584): (64, 8, 1, 8),
        (2560, 7168): (64, 8, 4, 8),
    },
}
# Drafter GEMMs of 9-64 rows by row bin (largest rows, tactics by (N, K)), as
# PROJECTIONS_WIDE; above the last bin cuBLAS (measured: NQ-NOFALLBACK coverage log).
# TP16 (release GEMM bench, us cuBLAS -> split-K): qkv 9-15 rows 4.14 -> 2.42, 18-30
# 3.99 -> 2.87, 33-63 4.26 -> 3.67; o_proj 9-15 2.28 -> 1.90, 18-30 2.25 -> 2.06
# (33-63: cuBLAS 2.37 vs 2.62, kept); context K/V 12-16 4.68 -> 2.65, 20-32 4.11 ->
# 2.92, 36-64 4.25 -> 3.70; the context fc shard (448, 35840; the preview's tables
# have no tactic above 8 rows) 12-16 6.90 -> 5.38, 20-32 6.68 -> 6.21 (N tiles of
# 16: published into the row mailbox by the GEMM, ``gqa_context_projection``;
# 36 rows: cuBLAS 6.74 vs 8.28, kept).
GQA_PROJECTIONS_WIDE = {
    16: (
        (
            16,
            {
                (512, 7168): (64, 8, 8, 7),
                (7168, 384): (64, 16, 1, 3),
                (640, 7168): (64, 8, 8, 7),
                (448, 35840): (64, 16, 14, 8),
                (1792, 7168): (64, 16, 4, 8),
                (7168, 896): (64, 16, 1, 8),
            },
        ),
        (
            32,
            {
                (512, 7168): (64, 16, 4, 10),
                (7168, 384): (64, 32, 1, 3),
                (640, 7168): (64, 16, 4, 10),
                (448, 35840): (64, 16, 10, 8),
                (1792, 7168): (64, 32, 4, 6),
            },
        ),
        (
            64,
            {
                (512, 7168): (128, 64, 6, 14, 14),
                (640, 7168): (128, 64, 6, 14, 14),
                (1792, 7168): (128, 64, 6, 14, 14),
            },
        ),
    ),
    # TP4 (us cuBLAS -> split-K): qkv 9-15 6.16 -> 3.92, 18-30 5.76 -> 4.55, 33-63
    # 6.59 -> 5.20; o_proj 9-15 3.93 -> 3.48, 18-30 4.06 -> 4.01 (33-63: cuBLAS 4.20
    # vs 4.94, kept); gate_up 9-15 12.11 -> 10.16, 18-30 12.61 -> 10.67, 33-63 13.74
    # -> 12.42; down 9-15 7.96 -> 5.99, 18-30 7.70 -> 6.62, 33-63 8.71 -> 7.57;
    # context K/V 12-16 6.44 -> 4.28, 20-32 6.73 -> 4.95, 36-64 7.47 -> 5.73. Above
    # 64 rows cuBLAS (the M64 split-K loses there: TP16 qkv at 70 rows 6.49 vs 4.21).
    4: (
        (
            16,
            {
                (2048, 7168): (64, 16, 4, 8),
                (7168, 1536): (64, 16, 1, 8),
                (7168, 7168): (64, 16, 1, 8),
                (7168, 3584): (64, 16, 1, 8),
                (2560, 7168): (64, 16, 4, 8),
            },
        ),
        (
            32,
            {
                (2048, 7168): (64, 32, 4, 8),
                (7168, 1536): (64, 32, 1, 8),
                (7168, 7168): (64, 32, 1, 8),
                (7168, 3584): (64, 32, 1, 8),
                (2560, 7168): (64, 32, 4, 8),
            },
        ),
        (
            64,
            {
                (2048, 7168): (128, 64, 6, 16, 14),
                (7168, 7168): (64, 64, 1, 6),
                (7168, 3584): (64, 64, 1, 6),
                (2560, 7168): (128, 64, 6, 16, 14),
            },
        ),
    ),
}
# One-shot site rows' ring o_proj / down_proj producers and their gate/up (+ SiLU)
# tactics (RING_PROJECTIONS' rule: every K tile staged before the PDL wait; o_proj's
# 3 K tiles, the sweep's best at 6-7 rows too).
GQA_RING_PROJECTIONS = {
    16: {
        (7168, 384): (64, 8, 1, 3),
        (7168, 896): RING_PROJECTIONS[16][(7168, 896)],
        (1792, 7168): RING_PROJECTIONS[16][(1792, 7168)],
    },
}
# Markov / LM head / context step / sites: the preview drafter's tables at TP8 and
# TP16 (the same kernels and rows).
GQA_FUSED_MARKOV = dict(FUSED_MARKOV)
GQA_SETTLED_MARKOV = dict(SETTLED_MARKOV)
GQA_FAST_LM_HEAD = dict(FAST_LM_HEAD)
GQA_WIDE_LM_HEAD = dict(WIDE_LM_HEAD)
GQA_FAST_CONTEXT = dict(FAST_CONTEXT)
GQA_FAST_AR = dict(FAST_AR)
GQA_GATED_AR = dict(GATED_AR)
GQA_SITE_DEFERRED = dict(SITE_DEFERRED)
GQA_VALID_AR = dict(VALID_AR)
GQA_TWO_SHOT_PRODUCERS = dict(TWO_SHOT_PRODUCERS)
GQA_TWO_SHOT_SILU = dict(TWO_SHOT_SILU)
GQA_TWO_SHOT_EARLY = dict(TWO_SHOT_EARLY)
# The context step on a side stream joined before the first attention core: none
# until measured on the GQA chain at 16 ranks (NQ-TUNE: the preview's 1-63 rows).
GQA_CONTEXT_OVERLAP: dict[int, str] = {}
# Requests per drafter graph whose Markov sampling runs ``VocabParallelMarkov`` (one
# set of launches per draft position, no co-resident CTAs), per TP: TP4 (C512's SP
# decode role), where the fused kernels' 320 CTAs per rank (Markov rank 512) exceed
# VR200's resident CTAs.
GQA_VP_MARKOV = {4: "1-:1"}
# TP4 (C512's SP decode role: TP4 = DCP4, DL3 B2-B128 per DP rank, query rows 6-384,
# context rows 8-512): the sharded context step (fc shard (1792, 35840), CtxGather at
# every context row), the Lamport sites (one-shot to 7 rows, two-shot 8-64, ArNormRows
# above) over stored partials. NQ-TUNE: the TP4 sites / producers on 4 ranks.
GQA_FAST_CONTEXT[4] = "1-:1"
GQA_FAST_AR[4] = "1-7:1,8-64:2,65-:3"


def gqa_projection_tactic(shape, tp, rows):
    """The GQA tables' split-K tactic of (N, K) at ``rows`` (None: cuBLAS)."""
    shape = tuple(shape)
    if rows <= PROJECTION_ROWS:
        return GQA_PROJECTIONS.get(tp, {}).get(shape)
    bins = GQA_PROJECTIONS_WIDE.get(tp, ())
    table = next((t for top, t in bins if rows <= top), {})
    return table.get(shape)


def gqa_projection(weight, tp, rows):
    """The prepared split-K projection of ``weight`` for ``rows`` (or None)."""
    from ..lt_pin import with_pins

    policies = {}
    for r in rows:
        tactic = gqa_projection_tactic(weight.shape, tp, r)
        if tactic is not None:
            policies[r] = tactic
    prepared = None
    if policies:
        from ..cutedsl.gemm.projection import PreparedProjection

        prepared = PreparedProjection(weight, policies).prepare()
    # Rows above the split-K bins: pinned cuBLASLt algorithms where the heuristic
    # pick is slow (lt_pin.PINS; e.g. TP4 gate_up at 288 rows 30.1 -> 17.1 us).
    return with_pins(prepared, weight, rows)


def install_gqa_projections(model, tp, query_rows, context_rows, rank=0):
    """Split-K projections for the drafter's per-layer linears (``query_rows``) and
    its fused context K/V weight (``context_rows``; ``model.model._context_projection``,
    the context step's K/V GEMM); returns how many were installed. The other rows run
    cuBLAS (logged per shape: a measured choice, NQ-NOFALLBACK coverage)."""
    inner = model.model
    linears = []
    for layer in inner.layers:
        attn, mlp = layer.self_attn, layer.mlp
        linears += [(attn.qkv_proj, query_rows), (attn.o_proj, query_rows)]
        linears += [(mlp.gate_up_proj, query_rows), (mlp.down_proj, query_rows)]
    installed = 0
    stock: dict[tuple, set[int]] = {}
    for linear, rows in linears:
        if getattr(linear, "bias", None) is not None:
            raise ValueError("Kimi K3.1 GQA DSpark drafter: biased linear")
        prepared = gqa_projection(linear.weight, tp, rows)
        served = set(prepared.kernels) if prepared is not None else set()
        stock.setdefault(tuple(linear.weight.shape), set()).update(
            r for r in rows if r not in served
        )
        if prepared is not None:
            linear.quant_method = DraftProjectionMethod(linear.quant_method, prepared)
            installed += 1
    weight = getattr(inner, "_fused_kv_weight", None)
    inner._context_projection = None
    if weight is not None:
        inner._context_projection = gqa_projection(weight, tp, context_rows)
        served = inner._context_projection.kernels if inner._context_projection else {}
        stock.setdefault(tuple(weight.shape), set()).update(
            r for r in context_rows if r not in served
        )
        installed += inner._context_projection is not None
    for shape, rows in sorted(stock.items()):
        if rows:
            log_coverage(
                rank,
                "draft linear",
                f"(N, K) {shape}: stock cuBLAS at rows {sorted(rows)} (measured: "
                "release GEMM bench)",
            )
    return installed


def gqa_context_projection(prepared, tp):
    """``PreparedContext``'s shard GEMM on the GQA tables' split-K tactics for its
    static rows (the release drafter's fc shard: (448, 35840) at TP16, as the
    preview's context_proj shard; (1792, 35840) at TP4), as
    ``PreparedContext.__init__`` sets it up from the preview's tables: rows whose
    tactic has the M tile 64 and N tiles of at most 16 publish into the row mailbox
    from the GEMM itself (``GatherProjection``, one 8-column fragment per epilogue
    thread; polled CtxGather), the others run the split-K GEMM and CtxGather's
    publish. Returns the (split-K, mailbox-publishing) rows."""
    from ..cutedsl.draft.context import CAP, GatherProjection, compile_gather
    from ..cutedsl.gemm.projection import PreparedProjection

    shard = prepared.shard
    tactics = {}
    for r in prepared.rows:
        tactic = gqa_projection_tactic(shard.shape, tp, r)
        if tactic is not None:
            tactics[r] = tactic
    ring = {r: t for r, t in tactics.items() if len(t) == 4 and t[1] <= 16}
    current = prepared.projection.policies if prepared.projection is not None else {}
    if not tactics or (tactics == current and set(ring) == set(prepared.polled)):
        from ..lt_pin import with_pins

        prepared.projection = with_pins(prepared.projection, shard, prepared.rows)
        return sorted(current), sorted(prepared.polled)
    index = prepared.device.index
    prepared.projection = PreparedProjection(shard, tactics).prepare()
    prepared.ring = GatherProjection(shard, ring).prepare() if ring else None
    prepared.polled = {r: compile_gather(r, shard.shape[0], index, True) for r in ring}
    prepared.shape_rows = torch.empty(
        (CAP, shard.shape[0]), dtype=torch.bfloat16, device=prepared.device
    )
    return sorted(tactics), sorted(ring)


class GqaContextWeights:
    """``cutedsl.draft.context.PreparedContext``'s view of the release drafter: its
    ``context_proj`` is ``fc`` and its ``context_norm`` is ``hidden_norm``
    (``combine`` = hidden_norm(fc(aux)))."""

    def __init__(self, inner):
        self.context_proj, self.context_norm = inner.fc, inner.hidden_norm


def deployment_config(vllm_config):
    """The deployment's (target's) config for a drafter's ``vllm_config``: a GQA draft
    runs at DCP 1 (DFlashSpeculator replaces the parallel config of its copy), while
    the deployment, and every strictness rule keyed on it, is the target's
    (``speculative_config.target_parallel_config``)."""
    import copy

    spec = vllm_config.speculative_config
    target = getattr(spec, "target_parallel_config", None) if spec is not None else None
    if target is None or target is vllm_config.parallel_config:
        return vllm_config
    config = copy.copy(vllm_config)
    config.parallel_config = target
    return config


class GqaAttention:
    """The fused GQA attention of the drafter's layers (``cutedsl.draft.gqa_*``,
    ``install_gqa_attention``), as the layers call it: ``forward(index, qkv,
    positions)`` (per-head q / k RMSNorm, RoPE, the K/V insert of the query block and
    the sliding-window attention: the layer's o_proj input [rows, q_size], or None
    when not covered) and ``store_context(kv, positions, slots)`` (the context K/V
    epilogue over the context K/V GEMM's rows [C, L x 2 x kv_size]: per-layer k_norm,
    RoPE, insert; False when not covered). ``state`` is the installer's prepared
    state with the same two methods. NQ-NOFALLBACK: inside a FULL capture on a strict
    deployment an uncovered call refuses instead of capturing the stock attention.
    ``ready``: the side-stream context step's event the first layer waits for."""

    def __init__(self, state, strict):
        self.state, self.strict = state, strict
        self.ready = None

    def forward(self, index, qkv, positions):
        if not torch.cuda.is_current_stream_capturing():
            return None
        if index == 0 and self.ready is not None:
            torch.cuda.current_stream().wait_event(self.ready)
            self.ready = None
        out = self.state.forward(index, qkv, positions)
        if out is None and self.strict and full_capture():
            refuse("DSpark drafter attention", "not covered", Shape(qkv.shape[0]))
        return out

    def store_context(self, kv, positions, slots):
        if not torch.cuda.is_current_stream_capturing():
            return False
        stored = bool(self.state.store_context(kv, positions, slots))
        if not stored and self.strict and full_capture():
            refuse("DSpark drafter context K/V", "not covered", Shape(kv.shape[0]))
        return stored

    def uncovered_context(self, rows, cause):
        """A context step the fused K/V insert cannot take (``cause``): refuses inside
        a FULL capture on a strict deployment (never a silent stock insert)."""
        if self.strict and full_capture():
            refuse("DSpark drafter context K/V", cause, Shape(rows))


def install_gqa_attention_state(model, vllm_config, query_rows, context_rows, strict):
    """The fused GQA attention's prepared state for the drafter's layers, or None
    when it is not available (``cutedsl.draft.gqa_*`` not installed).
    TODO(K31 dsrel track K): the fused GQA sliding-window attention kernel and its
    installer ``install_gqa_attention(layers, vllm_config, query_rows, context_rows,
    strict)`` (layers: the drafter's DFlashQwen3Attention modules) returning a state
    with ``forward`` / ``store_context`` (``GqaAttention``)."""
    try:
        from ..cutedsl.draft.gqa_attention import install_gqa_attention
    except ModuleNotFoundError as e:
        # Only the module's absence means "not installed"; a failing import inside it
        # is an error to see, not a missing piece.
        if not (e.name or "").endswith("cutedsl.draft.gqa_attention"):
            raise
        return None
    layers = [layer.self_attn for layer in model.model.layers]
    return install_gqa_attention(layers, vllm_config, query_rows, context_rows, strict)


class GqaSites:
    """The release drafter's forward on ``PreparedDraftAr``'s Lamport all-reduce +
    residual + RMSNorm sites (``ar``): ``PreparedDraftAr.run`` / ``run_polled`` with
    the GQA layers (``attention_core``: qkv, attention; ``attention``: + o_proj) and
    ``model.norm``; None (nothing launched) when not covered (strict: refuse inside a
    FULL capture). 11 sites per step: the embedding, two per layer, the final norm."""

    def __init__(self, ar):
        self.ar = ar

    @property
    def rows(self):
        return self.ar.rows

    def run(self, model, input_ids, positions):
        ar = self.ar
        rows = input_ids.shape[0]
        ar.stamp_index = 0
        if not torch.cuda.is_current_stream_capturing():
            return None  # eager: no graph, the stock layers and all-reduce
        weight = getattr(model.embed_tokens, "weight", None)
        checks = (
            (rows not in ar.reduce, "rows without prepared sites"),
            (input_ids.dtype != torch.int32, f"input ids {input_ids.dtype}"),
            (not input_ids.is_contiguous(), "strided input ids"),
            (weight is None, "no embedding"),
            (
                weight is not None and weight.shape[0] != ar.shard,
                "an embedding shard other than the vocabulary / TP rows",
            ),
            (weight is not None and not weight.is_contiguous(), "a strided embedding"),
            (model.has_separate_mask_embedding, "a separate mask embedding"),
        )
        reason = next((why for failed, why in checks if failed), None)
        if reason is not None:
            if ar.strict and full_capture():  # NQ-NOFALLBACK
                refuse("DSpark drafter all-reduce sites", reason, Shape(rows))
            return None
        empty = lambda: torch.empty(  # noqa: E731
            (rows, GQA_HIDDEN), dtype=weight.dtype, device=weight.device
        )
        # Site 0: the embedding rows from their owners (parity 0), normed.
        ar.ring(input_ids, weight, ar.rank * ar.shard, ar.vocab, ar.embed_address(rows))
        residual, hidden = empty(), empty()
        layers = model.layers
        embed = ar.polled_embed.get(rows, ar.embed[rows])
        ar.site(embed, residual, residual, layers[0].input_layernorm, hidden, 0)
        if rows in ar.polled:
            return self.run_polled(model, rows, residual, hidden, positions, empty)
        site = 1
        for index, layer in enumerate(layers):
            if index > 0:
                normed = empty()
                norm = layer.input_layernorm
                ar.site(ar.reduce[rows], hidden, residual, norm, normed, site % 2)
                residual, hidden, site = hidden, normed, site + 1
            attn = layer.attention(positions, hidden)
            normed = empty()
            norm = layer.post_attention_layernorm
            ar.site(ar.reduce[rows], attn, residual, norm, normed, site % 2)
            residual, site = attn, site + 1
            hidden = layer.mlp(normed)
        out = empty()
        ar.site(ar.reduce[rows], hidden, residual, model.norm, out, site % 2)
        return out

    def run_polled(self, model, rows, residual, hidden, positions, empty):
        """The layers with ring (or two-shot pushing) o_proj / down_proj partials and
        polled sites; the residual is updated in place."""
        ar = self.ar
        polled, site = ar.polled[rows], 1
        two = rows in ar.two_rows
        for index, layer in enumerate(model.layers):
            o_proj, down = (ar.two_producers if two else ar.producers)[index]
            if index > 0:  # the previous layer's down partials are in this site's slots
                hidden = empty()
                norm = layer.input_layernorm
                ar.site(polled, residual, residual, norm, hidden, site % 2)
                site += 1
            attn_out = layer.attention_core(positions, hidden)
            ar.publish(o_proj, attn_out, site % 2)
            normed = empty()
            norm = layer.post_attention_layernorm
            ar.site(polled, residual, residual, norm, normed, site % 2)
            site += 1
            mlp, gate_up = layer.mlp, None
            if two and ar.two_gate_up and rows in ar.two_gate_up[index].policies:
                gate_up = ar.two_gate_up[index]
            elif not two:
                gate_up = ar.gate_up[index]
            if gate_up is not None:
                act = gate_up.run(normed, normed.new_empty((rows, gate_up.half)))
            else:
                act = mlp.act_fn(mlp.gate_up_proj(normed)[0])
            ar.publish(down, act, site % 2)
        out = empty()
        ar.site(polled, residual, residual, model.norm, out, site % 2)
        return out


def gqa_checks(vllm_config, model):
    """Why the release drafter's pieces cannot serve this draft (None: they can): the
    geometry and math the kernels implement."""
    inner = model.model
    config = model.config
    layers = list(inner.layers)
    attn = layers[0].self_attn if layers else None
    spec = vllm_config.speculative_config
    sampled = spec.draft_sample_method == "probabilistic"
    checks = (
        (not layers, "no draft layers"),
        (getattr(config, "hidden_size", 0) != GQA_HIDDEN, "a hidden size but 7168"),
        (
            not getattr(config, "sample_from_anchor", False),
            "drafts that do not sample from the anchor",
        ),
        (getattr(config, "dspark_draft_topk", None) is not None, "top-k drafting"),
        (attn is not None and attn.v_scale is not None, "an attention value scale"),
        (
            attn is not None and attn.attention_sink_bias is not None,
            "attention sinks",
        ),
        (inner.has_separate_mask_embedding, "a separate mask embedding"),
        (
            any(
                getattr(layer.self_attn.qkv_proj, "bias", None) is not None
                for layer in layers
            ),  # fmt: skip
            "biased projections",
        ),
        (
            model.logits_processor.scale != 1.0
            or model.logits_processor.soft_cap is not None,
            "a logit scale or soft cap",
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
    return next((why for failed, why in checks if failed), None)


class GqaDraftFastPath(DraftFastPath):
    """Per-rank prepared pieces of the release (GQA) drafter; built after the draft
    weights load (``K3GqaDSparkForCausalLM.process_weights_after_loading``).
    ``attention``: the layers' fused GQA attention (``GqaAttention``) or None."""

    def __init__(
        self, markov=None, batches=(), steps=7, tp=1, all_batches=(), sampled=False,
        strict=False, rank=0, attention=None,
    ):  # fmt: skip
        super().__init__(markov, batches, steps, tp, all_batches, sampled, strict, rank)
        rows = lambda table: frozenset(  # noqa: E731
            b for b in all_batches if covered(table.get(tp, ""), b * steps, steps)
        )
        self.lm_batches = rows(GQA_FAST_LM_HEAD) & self.batches
        self.wide_batches = rows(GQA_WIDE_LM_HEAD) & self.batches
        self.sk_batches = (
            rows(GQA_SK_LM_HEAD) & self.batches - self.lm_batches - self.wide_batches
        )
        self.overlap_batches = rows(GQA_CONTEXT_OVERLAP)
        self.attention = attention

    def context_overlap(self, model, num_reqs):
        """GQA_CONTEXT_OVERLAP: the context step may run on a side stream (its
        sharded context step and the fused attention, which takes the join, are
        prepared)."""
        inner = model.model
        return (
            num_reqs in self.overlap_batches
            and self.attention is not None
            and getattr(inner, "_fast_context", None) is not None
        )

    def defer_context_join(self, model, ready):
        """The first layer's attention waits for ``ready`` before its core."""
        self.attention.ready = ready

    @classmethod
    def build(cls, vllm_config, model):
        """None unless the specialized decode is on and this device and deployment
        are covered. ``model``: the K3GqaDSparkForCausalLM (weights loaded)."""
        if not enabled(vllm_config):
            return None
        spec = vllm_config.speculative_config
        if spec is None or spec.method != "dspark":
            return None
        device = torch.device(torch.cuda.current_device())
        if torch.cuda.get_device_capability(device) != (10, 7):
            return None
        from vllm.distributed import get_tp_group

        from . import sp_topology

        check_draft_formats(model)
        group = get_tp_group()
        tp, rank = group.world_size, group.rank_in_group
        steps = spec.num_speculative_tokens
        all_batches = draft_batches(vllm_config, steps)
        # The deployment's topology is the target's: DFlashSpeculator hands a GQA draft
        # a copy of the config at DCP 1 (dflash/speculator.py), which would make every
        # strict deployment look non-strict here.
        deployment = deployment_config(vllm_config)
        dcp = deployment.parallel_config.decode_context_parallel_size
        # NQ-NOFALLBACK: on the target topology (Rubin TP16 = DCP16) and C512's SP
        # decode role (TP4 = DCP4) every drafter piece a captured graph runs is covered
        # or the load is refused; elsewhere (out of scope) uncovered pieces keep the
        # stock drafter path, logged.
        strict = target_topology(True, tp, dcp, steps + 1) or sp_topology(deployment)

        def fail(site, cause):
            if strict:
                refuse(site, cause)
            logger.warning_once(
                "Kimi K3.1 GQA DSpark drafter: %s keeps stock (%s)", site, cause
            )

        inner = model.model
        layers = list(inner.layers)
        reason = gqa_checks(vllm_config, model)
        if reason is not None:
            fail("DSpark drafter", reason)
            return None
        query_rows = [b * steps for b in all_batches]
        context_rows = [b * (steps + 1) for b in all_batches]
        top = max(all_batches, default=0)
        top_rows = top * steps  # (the open query-row ranges end here)
        policy = lambda table, r: covered(  # noqa: E731
            closed(table.get(tp, ""), top_rows), r, steps
        )

        # Markov sampling over this rank's vocabulary shard: the fused kernels, or the
        # per-position vocabulary-parallel launches (GQA_VP_MARKOV: TP4).
        sampled = spec.draft_sample_method == "probabilistic"
        head = inner.markov_head
        markov_args = (head, tp, rank, steps, all_batches, top, sampled, fail)
        if tp in GQA_VP_MARKOV:
            markov, batches = cls.vp_markov(*markov_args)
        else:
            markov, batches = cls.fused_markov(*markov_args, group, device)
        missing = [b for b in all_batches if b not in batches]
        if missing:
            fail("DSpark drafter Markov sampling", f"no Markov kernels for {missing}")

        # The context step: the fc shard + Lamport row gather + hidden_norm.
        context = [
            r
            for r in context_rows
            if covered(
                closed(GQA_FAST_CONTEXT.get(tp, ""), top * (steps + 1)), r, steps
            )
        ]
        if context != context_rows:
            fail(
                "DSpark drafter context step",
                f"GQA_FAST_CONTEXT misses {sorted(set(context_rows) - set(context))}",
            )
        if context and inner.fc.weight.shape[0] % tp:
            fail(
                "DSpark drafter context step",
                f"fc rows {inner.fc.weight.shape[0]} not divisible by TP{tp}",
            )
            context = []
        if context:
            from ..cutedsl.draft.context import PreparedContext

            prepared = PreparedContext(
                GqaContextWeights(inner), group.device_group, tp, rank, context
            )
            prepared.strict = strict  # NQ-NOFALLBACK
            gqa_context_projection(prepared, tp)
            inner._fast_context = prepared
            logger.info_once(
                "Kimi K3.1 GQA DSpark drafter: sharded context step for %s rows "
                "(persistent row gather at %s)",
                (*prepared.rows, *prepared.wide),
                prepared.persistent,
            )

        # The context step's K/V GEMM runs on the fused K/V weight of every layer
        # (``_build_fused_kv_buffers``, built before this build wraps the qkv
        # projections: ``K3GqaDSparkForCausalLM.process_weights_after_loading``).
        # Without it the stock step projects every layer's whole qkv over the context
        # rows (5 GEMMs + split-K reduces + a stack instead of one).
        if getattr(inner, "_fused_kv_weight", None) is None:
            fail(
                "DSpark drafter context K/V projection",
                "no fused K/V weight (the context buffers are not built or the qkv "
                "projections are not unquantized)",
            )
        installed = install_gqa_projections(model, tp, query_rows, context_rows, rank)
        if installed:
            logger.info_once(
                "Kimi K3.1 GQA DSpark drafter: split-K projections on %d linears",
                installed,
            )

        # The all-reduce + residual + RMSNorm sites (and their producers).
        ar_rows = [r for r in query_rows if policy(GQA_FAST_AR, r)]
        if query_rows and ar_rows != query_rows:
            fail(
                "DSpark drafter all-reduce sites",
                f"GQA_FAST_AR misses rows {sorted(set(query_rows) - set(ar_rows))}",
            )
        if ar_rows:
            from ..cutedsl.draft.allreduce import PreparedDraftAr

            prepared_ar = PreparedDraftAr(
                group.device_group,
                device,
                tp,
                rank,
                ar_rows,
                model.config.vocab_size,
                {r for r in ar_rows if policy(GQA_FAST_AR, r) == 2},
                {r for r in ar_rows if policy(GQA_GATED_AR, r)},
                {r for r in ar_rows if policy(GQA_TWO_SHOT_EARLY, r)},
                {r for r in ar_rows if policy(GQA_VALID_AR, r)},
                {r for r in ar_rows if policy(GQA_FAST_AR, r) == 3},
            )
            prepared_ar.strict = strict  # NQ-NOFALLBACK
            unserved = sorted(set(query_rows) - set(prepared_ar.rows))
            if unserved:
                fail(
                    "DSpark drafter all-reduce sites",
                    f"query rows {unserved} have no Lamport sites",
                )
            if prepared_ar.rows:
                cls.prepare_producers(prepared_ar, layers, tp, policy)
                inner._fast_ar = GqaSites(prepared_ar)
                logger.info_once(
                    "Kimi K3.1 GQA DSpark drafter: Lamport all-reduce sites for %s "
                    "rows (o_proj / down_proj publishing their partials for %s)",
                    tuple(prepared_ar.rows),
                    tuple(sorted(prepared_ar.polled)),
                )

        # The fused GQA attention (and the context K/V insert).
        state = install_gqa_attention_state(
            model, vllm_config, query_rows, context_rows, strict
        )
        attention = None
        missing_api = [
            name for name in ("forward", "store_context") if not hasattr(state, name)
        ]
        if state is None:
            fail(
                "DSpark drafter attention",
                "the fused GQA drafter attention (cutedsl.draft.gqa_attention "
                "install_gqa_attention) is not installed",
            )
        elif missing_api:
            fail(
                "DSpark drafter attention",
                f"the installed GQA attention state lacks {missing_api}",
            )
        else:
            attention = GqaAttention(state, strict)
            for layer in layers:
                layer.fast_attention = attention
            inner._fast_context_kv = attention
        logger.info_once(
            "Kimi K3.1 GQA DSpark drafter: fused %s Markov sampling for draft batches "
            "%s",
            "probabilistic" if sampled else "greedy",
            tuple(batches),
        )
        fast = cls(
            markov, batches, steps, tp, all_batches, sampled, strict, rank, attention
        )
        lm = sorted(b * steps for b in fast.lm_batches)
        wide_lm = sorted(b * steps for b in fast.wide_batches)
        sk_lm = sorted(b * steps for b in fast.sk_batches)
        stock_lm = sorted(
            b * steps
            for b in batches
            if b not in fast.lm_batches | fast.wide_batches | fast.sk_batches
        )
        log_coverage(
            rank,
            "draft LM head",
            f"GEMV rows {lm}, evict-first GEMM rows {wide_lm}, die-local stream-K "
            f"GEMM rows {sk_lm}, stock cuBLAS shard "
            f"GEMM rows {stock_lm} (GQA_WIDE_LM_HEAD: measured choices, as the "
            "preview drafter's)",
        )
        return fast

    @staticmethod
    def fused_markov(head, tp, rank, steps, all_batches, top, sampled, fail, group,
                     device):  # fmt: skip
        """(PreparedMarkov, its batches) for GQA_FUSED_MARKOV's batches at ``tp`` (the
        preview drafter's kernels and choices); each batch run once (every rank
        together)."""
        from ..cutedsl.draft.markov import (
            MAX_WIDTH,
            PreparedMarkov,
            device_limits,
            supports,
        )

        table = batch_table(closed(GQA_FUSED_MARKOV.get(tp, ""), top))
        settle = batch_table(GQA_SETTLED_MARKOV.get(tp, ""))
        batches = [b for b in all_batches if table.get(b)]
        w1, w2 = head.markov_w1.weight, head.markov_w2.weight
        if not batches:
            return None, []
        if not supports(w1.shape, w2.shape, tp, device_limits(device)):
            fail(
                "DSpark drafter Markov sampling",
                f"Markov head {tuple(w2.shape)} at TP{tp} (rank, vocabulary split or "
                "co-residency of the fused kernels)",
            )
            return None, []
        markov = PreparedMarkov(
            w1,
            w2,
            tp,
            rank,
            group.device_group,
            steps,
            gated=[b for b in batches if table.get(b) in (2, 3)],
            staged=[b for b in batches if table.get(b) == 3],
            settled=[b for b in batches if settle.get(b)],
            wide=max(batches) > MAX_WIDTH,
        )
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
        return (markov, batches) if batches else (None, [])

    @staticmethod
    def vp_markov(head, tp, rank, steps, all_batches, top, sampled, fail):
        """(VocabParallelMarkov, its batches) for GQA_VP_MARKOV's batches at ``tp``;
        each batch run once (every rank together)."""
        table = batch_table(closed(GQA_VP_MARKOV.get(tp, ""), top))
        batches = [b for b in all_batches if table.get(b)]
        if not batches:
            return None, []
        try:
            from vllm.distributed import get_tp_group

            # The polled per-position mailbox needs >= 2 draft positions: a peer
            # reuses position 0's slots after its select of the NEXT position, which
            # waits for this rank's keys; with one position (DSpark k1) the next is
            # the next run's position 0, so a fast peer overwrites keys this rank has
            # not taken yet (hang seen at k1, 2026-11-22 sdk-smk3-DCk1). k1 keeps the
            # in-graph all-gather of the keys (same keys and selection).
            group = get_tp_group().device_group if steps > 1 else None
            markov = VocabParallelMarkov(
                head.markov_w1.weight, head.markov_w2.weight, tp, rank, steps,
                group=group, cap=max(batches),
            )  # fmt: skip
        except ValueError as error:
            fail("DSpark drafter Markov sampling", str(error))
            return None, []
        for batch in batches:
            markov.prepare(batch, steps, steps, sampled)
        return markov, batches

    @staticmethod
    def prepare_producers(prepared_ar, layers, tp, policy):
        """The one-shot rows' ring o_proj / down_proj (+ fused gate/up SiLU), the
        two-shot rows' pushing producers (+ fused gate/up SiLU), as the preview
        drafter's build (``DraftFastPath.build``) with the GQA tables."""
        rings = GQA_RING_PROJECTIONS.get(tp, {})
        tactics = {r: rings for r in prepared_ar.rows if r <= PROJECTION_ROWS}
        ring_shapes = {
            tuple(layers[0].self_attn.o_proj.weight.shape),
            tuple(layers[0].mlp.down_proj.weight.shape),
        }
        if tactics and not ring_shapes <= set(rings):
            logger.warning_once(
                "Kimi K3.1 GQA DSpark drafter: no ring projection for %s at TP%d: "
                "rows %s keep stored partials (no polled sites)",
                tuple(sorted(ring_shapes - set(rings))),
                tp,
                tuple(sorted(tactics)),
            )
        deferred = {r for r in prepared_ar.valid_ar if policy(GQA_SITE_DEFERRED, r)}
        prepared_ar.prepare_producers(layers, tactics, deferred, trigger=True)
        from ..cutedsl.draft.allreduce import plain_layer
        from ..cutedsl.draft.producer import ROWS, TwoShotProducer

        pushed = sorted(
            r
            for r in prepared_ar.two_rows
            if policy(GQA_TWO_SHOT_PRODUCERS, r) and r in ROWS
        )
        if not pushed or not all(plain_layer(layer) for layer in layers):
            return
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
            if not policy(GQA_TWO_SHOT_SILU, r):
                continue
            tactic = gqa_projection_tactic(shape, tp, r)
            if tactic is not None:
                fused[r] = tactic
        prepared_ar.prepare_two_gate_up(layers, fused)


# ---------------------------------------------------------------------------------
# Vocabulary-parallel Markov sampling without co-residency (the TP4 drafter: C512's
# SP decode role). The fused kernels (cutedsl.draft.markov / markov_wide) keep every
# CTA of a rank resident (each polls every CTA's keys per position): at Markov rank 512
# a rank of TP4 owns 40960 vocabulary rows = 320 CTAs of 128 staged rows (~133 KB
# shared memory each, one per SM), above VR200's 212 SMs (``markov.supports``). Here
# each draft position is its own launches, so no kernel waits on a CTA of its own
# launch:
#   1. the Markov bias of this rank's vocabulary rows: F.linear(markov_w1[prev],
#      markov_w2 shard) (cuBLAS; the stock bias GEMM's columns, 1/TP of its weight),
#   2. ``_vp_markov_keys``: logits = base + bias (BF16, the stock add), each live
#      request's pre-temperature logits into this rank's columns of the draft-logits
#      cache, the stock Gumbel-max score (``gumbel_noised_argmax``: the stock Triton
#      code, bitwise its noise and scores) and one u64 key per request (FP32 score,
#      then the lowest token: ``gumbel.sampled_key``'s order) by atomic max,
#   3. the TP all-gather of the keys (TP x requests u64),
#   4. ``_vp_markov_select``: each request's largest key (stock's first maximum over
#      the vocabulary), its draft token, the next position's markov_w1 row; re-arms
#      the keys.
# Drafts as the stock loop's (``DSparkSpeculator._sample_sequential``: greedy
# argmax, or ``gumbel_sample``); only the bias GEMM's FP32 accumulation order may
# differ (cuBLAS on the shard's columns).

VP_MARKOV_BLOCK = 1024  # vocabulary columns per program (the stock sampler's block)


@triton.jit
def _vp_ordered(value):
    """Monotone u32 image of an FP32 ``value`` (-0.0 as +0.0)."""
    bits = value.to(tl.uint32, bitcast=True)
    bits = tl.where(bits == 0x80000000, 0, bits)
    return tl.where((bits & 0x80000000) != 0, ~bits, bits | 0x80000000)


@triton.jit
def _vp_markov_keys(
    keys_ptr,
    base_ptr,
    base_stride,
    bias_ptr,
    bias_stride,
    cache_ptr,
    cache_stride0,
    cache_stride1,
    slots_ptr,
    pos_ptr,
    temp_ptr,
    seeds_ptr,
    step,
    steps,
    lo,
    vocab_rank,
    count_ptr,
    peers_ptr,
    box_offset,
    programs,
    SAMPLED: tl.constexpr,
    BLOCK: tl.constexpr,
    TP: tl.constexpr,
    PUBLISH: tl.constexpr,
):
    req = tl.program_id(0)
    block = tl.program_id(1)
    cols = block * BLOCK + tl.arange(0, BLOCK)
    mask = cols < vocab_rank
    row = req * steps + step
    base = tl.load(base_ptr + row * base_stride + cols, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + req * bias_stride + cols, mask=mask, other=0.0)
    logits = (base.to(tl.float32) + bias.to(tl.float32)).to(tl.bfloat16)
    values = tl.where(mask, logits.to(tl.float32), float("-inf"))
    tokens = lo + cols
    if SAMPLED:
        slot = tl.load(slots_ptr + row)
        live = slot >= 0
        temp = tl.load(temp_ptr + slot, mask=live, other=0.0).to(tl.float32)
        tl.store(
            cache_ptr + slot * cache_stride0 + step * cache_stride1 + tokens,
            logits,
            mask=mask & live,
        )
        seed = tl.load(seeds_ptr + slot, mask=live, other=0)
        seed = spec_step_seed(seed, step)
        pos = tl.load(pos_ptr + row) - 1
        value, index = gumbel_noised_argmax(
            values,
            tokens,
            mask,
            seed,
            pos,
            temp,
            IS_DRAFTING=True,
            USE_FP64=False,
            APPLY_TEMPERATURE=True,
        )
    else:
        value, index = tl.max(values, axis=0, return_indices=True)
    token = (lo + block * BLOCK + index).to(tl.int64)
    key = (_vp_ordered(value).to(tl.int64) << 31) | (0x7FFFFFFF - token)
    if PUBLISH:
        # (relaxed: the count's acq_rel below orders it before the last program's
        # read; one fence per program)
        tl.atomic_max(keys_ptr + req, key, sem="relaxed", scope="gpu")
    else:
        tl.atomic_max(keys_ptr + req, key)
    if PUBLISH:
        # The request's last program on this rank publishes the rank's key (never 0)
        # into slot ``box_offset + req`` of every rank's mailbox and re-arms. A key
        # is its own flag (nothing else is read through the mailbox): plain stores,
        # no system-scope fence or remote atomic per peer.
        done = tl.atomic_add(count_ptr + req, 1, sem="acq_rel", scope="gpu")
        if done == programs - 1:
            final = tl.atomic_xchg(keys_ptr + req, 0, sem="acq_rel", scope="gpu")
            tl.atomic_xchg(count_ptr + req, 0, sem="relaxed", scope="gpu")
            for r in tl.static_range(TP):
                # (plain stores: a remote atomic waits for its reply, four in a row)
                peer = tl.load(peers_ptr + r) + (box_offset + req) * 8
                tl.store(peer.to(tl.pointer_type(tl.int64)), final)


@triton.jit
def _vp_take(slot):
    """A mailbox slot's key once a rank published it (never 0); re-arms the slot."""
    got = tl.atomic_add(slot, 0, sem="relaxed", scope="sys")
    while got == 0:
        got = tl.atomic_add(slot, 0, sem="relaxed", scope="sys")
    tl.store(slot, 0)
    return got


@triton.jit
def _vp_markov_select(
    gathered_ptr,
    keys_ptr,
    draft_ptr,
    draft_stride,
    emb_ptr,
    w1_ptr,
    num_reqs,
    step,
    TP: tl.constexpr,
    RANK: tl.constexpr,
    POLL: tl.constexpr,
):
    req = tl.program_id(0)
    if POLL:
        # This position's mailbox [TP, num_reqs] (``gathered_ptr``): wait for every
        # rank's published key, then re-arm the slots (the keys re-armed already).
        best = _vp_take(gathered_ptr + req)
        for r in tl.static_range(1, TP):
            best = tl.maximum(best, _vp_take(gathered_ptr + r * num_reqs + req))
    else:
        best = tl.load(gathered_ptr + req)
        for r in tl.static_range(1, TP):
            best = tl.maximum(best, tl.load(gathered_ptr + r * num_reqs + req))
        tl.store(keys_ptr + req, 0)
    token = 0x7FFFFFFF - (best & 0x7FFFFFFF)
    tl.store(draft_ptr + req * draft_stride + step, token)
    cols = tl.arange(0, RANK)
    row = tl.load(w1_ptr + token * RANK + cols)
    tl.store(emb_ptr + req * RANK + cols, row)


class VocabParallelMarkov:
    """Markov drafting over this rank's vocabulary shard, one set of launches per
    draft position (module section above): ``PreparedMarkov``'s interface
    (``covers`` / ``prepare`` / ``ready`` / ``sampling_args`` / ``run``, ``rank``,
    ``vocab_rank``, ``markov_rank``). ``gather``: the TP group's all-gather along
    dim 0 (in-graph)."""

    def __init__(self, w1, w2, tp, rank, steps, gather=None, group=None, cap=0):
        vocab, markov_rank = w2.shape
        if tuple(w1.shape) != tuple(w2.shape) or vocab % (tp * VP_MARKOV_BLOCK):
            raise ValueError(f"unsupported Markov head shapes {tuple(w2.shape)}")
        if markov_rank & (markov_rank - 1):
            raise ValueError(f"Markov rank {markov_rank} is not a power of two")
        self.tp, self.rank, self.steps = tp, rank, steps
        self.vocab, self.markov_rank = vocab, markov_rank
        self.vocab_rank = vocab // tp
        self.w1 = w1.detach().contiguous()
        lo = rank * self.vocab_rank
        self.w2 = w2.detach()[lo : lo + self.vocab_rank]
        self.device = w2.device
        if gather is None and group is None:
            from vllm.distributed import get_tp_group

            coordinator = get_tp_group()
            gather = lambda x: coordinator.all_gather(x, dim=0)  # noqa: E731
        self.gather = gather
        self.batches: set[tuple[int, bool]] = set()
        self.buffers: dict[int, tuple] = {}
        self.sk = None  # the die-local stream-K bias GEMM (GQA_SK_MARKOV batches)
        # ``group`` (a process group, with ``cap`` = the largest batch): the keys'
        # all-gather becomes peer stores into a mailbox [steps, TP, cap] polled by
        # the select (no collective launch). One mailbox per draft position: a peer
        # reuses position k's slots only after its select of position k + 1 (or of
        # the next run's first position), which waits for this rank's keys of that
        # position, published after this rank consumed (re-armed) position k. Two
        # parities would race at odd ``steps`` (position steps - 1, then 0 again).
        self.cap, self.box, self.peers = cap, None, None
        self.count = None
        if group is not None and tp > 1:
            from ..symm import peer_rendezvous
            import torch.distributed._symmetric_memory as symm_mem

            if cap <= 0:
                raise ValueError("Markov drafting: a mailbox needs the largest batch")
            slots = max(steps, 2)
            box = symm_mem.empty(
                (slots, tp, cap), dtype=torch.int64, device=self.device
            )
            box.zero_()
            handle = peer_rendezvous(box, group)
            size = box[0].numel() * 8
            self.box = box
            self.peers = torch.tensor(
                [[int(b) + p * size for b in handle.buffer_ptrs] for p in range(slots)],
                dtype=torch.int64,
                device=self.device,
            )
            self.count = torch.zeros(cap, dtype=torch.int32, device=self.device)
            torch.accelerator.synchronize(self.device)
            torch.distributed.barrier(group=group)

    def covers(self, batch, sampled=False):
        return batch > 0

    def ready(self, batch, anchor_stride, draft_stride, sampled=False):
        return (batch, sampled) in self.batches

    def _buffers(self, batch):
        if batch not in self.buffers:
            self.buffers[batch] = (
                torch.zeros(batch, dtype=torch.int64, device=self.device),
                torch.empty(
                    (batch, self.markov_rank), dtype=self.w1.dtype, device=self.device
                ),
            )
        return self.buffers[batch]

    def prepare_sk(self):
        """The GQA_SK_MARKOV batches' bias GEMM on the die-local stream-K kernel (on its
        die-homed copy of the shard); compiled here, outside CUDA graph captures."""
        if self.sk is not None:
            return
        from ..cutedsl.gemm import wide_sk

        rows = GQA_SK_MARKOV.get(self.tp, ())
        if not rows or self.w2.dtype != torch.bfloat16:
            return
        _, pairs = wide_sk.die_pairs(self.device)
        n, k = self.w2.shape
        rows = [r for r in rows if wide_sk.supports(r, n, k, pairs)]
        if rows:
            self.sk = wide_sk.PreparedWideSK(self.w2, rows).prepare()
            logger.info_once(
                "Kimi K3.1 DSpark drafter: die-local stream-K Markov bias GEMM for "
                "batches %s",
                tuple(rows),
            )

    def prepare(self, batch, anchor_stride, draft_stride, sampled=False):
        """Compile and run once (every rank together: the keys are all-gathered),
        so nothing compiles during CUDA graph capture."""
        self.prepare_sk()
        rows = batch * self.steps
        base = torch.zeros(
            (rows, self.vocab_rank), dtype=torch.bfloat16, device=self.device
        )
        anchors = torch.zeros(batch, dtype=torch.int32, device=self.device)
        draft = torch.zeros((batch, self.steps), dtype=torch.int64, device=self.device)
        sampling = None
        if sampled:
            sampling = self.sampling_args(
                torch.zeros(1, dtype=torch.float32, device=self.device),
                torch.zeros(1, dtype=torch.int64, device=self.device),
                torch.full((rows,), -1, dtype=torch.int64, device=self.device),
                torch.zeros(rows, dtype=torch.int64, device=self.device),
                torch.zeros(
                    (1, self.steps, self.vocab),
                    dtype=torch.bfloat16,
                    device=self.device,
                ),
            )
        self.batches.add((batch, sampled))
        self.run(base, anchors, draft, sampling)
        torch.accelerator.synchronize(self.device)

    def sampling_args(self, temperature, seeds, slots, positions, cache):
        """As ``PreparedMarkov.sampling_args``."""
        if (
            temperature.dtype != torch.float32
            or seeds.dtype != torch.int64
            or slots.dtype != torch.int64
            or positions.dtype != torch.int64
            or cache.dtype != torch.bfloat16
            or tuple(cache.shape[1:]) != (self.steps, self.vocab)
            or cache.stride(-1) != 1
            or not (slots.is_contiguous() and positions.is_contiguous())
        ):
            raise ValueError("Markov drafting: unexpected sampling arguments")
        return (temperature, seeds, slots, positions, cache)

    def run(self, base, anchors, draft, sampling=None):
        """Base [batch * steps, vocab_rank] BF16 (this rank's LM-head logits), anchors
        [batch] int32, draft [batch, steps] int64 (written); ``sampling``: a
        ``sampling_args`` tuple for probabilistic drafting (else greedy)."""
        batch = anchors.shape[0]
        if base.shape != (batch * self.steps, self.vocab_rank) or base.stride(1) != 1:
            raise ValueError(f"Markov drafting: unexpected base {tuple(base.shape)}")
        if draft.shape != (batch, self.steps) or draft.stride(1) != 1:
            raise ValueError(f"Markov drafting: unexpected draft {tuple(draft.shape)}")
        keys, emb = self._buffers(batch)
        torch.index_select(self.w1, 0, anchors.long(), out=emb)
        sampled = sampling is not None
        if sampled:
            temperature, seeds, slots, positions, cache = sampling
            strides = (cache.stride(0), cache.stride(1))
        else:
            temperature = seeds = slots = positions = cache = None
            strides = (0, 0)
        grid = (batch, triton.cdiv(self.vocab_rank, VP_MARKOV_BLOCK))
        lo = self.rank * self.vocab_rank
        poll = self.box is not None
        if poll and batch > self.cap:
            raise ValueError(f"Markov drafting: batch {batch} above the mailbox's {self.cap}")
        sk = self.sk if self.sk is not None and self.sk.takes(emb) else None
        for step in range(self.steps):
            if sk is not None:
                bias = sk.run(emb, emb.new_empty((batch, self.vocab_rank)))
            else:
                bias = torch.nn.functional.linear(emb, self.w2)
            _vp_markov_keys[grid](
                keys,
                base,
                base.stride(0),
                bias,
                bias.stride(0),
                cache,
                *strides,
                slots,
                positions,
                temperature,
                seeds,
                step,
                self.steps,
                lo,
                self.vocab_rank,
                self.count if poll else keys,
                self.peers[step] if poll else keys,
                self.rank * self.cap,
                grid[1],
                SAMPLED=sampled,
                BLOCK=VP_MARKOV_BLOCK,
                TP=self.tp,
                PUBLISH=poll,
            )
            _vp_markov_select[(batch,)](
                self.box[step] if poll else self.gather(keys),
                keys,
                draft,
                draft.stride(0),
                emb,
                self.w1,
                self.cap if poll else batch,
                step,
                TP=self.tp,
                RANK=self.markov_rank,
                POLL=poll,
            )
