# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode projection tactics (``decode.DecodeProjection``) of the sequence-parallel
decode (C512 decode role: TP4 = DCP4 = SP4, 24 of 96 heads per rank).

Per-rank BF16 weights [N, K] at TP4, derived from the model code (K3.1: hidden 7168,
q-LoRA 1536, kv-LoRA 512 + rope 64, 96 heads of 128 (qk 128 nope + 64 rope), 32
replicated indexer heads of 128, 96 KDA heads of 128, MoE latent 3584, two shared
experts of 3072); every caller reads N and K from the loaded weight, these are the
shapes the tables are keyed by:

  kind         N x K          rows   what
  dsa_in       5344 x 7168    R      q_a 1536 | kv_a 576 | gate 24 x 128 | wk 128 |
                                     wv 32 (``providers.pack_dsa_projection``)
  kda_in       12448 x 7168   R      q, k, v, gate 4 x 3072 | f_a 128 | beta 24 | 8 pad
  q_b          4608 x 1536    R      24 heads x 192 (MlaQuery's operand)
  o_proj       7168 x 3072    R      24 heads x 128 on K (``oproj.sp`` fuses the RS)
  wq_b         4096 x 1536    R      the replicated indexer query (``dsa.sp``)
  latent_down  3584 x 7168    n      MoE routed latent down (``runtime.sp_mlp``)
  latent_up    7168 x 3584    n      MoE latent up, beta-add of the shared output
  shared_gu    12288 x 7168   n      replicated shared experts' gate / up
  shared_down  7168 x 6144    n      replicated shared experts' down
  shared_*_side                      the same on the MegaMoE side stream (torch kinds)

R: the TP group's gathered rows (R = 4 n for n = 1..128 requests per DP group, 0.25..32
per GPU); n: this rank's shard rows (n = R / 4).

GENERIC rule (every R a structurally valid tactic): warp split-K up to 16 rows
(``decode.warp_tactic`` for the device's SM count and per-SM shared memory), the
cluster split-K (``GENERIC_SPLIT``: VR's TP16 bin tactics, M tile 64, split 2) at 17-64,
cuBLAS above 64; the beta-add family (latent_up) cuBLAS at every row. All of them sum
K in FP32 and round once to BF16.

TUNED: measured VR200 tables (``TUNED_KIND`` / ``TUNED_SPLIT``, comment below) override
the GENERIC rule where they name a row."""

import torch

from ...shapes import Context, Shape, Unsupported, rows_policy
from .decode import (
    SPLIT_K_ROWS,
    TORCH_KINDS,
    WARP_ROWS,
    ProjectionTactic,
    split_k_problem,
    warp_eligible,
    warp_tactic,
)
from .die_local import MAX_ROWS as DIE_LOCAL_MAX_ROWS
from .warp_splitk import WarpSplitKTactic, smem_limits

# Families: the attention block's projections over the TP group's R gathered rows
# (``R``), the MLP's dense BF16 GEMMs over this rank's n = R / 4 shard rows (``n``,
# ``runtime.sp_mlp``; the stock MegaMoE layer's latent down / up and replicated
# shared expert). ``latent_up`` is stock's beta-add GEMM (``residual.addmm_``): torch
# kinds only.
FAMILIES = {
    "dsa_in": "R", "kda_in": "R", "q_b": "R", "o_proj": "R", "wq_b": "R",
    "latent_down": "n", "latent_up": "n", "shared_gu": "n", "shared_down": "n",
    "shared_gu_side": "n", "shared_down_side": "n",
}  # fmt: skip
KINDS = tuple(FAMILIES)
RESIDUAL = frozenset({"latent_up"})
# The shared expert on the MegaMoE side stream (``KimiMoE._shared_overlap``: the
# Rubin MegaMoE grid capped to leave it SMs): torch kinds only. The split-K kernels'
# one-CTA-per-SM clusters fit beside the capped persistent grid poorly (D3 C512 step
# profile, 64 rows per rank: shared down split-K (64, 64, 1, 7) 65 us per call on
# the side stream vs 11 us chained alone, MegaMoE +2.6 us per layer); the beta-add
# family has no CuTe epilogue.
TORCH_ONLY = RESIDUAL | {"shared_gu_side", "shared_down_side"}
# Per-family weight-table key when two families share an [N, K] weight shape.
SIDE = {"shared_gu_side": "side", "shared_down_side": "side"}
# The TP size these tables are for (the SP decode role).
SP_TP = 4
# A weight placed on the two dies by halves (two-die parts, VR200: the KDA
# in-projection and the shared expert, ``die_local.place_layer_on_dies``): the
# die-local GEMM at every R up to its MAX_ROWS, on the side stream too.
DIE_LOCAL_KINDS = frozenset(
    {"kda_in", "shared_gu", "shared_down", "shared_gu_side", "shared_down_side"}
)

# GENERIC: the kind per R, then the split-K tactic (M tile, N tile, split, stages) per
# R: VR's TP16 bins (``providers.projection_policies``: 32 rows (64, 32, 2, 8), 64 rows
# (64, 64, 2, WIDE_B64_STAGES)); the N tile repeats over rows inside a bin. The
# residual family runs the stock GEMM (cuBLAS) everywhere it has no table row.
GENERIC_KIND = {"*": f"1-{WARP_ROWS}:warp,{WARP_ROWS + 1}-{SPLIT_K_ROWS}:split_k,"
                f"{SPLIT_K_ROWS + 1}-:cublas"}  # fmt: skip
GENERIC_SPLIT = {"*": f"{WARP_ROWS + 1}-32:64.32.2.8,33-{SPLIT_K_ROWS}:64.64.2.6"}

# TUNED (VR200, rel-1767cf7 + this change, 2026-11-18, ``spgemm`` lane): per weight
# (N, K) the kind per rows and, for split-K rows, the tactic. Chosen from a 1-GPU
# chained measurement: a CUDA graph of 8-32 back-to-back calls, each on its own cold
# weight copy (> 2 x L2), so PDL kernels (warp / split-K: weights requested before
# griddepcontrol.wait) overlap the previous call as in the step, and cuBLAS pays its
# launch-to-stream ramp and split-K reduce launch as in the step. Points measured (us
# per call, stock torch.mm -> chosen; results/spgemm chain2 / chain3):
#   kda_in   R16 15.5 cublas (warp 28.9), R32 19.8 -> 15.6 lt, R64 19.8 (split) ->
#            15.6 lt, R128 18.6 -> 17.6 lt, R200 21.9 -> 21.2 lt, R256 23.3 cublas,
#            R320 28.6 -> 23.6 lt, R384 24.1 cublas (lt 28.6)
#   dsa_in   R16 warp 7.9, R32 split 8.7, R64 split 9.9 (as GENERIC), R128 10.6
#            cublas, R200 12.4 -> 12.3 lt, R256 13.2 -> 12.3 lt, R320 16.2 -> 13.7
#            lt, R384 14.1 cublas
#   wq_b     R32 3.17 -> 3.01 (64.32.2.6), R64 4.24 -> 3.52 (64.32.1.9), R128 4.42
#            -> 3.89 lt, R200 7.61 -> 3.97 lt, R256 7.66 -> 4.51 cublas, R320 8.01
#            -> 4.36 lt, R384 8.55 -> 4.38 cublas (was split-K (64, 64, 1, 4): one
#            64-row token tile per CTA re-reads the weight per tile)
#   o_proj   (above the fused warp + RS rows) R128 7.1 / R200 7.9 cublas, R256 9.4
#            -> 8.2 lt, R320 11.7 -> 9.8 lt, R384 9.6 cublas
#   latent_down n4 8.0 -> 5.4 warp, n8 8.2 -> 5.5 warp, n16 8.4 -> 5.8 warp, n32 8.3
#            -> 6.8 (64.32.2.9), n50 11.0 -> 8.4 / n64 10.2 -> 8.5 (64.64.2.6), n80
#            10.6 -> 9.3 (64.80.2.5), n96 9.9 cublas
#   latent_up (beta-add) n4 / n8 10.1 / 10.0 cublas, n16 10.7 -> 8.5 lt, n32 8.4
#            cublas, n50 13.3 -> 9.1 lt, n64 13.6 -> 9.5 lt, n80 9.9 / n96 9.6 cublas
#   shared_gu n4 18.2 -> 16.0 / n8 17.7 -> 16.2 (64.8.1.12), n16 15.5 cublas, n32
#            20.0 -> 15.7 lt, n50 19.9 -> 15.7 lt, n64 20.1 -> 15.9 lt, n80 21.6 ->
#            16.6 (64.80.1.6), n96 21.5 -> 17.7 (64.96.1.5)
#   shared_down n4 11.1 -> 9.4 / n8 11.2 -> 9.4 (64.8.1.12), n16 11.5 -> 9.5
#            (64.16.1.11), n32 12.3 -> 9.8 (64.32.1.9), n50 13.3 -> 10.9 / n64 13.5
#            -> 11.0 (64.64.1.7), n80 11.5 / n96 11.3 cublas
# Bands split midway between measured points; rows past the last band keep the
# family's GENERIC rule. ``TUNED_WARP`` is empty: "warp" rows take ``warp_tactic``
# (the tactic measured). Every kind sums K in FP32 and rounds once to BF16.
# R65-96 (SPLHF D13, 2026-11-23, same chained method, Rhea, us per call cublas / lt /
# best split-K): dsa_in R65-80 10.4 / 9.9 / 9.9 (64.80.2.5), R84-96 10.45 / 10.3 /
# 10.8; o_proj R65-80 8.8 / 6.45 / 7.2 (64.80.1.5), R84-96 8.8 / 9.1 / 7.7 (64.96.1.5:
# not taken, SpOproj's partial GEMM is torch kinds only); wq_b R65-80 3.8 / 3.7, R84-96
# 3.8 / 3.9 (unchanged: within 0.1 us). latent_up n24-41 cublas 6.75-6.88 / lt
# 6.98-7.08: the cuBLAS band 25-40 stands.
TUNED_KIND: dict[tuple[int, int], dict] = {
    (12448, 7168): {"*": "1-16:cublas,17-164:lt,165-228:lt,229-288:cublas,"
                         "289-352:lt,353-:cublas"},
    (5344, 7168): {"*": "65-80:lt,81-164:cublas,165-352:lt,353-:cublas"},
    (4096, 1536): {"*": "1-64:split_k,65-228:lt,229-288:cublas,289-352:lt,"
                        "353-:cublas"},
    (7168, 3072): {"*": "65-80:lt,81-228:cublas,229-352:lt,353-:cublas"},
    (3584, 7168): {"*": "1-16:warp,17-80:split_k,81-:cublas"},
    (7168, 3584): {"*": "1-12:cublas,13-24:lt,25-40:cublas,41-72:lt,73-:cublas"},
    (12288, 7168): {"*": "1-12:split_k,13-24:cublas,25-64:lt,65-96:split_k,"
                         "97-:cublas"},
    (7168, 6144): {"*": "1-64:split_k,65-:cublas"},
}  # fmt: skip
# The side-stream shared expert (torch kinds; chained points above: shared_gu lt at
# n32-64, shared_down lt at n4-80 (n96 lt 16.5 vs cuBLAS 11.3)).
TUNED_SIDE: dict[tuple[int, int], dict] = {
    (12288, 7168): {"*": "25-64:lt,65-:cublas"},
    (7168, 6144): {"*": "1-88:lt,89-:cublas"},
}  # fmt: skip
TUNED_SPLIT: dict[tuple[int, int], dict] = {
    (4096, 1536): {"*": "1-8:128.8.3.4,9-16:64.16.2.6,17-32:64.32.2.6,"
                        "33-64:64.32.1.9"},
    (3584, 7168): {"*": "17-32:64.32.2.9,33-64:64.64.2.6,65-80:64.80.2.5"},
    (12288, 7168): {"*": "1-12:64.8.1.12,65-80:64.80.1.6,81-96:64.96.1.5"},
    (7168, 6144): {"*": "1-8:64.8.1.12,9-16:64.16.1.11,17-32:64.32.1.9,"
                        "33-64:64.64.1.7"},
}  # fmt: skip
TUNED_WARP: dict[tuple[int, int], dict] = {}


def _tuned(n, k, shape, sms=None, sm_bytes=None, table=None):
    """The measured tactic at ``shape`` for an [n, k] weight, or None."""
    table = TUNED_KIND if table is None else table
    kind = rows_policy(table.get((n, k), ""), shape, default=None)
    if kind is None:
        return None
    if kind == "warp":
        fields = rows_policy(TUNED_WARP.get((n, k), ""), shape, default=None)
        if fields is not None:
            return ("warp", WarpSplitKTactic(*fields))
        if sms is None:
            return None
        return ("warp", warp_tactic(shape.rows, n, k, sms, sm_bytes))
    if kind == "split_k":
        fields = rows_policy(TUNED_SPLIT.get((n, k), ""), shape, default=None)
        return None if fields is None else ("split_k", fields)
    return (kind, None) if kind in TORCH_KINDS else None


def measured(
    family: str,
    n: int,
    k: int,
    rows: int,
    tactic: ProjectionTactic,
    sms: int | None = None,
    sm_bytes: int | None = None,
) -> bool:
    """Whether ``tactic`` at ``rows`` for ``family``'s [n, k] weight is exactly the
    TUNED tactic of that family's table (``TUNED_SIDE`` for the side-stream kinds,
    ``TUNED_KIND`` otherwise; tables keyed "*"). A "warp" row without a
    ``TUNED_WARP`` entry needs ``sms`` / ``sm_bytes`` to derive the measured tactic."""
    if family not in FAMILIES:
        return False
    table = TUNED_SIDE if family in SIDE else TUNED_KIND
    tuned = _tuned(n, k, Shape(rows, 1), sms, sm_bytes, table)
    return tuned is not None and tuned == tuple(tactic)


def _generic(kind, n, k, shape, ctx, sm_bytes):
    if kind in TORCH_ONLY:
        return "cublas", None
    name = rows_policy(GENERIC_KIND, shape, name=f"{kind} kinds")
    if name == "warp":
        if not warp_eligible(n, k):
            raise Unsupported(
                kind, shape, f"warp split-K cannot take a [{n}, {k}] weight"
            )
        return "warp", warp_tactic(shape.rows, n, k, ctx.sms, sm_bytes)
    if name == "split_k":
        tactic = rows_policy(GENERIC_SPLIT, shape, name=f"{kind} split-K")
        problem = split_k_problem(tactic, k)
        if problem is not None:
            raise Unsupported(kind, shape, problem)
        return "split_k", tactic
    return "cublas", None


def die_local_tactics(
    kind: str, weight: torch.Tensor, tactics: dict[int, ProjectionTactic]
) -> dict[int, ProjectionTactic]:
    """``tactics`` with the die-local GEMM at every R up to its MAX_ROWS where
    ``weight`` is placed on the dies (``on_dies``) and ``kind`` is one of
    ``DIE_LOCAL_KINDS``; unchanged otherwise."""
    if kind not in DIE_LOCAL_KINDS or not getattr(weight, "on_dies", False):
        return tactics
    return {
        r: ("die_local", None) if r <= DIE_LOCAL_MAX_ROWS else tactic
        for r, tactic in tactics.items()
    }


def sp_projection_tactics(
    kind: str, n: int, k: int, rows: tuple[int, ...], ctx: Context
) -> dict[int, ProjectionTactic]:
    """A tactic for every R in ``rows`` for this rank's [n, k] ``kind`` weight
    (``KINDS``): the measured table's where it has one, the GENERIC rule's otherwise.
    Raises ``Unsupported`` outside the SP TP4 decode or when a row has no valid
    tactic (never a stock fallback)."""
    if kind not in FAMILIES:
        raise Unsupported(kind, None, f"no SP projection policy (kinds {KINDS})")
    if not ctx.sequence_parallel or ctx.tp != SP_TP:
        raise Unsupported(
            kind, None,
            f"the SP projection tables serve TP{SP_TP} sequence parallelism "
            f"(TP{ctx.tp}, sequence_parallel={ctx.sequence_parallel})",
        )  # fmt: skip
    if n <= 0 or k <= 0:
        raise Unsupported(kind, None, f"weight [{n}, {k}]")
    if not rows or min(rows) < 1:
        raise Unsupported(kind, None, f"rows {rows!r}")
    sm_bytes = smem_limits(torch.cuda.current_device()).sm
    tactics = {}
    for r in sorted(set(rows)):
        shape = Shape(r, ctx.q)
        table = TUNED_SIDE if kind in SIDE else None
        tactic = _tuned(n, k, shape, ctx.sms, sm_bytes, table)
        if tactic is not None and tactic[0] == "split_k":
            problem = split_k_problem(tactic[1], k)
            if problem is not None:
                raise Unsupported(kind, shape, f"TUNED {problem}")
        if tactic is not None and kind in TORCH_ONLY and tactic[0] not in TORCH_KINDS:
            raise Unsupported(kind, shape, f"{tactic[0]}: torch kinds only")
        tactics[r] = tactic or _generic(kind, n, k, shape, ctx, sm_bytes)
    return tactics
