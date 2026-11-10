# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode projection tactics (``decode.DecodeProjection``) of the EP_GROUP attention
blocks: C512's sequence-parallel decode role (TP4 = DCP4 = SP4, 24 of 96 heads per
rank, ``runtime.sp_attention``) and the replicated EP_GROUP step (TP4 / TP8 / TP16,
DCP 1 or TP: h = 24 / 12 / 6 heads per rank, ``runtime.group_ep``).

Per-rank BF16 weights [N, K], derived from the model code (K3.1: hidden 7168, q-LoRA
1536, kv-LoRA 512 + rope 64, 96 heads of 128 (qk 128 nope + 64 rope), 32 replicated
indexer heads of 128, 96 KDA heads of 128); every caller reads N and K from the
loaded weight, these are the shapes per heads h = 96 / G:

  kind    N x K at h 24 / 12 / 6       rows
  dsa_in  5344 / 3808 / 3040 x 7168    q_a 1536 | kv_a 576 | gate h x 128 | wk 128 |
                                       wv 32 (``providers.pack_dsa_projection``)
  kda_in  12448 / 6288 / 3216 x 7168   q, k, v, gate 4 x h x 128 | f_a 128 | beta h |
                                       pad rows (``kda.py``: 16-row BF16 alignment)
  q_b     4608 / 2304 / 1152 x 1536    h heads x 192 (MlaQuery's operand)
  o_proj  7168 x 3072 / 1536 / 768     h heads x 128 on K (row-parallel; ``oproj.sp``
                                       fuses the SP reduce-scatter; ``group_ep``
                                       writes the partial for stock's reduce-scatter)

DEP (``runtime.dep``: TP1, h = 96, ``Context.dep``, key (1, 1, N, K)): dsa_in 14560 x
7168, kda_in 49376 x 7168, q_b 18432 x 1536, o_proj 7168 x 12288 (the whole output,
no reduction).

Rows are the TP group's R = n q (n requests per DP group): 4..512 at C512, 1..64 in
the replicated step (``runtime.EP_GROUP_MAX_ROWS``), n <= 512 per rank at DEP
(``runtime.DEP_MAX_REQUESTS``).

GENERIC rule (every R a structurally valid tactic, at every group size and pitch):
warp split-K up to 16 rows (``decode.warp_tactic`` for the device's SM count and
per-SM shared memory), the cluster split-K (``GENERIC_SPLIT``: VR's TP16 bin
tactics, M tile 64, split 2) at 17-64, cuBLAS above 64 (UP GB300 TP8: split-K
re-reads the weight per 64-column tile and loses from 80 rows; cuBLAS is the stock
GEMM, a declared tactic). A weight warp split-K cannot take (K / 128 > 64: DEP's
o_proj, K = 12288) runs the cluster split-K at 1-16 rows too (``NARROW_SPLIT``: M
tile 64, N tile 8 up to 8 rows, 16 up to 16, split 2). All of them sum K in FP32 and
round once to BF16 (the stock BF16 linear's precision; the order differs).

The measured tables are keyed by (G, DCP, N, K): N and K fix the heads per rank, G
and DCP the group's communication around the projection (TP8 DCP1 and TP8 = DCP8
share h = 12 and the weights, not the overlap). A row no table names runs the GENERIC
rule.

NQ-TUNE: no VR200 measurement at these shapes. Sweep per (G, DCP), R and kind: warp
(output / token tile, stages, B loader warps) at R <= 32, split-K (M tile 64 / 128,
N tile 8-64, split 2 / 4 / 7, stages) and the Rubin tensor-memory split-K
(``projection.compile_projection_tmem``: 128 x 64-96, split 6 / 7) at R <= 128,
cuBLAS vs cuBLASLt above; 1 GPU, cold weights, behind an AttnRes-like PDL
predecessor (VR's in-projection method), then fill ``TUNED_KIND`` / ``TUNED_SPLIT``
/ ``TUNED_WARP`` (rows-keyed batch_table specs, q-keyed where q matters). Compare
against stock nvjet + splitKreduce at the same R."""

import torch

from ...shapes import Context, Shape, Unsupported, rows_policy
from .decode import (
    SPLIT_K_ROWS,
    WARP_ROWS,
    ProjectionTactic,
    split_k_problem,
    warp_eligible,
    warp_tactic,
)
from .warp_splitk import WarpSplitKTactic, smem_limits

KINDS = ("dsa_in", "kda_in", "q_b", "o_proj")
# The attention-group sizes these tables serve (EP_GROUP: TP4 / TP8 / TP16).
GROUPS = (4, 8, 16)
# The TP size of C512's sequence-parallel decode role (``runtime.sp_attention``).
SP_TP = 4

# GENERIC: the kind per R, then the split-K tactic (M tile, N tile, split, stages) per
# R: VR's TP16 bins (``providers.projection_policies``: 32 rows (64, 32, 2, 8), 64 rows
# (64, 64, 2, WIDE_B64_STAGES)); the N tile repeats over rows inside a bin.
GENERIC_KIND = {"*": f"1-{WARP_ROWS}:warp,{WARP_ROWS + 1}-{SPLIT_K_ROWS}:split_k,"
                f"{SPLIT_K_ROWS + 1}-:cublas"}  # fmt: skip
GENERIC_SPLIT = {"*": f"{WARP_ROWS + 1}-32:64.32.2.8,33-{SPLIT_K_ROWS}:64.64.2.6"}
# GENERIC at 1-16 rows for a weight warp split-K cannot take (``warp_eligible``
# False: DEP's o_proj, K = 12288): the cluster split-K with the narrow N tiles.
# NQ-TUNE (B5): seeds are sm_100's DEP_OPROJ (only N tiles 8 / 16 / 32 / 64 are legal
# on sm_107) and the Rubin tensor-memory split-K.
NARROW_SPLIT = {"*": f"1-8:64.8.2.8,9-{WARP_ROWS}:64.16.2.8"}
# The DEP mode's (G, DCP) key of the tables (TP1, DCP1).
DEP_GROUP = (1, 1)

# NQ-TUNE (all three): measured VR200 tables by (G, DCP, N, K): {q | "*": spec}.
# ``TUNED_KIND``: "warp" / "split_k" / "cublas" per R; ``TUNED_SPLIT``: the split-K
# tactic per R (4 fields, or Rubin's tensor-memory fields); ``TUNED_WARP``: (output
# tile, token tile, K tile, stages, B loader warps) per R. A row a TUNED table names
# is ``Path.TUNED`` in the coverage log; every other row runs the GENERIC rule.
TUNED_KIND: dict[tuple[int, int, int, int], dict] = {}
TUNED_SPLIT: dict[tuple[int, int, int, int], dict] = {}
TUNED_WARP: dict[tuple[int, int, int, int], dict] = {}


def _tuned(key, shape):
    """The measured tactic at ``shape`` for the (G, DCP, N, K) ``key``, or None."""
    kind = rows_policy(TUNED_KIND.get(key, ""), shape, default=None)
    if kind is None:
        return None
    if kind == "warp":
        fields = rows_policy(TUNED_WARP.get(key, ""), shape, default=None)
        return None if fields is None else ("warp", WarpSplitKTactic(*fields))
    if kind == "split_k":
        fields = rows_policy(TUNED_SPLIT.get(key, ""), shape, default=None)
        return None if fields is None else ("split_k", fields)
    return ("cublas", None) if kind == "cublas" else None


def measured(
    n: int, k: int, rows: int, tactic: ProjectionTactic, group=None
) -> bool:
    """Whether ``tactic`` at ``rows`` for an [n, k] weight is a TUNED table row (at
    any q the tables key) of the (G, DCP) ``group`` (None: of any group)."""
    keys = {
        key
        for t in (TUNED_KIND, TUNED_SPLIT, TUNED_WARP)
        for key in t
        if key[2:] == (n, k) and (group is None or key[:2] == tuple(group))
    }
    for key in sorted(keys):
        qs = {q for t in (TUNED_KIND, TUNED_SPLIT, TUNED_WARP)
              for q in t.get(key, {}) if q != "*"}  # fmt: skip
        for q in (*sorted(qs), 1):
            if _tuned(key, Shape(rows, q)) == tactic:
                return True
    return False


def _generic(kind, n, k, shape, ctx, sm_bytes):
    name = rows_policy(GENERIC_KIND, shape, name=f"{kind} kinds")
    split = GENERIC_SPLIT
    if name == "warp":
        if warp_eligible(n, k):
            return "warp", warp_tactic(shape.rows, n, k, ctx.sms, sm_bytes)
        # (the narrow split-K: every EP_GROUP weight is warp-eligible, so their
        # tactics do not change)
        name, split = "split_k", NARROW_SPLIT
    if name == "split_k":
        tactic = rows_policy(split, shape, name=f"{kind} split-K")
        problem = split_k_problem(tactic, k)
        if problem is not None:
            raise Unsupported(kind, shape, problem)
        return "split_k", tactic
    return "cublas", None


def sp_projection_tactics(
    kind: str, n: int, k: int, rows: tuple[int, ...], ctx: Context
) -> dict[int, ProjectionTactic]:
    """A tactic for every R in ``rows`` for this rank's [n, k] ``kind`` weight
    (``KINDS``): the measured table's of (G = TP, DCP, n, k) where it has one, the
    GENERIC rule's otherwise. Raises ``Unsupported`` outside an EP_GROUP decode step
    (``Context.ep_group_step``: the SP step or the replicated one) of a ``GROUPS``
    size and outside DEP (``Context.dep`` at TP1 = DCP1, key ``DEP_GROUP``), or when
    a row has no valid tactic (never a stock fallback)."""
    if kind not in KINDS:
        raise Unsupported(kind, None, f"no SP projection policy (kinds {KINDS})")
    group = ctx.ep_group_step and ctx.tp in GROUPS
    dep = ctx.dep and not ctx.ep_group_step and (ctx.tp, ctx.dcp) == DEP_GROUP
    if not (group or dep):
        raise Unsupported(
            kind, None,
            f"the EP_GROUP projection tables serve TP{'/'.join(map(str, GROUPS))} "
            f"attention groups under expert parallelism, the DEP ones TP1 = DCP1 "
            f"(TP{ctx.tp} DCP{ctx.dcp}, ep_group={ctx.ep_group}, "
            f"sequence_parallel={ctx.sequence_parallel}, dep={ctx.dep})",
        )  # fmt: skip
    if n <= 0 or k <= 0:
        raise Unsupported(kind, None, f"weight [{n}, {k}]")
    if not rows or min(rows) < 1:
        raise Unsupported(kind, None, f"rows {rows!r}")
    sm_bytes = smem_limits(torch.cuda.current_device()).sm
    key = (ctx.tp, ctx.dcp, n, k)
    tactics = {}
    for r in sorted(set(rows)):
        shape = Shape(r, ctx.q)
        tactic = _tuned(key, shape)
        tactics[r] = tactic or _generic(kind, n, k, shape, ctx, sm_bytes)
    return tactics
