# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode projection tactics (``decode.DecodeProjection``) of the sequence-parallel
attention block (C512 decode role: TP4 = DCP4 = SP4, 24 of 96 heads per rank).

Per-rank BF16 weights [N, K] at TP4, derived from the model code (K3.1: hidden 7168,
q-LoRA 1536, kv-LoRA 512 + rope 64, 96 heads of 128 (qk 128 nope + 64 rope), 32
replicated indexer heads of 128, 96 KDA heads of 128); every caller reads N and K
from the loaded weight, these are the shapes the tables are keyed by:

  kind    N x K          rows
  dsa_in  5344 x 7168    q_a 1536 | kv_a 576 | gate 24 x 128 | wk 128 | wv 32
                         (``providers.pack_dsa_projection``; TP16: 3040)
  kda_in  12448 x 7168   q, k, v, gate 4 x 3072 | f_a 128 | beta 24 | 8 pad rows
                         (``kda.py``: 16-row BF16 alignment; TP16: 3216)
  q_b     4608 x 1536    24 heads x 192 (MlaQuery's operand)
  o_proj  7168 x 3072    24 heads x 128 on K (row-parallel; ``oproj.sp`` fuses the RS)

Rows are the TP group's R = 4..512 (n = R / 4 requests per DP group, n / 4 per GPU).

GENERIC rule (every R a structurally valid tactic): warp split-K up to 16 rows
(``decode.warp_tactic`` for the device's SM count and per-SM shared memory), the
cluster split-K (``GENERIC_SPLIT``: VR's TP16 bin tactics, M tile 64, split 2) at 17-64,
cuBLAS above 64 (UP GB300 TP8: split-K re-reads the weight per 64-column tile and
loses from 80 rows; cuBLAS is the stock GEMM, a declared tactic). All of them sum K
in FP32 and round once to BF16.

NQ-TUNE: no VR200 measurement at these shapes. Sweep per R (4-512, step 4) and kind:
warp (output / token tile, stages, B loader warps) at R <= 32, split-K (M tile 64 /
128, N tile 8-64, split 2 / 4 / 7, stages) and the Rubin tensor-memory split-K
(``projection.compile_projection_tmem``: 128 x 64-96, split 6 / 7) at R <= 128,
cuBLAS vs cuBLASLt above; 1 GPU, cold weights, behind an AttnRes-like PDL
predecessor (VR's in-projection method), then fill ``TUNED_KIND`` / ``TUNED_SPLIT``
/ ``TUNED_WARP`` (keyed by (N, K), rows-keyed batch_table specs, q-keyed where q
matters). Compare against stock nvjet + splitKreduce at the same R."""

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
# The TP size these tables are for (the SP decode role).
SP_TP = 4

# GENERIC: the kind per R, then the split-K tactic (M tile, N tile, split, stages) per
# R: VR's TP16 bins (``providers.projection_policies``: 32 rows (64, 32, 2, 8), 64 rows
# (64, 64, 2, WIDE_B64_STAGES)); the N tile repeats over rows inside a bin.
GENERIC_KIND = {"*": f"1-{WARP_ROWS}:warp,{WARP_ROWS + 1}-{SPLIT_K_ROWS}:split_k,"
                f"{SPLIT_K_ROWS + 1}-:cublas"}  # fmt: skip
GENERIC_SPLIT = {"*": f"{WARP_ROWS + 1}-32:64.32.2.8,33-{SPLIT_K_ROWS}:64.64.2.6"}

# NQ-TUNE (all three): measured VR200 tables by weight (N, K): {q | "*": spec}.
# ``TUNED_KIND``: "warp" / "split_k" / "cublas" per R; ``TUNED_SPLIT``: the split-K
# tactic per R (4 fields, or Rubin's tensor-memory fields); ``TUNED_WARP``: (output
# tile, token tile, K tile, stages, B loader warps) per R. A row a TUNED table names
# is ``Path.TUNED`` in the coverage log; every other row runs the GENERIC rule.
TUNED_KIND: dict[tuple[int, int], dict] = {}
TUNED_SPLIT: dict[tuple[int, int], dict] = {}
TUNED_WARP: dict[tuple[int, int], dict] = {}


def _tuned(n, k, shape):
    """The measured tactic at ``shape`` for an [n, k] weight, or None."""
    kind = rows_policy(TUNED_KIND.get((n, k), ""), shape, default=None)
    if kind is None:
        return None
    if kind == "warp":
        fields = rows_policy(TUNED_WARP.get((n, k), ""), shape, default=None)
        return None if fields is None else ("warp", WarpSplitKTactic(*fields))
    if kind == "split_k":
        fields = rows_policy(TUNED_SPLIT.get((n, k), ""), shape, default=None)
        return None if fields is None else ("split_k", fields)
    return ("cublas", None) if kind == "cublas" else None


def measured(n: int, k: int, rows: int, tactic: ProjectionTactic) -> bool:
    """Whether ``tactic`` at ``rows`` for an [n, k] weight is a TUNED table row (at
    any q the tables key)."""
    qs = {key for t in (TUNED_KIND, TUNED_SPLIT, TUNED_WARP)
          for key in t.get((n, k), {}) if key != "*"}  # fmt: skip
    for q in (*sorted(qs), 1):
        if _tuned(n, k, Shape(rows, q)) == tactic:
            return True
    return False


def _generic(kind, n, k, shape, ctx, sm_bytes):
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


def sp_projection_tactics(
    kind: str, n: int, k: int, rows: tuple[int, ...], ctx: Context
) -> dict[int, ProjectionTactic]:
    """A tactic for every R in ``rows`` for this rank's [n, k] ``kind`` weight
    (``KINDS``): the measured table's where it has one, the GENERIC rule's otherwise.
    Raises ``Unsupported`` outside the SP TP4 decode or when a row has no valid
    tactic (never a stock fallback)."""
    if kind not in KINDS:
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
        tactic = _tuned(n, k, shape)
        tactics[r] = tactic or _generic(kind, n, k, shape, ctx, sm_bytes)
    return tactics
