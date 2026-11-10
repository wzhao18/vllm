# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The attention input projection of a decode step, one engine per prepared row count.

GB300 (UP), TP8 shard of the K3.1 KDA (6288 x 7168) and DSA (3808 x 7168)
projections, back-to-back launches over weights larger than L2, us:

  rows   KDA  split-K / warp / cuBLAS   DSA  split-K / warp / cuBLAS   (SOL KDA / DSA)
  1           14.54 / 13.41 / 16.89          10.93 /  8.60 / 12.23      11.02 / 6.68
  5           14.86 / 13.82 / 16.58          11.13 /  8.51 / 11.70      11.04 / 6.69
  10          15.30 / 14.64 / 16.89           9.82 /  8.91 / 11.54      11.05 / 6.70
  32          15.46 / 17.61 / 17.42          10.11 / 10.14 / 12.05      11.13 / 6.76
  64          16.58 /   -   / 16.18          11.95 /   -   / 11.75      11.23 / 6.85
  80          27.24 /   -   / 17.81          19.43 /   -   / 11.76      11.28 / 6.89
  160         30.72 /   -   / 21.81          27.72 /   -   / 12.67      11.55 / 7.10
  320         55.38 /   -   / 27.59          39.50 /   -   / 16.02      12.07 / 7.53

so up to 16 rows the warp split-K (many small co-resident CTAs keep the weight stream
at HBM speed), up to 64 the cluster split-K and past it cuBLAS (the split-K tiles stop
at 64 columns and re-read the weight per column tile).

VR port (C512 SP decode, Rubin): the caller names a tactic for every row count it
serves (``ProjectionTactic``; ``sp_policy.sp_projection_tactics``) and the engine is
built for exactly those. cuBLAS is a declared tactic, not a fallback; a row count
without a tactic, or a tactic the weight / operands cannot take, raises
``shapes.Unsupported`` at build (UP's warp -> cuBLAS switch on misaligned operands
is ``check_operands``, called once per buffer before capture). Every tactic sums K in
FP32 and rounds once to BF16 (the stock BF16 linear's precision; the order differs).

NQ-TUNE: no Rubin (SM107) TP4 numbers exist for any tactic here; the table above is
UP's GB300 TP8. Measure warp / split-K / cuBLAS per R at the C512 shapes
(``sp_policy`` docstring) and fill ``sp_policy``'s TUNED tables."""

from typing import Any

import torch

from ...shapes import Choice, Path, Unsupported
from .projection import PreparedProjection
from .splitk import CTA_K
from .warp_splitk import (
    _CTA_RESERVED_SMEM,
    _MAX_M,
    _SM_SMEM_BYTES,
    PreparedWarpSplitK,
    WarpSplitKTactic,
    _smem_bytes,
    check_tactic,
    smem_limits,
)

WARP_ROWS = 16  # warp split-K up to here
SPLIT_K_ROWS = 64  # cluster split-K up to here; cuBLAS past it

# ("warp", WarpSplitKTactic) | ("split_k", (M tile, N tile, split, stages[, Rubin
# tensor-memory fields])) | ("cublas", None)
ProjectionTactic = tuple[str, Any]
KINDS = ("warp", "split_k", "cublas")
# ``SplitKGemm``'s tiles (4-field tactics): weight rows per CTA, token columns per CTA
SPLIT_M_TILES, SPLIT_N_TILES = (64, 128), (8, 16, 32, 64)


def warp_tactic(m, n, k, sms, sm_bytes=_SM_SMEM_BYTES):
    """Warp split-K tactic of ``m`` rows of an [n, k] weight: 32-row output tiles
    when they fill at most two waves, the smallest token tile holding ``m``, 256-wide
    K tiles and the deepest ring that keeps every wave's CTAs resident on an SM of
    ``sm_bytes`` shared memory (``warp_splitk.smem_limits(device).sm``)."""
    output = 32 if n % 32 == 0 and n // 32 <= 2 * sms else 16
    token = 8 if m <= 8 else 16 if m <= 16 else 32
    k_tile = 256 if k % 256 == 0 and k // 256 != 2 else 128
    ctas = (n // output) * -(-m // token)
    resident = -(-ctas // sms)
    best = None
    for stages in range(2, min(16, k // k_tile) + 1):
        t = WarpSplitKTactic(output, token, k_tile, stages, 2)
        if resident * (_smem_bytes(t) + _CTA_RESERVED_SMEM) <= sm_bytes:
            best = t
    if best is None:
        best = WarpSplitKTactic(output, token, k_tile, 2, 2)
    check_tactic(best, n, k)
    return best


def warp_eligible(n, k):
    return n % 16 == 0 and k % 128 == 0 and k // 128 <= 64


def split_k_problem(tactic, k):
    """Why the split-K ``tactic`` cannot serve reduction length ``k`` (None: it can)."""
    if not isinstance(tactic, tuple) or len(tactic) < 4:
        return f"split-K tactic {tactic!r} is not (M tile, N tile, split, stages, ...)"
    if not all(isinstance(v, int) or v is None for v in tactic):
        return f"split-K tactic {tactic!r} has non-integer fields"
    m_tile, n_tile, split, stages = tactic[:4]
    if len(tactic) > 4:  # Rubin tensor-memory tactics: ``compile_projection_tmem``
        return None if split >= 1 and stages >= 1 else f"split-K tactic {tactic}"
    if m_tile not in SPLIT_M_TILES or n_tile not in SPLIT_N_TILES:
        return (
            f"split-K tiles {m_tile} x {n_tile} (supported {SPLIT_M_TILES} x "
            f"{SPLIT_N_TILES})"
        )
    if split < 1 or stages < 1 or k % (CTA_K * split):
        return f"split-K tactic {tactic}: K = {k} not a multiple of {CTA_K} x {split}"
    return None


class DecodeProjection:
    """``out[R, N] = x[R, K] @ weight[N, K]^T`` for each prepared R with the tactic the
    caller names for it: warp split-K (R <= 32), the cluster split-K
    (``PreparedProjection``) or cuBLAS (``torch.mm``). ``family`` names the projection
    in ``Unsupported`` messages. ``gather``: the AG-fused polled activation input
    (phase 1b, ``sp_comm.SpGatherSource``); not built in phase 1, so passing one
    raises."""

    def __init__(
        self,
        weight: torch.Tensor,
        rows: tuple[int, ...],
        tactics: dict[int, ProjectionTactic],
        *,
        gather: Any = None,
        family: str = "projection",
        group: tuple[int, int] | None = None,
    ):
        self.family = family
        # (G, DCP) of the measured tables this projection's coverage reads
        # (``sp_policy.measured``; None: any group's)
        self.group = group
        if weight.dim() != 2 or weight.dtype != torch.bfloat16:
            raise Unsupported(
                family, None,
                f"weight {tuple(weight.shape)} {weight.dtype}: a BF16 [N, K] matrix",
            )  # fmt: skip
        n, k = weight.shape
        if weight.stride() != (k, 1) or weight.data_ptr() % 32:
            raise Unsupported(
                family, None,
                f"weight [{n}, {k}] stride {weight.stride()} at "
                f"0x{weight.data_ptr():x}: packed rows, 32 B aligned",
            )  # fmt: skip
        if gather is not None:
            raise Unsupported(
                family, None, "the AG-fused polled input (phase 1b) is not built"
            )
        self.weight, self.gather = weight, gather
        self.rows = tuple(sorted(set(rows)))
        if not self.rows or self.rows[0] < 1:
            raise Unsupported(family, None, f"rows {rows!r}")
        missing = [r for r in self.rows if r not in tactics]
        if missing:
            raise Unsupported(family, missing[0], f"no tactic (rows {missing})")
        self.tactics = {r: tactics[r] for r in self.rows}
        self.kinds: dict[int, str] = {}
        warp: dict[int, WarpSplitKTactic] = {}
        split: dict[int, tuple] = {}
        cta_smem = smem_limits(weight.device.index).cta
        for r, entry in self.tactics.items():
            self.kinds[r] = kind = self._check(r, entry, n, k, cta_smem)
            if kind == "warp":
                warp[r] = entry[1]
            elif kind == "split_k":
                split[r] = entry[1]
        # Compiled before capture; codes are shared by (N, K, tactic) across layers.
        self.warp = PreparedWarpSplitK(weight, warp) if warp else None
        self.split = PreparedProjection(weight, split).prepare() if split else None
        prepared = {"warp": self.warp, "split_k": self.split, "cublas": self}
        self.engines = {r: prepared[kind] for r, kind in self.kinds.items()}

    def _check(self, r, entry, n, k, cta_smem):
        """``entry``'s kind, or raises naming why it cannot serve ``r`` rows."""
        if not isinstance(entry, tuple) or len(entry) != 2 or entry[0] not in KINDS:
            raise Unsupported(self.family, r, f"tactic {entry!r} (kinds {KINDS})")
        kind, tactic = entry
        if kind == "warp":
            if not isinstance(tactic, WarpSplitKTactic):
                raise Unsupported(self.family, r, f"warp tactic {tactic!r}")
            if r > _MAX_M or not warp_eligible(n, k):
                raise Unsupported(
                    self.family, r,
                    f"warp split-K serves R <= {_MAX_M}, N % 16 == 0, K % 128 == 0, "
                    f"K <= 8192 (weight [{n}, {k}])",
                )  # fmt: skip
            try:
                check_tactic(tactic, n, k, cta_smem)
            except ValueError as e:  # re-raised with the shape: never a fallback
                raise Unsupported(self.family, r, f"warp {tactic}: {e}") from e
        elif kind == "split_k":
            problem = split_k_problem(tactic, k)
            if problem is not None:
                raise Unsupported(self.family, r, problem)
        elif tactic is not None:
            raise Unsupported(self.family, r, f"cuBLAS takes no tactic ({tactic!r})")
        return kind

    def check_operands(self, x: torch.Tensor, out: torch.Tensor) -> None:
        """Build time, once per buffer: raises unless every prepared tactic takes
        ``x`` [rows, K] / ``out`` [rows, N] (BF16, packed rows) and their row slices
        ``[:R]``: ``rows`` is a prepared R or at least the largest one; warp split-K
        operands 32 B aligned, split-K operands 16 B aligned (TMA)."""
        n, k = self.weight.shape
        rows = x.shape[0] if x.dim() == 2 else -1
        problems = []
        if x.dtype != torch.bfloat16 or out.dtype != torch.bfloat16:
            problems.append(f"dtypes {x.dtype} / {out.dtype} (BF16)")
        if x.dim() != 2 or x.shape[1] != k or x.stride() != (k, 1):
            problems.append(f"x {tuple(x.shape)} stride {x.stride()}: packed [R, {k}]")
        if out.dim() != 2 or out.shape[1] != n or out.stride() != (n, 1):
            problems.append(
                f"out {tuple(out.shape)} stride {out.stride()}: packed [R, {n}]"
            )
        if out.dim() == 2 and out.shape[0] != rows:
            problems.append(f"x has {rows} rows, out {out.shape[0]}")
        if rows not in self.kinds and rows < self.rows[-1]:
            problems.append(f"{rows} rows: neither prepared nor >= {self.rows[-1]}")
        kinds = set(self.kinds.values())
        align = 32 if "warp" in kinds else 16 if "split_k" in kinds else 0
        for name, t in (("x", x), ("out", out)):
            if align and t.data_ptr() % align:
                problems.append(f"{name} at 0x{t.data_ptr():x}: {align} B aligned")
        if problems:
            raise Unsupported(self.family, None, "; ".join(problems))

    def cublas(self, x, out):
        return torch.mm(x, self.weight.t(), out=out)

    def run(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """``out = x @ weight^T`` with R = ``x.shape[0]``'s prepared tactic."""
        engine = self.engines.get(x.shape[0])
        if engine is None:
            raise Unsupported(
                self.family, x.shape[0], f"not prepared ({self.prepared()})"
            )
        if engine is self:
            return self.cublas(x, out)
        return engine.run(x, out)

    def prepared(self):
        return f"{len(self.rows)} prepared R in {self.rows[0]}-{self.rows[-1]}"

    def engine(self, rows):
        return self.kinds[rows]

    def choice(self, rows: int) -> Choice:
        """TUNED where ``sp_policy``'s measured table names this R's tactic, GENERIC
        otherwise (every kind, cuBLAS included, is a specialized-path decision)."""
        from .sp_policy import measured

        if rows not in self.kinds:
            raise Unsupported(self.family, rows, f"not prepared ({self.prepared()})")
        kind, tactic = self.tactics[rows]
        n, k = self.weight.shape
        tuned = measured(n, k, rows, (kind, tactic), self.group)
        path = Path.TUNED if tuned else Path.GENERIC
        return Choice(path, tactic=(kind, tactic), note=f"{kind} {n}x{k}")
