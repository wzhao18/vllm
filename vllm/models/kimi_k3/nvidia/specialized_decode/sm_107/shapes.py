# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode shapes, row-keyed policies and the specialized decode's coverage plan.

A deployment verifies ``q`` tokens per request (1 + draft length; 1 without
speculation) and captures one FULL decode graph per token-row count M (``Shape``).
Each kernel family decides per shape whether it runs a tuned kernel, a generic one,
or a stock seam (``Choice``); the coverage plan records those decisions for every
captured shape, and a shape that some family cannot run stays on the stock forward.

Under sequence parallelism (``Context.sequence_parallel``: C512's TP4 = DCP4 decode
role) nothing stays stock silently: every captured shape is specialized, a family that
cannot serve a shape raises ``Unsupported`` at build, and the stock modules the design
keeps (embedding, AttnRes, MegaMoE, ...) are ``Path.DECLARED`` choices from
``DECLARED_STOCK`` (``Coverage.assert_complete``).
"""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from vllm.logger import init_logger

from .cutedsl.common.policy import batch_table

logger = init_logger(__name__)

# Rows covered by the tuned tables and the 64-row mailboxes (B8 x q8 verify steps).
TUNED_ROWS = 64


@dataclass(frozen=True)
class Shape:
    """One captured decode graph: ``rows`` token rows of ``rows // q`` requests."""

    rows: int
    q: int = 1

    @property
    def requests(self) -> int:
        return self.rows // self.q

    def __str__(self) -> str:
        return f"M{self.rows}(q{self.q})"


class Path(Enum):
    TUNED = "tuned"  # a measured table row
    GENERIC = "generic"  # a specialized kernel, correct for the shape, untuned
    STOCK = "stock"  # the family's stock seam inside the specialized graph
    DECLARED = "declared"  # a stock module the SP design keeps (not a fallback)


@dataclass(frozen=True)
class Choice:
    """A family's decision for one shape; ``tactic`` is family-specific (hashable),
    ``note`` names its provenance in the coverage log."""

    path: Path
    tactic: Any = None
    note: str = ""

    def __str__(self) -> str:
        return f"{self.path.value}({self.note})" if self.note else self.path.value


@dataclass(frozen=True)
class Context:
    """Deployment facts every family's ``choose`` may read (built once per process)."""

    arch: str  # "rubin" (compute capability 10.7) or "gb300"
    tp: int
    rank: int
    dcp: int
    sms: int
    q: int  # the deployment's verify length
    max_rows: int  # the largest captured decode graph: capacity above TUNED_ROWS
    max_context: int
    bf16_state: bool
    # Sequence parallelism (C512): the DP size, the expert-parallel ranks (TP x DP)
    # and the attention heads per TP rank (24 at C512; 0 = not set).
    dp: int = 1
    ep: int = 1
    sequence_parallel: bool = False
    heads_per_rank: int = 0

    @property
    def target(self) -> bool:
        """The deployments the no-fallback rule covers (NQ-NOFALLBACK): Rubin, TP16 =
        DCP16, DSpark verify (q > 1). There every captured shape runs the specialized
        step or the process refuses to start; elsewhere (GB300, TP8, no DCP) a family's
        stock seam stays a logged coverage choice."""
        return target_topology(self.arch == "rubin", self.tp, self.dcp, self.q)


class Unsupported(ValueError):
    """A family cannot serve a captured shape of the SP decode: raised at build (never
    a fallback to stock), naming the family, the shape and the cause."""

    def __init__(self, family: str, shape: "Shape | int | None", cause: str):
        self.family, self.shape, self.cause = family, shape, cause
        where = "every shape" if shape is None else str(shape)
        super().__init__(f"Kimi K3 SP decode: {family} at {where}: {cause}")


def shard_rows(shape: Shape, ctx: Context) -> int:
    """Rows of one rank's SP shard of ``shape`` (rank r owns global rows [r n, (r+1)
    n), ``sp_shard`` order): the captured rows are whole multiples of the TP size."""
    if ctx.tp <= 0 or shape.rows % ctx.tp:
        raise Unsupported(
            "sp_shard", shape, f"{shape.rows} rows do not split over TP{ctx.tp}"
        )
    return shape.rows // ctx.tp


def target_topology(rubin: bool, tp: int, dcp: int, q: int) -> bool:
    """``Context.target`` from its parts (callers without a ``Context``)."""
    return rubin and tp == 16 and dcp == 16 and q > 1


def config_target(vllm_config: Any, tp: int, rubin: bool = True) -> bool:
    """``target_topology`` of a deployment's config (callers built outside the
    decode's ``Context``: verification). q = the speculative tokens + 1."""
    spec = vllm_config.speculative_config
    q = (getattr(spec, "num_speculative_tokens", 0) or 0) + 1 if spec else 1
    dcp = vllm_config.parallel_config.decode_context_parallel_size
    return target_topology(rubin, tp, dcp, q)


class Uncovered(RuntimeError):
    """NQ-NOFALLBACK: a shape the deployment can hit that the specialized decode does
    not cover. Raised at startup (build, prepare or capture) instead of letting the
    shape run the stock path."""


def refuse(site: str, cause: str, shape: Any = None) -> None:
    """Raise ``Uncovered`` naming the site, the shape and the cause."""
    at = f" at {shape}" if shape is not None else ""
    raise Uncovered(
        f"Kimi K3 specialized decode: {site}{at} is not covered ({cause}); it would "
        "run the stock path, which the specialized decode never does silently "
        "(NQ-NOFALLBACK). Cover the shape, or drop it from the deployment's captured "
        "sizes."
    )


def full_capture() -> bool:
    """Whether the current stream captures a FULL CUDA graph (the runner captures
    FULL graphs under ``CUDAGraphMode.NONE`` inside ``torch.cuda.graph``; PIECEWISE
    captures set PIECEWISE): the steps the specialized decode must serve."""
    import torch

    if not torch.cuda.is_current_stream_capturing():
        return False
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import get_forward_context, is_forward_context_available

    if not is_forward_context_available():
        return True
    mode = get_forward_context().cudagraph_runtime_mode
    return mode != CUDAGraphMode.PIECEWISE


def log_coverage(rank: int, family: str, text: str) -> None:
    """One coverage line outside the target step's plan (drafter, verification),
    with the same prefix as ``SpecializedDecode``'s (global rank 0 only)."""
    if rank == 0:
        logger.info("Kimi K3 decode coverage %s: %s", family, text)


class StockSteps:
    """NQ-NOFALLBACK: steps that run a whole stock forward at run time because no
    captured FULL graph covers them (mixed prefill + decode steps, partial drafts,
    eager steps). A run-time property of the schedule, not of a captured shape: each
    is counted per cause and logged at powers of two, so the exclusion is visible.

    Only serving steps count: the startup forwards (profile run, kernel warmup,
    PIECEWISE captures, the capture warmups and the post-capture profiling dummy
    runs) all come before the runner locks its workspace at the end of
    ``capture_model``, the runner's marker that capture is finished."""

    def __init__(self, name: str, rank: int):
        self.name, self.rank = name, rank
        self.counts: dict[str, int] = {}

    @staticmethod
    def serving() -> bool:
        """Whether the runner finished its (non-profiling) capture: it locks the
        workspace right after ``capture_model`` and never before."""
        from vllm.v1.worker.workspace import (
            current_workspace_manager,
            is_workspace_manager_initialized,
        )

        return (
            is_workspace_manager_initialized()
            and current_workspace_manager().is_locked()
        )

    def count(self, cause: str, rows: int) -> None:
        if not self.serving():
            return
        n = self.counts[cause] = self.counts.get(cause, 0) + 1
        if self.rank == 0 and n & (n - 1) == 0:
            logger.info(
                "Kimi K3 %s: stock step (%s), %d such steps so far (this one %d rows)",
                self.name, cause, n, rows,
            )  # fmt: skip


_MISSING = object()


def rows_policy(entry, shape: Shape, default: Any = _MISSING, name: str = "policy"):
    """The value of a rows-keyed policy ``entry`` at ``shape``.

    ``entry`` is a ``batch_table`` spec ("1-8:1,64:3.16.4.2"), the same for every q, or
    a dict {q | "*": spec} read at the exact q first, then at "*". A miss returns
    ``default``; without one it raises: the table must cover the shape but does not."""
    specs = entry if isinstance(entry, dict) else {"*": entry}
    for key in (shape.q, "*"):
        if key in specs:
            value = batch_table(specs[key]).get(shape.rows, _MISSING)
            if value is not _MISSING:
                return value
    if default is _MISSING:
        raise KeyError(f"{name}: no entry for {shape}")
    return default


def captured_rows(capture_sizes: Iterable[int], q: int, limit: int) -> tuple[int, ...]:
    """Token rows of the FULL decode graphs a deployment captures: the runner rounds
    each capture size up to whole q-token requests and drops those above ``limit``
    (max_num_seqs * q and the largest capture size)."""
    rows = {-(-size // q) * q for size in capture_sizes}
    return tuple(sorted(m for m in rows if m <= limit))


# Families the SP decode keeps as the stock module (``Path.DECLARED``): embedding,
# AttnRes, the MegaMoE / dense MLPs on the shard, the final row all-gather, the LM
# head, the sampler, verification under synthetic rejection and the DSpark drafter
# (phase 1).
DECLARED_STOCK = frozenset(
    {
        "embed",
        "attn_res",
        "moe",
        "dense",
        "final_gather",
        "lm_head",
        "sampler",
        "verify",
        "drafter",
    }
)


class Coverage:
    """Per captured shape, each family's ``Choice`` (``None`` for the whole shape: the
    step runs the stock forward, with the reason)."""

    def __init__(self, q: int):
        self.q = q
        self.plan: dict[int, dict[str, Choice] | None] = {}
        self.reasons: dict[int, str] = {}

    def specialize(self, rows: int, choices: dict[str, Choice]) -> None:
        self.plan[rows] = dict(choices)

    def stock(self, rows: int, reason: str) -> None:
        self.plan[rows], self.reasons[rows] = None, reason

    def declare(self, rows: int, family: str, note: str) -> None:
        """``family`` runs its stock module at ``rows`` by design (a ``Path.DECLARED``
        choice). A shape planned stock as a whole (``stock``) stays so:
        ``assert_complete`` reports it."""
        choices = self.plan.setdefault(rows, {})
        if choices is not None:
            choices[family] = Choice(Path.DECLARED, note=note)

    def assert_complete(
        self, captured: Iterable[int], declared: frozenset[str] = DECLARED_STOCK
    ) -> None:
        """Raises unless every captured shape is specialized: no shape missing or
        planned stock as a whole, no family on ``Path.STOCK``, and every
        ``Path.DECLARED`` family one of ``declared``."""
        problems = []
        for rows in sorted(set(captured)):
            shape = Shape(rows, self.q)
            if rows not in self.plan:
                problems.append(f"{shape}: not planned")
                continue
            choices = self.plan[rows]
            if choices is None:
                problems.append(f"{shape}: stock ({self.reasons.get(rows, '?')})")
                continue
            if not choices:
                problems.append(f"{shape}: no family planned")
            for name, choice in choices.items():
                if choice.path is Path.STOCK:
                    problems.append(f"{shape}: {name}={choice}")
                elif choice.path is Path.DECLARED and name not in declared:
                    problems.append(f"{shape}: {name}={choice} is not a declared stock")
        if problems:
            raise RuntimeError(
                "Kimi K3 SP decode: captured shapes not fully specialized:\n  "
                + "\n  ".join(problems)
            )

    def stock_rows(self) -> dict[int, str]:
        """Captured rows whose whole step runs the stock forward -> why."""
        return {rows: self.reasons[rows] for rows, c in self.plan.items() if c is None}

    def lines(self) -> list[str]:
        out = []
        for rows in sorted(self.plan):
            choices = self.plan[rows]
            shape = Shape(rows, self.q)
            if choices is None:
                out.append(f"{shape}: stock ({self.reasons[rows]})")
            else:
                body = " ".join(f"{name}={choice}" for name, choice in choices.items())
                out.append(f"{shape}: {body}")
        return out
