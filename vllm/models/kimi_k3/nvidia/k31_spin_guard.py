# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded spin waits for the Kimi K3.1 CuTe DSL kernels (the DSL counterpart of
k31_spin_guard.cuh; one helper for the specialized decode / prefill kernels and the
MegaMoE kernels).

A poll loop carries a failed-poll count and a clock start through ``spin_tick``:

    n, t0 = spin_start()
    while not done:
        ...                                   # the unchanged poll
        n, t0 = spin_tick(n, t0, observed, target, rank, size)

The clock is first read after ``SPIN_POLLS`` failed polls (a wait that resolves within
them never touches it), then every ``SPIN_POLLS``. A wait still pending
``SPIN_GUARD_NS`` after the first read prints

    K31_SPIN_TIMEOUT kernel=<name> site=<file:line>

and traps: the hang becomes a CUDA error naming the kernel, the wait and the rank.
``site`` is the source location of the poll loop, taken at trace time (the first
frame outside the poll helpers), so every wait has a distinct id with no hand-kept
numbering.

A trap from any thread ends the grid at once and drops device printf output still in
flight in other threads (several threads time out in the same clock window). DSL
kernels have no static device globals for a one-printer flag, so every timing-out
thread prints the (identical, constant) line and waits ``SPIN_FLUSH_NS`` before it
traps: every line in flight reaches the host. (Kernels with a workspace in scope,
e.g. MegaMoE, elect one printer per rank and site and also fill the host-visible
record of spin_record.py.) Nothing the kernels compute changes (same polls, same
order).

Scope: the spins written in the Kimi K3.1 kernels that wait on another thread's
progress through memory are bounded and named: cross-rank and cross-CTA flag,
Lamport-sentinel and counter polls (fabric counters, twoshot, sp_comm, DCP merges,
draft / verify exchanges, MoE tail / routing / experts, o-proj, GEMM split flags,
KDA fronts, latent-MoE tail, ...). Not guarded (documented residual):
- the decode mailbox helpers (common/mailbox.py poll / poll_warp: Lamport rows from
  peers): a guard there measured +0.26% decode step at B32 against a 0.2% budget.
  A dead peer is still named by the other guarded cross-rank waits of the step.
- intra-CTA / cluster mbarrier phase waits on TMA / MMA / warp completion, both the
  kernels' own (primitives.wait, the sparse MLA mbar_wait helpers, splitk, the
  fabric put-completion fabric.wait) and the
  CUTLASS / CuTe library ones (pipeline consumer_wait, producer_acquire,
  producer_tail, cute.arch.mbarrier_wait / cluster_wait). They complete unless the
  producing warp is stuck, and a producer stuck on a guarded spin traps, which ends
  the whole grid including these waiters. Guarding them costs: their loop sits on
  the critical path of every pipeline stage (decode step +0.25..1.9% measured,
  PIDX 2026-11-22), against a 0.2% budget."""

import ast
import os
import sys

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Uint32, Uint64

from vllm.models.kimi_k3.nvidia.guard_global import read_global_u64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

# 10 min: far above any legitimate wait (peers compiling, loading or capturing graphs),
# far below a forever-hang, and below MegaMoE's 15 min guard so a stuck non-MegaMoE wait
# reports on its own rank before healthy peers trap in MegaMoE (tests/models/kimi_k3/
# test_k31_spin_guard_order.py). Shorter only on never-merge stall-test overlays.
SPIN_GUARD_NS = 600 * 1000 * 1000 * 1000  # built-in default; runtime bound: guard_config.py (int64.3)
SPIN_POLLS = 1024  # failed polls before the first clock read
SPIN_FLUSH_NS = 1000 * 1000 * 1000  # print -> trap delay (lets every line land)

_HELPER_FILES: set[str] = {os.path.abspath(__file__)}
_HELPER_LINES: dict[str, list[tuple[int, int]]] = {}


def register_poll_helper(path: str, *functions: str) -> None:
    """Poll helpers (mailbox.poll, primitives.wait, ...) report their CALLER's
    location: every function of the file ``path``, or only the named ones. Named
    functions are matched by source line range (the DSL runs a loop body in a
    generated function, so frame names differ from the source's)."""
    path = os.path.abspath(path)
    if not functions:
        _HELPER_FILES.add(path)
        return
    with open(path) as f:
        tree = ast.parse(f.read())
    spans = [
        (node.lineno, node.end_lineno)
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in functions
    ]
    missing = set(functions) - {
        n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
    }
    if missing:
        raise ValueError(f"k31_spin_guard: no function {sorted(missing)} in {path}")
    _HELPER_LINES.setdefault(path, []).extend(spans)


def _is_helper(path: str, line: int) -> bool:
    if path in _HELPER_FILES:
        return True
    return any(a <= line <= b for a, b in _HELPER_LINES.get(path, ()))


def caller_site() -> str:
    """``dir/file.py:line`` of the first frame outside this module, the registered
    poll helpers and the DSL itself (trace time)."""
    f = sys._getframe(1)
    while f is not None:
        p = os.path.abspath(f.f_code.co_filename)
        if (
            not _is_helper(p, f.f_lineno)
            and f"{os.sep}cutlass{os.sep}" not in p
            and not p.startswith("<")
        ):
            where = f"{os.path.basename(os.path.dirname(p))}/{os.path.basename(p)}"
            return f"{where}:{f.f_lineno}"
        f = f.f_back
    return "?"


@dsl_user_op
def globaltimer_ns(*, loc=None, ip=None) -> Uint64:
    return Uint64(
        llvm.inline_asm(
            T.i64(), [], "mov.u64 $0, %globaltimer;", "=l", has_side_effects=True,
            asm_dialect=0, loc=loc, ip=ip,
        )
    )


@dsl_user_op
def smid(*, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(), [], "mov.u32 $0, %smid;", "=r", has_side_effects=True,
            asm_dialect=0, loc=loc, ip=ip,
        )
    )


@dsl_user_op
def warp_leader(*, loc=None, ip=None) -> Int32:
    """1 in the lowest active lane of the warp (one line per stuck warp), else 0."""
    return Int32(
        llvm.inline_asm(
            T.i32(), [],
            "{ .reg .b32 m, l; .reg .pred p; activemask.b32 m; mov.u32 l, %lanemask_lt;"
            " and.b32 m, m, l; setp.eq.u32 p, m, 0; selp.u32 $0, 1, 0, p; }",
            "=r", has_side_effects=True, asm_dialect=0, loc=loc, ip=ip,
        )
    )


@dsl_user_op
def nanosleep_us(*, loc=None, ip=None) -> None:
    llvm.inline_asm(None, [], "nanosleep.u32 1000;", "", has_side_effects=True,
                    asm_dialect=0, loc=loc, ip=ip)


@dsl_user_op
def trap(*, loc=None, ip=None) -> None:
    llvm.inline_asm(None, [], "trap;", "", has_side_effects=True, asm_dialect=0, loc=loc, ip=ip)


@cute.jit
def _timeout(kernel: cutlass.Constexpr[str], site: cutlass.Constexpr[str], observed,
             target, rank, size):
    # A constant line (no arguments): a printf with value arguments inlined at every
    # guarded poll cost +0.25% decode step (mailbox polls, PIDX 2026-11-22); the
    # process (rank) is named by the log line prefix. observed / target / rank / size
    # document the wait at the call site.
    cute.printf("K31_SPIN_TIMEOUT kernel=" + kernel + " site=" + site)
    t = globaltimer_ns()
    while globaltimer_ns() - t < Uint64(SPIN_FLUSH_NS):
        nanosleep_us()
    trap()


def spin_start():
    """(failed polls, clock start) of a new wait."""
    return Uint32(0), Uint64(0)


@cute.jit
def _tick(n, t0, kernel: cutlass.Constexpr[str], site: cutlass.Constexpr[str], observed,
          target, rank, size, polls: cutlass.Constexpr[int]):
    n = n + Uint32(polls)
    if n % Uint32(SPIN_POLLS) == Uint32(0):
        t = globaltimer_ns()
        # the clock starts once (%globaltimer is never 0): a wrap of the 32-bit poll
        # count (~430 s of 100 ns polls) must not restart it
        if t0 == Uint64(0):
            t0 = t
        elif t - t0 > read_global_u64("k31_guard_ns", SPIN_GUARD_NS):
            _timeout(kernel, site, observed, target, rank, size)
    return n, t0


def spin_tick(n, t0, observed=0, target=0, rank=-1, size=-1, kernel=None, site=None,
              polls=1):
    """Count ``polls`` (a divisor or multiple of SPIN_POLLS) failed polls of a wait;
    returns (n, t0). ``site`` defaults to the poll loop's source location
    (``caller_site``), ``kernel`` to its file."""
    if site is None:
        site = caller_site()
    if kernel is None:
        kernel = site.rsplit(".py:", 1)[0]
    return _tick(n, t0, kernel, site, observed, target, rank, size, polls)
