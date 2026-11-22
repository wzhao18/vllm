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

    K31_SPIN_TIMEOUT kernel=<name> site=<file:line> rank=<r> size=<s> observed=<v>
        target=<v> smid=<sm> block=<x> thread=<t>

and traps: the hang becomes a CUDA error naming the kernel, the wait and the rank.
``site`` is the source location of the poll loop, taken at trace time (the first
frame outside the poll helpers), so every wait has a distinct id with no hand-kept
numbering.

A trap from any thread ends the grid at once and drops device printf output still in
flight in other threads (several threads time out in the same clock window). DSL
kernels have no static device globals for a one-printer flag, so the lowest active
lane of each timing-out warp prints, and every timing-out thread waits
``SPIN_FLUSH_NS`` before it traps: every line in flight reaches the host. (Kernels
with a workspace in scope, e.g. MegaMoE, elect one printer per rank and site and also
fill the host-visible record of spin_record.py.) Nothing the kernels compute changes (same polls, same order).

Scope: only spins written in the Kimi K3.1 kernels themselves are bounded and named
(custom flag / Lamport / counter polls, the kernels' own mbarrier waits incl. inline
PTX loops). Waits inside CUTLASS / CuTe library code (PipelineTmaAsync / Umma
consumer_wait, producer_acquire, producer_tail, cute.arch.mbarrier_wait,
cute.arch.cluster_wait) stay unbounded: a hang there is not named by this guard
(documented residual; such a hang usually follows a named spin on another warp/CTA)."""

import ast
import os
import sys

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint32, Uint64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

# 15 min: far above any legitimate wait (peers compiling, loading or capturing graphs),
# far below a forever-hang. Shorter only on never-merge stall-test overlays.
SPIN_GUARD_NS = 900 * 1000 * 1000 * 1000
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
    if warp_leader() != Int32(0):
        cute.printf(
            "K31_SPIN_TIMEOUT kernel=" + kernel + " site=" + site
            + " rank={} size={} observed={} target={} smid={} block={} thread={}",
            Int32(rank), Int64(size), Uint64(observed), Uint64(target), smid(),
            Int32(cute.arch.block_idx()[0]), Int32(cute.arch.thread_idx()[0]),
        )
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
        elif t - t0 > Uint64(SPIN_GUARD_NS):
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


def _bar_address(bar, loc, ip):
    """32-bit shared address of an mbarrier given as a smem Pointer or an address."""
    if hasattr(bar, "toint"):
        return Int32(bar.toint(loc=loc, ip=ip)).ir_value(loc=loc, ip=ip)
    return Uint32(bar).ir_value(loc=loc, ip=ip)


@dsl_user_op
def mbar_try_n(bar, parity, sem: str = "", hint: str = "", *, loc=None, ip=None) -> Uint32:
    """Up to ``SPIN_POLLS`` parity polls of a CTA mbarrier in one PTX loop (the
    unguarded wait's loop, with an exit after SPIN_POLLS failures); nonzero once the
    phase completed. ``sem``: e.g. ".acquire.cluster"; ``hint``: e.g. ", 10000000"
    (suspend-time hint)."""
    code = (
        "{ .reg .pred p; .reg .u32 n; mov.u32 n, " + str(SPIN_POLLS) + "; "
        "W_${:uid}: mbarrier.try_wait.parity" + sem + ".shared::cta.b64 p, [$1], $2"
        + hint + "; @p bra D_${:uid}; sub.u32 n, n, 1; setp.ne.u32 p, n, 0; "
        "@p bra W_${:uid}; D_${:uid}: mov.u32 $0, n; }"
    )
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [_bar_address(bar, loc, ip), Uint32(parity).ir_value(loc=loc, ip=ip)],
            code, "=r,r,r,~{memory}", has_side_effects=True, asm_dialect=0,
            loc=loc, ip=ip,
        )
    )


@cute.jit
def _mbar_wait(bar, parity, sem: cutlass.Constexpr[str], hint: cutlass.Constexpr[str],
               kernel: cutlass.Constexpr[str], site: cutlass.Constexpr[str]):
    left = mbar_try_n(bar, parity, sem, hint)
    n, t0 = spin_start()
    while left == Uint32(0):
        left = mbar_try_n(bar, parity, sem, hint)
        n, t0 = _tick(n, t0, kernel, site, Uint32(parity) ^ Uint32(1), parity, -1, -1,
                      SPIN_POLLS)


def mbar_wait(bar, parity, sem="", hint="", kernel=None, site=None):
    """Bounded parity wait on a CTA mbarrier (smem Pointer or 32-bit address): the
    same try_wait polls as an unguarded loop; the clock is read once per SPIN_POLLS
    failed polls."""
    if site is None:
        site = caller_site()
    if kernel is None:
        kernel = site.rsplit(".py:", 1)[0]
    return _mbar_wait(bar, parity, sem, hint, kernel, site)
