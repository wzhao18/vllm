"""Bounded cross-rank spin waits (vLLM-local, BUG-E2E-3).

Every cross-rank wait of the MegaMoE kernel (the NVLink barrier signal and the
router sizes / metadata ready words) spins until peer ranks publish. If a peer never
publishes, the kernel used to spin forever and the whole EP group hung with the host
blocked in a stream sync. ``spin_guard`` bounds every such spin: after
``SPIN_GUARD_NS`` without reaching the target it prints the wait site, the observed
and target values and the SM id, then traps, so the failure surfaces as a CUDA error
with a diagnostic instead of a silent hang. No effect on any value the kernel
computes (the waits read the same words in the same order).
"""

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import Int32, Int64, T, dsl_user_op

# 15 min: far above any legitimate wait (peers compiling or loading before their first
# launch take minutes at most), far below a forever-hang.
SPIN_GUARD_NS = 900 * 1000 * 1000 * 1000


@dsl_user_op
def globaltimer_ns(*, loc=None, ip=None) -> cutlass.Uint64:
    return cutlass.Uint64(
        llvm.inline_asm(
            T.i64(), [], "mov.u64 $0, %globaltimer;", "=l", has_side_effects=True,
            asm_dialect=0, loc=loc, ip=ip,
        )
    )


@dsl_user_op
def _smid(*, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(), [], "mov.u32 $0, %smid;", "=r", has_side_effects=True,
            asm_dialect=0, loc=loc, ip=ip,
        )
    )


@dsl_user_op
def _trap(*, loc=None, ip=None) -> None:
    llvm.inline_asm(None, [], "trap;", "", has_side_effects=True, asm_dialect=0, loc=loc, ip=ip)


@cute.jit
def spin_guard(start_ns: cutlass.Uint64, site: int, word, target: Int32) -> None:
    """Trap with a diagnostic once a spin on the Int32 ``word`` (a pointer) that started
    at ``start_ns`` exceeds ``SPIN_GUARD_NS``; ``site`` is a constant id of the wait."""
    if globaltimer_ns() - start_ns > cutlass.Uint64(SPIN_GUARD_NS):
        cute.printf(
            "MEGAMOE_SPIN_TIMEOUT site={} observed={} target={} smid={}",
            Int32(site),
            cute.arch.load(word, Int32, sem="relaxed", scope="sys"),
            Int32(target),
            _smid(),
        )
        _trap()


@cute.jit
def spin_guard_value(start_ns: cutlass.Uint64, site: int, observed: Int32, target: Int32) -> None:
    """``spin_guard`` for waits whose condition is not one Int32 word (flag sets, 64-bit tokens, polled
    planes): the caller passes the observed and target values it compares (or a count of what is missing)."""
    if globaltimer_ns() - start_ns > cutlass.Uint64(SPIN_GUARD_NS):
        cute.printf(
            "MEGAMOE_SPIN_TIMEOUT site={} observed={} target={} smid={}",
            Int32(site),
            Int32(observed),
            Int32(target),
            _smid(),
        )
        _trap()


# -- missing-rank report (MEGAMOEFIX 2026-11-22) -------------------------------------------------------------
# Per rank and session, two Int32 progress words in the symmetric arena: [issued, entered]. ``issued`` is the
# host's count of launches it has issued (stamped by a 4-byte H2D copy on a side stream, eager launches only:
# graph replays do not advance it), ``entered`` the call id the router kernel stamps at entry. A cross-rank wait
# that times out reads every missing rank's words over NVLink and classifies it against this rank's call id:
#   issued < expected            -> H1: that rank's host never issued the call (host-side stall / launch-count skew)
#   entered < expected <= issued -> H2: issued, but its GPU never started the call (stuck in an earlier kernel)
#   entered >= expected          -> IN: it entered the call but has not published (stuck inside the router)


@dsl_user_op
def red_or_relaxed_sys_b32(address: Int64, value: Int32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None, [address.ir_value(), value.ir_value()], "red.relaxed.sys.global.or.b32 [$0], $1;", "l,r",
        has_side_effects=True, asm_dialect=0, loc=loc, ip=ip,
    )


@dsl_user_op
def ld_relaxed_sys_s32(address: Int64, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(), [address.ir_value()], "ld.relaxed.sys.global.s32 $0, [$1];", "=r,l",
            has_side_effects=True, asm_dialect=0, loc=loc, ip=ip,
        )
    )


@cute.jit
def spin_guard_ranks(
    start_ns: cutlass.Uint64,
    site: int,
    word,
    target: Int32,
    arrived_mask,
    progress,
    peer_rank_ptr_mapper,
    world_size: cutlass.Constexpr[int],
    expected: Int32,
) -> None:
    """``spin_guard`` for a wait on one arrival per rank: on timeout, besides the count, print one line per
    rank whose bit is missing in ``arrived_mask`` (Int32 pointer, local) with its progress words read
    remotely (``progress``: this rank's local [issued, entered] pointer; the peer mapper gives the peer's)."""
    if globaltimer_ns() - start_ns > cutlass.Uint64(SPIN_GUARD_NS):
        mask = cute.arch.load(arrived_mask, Int32, sem="relaxed", scope="sys")
        cute.printf(
            "MEGAMOE_SPIN_TIMEOUT site={} observed={} target={} arrived_mask={} expected_call={} smid={}",
            Int32(site), cute.arch.load(word, Int32, sem="relaxed", scope="sys"), Int32(target), mask,
            Int32(expected), _smid(),
        )
        base = progress.toint()
        for rank in cutlass.range_constexpr(world_size):
            if ((mask >> Int32(rank)) & Int32(1)) == Int32(0):
                issued = ld_relaxed_sys_s32(peer_rank_ptr_mapper.map(base, Int32(rank), Int64(0)))
                entered = ld_relaxed_sys_s32(peer_rank_ptr_mapper.map(base, Int32(rank), Int64(4)))
                if issued < expected:
                    cute.printf(
                        "MEGAMOE_SPIN_MISSING site={} rank={} issued={} entered={} expected={} verdict=H1_host_never_issued",
                        Int32(site), Int32(rank), issued, entered, Int32(expected),
                    )
                elif entered < expected:
                    cute.printf(
                        "MEGAMOE_SPIN_MISSING site={} rank={} issued={} entered={} expected={} verdict=H2_gpu_stuck_before_call",
                        Int32(site), Int32(rank), issued, entered, Int32(expected),
                    )
                else:
                    cute.printf(
                        "MEGAMOE_SPIN_MISSING site={} rank={} issued={} entered={} expected={} verdict=IN_call_not_published",
                        Int32(site), Int32(rank), issued, entered, Int32(expected),
                    )
        _trap()
