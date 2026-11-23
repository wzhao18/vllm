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

from vllm.models.kimi_k3.nvidia.guard_global import read_global_u64
from cutlass.cutlass_dsl import Int32, Int64, T, dsl_user_op

# 15 min: far above any legitimate wait (peers compiling or loading before their first
# launch take minutes at most), far below a forever-hang.
SPIN_GUARD_NS = 900 * 1000 * 1000 * 1000  # built-in default; runtime bound: guard_config.py (int64.3)


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
def _trap_now(*, loc=None, ip=None) -> None:
    llvm.inline_asm(None, [], "trap;", "", has_side_effects=True, asm_dialect=0, loc=loc, ip=ip)


# A trap from any thread kills the whole grid at once, including another thread's record stores or printf still in
# flight (PIDX, 2026-11-22). Every timed-out thread therefore waits before it traps: long enough for the elected
# printer of this site (and any other site's printer timing out in the same window) to finish its record.
TRAP_DELAY_NS = 1000 * 1000 * 1000


@cute.jit
def _trap() -> None:
    start = globaltimer_ns()
    while globaltimer_ns() - start < cutlass.Uint64(TRAP_DELAY_NS):
        llvm.inline_asm(None, [], "nanosleep.u32 100000;", "", has_side_effects=True, asm_dialect=0)
    _trap_now()


@dsl_user_op
def _atom_or_b32(address, value: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(), [address.toint(loc=loc, ip=ip).ir_value(), value.ir_value()], "atom.global.or.b32 $0, [$1], $2;",
            "=r,l,r", has_side_effects=True, asm_dialect=0, loc=loc, ip=ip,
        )
    )


@cute.jit
def _elect_printer(site: int, print_flag) -> cutlass.Boolean:
    """One printer per rank and site (the primary fix against trap-line floods; a larger printf FIFO is only
    the backstop). With ``print_flag`` (an Int32 workspace word, never reset: a trap ends the context) the
    first timed-out thread that sets the site's bit prints; every other spinner traps silently. Without one
    (generic helpers that do not see the workspace), only lane 0 of CTA 0 prints."""
    elected = cutlass.Boolean(False)
    if cutlass.const_expr(print_flag is None):
        bidx, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        elected = (bidx == Int32(0)) & ((tidx % Int32(32)) == Int32(0))
    else:
        bit = Int32(1) << Int32(site % 32)
        elected = (_atom_or_b32(print_flag, bit) & bit) == Int32(0)
    return elected


# -- host-visible records (shared contract: vllm/models/kimi_k3/nvidia/spin_record.py) ----------------------------
# printf from a trapping kernel is only flushed if the host later synchronizes the context, so the elected printer also
# writes a record into the process's pinned host buffer (UVA address in a workspace word), fences at system scope and
# marks it done before the trap; the host watcher thread logs it even after the context is lost.
RECORD_WORDS = 64
RECORD_DONE = 0x5350494E
RECORD_SOURCE_MEGAMOE = 1
RECORD_MAX_ENTRIES = (RECORD_WORDS - 10) // 4


@dsl_user_op
def _st_sys_s32(address: Int64, value: Int32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None, [address.ir_value(), value.ir_value()], "st.relaxed.sys.global.s32 [$0], $1;", "l,r",
        has_side_effects=True, asm_dialect=0, loc=loc, ip=ip,
    )


@dsl_user_op
def _st_sys_v4_s32(address: Int64, v0: Int32, v1: Int32, v2: Int32, v3: Int32, *, loc=None, ip=None) -> None:
    """One 16-byte relaxed system-scope store (``address`` 16-byte aligned)."""
    llvm.inline_asm(
        None, [address.ir_value(), v0.ir_value(), v1.ir_value(), v2.ir_value(), v3.ir_value()],
        "st.relaxed.sys.global.v4.s32 [$0], {$1, $2, $3, $4};", "l,r,r,r,r",
        has_side_effects=True, asm_dialect=0, loc=loc, ip=ip,
    )


@dsl_user_op
def _st_gpu_s32(address: Int64, value: Int32, *, loc=None, ip=None) -> None:
    """One 4-byte relaxed gpu-scope store (device memory: completes at L2, no system-scope wait)."""
    llvm.inline_asm(
        None, [address.ir_value(), value.ir_value()], "st.relaxed.gpu.global.s32 [$0], $1;", "l,r",
        has_side_effects=True, asm_dialect=0, loc=loc, ip=ip,
    )


@dsl_user_op
def _st_gpu_v4_s32(address: Int64, v0: Int32, v1: Int32, v2: Int32, v3: Int32, *, loc=None, ip=None) -> None:
    """One 16-byte relaxed gpu-scope store (``address`` 16-byte aligned, device memory)."""
    llvm.inline_asm(
        None, [address.ir_value(), v0.ir_value(), v1.ir_value(), v2.ir_value(), v3.ir_value()],
        "st.relaxed.gpu.global.v4.s32 [$0], {$1, $2, $3, $4};", "l,r,r,r,r",
        has_side_effects=True, asm_dialect=0, loc=loc, ip=ip,
    )


@dsl_user_op
def _cas_sys_b32(address: Int64, compare: Int32, value: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(), [address.ir_value(), compare.ir_value(), value.ir_value()],
            "atom.relaxed.sys.global.cas.b32 $0, [$1], $2, $3;", "=r,l,r,r",
            has_side_effects=True, asm_dialect=0, loc=loc, ip=ip,
        )
    )


@cute.jit
def _record_open(record, index: int) -> Int64:
    """Claim record ``index`` of the host buffer whose address the Int64 word ``record`` holds; 0 when
    there is no buffer or another thread already claimed it."""
    base = Int64(0)
    if cutlass.const_expr(record is not None):
        host = cute.arch.load(record, Int64, sem="relaxed", scope="gpu")
        if host != Int64(0):
            slot = host + Int64(index * RECORD_WORDS * 4)
            if _cas_sys_b32(slot, Int32(0), Int32(1)) == Int32(0):
                base = slot
    return base


@cute.jit
def _record_word(base: Int64, word: int, value: Int32) -> None:
    _st_sys_s32(base + Int64(word * 4), value)


@cute.jit
def _record_close(base: Int64) -> None:
    cute.arch.fence_acq_rel_sys()
    _st_sys_s32(base, Int32(RECORD_DONE))
    cute.arch.fence_acq_rel_sys()


@cute.jit
def _record_header(base: Int64, site: int, call: Int32, observed: Int32, target: Int32, aux: Int32) -> None:
    _record_word(base, 1, Int32(RECORD_SOURCE_MEGAMOE))
    _record_word(base, 2, Int32(site))
    _record_word(base, 4, call)
    _record_word(base, 5, observed)
    _record_word(base, 6, target)
    _record_word(base, 7, _smid())
    _record_word(base, 8, aux)


@cute.jit
def spin_guard(start_ns: cutlass.Uint64, site: int, word, target: Int32, print_flag=None, record=None) -> None:
    """Trap with a diagnostic once a spin on the Int32 ``word`` (a pointer) that started
    at ``start_ns`` exceeds ``SPIN_GUARD_NS``; ``site`` is a constant id of the wait."""
    if globaltimer_ns() - start_ns > read_global_u64("k31_megamoe_guard_ns", SPIN_GUARD_NS):
        if _elect_printer(site, print_flag):
            observed = cute.arch.load(word, Int32, sem="relaxed", scope="sys")
            base = _record_open(record, 1)
            if base != Int64(0):
                _record_header(base, site, Int32(-1), observed, Int32(target), Int32(0))
                _record_word(base, 9, Int32(0))
                _record_close(base)
            cute.printf(
                "MEGAMOE_SPIN_TIMEOUT site={} observed={} target={} smid={}",
                Int32(site), observed, Int32(target), _smid(),
            )
        _trap()


@cute.jit
def spin_guard_value(
    start_ns: cutlass.Uint64, site: int, observed: Int32, target: Int32, print_flag=None, record=None
) -> None:
    """``spin_guard`` for waits whose condition is not one Int32 word (flag sets, 64-bit tokens, polled
    planes): the caller passes the observed and target values it compares (or a count of what is missing)."""
    if globaltimer_ns() - start_ns > read_global_u64("k31_megamoe_guard_ns", SPIN_GUARD_NS):
        if _elect_printer(site, print_flag):
            base = _record_open(record, 1)
            if base != Int64(0):
                _record_header(base, site, Int32(-1), Int32(observed), Int32(target), Int32(0))
                _record_word(base, 9, Int32(0))
                _record_close(base)
            cute.printf(
                "MEGAMOE_SPIN_TIMEOUT site={} observed={} target={} smid={}",
                Int32(site), Int32(observed), Int32(target), _smid(),
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
    print_flag=None,
    record=None,
) -> None:
    """``spin_guard`` for a wait on one arrival per rank: on timeout, besides the count, print one line per
    rank whose bit is missing in ``arrived_mask`` (Int32 pointer, local) with its progress words read
    remotely (``progress``: this rank's local [issued, entered] pointer; the peer mapper gives the peer's)."""
    if globaltimer_ns() - start_ns > read_global_u64("k31_megamoe_guard_ns", SPIN_GUARD_NS):
        if _elect_printer(site, print_flag):
            mask = cute.arch.load(arrived_mask, Int32, sem="relaxed", scope="sys")
            observed = cute.arch.load(word, Int32, sem="relaxed", scope="sys")
            base_rec = _record_open(record, 0)
            if base_rec != Int64(0):
                _record_header(base_rec, site, Int32(expected), observed, Int32(target), mask)
            entries = Int32(0)
            cute.printf(
                "MEGAMOE_SPIN_TIMEOUT site={} observed={} target={} arrived_mask={} expected_call={} smid={}",
                Int32(site), observed, Int32(target), mask, Int32(expected), _smid(),
            )
            base = progress.toint()
            for rank in cutlass.range_constexpr(world_size):
                if ((mask >> Int32(rank)) & Int32(1)) == Int32(0):
                    issued = ld_relaxed_sys_s32(peer_rank_ptr_mapper.map(base, Int32(rank), Int64(0)))
                    entered = ld_relaxed_sys_s32(peer_rank_ptr_mapper.map(base, Int32(rank), Int64(4)))
                    verdict = Int32(3)
                    if issued < expected:
                        verdict = Int32(1)
                    elif entered < expected:
                        verdict = Int32(2)
                    if (base_rec != Int64(0)) & (entries < Int32(RECORD_MAX_ENTRIES)):
                        entry = base_rec + Int64(40) + Int64(entries) * Int64(16)
                        _st_sys_s32(entry, Int32(rank))
                        _st_sys_s32(entry + Int64(4), issued)
                        _st_sys_s32(entry + Int64(8), entered)
                        _st_sys_s32(entry + Int64(12), verdict)
                    entries = entries + Int32(1)
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
            if base_rec != Int64(0):
                _record_word(base_rec, 9, cutlass.min(entries, Int32(RECORD_MAX_ENTRIES)))
                _record_close(base_rec)
        _trap()
