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
from cutlass.cutlass_dsl import Int32, T, dsl_user_op

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
