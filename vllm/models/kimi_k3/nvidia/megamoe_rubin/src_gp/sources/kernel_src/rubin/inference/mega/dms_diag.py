"""DIAG ONLY (never lands; D-MOE-LADDER in-step stamps): per-SM %globaltimer stamps of the GenPhase main kernel,
kept for the last RING launches (ring slot = tail epoch mod RING) so a step bench can read them after its replays.

Always on in this diagnostic branch. Each SM owns SLOTS Int64 slots per ring entry in the local workspace region
REGION (indexed by %smid). Stamps are plain stores of %globaltimer (ns); slot S_EPOCH holds the epoch + 1 so a reader
can tell which launch filled the entry. Read back by session.dms_dump (eager launches only).
"""

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import Int32, Int64, dsl_user_op

from ......common.ptx_helpers import read_globaltimer_ns, read_smid

ON = True
REGION = "genphase.diag.stamps_ring"
RING = 64
SLOTS = 96
MAX_SM = 256
ENTRY_BYTES = MAX_SM * SLOTS * 8

# slot map (decoded by runs/D-MOE-LADDER/dmlt.py)
S_ENTRY, S_EXIT = 0, 1
S_SCHED0, S_SIZES, S_SCHED1 = 2, 3, 4
S_TA0, S_TA1 = 5, 6
S_TB0, S_TB1 = 10, 11
S_MMA0, S_MMA1 = 16, 17
S_EPI0, S_EPI1 = 24, 25
S_G0, S_G1 = 28, 29
S_NPUB = 31
S_GDW, S_SYNC, S_CLB, S_TAILBAR, S_ARRIVE = 40, 41, 42, 43, 44
S_L_COMM, S_L_RESET, S_L_EPOCH, S_L_FSYS, S_L_SIG = 45, 46, 47, 48, 49
S_CLB2, S_ISLAST = 50, 51
S_FPUB0, S_FUNIT0 = 32, 36  # first 4 publishes (time, unit code)
S_TOKREADY, S_FIRSTROUTED = 26, 27
S_PUB0, S_UNIT0 = 52, 56  # last 4 scheduler publishes: time (52..55) and unit code (56..59), index = n % 4
S_BLOCK, S_EPOCH = 62, 63
S_TAU0, S_TAPH0 = 64, 80
S_DIE = 7  # die + 1 of this CTA (die-aware builds)  # TMA-A unit k start time / phase, k < 16
# comm grid (token_comm) CTAs: entry, previous-tail wait passed, role work done, refine done, exit, role, epoch tag
C_ENTRY, C_EPOK, C_ROLE, C_REFINE, C_EXIT, C_ROLEID, C_EPOCH = 12, 13, 14, 15, 18, 20, 21


class _Bound:
    base = None


def register(device_workspace) -> None:
    device_workspace.register(
        REGION, cutlass.Int64, (RING * MAX_SM * SLOTS,), buffer_space="local", byte_alignment=128,
        reset="zero_on_first_allocate",
    )


def bind(device_workspace, epoch) -> None:
    """Call once at kernel entry by every thread, before any stamp; `epoch` = this call's tail epoch."""
    ring = Int64(epoch & Int32(RING - 1))
    _Bound.base = Int64(device_workspace.ptr(REGION).toint()) + ring * Int64(ENTRY_BYTES)


@dsl_user_op
def _st(address: Int64, value: Int64, predicate: Int32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [address.ir_value(), value.ir_value(), predicate.ir_value()],
        "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $2, 0;\n\t@p st.global.u64 [$0], $1;\n\t}",
        "l,l,r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


def _addr(slot: int):
    return _Bound.base + Int64(read_smid() * Int32(SLOTS * 8) + Int32(slot * 8))


def lane0():
    return Int32(cute.arch.lane_idx() == Int32(0))


def put_gt(slot: int, pred=None) -> None:
    if cutlass.const_expr(ON):
        _st(_addr(slot), Int64(read_globaltimer_ns()), lane0() if pred is None else Int32(pred))


def put_val(slot: int, value, pred=None) -> None:
    if cutlass.const_expr(ON):
        _st(_addr(slot), Int64(value), lane0() if pred is None else Int32(pred))


def _addr_dyn(slot):
    return _Bound.base + Int64(read_smid() * Int32(SLOTS * 8) + Int32(slot) * Int32(8))


def put_gt_dyn(slot, pred=None) -> None:
    if cutlass.const_expr(ON):
        _st(_addr_dyn(slot), Int64(read_globaltimer_ns()), lane0() if pred is None else Int32(pred))


def put_val_dyn(slot, value, pred=None) -> None:
    if cutlass.const_expr(ON):
        _st(_addr_dyn(slot), Int64(value), lane0() if pred is None else Int32(pred))


def unit_code(tile):
    """expert | phase_and_flags << 16 | tile_m << 24 | tile_n << 36 | valid tokens << 48 (fields absent -> 0)."""
    def f(name, mask):
        v = getattr(tile, name, None)
        return Int64(0) if v is None else (Int64(v) & Int64(mask))
    return (f("expert_idx", 0xFFFF) | (f("phase_and_flags", 0xFF) << Int64(16)) | (f("tile_m_idx", 0xFFF) << Int64(24))
            | (f("tile_n_idx", 0xFFF) << Int64(36)) | (f("valid_tokens_in_cta_tile", 0xFFFF) << Int64(48)))
