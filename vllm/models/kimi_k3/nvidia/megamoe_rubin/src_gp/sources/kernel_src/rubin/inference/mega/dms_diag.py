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
RING = 32
SLOTS = 64
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
S_PUB0, S_UNIT0 = 52, 56  # last 4 scheduler publishes: time (52..55) and unit code (56..59), index = n % 4
S_BLOCK, S_EPOCH = 62, 63


class _Bound:
    base = None


def register(device_workspace) -> None:
    device_workspace.register(
        REGION, cutlass.Int64, (RING * MAX_SM * SLOTS,), buffer_space="local", byte_alignment=128,
        reset="zero_on_first_allocate",
    )
    # D-MOE-STREAM reduce stamps: per reduce CTA (block index), same ring (epoch mod RING) as the main kernel.
    device_workspace.register(
        R_REGION, cutlass.Int64, (RING * R_MAX * R_SLOTS,), buffer_space="local", byte_alignment=128,
        reset="zero_on_first_allocate",
    )
    device_workspace.register(R_EPOCH, cutlass.Int32, (4,), buffer_space="local", byte_alignment=16,
                              reset="zero_on_first_allocate")


# Reduce ring (DIAG, D-MOE-STREAM): slot map per reduce CTA, decoded by runs-local/D-MOE-STREAM/rst.py
R_REGION = "genphase.diag.reduce_ring"
R_EPOCH = "genphase.diag.reduce_epoch"
R_MAX = 1024
R_SLOTS = 8
R_ENTRY_BYTES = R_MAX * R_SLOTS * 8
R_VALID, R_ENTRY, R_SYNC, R_CLUSTER, R_END, R_POLLS, R_SMID, R_STORE = 0, 1, 2, 3, 4, 5, 6, 7


class _RBound:
    base = None


def put_reduce_epoch(device_workspace, epoch) -> None:
    """Main kernel, every CTA's thread 0 at entry: the epoch its reduce files its stamps under."""
    if cutlass.const_expr(ON):
        _st(Int64(device_workspace.ptr(R_EPOCH).toint()), Int64(epoch), Int32(cute.arch.thread_idx()[0] == Int32(0)))


def rbind(ring_ptr_int: Int64, epoch_ptr, block: Int32) -> Int32:
    """Reduce kernel entry (every thread): this launch's ring entry, this CTA's slots. Returns the epoch."""
    epoch = cute.arch.load(epoch_ptr, Int32, sem="relaxed", scope="gpu")
    _RBound.base = (ring_ptr_int + Int64(epoch & Int32(RING - 1)) * Int64(R_ENTRY_BYTES)
                    + Int64(block) * Int64(R_SLOTS * 8))
    return epoch


def rput_gt(slot: int, pred) -> None:
    if cutlass.const_expr(ON):
        _st(_RBound.base + Int64(slot * 8), Int64(read_globaltimer_ns()), Int32(pred))


def rput_val(slot: int, value, pred) -> None:
    if cutlass.const_expr(ON):
        _st(_RBound.base + Int64(slot * 8), Int64(value), Int32(pred))


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
