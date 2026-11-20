"""DIAG ONLY (never lands; D-MOE-LADDER port of D-MOE-STREAM dms_diag): per-role %globaltimer / %clock64 stamps for the GenPhase kernel.

Enabled by DMS_STAMPS=1 at trace time. Each SM owns SLOTS Int64 slots in the local workspace region
REGION (indexed by %smid); roles add wait cycles with red.add.u64 from one lane.
"""

import os
from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import Int32, Int64, T, dsl_user_op

from ......common.ptx_helpers import read_clock64, read_globaltimer_ns, read_smid

ON = os.environ.get("DMS_STAMPS") == "1"
LAND = False
REGION = "genphase.diag.stamps"
SLOTS = 64
MAX_SM = 256


class _Bound:
    base = None


def register(device_workspace) -> None:
    if ON:
        device_workspace.register(
            REGION, cutlass.Int64, (MAX_SM * SLOTS,), buffer_space="local", byte_alignment=128,
            reset="zero_on_first_allocate",
        )


def bind(device_workspace) -> None:
    if ON:
        _Bound.base = device_workspace.ptr(REGION).toint()


@dsl_user_op
def _red_add(address: Int64, value: Int64, predicate: Int32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [address.ir_value(), value.ir_value(), predicate.ir_value()],
        "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $2, 0;\n\t@p red.relaxed.gpu.global.add.u64 [$0], $1;\n\t}",
        "l,l,r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


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
    return Int64(_Bound.base) + Int64(read_smid() * Int32(SLOTS * 8) + Int32(slot * 8))


@dsl_user_op
def _ld(address: Int64, *, loc=None, ip=None) -> Int64:
    return Int64(
        llvm.inline_asm(
            T.i64(), [address.ir_value()], "ld.volatile.global.u64 $0, [$1];", "=l,l",
            has_side_effects=True, asm_dialect=0, loc=loc, ip=ip,
        )
    )


def _addr_dyn(slot):
    return Int64(_Bound.base) + Int64(read_smid() * Int32(SLOTS * 8) + Int32(slot) * Int32(8))


S_STAGE0 = 48


def put_stage_issue(stage_index) -> None:
    """TMA-A: this SM's issue clock of ring stage `stage_index` (mod 16)."""
    if cutlass.const_expr(ON and LAND):
        _st(_addr_dyn(Int32(S_STAGE0) + Int32(stage_index) % Int32(16)), read_clock64(), lane0())


def stage_landing(stage_index, t_full):
    """MMA: clocks from this SM's TMA-A issue of `stage_index` to its full-barrier pass (0 when implausible)."""
    if cutlass.const_expr(ON and LAND):
        d = t_full - _ld(_addr_dyn(Int32(S_STAGE0) + Int32(stage_index) % Int32(16)))
        ok = (d > Int64(0)) & (d < Int64(100000))
        return Int64(ok) * d
    return Int64(0)


def lane0():
    return Int32(cute.arch.lane_idx() == Int32(0))


def clk():
    if cutlass.const_expr(ON):
        return read_clock64()
    return Int64(0)


def add(slot: int, value, pred=None) -> None:
    if cutlass.const_expr(ON):
        _red_add(_addr(slot), Int64(value), lane0() if pred is None else Int32(pred))


def add_since(slot: int, t0, pred=None) -> None:
    if cutlass.const_expr(ON):
        _red_add(_addr(slot), read_clock64() - t0, lane0() if pred is None else Int32(pred))


def put_gt(slot: int, pred=None) -> None:
    if cutlass.const_expr(ON):
        _st(_addr(slot), Int64(read_globaltimer_ns()), lane0() if pred is None else Int32(pred))


def put_clk(slot: int, pred=None) -> None:
    if cutlass.const_expr(ON):
        _st(_addr(slot), read_clock64(), lane0() if pred is None else Int32(pred))


# slot map (decoded by dms_stamps.py)
S_ENTRY, S_EXIT = 0, 1
S_SCHED0, S_SIZES, S_SCHED1 = 2, 3, 4
S_TA0, S_TA1, S_TA_EMPTY, S_TA_N, S_TA_CLAIM = 5, 6, 7, 8, 9
S_TB0, S_TB1, S_TB_EMPTY, S_TB_N, S_TB_DEPS1, S_TB_DEPS2 = 10, 11, 12, 13, 14, 15
S_MMA0, S_MMA1, S_MMA_FULL, S_MMA_N, S_MMA_ISSUE, S_MMA_ACC, S_MMA_CLK0, S_MMA_CLK1 = 16, 17, 18, 19, 20, 21, 22, 23
S_EPI0, S_EPI1, S_EPI_WAIT, S_EPI_N = 24, 25, 26, 27
S_G0, S_G1, S_G_BAR, S_G_N = 28, 29, 30, 31
S_MMA_TILES, S_TA_TILES, S_TB_CLAIM = 32, 33, 34
S_TA_TRY, S_TA_ISSUE, S_MMA_TRY, S_MMA_REL, S_MMA_CLAIM = 35, 36, 37, 38, 39
S_MMA_LAND = 40
