"""M1 EXPERIMENT ONLY (never merge): in-kernel phase stamps for the prefill MegaMoE main kernel.

Compile-time switch: M1_STAMPS=1 in the environment when the kernel object is built (bench-only; the kernel
name gains ``_m1st`` so the compiled-kernel caches never mix the two). One lane per warp role writes into a
per-CTA record of a ring indexed by the kernel's call count (local workspace, zeroed once):

  m1.ctrl  int32[4]               [0] call id (advanced by the last CTA to finish), [1] CTAs done this call
  m1.hdr   int32[SLOTS][HDR]      [0] call id, [1] local experts, [2..2+E) tokens per local expert
  m1.rec   int64[SLOTS][MAXCTA][REC]  per-CTA words (W_* below): globaltimer ns, clock64 sums, counts
"""

from __future__ import annotations

import os

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import Int32, Int64, T, dsl_user_op, extract_mlir_values, new_from_mlir_values

ENABLED = os.environ.get("M1_STAMPS", "0") == "1"
# bisect aid: comma list of the roles that stamp (default all)
ROLES = tuple(os.environ.get("M1_ROLES", "sched,tokin,tmaa,tmab,mma,epi,tb").split(","))


def on(role: str) -> bool:
    return ENABLED and role in ROLES

SLOTS = 192
MAXCTA = 256
REC = 64
HDR = 128
MAX_HDR_EXPERTS = HDR - 2

CTRL_REGION = "m1.ctrl"
HDR_REGION = "m1.hdr"
REC_REGION = "m1.rec"

# ---- record words ------------------------------------------------------------------------------------------
# globaltimer (ns) events; the per-tile FIRST/LAST words of TMA-A, TMA-B, MMA and the epilogue hold %clock64 (cheap)
W_ENTRY = 0            # scheduler warp entry
W_SIZES_READY = 1      # scheduler: wait_for_sizes_ready returned (router metadata of every rank landed)
W_TOKIN_SIZES = 2      # token_in warp 0: sizes ready
W_TOKIN_META = 3       # token_in warp 0: metadata ready
W_TOKIN_END = 4        # token_in warp 0: this CTA's payload pulls published
W_TMAA_FIRST = 5       # TMA-A: first weight stage acquired (min)
W_TMAA_LAST = 6        # TMA-A: last weight stage issued (max)
W_TMAB_FIRST = 7       # TMA-B: first token tile past its readiness wait (min)
W_MMA_FIRST = 8        # MMA: first tile's accumulator acquired (min)
W_MMA_LAST = 9         # MMA: last tile committed (max)
W_EPI_FIRST = 10       # epilogue: first tile start (min)
W_EPI_LAST = 11        # epilogue: last tile end (max)
W_TB_START = 12        # standalone token_back: work start
W_TB_END = 13          # standalone token_back: work end (last combine push issued)
W_DRAIN_START = 14     # tail: before the cross-rank drain barrier (this CTA's work all done)
W_DRAIN_END = 15       # tail: drain barrier passed (every CTA of every rank arrived)
W_TAIL_END = 16        # tail: publish barrier passed
W_SCHED_END = 17       # scheduler: tail published
W_TMAB_LAST = 18       # TMA-B: last token stage issued (max)
# clock64 cycle sums
C_MMA_ABWAIT = 20      # MMA warp: waiting for a full AB stage (weights + tokens landed)
C_MMA_ACCWAIT = 21     # MMA warp: waiting for a free TMEM accumulator (epilogue behind)
C_MMA_SCHED = 22       # MMA warp: waiting for the next work tile
C_TMAA_ACQ = 23        # TMA-A: waiting for an empty AB stage (stream back-pressured)
C_TMAA_SCHED = 24      # TMA-A: waiting for the next work tile
C_TMAB_FC1WAIT = 25    # TMA-B: FC1 token rows not yet dispatched (fc1_ready)
C_TMAB_FC2WAIT = 26    # TMA-B: FC2 input not yet produced (fc1_done of the token block)
C_TMAB_ACQ = 27        # TMA-B: waiting for an empty AB stage
C_TMAB_SCHED = 28      # TMA-B: waiting for the next work tile
C_EPI_ACCWAIT = 29     # epilogue: waiting for the MMA accumulator
C_EPI_TILE = 30        # epilogue: tile span (accumulator wait + epilogue work + drain + tile barrier)
C_EPI_SCHED = 31       # epilogue: waiting for the next work tile
C_TB_PREFIXWAIT = 32   # token_back: waiting for the fc2_done expert prefix
C_TB_PUSH = 33         # token_back: reduce + push of multi-copy rows
C_MMA_ABWAIT_FC2 = 34  # subset of C_MMA_ABWAIT on FC2 tiles
C_TOKIN_PULL = 35      # token_in warp 0: payload pull span (clock)
C_TMAA_ACQ_FC2 = 36    # subset of C_TMAA_ACQ on FC2 tiles
# counts
N_MMA_TILES = 40
N_FC1_TILES = 41
N_FC2_TILES = 42
N_TOKENS = 43          # sum of valid tokens of the CTA tiles the MMA ran (FC1 + FC2)
N_KTILES = 44
N_TOKENS_FC1 = 45
N_TMAA_TILES = 46
N_TMAB_TILES = 47
# clocks
CLK_ENTRY = 48
CLK_TAIL = 49
SMID = 50

ROLE_WORDS = {
    "sched": (W_ENTRY, W_SIZES_READY, W_SCHED_END, CLK_ENTRY, SMID),
    "tokin": (W_TOKIN_SIZES, W_TOKIN_META, W_TOKIN_END, W_DRAIN_START, W_DRAIN_END, W_TAIL_END, CLK_TAIL,
              C_TOKIN_PULL),
    "tmaa": (W_TMAA_LAST, C_TMAA_ACQ, C_TMAA_SCHED, N_TMAA_TILES, C_TMAA_ACQ_FC2),
    "tmab": (W_TMAB_LAST, C_TMAB_FC1WAIT, C_TMAB_FC2WAIT, C_TMAB_ACQ, C_TMAB_SCHED, N_TMAB_TILES),
    "mma": (W_MMA_LAST, C_MMA_ABWAIT, C_MMA_ACCWAIT, C_MMA_SCHED, N_MMA_TILES, N_FC1_TILES, N_FC2_TILES,
            N_TOKENS, N_KTILES, C_MMA_ABWAIT_FC2, N_TOKENS_FC1),
    "epi": (W_EPI_LAST, C_EPI_ACCWAIT, C_EPI_TILE, C_EPI_SCHED),
    "tb": (W_TB_START, W_TB_END, C_TB_PREFIXWAIT, C_TB_PUSH),
}
# "first" words start at INT64_MAX and take red.min
ROLE_MIN_WORDS = {
    "tmaa": (W_TMAA_FIRST,),
    "tmab": (W_TMAB_FIRST,),
    "mma": (W_MMA_FIRST,),
    "epi": (W_EPI_FIRST,),
}


def register(workspace) -> None:
    workspace.register(CTRL_REGION, cutlass.Int32, (4,), buffer_space="local", byte_alignment=16,
                       reset="zero_on_first_allocate")
    workspace.register(HDR_REGION, cutlass.Int32, (SLOTS * HDR,), buffer_space="local", byte_alignment=16,
                       reset="zero_on_first_allocate")
    workspace.register(REC_REGION, cutlass.Int64, (SLOTS * MAXCTA * REC,), buffer_space="local",
                       byte_alignment=16, reset="zero_on_first_allocate")


# ---- PTX ---------------------------------------------------------------------------------------------------
@dsl_user_op
def clk(*, loc=None, ip=None) -> Int64:
    return Int64(llvm.inline_asm(T.i64(), [], "mov.u64 $0, %clock64;", "=l", has_side_effects=True,
                                 asm_dialect=0, loc=loc, ip=ip))


@dsl_user_op
def gtime(*, loc=None, ip=None) -> Int64:
    return Int64(llvm.inline_asm(T.i64(), [], "mov.u64 $0, %globaltimer;", "=l", has_side_effects=True,
                                 asm_dialect=0, loc=loc, ip=ip))


@dsl_user_op
def smid(*, loc=None, ip=None) -> Int32:
    return Int32(llvm.inline_asm(T.i32(), [], "mov.u32 $0, %smid;", "=r", has_side_effects=True,
                                 asm_dialect=0, loc=loc, ip=ip))


@dsl_user_op
def ld_volatile_s32(address: Int64, *, loc=None, ip=None) -> Int32:
    return Int32(llvm.inline_asm(T.i32(), [address.ir_value(loc=loc, ip=ip)], "ld.volatile.global.s32 $0, [$1];",
                                 "=r,l", has_side_effects=True, asm_dialect=0, loc=loc, ip=ip))


def _pred_asm(op: str, address: Int64, value: Int64, pred: Int32, loc, ip) -> None:
    llvm.inline_asm(
        None,
        [address.ir_value(loc=loc, ip=ip), value.ir_value(loc=loc, ip=ip), pred.ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $2, 0;\n\t@p " + op + " [$0], $1;\n\t}",
        "l,l,r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_u64(address: Int64, value: Int64, pred: Int32, *, loc=None, ip=None) -> None:
    _pred_asm("st.relaxed.gpu.global.u64", address, value, pred, loc, ip)


@dsl_user_op
def red_add_u64(address: Int64, value: Int64, pred: Int32, *, loc=None, ip=None) -> None:
    _pred_asm("red.relaxed.gpu.global.add.u64", address, value, pred, loc, ip)


@dsl_user_op
def red_min_u64(address: Int64, value: Int64, pred: Int32, *, loc=None, ip=None) -> None:
    _pred_asm("red.relaxed.gpu.global.min.u64", address, value, pred, loc, ip)


@dsl_user_op
def red_max_u64(address: Int64, value: Int64, pred: Int32, *, loc=None, ip=None) -> None:
    _pred_asm("red.relaxed.gpu.global.max.u64", address, value, pred, loc, ip)


@dsl_user_op
def st_s32(address: Int64, value: Int32, pred: Int32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [address.ir_value(loc=loc, ip=ip), value.ir_value(loc=loc, ip=ip), pred.ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $2, 0;\n\t@p st.relaxed.gpu.global.s32 [$0], $1;\n\t}",
        "l,r,r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


class M1Writer:
    """One warp role's writer: ``rec`` is the Int64 address of this CTA's record of this call, ``pred`` the
    Int32 predicate (1 on the single writing lane)."""

    def __init__(self, rec: Int64, pred: Int32):
        self.rec = rec
        self.pred = pred

    def __extract_mlir_values__(self):
        return extract_mlir_values(self.rec) + extract_mlir_values(self.pred)

    def __new_from_mlir_values__(self, values):
        n = len(extract_mlir_values(self.rec))
        return M1Writer(new_from_mlir_values(self.rec, values[:n]), new_from_mlir_values(self.pred, values[n:]))

    def addr(self, word: int) -> Int64:
        return self.rec + Int64(word * 8)

    def st(self, word: int, value) -> None:
        st_u64(self.addr(word), Int64(value), self.pred)

    def add(self, word: int, value) -> None:
        red_add_u64(self.addr(word), Int64(value), self.pred)

    def vmin(self, word: int, value) -> None:
        red_min_u64(self.addr(word), Int64(value), self.pred)

    def vmax(self, word: int, value) -> None:
        red_max_u64(self.addr(word), Int64(value), self.pred)

    def init(self, role: str) -> None:
        for word in ROLE_WORDS[role]:
            self.st(word, Int64(0))
        for word in ROLE_MIN_WORDS.get(role, ()):
            self.st(word, Int64(0x7FFFFFFFFFFFFFFF))


def record_base(workspace, linear_cta_idx: Int32):
    """(slot, record address of this CTA) of the current call: every thread reads the call id at its start;
    it only advances after every CTA finished its tail (``finish``)."""
    ctrl = workspace.ptr(CTRL_REGION).toint()
    call = ld_volatile_s32(Int64(ctrl))
    slot = call % Int32(SLOTS)
    cta = cutlass.min(linear_cta_idx, Int32(MAXCTA - 1))
    rec = workspace.ptr(REC_REGION).toint() + (Int64(slot) * Int64(MAXCTA) + Int64(cta)) * Int64(REC * 8)
    return call, slot, Int64(rec)


def header_base(workspace, slot: Int32) -> Int64:
    return Int64(workspace.ptr(HDR_REGION).toint()) + Int64(slot) * Int64(HDR * 4)


@cute.jit
def finish(workspace, cta_count: int, writer_pred: Int32) -> None:
    """Last CTA of the call advances the call id (after every CTA's tail)."""
    if writer_pred != Int32(0):
        ctrl = workspace.ptr(CTRL_REGION)
        done = cute.arch.atomic_add(ctrl + Int32(1), Int32(1), sem="relaxed", scope="gpu")
        if done == Int32(cta_count - 1):
            cute.arch.atomic_add(ctrl + Int32(1), Int32(-cta_count), sem="relaxed", scope="gpu")
            cute.arch.atomic_add(ctrl, Int32(1), sem="relaxed", scope="gpu")
