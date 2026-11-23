# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Constants and inline-PTX helpers of the die-local decode GEMM (``cluster``)."""

from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

# L2 evict_first cache policy of the weight loads (streamed once a step)
EVICT_FIRST = 0x12F0000000000000
CTA_M = 128  # weight rows a CTA's MMA covers (MMA M)
CTA_K = 64
EPI_N = 16  # token columns per tensor-memory load
EPILOGUE_WARP = 4
OUT_BF16, OUT_BF16_ADD, OUT_F32 = 0, 1, 2


def _asm(result, operands: list, text: str, constraints: str, loc, ip):
    return llvm.inline_asm(
        result,
        operands,
        text,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def atom_add_u64(addr: Int64, value: Int64, *, loc=None, ip=None) -> Int64:
    """Add ``value`` to the 64-bit word at global address ``addr`` (relaxed, GPU
    scope); returns the old value."""
    operands = [
        Int64(addr).ir_value(loc=loc, ip=ip),
        Int64(value).ir_value(loc=loc, ip=ip),
    ]
    text = "atom.relaxed.gpu.global.add.u64 $0, [$1], $2;"
    return Int64(_asm(T.i64(), operands, text, "=l,l,l", loc, ip))


@dsl_user_op
def store_u64(addr: Int64, value: Int64, *, loc=None, ip=None) -> None:
    """Store the 64-bit ``value`` at global address ``addr`` (relaxed, GPU scope)."""
    operands = [
        Int64(addr).ir_value(loc=loc, ip=ip),
        Int64(value).ir_value(loc=loc, ip=ip),
    ]
    _asm(None, operands, "st.relaxed.gpu.global.u64 [$0], $1;", "l,l", loc, ip)


@dsl_user_op
def _smid(*, loc=None, ip=None) -> Int32:
    return Int32(_asm(T.i32(), [], "mov.u32 $0, %smid;", "=r", loc, ip))


@dsl_user_op
def _load_tmem16(addr: Int32, *, loc=None, ip=None):
    """16 FP32 tensor-memory columns of this lane from ``addr`` (32x32b.x16)."""
    regs = ",".join(f"${i}" for i in range(16))
    text = (
        f"tcgen05.ld.sync.aligned.32x32b.x16.b32 {{{regs}}},[$16]; "
        "tcgen05.wait::ld.sync.aligned;"
    )
    ty = llvm.StructType.get_literal([T.f32()] * 16)
    res = _asm(
        ty,
        [Int32(addr).ir_value(loc=loc, ip=ip)],
        text,
        ",".join(["=f"] * 16 + ["r"]),
        loc,
        ip,
    )
    return [
        Float32(llvm.extractvalue(T.f32(), res, [i], loc=loc, ip=ip)) for i in range(16)
    ]


@dsl_user_op
def _load_s32(addr: Int64, *, loc=None, ip=None) -> Int32:
    """The 32-bit word at global address ``addr`` of read-only data."""
    operands = [Int64(addr).ir_value(loc=loc, ip=ip)]
    return Int32(_asm(T.i32(), operands, "ld.global.nc.s32 $0, [$1];", "=r,l", loc, ip))
