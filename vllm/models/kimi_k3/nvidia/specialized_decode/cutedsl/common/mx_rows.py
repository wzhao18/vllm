# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP8 rows written by the thread that holds 8 consecutive BF16 columns of a
row (4 adjacent lanes = one 32-wide MXFP8 block, 16 adjacent lanes = one 4-byte
word of the F8_128x4-swizzled E8M0 scales): the AttnRes kernels write the MXFP8
input of the MXFP8 projection that follows them (``attn_res.sp_fused``) instead of a
standalone quantization kernel over their BF16 output.

Numerics: ``ops.mxfp8_act`` (``act_quant_kernel`` mode 0, bitwise FlashInfer's
``mxfp8_quantize``): block amax of the BF16 values, ``e = ue8m0(amax * fp32(1/448))``
(rounded up, capped at 254), values ``* 2^(127 - e)`` to E4M3 RN satfinite.

Lamport words (``common.mailbox``): an MXFP8 row published into a Lamport box never
holds the box's sentinel. Data words: the sentinel 0x80000000 (FP8 bytes +0, +0, +0,
-0) is stored as 0 (``positive_zero32``: the -0 becomes +0, a GEMM operand sign of
zero). Scale words: their own region armed with ``SCALE_SENTINEL`` 0xFFFFFFFF (E8M0
0xFF is NaN; ue8m0 never produces it), so scale bytes are stored unchanged.

Every lane of a warp must call ``quantize8`` / ``publish`` / ``store`` together
(shuffles)."""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int64, Uint32

from vllm.models.kimi_k3.nvidia.k31_spin_guard import (
    register_poll_helper,
    spin_start,
    spin_tick,
)

from .mailbox import SENTINEL
from .primitives import asm, bits_float, multicast_words, pack_fp8, store_words

BLOCK = 32
SCALE_SENTINEL = 0xFFFFFFFF
INV_448 = 0x3B124925  # fp32(1 / 448), round to nearest


def scale_bytes(rows: int, width: int) -> int:
    """Bytes of the F8_128x4 scales of ``rows`` x ``width`` (whole 128-row tiles)."""
    return -(-rows // 128) * 128 * (width // BLOCK)


@asm(Uint32, pure=True)
def ue8m0(value: Float32):
    """FlashInfer ``float_to_ue8m0_fast`` (``ops.mxfp8_act.float_to_ue8m0``)."""
    code = (
        "{ .reg .pred pz, pm, pe, pt, po; .reg .u32 b, e, m, u, r; "
        "setp.le.f32 pz, $1, 0f00000000; mov.b32 b, $1; shr.b32 e, b, 23; "
        "and.b32 e, e, 255; and.b32 m, b, 0x7FFFFF; setp.ne.u32 pm, m, 0; "
        "selp.u32 u, 1, 0, pm; setp.eq.u32 pe, e, 0; setp.le.u32 pt, m, 0x400000; "
        "and.pred pt, pe, pt; @pt mov.u32 u, 0; add.u32 r, e, u; "
        "setp.gt.u32 po, r, 254; selp.u32 r, 254, r, po; selp.u32 $0, 0, r, pz; }"
    )
    return code, "=r,f", (value,)


@asm(Uint32, pure=True)
def inverse_scale_bits(e: Uint32):
    """``ue8m0_to_inv_scale``: 0 for e = 0, else 2^(127 - e) (0 below 2^-126)."""
    code = (
        "{ .reg .pred p; .reg .s32 n; sub.s32 n, 254, $1; max.s32 n, n, 0; "
        "shl.b32 n, n, 23; setp.eq.u32 p, $1, 0; selp.b32 $0, 0, n, p; }"
    )
    return code, "=r,r", (e,)


@asm(Float32, pure=True)
def max_abs(a: Float32, b: Float32):
    """fmaxf(a, |b|) (a NaN operand yields the other, as fmaxf)."""
    return "{ .reg .f32 t; abs.f32 t, $2; max.f32 $0, $1, t; }", "=f,f,f", (a, b)


@asm(Float32, pure=True)
def fmax_f32(a: Float32, b: Float32):
    """fmaxf (a NaN operand yields the other)."""
    return "max.f32 $0, $1, $2;", "=f,f,f", (a, b)


@asm(Uint32, pure=True)
def shfl_down(value: Uint32, delta: cutlass.Constexpr[int]):
    code = f"shfl.sync.down.b32 $0, $1, {delta}, 0x1f, 0xffffffff;"
    return code, "=r,r", (value,)


@asm(Uint32, pure=True)
def positive_zero32(word):
    code = "{ .reg .pred p; setp.eq.u32 p,$1,0x80000000; selp.b32 $0,0,$1,p; }"
    return code, "=r,r", (Uint32(word),)


@cute.jit
def quantize8(values):
    """The 8 BF16-exact FP32 ``values`` of this lane -> (data word 0, data word 1,
    the 4-byte scale word of this lane's 128 columns (meaningful in lanes
    0 and 16 of the warp)). Warp-collective."""
    amax = Float32(0)
    for i in cutlass.range_constexpr(8):
        amax = max_abs(amax, values[i])
    amax = fmax_f32(amax, cute.arch.shuffle_sync_bfly(amax, 1))
    amax = fmax_f32(amax, cute.arch.shuffle_sync_bfly(amax, 2))
    e = ue8m0(amax * bits_float(Uint32(INV_448)))
    inv = bits_float(inverse_scale_bits(e))
    q = cute.make_rmem_tensor(8, Float32)
    for i in cutlass.range_constexpr(8):
        q[i] = values[i] * inv
    w0 = pack_fp8((q[0], q[1], q[2], q[3]))
    w1 = pack_fp8((q[4], q[5], q[6], q[7]))
    word = e | (shfl_down(e, 4) << 8) | (shfl_down(e, 8) << 16)
    word = word | (shfl_down(e, 12) << 24)
    return positive_zero32(w0), positive_zero32(w1), word


@cute.jit
def quantize8_plain(values):
    """``quantize8`` for plain (non-Lamport) stores: (data word 0, data word 1, this
    lane's block scale) with ``ops.mxfp8_act``'s quant8_store bits (-0 bytes kept).
    Every lane of 4 adjacent ones holds the scale; lane & 3 == 0 stores it.
    Warp-collective."""
    amax = Float32(0)
    for i in cutlass.range_constexpr(8):
        amax = max_abs(amax, values[i])
    amax = fmax_f32(amax, cute.arch.shuffle_sync_bfly(amax, 1))
    amax = fmax_f32(amax, cute.arch.shuffle_sync_bfly(amax, 2))
    e = ue8m0(amax * bits_float(Uint32(INV_448)))
    inv = bits_float(inverse_scale_bits(e))
    q = cute.make_rmem_tensor(8, Float32)
    for i in cutlass.range_constexpr(8):
        q[i] = values[i] * inv
    w0 = pack_fp8((q[0], q[1], q[2], q[3]))
    w1 = pack_fp8((q[4], q[5], q[6], q[7]))
    return w0, w1, e


def scale_offset(row, block, padded_cols: cutlass.Constexpr[int]):
    """Byte offset of scale ``block`` (32-column index) of ``row`` in the F8_128x4
    layout with ``padded_cols`` scale columns (``ops.mxfp8_act.swizzled_sf_offset``)."""
    row = Int64(row)
    block = Int64(block)
    return (
        (block & 3) + (block >> 2) * 512 + (row & 31) * 16 + ((row & 127) >> 5) * 4
        + (row >> 7) * (128 * padded_cols)
    )  # fmt: skip


@cute.jit
def lane_owns_scale():
    return (cute.arch.lane_idx() & 15) == 0


@cute.jit
def publish(values, data_mc, scale_mc, row, col, width: cutlass.Constexpr[int]):
    """Multicast this lane's 8 columns of MXFP8 ``row`` (columns col..col + 7) into
    every rank's box: data at ``data_mc`` [rows, width] FP8, scales at ``scale_mc``."""
    w0, w1, word = quantize8(values)
    offset = Int64(row) * width + col
    multicast_words(data_mc + offset, (w0, w1))
    if lane_owns_scale():
        where = scale_offset(row, col // BLOCK, width // BLOCK)
        multicast_words(scale_mc + where, (word,))


@cute.jit
def store(values, data, scale, row, col, width: cutlass.Constexpr[int]):
    """``publish`` into this GPU's own buffers (plain stores, no Lamport box)."""
    w0, w1, word = quantize8(values)
    store_words(data + (Int64(row) * width + col), (w0, w1))
    if lane_owns_scale():
        store_words(scale + scale_offset(row, col // BLOCK, width // BLOCK), (word,))


@asm()
def _load2(pointer):
    code = "ld.volatile.global.v2.b32 {$0,$1}, [$2];"
    return code, "=r,=r,l,~{memory}", (pointer,), (Uint32, Uint32)


@asm()
def _load1(pointer):
    return "ld.volatile.global.b32 $0, [$1];", "=r,l,~{memory}", (pointer,), Uint32


@cute.jit
def _landed(data, scale, row, col, width: cutlass.Constexpr[int]):
    """This lane's data words and its 128 columns' scale word (every lane loads
    the aligned word of its group) all published."""
    a, b = _load2(data + (Int64(row) * width + col))
    block = (col // BLOCK) & ~3
    word = _load1(scale + scale_offset(row, block, width // BLOCK))
    ok = (a != Uint32(SENTINEL)) & (b != Uint32(SENTINEL))
    return ok & (word != Uint32(SCALE_SENTINEL))


@cute.jit
def poll(data, scale, first, stride, count: cutlass.Constexpr[int], col,
         width: cutlass.Constexpr[int]):  # fmt: skip
    """Lamport-poll this lane's columns of rows first + i stride, i < count, of the
    local MXFP8 box until all landed (data words and scale words)."""
    pending = Uint32(0)
    for i in cutlass.range_constexpr(count):
        if not _landed(data, scale, first + i * stride, col, width):
            pending = pending | Uint32(1 << i)
    sgn, sgt = spin_start()
    while pending != 0:
        again = Uint32(0)
        for i in cutlass.range_constexpr(count):
            if (pending >> i) & 1:
                if not _landed(data, scale, first + i * stride, col, width):
                    again = again | Uint32(1 << i)
        pending = again
        sgn, sgt = spin_tick(sgn, sgt, observed=pending, target=0)


@cute.jit
def rearm(data, scale, row, col, width: cutlass.Constexpr[int]):
    """Re-arm this lane's columns of MXFP8 ``row`` in the local box."""
    store_words(data + (Int64(row) * width + col), (Uint32(SENTINEL),) * 2,
                scope="sys")  # fmt: skip
    if lane_owns_scale():
        where = scale + scale_offset(row, col // BLOCK, width // BLOCK)
        store_words(where, (Uint32(SCALE_SENTINEL),), scope="sys")


# spin timeouts name the caller of these shared polls
register_poll_helper(__file__, "poll")
