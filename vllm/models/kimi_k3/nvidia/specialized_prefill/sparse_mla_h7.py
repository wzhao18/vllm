# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 DSA prefill sparse MLA on Rubin (sm_107): split-row softmax.

Same pipeline, TMEM and shared-memory layout as sparse_mla_h6 (gather, QK and PV
warps unchanged), with a different softmax: both softmax warpgroups work on every
64-key subtile. Warp w (quarter q = w % 4, warpgroup g = w // 4) owns TMEM lanes
(heads) 32 q + 16 g .. + 15 and reads them with tcgen05.ld 16x32bx2: lanes l and
l + 16 of the warp hold the two 32-key halves of one head row, so the row max and
sum take one shuffle and the lazy max m stays in the thread (no warpgroup-to-
warpgroup handoff). Per subtile a thread exponentiates 32 values instead of 64
(h6: one warpgroup per subtile, ping-pong). Grown heads are OR-ed over the 8
softmax warps (bar.red) and rescaled column-wise after PV(u - 1): warpgroup g
covers O^T blocks 2 g and 2 g + 1. Warpgroup 0 copies the next token's Q into
TMEM and runs the epilogue (h6 code).
"""

import math
from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.primitives import asm
from vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla import (
    D_NOPE,
    H,
    LOG2E,
    ROW,
    cvt_e4m3x2,
    ex2,
    fence_async_smem,
    ld_shared_f32,
    ld_shared_u32,
    mbar_arrive,
    mbar_wait,
    named_bar,
    st_shared_f32,
    st_shared_u32,
    st_shared_v4,
    tc_fence_after,
    tc_fence_before,
)
from vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla_h5 import (
    _tree_max,
    fadd2,
    ffma2,
)
from vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla_h6 import (
    B_OREADY,
    B_PEMPTY,
    B_SFREE,
    B_SFULL,
    C_O,
    C_S,
    NP,
    OFF_AL,
    OFF_GMASK,
    OFF_INV,
    OFF_P,
    P_BYTES,
    P_PRE,
    THR,
    SparseMlaPrefillH6,
    _bar,
    _subs,
    ballot,
    bar_red_or,
    bfind,
)

BAR_GROW, BAR_TOKEN = 1, 2  # named barriers over the 8 softmax warps


@asm()
def tmem_ld_half_rows(addr):
    """tcgen05.ld 16x32bx2 x32: lanes 0-15 of the warp read columns [0, 32) of TMEM
    lanes addr.lane + l, lanes 16-31 read columns [32, 64) of the same TMEM lanes."""
    regs = ",".join(f"${i}" for i in range(32))
    code = (f"tcgen05.ld.sync.aligned.16x32bx2.x32.b32 {{{regs}}}, [$32], 32; "
            "tcgen05.wait::ld.sync.aligned;")
    return code, ",".join(["=f"] * 32 + ["r", "~{memory}"]), (Uint32(addr),), (Float32,) * 32


@asm()
def tmem_ld_o2(addr):
    """Column h of two adjacent O^T blocks (columns addr and addr + 96)."""
    code = ("tcgen05.ld.sync.aligned.32x32b.x1.b32 {$0}, [$2]; "
            "tcgen05.ld.sync.aligned.32x32b.x1.b32 {$1}, [$3]; "
            "tcgen05.wait::ld.sync.aligned;")
    a = Uint32(addr)
    return code, "=f,=f,r,r,~{memory}", (a, a + Uint32(96)), (Float32,) * 2


@asm()
def tmem_st_o2(addr, v0, v1):
    code = ("tcgen05.st.sync.aligned.32x32b.x1.b32 [$0], {$2}; "
            "tcgen05.st.sync.aligned.32x32b.x1.b32 [$1], {$3}; "
            "tcgen05.wait::st.sync.aligned;")
    a = Uint32(addr)
    return code, "r,r,f,f,~{memory}", (a, a + Uint32(96), Float32(v0), Float32(v1))


class SparseMlaPrefillH7(SparseMlaPrefillH6):
    def __init__(self, grid: int, knobs: tuple = ()):
        super().__init__(grid, knobs)
        self.sm_arrivals = 8

    @cute.jit
    def softmax(self, base, tid, warp, lane, taddr, n_mine, lengths, cta, scale_log2,
                out_scale, tm_o):
        lptr = cute.make_ptr(Int32, lengths, cute.AddressSpace.gmem, assumed_align=4)
        lens = cute.make_tensor(lptr, cute.make_layout(1 << 30))
        wg = warp // 4
        q = warp % 4
        half = lane // 16
        h = 32 * q + 16 * wg + lane % 16  # head row of this thread
        real = h < H
        if cutlass.const_expr(self.no_math):
            real = h < 0
        quarter_addr = taddr + (Uint32(32 * q) << Uint32(16))  # 32x32b (O^T, Q)
        row_addr = taddr + (Uint32(32 * q + 16 * wg) << Uint32(16))  # 16x32bx2 (S)
        xv = cute.make_rmem_tensor((32,), Float32)
        if wg == 0 and n_mine > 0:
            self.q_copy(base, tid % 128, lane, quarter_addr, Int32(0))
        u = Int32(0)
        for qi in range(n_mine):
            n = lens[cta + qi * self.grid]
            subs = _subs(n)
            mp = Float32(0.0)  # m - P_PRE of this head row (both row halves agree)
            lw = Float32(0.0)  # this half's sum, relative to mp
            for i in cutlass.range(subs, unroll=1):
                if i == subs - 1:
                    mp, lw = self.sub7(base, lane, wg, q, half, h, real, row_addr,
                                       quarter_addr, xv, scale_log2, u, i, n, mp, lw, True)
                else:
                    mp, lw = self.sub7(base, lane, wg, q, half, h, real, row_addr,
                                       quarter_addr, xv, scale_log2, u, i, n, mp, lw, False)
                u = u + 1
            lt = lw + cute.arch.shuffle_sync_bfly(lw, 16)
            if real and half == 0:
                iv = Float32(0.0)
                if lt > Float32(0.0):
                    iv = out_scale / lt
                st_shared_f32(base + Uint32(OFF_INV) + Uint32(h * 4), iv)
            named_bar(Uint32(BAR_TOKEN), 256)
            if wg == 0:
                if qi + 1 < n_mine:
                    self.q_copy(base, tid % 128, lane, quarter_addr, qi + 1)
                self.epilogue(base, 0, q, lane, quarter_addr, qi, cta, tm_o)

    @cute.jit
    def sub7(self, base, lane, wg, q, half, h, real, row_addr, quarter_addr, xv,
             scale_log2, u, i, n, mp, lw, masked: cutlass.Constexpr):
        mbar_wait(_bar(base, B_SFULL + u % 2), Uint32((u // 2) & 1))
        tc_fence_after()
        sa = tmem_ld_half_rows(row_addr + Uint32(C_S) + Uint32(64) * Uint32(u % 2))
        for k in cutlass.range_constexpr(32):
            xv[k] = sa[k]
        tc_fence_before()
        cute.arch.sync_warp()
        if lane == 0:
            mbar_arrive(_bar(base, B_SFREE + u % 2))
        grow = Int32(0)
        alpha = Float32(1.0)
        if real:
            if cutlass.const_expr(masked):
                for k in cutlass.range_constexpr(32):
                    if 64 * i + 32 * half + k >= n:
                        xv[k] = Float32(-math.inf)
            xl = _tree_max([xv[k] for k in range(32)])
            xm = cute.arch.fmax(xl, cute.arch.shuffle_sync_bfly(xl, 16)) * scale_log2
            if i == 0:
                mp = Float32(0.0)
                lw = Float32(0.0)
                if xm > Float32(-1e30):
                    mp = xm - Float32(P_PRE)
            elif xm - mp > Float32(P_PRE + THR):
                grow = Int32(1)
                mn = xm - Float32(P_PRE)
                alpha = ex2(mp - mn)
                lw = lw * alpha
                mp = mn
        # heads that grew: masks (low 16 ballot bits = this warp's 16 rows) and alphas,
        # OR-ed over the 8 softmax warps; the rescale runs after P
        par = Uint32(u % 2)
        gm = base + Uint32(OFF_GMASK) + par * Uint32(32)
        alb = base + Uint32(OFF_AL) + par * Uint32(H * 4)
        gw = ballot(Uint32(grow)) & Uint32(0xFFFF)
        if lane == 0 and q < 3:
            st_shared_u32(gm + Uint32(4) * Uint32(2 * q + wg), gw)
        if grow != 0 and half == 0:
            st_shared_f32(alb + Uint32(h * 4), alpha)
        anyg = bar_red_or(Uint32(BAR_GROW), Uint32(256), Uint32(grow))
        if u >= NP:
            mbar_wait(_bar(base, B_PEMPTY + u % NP), Uint32((u // NP - 1) & 1))
        if real:
            pbuf = base + Uint32(OFF_P) + Uint32(u % NP) * Uint32(P_BYTES)
            a0, a1 = Float32(0.0), Float32(0.0)
            b0, b1 = Float32(0.0), Float32(0.0)
            nmp = Float32(0.0) - mp
            for k in cutlass.range_constexpr(16):
                xv[2 * k], xv[2 * k + 1] = ffma2(xv[2 * k], xv[2 * k + 1], scale_log2,
                                                 scale_log2, nmp, nmp)
            for c2 in cutlass.range_constexpr(2):
                w = []
                for qq in cutlass.range_constexpr(4):
                    k = 16 * c2 + 4 * qq
                    p0 = ex2(xv[k])
                    p1 = ex2(xv[k + 1])
                    p2 = ex2(xv[k + 2])
                    p3 = ex2(xv[k + 3])
                    a0, a1 = fadd2(a0, a1, p0, p1)
                    b0, b1 = fadd2(b0, b1, p2, p3)
                    w.append(cvt_e4m3x2(p0, p1) | (cvt_e4m3x2(p2, p3) << Uint32(16)))
                ch = Uint32(2) * Uint32(half) + Uint32(c2)
                st_shared_v4(pbuf + Uint32(h * 64) + (
                    (ch ^ ((Uint32(h) >> Uint32(1)) & Uint32(3))) << Uint32(4)),
                    w[0], w[1], w[2], w[3])
            a0, a1 = fadd2(a0, a1, b0, b1)
            lw = lw + (a0 + a1)
        if anyg != Uint32(0):
            mbar_wait(_bar(base, B_PEMPTY + (u - 1) % NP), Uint32(((u - 1) // NP) & 1))
            tc_fence_after()
            for r in cutlass.range_constexpr(6):
                m = ld_shared_u32(gm + Uint32(4 * r))
                while m != Uint32(0):
                    j = bfind(m)
                    m = m ^ (Uint32(1) << j)
                    hh = Uint32(32 * (r // 2) + 16 * (r % 2)) + j
                    al = ld_shared_f32(alb + hh * Uint32(4))
                    col = quarter_addr + Uint32(C_O + 192 * wg) + hh
                    o0, o1 = tmem_ld_o2(col)
                    tmem_st_o2(col, o0 * al, o1 * al)
            tc_fence_before()
        fence_async_smem()
        cute.arch.sync_warp()
        if lane == 0:
            mbar_arrive(_bar(base, B_OREADY + u % NP))
        return mp, lw


@cache
def _compiled(grid: int, knobs: tuple = ()):
    kernel = SparseMlaPrefillH7(grid, knobs)
    args = (Int64(0), Int32(1), Int64(0), Int32(64), Int64(0), Int32(1), Int64(0), Int64(0),
            Float32(1.0), Float32(1.0), Int32(64), Int64(0), cuda.CUstream(0))
    return cute.compile(kernel, *args)


def sparse_mla_prefill_h7(q8: torch.Tensor, kv_rows: torch.Tensor, slots: torch.Tensor,
                          lengths: torch.Tensor, scale: float, out_scale: float,
                          unit_rows: int = 64, out: torch.Tensor | None = None,
                          stamps: torch.Tensor | None = None,
                          knobs: tuple = ()) -> torch.Tensor:
    """Same contract as ``sparse_mla_h6.sparse_mla_prefill_h6``."""
    t = q8.shape[0]
    if q8.shape[1:] != (H, ROW) or not q8.is_contiguous():
        raise ValueError(f"sparse_mla_prefill_h7: q8 must be contiguous [T, {H}, {ROW}]")
    if out is None:
        out = torch.empty((t, H, D_NOPE), dtype=torch.bfloat16, device=q8.device)
    if not out.is_contiguous() or out.shape != (t, H, D_NOPE):
        raise ValueError("sparse_mla_prefill_h7: out must be contiguous [T, 96, 512]")
    if t == 0:
        return out
    sms = torch.cuda.get_device_properties(q8.device).multi_processor_count
    grid = min(sms, t)
    fn = _compiled(grid, tuple(knobs))
    stream = cuda.CUstream(torch.cuda.current_stream(q8.device).cuda_stream)
    fn(Int64(q8.data_ptr()), Int32(t), Int64(kv_rows.data_ptr()), Int32(unit_rows),
       Int64(slots.data_ptr()), Int32(slots.stride(0)), Int64(lengths.data_ptr()),
       Int64(out.data_ptr()), Float32(scale * LOG2E), Float32(out_scale),
       Int32(kv_rows.shape[0]), Int64(0 if stamps is None else stamps.data_ptr()), stream)
    return out
