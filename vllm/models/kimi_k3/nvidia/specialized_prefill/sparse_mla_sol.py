# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 DSA prefill sparse MLA on Rubin (sm_107): gather-floor design.

Same math and tile form as ``sparse_mla.SparseMlaPrefill`` (keys on M, the 96
heads as the MMA N, one CTA per SM and per token so every top-k row is delivered
into exactly one SM once; the per-SM delivered-bytes cap of the random row gather
is the bound, P-MLA run10), restructured so nothing else is on the critical path:

* 3-stage KV ring (oversized shared memory) filled by 11 TMA ``tile::gather4``
  warps (the gather is issue-bound per warp);
* MMA order QK(t + 1) before PV(t) with two S^T buffers (576 TMEM columns,
  exclusive): the tensor pipe never waits for the softmax;
* lean softmax: one FFMA + one ex2 + one add per score in the common case (lazy
  max with a WG vote; the slow path runs only when a head's max grows by more
  than THR), l accumulated per thread in FP32 and reduced once per token;
* a dedicated epilogue warpgroup: O^T (TMEM) * 1/l -> BF16 in shared memory ->
  one TMA store per 128-dim block, each block's TMEM released as soon as it is
  read, so the next token's PV(0) waits only for its own block.
"""

import math
from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint8, Uint32, Uint64

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common import tensormap
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.primitives import asm
from vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla import (
    D_NOPE,
    e4m3x2_roundtrip,
    H,
    HW,
    IDESC_PV,
    IDESC_QK,
    LOG2E,
    P_DESC,
    ROW,
    SDESC_SW64,
    SDESC_SW128,
    STAGE,
    cvt_e4m3x2,
    ex2,
    fence_async_smem,
    gather4,
    ld_global_v4_s32,
    ld_shared_f32,
    ld_shared_u32,
    ld_shared_v4f,
    mbar_arrive,
    mbar_arrive_tx,
    mbar_init,
    mbar_wait,
    named_bar,
    redux_max_f32,
    st_shared_f32,
    st_shared_u32,
    st_shared_v4,
    stamp_clock,
    tc_commit,
    tc_fence_after,
    tc_fence_before,
    tc_mma,
    tma_tile2d,
    tmem_alloc_exclusive,
    tmem_free_exclusive,
    tmem_ld,
    tmem_st,
)

P_PRE, THR = 4, 4.5  # P = 2^(s - m + P_PRE) <= 2^(P_PRE + THR) = 362 < 448 (E4M3 max)
LOADERS = 11  # gather warps: warps 13..23 (warp 12 issues the MMAs)
NS = 3  # KV stages
THREADS = 32 * 24
# warp roles: 0-7 softmax (2 WG x 48 heads), 8-11 epilogue, 12 MMA, 13-23 gather
W_EPI, W_MMA, W_LD = 8, 12, 13
# setmaxnreg only moves registers inside the CTA's launch allocation (768 x 80):
# 256 x 136 + 128 x 80 + 384 x 40 = 60416 <= 61440 (an increase beyond what the
# decreases free blocks forever)
REG_LAUNCH = 65536 // THREADS // 8 * 8
REG_SOFTMAX, REG_LOAD = 136, 40
assert 256 * REG_SOFTMAX + 128 * REG_LAUNCH + 384 * REG_LOAD <= THREADS * REG_LAUNCH

# shared memory (bytes)
OFF_QN = 0  # Q latent: 4 x [96 x 128 B] SW128
OFF_QR = OFF_QN + 4 * H * 128  # Q RoPE [96 x 64 B] SW64
OFF_P = OFF_QR + H * 64  # 2 x P^T [128 keys x 96 heads] MN-major, no swizzle
OFF_RED = OFF_P + 2 * H * 128  # [2 WG][4 warps][48] per-warp maxima / l partials
OFF_MP = OFF_RED + 2 * 4 * HW * 4  # [2 WG][48] m - P_PRE (log2 units)
OFF_AL = OFF_MP + 2 * HW * 4  # [2 parity][2 WG][48] alpha of the tile
OFF_DL = OFF_AL + 2 * 2 * HW * 4  # [2 parity][2 WG][48] m_new - m_old
OFF_INV = OFF_DL + 2 * 2 * HW * 4  # [2 token parity][96] out_scale / l
OFF_FLAG = OFF_INV + 2 * H * 4  # [2 parity][2 WG] u32 slow-path flags
OFF_BAR = OFF_FLAG + 16
OFF_TMEM = OFF_BAR + 32 * 8
OFF_STG = (OFF_TMEM + 16 + 1023) // 1024 * 1024  # [96 heads x 128 dims] BF16 staging
OFF_KV = OFF_STG + H * 128 * 2
SMEM_BYTES = OFF_KV + NS * STAGE

# barriers
B_QFULL, B_QEMPTY, B_SFULL, B_PFULL, B_OFULL, B_PVDONE, B_OFREE, B_INV = (
    0, 1, 2, 4, 6, 7, 8, 12)  # INV: one per token parity (12, 13)
B_KVFULL, B_KVEMPTY = 16, 24
C_S, C_O = 0, 192  # TMEM: S^T 2 x 96 columns, O^T 4 x 96 columns

STAMP_CTAS, STAMP_TILES, STAMP_EV = 8, 64, 8


@asm()
def tma_store2d(tmap, src, col, row):
    code = ("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [$0, {$2, $3}], [$1];"
            " cp.async.bulk.commit_group;")
    return code, "l,r,r,r,~{memory}", (Int64(tmap), Uint32(src), Int32(col), Int32(row))


@asm()
def bulk_wait_read0():
    return "cp.async.bulk.wait_group.read 0;", "~{memory}", ()


@asm()
def bulk_wait0():
    return "cp.async.bulk.wait_group 0;", "~{memory}", ()


@asm()
def st_shared_bf16(addr, v):
    return ("{ .reg .b16 t; cvt.rn.bf16.f32 t, $1; st.shared.b16 [$0], t; }",
            "r,f,~{memory}", (Uint32(addr), Float32(v)))


def _bar(base, i):
    return base + Uint32(OFF_BAR + 8 * i)


@cute.jit
def _tiles(n):
    t = (n + 127) // 128
    if t < 1:
        t = Int32(1)
    return t


class SparseMlaPrefillSol:
    """Persistent: ``grid`` CTAs (one per SM), CTA c takes tokens c, c + grid, ..."""

    def __init__(self, grid: int, stamps: bool = False, knobs: tuple = (LOADERS, 0, 0)):
        self.grid, self.stamps = grid, stamps
        # diagnostics: (gather warps, no_gather (stale stage, arrive only),
        # no_store (epilogue reads TMEM, skips BF16 staging and TMA stores))
        # 4th knob mma_mode: 0 full kernel; 1 tensor only (QK + PV, no softmax /
        # epilogue dependencies), 2 QK only, 3 PV only
        knobs = tuple(knobs) + (0,) * (4 - len(knobs))
        self.nl, self.no_gather, self.no_store, self.mma_mode = knobs

    @cute.jit
    def __call__(self, q8: Int64, rows: Int32, kv: Int64, unit_rows: Int32, slots: Int64,
                 slot_stride: Int32, lengths: Int64, out: Int64, scale_log2: Float32,
                 out_scale: Float32, kv_count: Int32, stamp: Int64,
                 stream: cuda.CUstream):
        u8 = cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8
        bf16 = cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16
        l2 = cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B
        sw = cuda.CUtensorMapSwizzle
        qn = tensormap.encode_map(q8, u8, (D_NOPE, rows * H), (ROW,), (128, H),
                                  sw.CU_TENSOR_MAP_SWIZZLE_128B, l2)
        qr = tensormap.encode_map(q8 + D_NOPE, u8, (64, rows * H), (ROW,), (64, H),
                                  sw.CU_TENSOR_MAP_SWIZZLE_64B, l2)
        kvm = tensormap.encode_map(kv, u8, (ROW, kv_count), (ROW,), (128, 1),
                                   sw.CU_TENSOR_MAP_SWIZZLE_128B, l2)
        kvr = tensormap.encode_map(kv, u8, (ROW, kv_count), (ROW,), (64, 1),
                                   sw.CU_TENSOR_MAP_SWIZZLE_64B, l2)
        om = tensormap.encode_map(out, bf16, (D_NOPE, rows * H), (D_NOPE * 2,), (128, H),
                                  sw.CU_TENSOR_MAP_SWIZZLE_NONE, l2)
        self.kernel(qn, qr, kvm, kvr, om, rows, unit_rows, slots, slot_stride, lengths,
                    scale_log2,
                    out_scale, stamp).launch(
            grid=(self.grid, 1, 1), block=(THREADS, 1, 1), smem=SMEM_BYTES,
            stream=stream)

    @cute.kernel
    def kernel(self, qn: tensormap.GridMap, qr: tensormap.GridMap,
               kvm: tensormap.GridMap, kvr: tensormap.GridMap, om: tensormap.GridMap,
               rows: Int32, unit_rows: Int32, slots: Int64, slot_stride: Int32,
               lengths: Int64, scale_log2: Float32, out_scale: Float32, stamp: Int64):
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(tid // 32)
        lane = tid % 32
        smem = cutlass.utils.SmemAllocator()
        raw = smem.allocate_array(Uint8, SMEM_BYTES, byte_alignment=1024)
        base = Uint32(raw.toint())
        if tid == 0:
            for i, cnt in ((B_QFULL, 1), (B_QEMPTY, 1), (B_SFULL, 1), (B_SFULL + 1, 1),
                           (B_PFULL, 8), (B_PFULL + 1, 8), (B_OFULL, 1), (B_PVDONE, 1),
                           (B_OFREE, 4), (B_OFREE + 1, 4), (B_OFREE + 2, 4),
                           (B_OFREE + 3, 4), (B_INV, 8), (B_INV + 1, 8)):
                mbar_init(_bar(base, i), Uint32(cnt))
            for s in cutlass.range_constexpr(NS):
                mbar_init(_bar(base, B_KVFULL + s), Uint32(1))
                mbar_init(_bar(base, B_KVEMPTY + s), Uint32(1))
            cute.arch.mbarrier_init_fence()
        if warp == 0:
            tmem_alloc_exclusive(base + Uint32(OFF_TMEM))
        tc_fence_before()
        cute.arch.sync_threads()
        tc_fence_after()
        taddr = ld_shared_u32(base + Uint32(OFF_TMEM))
        n_mine = (rows - cta + self.grid - 1) // self.grid
        sv = cute.make_rmem_tensor((HW,), Float32)
        lv = cute.make_rmem_tensor((HW,), Float32)
        if warp < W_EPI:
            cute.arch.warpgroup_reg_alloc(REG_SOFTMAX)
            if cutlass.const_expr(self.mma_mode == 0):
                self.softmax(base, tid, warp, lane, taddr, n_mine, lengths, cta,
                             scale_log2, out_scale, sv, lv, stamp)
        elif warp < W_MMA:
            if cutlass.const_expr(self.mma_mode == 0):
                self.epilogue(base, tid, warp, lane, taddr, n_mine, cta,
                              tensormap.map_address(om), stamp)
        else:
            cute.arch.warpgroup_reg_dealloc(REG_LOAD)
            if warp == W_MMA:
                self.mma(base, lane, taddr, n_mine, lengths, cta, stamp)
            else:
                self.loader(base, warp - W_LD, lane, cta, n_mine,
                            tensormap.map_address(qn), tensormap.map_address(qr),
                            tensormap.map_address(kvm), tensormap.map_address(kvr),
                            unit_rows, slots, slot_stride, lengths, stamp)
        tc_fence_before()
        cute.arch.sync_threads()
        if warp == 0:
            tc_fence_after()
            tmem_free_exclusive(taddr)

    @cute.jit
    def stamp(self, ptr, cta, g, ev: cutlass.Constexpr):
        if cta < STAMP_CTAS and g < STAMP_TILES:
            stamp_clock(ptr + Int64(((cta * STAMP_TILES + g) * STAMP_EV + ev) * 8))

    # ------------------------------------------------------------------ gather
    @cute.jit
    def loader(self, base, pw, lane, cta, n_mine, tm_qn, tm_qr, tm_kv, tm_kr, unit_rows,
               slots, slot_stride, lengths, stamp):
        """Lane m of gather warp pw takes 4-row quad pw + 11 m of every tile: four
        128 B latent boxes (SW128) and the 64 B RoPE box (SW64) by tile::gather4;
        rows past the valid count read row -1 (TMA zero fill). Warp 0 also loads Q."""
        nl: cutlass.Constexpr = self.nl
        quads = (32 - pw + nl - 1) // nl
        if pw >= nl:
            quads = Int32(0)
        if cutlass.const_expr(self.no_gather):
            quads = Int32(0)
        lptr = cute.make_ptr(Int32, lengths, cute.AddressSpace.gmem, assumed_align=4)
        lens = cute.make_tensor(lptr, cute.make_layout(1 << 30))
        g = Int32(0)
        for qi in range(n_mine):
            row = cta + qi * self.grid
            n = lens[row]
            tiles = _tiles(n)
            if pw == 0:
                if lane == 0:
                    mbar_wait(_bar(base, B_QEMPTY), Uint32((qi & 1) ^ 1))
                    mbar_arrive_tx(_bar(base, B_QFULL), Uint32(4 * H * 128 + H * 64))
                    for j in cutlass.range_constexpr(4):
                        tma_tile2d(base + Uint32(OFF_QN + j * H * 128), tm_qn,
                                   _bar(base, B_QFULL), Int32(128 * j), row * H)
                    tma_tile2d(base + Uint32(OFF_QR), tm_qr, _bar(base, B_QFULL),
                               Int32(0), row * H)
            srow = slots + Int64(row) * Int64(slot_stride) * Int64(4)
            for t in range(tiles):
                s = g % NS
                st = base + Uint32(OFF_KV) + Uint32(s) * Uint32(STAGE)
                full = _bar(base, B_KVFULL + s)
                if lane == 0:
                    mbar_wait(_bar(base, B_KVEMPTY + s), Uint32(((g // NS) & 1) ^ 1))
                    if pw == 0:
                        if cutlass.const_expr(self.no_gather):
                            mbar_arrive(full)
                        else:
                            mbar_arrive_tx(full, Uint32(STAGE))
                        if cutlass.const_expr(self.stamps):
                            self.stamp(stamp, cta, g, 0)
                cute.arch.sync_warp()
                if lane < quads:
                    q = pw + nl * lane
                    key = t * 128 + 4 * q
                    r = ld_global_v4_s32(srow + Int64(key) * Int64(4))
                    # physical slot (block * 64 + offset) -> cache row (block stride
                    # unit_rows rows); negative slots stay negative (zero fill)
                    r0 = (r[0] >> 6) * unit_rows + (r[0] & 63)
                    r1 = (r[1] >> 6) * unit_rows + (r[1] & 63)
                    r2 = (r[2] >> 6) * unit_rows + (r[2] & 63)
                    r3 = (r[3] >> 6) * unit_rows + (r[3] & 63)
                    if r[0] < 0:
                        r0 = Int32(-1)
                    if r[1] < 0:
                        r1 = Int32(-1)
                    if r[2] < 0:
                        r2 = Int32(-1)
                    if r[3] < 0:
                        r3 = Int32(-1)
                    if key >= n:
                        r0 = Int32(-1)
                    if key + 1 >= n:
                        r1 = Int32(-1)
                    if key + 2 >= n:
                        r2 = Int32(-1)
                    if key + 3 >= n:
                        r3 = Int32(-1)
                    for a in cutlass.range_constexpr(4):
                        gather4(st + Uint32(a * 16384) + Uint32(q) * Uint32(512), tm_kv,
                                full, 128 * a, r0, r1, r2, r3)
                    gather4(st + Uint32(4 * 16384) + Uint32(q) * Uint32(256), tm_kr, full,
                            D_NOPE, r0, r1, r2, r3)
                g = g + 1

    # ------------------------------------------------------------------ MMA
    @cute.jit
    def mma(self, base, lane, taddr, n_mine, lengths, cta, stamp):
        lptr = cute.make_ptr(Int32, lengths, cute.AddressSpace.gmem, assumed_align=4)
        lens = cute.make_tensor(lptr, cute.make_layout(1 << 30))
        g = Int32(0)
        for qi in range(n_mine):
            row = cta + qi * self.grid
            tiles = _tiles(lens[row])
            mbar_wait(_bar(base, B_QFULL), Uint32(qi & 1))
            tc_fence_after()
            self.qk(base, lane, taddr, g, 0, tiles, cta, stamp)
            for t in range(tiles):
                if t + 1 < tiles:
                    self.qk(base, lane, taddr, g + 1, t + 1, tiles, cta, stamp)
                self.pv(base, lane, taddr, g, t, tiles, qi, cta, stamp)
                g = g + 1

    @cute.jit
    def qk(self, base, lane, taddr, g, t, tiles, cta, stamp):
        """S^T(g) [128 keys x 96 heads] = K Q^T into S^T buffer g % 2."""
        s = g % NS
        st = base + Uint32(OFF_KV) + Uint32(s) * Uint32(STAGE)
        mbar_wait(_bar(base, B_KVFULL + s), Uint32((g // NS) & 1))
        if cutlass.const_expr(self.stamps):
            if lane == 0:
                self.stamp(stamp, cta, g, 1)
        fence_async_smem()
        tc_fence_after()
        cs = Uint32(C_S) + Uint32(96) * Uint32(g % 2)
        if cutlass.const_expr(self.mma_mode != 3):
            if lane == 0:
                for j in cutlass.range_constexpr(4):
                    ad = Uint64(SDESC_SW128) | Uint64((st + Uint32(j * 16384)) >> Uint32(4))
                    bd = Uint64(SDESC_SW128) | Uint64(
                        (base + Uint32(OFF_QN + j * H * 128)) >> Uint32(4))
                    for kk in cutlass.range_constexpr(2):
                        tc_mma(taddr + cs, ad + Uint64(kk * 4), bd + Uint64(kk * 4),
                               Uint32(IDESC_QK), Uint32(1 if (j + kk) > 0 else 0))
                ad = Uint64(SDESC_SW64) | Uint64((st + Uint32(4 * 16384)) >> Uint32(4))
                bd = Uint64(SDESC_SW64) | Uint64((base + Uint32(OFF_QR)) >> Uint32(4))
                tc_mma(taddr + cs, ad, bd, Uint32(IDESC_QK), Uint32(1))
        if lane == 0:
            tc_commit(_bar(base, B_SFULL + g % 2))
            if t == tiles - 1:
                tc_commit(_bar(base, B_QEMPTY))
        cute.arch.sync_warp()

    @cute.jit
    def pv(self, base, lane, taddr, g, t, tiles, qi, cta, stamp):
        """O^T += V^T P^T(g); frees the KV stage; the token's first tile waits for
        each O^T block's release by the epilogue right before that block."""
        s = g % NS
        st = base + Uint32(OFF_KV) + Uint32(s) * Uint32(STAGE)
        if cutlass.const_expr(self.mma_mode == 0):
            mbar_wait(_bar(base, B_PFULL + g % 2), Uint32((g // 2) & 1))
        if cutlass.const_expr(self.stamps):
            if lane == 0:
                self.stamp(stamp, cta, g, 4)
        tc_fence_after()
        pbuf = base + Uint32(OFF_P) + Uint32(g % 2) * Uint32(H * 128)
        if lane == 0:
            bd0 = Uint64(P_DESC) | Uint64(pbuf >> Uint32(4))
            for b in cutlass.range_constexpr(4):
                if cutlass.const_expr(self.mma_mode == 0):
                    if t == 0:
                        mbar_wait(_bar(base, B_OFREE + b), Uint32((qi & 1) ^ 1))
                        tc_fence_after()
                ad0 = Uint64(SDESC_SW128) | Uint64((st + Uint32(b * 16384)) >> Uint32(4))
                for kk in cutlass.range_constexpr(2):
                    acc = Uint32(1)
                    if t == 0 and kk == 0:
                        acc = Uint32(0)
                    if cutlass.const_expr(self.mma_mode != 2):
                        tc_mma(taddr + Uint32(C_O + 96 * b), ad0 + Uint64(kk * 512),
                               bd0 + Uint64(kk * 64), Uint32(IDESC_PV), acc)
            tc_commit(_bar(base, B_KVEMPTY + s))
            tc_commit(_bar(base, B_PVDONE))
            if t == tiles - 1:
                tc_commit(_bar(base, B_OFULL))
        cute.arch.sync_warp()

    # ------------------------------------------------------------------ softmax
    @cute.jit
    def softmax(self, base, tid, warp, lane, taddr, n_mine, lengths, cta, scale_log2,
                out_scale, sv, lv, stamp):
        """Thread = key lane r of the tile, warpgroup wg = heads 48 wg .. 48 wg + 47."""
        lptr = cute.make_ptr(Int32, lengths, cute.AddressSpace.gmem, assumed_align=4)
        lens = cute.make_tensor(lptr, cute.make_layout(1 << 30))
        wg = warp // 4
        wq = warp % 4
        r = tid % 128
        h0 = wg * HW
        lane_addr = taddr + (Uint32(wq * 32) << Uint32(16))
        mp_base = base + Uint32(OFF_MP + wg * HW * 4)
        lim = Float32(P_PRE + THR)
        g = Int32(0)
        for qi in range(n_mine):
            row = cta + qi * self.grid
            n = lens[row]
            tiles = _tiles(n)
            lv.fill(0.0)
            if r < HW:
                # m - P_PRE starts at 0; tile 0 always takes the slow path, which
                # sets it from the tile's exact max
                st_shared_f32(mp_base + Uint32(r * 4), Float32(0.0))
            named_bar(1 + wg, 128)
            for t in range(tiles):
                par = g % 2
                mbar_wait(_bar(base, B_SFULL + par), Uint32((g // 2) & 1))
                if cutlass.const_expr(self.stamps):
                    if tid == 0:
                        self.stamp(stamp, cta, g, 2)
                tc_fence_after()
                cs = lane_addr + Uint32(C_S + h0) + Uint32(96) * Uint32(par)
                # x = s * scale_log2 - (m - P_PRE): P = 2^x (16 columns at a time)
                xm = Float32(-math.inf)
                for c in cutlass.range_constexpr(3):
                    a = tmem_ld(cs + Uint32(16 * c), 16)
                    for k4 in cutlass.range_constexpr(4):
                        m4 = ld_shared_v4f(mp_base + Uint32(64 * c + 16 * k4))
                        for u in cutlass.range_constexpr(4):
                            k = 16 * c + 4 * k4 + u
                            sv[k] = a[4 * k4 + u] * scale_log2 - m4[u]
                            xm = cute.arch.fmax(xm, sv[k])
                if t * 128 + r >= n:
                    for k in cutlass.range_constexpr(HW):
                        sv[k] = Float32(-math.inf)
                    xm = Float32(-math.inf)
                anyg = cute.arch.vote_any_sync(xm > lim)
                flag = base + Uint32(OFF_FLAG + (par * 2 + wg) * 4)
                if anyg and lane == 0:
                    st_shared_u32(flag, Uint32(g + 1))
                named_bar(1 + wg, 128)
                fl = ld_shared_u32(flag)
                if t == 0:
                    fl = Uint32(g + 1)
                if fl == Uint32(g + 1):
                    self.slow(base, wg, wq, lane, r, h0, g, t, par, lane_addr, sv, lv)
                # P (E4M3, as the stock kernel) to P^T buffer par; l += P (FP32)
                pbuf = base + Uint32(OFF_P) + Uint32(par) * Uint32(H * 128)
                for c in cutlass.range_constexpr(3):
                    w = []
                    for i in cutlass.range_constexpr(4):
                        k = 16 * c + 4 * i
                        p0 = ex2(sv[k])
                        p1 = ex2(sv[k + 1])
                        p2 = ex2(sv[k + 2])
                        p3 = ex2(sv[k + 3])
                        w0 = cvt_e4m3x2(p0, p1)
                        w1 = cvt_e4m3x2(p2, p3)
                        # l = sum of the E4M3 P that PV consumes
                        q0 = e4m3x2_roundtrip(w0)
                        q1 = e4m3x2_roundtrip(w1)
                        lv[k] = lv[k] + q0[0]
                        lv[k + 1] = lv[k + 1] + q0[1]
                        lv[k + 2] = lv[k + 2] + q1[0]
                        lv[k + 3] = lv[k + 3] + q1[1]
                        w.append(w0 | (w1 << Uint32(16)))
                    st_shared_v4(pbuf + Uint32((3 * wg + c) * 2048)
                                 + Uint32((r // 8) * 128 + (r % 8) * 16),
                                 w[0], w[1], w[2], w[3])
                fence_async_smem()
                tc_fence_before()
                cute.arch.sync_warp()
                if cutlass.const_expr(self.stamps):
                    if tid == 0:
                        self.stamp(stamp, cta, g, 3)
                if lane == 0:
                    mbar_arrive(_bar(base, B_PFULL + par))
                g = g + 1
            # ---- token end: l per head (sum over the 128 key lanes), inv = s / l
            red = base + Uint32(OFF_RED + (wg * 4 + wq) * HW * 4)
            for k in cutlass.range_constexpr(HW):
                x = lv[k]
                for sh in cutlass.range_constexpr(5):
                    x = x + cute.arch.shuffle_sync_bfly(x, 16 >> sh)
                if lane == 0:
                    st_shared_f32(red + Uint32(4 * k), x)
            named_bar(1 + wg, 128)
            if r < HW:
                lt = Float32(0.0)
                for w4 in cutlass.range_constexpr(4):
                    lt = lt + ld_shared_f32(
                        base + Uint32(OFF_RED + ((wg * 4 + w4) * HW + r) * 4))
                iv = Float32(0.0)
                if lt > Float32(0.0):
                    iv = out_scale / lt
                st_shared_f32(base + Uint32(OFF_INV + ((qi % 2) * H + h0 + r) * 4), iv)
            named_bar(1 + wg, 128)
            if lane == 0:
                mbar_arrive(_bar(base, B_INV + qi % 2))

    @cute.jit
    def slow(self, base, wg, wq, lane, r, h0, g, t, par, lane_addr, sv, lv):
        """Some head's max grew by more than THR: exact per-head tile max over the
        WG, raise m where it grew, shift x, rescale l and O^T (after PV(g - 1))."""
        red = base + Uint32(OFF_RED + (wg * 4 + wq) * HW * 4)
        for k in cutlass.range_constexpr(HW):
            mx = redux_max_f32(sv[k])
            if lane == 0:
                st_shared_f32(red + Uint32(4 * k), mx)
        named_bar(1 + wg, 128)
        al = base + Uint32(OFF_AL + (par * 2 + wg) * HW * 4)
        dl = base + Uint32(OFF_DL + (par * 2 + wg) * HW * 4)
        if r < HW:
            mt = ld_shared_f32(base + Uint32(OFF_RED + (wg * 4 * HW + r) * 4))
            for w4 in cutlass.range_constexpr(1, 4):
                mt = cute.arch.fmax(mt, ld_shared_f32(
                    base + Uint32(OFF_RED + ((wg * 4 + w4) * HW + r) * 4)))
            # mt = max x = max s - (m - P_PRE); grow when mt > P_PRE + THR
            d = Float32(0.0)
            alpha = Float32(1.0)
            grow = Int32(0)
            if mt > Float32(P_PRE + THR):
                grow = Int32(1)
            if t == 0:
                if mt > Float32(-1e30):  # not an all-masked tile
                    grow = Int32(1)
            if grow != 0:
                d = mt - Float32(P_PRE)
                alpha = ex2(-d)
                mpa = base + Uint32(OFF_MP + (wg * HW + r) * 4)
                st_shared_f32(mpa, ld_shared_f32(mpa) + d)
            if t == 0:
                alpha = Float32(1.0)  # l = 0 and O^T is overwritten by PV(0)
            st_shared_f32(al + Uint32(4 * r), alpha)
            st_shared_f32(dl + Uint32(4 * r), d)
        named_bar(1 + wg, 128)
        for k4 in cutlass.range_constexpr(HW // 4):
            d4 = ld_shared_v4f(dl + Uint32(16 * k4))
            a4 = ld_shared_v4f(al + Uint32(16 * k4))
            for u in cutlass.range_constexpr(4):
                k = 4 * k4 + u
                sv[k] = sv[k] - d4[u]
                lv[k] = lv[k] * a4[u]
        if t > 0:
            # PV(g - 1) is issued after QK(g): wait for it before touching O^T
            mbar_wait(_bar(base, B_PVDONE), Uint32((g - 1) & 1))
            tc_fence_after()
            for b in cutlass.range_constexpr(4):
                for c in cutlass.range_constexpr(3):
                    col = lane_addr + Uint32(C_O + 96 * b + h0 + 16 * c)
                    oa = tmem_ld(col, 16)
                    a4s = [ld_shared_v4f(al + Uint32(64 * c + 16 * q)) for q in range(4)]
                    tmem_st(col, [oa[k] * a4s[k // 4][k % 4] for k in range(16)])

    # ------------------------------------------------------------------ epilogue
    @cute.jit
    def epilogue(self, base, tid, warp, lane, taddr, n_mine, cta, tm_out, stamp):
        """Thread = dim r of a 128-dim O^T block: O * (s / l) -> BF16 staging
        [96 heads x 128 dims] -> TMA store; block b's TMEM is released (OFREE b)
        as soon as it has been read."""
        e = tid - 32 * W_EPI
        wq = warp - W_EPI
        lane_addr = taddr + (Uint32(wq * 32) << Uint32(16))
        for qi in range(n_mine):
            row = cta + qi * self.grid
            mbar_wait(_bar(base, B_OFULL), Uint32(qi & 1))
            # the softmax may finish token qi + 1 before this epilogue reads token
            # qi's 1/l (its PV waits only for OFREE): per-parity buffers
            mbar_wait(_bar(base, B_INV + qi % 2), Uint32((qi // 2) & 1))
            inv_base = base + Uint32(OFF_INV) + Uint32(qi % 2) * Uint32(H * 4)
            tc_fence_after()
            if cutlass.const_expr(self.stamps):
                if e == 0:
                    self.stamp(stamp, cta, Int32(qi), 5)
            for b in cutlass.range_constexpr(4):
                if e == 0:
                    bulk_wait_read0()
                named_bar(3, 128)
                for c in cutlass.range_constexpr(3):
                    o = tmem_ld(lane_addr + Uint32(C_O + 96 * b + 32 * c), 32)
                    for k8 in cutlass.range_constexpr(8):
                        iv = ld_shared_v4f(inv_base + Uint32((32 * c + 4 * k8) * 4))
                        for u in cutlass.range_constexpr(4):
                            h = 32 * c + 4 * k8 + u
                            if cutlass.const_expr(not self.no_store):
                                st_shared_bf16(
                                    base + Uint32(OFF_STG) + Uint32((h * 128 + e) * 2),
                                    o[4 * k8 + u] * iv[u])
                tc_fence_before()
                cute.arch.sync_warp()
                if lane == 0:
                    mbar_arrive(_bar(base, B_OFREE + b))
                fence_async_smem()
                named_bar(3, 128)
                if e == 0:
                    if cutlass.const_expr(not self.no_store):
                        tma_store2d(tm_out, base + Uint32(OFF_STG), Int32(128 * b), row * H)
            if cutlass.const_expr(self.stamps):
                if e == 0:
                    self.stamp(stamp, cta, Int32(qi), 6)
        if e == 0:
            bulk_wait0()


SOL_MIN_ROWS = 1024


@cache
def _compiled(grid: int, stamps: bool, knobs: tuple):
    kernel = SparseMlaPrefillSol(grid, stamps, knobs)
    args = (Int64(0), Int32(1), Int64(0), Int32(64), Int64(0), Int32(1), Int64(0), Int64(0),
            Float32(1.0), Float32(1.0), Int32(64), Int64(0), cuda.CUstream(0))
    return cute.compile(kernel, *args, options=f"--ptxas-options -maxrregcount={REG_LAUNCH}")


def sparse_mla_prefill_sol(q8: torch.Tensor, kv_rows: torch.Tensor, slots: torch.Tensor,
                           lengths: torch.Tensor, scale: float, out_scale: float,
                           unit_rows: int = 64,
                           out: torch.Tensor | None = None,
                           stamps: torch.Tensor | None = None,
                           knobs: tuple = (LOADERS, 0, 0)) -> torch.Tensor:
    """q8 [T, 96, 576] E4M3 (contiguous, per-tensor scale folded into ``scale``);
    kv_rows: the cache as rows of 576 B, block b's 64 rows starting at row
    b * unit_rows (``slots`` hold block * 64 + offset); slots
    [T, W] int32, lengths [T] int32 valid counts; ``scale`` = softmax scale x q
    scale x k scale; ``out_scale`` = k scale. Returns BF16 [T, 96, 512]."""
    t = q8.shape[0]
    if q8.shape[1:] != (H, ROW) or not q8.is_contiguous():
        raise ValueError(f"sparse_mla_prefill_sol: q8 must be contiguous [T, {H}, {ROW}]")
    if out is None:
        out = torch.empty((t, H, D_NOPE), dtype=torch.bfloat16, device=q8.device)
    if not out.is_contiguous() or out.shape != (t, H, D_NOPE):
        raise ValueError("sparse_mla_prefill_sol: out must be contiguous [T, 96, 512]")
    if t == 0:
        return out
    sms = torch.cuda.get_device_properties(q8.device).multi_processor_count
    grid = min(sms, t)
    fn = _compiled(grid, stamps is not None, tuple(knobs))
    stream = cuda.CUstream(torch.cuda.current_stream(q8.device).cuda_stream)
    fn(Int64(q8.data_ptr()), Int32(t), Int64(kv_rows.data_ptr()), Int32(unit_rows),
       Int64(slots.data_ptr()),
       Int32(slots.stride(0)), Int64(lengths.data_ptr()), Int64(out.data_ptr()),
       Float32(scale * LOG2E), Float32(out_scale), Int32(kv_rows.shape[0]),
       Int64(0 if stamps is None else stamps.data_ptr()), stream)
    return out
