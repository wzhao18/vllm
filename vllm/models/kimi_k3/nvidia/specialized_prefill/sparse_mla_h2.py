# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 DSA prefill sparse MLA on Rubin (sm_107): hybrid-orientation kernel.

One CTA per SM and per token (every top-k row is delivered into one SM once: the
delivered-bytes gather floor, P-MLA run10). The shared-memory operand traffic of
the all-transposed form (Q^T and P^T re-fetched by every SS MMA, P-MLA run15) and
its cross-lane softmax are removed by mixing orientations per GEMM:

* QK heads on M, A from TMEM: S [128 (96 heads) x 64 keys] = Q_lat (TMEM, .ts)
  K_lat^T + Q_rope K_rope^T (SS); only the K rows are read from shared memory.
* softmax row per thread (thread = head): max, lazy-max test, exp2, l all
  thread-local; P [96 heads x 64 keys] E4M3 written K-major (SW64) to SMEM.
* PV keys on M: O^T [4 x 128 dims x 96 heads] += V^T (SS, MN-major) P^T with the
  P^T operand held in the Rubin B collector across the 4 O^T blocks.
TMEM (exclusive 576 columns): Q latent 128 + S 64 + O^T 384. Gathers (TMA
gather4, 3-stage ring of 128-key stages), epilogue (BF16 staging + TMA store with
per-block O^T release) as in ``sparse_mla_sol``.
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
    H,
    LOG2E,
    ROW,
    SDESC_SW64,
    SDESC_SW128,
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
from vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla_sol import (
    bulk_wait0,
    bulk_wait_read0,
    st_shared_bf16,
    tc_mma_cb,
    tma_store2d,
)

P_PRE, THR = 4, 4.5  # P = 2^(s - m + P_PRE) <= 2^(P_PRE + THR) = 362 < 448 (E4M3 max)
LOADERS = 10  # gather warps; the 11th warp after the MMA warp loads Q
NS = 6  # 64-key KV stages (one per subtile: released after its PV)
STAGE64 = 4 * 64 * 128 + 64 * 64  # 36864: 4 latent SW128 atoms + RoPE SW64
# warps: 0-7 softmax (2 WG x 32 columns, thread = head; lanes 96-127 = padding rows),
# 8-11 epilogue, 12 MMA, 13-23 gather
W_EPI, W_MMA, W_LD = 8, 12, 13
THREADS = 32 * 24
REG_LAUNCH = 65536 // THREADS // 8 * 8  # 80
REG_SOFTMAX, REG_LOAD = 136, 40
assert 256 * REG_SOFTMAX + 128 * REG_LAUNCH + 384 * REG_LOAD <= THREADS * REG_LAUNCH

# shared memory (bytes)
OFF_QN = 0  # Q latent staging: 4 x [96 x 128 B] SW128 (copied into TMEM per token)
OFF_QR = OFF_QN + 4 * H * 128  # Q RoPE: [128 rows x 64 B] SW64 (A of the SS RoPE MMA)
OFF_P = OFF_QR + 128 * 64  # 2 x P [96 heads x 64 keys] E4M3, K-major SW64
P_BYTES = H * 64
OFF_AL = OFF_P + 2 * P_BYTES  # [2 parity][128] alpha of the subtile (slow path)
OFF_MX = OFF_AL + 2 * 128 * 4  # [2 parity][2 WG][128] partial row max; l at +2048
OFF_INV = OFF_MX + 3 * 1024  # [2 token parity][96] out_scale / l
OFF_FLAG = OFF_INV + 2 * H * 4  # [2 parity] u32 slow flags
OFF_BAR = OFF_FLAG + 16
OFF_TMEM = OFF_BAR + 32 * 8
OFF_STG = (OFF_TMEM + 16 + 1023) // 1024 * 1024
OFF_KV = OFF_STG + H * 128 * 2
SMEM_BYTES = OFF_KV + NS * STAGE64

(B_QFULL, B_QEMPTY, B_QTM, B_SFULL, B_SFREE, B_PFULL, B_OFULL, B_PVDONE, B_OFREE,
 B_INV) = (0, 1, 2, 3, 4, 5, 6, 7, 8, 12)
B_KVFULL, B_KVEMPTY = 16, 24
C_Q, C_S, C_O = 0, 128, 192

# instruction descriptors (dense kind::f8f6f4, E4M3 x E4M3 -> FP32, Rubin K64 bit 29)
IDESC_QK = (1 << 4) | (64 >> 3 << 17) | (128 >> 4 << 24) | (1 << 29)  # M128 N64
IDESC_PV = (1 << 4) | (H >> 3 << 17) | (128 >> 4 << 24) | (1 << 29) | (1 << 15)  # A MN

STAMP_CTAS, STAMP_TILES, STAMP_EV = 8, 64, 8


@asm()
def tc_mma_ts(d, a_tmem, b, idesc, acc):
    code = ("{ .reg .pred p; setp.ne.b32 p, $4, 0; "
            "tcgen05.mma.cta_group::1.kind::f8f6f4 [$0], [$1], $2, $3, p; }")
    return code, "r,r,l,r,r,~{memory}", (Uint32(d), Uint32(a_tmem), Uint64(b),
                                          Uint32(idesc), Uint32(acc))


@asm()
def tmem_st_u32(addr, vals):
    n = len(vals)
    regs = ",".join(f"${i + 1}" for i in range(n))
    code = (f"tcgen05.st.sync.aligned.32x32b.x{n}.b32 [$0], {{{regs}}}; "
            "tcgen05.wait::st.sync.aligned;")
    return code, "r" + ",r" * n + ",~{memory}", (Uint32(addr), *[Uint32(v) for v in vals])


@asm()
def ld_shared_v4u(addr):
    return ("ld.shared.v4.b32 {$0,$1,$2,$3}, [$4];", "=r,=r,=r,=r,r,~{memory}",
            (Uint32(addr),), (Uint32,) * 4)


def _bar(base, i):
    return base + Uint32(OFF_BAR + 8 * i)


@cute.jit
def _subs(n):
    """64-key subtiles of a row (at least one: empty rows still write zeros)."""
    t = (n + 63) // 64
    if t < 1:
        t = Int32(1)
    return t


class SparseMlaPrefillH2:
    def __init__(self, grid: int, stamps: bool = False, knobs: tuple = (0, 0)):
        self.grid, self.stamps = grid, stamps
        # diagnostics: no_store (epilogue skips staging stores + TMA stores),
        # no_mma (MMA warp only commits its barriers)
        self.no_store, self.no_mma = knobs

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
                    scale_log2, out_scale, stamp).launch(
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
            for i, cnt in ((B_QFULL, 1), (B_QEMPTY, 1), (B_QTM, 8), (B_SFULL, 1),
                           (B_SFREE, 8), (B_PFULL, 8), (B_OFULL, 1), (B_PVDONE, 1),
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
        xv = cute.make_rmem_tensor((32,), Float32)
        if warp < W_EPI:
            cute.arch.warpgroup_reg_alloc(REG_SOFTMAX)
            self.softmax(base, tid, warp, lane, taddr, n_mine, lengths, cta, scale_log2,
                         out_scale, xv, stamp)
        elif warp < W_MMA:
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
        quads = (16 - pw + LOADERS - 1) // LOADERS
        lptr = cute.make_ptr(Int32, lengths, cute.AddressSpace.gmem, assumed_align=4)
        lens = cute.make_tensor(lptr, cute.make_layout(1 << 30))
        if pw == LOADERS:
            # Q warp: the next token's Q waits for this token's last QK (QEMPTY);
            # kept off the gather warps so the KV ring runs ahead across tokens
            if lane == 0:
                for qi in range(n_mine):
                    row = cta + qi * self.grid
                    mbar_wait(_bar(base, B_QEMPTY), Uint32((qi & 1) ^ 1))
                    mbar_arrive_tx(_bar(base, B_QFULL), Uint32(4 * H * 128 + H * 64))
                    for j in cutlass.range_constexpr(4):
                        tma_tile2d(base + Uint32(OFF_QN + j * H * 128), tm_qn,
                                   _bar(base, B_QFULL), Int32(128 * j), row * H)
                    tma_tile2d(base + Uint32(OFF_QR), tm_qr, _bar(base, B_QFULL),
                               Int32(0), row * H)
        else:
            g = Int32(0)
            for qi in range(n_mine):
                row = cta + qi * self.grid
                n = lens[row]
                subs = _subs(n)
                srow = slots + Int64(row) * Int64(slot_stride) * Int64(4)
                for t in range(subs):
                    s = g % NS
                    st = base + Uint32(OFF_KV) + Uint32(s) * Uint32(STAGE64)
                    full = _bar(base, B_KVFULL + s)
                    q = pw + LOADERS * lane
                    if lane >= quads:
                        q = pw  # idle lanes read a valid slot of the tile (unused)
                    key = t * 64 + 4 * q
                    # slot loads before the stage wait: their latency hides under it
                    r = ld_global_v4_s32(srow + Int64(key) * Int64(4))
                    if lane == 0:
                        mbar_wait(_bar(base, B_KVEMPTY + s), Uint32(((g // NS) & 1) ^ 1))
                        if pw == 0:
                            mbar_arrive_tx(full, Uint32(STAGE64))
                            if cutlass.const_expr(self.stamps):
                                self.stamp(stamp, cta, g, 0)
                    cute.arch.sync_warp()
                    if lane < quads:
                        r0 = (r[0] >> 6) * unit_rows + (r[0] & 63)
                        r1 = (r[1] >> 6) * unit_rows + (r[1] & 63)
                        r2 = (r[2] >> 6) * unit_rows + (r[2] & 63)
                        r3 = (r[3] >> 6) * unit_rows + (r[3] & 63)
                        if r[0] < 0 or key >= n:
                            r0 = Int32(-1)
                        if r[1] < 0 or key + 1 >= n:
                            r1 = Int32(-1)
                        if r[2] < 0 or key + 2 >= n:
                            r2 = Int32(-1)
                        if r[3] < 0 or key + 3 >= n:
                            r3 = Int32(-1)
                        for a in cutlass.range_constexpr(4):
                            gather4(st + Uint32(a * 8192) + Uint32(q) * Uint32(512), tm_kv,
                                    full, 128 * a, r0, r1, r2, r3)
                        gather4(st + Uint32(4 * 8192) + Uint32(q) * Uint32(256), tm_kr,
                                full, D_NOPE, r0, r1, r2, r3)
                    if cutlass.const_expr(self.stamps):
                        if pw == 0 and lane == 0:
                            self.stamp(stamp, cta, g, 7)
                    g = g + 1

    # ------------------------------------------------------------------ MMA
    @cute.jit
    def mma(self, base, lane, taddr, n_mine, lengths, cta, stamp):
        """Global 64-key subtile counter u (two per KV stage). Per subtile: wait
        S read (SFREE u-1), QK(u); then PV(u-1) once its P is ready, so QK(u)
        overlaps softmax(u-1)."""
        lptr = cute.make_ptr(Int32, lengths, cute.AddressSpace.gmem, assumed_align=4)
        lens = cute.make_tensor(lptr, cute.make_layout(1 << 30))
        u = Int32(0)
        for qi in range(n_mine):
            row = cta + qi * self.grid
            subs = _subs(lens[row])
            mbar_wait(_bar(base, B_QTM), Uint32(qi & 1))
            tc_fence_after()
            for i in range(subs):
                if u > 0:
                    mbar_wait(_bar(base, B_SFREE), Uint32((u - 1) & 1))
                    tc_fence_after()
                self.qk(base, lane, taddr, u, i == subs - 1, cta, stamp)
                if i > 0:
                    self.pv(base, lane, taddr, u - 1, i - 1, False, qi, cta, stamp)
                u = u + 1
            self.pv(base, lane, taddr, u - 1, subs - 1, True, qi, cta, stamp)

    @cute.jit
    def qk(self, base, lane, taddr, u, last, cta, stamp):
        s = u % NS
        st = base + Uint32(OFF_KV) + Uint32(s) * Uint32(STAGE64)
        mbar_wait(_bar(base, B_KVFULL + s), Uint32((u // NS) & 1))
        if cutlass.const_expr(self.stamps):
            if lane == 0:
                self.stamp(stamp, cta, u, 1)
        fence_async_smem()
        tc_fence_after()
        if lane == 0:
            for j in cutlass.range_constexpr(4):
                bd = Uint64(SDESC_SW128) | Uint64((st + Uint32(j * 8192)) >> Uint32(4))
                for k2 in cutlass.range_constexpr(2):
                    if cutlass.const_expr(not self.no_mma):
                        tc_mma_ts(taddr + Uint32(C_S),
                                  taddr + Uint32(C_Q + 32 * j + 16 * k2),
                                  bd + Uint64(k2 * 4), Uint32(IDESC_QK),
                                  Uint32(1 if (j + k2) > 0 else 0))
            ad = Uint64(SDESC_SW64) | Uint64((base + Uint32(OFF_QR)) >> Uint32(4))
            bd = Uint64(SDESC_SW64) | Uint64((st + Uint32(4 * 8192)) >> Uint32(4))
            if cutlass.const_expr(not self.no_mma):
                tc_mma(taddr + Uint32(C_S), ad, bd, Uint32(IDESC_QK), Uint32(1))
            tc_commit(_bar(base, B_SFULL))
            if last:
                tc_commit(_bar(base, B_QEMPTY))
        cute.arch.sync_warp()

    @cute.jit
    def pv(self, base, lane, taddr, u, i, last, qi, cta, stamp):
        """O^T += V^T (64 keys of the stage) P^T(u); the P^T slice is fetched once
        into the B collector for the 4 O^T blocks."""
        s = u % NS
        st = base + Uint32(OFF_KV) + Uint32(s) * Uint32(STAGE64)
        mbar_wait(_bar(base, B_PFULL), Uint32(u & 1))
        if cutlass.const_expr(self.stamps):
            if lane == 0:
                self.stamp(stamp, cta, u, 4)
        tc_fence_after()
        pbuf = base + Uint32(OFF_P) + Uint32(u % 2) * Uint32(P_BYTES)
        if lane == 0:
            bd = Uint64(SDESC_SW64) | Uint64(pbuf >> Uint32(4))
            for b in cutlass.range_constexpr(4):
                if i == 0:
                    mbar_wait(_bar(base, B_OFREE + b), Uint32((qi & 1) ^ 1))
                    tc_fence_after()
                ad = Uint64(SDESC_SW128) | Uint64((st + Uint32(b * 8192)) >> Uint32(4))
                acc = Uint32(1)
                if i == 0:
                    acc = Uint32(0)
                op = "fill" if b == 0 else ("lastuse" if b == 3 else "use")
                if cutlass.const_expr(not self.no_mma):
                    tc_mma_cb(taddr + Uint32(C_O + 96 * b), ad, bd, Uint32(IDESC_PV), acc,
                              op)
            tc_commit(_bar(base, B_KVEMPTY + s))
            tc_commit(_bar(base, B_PVDONE))
            if last:
                tc_commit(_bar(base, B_OFULL))
        cute.arch.sync_warp()

    # ------------------------------------------------------------------ softmax
    @cute.jit
    def softmax(self, base, tid, warp, lane, taddr, n_mine, lengths, cta, scale_log2,
                out_scale, xv, stamp):
        """Two warpgroups: thread (wg, h) = head h (TMEM lane h), S columns
        32 wg .. 32 wg + 31 of the 64-key subtile. The row max is exchanged through
        shared memory, so both halves take the same lazy-max decisions. Lanes
        96-127 (warps 3 and 7) are padding rows: they skip the math but join the
        barriers and the O^T rescales (lanes = dims there)."""
        lptr = cute.make_ptr(Int32, lengths, cute.AddressSpace.gmem, assumed_align=4)
        lens = cute.make_tensor(lptr, cute.make_layout(1 << 30))
        wg = warp // 4
        h = tid % 128
        lane_addr = taddr + (Uint32((warp % 4) * 32) << Uint32(16))
        real = h < H
        lim = Float32(P_PRE + THR)
        u = Int32(0)
        for qi in range(n_mine):
            row = cta + qi * self.grid
            n = lens[row]
            subs = _subs(n)
            # ---- Q latent row h -> TMEM (WG wg copies atoms 2 wg, 2 wg + 1), after
            # the previous token's last QK (seen as its S) and this token's Q load
            mbar_wait(_bar(base, B_QFULL), Uint32(qi & 1))
            if real:
                for jj in cutlass.range_constexpr(2):
                    j = 2 * wg + jj
                    words = []
                    for c in cutlass.range_constexpr(8):
                        a = base + Uint32(OFF_QN) + Uint32(j * H * 128) + Uint32(h * 128) + (
                            (Uint32(c) ^ Uint32(h % 8)) << Uint32(4))
                        w4 = ld_shared_v4u(a)
                        words.extend([w4[0], w4[1], w4[2], w4[3]])
                    tmem_st_u32(lane_addr + Uint32(C_Q) + Uint32(32 * j), words)
            tc_fence_before()
            cute.arch.sync_warp()
            if lane == 0:
                mbar_arrive(_bar(base, B_QTM))
            mp = Float32(0.0)  # m - P_PRE; subtile 0 sets it from its exact max
            lsum = Float32(0.0)
            for i in range(subs):
                mbar_wait(_bar(base, B_SFULL), Uint32(u & 1))
                if cutlass.const_expr(self.stamps):
                    if tid == 0:
                        self.stamp(stamp, cta, u, 2)
                tc_fence_after()
                sa = tmem_ld(lane_addr + Uint32(C_S) + Uint32(32 * wg), 32)
                tc_fence_before()
                cute.arch.sync_warp()
                if lane == 0:
                    mbar_arrive(_bar(base, B_SFREE))
                key0 = 64 * i + 32 * wg
                for k in cutlass.range_constexpr(32):
                    xv[k] = sa[k] * scale_log2 - mp
                if key0 + 32 > n:
                    for k in cutlass.range_constexpr(32):
                        if key0 + k >= n:
                            xv[k] = Float32(-math.inf)
                m8 = [cute.arch.fmax(cute.arch.fmax(xv[4 * q], xv[4 * q + 1]),
                                     cute.arch.fmax(xv[4 * q + 2], xv[4 * q + 3]))
                      for q in range(8)]
                m2 = [cute.arch.fmax(cute.arch.fmax(m8[4 * q], m8[4 * q + 1]),
                                     cute.arch.fmax(m8[4 * q + 2], m8[4 * q + 3]))
                      for q in range(2)]
                xm = cute.arch.fmax(m2[0], m2[1])
                mxb = base + Uint32(OFF_MX) + Uint32(u % 2) * Uint32(1024)
                st_shared_f32(mxb + Uint32((wg * 128 + h) * 4), xm)
                named_bar(1, 256)
                xm = cute.arch.fmax(xm, ld_shared_f32(mxb + Uint32(((1 - wg) * 128 + h) * 4)))
                grow = Int32(0)
                d = Float32(0.0)
                if real:
                    if i == 0:
                        if xm > Float32(-1e30):
                            d = xm - Float32(P_PRE)
                    elif xm > lim:
                        grow = Int32(1)
                        d = xm - Float32(P_PRE)
                if d != Float32(0.0):
                    mp = mp + d
                    for k in cutlass.range_constexpr(32):
                        xv[k] = xv[k] - d
                alpha = Float32(1.0)
                if grow != 0:
                    alpha = ex2(-d)
                    lsum = lsum * alpha
                # O^T rescale (all 8 warps: lanes = dims; WG wg blocks 2 wg, 2 wg + 1)
                fl = base + Uint32(OFF_FLAG + (u % 2) * 4)
                alb = base + Uint32(OFF_AL) + Uint32(u % 2) * Uint32(512)
                if wg == 0:
                    st_shared_f32(alb + Uint32(h * 4), alpha)
                if cute.arch.vote_any_sync(grow != 0) and lane == 0:
                    st_shared_u32(fl, Uint32(u + 1))
                named_bar(1, 256)
                if ld_shared_u32(fl) == Uint32(u + 1):
                    mbar_wait(_bar(base, B_PVDONE), Uint32((u - 1) & 1))
                    tc_fence_after()
                    for bb in cutlass.range_constexpr(2):
                        b = 2 * wg + bb
                        for c in cutlass.range_constexpr(6):
                            col = lane_addr + Uint32(C_O + 16 * c) + Uint32(96) * Uint32(b)
                            oa = tmem_ld(col, 16)
                            a4s = [ld_shared_v4f(alb + Uint32(64 * c + 16 * q))
                                   for q in range(4)]
                            tmem_st(col, [oa[k] * a4s[k // 4][k % 4] for k in range(16)])
                    tc_fence_before()
                # P row h, keys 32 wg .. +31 (E4M3) -> P buffer u % 2 (K-major SW64);
                # l sums the FP32 P (same accuracy as summing the E4M3 P, run13/14)
                if real:
                    pbuf = base + Uint32(OFF_P) + Uint32(u % 2) * Uint32(P_BYTES)
                    for cc in cutlass.range_constexpr(2):
                        c = 2 * wg + cc
                        w = []
                        ps = []
                        for q in cutlass.range_constexpr(4):
                            k = 16 * cc + 4 * q
                            p0 = ex2(xv[k])
                            p1 = ex2(xv[k + 1])
                            p2 = ex2(xv[k + 2])
                            p3 = ex2(xv[k + 3])
                            ps.append((p0 + p1) + (p2 + p3))
                            w.append(cvt_e4m3x2(p0, p1) | (cvt_e4m3x2(p2, p3) << Uint32(16)))
                        lsum = lsum + ((ps[0] + ps[1]) + (ps[2] + ps[3]))
                        st_shared_v4(pbuf + Uint32(h * 64) + (
                            (Uint32(c) ^ ((Uint32(h) >> Uint32(1)) & Uint32(3))) << Uint32(4)),
                            w[0], w[1], w[2], w[3])
                fence_async_smem()
                tc_fence_before()
                cute.arch.sync_warp()
                if cutlass.const_expr(self.stamps):
                    if tid == 0:
                        self.stamp(stamp, cta, u, 3)
                if lane == 0:
                    mbar_arrive(_bar(base, B_PFULL))
                u = u + 1
            # ---- token end: l = both halves; 1/l for the epilogue
            lsb = base + Uint32(OFF_MX)
            if wg == 1:
                st_shared_f32(lsb + Uint32(2048 + h * 4), lsum)
            named_bar(1, 256)
            if wg == 0:
                if real:
                    lt = lsum + ld_shared_f32(lsb + Uint32(2048 + h * 4))
                    iv = Float32(0.0)
                    if lt > Float32(0.0):
                        iv = out_scale / lt
                    st_shared_f32(base + Uint32(OFF_INV + ((qi % 2) * H + h) * 4), iv)
            cute.arch.sync_warp()
            if lane == 0:
                mbar_arrive(_bar(base, B_INV + qi % 2))

    # ------------------------------------------------------------------ epilogue
    @cute.jit
    def epilogue(self, base, tid, warp, lane, taddr, n_mine, cta, tm_out, stamp):
        e = tid - 32 * W_EPI
        wq = warp - W_EPI
        lane_addr = taddr + (Uint32(wq * 32) << Uint32(16))
        for qi in range(n_mine):
            row = cta + qi * self.grid
            mbar_wait(_bar(base, B_OFULL), Uint32(qi & 1))
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
                        for q in cutlass.range_constexpr(4):
                            hh = 32 * c + 4 * k8 + q
                            if cutlass.const_expr(not self.no_store):
                                st_shared_bf16(
                                    base + Uint32(OFF_STG) + Uint32((hh * 128 + e) * 2),
                                    o[4 * k8 + q] * iv[q])
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


@cache
def _compiled(grid: int, stamps: bool, knobs: tuple = (0, 0)):
    kernel = SparseMlaPrefillH2(grid, stamps, knobs)
    args = (Int64(0), Int32(1), Int64(0), Int32(64), Int64(0), Int32(1), Int64(0), Int64(0),
            Float32(1.0), Float32(1.0), Int32(64), Int64(0), cuda.CUstream(0))
    return cute.compile(kernel, *args, options=f"--ptxas-options -maxrregcount={REG_LAUNCH}")


def sparse_mla_prefill_h2(q8: torch.Tensor, kv_rows: torch.Tensor, slots: torch.Tensor,
                          lengths: torch.Tensor, scale: float, out_scale: float,
                          unit_rows: int = 64, out: torch.Tensor | None = None,
                          stamps: torch.Tensor | None = None,
                          knobs: tuple = (0, 0)) -> torch.Tensor:
    """Same contract as ``sparse_mla_sol.sparse_mla_prefill_sol``."""
    t = q8.shape[0]
    if q8.shape[1:] != (H, ROW) or not q8.is_contiguous():
        raise ValueError(f"sparse_mla_prefill_h2: q8 must be contiguous [T, {H}, {ROW}]")
    if out is None:
        out = torch.empty((t, H, D_NOPE), dtype=torch.bfloat16, device=q8.device)
    if not out.is_contiguous() or out.shape != (t, H, D_NOPE):
        raise ValueError("sparse_mla_prefill_h2: out must be contiguous [T, 96, 512]")
    if t == 0:
        return out
    sms = torch.cuda.get_device_properties(q8.device).multi_processor_count
    grid = min(sms, t)
    fn = _compiled(grid, stamps is not None, tuple(knobs))
    stream = cuda.CUstream(torch.cuda.current_stream(q8.device).cuda_stream)
    fn(Int64(q8.data_ptr()), Int32(t), Int64(kv_rows.data_ptr()), Int32(unit_rows),
       Int64(slots.data_ptr()), Int32(slots.stride(0)), Int64(lengths.data_ptr()),
       Int64(out.data_ptr()), Float32(scale * LOG2E), Float32(out_scale),
       Int32(kv_rows.shape[0]), Int64(0 if stamps is None else stamps.data_ptr()), stream)
    return out
