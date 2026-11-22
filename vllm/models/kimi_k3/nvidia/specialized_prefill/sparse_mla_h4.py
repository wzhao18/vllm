# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 DSA prefill sparse MLA on Rubin (sm_107): decoupled-softmax kernel.

Same TMEM plan and GEMM orientations as ``sparse_mla_h2`` (one CTA per SM and per
token; QK heads on M with Q latent in TMEM, PV keys on M with P^T in the B
collector; TMEM Q 128 + S 64 + O^T 384 = 576 columns), with the softmax taken off
the tensor-core critical path:

* softmax: thread = head, two warps per sub-partition own 32 keys each of a
  subtile and exchange only their row max (64-thread pair barrier, no CTA-wide
  barrier); the lazy-max decision is identical in both halves;
* the output warpgroup applies the rare lazy-max O^T rescale (after PV(u - 1),
  before PV(u)), copies the next token's Q latent into TMEM and runs the epilogue;
* the MMA warp issues QK(u) as soon as S(u - 1) has been read and the KV stage has
  landed, and PV(u - 2) after it (three P buffers), so QK never waits for a softmax;
* every barrier that a consumer can reach late is indexed by its buffer (P x3,
  Q RoPE x2), so no waiter can fall two phases behind.
"""

import math
from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint8, Uint32, Uint64

from vllm.models.kimi_k3.nvidia import k31_spin_guard as k31sg
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common import tensormap
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.primitives import asm
from vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla import (
    D_NOPE,
    LOG2E,
    ROW,
    SDESC_SW64,
    SDESC_SW128,
    H,
    cvt_e4m3x2,
    ex2,
    fence_async_smem,
    gather4,
    ld_shared_f32,
    ld_shared_u32,
    ld_shared_v4f,
    mbar_arrive,
    mbar_arrive_tx,
    mbar_init,
    mbar_wait,
    named_bar,
    st_global_b16,
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
from vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla_h2 import (
    bulk_g2s,
    ld_shared_v4s,
    ld_shared_v4u,
    tc_mma_ts,
    tmem_st_u32,
)
from vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla_sol import (
    bulk_wait_read0,
    st_shared_bf16,
    tc_mma_cb,
    tma_store2d,
)

P_PRE, THR = 4, 4.5  # P = 2^(s - m + P_PRE) <= 2^(P_PRE + THR) = 362 < 448 (E4M3 max)
LOADERS = 10  # gather warps
NS = 6  # 64-key KV stages
NP = 3  # P buffers (PV(u - 2) issued after QK(u))
IDX_BYTES = 2048 * 4  # one token's slot row (top-k <= 2048), double-buffered in smem
STAGE64 = 4 * 64 * 128 + 64 * 64  # 36864: 4 latent SW128 atoms + RoPE SW64
# warps: 0-7 softmax (2 WG x 32 keys; thread = head, warps 3 and 7 = padding rows
# 96-127), 8-11 output (Q copy, O^T rescale, epilogue), 12 MMA, 13-22 gather,
# 23 Q + slot rows. ptxas keeps every warp at the launch register count (80).
W_OUT, W_MMA, W_LD = 8, 12, 13
THREADS = 32 * 24
REG_LAUNCH = 65536 // THREADS // 8 * 8  # 80

# shared memory (bytes)
QR_BYTES = 128 * 64
P_BYTES = H * 64
OFF_QN = 0  # Q latent staging: 4 x [96 x 128 B] SW128 (copied into TMEM per token)
OFF_QR = OFF_QN + 4 * H * 128  # 2 x Q RoPE [128 rows x 64 B] SW64 (A of the RoPE MMA)
OFF_P = OFF_QR + 2 * QR_BYTES  # 3 x P [96 heads x 64 keys] E4M3, K-major SW64
OFF_AL = OFF_P + NP * P_BYTES  # [3][128] alpha of the subtile (lazy-max rescale)
OFF_FLAG = OFF_AL + NP * 128 * 4  # [3] u32: u + 1 when subtile u rescales
OFF_INV = OFF_FLAG + 16  # [2 token parity][96] out_scale / l
OFF_MX = OFF_INV + 2 * H * 4  # [2 parity][2 WG][128] half-row max; [128] l of WG 1
OFF_BAR = OFF_MX + 5 * 128 * 4
OFF_TMEM = OFF_BAR + 48 * 8
OFF_IDX = (OFF_TMEM + 16 + 1023) // 1024 * 1024  # 2 x slot rows (bulk-copied per token)
OFF_KV = OFF_IDX + 2 * IDX_BYTES
SMEM_BYTES = OFF_KV + NS * STAGE64
assert OFF_KV % 1024 == 0 and SMEM_BYTES <= 327 * 1024

# barriers (count 1 = one arrive / commit per phase, 4 = one arrive per warp)
B_SFULL, B_SFREE, B_QTM, B_QLFREE, B_OFULL = 0, 1, 2, 3, 4
B_QFULL, B_QREMPTY, B_INV = 5, 7, 9  # x2 (token parity)
B_OFREE = 11  # x4 (O^T block)
B_PFULL, B_OREADY, B_PEMPTY = 15, 18, 21  # x3 (P buffer u % 3)
B_IDXFULL, B_IDXEMPTY = 24, 26  # x2 (token parity)
B_KVFULL, B_KVEMPTY = 28, 34  # x NS
B_EPIDONE = 40  # epi_tma: the epilogue's BF16 staging (in the Q latent buffer) is free
assert B_KVEMPTY + NS <= B_EPIDONE
C_Q, C_S, C_O = 0, 128, 192

# instruction descriptors (dense kind::f8f6f4, E4M3 x E4M3 -> FP32, Rubin K64 bit 29)
IDESC_QK = (1 << 4) | (64 >> 3 << 17) | (128 >> 4 << 24) | (1 << 29)  # M128 N64
IDESC_PV = (1 << 4) | (H >> 3 << 17) | (128 >> 4 << 24) | (1 << 29) | (1 << 15)  # A MN

STAMP_CTAS, STAMP_TILES, STAMP_EV = 8, 64, 8


def _bar(base, i):
    return base + Uint32(OFF_BAR + 8 * i)


def mbar_wait_hint(bar, parity):
    """Bounded parity wait with a 10 ms suspend-time hint (k31_spin_guard)."""
    return k31sg.mbar_wait(bar, parity, hint=", 10000000")


k31sg.register_poll_helper(__file__, "mbar_wait_hint")


@asm()
def bulk_wait_read1():
    return "cp.async.bulk.wait_group.read 1;", "~{memory}", ()


@cute.jit
def _subs(n):
    """64-key subtiles of a row (at least one: empty rows still write zeros)."""
    t = (n + 63) // 64
    if t < 1:
        t = Int32(1)
    return t


class SparseMlaPrefillH4:
    def __init__(self, grid: int, stamps: bool = False, knobs: tuple = ()):
        self.grid, self.stamps = grid, stamps
        # diagnostics (timing only): no_store (epilogue skips the global stores),
        # no_gather (KV stages are signalled without any gather: stale data);
        # epi_tma: BF16 staging in the Q latent buffer + one TMA store per O^T block
        # hint: mbarrier waits carry a suspend-time hint (the waiting thread sleeps
        # until the phase completes instead of re-polling shared memory)
        # no_math: softmax warps skip all math (publish P unchanged); no_mma: the MMA
        # warp issues no MMAs (commits only)
        knobs = tuple(knobs) + (0,) * (6 - len(knobs))
        (self.no_store, self.no_gather, self.epi_tma, self.hint, self.no_math,
         self.no_mma) = knobs[:6]

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
        self.kernel(qn, qr, kvm, kvr, om, out, rows, unit_rows, slots, slot_stride,
                    lengths, scale_log2, out_scale, stamp).launch(
            grid=(self.grid, 1, 1), block=(THREADS, 1, 1), smem=SMEM_BYTES,
            stream=stream)

    @cute.kernel
    def kernel(self, qn: tensormap.GridMap, qr: tensormap.GridMap,
               kvm: tensormap.GridMap, kvr: tensormap.GridMap, om: tensormap.GridMap,
               out: Int64, rows: Int32, unit_rows: Int32, slots: Int64, slot_stride: Int32,
               lengths: Int64, scale_log2: Float32, out_scale: Float32, stamp: Int64):
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(tid // 32)
        lane = tid % 32
        smem = cutlass.utils.SmemAllocator()
        raw = smem.allocate_array(Uint8, SMEM_BYTES, byte_alignment=1024)
        base = Uint32(raw.toint())
        if tid == 0:
            for i, cnt in ((B_SFULL, 1), (B_SFREE, 8), (B_QTM, 4), (B_QLFREE, 4),
                           (B_OFULL, 1), (B_EPIDONE, 1)):
                mbar_init(_bar(base, i), Uint32(cnt))
            for j in cutlass.range_constexpr(2):
                mbar_init(_bar(base, B_QFULL + j), Uint32(1))
                mbar_init(_bar(base, B_QREMPTY + j), Uint32(1))
                mbar_init(_bar(base, B_INV + j), Uint32(8))
                mbar_init(_bar(base, B_IDXFULL + j), Uint32(1))
                mbar_init(_bar(base, B_IDXEMPTY + j), Uint32(LOADERS))
            for j in cutlass.range_constexpr(4):
                mbar_init(_bar(base, B_OFREE + j), Uint32(4))
            for j in cutlass.range_constexpr(NP):
                mbar_init(_bar(base, B_PFULL + j), Uint32(8))
                mbar_init(_bar(base, B_OREADY + j), Uint32(4))
                mbar_init(_bar(base, B_PEMPTY + j), Uint32(1))
            for s in cutlass.range_constexpr(NS):
                mbar_init(_bar(base, B_KVFULL + s), Uint32(1))
                mbar_init(_bar(base, B_KVEMPTY + s), Uint32(1))
            for j in cutlass.range_constexpr(NP):
                st_shared_u32(base + Uint32(OFF_FLAG + 4 * j), Uint32(0))
            cute.arch.mbarrier_init_fence()
        if warp == 0:
            tmem_alloc_exclusive(base + Uint32(OFF_TMEM))
        tc_fence_before()
        cute.arch.sync_threads()
        tc_fence_after()
        taddr = ld_shared_u32(base + Uint32(OFF_TMEM))
        n_mine = (rows - cta + self.grid - 1) // self.grid
        if warp < W_OUT:
            self.softmax(base, tid, warp, lane, taddr, n_mine, lengths, cta,
                         scale_log2, out_scale, stamp)
        elif warp < W_MMA:
            self.output(base, tid, warp, lane, taddr, n_mine, lengths, cta, out,
                        tensormap.map_address(om), stamp)
        else:
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
    def wait(self, bar, parity):
        if cutlass.const_expr(self.hint):
            mbar_wait_hint(bar, parity)
        else:
            mbar_wait(bar, parity)

    @cute.jit
    def stamp(self, ptr, cta, g, ev: cutlass.Constexpr):
        if cutlass.const_expr(self.stamps):
            if cta < STAMP_CTAS and g < STAMP_TILES:
                stamp_clock(ptr + Int64(((cta * STAMP_TILES + g) * STAMP_EV + ev) * 8))

    @cute.jit
    def wstamp(self, ptr, cta, u, w, k: cutlass.Constexpr):
        """Per-warp stamps of CTA 0: w 0-7 softmax (k 0 S loaded, 1 max exchanged, 2
        P published, 3 rescale requested), 8 MMA (k 0 at the KV wait, 1 QK issued, 2
        PV issued, 3 O ready seen), 9 gather warps 0-3 (k = pw) done; output
        warp 0 at w 10 k 0 (P seen) / k 1 (O ready) / k 3 (rescaled)."""
        if cutlass.const_expr(self.stamps):
            if cta == 0 and u < STAMP_TILES:
                stamp_clock(ptr + Int64((STAMP_CTAS * STAMP_TILES * STAMP_EV
                                         + (u * 12 + w) * 4 + k) * 8))

    # ------------------------------------------------------------------ gather
    @cute.jit
    def loader(self, base, pw, lane, cta, n_mine, tm_qn, tm_qr, tm_kv, tm_kr, unit_rows,
               slots, slot_stride, lengths, stamp):
        quads = (16 - pw + LOADERS - 1) // LOADERS
        lptr = cute.make_ptr(Int32, lengths, cute.AddressSpace.gmem, assumed_align=4)
        lens = cute.make_tensor(lptr, cute.make_layout(1 << 30))
        if pw == LOADERS:
            # Q warp: latent staging (single) is free once the output WG copied the
            # previous token's Q into TMEM; RoPE buffer qi % 2 once token qi - 2's last
            # QK has completed
            if lane == 0:
                if n_mine > 0:
                    self.idx_copy(base, cta, Int32(0), lens, slots, slot_stride)
                for qi in range(n_mine):
                    row = cta + qi * self.grid
                    if qi + 1 < n_mine:
                        self.idx_copy(base, cta, qi + 1, lens, slots, slot_stride)
                    if qi >= 1:
                        self.wait(_bar(base, B_QLFREE), Uint32((qi - 1) & 1))
                    if qi >= 2:
                        self.wait(_bar(base, B_QREMPTY + qi % 2), Uint32((qi // 2 - 1) & 1))
                        if cutlass.const_expr(self.epi_tma):
                            self.wait(_bar(base, B_EPIDONE), Uint32((qi - 2) & 1))
                    full = _bar(base, B_QFULL + qi % 2)
                    mbar_arrive_tx(full, Uint32(4 * H * 128 + H * 64))
                    for j in cutlass.range_constexpr(4):
                        tma_tile2d(base + Uint32(OFF_QN + j * H * 128), tm_qn, full,
                                   Int32(128 * j), row * H)
                    tma_tile2d(base + Uint32(OFF_QR) + Uint32(qi % 2) * Uint32(QR_BYTES),
                               tm_qr, full, Int32(0), row * H)
        elif pw < LOADERS:
            g = Int32(0)
            for qi in range(n_mine):
                row = cta + qi * self.grid
                n = lens[row]
                subs = _subs(n)
                par = qi % 2
                ibuf = base + Uint32(OFF_IDX) + Uint32(par) * Uint32(IDX_BYTES)
                self.wait(_bar(base, B_IDXFULL + par), Uint32((qi // 2) & 1))
                for t in range(subs):
                    s = g % NS
                    st = base + Uint32(OFF_KV) + Uint32(s) * Uint32(STAGE64)
                    full = _bar(base, B_KVFULL + s)
                    q = pw + LOADERS * lane
                    if lane >= quads:
                        q = pw  # idle lanes read a valid slot of the tile (unused)
                    key = t * 64 + 4 * q
                    r = ld_shared_v4s(ibuf + Uint32(key * 4))
                    if lane == 0:
                        self.wait(_bar(base, B_KVEMPTY + s), Uint32(((g // NS) & 1) ^ 1))
                        if pw == 0:
                            if cutlass.const_expr(self.no_gather):
                                mbar_arrive(full)
                            else:
                                mbar_arrive_tx(full, Uint32(STAGE64))
                            self.stamp(stamp, cta, g, 0)
                    cute.arch.sync_warp()
                    if cutlass.const_expr(not self.no_gather):
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
                        if lane == 0 and pw < 4:
                            self.wstamp(stamp, cta, g, 9, pw)
                    g = g + 1
                cute.arch.sync_warp()
                if lane == 0:
                    mbar_arrive(_bar(base, B_IDXEMPTY + par))

    @cute.jit
    def idx_copy(self, base, cta, qi, lens, slots, slot_stride):
        """Slot row of token qi -> smem buffer qi % 2 (one bulk copy)."""
        row = cta + qi * self.grid
        par = qi % 2
        self.wait(_bar(base, B_IDXEMPTY + par), Uint32(((qi // 2) & 1) ^ 1))
        nb = Uint32(_subs(lens[row]) * 256)
        mbar_arrive_tx(_bar(base, B_IDXFULL + par), nb)
        bulk_g2s(base + Uint32(OFF_IDX) + Uint32(par) * Uint32(IDX_BYTES),
                 slots + Int64(row) * Int64(slot_stride) * Int64(4), nb,
                 _bar(base, B_IDXFULL + par))

    # ------------------------------------------------------------------ MMA
    @cute.jit
    def mma(self, base, lane, taddr, n_mine, lengths, cta, stamp):
        """Global 64-key subtile counter u: wait S(u - 1) read, QK(u), then PV(u - 2)
        (its P was published before softmax(u - 1) started)."""
        lptr = cute.make_ptr(Int32, lengths, cute.AddressSpace.gmem, assumed_align=4)
        lens = cute.make_tensor(lptr, cute.make_layout(1 << 30))
        u = Int32(0)
        # pending PVs: a = older, b = newer; (u, i, last, qi), u < 0 = none
        au, ai, al, aq = Int32(-1), Int32(0), Int32(0), Int32(0)
        bu, bi, bl, bq = Int32(-1), Int32(0), Int32(0), Int32(0)
        for qi in range(n_mine):
            row = cta + qi * self.grid
            subs = _subs(lens[row])
            self.wait(_bar(base, B_QTM), Uint32(qi & 1))
            tc_fence_after()
            for i in range(subs):
                if u > 0:
                    self.wait(_bar(base, B_SFREE), Uint32((u - 1) & 1))
                    tc_fence_after()
                self.qk(base, lane, taddr, u, qi, i == subs - 1, cta, stamp)
                if au >= 0:
                    self.pv(base, lane, taddr, au, ai, al, aq, cta, stamp)
                au, ai, al, aq = bu, bi, bl, bq
                bu, bi, bq = u, i, qi
                bl = Int32(0)
                if i == subs - 1:
                    bl = Int32(1)
                u = u + 1
        if au >= 0:
            self.pv(base, lane, taddr, au, ai, al, aq, cta, stamp)
        if bu >= 0:
            self.pv(base, lane, taddr, bu, bi, bl, bq, cta, stamp)

    @cute.jit
    def qk(self, base, lane, taddr, u, qi, last, cta, stamp):
        s = u % NS
        st = base + Uint32(OFF_KV) + Uint32(s) * Uint32(STAGE64)
        if lane == 0:
            self.wstamp(stamp, cta, u, 8, 0)
        self.wait(_bar(base, B_KVFULL + s), Uint32((u // NS) & 1))
        if lane == 0:
            self.stamp(stamp, cta, u, 1)
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
            qra = base + Uint32(OFF_QR) + Uint32(qi % 2) * Uint32(QR_BYTES)
            ad = Uint64(SDESC_SW64) | Uint64(qra >> Uint32(4))
            bd = Uint64(SDESC_SW64) | Uint64((st + Uint32(4 * 8192)) >> Uint32(4))
            if cutlass.const_expr(not self.no_mma):
                tc_mma(taddr + Uint32(C_S), ad, bd, Uint32(IDESC_QK), Uint32(1))
            self.wstamp(stamp, cta, u, 8, 1)
            tc_commit(_bar(base, B_SFULL))
            if last:
                tc_commit(_bar(base, B_QREMPTY + qi % 2))
        cute.arch.sync_warp()

    @cute.jit
    def pv(self, base, lane, taddr, u, i, last, qi, cta, stamp):
        """O^T += V^T (64 keys of the stage) P^T(u), after the output WG has seen
        P(u) (and applied its rescale); P^T is fetched once into the B collector."""
        s = u % NS
        st = base + Uint32(OFF_KV) + Uint32(s) * Uint32(STAGE64)
        self.wait(_bar(base, B_OREADY + u % NP), Uint32((u // NP) & 1))
        if lane == 0:
            self.stamp(stamp, cta, u, 4)
            self.wstamp(stamp, cta, u, 8, 3)
        tc_fence_after()
        pbuf = base + Uint32(OFF_P) + Uint32(u % NP) * Uint32(P_BYTES)
        if lane == 0:
            bd = Uint64(SDESC_SW64) | Uint64(pbuf >> Uint32(4))
            for b in cutlass.range_constexpr(4):
                if i == 0:
                    self.wait(_bar(base, B_OFREE + b), Uint32((qi & 1) ^ 1))
                    tc_fence_after()
                ad = Uint64(SDESC_SW128) | Uint64((st + Uint32(b * 8192)) >> Uint32(4))
                acc = Uint32(1)
                if i == 0:
                    acc = Uint32(0)
                op = "fill" if b == 0 else ("lastuse" if b == 3 else "use")
                if cutlass.const_expr(not self.no_mma):
                    tc_mma_cb(taddr + Uint32(C_O + 96 * b), ad, bd, Uint32(IDESC_PV), acc,
                              op)
            self.wstamp(stamp, cta, u, 8, 2)
            tc_commit(_bar(base, B_KVEMPTY + s))
            tc_commit(_bar(base, B_PEMPTY + u % NP))
            if last != 0:
                tc_commit(_bar(base, B_OFULL))
        cute.arch.sync_warp()

    # ------------------------------------------------------------------ softmax
    @cute.jit
    def softmax(self, base, tid, warp, lane, taddr, n_mine, lengths, cta, scale_log2,
                out_scale, stamp):
        """Thread (wg, h) = head h (TMEM lane h), S columns 32 wg .. 32 wg + 31 of the
        64-key subtile. Warps w and w + 4 (same sub-partition, same heads) exchange
        their half-row max through shared memory under a 64-thread barrier, so both
        halves take the same lazy-max decision. Warps 3 and 7 (padding rows) only
        join the mbarriers."""
        lptr = cute.make_ptr(Int32, lengths, cute.AddressSpace.gmem, assumed_align=4)
        lens = cute.make_tensor(lptr, cute.make_layout(1 << 30))
        wg = warp // 4
        h = tid % 128
        lane_addr = taddr + (Uint32((warp % 4) * 32) << Uint32(16))
        real = h < H
        real_m = h < (0 if self.no_math else H)  # no_math: no thread does softmax math
        pair = Uint32(1 + warp % 4)  # named barrier of the warp pair (1-3 used)
        lim = Float32(P_PRE + THR)
        xv = cute.make_rmem_tensor((32,), Float32)
        u = Int32(0)
        for qi in range(n_mine):
            row = cta + qi * self.grid
            n = lens[row]
            subs = _subs(n)
            mp = Float32(0.0)  # m - P_PRE; subtile 0 sets it from its exact max
            lsum = Float32(0.0)
            for i in range(subs):
                self.wait(_bar(base, B_SFULL), Uint32(u & 1))
                if tid == 0:
                    self.stamp(stamp, cta, u, 2)
                tc_fence_after()
                sa = tmem_ld(lane_addr + Uint32(C_S) + Uint32(32 * wg), 32)
                tc_fence_before()
                cute.arch.sync_warp()
                if lane == 0:
                    mbar_arrive(_bar(base, B_SFREE))
                    self.wstamp(stamp, cta, u, warp, 0)
                grow = Int32(0)
                alpha = Float32(1.0)
                if real_m:
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
                    named_bar(pair, 64)
                    xm = cute.arch.fmax(
                        xm, ld_shared_f32(mxb + Uint32(((1 - wg) * 128 + h) * 4)))
                    d = Float32(0.0)
                    if i == 0:
                        if xm > Float32(-1e30):
                            d = xm - Float32(P_PRE)
                    elif xm > lim:
                        grow = Int32(1)
                        d = xm - Float32(P_PRE)
                        alpha = ex2(-d)
                        lsum = lsum * alpha
                    if d != Float32(0.0):
                        mp = mp + d
                        for k in cutlass.range_constexpr(32):
                            xv[k] = xv[k] - d
                if lane == 0:
                    self.wstamp(stamp, cta, u, warp, 1)
                if u >= NP:
                    self.wait(_bar(base, B_PEMPTY + u % NP), Uint32((u // NP - 1) & 1))
                if real_m:
                    pbuf = base + Uint32(OFF_P) + Uint32(u % NP) * Uint32(P_BYTES)
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
                    if wg == 0:
                        st_shared_f32(base + Uint32(OFF_AL) + Uint32(u % NP) * Uint32(512)
                                      + Uint32(h * 4), alpha)
                if cute.arch.vote_any_sync(grow != 0) and lane == 0:
                    st_shared_u32(base + Uint32(OFF_FLAG) + Uint32(u % NP) * Uint32(4),
                                  Uint32(u + 1))
                    self.wstamp(stamp, cta, u, warp, 3)
                fence_async_smem()
                cute.arch.sync_warp()
                if tid == 0:
                    self.stamp(stamp, cta, u, 3)
                if lane == 0:
                    self.wstamp(stamp, cta, u, warp, 2)
                    mbar_arrive(_bar(base, B_PFULL + u % NP))
                u = u + 1
            # ---- token end: l = both halves; 1/l for the epilogue
            if real:
                lsb = base + Uint32(OFF_MX + 4 * 128 * 4)
                if wg == 1:
                    st_shared_f32(lsb + Uint32(h * 4), lsum)
                named_bar(pair, 64)
                if wg == 0:
                    lt = lsum + ld_shared_f32(lsb + Uint32(h * 4))
                    iv = Float32(0.0)
                    if lt > Float32(0.0):
                        iv = out_scale / lt
                    st_shared_f32(base + Uint32(OFF_INV + ((qi % 2) * H + h) * 4), iv)
                # WG 1 must not overwrite l (next token) before WG 0 read it: the next
                # token's first max exchange (same pair barrier) orders it
            cute.arch.sync_warp()
            if lane == 0:
                mbar_arrive(_bar(base, B_INV + qi % 2))

    # ------------------------------------------------------------------ output
    @cute.jit
    def output(self, base, tid, warp, lane, taddr, n_mine, lengths, cta, out, tm_o,
               stamp):
        """Per subtile u (in order): after P(u) is published, apply its lazy-max
        rescale to O^T if any head grew (after PV(u - 1)), then release PV(u). Q
        latent of token qi + 1 goes into TMEM once token qi's last QK is done; the
        epilogue of token qi runs before the subtiles of token qi + 1."""
        lptr = cute.make_ptr(Int32, lengths, cute.AddressSpace.gmem, assumed_align=4)
        lens = cute.make_tensor(lptr, cute.make_layout(1 << 30))
        wq = warp - W_OUT
        e = tid - 32 * W_OUT  # TMEM lane: head (Q copy) / dim within an O^T block
        lane_addr = taddr + (Uint32(wq * 32) << Uint32(16))
        if n_mine > 0:
            self.q_copy(base, e, lane, lane_addr, Int32(0))
        u = Int32(0)
        for qi in range(n_mine):
            row = cta + qi * self.grid
            subs = _subs(lens[row])
            for i in range(subs):
                if i == subs - 1:
                    if qi + 1 < n_mine:
                        self.q_copy(base, e, lane, lane_addr, qi + 1)
                self.wait(_bar(base, B_PFULL + u % NP), Uint32((u // NP) & 1))
                if wq == 0 and lane == 0:
                    self.wstamp(stamp, cta, u, 10, 0)
                if ld_shared_u32(base + Uint32(OFF_FLAG) + Uint32(u % NP) * Uint32(4)) == Uint32(u + 1):
                    # O^T(u - 1) complete: PV(u - 1) released P buffer (u - 1) % 3
                    self.wait(_bar(base, B_PEMPTY + (u - 1) % NP), Uint32(((u - 1) // NP) & 1))
                    tc_fence_after()
                    alb = base + Uint32(OFF_AL) + Uint32(u % NP) * Uint32(512)
                    for c in cutlass.range_constexpr(6):
                        al = []
                        for q in cutlass.range_constexpr(4):
                            a4 = ld_shared_v4f(alb + Uint32(64 * c + 16 * q))
                            al.extend([a4[0], a4[1], a4[2], a4[3]])
                        for b in cutlass.range_constexpr(4):
                            col = lane_addr + Uint32(C_O + 96 * b + 16 * c)
                            oa = tmem_ld(col, 16)
                            tmem_st(col, [oa[k] * al[k] for k in range(16)])
                    tc_fence_before()
                    if wq == 0 and lane == 0:
                        self.wstamp(stamp, cta, u, 10, 3)
                cute.arch.sync_warp()
                if lane == 0:
                    mbar_arrive(_bar(base, B_OREADY + u % NP))
                    if wq == 0:
                        self.wstamp(stamp, cta, u, 10, 1)
                u = u + 1
            if cutlass.const_expr(self.epi_tma):
                self.epilogue_tma(base, e, lane, lane_addr, qi, cta, tm_o, stamp)
            else:
                self.epilogue(base, e, lane, lane_addr, qi, cta, out, stamp)

    @cute.jit
    def q_copy(self, base, e, lane, lane_addr, qi):
        """Q latent of token qi (smem staging, SW128) -> TMEM columns C_Q (lane = head),
        once the previous token's last QK has completed."""
        self.wait(_bar(base, B_QFULL + qi % 2), Uint32((qi // 2) & 1))
        if qi >= 1:
            self.wait(_bar(base, B_QREMPTY + (qi - 1) % 2), Uint32(((qi - 1) // 2) & 1))
        tc_fence_after()
        if e < H:
            for j in cutlass.range_constexpr(4):
                words = []
                for c in cutlass.range_constexpr(8):
                    a = base + Uint32(OFF_QN) + Uint32(j * H * 128) + Uint32(e * 128) + (
                        (Uint32(c) ^ Uint32(e % 8)) << Uint32(4))
                    w4 = ld_shared_v4u(a)
                    words.extend([w4[0], w4[1], w4[2], w4[3]])
                tmem_st_u32(lane_addr + Uint32(C_Q) + Uint32(32 * j), words)
        tc_fence_before()
        cute.arch.sync_warp()
        if lane == 0:
            mbar_arrive(_bar(base, B_QLFREE))
            mbar_arrive(_bar(base, B_QTM))

    # ------------------------------------------------------------------ epilogue
    @cute.jit
    def epilogue(self, base, e, lane, lane_addr, qi, cta, out, stamp):
        """Thread = dim e of a 128-dim O^T block: O * (s / l) -> BF16 stored straight
        to global; block b's TMEM is released (OFREE b) as soon as it has been read."""
        row = cta + qi * self.grid
        self.wait(_bar(base, B_OFULL), Uint32(qi & 1))
        self.wait(_bar(base, B_INV + qi % 2), Uint32((qi // 2) & 1))
        inv_base = base + Uint32(OFF_INV) + Uint32(qi % 2) * Uint32(H * 4)
        tc_fence_after()
        if e == 0:
            self.stamp(stamp, cta, Int32(qi), 5)
        orow = out + Int64(row) * Int64(H * D_NOPE * 2) + Int64(e * 2)
        for b in cutlass.range_constexpr(4):
            for c in cutlass.range_constexpr(3):
                o = tmem_ld(lane_addr + Uint32(C_O + 96 * b + 32 * c), 32)
                if c == 2:
                    tc_fence_before()
                    cute.arch.sync_warp()
                    if lane == 0:
                        mbar_arrive(_bar(base, B_OFREE + b))
                for k8 in cutlass.range_constexpr(8):
                    iv = ld_shared_v4f(inv_base + Uint32((32 * c + 4 * k8) * 4))
                    for q in cutlass.range_constexpr(4):
                        hh = 32 * c + 4 * k8 + q
                        if cutlass.const_expr(not self.no_store):
                            st_global_b16(orow + Int64((hh * D_NOPE + 128 * b) * 2),
                                          o[4 * k8 + q] * iv[q])
        if e == 0:
            self.stamp(stamp, cta, Int32(qi), 6)


    @cute.jit
    def epilogue_tma(self, base, e, lane, lane_addr, qi, cta, tm_o, stamp):
        """Thread = dim e: O * (s / l) -> BF16 column e of a [96 heads x 128 dims]
        staging tile (two tiles in the Q latent buffer, free between this token's Q
        copy and the load of token qi + 2's Q), one TMA store per O^T block; block b's
        TMEM is released as soon as its last chunk has been read."""
        row = cta + qi * self.grid
        self.wait(_bar(base, B_OFULL), Uint32(qi & 1))
        self.wait(_bar(base, B_INV + qi % 2), Uint32((qi // 2) & 1))
        inv_base = base + Uint32(OFF_INV) + Uint32(qi % 2) * Uint32(H * 4)
        tc_fence_after()
        if e == 0:
            self.stamp(stamp, cta, Int32(qi), 5)
        # every thread's Q copy (reads of the same buffer) is done
        named_bar(Uint32(5), 128)
        for b in cutlass.range_constexpr(4):
            buf = base + Uint32(OFF_QN + (b % 2) * H * 256)
            if b >= 2:
                if e == 0:
                    bulk_wait_read1()
                named_bar(Uint32(5), 128)
            for c in cutlass.range_constexpr(3):
                o = tmem_ld(lane_addr + Uint32(C_O + 96 * b + 32 * c), 32)
                if c == 2:
                    tc_fence_before()
                    cute.arch.sync_warp()
                    if lane == 0:
                        mbar_arrive(_bar(base, B_OFREE + b))
                for k8 in cutlass.range_constexpr(8):
                    iv = ld_shared_v4f(inv_base + Uint32((32 * c + 4 * k8) * 4))
                    for q in cutlass.range_constexpr(4):
                        hh = 32 * c + 4 * k8 + q
                        st_shared_bf16(buf + Uint32(hh * 256 + e * 2), o[4 * k8 + q] * iv[q])
            fence_async_smem()
            named_bar(Uint32(5), 128)
            if e == 0:
                if cutlass.const_expr(not self.no_store):
                    tma_store2d(tm_o, buf, Int32(128 * b), row * H)
        if e == 0:
            bulk_wait_read0()
            mbar_arrive(_bar(base, B_EPIDONE))
            self.stamp(stamp, cta, Int32(qi), 6)


@cache
def _compiled(grid: int, stamps: bool, knobs: tuple = ()):
    kernel = SparseMlaPrefillH4(grid, stamps, knobs)
    args = (Int64(0), Int32(1), Int64(0), Int32(64), Int64(0), Int32(1), Int64(0), Int64(0),
            Float32(1.0), Float32(1.0), Int32(64), Int64(0), cuda.CUstream(0))
    return cute.compile(kernel, *args, options=f"--ptxas-options -maxrregcount={REG_LAUNCH}")


def sparse_mla_prefill_h4(q8: torch.Tensor, kv_rows: torch.Tensor, slots: torch.Tensor,
                          lengths: torch.Tensor, scale: float, out_scale: float,
                          unit_rows: int = 64, out: torch.Tensor | None = None,
                          stamps: torch.Tensor | None = None,
                          knobs: tuple = ()) -> torch.Tensor:
    """Same contract as ``sparse_mla_sol.sparse_mla_prefill_sol``."""
    t = q8.shape[0]
    if q8.shape[1:] != (H, ROW) or not q8.is_contiguous():
        raise ValueError(f"sparse_mla_prefill_h4: q8 must be contiguous [T, {H}, {ROW}]")
    if out is None:
        out = torch.empty((t, H, D_NOPE), dtype=torch.bfloat16, device=q8.device)
    if not out.is_contiguous() or out.shape != (t, H, D_NOPE):
        raise ValueError("sparse_mla_prefill_h4: out must be contiguous [T, 96, 512]")
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
