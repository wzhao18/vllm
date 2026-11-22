# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 DSA prefill sparse MLA on Rubin (sm_107): double-buffered-S kernel.

``sparse_mla_h4`` with two S buffers. One S buffer serializes every 64-key subtile on
the loop commit -> softmax wake -> S load -> release -> MMA wake -> QK -> commit,
about 0.9K clocks of handoffs alone plus the QK latency (P-MLA run45). Two S buffers
let QK(u + 1) run while softmax(u) works. The 64 TMEM columns come from the Q
latent: dims 0-255 stay in TMEM (TS MMAs), dims 256-511 and the RoPE part are read
from shared memory (SS MMAs, double-buffered per token). TMEM: Q 64 + S 2 x 64 +
O^T 384 = 576 columns. To fit six KV stages, the gather warps load their slot
indices with prefetched global loads (no slot-row buffer in shared memory).
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
    ld_global_v4_s32,
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
STAGE64 = 4 * 64 * 128 + 64 * 64  # 36864: 4 latent SW128 atoms + RoPE SW64
# warps: 0-7 softmax (2 WG x 32 keys; thread = head, warps 3 and 7 = padding rows
# 96-127), 8-11 output (Q copy, O^T rescale, epilogue), 12 MMA, 13-22 gather,
# 23 Q + slot rows. ptxas keeps every warp at the launch register count (80).
W_OUT, W_MMA, W_LD = 8, 12, 13
THREADS = 32 * 24
REG_LAUNCH = 65536 // THREADS // 8 * 8  # 80
REG_SM2, REG_LD2 = 136, 40  # pp2: softmax WGs / loader + MMA WGs (output WG stays 80)
assert 256 * REG_SM2 + 128 * REG_LAUNCH + 384 * REG_LD2 <= THREADS * REG_LAUNCH

# shared memory (bytes). SS A tiles have M = 128 rows; only 96 are loaded and rows
# 96-127 read whatever follows (valid shared memory, padding rows of S only).
ATOM = H * 128  # one Q latent SW128 atom (96 rows x 128 B)
QS_BYTES = 2 * ATOM + 128 * 64  # per token: latent atoms 2-3 + RoPE [128 x 64 B] SW64
P_BYTES = H * 64
OFF_QN = 0  # Q latent atoms 0-1 staging (copied into TMEM per token; epilogue staging)
OFF_QS = OFF_QN + 2 * ATOM  # 2 x QS_BYTES (token parity): A of the SS QK MMAs
OFF_P = OFF_QS + 2 * QS_BYTES  # 3 x P [96 heads x 64 keys] E4M3, K-major SW64
OFF_AL = OFF_P + NP * P_BYTES  # [3][96] alpha of the subtile (lazy-max rescale)
OFF_FLAG = OFF_AL + NP * H * 4  # [3] u32: u + 1 when subtile u rescales
OFF_INV = OFF_FLAG + 16  # [2 token parity][96] out_scale / l
# [2 parity][2 WG][96] half-row max; at a token end WG 1's l goes to the WG 0 slot of
# the next subtile's parity (WG 0 reads it before it writes that slot again)
OFF_MX = OFF_INV + 2 * H * 4
OFF_BAR = OFF_MX + 4 * H * 4
OFF_TMEM = OFF_BAR + 39 * 8
OFF_KV = (OFF_TMEM + 16 + 1023) // 1024 * 1024
SMEM_BYTES = OFF_KV + NS * STAGE64
# rows 96-127 of each RoPE A tile are read by the MMA but never loaded (padding rows of
# S): the ping-pong softmax keeps its lazy-max mailbox there ([2][96] f32, buffer 0) and
# its token-end (l, m) of WG 1 ([2 token parity][2][96] f32, buffer 1)
OFF_MBOX = OFF_QS + 2 * ATOM + H * 64
OFF_TOK = OFF_QS + QS_BYTES + 2 * ATOM + H * 64
assert 2 * H * 4 <= 32 * 64 and 4 * H * 4 <= 32 * 64
assert OFF_QS % 1024 == 0 and QS_BYTES % 1024 == 0 and ATOM % 1024 == 0
assert OFF_KV % 1024 == 0 and SMEM_BYTES <= 334848, SMEM_BYTES

# barriers (count 1 = one arrive / commit per phase, 4 / 8 = one arrive per warp)
B_SFULL, B_SFREE = 0, 2  # x2 (S buffer u % 2)
B_QTM, B_QLFREE, B_OFULL, B_EPIDONE = 4, 5, 6, 7
B_QFULL, B_QREMPTY, B_INV = 8, 10, 12  # x2 (token parity)
B_OFREE = 14  # x4 (O^T block)
B_PFULL, B_OREADY, B_PEMPTY = 18, 21, 24  # x3 (P buffer u % 3)
B_KVFULL, B_KVEMPTY = 27, 33  # x NS
assert B_KVEMPTY + NS <= 39
C_Q, C_S, C_O = 0, 64, 192  # Q latent dims 0-255; S buffers at 64 and 128; O^T

# instruction descriptors (dense kind::f8f6f4, E4M3 x E4M3 -> FP32, Rubin K64 bit 29)
IDESC_QK = (1 << 4) | (64 >> 3 << 17) | (128 >> 4 << 24) | (1 << 29)  # M128 N64
IDESC_PV = (1 << 4) | (H >> 3 << 17) | (128 >> 4 << 24) | (1 << 29) | (1 << 15)  # A MN

STAMP_CTAS, STAMP_TILES, STAMP_EV = 8, 64, 8


def _tree_max(vals):
    """Max of a list of traced values as a 4-ary tree (trace-time Python)."""
    vals = list(vals)
    while len(vals) > 1:
        vals = [cute.arch.fmax(cute.arch.fmax(vals[q], vals[q + 1]),
                               cute.arch.fmax(vals[q + 2], vals[q + 3]))
                if q + 3 < len(vals) else cute.arch.fmax(vals[q], vals[q + 1])
                for q in range(0, len(vals), 4)]
    return vals[0]


def _bar(base, i):
    return base + Uint32(OFF_BAR + 8 * i)


@asm()
def tmem_ld16x256(addr):
    """tcgen05.ld 16x256b.x4: thread t gets, for column groups g = 0..3 (8 columns
    each), (lane t / 4, cols 8 g + 2 (t % 4) + {0, 1}) and (lane 8 + t / 4, same)."""
    regs = ",".join(f"${i}" for i in range(16))
    code = (f"tcgen05.ld.sync.aligned.16x256b.x4.b32 {{{regs}}}, [$16]; "
            "tcgen05.wait::ld.sync.aligned;")
    return code, ",".join(["=f"] * 16 + ["r", "~{memory}"]), (Uint32(addr),), (Float32,) * 16


@asm(Uint32, pure=True)
def bf16x2(lo, hi):
    return "cvt.rn.bf16x2.f32 $0, $2, $1;", "=r,f,f", (Float32(lo), Float32(hi))


@asm()
def stmatrix_t4(addr, r0, r1, r2, r3):
    """stmatrix m8n8.x4.trans b16: matrix k's stored row j at the address of thread
    8 k + j holds column j of the fragment matrix k."""
    return ("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [$0], {$1, $2, $3, $4};",
            "r,r,r,r,r,~{memory}", (Uint32(addr), Uint32(r0), Uint32(r1), Uint32(r2),
                                     Uint32(r3)))


@asm()
def ld_shared_v2f(addr):
    return ("ld.shared.v2.f32 {$0, $1}, [$2];", "=f,=f,r,~{memory}", (Uint32(addr),),
            (Float32, Float32))


@asm()
def st_global_u32(ptr, v):
    return "st.global.u32 [$0], $1;", "l,r,~{memory}", (Int64(ptr), Uint32(v))


@asm()
def sstamp_clock(addr):
    return ("{ .reg .b32 c; mov.u32 c, %clock; st.shared.u32 [$0], c; }", "r,~{memory}",
            (Uint32(addr),))


@asm()
def named_bar_arrive(bid, n):
    return "bar.arrive $0, $1;", "r,r,~{memory}", (Uint32(bid), Uint32(n))


@asm((Float32, Float32), pure=True)
def ffma2(a0, a1, b0, b1, c0, c1):
    """(a0 * b0 + c0, a1 * b1 + c1): one packed FFMA2 (same rounding as two FFMAs)."""
    code = ("{ .reg .b64 a, b, c, d; mov.b64 a, {$2, $3}; mov.b64 b, {$4, $5}; "
            "mov.b64 c, {$6, $7}; fma.rn.f32x2 d, a, b, c; mov.b64 {$0, $1}, d; }")
    return code, "=f,=f,f,f,f,f,f,f", tuple(Float32(v) for v in (a0, a1, b0, b1, c0, c1))


@asm((Float32, Float32), pure=True)
def fadd2(a0, a1, b0, b1):
    """(a0 + b0, a1 + b1): one packed FADD2."""
    code = ("{ .reg .b64 a, b, d; mov.b64 a, {$2, $3}; mov.b64 b, {$4, $5}; "
            "add.rn.f32x2 d, a, b; mov.b64 {$0, $1}, d; }")
    return code, "=f,=f,f,f,f,f", tuple(Float32(v) for v in (a0, a1, b0, b1))


def mbar_wait_hint(bar, parity):
    """Bounded parity wait with a 10 ms suspend-time hint (k31_spin_guard)."""
    return k31sg.mbar_wait(bar, parity, hint=", 10000000")


k31sg.register_poll_helper(__file__, "mbar_wait_hint")


@cute.jit
def _subs(n):
    """64-key subtiles of a row (at least one: empty rows still write zeros)."""
    t = (n + 63) // 64
    if t < 1:
        t = Int32(1)
    return t


class SparseMlaPrefillH5:
    def __init__(self, grid: int, stamps: bool = False, knobs: tuple = ()):
        self.grid, self.stamps = grid, stamps
        # diagnostics (timing only): no_store (epilogue skips the global stores),
        # no_gather (KV stages are signalled without any gather: stale data);
        # epi_tma: 1 BF16 staging (scalar stores) + one TMA store per O^T block; 2
        # 16x256b TMEM loads + stmatrix.trans into SW128 staging + TMA stores
        # hint: mbarrier waits carry a suspend-time hint (the waiting thread sleeps
        # until the phase completes instead of re-polling shared memory)
        # no_math: softmax warps skip all math (publish P unchanged); no_mma: the MMA
        # warp issues no MMAs (commits only)
        # split: QK and PV are issued by two warps on different sub-partitions (QK warp
        # 15 next to the padding softmax warps, PV warp 12), 9 gather warps
        # pp: ping-pong softmax (WG u % 2 owns subtile u, 64-key rows, two S passes)
        # sstamp: CTA 0 records clock stamps of softmax warps 0 and 4 for 16 subtiles
        # into the last 1 KiB of shared memory (no global stores in the loop) and copies
        # them to the stamp buffer at the end (int32 [16 subtiles][2 warps][8 events])
        # pp2: ping-pong softmax with the 64-key row in registers (one S load, S released
        # at once; softmax WGs setmaxnreg 136, loader/MMA WGs 40; no -maxrregcount)
        knobs = tuple(knobs) + (0,) * (10 - len(knobs))
        (self.no_store, self.no_gather, self.epi_tma, self.hint, self.no_math,
         self.no_mma, self.split, self.pp, self.sstamp, self.pp2) = knobs[:10]
        if self.pp2:
            self.pp = 1
        self.smem = SMEM_BYTES + (1024 if self.sstamp else 0)
        self.nld = LOADERS - 1 if self.split else LOADERS

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
        if cutlass.const_expr(self.epi_tma == 2):
            om = tensormap.encode_map(out, bf16, (D_NOPE, rows * H), (D_NOPE * 2,), (64, H),
                                      sw.CU_TENSOR_MAP_SWIZZLE_128B, l2)
        else:
            om = tensormap.encode_map(out, bf16, (D_NOPE, rows * H), (D_NOPE * 2,),
                                      (128, H), sw.CU_TENSOR_MAP_SWIZZLE_NONE, l2)
        self.kernel(qn, qr, kvm, kvr, om, out, rows, unit_rows, slots, slot_stride,
                    lengths, scale_log2, out_scale, stamp).launch(
            grid=(self.grid, 1, 1), block=(THREADS, 1, 1), smem=self.smem,
            stream=stream, min_blocks_per_mp=1)

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
        raw = smem.allocate_array(Uint8, self.smem, byte_alignment=1024)
        base = Uint32(raw.toint())
        if tid == 0:
            for i, cnt in ((B_QTM, 4), (B_QLFREE, 4), (B_OFULL, 1), (B_EPIDONE, 1)):
                mbar_init(_bar(base, i), Uint32(cnt))
            for j in cutlass.range_constexpr(2):
                mbar_init(_bar(base, B_SFULL + j), Uint32(1))
                mbar_init(_bar(base, B_SFREE + j), Uint32(4 if self.pp else 8))
                mbar_init(_bar(base, B_QFULL + j), Uint32(1))
                mbar_init(_bar(base, B_QREMPTY + j), Uint32(1))
                mbar_init(_bar(base, B_INV + j), Uint32(8))
            for j in cutlass.range_constexpr(4):
                mbar_init(_bar(base, B_OFREE + j), Uint32(4))
            for j in cutlass.range_constexpr(NP):
                mbar_init(_bar(base, B_PFULL + j), Uint32(4 if self.pp else 8))
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
        if cutlass.const_expr(self.pp2):
            if warp >= W_MMA:
                cute.arch.warpgroup_reg_dealloc(REG_LD2)
        if warp < W_OUT:
            if cutlass.const_expr(self.pp2):
                cute.arch.warpgroup_reg_alloc(REG_SM2)
            if cutlass.const_expr(self.pp):
                self.softmax_pp(base, tid, warp, lane, taddr, n_mine, lengths, cta,
                                scale_log2, out_scale, stamp)
            else:
                self.softmax(base, tid, warp, lane, taddr, n_mine, lengths, cta,
                             scale_log2, out_scale, stamp)
        elif warp < W_MMA:
            self.output(base, tid, warp, lane, taddr, n_mine, lengths, cta, out,
                        tensormap.map_address(om), stamp)
        elif cutlass.const_expr(self.split):
            if warp == W_MMA:
                self.pv_loop(base, lane, taddr, n_mine, lengths, cta, stamp)
            elif warp == W_MMA + 3:
                self.qk_loop(base, lane, taddr, n_mine, lengths, cta, stamp)
            else:
                # Q warp 13 (pw = nld), gather warps 14, 16-23 (pw 0-8)
                pw = warp - (W_LD + 1)
                if warp == W_LD:
                    pw = Int32(self.nld)
                elif warp > W_MMA + 3:
                    pw = warp - (W_LD + 2)
                self.loader(base, pw, lane, cta, n_mine,
                            tensormap.map_address(qn), tensormap.map_address(qr),
                            tensormap.map_address(kvm), tensormap.map_address(kvr),
                            unit_rows, slots, slot_stride, lengths, stamp)
        else:
            if warp == W_MMA:
                self.mma(base, lane, taddr, n_mine, lengths, cta, stamp)
            else:
                self.loader(base, warp - W_LD, lane, cta, n_mine,
                            tensormap.map_address(qn), tensormap.map_address(qr),
                            tensormap.map_address(kvm), tensormap.map_address(kvr),
                            unit_rows, slots, slot_stride, lengths, stamp)
        if cutlass.const_expr(self.pp2):
            # every warpgroup back at the launch count before the common tail, so ptxas
            # sees one register count where the role paths merge
            if warp < W_OUT:
                cute.arch.warpgroup_reg_dealloc(REG_LAUNCH)
            elif warp >= W_MMA:
                cute.arch.warpgroup_reg_alloc(REG_LAUNCH)
        tc_fence_before()
        cute.arch.sync_threads()
        if cutlass.const_expr(self.sstamp):
            if cta == 0 and tid < 256 and stamp != Int64(0):
                v = ld_shared_u32(base + Uint32(SMEM_BYTES + 4 * tid))
                st_global_u32(stamp + Int64(4 * tid), v)
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
    def ss(self, base, cta, warp, lane, u, ev: cutlass.Constexpr):
        """sstamp: shared-memory clock stamp (CTA 0, subtiles 16-23, lane 0) into
        [8 subtiles][32 slots]: slots 0-7 softmax warp 0 events, 8-15 softmax warp 4
        (ev + 8 passed by the caller as warp 4), 16+ the other roles' events."""
        if cutlass.const_expr(self.sstamp):
            if cta == 0 and lane == 0 and u >= 16 and u < 24:
                slot = Int32(ev)
                if cutlass.const_expr(ev < 8):
                    slot = Int32(ev) + Int32(8) * (warp // 4)
                    if warp % 4 == 0:
                        sstamp_clock(base + Uint32(SMEM_BYTES) + Uint32(
                            ((u - 16) * 32 + slot) * 4))
                else:
                    sstamp_clock(base + Uint32(SMEM_BYTES) + Uint32(((u - 16) * 32 + slot) * 4))

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
        nld = self.nld
        quads = (16 - pw + nld - 1) // nld
        lptr = cute.make_ptr(Int32, lengths, cute.AddressSpace.gmem, assumed_align=4)
        lens = cute.make_tensor(lptr, cute.make_layout(1 << 30))
        if pw == nld:
            # Q warp: atoms 0-1 staging (single) is free once the output WG copied the
            # previous token's Q into TMEM (and, epi_tma, once the epilogue of token
            # qi - 2 stored it); SS buffer qi % 2 once token qi - 2's last QK is done
            if lane == 0:
                for qi in range(n_mine):
                    row = cta + qi * self.grid
                    if qi >= 1:
                        self.wait(_bar(base, B_QLFREE), Uint32((qi - 1) & 1))
                    if qi >= 2:
                        self.wait(_bar(base, B_QREMPTY + qi % 2), Uint32((qi // 2 - 1) & 1))
                        if cutlass.const_expr(self.epi_tma):
                            self.wait(_bar(base, B_EPIDONE), Uint32((qi - 2) & 1))
                    full = _bar(base, B_QFULL + qi % 2)
                    qs = base + Uint32(OFF_QS) + Uint32(qi % 2) * Uint32(QS_BYTES)
                    mbar_arrive_tx(full, Uint32(4 * H * 128 + H * 64))
                    for j in cutlass.range_constexpr(4):
                        dst = base + Uint32(OFF_QN + j * ATOM)
                        if j >= 2:
                            dst = qs + Uint32((j - 2) * ATOM)
                        tma_tile2d(dst, tm_qn, full, Int32(128 * j), row * H)
                    tma_tile2d(qs + Uint32(2 * ATOM), tm_qr, full, Int32(0), row * H)
        elif pw < nld:
            # lane l gathers quad q = pw + nld l of every 64-key stage; its 4 slot
            # indices are loaded one stage ahead (global loads, latency under the
            # stage-free wait)
            q = pw + nld * lane
            if lane >= quads:
                q = pw  # idle lanes read a valid slot of the stage (unused)
            g = Int32(0)
            for qi in range(n_mine):
                row = cta + qi * self.grid
                n = lens[row]
                subs = _subs(n)
                srow = slots + Int64(row) * Int64(slot_stride) * Int64(4) + Int64(16 * q)
                c0, c1, c2, c3 = ld_global_v4_s32(srow)
                for t in range(subs):
                    s = g % NS
                    st = base + Uint32(OFF_KV) + Uint32(s) * Uint32(STAGE64)
                    full = _bar(base, B_KVFULL + s)
                    key = t * 64 + 4 * q
                    r = (c0, c1, c2, c3)
                    if t + 1 < subs:
                        c0, c1, c2, c3 = ld_global_v4_s32(srow + Int64((t + 1) * 256))
                    if lane == 0:
                        self.wait(_bar(base, B_KVEMPTY + s), Uint32(((g // NS) & 1) ^ 1))
                        if pw == 0:
                            self.ss(base, cta, Int32(0), lane, g, 22)
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
                    if pw == 0:
                        self.ss(base, cta, Int32(0), lane, g, 23)
                    if cutlass.const_expr(self.stamps):
                        if pw == 0 and lane == 0:
                            self.stamp(stamp, cta, g, 7)
                        if lane == 0 and pw < 4:
                            self.wstamp(stamp, cta, g, 9, pw)
                    g = g + 1

    # ------------------------------------------------------------------ MMA
    @cute.jit
    def mma(self, base, lane, taddr, n_mine, lengths, cta, stamp):
        """Global 64-key subtile counter u: wait S buffer u % 2 free (S(u - 2) read),
        QK(u), then PV(u - 2) (its P was published before softmax(u - 1) started)."""
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
                if u >= 2:
                    self.wait(_bar(base, B_SFREE + u % 2), Uint32((u // 2 - 1) & 1))
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
    def qk_loop(self, base, lane, taddr, n_mine, lengths, cta, stamp):
        """split: QK(u) as soon as S buffer u % 2 is free and stage u has landed."""
        lptr = cute.make_ptr(Int32, lengths, cute.AddressSpace.gmem, assumed_align=4)
        lens = cute.make_tensor(lptr, cute.make_layout(1 << 30))
        u = Int32(0)
        for qi in range(n_mine):
            subs = _subs(lens[cta + qi * self.grid])
            self.wait(_bar(base, B_QTM), Uint32(qi & 1))
            tc_fence_after()
            for i in range(subs):
                self.ss(base, cta, Int32(0), lane, u, 25)
                if u >= 2:
                    self.wait(_bar(base, B_SFREE + u % 2), Uint32((u // 2 - 1) & 1))
                    tc_fence_after()
                self.ss(base, cta, Int32(0), lane, u, 24)
                self.qk(base, lane, taddr, u, qi, i == subs - 1, cta, stamp)
                u = u + 1

    @cute.jit
    def pv_loop(self, base, lane, taddr, n_mine, lengths, cta, stamp):
        """split: PV(u) as soon as the output WG released it."""
        lptr = cute.make_ptr(Int32, lengths, cute.AddressSpace.gmem, assumed_align=4)
        lens = cute.make_tensor(lptr, cute.make_layout(1 << 30))
        u = Int32(0)
        for qi in range(n_mine):
            subs = _subs(lens[cta + qi * self.grid])
            for i in range(subs):
                last = Int32(0)
                if i == subs - 1:
                    last = Int32(1)
                self.pv(base, lane, taddr, u, i, last, qi, cta, stamp)
                u = u + 1

    @cute.jit
    def qk(self, base, lane, taddr, u, qi, last, cta, stamp):
        s = u % NS
        st = base + Uint32(OFF_KV) + Uint32(s) * Uint32(STAGE64)
        if lane == 0:
            self.wstamp(stamp, cta, u, 8, 0)
        self.ss(base, cta, Int32(0), lane, u, 16)
        self.wait(_bar(base, B_KVFULL + s), Uint32((u // NS) & 1))
        if lane == 0:
            self.stamp(stamp, cta, u, 1)
        tc_fence_after()
        # descriptors in converged code (warp-uniform values), MMAs from lane 0
        sc = taddr + Uint32(C_S) + Uint32(64) * Uint32(u % 2)
        qs = base + Uint32(OFF_QS) + Uint32(qi % 2) * Uint32(QS_BYTES)
        bd0 = Uint64(SDESC_SW128) | Uint64(st >> Uint32(4))
        ad0 = Uint64(SDESC_SW128) | Uint64(qs >> Uint32(4))
        if lane == 0:
            for j in cutlass.range_constexpr(4):
                bd = bd0 + Uint64(j * 512)
                if cutlass.const_expr(j >= 2):
                    ad = ad0 + Uint64((j - 2) * ATOM // 16)
                for k2 in cutlass.range_constexpr(2):
                    if cutlass.const_expr(not self.no_mma):
                        if cutlass.const_expr(j < 2):
                            # latent dims 0-255: A from TMEM
                            tc_mma_ts(sc, taddr + Uint32(C_Q + 32 * j + 16 * k2),
                                      bd + Uint64(k2 * 4), Uint32(IDESC_QK),
                                      Uint32(1 if (j + k2) > 0 else 0))
                        else:
                            # latent dims 256-511: A from shared memory
                            tc_mma(sc, ad + Uint64(k2 * 4), bd + Uint64(k2 * 4),
                                   Uint32(IDESC_QK), Uint32(1))
            ad = (ad0 + Uint64(2 * ATOM // 16)) ^ Uint64(SDESC_SW128 ^ SDESC_SW64)
            bd = (bd0 + Uint64(4 * 512)) ^ Uint64(SDESC_SW128 ^ SDESC_SW64)
            if cutlass.const_expr(not self.no_mma):
                tc_mma(sc, ad, bd, Uint32(IDESC_QK), Uint32(1))
            self.wstamp(stamp, cta, u, 8, 1)
            self.ss(base, cta, Int32(0), lane, u, 17)
            tc_commit(_bar(base, B_SFULL + u % 2))
            if last:
                tc_commit(_bar(base, B_QREMPTY + qi % 2))
        cute.arch.sync_warp()

    @cute.jit
    def pv(self, base, lane, taddr, u, i, last, qi, cta, stamp):
        """O^T += V^T (64 keys of the stage) P^T(u), after the output WG has seen
        P(u) (and applied its rescale); P^T is fetched once into the B collector."""
        s = u % NS
        st = base + Uint32(OFF_KV) + Uint32(s) * Uint32(STAGE64)
        self.ss(base, cta, Int32(0), lane, u, 26)
        self.wait(_bar(base, B_OREADY + u % NP), Uint32((u // NP) & 1))
        self.ss(base, cta, Int32(0), lane, u, 18)
        if lane == 0:
            self.stamp(stamp, cta, u, 4)
            self.wstamp(stamp, cta, u, 8, 3)
        tc_fence_after()
        pbuf = base + Uint32(OFF_P) + Uint32(u % NP) * Uint32(P_BYTES)
        bd = Uint64(SDESC_SW64) | Uint64(pbuf >> Uint32(4))
        ad0 = Uint64(SDESC_SW128) | Uint64(st >> Uint32(4))
        if lane == 0:
            for b in cutlass.range_constexpr(4):
                if i == 0:
                    self.wait(_bar(base, B_OFREE + b), Uint32((qi & 1) ^ 1))
                    tc_fence_after()
                ad = ad0 + Uint64(b * 512)
                acc = Uint32(1)
                if i == 0:
                    acc = Uint32(0)
                op = "fill" if b == 0 else ("lastuse" if b == 3 else "use")
                if cutlass.const_expr(not self.no_mma):
                    tc_mma_cb(taddr + Uint32(C_O + 96 * b), ad, bd, Uint32(IDESC_PV), acc,
                              op)
            self.wstamp(stamp, cta, u, 8, 2)
            self.ss(base, cta, Int32(0), lane, u, 19)
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
        join the mbarriers. Only a token's last subtile can be partial: it runs a
        separate copy of the body with the key mask."""
        lptr = cute.make_ptr(Int32, lengths, cute.AddressSpace.gmem, assumed_align=4)
        lens = cute.make_tensor(lptr, cute.make_layout(1 << 30))
        wg = warp // 4
        h = tid % 128
        lane_addr = taddr + (Uint32((warp % 4) * 32) << Uint32(16))
        real = h < H
        real_m = h < (0 if self.no_math else H)  # no_math: no thread does softmax math
        xv = cute.make_rmem_tensor((32,), Float32)
        u = Int32(0)
        for qi in range(n_mine):
            row = cta + qi * self.grid
            n = lens[row]
            subs = _subs(n)
            mp = Float32(0.0)  # m - P_PRE; subtile 0 sets it from its exact max
            lsum = Float32(0.0)
            for i in range(subs - 1):
                mp, lsum = self.sm_sub(base, tid, warp, lane, wg, h, lane_addr, real_m, xv,
                                       scale_log2, u, i, n, mp, lsum, cta, stamp, False)
                u = u + 1
            mp, lsum = self.sm_sub(base, tid, warp, lane, wg, h, lane_addr, real_m, xv,
                                   scale_log2, u, subs - 1, n, mp, lsum, cta, stamp, True)
            u = u + 1
            # ---- token end: l = both halves; 1/l for the epilogue
            if real:
                pair = Uint32(1 + warp % 4)
                lsb = base + Uint32(OFF_MX) + Uint32(u % 2) * Uint32(2 * H * 4)
                if wg == 1:
                    st_shared_f32(lsb + Uint32(h * 4), lsum)
                named_bar(pair, 64)
                if wg == 0:
                    lt = lsum + ld_shared_f32(lsb + Uint32(h * 4))
                    iv = Float32(0.0)
                    if lt > Float32(0.0):
                        iv = out_scale / lt
                    st_shared_f32(base + Uint32(OFF_INV + ((qi % 2) * H + h) * 4), iv)
            cute.arch.sync_warp()
            if lane == 0:
                mbar_arrive(_bar(base, B_INV + qi % 2))

    @cute.jit
    def sm_sub(self, base, tid, warp, lane, wg, h, lane_addr, real_m, xv, scale_log2, u, i,
               n, mp, lsum, cta, stamp, masked: cutlass.Constexpr):
        """Softmax of subtile u (subtile i of its token): S -> x = s * scale - mp (packed
        FFMA), half-row max exchange, lazy-max decision, P = 2^x -> E4M3 P buffer u % 3,
        l += sum P (packed adds; FP32 P). Returns the updated (mp, lsum)."""
        self.ss(base, cta, warp, lane, u, 0)
        self.wait(_bar(base, B_SFULL + u % 2), Uint32((u // 2) & 1))
        self.ss(base, cta, warp, lane, u, 1)
        if tid == 0:
            self.stamp(stamp, cta, u, 2)
        tc_fence_after()
        sa = tmem_ld(lane_addr + Uint32(C_S) + Uint32(64) * Uint32(u % 2) + Uint32(32 * wg),
                     32)
        tc_fence_before()
        cute.arch.sync_warp()
        self.ss(base, cta, warp, lane, u, 2)
        if lane == 0:
            mbar_arrive(_bar(base, B_SFREE + u % 2))
        grow = Int32(0)
        alpha = Float32(1.0)
        if real_m:
            nmp = Float32(0.0) - mp
            for k in cutlass.range_constexpr(16):
                xv[2 * k], xv[2 * k + 1] = ffma2(sa[2 * k], sa[2 * k + 1], scale_log2,
                                                 scale_log2, nmp, nmp)
            if cutlass.const_expr(masked):
                key0 = 64 * i + 32 * wg
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
            mxb = base + Uint32(OFF_MX) + Uint32(u % 2) * Uint32(2 * H * 4)
            st_shared_f32(mxb + Uint32((wg * H + h) * 4), xm)
            self.ss(base, cta, warp, lane, u, 3)
            named_bar(Uint32(1 + warp % 4), 64)
            xm = cute.arch.fmax(xm, ld_shared_f32(mxb + Uint32(((1 - wg) * H + h) * 4)))
            self.ss(base, cta, warp, lane, u, 4)
            d = Float32(0.0)
            if i == 0:
                if xm > Float32(-1e30):
                    d = xm - Float32(P_PRE)
            elif xm > Float32(P_PRE + THR):
                grow = Int32(1)
                d = xm - Float32(P_PRE)
                alpha = ex2(Float32(0.0) - d)
                lsum = lsum * alpha
            if d != Float32(0.0):
                mp = mp + d
                nd = Float32(0.0) - d
                for k in cutlass.range_constexpr(16):
                    xv[2 * k], xv[2 * k + 1] = fadd2(xv[2 * k], xv[2 * k + 1], nd, nd)
        if u >= NP:
            self.wait(_bar(base, B_PEMPTY + u % NP), Uint32((u // NP - 1) & 1))
        self.ss(base, cta, warp, lane, u, 5)
        if real_m:
            pbuf = base + Uint32(OFF_P) + Uint32(u % NP) * Uint32(P_BYTES)
            a0, a1 = Float32(0.0), Float32(0.0)
            b0, b1 = Float32(0.0), Float32(0.0)
            for cc in cutlass.range_constexpr(2):
                c = 2 * wg + cc
                w = []
                for q in cutlass.range_constexpr(4):
                    k = 16 * cc + 4 * q
                    p0 = ex2(xv[k])
                    p1 = ex2(xv[k + 1])
                    p2 = ex2(xv[k + 2])
                    p3 = ex2(xv[k + 3])
                    a0, a1 = fadd2(a0, a1, p0, p1)
                    b0, b1 = fadd2(b0, b1, p2, p3)
                    w.append(cvt_e4m3x2(p0, p1) | (cvt_e4m3x2(p2, p3) << Uint32(16)))
                st_shared_v4(pbuf + Uint32(h * 64) + (
                    (Uint32(c) ^ ((Uint32(h) >> Uint32(1)) & Uint32(3))) << Uint32(4)),
                    w[0], w[1], w[2], w[3])
            a0, a1 = fadd2(a0, a1, b0, b1)
            lsum = lsum + (a0 + a1)
            if wg == 0:
                st_shared_f32(base + Uint32(OFF_AL) + Uint32(u % NP) * Uint32(H * 4)
                              + Uint32(h * 4), alpha)
        if cute.arch.vote_any_sync(grow != 0) and lane == 0:
            st_shared_u32(base + Uint32(OFF_FLAG) + Uint32(u % NP) * Uint32(4),
                          Uint32(u + 1))
        self.ss(base, cta, warp, lane, u, 6)
        fence_async_smem()
        cute.arch.sync_warp()
        self.ss(base, cta, warp, lane, u, 7)
        if tid == 0:
            self.stamp(stamp, cta, u, 3)
        if lane == 0:
            mbar_arrive(_bar(base, B_PFULL + u % NP))
        return mp, lsum

    # --------------------------------------------------------- ping-pong softmax
    @cute.jit
    def softmax_pp(self, base, tid, warp, lane, taddr, n_mine, lengths, cta, scale_log2,
                   out_scale, stamp):
        """WG u % 2 owns subtile u: thread h (TMEM lane h) handles the 64 keys of its
        row. Pass 1 reads S for the row max, the lazy-max decision is handed from WG to
        WG (mailbox + named barrier: producer bar.arrive, consumer bar.sync, so the
        producer never waits), pass 2 re-reads S for P. Each WG keeps l relative to the
        last m it used; the two partial l are combined at the token end. The two warps
        of a sub-partition work on different subtiles. Warps 3 and 7 (padding rows) only
        join the mbarriers of their WG's subtiles."""
        lptr = cute.make_ptr(Int32, lengths, cute.AddressSpace.gmem, assumed_align=4)
        lens = cute.make_tensor(lptr, cute.make_layout(1 << 30))
        wg = warp // 4
        sp = warp % 4
        h = tid % 128
        lane_addr = taddr + (Uint32(sp * 32) << Uint32(16))
        real = h < H
        real_m = h < (0 if self.no_math else H)
        xv = cute.make_rmem_tensor((64 if self.pp2 else 32,), Float32)
        total = Int32(0)
        for qi in range(n_mine):
            total = total + _subs(lens[cta + qi * self.grid])
        u = Int32(0)
        for qi in range(n_mine):
            row = cta + qi * self.grid
            n = lens[row]
            subs = _subs(n)
            mp_last = Float32(-1e30)  # m - P_PRE that this WG's l is relative to
            lw = Float32(0.0)
            for i in range(subs):
                if u % 2 == wg:
                    if i == subs - 1:
                        mp_last, lw = self.pp_sub(base, tid, warp, lane, wg, sp, h, lane_addr,
                                                  real_m, xv, scale_log2, u, i, n, total,
                                                  mp_last, lw, cta, stamp, True)
                    else:
                        mp_last, lw = self.pp_sub(base, tid, warp, lane, wg, sp, h, lane_addr,
                                                  real_m, xv, scale_log2, u, i, n, total,
                                                  mp_last, lw, cta, stamp, False)
                u = u + 1
            # ---- token end: l = l0 2^(m0 - m) + l1 2^(m1 - m), m = max(m0, m1)
            if real:
                tok = base + Uint32(OFF_TOK) + Uint32(qi % 2) * Uint32(2 * H * 4)
                if wg == 1:
                    st_shared_f32(tok + Uint32(h * 4), lw)
                    st_shared_f32(tok + Uint32((H + h) * 4), mp_last)
                named_bar(Uint32(7 + sp), 64)
                if wg == 0:
                    l1 = ld_shared_f32(tok + Uint32(h * 4))
                    m1 = ld_shared_f32(tok + Uint32((H + h) * 4))
                    mf = cute.arch.fmax(mp_last, m1)
                    lt = lw * ex2(mp_last - mf) + l1 * ex2(m1 - mf)
                    iv = Float32(0.0)
                    if lt > Float32(0.0):
                        iv = out_scale / lt
                    st_shared_f32(base + Uint32(OFF_INV + ((qi % 2) * H + h) * 4), iv)
            cute.arch.sync_warp()
            if lane == 0:
                mbar_arrive(_bar(base, B_INV + qi % 2))

    @cute.jit
    def pp_sub(self, base, tid, warp, lane, wg, sp, h, lane_addr, real_m, xv, scale_log2, u,
               i, n, total, mp_last, lw, cta, stamp, masked: cutlass.Constexpr):
        """Subtile u (subtile i of its token) on WG u % 2. Returns (m, l) of this WG."""
        self.wait(_bar(base, B_SFULL + u % 2), Uint32((u // 2) & 1))
        if tid % 128 == 0:
            self.stamp(stamp, cta, u, 2)
        tc_fence_after()
        scol = lane_addr + Uint32(C_S) + Uint32(64) * Uint32(u % 2)
        mbx = base + Uint32(OFF_MBOX)
        grow = Int32(0)
        alpha = Float32(1.0)
        mp = Float32(0.0)
        if cutlass.const_expr(self.pp2):
            # the whole 64-key row into registers; S buffer released at once
            sa = tmem_ld(scol, 64)
            for k in cutlass.range_constexpr(64):
                xv[k] = sa[k]
            tc_fence_before()
            cute.arch.sync_warp()
            if lane == 0:
                mbar_arrive(_bar(base, B_SFREE + u % 2))
            if cutlass.const_expr(masked):
                for k in cutlass.range_constexpr(64):
                    if 64 * i + k >= n:
                        xv[k] = Float32(-math.inf)
        if real_m:
            # pass 1: row max of the raw scores (scale > 0 commutes with max)
            sm = Float32(-math.inf)
            for c in cutlass.range_constexpr(1 if self.pp2 else 2):
                if cutlass.const_expr(self.pp2):
                    vals = [xv[k] for k in range(64)]
                else:
                    sa = tmem_ld(scol + Uint32(32 * c), 32)
                    vals = sa
                if cutlass.const_expr(masked and not self.pp2):
                    for k in cutlass.range_constexpr(32):
                        xv[k] = sa[k]
                    if 64 * i + 32 * c + 32 > n:
                        for k in cutlass.range_constexpr(32):
                            if 64 * i + 32 * c + k >= n:
                                xv[k] = Float32(-math.inf)
                    vals = [xv[k] for k in range(32)]
                sm = cute.arch.fmax(sm, _tree_max(vals))
            # lazy-max decision after the previous subtile's (other WG) decision
            if u >= 1:
                named_bar(Uint32(1 + sp + 3 * (1 - wg)), 64)
            mprev = ld_shared_f32(mbx + Uint32(((u + 1) % 2) * H * 4 + h * 4))
            xm = sm * scale_log2
            if i == 0:
                if xm > Float32(-1e30):
                    mp = xm - Float32(P_PRE)
            else:
                mp = mprev
                if xm - mprev > Float32(P_PRE + THR):
                    grow = Int32(1)
                    mp = xm - Float32(P_PRE)
                    alpha = ex2(mprev - mp)
            st_shared_f32(mbx + Uint32((u % 2) * H * 4 + h * 4), mp)
            if u + 1 < total:
                named_bar_arrive(Uint32(1 + sp + 3 * wg), 64)
            # this WG's l is kept relative to the m it last used
            if mp != mp_last:
                lw = lw * ex2(mp_last - mp)
                mp_last = mp
        if u >= NP:
            self.wait(_bar(base, B_PEMPTY + u % NP), Uint32((u // NP - 1) & 1))
        if real_m:
            # pass 2: P = 2^(s scale - m) -> E4M3 row h of P buffer u % 3
            pbuf = base + Uint32(OFF_P) + Uint32(u % NP) * Uint32(P_BYTES)
            a0, a1 = Float32(0.0), Float32(0.0)
            b0, b1 = Float32(0.0), Float32(0.0)
            nmp = Float32(0.0) - mp
            for c in cutlass.range_constexpr(2):
                if cutlass.const_expr(self.pp2):
                    # xv holds the row (masked keys -inf); x = s scale - m in place
                    for k in cutlass.range_constexpr(16):
                        xv[32 * c + 2 * k], xv[32 * c + 2 * k + 1] = ffma2(
                            xv[32 * c + 2 * k], xv[32 * c + 2 * k + 1], scale_log2,
                            scale_log2, nmp, nmp)
                else:
                    sa = tmem_ld(scol + Uint32(32 * c), 32)
                    for k in cutlass.range_constexpr(16):
                        xv[2 * k], xv[2 * k + 1] = ffma2(sa[2 * k], sa[2 * k + 1],
                                                         scale_log2, scale_log2, nmp, nmp)
                if cutlass.const_expr(masked and not self.pp2):
                    if 64 * i + 32 * c + 32 > n:
                        for k in cutlass.range_constexpr(32):
                            if 64 * i + 32 * c + k >= n:
                                xv[k] = Float32(-math.inf)
                for cc in cutlass.range_constexpr(2):
                    w = []
                    for q in cutlass.range_constexpr(4):
                        k = 16 * cc + 4 * q + (32 * c if self.pp2 else 0)
                        p0 = ex2(xv[k])
                        p1 = ex2(xv[k + 1])
                        p2 = ex2(xv[k + 2])
                        p3 = ex2(xv[k + 3])
                        a0, a1 = fadd2(a0, a1, p0, p1)
                        b0, b1 = fadd2(b0, b1, p2, p3)
                        w.append(cvt_e4m3x2(p0, p1) | (cvt_e4m3x2(p2, p3) << Uint32(16)))
                    ch = 2 * c + cc
                    st_shared_v4(pbuf + Uint32(h * 64) + (
                        (Uint32(ch) ^ ((Uint32(h) >> Uint32(1)) & Uint32(3))) << Uint32(4)),
                        w[0], w[1], w[2], w[3])
            a0, a1 = fadd2(a0, a1, b0, b1)
            lw = lw + (a0 + a1)
            st_shared_f32(base + Uint32(OFF_AL) + Uint32(u % NP) * Uint32(H * 4)
                          + Uint32(h * 4), alpha)
        if cutlass.const_expr(not self.pp2):
            tc_fence_before()
            cute.arch.sync_warp()
            if lane == 0:
                mbar_arrive(_bar(base, B_SFREE + u % 2))
        if cute.arch.vote_any_sync(grow != 0) and lane == 0:
            st_shared_u32(base + Uint32(OFF_FLAG) + Uint32(u % NP) * Uint32(4),
                          Uint32(u + 1))
        fence_async_smem()
        cute.arch.sync_warp()
        if tid % 128 == 0:
            self.stamp(stamp, cta, u, 3)
        if lane == 0:
            mbar_arrive(_bar(base, B_PFULL + u % NP))
        return mp_last, lw

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
                if wq == 0:
                    self.ss(base, cta, Int32(0), lane, u, 20)
                if wq == 0 and lane == 0:
                    self.wstamp(stamp, cta, u, 10, 0)
                if ld_shared_u32(base + Uint32(OFF_FLAG) + Uint32(u % NP) * Uint32(4)) == Uint32(u + 1):
                    # O^T(u - 1) complete: PV(u - 1) released P buffer (u - 1) % 3
                    self.wait(_bar(base, B_PEMPTY + (u - 1) % NP), Uint32(((u - 1) // NP) & 1))
                    tc_fence_after()
                    alb = base + Uint32(OFF_AL) + Uint32(u % NP) * Uint32(H * 4)
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
                if wq == 0:
                    self.ss(base, cta, Int32(0), lane, u, 21)
                u = u + 1
            if cutlass.const_expr(self.epi_tma == 2):
                self.epilogue_stm(base, e, wq, lane, lane_addr, qi, cta, tm_o, stamp)
            elif cutlass.const_expr(self.epi_tma == 1):
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
            for j in cutlass.range_constexpr(2):
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
        staging tile (the Q latent atoms 0-1 staging, free between token qi + 1's Q
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
        named_bar(Uint32(10), 128)
        buf = base + Uint32(OFF_QN)
        for b in cutlass.range_constexpr(4):
            if b >= 1:
                if e == 0:
                    bulk_wait_read0()
                named_bar(Uint32(10), 128)
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
            named_bar(Uint32(10), 128)
            if e == 0:
                if cutlass.const_expr(not self.no_store):
                    tma_store2d(tm_o, buf, Int32(128 * b), row * H)
        if e == 0:
            bulk_wait_read0()
            mbar_arrive(_bar(base, B_EPIDONE))
            self.stamp(stamp, cta, Int32(qi), 6)


    @cute.jit
    def epilogue_stm(self, base, e, wq, lane, lane_addr, qi, cta, tm_o, stamp):
        """O^T block b (lanes = dims, columns = heads) -> BF16 [heads x dims]: 16x256b
        TMEM loads give each thread two adjacent heads of one dim (an m8n8 b16
        fragment with dims as rows), scaled by s / l of the two heads and packed;
        stmatrix.trans writes them as head rows into two SW128 [96 x 64 dims] staging
        halves (the Q latent atoms 0-1 staging), each stored by one TMA store."""
        row = cta + qi * self.grid
        self.wait(_bar(base, B_OFULL), Uint32(qi & 1))
        self.wait(_bar(base, B_INV + qi % 2), Uint32((qi // 2) & 1))
        inv_base = base + Uint32(OFF_INV) + Uint32(qi % 2) * Uint32(H * 4)
        tc_fence_after()
        if e == 0:
            self.stamp(stamp, cta, Int32(qi), 5)
        named_bar(Uint32(10), 128)  # every thread's Q copy reads of the staging are done
        t4 = lane % 4
        j = lane % 8
        mk = lane // 8  # matrix this lane addresses in stmatrix.x4
        for b in cutlass.range_constexpr(4):
            if cutlass.const_expr(b >= 1):
                if e == 0:
                    bulk_wait_read0()
                named_bar(Uint32(10), 128)
            for hh in cutlass.range_constexpr(2):
                d0 = 32 * wq + 16 * hh  # first dim (lane) of this 16-lane half
                for c in cutlass.range_constexpr(3):
                    v = tmem_ld16x256(lane_addr + Uint32(16 * hh << 16)
                                      + Uint32(C_O + 96 * b + 32 * c))
                    if cutlass.const_expr(hh == 1 and c == 2):
                        tc_fence_before()
                        cute.arch.sync_warp()
                        if lane == 0:
                            mbar_arrive(_bar(base, B_OFREE + b))
                    frag = []
                    for g in cutlass.range_constexpr(4):
                        i0, i1 = ld_shared_v2f(inv_base + Uint32((32 * c + 8 * g + 2 * t4) * 4))
                        frag.append(bf16x2(v[4 * g] * i0, v[4 * g + 1] * i1))
                        frag.append(bf16x2(v[4 * g + 2] * i0, v[4 * g + 3] * i1))
                    # stmatrix x4 #s: matrices (g = 2 s + k // 2, dims d0 + 8 (k % 2))
                    for sx in cutlass.range_constexpr(2):
                        g = 2 * sx + mk // 2
                        dd = d0 + 8 * (mk % 2)  # first dim of the matrix (0-127)
                        hrow = 32 * c + 8 * g + j  # stored row = head
                        half = base + Uint32(OFF_QN) + Uint32(dd // 64) * Uint32(H * 128)
                        addr = half + Uint32(hrow * 128) + (
                            ((Uint32((dd % 64) // 8)) ^ (Uint32(hrow) & Uint32(7))) << Uint32(4))
                        stmatrix_t4(addr, frag[4 * sx], frag[4 * sx + 1], frag[4 * sx + 2],
                                    frag[4 * sx + 3])
            fence_async_smem()
            named_bar(Uint32(10), 128)
            if e == 0:
                if cutlass.const_expr(not self.no_store):
                    for hf in cutlass.range_constexpr(2):
                        tma_store2d(tm_o, base + Uint32(OFF_QN + hf * H * 128),
                                    Int32(128 * b + 64 * hf), row * H)
        if e == 0:
            bulk_wait_read0()
            mbar_arrive(_bar(base, B_EPIDONE))
            self.stamp(stamp, cta, Int32(qi), 6)

@cache
def _compiled(grid: int, stamps: bool, knobs: tuple = ()):
    kernel = SparseMlaPrefillH5(grid, stamps, knobs)
    args = (Int64(0), Int32(1), Int64(0), Int32(64), Int64(0), Int32(1), Int64(0), Int64(0),
            Float32(1.0), Float32(1.0), Int32(64), Int64(0), cuda.CUstream(0))
    if kernel.pp2:
        # per-warpgroup budgets come from setmaxnreg; .reqntid caps the launch at 80
        return cute.compile(kernel, *args)
    return cute.compile(kernel, *args, options=f"--ptxas-options -maxrregcount={REG_LAUNCH}")


def sparse_mla_prefill_h5(q8: torch.Tensor, kv_rows: torch.Tensor, slots: torch.Tensor,
                          lengths: torch.Tensor, scale: float, out_scale: float,
                          unit_rows: int = 64, out: torch.Tensor | None = None,
                          stamps: torch.Tensor | None = None,
                          knobs: tuple = ()) -> torch.Tensor:
    """Same contract as ``sparse_mla_sol.sparse_mla_prefill_sol``."""
    t = q8.shape[0]
    if q8.shape[1:] != (H, ROW) or not q8.is_contiguous():
        raise ValueError(f"sparse_mla_prefill_h5: q8 must be contiguous [T, {H}, {ROW}]")
    if out is None:
        out = torch.empty((t, H, D_NOPE), dtype=torch.bfloat16, device=q8.device)
    if not out.is_contiguous() or out.shape != (t, H, D_NOPE):
        raise ValueError("sparse_mla_prefill_h5: out must be contiguous [T, 96, 512]")
    if t == 0:
        return out
    sms = torch.cuda.get_device_properties(q8.device).multi_processor_count
    grid = min(sms, t)
    sst = len(knobs) > 8 and bool(knobs[8])  # shared-memory softmax stamps only
    fn = _compiled(grid, stamps is not None and not sst, tuple(knobs))
    stream = cuda.CUstream(torch.cuda.current_stream(q8.device).cuda_stream)
    fn(Int64(q8.data_ptr()), Int32(t), Int64(kv_rows.data_ptr()), Int32(unit_rows),
       Int64(slots.data_ptr()), Int32(slots.stride(0)), Int64(lengths.data_ptr()),
       Int64(out.data_ptr()), Float32(scale * LOG2E), Float32(out_scale),
       Int32(kv_rows.shape[0]), Int64(0 if stamps is None else stamps.data_ptr()), stream)
    return out
