# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 DSA prefill sparse MLA on Rubin (sm_107): ping-pong softmax kernel.

One CTA per SM and per token (every top-k KV row is delivered into one SM once),
QK heads on M (Q latent dims 0-255 from TMEM, dims 256-511 and RoPE as SS operands),
PV keys on M with P^T in the Rubin B collector. TMEM: Q 64 + S 2 x 64 + O^T 384 = 576.

Pipeline (P-MLA runs 45-61):
* QK and PV are issued by two warps on different sub-partitions (one MMA warp was the
  serializer: ~10 dependent SASS per tcgen05.mma issue plus all waits and commits).
* Softmax ping-pong: warpgroup u % 2 owns 64-key subtile u. A thread (= head, TMEM
  lane) loads its whole 64-key row with one tcgen05.ld (S released at once), takes
  the row max, receives the lazy-max m of subtile u - 1 from the other warpgroup
  (mailbox + producer bar.arrive / consumer bar.sync), and writes E4M3 P. The two
  warps of a sub-partition therefore work on different subtiles. A 64-value row needs
  more than 80 registers: the CTA has 20 warps (ptxas only honours setmaxnreg region
  budgets above the launch count at <= 640 threads), softmax warpgroups run at 144.
* Each softmax warpgroup keeps l relative to the last m it used; both are combined
  at the token end. The lazy-max O^T rescale (rare) and, for the warpgroup that owns
  a token's last subtile, the next token's Q copy into TMEM and the epilogue
  (16x256b TMEM loads -> stmatrix.trans -> SW128 staging -> TMA store) run in the
  softmax warpgroups: there is no separate output warpgroup.
* 9 gather warps (TMA tile::gather4), slot indices by prefetched global loads.
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
    mbar_arrive,
    mbar_arrive_tx,
    mbar_wait,
    mbar_init,
    named_bar,
    st_shared_f32,
    st_shared_u32,
    st_shared_v4,
    tc_fence_after,
    tc_fence_before,
    tma_tile2d,
    tmem_alloc_exclusive,
    tmem_free_exclusive,
    tmem_ld,
)
from vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla_h2 import (
    ld_shared_v4u,
    tmem_st_u32,
)
from vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla_h5 import (
    _tree_max,
    bf16x2,
    fadd2,
    ffma2,
    ld_shared_v2f,
    named_bar_arrive,
    st_global_u32,
    sstamp_clock,
    stmatrix_t4,
    tmem_ld16x256,
)
from vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla_sol import (
    bulk_wait_read0,
    tma_store2d,
)

P_PRE, THR = 4, 4.5  # P = 2^(s - m + P_PRE) <= 2^(P_PRE + THR) = 362 < 448 (E4M3 max)
NS = 6  # 64-key KV stages
NP = 3  # P buffers
NLD = 9  # gather warps
STAGE64 = 4 * 64 * 128 + 64 * 64  # 36864: 4 latent SW128 atoms + RoPE SW64
# warps: 0-3 / 4-7 softmax warpgroups (thread = head; warps 3 and 7 hold padding rows
# 96-127 and only take part in barriers, rescales, Q copies and epilogues), 8 PV,
# 9 Q loader, 10 and 12-19 gather, 11 QK
W_SM, W_PV, W_Q, W_QK = 8, 8, 9, 11
THREADS = 32 * 20
REG_LAUNCH = 65536 // THREADS // 8 * 8  # 96
REG_SM, REG_LOW = 144, 56
assert 256 * REG_SM + 384 * REG_LOW <= THREADS * REG_LAUNCH

# shared memory (bytes). SS A tiles have M = 128 rows; only 96 are loaded and rows
# 96-127 read whatever follows (valid shared memory, padding rows of S only).
ATOM = H * 128  # one Q latent SW128 atom (96 rows x 128 B)
QS_BYTES = 2 * ATOM + 128 * 64  # per token: latent atoms 2-3 + RoPE [128 x 64 B] SW64
P_BYTES = H * 64
OFF_QN = 0  # Q latent atoms 0-1 staging (copied into TMEM per token; epilogue staging)
OFF_QS = OFF_QN + 2 * ATOM  # 2 x QS_BYTES (token parity): A of the SS QK MMAs
OFF_P = OFF_QS + 2 * QS_BYTES  # 3 x P [96 heads x 64 keys] E4M3, K-major SW64
OFF_AL = OFF_P + NP * P_BYTES  # [2 WG][2 (u // 2 parity)][96] alpha (lazy-max rescale)
OFF_FLAG = OFF_AL + 4 * H * 4  # [3] u32 (unused slots kept for alignment)
OFF_INV = OFF_FLAG + 16  # [96] out_scale / l of the token in its epilogue
OFF_TOK0 = OFF_INV + H * 4  # token parity 0: [2 WG][2 (l, m)][96]
OFF_BAR = OFF_TOK0 + 4 * H * 4
OFF_TMEM = OFF_BAR + 40 * 8
OFF_GMASK = OFF_TMEM + 16  # [2 WG][2 (u // 2 parity)][4] u32: heads that grew (ballots)
OFF_KV = (OFF_GMASK + 64 + 1023) // 1024 * 1024
SMEM_BYTES = OFF_KV + NS * STAGE64
# RoPE A tile rows 96-127 of each QS buffer (read by the MMA, never loaded): the
# lazy-max mailbox ([2][96] f32, buffer 0) and token parity 1 of the l/m exchange
OFF_MBOX = OFF_QS + 2 * ATOM + H * 64
OFF_TOK1 = OFF_QS + QS_BYTES + 2 * ATOM + H * 64
assert 2 * H * 4 <= 32 * 64 and 4 * H * 4 <= 32 * 64
assert OFF_QS % 1024 == 0 and QS_BYTES % 1024 == 0 and ATOM % 1024 == 0
assert OFF_KV % 1024 == 0 and SMEM_BYTES <= 334848 - 1024, SMEM_BYTES

# barriers (count 1 = one arrive / commit per phase, 4 = one arrive per warp of a WG)
B_SFULL, B_SFREE = 0, 2  # x2 (S buffer u % 2)
B_QTM, B_QLFREE, B_OFULL, B_EPIDONE = 4, 5, 6, 7
B_QFULL, B_QREMPTY = 8, 10  # x2 (token parity)
B_OFREE = 12  # x4 (O^T block)
B_OREADY, B_PEMPTY = 16, 19  # x3 (P buffer u % 3)
B_KVFULL, B_KVEMPTY = 22, 28  # x NS
assert B_KVEMPTY + NS <= 40
# named barriers: 1-3 / 4-6 m handoff WG0 -> WG1 / WG1 -> WG0 (per sub-partition),
# 7-9 token-end l/m exchange (per sub-partition), 10 + wg warpgroup-wide
C_Q, C_S, C_O = 0, 64, 192  # Q latent dims 0-255; S buffers at 64 and 128; O^T

# instruction descriptors (dense kind::f8f6f4, E4M3 x E4M3 -> FP32, Rubin K64 bit 29)
IDESC_QK = (1 << 4) | (64 >> 3 << 17) | (128 >> 4 << 24) | (1 << 29)  # M128 N64
IDESC_PV = (1 << 4) | (H >> 3 << 17) | (128 >> 4 << 24) | (1 << 29) | (1 << 15)  # A MN


def _bar(base, i):
    return base + Uint32(OFF_BAR + 8 * i)


@asm()
def clock_read():
    return "{ .reg .b32 c; mov.u32 c, %clock; }", "~{memory}", ()


@asm()
def compiler_fence():
    return "", "~{memory}", ()


@asm(Uint32, pure=False)
def bar_red_or(bid, n, pred):
    """bar.red.or over the threads of a named barrier: 1 if any thread passed pred != 0."""
    code = ("{ .reg .pred p, q; setp.ne.u32 p, $3, 0; bar.red.or.pred q, $1, $2, p; "
            "selp.u32 $0, 1, 0, q; }")
    return code, "=r,r,r,r,~{memory}", (Uint32(bid), Uint32(n), Uint32(pred))


# MMA issue by the whole warp with one elected thread: the operands stay warp-uniform
# (uniform registers) instead of an ELECT + R2UR.BROADCAST waterfall per MMA under a
# lane-0 branch (~100 clk per MMA issue, P-MLA run83 stamps)
_ELECT = "elect.sync _|e, 0xffffffff; "


@asm()
def mma_ts_e(d, a_tmem, b, idesc, acc):
    code = ("{ .reg .pred p, e; " + _ELECT + "setp.ne.b32 p, $4, 0; "
            "@e tcgen05.mma.cta_group::1.kind::f8f6f4 [$0], [$1], $2, $3, p; }")
    return code, "r,r,l,r,r,~{memory}", (Uint32(d), Uint32(a_tmem), Uint64(b),
                                          Uint32(idesc), Uint32(acc))


@asm()
def mma_ss_e(d, a, b, idesc, acc):
    code = ("{ .reg .pred p, e; " + _ELECT + "setp.ne.b32 p, $4, 0; "
            "@e tcgen05.mma.cta_group::1.kind::f8f6f4 [$0], $1, $2, $3, p; }")
    return code, "r,l,l,r,r,~{memory}", (Uint32(d), Uint64(a), Uint64(b), Uint32(idesc),
                                          Uint32(acc))


@asm()
def mma_cb_e(d, a, b, idesc, acc, op: str):
    code = ("{ .reg .pred p, e; " + _ELECT + "setp.ne.b32 p, $4, 0; "
            f"@e tcgen05.mma.cta_group::1.kind::f8f6f4.collector::b::{op} [$0], $1, $2, $3, p; }}")
    return code, "r,l,l,r,r,~{memory}", (Uint32(d), Uint64(a), Uint64(b), Uint32(idesc),
                                          Uint32(acc))


@asm()
def commit_e(bar):
    code = ("{ .reg .pred e; " + _ELECT + "@e tcgen05.commit.cta_group::1.mbarrier::arrive::one"
            ".shared::cluster.b64 [$0]; }")
    return code, "r,~{memory}", (Uint32(bar),)


@asm(Uint32)
def ballot(pred):
    code = "{ .reg .pred p; setp.ne.u32 p, $1, 0; vote.sync.ballot.b32 $0, p, 0xffffffff; }"
    return code, "=r,r", (Uint32(pred),)


@asm(Uint32)
def bfind(v):
    return "bfind.u32 $0, $1;", "=r,r", (Uint32(v),)


@asm()
def tmem_ld_o4(addr):
    """Column h of the four O^T blocks (columns addr + 96 b)."""
    code = ("tcgen05.ld.sync.aligned.32x32b.x1.b32 {$0}, [$4]; "
            "tcgen05.ld.sync.aligned.32x32b.x1.b32 {$1}, [$5]; "
            "tcgen05.ld.sync.aligned.32x32b.x1.b32 {$2}, [$6]; "
            "tcgen05.ld.sync.aligned.32x32b.x1.b32 {$3}, [$7]; "
            "tcgen05.wait::ld.sync.aligned;")
    a = Uint32(addr)
    return (code, "=f,=f,=f,=f,r,r,r,r,~{memory}",
            (a, a + Uint32(96), a + Uint32(192), a + Uint32(288)), (Float32,) * 4)


@asm()
def tmem_st_o4(addr, v0, v1, v2, v3):
    code = ("tcgen05.st.sync.aligned.32x32b.x1.b32 [$0], {$4}; "
            "tcgen05.st.sync.aligned.32x32b.x1.b32 [$1], {$5}; "
            "tcgen05.st.sync.aligned.32x32b.x1.b32 [$2], {$6}; "
            "tcgen05.st.sync.aligned.32x32b.x1.b32 [$3], {$7}; "
            "tcgen05.wait::st.sync.aligned;")
    a = Uint32(addr)
    return code, "r,r,r,r,f,f,f,f,~{memory}", (a, a + Uint32(96), a + Uint32(192),
                                               a + Uint32(288), Float32(v0), Float32(v1),
                                               Float32(v2), Float32(v3))


@asm()
def tmem_ld_max64(addr):
    """64 columns of this thread's TMEM lane and their maximum (tcgen05.ld.red)."""
    regs = ",".join(f"${i}" for i in range(64))
    code = (f"tcgen05.ld.red.sync.aligned.32x32b.x64.max.f32 {{{regs}}}, $64, [$65]; "
            "tcgen05.wait::ld.sync.aligned;")
    return (code, ",".join(["=f"] * 65 + ["r", "~{memory}"]), (Uint32(addr),),
            (Float32,) * 65)


@cute.jit
def _subs(n):
    """64-key subtiles of a row (at least one: empty rows still write zeros)."""
    t = (n + 63) // 64
    if t < 1:
        t = Int32(1)
    return t


class SparseMlaPrefillH6:
    def __init__(self, grid: int, knobs: tuple = ()):
        self.grid = grid
        # diagnostics (timing only): no_store (no TMA stores), no_gather (stages are
        # signalled without gathers: stale data), no_math (softmax warps skip all math),
        # no_mma (MMA warps only commit), sstamp (CTA 0 shared-memory clock stamps)
        # pad: 1 KiB of extra dynamic shared memory (no stamps)
        # ssmask: roles whose sstamp sites are compiled (1 softmax, 2 QK, 4 PV, 8 gather)
        # pin: loader code-layout experiment (1: clock read only at the two gather stamp
        # sites, 2: the guarded stamp store to a scratch word)
        knobs = tuple(knobs) + (0,) * (8 - len(knobs))
        (self.no_store, self.no_gather, self.no_math, self.no_mma, self.sstamp,
         self.pad, self.ssmask, self.pin) = knobs[:8]
        if self.sstamp and not self.ssmask:
            self.ssmask = 15
        self.smem = SMEM_BYTES + (1024 if self.sstamp or self.pad else 0)
        self.sm_arrivals = 4  # S-free / O-ready arrivals per subtile (one per softmax warp)

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
        om = tensormap.encode_map(out, bf16, (D_NOPE, rows * H), (D_NOPE * 2,), (64, H),
                                  sw.CU_TENSOR_MAP_SWIZZLE_128B, l2)
        self.kernel(qn, qr, kvm, kvr, om, rows, unit_rows, slots, slot_stride, lengths,
                    scale_log2, out_scale, stamp).launch(
            grid=(self.grid, 1, 1), block=(THREADS, 1, 1), smem=self.smem,
            stream=stream, min_blocks_per_mp=1)

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
        raw = smem.allocate_array(Uint8, self.smem, byte_alignment=1024)
        base = Uint32(raw.toint())
        if tid == 0:
            for i, cnt in ((B_QTM, 4), (B_QLFREE, 4), (B_OFULL, 1), (B_EPIDONE, 1)):
                mbar_init(_bar(base, i), Uint32(cnt))
            for j in cutlass.range_constexpr(2):
                mbar_init(_bar(base, B_SFULL + j), Uint32(1))
                mbar_init(_bar(base, B_SFREE + j), Uint32(self.sm_arrivals))
                mbar_init(_bar(base, B_QFULL + j), Uint32(1))
                mbar_init(_bar(base, B_QREMPTY + j), Uint32(1))
            for j in cutlass.range_constexpr(4):
                mbar_init(_bar(base, B_OFREE + j), Uint32(4))
            for j in cutlass.range_constexpr(NP):
                mbar_init(_bar(base, B_OREADY + j), Uint32(self.sm_arrivals))
                mbar_init(_bar(base, B_PEMPTY + j), Uint32(1))
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
        if warp < W_SM:
            cute.arch.warpgroup_reg_alloc(REG_SM)
            self.softmax(base, tid, warp, lane, taddr, n_mine, lengths, cta, scale_log2,
                         out_scale, tensormap.map_address(om))
            cute.arch.warpgroup_reg_dealloc(REG_LAUNCH)
        else:
            cute.arch.warpgroup_reg_dealloc(REG_LOW)
            taddr = Uint32(cute.arch.make_warp_uniform(taddr))
            if warp == W_PV:
                self.pv_loop(base, lane, taddr, n_mine, lengths, cta)
            elif warp == W_QK:
                self.qk_loop(base, lane, taddr, n_mine, lengths, cta)
            else:
                pw = warp - (W_Q + 1)  # gather 10 -> 0, 12-19 -> 1-8; Q warp 9 -> NLD
                if warp == W_Q:
                    pw = Int32(NLD)
                elif warp > W_QK:
                    pw = warp - (W_Q + 2)
                self.loader(base, pw, lane, cta, n_mine, tensormap.map_address(qn),
                            tensormap.map_address(qr), tensormap.map_address(kvm),
                            tensormap.map_address(kvr), unit_rows, slots, slot_stride,
                            lengths)
            cute.arch.warpgroup_reg_alloc(REG_LAUNCH)
        tc_fence_before()
        cute.arch.sync_threads()
        if cutlass.const_expr(self.sstamp):
            if cta == 0 and tid < 256 and stamp != Int64(0):
                st_global_u32(stamp + Int64(4 * tid),
                              ld_shared_u32(base + Uint32(SMEM_BYTES + 4 * tid)))
        if warp == 0:
            tc_fence_after()
            tmem_free_exclusive(taddr)

    @cute.jit
    def ss(self, base, cta, lane, u, slot, role: cutlass.Constexpr = 15):
        """sstamp: CTA 0, subtiles 16-23, [8][32] u32 clock slots (lane 0 of the caller)."""
        if cutlass.const_expr(self.sstamp and (self.ssmask & role)):
            if cta == 0 and lane == 0 and u >= 16 and u < 24:
                sstamp_clock(base + Uint32(SMEM_BYTES)
                             + Uint32(((u - 16) * 32 + Int32(slot)) * 4))

    @cute.jit
    def lpin(self, base, cta, lane, g):
        if cutlass.const_expr(self.pin == 1):
            clock_read()
        if cutlass.const_expr(self.pin == 2):
            if cta == 0 and lane == 0 and g >= 16 and g < 24:
                sstamp_clock(base + Uint32(OFF_FLAG + 12))

    # ------------------------------------------------------------------ gather
    @cute.jit
    def loader(self, base, pw, lane, cta, n_mine, tm_qn, tm_qr, tm_kv, tm_kr, unit_rows,
               slots, slot_stride, lengths):
        quads = (16 - pw + NLD - 1) // NLD
        lptr = cute.make_ptr(Int32, lengths, cute.AddressSpace.gmem, assumed_align=4)
        lens = cute.make_tensor(lptr, cute.make_layout(1 << 30))
        if pw == NLD:
            # Q warp: atoms 0-1 staging (single) is free once token qi - 1's Q is in TMEM
            # and token qi - 2's epilogue has stored its staging; SS buffer qi % 2 once
            # token qi - 2's last QK is done
            if lane == 0:
                for qi in range(n_mine):
                    row = cta + qi * self.grid
                    if qi >= 1:
                        mbar_wait(_bar(base, B_QLFREE), Uint32((qi - 1) & 1))
                    if qi >= 2:
                        mbar_wait(_bar(base, B_QREMPTY + qi % 2), Uint32((qi // 2 - 1) & 1))
                        mbar_wait(_bar(base, B_EPIDONE), Uint32((qi - 2) & 1))
                    full = _bar(base, B_QFULL + qi % 2)
                    qs = base + Uint32(OFF_QS) + Uint32(qi % 2) * Uint32(QS_BYTES)
                    mbar_arrive_tx(full, Uint32(4 * H * 128 + H * 64))
                    for j in cutlass.range_constexpr(4):
                        dst = base + Uint32(OFF_QN + j * ATOM)
                        if j >= 2:
                            dst = qs + Uint32((j - 2) * ATOM)
                        tma_tile2d(dst, tm_qn, full, Int32(128 * j), row * H)
                    tma_tile2d(qs + Uint32(2 * ATOM), tm_qr, full, Int32(0), row * H)
        else:
            # lane l gathers quad q = pw + NLD l of every 64-key stage; its 4 slot indices
            # are loaded one stage ahead
            q = pw + NLD * lane
            if lane >= quads:
                q = pw  # idle lanes read a valid slot of the stage (unused)
            g = Int32(0)
            for qi in range(n_mine):
                row = cta + qi * self.grid
                n = lens[row]
                subs = _subs(n)
                srow = slots + Int64(row) * Int64(slot_stride) * Int64(4) + Int64(16 * q)
                c0, c1, c2, c3 = ld_global_v4_s32(srow)
                # unroll=1: the 2x LLVM unroll (register stage addresses, BRA.DIV gather
                # blocks) ran the kernel at 3.63 vs 2.02 ms/layer (P-MLA runs 64-71)
                for t in cutlass.range(subs, unroll=1):
                    s = g % NS
                    st = base + Uint32(OFF_KV) + Uint32(s) * Uint32(STAGE64)
                    full = _bar(base, B_KVFULL + s)
                    key = t * 64 + 4 * q
                    r = (c0, c1, c2, c3)
                    if t + 1 < subs:
                        c0, c1, c2, c3 = ld_global_v4_s32(srow + Int64((t + 1) * 256))
                    # keep the prefetch here (issued before the stage-free wait): without
                    # a barrier the compiler sinks it to its use, exposing an L2 round trip
                    # per stage (2.0 vs 3.6 ms, P-MLA run66)
                    compiler_fence()
                    if lane == 0:
                        mbar_wait(_bar(base, B_KVEMPTY + s), Uint32(((g // NS) & 1) ^ 1))
                        if pw == 0:
                            self.ss(base, cta, lane, g, 22, 8)
                            self.lpin(base, cta, lane, g)
                            if cutlass.const_expr(self.no_gather):
                                mbar_arrive(full)
                            else:
                                mbar_arrive_tx(full, Uint32(STAGE64))
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
                        self.ss(base, cta, lane, g, 23, 8)
                        self.lpin(base, cta, lane, g)
                    g = g + 1

    # ------------------------------------------------------------------ MMA
    @cute.jit
    def qk_loop(self, base, lane, taddr, n_mine, lengths, cta):
        """QK(u) as soon as S buffer u % 2 is free and stage u has landed."""
        lptr = cute.make_ptr(Int32, lengths, cute.AddressSpace.gmem, assumed_align=4)
        lens = cute.make_tensor(lptr, cute.make_layout(1 << 30))
        u = Int32(0)
        for qi in range(n_mine):
            subs = _subs(lens[cta + qi * self.grid])
            mbar_wait(_bar(base, B_QTM), Uint32(qi & 1))
            tc_fence_after()
            for i in range(subs):
                if u >= 2:
                    mbar_wait(_bar(base, B_SFREE + u % 2), Uint32((u // 2 - 1) & 1))
                    tc_fence_after()
                self.qk(base, lane, taddr, u, qi, i == subs - 1, cta)
                u = u + 1

    @cute.jit
    def qk(self, base, lane, taddr, u, qi, last, cta):
        s = u % NS
        st = base + Uint32(OFF_KV) + Uint32(s) * Uint32(STAGE64)
        self.ss(base, cta, lane, u, 16, 2)
        mbar_wait(_bar(base, B_KVFULL + s), Uint32((u // NS) & 1))
        tc_fence_after()
        sc = taddr + Uint32(C_S) + Uint32(64) * Uint32(u % 2)
        qs = base + Uint32(OFF_QS) + Uint32(qi % 2) * Uint32(QS_BYTES)
        bd0 = Uint64(SDESC_SW128) | Uint64(st >> Uint32(4))
        ad0 = Uint64(SDESC_SW128) | Uint64(qs >> Uint32(4))
        for j in cutlass.range_constexpr(4):
            bd = bd0 + Uint64(j * 512)
            if cutlass.const_expr(j >= 2):
                ad = ad0 + Uint64((j - 2) * ATOM // 16)
            for k2 in cutlass.range_constexpr(2):
                if cutlass.const_expr(not self.no_mma):
                    if cutlass.const_expr(j < 2):
                        mma_ts_e(sc, taddr + Uint32(C_Q + 32 * j + 16 * k2),
                                 bd + Uint64(k2 * 4), Uint32(IDESC_QK),
                                 Uint32(1 if (j + k2) > 0 else 0))
                    else:
                        mma_ss_e(sc, ad + Uint64(k2 * 4), bd + Uint64(k2 * 4),
                                 Uint32(IDESC_QK), Uint32(1))
        ad = (ad0 + Uint64(2 * ATOM // 16)) ^ Uint64(SDESC_SW128 ^ SDESC_SW64)
        bd = (bd0 + Uint64(4 * 512)) ^ Uint64(SDESC_SW128 ^ SDESC_SW64)
        if cutlass.const_expr(not self.no_mma):
            mma_ss_e(sc, ad, bd, Uint32(IDESC_QK), Uint32(1))
        self.ss(base, cta, lane, u, 17, 2)
        commit_e(_bar(base, B_SFULL + u % 2))
        if last:
            commit_e(_bar(base, B_QREMPTY + qi % 2))

    @cute.jit
    def pv_loop(self, base, lane, taddr, n_mine, lengths, cta):
        """PV(u) once the owning softmax warpgroup released it (P written, O^T
        rescaled if needed); P^T is fetched once into the B collector."""
        lptr = cute.make_ptr(Int32, lengths, cute.AddressSpace.gmem, assumed_align=4)
        lens = cute.make_tensor(lptr, cute.make_layout(1 << 30))
        u = Int32(0)
        for qi in range(n_mine):
            subs = _subs(lens[cta + qi * self.grid])
            for i in range(subs):
                s = u % NS
                st = base + Uint32(OFF_KV) + Uint32(s) * Uint32(STAGE64)
                mbar_wait(_bar(base, B_OREADY + u % NP), Uint32((u // NP) & 1))
                self.ss(base, cta, lane, u, 18, 4)
                tc_fence_after()
                pbuf = base + Uint32(OFF_P) + Uint32(u % NP) * Uint32(P_BYTES)
                bd = Uint64(SDESC_SW64) | Uint64(pbuf >> Uint32(4))
                ad0 = Uint64(SDESC_SW128) | Uint64(st >> Uint32(4))
                for b in cutlass.range_constexpr(4):
                    if i == 0:
                        mbar_wait(_bar(base, B_OFREE + b), Uint32((qi & 1) ^ 1))
                        tc_fence_after()
                    acc = Uint32(1)
                    if i == 0:
                        acc = Uint32(0)
                    op = "fill" if b == 0 else ("lastuse" if b == 3 else "use")
                    if cutlass.const_expr(not self.no_mma):
                        mma_cb_e(taddr + Uint32(C_O + 96 * b), ad0 + Uint64(b * 512), bd,
                                 Uint32(IDESC_PV), acc, op)
                self.ss(base, cta, lane, u, 19, 4)
                commit_e(_bar(base, B_KVEMPTY + s))
                commit_e(_bar(base, B_PEMPTY + u % NP))
                if i == subs - 1:
                    commit_e(_bar(base, B_OFULL))
                u = u + 1

    # ------------------------------------------------------------------ softmax
    @cute.jit
    def softmax(self, base, tid, warp, lane, taddr, n_mine, lengths, cta, scale_log2,
                out_scale, tm_o):
        lptr = cute.make_ptr(Int32, lengths, cute.AddressSpace.gmem, assumed_align=4)
        lens = cute.make_tensor(lptr, cute.make_layout(1 << 30))
        wg = warp // 4
        sp = warp % 4
        h = tid % 128  # TMEM lane: head (S, Q) / dim (O^T)
        lane_addr = taddr + (Uint32(sp * 32) << Uint32(16))
        real = h < H
        real_m = h < (0 if self.no_math else H)
        xv = cute.make_rmem_tensor((64,), Float32)
        total = Int32(0)
        for qi in range(n_mine):
            total = total + _subs(lens[cta + qi * self.grid])
        if wg == 0 and n_mine > 0:
            self.q_copy(base, h, lane, lane_addr, Int32(0))
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
                        mp_last, lw = self.sub(base, warp, lane, wg, sp, h, lane_addr, real_m,
                                               xv, scale_log2, u, i, n, total, mp_last, lw,
                                               cta, True)
                    else:
                        mp_last, lw = self.sub(base, warp, lane, wg, sp, h, lane_addr, real_m,
                                               xv, scale_log2, u, i, n, total, mp_last, lw,
                                               cta, False)
                u = u + 1
            # ---- token end. The WG owning the last subtile (u - 1) combines
            # l = l0 2^(m0 - m) + l1 2^(m1 - m), m = max(m0, m1), copies the next token's
            # Q into TMEM and runs the epilogue; the other WG hands over (l, m) and goes on.
            owner = (u - 1) % 2 == wg
            tok = base + Uint32(OFF_TOK0)
            if qi % 2 == 1:
                tok = base + Uint32(OFF_TOK1)
            # token-parity barriers (7-9 / 12-14), see sparse_mla_h8
            xbar = Uint32(7 + sp) + Uint32(5) * Uint32(qi % 2)
            if real:
                st_shared_f32(tok + Uint32((wg * 2 * H + h) * 4), lw)
                st_shared_f32(tok + Uint32((wg * 2 * H + H + h) * 4), mp_last)
                if owner:
                    named_bar(xbar, 64)
                    o = 1 - wg
                    l1 = ld_shared_f32(tok + Uint32((o * 2 * H + h) * 4))
                    m1 = ld_shared_f32(tok + Uint32((o * 2 * H + H + h) * 4))
                    mf = cute.arch.fmax(mp_last, m1)
                    lt = lw * ex2(mp_last - mf) + l1 * ex2(m1 - mf)
                    iv = Float32(0.0)
                    if lt > Float32(0.0):
                        iv = out_scale / lt
                    st_shared_f32(base + Uint32(OFF_INV + h * 4), iv)
                else:
                    named_bar_arrive(xbar, 64)
            if owner:
                if qi + 1 < n_mine:
                    self.q_copy(base, h, lane, lane_addr, qi + 1)
                self.epilogue(base, wg, sp, lane, lane_addr, qi, cta, tm_o)

    @cute.jit
    def sub(self, base, warp, lane, wg, sp, h, lane_addr, real_m, xv, scale_log2, u, i, n,
            total, mp_last, lw, cta, masked: cutlass.Constexpr):
        """Subtile u (subtile i of its token) on WG u % 2. Returns (m, l) of this WG."""
        self.ss(base, cta, lane, u, 8 * wg + 0, 1)
        mbar_wait(_bar(base, B_SFULL + u % 2), Uint32((u // 2) & 1))
        self.ss(base, cta, lane, u, 8 * wg + 1, 1)
        tc_fence_after()
        s_addr = lane_addr + Uint32(C_S) + Uint32(64) * Uint32(u % 2)
        smax = Float32(0.0)
        if cutlass.const_expr(masked):
            sa = tmem_ld(s_addr, 64)
        else:
            sm = tmem_ld_max64(s_addr)
            sa = sm[:64]
            smax = sm[64]
        for k in cutlass.range_constexpr(64):
            xv[k] = sa[k]
        tc_fence_before()
        cute.arch.sync_warp()
        if lane == 0:
            mbar_arrive(_bar(base, B_SFREE + u % 2))
        self.ss(base, cta, lane, u, 8 * wg + 2, 1)
        mbx = base + Uint32(OFF_MBOX)
        grow = Int32(0)
        alpha = Float32(1.0)
        mp = Float32(0.0)
        if real_m:
            if cutlass.const_expr(masked):
                for k in cutlass.range_constexpr(64):
                    if 64 * i + k >= n:
                        xv[k] = Float32(-math.inf)
                xm = _tree_max([xv[k] for k in range(64)]) * scale_log2
            else:
                xm = smax * scale_log2
            # lazy-max decision after the previous subtile's (other WG) decision
            if u >= 1:
                named_bar(Uint32(1 + sp + 3 * (1 - wg)), 64)
            mprev = ld_shared_f32(mbx + Uint32(((u + 1) % 2) * H * 4 + h * 4))
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
            if mp != mp_last:
                lw = lw * ex2(mp_last - mp)
                mp_last = mp
        # publish the heads that grew (ballot masks, alphas) and OR them over the WG
        # here, where the warps are already in step; the rescale runs after P
        par = Uint32(wg * 2) + Uint32((u // 2) % 2)
        gm = base + Uint32(OFF_GMASK) + par * Uint32(16)
        alb = base + Uint32(OFF_AL) + par * Uint32(H * 4)
        gw = ballot(Uint32(grow))
        if lane == 0 and sp < 3:
            st_shared_u32(gm + Uint32(4 * sp), gw)
        if grow != 0:
            st_shared_f32(alb + Uint32(h * 4), alpha)
        anyg = bar_red_or(Uint32(10 + wg), Uint32(128), Uint32(grow))
        self.ss(base, cta, lane, u, 8 * wg + 3, 1)
        if u >= NP:
            mbar_wait(_bar(base, B_PEMPTY + u % NP), Uint32((u // NP - 1) & 1))
        self.ss(base, cta, lane, u, 8 * wg + 4, 1)
        if real_m:
            pbuf = base + Uint32(OFF_P) + Uint32(u % NP) * Uint32(P_BYTES)
            a0, a1 = Float32(0.0), Float32(0.0)
            b0, b1 = Float32(0.0), Float32(0.0)
            nmp = Float32(0.0) - mp
            for k in cutlass.range_constexpr(32):
                xv[2 * k], xv[2 * k + 1] = ffma2(xv[2 * k], xv[2 * k + 1], scale_log2,
                                                 scale_log2, nmp, nmp)
            for ch in cutlass.range_constexpr(4):
                w = []
                for q in cutlass.range_constexpr(4):
                    k = 16 * ch + 4 * q
                    p0 = ex2(xv[k])
                    p1 = ex2(xv[k + 1])
                    p2 = ex2(xv[k + 2])
                    p3 = ex2(xv[k + 3])
                    a0, a1 = fadd2(a0, a1, p0, p1)
                    b0, b1 = fadd2(b0, b1, p2, p3)
                    w.append(cvt_e4m3x2(p0, p1) | (cvt_e4m3x2(p2, p3) << Uint32(16)))
                st_shared_v4(pbuf + Uint32(h * 64) + (
                    (Uint32(ch) ^ ((Uint32(h) >> Uint32(1)) & Uint32(3))) << Uint32(4)),
                    w[0], w[1], w[2], w[3])
            a0, a1 = fadd2(a0, a1, b0, b1)
            lw = lw + (a0 + a1)
        self.ss(base, cta, lane, u, 8 * wg + 5, 1)
        # heads of this subtile that grew: rescale their O^T columns (one per block)
        # after PV(u - 1), before PV(u); every other column has alpha = 1
        if anyg != Uint32(0):
            mbar_wait(_bar(base, B_PEMPTY + (u - 1) % NP), Uint32(((u - 1) // NP) & 1))
            tc_fence_after()
            for w in cutlass.range_constexpr(3):
                m = ld_shared_u32(gm + Uint32(4 * w))
                while m != Uint32(0):
                    j = bfind(m)
                    m = m ^ (Uint32(1) << j)
                    hh = Uint32(32 * w) + j
                    al = ld_shared_f32(alb + hh * Uint32(4))
                    col = lane_addr + Uint32(C_O) + hh
                    o0, o1, o2, o3 = tmem_ld_o4(col)
                    tmem_st_o4(col, o0 * al, o1 * al, o2 * al, o3 * al)
            tc_fence_before()
        fence_async_smem()
        cute.arch.sync_warp()
        self.ss(base, cta, lane, u, 8 * wg + 6, 1)
        if lane == 0:
            mbar_arrive(_bar(base, B_OREADY + u % NP))
        return mp_last, lw

    @cute.jit
    def q_copy(self, base, h, lane, lane_addr, qi):
        """Q latent atoms 0-1 of token qi (staging, SW128) -> TMEM columns C_Q (lane =
        head), once the previous token's last QK has completed."""
        mbar_wait(_bar(base, B_QFULL + qi % 2), Uint32((qi // 2) & 1))
        if qi >= 1:
            mbar_wait(_bar(base, B_QREMPTY + (qi - 1) % 2), Uint32(((qi - 1) // 2) & 1))
        tc_fence_after()
        if h < H:
            for j in cutlass.range_constexpr(2):
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
            mbar_arrive(_bar(base, B_QLFREE))
            mbar_arrive(_bar(base, B_QTM))

    @cute.jit
    def epilogue(self, base, wg, sp, lane, lane_addr, qi, cta, tm_o):
        """O^T block b (lanes = dims, columns = heads) -> BF16 [heads x dims]: 16x256b
        TMEM loads give each thread two adjacent heads of one dim (an m8n8 b16
        fragment with dims as rows), scaled by s / l and packed; stmatrix.trans writes
        head rows into two SW128 [96 x 64 dims] staging halves (the Q latent atoms 0-1
        staging), one TMA store each. Block b is released once read."""
        row = cta + qi * self.grid
        mbar_wait(_bar(base, B_OFULL), Uint32(qi & 1))
        tc_fence_after()
        bid = Uint32(10 + wg)
        named_bar(bid, 128)  # 1/l of every head written; this WG's Q copy reads done
        t4 = lane % 4
        j = lane % 8
        mk = lane // 8  # matrix this lane addresses in stmatrix.x4
        inv_base = base + Uint32(OFF_INV)
        for b in cutlass.range_constexpr(4):
            if cutlass.const_expr(b >= 1):
                if sp == 0 and lane == 0:
                    bulk_wait_read0()
                named_bar(bid, 128)
            for hh in cutlass.range_constexpr(2):
                d0 = 32 * sp + 16 * hh  # first dim (lane) of this 16-lane half
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
            named_bar(bid, 128)
            if sp == 0 and lane == 0:
                if cutlass.const_expr(not self.no_store):
                    for hf in cutlass.range_constexpr(2):
                        tma_store2d(tm_o, base + Uint32(OFF_QN + hf * H * 128),
                                    Int32(128 * b + 64 * hf), row * H)
        if sp == 0 and lane == 0:
            bulk_wait_read0()
            mbar_arrive(_bar(base, B_EPIDONE))


@cache
def _compiled(grid: int, knobs: tuple = ()):
    kernel = SparseMlaPrefillH6(grid, knobs)
    args = (Int64(0), Int32(1), Int64(0), Int32(64), Int64(0), Int32(1), Int64(0), Int64(0),
            Float32(1.0), Float32(1.0), Int32(64), Int64(0), cuda.CUstream(0))
    # register budgets come from setmaxnreg (.reqntid 640 -> 96 at launch)
    return cute.compile(kernel, *args)


def sparse_mla_prefill_h6(q8: torch.Tensor, kv_rows: torch.Tensor, slots: torch.Tensor,
                          lengths: torch.Tensor, scale: float, out_scale: float,
                          unit_rows: int = 64, out: torch.Tensor | None = None,
                          stamps: torch.Tensor | None = None,
                          knobs: tuple = ()) -> torch.Tensor:
    """Same contract as ``sparse_mla_sol.sparse_mla_prefill_sol``: q8 [T, 96, 576]
    E4M3 (latent 512 + RoPE 64), kv_rows [rows, 576] E4M3, slots [T, >= top-k] int32
    (paged slot = page * unit_rows + offset, < 0 or >= lengths[t] masked), out [T, 96,
    512] BF16 = softmax(scale q k^T) v * out_scale. stamps (int32 [256]) only with the
    sstamp knob."""
    t = q8.shape[0]
    if q8.shape[1:] != (H, ROW) or not q8.is_contiguous():
        raise ValueError(f"sparse_mla_prefill_h6: q8 must be contiguous [T, {H}, {ROW}]")
    if out is None:
        out = torch.empty((t, H, D_NOPE), dtype=torch.bfloat16, device=q8.device)
    if not out.is_contiguous() or out.shape != (t, H, D_NOPE):
        raise ValueError("sparse_mla_prefill_h6: out must be contiguous [T, 96, 512]")
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
