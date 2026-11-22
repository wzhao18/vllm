# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 DSA sparse MLA decode core on Rubin (sm_107): the SP decode's per-rank
sparse attention (DCP shard of the KV, all gathered query heads), in place of
FlashInfer trtllm-gen.

Per row r (a verify token) with ``valid[r]`` local entries of its top-k flat KV rows
``slots[r]``: O[r, h] = softmax_k(q[r, h] . K[k] * scale) V[k] over the per-tensor FP8
MLA cache (576 B rows: 512 E4M3 latent = V | 64 E4M3 RoPE) and the FP8 query rows of
the q mailbox ([R, Hq, 576], heads 0..95 real). Output: the normalized latent
[R, Hq, 512] BF16 times ``out_scale`` (heads 0..95 written) and its LSE [R, Hq] in log2
units (``exp2(lse)`` = the row's softmax sum), as trtllm-gen returns them to DcpValue.

Layout (one row per cluster of two CTAs at a time, persistent over rows):
- the pair's CTA v owns value columns 256 v .. 256 v + 255 (O: 256 FP32 TMEM columns);
  both run the same QK and softmax (the 2-CTA split of V that keeps O, S, P and Q in
  one CTA's 576-column exclusive TMEM: O 256 + S 128 + P 32 + Q 144 = 560);
- KV tiles of 128 rows: each CTA gathers half of a tile's rows with TMA tile::gather4
  (four 128 B latent boxes, SW128, and one 64 B RoPE box, SW64) multicast to both CTAs,
  so every row is read from L2 once per pair; ``stages`` tiles in flight;
- QK: S [128 heads (96 real), 128 keys] = Q K^T, tcgen05 kind::f8f6f4 in the Rubin 2x-K
  form (K64 per MMA, FP32 accumulation) with Q (A) in TMEM, K (B) K-major in SMEM;
- softmax: warps 4..11, two per TMEM lane quarter (64 score columns each; quarter 3 =
  the padded heads 96..127 idles); running row max in log2 units, P = E4M3(448 x 2^(x -
  m)) against the running max (as trtllm-gen), FP32 row sums; O rescaled in TMEM when
  the max grows;
- PV: O += P V with P (A) in TMEM, V (B) MN-major from the same SMEM tile (the value
  half of this CTA), FP32 accumulation;
- epilogue: O / (448 l) x out_scale -> BF16, LSE = m + log2 l.
Rows with no local entry are skipped (DcpValue reads the counts and drops them).
"""

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint8, Uint32, Uint64

from vllm.models.kimi_k3.nvidia import k31_spin_guard as k31sg

from ..common import tensormap
from ..common.primitives import asm

HQ = 128  # MMA M: the query heads of a row (96 real, zero padded)
HR = 96  # real heads (quarters 0..2)
ROW = 576  # FP8 bytes per KV / query row
D_NOPE = 512
TN = 128  # keys per KV tile (QK's N, PV's K)
LOG2E = 1.4426950408889634
P_SCALE = 448.0  # E4M3 max: P = E4M3(448 x 2^(x - m)) as trtllm-gen

# SMEM (bytes): KV stages, then the Q stage, then the reduction slots and barriers.
STAGE = 4 * TN * 128 + TN * 64  # 4 SW128 latent blocks [128 x 128 B] + SW64 RoPE
QBYTES = HR * ROW  # the row's 96 real query heads, linear [96, 576]
NW = 12  # warps: 0 Q loader, 1 MMA, 2 TMEM owner, 3 spare, 4..11 softmax, 12.. KV
THREADS = 32 * NW
SOFT0 = 4  # first softmax warp
MAX_ROWS = 16  # rows per CTA pair

# Barriers (8 B each)
B_QFULL, B_QEMPTY, B_QTMEM, B_SFULL, B_SEMPTY, B_PFULL, B_PVDONE, B_OEMPTY = range(8)
B_KVFULL = 8  # + stage
B_KVEMPTY = 16  # + stage (8 stages max)
B_IDXFULL = 24  # + row parity
B_IDXEMPTY = 26  # + row parity
NBAR = 28
IDXB = 2048 * 4  # a row's slot list (top-k width <= 2048)

# TMEM columns: O 256 | S 128 | P 32 (FP8, 4 per column) | Q 144 (FP8)
C_O, C_S, C_P, C_Q = 0, 256, 384, 416

# Shared memory descriptors (version bit 46): SW128 8 rows x 128 B (SBO 1024), SW64
# 8 rows x 64 B (SBO 512).
SDESC_SW128 = (1024 >> 4 << 32) | (1 << 46) | (2 << 61)
SDESC_SW64 = (512 >> 4 << 32) | (1 << 46) | (4 << 61)
# kind::f8f6f4 instruction descriptors: FP32 D (bit 4), E4M3 A and B, M = 128 (bit 24),
# N at bit 17, bit 29 = Rubin 2x-K (K64 per instruction); bit 16 = B MN-major.
IDESC_QK = (1 << 4) | ((TN >> 3) << 17) | ((128 >> 4) << 24) | (1 << 29)
IDESC_PV = (1 << 4) | ((128 >> 3) << 17) | ((128 >> 4) << 24) | (1 << 29) | (1 << 16)


def smem_layout(stages):
    off_kv = 0
    off_idx = off_kv + stages * STAGE  # [2][2048] int32 slot lists
    off_q = off_idx + 2 * IDXB
    off_red = off_q + QBYTES  # [2 parity][4 quarters][2 halves][32] FP32 tile maxima
    off_lsum = off_red + 2 * 4 * 2 * 32 * 4  # [4][2][32] FP32 row sums
    off_n = off_lsum + 4 * 2 * 32 * 4  # [MAX_ROWS] int32 local counts of my rows
    off_bar = off_n + 4 * MAX_ROWS
    off_tmem = off_bar + NBAR * 8
    return off_kv, off_idx, off_q, off_red, off_lsum, off_n, off_bar, off_tmem, off_tmem + 16


# PTX ------------------------------------------------------------------------------
@asm()
def mbar_init(bar, count):
    return "mbarrier.init.shared::cta.b64 [$0], $1;", "r,r,~{memory}", (
        Uint32(bar),
        Uint32(count),
    )


@asm()
def mbar_arrive(bar):
    return "mbarrier.arrive.shared::cta.b64 _, [$0];", "r,~{memory}", (Uint32(bar),)


@asm()
def mbar_arrive_tx(bar, nbytes):
    return (
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [$0], $1;",
        "r,r,~{memory}",
        (Uint32(bar), Uint32(nbytes)),
    )


def mbar_wait(bar, parity):
    """Bounded parity wait (k31_spin_guard); timeouts name the caller's line."""
    return k31sg.mbar_wait(bar, parity)


k31sg.register_poll_helper(__file__, "mbar_wait")


@asm()
def named_bar(bid, n):
    return "bar.sync $0, $1;", "r,r,~{memory}", (Uint32(bid), Uint32(n))


@asm()
def tc_fence_after():
    return "tcgen05.fence::after_thread_sync;", "~{memory}", ()


@asm()
def tc_fence_before():
    return "tcgen05.fence::before_thread_sync;", "~{memory}", ()


@asm()
def tc_commit(bar):
    return (
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [$0];",
        "r,~{memory}",
        (Uint32(bar),),
    )


@asm()
def tc_commit_pair(bar):
    """Commit arriving on ``bar`` in both CTAs of the pair."""
    code = (
        "{ .reg .b16 m; mov.b16 m, 3; tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.multicast::cluster.b64 [$0], m; }"
    )
    return code, "r,~{memory}", (Uint32(bar),)


@asm()
def mma_ts(d, a, b, idesc, acc):
    """D (TMEM) (+)= A (TMEM, K-major, 4 E4M3 per column) . B (SMEM descriptor)."""
    code = (
        "{ .reg .pred p; setp.ne.b32 p, $4, 0; "
        "tcgen05.mma.cta_group::1.kind::f8f6f4 [$0], [$1], $2, $3, p; }"
    )
    return code, "r,r,l,r,r,~{memory}", (
        Uint32(d),
        Uint32(a),
        Uint64(b),
        Uint32(idesc),
        Uint32(acc),
    )


@asm()
def tmem_ld(addr, n):
    regs = ",".join(f"${i}" for i in range(n))
    code = (
        f"tcgen05.ld.sync.aligned.32x32b.x{n}.b32 {{{regs}}}, [${n}]; "
        "tcgen05.wait::ld.sync.aligned;"
    )
    return code, ",".join(["=f"] * n + ["r", "~{memory}"]), (Uint32(addr),), (
        Float32,
    ) * n


@asm()
def tmem_st_f(addr, vals):
    n = len(vals)
    regs = ",".join(f"${i + 1}" for i in range(n))
    code = (
        f"tcgen05.st.sync.aligned.32x32b.x{n}.b32 [$0], {{{regs}}}; "
        "tcgen05.wait::st.sync.aligned;"
    )
    return code, "r" + ",f" * n + ",~{memory}", (
        Uint32(addr),
        *[Float32(v) for v in vals],
    )


@asm()
def tmem_st_u(addr, words):
    n = len(words)
    regs = ",".join(f"${i + 1}" for i in range(n))
    code = (
        f"tcgen05.st.sync.aligned.32x32b.x{n}.b32 [$0], {{{regs}}}; "
        "tcgen05.wait::st.sync.aligned;"
    )
    return code, "r" + ",r" * n + ",~{memory}", (
        Uint32(addr),
        *[Uint32(w) for w in words],
    )


@asm()
def ld_shared_v4(addr):
    return (
        "ld.shared.v4.b32 {$0,$1,$2,$3}, [$4];",
        "=r,=r,=r,=r,r,~{memory}",
        (Uint32(addr),),
        (Uint32,) * 4,
    )


@asm(Float32, pure=False)
def ld_shared_f32(addr):
    return "ld.shared.f32 $0, [$1];", "=f,r,~{memory}", (Uint32(addr),)


@asm(Uint32, pure=False)
def ld_shared_u32(addr):
    return "ld.shared.b32 $0, [$1];", "=r,r,~{memory}", (Uint32(addr),)


@asm()
def st_shared_f32(addr, v):
    return "st.shared.f32 [$0], $1;", "r,f,~{memory}", (Uint32(addr), Float32(v))


@asm()
def st_shared_u32(addr, w):
    return "st.shared.b32 [$0], $1;", "r,r,~{memory}", (Uint32(addr), Uint32(w))


@asm()
def st_global_v4(ptr, w0, w1, w2, w3):
    return (
        "st.global.v4.b32 [$0], {$1,$2,$3,$4};",
        "l,r,r,r,r,~{memory}",
        (Int64(ptr), Uint32(w0), Uint32(w1), Uint32(w2), Uint32(w3)),
    )


@asm()
def st_global_f32(ptr, v):
    return "st.global.f32 [$0], $1;", "l,f,~{memory}", (Int64(ptr), Float32(v))


@asm(Float32, pure=True)
def ex2(v):
    return "ex2.approx.ftz.f32 $0, $1;", "=f,f", (Float32(v),)


@asm(Float32, pure=True)
def lg2(v):
    return "lg2.approx.ftz.f32 $0, $1;", "=f,f", (Float32(v),)


@asm(Float32, pure=True)
def fmax(a, b):
    return "max.f32 $0, $1, $2;", "=f,f,f", (Float32(a), Float32(b))


@asm(Uint32, pure=True)
def e4m3x4(a, b, c, d):
    """Four FP32 -> four E4M3 bytes (a lowest), round to nearest, saturating."""
    code = (
        "{ .reg .b16 lo, hi; cvt.rn.satfinite.e4m3x2.f32 lo, $2, $1; "
        "cvt.rn.satfinite.e4m3x2.f32 hi, $4, $3; mov.b32 $0, {lo, hi}; }"
    )
    return code, "=r,f,f,f,f", (Float32(a), Float32(b), Float32(c), Float32(d))


@asm(Uint32, pure=True)
def bf16x2(lo, hi):
    return "cvt.rn.bf16x2.f32 $0, $2, $1;", "=r,f,f", (Float32(lo), Float32(hi))


@asm(Uint32, pure=False)
def vote_any(flag):
    code = (
        "{ .reg .pred p, q; setp.ne.u32 p, $1, 0; vote.sync.any.pred q, p, 0xffffffff; "
        "selp.u32 $0, 1, 0, q; }"
    )
    return code, "=r,r", (Uint32(flag),)


@asm(Uint32, pure=True)
def cluster_rank():
    return "mov.u32 $0, %cluster_ctarank;", "=r", ()


@asm()
def cluster_sync():
    return (
        "barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;",
        "~{memory}",
        (),
    )


@asm()
def fence_mbar_init():
    return "fence.mbarrier_init.release.cluster;", "~{memory}", ()


@asm()
def gather4_pair(dst, desc, bar, col, r0, r1, r2, r3):
    """TMA tile::gather4 of rows r0..r3 (-1: zero fill) of one box into ``dst`` of
    both CTAs of the pair, completing on ``bar`` in each."""
    code = (
        "{ .reg .b16 m; mov.b16 m, 3; "
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
        ".mbarrier::complete_tx::bytes.multicast::cluster"
        " [$0], [$1, {$3, $4, $5, $6, $7}], [$2], m; }"
    )
    ops = (
        Uint32(dst),
        Int64(desc),
        Uint32(bar),
        Int32(col),
        Int32(r0),
        Int32(r1),
        Int32(r2),
        Int32(r3),
    )
    return code, "r,l,r,r,r,r,r,r,~{memory}", ops


@asm()
def gather4(dst, desc, bar, col, r0, r1, r2, r3):
    code = (
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
        ".mbarrier::complete_tx::bytes [$0], [$1, {$3, $4, $5, $6, $7}], [$2];"
    )
    ops = (Uint32(dst), Int64(desc), Uint32(bar), Int32(col), Int32(r0), Int32(r1),
           Int32(r2), Int32(r3))
    return code, "r,l,r,r,r,r,r,r,~{memory}", ops


@asm()
def bulk_load(dst, src, nbytes, bar):
    code = (
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[$0], [$1], $2, [$3];"
    )
    return code, "r,l,r,r,~{memory}", (
        Uint32(dst),
        Int64(src),
        Uint32(nbytes),
        Uint32(bar),
    )


@asm()
def ld_global_v4_s32(ptr):
    return (
        "ld.global.nc.v4.b32 {$0,$1,$2,$3}, [$4];",
        "=r,=r,=r,=r,l",
        (Int64(ptr),),
        (Int32,) * 4,
    )


@asm(Int32, pure=False)
def ld_global_s32(ptr):
    return "ld.global.nc.s32 $0, [$1];", "=r,l", (Int64(ptr),)


@asm()
def tmem_alloc_exclusive(dst):
    code = (
        "tcgen05.alloc.exclusive.cta_group::1.sync.aligned.shared::cta.b32 [$0], 576; "
        "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"
    )
    return code, "r,~{memory}", (Uint32(dst),)


@asm()
def tmem_free_exclusive(taddr):
    return (
        "tcgen05.dealloc.exclusive.cta_group::1.sync.aligned.b32 $0, 576;",
        "r,~{memory}",
        (Uint32(taddr),),
    )


@asm(Uint64, pure=False)
def globaltimer():
    return "mov.u64 $0, %globaltimer;", "=l,~{memory}", ()


@asm()
def st_global_u64(ptr, value):
    return "st.global.u64 [$0], $1;", "l,l,~{memory}", (Int64(ptr), Uint64(value))


def _tree(values, op):
    """Pairwise reduction (Python-level: unrolled at trace time)."""
    values = list(values)
    while len(values) > 1:
        values = [op(values[2 * k], values[2 * k + 1]) for k in range(len(values) // 2)]
    return values[0]


def _add(a, b):
    return a + b


@cute.jit
def _tiles(n):
    return (n + (TN - 1)) // TN


class SparseMlaDecode:
    """``clusters`` persistent CTA pairs; ``stages`` KV tiles in flight per CTA."""

    def __init__(self, clusters: int, stages: int = 3, stamps: bool = False,
                 loaders: int = 8, multicast: bool = True):
        assert 2 <= stages <= 8 and 1 <= loaders <= 16
        # ``loaders`` KV gather4-issuing warps (12 ..); ``multicast``: each CTA of
        # the pair gathers half of a tile into both, else every CTA all of it.
        self.loaders, self.mc = loaders, multicast
        self.threads = 32 * (NW + loaders)
        self.clusters, self.stages = clusters, stages
        # ``stamps`` (benchmarks only): %globaltimer phase stamps, 64 per CTA
        self.stamps = stamps
        (self.off_kv, self.off_idx, self.off_q, self.off_red, self.off_lsum, self.off_n,
         self.off_bar, self.off_tmem, self.smem) = smem_layout(stages)

    @cute.jit
    def nrow(self, base, i):
        return Int32(ld_shared_u32(base + Uint32(self.off_n) + Uint32(4) * Uint32(i)))

    def bar(self, base, i):
        return base + Uint32(self.off_bar + 8 * i)

    @cute.jit
    def mark(self, stamp, bid, i):
        if cutlass.const_expr(self.stamps):
            st_global_u64(stamp + (Int64(bid) * Int64(64) + Int64(i)) * Int64(8),
                          globaltimer())

    @cute.jit
    def __call__(
        self,
        q: Int64,
        q_stride: Int32,
        rows: Int32,
        kv: Int64,
        kv_count: Int32,
        slots: Int64,
        slot_stride: Int32,
        valid: Int64,
        out: Int64,
        out_stride: Int32,
        lse: Int64,
        lse_stride: Int32,
        scale_log2: Float32,
        out_scale: Float32,
        stamp: Int64,
        stream: cuda.CUstream,
    ):
        u8 = cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8
        l2 = cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B
        sw = cuda.CUtensorMapSwizzle
        kvm = tensormap.encode_map(
            kv, u8, (ROW, kv_count), (ROW,), (128, 1), sw.CU_TENSOR_MAP_SWIZZLE_128B, l2
        )
        kvr = tensormap.encode_map(
            kv, u8, (ROW, kv_count), (ROW,), (64, 1), sw.CU_TENSOR_MAP_SWIZZLE_64B, l2
        )
        self.kernel(
            kvm, kvr, q, q_stride, rows, slots, slot_stride, valid, out, out_stride,
            lse, lse_stride, scale_log2, out_scale, stamp,
        ).launch(
            grid=(2 * self.clusters, 1, 1),
            block=(self.threads, 1, 1),
            cluster=(2, 1, 1),
            smem=self.smem,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        kvm: tensormap.GridMap,
        kvr: tensormap.GridMap,
        q: Int64,
        q_stride: Int32,
        rows: Int32,
        slots: Int64,
        slot_stride: Int32,
        valid: Int64,
        out: Int64,
        out_stride: Int32,
        lse: Int64,
        lse_stride: Int32,
        scale_log2: Float32,
        out_scale: Float32,
        stamp: Int64,
    ):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(tid // 32)
        lane = tid % 32
        v = cluster_rank()
        pair = bid // 2
        smem = cutlass.utils.SmemAllocator()
        raw = smem.allocate_array(Uint8, self.smem, byte_alignment=1024)
        base = Uint32(raw.toint())
        if tid == 0:
            self.mark(stamp, bid, 0)
        if tid == 0:
            for i, count in (
                (B_QFULL, 1), (B_QEMPTY, 8), (B_QTMEM, 8), (B_SFULL, 1),
                (B_SEMPTY, 8), (B_PFULL, 8), (B_PVDONE, 1), (B_OEMPTY, 8),
            ):
                mbar_init(self.bar(base, i), Uint32(count))
            for s in cutlass.range_constexpr(self.stages):
                mbar_init(self.bar(base, B_KVFULL + s), Uint32(1))
                mbar_init(self.bar(base, B_KVEMPTY + s), Uint32(2 if self.mc else 1))
            for b in cutlass.range_constexpr(2):
                mbar_init(self.bar(base, B_IDXFULL + b), Uint32(1))
                mbar_init(self.bar(base, B_IDXEMPTY + b), Uint32(self.loaders))
            fence_mbar_init()
        if warp == 2:
            tmem_alloc_exclusive(base + Uint32(self.off_tmem))
        n_mine = (rows - pair + self.clusters - 1) // self.clusters
        if warp == 3:
            if lane < n_mine:
                st_shared_u32(base + Uint32(self.off_n + 4 * lane), Uint32(
                    ld_global_s32(valid + Int64(pair + lane * self.clusters) * Int64(4))))
        tc_fence_before()
        cute.arch.sync_threads()
        cluster_sync()  # both CTAs' barriers initialized before any multicast
        if tid == 0:
            self.mark(stamp, bid, 1)
        tc_fence_after()
        taddr = ld_shared_u32(base + Uint32(self.off_tmem))
        if warp == 0:
            self.qloader(base, lane, pair, n_mine, q, q_stride, valid)
        if warp == 3:
            if lane == 0:
                self.idxloader(base, pair, n_mine, slots, slot_stride)
        if warp >= NW:
            self.loader(base, warp - NW, lane, v, pair, n_mine, slots, slot_stride,
                        valid, tensormap.map_address(kvm), tensormap.map_address(kvr))
        if warp == 1:
            if lane == 0:
                self.mma(base, v, taddr, pair, n_mine, valid, stamp, bid)
        if (warp >= SOFT0) & (warp < NW):
            self.softmax(base, warp, lane, v, taddr, pair, n_mine, valid, out,
                         out_stride, lse, lse_stride, scale_log2, out_scale, stamp,
                         bid)
        tc_fence_before()
        cute.arch.sync_threads()
        cluster_sync()  # every multicast into / remote arrive on this CTA has landed
        if warp == 2:
            tc_fence_after()
            tmem_free_exclusive(taddr)

    # ------------------------------------------------------------------ loaders
    @cute.jit
    def qloader(self, base, lane, pair, n_mine, q, q_stride, valid):
        """Every row's query rows into the Q stage as soon as softmax has moved the
        previous one into TMEM (18 bulk copies of 3 KB, one per lane)."""
        for i in range(n_mine):
            row = pair + i * self.clusters
            if lane == 0:
                mbar_wait(self.bar(base, B_QEMPTY), (i & 1) ^ 1)
                mbar_arrive_tx(self.bar(base, B_QFULL), Uint32(QBYTES))
            cute.arch.sync_warp()
            if lane < 18:
                chunk = QBYTES // 18
                bulk_load(base + Uint32(self.off_q) + Uint32(lane * chunk),
                          q + Int64(row) * Int64(q_stride) + Int64(lane * chunk),
                          Uint32(chunk), self.bar(base, B_QFULL))

    @cute.jit
    def idxloader(self, base, pair, n_mine, slots, slot_stride):
        """Each non-empty row's slot list (its valid prefix, 16 B granules) into the
        two-row index ring, ahead of the loaders."""
        rk = Int32(0)
        for i in range(n_mine):
            n = self.nrow(base, i)
            if n > 0:
                row = pair + i * self.clusters
                b = rk & 1
                mbar_wait(self.bar(base, B_IDXEMPTY + b), ((rk >> 1) & 1) ^ 1)
                nbytes = Uint32(((n + 3) // 4) * 16)
                mbar_arrive_tx(self.bar(base, B_IDXFULL + b), nbytes)
                bulk_load(base + Uint32(self.off_idx) + Uint32(b) * Uint32(IDXB),
                          slots + Int64(row) * Int64(slot_stride) * Int64(4), nbytes,
                          self.bar(base, B_IDXFULL + b))
                rk = rk + 1

    @cute.jit
    def loader(self, base, lw, lane, v, pair, n_mine, slots, slot_stride, valid, tm_kv,
               tm_kr):
        """Loader warp lw, lane m: quad lw + L m of this CTA's share of every tile (the
        pair's half v with multicast, else the whole tile); warp 0 arms the stage."""
        quads: cutlass.Constexpr = 16 if self.mc else 32
        per: cutlass.Constexpr = (quads + self.loaders - 1) // self.loaders
        g = Int32(0)
        rk = Int32(0)
        for i in range(n_mine):
            n = self.nrow(base, i)
            if n > 0:
                tiles = _tiles(n)
                b = rk & 1
                mbar_wait(self.bar(base, B_IDXFULL + b), (rk >> 1) & 1)
                srow = base + Uint32(self.off_idx) + Uint32(b) * Uint32(IDXB)
                for t in range(tiles):
                    s = g % self.stages
                    st = base + Uint32(self.off_kv) + Uint32(s) * Uint32(STAGE)
                    full = self.bar(base, B_KVFULL + s)
                    if lane == 0:
                        mbar_wait(self.bar(base, B_KVEMPTY + s),
                                  ((g // self.stages) & 1) ^ 1)
                        if lw == 0:
                            mbar_arrive_tx(full, Uint32(STAGE))
                    cute.arch.sync_warp()
                    local = lw + self.loaders * lane
                    if (lane < per) & (local < quads):
                        quad = local
                        if cutlass.const_expr(self.mc):
                            quad = Int32(16) * Int32(v) + local
                        key = t * TN + 4 * quad
                        r = ld_shared_v4(srow + Uint32(key) * Uint32(4))
                        r0, r1, r2, r3 = Int32(r[0]), Int32(r[1]), Int32(r[2]), Int32(r[3])
                        if key >= n:
                            r0 = Int32(-1)
                        if key + 1 >= n:
                            r1 = Int32(-1)
                        if key + 2 >= n:
                            r2 = Int32(-1)
                        if key + 3 >= n:
                            r3 = Int32(-1)
                        for a in cutlass.range_constexpr(5):
                            dst = st + Uint32(a * TN * 128) + Uint32(quad) * 512
                            desc, col = tm_kv, 128 * a
                            if cutlass.const_expr(a == 4):
                                dst = st + Uint32(4 * TN * 128) + Uint32(quad) * 256
                                desc, col = tm_kr, D_NOPE
                            if cutlass.const_expr(self.mc):
                                gather4_pair(dst, desc, full, col, r0, r1, r2, r3)
                            else:
                                gather4(dst, desc, full, col, r0, r1, r2, r3)
                    g = g + 1
                cute.arch.sync_warp()
                if lane == 0:
                    mbar_arrive(self.bar(base, B_IDXEMPTY + b))
                rk = rk + 1
        # Producer tail: the last commits (the pair's too) on this CTA's empty barriers.
        if (lw == 0) & (lane == 0):
            for j in cutlass.range_constexpr(self.stages):
                gj = g + j
                mbar_wait(self.bar(base, B_KVEMPTY + gj % self.stages),
                          ((gj // self.stages) & 1) ^ 1)

    # ------------------------------------------------------------------ MMA
    @cute.jit
    def qk(self, base, taddr, g, stamp, bid, mark):
        s = g % self.stages
        st = base + Uint32(self.off_kv) + Uint32(s) * Uint32(STAGE)
        mbar_wait(self.bar(base, B_KVFULL + s), (g // self.stages) & 1)
        if mark < 8:
            self.mark(stamp, bid, 48 + mark)
        mbar_wait(self.bar(base, B_SEMPTY), (g & 1) ^ 1)
        tc_fence_after()
        for kk in cutlass.range_constexpr(8):
            bd = Uint64(SDESC_SW128) | Uint64(
                (st + Uint32((kk // 2) * TN * 128 + (kk % 2) * 64)) >> Uint32(4))
            mma_ts(taddr + Uint32(C_S), taddr + Uint32(C_Q + 16 * kk), bd,
                   Uint32(IDESC_QK), Uint32(1 if kk > 0 else 0))
        bd = Uint64(SDESC_SW64) | Uint64((st + Uint32(4 * TN * 128)) >> Uint32(4))
        mma_ts(taddr + Uint32(C_S), taddr + Uint32(C_Q + 128), bd, Uint32(IDESC_QK),
               Uint32(1))
        tc_commit(self.bar(base, B_SFULL))

    @cute.jit
    def mma(self, base, v, taddr, pair, n_mine, valid, stamp, bid):
        """One thread: QK(g + 1) is issued before PV(g), so softmax(g + 1) overlaps
        PV(g); a row's first QK follows its Q (in TMEM from the previous row's last
        tile on), so it overlaps the previous row's epilogue."""
        g = Int32(0)
        ro = Int32(0)
        for i in range(n_mine):
            n = self.nrow(base, i)
            mbar_wait(self.bar(base, B_QTMEM), i & 1)
            if n > 0:
                tiles = _tiles(n)
                if ro < 2:
                    self.mark(stamp, bid, 40 + 4 * ro)
                self.qk(base, taddr, g, stamp, bid, ro * 8)
                for t in range(tiles):
                    if t + 1 < tiles:
                        self.qk(base, taddr, g + 1, stamp, bid, ro * 8 + t + 1)
                    if t == 0:
                        mbar_wait(self.bar(base, B_OEMPTY), (ro & 1) ^ 1)
                    mbar_wait(self.bar(base, B_PFULL), g & 1)
                    if ro * 8 + t < 8:
                        self.mark(stamp, bid, 56 + t)
                    tc_fence_after()
                    s = g % self.stages
                    st = base + Uint32(self.off_kv) + Uint32(s) * Uint32(STAGE)
                    for nb in cutlass.range_constexpr(2):
                        vblock = st + (Uint32(2) * v + Uint32(nb)) * Uint32(TN * 128)
                        for kk in cutlass.range_constexpr(2):
                            bd = Uint64(SDESC_SW128) | Uint64(
                                (vblock + Uint32(kk * 64 * 128)) >> Uint32(4))
                            acc = Uint32(1)
                            if t == 0:
                                acc = Uint32(kk)
                            mma_ts(taddr + Uint32(C_O + 128 * nb),
                                   taddr + Uint32(C_P + 16 * kk), bd,
                                   Uint32(IDESC_PV), acc)
                    if cutlass.const_expr(self.mc):
                        tc_commit_pair(self.bar(base, B_KVEMPTY + s))
                    else:
                        tc_commit(self.bar(base, B_KVEMPTY + s))
                    tc_commit(self.bar(base, B_PVDONE))
                    g = g + 1
                ro = ro + 1

    # ------------------------------------------------------------------ softmax
    @cute.jit
    def copy_q(self, base, warp, lane, taddr, qi, src, keep, col, stamp, bid):
        """Row qi's query (this warp's half of its head) Q stage -> TMEM; the previous
        row's last QK is complete (its S_FULL was waited)."""
        mbar_wait(self.bar(base, B_QFULL), qi & 1)
        if (warp == SOFT0) & (lane == 0) & (qi < 2):
            self.mark(stamp, bid, 2 + 16 * qi)
        qwords = []
        for c in cutlass.range_constexpr(18):
            w4 = ld_shared_v4(src + Uint32(16 * c))
            for e in cutlass.range_constexpr(4):
                qwords.append(w4[e] & keep)
        tmem_st_u(col, qwords[:64])
        tmem_st_u(col + Uint32(64), qwords[64:])
        tc_fence_before()
        cute.arch.sync_warp()
        if lane == 0:
            mbar_arrive(self.bar(base, B_QTMEM))
            mbar_arrive(self.bar(base, B_QEMPTY))

    @cute.jit
    def softmax(self, base, warp, lane, v, taddr, pair, n_mine, valid, out,
                out_stride, lse, lse_stride, scale_log2, out_scale, stamp, bid):
        """Warp w: TMEM lane quarter w % 4 (head = 32 quarter + lane), score half
        (w - 4) // 4. Quarter 3 (padded heads 96..127) runs on zero query rows and
        stores nothing."""
        quarter = warp % 4
        half = (warp - SOFT0) // 4
        real = quarter < 3
        keep = Uint32(0) - Uint32(real)
        head = quarter * 32 + lane
        qhead = head - 32 * (quarter // 3)  # (quarter 3 reads a real row, then zeroes it)
        lanes = Uint32(quarter * 32) << Uint32(16)
        qsrc = base + Uint32(self.off_q) + Uint32(qhead * ROW + half * 288)
        qcol = taddr + lanes + Uint32(C_Q + half * 72)
        red = base + Uint32(self.off_red) + Uint32(quarter * 256 + half * 128 + lane * 4)
        red_other = base + Uint32(self.off_red) + Uint32(
            quarter * 256 + (1 - half) * 128 + lane * 4)
        lsum = base + Uint32(self.off_lsum) + Uint32(quarter * 256 + half * 128 + lane * 4)
        lsum_other = base + Uint32(self.off_lsum) + Uint32(
            quarter * 256 + (1 - half) * 128 + lane * 4)
        if n_mine > 0:
            self.copy_q(base, warp, lane, taddr, Int32(0), qsrc, keep, qcol, stamp, bid)
        g = Int32(0)
        ro = Int32(0)
        for i in range(n_mine):
            row = pair + i * self.clusters
            n = self.nrow(base, i)
            more = i + 1 < n_mine
            if n > 0:
                tiles = _tiles(n)
                m = Float32(-float("inf"))
                l = Float32(0.0)
                for t in range(tiles):
                    mbar_wait(self.bar(base, B_SFULL), g & 1)
                    if (warp == SOFT0) & (lane == 0) & (ro < 2) & (t < 10):
                        self.mark(stamp, bid, 3 + 16 * ro + t)
                    if (t == tiles - 1) & more:  # QK(row) done: next row's Q -> TMEM
                        self.copy_q(base, warp, lane, taddr, i + 1, qsrc, keep, qcol,
                                    stamp, bid)
                    tc_fence_after()
                    sv = tmem_ld(taddr + lanes + Uint32(C_S + 64 * half), 64)
                    tc_fence_before()
                    cute.arch.sync_warp()
                    if lane == 0:
                        mbar_arrive(self.bar(base, B_SEMPTY))
                    # valid columns of this half: keys < n
                    left = n - (t * TN + 64 * half)
                    x = []
                    for j in cutlass.range_constexpr(64):
                        xj = sv[j] * scale_log2
                        x.append(cutlass.select_(left > j, xj, Float32(-float("inf"))))
                    tmax = _tree(x, fmax)
                    parity = Uint32((g & 1) * 1024)
                    st_shared_f32(red + parity, tmax)
                    named_bar(Uint32(1) + Uint32(quarter), Uint32(64))
                    other = ld_shared_f32(red_other + parity)
                    m_new = fmax(m, fmax(tmax, other))
                    alpha = ex2(m - m_new)
                    e = [ex2(x[j] - m_new) for j in range(64)]
                    l = l * alpha + _tree(e, _add)
                    pw = Float32(P_SCALE)
                    pwords = [
                        e4m3x4(e[4 * w] * pw, e[4 * w + 1] * pw, e[4 * w + 2] * pw,
                               e[4 * w + 3] * pw)
                        for w in range(16)
                    ]
                    grew = vote_any(Uint32(m_new > m))
                    if g > 0:  # PV(g - 1) is complete: P is free, O is stable
                        mbar_wait(self.bar(base, B_PVDONE), (g - 1) & 1)
                    tc_fence_after()
                    if (t > 0) & (grew != Uint32(0)):
                        for c in cutlass.range_constexpr(4):
                            ocol = taddr + lanes + Uint32(C_O + 128 * half + 32 * c)
                            rv = tmem_ld(ocol, 32)
                            tmem_st_f(ocol, [rv[j] * alpha for j in range(32)])
                    tmem_st_u(taddr + lanes + Uint32(C_P + 16 * half), pwords)
                    m = m_new
                    tc_fence_before()
                    cute.arch.sync_warp()
                    if lane == 0:
                        mbar_arrive(self.bar(base, B_PFULL))
                    g = g + 1
                # Epilogue: O is complete after the row's last PV.
                mbar_wait(self.bar(base, B_PVDONE), (g - 1) & 1)
                if (warp == SOFT0) & (lane == 0) & (ro < 2):
                    self.mark(stamp, bid, 13 + 16 * ro)
                tc_fence_after()
                st_shared_f32(lsum, l)
                named_bar(Uint32(1) + Uint32(quarter), Uint32(64))
                l_all = l + ld_shared_f32(lsum_other)
                inv = out_scale / (l_all * Float32(P_SCALE))
                packed = []
                for c in cutlass.range_constexpr(4):
                    ov = tmem_ld(taddr + lanes + Uint32(C_O + 128 * half + 32 * c), 32)
                    for j in cutlass.range_constexpr(16):
                        packed.append(bf16x2(ov[2 * j] * inv, ov[2 * j + 1] * inv))
                tc_fence_before()
                cute.arch.sync_warp()
                if lane == 0:
                    mbar_arrive(self.bar(base, B_OEMPTY))
                if (warp == SOFT0) & (lane == 0) & (ro < 2):
                    self.mark(stamp, bid, 14 + 16 * ro)
                if real:
                    if (v == Uint32(0)) & (half == 0):
                        st_global_f32(
                            lse + (Int64(row) * Int64(lse_stride) + Int64(head)) * Int64(4),
                            m + lg2(l_all),
                        )
                    dst = out + (
                        Int64(row) * Int64(out_stride)
                        + Int64(head) * Int64(D_NOPE)
                        + Int64(v) * Int64(256)
                        + Int64(half * 128)
                    ) * Int64(2)
                    for e4 in cutlass.range_constexpr(16):
                        st_global_v4(dst + Int64(16 * e4), packed[4 * e4],
                                     packed[4 * e4 + 1], packed[4 * e4 + 2],
                                     packed[4 * e4 + 3])
                ro = ro + 1
            else:
                if more:  # no QK for this row: the next row's Q follows at once
                    self.copy_q(base, warp, lane, taddr, i + 1, qsrc, keep, qcol, stamp,
                                bid)


@cache
def _compiled(clusters: int, stages: int, stamps: bool = False, loaders: int = 8,
              multicast: bool = True):
    kernel = SparseMlaDecode(clusters, stages, stamps, loaders, multicast)
    args = (
        Int64(0), Int32(1), Int32(1), Int64(0), Int32(64), Int64(0), Int32(1),
        Int64(0), Int64(0), Int32(1), Int64(0), Int32(1), Float32(1.0), Float32(1.0), Int64(0),
        cuda.CUstream(0),
    )
    opts = f"--ptxas-options -maxrregcount={min(255, 65536 // kernel.threads // 8 * 8)}"
    return cute.compile(kernel, *args, options=opts)


def sparse_mla_decode(query, kv_rows, slots, valid, latent, lse, scale, out_scale,
                      clusters=None, stages=3, stamps=None, loaders=8,
                      multicast=True):
    """``query`` [R, Hq, 576] FP8 (bytes or E4M3; Hq >= 96, heads 0..95 real, rows
    contiguous); ``kv_rows`` [N, 576] the flat cache rows; ``slots`` [R, W] int32 flat
    rows (W a multiple of 128, a compacted prefix of ``valid[r]`` entries per row);
    ``latent`` [R, Hq, 512] BF16 and ``lse`` [R, Hq] FP32 outputs (heads 0..95 of rows
    with a local entry written); ``scale`` = softmax x q x k scale, ``out_scale`` = k
    scale (trtllm-gen's bmm1 / bmm2 scales)."""
    rows = query.shape[0]
    assert query.shape[2] == ROW and query.shape[1] >= HR and query.stride(1) == ROW
    assert kv_rows.shape[1] == ROW and kv_rows.stride(0) == ROW
    assert slots.shape[1] % TN == 0 and slots.shape[1] <= IDXB // 4
    assert slots.stride(1) == 1 and slots.stride(0) % 4 == 0
    assert latent.shape[2] == D_NOPE and latent.stride(1) == D_NOPE
    if rows == 0:
        return latent, lse
    if clusters is None:
        sms = torch.cuda.get_device_properties(query.device).multi_processor_count
        clusters = min(sms // 2, rows)
    if rows > MAX_ROWS * clusters:
        raise ValueError(f"sparse_mla_decode: {rows} rows exceed {MAX_ROWS} per CTA pair")
    fn = _compiled(clusters, stages, stamps is not None, loaders, multicast)
    stream = cuda.CUstream(torch.cuda.current_stream(query.device).cuda_stream)
    fn(
        Int64(query.data_ptr()), Int32(query.stride(0)), Int32(rows),
        Int64(kv_rows.data_ptr()), Int32(kv_rows.shape[0]), Int64(slots.data_ptr()),
        Int32(slots.stride(0)), Int64(valid.data_ptr()), Int64(latent.data_ptr()),
        Int32(latent.stride(0)), Int64(lse.data_ptr()), Int32(lse.stride(0)),
        Float32(scale * LOG2E), Float32(out_scale),
        Int64(0 if stamps is None else stamps.data_ptr()), stream,
    )
    return latent, lse
