# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 DSA sparse MLA decode core on Rubin (sm_107), keys on M: the SP decode's
per-rank sparse attention over this rank's DCP shard of the KV, all 96 gathered query
heads, in place of FlashInfer trtllm-gen (same inputs and outputs as
``sparse_mla.sparse_mla_decode``).

One CTA per SM, persistent over row segments (split-K: the rows' key tiles cut evenly
over the CTAs, partials merged by each row's first CTA, see MAX_SEG); every gathered KV
row lands in one SM once:
- KV tiles of 128 rows by TMA tile::gather4 (four 128 B latent boxes, SW128, one 64 B
  RoPE box, SW64) from ``loaders`` warps, ``stages`` tiles in flight; each row's slot
  list is bulk-copied into a two-row SMEM ring ahead of the loaders;
- QK: S^T [128 keys, 96 heads] = K Q^T, tcgen05 kind::f8f6f4 (Rubin 2x-K, K64 per
  MMA, FP32 accumulation), A = the gathered K tile (K-major), B = the row's 96 FP8
  query heads (TMA, K-major): no head padding; two S^T buffers in TMEM;
- softmax: three warpgroups of 32 heads, thread = key; the running max per head in
  log2 units (warp max reductions + a cross-quarter combine), P = E4M3(448 x 2^(x - m))
  against the running max as trtllm-gen, FP32 row sums; O^T rescaled in TMEM where a
  head's max grew, by separate correction warps (off the softmax path);
- PV: O^T [512 dims, 96 heads] += V^T P^T, 4 MMAs of M = 128 dims, A = V^T (the same
  gathered tile read MN-major), B = P^T (SMEM, MN-major), FP32 accumulation;
- epilogue (the correction warps, under the next row's softmax): O / (448 l) x
  out_scale -> BF16 latent [R, Hq, 512] (heads 0..95), LSE = m + log2 l -> [R, Hq].
TMEM (exclusive, 576 columns): S^T 2 x 96 | O^T 4 x 96.
"""

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint8, Uint32, Uint64

from ..common import tensormap
from ..common.primitives import asm
from .sparse_mla import (
    D_NOPE,
    IDXB,
    LOG2E,
    P_SCALE,
    ROW,
    SDESC_SW64,
    SDESC_SW128,
    STAGE,
    TN,
    _tiles,
    bulk_load,
    e4m3x4,
    ex2,
    fence_mbar_init,
    fmax,
    gather4,
    globaltimer,
    ld_shared_f32,
    ld_shared_u32,
    ld_shared_v4,
    lg2,
    mbar_arrive,
    mbar_arrive_tx,
    mbar_init,
    mbar_wait,
    named_bar,
    st_global_f32,
    st_global_u64,
    st_shared_f32,
    st_shared_u32,
    tc_commit,
    tc_fence_after,
    tc_fence_before,
    tmem_alloc_exclusive,
    tmem_free_exclusive,
)

H = 96  # real query heads (the MMA's N)
HW = 32  # heads per softmax group (4 warps, one per TMEM lane quarter)
NG = H // HW  # softmax groups
# Warpgroup-aligned roles (setmaxnreg is per warpgroup): WG0 = warp 0 (TMEM owner +
# Q / index loader), warp 1 (QK MMAs), warp 2 (PV MMAs), warp 3 (KV gathers); WG1-3
# = warps 4..15 (softmax and row sums of each group's heads); WG4-6 = warps 16..27
# (O^T correction and the epilogue of each group's heads, off the softmax path: the
# next row's softmax runs under a row's epilogue); WG7 = warps 28..31 (KV gathers).
NW = 16
SOFT0 = 4
CORR0 = 16
LOAD0 = 28
# Register pool: 64 per thread x 1024 threads = 65536 >= 640 x 40 (lean: TMEM / Q /
# MMA / gathers / corrections) + 384 x 104 (softmax).
LEAN_REGS, SOFT_REGS, CORR_REGS = 40, 80, 64
LOADERS = 1 + 4
# Per KV row: latent blocks 0-1 by TMA gather4 (4 rows per op); latent blocks 2-3 and
# the RoPE bytes by cp.async (16 B, LSU path) from every loader thread: the per-SM TMA
# op rate alone (~50 gather4 per us, 512 B each) held a 72 KB tile to ~2.5 us.
CPA_THREADS = 32 * LOADERS
NSOFT = 4 * NG  # softmax warps
# Split-K deal: the rows' 128-key tiles, concatenated in row order, are cut into
# ``grid`` contiguous ranges of Q = max(ceil(tiles / grid), MIN_Q) tiles; a CTA
# runs the row segments of its range in row order. A row cut by a range boundary
# is computed by several CTAs: its first segment (the "head", the last segment of
# its CTA) merges the partial (BF16 O, FP32 LSE) of each later segment (a "tail",
# the first segment of the next CTAs, written to a per-CTA workspace slot and
# released by a flag) as the DCP merge does (FP32 weights 2^(lse - LSE), one BF16
# rounding). Rows without keys: one segment each on CTA row % grid.
ROWS_MAX = 512  # rows per call
MAX_SEG = 48  # segments per CTA (<= Q + 1 + ceil(512 / grid))
MIN_Q = 4  # tiles per CTA at least: a head has at most ceil(15 / MIN_Q) tails
NT_MAX = 4  # tails per head
MB = 4  # merge pass: 16 B chunks per thread in flight
TAIL = -1  # segment role: -1 tail, 0 whole row, > 0 head (its tail count)
QBLK = H * 128  # one 128 B column block of the query rows [96 x 128 B]
QBYTES = 4 * QBLK + H * 64
PBYTES = TN * H  # P^T [128 keys x 96 heads] E4M3

# Barriers
NCORR = 4 * NG  # correction warps
B_QEMPTY = 1
B_PFULL = 2  # + P buffer (tile parity): P^T, the tile's rescale factors and flags
B_PVDONE = 4  # + P buffer: PV of that tile complete
B_SFULL = 6  # + buffer
B_SEMPTY = 8  # + buffer
B_IDXFULL = 10  # + row parity
B_IDXEMPTY = 12  # + row parity
B_KVFULL = 14  # + stage
B_KVEMPTY = 22  # + stage
B_PVBLK = 30  # + O^T block: that block's PV of the tile complete
# + O^T block (tiles of parity 0; parity 1 at B_OREADY1): corrected for that tile's
# PV. One barrier per tile parity: the correction of tile g needs the factors of tile g
# (published after PV(g - 2) completed), not PV(g - 1), so it can run one tile ahead of
# the PV issuer; with one barrier its second arrival would complete the phase the PV of
# tile g - 1 still waits for (parity aliasing: a hang, or a PV before its correction).
B_OREADY = 34
B_OEMPTY = 38  # + O^T block: read by the row's epilogue
B_ALPHA = 42  # + NG x tile parity + group: the tile's rescale factors published
B_LFULL = 48  # + group: the row's 1 / (448 l) published
B_LEMPTY = 51  # + group: the epilogue read them
B_QBLK = 54  # + query block (4 latent column blocks, RoPE): landed
B_OREADY1 = 59  # + O^T block
NBAR = 63

C_S, C_O = 0, 192  # TMEM: S^T 2 x 96 | O^T 4 x 96

# Split-K workspace (one per device and grid, ``workspace``): [grid] slots of a
# tail's BF16 latent [96, 512], then [grid, 96] FP32 LSE, then [grid] int32 flags
# (zero between launches: the tail's 12 correction warps add 1 each, the head's 12
# readers add 1 each and the last one re-arms it to 0).
WS_SLOT = H * D_NOPE * 2
GRID_MAX = 1024
WS_LSE = GRID_MAX * WS_SLOT
WS_FLAGS = WS_LSE + GRID_MAX * H * 4
WS_BYTES = WS_FLAGS + GRID_MAX * 4
_WORKSPACE: dict = {}


def workspace(device) -> torch.Tensor:
    """The split-K workspace of ``device`` (flags zeroed once at allocation)."""
    key = torch.device(device).index
    ws = _WORKSPACE.get(key)
    if ws is None:
        ws = torch.zeros(WS_BYTES, dtype=torch.uint8, device=device)
        _WORKSPACE[key] = ws
    return ws

# P^T: MN-major, no swizzle, core matrices of 8 keys x 16 heads (128 B); LBO = the K
# (key) direction core stride 128 B, SBO = the N (16-head chunk) stride 2048 B.
P_DESC = (128 >> 4 << 16) | (2048 >> 4 << 32) | (1 << 46)
# kind::f8f6f4: FP32 D, E4M3 A and B, M = 128, N = 96, 2x-K (bit 29); PV: A (V^T) and
# B (P^T) MN-major (bits 15, 16).
IDESC_QK = (1 << 4) | ((H >> 3) << 17) | ((128 >> 4) << 24) | (1 << 29)
IDESC_PV = IDESC_QK | (1 << 15) | (1 << 16)


@asm()
def tma_tile2d(dst, tmap, bar, col, row):
    code = (
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx"
        "::bytes [$0], [$1, {$3, $4}], [$2];"
    )
    return code, "r,l,r,r,r,~{memory}", (
        Uint32(dst), Int64(tmap), Uint32(bar), Int32(col), Int32(row))


@asm()
def mma_ss(d, a, b, idesc, acc):
    code = (
        "{ .reg .pred p; setp.ne.b32 p, $4, 0; "
        "tcgen05.mma.cta_group::1.kind::f8f6f4 [$0], $1, $2, $3, p; }"
    )
    return code, "r,l,l,r,r,~{memory}", (
        Uint32(d), Uint64(a), Uint64(b), Uint32(idesc), Uint32(acc))


@asm()
def tmem_ld48(addr):
    regs = ",".join(f"${i}" for i in range(48))
    code = (
        "{ .reg .b32 t; add.u32 t, $48, 32; "
        f"tcgen05.ld.sync.aligned.32x32b.x32.b32 {{{regs.split(',$32')[0]}}}, [$48]; "
        "tcgen05.ld.sync.aligned.32x32b.x16.b32 {"
        + ",".join(f"${i}" for i in range(32, 48))
        + "}, [t]; tcgen05.wait::ld.sync.aligned; }"
    )
    return code, ",".join(["=f"] * 48 + ["r", "~{memory}"]), (Uint32(addr),), (
        Float32,) * 48


@asm()
def tmem_st48(addr, vals):
    code = (
        "{ .reg .b32 t; add.u32 t, $0, 32; "
        "tcgen05.st.sync.aligned.32x32b.x32.b32 [$0], {"
        + ",".join(f"${i + 1}" for i in range(32))
        + "}; tcgen05.st.sync.aligned.32x32b.x16.b32 [t], {"
        + ",".join(f"${i + 33}" for i in range(16))
        + "}; tcgen05.wait::st.sync.aligned; }"
    )
    return code, "r" + ",f" * 48 + ",~{memory}", (
        Uint32(addr), *[Float32(v) for v in vals])


@asm()
def tmem_ld16(addr):
    regs = ",".join(f"${i}" for i in range(16))
    code = (f"tcgen05.ld.sync.aligned.32x32b.x16.b32 {{{regs}}}, [$16]; "
            "tcgen05.wait::ld.sync.aligned;")
    return code, ",".join(["=f"] * 16 + ["r", "~{memory}"]), (Uint32(addr),), (
        Float32,) * 16


@asm()
def tmem_st16(addr, vals):
    regs = ",".join(f"${i + 1}" for i in range(16))
    code = (f"tcgen05.st.sync.aligned.32x32b.x16.b32 [$0], {{{regs}}}; "
            "tcgen05.wait::st.sync.aligned;")
    return code, "r" + ",f" * 16 + ",~{memory}", (
        Uint32(addr), *[Float32(v) for v in vals])


@asm(Float32, pure=True)
def redux_max(v):
    return "redux.sync.max.f32 $0, $1, 0xffffffff;", "=f,f", (Float32(v),)


@asm(Float32, pure=True)
def shfl_xor(v, m):
    return "shfl.sync.bfly.b32 $0, $1, $2, 0x1f, 0xffffffff;", "=f,f,r", (
        Float32(v), Uint32(m))


@asm()
def st_shared_v4(addr, w0, w1, w2, w3):
    return "st.shared.v4.b32 [$0], {$1,$2,$3,$4};", "r,r,r,r,r,~{memory}", (
        Uint32(addr), Uint32(w0), Uint32(w1), Uint32(w2), Uint32(w3))


@asm()
def st_global_bf16(ptr, v):
    return ("{ .reg .b16 t; cvt.rn.bf16.f32 t, $1; st.global.b16 [$0], t; }",
            "l,f,~{memory}", (Int64(ptr), Float32(v)))


@asm()
def st_global_bf16x2(ptr, lo, hi):
    return ("{ .reg .b32 t; cvt.rn.bf16x2.f32 t, $2, $1; st.global.b32 [$0], t; }",
            "l,f,f,~{memory}", (Int64(ptr), Float32(lo), Float32(hi)))


@asm()
def cp_async16(dst, src, nbytes):
    return ("cp.async.cg.shared.global [$0], [$1], 16, $2;", "r,l,r,~{memory}",
            (Uint32(dst), Int64(src), Uint32(nbytes)))


@asm()
def cp_async_arrive(bar):
    return ("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [$0];", "r,~{memory}",
            (Uint32(bar),))


@asm()
def prefetch_l2(ptr):
    return "prefetch.global.L2 [$0];", "l,~{memory}", (Int64(ptr),)


@asm()
def fence_async_smem():
    return "fence.proxy.async.shared::cta;", "~{memory}", ()


class SparseMlaDecodeKM:
    """``grid`` persistent CTAs (one per SM); ``stages`` KV tiles in flight;
    ``loaders`` gather4 warps."""

    def __init__(self, grid: int, stages: int = 3, loaders: int = 6,
                 stamps: bool = False, cpa: int = 0, prefetch: int = 3,
                 gather_only: bool = False, epi_store: bool = True,
                 no_gather: bool = False):
        assert 2 <= stages <= 8 and loaders == LOADERS
        self.grid, self.stages, self.loaders, self.stamps = grid, stages, loaders, stamps
        # cpa > 0: latent blocks 0 .. cpa - 1 by TMA, the rest of the row by cp.async
        self.cpa, self.prefetch = cpa, prefetch
        self.gather_only = gather_only
        # bench knob: KV stages complete without loading (the MMA / softmax chain on
        # stale stage bytes: the attention time if the gather were free)
        self.no_gather = no_gather
        self.epi_store = epi_store  # bench knob: False drops the epilogue's stores
        self.tb = cpa if cpa else 4
        self.cpa_chunks = (4 - self.tb) * 8 + 4  # 16 B chunks per row by cp.async
        self.cpa_per = (TN * self.cpa_chunks + CPA_THREADS - 1) // CPA_THREADS
        self.threads = 32 * (NW + 16)
        off = 0
        self.off_kv = off
        off += stages * STAGE
        self.off_q = off
        off += QBYTES
        self.off_p = off  # 2 x P^T (tile parity)
        off += 2 * PBYTES
        self.off_idx = off
        off += 2 * IDXB
        self.off_tmax = off  # [3 groups][4 quarters][32] FP32 tile maxima / sums
        off += NG * 4 * HW * 4
        self.off_bias = off  # [96] m - log2(448): P = 2^(x - bias) = 448 x 2^(x - m)
        off += H * 4
        self.off_alpha = off  # [2][96] 2^(m_old - m_new) of a tile (tile parity)
        off += 2 * H * 4
        self.off_flag = off  # [2][3] a head's max grew in the tile (tile parity)
        off += 32
        self.off_l = off  # [96] final row sums
        off += H * 4
        self.off_n = off  # [MAX_SEG] keys of this CTA's segments (``deal``)
        off += 4 * MAX_SEG
        self.off_row = off  # [MAX_SEG] their row ids
        off += 4 * MAX_SEG
        self.off_k0 = off  # [MAX_SEG] their first key in the row
        off += 4 * MAX_SEG
        self.off_role = off  # [MAX_SEG] TAIL, 0 (whole row) or the head's tail count
        off += 4 * MAX_SEG
        self.off_nseg = off  # this CTA's segment count
        off += 16
        self.off_w = off  # [NT_MAX + 1][96] a head's merge weights: its tails, then own
        off += 4 * (NT_MAX + 1) * H
        self.off_bar = off
        off += NBAR * 8
        self.off_tmem = off
        self.smem = off + 16

    def bar(self, base, i):
        return base + Uint32(self.off_bar + 8 * i)

    @cute.jit
    def nrow(self, base, i):
        return Int32(ld_shared_u32(base + Uint32(self.off_n) + Uint32(4) * Uint32(i)))

    @cute.jit
    def rowid(self, base, i):
        return Int32(ld_shared_u32(base + Uint32(self.off_row) + Uint32(4) * Uint32(i)))

    @cute.jit
    def k0(self, base, i):
        return Int32(ld_shared_u32(base + Uint32(self.off_k0) + Uint32(4) * Uint32(i)))

    @cute.jit
    def role(self, base, i):
        return Int32(ld_shared_u32(base + Uint32(self.off_role) + Uint32(4) * Uint32(i)))

    @cute.jit
    def put(self, base, idx, row, n, k0, role):
        st_shared_u32(base + Uint32(self.off_row) + Uint32(4) * Uint32(idx), Uint32(row))
        st_shared_u32(base + Uint32(self.off_n) + Uint32(4) * Uint32(idx), Uint32(n))
        st_shared_u32(base + Uint32(self.off_k0) + Uint32(4) * Uint32(idx), Uint32(k0))
        st_shared_u32(base + Uint32(self.off_role) + Uint32(4) * Uint32(idx),
                      Uint32(role))

    @cute.jit
    def deal(self, base, lane, cta, rows, valid):
        """Warp 1: this CTA's row segments into SMEM (off_row / off_n / off_k0 /
        off_role, count at off_nseg): the tiles of all rows in row order cut into
        ``grid`` contiguous ranges of Q tiles (split-K, see MAX_SEG)."""
        lt = lanemask_lt()
        # Every row count in one batch of loads (rows <= 512: 16 per lane), not one
        # dependent L2 round trip per 32 rows and pass.
        ns = []
        for c in cutlass.range_constexpr(ROWS_MAX // 32):
            r = c * 32 + lane
            n = Int32(0)
            if r < rows:
                n = ld_global_s32(valid + Int64(r) * Int64(4))
            ns.append(n)
        tot = Int32(0)
        for c in cutlass.range_constexpr(ROWS_MAX // 32):
            tot = tot + Int32(_tiles(ns[c]))
        for sh in cutlass.range_constexpr(5):
            tot = tot + shfl_xor_i(tot, Uint32(1 << sh))
        q = (tot + Int32(self.grid - 1)) // Int32(self.grid)
        if q < MIN_Q:
            q = Int32(MIN_Q)
        lo = cta * q
        hi = lo + q
        if hi > tot:
            hi = tot
        cnt = Int32(0)
        pre = Int32(0)  # tiles of the rows before this step
        for c in cutlass.range_constexpr(ROWS_MAX // 32):
            r = c * 32 + lane
            n = ns[c]
            t = Int32(_tiles(n))
            inc = t
            for sh in cutlass.range_constexpr(5):
                up = shfl_up_i(inc, Uint32(1 << sh))
                if lane >= (1 << sh):
                    inc = inc + up
            p = pre + inc - t
            s_lo = p
            if s_lo < lo:
                s_lo = lo
            s_hi = p + t
            if s_hi > hi:
                s_hi = hi
            has = s_lo < s_hi
            bal = ballot(cutlass.select_(has, Uint32(1), Uint32(0)))
            if has:
                k0 = (s_lo - p) * TN
                nk = (s_hi - p) * TN
                if nk > n:
                    nk = n
                role = Int32(0)
                if s_lo > p:
                    role = Int32(TAIL)
                elif p + t > hi:
                    role = (p + t - hi + q - 1) // q
                self.put(base, cnt + Int32(popc(bal & lt)), r, nk - k0, k0, role)
            cnt = cnt + Int32(popc(bal))
            pre = pre + shfl_idx_i(inc, Uint32(31))
        for c in cutlass.range_constexpr(ROWS_MAX // 32):  # rows without keys: LSE -inf
            r = c * 32 + lane  # and a zero latent
            z = Int32(0)
            if r < rows:
                if ns[c] <= 0:
                    if r % self.grid == cta:
                        z = Int32(1)
            bal = ballot(Uint32(z))
            if z == 1:
                self.put(base, cnt + Int32(popc(bal & lt)), r, 0, 0, 0)
            cnt = cnt + Int32(popc(bal))
        if lane == 0:
            st_shared_u32(base + Uint32(self.off_nseg), Uint32(cnt))
        sync_warp()

    @cute.jit
    def mark(self, stamp, bid, i):
        if cutlass.const_expr(self.stamps):
            st_global_u64(stamp + (Int64(bid) * Int64(64) + Int64(i)) * Int64(8),
                          globaltimer())

    @cute.jit
    def __call__(self, q: Int64, q_heads: Int32, rows: Int32, kv: Int64,
                 kv_count: Int32, slots: Int64, slot_stride: Int32, valid: Int64,
                 out: Int64, out_stride: Int32, lse: Int64, lse_stride: Int32,
                 scale_log2: Float32, out_scale: Float32, stamp: Int64,
                 ws: Int64, stream: cuda.CUstream):
        u8 = cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8
        l2 = cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B
        sw = cuda.CUtensorMapSwizzle
        kvm = tensormap.encode_map(
            kv, u8, (ROW, kv_count), (ROW,), (128, 1), sw.CU_TENSOR_MAP_SWIZZLE_128B, l2)
        kvr = tensormap.encode_map(
            kv, u8, (ROW, kv_count), (ROW,), (64, 1), sw.CU_TENSOR_MAP_SWIZZLE_64B, l2)
        qn = tensormap.encode_map(
            q, u8, (D_NOPE, rows * q_heads), (ROW,), (128, H),
            sw.CU_TENSOR_MAP_SWIZZLE_128B, l2)
        qr = tensormap.encode_map(
            q + D_NOPE, u8, (64, rows * q_heads), (ROW,), (64, H),
            sw.CU_TENSOR_MAP_SWIZZLE_64B, l2)
        self.kernel(kvm, kvr, qn, qr, kv, q, q_heads, rows, slots, slot_stride, valid, out,
                    out_stride, lse, lse_stride, scale_log2, out_scale, stamp, ws).launch(
            grid=(self.grid, 1, 1), block=(self.threads, 1, 1), smem=self.smem,
            stream=stream)

    @cute.kernel
    def kernel(self, kvm: tensormap.GridMap, kvr: tensormap.GridMap,
               qn: tensormap.GridMap, qr: tensormap.GridMap, kv: Int64, qp: Int64,
               q_heads: Int32,
               rows: Int32, slots: Int64, slot_stride: Int32, valid: Int64, out: Int64,
               out_stride: Int32, lse: Int64, lse_stride: Int32, scale_log2: Float32,
               out_scale: Float32, stamp: Int64, ws: Int64):
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(tid // 32)
        lane = tid % 32
        smem = cutlass.utils.SmemAllocator()
        raw = smem.allocate_array(Uint8, self.smem, byte_alignment=1024)
        base = Uint32(raw.toint())
        if tid == 0:
            self.mark(stamp, cta, 0)
            mbar_init(self.bar(base, B_QEMPTY), Uint32(1))
            for j in cutlass.range_constexpr(5):
                mbar_init(self.bar(base, B_QBLK + j), Uint32(1))
            for blk in cutlass.range_constexpr(4):
                mbar_init(self.bar(base, B_PVBLK + blk), Uint32(1))
                mbar_init(self.bar(base, B_OREADY + blk), Uint32(NCORR))
                mbar_init(self.bar(base, B_OREADY1 + blk), Uint32(NCORR))
                mbar_init(self.bar(base, B_OEMPTY + blk), Uint32(NCORR))
            for j in cutlass.range_constexpr(2 * NG):
                mbar_init(self.bar(base, B_ALPHA + j), Uint32(32))
            for j in cutlass.range_constexpr(NG):
                mbar_init(self.bar(base, B_LFULL + j), Uint32(32))
                mbar_init(self.bar(base, B_LEMPTY + j), Uint32(4))
            for b in cutlass.range_constexpr(2):
                mbar_init(self.bar(base, B_PFULL + b), Uint32(NSOFT))
                mbar_init(self.bar(base, B_PVDONE + b), Uint32(1))
                mbar_init(self.bar(base, B_SFULL + b), Uint32(1))
                mbar_init(self.bar(base, B_SEMPTY + b), Uint32(NSOFT))
                mbar_init(self.bar(base, B_IDXFULL + b), Uint32(1))
                mbar_init(self.bar(base, B_IDXEMPTY + b), Uint32(self.loaders))
            for s in cutlass.range_constexpr(self.stages):
                mbar_init(self.bar(base, B_KVFULL + s),
                          Uint32(1 + CPA_THREADS if self.cpa else 1))
                mbar_init(self.bar(base, B_KVEMPTY + s), Uint32(1))
            fence_mbar_init()
        if warp == 0:
            tmem_alloc_exclusive(base + Uint32(self.off_tmem))
        if warp == 1:
            self.deal(base, lane, cta, rows, valid)
        tc_fence_before()
        cute.arch.sync_threads()
        tc_fence_after()
        n_mine = Int32(ld_shared_u32(base + Uint32(self.off_nseg)))
        if tid == 0:
            self.mark(stamp, cta, 1)
        taddr = ld_shared_u32(base + Uint32(self.off_tmem))
        if (warp >= SOFT0) & (warp < NW):
            cute.arch.warpgroup_reg_alloc(SOFT_REGS)
            if cutlass.const_expr(not self.gather_only):
                self.softmax(base, warp, lane, taddr, cta, n_mine, out, out_stride, lse,
                             lse_stride, scale_log2, out_scale, stamp, ws)
        elif (warp >= CORR0) & (warp < LOAD0):
            # Its own register budget in a region only the correction warps enter
            # (ptxas sizes a region by the smallest setmaxnreg that may reach it).
            cute.arch.warpgroup_reg_dealloc(CORR_REGS)
            if cutlass.const_expr(not self.gather_only):
                self.correct(base, warp, lane, taddr, cta, n_mine, out, out_stride,
                             stamp, ws)
        else:
            cute.arch.warpgroup_reg_dealloc(LEAN_REGS)
            if warp == 0:
                if lane == 0:
                    self.qloader(base, cta, n_mine, q_heads, tensormap.map_address(qn),
                                 tensormap.map_address(qr), slots, slot_stride)
            if warp == 1:
                if lane == 0:
                    if cutlass.const_expr(self.gather_only):
                        self.drain(base, n_mine, stamp, cta)
                    else:
                        self.mma(base, taddr, n_mine, stamp, cta)
            if warp == 2:
                if lane == 0:
                    if cutlass.const_expr(not self.gather_only):
                        self.pv(base, taddr, n_mine, stamp, cta)
            if (warp == 3) | (warp >= LOAD0):
                lw = cutlass.select_(warp >= LOAD0, warp - LOAD0 + 1, 0)
                self.loader(base, lw, lane, cta, n_mine, tensormap.map_address(kvm),
                            tensormap.map_address(kvr), kv, qp, q_heads)
        tc_fence_before()
        cute.arch.sync_threads()
        if warp == 0:
            tc_fence_after()
            tmem_free_exclusive(taddr)

    # ------------------------------------------------------------------ loaders
    @cute.jit
    def qloader(self, base, cta, n_mine, q_heads, tm_qn, tm_qr, slots, slot_stride):
        """Per non-empty row: its slot list into the two-row index ring (ahead of the
        gathers), then its 96 query heads (B of QK) once the previous row's last QK
        is complete."""
        rq = Int32(0)
        for i in range(n_mine):
            n = self.nrow(base, i)
            if n > 0:
                row = self.rowid(base, i)
                b = rq & 1
                mbar_wait(self.bar(base, B_IDXEMPTY + b), ((rq >> 1) & 1) ^ 1)
                nbytes = Uint32(((n + 3) // 4) * 16)
                mbar_arrive_tx(self.bar(base, B_IDXFULL + b), nbytes)
                bulk_load(base + Uint32(self.off_idx) + Uint32(b) * Uint32(IDXB),
                          slots + (Int64(row) * Int64(slot_stride)
                                   + Int64(self.k0(base, i))) * Int64(4), nbytes,
                          self.bar(base, B_IDXFULL + b))
                if cutlass.const_expr(not self.gather_only):
                    mbar_wait(self.bar(base, B_QEMPTY), (rq & 1) ^ 1)
                    # Block by block: the row's first QK starts on the first block.
                    for j in cutlass.range_constexpr(4):
                        full = self.bar(base, B_QBLK + j)
                        mbar_arrive_tx(full, Uint32(QBLK))
                        tma_tile2d(base + Uint32(self.off_q + j * QBLK), tm_qn, full,
                                   Int32(128 * j), row * q_heads)
                    full = self.bar(base, B_QBLK + 4)
                    mbar_arrive_tx(full, Uint32(H * 64))
                    tma_tile2d(base + Uint32(self.off_q + 4 * QBLK), tm_qr, full, Int32(0),
                               row * q_heads)
                rq = rq + 1

    @cute.jit
    def prefetch_keys(self, srow, n, k0, me, kv):
        """L2 prefetch of the 128-key tile from key k0 of a row (slot list at
        ``srow``, n keys): five 128 B lines cover any 576 B row."""
        k = k0 + me
        if (me < TN) & (k < n):
            a = kv + Int64(Int32(ld_shared_u32(srow + Uint32(k) * Uint32(4)))) * Int64(ROW)
            for j in cutlass.range_constexpr(5):
                prefetch_l2(a + Int64(128 * j))

    @cute.jit
    def loader(self, base, lw, lane, cta, n_mine, tm_kv, tm_kr, kv, qp, q_heads):
        """Loader warp lw, lane m: 4-row quad lw + L m of every tile; warp 0 arms.
        With ``prefetch``: a row's tiles 1.. and the next row's keys are prefetched
        into L2 once their slot lists are in SMEM."""
        per: cutlass.Constexpr = (32 + self.loaders - 1) // self.loaders
        me = lw * 32 + lane
        g = Int32(0)
        rk = Int32(0)
        for i in range(n_mine):
            n = self.nrow(base, i)
            if n > 0:
                tiles = _tiles(n)
                b = rk & 1
                mbar_wait(self.bar(base, B_IDXFULL + b), (rk >> 1) & 1)
                srow = base + Uint32(self.off_idx) + Uint32(b) * Uint32(IDXB)
                if (i + 1 < n_mine) & (me < QBYTES // 128):
                    # The next row's query heads into L2: its TMA load (after this
                    # row's last QK) then waits on L2, not HBM.
                    prefetch_l2(qp + (Int64(self.rowid(base, i + 1)) * Int64(q_heads)
                                      * Int64(ROW) + Int64(me * 128)))
                for t in range(tiles):
                    s = g % self.stages
                    st = base + Uint32(self.off_kv) + Uint32(s) * Uint32(STAGE)
                    full = self.bar(base, B_KVFULL + s)
                    if lane == 0:
                        mbar_wait(self.bar(base, B_KVEMPTY + s),
                                  ((g // self.stages) & 1) ^ 1)
                        if cutlass.const_expr(self.no_gather):
                            if lw == 0:
                                mbar_arrive(full)
                        else:
                            if lw == 0:
                                if cutlass.const_expr(self.cpa):
                                    mbar_arrive_tx(full, Uint32(self.tb * TN * 128))
                                else:
                                    mbar_arrive_tx(full, Uint32(STAGE))
                    cute.arch.sync_warp()
                    quad = lw + self.loaders * lane
                    if (lane < per) & (quad < (0 if self.no_gather else 32)):
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
                        if cutlass.const_expr(self.cpa):
                            for a in cutlass.range_constexpr(self.tb):
                                gather4(st + Uint32(a * TN * 128) + Uint32(quad) * 512,
                                        tm_kv, full, 128 * a, r0, r1, r2, r3)
                        else:
                            for a in cutlass.range_constexpr(4):
                                gather4(st + Uint32(a * TN * 128) + Uint32(quad) * 512,
                                        tm_kv, full, 128 * a, r0, r1, r2, r3)
                            gather4(st + Uint32(4 * TN * 128) + Uint32(quad) * 256, tm_kr,
                                    full, D_NOPE, r0, r1, r2, r3)
                    if cutlass.const_expr(self.cpa):
                        # The rest of each row by cp.async into the same swizzled
                        # layout (SW128 latent blocks: 16 B chunk c of row r at
                        # c ^ (r % 8); SW64 RoPE: chunk p at p ^ ((r / 2) % 4)); keys
                        # past n zero-filled.
                        for j in cutlass.range_constexpr(self.cpa_per):
                            idx = me + j * CPA_THREADS
                            if idx < TN * self.cpa_chunks:
                                r = idx // self.cpa_chunks
                                c = idx - r * self.cpa_chunks
                                key = t * TN + r
                                ok = key < n
                                slot = Int32(ld_shared_u32(srow + Uint32(key) * Uint32(4)))
                                slot = cutlass.select_(ok, slot, Int32(0))
                                nb = cutlass.select_(ok, Uint32(16), Uint32(0))
                                blk = self.tb + (c >> 3)
                                dst = st + Uint32(blk * TN * 128 + r * 128
                                                  + (((c & 7) ^ (r & 7)) << 4))
                                off = self.tb * 128 + c * 16
                                if c >= (4 - self.tb) * 8:
                                    p = c - (4 - self.tb) * 8
                                    dst = st + Uint32(4 * TN * 128 + r * 64
                                                      + ((p ^ ((r >> 1) & 3)) << 4))
                                    off = D_NOPE + p * 16
                                cp_async16(dst, kv + Int64(slot) * Int64(ROW) + Int64(off),
                                           nb)
                        cp_async_arrive(full)
                    if cutlass.const_expr(self.prefetch > 0):
                        # Rolling L2 prefetch of the tile ``prefetch`` tiles ahead
                        # (this row's, else the next row's once its slot list landed)
                        ahead = t + self.prefetch
                        if ahead < tiles:
                            self.prefetch_keys(srow, n, ahead * TN, me, kv)
                        elif i + 1 < n_mine:
                            n1 = self.nrow(base, i + 1)
                            if (n1 > 0) & ((ahead - tiles) * TN < n1):
                                b1 = (rk + 1) & 1
                                mbar_wait(self.bar(base, B_IDXFULL + b1),
                                          ((rk + 1) >> 1) & 1)
                                self.prefetch_keys(
                                    base + Uint32(self.off_idx) + Uint32(b1)
                                    * Uint32(IDXB), n1, (ahead - tiles) * TN, me, kv)
                    g = g + 1
                cute.arch.sync_warp()
                if lane == 0:
                    mbar_arrive(self.bar(base, B_IDXEMPTY + b))
                rk = rk + 1

    # ------------------------------------------------------------------ MMA
    @cute.jit
    def qk(self, base, taddr, g, stamp, cta, mark, last, first, ro):
        s = g % self.stages
        st = base + Uint32(self.off_kv) + Uint32(s) * Uint32(STAGE)
        b = g & 1
        mbar_wait(self.bar(base, B_KVFULL + s), (g // self.stages) & 1)
        fence_async_smem()  # cp.async (generic proxy) writes -> tcgen05.mma reads
        if mark < 8:
            self.mark(stamp, cta, 48 + mark)
        mbar_wait(self.bar(base, B_SEMPTY + b), ((g >> 1) & 1) ^ 1)
        tc_fence_after()
        d = taddr + Uint32(C_S) + Uint32(96) * Uint32(b)
        for kk in cutlass.range_constexpr(8):
            if cutlass.const_expr(kk % 2 == 0):
                if first:
                    mbar_wait(self.bar(base, B_QBLK + kk // 2), ro & 1)
            off = (kk // 2) * TN * 128 + (kk % 2) * 64
            ad = Uint64(SDESC_SW128) | Uint64((st + Uint32(off)) >> Uint32(4))
            qoff = self.off_q + (kk // 2) * QBLK + (kk % 2) * 64
            bd = Uint64(SDESC_SW128) | Uint64((base + Uint32(qoff)) >> Uint32(4))
            mma_ss(d, ad, bd, Uint32(IDESC_QK), Uint32(1 if kk > 0 else 0))
        if first:
            mbar_wait(self.bar(base, B_QBLK + 4), ro & 1)
        ad = Uint64(SDESC_SW64) | Uint64((st + Uint32(4 * TN * 128)) >> Uint32(4))
        bd = Uint64(SDESC_SW64) | Uint64((base + Uint32(self.off_q + 4 * QBLK)) >> Uint32(4))
        mma_ss(d, ad, bd, Uint32(IDESC_QK), Uint32(1))
        tc_commit(self.bar(base, B_SFULL + b))
        if last:  # the row's last QK: Q is free once it completes
            tc_commit(self.bar(base, B_QEMPTY))

    @cute.jit
    def drain(self, base, n_mine, stamp, cta):
        """Benchmark knob ``gather_only``: release every KV tile on landing (the
        gather rate of this kernel's loaders without the attention chain)."""
        g = Int32(0)
        for i in range(n_mine):
            n = self.nrow(base, i)
            if n > 0:
                for t in range(_tiles(n)):
                    s = g % self.stages
                    mbar_wait(self.bar(base, B_KVFULL + s), (g // self.stages) & 1)
                    if g < 8:
                        self.mark(stamp, cta, 48 + g)
                    mbar_arrive(self.bar(base, B_KVEMPTY + s))
                    g = g + 1

    @cute.jit
    def mma(self, base, taddr, n_mine, stamp, cta):
        """The QK issuer: every tile's S^T into the free one of two TMEM buffers."""
        g = Int32(0)
        ro = Int32(0)
        for i in range(n_mine):
            n = self.nrow(base, i)
            if n > 0:
                tiles = _tiles(n)
                for t in range(tiles):
                    self.qk(base, taddr, g, stamp, cta, ro * 8 + t, t + 1 == tiles,
                            t == 0, ro)
                    g = g + 1
                ro = ro + 1

    @cute.jit
    def pv(self, base, taddr, n_mine, stamp, cta):
        """The PV issuer (its own warp: a PV never waits behind the next tile's QK,
        whose KV may still be in flight)."""
        g = Int32(0)
        gc0 = Int32(0)  # corrected tiles (every tile but a row's first) of parity 0
        gc1 = Int32(0)  # ... of parity 1
        ro = Int32(0)
        for i in range(n_mine):
            n = self.nrow(base, i)
            if n > 0:
                tiles = _tiles(n)
                for t in range(tiles):
                    b = g & 1
                    mbar_wait(self.bar(base, B_PFULL + b), (g >> 1) & 1)
                    if ro * 8 + t < 8:
                        self.mark(stamp, cta, 56 + t)
                    tc_fence_after()
                    s = g % self.stages
                    st = base + Uint32(self.off_kv) + Uint32(s) * Uint32(STAGE)
                    pdesc = Uint64(P_DESC) | Uint64(
                        (base + Uint32(self.off_p) + Uint32(b) * Uint32(PBYTES))
                        >> Uint32(4))
                    # Block by block: the correction groups rescale O^T block blk (by
                    # this tile's factors, after the previous PV of blk) while the
                    # tensor pipe runs the other blocks; a row's first PV of a block
                    # follows the previous row's epilogue read of it.
                    for blk in cutlass.range_constexpr(4):
                        if t > 0:
                            if b == 0:
                                mbar_wait(self.bar(base, B_OREADY + blk), gc0 & 1)
                            else:
                                mbar_wait(self.bar(base, B_OREADY1 + blk), gc1 & 1)
                            tc_fence_after()
                        else:
                            mbar_wait(self.bar(base, B_OEMPTY + blk), (ro & 1) ^ 1)
                            tc_fence_after()
                        vd = Uint64(SDESC_SW128) | Uint64(
                            (st + Uint32(blk * TN * 128)) >> Uint32(4))
                        for kk in cutlass.range_constexpr(2):
                            acc = Uint32(1)
                            if t == 0:
                                acc = Uint32(kk)
                            mma_ss(taddr + Uint32(C_O + 96 * blk), vd + Uint64(kk * 512),
                                   pdesc + Uint64(kk * 64), Uint32(IDESC_PV), acc)
                        tc_commit(self.bar(base, B_PVBLK + blk))
                    tc_commit(self.bar(base, B_KVEMPTY + s))
                    tc_commit(self.bar(base, B_PVDONE + b))
                    if t > 0:
                        if b == 0:
                            gc0 = gc0 + 1
                        else:
                            gc1 = gc1 + 1
                    g = g + 1
                ro = ro + 1

    # ------------------------------------------------------------------ correction
    @cute.jit
    def correct(self, base, warp, lane, taddr, cta, n_mine, out, out_stride, stamp, ws):
        """Group k (warps CORR0 + 4k ..) owns the O^T columns of heads 32k .. 32k + 31;
        its warp of TMEM lane quarter q owns dims 32q .. 32q + 31 of every 128-dim
        block. Per tile after a row's first: once the softmax group published the
        tile's factors, O^T block by block x 2^(m_old - m_new) where a head's max
        grew (after the previous PV of the block). Per row: the epilogue O / (448 l) x
        out_scale -> BF16 while the softmax group goes on with the next row."""
        k = (warp - CORR0) // 4
        quarter = warp % 4
        lanes = Uint32(quarter * 32) << Uint32(16)
        h0 = k * HW
        alpha_g = base + Uint32(self.off_alpha) + Uint32(4 * h0)
        flag_g = base + Uint32(self.off_flag) + Uint32(4 * k)
        fin = base + Uint32(self.off_l) + Uint32(4 * h0)
        ocol = taddr + lanes + Uint32(C_O + HW * k)
        odd = lane & 1
        g = Int32(0)
        ro = Int32(0)
        for i in range(n_mine):
            n = self.nrow(base, i)
            if n > 0:
                row = self.rowid(base, i)
                tiles = _tiles(n)
                for t in range(tiles):
                    if t > 0:
                        b = g & 1
                        mbar_wait(self.bar(base, B_ALPHA + NG * b + k), (g >> 1) & 1)
                        alpha_s = alpha_g + Uint32(b) * Uint32(H * 4)
                        grew = ld_shared_u32(flag_g + Uint32(b) * Uint32(16)) != Uint32(0)
                        for blk in cutlass.range_constexpr(4):
                            if grew:
                                # PVs of a block run in order: an unchanged block needs
                                # no wait for the previous one.
                                mbar_wait(self.bar(base, B_PVBLK + blk), (g - 1) & 1)
                                tc_fence_after()
                                for j in cutlass.range_constexpr(2):
                                    col = ocol + Uint32(96 * blk + 16 * j)
                                    ov = tmem_ld16(col)
                                    vals = []
                                    for c in cutlass.range_constexpr(4):
                                        a4 = ld_shared_v4f(alpha_s + Uint32(64 * j + 16 * c))
                                        vals.extend([ov[4 * c + e] * a4[e] for e in range(4)])
                                    tmem_st16_nw(col, vals)
                                tmem_wait_st()
                                tc_fence_before()
                            cute.arch.sync_warp()
                            if lane == 0:
                                mbar_arrive(self.bar(base, cutlass.select_(
                                    b == 0, B_OREADY, B_OREADY1) + blk))
                    g = g + 1
                # This group's 32 heads of the latent once the row's last PV is
                # complete: thread = dim 32 quarter + lane of every 128-dim block, each
                # O^T block released after its read. Lane pairs swap one value so each
                # lane stores two adjacent dims of one head (BF16x2): even lanes head h,
                # odd lanes head h + 1.
                mbar_wait(self.bar(base, B_LFULL + k), ro & 1)
                gl = g - 1
                mbar_wait(self.bar(base, B_PVDONE + (gl & 1)), (gl >> 1) & 1)
                tc_fence_after()
                role = self.role(base, i)
                me = Int64(quarter * 32 + lane - odd) * Int64(2)
                dst = out + Int64(row) * Int64(out_stride) * Int64(2) + me
                if role == TAIL:  # this CTA's workspace slot
                    dst = ws + Int64(cta) * Int64(WS_SLOT) + me
                for blk in cutlass.range_constexpr(4):
                    for j in cutlass.range_constexpr(2):
                        ov = tmem_ld16(ocol + Uint32(96 * blk + 16 * j))
                        if j == 1:
                            tc_fence_before()
                            cute.arch.sync_warp()
                            if lane == 0:
                                mbar_arrive(self.bar(base, B_OEMPTY + blk))
                        # Each pair stored as soon as formed (40-register warps: no
                        # 16-value staging).
                        for e in cutlass.range_constexpr(4):
                            i4 = ld_shared_v4f(fin + Uint32(64 * j + 16 * e))
                            for f in cutlass.range_constexpr(2):
                                h = 4 * e + 2 * f
                                a = ov[h] * i4[2 * f]
                                b2 = ov[h + 1] * i4[2 * f + 1]
                                x = shfl_xor(cutlass.select_(odd == 1, a, b2), 1)
                                if cutlass.const_expr(self.epi_store):
                                    st_global_bf16x2(
                                        dst + Int64(((h0 + 16 * j + h + odd) * D_NOPE
                                                     + 128 * blk) * 2),
                                        cutlass.select_(odd == 1, x, a),
                                        cutlass.select_(odd == 1, b2, x))
                if role == TAIL:  # release this slot: the partial latent is stored
                    fence_gpu()
                cute.arch.sync_warp()
                if lane == 0:
                    mbar_arrive(self.bar(base, B_LEMPTY + k))
                    if role == TAIL:
                        red_release_add(ws + Int64(WS_FLAGS) + Int64(4) * Int64(cta), 1)
                if (warp == CORR0) & (lane == 0) & (ro < 2):
                    self.mark(stamp, cta, 14 + 16 * ro)
                ro = ro + 1

    @cute.jit
    def tail_lse(self, ws, slot, head):
        return ld_relaxed_f32(ws + Int64(WS_LSE) + (Int64(slot) * Int64(H) + Int64(head))
                              * Int64(4))

    @cute.jit
    def merge(self, base, wt, h0, row, role, cta, out, out_stride, ws):
        """Thread wt of a softmax group: 8-dim chunks wt, wt + 128, .. of the group's
        32 heads x 512 dims: own BF16 partial x its weight + each tail's BF16 partial x
        its weight in FP32 -> BF16 latent."""
        wts = base + Uint32(self.off_w)
        dst = out + Int64(row) * Int64(out_stride) * Int64(2) + Int64(h0 * D_NOPE * 2)
        # Batches of MB chunks: every load of a batch in flight before its first use
        # (L2 latency, not bandwidth, bounds a dependent load chain).
        for vb in cutlass.range_constexpr(HW * D_NOPE // 8 // 128 // MB):
            cs = [(vb * MB + v) * 128 + wt for v in range(MB)]  # 8-dim chunks
            hs = [c // (D_NOPE // 8) for c in cs]
            own = [ld_cg_v4(dst + Int64(c * 16)) for c in cs]
            acc = []
            for v in cutlass.range_constexpr(MB):
                w0 = ld_shared_f32(wts + Uint32(4) * Uint32(NT_MAX * H + h0 + hs[v]))
                for e in cutlass.range_constexpr(4):
                    lo, hi = unpack_bf16(own[v][e])
                    acc.extend([w0 * lo, w0 * hi])
            for jt in range(role):
                src = ws + Int64(cta + 1 + jt) * Int64(WS_SLOT) + Int64(h0 * D_NOPE * 2)
                tws = [ld_cg_v4(src + Int64(c * 16)) for c in cs]
                for v in cutlass.range_constexpr(MB):
                    wj = ld_shared_f32(wts + Uint32(4) * Uint32(jt * H + h0 + hs[v]))
                    for e in cutlass.range_constexpr(4):
                        lo, hi = unpack_bf16(tws[v][e])
                        acc[8 * v + 2 * e] = acc[8 * v + 2 * e] + wj * lo
                        acc[8 * v + 2 * e + 1] = acc[8 * v + 2 * e + 1] + wj * hi
            for v in cutlass.range_constexpr(MB):
                a8 = acc[8 * v: 8 * v + 8]
                st_global_v4(dst + Int64(cs[v] * 16), pack_bf16x2(a8[0], a8[1]),
                             pack_bf16x2(a8[2], a8[3]), pack_bf16x2(a8[4], a8[5]),
                             pack_bf16x2(a8[6], a8[7]))

    # ------------------------------------------------------------------ softmax
    @cute.jit
    def softmax(self, base, warp, lane, taddr, cta, n_mine, out, out_stride, lse,
                lse_stride, scale_log2, out_scale, stamp, ws):
        """Group k (warps 2 + 4k ..) owns heads 32k .. 32k + 31; its warp of TMEM lane
        quarter q = warp % 4 owns keys 32q .. 32q + 31 of a tile (S^T lanes) and dims
        32q .. 32q + 31 of every 128-dim block of O^T. Per tile: warp maxima of the raw
        scores -> the group's quarter-0 warp (lane = head) keeps the running max in
        log2 units and publishes P's exponent bias (m - log2 448) and the rescale
        factor -> every thread forms P = 2^(s x scale - bias) (448 x 2^(x - m), E4M3
        as trtllm-gen) and its row-sum partials (in units of 448)."""
        k = (warp - SOFT0) // 4
        quarter = warp % 4
        wt = quarter * 32 + lane
        lanes = Uint32(quarter * 32) << Uint32(16)
        h0 = k * HW
        tbuf = base + Uint32(self.off_tmax) + Uint32((k * 4) * HW * 4)
        bias_s = base + Uint32(self.off_bias) + Uint32(4 * h0)
        alpha_g = base + Uint32(self.off_alpha) + Uint32(4 * h0)
        flag_g = base + Uint32(self.off_flag) + Uint32(4 * k)
        lfin = base + Uint32(self.off_l)
        key = wt
        combiner = quarter == 0  # lane = head h0 + lane
        pbase = base + Uint32(self.off_p) + Uint32((key // 8) * 128 + (key % 8) * 16)
        neg = Float32(-float("inf"))
        log2_448 = Float32(8.807354922057604)
        g = Int32(0)
        ro = Int32(0)
        for i in range(n_mine):
            n = self.nrow(base, i)
            if n > 0:
                row = self.rowid(base, i)
                tiles = _tiles(n)
                lp = [Float32(0.0)] * HW  # row-sum partials of this key lane
                m = neg  # combiner: running max of head h0 + lane (log2 units)
                for t in range(tiles):
                    b = g & 1
                    alpha_s = alpha_g + Uint32(b) * Uint32(H * 4)
                    flag_s = flag_g + Uint32(b) * Uint32(16)
                    mbar_wait(self.bar(base, B_SFULL + b), (g >> 1) & 1)
                    if (warp == SOFT0) & (lane == 0) & (ro < 2) & (t < 10):
                        self.mark(stamp, cta, 3 + 16 * ro + t)
                    tc_fence_after()
                    sv = tmem_ld32(taddr + lanes + Uint32(C_S + 96 * b + HW * k))
                    tc_fence_before()
                    cute.arch.sync_warp()
                    if lane == 0:
                        mbar_arrive(self.bar(base, B_SEMPTY + b))
                    sub = (warp == SOFT0) & (lane == 0) & (ro == 0) & (t == 1)
                    if sub:
                        self.mark(stamp, cta, 31)
                    x = [sv[h] for h in range(HW)]
                    if t == tiles - 1:  # keys past the row's count
                        ok = key < n - t * TN
                        x = [cutlass.select_(ok, sv[h], neg) for h in range(HW)]
                    for c in cutlass.range_constexpr(HW // 4):
                        r4 = [_bits(redux_max(x[4 * c + e])) for e in range(4)]
                        if lane == 0:
                            st_shared_v4(tbuf + Uint32(quarter * HW * 4 + 16 * c),
                                         r4[0], r4[1], r4[2], r4[3])
                    if sub:
                        self.mark(stamp, cta, 32)
                    if g >= 2:  # PV(g - 2) is complete: its P^T buffer, rescale
                        # factors and flags are free
                        mbar_wait(self.bar(base, B_PVDONE + b), ((g >> 1) & 1) ^ 1)
                    named_bar(Uint32(1) + Uint32(k), Uint32(128))
                    if sub:
                        self.mark(stamp, cta, 33)
                    if combiner:
                        tm = ld_shared_f32(tbuf + Uint32(4) * Uint32(lane))
                        for qq in cutlass.range_constexpr(1, 4):
                            tm = fmax(tm, ld_shared_f32(tbuf + Uint32(qq * HW * 4)
                                                        + Uint32(4) * Uint32(lane)))
                        m_new = fmax(m, tm * scale_log2)
                        al = ex2(m - m_new)
                        m = m_new
                        st_shared_f32(bias_s + Uint32(4) * Uint32(lane), m_new - log2_448)
                        st_shared_f32(alpha_s + Uint32(4) * Uint32(lane), al)
                        grown = vote_any(Uint32(al < Float32(1.0)))
                        if lane == 0:
                            st_shared_u32(flag_s, grown)
                        cute.arch.sync_warp()
                        mbar_arrive(self.bar(base, B_ALPHA + NG * b + k))
                    named_bar(Uint32(1) + Uint32(k), Uint32(128))
                    if sub:
                        self.mark(stamp, cta, 34)
                    # Row sums: l = l x 2^(m_old - m_new) + p, one FMA per head (the
                    # factor is exactly 1 where the max held, 0 on a row's first tile).
                    pwords = []
                    lnew = []
                    for c in cutlass.range_constexpr(HW // 4):
                        b4 = ld_shared_v4f(bias_s + Uint32(16 * c))
                        a4 = ld_shared_v4f(alpha_s + Uint32(16 * c))
                        p4 = [ex2(fma_f(x[4 * c + e], scale_log2, -b4[e])) for e in range(4)]
                        pwords.append(e4m3x4(p4[0], p4[1], p4[2], p4[3]))
                        for e in cutlass.range_constexpr(4):
                            lnew.append(fma_f(lp[4 * c + e], a4[e], p4[e]))
                    if sub:
                        self.mark(stamp, cta, 35)
                    if sub:
                        self.mark(stamp, cta, 36)
                    pb = pbase + Uint32(b) * Uint32(PBYTES)
                    for c in cutlass.range_constexpr(2):
                        st_shared_v4(pb + Uint32((2 * k + c) * 2048), pwords[4 * c],
                                     pwords[4 * c + 1], pwords[4 * c + 2], pwords[4 * c + 3])
                    if sub:
                        self.mark(stamp, cta, 37)
                    lp = lnew
                    fence_async_smem()
                    cute.arch.sync_warp()
                    if lane == 0:
                        mbar_arrive(self.bar(base, B_PFULL + b))
                    g = g + 1
                # Row end: the row sums (a transposing warp reduction: lane = head,
                # then the 4 key quarters), LSE and 1 / (448 l) for the correction
                # groups, which write O^T while these warps go on with the next row.
                ep = (warp == SOFT0) & (lane == 0) & (ro == 0)
                if ep:
                    self.mark(stamp, cta, 15)
                v = lp
                for sh in cutlass.range_constexpr(5):
                    half = len(v) // 2
                    hi = (lane & (16 >> sh)) != 0
                    nv = []
                    for j in cutlass.range_constexpr(half):
                        send = cutlass.select_(hi, v[j], v[j + half])
                        keep = cutlass.select_(hi, v[j + half], v[j])
                        nv.append(keep + shfl_xor(send, Uint32(16 >> sh)))
                    v = nv
                named_bar(Uint32(1) + Uint32(k), Uint32(128))  # tile maxima all read
                st_shared_f32(tbuf + Uint32(quarter * HW * 4) + Uint32(4) * Uint32(lane), v[0])
                named_bar(Uint32(1) + Uint32(k), Uint32(128))
                if combiner:
                    lt = ld_shared_f32(tbuf + Uint32(4) * Uint32(lane))
                    for qq in cutlass.range_constexpr(1, 4):
                        lt = lt + ld_shared_f32(tbuf + Uint32(qq * HW * 4)
                                                + Uint32(4) * Uint32(lane))
                    # lt = 448 l: LSE = m + log2 l; O = O~ / (448 l) x out_scale
                    lse_v = m + lg2(lt) - log2_448
                    fin_v = out_scale / lt
                    role = self.role(base, i)
                    if role == TAIL:  # the partial's LSE into this CTA's slot
                        st_global_f32(ws + Int64(WS_LSE) + (Int64(cta) * Int64(H)
                                                            + Int64(h0 + lane)) * Int64(4),
                                      lse_v)
                        fence_gpu()
                    elif role > 0:  # head: wait for the tails, merge weights
                        big = lse_v
                        for jt in range(role):
                            fl = ws + Int64(WS_FLAGS) + Int64(4) * Int64(cta + 1 + jt)
                            seen = ld_acquire_s32(fl)
                            while seen < NCORR:
                                seen = ld_acquire_s32(fl)
                            big = fmax(big, self.tail_lse(ws, cta + 1 + jt, h0 + lane))
                        tot_w = ex2(lse_v - big)
                        for jt in range(role):
                            tot_w = tot_w + ex2(self.tail_lse(ws, cta + 1 + jt, h0 + lane)
                                                - big)
                        lse_v = big + lg2(tot_w)
                    if role != TAIL:
                        st_global_f32(lse + (Int64(row) * Int64(lse_stride)
                                             + Int64(h0 + lane)) * Int64(4), lse_v)
                    if ro > 0:  # the previous row's epilogue read its factors
                        mbar_wait(self.bar(base, B_LEMPTY + k), (ro - 1) & 1)
                    st_shared_f32(lfin + Uint32(4) * Uint32(h0 + lane), fin_v)
                    if role > 0:
                        for jt in range(role):
                            st_shared_f32(base + Uint32(self.off_w)
                                          + Uint32(4) * Uint32(jt * H + h0 + lane),
                                          ex2(self.tail_lse(ws, cta + 1 + jt, h0 + lane)
                                              - lse_v))
                        st_shared_f32(base + Uint32(self.off_w)
                                      + Uint32(4) * Uint32(NT_MAX * H + h0 + lane),
                                      ex2(m + lg2(lt) - log2_448 - lse_v))
                    mbar_arrive(self.bar(base, B_LFULL + k))
                named_bar(Uint32(1) + Uint32(k), Uint32(128))  # the sums are read
                role = self.role(base, i)
                if role > 0:
                    # A head is its CTA's last keyed segment: once the correction
                    # group stored this group's own partial latent (normalized alone),
                    # merge the tails' partials into it as DCP does (FP32 weights
                    # 2^(lse - LSE), one BF16 rounding).
                    mbar_wait(self.bar(base, B_LEMPTY + k), ro & 1)
                    self.merge(base, wt, h0, row, role, cta, out, out_stride, ws)
                    named_bar(Uint32(1) + Uint32(k), Uint32(128))
                    if wt < role:  # the reads are done: re-arm the tail's flag
                        fl = ws + Int64(WS_FLAGS) + Int64(4) * Int64(cta + 1 + wt)
                        if atom_add(fl, 1) == NCORR + NG - 1:
                            st_relaxed_s32(fl, 0)
                if ep:
                    self.mark(stamp, cta, 16)
                ro = ro + 1
            else:  # no keys: LSE -inf and a zero latent (as trtllm-gen)
                row = self.rowid(base, i)
                if combiner:
                    st_global_f32(lse + (Int64(row) * Int64(lse_stride) + Int64(h0 + lane))
                                  * Int64(4), neg)
                dst = out + (Int64(row) * Int64(out_stride) + Int64(h0) * Int64(D_NOPE)
                             + Int64(key)) * Int64(2)
                for blk in cutlass.range_constexpr(4):
                    for h in cutlass.range_constexpr(HW):
                        st_global_bf16(dst + Int64((h * D_NOPE + 128 * blk) * 2),
                                       Float32(0.0))


@asm(Float32, pure=True)
def fma_f(a, b, c):
    return "fma.rn.f32 $0, $1, $2, $3;", "=f,f,f,f", (Float32(a), Float32(b), Float32(c))


@asm(Uint32, pure=False)
def match_any(v):
    return "match.any.sync.b32 $0, $1, 0xffffffff;", "=r,r", (Uint32(v),)


@asm(Uint32, pure=True)
def lanemask_lt():
    return "mov.u32 $0, %lanemask_lt;", "=r", ()


@asm(Int32, pure=True)
def shfl_xor_i(v, m):
    return "shfl.sync.bfly.b32 $0, $1, $2, 0x1f, 0xffffffff;", "=r,r,r", (
        Int32(v), Uint32(m))


@asm(Int32, pure=True)
def shfl_up_i(v, d):
    return "shfl.sync.up.b32 $0, $1, $2, 0, 0xffffffff;", "=r,r,r", (Int32(v), Uint32(d))


@asm(Int32, pure=True)
def shfl_idx_i(v, src):
    return "shfl.sync.idx.b32 $0, $1, $2, 0x1f, 0xffffffff;", "=r,r,r", (
        Int32(v), Uint32(src))


@asm(Uint32, pure=False)
def ballot(flag):
    code = ("{ .reg .pred p; setp.ne.u32 p, $1, 0; "
            "vote.sync.ballot.b32 $0, p, 0xffffffff; }")
    return code, "=r,r", (Uint32(flag),)


@asm(Int32, pure=False)
def ld_acquire_s32(ptr):
    return "ld.acquire.gpu.global.s32 $0, [$1];", "=r,l,~{memory}", (Int64(ptr),)


@asm(Float32, pure=False)
def ld_relaxed_f32(ptr):
    return "ld.relaxed.gpu.global.f32 $0, [$1];", "=f,l,~{memory}", (Int64(ptr),)


@asm((Uint32, Uint32, Uint32, Uint32), pure=False)
def ld_cg_v4(ptr):
    return "ld.global.cg.v4.b32 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,l", (Int64(ptr),)


@asm()
def st_global_v4(ptr, a, b, c, d):
    return ("st.global.v4.b32 [$0], {$1, $2, $3, $4};", "l,r,r,r,r,~{memory}",
            (Int64(ptr), Uint32(a), Uint32(b), Uint32(c), Uint32(d)))


@asm(Uint32, pure=True)
def pack_bf16x2(lo, hi):
    return "cvt.rn.bf16x2.f32 $0, $2, $1;", "=r,f,f", (Float32(lo), Float32(hi))


@asm((Float32, Float32), pure=True)
def unpack_bf16(word):
    return "shl.b32 $0,$2,16; and.b32 $1,$2,0xffff0000;", "=f,=f,r", (Uint32(word),)


@asm()
def red_release_add(ptr, v):
    return "red.release.gpu.global.add.s32 [$0], $1;", "l,r,~{memory}", (
        Int64(ptr), Int32(v))


@asm(Int32, pure=False)
def atom_add(ptr, v):
    return "atom.relaxed.gpu.global.add.s32 $0, [$1], $2;", "=r,l,r,~{memory}", (
        Int64(ptr), Int32(v))


@asm()
def st_relaxed_s32(ptr, v):
    return "st.relaxed.gpu.global.s32 [$0], $1;", "l,r,~{memory}", (Int64(ptr), Int32(v))


@asm()
def fence_gpu():
    return "fence.acq_rel.gpu;", "~{memory}", ()


@asm(Uint32, pure=True)
def popc(v):
    return "popc.b32 $0, $1;", "=r,r", (Uint32(v),)


@asm()
def sync_warp():
    return "bar.warp.sync 0xffffffff;", "~{memory}", ()


@asm(Uint32, pure=False)
def vote_any(flag):
    code = (
        "{ .reg .pred p, q; setp.ne.u32 p, $1, 0; vote.sync.any.pred q, p, 0xffffffff; "
        "selp.u32 $0, 1, 0, q; }"
    )
    return code, "=r,r", (Uint32(flag),)


@asm()
def tmem_ld32(addr):
    regs = ",".join(f"${i}" for i in range(32))
    code = (f"tcgen05.ld.sync.aligned.32x32b.x32.b32 {{{regs}}}, [$32]; "
            "tcgen05.wait::ld.sync.aligned;")
    return code, ",".join(["=f"] * 32 + ["r", "~{memory}"]), (Uint32(addr),), (
        Float32,) * 32


@asm()
def tmem_st32(addr, vals):
    regs = ",".join(f"${i + 1}" for i in range(32))
    code = (f"tcgen05.st.sync.aligned.32x32b.x32.b32 [$0], {{{regs}}}; "
            "tcgen05.wait::st.sync.aligned;")
    return code, "r" + ",f" * 32 + ",~{memory}", (
        Uint32(addr), *[Float32(v) for v in vals])


@asm()
def tmem_st32_nw(addr, vals):
    regs = ",".join(f"${i + 1}" for i in range(32))
    code = f"tcgen05.st.sync.aligned.32x32b.x32.b32 [$0], {{{regs}}};"
    return code, "r" + ",f" * 32 + ",~{memory}", (
        Uint32(addr), *[Float32(v) for v in vals])


@asm()
def tmem_st16_nw(addr, vals):
    regs = ",".join(f"${i + 1}" for i in range(16))
    code = f"tcgen05.st.sync.aligned.32x32b.x16.b32 [$0], {{{regs}}};"
    return code, "r" + ",f" * 16 + ",~{memory}", (
        Uint32(addr), *[Float32(v) for v in vals])


@asm()
def tmem_wait_st():
    return "tcgen05.wait::st.sync.aligned;", "~{memory}", ()


@asm()
def ld_shared_v4f(addr):
    return ("ld.shared.v4.f32 {$0,$1,$2,$3}, [$4];", "=f,=f,=f,=f,r,~{memory}",
            (Uint32(addr),), (Float32,) * 4)


@asm(Float32, pure=True)
def fmin(a, b):
    return "min.f32 $0, $1, $2;", "=f,f,f", (Float32(a), Float32(b))


@asm(Uint32, pure=True)
def _bits(v):
    return "mov.b32 $0, $1;", "=r,f", (Float32(v),)


@asm(Int32, pure=False)
def ld_global_s32(ptr):
    return "ld.global.nc.s32 $0, [$1];", "=r,l", (Int64(ptr),)


@cache
def _compiled(grid: int, stages: int, loaders: int, stamps: bool, cpa: int = 0,
              prefetch: int = 3, gather_only: bool = False, epi_store: bool = True,
              no_gather: bool = False):
    kernel = SparseMlaDecodeKM(grid, stages, loaders, stamps, cpa, prefetch, gather_only,
                               epi_store, no_gather)
    args = (Int64(0), Int32(128), Int32(1), Int64(0), Int32(64), Int64(0), Int32(128),
            Int64(0), Int64(0), Int32(1), Int64(0), Int32(1), Float32(1.0), Float32(1.0),
            Int64(0), Int64(0), cuda.CUstream(0))
    opts = f"--ptxas-options -maxrregcount={min(255, 65536 // kernel.threads // 8 * 8)}"
    return cute.compile(kernel, *args, options=opts)


def sparse_mla_decode_km(query, kv_rows, slots, valid, latent, lse, scale, out_scale,
                         grid=None, stages=3, loaders=LOADERS, stamps=None, cpa=0,
                         prefetch=3, gather_only=False, epi_store=True,
                         no_gather=False):
    """As ``sparse_mla.sparse_mla_decode`` (``query`` [R, Hq, 576] contiguous)."""
    rows, q_heads = query.shape[0], query.shape[1]
    assert query.shape[2] == ROW and query.is_contiguous() and q_heads >= H
    assert kv_rows.shape[1] == ROW and kv_rows.stride(0) == ROW
    assert slots.shape[1] % TN == 0 and slots.shape[1] <= IDXB // 4
    assert slots.stride(1) == 1 and slots.stride(0) % 4 == 0
    assert latent.shape[2] == D_NOPE and latent.stride(1) == D_NOPE
    assert latent.shape[1] == q_heads and latent.stride(0) == q_heads * D_NOPE
    if rows == 0:
        return latent, lse
    if grid is None:  # one compile for every row count: CTAs past the rows exit
        grid = torch.cuda.get_device_properties(query.device).multi_processor_count
    if grid > GRID_MAX or rows > ROWS_MAX:
        raise ValueError(f"sparse_mla_decode_km: grid {grid} / {rows} rows unsupported")
    ws = workspace(query.device)
    fn = _compiled(grid, stages, loaders, stamps is not None, cpa, prefetch, gather_only,
                   epi_store, no_gather)
    stream = cuda.CUstream(torch.cuda.current_stream(query.device).cuda_stream)
    fn(Int64(query.data_ptr()), Int32(q_heads), Int32(rows), Int64(kv_rows.data_ptr()),
       Int32(kv_rows.shape[0]), Int64(slots.data_ptr()), Int32(slots.stride(0)),
       Int64(valid.data_ptr()), Int64(latent.data_ptr()), Int32(latent.stride(0)),
       Int64(lse.data_ptr()), Int32(lse.stride(0)), Float32(scale * LOG2E),
       Float32(out_scale), Int64(0 if stamps is None else stamps.data_ptr()),
       Int64(ws.data_ptr()), stream)
    return latent, lse


def kernel_info(device, grid, stages=3, loaders=LOADERS, cpa=0, prefetch=3):
    """(registers per thread, local memory bytes per thread) of the compiled kernel
    (benchmarks: spill check)."""
    from ..common.host import _checked, resolve_kernel

    _, function = resolve_kernel(
        _compiled(grid, stages, loaders, False, cpa, prefetch), device)
    attr = cuda.CUfunction_attribute
    (regs,) = _checked(cuda.cuFuncGetAttribute(attr.CU_FUNC_ATTRIBUTE_NUM_REGS, function))
    (local,) = _checked(
        cuda.cuFuncGetAttribute(attr.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, function))
    return regs, local
