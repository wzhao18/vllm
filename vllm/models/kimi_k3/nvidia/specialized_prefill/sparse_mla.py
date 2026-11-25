# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 DSA prefill sparse MLA on Rubin (sm_107), native 96 heads.

Per query row (token) t with ``lengths[t]`` valid entries of its top-k physical
slots: O[t] = softmax(q_t . K[sel]^T * scale) . V[sel] over the standard FP8 MLA
cache (576 B rows: 512 E4M3 latent | 64 E4M3 RoPE, one per-tensor scale) and the
per-tensor FP8 query of the stock layer (``decode_query_quant_pad``, 96 heads,
no padding). Output BF16 [T, 96, 512].

Transposed ("keys on M") tile: per 128-row KV tile
  S^T [128 keys, 96 heads] = K [128 x 576] (A, K-major SW128 + SW64 RoPE) . Q^T (B)
  O^T [512 dims, 96 heads] += V^T [4 x 128 dims x 128 keys] (A, MN-major) . P^T (B)
with tcgen05 dense kind::f8f6f4 MMAs in the Rubin 2x-K form (K64 per instruction,
FP32 accumulation in TMEM: S^T 96 columns, O^T 4 x 96 columns). No head padding:
the 96 heads are the MMA's N. One gathered SMEM tile serves QK (K-major) and PV
(MN-major V^T). Softmax: thread = key lane, two warpgroups of 48 heads; running max
per head kept in shared memory, raised only when a tile's max grows by more than
THR (log2), else P is formed against the stale max (P <= 2^(P_PRE + THR) < 448,
E4M3 as the stock kernel's P); l = sum of the E4M3 P that PV consumes.

Derived from the GB300 T96 v2 prototype (transposed, cp.async row loaders); Rubin
changes: dense 2x-K FP8 MMAs (no unit block scales), per-tensor FP8 Q, more loader
warps, physical slots and valid counts as the stock backend passes them.
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

H = 96  # query heads per GPU (DEP16: TP1)
HW = 48  # heads per softmax warpgroup
ROW = 576
D_NOPE = 512
LOG2E = 1.4426950408889634
P_PRE = 7  # P is formed as 2^(x - m + P_PRE); O / l is invariant
THR = 1.0  # lazy max: P <= 2^(P_PRE + THR) = 256 < 448 (E4M3 max)
SOFTMAX_WARPS = 8

# SMEM layout (bytes) -------------------------------------------------------------
OFF_QN = 0  # Q latent: 4 x [96 rows x 128 B] SW128 (B operand of QK, K-major)
OFF_QR = OFF_QN + 4 * H * 128  # Q RoPE: [96 x 64 B] SW64
STAGE = 4 * 128 * 128 + 128 * 64  # KV tile: latent SW128 atoms + RoPE SW64
OFF_P = OFF_QR + H * 64  # 2 x P^T [128 keys x 96 heads] MN-major, no swizzle
OFF_RED = OFF_P + 2 * H * 128
OFF_MT = OFF_RED + 2 * 4 * HW * 4
OFF_L = OFF_MT + 2 * HW * 4
OFF_M = OFF_L + 2 * 4 * HW * 4  # [2 WG][48] running max (log2 units)
OFF_AL = OFF_M + 2 * HW * 4  # [2 parity][2 WG][48] alpha of this tile
OFF_FLAG = OFF_AL + 2 * 2 * HW * 4
OFF_BAR = OFF_FLAG + 16
OFF_TMEM = OFF_BAR + 32 * 8
OFF_KV = (OFF_TMEM + 16 + 1023) // 1024 * 1024  # NS stages of STAGE bytes


def smem_bytes(stages: int) -> int:
    return OFF_KV + stages * STAGE

B_QFULL, B_QEMPTY, B_KVFULL, B_KVEMPTY, B_SFULL, B_PFULL, B_OFULL, B_OFREE = (
    0, 1, 16, 24, 6, 7, 8, 9,
)  # KV full / empty: one barrier per stage (up to 8 stages)
# pipelined: second S^T / P^T buffers' barriers and the per-tile PV completion
B_SFULL2, B_PFULL2, B_PVDONE = 10, 11, 12
C_S = 0  # TMEM columns: S^T 96 (pipelined: two S^T buffers, 0 and 96)
C_O, C_O_PIPE = 128, 192  # O^T 4 x 96

# tcgen05 shared-memory descriptors (version bit 46)
SDESC_SW128 = (1024 >> 4 << 32) | (1 << 46) | (2 << 61)  # 8 rows x 128 B
SDESC_SW64 = (512 >> 4 << 32) | (1 << 46) | (4 << 61)  # 8 rows x 64 B
# P^T: MN-major, no swizzle, core matrices of 8 keys x 16 heads (128 B); LBO = K
# direction core stride 128 B, SBO = N direction (16-head chunk) stride 2048 B.
P_DESC = (128 >> 4 << 16) | (2048 >> 4 << 32) | (1 << 46)
# Dense kind::f8f6f4 instruction descriptor: FP32 D, E4M3 A and B, M = 128, N = 96,
# bit 29 = Rubin 2x-K (K64 per instruction).
IDESC_QK = (1 << 4) | (H >> 3 << 17) | (128 >> 4 << 24) | (1 << 29)
IDESC_PV = IDESC_QK | (1 << 15) | (1 << 16)  # A (V^T) and B (P^T) MN-major


# PTX ------------------------------------------------------------------------------
@asm()
def mbar_init(bar, count):
    return "mbarrier.init.shared::cta.b64 [$0], $1;", "r,r,~{memory}", (
        Uint32(bar), Uint32(count))


@asm()
def mbar_arrive(bar):
    return "mbarrier.arrive.shared::cta.b64 _, [$0];", "r,~{memory}", (Uint32(bar),)


@asm()
def mbar_arrive_tx(bar, nbytes):
    return ("mbarrier.arrive.expect_tx.shared::cta.b64 _, [$0], $1;", "r,r,~{memory}",
            (Uint32(bar), Uint32(nbytes)))


@asm()
def mbar_wait(bar, parity):
    code = ("{ .reg .pred p; WAIT_${:uid}: mbarrier.try_wait.parity.shared::cta.b64 "
            "p, [$0], $1; @!p bra WAIT_${:uid}; }")
    return code, "r,r,~{memory}", (Uint32(bar), Uint32(parity))


@asm()
def named_bar(bid, n):
    return "bar.sync $0, $1;", "r,r,~{memory}", (Uint32(bid), Uint32(n))


@asm()
def tma_tile2d(dst, tmap, bar, col, row):
    code = ("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx"
            "::bytes [$0], [$1, {$3, $4}], [$2];")
    return code, "r,l,r,r,r,~{memory}", (Uint32(dst), Int64(tmap), Uint32(bar),
                                          Int32(col), Int32(row))


@asm()
def tc_fence_after():
    return "tcgen05.fence::after_thread_sync;", "~{memory}", ()


@asm()
def tc_fence_before():
    return "tcgen05.fence::before_thread_sync;", "~{memory}", ()


@asm()
def fence_async_smem():
    return "fence.proxy.async.shared::cta;", "~{memory}", ()


@asm()
def tc_commit(bar):
    return ("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [$0];",
            "r,~{memory}", (Uint32(bar),))


@asm()
def tc_mma(d, a, b, idesc, acc):
    code = ("{ .reg .pred p; setp.ne.b32 p, $4, 0; "
            "tcgen05.mma.cta_group::1.kind::f8f6f4 [$0], $1, $2, $3, p; }")
    return code, "r,l,l,r,r,~{memory}", (Uint32(d), Uint64(a), Uint64(b), Uint32(idesc),
                                          Uint32(acc))


@asm()
def tmem_ld(addr, n):
    regs = ",".join(f"${i}" for i in range(n))
    code = (f"tcgen05.ld.sync.aligned.32x32b.x{n}.b32 {{{regs}}}, [${n}]; "
            "tcgen05.wait::ld.sync.aligned;")
    return code, ",".join(["=f"] * n + ["r", "~{memory}"]), (Uint32(addr),), (Float32,) * n


@asm()
def tmem_st(addr, vals):
    n = len(vals)
    regs = ",".join(f"${i + 1}" for i in range(n))
    code = (f"tcgen05.st.sync.aligned.32x32b.x{n}.b32 [$0], {{{regs}}}; "
            "tcgen05.wait::st.sync.aligned;")
    return code, "r" + ",f" * n + ",~{memory}", (Uint32(addr), *[Float32(v) for v in vals])


@asm()
def ld_shared_f32(addr):
    return "ld.shared.f32 $0, [$1];", "=f,r,~{memory}", (Uint32(addr),), Float32


@asm()
def ld_shared_v4f(addr):
    return ("ld.shared.v4.f32 {$0,$1,$2,$3}, [$4];", "=f,=f,=f,=f,r,~{memory}",
            (Uint32(addr),), (Float32,) * 4)


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
def st_shared_v4(addr, w0, w1, w2, w3):
    return ("st.shared.v4.b32 [$0], {$1,$2,$3,$4};", "r,r,r,r,r,~{memory}",
            (Uint32(addr), Uint32(w0), Uint32(w1), Uint32(w2), Uint32(w3)))


@asm()
def st_global_b16(ptr, v):
    return ("{ .reg .b16 t; cvt.rn.bf16.f32 t, $1; st.global.b16 [$0], t; }",
            "l,f,~{memory}", (Int64(ptr), Float32(v)))


@asm(Float32, pure=True)
def ex2(v):
    return "ex2.approx.ftz.f32 $0, $1;", "=f,f", (Float32(v),)


@asm(Float32, pure=True)
def redux_max_f32(v):
    return "redux.sync.max.f32 $0, $1, 0xffffffff;", "=f,f", (Float32(v),)


@asm(Uint32, pure=True)
def cvt_e4m3x2(lo, hi):
    return ("{ .reg .b16 t; cvt.rn.satfinite.e4m3x2.f32 t, $2, $1; cvt.u32.u16 $0, t; }",
            "=r,f,f", (Float32(lo), Float32(hi)))


@asm((Float32, Float32), pure=True)
def e4m3x2_roundtrip(pair):
    code = ("{ .reg .b16 t; .reg .b32 h; .reg .f16 a,b; cvt.u16.u32 t, $2; "
            "cvt.rn.f16x2.e4m3x2 h, t; mov.b32 {a,b}, h; cvt.f32.f16 $0, a; "
            "cvt.f32.f16 $1, b; }")
    return code, "=f,=f,r", (Uint32(pair),)


@asm(Int32, pure=False)
def ld_global_s32_pred(ptr, ok):
    """``*ptr`` if ok else -1 (no load issued when not ok)."""
    code = ("{ .reg .pred p; setp.ne.b32 p, $2, 0; mov.b32 $0, -1; "
            "@p ld.global.nc.s32 $0, [$1]; }")
    return code, "=r,l,r", (Int64(ptr), Int32(ok))


@asm()
def cp_rows4(lo, hi, src_lanes, dsts, lane16, rope_dsts, lane):
    """For 4 rows: the row's global byte address (lo, hi) is shuffled from lane
    ``src_lanes[k]``; hi < 0 marks an invalid row (zero fill). Every lane copies 16-B
    latent chunk ``lane`` to dsts[k]; lanes 0-3 also copy RoPE chunk ``lane``."""
    code = "{ .reg .b32 l<4>, h<4>, n<4>; .reg .b64 a<4>; .reg .pred p<4>, q; "
    code += "setp.lt.u32 q, $13, 4; "
    for k in range(4):
        code += (f"shfl.sync.idx.b32 l{k}, $0, {src_lanes[k]}, 31, -1; "
                 f"shfl.sync.idx.b32 h{k}, $1, {src_lanes[k]}, 31, -1; ")
    for k in range(4):
        code += (f"setp.lt.s32 p{k}, h{k}, 0; selp.b32 n{k}, 0, 16, p{k}; "
                 f"mov.b64 a{k}, {{l{k}, h{k}}}; @p{k} mov.b64 a{k}, 0; "
                 f"add.s64 a{k}, a{k}, $2; ")
        code += f"cp.async.cg.shared.global [${3 + k}], [a{k}], 16, n{k}; "
        code += f"@q cp.async.cg.shared.global [${7 + k}], [a{k}+512], 16, n{k}; "
    code += "}"
    ops = (Int32(lo), Int32(hi), Int64(lane16), *[Uint32(d) for d in dsts],
           *[Uint32(d) for d in rope_dsts], Int32(0), Int32(0), Uint32(lane))
    return code, "r,r,l,r,r,r,r,r,r,r,r,r,r,r", ops


@asm()
def gather4(dst, desc, bar, col, r0, r1, r2, r3):
    """TMA tile::gather4: rows r0..r3 (out of range, e.g. -1: zero fill) of one box."""
    code = ("cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
            ".mbarrier::complete_tx::bytes [$0],[$1,{$3,$4,$5,$6,$7}],[$2];")
    ops = (Uint32(dst), Int64(desc), Uint32(bar), Int32(col), Int32(r0), Int32(r1),
           Int32(r2), Int32(r3))
    return code, "r,l,r,r,r,r,r,r,~{memory}", ops


@asm()
def ld_global_v4_s32(ptr):
    return ("ld.global.nc.v4.b32 {$0,$1,$2,$3}, [$4];", "=r,=r,=r,=r,l", (Int64(ptr),),
            (Int32,) * 4)


@asm()
def cp_async_arrive(bar):
    return "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [$0];", "r,~{memory}", (
        Uint32(bar),)


@asm()
def tmem_alloc_exclusive(dst):
    """All 576 TMEM columns (Rubin exclusive allocation: one CTA per SM)."""
    code = ("tcgen05.alloc.exclusive.cta_group::1.sync.aligned.shared::cta.b32 [$0], 576; "
            "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;")
    return code, "r,~{memory}", (Uint32(dst),)


@asm()
def tmem_free_exclusive(taddr):
    return ("tcgen05.dealloc.exclusive.cta_group::1.sync.aligned.b32 $0, 576;",
            "r,~{memory}", (Uint32(taddr),))


@asm()
def stamp_clock(ptr):
    return ("{ .reg .b64 c; mov.u64 c, %clock64; st.global.u64 [$0], c; }", "l,~{memory}",
            (Int64(ptr),))


STAMP_CTAS, STAMP_TILES, STAMP_EV = 8, 64, 8


def _bar(base, i):
    return base + Uint32(OFF_BAR + 8 * i)


@cute.jit
def _tiles(n):
    t = (n + 127) // 128
    if t < 1:
        t = Int32(1)
    return t


class SparseMlaPrefill:
    """``loaders`` cp.async row-loader warps; ``grid`` persistent CTAs (one per SM)."""

    def __init__(self, loaders: int, grid: int, pipelined: bool = False,
                 tma: bool = False, regs: tuple = (),
                 softmax_shift: tuple = (P_PRE, THR), stages: int = 2,
                 stamps: bool = False):
        self.loaders, self.grid, self.pipelined = loaders, grid, pipelined
        # stages: KV ring depth (3 needs the oversized shared-memory carveout);
        # stamps: record clock64 phase stamps of the first CTAs (diagnostics)
        self.stages, self.stamps = stages, stamps
        self.smem = smem_bytes(stages)
        # tma: loaders issue TMA tile::gather4 (one lane per 4-row quad) instead of
        # cp.async rows; regs: (softmax, other) setmaxnreg counts per warpgroup
        self.tma, self.regs = tma, regs
        # (P_PRE, THR): P = 2^(x - m + P_PRE) <= 2^(P_PRE + THR) < 448 (E4M3 max)
        self.p_pre, self.thr = softmax_shift
        # pipelined: two S^T and P^T buffers (all 576 TMEM columns, exclusive) and
        # the MMA order QK(t + 1) before PV(t), so softmax(t + 1) overlaps PV(t)
        self.c_o = C_O_PIPE if pipelined else C_O
        self.threads = 32 * (SOFTMAX_WARPS + 1 + loaders)

    @cute.jit
    def __call__(self, q8: Int64, rows: Int32, kv: Int64, unit_rows: Int32, slots: Int64,
                 slot_stride: Int32, lengths: Int64, out: Int64, scale_log2: Float32,
                 out_scale: Float32, kv_count: Int32, stamp: Int64,
                 stream: cuda.CUstream):
        u8 = cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8
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
        self.kernel(qn, qr, kvm, kvr, rows, kv, unit_rows, slots, slot_stride, lengths,
                    out, scale_log2, out_scale, stamp).launch(
            grid=(self.grid, 1, 1), block=(self.threads, 1, 1), smem=self.smem,
            stream=stream)

    @cute.kernel
    def kernel(self, qn: tensormap.GridMap, qr: tensormap.GridMap, kvm: tensormap.GridMap,
               kvr: tensormap.GridMap, rows: Int32, kv: Int64,
               unit_rows: Int32, slots: Int64, slot_stride: Int32, lengths: Int64,
               out: Int64, scale_log2: Float32, out_scale: Float32, stamp: Int64):
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(tid // 32)
        lane = tid % 32
        smem = cutlass.utils.SmemAllocator()
        raw = smem.allocate_array(Uint8, self.smem, byte_alignment=1024)
        base = Uint32(raw.toint())
        if tid == 0:
            kvf = 1 if cutlass.const_expr(self.tma) else 32 * self.loaders
            for st in cutlass.range_constexpr(self.stages):
                mbar_init(_bar(base, B_KVFULL + st), Uint32(kvf))
                mbar_init(_bar(base, B_KVEMPTY + st), Uint32(1))
            for i, cnt in ((B_QFULL, 1), (B_QEMPTY, 1), (B_SFULL, 1), (B_PFULL, SOFTMAX_WARPS),
                           (B_OFULL, 1), (B_OFREE, SOFTMAX_WARPS), (B_SFULL2, 1),
                           (B_PFULL2, SOFTMAX_WARPS), (B_PVDONE, 1)):
                mbar_init(_bar(base, i), Uint32(cnt))
            cute.arch.mbarrier_init_fence()
        if warp == 0:
            if cutlass.const_expr(self.pipelined):
                tmem_alloc_exclusive(base + Uint32(OFF_TMEM))
            else:
                cute.arch.alloc_tmem(512, cute.make_ptr(
                    Int32, base + Uint32(OFF_TMEM), cute.AddressSpace.smem))
                cute.arch.relinquish_tmem_alloc_permit()
        tc_fence_before()
        cute.arch.sync_threads()
        tc_fence_after()
        taddr = ld_shared_u32(base + Uint32(OFF_TMEM))
        n_mine = (rows - cta + self.grid - 1) // self.grid
        regs = (
            cute.make_rmem_tensor((HW,), Float32),
            cute.make_rmem_tensor((HW,), Float32),
            cute.make_rmem_tensor((HW,), Float32),
            cute.make_rmem_tensor((12,), Uint32),
        )
        if cutlass.const_expr(len(self.regs) == 2):
            if warp < SOFTMAX_WARPS:
                cute.arch.warpgroup_reg_alloc(self.regs[0])
            else:
                cute.arch.warpgroup_reg_dealloc(self.regs[1])
        for pw in cutlass.range_constexpr(self.loaders):
            if warp == SOFTMAX_WARPS + 1 + pw:
                if cutlass.const_expr(self.tma):
                    self.loader_tma(base, pw, lane, cta, n_mine,
                                    tensormap.map_address(qn), tensormap.map_address(qr),
                                    tensormap.map_address(kvm), tensormap.map_address(kvr),
                                    slots, slot_stride, lengths, stamp)
                else:
                    self.loader(base, pw, lane, cta, n_mine, tensormap.map_address(qn),
                                tensormap.map_address(qr), kv, unit_rows, slots,
                                slot_stride, lengths)
        if warp == SOFTMAX_WARPS:
            self.mma(base, lane, taddr, n_mine, lengths, cta, stamp)
        if warp < SOFTMAX_WARPS:
            self.softmax(base, tid, warp, lane, taddr, n_mine, lengths, cta, out,
                         scale_log2, out_scale, regs, stamp)
        tc_fence_before()
        cute.arch.sync_threads()
        if warp == 0:
            tc_fence_after()
            if cutlass.const_expr(self.pipelined):
                tmem_free_exclusive(taddr)
            else:
                cute.arch.dealloc_tmem(
                    cute.make_ptr(Float32, taddr, cute.AddressSpace.tmem), 512)

    @cute.jit
    def stamp(self, ptr, cta, g, ev: cutlass.Constexpr):
        """clock64 of event ``ev`` for tile ``g`` of CTA ``cta`` (first CTAs/tiles)."""
        if cta < STAMP_CTAS and g < STAMP_TILES:
            stamp_clock(ptr + Int64(((cta * STAMP_TILES + g) * STAMP_EV + ev) * 8))

    # ------------------------------------------------------------------ loaders
    @cute.jit
    def loader(self, base, pw: cutlass.Constexpr, lane, cta, n_mine, tm_qn, tm_qr, kv,
               unit_rows, slots, slot_stride, lengths):
        nl: cutlass.Constexpr = self.loaders
        per: cutlass.Constexpr = (128 - pw + nl - 1) // nl  # tile rows of this warp
        batches: cutlass.Constexpr = (per + 31) // 32
        lptr = cute.make_ptr(Int32, lengths, cute.AddressSpace.gmem, assumed_align=4)
        lens = cute.make_tensor(lptr, cute.make_layout(1 << 30))
        g = Int32(0)
        for qi in range(n_mine):
            row = cta + qi * self.grid
            n = lens[row]
            tiles = _tiles(n)
            if cutlass.const_expr(pw == 0):
                mbar_wait(_bar(base, B_QEMPTY), Uint32((qi & 1) ^ 1))
                if lane == 0:
                    mbar_arrive_tx(_bar(base, B_QFULL), Uint32(4 * H * 128 + H * 64))
                    for j in cutlass.range_constexpr(4):
                        tma_tile2d(base + Uint32(OFF_QN + j * H * 128), tm_qn,
                                   _bar(base, B_QFULL), Int32(128 * j), row * H)
                    tma_tile2d(base + Uint32(OFF_QR), tm_qr, _bar(base, B_QFULL),
                               Int32(0), row * H)
            srow = slots + Int64(row) * Int64(slot_stride) * Int64(4)
            for t in range(tiles):
                s = g % self.stages
                st = base + Uint32(OFF_KV) + Uint32(s) * Uint32(STAGE)
                mbar_wait(_bar(base, B_KVEMPTY + s), Uint32(((g // self.stages) & 1) ^ 1))
                # warp pw copies tile rows pw + nl * m (m < per): one row per warp
                # instruction (lane = 16-B latent chunk; lanes 0-3 the RoPE chunks).
                j = lane // 8
                cc = Uint32(lane % 8)
                for bt in cutlass.range_constexpr(batches):
                    rl = pw + nl * (32 * bt + lane)
                    key = t * 128 + rl
                    slot = ld_global_s32_pred(srow + Int64(key) * Int64(4),
                                              Int32(rl < 128) & Int32(key < n))
                    off = (Int64(slot >> 6) * Int64(unit_rows) + Int64(slot & 63)) * Int64(ROW)
                    if slot < 0:
                        off = Int64(-1) << Int64(40)
                    lo = Int32(off & Int64(0xFFFFFFFF))
                    hi = Int32(off >> Int64(32))
                    for i0 in cutlass.range_constexpr(0, 32, 4):
                        ms = [32 * bt + i0 + k for k in range(4)]
                        if cutlass.const_expr(ms[0] < per):
                            # rows past the warp's share repeat its first row of the
                            # group (an idempotent copy)
                            lanes4 = [i0 + k if ms[k] < per else i0 for k in range(4)]
                            ms = [x if x < per else ms[0] for x in ms]
                            dsts = []
                            rds = []
                            for x in ms:
                                r = pw + nl * x
                                dsts.append(st + Uint32(j * 16384) + Uint32(r * 128)
                                            + ((cc ^ Uint32(r % 8)) << Uint32(4)))
                                rds.append(st + Uint32(4 * 16384) + Uint32(r * 64)
                                           + ((Uint32(lane % 4) ^ Uint32((r >> 1) % 4))
                                              << Uint32(4)))
                            cp_rows4(lo, hi, lanes4, dsts, kv + Int64(lane * 16), rds,
                                     lane)
                cp_async_arrive(_bar(base, B_KVFULL + s))
                g = g + 1

    @cute.jit
    def loader_tma(self, base, pw: cutlass.Constexpr, lane, cta, n_mine, tm_qn, tm_qr,
                   tm_kv, tm_kr, slots, slot_stride, lengths, stamp):
        """Lane m of loader warp pw gathers 4-row quad pw + nl * m of each tile: four
        128 B latent boxes (SW128) and the 64 B RoPE box (SW64) by tile::gather4;
        rows past the valid count read row -1 (TMA zero fill)."""
        nl: cutlass.Constexpr = self.loaders
        quads: cutlass.Constexpr = (32 - pw + nl - 1) // nl
        lptr = cute.make_ptr(Int32, lengths, cute.AddressSpace.gmem, assumed_align=4)
        lens = cute.make_tensor(lptr, cute.make_layout(1 << 30))
        g = Int32(0)
        for qi in range(n_mine):
            row = cta + qi * self.grid
            n = lens[row]
            tiles = _tiles(n)
            if cutlass.const_expr(pw == 0):
                mbar_wait(_bar(base, B_QEMPTY), Uint32((qi & 1) ^ 1))
                if lane == 0:
                    mbar_arrive_tx(_bar(base, B_QFULL), Uint32(4 * H * 128 + H * 64))
                    for j in cutlass.range_constexpr(4):
                        tma_tile2d(base + Uint32(OFF_QN + j * H * 128), tm_qn,
                                   _bar(base, B_QFULL), Int32(128 * j), row * H)
                    tma_tile2d(base + Uint32(OFF_QR), tm_qr, _bar(base, B_QFULL),
                               Int32(0), row * H)
            srow = slots + Int64(row) * Int64(slot_stride) * Int64(4)
            for t in range(tiles):
                s = g % self.stages
                st = base + Uint32(OFF_KV) + Uint32(s) * Uint32(STAGE)
                full = _bar(base, B_KVFULL + s)
                if lane == 0:
                    mbar_wait(_bar(base, B_KVEMPTY + s), Uint32(((g // self.stages) & 1) ^ 1))
                    if cutlass.const_expr(self.stamps and pw == 0):
                        self.stamp(stamp, cta, g, 0)
                    if cutlass.const_expr(pw == 0):
                        mbar_arrive_tx(full, Uint32(STAGE))
                cute.arch.sync_warp()
                if lane < quads:
                    q = pw + nl * lane
                    key = t * 128 + 4 * q
                    r = ld_global_v4_s32(srow + Int64(key) * Int64(4))
                    r0, r1, r2, r3 = r[0], r[1], r[2], r[3]
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
            n = lens[row]
            tiles = _tiles(n)
            mbar_wait(_bar(base, B_QFULL), Uint32(qi & 1))
            tc_fence_after()
            if cutlass.const_expr(self.pipelined):
                # QK(0), then QK(t + 1) before PV(t): softmax(t + 1) overlaps PV(t)
                self.qk(base, lane, taddr, g, 0, tiles, cta, stamp)
                for t in range(tiles):
                    if t + 1 < tiles:
                        self.qk(base, lane, taddr, g + 1, t + 1, tiles, cta, stamp)
                    self.pv(base, lane, taddr, g, t, tiles, qi, cta, stamp)
                    g = g + 1
            else:
                for t in range(tiles):
                    self.qk(base, lane, taddr, g, t, tiles, cta, stamp)
                    self.pv(base, lane, taddr, g, t, tiles, qi, cta, stamp)
                    g = g + 1

    @cute.jit
    def qk(self, base, lane, taddr, g, t, tiles, cta, stamp):
        """S^T(tile g) = K Q^T into its S^T buffer; frees Q after the row's last."""
        s = g % self.stages
        st = base + Uint32(OFF_KV) + Uint32(s) * Uint32(STAGE)
        mbar_wait(_bar(base, B_KVFULL + s), Uint32((g // self.stages) & 1))
        if cutlass.const_expr(self.stamps):
            if lane == 0:
                self.stamp(stamp, cta, g, 1)
        fence_async_smem()
        tc_fence_after()
        cs = Uint32(C_S)
        sfull = _bar(base, B_SFULL)
        if cutlass.const_expr(self.pipelined):
            cs = Uint32(C_S) + Uint32(96) * Uint32(g % 2)
            if g % 2 == 1:
                sfull = _bar(base, B_SFULL2)
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
            tc_commit(sfull)
            if t == tiles - 1:
                tc_commit(_bar(base, B_QEMPTY))
        cute.arch.sync_warp()

    @cute.jit
    def pv(self, base, lane, taddr, g, t, tiles, qi, cta, stamp):
        """O^T += V^T P^T(tile g); frees the KV stage (and P^T buffer)."""
        s = g % self.stages
        st = base + Uint32(OFF_KV) + Uint32(s) * Uint32(STAGE)
        if t == 0:
            mbar_wait(_bar(base, B_OFREE), Uint32((qi & 1) ^ 1))
        pfull = _bar(base, B_PFULL)
        pbuf = base + Uint32(OFF_P)
        if cutlass.const_expr(self.pipelined):
            mbar_wait_sel = g % 2
            if mbar_wait_sel == 1:
                pfull = _bar(base, B_PFULL2)
                pbuf = base + Uint32(OFF_P + H * 128)
            mbar_wait(pfull, Uint32((g // 2) & 1))
        else:
            mbar_wait(pfull, Uint32(g & 1))
        tc_fence_after()
        if cutlass.const_expr(self.stamps):
            if lane == 0:
                self.stamp(stamp, cta, g, 4)
        if lane == 0:
            bd0 = Uint64(P_DESC) | Uint64(pbuf >> Uint32(4))
            for b in cutlass.range_constexpr(4):
                ad0 = Uint64(SDESC_SW128) | Uint64((st + Uint32(b * 16384)) >> Uint32(4))
                for kk in cutlass.range_constexpr(2):
                    acc = Uint32(1)
                    if t == 0 and kk == 0:
                        acc = Uint32(0)
                    # K64 = 64 keys: V^T +8 KB (8 x 1024-B row groups),
                    # P^T +1 KB (8 x 128-B core matrices along K)
                    tc_mma(taddr + Uint32(self.c_o + 96 * b), ad0 + Uint64(kk * 512),
                           bd0 + Uint64(kk * 64), Uint32(IDESC_PV), acc)
            tc_commit(_bar(base, B_KVEMPTY + s))
            if cutlass.const_expr(self.pipelined):
                tc_commit(_bar(base, B_PVDONE))
            if t == tiles - 1:
                tc_commit(_bar(base, B_OFULL))
        cute.arch.sync_warp()

    # ------------------------------------------------------------------ softmax
    @cute.jit
    def softmax(self, base, tid, warp, lane, taddr, n_mine, lengths, cta, out,
                scale_log2, out_scale, regs, stamp):
        lptr = cute.make_ptr(Int32, lengths, cute.AddressSpace.gmem, assumed_align=4)
        lens = cute.make_tensor(lptr, cute.make_layout(1 << 30))
        wg = warp // 4
        r = tid % 128
        wq = warp % 4
        h0 = wg * HW
        lane_addr = taddr + (Uint32(wq * 32) << Uint32(16))
        g = Int32(0)
        l, sv, inv, words = regs
        for qi in range(n_mine):
            row = cta + qi * self.grid
            n = lens[row]
            tiles = _tiles(n)
            l.fill(0.0)
            if r < HW:
                st_shared_f32(base + Uint32(OFF_M + (wg * HW + r) * 4), Float32(-1e30))
            named_bar(1 + wg, 128)
            for t in range(tiles):
                cs = Uint32(C_S + h0)
                pbuf = base + Uint32(OFF_P)
                pfull = _bar(base, B_PFULL)
                if cutlass.const_expr(self.pipelined):
                    sfull = _bar(base, B_SFULL)
                    if g % 2 == 1:
                        sfull = _bar(base, B_SFULL2)
                        cs = Uint32(C_S + 96 + h0)
                        pbuf = base + Uint32(OFF_P + H * 128)
                        pfull = _bar(base, B_PFULL2)
                    mbar_wait(sfull, Uint32((g // 2) & 1))
                else:
                    mbar_wait(_bar(base, B_SFULL), Uint32(g & 1))
                if cutlass.const_expr(self.stamps):
                    if tid == 0:
                        self.stamp(stamp, cta, g, 2)
                tc_fence_after()
                a = tmem_ld(lane_addr + cs, 32)
                b16 = tmem_ld(lane_addr + cs + Uint32(32), 16)
                for k in cutlass.range_constexpr(32):
                    sv[k] = a[k] * scale_log2
                for k in cutlass.range_constexpr(16):
                    sv[32 + k] = b16[k] * scale_log2
                if t * 128 + r >= n:
                    for k in cutlass.range_constexpr(HW):
                        sv[k] = Float32(-math.inf)
                # fast path: no head's max grows by more than THR -> alpha = 1
                grow = Int32(0)
                for k4 in cutlass.range_constexpr(HW // 4):
                    m4 = ld_shared_v4f(base + Uint32(OFF_M + (wg * HW + 4 * k4) * 4))
                    for u in cutlass.range_constexpr(4):
                        if sv[4 * k4 + u] > m4[u] + Float32(self.thr):
                            grow = Int32(1)
                anyg = cute.arch.vote_any_sync(grow != 0)
                alb = base + Uint32(OFF_AL) + Uint32(g % 2) * Uint32(2 * HW * 4)
                if anyg and lane == 0:
                    st_shared_u32(base + Uint32(OFF_FLAG + wg * 4), Uint32(g + 1))
                if r < HW:
                    st_shared_f32(alb + Uint32((wg * HW + r) * 4), Float32(1.0))
                named_bar(1 + wg, 128)
                slow = ld_shared_u32(base + Uint32(OFF_FLAG + wg * 4)) == Uint32(g + 1)
                if slow:
                    for k in cutlass.range_constexpr(HW):
                        mx = redux_max_f32(sv[k])
                        if lane == 0:
                            st_shared_f32(
                                base + Uint32(OFF_RED + ((wg * 4 + wq) * HW + k) * 4), mx)
                    named_bar(1 + wg, 128)
                    if r < HW:
                        mt = ld_shared_f32(base + Uint32(OFF_RED + ((wg * 4) * HW + r) * 4))
                        for w in cutlass.range_constexpr(1, 4):
                            mt = cute.arch.fmax(mt, ld_shared_f32(
                                base + Uint32(OFF_RED + ((wg * 4 + w) * HW + r) * 4)))
                        mo = ld_shared_f32(base + Uint32(OFF_M + (wg * HW + r) * 4))
                        if mt > mo + Float32(self.thr):
                            st_shared_f32(alb + Uint32((wg * HW + r) * 4), ex2(mo - mt))
                            st_shared_f32(base + Uint32(OFF_M + (wg * HW + r) * 4), mt)
                    named_bar(1 + wg, 128)
                # O^T rescale (PV(t-1) finished before QK(t): in-order tensor pipe)
                if t > 0 and slow:
                    if cutlass.const_expr(self.pipelined):
                        # PV(t - 1) runs after QK(t) here: wait for it
                        mbar_wait(_bar(base, B_PVDONE), Uint32((g - 1) & 1))
                        tc_fence_after()
                    for b in cutlass.range_constexpr(4):
                        for half in cutlass.range_constexpr(3):
                            col = lane_addr + Uint32(self.c_o + 96 * b + h0 + 16 * half)
                            oa = tmem_ld(col, 16)
                            al4 = [ld_shared_v4f(alb + Uint32((wg * HW + 16 * half + 4 * q) * 4))
                                   for q in range(4)]
                            tmem_st(col, [oa[k] * al4[k // 4][k % 4] for k in range(16)])
                # P (x 2^P_PRE) -> E4M3; l += the E4M3 P
                for i in cutlass.range_constexpr(12):
                    k = 4 * i
                    m4 = ld_shared_v4f(base + Uint32(OFF_M + (wg * HW + k) * 4))
                    a4 = ld_shared_v4f(alb + Uint32((wg * HW + k) * 4))
                    w0 = cvt_e4m3x2(ex2(sv[k] - m4[0] + Float32(self.p_pre)),
                                    ex2(sv[k + 1] - m4[1] + Float32(self.p_pre)))
                    w1 = cvt_e4m3x2(ex2(sv[k + 2] - m4[2] + Float32(self.p_pre)),
                                    ex2(sv[k + 3] - m4[3] + Float32(self.p_pre)))
                    q0 = e4m3x2_roundtrip(w0)
                    q1 = e4m3x2_roundtrip(w1)
                    l[k] = l[k] * a4[0] + q0[0]
                    l[k + 1] = l[k + 1] * a4[1] + q0[1]
                    l[k + 2] = l[k + 2] * a4[2] + q1[0]
                    l[k + 3] = l[k + 3] * a4[3] + q1[1]
                    words[i] = w0 | (w1 << Uint32(16))
                for c in cutlass.range_constexpr(3):
                    cn = 3 * wg + c
                    st_shared_v4(pbuf + Uint32(cn * 2048)
                                 + Uint32((r // 8) * 128 + (r % 8) * 16),
                                 words[4 * c], words[4 * c + 1], words[4 * c + 2],
                                 words[4 * c + 3])
                fence_async_smem()
                tc_fence_before()
                cute.arch.sync_warp()
                if cutlass.const_expr(self.stamps):
                    if tid == 0:
                        self.stamp(stamp, cta, g, 3)
                if lane == 0:
                    mbar_arrive(pfull)
                g = g + 1
            # ---- epilogue: O = O^T * out_scale / l
            for k in cutlass.range_constexpr(HW):
                x = l[k]
                for sh in cutlass.range_constexpr(5):
                    x = x + cute.arch.shuffle_sync_bfly(x, 16 >> sh)
                if lane == 0:
                    st_shared_f32(base + Uint32(OFF_L + ((wg * 4 + wq) * HW + k) * 4), x)
            named_bar(1 + wg, 128)
            if r < HW:
                lt = Float32(0.0)
                for w in cutlass.range_constexpr(4):
                    lt = lt + ld_shared_f32(base + Uint32(OFF_L + ((wg * 4 + w) * HW + r) * 4))
                iv = Float32(0.0)
                if lt > Float32(0.0):
                    iv = out_scale / lt
                st_shared_f32(base + Uint32(OFF_MT + (wg * HW + r) * 4), iv)
            named_bar(1 + wg, 128)
            mbar_wait(_bar(base, B_OFULL), Uint32(qi & 1))
            if cutlass.const_expr(self.stamps):
                if tid == 0:
                    self.stamp(stamp, cta, g - 1, 5)
            tc_fence_after()
            for k4 in cutlass.range_constexpr(HW // 4):
                v4 = ld_shared_v4f(base + Uint32(OFF_MT + (wg * HW + 4 * k4) * 4))
                for u in cutlass.range_constexpr(4):
                    inv[4 * k4 + u] = v4[u]
            orow = out + Int64(row) * Int64(H * D_NOPE * 2)
            for b in cutlass.range_constexpr(4):
                oa = tmem_ld(lane_addr + Uint32(self.c_o + 96 * b + h0), 32)
                ob = tmem_ld(lane_addr + Uint32(self.c_o + 96 * b + h0 + 32), 16)
                for k in cutlass.range_constexpr(HW):
                    v = oa[k] if k < 32 else ob[k - 32]
                    st_global_b16(orow + Int64(((h0 + k) * D_NOPE + 128 * b + r) * 2),
                                  v * inv[k])
            tc_fence_before()
            cute.arch.sync_warp()
            if cutlass.const_expr(self.stamps):
                if tid == 0:
                    self.stamp(stamp, cta, g - 1, 6)
            if lane == 0:
                mbar_arrive(_bar(base, B_OFREE))


# Per-shape dispatch: calls of at least MIN_ROWS rows (prefill chunks) run the h8
# persistent kernel; smaller ones (decode, short chunks) keep the stock kernel.
MIN_ROWS = 1024
LOADERS = 3
PIPELINED = False
TMA = False
REGS: tuple = ()


def supported(num_heads: int, kv_cache_dtype: str, kv_lora_rank: int, rope_dim: int) -> bool:
    """Whether a DCP-1, non-ShardQ FLASHINFER_MLA_SPARSE layer's prefill rows run
    here: 96 heads (Kimi K3.1 at TP1), the per-tensor FP8 MLA cache, D 512 + 64, on
    Rubin (sm_107)."""
    from vllm.platforms import current_platform

    return (
        num_heads == H
        and kv_cache_dtype in ("fp8", "fp8_e4m3")
        and (kv_lora_rank, rope_dim) == (D_NOPE, ROW - D_NOPE)
        and current_platform.is_cuda()
        and current_platform.is_device_capability(107)
    )


@cache
def _compiled(loaders: int, grid: int, pipelined: bool, tma: bool, regs: tuple,
              shift: tuple, stages: int, stamps: bool):
    kernel = SparseMlaPrefill(loaders, grid, pipelined, tma, regs, shift, stages, stamps)
    args = (Int64(0), Int32(1), Int64(0), Int32(64), Int64(0), Int32(1), Int64(0),
            Int64(0), Float32(1.0), Float32(1.0), Int32(64), Int64(0), cuda.CUstream(0))
    opts = f"--ptxas-options -maxrregcount={min(255, 65536 // kernel.threads // 8 * 8)}"
    return cute.compile(kernel, *args, options=opts)


def precompile() -> None:
    """Compile the prefill-sized launch of the dispatched kernel (calls of at least
    MIN_ROWS rows run one CTA per SM) on the current device."""
    from vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla_h8 import _compiled

    sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    _compiled(min(sms, MIN_ROWS), ())


def sparse_mla_prefill(q8: torch.Tensor, kv_rows: torch.Tensor, unit_rows: int,
                       slots: torch.Tensor, lengths: torch.Tensor, scale: float,
                       out_scale: float, out: torch.Tensor | None = None) -> torch.Tensor:
    """Prefill-sized DSA sparse MLA on Rubin (the dispatched kernel, see
    sparse_mla_h8). Same contract as sparse_mla_prefill_v1."""
    from vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla_h8 import (
        sparse_mla_prefill_h8,
    )

    return sparse_mla_prefill_h8(q8, kv_rows, slots, lengths, scale, out_scale,
                                 unit_rows=unit_rows, out=out)


def sparse_mla_prefill_logical(q8: torch.Tensor, kv_rows: torch.Tensor,
                               topk_indices: torch.Tensor, req_ids: torch.Tensor,
                               block_table: torch.Tensor, block_size: int,
                               block_stride_rows: int, scale: float,
                               out_scale: float) -> torch.Tensor:
    """``sparse_mla_prefill`` on the indexer's request-relative top-k: the index remap
    runs inside the kernel's gather (see sparse_mla_h8.sparse_mla_prefill_h8_logical)."""
    from vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla_h8 import (
        sparse_mla_prefill_h8_logical,
    )

    return sparse_mla_prefill_h8_logical(q8, kv_rows, topk_indices, req_ids, block_table,
                                         block_size, block_stride_rows, scale, out_scale)


def sparse_mla_prefill_v1(q8: torch.Tensor, kv_rows: torch.Tensor, unit_rows: int,
                       slots: torch.Tensor, lengths: torch.Tensor, scale: float,
                       out_scale: float, out: torch.Tensor | None = None,
                       loaders: int = LOADERS, pipelined: bool = PIPELINED,
                       tma: bool = TMA, regs: tuple = REGS,
                       shift: tuple = (P_PRE, THR), stages: int = 2,
                       stamps: torch.Tensor | None = None) -> torch.Tensor:
    """q8 [T, 96, 576] E4M3 (contiguous); kv_rows: the cache as uint8/E4M3 rows of
    576 B; slots [T, W] int32 physical slots (block * 64 + offset, -1 holes past the
    valid prefix), lengths [T] int32 valid counts; ``scale`` = softmax scale x q
    scale x k scale; ``out_scale`` = k scale. Returns BF16 [T, 96, 512]."""
    t = q8.shape[0]
    if q8.shape[1:] != (H, ROW) or not q8.is_contiguous():
        raise ValueError(f"sparse_mla_prefill_v1: q8 must be contiguous [T, {H}, {ROW}]")
    if out is None:
        out = torch.empty((t, H, D_NOPE), dtype=torch.bfloat16, device=q8.device)
    if t == 0:
        return out
    sms = torch.cuda.get_device_properties(q8.device).multi_processor_count
    grid = min(sms, t)
    fn = _compiled(loaders, grid, pipelined, tma, tuple(regs), tuple(shift), stages,
                   stamps is not None)
    stream = cuda.CUstream(torch.cuda.current_stream(q8.device).cuda_stream)
    fn(Int64(q8.data_ptr()), Int32(t), Int64(kv_rows.data_ptr()), Int32(unit_rows),
       Int64(slots.data_ptr()), Int32(slots.stride(0)), Int64(lengths.data_ptr()),
       Int64(out.data_ptr()), Float32(scale * LOG2E), Float32(out_scale), Int32(kv_rows.shape[0]),
       Int64(0 if stamps is None else stamps.data_ptr()), stream)
    return out
