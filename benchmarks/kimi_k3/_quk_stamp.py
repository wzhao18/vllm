# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LocalQuery's W_UK and query pack in one tcgen05 kernel (``dsa.local_query``).

Stock (and LocalQuery before): ``torch.bmm(q_nope [G, R, 128], W_UK_T [G, 128, 512])``
-> BF16 latent [G, R, 512], then ``pack_query``: cat(latent, q_pe) * (1 / q_scale),
clamped to +-448, E4M3 round to nearest -> query [R, P, 576]. The BF16 latent is
written and read back once (2 x 2 R G KB), the pack is a separate kernel.

Here CTA (h, n) of 2 G owns gathered head h's latent columns 256 n .. + 255 for every
row: its W_UK_T slice [256 x 128] (K-major, staged once) times the q_nope rows of
each 128-row tile (TMA, two stages) on tcgen05 (kind::f16, FP32 accumulation in
TMEM). The accumulator is two 128-column halves with their own full/free barriers
and epilogue warp sets (8 warps each, thread = row), so the MMAs of one half run under
the other half's epilogue. The epilogue rounds to BF16 as the bmm output, scales,
clamps and stores E4M3 (32 B per store) straight into the query rows, and q_pe
values 32 n .. + 31.
Same arithmetic as the stock bmm + pack (the K order of the 128-term dot products may
differ)."""

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, Uint32, Uint64

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common import tensormap
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.primitives import asm, div_rn
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.dcp_value import _mma_f16, _tma3d, _tmem_ld32
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.sparse_mla import (
    SDESC_SW128,
    fence_mbar_init,
    ld_shared_u32,
    mbar_arrive,
    mbar_arrive_tx,
    mbar_init,
    mbar_wait,
    tc_commit,
    tc_fence_after,
    tc_fence_before,
)

QK_NOPE, QK_ROPE, LATENT, QK = 128, 64, 512, 192
ROW = LATENT + QK_ROPE  # 576 FP8 bytes per query head
A_BLOCK = 128 * 64 * 2  # [128 rows x 64] BF16, SW128
A_BYTES = 3 * A_BLOCK  # a row tile's q_nope (two K blocks) and q_pe
B_BYTES = 256 * 128 * 2  # W_UK_T slice [256 x 128] BF16 (two SW128 K blocks)
B_HALF = 128 * 128  # one 128-column half's rows inside a B K block
STAGES = 2
# warp 0 TMA, warp 1 MMA, warps 4-11 the epilogue of accumulator half 0 (warps 4-7
# its columns 0-63 and the q_pe, 8-11 columns 64-127), warps 12-19 of half 1 (each
# set of four covers the TMEM lane quadrants)
THREADS = 640
EPI = 4
# kind::f16: FP32 D, BF16 A and B (K-major), N = 128 (one accumulator half), M = 128
IDESC = (1 << 4) | (1 << 7) | (1 << 10) | ((128 >> 3) << 17) | ((128 >> 4) << 24)


@asm()
def _tmem_alloc(dst):
    code = (
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [$0], 256; "
        "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"
    )
    return code, "r,~{memory}", (Uint32(dst),)


@asm()
def _tmem_free(taddr):
    return ("tcgen05.dealloc.cta_group::1.sync.aligned.b32 $0, 256;", "r,~{memory}",
            (Uint32(taddr),))


# The stock pack's E4M3 of x * (1 / scale) clamped to +-448, round to nearest, for four
# values -> four bytes (the first lowest). The stock clamp is fmaxf(-448, fminf(y, 448)):
# min.f32 (NaN -> 448, as fminf) then the saturating convert (below -448 and -inf ->
# -448) give the same byte for every input.
_CLAMP = "mul.rn.f32 {v}, {v}, $5; min.f32 {v}, {v}, 0f43E00000; "

_PACK = ("cvt.rn.satfinite.e4m3x2.f32 l, f1, f0; cvt.rn.satfinite.e4m3x2.f32 h, f3, f2; "
         "mov.b32 $0, {l, h}; }")
_SPLIT = ("shl.b32 x0, p, 16; and.b32 x1, p, 0xFFFF0000; shl.b32 x2, r, 16; "
          "and.b32 x3, r, 0xFFFF0000; mov.b32 f0, x0; mov.b32 f1, x1; mov.b32 f2, x2; "
          "mov.b32 f3, x3; ")
_REGS = "{ .reg .b32 p, r, x0, x1, x2, x3; .reg .f32 f0, f1, f2, f3; .reg .b16 l, h; "


@asm(Uint32, pure=True)
def _q4_acc(a, b, c, d, inv):
    """Four FP32 accumulators -> BF16 (the bmm output, RN) -> the stock pack's E4M3."""
    code = (_REGS + "cvt.rn.bf16x2.f32 p, $2, $1; cvt.rn.bf16x2.f32 r, $4, $3; "
            + _SPLIT + "".join(_CLAMP.format(v=f"f{i}") for i in range(4)) + _PACK)
    return code, "=r,f,f,f,f,f", (Float32(a), Float32(b), Float32(c), Float32(d),
                                  Float32(inv))


@asm(Uint32, pure=True)
def _q4_bf16(w0, w1, inv):
    """Two BF16x2 words (q_pe, the lower element first) -> the stock pack's E4M3."""
    code = (_REGS + "mov.b32 p, $1; mov.b32 r, $2; " + _SPLIT
            + "".join(_CLAMP.format(v=f"f{i}").replace("$5", "$3") for i in range(4))
            + _PACK)
    return code, "=r,r,r,f", (Uint32(w0), Uint32(w1), Float32(inv))


@asm()
def _st_global8(addr, *w):
    return ("st.global.v8.b32 [$0], {$1, $2, $3, $4, $5, $6, $7, $8};",
            "l," + ",".join(["r"] * 8) + ",~{memory}",
            (Int64(addr), *(Uint32(x) for x in w)))


@asm()
def _ld_shared_v4(addr):
    return ("ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,r,~{memory}",
            (Uint32(addr),), (Uint32,) * 4)


@asm(Float32, pure=False)
def _ld_global_f32(addr):
    return "ld.global.f32 $0, [$1];", "=f,l", (Int64(addr),)


@asm()
def _stamp(addr):
    return ("{ .reg .u64 t; mov.u64 t, %globaltimer; st.global.u64 [$0], t; }",
            "l,~{memory}", (Int64(addr),))


class QueryUK:
    def __init__(self, rows, gathered, q_heads, q_stride):
        self.rows, self.gathered = rows, gathered
        self.q_heads, self.q_stride = q_heads, q_stride
        self.tiles = -(-rows // 128)

    @cute.jit
    def __call__(self, q: Int64, uk_t: Int64, out: Int64, scale: Int64,
                 stream: cuda.CUstream):
        bf = cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16
        sw = cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B
        l2 = cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B
        # A: q rows [R, G * 192]: [128 rows x 64] boxes at column 192 h + 64 k
        am = tensormap.encode_map(q, bf, (self.gathered * QK, self.rows, 1),
                                  (self.q_stride * 2, self.rows * self.q_stride * 2),
                                  (64, 128, 1), sw, l2)
        # B: W_UK_T K-major [G, 512, 128]: [256 x 64] boxes
        bm = tensormap.encode_map(uk_t, bf, (QK_NOPE, LATENT, self.gathered),
                                  (QK_NOPE * 2, LATENT * QK_NOPE * 2), (64, 256, 1),
                                  sw, l2)
        self.kernel(am, bm, q, out, scale).launch(
            grid=(2 * self.gathered, 1, 1), block=(THREADS, 1, 1), stream=stream,
            use_pdl=True)

    @cute.kernel
    def kernel(self, am: tensormap.GridMap, bm: tensormap.GridMap, q: Int64,
               out: Int64, scale: Int64):
        R, P = self.rows, self.q_heads
        smem = cutlass.utils.SmemAllocator()
        base = Uint32(smem.allocate_array(
            Uint8, B_BYTES + STAGES * A_BYTES + 128, byte_alignment=1024).toint())
        b_smem = base
        a_smem = base + Uint32(B_BYTES)
        bars = a_smem + Uint32(STAGES * A_BYTES)
        # b_full, a_full[STAGES], a_free[STAGES], acc_full[2], acc_free[2], slot
        b_full = bars
        a_full = bars + Uint32(8)
        a_free = a_full + Uint32(8 * STAGES)
        acc_full = a_free + Uint32(8 * STAGES)
        acc_free = acc_full + Uint32(16)
        slot = acc_free + Uint32(16)
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(tid // 32)
        lane = tid % 32
        h, n = cta // 2, cta % 2
        if tid == 0:
            _stamp(scale + Int64(256) + (Int64(cta) * 32 + Int64(0)) * 8)
        tm_a, tm_b = tensormap.map_address(am), tensormap.map_address(bm)
        if tid == 0:
            mbar_init(b_full, Uint32(1))
            for st in cutlass.range_constexpr(STAGES):
                mbar_init(a_full + Uint32(8 * st), Uint32(1))
                # the MMAs' commit and the 128 q_pe epilogue threads (warps 4-7)
                mbar_init(a_free + Uint32(8 * st), Uint32(129))
            for c in cutlass.range_constexpr(2):
                mbar_init(acc_full + Uint32(8 * c), Uint32(1))
                mbar_init(acc_free + Uint32(8 * c), Uint32(256))
            fence_mbar_init()
        cute.arch.sync_threads()
        if warp == 0:
            if lane == 0:  # the weights do not depend on the predecessor
                mbar_arrive_tx(b_full, Uint32(B_BYTES))
                for k in cutlass.range_constexpr(2):
                    _tma3d(b_smem + Uint32(k * B_BYTES // 2), tm_b, b_full, 64 * k,
                           256 * n, h)
        # TMEM only once the predecessor grid is complete (dcp_value.kernel_tc).
        cute.arch.griddepcontrol_wait()
        if warp == 0:
            _tmem_alloc(slot)
        tc_fence_before()
        cute.arch.sync_threads()
        tc_fence_after()
        taddr = ld_shared_u32(slot)
        if warp == 0:
            if lane == 0:
                for m in range(self.tiles):
                    st = m % STAGES
                    mbar_wait(a_free + Uint32(8) * Uint32(st), ((m // STAGES) & 1) ^ 1)
                    full = a_full + Uint32(8) * Uint32(st)
                    mbar_arrive_tx(full, Uint32(A_BYTES))
                    dst = a_smem + Uint32(A_BYTES) * Uint32(st)
                    for k in cutlass.range_constexpr(3):
                        _tma3d(dst + Uint32(k * A_BLOCK), tm_a, full,
                               QK * h + 64 * k, 128 * m, 0)
        if warp == 1:
            if lane == 0:
                mbar_wait(b_full, 0)
                _stamp(scale + Int64(256) + (Int64(cta) * 32 + Int64(1)) * 8)
                for m in range(self.tiles):
                    st = m % STAGES
                    mbar_wait(a_full + Uint32(8) * Uint32(st), (m // STAGES) & 1)
                    _stamp(scale + Int64(256) + (Int64(cta) * 32 + Int64(2 + m)) * 8)
                    src = a_smem + Uint32(A_BYTES) * Uint32(st)
                    # accumulator half c (TMEM columns 128 c ..): latent columns
                    # 256 n + 128 c .. of the tile, its own full/free pair, so the
                    # MMAs of one half run under the other half's epilogue
                    for c in cutlass.range_constexpr(2):
                        mbar_wait(acc_free + Uint32(8 * c), (m & 1) ^ 1)
                        tc_fence_after()
                        for k in cutlass.range_constexpr(2):
                            for kk in cutlass.range_constexpr(4):
                                ad = Uint64(SDESC_SW128) | Uint64(
                                    (src + Uint32(k * A_BLOCK + 32 * kk)) >> 4)
                                bd = Uint64(SDESC_SW128) | Uint64(
                                    (b_smem + Uint32(k * B_BYTES // 2 + c * B_HALF
                                                     + 32 * kk)) >> 4)
                                _mma_f16(taddr + Uint32(128 * c), ad, bd, Uint32(IDESC),
                                         Uint32(1 if k + kk > 0 else 0))
                        tc_commit(acc_full + Uint32(8 * c))
                    tc_commit(a_free + Uint32(8) * Uint32(st))
        if warp >= EPI:
            e = (warp - EPI) // 4  # epilogue set: accumulator half e // 2
            g, sub = e // 2, e % 2  # columns 64 sub .. + 63 of the half
            qd = (warp - EPI) % 4  # TMEM lane quadrant
            row = 32 * qd + lane
            quad = Uint32(32 * qd) << Uint32(16)
            inv = div_rn(Float32(1), _ld_global_f32(scale))
            for m in range(self.tiles):
                t = 128 * m + row
                live = t < R
                dst = out + (Int64(t) * P + Int64(h)) * ROW
                if e == 0:
                    # q_pe values 32 n .. + 31 (the A stage's third block, SW128: 16 B
                    # chunk j of row r at j ^ (r % 8)) -> bytes 512 + 32 n ..
                    st = m % STAGES
                    mbar_wait(a_full + Uint32(8) * Uint32(st), (m // STAGES) & 1)
                    pe = (a_smem + Uint32(A_BYTES) * Uint32(st) + Uint32(2 * A_BLOCK)
                          + Uint32(128) * Uint32(row))
                    words = [
                        _ld_shared_v4(pe + Uint32(16) * Uint32((4 * n + i) ^ (row % 8)))
                        for i in range(4)
                    ]
                    mbar_arrive(a_free + Uint32(8) * Uint32(st))
                    if live:
                        ws = [
                            _q4_bf16(words[i // 2][2 * (i % 2)],
                                     words[i // 2][2 * (i % 2) + 1], inv)
                            for i in range(8)
                        ]
                        _st_global8(dst + LATENT + 32 * n, *ws)
                full = acc_full + Uint32(8) * Uint32(g)
                mbar_wait(full, m & 1)
                tc_fence_after()
                if tid % 128 == 0:
                    _stamp(scale + Int64(256) + (Int64(cta) * 32 + Int64(6 + 2 * m + g)) * 8)
                for c4 in cutlass.range_constexpr(2):  # (warp-collective TMEM loads)
                    v = _tmem_ld32(taddr + quad + Uint32(128) * Uint32(g)
                                   + Uint32(64) * Uint32(sub) + Uint32(32 * c4))
                    if live:
                        ws = [_q4_acc(v[4 * i], v[4 * i + 1], v[4 * i + 2], v[4 * i + 3],
                                      inv) for i in range(8)]
                        col = 256 * n + 128 * g + 64 * sub + 32 * c4
                        _st_global8(dst + col, *ws)
                tc_fence_before()
                mbar_arrive(acc_free + Uint32(8) * Uint32(g))
                if tid % 128 == 0:
                    _stamp(scale + Int64(256) + (Int64(cta) * 32 + Int64(14 + 2 * m + g)) * 8)
        tc_fence_before()
        cute.arch.sync_threads()
        if warp == 0:
            tc_fence_after()
            _tmem_free(taddr)
        if tid == 0:
            _stamp(scale + Int64(256) + (Int64(cta) * 32 + Int64(30)) * 8)


@cache
def _compiled(rows, gathered, q_heads, q_stride, device):
    kernel = QueryUK(rows, gathered, q_heads, q_stride)
    args = (Int64(0), Int64(0), Int64(0), Int64(0), cuda.CUstream(0))
    with torch.accelerator.device_index(device):
        return cute.compile(kernel, *args)


def query_uk(q, uk_t, scale, out):
    """``q`` [R, G x 192] BF16 (q_b rows, row-major), ``uk_t`` [G, 512, 128] BF16
    (W_UK_T K-major), ``scale`` [1] FP32 (q_scale) -> ``out`` [R, P, 576] E4M3 heads
    0..G-1 = cat(q_nope W_UK_T, q_pe) / q_scale as the stock bmm + pack."""
    rows, width = q.shape
    gathered = uk_t.shape[0]
    if (width != gathered * QK or q.stride(1) != 1 or tuple(uk_t.shape[1:]) !=
            (LATENT, QK_NOPE) or not uk_t.is_contiguous() or not out.is_contiguous()
            or out.shape[1] < gathered or out.shape[2] != ROW
            or q.dtype != torch.bfloat16 or uk_t.dtype != torch.bfloat16):
        raise ValueError(
            f"query_uk: q {tuple(q.shape)} {q.dtype}, uk_t {tuple(uk_t.shape)}, "
            f"out {tuple(out.shape)}")
    fn = _compiled(rows, gathered, out.shape[1], q.stride(0), q.device.index)
    stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
    fn(Int64(q.data_ptr()), Int64(uk_t.data_ptr()), Int64(out.data_ptr()),
       Int64(scale.data_ptr()), stream)
    return out
