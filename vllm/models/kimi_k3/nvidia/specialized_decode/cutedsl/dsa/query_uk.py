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
TMEM), and the epilogue (thread = row) rounds to BF16 as the bmm output, scales,
clamps and stores E4M3 straight into the query rows, and q_pe values 32 n .. + 31.
Same arithmetic as the stock bmm + pack (the K order of the 128-term dot products may
differ)."""

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, Uint32, Uint64

from ..common import tensormap
from ..common.primitives import asm, bits_float, div_rn
from .dcp_value import _mma_f16, _tma3d, _tmem_ld32
from .sparse_mla import (
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
STAGES = 2
THREADS = 256  # warp 0 TMA, warp 1 MMA, warps 4-7 epilogue (TMEM lane quadrants)
EPI = 4
# kind::f16: FP32 D, BF16 A and B (K-major), N = 256, M = 128
IDESC = (1 << 4) | (1 << 7) | (1 << 10) | ((256 >> 3) << 17) | ((128 >> 4) << 24)


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


@asm(Uint32, pure=True)
def _e4m3x4(a, b, c, d):
    """Four floats -> four E4M3 bytes (a lowest), round to nearest."""
    code = ("{ .reg .b16 l, h; cvt.rn.satfinite.e4m3x2.f32 l, $2, $1; "
            "cvt.rn.satfinite.e4m3x2.f32 h, $4, $3; mov.b32 $0, {l, h}; }")
    return code, "=r,f,f,f,f", (Float32(a), Float32(b), Float32(c), Float32(d))


@asm()
def _st_global4(addr, w0, w1, w2, w3):
    return "st.global.v4.b32 [$0], {$1, $2, $3, $4};", "l,r,r,r,r,~{memory}", (
        Int64(addr), Uint32(w0), Uint32(w1), Uint32(w2), Uint32(w3))


@asm()
def _ld_shared_v4(addr):
    return ("ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,r,~{memory}",
            (Uint32(addr),), (Uint32,) * 4)


@asm(Float32, pure=False)
def _ld_global_f32(addr):
    return "ld.global.f32 $0, [$1];", "=f,l", (Int64(addr),)


@asm(Float32, pure=True)
def _quant(x, inv):
    """The stock pack's x * (1 / scale) clamped to +-448 (x: a BF16 value; 0fC3E00000
    = -448)."""
    code = ("{ .reg .f32 y; mul.rn.f32 y, $1, $2; max.f32 y, y, 0fC3E00000; "
            "min.f32 $0, y, 0f43E00000; }")
    return code, "=f,f,f", (Float32(x), Float32(inv))


@asm()
def _ld_global_v4(addr):
    return ('ld.global.nc.v4.b32 {$0, $1, $2, $3}, [$4];', '=r,=r,=r,=r,l,~{memory}',
            (Int64(addr),), (Uint32,) * 4)


@asm(Int64, pure=False)
def _ld_global_s64(addr):
    return 'ld.global.nc.s64 $0, [$1];', '=l,l', (Int64(addr),)


@asm()
def _st_global2(addr, w0, w1):
    return 'st.global.v2.b32 [$0], {$1, $2};', 'l,r,r,~{memory}', (
        Int64(addr), Uint32(w0), Uint32(w1))


@asm(Float32, pure=True)
def _inv_rms512(total, eps):
    """The stock rms_norm's rsqrtf(sum / 512 + eps)."""
    code = ('{ .reg .f32 r; div.rn.f32 r, $1, 0f44000000; add.f32 r, r, $2; '
            'rsqrt.approx.f32 $0, r; }')
    return code, '=f,f,f', (Float32(total), Float32(eps))


class QueryUK:
    def __init__(self, rows, gathered, q_heads, q_stride, insert=None):
        self.rows, self.gathered = rows, gathered
        self.q_heads, self.q_stride = q_heads, q_stride
        self.tiles = -(-rows // 128)
        # (CTAs, kv_a row pitch, cache block stride, block size): the KV insert on
        # extra CTAs beside the 2 G projection CTAs, else None
        self.insert = insert

    @cute.jit
    def __call__(self, q: Int64, uk_t: Int64, out: Int64, scale: Int64, kva: Int64,
                 kvw: Int64, kscale: Int64, cache: Int64, slots: Int64, eps: Float32,
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
        extra = 0 if self.insert is None else self.insert[0]
        self.kernel(am, bm, q, out, scale, kva, kvw, kscale, cache, slots, eps).launch(
            grid=(2 * self.gathered + extra, 1, 1), block=(THREADS, 1, 1),
            stream=stream, use_pdl=True)

    @cute.kernel
    def kernel(self, am: tensormap.GridMap, bm: tensormap.GridMap, q: Int64,
               out: Int64, scale: Int64, kva: Int64, kvw: Int64, kscale: Int64,
               cache: Int64, slots: Int64, eps: Float32):
        R, P = self.rows, self.q_heads
        smem = cutlass.utils.SmemAllocator()
        base = Uint32(smem.allocate_array(
            Uint8, B_BYTES + STAGES * A_BYTES + 128, byte_alignment=1024).toint())
        b_smem = base
        a_smem = base + Uint32(B_BYTES)
        bars = a_smem + Uint32(STAGES * A_BYTES)
        b_full = bars  # then a_full[STAGES], a_free[STAGES], acc_full, acc_free
        a_full = bars + Uint32(8)
        a_free = a_full + Uint32(8 * STAGES)
        acc_full = a_free + Uint32(8 * STAGES)
        acc_free = acc_full + Uint32(8)
        slot = acc_free + Uint32(8)
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(tid // 32)
        lane = tid % 32
        if cta < 2 * self.gathered:
            h, n = cta // 2, cta % 2
            tm_a, tm_b = tensormap.map_address(am), tensormap.map_address(bm)
            if tid == 0:
                mbar_init(b_full, Uint32(1))
                for st in cutlass.range_constexpr(STAGES):
                    mbar_init(a_full + Uint32(8 * st), Uint32(1))
                    # the MMAs' commit and the 128 epilogue threads (q_pe read)
                    mbar_init(a_free + Uint32(8 * st), Uint32(129))
                mbar_init(acc_full, Uint32(1))
                mbar_init(acc_free, Uint32(128))
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
                        mbar_wait(a_free + Uint32(8) * Uint32(st),
                                  ((m // STAGES) & 1) ^ 1)
                        full = a_full + Uint32(8) * Uint32(st)
                        mbar_arrive_tx(full, Uint32(A_BYTES))
                        dst = a_smem + Uint32(A_BYTES) * Uint32(st)
                        for k in cutlass.range_constexpr(3):
                            _tma3d(dst + Uint32(k * A_BLOCK), tm_a, full,
                                   QK * h + 64 * k, 128 * m, 0)
            if warp == 1:
                if lane == 0:
                    mbar_wait(b_full, 0)
                    for m in range(self.tiles):
                        st = m % STAGES
                        mbar_wait(acc_free, (m & 1) ^ 1)
                        mbar_wait(a_full + Uint32(8) * Uint32(st), (m // STAGES) & 1)
                        tc_fence_after()
                        src = a_smem + Uint32(A_BYTES) * Uint32(st)
                        for k in cutlass.range_constexpr(2):
                            for kk in cutlass.range_constexpr(4):
                                ad = Uint64(SDESC_SW128) | Uint64(
                                    (src + Uint32(k * A_BLOCK + 32 * kk)) >> 4)
                                bd = Uint64(SDESC_SW128) | Uint64(
                                    (b_smem + Uint32(k * B_BYTES // 2 + 32 * kk)) >> 4)
                                _mma_f16(taddr, ad, bd, Uint32(IDESC),
                                         Uint32(1 if k + kk > 0 else 0))
                        tc_commit(a_free + Uint32(8) * Uint32(st))
                        tc_commit(acc_full)
            if (warp >= EPI) & (warp < EPI + 4):
                row = tid - 32 * EPI
                quad = Uint32(32 * (warp - EPI)) << Uint32(16)
                inv = div_rn(Float32(1), _ld_global_f32(scale))
                for m in range(self.tiles):
                    t = 128 * m + row
                    live = t < R
                    dst = out + (Int64(t) * P + Int64(h)) * ROW
                    # q_pe values 32 n .. + 31 (the A stage's third block, SW128: 16 B
                    # chunk j of row r at j ^ (r % 8)) -> bytes 512 + 32 n .. (the stock
                    # pack's arithmetic)
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
                        for c in cutlass.range_constexpr(2):  # 16 values per store
                            ws = []
                            for i in cutlass.range_constexpr(4):
                                w2 = words[2 * c + i // 2]
                                pair = w2[2 * (i % 2): 2 * (i % 2) + 2]
                                v = [
                                    _quant(bits_float(w & Uint32(0xFFFF0000) if s
                                                      else w << Uint32(16)), inv)
                                    for w in pair for s in (0, 1)
                                ]
                                ws.append(_e4m3x4(*v))
                            _st_global4(dst + LATENT + 32 * n + 16 * c, *ws)
                    mbar_wait(acc_full, m & 1)
                    tc_fence_after()
                    # (warp-collective TMEM loads)
                    for c4 in cutlass.range_constexpr(8):
                        v = _tmem_ld32(taddr + quad + Uint32(32 * c4))
                        if live:
                            for half in cutlass.range_constexpr(2):
                                ws = []
                                for i in cutlass.range_constexpr(4):
                                    x = [
                                        _quant(Float32(BFloat16(
                                            v[16 * half + 4 * i + j])),
                                               inv)
                                        for j in range(4)
                                    ]
                                    ws.append(_e4m3x4(*x))
                                col = 256 * n + 32 * c4 + 16 * half
                                _st_global4(dst + col, *ws)
                    tc_fence_before()
                    mbar_arrive(acc_free)
            tc_fence_before()
            cute.arch.sync_threads()
            if warp == 0:
                tc_fence_after()
                _tmem_free(taddr)
        else:
            if cutlass.const_expr(self.insert is not None):
                self.kv_insert(cta - 2 * self.gathered, warp, lane, kva, kvw, kscale,
                               cache, slots, eps)


    @cute.jit
    def kv_insert(self, c, warp, lane, kva, kvw, kscale, cache, slots, eps):
        """Extra CTA c: this rank's KV rows into the paged FP8 cache, a warp per row
        (stock rms_norm + concat_and_cache_mla arithmetic: y = BF16(x rsqrt(mean(x^2)
        + eps) w), E4M3(y / k_scale) saturating, k_pe E4M3(k_pe / k_scale); the
        sum-of-squares order differs). Lane l owns latent values 16 l .. + 15 and,
        for l < 8, RoPE values 8 l .. + 7."""
        ctas, pitch, bstride, bsize = self.insert
        cute.arch.griddepcontrol_wait()
        ks = _ld_global_f32(kscale)
        wv = [_ld_global_v4(kvw + Int64(32 * lane + 16 * j)) for j in range(2)]
        for r in range(c * (THREADS // 32) + warp, self.rows, ctas * (THREADS // 32)):
            src = kva + Int64(r) * Int64(2 * pitch)
            xv = [_ld_global_v4(src + Int64(32 * lane + 16 * j)) for j in range(2)]
            pe = _ld_global_v4(src + Int64(2 * LATENT) + Int64(16 * (lane & 7)))
            slot = _ld_global_s64(slots + Int64(r) * Int64(8))
            x = []
            for j in cutlass.range_constexpr(2):
                for i in cutlass.range_constexpr(4):
                    x.append(bits_float(xv[j][i] << Uint32(16)))
                    x.append(bits_float(xv[j][i] & Uint32(0xFFFF0000)))
            ss = Float32(0.0)
            for i in cutlass.range_constexpr(16):
                ss = ss + x[i] * x[i]
            for sh in cutlass.range_constexpr(5):
                ss = ss + cute.arch.shuffle_sync_bfly(ss, 16 >> sh)
            s = _inv_rms512(ss, eps)
            ws = []
            for i in cutlass.range_constexpr(4):
                v = []
                for e in cutlass.range_constexpr(4):
                    k = 4 * i + e
                    wword = wv[k // 8][(k % 8) // 2]
                    wf = bits_float(wword & Uint32(0xFFFF0000) if k % 2
                                    else wword << Uint32(16))
                    y = Float32(BFloat16(x[k] * s * wf))
                    v.append(div_rn(y, ks))
                ws.append(_e4m3x4(*v))
            if slot >= Int64(0):
                dst = (cache + (slot // Int64(bsize)) * Int64(bstride)
                       + (slot % Int64(bsize)) * Int64(ROW))
                _st_global4(dst + Int64(16 * lane), *ws)
                if lane < 8:
                    pv = []
                    for i in cutlass.range_constexpr(4):
                        pv.append(div_rn(bits_float(pe[i] << Uint32(16)), ks))
                        pv.append(div_rn(bits_float(pe[i] & Uint32(0xFFFF0000)), ks))
                    _st_global2(dst + Int64(LATENT + 8 * lane), _e4m3x4(*pv[:4]),
                                _e4m3x4(*pv[4:]))

@cache
def _compiled(rows, gathered, q_heads, q_stride, device, insert=None):
    kernel = QueryUK(rows, gathered, q_heads, q_stride, insert)
    args = (Int64(0), Int64(0), Int64(0), Int64(0), Int64(0), Int64(0), Int64(0),
            Int64(0), Int64(0), Float32(0.0), cuda.CUstream(0))
    with torch.accelerator.device_index(device):
        return cute.compile(kernel, *args)


def _insert_ctas(device, gathered):
    """The SMs left beside the 2 G projection CTAs (at least 8)."""
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    return max(8, sms - 2 * gathered)


def query_uk(q, uk_t, scale, out, insert=None):
    """``q`` [R, G x 192] BF16 (q_b rows, row-major), ``uk_t`` [G, 512, 128] BF16
    (W_UK_T K-major), ``scale`` [1] FP32 (q_scale) -> ``out`` [R, P, 576] E4M3 heads
    0..G-1 = cat(q_nope W_UK_T, q_pe) / q_scale as the stock bmm + pack.

    ``insert`` = (kv_a [R, >= 576] BF16 (latent then k_pe columns), norm weight
    [512] BF16, eps, k_scale [1] FP32, cache [blocks, block size, 576] FP8, slots [R]
    int64): also the stock ``rms_norm`` + ``concat_and_cache_mla`` (fp8) of those
    rows, on extra CTAs of the same launch."""
    rows, width = q.shape
    gathered = uk_t.shape[0]
    if (width != gathered * QK or q.stride(1) != 1 or tuple(uk_t.shape[1:]) !=
            (LATENT, QK_NOPE) or not uk_t.is_contiguous() or not out.is_contiguous()
            or out.shape[1] < gathered or out.shape[2] != ROW
            or q.dtype != torch.bfloat16 or uk_t.dtype != torch.bfloat16):
        raise ValueError(
            f"query_uk: q {tuple(q.shape)} {q.dtype}, uk_t {tuple(uk_t.shape)}, "
            f"out {tuple(out.shape)}")
    ptrs, spec, eps = (0,) * 5, None, 0.0
    if insert is not None:
        kva, weight, eps, k_scale, cache, slots = insert
        if (kva.shape[0] < rows or kva.shape[1] < ROW or kva.stride(1) != 1
                or kva.stride(0) % 8 or kva.data_ptr() % 16
                or kva.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16
                or tuple(weight.shape) != (LATENT,) or not weight.is_contiguous()
                or cache.element_size() != 1 or tuple(cache.shape[2:]) != (ROW,)
                or cache.stride(1) != ROW or cache.stride(0) % 16
                or slots.dtype != torch.int64 or slots.shape[0] < rows
                or not slots.is_contiguous()):
            raise ValueError(
                f"query_uk insert: kv_a {tuple(kva.shape)} {kva.stride()}, cache "
                f"{tuple(cache.shape)} {cache.stride()}, slots {slots.dtype}")
        spec = (_insert_ctas(q.device, gathered), kva.stride(0), cache.stride(0),
                cache.shape[1])
        ptrs = (kva.data_ptr(), weight.data_ptr(), k_scale.data_ptr(),
                cache.data_ptr(), slots.data_ptr())
    fn = _compiled(rows, gathered, out.shape[1], q.stride(0), q.device.index, spec)
    stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
    fn(Int64(q.data_ptr()), Int64(uk_t.data_ptr()), Int64(out.data_ptr()),
       Int64(scale.data_ptr()), *(Int64(p) for p in ptrs), Float32(eps), stream)
    return out
