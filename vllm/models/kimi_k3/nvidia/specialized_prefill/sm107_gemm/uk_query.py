"""Prefill DSA query in one kernel: W_UK absorption + RoPE concat + static FP8 quant.

The stock MLA prefill path builds the sparse-MLA FP8 query in three passes
(``mla_attention``: ``torch.bmm(q_nope, W_UK_T)`` into the NoPE columns of a
BF16 ``(T, H, L + R)`` buffer, the RoPE part copied beside it, then
``_DecodeConcatQuantFP8`` / ``QuantFP8`` static per-tensor quant). ``UkQueryFp8``
writes the FP8 query directly: a batched (one batch per head) BF16 tcgen05 GEMM
whose epilogue rounds each FP32 accumulator to BF16 (the bmm output dtype), then
quantizes exactly as ``QuantFP8`` (``x * (1 / scale)`` in FP32, clamp to the E4M3
range, round to nearest even); the CTAs of the first N tile also quantize the
RoPE columns of their rows. Same math as the three stock passes (the GEMM's
FP32 accumulation order is the tensor-core order, as for any GEMM backend).
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
import torch
from cutlass import Float32, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import make_ptr
from cutlass.cutlass_dsl import T, dsl_user_op

from .dense_gemm_bf16_sm107 import PersistentDenseGemmKernel

FP8_MAX = 448.0
# (mma tile M, N), (cluster M, N), TMA store, direct-store epilogue N, max A/B stages
DEFAULT_TACTIC = ((256, 256), (2, 1), False, 64, None)
_COMPILED: dict = {}


def _q8_ptx(from_f32: bool, scaled: bool) -> str:
    """4 values -> 4 E4M3 bytes (one b32). from_f32: inputs $1..$4 are FP32
    accumulators rounded to BF16 first (cvt.rn.bf16x2); else $1, $2 are two
    bf16x2 words. scaled: multiply by $last (1 / q_scale) in FP32. The E4M3
    conversion saturates to +-448 (= QuantFP8's clamp, then round to nearest even)."""
    s = "{\n.reg .b32 h0, h1, a0, a1, a2, a3;\n.reg .b16 p0, p1;\n"
    if from_f32:
        s += "cvt.rn.bf16x2.f32 h0, $2, $1;\ncvt.rn.bf16x2.f32 h1, $4, $3;\n"
        sc = "$5"
    else:
        s += "mov.b32 h0, $1;\nmov.b32 h1, $2;\n"
        sc = "$3"
    s += (
        "shl.b32 a0, h0, 16;\nand.b32 a1, h0, 0xFFFF0000;\n"
        "shl.b32 a2, h1, 16;\nand.b32 a3, h1, 0xFFFF0000;\n"
    )
    if scaled:
        s += (
            ".reg .b64 A, B, S;\nmov.b64 S, {%s, %s};\n" % (sc, sc)
            + "mov.b64 A, {a0, a1};\nmov.b64 B, {a2, a3};\n"
            "mul.rn.f32x2 A, A, S;\nmul.rn.f32x2 B, B, S;\n"
            "mov.b64 {a0, a1}, A;\nmov.b64 {a2, a3}, B;\n"
        )
    s += (
        "cvt.rn.satfinite.e4m3x2.f32 p0, a1, a0;\n"
        "cvt.rn.satfinite.e4m3x2.f32 p1, a3, a2;\n"
        "mov.b32 $0, {p0, p1};\n}"
    )
    return s


@dsl_user_op
def _q8_f32x4(x0, x1, x2, x3, inv, scaled: bool, *, loc=None, ip=None) -> Uint32:
    ops = [Float32(v).ir_value(loc=loc, ip=ip) for v in (x0, x1, x2, x3)]
    cons = "=r,f,f,f,f"
    if scaled:
        ops.append(Float32(inv).ir_value(loc=loc, ip=ip))
        cons += ",f"
    return Uint32(
        llvm.inline_asm(
            T.i32(), ops, _q8_ptx(True, scaled), cons,
            has_side_effects=False, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )


@dsl_user_op
def _q8_bf16x4(w0, w1, inv, scaled: bool, *, loc=None, ip=None) -> Uint32:
    ops = [Uint32(v).ir_value(loc=loc, ip=ip) for v in (w0, w1)]
    cons = "=r,r,r"
    if scaled:
        ops.append(Float32(inv).ir_value(loc=loc, ip=ip))
        cons += ",f"
    return Uint32(
        llvm.inline_asm(
            T.i32(), ops, _q8_ptx(False, scaled), cons,
            has_side_effects=False, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )


class _UkQueryKernel(PersistentDenseGemmKernel):
    """C = FP8(BF16(A @ B) * inv_scale); ep_t0 = inv_scale (1 FP32), ep_t1 / ep_t2 =
    RoPE part in (BF16) / out (FP8) as (row, 16 B chunk, 8, head)."""

    plain = False  # microbench only: plain FP32 -> FP8 conversion
    unit_scale = False  # q_scale == 1: x * (1 / q_scale) is x

    def epi_convert_regs(self, src, dst, epilogue_op, ep_t0):
        if cutlass.const_expr(self.plain):
            dst.store(src.load().to(self.c_dtype))
        else:
            inv = None if self.unit_scale else ep_t0[0]
            d32 = cute.recast_tensor(dst, Uint32)
            for i in range(cute.size(d32)):
                d32[i] = _q8_f32x4(
                    src[4 * i], src[4 * i + 1], src[4 * i + 2], src[4 * i + 3],
                    inv, not self.unit_scale,
                )

    def _rope_rows(self, epi_tidx, cta_tile_coord):
        lane = epi_tidx % 32
        m0 = cta_tile_coord[0] * self.cta_tile_shape_mnk[0] + (epi_tidx // 32) * 4 + lane // 8
        return m0, lane % 8

    def epi_tile_alloc(self, ep_t1):
        if cutlass.const_expr(ep_t1 is None):
            return None
        # (8 elements, rows per thread): one row's 16 B contiguous
        return cute.make_rmem_tensor((8, self.cta_tile_shape_mnk[0] // 16), cutlass.BFloat16)

    @cute.jit
    def epi_tile_prefetch(self, epi_tidx, tile_coord, valid, ep_t1, pf):
        # RoPE part of a tile's rows (CTAs of N tile 0), loaded one tile ahead:
        # 8 lanes per row (16 B each), 4 rows per warp per step. Branch-free row
        # clamp (rows past the end re-read the last row; their stores are skipped).
        if cutlass.const_expr(ep_t1 is not None):
            if valid:
                if tile_coord[1] == 0:
                    m0, chunk = self._rope_rows(epi_tidx, tile_coord)
                    rows = cute.size(ep_t1, mode=[0])
                    for it in cutlass.range_constexpr(self.cta_tile_shape_mnk[0] // 16):
                        r = cutlass.min(m0 + it * 16, rows - 1)
                        pf[(None, it)].store(ep_t1[(r, chunk, None, tile_coord[2])].load())

    @cute.jit
    def epi_tile_extra(self, epi_tidx, cta_tile_coord, ep_t0, ep_t1, ep_t2, pf):
        if cutlass.const_expr(ep_t1 is not None):
            self._rope_store(epi_tidx, cta_tile_coord, ep_t0, ep_t1, ep_t2, pf)

    @cute.jit
    def _rope_store(self, epi_tidx, cta_tile_coord, ep_t0, ep_t1, ep_t2, pf):
        n_it = self.cta_tile_shape_mnk[0] // 16
        m0, chunk = self._rope_rows(epi_tidx, cta_tile_coord)
        rows = cute.size(ep_t1, mode=[0])
        inv = None if self.unit_scale else ep_t0[0]
        w = cute.recast_tensor(pf, Uint32)
        if cta_tile_coord[1] == 0:
            for it in cutlass.range_constexpr(n_it):
                if m0 + it * 16 < rows:
                    o = cute.make_rmem_tensor((2,), Uint32)
                    o[0] = _q8_bf16x4(w[(0, it)], w[(1, it)], inv, not self.unit_scale)
                    o[1] = _q8_bf16x4(w[(2, it)], w[(3, it)], inv, not self.unit_scale)
                    dst = cute.recast_tensor(
                        ep_t2[(m0 + it * 16, chunk, None, cta_tile_coord[2])], Uint32
                    )
                    dst.store(o.load())


@cute.jit
def _launch(
    gemm: cutlass.Constexpr,
    q_ptr: cute.Pointer,
    w_ptr: cute.Pointer,
    out_ptr: cute.Pointer,
    inv_ptr: cute.Pointer,
    qpe_ptr: cute.Pointer,
    ope_ptr: cute.Pointer,
    m: cutlass.Int32,
    dims: cutlass.Constexpr,
    mac: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    h, p, r, lat, qs, os_ = dims
    # A (M, K=P, H): q[:, h, :P]; B (N=L, K=P, H): W_UK_T[h] (N-major); C (M, L, H).
    a = cute.make_tensor(q_ptr, cute.make_layout((m, p, h), stride=(h * qs, 1, qs)))
    b = cute.make_tensor(w_ptr, cute.make_layout((lat, p, h), stride=(1, lat, p * lat)))
    c = cute.make_tensor(out_ptr, cute.make_layout((m, lat, h), stride=(h * os_, 1, os_)))
    # RoPE part as (row, 16 B chunk, element, head)
    qpe = cute.make_tensor(
        qpe_ptr, cute.make_layout((m, r // 8, 8, h), stride=(h * qs, 8, 1, qs))
    )
    ope = cute.make_tensor(
        ope_ptr, cute.make_layout((m, r // 8, 8, h), stride=(h * os_, 8, 1, os_))
    )
    inv = cute.make_tensor(inv_ptr, cute.make_layout((1,)))
    if cutlass.const_expr(gemm.skip_rope):
        gemm(a, b, c, mac, stream, lambda x: x, inv, None, None)
    else:
        gemm(a, b, c, mac, stream, lambda x: x, inv, qpe, ope)


def _ptr(dtype, t: torch.Tensor, align: int = 16):
    return make_ptr(dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=align)


class UkQueryFp8:
    """``out[t, h] = FP8([BF16(q[t, h, :P] @ W_UK_T[h]) | q[t, h, P:]] / q_scale)``.

    q: (T, H, P + R) BF16 with the RoPE part already applied (token stride H*(P+R),
    head stride P+R); w_uk_t: (H, P, L) BF16 contiguous; out: (T, H, L + R)
    float8_e4m3fn contiguous; inv_scale: (1,) FP32 = 1 / q_scale.
    """

    def __init__(
        self, tactic=DEFAULT_TACTIC, plain=False, skip_rope=False, unit_scale=False,
        clamp=True,
    ):
        self.tactic = tactic
        self.unit_scale = unit_scale
        self.clamp = clamp
        self.plain = plain  # microbench arms only
        self.skip_rope = skip_rope

    def _compiled(self, dims, stream, args):
        key = (self.tactic, dims, self.plain, self.skip_rope, self.unit_scale, self.clamp)
        fn = _COMPILED.get(key)
        if fn is None:
            (tm, tn), cluster = self.tactic[:2]
            tma_store = self.tactic[2] if len(self.tactic) > 2 else True
            gemm = _UkQueryKernel(cutlass.Float32, tm == 256, (tm, tn), cluster, tma_store)
            gemm.direct_epi_n = self.tactic[3] if len(self.tactic) > 3 else 64
            gemm.max_ab_stage = self.tactic[4] if len(self.tactic) > 4 else None
            gemm.plain = self.plain
            gemm.unit_scale = self.unit_scale
            gemm.clamp = self.clamp
            gemm.skip_rope = self.skip_rope
            self.gemm = gemm
            mac = cutlass_utils.HardwareInfo().get_max_active_clusters(
                cluster[0] * cluster[1]
            )
            fn = cute.compile(_launch, gemm, *args, dims, mac, stream)
            _COMPILED[key] = fn
        return fn

    def __call__(self, q, w_uk_t, inv_scale, out):
        t, h, d = q.shape
        h2, p, lat = w_uk_t.shape
        r = d - p
        assert h2 == h and out.shape == (t, h, lat + r) and q.stride(2) == 1
        assert q.stride(1) == d and q.stride(0) == h * d and w_uk_t.is_contiguous()
        assert q.data_ptr() % 16 == 0 and out.data_ptr() % 16 == 0 and d % 8 == 0
        assert p % 8 == 0 and lat % 16 == 0 and r % 16 == 0
        assert out.is_contiguous() and q.dtype == torch.bfloat16
        assert out.dtype == torch.float8_e4m3fn and inv_scale.dtype == torch.float32
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        args = (
            _ptr(cutlass.BFloat16, q),
            _ptr(cutlass.BFloat16, w_uk_t),
            _ptr(cutlass.Float8E4M3FN, out),
            _ptr(cutlass.Float32, inv_scale, 4),
            # RoPE columns: their own 16-B aligned pointers (vector loads/stores)
            make_ptr(cutlass.BFloat16, q.data_ptr() + 2 * p, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Float8E4M3FN, out.data_ptr() + lat, cute.AddressSpace.gmem, assumed_align=16),
            cutlass.Int32(t),
        )
        dims = (h, p, r, lat, d, lat + r)
        fn = self._compiled(dims, stream, args)
        fn(*args, stream)
        return out


def reference(q, w_uk_t, q_scale, out_dtype=torch.float8_e4m3fn):
    """The stock three passes: bmm into the NoPE columns, RoPE copy, static quant."""
    t, h, d = q.shape
    p, lat = w_uk_t.shape[1], w_uk_t.shape[2]
    cat = q.new_empty((t, h, lat + d - p))
    torch.bmm(q[..., :p].transpose(0, 1), w_uk_t, out=cat[..., :lat].transpose(0, 1))
    cat[..., lat:].copy_(q[..., p:])
    x = cat.float() * q_scale.float().reciprocal()
    return x.clamp(-FP8_MAX, FP8_MAX).to(out_dtype)
