"""Prefill DSA query in one kernel: W_UK absorption + RoPE concat + static FP8 quant.

The stock MLA prefill path builds the sparse-MLA FP8 query in three passes
(``mla_attention``: ``torch.bmm(q_nope, W_UK_T)`` into the NoPE columns of a
BF16 ``(T, H, L + R)`` buffer, the RoPE part copied beside it, then
``_DecodeConcatQuantFP8`` / ``QuantFP8`` static per-tensor quant). ``UkQueryFp8``
writes the FP8 query in two passes: a batched (one batch per head) BF16 tcgen05
GEMM whose epilogue rounds each FP32 accumulator to BF16 (the bmm output dtype),
then quantizes exactly as ``QuantFP8`` (``x * (1 / scale)`` in FP32, clamp to the
E4M3 range, round to nearest even) straight into the NoPE columns, and a small
pass that quantizes the RoPE columns the same way. Bitwise equal to the stock
passes (the GEMM's FP32 accumulation order is the tensor-core order, as for any
GEMM backend). F16K (16383 rows x 96 heads): 178 vs 473 us per layer (1 GPU).
"""

import weakref

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
import torch
import triton
import triton.language as tl
from cutlass import Float32, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import make_ptr
from cutlass.cutlass_dsl import T, dsl_user_op

from vllm.logger import init_logger

from .dense_gemm_bf16_sm107 import PersistentDenseGemmKernel

logger = init_logger(__name__)

FP8_MAX = 448.0
# (mma tile M, N), (cluster M, N), TMA store, direct-store epilogue N, max A/B stages
DEFAULT_TACTIC = ((256, 256), (2, 1), True, 64, None)
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


class _UkQueryKernel(PersistentDenseGemmKernel):
    """C = FP8(BF16(A @ B) * inv_scale); ep_t0 = inv_scale (1 FP32)."""

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


@cute.jit
def _launch(
    gemm: cutlass.Constexpr,
    q_ptr: cute.Pointer,
    w_ptr: cute.Pointer,
    out_ptr: cute.Pointer,
    inv_ptr: cute.Pointer,
    m: cutlass.Int32,
    dims: cutlass.Constexpr,
    mac: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    h, p, lat, qs, os_, ws = dims
    # A (M, K=P, H): q[:, h, :P]; B (N=L, K=P, H): W_UK_T[h] (its strides: the
    # layer's W_UK_T is a permuted view, K-major); C (M, L, H).
    a = cute.make_tensor(q_ptr, cute.make_layout((m, p, h), stride=(h * qs, 1, qs)))
    b = cute.make_tensor(w_ptr, cute.make_layout((lat, p, h), stride=(ws[2], ws[1], ws[0])))
    c = cute.make_tensor(out_ptr, cute.make_layout((m, lat, h), stride=(h * os_, 1, os_)))
    inv = cute.make_tensor(inv_ptr, cute.make_layout((1,)))
    gemm(a, b, c, mac, stream, lambda x: x, inv, None, None)


@triton.jit
def _rope_fp8_kernel(
    q_ptr, out_ptr, inv_ptr, n_rows,
    D: tl.constexpr, P: tl.constexpr, R: tl.constexpr, OD: tl.constexpr, LAT: tl.constexpr,
    BLOCK: tl.constexpr, UNIT: tl.constexpr,
):
    # RoPE columns of the FP8 query: QuantFP8 static (x * (1 / scale), clamp, RNE).
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    cols = tl.arange(0, R)
    mask = rows[:, None] < n_rows
    x = tl.load(q_ptr + rows[:, None] * D + P + cols[None, :], mask=mask).to(tl.float32)
    if not UNIT:
        x = x * tl.load(inv_ptr)
    x = tl.minimum(tl.maximum(x, -448.0), 448.0)
    tl.store(out_ptr + rows[:, None] * OD + LAT + cols[None, :], x.to(tl.float8e4nv), mask=mask)


def _w_ok(w: torch.Tensor) -> bool:
    """W_UK_T (H, P, L) strides the GEMM's B operand takes: P or L contiguous, the
    rest 16-byte multiples."""
    s = w.stride()
    return (s[1] == 1 or s[2] == 1) and all(x == 1 or x % 8 == 0 for x in s) and (
        w.data_ptr() % 16 == 0
    )


def _ptr(dtype, t: torch.Tensor, align: int = 16):
    return make_ptr(dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=align)


class UkQueryFp8:
    """``out[t, h] = FP8([BF16(q[t, h, :P] @ W_UK_T[h]) | q[t, h, P:]] / q_scale)``.

    q: (T, H, P + R) BF16 with the RoPE part already applied (token stride H*(P+R),
    head stride P+R); w_uk_t: (H, P, L) BF16 contiguous; out: (T, H, L + R)
    float8_e4m3fn contiguous; inv_scale: (1,) FP32 = 1 / q_scale. unit_scale (a
    q_scale of exactly 1) skips the multiply. The GEMM writes the NoPE columns, a
    small pass the RoPE columns (``rope=False``: GEMM only, microbench).
    """

    def __init__(self, tactic=DEFAULT_TACTIC, unit_scale=False, rope=True, plain=False):
        self.tactic = tactic
        self.unit_scale = unit_scale
        self.rope = rope
        self.plain = plain  # microbench only

    def _compiled(self, dims, stream, args):
        key = (self.tactic, dims, self.unit_scale, self.plain)
        fn = _COMPILED.get(key)
        if fn is None:
            (tm, tn), cluster = self.tactic[:2]
            tma_store = self.tactic[2] if len(self.tactic) > 2 else True
            gemm = _UkQueryKernel(cutlass.Float32, tm == 256, (tm, tn), cluster, tma_store)
            gemm.direct_epi_n = self.tactic[3] if len(self.tactic) > 3 else 64
            gemm.max_ab_stage = self.tactic[4] if len(self.tactic) > 4 else None
            gemm.unit_scale = self.unit_scale
            gemm.plain = self.plain
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
        assert q.stride(1) == d and q.stride(0) == h * d and _w_ok(w_uk_t)
        assert q.data_ptr() % 16 == 0 and out.data_ptr() % 16 == 0 and d % 8 == 0
        assert out.is_contiguous() and q.dtype == torch.bfloat16
        assert out.dtype == torch.float8_e4m3fn and inv_scale.dtype == torch.float32
        torch_stream = torch.cuda.current_stream()
        stream = cuda.CUstream(torch_stream.cuda_stream)
        args = (
            _ptr(cutlass.BFloat16, q),
            _ptr(cutlass.BFloat16, w_uk_t),
            _ptr(cutlass.Float8E4M3FN, out),
            _ptr(cutlass.Float32, inv_scale, 4),
            cutlass.Int32(t),
        )
        dims = (h, p, lat, d, lat + r, tuple(w_uk_t.stride()))
        self._compiled(dims, stream, args)(*args, stream)
        if self.rope and r > 0:
            n = t * h
            block = 64
            _rope_fp8_kernel[(triton.cdiv(n, block),)](
                q, out, inv_scale, n, d, p, r, lat + r, lat, block, self.unit_scale,
                num_warps=4,
            )
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


_KERNEL = UkQueryFp8()
_UNIT_KERNEL = UkQueryFp8(unit_scale=True)
# q_scale tensor -> (1 / q_scale, q_scale == 1); weak keys: a freed scale tensor's
# address can be reused by another one.
_SCALES: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def fused_query(
    q: torch.Tensor, w_uk_t: torch.Tensor, q_scale: torch.Tensor
) -> torch.Tensor | None:
    """The prefill sparse-MLA FP8 query of ``q`` (T, H, P + R) or None when the
    shapes are outside this kernel (then the stock passes run)."""
    t, h, d = q.shape
    if (
        q.dtype != torch.bfloat16
        or w_uk_t.dtype != torch.bfloat16
        or w_uk_t.dim() != 3
        or w_uk_t.shape[0] != h
        or not _w_ok(w_uk_t)
        or q.stride() != (h * d, d, 1)
        or q.data_ptr() % 16
        or d % 8
        or w_uk_t.shape[2] % 16
        or q_scale.numel() != 1
    ):
        return None
    ent = _SCALES.get(q_scale)
    if ent is None:
        inv = q_scale.detach().float().reciprocal().reshape(1)
        # q_scale is a fixed per-layer constant: read once (outside capture).
        unit = not torch.cuda.is_current_stream_capturing() and bool(q_scale == 1)
        ent = _SCALES[q_scale] = (inv, unit)
    inv, unit = ent
    logger.info_once("Kimi K3.1 prefill: fused W_UK / RoPE / FP8 query (uk_query) active")
    out = torch.empty(
        (t, h, w_uk_t.shape[2] + d - w_uk_t.shape[1]),
        dtype=torch.float8_e4m3fn,
        device=q.device,
    )
    return (_UNIT_KERNEL if unit else _KERNEL)(q, w_uk_t, inv, out)
