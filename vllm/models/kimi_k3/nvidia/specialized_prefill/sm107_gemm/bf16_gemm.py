"""BF16 GEMMs on the sm_107 CuTeDSL persistent base (dense_gemm_bf16_sm107).

``Bf16Gemm``: ``out = x @ w.T`` (x (M, K), w (N, K) row-major BF16, out (M, N) BF16
row-major) as the base computes it (FP32 accumulation, RNE to BF16). The prefill
epilogue fusions (e.g. the DSA output gate) subclass its kernel.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
import torch
from cutlass.cute.runtime import make_ptr

from .dense_gemm_bf16_sm107 import PersistentDenseGemmKernel

# (mma tile M, N), (cluster M, N), max A/B stages (None = fill smem)
DEFAULT_TACTIC = ((256, 256), (2, 1), None)
_COMPILED: dict = {}


@cute.jit
def _launch(
    gemm: cutlass.Constexpr,
    x_ptr: cute.Pointer,
    w_ptr: cute.Pointer,
    out_ptr: cute.Pointer,
    aux_ptr: cute.Pointer,
    m: cutlass.Int32,
    nk: cutlass.Constexpr,
    mac: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    n, k, ldx, ldo, ldaux = nk[:5]
    # L = 1: static zero batch strides (dynamic ones hide the 16-byte alignment)
    a = cute.make_tensor(x_ptr, cute.make_layout((m, k, 1), stride=(ldx, 1, 0)))
    b = cute.make_tensor(w_ptr, cute.make_layout((n, k, 1), stride=(k, 1, 0)))
    c = cute.make_tensor(out_ptr, cute.make_layout((m, n, 1), stride=(ldo, 1, 0)))
    if cutlass.const_expr(ldaux is None):
        gemm(a, b, c, mac, stream, lambda v: v, None, None, None)
    else:
        aux = cute.make_tensor(
            aux_ptr, cute.make_layout((m, n, 1), stride=(ldaux, 1, 0))
        )
        gemm(a, b, c, mac, stream, lambda v: v, None, aux, None)


def _ptr(t: torch.Tensor):
    return make_ptr(cutlass.BFloat16, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)


class Bf16Gemm:
    kernel_cls = PersistentDenseGemmKernel

    def __init__(self, tactic=DEFAULT_TACTIC):
        self.tactic = tactic

    def _fn(self, nk, stream, args):
        key = (type(self).__name__, self.tactic, nk)
        fn = _COMPILED.get(key)
        if fn is None:
            (tm, tn), cluster, max_ab = self.tactic
            gemm = self.kernel_cls(cutlass.Float32, tm == 256, (tm, tn), cluster, True)
            gemm.max_ab_stage = max_ab
            mac = cutlass_utils.HardwareInfo().get_max_active_clusters(
                cluster[0] * cluster[1]
            )
            fn = cute.compile(_launch, gemm, *args, nk, mac, stream)
            _COMPILED[key] = fn
        return fn

    def __call__(self, x, w, out, aux=None):
        m, k = x.shape
        n = w.shape[0]
        assert w.shape[1] == k and out.shape == (m, n) and x.stride(1) == 1
        assert w.is_contiguous() and out.stride(1) == 1
        assert x.dtype == w.dtype == out.dtype == torch.bfloat16
        for t in (x, w, out) + (() if aux is None else (aux,)):
            assert t.data_ptr() % 16 == 0
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        args = (
            _ptr(x), _ptr(w), _ptr(out), _ptr(out if aux is None else aux),
            cutlass.Int32(m),
        )
        nk = (n, k, x.stride(0), out.stride(0), None if aux is None else aux.stride(0))
        self._fn(nk, stream, args)(*args, stream)
        return out


class _GateKernel(PersistentDenseGemmKernel):
    """out = aux * BF16(sigmoid(BF16(acc))): the DSA output gate on the g_proj
    accumulator, as ATen computes ``attn * g_proj(x).sigmoid()`` in BF16 (BF16 GEMM
    output; sigmoid 1 / (1 + expf(-g)) in FP32 rounded to BF16; FP32 product
    rounded to BF16)."""

    aux_t2r = True
    ablate = 0  # microbench only: 1 = out = aux (load, no math), 2 = out = sigmoid (no aux use)

    def epi_convert_regs_aux(self, acc, aux, dst, tiled_copy_r2s):
        out = cute.make_rmem_tensor(acc.shape, self.c_dtype)
        if cutlass.const_expr(self.ablate == 1):
            out.store(aux.load())
        else:
            g = acc.load().to(cutlass.BFloat16).to(cutlass.Float32)
            e = cute.math.exp(g * -1.0, fastmath=False)
            sig = (cute.full_like(e, 1.0) / (e + 1.0)).to(cutlass.BFloat16)
            if cutlass.const_expr(self.ablate == 2):
                out.store(sig)
            else:
                out.store((aux.load().to(cutlass.Float32) * sig.to(cutlass.Float32)).to(self.c_dtype))
        dst.store(tiled_copy_r2s.retile(out).load())


class Bf16GateGemm(Bf16Gemm):
    """``out = attn * (x @ w.T).sigmoid()`` (BF16, ATen rounding) in one GEMM: the
    DSA g_proj with the output gate in its epilogue (no gate tensor round trip)."""

    kernel_cls = _GateKernel

    def __init__(self, tactic=DEFAULT_TACTIC, ablate=0):
        super().__init__(tactic)
        self.ablate = ablate

    def _fn(self, nk, stream, args):
        _GateKernel.ablate = self.ablate
        return super()._fn(nk + (self.ablate,), stream, args)

    def __call__(self, x, w, attn, out):
        assert attn.shape == out.shape and attn.stride(1) == 1
        return super().__call__(x, w, out, aux=attn)

