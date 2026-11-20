"""Prefill DSA value up-projection with the output gate in its epilogue.

The stock DSA prefill path writes the gated attention output in two passes after
the sparse MLA: ``mla_attention._v_up_proj`` (``torch.bmm(latent, W_UV)`` into a
BF16 ``(T, H * V)`` buffer), then the DSA output gate ``attn * g.sigmoid()``
(``g = g_proj(x)``; one more read of that buffer and of ``g``, one more write).
``UvGate`` writes the gated output in one pass: a batched (one batch per head)
BF16 tcgen05 GEMM whose epilogue rounds each FP32 accumulator to BF16 (the bmm
output dtype), reads ``g`` as an aux operand in the accumulator layout and
applies the gate as eager PyTorch computes it (sigmoid in FP32 rounded to BF16,
then the FP32 product rounded to BF16). The gate's dependency on ``g_proj``
moves the g_proj GEMM ahead of the attention (``dsa.py``).
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
import torch
from cutlass.cute.runtime import make_ptr

from vllm.logger import init_logger

from .dense_gemm_bf16_sm107 import PersistentDenseGemmKernel

logger = init_logger(__name__)

# (mma tile M, N), (cluster M, N), max A/B stages (None = fill smem)
DEFAULT_TACTIC = ((256, 128), (2, 1), None)
# Prefill rows only: below this the stock bmm + gate kernel run.
MIN_ROWS = 512
_COMPILED: dict = {}


class _UvGateKernel(PersistentDenseGemmKernel):
    """C = BF16(BF16(acc) * BF16(sigmoid(aux))): the W_UV bmm with the DSA output
    gate (aux = g_proj(x), BF16, laid out like C)."""

    aux_t2r = True
    # K = kv_lora_rank (512): a tile's mainloop is too short to hide its aux
    # loads; load the whole tile's gate one tile ahead.
    aux_tile_ahead = True

    def epi_convert_regs_aux(self, acc, aux, dst, tiled_copy_r2s):
        out = cute.make_rmem_tensor(acc.shape, self.c_dtype)
        attn = acc.load().to(cutlass.BFloat16).to(cutlass.Float32)
        g = aux.load().to(cutlass.Float32)
        e = cute.math.exp(g * -1.0, fastmath=False)
        sig = (cute.full_like(e, 1.0) / (e + 1.0)).to(cutlass.BFloat16)
        out.store((attn * sig.to(cutlass.Float32)).to(self.c_dtype))
        dst.store(tiled_copy_r2s.retile(out).load())


@cute.jit
def _launch(
    gemm: cutlass.Constexpr,
    lat_ptr: cute.Pointer,
    w_ptr: cute.Pointer,
    out_ptr: cute.Pointer,
    g_ptr: cute.Pointer,
    m: cutlass.Int32,
    dims: cutlass.Constexpr,
    mac: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    h, lat, v, ls, hs, os_, gs, ws = dims
    # A (M, K=L, H): latent[t, h, :]; B (N=V, K=L, H): W_UV[h] (L, V), any
    # 16-byte-aligned strides with L or V contiguous; C / aux (M, V, H): column
    # h * V + n of the (T, H * V) output / gate.
    a = cute.make_tensor(lat_ptr, cute.make_layout((m, lat, h), stride=(ls, 1, hs)))
    b = cute.make_tensor(w_ptr, cute.make_layout((v, lat, h), stride=(ws[2], ws[1], ws[0])))
    c = cute.make_tensor(out_ptr, cute.make_layout((m, v, h), stride=(os_, 1, v)))
    aux = cute.make_tensor(g_ptr, cute.make_layout((m, v, h), stride=(gs, 1, v)))
    gemm(a, b, c, mac, stream, lambda x: x, None, aux, None)


def _ptr(t: torch.Tensor):
    return make_ptr(cutlass.BFloat16, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)


def _w_ok(w: torch.Tensor) -> bool:
    """W_UV (H, L, V) strides the GEMM's B operand takes: L or V contiguous, the
    rest 16-byte multiples."""
    s = w.stride()
    return (s[1] == 1 or s[2] == 1) and all(x == 1 or x % 8 == 0 for x in s) and (
        w.data_ptr() % 16 == 0
    )


class UvGate:
    """``out = BF16(bmm(latent, w_uv)) * g.sigmoid()`` (eager PyTorch rounding).

    latent: (T, H, L) BF16, L contiguous (token / head strides multiples of 8);
    w_uv: (H, L, V) BF16 (L or V contiguous); g, out: (T, H * V) BF16 row-major.
    """

    def __init__(self, tactic=DEFAULT_TACTIC, tile_ahead=True):
        self.tactic = tactic
        self.tile_ahead = tile_ahead  # microbench: False = per-subtile aux loads

    def _compiled(self, dims, stream, args):
        key = (self.tactic, dims, self.tile_ahead)
        fn = _COMPILED.get(key)
        if fn is None:
            (tm, tn), cluster, max_ab = self.tactic
            gemm = _UvGateKernel(cutlass.Float32, tm == 256, (tm, tn), cluster, True)
            gemm.max_ab_stage = max_ab
            gemm.aux_tile_ahead = self.tile_ahead
            mac = cutlass_utils.HardwareInfo().get_max_active_clusters(
                cluster[0] * cluster[1]
            )
            fn = cute.compile(_launch, gemm, *args, dims, mac, stream)
            _COMPILED[key] = fn
        return fn

    def __call__(self, latent, w_uv, g, out):
        t, h, lat = latent.shape
        h2, lat2, v = w_uv.shape
        assert h2 == h and lat2 == lat and out.shape == g.shape == (t, h * v)
        assert latent.stride(2) == 1 and out.stride(1) == 1 and g.stride(1) == 1
        assert _w_ok(w_uv)
        for x in (latent, out, g):
            assert x.dtype == torch.bfloat16 and x.data_ptr() % 16 == 0
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        args = (_ptr(latent), _ptr(w_uv), _ptr(out), _ptr(g), cutlass.Int32(t))
        dims = (
            h, lat, v, latent.stride(0), latent.stride(1), out.stride(0), g.stride(0),
            tuple(w_uv.stride()),
        )
        self._compiled(dims, stream, args)(*args, stream)
        return out


def reference(latent, w_uv, g):
    """The stock passes: bmm into a (T, H * V) buffer, then ``attn * g.sigmoid()``."""
    t, h, _ = latent.shape
    v = w_uv.shape[2]
    attn = latent.new_empty((t, h * v))
    torch.bmm(latent.transpose(0, 1), w_uv, out=attn.view(t, h, v).transpose(0, 1))
    return attn * g.sigmoid()


_KERNEL = UvGate()


def supports(latent: torch.Tensor, w_uv: torch.Tensor, g: torch.Tensor, out: torch.Tensor) -> bool:
    """Per-shape policy + capability of :func:`v_up_gate`."""
    if latent.dim() != 3 or w_uv.dim() != 3 or g.dim() != 2 or out.dim() != 2:
        return False
    t, h, lat = latent.shape
    v = w_uv.shape[2]
    return (
        t >= MIN_ROWS
        and w_uv.shape[:2] == (h, lat)
        and g.shape == out.shape == (t, h * v)
        and all(x.dtype == torch.bfloat16 for x in (latent, w_uv, g, out))
        and latent.stride(2) == 1
        and latent.stride(0) % 8 == 0
        and latent.stride(1) % 8 == 0
        and g.stride(1) == 1
        and out.stride(1) == 1
        and g.stride(0) % 8 == 0
        and out.stride(0) % 8 == 0
        and all(x.data_ptr() % 16 == 0 for x in (latent, g, out))
        and lat % 64 == 0
        and v % 16 == 0
        and _w_ok(w_uv)
    )


def v_up_gate(
    latent: torch.Tensor, w_uv: torch.Tensor, g: torch.Tensor, out: torch.Tensor
) -> bool:
    """Write ``bmm(latent, w_uv) * g.sigmoid()`` (the gated DSA attention output)
    into ``out``; False (nothing written) when the shapes are outside the kernel."""
    if not supports(latent, w_uv, g, out):
        return False
    logger.info_once("Kimi K3.1 prefill: W_UV up-projection with the output gate (uv_gate) active")
    _KERNEL(latent, w_uv, g, out)
    return True
