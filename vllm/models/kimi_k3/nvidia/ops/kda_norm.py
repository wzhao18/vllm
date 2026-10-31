# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 KDA output gate norm (FusedRMSNormGated, sigmoid, head_dim 128),
bitwise equal to FLA's ``layer_norm_gated_fwd_kernel`` as KDA calls it
(vllm/third_party/flash_linear_attention/ops/kda.py: rms_norm_gated, in place,
no bias / residual).

The FLA kernel gives each program one [BT = 16 rows, 128] tile (8 warps); at a
16k-token prefill (193k rows on one TP8 rank) its 12k programs are latency
bound: 27.9 us per call in-model for ~149 MB. Here a program walks TILES such
tiles in a software-pipelined loop; each tile is the FLA body unchanged (same
tile shape, warps and expressions, so the same layout and reduction order).
The rstd / mean side outputs, which KDA does not read, are not written.
"""

import os

import torch

from vllm.triton_utils import tl, triton

_BT = 16  # rows per tile: FLA's choice for D <= 512
_TILES = int(
    os.environ.get("VLLM_K3_KDA_NORM_TILES_MAX", "8")
)  # tiles per program, at most
_STAGES = int(os.environ.get("VLLM_K3_KDA_NORM_STAGES", "3"))
_MIN_PROGRAMS = 848  # keep >= 4 programs per SM (212 SMs) before taking more tiles each


@triton.jit(do_not_specialize=["T"])
def _kda_rmsnorm_gated_kernel(
    x,
    g,
    y,
    w,
    eps,
    T,
    H: tl.constexpr,
    g_stride_n: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    TILES: tl.constexpr,
    STAGES: tl.constexpr,
):
    for k in tl.range(0, TILES, num_stages=STAGES):
        i_t = tl.program_id(0).to(tl.int64) * TILES + k

        o_t = i_t * BT + tl.arange(0, BT)
        o_d = tl.arange(0, BD)
        m_d = o_d < D
        m_t = o_t < T
        m_x = m_t[:, None] & m_d[None, :]

        p_x = x + o_t[:, None] * D + o_d[None, :]
        b_x = tl.load(p_x, mask=m_x, other=0.0).to(tl.float32)
        b_xbar = tl.where(m_d[None, :], b_x, 0.0)
        b_var = tl.sum(b_xbar * b_xbar, axis=1) / D
        b_rstd = 1 / tl.sqrt(b_var + eps)

        b_w = tl.load(w + o_d, mask=m_d).to(tl.float32)
        b_x_hat = b_x * b_rstd[:, None]
        b_y = b_x_hat * b_w[None, :]

        o_g = (o_t // H) * g_stride_n + (o_t % H) * D
        b_g = tl.load(
            g + o_g[:, None] + o_d[None, :],
            mask=(o_t[:, None] < T) & m_d[None, :],
            other=0.0,
        ).to(tl.float32)
        b_y = b_y * tl.sigmoid(b_g)

        p_y = y + o_t[:, None] * D + o_d[None, :]
        tl.store(p_y, b_y.to(p_y.dtype.element_ty), mask=m_x)


def supported(x: torch.Tensor, g: torch.Tensor, norm) -> bool:
    # Only where the op would run FLA's Triton path (forward_cuda).
    fwd = getattr(norm, "_forward_method", None)
    return (
        x.is_cuda
        and getattr(fwd, "__func__", None) is type(norm).forward_cuda
        and x.shape[-1] == 128
        and x.is_contiguous()
        and norm.bias is None
        and norm.activation == "sigmoid"
        and norm.weight.is_contiguous()
        and g.stride(-1) == 1
    )


def _launch(x2, gv, weight, eps, T, H, tiles):
    D = x2.shape[-1]
    _kda_rmsnorm_gated_kernel[(triton.cdiv(T, _BT * tiles),)](
        x2,
        gv,
        x2,
        weight,
        eps,
        T,
        H=H,
        g_stride_n=gv.stride(0),
        D=D,
        BT=_BT,
        BD=triton.next_power_of_2(D),
        TILES=tiles,
        STAGES=_STAGES,
        num_warps=8,
    )


def rmsnorm_gated_(x: torch.Tensor, g: torch.Tensor, weight: torch.Tensor, eps: float):
    """In place on x (FLA's y = x): rms_norm_gated(x, g, weight, None,
    "sigmoid", eps=eps) for D = 128."""
    D = x.shape[-1]
    x2 = x.reshape(-1, D)
    H = 1 if g.ndim == 2 else g.shape[-2]
    gv = g.view(-1, H, D)
    T = x2.shape[0]
    if T == 0:
        return x
    # Tiles per program: a power of two up to _TILES that keeps >= _MIN_PROGRAMS
    # programs (small calls stay one tile per program, like FLA).
    tiles = 1
    while tiles < _TILES and triton.cdiv(T, _BT * tiles * 2) >= _MIN_PROGRAMS:
        tiles *= 2
    _launch(x2, gv, weight, eps, T, H, tiles)
    return x


def warmup(heads: int, g_row_stride: int, weight: torch.Tensor, eps: float) -> None:
    """Compile every TILES variant for gates that are row-strided views of a
    projection output with this row stride, built the way KDA builds them (a
    column slice, then (h d) -> h d), for 1- and 2-token calls: a 1-token view
    can carry a different leading stride."""
    D = weight.shape[0]
    for n in (1, 2):
        proj = torch.zeros((n, g_row_stride), dtype=weight.dtype, device=weight.device)
        gv = proj[:, : heads * D].view(n, heads, D).view(-1, heads, D)
        x = torch.zeros((n * heads, D), dtype=weight.dtype, device=weight.device)
        tiles = 1
        while tiles <= _TILES:
            _launch(x, gv, weight, eps, n * heads, heads, tiles)
            tiles *= 2


# ---------------------------------------------------------------------------
# MXFP8 output: the gate norm writes o_proj's input as MXFP8 (FP8 values +
# F8_128x4-swizzled E8M0 scales, FlashInfer mxfp8_quantize semantics) instead
# of BF16 in place, so an online-MXFP8 o_proj skips its own quantization.
# The tile body is the BF16 kernel's; the value is rounded to BF16 exactly as
# the in-place store would, then quantized in 32-wide blocks (4 per head).


@triton.jit
def _ue8m0_and_inv(amax):
    """FlashInfer float_to_ue8m0_fast(amax * fp32(1/448)) and its inverse
    2^(127 - e) (0 for e == 0)."""
    nm = amax * 0.002232142857142857
    bits = nm.to(tl.uint32, bitcast=True)
    exp = (bits >> 23) & 255
    mant = bits & 0x7FFFFF
    bump = tl.where(mant != 0, 1, 0).to(tl.uint32)
    bump = tl.where((exp == 0) & (mant <= 0x400000), 0, bump).to(tl.uint32)
    e = tl.minimum(exp + bump, 254)
    e = tl.where(nm <= 0.0, 0, e).to(tl.uint32)
    inv = ((254 - e) << 23).to(tl.float32, bitcast=True)
    inv = tl.where(e == 0, 0.0, inv)
    return e, inv


@triton.jit(do_not_specialize=["T"])
def _kda_rmsnorm_gated_mxfp8_kernel(
    x,
    g,
    q,
    sf,
    w,
    eps,
    T,
    q_stride,
    H: tl.constexpr,
    g_stride_n: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    SF_COLS: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    o_t = i_t * BT + tl.arange(0, BT)
    o_d = tl.arange(0, D)
    m_t = o_t < T

    b_x = tl.load(x + o_t[:, None] * D + o_d[None, :], mask=m_t[:, None], other=0.0)
    b_x = b_x.to(tl.float32)
    b_var = tl.sum(b_x * b_x, axis=1) / D
    b_rstd = 1 / tl.sqrt(b_var + eps)
    b_w = tl.load(w + o_d).to(tl.float32)
    b_y = b_x * b_rstd[:, None] * b_w[None, :]
    o_g = (o_t // H) * g_stride_n + (o_t % H) * D
    b_g = tl.load(g + o_g[:, None] + o_d[None, :], mask=m_t[:, None], other=0.0)
    b_y = b_y * tl.sigmoid(b_g.to(tl.float32))
    # The BF16 value the in-place kernel stores.
    b_y = b_y.to(tl.bfloat16).to(tl.float32)

    b_y3 = tl.reshape(b_y, (BT, D // 32, 32))
    amax = tl.max(tl.abs(b_y3), axis=2)
    e, inv = _ue8m0_and_inv(amax)
    b_q = (b_y3 * inv[:, :, None]).to(tl.float8e4nv)
    b_q = tl.reshape(b_q, (BT, D))

    tok = o_t // H
    head = o_t % H
    tl.store(
        q + tok[:, None] * q_stride + head[:, None] * D + o_d[None, :],
        b_q,
        mask=m_t[:, None],
    )
    # Scale column of block j of this head, in the swizzled layout.
    col = head[:, None] * (D // 32) + tl.arange(0, D // 32)[None, :]
    off = (
        (col % 4)
        + (col // 4) * 512
        + (tok[:, None] % 32) * 16
        + ((tok[:, None] % 128) // 32) * 4
        + (tok[:, None] // 128) * (128 * SF_COLS)
    )
    tl.store(sf + off, e.to(tl.uint8), mask=m_t[:, None])


def rmsnorm_gated_mxfp8(
    x: torch.Tensor,
    g: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    out=None,
):
    """``rmsnorm_gated_(x, g, ...)`` (D = 128) written as the MXFP8 activation
    ``[tokens, heads * 128]`` of the following o_proj instead of in place.
    ``out``: a preallocated ``mxfp8_act.Mxfp8Buffer`` (needed under CUDA-graph
    capture across an eager break); returns its QuantizedActivation. Unlike
    the mxfp8_act producers, the padding rows' scales are left unwritten."""
    from vllm.models.kimi_k3.nvidia.ops.mxfp8_act import Mxfp8Buffer

    D = x.shape[-1]
    assert D == 128
    x2 = x.reshape(-1, D)
    H = 1 if g.ndim == 2 else g.shape[-2]
    gv = g.view(-1, H, D)
    T = x2.shape[0]
    tokens = T // H
    if out is None:
        out = Mxfp8Buffer(tokens, H * D, x.device, x.dtype)
    assert out.rows == tokens and out.k == H * D
    if tokens:
        # The scales of the padding rows of the last 128-row tile stay
        # unwritten: the GEMM's rows are independent and drops those rows.
        _kda_rmsnorm_gated_mxfp8_kernel[(triton.cdiv(T, _BT),)](
            x2,
            gv,
            out.data,
            out.scale,
            weight,
            eps,
            T,
            out.data.stride(0),
            H=H,
            g_stride_n=gv.stride(0),
            D=D,
            BT=_BT,
            SF_COLS=out.padded_cols,
            num_warps=8,
        )
    return out.activation()
