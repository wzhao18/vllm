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
