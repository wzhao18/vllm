# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Residual-stream producers of the prefill AttnRes (Kimi K3.1, sm_107).

Today every AttnRes call that follows a layer output reads the AttnRes prefix
``P`` and the layer output ``y``, forms ``u = bf16(P + y)`` and writes ``u``
back as the new prefix: one extra row read and one row write per call in the
HBM-bound AttnRes. Here the producer GEMM of ``y`` updates the prefix itself:
the sm_107 MXFP8 GEMM keeps its plain epilogue (accumulator -> BF16 tile in
shared memory) and its TMA store becomes a reduce-add
(``cp.reduce.async.bulk.tensor`` ``.add``), so L2 forms
``P <- bf16(f32(P) + f32(bf16(acc)))`` - the stock AttnRes prefix rounding - in
the die-local prefix buffer (``attn_res_dielocal``), and the AttnRes call after
the GEMM streams ``P`` alone (``attn_res_lp`` with ``delta=None``).

Producers (online / checkpoint MXFP8 linears with the [N, K] K-major weight):
  * KDA o_proj: ``add_into`` (or ``store_into`` at AttnRes block-write layers,
    where the attention output starts the new prefix);
  * MoE shared expert down_proj: ``add_into``; the latent up-projection then
    beta-adds the routed output into the same prefix (``addmm_``), so a MoE
    layer forms ``bf16(bf16(P + S) + acc_up)`` where the stock order is
    ``bf16(P + bf16(S + acc_up))``: the same three roundings (GEMM output,
    BF16 + BF16, BF16 + FP32 beta-add), reassociated.
"""

import torch

from vllm.platforms import current_platform

# (mma tile M, N), cluster (M, N), MMA inst M, swizzle, raster, split producer,
# swap, epilogue flags: the 512x256 B-reuse tile on (2, 2) clusters is the
# MXFP8 host at nvjet parity for the N = 7168 prefill shapes (P-GEMM R4-M3).
TACTIC = ((512, 256), (2, 2), 256, 8, "n", 0, 0, "")
MIN_ROWS = 4096


def _weight_rows(linear) -> tuple[torch.Tensor, torch.Tensor] | None:
    """The linear's E4M3 weight as a contiguous [N, K] tensor and its
    128x4-swizzled UE8M0 scales (flat), or None for another layout."""
    from vllm.models.kimi_k3.nvidia.mxfp8 import _kernel_layout, is_mxfp8_linear

    if not is_mxfp8_linear(linear):
        return None
    layout = _kernel_layout(linear)
    if layout is None:
        return None
    w = linear.weight.data
    if layout == "kn":
        w = w.t()
    if not w.is_contiguous() or w.dtype != torch.float8_e4m3fn:
        return None
    scale = getattr(linear, "weight_scale", None)
    if scale is None:
        return None
    return w, scale.data.view(torch.uint8).reshape(-1)


def supported(linear) -> bool:
    """sm_107 and an MXFP8 linear this GEMM can run (N % 256, K % 128)."""
    if not current_platform.is_device_capability((10, 7)):
        return False
    rows = _weight_rows(linear)
    if rows is None:
        return False
    n, k = rows[0].shape
    return n % 256 == 0 and k % 128 == 0


class ResidGemm:
    """``x @ W.T`` (MXFP8 activation ``qa``) added into / stored as the AttnRes
    prefix. ``linear`` must satisfy :func:`supported`."""

    def __init__(self, linear) -> None:
        rows = _weight_rows(linear)
        if rows is None:
            raise ValueError("attn_res_resid: linear is not an MXFP8 [N, K] linear")
        self.weight, self.scale = rows
        self.n, self.k = self.weight.shape
        self._gemm: dict[bool, object] = {}

    def _plain(self, reduce: bool):
        g = self._gemm.get(reduce)
        if g is None:
            from .sm107_gemm import kda_conv_gemm as kcg

            tac = TACTIC[:7] + ("red" if reduce else TACTIC[7],)
            g = self._gemm[reduce] = kcg.PlainGemm(self.weight, self.scale, tac)
        return g

    def _run(self, qa, out: torch.Tensor, reduce: bool) -> torch.Tensor:
        x = qa.data
        if x.dtype != torch.float8_e4m3fn:
            x = x.view(torch.float8_e4m3fn)
        m = x.shape[0]
        if x.shape[1] != self.k or x.stride(1) != 1:
            raise ValueError(f"attn_res_resid: activation {tuple(x.shape)} vs K {self.k}")
        if out.shape != (m, self.n) or out.stride(1) != 1 or out.stride(0) != self.n:
            raise ValueError(f"attn_res_resid: prefix {tuple(out.shape)} vs ({m}, {self.n})")
        if out.dtype != torch.bfloat16:
            raise ValueError("attn_res_resid: the prefix must be BF16")
        self._plain(reduce)(x, qa.scale, out)
        return out

    def add_into(self, qa, prefix: torch.Tensor) -> torch.Tensor:
        """prefix <- bf16(prefix + bf16(qa @ W.T)) in place; returns prefix."""
        return self._run(qa, prefix, True)

    def store_into(self, qa, out: torch.Tensor) -> torch.Tensor:
        """out <- bf16(qa @ W.T); returns out."""
        return self._run(qa, out, False)


def gemm_for(linear) -> ResidGemm | None:
    """The cached ResidGemm of ``linear`` (None when unsupported)."""
    g = getattr(linear, "_attn_res_resid_gemm", None)
    if g is None:
        if not supported(linear):
            return None
        g = ResidGemm(linear)
        linear._attn_res_resid_gemm = g
    return g
