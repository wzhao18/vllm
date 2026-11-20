# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill KDA in_proj with the short conv in the GEMM epilogue.

On a prefill worker with an SM107 GPU, the online-MXFP8 KDA in_proj
(q|k|v|g|f_a|beta rows) and FP32 conv weights, prefill-only batches of at
least ``MIN_ROWS`` rows run the q|k|v rows through one sm_107 GEMM whose
epilogue applies the width-4 causal conv + SiLU and updates the conv state
(``sm107_gemm.kda_conv_gemm``: bit for bit the in_proj GEMM followed by
``ops/kda_conv.qkv_conv``), so neither the BF16 q|k|v GEMM output nor the
separate conv pass exist. The g|f_a|beta rows keep a cuBLASLt MXFP8 GEMM on a
view of the same weight. Batches with decode or speculative tokens, and
smaller batches, keep the stock in_proj + conv.
"""

import torch

from vllm.forward_context import get_forward_context
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

MIN_ROWS = 2048
#: 512x256 B-reuse tile, 2x4 cluster (A multicast across N), drain epilogue.
TACTIC = ((512, 256), (2, 4), 256, 4, "m", 0, 0, "")


def _qkv_rows(lin) -> torch.Tensor:
    """The online-MXFP8 in_proj weight as K-major [N, K] FP8 rows."""
    from vllm.models.kimi_k3.nvidia.mxfp8 import _kernel_layout

    w = lin.weight.data
    return w.t() if _kernel_layout(lin) == "kn" else w


def unsupported_reason(layer) -> str | None:
    """None when the layer can run the fused GEMM: SM107, online-MXFP8 in_proj
    with K-major [N, K] rows (either kernel layout), FP32 width-4 conv
    weights, 256-multiple planes; else why not."""
    from vllm.platforms import current_platform

    if not current_platform.is_device_capability((10, 7)):
        return "not SM107"
    from vllm.models.kimi_k3.nvidia.mxfp8 import _kernel_layout, is_mxfp8_linear

    lin = layer.in_proj_qkvgfab
    if not is_mxfp8_linear(lin) or _kernel_layout(lin) is None:
        return "in_proj is not an online-MXFP8 [N, K] / [K, N] linear"
    if not _qkv_rows(lin).is_contiguous():
        return "in_proj rows are not K-major"
    w = layer.conv1d.weight
    if w.dtype != torch.float32 or w.shape[-1] != 4:
        return f"conv weight {w.dtype} {tuple(w.shape)} (FP32 width 4 only)"
    if layer.local_projection_size % 256 or lin.input_size_per_partition % 128:
        return "plane / K size"
    return None


class PrefillKdaConvInProj:
    """Fused q|k|v in_proj + conv of one KDA layer (``unsupported_reason`` None)."""

    def __init__(self, layer) -> None:
        self.layer = layer
        self.p = layer.local_projection_size
        self._gemm = None

    def prepare(self) -> None:
        """After weight loading, before any forward: views of the q|k|v and
        g|f_a|beta rows and their F8_128x4 scales; compile the fused GEMM."""
        if self._gemm is not None:
            return
        from .sm107_gemm.kda_conv_gemm import KdaConvGemm

        lin = self.layer.in_proj_qkvgfab
        w = _qkv_rows(lin)
        n, k = w.shape
        nq = 3 * self.p
        # F8_128x4 scales are 128-row blocks of K / 32 bytes: q|k|v (a whole
        # number of blocks) is a prefix of the flat scale tensor.
        s = lin.weight_scale.data.view(torch.uint8).reshape(-1)
        cut = nq * (k // 32)
        self.w_rest, self.s_rest = w[nq:], s[cut:]
        cw = self.layer.conv1d.weight
        self.conv_w = cw.view(cw.shape[0], cw.shape[-1])
        self._gemm = KdaConvGemm(w[:nq], s[:cut], self.conv_w, self.p, tactic=TACTIC)
        dev = w.device
        rows = MIN_ROWS
        x = torch.zeros((rows, k), dtype=torch.float8_e4m3fn, device=dev)
        sx = torch.full((rows * k // 32,), 127, dtype=torch.uint8, device=dev)
        qsl = torch.tensor([0, rows], dtype=torch.int32, device=dev)
        lines = torch.zeros(1, dtype=torch.int32, device=dev)
        init = torch.zeros(1, dtype=torch.bool, device=dev)
        st = torch.zeros((1, nq, 3), dtype=torch.bfloat16, device=dev)
        self._gemm(x, sx, st, qsl, lines, init, null_block_id=NULL_BLOCK_ID)

    def fused(self, qa):
        """``((3, T, P) conv + SiLU q|k|v planes, (T, rest) g|f_a|beta)`` for a
        prefill-only batch of at least ``MIN_ROWS`` rows, else None."""
        gemm = self._gemm
        assert gemm is not None, "PrefillKdaConvInProj.prepare() not run"
        rows = qa.data.shape[0]
        if rows < MIN_ROWS:
            return None
        md = get_forward_context().attn_metadata
        if not isinstance(md, dict):
            return None
        m = md.get(self.layer.prefix)
        if (
            m is None
            or m.num_prefills == 0
            or m.num_decodes != 0
            or m.num_spec_decodes != 0
        ):
            return None
        from vllm.model_executor.layers.mamba.mamba_utils import (
            is_conv_state_dim_first,
        )

        conv_state = self.layer.kv_cache[0]
        if not is_conv_state_dim_first():
            conv_state = conv_state.transpose(-1, -2)
        if conv_state.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError(f"KDA conv GEMM: conv state dtype {conv_state.dtype}")
        x_scale = qa.scale.view(torch.uint8).reshape(-1)
        planes, _ = gemm(
            qa.data, x_scale, conv_state, m.non_spec_query_start_loc,
            m.non_spec_state_indices_tensor, m.has_initial_state,
            null_block_id=NULL_BLOCK_ID,
        )  # fmt: skip
        rest = torch._scaled_mm(
            qa.data,
            self.w_rest.t(),
            scale_a=qa.scale.view(torch.float8_e8m0fnu),
            scale_b=self.s_rest.view(torch.float8_e8m0fnu),
            out_dtype=torch.bfloat16,
        )
        return planes, rest


def contiguous_rows(x: torch.Tensor) -> torch.Tensor:
    """``x.contiguous()`` of a row-strided 2-D slice, copied as 16-byte
    elements when rows and strides allow it (byte-identical; the generic
    strided copy moves 2-byte elements: 7.7 vs ~1.5 us for the KDA beta
    columns at F16K)."""
    if x.is_contiguous():
        return x
    nbytes = x.size(-1) * x.element_size()
    if (
        x.dim() == 2
        and x.stride(-1) == 1
        and nbytes % 16 == 0
        and (x.stride(0) * x.element_size()) % 16 == 0
        and x.data_ptr() % 16 == 0
    ):
        return x.view(torch.complex128).contiguous().view(x.dtype)
    return x.contiguous()

