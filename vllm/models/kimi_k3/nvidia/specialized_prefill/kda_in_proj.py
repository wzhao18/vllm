# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KDA prefill in_proj with the Q|K|V short conv in the GEMM epilogue (SM107).

For a prefill-only batch whose in_proj input is the MXFP8 AttnRes output, the
Q|K|V rows of in_proj run ``sm107_gemm.KdaConvGemm`` (GEMM + conv + conv state
update, bit for bit the stock conv) and the remaining rows (g, f_a, beta) one
cuBLASLt MXFP8 GEMM on the weight row slice.
"""

import torch

from vllm.forward_context import get_forward_context
from vllm.platforms import current_platform

#: Fewer rows than this keep the unfused in_proj + qkv_conv path.
MIN_ROWS = 2048


class KdaPrefillInProj:
    def __init__(self, layer) -> None:
        from vllm.models.kimi_k3.nvidia.ops.kda_conv import NULL_BLOCK_ID

        from .sm107_gemm.kda_conv_gemm import KdaConvGemm

        lin = layer.in_proj_qkvgfab
        p = layer.local_projection_size
        w = lin.weight.data
        k = layer.hidden_size
        w_nk = w.t() if w.shape[0] == k and w.shape[1] != k else w
        if not w_nk.is_contiguous() or w_nk.shape[1] != k:
            raise RuntimeError(
                f"KDA conv GEMM needs the [N, K] K-major in_proj weight, got "
                f"{tuple(w.shape)} strides {w.stride()}"
            )
        n3 = 3 * p
        sf = lin.weight_scale.data.reshape(-1).view(torch.uint8)
        sf_row = k // 32
        conv_w = layer.conv1d.weight.data
        conv_w = conv_w.view(conv_w.size(0), conv_w.size(-1))
        self.qkv = KdaConvGemm(w_nk[:n3], sf[: n3 * sf_row], conv_w, p)
        self.w_rest = w_nk[n3:]
        self.sf_rest = sf[n3 * sf_row :]
        self.null_block_id = NULL_BLOCK_ID
        self.prefix = layer.prefix

    @staticmethod
    def supported(layer) -> bool:
        return (
            current_platform.is_device_capability((10, 7))
            and layer.local_projection_size % 256 == 0
            and layer.conv1d.weight.dtype == torch.float32
            and layer.conv1d.weight.size(-1) == 4
        )

    def __call__(self, layer, qa, num_tokens: int):
        """``(qkv (3, M, P), rest (M, N - 3P))`` or None (not a prefill-only
        batch of >= MIN_ROWS rows)."""
        if num_tokens < MIN_ROWS:
            return None
        md = get_forward_context().attn_metadata
        m = md.get(self.prefix) if isinstance(md, dict) else None
        if (
            m is None
            or m.num_spec_decodes > 0
            or m.num_decodes > 0
            or m.num_prefills == 0
            or m.non_spec_query_start_loc is None
            or m.non_spec_state_indices_tensor is None
            or m.has_initial_state is None
        ):
            return None
        conv_state = layer.kv_cache[0]
        from vllm.model_executor.layers.mamba.mamba_utils import (
            is_conv_state_dim_first,
        )

        if not is_conv_state_dim_first():
            conv_state = conv_state.transpose(-1, -2)
        if conv_state.dtype != torch.float32:
            return None
        # The full (graph-padded) metadata, as the stock conv reads it.
        qkv, _ = self.qkv(
            qa.data,
            qa.scale,
            conv_state,
            m.non_spec_query_start_loc,
            m.non_spec_state_indices_tensor,
            m.has_initial_state,
            null_block_id=self.null_block_id,
        )
        e8 = torch.float8_e8m0fnu
        rest = torch._scaled_mm(
            qa.data,
            self.w_rest.t(),
            scale_a=qa.scale.view(e8),
            scale_b=self.sf_rest.view(e8),
            out_dtype=qa.orig_dtype,
        )
        return qkv, rest
