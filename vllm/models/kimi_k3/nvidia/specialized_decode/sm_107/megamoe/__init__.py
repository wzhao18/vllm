# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 MegaMoE (DeepGEMM FP8 x FP4 SiTU) wide-EP decode tuning.

``prepare_megamoe`` runs once per worker after the weights are loaded, before
the runner's warmup:

- The wide-EP kernel: every FP8 x FP4 SiTU ``DeepGemmMegaMoEExperts`` of the model
  launches the SM107 decode's own MegaMoE kernel (``kernel.SituMegaMoE``: the
  header ``situ_patch`` derives from the installed stock one; bitwise-identical
  output, -5..-12% per launch at EP16/EP32 per-rank shapes). DeepGEMM stays as
  installed: its headers are read-only includes and its JIT, include root and
  kernels are untouched. The SM107 specialized decode raises when the installed
  DeepGEMM's header or heuristics do not match (it never falls back to the stock
  kernel silently).
"""

from __future__ import annotations

import types

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

def _situ_forward(
    self,
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    shared_experts: torch.nn.Module | None = None,
) -> torch.Tensor:
    """``DeepGemmMegaMoEExperts.forward`` (MX path) launching the SM107 decode's
    wide-EP kernel instead of ``deep_gemm.fp8_fp4_mega_moe``: the same staging,
    buffer and arguments (stock's helpers), one different kernel."""
    from vllm.model_executor.layers.fused_moe import deep_gemm_mega_moe as stock

    if shared_experts is not None and self.has_fused_shared_experts:
        raise ValueError(
            f"{self.prefix} fuses the shared expert into MegaMoE; pass shared_experts=None."
        )
    num_tokens = hidden_states.shape[0]
    if num_tokens > self.max_num_tokens:
        raise ValueError(
            f"MegaMoE got {num_tokens} tokens, but the symmetric buffer was "
            f"sized for {self.max_num_tokens}."
        )
    self._synchronize_first_launch()
    is_padding = None
    if stock.envs.VLLM_MOE_SKIP_PADDING and stock.is_forward_context_available():
        is_padding = stock.get_forward_context().is_padding
        if is_padding is not None:
            is_padding = is_padding[:num_tokens]
    if self.capture_fn is not None:
        self.capture_fn(topk_ids)
    fmt = self.weight_format
    symm_buffer = self.get_symm_buffer()
    shared_x_sf = shared_block_m = None
    if self.shared_spec is not None and self.shared_spec.scale_group_k:
        shared_x_sf = symm_buffer.shared_l1_acts_sf
        shared_block_m = self.get_block_m(num_tokens)
    stock.prepare_megamoe_inputs(
        hidden_states,
        topk_weights,
        topk_ids,
        symm_buffer.x[:num_tokens],
        symm_buffer.x_sf[:num_tokens] if fmt.hidden_quant is not None else None,
        symm_buffer.topk_idx[:num_tokens],
        symm_buffer.topk_weights[:num_tokens],
        is_padding=is_padding,
        shared_x_sf=shared_x_sf,
        shared_block_m=shared_block_m,
        hidden_quant=fmt.hidden_quant,
        eplb_state=self.eplb_state,
    )
    kwargs = dict(self._kernel_kwargs)
    if self.has_fused_shared_experts:
        kwargs["shared_l1_weights"] = self._kernel_weights("shared_l1")
        kwargs["shared_l2_weights"] = self._kernel_weights("shared_l2")
    y = torch.empty_like(hidden_states, dtype=torch.bfloat16)
    self._k3_situ(y, self.l1_weights, self.l2_weights, symm_buffer,
                  fast_math=self.fast_math, **kwargs)  # fmt: skip
    if shared_experts is not None:
        y += shared_experts(hidden_states)
    return y


def prepare_megamoe(model: torch.nn.Module, vllm_config) -> None:
    """Launch the SM107 wide-EP kernel from every FP8 x FP4 SiTU ``DeepGemmMegaMoEExperts``.
    DeepGEMM MegaMoE only: nothing to do for any other MoE backend
    (``kimi_k3_mega_moe`` runs its own vendored kernels)."""
    if vllm_config.kernel_config.moe_backend != "deep_gemm_mega_moe":
        return
    from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import (
        DeepGemmMegaMoEExperts,
    )

    experts = [m for m in model.modules() if isinstance(m, DeepGemmMegaMoEExperts)]
    if not experts:
        return
    from vllm.utils.deep_gemm import _import_deep_gemm

    deep_gemm = _import_deep_gemm()
    if deep_gemm is None:
        return
    fmt = experts[0].weight_format
    if fmt.kernel == "fp8_fp4_mega_moe" and experts[0].activation.name == "situ":
        if fmt.uses_nvfp4_api or fmt.hidden_quant is None:
            raise RuntimeError(
                f"Kimi K3 specialized decode (SM107): MegaMoE format {fmt.name} is not "
                "the MXFP8 x FP4 SiTU the wide-EP kernel serves"
            )
        from .kernel import SituMegaMoE

        situ = SituMegaMoE(deep_gemm)
        for e in experts:
            e._k3_situ = situ
            e.forward = types.MethodType(_situ_forward, e)
        logger.info_once(
            "Kimi K3 MegaMoE: K3.1 wide-EP kernel (SM107 decode build, %s)",
            situ.include[0],
        )
