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
- ``VLLM_K3_MEGAMOE_LOCALIZE`` (default 0; set 1 to opt in): place each rank's routed expert weights
  in the GPU's two locality domains (MLOPart), so every die streams its half of N
  from its own HBM (bitwise-identical output). Full K3.1 DEP16 on VR: -0.57..-0.65 ms
  per step on one 4-node block (MegaMoE 119.7 -> 113.7 us/launch at 128 rows), 5 of
  5 processes clean. The gain depends on the node: DEP4 8-layer B/C repeats on three
  dedicated nodes saved -334..-381 us per step on two and -179..-211 us on the third.
  The KV pool is 0.1-3% smaller per GPU and varies from one start to the next (full
  DEP16: 102.34 -> 101.96-102.21 GiB; DEP4 8-layer: 202.54 -> 196.1-201.5 GiB). The
  MLOPart weights live outside the PyTorch allocator and vLLM's profile counts
  them, plus a start-dependent MLOPart overhead, as non-torch memory. Falls back
  to the plain placement without MLOPart or under EPLB (the weights change shape to
  ``(2, E, N / 2, K)``, which EPLB cannot move). Caveat: on a shared DEP4 node
  (224 experts/rank) 4 of 8 localized processes had cases 17-25% slower (one rank's
  drafter MLA decode ~1.4x slower) while other jobs used the same GPUs.
"""

from __future__ import annotations

import os
import types

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_LOCALIZE = os.environ.get("VLLM_K3_MEGAMOE_LOCALIZE", "0") == "1"


def _localize_experts(experts, deep_gemm) -> None:
    """Routed weights ``(E, N, K)`` -> ``(2, E, N / 2, K)``, one N half per locality
    domain; the block scales stay as they are."""
    for name in ("w13_weight", "w2_weight"):
        w = getattr(experts, name)
        if not deep_gemm.is_localized(w.data):
            w.data = deep_gemm.localize(w.data)


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
    """Apply the env-selected MegaMoE tuning to every ``DeepGemmMegaMoEExperts``."""
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
    if _LOCALIZE:
        if vllm_config.parallel_config.enable_eplb:
            logger.info_once("Kimi K3 MegaMoE: routed expert weights not localized (EPLB)")
            return
        if not deep_gemm.is_localization_available():
            logger.info_once("Kimi K3 MegaMoE: routed expert weights not localized (no MLOPart)")
            return
        for e in experts:
            _localize_experts(e, deep_gemm)
            torch.cuda.empty_cache()
        deep_gemm.destroy_localizer()
        logger.info_once(
            "Kimi K3 MegaMoE: localized the routed expert weights of %d layers", len(experts)
        )
