# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MoE layers of the NVFP4 checkpoint (``modelopt_mixed``) on the TP (non-SP) decode:
MoE tensor parallel, no expert parallelism (TP16 = DCP16, BUG-BENCH-1).

That checkpoint's MoE layer differs from the K3 / MXFP4 one the tuned providers
(``SharedFront`` -> ``PersistentExperts`` -> ``PreparedTail``) were built for in two
places, both handled here with the stock layer's own kernels on its own weights:

- routed experts: NVFP4 W4A4 (E2M1 weights, E4M3 scales per 16 elements, per-expert
  global scales; ``ModelOptNvFp4FusedMoE`` on the FlashInfer TRT-LLM-Gen backend).
  ``Nvfp4RoutedExperts`` runs the stock experts' routed-ids entry (vLLM's
  ``TrtLlmNvFp4ExpertsModular`` -> FlashInfer's public
  ``trtllm_fp4_block_scale_routed_moe``): the layer's weights and block scales, the
  activation quantized with the layer's input global scale (its prepare), the folded
  alphas (``g1_alphas`` / ``g2_alphas``), the SiTU alpha / beta and ``g1_scale_c`` of
  the stock experts object. The stock layer runs the same experts through the
  monolithic entry, which routes from the logits inside the kernel; here the routes
  are the stock router's (``prepared.stock_router``: the fused grouped top-k op, as
  the wide MoE rows take them). The top-k weighted sum is finalized in the kernel
  (stock defers it to its tail op at small batches: FP32 summation order differs).
- shared experts: online MXFP8 W8A8 (32-element E8M0 blocks). ``Mxfp8SharedExpert``
  is stock ``KimiMLP.forward`` at TP (``reduce_results=False``: this rank's partial):
  ``gemm.mxfp8.Mxfp8Projection`` gate_up (BF16 input quantized bitwise as FlashInfer's
  ``mxfp8_quantize``, stock's per-R MXFP8 kernel), ``KimiMLP.act_mxfp8_into`` (the
  stock SiTU-and-mul + MXFP8 quantize op writing the down projection's input), then
  the MXFP8 down projection.

``Nvfp4MoE`` is the layer's provider for every captured MoE batch: the stock router
and latent down projection (``KimiMoE._maybe_overlap_router_and_down_proj``, BF16, as
the wide rows' "stock front"), the stock routes, the NVFP4 routed experts' partial
into the wide MoE tail (``tail_rows.PreparedTailRows``: two-shot latent all-reduce,
RMS norm, up projection, output ring the next AttnRes polls), and the MXFP8 shared
expert on the side stream with its reduce-scatter push. A layer this path cannot
reproduce raises ``Unsupported`` at build (``nvfp4_moe_cause``); nothing falls back to
the stock layer at run."""

import copy

import torch

from ...shapes import Choice, Path, Unsupported

TOPK, LATENT, HIDDEN = 16, 3584, 7168


def nvfp4_experts_cause(routed):
    """Why ``routed`` (a MoE layer's ``RoutedExperts``) is not the stock NVFP4 W4A4
    experts ``Nvfp4RoutedExperts`` runs, or None."""
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import NvFp4MoeBackend
    from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4FusedMoE

    method = getattr(routed, "quant_method", None)
    if not isinstance(method, ModelOptNvFp4FusedMoE):
        return f"{type(method).__name__} is not the ModelOpt NVFP4 experts method"
    if method.nvfp4_backend != NvFp4MoeBackend.FLASHINFER_TRTLLM:
        return f"NVFP4 backend {method.nvfp4_backend.value} (needs FLASHINFER_TRTLLM)"
    if method.use_a16:
        return "W4A16 NVFP4 experts (the decode runs W4A4)"
    if method.moe_quant_config is None or method.moe_kernel is None:
        return "the experts' kernel is not built (weights not processed)"
    if not method.uses_input_scales:
        return "no activation input scales"
    config = routed.moe_config
    if config.moe_parallel_config.use_ep or routed.expert_map is not None:
        return "expert parallelism (the TP decode needs every expert local)"
    if routed.local_num_experts != routed.global_num_experts:
        return f"{routed.local_num_experts} of {routed.global_num_experts} experts"
    if routed.activation != MoEActivation.SITU:
        return f"activation {routed.activation} (Kimi K3: SiTU)"
    if routed.apply_router_weight_on_input:
        return "router weights applied on the input"
    if config.experts_per_token != TOPK:
        return f"top-{config.experts_per_token} (needs top-{TOPK})"
    if (config.hidden_dim_unpadded or config.hidden_dim) != LATENT:
        return f"hidden {config.hidden_dim} (latent {LATENT})"
    for name in ("w13_bias", "w2_bias"):
        if getattr(routed, name, None) is not None:
            return f"expert bias {name}"
    return None


class Nvfp4RoutedExperts:
    """The stock NVFP4 routed experts on given routes (module docstring). ``run``
    returns this rank's routed partial [M, 3584] BF16: the top-k weighted sum over the
    rank's share of the intermediate dimension, before the TP reduction, the routed
    norm and the up projection. Build outside CUDA graph captures."""

    def __init__(self, routed):
        from vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe import (
            TrtLlmNvFp4ExpertsModular,
        )
        from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
            NvFp4MoeBackend,
            make_nvfp4_moe_kernel,
        )

        cause = nvfp4_experts_cause(routed)
        if cause is not None:
            where = getattr(routed, "layer_name", None) or type(routed).__name__
            raise Unsupported("moe", None, f"NVFP4 routed experts {where}: {cause}")
        method = routed.quant_method
        # A private copy of the layer's MoE config: the modular experts lower its
        # deferred-finalize cap, and these calls always finalize into their output.
        config = copy.copy(method.moe)
        config._defer_moe_finalize = False
        self.kernel = make_nvfp4_moe_kernel(
            moe_quant_config=method.moe_quant_config,
            moe_config=config,
            experts_cls=TrtLlmNvFp4ExpertsModular,
            backend=NvFp4MoeBackend.FLASHINFER_TRTLLM,
            routing_tables=routed._expert_routing_tables(),
            per_token_activation=method.per_token_activation,
        )
        # The per-expert kernel scalars the stock experts registered on the layer at
        # load (g1_scale_c, the SiTU alpha / beta, a clamp): the same tensors, not a
        # second fold (``process_weights_after_loading`` already folded the input
        # scales into w13 / w2_weight_scale_2, which the quant config references).
        stock = method.moe_kernel.fused_experts
        experts = self.kernel.fused_experts
        for name in ("g1_scale_c", "gemm1_alpha", "gemm1_beta", "gemm1_clamp_limit"):
            setattr(experts, name, getattr(stock, name))
        self.routed = routed

    def run(self, latent, topk_weights, topk_ids):
        """``latent`` [M, 3584] BF16 (the routed experts' input), ``topk_weights``
        [M, 16] and ``topk_ids`` [M, 16] int32 from the stock router."""
        routed = self.routed
        partial = self.kernel.apply(
            hidden_states=latent,
            w1=routed.w13_weight,
            w2=routed.w2_weight,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            activation=routed.activation,
            global_num_experts=routed.global_num_experts,
            expert_map=routed.expert_map,
            apply_router_weight_on_input=routed.apply_router_weight_on_input,
        )
        if not isinstance(partial, torch.Tensor):
            raise RuntimeError("Nvfp4RoutedExperts: the experts deferred their top-k")
        return partial


def mxfp8_shared_cause(shared):
    """Why ``shared`` (a MoE layer's ``KimiMLP``) is not the TP online-MXFP8 shared
    expert ``Mxfp8SharedExpert`` reproduces, or None."""
    from ....mxfp8 import is_mxfp8_linear

    if shared is None:
        return "no shared expert"
    linears = (shared.gate_up_proj, shared.down_proj)
    if not all(is_mxfp8_linear(linear) for linear in linears):
        return "not online MXFP8 (gate_up and down)"
    if getattr(shared, "shard_sequence_parallel", False):
        return "a sequence-parallel sharded shared expert"
    if getattr(shared, "gemm_rs_ar", None) is not None:
        return "a GEMM-RS/AR shared down projection"
    if shared.down_proj.reduce_results:
        return "a reducing down projection (the tail reduces the shared partial)"
    if not shared._act_mxfp8_ok():
        return "the activation cannot write the down projection's MXFP8 input"
    return None


class Mxfp8SharedExpert:
    """Stock ``KimiMLP.forward`` of a TP online-MXFP8 shared expert (module docstring)
    at the prepared ``rows``: ``run(x [R, 7168] BF16, out [R, 7168] BF16)`` writes this
    rank's unreduced partial."""

    def __init__(self, shared, rows, *, cap=None):
        from ..gemm.mxfp8 import Mxfp8Projection

        cause = mxfp8_shared_cause(shared)
        if cause is not None:
            where = getattr(shared.gate_up_proj, "prefix", "shared experts")
            raise Unsupported("moe", None, f"MXFP8 shared expert {where}: {cause}")
        self.shared, self.rows = shared, tuple(sorted(set(rows)))
        # Families of their own: their MXFP8 input scratch is shared by every MoE
        # layer's projection of that family (the SP decode's shared_gu / shared_down
        # take another K).
        self.gate_up = Mxfp8Projection(
            shared.gate_up_proj, self.rows, family="tp_shared_gu", cap=cap
        )
        self.down = Mxfp8Projection(
            shared.down_proj, self.rows, family="tp_shared_down", cap=cap
        )
        self.inter2, self.hidden = self.gate_up.n, self.down.n
        if self.down.k * 2 != self.inter2 or self.gate_up.k != self.hidden:
            raise Unsupported(
                "moe", None, f"MXFP8 shared expert gate_up [{self.inter2}, "
                f"{self.gate_up.k}] / down [{self.hidden}, {self.down.k}]",
            )  # fmt: skip
        cap = cap or self.rows[-1]
        device = shared.gate_up_proj.weight.device
        self.gate_up_out = torch.empty(
            (cap, self.inter2), dtype=torch.bfloat16, device=device
        )

    def run(self, x, out):
        n = x.shape[0]
        gate_up = self.gate_up.run(x, self.gate_up_out[:n])
        act = self.down.input.view(n)
        self.shared.act_mxfp8_into(gate_up, act)
        return self.down.run(act, out)

    def choice(self, rows):
        gu, dn = self.gate_up.choice(rows), self.down.choice(rows)
        return f"{gu.note} -> act_mxfp8 -> {dn.note}"


def nvfp4_moe_cause(module):
    """Why a MoE layer (stock ``KimiMoE``) is not the NVFP4 layer ``Nvfp4MoE`` runs,
    or None. None of its parts may be another format: a BF16 / MXFP4 layer keeps the
    tuned providers."""
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    if getattr(module, "use_mega_moe", False):
        return "MegaMoE (expert parallel)"
    if getattr(module, "moe_token_shard", 0):
        return "a token-sharded MoE"
    if getattr(module, "_packed_moe", None) is not None:
        return "packed MoE projections"
    runner = getattr(module, "experts", None)
    routed = getattr(runner, "routed_experts", None)
    if routed is None:
        return "no routed experts"
    cause = nvfp4_experts_cause(routed)
    if cause is not None:
        return cause
    cause = mxfp8_shared_cause(module.shared_experts)
    if cause is not None:
        return f"shared expert: {cause}"
    transform = getattr(module, "routed_output_transform", None)
    down = getattr(module, "routed_expert_down_proj", None)
    if transform is None or transform.norm is None or down is None:
        return "no latent MoE projections with the routed norm"
    for linear in (down, transform.up_proj, module.gate):
        method = getattr(linear, "quant_method", None)
        if method is not None and not isinstance(method, UnquantizedLinearMethod):
            return f"quantized {type(method).__name__} latent / router linear"
        if linear.weight.dtype != torch.bfloat16:
            return f"{linear.weight.dtype} latent / router weight"
    return None


def is_nvfp4_moe(module):
    """A MoE layer with NVFP4 routed experts (the NVFP4 checkpoint), whether or not
    ``Nvfp4MoE`` can run it (it refuses the others at build)."""
    from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4FusedMoE

    routed = getattr(getattr(module, "experts", None), "routed_experts", None)
    return isinstance(getattr(routed, "quant_method", None), ModelOptNvFp4FusedMoE)


class Nvfp4MoE:
    """One NVFP4 MoE layer's provider for every captured MoE batch (module
    docstring). ``rows``: the layer's wide tail (``tail_rows.PreparedTailRows``) over
    the model's ``TailRowsBox``; ``state``: the model's ``MoEScratch`` (side stream and
    its events)."""

    def __init__(self, module, state, rows, batches, *, prefetch=None, layer_idx=None):
        from .prepared import stock_router

        cause = nvfp4_moe_cause(module)
        if cause is not None:
            raise Unsupported("moe", None, f"NVFP4 MoE layer: {cause}")
        self.module, self.state, self.rows = module, state, rows
        self.prefetch, self.layer_idx = prefetch, layer_idx
        self.batches = frozenset(batches)
        self.route = stock_router(module.experts.router)
        self.experts = Nvfp4RoutedExperts(module.experts.routed_experts)
        cap = max(self.batches)
        self.shared = Mxfp8SharedExpert(module.shared_experts, self.batches, cap=cap)
        self.shared_out = torch.empty(
            (cap, HIDDEN), dtype=torch.bfloat16, device=state.device
        )

    def _shared(self, x):
        shared = self.shared_out[: x.shape[0]]
        self.shared.run(x, shared)
        self.rows.push(shared)

    def run(self, x, out, *, consume=None):
        """The layer on ``x`` [M, 7168]: its output lands in the wide tail's ring, which
        ``consume`` (the next AttnRes's ``run_ring``) reads."""
        from .prepared import PreparedMoE

        b, module = x.shape[0], self.module
        if b not in self.batches:
            raise Unsupported("moe", b, f"NVFP4 MoE: not prepared ({sorted(self.batches)})")
        if consume is None:
            raise RuntimeError("NVFP4 MoE: the output must be consumed from its ring")
        # (the shared rows' RS push on the side stream; joined before NormRows)
        with PreparedMoE.beside(self, lambda: self._shared(x)) as join:
            latent, logits, _ = module._maybe_overlap_router_and_down_proj(x)
            weights, ids = self.route(logits)
            partial = self.experts.run(latent, weights, ids)
            if self.prefetch is not None:  # (as PreparedMoE.finish: after the experts)
                self.prefetch.fork("next_attention", self.layer_idx, b)
            self.rows.run(partial, b, join)
        if consume(out) is not out:
            raise RuntimeError("The wide tail output must be consumed from its ring")
        return out

    def choice(self, rows):
        return Choice(
            Path.GENERIC,
            ("nvfp4", "stock-front"),
            f"stock front + routes, NVFP4 TRT-LLM-Gen routed experts, MXFP8 shared "
            f"({self.shared.choice(rows)}), wide tail",
        )
