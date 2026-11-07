# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""How the SM107 specialized decode plugs into this tree's Kimi K3 model.

The source branch built its specialized decode from edits to the stock model
(``process_weights_after_loading``: the MegaMoE header patch and the DcpExchange;
``build_specialized_decode``: the DSA sparse core, the decode, the vocabulary-sharded
verification), its model state, its runner and its rejection sampler. Here the model
only asks the decode config to build itself (``Sm107DecodeConfig.build``) and for its
model state class (``Sm107DecodeConfig.model_state_cls``); everything else lives in
this package:

- ``build``: the vendored DCP top-k merge (round-1 mailbox), the MegaMoE header
  patch, DcpExchange, the DSA sparse core, then ``Sm107SpecializedDecode`` and the
  verification pieces (``VerifyArgmax``, ``SampledVerify``, ``warm_sampling``), in
  the source branch's order.
- ``model_state.Sm107KimiK3ModelState``: greedy and vocabulary-sharded verification
  through this tree's model-state sampler hooks (``custom_sampler``,
  ``compute_sample_logits``) instead of runner edits.
- ``drafter``: the DSpark drafter's fast path as a registered drafter subclass.
"""

from dataclasses import dataclass, fields
from typing import Any

import torch

from vllm.logger import init_logger

from .runtime import DecodeConfig
from .runtime.decode import SpecializedDecode

logger = init_logger(__name__)


class Sm107SpecializedDecode(SpecializedDecode):
    """The source branch's ``SpecializedDecode`` with the model-state binding this
    tree's ``KimiK3ModelState`` calls after the build. The SM107 decode keeps no
    model-state hooks (RecoverSSM runs undeferred under SP): it only remembers it."""

    model_state = None

    def bind_model_state(self, state) -> None:
        import weakref

        self.model_state = weakref.ref(state)


@dataclass(frozen=True)
class Sm107DecodeConfig(DecodeConfig):
    """``runtime.DecodeConfig`` that builds the SM107 decode (``build``) and names
    its model state (``model_state_cls``)."""

    @property
    def model_state_cls(self):
        from .model_state import Sm107KimiK3ModelState

        return Sm107KimiK3ModelState

    def build(self, lm) -> None:
        """Build the decode on ``lm`` (KimiLinearForCausalLM, weights loaded, aux
        layers set): every rank together, outside CUDA graph capture."""
        backbone, vllm_config = lm.model, lm.vllm_config
        # The DSA indexer runs at stock Kimi K3 precision (FP32 logits): no
        # one-round DCP top-k merge (#63, BF16 logits only) is set up; the DCP
        # top-k is the FP32 local top-K and merge (cutedsl/dsa/dcp_topk.py).
        # MegaMoE wide-EP kernel header / expert placement (env-selected), before
        # the runner's warmup compiles MegaMoE.
        from .megamoe import prepare_megamoe

        prepare_megamoe(backbone, vllm_config)
        # DSA decode under DCP: DcpExchange (VLLM_K31_DCP_EXCHANGE, off by default)
        # selects BF16 indexer logits only; stock Kimi K3's are FP32, which the
        # FP32 DCP top-k serves. Switched on, it raises.
        from .dsa_kernels.dcp_exchange.decode import ENV as DCPX_ENV
        from .dsa_kernels.dcp_exchange.decode import wanted as dcpx_wanted

        if dcpx_wanted(vllm_config):
            raise ValueError(
                f"Kimi K3 specialized decode (SM107): {DCPX_ENV} turns on DcpExchange, "
                "whose top-k selects BF16 indexer logits; stock Kimi K3's are FP32 "
                "(the FP32 DCP top-k serves DCP): unset it"
            )
        dcpx = None
        # The DSA sparse core (VLLM_K31_DSA_CORE=1), which the DSA layers'
        # PreparedDsa picks up from their indexers.
        from .dsa_kernels.dsa_core.decode import build as build_dsa_core

        build_dsa_core(backbone)
        decode = Sm107SpecializedDecode(backbone, self)
        decode.dcpx = dcpx
        lm.specialized_decode = decode
        # From here on forward() returns normalized hidden states on every step.
        lm.needs_norm_in_logits = False
        from .runtime.verify import VerifyArgmax

        lm.verify_logits = VerifyArgmax.build(vllm_config, lm)
        lm.sampled_verify = None
        if lm.verify_logits is not None:  # Rubin, speculating
            from .runtime.sampled_verify import SampledVerify
            from .runtime.warmup import warm_sampling

            lm.sampled_verify = SampledVerify.build(vllm_config, lm)
            # Compile the verification kernels now (every rank together).
            warm_sampling(vllm_config, lm)


def sm107_decode_config(vllm_config) -> Sm107DecodeConfig:
    """The SM107 decode's config (raises for an unsupported deployment), with the
    SM107 drafter registered when speculating."""
    from .runtime import specialized_decode_config

    config = specialized_decode_config(vllm_config)
    assert config is not None  # (decode_backend=latency_specialized)
    pcp = vllm_config.parallel_config.prefill_context_parallel_size
    if pcp > 1:
        # The runner rebuilds the InputBatch for sampling under PCP, which the fast
        # verification (``model_state``) cannot follow.
        raise ValueError(
            "Kimi K3 specialized decode (SM107) does not support prefill context "
            f"parallelism (PCP{pcp})"
        )
    if vllm_config.speculative_config is not None:
        from .drafter import register

        register()
    logger.info_once(
        "Kimi K3 specialized decode (SM107): verify graphs at %s rows (q%d)",
        config.captured,
        config.q,
    )
    return Sm107DecodeConfig(
        **{f.name: getattr(config, f.name) for f in fields(config)}
    )


# -- verification (the source branch's model methods) ------------------------------
def get_top_tokens(lm, hidden_states: torch.Tensor) -> torch.Tensor | None:
    """``compute_logits(hidden_states).argmax(-1)`` exactly (the lowest index among
    equal maxima) without gathering the full-vocabulary logits; every TP rank calls
    it together. None: no fast path for this call (nothing launched)."""
    if lm.verify_logits is None:
        return None
    if lm.needs_norm_in_logits:
        hidden_states = lm.model.norm(hidden_states, None)
    return lm.verify_logits.top_tokens(lm.logits_processor, lm.lm_head, hidden_states)


def verify_sampled(lm, hidden_states: torch.Tensor, steps: int, **kwargs: Any):
    """``RejectionSampler.rejection_sample``'s result for a decode batch of greedy
    and temperature-1 requests from this rank's shard of the target logits; every TP
    rank calls it together. None: no fast path for this call (nothing launched)."""
    verify = lm.sampled_verify
    if (
        verify is None
        or lm.verify_logits is None
        or not verify.takes(hidden_states.shape[0])
    ):
        return None
    if lm.needs_norm_in_logits:
        hidden_states = lm.model.norm(hidden_states, None)
    logits = lm.verify_logits.shard_logits(
        lm.logits_processor, lm.lm_head, hidden_states
    )
    return verify.run(logits, steps, **kwargs)
