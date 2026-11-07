# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The SM107 specialized decode's model state: the stock Kimi K3 model state plus the
source branch's fast spec-decode verification.

On the source branch the runner's ``sample`` asked the rejection sampler whether a
verify batch could skip the full-vocabulary logits (``can_verify_greedy``: every
request greedy; ``can_verify_sharded``: greedy and temperature-1 requests) and then
called the model's ``get_top_tokens`` / ``verify_sampled``. This tree's runner has
two model-state hooks around the same point, so the runner stays unchanged:

- ``compute_sample_logits`` (the runner's logits projection) decides from the step's
  batch, recorded by ``prepare_inputs``, with the source branch's predicates. On a
  hit it keeps the hidden states and returns an empty ``[rows, 0]`` logits tensor
  (nothing launched: the full-vocabulary logits and their all-gather are skipped).
- ``custom_sampler`` installs ``Sm107RejectionSampler``, whose call runs the fast
  verification on the kept hidden states. Where the model declines (None, nothing
  launched) it computes the stock logits from them and verifies on the stock path,
  as the source branch's runner did.

Every decision depends only on host state that all TP ranks share, so the ranks
always take the same path.
"""

from typing import Any

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.models.kimi_k3.nvidia.model_state import KimiK3ModelState
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

logger = init_logger(__name__)

GREEDY, SHARDED = "greedy", "sharded"


class Sm107KimiK3ModelState(KimiK3ModelState):
    def __init__(self, vllm_config, model, encoder_cache, device):
        super().__init__(vllm_config, model, encoder_cache, device)
        # Requests with structured outputs (req_id -> state index): their logits
        # take a grammar bitmask, so their batches keep the stock logits.
        self._structured: dict[str, int] = {}
        # The step's InputBatch (prepare_inputs), read once by compute_sample_logits.
        self._batch = None
        # (kind, sample hidden states, batch) of a verify step on the fast path.
        self._stash: tuple[str, torch.Tensor, Any] | None = None
        self._rejection_sampler: Sm107RejectionSampler | None = None

    # -- request bookkeeping -----------------------------------------------------
    def add_request(self, req_index, new_req_data) -> None:
        super().add_request(req_index, new_req_data)
        params = new_req_data.sampling_params
        if params is not None and params.structured_outputs is not None:
            self._structured[new_req_data.req_id] = req_index
        else:
            self._structured.pop(new_req_data.req_id, None)

    def remove_request(self, req_id: str) -> None:
        super().remove_request(req_id)
        self._structured.pop(req_id, None)

    def prepare_inputs(self, input_batch, req_states):
        self._batch = input_batch
        return super().prepare_inputs(input_batch, req_states)

    # -- sampler hooks ---------------------------------------------------------------
    def custom_sampler(self, sampler: Any) -> tuple[Any, Any] | None:
        lm = self.language_model
        spec = self.vllm_config.speculative_config
        if lm.specialized_decode is None or spec is None:
            return super().custom_sampler(sampler)
        if self.vllm_config.watermark_config is not None:
            # The source branch's predicates refuse watermarked verification, and
            # the drafter refuses watermarked drafts (DraftFastPath.build).
            return super().custom_sampler(sampler)
        self._rejection_sampler = Sm107RejectionSampler(
            sampler, spec, self.device, state=self
        )
        return sampler, self._rejection_sampler

    def compute_sample_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, self._batch = self._batch, None
        self._stash = None
        kind = self._fast_kind(batch, hidden_states)
        if kind is None:
            return super().compute_sample_logits(hidden_states)
        self._stash = (kind, hidden_states, batch)
        return hidden_states.new_empty((hidden_states.shape[0], 0))

    def take_stash(self):
        stash, self._stash = self._stash, None
        return stash

    def _fast_kind(self, batch, hidden_states) -> str | None:
        """The source branch's runner conditions for the fast verification of this
        step's logits rows (None: the stock logits)."""
        rs, lm = self._rejection_sampler, self.language_model
        if (
            rs is None
            or batch is None
            or batch.num_reqs == 0
            or batch.num_draft_tokens <= 0
            or hidden_states.shape[0] != batch.logits_indices.shape[0]
            or self._has_structured(batch)
        ):
            return None
        if lm.verify_logits is not None and rs.can_verify_greedy(batch):
            return GREEDY
        if rs.can_verify_sharded(batch):
            return SHARDED
        return None

    def _has_structured(self, batch) -> bool:
        if not self._structured:
            return False
        live = np.fromiter(self._structured.values(), dtype=np.int64)
        return bool(np.isin(batch.idx_mapping_np, live).any())


class Sm107RejectionSampler(RejectionSampler):
    """The stock rejection sampler with the source branch's fast verifications
    (``verify_greedy`` from the target argmax, ``verify_sharded`` from this rank's
    vocabulary shard) for the steps ``Sm107KimiK3ModelState.compute_sample_logits``
    kept on the fast path."""

    def __init__(self, sampler, spec_config, device, *, state):
        super().__init__(sampler, spec_config, device, watermark_key=None)
        self._state = state

    @property
    def _lm(self):
        return self._state.language_model

    def _plain(self) -> bool:
        """No watermark, adaptive verification, FLy windows, NaN counting or
        sampling mask (this tree's ``fly`` method did not exist on the source
        branch; it reads full logits)."""
        sampler = self.sampler
        return (
            self.watermark_key is None
            and not self.enable_adaptive_verification
            and self.fly_window_size == 0
            and not sampler.compute_nans
            and not sampler.return_sampling_mask
        )

    def can_verify_greedy(self, input_batch) -> bool:
        """Every request greedy and nothing else reads or changes the logits."""
        sampler = self.sampler
        idx = input_batch.idx_mapping_np
        return (
            self._lm.verify_logits is not None
            and self._plain()
            and bool(np.all(sampler.sampling_states.temperature.np[idx] == 0.0))
            and not np.any(sampler.needs_logits_processing[idx])
            and sampler.sampling_states.max_num_logprobs(idx) == NO_LOGPROBS
        )

    def can_verify_sharded(self, input_batch) -> bool:
        """A decode batch (every request with ``num_speculative_steps`` drafts) of
        greedy or temperature-1 requests with no other logits processing, no
        logprobs, FP32 draws."""
        lm = self._lm
        sampler = self.sampler
        idx = input_batch.idx_mapping_np
        num_reqs = input_batch.num_reqs
        return (
            lm.sampled_verify is not None
            and lm.verify_logits is not None
            and num_reqs > 0
            and self._plain()
            and not sampler.use_fp64_gumbel
            and not np.any(sampler.needs_logits_processing[idx])
            and sampler.sampling_states.max_num_logprobs(idx) == NO_LOGPROBS
            and np.array_equal(
                input_batch.cu_num_logits_np,
                np.arange(num_reqs + 1) * (self.num_speculative_steps + 1),
            )
        )

    def verify_greedy(self, target_argmax, input_batch) -> SamplerOutput:
        from .compat.greedy_verify import greedy_verify

        sampled, num_sampled, num_rejected = greedy_verify(
            target_argmax,
            input_batch.input_ids,
            input_batch.positions,
            input_batch.logits_indices,
            input_batch.expanded_local_pos,
            input_batch.cu_num_logits,
            input_batch.idx_mapping,
            input_batch.seq_lens,
            self.sampler.req_states.prefill_len.gpu,
            self.sampler.sampling_states.seeds.gpu,
            self.num_speculative_steps,
            self.synthetic_conditional_rates,
        )
        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )

    def verify_sharded(self, hidden_states, input_batch, draft_logits):
        from .integration import verify_sampled

        if draft_logits is not None and draft_logits.row_mapping is not None:
            return None  # (the model reads draft logits by request state index)
        sampler = self.sampler
        result = verify_sampled(
            self._lm,
            hidden_states,
            self.num_speculative_steps,
            input_ids=input_batch.input_ids,
            positions=input_batch.positions,
            logits_indices=input_batch.logits_indices,
            expanded_local_pos=input_batch.expanded_local_pos,
            idx_mapping=input_batch.idx_mapping,
            seq_lens=input_batch.seq_lens,
            prefill_len=sampler.req_states.prefill_len.gpu,
            temperature=sampler.sampling_states.temperature.gpu,
            seeds=sampler.sampling_states.seeds.gpu,
            draft_logits=None if draft_logits is None else draft_logits.values,
            synthetic_rates=self.synthetic_conditional_rates,
            block_verification=self.use_block_verification,
        )
        if result is None:
            return None
        sampled, num_sampled, num_rejected = result
        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )

    def __call__(self, logits, input_batch, draft_logits=None) -> SamplerOutput:
        stash = self._state.take_stash()
        if stash is not None:
            kind, hidden, batch = stash
            if batch is not input_batch:
                # A wiring bug (a runner feature rewrote the batch between
                # compute_sample_logits and the sampler), not a step the model can
                # decline: refuse rather than verify on the stock sampler.
                raise RuntimeError(
                    "Kimi K3 specialized decode (SM107): the rejection sampler got a "
                    "different InputBatch than the one whose logits rows were kept "
                    "on the fast verification path"
                )
            if kind == GREEDY:
                from .integration import get_top_tokens

                top = get_top_tokens(self._lm, hidden)
                if top is not None:
                    return self.verify_greedy(top, input_batch)
            else:
                out = self.verify_sharded(hidden, input_batch, draft_logits)
                if out is not None:
                    return out
            # The model declined (nothing launched): the stock logits.
            logits = self._state.model.compute_logits(hidden)
        self._complete_draft_logits(input_batch, draft_logits)
        return super().__call__(logits, input_batch, draft_logits)

    @staticmethod
    def _complete_draft_logits(input_batch, draft_logits) -> None:
        """Before the stock rejection sampler reads the draft logits: a drafter
        whose fused probabilistic drafting wrote only this rank's vocabulary columns
        completes those rows (the source branch's ``prepare_draft_logits``)."""
        if draft_logits is None or draft_logits.row_mapping is not None:
            return
        from .drafter import current_drafter

        drafter = current_drafter()
        if drafter is not None and drafter.shards_draft_logits:
            drafter.gather_draft_logits(draft_logits.values, input_batch.idx_mapping)
