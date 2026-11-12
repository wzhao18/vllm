# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3's model runner state: the hybrid (KDA + MLA) state, plus the lifecycle of
the specialized decode when ``decode_backend=latency_specialized`` selects it."""

import torch

from vllm import envs
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

logger = init_logger(__name__)


class KimiK3ModelState(MambaHybridModelState):
    """Builds the specialized decode once the weights are loaded and the aux layers
    set, and prepares it for every bound KV cache (the profiling cache, then the
    real one).

    On a steady speculative decode step (FULL graph, every request verifying its
    drafts) the step prep runs as one kernel (``step_prep.py``): the model state
    takes it over before the runner prepares the inputs (``defers_step_prep``: the
    runner skips the launches of its step inputs, block tables, slot mappings and DCP
    local lengths) and ``preprocess_state`` skips the Mamba align preprocess.
    ``prepare_attn`` then launches the fused kernel and the state pre-copy, and hands
    its outputs to the stock metadata path (the KDA builders skip their own
    staging)."""

    def __init__(self, vllm_config, model, encoder_cache, device):
        super().__init__(vllm_config, model, encoder_cache, device)
        if hasattr(model, "get_language_model"):
            model = model.get_language_model()
        self.language_model = model
        if model.specialized_decode_config is not None:
            model.build_specialized_decode()
            # Deferred RecoverSSM (the SP decode's KDA cores fold the commit).
            model.specialized_decode.bind_model_state(self)
        self._step_prep = None  # step_prep.KdaStepPrep, built at the first step
        self._step_prep_checked = False
        self._prepped = None  # (num_accepted, aligned indices) of this step
        self._block_tables = None  # the runner's BlockTables
        # (InputBuffers, RequestState) of a step whose prep this state took over.
        self._deferred = None
        # That step's staged device copies of the host arrays its prep reads.
        self._staged = None
        # The runner's (InputBuffers, RequestState), seen at any step: the fused step
        # prep compiles against them as soon as it is built (_compile_step_prep).
        self._io = None

    def initialize_kv_cache(self, kv_cache_config, block_tables):
        super().initialize_kv_cache(kv_cache_config, block_tables)
        self._block_tables = block_tables
        decode = self.language_model.specialized_decode
        if decode is not None:
            decode.prepare_kv()
        self._step_prep = None
        self._step_prep_checked = False

    def _build_step_prep(self, attn_groups, kv_cache_config):
        """The fused KDA step prep, when every aligned-index builder is a K3 KDA
        builder that stages FULL-graph buffers without RecoverSSM."""
        from .kda_metadata import KimiK3KDAMetadataBuilder
        from .step_prep import KdaStepPrep

        self._step_prep_checked = True
        if (
            not self._align_mode
            or self._mamba_ctx is None
            or self.recoverssm is not None
            or self.vllm_config.num_speculative_tokens <= 0
        ):
            return None
        mamba_group_ids, mamba_spec = self._get_mamba_group_info(kv_cache_config)
        builders = []
        for group_idx, group_id in enumerate(mamba_group_ids):
            for group in attn_groups[group_id]:
                builder = group.get_metadata_builder(0)
                if not hasattr(builder, "mamba_aligned_state_indices"):
                    continue
                if (
                    not isinstance(builder, KimiK3KDAMetadataBuilder)
                    or builder.use_recoverssm
                    or not builder.use_full_cuda_graph
                ):
                    return None
                builders.append((group_idx, builder))
        if not builders:
            return None
        # Attention groups whose kernel block tables the launch maps too.
        kernel_groups = [
            group
            for groups in attn_groups
            for group in groups
            if group.kernel_block_size is not None
        ]
        logger.info_once(
            "Kimi K3: fused KDA step prep for %d KDA metadata builders and %d kernel "
            "block tables",
            len(builders),
            len(kernel_groups),
        )
        return KdaStepPrep(
            self._mamba_ctx,
            builders,
            mamba_spec.block_size,
            self.device,
            kernel_groups,
        )

    def defers_step_prep(self, batch_req_state, batch_desc, input_buffers, req_states):
        """A steady spec-decode step (every request verifies drafts, no prefill) in
        a FULL graph that the fused step prep covers."""
        self._io = (input_buffers, req_states)
        step_prep = self._step_prep
        if (
            step_prep is None
            or batch_desc.cg_mode != CUDAGraphMode.FULL
            or batch_desc.num_ubatches != 1
            or batch_req_state.has_prefill
            or batch_req_state.num_draft_tokens_np is None
        ):
            return False
        num_rows = batch_desc.num_reqs or len(batch_req_state.req_ids)
        num_tokens = batch_req_state.num_tokens
        scheduled = batch_req_state.num_scheduled_tokens
        max_query_len = self.vllm_config.num_speculative_tokens + 1
        if (
            num_rows > step_prep.capacity
            or any(num_tokens > b.decode_cudagraph_max_bs for b in step_prep.builders)
            or not (scheduled > 0).all()
            or not (scheduled <= max_query_len).all()
        ):
            return False
        self._deferred = (input_buffers, req_states)
        logger.info_once(
            "Kimi K3: fused step prep fills the step inputs, block tables, slot "
            "mappings and DCP local lengths"
        )
        return True

    def stage_step_inputs(self, h2d):
        """The deferred step prep's host arrays (the block tables' per-group block
        counts and the requests' prefill lengths: pinned host memory, read through
        UVA), staged with the step's other inputs so the fused kernel reads device
        copies. Each of its row programs reads them in dependent loads, and a rank
        whose pinned buffers sit on the other socket paid a remote round trip per
        level (16 ranks, DL3 B32: 19.4 against 6.7 us per launch; the slowest rank
        holds every rank's next verify step). 1 GPU, graph of 50, 2-32 requests:
        3.3-3.8 us from the copies against 3.9-4.6 (local) / 4.7-6.2 (remote) UVA."""
        if self._deferred is None:
            return
        _, req_states = self._deferred
        self._staged = (
            h2d.stage(self._block_tables.num_blocks.np, torch.int32),
            h2d.stage(req_states.prefill_len.np, torch.int32),
        )

    def preprocess_state(
        self, input_batch, block_tables, kv_cache_config, num_computed_tokens
    ):
        if self._deferred is None:  # else prepare_attn runs it, fused
            super().preprocess_state(
                input_batch, block_tables, kv_cache_config, num_computed_tokens
            )

    def _run_step_prep(
        self, input_batch, input_buffers, req_states, compile_only=False
    ):
        """The fused step prep (filling everything ``defers_step_prep`` deferred)
        and the state pre-copy; the KDA builders take its staged buffers.
        ``compile_only``: compile the fused kernel for this deployment (scratch
        stand-ins for the staged copies; nothing launches or is written), so the
        first steady decode step does not stop for the JIT."""
        from .step_prep import StepInputs

        step_prep, bt = self._step_prep, self._block_tables
        if compile_only:
            num_blocks = torch.empty(
                bt.num_blocks.np.shape, dtype=torch.int32, device=self.device
            )
            prefill_len = torch.empty(
                req_states.prefill_len.np.shape, dtype=torch.int32, device=self.device
            )
        else:
            num_blocks, prefill_len = self._staged
            self._staged = None
        num_rows = input_batch.num_reqs_after_padding
        num_new_sampled = self.num_new_sampled_tokens_per_step
        inputs = StepInputs(
            block_tables=bt,
            dcp_local_seq_lens=(
                input_buffers.dcp_local_seq_lens if bt.cp_size > 1 else None
            ),
            cu_num_logits=input_batch.cu_num_logits,
            expanded_idx_mapping=input_batch.expanded_idx_mapping,
            expanded_local_pos=input_batch.expanded_local_pos,
            logits_indices=input_batch.logits_indices,
            input_ids=input_buffers.input_ids,
            positions=input_buffers.positions,
            seq_lens=input_buffers.seq_lens,
            last_sampled_tokens=req_states.last_sampled_tokens,
            num_blocks=num_blocks,
            prefill_len=prefill_len,
            draft_tokens=req_states.draft_tokens,
            num_new_sampled=num_new_sampled,
            max_expand_len=self.vllm_config.num_speculative_tokens + num_new_sampled,
            # prepare_inputs fills is_padding under the same condition.
            is_padding=input_buffers.is_padding if envs.VLLM_MOE_SKIP_PADDING else None,
            num_tokens_padded=input_batch.num_tokens_after_padding,
        )
        self._prepped = step_prep.run(
            input_batch,
            num_rows,
            req_states.num_computed_tokens.gpu,
            self._mamba_state_idx_gpu,
            self.num_accepted_tokens_gpu,
            self._mamba_src_col_gpu,
            self._mamba_src_off_gpu,
            inputs,
            compile_only=compile_only,
        )
        if compile_only:
            self._prepped = None
            return
        self._mamba_ctx.run_fused_precopy(
            input_batch.num_reqs,
            self._mamba_state_idx_gpu,
            self._mamba_src_col_gpu,
            self._mamba_src_off_gpu,
            input_batch.idx_mapping,
        )
        for builder in step_prep.builders:
            builder.prestaged = (num_rows, input_batch.num_reqs)
        for group in step_prep.kernel_groups:
            group.kernel_block_table_prestaged = True

    def prepare_attn(
        self,
        input_batch,
        cudagraph_mode,
        block_tables,
        slot_mappings,
        attn_groups,
        kv_cache_config,
        for_capture=False,
        ubatch_idx=0,
        model_specific_attn_metadata=None,
    ):
        if not self._step_prep_checked and not for_capture:
            self._step_prep = self._build_step_prep(attn_groups, kv_cache_config)
            # Rubin (the sampled / greedy verification paths are built): compile the
            # fused step prep now rather than at the first steady decode step.
            if (
                self._step_prep is not None
                and self._io is not None
                and self.language_model.verify_logits is not None
            ):
                self._run_step_prep(input_batch, *self._io, compile_only=True)
        deferred, self._deferred = self._deferred, None
        if deferred is not None:
            self._run_step_prep(input_batch, *deferred)
        try:
            return super().prepare_attn(
                input_batch,
                cudagraph_mode,
                block_tables,
                slot_mappings,
                attn_groups,
                kv_cache_config,
                for_capture=for_capture,
                ubatch_idx=ubatch_idx,
                model_specific_attn_metadata=model_specific_attn_metadata,
            )
        finally:
            self._prepped = None
            if self._step_prep is not None:
                for builder in self._step_prep.builders:
                    builder.prestaged = None
                for group in self._step_prep.kernel_groups:
                    group.kernel_block_table_prestaged = False

    def gather_num_accepted_tokens(self, input_batch, num_reqs):
        if self._prepped is not None:
            assert self._prepped[0].shape[0] == num_reqs
            return self._prepped[0]
        return super().gather_num_accepted_tokens(input_batch, num_reqs)

    def compute_aligned_state_indices(self, input_batch, num_reqs):
        if self._prepped is not None:
            assert self._prepped[1].shape[1] == num_reqs
            return self._prepped[1]
        return super().compute_aligned_state_indices(input_batch, num_reqs)
