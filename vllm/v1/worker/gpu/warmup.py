# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from typing import Any

import numpy as np
import torch

from vllm import PoolingParams, SamplingParams
from vllm.config.model import PROCESSED_LOGPROBS_MODES
from vllm.logger import init_logger
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.platforms import current_platform
from vllm.sampling_params import MAX_LOGPROB_TOKEN_IDS
from vllm.utils.math_utils import cdiv
from vllm.v1.core.sched.output import (
    CachedRequestData,
    GrammarOutput,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    CrossAttentionSpec,
    KVCacheSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.request import Request
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.sample.logprob import (
    warmup_logprob_token_ids,
    warmup_token_logprobs,
)

logger = init_logger(__name__)


def _reserved_block_count(
    num_tokens: int,
    kvcache_spec: KVCacheSpec,
    *,
    num_lookahead_tokens: int,
    max_model_len: int,
    max_encoder_len: int,
    dcp_world_size: int = 1,
) -> int:
    """Number of blocks the scheduler would hold for a request of `num_tokens`.

    Warmup hand-builds its `SchedulerOutput`s, so it must reserve what
    `KVCacheManager.allocate_slots` reserves: the token range plus
    `num_lookahead_tokens`, where the speculator writes the KV of its drafts.
    A DCP-sharded group's block holds `block_size * dcp_world_size` tokens of
    the request (each rank stores its `block_size` share), as in
    `SingleTypeKVCacheManager.block_size` and the worker's block-table rows.
    """
    if isinstance(kvcache_spec, UniformTypeKVCacheSpecs):
        kvcache_spec = kvcache_spec.first_spec
    block_size = kvcache_spec.block_size
    if kvcache_spec.dcp_sharded:
        block_size *= dcp_world_size
    if isinstance(kvcache_spec, CircularBufferSpec):
        # Circular caches keep one physical ring block for the request lifetime.
        return 1
    if isinstance(kvcache_spec, CrossAttentionSpec):
        # Cross-attention blocks cover the encoder sequence only.
        return cdiv(max_encoder_len, block_size)
    num_speculative_blocks = 0
    if isinstance(kvcache_spec, MambaSpec):
        # MambaManager appends speculative running-state blocks in every cache
        # mode; align mode sizes from the uncapped, lookahead-free token range.
        num_speculative_blocks = kvcache_spec.num_speculative_blocks
        if kvcache_spec.mamba_cache_mode == "align":
            return cdiv(num_tokens, block_size) + num_speculative_blocks
    num_tokens = min(num_tokens + num_lookahead_tokens, max_model_len)
    return cdiv(num_tokens, block_size) + num_speculative_blocks


def _warmup_block_counter(
    model_runner: GPUModelRunner,
) -> Callable[[int, KVCacheSpec], int]:
    """Bind `_reserved_block_count` to `model_runner`'s reservation policy."""
    num_lookahead_tokens = model_runner.vllm_config.num_lookahead_tokens
    max_model_len = model_runner.max_model_len
    max_encoder_len = getattr(model_runner.model_state, "max_encoder_len", 0)
    dcp_world_size = model_runner.dcp_size

    def block_count(num_tokens: int, kvcache_spec: KVCacheSpec) -> int:
        return _reserved_block_count(
            num_tokens,
            kvcache_spec,
            num_lookahead_tokens=num_lookahead_tokens,
            max_model_len=max_model_len,
            max_encoder_len=max_encoder_len,
            dcp_world_size=dcp_world_size,
        )

    return block_count


class _WarmupBlocks:
    """Block reservations of hand-built warmup steps, in the scheduler's units.

    ``counts`` is what each KV cache group's manager holds for a request
    (``_reserved_block_count``). A slot-packed group (``blocks_per_pool_block``
    S > 1, e.g. a draft's small blocks packed into the block-outermost rows)
    addresses slot ``j`` in pool block ``j // S`` and takes pool blocks on
    demand, so its blocks cost ``cdiv(count, S)`` pool blocks, not ``count``.
    ``pool_blocks`` (the cost against ``KVCacheConfig.num_blocks``) and
    ``alloc`` (the ids handed to the worker) use that one rule: each request's
    group takes whole pool blocks from one cursor (pool block 0, the null
    block, is never handed out) and its block ids are the slots inside them.
    """

    def __init__(self, model_runner: GPUModelRunner) -> None:
        self._count = _warmup_block_counter(model_runner)
        groups = model_runner.kv_cache_config.kv_cache_groups
        self.specs = [g.kv_cache_spec for g in groups]
        self.slots = [max(1, getattr(g, "blocks_per_pool_block", 1)) for g in groups]
        self.next_pool_block = 1

    def counts(self, num_tokens: int) -> list[int]:
        return [self._count(num_tokens, spec) for spec in self.specs]

    def pool_blocks_of(self, counts: list[int]) -> int:
        return sum(cdiv(n, s) for n, s in zip(counts, self.slots))

    def pool_blocks(self, num_tokens: int) -> int:
        return self.pool_blocks_of(self.counts(num_tokens))

    def grown_pool_blocks(self, start: list[int], end: list[int]) -> int:
        """Upper bound on the pool blocks of a request allocated ``start``
        blocks, then grown to ``end`` by ``alloc_counts`` of the deltas (each
        nonzero delta of a slot-packed group takes at least one pool block)."""
        return sum(cdiv(a, s) + b - a for a, b, s in zip(start, end, self.slots))

    def alloc_counts(self, counts: list[int]) -> tuple[list[int], ...]:
        block_ids = []
        for n, s in zip(counts, self.slots):
            first = self.next_pool_block * s
            block_ids.append(list(range(first, first + n)))
            self.next_pool_block += cdiv(n, s)
        return tuple(block_ids)

    def alloc(self, num_tokens: int) -> tuple[list[int], ...]:
        return self.alloc_counts(self.counts(num_tokens))


def run_mixed_prefill_decode_warmup(
    model_runner: GPUModelRunner,
    worker_execute_model: Callable[[SchedulerOutput], Any],
    worker_sample_tokens: Callable[[GrammarOutput | None], Any],
    num_tokens: int,
    *,
    mixed_step_context: AbstractContextManager[object] | None = None,
    req_id_prefix: str = "_v2_mixed_warmup",
) -> bool:
    """Run a V2 mixed prefill+decode step through normal scheduler inputs."""
    if model_runner.is_pooling_model or model_runner.max_num_reqs < 2 or num_tokens < 3:
        return False

    decode_req_id = f"{req_id_prefix}_decode_"
    prefill_req_id = f"{req_id_prefix}_prefill_"
    decode_prompt_len = 2
    decode_scheduled_tokens = 1
    prefill_len = num_tokens - decode_scheduled_tokens
    decode_token_ids = list(range(decode_prompt_len))
    prefill_token_ids = list(range(prefill_len))

    blocks = _WarmupBlocks(model_runner)
    decode_prefill_block_counts = blocks.counts(decode_prompt_len)
    decode_block_counts = blocks.counts(decode_prompt_len + decode_scheduled_tokens)
    decode_block_deltas = [
        decode - prefill
        for decode, prefill in zip(decode_block_counts, decode_prefill_block_counts)
    ]
    prefill_block_counts = blocks.counts(prefill_len)
    required_blocks = (
        blocks.pool_blocks_of(decode_prefill_block_counts)
        + blocks.pool_blocks_of(decode_block_deltas)
        + blocks.pool_blocks_of(prefill_block_counts)
    )
    if model_runner.kv_cache_config.num_blocks <= required_blocks:
        logger.warning(
            "Skipping V2 mixed prefill+decode warmup because only %d KV blocks "
            "are available for %d required warmup blocks.",
            model_runner.kv_cache_config.num_blocks,
            required_blocks,
        )
        return False

    sampling_params = SamplingParams(
        max_tokens=2,
        temperature=0.0,
        watermarking=False,
    )

    decode_prefill_output = SchedulerOutput.make_empty()
    decode_prefill_output.num_spec_tokens_to_schedule = (
        model_runner.num_speculative_steps
    )
    decode_prefill_output.scheduled_new_reqs = [
        NewRequestData(
            req_id=decode_req_id,
            prompt_token_ids=decode_token_ids,
            mm_features=[],
            sampling_params=sampling_params,
            pooling_params=None,
            block_ids=blocks.alloc_counts(decode_prefill_block_counts),
            num_computed_tokens=0,
            lora_request=None,
            prefill_token_ids=decode_token_ids,
        ),
    ]
    decode_prefill_output.num_scheduled_tokens = {
        decode_req_id: decode_prompt_len,
    }
    decode_prefill_output.total_num_scheduled_tokens = decode_prompt_len

    decode_new_blocks = blocks.alloc_counts(decode_block_deltas)
    cached_decode_req = CachedRequestData.make_empty()
    cached_decode_req.req_ids = [decode_req_id]
    cached_decode_req.num_computed_tokens = [decode_prompt_len]
    cached_decode_req.num_output_tokens = [1]
    cached_decode_req.new_block_ids = [
        decode_new_blocks if any(decode_block_deltas) else None
    ]

    mixed_output = SchedulerOutput.make_empty()
    mixed_output.num_spec_tokens_to_schedule = model_runner.num_speculative_steps
    mixed_output.scheduled_cached_reqs = cached_decode_req
    mixed_output.scheduled_new_reqs = [
        NewRequestData(
            req_id=prefill_req_id,
            prompt_token_ids=prefill_token_ids,
            mm_features=[],
            sampling_params=sampling_params,
            pooling_params=None,
            block_ids=blocks.alloc_counts(prefill_block_counts),
            num_computed_tokens=0,
            lora_request=None,
            prefill_token_ids=prefill_token_ids,
        ),
    ]
    mixed_output.num_scheduled_tokens = {
        decode_req_id: decode_scheduled_tokens,
        prefill_req_id: prefill_len,
    }
    mixed_output.total_num_scheduled_tokens = num_tokens

    cleanup_output = SchedulerOutput.make_empty()
    cleanup_output.finished_req_ids = {decode_req_id, prefill_req_id}

    context = mixed_step_context or nullcontext()
    model_runner.kv_connector.set_disabled(True)
    try:
        worker_execute_model(decode_prefill_output)
        worker_sample_tokens(None)
        with context:
            worker_execute_model(mixed_output)
            worker_sample_tokens(None)
        worker_execute_model(cleanup_output)
    finally:
        model_runner.kv_connector.set_disabled(False)
    _zero_warmup_blocks(model_runner, blocks.next_pool_block)
    return True


def _zero_warmup_blocks(model_runner: GPUModelRunner, num_used_blocks: int) -> None:
    """Re-zero every block a warmup wrote (ids 0 .. num_used_blocks - 1).

    Warmup steps bypass the scheduler, so the blocks they fill are never
    recorded for zeroing. Normal allocations zero them on reuse, but a block
    an async KV load fills skips that zeroing, and the pages the load does not
    write would keep the dummy KV and Mamba states (BF16 bytes that read as
    FP8 NaN in attention).
    """
    if model_runner.kv_block_zeroer is not None and num_used_blocks > 0:
        model_runner.kv_block_zeroer.zero_block_ids(list(range(num_used_blocks)))


@torch.inference_mode()
def warmup_kernels(
    model_runner: GPUModelRunner,
    worker_execute_model: Callable[[SchedulerOutput], Any],
    worker_sample_tokens: Callable[[GrammarOutput | None], Any],
) -> None:
    """Run scheduler-realistic prefill and decode steps to JIT compile kernels.

    We must call the provided worker's execute_model for pipeline parallel
    coordination.
    """
    # Adaptive costs are calibrated during capture, after this warmup. Exercise
    # fixed draft counts here, then restore the manager for capture and serving.
    adaptive_verification = model_runner.adaptive_verification
    model_runner.adaptive_verification = None
    rejection_sampler = model_runner.rejection_sampler
    adaptive_sampling = (
        rejection_sampler is not None and rejection_sampler.enable_adaptive_verification
    )
    if adaptive_sampling:
        assert rejection_sampler is not None
        rejection_sampler.enable_adaptive_verification = False
    try:
        _warmup_kernels(model_runner, worker_execute_model, worker_sample_tokens)
        # All PP ranks must run the same forwards to match sends and receives.
        if model_runner.speculative_config is not None:
            # Greedy batches skip logits processing, so rejection kernels
            # also see model-dtype logits instead of FP32.
            _warmup_kernels(
                model_runner, worker_execute_model, worker_sample_tokens, greedy=True
            )
    finally:
        model_runner.adaptive_verification = adaptive_verification
        if adaptive_sampling:
            assert rejection_sampler is not None
            rejection_sampler.enable_adaptive_verification = True
    if adaptive_verification is not None:
        adaptive_verification.warmup_budget_kernel()
    acceptance_estimator = getattr(
        model_runner.speculator, "acceptance_estimator", None
    )
    if acceptance_estimator is not None:
        acceptance_estimator.warmup()


def _warmup_logprobs_kernels(model_runner: GPUModelRunner) -> None:
    """Compile the logprobs kernels for every token-id count a request can ask for.

    The steps above score only the counts of ``SamplingParams.for_sampler_warmup``
    (5 sample and 1 prompt logprob, no ``logprob_token_ids``), so any other
    ``logprobs`` / ``prompt_logprobs`` / ``logprob_token_ids`` request would
    compile in the serving loop. Sample logprobs score 1 + up to
    max(max_logprobs, MAX_LOGPROB_TOKEN_IDS) token ids per row, prompt logprobs
    1 + up to max_logprobs, and prompt_logprob_token_ids up to max_logprobs
    (from any row offset). Rows come from ``compute_logits`` (prompt logprobs,
    raw modes), the batch-sharded sampler's all-to-all view, and the FP32
    processed logits (processed modes), each also one row in when a row is not a
    multiple of 16 bytes (the rejection sampler scores row chunks). The logits
    modes gather instead of running the log-softmax kernel.
    """
    sampler = model_runner.sampler
    if (
        current_platform.is_cpu()
        or model_runner.is_pooling_model
        or not model_runner.is_last_pp_rank
        or sampler is None
    ):
        return
    model_config = model_runner.model_config
    device = model_runner.device
    vocab_size = model_runner.vocab_size
    max_logprobs = model_config.max_logprobs
    if max_logprobs < 0:
        max_logprobs = vocab_size
    max_num_cols = max(max_logprobs, MAX_LOGPROB_TOKEN_IDS)
    # The sampler's sampled ids and request mapping are int64.
    int64_ids = torch.zeros(1, dtype=torch.int64, device=device)
    warmup_logprob_token_ids(
        sampler.logprob_token_ids_state, int64_ids, int64_ids, max_num_cols
    )
    if sampler.logprobs_mode in ("raw_logits", "processed_logits"):
        return
    hidden_states = torch.zeros(
        1, model_config.get_hidden_size(), dtype=model_config.dtype, device=device
    )
    # Every TP rank computes the logits together (gathered across the group).
    logits = model_runner.model.compute_logits(hidden_states)
    if logits is None:
        return
    layouts = [logits]
    if model_runner.batch_sharder is not None:
        shard = model_runner.model.compute_logits_local(hidden_states)
        tp_size = model_runner.parallel_config.tensor_parallel_size
        layouts.append(shard.new_empty(1, tp_size * shard.shape[-1])[:, :vocab_size])
    if sampler.logprobs_mode in PROCESSED_LOGPROBS_MODES:
        layouts.append(torch.empty_like(logits, dtype=torch.float32))
    for layout in list(layouts):
        if layout.stride(0) * layout.element_size() % 16:
            rows = layout.new_empty((2, layout.stride(0)))
            layouts.append(rows[1:, : layout.shape[1]])
    for layout in layouts:
        # Only prompt_logprob_token_ids (scored on compute_logits rows) slices.
        unaligned = max_logprobs if layout is logits else 0
        warmup_token_logprobs(layout, 1 + max_num_cols, max_unaligned_count=unaligned)
    del logits, layouts


def _warmup_kernels(
    model_runner: GPUModelRunner,
    worker_execute_model: Callable[[SchedulerOutput], Any],
    worker_sample_tokens: Callable[[GrammarOutput | None], Any],
    greedy: bool = False,
) -> None:
    if model_runner.vllm_config.is_mm_encoder_only:
        return

    num_spec_steps = model_runner.num_speculative_steps
    decode_query_len = model_runner.decode_query_len
    # Use decode_query_len + 1 tokens so the prefill batch's per-request query
    # length exceeds decode_query_len, preventing it from being misclassified as
    # a uniform decode batch.
    prompt_len = decode_query_len + 1
    prompt_token_ids = list(range(prompt_len))
    # Upper bound on the decode steps built in `decode_steps` below.
    num_decode_steps = 1
    if not model_runner.is_pooling_model:
        num_decode_steps = 5 if num_spec_steps > 0 else 3
    # Size the block allocation for the worst case: every request advancing
    # decode_query_len tokens on every decode step.
    decode_len = prompt_len + num_decode_steps * decode_query_len

    kv_cache_groups = model_runner.kv_cache_config.kv_cache_groups

    # Encoder-decoder models: give each warmup request a dummy encoder input so
    # cross-attention warms up over a realistic, non-empty key sequence.
    # The dummy mm_feature is registered in the encoder cache and only its encoder
    # length is read (not the inputs themselves); the encoder itself is not scheduled.
    max_encoder_len = getattr(model_runner.model_state, "max_encoder_len", 0)
    warmup_mm_features: list[MultiModalFeatureSpec] = []
    if model_runner.is_encoder_decoder and max_encoder_len:
        warmup_mm_features = [
            MultiModalFeatureSpec(
                data=None,
                modality="",
                identifier="_warmup_encoder",
                mm_position=PlaceholderRange(offset=0, length=max_encoder_len),
            )
        ]

    # Compute per-request block counts for each KV cache group.
    blocks = _WarmupBlocks(model_runner)
    block_count = _warmup_block_counter(model_runner)
    kv_cache_specs = blocks.specs
    prefill_block_counts = blocks.counts(prompt_len)
    decode_block_counts = blocks.counts(decode_len)
    max_blocks_per_req = blocks.grown_pool_blocks(
        prefill_block_counts, decode_block_counts
    )

    num_reqs = min(
        model_runner.scheduler_config.max_num_seqs,
        model_runner.scheduler_config.max_num_batched_tokens
        // max(prompt_len, decode_query_len),
    )
    block_tables = getattr(model_runner, "block_tables", None)
    null_blocks = block_tables is not None and (
        block_tables.redirect_writes_to_null_block
    )
    if max_blocks_per_req > 0 and not null_blocks:
        # Reserve block 0 (null block) and ensure we have enough blocks.
        # Encoder-only models allocate no KV blocks, so this cap doesn't apply.
        num_reqs = min(
            num_reqs,
            max(1, (model_runner.kv_cache_config.num_blocks - 1) // max_blocks_per_req),
        )

    req_ids = [f"_warmup_{i}_" for i in range(num_reqs)]

    # SamplingParams exercising all sampling features.
    if model_runner.is_pooling_model:
        sampling_params = None
        pooling_task = model_runner.model_config.get_pooling_task(
            model_runner.get_supported_tasks()
        )
        pooling_params = PoolingParams(task=pooling_task)
        pooling_params.verify(model_runner.model_config)
    else:
        sampling_params = (
            SamplingParams(temperature=0)
            if greedy
            else SamplingParams.for_sampler_warmup()
        )
        pooling_params = None

    # Distinct block IDs per request per group (``_WarmupBlocks.alloc_counts``;
    # pool block 0 is the null block).
    # The KV-block zeroing kernel is driven by the scheduler's
    # new_block_ids_to_zero, so none of the steps below reach it.
    if model_runner.kv_block_zeroer is not None:
        model_runner.kv_block_zeroer.warmup(model_runner.kv_cache_config.num_blocks)

    # Step 1: Prefill all requests with 1 + decode_query_len prompt tokens each.
    new_reqs = [
        NewRequestData.from_request(
            Request(
                req_ids[i],
                prompt_token_ids,
                sampling_params,
                pooling_params,
                mm_features=warmup_mm_features,
            ),
            block_ids=blocks.alloc_counts(prefill_block_counts),
            prefill_token_ids=prompt_token_ids,
        )
        for i in range(num_reqs)
    ]

    prefill_output = SchedulerOutput.make_empty()
    prefill_output.num_spec_tokens_to_schedule = num_spec_steps
    prefill_output.scheduled_new_reqs = new_reqs
    prefill_output.num_scheduled_tokens = {rid: prompt_len for rid in req_ids}
    prefill_output.total_num_scheduled_tokens = prompt_len * num_reqs

    # Disable KV connector for warmup run.
    model_runner.kv_connector.set_disabled(True)
    worker_execute_model(prefill_output)

    if not model_runner.is_pooling_model:
        # Warm up sampler and perform a decode step for non-pooling models.

        grammar_output = None
        if model_runner.is_last_pp_rank:
            # Build a GrammarOutput to exercise the structured output bitmask
            # kernel during the prefill step.
            vocab_size = model_runner.model_config.get_vocab_size()
            bitmask_width = (vocab_size + 31) // 32
            grammar_bitmask = np.full(
                (len(req_ids) * model_runner.decode_query_len, bitmask_width),
                fill_value=-1,
                dtype=np.int32,
            )
            grammar_output = GrammarOutput(
                structured_output_request_ids=req_ids, grammar_bitmask=grammar_bitmask
            )

        worker_sample_tokens(grammar_output)

        # Per-request state carried across the decode steps.
        req_computed = [prompt_len] * num_reqs
        req_blocks = [list(prefill_block_counts) for _ in range(num_reqs)]

        def _run_decode_step(indices: list[int], spec_flags: list[bool]) -> None:
            """Decode `indices`, spec-decoding the ones flagged in `spec_flags`."""
            cached_req_data = CachedRequestData.make_empty()
            cached_req_data.req_ids = [req_ids[i] for i in indices]
            cached_req_data.num_computed_tokens = [req_computed[i] for i in indices]
            cached_req_data.num_output_tokens = [1] * len(indices)
            cached_req_data.new_block_ids = []

            step_num_scheduled_tokens: dict[str, int] = {}
            step_spec_tokens: dict[str, list[int]] = {}
            for i, use_spec in zip(indices, spec_flags):
                num_tokens = decode_query_len if use_spec else 1
                after = req_computed[i] + num_tokens
                deltas = [
                    block_count(after, spec) - held
                    for spec, held in zip(kv_cache_specs, req_blocks[i])
                ]
                cached_req_data.new_block_ids.append(
                    blocks.alloc_counts(deltas) if any(deltas) else None
                )
                req_blocks[i] = [
                    held + delta for held, delta in zip(req_blocks[i], deltas)
                ]
                step_num_scheduled_tokens[req_ids[i]] = num_tokens
                if use_spec:
                    step_spec_tokens[req_ids[i]] = [0] * num_spec_steps

            decode_output = SchedulerOutput.make_empty()
            decode_output.num_spec_tokens_to_schedule = num_spec_steps
            decode_output.scheduled_cached_reqs = cached_req_data
            decode_output.num_scheduled_tokens = step_num_scheduled_tokens
            decode_output.scheduled_spec_decode_tokens = step_spec_tokens
            decode_output.total_num_scheduled_tokens = sum(
                step_num_scheduled_tokens.values()
            )

            worker_execute_model(decode_output)
            worker_sample_tokens(None)

            for i, use_spec in zip(indices, spec_flags):
                req_computed[i] += decode_query_len if use_spec else 1

        all_indices = list(range(num_reqs))
        use_spec_decode = num_spec_steps > 0

        # Decode steps to warm, as (request indices, per-request spec flag).
        # Under spec decoding the scheduler drops requests the drafter proposed
        # nothing for, so warm each batch shape with and without draft tokens.
        decode_steps: list[tuple[list[int], list[bool]]] = [
            (all_indices, [use_spec_decode] * num_reqs),
        ]
        if num_reqs >= 2:
            # Mixed spec / non-spec: GDN and KDA reclassify the non-spec decode
            # as a prefill and split the batch into spec/non-spec token indices.
            decode_steps.append(([0, 1], [use_spec_decode, False]))
            if use_spec_decode:
                # Exercise the model paths that split a batch by whether each
                # request received draft tokens.
                decode_steps.append(([0, 1], [False, False]))
        if num_reqs > 1:
            decode_steps.append(([0], [use_spec_decode]))
            if use_spec_decode:
                decode_steps.append(([0], [False]))
        elif use_spec_decode:
            decode_steps.append(([0], [False]))

        for step_indices, step_spec_flags in decode_steps:
            _run_decode_step(step_indices, step_spec_flags)

        if not greedy:
            _warmup_logprobs_kernels(model_runner)

    # Clean up - process finish_req_ids.
    cleanup_output = SchedulerOutput.make_empty()
    cleanup_output.finished_req_ids = set(req_ids)
    worker_execute_model(cleanup_output)
    model_runner.kv_connector.set_disabled(False)
    # Block 0 (the null block) included.
    _zero_warmup_blocks(model_runner, blocks.next_pool_block)
    torch.accelerator.synchronize()


# ---------------------------------------------------------------------------
# Serving-shape warmup and its startup verification
# ---------------------------------------------------------------------------
#
# The steps above warm short prompts and decode batches. A serving step also
# mixes full-budget prompt chunks with spec-decode rows: a chunk long enough
# for sparse-attention top-k, a continuation chunk over a long context, a
# prefix-cache hit (or P/D KV load) whose few new tokens attend a long context,
# per-request sampling mixes, and, under DP, ranks that pad to each other's
# token counts. Those steps take paths (DCP collectives beyond the direct
# workspace, top-k candidate packing, RecoverSSM commit with request indices,
# FP32-logit sampling, ...) that first JIT-compile in the serving loop and
# stall it for a minute. These steps run them at startup through the real
# step path, and a last step with different shapes verifies that nothing
# compiles any more.


def _serving_sampling_params() -> tuple[SamplingParams, ...]:
    """Per-request sampling mixes of a serving step (round-robin)."""
    common = dict(max_tokens=64, watermarking=False)
    return (
        # Greedy: model-dtype logits, no fused sampler for the batch.
        SamplingParams(temperature=0.0, **common),
        # Logits processing: FP32 logits into the Gumbel sampler.
        SamplingParams(
            temperature=0.8,
            repetition_penalty=1.1,
            frequency_penalty=0.1,
            **common,
        ),
        # Temperature only: model-dtype logits, temperature in the sampler.
        SamplingParams(temperature=0.8, **common),
        # Top-k / top-p.
        SamplingParams(temperature=0.8, top_p=0.95, top_k=50, **common),
    )


def _serving_step_shapes(
    model_runner: GPUModelRunner, variant: int, max_decodes: int | None = None
) -> tuple[int, int, int, int, int] | None:
    """(decode requests, first chunk, second chunk, cache-hit context, cache-hit
    new tokens) of serving-shape step set ``variant``.

    Variant 0 fills the token budget with the same shapes on every rank;
    later variants shrink the decode batch and chunks by rank-dependent odd
    amounts, so DP ranks pad to each other's token counts and alignments,
    batch-size buckets and row counts differ from variant 0. ``max_decodes``
    caps the decode batch before that shrink (``_serving_decode_cap``: the
    largest batch whose KV blocks the server has).
    """
    budget = model_runner.scheduler_config.max_num_batched_tokens
    query_len = model_runner.decode_query_len
    max_model_len = model_runner.max_model_len
    lookahead = model_runner.vllm_config.num_lookahead_tokens
    dp_rank = model_runner.vllm_config.parallel_config.data_parallel_rank
    shift = 0 if variant == 0 else variant * 7 + (dp_rank % 4) * 3 + 1

    # Two requests (the long prompt and the cache hit) besides the decodes.
    num_decodes = min(model_runner.max_num_reqs - 2, (budget // 2) // query_len)
    if max_decodes is not None:
        num_decodes = min(num_decodes, max_decodes)
    num_decodes -= min(num_decodes - 1, shift // 2) if num_decodes > 1 else 0
    if num_decodes < 1:
        return None
    hit_new = 1 if variant == 0 else query_len + 1 + shift % 5
    free = budget - num_decodes * query_len - hit_new
    longest = max_model_len - lookahead - 1
    first = min(free, longest // 2) - shift
    second = min(free, longest - first) - 2 * shift
    hit_context = min(first, longest - hit_new)
    if min(first, second, hit_context) < 1:
        return None
    return num_decodes, first, second, hit_context, hit_new


def _serving_ladder_shapes(
    model_runner: GPUModelRunner,
) -> list[tuple[int, int, int, int, int, int]]:
    """Step sets for the token counts between decode batches and the
    budget-filling sets: serving steps of 2^k + 1 and 3 * 2^(k - 1) tokens
    (65 up to the token budget), three counts per set: a single-sequence
    prefill of the decode request's prompt (step 1), one spec-decode row + a
    prompt chunk (step 2), one spec-decode row + the next chunk + a one-token
    cache hit (step 3). Kernels that pick a tile, a split or a specialization
    by token or sequence count compile here, not in the first small mixed
    serving step (what P/D decode and prefill servers mostly run). Shapes are
    the same on every rank; see ``run_serving_shape_warmup`` for the tuple.
    """
    budget = model_runner.scheduler_config.max_num_batched_tokens
    query_len = model_runner.decode_query_len
    longest = (
        model_runner.max_model_len - model_runner.vllm_config.num_lookahead_tokens - 1
    )
    counts = sorted(
        {
            c
            for k in range(6, budget.bit_length())
            for c in (2**k + 1, 3 << (k - 1))
            if query_len + 2 <= c <= budget
        }
    )
    sets = []
    for i in range(0, len(counts), 3):
        t1, t2, t3 = (counts[i : i + 3] + counts[-1:] * 2)[:3]
        first, second = t2 - query_len, t3 - query_len - 1
        if min(first, second) >= 1 and first + second <= longest:
            sets.append((1, first, second, first, 1, t1))
    return sets


def _serving_step_blocks(
    model_runner: GPUModelRunner, shapes: tuple[int, int, int, int, int]
) -> tuple[int, int]:
    """(KV pool blocks a serving-shape step set holds, pool blocks per decode
    request), as ``run_serving_shape_warmup`` allocates them."""
    num_decodes, first, second, hit_context, hit_new = shapes
    query_len = model_runner.decode_query_len
    decode_len = query_len + 1 + 2 * query_len
    blocks = _WarmupBlocks(model_runner)
    per_decode = blocks.pool_blocks(decode_len)
    required = (
        num_decodes * per_decode
        + blocks.pool_blocks(first + second)
        + blocks.pool_blocks(hit_context + hit_new)
    )
    return required, per_decode


def _serving_decode_cap(model_runner: GPUModelRunner) -> int | None:
    """The largest decode batch whose serving-shape steps fit the KV blocks
    the server has (block 0 is the null block), or None when the full batch
    fits. A decode server's KV pool can hold fewer per-request blocks (state
    groups, speculative blocks) than ``max_num_seqs`` warmup requests need:
    the steps then run a smaller batch rather than not at all, so the
    verification step always runs. Same on every rank (``num_blocks`` is)."""
    available = model_runner.kv_cache_config.num_blocks - 1
    cap: int | None = None
    while True:
        shapes = _serving_step_shapes(model_runner, 0, cap)
        if shapes is None:
            return cap
        required, per_decode = _serving_step_blocks(model_runner, shapes)
        if required <= available:
            if cap is not None:
                logger.info(
                    "Serving-shape warmup: %d decode requests fit the %d KV "
                    "blocks (%d needed, %d per decode request).",
                    shapes[0],
                    available,
                    required,
                    per_decode,
                )
            return cap
        num_decodes = shapes[0]
        cap = num_decodes - max(1, cdiv(required - available, per_decode))
        if cap < 1:
            raise RuntimeError(
                "Serving-shape warmup cannot fit one decode request with its "
                f"prompt chunks in the {available} KV blocks the server has "
                f"({required} needed with {num_decodes} decode requests); the "
                "warmup verification would not run."
            )


def run_serving_shape_warmup(
    model_runner: GPUModelRunner,
    worker_execute_model: Callable[[SchedulerOutput], Any],
    worker_sample_tokens: Callable[[GrammarOutput | None], Any],
    variant: int,
    ladder: tuple[int, int, int, int, int, int] | None = None,
) -> bool:
    """Run one serving-shape step set (see ``_serving_step_shapes``).

    Steps: (1) prefill the decode requests; (2) spec-decode rows + the first
    chunk of a long prompt; (3) spec-decode rows + the prompt's last chunk
    over the first one + a new request with a long cached context and a few
    new tokens. Returns False when the token budget or max_model_len leaves no
    room for the shapes; the decode batch is sized to the KV blocks.
    ``ladder``: explicit (decode requests, first chunk, second chunk, cache-hit
    context, cache-hit new tokens, decode prompt length) of a
    ``_serving_ladder_shapes`` set instead of the variant's shapes.
    """
    if (
        model_runner.is_pooling_model
        or model_runner.vllm_config.is_mm_encoder_only
        or model_runner.is_encoder_decoder
    ):
        return False
    query_len = model_runner.decode_query_len
    prompt_len = query_len + 1
    if ladder is not None:
        num_decodes, first, second, hit_context, hit_new, prompt_len = ladder
        shapes = ladder[:5]
    else:
        shapes = _serving_step_shapes(
            model_runner, variant, _serving_decode_cap(model_runner)
        )
    if shapes is None:
        logger.warning(
            "Skipping serving-shape warmup %d: the token budget (%d) or "
            "max_model_len (%d) leaves no room for its steps.",
            variant,
            model_runner.scheduler_config.max_num_batched_tokens,
            model_runner.max_model_len,
        )
        return False
    num_decodes, first, second, hit_context, hit_new = shapes
    num_spec = model_runner.num_speculative_steps
    decode_len = prompt_len + 2 * query_len
    long_len = first + second
    hit_len = hit_context + hit_new

    blocks = _WarmupBlocks(model_runner)
    required = (
        num_decodes * blocks.pool_blocks(decode_len)
        + blocks.pool_blocks(long_len)
        + blocks.pool_blocks(hit_len)
    )
    if model_runner.kv_cache_config.num_blocks <= required:
        # _serving_decode_cap sized the batch to the pool and the ladder sets
        # run one decode request: never skip.
        raise RuntimeError(
            f"Serving-shape warmup {variant}: {required} KV blocks needed, "
            f"{model_runner.kv_cache_config.num_blocks} available after sizing "
            "the decode batch to the pool."
        )
    alloc = blocks.alloc

    params = _serving_sampling_params()
    vocab = model_runner.model_config.get_vocab_size()

    def token_ids(n: int, seed: int) -> list[int]:
        return [(seed * 7919 + i * 31) % (vocab - 1) + 1 for i in range(n)]

    prefix = f"_serving_warmup_{variant}"
    decode_ids = [f"{prefix}_d{i}_" for i in range(num_decodes)]
    long_id, hit_id = f"{prefix}_long_", f"{prefix}_hit_"
    long_tokens = token_ids(long_len, 1)
    hit_tokens = token_ids(hit_len, 2)

    def new_req(
        req_id: str,
        tokens: list[int],
        sampling_params: SamplingParams,
        num_tokens_for_blocks: int,
        num_computed: int = 0,
    ) -> NewRequestData:
        return NewRequestData(
            req_id=req_id,
            prompt_token_ids=tokens,
            mm_features=[],
            sampling_params=sampling_params,
            pooling_params=None,
            block_ids=alloc(num_tokens_for_blocks),
            num_computed_tokens=num_computed,
            lora_request=None,
            prefill_token_ids=tokens,
        )

    def output(
        new_reqs: list[NewRequestData],
        cached_ids: list[str],
        cached_computed: list[int],
        cached_outputs: list[int],
        scheduled: dict[str, int],
    ) -> SchedulerOutput:
        out = SchedulerOutput.make_empty()
        out.num_spec_tokens_to_schedule = num_spec
        out.scheduled_new_reqs = new_reqs
        cached = CachedRequestData.make_empty()
        cached.req_ids = cached_ids
        cached.num_computed_tokens = cached_computed
        cached.num_output_tokens = cached_outputs
        cached.new_block_ids = [None] * len(cached_ids)
        out.scheduled_cached_reqs = cached
        out.num_scheduled_tokens = scheduled
        if num_spec > 0:
            out.scheduled_spec_decode_tokens = {
                rid: [0] * num_spec for rid in cached_ids if rid in decode_ids
            }
        out.total_num_scheduled_tokens = sum(scheduled.values())
        return out

    # Variant 1 is homogeneous (every request processes its logits, none is
    # greedy: FP32 logits, unfused sampling); the others mix all four.
    if variant == 1:
        params = (params[1],) * len(params)
    rotation = variant % len(params)
    decode_params = [params[(i + rotation) % len(params)] for i in range(num_decodes)]
    # Step 1: the decode requests' prompts (all blocks reserved up front).
    step1 = output(
        [
            new_req(rid, token_ids(prompt_len, 10 + i), decode_params[i], decode_len)
            for i, rid in enumerate(decode_ids)
        ],
        [],
        [],
        [],
        {rid: prompt_len for rid in decode_ids},
    )
    decode_tokens = query_len if num_spec > 0 else 1
    # Step 2: spec-decode rows + the long prompt's first chunk.
    step2_sched = {rid: decode_tokens for rid in decode_ids}
    step2_sched[long_id] = first
    step2 = output(
        [new_req(long_id, long_tokens, params[(1 + rotation) % len(params)], long_len)],
        list(decode_ids),
        [prompt_len] * num_decodes,
        [1] * num_decodes,
        step2_sched,
    )
    # Step 3: spec-decode rows + the long prompt's last chunk over its first
    # chunk + a cache hit (long context, few new tokens).
    step3_sched = {rid: decode_tokens for rid in decode_ids}
    step3_sched[long_id] = second
    step3_sched[hit_id] = hit_new
    step3 = output(
        [
            new_req(
                hit_id,
                hit_tokens,
                params[(2 + rotation) % len(params)],
                hit_len,
                num_computed=hit_context,
            )
        ],
        [*decode_ids, long_id],
        [prompt_len + decode_tokens] * num_decodes + [first],
        [2] * num_decodes + [0],
        step3_sched,
    )
    cleanup = SchedulerOutput.make_empty()
    cleanup.finished_req_ids = {*decode_ids, long_id, hit_id}

    logger.info(
        "Serving-shape warmup %d: %d spec-decode rows x %d tokens, prompt "
        "chunks %d + %d, cache hit %d + %d tokens.",
        variant,
        num_decodes,
        decode_tokens,
        first,
        second,
        hit_context,
        hit_new,
    )
    model_runner.kv_connector.set_disabled(True)
    try:
        for step in (step1, step2, step3):
            worker_execute_model(step)
            worker_sample_tokens(None)
        worker_execute_model(cleanup)
    finally:
        model_runner.kv_connector.set_disabled(False)
    _zero_warmup_blocks(model_runner, blocks.next_pool_block)
    torch.accelerator.synchronize()
    return True


# Serving-shape step sets run to compile; the next one verifies.
NUM_SERVING_WARMUP_VARIANTS = 2


class JitWarmupGapError(RuntimeError):
    """A kernel compiled in the serving-shape verification step after warmup."""


@torch.inference_mode()
def serving_shape_warmup_kernels(
    model_runner: GPUModelRunner,
    worker_execute_model: Callable[[SchedulerOutput], Any],
    worker_sample_tokens: Callable[[GrammarOutput | None], Any],
) -> None:
    """``serving_shape_warmup`` with fixed draft counts (as ``warmup_kernels``:
    adaptive verification is calibrated during capture, after this)."""
    if not current_platform.is_cuda_alike():
        return
    adaptive_verification = model_runner.adaptive_verification
    model_runner.adaptive_verification = None
    rejection_sampler = model_runner.rejection_sampler
    adaptive_sampling = (
        rejection_sampler is not None and rejection_sampler.enable_adaptive_verification
    )
    if adaptive_sampling:
        assert rejection_sampler is not None
        rejection_sampler.enable_adaptive_verification = False
    try:
        serving_ladder_warmup(model_runner, worker_execute_model, worker_sample_tokens)
        serving_shape_warmup(model_runner, worker_execute_model, worker_sample_tokens)
    finally:
        model_runner.adaptive_verification = adaptive_verification
        if adaptive_sampling:
            assert rejection_sampler is not None
            rejection_sampler.enable_adaptive_verification = True


def serving_ladder_warmup(
    model_runner: GPUModelRunner,
    worker_execute_model: Callable[[SchedulerOutput], Any],
    worker_sample_tokens: Callable[[GrammarOutput | None], Any],
) -> None:
    """Run the ``_serving_ladder_shapes`` step sets (small serving steps)."""
    for i, shapes in enumerate(_serving_ladder_shapes(model_runner)):
        run_serving_shape_warmup(
            model_runner,
            worker_execute_model,
            worker_sample_tokens,
            NUM_SERVING_WARMUP_VARIANTS + 1 + i,
            ladder=shapes,
        )


def serving_shape_warmup(
    model_runner: GPUModelRunner,
    worker_execute_model: Callable[[SchedulerOutput], Any],
    worker_sample_tokens: Callable[[GrammarOutput | None], Any],
) -> None:
    """Compile what serving steps launch, then verify with new shapes.

    The verification step set (rank-dependent shapes that no warmup step
    used) runs with the JIT recorder; any Triton / CuTe DSL / TileLang
    compile in it means a serving step would compile too, which stalls the
    serving loop, so startup fails with the kernel names.
    """
    from vllm.utils import jit_monitor

    for variant in range(NUM_SERVING_WARMUP_VARIANTS):
        run_serving_shape_warmup(
            model_runner, worker_execute_model, worker_sample_tokens, variant
        )
    with jit_monitor.record_jit_events() as events:
        verified = run_serving_shape_warmup(
            model_runner,
            worker_execute_model,
            worker_sample_tokens,
            NUM_SERVING_WARMUP_VARIANTS,
        )
    if events:
        for event in events:
            logger.error("JIT_WARMUP_GAP %s", event)
        raise JitWarmupGapError(
            "JIT_WARMUP_GAP: kernels compiled in the serving-shape verification "
            f"step after warmup ({len(events)}): {sorted(set(events))}. A "
            "serving step would compile them too; extend warmup to cover them."
        )
    if verified:
        logger.info(
            "JIT_WARMUP_VERIFIED: no kernel compiled in the serving-shape "
            "verification step."
        )
