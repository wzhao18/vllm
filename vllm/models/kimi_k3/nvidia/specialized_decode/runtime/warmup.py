# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compile the spec-decode sampling kernels before the first decode step.

The verification kernels (Triton) otherwise compile at the first decode step that
needs them, which stalls that request's first tokens by about a second (and the
AgentX warmup sends only one-token primers, so the stall lands on a profiled
request). ``warm_sampling`` runs each of them once, at engine init on every rank
together, on scratch tensors of the deployment's dtypes and shapes (the same
specializations as serving), so nothing it writes is real state:

- the stock rejection sampler (``gather_draft_sampled``, ``rejection_sample`` in the
  configured mode with BF16 logits, and with FP32 logits as processed sampling
  parameters produce them, ``apply_temperature``, ``get_num_sampled_and_rejected``)
  and greedy verification (``greedy_verify``);
- the vocabulary-sharded sampled verification (``SampledVerify.run``, one batch per
  segment layout up to the deployment's batch size; its exchanges run on every rank
  together, as at serving).
"""

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def _rates(spec, device):
    from vllm.config import SpeculativeConfig
    from vllm.v1.spec_decode.utils import unconditional_to_conditional_rates

    if spec.rejection_sample_method != "synthetic":
        return None
    rates = spec.synthetic_acceptance_rates
    if rates is None:
        rates = SpeculativeConfig._acceptance_length_to_rates(
            spec.synthetic_acceptance_length, spec.num_speculative_tokens
        )
    return torch.tensor(
        unconditional_to_conditional_rates(rates), dtype=torch.float32, device=device
    )


def _batch(num_reqs, steps, device):
    """A decode batch of ``num_reqs`` requests with ``steps`` drafts each, every
    request on request slot 0 (scratch state), as the runner's InputBatch fields."""
    q = steps + 1
    rows = num_reqs * q
    return dict(
        input_ids=torch.zeros(rows, dtype=torch.int32, device=device),
        positions=torch.arange(rows, dtype=torch.int64, device=device) + 64,
        logits_indices=torch.arange(rows, dtype=torch.int64, device=device),
        expanded_local_pos=torch.arange(q, dtype=torch.int32, device=device).repeat(
            num_reqs
        ),
        expanded_idx_mapping=torch.zeros(rows, dtype=torch.int64, device=device),
        idx_mapping=torch.zeros(num_reqs, dtype=torch.int64, device=device),
        seq_lens=torch.full((num_reqs,), 1 << 20, dtype=torch.int32, device=device),
        cu_num_logits=torch.arange(0, rows + 1, q, dtype=torch.int32, device=device),
    )


def warm_sampling(vllm_config, model) -> None:
    """Every rank of the TP group together, after the sampling paths are built."""
    spec = vllm_config.speculative_config
    if spec is None:
        return
    from vllm.v1.worker.gpu.input_batch import get_num_sampled_and_rejected
    from vllm.v1.worker.gpu.sample.gumbel import apply_temperature
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler import gather_draft_sampled
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
        greedy_verify,
        rejection_sample,
    )

    device = torch.device(torch.cuda.current_device())
    steps = spec.num_speculative_tokens
    vocab = vllm_config.model_config.get_vocab_size()
    dtype = vllm_config.model_config.dtype
    rates = _rates(spec, device)
    block = spec.rejection_sample_method == "block"
    use_fp64 = vllm_config.model_config.use_fp64_gumbel
    draft_logits = None
    if spec.draft_sample_method == "probabilistic":
        draft_logits = torch.zeros((1, steps, vocab), dtype=dtype, device=device)
    temperature = torch.ones(1, dtype=torch.float32, device=device)
    seeds = torch.full((1,), 17, dtype=torch.int64, device=device)
    prefill_len = torch.zeros(1, dtype=torch.int32, device=device)

    # Stock rejection sampler: BF16 logits (no sampling parameters to apply) and FP32
    # (apply_sampling_params' copy), and greedy verification. One request and 16
    # (integer arguments such as the row count specialize on divisibility by 16).
    for num_reqs in (1, 16):
        b = _batch(num_reqs, steps, device)
        rows = num_reqs * (steps + 1)
        for logits_dtype in (dtype, torch.float32):
            logits = torch.zeros((rows, vocab), dtype=logits_dtype, device=device)
            if logits_dtype == torch.float32:
                apply_temperature(logits, b["expanded_idx_mapping"], temperature * 0.5)
            drafts, pos = gather_draft_sampled(
                b["input_ids"],
                b["positions"],
                b["logits_indices"],
                b["expanded_idx_mapping"],
                b["expanded_local_pos"],
                prefill_len,
            )
            _, num_sampled = rejection_sample(
                logits,
                draft_logits,
                drafts,
                b["cu_num_logits"],
                pos,
                b["idx_mapping"],
                b["expanded_idx_mapping"],
                b["expanded_local_pos"],
                temperature,
                seeds,
                steps,
                rates,
                use_fp64=use_fp64,
                use_block_verification=block,
            )
            get_num_sampled_and_rejected(
                num_sampled,
                b["seq_lens"],
                b["cu_num_logits"],
                b["idx_mapping"],
                prefill_len,
            )
        greedy_verify(
            torch.zeros(rows, dtype=torch.int64, device=device),
            b["input_ids"],
            b["positions"],
            b["logits_indices"],
            b["expanded_local_pos"],
            b["cu_num_logits"],
            b["idx_mapping"],
            b["seq_lens"],
            prefill_len,
            seeds,
            steps,
            rates,
        )
        del logits

    # Vocabulary-sharded sampled verification: one batch per segment layout.
    verify = getattr(model, "sampled_verify", None)
    warmed = []
    if verify is not None:
        from .sampled_verify import segments

        max_reqs = vllm_config.scheduler_config.max_num_seqs
        layouts = {}
        for num_reqs in range(1, max_reqs + 1):
            rows = num_reqs * (steps + 1)
            if not verify.takes(rows):
                break
            key = (segments(rows), segments(rows if block else num_reqs))
            layouts.setdefault(key, num_reqs)
        for num_reqs in sorted(layouts.values()):
            b = _batch(num_reqs, steps, device)
            rows = num_reqs * (steps + 1)
            logits = torch.zeros((rows, verify.vocab_rank), dtype=dtype, device=device)
            verify.run(
                logits,
                steps,
                input_ids=b["input_ids"],
                positions=b["positions"],
                logits_indices=b["logits_indices"],
                expanded_local_pos=b["expanded_local_pos"],
                idx_mapping=b["idx_mapping"],
                seq_lens=b["seq_lens"],
                prefill_len=prefill_len,
                temperature=temperature,
                seeds=seeds,
                draft_logits=draft_logits,
                synthetic_rates=rates,
                block_verification=block,
            )
            warmed.append(num_reqs)
    torch.accelerator.synchronize(device)
    logger.info_once(
        "Kimi K3: compiled the spec-decode sampling kernels (%s rejection, %s "
        "drafts; sharded verification for batches %s)",
        spec.rejection_sample_method,
        spec.draft_sample_method,
        tuple(warmed),
    )
