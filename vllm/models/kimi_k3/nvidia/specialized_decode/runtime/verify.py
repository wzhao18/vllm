# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Greedy spec-decode verification for the K3 target, with the specialized decode on
Rubin-class devices: ``get_top_tokens`` (the target argmax of each logits row, which is
all that greedy verification reads; ``RejectionSampler.verify_greedy``) as the stock
LM-head shard GEMM plus one vocabulary-sharded argmax kernel (cutedsl.verify.argmax),
instead of a full-vocabulary logits all-gather and the full-vocabulary rejection
sampler kernels. Up to 8 rows (B1) take the fused LM-head argmax instead, and 10-64
rows the shard GEMM with evict-first weight loads (cutedsl.verify.lmhead; FP32
accumulation orders stated there). Above 64 rows the stock shard GEMM feeds the argmax
kernel in launches of ``argmax.MAX_ROWS`` rows: one per verify step.
"""

import torch

from vllm.logger import init_logger

from ..cutedsl.verify.argmax import MAX_ROWS, THREADS, VECTOR
from ..shapes import config_target, log_coverage, refuse

logger = init_logger(__name__)

# Tensor-parallel sizes served by the vocabulary-sharded argmax on Rubin (compute
# capability 10.7). GB300 keeps the gathered logits.
FAST_ARGMAX_TP = (4, 8, 16)


class VerifyArgmax:
    """Per-rank prepared argmax of the target's logits rows; built with the
    specialized decode (after the weights load, before CUDA graph capture)."""

    def __init__(self, prepared, fused=None, wide=None, logits=None):
        self.prepared = prepared
        self.fused = fused
        self.wide = wide
        # Up to 8 rows: the fused LM head's GEMV with BF16 logits (LmHeadLogits: the
        # same projection as ``fused``), for sampled verification.
        self.logits = logits

    @classmethod
    def build(cls, vllm_config, model):
        """None unless this deployment speculates and this device and TP size are
        covered. ``model``: the KimiLinearForCausalLM (weights loaded)."""
        if vllm_config.speculative_config is None:
            return None
        device = torch.device(torch.cuda.current_device())
        if torch.cuda.get_device_capability(device) != (10, 7):
            return None
        from vllm.distributed import get_tp_group

        group = get_tp_group()
        tp, rank = group.world_size, group.rank_in_group
        lm_head, processor = model.lm_head, model.logits_processor
        shard = getattr(lm_head, "shard_indices", None)
        # NQ-NOFALLBACK: on Rubin with speculation, a deployment these kernels do not
        # cover would verify every step on the stock full-vocabulary sampler: on the
        # target topology refuse it at load, naming the cause.
        checks = (
            (tp not in FAST_ARGMAX_TP, f"TP{tp}"),
            (shard is None, "an LM head without vocabulary shards"),
            (shard is not None and shard.num_org_vocab_padding != 0, "vocab padding"),
            (processor.scale != 1.0, "a logit scale"),
            (processor.soft_cap is not None, "a logit soft cap"),
            (processor.logits_as_input, "logits as input"),
            (processor.head_dtype not in (None, torch.bfloat16), "a non-BF16 head"),
            (vllm_config.model_config.dtype != torch.bfloat16, "a non-BF16 model"),
            (shard is not None and shard.num_added_elements != 0, "added vocabulary"),
        )
        # NQ-NOFALLBACK: refused on the target topology only; elsewhere the stock
        # sampler keeps verifying, as before, logged.
        strict = config_target(vllm_config, tp)

        def decline(cause):
            if strict:
                refuse("target verification (VerifyArgmax)", cause)
            logger.info_once(
                "Kimi K3 verification: stock full-vocabulary sampler (%s)", cause
            )

        reason = next((why for failed, why in checks if failed), None)
        if reason is not None:
            decline(reason)
            return None
        vocab_rank = shard.org_vocab_end_index - shard.org_vocab_start_index
        if (
            vocab_rank % (THREADS * VECTOR)
            or shard.org_vocab_start_index != rank * vocab_rank
            or processor.org_vocab_size != tp * vocab_rank
        ):
            decline(
                f"vocabulary shard of {vocab_rank} columns at rank {rank} is not "
                f"{THREADS * VECTOR}-aligned and rank-contiguous"
            )
            return None
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            UnquantizedEmbeddingMethod,
        )

        from ..cutedsl.verify import lmhead
        from ..cutedsl.verify.argmax import PreparedArgmax

        prepared = PreparedArgmax(tp, rank, group.device_group, vocab_rank, device)
        prepared.prepare()
        fused = wide = logits = None
        weight = lm_head.weight[:vocab_rank]
        if (
            isinstance(lm_head.quant_method, UnquantizedEmbeddingMethod)
            and getattr(lm_head, "bias", None) is None
            and lmhead.supports(weight)
        ):
            fused = lmhead.PreparedLmHeadArgmax(tp, rank, prepared.box, weight, device)
            wide = lmhead.PreparedWideLmHead(weight, device)
            logits = lmhead.PreparedLmHeadLogits(weight, device)
        logger.info_once(
            "Kimi K3: vocabulary-sharded greedy verification (launches of up to %d "
            "rows, fused LM head up to %d, evict-first LM head GEMM above)",
            MAX_ROWS,
            lmhead.TOKENS if fused is not None else 0,
        )
        # NQ-NOFALLBACK: the logits shard GEMM's stock rows, with their reason.
        log_coverage(
            rank,
            "verify",
            f"LM-head shard: 1-{lmhead.TOKENS} rows fused GEMV, "
            f"{lmhead.WIDE_ROWS[0]}-{lmhead.WIDE_ROWS[-1]} rows evict-first GEMM, "
            f"above {lmhead.WIDE_ROWS[-1]} rows stock cuBLAS shard GEMM (bitwise "
            "compute_logits; the evict-first GEMM is not tuned there: NQ-TUNE); "
            "argmax / sampled verification vocabulary-sharded at every row"
            if fused is not None
            else "LM-head shard: stock cuBLAS shard GEMM at every row (the LM-head "
            "weight is not the plain BF16 shard the fused kernels read)",
        )
        return cls(prepared, fused, wide, logits)

    def shard_logits(self, processor, lm_head, hidden_states):
        """This rank's vocabulary shard of the logits rows [rows, vocab_rank] BF16, as
        the argmax path computes them: up to 8 rows the fused LM head's GEMV (its FP32
        accumulation order, as greedy verification's argmax), 10-64 rows the evict-first
        shard GEMM, else the stock LM-head GEMM without its all-gather (bitwise
        compute_logits')."""
        if self.logits is not None and self.logits.takes(hidden_states):
            return self.logits.run(hidden_states)
        if self.wide is not None and self.wide.takes(hidden_states):
            return self.wide.run(hidden_states)
        return processor._get_logits(hidden_states, lm_head, None, skip_gather=True)

    def top_tokens(self, processor, lm_head, hidden_states):
        """The global argmax [rows] int64 of each row's logits, or None (nothing
        launched) when this call is not covered."""
        rows = hidden_states.shape[0]
        if rows > MAX_ROWS:
            return self.top_tokens_wide(processor, lm_head, hidden_states)
        if rows <= 0:
            return None
        if self.fused is not None and self.fused.takes(hidden_states):
            return self.fused.run(hidden_states)
        if self.wide is not None and self.wide.takes(hidden_states):
            # 10-64 rows: the shard GEMM with evict-first weight loads.
            local = self.wide.run(hidden_states)
        else:
            # The stock LM-head GEMM without its all-gather: this rank's vocabulary
            # shard of the logits, bitwise as compute_logits computes it.
            local = processor._get_logits(
                hidden_states, lm_head, None, skip_gather=True
            )
        return self.prepared.run(local.contiguous())

    def top_tokens_wide(self, processor, lm_head, hidden_states):
        """More rows than one argmax launch takes: this rank's logits shard as
        ``compute_logits`` computes it, then the argmax in launches of ``MAX_ROWS`` rows
        (each launch advances its rows' mailbox parities, as consecutive steps do)."""
        logits = processor._get_logits(hidden_states, lm_head, None, skip_gather=True)
        logits = logits.contiguous()
        rows = logits.shape[0]
        out = torch.empty(rows, dtype=torch.int64, device=logits.device)
        for start in range(0, rows, MAX_ROWS):
            end = min(start + MAX_ROWS, rows)
            self.prepared.run(logits[start:end], out[start:end])
        return out
