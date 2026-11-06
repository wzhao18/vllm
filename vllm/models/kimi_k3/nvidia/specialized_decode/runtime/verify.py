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

Under sequence parallelism (C512's TP4 = DCP4 decode role, ``runtime.sp_topology``)
synthetic rejection keeps the stock sampler, a declared stock module (``verify`` in
``shapes.DECLARED_STOCK``); otherwise the build either covers TP4 at every row count or
raises ``shapes.Unsupported`` naming the cause: never the silent stock path.
"""

import torch

from vllm.logger import init_logger

from ..cutedsl.verify.argmax import MAX_ROWS, THREADS, VECTOR

logger = init_logger(__name__)

# Tensor-parallel sizes served by the vocabulary-sharded argmax on Rubin (compute
# capability 10.7). GB300 keeps the gathered logits.
FAST_ARGMAX_TP = (4, 8, 16)


def _uncovered(sp: bool, cause: str):
    """The uncovered build: None (the stock sampler) without sequence parallelism;
    under it a raise (``verify`` is declared stock only under synthetic rejection)."""
    if not sp:
        return None
    from ..shapes import Unsupported

    raise Unsupported("verify", None, cause)


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
        covered. ``model``: the KimiLinearForCausalLM (weights loaded). Under
        sequence parallelism: None (declared stock) with synthetic rejection, else
        the argmax or ``shapes.Unsupported``."""
        if vllm_config.speculative_config is None:
            return None
        from . import sp_topology

        sp = sp_topology(vllm_config)
        method = vllm_config.speculative_config.rejection_sample_method
        if sp and method == "synthetic":
            logger.info_once(
                "Kimi K3 SP decode: verify: declared stock (synthetic rejection)"
            )
            return None
        device = torch.device(torch.cuda.current_device())
        if torch.cuda.get_device_capability(device) != (10, 7):
            return _uncovered(sp, "this GPU (Rubin SM107 only)")
        from vllm.distributed import get_tp_group

        group = get_tp_group()
        tp, rank = group.world_size, group.rank_in_group
        lm_head, processor = model.lm_head, model.logits_processor
        shard = getattr(lm_head, "shard_indices", None)
        if (
            tp not in FAST_ARGMAX_TP
            or shard is None
            or shard.num_org_vocab_padding != 0
            or processor.scale != 1.0
            or processor.soft_cap is not None
            or processor.logits_as_input
            or processor.head_dtype not in (None, torch.bfloat16)
            or vllm_config.model_config.dtype != torch.bfloat16
            or shard.num_added_elements != 0
        ):
            return _uncovered(
                sp,
                f"TP{tp} or the LM head / logits processor (a padded, extended or "
                "unsharded vocabulary, a logits scale or soft cap, logits as input, "
                "a non-BF16 head)",
            )
        vocab_rank = shard.org_vocab_end_index - shard.org_vocab_start_index
        if (
            vocab_rank % (THREADS * VECTOR)
            or shard.org_vocab_start_index != rank * vocab_rank
            or processor.org_vocab_size != tp * vocab_rank
        ):
            return _uncovered(
                sp,
                f"the vocabulary shard ({vocab_rank} columns per rank at rank {rank}, "
                f"vocabulary {processor.org_vocab_size}): not {THREADS * VECTOR}-"
                "aligned contiguous rank-order shards",
            )
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
