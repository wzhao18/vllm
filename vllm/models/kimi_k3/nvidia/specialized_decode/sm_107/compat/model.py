# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-level predicates of the source branch's ``nvidia/model.py`` that the
specialized decode reads and the base model does not define."""

from vllm.config import VllmConfig

from . import envs


def use_moe_token_shard(vllm_config: VllmConfig) -> bool:
    """Whether the routed MoE runs on token shards over a replicated stream.

    Same predicate as the source branch's ``nvidia.model.use_moe_token_shard``:
    ``VLLM_KIMI_K3_MOE_TOKEN_SHARD`` with EP + MegaMoE at TP > 1, PP 1 and DP 1.
    """
    parallel_config = vllm_config.parallel_config
    return (
        envs.VLLM_KIMI_K3_MOE_TOKEN_SHARD
        and vllm_config.kernel_config.moe_backend == "deep_gemm_mega_moe"
        and parallel_config.enable_expert_parallel
        and parallel_config.pipeline_parallel_size == 1
        and parallel_config.data_parallel_size == 1
        and parallel_config.tensor_parallel_size > 1
    )
