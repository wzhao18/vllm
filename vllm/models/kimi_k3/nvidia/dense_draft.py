# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dense MLA draft support for Kimi K3.1 (DSA target).

Dense EAGLE3 and DSpark drafts keep their own fp8 MLA cache and decode with
FLASHINFER_MLA, or ROCM_AITER_MLA on ROCm (``configure_dense_draft``). The release
GQA DSpark draft (Inferact/Kimi-K3.1-Pro-DSpark: a speculators ``dspark`` checkpoint,
``Qwen3DSparkModel``) loads as ``dspark_gqa.K3GqaDSparkForCausalLM`` on CUDA.
"""

from typing import Any

from vllm.config import SpeculativeConfig
from vllm.platforms import current_platform
from vllm.v1.attention.backends.registry import AttentionBackendEnum

# The release GQA DSpark draft's architecture on Kimi K3.1 (model registry:
# vllm.models.kimi_k3.nvidia.dspark_gqa.K3GqaDSparkForCausalLM).
GQA_DSPARK_ARCHITECTURE = "K3GqaDSparkModel"


def is_gqa_dspark(spec_config: SpeculativeConfig | None) -> bool:
    """Whether the deployment drafts with the release GQA DSpark draft."""
    model = getattr(spec_config, "draft_model_config", None)
    architectures = getattr(model, "architectures", None) or ()
    return (
        getattr(spec_config, "method", None) == "dspark"
        and GQA_DSPARK_ARCHITECTURE in architectures
    )


def _route_gqa_dspark(spec_config: SpeculativeConfig) -> None:
    """A Qwen3 DSpark GQA draft (speculators ``dspark`` -> ``Qwen3DSparkModel``) on a
    Kimi K3.1 target loads as ``K3GqaDSparkForCausalLM`` (CUDA): the stock math with
    the specialized decode's drafter hooks."""
    config = spec_config.draft_model_config
    if (
        spec_config.method == "dspark"
        and current_platform.is_cuda()
        and list(config.architectures or ()) == ["Qwen3DSparkModel"]
    ):
        config.hf_config.architectures = [GQA_DSPARK_ARCHITECTURE]
        spec_config.update_arch_()


def configure_dense_draft(spec_config: SpeculativeConfig, text_config: Any) -> None:
    """Default an EAGLE3/DSpark draft's attention backend and KV dtype on a
    KDA-hybrid DSA target: dense MLA + fp8 for MLA drafts, FLASH_ATTN +
    model dtype for GQA drafts. MTP is unchanged."""
    if (
        spec_config.method not in ("eagle3", "dspark")
        or getattr(text_config, "linear_attn_config", None) is None
    ):
        return
    draft_text_config = spec_config.draft_model_config.hf_text_config
    is_mla_draft = getattr(draft_text_config, "kv_lora_rank", None) is not None
    if is_mla_draft:
        # MLA draft: no dense MLA kernel reads the target's fp8_ds_mla, so the
        # draft gets its own plain fp8 cache on the platform's dense MLA backend.
        backend = (
            AttentionBackendEnum.ROCM_AITER_MLA
            if current_platform.is_rocm()
            else AttentionBackendEnum.FLASHINFER_MLA
        )
        kv_cache_dtype = "fp8"
    else:
        # GQA draft: its own cache, not the target's sparse backend or fp8_ds_mla.
        backend, kv_cache_dtype = AttentionBackendEnum.FLASH_ATTN, "auto"
        _route_gqa_dspark(spec_config)
    spec_config.attention_backend = spec_config.attention_backend or backend
    spec_config.kv_cache_dtype = spec_config.kv_cache_dtype or kv_cache_dtype
