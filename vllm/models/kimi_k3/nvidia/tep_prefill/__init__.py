# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 TEP prefill: specialized kernels for the tensor/expert-parallel prefill
of disaggregated serving, enabled by
``additional_config={"prefill_backend": "tep_fused"}``.

Agentic prefill computes a few thousand new tokens per request on top of
100K+ cached context tokens, so the per-request preparation of that context
(gathering the paged latent, up-projecting it to per-head K/V and packing them
for the attention kernel) is a large share of MLA prefill. Here it is one fused
kernel (``context_kv``). Every other part of the step runs the stock path.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TepPrefillConfig:
    pass


def tep_prefill_config(vllm_config) -> TepPrefillConfig | None:
    """The deployment's TEP prefill settings, None when it is not selected.
    Raises when it is requested for an unsupported deployment."""
    extra = vllm_config.additional_config
    backend = (
        extra.get("prefill_backend", "auto") if isinstance(extra, dict) else "auto"
    )
    if backend == "auto":
        return None
    if backend != "tep_fused":
        raise ValueError("Supported prefill_backend values are auto/tep_fused")
    reason = unsupported(vllm_config)
    if reason is not None:
        raise ValueError(f"Kimi K3 TEP prefill does not support {reason}")
    return TepPrefillConfig()


def unsupported(vllm_config) -> str | None:
    from vllm.platforms import current_platform

    parallel = vllm_config.parallel_config
    attention = vllm_config.attention_config
    checks = (
        (not current_platform.is_device_capability_family(100), "this GPU (SM100/103)"),
        (parallel.decode_context_parallel_size != 1, "decode context parallel"),
        (
            vllm_config.cache_config.cache_dtype not in ("fp8", "fp8_e4m3"),
            "MLA KV caches other than plain FP8",
        ),
        (
            not attention.use_prefill_query_quantization,
            "a BF16 prefill query (set use_prefill_query_quantization)",
        ),
    )
    return next((reason for failed, reason in checks if failed), None)
