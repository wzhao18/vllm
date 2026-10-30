# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 specialized decode: a CuTe decode executor over the stock backbone, enabled
by ``additional_config={"decode_backend": "latency_specialized"}``."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DecodeConfig:
    max_context: int
    mla_core: str
    dsa_core: str
    dcp: int
    # Tokens per request in a verify step (1 + draft tokens; 1 without speculation).
    q: int
    capture_sizes: tuple[int, ...]
    # KDA recurrent state in BF16 (--mamba-ssm-cache-dtype bfloat16; Rubin only): the
    # KdaHead layers need it.
    bf16_state: bool = False
    # Token rows of the runner's FULL decode graphs (``shapes.captured_rows``).
    captured: tuple[int, ...] = ()


def specialized_decode_config(vllm_config) -> DecodeConfig | None:
    """The deployment's specialized decode settings, None for the stock decode.
    Raises when the specialized decode is requested for an unsupported deployment."""
    extra = vllm_config.additional_config
    backend = extra.get("decode_backend", "auto") if isinstance(extra, dict) else "auto"
    if backend == "auto":
        return None
    if backend != "latency_specialized":
        raise ValueError("Supported decode_backend values are auto/latency_specialized")
    import torch

    reason = unsupported(vllm_config)
    if reason is not None:
        raise ValueError(f"Kimi K3 specialized decode does not support {reason}")
    from ..shapes import captured_rows

    spec, compilation = vllm_config.speculative_config, vllm_config.compilation_config
    q = 1 + spec.num_speculative_tokens if spec is not None else 1
    sizes = tuple(compilation.cudagraph_capture_sizes or ())
    # The runner's uniform decode graphs (cudagraph_utils): at most max_num_seqs
    # requests and the largest capture size.
    limit = vllm_config.scheduler_config.max_num_seqs * q
    limit = min(limit, compilation.max_cudagraph_capture_size or max(sizes, default=0))
    return DecodeConfig(
        max_context=vllm_config.model_config.max_model_len,
        mla_core=extra.get("mla_core", "auto"),
        dsa_core=extra.get("dsa_core", "auto"),
        dcp=vllm_config.parallel_config.decode_context_parallel_size,
        q=q,
        capture_sizes=sizes,
        bf16_state=kda_dtypes(vllm_config)[1] == torch.bfloat16,
        captured=captured_rows(sizes, q, limit),
    )


def kda_dtypes(vllm_config):
    """(conv, recurrent) KDA state dtypes the deployment's cache config selects."""
    from vllm.model_executor.layers.mamba.mamba_utils import MambaStateDtypeCalculator

    cache = vllm_config.cache_config
    return MambaStateDtypeCalculator.kda_state_dtype(
        vllm_config.model_config.dtype,
        cache.mamba_cache_dtype,
        cache.mamba_ssm_cache_dtype,
    )


def unsupported(vllm_config) -> str | None:
    import torch

    from vllm import envs
    from vllm.config import CompilationMode, CUDAGraphMode
    from vllm.platforms import current_platform

    parallel, spec = vllm_config.parallel_config, vllm_config.speculative_config
    text = vllm_config.model_config.hf_text_config
    moe = tuple(
        getattr(text, name, None)
        for name in (
            "hidden_act",
            "activation_situ_beta",
            "activation_situ_linear_beta",
            "num_experts",
            "num_experts_per_token",
            "num_expert_group",
            "topk_group",
            "moe_router_activation_func",
        )
    )
    compilation = vllm_config.compilation_config
    graphs = compilation.cudagraph_mode
    # KdaSplit's BF16-state variant runs on Rubin (compute capability 10.7, the key of
    # its schedules, ``kda.prepared.BF16_DEVICE_SCHEDULE``) only; elsewhere the
    # recurrent state stays FP32.
    states = [torch.float32]
    if current_platform.is_device_capability((10, 7)):
        states.append(torch.bfloat16)
    # Kimi K3.1: sparse (DSA) MLA layers over the packed MLA and indexer caches,
    # fp8_ds_mla on FlashMLA sparse or per-tensor FP8 on FlashInfer sparse.
    dsa = getattr(text, "dsa_config", None) is not None
    checks = (
        (not current_platform.is_device_capability_family(100), "this GPU (SM100/103)"),
        (
            moe != ("situ", 4.0, 25.0, 896, 16, 1, 1, "sigmoid"),
            "MoE configs but the one-group sigmoid top-16 SiTU MoE of Kimi K3",
        ),
        (parallel.tensor_parallel_size not in (8, 16), "TP other than 8/16"),
        (parallel.pipeline_parallel_size != 1, "pipeline parallelism"),
        (parallel.enable_expert_parallel, "expert parallelism"),
        (parallel.use_ubatching, "DBO"),
        (
            parallel.decode_context_parallel_size
            not in (1, parallel.tensor_parallel_size),
            "decode context parallel other than 1 or TP",
        ),
        (vllm_config.model_config.enable_sleep_mode, "sleep mode"),
        (vllm_config.lora_config is not None, "LoRA"),
        # The runner's ModelState hooks build the executor and prepare the KV layouts.
        (not vllm_config.use_v2_model_runner, "the V1 model runner"),
        # The decode loop runs in forward; a compiled forward would trace it.
        (compilation.mode == CompilationMode.STOCK_TORCH_COMPILE, "torch.compile"),
        # The loop runs only in FULL decode graphs; eager and PIECEWISE steps run stock.
        (
            graphs is None or graphs.decode_mode() != CUDAGraphMode.FULL,
            "deployments without FULL decode CUDA graphs (e.g. enforce-eager)",
        ),
        (
            spec is not None and spec.method != "dspark",
            "speculative methods but DSpark",
        ),
        (
            spec is not None and spec.enable_adaptive_verification,
            "adaptive verification",
        ),
        # KdaSplit keeps a BF16 conv state and an FP32 (Rubin: or BF16) recurrent
        # state in place; it maintains neither other cache dtypes nor ReplaySSM /
        # RecoverSSM records.
        (
            kda_dtypes(vllm_config) not in [(torch.bfloat16, s) for s in states],
            "KDA cache dtypes other than a BF16 conv state and an FP32 recurrent "
            "state (mamba cache dtype; a BF16 recurrent state on Rubin only)",
        ),
        (
            vllm_config.cache_config.use_replayssm
            or vllm_config.cache_config.use_kda_recoverssm,
            "ReplaySSM / RecoverSSM (--use-replayssm)",
        ),
        # The verify kernels take at most 8 tokens per request (KdaSplit's barriers).
        (
            spec is not None and spec.num_speculative_tokens > 7,
            "more than 7 speculative tokens",
        ),
        (
            spec is not None and envs.VLLM_KIMI_K3_AUX_ATTN_RES_STREAM,
            "the AttnRes aux stream",
        ),
        (
            dsa
            and vllm_config.cache_config.cache_dtype
            not in ("fp8_ds_mla", "fp8", "fp8_e4m3", "nvfp4_ds_mla"),
            "DSA layers on KV cache dtypes other than fp8_ds_mla (FlashMLA sparse), "
            "per-tensor fp8 (FlashInfer sparse) or nvfp4_ds_mla (FlashInfer sparse "
            "prims_ts)",
        ),
    )
    reason = next((reason for failed, reason in checks if failed), None)
    if (
        reason is None
        and dsa
        and vllm_config.cache_config.cache_dtype == "nvfp4_ds_mla"
    ):
        reason = nvfp4_unsupported(vllm_config)
    return reason


def nvfp4_unsupported(vllm_config) -> str | None:
    """Why the DSA layers cannot run on an nvfp4_ds_mla cache: the specialized
    decode writes it only for FlashInfer sparse's prims_ts kernel."""
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
        _prims_ts_unsupported_reason,
    )
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    backend = vllm_config.attention_config.backend
    if backend != AttentionBackendEnum.FLASHINFER_MLA_SPARSE:
        return (
            "nvfp4_ds_mla DSA layers on attention backends but "
            "FLASHINFER_MLA_SPARSE (prims_ts)"
        )
    if vllm_config.kernel_config.flashinfer_sparse_mla_kernel == "trtllm":
        return "nvfp4_ds_mla DSA layers on the trtllm-gen sparse MLA kernel"
    reason = _prims_ts_unsupported_reason(vllm_config, "nvfp4_ds_mla")
    if reason is not None:
        return f"nvfp4_ds_mla DSA layers without FlashInfer prims_ts: {reason}"
    return None
