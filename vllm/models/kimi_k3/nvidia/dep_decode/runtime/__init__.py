# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 DEP decode: a decode executor for data-parallel attention with
expert-parallel MegaMoE (TP1 attention, EP over the DP ranks), enabled by
``additional_config={"decode_backend": "dep_throughput"}``.

Like the latency specialized decode (``specialized_decode/``), it runs every layer of a
planned decode step from prepared providers and falls back to the stock forward
otherwise; it targets the throughput end instead: TP1 projections, which are weight
streaming bound, on the backend's own MXFP8 weights, KDA verify with RecoverSSM (the
stock KDA layer without it), and
the MegaMoE block. Unlike the TP backend, its ranks share no mailboxes: each DP rank
may take either path on any step, since both run the same MegaMoE collective.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DepDecodeConfig:
    # Tokens per request in a verify step (1 + draft tokens; 1 without speculation).
    q: int
    capture_sizes: tuple[int, ...]
    # Cross-rank MLA work sharing (``dep_mla_share``: true or a dict of MlaShare
    # options), as sorted (option, value) pairs; None when off.
    mla_share: tuple | None = None


def dep_decode_config(vllm_config) -> DepDecodeConfig | None:
    """The deployment's DEP decode settings, None when another backend is selected.
    Raises when the DEP decode is requested for an unsupported deployment."""
    extra = vllm_config.additional_config
    backend = extra.get("decode_backend", "auto") if isinstance(extra, dict) else "auto"
    if backend != "dep_throughput":
        return None
    reason = unsupported(vllm_config)
    if reason is not None:
        raise ValueError(f"Kimi K3 DEP decode does not support {reason}")
    spec = vllm_config.speculative_config
    share = extra.get("dep_mla_share", False)
    if share is True:
        share = {}
    elif not share:
        share = None
    elif not isinstance(share, dict):
        raise ValueError("dep_mla_share takes true or a dict of MlaShare options")
    return DepDecodeConfig(
        q=1 + spec.num_speculative_tokens if spec is not None else 1,
        capture_sizes=tuple(
            vllm_config.compilation_config.cudagraph_capture_sizes or ()
        ),
        mla_share=tuple(sorted(share.items())) if share is not None else None,
    )


def unsupported(vllm_config) -> str | None:
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
    checks = (
        (not current_platform.is_device_capability_family(100), "this GPU (SM100/103)"),
        (
            moe != ("situ", 4.0, 25.0, 896, 16, 1, 1, "sigmoid"),
            "MoE configs but the one-group sigmoid top-16 SiTU MoE of Kimi K3",
        ),
        (parallel.tensor_parallel_size != 1, "attention TP other than 1"),
        (not parallel.enable_expert_parallel, "deployments without expert parallelism"),
        (
            vllm_config.kernel_config.moe_backend != "deep_gemm_mega_moe",
            "MoE backends but deep_gemm_mega_moe",
        ),
        (parallel.pipeline_parallel_size != 1, "pipeline parallelism"),
        (parallel.decode_context_parallel_size != 1, "decode context parallel"),
        (parallel.use_ubatching, "DBO"),
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
        # The KDA verify kernel takes at most 8 tokens per request.
        (
            spec is not None and spec.num_speculative_tokens > 7,
            "more than 7 speculative tokens",
        ),
        (
            not str(vllm_config.cache_config.cache_dtype).startswith("fp8"),
            "MLA KV caches other than FP8",
        ),
    )
    return next((reason for failed, reason in checks if failed), None)
