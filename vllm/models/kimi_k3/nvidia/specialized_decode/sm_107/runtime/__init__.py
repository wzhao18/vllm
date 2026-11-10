# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 specialized decode: a CuTe decode executor over the stock backbone, enabled
by ``additional_config={"decode_backend": "latency_specialized"}``.

Two topologies: plain TP8 / TP16 (DCP 1 or = TP, no expert parallelism), and the
sequence-parallel (SP) decode role of K3.1's C512 recipe (``sp_topology``): TP4 = DCP4
attention groups over DP4 with expert-parallel MegaMoE (EP16), where the stock backbone
keeps the residual stream row-sharded over the TP group (``sp_shard``) and the
specialized decode replaces each DSA / KDA layer's all-gather -> attention -> o_proj ->
reduce-scatter (``runtime.sp_attention``).
"""

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
    # The SP topology (``sp_topology``): TP4 = DCP4 attention, residual row-sharded.
    sequence_parallel: bool = False
    dp: int = 1
    # Expert-parallel ranks of the MoE (TP x DP under expert parallelism, else 1).
    ep: int = 1
    # ReplaySSM / RecoverSSM records on the KDA layers (--use-replayssm; SP only).
    replayssm: bool = False


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
    spec, compilation = vllm_config.speculative_config, vllm_config.compilation_config
    parallel, cache = vllm_config.parallel_config, vllm_config.cache_config
    q = 1 + spec.num_speculative_tokens if spec is not None else 1
    sizes = tuple(compilation.cudagraph_capture_sizes or ())
    return DecodeConfig(
        max_context=vllm_config.model_config.max_model_len,
        mla_core=extra.get("mla_core", "auto"),
        dsa_core=extra.get("dsa_core", "auto"),
        dcp=vllm_config.parallel_config.decode_context_parallel_size,
        q=q,
        capture_sizes=sizes,
        bf16_state=kda_dtypes(vllm_config)[1] == torch.bfloat16,
        captured=target_decode_rows(vllm_config, q),
        sequence_parallel=sp_topology(vllm_config),
        dp=parallel.data_parallel_size,
        ep=(
            parallel.tensor_parallel_size * parallel.data_parallel_size
            if parallel.enable_expert_parallel
            else 1
        ),
        replayssm=cache.use_replayssm or cache.use_kda_recoverssm,
    )


def decode_graph_requests(vllm_config, query_lens) -> list[int]:
    """Request counts of the FULL uniform decode graphs a runner captures at the
    query lengths ``query_lens`` ({q: most requests}): this tree's runners key
    decode graphs by requests (``decode_cudagraph_capture_request_sizes``, resolved
    by the config), not by the token counts the source branch's runner rounded."""
    from vllm.config.compilation import (
        decode_cudagraph_capture_max_tokens,
        decode_cudagraph_capture_shapes,
    )

    compilation = vllm_config.compilation_config
    shapes = decode_cudagraph_capture_shapes(
        compilation.decode_cudagraph_capture_request_sizes or (),
        query_lens,
        decode_cudagraph_capture_max_tokens(compilation),
    )
    return sorted({n for n, _ in shapes})


def target_decode_rows(vllm_config, q) -> tuple[int, ...]:
    """Token rows of the target runner's FULL verify graphs (``captured``): n * q for
    each request count n of its uniform decode graphs at query length q."""
    query_lens = vllm_config.get_decode_cudagraph_query_lens(q)
    if set(query_lens) != {q}:
        # (dynamic speculative decoding: decode graphs at several query lengths)
        raise ValueError(
            "Kimi K3 specialized decode (SM107): decode graphs at query lengths "
            f"{sorted(query_lens)}; only one ({q}) is supported"
        )
    return tuple(n * q for n in decode_graph_requests(vllm_config, query_lens))


def sp_topology(vllm_config) -> bool:
    """Whether the deployment is the sequence-parallel decode topology: TP4 = DCP4
    attention groups (one pipeline stage) over expert-parallel MegaMoE, i.e. the stock
    K3 decoder's ``use_sequence_parallel`` at TP4 = DCP4 (this tree's stock model
    has no MoE token shard, so no switch can turn it off). Any native MegaMoE
    backend (DeepGEMM, or Rubin on SM107: ``kimi_k3_mega_moe``): the SP step runs
    the stock MoE module on the shard either way."""
    from vllm.config.kernel import NATIVE_MEGA_MOE_BACKENDS

    parallel = vllm_config.parallel_config
    return (
        parallel.tensor_parallel_size == 4
        and parallel.decode_context_parallel_size == 4
        and parallel.pipeline_parallel_size == 1
        and parallel.enable_expert_parallel
        and vllm_config.kernel_config.moe_backend in NATIVE_MEGA_MOE_BACKENDS
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


def online_linear_quantization(quant_args) -> bool:
    """Whether ``quantization_config`` quantizes linear layers online (a linear
    weight spec or per-layer targets): the decode reads the BF16 projection
    weights directly. An activation-only override (``moe.activation`` for the
    MXFP4 experts) is fine."""
    linear = getattr(quant_args, "linear", None)
    return bool(getattr(quant_args, "targets", None)) or (
        getattr(linear, "weight", None) is not None
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
    # TP4 with expert parallelism and ReplaySSM records only as the SP topology
    # (``sp_unsupported`` narrows it to C512's decode role).
    sp = sp_topology(vllm_config)
    checks = (
        (not current_platform.is_device_capability_family(100), "this GPU (SM100/103)"),
        (
            moe != ("situ", 4.0, 25.0, 896, 16, 1, 1, "sigmoid"),
            "MoE configs but the one-group sigmoid top-16 SiTU MoE of Kimi K3",
        ),
        (
            parallel.tensor_parallel_size not in (8, 16) and not sp,
            "TP other than 8/16 (TP4 only as TP4 = DCP4 sequence parallelism over "
            "expert-parallel MegaMoE)",
        ),
        (
            online_linear_quantization(vllm_config.model_config.quantization_config),
            "online quantization of linear layers (quantization_config.targets or "
            "a linear weight spec; its projections read BF16 weights)",
        ),
        (parallel.pipeline_parallel_size != 1, "pipeline parallelism"),
        (
            parallel.enable_expert_parallel and not sp,
            "expert parallelism (except TP4 = DCP4 sequence parallelism over MegaMoE)",
        ),
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
        # RecoverSSM records (the SP topology's KDA core writes the records).
        (
            kda_dtypes(vllm_config) not in [(torch.bfloat16, s) for s in states],
            "KDA cache dtypes other than a BF16 conv state and an FP32 recurrent "
            "state (mamba cache dtype; a BF16 recurrent state on Rubin only)",
        ),
        (
            (
                vllm_config.cache_config.use_replayssm
                or vllm_config.cache_config.use_kda_recoverssm
            )
            and not sp,
            "ReplaySSM / RecoverSSM (--use-replayssm) outside TP4 = DCP4 sequence "
            "parallelism",
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
        # NQ-NOFALLBACK: batch-sharded sampling bypasses both fast verifications
        # (greedy argmax and sampled), so every verify step would run stock.
        (
            spec is not None
            and getattr(parallel, "enable_batch_sharded_sampling", False),
            "batch-sharded sampling (--enable-batch-sharded-sampling)",
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
    if reason is None and sp:
        reason = sp_unsupported(vllm_config)
    return reason


# The SP topology's verify rows: max_num_seqs x q (the shared scratch, the DSA box
# and the o_proj mailboxes are sized for 512 rows, 128 per SP shard).
SP_MAX_ROWS = 512


def sp_unsupported(vllm_config) -> str | None:
    """Why the SP topology (``sp_topology``) is not C512's decode role, the one the
    SP specialized decode covers: Rubin, K3.1 DSA layers on a per-tensor FP8 KV cache
    under FlashInfer sparse's trtllm-gen kernel, a BF16 conv / FP32 recurrent KDA
    state, DSpark q4, at most 512 verify rows, the stock (unfused) reduce-scatter
    o_proj and DCP = TP."""
    import torch

    from vllm import envs
    from vllm.platforms import current_platform
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    parallel, spec = vllm_config.parallel_config, vllm_config.speculative_config
    cache = vllm_config.cache_config
    q = 1 + spec.num_speculative_tokens if spec is not None else 1
    fi_sparse = (
        vllm_config.attention_config.backend
        == AttentionBackendEnum.FLASHINFER_MLA_SPARSE
    )
    dsa = getattr(vllm_config.model_config.hf_text_config, "dsa_config", None)
    fp8 = cache.cache_dtype in ("fp8", "fp8_e4m3")
    checks = (
        (
            not current_platform.is_device_capability((10, 7)),
            "this GPU (Rubin SM107 only)",
        ),
        (dsa is None, "models without DSA layers (Kimi K3 MLA)"),
        (
            not fp8,
            "KV cache dtypes other than per-tensor fp8 (FlashInfer sparse trtllm-gen)",
        ),
        (not fi_sparse, "attention backends but FLASHINFER_MLA_SPARSE"),
        # The SP DSA core runs trtllm-gen; stock must too (parity), also under
        # "auto", which picks prims_ts under DCP where it can run.
        (
            vllm_config.kernel_config.flashinfer_sparse_mla_kernel == "prims_ts"
            or (fi_sparse and fp8 and _stock_prims_ts(vllm_config)),
            "FlashInfer sparse's prims_ts kernel (trtllm-gen only; "
            "flashinfer_sparse_mla_kernel=trtllm)",
        ),
        (
            kda_dtypes(vllm_config) != (torch.bfloat16, torch.float32),
            "KDA cache dtypes other than a BF16 conv state and an FP32 recurrent state",
        ),
        (
            spec is None or spec.method != "dspark" or spec.num_speculative_tokens != 3,
            "speculation other than DSpark with 3 draft tokens (q4)",
        ),
        (
            vllm_config.scheduler_config.max_num_seqs * q > SP_MAX_ROWS,
            f"more than {SP_MAX_ROWS} verify rows (max_num_seqs x q)",
        ),
        (
            envs.VLLM_ENABLE_GEMM_RS,
            "the fused GEMM + reduce-scatter o_proj (VLLM_ENABLE_GEMM_RS)",
        ),
        (
            parallel.decode_context_parallel_size != parallel.tensor_parallel_size,
            "decode context parallel other than TP",
        ),
    )
    reason = next((reason for failed, reason in checks if failed), None)
    return None if reason is None else f"sequence parallelism on {reason}"


def _stock_prims_ts(vllm_config) -> bool:
    """Whether stock FLASHINFER_MLA_SPARSE resolves the per-tensor FP8 cache to
    prims_ts (``use_prims_ts``; "prims_ts" itself is refused before)."""
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import use_prims_ts

    return use_prims_ts(vllm_config, vllm_config.cache_config.cache_dtype)


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
