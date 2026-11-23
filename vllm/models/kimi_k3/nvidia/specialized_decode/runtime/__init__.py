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
    # The SP topology's DSA-DP variant (``dsa_dp_topology``): TP4 attention groups at
    # DCP1 whose DSA layers run data-parallel inside the TP group (ops/dsa_dp.py).
    dsa_dp: bool = False


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
    parallel, cache = vllm_config.parallel_config, vllm_config.cache_config
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
        # DSA-DP: the runner rounds every FULL decode graph to whole q-token requests
        # per TP rank (``row_multiple``; ops/dsa_dp.py dep_order).
        captured=captured_rows(sizes, q, limit, multiple=dsa_dp_rows(vllm_config)),
        sequence_parallel=sp_topology(vllm_config),
        dsa_dp=dsa_dp_topology(vllm_config),
        dp=parallel.data_parallel_size,
        ep=(
            parallel.tensor_parallel_size * parallel.data_parallel_size
            if parallel.enable_expert_parallel
            else 1
        ),
        replayssm=cache.use_replayssm or cache.use_kda_recoverssm,
    )


def sp_layout(vllm_config) -> bool:
    """TP4 attention groups (one pipeline stage) over expert-parallel MegaMoE without
    the MoE token shard: the stock K3 decoder's ``use_sequence_parallel`` at TP4. Any
    native MegaMoE backend (DeepGEMM, or Rubin on SM107): the SP step runs the stock
    MoE module on the shard either way. Its DCP is ``sp_topology``'s."""
    from vllm.config.kernel import NATIVE_MEGA_MOE_BACKENDS

    parallel = vllm_config.parallel_config
    if not (
        parallel.tensor_parallel_size == 4
        and parallel.pipeline_parallel_size == 1
        and parallel.enable_expert_parallel
        and vllm_config.kernel_config.moe_backend in NATIVE_MEGA_MOE_BACKENDS
    ):
        return False
    from ...model import use_moe_token_shard

    return not use_moe_token_shard(vllm_config)


def dsa_dp_topology(vllm_config) -> bool:
    """The SP layout at DCP1 on a model with DSA layers (K3.1): DSA-DP decode ("DSA
    DP, GEMM TP", ops/dsa_dp.py). Each request's DSA rows run on its owner card with
    all heads (stock ``dsa_dp.run_segment``, declared); KDA, MoE and the layer
    boundaries are the SP step's. ``dsa_dp.supported`` admits DSA-DP under the
    specialized decode on this layout only (its plain TP8 / TP16 layouts keep the
    replicated DSA KV). Config values only."""
    parallel = vllm_config.parallel_config
    text = vllm_config.model_config.hf_text_config
    return (
        getattr(text, "dsa_config", None) is not None
        and parallel.decode_context_parallel_size == 1
        and getattr(parallel, "prefill_context_parallel_size", 1) == 1
        and sp_layout(vllm_config)
    )


def dsa_dp_rows(vllm_config) -> int:
    """Requests per FULL decode graph are a multiple of this (the runner's DSA-DP
    ``row_multiple``: TP), 1 outside DSA-DP."""
    if not dsa_dp_topology(vllm_config):
        return 1
    return vllm_config.parallel_config.tensor_parallel_size


def sp_topology(vllm_config) -> bool:
    """Whether the deployment is the sequence-parallel decode topology: the SP layout
    (``sp_layout``) at TP4 = DCP4 (C512's decode role: SP DSA blocks over the DCP4
    DSA KV), or at DCP1 with DSA-DP (``dsa_dp_topology``)."""
    if not sp_layout(vllm_config):
        return False
    dcp = vllm_config.parallel_config.decode_context_parallel_size
    return dcp == 4 or dsa_dp_topology(vllm_config)


def conv_dtypes():
    """KDA conv-state dtypes the specialized decode serves."""
    import torch

    return (torch.bfloat16, torch.float32)


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
            "TP other than 8/16 (TP4 only as TP4 = DCP4 or DSA-DP DCP1 sequence "
            "parallelism over expert-parallel MegaMoE)",
        ),
        (
            online_linear_quantization(vllm_config.model_config.quantization_config),
            "online quantization of linear layers (quantization_config.targets or "
            "a linear weight spec; its projections read BF16 weights)",
        ),
        (parallel.pipeline_parallel_size != 1, "pipeline parallelism"),
        (
            parallel.enable_expert_parallel and not sp,
            "expert parallelism (except TP4 = DCP4 or DSA-DP DCP1 sequence "
            "parallelism over MegaMoE)",
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
        # The KDA cores keep a BF16 or FP32 conv state (``--mamba-cache-dtype``; the
        # cores read and write it at its own dtype, the window holds the BF16
        # in-projection inputs either way) and an FP32 (Rubin: or BF16) recurrent
        # state in place; no other cache dtypes, nor ReplaySSM / RecoverSSM records
        # outside SP (the SP topology's KDA core writes the records).
        (
            kda_dtypes(vllm_config)
            not in [(c, s) for c in conv_dtypes() for s in states],
            "KDA cache dtypes other than a BF16 / FP32 conv state and an FP32 "
            "recurrent state (mamba cache dtype; a BF16 recurrent state on Rubin "
            "only)",
        ),
        (
            (
                vllm_config.cache_config.use_replayssm
                or vllm_config.cache_config.use_kda_recoverssm
            )
            and not sp,
            "ReplaySSM / RecoverSSM (--use-replayssm) outside TP4 sequence "
            "parallelism (DCP4 or DSA-DP)",
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
    if reason is None:
        reason = drafter_unsupported(vllm_config, sp)
    return reason


# DSpark draft architectures whose drafters the specialized decode specializes
# (``runtime.draft``): the preview MLA drafter (``dspark_mla``) and the release GQA
# drafter (``dspark_gqa``).
KIMI_DRAFTERS = ("K3DSparkModel", "K3GqaDSparkModel")


def drafter_unsupported(vllm_config, sp) -> str | None:
    """NQ-NOFALLBACK: on the deployments the no-fallback rule covers (Rubin TP16 =
    DCP16 with DSpark, and the SP topology) a DSpark draft must be one of
    KIMI_DRAFTERS; any other would draft every step on unspecialized stock modules."""
    from vllm.platforms import current_platform

    from ..shapes import target_topology

    spec, parallel = vllm_config.speculative_config, vllm_config.parallel_config
    draft = getattr(spec, "draft_model_config", None)
    if spec is None or spec.method != "dspark" or draft is None:
        return None
    strict = sp or target_topology(
        current_platform.is_device_capability((10, 7)),
        parallel.tensor_parallel_size,
        parallel.decode_context_parallel_size,
        1 + spec.num_speculative_tokens,
    )
    architectures = tuple(draft.architectures or ())
    if strict and not set(architectures) & set(KIMI_DRAFTERS):
        return (
            f"DSpark drafts {architectures} (the specialized drafters: "
            f"{', '.join(KIMI_DRAFTERS)}; any other would draft on stock modules)"
        )
    return None


# The SP topology's verify rows: max_num_seqs x q. The shared scratch, the DSA box,
# the SP mailboxes and the per-family kernels are sized at build from the captured
# rows (rows[-1]); a family that cannot serve a captured shape raises Unsupported at
# build. The bound here is the layer-0 DenseFlow's (dense_flow.MAX_ROWS shard rows
# per GPU x TP 4 = 1024 verify rows); above it the SP topology is refused at load.
SP_MAX_ROWS = 1024


def sp_unsupported(vllm_config) -> str | None:
    """Why the SP topology (``sp_topology``) is not C512's decode role, the one the
    SP specialized decode covers: Rubin, K3.1 DSA layers on a per-tensor FP8 KV cache
    under FlashInfer sparse's trtllm-gen kernel, a BF16 conv / FP32 recurrent KDA
    state, DSpark q2..q8 on captured rows that are multiples of lcm(q, TP), at most
    SP_MAX_ROWS verify rows and DCP = TP. (The stock unfused
    reduce-scatter o_proj it also needs is a model policy, checked per layer at
    build: ``sp_attention.sp_layer``.)"""
    import torch

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
    dsa_dp = dsa_dp_topology(vllm_config)
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
            kda_dtypes(vllm_config) not in [(c, torch.float32) for c in conv_dtypes()],
            "KDA cache dtypes other than a BF16 / FP32 conv state and an FP32 "
            "recurrent state",
        ),
        (
            spec is None
            or spec.method != "dspark"
            or not 1 <= spec.num_speculative_tokens <= 7,
            "speculation other than DSpark with 1..7 draft tokens (q2..q8)",
        ),
        (
            vllm_config.scheduler_config.max_num_seqs * q > SP_MAX_ROWS,
            f"more than {SP_MAX_ROWS} verify rows (max_num_seqs x q)",
        ),
        (
            parallel.decode_context_parallel_size != parallel.tensor_parallel_size
            and not dsa_dp,
            "decode context parallel other than TP (or DCP1 with DSA-DP)",
        ),
        # DSA-DP: the release (GQA) drafter's strict build (``runtime.draft``); the
        # preview drafter's specialized attention (DraftDcp) needs DCP = TP.
        (
            dsa_dp and not _gqa_drafter(spec),
            "DSA-DP with a DSpark drafter but the release GQA one (K3GqaDSparkModel)",
        ),
    )
    reason = next((reason for failed, reason in checks if failed), None)
    if reason is None:
        tp, multiple = parallel.tensor_parallel_size, dsa_dp_rows(vllm_config)
        reason = sp_unaligned(vllm_config, q, tp, multiple=multiple)
    return None if reason is None else f"sequence parallelism on {reason}"


def sp_row_step(q: int, tp: int, multiple: int = 1) -> int:
    """The SP decode's captured rows are multiples of this: lcm(q, TP) (whole q-token
    requests that split evenly over the TP ranks), and under DSA-DP (``multiple`` =
    TP) whole requests per TP rank, q x TP (ops/dsa_dp.py dep_order)."""
    import math

    return math.lcm(q, tp, q * multiple)


def sp_aligned_rows(
    q: int, tp: int, top: int, multiple: int = 1
) -> tuple[int, ...]:
    """The verify row counts the SP decode can capture up to ``top``
    (``sp_row_step``)."""
    step = sp_row_step(q, tp, multiple)
    return tuple(range(step, top + 1, step))


def sp_unaligned(vllm_config, q: int, tp: int, multiple: int = 1) -> str | None:
    """Why the deployment's FULL decode graphs do not suit the SP decode: it shards
    every captured R = B x q verify rows evenly over the TP ranks (rank r owns rows
    [r R / TP, (r + 1) R / TP)), so every R must be a multiple of ``sp_row_step``.
    Batches between two captured sizes run the next captured graph with the runner's
    padded requests (their rows are discarded). Where the deployment captures its
    largest batch (max_num_seqs x q within the largest capture size), that batch needs
    an aligned graph too, else it would run the stock forward. None when aligned."""
    from ..shapes import captured_rows

    compilation = vllm_config.compilation_config
    sizes = tuple(compilation.cudagraph_capture_sizes or ())
    seqs = vllm_config.scheduler_config.max_num_seqs
    limit = min(seqs * q, compilation.max_cudagraph_capture_size or max(sizes, default=0))
    rows = captured_rows(sizes, q, limit, multiple=multiple)
    step = sp_row_step(q, tp, multiple)
    bad = [r for r in rows if r % step]
    top = seqs * q
    # An unaligned largest batch within the capture range has no FULL graph at any
    # capture list. (A batch above the largest capture size, or one whose aligned
    # size the list skips, runs eager by the deployment's own choice at any q, k3
    # included: counted by StockSteps, not this check's concern.)
    cap = compilation.max_cudagraph_capture_size or max(sizes, default=0)
    top_ok = top > cap or top % step == 0
    if not bad and top_ok:
        return None
    want = sp_aligned_rows(q, tp, limit - limit % step, multiple)
    hint = (
        f"capture sizes {list(want[:3])} .. {want[-1] if want else step} (every "
        f"{step} rows) and max_num_seqs a multiple of {step // q}"
    )
    if bad:
        return (
            f"captured q{q} decode rows {bad[:6]}{' ...' if len(bad) > 6 else ''} "
            f"that do not split over TP{tp} (rows must be multiples of {step}; use "
            f"{hint})"
        )
    return (
        f"max_num_seqs {seqs} x q{q} = {top} rows without an aligned FULL decode "
        f"graph (rows must be multiples of {step}, the largest batch captured; use "
        f"{hint})"
    )


def _gqa_drafter(spec) -> bool:
    draft = getattr(spec, "draft_model_config", None)
    architectures = getattr(draft, "architectures", None) or ()
    return "K3GqaDSparkModel" in architectures


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
