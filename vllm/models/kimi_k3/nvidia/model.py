# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 multimodal model implementation for vLLM."""

import math
from collections.abc import Hashable, Iterable
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import nn

import vllm._custom_ops as ops
import vllm.envs as envs
from vllm.models.kimi_k3.nvidia import breadcrumb
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.config.kernel import KIMI_K3_MEGA_MOE, NATIVE_MEGA_MOE_BACKENDS
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul, SituAndMul
from vllm.model_executor.layers.fused_moe import (
    FusedMoEFactory,
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import (
    MEGA_MOE_MXFP4,
    MEGA_MOE_NVFP4,
    DeepGemmMegaMoEExperts,
    MegaMoEActivation,
    MegaMoEWeightFormat,
)
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
    fused_grouped_topk,
)
from vllm.model_executor.layers.fusion.mm_input_norm import build_mm_input_norm
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    LinearBase,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import (
    KimiGatedDeltaNetAttention as KimiLinearGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.compressed_tensors import (
    compressed_tensors,
)
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptMixedPrecisionConfig,
    ModelOptNvFp4Config,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.interfaces import (
    EagleModelMixin,
    HasInnerState,
    IsHybrid,
    MambaStateShapes,
    MixtureOfExperts,
    SupportsEagle3,
    SupportsEncoderCudaGraph,
    SupportsMultiModal,
    SupportsPP,
    SupportsQuant,
    SupportsReplaySSM,
)
from vllm.model_executor.models.kimi_k25 import KimiK25MediaPixelInputs
from vllm.model_executor.models.kimi_k25_vit import (
    KimiK25MultiModalProjector,
    MoonViT3dPretrainedModel,
    vision_tower_forward,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    init_vllm_registered_model,
    is_pp_missing_parameter,
    make_layers,
    maybe_prefix,
    spec_decode_needs_target_embed,
)
from vllm.model_executor.models.vision import is_vit_use_data_parallel
from vllm.models.common.ops.sequence_parallel import (
    sp_all_gather,
    sp_padding_mask,
    sp_reduce_scatter,
    sp_shard,
)
from vllm.models.kimi_k3.common.dsa_cache import packed_dsa_block_size
from vllm.models.kimi_k3.nvidia import dense_dwdp
from vllm.models.kimi_k3.nvidia.dsa import KimiK3DSAAttention
from vllm.models.kimi_k3.nvidia.gqa_draft import GQADraftKVCacheGrouping
from vllm.models.kimi_k3.nvidia.kda import KimiK3DeltaAttention
from vllm.models.kimi_k3.nvidia.latent_moe_runner import (
    LatentMoERunner,
)
from vllm.models.kimi_k3.nvidia.low_latency_gemm import (
    enable_kimi_k3_low_latency_gemm,
)
from vllm.models.kimi_k3.nvidia.mla import MultiHeadLatentAttention
from vllm.models.kimi_k3.nvidia.mxfp8 import (
    PackedMxfp8Linear,
    can_pack_mxfp8,
    takes_mxfp8_activation,
)
from vllm.models.kimi_k3.nvidia.ops import (
    attn_res,
    dsa_dp,
    mxfp8_act,
    prefill_glue,
    sp_local_aux,
)
from vllm.models.kimi_k3.nvidia.ops import sp_all_gather as sp_ag
from vllm.models.kimi_k3.nvidia.specialized_prefill import (
    attn_res_dielocal as prefill_attn_res_dielocal,
)
from vllm.models.kimi_k3.nvidia.specialized_prefill import (
    attn_res_lp as prefill_attn_res_lp,
)
from vllm.models.kimi_k3.nvidia.specialized_prefill import (
    attn_res_mxfp8 as prefill_attn_res_mxfp8,
)
from vllm.models.kimi_k3.nvidia.specialized_prefill import (
    attn_res_resid as prefill_attn_res_resid,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import NestedTensors
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.kimi_k3 import KimiK3Config
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig
from vllm.utils.math_utils import cdiv
from vllm.utils.multi_stream_utils import maybe_execute_in_parallel
from vllm.utils.torch_utils import aux_stream, is_meta_module

from ..common.mm_preprocess import (
    KimiK3DummyInputsBuilder,
    KimiK3MultiModalProcessor,
    KimiK3ProcessingInfo,
)
from .specialized_decode.runtime import specialized_decode_config

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.model_states.interface import ModelState

    from .specialized_decode.runtime.decode import SpecializedDecode


def _attn_res(
    prefix: torch.Tensor, delta: torch.Tensor | None, blocks: torch.Tensor,
    *args, call: int | None = None, prefix_out: torch.Tensor | None = None,
    **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """AttnRes with a BF16 output -> (output, updated prefix): on a prefill
    forward planned by ``attn_res_lp.begin`` the precomputed-logit kernel
    (AttnRes call index ``call``; the prefix moves to the die-local prefix
    buffer, or to ``prefix_out`` when given), the native one otherwise."""
    if call is not None and prefill_attn_res_lp.active(blocks, prefix):
        keep = prefix_out is not None and _aux_copy_keeps_prefix(prefix, delta, blocks)
        if prefix_out is None:
            prefix_out = prefill_attn_res_dielocal.prefix_out(prefix, blocks)
        out, _ = prefill_attn_res_lp.attn_res(
            prefix, delta, blocks, call, kwargs["num_blocks"],
            kwargs["block_write_idx"], args[2], kwargs["output_norm_eps"],
            bf16_out=True, quant_out=False, prefix_out=prefix_out,
        )
        return out, prefix if prefix_out is None or keep else prefix_out
    if prefix_out is not None:
        raise RuntimeError(
            "Kimi K3 AttnRes: an explicit prefix_out needs the planned prefill "
            "(attn_res_lp) kernel"
        )
    return attn_res(prefix, delta, blocks, *args, **kwargs), prefix


def _aux_copy_keeps_prefix(
    prefix: torch.Tensor, delta: torch.Tensor | None, blocks: torch.Tensor
) -> bool:
    """For an AttnRes call that writes the DSpark aux capture (an explicit
    ``prefix_out``): whether the capture is a copy and the die-local prefix
    stays the prefix. True under prefill dataflow C (no delta: the producers
    already added the previous outputs into the die-local prefix in place):
    the call writes the unchanged prefix to the capture as a side output and
    the next producers keep adding into the die-local prefix (they never
    touch the capture). Otherwise (a delta) the updated prefix moves to the
    capture and the call after it moves it back."""
    return delta is None and prefill_attn_res_dielocal.is_prefix(prefix, blocks)


logger = init_logger(__name__)

# Token-count cutoff for overlapping the MoE router gate with the routed-expert
# down projection on a separate CUDA stream (latent MoE). At or below this many
# tokens the launch-bound decode path benefits from multi-stream overlap; above
# it the GEMMs saturate the device and the cross-stream sync is pure overhead,
# so it falls back to sequential.
_ROUTED_DOWN_PROJ_STREAM_TOKEN_THRESHOLD = 256
# Prefill chunks from this many tokens overlap again: the router GEMM (N = 896,
# a few waves) and its latency-bound top-k fill the latent down GEMM's last
# waves instead of running alone.
_ROUTED_DOWN_PROJ_STREAM_PREFILL_TOKENS = 4096


def shard_sequence_parallel_mlp(
    hidden_size: int,
    intermediate_size: int,
    use_sequence_parallel: bool,
    eligible: bool,
) -> bool:
    """Whether to TP-shard a sequence-parallel MLP instead of replicating it.

    Opt-in via ``VLLM_KIMI_K3_SHARD_SP_SHARED_EXPERT``; see :class:`KimiMLP` for
    the trade-off and :mod:`vllm.envs` for when it is worth enabling.
    """
    enabled = envs.VLLM_KIMI_K3_SHARD_SP_SHARED_EXPERT
    if not (use_sequence_parallel and eligible and enabled):
        return False
    tp_size = get_tensor_model_parallel_world_size()
    return (
        tp_size > 1 and intermediate_size % tp_size == 0 and hidden_size % tp_size == 0
    )


def can_shard_sequence_parallel_shared_expert(vllm_config: VllmConfig) -> bool:
    """Whether the active MoE backend preserves the SP shared-expert contract.

    MegaMoE invokes the shared expert directly. DeepEP v2 invokes it through
    ``MoERunner``, but its EP combine returns the routed output in the local
    sequence-sharded layout, so the shared expert can use the same
    all-gather/partial-GEMM/reduce-scatter implementation.
    """
    return (
        vllm_config.kernel_config.moe_backend in NATIVE_MEGA_MOE_BACKENDS
        or vllm_config.parallel_config.all2all_backend == "deepep_v2"
    )


def decode_weights_on_dies(vllm_config: VllmConfig) -> bool:
    """The SP specialized decode on a two-die part (VR200): its die-local GEMMs' BF16
    weights are placed on the dies as each decoder layer is built."""
    if specialized_decode_config(vllm_config) is None:
        return False
    from .specialized_decode.cutedsl.gemm.die_local import two_dies
    from .specialized_decode.runtime import sp_topology

    return sp_topology(vllm_config) and two_dies(torch.cuda.current_device())


def pd_prefill_worker(vllm_config: VllmConfig) -> bool:
    """Whether this instance is the prefill side of a P/D deployment: a KV producer
    (kv_producer, or kv_both) that is not a decode instance. A kv_both instance on the
    latency-specialized decode backend is the decode side (Dynamo maps its decode
    workers to kv_consumer, but kv_both decode recipes exist): it keeps the decode
    MegaMoE buckets and die-local streams."""
    kv_transfer = getattr(vllm_config, "kv_transfer_config", None)
    if kv_transfer is None or not kv_transfer.is_kv_producer:
        return False
    extra = getattr(vllm_config, "additional_config", None)
    backend = extra.get("decode_backend", "auto") if isinstance(extra, dict) else "auto"
    return not (backend == "latency_specialized" and kv_transfer.is_kv_consumer)


def prefill_full_grid_shared_overlap(vllm_config: VllmConfig) -> bool:
    """Whether the sequence-parallel MegaMoE layers keep the full persistent grid and
    overlap the replicated shared expert by stream priority (``KimiMoE._overlapped_forward``)
    instead of capping MegaMoE at RUBIN_SHARED_OVERLAP_MAX_SMS beside a side-stream shared
    expert. P/D prefill workers: their steps are SM-throughput-bound in MegaMoE, so the
    cap costs more than it hides (TP8+DCP8 NVFP4: 2K / 4K tokens per GPU -11.4 / -23.6 ms
    per step eager, 128-1024 rows per rank no slower). Decode and unified serving keep
    the decode SP measurement's cap. Decided by role, no switch."""
    return pd_prefill_worker(vllm_config)


_routed_priority_streams: dict[int, torch.cuda.Stream] = {}


def routed_priority_stream() -> torch.cuda.Stream:
    """The current device's high-priority stream (above the default priority 0 of
    the forward's stream) for the MegaMoE routing chain beside the shared expert
    (``KimiMoE._overlapped_forward``), one per device."""
    device = torch.accelerator.current_device_index()
    stream = _routed_priority_streams.get(device)
    if stream is None:
        stream = _routed_priority_streams[device] = torch.cuda.Stream(priority=-1)
    return stream


def use_sequence_parallel(vllm_config: VllmConfig) -> bool:
    """Whether the decoder runs sequence parallelism: each TP rank owns a 1/TP token
    shard between the attention blocks (o_proj GEMM-RS, SP all-gather before the
    next layer's projections) and the routed MoE runs on that shard.

    Expert-parallel MegaMoE (or any DP > 1 EP layout) at TP > 1 in one pipeline
    stage, for every role: P/D prefill workers (kv_producer / kv_both), decode
    instances and unified serving alike."""
    parallel_config = vllm_config.parallel_config
    use_mega_moe = vllm_config.kernel_config.moe_backend in NATIVE_MEGA_MOE_BACKENDS
    return (
        parallel_config.pipeline_parallel_size == 1
        and parallel_config.enable_expert_parallel
        and parallel_config.tensor_parallel_size > 1
        and (use_mega_moe or parallel_config.data_parallel_size > 1)
    )


# Prefill-sized SP batches (>= 2048 tokens: >= 29 MB of bf16 rows per rank, past
# the one-shot all-reduce range) reduce-scatter the embedding lookup.
_SP_EMBED_RS_MIN_TOKENS = 2048


def sp_embed(embed: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """This rank's sequence-parallel token shard of ``embed(input_ids)``.

    The vocab-parallel lookup leaves each token's row on the one rank whose
    vocab shard owns it (zeros elsewhere), so reduce-scattering the shards'
    partial rows over the TP group gives exactly the all-reduced rows of this
    rank's shard: half the bytes of the all-reduce, and the full-batch rows that
    ``sp_shard`` would drop are never reduced. Embeddings that are not the plain
    fused bf16/fp16 lookup keep the all-reduce + shard, and so do batches below
    ``_SP_EMBED_RS_MIN_TOKENS`` (decode-sized: there the all-reduce is the
    one-shot custom kernel, whose latency a ring reduce-scatter does not beat)."""
    if not (
        input_ids.shape[0] >= _SP_EMBED_RS_MIN_TOKENS
        and isinstance(embed, VocabParallelEmbedding)
        and embed.use_fused_embedding
        and embed.reduce_results
        and embed.parallel_group is None
        and embed.weight.dtype in (torch.bfloat16, torch.float16)
        and input_ids.ndim == 1
    ):
        return sp_shard(embed(input_ids))
    shards = embed.shard_indices
    partial = ops.vocab_parallel_embedding(
        input_ids,
        embed.weight,
        shards.org_vocab_start_index,
        shards.org_vocab_end_index,
        shards.num_org_vocab_padding,
        shards.added_vocab_start_index,
        shards.added_vocab_end_index,
    )
    return sp_reduce_scatter(partial)


def mega_moe_max_num_tokens(
    max_num_batched_tokens: int, tp_size: int, token_sharded: bool
) -> int:
    """Most tokens one rank's MegaMoE layer sees in a forward.

    Under sequence parallelism each rank's MoE gets only its 1/TP token
    shard (sp_shard pads to a multiple of the TP size), so
    the symmetric dispatch / combine buffer needs ceil(MNBT / TP) tokens per
    rank, not MNBT. Its combine slots alone are top_k x tokens x latent x 2 B
    (1.9 GB per rank at MNBT 16384). Divide exactly once: dividing again sized
    the buffer for ceil(MNBT / TP^2) and rejected full batches.
    """
    if token_sharded:
        return cdiv(max_num_batched_tokens, tp_size)
    return max_num_batched_tokens


def mega_moe_experts_cls(vllm_config: VllmConfig) -> type[nn.Module]:
    """The routed-experts module of the native MegaMoE backend: DeepGEMM MegaMoE, or
    on ``kimi_k3_mega_moe`` the same interface on the Rubin (SM107) kernels (raises
    off SM107), with its own MXFP8 x MXFP4 / exact NVFP4 policy per tokens per
    rank."""
    if vllm_config.kernel_config.moe_backend == KIMI_K3_MEGA_MOE:
        from .megamoe_rubin.experts import KimiK3MegaMoEExperts

        return KimiK3MegaMoEExperts
    return DeepGemmMegaMoEExperts


def _rubin_physical_experts(
    vllm_config: VllmConfig, num_experts: int, ep_size: int
) -> int:
    """Physical routed experts per layer on the Rubin MegaMoE backend: the logical
    count plus redundant slots: ``eplb_config.num_redundant_experts`` under EPLB,
    else just enough to make the count divisible by the EP size (e.g. 896 + 4 at
    EP12). Config-only, so every rank agrees. Other backends keep the logical
    count."""
    if vllm_config.kernel_config.moe_backend != KIMI_K3_MEGA_MOE:
        return num_experts
    parallel_config = vllm_config.parallel_config
    if parallel_config.enable_eplb:
        return num_experts + parallel_config.eplb_config.num_redundant_experts
    return num_experts + (-num_experts) % ep_size


def maybe_init_gemm_rs_ar(vllm_config: VllmConfig, use_sequence_parallel: bool) -> bool:
    """Bind the row-parallel projections (attention o_proj, MLP / shared-expert
    down_proj) to the fused GEMM + multicast reduction where the shared kernel can
    run (its own capability checks): GEMM-RS under sequence parallelism, GEMM-AR
    otherwise. The worker's static SP topology binds the singleton to one mode.

    Both modes are on by policy (no switch): the fused kernel computes the same
    BF16 GEMM and reduction (summation order over the TP ranks may differ, as for
    the stock NCCL collectives). GEMM-RS replaced the separate GEMM + NCCL ring
    reduce-scatter on the 40-layer TP4 SP prefill step: -4.4% at 17.4K tokens,
    -3.5% at 8.2K tokens.

    The one exception is the SP topology of the specialized decode: its stock
    (eager / piecewise) steps keep the unfused reduce-scatter o_proj its SP step
    reproduces (``specialized_decode.runtime.sp_unsupported``).
    """
    all_reduce = not use_sequence_parallel
    if not all_reduce and specialized_decode_config(vllm_config) is not None:
        return False

    # The kernel module pulls in cute_dsl, so import it only when used.
    from vllm.model_executor.kernels.linear.cute_dsl.gemm_rs_ar import (
        maybe_init_gemm_rs_ar as maybe_init_shared_gemm_rs_ar,
    )

    config = vllm_config.model_config.hf_text_config
    return maybe_init_shared_gemm_rs_ar(
        vllm_config, N=config.hidden_size, all_reduce=all_reduce
    )


def dsa_runs_gemm_rs_ar(vllm_config: VllmConfig, run_gemm_rs_ar: bool) -> bool:
    """Whether the DSA o_proj binds the fused GEMM-RS/AR (prefill and stock
    serving). A specialized-decode deployment keeps the DSA o_proj unfused: its
    DSA layers reproduce the stock GEMM + all-reduce o_proj (TP16 non-SP C256
    decode), so the binding must not change that role's math or timing."""
    return run_gemm_rs_ar and specialized_decode_config(vllm_config) is None


def _gemm_ar_owns(t: torch.Tensor) -> bool:
    """Whether ``t`` is a view of the GEMM-AR workspace: the GEMM-AR o_proj hands
    its reduced output back as such a view (no copy; bitwise),
    and a layer that keeps it past the next GEMM-AR copies it."""
    from vllm.model_executor.kernels.linear.cute_dsl import gemm_rs_ar

    op = gemm_rs_ar._gemm_rs_ar
    return op is not None and op.all_reduce and op.owns(t)


class KimiVecSituAndMul(nn.Module):
    """SituAndMul through the vectorized kernel in ops/situ_and_mul.py (bitwise
    equal to the CUDA op for every input), for every SiTU MLP.

    For small inputs (up to 2048 rows of the TP8 shared expert's 768 columns,
    the CUDA-graph range) the CUDA op's one-block-per-row layout keeps more
    threads busy and wins (128-token steps: +7 us per call with the vectorized
    kernel); larger ones (eager prefill steps, and hybrid-SP steps' ~2k
    rows x 6144 columns of the replicated shared expert: 34.3 -> 24.7 us) take
    the vectorized kernel. The small inputs (every decode step) run the stock
    SituAndMul module itself, through its own dispatch; the vectorized kernel is
    bitwise its CUDA op (forward_cuda). Where the custom op dispatches to
    forward_native instead, the large inputs keep the CUDA op's bits (the same FP32
    math, one BF16 rounding).

    Prefill chunks (from ``_FAST_MIN_ROWS`` rows; the dense layer is 16K x 33792
    at F16K) use the routed experts' MUFU-light SiTU form, as the shared experts'
    MXFP8 SiTU does at prefill (``mxfp8_act.SITU_FAST_MIN_ROWS``): the
    precise-tanhf kernel is FMA-issue bound at ~5x its HBM floor there, and the
    fast form stays within one BF16 ulp of it."""

    _VEC_MIN_OUT_ELEMS = 2048 * 768 + 1
    _FAST_MIN_ROWS = 4096

    def __init__(self, beta: float, linear_beta: float | None):
        super().__init__()
        self.beta = float(beta)
        self.linear_beta = None if linear_beta is None else float(linear_beta)
        self.small = SituAndMul(beta=self.beta, linear_beta=self.linear_beta)
        from vllm.models.kimi_k3.nvidia.ops import mxfp8_act
        from vllm.models.kimi_k3.nvidia.ops.situ_and_mul import warmup

        warmup()  # build / load the JIT extensions at model init, not mid-step
        mxfp8_act.warmup()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() // 2 < self._VEC_MIN_OUT_ELEMS:
            return self.small(x)
        if x.numel() // x.shape[-1] >= self._FAST_MIN_ROWS:
            from vllm.models.kimi_k3.nvidia.ops import mxfp8_act

            if mxfp8_act.situ_fast_bf16_supported(x.shape[-1] // 2):
                return mxfp8_act.situ_and_mul_fast_bf16(
                    x, self.beta, self.linear_beta
                )
        from vllm.models.kimi_k3.nvidia.ops.situ_and_mul import situ_and_mul

        return situ_and_mul(x, self.beta, self.linear_beta)


class KimiMLP(nn.Module):
    """Dense / shared-expert MLP, optionally TP-sharded under sequence parallel.

    Under sequence parallelism each rank owns a distinct slice of the tokens, so
    by default both projections are replicated (``disable_tp``) and the block
    needs no collective. That makes every rank stream the entire weight to serve
    its own token shard.

    With ``VLLM_KIMI_K3_SHARD_SP_SHARED_EXPERT`` the weights are TP-sharded
    instead. A rank then holds only a slice of the intermediate dim, so it
    cannot finish its own tokens alone: ``forward`` all-gathers the full token
    set, computes this rank's partial, and reduce-scatters. The reduce-scatter
    sums across TP and restores the sequence sharding in one collective, so the
    block still ends with one collective per direction.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        use_sequence_parallel: bool = False,
        can_shard_sequence_parallel: bool = False,
        run_gemm_rs_ar: bool = False,
        prefix: str = "",
        activation_situ_beta: float | None = None,
        activation_situ_linear_beta: float | None = None,
    ) -> None:
        super().__init__()

        self.shard_sequence_parallel = shard_sequence_parallel_mlp(
            hidden_size,
            intermediate_size,
            use_sequence_parallel,
            can_shard_sequence_parallel,
        )
        replicate = use_sequence_parallel and not self.shard_sequence_parallel

        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            disable_tp=replicate,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            # Sharded sequence parallel reduces via the reduce-scatter in
            # forward(), which also restores the sequence sharding.
            reduce_results=False if self.shard_sequence_parallel else reduce_results,
            disable_tp=replicate,
            prefix=f"{prefix}.down_proj",
        )
        self.gemm_rs_ar = None
        # RS requires sequence sharding; AR operates on replicated tokens.
        use_gemm_rs_ar = self.shard_sequence_parallel or (
            not use_sequence_parallel and reduce_results
        )
        if use_gemm_rs_ar and run_gemm_rs_ar:
            from vllm.model_executor.kernels.linear.cute_dsl.gemm_rs_ar import (
                get_gemm_rs_ar,
            )

            gemm_rs_ar = get_gemm_rs_ar()
            if gemm_rs_ar.can_run(self.down_proj):
                self.gemm_rs_ar = gemm_rs_ar
            else:
                gemm_rs_ar.warn_incompatible_projection()
        if hidden_act == "silu":
            self.act_fn = SiluAndMul()
        elif hidden_act == "situ":
            self.act_fn = KimiVecSituAndMul(
                beta=activation_situ_beta or 1.0,
                linear_beta=activation_situ_linear_beta,
            )
        else:
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu and situ are supported."
            )

    # specialized_prefill.shared_situ.PrefillSharedSitu (prefill workers only).
    _prefill_situ = None
    # gate_up_proj's weight rows are in the Rubin MegaMoE fused shared expert's 16-row
    # gate/up stripes (specialized_decode.runtime.sp_mlp.stripe_shared_gate_up: its
    # fused FC1 views this weight); its GEMM output is de-striped here.
    gate_up_striped = False

    def _gate_up(self, x) -> torch.Tensor:
        """``[gate | up]`` = gate_up_proj(x)."""
        gate_up, _ = self.gate_up_proj(x)
        if self.gate_up_striped:
            from .megamoe_rubin.weights import deinterleave_gate_up_cols_16

            gate_up = deinterleave_gate_up_cols_16(gate_up)
        return gate_up

    def mxfp8_input(self) -> bool:
        """gate_up_proj takes an MXFP8 activation (QuantizedActivation)."""
        return not self.shard_sequence_parallel and takes_mxfp8_activation(
            self.gate_up_proj
        )

    def _act_mxfp8_ok(self) -> bool:
        """The activation kernel can write down_proj's MXFP8 input."""
        fused = getattr(self, "_fused_act_quant", None)
        if fused is None:
            fused = (
                self.gemm_rs_ar is None
                and not self.shard_sequence_parallel
                and isinstance(self.act_fn, (SiluAndMul, SituAndMul, KimiVecSituAndMul))
                and takes_mxfp8_activation(self.down_proj)
            )
            self._fused_act_quant = fused
        return fused

    def act_mxfp8_into(self, gate_up: torch.Tensor, out, col: int = 0) -> None:
        """Write ``act_fn(gate_up)`` as MXFP8 into columns ``[col, col + d)``
        of ``out`` (an ``mxfp8_act.Mxfp8Buffer``)."""
        if isinstance(self.act_fn, SiluAndMul):
            mxfp8_act.silu_and_mul_quantize(gate_up, out=out, col=col)
        else:
            mxfp8_act.situ_and_mul_quantize_auto(
                gate_up, self.act_fn.beta, self.act_fn.linear_beta, out=out, col=col
            )

    def _act_mxfp8(self, gate_up: torch.Tensor):
        """The activation written as down_proj's MXFP8 input, or None."""
        if not self._act_mxfp8_ok():
            return None
        if isinstance(self.act_fn, SiluAndMul):
            return mxfp8_act.silu_and_mul_quantize(gate_up)
        return mxfp8_act.situ_and_mul_quantize_auto(
            gate_up, self.act_fn.beta, self.act_fn.linear_beta
        )

    def resid_capable(self) -> bool:
        """``forward_resid`` can run: the MXFP8 activation and an
        attn_res_resid GEMM for down_proj (prefill AttnRes dataflow C)."""
        return (
            not self.shard_sequence_parallel
            and self._act_mxfp8_ok()
            and prefill_attn_res_resid.supported(self.down_proj)
        )

    def forward_resid(self, x, prefix: torch.Tensor) -> torch.Tensor:
        """``prefix <- bf16(prefix + forward(x))`` in place, the down_proj
        adding its BF16 output into ``prefix`` in its GEMM store
        (specialized_prefill.attn_res_resid). Returns ``prefix``."""
        situ = self._prefill_situ
        qa = situ.fused(x) if situ is not None else None
        if qa is None:
            gate_up = self._gate_up(x)
            if situ is not None:
                gate_up = situ.deinterleave(gate_up)
            qa = self._act_mxfp8(gate_up)
        gemm = prefill_attn_res_resid.gemm_for(self.down_proj)
        if qa is None or gemm is None:
            raise RuntimeError("shared expert: no MXFP8 down_proj input for the AttnRes prefix")
        return gemm.add_into(qa, prefix)

    def forward(self, x):
        if self.shard_sequence_parallel:
            # Each rank holds a weight shard but only its own tokens, so it
            # cannot finish those tokens alone: gather the full token set,
            # compute this rank's partial for all of them, then reduce-scatter,
            # which sums across TP and restores the sequence sharding.
            x = sp_all_gather(x)
        situ = self._prefill_situ
        if situ is not None:
            # Prefill shared expert: the gate_up GEMM writes the MXFP8 SiTU
            # (specialized_prefill.shared_situ) at prefill-sized batches.
            qa = situ.fused(x)
            if qa is not None:
                return self.down_proj(qa)[0]
        gate_up = self._gate_up(x)
        if situ is not None:
            gate_up = situ.deinterleave(gate_up)
        # Online-MXFP8 down_proj: the activation kernel writes its MXFP8 input.
        qa = self._act_mxfp8(gate_up)
        if qa is not None:
            return self.down_proj(qa)[0]
        x = self.act_fn(gate_up)

        if self.gemm_rs_ar is not None and self.gemm_rs_ar.should_run(x):
            return self.gemm_rs_ar.apply(x, self.down_proj)

        x, _ = self.down_proj(x)
        if self.shard_sequence_parallel:
            x = sp_reduce_scatter(x)
        return x


class KimiRoutedOutputTransform(nn.Module):
    def __init__(
        self,
        norm: RMSNorm | None,
        up_proj: ReplicatedLinear,
    ) -> None:
        super().__init__()
        self.norm = norm
        self.up_proj = up_proj
        self._fused_norm_quant: bool | None = None

    def _norm_mxfp8(self) -> bool:
        if self._fused_norm_quant is None:
            norm = self.norm
            self._fused_norm_quant = (
                norm is not None
                and getattr(norm, "has_weight", True)
                and norm.variance_size_override is None
                and norm.weight.shape[0] % 256 == 0
                and norm.weight.shape[0] <= 8192
                and takes_mxfp8_activation(self.up_proj)
            )
        return self._fused_norm_quant

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Project the routed latent back to the hidden dim.

        Args:
            hidden_states: Routed expert output in latent space.
            residual: Optional tensor of the up-projection's output shape to
                accumulate into. Unquantized projections fuse the addition
                into the GEMM epilogue.

        """
        if self.norm is not None:
            if self._norm_mxfp8():
                # Online-MXFP8 up_proj: the latent norm writes its MXFP8 input.
                qa = mxfp8_act.rms_norm_quantize(
                    hidden_states, self.norm.weight, self.norm.variance_epsilon
                )
                out, _ = self.up_proj(qa)
                return out if residual is None else residual.add_(out)
            hidden_states = self.norm(hidden_states)
        if residual is not None and isinstance(
            self.up_proj.quant_method, UnquantizedLinearMethod
        ):
            return residual.addmm_(hidden_states, self.up_proj.weight.t())
        hidden_states, _ = self.up_proj(hidden_states)
        if residual is not None:
            return residual.add_(hidden_states)
        return hidden_states


def kimi_k3_mega_moe_weight_format(
    quant_config: QuantizationConfig | None, prefix: str
) -> MegaMoEWeightFormat:
    """Routed-expert checkpoint format of the MoE block at prefix.

    ModelOpt checkpoints name it per layer (MIXED_PRECISION quantized_layers
    entry of {prefix}.experts, or a whole-model NVFP4 export): NVFP4 runs the
    native NVFP4 MegaMoE path. Every other checkpoint is the canonical MXFP4 one.
    Raises for ModelOpt expert formats MegaMoE cannot load.
    """
    if isinstance(quant_config, ModelOptMixedPrecisionConfig):
        algo = quant_config._resolve_quant_algo(f"{prefix}.experts")
    elif isinstance(quant_config, ModelOptNvFp4Config):
        algo = quant_config.quant_method.upper()
    else:
        return MEGA_MOE_MXFP4
    if algo == "NVFP4":
        return MEGA_MOE_NVFP4
    raise NotImplementedError(
        f"{prefix}.experts: MegaMoE has no path for ModelOpt {algo} experts."
    )


# Checkpoint suffix -> parameter suffix of the MegaMoE experts, per format.
_MEGA_MOE_EXPERT_SUFFIXES = {
    MEGA_MOE_MXFP4.name: (("weight_packed", "weight"), ("weight_scale", "weight_scale")),
    MEGA_MOE_NVFP4.name: (
        ("weight_scale_2", "weight_scale_2"),
        ("weight_scale", "weight_scale"),
        ("input_scale", "input_scale"),
        ("weight", "weight"),
    ),
}


def make_kimi_k3_mega_moe_expert_params_mapping(
    num_experts: int,
    weight_format: MegaMoEWeightFormat = MEGA_MOE_MXFP4,
) -> list[tuple[str, str, int, str]]:
    suffixes = _MEGA_MOE_EXPERT_SUFFIXES.get(weight_format.name)
    if suffixes is None:
        raise NotImplementedError(
            f"No Kimi K3 MegaMoE expert mapping for {weight_format.name!r} weights."
        )
    mapping = []
    for expert_id in range(num_experts):
        for shard_id in ("w1", "w2", "w3"):
            param_prefix = "w13" if shard_id in ("w1", "w3") else "w2"
            for suffix, param_suffix in suffixes:
                mapping.append(
                    (
                        f"experts.{param_prefix}_{param_suffix}",
                        f"experts.{expert_id}.{shard_id}.{suffix}",
                        expert_id,
                        shard_id,
                    )
                )
    return mapping


class KimiMoE(nn.Module):
    def __init__(
        self,
        config: KimiLinearConfig,
        vllm_config: VllmConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        layer_idx: int = 0,
        use_sequence_parallel: bool = False,
        run_gemm_rs_ar: bool = False,
    ):
        super().__init__()
        hidden_size = config.hidden_size
        moe_intermediate_size = config.moe_intermediate_size
        num_experts = config.num_experts
        num_experts_per_token = config.num_experts_per_token
        assert moe_intermediate_size is not None
        assert num_experts is not None
        assert num_experts_per_token is not None
        moe_renormalize = config.moe_renormalize
        routed_expert_hidden_size = config.routed_expert_hidden_size
        self.use_latent_moe = routed_expert_hidden_size is not None
        self.moe_hidden_size = (
            routed_expert_hidden_size
            if routed_expert_hidden_size is not None
            else hidden_size
        )
        self.latent_moe_use_norm = config.latent_moe_use_norm
        self.tp_size = get_tensor_model_parallel_world_size()
        self.use_sequence_parallel = use_sequence_parallel
        self.routed_scaling_factor = config.routed_scaling_factor
        self.moe_renormalize = moe_renormalize
        self.use_grouped_topk = config.use_grouped_topk
        self.num_expert_group = config.num_expert_group
        self.topk_group = config.topk_group
        self.moe_router_activation_func = config.moe_router_activation_func
        self.num_shared_experts = config.num_shared_experts
        self.layer_idx = layer_idx
        # Native MegaMoE (DeepGEMM, or Rubin on SM107): the router, latent
        # projections and shared expert run here, the experts in one launch.
        self.use_mega_moe = (
            vllm_config.kernel_config.moe_backend in NATIVE_MEGA_MOE_BACKENDS
        )
        # Prefill-worker MegaMoE policy (specialized_prefill.moe); None: stock.
        self._prefill_moe = None
        self._prefill_subcalls = None
        if self.use_mega_moe and not vllm_config.parallel_config.enable_expert_parallel:
            raise NotImplementedError(
                "Kimi K3 MegaMoE requires expert parallel. Enable it with "
                "--enable-expert-parallel."
            )
        if self.use_mega_moe and config.hidden_act != "situ":
            raise ValueError("Kimi K3 MegaMoE requires SITU activation.")
        if self.use_mega_moe and not self.use_latent_moe:
            raise ValueError("Kimi K3 MegaMoE requires latent MoE projections.")
        if self.use_mega_moe and not self.use_grouped_topk:
            raise ValueError("Kimi K3 MegaMoE requires grouped top-k routing.")
        if self.use_mega_moe and (self.num_expert_group != 1 or self.topk_group != 1):
            raise NotImplementedError(
                "Kimi K3 MegaMoE currently requires one expert group."
            )
        self.padded_moe_intermediate_size = moe_intermediate_size
        min_moe_intermediate_per_partition = getattr(
            config, "min_moe_intermediate_per_partition", 256
        )
        if self.tp_size > 1 and not vllm_config.parallel_config.enable_expert_parallel:
            moe_intermediate_per_partition = moe_intermediate_size // self.tp_size
            if moe_intermediate_per_partition < min_moe_intermediate_per_partition:
                self.padded_moe_intermediate_size = (
                    min_moe_intermediate_per_partition * self.tp_size
                )
        activation_situ_beta = (
            config.activation_situ_beta if config.hidden_act == "situ" else None
        )
        activation_situ_linear_beta = (
            config.activation_situ_linear_beta if config.hidden_act == "situ" else None
        )

        # Route with fp32 logits for numerically stable expert selection.
        self.gate = GateLinear(
            input_size=hidden_size,
            output_size=num_experts,
            bias=False,
            out_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )

        self.gate.e_score_correction_bias = nn.Parameter(
            torch.empty(num_experts, dtype=torch.float32)
        )

        if self.num_shared_experts is not None:
            shared_intermediate_size = moe_intermediate_size * self.num_shared_experts
            self.shared_experts = KimiMLP(
                hidden_size=config.hidden_size,
                intermediate_size=shared_intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                use_sequence_parallel=use_sequence_parallel,
                # MegaMoE calls the shared expert directly. DeepEP v2 hands it
                # to MoERunner, but preserves the local SP output layout needed
                # by the sharded shared-expert path.
                can_shard_sequence_parallel=can_shard_sequence_parallel_shared_expert(
                    vllm_config
                ),
                run_gemm_rs_ar=run_gemm_rs_ar,
                prefix=f"{prefix}.shared_experts",
                activation_situ_beta=activation_situ_beta,
                activation_situ_linear_beta=activation_situ_linear_beta,
            )
        else:
            self.shared_experts = None

        self.routed_expert_down_proj: ReplicatedLinear | None
        self.routed_expert_norm: RMSNorm | None
        self.routed_expert_up_proj: ReplicatedLinear | None
        self.routed_output_transform: KimiRoutedOutputTransform | None
        if self.use_latent_moe:
            self.routed_expert_down_proj = ReplicatedLinear(
                hidden_size,
                self.moe_hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.routed_expert_down_proj",
            )
            self.routed_expert_norm = (
                prefill_glue.KimiRowsRMSNorm(
                    self.moe_hidden_size, eps=config.rms_norm_eps
                )
                if self.latent_moe_use_norm
                else None
            )
            # Replicated up-proj: the full weight lives on every rank and
            # produces the full hidden dim locally. This lets LatentMoERunner
            # fuse the latent and shared reductions into a single all-reduce
            # (concat the two partials, reduce once), then run the up-proj and
            # shared add locally with no further collective.
            self.routed_expert_up_proj = ReplicatedLinear(
                self.moe_hidden_size,
                hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.routed_expert_up_proj",
            )

            self.routed_output_transform = KimiRoutedOutputTransform(
                self.routed_expert_norm, self.routed_expert_up_proj
            )
            # Auxiliary CUDA stream to overlap the router gate with the routed
            # down projection on decode-sized batches (gated by
            # _ROUTED_DOWN_PROJ_STREAM_TOKEN_THRESHOLD).
            self._down_proj_stream: torch.cuda.Stream | None = aux_stream()
            self._down_proj_events = (torch.cuda.Event(), torch.cuda.Event())
            # Shared experts beside the routing front of prefill chunks.
            self._prefill_shared_side_stream = torch.cuda.Stream()
            self._prefill_shared_events = (torch.cuda.Event(), torch.cuda.Event())
        else:
            self.routed_expert_down_proj = None
            self.routed_expert_norm = None
            self.routed_expert_up_proj = None
            self.routed_output_transform = None

        if self.use_mega_moe:
            ep_group = get_ep_group()
            ep_size = ep_group.world_size
            ep_rank = ep_group.rank_in_group
            num_physical_experts = _rubin_physical_experts(
                vllm_config, num_experts, ep_size
            )
            if num_physical_experts % ep_size != 0:
                raise ValueError(
                    f"Kimi K3 num_experts={num_experts} must be divisible by "
                    f"EP size {ep_size}."
                )
            num_local_experts = num_physical_experts // ep_size
            max_num_tokens = mega_moe_max_num_tokens(
                vllm_config.scheduler_config.max_num_batched_tokens,
                self.tp_size,
                token_sharded=use_sequence_parallel,
            )
            from .specialized_prefill.moe import prefill_moe_policy

            self._prefill_moe = prefill_moe_policy(
                vllm_config,
                tp_size=self.tp_size,
                sequence_parallel=use_sequence_parallel,
                max_num_tokens=max_num_tokens,
            )
            self.experts = mega_moe_experts_cls(vllm_config)(
                vllm_config,
                max_num_tokens=(
                    max_num_tokens
                    if self._prefill_moe is None
                    else self._prefill_moe.capacity
                ),
                num_experts=num_physical_experts,
                num_logical_experts=num_experts,
                num_local_experts=num_local_experts,
                experts_start_idx=ep_rank * num_local_experts,
                top_k=num_experts_per_token,
                hidden_size=self.moe_hidden_size,
                intermediate_size=self.padded_moe_intermediate_size,
                activation=MegaMoEActivation(
                    "situ",
                    alpha=activation_situ_beta or 1.0,
                    beta=activation_situ_linear_beta or 0.0,
                ),
                weight_format=kimi_k3_mega_moe_weight_format(quant_config, prefix),
                prefix=f"{prefix}.experts",
            )
        else:
            self.experts = FusedMoEFactory(
                shared_experts=self.shared_experts,
                num_experts=num_experts,
                top_k=num_experts_per_token,
                hidden_size=self.moe_hidden_size,
                intermediate_size=self.padded_moe_intermediate_size,
                activation=config.hidden_act,
                activation_situ_beta=activation_situ_beta,
                activation_situ_linear_beta=activation_situ_linear_beta,
                renormalize=moe_renormalize,
                quant_config=quant_config,
                use_grouped_topk=config.use_grouped_topk,
                num_expert_group=config.num_expert_group,
                topk_group=config.topk_group,
                prefix=f"{prefix}.experts",
                scoring_func=config.moe_router_activation_func,
                e_score_correction_bias=self.gate.e_score_correction_bias,
                routed_scaling_factor=self.routed_scaling_factor,
                # Down projection runs outside MoERunner so it can overlap the
                # router gate on the aux stream (see forward()); the original
                # hidden states are passed to forward() as shared_experts_input
                # so shared experts still see the untransformed input.
                routed_input_transform=None,
                routed_output_transform=self.routed_output_transform,
                is_sequence_parallel=use_sequence_parallel,
                skip_padding=True,
                runner_cls=LatentMoERunner if self.use_latent_moe else None,
            )
        # Sequence-parallel Rubin MegaMoE: the replicated shared expert (no
        # collective) runs on a side stream concurrently with the MegaMoE launch,
        # whose persistent grid is capped to leave it SMs; joined before the
        # up-projection adds it. Same kernels and math, only concurrent.
        self._shared_overlap = (
            self.use_mega_moe
            and use_sequence_parallel
            and vllm_config.kernel_config.moe_backend == KIMI_K3_MEGA_MOE
            and isinstance(self.shared_experts, KimiMLP)
            and not self.shared_experts.shard_sequence_parallel
        )
        if self._shared_overlap:
            if prefill_full_grid_shared_overlap(vllm_config):
                self.experts.full_grid_beside_shared_expert()
            else:
                self.experts.reserve_sms_for_shared_expert()
            self._shared_stream = aux_stream()
            self._shared_events = (torch.cuda.Event(), torch.cuda.Event())
            # Full-grid buckets: the routing chain runs on a high-priority stream
            # and the shared expert on this one (see _overlapped_forward).
            self._routed_stream = routed_priority_stream()
        if self._prefill_moe is not None and self._prefill_moe.subcalls:
            from .specialized_prefill.moe import PrefillSubcallExperts

            self._prefill_subcalls = PrefillSubcallExperts(
                self.experts,
                self._prefill_moe.capacity,
                vllm_config.parallel_config.data_parallel_size,
            )
        if self.padded_moe_intermediate_size != moe_intermediate_size:
            w13_weight = getattr(self.experts, "w13_weight", None)
            if w13_weight is None:
                w13_weight = getattr(self.experts, "w13_weight_packed", None)
            w2_weight = getattr(self.experts, "w2_weight", None)
            if w2_weight is None:
                w2_weight = getattr(self.experts, "w2_weight_packed", None)
            if w13_weight is not None:
                w13_weight.data.zero_()
            if w2_weight is not None:
                w2_weight.data.zero_()
            self.experts.moe_config.intermediate_size_per_partition_unpadded = (
                moe_intermediate_size // self.tp_size
            )

    def _shared_mxfp8_input(self) -> bool:
        """The latent down_proj and the shared experts' gate_up_proj both read
        the layer input as online MXFP8: quantize it once for both."""
        fused = getattr(self, "_fused_input_quant", None)
        if fused is None:
            shared = self.shared_experts
            fused = (
                self.use_mega_moe
                and self.routed_expert_down_proj is not None
                and isinstance(shared, KimiMLP)
                and shared.mxfp8_input()
                and takes_mxfp8_activation(self.routed_expert_down_proj)
            )
            self._fused_input_quant = fused
        return fused

    def pack_projections(self) -> None:
        """TP1 latent MegaMoE with online-MXFP8 shared experts (wide-EP DEP):
        pack the four dense MoE GEMMs into two (wep-proj-gemm-p2).

        * along N: ``[routed latent down ; shared gate_up]`` on the layer
          input (one MXFP8 quantization, one weight stream);
        * along K: ``[shared down | routed latent up]`` on
          ``[act(shared gate_up) | norm(MegaMoE output)]``, which is the layer
          output ``shared + routed`` in one GEMM (FP32 accumulation; replaces
          the up projection's BF16 beta-add of the shared output)."""
        self._packed_moe = None
        shared = self.shared_experts
        transform = self.routed_output_transform
        if (
            not self.use_mega_moe
            or self.tp_size != 1
            or transform is None
            or self.routed_expert_down_proj is None
            or not isinstance(shared, KimiMLP)
            or not shared.mxfp8_input()
            or shared._act_mxfp8_ok() is False
            or not transform._norm_mxfp8()
            or not takes_mxfp8_activation(
                self.routed_expert_down_proj, shared.down_proj
            )
        ):
            return
        down, gate_up = self.routed_expert_down_proj, shared.gate_up_proj
        if not (can_pack_mxfp8([down, gate_up])
                and can_pack_mxfp8([shared.down_proj, transform.up_proj],
                                   along_k=True)):  # fmt: skip
            return
        latent = down.output_size_per_partition
        inter = shared.down_proj.input_size_per_partition
        self._packed_moe = (
            PackedMxfp8Linear([down, gate_up]),
            PackedMxfp8Linear([shared.down_proj, transform.up_proj], along_k=True),
            latent,
            inter,
        )

    def _grouped_topk(
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """MegaMoE routing: ``fused_grouped_topk``; prefill-sized batches run
        its bitwise twin ``prefill_glue.router_topk`` (one block per token)."""
        bias = self.gate.e_score_correction_bias.data
        if prefill_glue.use_router_topk(
            router_logits,
            bias,
            self.experts.top_k,
            self.num_expert_group,
            self.topk_group,
            self.moe_router_activation_func,
        ):
            return prefill_glue.router_topk(
                router_logits,
                bias,
                self.experts.top_k,
                self.moe_renormalize,
                self.routed_scaling_factor,
            )
        return fused_grouped_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=self.experts.top_k,
            renormalize=self.moe_renormalize,
            e_score_correction_bias=bias,
            num_expert_group=self.num_expert_group,
            topk_group=self.topk_group,
            scoring_func=self.moe_router_activation_func,
            routed_scaling_factor=self.routed_scaling_factor,
        )

    def _router(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        router_logits, _ = self.gate(hidden_states)
        if not self.use_mega_moe:
            return router_logits, None
        return self._grouped_topk(hidden_states, router_logits)

    def _routed_experts(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """MegaMoE routed experts; prefill workers split the batch into
        EP-uniform capacity sub-calls (specialized_prefill.moe)."""
        subcalls = self._prefill_subcalls
        if subcalls is None:
            return self.experts(hidden_states, topk_weights, topk_ids)
        return subcalls(hidden_states, topk_weights, topk_ids)

    def _packed_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """MoE layer on the packed GEMMs of pack_projections()."""
        pack_n, pack_k, latent, inter = self._packed_moe
        num_tokens = hidden_states.shape[0]
        x = mxfp8_act.quantize(hidden_states)
        (router_output, topk_ids), y = maybe_execute_in_parallel(
            lambda: self._router(hidden_states),
            lambda: pack_n(x),
            self._down_proj_events[0],
            self._down_proj_events[1],
            self._down_proj_stream
            if num_tokens <= _ROUTED_DOWN_PROJ_STREAM_TOKEN_THRESHOLD
            else None,
        )
        # [act(shared gate_up) | norm(routed latent out)]: the packed down/up
        # GEMM input; the shared half is written before MegaMoE runs.
        act = mxfp8_act.Mxfp8Buffer(
            num_tokens, inter + latent, hidden_states.device, hidden_states.dtype
        )
        self.shared_experts.act_mxfp8_into(y[:, latent:], act, 0)
        routed = self._routed_experts(y[:, :latent], router_output, topk_ids)
        norm = self.routed_output_transform.norm
        mxfp8_act.rms_norm_quantize(
            routed, norm.weight, norm.variance_epsilon, out=act, col=inter
        )
        return pack_k(act.activation())

    def _overlapped_forward(
        self,
        hidden_states: torch.Tensor,
        mxfp8_input: "QuantizedActivation | None",
        shared_input: "torch.Tensor | QuantizedActivation",
    ) -> torch.Tensor:
        """Sequence-parallel MegaMoE (``_shared_overlap``) with the replicated
        shared expert concurrent with the routed experts; same kernels, operands
        and up-projection epilogue as the serial order, only concurrent.

        * SM-capped bucket (``experts.shared_overlap``: decode, unified serving):
          the MegaMoE grid leaves the shared expert its SMs; the shared expert runs
          on the side stream from the dispatch on.
        * Full-grid bucket (P/D prefill workers,
          ``prefill_full_grid_shared_overlap``): MegaMoE needs every SM, so
          the whole routing chain (router, top-k, latent down, dispatch, MegaMoE,
          reduce) runs on a high-priority stream and the shared expert on the
          current stream from the layer input on: the block scheduler hands each
          SM freed by the chain to its next kernel first, so the shared GEMMs take
          only what the chain leaves (router/top-k/dispatch gaps) and the MegaMoE
          grid becomes resident as the shared CTAs retire (the shared kernels never
          wait on MegaMoE, so its spin-waits cannot block on them).

        Both forms fork and join inside this call (no eager break between), so
        they are also legal inside breakable CUDA graph segments.

        Raises if the shared expert would run a persistent or collective kernel
        (GEMM-RS/AR, the fused SiTU GEMM) beside the persistent MegaMoE grid, or if
        an eager break split a breakable capture segment between fork and join."""
        shared = self.shared_experts
        if shared.gemm_rs_ar is not None or shared._prefill_situ is not None:
            raise RuntimeError(
                "KimiMoE shared-expert overlap: the shared expert runs a persistent "
                "kernel (GEMM-RS/AR or fused SiTU) that may not share SMs with the "
                "persistent MegaMoE grid."
            )
        from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture

        capture = BreakableCUDAGraphCapture.current()
        breaks = None if capture is None else capture.num_eager_breaks
        start, done = self._shared_events
        if self.experts.shared_overlap(hidden_states.shape[0]):
            routed_hidden_states, router_output, topk_ids = (
                self._maybe_overlap_router_and_down_proj(hidden_states, mxfp8_input)
            )
            start.record()
            routed = self._routed_experts(routed_hidden_states, router_output, topk_ids)
            with torch.cuda.stream(self._shared_stream):
                start.wait()
                shared_output = self.shared_experts(shared_input)
                done.record()
        else:
            start.record()
            with torch.cuda.stream(self._routed_stream):
                start.wait()
                routed_hidden_states, router_output, topk_ids = (
                    self._maybe_overlap_router_and_down_proj(hidden_states, mxfp8_input)
                )
                routed = self._routed_experts(
                    routed_hidden_states, router_output, topk_ids
                )
                done.record()
            shared_output = self.shared_experts(shared_input)
        done.wait()
        if capture is not None and capture.num_eager_breaks != breaks:
            raise RuntimeError(
                "KimiMoE shared-expert overlap: an eager break inside the MoE layer "
                "split the breakable CUDA graph segment between fork and join."
            )
        assert self.routed_output_transform is not None
        return self.routed_output_transform(routed, residual=shared_output)

    def _prefill_shared_stream(self, num_tokens: int) -> torch.cuda.Stream | None:
        """The stream that overlaps the shared experts with the routing front
        of a MegaMoE prefill chunk (from _ROUTED_DOWN_PROJ_STREAM_PREFILL_TOKENS
        tokens), else None."""
        if (
            not self.use_mega_moe
            or self._shared_overlap
            or self.shared_experts is None
            or self.routed_expert_down_proj is None
            or num_tokens < _ROUTED_DOWN_PROJ_STREAM_PREFILL_TOKENS
        ):
            return None
        from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture

        if BreakableCUDAGraphCapture.is_active():
            return None
        return self._prefill_shared_side_stream

    def _maybe_overlap_router_and_down_proj(
        self,
        hidden_states: torch.Tensor,
        down_proj_input: "torch.Tensor | QuantizedActivation | None" = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Compute the routed-expert down projection alongside the router,
        overlapping them on separate CUDA streams when latent MoE is enabled.

        The router gate and the down projection both read ``hidden_states``, so
        the gate runs on the default stream and the down projection on the aux
        stream, joined via ``maybe_execute_in_parallel``. For MegaMoE the
        grouped top-k selection consumes only the gate logits, so it also runs
        on the default stream and overlaps the down projection.

        Returns:
            ``(routed_hidden_states, router_output, topk_ids)``.
            ``routed_hidden_states`` is the down-projected latent (or the
            original ``hidden_states`` when latent MoE is disabled). For MegaMoE
            ``router_output`` holds the grouped top-k weights and ``topk_ids``
            the selected experts; otherwise ``router_output`` holds the raw gate
            logits and ``topk_ids`` is ``None``.

        """

        _router = self._router

        down_proj = self.routed_expert_down_proj
        if down_proj is None:
            router_output, topk_ids = _router(hidden_states)
            return hidden_states, router_output, topk_ids
        num_tokens = hidden_states.shape[0]
        if down_proj_input is None:
            down_proj_input = hidden_states
        (router_output, topk_ids), (routed_hidden_states, _) = (
            maybe_execute_in_parallel(
                lambda: _router(hidden_states),
                lambda: down_proj(down_proj_input),
                self._down_proj_events[0],
                self._down_proj_events[1],
                self._down_proj_stream
                if num_tokens <= _ROUTED_DOWN_PROJ_STREAM_TOKEN_THRESHOLD
                or num_tokens >= _ROUTED_DOWN_PROJ_STREAM_PREFILL_TOKENS
                else None,
            )
        )
        return routed_hidden_states, router_output, topk_ids

    def takes_shared_mxfp8_input(self) -> bool:
        """forward() can take the shared experts' gate_up input as MXFP8
        from the producer of the layer input (``shared_mxfp8_input``): the
        MegaMoE path with online-MXFP8 shared experts that quantizes nothing
        else from the layer input (BF16 router and latent down_proj)."""
        return (
            self.use_mega_moe
            and getattr(self, "_packed_moe", None) is None
            and isinstance(self.shared_experts, KimiMLP)
            and self.shared_experts.mxfp8_input()
            and not self._shared_mxfp8_input()
        )

    def resid_capable(self) -> bool:
        """``forward(resid_prefix=...)`` can run (prefill AttnRes dataflow C):
        MegaMoE with the shared expert's down_proj on an attn_res_resid GEMM
        and the BF16 latent up-projection's beta-add epilogue."""
        cap = getattr(self, "_resid_capable", None)
        if cap is None:
            t = self.routed_output_transform
            cap = self._resid_capable = (
                self.use_mega_moe
                and getattr(self, "_packed_moe", None) is None
                and isinstance(self.shared_experts, KimiMLP)
                and self.shared_experts.resid_capable()
                and t is not None
                and not t._norm_mxfp8()
                and isinstance(t.up_proj.quant_method, UnquantizedLinearMethod)
            )
        return cap

    def forward(
        self,
        hidden_states: torch.Tensor,
        shared_mxfp8_input: QuantizedActivation | None = None,
        resid_prefix: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """resid_prefix (prefill AttnRes dataflow C): the AttnRes prefix the
        layer output is added into. The shared expert's down_proj adds its
        output into it in its GEMM store and the latent up-projection
        beta-adds the routed output (the stock P + (S + R) reassociated as
        (P + S) + R: the same roundings); returns the prefix."""
        num_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)
        if getattr(self, "_packed_moe", None) is not None:
            return self._packed_forward(hidden_states).view(num_tokens, hidden_size)
        # Online MXFP8: one quantization of the layer input for the latent
        # down_proj and the shared gate_up_proj (each would quantize its own).
        mxfp8_input = (
            mxfp8_act.quantize(hidden_states) if self._shared_mxfp8_input() else None
        )
        if shared_mxfp8_input is not None:
            assert mxfp8_input is None and self.use_mega_moe
        # The shared expert reads the un-projected hidden states, so it stays
        # outside the latent-space MegaMoE launch.
        shared_input = (
            shared_mxfp8_input
            if shared_mxfp8_input is not None
            else hidden_states
            if mxfp8_input is None
            else mxfp8_input
        )
        if self._shared_overlap and resid_prefix is None:
            return self._overlapped_forward(
                hidden_states, mxfp8_input, shared_input
            ).view(num_tokens, hidden_size)
        prefill_shared = self._prefill_shared_stream(num_tokens)
        if prefill_shared is not None:
            # The shared experts run beside the router, its top-k and the
            # latent down GEMM and join before MegaMoE (whose persistent grid
            # takes every SM): same kernels and math, only concurrent.
            start, done = self._prefill_shared_events
            start.record()
            with torch.cuda.stream(prefill_shared):
                start.wait()
                if resid_prefix is not None:
                    shared_output = self.shared_experts.forward_resid(
                        shared_input, resid_prefix
                    )
                else:
                    shared_output = self.shared_experts(shared_input)
                done.record()
        # Overlap the gate with the routed down projection; the returned hidden
        # states are already down-projected.
        routed_hidden_states, router_output, topk_ids = (
            self._maybe_overlap_router_and_down_proj(hidden_states, mxfp8_input)
        )
        if prefill_shared is not None:
            done.wait()
            assert self.routed_output_transform is not None
            final_hidden_states = self._routed_experts(
                routed_hidden_states, router_output, topk_ids
            )
            final_hidden_states = self.routed_output_transform(
                final_hidden_states, residual=shared_output
            )
        elif self.use_mega_moe and resid_prefix is not None:
            assert self.routed_output_transform is not None
            final_hidden_states = self._routed_experts(
                routed_hidden_states, router_output, topk_ids
            )
            shared_output = self.shared_experts.forward_resid(shared_input, resid_prefix)
            final_hidden_states = self.routed_output_transform(
                final_hidden_states, residual=shared_output
            )
        elif self.use_mega_moe:
            assert self.routed_output_transform is not None
            assert topk_ids is not None
            final_hidden_states = self._routed_experts(
                routed_hidden_states, router_output, topk_ids
            )
            shared_output = (
                self.shared_experts(shared_input)
                if self.shared_experts is not None
                else None
            )
            # The shared output is folded into the up-projection GEMM's beta-add
            # epilogue, so combining the two branches costs no extra kernel.
            final_hidden_states = self.routed_output_transform(
                final_hidden_states, residual=shared_output
            )
        else:
            # Routed experts consume the down-projected latent; shared experts
            # (inside MoERunner) get the original hidden states via
            # shared_experts_input.
            final_hidden_states = self.experts(
                hidden_states=routed_hidden_states,
                router_logits=router_output,
                shared_experts_input=hidden_states,
            )
        return final_hidden_states.view(num_tokens, hidden_size)


def warm_prefill_indexer(model: nn.Module) -> None:
    """Check at load, when the model has DSA layers and
    sparse_indexer_topk_backend="trtllm_gvr2", that DeepSelect ranks every
    prefill chunk and decode step (up to max_model_len columns): that backend's
    prefill and decode top-k are DeepSelect, the only exact and deterministic
    one (the 16-bit GVR V2 engines change cutoff ties and row order per call,
    BUG-DCP8-2 / BUG-STRESS-2). Raises instead of serving with a fallback; no
    GVR V2 engine is compiled. Under "trtllm_gvr2_nondet" the prefill top-k is
    the GVR V2 prefill engine instead: compile its engines at load, on every
    rank (they never compile under CUDA graph capture, and a mid-serving JIT on
    one DCP rank would stall its peers).

    Kimi K3's DSA indexer (KimiK3DSAIndexer) is not a SparseAttnIndexer, so
    both are matched by their topk_backend attribute."""
    if not any(isinstance(m, KimiK3DSAAttention) for m in model.modules()):
        return
    from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer

    from .dsa import KimiK3DSAIndexer

    indexer = next(
        (
            m
            for m in model.modules()
            if isinstance(m, (KimiK3DSAIndexer, SparseAttnIndexer))
            and m.topk_backend in ("trtllm_gvr2", "trtllm_gvr2_nondet")
        ),
        None,
    )
    if indexer is None:
        return
    from vllm.model_executor.layers.indexer_topk import (
        deep_select_prefill_unsupported,
        warmup_trtllm_gvr2_prefill,
    )

    if indexer.topk_backend == "trtllm_gvr2_nondet":
        warmup_trtllm_gvr2_prefill(indexer.topk_tokens, indexer.max_model_len)
        return

    why = deep_select_prefill_unsupported(indexer.topk_tokens, indexer.max_model_len)
    if why is not None:
        raise RuntimeError(
            "sparse_indexer_topk_backend='trtllm_gvr2' ranks the prefill and "
            f"decode rows with DeepSelect, which cannot run here: {why}"
        )


class KimiDecoderLayer(nn.Module):
    def __init__(
        self,
        config: KimiLinearConfig,
        vllm_config: VllmConfig,
        prefix: str = "",
        aux_stream: torch.cuda.Stream | None = None,
        run_gemm_rs_ar: bool = False,
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = int(prefix.rsplit(".", 1)[1])

        self.is_moe = config.is_moe
        layer_idx = self.layer_idx
        self._layer_kind = (
            "kda" if config.is_kda_layer(self.layer_idx)
            else ("dsa" if config.is_dsa_layer(self.layer_idx) else "mla")
        )
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config
        self.is_moe_layer = (
            self.is_moe
            and config.num_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0
        )

        self.use_sequence_parallel = use_sequence_parallel(vllm_config)
        if config.is_kda_layer(layer_idx):
            kda_config = config.linear_attn_config
            assert kda_config is not None
            # This class also serves standalone Kimi-Linear through the model
            # registry. Only Kimi-K3's full-rank gate uses the private KDA path.
            if kda_config.get("use_full_rank_gate", False):
                self.self_attn = KimiK3DeltaAttention(
                    config,
                    vllm_config,
                    prefix=f"{prefix}.self_attn",
                    aux_stream=aux_stream,
                    run_gemm_rs_ar=run_gemm_rs_ar,
                )
            else:
                self.self_attn = KimiLinearGatedDeltaNetAttention(
                    config,
                    vllm_config,
                    prefix=f"{prefix}.self_attn",
                )
        else:
            qk_nope_head_dim = config.qk_nope_head_dim
            qk_rope_head_dim = config.qk_rope_head_dim
            v_head_dim = config.v_head_dim
            kv_lora_rank = config.kv_lora_rank
            mla_use_nope = config.mla_use_nope
            assert qk_nope_head_dim is not None
            assert qk_rope_head_dim is not None
            assert v_head_dim is not None
            assert kv_lora_rank is not None
            assert mla_use_nope, "Kimi-K3 MLA (MultiHeadLatentAttention) is NoPE-only"
            self._dsa_dp = (
                config.is_dsa_layer(layer_idx)
                and dsa_dp.ENABLED
                and parallel_config.tensor_parallel_size > 1
                and parallel_config.decode_context_parallel_size == 1
            )
            # DSA-DP input scatter (dsa_dp.push_in): this layer issues no SP
            # all-gather, which KimiLinearModel.forward leaves out of its count.
            self._dsa_scatter = (
                self._dsa_dp and dsa_dp.SCATTER_IN and self.use_sequence_parallel
            )
            if config.is_dsa_layer(layer_idx):
                self.self_attn = KimiK3DSAAttention(
                    config=config,
                    hidden_size=self.hidden_size,
                    num_heads=config.num_attention_heads,
                    qk_nope_head_dim=qk_nope_head_dim,
                    qk_rope_head_dim=qk_rope_head_dim,
                    v_head_dim=v_head_dim,
                    q_lora_rank=config.q_lora_rank,
                    kv_lora_rank=kv_lora_rank,
                    cache_config=cache_config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.self_attn",
                    topk_indices_buffer=topk_indices_buffer,
                    aux_stream=aux_stream,
                    run_gemm_rs_ar=dsa_runs_gemm_rs_ar(vllm_config, run_gemm_rs_ar),
                    local_full_heads=self._dsa_dp,
                )
                if self._dsa_dp:
                    dsa_dp.DSA_LAYER_NAMES.add(self.self_attn.mla_attn.mla_attn.layer_name)
                    dsa_dp.DSA_LAYER_NAMES.add(self.self_attn.dsa_indexer.k_cache.prefix)
                    self._dsa_dp_idx = dsa_dp.register_eager(
                        self._dsa_scatter_rows
                        if self._dsa_scatter
                        else self._run_dsa_dp
                    )
            # q_lora_rank may be None (Kimi-Linear): the MLA layer then uses an
            # uncompressed q_proj instead of the fused q-LoRA front-end.
            else:
                self.self_attn = MultiHeadLatentAttention(
                    config=config,
                    hidden_size=self.hidden_size,
                    num_heads=config.num_attention_heads,
                    qk_nope_head_dim=qk_nope_head_dim,
                    qk_rope_head_dim=qk_rope_head_dim,
                    v_head_dim=v_head_dim,
                    q_lora_rank=config.q_lora_rank,
                    kv_lora_rank=kv_lora_rank,
                    use_output_gate=bool(config.mla_use_output_gate),
                    cache_config=cache_config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.self_attn",
                    aux_stream=aux_stream,
                    run_gemm_rs_ar=run_gemm_rs_ar,
                )

        if self.use_sequence_parallel:
            self.self_attn.o_proj.reduce_results = False

        if self.is_moe_layer:
            self.block_sparse_moe = KimiMoE(
                config=config,
                vllm_config=vllm_config,
                quant_config=quant_config,
                prefix=f"{prefix}.block_sparse_moe",
                layer_idx=layer_idx,
                use_sequence_parallel=self.use_sequence_parallel,
                run_gemm_rs_ar=run_gemm_rs_ar,
            )
            self.mlp = self.block_sparse_moe
        else:
            self.mlp = KimiMLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                use_sequence_parallel=self.use_sequence_parallel,
                can_shard_sequence_parallel=True,
                run_gemm_rs_ar=run_gemm_rs_ar,
                activation_situ_beta=config.activation_situ_beta,
                activation_situ_linear_beta=config.activation_situ_linear_beta,
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        attn_res_block_size = config.attn_res_block_size
        self.use_attn_res = attn_res_block_size is not None
        if self.use_attn_res:
            assert attn_res_block_size is not None
            self.attn_res_block_size = attn_res_block_size
            self.is_block_write_layer = layer_idx % self.attn_res_block_size == 0
            self.block_write_idx = layer_idx // self.attn_res_block_size
            self.prev_valid_blocks = cdiv(layer_idx, self.attn_res_block_size)
            # Set by the model for one forward when the previous layer is a
            # DSpark aux layer: the pre-attention AttnRes writes its updated
            # prefix (= the aux state, one BF16 add) there (_aux_prefix_dst).
            self._aux_prefix_out: torch.Tensor | None = None
            self._aux_prefix_used = False
            self.self_attention_res_norm = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.mlp_res_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.self_attention_res_proj = ReplicatedLinear(
                config.hidden_size,
                1,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.self_attention_res_proj",
            )
            self.mlp_res_proj = ReplicatedLinear(
                config.hidden_size,
                1,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.mlp_res_proj",
            )
        if decode_weights_on_dies(vllm_config):
            from .specialized_decode.cutedsl.gemm.die_local import place_layer_on_dies

            place_layer_on_dies(self)

    def _dsa_scatter_rows(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """DSA-DP under SP: owned rows in from, and out to, the sequence shards."""
        num_rows = positions.shape[0]
        if dsa_dp.STEP.num_tokens < 0:
            return dsa_dp.local_rows_out(
                self.self_attn, hidden_states, positions, num_rows
            )
        own_in = dsa_dp.push_in(hidden_states, num_rows)
        own = None
        if own_in is not None:
            own = self.self_attn(
                hidden_states=own_in,
                positions=positions.index_select(0, dsa_dp.STEP.own_idx),
            )
        return dsa_dp.push_out(
            own, num_rows, self.hidden_size, hidden_states.dtype, hidden_states.device
        )

    def _run_dsa_dp(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """DSA-DP: run the layer on this card's owned rows only (all heads), other
        rows zero, so the TP reduction leaves each row with its owner's output."""
        if dsa_dp.graph_mode() == CUDAGraphMode.FULL:
            return dsa_dp.run_segment_rows(self.self_attn, positions, hidden_states)
        num_tokens = hidden_states.shape[0]
        out = hidden_states.new_zeros((num_tokens, self.hidden_size))
        rows = dsa_dp.rows_for(num_tokens, hidden_states.device)
        if rows is not None and rows.numel() > 0:
            own = self.self_attn(
                hidden_states=hidden_states.index_select(0, rows),
                positions=positions.index_select(0, rows),
            )
            out.index_copy_(0, rows, own)
        return out

    def _run_self_attn(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        resid_prefix: tuple[torch.Tensor, bool] | None = None,
    ) -> torch.Tensor:
        if resid_prefix is not None:
            return self.self_attn(
                hidden_states=hidden_states,
                positions=positions,
                resid_prefix=resid_prefix,
            )
        return self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
        )

    def _resid_producers(
        self, prefix_sum: torch.Tensor | None, residual: torch.Tensor | None
    ) -> tuple[bool, bool]:
        """(attention, mlp): which of this layer's output producers add into
        the die-local AttnRes prefix in their GEMM store (prefill dataflow C,
        specialized_prefill.attn_res_resid). Only while ``prefix_sum`` is the
        die-local prefix buffer of an attn_res_lp forward of at least
        ``attn_res_resid.MIN_ROWS`` rows (``use_for``: shorter steps keep the
        stock GEMM + AttnRes with a delta, which is faster there)."""
        caps = getattr(self, "_resid_caps", None)
        if caps is None:
            attn = (
                self.use_attn_res
                and not self.use_sequence_parallel
                and isinstance(self.self_attn, KimiK3DeltaAttention)
                and self.self_attn.resid_capable()
            )
            mlp = (
                self.use_attn_res
                and not self.use_sequence_parallel
                and isinstance(self.mlp, KimiMoE)
                and self.mlp.resid_capable()
            )
            caps = self._resid_caps = (attn, mlp)
        if not (caps[0] or caps[1]) or prefix_sum is None or residual is None:
            return False, False
        if not prefill_attn_res_resid.use_for(prefix_sum.shape[0]):
            return False, False
        if not (
            prefill_attn_res_lp.active(residual, prefix_sum)
            and prefill_attn_res_dielocal.is_prefix(prefix_sum, residual)
        ):
            return False, False
        return caps

    def _pre_attn_norm(
        self,
        hidden_states: torch.Tensor | None,
        residual: torch.Tensor | None,
        prefix_sum: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        if not self.use_attn_res:
            assert hidden_states is not None
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(hidden_states, residual)
            return hidden_states, prefix_sum, residual

        assert prefix_sum is not None
        assert residual is not None
        hidden_states, prefix_sum = _attn_res(
            prefix_sum,
            hidden_states,
            residual,
            self.self_attention_res_norm.weight,
            self.self_attention_res_proj.weight.squeeze(0),
            self.input_layernorm.weight,
            num_blocks=self.prev_valid_blocks,
            block_write_idx=(self.block_write_idx if self.is_block_write_layer else -1),
            eps=self.self_attention_res_norm.variance_epsilon,
            output_norm_eps=self.input_layernorm.variance_epsilon,
            call=2 * self.layer_idx,
            prefix_out=self._take_aux_prefix_out(),
        )
        return hidden_states, prefix_sum, residual

    def _take_aux_prefix_out(self) -> torch.Tensor | None:
        """The aux capture buffer this forward's pre-attention AttnRes writes
        its updated prefix to (consumed once), or None."""
        dst = self._aux_prefix_out
        if dst is not None:
            self._aux_prefix_out = None
            self._aux_prefix_used = True
        return dst

    def _post_attn_norm(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        prefix_sum: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        if not self.use_attn_res:
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )
            return hidden_states, prefix_sum, residual

        assert prefix_sum is not None
        if self.is_block_write_layer:
            # The prefix sum lives across later layers' GEMM-ARs: never keep a
            # view of the GEMM-AR workspace (``_gemm_ar_owns``).
            if _gemm_ar_owns(hidden_states):
                hidden_states = hidden_states.clone()
            prefix_sum = hidden_states
            prefix_delta = None
        else:
            prefix_delta = hidden_states
        mlp_valid_blocks = self.prev_valid_blocks + self.is_block_write_layer
        hidden_states, prefix_sum = _attn_res(
            prefix_sum,
            prefix_delta,
            residual,
            self.mlp_res_norm.weight,
            self.mlp_res_proj.weight.squeeze(0),
            self.post_attention_layernorm.weight,
            num_blocks=mlp_valid_blocks,
            block_write_idx=-1,
            eps=self.mlp_res_norm.variance_epsilon,
            output_norm_eps=self.post_attention_layernorm.variance_epsilon,
            call=2 * self.layer_idx + 1,
        )
        return hidden_states, prefix_sum, residual

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | None,
        residual: torch.Tensor | None,
        prefix_sum: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        evs = dsa_dp.P.new_layer(self._layer_kind) if dsa_dp.PROF else None
        dsa_dp.rec(evs)
        pre_mxfp8, post_mxfp8 = self._attn_res_mxfp8_consumers(prefix_sum)
        sp_mxfp8 = self._sp_mxfp8_gather(prefix_sum)
        if sp_mxfp8:
            # SP prefill: the AttnRes writes the shard of the KDA in_proj's
            # MXFP8 input and the gather pushes MXFP8 (ops/sp_all_gather.py
            # kimi_sp_all_gather_mxfp8: bitwise the quantization of the
            # gathered BF16 rows that in_proj ran before).
            _, hidden_states, prefix_sum = self._attn_res_mxfp8(
                prefix_sum, hidden_states, residual, pre=True
            )
        elif pre_mxfp8:
            # Prefill: the AttnRes writes the KDA in_proj's MXFP8 input (its
            # only consumer) instead of BF16.
            _, hidden_states, prefix_sum = self._attn_res_mxfp8(
                prefix_sum, hidden_states, residual, pre=True
            )
        else:
            hidden_states, prefix_sum, residual = self._pre_attn_norm(
                hidden_states, residual, prefix_sum
            )
        assert hidden_states is not None

        M = None
        if getattr(self, "_dsa_scatter", False):
            dsa_dp.rec(evs)
            mode = dsa_dp.graph_mode()
            if mode == CUDAGraphMode.FULL:
                hidden_states = dsa_dp.run_segment(
                    self.self_attn, positions, hidden_states
                )
            elif mode == CUDAGraphMode.PIECEWISE:
                out = torch.empty_like(hidden_states)
                dsa_dp.run_eager(self._dsa_dp_idx, positions, hidden_states, out)
                hidden_states = out
            else:
                hidden_states = self._dsa_scatter_rows(positions, hidden_states)
            dsa_dp.rec(evs)
            dsa_dp.rec(evs)
            hidden_states, prefix_sum, residual = self._post_attn_norm(
                hidden_states, residual, prefix_sum
            )
            hidden_states = self.mlp(hidden_states)
            dsa_dp.rec(evs)
            return hidden_states, prefix_sum, residual
        if sp_mxfp8:
            M = positions.shape[0]
            hidden_states = sp_ag.kimi_sp_all_gather_mxfp8(hidden_states, M)
        elif self.use_sequence_parallel:
            hidden_states = sp_ag.kimi_sp_all_gather(hidden_states)
            # Remove SP padding before attention.
            hidden_states = hidden_states[: positions.shape[0]]
            M = hidden_states.shape[0]
        dsa_dp.rec(evs)

        # Prefill dataflow C: the KDA o_proj / MoE shared down_proj add their
        # outputs into the die-local AttnRes prefix (attn_res_resid), and the
        # AttnRes calls after them read the prefix alone.
        resid_attn, resid_mlp = self._resid_producers(prefix_sum, residual)

        # Attention.
        if (
            getattr(self, "_dsa_dp", False)
            and dsa_dp.PUSH
            and self.use_sequence_parallel
            and dsa_dp.STEP.num_tokens >= 0
        ):
            # Owned rows only; outputs pushed straight to their sequence shards.
            rows = dsa_dp.rows_for(M, hidden_states.device)
            own = None
            if rows is not None and rows.numel() > 0:
                own = self.self_attn(
                    hidden_states=hidden_states.index_select(0, rows),
                    positions=positions.index_select(0, rows),
                )
            hidden_states = dsa_dp.push_out(
                own, M, self.hidden_size, hidden_states.dtype, hidden_states.device
            )
            M = None
        elif getattr(self, "_dsa_dp", False):
            if dsa_dp.graph_mode() == CUDAGraphMode.PIECEWISE:
                out = hidden_states.new_empty(
                    (hidden_states.shape[0], self.hidden_size)
                )
                dsa_dp.run_eager(self._dsa_dp_idx, positions, hidden_states, out)
                hidden_states = out
            else:
                hidden_states = self._run_dsa_dp(positions, hidden_states)
        else:
            hidden_states = self._run_self_attn(
                positions,
                hidden_states,
                (prefix_sum, not self.is_block_write_layer) if resid_attn else None,
            )
        dsa_dp.rec(evs)

        # GEMM-RS returns the local sequence shard; standard O-proj preserves M.
        if self.use_sequence_parallel and hidden_states.shape[0] == M:
            # Add SP padding if needed, and then perform reduce scatter.
            hidden_states = sp_reduce_scatter(hidden_states)
        elif getattr(self, "_dsa_dp", False) and not self.use_sequence_parallel:
            hidden_states = tensor_model_parallel_all_reduce(hidden_states)
        dsa_dp.rec(evs)

        if post_mxfp8:
            # Prefill: the AttnRes also writes the shared experts' MXFP8
            # gate_up input (the router / latent down_proj read the BF16).
            if self.is_block_write_layer:
                if _gemm_ar_owns(hidden_states):
                    hidden_states = hidden_states.clone()
                prefix_sum = hidden_states
            hidden_states, shared_mxfp8, prefix_sum = self._attn_res_mxfp8(
                prefix_sum, hidden_states, residual, pre=False, attn_resid=resid_attn
            )
            if resid_mlp:
                self.mlp(
                    hidden_states,
                    shared_mxfp8_input=shared_mxfp8,
                    resid_prefix=prefix_sum,
                )
                return None, prefix_sum, residual
            hidden_states = self.mlp(hidden_states, shared_mxfp8_input=shared_mxfp8)
            return hidden_states, prefix_sum, residual

        hidden_states, prefix_sum, residual = self._post_attn_norm(
            hidden_states, residual, prefix_sum
        )

        # MoE/MLP.
        hidden_states = self.mlp(hidden_states)
        dsa_dp.rec(evs)
        return hidden_states, prefix_sum, residual

    def _attn_res_mxfp8_capabilities(self) -> tuple[bool, bool]:
        """(pre, post): the pre-attention / post-attention AttnRes output feeds
        an online-MXFP8 projection that can take it as MXFP8 from the AttnRes
        epilogue (specialized_prefill ``attn_res_mxfp8``): the KDA in_proj (the
        pre-attention AttnRes' only consumer) and the MegaMoE shared experts'
        gate_up."""
        caps = getattr(self, "_attn_res_mxfp8_caps", None)
        if caps is None:
            base = self.use_attn_res and not self.use_sequence_parallel
            pre = (
                base
                and isinstance(self.self_attn, KimiK3DeltaAttention)
                and takes_mxfp8_activation(self.self_attn.in_proj_qkvgfab)
            )
            post = (
                base
                and isinstance(self.mlp, KimiMoE)
                and self.mlp.takes_shared_mxfp8_input()
            )
            caps = self._attn_res_mxfp8_caps = (pre, post)
        return caps

    def _sp_mxfp8_gather_capable(self) -> bool:
        """Under SP, the KDA in_proj (the pre-attention AttnRes' only consumer)
        is an online-MXFP8 projection that takes a pre-quantized activation:
        its input can be quantized per shard and gathered as MXFP8."""
        cap = getattr(self, "_sp_mxfp8_cap", None)
        if cap is None:
            cap = self._sp_mxfp8_cap = (
                self.use_attn_res
                and self.use_sequence_parallel
                and not getattr(self, "_dsa_scatter", False)
                and not getattr(self, "_dsa_dp", False)
                and isinstance(self.self_attn, KimiK3DeltaAttention)
                and takes_mxfp8_activation(self.self_attn.in_proj_qkvgfab)
            )
        return cap

    def _sp_mxfp8_gather(self, prefix_sum: torch.Tensor | None) -> bool:
        """This step's SP gather goes as MXFP8: the capability, a prefill
        shard of the native AttnRes contract, and a shard size the MXFP8
        gather takes (``sp_ag.mxfp8_gather_rows_ok``: a multiple of 128 rows
        on the symmetric-memory push; the Lamport all-gather stays BF16)."""
        if prefix_sum is None or not self._sp_mxfp8_gather_capable():
            return False
        return prefill_attn_res_mxfp8.supported(
            prefix_sum, 0, True
        ) and sp_ag.mxfp8_gather_rows_ok(prefix_sum.shape[0], prefix_sum.shape[1])

    def _attn_res_mxfp8_consumers(
        self, prefix_sum: torch.Tensor | None
    ) -> tuple[bool, bool]:
        """The capabilities, for this step's shape (prefill chunks only)."""
        caps = self._attn_res_mxfp8_capabilities()
        if not (caps[0] or caps[1]) or prefix_sum is None:
            return False, False
        if not prefill_attn_res_mxfp8.use_for(prefix_sum):
            return False, False
        return caps

    def _attn_res_mxfp8(
        self,
        prefix_sum: torch.Tensor,
        hidden_states: torch.Tensor | None,
        residual: torch.Tensor,
        pre: bool,
        attn_resid: bool = False,
    ):
        """``_pre_attn_norm`` (``pre``) / ``_post_attn_norm`` AttnRes with
        the MXFP8 output (BF16 output only after attention). ``attn_resid``:
        the attention output is already in the prefix (dataflow C)."""
        if prefill_attn_res_lp.active(residual, prefix_sum):
            prefix_out = self._take_aux_prefix_out() if pre else None
            keep = prefix_out is not None and _aux_copy_keeps_prefix(
                prefix_sum, hidden_states, residual
            )
            if prefix_out is None:
                prefix_out = prefill_attn_res_dielocal.prefix_out(prefix_sum, residual)
            new_prefix = prefix_sum if prefix_out is None or keep else prefix_out
            if pre:
                delta = hidden_states
                num_blocks = self.prev_valid_blocks
                widx = self.block_write_idx if self.is_block_write_layer else -1
                onw = self.input_layernorm
            else:
                delta = (
                    None if self.is_block_write_layer or attn_resid else hidden_states
                )
                num_blocks = self.prev_valid_blocks + self.is_block_write_layer
                widx = -1
                onw = self.post_attention_layernorm
            out, qa = prefill_attn_res_lp.attn_res(
                prefix_sum, delta, residual, 2 * self.layer_idx + (not pre),
                num_blocks, widx, onw.weight, onw.variance_epsilon,
                bf16_out=not pre, quant_out=True, prefix_out=prefix_out,
            )
            return out, qa, new_prefix
        if pre:
            out, qa = prefill_attn_res_mxfp8.attn_res_mxfp8(
                prefix_sum,
                hidden_states,
                residual,
                self.self_attention_res_norm.weight,
                self.self_attention_res_proj.weight.squeeze(0),
                self.input_layernorm.weight,
                num_blocks=self.prev_valid_blocks,
                block_write_idx=(
                    self.block_write_idx if self.is_block_write_layer else -1
                ),
                eps=self.self_attention_res_norm.variance_epsilon,
                output_norm_eps=self.input_layernorm.variance_epsilon,
                bf16_out=False,
            )
            return out, qa, prefix_sum
        prefix_delta = None if self.is_block_write_layer else hidden_states
        out, qa = prefill_attn_res_mxfp8.attn_res_mxfp8(
            prefix_sum,
            prefix_delta,
            residual,
            self.mlp_res_norm.weight,
            self.mlp_res_proj.weight.squeeze(0),
            self.post_attention_layernorm.weight,
            num_blocks=self.prev_valid_blocks + self.is_block_write_layer,
            block_write_idx=-1,
            eps=self.mlp_res_norm.variance_epsilon,
            output_norm_eps=self.post_attention_layernorm.variance_epsilon,
            bf16_out=True,
        )
        return out, qa, prefix_sum



# (param_name, shard_name, shard_id) for the merged DSA wkv_index.
_DSA_STACKED_PARAMS_MAPPING = [
    (".self_attn.dsa_indexer.wkv_index.", ".self_attn.indexer.wk.", 0),
    (".self_attn.dsa_indexer.wkv_index.", ".self_attn.indexer.weights_proj.", 1),
]

# DSA indexer weight rename mapping. Specific rules must come before the
# generic ".self_attn.indexer." rule: the loop breaks on the first match.
_DSA_RENAME_PARAMS_MAPPING = [
    (".self_attn.dsa_indexer.wq_index.", ".self_attn.indexer.wq_b."),
    (".self_attn.dsa_indexer.k_index_norm.", ".self_attn.indexer.k_norm."),
    # Generic fallback for already-nested indexer names
    (".self_attn.dsa_indexer.", ".self_attn.indexer."),
]


class KimiLinearModel(nn.Module, EagleModelMixin, SupportsQuant):
    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvgfab": ["q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj"],
        "conv1d": ["q_conv1d", "k_conv1d", "v_conv1d"],
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
        "fused_qkv_a_g_proj": ["q_a_proj", "kv_a_proj_with_mqa", "g_proj"],
    }

    supports_aux_hidden_states_over_pp = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_text_config
        self.config = config
        self.attn_res_block_size: int | None = config.attn_res_block_size
        self.use_attn_res = self.attn_res_block_size is not None
        parallel_config = vllm_config.parallel_config
        # DWDP layout (ParallelConfig.dwdp) before the MegaMoE experts are built.
        dense_dwdp.edwdp.configure(parallel_config)
        self.use_sequence_parallel = use_sequence_parallel(vllm_config)

        self.vocab_size = config.vocab_size

        # GEMM-RS/AR uses NCCL symmetric-memory multicast, which requires all
        # TP ranks to belong to one NVLink domain.
        self.run_gemm_rs_ar = maybe_init_gemm_rs_ar(
            vllm_config, self.use_sequence_parallel
        )
        # The decoder layers' SP all-gather pushes into a symmetric-memory
        # buffer of the largest stock step above the custom all-gather's size,
        # if this instance runs one (ops/sp_all_gather.py).
        if self.use_sequence_parallel:
            sp_ag.init_sp_all_gather(
                vllm_config, config.hidden_size, vllm_config.model_config.dtype
            )

        if get_pp_group().is_first_rank or spec_decode_needs_target_embed(vllm_config):
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                prefix=f"{prefix}.embed_tokens",
            )
            dense_dwdp.shard_table(self.embed_tokens.weight)
        else:
            self.embed_tokens = PPMissingLayer()

        # Aux stream for overlapping the MLA g_proj output-gate GEMM with the
        # attention front-end; created at the model level and threaded into
        # each attention layer.
        aux_stream = torch.cuda.Stream()

        # Single model-level top-k buffer shared by all DSA layers.
        # [max_num_batched_tokens + dcp_kg_slack, index_topk] int32.
        self.topk_indices_buffer: torch.Tensor | None = None
        if config.dsa_config is not None and any(
            config.is_dsa_layer(layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ):
            topk_buffer_device = "cuda" if torch.cuda.is_available() else "cpu"
            # In allgather_k mode prefill rows are padded to ceil(P/W) per
            # rank, so the all-gathered output takes num_decode_tokens +
            # W*ceil(P/W) rows, at most W-1 more than num_actual_tokens.
            dcp_kg_slack = (
                parallel_config.decode_context_parallel_size - 1
                if parallel_config.decode_context_parallel_size > 1
                else 0
            )
            self.topk_indices_buffer = torch.empty(
                (
                    vllm_config.scheduler_config.max_num_batched_tokens + dcp_kg_slack,
                    config.dsa_config.index_topk,
                ),
                dtype=torch.int32,
                device=topk_buffer_device,
            )

        def get_layer(prefix: str):
            layer = KimiDecoderLayer(
                config,
                vllm_config,
                prefix,
                aux_stream=aux_stream,
                run_gemm_rs_ar=self.run_gemm_rs_ar,
                topk_indices_buffer=self.topk_indices_buffer,
            )
            # Non-expert DWDP shards each layer's linear weights once it exists.
            dense_dwdp.shard_layer(layer, extract_layer_index(prefix))
            return layer

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            get_layer,
            prefix=f"{prefix}.layers",
        )
        # Served EP-balance log: records map to MoE layers by call order (megamoe_rubin/ep_balance.py).
        n_mega = sum(
            bool(getattr(getattr(layer, "block_sparse_moe", None), "use_mega_moe", False)
                 or getattr(getattr(layer, "mlp", None), "use_mega_moe", False))
            for layer in self.layers[self.start_layer : self.end_layer]
        )
        if n_mega:
            from vllm.models.kimi_k3.nvidia.megamoe_rubin import ep_balance

            ep_balance.configure(n_mega)

        if config.dsa_config is not None:
            # A DSA layer silently built as dense MLA would be invisible in
            # outputs; check the wiring matches dsa_config.
            n_dsa_built = sum(
                isinstance(layer.self_attn, KimiK3DSAAttention)
                for layer in self.layers[self.start_layer : self.end_layer]
            )
            n_dsa_expected = sum(
                config.is_dsa_layer(layer_idx)
                for layer_idx in range(self.start_layer, self.end_layer)
            )
            assert n_dsa_built == n_dsa_expected, (
                f"DSA wiring broken: built {n_dsa_built} DSA attention layers "
                f"but dsa_config expects {n_dsa_expected}."
            )

        self.num_attn_res_blocks = (
            cdiv(self.end_layer, self.attn_res_block_size)
            if self.attn_res_block_size is not None
            else 0
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if self.use_attn_res:
                self.output_attn_res_norm = RMSNorm(
                    config.hidden_size, eps=config.rms_norm_eps
                )
                self.output_attn_res_proj = ReplicatedLinear(
                    config.hidden_size,
                    1,
                    bias=False,
                    quant_config=None,
                    prefix=f"{prefix}.output_attn_res_proj",
                )
        else:
            self.norm = PPMissingLayer()
            if self.use_attn_res:
                self.output_attn_res_norm = PPMissingLayer()
                self.output_attn_res_proj = PPMissingLayer()

        world_size = get_tensor_model_parallel_world_size()
        assert config.num_attention_heads % world_size == 0, (
            "num_attention_heads must be divisible by world_size"
        )

    def _attn_res_lp_plan(self) -> "prefill_attn_res_lp.Plan":
        """Per-call AttnRes queries of the whole forward (layer l: 2 l
        pre-attention, 2 l + 1 post-attention; the final AttnRes last), built
        once from the loaded weights."""
        plan = getattr(self, "_attn_res_lp_plan_cache", None)
        if plan is None:
            norms = []
            for layer in self.layers[self.start_layer : self.end_layer]:
                norms.append((layer.self_attention_res_norm, layer.self_attention_res_proj))
                norms.append((layer.mlp_res_norm, layer.mlp_res_proj))
            norms.append((self.output_attn_res_norm, self.output_attn_res_proj))
            assert self.attn_res_block_size is not None
            plan = self._attn_res_lp_plan_cache = prefill_attn_res_lp.Plan(
                [(nm.weight, pj.weight.squeeze(0)) for nm, pj in norms],
                [nm.variance_epsilon for nm, _ in norms],
                self.attn_res_block_size,
                self.num_attn_res_blocks,
            )
        return plan

    def make_empty_intermediate_tensors(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        residual_shape: tuple[int, ...] = (batch_size, self.config.hidden_size)
        if self.use_attn_res:
            assert self.attn_res_block_size is not None
            residual_shape = (
                batch_size,
                cdiv(self.start_layer, self.attn_res_block_size),
                self.config.hidden_size,
            )
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros(
                    (batch_size, self.config.hidden_size), dtype=dtype, device=device
                ),
                "residual": torch.zeros(residual_shape, dtype=dtype, device=device),
            }
        )

    # Set by a drafter that takes sequence-shard aux states (ops/sp_local_aux.py).
    drafter_takes_local_aux = False

    def _set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        super()._set_aux_hidden_state_layers(layers)
        if self.use_attn_res and self._aux_attn_res_stream:
            pp = get_pp_group()
            if not pp.is_last_rank and self.end_layer in self.aux_hidden_state_layers:
                raise ValueError(
                    f"Auxiliary layer {self.end_layer} cannot end a non-final PP "
                    "stage when VLLM_KIMI_K3_AUX_ATTN_RES_STREAM=1"
                )
        if self.use_attn_res:
            logger.info_once(
                "Kimi-K3 aux hidden capture: layers=%s mode=%s "
                "(VLLM_KIMI_K3_AUX_ATTN_RES_STREAM=%d)",
                self.aux_hidden_state_layers,
                "attn_res_stream" if self._aux_attn_res_stream else "prefix_only",
                int(self._aux_attn_res_stream),
            )

    @property
    def _aux_attn_res_stream(self) -> bool:
        return envs.VLLM_KIMI_K3_AUX_ATTN_RES_STREAM

    def _aux_from_next_attn_res(
        self,
        layer_idx: int,
        prefix_sum: torch.Tensor | None,
        block_residual: torch.Tensor | None,
    ) -> bool:
        """Whether the aux state after ``layer_idx`` (prefix_sum + the pending
        MLP output) is captured as the updated prefix the next AttnRes call of
        this rank writes (prefill forwards on the attn_res_lp plan with the
        die-local buffers: the call after it then writes the die-local prefix
        buffer, never the capture in place)."""
        return (
            self.use_attn_res
            and not self._aux_attn_res_stream
            and prefix_sum is not None
            and block_residual is not None
            and (layer_idx + 1 < self.end_layer or get_pp_group().is_last_rank)
            and prefill_attn_res_lp.active(block_residual, prefix_sum)
            and prefill_attn_res_dielocal.kernel_args(block_residual)[0] is not None
        )

    def _capture_aux_hidden_stream(
        self,
        layer_idx: int,
        prefix_sum: torch.Tensor,
        pending_mlp_out: torch.Tensor | None,
        block_residual: torch.Tensor,
    ) -> torch.Tensor:
        """Return the AttnRes stream after ``layer_idx``."""
        prefix = prefix_sum if pending_mlp_out is None else prefix_sum + pending_mlp_out
        if not (self._aux_attn_res_stream and self.use_attn_res):
            # Dataflow C (no pending output): the prefix buffer itself is
            # updated in place by the next layers' producers.
            return prefix.clone() if pending_mlp_out is None else prefix

        if layer_idx + 1 < self.end_layer:
            consumer = self.layers[layer_idx + 1]
            score_norm = consumer.self_attention_res_norm
            score_proj = consumer.self_attention_res_proj
            num_blocks = consumer.prev_valid_blocks
        elif get_pp_group().is_last_rank:
            score_norm = self.output_attn_res_norm
            score_proj = self.output_attn_res_proj
            num_blocks = self.num_attn_res_blocks
        else:
            raise RuntimeError("Auxiliary AttnRes capture crossed a PP boundary")

        return attn_res(
            prefix,
            None,
            block_residual,
            score_norm.weight,
            score_proj.weight.squeeze(0),
            None,
            num_blocks=num_blocks,
            block_write_idx=-1,
            eps=score_norm.variance_epsilon,
            output_norm_eps=0.0,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        # Under SP the lookup's shards are reduce-scattered straight to this
        # rank's token shard (sp_embed) instead of all-reduced and sharded.
        sp_lookup = (
            self.use_sequence_parallel
            and get_pp_group().is_first_rank
            and inputs_embeds is None
        )
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            elif sp_lookup:
                assert input_ids is not None
                hidden_states = sp_embed(self.embed_tokens, input_ids)
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        assert hidden_states is not None

        full_num_tokens = positions.shape[0]
        if self.use_sequence_parallel:
            # One decoder-layer SP all-gather per layer: their buffer slots.
            sp_ag.begin_forward(
                sum(
                    not getattr(layer, "_dsa_scatter", False)
                    for layer in self.layers[self.start_layer : self.end_layer]
                )
            )
            if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
                forward_context = get_forward_context()
                forward_context.is_padding = sp_padding_mask(
                    forward_context.is_padding,
                    input_ids if sp_lookup else hidden_states,
                )
            if not sp_lookup:
                hidden_states = sp_shard(hidden_states)
            assert residual is None, "Currently, SP is not supported with PP"

        remote_aux = self.collect_remote_aux_hidden_states(intermediate_tensors)

        # sharded aux hidden states when sp is enabled
        aux_hidden_states: list[torch.Tensor] = []
        if (
            get_pp_group().is_first_rank
            and self.start_layer in self.aux_hidden_state_layers
        ):
            if self.use_attn_res or residual is None:
                aux_hidden_states.append(hidden_states)
            else:
                aux_hidden_states.append(hidden_states + residual)

        prefix_sum = None
        if self.use_attn_res:
            block_residual = None
            if residual is None and prefill_attn_res_mxfp8.use_for(hidden_states):
                # Prefill: block rows homed per die for die-local AttnRes reads.
                block_residual = prefill_attn_res_dielocal.block_residual(
                    hidden_states, self.num_attn_res_blocks
                )
            if block_residual is None:
                block_residual = hidden_states.new_empty(
                    hidden_states.size(0),
                    self.num_attn_res_blocks,
                    hidden_states.size(1),
                )
            if residual is not None:
                block_residual[:, : residual.size(1), :].copy_(residual)
            elif (
                prefill_attn_res_lp.use_for(hidden_states)
                and self.start_layer == 0
                and get_pp_group().is_last_rank
            ):
                # Prefill: block logits precomputed per block write.
                prefill_attn_res_lp.begin(self._attn_res_lp_plan(), block_residual)
            prefix_sum = hidden_states
            hidden_states = None
            residual = block_residual

        # Prefill (attn_res_lp plan): the aux state after layer i is the prefix
        # the next AttnRes call (layer i + 1's pre-attention one, or the final
        # one) writes anyway (prefix + MLP output, one BF16 add): that call
        # writes it to a fresh buffer instead of the die-local prefix buffer
        # (the call after it moves the prefix back), so the capture costs no
        # extra pass over [tokens, hidden]. Under dataflow C (the producers
        # already added the outputs into the die-local prefix) that call writes
        # the capture as a copy and the prefix stays die-local
        # (_aux_copy_keeps_prefix).
        aux_dst: torch.Tensor | None = None
        dense = dense_dwdp.get()
        breadcrumb.crumb(breadcrumb.OP_FWD_BEGIN, self.start_layer)
        for layer_idx, layer in enumerate(
            self.layers[self.start_layer : self.end_layer],
            start=self.start_layer,
        ):
            breadcrumb.crumb(breadcrumb.OP_LAYER, layer_idx)
            if aux_dst is not None:
                layer._aux_prefix_out = aux_dst
                layer._aux_prefix_used = False
            if dense is not None:
                dense.layer_start(layer_idx)
            hidden_states, prefix_sum, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                prefix_sum=prefix_sum,
                residual=residual,
            )
            if aux_dst is not None:
                layer._aux_prefix_out = None
                if not layer._aux_prefix_used:
                    raise RuntimeError(
                        f"Kimi K3 aux capture: layer {layer_idx} did not write the "
                        "aux state (its pre-attention AttnRes left the LP kernel)"
                    )
                aux_hidden_states.append(aux_dst)
                aux_dst = None
            if (layer_idx + 1) in self.aux_hidden_state_layers:
                if self._aux_from_next_attn_res(layer_idx, prefix_sum, residual):
                    # Appended once the next AttnRes call has written it.
                    aux_dst = torch.empty_like(prefix_sum)
                    continue
                if self.use_attn_res:
                    assert prefix_sum is not None
                    assert residual is not None
                    aux_hidden_state = self._capture_aux_hidden_stream(
                        layer_idx, prefix_sum, hidden_states, residual
                    )
                else:
                    assert residual is not None
                    aux_hidden_state = hidden_states + residual

                aux_hidden_states.append(aux_hidden_state)

        breadcrumb.crumb(breadcrumb.OP_FWD_END)
        assert hidden_states is not None or prefix_sum is not None
        assert residual is not None
        if not get_pp_group().is_last_rank:
            assert not self.use_sequence_parallel, (
                "Currently, SP is not supported with PP"
            )
            if prefix_sum is not None:
                hidden_states = (
                    prefix_sum.clone()
                    if hidden_states is None
                    else hidden_states + prefix_sum
                )
            return IntermediateTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                    **self.pack_local_aux_hidden_states(aux_hidden_states),
                }
            )

        if self.use_attn_res:
            assert prefix_sum is not None
            hidden_states, _ = _attn_res(
                prefix_sum,
                hidden_states,
                residual,
                self.output_attn_res_norm.weight,
                self.output_attn_res_proj.weight.squeeze(0),
                None,
                num_blocks=self.num_attn_res_blocks,
                block_write_idx=-1,
                eps=self.output_attn_res_norm.variance_epsilon,
                output_norm_eps=0.0,
                call=2 * self.end_layer,
                prefix_out=aux_dst,
            )
            if aux_dst is not None:  # the last layer is an aux layer
                aux_hidden_states.append(aux_dst)
                aux_dst = None
        else:
            hidden_states = hidden_states + residual

        local_aux = (
            bool(aux_hidden_states)
            and self.use_sequence_parallel
            and self.drafter_takes_local_aux
            and sp_local_aux.use_local_aux(full_num_tokens)
        )
        if local_aux:
            hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]
        elif self.use_sequence_parallel:
            if aux_hidden_states:
                hidden_size = hidden_states.shape[-1]
                packed_hidden_states = torch.cat(
                    [hidden_states, *aux_hidden_states], dim=-1
                )
                packed_hidden_states = sp_all_gather(packed_hidden_states)
                packed_hidden_states = packed_hidden_states[:full_num_tokens]
                hidden_states, *aux_hidden_states = packed_hidden_states.split(
                    hidden_size, dim=-1
                )
            else:
                hidden_states = sp_all_gather(hidden_states)
                hidden_states = hidden_states[:full_num_tokens]

        # NOTE: the final norm is applied in compute_logits instead of here.
        if local_aux:
            assert not remote_aux, "SP is not supported with PP"
            aux_hidden_states = sp_local_aux.SpLocalAux(
                aux_hidden_states, full_num_tokens
            )
        else:
            aux_hidden_states = remote_aux + aux_hidden_states
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states

    def load_weights(
        self,
        weights: Iterable[
            tuple[str, torch.Tensor] | tuple[str, torch.Tensor, dict[str, Any]]
        ],
    ) -> set[str]:
        kda_config = self.config.linear_attn_config
        use_full_rank_gate = bool(
            kda_config and kda_config.get("use_full_rank_gate", False)
        )
        beta_shard_id = 5 if use_full_rank_gate else 3
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".in_proj_qkvgfab", ".q_proj", 0),
            (".in_proj_qkvgfab", ".k_proj", 1),
            (".in_proj_qkvgfab", ".v_proj", 2),
            (".in_proj_qkvgfab", ".b_proj", beta_shard_id),
            (".in_proj_qkvgfab", ".f_a_proj", 4),
            (".conv1d", ".q_conv1d", 0),
            (".conv1d", ".k_conv1d", 1),
            (".conv1d", ".v_conv1d", 2),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        if use_full_rank_gate:
            stacked_params_mapping.append((".in_proj_qkvgfab", ".g_proj", 3))
        if getattr(self.config, "q_lora_rank", None) is not None:
            if self.config.mla_use_output_gate:
                stacked_params_mapping += [
                    (".fused_qkv_a_g_proj", ".q_a_proj", 0),
                    (".fused_qkv_a_g_proj", ".kv_a_proj_with_mqa", 1),
                    (".fused_qkv_a_g_proj", ".g_proj", 2),
                ]
            else:
                stacked_params_mapping += [
                    (".fused_qkv_a_proj", ".q_a_proj", 0),
                    (".fused_qkv_a_proj", ".kv_a_proj_with_mqa", 1),
                ]
            if self.config.dsa_config is not None and self.config.mla_use_output_gate:
                # DSA layers have no fused_qkv_a_g_proj param; fall back to
                # the plain fused_qkv_a_proj mapping.
                stacked_params_mapping += [
                    (".fused_qkv_a_proj", ".q_a_proj", 0),
                    (".fused_qkv_a_proj", ".kv_a_proj_with_mqa", 1),
                ]
        if self.config.dsa_config is not None:
            stacked_params_mapping += _DSA_STACKED_PARAMS_MAPPING
        mega_moe_formats = {
            module.experts.weight_format
            for module in self.modules()
            if isinstance(module, KimiMoE) and module.use_mega_moe
        }
        use_mega_moe = bool(mega_moe_formats)
        if len(mega_moe_formats) > 1:
            raise NotImplementedError(
                "Kimi K3 MegaMoE layers with different expert formats: "
                f"{sorted(f.name for f in mega_moe_formats)}."
            )
        if self.config.is_moe and use_mega_moe:
            expert_params_mapping = make_kimi_k3_mega_moe_expert_params_mapping(
                self.config.num_experts, next(iter(mega_moe_formats))
            )
        elif self.config.is_moe:
            # Params for weights, fp8 weight scales, fp8 activation scales
            # (param_name, weight_name, expert_id, shard_id)
            expert_params_mapping = fused_moe_make_expert_params_mapping(
                self,
                ckpt_gate_proj_name="w1",
                ckpt_down_proj_name="w2",
                ckpt_up_proj_name="w3",
                num_experts=self.config.num_experts,
            )
        else:
            expert_params_mapping = []
        params_dict = dict(self.named_parameters())

        # Under the MXFP4 quant interface the routed experts register unpacked
        # params (``w13_weight``), while the compressed-tensors checkpoint names
        # them ``.weight_packed``. Rebind so the expert mapping resolves; scales
        # already share the ``.weight_scale`` suffix.
        experts_unpacked = not use_mega_moe and not any(
            n.endswith("w13_weight_packed") for n in params_dict
        )
        loaded_params: set[str] = set()
        for args in weights:
            name, loaded_weight = args[0], args[1]
            kwargs: dict[str, Any] = args[2] if len(args) > 2 else {}
            if "rotary_emb.inv_freq" in name:
                continue
            if experts_unpacked and name.endswith(".weight_packed"):
                name = name.replace(".weight_packed", ".weight")

            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                name_mapped = name.replace(weight_name, param_name)
                # Packed projections are only present on compatible layers.
                if name_mapped not in params_dict:
                    continue
                name = name_mapped
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for (
                    expert_param_name,
                    expert_weight_name,
                    expert_id,
                    expert_shard_id,
                ) in expert_params_mapping:
                    if expert_weight_name not in name:
                        continue
                    name = name.replace(expert_weight_name, expert_param_name)
                    if is_pp_missing_parameter(name, self):
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        expert_id=expert_id,
                        shard_id=expert_shard_id,
                    )
                    break
                else:
                    # Rename indexer checkpoint names into the module tree
                    # before the param lookup below.
                    if self.config.dsa_config is not None:
                        for param_name, weight_name in _DSA_RENAME_PARAMS_MAPPING:
                            if weight_name in name:
                                name = name.replace(weight_name, param_name)
                                break
                    # Skip loading extra bias for GPTQ models.
                    if (
                        name.endswith(".bias")
                        and name not in params_dict
                        and not self.config.is_linear_attn
                    ):  # noqa: E501
                        continue
                    # Remapping the name of FP8 kv-scale.
                    remapped_name = maybe_remap_kv_scale_name(name, params_dict)
                    if remapped_name is None:
                        continue
                    name = remapped_name
                    if is_pp_missing_parameter(name, self):
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight, **kwargs)
            loaded_params.add(name)
        dense_dwdp.loaded()
        return loaded_params


def log_kimi_k3_quant_kinds(model: nn.Module) -> dict[str, int]:
    """Log (once) and return how many linears / MoE blocks run each precision:
    the checkpoint's per-layer quantization as this process built it."""
    kinds: dict[str, int] = {}
    for module in model.modules():
        if isinstance(module, KimiMoE) and module.use_mega_moe:
            experts = module.experts
            kind = f"MegaMoE {experts.weight_format.name} experts"
            if hasattr(experts, "quant_kind"):
                kind += f" ({experts.quant_kind} kernel)"
        elif isinstance(module, LinearBase):
            method = module.quant_method
            spec = getattr(method, "spec", None)
            kind = type(method).__name__
            if spec is not None:
                kind += f"[{spec.weight}]"
        else:
            continue
        kinds[kind] = kinds.get(kind, 0) + 1
    logger.info_once(
        "Kimi K3 quantization: %s",
        "; ".join(f"{n} x {k}" for k, n in sorted(kinds.items())),
    )
    return kinds


class KimiLinearForCausalLM(
    nn.Module,
    HasInnerState,
    SupportsPP,
    MixtureOfExperts,
    IsHybrid,
    SupportsEagle3,
    SupportsReplaySSM,
):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.model_config = vllm_config.model_config
        self.vllm_config = vllm_config
        self.config = self.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.quant_config = quant_config
        self.model = KimiLinearModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                self.config.vocab_size,
                self.config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            dense_dwdp.shard_table(self.lm_head.weight)
        else:
            self.lm_head = PPMissingLayer()
        dense_dwdp.log_construction()
        log_kimi_k3_quant_kinds(self)
        enable_kimi_k3_low_latency_gemm(self, self.model_config.dtype)
        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(
            self.config.vocab_size, scale=logit_scale
        )
        # decode_backend=latency_specialized: a specialized decode executor over
        # self.model, built by the model state once the weights are loaded.
        self.specialized_decode_config = specialized_decode_config(vllm_config)
        self.specialized_decode: SpecializedDecode | None = None
        # The vocabulary-sharded logits and sampled verification of verify_sampled
        # (built with the specialized decode on Rubin when speculating).
        self.verify_logits = None
        self.sampled_verify = None
        # The final norm is applied in compute_logits unless the specialized decode
        # applies it in forward.
        self.needs_norm_in_logits = True
        self._init_moe_metadata()

    def _init_moe_metadata(self) -> None:
        """MixtureOfExperts attributes for EPLB, over the Rubin MegaMoE routed experts
        (the only backend here that rearranges experts in place)."""
        from .megamoe_rubin.experts import KimiK3MegaMoEExperts

        self.moe_layers = [
            m for m in self.model.modules() if isinstance(m, KimiK3MegaMoEExperts)
        ]
        self.num_moe_layers = len(self.moe_layers)
        self.expert_weights = []
        self.num_expert_groups = getattr(self.config, "num_expert_group", 1) or 1
        self.num_shared_experts = getattr(self.config, "num_shared_experts", 0) or 0
        if not self.moe_layers:
            self.num_logical_experts = self.num_physical_experts = 0
            self.num_local_physical_experts = self.num_routed_experts = 0
            self.num_redundant_experts = 0
            return
        first = self.moe_layers[0]
        self.num_logical_experts = first.num_logical_experts
        self.num_physical_experts = first.num_experts
        self.num_local_physical_experts = first.num_local_experts
        self.num_routed_experts = first.num_logical_experts
        self.num_redundant_experts = first.num_redundant_experts

    def update_physical_experts_metadata(
        self, num_physical_experts: int, num_local_physical_experts: int
    ) -> None:
        raise NotImplementedError(
            "Kimi K3 Rubin MegaMoE: elastic EP (changing the physical expert count) "
            "is not supported."
        )

    def process_weights_after_loading(self) -> None:
        from .mxfp8 import enable_kimi_k3_mxfp8_dispatch

        if switched := enable_kimi_k3_mxfp8_dispatch(self):
            logger.info_once(
                "Kimi K3: %d online MXFP8 linears use the CuTe DSL (<= 16 rows) / "
                "cuBLASLt MXFP8 dispatch.",
                switched,
            )
        from .specialized_prefill import gemm_tactics

        if pinned := gemm_tactics.enable(self):
            logger.info_once(
                "Kimi K3: %d BF16 linears use pinned cuBLASLt tactics at >= %d rows.",
                pinned,
                gemm_tactics.MIN_ROWS,
            )
        if any(
            takes_mxfp8_activation(m)
            for m in self.modules()
            if isinstance(m, LinearBase)
        ):
            # Build the fused MXFP8 activation producers at load, not mid-step.
            mxfp8_act.warmup()
            if any(
                any(layer._attn_res_mxfp8_capabilities())
                or layer._sp_mxfp8_gather_capable()
                for layer in self.modules()
                if isinstance(layer, KimiDecoderLayer)
            ):
                prefill_attn_res_mxfp8.warmup()
            logger.info_once(
                "Kimi K3: online-MXFP8 projections take fused MXFP8 activations."
            )
        # The specialized decode packs the DSA projections its own way.
        if self.specialized_decode_config is None:
            for module in self.model.modules():
                if isinstance(module, KimiK3DSAAttention):
                    module.pack_projections()
        # MegaMoE wide-EP kernel header / expert weight placement (env-selected).
        from .megamoe import prepare_megamoe
        from .megamoe_rubin.experts import prepare_rubin_megamoe

        # Expert DWDP (two engines): the non-expert DWDP below goes first, so no die
        # holds the full non-expert copy and every expert half at once.
        experts_late = dense_dwdp.edwdp.get() is not None
        if not experts_late:
            prepare_megamoe(self.model, self.vllm_config)
            # Rubin MegaMoE: kernel-layout experts + the collective warmup, before
            # any forward (profile run, CUDA graph capture).
            prepare_rubin_megamoe(self.model)
        # P/D prefill workers: the shared gate_up GEMM with the SiTU + MXFP8
        # epilogue (not beside a concurrent SM-capped MegaMoE (SP _shared_overlap:
        # the persistent full-grid GEMM is not capped to the reserved SMs)), and
        # the KDA q|k|v in_proj GEMM with the conv + SiLU in its epilogue, at every
        # layout whose shape the kernel supports (kda_conv_in_proj.unsupported_reason).
        from .specialized_prefill import kda_conv_in_proj, shared_situ

        prefill_worker = pd_prefill_worker(self.vllm_config)
        counts = {"packed_moe": 0, "fused_situ": 0, "fused_conv": 0}

        def prepare_layer(layer: nn.Module) -> None:
            """One decoder layer's weight transforms (under the non-expert DWDP, in
            the layer's load window: dense_dwdp.install)."""
            for module in layer.modules():
                if isinstance(module, KimiMoE):
                    module.pack_projections()
                    counts["packed_moe"] += module._packed_moe is not None
            for module in layer.modules() if prefill_worker else ():
                if (
                    isinstance(module, KimiMoE)
                    and not module._shared_overlap
                    and getattr(module, "_packed_moe", None) is None
                    and isinstance(module.shared_experts, KimiMLP)
                    and not module.shared_experts.shard_sequence_parallel
                    and shared_situ.supported(module.shared_experts)
                ):
                    situ = shared_situ.PrefillSharedSitu(module.shared_experts)
                    situ.prepare()
                    module.shared_experts._prefill_situ = situ
                    counts["fused_situ"] += 1
            for module in layer.modules() if prefill_worker else ():
                if not isinstance(module, KimiK3DeltaAttention):
                    continue
                reason = kda_conv_in_proj.unsupported_reason(module)
                if reason is not None:
                    logger.info_once(
                        "Kimi K3: prefill KDA in_proj keeps the separate conv: %s",
                        reason,
                    )
                    continue
                conv_in_proj = kda_conv_in_proj.PrefillKdaConvInProj(module)
                conv_in_proj.prepare(self.vllm_config.scheduler_config.max_num_seqs)
                module._prefill_conv_in_proj = conv_in_proj
                counts["fused_conv"] += 1

        # K3.1 DSA decode under DCP: the NCCL-free top-k exchange (on by policy).
        from .dsa_kernels.dcp_exchange.decode import build as build_dcpx

        self.dcpx = build_dcpx(self.model, self.vllm_config)
        # trtllm_gvr2: DeepSelect must rank every prefill chunk (checked);
        # trtllm_gvr2_nondet: its GVR V2 prefill engines, built before any forward.
        warm_prefill_indexer(self.model)
        # The per-layer transforms, under the non-expert DWDP each in its layer's
        # load window; after the DSA projections are packed, before the profile run.
        dense_dwdp.install(
            self.model.layers,
            self.model.start_layer,
            self.model.end_layer,
            prepare_layer,
            self.model_config.dtype,
        )
        if counts["packed_moe"]:
            logger.info_once(
                "Kimi K3: %d MoE layers run the packed MXFP8 shared/latent GEMMs.",
                counts["packed_moe"],
            )
        if counts["fused_situ"]:
            logger.info_once(
                "Kimi K3: %d prefill shared experts run the gate_up GEMM with the "
                "SiTU + MXFP8 epilogue (sm_107).",
                counts["fused_situ"],
            )
        else:
            logger.info_once(
                "Kimi K3: prefill shared experts keep the separate SiTU: %s",
                "not a P/D prefill worker"
                if not prefill_worker
                else "unsupported layout or weights",
            )
        if not prefill_worker:
            logger.info_once(
                "Kimi K3: prefill KDA in_proj keeps the separate conv: "
                "not a P/D prefill worker"
            )
        if counts["fused_conv"]:
            logger.info_once(
                "Kimi K3: %d prefill KDA layers run the q|k|v in_proj GEMM with the "
                "conv + SiLU epilogue (sm_107).",
                counts["fused_conv"],
            )
        if experts_late:
            prepare_megamoe(self.model, self.vllm_config)
            prepare_rubin_megamoe(self.model)

    def build_specialized_decode(self) -> None:
        from .dsa_kernels.dsa_core.decode import build as build_dsa_core
        from .specialized_decode.runtime.decode import SpecializedDecode

        # The DSA sparse core (on by policy where it can run: dsa_core.decode.build),
        # which the DSA layers' PreparedDsa picks up from their indexers.
        build_dsa_core(self.model, self.specialized_decode_config)
        self.specialized_decode = SpecializedDecode(
            self.model, self.specialized_decode_config
        )
        # From here on forward() returns normalized hidden states on every step: the
        # specialized decode's last AttnRes applies the norm, and forward() applies it
        # to each step that falls back to the stock forward (prefills, larger batches,
        # PIECEWISE steps).
        self.needs_norm_in_logits = False
        from .specialized_decode.runtime.verify import VerifyArgmax as VerifyLogits

        self.verify_logits = VerifyLogits.build(self.vllm_config, self)
        if self.verify_logits is not None:  # Rubin, speculating
            from .specialized_decode.runtime.sampled_verify import SampledVerify
            from .specialized_decode.runtime.warmup import warm_sampling

            self.sampled_verify = SampledVerify.build(self.vllm_config, self)
            # Compile the verification kernels now (every rank together), not at
            # the first decode step.
            warm_sampling(self.vllm_config, self)

    def get_model_state_cls(self) -> "type[ModelState]":
        from .model_state import KimiK3ModelState

        return KimiK3ModelState

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.specialized_decode is not None:
            return self.specialized_decode.embed_input_ids(input_ids)
        return self.model.embed_input_ids(input_ids)

    def make_empty_intermediate_tensors(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        return self.model.make_empty_intermediate_tensors(batch_size, dtype, device)

    def forward(  # type: ignore[override]
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        decode = self.specialized_decode
        if decode is not None:
            step = decode.plan(positions, intermediate_tensors)
            if step is not None:
                return decode.run(step, input_ids, inputs_embeds)
        if is_forward_context_available():
            from vllm.models.kimi_k3.nvidia.specialized_prefill import indexer_step

            indexer_step.prefetch_host(get_forward_context().attn_metadata)
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs
        )
        if not self.needs_norm_in_logits:
            assert decode is not None
            hidden_states = decode.normalize(hidden_states)
        return hidden_states

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, ...]:
        return MambaStateDtypeCalculator.kda_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> MambaStateShapes:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.kda_state_shape(
            tp_size,
            hf_config.linear_attn_config["num_heads"],
            hf_config.linear_attn_config["head_dim"],
            conv_kernel_size=hf_config.linear_attn_config["short_conv_kernel_size"],
            num_spec=num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.kda_state_copy_func()

    def enable_sp_local_aux(self) -> None:
        self.model.drafter_takes_local_aux = True
        if self.model.use_sequence_parallel:
            logger.info_once("The drafter projects sequence-shard aux states.")

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        # The model's final norm is applied here, not at the end of forward.
        if self.needs_norm_in_logits:
            hidden_states = self.model.norm(hidden_states, None)
        return self.logits_processor(self.lm_head, hidden_states)

    @property
    def prefetch_step_inputs(self) -> bool:
        """Opt-in for the runner's PrefetchedH2D (the step's small input arrays copied
        on a side stream ahead of the GPU): Rubin with the specialized decode only, so
        GB300 keeps the stock copies."""
        return (
            self.specialized_decode is not None
            and torch.cuda.get_device_capability() == (10, 7)
        )

    def get_top_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        """``compute_logits(hidden_states).argmax(-1)`` exactly (the lowest index among
        equal maxima) without gathering the full-vocabulary logits, for greedy
        spec-decode verification; every TP rank calls it together. None: no fast path
        for this call, the runner keeps the stock sampler."""
        if self.verify_logits is None:
            return None
        if self.needs_norm_in_logits:
            hidden_states = self.model.norm(hidden_states, None)
        return self.verify_logits.top_tokens(
            self.logits_processor, self.lm_head, hidden_states
        )

    def verify_sampled(
        self,
        hidden_states: torch.Tensor,
        steps: int,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """``RejectionSampler.rejection_sample``'s result for a decode batch of greedy
        and temperature-1 requests with ``steps`` drafts each, from this rank's shard of
        the target logits (``runtime.sampled_verify``); every TP rank calls it together.
        None: no fast path for this call (nothing launched)."""
        verify = self.sampled_verify
        if (
            verify is None
            or self.verify_logits is None
            or not verify.takes(hidden_states.shape[0])
        ):
            return None
        if self.needs_norm_in_logits:
            hidden_states = self.model.norm(hidden_states, None)
        logits = self.verify_logits.shard_logits(
            self.logits_processor, self.lm_head, hidden_states
        )
        return verify.run(logits, steps, **kwargs)

    def restrict_local_expert_ids(self, local_expert_ids):
        """Loader hook: skip the expert weights this rank does not hold (expert
        DWDP keeps half of the EP-local experts per layer)."""
        from .megamoe_rubin.experts import restrict_local_expert_ids

        return restrict_local_expert_ids(self, local_expert_ids)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)


@MULTIMODAL_REGISTRY.register_processor(
    KimiK3MultiModalProcessor,
    info=KimiK3ProcessingInfo,
    dummy_inputs=KimiK3DummyInputsBuilder,
)
class KimiK3ForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsEncoderCudaGraph,
    SupportsPP,
    SupportsQuant,
    SupportsEagle3,
    HasInnerState,
    IsHybrid,
    SupportsReplaySSM,
    GQADraftKVCacheGrouping,
):
    """Kimi-K3 model with Kimi-K2.5 vision and KimiLinear text."""

    supports_encoder_tp_data = True
    supports_mm_device_do_normalize = True

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "language_model.layers.": "language_model.model.layers.",
            "mm_projector.proj.0": "mm_projector.linear_1",
            "mm_projector.proj.2": "mm_projector.linear_2",
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality == "image":
            return "<|kimi_image_placeholder|>"
        raise ValueError(f"Unsupported modality: {modality}")

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        model_config = vllm_config.model_config
        config: KimiK3Config = model_config.hf_config
        self.config = config
        self.model_config = model_config
        quant_config = vllm_config.quant_config

        multimodal_config = model_config.multimodal_config
        assert multimodal_config is not None
        self.use_data_parallel = is_vit_use_data_parallel(
            config.vision_config.num_attention_heads
        )
        self.hidden_size = config.text_config.hidden_size
        self.device = current_platform.current_device()

        with self._mark_tower_model(vllm_config, "image"):
            self.vision_tower = MoonViT3dPretrainedModel(
                config.vision_config,
                quant_config=self._maybe_ignore_quant_config(quant_config),
                input_norm=build_mm_input_norm(vllm_config.model_config),
                prefix=maybe_prefix(prefix, "vision_tower"),
            )
            if is_meta_module(self.vision_tower):
                pass
            elif self._maybe_ignore_quant_config(quant_config) is not None:
                self.vision_tower = self.vision_tower.to(device=self.device)
            else:
                self.vision_tower = self.vision_tower.to(
                    device=self.device, dtype=model_config.dtype
                )

            vision_attn = self.vision_tower.encoder.blocks[0].attn
            if vision_attn.is_flash_attn_backend and vision_attn._fa_version == 4:
                from vllm.models.kimi_k3.nvidia.ops.vision_fa4_warmup import (
                    KimiK3VisionFA4WarmupConfig,
                    register_kimi_k3_vision_fa4_warmup,
                )

                merge_height, merge_width = config.vision_config.merge_kernel_size
                mm_config = model_config.get_multimodal_config()
                assert mm_config is not None
                register_kimi_k3_vision_fa4_warmup(
                    KimiK3VisionFA4WarmupConfig(
                        num_heads=vision_attn.num_heads,
                        head_dim=vision_attn.head_size,
                        dtype=vision_attn.dtype,
                        max_batch_size=(
                            vllm_config.scheduler_config.max_num_seqs
                            * mm_config.get_limit_per_prompt("image")
                        ),
                        max_seqlen=(
                            vllm_config.scheduler_config.max_num_encoder_input_tokens
                            * merge_height
                            * merge_width
                        ),
                    )
                )

            self.mm_projector = KimiK25MultiModalProjector(
                config=config.vision_config,
                out_hidden_size=config.text_config.hidden_size,
                use_data_parallel=self.use_data_parallel,
                quant_config=self._maybe_ignore_quant_config(quant_config),
                prefix=maybe_prefix(prefix, "mm_projector"),
            )
            if not is_meta_module(self.mm_projector):
                self.mm_projector = self.mm_projector.to(
                    device=self.device, dtype=model_config.dtype
                )

        self.quant_config = quant_config
        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=["KimiLinearForCausalLM"],
            )
        self.make_empty_intermediate_tensors = (  # type: ignore[method-assign]
            self.language_model.make_empty_intermediate_tensors
        )
        self.media_placeholder: int = self.config.media_placeholder_token_id

    def process_weights_after_loading(self) -> None:
        self.language_model.process_weights_after_loading()

    # -- SupportsEncoderCudaGraph protocol methods --

    def get_encoder_cudagraph_config(self):
        from vllm.v1.worker.encoder_cudagraph_defs import EncoderCudaGraphConfig

        return EncoderCudaGraphConfig(
            modalities=["image"],
            buffer_keys=[
                "pixel_values",
                "pos_embeds",
                "rope_freqs_cis",
                "cu_seqlens",
                "max_seqlen",
                "sequence_lengths",
                "merge_gather_idx",
            ],
            out_hidden_size=self.hidden_size,
        )

    def get_encoder_cudagraph_budget_range(
        self, vllm_config: VllmConfig
    ) -> tuple[int, int]:
        min_budget = 64
        max_budget = min(
            vllm_config.scheduler_config.max_num_batched_tokens,
            self.model_config.max_model_len,
        )
        return min_budget, max_budget

    @staticmethod
    def _get_grid_thws(mm_kwargs: dict[str, Any]) -> list[list[int]]:
        grid_thws = mm_kwargs["grid_thws"]
        if not isinstance(grid_thws, list):
            grid_thws = grid_thws.tolist()
        return grid_thws

    @staticmethod
    def _get_pixel_values(mm_kwargs: dict[str, Any]) -> torch.Tensor:
        pixel_values = mm_kwargs["pixel_values"]
        if isinstance(pixel_values, list):
            pixel_values = torch.cat(pixel_values)
        if pixel_values.ndim in (3, 5):
            pixel_values = pixel_values.reshape(
                pixel_values.shape[0] * pixel_values.shape[1],
                *pixel_values.shape[2:],
            )
        return pixel_values

    def get_encoder_cudagraph_item_specs(self, mm_kwargs: dict[str, Any]):
        from vllm.v1.worker.encoder_cudagraph_defs import EncoderItemSpec

        kh, kw = self.config.vision_config.merge_kernel_size
        return [
            EncoderItemSpec(
                input_size=t * h * w,
                output_tokens=(h // kh) * (w // kw),
            )
            for t, h, w in self._get_grid_thws(mm_kwargs)
        ]

    def select_encoder_cudagraph_items(
        self, mm_kwargs: dict[str, Any], indices: list[int]
    ) -> dict[str, Any]:
        grid_thws = self._get_grid_thws(mm_kwargs)
        pixel_values = self._get_pixel_values(mm_kwargs)
        source_grid = mm_kwargs["grid_thws"]

        if not indices:
            empty_grid = (
                source_grid[:0] if isinstance(source_grid, torch.Tensor) else []
            )
            return {"pixel_values": pixel_values[:0], "grid_thws": empty_grid}

        patch_counts = [t * h * w for t, h, w in grid_thws]
        offsets = [0]
        for count in patch_counts:
            offsets.append(offsets[-1] + count)
        selected_pixel_values = torch.cat(
            [pixel_values[offsets[i] : offsets[i + 1]] for i in indices]
        )
        grid_device = (
            source_grid.device if isinstance(source_grid, torch.Tensor) else None
        )
        selected_grid = torch.tensor(
            [grid_thws[i] for i in indices],
            dtype=torch.long,
            device=grid_device,
        )
        return {"pixel_values": selected_pixel_values, "grid_thws": selected_grid}

    def prepare_encoder_cudagraph_capture_inputs(
        self,
        token_budget: int,
        max_batch_size: int,
        max_frames_per_batch: int,
        device: torch.device,
        dtype: torch.dtype,
        path: str = "default",
        axis_keys: tuple[Hashable, ...] | None = None,
    ):
        from vllm.v1.worker.encoder_cudagraph_defs import (
            EncoderCudaGraphCaptureInputs,
        )

        kh, kw = self.config.vision_config.merge_kernel_size
        per_item_output = (token_budget + max_batch_size - 1) // max_batch_size
        rope = self.vision_tower.encoder.rope_2d
        max_output_width = rope.max_width // kw
        max_output_height = rope.max_height // kh
        output_width = min(math.ceil(math.sqrt(per_item_output)), max_output_width)
        output_height = (per_item_output + output_width - 1) // output_width
        if output_height > max_output_height:
            output_height = max_output_height
            output_width = (per_item_output + output_height - 1) // output_height
        if output_width > max_output_width:
            raise ValueError(
                f"Encoder CUDA graph budget {token_budget} exceeds K3 RoPE "
                f"capacity for max_batch_size={max_batch_size}"
            )
        grid_thws = [
            [1, output_height * kh, output_width * kw] for _ in range(max_batch_size)
        ]

        patch_size: int | tuple[int, int] = self.config.vision_config.patch_size
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        total_patches = sum(t * h * w for t, h, w in grid_thws)
        pixel_values = torch.zeros(
            total_patches,
            3,
            patch_size[0],
            patch_size[1],
            device=device,
            dtype=self.vision_tower.patch_embed.input_norm.input_dtype or dtype,
        )
        metadata = self.vision_tower.prepare_encoder_cudagraph_metadata(
            grid_thws,
            max_batch_size=max_batch_size,
            max_seqlen_override=max(
                token_budget * kh * kw,
                max(t * h * w for t, h, w in grid_thws),
            ),
            device=device,
        )
        return EncoderCudaGraphCaptureInputs(
            values=metadata | {"pixel_values": pixel_values}
        )

    def prepare_encoder_cudagraph_replay_buffers(
        self,
        mm_kwargs: dict[str, Any],
        max_batch_size: int,
        max_frames_per_batch: int,
        path: str = "default",
    ):
        from vllm.v1.worker.encoder_cudagraph_defs import (
            EncoderCudaGraphReplayBuffers,
        )

        pixel_values = self._get_pixel_values(mm_kwargs)
        metadata = self.vision_tower.prepare_encoder_cudagraph_metadata(
            self._get_grid_thws(mm_kwargs),
            max_batch_size=max_batch_size,
            device=pixel_values.device,
        )
        return EncoderCudaGraphReplayBuffers(
            values=metadata | {"pixel_values": pixel_values}
        )

    def _project_encoder_features(self, image_features: torch.Tensor) -> torch.Tensor:
        projector_dtype = next(self.mm_projector.parameters()).dtype
        if image_features.dtype != projector_dtype:
            image_features = image_features.to(projector_dtype)
        output = self.mm_projector(image_features)
        return output.reshape(-1, output.shape[-1])

    def encoder_cudagraph_forward(
        self,
        values: dict[str, torch.Tensor],
        path: str = "default",
    ) -> torch.Tensor:
        pixel_values = values.pop("pixel_values")
        image_features = self.vision_tower(pixel_values, None, encoder_metadata=values)
        return self._project_encoder_features(image_features)

    def encoder_eager_forward(
        self,
        mm_kwargs: dict[str, Any],
        path: str = "default",
    ) -> torch.Tensor:
        image_features = self.vision_tower(
            self._get_pixel_values(mm_kwargs),
            self._get_grid_thws(mm_kwargs),
        )
        return self._project_encoder_features(torch.cat(image_features))

    def _maybe_ignore_quant_config(
        self, quant_config: QuantizationConfig | None
    ) -> QuantizationConfig | None:
        if isinstance(quant_config, compressed_tensors.CompressedTensorsConfig):
            return None
        return quant_config

    def _parse_and_validate_media_input(
        self, **kwargs: object
    ) -> KimiK25MediaPixelInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        grid_thws = kwargs.pop("grid_thws", None)
        if pixel_values is None:
            return None

        if isinstance(pixel_values, list):
            pixel_values = torch.cat(cast(list[torch.Tensor], pixel_values), dim=0)
        if not isinstance(pixel_values, torch.Tensor):
            raise TypeError(
                "pixel_values must be a tensor or a list of tensors, "
                f"got {type(pixel_values)}"
            )

        if len(pixel_values.shape) == 5 or len(pixel_values.shape) == 3:
            pixel_values = pixel_values.reshape(
                pixel_values.shape[0] * pixel_values.shape[1], *pixel_values.shape[2:]
            )

        assert isinstance(grid_thws, torch.Tensor), (
            f"expect grid_thws to be a tensor, got {type(grid_thws)}"
        )
        grid_thws = grid_thws.reshape(-1, grid_thws.shape[-1])
        assert grid_thws.ndim == 2 and grid_thws.size(1) == 3, (
            f"unexpected shape for grid_thws: {grid_thws.shape}"
        )

        return KimiK25MediaPixelInputs(
            type="pixel_values",
            pixel_values=pixel_values,
            grid_thws=grid_thws,
        )

    def _process_media_input(
        self, media_input: KimiK25MediaPixelInputs
    ) -> list[torch.Tensor]:
        media_features = vision_tower_forward(
            self.vision_tower,
            media_input["pixel_values"],
            media_input["grid_thws"],
            mm_projector=self.mm_projector,
            use_data_parallel=self.use_data_parallel,
        )
        return media_features

    def embed_multimodal(self, **kwargs: object) -> NestedTensors | None:
        media_input = self._parse_and_validate_media_input(**kwargs)
        if media_input is None:
            return None
        return self._process_media_input(media_input)

    def forward(  # type: ignore[override]
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if intermediate_tensors is not None:
            inputs_embeds = None
        return self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.language_model.compute_logits(hidden_states)

    def enable_sp_local_aux(self) -> None:
        self.language_model.enable_sp_local_aux()

    @property
    def prefetch_step_inputs(self) -> bool:
        return self.language_model.prefetch_step_inputs

    def get_top_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.get_top_tokens(hidden_states)

    def verify_sampled(
        self, hidden_states: torch.Tensor, steps: int, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        return self.language_model.verify_sampled(hidden_states, steps, **kwargs)

    def get_model_state_cls(self) -> "type[ModelState]":
        return self.language_model.get_model_state_cls()

    def copy_inputs_before_cuda_graphs(self, input_buffers, **kwargs):
        return self.language_model.mamba_cache.copy_inputs_before_cuda_graphs(
            input_buffers, **kwargs
        )

    def get_seqlen_agnostic_capture_inputs(self, batch_size: int):
        return self.language_model.mamba_cache.get_seqlen_agnostic_capture_inputs(
            batch_size
        )

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig):
        text_config = vllm_config.model_config.hf_config.text_config
        temp_vllm_config = vllm_config.with_hf_config(text_config)
        return KimiLinearForCausalLM.get_mamba_state_dtype_from_config(temp_vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):
        text_config = vllm_config.model_config.hf_config.text_config
        temp_vllm_config = vllm_config.with_hf_config(text_config)
        return KimiLinearForCausalLM.get_mamba_state_shape_from_config(temp_vllm_config)

    @classmethod
    def get_mamba_state_copy_func(cls):
        return KimiLinearForCausalLM.get_mamba_state_copy_func()

    @classmethod
    def get_packed_hybrid_block_size(
        cls,
        vllm_config: VllmConfig,
        attn_page_size_1_token: int,
        mamba_page_size: int,
        alignment: int,
    ) -> int | None:
        return packed_dsa_block_size(
            vllm_config, attn_page_size_1_token, mamba_page_size, alignment
        )

    def restrict_local_expert_ids(self, local_expert_ids):
        """Loader hook: skip the expert weights this rank does not hold (expert
        DWDP keeps half of the EP-local experts per layer)."""
        from .megamoe_rubin.experts import restrict_local_expert_ids

        return restrict_local_expert_ids(self, local_expert_ids)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
