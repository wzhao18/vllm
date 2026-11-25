# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``KimiK3MegaMoEExperts``: the Kimi K3 routed experts on the Rubin (SM107) MegaMoE
backend, ``--moe-backend kimi_k3_mega_moe``.

A drop-in for :class:`DeepGemmMegaMoEExperts` in ``KimiMoE`` (same constructor
arguments, parameter names / shapes / dtypes, weight loader, EPLB hooks and
``forward(hidden_states, topk_weights, topk_ids)``), running the vendored FlashInfer
sm107 CuTe DSL MegaMoE through :class:`RubinMegaMoE` (fused dispatch + FC1 + SiTU +
FC2 + BF16 combine on torch symmetric memory).

Differences from DeepGEMM that the caller sees:

- The checkpoint shards are loaded as stored (canonical MXFP4: E2M1 codes and UE8M0
  scales; or ModelOpt NVFP4: E2M1 codes, E4M3 block-16 scales and per-expert
  ``weight_scale_2`` / ``input_scale``; gate rows then up rows);
  :meth:`process_weights_after_loading` converts them to the kernel layout once
  (``prepare_rubin_megamoe`` runs it for every layer, then the collective warmup)
  and frees the checkpoint copy.
- The routed-expert precision follows the checkpoint (:func:`rubin_quant_kind`, no
  flag): an NVFP4 checkpoint runs NVFP4 at every size; an MXFP4 one runs MXFP8
  activations x MXFP4 weights up to :data:`RUBIN_MX_MAX_TOKENS_PER_RANK` tokens per
  rank, else an exact MXFP4 -> NVFP4 conversion at load
  (:class:`NonRepresentableError` when a weight is not exactly representable;
  nothing is re-quantized).
- One compiled bucket sized to ``max_num_tokens``, plus, on decoding (not P/D
  prefill) inference-kernel servers whose capacity exceeds the token_comm
  decode-landing limit, one decode bucket at that limit
  (:func:`rubin_bucket_capacities`); each forward runs the smallest bucket holding
  its EP-uniform token count (:meth:`KimiK3MegaMoEExperts.select_bucket`).
- The shared expert is never fused (the Rubin kernel has no shared-expert path).
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar

import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.distributed import get_ep_group, get_tensor_model_parallel_world_size
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import (
    MEGA_MOE_MXFP4,
    MEGA_MOE_NVFP4,
    MegaMoEActivation,
    MegaMoEWeightFormat,
)
from vllm.model_executor.models.utils import extract_layer_index
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

from . import compact_scales, dwdp
from .config import (
    DEDUP_VARIANTS,
    SHIPPED_KNOBS,
    QuantKind,
    RubinBucket,
    RubinKnobs,
    RubinMoeConfig,
    RubinMoeProblem,
    Variant,
    make_buckets,
)
from .session import ep_mean_tokens
from .staging import EplbStage
from .weights import MX_BLOCK, NV_BLOCK, RubinMoeWeights

if TYPE_CHECKING:
    from vllm.config import VllmConfig

    from .runtime import RubinMegaMoE

logger = init_logger(__name__)

KIMI_K3_MEGA_MOE_BACKEND = "kimi_k3_mega_moe"

# Format policy: the largest tokens per rank (the value the layer's workspace is
# sized for) that runs MXFP8 activations x MXFP4 weights; larger servers run NVFP4
# (exact MXFP4 -> NVFP4 at load). Measured crossover on VR200, EP16, Kimi K3.1
# shapes (bench_megamoe_rubin.py, CUDA graph): MX is as fast or faster up to 256
# tokens/rank (256 tok: NVFP4 132.3 us vs MX 131.9 us) and NVFP4 wins from 1024 on
# (1024 tok: NVFP4 223.9 us vs MX 251.7 us); 512 sits between and keeps the
# checkpoint precision.
RUBIN_MX_MAX_TOKENS_PER_RANK = 512

# Kernel variant: the generation-phase kernel up to its 1024-token limit
# (RubinBucket), the inference kernel above. Shipped VR knob records at the EP16
# 1024-token capacity: genphase NVFP4 346.6 us vs inference 413.2 us, MX 381.8 vs
# 477.7 us.
RUBIN_GENPHASE_MAX_TOKENS_PER_RANK = 1024

# Decode bucket: the kernel's token_comm decode paths (S0 dynamic token_in claim, S1
# decode landing) are traced in only when capacity x top_k <= this many routes
# (token_comm ``decode_landing_max_routes`` / ``token_in_dynamic_claim_max_routes``;
# a unit test keeps them equal). A decoding server sized above it (C512 decode: MNBT
# 8192 / TP 4 = 2048 tokens/rank) also compiles a bucket at this limit (4096 / top-16
# = 256 tokens per rank) with the same kernel variant, quant kind and knob record as
# its main bucket, so decode steps (52-128 tokens/rank at C512) run with both paths
# on and a 256-row stager grid. Same math, bitwise-identical outputs. Measured on
# VR200 (EP16 per-rank shape on EP4: 56 local experts, top-16, NVFP4, CUDA graph,
# stage+kernel p50) at 52 tokens/rank: 2048 bucket 128.3 us, 256 bucket 110.4 us
# (64: 128.9 / 110.9, 128: 130.0 / 113.6, 256: 135.1 / 122.7). The genphase kernel
# is not used for it (139.7-225.5 us at that shape). P/D prefill workers (C512
# prefill: SP shard of 8192 tokens/rank at EP16 on the i8192 record; C256 prefill:
# token shard of 2048 at EP8 on the i2048 record with the lowest p50, the
# phase-interleaved N256 one) run prefill chunks, where the bucket was not measured,
# and keep their one bucket (no extra compile or arena).
RUBIN_DECODE_BUCKET_MAX_ROUTES = 4096

# Wide EP (RUBIN_INFERENCE_ONLY_EP_SIZES, the D3-32 decode role: SP shards of at most
# 128 tokens per rank up to r32): the decode bucket is 128 tokens per rank with its own
# record (N128 tile). Measured on the EP32 per-GPU shape (1 GPU, 28 experts, NVFP4,
# FlashInfer a368c5f66, hint 64, CUDA graph): capacity 128 / N128 at 32 / 96 / 128
# live tokens 44.9 / 47.8 / 50.7 us vs capacity 256 / N256 51.0 / 52.8 / 55.9 us.
RUBIN_WIDE_EP_DECODE_TOKENS = 128

# EP32 decode: the <=128-token bucket runs the refreshed GenPhase kernel (src_gp) on its
# decode scheduler, die-aware (expert_half) on decode servers. Measured at real EP32
# (8 nodes x 4 VR200, 896 experts, NVFP4, 128-token capacity, random routing, CUDA graph
# of the kernel, p50 of the slowest rank, die-local weights; D-MOE runs/D-MOE/out/ep32
# g32inf / g32gpa): inference kernel 67.8 / 70.5 / 68.7 us vs GenPhase 62.7 / 63.8 /
# 65.6 us at 32 / 96 / 128 tokens per rank (rel-L2 vs the bf16 reference 0.166 both).
# EP28 keeps the inference kernel (not measured).
RUBIN_GENPHASE_DECODE_EP_SIZES = (32,)

# Prefill-sized NVFP4 inference buckets (at least RUBIN_DEDUP_MIN_TOKENS tokens per rank:
# the P3-16 8192-token sub-call) run the refreshed inference kernel with the deduplicated
# combine and two epilogue warpgroups (variant inference_dedup_wg2, token back from
# standalone warps): one BF16 row per (owner rank, token) crosses NVLink instead of one
# per route (EP16 top-16: ~10.3 instead of 16 rows per token), and the FC1/FC2 epilogue
# (the compute-side bound) runs on 8 warps. Measured on VR200 (NVFP4, 56 experts/GPU,
# 2 x 8192 tokens/rank, random routing): EP4 2110 (shipped) -> 1893 (dedup) -> 1736 us.
RUBIN_DEDUP_VARIANT = "inference_dedup_wg2"
RUBIN_DEDUP_TOKEN_BACK_MODE = "standalone_warps"
RUBIN_DEDUP_MIN_TOKENS = 4096
# Cluster (2, 1): 106 CTA pairs cover all 212 SMs (cluster (4, 1) packs 46 clusters = 184 CTAs); dynamic B
# gather on the pair. EP1 (28 experts, NVFP4, fused MXFP8 shared, same session ABBA, bitwise-equal outputs):
# 55.4 / 56.4 / 59.6 vs 56.7 / 58.2 / 60.9 us at 32 / 96 / 128 tokens per rank.
RUBIN_GENPHASE_DECODE_KNOBS = RubinKnobs(
    mma_tiler_mnk=(256, 128, 256),
    cluster_shape_mn=(2, 1),
    fallback_cluster_shape_mn=None,
    schedule_policy=("decode", None),
    work_id_mode="atomic_counter",
    fc2_use_bulk=True,
    fc2_tma_stages=2,
    epi_flag_batches=(1, 4),
    token_in_flag_batch=1,
    token_back_mode="epi_warps",
    reduce_topk_in_kernel=False,
)

# The decode bucket with the fused MXFP8 shared expert (enable_fused_shared): the decode
# scheduler (die-aware on decode servers) with the shared tile stream ahead of the routed
# sources (VENDOR.md S-SH): shared FC1 streams under the dispatch.
RUBIN_GENPHASE_FUSED_SHARED_KNOBS = RUBIN_GENPHASE_DECODE_KNOBS


def rubin_bucket_capacities(
    max_tokens_per_rank: int, top_k: int, *, prefill_worker: bool, world_size: int = 16
) -> tuple[int, ...]:
    """Ascending token capacities per rank of a server's compiled buckets
    (automatic, no flag): ``max_tokens_per_rank``, preceded by the decode bucket
    (``RUBIN_DECODE_BUCKET_MAX_ROUTES // top_k``) on decoding (not P/D prefill,
    ``prefill_worker``) inference-kernel servers sized above it."""
    if top_k <= 0:
        raise ValueError(f"top_k must be positive: {top_k}")
    decode = RUBIN_DECODE_BUCKET_MAX_ROUTES // top_k
    if world_size in RUBIN_INFERENCE_ONLY_EP_SIZES:
        decode = min(decode, RUBIN_WIDE_EP_DECODE_TOKENS)
    if (
        not prefill_worker
        and rubin_kernel_variant(max_tokens_per_rank, world_size) == "inference"
        and 0 < decode < max_tokens_per_rank
    ):
        return (decode, max_tokens_per_rank)
    return (max_tokens_per_rank,)


# Persistent-grid cap (SMs) when the caller runs the BF16 shared expert concurrently
# on a side stream (KimiMoE under sequence parallelism): the remaining SMs serve the
# shared expert's GEMMs. Measured on VR200 (EP16 decode shape emulated on EP4: 56
# experts/rank, top-16; CUDA graph, stage + kernel, per layer): the shipped EP16
# NVFP4 inference 2048 record (cluster 4 + fallback 2) at 32 / 52 / 64 tokens/rank
# runs 125.2 / 126.6 / 126.9 us on the full grid and 123.9 / 124.3 / 124.6 us capped
# at 148 SMs without fallback clusters; MegaMoE + shared expert + latent up-projection
# drops from 186.4 / 198.3 / 199.4 us serial to 168.3 / 183.3 / 179.2 us overlapped
# (bitwise-equal outputs).
RUBIN_SHARED_OVERLAP_MAX_SMS = 148

# P/D prefill workers under sequence parallelism: steps of at most this many tokens per
# rank keep the decode-style SM-capped MegaMoE + side-stream shared expert (a small
# bucket capped at RUBIN_SHARED_OVERLAP_MAX_SMS); larger steps run MegaMoE on the full
# grid and the shared expert after it. Measured on VR200, TP8+DCP8 NVFP4 prefill EP8
# (rows per rank, capped-overlap -> full-grid serial step): 128 rows 45.24 -> 46.63 ms,
# 512 rows 77.72 -> 78.09, 1024 rows 125.10 -> 126.37 (all worse on the full grid);
# 2048 rows 225.26 -> 218.22, 4096 rows 405.43 -> 385.37 (better). The crossover lies
# between 1024 and 2048 rows; the small bucket ends at 1024.
RUBIN_PREFILL_SHARED_OVERLAP_MAX_TOKENS = 1024


# additional_config["megamoe_quant"] -> quant kind (FlashInfer's kind names).
_MEGAMOE_QUANT: dict[str, QuantKind] = {
    "nvfp4": "nvfp4",
    "mxfp4_mxfp8": "mxfp4_mxfp8",
    "mxfp4_mxfp4": "mxfp4",
}


def rubin_quant_kind(
    max_tokens_per_rank: int,
    weight_format: MegaMoEWeightFormat = MEGA_MOE_MXFP4,
    requested: str = "auto",
) -> QuantKind:
    """Routed-expert precision of a server whose layer sees at most
    ``max_tokens_per_rank`` tokens per forward: automatic unless ``requested``
    (``additional_config["megamoe_quant"]``: ``auto`` | ``nvfp4`` | ``mxfp4_mxfp8``
    | ``mxfp4_mxfp4``; the exact NVFP4 conversion, or the MXFP4 weights as in the
    checkpoint with MXFP8 or MXFP4 activations). An NVFP4 checkpoint runs the NVFP4
    kernel at every size (it has no exact MXFP4 form)."""
    if max_tokens_per_rank <= 0:
        raise ValueError(f"max_tokens_per_rank must be positive: {max_tokens_per_rank}")
    if weight_format is MEGA_MOE_NVFP4:
        if requested not in ("auto", "nvfp4"):
            raise ValueError(f"megamoe_quant {requested!r} needs MXFP4 weights.")
        return "nvfp4"
    if weight_format is not MEGA_MOE_MXFP4:
        raise ValueError(f"no Rubin MegaMoE kernel for {weight_format.name!r} weights.")
    if requested != "auto":
        if requested not in _MEGAMOE_QUANT:
            raise ValueError(
                f"megamoe_quant must be auto or one of {tuple(_MEGAMOE_QUANT)}, got "
                f"{requested!r}."
            )
        return _MEGAMOE_QUANT[requested]
    if max_tokens_per_rank <= RUBIN_MX_MAX_TOKENS_PER_RANK:
        return "mxfp4_mxfp8"
    return "nvfp4"


# EP sizes whose shipped records are the inference kernel only (FlashInfer a368c5f66
# default knobs measured at EP28 / EP32, moe_ep32): the generation-phase kernel has no
# record there, so these servers run the inference kernel at every capacity.
RUBIN_INFERENCE_ONLY_EP_SIZES = (28, 32)


def rubin_kernel_variant(max_tokens_per_rank: int, world_size: int = 16) -> Variant:
    if max_tokens_per_rank <= 0:
        raise ValueError(f"max_tokens_per_rank must be positive: {max_tokens_per_rank}")
    if world_size in RUBIN_INFERENCE_ONLY_EP_SIZES:
        return "inference"
    if max_tokens_per_rank <= RUBIN_GENPHASE_MAX_TOKENS_PER_RANK:
        return "genphase"
    return "inference"


@functools.cache
def select_rubin_bucket(
    problem: RubinMoeProblem, world_size: int, max_tokens: int
) -> tuple[RubinBucket, tuple[str, ...]]:
    """The one compiled bucket of a server: capacity ``max_tokens`` per rank, the
    :func:`rubin_kernel_variant` and the shipped VR knob record for it (FlashInfer
    lookup order; :class:`KnobLookupError` when the EP size / variant / precision has
    no record)."""
    buckets, notes = make_buckets(
        SHIPPED_KNOBS,
        problem=problem,
        world_size=world_size,
        points=[(rubin_kernel_variant(max_tokens, world_size), max_tokens, None)],
    )
    return buckets[0], tuple(notes)


def select_rubin_buckets(
    problem: RubinMoeProblem,
    world_size: int,
    max_tokens: int,
    *,
    prefill_worker: bool,
    knob_world_size: int | None = None,
) -> tuple[tuple[RubinBucket, ...], tuple[str, ...]]:
    """Every compiled bucket of a server, ascending (:func:`rubin_bucket_capacities`):
    the main bucket of :func:`select_rubin_bucket`, and below it the decode bucket,
    which keeps the main bucket's variant and knob record (only the capacity
    differs). Raises (``ValueError``) when the kernel's admission rules refuse that
    record at the decode capacity.

    ``knob_world_size`` (benches only; servers leave it None): pick capacities and
    records as at that EP size while the kernel runs at ``world_size``. A 1-node
    slice that emulates EP32 per-GPU work (28 active experts per rank) passes 32 so
    its MegaMoE runs the EP32 configuration (128-token decode bucket on its own
    record, phase-interleaved, die-local) instead of the EP4 prefill records."""
    if knob_world_size is not None:
        world_size = knob_world_size
    main, notes = select_rubin_bucket(problem, world_size, max_tokens)
    caps = rubin_bucket_capacities(
        max_tokens, problem.top_k, prefill_worker=prefill_worker, world_size=world_size
    )
    if world_size in RUBIN_INFERENCE_ONLY_EP_SIZES:
        # wide EP: the decode bucket takes its own record (per-shape tile)
        buckets = [select_rubin_bucket(problem, world_size, cap)[0] for cap in caps[:-1]]
        own = "its own record"
    else:
        buckets = [RubinBucket(main.variant, cap, main.knobs) for cap in caps[:-1]]
        own = f"knobs of the {main.max_tokens} bucket"
    if world_size in RUBIN_GENPHASE_DECODE_EP_SIZES and problem.quant_kind == "nvfp4":
        buckets = [_genphase_decode(b) for b in buckets]
        main = _genphase_decode(main)
    if problem.quant_kind == "nvfp4":
        buckets = [_prefill_dedup(b) for b in buckets]
        main = _prefill_dedup(main)
    for bucket in (*buckets, main):
        RubinMoeConfig.make(problem, bucket, rank=0, world_size=world_size)
    extra = tuple(f"[{b.variant} {b.max_tokens}] decode bucket, {own}" for b in buckets)
    return (*buckets, main), notes + extra


def _prefill_dedup(bucket: RubinBucket) -> RubinBucket:
    """A prefill-sized inference bucket on the refreshed dedup-combine kernel (same
    tile/schedule knobs, token back from standalone warps); others unchanged."""
    if (
        bucket.variant != "inference"
        or bucket.die_sm_mask32 is not None
        or bucket.max_tokens < RUBIN_DEDUP_MIN_TOKENS
    ):
        return bucket
    return dataclasses.replace(
        bucket,
        variant=RUBIN_DEDUP_VARIANT,
        knobs=dataclasses.replace(
            bucket.knobs, token_back_mode=RUBIN_DEDUP_TOKEN_BACK_MODE
        ),
    )


def _genphase_decode(bucket: RubinBucket) -> RubinBucket:
    """A wide-EP bucket of at most :data:`RUBIN_WIDE_EP_DECODE_TOKENS` tokens per rank
    on the refreshed GenPhase kernel (src_gp, decode scheduler; die-aware
    ``expert_half`` once the bucket is die-local); larger buckets unchanged."""
    if bucket.variant != "inference" or bucket.max_tokens > RUBIN_WIDE_EP_DECODE_TOKENS:
        return bucket
    return dataclasses.replace(
        bucket, variant="genphase", knobs=RUBIN_GENPHASE_DECODE_KNOBS
    )


# The forward's DPMetadata (held, so its id is never reused) and whether its DP batch
# sizes are equal: computed once per forward, not once per MoE layer.
_dp_uniform_memo: tuple[object, bool] | None = None


def _dp_batches_uniform(dp_metadata) -> bool:
    """Whether every DP rank runs the same token count this forward. A forward
    context builds a new DPMetadata per forward (its counts are never written in
    place), so the answer is cached by object identity; host-only, so a CUDA graph
    capture bakes in the bucket of its own padded size."""
    global _dp_uniform_memo
    memo = _dp_uniform_memo
    if memo is not None and memo[0] is dp_metadata:
        return memo[1]
    counts = dp_metadata.num_tokens_across_dp_cpu
    uniform = int(counts.min()) == int(counts.max())
    _dp_uniform_memo = (dp_metadata, uniform)
    return uniform


# The forward's DPMetadata (held), the sequence-parallel size and the EP-mean live
# tokens per rank: computed once per forward, not once per MoE layer.
_ep_mean_memo: tuple[object, int, int] | None = None


def _ep_mean_tokens(dp_metadata, sp_size: int) -> int:
    """:func:`ep_mean_tokens` of the forward's DP batch sizes (host-only, identical on
    every rank), cached per DPMetadata object."""
    global _ep_mean_memo
    memo = _ep_mean_memo
    if memo is not None and memo[0] is dp_metadata and memo[1] == sp_size:
        return memo[2]
    mean = ep_mean_tokens(dp_metadata.num_tokens_across_dp_cpu, sp_size)
    _ep_mean_memo = (dp_metadata, sp_size, mean)
    return mean


def check_kimi_k3_mega_moe_supported() -> None:
    """Raise unless this process runs on an SM107 (VR200) GPU."""
    if not current_platform.is_cuda() or not current_platform.is_device_capability(
        (10, 7)
    ):
        raise NotImplementedError(
            f"moe_backend={KIMI_K3_MEGA_MOE_BACKEND!r} runs the Rubin MegaMoE kernels "
            "and requires an SM107 (VR200) GPU; use deep_gemm_mega_moe on SM100."
        )


class KimiK3MegaMoEExperts(nn.Module):
    """Routed experts of one Kimi K3 MoE layer on the Rubin MegaMoE backend.

    Args (as :class:`DeepGemmMegaMoEExperts`):
        num_experts: Physical experts across EP ranks (no redundant experts: the
            kernel places expert ``g`` on rank ``g // num_local_experts``).
        weight_format: Checkpoint precision, ``MEGA_MOE_MXFP4`` or
            ``MEGA_MOE_NVFP4`` (ModelOpt export).
        shared_experts: Must be ``None``; the shared expert runs outside the launch.
        max_num_tokens: Most tokens one rank passes to ``forward``; sizes the bucket
            and picks the precision. Defaults to ``max_num_batched_tokens``.

    """

    _runtimes: ClassVar[dict[tuple, RubinMegaMoE]] = {}

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        num_experts: int,
        num_local_experts: int,
        experts_start_idx: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        activation: MegaMoEActivation,
        weight_format: MegaMoEWeightFormat = MEGA_MOE_MXFP4,
        shared_experts: nn.Module | None = None,
        num_logical_experts: int | None = None,
        max_num_tokens: int | None = None,
        prefix: str = "",
    ):
        super().__init__()
        check_kimi_k3_mega_moe_supported()
        if weight_format not in (MEGA_MOE_MXFP4, MEGA_MOE_NVFP4):
            raise NotImplementedError(
                "Rubin MegaMoE loads MXFP4 or NVFP4 expert checkpoints, not "
                f"{weight_format.name!r}."
            )
        if activation.name != "situ" or activation.beta <= 0:
            raise NotImplementedError(
                "Rubin MegaMoE runs SiTU with a positive linear beta, got "
                f"{activation}."
            )
        if shared_experts is not None:
            raise NotImplementedError(
                "Rubin MegaMoE does not fuse the shared expert; pass it to forward()."
            )
        # Redundant experts: ``num_experts`` physical slots hold ``num_logical_experts``
        # logical experts; physical slot ``p`` holds logical ``p % num_logical`` (the
        # default EPLB placement), still linear over the EP ranks. Routed ids are
        # logical; without an EPLB map they address the primary slots ``[0, L)``, so
        # the extra slots only pad the expert count to a multiple of the EP size.
        if num_logical_experts is None:
            num_logical_experts = num_experts
        if not 0 < num_logical_experts <= num_experts:
            raise ValueError(
                f"Rubin MegaMoE: num_logical_experts {num_logical_experts} must be in "
                f"(0, {num_experts}]."
            )
        ep_group = get_ep_group()
        ep_size, ep_rank = ep_group.world_size, ep_group.rank_in_group
        if (
            num_local_experts * ep_size != num_experts
            or experts_start_idx != ep_rank * num_local_experts
        ):
            raise ValueError(
                f"Rubin MegaMoE needs linear expert placement: rank {ep_rank} of EP"
                f"{ep_size} must own experts [{ep_rank * num_local_experts}, "
                f"{(ep_rank + 1) * num_local_experts}) of {num_experts}, got start "
                f"{experts_start_idx} x {num_local_experts}."
            )

        self.prefix = prefix
        self.capture_fn: Callable[[torch.Tensor], None] | None = None
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.experts_start_idx = experts_start_idx
        self.experts_end_idx = experts_start_idx + num_local_experts
        self.num_logical_experts = num_logical_experts
        self.num_redundant_experts = num_experts - num_logical_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_num_tokens = (
            max_num_tokens
            if max_num_tokens is not None
            else vllm_config.scheduler_config.max_num_batched_tokens
        )
        self.weight_format = weight_format
        self.activation = activation
        self.num_shared_experts = 0
        self.shared_spec = None
        self.eplb_state = EplbLayerState()
        self.ep_size = ep_size
        additional_config = getattr(vllm_config, "additional_config", None)
        self.quant_kind = rubin_quant_kind(
            self.max_num_tokens,
            weight_format,
            additional_config.get("megamoe_quant", "auto")
            if isinstance(additional_config, dict)
            else "auto",
        )
        self.problem = RubinMoeProblem(
            num_experts=num_experts,
            top_k=top_k,
            hidden=hidden_size,
            intermediate=intermediate_size,
            situ_beta=float(activation.alpha),
            situ_linear_beta=float(activation.beta),
            quant_kind=self.quant_kind,
        )
        from ..model import pd_prefill_worker  # (model.py imports this lazily)

        self.prefill_worker = pd_prefill_worker(vllm_config)
        # NVFP4 scale planes kept compact, read by the kernel (prepare_rubin_megamoe
        # compacts them when every pair is equal: compact_scales).
        self.compact_scales = compact_scales.wanted(vllm_config, self.prefill_worker)
        model_config = getattr(vllm_config, "model_config", None)
        text_config = getattr(model_config, "hf_text_config", None)
        # MoE layers of the model: an upper bound for the die-local budget check
        # (1 without a model config, e.g. a standalone experts module).
        self.num_moe_layers = getattr(text_config, "num_hidden_layers", 1)
        self.buckets, notes = select_rubin_buckets(
            self.problem,
            ep_size,
            self.max_num_tokens,
            prefill_worker=self.prefill_worker,
        )
        self.bucket = self.buckets[-1]
        self.dp_size = vllm_config.parallel_config.data_parallel_size
        # Whether forward gets a 1/TP sequence-parallel shard (see ep_tokens).
        self.sequence_parallel = False
        if ep_rank == 0:
            logger.info_once(
                "Rubin MegaMoE: %s checkpoint, %s %s buckets of %s tokens/rank at "
                "EP%d (%s).",
                weight_format.name,
                self.quant_kind,
                self.bucket.variant,
                tuple(b.max_tokens for b in self.buckets),  # info_once hashes args
                ep_size,
                "; ".join(notes),
            )
        # Kernel-layout weights, set by process_weights_after_loading.
        self.kernel_weights: RubinMoeWeights | None = None
        # Local experts [lo, hi) this rank holds: all of them, except under expert
        # DWDP, where each layer after the first keeps only the half its engine
        # owns (dwdp.ExpertDwdp.claim). NVFP4 checkpoints then allocate and load only
        # that half of the codes and block scales (the per-expert global scales stay
        # whole: they are the kernel's alphas for both halves).
        self.resident_experts = (0, num_local_experts)
        expert_dwdp = dwdp.get()
        if expert_dwdp is not None:
            self._dwdp_layer, lo, hi = expert_dwdp.claim(num_local_experts)
            self.resident_experts = (lo, hi)
        self.sharded_experts = weight_format is MEGA_MOE_NVFP4 and (
            self.resident_experts != (0, num_local_experts)
        )

        if weight_format is MEGA_MOE_NVFP4:
            self._create_nvfp4_weights()
        else:
            self._create_weights()
        for name, param in self.named_parameters(recurse=False):
            set_weight_attrs(param, {"weight_loader": self.weight_loader})
            if name.endswith("weight_scale"):
                param.quant_method = "block"

        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    @property
    def has_fused_shared_experts(self) -> bool:
        return False

    @property
    def layer_id(self) -> int:
        return extract_layer_index(self.prefix)

    # -- checkpoint-layout parameters -------------------------------------------------

    def _create_weights(self) -> None:
        """Canonical MXFP4 parameters, local expert on dim 0 (DeepGEMM's names,
        shapes and dtypes): ``w13`` rows ``[0, I)`` gate (w1), ``[I, 2I)`` up (w3);
        packed E2M1 codes as int8 ``(rows, K/2)``; UE8M0 scales row-major as int32
        ``(rows, K/128)``, i.e. bytes ``(rows, K/32)``."""
        e, h, i = self.num_local_experts, self.hidden_size, self.intermediate_size
        group = 4 * MX_BLOCK
        if h % group or i % group:
            raise ValueError(
                f"Rubin MegaMoE needs hidden and intermediate sizes that are multiples "
                f"of {group}, got {h} and {i}."
            )
        shapes = {
            "w13_weight": ((e, 2 * i, h // 2), torch.int8),
            "w13_weight_scale": ((e, 2 * i, h // group), torch.int32),
            "w2_weight": ((e, h, i // 2), torch.int8),
            "w2_weight_scale": ((e, h, i // group), torch.int32),
        }
        for name, (shape, dtype) in shapes.items():
            data = torch.zeros(shape, dtype=dtype)
            self.register_parameter(name, nn.Parameter(data, requires_grad=False))

    # Dummy values of the NVFP4 scale parameters (``--load-format dummy``): finite,
    # positive power-of-two scales like the ModelOpt export's (random E4M3 bytes
    # include NaN; zero scales would zero the experts).
    _NVFP4_DUMMY_SCALES: ClassVar[dict[str, float]] = {
        "weight_scale": 1.0,
        "weight_scale_2": 2.0**-13,
        "input_scale": 1.0,
    }

    def _create_nvfp4_weights(self) -> None:
        """ModelOpt NVFP4 parameters as stored, local expert on dim 0: ``w13`` rows
        ``[0, I)`` gate (w1), ``[I, 2I)`` up (w3); packed E2M1 codes uint8 ``(rows,
        K/2)``; E4M3 block-16 scales ``(rows, K/16)``; per-expert fp32
        ``weight_scale_2`` and ``input_scale`` (``w13_*`` columns: w1, w3)."""
        e, h, i = self.num_local_experts, self.hidden_size, self.intermediate_size
        group = 4 * MX_BLOCK
        if h % group or i % group:
            raise ValueError(
                f"Rubin MegaMoE needs hidden and intermediate sizes that are multiples "
                f"of {group}, got {h} and {i}."
            )
        lo, hi = self.resident_experts
        r = hi - lo  # resident experts (codes and block scales)
        e4m3 = torch.float8_e4m3fn
        shapes = {
            "w13_weight": ((r, 2 * i, h // 2), torch.uint8),
            "w13_weight_scale": ((r, 2 * i, h // NV_BLOCK), e4m3),
            "w13_weight_scale_2": ((e, 2), torch.float32),
            "w13_input_scale": ((e, 2), torch.float32),
            "w2_weight": ((r, h, i // 2), torch.uint8),
            "w2_weight_scale": ((r, h, i // NV_BLOCK), e4m3),
            "w2_weight_scale_2": ((e,), torch.float32),
            "w2_input_scale": ((e,), torch.float32),
        }
        for name, (shape, dtype) in shapes.items():
            param = nn.Parameter(torch.zeros(shape, dtype=dtype), requires_grad=False)
            self.register_parameter(name, param)
            suffix = name.split("_", 1)[1]
            if suffix in self._NVFP4_DUMMY_SCALES:
                param.dummy_weight_value = self._NVFP4_DUMMY_SCALES[suffix]

    def _map_global_expert_id(self, expert_id: int) -> list[int]:
        """Local slots that logical expert ``expert_id`` occupies on this rank."""
        physical_ids: list[int] = []
        for p in range(self.experts_start_idx, self.experts_end_idx):
            if p % self.num_logical_experts == expert_id:
                physical_ids.append(p - self.experts_start_idx)
        return physical_ids

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: bool = False,
    ) -> bool | None:
        if self.kernel_weights is not None:
            raise RuntimeError(
                f"{self.prefix}: Rubin MegaMoE converted its experts to the kernel "
                "layout at load; reloading weights is not supported."
            )
        nvfp4 = self.weight_format is MEGA_MOE_NVFP4
        is_global = weight_name.endswith(("weight_scale_2", "input_scale"))
        if is_global and not nvfp4:
            raise ValueError(
                f"{weight_name}: this Rubin MegaMoE layer loads MXFP4 experts (codes + "
                "UE8M0 scales) only."
            )
        if shard_id not in ("w1", "w2", "w3"):
            raise ValueError(f"Unsupported expert shard id: {shard_id}")
        local_expert_ids = self._map_global_expert_id(expert_id)
        param_prefix = "w2_" if shard_id == "w2" else "w13_"
        if not local_expert_ids or param_prefix not in weight_name:
            return False if return_success else None
        if is_global:
            if loaded_weight.numel() != 1:
                raise ValueError(
                    f"{weight_name}: expected one value per expert, got shape "
                    f"{tuple(loaded_weight.shape)}."
                )
            value = loaded_weight.reshape(()).to(torch.float32)
            for local_expert_id in local_expert_ids:
                slot = param.data[local_expert_id]
                if shard_id != "w2":
                    slot = slot[0 if shard_id == "w1" else 1]
                slot.copy_(value)
            return True if return_success else None
        is_scale = weight_name.endswith("weight_scale")
        scale_dtype = torch.float8_e4m3fn if nvfp4 else torch.int32
        if is_scale != (param.dtype == scale_dtype):
            raise ValueError(f"{weight_name} does not match its {param.dtype} param.")
        if nvfp4 and is_scale and loaded_weight.dtype not in (
            torch.float8_e4m3fn,
            torch.uint8,
        ):
            raise ValueError(
                f"{weight_name}: NVFP4 block scales must be E4M3, got "
                f"{loaded_weight.dtype}."
            )
        if self.sharded_experts:  # only the resident experts' rows exist
            lo, hi = self.resident_experts
            local_expert_ids = [x - lo for x in local_expert_ids if lo <= x < hi]
            if not local_expert_ids:
                return False if return_success else None
        for local_expert_id in local_expert_ids:
            self._load_weight(param.data[local_expert_id], loaded_weight, shard_id)
        return True if return_success else None

    @staticmethod
    def _load_weight(dst: torch.Tensor, src: torch.Tensor, shard_id: str) -> None:
        """Copy checkpoint shard ``src`` (one expert's w1 / w3 / w2 codes or scales,
        one byte per element) into its rows of one expert's ``dst`` as stored."""
        if shard_id not in ("w1", "w2", "w3"):
            raise ValueError(f"Unsupported expert shard id: {shard_id}")
        if src.element_size() != 1:
            raise ValueError(
                f"Rubin MegaMoE expects 1-byte FP4 codes / block scales, got "
                f"{src.dtype}."
            )
        dst = dst.view(torch.uint8)
        rows = dst.shape[0] // (1 if shard_id == "w2" else 2)
        if tuple(src.shape) != (rows, dst.shape[1]):
            raise ValueError(
                f"Rubin MegaMoE {shard_id} shard shape {tuple(src.shape)} does not "
                f"match the expert slice ({rows}, {dst.shape[1]})."
            )
        row0 = rows if shard_id == "w3" else 0
        dst[row0 : row0 + rows].copy_(src.view(torch.uint8))

    def canonical_weights(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(w13, w13_scale, w2, w2_scale)`` uint8 views in the
        ``megamoe_rubin.weights`` canonical layout."""
        return (
            self.w13_weight.data.view(torch.uint8),
            self.w13_weight_scale.data.view(torch.uint8),
            self.w2_weight.data.view(torch.uint8),
            self.w2_weight_scale.data.view(torch.uint8),
        )

    def _nvfp4_alphas(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-expert FC1 / FC2 global weight scales of the loaded NVFP4 experts.

        The kernel takes one FC1 scale per expert and quantizes activations with
        unit global scales, so every ``w1`` / ``w3`` ``weight_scale_2`` pair must be
        equal and every ``input_scale`` 1.0 (the Kimi K3.1 ModelOpt export stores
        exactly that); anything else raises instead of changing the math.
        """
        ws13 = self.w13_weight_scale_2.data
        if not torch.equal(ws13[:, 0], ws13[:, 1]):
            bad = (ws13[:, 0] != ws13[:, 1]).nonzero().flatten()[:8].tolist()
            raise ValueError(
                f"{self.prefix}: w1 and w3 weight_scale_2 differ for local experts "
                f"{bad}; the Rubin NVFP4 kernel takes one FC1 scale per expert."
            )
        for name in ("w13_input_scale", "w2_input_scale"):
            scales = getattr(self, name).data
            if not bool((scales == 1.0).all()):
                raise NotImplementedError(
                    f"{self.prefix}: {name} values {scales.unique()[:8].tolist()}; the "
                    "Rubin NVFP4 path takes unit activation scales only."
                )
        return ws13[:, 0].clone(), self.w2_weight_scale_2.data.clone()

    def _prepare_resident_nvfp4(self, runtime: RubinMegaMoE) -> RubinMoeWeights:
        """Expert DWDP, NVFP4: the kernel layout of the resident half (the layout is
        per expert, so the half converts to the same bytes as the same experts of a
        whole conversion), with every local expert's alphas."""
        lo, hi = self.resident_experts
        fc1_alpha, fc2_alpha = self._nvfp4_alphas()
        for name, g in (("w13", fc1_alpha), ("w2", fc2_alpha)):
            if not bool(torch.isfinite(g).all()) or not bool((g > 0).all()):
                raise ValueError(
                    f"{self.prefix}: {name} weight_scale_2 must be finite and positive."
                )
        weights = runtime.prepare_nvfp4_checkpoint_weights(
            *self.canonical_weights(),
            fc1_alpha[lo:hi].contiguous(),
            fc2_alpha[lo:hi].contiguous(),
            num_experts=hi - lo,
        )
        return weights._replace(
            fc1_alpha=fc1_alpha,
            fc2_alpha=fc2_alpha,
            fc1_norm_const=torch.ones_like(fc1_alpha),
        )

    def reserve_sms_for_shared_expert(
        self,
        max_sm_count: int = RUBIN_SHARED_OVERLAP_MAX_SMS,
        *,
        die_local: bool = True,
        up_to_tokens: int | None = None,
    ) -> None:
        """Cap this layer's persistent grid at ``max_sm_count`` (default
        :data:`RUBIN_SHARED_OVERLAP_MAX_SMS`, the decode SP measurement) so a shared
        expert the caller runs on another stream gets the remaining SMs.
        ``die_local=False`` keeps the cap where the decode die-local policy would
        take the full grid (the prefill-sized wide-EP layers).
        Every layer of the EP group must make the same call (the bucket keys the
        shared runtime); it must come before the runtime is built.
        ``up_to_tokens`` (P/D prefill workers): only a bucket of at most that many
        tokens per rank is capped; it is added below the main bucket (same variant and
        knob record, its own capacity-sized arena share), and the main bucket keeps the
        full grid. :meth:`shared_overlap` says per forward which one runs."""
        if self.kernel_weights is not None:
            raise RuntimeError(
                f"{self.prefix}: reserve_sms_for_shared_expert after the runtime "
                "was built."
            )
        # A capped grid launches preferred-shape clusters only: the mixed-CGA
        # fallback clusters exist to fill the SMs the cap leaves to the shared
        # expert. Tiles and their accumulation order do not depend on the grid.
        # Every compiled bucket (the decode bucket included): forward picks one per
        # batch, and each must leave the shared expert its SMs.
        # Decode servers also stream each die's experts from its own HBM
        # (``die_local``; the phase-interleaved schedule on one cluster shape), when
        # the device has a trusted die map. Die-local streams then take the full grid
        # (no SM cap): measured on VR200 (EP16 decode shape on EP4, 8 layers of
        # MegaMoE + side-stream shared expert + latent up, CUDA graph, p50 us/layer at
        # 32 / 52 / 64 tok/rank): capped die-oblivious 151.8 / 161.7 / 162.4, capped
        # die-local 148.3 / 158.6 / 158.9, full-grid die-local 140.8 / 149.8 / 150.5
        # (bitwise equal). Capped, each die's stream gets the clusters the side
        # stream leaves on that die; full, both dies' HBM is streamed by all SMs.
        # The prefill-sized dedup buckets have no die-split schedule: they keep the
        # die-oblivious form and leave the decode buckets' die-local choice alone.
        # Expert DWDP homes each half of a layer's experts on its die, so every
        # phase-interleaved inference bucket streams die-locally, prefill included.
        def die_streams(bucket: RubinBucket) -> bool:
            if bucket.variant in DEDUP_VARIANTS:
                return False
            if dwdp.ENABLED:
                return (
                    bucket.variant == "inference"
                    and tuple(bucket.knobs.schedule_policy)[0] == "phase_interleave"
                )
            return True

        die_mask = None
        if (dwdp.ENABLED and any(die_streams(b) for b in self.buckets)) or (
            die_local
            and not self.prefill_worker
            and all(
                tuple(bucket.knobs.schedule_policy)[0]
                == ("decode" if bucket.variant == "genphase" else "phase_interleave")
                for bucket in self.buckets
                if bucket.variant not in DEDUP_VARIANTS
            )
        ):
            from . import die_local

            die_mask = die_local.current_die_sm_mask32()

        def _placed(bucket: RubinBucket) -> RubinBucket:
            mask = die_mask if die_streams(bucket) else None
            return dataclasses.replace(
                bucket,
                knobs=dataclasses.replace(bucket.knobs, fallback_cluster_shape_mn=None),
                max_sm_count=max_sm_count if mask is None else None,
                die_sm_mask32=mask,
            )

        if up_to_tokens is None:
            self.buckets = tuple(_placed(bucket) for bucket in self.buckets)
        else:
            main = self.buckets[-1]
            if len(self.buckets) != 1 or not 0 < up_to_tokens < main.max_tokens:
                raise ValueError(
                    f"{self.prefix}: a capped {up_to_tokens}-token bucket needs one "
                    f"larger bucket, got {[b.max_tokens for b in self.buckets]}."
                )
            small = dataclasses.replace(main, max_tokens=up_to_tokens)
            self.buckets = (
                dataclasses.replace(
                    small,
                    knobs=dataclasses.replace(
                        small.knobs, fallback_cluster_shape_mn=None
                    ),
                    max_sm_count=max_sm_count,
                    die_sm_mask32=None,
                ),
                main,
            )
        self.bucket = self.buckets[-1]
        # The overlap runs under sequence parallelism: each TP rank holds a 1/TP
        # shard of its DP batch (sequence_parallel_chunk).
        self.sequence_parallel = True

    def shared_overlap(self, num_tokens: int) -> bool:
        """Whether a forward of ``num_tokens`` tokens on this rank runs on an SM-capped
        bucket (the caller then runs the shared expert beside it)."""
        return self.select_bucket(num_tokens).max_sm_count is not None

    def enable_fused_shared(self, hidden: int, intermediate: int) -> bool:
        """Switch this layer's wide-EP decode bucket (``RUBIN_WIDE_EP_DECODE_TOKENS``
        tokens per rank, the D3-32 SP shard; added below a larger GenPhase bucket,
        the EP16 SP decode shard) to the refreshed GenPhase kernel with an
        MXFP8 E4M3 shared expert of ``(hidden, intermediate)`` fused in (its FC1 tiles
        run under the dispatch; the side-stream shared expert then has nothing left
        to overlap). Per-shape: larger buckets keep their kernel. Returns whether a
        bucket switched. Same call on every layer of the EP group (the buckets key
        the runtime); then :meth:`warmup_fused_shared` before CUDA graph capture."""
        if self.prefill_worker:
            return False
        first = self.buckets[0]
        if (
            first.variant == "genphase"
            and first.max_tokens > RUBIN_WIDE_EP_DECODE_TOKENS
            and self.problem.quant_kind == "nvfp4"
        ):
            # A GenPhase server sized above the fused bucket (EP16 SP decode: shards
            # of 192 / 242 tokens per rank) adds it: steps of at most
            # RUBIN_WIDE_EP_DECODE_TOKENS tokens per rank run the fused shared
            # expert, larger ones keep the GenPhase bucket and the side-stream
            # shared expert. Same kernel family and math; one more bucket (arena,
            # workspace) in the runtime built for the fused buckets.
            self.buckets = (
                dataclasses.replace(first, max_tokens=RUBIN_WIDE_EP_DECODE_TOKENS),
                *self.buckets,
            )
            logger.info_once(
                "Rubin MegaMoE: EP%d GenPhase decode adds the %d-token fused-shared "
                "bucket below its %d bucket.",
                self.ep_size,
                RUBIN_WIDE_EP_DECODE_TOKENS,
                first.max_tokens,
            )
        switched = False
        buckets = []
        for bucket in self.buckets:
            if (
                bucket.max_tokens <= RUBIN_WIDE_EP_DECODE_TOKENS
                and self.problem.quant_kind == "nvfp4"
            ):
                bucket = dataclasses.replace(
                    bucket,
                    variant="genphase",
                    knobs=RUBIN_GENPHASE_FUSED_SHARED_KNOBS,
                    max_sm_count=None,
                    shared=(hidden, intermediate),
                )
                switched = True
            buckets.append(bucket)
        self.buckets = tuple(buckets)
        self.bucket = self.buckets[-1]
        for bucket in self.buckets:  # the kernel's admission rules (raises)
            if bucket.shared is not None:
                RubinMoeConfig.make(
                    self.problem, bucket, rank=0, world_size=self.ep_size
                )
        return switched

    def fused_shared(self, num_tokens: int) -> bool:
        """Whether a forward of ``num_tokens`` runs the fused shared expert (it then
        takes ``fused_shared=``)."""
        return self.select_bucket(num_tokens).shared is not None

    def home_fused_shared_rows(self, rows: torch.Tensor) -> torch.Tensor:
        """The fused shared expert's FC1 weight rows ``[N, K]`` (contiguous, 16-row
        gate/up stripes) homed per die (feature halves, as :meth:`warmup_fused_shared`
        homes FC1) on die-local buckets: a die-local copy the caller swaps in for
        ``rows``; ``rows`` itself otherwise."""
        if self.bucket.die_sm_mask32 is None:
            return rows
        from .die_local import place_halves

        return place_halves(rows)

    def warmup_fused_shared(self, shared, placed: bool = False):
        """Collective: build, compile and warm the fused-shared runtime with one
        layer's ``session.FusedSharedArgs``; returns the arguments to launch with
        (FC1 homed per die on die-local buckets, unless ``placed``: FC1 views rows
        already homed by :meth:`home_fused_shared_rows`)."""
        if self.kernel_weights is None:
            raise RuntimeError(f"{self.prefix}: warmup_fused_shared before weights.")
        if self.bucket.die_sm_mask32 is not None and not placed:
            from .die_local import place_halves

            # FC1 [1, K, N] is a K-major view of the striped [1, N, K] copy: home its
            # feature halves per die (the decode scheduler claims them die-local).
            w = shared.fc1[0]
            placed = place_halves(w.permute(0, 2, 1)).permute(0, 2, 1)
            shared = shared._replace(fc1=(placed, shared.fc1[1]))
        self.get_runtime().warmup(self.kernel_weights, shared=shared)
        self._warm_fused_shared = shared
        return shared

    # -- runtime / kernel layout -----------------------------------------------------

    def get_runtime(self) -> RubinMegaMoE:
        """The EP group's shared :class:`RubinMegaMoE` for this layer's problem and
        buckets (collective on first use: symmetric arena, compile objects)."""
        from .runtime import RubinMegaMoE

        group = get_ep_group().device_group
        key = (
            id(group),
            torch.accelerator.current_device_index(),
            self.problem,
            self.buckets,
        )
        runtime = self._runtimes.get(key)
        if runtime is None:
            runtime = RubinMegaMoE(self.problem, group, list(self.buckets))
            self._runtimes[key] = runtime
        return runtime

    def process_weights_after_loading(self) -> None:
        """Canonical MXFP4 -> kernel layout for :attr:`quant_kind` (exact NVFP4
        conversion or :class:`NonRepresentableError`), then free the checkpoint
        copy. Collective over the EP group (the runtime's arena on first use, and a
        failure on any rank raises on every rank instead of hanging the others)."""
        if self.kernel_weights is not None:
            return
        runtime = self.get_runtime()
        error: Exception | None = None
        weights = None
        try:
            if self.weight_format is MEGA_MOE_NVFP4 and self.sharded_experts:
                weights = self._prepare_resident_nvfp4(runtime)
            elif self.weight_format is MEGA_MOE_NVFP4:
                alphas = self._nvfp4_alphas()
                weights = runtime.prepare_nvfp4_checkpoint_weights(
                    *self.canonical_weights(), *alphas
                )
            else:
                weights = runtime.prepare_weights(
                    *self.canonical_weights(), layer_id=self.layer_id
                )
        except Exception as exc:  # re-raised below, after the ranks agree
            error = exc
        ok = torch.tensor([0 if error is not None else 1], dtype=torch.int32)
        cpu_group = get_ep_group().cpu_group
        torch.distributed.all_reduce(
            ok, op=torch.distributed.ReduceOp.MIN, group=cpu_group
        )
        if error is not None:
            raise error
        if not int(ok):
            raise RuntimeError(
                f"{self.prefix}: Rubin MegaMoE weight preparation failed on another "
                "EP rank (see its log)."
            )
        assert weights is not None
        expert_dwdp = dwdp.get()
        if expert_dwdp is not None:
            lo, hi = self.resident_experts
            if not self.sharded_experts and (lo, hi) != (0, self.num_local_experts):
                # Converted whole (MXFP4: the alphas come from the conversion):
                # keep the home half.
                weights = weights._replace(
                    fc1=tuple(t[lo:hi] for t in weights.fc1),
                    fc2=tuple(t[lo:hi] for t in weights.fc2),
                )
            weights = expert_dwdp.register(self, weights)
        elif self.bucket.die_sm_mask32 is not None and _die_local_fits(
            weights, self.num_moe_layers
        ):
            from .die_local import place_experts

            weights = place_experts(weights)
        self.kernel_weights = weights
        if weights.report is not None and get_ep_group().rank_in_group == 0:
            logger.debug("%s: %s", self.prefix, weights.report.summary())
        # The kernel copy replaces the checkpoint layout (the MX path may alias w2).
        for _, param in self.named_parameters(recurse=False):
            param.data = torch.empty(0, dtype=param.dtype, device=param.device)
        if self.bucket.die_sm_mask32 is not None or dwdp.ENABLED:
            del weights
            torch.accelerator.empty_cache()  # the default-allocated originals

    # -- EPLB -------------------------------------------------------------------------

    def set_eplb_state(
        self,
        moe_layer_idx: int,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ) -> None:
        self.eplb_state.set_layer_state(
            moe_layer_idx,
            expert_load_view,
            logical_to_physical_map,
            logical_replica_count,
        )

    def get_expert_weights(self) -> list[torch.Tensor]:
        """Every kernel-layout tensor as a contiguous ``(num_local_experts, -1)``
        view (EPLB moves whole local-expert rows; the bound launch arguments keep
        their pointers)."""
        w = self.kernel_weights
        if w is None:
            raise RuntimeError(
                f"{self.prefix}: EPLB needs the kernel-layout experts; run "
                "process_weights_after_loading first."
            )
        tensors = [
            w.fc1[0].permute(0, 2, 1),  # logical (E, K/2, N) of K-major storage
            w.fc1[1],  # full or compact scale planes, as the kernel reads them
            w.fc2[0].permute(0, 2, 1),
            w.fc2[1],
            *(t for t in (w.fc1_alpha, w.fc2_alpha, w.fc1_norm_const) if t is not None),
        ]
        views = []
        for t in tensors:
            if t.shape[0] != self.num_local_experts or not t.is_contiguous():
                raise AssertionError(
                    f"Rubin MegaMoE EPLB: unexpected expert tensor layout "
                    f"shape={tuple(t.shape)} stride={tuple(t.stride())}"
                )
            views.append(t.view(self.num_local_experts, -1))
        return views

    def update_expert_map(self) -> None:
        pass

    # -- forward ----------------------------------------------------------------------

    def select_bucket(self, num_tokens: int) -> RubinBucket:
        """The bucket for a forward of ``num_tokens`` tokens on this rank.

        The kernel's lock-step contract needs the same bucket on every EP rank. An EP
        rank's token count is its DP rank's (DP-padded) batch, or its 1/TP shard of
        it, the same on every TP rank of the DP group; so equal DP batch sizes make
        ``num_tokens`` EP-uniform and the smallest bucket holding it is taken. With
        unequal DP batch sizes (identical ``num_tokens_across_dp`` on every rank) all
        ranks take the largest bucket. Raises when DP > 1 without DP metadata.
        """
        if num_tokens > self.max_num_tokens:
            raise ValueError(
                f"Rubin MegaMoE got {num_tokens} tokens, but its buckets were sized "
                f"for {self.max_num_tokens}."
            )
        buckets = self.buckets
        if len(buckets) == 1:
            return buckets[0]
        if self.dp_size > 1:
            dp_metadata = (
                get_forward_context().dp_metadata
                if is_forward_context_available()
                else None
            )
            if dp_metadata is None:
                raise RuntimeError(
                    f"{self.prefix}: Rubin MegaMoE at DP{self.dp_size} needs the "
                    "forward context's DP metadata to pick an EP-uniform bucket."
                )
            if not _dp_batches_uniform(dp_metadata):
                return buckets[-1]
        for b in buckets:
            if num_tokens <= b.max_tokens:
                return b
        raise AssertionError("unreachable: the largest bucket holds max_num_tokens")

    def ep_tokens(self, num_tokens: int) -> int | None:
        """EP-uniform live tokens per rank of this forward (the EP mean, which sets
        each rank's expert work); picks a capped prefill bucket's epilogue variant
        (``session.RubinMegaMoeSession.use_s2``). ``num_tokens`` itself at DP1 (every
        EP rank holds the same count); None (the bucket capacity) at DP > 1 without
        DP metadata, which every rank sees alike."""
        if self.dp_size == 1:
            return num_tokens
        dp_metadata = (
            get_forward_context().dp_metadata
            if is_forward_context_available()
            else None
        )
        if dp_metadata is None:
            return None
        sp_size = (
            get_tensor_model_parallel_world_size() if self.sequence_parallel else 1
        )
        return _ep_mean_tokens(dp_metadata, sp_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: nn.Module | None = None,
        is_padding: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
        ep_tokens: int | None = None,
        output_rmsnorm: nn.Module | None = None,
        fused_shared=None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run the routed experts (and an unfused ``shared_experts``) on this rank's
        tokens; arguments and result as :meth:`DeepGemmMegaMoEExperts.forward`.

        ``output_rmsnorm``: the routed output's consumer RMSNorm (the latent norm), applied
        inside the kernel's top-k reduce (result = ``output_rmsnorm(routed output)``,
        stock ``rms_norm`` numerics); :meth:`prepare_output_rmsnorm` it before capture.

        ``topk_weights`` carry ``routed_scaling_factor``; padding rows (``is_padding``
        or the forward context's under ``VLLM_MOE_SKIP_PADDING``) are masked to -1.
        Every EP rank must call each layer the same number of times (the kernel
        synchronizes the group), as with DeepGEMM MegaMoE.
        ``out``: optional output rows; ``ep_tokens``: EP-uniform live tokens per
        rank of this launch (default: from the forward's DP batch sizes), set by
        callers that split a batch into several launches.
        """
        weights = self.kernel_weights
        if weights is None:
            raise RuntimeError(
                f"{self.prefix}: Rubin MegaMoE used before "
                "process_weights_after_loading (prepare_rubin_megamoe)."
            )
        expert_dwdp = dwdp.get()
        if expert_dwdp is not None:
            expert_dwdp.before_experts(self)
        num_tokens = hidden_states.shape[0]
        # Static per captured CUDA graph size (shape-only + host DP metadata).
        bucket = self.select_bucket(num_tokens)
        if (
            is_padding is None
            and envs.VLLM_MOE_SKIP_PADDING
            and is_forward_context_available()
        ):
            is_padding = get_forward_context().is_padding
        if is_padding is not None:
            is_padding = is_padding[:num_tokens]

        if self.capture_fn is not None:
            self.capture_fn(topk_ids)

        eplb_state = self.eplb_state
        eplb = None
        if eplb_state.logical_to_physical_map is not None:
            assert eplb_state.expert_load_view is not None
            assert eplb_state.logical_replica_count is not None
            assert eplb_state.should_record_tensor is not None
            # The stager masks the padding, then maps the logical ids and records
            # the load (eplb_map_to_physical_and_record) in its one launch.
            eplb = EplbStage(
                eplb_state.logical_to_physical_map,
                eplb_state.logical_replica_count,
                eplb_state.expert_load_view,
                eplb_state.should_record_tensor,
                eplb_state.num_unpadded_tokens_tensors[dbo_current_ubatch_id()]
                if eplb_state.num_unpadded_tokens_tensors is not None
                else None,
            )

        extra = {}
        if output_rmsnorm is not None:
            if shared_experts is not None:
                raise ValueError("output_rmsnorm with an unfused shared expert is not supported.")
            extra["rmsnorm"] = _rmsnorm_args(output_rmsnorm)
        drop_shared = False
        if fused_shared is not None:
            if shared_experts is not None or bucket.shared is None:
                raise ValueError(f"{self.prefix}: no fused shared expert at {num_tokens} tokens.")
            extra["shared"] = fused_shared
        elif bucket.shared is not None:
            # A caller outside the specialized decoder (the stock forward, e.g. a
            # bench's context prefill) runs its own shared expert: the fused bucket's
            # shared tiles run on the warmed operands and their output is dropped.
            extra["shared"] = getattr(self, "_warm_fused_shared", None)
            if extra["shared"] is None:
                raise ValueError(f"{self.prefix}: the {bucket.max_tokens} bucket fuses the shared expert.")
            drop_shared = True
        y = self.get_runtime().forward(
            hidden_states,
            topk_ids,
            topk_weights,
            weights,
            bucket=bucket,
            out=out,
            is_padding=is_padding,
            owner=self.prefix,
            ep_tokens=self.ep_tokens(num_tokens) if ep_tokens is None else ep_tokens,
            eplb=eplb,
            **extra,
        )
        if drop_shared:
            y = y[0]
        if shared_experts is not None:
            y += shared_experts(hidden_states)
        if expert_dwdp is not None:
            expert_dwdp.after_experts(self)
        return y


def _rmsnorm_args(norm: nn.Module) -> tuple[torch.Tensor, float]:
    """``(weight, eps)`` of a plain weighted vLLM ``RMSNorm`` (no residual, no variance
    override) for the fused reduce; anything else raises (never a silent fallback)."""
    weight = getattr(norm, "weight", None)
    if (
        weight is None
        or not getattr(norm, "has_weight", True)
        or getattr(norm, "variance_size_override", None) is not None
        or weight.dtype != torch.bfloat16
        or weight.dim() != 1
    ):
        raise ValueError(f"cannot fuse {type(norm).__name__} into the MegaMoE reduce.")
    return weight.data, float(norm.variance_epsilon)


def prepare_output_rmsnorm(experts: nn.Module, norm: nn.Module) -> None:
    """Compile ``experts``' fused-RMSNorm reduce for ``norm`` (before CUDA graph capture)."""
    weights = experts.kernel_weights
    if weights is None:
        raise RuntimeError(f"{experts.prefix}: prepare_output_rmsnorm before weight preparation.")
    experts.get_runtime().prepare_rmsnorm(weights, *_rmsnorm_args(norm))


KimiK3MegaMoEExperts.weight_loader.supports_moe_loading = True  # type: ignore[attr-defined]


def prepare_rubin_megamoe(model: nn.Module) -> int:
    """Convert every ``KimiK3MegaMoEExperts`` of ``model`` to the kernel layout (module
    order, identical on every rank), then run the EP group's collective warmup
    (stager + kernel compile, route-id self-test, one empty launch per bucket)
    before any forward or CUDA graph capture. Returns the number of layers."""
    experts = [m for m in model.modules() if isinstance(m, KimiK3MegaMoEExperts)]
    for module in experts:
        module.process_weights_after_loading()
    if experts:
        compact_scales.install(experts, get_ep_group().cpu_group)
    warmed: set[int] = set()
    for module in experts:
        runtime = module.get_runtime()
        if id(runtime) in warmed:
            continue
        assert module.kernel_weights is not None
        runtime.warmup(module.kernel_weights)
        warmed.add(id(runtime))
    expert_dwdp = dwdp.get()
    if experts and expert_dwdp is not None:
        expert_dwdp.finalize()  # cross-engine, before any forward
    if experts:
        logger.info_once(
            "Rubin MegaMoE: %d layers in the %s kernel layout, %d runtime(s) warmed.",
            len(experts),
            experts[0].quant_kind,
            len(warmed),
        )
    return len(experts)


_DIE_LOCAL_FITS: bool | None = None


def _die_local_fits(weights: Any, num_moe_layers: int) -> bool:
    """Decided once, at the first layer: whether the die-localized copies of every
    MoE layer's experts fit in the device memory left after loading. The
    checkpoint layouts they replace are freed into the allocator's pool, not to
    the driver the copies come from, so placement needs the copies' bytes on top
    of the loaded model (e.g. TP4 x EP12 of the full K3.1 does not fit, EP16 does).
    Without room every layer keeps the default allocation."""
    global _DIE_LOCAL_FITS
    if _DIE_LOCAL_FITS is None:
        layer = sum(
            t.numel() * t.element_size()
            for t in (*weights.fc1, *weights.fc2)
            if isinstance(t, torch.Tensor)
        )
        need = layer * num_moe_layers + (8 << 30)
        free, _ = torch.cuda.mem_get_info()
        _DIE_LOCAL_FITS = free >= need
        if not _DIE_LOCAL_FITS:
            logger.warning(
                "Rubin MegaMoE: die-local experts need %.1f GiB, %.1f GiB free "
                "after loading; keeping the default allocation.",
                need / 2**30,
                free / 2**30,
            )
    return _DIE_LOCAL_FITS


def restrict_local_expert_ids(model: nn.Module, local_expert_ids: Any) -> Any:
    """The loader's EP expert filter narrowed to what each MoE layer allocated:
    under expert DWDP a layer after the first holds only its home half, so the
    other half's weights are skipped before they are read."""
    from vllm.model_executor.model_loader.ep_weight_filter import LayerExpertIds

    layers: dict[int, set[int]] = {}
    for m in model.modules():
        if isinstance(m, KimiK3MegaMoEExperts) and m.sharded_experts:
            lo, hi = m.resident_experts
            first = m.experts_start_idx
            layers[m.layer_id] = {
                p % m.num_logical_experts for p in range(first + lo, first + hi)
            }
    if not layers or isinstance(local_expert_ids, LayerExpertIds):
        return local_expert_ids
    return LayerExpertIds(set(local_expert_ids), layers)
