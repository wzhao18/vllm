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
  scales, gate rows then up rows); :meth:`process_weights_after_loading` converts them
  to the kernel layout once (``prepare_rubin_megamoe`` runs it for every layer, then
  the collective warmup) and frees the checkpoint copy.
- The routed-expert precision is chosen per server by :func:`rubin_quant_kind` from
  the tokens per rank the layer is sized for (no flag): MXFP8 activations x MXFP4
  weights up to :data:`RUBIN_MX_MAX_TOKENS_PER_RANK`, else an exact MXFP4 -> NVFP4
  conversion at load (:class:`NonRepresentableError` when a weight is not exactly
  representable; nothing is re-quantized).
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
from typing import TYPE_CHECKING, ClassVar

import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.distributed import get_ep_group, get_tensor_model_parallel_world_size
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import (
    MEGA_MOE_MXFP4,
    MegaMoEActivation,
    MegaMoEWeightFormat,
)
from vllm.model_executor.layers.fused_moe.router.base_router import (
    eplb_map_to_physical_and_record,
)
from vllm.model_executor.models.utils import extract_layer_index
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

from .config import (
    SHIPPED_KNOBS,
    QuantKind,
    RubinBucket,
    RubinMoeConfig,
    RubinMoeProblem,
    Variant,
    make_buckets,
)
from .session import ep_mean_tokens
from .weights import MX_BLOCK, RubinMoeWeights

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


def rubin_bucket_capacities(
    max_tokens_per_rank: int, top_k: int, *, prefill_worker: bool
) -> tuple[int, ...]:
    """Ascending token capacities per rank of a server's compiled buckets
    (automatic, no flag): ``max_tokens_per_rank``, preceded by the decode bucket
    (``RUBIN_DECODE_BUCKET_MAX_ROUTES // top_k``) on decoding (not P/D prefill,
    ``prefill_worker``) inference-kernel servers sized above it."""
    if top_k <= 0:
        raise ValueError(f"top_k must be positive: {top_k}")
    decode = RUBIN_DECODE_BUCKET_MAX_ROUTES // top_k
    if (
        not prefill_worker
        and rubin_kernel_variant(max_tokens_per_rank) == "inference"
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


def rubin_quant_kind(max_tokens_per_rank: int) -> QuantKind:
    """Routed-expert precision of a server whose layer sees at most
    ``max_tokens_per_rank`` tokens per forward (automatic, no flag)."""
    if max_tokens_per_rank <= 0:
        raise ValueError(f"max_tokens_per_rank must be positive: {max_tokens_per_rank}")
    if max_tokens_per_rank <= RUBIN_MX_MAX_TOKENS_PER_RANK:
        return "mxfp4_mxfp8"
    return "nvfp4"


def rubin_kernel_variant(max_tokens_per_rank: int) -> Variant:
    if max_tokens_per_rank <= 0:
        raise ValueError(f"max_tokens_per_rank must be positive: {max_tokens_per_rank}")
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
        points=[(rubin_kernel_variant(max_tokens), max_tokens, None)],
    )
    return buckets[0], tuple(notes)


def select_rubin_buckets(
    problem: RubinMoeProblem,
    world_size: int,
    max_tokens: int,
    *,
    prefill_worker: bool,
) -> tuple[tuple[RubinBucket, ...], tuple[str, ...]]:
    """Every compiled bucket of a server, ascending (:func:`rubin_bucket_capacities`):
    the main bucket of :func:`select_rubin_bucket`, and below it the decode bucket,
    which keeps the main bucket's variant and knob record (only the capacity
    differs). Raises (``ValueError``) when the kernel's admission rules refuse that
    record at the decode capacity."""
    main, notes = select_rubin_bucket(problem, world_size, max_tokens)
    caps = rubin_bucket_capacities(
        max_tokens, problem.top_k, prefill_worker=prefill_worker
    )
    buckets = [RubinBucket(main.variant, cap, main.knobs) for cap in caps[:-1]]
    for bucket in buckets:
        RubinMoeConfig.make(problem, bucket, rank=0, world_size=world_size)
    extra = tuple(
        f"[{b.variant} {b.max_tokens}] decode bucket, knobs of the "
        f"{main.max_tokens} bucket"
        for b in buckets
    )
    return (*buckets, main), notes + extra


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
        weight_format: Checkpoint precision; only ``MEGA_MOE_MXFP4``.
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
        if weight_format is not MEGA_MOE_MXFP4:
            raise NotImplementedError(
                "Rubin MegaMoE loads MXFP4 expert checkpoints only, not "
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
        if num_logical_experts not in (None, num_experts):
            raise NotImplementedError(
                "Rubin MegaMoE places experts linearly without redundant experts "
                f"(num_logical_experts {num_logical_experts} != {num_experts})."
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
        self.num_logical_experts = num_experts
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
        self.quant_kind = rubin_quant_kind(self.max_num_tokens)
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
                "Rubin MegaMoE: %s %s buckets of %s tokens/rank at EP%d (%s).",
                self.quant_kind,
                self.bucket.variant,
                tuple(b.max_tokens for b in self.buckets),  # info_once hashes args
                ep_size,
                "; ".join(notes),
            )
        # Kernel-layout weights, set by process_weights_after_loading.
        self.kernel_weights: RubinMoeWeights | None = None

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
        if weight_name.endswith(("weight_scale_2", "input_scale")):
            raise ValueError(
                f"{weight_name}: Rubin MegaMoE loads MXFP4 experts (codes + UE8M0 "
                "scales) only."
            )
        local_expert_ids = self._map_global_expert_id(expert_id)
        param_prefix = "w2_" if shard_id == "w2" else "w13_"
        if not local_expert_ids or param_prefix not in weight_name:
            return False if return_success else None
        if weight_name.endswith("weight_scale") != (param.dtype == torch.int32):
            raise ValueError(f"{weight_name} does not match its {param.dtype} param.")
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
                f"Rubin MegaMoE expects 1-byte MXFP4 codes / UE8M0 scales, got "
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

    def reserve_sms_for_shared_expert(self) -> None:
        """Cap this layer's persistent grid at :data:`RUBIN_SHARED_OVERLAP_MAX_SMS`
        so a shared expert the caller runs on another stream gets the remaining SMs.
        Every layer of the EP group must make the same call (the bucket keys the
        shared runtime); it must come before the runtime is built."""
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
        die_mask = None
        if not self.prefill_worker and all(
            bucket.variant == "inference"
            and tuple(bucket.knobs.schedule_policy)[0] == "phase_interleave"
            for bucket in self.buckets
        ):
            from . import die_local

            die_mask = die_local.current_die_sm_mask32()
        self.buckets = tuple(
            dataclasses.replace(
                bucket,
                knobs=dataclasses.replace(bucket.knobs, fallback_cluster_shape_mn=None),
                max_sm_count=(
                    RUBIN_SHARED_OVERLAP_MAX_SMS if die_mask is None else None
                ),
                die_sm_mask32=die_mask,
            )
            for bucket in self.buckets
        )
        self.bucket = self.buckets[-1]
        # The overlap runs under sequence parallelism: each TP rank holds a 1/TP
        # shard of its DP batch (sequence_parallel_chunk).
        self.sequence_parallel = True

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
        if self.bucket.die_sm_mask32 is not None:
            from .die_local import place_experts

            weights = place_experts(weights)
        self.kernel_weights = weights
        if weights.report is not None and get_ep_group().rank_in_group == 0:
            logger.debug("%s: %s", self.prefix, weights.report.summary())
        # The kernel copy replaces the checkpoint layout (the MX path may alias w2).
        for _, param in self.named_parameters(recurse=False):
            param.data = torch.empty(0, dtype=param.dtype, device=param.device)
        if self.bucket.die_sm_mask32 is not None:
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
            w.fc1[1],
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
    ) -> torch.Tensor:
        """Run the routed experts (and an unfused ``shared_experts``) on this rank's
        tokens; arguments and result as :meth:`DeepGemmMegaMoEExperts.forward`.

        ``topk_weights`` carry ``routed_scaling_factor``; padding rows (``is_padding``
        or the forward context's under ``VLLM_MOE_SKIP_PADDING``) are masked to -1.
        Every EP rank must call each layer the same number of times (the kernel
        synchronizes the group), as with DeepGEMM MegaMoE.
        """
        weights = self.kernel_weights
        if weights is None:
            raise RuntimeError(
                f"{self.prefix}: Rubin MegaMoE used before "
                "process_weights_after_loading (prepare_rubin_megamoe)."
            )
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
        if is_padding is not None and eplb_state.logical_to_physical_map is not None:
            # EPLB maps and records the masked ids: mask before it.
            topk_ids = torch.where(is_padding.unsqueeze(1), -1, topk_ids)
            is_padding = None
        if eplb_state.logical_to_physical_map is not None:
            assert eplb_state.expert_load_view is not None
            assert eplb_state.logical_replica_count is not None
            assert eplb_state.should_record_tensor is not None
            topk_ids = eplb_map_to_physical_and_record(
                topk_ids=topk_ids,
                expert_load_view=eplb_state.expert_load_view,
                logical_to_physical_map=eplb_state.logical_to_physical_map,
                logical_replica_count=eplb_state.logical_replica_count,
                record_enabled=eplb_state.should_record_tensor,
                num_unpadded_tokens=eplb_state.num_unpadded_tokens_tensors[
                    dbo_current_ubatch_id()
                ]
                if eplb_state.num_unpadded_tokens_tensors is not None
                else None,
            )

        # Without EPLB the padding mask is folded into the stager launch.
        y = self.get_runtime().forward(
            hidden_states,
            topk_ids,
            topk_weights,
            weights,
            bucket=bucket,
            is_padding=is_padding,
            owner=self.prefix,
            ep_tokens=self.ep_tokens(num_tokens),
        )
        if shared_experts is not None:
            y += shared_experts(hidden_states)
        return y


KimiK3MegaMoEExperts.weight_loader.supports_moe_loading = True  # type: ignore[attr-defined]


def prepare_rubin_megamoe(model: nn.Module) -> int:
    """Convert every ``KimiK3MegaMoEExperts`` of ``model`` to the kernel layout (module
    order, identical on every rank), then run the EP group's collective warmup
    (stager + kernel compile, route-id self-test, one empty launch per bucket)
    before any forward or CUDA graph capture. Returns the number of layers."""
    experts = [m for m in model.modules() if isinstance(m, KimiK3MegaMoEExperts)]
    for module in experts:
        module.process_weights_after_loading()
    warmed: set[int] = set()
    for module in experts:
        runtime = module.get_runtime()
        if id(runtime) in warmed:
            continue
        assert module.kernel_weights is not None
        runtime.warmup(module.kernel_weights)
        warmed.add(id(runtime))
    if experts:
        logger.info_once(
            "Rubin MegaMoE: %d layers in the %s kernel layout, %d runtime(s) warmed.",
            len(experts),
            experts[0].quant_kind,
            len(warmed),
        )
    return len(experts)
