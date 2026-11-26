# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Problem, knobs, buckets and the validated kernel configuration (CPU-safe).

``RubinMoeConfig`` ports every admission rule of FlashInfer's
``Sm107BlockScaledMoeConfig.__post_init__`` (shim/block_scaled.py, hash in VENDOR.md)
for the quant kinds this backend runs (``nvfp4``, ``mxfp4_mxfp8``, ``mxfp4``) with
BF16 combine, SiTU and separate top-k reduction. The knob reader consumes FlashInfer's
v2 knob-cache JSON (``knobs/k31_vr.json`` ships the VR tuner winners) with
FlashInfer's lookup order; a miss raises, there is no silent default.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import torch

from .weights import MX_BLOCK, NV_BLOCK, fp4_storage_dtype, is_power_of_two

QuantKind = Literal["mxfp4_mxfp8", "nvfp4", "mxfp4"]
Variant = Literal["inference", "genphase", "inference_dedup", "inference_dedup_wg2"]
# "inference_dedup": the refreshed inference kernel (src_gp, FlashInfer main) with the
# deduplicated combine: the owner rank pre-reduces a token's local expert rows (FP32,
# one BF16 rounding) and returns one row per (source rank, token); the separate reduce
# sums the rank rows in ascending rank order (TopkReduceDedup).
VARIANTS = ("inference", "genphase", "inference_dedup", "inference_dedup_wg2")
DEDUP_VARIANTS = ("inference_dedup", "inference_dedup_wg2")  # prefill only

# (act data dtype or None = packed fp4, act/weight SF dtype, sf_vec_size,
#  2x-mode instr K)
_KIND_TABLE: dict[str, tuple[Any, torch.dtype, int, int]] = {
    "nvfp4": (None, torch.float8_e4m3fn, NV_BLOCK, 128),
    "mxfp4_mxfp8": (torch.float8_e4m3fn, torch.float8_e8m0fnu, MX_BLOCK, 64),
    "mxfp4": (None, torch.float8_e8m0fnu, MX_BLOCK, 128),
}

KNOB_KEYS = (
    "mma_tiler_mnk",
    "cluster_shape_mn",
    "fallback_cluster_shape_mn",
    "schedule_policy",
    "work_id_mode",
    "fc2_use_bulk",
    "fc2_tma_stages",
    "epi_flag_batches",
    "token_in_flag_batch",
    "token_back_mode",
    "reduce_topk_in_kernel",
)

# FlashInfer shim/knob_cache.py BACKEND_REVISION / core/tuning.py KNOB_CACHE_VERSION of
# the vendored drop; a record from another revision describes another kernel.
BACKEND_REVISION = "sm107-block-scaled-1667b47a-runtime-options-v1"
KNOB_CACHE_VERSION = 2

SHIPPED_KNOBS = os.path.join(os.path.dirname(__file__), "knobs", "k31_vr.json")


# --------------------------------------------------------------------------------------
# Public problem / knobs / bucket types
# --------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RubinMoeProblem:
    """The Kimi K3.1 routed-MoE contract (defaults) and the NVFP4 staging constants."""

    num_experts: int = 896
    top_k: int = 16
    hidden: int = 3584
    intermediate: int = 3072
    situ_beta: float = 4.0
    situ_linear_beta: float = 25.0
    quant_kind: QuantKind = "nvfp4"
    # NVFP4 activation staging scale and FC1->FC2 requantization scale; powers of two
    # so the per-expert alphas stay exact. Rank-uniform by construction (one value).
    # The defaults are FlashInfer's contract (unit global scales). A served model must
    # not keep fc1_norm_const at 1.0: the kernel requantizes the FC1 output AFTER the
    # routing weight, so the E4M3 block scales of real wires flush to zero; the model
    # passes :func:`nvfp4_fc1_norm_const` (KimiK3MegaMoEExperts).
    input_norm_const: float = 1.0
    fc1_norm_const: float = 1.0

    def __post_init__(self) -> None:
        if self.quant_kind not in _KIND_TABLE:
            raise ValueError(
                f"quant_kind must be one of {tuple(_KIND_TABLE)}, got "
                f"{self.quant_kind!r}."
            )
        for name in ("num_experts", "top_k", "hidden", "intermediate"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive int, got {value!r}.")
        for name in ("situ_beta", "situ_linear_beta"):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be positive and finite, got {value!r}.")
        for name in ("input_norm_const", "fc1_norm_const"):
            if not is_power_of_two(getattr(self, name)):
                raise ValueError(f"{name} must be a positive power of two.")
        if self.quant_kind != "nvfp4" and (
            self.input_norm_const != 1.0 or self.fc1_norm_const != 1.0
        ):
            raise ValueError(
                "norm constants apply to nvfp4 only; keep them 1.0 for MX."
            )


@dataclasses.dataclass(frozen=True)
class RubinKnobs:
    """FlashInfer KNOB_KEYS, every field explicit (no defaults applied silently)."""

    mma_tiler_mnk: tuple[int, int, int]
    cluster_shape_mn: tuple[int, int]
    fallback_cluster_shape_mn: tuple[int, int] | None
    schedule_policy: tuple[str, int | None]
    work_id_mode: str
    fc2_use_bulk: bool
    fc2_tma_stages: int | None
    epi_flag_batches: tuple[int, int]
    token_in_flag_batch: int
    token_back_mode: str
    reduce_topk_in_kernel: bool

    def __post_init__(self) -> None:
        if self.reduce_topk_in_kernel:
            raise ValueError(
                "reduce_topk_in_kernel=True is nondeterministic and not supported by "
                "this backend (separate BF16 top-k reduction only)."
            )

    @classmethod
    def from_dict(cls, knobs: Mapping[str, Any]) -> RubinKnobs:
        unknown = set(knobs) - set(KNOB_KEYS)
        missing = set(KNOB_KEYS) - set(knobs)
        if unknown or missing:
            raise ValueError(
                f"knobs need exactly {KNOB_KEYS}; unknown {sorted(unknown)}, missing "
                f"{sorted(missing)}."
            )

        def tup(v):
            return tuple(v) if isinstance(v, (list, tuple)) else v

        return cls(**{k: tup(knobs[k]) for k in KNOB_KEYS})

    def as_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in KNOB_KEYS}


@dataclasses.dataclass(frozen=True)
class RubinBucket:
    """One compiled session: kernel variant, token capacity per rank and its knobs.

    ``max_sm_count``: cap on the persistent grid (SMs, whole clusters), or None for
    the occupancy limit. A server policy rather than a knob: it leaves SMs to work
    the caller runs concurrently on another stream; the math is unchanged.

    die_sm_mask32: die-local expert streams (die_local): bit s of the
    32-bit words set when SM s is on die 1; the layer's local experts
    [0, E/2) are homed on die 0 and [E/2, E) on die 1. None: die-oblivious.
    Also a server policy (the device's die map), same tiles and math.

    shared: ``(hidden, intermediate)`` of an MXFP8 E4M3 shared expert run inside the
    GenPhase kernel (its tiles overlap the dispatch; its BF16 output lands in its own
    tensor), or None."""

    variant: Variant
    max_tokens: int
    knobs: RubinKnobs
    max_sm_count: int | None = None
    die_sm_mask32: tuple[int, ...] | None = None
    shared: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(f"unknown variant {self.variant!r}.")
        if type(self.max_tokens) is not int or self.max_tokens <= 0:
            raise ValueError("max_tokens must be a positive int.")
        if self.variant == "genphase" and self.max_tokens > 1024:
            raise ValueError("genphase buckets hold at most 1024 tokens per rank.")


# --------------------------------------------------------------------------------------
# Validated kernel configuration (port of Sm107BlockScaledMoeConfig)
# --------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RubinMoeConfig:
    num_total_experts: int
    max_tokens_per_rank: int
    num_topk: int
    hidden: int
    intermediate: int  # width after activation; FC1 GEMM N is 2*intermediate
    rank: int
    world_size: int
    quant_kind: QuantKind
    mma_tiler_mnk: tuple[int, int, int]
    cluster_shape_mn: tuple[int, int]
    fallback_cluster_shape_mn: tuple[int, int] | None
    schedule_policy: tuple[str, int | None]
    work_id_mode: str
    fc2_use_bulk: bool
    fc2_tma_stages: int | None
    epi_flag_batches: tuple[int, int]
    token_in_flag_batch: int
    token_back_mode: str
    situ_beta: float
    situ_linear_beta: float
    kernel_variant: Variant
    token_padding_block: int = 64
    sf_padding_block: int = 128
    # fixed by this backend (kept as fields so the admission rules read like the port)
    reduce_topk_in_kernel: bool = False
    apply_routing_weights_before_fc2: bool = True
    combine_dtype: str = "bf16"
    gate_up_clamp: float | None = None
    max_sm_count: int | None = None
    die_sm_mask32: tuple[int, ...] | None = None
    shared: tuple[int, int] | None = None

    @classmethod
    def make(
        cls,
        problem: RubinMoeProblem,
        bucket: RubinBucket,
        *,
        rank: int,
        world_size: int,
    ) -> RubinMoeConfig:
        k = bucket.knobs
        return cls(
            num_total_experts=problem.num_experts,
            max_tokens_per_rank=bucket.max_tokens,
            num_topk=problem.top_k,
            hidden=problem.hidden,
            intermediate=problem.intermediate,
            rank=rank,
            world_size=world_size,
            quant_kind=problem.quant_kind,
            mma_tiler_mnk=tuple(k.mma_tiler_mnk),
            cluster_shape_mn=tuple(k.cluster_shape_mn),
            fallback_cluster_shape_mn=(
                None
                if k.fallback_cluster_shape_mn is None
                else tuple(k.fallback_cluster_shape_mn)
            ),
            schedule_policy=tuple(k.schedule_policy),
            work_id_mode=k.work_id_mode,
            fc2_use_bulk=k.fc2_use_bulk,
            fc2_tma_stages=k.fc2_tma_stages,
            epi_flag_batches=tuple(k.epi_flag_batches),
            token_in_flag_batch=k.token_in_flag_batch,
            token_back_mode=k.token_back_mode,
            situ_beta=float(problem.situ_beta),
            situ_linear_beta=float(problem.situ_linear_beta),
            kernel_variant=bucket.variant,
            reduce_topk_in_kernel=k.reduce_topk_in_kernel,
            max_sm_count=bucket.max_sm_count,
            die_sm_mask32=bucket.die_sm_mask32,
            shared=bucket.shared,
        )

    def __post_init__(self) -> None:  # noqa: C901 - verbatim rule list
        if self.quant_kind not in _KIND_TABLE:
            raise ValueError(f"unsupported quant_kind {self.quant_kind!r}.")
        if self.kernel_variant not in VARIANTS:
            raise ValueError(f"unsupported kernel_variant {self.kernel_variant!r}.")
        if self.shared is not None and self.kernel_variant != "genphase":
            raise ValueError("the fused shared expert runs on the GenPhase kernel only.")
        if self.combine_dtype != "bf16":
            raise ValueError("this backend runs BF16 combine only.")
        if self.reduce_topk_in_kernel:
            raise ValueError("this backend runs separate top-k reduction only.")
        if not self.apply_routing_weights_before_fc2:
            raise ValueError("this backend applies routing weights before FC2.")
        if self.gate_up_clamp is not None:
            raise ValueError("SiTU does not support gate_up_clamp.")
        for name in ("situ_beta", "situ_linear_beta"):
            value = getattr(self, name)
            if value is None or not math.isfinite(value) or value <= 0:
                raise ValueError(f"SiTU requires positive, finite {name}.")
        for name in (
            "num_total_experts",
            "max_tokens_per_rank",
            "num_topk",
            "hidden",
            "intermediate",
            "world_size",
            "token_padding_block",
            "sf_padding_block",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive Python integer.")
        if type(self.rank) is not int or not (0 <= self.rank < self.world_size):
            raise ValueError(f"invalid rank/world_size {self.rank}/{self.world_size}.")
        if type(self.fc2_use_bulk) is not bool:
            raise ValueError("fc2_use_bulk must be a Python bool.")
        if self.num_total_experts > 16384:
            raise ValueError("SM107 routing supports at most 16384 global experts.")
        if self.num_topk > self.num_total_experts:
            raise ValueError("num_topk must not exceed num_total_experts.")
        if self.padded_tokens_per_rank * self.num_topk > 2097152:
            raise ValueError("SM107 routing supports at most 2097152 routes per rank.")
        if self.num_total_experts % self.world_size != 0:
            raise ValueError(
                f"num_total_experts ({self.num_total_experts}) must divide evenly "
                f"across world_size ({self.world_size})."
            )
        vec = self.sf_vec_size
        if self.hidden % (4 * vec) != 0:
            raise ValueError(
                f"hidden ({self.hidden}) must be a multiple of {4 * vec} for "
                f"{self.quant_kind}."
            )
        if self.intermediate % (2 * vec) != 0 or self.intermediate % 32 != 0:
            raise ValueError(
                f"intermediate ({self.intermediate}) must be a multiple of "
                f"{max(2 * vec, 32)} for {self.quant_kind}."
            )
        if self.quant_kind in ("mxfp4", "mxfp4_mxfp8") and self.intermediate % 128:
            raise ValueError(
                f"{self.quant_kind} intermediate must be a multiple of 128."
            )
        tiler = self.mma_tiler_mnk
        instruction_k = self.instruction_k
        if len(tiler) != 3 or any(type(d) is not int or d <= 0 for d in tiler):
            raise ValueError("mma_tiler_mnk must be three positive Python integers.")
        if tiler[0] not in (128, 256):
            raise ValueError(f"Rubin instruction M must be 128 or 256, got {tiler[0]}.")
        if tiler[1] not in (64, 128, 256):
            raise ValueError(
                f"Rubin instruction N must be 64, 128 or 256, got {tiler[1]}."
            )
        if tiler[2] not in (2 * instruction_k, 4 * instruction_k):
            raise ValueError(
                f"Rubin {self.quant_kind} tile K must be {2 * instruction_k} or "
                f"{4 * instruction_k}, got {tiler[2]}."
            )
        if len(self.cluster_shape_mn) != 2:
            raise ValueError("cluster_shape_mn must be a 2-tuple.")
        cm, cn = self.cluster_shape_mn
        if type(cm) is not int or cm <= 0 or cm > 16 or cm & (cm - 1):
            raise ValueError("cluster M must be a power of two no greater than 16.")
        if type(cn) is not int or cn != 1:
            raise ValueError("the swap-AB FC12 path requires cluster N=1.")
        if tiler[0] == 256 and cm % 2 != 0:
            raise ValueError("instruction M 256 requires an even cluster M.")
        if self.fallback_cluster_shape_mn is not None:
            fallback = tuple(self.fallback_cluster_shape_mn)
            if fallback == tuple(self.cluster_shape_mn):
                object.__setattr__(self, "fallback_cluster_shape_mn", None)
            else:
                if len(fallback) != 2 or any(
                    type(d) is not int or d <= 0 for d in fallback
                ):
                    raise ValueError(
                        "fallback_cluster_shape_mn must be 2 positive ints."
                    )
                if fallback[1] != 1:
                    raise ValueError(
                        "the swap-AB FC12 fallback path requires cluster N=1."
                    )
                if any(p % f for p, f in zip(self.cluster_shape_mn, fallback)):
                    raise ValueError(
                        "every preferred cluster dimension must be divisible by its "
                        "fallback dimension."
                    )
                if tiler[0] == 256 and fallback[0] % 2 != 0:
                    raise ValueError(
                        "instruction M 256 requires an even fallback cluster M."
                    )
                if self.max_sm_count is not None:
                    raise ValueError(
                        "max_sm_count is not supported with fallback_cluster_shape_mn."
                    )
        if self.token_padding_block % 64 != 0:
            raise ValueError("token_padding_block must be a multiple of 64.")
        if self.sf_padding_block % 128 != 0:
            raise ValueError("sf_padding_block must be a multiple of 128.")
        if tiler[1] % self.token_padding_block != 0:
            raise ValueError(
                f"mma tiler N ({tiler[1]}) must be a whole number of token padding "
                f"blocks ({self.token_padding_block})."
            )
        if len(self.schedule_policy) != 2:
            raise ValueError("schedule_policy requires (mode, hint).")
        mode, hint = self.schedule_policy
        modes = ("grouped", "phase_interleave") + (
            ("decode",) if self.kernel_variant == "genphase" else ()
        )
        if mode not in modes:
            raise ValueError(f"unknown schedule mode {mode!r}.")
        if hint is not None and (type(hint) is not int or hint <= 0):
            raise ValueError("schedule hint must be a positive Python integer or None.")
        if self.work_id_mode not in ("grid_stride", "atomic_counter"):
            raise ValueError(f"unknown work_id_mode {self.work_id_mode!r}.")
        if self.token_back_mode not in (
            "epi_warps",
            "standalone_warps",
            "reuse_dispatch_warps",
        ):
            raise ValueError(f"unknown token_back_mode {self.token_back_mode!r}.")
        if mode == "phase_interleave" and self.work_id_mode != "atomic_counter":
            raise ValueError(
                "schedule_policy 'phase_interleave' requires "
                "work_id_mode='atomic_counter'."
            )
        if len(self.epi_flag_batches) != 2 or any(
            type(b) is not int or not 1 <= b <= 4 for b in self.epi_flag_batches
        ):
            raise ValueError("Rubin epi_flag_batches must be two values in [1, 4].")
        if type(self.token_in_flag_batch) is not int or not (
            1 <= self.token_in_flag_batch <= 32
        ):
            raise ValueError("token_in_flag_batch must be in [1, 32].")
        if self.max_sm_count is not None and (
            type(self.max_sm_count) is not int or self.max_sm_count < cm
        ):
            raise ValueError("max_sm_count must accommodate at least one cluster.")
        if self.die_sm_mask32 is not None:
            # GenPhase: the decode scheduler's expert_half form (each die claims the
            # tiles of its homed experts, the die from %locality_domain_id).
            schedule = "decode" if self.kernel_variant == "genphase" else (
                "phase_interleave"
            )
            if (
                tuple(self.schedule_policy)[0] != schedule
                or self.fallback_cluster_shape_mn is not None
            ):
                raise ValueError(
                    f"die-local expert streams need the {self.kernel_variant} "
                    f"kernel's {schedule} schedule with one cluster shape."
                )
            if not self.die_sm_mask32 or not all(
                type(w) is int and 0 <= w < (1 << 32) for w in self.die_sm_mask32
            ):
                raise ValueError("die_sm_mask32 must be a tuple of 32-bit ints.")
        if self.fc2_tma_stages is not None and not (
            type(self.fc2_tma_stages) is int
            and 1 <= self.fc2_tma_stages <= tiler[1] // 64
        ):
            raise ValueError(
                f"fc2_tma_stages must be in [1, {tiler[1] // 64}] for tiler N "
                f"{tiler[1]}."
            )

        if self.kernel_variant == "genphase":
            # The refreshed GenPhase kernel (src_gp) validates its cluster, tile and
            # schedule geometry itself (decode scheduler, die-aware forms).
            if self.padded_tokens_per_rank > 1024:
                raise ValueError("GenPhase supports at most 1024 tokens per rank.")
            if not self.fc2_use_bulk:
                raise ValueError("GenPhase requires fc2_use_bulk=True.")
            if tiler[1] == 64 and self.world_size > 1 and mode == "phase_interleave":
                raise ValueError(
                    "GenPhase N64 with world_size > 1 requires grouped or decode "
                    "scheduling; multirank phase_interleave is not qualified."
                )
            if tiler[2] != 2 * instruction_k:
                raise ValueError(
                    f"GenPhase {self.quant_kind} requires tile K {2 * instruction_k}."
                )
            if self.sf_padding_block != 128 or 128 % self.token_padding_block:
                raise ValueError(
                    "GenPhase requires SF padding 128 and token padding 64 or 128."
                )
            if self.token_back_mode != "epi_warps" or self.token_in_flag_batch != 1:
                raise ValueError(
                    "GenPhase does not use token-back or token-in flag knobs."
                )

        if self.kernel_variant in DEDUP_VARIANTS:
            if self.token_back_mode == "epi_warps":
                raise ValueError(
                    "the dedup combine pushes rank rows from the combine warps "
                    "(standalone_warps or reuse_dispatch_warps)."
                )
            if not self.fc2_use_bulk:
                raise ValueError("the dedup combine requires fc2_use_bulk=True.")
            if self.die_sm_mask32 is not None:
                raise ValueError("the dedup inference kernel has no die-split schedule.")
            if self.world_size > 32:
                raise ValueError("the dedup combine supports at most 32 ranks.")

    # -- derived ----------------------------------------------------------------------

    @property
    def experts_per_rank(self) -> int:
        return self.num_total_experts // self.world_size

    @property
    def padded_tokens_per_rank(self) -> int:
        """Physical capacity keeps the router's four-int vector loads in bounds."""
        multiple = 4 // math.gcd(self.num_topk, 4)
        return (self.max_tokens_per_rank + multiple - 1) // multiple * multiple

    @property
    def sf_vec_size(self) -> int:
        return _KIND_TABLE[self.quant_kind][2]

    @property
    def instruction_k(self) -> int:
        return _KIND_TABLE[self.quant_kind][3]

    @property
    def torch_act_data_dtype(self) -> torch.dtype:
        dtype = _KIND_TABLE[self.quant_kind][0]
        return fp4_storage_dtype() if dtype is None else dtype

    @property
    def torch_sf_dtype(self) -> torch.dtype:
        return _KIND_TABLE[self.quant_kind][1]

    @property
    def activation_row_bytes(self) -> int:
        packed_fp4 = _KIND_TABLE[self.quant_kind][0] is None
        return self.hidden // 2 if packed_fp4 else self.hidden


# --------------------------------------------------------------------------------------
# Knob JSON reader (FlashInfer v2 knob cache, read-only)
# --------------------------------------------------------------------------------------


class KnobLookupError(LookupError):
    pass


class RouteIdError(ValueError):
    """A routed expert id other than -1 outside ``[0, num_experts)``.

    ``num_experts`` is the PHYSICAL expert count of the EP group (after any EPLB
    logical-to-physical mapping) and placement is linear (global id ``g`` on rank
    ``g // E_local``); ids are never silently dropped."""

    def __init__(self, expert_id: int, num_experts: int):
        self.expert_id = int(expert_id)
        self.num_experts = int(num_experts)
        super().__init__(
            f"routed expert id {self.expert_id} is outside [0, {self.num_experts}) "
            "and is not -1 (masked); num_experts is the physical expert count with "
            "linear EP placement."
        )


_MATCH_FIELDS = (
    "backend_revision",
    "kernel_variant",
    "allow_nondeterministic",
    "apply_routing_weights_before_fc2",
    "dtype",
    "world_size",
    "hidden",
    "intermediate",
    "num_experts",
    "topk",
    "combine_dtype",
    "activation",
    "situ_beta",
    "situ_linear_beta",
    "gate_up_clamp",
)


def load_knob_source(source: str | Mapping[str, Any]) -> Any:
    """A path, an inline JSON string or an already-parsed mapping."""
    if isinstance(source, Mapping):
        return dict(source)
    text = source.strip()
    if text.startswith("{"):
        return json.loads(text)
    with open(os.path.expanduser(source)) as f:
        return json.load(f)


def _entries_of(data: Any) -> list[dict[str, Any]] | None:
    """Entries of a v2 file / single record, or None for a bare knob dict."""
    if not isinstance(data, Mapping):
        raise ValueError("knob source must be a JSON object.")
    if "entries" in data:
        if data.get("version") != KNOB_CACHE_VERSION:
            raise ValueError(
                f"knob file version {data.get('version')!r} != {KNOB_CACHE_VERSION}."
            )
        return [e for e in data["entries"] if isinstance(e, Mapping)]
    if "knobs" in data:
        return [dict(data)]
    if set(data) <= set(KNOB_KEYS):
        return None
    raise ValueError(
        "knob source is neither a v2 knob file, a record, nor a knob dict."
    )


def _live_buckets(entries: list[dict[str, Any]], capacity: int) -> list[int]:
    return sorted(
        {
            e.get("live_tokens", capacity)
            for e in entries
            if e["max_tokens"] == capacity
            and type(e.get("live_tokens", capacity)) is int
            and 0 <= e.get("live_tokens", capacity) <= capacity
        }
    )


def _select_token_entries(
    entries: list[dict[str, Any]], capacity: int, live_tokens: int | None
) -> list[dict[str, Any]]:
    """FlashInfer core/tuning.py ``select_token_entries``."""
    if live_tokens is not None and not 0 <= live_tokens <= capacity:
        raise ValueError("live_tokens must be between zero and max_tokens.")
    buckets = _live_buckets(entries, capacity)
    if buckets:
        workload = capacity if live_tokens is None else live_tokens
        bucket = next((b for b in buckets if b >= workload), buckets[-1])
        return [
            e
            for e in entries
            if e["max_tokens"] == capacity and e.get("live_tokens", capacity) == bucket
        ]
    return [
        e
        for e in entries
        if e.get("live_tokens") is None or e.get("live_tokens") == e["max_tokens"]
    ]


def resolve_knobs(
    source: str | Mapping[str, Any],
    *,
    problem: RubinMoeProblem,
    variant: Variant,
    world_size: int,
    max_tokens: int,
    live_tokens: int | None = None,
    device_name: str | None = None,
) -> tuple[RubinKnobs, dict[str, Any] | None, list[str]]:
    """``(knobs, matched record or None for an explicit knob dict, warnings)``.

    Matching follows FlashInfer ``lookup_record``: every contract field must match;
    among the records selected for (``max_tokens``, ``live_tokens``) the smallest
    capacity >= ``max_tokens`` wins, else the largest smaller one. Equal-capacity ties
    (two tuner runs of one shape) go to the lowest recorded p50. The record's ``device``
    is compared with ``sm107:<device_name>`` and only warns on mismatch.
    """
    entries = _entries_of(load_knob_source(source))
    warnings: list[str] = []
    if entries is None:
        return RubinKnobs.from_dict(load_knob_source(source)), None, warnings
    matches = _matching_records(entries, problem, variant, world_size)
    if not matches:
        # Redundant experts (num_experts padded past the checkpoint's 896) or an EP
        # size without tuned records (EP12, EP20, EP24): borrow the nearest tuned
        # contract of the same variant / dtype (records tuned at the logical expert
        # count, the closest world size with per-rank experts >= this one's). The
        # knobs are tile / schedule choices valid for any expert count; only their
        # tuning is borrowed.
        matches, note = _nearest_contract_records(entries, problem, variant, world_size)
        if note:
            warnings.append(note)
    selected = _select_token_entries(matches, max_tokens, live_tokens)
    if not selected:
        available = sorted(
            {
                (
                    e.get("kernel_variant"),
                    e.get("dtype"),
                    e.get("world_size"),
                    e.get("max_tokens"),
                    e.get("live_tokens"),
                )
                for e in entries
            },
            key=str,
        )
        raise KnobLookupError(
            f"no knob record for variant={variant} dtype={problem.quant_kind} "
            f"world_size={world_size} max_tokens={max_tokens} "
            f"live_tokens={live_tokens} (H={problem.hidden} I={problem.intermediate} "
            f"E={problem.num_experts} topk={problem.top_k} SiTU "
            f"{problem.situ_beta}/{problem.situ_linear_beta}, revision "
            f"{BACKEND_REVISION}). Available (variant, dtype, world, max, live): "
            f"{available}"
        )
    above = [e for e in selected if e["max_tokens"] >= max_tokens]
    if above:
        cap = min(e["max_tokens"] for e in above)
    else:
        cap = max(e["max_tokens"] for e in selected)
    tied = [e for e in selected if e["max_tokens"] == cap]
    best = min(
        enumerate(tied),
        key=lambda ie: (
            float("inf") if ie[1].get("p50_us") is None else ie[1]["p50_us"],
            ie[0],
        ),
    )[1]
    if device_name is not None and best.get("device") != "sm107:" + device_name:
        warnings.append(
            f"knob record device {best.get('device')!r} != 'sm107:{device_name}'"
        )
    if best["max_tokens"] != max_tokens:
        warnings.append(
            f"knob record tuned at max_tokens {best['max_tokens']} used for "
            f"{max_tokens}"
        )
    return RubinKnobs.from_dict(best["knobs"]), dict(best), warnings


def _matching_records(
    entries: list[dict[str, Any]],
    problem: RubinMoeProblem,
    variant: Variant,
    world_size: int,
) -> list[dict[str, Any]]:
    """Records whose contract fields all equal this problem's (FlashInfer
    ``lookup_record`` key match), before token-bucket selection."""
    key = dict(
        backend_revision=BACKEND_REVISION,
        kernel_variant=variant,
        allow_nondeterministic=False,
        apply_routing_weights_before_fc2=True,
        dtype=problem.quant_kind,
        world_size=world_size,
        hidden=problem.hidden,
        intermediate=problem.intermediate,
        num_experts=problem.num_experts,
        topk=problem.top_k,
        combine_dtype="bf16",
        activation="situ",
        situ_beta=float(problem.situ_beta),
        situ_linear_beta=float(problem.situ_linear_beta),
        gate_up_clamp=None,
    )
    return [
        e
        for e in entries
        if all(e.get(f) == key[f] for f in _MATCH_FIELDS)
        and isinstance(e.get("knobs"), Mapping)
        and isinstance(e.get("max_tokens"), int)
        and not e["knobs"].get("reduce_topk_in_kernel", False)
    ]


def _nearest_contract_records(
    entries: list[dict[str, Any]],
    problem: RubinMoeProblem,
    variant: Variant,
    world_size: int,
) -> tuple[list[dict[str, Any]], str | None]:
    """Records of the tuned contract closest to (``problem``, ``world_size``) when none
    match exactly: same fields except ``num_experts`` and ``world_size``; among the
    tuned (num_experts, world_size) pairs, the one whose experts per rank is closest
    to ``problem.num_experts / world_size`` (ties: larger world size)."""
    base = {}
    for ws in {e.get("world_size") for e in entries}:
        for ne in {e.get("num_experts") for e in entries}:
            if not isinstance(ws, int) or not isinstance(ne, int):
                continue
            probe = dataclasses.replace(problem, num_experts=ne)
            recs = _matching_records(entries, probe, variant, ws)
            if recs:
                base[(ne, ws)] = recs
    if not base:
        return [], None
    want = problem.num_experts / world_size
    (ne, ws), recs = min(
        base.items(), key=lambda kv: (abs(kv[0][0] / kv[0][1] - want), -kv[0][1])
    )
    return recs, (
        f"no knob record for E={problem.num_experts} at EP{world_size}; borrowed the "
        f"E={ne} EP{ws} records ({ne // ws} experts/rank vs {want:.1f})"
    )


def default_capacity(tokens: int) -> int:
    """The tuner's capacity rounding for a live token count (128 below 128, else a
    multiple of 256)."""
    step = 128 if tokens <= 128 else 256
    return max(step, -(-tokens // step) * step)


def tuned_capacity(
    source: str | Mapping[str, Any],
    *,
    problem: RubinMoeProblem,
    variant: Variant,
    world_size: int,
    live_tokens: int,
) -> int | None:
    """``max_tokens`` of the knob record tuned at exactly ``live_tokens`` for this
    contract (the capacity the tuner compiled for that point), or None when no record
    has that live count (or the source is a bare knob dict). Several such records with
    different capacities resolve to the one with the lowest recorded p50."""
    entries = _entries_of(load_knob_source(source))
    if entries is None:
        return None
    exact = [
        e
        for e in _matching_records(entries, problem, variant, world_size)
        if e.get("live_tokens", e["max_tokens"]) == live_tokens
        and e["max_tokens"] >= live_tokens
    ]
    if not exact:
        return None
    best = min(
        enumerate(exact),
        key=lambda ie: (
            float("inf") if ie[1].get("p50_us") is None else ie[1]["p50_us"],
            ie[0],
        ),
    )[1]
    return int(best["max_tokens"])


def bench_capacity(
    source: str | Mapping[str, Any],
    *,
    problem: RubinMoeProblem,
    variant: Variant,
    world_size: int,
    live_tokens: int,
) -> int:
    """Capacity for a benchmark point without an explicit one: the tuned record's
    ``max_tokens`` for exactly this live count, else :func:`default_capacity`."""
    tuned = tuned_capacity(
        source,
        problem=problem,
        variant=variant,
        world_size=world_size,
        live_tokens=live_tokens,
    )
    return default_capacity(live_tokens) if tuned is None else tuned


# ---------------------------------------------------------------------------------
# NVFP4 norm constants from activation statistics
# ---------------------------------------------------------------------------------

# Largest finite E2M1 magnitude x largest finite E4M3 block scale.
NVFP4_RANGE = 6.0 * 448.0


def nvfp4_norm_const_from_amax(amax: float, *, headroom: float = 1.0) -> float:
    """Power-of-two NVFP4 norm constant for a tensor whose absolute max is ``amax``.

    Staging (and the in-kernel FC1->FC2 requantization) encode a 16-element block with
    the E4M3 scale ``s = e4m3(amax_block * n / 6)`` and codes ``v * n / s``. A larger
    ``n`` moves ``s`` out of the E4M3 subnormal range (below 2^-6) and away from the
    zero flush (below 2^-9), but ``s`` saturates at 448 once ``amax_block * n / 6 >
    448``. The returned ``n`` is the largest power of two with
    ``amax * headroom * n <= 6 * 448``, so the largest block of the calibration tensor
    still has a finite, unsaturated scale.

    Pure function. ``amax`` must come from the caller's own statistics: for
    ``input_norm_const`` the absolute max of the BF16 MoE input, for
    ``fc1_norm_const`` that of the FC1 output after SiTU and the routing weight (the
    FC1->FC2 wire). ``headroom >= 1`` leaves room for unseen larger activations (2.0
    halves ``n``). The runtime never derives constants itself: ``RubinMoeProblem``
    takes explicit values and defaults to FlashInfer's 1.0. Served layers use
    :func:`nvfp4_fc1_norm_const` (no calibration data needed).
    """
    amax = float(amax)
    headroom = float(headroom)
    if not math.isfinite(amax) or amax <= 0.0:
        raise ValueError(f"amax must be positive and finite, got {amax!r}.")
    if not math.isfinite(headroom) or headroom < 1.0:
        raise ValueError(f"headroom must be finite and >= 1, got {headroom!r}.")
    exponent = math.floor(math.log2(NVFP4_RANGE / (amax * headroom)))
    # log2 rounding at exact powers of two: step down until the bound holds.
    while amax * headroom * 2.0**exponent > NVFP4_RANGE:
        exponent -= 1
    while amax * headroom * 2.0 ** (exponent + 1) <= NVFP4_RANGE:
        exponent += 1
    return 2.0**exponent


def _pow2_floor(x: float) -> float:
    exponent = math.floor(math.log2(x))
    while 2.0**exponent > x:
        exponent -= 1
    while 2.0 ** (exponent + 1) <= x:
        exponent += 1
    return 2.0**exponent


def nvfp4_fc1_norm_const(
    *,
    top_k: int,
    situ_beta: float,
    situ_linear_beta: float,
    fc2_input_gscale: float = 1.0,
    routed_scaling_factor: float = 1.0,
) -> float:
    """``fc1_norm_const`` of an NVFP4 MegaMoE layer: the FC1->FC2 wire global scale.

    Stock NVFP4 MoE (vLLM ModelOpt -> FlashInfer TRT-LLM-Gen,
    ``fused_moe/oracle/nvfp4.py:602`` ``a2_gscale = 1 / w2_input_scale``,
    ``experts/trtllm_nvfp4_moe.py:144-145`` SiTU ``output1_scale_scalar =
    a2_gscale``) requantizes the activation output ``u = SiTU(fc1)`` with the global
    scale ``g = fc2_input_gscale`` and applies the routing weight ``w`` after FC2, in
    finalize (``trtllm_nvfp4_moe.py:541-546``). This kernel requantizes ``w * u``
    instead, so its E4M3 block scales ``amax(w * u) * n / 6`` equal stock's
    ``amax(u) * g / 6`` iff ``n = g / w``. With ``n = g`` (FlashInfer's 1.0) the
    wire sits ``1 / w`` (16x on average at top-16) below stock and its block scales
    flush to zero (measured on K3.1 real-text prefill: 94-99.7 % of the 16-blocks).

    Returned: the largest power of two (exact alpha folding) not above
    ``max(g / max(rsf, 1), min(g * top_k / rsf, 6 * 448 / (beta * beta_lin * rsf)))``:

    - ``g * top_k / rsf``: stock's block scales at the mean routing weight
      ``rsf / top_k`` (renormalized top-k); heavier routes get larger scales than stock.
    - ``6 * 448 / (beta * beta_lin * rsf)``: no block scale can saturate.
      ``|SiTU(gate, up)| < beta * beta_lin`` (``beta tanh(gate / beta) sigmoid(gate)
      < beta``, ``beta_lin tanh(up / beta_lin) < beta_lin``) and ``w <= rsf``, so
      ``amax_block(w * u) * n / 6 <= 448``.
    - ``g / max(rsf, 1)``: a calibrated ``g`` (stock's own non-saturating scale for
      ``u``) is never lowered further, since ``w * u <= rsf * u``.

    Kimi K3.1 (top-16, beta 4, beta_lin 25, rsf 1.0, checkpoint ``w2_input_scale``
    1.0 or no activation scale at all): ``min(16, 26.88) = 16``, i.e. stock's scale.
    """
    values = (
        ("situ_beta", situ_beta),
        ("situ_linear_beta", situ_linear_beta),
        ("fc2_input_gscale", fc2_input_gscale),
        ("routed_scaling_factor", routed_scaling_factor),
    )
    for name, value in values:
        value = float(value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite, got {value!r}.")
    if type(top_k) is not int or top_k <= 0:
        raise ValueError(f"top_k must be a positive int, got {top_k!r}.")
    g = float(fc2_input_gscale)
    rsf = float(routed_scaling_factor)
    unsaturated = NVFP4_RANGE / (float(situ_beta) * float(situ_linear_beta) * rsf)
    return _pow2_floor(max(g / max(rsf, 1.0), min(g * top_k / rsf, unsaturated)))


# ---------------------------------------------------------------------------------
# Knob file merge (tooling: produces knobs/k31_vr.json from tuner outputs)
# ---------------------------------------------------------------------------------

# Record fields kept in the merged file (``protocol_contract`` is reduced to the
# environment versions recorded under ``provenance``).
_MERGED_FIELDS = (
    *_MATCH_FIELDS,
    "device",
    "max_tokens",
    "live_tokens",
    "knobs",
    "p50_us",
    "source",
    "tuned_at",
    "measurement_policy",
    "profiling_policy",
)


def merge_knob_files(
    files: Sequence[tuple[str, bytes]], *, comment: str
) -> dict[str, Any]:
    """Merge FlashInfer v2 knob-cache files into one v2 file.

    ``files`` are ``(provenance name, raw file bytes)``; every record keeps its
    contract fields, knobs, p50 and timer policy, plus ``provenance`` = (file name,
    sha256 of the file bytes, FlashInfer / CUDA versions). Raises unless every source
    file is a v2 file contributing at least one record and the merged record count
    equals the sum of the source record counts (nothing dropped or duplicated).
    """
    names = [name for name, _ in files]
    if len(set(names)) != len(names):
        raise ValueError("duplicate source file names.")
    merged: list[dict[str, Any]] = []
    expected = 0
    per_file: dict[str, int] = {}
    for name, raw in files:
        data = json.loads(raw)
        if not isinstance(data, Mapping) or data.get("version") != KNOB_CACHE_VERSION:
            raise ValueError(f"{name}: not a v{KNOB_CACHE_VERSION} knob file.")
        entries = [e for e in data.get("entries", []) if isinstance(e, Mapping)]
        if not entries or len(entries) != len(data.get("entries", [])):
            raise ValueError(f"{name}: no (or malformed) records.")
        digest = hashlib.sha256(raw).hexdigest()
        expected += len(entries)
        per_file[name] = len(entries)
        for e in entries:
            env = (e.get("protocol_contract") or {}).get("environment") or {}
            record = {f: e.get(f) for f in _MERGED_FIELDS if f in e}
            record["provenance"] = {
                "file": name,
                "sha256": digest,
                "flashinfer_version": env.get("flashinfer_version"),
                "cuda_version": env.get("cuda_version"),
            }
            merged.append(record)
    if len(merged) != expected or set(per_file) != set(names):
        raise AssertionError(
            f"knob merge lost records: {len(merged)} merged, {expected} in sources."
        )
    return {
        "version": KNOB_CACHE_VERSION,
        "comment": comment,
        "source_files": len(files),
        "entries": merged,
    }


def make_buckets(
    source: str | Mapping[str, Any],
    *,
    problem: RubinMoeProblem,
    world_size: int,
    points: Sequence[tuple[Variant, int, int | None]],
    device_name: str | None = None,
) -> tuple[list[RubinBucket], list[str]]:
    """Buckets for ``(variant, max_tokens, live_tokens)`` points from one knob source.
    """
    buckets, notes = [], []
    for variant, max_tokens, live in points:
        knobs, record, warns = resolve_knobs(
            source,
            problem=problem,
            variant=variant,
            world_size=world_size,
            max_tokens=max_tokens,
            live_tokens=live,
            device_name=device_name,
        )
        buckets.append(RubinBucket(variant, max_tokens, knobs))
        notes.extend(f"[{variant} {max_tokens}] {w}" for w in warns)
        if record is not None:
            prov = record.get("provenance", {}).get("file", "?")
            notes.append(
                f"[{variant} {max_tokens}] knobs from {prov} (tuned "
                f"{record['max_tokens']}"
                f"/{record.get('live_tokens')}, p50 {record.get('p50_us')} us)"
            )
    return buckets, notes
