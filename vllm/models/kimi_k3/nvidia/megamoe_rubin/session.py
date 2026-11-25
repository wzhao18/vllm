# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One compiled Rubin MegaMoE kernel plus its workspace (a "session").

Ported from FlashInfer ``shim/block_scaled.py`` (``Sm107BlockScaledSymmBuffer``:
``_build_kernel``, ``_runtime_kwargs``, ``prepare`` / ``launch``; hash in VENDOR.md)
with the NVSHMEM allocations replaced by views of the :class:`SymmetricArena`:

- symmetric (peer-read / peer-written): ``x``, ``x_sf``, ``topk_weights`` and the
  kernel's shared workspace (router sizes, token metadata, FC1 scores, the BF16 combine
  landing zone, NVLink barrier signals)
- local: ``topk_idx`` (int32, initialized to -1) and the local workspace (grid / phase
  counters, FC1 pools); both persistent and never shared between sessions

A launch is ``compiled(**kwargs)``: router, the PDL MegaMoE kernel, then the separate
BF16 ``TopkReduce`` writing ``out[:n]`` directly.

One kernel per session (the a368c5f66 drop has a single epilogue variant). No host sync, allocation or masked
fill (BF16 separate reduction masks invalid slots in-kernel: asserted at build), so it
is CUDA graph safe after warmup. Every EP rank must launch the same session the same
number of times in the same order (the barrier state persists across launches).
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, NamedTuple

import torch

from .config import DEDUP_VARIANTS, RubinMoeConfig
from .symm_arena import ALIGN, SymmetricArena, aligned, ensure_not_capturing
from .weights import RubinMoeWeights, validate_kernel_weights

_SCALAR_NAMES = ("fc1_alpha", "fc2_alpha", "fc1_norm_const")
# FlashInfer ``die_local.SM_LOCALITY_MAP_SIZE`` (the kernel's die map extent).
SM_LOCALITY_MAP_SIZE = 1024
_MAX_ACTIVE_CLUSTERS: dict[tuple[int, int], int] = {}
_COMPILED: dict[tuple, Any] = {}
_BIND_CACHE_LIMIT = 1024
_OUTPUT_CACHE_LIMIT = 4



def ep_mean_tokens(num_tokens_across_dp: torch.Tensor, sp_size: int) -> int:
    """Live tokens per EP rank averaged over the group, rounded up: each DP batch of
    ``n`` tokens is split over ``sp_size`` ranks as ``ceil(n / sp_size)`` rows
    (sequence_parallel_chunk; ``sp_size`` 1 without sequence parallelism)."""
    if sp_size < 1 or num_tokens_across_dp.numel() == 0:
        raise ValueError("need sp_size >= 1 and at least one DP rank.")
    shards = (num_tokens_across_dp.to(torch.int64) + sp_size - 1) // sp_size
    return -(-int(shards.sum()) // shards.numel())


# Fused shared expert: shared FC2 after the routed work ("tail") or right after shared
# FC1, under the dispatch ("early").
RUBIN_FUSED_SHARED_L2_POSITION = "early"


def compile_cache_key(
    cfg: RubinMoeConfig, kernel_name: str, cutlass_version: str, device: int
) -> tuple:
    """Process-wide compiled-kernel key. The kernel name encodes the constexprs of
    both variants, except that the genphase name omits the SiTU betas (codegen
    constants) and the die-local SM mask (per GPU: its die map; the name, which
    also tags the EP-uniform arena layout, only says "split"), so they, the quant
    kind and the device are keyed explicitly."""
    return (
        kernel_name,
        str(cutlass_version),
        cfg.quant_kind,
        float(cfg.situ_beta),
        float(cfg.situ_linear_beta),
        int(device),
        getattr(cfg, "die_sm_mask32", None),
        cfg.kernel_variant,
        getattr(cfg, "shared", None),
    )


def layout_tag(index: int, cfg: RubinMoeConfig, kernel_name: str) -> str:
    """Arena carve tag of a session. It carries the kernel name (which encodes every
    constexpr / knob) plus the quant kind and SiTU betas (absent from the genphase
    name), so ``SymmetricArena.verify_layout`` also catches ranks that resolved
    different knobs or problems with equal workspace sizes."""
    return (
        f"b{index}:{cfg.kernel_variant}:{cfg.max_tokens_per_rank}:{cfg.quant_kind}:"
        f"{float(cfg.situ_beta)}/{float(cfg.situ_linear_beta)}:{kernel_name}"
    )


def output_binding_key(output: torch.Tensor) -> tuple:
    """Output binding cache key: everything ``_check_output`` validates, so a hit
    never skips validation of a differently typed / strided / placed tensor."""
    return (
        output.data_ptr(),
        tuple(output.shape),
        tuple(output.stride()),
        output.dtype,
        output.device,
    )


def _max_active_clusters(cluster_size: int) -> int:
    key = (torch.cuda.current_device(), cluster_size)
    cached = _MAX_ACTIVE_CLUSTERS.get(key)
    if cached is None:
        import cutlass.utils as cutlass_utils

        cached = int(cutlass_utils.HardwareInfo().get_max_active_clusters(cluster_size))
        _MAX_ACTIVE_CLUSTERS[key] = cached
    return cached


def _src(cfg: RubinMoeConfig):
    """The vendored source tree of ``cfg``'s kernel: the refreshed FlashInfer main
    export (``src_gp``, VENDOR.md) for GenPhase, the a368c5f66 drop (``src``) for
    the inference kernel."""
    import importlib

    name = "src" if cfg.kernel_variant == "inference" else "src_gp"
    pkg = importlib.import_module(f"{__package__}.{name}")
    importlib.import_module(f"{__package__}.{name}.sources.api")
    importlib.import_module(f"{__package__}.{name}.sources.quant_def")
    importlib.import_module(
        f"{__package__}.{name}.sources.communication.nvlink_domain.symmetric_buffer"
    )
    return pkg


def build_kernel(cfg: RubinMoeConfig, epilogue_warpgroups: int | None = None):
    """FlashInfer ``Sm107BlockScaledSymmBuffer._build_kernel`` (BF16 combine, SiTU).

    ``epilogue_warpgroups``: None or 1 (the a368c5f66 kernels have one epilogue
    warpgroup; the S2 variant of the previous drop is gone)."""
    import cutlass
    from cutlass.cute.nvgpu import OperandMajorMode

    src = _src(cfg)
    ImplDesc, ProblemDesc = src.sources.api.ImplDesc, src.sources.api.ProblemDesc
    CombineFormat = src.sources.quant_def.CombineFormat

    tiler = tuple(cfg.mma_tiler_mnk)
    cluster_size = cfg.cluster_shape_mn[0] * cfg.cluster_shape_mn[1]
    preferred_count = fallback_count = None
    if cfg.fallback_cluster_shape_mn is None:
        launch_clusters = _max_active_clusters(cluster_size)
        if cfg.max_sm_count is not None:
            requested = cfg.max_sm_count // cluster_size
            if requested <= 0:
                raise ValueError("max_sm_count must cover at least one full cluster.")
            launch_clusters = min(launch_clusters, requested)
    else:
        # Mixed CGA: preferred clusters at their occupancy limit, leftover fallback CTA
        # capacity converted into whole preferred-sized groups of fallback clusters.
        fallback_size = (
            cfg.fallback_cluster_shape_mn[0] * cfg.fallback_cluster_shape_mn[1]
        )
        if cluster_size % fallback_size:
            raise ValueError(
                "preferred cluster size must be a multiple of the fallback cluster "
                "size."
            )
        preferred_count = _max_active_clusters(cluster_size)
        fallback_capacity = _max_active_clusters(fallback_size) * fallback_size
        preferred_capacity = preferred_count * cluster_size
        if fallback_capacity < preferred_capacity:
            raise ValueError(
                "fallback cluster CTA capacity must not be smaller than the preferred "
                "one."
            )
        split_factor = cluster_size // fallback_size
        remaining = (fallback_capacity - preferred_capacity) // fallback_size
        fallback_count = remaining // split_factor * split_factor
        launch_clusters = preferred_count + fallback_count // split_factor

    problem_desc = ProblemDesc(
        {
            "expert_count": cfg.num_total_experts,
            "intermediate_gateup_size": 2 * cfg.intermediate,
            "hidden_size": cfg.hidden,
            "quant_kind": cfg.quant_kind,
            "a_major_mode": OperandMajorMode.K,
            "b_major_mode": OperandMajorMode.K,
            "combine_format": CombineFormat.parse("bf16"),
            "gate_up_clamp": None,
            "situ_beta": cfg.situ_beta,
            "situ_linear_beta": cfg.situ_linear_beta,
            "world_size": cfg.world_size,
            "topk": cfg.num_topk,
            "topk_index_dtype": cutlass.Int32,
            "max_tokens_per_rank": cfg.padded_tokens_per_rank,
            "apply_topk_at_fc1": True,
            **(
                dict(
                    shared_expert=True,
                    shared_quant_kind="mxfp8_e4m3",
                    shared_hidden_size=cfg.shared[0],
                    shared_intermediate_gateup_size=2 * cfg.shared[1],
                    shared_output_size=cfg.shared[0],
                    reduce_shared_output=False,
                )
                if cfg.shared is not None
                else {}
            ),
        }
    )
    impl_fields: dict[str, Any] = {
        "mma_instruction_mnk": (tiler[0], tiler[1], cfg.instruction_k),
        "mma_tiler_mnk": tiler,
        "mma_k_mode": "2x",
        "cluster_shape_mn": tuple(cfg.cluster_shape_mn),
        "use_2cta_instrs": tiler[0] == 256,
        "schedule_policy": tuple(cfg.schedule_policy),
        "token_padding_block": cfg.token_padding_block,
        "sf_padding_block": cfg.sf_padding_block,
        "work_id_mode": cfg.work_id_mode,
        "fc2_use_bulk": cfg.fc2_use_bulk,
        "epi_flag_batches": tuple(cfg.epi_flag_batches),
        "launch_cluster_count": launch_clusters,
        "token_in_flag_batch": cfg.token_in_flag_batch,
        "token_back_mode": cfg.token_back_mode,
        "reduce_topk_in_kernel": False,
    }
    if cfg.fc2_tma_stages is not None:
        impl_fields["fc2_tma_stages"] = cfg.fc2_tma_stages
    if epilogue_warpgroups not in (None, 1):
        # FlashInfer a368c5f66 kernels have one epilogue warpgroup (the vLLM-local
        # S2 of the 0.7.3.dev20261031 drop is not carried over; VENDOR.md).
        raise ValueError(f"unsupported epilogue_warpgroups={epilogue_warpgroups}.")
    if getattr(cfg, "shared", None) is not None:
        impl_fields["shared_l2_position"] = RUBIN_FUSED_SHARED_L2_POSITION
    if cfg.die_sm_mask32 is not None and cfg.kernel_variant == "genphase":
        # Die-local expert streams on the refreshed GenPhase decode scheduler: each
        # die claims the tiles of the experts homed on it (die_local.place_experts:
        # [0, E/2) on die 0), reading its die from %locality_domain_id.
        impl_fields["die_aware"] = "expert_half"
        impl_fields["die_sm_mask32"] = tuple(cfg.die_sm_mask32)
    elif cfg.die_sm_mask32 is not None:
        # Die-local expert streams: FlashInfer's two-ended expert claim over the
        # die-homed weights (die_local.place_experts); the kernel reads the die of
        # each SM from the ``sm_locality_domain`` launch argument.
        if tuple(cfg.schedule_policy)[0] != "phase_interleave":
            raise ValueError("die-local streams need the phase-interleaved schedule.")
        impl_fields["die_split"] = True
    if cfg.fallback_cluster_shape_mn is not None:
        impl_fields["fallback_cluster_shape_mn"] = tuple(cfg.fallback_cluster_shape_mn)
        impl_fields["preferred_cluster_count"] = preferred_count
        impl_fields["fallback_cluster_count"] = fallback_count
    if cfg.kernel_variant == "genphase":
        for name in ("token_in_flag_batch", "token_back_mode"):
            impl_fields.pop(name)
    if cfg.kernel_variant == "inference_dedup_wg2":
        impl_fields["epilogue_warpgroups"] = 2
    if cfg.kernel_variant in DEDUP_VARIANTS:
        impl_fields["combine_dedup"] = True
        impl_fields["token_back_ready_granularity"] = "expert"
    kernel_cls = {  # src_gp exports at the package root
        "genphase": lambda: src.RubinInferenceGenphaseMegaMoE,
        "inference_dedup": lambda: src.RubinInferenceMegaMoE,
        "inference_dedup_wg2": lambda: src.RubinInferenceMegaMoE,
        "inference": lambda: src.sources.RubinInferenceMegaMoE,
    }[cfg.kernel_variant]()
    return kernel_cls(problem_desc, ImplDesc(impl_fields))


def build_kernels(cfg: RubinMoeConfig) -> dict[bool, Any]:
    """The session's kernel objects keyed by ``use_s2``: ``{False: kernel}`` (the
    FlashInfer a368c5f66 drop has one epilogue variant, VENDOR.md), plus
    ``True: narrow kernel`` when the bucket has one (``cfg.narrow``: the same variant
    built with other tile knobs, sharing the session's workspace; checked by
    :func:`narrow_local_offsets`)."""
    kernels = {False: build_kernel(cfg)}
    if cfg.narrow is not None:
        kernels[True] = build_kernel(cfg.narrow_config())
        narrow_local_offsets(kernels[False], kernels[True])
    return kernels


def narrow_local_offsets(wide, narrow) -> tuple[int, int]:
    """``(wide, narrow)`` byte offsets of the two kernels' local workspaces in ONE
    local buffer, so both kernels run over the same workspace (no second arena).

    A space is laid out tail_reset span (zeroed by every launch's tail), then the
    persistent kernel state (zero_on_first_allocate: call epochs, phase counters,
    the host-written record addresses), then per-launch data (DeviceWorkspace.
    finalize). Only the tail_reset span depends on the tile shape. Placing the
    kernel with the shorter span at the length difference makes every persistent
    and data region coincide, and its tail span lie inside the other's, which each
    launch leaves zeroed. So the per-call protocol state (CallTail epoch + shared
    done signals, software-sync phase + signals) is one state whichever kernel
    runs. Raises unless the shared spaces are identical and the local spaces agree
    past their tail spans."""
    lw, ln = _workspace_layout(wide), _workspace_layout(narrow)
    shared_w = next(x for x in lw if x[0] == "shared")
    shared_n = next(x for x in ln if x[0] == "shared")
    if shared_w != shared_n:
        raise RuntimeError("narrow kernel: shared workspace layouts differ.")
    ww, wn = wide._device_workspace, narrow._device_workspace
    tail_w, tail_n = ww.tail_reset_bytes("local"), wn.tail_reset_bytes("local")

    def past_tail(ws, tail):
        return sorted(
            (r.name, ws.offset(r.name) - tail, ws.nbytes(r.name), r.reset, str(r.dtype))
            for r in ws.regions("local")
            if r.reset != "tail_reset"
        )

    if (
        past_tail(ww, tail_w) != past_tail(wn, tail_n)
        or ww.total_bytes("local") - tail_w != wn.total_bytes("local") - tail_n
    ):
        raise RuntimeError("narrow kernel: local workspace state layouts differ.")
    offsets = (max(tail_n - tail_w, 0), max(tail_w - tail_n, 0))
    if any(o % 128 for o in offsets):
        raise RuntimeError(f"narrow kernel: local offsets {offsets} not 128-aligned.")
    return offsets


def _workspace_layout(kernel) -> tuple:
    ws = kernel._device_workspace
    return tuple(
        (
            space,
            ws.total_bytes(space),
            tuple(
                (r.name, ws.offset(r.name), ws.nbytes(r.name), r.reset, str(r.dtype))
                for r in ws.regions(space)
            ),
        )
        for space in ("local", "shared")
    )


def _to_cute(
    tensor: torch.Tensor, assumed_align: int = 16, dynamic_leading: bool = True
):
    import cutlass.torch as cutlass_torch

    cute_tensor = cutlass_torch.from_dlpack(tensor, assumed_align=assumed_align)
    if not dynamic_leading:
        return cute_tensor
    return cute_tensor.mark_layout_dynamic(
        leading_dim=cutlass_torch.get_leading_dim(tensor)
    )


def _to_cute_ptr(tensor: torch.Tensor, assumed_align: int = 128):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.typing import AddressSpace

    return cute.runtime.make_ptr(
        cutlass.Uint8, tensor.data_ptr(), AddressSpace.gmem, assumed_align=assumed_align
    )


def _cu_stream(stream: torch.cuda.Stream):
    import cuda.bindings.driver as cuda

    return cuda.CUstream(stream.cuda_stream)


class FusedSharedArgs(NamedTuple):
    """One launch's fused MXFP8 shared expert: ``x`` FP8 E4M3 [rows, hidden] and
    ``x_sf`` its F8_128x4 E8M0 scales (a capacity buffer, rows >= the bucket's);
    ``fc1`` / ``fc2`` the kernel-layout (weight, scale) pairs
    (``weights.prepare_mxfp8_shared_weights``)."""

    x: torch.Tensor
    x_sf: torch.Tensor
    fc1: tuple[torch.Tensor, torch.Tensor]
    fc2: tuple[torch.Tensor, torch.Tensor]


class RubinMegaMoeSession:
    """Kernel object + its carved workspace + bound launch arguments."""

    def __init__(self, cfg: RubinMoeConfig):
        ensure_not_capturing("Rubin MegaMoE session build")
        self.cfg = cfg
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.kernels = build_kernels(cfg)
        # Sizing, layout and token_comm come from the stock kernel when there is one
        # (identical in both variants: asserted in build_kernels).
        self.kernel = self.kernels.get(False, self.kernels.get(True))
        if not all(
            getattr(k, "masks_invalid_combine_slots", False)
            or getattr(k, "combine_dedup", False)
            for k in self.kernels.values()
        ):
            raise RuntimeError(
                "the vendored kernel does not mask invalid BF16 combine slots; this "
                "backend relies on it (no per-launch pre-clear)."
            )
        tokens = cfg.padded_tokens_per_rank
        token_comm = self.kernel.token_comm
        self.sf_cols = int(
            token_comm.plain_activation_sf_row_elements
            if cfg.kernel_variant == "genphase"
            else token_comm.activation_sf_hidden_padded
        )
        if self.sf_cols < cfg.hidden // cfg.sf_vec_size:
            raise RuntimeError(f"kernel activation SF row {self.sf_cols} < hidden/vec.")
        workspace = self.kernel._device_workspace
        self.local_bytes = int(workspace.total_bytes("local"))
        # One local buffer for both kernels of a narrow bucket (narrow_local_offsets).
        self._local_offsets = (0, 0)
        if True in self.kernels and False in self.kernels:
            self._local_offsets = narrow_local_offsets(
                self.kernels[False], self.kernels[True]
            )
            self.local_bytes = max(
                off + int(k._device_workspace.total_bytes("local"))
                for off, k in zip(self._local_offsets, (self.kernels[False], self.kernels[True]))
            )
        self.shared_bytes = int(workspace.total_bytes("shared"))
        # (name, shape, dtype) carved from the arena in this order on every rank.
        self.symmetric_plan: list[tuple[str, tuple[int, ...], torch.dtype]] = [
            ("x", (tokens, cfg.activation_row_bytes), cfg.torch_act_data_dtype),
            ("x_sf", (tokens, self.sf_cols), cfg.torch_sf_dtype),
            ("topk_weights", (tokens, cfg.num_topk), torch.float32),
            ("shared_workspace", (max(self.shared_bytes, 1),), torch.uint8),
        ]
        self.name = self.kernel.name()
        self.names = {s2: k.name() for s2, k in self.kernels.items()}
        self._carved = False
        self.compiled: dict[tuple[bool, bool], Any] = {}
        self._norm_bound: OrderedDict[tuple, dict] = OrderedDict()
        self._bound: OrderedDict[tuple, tuple[dict, RubinMoeWeights]] = OrderedDict()
        self._out_bound: OrderedDict[tuple, Any] = OrderedDict()
        self._shared_bound: OrderedDict[tuple, dict] = OrderedDict()

    # -- sizing / carving --------------------------------------------------------------

    def symmetric_bytes(self) -> int:
        total = 0
        for _, shape, dtype in self.symmetric_plan:
            n = dtype.itemsize
            for d in shape:
                n *= d
            total += aligned(n)
        return total

    def carve(self, arena: SymmetricArena, tag: str) -> None:
        cfg = self.cfg
        if self._carved:
            raise RuntimeError("session already carved.")
        if arena.world != cfg.world_size or arena.rank != cfg.rank:
            raise RuntimeError("arena group does not match the session's rank/world.")
        for name, shape, dtype in self.symmetric_plan:
            setattr(self, name, arena.alloc(shape, dtype, tag=f"{tag}.{name}"))
        tokens = cfg.padded_tokens_per_rank
        self.topk_idx = torch.full(
            (tokens, cfg.num_topk), -1, dtype=torch.int32, device=self.device
        )
        self._local_buffer = torch.zeros(
            (max(self.local_bytes, 1),), dtype=torch.uint8, device=self.device
        )
        # The main kernel's view; a narrow kernel launches on its own view of the
        # same buffer (same persistent state, narrow_local_offsets).
        self.local_workspace = self._local_buffer[self._local_offsets[0] :]
        self._narrow_local = self._local_buffer[self._local_offsets[1] :]
        self._write_spin_record_address()
        self._write_ep_balance_address()
        # Per-launch scratch of the refreshed inference kernel (src_gp; any content
        # at launch, nothing survives it). The caching allocator aligns to 512 B.
        self.tmp_buffer = (
            torch.empty(
                (
                    max(
                        [int(k.get_tmp_buffer_bytes()) for k in self.kernels.values()]
                        + [1]
                    ),
                ),
                dtype=torch.uint8,
                device=self.device,
            )
            if getattr(self.kernel, "separate_tmp_buffer", False)
            else None
        )
        # The fused shared expert's BF16 output rows (local; read after the launch).
        self.shared_output = (
            torch.empty(
                (tokens, cfg.shared[0]), dtype=torch.bfloat16, device=self.device
            )
            if cfg.shared is not None
            else None
        )
        # Output for launches with no live row (every rank must launch, R1).
        self.empty_output = torch.zeros(
            (1, cfg.hidden), dtype=torch.bfloat16, device=self.device
        )
        if (
            self.topk_idx.data_ptr() % 16
            or self.local_workspace.data_ptr() % 128
            or self._narrow_local.data_ptr() % 128
            or self.shared_workspace.data_ptr() % ALIGN
        ):
            raise RuntimeError("Rubin MegaMoE workspace alignment violated.")
        SymmetricBufferHost = _src(
            cfg
        ).sources.communication.nvlink_domain.symmetric_buffer.SymmetricBufferHost
        self.peer_mapper = SymmetricBufferHost(
            base_address=self.shared_workspace.data_ptr(),
            offsets=tuple(arena.offsets),
            rank=cfg.rank,
            max_ranks=cfg.world_size,
        )
        self._carved = True

    # -- bind / compile / launch ------------------------------------------------------

    def _base_kwargs(self, w: RubinMoeWeights) -> dict:
        """Everything but the output (FlashInfer ``_runtime_kwargs``)."""
        kwargs = {
            "activation": _to_cute(self.x),
            "activation_sf": _to_cute(self.x_sf),
            "topk_indices": _to_cute(self.topk_idx),
            "topk_scores": _to_cute(self.topk_weights, assumed_align=4),
            "fc1_weight": _to_cute(w.fc1[0]),
            "fc1_weight_sf": _to_cute(w.fc1[1]),
            "fc2_weight": _to_cute(w.fc2[0]),
            "fc2_weight_sf": _to_cute(w.fc2[1]),
            "local_workspace": _to_cute_ptr(self.local_workspace),
            "shared_workspace": _to_cute_ptr(self.shared_workspace),
            **(
                {"tmp_buffer": _to_cute_ptr(self.tmp_buffer)}
                if self.tmp_buffer is not None
                else {}
            ),
            "peer_rank_ptr_mapper_host": self.peer_mapper,
            "stream": _cu_stream(torch.cuda.current_stream()),
        }
        if self.cfg.quant_kind == "nvfp4":
            for name in _SCALAR_NAMES:
                kwargs[name] = _to_cute(getattr(w, name), assumed_align=4)
        if self.cfg.die_sm_mask32 is not None and self.cfg.kernel_variant != "genphase":
            kwargs["sm_locality_domain"] = _to_cute(self._sm_locality_domain())
        return kwargs

    def _sm_locality_domain(self) -> torch.Tensor:
        """uint8 ``[SM_LOCALITY_MAP_SIZE]`` die of each ``%smid`` (the die_split
        scheduler's map; FlashInfer ``die_local.sm_locality_domain_map`` layout),
        from the session's trusted die-1 SM mask."""
        t = getattr(self, "_sm_domains", None)
        if t is None:
            mask = self.cfg.die_sm_mask32
            domains = [0] * SM_LOCALITY_MAP_SIZE
            for sm in range(min(32 * len(mask), SM_LOCALITY_MAP_SIZE)):
                if (mask[sm // 32] >> (sm % 32)) & 1:
                    domains[sm] = 1
            t = torch.tensor(domains, dtype=torch.uint8, device=self.device)
            self._sm_domains = t
        return t

    def _check_output(self, output: torch.Tensor) -> None:
        if (
            output.ndim != 2
            or not (0 < output.shape[0] <= self.cfg.max_tokens_per_rank)
            or output.shape[1] != self.cfg.hidden
            or output.dtype != torch.bfloat16
            or output.device != self.device
            or not output.is_contiguous()
            or output.data_ptr() % 16
        ):
            raise ValueError(
                "output must be a contiguous, 16-byte aligned BF16 (n, hidden) matrix "
                f"with 0 < n <= {self.cfg.max_tokens_per_rank} on {self.device}."
            )

    def use_s2(self, ep_tokens: int | None) -> bool:
        """The kernel for a launch whose EP-uniform live tokens per rank are
        ``ep_tokens`` (None: the capacity): the narrow kernel (True) below the
        bucket's narrow rows, else the main one. Same answer on every EP rank."""
        if self.cfg.narrow is not None and True in self.kernels:
            return ep_tokens is not None and ep_tokens < self.cfg.narrow[0]
        return next(iter(self.kernels))

    def prepare(
        self,
        w: RubinMoeWeights,
        output: torch.Tensor,
        use_s2: bool | None = None,
        rmsnorm: tuple[torch.Tensor, float] | None = None,
        shared: "FusedSharedArgs | None" = None,
    ) -> dict:
        """Compile on first use (never during capture), bind (host-only; legal during
        capture for an already compiled session) and cache the launch arguments.
        ``use_s2`` picks the epilogue variant (None: :meth:`use_s2` at capacity).

        Weight bindings are cached per (layer weights, stream) for the session's
        lifetime (bounded LRU; they keep their weights alive, which the model does
        anyway). The output binding is cached per (pointer, shape, stride, dtype,
        device) in a small LRU: eager
        callers hand a fresh output each call, and CUDA graph replays reuse their pool
        addresses, so a large output cache would only pin freed activations.

        ``rmsnorm`` ``(weight [hidden] BF16, eps)``: the output is the stock vLLM
        ``rms_norm`` of the reduced row (vLLM-local E2a, fused into the reduce; its own
        compiled variant, compiled on first use like the plain one).
        """
        if not self._carved:
            raise RuntimeError("session used before its workspace was carved.")
        if self.device.index != torch.cuda.current_device():
            raise RuntimeError("a session must launch on the device that owns it.")
        wkey = (
            tuple(t.data_ptr() for t in (*w.fc1, *w.fc2)),
            tuple(
                getattr(w, n).data_ptr()
                for n in _SCALAR_NAMES
                if getattr(w, n) is not None
            ),
            torch.cuda.current_stream().cuda_stream,
        )
        hit = self._bound.get(wkey)
        if hit is not None:
            self._bound.move_to_end(wkey)
            base = hit[0]
        else:
            cfg = self.cfg
            validate_kernel_weights(
                w,
                quant_kind=cfg.quant_kind,
                experts=cfg.experts_per_rank,
                hidden=cfg.hidden,
                intermediate=cfg.intermediate,
                device=self.device,
            )
            base = self._base_kwargs(w)
            self._bound[wkey] = (base, w)
            if len(self._bound) > _BIND_CACHE_LIMIT:
                self._bound.popitem(last=False)
        okey = output_binding_key(output)
        out_cute = self._out_bound.get(okey)
        if out_cute is None:
            self._check_output(output)
            out_cute = _to_cute(output, dynamic_leading=False)
            out_cute.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
            self._out_bound[okey] = out_cute
            if len(self._out_bound) > _OUTPUT_CACHE_LIMIT:
                self._out_bound.popitem(last=False)
        else:
            self._out_bound.move_to_end(okey)
        kwargs = {**base, "output_activation": out_cute}
        if (shared is None) != (self.cfg.shared is None):
            raise ValueError("a fused-shared session launches with its shared arguments.")
        if shared is not None:
            kwargs.update(self._shared_kwargs(shared))
        if use_s2 is None:
            use_s2 = self.use_s2(None)
        if use_s2 and self.cfg.narrow is not None:
            kwargs["local_workspace"] = self._narrow_local_ptr()
        variant = (use_s2, rmsnorm is not None)
        if rmsnorm is not None and self.cfg.kernel_variant in DEDUP_VARIANTS:
            raise ValueError("the dedup reduce has no fused RMSNorm (E2a).")
        if rmsnorm is not None:
            kwargs.update(self._rmsnorm_kwargs(*rmsnorm))
        if variant not in self.compiled:
            kernel = self.kernels[use_s2]
            import cutlass

            ckey = compile_cache_key(
                self.cfg, self.names[use_s2], cutlass.__version__, self.device.index
            ) + (variant[1],)
            compiled = _COMPILED.get(ckey)
            if compiled is None:
                ensure_not_capturing("Rubin MegaMoE kernel compile")
                import cutlass.cute as cute

                compiled = cute.compile(kernel, **kwargs)
                _COMPILED[ckey] = compiled
            self.compiled[variant] = compiled
        return kwargs

    def _narrow_local_ptr(self):
        ptr = getattr(self, "_narrow_local_cute", None)
        if ptr is None:
            ptr = self._narrow_local_cute = _to_cute_ptr(self._narrow_local)
        return ptr

    def _shared_kwargs(self, shared: "FusedSharedArgs") -> dict:
        """The fused shared expert's launch arguments (cached by pointers): the MXFP8
        input as the caller's capacity buffer (rows >= the bucket's, F8_128x4 scales),
        the kernel-layout weights and this session's BF16 output rows."""
        key = (
            shared.x.data_ptr(), shared.x_sf.data_ptr(),
            *(t.data_ptr() for t in (*shared.fc1, *shared.fc2)),
        )  # fmt: skip
        hit = self._shared_bound.get(key)
        if hit is None:
            tokens, (hidden, inter) = self.cfg.padded_tokens_per_rank, self.cfg.shared
            sf_numel = -(-tokens // 128) * 128 * (-(-(hidden // 32) // 4) * 4)
            if (
                shared.x.dtype != torch.float8_e4m3fn
                or shared.x.shape[0] < tokens
                or shared.x.shape[1] != hidden
                or shared.x_sf.numel() < sf_numel
            ):
                raise ValueError(
                    f"shared input must be MXFP8 E4M3 (>= {tokens}, {hidden}) with "
                    f">= {sf_numel} F8_128x4 scales."
                )
            sf = shared.x_sf.view(torch.float8_e8m0fnu).reshape(-1)[:sf_numel]
            hit = {
                "shared_activation": _to_cute(shared.x[:tokens]),
                "shared_activation_sf": _to_cute(sf),
                "shared_fc1_weight": _to_cute(shared.fc1[0]),
                "shared_fc1_weight_sf": _to_cute(shared.fc1[1]),
                "shared_fc2_weight": _to_cute(shared.fc2[0]),
                "shared_fc2_weight_sf": _to_cute(shared.fc2[1]),
                "shared_output": _to_cute(self.shared_output, assumed_align=32),
            }
            self._shared_bound[key] = hit
            if len(self._shared_bound) > _BIND_CACHE_LIMIT:
                self._shared_bound.popitem(last=False)
        else:
            self._shared_bound.move_to_end(key)
        return hit

    def _rmsnorm_kwargs(self, weight: torch.Tensor, eps: float) -> dict:
        if (
            weight.shape != (self.cfg.hidden,)
            or weight.dtype != torch.bfloat16
            or weight.device != self.device
            or not weight.is_contiguous()
            or weight.data_ptr() % 16
        ):
            raise ValueError(
                f"rmsnorm weight must be a contiguous, 16-byte aligned BF16 ({self.cfg.hidden},) "
                f"vector on {self.device}."
            )
        key = (weight.data_ptr(), float(eps))
        hit = self._norm_bound.get(key)
        if hit is None:
            import cutlass

            hit = {
                "rmsnorm_weight": _to_cute(weight, dynamic_leading=False),
                "rmsnorm_eps": cutlass.Float32(float(eps)),
            }
            self._norm_bound[key] = hit
            if len(self._norm_bound) > _BIND_CACHE_LIMIT:
                self._norm_bound.popitem(last=False)
        else:
            self._norm_bound.move_to_end(key)
        return hit

    def launch(
        self,
        w: RubinMoeWeights,
        output: torch.Tensor,
        ep_tokens: int | None = None,
        rmsnorm: tuple[torch.Tensor, float] | None = None,
        shared: "FusedSharedArgs | None" = None,
    ) -> None:
        """One launch; ``ep_tokens`` (EP-uniform live tokens per rank, None = the
        capacity) picks the epilogue variant (:meth:`use_s2`); ``rmsnorm`` as in
        :meth:`prepare`."""
        use_s2 = self.use_s2(ep_tokens)
        kwargs = self.prepare(w, output, use_s2, rmsnorm, shared)
        self._stamp_issued()
        self.compiled[(use_s2, rmsnorm is not None)](**kwargs)

    # Host "issued" stamp (MEGAMOEFIX 2026-11-22, VENDOR.md S9): this rank's count of launches of this session,
    # written into its [issued, entered] progress word in the symmetric arena by a 4-byte H2D copy on a side
    # stream (copy engine: needs no SM and no sync, so it lands even when this GPU is stuck in an earlier
    # kernel). A peer's spin guard reads it over NVLink on timeout to tell a host that never issued the call (H1)
    # from a GPU that never started it (H2). Eager launches only: during CUDA graph capture the host does not
    # issue per launch, so graph replays leave it behind (the trap line reports raw ids).
    _PROGRESS_REGION = "nvlink.token_comm_dedup.call_progress"

    _RECORD_REGION = "nvlink.token_comm_dedup.spin_record_addr"

    def _write_spin_record_address(self) -> None:
        """Point this session's guards at the process's host spin-record buffer (spin_record.py): a trap's
        diagnostic then survives the context loss (printf alone is lost when the host never synchronizes)."""
        ws = self.kernel._device_workspace
        if self._RECORD_REGION not in {r.name for r in ws.regions("local")}:
            return
        from vllm.models.kimi_k3.nvidia import spin_record

        records = spin_record.get(self.cfg.rank)
        off = ws.offset(self._RECORD_REGION)
        self.local_workspace[off : off + 8].view(torch.int64).fill_(records.address)

    # dedup router (prefill: global counts) / genphase tail (decode: this rank's local counts)
    _EPBAL_REGIONS = ("nvlink.token_comm_dedup.epbal_ring_addr", "rubin.genphase_tail.epbal_ring_addr")

    def _write_ep_balance_address(self) -> None:
        """Point this session's router at the process's served EP-balance ring (ep_balance.py)."""
        ws = self.kernel._device_workspace
        names = {r.name for r in ws.regions("local")}
        regions = [r for r in self._EPBAL_REGIONS if r in names]
        if not regions:
            return
        from vllm.models.kimi_k3.nvidia.megamoe_rubin import ep_balance

        self._ep_balance_ring = ep_balance.new_ring(f"{self.cfg.kernel_variant}-{self.cfg.max_tokens_per_rank}", self.device)
        for region in regions:
            off = ws.offset(region)
            self.local_workspace[off : off + 8].view(torch.int64).fill_(self._ep_balance_ring.address)

    def _stamp_issued(self) -> None:
        if torch.cuda.is_current_stream_capturing():
            return
        view = getattr(self, "_issued_view", None)
        if view is None:
            ws = self.kernel._device_workspace
            names = {r.name for r in ws.regions("shared")}
            if self._PROGRESS_REGION not in names:
                self._issued_view = False  # this kernel variant has no progress word
                return
            off = ws.offset(self._PROGRESS_REGION)
            view = self.shared_workspace[off : off + 4].view(torch.int32)
            self._issued_view = view
            self._issued = 0
            self._issued_host = torch.zeros(1, dtype=torch.int32, pin_memory=True)
            self._issued_stream = torch.cuda.Stream(device=self.device)
        elif view is False:
            return
        self._issued += 1
        self._issued_host[0] = self._issued
        with torch.cuda.stream(self._issued_stream):
            view.copy_(self._issued_host, non_blocking=True)

    def release(self) -> None:
        self._issued_view = None  # the progress word lives in the arena being released
        self._bound.clear()
        self._out_bound.clear()
        self._norm_bound.clear()
        self._carved = False
        for name, _, _ in self.symmetric_plan:
            setattr(self, name, None)
        self._shared_bound.clear()
        self.topk_idx = self.local_workspace = self.empty_output = None
        self._local_buffer = self._narrow_local = self._narrow_local_cute = None
        self.tmp_buffer = None
        self.shared_output = None
