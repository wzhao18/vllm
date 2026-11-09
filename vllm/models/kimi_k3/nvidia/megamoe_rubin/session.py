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
BF16 ``TopkReduce`` writing ``out[:n]`` directly. No host sync, allocation or masked
fill (BF16 separate reduction masks invalid slots in-kernel: asserted at build), so it
is CUDA graph safe after warmup. Every EP rank must launch the same session the same
number of times in the same order (the barrier state persists across launches).
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch

from .config import RubinMoeConfig
from .symm_arena import ALIGN, SymmetricArena, aligned, ensure_not_capturing
from .weights import RubinMoeWeights, validate_kernel_weights

_SCALAR_NAMES = ("fc1_alpha", "fc2_alpha", "fc1_norm_const")
_MAX_ACTIVE_CLUSTERS: dict[tuple[int, int], int] = {}
_COMPILED: dict[tuple, Any] = {}
_BIND_CACHE_LIMIT = 1024
_OUTPUT_CACHE_LIMIT = 4


def compile_cache_key(
    cfg: RubinMoeConfig, kernel_name: str, cutlass_version: str, device: int
) -> tuple:
    """Process-wide compiled-kernel key. The kernel name encodes the constexprs of
    both variants, except that the genphase name omits the SiTU betas (codegen
    constants), so they, the quant kind and the device are keyed explicitly."""
    return (
        kernel_name,
        str(cutlass_version),
        cfg.quant_kind,
        float(cfg.situ_beta),
        float(cfg.situ_linear_beta),
        int(device),
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


def build_kernel(cfg: RubinMoeConfig):
    """FlashInfer ``Sm107BlockScaledSymmBuffer._build_kernel`` (BF16 combine, SiTU)."""
    import cutlass
    from cutlass.cute.nvgpu import OperandMajorMode

    from .src.sources import RubinInferenceGenphaseMegaMoE, RubinInferenceMegaMoE
    from .src.sources.api import ImplDesc, ProblemDesc
    from .src.sources.quant_def import CombineFormat

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
    if cfg.fallback_cluster_shape_mn is not None:
        impl_fields["fallback_cluster_shape_mn"] = tuple(cfg.fallback_cluster_shape_mn)
        impl_fields["preferred_cluster_count"] = preferred_count
        impl_fields["fallback_cluster_count"] = fallback_count
    kernel_cls = (
        RubinInferenceGenphaseMegaMoE
        if cfg.kernel_variant == "genphase"
        else RubinInferenceMegaMoE
    )
    return kernel_cls(problem_desc, ImplDesc(impl_fields))


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


class RubinMegaMoeSession:
    """Kernel object + its carved workspace + bound launch arguments."""

    def __init__(self, cfg: RubinMoeConfig):
        ensure_not_capturing("Rubin MegaMoE session build")
        self.cfg = cfg
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.kernel = build_kernel(cfg)
        if not getattr(self.kernel, "masks_invalid_combine_slots", False):
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
        self.shared_bytes = int(workspace.total_bytes("shared"))
        # (name, shape, dtype) carved from the arena in this order on every rank.
        self.symmetric_plan: list[tuple[str, tuple[int, ...], torch.dtype]] = [
            ("x", (tokens, cfg.activation_row_bytes), cfg.torch_act_data_dtype),
            ("x_sf", (tokens, self.sf_cols), cfg.torch_sf_dtype),
            ("topk_weights", (tokens, cfg.num_topk), torch.float32),
            ("shared_workspace", (max(self.shared_bytes, 1),), torch.uint8),
        ]
        self.name = self.kernel.name()
        self._carved = False
        self.compiled = None
        self._bound: OrderedDict[tuple, tuple[dict, RubinMoeWeights]] = OrderedDict()
        self._out_bound: OrderedDict[tuple, Any] = OrderedDict()

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
        self.local_workspace = torch.zeros(
            (max(self.local_bytes, 1),), dtype=torch.uint8, device=self.device
        )
        # Output for launches with no live row (every rank must launch, R1).
        self.empty_output = torch.zeros(
            (1, cfg.hidden), dtype=torch.bfloat16, device=self.device
        )
        if (
            self.topk_idx.data_ptr() % 16
            or self.local_workspace.data_ptr() % 128
            or self.shared_workspace.data_ptr() % ALIGN
        ):
            raise RuntimeError("Rubin MegaMoE workspace alignment violated.")
        from .src.sources.communication.nvlink_domain.symmetric_buffer import (
            SymmetricBufferHost,
        )

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
            "peer_rank_ptr_mapper_host": self.peer_mapper,
            "stream": _cu_stream(torch.cuda.current_stream()),
        }
        if self.cfg.quant_kind == "nvfp4":
            for name in _SCALAR_NAMES:
                kwargs[name] = _to_cute(getattr(w, name), assumed_align=4)
        return kwargs

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

    def prepare(self, w: RubinMoeWeights, output: torch.Tensor) -> dict:
        """Compile on first use (never during capture), bind (host-only; legal during
        capture for an already compiled session) and cache the launch arguments.

        Weight bindings are cached per (layer weights, stream) for the session's
        lifetime (bounded LRU; they keep their weights alive, which the model does
        anyway). The output binding is cached per (pointer, shape, stride, dtype,
        device) in a small LRU: eager
        callers hand a fresh output each call, and CUDA graph replays reuse their pool
        addresses, so a large output cache would only pin freed activations.
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
        if self.compiled is None:
            import cutlass

            ckey = compile_cache_key(
                self.cfg, self.name, cutlass.__version__, self.device.index
            )
            compiled = _COMPILED.get(ckey)
            if compiled is None:
                ensure_not_capturing("Rubin MegaMoE kernel compile")
                import cutlass.cute as cute

                compiled = cute.compile(self.kernel, **kwargs)
                _COMPILED[ckey] = compiled
            self.compiled = compiled
        return kwargs

    def launch(self, w: RubinMoeWeights, output: torch.Tensor) -> None:
        kwargs = self.prepare(w, output)
        self.compiled(**kwargs)

    def release(self) -> None:
        self._bound.clear()
        self._out_bound.clear()
        self._carved = False
        for name, _, _ in self.symmetric_plan:
            setattr(self, name, None)
        self.topk_idx = self.local_workspace = self.empty_output = None
