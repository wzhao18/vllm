# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``RubinMegaMoE``: the Kimi K3.1 routed MoE on VR200 (sm_107a) through the vendored
FlashInfer Rubin CuTe DSL MegaMoE kernels (fused dispatch + FC1 + SiTU + FC2 + BF16
combine), on torch symmetric memory. No flashinfer / nvshmem dependency.

One object per EP process group serves every MoE layer: it owns one session per
token bucket (compiled kernel + workspace, all carved from ONE symmetric arena), and
takes per-layer weights at call time.

Lock-step contract (deadlock otherwise): every EP rank must call :meth:`forward` with
the SAME bucket, in the same order, the same number of times (barrier state persists
in the workspace). Choose the bucket from an EP-uniform token count (the
DP-synchronized max / CUDA graph padded size), never the local count; ranks with no
tokens still call.

Expert ids are GLOBAL PHYSICAL ids over ``problem.num_experts`` (the physical expert
count, after any EPLB logical-to-physical mapping) with linear placement: id ``g``
lives on rank ``g // experts_per_rank``. -1 masks a route; any other out-of-range id
is an error (:class:`RouteIdError`), never a silent drop. The stager records it in a
sticky device flag; it is raised by :meth:`check_route_errors` (syncs: warmup, debug)
and, without a sync, by the next eager :meth:`forward` once the asynchronous host
mirror of the flag (copied after every eager forward) has landed. Graph replays
update the device flag only: call :meth:`check_route_errors` (or
:meth:`poll_route_errors` once per step) to surface errors raised inside a graph.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.distributed as dist

from vllm.logger import init_logger

from .arch import require_sm107
from .config import RouteIdError, RubinBucket, RubinMoeConfig, RubinMoeProblem
from .session import RubinMegaMoeSession, layout_tag
from .staging import PREFILL_STAGE_MIN_TOKENS, stage_inputs
from .symm_arena import SymmetricArena, ensure_not_capturing, group_barrier
from .weights import RubinMoeWeights, prepare_weights

logger = init_logger(__name__)


def _rank_world(group: dist.ProcessGroup | None) -> tuple[int, int]:
    if group is None:
        if dist.is_initialized() and dist.get_world_size() > 1:
            raise ValueError("pass the EP process group explicitly.")
        return 0, 1
    return dist.get_rank(group), dist.get_world_size(group)


def _sorted_buckets(buckets: Sequence[RubinBucket]) -> list[RubinBucket]:
    if not buckets:
        raise ValueError("at least one bucket is required.")
    ordered = sorted(buckets, key=lambda b: (b.max_tokens, b.variant))
    caps = [b.max_tokens for b in ordered]
    if len(set(caps)) != len(caps):
        raise ValueError(f"bucket capacities must be distinct, got {caps}.")
    return ordered


class RubinMegaMoE:
    def __init__(
        self,
        problem: RubinMoeProblem,
        group: dist.ProcessGroup | None,
        buckets: Sequence[RubinBucket],
    ):
        require_sm107()
        ensure_not_capturing("RubinMegaMoE construction")
        self.problem = problem
        self.group = group
        self.rank, self.world = _rank_world(group)
        if problem.num_experts % self.world:
            raise ValueError(
                f"{problem.num_experts} experts do not divide over EP {self.world}."
            )
        self.experts_per_rank = problem.num_experts // self.world
        self.buckets = _sorted_buckets(buckets)
        self.sessions: dict[RubinBucket, RubinMegaMoeSession] = {
            b: RubinMegaMoeSession(
                RubinMoeConfig.make(problem, b, rank=self.rank, world_size=self.world)
            )
            for b in self.buckets
        }
        symmetric = sum(s.symmetric_bytes() for s in self.sessions.values())
        self.arena = SymmetricArena(group, symmetric)
        for i, b in enumerate(self.buckets):
            s = self.sessions[b]
            s.carve(self.arena, layout_tag(i, s.cfg, s.name))
        self.arena.verify_layout()
        device = torch.device("cuda", torch.cuda.current_device())
        # Sticky route-id error flag [flag, offending id] + its async host mirror.
        self.route_error = torch.zeros(2, dtype=torch.int64, device=device)
        self._route_error_host = torch.zeros(2, dtype=torch.int64, pin_memory=True)
        self._route_error_event: torch.cuda.Event | None = None
        local = sum(s.local_bytes for s in self.sessions.values())
        # Sessions' local tensors (topk_idx fill, local workspace zeros) and the flag
        # are initialized on the construction stream; finish them before any other
        # stream (warmup / forward) can launch.
        torch.cuda.synchronize()
        if self.rank == 0:
            logger.info(
                "Rubin MegaMoE (%s, EP%d): buckets %s; symmetric arena %.1f MiB, "
                "local workspaces %.1f MiB per rank.",
                problem.quant_kind,
                self.world,
                [(b.variant, b.max_tokens) for b in self.buckets],
                self.arena.nbytes / 2**20,
                local / 2**20,
            )

    # -- sizing ---------------------------------------------------------------------

    @staticmethod
    def arena_bytes(
        problem: RubinMoeProblem, world_size: int, buckets: Sequence[RubinBucket]
    ) -> tuple[int, int]:
        """``(symmetric, local)`` bytes per rank, for memory budgeting before KV-cache
        sizing. Builds the kernel objects (host only + the occupancy probe)."""
        require_sm107()
        sym = loc = 0
        for b in _sorted_buckets(buckets):
            s = RubinMegaMoeSession(
                RubinMoeConfig.make(problem, b, rank=0, world_size=world_size)
            )
            sym += s.symmetric_bytes()
            loc += s.local_bytes
        return sym, loc

    # -- weights --------------------------------------------------------------------

    def prepare_weights(
        self,
        w13: torch.Tensor,
        w13_scale: torch.Tensor,
        w2: torch.Tensor,
        w2_scale: torch.Tensor,
        *,
        layer_id: int | None = None,
    ) -> RubinMoeWeights:
        """Canonical MXFP4 local-expert weights -> kernel layout for the problem's
        quant kind (``nvfp4``: exact conversion or :class:`NonRepresentableError`)."""
        p = self.problem
        if w13.shape[0] != self.experts_per_rank:
            raise ValueError(
                f"expected {self.experts_per_rank} local experts (linear EP "
                f"placement), got {w13.shape[0]}."
            )
        return prepare_weights(
            p.quant_kind,
            w13,
            w13_scale,
            w2,
            w2_scale,
            hidden=p.hidden,
            intermediate=p.intermediate,
            input_norm_const=p.input_norm_const,
            fc1_norm_const=p.fc1_norm_const,
            layer_id=layer_id,
            expert_offset=self.rank * self.experts_per_rank,
        )

    # -- route-id errors ------------------------------------------------------------

    def _raise_route_error(self, flag: torch.Tensor) -> None:
        if int(flag[0]):
            raise RouteIdError(int(flag[1]), self.problem.num_experts)

    def check_route_errors(self) -> None:
        """Synchronizing check of the sticky route-id flag (warmup, debug, after graph
        replays); raises :class:`RouteIdError`."""
        ensure_not_capturing("RubinMegaMoE.check_route_errors")
        self._raise_route_error(self.route_error.cpu())

    def poll_route_errors(self) -> None:
        """Non-blocking check: raise if the last landed host mirror of the flag shows
        an error, then queue a fresh mirror copy on the current stream. No-op during
        capture (graph replays are surfaced by the next poll / check). The mirror copy
        is 16 bytes D2H, one per eager forward."""
        if torch.cuda.is_current_stream_capturing():
            return
        event = self._route_error_event
        if event is not None and event.query():
            self._raise_route_error(self._route_error_host)
        self._route_error_host.copy_(self.route_error, non_blocking=True)
        if event is None:
            event = self._route_error_event = torch.cuda.Event()
        event.record()

    def reset_route_errors(self) -> None:
        ensure_not_capturing("RubinMegaMoE.reset_route_errors")
        torch.cuda.synchronize()
        self.route_error.zero_()
        self._route_error_host.zero_()
        torch.cuda.synchronize()

    # -- execution ------------------------------------------------------------------

    def select_bucket(self, num_tokens_uniform: int) -> RubinBucket:
        """Smallest bucket holding ``num_tokens_uniform`` (an EP-uniform count)."""
        for b in self.buckets:
            if num_tokens_uniform <= b.max_tokens:
                return b
        raise ValueError(
            f"{num_tokens_uniform} tokens exceed the largest bucket "
            f"({self.buckets[-1].max_tokens})."
        )

    def _stage(self, session, hidden, topk_ids, topk_weights, is_padding=None) -> None:
        stage_inputs(
            hidden,
            topk_ids,
            topk_weights,
            session.x,
            session.x_sf,
            session.topk_idx,
            session.topk_weights,
            route_error=self.route_error,
            quant_kind=self.problem.quant_kind,
            num_experts=self.problem.num_experts,
            input_norm_const=self.problem.input_norm_const,
            is_padding=is_padding,
        )

    def warmup(
        self,
        weights: RubinMoeWeights,
        *,
        ids_dtypes: Sequence[torch.dtype] = (torch.int32, torch.int64),
        weight_dtypes: Sequence[torch.dtype] = (torch.float32, torch.bfloat16),
    ) -> None:
        """Collective; run on all EP ranks before CUDA graph capture.

        1. Compiles the stager for every (ids dtype, weights dtype) the caller will
           pass (token count and strides are not specialized), on the real shape: one
           live row whose routes are all -1, staged locally (no kernel launch).
        2. Self-tests the route-id error path: one row routed to id ``num_experts``
           must set the device flag (raises otherwise), which is then cleared.
        3. Compiles + binds every bucket and launches it once with no routed token
           (all ids -1), then checks the route-id flag.
        """
        ensure_not_capturing("RubinMegaMoE warmup")
        p = self.problem
        dev = self.route_error.device
        row = torch.zeros((1, p.hidden), dtype=torch.bfloat16, device=dev)
        first = self.sessions[self.buckets[0]]
        last = self.sessions[self.buckets[-1]]
        # The multi-row prefill grid is its own specialization: compile it too when
        # the largest bucket can take that many rows.
        wide = PREFILL_STAGE_MIN_TOKENS
        wide_rows = None
        if self.buckets[-1].max_tokens >= wide:
            wide_rows = torch.zeros((wide, p.hidden), dtype=torch.bfloat16, device=dev)
        for ids_dtype in ids_dtypes:
            for w_dtype in weight_dtypes:
                ids = torch.full((1, p.top_k), -1, dtype=ids_dtype, device=dev)
                wts = torch.zeros((1, p.top_k), dtype=w_dtype, device=dev)
                self._stage(first, row, ids, wts)
                if wide_rows is not None:
                    ids = torch.full((wide, p.top_k), -1, dtype=ids_dtype, device=dev)
                    wts = torch.zeros((wide, p.top_k), dtype=w_dtype, device=dev)
                    self._stage(last, wide_rows, ids, wts)
        self.check_route_errors()
        ids = torch.full((1, p.top_k), -1, dtype=ids_dtypes[0], device=dev)
        ids[0, 0] = p.num_experts
        wts = torch.zeros((1, p.top_k), dtype=weight_dtypes[0], device=dev)
        self._stage(first, row, ids, wts)
        flag = self.route_error.cpu()
        if int(flag[0]) != 1 or int(flag[1]) != p.num_experts:
            raise RuntimeError(
                f"route-id error self-test failed (flag {flag.tolist()}): the stager "
                "did not report an out-of-range expert id."
            )
        self.reset_route_errors()
        empty = torch.empty((0, p.hidden), dtype=torch.bfloat16, device=dev)
        ids = torch.empty((0, p.top_k), dtype=torch.int32, device=dev)
        wts = torch.empty((0, p.top_k), dtype=torch.float32, device=dev)
        for b in self.buckets:
            self.forward(empty, ids, wts, weights, bucket=b)
        group_barrier(self.group)
        self.check_route_errors()

    def forward(
        self,
        hidden: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        weights: RubinMoeWeights,
        *,
        bucket: RubinBucket,
        out: torch.Tensor | None = None,
        is_padding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``out[:n] = sum_k topk_weights[:, k] * expert_{topk_ids[:, k]}(hidden)``.

        ``hidden`` (n, H) BF16; ``topk_ids`` (n, top_k) int32/int64 GLOBAL PHYSICAL
        expert ids (-1 = masked; any other id outside ``[0, num_experts)`` raises
        :class:`RouteIdError`, see the module docstring); ``topk_weights`` (n, top_k)
        fp32/bf16/fp16 (``routed_scaling_factor`` folded in by the caller);
        ``is_padding`` optional bool (n,): rows whose routes are masked to -1 in the
        stager (as ``torch.where(is_padding[:, None], -1, topk_ids)``). Output
        BF16. Both the staging and the kernel run on the current stream.
        """
        session = self.sessions.get(bucket)
        if session is None:
            raise ValueError(f"unknown bucket {bucket}.")
        n = hidden.shape[0]
        if hidden.dim() != 2 or hidden.shape[1] != self.problem.hidden:
            raise ValueError(f"hidden must be (n, {self.problem.hidden}).")
        if n > bucket.max_tokens:
            raise ValueError(f"{n} tokens exceed bucket capacity {bucket.max_tokens}.")
        self._stage(session, hidden, topk_ids, topk_weights, is_padding)
        if n == 0:
            # Every rank launches (lock step); the reduce writes the scratch row only.
            session.launch(weights, session.empty_output)
            result = hidden.new_empty((0, self.problem.hidden))
        else:
            if out is None:
                out = torch.empty(
                    (n, self.problem.hidden),
                    dtype=torch.bfloat16,
                    device=hidden.device,
                )
            elif out.shape[0] < n:
                raise ValueError("out has fewer rows than tokens.")
            result = out[:n]
            session.launch(weights, result)
        self.poll_route_errors()
        return result

    def destroy(self) -> None:
        ensure_not_capturing("RubinMegaMoE teardown")
        group_barrier(self.group)
        for s in self.sessions.values():
            s.release()
        self.sessions.clear()
        self.arena.destroy()
