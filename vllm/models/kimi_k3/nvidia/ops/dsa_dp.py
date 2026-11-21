# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA data parallelism inside the TP group (``VLLM_K3_TP_DSA_DP=1``).

Without DCP, every request is owned by one card of its TP group. The DSA layers
run as under DEP: the owner computes the request's DSA rows with all heads (input
projections, indexer and top-k, sparse attention, output gate and o_proj), and it
is the only card that writes or reads the request's DSA and indexer KV. The
other cards never touch the request in a DSA layer. KDA layers, dense GEMMs, the
shared expert and MoE are unchanged (TP / EP).

Collectives: under sequence parallelism every card pushes its shard's rows of a
DSA layer straight to their owner (push_in) and each owner pushes its output rows
back (push_out), through symmetric memory; nothing else is added.

KV: by default each card has its own DSA KV region (``v1/core/dsa_dp_regions.py``)
and a request's DSA KV lives only in its owner's region, so the TP group keeps one
copy, as DEP does. Block hashes are not salted, so prefix-cache and Mooncake keys
do not depend on the owner; the owner is the card holding the longest prefix hit,
ties broken by queued load. ``VLLM_K3_TP_DSA_DP_KV=0`` keeps replicated DSA KV
instead, with owner-salted hashes and owners sticky per ``Request.session_id``.

Per step, ``prepare_step`` (model runner, after the regular attention metadata)
builds the DSA groups' metadata and slot mappings for the owned requests only and
records the owned token rows for the DSA layers' forward.
"""

from __future__ import annotations

import dataclasses
import os
import zlib
from typing import Any

import numpy as np
import torch

ENABLED = os.environ.get("VLLM_K3_TP_DSA_DP", "0") == "1"
# Phase 2: per-card DSA KV (v1/core/dsa_dp_regions.py). The DSA groups' blocks of
# a request live only in its owner card's region, so no card reserves another
# card's DSA KV.
KV_ENABLED = ENABLED and os.environ.get("VLLM_K3_TP_DSA_DP_KV", "1") == "1"
# DEP-like owner choice over the regions (scheduler ``_assign_dsa_owner_by_prefix``):
# no owner salt (block hashes and Mooncake keys are owner-independent), owner = the
# region with the longest prefix hit, ties by queued load. Ignores ``session_id``.
PREFIX_OWNER = KV_ENABLED and (
    os.environ.get("VLLM_K3_TP_DSA_DP_PREFIX_OWNER", "1") == "1"
)
# Output exchange of the DSA layers under SP: symmetric-memory push (push_out)
# instead of the zero-padded reduce-scatter.
PUSH = os.environ.get("VLLM_K3_DSADP_PUSH", "1") == "1"
# Input exchange of the DSA layers under SP (needs PUSH): every card pushes the
# rows of its sequence shard straight to their owner card (push_in) instead of
# the all-gather of the whole step, of which the owner keeps only its rows.
SCATTER_IN = ENABLED and PUSH

# Attention and indexer KV-cache layer names of the DSA layers (model init).
DSA_LAYER_NAMES: set[str] = set()
# req_id -> owner card (model runner, from NewRequestData.dsa_owner).
REQ_OWNER: dict[str, int] = {}
_LOGGED_PUSH: list[int] = []


# Debug profiling (VLLM_K3_DSADP_PROF=1): per-owner token / indexer-cost balance of
# each step (rank 0) and sampled per-segment GPU times of the decoder layers (every
# rank), logged every few hundred steps.
PROF = os.environ.get("VLLM_K3_DSADP_PROF", "0") == "1"
PROF_EVERY = int(os.environ.get("VLLM_K3_DSADP_PROF_EVERY", "10"))


class _Prof:
    def __init__(self) -> None:
        self.step = 0
        self.sample = False
        # Pending events of the sampled step: [(kind, [ev0, ev1, ...])].
        self.events: list[tuple[str, list[Any]]] = []
        self.seg: dict[tuple[str, int], float] = {}
        self.nsamples = 0
        self.bal: list[tuple[float, float, int]] = []
        self.host_ms = 0.0
        self.nhost = 0
        self._last: list[tuple[str, list[Any]]] = []
        self.fit: list[tuple[int, float, float]] = []
        self.own_tok = 0
        self.own_cost = 0.0
        self.host_split = [0.0, 0.0]

    def new_layer(self, kind: str) -> list[Any] | None:
        if not self.sample:
            return None
        evs: list[Any] = []
        self.events.append((kind, evs))
        return evs

    def flush(self, rank: int) -> None:
        if not self.events:
            return
        if self.sample:
            return
        # Never block the host: collect a sampled step once the GPU has passed it.
        if not self.events[-1][1][-1].query():
            return
        self._last = list(self.events)
        for kind, evs in self.events:
            for i in range(len(evs) - 1):
                k = (kind, i)
                self.seg[k] = self.seg.get(k, 0.0) + evs[i].elapsed_time(evs[i + 1])
        self.events.clear()
        dsa_ms = (
            sum(
                evs[2 - 1].elapsed_time(evs[2])
                for kind, evs in self._last
                if kind == "dsa"
            )
            if self._last
            else 0.0
        )
        self.fit.append((self.own_tok, self.own_cost, dsa_ms))
        self.nsamples += 1
        if self.nsamples % 30 == 0:
            from vllm.logger import init_logger

            per = {
                f"{k[0]}.{k[1]}": round(v / self.nsamples, 2)
                for k, v in sorted(self.seg.items())
            }
            init_logger(__name__).info(
                "DSADP_PROF rank %d: ms per step by (layer kind.segment) over %d "
                "sampled steps: %s",
                rank,
                self.nsamples,
                per,
            )
            f = np.array(self.fit, dtype=np.float64)
            f = f[f[:, 0] > 0]
            if len(f) >= 8:
                # dsa_ms ~ a * tokens + b * sum(n * ctx) + c
                ctxw = (f[:, 1] - f[:, 0]) * COST_C0
                A = np.stack([f[:, 0], ctxw, np.ones(len(f))], 1)
                coef = np.linalg.lstsq(A, f[:, 2], rcond=None)[0]
                init_logger(__name__).info(
                    "DSADP_FIT rank %d (%d samples): DSA ms = %.4g*tok + %.4g*tok*ctx "
                    "+ %.3g "
                    "-> fitted C0 %.0f; own tok mean %.0f dsa ms mean %.2f",
                    rank,
                    len(f),
                    coef[0],
                    coef[1],
                    coef[2],
                    coef[0] / coef[1] if coef[1] > 0 else -1,
                    f[:, 0].mean(),
                    f[:, 2].mean(),
                )


P = _Prof()


def prof_tick() -> None:
    """Per forward step: collect a finished sampled step, maybe sample this one."""
    P.sample = False
    P.flush(_tp()[1])
    P.step += 1
    # Sample only once the previous sample is collected (events of one step).
    P.sample = not P.events and P.step % PROF_EVERY == 0


def rec(evs: list[Any] | None) -> None:
    if evs is not None:
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        evs.append(e)


# DSA cost model of a chunk of n tokens after ctx cached tokens, in token units:
# a fixed per-token part (projections, top-2048 sparse attention, o_proj) plus the
# indexer, which scores every earlier token. COST_C0 is the context length at which
# the indexer costs as much as the fixed part: fitted on the VR200 TP4 serving step
# (6 DSA layers: 2.1e-3 ms per token + 1.18e-8 ms per token x context, C0 ~180K).
# Prompt chunk size under DSA-DP (replaces the long-prefill threshold).
CHUNK = int(os.environ.get("VLLM_K3_DSADP_CHUNK", "4096"))
COST_C0 = float(os.environ.get("VLLM_K3_DSADP_COST_C0", "180000"))


def chunk_cost(n: int, ctx: int) -> float:
    return n * (1.0 + (ctx + 0.5 * n) / COST_C0)


def max_tokens_for_cost(cost: float, ctx: int) -> int:
    """Largest n with chunk_cost(n, ctx) <= cost."""
    if cost <= 0:
        return 0
    a = 0.5 / COST_C0
    b = 1.0 + ctx / COST_C0
    return max(0, int((-b + (b * b + 4 * a * cost) ** 0.5) / (2 * a)))


_SCHED: list[tuple[float, float, float]] = []


def sched_stat(cap: float, used: list[float], busiest: float) -> None:
    """Scheduler side (PROF): per-owner cost before the DP pull-up vs the cap and
    the DP group's busiest card."""
    m = sum(used) / len(used)
    if m <= 0:
        return
    _SCHED.append(
        (max(used) / m, max(used) / max(cap, 1.0), busiest / max(max(used), 1.0))
    )
    if len(_SCHED) >= 200:
        from vllm.logger import init_logger

        a = np.array(_SCHED)
        init_logger(__name__).info(
            "DSADP_SCHED over %d steps: owner cost max/mean before pull-up %.3f "
            "(p90 %.3f); busiest owner / cap %.3f; DP busiest / own busiest %.3f",
            len(a),
            a[:, 0].mean(),
            np.quantile(a[:, 0], 0.9),
            a[:, 1].mean(),
            a[:, 2].mean(),
        )
        _SCHED.clear()


_ADMIT = [0, 0]


def count_admission(local_hit: bool) -> None:
    _ADMIT[local_hit] += 1
    if sum(_ADMIT) % 500 == 0:
        from vllm.logger import init_logger

        init_logger(__name__).info(
            "DSA-DP admissions: %d to the card holding their prefix, %d by load",
            _ADMIT[1],
            _ADMIT[0],
        )


def owner_of(key: str, tp: int) -> int:
    return zlib.crc32(key.encode()) % tp


@dataclasses.dataclass
class StepState:
    # Token count of the step the rows belong to.
    num_tokens: int = -1
    # This card's owned token rows (ascending) on the device, or None.
    own_idx: torch.Tensor | None = None
    n_own: int = 0
    # push_out's per-step source offsets / destination addresses.
    push_key: Any = None
    push_src: torch.Tensor | None = None
    push_dst: torch.Tensor | None = None
    # SCATTER_IN: owner card and position among that owner's rows of every token
    # of the step (device, [num_tokens]), the largest owner's row count, and
    # push_in's per-step source offsets / destination addresses.
    tok_owner: torch.Tensor | None = None
    tok_pos: torch.Tensor | None = None
    max_own: int = 0
    in_key: Any = None
    in_src: torch.Tensor | None = None
    in_dst: torch.Tensor | None = None


STEP = StepState()


def _tp() -> tuple[int, int]:
    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )

    return get_tensor_model_parallel_world_size(), get_tensor_model_parallel_rank()


def fallback_rows(num_tokens: int, device: torch.device) -> torch.Tensor | None:
    """Owned rows of a step prepare_step did not see (dummy / profile runs without
    metadata): a contiguous 1/tp slice, so the cards split the dummy work."""
    tp, rank = _tp()
    per = (num_tokens + tp - 1) // tp
    lo, hi = min(rank * per, num_tokens), min((rank + 1) * per, num_tokens)
    if hi <= lo:
        return None
    return torch.arange(lo, hi, device=device)


def rows_for(num_tokens: int, device: torch.device) -> torch.Tensor | None:
    # The layer sees padded rows (DP/SP padding) past the step's real tokens;
    # padding rows are never owned.
    if 0 <= STEP.num_tokens <= num_tokens:
        return STEP.own_idx
    return fallback_rows(num_tokens, device)


def _h2d(x: np.ndarray, device: torch.device) -> torch.Tensor:
    # Pinned staging: a pageable H2D copy blocks the host until the stream drains,
    # which would serialize this step's host prep behind the previous forward.
    return torch.from_numpy(x).pin_memory().to(device, non_blocking=True)


def _sub(x: Any, idx: np.ndarray) -> Any:
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x[idx]
    if isinstance(x, torch.Tensor):
        if x.device.type == "cpu":
            return x[torch.from_numpy(idx)]
        return x.index_select(0, _h2d(idx, x.device))
    raise TypeError(type(x))


def prepare_step(
    runner: Any,
    input_batch: Any,
    block_tables: tuple[torch.Tensor, ...],
    slot_mappings: Any,
    attn_groups: list[list[Any]],
    attn_metadata: dict[str, Any] | None,
    slot_mappings_by_layer: dict[str, torch.Tensor] | None,
) -> None:
    """Restrict this step's DSA metadata and slot mappings to the owned requests and
    record the owned rows. Called on every TP rank with identical inputs."""
    from vllm.v1.worker.gpu.attn_utils import (
        build_attn_metadata,
        build_slot_mappings_by_layer,
    )
    from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridAttnMetadata

    tp, rank = _tp()
    if PROF:
        import time

        prof_tick()
        _t0 = time.perf_counter()
    n = input_batch.num_reqs
    num_tokens = input_batch.num_tokens
    device = input_batch.positions.device
    STEP.num_tokens = num_tokens
    STEP.own_idx = None
    STEP.n_own = 0
    STEP.push_key = None
    STEP.tok_owner = STEP.tok_pos = None
    STEP.max_own = 0
    STEP.in_key = None
    if attn_metadata is None or n == 0:
        # Dummy step: no owned rows, the DSA layers skip their work (they issue no
        # collectives of their own, so ranks stay in lockstep) and output zeros.
        STEP.num_tokens = 0
        return
    owners = np.array(
        [REQ_OWNER.get(r, i % tp) for i, r in enumerate(input_batch.req_ids[:n])],
        dtype=np.int64,
    )
    reqs = np.nonzero(owners == rank)[0]
    qsl = input_batch.query_start_loc_np[: n + 1].astype(np.int64)
    if SCATTER_IN:
        # Every card needs the whole map: it pushes its shard's rows to their owners.
        tok_owner, tok_pos, counts = token_owner_pos(owners, qsl, tp)
        STEP.tok_owner = _h2d(tok_owner, device)
        STEP.tok_pos = _h2d(tok_pos, device)
        STEP.max_own = int(counts.max())
    if PROF and rank == 0:
        nt = (qsl[1:] - qsl[:-1]).astype(np.float64)
        ctx = input_batch.num_computed_tokens_np[:n].astype(np.float64)
        tok_o = np.bincount(owners, weights=nt, minlength=tp)
        cost_o = np.bincount(
            owners, weights=nt * (1.0 + (ctx + nt / 2) / COST_C0), minlength=tp
        )
        if tok_o.sum() >= 1024:
            P.bal.append(
                (
                    tok_o.max() / tok_o.mean(),
                    cost_o.max() / max(cost_o.mean(), 1.0),
                    int(tok_o.sum()),
                )
            )
        if len(P.bal) >= 200:
            from vllm.logger import init_logger

            b = np.array(P.bal)
            init_logger(__name__).info(
                "DSADP_BAL over %d steps: owned tokens max/mean mean %.3f p90 %.3f; "
                "owned "
                "modeled DSA cost max/mean mean %.3f p90 %.3f; tokens/step %.0f; "
                "prepare_step host %.2f ms (sub-batch %.2f, metadata build %.2f)",
                len(b),
                b[:, 0].mean(),
                np.quantile(b[:, 0], 0.9),
                b[:, 1].mean(),
                np.quantile(b[:, 1], 0.9),
                b[:, 2].mean(),
                P.host_ms / max(P.nhost, 1),
                P.host_split[0] / max(P.nhost, 1),
                P.host_split[1] / max(P.nhost, 1),
            )
            P.host_split = [0.0, 0.0]
            P.bal.clear()
            P.host_ms = 0.0
            P.nhost = 0
    lens = qsl[reqs + 1] - qsl[reqs]
    if PROF:
        P.own_tok, P.own_cost = 0, 0.0
    if reqs.size == 0 or lens.sum() == 0:
        # Nothing owned: the DSA layers skip their work on this card.
        return
    tok = np.concatenate([np.arange(qsl[r], qsl[r + 1]) for r in reqs])
    new_qsl = np.zeros(reqs.size + 1, dtype=np.int32)
    np.cumsum(lens, out=new_qsl[1:])
    own_tok = _h2d(tok, device)
    reqs_dev = _h2d(reqs, device)
    n_own_tok = int(tok.size)

    sub = dataclasses.replace(
        input_batch,
        req_ids=[input_batch.req_ids[i] for i in reqs],
        num_reqs=int(reqs.size),
        num_reqs_after_padding=int(reqs.size),
        idx_mapping=input_batch.idx_mapping.index_select(0, reqs_dev),
        idx_mapping_np=input_batch.idx_mapping_np[reqs],
        num_scheduled_tokens=input_batch.num_scheduled_tokens[reqs],
        num_tokens=n_own_tok,
        num_tokens_after_padding=n_own_tok,
        num_draft_tokens_per_req=_sub(input_batch.num_draft_tokens_per_req, reqs),
        query_start_loc=_h2d(new_qsl, device),
        query_start_loc_np=new_qsl,
        seq_lens=input_batch.seq_lens.index_select(0, reqs_dev),
        seq_lens_cpu_upper_bound=_sub(input_batch.seq_lens_cpu_upper_bound, reqs),
        dcp_local_seq_lens=_sub(input_batch.dcp_local_seq_lens, reqs),
        num_computed_tokens_np=input_batch.num_computed_tokens_np[reqs],
        prefill_len_np=input_batch.prefill_len_np[reqs],
        num_computed_prefill_tokens_np=input_batch.num_computed_prefill_tokens_np[reqs],
        is_prefilling_np=input_batch.is_prefilling_np[reqs],
        positions=input_batch.positions.index_select(0, own_tok),
        prompt_lens=_sub(input_batch.prompt_lens, reqs),
        max_seq_len_np=_sub(input_batch.max_seq_len_np, reqs),
        max_query_len=None,
        prefill_runs_as_decode_np=_sub(input_batch.prefill_runs_as_decode_np, reqs),
    )
    groups = [
        [g for g in gs if any(name in DSA_LAYER_NAMES for name in g.layer_names)]
        for gs in attn_groups
    ]
    sub_block_tables = tuple(
        bt.index_select(0, reqs_dev) if bt is not None else None for bt in block_tables
    )
    if isinstance(slot_mappings, torch.Tensor):
        sub_slots = slot_mappings.index_select(1, own_tok)
    else:
        sub_slots = type(slot_mappings)(
            s.index_select(0, own_tok) if s is not None else None for s in slot_mappings
        )
    kv_cfg = runner.kv_cache_config
    if kv_cfg.dsa_dp_region_blocks > 0:
        _check_dsa_groups(kv_cfg)
        sub_block_tables, sub_slots = localize_tables(
            sub_block_tables, sub_slots, kv_cfg, rank
        )
    is_prefilling = torch.from_numpy(sub.is_prefilling_np.copy())
    if sub.prefill_runs_as_decode_np is not None:
        is_prefilling &= ~torch.from_numpy(sub.prefill_runs_as_decode_np.copy())
    seq_ub = sub.seq_lens_cpu_upper_bound
    if PROF:
        _t1 = time.perf_counter()
    md = build_attn_metadata(
        attn_groups=groups,
        num_reqs=sub.num_reqs,
        num_tokens=n_own_tok,
        query_start_loc_gpu=sub.query_start_loc,
        query_start_loc_cpu=torch.from_numpy(new_qsl),
        max_query_len=int(lens.max()),
        seq_lens=sub.seq_lens,
        max_seq_len=int(seq_ub.max().item()),
        block_tables=sub_block_tables,
        slot_mappings=sub_slots,
        kv_cache_config=runner.kv_cache_config,
        seq_lens_cpu_upper_bound=seq_ub,
        dcp_local_seq_lens=sub.dcp_local_seq_lens,
        positions=sub.positions,
        model_specific_attn_metadata=MambaHybridAttnMetadata(
            is_prefilling=is_prefilling
        ),
        rswa_prefix_lens=sub.prompt_lens,
    )
    if PROF:
        _t2 = time.perf_counter()
        P.host_split[0] += (_t1 - _t0) * 1e3
        P.host_split[1] += (_t2 - _t1) * 1e3
    for name in DSA_LAYER_NAMES:
        if name in md:
            attn_metadata[name] = md[name]
    if slot_mappings_by_layer is not None:
        sm = build_slot_mappings_by_layer(sub_slots, runner.kv_cache_config, groups)
        for name in DSA_LAYER_NAMES:
            if name in sm:
                slot_mappings_by_layer[name] = sm[name]
    STEP.own_idx = own_tok
    STEP.n_own = n_own_tok
    if PROF:
        ctx_own = input_batch.num_computed_tokens_np[reqs].astype(np.float64)
        P.own_tok = n_own_tok
        P.own_cost = float(
            sum(chunk_cost(int(a), int(b)) for a, b in zip(lens, ctx_own))
        )
        P.host_ms += (time.perf_counter() - _t0) * 1e3
        P.nhost += 1


def token_owner_pos(
    owners: np.ndarray, qsl: np.ndarray, tp: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per token of the step: its owner card and its index among that owner's
    tokens in ascending order (the row order of ``own_idx`` on the owner); and
    each owner's token count."""
    lens = qsl[1:] - qsl[:-1]
    tok_owner = np.repeat(owners, lens)
    counts = np.bincount(tok_owner, minlength=tp)
    tok_pos = np.empty_like(tok_owner)
    for o in range(tp):
        m = tok_owner == o
        tok_pos[m] = np.arange(counts[o], dtype=tok_pos.dtype)
    return tok_owner, tok_pos, counts


def shard_rows(num_rows: int, real_rows: int, rank: int, tp: int) -> tuple[int, int]:
    """(shard size, real rows) of ``rank``'s sequence shard of a step of
    ``num_rows`` rows (SP-padded to a multiple of tp) of which the first
    ``real_rows`` are tokens."""
    shard = (num_rows + tp - 1) // tp
    return shard, max(0, min(shard, real_rows - rank * shard))


def push_in_plan(
    tok_owner: torch.Tensor,
    tok_pos: torch.Tensor,
    peer_ptrs: torch.Tensor,
    row0: int,
    real: int,
    row_bytes: int,
    hidden: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """push_in's (source element offsets, destination byte addresses) of the
    local rows ``[0, real)``, i.e. step rows ``[row0, row0 + real)``."""
    g = torch.arange(real, device=tok_owner.device, dtype=torch.int64) + row0
    src = torch.arange(real, device=tok_owner.device, dtype=torch.int64) * hidden
    return src, peer_ptrs[tok_owner[g]] + tok_pos[g] * row_bytes


# Input exchange (SP, SCATTER_IN): the card whose sequence shard holds row g
# pushes it straight to its owner, at the owner's position of g, over NVLink
# symmetric memory. Replaces the layer's all-gather of every row to every card:
# each row crosses NVLink at most once (a quarter of the all-gather's traffic at
# TP4). Returns the owner's rows (own_idx order), a view of the exchange buffer
# that stays valid until the next push_in, or None if it owns no row.
def push_in(local: torch.Tensor, num_rows: int) -> torch.Tensor | None:
    from vllm.distributed import get_tp_group
    from vllm.v1.attention.ops import shardq

    tp, rank = _tp()
    group = get_tp_group().device_group
    hidden = local.shape[1]
    shard, real = shard_rows(num_rows, STEP.num_tokens, rank, tp)
    assert local.shape[0] == shard, (local.shape, num_rows)
    # Same size on every card (step metadata), so a growth rendezvouses in
    # lockstep. Two shards' worth covers the usual owner skew (max/mean ~1.4) of
    # the largest step, so the buffer grows (a collective allocation) at most a
    # few times early on.
    buf = shardq._symm_buffer(
        group,
        "dsadp_in",
        max(STEP.max_own, 2 * shard) * hidden,
        local.dtype,
        local.device,
    )
    key = (shard, buf.storage.data_ptr(), real)
    if real and STEP.in_key != key:
        STEP.in_src, STEP.in_dst = push_in_plan(
            STEP.tok_owner,
            STEP.tok_pos,
            buf.peer_ptrs,
            rank * shard,
            real,
            hidden * local.element_size(),
            hidden,
        )
        STEP.in_key = key
    if real:
        src = local if local.is_contiguous() else local.contiguous()
        shardq._push_rows(src, STEP.in_src, STEP.in_dst, hidden, buf.handle)
    else:
        buf.handle.barrier()
        buf.handle.barrier()
    if not STEP.n_own:
        return None
    return buf.storage[: STEP.n_own * hidden].view(STEP.n_own, hidden)


def local_rows_out(
    attn: Any, local: torch.Tensor, positions: torch.Tensor, num_rows: int
) -> torch.Tensor:
    """SCATTER_IN on a step prepare_step did not see (profile / dummy runs
    without metadata): ``fallback_rows`` are exactly this card's sequence shard,
    so the layer runs on its local rows with no exchange; padding rows are 0."""
    tp, rank = _tp()
    shard, real = shard_rows(num_rows, num_rows, rank, tp)
    out = local.new_zeros((shard, local.shape[1]))
    if real:
        lo = rank * shard
        out[:real] = attn(
            hidden_states=local[:real], positions=positions[lo : lo + real]
        )
    return out


# Output exchange (SP): the owner of row g pushes its DSA output straight to the
# rank whose sequence shard holds g (rank g // S, row g % S), over NVLink symmetric
# memory, instead of a zero-padded [T, hidden] reduce-scatter. Each output row
# crosses NVLink at most once (a quarter of the reduce-scatter's traffic at TP4)
# and nothing is reduced. Rows nobody owns (DP / SP padding) are zeroed.
def push_out(
    own_out: torch.Tensor | None,
    num_rows: int,
    hidden: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    from vllm.distributed import get_tp_group
    from vllm.v1.attention.ops import shardq

    tp, rank = _tp()
    group = get_tp_group().device_group
    shard = (num_rows + tp - 1) // tp
    if not _LOGGED_PUSH:
        _LOGGED_PUSH.append(1)
        from vllm.logger import init_logger

        init_logger(__name__).info("DSA-DP: DSA outputs pushed over symmetric memory")
    buf = shardq._symm_buffer(group, "dsadp_out", shard * hidden, dtype, device)
    n_own = 0 if own_out is None else own_out.shape[0]
    key = (shard, buf.storage.data_ptr(), n_own)
    if STEP.push_key != key:
        idx = STEP.own_idx
        if n_own:
            STEP.push_src = (
                torch.arange(n_own, device=device, dtype=torch.int64) * hidden
            )
            STEP.push_dst = buf.peer_ptrs[idx // shard] + (idx % shard) * (
                hidden * own_out.element_size()
            )
        STEP.push_key = key
    if n_own:
        assert own_out.is_contiguous() and own_out.shape[1] == hidden
        shardq._push_rows(own_out, STEP.push_src, STEP.push_dst, hidden, buf.handle)
    else:
        buf.handle.barrier()
        buf.handle.barrier()
    out = buf.storage[: shard * hidden].view(shard, hidden)
    # Rows of this shard past the step's real tokens are owned by nobody.
    real = max(0, min(shard, STEP.num_tokens - rank * shard))
    res = torch.empty_like(out)
    res[:real].copy_(out[:real])
    if real < shard:
        res[real:].zero_()
    return res


# Phase 2 (KV_ENABLED): a card backs its own DSA region [S + rank * D, S +
# (rank + 1) * D) at local pages [S, S + D). Shared ids [0, S) are unchanged.


def to_local_block_id(block_id: int, shared: int, region: int, rank: int) -> int:
    """Local page of a block id on card ``rank``, or -1 for another card's."""
    if block_id < shared:
        return block_id
    local = block_id - rank * region
    return local if shared <= local < shared + region else -1


def local_block_ids(block_ids: list[int], kv_cfg: Any, rank: int) -> list[int]:
    """This card's pages among scheduler block ids (other cards' dropped)."""
    if kv_cfg.dsa_dp_region_blocks <= 0:
        return block_ids
    s, d = kv_cfg.dsa_dp_shared_blocks, kv_cfg.dsa_dp_region_blocks
    out = (to_local_block_id(b, s, d, rank) for b in block_ids)
    return [b for b in out if b >= 0]


def region_block_ids(
    block_ids: list[int], kv_cfg: Any, rank: int, to_global: bool = False
) -> list[int]:
    """Position-preserving map of one DSA group's block ids of a request this
    card owns: scheduler ids to local pages, or back with ``to_global``."""
    s, d = kv_cfg.dsa_dp_shared_blocks, kv_cfg.dsa_dp_region_blocks
    if to_global:
        return [b + rank * d if b >= s else b for b in block_ids]
    out = [to_local_block_id(b, s, d, rank) for b in block_ids]
    assert min(out, default=0) >= 0, "DSA block of another card's region"
    return out


def local_block_copies(copies: list[Any], kv_cfg: Any, rank: int) -> list[Any]:
    """Block copies on this card's pages; copies within another card's region
    are dropped."""
    if kv_cfg.dsa_dp_region_blocks <= 0:
        return copies
    s, d = kv_cfg.dsa_dp_shared_blocks, kv_cfg.dsa_dp_region_blocks
    out = []
    for c in copies:
        src = to_local_block_id(c.src_block_id, s, d, rank)
        dst = to_local_block_id(c.dst_block_id, s, d, rank)
        if src >= 0 and dst >= 0:
            out.append(c._replace(src_block_id=src, dst_block_id=dst))
    return out


def localize_tables(
    block_tables: tuple[torch.Tensor | None, ...],
    slot_mappings: Any,
    kv_cfg: Any,
    rank: int,
) -> tuple[tuple[torch.Tensor | None, ...], Any]:
    """Map the DSA groups' block tables and slot mappings (owned requests only)
    from region ids to this card's local pages."""
    s, d = kv_cfg.dsa_dp_shared_blocks, kv_cfg.dsa_dp_region_blocks
    tables = list(block_tables)
    slots = (
        slot_mappings
        if isinstance(slot_mappings, torch.Tensor)
        else list(slot_mappings)
    )
    for g in kv_cfg.dsa_dp_group_ids:
        bt = tables[g]
        if bt is not None:
            tables[g] = torch.where(bt >= s, bt - rank * d, bt)
        sm = slots[g]
        if sm is not None:
            bs = kv_cfg.kv_cache_groups[g].kv_cache_spec.block_size
            slots[g] = torch.where(sm >= s * bs, sm - rank * d * bs, sm)
    if not isinstance(slot_mappings, torch.Tensor):
        slots = type(slot_mappings)(slots)
    return tuple(tables), slots


_CHECKED: list[int] = []


def _check_dsa_groups(kv_cfg: Any) -> None:
    """The region groups must hold DSA layers only: other layers run on every
    card and could not read a request's KV from its owner's region."""
    if _CHECKED:
        return
    allowed = DSA_LAYER_NAMES
    for g in kv_cfg.dsa_dp_group_ids:
        names = set(kv_cfg.kv_cache_groups[g].layer_names)
        if not names <= allowed:
            raise RuntimeError(
                f"DSA-DP KV: cache group {g} has non-DSA layers "
                f"{sorted(names - allowed)[:4]}"
            )
    _CHECKED.append(1)
