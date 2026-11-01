# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``VLLM_DSA_BPRIME_CHECK=1``: the one-round DCP decode top-k merge against
the generic merge, on the live decode logits.

Each decode step that takes ``dcp_topk_merge`` also runs the generic merge
(the backend's local top-K, then ``_merge_dcp_topk_global``) of the same
logits, and ``check_dcp_topk_merge`` compares the two per row. Both pick the
top-K by bf16 score, and equal scores at the lowest selected one (``tau``) may
resolve to different tokens. So a row passes when both hold as many distinct
ids, every id of the merge scores at least ``tau`` and every reference id
scoring above ``tau`` is in the merge: the two score multisets are equal.
Each rank looks up the ids it owns (``id % W == rank``, interleave 1) in its
own logits; the verdict of a row is combined over the DCP group. Rows with a
NaN logit are counted apart (outside the merge's exactness contract).

All of it runs on the device without host syncs, so CUDA graphs keep it.
Per-layer counters (rows, rows that differ only at ties, missed rows,
saturated rows and ranks from the select's ``sat_word``) and the first
misses as (layer, step, row) land in pinned host memory the device writes
directly; a daemon thread of DCP rank 0 logs them (and a final summary at
exit). ``step`` counts the forward passes the check saw. Debug only: it adds
the generic merge and two small all-gathers to every decode step.
"""

import atexit
import threading
import time
from collections.abc import Callable
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

MAX_LAYERS = 128
MAX_MISSES = 64
# Per-layer counters; MAX_SAT_RANKS and MAX_MISSING keep the maximum.
(
    CALLS,
    ROWS,
    TIE_ROWS,
    MISS_ROWS,
    NAN_ROWS,
    SAT_ROWS,
    MAX_SAT_RANKS,
    MAX_MISSING,
) = range(8)
_NUM_STATS = 8
# A recorded miss: layer, step, row, saturated ranks, reference ids missing.
_MISS_FIELDS = 5
_HEADER = 2  # forward passes, recorded misses
_POLL_SECONDS = 5.0
_SUMMARY_SECONDS = 60.0
_NO_KEY = 1 << 16


def _bf16_keys(bits: torch.Tensor) -> torch.Tensor:
    """bf16 bit patterns (int32 in [0, 0xFFFF]) as order-preserving keys, as
    the merge's packed words hold them, except that -0.0 keys as +0.0: the
    top-k producers order the two as equal, so they tie."""
    bits = torch.where(bits == 0x8000, 0, bits)
    return torch.where(bits >= 0x8000, ~bits & 0xFFFF, bits | 0x8000)


def saturated_ranks(sat_word: torch.Tensor) -> torch.Tensor:
    """Per row, the ranks whose local top-N the select found saturated."""
    mask = (sat_word.long() >> 16) & 0xFFFF
    bits = torch.arange(16, device=sat_word.device)
    return ((mask[:, None] >> bits) & 1).sum(1)


def compare_dcp_topk_merge(
    logits: torch.Tensor,
    row_ends: torch.Tensor,
    merged: torch.Tensor,
    reference: torch.Tensor,
    rank: int,
    world_size: int,
    all_gather: Callable[[torch.Tensor], torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Per row of the merge's global ``merged`` ids against the generic
    merge's ``reference`` ids (both ``[T, K]`` int32, -1 padded, the same on
    every rank), with this rank's bf16 ``logits`` ``[T, S]`` (``S`` covering
    every local row end) and ``row_ends`` ``[T]``.

    ``all_gather`` maps an ``[F, T]`` int32 tensor to the group's
    ``[W, F, T]``; every rank must call this. Returns ``[T]`` tensors: miss,
    tie (differs only at ``tau``), nan, missing (reference ids not in the
    merge) and valid (the row has reference ids)."""
    num_rows, width = logits.shape
    topk = merged.shape[1]
    bits = logits.view(torch.int16)
    ends = row_ends[:num_rows].long()[:, None]

    def owned(ids: torch.Tensor) -> tuple[torch.Tensor, ...]:
        own = (ids >= 0) & (ids % world_size == rank)
        local = torch.where(own, ids // world_size, 0).long()
        beyond = own & (local >= ends)
        local = torch.where(beyond, 0, local).clamp_(max=width - 1)
        keys = _bf16_keys(torch.gather(bits, 1, local).int() & 0xFFFF)
        return own & ~beyond, keys, beyond.any(1)

    own_m, key_m, beyond_m = owned(merged)
    own_r, key_r, beyond_r = owned(reference)
    in_span = torch.arange(width, device=logits.device)[None, :] < ends
    nan_local = (torch.isnan(logits) & in_span).any(1)
    tau_local = torch.where(own_r, key_r, _NO_KEY).amin(1)
    first = all_gather(torch.stack([tau_local.int(), nan_local.int()]))
    tau = first[:, 0].amin(0)[:, None]
    nan = first[:, 1].amax(0) > 0

    sorted_m = torch.sort(merged, dim=1).values
    duplicate = ((sorted_m[:, 1:] == sorted_m[:, :-1]) & (sorted_m[:, 1:] >= 0)).any(1)
    pos = torch.searchsorted(sorted_m, reference.contiguous()).clamp_(max=topk - 1)
    found = torch.gather(sorted_m, 1, pos) == reference
    below = (own_m & (key_m < tau)).any(1)
    lost = (own_r & (key_r > tau) & ~found).any(1)
    bad_local = below | lost | beyond_m | beyond_r
    second = all_gather(bad_local.int()[None])
    missing = ((reference >= 0) & ~found).sum(1)
    count_differs = (merged >= 0).sum(1) != (reference >= 0).sum(1)
    miss = ((second[:, 0].amax(0) > 0) | count_differs | duplicate) & ~nan
    return {
        "miss": miss,
        "tie": (missing > 0) & ~miss & ~nan,
        "nan": nan,
        "missing": missing,
        "valid": (reference >= 0).any(1),
    }


class DcpTopkMergeCheck:
    """The check's counters, in host memory that the device writes."""

    def __init__(self, device: torch.device) -> None:
        size = _HEADER + MAX_LAYERS * _NUM_STATS + (MAX_MISSES + 1) * _MISS_FIELDS
        pinned = device.type == "cuda"
        self.host = torch.zeros(
            size, dtype=torch.int64, device="cpu", pin_memory=pinned
        )
        self.host[_HEADER + MAX_LAYERS * _NUM_STATS :] = -1
        view = self.host
        if pinned:
            from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

            view = get_accelerator_view_from_cpu_tensor(self.host)
        self.steps = view[0:1]
        self.num_misses = view[1:2]
        stats_end = _HEADER + MAX_LAYERS * _NUM_STATS
        self.stats = view[_HEADER:stats_end].view(MAX_LAYERS, _NUM_STATS)
        # The last row takes the writes of rows that are not recorded misses.
        self.misses = view[stats_end:].view(MAX_MISSES + 1, _MISS_FIELDS)
        self.keep_max = torch.zeros(_NUM_STATS, dtype=torch.bool, device=device)
        self.keep_max[[MAX_SAT_RANKS, MAX_MISSING]] = True
        self.layers: dict[str, int] = {}
        self.names: list[str] = []
        self._logged_misses = 0
        self._last_summary: tuple[int, ...] | None = None
        self._last_summary_time = 0.0

    def layer_id(self, name: str) -> int:
        if name not in self.layers:
            self.layers[name] = min(len(self.names), MAX_LAYERS - 1)
            self.names.append(name)
        return self.layers[name]

    def record(self, layer: int, rows: dict[str, torch.Tensor], sat: torch.Tensor):
        """Add one check of ``layer`` (``compare_dcp_topk_merge``'s rows and
        the saturated ranks per row); the first layer counts a pass."""
        if layer == 0:
            self.steps.add_(1)
        miss = rows["miss"]
        update = torch.stack(
            [
                torch.ones_like(rows["missing"][0]),
                rows["valid"].sum(),
                rows["tie"].sum(),
                miss.sum(),
                rows["nan"].sum(),
                (sat > 0).sum(),
                sat.amax(),
                rows["missing"].amax(),
            ]
        ).long()
        stats = self.stats[layer]
        stats.copy_(
            torch.where(self.keep_max, torch.maximum(stats, update), stats + update)
        )
        order = self.num_misses + torch.cumsum(miss.long(), 0) - 1
        slot = torch.where(miss & (order < MAX_MISSES), order, MAX_MISSES)
        row = torch.arange(miss.shape[0], device=miss.device)
        entries = torch.stack(
            [
                torch.full_like(row, layer),
                self.steps.expand_as(row),
                row,
                sat.long(),
                rows["missing"].long(),
            ],
            dim=1,
        )
        self.misses.index_copy_(0, slot, entries)
        self.num_misses.add_(miss.long().sum())

    def snapshot(self) -> dict[str, Any]:
        """The counters as the host sees them now (the device may be ahead)."""
        values = self.host.numpy().copy()
        stats_end = _HEADER + MAX_LAYERS * _NUM_STATS
        stats = values[_HEADER:stats_end].reshape(MAX_LAYERS, _NUM_STATS)
        totals = stats.sum(0)
        totals[[MAX_SAT_RANKS, MAX_MISSING]] = stats[
            :, [MAX_SAT_RANKS, MAX_MISSING]
        ].max(0)
        misses = []
        for layer, step, row, sat, missing in values[stats_end:].reshape(
            MAX_MISSES + 1, _MISS_FIELDS
        )[: min(int(values[1]), MAX_MISSES)]:
            if layer < 0:  # not visible yet
                break
            name = self.names[layer] if layer < len(self.names) else str(layer)
            misses.append((name, int(step), int(row), int(sat), int(missing)))
        return {
            "steps": int(values[0]),
            "num_misses": int(values[1]),
            "layers": min(len(self.names), MAX_LAYERS),
            "totals": tuple(int(v) for v in totals),
            "misses": misses,
        }

    def log(self, final: bool = False) -> None:
        snap = self.snapshot()
        for name, step, row, sat, missing in snap["misses"][self._logged_misses :]:
            logger.warning(
                "DSA DCP top-k merge check: MISS layer=%s step=%d row=%d "
                "saturated_ranks=%d missing_ids=%d",
                name,
                step,
                row,
                sat,
                missing,
            )
        self._logged_misses = len(snap["misses"])
        totals = snap["totals"]
        now = time.monotonic()
        summary = (snap["steps"], *totals)
        if summary == self._last_summary and not final:
            return
        if not (
            final
            or totals[MISS_ROWS]
            or now - self._last_summary_time >= _SUMMARY_SECONDS
        ):
            return
        self._last_summary, self._last_summary_time = summary, now
        (logger.warning if totals[MISS_ROWS] else logger.info)(
            "DSA DCP top-k merge check%s: steps=%d rows=%d layers=%d missed=%d "
            "tie_only=%d nan=%d saturated=%d max_saturated_ranks=%d "
            "max_missing_ids=%d recorded_misses=%d/%d",
            " (final)" if final else "",
            snap["steps"],
            totals[ROWS],
            snap["layers"],
            totals[MISS_ROWS],
            totals[TIE_ROWS],
            totals[NAN_ROWS],
            totals[SAT_ROWS],
            totals[MAX_SAT_RANKS],
            totals[MAX_MISSING],
            len(snap["misses"]),
            snap["num_misses"],
        )

    def start_logging(self) -> None:
        def poll() -> None:
            while True:
                time.sleep(_POLL_SECONDS)
                try:
                    self.log()
                except Exception:
                    logger.exception("DSA DCP top-k merge check: logging failed")

        threading.Thread(target=poll, name="dcp-topk-merge-check", daemon=True).start()
        atexit.register(self.log, final=True)


_STATE: DcpTopkMergeCheck | None = None


def init_dcp_topk_merge_check(device: torch.device, rank_in_group: int) -> None:
    """Turn the check on (once per process, at model construction); DCP rank 0
    logs."""
    global _STATE
    if _STATE is not None:
        return
    _STATE = DcpTopkMergeCheck(device)
    if rank_in_group == 0:
        _STATE.start_logging()
    logger.warning_once(
        "VLLM_DSA_BPRIME_CHECK: every one-round DCP top-k merge also runs the "
        "generic merge for comparison (debug only, slower decode)."
    )


def dcp_topk_merge_check_enabled() -> bool:
    return _STATE is not None


def get_dcp_topk_merge_check() -> DcpTopkMergeCheck | None:
    return _STATE


def _dcp_all_gather(world_size: int) -> Callable[[torch.Tensor], torch.Tensor]:
    from vllm.distributed import get_dcp_group

    group = get_dcp_group()

    def gather(tensor: torch.Tensor) -> torch.Tensor:
        return group.all_gather(tensor, dim=0).view(world_size, *tensor.shape)

    return gather


def check_dcp_topk_merge(
    layer_name: str,
    logits: torch.Tensor,
    row_ends: torch.Tensor,
    merged: torch.Tensor,
    reference: torch.Tensor,
    sat_word: torch.Tensor,
    rank: int,
    world_size: int,
    all_gather: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> None:
    """Compare and record one merge call (see ``compare_dcp_topk_merge``).
    Collective over the DCP group."""
    state = _STATE
    assert state is not None, "init_dcp_topk_merge_check was not called"
    rows = compare_dcp_topk_merge(
        logits,
        row_ends,
        merged,
        reference,
        rank,
        world_size,
        all_gather or _dcp_all_gather(world_size),
    )
    state.record(
        state.layer_id(layer_name), rows, saturated_ranks(sat_word[: merged.shape[0]])
    )


def exchange_slot_ids(
    slots: torch.Tensor,
    counts: torch.Tensor,
    block_table: torch.Tensor,
    req_of_row: torch.Tensor,
    block_size: int,
    block_stride: int,
    row_ends: torch.Tensor,
    rank: int,
    world_size: int,
) -> torch.Tensor:
    """A top-k given as this rank's KV slots (Kimi K3.1 DcpExchange: ``slots``
    [T, K], the first ``counts`` [T] of a row valid, slot = block_table[req,
    pos // block_size] * block_stride + pos % block_size) back as global ids
    (interleave 1: ``pos * W + rank``). Returns int32 [T, K + 1]: the ids, -1
    padded, and a last column set on rows with a slot that maps to no position
    of the row below ``row_ends`` (``merge_exchange_ids`` then fails the row).
    No host syncs."""
    num_rows, topk = slots.shape
    device = slots.device
    taken = torch.arange(topk, device=device)[None, :] < counts[:num_rows, None]
    sel = slots.long()
    block = torch.div(sel, block_stride, rounding_mode="floor")
    offset = sel - block * block_stride
    ends = row_ends[:num_rows].long()[:, None]
    table = block_table[req_of_row[:num_rows].long()].long()
    first = torch.arange(table.shape[1], device=device)[None, :] * block_size
    # A row's blocks in use, sorted: the table's tail may hold stale ids.
    keys = torch.where(first < ends, table, torch.iinfo(torch.int64).max)
    keys, order = torch.sort(keys, dim=1)
    at = torch.searchsorted(keys, block.contiguous()).clamp_(max=keys.shape[1] - 1)
    local = torch.gather(order, 1, at) * block_size + offset
    ok = (torch.gather(keys, 1, at) == block) & (offset < block_size) & (local < ends)
    ids = torch.where(taken & ok, local * world_size + rank, -1)
    bad = (taken & ~ok).any(1, keepdim=True)
    return torch.cat([ids, bad.long()], 1).int()


def merge_exchange_ids(every: torch.Tensor) -> torch.Tensor:
    """Every rank's ``exchange_slot_ids`` ([W, T, K + 1]) as the merged global ids
    [T, K] for ``compare_dcp_topk_merge`` (the same on every rank). A row with a
    flagged slot or more than K ids in all comes out empty, so the comparison
    counts it as missed."""
    world_size, num_rows, width = every.shape
    topk = width - 1
    ids = every[..., :topk].permute(1, 0, 2).reshape(num_rows, world_size * topk)
    ids = torch.sort(ids, dim=1, descending=True).values
    broken = (every[..., topk] > 0).any(0) | (ids[:, topk:] >= 0).any(1)
    return torch.where(broken[:, None], -1, ids[:, :topk])
