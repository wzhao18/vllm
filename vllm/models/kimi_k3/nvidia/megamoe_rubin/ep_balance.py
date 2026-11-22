# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Served EP (MoE) balance log, host side (int64.3 item 3).

The dedup router's waiting warp writes, on one MegaMoE call in ``PERIOD`` (call ids are per session, 1-based), the
GLOBAL per-expert token counts of that call (complete on every rank once ``sizes_ready == world_size``) and a
%globaltimer stamp into a slot of its session's host-mapped ring (token_comm_dedup.py ``_log_ep_balance``, same
constants). ``drain_all()`` reads completed slots with plain loads on the caller's thread (no watcher thread) and
returns per-call summaries: per-EP-rank token load and max / mean over ranks. The worker iterlog writes them as
"E" lines (rank 0 is enough: every rank sees the global counts).

Assumptions (recorded in the iterlog header): experts are owned contiguously, rank r holds experts
[r * experts_per_rank, (r + 1) * experts_per_rank) (static placement: dynamic EPLB is off); a session's calls cycle
through the model's MoE layers in order, so layer = (call - 1) % moe_layers once ``configure`` set the count. Always on; the device cost is one sampled slot
of stores per 64 calls, the host cost one word read per ring per step (iterlog only).
"""

from __future__ import annotations

import ctypes
import threading

import torch

SHIFT = 6
PERIOD = 1 << SHIFT
SLOTS = 256
HEADER_WORDS = 8
MAX_EXPERTS = 512
SLOT_WORDS = HEADER_WORDS + MAX_EXPERTS

_lock = threading.Lock()
_rings: list["EpBalanceRing"] = []
_moe_layers = 0
RESCAN_EVERY = 64  # drains without progress between full scans (resync after a call-id reset)


def configure(moe_layers: int) -> None:
    """MoE layer count of the model (layer index of each record)."""
    global _moe_layers
    _moe_layers = int(moe_layers)


class EpBalanceRing:
    """One MegaMoE session's ring (sessions keep separate call ids)."""

    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.buffer = torch.zeros(SLOTS * SLOT_WORDS, dtype=torch.int32, pin_memory=True)
        self._last = 0  # last call id drained
        self._idle = 0  # drains without progress
        self.lost = 0  # sampled calls overwritten before they were drained

    @property
    def address(self) -> int:
        """Device-accessible address of slot 0 (UVA: the pinned host pointer)."""
        return self.buffer.data_ptr()

    def _word(self, i: int, j: int) -> int:
        return ctypes.c_int32.from_address(self.buffer.data_ptr() + 4 * (i * SLOT_WORDS + j)).value

    def _slot(self, i: int) -> list[int]:
        base = self.buffer.data_ptr() + 4 * i * SLOT_WORDS
        return list((ctypes.c_int32 * SLOT_WORDS).from_address(base))

    def drain(self, _retry: bool = True) -> list[dict]:
        """Summaries of the calls completed since the last drain, in call order. The next sampled call has id
        last + PERIOD in slot ((last >> SHIFT) + 1) % SLOTS: one word read per drain when nothing is new. A lapped
        slot (a newer call) is taken and the skipped calls are counted as lost; after RESCAN_EVERY drains without
        progress all slots are scanned, which resyncs after a call-id reset (session re-carve)."""
        out = []
        for _ in range(SLOTS):
            i = ((self._last >> SHIFT) + 1) % SLOTS
            call = self._word(i, 0)
            if call <= self._last:
                break
            w = self._slot(i)
            if w[0] != call:  # being rewritten: take it on the next drain
                break
            self.lost += max(0, (call - self._last) // PERIOD - 1)
            self._last = call
            out.append(self._record(w))
        if out:
            self._idle = 0
            return out
        self._idle += 1
        if self._idle >= RESCAN_EVERY:
            self._idle = 0
            calls = [self._word(i, 0) for i in range(SLOTS)]
            stamps = [(self._word(i, 1) & 0xFFFFFFFF) | ((self._word(i, 2) & 0xFFFFFFFF) << 32) for i in range(SLOTS)]
            newest = max(range(SLOTS), key=lambda i: stamps[i] if calls[i] > 0 else -1)
            if 0 < calls[newest] < self._last:
                # the most recently written slot holds an older id: the session restarted its call ids
                self._last = calls[newest] - PERIOD
            else:
                newer = sorted(c for c in calls if c > self._last)
                if newer and newer[0] > self._last + PERIOD:  # stuck behind a skipped slot: jump to the oldest newer
                    self.lost += max(0, (newer[0] - self._last) // PERIOD - 1)
                    self._last = newer[0] - PERIOD
            if _retry:
                return self.drain(_retry=False)
        return out

    def _record(self, w: list[int]) -> dict:
        r = summarize(w)
        r["session"] = self.tag
        r["layer"] = (r["call"] - 1) % _moe_layers if _moe_layers else -1
        r["lost"] = self.lost
        return r


def summarize(w: list[int]) -> dict:
    """One slot -> per-EP-rank loads (experts are owned contiguously: rank r has [r*epr, (r+1)*epr))."""
    """Global record (word 6 == 0, dedup router / prefill): every expert's count, so per-rank loads and max/mean.
    Local record (word 6 == local_rank + 1, genphase / decode): this rank's experts only; rank_loads holds this rank's
    load and max_over_mean is -1 (the host merges the ranks' records by call id, see merge_local)."""
    t_ready = (w[1] & 0xFFFFFFFF) | ((w[2] & 0xFFFFFFFF) << 32)
    entries, world, epr, local = w[3], w[4], w[5], w[6]
    sizes = w[HEADER_WORDS : HEADER_WORDS + entries]
    if local > 0:
        loads, mm = [sum(sizes)], -1.0
    else:
        loads = [sum(sizes[r * epr : (r + 1) * epr]) for r in range(world)] if world and epr else []
        mean = sum(loads) / len(loads) if loads else 0.0
        mm = (max(loads) / mean) if mean > 0 else 0.0
    return {
        "call": w[0],
        "t_ready_ns": t_ready,
        "world": world,
        "experts_per_rank": epr,
        "local_rank": local - 1,
        "routes": sum(sizes),
        "rank_loads": loads,
        "max_over_mean": mm,
        "expert_max": max(sizes) if sizes else 0,
    }


def merge_local(records_by_rank: list[list[dict]]) -> list[dict]:
    """Merge local (decode) records of all ranks by (session, call): per-rank loads, routes and max/mean."""
    by_call: dict = {}
    for recs in records_by_rank:
        for r in recs:
            if r["local_rank"] >= 0:
                by_call.setdefault((r.get("session"), r["call"]), {})[r["local_rank"]] = r
    out = []
    for (session, call), ranks in sorted(by_call.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        world = next(iter(ranks.values()))["world"]
        loads = [ranks[r]["routes"] if r in ranks else None for r in range(world)]
        have = [x for x in loads if x is not None]
        mean = sum(have) / len(have) if have else 0.0
        out.append({"session": session, "call": call, "rank_loads": loads, "complete": len(have) == world,
                    "routes": sum(have), "max_over_mean": (max(have) / mean) if mean > 0 else 0.0})
    return out


def new_ring(tag: str) -> EpBalanceRing:
    """A ring for one session (before its first launch); registers the iterlog source once."""
    with _lock:
        ring = EpBalanceRing(tag)
        _rings.append(ring)
        if len(_rings) == 1:
            try:
                from vllm.v1.metrics import iterlog

                iterlog.register_worker_source(_iterlog_records)
            except Exception:  # noqa: BLE001 - logging only
                pass
        return ring


def drain_all() -> list[dict]:
    out = []
    for ring in list(_rings):
        out.extend(ring.drain())
    return out


def _iterlog_records() -> list[tuple]:
    return [("E", r["session"], r["call"], r["layer"], r["local_rank"], r["t_ready_ns"], r["routes"],
             round(r["max_over_mean"], 4), r["expert_max"], r["lost"], r["rank_loads"]) for r in drain_all()]
