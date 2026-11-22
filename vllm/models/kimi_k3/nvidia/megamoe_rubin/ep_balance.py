# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Served EP (MoE) balance log, host side (int64.3 item 3).

The dedup router's waiting warp writes, on one MegaMoE call in ``PERIOD`` (call ids are per session, 1-based), the
GLOBAL per-expert token counts of that call (complete on every rank once ``sizes_ready == world_size``) and a
%globaltimer stamp into a slot of its session's DEVICE-memory ring (token_comm_dedup.py ``log_ep_balance``; the
genphase helper writes this rank's local counts, token_comm_gen_specialized.py ``_log_ep_balance``; same constants).
Kernels use gpu-scope stores and no fence (pinned-host stores cost ~100 us under MegaMoE load to whatever next
orders at system scope: MEGAMOEFIX tasks C-Gp). ``drain_all()`` takes the slots copied out (asynchronously, on the
ring's own stream, never waiting for the GPU) by the previous drain when the call id and the checksum agree, issues
the next copy, and returns per-call summaries: per-EP-rank token load and max / mean over ranks. The worker iterlog writes them as
"E" lines (rank 0 is enough: every rank sees the global counts).

Assumptions (recorded in the iterlog header): experts are owned contiguously, rank r holds experts
[r * experts_per_rank, (r + 1) * experts_per_rank) (static placement: dynamic EPLB is off); a session's calls cycle
through the model's MoE layers in order, so layer = (call - 1) % moe_layers once ``configure`` set the count. Always on; the device cost is one sampled slot
of stores per 64 calls, the host cost one 8 KB device-to-host copy per ring per drain (iterlog only).
"""

from __future__ import annotations

import threading

import torch

from .src_gp.common.ep_balance_layout import (  # the device layout (pure Python: no CuTe import)
    EPBAL_HEADER_WORDS as HEADER_WORDS,
    EPBAL_MAX_EXPERTS as MAX_EXPERTS,
    EPBAL_PERIOD as PERIOD,
    EPBAL_SHIFT as SHIFT,
    EPBAL_SLOT_WORDS as SLOT_WORDS,
    EPBAL_SLOTS as SLOTS,
    epbal_checksum,
)

_lock = threading.Lock()
_rings: list["EpBalanceRing"] = []
_moe_layers = 0
RESCAN_EVERY = 64  # drains without progress between full scans (resync after a call-id reset)
WINDOW = 4  # slots copied per drain (decode: ~1.4 sampled calls per ring per step)


def configure(moe_layers: int) -> None:
    """MoE layer count of the model (layer index of each record)."""
    global _moe_layers
    _moe_layers = int(moe_layers)


class EpBalanceRing:
    """One MegaMoE session's ring (sessions keep separate call ids), in device memory.

    The host never waits for the GPU: each drain issues an asynchronous copy of the next WINDOW slots (or of the whole
    ring for a resync) on the ring's own stream, records an event, and takes the data at a later drain once the event
    has completed (copy engines can be busy with bulk KV traffic; a small D2H must not block the worker thread)."""

    def __init__(self, tag: str, device: torch.device | str | int | None = None) -> None:
        self.tag = tag
        device = torch.device("cuda", torch.cuda.current_device()) if device is None else torch.device(device)
        self.buffer = torch.zeros(SLOTS * SLOT_WORDS, dtype=torch.int32, device=device)
        self._host = torch.zeros(SLOTS * SLOT_WORDS, dtype=torch.int32, pin_memory=True)  # copy-out staging
        self._stream = torch.cuda.Stream(device=device)  # copy-outs never wait for (or block) the compute stream
        self._event = torch.cuda.Event()
        self._pending: tuple[str, int] | None = None  # ("window", first slot) or ("scan", 0) in flight
        self._last = 0  # last call id drained
        self._idle = 0  # drains without progress
        self.lost = 0  # sampled calls overwritten before they were drained

    @property
    def address(self) -> int:
        """Device address of slot 0."""
        return self.buffer.data_ptr()

    def _issue(self, kind: str, first: int) -> None:
        """Copy WINDOW slots from ``first`` (wrapping), or the whole ring, to the host staging on the ring's stream.
        The copy races the kernels by design: the checksum rejects a slot whose stores were still landing."""
        with torch.cuda.stream(self._stream):
            if kind == "scan":
                self._host.copy_(self.buffer, non_blocking=True)
            else:
                for k in range(WINDOW):
                    i = (first + k) % SLOTS
                    lo, hi = i * SLOT_WORDS, (i + 1) * SLOT_WORDS
                    self._host[lo:hi].copy_(self.buffer[lo:hi], non_blocking=True)
            self._event.record(self._stream)
        self._pending = (kind, first)

    def _host_slot(self, i: int) -> list[int]:
        return self._host[i * SLOT_WORDS : (i + 1) * SLOT_WORDS].tolist()

    def drain(self) -> list[dict]:
        """Summaries of the calls completed since the last drain, in call order (from the copy issued by an earlier
        drain; then the next copy is issued). The next sampled call has id last + PERIOD in slot
        ((last >> SHIFT) + 1) % SLOTS. A lapped slot (a newer call) is taken and the skipped calls are counted as
        lost; after RESCAN_EVERY drains without progress the whole ring is copied and scanned, which resyncs after a
        call-id reset (session re-carve)."""
        if torch.cuda.is_current_stream_capturing():
            return []
        out: list[dict] = []
        if self._pending is not None:
            if not self._event.query():
                return out  # copy still in flight: take it next time
            kind, first = self._pending
            self._pending = None
            if kind == "scan":
                self._resync()
            else:
                for k in range(WINDOW):
                    i = (first + k) % SLOTS
                    if i != ((self._last >> SHIFT) + 1) % SLOTS:
                        break
                    w = self._host_slot(i)
                    call = w[0]
                    if call <= self._last or not slot_complete(w):  # nothing new, or still landing
                        break
                    self.lost += max(0, (call - self._last) // PERIOD - 1)
                    self._last = call
                    out.append(self._record(w))
                if out:
                    self._idle = 0
                else:
                    self._idle += 1
        if self._idle >= RESCAN_EVERY:
            self._idle = 0
            self._issue("scan", 0)
        else:
            self._issue("window", ((self._last >> SHIFT) + 1) % SLOTS)
        return out

    def _resync(self) -> None:
        calls = [int(self._host[i * SLOT_WORDS]) for i in range(SLOTS)]
        stamps = [(int(self._host[i * SLOT_WORDS + 1]) & 0xFFFFFFFF) | ((int(self._host[i * SLOT_WORDS + 2]) & 0xFFFFFFFF) << 32)
                  for i in range(SLOTS)]
        newest = max(range(SLOTS), key=lambda i: stamps[i] if calls[i] > 0 else -1)
        if 0 < calls[newest] < self._last:
            # the most recently written slot holds an older id: the session restarted its call ids
            self._last = calls[newest] - PERIOD
        else:
            newer = sorted(c for c in calls if c > self._last)
            if newer and newer[0] > self._last + PERIOD:  # stuck behind a skipped slot: jump to the oldest newer
                self.lost += max(0, (newer[0] - self._last) // PERIOD - 1)
                self._last = newer[0] - PERIOD

    def _record(self, w: list[int]) -> dict:
        r = summarize(w)
        r["session"] = self.tag
        r["layer"] = (r["call"] - 1) % _moe_layers if _moe_layers else -1
        r["lost"] = self.lost
        return r


def slot_complete(w: list[int]) -> bool:
    """Records are written without a fence (and copied out while kernels run), so a slot is complete only when
    word 7 matches the position-weighted checksum of its call id and counts (a record whose counts are still
    landing, or mix two calls of the same size, equal sums, fails)."""
    entries = w[3]
    if not 0 < entries <= MAX_EXPERTS:
        return False
    return w[7] == epbal_checksum(w[0], w[HEADER_WORDS : HEADER_WORDS + entries])


def summarize(w: list[int]) -> dict:
    """One slot -> per-EP-rank loads (experts are owned contiguously: rank r has [r*epr, (r+1)*epr))."""
    """Global record (word 6 == 0, dedup router / prefill): every expert's count, so per-rank loads and max/mean.
    Local record (word 6 == local_rank + 1, genphase / decode): this rank's experts only; rank_loads holds this rank's
    load and max_over_mean is -1 (the host merges the ranks' records by call id, see merge_local).
    ``active_per_rank``: experts with at least one token, per EP rank ([this rank] for a local record)."""
    t_ready = (w[1] & 0xFFFFFFFF) | ((w[2] & 0xFFFFFFFF) << 32)
    entries, world, epr, local = w[3], w[4], w[5], w[6]
    sizes = w[HEADER_WORDS : HEADER_WORDS + entries]
    if local > 0:
        loads, mm = [sum(sizes)], -1.0
        active = [sum(1 for c in sizes if c > 0)]
    else:
        slices = [sizes[r * epr : (r + 1) * epr] for r in range(world)] if world and epr else []
        loads = [sum(x) for x in slices]
        active = [sum(1 for c in x if c > 0) for x in slices]
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
        # ACTIVE experts (tokens > 0) per rank: the decode MegaMoE cost variable (weight streaming, ~+1.17 us per
        # active expert per layer; EPLB.md 15:04). Local record: [this rank's count]; global: one per EP rank.
        "active_per_rank": active,
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
        active = [ranks[r]["active_per_rank"][0] if r in ranks else None for r in range(world)]
        have = [x for x in loads if x is not None]
        act = [x for x in active if x is not None]
        mean = sum(have) / len(have) if have else 0.0
        out.append({"session": session, "call": call, "rank_loads": loads, "complete": len(have) == world,
                    "routes": sum(have), "max_over_mean": (max(have) / mean) if mean > 0 else 0.0,
                    "expert_max": max((ranks[r]["expert_max"] for r in ranks), default=0),
                    "active_per_rank": active, "active_busiest": max(act, default=0),
                    "active_mean": (sum(act) / len(act)) if act else 0.0})
    return out


def new_ring(tag: str, device: torch.device | str | int | None = None) -> EpBalanceRing:
    """A ring for one session (before its first launch), on ``device``; registers the iterlog source once."""
    with _lock:
        ring = EpBalanceRing(tag, device)
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
             round(r["max_over_mean"], 4), r["expert_max"], r["lost"], r["rank_loads"], r["active_per_rank"])
            for r in drain_all()]
