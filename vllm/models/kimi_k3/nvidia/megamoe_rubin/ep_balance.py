# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Served EP (MoE) balance log, host side (int64.3 item 3).

The dedup router's waiting warp writes, on one MegaMoE call in ``PERIOD`` (call ids are per session, 1-based), the
GLOBAL per-expert token counts of that call (complete on every rank once ``sizes_ready == world_size``) and a
%globaltimer stamp into a slot of its session's host-mapped ring (token_comm_dedup.py ``_log_ep_balance``, same
constants). ``drain_all()`` reads completed slots with plain loads on the caller's thread (no watcher thread) and
returns per-call summaries: per-EP-rank token load and max / mean over ranks. The worker iterlog writes them as
"E" lines (rank 0 is enough: every rank sees the global counts). Always on; the device cost is one sampled slot
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


class EpBalanceRing:
    """One MegaMoE session's ring (sessions keep separate call ids)."""

    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.buffer = torch.zeros(SLOTS * SLOT_WORDS, dtype=torch.int32, pin_memory=True)
        self._last = 0  # last call id drained

    @property
    def address(self) -> int:
        """Device-accessible address of slot 0 (UVA: the pinned host pointer)."""
        return self.buffer.data_ptr()

    def _word(self, i: int, j: int) -> int:
        return ctypes.c_int32.from_address(self.buffer.data_ptr() + 4 * (i * SLOT_WORDS + j)).value

    def _slot(self, i: int) -> list[int]:
        base = self.buffer.data_ptr() + 4 * i * SLOT_WORDS
        return list((ctypes.c_int32 * SLOT_WORDS).from_address(base))

    def drain(self) -> list[dict]:
        """Summaries of the calls completed since the last drain, in call order. The next sampled call has id
        last + PERIOD in slot ((last >> SHIFT) + 1) % SLOTS: one word read per drain when nothing is new."""
        out = []
        for _ in range(SLOTS):
            i = ((self._last >> SHIFT) + 1) % SLOTS
            call = self._word(i, 0)
            if call <= self._last:
                break
            w = self._slot(i)
            if w[0] != call:  # being rewritten: take it on the next drain
                break
            self._last = call
            r = summarize(w)
            r["session"] = self.tag
            out.append(r)
        return out


def summarize(w: list[int]) -> dict:
    """One slot -> per-EP-rank loads (experts are owned contiguously: rank r has [r*epr, (r+1)*epr))."""
    t_ready = (w[1] & 0xFFFFFFFF) | ((w[2] & 0xFFFFFFFF) << 32)
    padded, world, epr = w[3], w[4], w[5]
    sizes = w[HEADER_WORDS : HEADER_WORDS + padded]
    loads = [sum(sizes[r * epr : (r + 1) * epr]) for r in range(world)] if world and epr else []
    mean = sum(loads) / len(loads) if loads else 0.0
    return {
        "call": w[0],
        "t_ready_ns": t_ready,
        "world": world,
        "experts_per_rank": epr,
        "routes": sum(sizes),
        "rank_loads": loads,
        "max_over_mean": (max(loads) / mean) if mean > 0 else 0.0,
        "expert_max": max(sizes) if sizes else 0,
    }


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
    return [("E", r["session"], r["call"], r["t_ready_ns"], r["routes"], round(r["max_over_mean"], 4),
             r["expert_max"], r["rank_loads"]) for r in drain_all()]
