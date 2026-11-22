# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bookkeeping for async KV loads admitted in the step that issues them.

A connector may join an async load to the step that allocated its blocks
(KVConnectorBase_V1.join_load_in_step) when it can have the KV in the blocks
before that step's forward reads them. The scheduler keeps only the hook
points; this module holds the rest:

- InflightStepClock: how long the steps in flight still run (the headroom the
  connector compares its load-time prediction against);
- JoinedLoadTracker: which joined loads have not reported done (their blocks
  stay uncached, and parked instead of freed if the request is preempted or
  finished, since the transfer writes them outside the GPU stream), the
  handling of their done/failed reports, and a count of store reads of blocks a
  joined load is still loading.
"""

import time
from collections import Counter, deque
from dataclasses import replace
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import BlockHash, KVCacheBlock
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request, RequestStatus

if TYPE_CHECKING:
    from vllm.v1.core.sched.scheduler import Scheduler

logger = init_logger(__name__)


class InflightStepClock:
    """Predicts when the steps in flight end, for join_load_in_step.

    A step is in flight from schedule() returning it until its output is
    processed. Its GPU work starts at the later of its dispatch and the end of
    the step before it; its duration is taken as a low quantile of recent
    measured durations (start to output processed), so the prediction errs
    early: a load predicted to beat it usually does.
    """

    # Steps never processed (e.g. dropped on a reset) leave after this long.
    STALE_S = 30.0

    def __init__(self, window: int = 64, quantile: float = 0.2, min_samples: int = 8):
        self._inflight: dict[int, float] = {}  # id(scheduler_output) -> dispatch
        self._last_done: float | None = None
        self._durations: deque[float] = deque(maxlen=window)
        self._quantile = quantile
        self._min_samples = min_samples

    def dispatched(self, step_key: int, now: float) -> None:
        self._inflight[step_key] = now

    def done(self, step_key: int, now: float) -> None:
        dispatch = self._inflight.pop(step_key, None)
        if dispatch is None:
            return
        start = dispatch if self._last_done is None else max(dispatch, self._last_done)
        self._durations.append(now - start)
        self._last_done = now

    def headroom(self, now: float) -> float:
        """Seconds until the steps in flight are predicted to end (0 when
        none is in flight or too few steps were measured)."""
        for key in [k for k, d in self._inflight.items() if now - d > self.STALE_S]:
            del self._inflight[key]
        if not self._inflight or len(self._durations) < self._min_samples:
            return 0.0
        ordered = sorted(self._durations)
        step = ordered[min(len(ordered) - 1, int(self._quantile * len(ordered)))]
        end = self._last_done if self._last_done is not None else now
        for dispatch in self._inflight.values():
            end = max(end, dispatch) + step
        return max(end - now, 0.0)


class JoinedLoadTracker:
    """Joined async loads of one scheduler, from the join until the load's
    done report (see the module docstring)."""

    def __init__(self, scheduler: "Scheduler") -> None:
        self._scheduler = scheduler
        self.clock = InflightStepClock()
        # Joined loads not reported done: their requests' blocks stay uncached.
        self._loading: set[str] = set()
        # Blocks of a request preempted or finished while its joined load had
        # not reported; the request is not admitted again until it does (a new
        # load under its id would be confused with the old one's report).
        self._parked: dict[str, list[list[KVCacheBlock]]] = {}
        self._hashes: dict[str, list[BlockHash]] = {}
        self._hash_refs: Counter[BlockHash] = Counter()
        self.stats = {"joined": 0, "dup_reads": 0, "dup_read_blocks": 0}

    def headroom(self) -> float:
        return self.clock.headroom(time.monotonic())

    def dispatched(self, step_key: int) -> None:
        self.clock.dispatched(step_key, time.monotonic())

    def done(self, step_key: int) -> None:
        self.clock.done(step_key, time.monotonic())

    def loading(self, req_id: str) -> bool:
        """Its joined load has not reported done (keep its blocks uncached)."""
        return req_id in self._loading

    def parked(self, req_id: str) -> bool:
        """Its request waits for an earlier joined load to report."""
        return req_id in self._parked

    def park(self, req_id: str, blocks: list[KVCacheBlock]) -> None:
        self._parked.setdefault(req_id, []).append(blocks)

    def on_async_load(
        self, request: Request, start_tokens: int, end_tokens: int, joined: bool
    ) -> None:
        """An async load of ``request``'s tokens [start, end) was allocated."""
        size = self._scheduler.hash_block_size
        hashes = list(request.block_hashes[start_tokens // size : end_tokens // size])
        dup = sum(1 for hash_ in hashes if hash_ in self._hash_refs)
        if dup:
            # Read from the store while a joined load still loads them (they
            # are unhittable until it reports clean).
            self.stats["dup_reads"] += 1
            self.stats["dup_read_blocks"] += dup
            if self.stats["dup_reads"] % 100 == 1:
                logger.info("Same-step joined loads: %s", dict(self.stats))
        if joined:
            self._loading.add(request.request_id)
            self._hashes[request.request_id] = hashes
            self._hash_refs.update(hashes)
            self.stats["joined"] += 1

    def settle(
        self, kv_connector_output: KVConnectorOutput
    ) -> tuple[KVConnectorOutput, set[str]]:
        """Take the reports of joined loads out of ``kv_connector_output``
        before the scheduler reads it.

        A failed load (reported by its first step's output at the latest; a
        worker that gives up waiting reports it failed while the transfer may
        still run) was admitted with KV that never arrived: its output is
        skipped and, under the "recompute" policy, it is preempted (in-flight
        output dropped) to recompute from scratch, else it fails as any failed
        load does. Its blocks were never cached, so no other request read them,
        and they stay parked until the load reports done.

        A done report ends the load: a request still admitted gets its blocks
        cached (held back so far); a preempted or finished one gets its parked
        blocks freed (after the steps in flight) and may be admitted again.
        Returns the filtered output and the ids whose output this step skips.
        """
        reported_failed = kv_connector_output.failed_recving or set()
        reported_done = kv_connector_output.finished_recving or set()
        failed = reported_failed & self._loading
        done = reported_done & self._loading
        if not failed and not done:
            return kv_connector_output, set()
        sched = self._scheduler
        skip: set[str] = set()
        timestamp = time.monotonic()
        for req_id in failed:
            request = sched.requests.get(req_id)
            if request is None or request.is_finished():
                continue
            skip.add(req_id)
            if req_id in self._parked or not sched.recompute_kv_load_failures:
                # Already reset by an earlier report, or policy "fail":
                # update_from_output finishes it with an error.
                continue
            if request.status == RequestStatus.RUNNING:
                sched.running.remove(request)
                sched._preempt_request(request, timestamp, drop_stale_output=True)
            else:
                # Admitted but not scheduled yet: back to a fresh start.
                sched._free_request_blocks(request)
                sched._inflight_prefills.discard(request)
                request.num_computed_tokens = 0
            self._parked.setdefault(req_id, [])
        for req_id in done:
            self._loading.discard(req_id)
            for hash_ in self._hashes.pop(req_id, ()):
                self._hash_refs[hash_] -= 1
                if self._hash_refs[hash_] <= 0:
                    del self._hash_refs[hash_]
            parked = self._parked.pop(req_id, None)
            request = sched.requests.get(req_id)
            if parked is not None:
                for blocks in parked:
                    sched._free_parked_blocks(blocks)
                if request is not None and request.is_finished():
                    # Its free waited for this report.
                    del sched.requests[req_id]
                continue
            if (
                request is not None
                and req_id not in failed
                and request.status in (RequestStatus.RUNNING, RequestStatus.WAITING)
                and request.num_computed_tokens > 0
            ):
                sched.kv_cache_manager.cache_blocks(
                    request, min(request.num_computed_tokens, request.num_tokens)
                )
        settled = failed | done
        return (
            replace(
                kv_connector_output,
                finished_recving=(reported_done - settled) or None,
                failed_recving=reported_failed - settled,
            ),
            skip,
        )
