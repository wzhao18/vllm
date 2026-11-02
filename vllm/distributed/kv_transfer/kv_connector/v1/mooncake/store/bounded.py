# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded Mooncake store calls: a store get / put that never returns (e.g. a
transfer stuck on a paused RDMA rail) must not hang its request forever.

``StoreCallGuard.call`` runs one store call on a dedicated helper thread and
waits at most ``timeout_s``. On timeout it raises ``StoreCallTimeout``, which
the transfer threads' existing error paths already turn into a load error (the
request recomputes those tokens) or a dropped put. The stuck call cannot be
cancelled, so the guard stays *degraded* until it returns: every further call
fails fast instead of queueing behind it (cache miss / dropped put), which
bounds the stuck helper threads to one per transfer thread.

When a timed-out call completes late, ``on_late`` runs with the operation name,
its keys and its outcome (the put path removes those keys, since the source
blocks may have been reused meanwhile). A late *get* writes into KV blocks the
request no longer waits for; it is logged as an error and counted so a run can
flag it.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from typing import Any, TypeVar

from vllm.logger import init_logger

logger = init_logger(__name__)

_T = TypeVar("_T")

DEFAULT_STORE_OP_TIMEOUT_S = 30.0


class StoreCallTimeout(RuntimeError):
    """A store call timed out, or the guard is degraded by an earlier one."""


@dataclass
class StoreCallCounters:
    timeouts: int = 0
    degraded_skips: int = 0
    late_completions: int = 0
    late_gets: int = 0


LateCallback = Callable[[str, Sequence[str], BaseException | None], None]


class StoreCallGuard:
    """Runs store calls with a deadline; one guard per transfer thread."""

    def __init__(
        self,
        name: str,
        timeout_s: float,
        on_late: LateCallback | None = None,
    ) -> None:
        if not timeout_s > 0:
            raise ValueError(f"store op timeout must be > 0, got {timeout_s}")
        self.name = name
        self.timeout_s = float(timeout_s)
        self.on_late = on_late
        self.counters = StoreCallCounters()
        self._lock = threading.Lock()
        self._stuck: Future[Any] | None = None
        self._stuck_op = ""
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"{name}-store-call"
        )

    @property
    def degraded(self) -> bool:
        with self._lock:
            return self._stuck is not None and not self._stuck.done()

    def call(
        self,
        op: str,
        fn: Callable[..., _T],
        *args: Any,
        keys: Sequence[str] = (),
        timeout_s: float | None = None,
    ) -> _T:
        timeout = self.timeout_s if timeout_s is None else timeout_s
        with self._lock:
            if self._stuck is not None and not self._stuck.done():
                self.counters.degraded_skips += 1
                raise StoreCallTimeout(
                    f"{self.name}: {op} skipped, an earlier {self._stuck_op} "
                    f"has not returned (store degraded)"
                )
            future = self._executor.submit(fn, *args)
        try:
            return future.result(timeout=timeout)
        except FutureTimeoutError:
            pass
        with self._lock:
            self._stuck = future
            self._stuck_op = op
            self.counters.timeouts += 1
            counters = self.counters
        logger.warning(
            "Mooncake store failover: %s %s of %d keys timed out after %.1f s "
            "(treated as %s; store degraded until it returns; timeouts=%d)",
            self.name,
            op,
            len(keys),
            timeout,
            "a dropped put" if "put" in op else "a cache miss",
            counters.timeouts,
        )
        future.add_done_callback(
            lambda f, op=op, keys=list(keys): self._late(op, keys, f)  # type: ignore[misc]
        )
        raise StoreCallTimeout(f"{self.name}: {op} timed out after {timeout:.1f} s")

    def _late(self, op: str, keys: list[str], future: Future[Any]) -> None:
        error = future.exception()
        with self._lock:
            self.counters.late_completions += 1
            if "get" in op:
                self.counters.late_gets += 1
        log = logger.error if "get" in op else logger.warning
        log(
            "Mooncake store failover: %s %s of %d keys returned late (%s)%s",
            self.name,
            op,
            len(keys),
            f"error: {error}" if error is not None else "completed",
            "; it may have written KV blocks after its request moved on"
            if "get" in op and error is None
            else "",
        )
        if self.on_late is not None:
            try:
                self.on_late(op, keys, error)
            except Exception as e:  # noqa: BLE001 - cleanup is best effort
                logger.warning("Mooncake store failover cleanup failed: %s", e)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
