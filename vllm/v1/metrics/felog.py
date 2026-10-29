# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request API-server stage log and process GC / RSS log (off by default).

Set ``VLLM_FELOG_DIR=<dir>`` to turn it on. The API server writes
``api-<host>-<pid>.jsonl`` and EngineCore ``engine-<host>-<pid>.jsonl`` with the
iterlog writer (a deque on the hot path, a daemon thread doing the I/O; the
same ``H`` / ``A`` line conventions). Off, every hook is one ``ON`` test.

Records:

- ``Q`` (API server, one per request handed to EngineCore): ``time.perf_counter``
  stamps of each stage (see ``Q_FIELDS``), the render thread's CPU time, CPU id
  and context switches, the K3 token cache's per-call numbers, and a digest of
  the prompt token ids (``VLLM_FELOG_DIGEST``, default 1).
- ``G`` (both processes): one per garbage collection (start / stop, generation,
  collected, uncollectable).
- ``M`` (both processes, every ``VLLM_FELOG_SAMPLE_S`` seconds, default 5):
  RSS, ``gc.get_count()``, collections per generation, thread count.

``FelogMiddleware`` (``--middleware vllm.v1.metrics.felog.FelogMiddleware``)
stamps the HTTP receive start and the end of the request body. Without it, a
request's record starts at the chat handler.
"""

import contextvars
import ctypes
import gc
import hashlib
import os
import resource
import threading
import time
from array import array
from typing import Any

from vllm.v1.metrics.iterlog import _Writer

FELOG_DIR = os.environ.get("VLLM_FELOG_DIR") or None
ON = FELOG_DIR is not None
DIGEST = os.environ.get("VLLM_FELOG_DIGEST", "1") not in ("", "0")

Q_STAMPS = (
    "t_recv0",  # ASGI request start (middleware)
    "t_body1",  # last body chunk received (middleware)
    "t_handler",  # chat handler entry (body parsed and validated)
    "t_k3_0",  # K3 renderer entry
    "t_parse1",  # messages parsed and normalized
    "t_submit",  # render + tokenize submitted to the tokenizer thread
    "t_th0",  # tokenizer thread starts it
    "t_th1",  # tokenizer thread done
    "t_k3_1",  # back on the event loop
    "t_rendered",  # render_chat_request returned (incl. engine-input processing)
    "t_gen",  # engine_client.generate() created
    "t_add0",  # AsyncLLM.add_request entry
    "t_inproc1",  # input processor done
    "t_outproc1",  # output processor registered the request
    "t_sent",  # EngineCore request sent (ZMQ)
)
Q_FIELDS = " ".join(
    (
        "rid",
        "session",
        *Q_STAMPS,
        "body_bytes n_msgs prompt_tokens th_cpu_ms th_cpu0 th_cpu1 th_nvcsw "
        "th_nivcsw loop_cpu0 loop_cpu1 tc_runs tc_hit_runs tc_hit_tokens "
        "tc_miss_tokens tc_s tc_bytes tc_entries tc_evictions ids_sha",
    )
)
FIELDS = {
    "Q": Q_FIELDS,
    "G": "t0 t1 gen collected uncollectable",
    "M": "t rss_bytes gc0 gc1 gc2 coll0 coll1 coll2 n_threads",
}

_rec: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "vllm_felog_rec", default=None
)
_writer: _Writer | None = None
_writer_lock = threading.Lock()

try:
    _sched_getcpu = ctypes.CDLL(None, use_errno=True).sched_getcpu
    _sched_getcpu.restype = ctypes.c_int
except (OSError, AttributeError):  # pragma: no cover
    _sched_getcpu = None


def getcpu() -> int:
    return _sched_getcpu() if _sched_getcpu is not None else -1


def _rss_bytes() -> int:
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except OSError:
        return -1


def install_process_hooks(kind: str) -> _Writer:
    """Create this process's writer, GC callback and RSS sampler (once)."""
    global _writer
    with _writer_lock:
        if _writer is not None:
            return _writer
        try:
            affinity = sorted(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            affinity = []
        meta: dict[str, Any] = {
            "cpu_count": os.cpu_count(),
            "affinity_n": len(affinity),
            "affinity": affinity if len(affinity) <= 64 else affinity[:64],
            "gc_threshold": gc.get_threshold(),
            "k3_env": {
                k: v
                for k, v in os.environ.items()
                if k.startswith(("VLLM_K3_", "VLLM_FELOG", "VLLM_ITERLOG"))
            },
        }
        w = _Writer(kind, FIELDS, meta, out_dir=FELOG_DIR)
        put = w.put
        starts: dict[int, float] = {}

        def on_gc(phase: str, info: dict) -> None:
            tid = threading.get_ident()
            if phase == "start":
                starts[tid] = time.perf_counter()
            else:
                t0 = starts.pop(tid, None)
                put(
                    (
                        "G",
                        t0,
                        time.perf_counter(),
                        info.get("generation"),
                        info.get("collected"),
                        info.get("uncollectable"),
                    )
                )

        gc.callbacks.append(on_gc)
        period = float(os.environ.get("VLLM_FELOG_SAMPLE_S", "5"))

        def sample() -> None:
            while True:
                c = gc.get_count()
                s = gc.get_stats()
                put(
                    (
                        "M",
                        time.perf_counter(),
                        _rss_bytes(),
                        *c,
                        *(g["collections"] for g in s[:3]),
                        threading.active_count(),
                    )
                )
                time.sleep(period)

        threading.Thread(target=sample, name=f"felog-{kind}", daemon=True).start()
        _writer = w
        return w


def record() -> dict:
    """This request's record, created on first use."""
    r = _rec.get()
    if r is None:
        r = {}
        _rec.set(r)
        if _writer is None:
            install_process_hooks("api")
    return r


def stamp(name: str) -> None:
    record()[name] = time.perf_counter()


def thread_begin(r: dict) -> tuple:
    """Tokenizer-thread side: call at the start of the offloaded work."""
    r["t_th0"] = time.perf_counter()
    ru = resource.getrusage(resource.RUSAGE_THREAD)
    r["th_cpu0"] = getcpu()
    return time.thread_time(), ru.ru_nvcsw, ru.ru_nivcsw


def thread_end(r: dict, begin: tuple, accel=None) -> None:
    ru = resource.getrusage(resource.RUSAGE_THREAD)
    r["th_cpu_ms"] = (time.thread_time() - begin[0]) * 1e3
    r["th_nvcsw"] = ru.ru_nvcsw - begin[1]
    r["th_nivcsw"] = ru.ru_nivcsw - begin[2]
    r["th_cpu1"] = getcpu()
    if accel is not None:
        r["tc"] = accel.last_call()
    r["t_th1"] = time.perf_counter()


def emit(rid: str, prompt_ids: list[int] | None) -> None:
    """After the request was sent to EngineCore. With ``VLLM_FELOG_DIGEST=1``
    (default) the record also carries a SHA-256 prefix of the prompt token
    ids as int32, computed after the send stamp (A/B identity checks)."""
    r = record()
    r["t_sent"] = time.perf_counter()
    r["loop_cpu1"] = getcpu()
    tc = r.get("tc") or (None,) * 8
    w = _writer
    if w is None:
        return
    ids_sha = None
    if prompt_ids is not None and DIGEST:
        ids_sha = hashlib.sha256(array("i", prompt_ids).tobytes()).hexdigest()[:16]
    w.put(
        (
            "Q",
            rid,
            r.get("session"),
            *(r.get(k) for k in Q_STAMPS),
            r.get("body_bytes"),
            r.get("n_msgs"),
            len(prompt_ids) if prompt_ids is not None else r.get("prompt_tokens"),
            r.get("th_cpu_ms"),
            r.get("th_cpu0"),
            r.get("th_cpu1"),
            r.get("th_nvcsw"),
            r.get("th_nivcsw"),
            r.get("loop_cpu0"),
            r.get("loop_cpu1"),
            *tc,
            ids_sha,
        )
    )


class FelogMiddleware:
    """Pure ASGI middleware: HTTP receive start and end of body per request."""

    def __init__(self, app) -> None:
        self.app = app
        if ON:
            install_process_hooks("api")

    async def __call__(self, scope, receive, send) -> None:
        if not ON or scope["type"] != "http" or scope.get("method") != "POST":
            await self.app(scope, receive, send)
            return
        r = {"t_recv0": time.perf_counter(), "body_bytes": 0}
        token = _rec.set(r)

        async def recv():
            msg = await receive()
            if msg["type"] == "http.request":
                r["body_bytes"] += len(msg.get("body", b""))
                if not msg.get("more_body", False):
                    r["t_body1"] = time.perf_counter()
            return msg

        try:
            await self.app(scope, recv, send)
        finally:
            _rec.reset(token)
