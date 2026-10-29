# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-iteration JSONL log of the engine loop and of worker rank 0 (off by default).

Set ``VLLM_ITERLOG_DIR=<dir>`` to turn it on. Each process writes one file:
``engine-<host>-<pid>.jsonl`` (EngineCore: one record per scheduled batch, request
lifecycle, idle gaps) and ``worker-r0-<host>-<pid>.jsonl`` (global rank 0: host
phase times and CUDA-event GPU times per executed batch).

``VLLM_ITERLOG_GPU`` (worker): 2 (default) times six points per step with CUDA
events, 1 only step begin / end, 0 none (host times only).

The hot path only appends a tuple to a deque. A daemon thread encodes and writes
the records every 0.25 s and yields the GIL every 16 records. Every line is a
JSON array whose first element is the record type; ``H`` lines name the fields
of each type, ``A`` lines anchor ``time.perf_counter()`` (CLOCK_MONOTONIC, shared
by all processes of a node) to wall time.
"""

import atexit
import json
import os
import socket
import threading
import time
from collections import deque
from typing import Any

ITERLOG_DIR = os.environ.get("VLLM_ITERLOG_DIR") or None

ENGINE_FIELDS = {
    # one per scheduled batch, written when its output has been processed
    "S": (
        "k t_sched0 t_sched1 t_disp n_running n_waiting n_reqs n_tokens "
        "n_decode n_decode_tokens n_prefill n_prefill_tokens n_preempted "
        "kv_usage prefills t_result t_update n_out_tokens n_finished"
    ),
    # prefills: [[req_idx, new_tokens, computed_before, prompt_tokens], ...]
    # (computed_before of a request's first chunk = its prefix-cache hit)
    "RA": "idx req_id session_id t_add arrival_wall prompt_tokens max_tokens",
    "RT": "idx t k",  # first output token processed by the engine
    "RF": "idx t k finish_reason n_out_tokens",
    "RX": "idx t",  # aborted
    "I": "t_idle0 t_idle1",  # engine waiting for work (no requests)
}
WORKER_FIELDS = {
    "W": (
        "seq h_entry h_begin h_fwd0 h_fwd1 h_samp0 h_draft0 h_end n_reqs n_tokens "
        "n_tokens_padded cg_mode has_prefill g_prep g_fwd g_mid g_samp g_draft "
        "g_gap_prev"
    ),
}


class _Writer:
    """Buffered JSONL writer: ``put`` is the only hot-path call."""

    def __init__(
        self,
        kind: str,
        fields: dict[str, str],
        meta: dict[str, Any],
        out_dir: str | None = None,
    ):
        out_dir = out_dir or ITERLOG_DIR
        assert out_dir is not None
        os.makedirs(out_dir, exist_ok=True)
        host = socket.gethostname().split(".")[0]
        self.path = os.path.join(out_dir, f"{kind}-{host}-{os.getpid()}.jsonl")
        self._q: deque = deque()
        self._f = open(self.path, "w", buffering=1 << 20)  # noqa: SIM115
        self._enc = json.JSONEncoder(separators=(",", ":")).encode
        self._f.write(self._enc(["H", {"kind": kind, "fields": fields, **meta}]))
        self._f.write("\n")
        self._anchor()
        self._f.flush()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name=f"iterlog-{kind}", daemon=True
        )
        self._thread.start()
        atexit.register(self.close)

    def put(self, rec: tuple) -> None:
        self._q.append(rec)

    def _anchor(self) -> None:
        self._f.write(self._enc(["A", time.perf_counter(), time.time()]))
        self._f.write("\n")

    def _drain(self) -> None:
        q, enc, lines = self._q, self._enc, []
        while q:
            lines.append(enc(q.popleft()))
            if len(lines) == 16:
                self._f.write("\n".join(lines))
                self._f.write("\n")
                lines = []
                time.sleep(0)  # let the engine / worker loop take the GIL
        if lines:
            self._f.write("\n".join(lines))
            self._f.write("\n")

    def _run(self) -> None:
        last_anchor = time.perf_counter()
        while not self._stop.wait(0.25):
            self._drain()
            if time.perf_counter() - last_anchor > 30.0:
                self._anchor()
                last_anchor = time.perf_counter()
            self._f.flush()

    def close(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        self._thread.join(timeout=5)
        self._drain()
        self._anchor()
        self._f.flush()
        self._f.close()


class EngineIterLog:
    """EngineCore side: batches, request lifecycle and idle gaps."""

    def __init__(self, meta: dict[str, Any]):
        self._w = _Writer("engine", ENGINE_FIELDS, meta)
        self._put = self._w.put
        self._k = 0
        self._pending: deque = deque()
        # request_id -> [idx, n_out_tokens]
        self._reqs: dict[str, list[int]] = {}
        self._next_idx = 0

    @classmethod
    def maybe_create(cls, vllm_config) -> "EngineIterLog | None":
        if ITERLOG_DIR is None:
            return None
        spec = vllm_config.speculative_config
        meta = {
            "max_num_seqs": vllm_config.scheduler_config.max_num_seqs,
            "max_num_batched_tokens": (
                vllm_config.scheduler_config.max_num_batched_tokens
            ),
            "async_scheduling": vllm_config.scheduler_config.async_scheduling,
            "num_spec_tokens": spec.num_speculative_tokens if spec else 0,
            "tp": vllm_config.parallel_config.tensor_parallel_size,
            "dcp": vllm_config.parallel_config.decode_context_parallel_size,
        }
        return cls(meta)

    def add_request(self, request) -> None:
        idx = self._next_idx
        self._next_idx = idx + 1
        self._reqs[request.request_id] = [idx, 0]
        self._put(
            (
                "RA",
                idx,
                request.request_id,
                request.session_id,
                time.perf_counter(),
                request.arrival_time,
                request.num_prompt_tokens,
                request.max_tokens,
            )
        )

    def abort(self, request_ids) -> None:
        t = time.perf_counter()
        for rid in request_ids:
            r = self._reqs.pop(rid, None)
            if r is not None:
                self._put(("RX", r[0], t))

    def idle(self, t0: float, t1: float) -> None:
        self._put(("I", t0, t1))

    def on_schedule(self, so, t0: float, t1: float, t2: float, scheduler) -> None:
        """After ``schedule()`` and the dispatch of its batch."""
        requests = scheduler.requests
        reqs = self._reqs
        n_dec = n_dec_tok = n_pf = n_pf_tok = 0
        prefills = None
        for rid, n in so.num_scheduled_tokens.items():
            r = requests.get(rid)
            if r is None:
                continue
            before = r.num_computed_tokens - n
            if before < r.num_prompt_tokens:
                n_pf += 1
                n_pf_tok += n
                ri = reqs.get(rid)
                if prefills is None:
                    prefills = []
                prefills.append((ri[0] if ri else -1, n, before, r.num_prompt_tokens))
            else:
                n_dec += 1
                n_dec_tok += n
        preempted = so.preempted_req_ids
        self._pending.append(
            [
                so,
                self._k,
                t0,
                t1,
                t2,
                len(scheduler.running),
                len(scheduler.waiting),
                len(so.num_scheduled_tokens),
                so.total_num_scheduled_tokens,
                n_dec,
                n_dec_tok,
                n_pf,
                n_pf_tok,
                len(preempted) if preempted else 0,
                scheduler.kv_cache_manager.usage,
                prefills,
            ]
        )
        self._k += 1

    def on_update(self, so, t3: float, t4: float, outputs) -> None:
        """After ``update_from_output()`` of a batch."""
        pending = self._pending
        if not pending:
            return
        rec = pending.popleft()
        if rec[0] is not so:  # out of order (not expected): find it
            pending.appendleft(rec)
            for i, cand in enumerate(pending):
                if cand[0] is so:
                    rec = cand
                    del pending[i]
                    break
            else:
                return
        k = rec[1]
        n_out = n_fin = 0
        reqs, put = self._reqs, self._put
        for eco in outputs.values():
            for o in eco.outputs:
                nt = len(o.new_token_ids)
                n_out += nt
                r = reqs.get(o.request_id)
                if r is None:
                    continue
                if nt and r[1] == 0:
                    put(("RT", r[0], t4, k))
                r[1] += nt
                if o.finish_reason is not None:
                    put(("RF", r[0], t4, k, int(o.finish_reason), r[1]))
                    del reqs[o.request_id]
                    n_fin += 1
        rec[0] = "S"
        rec.extend((t3, t4, n_out, n_fin))
        put(tuple(rec))


class WorkerIterLog:
    """Worker side (global rank 0): host phase times and GPU times per batch."""

    RING = 64

    def __init__(self, level: int):
        import torch

        self._torch = torch
        self._level = level
        self._w = _Writer("worker-r0", WORKER_FIELDS, {"gpu_level": level})
        self._put = self._w.put
        n_ev = 6 if level >= 2 else (2 if level == 1 else 0)
        self._ring = [
            [torch.cuda.Event(enable_timing=True) for _ in range(n_ev)]
            for _ in range(self.RING)
        ]
        self._pos = 0
        self._seq = 0
        self._cur: list | None = None
        self._inflight: deque = deque()
        self._last_end = None

    @classmethod
    def maybe_create(cls) -> "WorkerIterLog | None":
        if ITERLOG_DIR is None:
            return None
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return None
        return cls(int(os.environ.get("VLLM_ITERLOG_GPU", "2")))

    def begin(self, h_entry: float, meta: tuple) -> None:
        """execute_model, once the batch is known (``meta``: n_reqs, n_tokens,
        n_tokens_padded, cg_mode, has_prefill)."""
        if self._inflight:
            self._resolve()
        ev = None
        if self._level and len(self._inflight) < self.RING - 2:
            ev = self._ring[self._pos]
            self._pos = (self._pos + 1) % self.RING
            ev[0].record()
        h = [h_entry, time.perf_counter(), 0.0, 0.0, 0.0, 0.0, 0.0]
        self._cur = [self._seq, ev, h, meta]
        self._seq += 1

    def mark(self, i: int) -> None:
        """1: before the forward, 2: after it, 3: sample_tokens entry,
        4: before the drafter."""
        cur = self._cur
        if cur is None:
            return
        cur[2][i + 1] = time.perf_counter()
        ev = cur[1]
        if ev is not None and self._level >= 2:
            ev[i].record()

    def end(self) -> None:
        cur = self._cur
        if cur is None:
            return
        self._cur = None
        ev = cur[1]
        if ev is not None:
            ev[-1].record()
        cur[2][6] = time.perf_counter()
        if ev is None:
            self._emit(cur, (-1.0,) * 6)
        else:
            self._inflight.append(cur)

    def _emit(self, cur: list, gpu: tuple) -> None:
        self._put(("W", cur[0], *cur[2], *cur[3], *gpu))

    def _resolve(self) -> None:
        inflight = self._inflight
        while inflight:
            cur = inflight[0]
            ev = cur[1]
            if not ev[-1].query():
                return
            inflight.popleft()
            last = self._last_end
            gap = last.elapsed_time(ev[0]) if last is not None else -1.0
            if len(ev) == 6:
                gpu = (
                    ev[0].elapsed_time(ev[1]),
                    ev[1].elapsed_time(ev[2]),
                    ev[2].elapsed_time(ev[3]),
                    ev[3].elapsed_time(ev[4]),
                    ev[4].elapsed_time(ev[5]),
                    gap,
                )
            else:
                gpu = (-1.0, ev[0].elapsed_time(ev[1]), -1.0, -1.0, -1.0, gap)
            self._last_end = ev[-1]
            self._emit(cur, gpu)
