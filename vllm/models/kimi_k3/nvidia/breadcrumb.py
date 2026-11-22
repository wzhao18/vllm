# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stream-ordered progress breadcrumbs (MEGAMOEFIX, DCRIT r44/r45): self-diagnosis of startup stalls.

The host stamps a 4-byte code (capture size, layer, op) into a pinned word with a stream memory
operation (``cuStreamWriteValue32``) at each instrumented point. The GPU writes the code when the
stream reaches that point, and the host remembers the last code it issued. A watcher thread compares
the two words:

* GPU behind (the GPU's code lags the host's and does not move for ``STALL_S``):
  ``K31_BREADCRUMB gpu_stuck gpu_at=<where> host_at=<where>``, i.e. the GPU is stuck in the work
  enqueued after ``gpu_at``.
* Host stuck inside a forward (no new code issued for ``STALL_S`` and the GPU has caught up):
  ``K31_BREADCRUMB host_stuck at=<where>`` plus the main thread's Python stack.

A silent rank (never traps, a1234772 DP12-type) reports itself this way.

Codes are written in EAGER execution only. While a CUDA graph is being captured, ``crumb`` is a no-op,
so graph replays carry no extra nodes (decode step cost unchanged). Every recorded startup stall
(a1234772: the eager warmup forward of a capture size; 90758: an eager break) is eager work.
"""

from __future__ import annotations

import sys
import threading
import time
import traceback

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

STALL_S = 60.0  # REPRO ONLY: report before the 120 s repro guards trap the peers
OP_FWD_BEGIN, OP_FWD_END, OP_LAYER, OP_MOE_IN, OP_MOE_OUT, OP_CAPTURE_SIZE, OP_ATTN_OUT = 1, 2, 3, 4, 5, 6, 7
_OPS = {1: "forward_begin", 2: "forward_end", 3: "layer", 4: "megamoe_in", 5: "megamoe_out", 6: "capture_size", 7: "attn_out"}


def decode(code: int) -> str:
    code &= 0xFFFFFFFF
    size, layer, op = code >> 24, (code >> 8) & 0xFFFF, code & 0xFF
    return f"size#{size - 1 if size else '-'}/layer{layer}/{_OPS.get(op, op)}"


class _Crumbs:
    def __init__(self) -> None:
        self.word = torch.zeros(4, dtype=torch.int32, pin_memory=True)  # [0] GPU-side code
        self.addr = self.word.data_ptr()
        self.host_code = 0
        self.host_t = time.monotonic()
        self.in_forward = False
        self.size_idx = 0  # 1 + capture-size index while capturing sizes, else 0
        self.layer = 0
        self.ok = True
        self.streams: dict[int, tuple[str, object]] = {}  # handle -> (name, stream): queried in the stall report
        self._main = threading.main_thread().ident
        self._thread = threading.Thread(target=self._watch, name="k31-breadcrumbs", daemon=True)
        self._thread.start()

    def write(self, op: int) -> None:
        if not self.ok or torch.cuda.is_current_stream_capturing():
            return
        code = ((self.size_idx & 0xFF) << 24) | ((self.layer & 0xFFFF) << 8) | (op & 0xFF)
        try:
            from cuda.bindings import driver

            cur = torch.cuda.current_stream()
            if cur.cuda_stream not in self.streams:
                self.streams[cur.cuda_stream] = (f"crumb_stream_{len(self.streams)}", cur)
            (err,) = driver.cuStreamWriteValue32(
                driver.CUstream(cur.cuda_stream),
                driver.CUdeviceptr(self.addr), code, 0,
            )
            if err != driver.CUresult.CUDA_SUCCESS:
                raise RuntimeError(err)
        except Exception as e:  # noqa: BLE001 - diagnostics: never break the forward
            self.ok = False
            logger.warning("Kimi K3 breadcrumbs disabled: cuStreamWriteValue32 failed (%s)", e)
            return
        self.host_code, self.host_t = code, time.monotonic()

    def _watch(self) -> None:
        gpu_prev, gpu_t, reported = None, time.monotonic(), None
        while True:
            time.sleep(5.0)
            now = time.monotonic()
            gpu = int(self.word[0]) & 0xFFFFFFFF
            if gpu != gpu_prev:
                gpu_prev, gpu_t = gpu, now
            if gpu != self.host_code and now - gpu_t >= STALL_S:
                key = ("gpu", gpu, self.host_code)
                if key != reported:
                    reported = key
                    self._log(
                        f"K31_BREADCRUMB gpu_stuck {now - gpu_t:.0f}s gpu_at={decode(gpu)} host_at={decode(self.host_code)} "
                        f"gpu_state=[{self._gpu_state()}] streams=[{self._stream_state()}]"
                    )
            elif self.in_forward and gpu == self.host_code and now - self.host_t >= STALL_S:
                key = ("host", self.host_code)
                if key != reported:
                    reported = key
                    frame = sys._current_frames().get(self._main)
                    stack = "".join(traceback.format_stack(frame)[-25:]) if frame else "?"
                    self._log(
                        f"K31_BREADCRUMB host_stuck {now - self.host_t:.0f}s at={decode(self.host_code)} "
                        f"(or a long host-side compile: see the stack)\n{stack}"
                    )

    def _gpu_state(self) -> str:
        """DCRIT r47: busy (~100% util, high power) = a looping kernel; idle = a stream / event wait."""
        try:
            import subprocess

            uuid = str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)
            uuid = uuid if uuid.startswith("GPU-") else "GPU-" + uuid
            out = subprocess.run(
                ["nvidia-smi", "-i", uuid, "--query-gpu=utilization.gpu,power.draw,clocks.sm",
                 "--format=csv,noheader"], capture_output=True, text=True, timeout=30,
            )
            return (out.stdout or out.stderr).strip().replace("\n", " ")
        except Exception as e:  # noqa: BLE001
            return f"n/a ({e})"

    def _stream_state(self) -> str:
        """cuStreamQuery per known stream: pending = work enqueued and not finished, done = idle."""
        try:
            from cuda.bindings import driver
        except Exception:  # noqa: BLE001
            return "n/a"
        parts = []
        for handle, (name, _) in list(self.streams.items()):
            (err,) = driver.cuStreamQuery(driver.CUstream(handle))
            state = {driver.CUresult.CUDA_SUCCESS: "done", driver.CUresult.CUDA_ERROR_NOT_READY: "pending"}.get(err, str(err))
            parts.append(f"{name}:{state}")
        return " ".join(parts)

    @staticmethod
    def _log(line: str) -> None:
        logger.error("%s", line)
        print(line, file=sys.stderr, flush=True)


_crumbs: _Crumbs | None = None
_lock = threading.Lock()


def _get() -> _Crumbs:
    global _crumbs
    if _crumbs is None:
        with _lock:
            if _crumbs is None:
                _crumbs = _Crumbs()
    return _crumbs


def crumb(op: int, layer: int | None = None) -> None:
    c = _get()
    if layer is not None:
        c.layer = layer
    if op == OP_FWD_BEGIN:
        c.in_forward = True
    c.write(op)
    if op == OP_FWD_END:
        c.in_forward = False


def register_stream(name: str, stream) -> None:
    """A side stream to query in the stall report (prefill shared experts, aux, attention aux, ...)."""
    if stream is not None:
        _get().streams[stream.cuda_stream] = (name, stream)


def capture_size(index: int | None) -> None:
    """Index of the capture size being warmed up / captured (None: outside the capture loop)."""
    c = _get()
    c.size_idx = 0 if index is None else min(index + 1, 255)
    if index is not None:
        c.write(OP_CAPTURE_SIZE)
