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

import atexit
import ctypes
import sys
import threading
import time
import traceback

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

STALL_S = 120.0
OP_FWD_BEGIN, OP_FWD_END, OP_LAYER, OP_MOE_IN, OP_MOE_OUT, OP_CAPTURE_SIZE = 1, 2, 3, 4, 5, 6
_OPS = {1: "forward_begin", 2: "forward_end", 3: "layer", 4: "megamoe_in", 5: "megamoe_out", 6: "capture_size"}


def decode(code: int) -> str:
    code &= 0xFFFFFFFF
    size, layer, op = code >> 24, (code >> 8) & 0xFFFF, code & 0xFF
    return f"size#{size - 1 if size else '-'}/layer{layer}/{_OPS.get(op, op)}"


class _Crumbs:
    def __init__(self) -> None:
        # Explicit device: the first crumb can come from model construction, under a CUDA default device.
        self.word = torch.zeros(4, dtype=torch.int32, device="cpu", pin_memory=True)  # [0] GPU-side code
        self.addr = self.word.data_ptr()
        self.host_code = 0
        self.host_t = time.monotonic()
        self.in_forward = False
        self.size_idx = 0  # 1 + capture-size index while capturing sizes, else 0
        self.layer = 0
        self.ok = True
        self._main = threading.main_thread().ident
        # The watcher never touches torch (the word is read through ctypes) and is stopped and joined at
        # exit: a daemon thread inside a torch call during interpreter teardown aborted the process
        # ("terminate called without an active exception", MegaMoE bench at exit, 2026-11-22).
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._watch, name="k31-breadcrumbs", daemon=True)
        self._thread.start()
        atexit.register(self.close)

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=10.0)

    def gpu_word(self) -> int:
        """Last code the GPU stream wrote (pinned host word, plain load)."""
        return ctypes.c_uint32.from_address(self.addr).value

    def write(self, op: int) -> None:
        if not self.ok or torch.cuda.is_current_stream_capturing():
            return
        code = ((self.size_idx & 0xFF) << 24) | ((self.layer & 0xFFFF) << 8) | (op & 0xFF)
        try:
            from cuda.bindings import driver

            (err,) = driver.cuStreamWriteValue32(
                driver.CUstream(torch.cuda.current_stream().cuda_stream),
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
        while not self._stop.wait(5.0):
            now = time.monotonic()
            gpu = self.gpu_word()
            if gpu != gpu_prev:
                gpu_prev, gpu_t = gpu, now
            if gpu != self.host_code and now - gpu_t >= STALL_S:
                key = ("gpu", gpu, self.host_code)
                if key != reported:
                    reported = key
                    self._log(f"K31_BREADCRUMB gpu_stuck {now - gpu_t:.0f}s gpu_at={decode(gpu)} host_at={decode(self.host_code)}")
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


def capture_size(index: int | None) -> None:
    """Index of the capture size being warmed up / captured (None: outside the capture loop). At the end of the
    capture loop the host and GPU words are resynchronized, so a crumb that did not reach the word during capture
    cannot leave a stale gap behind into serving."""
    c = _get()
    c.size_idx = 0 if index is None else min(index + 1, 255)
    if index is not None:
        c.write(OP_CAPTURE_SIZE)
    elif c.ok and not torch.cuda.is_current_stream_capturing():
        torch.cuda.synchronize()
        c.host_code, c.host_t = c.gpu_word(), time.monotonic()
        c.in_forward = False
