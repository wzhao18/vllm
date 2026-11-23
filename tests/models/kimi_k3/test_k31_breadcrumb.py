# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 progress breadcrumbs (vllm/models/kimi_k3/nvidia/breadcrumb.py): the GPU word follows the stream,
a stuck stream is reported with where it stopped, and nothing is recorded into a CUDA graph."""

import os
import subprocess
import sys
import textwrap
import time

import pytest
import torch


def test_decode_names_size_layer_op():
    from vllm.models.kimi_k3.nvidia import breadcrumb as B

    code = (5 << 24) | (61 << 8) | B.OP_MOE_IN
    assert B.decode(code) == "size#4/layer61/megamoe_in"
    assert B.decode(B.OP_FWD_BEGIN) == "size#-/layer0/forward_begin"


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@cuda
def test_crumb_reaches_the_gpu_word_in_stream_order():
    from vllm.models.kimi_k3.nvidia import breadcrumb as B

    B.crumb(B.OP_LAYER, 7)
    torch.cuda.synchronize()
    c = B._get()
    assert c.ok
    assert int(c.word[0]) & 0xFFFFFFFF == c.host_code == ((7 << 8) | B.OP_LAYER)


@cuda
def test_first_use_under_a_cuda_default_device():
    # Model construction runs under torch.device("cuda"); a crumb registered there must still get a pinned host word.
    from vllm.models.kimi_k3.nvidia import breadcrumb as B

    with torch.device("cuda"):
        c = B._Crumbs()
    assert c.word.device.type == "cpu" and c.word.is_pinned()
    c.ok = False  # stop the extra watcher from reporting


@cuda
def test_no_crumb_is_recorded_into_a_cuda_graph():
    from vllm.models.kimi_k3.nvidia import breadcrumb as B

    B.crumb(B.OP_LAYER, 1)
    torch.cuda.synchronize()
    x = torch.zeros(4, device="cuda")
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        x.add_(1)  # warm up off the capture
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        B.crumb(B.OP_LAYER, 99)
        x.add_(1)
    g.replay()
    torch.cuda.synchronize()
    assert int(B._get().word[0]) & 0xFFFF == (1 << 8) | B.OP_LAYER  # the replay wrote nothing


@cuda
def test_stuck_stream_is_reported_with_its_position():
    code = textwrap.dedent(
        """
        import time, torch
        from vllm.models.kimi_k3.nvidia import breadcrumb as B
        B.STALL_S = 2.0
        B.crumb(B.OP_FWD_BEGIN, 0)
        B.crumb(B.OP_LAYER, 3)
        torch.cuda.synchronize()
        torch.cuda._sleep(60_000_000_000)  # the stream is stuck in this kernel
        B.crumb(B.OP_MOE_IN, 4)            # issued by the host, not yet reached by the GPU
        time.sleep(30)
        """
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    text = r.stdout + r.stderr
    assert "K31_BREADCRUMB gpu_stuck" in text, text[-2000:]
    assert "gpu_at=size#-/layer3/layer host_at=size#-/layer4/megamoe_in" in text, text[-2000:]


@cuda
def test_stamps_land_in_an_eager_break_of_a_breakable_capture():
    """DCRIT r46 (a): the no-op keys on the stream actually capturing, so an eager break inside a
    PIECEWISE (breakable) capture forward still stamps; the 90758 stall sat in such a break."""
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
    from vllm.models.kimi_k3.nvidia import breadcrumb as B

    x = torch.zeros(4, device="cuda")
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        x.add_(1)
        torch.cuda.synchronize()
        with BreakableCUDAGraphCapture() as cap:
            x.add_(1)  # captured segment
            B.crumb(B.OP_LAYER, 5)  # captured: no stamp
            cap.add_eager(lambda: B.crumb(B.OP_MOE_IN, 42))  # eager break: stamps
            x.add_(1)  # captured segment
    torch.cuda.synchronize()
    assert int(B._get().word[0]) & 0xFFFFFFFF == (42 << 8) | B.OP_MOE_IN


def _run(code: str) -> str:
    r = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True, timeout=300)
    return r.stdout + r.stderr


@cuda
def test_host_stuck_inside_a_forward_is_reported_with_its_stack():
    text = _run(
        """
        import time, torch
        from vllm.models.kimi_k3.nvidia import breadcrumb as B
        B.STALL_S = 2.0
        def blocked_in_host():
            time.sleep(15)  # the host stops issuing inside the forward; the GPU has caught up
        B.crumb(B.OP_FWD_BEGIN, 0)
        B.crumb(B.OP_LAYER, 9)
        torch.cuda.synchronize()
        blocked_in_host()
        """
    )
    assert "K31_BREADCRUMB host_stuck" in text and "at=size#-/layer9/layer" in text, text[-2000:]
    assert "blocked_in_host" in text, text[-2000:]  # the main thread's stack names where the host waits


@cuda
def test_stall_inside_a_capture_eager_break_is_reported():
    text = _run(
        """
        import time, torch
        from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
        from vllm.models.kimi_k3.nvidia import breadcrumb as B
        B.STALL_S = 2.0
        B.capture_size(4)
        x = torch.zeros(4, device="cuda")
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            x.add_(1); torch.cuda.synchronize()
            with BreakableCUDAGraphCapture() as cap:
                x.add_(1)
                def eager_break():
                    B.crumb(B.OP_LAYER, 61)
                    torch.cuda._sleep(60_000_000_000)  # the break's GPU work never finishes in time
                    B.crumb(B.OP_MOE_IN, 61)
                cap.add_eager(eager_break)
                time.sleep(30)
        """
    )
    assert "K31_BREADCRUMB gpu_stuck" in text, text[-2000:]
    assert "gpu_at=size#4/layer61/layer host_at=size#4/layer61/megamoe_in" in text, text[-2000:]


@cuda
def test_process_exits_cleanly_after_crumbs():
    # The watcher thread must be stopped before interpreter teardown: a daemon thread inside a torch
    # call at exit aborted processes ("terminate called without an active exception").
    r = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(
            """
            import torch
            from vllm.models.kimi_k3.nvidia import breadcrumb as B
            B.crumb(B.OP_FWD_BEGIN, 0)
            B.crumb(B.OP_LAYER, 1)
            torch.cuda.synchronize()
            assert B._get().gpu_word() == B._get().host_code
            """
        )],
        capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, (r.returncode, (r.stdout + r.stderr)[-2000:])
    assert "terminate called" not in r.stdout + r.stderr


@cuda
def test_sigterm_shutdown_exits_cleanly_after_crumbs():
    # Workers are stopped by SIGTERM with a handler that exits the interpreter (vLLM / srtctl teardown):
    # the watcher must be joined on that path too, with no "terminate called" abort.
    import signal

    p = subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent(
            """
            import signal, sys, time, torch
            from vllm.models.kimi_k3.nvidia import breadcrumb as B
            signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))
            B.crumb(B.OP_FWD_BEGIN, 0)
            B.crumb(B.OP_LAYER, 1)
            torch.cuda.synchronize()
            print("READY", flush=True)
            time.sleep(60)
            """
        )],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    seen = []
    for line in p.stdout:  # warnings may come first (stderr is merged)
        seen.append(line)
        if line.strip().endswith("READY"):
            break
    assert seen and seen[-1].strip().endswith("READY"), "".join(seen)[-2000:]
    time.sleep(6)  # at least one watcher poll in flight
    p.send_signal(signal.SIGTERM)
    out, _ = p.communicate(timeout=60)
    assert p.returncode == 0, (p.returncode, out[-2000:])
    assert "terminate called" not in out


@cuda
def test_capture_loop_end_resyncs_host_and_gpu_words():
    # REGRESS 2026-11-23: a host code the GPU word never reached (lost during capture) left gpu != host into serving
    # and the watcher reported 'gpu_stuck' on healthy workers. The end of the capture loop resyncs the two words.
    from vllm.models.kimi_k3.nvidia import breadcrumb as B

    B.crumb(B.OP_LAYER, 5)
    torch.cuda.synchronize()
    c = B._get()
    c.host_code = (6 << 24) | (92 << 8) | B.OP_FWD_BEGIN  # a host code that never reached the GPU word
    c.in_forward = True
    B.capture_size(None)
    assert c.host_code == c.gpu_word() and not c.in_forward

