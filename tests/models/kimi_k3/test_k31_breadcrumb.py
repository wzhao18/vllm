# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 progress breadcrumbs (vllm/models/kimi_k3/nvidia/breadcrumb.py): the GPU word follows the stream,
a stuck stream is reported with where it stopped, and nothing is recorded into a CUDA graph."""

import os
import subprocess
import sys
import textwrap

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
        torch.cuda._sleep(20_000_000_000)  # the stream is stuck in this kernel
        B.crumb(B.OP_MOE_IN, 4)            # issued by the host, not yet reached by the GPU
        time.sleep(12)
        """
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    text = r.stdout + r.stderr
    assert "K31_BREADCRUMB gpu_stuck" in text, text[-2000:]
    assert "gpu_at=size#-/layer3/layer host_at=size#-/layer4/megamoe_in" in text, text[-2000:]
