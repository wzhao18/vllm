# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MoE-front weight touch for the SP decode step (``cutedsl/common/touch.py``).

Each MegaMoE layer's SP reduce-scatter + AttnRes (``SpReduceAttnRes``) is NVLink-
and latency-bound and leaves HBM idle, and the router | latent down GEMM right after
it streams 64 MB of BF16 weights that it only starts reading once the reduce ends.
Before each layer's reduce, the main stream forks a side stream (highest priority)
that TMA-loads the layer's front weights (the packed router | latent down rows, else
the router and latent down weights) into discarded shared memory, so the GEMM reads
them from L2. The side stream rejoins the main stream at the end of the step, so a
captured graph always joins it. No result depends on the touch."""

import cuda.bindings.driver as cuda
import torch

from vllm.models.common.ops.moe_tail_fusion.primitives import to_cute

from ..cutedsl.common.touch import compile_touch, sm_count, touch_table


def front_ranges(mlp):
    """The front GEMM's weights of a stock MoE layer: the packed router | latent
    rows when ``sp_mlp.pack_router_latent`` packed them, else both weights."""
    packed = getattr(mlp, "packed_router_latent", None)
    if packed is not None:
        return [packed]
    return [mlp.gate.weight, mlp.routed_expert_down_proj.weight]


class FrontTouch:
    def __init__(self, mlps, device):
        self.device = device
        _, high = torch.cuda.Stream.priority_range()
        self.stream = torch.cuda.Stream(device=device, priority=high)
        grid = sm_count(device)
        self.layers = {}
        for index, sp_mlp in mlps.items():
            table = touch_table(front_ranges(sp_mlp.mlp), device)
            compiled = compile_touch(table.shape[0], grid, device.index)
            self.layers[index] = (table, to_cute(table, 16), compiled)
        self.forked = False

    def fork(self, index):
        """Touch layer ``index``'s front weights after the work queued so far."""
        entry = self.layers.get(index)
        if entry is None:
            return
        _, views, compiled = entry
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        compiled(views, cuda.CUstream(self.stream.cuda_stream))
        self.forked = True

    def join(self):
        if self.forked:
            torch.cuda.current_stream(self.device).wait_stream(self.stream)
            self.forked = False
