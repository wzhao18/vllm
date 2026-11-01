# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routed experts on one scratch, as one decode instance serving changing batch sizes:
``ExpertsKernel`` rows (B1-7, e.g. one q4 request) and the fast-exchange
``RubinExpertsKernel`` (M8, two q4 requests) must not read each other's stale hidden
images (``PersistentExperts.kernel_workspace``). Adapted from Wei Zhao's
test_k31_experts_rubin.py mixed-batch tests (44b752867f) for a tree without
``RUBIN_ROWS``.

One MoE layer of random MXFP4 shards (TP16), random latent rows, routes drawn per token
without replacement; every call is compared bitwise with the same input on a fresh
scratch."""

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda()
    or torch.cuda.get_device_capability() != (10, 7)
    or torch.cuda.get_device_properties(0).multi_processor_count < 212,
    reason="Rubin (SM107, 212 SMs) routed-experts kernels",
)

E = 896


@pytest.fixture(scope="module")
def layer():
    dev = torch.device("cuda", 0)
    gen = torch.Generator(device=dev)
    gen.manual_seed(11)
    u8 = torch.uint8
    w13 = torch.randint(0, 256, (E, 512, 1792), dtype=u8, device=dev, generator=gen)
    w2 = torch.randint(0, 256, (E, 3584, 128), dtype=u8, device=dev, generator=gen)
    s13 = torch.randint(118, 127, (E, 512, 112), dtype=u8, device=dev, generator=gen)
    s2 = torch.randint(118, 127, (E, 3584, 8), dtype=u8, device=dev, generator=gen)
    return (w13, s13, w2, s2), dev


def routes(rows, seed):
    rng = np.random.default_rng(seed)
    return np.stack([rng.choice(E, 16, replace=False) for _ in range(rows)])


def calls(weights, dev, q, sequence, inputs):
    """Run ``sequence`` (row counts) on one ``PersistentExperts`` and one scratch."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        stream,
        to_cute,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe import (
        experts_runtime as er,
    )

    buffers = er.scratch(16, dev)
    experts = er.PersistentExperts(*weights, q=q)
    outs = {}
    for rows in sorted(set(sequence)):
        outs[rows] = torch.empty(rows * 16, 3584, dtype=torch.bfloat16, device=dev)
        experts.prepare(rows, buffers, outs[rows], (), planned=True)
    seen = dict.fromkeys(outs, 0)
    results = []
    for rows in sequence:
        x, ids = inputs[rows][seen[rows] % len(inputs[rows])]
        seen[rows] += 1
        writer = None
        if rows == 8 and 8 in experts.rubin:
            writer = er.compile_image_plan(8, dev.index)
        elif rows > 8:
            writer = er.compile_plan(rows, dev.index)
        if writer is not None:
            writer(to_cute(x, 16), to_cute(buffers["plan"], 16), stream(dev))
        if rows == 8 and er.polled(16, 8, dev):  # the routing kernel's mailbox
            box = torch.full((er.ROUTE_WORDS,), -1, dtype=torch.int32, device=dev)
            box[:128] = ids.reshape(-1)
            box[128] = 15  # every copy CTA's image part written (2 x 8 - 1)
            er.route_mailbox(buffers["workspace"]).copy_(box)
        experts.run(x, ids)
        torch.cuda.synchronize()
        results.append(outs[rows].clone())
    return results


@pytest.mark.parametrize("others", [(4,), (2, 6), (4, 16, 64)])
def test_experts_kernel_rows_and_fast_rubin_one_scratch_bitwise(layer, others):
    weights, dev = layer
    torch.manual_seed(5)
    inputs = {
        rows: [
            (
                torch.randn(rows, 3584, dtype=torch.bfloat16, device=dev),
                torch.tensor(routes(rows, 1000 * rows + i), dtype=torch.int32,
                             device=dev),
            )
            for i in range(3)
        ]
        for rows in (8, *others)
    }  # fmt: skip
    sequence = [8, 8]
    for run in (1, 2, 3, 1):
        sequence += [others[i % len(others)] for i in range(run)] + [8]
    sequence += [others[0], 8, others[0], 8, 8, others[-1], others[-1], 8]
    fresh = {
        (rows, i): calls(weights, dev, 4, [rows], {rows: [inputs[rows][i]]})[0]
        for rows in inputs
        for i in range(3)
    }
    got = calls(weights, dev, 4, sequence, inputs)
    seen = dict.fromkeys(inputs, 0)
    for k, rows in enumerate(sequence):
        ref = fresh[(rows, seen[rows] % 3)]
        seen[rows] += 1
        bad = (got[k].view(torch.int16) != ref.view(torch.int16)).any(1).sum()
        assert bad.item() == 0, (k, rows, bad.item())
