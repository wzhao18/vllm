# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Ported from the source branch's tests/models/kimi_k3/test_k31_experts_workspace.py for the SM107
# specialized decode (vllm/models/kimi_k3/nvidia/specialized_decode/sm_107).
"""Routed experts on one scratch, as one decode instance serving changing batch sizes:
``ExpertsKernel`` rows (B1-7, e.g. one q4 request) must not read stale hidden images
left by the fast-exchange ``RubinExpertsKernel`` (M8, two q4 requests) or
``RubinGroupedExpertsKernel`` (M12-M64), nor leave stale images to the fast-exchange
M8 (``PersistentExperts.kernel_workspace``). Adapted from Wei Zhao's
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

# The SM107 specialized decode (specialized_decode/sm_107) serves VR200 only.
_SM107_ONLY = pytest.mark.skipif(
    not current_platform.is_device_capability((10, 7)),
    reason="the SM107 specialized decode runs on compute capability 10.7 only",
)
pytestmark = [
    *(pytestmark if isinstance(pytestmark, list) else [pytestmark]),
    _SM107_ONLY,
]

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
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.common.host import (
        stream,
        to_cute,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.moe import (
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


@pytest.mark.parametrize(
    "anchor, others", [(8, (4,)), (8, (2, 6)), (8, (4, 16, 64)), (4, (12, 16))]
)
def test_experts_kernel_rows_beside_rubin_rows_one_scratch_bitwise(
    layer, anchor, others
):
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
        for rows in (anchor, *others)
    }  # fmt: skip
    sequence = [anchor, anchor]
    for run in (1, 2, 3, 1):
        sequence += [others[i % len(others)] for i in range(run)] + [anchor]
    sequence += [others[0], anchor, others[0], anchor, anchor]
    sequence += [others[-1], others[-1], anchor]
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


@pytest.mark.parametrize("q", [4, 8])
def test_decode_prepare_loop_one_scratch(layer, q, monkeypatch):
    """``SpecializedDecode`` prepares every tuned row count (``BATCHES``) of a verify
    instance on one ``PersistentExperts`` and one ``scratch`` (``moe.prepared``):
    ``ExpertsKernel`` rows take their own workspace, the Rubin kernels the shared
    one."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.cutedsl.moe import (
        experts_runtime as er,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.runtime.providers import (
        BATCHES,
    )

    weights, dev = layer
    experts = er.PersistentExperts(*weights, q=q)
    buffers = er.scratch(16, dev)
    shared = buffers["workspace"].data_ptr()
    monkeypatch.setattr(er, "tensor", lambda t: t)  # prepared[b][2]: the torch view
    for b in BATCHES:
        out = torch.empty(b * 16, 3584, dtype=torch.bfloat16, device=dev)
        experts.prepare(b, buffers, out, (), planned=True)
    assert experts.fast_rubin and 8 in experts.rubin
    kernel = buffers["kernel_workspace"].data_ptr()
    assert kernel != shared
    for b in BATCHES:
        on_kernel = b not in experts.rubin and not experts.grouped64(b)
        ws = experts.prepared[b][2]
        assert ws.data_ptr() == (kernel if on_kernel else shared), b
