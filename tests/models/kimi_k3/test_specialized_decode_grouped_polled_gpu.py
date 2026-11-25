# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 specialized decode, TP16 verify rows on ``RubinGroupedExpertsKernel``
(``experts_runtime.POLLED64``): the routing kernel (``RouteSitu`` ids_box) also writes
the ids into the grouped ids mailbox and the experts poll it, grouping before their
``griddepcontrol.wait``. Against the same chain with the experts reading routing's ids
after the wait (no mailbox), at 16 / 32 / 64 rows on one GPU (the 16 ranks' front
cells pre-landed in a plain front box): the ids, routing weights, plan and the
experts' expanded rows are bitwise equal, the mailbox's ids are routing's, and after
every call the mailbox is re-armed (ids all ones, read count 0), over repeated calls.
Production expert weights and verify routes from the wide-experts test helpers."""

import importlib.util
import os

import pytest
import torch

pytest.importorskip("cutlass")

ROWS = (16, 32, 64)
HERE = os.path.dirname(os.path.abspath(__file__))


def _helpers():
    path = os.path.join(HERE, "test_specialized_decode_wide_experts_gpu.py")
    spec = importlib.util.spec_from_file_location("wide_experts_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def built():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("Rubin (cc 10.7) grouped experts")
    t = _helpers()
    dev = torch.device("cuda", 0)
    torch.accelerator.set_device_index(dev)
    return t, dev, t.production_experts(t.logical_weights(dev), dev)


def _chain(t, dev, built, rows, polled):
    """(ids, weights, plan, expanded, mailbox) after routing + experts, twice over."""
    import cutlass as c

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        prepare_shared_carveout,
        stream,
        to_cute,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe import (
        experts_runtime as er,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe import front as F
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe.experts_rubin64 import (  # noqa: E501
        IDS_COUNT,
    )

    ids_ref, _, latent = t.verify_routes(rows, dev, q=8)
    gen = torch.Generator(device=dev).manual_seed(rows)
    logits = torch.randn(rows, 896, device=dev, generator=gen)
    bias = torch.randn(896, device=dev, generator=gen) * 0.1
    buffers = er.scratch(16, dev)
    if not polled:
        buffers.pop("ids")
    experts = er.PersistentExperts(
        *built["weights"], beta=t.BETA, linear_beta=t.LINEAR_BETA,
        dies=built["dies"], q=8,
    )  # fmt: skip
    expanded = torch.empty(rows * 16, 3584, dtype=torch.bfloat16, device=dev)
    experts.prepare(rows, buffers, expanded, (), True, False, True)
    assert (rows in experts.ids_box) == polled
    opts = dict(rcp=rows <= 56, halves=rows <= 56, sync_copy=rows > 56, early="front")
    if polled:
        opts.update(signal=True, ids_box=True)
    route = F.compile_route(rows, 384, True, (True, True, False), dev.index, True, True,
                            None, **opts)  # fmt: skip
    route = prepare_shared_carveout(route, dev.index, 100)
    box = torch.empty(1, 64, F.WIDTH, dtype=torch.bfloat16, device=dev)
    cells = torch.empty_like(box)
    cells[0, :rows, :3584] = latent
    cells.view(torch.float32)[0, :rows, 1792 : 1792 + 896] = logits
    f32, bf16, i32 = torch.float32, torch.bfloat16, torch.int32
    ids = torch.empty(rows, 16, dtype=i32, device=dev)
    weights = torch.empty(rows, 16, dtype=bf16, device=dev)
    lat = torch.empty(rows, 3584, dtype=bf16, device=dev)
    gate = torch.empty(rows, 768, dtype=f32, device=dev)
    h = torch.empty(rows, 384, dtype=bf16, device=dev)
    lg = torch.empty(rows, 896, dtype=f32, device=dev)
    args = tuple(map(to_cute, (lg, bias, ids, weights, box, lat, gate, h)))
    args += tuple(map(c.Float32, (2.827, 1.0, 1.0)))
    extra = (to_cute(buffers["plan"]),)
    if polled:
        extra += (to_cute(buffers["ids"]),)
    outs = []
    for _ in range(2):
        box.copy_(cells)
        route(*args, *extra, stream(dev))
        seen = buffers["ids"][: rows * 16].clone() if polled else None
        experts.group(ids)
        experts.run(lat, ids)
        torch.cuda.synchronize()
        if polled:
            mailbox = buffers["ids"]
            assert bool((mailbox[: rows * 16] == -1).all()), "ids not re-armed"
            assert int(mailbox[IDS_COUNT]) == 0, "read count not reset"
        outs.append((ids.clone(), weights.clone(), buffers["plan"].clone(),
                     expanded.clone(), seen))  # fmt: skip
    return outs


@pytest.mark.parametrize("rows", ROWS)
def test_grouped_experts_polled_ids_bitwise(built, rows):
    t, dev, weights = built
    ref = _chain(t, dev, weights, rows, polled=False)
    got = _chain(t, dev, weights, rows, polled=True)
    for (i0, w0, p0, e0, _), (i1, w1, p1, e1, seen) in zip(ref, got):
        assert torch.equal(i0, i1) and torch.equal(w0, w1)
        assert torch.equal(p0, p1)
        assert torch.equal(e0.view(torch.int16), e1.view(torch.int16))
        assert torch.equal(seen.view(rows, 16), i1), "mailbox ids != routing's ids"
    # both calls of each arm agree (the re-armed mailbox serves the second call)
    assert torch.equal(got[0][3].view(torch.int16), got[1][3].view(torch.int16))
