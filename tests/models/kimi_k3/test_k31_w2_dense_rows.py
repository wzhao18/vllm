# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dense routed w2 rows (``experts_runtime.dense_w2``): the TP16 experts kernels read
the w2 data bytes at the dense pitch (96 B) instead of the parameter's padded rows
(128 B), where ``dense_w2_rows`` admits the layer's layout; the stock MoE reads the
padded parameter, now one shared padded copy refreshed per stock call.

- ``test_dense_w2_rows_layouts``: the capability (TP16 padded rows only; TP8's 192 B
  rows and unpadded intermediates keep the parameter).
- ``test_dense_rows_bitwise``: one MoE layer of random MXFP4 shards whose w2 padding
  holds random bytes (a kernel reading the padding would differ), die-placed as
  ``runtime.decode`` does (``place``, then ``dense_w2`` with the die map); the routed
  experts at every TP16 k7 graph row count (8 x B, B = 1..32: the 8-row Rubin kernel,
  the grouped kernel, the wide engines) and the tuned rows 1-7 (``ExpertsKernel``) on
  the padded parameter and on the dense rows, compared bitwise.
- ``test_tp8_layout_untouched``: TP8's 192 B rows are not admitted and its
  ``PersistentExperts`` builds and prepares as before.
- ``test_dense_rows_stock_path``: two production stock layers (``Mxfp4MoEMethod`` on
  the TRT-LLM-Gen MXFP4 x MXFP8 experts) converted to dense rows share one padded copy;
  after each layer's refresh its stock output equals the padded parameter's bitwise.

Rubin (cc 10.7, 212 SMs) with CuTe DSL."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform

E, H, TOPK, Q = 896, 3584, 16, 8
# TP16 k7 graph row counts (8 x B for B = 1..32) and the tuned rows below 8
# (``providers.BATCHES``: ``ExpertsKernel``)
ROWS = (*range(1, 8), *(8 * b for b in range(1, 33)))

rubin = pytest.mark.skipif(
    not current_platform.is_cuda()
    or torch.cuda.get_device_capability() != (10, 7)
    or torch.cuda.get_device_properties(0).multi_processor_count < 212,
    reason="Rubin (SM107, 212 SMs) routed-experts kernels",
)


def routed_layer(w13, w2, unpadded=192):
    param = lambda t: torch.nn.Parameter(t, requires_grad=False)  # noqa: E731
    config = SimpleNamespace(intermediate_size_per_partition_unpadded=unpadded)
    weights = dict(w13_weight=param(w13), w2_weight=param(w2))
    return SimpleNamespace(**weights, moe_config=config)


@rubin
def test_dense_w2_rows_layouts():
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe import (
        experts_runtime as er,
    )

    dev, u8 = torch.device("cuda", 0), torch.uint8
    empty = lambda *s: torch.empty(*s, dtype=u8, device=dev)  # noqa: E731
    assert er.dense_w2_rows(routed_layer(empty(4, 512, 1792), empty(4, H, 128)))
    # TP8: 384-row intermediate, 192 B w2 rows (no padding)
    tp8 = routed_layer(empty(4, 768, 1792), empty(4, H, 192), unpadded=384)
    assert not er.dense_w2_rows(tp8)
    # a 256-row intermediate (no padding)
    full = routed_layer(empty(4, 512, 1792), empty(4, H, 128), unpadded=256)
    assert not er.dense_w2_rows(full)
    cpu = routed_layer(torch.empty(4, 512, 1792, dtype=u8), torch.empty(4, H, 128))
    assert not er.dense_w2_rows(cpu)


@pytest.fixture(scope="module")
def layers():
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe import (
        experts_runtime as er,
    )

    dev = torch.device("cuda", 0)
    torch.accelerator.set_device_index(dev)
    gen = torch.Generator(device=dev).manual_seed(11)
    u8 = torch.uint8
    rand = lambda s, lo=0, hi=256: torch.randint(  # noqa: E731
        lo, hi, s, dtype=u8, device=dev, generator=gen
    )
    w13, w2 = rand((E, 512, 1792)), rand((E, H, 128))  # w2 padding: random bytes
    s13, s2 = rand((E, 512, 112), 118, 127), rand((E, H, 8), 118, 127)
    padded = routed_layer(w13.clone(), w2.clone())
    dense = routed_layer(w13.clone(), w2.clone())
    dies = er.place(padded)
    assert er.place(dense) == dies
    w2_dense = er.dense_w2(dense, dies)
    assert w2_dense.shape == (E, H, 96)
    assert torch.equal(w2_dense, w2[..., :96])
    # the parameter: the shared padded copy (zero padding until a stock call)
    assert dense.w2_weight.shape == w2.shape and not dense.w2_weight[..., 96:].any()
    arms = {
        "padded": (padded.w13_weight, s13, padded.w2_weight, s2),
        "dense": (dense.w13_weight, s13, w2_dense, s2),
    }
    return arms, tuple(dies), dev


def inputs(rows, dev, seed):
    rng = np.random.default_rng(seed)
    ids = np.stack([rng.choice(E, TOPK, replace=False) for _ in range(rows)])
    gen = torch.Generator(device=dev).manual_seed(seed)
    w = torch.rand(rows, TOPK, device=dev, generator=gen)
    w = (w / w.sum(1, keepdim=True)).bfloat16()
    x = (torch.randn(rows, H, device=dev, generator=gen) * 0.5).bfloat16()
    return x, torch.tensor(ids, dtype=torch.int32, device=dev), w


def run(weights, dies, dev, rows, x, ids, w):
    """The routed experts at ``rows`` as ``moe.prepared`` prepares them at q8 (planned;
    above 64 rows the wide engines into a local partial)."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        stream,
        to_cute,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe import (
        experts_runtime as er,
    )

    buffers = er.scratch(16, dev, rows=max(rows, 64))
    experts = er.PersistentExperts(*weights, dies=dies, q=Q)
    if rows > 64:
        partial = torch.empty(rows, H, dtype=torch.bfloat16, device=dev)
        wide = er.WideExperts(experts, buffers, {rows: partial}, planned=True)
        writer = er.compile_wide_plan(rows, dev.index)
        writer(to_cute(x, 16), er.tensor(buffers["plan"]), stream(dev))
        out = wide.run(x, w, ids)
        torch.cuda.synchronize()
        return out[:rows].clone(), "wide"
    out = torch.empty(rows * TOPK, H, dtype=torch.bfloat16, device=dev)
    experts.prepare(rows, buffers, out, (), True, False, False)
    kind = "grouped" if experts.grouped64(rows) else "experts"
    if rows in experts.rubin:
        kind = "rubin"
        writer = er.compile_image_plan(rows, dev.index)
    elif rows > 8:
        writer = er.compile_plan(rows, dev.index)
    else:
        writer = None
    if writer is not None:
        writer(to_cute(x, 16), to_cute(buffers["plan"], 16), stream(dev))
    if rows == 8 and er.polled(16, 8, dev):  # the routing kernel's mailbox
        box = torch.full((er.ROUTE_WORDS,), -1, dtype=torch.int32, device=dev)
        box[:128] = ids.reshape(-1)
        box[128] = 15  # every copy CTA's image part written (2 x 8 - 1)
        er.route_mailbox(buffers["workspace"]).copy_(box)
    experts.run(x, ids)
    torch.cuda.synchronize()
    return out.clone(), kind


@rubin
@pytest.mark.parametrize("rows", ROWS)
def test_dense_rows_bitwise(layers, rows):
    arms, dies, dev = layers
    x, ids, w = inputs(rows, dev, 1000 + rows)
    padded, kind = run(arms["padded"], dies, dev, rows, x, ids, w)
    dense, kind2 = run(arms["dense"], dies, dev, rows, x, ids, w)
    assert kind == kind2
    bad = (padded.view(torch.int16) != dense.view(torch.int16)).any(1).sum().item()
    assert bad == 0, (rows, kind, bad)
    assert padded.abs().sum() > 0


@rubin
def test_dense_rows_stock_path():
    """Two stock layers on one shared padded copy: each layer's refresh restores its
    stock output bitwise (real-checkpoint layout: zero padding)."""
    from tests.models.kimi_k3 import test_specialized_decode_wide_experts_gpu as wx
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe import (
        experts_runtime as er,
    )

    dev = torch.device("cuda", 0)
    torch.accelerator.set_device_index(dev)
    ids, topk_w, latent = wx.verify_routes(64, dev)
    built = [wx.production_experts(wx.logical_weights(dev, s), dev) for s in (11, 12)]
    stock = [wx.Stock(b["layer"], b["config"]) for b in built]
    before = [s(latent, topk_w, ids) for s in stock]
    assert not torch.equal(before[0], before[1])
    for b in built:
        layer = b["layer"]
        assert er.dense_w2_rows(layer)
        er.dense_w2(layer, b["dies"])
    shared = [b["layer"].w2_weight.data_ptr() for b in built]
    assert shared[0] == shared[1]
    for i in (0, 1, 0):
        built[i]["layer"]._dense_w2_refresh()
        after = stock[i](latent, topk_w, ids)
        assert torch.equal(after.view(torch.int16), before[i].view(torch.int16)), i


@rubin
def test_tp8_layout_untouched():
    """TP8 (DFn TP8 = DCP8): w13 768 rows, w2 rows of 192 bytes (no padding). The
    layout is not admitted (the parameter stays), and the TP8 experts build their
    descriptors and prepare at the padded-free pitch as before."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe import (
        experts_runtime as er,
    )

    dev, u8 = torch.device("cuda", 0), torch.uint8
    w13 = torch.zeros(E, 768, 1792, dtype=u8, device=dev)
    w2 = torch.zeros(E, H, 192, dtype=u8, device=dev)
    layer = routed_layer(w13, w2, unpadded=384)
    assert not er.dense_w2_rows(layer)
    assert er.place(layer) == ()
    s13 = torch.full((E, 768, 112), 127, dtype=u8, device=dev)
    s2 = torch.full((E, H, 12), 127, dtype=u8, device=dev)
    experts = er.PersistentExperts(w13, s13, w2, s2, q=Q)
    assert experts.tp == 8
    buffers = er.scratch(8, dev)
    for rows in (1, 8, 64):
        out = torch.empty(rows * TOPK, H, dtype=torch.bfloat16, device=dev)
        experts.prepare(rows, buffers, out, (), planned=True)
