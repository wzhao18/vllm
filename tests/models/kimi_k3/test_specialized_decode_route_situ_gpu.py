# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 specialized decode ``RouteSitu`` route role (TP16 polled logits): the lean
top16 (warp 0 alone from the cutoff to the published ids, ``RouteSitu.top16``) gives
the same ids (rank order) and BF16 weights, bit for bit, as the non-lean path, and the
same expert set as stock ``fused_grouped_topk`` on the FP32 logits. Continuous logits
(<= 32 candidates) and flat / tied logits (> 32 candidates, the round-based ranking).
One GPU, well under a minute."""

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
    pytest.skip("needs an sm_10x GPU", allow_module_level=True)

import cutlass as c  # noqa: E402

from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (  # noqa: E402
    fused_grouped_topk,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (  # noqa: E402
    stream,
    to_cute,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe.front import (  # noqa: E402
    WIDTH,
    compile_route,
)

E, TOPK, SHARED, SCALE = 896, 16, 384, 2.827
SENTINEL = -(2**31)


def _route(batch, logits, bias, lean, rcp):
    dev = logits.device
    kernel = compile_route(
        batch, SHARED, True, (True, False, False), dev.index, polled=True, lean=lean,
        rcp=rcp,
    )  # fmt: skip
    box = torch.empty(1, 64, WIDTH, dtype=torch.bfloat16, device=dev)
    box.view(torch.int32).fill_(SENTINEL)
    box.view(torch.int32)[0, :batch, 1792 : 1792 + E] = logits.view(torch.int32)
    ids = torch.full((batch, TOPK), -1, dtype=torch.int32, device=dev)
    weights = torch.zeros(batch, TOPK, dtype=torch.bfloat16, device=dev)
    keep = (
        torch.zeros(batch, E, dtype=torch.float32, device=dev),
        torch.zeros(batch, 3584, dtype=torch.bfloat16, device=dev),
        torch.zeros(batch, 2 * SHARED, dtype=torch.float32, device=dev),
        torch.zeros(batch, SHARED, dtype=torch.bfloat16, device=dev),
    )
    args = (to_cute(keep[0]), to_cute(bias), to_cute(ids), to_cute(weights))
    args += (to_cute(box), to_cute(keep[1]), to_cute(keep[2]), to_cute(keep[3]))
    args += tuple(c.Float32(v) for v in (SCALE, 4.0, 25.0))
    kernel(*args, stream(dev))
    torch.accelerator.synchronize()
    # the polled cells were re-armed
    assert (box.view(torch.int32)[0, :batch, 1792 : 1792 + E] == SENTINEL).all()
    return ids, weights


@pytest.mark.parametrize("batch", [8, 64])
@pytest.mark.parametrize("kind", ["continuous", "ties", "flat"])
@pytest.mark.parametrize("rcp", [False, True])
def test_lean_top16_matches_non_lean(batch, kind, rcp):
    dev = torch.device("cuda")
    torch.manual_seed(batch + 100 * len(kind) + 7 * rcp)
    logits = torch.randn(batch, E, device=dev) * 3.0
    bias = (torch.randn(E, device=dev) * 0.1).float()
    if kind == "ties":  # many equal keys: > 32 candidates on some rows
        logits = torch.round(logits)
        logits[: batch // 2] = 0.5
        bias.zero_()
    elif kind == "flat":  # every key equal: all 896 experts are candidates
        logits.fill_(-1.25)
        bias.zero_()
    lean = _route(batch, logits, bias, True, rcp)
    ref = _route(batch, logits, bias, False, False)
    assert torch.equal(lean[0], ref[0]), (lean[0] != ref[0]).nonzero()[:4].tolist()
    a, b = lean[1].view(torch.int16), ref[1].view(torch.int16)
    assert torch.equal(a, b), (a != b).nonzero()[:4].tolist()
    if kind == "continuous":
        _, stock = fused_grouped_topk(
            hidden_states=logits.to(torch.bfloat16),
            gating_output=logits.contiguous(),
            topk=TOPK,
            renormalize=True,
            e_score_correction_bias=bias,
            num_expert_group=1,
            topk_group=1,
            scoring_func="sigmoid",
            routed_scaling_factor=SCALE,
        )
        got = lean[0].long().sort(1).values
        want = stock.long().sort(1).values
        assert torch.equal(got, want)
    # ties break to the lower expert, ranks in key order
    assert (lean[0] >= 0).all() and (lean[0] < E).all()
    if kind == "flat":
        want = torch.arange(TOPK, device=dev, dtype=torch.int32).expand(batch, TOPK)
        assert torch.equal(lean[0], want)
