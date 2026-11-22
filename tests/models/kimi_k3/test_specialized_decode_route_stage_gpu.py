# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 SP decode ``route_stage`` (router top-k + MegaMoE staging in one launch,
``specialized_decode.cutedsl.moe.route_stage``) is bitwise the stock chain it
replaces: ``fused_grouped_topk`` on the FP32 router logits, the latent rounded to
BF16, then ``megamoe_rubin.staging.stage_inputs`` (NVFP4 and MXFP8 staging, padding
rows, dead rows, SF pad columns, tied scores, the route-error flag untouched).
One GPU, a few seconds."""

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
    pytest.skip("needs an sm_10x GPU", allow_module_level=True)

from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (  # noqa: E402
    fused_grouped_topk,
)
from vllm.model_executor.layers.fused_moe.router.base_router import (  # noqa: E402
    eplb_map_to_physical_and_record,
)
from vllm.models.kimi_k3.nvidia.megamoe_rubin.staging import (  # noqa: E402
    stage_inputs,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe import (  # noqa: E402
    route_stage,
)

E, TOPK, LATENT, CAP, SCALE = 896, 16, 3584, 128, 2.827


def _buffers(nvfp4, dev):
    vec = 16 if nvfp4 else 32
    g = torch.Generator(device=dev).manual_seed(7)
    x = torch.randint(0, 256, (CAP, LATENT // (2 if nvfp4 else 1)), generator=g,
                      device=dev, dtype=torch.uint8)  # fmt: skip
    sf = torch.randint(0, 256, (CAP, LATENT // vec + 4), generator=g, device=dev,
                       dtype=torch.uint8)  # fmt: skip
    idx = torch.randint(0, E, (CAP, TOPK), generator=g, device=dev, dtype=torch.int32)
    sw = torch.rand((CAP, TOPK), generator=g, device=dev)
    err = torch.zeros(2, dtype=torch.int64, device=dev)
    return [x, sf, idx, sw, err]


@pytest.mark.parametrize("nvfp4", [True, False])
@pytest.mark.parametrize("n", [1, 8, 24, 32, 96, 128])
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("ties", [False, True])
def test_route_stage_bitwise(nvfp4, n, padded, ties):
    dev = torch.device("cuda")
    torch.manual_seed(n + 2 * nvfp4 + 4 * padded + 8 * ties)
    y = torch.randn((n, E + LATENT), device=dev) * 3.0
    if ties:  # many equal selection scores (exercises the tie-heavy branch)
        y[:, :E] = torch.round(y[:, :E])
        y[: n // 2, :E] = 0.5
    y[:, E + 5 :: 97] *= 1e4  # some large blocks
    y[:, E : E + 64] = 0.0  # zero blocks
    bias = (torch.randn(E, device=dev) * 0.1).float()
    if ties:
        bias.zero_()
    pad = (torch.rand(n, device=dev) < 0.3) if padded else None

    ref = _buffers(nvfp4, dev)
    wts, ids = fused_grouped_topk(
        hidden_states=y[:, E:].to(torch.bfloat16),
        gating_output=y[:, :E].contiguous(),
        topk=TOPK,
        renormalize=True,
        e_score_correction_bias=bias,
        num_expert_group=1,
        topk_group=1,
        scoring_func="sigmoid",
        routed_scaling_factor=SCALE,
    )
    stage_inputs(
        y[:, E:].to(torch.bfloat16),
        ids,
        wts,
        ref[0],
        ref[1],
        ref[2],
        ref[3],
        route_error=ref[4],
        quant_kind="nvfp4" if nvfp4 else "mxfp4_mxfp8",
        num_experts=E,
        input_norm_const=1.0,
        is_padding=pad,
    )
    got = _buffers(nvfp4, dev)
    route_stage._module().route_stage(
        y, E, bias, got[0], got[1], got[2], got[3], got[4], pad, E, True, SCALE,
        nvfp4, 1.0, None, None, None, None, None,
    )  # fmt: skip
    torch.accelerator.synchronize()
    for name, a, b in zip(("x", "x_sf", "topk_idx", "topk_weights", "err"), got, ref):
        if a.dtype == torch.float32:
            a, b = a.view(torch.int32), b.view(torch.int32)
        bad = (a != b).nonzero()
        assert bad.numel() == 0, f"{name}: {bad.shape[0]} mismatches, first {bad[:4].tolist()}"


@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("record", [False, True])
def test_route_stage_eplb(padded, record):
    """EPLB in the kernel: the stock padding mask, then stock
    ``eplb_map_to_physical_and_record`` (replica choice and load record), then
    staging with the physical expert count."""
    dev, n, slots = torch.device("cuda"), 96, 4
    torch.manual_seed(11 + padded + 2 * record)
    y = torch.randn((n, E + LATENT), device=dev) * 3.0
    bias = (torch.randn(E, device=dev) * 0.1).float()
    pad = (torch.rand(n, device=dev) < 0.3) if padded else None
    reps = torch.randint(1, slots, (E,), device=dev)
    reps[:8] = 1
    phys = int(reps.sum())
    perm = torch.randperm(phys, device=dev)
    l2p = torch.full((E, slots), -1, dtype=torch.int64, device=dev)
    start = torch.cumsum(reps, 0) - reps
    for e in range(E):
        l2p[e, : reps[e]] = perm[start[e] : start[e] + reps[e]]
    rec = torch.tensor(record, device=dev)
    unpadded = torch.tensor(n - 7, dtype=torch.int32, device=dev)

    wts, ids = fused_grouped_topk(
        hidden_states=y[:, E:].to(torch.bfloat16),
        gating_output=y[:, :E].contiguous(),
        topk=TOPK,
        renormalize=True,
        e_score_correction_bias=bias,
        num_expert_group=1,
        topk_group=1,
        scoring_func="sigmoid",
        routed_scaling_factor=SCALE,
    )
    if pad is not None:
        ids = torch.where(pad.unsqueeze(1), -1, ids)
    load_ref = torch.zeros(phys, dtype=torch.int32, device=dev)
    ids = eplb_map_to_physical_and_record(
        topk_ids=ids,
        expert_load_view=load_ref,
        logical_to_physical_map=l2p,
        logical_replica_count=reps,
        record_enabled=rec,
        num_unpadded_tokens=unpadded,
    )
    ref = _buffers(True, dev)
    stage_inputs(
        y[:, E:].to(torch.bfloat16), ids, wts, *ref[:4], route_error=ref[4],
        quant_kind="nvfp4", num_experts=phys, input_norm_const=1.0, is_padding=None,
    )  # fmt: skip
    got = _buffers(True, dev)
    load = torch.zeros(phys, dtype=torch.int32, device=dev)
    route_stage._module().route_stage(
        y, E, bias, *got, pad, phys, True, SCALE, True, 1.0, l2p, reps, load, rec,
        unpadded,
    )  # fmt: skip
    torch.accelerator.synchronize()
    assert phys > E
    for name, a, b in zip(("x", "x_sf", "topk_idx", "topk_weights", "err"), got, ref):
        if a.dtype == torch.float32:
            a, b = a.view(torch.int32), b.view(torch.int32)
        bad = (a != b).nonzero()
        assert bad.numel() == 0, f"{name}: {bad.shape[0]} mismatches, first {bad[:4].tolist()}"
    assert torch.equal(load, load_ref)
    live = n - 7 - (int(pad[: n - 7].sum()) if padded else 0)
    assert int(load.sum()) == record * live * TOPK


@pytest.mark.parametrize("nvfp4", [True, False])
@pytest.mark.parametrize("n,splits", [(32, 3), (96, 5)])
def test_route_stage_split_partials(nvfp4, n, splits):
    """Split-K partials [splits, n, E + latent] summed in split order give exactly
    what the summed rows give (the fused split-K GEMM front at n96)."""
    dev = torch.device("cuda")
    torch.manual_seed(n + splits + 2 * nvfp4)
    parts = torch.randn((splits, n, E + LATENT), device=dev)
    y = parts[0].clone()
    for s in range(1, splits):
        y += parts[s]
    bias = (torch.randn(E, device=dev) * 0.1).float()
    outs = []
    for src in (y, parts):
        got = _buffers(nvfp4, dev)
        route_stage._module().route_stage(
            src, E, bias, got[0], got[1], got[2], got[3], got[4], None, E, True,
            SCALE, nvfp4, 1.0, None, None, None, None, None,
        )  # fmt: skip
        outs.append(got)
    torch.accelerator.synchronize()
    for name, a, b in zip(("x", "x_sf", "topk_idx", "topk_weights", "err"), *outs):
        if a.dtype == torch.float32:
            a, b = a.view(torch.int32), b.view(torch.int32)
        assert torch.equal(a, b), name


def test_route_stage_rejects_bad_shapes():
    dev = torch.device("cuda")
    y = torch.zeros((CAP + 1, E + LATENT), device=dev)
    b = _buffers(True, dev)
    with pytest.raises(RuntimeError):
        route_stage._module().route_stage(
            y, E, torch.zeros(E, device=dev), *b, None, E, True, 1.0, True, 1.0,
            None, None, None, None, None,
        )  # fmt: skip
