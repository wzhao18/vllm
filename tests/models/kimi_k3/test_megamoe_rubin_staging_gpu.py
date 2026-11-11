# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rubin MegaMoE Triton stager (one GPU): the live-row grid + dead-row route fill and
the folded ``is_padding`` mask match the pure-torch twin applied after the caller's
``torch.where(is_padding[:, None], -1, topk_ids)``, over the whole bucket capacity."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="the stager is a Triton CUDA kernel"
)

H, TOPK, E = 7168, 16, 64


@pytest.mark.parametrize("quant_kind", ["nvfp4", "mxfp4_mxfp8"])
@pytest.mark.parametrize(("n", "capacity"), [(0, 256), (1, 256), (52, 2048), (90, 256)])
@pytest.mark.parametrize("with_padding", [False, True])
def test_stager_live_rows_and_padding_mask(quant_kind, n, capacity, with_padding):
    from vllm.models.kimi_k3.nvidia.megamoe_rubin import reference as R
    from vllm.models.kimi_k3.nvidia.megamoe_rubin.staging import stage_inputs

    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(n + capacity)
    nvfp4 = quant_kind == "nvfp4"
    vec = 16 if nvfp4 else 32
    sf_cols = H // vec + 4
    hid = (torch.randn((n, H), device=dev, generator=g) * 3).to(torch.bfloat16)
    ids = torch.randint(-1, E, (n, TOPK), device=dev, generator=g, dtype=torch.int64)
    wts = torch.rand((n, TOPK), device=dev, generator=g)
    pad = torch.zeros(n, dtype=torch.bool, device=dev)
    pad[n - n // 4 :] = True
    if n:
        ids[-1, 0] = E + 3  # out of range, but on a padding row when padded
    # stale buffers: every staged byte the contract defines must be rewritten
    x = torch.randint(0, 256, (capacity, H // 2 if nvfp4 else H), dtype=torch.uint8,
                      device=dev, generator=g)  # fmt: skip
    x_sf = torch.randint(0, 256, (capacity, sf_cols), dtype=torch.uint8, device=dev,
                         generator=g)  # fmt: skip
    idx = torch.randint(0, E, (capacity, TOPK), dtype=torch.int32, device=dev,
                        generator=g)  # fmt: skip
    sw = torch.rand((capacity, TOPK), device=dev, generator=g)
    err = torch.zeros(2, dtype=torch.int64, device=dev)
    stage_inputs(hid, ids, wts, x, x_sf, idx, sw, route_error=err,
                 quant_kind=quant_kind, num_experts=E,
                 is_padding=pad if with_padding else None)  # fmt: skip
    masked = torch.where(pad.unsqueeze(1), -1, ids) if with_padding else ids
    expect_error = n > 0 and not (with_padding and bool(pad[-1]))
    torch.cuda.synchronize()
    assert err.tolist() == ([1, E + 3] if expect_error else [0, 0])
    if expect_error:
        masked = masked.clone()
        masked[-1, 0] = -1  # the stager masks the offending route
    rx, rsf, ridx, rsw = R.stage_inputs_torch(
        hid, masked, wts, quant_kind=quant_kind, capacity=capacity, sf_cols=sf_cols,
        num_experts=E,
    )  # fmt: skip
    assert torch.equal(idx, ridx)  # live rows and every dead row of the bucket
    assert torch.equal(sw[:n], rsw[:n])
    assert torch.equal(x[:n], rx[:n])
    assert torch.equal(x_sf[:n], rsf[:n])
    with pytest.raises(ValueError, match="is_padding"):
        stage_inputs(hid, ids, wts, x, x_sf, idx, sw, route_error=err,
                     quant_kind=quant_kind, num_experts=E,
                     is_padding=pad.to(torch.int32))  # fmt: skip
