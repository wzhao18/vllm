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


# Prefill row counts (multi-row grid from PREFILL_STAGE_MIN_TOKENS rows) at the
# deployed latent width: C256 token shard 2048 rows, C512 SP shard 4350 rows of 8192.
@pytest.mark.parametrize("quant_kind", ["nvfp4", "mxfp4_mxfp8"])
@pytest.mark.parametrize(("n", "capacity"), [(511, 2048), (512, 2048), (2047, 2048),
                                             (4350, 8192)])  # fmt: skip
@pytest.mark.parametrize("with_padding", [False, True])
def test_stager_prefill_rows(quant_kind, n, capacity, with_padding):
    from vllm.models.kimi_k3.nvidia.megamoe_rubin import reference as R
    from vllm.models.kimi_k3.nvidia.megamoe_rubin.staging import stage_inputs

    h = 3584
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(n + capacity + 1)
    nvfp4 = quant_kind == "nvfp4"
    vec = 16 if nvfp4 else 32
    sf_cols = h // vec + 4
    hid = (torch.randn((n, h), device=dev, generator=g) * 3).to(torch.bfloat16)
    hid[n // 3, :64] = 0.0  # all-zero blocks (scale 0)
    ids = torch.randint(-1, E, (n, TOPK), device=dev, generator=g, dtype=torch.int64)
    wts = torch.rand((n, TOPK), device=dev, generator=g).to(torch.bfloat16)
    pad = torch.zeros(n, dtype=torch.bool, device=dev)
    pad[n - n // 5 :] = True
    x = torch.randint(0, 256, (capacity, h // 2 if nvfp4 else h), dtype=torch.uint8,
                      device=dev, generator=g)  # fmt: skip
    x_sf = torch.randint(0, 256, (capacity, sf_cols), dtype=torch.uint8, device=dev,
                         generator=g)  # fmt: skip
    idx = torch.randint(0, E, (capacity, TOPK), dtype=torch.int32, device=dev,
                        generator=g)  # fmt: skip
    sw = torch.rand((capacity, TOPK), device=dev, generator=g)
    err = torch.zeros(2, dtype=torch.int64, device=dev)
    # a strided view, as the token-shard / SP slices are
    wide = torch.empty((n, h + 128), dtype=torch.bfloat16, device=dev)
    wide[:, :h] = hid
    stage_inputs(wide[:, :h], ids, wts, x, x_sf, idx, sw, route_error=err,
                 quant_kind=quant_kind, num_experts=E,
                 is_padding=pad if with_padding else None)  # fmt: skip
    torch.cuda.synchronize()
    assert err.tolist() == [0, 0]
    masked = torch.where(pad.unsqueeze(1), -1, ids) if with_padding else ids
    rx, rsf, ridx, rsw = R.stage_inputs_torch(
        hid, masked, wts, quant_kind=quant_kind, capacity=capacity, sf_cols=sf_cols,
        num_experts=E,
    )  # fmt: skip
    assert torch.equal(idx, ridx)
    assert torch.equal(sw[:n], rsw[:n])
    assert torch.equal(x[:n], rx[:n])
    assert torch.equal(x_sf[:n], rsf[:n])


@pytest.mark.parametrize("n", [1, 600])
def test_stager_nvfp4_rounding_ties(n):
    """E2M1 ties (0.25, 0.75, ... 5), saturation and signed zeros at scale 1 (block
    amax 6 -> E4M3 scale 1.0): the hardware conversion matches the torch twin."""
    from vllm.models.kimi_k3.nvidia.megamoe_rubin import reference as R
    from vllm.models.kimi_k3.nvidia.megamoe_rubin.staging import stage_inputs

    h, dev = 3584, torch.device("cuda")
    ties = [6.0, 0.25, -0.25, 0.75, -0.75, 1.25, -1.25, 1.75, -1.75, 2.5, -2.5,
            3.5, -3.5, 5.0, -5.0, -0.0]  # fmt: skip
    near = [-6.0, 0.1, -0.1, 0.0, 0.5, -0.5, 4.9, -5.1, 2.75, -2.25, 0.2578125,
            -0.7421875, 1.0, -3.0, 5.5, -1e-30]  # fmt: skip
    row = torch.tensor(ties + near, dtype=torch.float32, device=dev).repeat(h // 32)
    hid = row.expand(n, h).to(torch.bfloat16).contiguous()
    sf_cols = h // 16 + 4
    ids = torch.full((n, TOPK), -1, dtype=torch.int32, device=dev)
    wts = torch.zeros((n, TOPK), device=dev)
    x = torch.zeros((n, h // 2), dtype=torch.uint8, device=dev)
    x_sf = torch.zeros((n, sf_cols), dtype=torch.uint8, device=dev)
    idx = torch.zeros((n, TOPK), dtype=torch.int32, device=dev)
    sw = torch.zeros((n, TOPK), device=dev)
    err = torch.zeros(2, dtype=torch.int64, device=dev)
    stage_inputs(hid, ids, wts, x, x_sf, idx, sw, route_error=err,
                 quant_kind="nvfp4", num_experts=E)  # fmt: skip
    rx, rsf, _, _ = R.stage_inputs_torch(
        hid, ids, wts, quant_kind="nvfp4", capacity=n, sf_cols=sf_cols, num_experts=E
    )
    torch.cuda.synchronize()
    assert torch.equal(x, rx)
    assert torch.equal(x_sf, rsf)
