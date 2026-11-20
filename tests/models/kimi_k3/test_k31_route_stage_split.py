# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""route_stage on split input (FP32 router logits + BF16 latent rows, the stock
router GEMM || latent down pair) vs stock: ``fused_grouped_topk`` then the MegaMoE
stager ``stage_inputs``, bitwise; and vs route_stage on the concatenated FP32 rows."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.has_device_capability(100),
    reason="route_stage needs sm_100 family",
)

E, TOPK, HIDDEN = 896, 16, 3584


def _buffers(capacity, nvfp4, device):
    vec = 16 if nvfp4 else 32
    sf_cols = HIDDEN // vec + 4  # SF pad columns (zeroed by the stager)
    g = torch.Generator(device=device).manual_seed(7)

    def junk(shape, dtype):
        t = torch.randint(0, 255, shape, generator=g, device=device, dtype=torch.int32)
        return t.to(torch.uint8).view(dtype) if dtype != torch.uint8 else t.to(dtype)

    x = junk((capacity, HIDDEN // 2 if nvfp4 else HIDDEN), torch.uint8)
    sf = junk((capacity, sf_cols), torch.uint8)
    idx = torch.full((capacity, TOPK), 7, dtype=torch.int32, device=device)
    sw = torch.full((capacity, TOPK), 3.0, dtype=torch.float32, device=device)
    err = torch.zeros(2, dtype=torch.int64, device=device)
    return [x, sf, idx, sw, err]


@pytest.mark.parametrize("nvfp4", [True, False])
@pytest.mark.parametrize("n,capacity", [(1, 64), (33, 128), (96, 128), (96, 384)])
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("ties", [False, True])
def test_route_stage_split_vs_stock(nvfp4, n, capacity, renormalize, ties):
    from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
        fused_grouped_topk,
    )
    from vllm.models.kimi_k3.nvidia.megamoe_rubin.staging import stage_inputs
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe.route_stage import (
        _module,
    )

    device = torch.device("cuda")
    torch.manual_seed(n * 31 + capacity)
    logits = torch.randn(n, E, device=device) * 3
    logits[:, 5] = logits[:, 6]  # exact ties: stock tie order
    bias = torch.randn(E, device=device) * 0.1
    if ties:  # heavy exact ties: a few distinct scores, many equal lane maxima
        logits = torch.randint(-2, 3, (n, E), device=device).float()
        bias = torch.zeros(E, device=device)
    latent = (torch.randn(n, HIDDEN, device=device) * 4).to(torch.bfloat16)
    latent[:, 64:96] = 0  # all-zero quantization blocks
    norm_const, scale = 2.5, 2.827
    quant = "nvfp4" if nvfp4 else "mxfp8"

    ref = _buffers(capacity, nvfp4, device)
    w, ids = fused_grouped_topk(
        latent, logits, TOPK, renormalize, bias, 1, 1, "sigmoid", scale
    )
    stage_inputs(
        latent, ids, w, ref[0], ref[1], ref[2], ref[3], route_error=ref[4],
        quant_kind=quant, num_experts=E, input_norm_const=norm_const,
    )  # fmt: skip

    def ours(y, lat):
        out = _buffers(capacity, nvfp4, device)
        _module().route_stage(
            y, E, bias, out[0], out[1], out[2], out[3], out[4], None, E,
            renormalize, scale, nvfp4, norm_const, lat,
        )  # fmt: skip
        return out

    split = ours(logits, latent)
    concat = ours(torch.cat([logits, latent.float()], dim=1), None)
    torch.accelerator.synchronize()
    names = ("x", "sf", "topk_idx", "topk_weights", "route_error")
    for name, a, b, c in zip(names, ref, split, concat):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), name
        assert torch.equal(b.view(torch.uint8), c.view(torch.uint8)), name
