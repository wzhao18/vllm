# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The SP MoE front's router | latent down GEMM on the split-K kernel
(``sp_mlp.SplitKFront``) and its per-rows dispatch (``sp_mlp.FrontGemms``): FP32
outputs within FP32-accumulation noise of an FP64 reference, for every split-K row
of ``sp_mlp.ROUTER_LATENT``, and deterministic across calls."""

import pytest
import torch

from vllm.models.kimi_k3.nvidia.specialized_decode.runtime import sp_mlp
from vllm.models.kimi_k3.nvidia.specialized_decode.shapes import Shape
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.policy import (
    batch_table,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="Rubin / Blackwell GPU required",
)

N, K = 4480, 7168  # [router 896 | latent down 3584] x hidden


def split_rows():
    table = batch_table(sp_mlp.ROUTER_LATENT["*"])
    rows = [r for r in (32, 64, 96) if isinstance(table.get(r), tuple)]
    return {r: sp_mlp.rows_policy(sp_mlp.ROUTER_LATENT, Shape(r, 1)) for r in rows}


@pytest.mark.parametrize("rows", sorted(split_rows()) or [None])
def test_split_k_front_matches_fp64(rows):
    if rows is None:
        pytest.skip("no split-K rows in ROUTER_LATENT")
    torch.manual_seed(rows)
    dev = torch.device("cuda")
    packed = (torch.randn(N, K, device=dev) * 0.02).to(torch.bfloat16)
    x = torch.randn(rows, K, device=dev).to(torch.bfloat16)
    front = sp_mlp.FrontGemms([sp_mlp.SplitKFront(packed, {rows: split_rows()[rows]})])
    assert set(front.plans) == {rows}
    y = torch.empty(rows, N, device=dev, dtype=torch.float32)
    front.run(packed, x, y)
    ref = x.double() @ packed.double().T
    err = (y.double() - ref).abs().max().item() / ref.abs().max().item()
    assert err < 2e-5, err
    again = torch.empty_like(y)
    front.run(packed, x, again)
    assert torch.equal(y, again)  # fixed split-K summation order
