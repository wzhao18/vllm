# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""dsa.query_uk (LocalQuery's W_UK + FP8 query pack) vs the stock BF16 bmm + pack."""

import pytest
import torch

from vllm.platforms import current_platform

G, P, QK, NOPE, LAT, ROW = 96, 128, 192, 128, 512, 576

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="tcgen05 (sm_10x) only",
)


def _stock(q, uk_t, scale, rows):
    qv = q.view(rows, G, QK)
    lat = torch.bmm(qv[:, :, :NOPE].transpose(0, 1), uk_t.transpose(1, 2))
    full = torch.cat([lat.transpose(0, 1), qv[:, :, NOPE:]], dim=-1).float()
    y = (full * (1.0 / scale.float())).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    out = torch.zeros((rows, P, ROW), dtype=torch.float8_e4m3fn, device=q.device)
    out[:, :G] = y
    return out


@pytest.mark.parametrize("rows", [72, 128, 200, 384])
def test_query_uk_matches_stock(rows):
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.query_uk import (
        query_uk,
    )

    torch.manual_seed(rows)
    dev = torch.device("cuda")
    q = (torch.randn(rows, G * QK, device=dev) * 2).to(torch.bfloat16)
    # q_pe values the stock clamp saturates (fmaxf(-448, fminf(y, 448)))
    q[1, NOPE + 3] = -1e30
    q[2, QK + NOPE + 5] = float("inf")
    q[3, NOPE + 6] = float("-inf")
    uk_t = (torch.randn(G, LAT, NOPE, device=dev) * 0.1).to(torch.bfloat16)
    scale = torch.tensor([0.02], device=dev)
    out = torch.zeros((rows, P, ROW), dtype=torch.float8_e4m3fn, device=dev)
    query_uk(q, uk_t, scale, out)
    ref = _stock(q, uk_t, scale, rows)
    a, b = out.view(torch.uint8), ref.view(torch.uint8)
    # q_pe and the padding heads: bitwise
    assert torch.equal(a[:, :, LAT:], b[:, :, LAT:])
    assert torch.equal(a[:, G:], b[:, G:])
    # latent: the same BF16 bmm up to the K order of the 128-term FP32 dot products,
    # which can flip a BF16 rounding (then at most one E4M3 code)
    diff = (a[:, :G, :LAT].int() - b[:, :G, :LAT].int()).abs()
    assert (diff > 1).sum().item() == 0
    assert (diff != 0).float().mean().item() < 2e-3
