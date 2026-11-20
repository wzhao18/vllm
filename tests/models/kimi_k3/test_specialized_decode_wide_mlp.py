# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DSpark drafter MLP as one die-local persistent grid (``cutedsl.gemm.wide_mlp``)
against the same math run as separate launches: the stream-K ``spread`` gate_up GEMM
(same tiles and K order), vLLM's silu_and_mul, the ``spread`` down GEMM. The fused
grid must be bitwise identical to that chain (its gate_up -> down hand-off is per
64-column block, its exchanges read partials by vector loads) and repeat bitwise
(the slot claim, tile counters and block counters re-arm for graph replays)."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not (current_platform.is_cuda() and current_platform.has_device_capability(100)),
    reason="tcgen05 2-CTA GEMM targets SM100+",
)


@pytest.mark.parametrize(
    "rows,hidden,inter", [(96, 2048, 1024), (192, 2048, 1024), (288, 2048, 1024),
                          (288, 7168, 3584)],
)  # fmt: skip
def test_wide_mlp(rows, hidden, inter):
    import vllm._custom_ops  # noqa: F401
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm import (
        wide_mlp,
        wide_sk,
    )

    try:
        _, pairs = wide_sk.die_pairs("cuda")
    except ValueError:
        pytest.skip("die-local GEMM needs a two-die device")
    if not wide_mlp.supports(rows, hidden, inter, pairs):
        pytest.skip(f"unsupported {rows}x{hidden}x{inter} at pairs {pairs}")
    torch.manual_seed(rows + hidden)
    gate_up = torch.randn(2 * inter, hidden, device="cuda", dtype=torch.bfloat16) * 0.05
    down = torch.randn(hidden, inter, device="cuda", dtype=torch.bfloat16) * 0.05
    x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    gu = wide_sk.PreparedWideSK(gate_up, [rows], spread=True, silu=True).prepare()
    dn = wide_sk.PreparedWideSK(down, [rows], spread=True).prepare()
    act = torch.empty(rows, inter, device="cuda", dtype=torch.bfloat16)
    ref = torch.empty(rows, hidden, device="cuda", dtype=torch.bfloat16)
    gu.run(x, act)
    dn.run(act, ref)
    fused = wide_mlp.PreparedWideMlp(gate_up, down, [rows]).prepare()
    out = torch.empty_like(ref)
    for _ in range(3):
        out.fill_(float("nan"))
        fused.run(x, out)
        torch.cuda.synchronize()
        assert torch.equal(out, ref)
    stock = torch.empty_like(act)
    torch.ops._C.silu_and_mul(stock, torch.nn.functional.linear(x, gate_up))
    stock = torch.nn.functional.linear(stock, down).float()
    err = (out.float() - stock).abs().max() / stock.abs().max()
    assert err < 2e-2, err.item()
