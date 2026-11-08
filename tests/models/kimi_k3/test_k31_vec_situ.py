# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KimiVecSituAndMul (every SiTU MLP of the Kimi K3 model, decode instances included)
is bitwise the stock SituAndMul CUDA op: decode-sized inputs (up to 2048 x 768 output
elements) run that op itself, larger ones the vectorized kernel of ops/situ_and_mul.py,
which must give the same bits."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="CUDA situ_and_mul op"
)


@pytest.mark.parametrize("beta,linear_beta", [(1.0, None), (2.0, 4.0)])
@pytest.mark.parametrize(
    "rows,cols",
    [
        (8, 768),  # decode rows: the stock op itself
        (64, 384),
        (512, 768),
        (2048, 768),  # the largest stock-op input
        (2049, 768),  # the vectorized kernel from here on
        (4096, 768),
        (2048, 6144),  # hybrid-SP replicated shared expert
        (16384, 384),
    ],
)
def test_vec_situ_bitwise_stock(rows, cols, beta, linear_beta):
    from vllm.model_executor.layers.activation import SituAndMul
    from vllm.models.kimi_k3.nvidia.model import KimiVecSituAndMul

    g = torch.Generator(device="cuda").manual_seed(rows * 7 + cols)
    x = torch.randn(rows, 2 * cols, generator=g, device="cuda") * 6
    x[0, :8] = torch.tensor([0.0, -0.0, 1e-30, -1e-30, 60.0, -60.0, 3e4, -3e4])
    x = x.to(torch.bfloat16)
    ours = KimiVecSituAndMul(beta=beta, linear_beta=linear_beta)(x)
    stock = SituAndMul(beta=beta, linear_beta=linear_beta)
    for ref in (stock(x), stock.forward_cuda(x)):  # (its dispatch, and the CUDA op)
        assert ours.dtype == ref.dtype and ours.shape == ref.shape
        assert torch.equal(ours.view(torch.int16), ref.view(torch.int16))
