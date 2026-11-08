# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KimiVecSituAndMul (every SiTU MLP of the Kimi K3 model, decode instances included).

Decode-sized inputs (up to 2048 x 768 output elements: every decode step's shared
expert and drafter MLP) run the stock SituAndMul module itself, through its own
dispatch, so they give the stock model's bits whatever that dispatch is. Larger inputs
(prefill) run the vectorized kernel of ops/situ_and_mul.py, which is bitwise the stock
SituAndMul CUDA op (forward_cuda) at every size. Under a config whose custom ops are
off (SituAndMul dispatching to forward_native), the large inputs therefore keep the
CUDA op's bits rather than forward_native's (both FP32 math rounded once to BF16;
the prefill recipes ran this kernel already, VLLM_KIMI_K3_VEC_SITU=1)."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="CUDA situ_and_mul op"
)


def _bits(t):
    return t.view(torch.int16)


@pytest.mark.parametrize("beta,linear_beta", [(1.0, None), (2.0, 4.0)])
@pytest.mark.parametrize(
    "rows,cols",
    [
        (8, 768),  # decode rows: the stock module itself
        (64, 384),
        (512, 768),
        (2048, 768),  # the largest input on the stock module
        (2049, 768),  # the vectorized kernel from here on
        (4096, 768),
        (2048, 6144),  # hybrid-SP replicated shared expert
        (16384, 384),
    ],
)
def test_vec_situ_bitwise_stock(rows, cols, beta, linear_beta):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.activation import SituAndMul
    from vllm.models.kimi_k3.nvidia.model import KimiVecSituAndMul
    from vllm.models.kimi_k3.nvidia.ops.situ_and_mul import situ_and_mul

    g = torch.Generator(device="cuda").manual_seed(rows * 7 + cols)
    x = torch.randn(rows, 2 * cols, generator=g, device="cuda") * 6
    x[0, :8] = torch.tensor([0.0, -0.0, 1e-30, -1e-30, 60.0, -60.0, 3e4, -3e4])
    x = x.to(torch.bfloat16)
    small = rows * cols < KimiVecSituAndMul._VEC_MIN_OUT_ELEMS
    with set_current_vllm_config(VllmConfig()):
        module = KimiVecSituAndMul(beta=beta, linear_beta=linear_beta)
        ours = module(x)
        stock = SituAndMul(beta=beta, linear_beta=linear_beta)
        dispatched, cuda_op = stock(x), stock.forward_cuda(x)
    assert type(module.small) is SituAndMul
    assert ours.dtype == x.dtype and ours.shape == (rows, cols)
    # the vectorized kernel is the CUDA op, bit for bit, at every size
    vec = situ_and_mul(x, beta, linear_beta)
    assert torch.equal(_bits(vec), _bits(cuda_op))
    if small:  # the stock module and its dispatch
        assert torch.equal(_bits(ours), _bits(dispatched))
    else:
        assert torch.equal(_bits(ours), _bits(cuda_op))
