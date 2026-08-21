# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8_packed_for_deepgemm,
    situ_mul_quant_fp8_packed_triton,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="The packed SiTU quantization kernel requires CUDA.",
)


@pytest.mark.parametrize("num_tokens", [1, 32, 160, 256])
@torch.inference_mode()
def test_situ_mul_fp8_quant_packed(num_tokens: int):
    torch.manual_seed(42)
    intermediate_size = 6144
    gate_up = torch.randn(
        (num_tokens, 2 * intermediate_size),
        dtype=torch.bfloat16,
        device="cuda",
    )

    activated = torch.empty(
        (num_tokens, intermediate_size),
        dtype=gate_up.dtype,
        device=gate_up.device,
    )
    torch.ops._C.situ_and_mul(activated, gate_up, 4.0, 25.0)
    expected_q, expected_scale = per_token_group_quant_fp8_packed_for_deepgemm(
        activated,
        group_size=128,
        use_ue8m0=True,
    )
    actual_q, actual_scale = situ_mul_quant_fp8_packed_triton(
        gate_up,
        group_size=128,
        beta=4.0,
        linear_beta=25.0,
    )

    torch.testing.assert_close(
        actual_q.float(), expected_q.float(), rtol=0.0, atol=0.125
    )
    assert torch.equal(actual_scale, expected_scale)
