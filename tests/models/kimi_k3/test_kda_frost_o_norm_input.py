# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The KDA prefill core output must reach the o_norm that reads it.

On a pure prefill step the FROST / persistent core may leave its output in its
own buffer and skip the copy into ``core_attn_out`` only when the o_norm is the
MXFP8 prefill kernel (which reads that buffer). With a BF16 o_proj input (the TP
fused GEMM-AR o_proj, TP8 + DCP8 prefill) the in-place gated RMSNorm reads
``core_attn_out``: skipping the copy there normalized uninitialized memory
(NaN / non-reproducible prefill steps).
"""

import pytest
import torch

from vllm.models.kimi_k3.nvidia.kda import frost_out_feeds_o_norm
from vllm.models.kimi_k3.nvidia.ops.mxfp8_act import Mxfp8Buffer

T, H, D = 64, 2, 128


def _buffers(device: str):
    core_attn_out = torch.empty(1, T, H, D, dtype=torch.bfloat16, device=device)
    frost_out = torch.randn(1, T, H, D, dtype=torch.bfloat16, device=device)
    return core_attn_out, frost_out


def test_alias_only_for_the_mxfp8_o_norm():
    core_attn_out, frost_out = _buffers("cpu")
    # BF16 o_proj input (o_mxfp8 None): the in-place o_norm reads core_attn_out.
    assert not frost_out_feeds_o_norm(core_attn_out, core_attn_out, frost_out, None)
    o_mxfp8 = object.__new__(Mxfp8Buffer)
    # MXFP8 prefill o_norm reads the FROST output directly.
    assert frost_out_feeds_o_norm(core_attn_out, core_attn_out, frost_out, o_mxfp8)
    # Not a pure prefill step (output is a slice of another buffer).
    other = torch.empty_like(core_attn_out)
    assert not frost_out_feeds_o_norm(other, core_attn_out, frost_out, o_mxfp8)
    assert not frost_out_feeds_o_norm(
        core_attn_out, core_attn_out, frost_out[:, : T // 2], o_mxfp8
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@torch.inference_mode()
def test_bf16_o_norm_reads_the_core_output():
    """The KDA prefill tail with a BF16 o_proj input, starting from a poisoned
    ``core_attn_out``: the normalized output must equal the gated RMSNorm of
    the core output, bit for bit, and be finite."""
    from vllm.models.kimi_k3.nvidia.ops import kda_norm

    torch.manual_seed(0)
    core_attn_out, frost_out = _buffers("cuda")
    core_attn_out.fill_(float("nan"))
    g2 = torch.randn(1, T, H, D, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(D, dtype=torch.float32, device="cuda")
    eps = 1e-5

    # As KimiK3DeltaAttention._forward does after the FROST / persistent core.
    if not frost_out_feeds_o_norm(core_attn_out, core_attn_out, frost_out, None):
        core_attn_out.copy_(frost_out)
    kda_norm.rmsnorm_gated_(core_attn_out, g2, weight, eps)

    expected = frost_out.clone()
    kda_norm.rmsnorm_gated_(expected, g2, weight, eps)
    assert torch.isfinite(core_attn_out).all()
    torch.testing.assert_close(core_attn_out, expected, atol=0, rtol=0)
