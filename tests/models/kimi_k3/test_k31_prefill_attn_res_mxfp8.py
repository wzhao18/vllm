# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""specialized_prefill.attn_res_mxfp8 == mxfp8_quantize(native AttnRes), bit
for bit (values, swizzled scales, padded scale rows), and its BF16 output ==
the native AttnRes output."""

import pytest
import torch

from vllm.platforms import current_platform

HIDDEN = 7168
MAX_BLOCKS = 9
EPS = 1e-6

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability_family(100)
    or not hasattr(torch.ops._C, "kimi_k3_attn_res"),
    reason="native AttnRes requires an SM100-family build",
)


def _inputs(rows, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def rn(*s, scale=1.0):
        return (torch.randn(*s, device="cuda", generator=g) * scale).to(torch.bfloat16)

    prefix = rn(rows, HIDDEN)
    delta = rn(rows, HIDDEN, scale=0.5)
    blocks = rn(rows, MAX_BLOCKS, HIDDEN)
    norm_w = 1.0 + rn(HIDDEN, scale=0.1).float()
    qk_w = rn(HIDDEN, scale=HIDDEN**-0.5)
    out_norm_w = 1.0 + rn(HIDDEN, scale=0.1).float()
    return prefix, delta, blocks, norm_w.bfloat16(), qk_w, out_norm_w.bfloat16()


@pytest.mark.parametrize("rows", [1, 77, 128, 1000, 4099])
@pytest.mark.parametrize(
    "num_blocks,has_delta,write_idx",
    [(0, False, -1), (3, True, -1), (5, False, 5), (8, True, -1), (6, True, 6)],
)
@pytest.mark.parametrize("bf16_out", [True, False])
def test_attn_res_mxfp8_bitwise(rows, num_blocks, has_delta, write_idx, bf16_out):
    from vllm import _custom_ops as ops
    from vllm.models.kimi_k3.nvidia.ops import mxfp8_act
    from vllm.models.kimi_k3.nvidia.specialized_prefill.attn_res_mxfp8 import (
        attn_res_mxfp8,
    )

    prefix, delta, blocks, norm_w, qk_w, out_norm_w = _inputs(rows)
    delta = delta if has_delta else None
    blocks_ref = blocks.clone()
    ref = ops.kimi_k3_attn_res(
        prefix, delta, blocks_ref, norm_w, qk_w, out_norm_w,
        num_blocks, write_idx, EPS, EPS,
    )
    ref_q = mxfp8_act.quantize(ref)
    blocks_new = blocks.clone()
    out, qa = attn_res_mxfp8(
        prefix, delta, blocks_new, norm_w, qk_w, out_norm_w,
        num_blocks, write_idx, EPS, EPS, bf16_out=bf16_out,
    )
    torch.accelerator.synchronize()
    # The block write (when requested) is the native kernel's.
    assert torch.equal(blocks_new, blocks_ref)
    if bf16_out:
        assert torch.equal(out, ref)
    else:
        assert out is None
    assert torch.equal(qa.data.view(torch.uint8), ref_q.data.view(torch.uint8))
    n = ref_q.scale.numel()
    assert qa.scale.numel() == n
    assert torch.equal(qa.scale.view(torch.uint8), ref_q.scale.view(torch.uint8))


def test_matches_flashinfer_quantize():
    """The fused scales/values also equal FlashInfer's mxfp8_quantize."""
    from flashinfer import mxfp8_quantize

    from vllm import _custom_ops as ops
    from vllm.models.kimi_k3.nvidia.specialized_prefill.attn_res_mxfp8 import (
        attn_res_mxfp8,
    )

    prefix, delta, blocks, norm_w, qk_w, out_norm_w = _inputs(300, seed=1)
    ref = ops.kimi_k3_attn_res(
        prefix, delta, blocks.clone(), norm_w, qk_w, out_norm_w, 4, -1, EPS, EPS
    )
    fq, fs = mxfp8_quantize(ref, is_sf_swizzled_layout=True)
    _, qa = attn_res_mxfp8(
        prefix, delta, blocks.clone(), norm_w, qk_w, out_norm_w, 4, -1, EPS, EPS,
        bf16_out=False,
    )
    assert torch.equal(qa.data.view(torch.uint8), fq.view(torch.uint8))
    assert torch.equal(
        qa.scale.view(torch.uint8)[: fs.numel()], fs.view(torch.uint8).flatten()
    )
