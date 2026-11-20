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
    # The kernel updates the prefix in place (prefix += delta): own copies.
    prefix_ref, prefix_new = prefix.clone(), prefix.clone()
    blocks_ref = blocks.clone()
    ref = ops.kimi_k3_attn_res(
        prefix_ref, delta, blocks_ref, norm_w, qk_w, out_norm_w,
        num_blocks, write_idx, EPS, EPS,
    )
    ref_q = mxfp8_act.quantize(ref)
    blocks_new = blocks.clone()
    out, qa = attn_res_mxfp8(
        prefix_new, delta, blocks_new, norm_w, qk_w, out_norm_w,
        num_blocks, write_idx, EPS, EPS, bf16_out=bf16_out,
    )
    torch.accelerator.synchronize()
    # The block write (when requested) is the native kernel's.
    assert torch.equal(blocks_new, blocks_ref)
    assert torch.equal(prefix_new, prefix_ref)
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
        prefix.clone(), delta, blocks.clone(), norm_w, qk_w, out_norm_w, 4, -1, EPS, EPS
    )
    fq, fs = mxfp8_quantize(ref, is_sf_swizzled_layout=True)
    _, qa = attn_res_mxfp8(
        prefix.clone(), delta, blocks.clone(), norm_w, qk_w, out_norm_w, 4, -1, EPS, EPS,
        bf16_out=False,
    )
    assert torch.equal(qa.data.view(torch.uint8), fq.view(torch.uint8))
    assert torch.equal(
        qa.scale.view(torch.uint8)[: fs.numel()], fs.view(torch.uint8).flatten()
    )


@pytest.mark.parametrize("rows", [1, 77, 1000, 4099])
@pytest.mark.parametrize(
    "num_blocks,has_delta,write_idx",
    [(0, False, 0), (2, True, -1), (7, True, 7), (8, True, -1)],
)
@pytest.mark.parametrize("out_norm", [True, False])
def test_attn_res_pf_bf16_bitwise(rows, num_blocks, has_delta, write_idx, out_norm):
    """The BF16-only modes of the prefill kernel (with / without the output
    RMSNorm) == the native AttnRes, bit for bit (output, prefix, blocks)."""
    from vllm import _custom_ops as ops
    from vllm.models.kimi_k3.nvidia.specialized_prefill.attn_res_mxfp8 import (
        _pf_module,
    )

    prefix, delta, blocks, norm_w, qk_w, out_norm_w = _inputs(rows, seed=2)
    delta = delta if has_delta else None
    onw = out_norm_w if out_norm else None
    prefix_ref, prefix_new = prefix.clone(), prefix.clone()
    blocks_ref, blocks_new = blocks.clone(), blocks.clone()
    ref = ops.kimi_k3_attn_res(
        prefix_ref, delta, blocks_ref, norm_w, qk_w, onw, num_blocks, write_idx,
        EPS, EPS,
    )
    out = torch.empty_like(prefix)
    _pf_module().attn_res_pf(
        prefix_new, delta, blocks_new, norm_w, qk_w, onw, out, None, None,
        num_blocks, write_idx, EPS, EPS, None, None, None,
    )
    torch.accelerator.synchronize()
    assert torch.equal(blocks_new, blocks_ref)
    assert torch.equal(prefix_new, prefix_ref)
    assert torch.equal(out, ref)


@pytest.mark.parametrize("rows", [1500, 4099])
def test_attn_res_die_local_bitwise(rows):
    """Die-local block buffer + per-die token claims (two-die devices): every
    token is processed exactly once and the results equal the native kernel,
    over back-to-back calls (the claim counters reset themselves)."""
    from vllm import _custom_ops as ops
    from vllm.models.kimi_k3.nvidia.specialized_prefill import attn_res_dielocal
    from vllm.models.kimi_k3.nvidia.specialized_prefill.attn_res_mxfp8 import (
        attn_res_bf16,
        attn_res_mxfp8,
    )

    prefix, delta, blocks, norm_w, qk_w, out_norm_w = _inputs(rows, seed=3)
    blocks = blocks[:, :8].contiguous()
    dl = attn_res_dielocal.block_residual(prefix, 8)
    if dl is None:
        pytest.skip("single-die device")
    assert attn_res_dielocal.kernel_args(dl)[0] is not None
    dl.copy_(blocks)
    prefix_ref, prefix_new = prefix.clone(), prefix.clone()
    for nb, d, widx, mode in [(5, delta, -1, "q"), (2, delta, 2, "bq"),
                              (8, delta, -1, "bf16"), (8, None, -1, "bq")]:
        # The first call moves the prefix into the die-local prefix buffer.
        po = attn_res_dielocal.prefix_out(prefix_new, dl)
        ref = ops.kimi_k3_attn_res(prefix_ref, d, blocks, norm_w, qk_w,
                                   out_norm_w, nb, widx, EPS, EPS)
        if mode == "bf16":
            out = attn_res_bf16(prefix_new, d, dl, norm_w, qk_w, out_norm_w, nb,
                                widx, EPS, EPS, prefix_out=po)
            torch.accelerator.synchronize()
            assert torch.equal(out, ref)
        else:
            out, qa = attn_res_mxfp8(prefix_new, d, dl, norm_w, qk_w, out_norm_w,
                                     nb, widx, EPS, EPS, bf16_out=mode == "bq",
                                     prefix_out=po)
            from vllm.models.kimi_k3.nvidia.ops import mxfp8_act

            ref_q = mxfp8_act.quantize(ref)
            torch.accelerator.synchronize()
            if mode == "bq":
                assert torch.equal(out, ref)
            assert torch.equal(qa.data.view(torch.uint8), ref_q.data.view(torch.uint8))
        if po is not None:
            prefix_new = po
        assert torch.equal(prefix_new, prefix_ref)
        assert torch.equal(dl, blocks)
    assert prefix_new.data_ptr() != prefix.data_ptr()
