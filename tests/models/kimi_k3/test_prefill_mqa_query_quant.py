# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The MQA query of a DCP-1 FLASHINFER_MLA_SPARSE layer (Kimi K3.1 DEP16 prefill):
MLAAttention skips the in-place RoPE concat when the backend writes the FP8 query
(concat + static quant + head pad) in one kernel, and the result is bitwise the
in-place path's (bmm into the concat buffer, RoPE copy, scaled_fp8_quant, head pad)."""

from types import SimpleNamespace

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip("CUDA only", allow_module_level=True)

from vllm.v1.attention.backends.mla import flashinfer_mla_sparse as fims  # noqa: E402


def _impl(dcp_world_size: int, use_shardq: bool):
    impl = object.__new__(fims.FlashInferMLASparseImpl)
    impl.dcp_world_size = dcp_world_size
    impl.use_shardq = use_shardq
    return impl


@pytest.mark.parametrize(
    "dcp,shardq,replicated,expected",
    [(1, False, False, True), (2, False, False, False), (1, True, False, False),
     (1, False, True, False)],
)
def test_layer_skips_inplace_concat_only_when_fused(dcp, shardq, replicated, expected):
    layer = SimpleNamespace(impl=_impl(dcp, shardq))
    q_rep = torch.empty(1) if replicated else None
    assert MLAAttention._impl_quantizes_query(layer, q_rep) is expected
    # A backend without the hook keeps the in-place concat.
    assert MLAAttention._impl_quantizes_query(SimpleNamespace(impl=object()), None) is False


@pytest.mark.parametrize("tokens", [1, 333, 4096])
@pytest.mark.parametrize("q_scale", [0.05, 1.0])
def test_fused_query_bitwise_equals_inplace_path(tokens, q_scale):
    torch.manual_seed(tokens)
    heads, nope, rope, latent = 96, 128, 64, 512
    q = torch.randn(tokens, heads, nope + rope, device="cuda", dtype=torch.bfloat16)
    q[:7] *= 300.0  # saturating rows (|x| / scale > 448)
    w_uk_t = torch.randn(heads, nope, latent, device="cuda", dtype=torch.bfloat16)
    w_uk_t *= 0.05
    scale = torch.tensor([q_scale], device="cuda", dtype=torch.float32)
    q_nope, q_pe = q.split([nope, rope], dim=-1)

    # In-place path (mla_attention before this change).
    cat = q.new_empty(tokens, heads, latent + rope)
    torch.bmm(q_nope.transpose(0, 1), w_uk_t, out=cat[..., :latent].transpose(0, 1))
    cat[..., latent:].copy_(q_pe)
    q8, _ = ops.scaled_fp8_quant(cat.reshape(tokens, -1), scale)
    ref = fims._pad_query_heads(q8.view(tokens, heads, latent + rope), 128)

    # Fused path: bmm into the token-major latent buffer, one quant + pad kernel.
    ql_nope = q.new_empty(tokens, heads, latent)
    torch.bmm(q_nope.transpose(0, 1), w_uk_t, out=ql_nope.transpose(0, 1))
    out = fims.decode_query_quant_pad(ql_nope, q_pe, scale, 128)
    assert out is not None
    assert torch.equal(out.view(torch.uint8), ref.view(torch.uint8))
