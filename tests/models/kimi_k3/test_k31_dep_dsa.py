# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DSA-DP SP decode's DSA layer (``dep_dsa.DepDsa``): packing turns the stock
linears into views of one weight (same values, so the stock forward of other steps and
a weight reload keep working), a layer it cannot reproduce is refused at build, and
``ops.dsa_dp.run_segment`` calls it as it calls the stock module."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.models.kimi_k3.nvidia.specialized_decode.runtime import dep_dsa
from vllm.models.kimi_k3.nvidia.specialized_decode.shapes import Unsupported


def _linear(n, k, seed):
    g = torch.Generator().manual_seed(seed)
    linear = torch.nn.Module()
    weight = torch.randn(n, k, generator=g).bfloat16()
    linear.weight = torch.nn.Parameter(weight, requires_grad=False)
    linear.quant_method = UnquantizedLinearMethod()
    linear.bias = None
    return linear


def test_pack_linears_makes_views_of_one_weight():
    a, b = _linear(24, 64, 0), _linear(8, 64, 1)
    want = torch.cat([a.weight.data.clone(), b.weight.data.clone()])
    packed = dep_dsa.pack_linears([a, b], "test")
    assert torch.equal(packed, want)
    assert a.weight.data_ptr() == packed.data_ptr()
    assert b.weight.data_ptr() == packed[24:].data_ptr()
    packed.zero_()  # a reload refills the packed storage through the linears
    assert not a.weight.any() and not b.weight.any()


def test_pack_linears_refuses_mixed_k_and_quantized():
    with pytest.raises(Unsupported, match="K"):
        dep_dsa.pack_linears([_linear(8, 64, 0), _linear(8, 32, 1)], "test")
    quantized = _linear(8, 64, 2)
    quantized.quant_method = object()
    with pytest.raises(Unsupported, match="unquantized"):
        dep_dsa.pack_linears([quantized], "test")


def test_refuses_a_tp_sharded_dsa_layer():
    wrapper = SimpleNamespace(
        prefix="model.layers.3.self_attn", rotary_emb=None, g_proj=object(),
        dcp_q_replicate=False, skip_topk=False, is_sparse=True,
    )  # fmt: skip
    indexer = SimpleNamespace(index_rope_dim=0, dcp_world_size=1)
    attention = SimpleNamespace(
        mla_attn=wrapper, dsa_indexer=indexer, local_full_heads=False,
        q_lora_rank=1536, gemm_rs_ar=None, _packed=None,
        o_proj=SimpleNamespace(reduce_results=False),
    )  # fmt: skip
    with pytest.raises(Unsupported, match="not all heads"):
        dep_dsa.DepDsa(attention)


def test_run_segment_calls_it_as_the_stock_module(monkeypatch):
    calls = []
    monkeypatch.setattr(
        dep_dsa.DepDsa, "run", lambda self, x, positions: calls.append((x, positions))
    )
    dep = object.__new__(dep_dsa.DepDsa)
    dep(hidden_states="x", positions="p")
    assert calls == [("x", "p")]
