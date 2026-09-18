# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Context cast/concat contract, including V view layout and fallback dispatch."""

from types import MethodType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.model_executor.layers.attention import mla_attention
from vllm.platforms import current_platform
from vllm.v1.attention.ops.mla_context_cast import fused_context_cast_concat

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.is_device_capability(103),
    reason="The fused context conversion is currently enabled only on SM103",
)


def _impl():
    result = SimpleNamespace(
        _use_flashinfer_concat_mla_k=False,
        qk_nope_head_dim=128,
        v_head_dim=128,
        qk_rope_head_dim=64,
    )
    result._concat_k_nope_k_pe = MethodType(
        mla_attention.MLACommonBaseImpl._concat_k_nope_k_pe, result
    )
    result.prepare = MethodType(
        mla_attention.MLACommonBaseImpl._prepare_prefill_context_kv, result
    )
    return result


def _original(impl, kv, rope, dtype):
    if dtype is not None:
        kv, rope = kv.to(dtype), rope.to(dtype)
    nope, value = kv.split((128, 128), dim=-1)
    return impl._concat_k_nope_k_pe(nope, rope), value


def _exact(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    integer_dtype = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
    bits = integer_dtype[actual.element_size()]
    assert torch.equal(actual.view(bits), expected.view(bits))


def _check(impl, kv, rope, expected_fused, dtype=torch.float8_e4m3fn):
    before_kv, before_rope = kv.clone(), rope.clone()
    expected = _original(impl, kv, rope, dtype)
    with patch.object(
        mla_attention, "fused_context_cast_concat", wraps=fused_context_cast_concat
    ) as spy:
        actual = impl.prepare(kv, rope, dtype)
        assert spy.call_count == int(expected_fused)
    for result, reference in zip(actual, expected):
        _exact(result, reference)
    _exact(kv, before_kv)
    _exact(rope, before_rope)
    assert actual[0].is_contiguous()
    assert actual[1].stride() == expected[1].stride()
    assert actual[1].storage_offset() == expected[1].storage_offset()
    return actual


@pytest.mark.parametrize("heads", [1, 4, 12, 16])
@pytest.mark.parametrize("tokens", [1, 128, 4096])
@pytest.mark.parametrize("strided", [False, True])
def test_context_cast_preserves_bytes_layout_and_inputs(heads, tokens, strided):
    if strided:
        kv = torch.randn((tokens, heads, 514), device="cuda", dtype=torch.bfloat16)[
            ..., 1:513:2
        ]
        rope = torch.randn((tokens, 1, 130), device="cuda", dtype=torch.bfloat16)[
            ..., 1:129:2
        ]
    else:
        kv = torch.randn((tokens, heads, 256), device="cuda", dtype=torch.bfloat16)
        rope = torch.randn((tokens, 1, 64), device="cuda", dtype=torch.bfloat16)
    impl = _impl()
    _check(impl, kv, rope, True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = impl.prepare(kv, rope, torch.float8_e4m3fn)
    kv.add_(0.125)
    rope.sub_(0.0625)
    before_kv, before_rope = kv.clone(), rope.clone()
    graph.replay()
    expected = _original(impl, kv, rope, torch.float8_e4m3fn)
    for result, reference in zip(actual, expected):
        _exact(result, reference)
    _exact(kv, before_kv)
    _exact(rope, before_rope)


def test_context_cast_all_bf16_patterns():
    bits = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.int16)
    kv = torch.stack((bits.view(512, 128), bits.view(512, 128)), dim=1)
    kv = kv.reshape(512, 1, 256).repeat(2, 1, 1).view(torch.bfloat16)
    rope = bits.view(torch.bfloat16).view(1024, 1, 64)
    _check(_impl(), kv, rope, True)


@pytest.mark.parametrize(
    "case", ["no_cast", "bf16_output", "fp16_input", "permuted", "empty", "hardware"]
)
def test_context_cast_unsupported_inputs_keep_original_path(case):
    impl = _impl()
    kv = torch.randn((128, 4, 256), device="cuda", dtype=torch.bfloat16)
    rope = torch.randn((128, 1, 64), device="cuda", dtype=torch.bfloat16)
    dtype = torch.float8_e4m3fn
    if case == "no_cast":
        dtype = None
    elif case == "bf16_output":
        dtype = torch.bfloat16
    elif case == "fp16_input":
        kv, rope = kv.half(), rope.half()
    elif case == "permuted":
        kv = torch.randn((4, 128, 256), device="cuda", dtype=torch.bfloat16).transpose(
            0, 1
        )
    elif case == "empty":
        kv, rope = kv[:0], rope[:0]
    elif case == "hardware":
        with patch.object(current_platform, "is_device_capability", return_value=False):
            _check(impl, kv, rope, False, dtype)
        return
    _check(impl, kv, rope, False, dtype)


def test_context_cast_retains_flashinfer_precedence():
    impl = _impl()
    impl._use_flashinfer_concat_mla_k = True
    kv = torch.randn((128, 128, 256), device="cuda", dtype=torch.bfloat16)
    rope = torch.randn((128, 1, 64), device="cuda", dtype=torch.bfloat16)
    _check(impl, kv, rope, False)
