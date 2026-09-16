# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.attention.backends.mla.kimi_k3_nvfp4_mla import (
    KimiK3NVFP4MLABackend,
    _expand_dcp_causal_seq_lens,
)
from vllm.v1.kv_cache_interface import (
    MLAAttentionSpec,
    get_kv_quant_mode,
)


def test_kimi_k3_nvfp4_cache_spec_uses_native_layout() -> None:
    spec = MLAAttentionSpec(
        block_size=1,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.uint8,
        cache_dtype_str="nvfp4_kimi_k3",
        kv_quant_mode=get_kv_quant_mode("nvfp4_kimi_k3"),
    )

    native_spec = KimiK3NVFP4MLABackend.customize_spec(spec)

    assert native_spec.state_content_bytes == 392
    assert native_spec.storage_block_size == 128
    assert native_spec.page_size_bytes == 392
    assert KimiK3NVFP4MLABackend.get_supported_kernel_block_sizes() == [128]


def test_dcp_multi_token_causal_lengths_are_localized_per_query() -> None:
    # With world size 2, rank 1 owns global token indices 1, 3, 5, ... .
    # Localizing final length 10 before subtracting [2, 1, 0] would incorrectly
    # produce [3, 4, 5]. Localize each global bound independently instead.
    global_final_lens = torch.tensor([10], dtype=torch.int32)
    offsets = torch.arange(3, dtype=torch.int32)
    output = torch.empty(3, dtype=torch.int32)

    actual = _expand_dcp_causal_seq_lens(
        global_final_lens,
        query_len=3,
        dcp_world_size=2,
        dcp_rank=1,
        interleave_size=1,
        token_offsets=offsets,
        output=output,
    )

    assert actual.tolist() == [4, 4, 5]
    assert actual.data_ptr() == output.data_ptr()
