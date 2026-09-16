# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.attention.backends.mla.kimi_k3_nvfp4_mla import (
    KimiK3NVFP4MLABackend,
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
