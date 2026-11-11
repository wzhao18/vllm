# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Token-row layouts that Kimi-K3.1's cache backends declare for their pages.
`test_dsa_indexer_fused.py` checks the indexer's against the key inserts."""

import pytest
import torch

from vllm.models.kimi_k3.common import dsa_cache
from vllm.models.kimi_k3.common.dsa_cache import (
    KimiK3IndexerBackend,
    indexer_cache_entry_bytes,
)
from vllm.v1.kv_cache_interface import KVRowLayout, MLAAttentionSpec, SparseCacheRole

pytestmark = pytest.mark.cpu_test

BLOCK_SIZE = 640  # The packed K3.1 block at TP16.


def _indexer_spec(entry_bytes: int) -> MLAAttentionSpec:
    return MLAAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=entry_bytes,
        dtype=torch.uint8,
        cache_role=SparseCacheRole.INDEXER,
    )


@pytest.mark.parametrize("use_fp4", [False, True])
def test_indexer_pages_store_values_then_scales(use_fp4):
    """Each 64-row kernel page holds its rows' key values, then their 4 scale
    bytes (one fp32 scale, or four UE8M0 scales of a 128-dim MXFP4 key)."""
    entry_bytes = indexer_cache_entry_bytes(128, use_fp4)
    value_bytes = entry_bytes - 4
    spec = _indexer_spec(entry_bytes)
    layout = KimiK3IndexerBackend.get_kv_cache_row_layout(spec, 64)
    assert layout == KVRowLayout(64, ((0, value_bytes), (64 * value_bytes, 4)))
    layout.validate(spec)
    # Row 70 is row 6 of the second kernel page.
    page = 64 * entry_bytes
    assert layout.row_byte_ranges(70) == [
        (page + 6 * value_bytes, value_bytes),
        (page + 64 * value_bytes + 6 * 4, 4),
    ]


def test_indexer_kernel_pages_follow_the_kernel_block():
    spec = _indexer_spec(indexer_cache_entry_bytes(128, False))
    layout = KimiK3IndexerBackend.get_kv_cache_row_layout(spec, 128)
    assert layout.kernel_page_rows == 128
    layout.validate(spec)


def test_indexer_rows_undeclared_on_rocm(monkeypatch):
    monkeypatch.setattr(dsa_cache.current_platform, "is_rocm", lambda: True)
    spec = _indexer_spec(indexer_cache_entry_bytes(128, False))
    assert KimiK3IndexerBackend.get_kv_cache_row_layout(spec, 64) is None


def test_row_layout_validation():
    spec = MLAAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=656,
        dtype=torch.uint8,
        cache_dtype_str="fp8_ds_mla",
    )
    KVRowLayout.token_major(656).validate(spec)
    with pytest.raises(ValueError, match="does not tile a page"):
        KVRowLayout.token_major(576).validate(spec)
    with pytest.raises(ValueError, match="does not tile its kernel page"):
        KVRowLayout(64, ((0, 600), (64 * 600 + 8, 56))).validate(spec)
    # A kernel page that does not divide the block.
    with pytest.raises(ValueError, match="does not tile a page"):
        KVRowLayout(256, ((0, 600), (256 * 600, 56))).validate(spec)
