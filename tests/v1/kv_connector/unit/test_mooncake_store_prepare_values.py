# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ChunkedTokenDatabase.prepare_values."""

import random

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    ChunkedTokenDatabase,
    KeyMetadata,
)
from vllm.utils.math_utils import cdiv

BLOCK_SIZE = 128


def _reference_prepare_value(
    db: ChunkedTokenDatabase, start: int, end: int, block_ids: list[int]
) -> tuple[list[int], list[int], int]:
    """Compute a token range with the original scalar implementation."""
    addr_list = []
    size_list = []
    first_block = start // db.block_size
    block_count = cdiv(end - start, db.block_size)
    chunk_block_ids = block_ids[first_block : first_block + block_count]
    block_id = chunk_block_ids[0]
    length = len(db.block_len)
    consecutive = all(
        right == left + 1
        for left, right in zip(chunk_block_ids, chunk_block_ids[1:])
    )
    if consecutive:
        for index, base_addr in enumerate(db.kv_caches_base_addr):
            addr_list.append(base_addr + block_id * db.block_len[index % length])
            size_list.append(db.block_len[index % length] * block_count)
    else:
        for current_block_id in chunk_block_ids:
            for index, base_addr in enumerate(db.kv_caches_base_addr):
                addr_list.append(
                    base_addr + current_block_id * db.block_len[index % length]
                )
                size_list.append(db.block_len[index % length])
    return addr_list, size_list, block_id


def _make_db(num_regions: int, num_block_lens: int) -> ChunkedTokenDatabase:
    md = KeyMetadata(model_name="t", tp_rank=1, pcp_rank=0, dcp_rank=0, pp_rank=0)
    db = ChunkedTokenDatabase(md, BLOCK_SIZE)
    db.set_kv_caches_base_addr(
        [0x7F00_0000_0000 + i * (1 << 30) for i in range(num_regions)]
    )
    # Exercise repeated block lengths when there are more cache regions.
    db.set_block_len([30_208 + 512 * i for i in range(num_block_lens)])
    return db


@pytest.mark.parametrize("num_regions,num_block_lens", [(96, 96), (96, 2), (1, 1)])
def test_prepare_values_matches_reference(num_regions: int, num_block_lens: int):
    db = _make_db(num_regions, num_block_lens)
    rng = random.Random(0)
    n_blocks = 300
    block_ids = [rng.randrange(0, 1 << 20) for _ in range(n_blocks)]
    chunks = []
    b = 0
    while b < n_blocks - 4:
        span = rng.choice([1, 1, 1, 2, 4])
        chunks.append((b * BLOCK_SIZE, (b + span) * BLOCK_SIZE))
        b += span + rng.choice([0, 1])

    addrs, sizes, bids = db.prepare_values(chunks, block_ids)
    assert len(addrs) == len(sizes) == len(bids) == len(chunks)
    for (start, end), addr, size, bid in zip(chunks, addrs, sizes, bids):
        ref_addr, ref_size, ref_bid = _reference_prepare_value(
            db, start, end, block_ids
        )
        assert addr == ref_addr
        assert size == ref_size
        assert bid == ref_bid
        # Native bindings require Python ints rather than numpy scalars.
        assert all(type(a) is int for a in addr)
        assert type(bid) is int


def test_prepare_value_single_matches_reference():
    db = _make_db(8, 8)
    block_ids = list(range(64))
    got = db.prepare_value(5 * BLOCK_SIZE, 7 * BLOCK_SIZE, block_ids)
    assert got == _reference_prepare_value(
        db, 5 * BLOCK_SIZE, 7 * BLOCK_SIZE, block_ids
    )


def test_prepare_values_empty():
    db = _make_db(4, 4)
    assert db.prepare_values([], [1, 2, 3]) == ([], [], [])


def test_prepare_values_rejects_unaligned_chunk():
    db = _make_db(4, 4)
    with pytest.raises(AssertionError):
        db.prepare_values([(0, BLOCK_SIZE + 1)], [0, 1])


def test_prepare_values_separates_block_stride_from_transfer_length():
    db = ChunkedTokenDatabase(
        KeyMetadata(model_name="t", tp_rank=0, pcp_rank=0, dcp_rank=0, pp_rank=0),
        BLOCK_SIZE,
    )
    db.set_kv_cache_regions(
        base_addrs=[0x1000, 0x8000],
        block_strides=[0x100, 0x200],
        transfer_lens=[0xC0, 0x180],
    )

    addrs, sizes, block_ids = db.prepare_values(
        [(0, BLOCK_SIZE), (BLOCK_SIZE, 2 * BLOCK_SIZE)], [3, 9]
    )

    assert block_ids == [3, 9]
    assert addrs == [[0x1300, 0x8600], [0x1900, 0x9200]]
    assert sizes == [[0xC0, 0x180], [0xC0, 0x180]]


def test_compact_regions_reject_multiblock_value():
    db = ChunkedTokenDatabase(
        KeyMetadata(model_name="t", tp_rank=0, pcp_rank=0, dcp_rank=0, pp_rank=0),
        BLOCK_SIZE,
    )
    db.set_kv_cache_regions(
        base_addrs=[0x1000],
        block_strides=[0x100],
        transfer_lens=[0xC0],
    )
    with pytest.raises(ValueError, match="exactly one database block"):
        db.prepare_values([(0, 2 * BLOCK_SIZE)], [3, 9])


def test_compact_regions_round_trip_noncontiguous_blocks_and_preserve_padding():
    db = ChunkedTokenDatabase(
        KeyMetadata(model_name="t", tp_rank=0, pcp_rank=0, dcp_rank=0, pp_rank=0),
        BLOCK_SIZE,
    )
    db.set_kv_cache_regions(
        base_addrs=[0, 0x1000],
        block_strides=[32, 48],
        transfer_lens=[24, 40],
    )
    source = bytearray(0x2000)
    for index in range(len(source)):
        source[index] = index % 251
    restored = bytearray([0xA5] * len(source))

    addrs, sizes, _ = db.prepare_values(
        [(0, BLOCK_SIZE), (BLOCK_SIZE, 2 * BLOCK_SIZE)], [5, 1]
    )
    value = b"".join(
        source[addr : addr + size]
        for chunk_addrs, chunk_sizes in zip(addrs, sizes, strict=True)
        for addr, size in zip(chunk_addrs, chunk_sizes, strict=True)
    )
    offset = 0
    for chunk_addrs, chunk_sizes in zip(addrs, sizes, strict=True):
        for addr, size in zip(chunk_addrs, chunk_sizes, strict=True):
            restored[addr : addr + size] = value[offset : offset + size]
            offset += size

    for chunk_addrs, chunk_sizes in zip(addrs, sizes, strict=True):
        for addr, size in zip(chunk_addrs, chunk_sizes, strict=True):
            assert restored[addr : addr + size] == source[addr : addr + size]
    for block_id in (5, 1):
        assert restored[block_id * 32 + 24 : (block_id + 1) * 32] == bytes(
            [0xA5] * 8
        )
        assert restored[
            0x1000 + block_id * 48 + 40 : 0x1000 + (block_id + 1) * 48
        ] == bytes([0xA5] * 8)


def test_set_kv_cache_regions_rejects_invalid_geometry():
    db = ChunkedTokenDatabase(
        KeyMetadata(model_name="t", tp_rank=0, pcp_rank=0, dcp_rank=0, pp_rank=0),
        BLOCK_SIZE,
    )
    with pytest.raises(ValueError, match="lengths must match"):
        db.set_kv_cache_regions([0x1000], [256, 256], [128])
    with pytest.raises(ValueError, match="cannot exceed"):
        db.set_kv_cache_regions([0x1000], [128], [256])
