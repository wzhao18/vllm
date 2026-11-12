# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The lookup server's batched key building and result check match the
per-hash reference (same keys in the same order, same present set)."""

import dataclasses
import hashlib
import random
import threading

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import (
    worker as msw,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    BlobBlockHashes,
    ChunkedTokenDatabase,
    KeyMetadata,
    PoolKey,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.metrics import (
    MooncakeStoreConnectorStats,
)
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
)

UNIT = 16


def _hashes(n: int, seed: int = 0, hash_len: int = 32) -> list[bytes]:
    out, h = [], seed.to_bytes(8, "little")
    for i in range(n):
        h = hashlib.sha256(h + i.to_bytes(4, "little")).digest()[:hash_len]
        out.append(h)
    return out


@pytest.mark.parametrize("hash_len", [8, 32])
def test_blob_hex_and_bytes_matches_per_item(hash_len):
    hashes = _hashes(37, hash_len=hash_len)
    blob = BlobBlockHashes(memoryview(b"".join(hashes)), hash_len)
    for start, stop in [(0, 37), (5, 6), (3, 30), (36, 37), (10, 10), (20, 5),
                        (-3, 4), (30, 99)]:
        hexes, raws = blob.hex_and_bytes(start, stop)
        ids = range(max(0, start), min(37, stop))
        assert raws == [bytes(blob[i]) for i in ids]
        assert hexes == [blob[i].hex() for i in ids]
        assert all(type(r) is bytes for r in raws)


def test_build_key_strings_is_hash_major():
    prefixes = ["p@tp_rank:0", "p@tp_rank:1", "p@tp_rank:2"]
    hexes = ["aa", "bb"]
    assert PoolKey.build_key_strings(prefixes, hexes) == [
        PoolKey.build_key_string(p, h) for h in hexes for p in prefixes
    ]


def _make_worker(*, dcp: int, fine_grained: bool, mamba: bool):
    w = object.__new__(msw.MooncakeStoreWorker)
    w._capacity_only = False
    w._selective_lease = None
    w.block_size = 4 * UNIT
    w.hash_block_size = UNIT
    w.tp_size = 4
    w.dcp_size = dcp
    w.pcp_size = 1
    w.pp_size = 1
    w.tp_rank = 0
    w.num_kv_head = 2
    w.store_tp_size = None
    w._kv_connector_stats_lock = threading.Lock()
    w.kv_connector_stats = MooncakeStoreConnectorStats()
    groups = [
        KVCacheGroupSpec(
            ["mla"],
            MLAAttentionSpec(
                block_size=4 * UNIT, num_kv_heads=1, head_size=8,
                dtype=torch.uint8,
            ),
        ),
        KVCacheGroupSpec(
            ["gqa"],
            FullAttentionSpec(
                block_size=2 * UNIT, num_kv_heads=8, head_size=8,
                dtype=torch.uint8,
            ),
        ),
    ]
    if mamba:
        groups.append(
            KVCacheGroupSpec(
                ["kda"],
                MambaSpec(
                    block_size=4 * UNIT, shapes=((1, 1),),
                    dtypes=(torch.float32,), mamba_cache_mode="align",
                ),
            )
        )
    w._kv_cache_groups = groups
    w.coord = msw.MooncakeStoreCoordinator(
        groups,
        scheduler_block_size=4 * UNIT,
        hash_block_size=UNIT,
        enable_partial_hash_hits=fine_grained,
    )
    w._group_tp_replication_factors = w._compute_group_tp_replication_factors()
    meta = KeyMetadata("m", 0, 0, 0, 0, cache_prefix="pfx")
    w.token_dbs = [
        ChunkedTokenDatabase(
            dataclasses.replace(meta, group_id=g), g_spec.kv_cache_spec.block_size,
            hash_block_size=UNIT,
        )
        for g, g_spec in enumerate(groups)
    ]
    w._init_lookup_key_prefixes()
    return w


def _reference(w, num_tokens, block_hashes, known_prefix_tokens, res_fn):
    """The per-hash key loop and result check the server used before."""
    token_len = w.coord.align_lookup_length(num_tokens)
    fine = w.coord.enable_partial_hash_hits
    masks = None if fine else w.coord.lookup_mask(token_len)
    start_unit = 0
    if fine and known_prefix_tokens > 0:
        start_unit = (
            known_prefix_tokens // w.block_size * w.block_size
        ) // w.hash_block_size
    keys, meta, trusted = [], [], set()
    for g_idx, db in enumerate(w.token_dbs):
        spec = w._kv_cache_groups[g_idx].kv_cache_spec
        prefixes = w._lookup_key_prefixes[g_idx]
        if fine:
            max_units = min(len(block_hashes), token_len // w.hash_block_size)
            first = min(start_unit, max_units)
            ids = range(first, max_units)
            gh = block_hashes
            if first and not msw._has_state_spec(spec):
                trusted.update((g_idx, bytes(block_hashes[u])) for u in range(first))
        else:
            mask = masks[g_idx]
            gh = w.coord.block_hashes_for_spec(block_hashes, spec)
            mc = min(len(gh), -(-token_len // db.block_size))
            lim = mc if mask is None else min(mc, len(mask))
            ids = [c for c in range(lim) if mask is None or mask[c]]
        for c in ids:
            h = gh[c]
            for p in prefixes:
                keys.append(PoolKey.build_key_string(p, h.hex()))
            meta.append((g_idx, bytes(h)))
    res = res_fn(keys)
    exists, pos = set(trusted), 0
    for g_idx, hb in meta:
        cnt = len(w._lookup_key_prefixes[g_idx])
        if all(res[pos + j] == 1 for j in range(cnt)):
            exists.add((g_idx, hb))
        pos += cnt
    return keys, exists


@pytest.mark.parametrize("dcp", [1, 4])
@pytest.mark.parametrize("fine_grained", [True, False])
@pytest.mark.parametrize("mamba", [True, False])
@pytest.mark.parametrize("blob", [True, False])
@pytest.mark.parametrize("known", [0, 4 * UNIT + 3, 9 * UNIT])
def test_lookup_keys_and_present_set_match_reference(
    dcp, fine_grained, mamba, blob, known
):
    if known and not fine_grained:
        known = 0
    w = _make_worker(dcp=dcp, fine_grained=fine_grained, mamba=mamba)
    rng = random.Random(dcp * 131 + fine_grained * 7 + mamba * 3 + known)
    raw = _hashes(23, seed=dcp + known)
    num_tokens = 23 * UNIT - 5
    block_hashes = (
        BlobBlockHashes(memoryview(b"".join(raw)), 32)
        if blob
        else [BlockHash(h) for h in raw]
    )
    # Mostly present, some single-namespace holes, some error codes.
    answers: dict[str, int] = {}

    def res_fn(keys):
        for k in keys:
            if k not in answers:
                answers[k] = rng.choice([1] * 12 + [0, -1])
        return [answers[k] for k in keys]

    exp_keys, exp_exists = _reference(w, num_tokens, block_hashes, known, res_fn)

    class Store:
        calls: list[list[str]] = []

        def batch_is_exist(self, keys):
            self.calls.append(list(keys))
            return res_fn(keys)

    w.store = Store()
    seen = {}
    real_longest = w._longest_hit

    def spy(num_tokens, token_len, bh, kpt, exists_set):
        seen["exists"] = set(exists_set)
        return real_longest(num_tokens, token_len, bh, kpt, exists_set)

    w._longest_hit = spy
    w.lookup(num_tokens, block_hashes, known)
    assert w.store.calls[0] == exp_keys
    assert seen["exists"] == exp_exists


def test_lookup_short_store_answer_is_a_reported_failure():
    w = _make_worker(dcp=4, fine_grained=True, mamba=False)
    raw = _hashes(4)

    class Store:
        def batch_is_exist(self, keys):
            return [1] * (len(keys) - 1)

    w.store = Store()
    with pytest.raises(ValueError):
        w.lookup(4 * UNIT, [BlockHash(h) for h in raw], 0)
