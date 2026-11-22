# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P/D block-hash handover: the KV consumer adopts the producer's prompt block
hashes only when they are exactly what it computes, and says so when it does not."""

import base64
import logging

import pytest

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core import kv_cache_utils as kcu
from vllm.v1.request import Request

pytestmark = pytest.mark.cpu_test

BS = 16


@pytest.fixture(autouse=True)
def _none_hash():
    kcu.init_none_hash(sha256)
    kcu._remote_hash_counts.clear()


def _req(tokens, params=None, block_size=BS, cache_salt=None, rid="r"):
    sp = SamplingParams(max_tokens=4)
    if params is not None:
        sp.extra_args = {"kv_transfer_params": params}
    return Request(
        request_id=rid,
        prompt_token_ids=list(tokens),
        sampling_params=sp,
        pooling_params=None,
        cache_salt=cache_salt,
        block_hasher=kcu.get_request_block_hasher(block_size, sha256),
    )


def _handover(producer, **extra):
    return {"do_remote_prefill": True, **kcu.export_block_hashes(producer, BS, sha256), **extra}


def _count_hashes(monkeypatch):
    calls = []
    real = kcu.hash_block_tokens

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(kcu, "hash_block_tokens", counting)
    return calls


def test_consumer_adopts_exact_hashes_and_hashes_only_the_witness(monkeypatch):
    tokens = list(range(1000, 1000 + 10 * BS + 5))
    producer = _req(tokens)
    params = _handover(producer)
    calls = _count_hashes(monkeypatch)
    consumer = _req(tokens, params)
    assert consumer.block_hashes == producer.block_hashes
    assert len(calls) == 1  # the witness block only
    assert kcu._remote_hash_counts == {"exported": 1, "adopted": 1}


def test_longer_consumer_prompt_hashes_the_extra_block_locally(monkeypatch):
    tokens = list(range(5000, 5000 + 4 * BS))
    producer = _req(tokens)
    # The consumer's prompt also carries the producer's first token and fills
    # one more block.
    longer = tokens + list(range(9000, 9000 + BS))
    calls = _count_hashes(monkeypatch)
    consumer = _req(longer, _handover(producer))
    assert consumer.block_hashes == _req(longer).block_hashes
    assert consumer.block_hashes[:4] == producer.block_hashes
    assert len(calls) == 1 + 1 + 5  # witness + the new block, + the reference request
    assert kcu._remote_hash_counts["adopted"] == 1


def test_only_prompt_blocks_are_exported():
    producer = _req(range(3 * BS + 3))
    producer.append_output_token_ids(list(range(BS)))  # output fills a 4th block
    assert len(producer.block_hashes) == 4
    raw = base64.b64decode(kcu.export_block_hashes(producer, BS, sha256)["block_hashes"])
    assert len(raw) == 3 * len(kcu.NONE_HASH)


@pytest.mark.parametrize(
    "mutate, reason",
    [
        (lambda p: p.update(block_hash_fp=p["block_hash_fp"].replace(f"/{BS}/", "/32/")),
         "fingerprint mismatch"),
        (lambda p: p.update(block_hashes=p["block_hashes"][:-4] + "AAAA"), "witness mismatch"),
        (lambda p: p.update(block_hashes="not base64!"), "malformed"),
    ],
)
def test_mismatch_falls_back_to_local_hashes_and_warns(mutate, reason, caplog):
    tokens = list(range(7, 7 + 6 * BS))
    producer = _req(tokens)
    params = _handover(producer)
    mutate(params)
    with caplog.at_level(logging.WARNING):
        consumer = _req(tokens, params)
    assert consumer.block_hashes == producer.block_hashes  # recomputed, exact
    assert kcu._remote_hash_counts == {"exported": 1, reason: 1}
    assert reason in caplog.text
    assert f"fallback 1 ({reason} 1)" in kcu._remote_hash_summary()


def test_different_cache_salt_is_never_adopted():
    tokens = list(range(3, 3 + 4 * BS))
    producer = _req(tokens, cache_salt="a")
    consumer = _req(tokens, _handover(producer), cache_salt="b")
    assert consumer.block_hashes == _req(tokens, cache_salt="b").block_hashes
    assert kcu._remote_hash_counts == {"exported": 1, "fingerprint mismatch": 1}


def test_more_hashes_than_consumer_blocks_is_rejected():
    producer = _req(range(8 * BS))
    consumer = _req(range(4 * BS), _handover(producer))
    assert consumer.block_hashes == _req(range(4 * BS)).block_hashes
    assert kcu._remote_hash_counts == {"exported": 1, "more blocks than tokens": 1}


def test_adoption_is_logged_positively(caplog):
    tokens = list(range(2 * BS))
    with caplog.at_level(logging.INFO):
        # 500 handovers = 1000 events (an export and an adoption each): the
        # summary is logged at the first event and again at the 1000th.
        for i in range(500):
            _req(tokens, _handover(_req(tokens, rid=f"p{i}")), rid=f"d{i}")
    summary = kcu._remote_hash_summary()
    assert summary == (
        "pd-block-hashes: adopted 500 / fallback 0 (); exported 500, not exported 0"
    )
    lines = [r.getMessage() for r in caplog.records if "pd-block-hashes" in r.getMessage()]
    assert lines[0] == "pd-block-hashes: adopted 0 / fallback 0 (); exported 1, not exported 0"
    assert lines[-1] == summary and len(lines) == 2


def test_requests_without_handover_are_unchanged():
    tokens = list(range(5 * BS))
    assert _req(tokens, {"do_remote_prefill": True}).block_hashes == _req(tokens).block_hashes
    assert kcu._remote_hash_counts == {}
