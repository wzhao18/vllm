# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Row-count enumeration of the DSA indexer metadata warmup (DeepGEMM paged
MQA logits metadata). DeepGEMM is mocked, so no GPU."""

from types import SimpleNamespace
from unittest import mock

import pytest

from vllm.model_executor.warmup import dsa_indexer_metadata_warmup as w

pytestmark = pytest.mark.cpu_test

# DeepGEMM's SM100 metadata kernel picks its block size (the compiled variant)
# from the number of Q rows.
_SM100_CLASSES = ((1, 512), (513, 2048), (2049, 1 << 30))


def _classes_hit(counts):
    return {
        i for i, (lo, hi) in enumerate(_SM100_CLASSES) for c in counts if lo <= c <= hi
    }


@pytest.mark.parametrize("max_rows", [1, 7, 512, 513, 640, 2048, 2049, 16384])
def test_row_counts_reach_every_class_up_to_max(max_rows):
    counts = w.metadata_row_counts(max_rows)
    assert counts == sorted(set(counts))
    assert counts[0] == 1 and counts[-1] == max_rows
    assert all(1 <= c <= max_rows for c in counts)
    reachable = _classes_hit(range(1, max_rows + 1))
    assert _classes_hit(counts) == reachable
    # Each class boundary that max_rows crosses is hit on both sides.
    for edge in (512, 2048):
        if max_rows > edge:
            assert edge in counts and edge + 1 in counts


def test_row_counts_empty():
    assert w.metadata_row_counts(0) == []


def test_max_decode_rows_caps_by_budget_and_seqs():
    cfg = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16384, max_num_seqs=160)
    )
    assert w.max_decode_rows(cfg, 4) == 640
    cfg.scheduler_config.max_num_batched_tokens = 512
    assert w.max_decode_rows(cfg, 8) == 512


class _FakeBuilder:
    def __init__(self, block, sms, next_n, supports_varlen=False):
        self.kv_cache_spec = SimpleNamespace(num_states=block)
        self.num_sms = sms
        self.decode_threshold = next_n
        self.supports_varlen = supports_varlen
        self.device = "cpu"


def test_warmup_builds_each_distinct_builder_once_at_every_row_count():
    cfg = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8192, max_num_seqs=160)
    )
    b1 = _FakeBuilder(64, 148, 4)
    b2 = _FakeBuilder(64, 148, 4)  # same configuration: skipped
    b3 = _FakeBuilder(64, 148, 8)
    groups = [
        [SimpleNamespace(metadata_builders=[b1, object()])],
        [SimpleNamespace(metadata_builders=[b2, b3])],
    ]
    calls = []

    def fake_metadata(context_lens, block, sms, indices=None):
        calls.append((tuple(context_lens.shape), block, sms))

    with (
        mock.patch(
            "vllm.v1.attention.backends.mla.indexer.DeepseekV32IndexerMetadataBuilder",
            _FakeBuilder,
        ),
        mock.patch("vllm.utils.deep_gemm.get_paged_mqa_logits_metadata", fake_metadata),
        mock.patch("torch.accelerator.synchronize"),
    ):
        n = w.warmup_paged_mqa_logits_metadata(groups, cfg)
    assert n == len(calls)
    for next_n in (4, 8):
        rows_max = min(8192, 160 * next_n)
        for query_len in (1, next_n):
            rows = {
                s[0] * s[1]
                for s, _, _ in calls
                if s[1] == query_len and (query_len == 1 or query_len == next_n)
            }
            assert _classes_hit(rows) >= _classes_hit(range(1, rows_max + 1))
    assert {(blk, sms) for _, blk, sms in calls} == {(64, 148)}
    # b2 adds nothing: 2 builder configurations x 2 query lengths.
    expected = sum(len(w.metadata_row_counts(min(8192, 160 * nn))) * 2 for nn in (4, 8))
    assert n == expected


def test_warmup_varlen_builds_the_serving_form():
    """A varlen builder (DeepGEMM varlen paged MQA logits, as served): one
    (rows, 1) build per row count, with per-token request indices."""
    cfg = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8192, max_num_seqs=160)
    )
    b = _FakeBuilder(64, 148, 4, supports_varlen=True)
    groups = [[SimpleNamespace(metadata_builders=[b])]]
    calls = []

    def fake_metadata(context_lens, block, sms, indices=None):
        calls.append((tuple(context_lens.shape), block, sms, indices))

    with (
        mock.patch(
            "vllm.v1.attention.backends.mla.indexer.DeepseekV32IndexerMetadataBuilder",
            _FakeBuilder,
        ),
        mock.patch("vllm.utils.deep_gemm.get_paged_mqa_logits_metadata", fake_metadata),
        mock.patch("torch.accelerator.synchronize"),
    ):
        n = w.warmup_paged_mqa_logits_metadata(groups, cfg)
    counts = w.metadata_row_counts(min(8192, 160 * 4))
    assert n == len(calls) == len(counts)
    assert [c[0] for c in calls] == [(r, 1) for r in counts]
    for (rows, _), _, _, indices in calls:
        assert indices is not None and indices.shape == (rows,)
        assert indices.tolist() == [i // 4 for i in range(rows)]
