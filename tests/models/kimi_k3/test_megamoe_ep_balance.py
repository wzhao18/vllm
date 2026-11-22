# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Served EP-balance ring (megamoe_rubin/ep_balance.py): slot layout, drain order, per-rank loads."""

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="pinned host memory needs CUDA")


def _write(ring, call, sizes, world, epr, t=123456789):
    from vllm.models.kimi_k3.nvidia.megamoe_rubin import ep_balance as E

    i = (call >> E.SHIFT) % E.SLOTS
    base = i * E.SLOT_WORDS
    buf = ring.buffer
    buf[base + E.HEADER_WORDS : base + E.HEADER_WORDS + len(sizes)] = torch.tensor(sizes, dtype=torch.int32)
    buf[base + 1] = t & 0x7FFFFFFF
    buf[base + 2] = 0
    buf[base + 3] = len(sizes)
    buf[base + 4] = world
    buf[base + 5] = epr
    buf[base + 6] = 0  # global record
    buf[base + 7] = E.epbal_checksum(call, sum(sizes))
    buf[base] = call


@cuda
def test_drain_returns_completed_calls_in_order_with_rank_loads():
    from vllm.models.kimi_k3.nvidia.megamoe_rubin import ep_balance as E

    ring = E.EpBalanceRing("test")
    assert ring.drain() == []
    # 4 ranks x 2 experts: rank loads [3, 7, 0, 6] -> mean 4, max/mean 1.75
    _write(ring, 64, [1, 2, 3, 4, 0, 0, 5, 1], world=4, epr=2)
    _write(ring, 128, [2, 2, 2, 2, 2, 2, 2, 2], world=4, epr=2)
    out = ring.drain()
    assert [r["call"] for r in out] == [64, 128]
    assert out[0]["rank_loads"] == [3, 7, 0, 6] and out[0]["routes"] == 16
    assert out[0]["max_over_mean"] == pytest.approx(1.75) and out[0]["expert_max"] == 5
    assert out[1]["max_over_mean"] == pytest.approx(1.0)
    assert ring.drain() == []  # nothing new


@cuda
def test_slot_being_rewritten_waits_for_the_next_drain():
    from vllm.models.kimi_k3.nvidia.megamoe_rubin import ep_balance as E

    ring = E.EpBalanceRing("test")
    _write(ring, 64, [1, 1], world=2, epr=1)
    assert [r["call"] for r in ring.drain()] == [64]
    # call 128's slot: payload written, call word not yet (the kernel writes it last)
    i = (128 >> E.SHIFT) % E.SLOTS
    ring.buffer[i * E.SLOT_WORDS + E.HEADER_WORDS] = 9
    assert ring.drain() == []
    _write(ring, 128, [9, 1], world=2, epr=1)
    out = ring.drain()
    assert [r["call"] for r in out] == [128] and out[0]["rank_loads"] == [9, 1]


@cuda
def test_global_record_without_matching_checksum_waits():
    """The dedup router writes global records without a system fence: a slot whose call id is visible but whose
    counts (or checksum) are still landing fails the checksum and is taken on a later drain."""
    from vllm.models.kimi_k3.nvidia.megamoe_rubin import ep_balance as E

    ring = E.EpBalanceRing("test")
    _write(ring, 64, [4, 4], world=2, epr=1)
    i = (64 >> E.SHIFT) % E.SLOTS
    ring.buffer[i * E.SLOT_WORDS + E.HEADER_WORDS + 1] = 3  # one count not landed yet
    assert ring.drain() == []
    ring.buffer[i * E.SLOT_WORDS + E.HEADER_WORDS + 1] = 4
    out = ring.drain()
    assert [r["call"] for r in out] == [64] and out[0]["routes"] == 8


def test_checksum_matches_device_int32_arithmetic():
    from vllm.models.kimi_k3.nvidia.megamoe_rubin import ep_balance as E

    def dev(call, total):  # (total ^ SALT) + call * MUL in wrapping int32, as the kernel computes it
        v = (((total ^ 0x5BD1E995) & 0xFFFFFFFF) + ((call * 65599) & 0xFFFFFFFF)) & 0xFFFFFFFF
        return v - (1 << 32) if v >= 1 << 31 else v

    for call, total in ((64, 0), (1 << 30, 123456), (64 * 999_999, 2_000_000_000)):
        assert E.epbal_checksum(call, total) == dev(call, total)
        assert -(1 << 31) <= E.epbal_checksum(call, total) < 1 << 31


@cuda
def test_lapped_slot_counts_lost_calls_and_layer_index():
    from vllm.models.kimi_k3.nvidia.megamoe_rubin import ep_balance as E

    E.configure(4)
    ring = E.EpBalanceRing("test")
    _write(ring, 64, [1, 1], world=2, epr=1)
    assert [(r["call"], r["layer"]) for r in ring.drain()] == [(64, (64 - 1) % 4)]
    # the next slot (call 128's) was lapped by call 128 + SLOTS * PERIOD before this drain
    lapped = 128 + E.SLOTS * E.PERIOD
    _write(ring, lapped, [2, 2], world=2, epr=1)
    out = ring.drain()
    assert [r["call"] for r in out] == [lapped]
    assert out[0]["lost"] == E.SLOTS  # calls 128 .. lapped - PERIOD were overwritten
    E.configure(0)


@cuda
def test_call_id_reset_resyncs_after_rescan():
    from vllm.models.kimi_k3.nvidia.megamoe_rubin import ep_balance as E

    ring = E.EpBalanceRing("test")
    _write(ring, 64 * 10, [1, 1], world=2, epr=1, t=100)
    ring._last = 64 * 10  # as if drained
    _write(ring, 64, [3, 1], world=2, epr=1, t=200)  # later: the session restarted its call ids
    for _ in range(E.RESCAN_EVERY - 1):
        assert ring.drain() == []
    out = ring.drain()  # the rescan resyncs the cursor; the restarted sequence drains from the next call
    assert [r["call"] for r in out] == [64] and out[0]["rank_loads"] == [3, 1]


def test_local_decode_records_merge_by_call():
    from vllm.models.kimi_k3.nvidia.megamoe_rubin import ep_balance as E

    def slot(call, sizes, rank, world=2):
        w = [0] * E.SLOT_WORDS
        w[0], w[3], w[4], w[5], w[6] = call, len(sizes), world, len(sizes), rank + 1
        w[E.HEADER_WORDS : E.HEADER_WORDS + len(sizes)] = sizes
        return w

    r0 = E.summarize(slot(64, [3, 1], rank=0))
    r1 = E.summarize(slot(64, [6, 2], rank=1))
    assert r0["local_rank"] == 0 and r0["rank_loads"] == [4] and r0["max_over_mean"] == -1.0
    for r in (r0, r1):
        r["session"] = "genphase-128"
    lone = E.summarize(slot(128, [1, 1], rank=0))
    lone["session"] = "genphase-128"
    merged = E.merge_local([[r0, lone], [r1]])
    assert [(m["call"], m["complete"]) for m in merged] == [(64, True), (128, False)]
    assert merged[0]["rank_loads"] == [4, 8] and merged[0]["routes"] == 12
    assert merged[0]["max_over_mean"] == pytest.approx(8 / 6)
