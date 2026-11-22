# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Served EP-balance ring (megamoe_rubin/ep_balance.py): slot layout, drain order, per-rank loads."""

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="pinned host memory needs CUDA")


def _d(ring):
    """One host drain cycle: drain() takes the copy issued by the previous drain and issues the next one, so a
    drain, a device sync and a second drain see everything written before the first."""
    out = ring.drain()
    torch.cuda.synchronize()
    return out + ring.drain()


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
    buf[base + 7] = E.epbal_checksum(call, sizes)
    buf[base] = call


@cuda
def test_drain_returns_completed_calls_in_order_with_rank_loads():
    from vllm.models.kimi_k3.nvidia.megamoe_rubin import ep_balance as E

    ring = E.EpBalanceRing("test")
    assert _d(ring) == []
    # 4 ranks x 2 experts: rank loads [3, 7, 0, 6] -> mean 4, max/mean 1.75
    _write(ring, 64, [1, 2, 3, 4, 0, 0, 5, 1], world=4, epr=2)
    _write(ring, 128, [2, 2, 2, 2, 2, 2, 2, 2], world=4, epr=2)
    out = _d(ring)
    assert [r["call"] for r in out] == [64, 128]
    assert out[0]["rank_loads"] == [3, 7, 0, 6] and out[0]["routes"] == 16
    assert out[0]["max_over_mean"] == pytest.approx(1.75) and out[0]["expert_max"] == 5
    assert out[1]["max_over_mean"] == pytest.approx(1.0)
    assert _d(ring) == []  # nothing new


@cuda
def test_slot_being_rewritten_waits_for_the_next_drain():
    from vllm.models.kimi_k3.nvidia.megamoe_rubin import ep_balance as E

    ring = E.EpBalanceRing("test")
    _write(ring, 64, [1, 1], world=2, epr=1)
    assert [r["call"] for r in _d(ring)] == [64]
    # call 128's slot: payload written, call word not yet (the kernel writes it last)
    i = (128 >> E.SHIFT) % E.SLOTS
    ring.buffer[i * E.SLOT_WORDS + E.HEADER_WORDS] = 9
    assert _d(ring) == []
    _write(ring, 128, [9, 1], world=2, epr=1)
    out = _d(ring)
    assert [r["call"] for r in out] == [128] and out[0]["rank_loads"] == [9, 1]


@cuda
def test_global_record_without_matching_checksum_waits():
    """Records are written without a fence and copied out while kernels run: a slot whose call id is visible but
    whose counts (or checksum) are still landing fails the checksum and is taken on a later drain."""
    from vllm.models.kimi_k3.nvidia.megamoe_rubin import ep_balance as E

    ring = E.EpBalanceRing("test")
    _write(ring, 64, [4, 4], world=2, epr=1)
    i = (64 >> E.SHIFT) % E.SLOTS
    ring.buffer[i * E.SLOT_WORDS + E.HEADER_WORDS + 1] = 3  # one count not landed yet
    assert _d(ring) == []
    ring.buffer[i * E.SLOT_WORDS + E.HEADER_WORDS + 1] = 4
    out = _d(ring)
    assert [r["call"] for r in out] == [64] and out[0]["routes"] == 8


def test_checksum_matches_device_int32_arithmetic():
    """Reference: the kernel's wrapping int32 ops (weights (i * POS_MUL + POS_ADD) | 1 as signed int32)."""
    import numpy as np
    from vllm.models.kimi_k3.nvidia.megamoe_rubin import ep_balance as E
    from vllm.models.kimi_k3.nvidia.megamoe_rubin.src_gp.common import ep_balance_layout as L

    rng = np.random.default_rng(0)
    for call, n in ((64, 8), (1 << 30, 224), (64 * 999_999, 512)):
        counts = rng.integers(0, 50_000, n).astype(np.int32)
        with np.errstate(over="ignore"):
            i = np.arange(n, dtype=np.int32)
            w = (i * np.int32(L.EPBAL_POS_MUL_S32) + np.int32(L.EPBAL_POS_ADD)) | np.int32(1)
            acc = np.sum(counts * w, dtype=np.int32)
            dev = (acc ^ np.int32(L.EPBAL_CHECKSUM_SALT)) + np.int32(call) * np.int32(L.EPBAL_CHECKSUM_MUL)
        assert E.epbal_checksum(call, counts.tolist()) == int(dev)


def test_checksum_separates_equal_sum_records():
    """tokens x top-k is the same for every call of one size: records with equal sums but different routing, or
    with the same counts at swapped positions, must get different checksums."""
    from vllm.models.kimi_k3.nvidia.megamoe_rubin import ep_balance as E

    a = [5, 3, 0, 8, 2, 6, 1, 7]
    b = [3, 5, 0, 8, 2, 6, 1, 7]  # two experts swapped
    c = [4, 4, 1, 7, 3, 5, 2, 6]  # same sum, different distribution
    assert sum(a) == sum(b) == sum(c)
    assert len({E.epbal_checksum(64, a), E.epbal_checksum(64, b), E.epbal_checksum(64, c)}) == 3
    assert E.epbal_checksum(64, a) != E.epbal_checksum(128, a)  # bound to the call id


@cuda
def test_torn_record_mixing_two_same_size_calls_is_rejected():
    """Slot reuse without a fence: the slot holds call 64's record when call 64 + 64 * SLOTS (same size, same
    sum) lands; the host reads the new call id + checksum with half of the old counts still visible."""
    from vllm.models.kimi_k3.nvidia.megamoe_rubin import ep_balance as E

    ring = E.EpBalanceRing("test")
    old = [6, 2, 4, 4, 1, 7, 3, 5]
    new = [2, 6, 4, 4, 7, 1, 5, 3]
    assert sum(old) == sum(new)
    _write(ring, 64, old, world=4, epr=2)
    assert [r["call"] for r in _d(ring)] == [64]
    nxt = 64 + 64 * E.SLOTS
    _write(ring, nxt, new, world=4, epr=2)
    i = (nxt >> E.SHIFT) % E.SLOTS
    base = i * E.SLOT_WORDS + E.HEADER_WORDS
    ring.buffer[base : base + 4] = torch.tensor(old[:4], dtype=torch.int32)  # first half still the old call's
    ring._last = nxt - 64  # cursor just before the new call
    assert _d(ring) == []  # torn: rejected, retried later
    ring.buffer[base : base + 4] = torch.tensor(new[:4], dtype=torch.int32)
    out = _d(ring)
    assert [r["call"] for r in out] == [nxt] and out[0]["rank_loads"] == [8, 8, 8, 8]



@cuda
def test_lapped_slot_counts_lost_calls_and_layer_index():
    from vllm.models.kimi_k3.nvidia.megamoe_rubin import ep_balance as E

    E.configure(4)
    ring = E.EpBalanceRing("test")
    _write(ring, 64, [1, 1], world=2, epr=1)
    assert [(r["call"], r["layer"]) for r in _d(ring)] == [(64, (64 - 1) % 4)]
    # the next slot (call 128's) was lapped by call 128 + SLOTS * PERIOD before this drain
    lapped = 128 + E.SLOTS * E.PERIOD
    _write(ring, lapped, [2, 2], world=2, epr=1)
    out = _d(ring)
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
    cycles, out = 0, []
    while not out and cycles < 3 * E.RESCAN_EVERY:
        out = _d(ring)  # the rescan resyncs the cursor; the restarted sequence drains from the next call
        cycles += 1
    assert cycles >= E.RESCAN_EVERY // 2  # only the rescan finds it
    assert [r["call"] for r in out] == [64] and out[0]["rank_loads"] == [3, 1]


@cuda
def test_local_record_needs_its_checksum_too():
    """Genphase local records are fence-free as well: a local slot is taken only with a matching checksum."""
    from vllm.models.kimi_k3.nvidia.megamoe_rubin import ep_balance as E

    ring = E.EpBalanceRing("test")
    _write(ring, 64, [5, 2], world=8, epr=2)
    i = (64 >> E.SHIFT) % E.SLOTS
    ring.buffer[i * E.SLOT_WORDS + 6] = 3  # local record of rank 2
    ring.buffer[i * E.SLOT_WORDS + 7] = 0  # checksum not landed
    assert _d(ring) == []
    ring.buffer[i * E.SLOT_WORDS + 7] = E.epbal_checksum(64, [5, 2])
    out = _d(ring)
    assert [(r["call"], r["local_rank"], r["rank_loads"]) for r in out] == [(64, 2, [7])]


@cuda
def test_ring_lives_in_device_memory_and_drain_never_blocks():
    from vllm.models.kimi_k3.nvidia.megamoe_rubin import ep_balance as E

    ring = E.EpBalanceRing("test")
    assert ring.buffer.is_cuda and ring.address == ring.buffer.data_ptr()
    # a long kernel on the compute stream: the drain issues its copy on its own stream and returns at once
    x = torch.randn(4096, 4096, device="cuda")
    for _ in range(20):
        x = x @ x.T / 4096
    assert ring.drain() == [] and ring._pending is not None


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
