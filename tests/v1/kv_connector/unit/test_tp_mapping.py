# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for TP mapping and transfer plan utilities.

These tests verify that TP mapping produces correct outputs
(source ranks, split handles, desc IDs).
No GPU or NIXL required.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vllm.distributed.kv_transfer.kv_connector.dcp_layout import DcpLayout
from vllm.distributed.kv_transfer.kv_connector.utils import TransferTopology
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import (
    DcpTransferMode,
    TPMapping,
    compute_tp_mapping,
    select_dcp_transfer_mode,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils import (
    NixlPeerIncompatibleError,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import (
    NixlConnectorWorker,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

# ======================================================================
# Test fixtures / helpers
# ======================================================================


def _compute_mapping(
    tp_rank: int = 0,
    tp_size: int = 1,
    remote_tp_size: int = 1,
    is_mla: bool = False,
    num_kv_heads: int = 8,
    group_spec_types: tuple[type, ...] = (FullAttentionSpec,),
    dcp_size: int = 1,
    remote_dcp_size: int = 1,
    head_sharded_kv_heads: int | None = None,
) -> TPMapping:
    transfer_topology = object.__new__(TransferTopology)
    transfer_topology.tp_rank = tp_rank
    transfer_topology.tp_size = tp_size
    transfer_topology.is_mla = is_mla
    transfer_topology.total_num_kv_heads = num_kv_heads
    transfer_topology.dcp_size = dcp_size
    transfer_topology.is_mamba = any(t is MambaSpec for t in group_spec_types)
    return compute_tp_mapping(
        transfer_topology=transfer_topology,
        remote_tp_size=remote_tp_size,
        group_spec_types=group_spec_types,
        remote_dcp_size=remote_dcp_size,
        head_sharded_kv_heads=head_sharded_kv_heads,
    )


# ======================================================================
# TP mapping structure tests
# ======================================================================


class TestTPMappingStructure:
    def test_source_ranks_homogeneous(self):
        m = _compute_mapping(tp_size=2, tp_rank=1, remote_tp_size=2)
        assert m.all_source_ranks == (1,)

    def test_source_ranks_d_gt_p(self):
        m = _compute_mapping(tp_size=4, tp_rank=2, remote_tp_size=2)
        assert m.all_source_ranks == (1,)

    def test_source_ranks_p_gt_d(self):
        m = _compute_mapping(tp_size=1, tp_rank=0, remote_tp_size=2)
        assert m.all_source_ranks == (0, 1)


@pytest.mark.parametrize(
    "tp_rank,tp_size,remote_tp_size,draft_kv_heads,expected_offset",
    [
        (3, 8, 1, 8, 3),  # one draft head per decode rank
        (5, 8, 2, 8, 1),  # rank 5 reads P rank 1, second head slot
        (5, 16, 1, 8, 2),  # D_TP > heads: two ranks share each head
    ],
)
def test_head_sharded_draft_under_mla_target_gets_gqa_offset(
    tp_rank, tp_size, remote_tp_size, draft_kv_heads, expected_offset
):
    """An MLA target's mapping reads every head at offset 0, which is wrong
    for a GQA draft's head-sharded KV. Mapping the draft as GQA with its own
    head count recovers each rank's slice from the same source rank."""
    mla = _compute_mapping(
        tp_rank=tp_rank,
        tp_size=tp_size,
        remote_tp_size=remote_tp_size,
        is_mla=True,
        num_kv_heads=1,
    )
    draft = _compute_mapping(
        tp_rank=tp_rank,
        tp_size=tp_size,
        remote_tp_size=remote_tp_size,
        is_mla=True,
        num_kv_heads=1,
        head_sharded_kv_heads=draft_kv_heads,
    )
    assert mla.rank_offset_factor == 0
    assert draft.rank_offset_factor == expected_offset
    assert draft.all_source_ranks == mla.all_source_ranks


@pytest.mark.parametrize(
    "tp_rank,tp_size,remote_tp_size,dcp_size,remote_dcp_size,expected_ranks",
    [
        (0, 1, 8, 1, 8, tuple(range(8))),
        (0, 4, 4, 1, 4, (0, 1, 2, 3)),
        (2, 4, 4, 4, 4, (2,)),
        (0, 2, 4, 2, 4, (0, 2)),
        (3, 4, 2, 4, 2, (1,)),
    ],
)
def test_mla_dcp_source_ranks(
    tp_rank,
    tp_size,
    remote_tp_size,
    dcp_size,
    remote_dcp_size,
    expected_ranks,
):
    mapping = _compute_mapping(
        tp_rank=tp_rank,
        tp_size=tp_size,
        remote_tp_size=remote_tp_size,
        is_mla=True,
        num_kv_heads=1,
        dcp_size=dcp_size,
        remote_dcp_size=remote_dcp_size,
    )

    assert mapping.all_source_ranks == expected_ranks


@pytest.mark.parametrize(
    "tp_size,remote_tp_size,dcp_size,remote_dcp_size",
    [
        (4, 2, 4, 2),
        (4, 4, 4, 4),
        (4, 1, 4, 1),
        (4, 4, 1, 4),
        (2, 4, 2, 4),
    ],
)
def test_dcp_consumer_count_matches_readers(
    tp_size, remote_tp_size, dcp_size, remote_dcp_size
):
    """Each producer waits for exactly the local ranks that read from it."""
    mappings = [
        _compute_mapping(
            tp_rank=tp_rank,
            tp_size=tp_size,
            remote_tp_size=remote_tp_size,
            is_mla=True,
            num_kv_heads=1,
            dcp_size=dcp_size,
            remote_dcp_size=remote_dcp_size,
        )
        for tp_rank in range(tp_size)
    ]

    for remote_rank in range(remote_tp_size):
        readers = [m for m in mappings if remote_rank in m.all_source_ranks]
        for mapping in readers:
            assert mapping.consumers(remote_rank) == len(readers)


# ======================================================================
# Hybrid (attention + Mamba state) source ranks under DCP
# ======================================================================

HYBRID_GROUPS = (FullAttentionSpec, MambaSpec)


def _hybrid_mappings(tp_size: int, remote_tp_size: int) -> list[TPMapping]:
    """Mappings of every local rank of an MLA+Mamba model with TP = DCP on
    both sides."""
    return [
        _compute_mapping(
            tp_rank=tp_rank,
            tp_size=tp_size,
            remote_tp_size=remote_tp_size,
            is_mla=True,
            num_kv_heads=1,
            group_spec_types=HYBRID_GROUPS,
            dcp_size=tp_size,
            remote_dcp_size=remote_tp_size,
        )
        for tp_rank in range(tp_size)
    ]


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("tp_size", "remote_tp_size"), [(16, 8), (8, 16), (16, 4), (4, 16), (8, 8)]
)
def test_hybrid_dcp_attention_and_state_sources_split(tp_size, remote_tp_size):
    """Attention KV comes from the remote ranks holding this rank's DCP
    positions (rank mod DCP), Mamba state from those holding its TP shard of
    the heads (TP ratio); each producer counts every local rank reading any
    kind from it."""
    mappings = _hybrid_mappings(tp_size, remote_tp_size)
    for tp_rank, mapping in enumerate(mappings):
        attn, state = mapping.source_ranks_per_group
        if tp_size >= remote_tp_size:
            assert attn == (tp_rank % remote_tp_size,)
            assert state == (tp_rank * remote_tp_size // tp_size,)
        else:
            ratio = remote_tp_size // tp_size
            assert attn == tuple(range(tp_rank, remote_tp_size, tp_size))
            assert state == tuple(range(tp_rank * ratio, (tp_rank + 1) * ratio))
        assert mapping.all_source_ranks == tuple(sorted(set(attn) | set(state)))
    # Per producer: the count equals the brute-force reader set, and every
    # reader of a producer sends the same count.
    for remote_rank in range(remote_tp_size):
        readers = [m for m in mappings if remote_rank in m.all_source_ranks]
        assert readers
        assert {m.consumers(remote_rank) for m in readers} == {len(readers)}


@pytest.mark.cpu_test
def test_hybrid_dcp_p8_to_d16_consumers():
    """P TP8 -> D TP16: P rank 3 serves attention to D ranks {3, 11} and state
    to D ranks {6, 7}, so it waits for four notifications."""
    mappings = _hybrid_mappings(16, 8)
    attention_readers = {
        r for r, m in enumerate(mappings) if 3 in m.source_ranks_per_group[0]
    }
    state_readers = {
        r for r, m in enumerate(mappings) if 3 in m.source_ranks_per_group[1]
    }
    assert attention_readers == {3, 11}
    assert state_readers == {6, 7}
    assert mappings[3].consumers(3) == mappings[6].consumers(3) == 4


@pytest.mark.cpu_test
def test_handshake_targets_cover_attention_and_state_sources():
    topology = object.__new__(TransferTopology)
    topology.tp_rank, topology.tp_size, topology.dcp_size = 6, 16, 16
    topology.is_mamba = True
    assert topology.handshake_target_ranks(8, 8) == [3, 6]
    topology.is_mamba = False
    assert topology.handshake_target_ranks(8, 8) == [6]


# ======================================================================
# Split handle tests
# ======================================================================


def _make_mock_worker_for_splits(group_spec_types):
    """Build a mock NixlConnectorWorker with _group_spec_types for split tests.

    No per-region replicate flags are configured (`block_len_per_layer` empty
    and `num_regions == 0`), so `_build_local_splits_from_plan` treats every
    FA descriptor as SPLIT, matching the legacy behavior these tests assert.
    """
    worker = object.__new__(NixlConnectorWorker)
    worker._group_spec_types = group_spec_types
    worker.transfer_topo = SimpleNamespace(virtually_split_kv_in_blocks=False)
    worker.block_len_per_layer = []
    worker.num_regions = 0
    worker._region_is_mla = []
    worker._conv_decomp = SimpleNamespace(local_conv_offsets=())
    worker._ssm_region_indices = [0] if MambaSpec in group_spec_types else []
    worker._ple_region_index = None
    return worker


class TestBuildSrcSplitHandles:
    @pytest.mark.parametrize("remote_tp_size", [2, 4])
    def test_build_src_split_handles(self, remote_tp_size):
        tp_rank = 0
        tp_size = 1

        plan = _compute_mapping(
            tp_rank=tp_rank,
            tp_size=tp_size,
            remote_tp_size=remote_tp_size,
        )

        worker = _make_mock_worker_for_splits((FullAttentionSpec,))
        # One region of 8 dense 1024-byte blocks as a single strided run.
        src_blocks_data = np.array([(0x2000, 1024, 0, 1024, 8)], dtype=np.uint64)
        num_descs = 8
        splits = list(
            worker._build_local_splits_from_plan(
                plan,
                src_blocks_data,
                num_descs,
            )
        )

        assert len(splits) == remote_tp_size
        for handle in splits:
            assert handle.shape == src_blocks_data.shape
            # The head slice shrinks the length; block stride and count are kept.
            assert handle[:, 1].tolist() == [1024 // remote_tp_size]
            assert handle[:, 3:].tolist() == [[1024, 8]]


class TestMambaPlanSplitHandles:
    """Verify split handles for Mamba with FA/SSM distinction."""

    def test_ssm_slice_follows_state_source_index(self):
        """Under DCP the attention sources are not the state sources (D TP8
        <- P TP16, D rank 1: attention {1, 9}, state {2, 3}): a state source
        takes its slice by its index among the state sources, not among all
        sources."""
        plan = _compute_mapping(
            tp_rank=1,
            tp_size=8,
            remote_tp_size=16,
            is_mla=True,
            num_kv_heads=1,
            group_spec_types=(FullAttentionSpec, MambaSpec),
            dcp_size=8,
            remote_dcp_size=16,
        )
        assert plan.all_source_ranks == (1, 2, 3, 9)
        worker = _make_mock_worker_for_splits((FullAttentionSpec, MambaSpec))
        worker._region_is_mla = [True]
        worker.block_len_per_layer = [200]
        worker.num_regions = 1
        worker.region_num_blocks = [1]
        src_blocks_data = np.array(
            [(1000, 200, 0, 200, 1), (3000, 400, 0, 400, 1)], dtype=np.uint64
        )
        splits = dict(
            zip(
                plan.all_source_ranks,
                worker._build_local_splits_from_plan(plan, src_blocks_data, 1),
            )
        )
        # MLA attention is read whole from every attention source.
        assert all(h[0].tolist() == [1000, 200, 0, 200, 1] for h in splits.values())
        assert splits[2][1].tolist() == [3000, 200, 0, 400, 1]
        assert splits[3][1].tolist() == [3200, 200, 0, 400, 1]

    def test_fa_and_ssm_different_split_factors(self):
        """Section 0 split by num_attn_reads, section 1 by abs_tp."""
        fa_readers = (0,)
        ssm_readers = (0, 1)
        plan = TPMapping(
            source_ranks_per_group=(fa_readers, ssm_readers),
            all_source_ranks=(0, 1),
            rank_to_attention_slot={0: 0, 1: 0},
            rank_offset_factor=0,
        )

        worker = _make_mock_worker_for_splits((FullAttentionSpec, MambaSpec))
        # 2 FA runs + 1 SSM run of one block each: (addr, len, dev, stride, count)
        src_blocks_data = np.array(
            [
                (1000, 200, 0, 200, 1),  # FA desc 0
                (2000, 200, 0, 200, 1),  # FA desc 1
                (3000, 400, 0, 400, 1),  # SSM desc 0
            ],
            dtype=np.uint64,
        )

        splits = list(worker._build_local_splits_from_plan(plan, src_blocks_data, 2))

        assert len(splits) == 2  # 2 source ranks

        # Rank 0 (FA source, p_idx=0):
        # FA: chunk=200//1=200, slot=0 → (1000, 200), (2000, 200)
        # SSM: chunk=400//2=200, idx=0 → (3000, 200); block stride stays 400
        assert splits[0].tolist() == [
            [1000, 200, 0, 200, 1],
            [2000, 200, 0, 200, 1],
            [3000, 200, 0, 400, 1],
        ]

        # Rank 1 (not FA source, p_idx=1):
        # FA: chunk=200//1=200, slot=0 (skip_fa) → (1000, 200), (2000, 200)
        # SSM: chunk=400//2=200, idx=1 → (3200, 200); block stride stays 400
        assert splits[1].tolist() == [
            [1000, 200, 0, 200, 1],
            [2000, 200, 0, 200, 1],
            [3200, 200, 0, 400, 1],
        ]

    def test_hetero_block_size_splits(self):
        """With a block-size ratio, single-source FA sub-block descs pass
        through whole; SSM descs are unexpanded and split per source."""
        plan = TPMapping(
            source_ranks_per_group=((0,), (0, 1)),
            all_source_ranks=(0, 1),
            rank_to_attention_slot={0: 0, 1: 0},
            rank_offset_factor=0,
        )

        worker = _make_mock_worker_for_splits((FullAttentionSpec, MambaSpec))
        # 2 FA blocks x ratio 2 sub-blocks (one run of 2 per block) + 1 SSM desc
        # (never expanded).
        src_blocks_data = np.array(
            [
                (1000, 100, 0, 100, 2),
                (2000, 100, 0, 100, 2),
                (3000, 400, 0, 400, 1),
            ],
            dtype=np.uint64,
        )

        splits = list(worker._build_local_splits_from_plan(plan, src_blocks_data, 4, 2))

        assert len(splits) == 2
        fa_passthrough = [[1000, 100, 0, 100, 2], [2000, 100, 0, 100, 2]]
        assert splits[0].tolist() == fa_passthrough + [[3000, 200, 0, 400, 1]]
        assert splits[1].tolist() == fa_passthrough + [[3200, 200, 0, 400, 1]]

    def test_hetero_block_size_head_sharded_asserts(self):
        """Head-sharded FA reads (multiple FA sources) are incompatible with
        a block-size mismatch and must fail loudly."""
        plan = TPMapping(
            source_ranks_per_group=((0, 1), (0, 1)),
            all_source_ranks=(0, 1),
            rank_to_attention_slot={0: 0, 1: 1},
            rank_offset_factor=0,
        )

        worker = _make_mock_worker_for_splits((FullAttentionSpec, MambaSpec))
        src_blocks_data = np.array(
            [(1000, 100, 0, 100, 2), (3000, 400, 0, 400, 1)],
            dtype=np.uint64,
        )

        with pytest.raises(AssertionError, match="Head-sharded"):
            list(worker._build_local_splits_from_plan(plan, src_blocks_data, 2, 2))


@pytest.mark.parametrize(
    ("total_kv_heads", "local_tp", "remote_tp", "compatible"),
    [
        # 2 heads: TP1 packs both heads into a page, TP >= 2 packs one.
        (2, 1, 2, False),
        (2, 2, 1, False),
        (2, 1, 1, True),
        (2, 2, 4, True),
        (2, 4, 2, True),
        # 4 heads: the boundary moves with the head count.
        (4, 2, 4, False),
        (4, 2, 2, True),
        (4, 4, 8, True),
    ],
)
def test_csa_linear_tp_layout_boundary(total_kv_heads, local_tp, remote_tp, compatible):
    worker = object.__new__(NixlConnectorWorker)
    worker.world_size = local_tp
    worker._is_csa_linear = True
    worker.transfer_topo = SimpleNamespace(total_num_kv_heads=total_kv_heads)

    if compatible:
        worker._validate_csa_linear_tp_layout(remote_tp)
    else:
        with pytest.raises(ValueError, match="KV-head sharding boundary"):
            worker._validate_csa_linear_tp_layout(remote_tp)


# ======================================================================
# DCP transfer mode selection
# ======================================================================


@pytest.mark.parametrize(
    ("local", "remote"),
    [
        # No DCP on either side, whatever the blocks.
        ((1, 1, 16), (1, 1, 64)),
        # Same DCP layout (homogeneous DCP, e.g. Kimi K3.1 DCP8 <-> DCP8).
        ((8, 1, 1280), (8, 1, 1280)),
        ((4, 16, 64), (4, 16, 64)),
        # Same DCP size and interleave, other per-rank blocks: rows pair.
        ((8, 1, 640), (8, 1, 1280)),
        ((4, 1, 2560), (4, 1, 640)),
        ((4, 16, 64), (4, 16, 128)),
        # Block-interleaved pages of equal size over different DCP sizes.
        ((1, 1, 16), (2, 16, 16)),
        ((4, 16, 16), (2, 16, 16)),
        ((2, 64, 64), (8, 64, 64)),
    ],
)
def test_dcp_transfer_mode_direct(local, remote):
    mode = select_dcp_transfer_mode(DcpLayout(*local), DcpLayout(*remote))
    assert mode is DcpTransferMode.DIRECT


@pytest.mark.parametrize(
    ("local", "remote", "match"),
    [
        # Both shard, but rank turns differ: positions land on other ranks.
        ((4, 1, 64), (4, 64, 64), "interleave"),
        ((4, 1, 64), (2, 64, 64), "interleave"),
        # Different DCP sizes move rows between pages: token-level interleave
        # (Kimi K3.1 P8 -> D16), or block-level pages of different sizes.
        # Without TP sizes proving TP = DCP no relayout is possible either.
        ((16, 1, 640), (8, 1, 1280), "TP = DCP on both sides"),
        ((16, 1, 640), (8, 1, 640), "TP = DCP on both sides"),
        ((1, 1, 64), (2, 1, 64), "TP = DCP on both sides"),
        ((1, 1, 32), (4, 64, 64), "TP = DCP on both sides"),
        ((4, 64, 64), (2, 128, 128), "interleave"),
    ],
)
def test_dcp_transfer_mode_rejects(local, remote, match):
    with pytest.raises(NixlPeerIncompatibleError, match=match):
        select_dcp_transfer_mode(DcpLayout(*local), DcpLayout(*remote))


@pytest.mark.parametrize(
    ("local", "remote"),
    [
        # Kimi K3.1 P TP8=DCP8 -> D TP16=DCP16 with TP-scaled blocks, and back.
        ((16, 1, 640), (8, 1, 1280)),
        ((8, 1, 1280), (16, 1, 640)),
        # Token-level interleave over different DCP sizes, equal blocks.
        ((16, 1, 640), (8, 1, 640)),
        ((1, 1, 64), (2, 1, 64)),
        # Block-interleaved pages of different sizes.
        ((4, 64, 64), (2, 64, 128)),
    ],
)
def test_dcp_transfer_mode_relayout(local, remote):
    mode = select_dcp_transfer_mode(
        DcpLayout(*local), DcpLayout(*remote), tp_sizes=(local[0], remote[0])
    )
    assert mode is DcpTransferMode.RELAYOUT


def test_dcp_transfer_mode_prefers_direct():
    """Layouts whose pages pair whole never pay for a relayout."""
    for local, remote in [
        ((8, 1, 1280), (8, 1, 1280)),
        ((8, 1, 640), (8, 1, 1280)),
        ((2, 64, 64), (8, 64, 64)),
        ((4, 64, 64), (2, 64, 64)),
    ]:
        mode = select_dcp_transfer_mode(
            DcpLayout(*local), DcpLayout(*remote), tp_sizes=(local[0], remote[0])
        )
        assert mode is DcpTransferMode.DIRECT


@pytest.mark.parametrize(
    ("local", "remote", "tp_sizes", "match"),
    [
        # TP != DCP on one side.
        ((16, 1, 640), (8, 1, 1280), (16, 16), "TP = DCP on both sides"),
        ((4, 1, 640), (8, 1, 1280), (8, 8), "TP = DCP on both sides"),
        # DCP sizes that do not divide one another.
        ((6, 1, 64), (4, 1, 64), (6, 4), "divide one another"),
        # The interleave of both sides must agree whatever the mode.
        ((16, 1, 640), (8, 2, 1280), (16, 8), "interleave"),
    ],
)
def test_dcp_transfer_mode_relayout_rejects(local, remote, tp_sizes, match):
    with pytest.raises(NixlPeerIncompatibleError, match=match):
        select_dcp_transfer_mode(
            DcpLayout(*local), DcpLayout(*remote), tp_sizes=tp_sizes
        )
