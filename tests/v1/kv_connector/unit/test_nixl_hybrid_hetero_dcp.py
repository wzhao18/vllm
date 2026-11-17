# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hybrid MLA + Mamba (GDN/KDA) NIXL pulls between peers with different DCP sizes,
e.g. a TP8/DCP8 prefiller feeding a TP16/DCP16 decoder.

MLA KV is DCP-sharded in interleave-sized chunks and matched by global position;
the Mamba state is TP-sharded by heads and never DCP-sharded, so a decoder rank can
read its MLA blocks and its Mamba state from different prefiller ranks. The
handshake accepts block-sized interleaves, or interleave 1 with the decoder's DCP
pair exchange; any other interleave stops the decoder.
No GPU or NIXL required.
"""

from __future__ import annotations

import contextlib
import operator
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import msgspec
import numpy as np
import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.utils import TransferTopology
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import base_worker
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
    NixlPeerConfigMismatch,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlAgentMetadata,
    NixlHandshakePayload,
    RemoteMeta,
    ReqMeta,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import (
    compute_tp_mapping,
    head_sharded_source_ranks,
)
from vllm.v1.kv_cache_interface import (
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
)

# Kimi K3's hybrid layout: one MLA group, then KDA (Mamba) groups.
HYBRID_GROUPS = (MLAAttentionSpec, MambaSpec, MambaSpec, MambaSpec)
# With the release DSpark draft: its sliding-window GQA group (16 KV heads).
DRAFT_GROUPS = (*HYBRID_GROUPS, SlidingWindowSpec)
DRAFT_GROUP, DRAFT_KV_HEADS = len(HYBRID_GROUPS), 16


def _topology(
    tp_rank: int, tp_size: int, dcp_size: int, is_mamba: bool = True
) -> TransferTopology:
    topo = object.__new__(TransferTopology)
    topo.tp_rank = tp_rank
    topo.tp_size = tp_size
    topo.is_mla = True
    topo.is_mamba = is_mamba
    topo.total_num_kv_heads = 1
    topo.dcp_size = dcp_size
    return topo


def _mapping(tp_rank, tp_size, dcp_size, remote_tp_size, remote_dcp_size):
    return compute_tp_mapping(
        transfer_topology=_topology(tp_rank, tp_size, dcp_size),
        remote_tp_size=remote_tp_size,
        group_spec_types=HYBRID_GROUPS,
        remote_dcp_size=remote_dcp_size,
    )


@pytest.mark.parametrize("tp_rank", range(16))
def test_hetero_dcp_attention_and_ssm_sources(tp_rank):
    """D TP16/DCP16 <- P TP8/DCP8: MLA from the rank owning the same DCP chunks
    (r % 8), Mamba state from the rank owning the same heads (r // 2)."""
    m = _mapping(tp_rank, 16, 16, 8, 8)
    attn, ssm = (tp_rank % 8,), (tp_rank // 2,)
    assert m.source_ranks_per_group == (attn, ssm, ssm, ssm)
    assert m.all_source_ranks == tuple(sorted(set(attn) | set(ssm)))
    topo = _topology(tp_rank, 16, 16)
    assert topo.handshake_target_ranks(8, 8) == list(m.all_source_ranks)


def test_hetero_dcp_consumer_counts():
    """Each prefiller rank waits for exactly the decoder ranks reading from it
    (attention or Mamba): ranks 0 and 7 have 3 readers, the others 4."""
    mappings = [_mapping(r, 16, 16, 8, 8) for r in range(16)]
    for remote_rank in range(8):
        readers = [m for m in mappings if remote_rank in m.all_source_ranks]
        assert len(readers) == (3 if remote_rank in (0, 7) else 4)
        for m in readers:
            assert m.consumers_for(remote_rank) == len(readers)


@pytest.mark.parametrize("tp_size,dcp_size", [(8, 8), (16, 16)])
def test_homogeneous_hybrid_dcp_unchanged(tp_size, dcp_size):
    """Equal DCP sizes: Mamba and MLA sources coincide, a single consumer count."""
    for r in range(tp_size):
        m = _mapping(r, tp_size, dcp_size, tp_size, dcp_size)
        assert m.source_ranks_per_group == ((r,),) * 4
        assert m.consumers_by_rank is None
        assert m.consumers_for(r) == 1


def test_hetero_tp_hybrid_without_dcp_unchanged():
    """No DCP (D TP4 <- P TP2): Mamba follows the head owner, as before."""
    for r in range(4):
        m = _mapping(r, 4, 1, 2, 1)
        assert m.source_ranks_per_group == ((r // 2,),) * 4
        assert m.consumers_by_rank is None


def _draft_mappings(
    tp_rank, tp_size, remote_tp_size, groups=DRAFT_GROUPS, is_mamba=True
):
    """A decoder rank's target mapping (counting the draft's readers) and its
    draft mapping, as add_remote_agent builds them; DCP = TP on both sides."""
    topo = _topology(tp_rank, tp_size, tp_size, is_mamba=is_mamba)
    target = compute_tp_mapping(
        topo, remote_tp_size, groups, remote_tp_size, draft_kv_heads=DRAFT_KV_HEADS
    )
    draft = compute_tp_mapping(
        topo,
        remote_tp_size,
        groups,
        remote_tp_size,
        head_sharded_kv_heads=DRAFT_KV_HEADS,
    )
    return topo, target, draft


@pytest.mark.parametrize("tp_rank", range(16))
def test_release_draft_reads_its_heads_owner_tp8_to_tp16(tp_rank):
    """The release draft at DCP 1 on D TP16/DCP16 <- P TP8/DCP8 (C1 ... C160):
    D rank r holds draft head r, which P rank r // 2 holds as its local head
    r % 2, whatever the DCP sizes; the target keeps its MLA (r % 8) and KDA
    (r // 2) sources, which the handshake already covers."""
    topo, target, draft = _draft_mappings(tp_rank, 16, 8)
    assert draft.source_ranks_per_group[DRAFT_GROUP] == (tp_rank // 2,)
    assert draft.rank_offset_factor == tp_rank % 2
    assert head_sharded_source_ranks(topo, 8, DRAFT_KV_HEADS) == [tp_rank // 2]
    attn, ssm = (tp_rank % 8,), (tp_rank // 2,)
    assert target.source_ranks_per_group[: len(HYBRID_GROUPS)] == (attn, *[ssm] * 3)
    assert set(topo.handshake_target_ranks(8, 8)) == {tp_rank % 8, tp_rank // 2}


def test_release_draft_consumer_counts_tp8_to_tp16():
    """Counting the draft's readers changes nothing for the hybrid target (the
    KDA state comes from the same rank): P ranks 0 and 7 wait for 3 decoder
    ranks, the others for 4."""
    plans = [_draft_mappings(r, 16, 8)[1] for r in range(16)]
    for p in range(8):
        readers = [r for r in range(16) if p in {r % 8, r // 2}]
        assert len(readers) == (3 if p in (0, 7) else 4)
        assert {plans[r].consumers_for(p) for r in readers} == {len(readers)}


def test_draft_readers_counted_without_kda_state():
    """A pure MLA target pulls the draft from other ranks than its MLA KV: P
    rank p is read by decoder ranks p and p + 8 (MLA) and 2p, 2p + 1 (draft),
    so it must wait for all of them, not for the DCP ratio of 2."""
    groups = (MLAAttentionSpec, SlidingWindowSpec)
    plans = [_draft_mappings(r, 16, 8, groups, is_mamba=False)[1] for r in range(16)]
    for p in range(8):
        readers = {p, p + 8, 2 * p, 2 * p + 1}
        assert {plans[r].consumers_for(p) for r in readers} == {len(readers)}
    topo = _topology(0, 16, 16, is_mamba=False)
    assert compute_tp_mapping(topo, 8, groups, 8).consumers_for(0) == 2


@pytest.mark.parametrize("tp_rank", range(4))
def test_release_draft_tp4dcp4_reads_same_rank(tp_rank):
    """C512's DP ranks run TP4/DCP4 on both sides: each rank reads its 4 draft
    heads whole from the same rank, which waits for that one reader."""
    _, target, draft = _draft_mappings(tp_rank, 4, 4)
    assert draft.source_ranks_per_group[DRAFT_GROUP] == (tp_rank,)
    assert draft.rank_offset_factor == 0
    assert target.all_source_ranks == (tp_rank,)
    assert target.consumers_for(tp_rank) == 1


def _owner_chunks(rank: int, dcp_size: int, num_chunks: int) -> list[int]:
    """Global chunk index held by each per-rank block (interleave == block)."""
    return list(range(rank, num_chunks, dcp_size))


@pytest.mark.parametrize("num_chunks", [1, 7, 8, 9, 16, 17, 37, 64])
@pytest.mark.parametrize("local_num_computed_blocks", [0, 1, 2])
def test_hetero_dcp_blocks_match_global_chunks(num_chunks, local_num_computed_blocks):
    """With interleave == per-rank block on both sides, every matched
    (decoder block, prefiller block) pair holds the same global token chunk."""
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import (
        NixlConnectorWorker,
    )

    worker = object.__new__(NixlConnectorWorker)
    for d_rank in range(16):
        d_chunks = _owner_chunks(d_rank, 16, num_chunks)
        if len(d_chunks) <= local_num_computed_blocks:
            continue
        p_rank = _mapping(d_rank, 16, 16, 8, 8).source_ranks_per_group[0][0]
        p_chunks = _owner_chunks(p_rank, 8, num_chunks)
        local_ids = [1000 + j for j in range(len(d_chunks))]
        remote_ids = [2000 + i for i in range(len(p_chunks))]
        local, remote = worker._apply_dcp_prefix_caching(
            local_ids[local_num_computed_blocks:],
            remote_ids,
            remote_rank=p_rank,
            local_dcp_size=16,
            local_dcp_rank=d_rank,
            remote_dcp_size=8,
            local_num_computed_blocks=local_num_computed_blocks,
        )
        assert len(local) == len(d_chunks) - local_num_computed_blocks
        for lid, rid in zip(local, remote, strict=True):
            assert d_chunks[lid - 1000] == p_chunks[rid - 2000]


@pytest.fixture
def ds_conv_layout(monkeypatch):
    from vllm.model_executor.layers.mamba import mamba_utils

    monkeypatch.setenv("VLLM_SSM_CONV_STATE_LAYOUT", "DS")
    mamba_utils.get_conv_state_layout.cache_clear()
    yield
    mamba_utils.get_conv_state_layout.cache_clear()


def _kda_spec(tp: int, heads: int = 96, head_dim: int = 128, num_spec: int = 7):
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

    conv_rows = 3 + num_spec
    return MambaSpec(
        block_size=896,
        shapes=(
            (3 * heads * head_dim // tp, conv_rows),
            (heads // tp, head_dim, head_dim),
        ),
        dtypes=(torch.bfloat16, torch.bfloat16),
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
    )


@pytest.mark.parametrize("d_rank", range(16))
def test_kda_state_slices_tp8_to_tp16(ds_conv_layout, d_rank):
    """The decoder rank's conv sub-projections ([Q|K|V], DS layout) and recurrent
    heads, read as byte ranges from the prefiller rank owning its heads, equal
    the state the decoder rank itself would hold (Kimi K3: 96 heads x 128, conv
    width 4, 7 speculative columns)."""
    from vllm.distributed.kv_transfer.kv_connector.v1.ssm_conv_transfer_utils import (
        derive_mamba_conv_split,
    )

    heads, hd, rows = 96, 128, 10
    p_rank, j = d_rank // 2, d_rank % 2

    def conv_ids(tp, rank):  # (proj, global channel, column) ids, DS layout
        n = heads * hd // tp
        ch = np.arange(rank * n, (rank + 1) * n)
        return np.concatenate(
            [
                ((t * heads * hd + ch)[:, None] * rows + np.arange(rows))
                for t in range(3)
            ]
        ).reshape(-1)

    def ssm_ids(tp, rank):
        n = heads // tp
        return np.arange(rank * n * hd * hd, (rank + 1) * n * hd * hd)

    d_split = derive_mamba_conv_split(_kda_spec(16), local_tp=16)
    p_split = derive_mamba_conv_split(_kda_spec(8), local_tp=8)
    assert d_split.local_proj_dims == (768, 768, 768)
    assert p_split.local_proj_dims == (1536, 1536, 1536)

    # One P page: conv state, then the recurrent state (element = 2 bytes).
    p_page = np.concatenate([conv_ids(8, p_rank), ssm_ids(8, p_rank)])
    got = [
        p_page[off // 2 : (off + size) // 2]
        for off, size in d_split.remote_conv_offsets(j, tp_ratio=2)
    ]
    conv_size_remote, ssm_read = p_split.ssm_sizes[0], d_split.ssm_sizes[1]
    start = (conv_size_remote + j * ssm_read) // 2
    got.append(p_page[start : start + ssm_read // 2])
    assert np.array_equal(np.concatenate(got[:-1]), conv_ids(16, d_rank))
    assert np.array_equal(got[-1], ssm_ids(16, d_rank))


def _worker_stub(
    block=1024, phys=1, interleave=1024, tp=16, dcp=16, pair=False, mode="pull"
):
    """Decoder worker (TP16/DCP16 by default) for the handshake validators."""
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import (
        NixlConnectorWorker,
    )

    w = object.__new__(NixlConnectorWorker)
    w.block_size = block // phys
    w._physical_blocks_per_logical_kv_block = phys
    w.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=interleave)
    )
    w.use_mla = True
    w.world_size, w.dcp_size, w.pcp_size = tp, dcp, 1
    w._dcp_pair_exchange = pair
    w._TRANSFER_MODE = mode
    return w


def _meta(block=1024, phys=1, interleave=1024, dcp=8, pcp=1):
    """Prefiller handshake metadata (DCP8 by default); TP equals DCP."""
    return SimpleNamespace(
        engine_id="P",
        block_size=block // phys,
        physical_blocks_per_logical_kv_block=phys,
        cp_kv_cache_interleave_size=interleave,
        dcp_size=dcp,
        pcp_size=pcp,
    )


def _validate_hybrid(worker_kwargs, meta_kwargs):
    meta = _meta(**meta_kwargs)
    _worker_stub(**worker_kwargs)._validate_hybrid_hetero_dcp(
        meta, meta.dcp_size, meta.dcp_size
    )


# Decoder at interleave 1 with the DCP pair exchange on.
PAIR = {"interleave": 1, "pair": True}


@pytest.mark.parametrize(
    "worker_kwargs,meta_kwargs",
    [({}, {}), (PAIR, {"interleave": 1})],
    ids=["block-mode", "pair-mode"],
)
def test_validate_hybrid_hetero_dcp_accepts(worker_kwargs, meta_kwargs):
    _validate_hybrid(worker_kwargs, meta_kwargs)


@pytest.mark.parametrize(
    "worker_kwargs,meta_kwargs",
    [
        ({"interleave": 1}, {"interleave": 1}),
        (PAIR, {"interleave": 1024}),
        ({"pair": True}, {"interleave": 1}),
        (PAIR, {"interleave": None}),
        ({**PAIR, "mode": "push"}, {"interleave": 1}),
        (PAIR, {"interleave": 1, "dcp": 4}),
        ({"interleave": 16, "pair": True}, {"interleave": 16}),
        ({"block": 512, "phys": 8, "interleave": 1}, {}),
        # Pair mode moves half-blocks of equal per-rank blocks, without PCP.
        (PAIR, {"interleave": 1, "block": 512}),
        (PAIR, {"interleave": 1, "pcp": 2}),
    ],
    ids=[
        "no-opt-in",
        "peer-at-block-interleave",
        "decoder-at-block-interleave",
        "peer-without-interleave",
        "push-mode",
        "dcp-ratio-not-2",
        "interleave-neither-1-nor-block",
        "with-block-size-mismatch",
        "pair-with-block-size-mismatch",
        "pair-with-peer-pcp",
    ],
)
def test_validate_hybrid_hetero_dcp_interleave_mismatch_is_fatal(
    worker_kwargs, meta_kwargs
):
    """Any other interleave combination needs a different mapping, so it stops
    the decoder (NixlPeerConfigMismatch) even alongside non-fatal problems."""
    with pytest.raises(NixlPeerConfigMismatch, match="cp_kv_cache_interleave_size"):
        _validate_hybrid(worker_kwargs, meta_kwargs)


@pytest.mark.parametrize(
    "worker_kwargs,meta_kwargs,match",
    [
        ({"block": 512, "phys": 8, "interleave": 512}, {}, "block sizes differ"),
        ({"dcp": 8}, {}, "DCP must span TP"),
        ({"tp": 8, "dcp": 8}, {"dcp": 16}, "must be >="),
    ],
)
def test_validate_hybrid_hetero_dcp_rejects(worker_kwargs, meta_kwargs, match):
    """With block-sized interleaves the other problems still only fail the
    handshake, as before."""
    with pytest.raises(RuntimeError, match=match) as exc_info:
        _validate_hybrid(worker_kwargs, meta_kwargs)
    assert not isinstance(exc_info.value, NixlPeerConfigMismatch)


@pytest.mark.parametrize(
    "remote_tp,use_mla,match",
    [(16, True, "DCP must span TP"), (8, False, "must be MLA")],
)
def test_validate_hybrid_hetero_dcp_pair_mode_problems_are_fatal(
    remote_tp, use_mla, match
):
    """Pair mode keeps no per-request fallback: any problem stops the decoder."""
    w = _worker_stub(**PAIR)
    w.use_mla = use_mla
    with pytest.raises(NixlPeerConfigMismatch, match=match):
        w._validate_hybrid_hetero_dcp(_meta(interleave=1), remote_tp, 8)


@pytest.mark.parametrize(
    "local_dcp,local_interleave,remote_dcp,remote_interleave,fatal",
    [
        (16, 1, 8, 1, False),
        (16, 1024, 8, 1024, False),
        (16, 1, 8, 1024, True),
        (16, 1024, 8, 1, True),
        (16, 1, 8, None, True),
        (16, 1024, 8, None, True),
        # A side with DCP 1 never resolves its interleave and does not use it.
        (16, 1, 1, 1024, False),
        (16, 1024, 1, None, False),
        (1, 1, 8, 1024, False),
    ],
)
def test_validate_remote_parallel_config_interleave(
    local_dcp, local_interleave, remote_dcp, remote_interleave, fatal
):
    """DCP peers must use the same interleave (vllm#60484); a DCP peer that does
    not advertise one predates the field and is rejected too."""
    w = _worker_stub(interleave=local_interleave, dcp=local_dcp)
    meta = _meta(interleave=remote_interleave, dcp=remote_dcp)
    if not fatal:
        w._validate_remote_parallel_config(meta)
        return
    with pytest.raises(NixlPeerConfigMismatch, match="same cp_kv_cache_interleave"):
        w._validate_remote_parallel_config(meta)


@pytest.mark.parametrize(
    "worker_kwargs,meta_kwargs,ok",
    [
        ({}, {}, True),
        (PAIR, {"interleave": 1}, True),
        ({"interleave": 1}, {"interleave": 1}, False),
        (PAIR, {"interleave": 1, "block": 512}, False),
        (PAIR, {"interleave": 1, "pcp": 2}, False),
        # DCP 1 <-> DCP n: only the DCP-sharded side's interleave matters.
        ({"tp": 4, "dcp": 4}, {"dcp": 1, "interleave": 1}, True),
        ({"tp": 4, "dcp": 4, "interleave": 1}, {"dcp": 1, "interleave": 1}, False),
        ({"tp": 4, "dcp": 1, "interleave": 1}, {"dcp": 4}, True),
        ({"tp": 4, "dcp": 1}, {"dcp": 4, "interleave": 1}, False),
        ({"tp": 4, "dcp": 1}, {"dcp": 4, "interleave": None}, False),
    ],
)
def test_pure_mla_hetero_dcp_interleave(worker_kwargs, meta_kwargs, ok):
    """Pure MLA across DCP sizes needs block-sized interleaves (vllm#60483),
    with pair mode as the one exception; a mismatch used to mis-pair silently
    and now stops the decoder."""
    w = _worker_stub(**worker_kwargs)
    meta = _meta(**meta_kwargs)
    if ok:
        assert w._hetero_dcp_interleave_problem(meta, meta.dcp_size) is None
        return
    w._has_mamba = False
    w.transfer_topo = MagicMock()
    w.transfer_topo.get_engine_info.return_value = SimpleNamespace(
        remote_tp_size=meta.dcp_size, remote_dcp_size=meta.dcp_size
    )
    with pytest.raises(NixlPeerConfigMismatch, match="different DCP sizes"):
        w._validate_remote_agent_handshake(meta, meta.dcp_size, meta.dcp_size)


_AGENT_FIELDS = dict(
    engine_id="e",
    agent_metadata=b"",
    kv_caches_base_addr=[0],
    device_id=0,
    num_blocks=1,
    block_lens=[1],
    block_strides=[1],
    kv_cache_layout="HND",
    block_size=16,
    ssm_sizes=(0, 0),
    attn_backend_name="x",
    physical_blocks_per_logical_kv_block=1,
)


def _handshake_worker(monkeypatch, worker_kwargs, peer_interleave):
    """Decoder rank whose real handshake thread talks to a mocked TP8/DCP8
    prefiller; ``add_remote_agent`` runs only the hybrid hetero-DCP check."""
    peer = {**_AGENT_FIELDS, "engine_id": "P", "block_size": 1024, "dcp_size": 8}
    if peer_interleave is not None:
        peer["cp_kv_cache_interleave_size"] = peer_interleave
    payload = NixlHandshakePayload("hash", msgspec.msgpack.encode(peer))
    sock = MagicMock()
    sock.recv_multipart.return_value = [
        msgspec.msgpack.encode(payload),
        msgspec.msgpack.encode(time.perf_counter()),
    ]
    monkeypatch.setattr(base_worker, "zmq_ctx", lambda *_: contextlib.nullcontext(sock))
    monkeypatch.setattr(
        base_worker, "current_platform", SimpleNamespace(set_device=lambda _: None)
    )

    w = _worker_stub(**worker_kwargs)
    w.engine_id, w.device_id, w._is_csa_linear, w.use_host_buffer = "D", 0, False, False
    w.compat_hash, w.enforce_compat_hash = "hash", True
    w.transfer_topo = MagicMock()
    w.transfer_topo.handshake_target_ranks.return_value = [0]
    w.add_remote_agent = lambda meta, rank, tp, dcp: w._validate_hybrid_hetero_dcp(
        meta, tp, dcp
    )
    w._engine_ttl = 0
    w._handshake_lock = threading.RLock()
    w._handshake_initiation_executor = ThreadPoolExecutor(max_workers=1)
    w._handshake_futures, w._remote_agents, w._recving_metadata = {}, {}, {}
    w._engine_clock_offset, w._engine_last_active, w._engine_by_address = {}, {}, {}
    w._ready_requests, w._failed_recv_reqs = queue.Queue(), queue.Queue()
    w.xfer_stats = MagicMock()
    return w


def _pull_from_peer(w) -> None:
    meta = ReqMeta(
        local_block_ids=([1],),
        local_physical_block_ids=([1],),
        tp_size=8,
        dcp_size=8,
        remote=RemoteMeta(([2],), "p", 1, "P", "p-req"),
    )
    w._background_nixl_handshake("req", "P", meta)
    # Joins the handshake thread, after its done callbacks.
    w._handshake_initiation_executor.shutdown(wait=True)


@pytest.mark.parametrize(
    "worker_kwargs,peer_interleave",
    [(PAIR, None), (PAIR, 1024), ({"interleave": 1}, 1)],
    ids=["peer-without-interleave", "peer-at-block-interleave", "no-opt-in"],
)
def test_handshake_interleave_mismatch_stops_the_decoder(
    monkeypatch, worker_kwargs, peer_interleave
):
    """A mismatch found in the handshake thread never fails the request into a
    recompute: the main thread's next completion poll raises, and keeps
    raising."""
    w = _handshake_worker(monkeypatch, worker_kwargs, peer_interleave)
    _pull_from_peer(w)
    assert w._ready_requests.empty() and w._failed_recv_reqs.empty()
    for poll in (w.get_transfer_results, w.get_finished):
        with pytest.raises(NixlPeerConfigMismatch, match="cp_kv_cache_interleave"):
            poll()


def test_handshake_accepts_pair_mode(monkeypatch):
    w = _handshake_worker(monkeypatch, PAIR, 1)
    _pull_from_peer(w)
    assert w._ready_requests.get_nowait()[0] == "req"
    assert "P" in w._remote_agents and w._peer_config_mismatch is None


def test_handshake_other_failures_still_fail_the_request(monkeypatch):
    w = _handshake_worker(monkeypatch, PAIR, 1)
    w.add_remote_agent = MagicMock(side_effect=RuntimeError("NIXL error"))
    _pull_from_peer(w)
    assert w._failed_recv_reqs.get_nowait() == "req"
    assert w._peer_config_mismatch is None


@pytest.mark.parametrize("draft", [False, True])
def test_handshake_covers_the_draft_heads_owner(monkeypatch, draft):
    """Without KDA state, decoder rank 5 (TP16/DCP16) reads its MLA KV from P
    rank 5; a head-sharded draft at DCP 1 adds P rank 2, which owns its draft
    head, so the handshake loads that agent too."""
    w = _handshake_worker(monkeypatch, PAIR, 1)
    w.transfer_topo = _topology(5, 16, 16, is_mamba=False)
    w.add_remote_agent = lambda meta, rank, tp, dcp: f"agent-{rank}"
    if draft:
        w._head_sharded_draft_kv_heads = DRAFT_KV_HEADS
        w._head_sharded_draft_regions = {1}
        w._head_sharded_draft_groups = {1}
    sock = base_worker.zmq_ctx().__enter__()
    agents, _ = w._nixl_handshake("p", 1, 8, "P", 8)
    asked = sorted(msgspec.msgpack.decode(c.args[0])[2] for c in sock.send.mock_calls)
    assert asked == ([2, 5] if draft else [5])
    assert sorted(agents) == [(0, rank) for rank in asked]


_MLA_SPEC = MLAAttentionSpec(
    block_size=1024, num_kv_heads=1, head_size=576, dtype=torch.uint8
)


def _pair_startup_worker():
    """A K3.1 decoder rank: TP16/DCP16, interleave 1, V2 runner, pull."""
    w = _worker_stub(**PAIR)
    w.vllm_config.use_v2_model_runner = True
    w.vllm_config.model_config = SimpleNamespace(enable_sleep_mode=False)
    w.pp_size, w.use_host_buffer = 1, False
    w.kv_cache_config = SimpleNamespace(
        transfer_group_ids=(0, 1), kv_cache_groups=[None, None]
    )
    w._layer_specs = {"layers.0.mla": _MLA_SPEC, "layers.1.kda": _kda_spec(16)}
    return w


@pytest.mark.parametrize(
    "path,attr,value,match",
    [
        ("", "_dcp_pair_exchange", "false", "JSON boolean"),
        ("vllm_config.parallel_config", "cp_kv_cache_interleave_size", 1024, "size 1"),
        ("", "dcp_size", 8, "DCP must equal TP"),
        ("", "pp_size", 2, "PP 1"),
        ("vllm_config", "use_v2_model_runner", False, "V2 model runner"),
        ("", "_TRANSFER_MODE", "push", "NixlPullConnector"),
        ("", "use_host_buffer", True, "host transfer buffers"),
        ("vllm_config.model_config", "enable_sleep_mode", True, "sleep mode"),
        ("", "_physical_blocks_per_logical_kv_block", 2, "kernel block"),
        ("kv_cache_config", "transfer_group_ids", (0,), "transfer group"),
        (
            "",
            "_layer_specs",
            {
                "layers.0.mla": _MLA_SPEC,
                "draft.mla": replace(_MLA_SPEC, dcp_sharded=False),
            },
            "DCP-sharded.*draft.mla",
        ),
    ],
)
def test_dcp_pair_exchange_startup_checks(path, attr, value, match):
    w = _pair_startup_worker()
    w._validate_dcp_pair_exchange_config()
    setattr(operator.attrgetter(path)(w) if path else w, attr, value)
    with pytest.raises(ValueError, match=match):
        w._validate_dcp_pair_exchange_config()


# The release DSpark draft at TP16: one of its 16 KV heads of 64 per rank, a
# 2048-token window, DCP 1 under the DCP target. K3.1 numbers its layers 0-92;
# the DFlash draft's 5 layers follow them.
_DRAFT_SPEC = SlidingWindowSpec(
    block_size=1024,
    num_kv_heads=1,
    head_size=64,
    dtype=torch.bfloat16,
    sliding_window=2048,
    dcp_sharded=False,
)
_TARGET_LAYERS, _NUM_DRAFT_LAYERS = 93, 5
_DRAFT_LAYERS = {
    f"model.layers.{_TARGET_LAYERS + i}.self_attn.attn": _DRAFT_SPEC
    for i in range(_NUM_DRAFT_LAYERS)
}


def test_pair_startup_exempts_the_release_draft():
    """Pair mode exchanges the DCP-sharded target pages. The release draft at
    DCP 1 holds every token on every rank and is pulled whole, so its
    unsharded layers pass the startup check; any other unsharded attention
    layer still fails it, as does the draft without a head-sharded draft
    config (it would not be pulled whole)."""
    w = _pair_startup_worker()
    w.model_config = SimpleNamespace(get_total_num_hidden_layers=lambda: _TARGET_LAYERS)
    w.vllm_config.speculative_config = SimpleNamespace(
        draft_model_config=SimpleNamespace(
            get_total_num_hidden_layers=lambda: _NUM_DRAFT_LAYERS
        )
    )
    w._head_sharded_draft_kv_heads = DRAFT_KV_HEADS
    w.kv_cache_config = SimpleNamespace(
        transfer_group_ids=(0, 1, 2), kv_cache_groups=[None, None, None]
    )
    w._layer_specs = {**w._layer_specs, **_DRAFT_LAYERS}
    w._validate_dcp_pair_exchange_config()

    w._layer_specs["model.layers.7.self_attn.attn"] = _DRAFT_SPEC
    with pytest.raises(ValueError, match="DCP-sharded.*model.layers.7"):
        w._validate_dcp_pair_exchange_config()
    del w._layer_specs["model.layers.7.self_attn.attn"]
    w._head_sharded_draft_kv_heads = None
    with pytest.raises(ValueError, match="DCP-sharded.*model.layers.93"):
        w._validate_dcp_pair_exchange_config()


# Regions of the decoder stub below: MLA (0, 1), KDA state (2-4), the draft (5);
# transfer groups: MLA, 3 x KDA, the draft (4).
_DRAFT_REGION, _DRAFT_TRANSFER_GROUP = 5, 4
_TARGET_REGIONS_SHARDED = [True, True, False, False, False]


def _draft_decoder(replicated: bool = True, tp: int = 16, **worker_kwargs):
    """A K3.1 decoder rank (TP16/DCP16 in pair mode by default) with the
    release draft, whose handshake reaches the draft checks."""
    w = _worker_stub(tp=tp, dcp=tp, **{**PAIR, **worker_kwargs})
    w._has_mamba, w._mamba_state_geometry = True, None
    w._head_sharded_draft_kv_heads = DRAFT_KV_HEADS
    w._head_sharded_draft_regions = {_DRAFT_REGION}
    w._group_region_indices = [[0, 1], [2], [3], [4], [_DRAFT_REGION]]
    w._dcp_replicated_draft_groups = frozenset(
        {_DRAFT_TRANSFER_GROUP} if replicated else ()
    )
    return w


def _draft_peer(draft_sharded: bool | None = False, **meta_kwargs):
    """A prefiller (TP8/DCP8 by default) advertising whether DCP shards its
    draft region (None: a peer predating the field)."""
    meta = _meta(**{"interleave": 1, **meta_kwargs})
    meta.mamba_state_geometry = None
    meta.region_dcp_sharded = (
        None if draft_sharded is None else [*_TARGET_REGIONS_SHARDED, draft_sharded]
    )
    return meta


def _handshake_draft(w, meta, tp_ratio: int = 2, block_size_ratio: int = 1):
    w.transfer_topo = MagicMock(tp_size=w.world_size)
    w.transfer_topo.tp_ratio.return_value = tp_ratio
    w.transfer_topo.block_size_ratio.return_value = block_size_ratio
    w.transfer_topo.get_engine_info.return_value = SimpleNamespace(
        remote_tp_size=meta.dcp_size, remote_dcp_size=meta.dcp_size
    )
    w._validate_remote_agent_handshake(meta, meta.dcp_size, meta.dcp_size)


@pytest.mark.parametrize(
    "worker_kwargs,meta_kwargs,tp_ratio",
    [
        ({}, {}, 2),
        ({"tp": 4, "interleave": 1}, {"dcp": 4}, 1),
        ({"tp": 4, "interleave": 1}, {"dcp": 4, "draft_sharded": None}, 1),
    ],
    ids=["tp8dcp8-to-tp16dcp16-pair", "tp4dcp4-both", "tp4dcp4-both-older-peer"],
)
def test_release_draft_handshake_accepted(worker_kwargs, meta_kwargs, tp_ratio):
    """The chosen topologies pass the draft checks: P TP8/DCP8 -> D
    TP16/DCP16 in pair mode, and TP4/DCP4 on both sides (C512), where a peer
    predating the advertised DCP sharding lays the draft out alike."""
    w = _draft_decoder(**worker_kwargs)
    meta = _draft_peer(**meta_kwargs)
    assert w._head_sharded_draft_dcp_problem(meta, meta.dcp_size, tp_ratio, 1) is None


@pytest.mark.parametrize(
    "decoder_kwargs,meta_kwargs,match",
    [
        ({}, {"draft_sharded": True}, r"DCP shards the peer's draft regions \[5\]"),
        ({}, {"draft_sharded": None}, "does not advertise"),
        ({"replicated": False}, {}, "a draft that DCP shards needs equal TP"),
    ],
    ids=["peer-draft-dcp-sharded", "older-peer", "local-draft-dcp-sharded"],
)
@pytest.mark.parametrize("pair", [True, False], ids=["pair-mode", "block-mode"])
def test_release_draft_handshake_rejected(decoder_kwargs, meta_kwargs, match, pair):
    """A draft whose pages do not hold every token on both sides cannot pair
    ranks by heads alone. Pair mode stops the decoder (no per-request
    fallback); block mode (interleave = block) fails the handshake as before."""
    if pair:
        w = _draft_decoder(**decoder_kwargs)
        meta = _draft_peer(**meta_kwargs)
        error = NixlPeerConfigMismatch
    else:
        w = _draft_decoder(**decoder_kwargs, interleave=1024, pair=False)
        meta = _draft_peer(**{**meta_kwargs, "interleave": 1024})
        error = NotImplementedError
    with pytest.raises(error, match=match) as exc_info:
        _handshake_draft(w, meta)
    assert "head-sharded draft KV under an MLA target with DCP" in str(exc_info.value)


@pytest.mark.parametrize(
    "decoder_kwargs,args,match",
    [
        ({"mode": "push"}, (2, 1), "push mode pairs ranks 1:1"),
        ({}, (2, 2), "per-rank blocks differ"),
        ({}, (-2, 1), "local TP must be >= remote TP"),
    ],
)
def test_release_draft_problems(decoder_kwargs, args, match):
    """The other limits of pulling the draft by heads under DCP."""
    w = _draft_decoder(**decoder_kwargs)
    problem = w._head_sharded_draft_dcp_problem(_draft_peer(), 8, *args)
    assert problem is not None and match in problem


# P's draft list as request_finished sends it: null blocks (0) before its
# window, then its blocks up to a speculative lookahead block past its 1100
# tokens (block size 64: positions 13-17, lookahead 18).
_P_DRAFT_BLOCKS = [0] * 13 + [113, 114, 115, 116, 117, 118]


@pytest.mark.parametrize(
    "local,remote,num_remote_blocks,expected",
    [
        ([16, 17], _P_DRAFT_BLOCKS, 18, ([16, 17], [116, 117])),
        ([4, 5, 6, 7], _P_DRAFT_BLOCKS, 18, ([4, 5, 6, 7], [114, 115, 116, 117])),
        ([44, 45, 46], [143, 144, 145, 146], 47, ([44, 45, 46], [144, 145, 146])),
        ([1, 2, 3], [10, 11], None, ([2, 3], [10, 11])),
        ([], [10, 11], 2, ([], [])),
    ],
    ids=["hit-in-window", "no-hit", "window-clipped", "local-longer", "full-hit"],
)
def test_pair_blocks_from_end(local, remote, num_remote_blocks, expected):
    """A DCP-replicated draft's blocks pair by position from the end of both
    lists, after dropping the prefiller's blocks past its tokens."""
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import (
        NixlConnectorWorker,
    )

    got = NixlConnectorWorker._pair_blocks_from_end(local, remote, num_remote_blocks)
    assert got == expected


def _geometry(tp: int, num_spec: int = 7, dtype: torch.dtype = torch.bfloat16):
    spec = _kda_spec(tp, num_spec=num_spec)
    return tp, [list(s) for s in spec.shapes], [str(dtype)] * len(spec.dtypes)


def _geometry_check(local, remote):
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import (
        NixlConnectorWorker,
    )

    w = object.__new__(NixlConnectorWorker)
    w._mamba_state_geometry = local
    meta = SimpleNamespace(engine_id="remote", mamba_state_geometry=remote)
    w._validate_mamba_state_geometry(meta)


@pytest.mark.parametrize(
    "local,remote",
    [
        (_geometry(8), _geometry(16)),
        (_geometry(16), _geometry(8)),
        (_geometry(16, num_spec=3), _geometry(16, num_spec=3)),
        (_geometry(8), None),
        (None, _geometry(16)),
    ],
)
def test_mamba_state_geometry_accepts(local, remote):
    """Same per-head state and conv rows across TP sizes, or a peer (or local
    side) without Mamba geometry, passes."""
    _geometry_check(local, remote)


@pytest.mark.parametrize(
    "local,remote",
    [
        (_geometry(8, num_spec=3), _geometry(16, num_spec=7)),
        (_geometry(16, num_spec=7), _geometry(8, num_spec=3)),
        (_geometry(8, num_spec=7), _geometry(16, num_spec=3)),
        (_geometry(16, num_spec=3), _geometry(8, num_spec=7)),
        (_geometry(8), _geometry(16, dtype=torch.float32)),
    ],
)
def test_mamba_state_geometry_rejects(local, remote):
    """A DL3 prefiller with a DL7 decoder (either side, either TP direction)
    differs in conv rows, and a state dtype mismatch differs in bytes: both
    would transfer misaligned state, so the handshake must fail."""
    with pytest.raises(RuntimeError, match="Mamba state geometry differs"):
        _geometry_check(local, remote)


def test_agent_metadata_carries_mamba_geometry():
    """The geometry survives the msgpack handshake encoding; a peer built
    before the field existed decodes as None (check skipped)."""
    geometry = _geometry(8)
    sent = NixlAgentMetadata(**_AGENT_FIELDS, mamba_state_geometry=geometry)
    got = msgspec.msgpack.decode(msgspec.msgpack.encode(sent), type=NixlAgentMetadata)
    assert tuple(got.mamba_state_geometry) == tuple(geometry)
    old = msgspec.msgpack.decode(
        msgspec.msgpack.encode(NixlAgentMetadata(**_AGENT_FIELDS)),
        type=NixlAgentMetadata,
    )
    assert old.mamba_state_geometry is None


def test_agent_metadata_carries_interleave():
    """The interleave survives the handshake encoding; a peer built before the
    field existed decodes as None, which the DCP checks treat as unknown."""
    sent = NixlAgentMetadata(**_AGENT_FIELDS, cp_kv_cache_interleave_size=1)
    got = msgspec.msgpack.decode(msgspec.msgpack.encode(sent), type=NixlAgentMetadata)
    assert got.cp_kv_cache_interleave_size == 1
    older = msgspec.msgpack.decode(
        msgspec.msgpack.encode(_AGENT_FIELDS), type=NixlAgentMetadata
    )
    assert older.cp_kv_cache_interleave_size is None


def test_agent_metadata_carries_region_dcp_sharding():
    """Which regions DCP shards survives the handshake encoding; a peer built
    before the field existed decodes as None (unknown)."""
    sent = NixlAgentMetadata(**_AGENT_FIELDS, region_dcp_sharded=[False])
    got = msgspec.msgpack.decode(msgspec.msgpack.encode(sent), type=NixlAgentMetadata)
    assert got.region_dcp_sharded == [False]
    older = msgspec.msgpack.decode(
        msgspec.msgpack.encode(_AGENT_FIELDS), type=NixlAgentMetadata
    )
    assert older.region_dcp_sharded is None


@pytest.mark.parametrize(
    "gpu,expected",
    [
        (0, "mlx5_0:1,mlx5_1:1"),
        (1, "mlx5_0:1,mlx5_1:1"),
        (2, "mlx5_6:1"),
        (3, "mlx5_6:1"),
        (4, "inherited"),
    ],
)
def test_select_ucx_net_devices(monkeypatch, gpu, expected):
    """Per-GPU UCX_NET_DEVICES from the connector's ucx_net_devices_by_gpu table."""
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import (
        NixlConnectorWorker,
    )

    monkeypatch.setenv("UCX_NET_DEVICES", "inherited")
    NixlConnectorWorker._select_ucx_net_devices(
        "0-1=mlx5_0:1,mlx5_1:1; 2-3=mlx5_6:1", gpu=gpu
    )
    import os

    assert os.environ["UCX_NET_DEVICES"] == expected
    NixlConnectorWorker._select_ucx_net_devices(None, gpu=gpu)
    assert os.environ["UCX_NET_DEVICES"] == expected


def test_select_ucx_net_devices_plus_separator(monkeypatch):
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import (
        NixlConnectorWorker,
    )

    monkeypatch.setenv("UCX_NET_DEVICES", "inherited")
    NixlConnectorWorker._select_ucx_net_devices("0-1=a:1+b:1;2-3=c:1+d:1", gpu=3)
    import os

    assert os.environ["UCX_NET_DEVICES"] == "c:1,d:1"
