# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hybrid MLA + Mamba (GDN/KDA) NIXL pulls between peers with different DCP sizes,
e.g. a TP8/DCP8 prefiller feeding a TP16/DCP16 decoder.

MLA KV is DCP-sharded in interleave-sized chunks and matched by global position;
the Mamba state is TP-sharded by heads and never DCP-sharded, so a decoder rank can
read its MLA blocks and its Mamba state from different prefiller ranks.
No GPU or NIXL required.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.utils import TransferTopology
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import (
    compute_tp_mapping,
)
from vllm.v1.kv_cache_interface import MambaSpec, MLAAttentionSpec

# Kimi K3's hybrid layout: one MLA group, then KDA (Mamba) groups.
HYBRID_GROUPS = (MLAAttentionSpec, MambaSpec, MambaSpec, MambaSpec)


def _topology(tp_rank: int, tp_size: int, dcp_size: int) -> TransferTopology:
    topo = object.__new__(TransferTopology)
    topo.tp_rank = tp_rank
    topo.tp_size = tp_size
    topo.is_mla = True
    topo.is_mamba = True
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


def _worker_stub(block=896, phys=14, interleave=896, tp=16, dcp=16):
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
    return w


def _meta(block=896, phys=14, interleave=896):
    return SimpleNamespace(
        block_size=block // phys,
        physical_blocks_per_logical_kv_block=phys,
        cp_kv_cache_interleave_size=interleave,
        pcp_size=1,
    )


def test_validate_hybrid_hetero_dcp_accepts_block_interleave():
    _worker_stub()._validate_hybrid_hetero_dcp(_meta(), 8, 8)


@pytest.mark.parametrize(
    "worker_kwargs,meta_kwargs,remote_tp,match",
    [
        ({"interleave": 1}, {}, 8, "interleave"),
        ({}, {"interleave": 1}, 8, "interleave"),
        ({"block": 512, "phys": 8, "interleave": 512}, {}, 8, "block sizes differ"),
        ({"dcp": 8}, {}, 8, "DCP must span TP"),
        ({"tp": 8, "dcp": 8}, {}, 16, "must be >="),
    ],
)
def test_validate_hybrid_hetero_dcp_rejects(
    worker_kwargs, meta_kwargs, remote_tp, match
):
    with pytest.raises(RuntimeError, match=match):
        _worker_stub(**worker_kwargs)._validate_hybrid_hetero_dcp(
            _meta(**meta_kwargs), remote_tp, remote_tp
        )


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
    import msgspec

    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
        NixlAgentMetadata,
    )

    fields = dict(
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
    geometry = _geometry(8)
    sent = NixlAgentMetadata(**fields, mamba_state_geometry=geometry)
    got = msgspec.msgpack.decode(msgspec.msgpack.encode(sent), type=NixlAgentMetadata)
    assert tuple(got.mamba_state_geometry) == tuple(geometry)
    old = msgspec.msgpack.decode(
        msgspec.msgpack.encode(NixlAgentMetadata(**fields)), type=NixlAgentMetadata
    )
    assert old.mamba_state_geometry is None
