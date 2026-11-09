# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP mapping computation for NIXL KV cache transfers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vllm.distributed.kv_transfer.kv_connector.utils import (
    BlockIds,
    TransferTopology,
)
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheSpec, MambaSpec

# ======================================================================
# Data structures
# ======================================================================


@dataclass(frozen=True)
class ReadSpec:
    """Specification for a single remote block read operation."""

    remote_rank: int
    local_block_ids: BlockIds
    remote_block_ids: BlockIds
    block_ids_by_region: bool = False


def _is_attention_spec(spec_type: type[KVCacheSpec]) -> bool:
    return issubclass(spec_type, AttentionSpec)


def _is_ssm_spec(spec_type: type[KVCacheSpec]) -> bool:
    return issubclass(spec_type, MambaSpec)


@dataclass(frozen=True)
class TPMapping:
    """Complete local-to-remote TP mapping for one remote engine.

    Generated once per remote engine during handshake.
    """

    # Remote TP ranks that this local rank reads from, per group.
    # Position = local piece index.
    source_ranks_per_group: tuple[tuple[int, ...], ...]

    # Superset of all source ranks (union of all groups).
    all_source_ranks: tuple[int, ...]

    # Maps each source rank to its FA head slot index.
    rank_to_attention_slot: dict[int, int]

    # FA head offset factor for hetero-TP (D_TP > P_TP).
    rank_offset_factor: int

    # Local ranks (in aggregate) that read from a given source rank. The producer frees
    # a request's blocks only once that many notifications have come in.
    local_consumers: int = 1

    # Per-source-rank override of ``local_consumers``. Hybrid attention+SSM with
    # different DCP sizes reads attention KV and SSM state from different remote
    # ranks, so remote ranks have different reader counts.
    consumers_by_rank: dict[int, int] | None = None

    def consumers_for(self, remote_rank: int) -> int:
        """Local ranks (in aggregate) that read from ``remote_rank``."""
        if self.consumers_by_rank is None:
            return self.local_consumers
        return self.consumers_by_rank.get(remote_rank, self.local_consumers)


# ======================================================================
# TP mapping computation
# ======================================================================


def compute_head_offset(
    tp_rank: int, tp_size: int, remote_tp_size: int, total_num_kv_heads: int
) -> int:
    """Return the offset in local head slices within the remote KV block."""
    if tp_size <= remote_tp_size:
        return 0
    if tp_size > total_num_kv_heads:
        remote_rank = tp_rank * remote_tp_size // tp_size
        local_head = tp_rank * total_num_kv_heads // tp_size
        remote_head = remote_rank * total_num_kv_heads // remote_tp_size
        return local_head - remote_head
    return tp_rank % (tp_size // remote_tp_size)


def compute_tp_mapping(
    transfer_topology: TransferTopology,
    remote_tp_size: int,
    group_spec_types: tuple[type[KVCacheSpec], ...],
    remote_dcp_size: int = 1,
    head_sharded_kv_heads: int | None = None,
) -> TPMapping:
    """Build the complete local-to-remote TP mapping.

    Computes source ranks, head slot assignments, and the rank offset
    factor in a single pass.

    DCP support is scoped to MLA only, with a side is either fully replicated or fully
    sharded. DCP-branch reuses the same rank set used at handshake selection.

    ``head_sharded_kv_heads`` maps head-sharded KV with that many total KV heads
    instead of the topology's, e.g. a GQA draft under an MLA target.
    """
    tp_rank = transfer_topology.tp_rank
    tp_size = transfer_topology.tp_size
    is_mla = transfer_topology.is_mla and head_sharded_kv_heads is None
    total_num_kv_heads = head_sharded_kv_heads or transfer_topology.total_num_kv_heads
    attn_ranks = _attention_source_ranks(
        transfer_topology,
        tp_rank,
        remote_tp_size,
        remote_dcp_size,
        is_mla=is_mla,
        total_num_kv_heads=total_num_kv_heads,
    )

    # --- SSM source ranks ---
    # SSM state is head-sharded over TP and never DCP-sharded, so its source
    # ranks follow the TP ratio even when DCP moves the attention source.
    has_ssm = any(_is_ssm_spec(t) for t in group_spec_types)
    ssm_ranks = (
        transfer_topology.ssm_source_ranks(remote_tp_size, tp_rank=tp_rank)
        if has_ssm
        else []
    )

    all_ranks = sorted(set(attn_ranks) | set(ssm_ranks))

    # --- Per-group ordered source ranks ---
    source_ranks_per_group = tuple(
        tuple(ssm_ranks) if _is_ssm_spec(t) else tuple(attn_ranks)
        for t in group_spec_types
    )

    # --- Attention head slots ---
    head_to_slot: dict[int, int] = {}
    for i, r in enumerate(attn_ranks):
        head_to_slot[r * total_num_kv_heads // remote_tp_size] = i
    rank_to_attention_slot = {
        r: head_to_slot.get(r * total_num_kv_heads // remote_tp_size, 0)
        for r in all_ranks
    }

    # --- Rank offset factor ---
    rank_offset_factor = (
        0
        if is_mla
        else compute_head_offset(tp_rank, tp_size, remote_tp_size, total_num_kv_heads)
    )

    local_consumers = transfer_topology.dcp_consumer_count(
        remote_tp_size, remote_dcp_size
    )

    # Hybrid attention+SSM: count, for every remote rank, the local ranks that
    # read anything (attention KV or SSM state) from it. With equal DCP sizes
    # (or no DCP) the attention and SSM sources coincide and every count equals
    # ``local_consumers``; with different DCP sizes they do not.
    consumers_by_rank = None
    if has_ssm and transfer_topology.dcp_size != remote_dcp_size:
        readers: dict[int, set[int]] = {}
        for rank in range(tp_size):
            sources = set(
                _attention_source_ranks(
                    transfer_topology,
                    rank,
                    remote_tp_size,
                    remote_dcp_size,
                    is_mla=is_mla,
                    total_num_kv_heads=total_num_kv_heads,
                )
            ) | set(transfer_topology.ssm_source_ranks(remote_tp_size, tp_rank=rank))
            for source in sources:
                readers.setdefault(source, set()).add(rank)
        consumers_by_rank = {source: len(r) for source, r in readers.items()}

    return TPMapping(
        source_ranks_per_group=source_ranks_per_group,
        all_source_ranks=tuple(all_ranks),
        rank_to_attention_slot=rank_to_attention_slot,
        rank_offset_factor=rank_offset_factor,
        local_consumers=local_consumers,
        consumers_by_rank=consumers_by_rank,
    )


def _attention_source_ranks(
    transfer_topology: TransferTopology,
    tp_rank: int,
    remote_tp_size: int,
    remote_dcp_size: int,
    is_mla: bool | None = None,
    total_num_kv_heads: int | None = None,
) -> list[int]:
    """Remote ranks that local rank ``tp_rank`` reads attention KV from.
    ``is_mla`` / ``total_num_kv_heads`` default to the topology's."""
    tp_size = transfer_topology.tp_size
    if is_mla is None:
        is_mla = transfer_topology.is_mla
    if total_num_kv_heads is None:
        total_num_kv_heads = transfer_topology.total_num_kv_heads
    if is_mla or tp_size >= remote_tp_size:
        if is_mla and remote_dcp_size > 1:
            return transfer_topology.dcp_source_ranks(
                remote_tp_size,
                remote_dcp_size,
                local_dcp_rank=tp_rank % transfer_topology.dcp_size,
            )
        # D (local TP) > P (remote TP): multiple local ranks read different chunks
        # from *one* remote rank, corresponding to different kv heads.
        # For MLA, we only need one remote since cache is duplicated. When
        # P TP=k*TP k, this will spread mla ranks to read from remote k*tp_rank.
        return [tp_rank * remote_tp_size // tp_size]
    # P (remote TP) > D (local TP): one local rank
    # reads from multiple remote ranks.
    # GQA dedup: when K < remote_tp_size, several remote ranks
    # hold the same KV head.  np.unique keeps only the first
    # rank per unique head so we don't issue redundant reads.
    abs_tp = remote_tp_size // tp_size
    start = tp_rank * abs_tp
    heads = np.arange(start, start + abs_tp) * total_num_kv_heads // remote_tp_size
    _, unique_idx = np.unique(heads, return_index=True)
    return (start + np.sort(unique_idx)).tolist()
