# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP mapping computation for NIXL KV cache transfers."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

import numpy as np

from vllm.distributed.kv_transfer.kv_connector.dcp_layout import DcpLayout
from vllm.distributed.kv_transfer.kv_connector.utils import (
    BlockIds,
    TransferTopology,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils import (
    NixlPeerIncompatibleError,
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

    # Remote rank -> local ranks (in aggregate) that notify it about a request,
    # for every remote rank this rank notifies (`compute_tp_mapping` fills it).
    # The producer frees a request's blocks only once that many notifications
    # have come in. Counts differ per producer when attention and Mamba state
    # come from different ranks.
    consumers_per_rank: dict[int, int] = field(default_factory=dict)

    def consumers(self, remote_rank: int) -> int:
        """Notifications `remote_rank` waits for per request; a rank without a
        count has this rank as its only reader (a one-to-one mapping)."""
        return self.consumers_per_rank.get(remote_rank, 1)


class DcpTransferMode(enum.Enum):
    """How attention KV moves between a remote and a local DCP layout."""

    DIRECT = "direct"
    """Local pages read remote pages as they are; nothing is copied after the
    transfer."""


def select_dcp_transfer_mode(local: DcpLayout, remote: DcpLayout) -> DcpTransferMode:
    """Pick how attention KV moves from `remote` to `local`, or raise
    `NixlPeerIncompatibleError` for a pair of layouts no mode supports.

    `block_size` is each side's logical block (rows per page on one rank).
    DIRECT pairs pages of one rank with pages of another, with no copy after
    the transfer:

    - Same DCP size and interleave, any block sizes: a rank's row stream is
      the same on both sides, so local row `j` is remote row `j` of the same
      rank, and pages pair at kernel-page granularity whatever the logical
      blocks are (`_pair_same_dcp_layout_pages`).
    - Pages of consecutive positions on both sides (`interleave ==
      block_size`, or no DCP) and equal blocks: page `j` of any rank is one
      run of positions, so it maps to one remote page whatever the DCP sizes
      (`_apply_dcp_prefix_caching`).

    Attention KV only: Mamba state is never DCP-sharded, and comes from its
    own source ranks (`TransferTopology.state_source_ranks`) whatever the
    mode. Mode selection assumes each side's DCP group spans its TP ranks
    (`dcp_size` 1 or TP).
    """
    if local.dcp_size == 1 and remote.dcp_size == 1:
        return DcpTransferMode.DIRECT
    if (
        local.dcp_size > 1
        and remote.dcp_size > 1
        and local.interleave != remote.interleave
    ):
        raise NixlPeerIncompatibleError(
            "Both P and D shard KV with decode context parallelism but their "
            f"cp_kv_cache_interleave_size differ: local {local.interleave}, "
            f"remote {remote.interleave}. Use the same interleave on both sides."
        )
    if local.dcp_size == remote.dcp_size:
        return DcpTransferMode.DIRECT
    if (
        local.block_size == remote.block_size
        and local.pages_hold_contiguous_positions
        and remote.pages_hold_contiguous_positions
    ):
        return DcpTransferMode.DIRECT
    # The only layouts left move rows between pages after the transfer (e.g.
    # Kimi K3.1 P DCP8 -> D DCP16 at interleave 1).
    raise NixlPeerIncompatibleError(
        "NIXL transfers between different DCP sizes need pages of consecutive "
        "positions of equal size on both sides (cp_kv_cache_interleave_size "
        "equal to the per-rank KV block size, or no DCP); other layouts need "
        "KV rows moved between pages after the transfer, which is not "
        f"supported yet: local {local}, remote {remote}."
    )


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

    `head_sharded_kv_heads` maps head-sharded KV with that many total KV heads
    instead of the topology's, e.g. a GQA draft under an MLA target.
    """
    tp_rank = transfer_topology.tp_rank
    tp_size = transfer_topology.tp_size
    is_mla = transfer_topology.is_mla and head_sharded_kv_heads is None
    total_num_kv_heads = head_sharded_kv_heads or transfer_topology.total_num_kv_heads
    # --- Attention source ranks ---
    if is_mla or tp_size >= remote_tp_size:
        if is_mla and remote_dcp_size > 1:
            attn_ranks = transfer_topology.dcp_source_ranks(
                remote_tp_size, remote_dcp_size
            )
        else:
            # D (local TP) > P (remote TP): multiple local ranks read different chunks
            # from *one* remote rank, corresponding to different kv heads.
            # For MLA, we only need one remote since cache is duplicated. When
            # P TP=k*TP k, this will spread mla ranks to read from remote k*tp_rank.
            attn_ranks = [tp_rank * remote_tp_size // tp_size]
    else:
        # P (remote TP) > D (local TP): one local rank
        # reads from multiple remote ranks.
        # GQA dedup: when K < remote_tp_size, several remote ranks
        # hold the same KV head.  np.unique keeps only the first
        # rank per unique head so we don't issue redundant reads.
        abs_tp = remote_tp_size // tp_size
        start = tp_rank * abs_tp
        heads = np.arange(start, start + abs_tp) * total_num_kv_heads // remote_tp_size
        _, unique_idx = np.unique(heads, return_index=True)
        attn_ranks = (start + np.sort(unique_idx)).tolist()

    # --- SSM source ranks ---
    # Mamba state follows TP even where attention KV follows DCP.
    has_ssm = any(_is_ssm_spec(t) for t in group_spec_types)
    ssm_ranks = transfer_topology.state_source_ranks(remote_tp_size) if has_ssm else []

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

    consumers_per_rank = {
        rank: transfer_topology.consumer_count(rank, remote_tp_size, remote_dcp_size)
        for rank in transfer_topology.handshake_target_ranks(
            remote_tp_size, remote_dcp_size
        )
    }

    return TPMapping(
        source_ranks_per_group=source_ranks_per_group,
        all_source_ranks=tuple(all_ranks),
        rank_to_attention_slot=rank_to_attention_slot,
        rank_offset_factor=rank_offset_factor,
        consumers_per_rank=consumers_per_rank,
    )
