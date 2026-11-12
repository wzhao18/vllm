# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP mapping computation for NIXL KV cache transfers."""

from __future__ import annotations

import enum
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from vllm.distributed.kv_transfer.kv_connector.dcp_layout import DcpLayout
from vllm.distributed.kv_transfer.kv_connector.utils import (
    BlockIds,
    TransferTopology,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils import (
    get_dcp_replicated_group_ids,
    get_representative_spec_type,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    CircularBufferSpec,
    KVCacheSpec,
    MambaSpec,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

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


class DcpTransferMode(enum.Enum):
    """How attention KV moves between a remote and a local DCP layout
    (`select_dcp_transfer_mode`); each mode has its own page pairing."""

    DIRECT = "direct"
    """Pages pair whole by position, nothing is copied after the transfer:
    equal per-rank logical blocks of the same layout, or pages of consecutive
    positions (`_apply_dcp_prefix_caching`)."""
    DIRECT_ROWS = "direct_rows"
    """Same DCP size and interleave but other per-rank logical blocks: a rank
    holds the same rows in the same order on both sides, so kernel pages pair
    by row (`_pair_same_dcp_layout_pages`); nothing is copied either."""
    RELAYOUT = "relayout"
    """Local ranks read the remote pages holding their rows into staging
    blocks, then gather the rows into their own pages
    (`kv_connector.dcp_relayout`)."""


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

    # How attention KV moves from this remote (`select_dcp_transfer_mode`).
    dcp_mode: DcpTransferMode = DcpTransferMode.DIRECT

    def consumers(self, remote_rank: int) -> int:
        """Notifications `remote_rank` waits for per request. A mapping with
        counts has one for every rank this rank notifies; a missing one would
        let the producer free a request's blocks too early, so it is an
        error. A mapping built without counts has one reader per producer."""
        if not self.consumers_per_rank:
            return 1
        return self.consumers_per_rank[remote_rank]

    def rank_block_ids(
        self,
        block_ids: BlockIds,
        rank: int,
        key_groups: Sequence[int] | None = None,
    ) -> list[list[int]]:
        """`block_ids` with the entries `rank` is no source of emptied.

        Entries are per cache group, or per region with `key_groups` giving
        each region's cache group (-1: a region shared by several groups,
        read from every source rank). Attention groups come from their
        attention sources and Mamba state groups from their state sources
        (`TransferTopology.state_source_ranks`), whatever the transfer mode.
        """
        groups = range(len(block_ids)) if key_groups is None else key_groups
        return [
            list(ids)
            if rank
            in (self.source_ranks_per_group[g] if g >= 0 else self.all_source_ranks)
            else []
            for ids, g in zip(block_ids, groups, strict=True)
        ]


def engine_dcp_layout(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
) -> DcpLayout:
    """Where an engine's DCP ranks hold a request's attention KV rows: its
    interleave, and the per-rank logical block of its attention cache groups
    (the lcm of their block sizes; Mamba state and circular buffers are not
    placed by position).

    The scheduler (which advertises it in `kv_transfer_params` and sizes the
    relayout staging from it) and the workers (which advertise it at the
    handshake and plan the transfers with it) both derive it here.
    """
    block_sizes = [
        group.kv_cache_spec.block_size
        for group in kv_cache_config.transfer_groups
        if get_representative_spec_type(group.kv_cache_spec)
        not in (MambaSpec, CircularBufferSpec)
    ]
    parallel_config = vllm_config.parallel_config
    return DcpLayout(
        dcp_size=parallel_config.decode_context_parallel_size,
        interleave=parallel_config.cp_kv_cache_interleave_size,
        block_size=math.lcm(*block_sizes)
        if block_sizes
        else vllm_config.cache_config.block_size,
    )


def relayout_config_problem(
    kv_cache_config: KVCacheConfig,
    *,
    transfer_mode: str,
    use_host_buffer: bool,
    pcp_size: int,
) -> str | None:
    """Why an engine with this configuration cannot receive KV through a
    `DcpTransferMode.RELAYOUT`, or None.

    Shared by the scheduler (which then reserves no staging) and the workers
    (which reject such a peer at the handshake, adding the checks only the
    registered caches can answer).
    """
    if transfer_mode != "pull":
        return "push transfers write whole pages."
    if use_host_buffer:
        return "KV staged through a host buffer is not relayouted."
    groups = kv_cache_config.transfer_groups
    if any(group.host_resident for group in groups):
        return "host-resident KV cache groups (HiSparse) are not relayouted."
    if pcp_size > 1:
        return "prefill context parallelism is not supported."
    if get_dcp_replicated_group_ids(groups):
        return (
            "DCP-replicated KV cache groups (e.g. a separate MLA draft cache) "
            "are not supported."
        )
    num_attention_groups = sum(
        _is_attention_spec(get_representative_spec_type(group.kv_cache_spec))
        for group in groups
    )
    if num_attention_groups != 1:
        return (
            "it needs exactly one attention KV cache group, got "
            f"{num_attention_groups}."
        )
    return None


def select_dcp_transfer_mode(
    local: DcpLayout,
    remote: DcpLayout,
    *,
    tp_sizes: tuple[int, int],
) -> DcpTransferMode:
    """Pick how attention KV moves from `remote` to `local`, or raise
    `NotImplementedError` for a pair of layouts no mode supports.

    `block_size` is each side's per-rank logical block (rows per page on one
    rank) and `tp_sizes` are the (local, remote) transfer TP sizes.

    Supported pairs:

    - Different DCP sizes need each side's DCP group to be 1 or its whole TP
      group, and the sizes to divide one another: the source-rank and
      consumer-count math (`TransferTopology`) pairs one DCP rank per TP rank.
    - Both sides sharding (DCP > 1) need the same interleave. RELAYOUT itself
      maps rows between any interleaves, but every supported model declares
      its interleave, so other interleaves mean other model configurations.

    The mode then follows from the layouts:

    - DIRECT: no DCP; or same layout and per-rank blocks; or pages of
      consecutive positions on both sides (`interleave == block_size`, or no
      DCP on that side) of equal size, so page `j` of any rank is one run of
      positions and maps to one remote page whatever the DCP sizes.
    - DIRECT_ROWS: same DCP size and interleave, other per-rank blocks.
    - RELAYOUT: everything else (e.g. token-level interleave over different
      DCP sizes, as Kimi K3.1 P DCP8 -> D DCP16).

    Whether the engine can receive through a relayout is for the caller to
    check (`relayout_config_problem`). Attention KV only: Mamba state is never
    DCP-sharded, and comes from its own source ranks
    (`TransferTopology.state_source_ranks`) whatever the mode.
    """
    if local.dcp_size == 1 and remote.dcp_size == 1:
        return DcpTransferMode.DIRECT
    if local.dcp_size != remote.dcp_size:
        for side, layout, tp_size in (
            ("local", local, tp_sizes[0]),
            ("remote", remote, tp_sizes[1]),
        ):
            if layout.dcp_size not in (1, tp_size):
                raise NotImplementedError(
                    "NIXL transfers between different DCP sizes need DCP equal "
                    f"to TP (or no DCP) on each side; the {side} side has TP "
                    f"{tp_size} and DCP {layout.dcp_size}."
                )
        if local.dcp_size % remote.dcp_size and remote.dcp_size % local.dcp_size:
            raise NotImplementedError(
                "NIXL transfers between different DCP sizes need sizes that "
                f"divide one another, got local {local.dcp_size}, remote "
                f"{remote.dcp_size}."
            )
    if (
        local.dcp_size > 1
        and remote.dcp_size > 1
        and local.interleave != remote.interleave
    ):
        raise NotImplementedError(
            "Both P and D shard KV with decode context parallelism but their "
            f"cp_kv_cache_interleave_size differ: local {local.interleave}, "
            f"remote {remote.interleave}. Use the same interleave on both sides."
        )
    if local.dcp_size == remote.dcp_size:
        if local.block_size == remote.block_size:
            return DcpTransferMode.DIRECT
        return DcpTransferMode.DIRECT_ROWS
    if (
        local.block_size == remote.block_size
        and local.pages_hold_contiguous_positions
        and remote.pages_hold_contiguous_positions
    ):
        return DcpTransferMode.DIRECT
    return DcpTransferMode.RELAYOUT


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
    dcp_mode: DcpTransferMode = DcpTransferMode.DIRECT,
) -> TPMapping:
    """Build the complete local-to-remote TP mapping.

    Computes source ranks, head slot assignments, and the rank offset
    factor in a single pass.

    DCP support is scoped to MLA only, with a side is either fully replicated or fully
    sharded. DCP-branch reuses the same rank set used at handshake selection.

    `head_sharded_kv_heads` maps head-sharded KV with that many total KV heads
    instead of the topology's, e.g. a GQA draft under an MLA target.
    `dcp_mode` is how attention KV moves (`select_dcp_transfer_mode`).
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
        dcp_mode=dcp_mode,
    )
