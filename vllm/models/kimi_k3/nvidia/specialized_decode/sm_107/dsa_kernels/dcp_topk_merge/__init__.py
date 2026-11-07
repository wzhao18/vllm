# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The exact one-round DCP decode top-k merge as the SM107 specialized decode runs
it (``IndexerStage.merge_rows``), vendored from the source branch.

This tree's stock ``vllm.v1.attention.ops.dcp_topk_merge`` has the same packed
layout and merge, but its round 1 is always the NCCL all-gather. The source branch
added a round-1 multicast mailbox (``VLLM_DSA_BPRIME_MAILBOX=1``, ``kernels.py``:
the pack publishes each row into every rank's symmetric mailbox, the select polls
it). The specialized decode keeps its own copy, with its own rescue workspace and
mailbox, so the stock indexer op (which keeps the stock module and its workspace)
is untouched. The merge is bit-identical either way.

Packed candidate rows (the "B'" layout), int32 ``[T, N + N/32]`` per rank: see the
stock module.
"""

import os
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

# The packed candidate word keeps local[15:0]; a bit plane carries local[16].
LOCAL_INDEX_BITS = 17
# The saturation word holds a 16-bit rank mask.
MAX_WORLD_SIZE = 16
_WARP = 32

# Env reads kept as the source branch's ``vllm.envs`` parsed them (parity).
MAILBOX_ENV = "VLLM_DSA_BPRIME_MAILBOX"
CHECK_ENV = "VLLM_DSA_BPRIME_CHECK"


def _env_flag(name: str) -> bool:
    return bool(int(os.getenv(name, "0")))


def packed_row_words(local_topk: int) -> int:
    """int32 words per packed row: N candidate words + N/32 bit-plane words."""
    return local_topk + local_topk // _WARP


def rescue_slots(topk: int, local_topk: int) -> int:
    """Upper bound on the saturated ranks of one row in the exact merge: a
    saturated rank sent N words, all strictly above the K-th selected key, so at
    most ``(K - 1) // N``."""
    return (topk - 1) // local_topk


# (topk, local_topk) -> rows of this copy's rescue workspace.
_MERGE_MAX_ROWS: dict[tuple[int, int], int] = {}
# (topk, local_topk) -> whether round 1 runs through the mailbox.
_MAILBOX: dict[tuple[int, int], bool] = {}


def dcp_topk_merge_max_rows(topk: int, local_topk: int) -> int:
    """Rows one merge call may hold (0 when this copy was not initialized)."""
    return _MERGE_MAX_ROWS.get((topk, local_topk), 0)


def dcp_topk_merge_fits(num_rows: int, topk: int, local_topk: int) -> bool:
    """Whether a decode step of ``num_rows`` rows can take the one-round merge
    (identical on every DCP rank)."""
    return local_topk > 0 and num_rows <= dcp_topk_merge_max_rows(topk, local_topk)


def dcp_topk_merge_mailbox(topk: int, local_topk: int) -> bool:
    return _MAILBOX.get((topk, local_topk), False)


def interleave_one(vllm_config: Any) -> bool:
    """The source branch's ``cp_kv_cache_interleave_one`` gate for this copy: the
    packed ids decode as ``local * W + rank`` only at interleave 1. This tree
    finalizes the interleave before the model is built, so the current value is
    the final one."""
    return vllm_config.parallel_config.cp_kv_cache_interleave_size == 1


def init_dcp_topk_merge(vllm_config: Any, topk: int, local_topk: int) -> int:
    """Set up this copy of the merge for the stock indexer's N (``local_topk``,
    the stock ``init_dcp_topk_merge`` result) and return N, or 0 when the merge is
    unavailable (N 0, interleave other than 1, ubatching, no symmetric memory).
    Collective over the DCP group, outside CUDA graph capture, after the weights
    load. Raises for ``VLLM_DSA_BPRIME_CHECK=1`` without a usable merge."""
    key = (topk, local_topk)
    if key in _MERGE_MAX_ROWS or not local_topk or not interleave_one(vllm_config):
        return local_topk if key in _MERGE_MAX_ROWS else 0
    parallel_config = vllm_config.parallel_config
    if parallel_config.use_ubatching:
        logger.info_once("DSA DCP top-k merge (SM107): not set up under ubatching.")
        return 0
    import torch

    from vllm.distributed import get_dcp_group
    from vllm.v1.attention.ops.cp_common import direct_cp_enabled
    from vllm.v1.attention.ops.dcp import get_dcp_workspace_max_num_tokens

    group = get_dcp_group()
    max_rows = get_dcp_workspace_max_num_tokens(vllm_config)
    agree = torch.tensor(
        [int(direct_cp_enabled(group, torch.int32, None))], dtype=torch.int32
    )
    torch.distributed.all_reduce(
        agree, op=torch.distributed.ReduceOp.MIN, group=group.cpu_group
    )
    if not agree.item():
        logger.warning_once(
            "DSA DCP top-k merge (SM107): no symmetric memory over the DCP group; "
            "the specialized decode runs the stock op and the local-slot filter."
        )
        return 0
    from .kernels import init_dcp_topk_mailbox, init_dcp_topk_rescue_workspace

    init_dcp_topk_rescue_workspace(max_rows, topk, local_topk, group)
    _MERGE_MAX_ROWS[key] = max_rows
    kv_transfer = vllm_config.kv_transfer_config
    prefill_only = (
        kv_transfer is not None
        and kv_transfer.is_kv_producer
        and not kv_transfer.is_kv_consumer
    )
    mailbox = (
        _env_flag(MAILBOX_ENV)
        and not prefill_only
        and init_dcp_topk_mailbox(topk, local_topk, group)
    )
    _MAILBOX[key] = bool(mailbox)
    if _env_flag(CHECK_ENV):
        from ...compat.dcp_topk_merge_check import init_dcp_topk_merge_check

        init_dcp_topk_merge_check(
            torch.device("cuda", torch.accelerator.current_device_index()),
            group.rank_in_group,
        )
    logger.info_once(
        "DSA DCP top-k merge (SM107 specialized decode): exact one-round merge of "
        "each rank's local top-%d (rescue workspace for %d rows; round 1 by %s).",
        local_topk,
        max_rows,
        "multicast mailbox" if mailbox else "all-gather",
    )
    if not mailbox and _env_flag(MAILBOX_ENV):
        logger.warning_once(
            "DSA DCP top-k merge (SM107): %s is set, but round 1 stays on the "
            "all-gather (%s).",
            MAILBOX_ENV,
            "prefill-only instance"
            if prefill_only
            else "no multicast memory over the DCP group",
        )
    return local_topk
