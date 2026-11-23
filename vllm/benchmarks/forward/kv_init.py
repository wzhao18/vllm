# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV blocks of a benchmark case: initialization and the blocks a step writes.

Only the case's blocks are touched, group by group: groups can alias one
allocation with different layouts, and the batch builder keeps their block IDs
disjoint. Zero KV times within a few percent of random KV, even for sparse
attention (DeepSeek-V4.1, ~100K context), so zeros are the only init for now.
"""

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import torch

from vllm.benchmarks.forward.batch import BenchBatch, kv_block_size
from vllm.v1.kv_cache_interface import MambaSpec

if TYPE_CHECKING:
    from vllm.v1.worker.utils import AttentionGroup


def _tensors(kv_cache) -> Iterator[torch.Tensor]:
    if isinstance(kv_cache, torch.Tensor):
        if kv_cache.numel():
            yield kv_cache
    elif isinstance(kv_cache, (list, tuple)):
        for item in kv_cache:
            yield from _tensors(item)


def _block_dim(tensor: torch.Tensor, num_blocks: int) -> tuple[int, int]:
    """Return (block dim, kernel blocks per allocator block)."""
    for dim in range(min(2, tensor.dim())):
        if tensor.shape[dim] >= num_blocks and tensor.shape[dim] % num_blocks == 0:
            return dim, tensor.shape[dim] // num_blocks
    raise ValueError(f"No block dim of size {num_blocks} in {tuple(tensor.shape)}")


def _group_tensors(
    kv_caches: dict[str, Any], group: "AttentionGroup", num_blocks: int
) -> Iterator[torch.Tensor]:
    for name in group.layer_names:
        cache = kv_caches.get(name)
        if group.kernel_block_stride:
            assert isinstance(cache, torch.Tensor)
            cache = group.map_kv_cache_to_manager_blocks(cache, num_blocks)
        yield from _tensors(cache)


@torch.inference_mode()
def zero_kv_blocks(
    kv_caches: dict[str, Any],
    attn_groups: list["AttentionGroup"],
    batch: BenchBatch,
    num_blocks: int,
) -> None:
    for group in attn_groups:
        ranges = batch.group_block_ranges[group.kv_cache_group_id]
        for tensor in _group_tensors(kv_caches, group, num_blocks):
            dim, factor = _block_dim(tensor, num_blocks)
            for begin, end in ranges:
                tensor.narrow(dim, begin * factor, (end - begin) * factor).zero_()


def touched_kv_views(
    kv_caches: dict[str, Any],
    attn_groups: list["AttentionGroup"],
    batch: BenchBatch,
    num_blocks: int,
    dcp_size: int,
    lookahead: int = 0,
) -> list[torch.Tensor]:
    """Views a step may write: attention tails and all recurrent state slots."""
    scheduled_tokens = [
        batch.expected[req_id][0] + lookahead for req_id in batch.req_ids
    ]
    views = []
    for group in attn_groups:
        ranges = batch.group_block_ranges[group.kv_cache_group_id]
        block_size = kv_block_size(group.kv_cache_spec, dcp_size)
        for tensor in _group_tensors(kv_caches, group, num_blocks):
            dim, factor = _block_dim(tensor, num_blocks)
            for (begin, end), tokens in zip(ranges, scheduled_tokens):
                first = (
                    begin
                    if isinstance(group.kv_cache_spec, MambaSpec)
                    else max(begin, end - (-(-tokens // block_size) + 2))
                )
                views.append(tensor.narrow(dim, first * factor, (end - first) * factor))
    return views


def snapshot_state(
    views: list[torch.Tensor], free_bytes: int | None = None
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], str]:
    """Restore copies of the state a step writes, and where they live.

    Kept on the device when the copies fit in half the free device memory (the
    other half stays for capture and the timed run); otherwise in pinned host
    memory, restored by an H2D copy. Both restore the same bytes; the host copy
    only lets large recurrent-state snapshots (e.g. FP32 KDA state of B256 x
    (k+1) slots at u132k, ~45 GiB per rank) coexist with the KV pool."""
    total = sum(v.numel() * v.element_size() for v in views)
    if free_bytes is None:
        free_bytes = (
            torch.cuda.mem_get_info(views[0].device)[0]
            if views and views[0].is_cuda
            else total * 2
        )
    if total * 2 <= free_bytes:
        return [(v, v.clone()) for v in views], "device"
    pin = torch.cuda.is_available()
    saved = []
    for v in views:
        host = torch.empty(v.shape, dtype=v.dtype, device="cpu", pin_memory=pin)
        host.copy_(v)
        saved.append((v, host))
    return saved, "host"


def restore_state(snapshots: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
    for view, saved in snapshots:
        view.copy_(saved, non_blocking=True)

