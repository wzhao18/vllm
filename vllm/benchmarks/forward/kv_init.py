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
    import os

    if os.environ.get("DCP8FIX_RANDKV"):
        _random_kv_blocks(kv_caches, attn_groups, batch, num_blocks)


def _random_kv_blocks(kv_caches, attn_groups, batch, num_blocks) -> None:
    """DIAG (DCP8FIX r2): non-zero random KV in the case's blocks, finite in every
    format: BF16/FP32 tensors N(0, s); FP8 MLA latent bytes = E4M3 of N(0, 1);
    the MXFP4 indexer cache ([B x 64 B E2M1 | B x 4 B UE8M0] per block) gets
    random E2M1 bytes and UE8M0 scales 2^-4..2^0. Seeded per rank (fixed)."""
    import torch.distributed as dist

    rank = dist.get_rank() if dist.is_initialized() else 0
    gen = torch.Generator(device="cuda").manual_seed(1234 + rank)
    seen = []
    for group in attn_groups:
        ranges = batch.group_block_ranges[group.kv_cache_group_id]
        mamba = isinstance(group.kv_cache_spec, MambaSpec)
        for name in group.layer_names:
            cache = kv_caches.get(name)
            if group.kernel_block_stride:
                cache = group.map_kv_cache_to_manager_blocks(cache, num_blocks)
            for tensor in _tensors(cache):
                dim, factor = _block_dim(tensor, num_blocks)
                for begin, end in ranges:
                    v = tensor.narrow(dim, begin * factor, (end - begin) * factor)
                    _fill_random(v, name, mamba, gen)
                if len(seen) < 12 and (name.split(".")[-1], tensor.dtype) not in seen:
                    seen.append((name.split(".")[-1], tensor.dtype))
                    print(f"[randkv] {name} {tuple(tensor.shape)} {tensor.dtype} "
                          f"stride {tensor.stride()} mamba {mamba}", flush=True)


def _fill_random(v: torch.Tensor, name: str, mamba: bool, gen) -> None:
    if v.numel() == 0:
        return
    if v.dtype in (torch.bfloat16, torch.float16, torch.float32):
        scale = 0.02 if (mamba and v.dtype == torch.float32) else 0.5
        v.copy_(torch.randn(v.shape, generator=gen, device=v.device) * scale)
        return
    if v.dtype == torch.float8_e4m3fn:
        v.copy_(torch.randn(v.shape, generator=gen, device=v.device).to(v.dtype))
        return
    assert v.dtype == torch.uint8, (name, v.dtype)
    if not v.is_contiguous():
        # Block stride wider than a block (shared allocation): fill per block.
        for i in range(v.shape[0]):
            _fill_random(v[i : i + 1], name, mamba, gen)
        return
    blk = v.view(v.shape[0], -1)
    nbytes = blk.shape[1]
    if nbytes % 68 == 0 and "indexer" in name:
        tokens = nbytes // 68
        vals = torch.randint(0, 256, (blk.shape[0], tokens * 64), generator=gen,
                             device=v.device, dtype=torch.int32).to(torch.uint8)
        sc = torch.randint(123, 128, (blk.shape[0], tokens * 4), generator=gen,
                           device=v.device, dtype=torch.int32).to(torch.uint8)
        blk[:, : tokens * 64].copy_(vals)
        blk[:, tokens * 64 : tokens * 68].copy_(sc)
        return
    # FP8 (E4M3) MLA latent + rope stored as bytes.
    e4m3 = torch.randn(blk.shape, generator=gen, device=v.device).to(torch.float8_e4m3fn)
    blk.copy_(e4m3.view(torch.uint8))


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
