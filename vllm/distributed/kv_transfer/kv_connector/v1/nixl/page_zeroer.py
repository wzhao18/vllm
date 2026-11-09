# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Zero the local attention pages a NIXL pull leaves unwritten.

The scheduler skips alloc-time zeroing for the blocks an async load fills
(``skip_zeroing_group_ids``), since zeroing would race the RDMA write. Any page
of those blocks that the pull does not write therefore keeps its previous
owner's bytes: another request's KV, or a Mamba state that now reads as KV.
Attention kernels can consume whole tiles past the sequence end and mask them
by multiplication, so a stale NaN still leaks into the output. The pull worker
zeroes the complement of the pages it writes, and on failure every page.
"""

import itertools
from collections.abc import Mapping, Sequence

import torch

from vllm.logger import init_logger
from vllm.utils.torch_utils import async_tensor_h2d

logger = init_logger(__name__)

# (absolute address, block stride, page bytes, number of blocks), all in bytes.
Segment = tuple[int, int, int, int]


def unwritten_blocks(
    local_block_ids: Sequence[int], written: set[int], ratio: int = 1
) -> list[int]:
    """Return the blocks of ``local_block_ids`` that no written unit touches.

    ``written`` holds the units the pull's descriptors cover. These are kernel
    block ids when ``ratio`` is 1, and remote-sized sub-blocks
    ``block * ratio + i`` otherwise. A partly covered block counts as touched:
    the receive post-process zeroes its token tail after any layout conversion.
    """
    touched = {unit // ratio for unit in written} if ratio > 1 else written
    return list(dict.fromkeys(b for b in local_block_ids if b not in touched))


def page_segments(cache: torch.Tensor, flat_page_bytes: int | None) -> list[Segment]:
    """Byte segments that ``cache.index_fill_(0, ids, 0)`` would clear.

    ``cache`` is block-major at dim 0. Dims laid out outside the block stride
    (head groups under LHBNC, say) each get their own segment, as in
    ``KVBlockZeroer``. A flat 1-D byte view has no block dim, so its pages
    abut and are ``flat_page_bytes`` long.
    """
    el = cache.element_size()
    if cache.ndim == 1:
        assert flat_page_bytes, "a flat cache view needs its page size"
        num_blocks = cache.numel() * el // flat_page_bytes
        return [(cache.data_ptr(), flat_page_bytes, flat_page_bytes, num_blocks)]
    stride = cache.stride(0) * el
    outer = [d for d in range(1, cache.ndim) if cache.stride(d) * el > stride]
    inner = [d for d in range(1, cache.ndim) if d not in outer]
    page = el + sum((cache.shape[d] - 1) * cache.stride(d) * el for d in inner)
    segments = []
    for index in itertools.product(*(range(cache.shape[d]) for d in outer)):
        offset = sum(i * cache.stride(d) * el for i, d in zip(index, outer))
        segments.append((cache.data_ptr() + offset, stride, page, cache.shape[0]))
    return segments


class PageZeroer:
    """Zero the attention pages of given kernel blocks, per transfer group.

    The segment table is built once, after the KV caches are registered. On
    CUDA, each group is one launch of the stock KV-block zeroing kernel.
    Elsewhere, and for pages that are not 4-byte aligned, each segment gets a
    strided byte view and an ``index_fill_``.

    ``zero_async`` runs on a side stream that first waits for the current
    stream, so it is ordered after every earlier use of the blocks yet overlaps
    the next forward pass instead of running before it. ``step_fence``, called
    once per step after the forward is queued, makes the current stream wait
    for the zeroing of earlier steps, which has finished by then.
    """

    def __init__(
        self,
        caches_by_group: Mapping[int, Sequence[torch.Tensor]],
        flat_page_bytes_by_group: Mapping[int, int | None],
    ):
        self._segments: dict[int, list[Segment]] = {}
        self._byte_views: dict[int, list[torch.Tensor]] = {}
        self._kernel_meta: dict[int, tuple] = {}
        self.device: torch.device | None = None
        for group, caches in caches_by_group.items():
            widest: dict[int, int] = {}
            views: dict[int, torch.Tensor] = {}
            segments: dict[int, Segment] = {}
            for cache in caches:
                if self.device is None:
                    self.device = cache.device
                storage = cache.untyped_storage()
                for seg in page_segments(cache, flat_page_bytes_by_group.get(group)):
                    addr, stride, page, num_blocks = seg
                    # Overlaid layers share an address; keep the widest page.
                    if addr in segments and page <= widest[addr]:
                        assert segments[addr][1] == stride
                        continue
                    widest[addr] = page
                    segments[addr] = seg
                    views[addr] = torch.empty(
                        0, dtype=torch.uint8, device=cache.device
                    ).set_(
                        storage,
                        addr - storage.data_ptr(),
                        (num_blocks, page),
                        (stride, 1),
                    )
            if segments:
                self._segments[group] = list(segments.values())
                self._byte_views[group] = list(views.values())
        self._stream = None
        # Last zeroing event queued in this step, and in earlier steps.
        self._fresh: torch.cuda.Event | None = None
        self._aged: torch.cuda.Event | None = None
        if self.device is not None and self.device.type == "cuda":
            self._build_kernel_meta()
            self._stream = torch.cuda.Stream(device=self.device)

    def _build_kernel_meta(self) -> None:
        for group, segments in self._segments.items():
            if any(
                addr % 4 or stride % 4 or page % 4 for addr, stride, page, _ in segments
            ):
                continue
            max_page = max(page for _, _, page, _ in segments) // 4
            blk = min(1 << (max_page - 1).bit_length(), 1024)
            self._kernel_meta[group] = (
                torch.tensor(
                    [s[0] for s in segments], dtype=torch.uint64, device=self.device
                ),
                torch.tensor(
                    [s[1] // 4 for s in segments], dtype=torch.int64, device=self.device
                ),
                torch.tensor(
                    [s[2] // 4 for s in segments], dtype=torch.int64, device=self.device
                ),
                (max_page + blk - 1) // blk,
                blk,
                len(segments),
            )

    @property
    def groups(self) -> tuple[int, ...]:
        return tuple(self._segments)

    def zero(self, ids_by_group: Mapping[int, Sequence[int]]) -> None:
        """Zero the pages of the given blocks on the current stream."""
        for group, block_ids in ids_by_group.items():
            if not block_ids or group not in self._segments:
                continue
            assert self.device is not None
            idx = async_tensor_h2d(list(block_ids), self.device, torch.int64)
            meta = self._kernel_meta.get(group)
            if meta is None:
                for view in self._byte_views[group]:
                    view.index_fill_(0, idx, 0)
                continue
            from vllm.v1.worker.utils import _zero_kv_blocks_kernel

            addrs, strides, pages, chunks, blk, num_segments = meta
            _zero_kv_blocks_kernel[(len(block_ids), num_segments, chunks)](
                addrs, strides, pages, idx, BLOCK_SIZE=blk
            )

    def zero_async(self, ids_by_group: Mapping[int, Sequence[int]]) -> None:
        """Zero off the current stream, ordered after the work queued on it."""
        if not any(ids_by_group.values()):
            return
        if self._stream is None:
            self.zero(ids_by_group)
            return
        self._stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self._stream):
            self.zero(ids_by_group)
            event = torch.cuda.Event()
            event.record(self._stream)
        self._fresh = event

    def step_fence(self, include_fresh: bool = False) -> None:
        """Order the current stream after the zeroing of earlier steps.

        With ``include_fresh``, also after this step's: needed only when a
        block zeroed this step may be read by the next forward. Events on the
        side stream complete in order, so waiting on the newest one suffices.
        """
        event = self._fresh if include_fresh and self._fresh else self._aged
        if event is not None:
            torch.cuda.current_stream(self.device).wait_event(event)
        self._aged = None if include_fresh else self._fresh
        self._fresh = None
