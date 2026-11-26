# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sequence-parallel all-gather of the Kimi K3 decoder layers over NVLink
symmetric memory.

Under sequence parallelism every decoder layer gathers the TP ranks' token
shards of its attention input ([T / tp, hidden] per rank -> [T, hidden]).
Above the size the stock custom (IPC) all-gather takes, that went to NCCL,
which runs it as RING / SIMPLE over 16 channels: 16 CTAs at ~25 GB/s each,
0.40 TB/s of the 1.25 TB/s NVLink ingress (TP4 C512 prefill at 17.4K tokens:
463 us per layer, 94 layers per step).

Here each rank pushes its shard with 16-byte stores straight into every
rank's symmetric output buffer (one persistent kernel over all SMs), then one
symmetric-memory barrier makes all shards visible: 191 us at 17.4K tokens
(0.98 TB/s). The output is a view of the buffer, so the gather does no extra
copy. The result is bitwise the NCCL all-gather (a pure copy). On the
40-layer TP4 SP prefill step: -11.7 ms (-5.8%) at 17.4K tokens, -5.4 ms
(-4.7%) at 8.2K.

Allocation (``sp_all_gather_rows``): a prefill or unified instance allocates,
at model init (so the KV sizing accounts for it), two slots of the token shard
of a max_num_batched_tokens step (896 MiB per GPU at 32K tokens, TP4) when that
shard is above the custom all-gather's size. No step of the instance gathers
more, so a larger shard raises. A P/D decode instance (KV consumer only)
allocates nothing: its decode steps are small (C512: 128 rows per rank, inside
the custom all-gather; the specialized decode's SP step has its own
communication), a buffer for its rare local prefills (preemption recompute) up
to max_num_batched_tokens would cost KV capacity, and a smaller one would not
cover them. Without a buffer (also: no symmetric memory over the TP group,
logged) every gather takes the stock all-gather.

The final gather of the model (``KimiK3Model.forward``, with the drafter's
auxiliary hidden states packed beside the hidden states, so three times wider)
keeps the stock path: a slot of its width would add 1.5x the buffer for one
gather of 41 (~0.8 ms of a 190 ms step).

MXFP8 gather (``kimi_sp_all_gather_mxfp8``): a KDA layer's in_proj is an
online-MXFP8 linear that quantizes the gathered BF16 rows with FlashInfer's
``mxfp8_quantize`` (E4M3 + F8_128x4-swizzled E8M0 scales, 1x32 blocks along
the row). Each scale depends on its row's own values, so the ranks quantize
their shards (the AttnRes epilogue, bit for bit ``mxfp8_quantize``) and push
the FP8 rows + scale bytes (1.03 B per element instead of 2). With a shard of
a multiple of 128 rows the F8_128x4 row tiles of the gathered tensor are the
ranks' local tiles in rank order (the row-tile index is the outermost of the
layout), so the gathered scale bytes are the shards' local swizzled scales
concatenated: bitwise the quantization of the gathered BF16. Other shard sizes,
and shards the custom (Lamport) all-gather takes (BF16 only: FP8 bytes can
alias its sentinel), keep the BF16 gather (``mxfp8_gather_rows_ok``). The FP8
rows and scales share the BF16 slot (1 + 1/32 <= 2 bytes per element): rows
from byte 0, scales from byte ``world * max_rows * hidden``.

Buffer reuse: the two slots alternate over the gathers of a forward
(``begin_forward`` resets the count; the slot sequence is fixed per forward,
so CUDA graphs capture it). The push of a gather overwrites a peer's slot only
after the barrier of an earlier gather on the other slot, which that peer
reached after its stream finished every read of the slot's previous gather.
So a gathered tensor stays valid until the gather two later, and it is
consumed inside its layer. Across forwards: with an even number of gathers per
forward the last uses slot 1 and the next forward's first slot 0; with an odd
number both use slot 0, so the first push of such a forward waits on one more
barrier first (one per forward, a few us).
"""

from typing import Any

import torch

from vllm.distributed import get_tp_group
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.ops.cp_common import direct_cp_enabled, symm_mem

logger = init_logger(__name__)

# Persistent grid of one program per SM, 8192-element (16 KiB) tiles, 4 warps:
# 98 / 191 / 350 us at 2050 / 4350 / 8192 rows x 7168 per rank on TP4 VR200
# (non-persistent tiles: 117-141 / 221-264 / 404-484 us).
_BLOCK = 8192
_NUM_WARPS = 4
_NUM_SLOTS = 2


@triton.jit
def _push_shard_kernel(
    src,
    dst_addresses,
    dst_offset,
    numel,
    rank,
    WORLD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Copy ``numel`` elements of ``src`` to ``dst_offset`` of every peer
    buffer in ``dst_addresses`` (absolute addresses, [WORLD]). Persistent: one
    program per SM strides over the tiles. Peers are visited from ``rank + 1``
    on so the ranks' first stores spread over the peers."""
    for tile in range(tl.program_id(0), tl.cdiv(numel, BLOCK), tl.num_programs(0)):
        offs = tile.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < numel
        x = tl.load(src + offs, mask=mask)
        for i in tl.static_range(WORLD):
            peer = (rank + 1 + i) % WORLD
            dst = tl.load(dst_addresses + peer).to(
                tl.pointer_type(src.dtype.element_ty)
            )
            tl.store(dst + dst_offset + offs, x, mask=mask)


@triton.jit
def _push_shard2_kernel(
    src_a,
    src_b,
    dst_addresses,
    dst_offset_a,
    dst_offset_b,
    numel_a,
    numel_b,
    rank,
    WORLD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """``_push_shard_kernel`` over two segments in one persistent grid: the
    tiles of ``src_a`` (``numel_a`` elements to ``dst_offset_a``), then those
    of ``src_b``. Same element type for both (32-bit words of the MXFP8 rows
    and scale bytes)."""
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    for tile in range(pid, tl.cdiv(numel_a, BLOCK), nprog):
        offs = tile.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < numel_a
        x = tl.load(src_a + offs, mask=mask)
        for i in tl.static_range(WORLD):
            peer = (rank + 1 + i) % WORLD
            dst = tl.load(dst_addresses + peer).to(
                tl.pointer_type(src_a.dtype.element_ty)
            )
            tl.store(dst + dst_offset_a + offs, x, mask=mask)
    # Segment b continues the round robin where segment a's tiles end.
    start_b = (pid + nprog - tl.cdiv(numel_a, BLOCK) % nprog) % nprog
    for tile in range(start_b, tl.cdiv(numel_b, BLOCK), nprog):
        offs = tile.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < numel_b
        x = tl.load(src_b + offs, mask=mask)
        for i in tl.static_range(WORLD):
            peer = (rank + 1 + i) % WORLD
            dst = tl.load(dst_addresses + peer).to(
                tl.pointer_type(src_b.dtype.element_ty)
            )
            tl.store(dst + dst_offset_b + offs, x, mask=mask)


# MXFP8: 1x32 blocks along the row, F8_128x4 scale tiles of 128 rows.
MXFP8_BLOCK = 32
MXFP8_ROW_TILE = 128
_WORD_BLOCK = _BLOCK // 2  # 32-bit words per tile (16 KiB, as the BF16 tiles)


class SymmSpAllGather:
    """Two-slot symmetric output buffer of the TP group, sized for
    ``max_rows`` rows per rank (the token shard of the largest step)."""

    def __init__(
        self, max_rows: int, hidden_size: int, dtype: torch.dtype, device
    ) -> None:
        tp = get_tp_group()
        self.world = tp.world_size
        self.rank = tp.rank_in_group
        self.max_rows = max_rows
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.slot_numel = self.world * max_rows * hidden_size
        self.num_sms = torch.cuda.get_device_properties(device).multi_processor_count
        self.storage = symm_mem.empty(
            _NUM_SLOTS * self.slot_numel, dtype=dtype, device=device
        )
        self.handle = symm_mem.rendezvous(self.storage, tp.device_group.group_name)
        assert self.handle is not None, "SP all-gather rendezvous returned None"
        numel = _NUM_SLOTS * self.slot_numel
        bases = [
            self.handle.get_buffer(peer, [numel], dtype, 0).data_ptr()
            for peer in range(self.world)
        ]
        esize = torch.empty((), dtype=dtype).element_size()
        # [slot, peer] absolute address of the slot in the peer's buffer.
        self.peer_ptrs = torch.tensor(
            [
                [base + slot * self.slot_numel * esize for base in bases]
                for slot in range(_NUM_SLOTS)
            ],
            dtype=torch.int64,
            device=self.storage.device,
        )
        # Gathers of the current forward: announced and pushed so far.
        self.num_gathers = 0
        self.count = 0
        torch.accelerator.synchronize()
        self.handle.barrier()

    def begin_forward(self, num_gathers: int) -> None:
        self.num_gathers = num_gathers
        self.count = 0

    def _check_shard(self, rows: int, hidden: int, dtype: torch.dtype) -> int:
        """Validate a gather of ``rows`` x ``hidden`` ``dtype`` rows per rank
        and take its slot."""
        if hidden != self.hidden_size or dtype != self.dtype:
            raise ValueError(
                f"SP all-gather buffer holds {self.hidden_size}-wide {self.dtype} "
                f"rows, got {rows} x {hidden} {dtype}"
            )
        if rows > self.max_rows:
            # The buffer covers every step of the instance
            # (max_num_batched_tokens): a larger shard is a bug.
            raise ValueError(
                f"SP all-gather shard of {rows} rows exceeds the buffer's "
                f"{self.max_rows} rows per rank (max_num_batched_tokens)"
            )
        # The slot schedule (module docstring) holds only for the gathers the
        # forward announced.
        assert self.count < self.num_gathers, (
            f"SP all-gather {self.count + 1} of a forward that announced "
            f"{self.num_gathers} (begin_forward)"
        )
        slot = self.count % _NUM_SLOTS
        if self.count == 0 and self.num_gathers % _NUM_SLOTS:
            # The previous forward's last push used slot 0 as well.
            self.handle.barrier()
        self.count += 1
        return slot

    def gather_mxfp8(
        self, data: torch.Tensor, scale: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather an MXFP8 shard: ``data`` [rows, hidden] E4M3 (rows a
        multiple of 128) and ``scale`` its F8_128x4-swizzled E8M0 bytes
        (rows * hidden / 32). Returns the gathered [world * rows, hidden]
        E4M3 rows and the [world * rows * hidden / 32] uint8 swizzled scales
        (views of the slot; module docstring)."""
        rows, hidden = data.shape
        if rows % MXFP8_ROW_TILE or data.element_size() != 1:
            raise ValueError(
                f"SP MXFP8 all-gather: {rows} x {hidden} {data.dtype} shard "
                f"(1-byte rows, a multiple of {MXFP8_ROW_TILE})"
            )
        sf_bytes = rows * hidden // MXFP8_BLOCK
        scale = scale.reshape(-1)
        if scale.numel() * scale.element_size() != sf_bytes:
            raise ValueError(
                f"SP MXFP8 all-gather: {scale.numel()} scale elements for a "
                f"{rows} x {hidden} shard (expected {sf_bytes} bytes)"
            )
        slot = self._check_shard(rows, hidden, self.dtype)
        data_words = data.contiguous().view(torch.int32).reshape(-1)
        sf_words = scale.contiguous().view(torch.int32)
        # Slot layout in bytes: rows from 0, scales from world * max_rows *
        # hidden (inside the BF16 slot of 2 * world * max_rows * hidden).
        sf_base = self.world * self.max_rows * hidden
        _push_shard2_kernel[
            (
                min(
                    self.num_sms,
                    cdiv(data_words.numel(), _WORD_BLOCK)
                    + cdiv(sf_words.numel(), _WORD_BLOCK),
                ),
            )
        ](
            data_words,
            sf_words,
            self.peer_ptrs[slot],
            self.rank * data_words.numel(),
            (sf_base + self.rank * sf_bytes) // 4,
            data_words.numel(),
            sf_words.numel(),
            self.rank,
            WORLD=self.world,
            BLOCK=_WORD_BLOCK,
            num_warps=_NUM_WARPS,
        )
        self.handle.barrier()
        start = slot * self.slot_numel * self.storage.element_size()
        raw = self.storage.view(torch.uint8)
        out = raw[start : start + self.world * rows * hidden]
        sf = raw[start + sf_base : start + sf_base + self.world * sf_bytes]
        return out.view(data.dtype).view(self.world * rows, hidden), sf

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        rows, hidden = x.shape
        slot = self._check_shard(rows, hidden, x.dtype)
        x = x.contiguous()
        numel = rows * hidden
        src, block = x.reshape(-1), _BLOCK
        if (hidden * x.element_size()) % 4 == 0:
            # Push 32-bit words (the same bytes): Triton's 16-bit copy code
            # moves 0.86 TB/s against 1.06 for 32-bit words at 8 ranks.
            src, block = x.view(torch.int32).reshape(-1), _WORD_BLOCK
        words = src.numel()
        _push_shard_kernel[(min(self.num_sms, cdiv(words, block)),)](
            src,
            self.peer_ptrs[slot],
            self.rank * words,
            words,
            self.rank,
            WORLD=self.world,
            BLOCK=block,
            num_warps=_NUM_WARPS,
        )
        self.handle.barrier()
        start = slot * self.slot_numel
        return self.storage[start : start + self.world * numel].view(
            self.world * rows, hidden
        )


_sp_all_gather: SymmSpAllGather | None = None


def is_decode_instance(vllm_config: Any) -> bool:
    """A P/D decode instance: KV consumer, not producer."""
    kv_transfer = getattr(vllm_config, "kv_transfer_config", None)
    return (
        kv_transfer is not None
        and kv_transfer.is_kv_consumer
        and not kv_transfer.is_kv_producer
    )


def sp_all_gather_rows(vllm_config: Any, hidden_size: int, dtype: torch.dtype) -> int:
    """Rows per rank the push buffer holds for this instance (module
    docstring): the token shard of a max_num_batched_tokens step on a prefill or
    unified instance, or 0 (no buffer) on a P/D decode instance or when that
    shard fits the custom all-gather, which then takes every step's gather."""
    if is_decode_instance(vllm_config):
        return 0
    tp_size = vllm_config.parallel_config.tensor_parallel_size
    rows = cdiv(vllm_config.scheduler_config.max_num_batched_tokens, tp_size)
    return 0 if _fits_custom_all_gather(rows, hidden_size, dtype) else rows


def _fits_custom_all_gather(rows: int, hidden_size: int, dtype: torch.dtype) -> bool:
    communicator = get_tp_group().device_communicator
    ca_comm = getattr(communicator, "ca_comm", None)
    return ca_comm is not None and ca_comm.should_custom_all_gather(
        torch.empty((rows, hidden_size), dtype=dtype, device="meta")
    )


def init_sp_all_gather(vllm_config: Any, hidden_size: int, dtype: torch.dtype) -> bool:
    """Collectively allocate the SP all-gather buffer this instance needs
    (``sp_all_gather_rows``; none needed: False). Capability check only:
    without NVLink symmetric memory over the TP group the stock all-gather
    stays (logged)."""
    max_rows = sp_all_gather_rows(vllm_config, hidden_size, dtype)
    if max_rows == 0:
        logger.info_once(
            "Kimi K3 SP all-gather: no symmetric-memory buffer on this instance "
            "(P/D decode, or every step fits the custom all-gather); stock "
            "all-gather."
        )
        return False
    return init_sp_all_gather_buffer(max_rows, hidden_size, dtype)


def init_sp_all_gather_buffer(
    max_rows: int, hidden_size: int, dtype: torch.dtype
) -> bool:
    """Collectively allocate the two-slot buffer for ``max_rows`` rows per rank
    (False: no symmetric memory over the TP group, stock all-gather, logged)."""
    global _sp_all_gather
    if _sp_all_gather is not None:
        assert (
            _sp_all_gather.max_rows >= max_rows
            and _sp_all_gather.hidden_size == hidden_size
            and _sp_all_gather.dtype == dtype
        )
        return True
    tp = get_tp_group()
    if symm_mem is None or not direct_cp_enabled(tp, dtype, None):
        logger.warning_once(
            "Kimi K3 SP all-gather: the TP group has no symmetric memory; "
            "using the stock all-gather."
        )
        return False
    device = torch.device("cuda", torch.accelerator.current_device_index())
    _sp_all_gather = SymmSpAllGather(max_rows, hidden_size, dtype, device)
    logger.info_once(
        "Kimi K3 SP all-gather: symmetric-memory push, %d rows x %d per rank, %.0f MiB",
        max_rows,
        hidden_size,
        _sp_all_gather.storage.numel() * _sp_all_gather.storage.element_size() / 2**20,
    )
    return True


def begin_forward(num_gathers: int) -> None:
    """Announce a forward of ``num_gathers`` decoder-layer gathers (one per
    layer, so the same count every forward, which the cross-forward slot rule
    assumes); restarts the slot schedule (module docstring)."""
    if _sp_all_gather is not None:
        _sp_all_gather.begin_forward(num_gathers)


def kimi_sp_all_gather(x: torch.Tensor) -> torch.Tensor:
    """The decoder layer's SP all-gather. Without the buffer (P/D decode
    instance, small steps only, or no symmetric memory) the stock all-gather.
    With it: shards the stock custom (IPC) all-gather takes (small, e.g.
    graph-captured steps) keep it, larger ones take the symmetric-memory push;
    a shard above the buffer raises (``SymmSpAllGather.__call__``)."""
    from vllm.models.common.ops.sequence_parallel import (
        _custom_collective,
        sp_all_gather,
    )

    gather = _sp_all_gather
    if gather is None:
        return sp_all_gather(x)
    output = _custom_collective("custom_all_gather", x)
    if output is not None:
        return output
    return gather(x)


def mxfp8_gather_rows_ok(rows: int, hidden: int) -> bool:
    """A decoder-layer gather of ``rows`` x ``hidden`` BF16 rows per rank may
    go as MXFP8 (``kimi_sp_all_gather_mxfp8``): the symmetric-memory buffer
    takes the shard (the stock custom / Lamport all-gather would not: it stays
    BF16) and ``rows`` is a positive multiple of the 128-row scale tile."""
    gather = _sp_all_gather
    if (
        gather is None
        or gather.dtype != torch.bfloat16
        or hidden != gather.hidden_size
        or hidden % (4 * MXFP8_BLOCK)
        or rows <= 0
        or rows % MXFP8_ROW_TILE
        or rows > gather.max_rows
    ):
        return False
    return not _takes_custom_all_gather(rows, hidden, gather.dtype)


def _takes_custom_all_gather(rows: int, hidden: int, dtype: torch.dtype) -> bool:
    """``kimi_sp_all_gather`` sends this BF16 shard to the stock custom
    all-gather (the predicate of ``custom_all_gather``)."""
    communicator = get_tp_group().device_communicator
    ca_comm = getattr(communicator, "ca_comm", None)
    return ca_comm is not None and ca_comm.should_custom_all_gather(
        torch.empty((rows, hidden), dtype=dtype, device="meta")
    )


def kimi_sp_all_gather_mxfp8(qa: Any, num_tokens: int) -> Any:
    """The decoder layer's SP all-gather of an MXFP8 activation (module
    docstring) for a shard ``mxfp8_gather_rows_ok`` accepted: the gathered
    :class:`QuantizedActivation` of the first ``num_tokens`` rows (SP padding
    removed), bitwise ``mxfp8_quantize(all_gather(bf16))[:num_tokens]`` on
    every real row."""
    from dataclasses import replace

    gather = _sp_all_gather
    assert gather is not None
    rows, hidden = qa.data.shape
    data, sf = gather.gather_mxfp8(qa.data, qa.scale)
    sf_rows = cdiv(num_tokens, MXFP8_ROW_TILE) * MXFP8_ROW_TILE
    return replace(
        qa,
        data=data[:num_tokens],
        scale=sf[: sf_rows * hidden // MXFP8_BLOCK],
        orig_shape=torch.Size((num_tokens, hidden)),
    )
