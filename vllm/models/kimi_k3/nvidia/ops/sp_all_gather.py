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

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        rows, hidden = x.shape
        if hidden != self.hidden_size or x.dtype != self.dtype:
            raise ValueError(
                f"SP all-gather buffer holds {self.hidden_size}-wide {self.dtype} "
                f"rows, got {tuple(x.shape)} {x.dtype}"
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
        x = x.contiguous()
        numel = rows * hidden
        _push_shard_kernel[(min(self.num_sms, cdiv(numel, _BLOCK)),)](
            x,
            self.peer_ptrs[slot],
            self.rank * numel,
            numel,
            self.rank,
            WORLD=self.world,
            BLOCK=_BLOCK,
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
