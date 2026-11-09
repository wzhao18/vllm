# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One torch symmetric-memory arena per EP process group, replacing FlashInfer's NVSHMEM
heap (shim/comm.py: ``sym_zeros``, ``compute_peer_offsets``).

The kernel reaches a peer's copy of ANY symmetric tensor as ``local address +
offsets[peer]`` with ONE offset table (FlashInfer computes it from one allocation and
relies on NVSHMEM's identical heap layout). Here every symmetric tensor is carved from a
single ``symm_mem.empty`` allocation by a deterministic bump allocator, at the same byte
offset on every rank, so ``offsets[p] = buffer_ptrs[p] - buffer_ptrs[rank]`` holds for
all of them. :meth:`SymmetricArena.verify_layout` all-gathers a hash of the carve layout
and raises if any rank differs.

The arena is zeroed, the device synchronized and the group joined before anyone's first
launch (the kernel's NVLink barrier signals and router sizes are
``zero_on_first_allocate`` state: a late zero would erase a peer's early
``red.add.sys``). It is rendezvoused without a multicast object (``peer_rendezvous``):
the kernel only uses the peer mappings. Nothing here allocates, rendezvouses or zeroes
during CUDA graph capture.
"""

from __future__ import annotations

import hashlib

import torch
import torch.distributed as dist

ALIGN = 256


def ensure_not_capturing(what: str) -> None:
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            f"{what} needs host-side compile/alloc and cannot run during CUDA graph "
            "capture. Warm up (RubinMegaMoE.warmup) on ALL EP ranks before capture."
        )


def aligned(nbytes: int, align: int = ALIGN) -> int:
    return -(-max(int(nbytes), 1) // align) * align


def group_barrier(group: dist.ProcessGroup | None) -> None:
    """Device-synchronized barrier over ``group`` (an all-reduce, NCCL-safe)."""
    torch.cuda.synchronize()
    if group is None or dist.get_world_size(group) == 1:
        return
    flag = torch.ones(1, dtype=torch.int32, device="cuda")
    dist.all_reduce(flag, group=group)
    torch.cuda.synchronize()


class SymmetricArena:
    def __init__(self, group: dist.ProcessGroup | None, nbytes: int):
        ensure_not_capturing("symmetric arena allocation")
        self.group = group
        self.world = 1 if group is None else dist.get_world_size(group)
        self.rank = 0 if group is None else dist.get_rank(group)
        self.nbytes = aligned(nbytes)
        device = torch.device("cuda", torch.cuda.current_device())
        self.handle = None
        if self.world == 1:
            self.buf = torch.empty(self.nbytes, dtype=torch.uint8, device=device)
            ptrs = [self.buf.data_ptr()]
        else:
            import torch.distributed._symmetric_memory as symm_mem

            from vllm.models.kimi_k3.nvidia.specialized_decode.symm import (
                peer_rendezvous,
            )

            self.buf = symm_mem.empty(self.nbytes, dtype=torch.uint8, device=device)
            self.handle = peer_rendezvous(self.buf, group.group_name)
            ptrs = [int(p) for p in self.handle.buffer_ptrs]
            if len(ptrs) != self.world:
                raise RuntimeError(
                    f"symmetric rendezvous returned {len(ptrs)} peer pointers for a "
                    f"group of {self.world}."
                )
        local = int(ptrs[self.rank])
        if local != self.buf.data_ptr():
            raise RuntimeError(
                "symmetric handle's local pointer != the arena's address."
            )
        if local % ALIGN:
            raise RuntimeError(f"symmetric arena base is not {ALIGN}-byte aligned.")
        # Signed per-peer displacement; the kernel adds it as Int64.
        self.offsets: tuple[int, ...] = tuple(int(p) - local for p in ptrs)
        self.base = local
        self.layout: list[tuple[str, int, int]] = []
        self._cursor = 0
        self.zero_all()

    # -- carving ----------------------------------------------------------------------

    def alloc(
        self, shape: tuple[int, ...], dtype: torch.dtype, *, tag: str
    ) -> torch.Tensor:
        """A view of the next free, ALIGN-aligned region (never freed individually)."""
        ensure_not_capturing("symmetric arena carve")
        elem = dtype.itemsize
        nbytes = elem
        for d in shape:
            nbytes *= int(d)
        size = aligned(nbytes)
        if self._cursor + size > self.nbytes:
            raise RuntimeError(
                f"symmetric arena exhausted carving {tag!r} ({size} B at "
                f"{self._cursor} of {self.nbytes} B); size the arena from the "
                "sessions' plans."
            )
        offset = self._cursor
        self._cursor += size
        self.layout.append((tag, offset, size))
        region = self.buf[offset : offset + max(nbytes, 1)]
        return region[:nbytes].view(dtype).view(shape)

    @property
    def used_bytes(self) -> int:
        return self._cursor

    def verify_layout(self) -> None:
        """Raise unless every rank carved the same (tag, offset, size) sequence and the
        arena has the same size everywhere (the single-offset-table invariant)."""
        if self.world == 1:
            return
        text = repr((self.nbytes, self.layout)).encode()
        digest = torch.tensor(
            list(hashlib.sha256(text).digest()), dtype=torch.uint8, device="cuda"
        )
        gathered = torch.empty(
            (self.world, digest.numel()), dtype=torch.uint8, device="cuda"
        )
        dist.all_gather_into_tensor(gathered, digest, group=self.group)
        if not bool((gathered == digest).all()):
            bad = [r for r in range(self.world) if not torch.equal(gathered[r], digest)]
            raise RuntimeError(
                f"symmetric arena layout differs on ranks {bad} (local layout "
                f"{self.layout}); every EP rank must build the same buckets in the "
                "same order."
            )

    # -- zeroing ----------------------------------------------------------------------

    def zero_all(self) -> None:
        """Zero the whole arena on every rank, visible before any peer's next write.
        Collective; only at (re)initialization, never while sessions are in flight."""
        ensure_not_capturing("symmetric arena zeroing")
        self.buf.zero_()
        group_barrier(self.group)

    def zero_(self, tensor: torch.Tensor) -> None:
        """Re-zero one carved region with the same protocol (collective)."""
        ensure_not_capturing("symmetric arena zeroing")
        # Byte-level: FP8/FP4 scale dtypes (e.g. E8M0) have no zero value to fill.
        tensor.view(torch.uint8).zero_()
        group_barrier(self.group)

    def destroy(self) -> None:
        """Drain and drop the references; torch releases the symmetric allocation."""
        group_barrier(self.group)
        self.layout.clear()
        self.handle = None
        self.buf = None
