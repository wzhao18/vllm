# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resident grids: ``per_sm`` CTAs of one kernel on every SM at once, each knowing its
die, its rank among its die's SMs and its slot on its SM, so work can be assigned
statically by (die, rank, slot) with no atomics on a shared address and no grid
barrier.

Protocol (device helpers ``land`` / ``wait_landed``):

- ``land`` (before the PDL wait): the SM's table entry gives (die, rank); a ticket
  from the SM's own counter gives the slot (ticket mod ``per_sm``) and the launch
  index (ticket div ``per_sm``); a fire-and-forget add counts the CTA as landed.
- ``wait_landed`` (before the CTA exits): every CTA of this launch has landed. With
  the grid exactly ``per_sm`` x SMs and the occupancy exactly ``per_sm`` (checked at
  load, raising otherwise), no CTA can then land on an SM after one of its CTAs left:
  every SM hosts exactly ``per_sm`` CTAs of each launch, so the tickets of a launch
  on one SM are consecutive and the slots distinct, whatever the placement order
  (a CTA delayed by another kernel's CTA lands on that SM once it frees).

The counters only grow (u32 tickets: 2^32 / per_sm launches; u64 landings), so no
launch resets anything and back-to-back launches need no ordering between them.
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, Uint32, Uint64

from vllm.models.kimi_k3.nvidia.k31_spin_guard import (
    register_poll_helper,
    spin_start,
    spin_tick,
)

from . import dies
from .host import _checked, resolve_kernel
from .primitives import asm

MAX_SMID = 1024


def per_sm_fit(compiled, index: int, threads: int, smem: int) -> int:
    """CTAs of ``compiled`` (``threads`` each, ``smem`` dynamic shared memory bytes)
    one SM holds, from the function's registers and the SM's limits (65536 registers
    allocated per warp in 256s, the shared memory plus 1 KiB per CTA, the threads),
    with the maximum shared memory carveout requested (cuOccupancy* reports 1 for
    CuTe functions that demonstrably run two per SM)."""
    import cuda.bindings.driver as cuda

    _, function = resolve_kernel(compiled, index)
    attribute = cuda.CUfunction_attribute
    carveout = attribute.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT
    _checked(cuda.cuFuncSetAttribute(function, carveout, 100))
    (regs,) = _checked(
        cuda.cuFuncGetAttribute(attribute.CU_FUNC_ATTRIBUTE_NUM_REGS, function)
    )
    (static,) = _checked(
        cuda.cuFuncGetAttribute(attribute.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, function)
    )
    props = torch.cuda.get_device_properties(index)
    warps = -(-threads // 32)
    per_warp = -(-regs * 32 // 256) * 256
    return min(
        props.regs_per_multiprocessor // (warps * per_warp),
        props.shared_memory_per_multiprocessor // (smem + static + 1024),
        props.max_threads_per_multi_processor // (warps * 32),
    )


class ResidentGrid:
    """Host side: the SM table and counters of resident ``per_sm``-per-SM grids on
    ``device`` (shared by every kernel using it on the device's stream order)."""

    def __init__(self, device: torch.device, per_sm: int):
        index = torch.device(device).index
        self.per_sm = per_sm
        self.sms = torch.cuda.get_device_properties(index).multi_processor_count
        domains = dies.sm_domains(index) if dies.domain_count(index) > 1 else None
        if domains is None:
            domains = (0,) * self.sms
        entries = [-1] * MAX_SMID
        counts = [0, 0]
        for sm, die in enumerate(domains):
            entries[sm] = (die << 16) | counts[die]
            counts[die] += 1
        self.die_sms = tuple(counts)
        self.table = torch.tensor(entries, dtype=torch.int32, device=device)
        self.tickets = torch.zeros(MAX_SMID, dtype=torch.int32, device=device)
        self.landed = torch.zeros(1, dtype=torch.int64, device=device)

    @property
    def grid(self) -> int:
        return self.per_sm * self.sms

    def check(self, compiled, threads: int, smem: int, what: str):
        """Raise unless exactly ``per_sm`` CTAs of ``compiled`` (``threads`` each,
        ``smem`` dynamic shared memory bytes) fit an SM: the static assignment's
        completeness rests on it (module notes)."""
        fits = per_sm_fit(compiled, self.table.device.index, threads, smem)
        if fits != self.per_sm:
            raise RuntimeError(
                f"{what}: {fits} CTAs per SM fit ({threads} threads, {smem} B shared "
                f"memory); the resident grid needs exactly {self.per_sm}"
            )

    def addresses(self):
        return (self.table.data_ptr(), self.tickets.data_ptr(), self.landed.data_ptr())


@asm(Uint32)
def _smid():
    return "mov.u32 $0, %smid;", "=r", ()


@asm(Int32)
def _table_entry(address):
    return "ld.global.cg.b32 $0, [$1];", "=r,l,~{memory}", (Int64(address),)


@asm(Uint32)
def _ticket(address):
    code = "atom.add.relaxed.gpu.global.u32 $0, [$1], 1;"
    return code, "=r,l,~{memory}", (Int64(address),)


@asm()
def _count_landed(address):
    return "red.relaxed.gpu.global.add.u64 [$0], 1;", "l,~{memory}", (Int64(address),)


@asm(Uint64)
def _landed(address):
    return "ld.acquire.gpu.global.u64 $0, [$1];", "=l,l,~{memory}", (Int64(address),)


@cute.jit
def land(table, tickets, landed, per_sm: cutlass.Constexpr[int]):
    """(die, rank of the SM in its die, slot on the SM, launch index) of this CTA;
    one thread calls it and shares the result."""
    sm = _smid()
    entry = _table_entry(Int64(table) + Int64(sm) * 4)
    ticket = _ticket(Int64(tickets) + Int64(sm) * 4)
    _count_landed(landed)
    return (entry >> 16, entry & 0xFFFF, Int32(ticket % Uint32(per_sm)),
            Int32(ticket // Uint32(per_sm)))  # fmt: skip


@cute.jit
def wait_landed(landed, launch, grid: cutlass.Constexpr[int]):
    """Spin until every CTA of launch ``launch`` has landed (one thread; normally
    long true by the time a CTA is done)."""
    target = (Uint64(launch) + 1) * grid
    seen = _landed(landed)
    sgn, sgt = spin_start()
    while seen < target:
        seen = _landed(landed)
        sgn, sgt = spin_tick(sgn, sgt, observed=seen, target=0)


# spin timeouts name the caller of these shared polls
register_poll_helper(__file__, "wait_landed")
