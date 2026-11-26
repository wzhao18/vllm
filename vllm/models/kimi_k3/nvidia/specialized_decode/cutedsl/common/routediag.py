# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DIAGNOSTIC ONLY (never merge): per-CTA %globaltimer + %smid phase stamps of the MoE
front chain (lane MOEFT / ROUTE: WsGemm front, RouteSitu, SplitKSitu, SplitKGemm,
RubinExpertsKernel entry), untraced, same plain-store style as the STAMP arm.

Per (tag, rows) a device ring ``[RING, CTAS, SLOTS]`` int64 plus a per-CTA call counter.
``begin`` (one thread per CTA): call = atomicAdd(counter[cta], 1); slot 0 = entry time,
slot 6 = %smid; ``stamp(addr, s)``: slot s = time. Every CTA of a grid runs once per call,
so ``call`` numbers launches identically on every CTA. Several tags per kernel (one per
stamping warp) keep the warps independent (no shared address). ``dump(path)`` saves all
rings (host side, after a synchronize). Buffers are allocated at trace time and their
addresses baked into the code, so the process compiles into a private cute cache."""

import os

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int64, Uint32

from .primitives import asm

RING, CTAS, SLOTS = 512, 256, 8
os.environ["VLLM_KIMI_K3_CUTE_CACHE_DIR"] = f"/tmp/routediag-cutecache-{os.getpid()}"
_BUFS: dict = {}


@asm(Int64)
def gtimer():
    return "mov.u64 $0, %globaltimer;", "=l,~{memory}", ()


@asm(Uint32)
def smid():
    return "mov.u32 $0, %smid;", "=r", ()


@asm(Uint32)
def atom_inc(addr):
    return "atom.global.add.u32 $0, [$1], 1;", "=r,l,~{memory}", (addr,)


@asm()
def st64(addr, value):
    return "st.global.u64 [$0], $1;", "l,l,~{memory}", (addr, value)


def buffer(tag, rows=-1):
    """(ring address, counter address) of (tag, rows) as Python ints (trace time)."""
    try:
        rows = int(rows)
    except Exception:
        rows = -1
    key = (tag, rows)
    if key not in _BUFS:
        dev = torch.cuda.current_device()
        ring = torch.zeros(RING, CTAS, SLOTS, dtype=torch.int64, device=dev)
        ctr = torch.zeros(CTAS, dtype=torch.int32, device=dev)
        _BUFS[key] = (ring, ctr)
    ring, ctr = _BUFS[key]
    return ring.data_ptr(), ctr.data_ptr()


@cute.jit
def begin(base: cutlass.Constexpr, ctr: cutlass.Constexpr, cta, leader):
    """Slot 0 (time) + slot 6 (%smid) by ``leader`` of CTAs < CTAS; returns the row
    address (0 elsewhere)."""
    addr = Int64(0)
    if leader & (cta < CTAS):
        call = atom_inc(Int64(ctr) + Int64(cta) * 4) % RING
        addr = Int64(base) + (Int64(call) * CTAS + Int64(cta)) * (SLOTS * 8)
        st64(addr, gtimer())
        st64(addr + 48, Int64(smid()))
    return addr


@cute.jit
def stamp(addr, slot: cutlass.Constexpr):
    if addr != Int64(0):
        st64(addr + slot * 8, gtimer())


def dump(path):
    torch.cuda.synchronize()
    data = {f"{k[0]}:{k[1]}": (s.cpu(), c.cpu()) for k, (s, c) in _BUFS.items()}
    torch.save(data, path)


def reset():
    for s, c in _BUFS.values():
        s.zero_()
        c.zero_()
