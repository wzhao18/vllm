# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DIAGNOSTIC ONLY (never merge): %globaltimer phase stamps of the o-projection /
AttnRes kernels in the 16-rank step (lane OA, loop B-OA-2).

Per (tag, rows) a device ring ``[RING, CTAS, SLOTS]`` int64 plus a per-CTA call
counter. A stamping thread takes ``call = atomicAdd(counter[cta], 1)`` at kernel
entry and writes %globaltimer into ``ring[call % RING, cta, slot]``; every CTA of a
grid runs once per call, so ``call`` numbers the launches identically on every CTA
and every rank (all ranks replay the same graph). Dumped after each bench case by ``dump``
(``$OA_DIAG_DIR/<job>.<step>/<case>.rank<SLURM_PROCID>.pt``). Buffers are allocated at trace
time on the current device; pointers are baked into the compiled kernels."""

import os

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int64, Uint32

from .primitives import asm

RING, CTAS, SLOTS = 256, 128, 8
# Stamp buffer addresses are baked into the compiled code, so a shared (Lustre) cute
# cache entry compiled by one rank is an illegal address on every other rank: this
# process compiles into its own node-local cache (read per compile by cute_cache).
os.environ["VLLM_KIMI_K3_CUTE_CACHE_DIR"] = f"/tmp/oadiag-cutecache-{os.getpid()}"
_BUFS: dict = {}


@asm(Int64)
def gtimer():
    return "mov.u64 $0, %globaltimer;", "=l,~{memory}", ()


@asm(Uint32)
def atom_inc(addr):
    return "atom.global.add.u32 $0, [$1], 1;", "=r,l,~{memory}", (addr,)


@asm()
def st64(addr, value):
    return "st.global.u64 [$0], $1;", "l,l,~{memory}", (addr, value)


def dump(case):
    """Save every (tag, rows) ring + counter to
    ``$OA_DIAG_DIR/<job>.<step>/<case>.rank<R>.pt`` (called by the forward bench
    after each case; nq_step exits with os._exit, so no atexit hook)."""
    out = os.environ.get(
        "OA_DIAG_DIR", "/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/sol16/OA/diag"
    )
    run = os.environ.get("SLURM_JOB_ID", "j") + "." + os.environ.get("SLURM_STEP_ID", "s")
    os.makedirs(f"{out}/{run}", exist_ok=True)
    rank = os.environ.get("SLURM_PROCID", os.environ.get("RANK", "0"))
    torch.cuda.synchronize()
    data = {f"{k[0]}:{k[1]}": (s.cpu(), c.cpu()) for k, (s, c) in _BUFS.items()}
    torch.save(data, f"{out}/{run}/{case}.rank{rank}.pt")
    if _RAW:  # STAMP arm: raw rings / accumulators + anchor sites
        dump_raw(f"{out}/{run}/{case}.rank{rank}.raw.pt")


def buffer(tag, rows):
    """(ring address, counter address) of (tag, rows) as Python ints (trace time)."""
    try:  # static rows (a per-shape kernel); a dynamic-rows kernel serves every
        rows = int(rows)  # batch with one trace: key -1, the per-case dump separates them
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
    """Entry stamp (slot 0) by the ``leader`` thread of CTAs < CTAS; returns this
    thread's stamp row address (0 elsewhere)."""
    addr = Int64(0)
    if leader & (cta < CTAS):
        call = atom_inc(Int64(ctr) + Int64(cta) * 4) % RING
        addr = Int64(base) + (Int64(call) * CTAS + Int64(cta)) * (SLOTS * 8)
        st64(addr, gtimer())
    return addr


@cute.jit
def stamp(addr, slot: cutlass.Constexpr):
    if addr != Int64(0):
        st64(addr + slot * 8, gtimer())

