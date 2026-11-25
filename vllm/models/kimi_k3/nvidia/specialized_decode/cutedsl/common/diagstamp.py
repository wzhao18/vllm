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


# ---------------------------------------------------------------------------------
# DIAG (STAMP arm, never merge): raw per-process buffers, %gridid-keyed rings and the
# untraced step anchor (every specialized-decode kernel's PDL-wait return).
# ---------------------------------------------------------------------------------
import json  # noqa: E402
import sys  # noqa: E402
import zlib  # noqa: E402

from cutlass import Uint64  # noqa: E402

_RAW: dict = {}
_SITES: dict = {}
ANCHOR_LOG2 = 16  # 65536 launches x 16 B (gridid-keyed)


@asm(Uint64)
def gridid():
    """%gridid: the per-launch temporal grid id (uniform over the grid)."""
    return "mov.u64 $0, %gridid;", "=l", ()


def raw_buffer(tag, rows, nbytes, reset=True):
    """A zeroed device buffer of ``nbytes`` for (tag, rows) as a Python int (trace
    time; baked into the kernel). Dumped with the rings; ``reset``: zeroed after
    each dump (accumulators), else kept (gridid-keyed rings)."""
    key = (tag, int(rows))
    if key not in _RAW:
        dev = torch.cuda.current_device()
        _RAW[key] = (torch.zeros(nbytes // 8, dtype=torch.int64, device=dev), reset)
    return _RAW[key][0].data_ptr()


def _anchor_site(frame):
    name = frame.f_code.co_filename
    rel = name.split("specialized_decode/", 1)[-1]
    text = f"{rel}:{frame.f_lineno}:{frame.f_code.co_name}"
    site = zlib.crc32(text.encode()) & 0x7FFFFF
    _SITES[site] = text
    return site


def _anchor_code(site):
    base = raw_buffer("anchor", 0, 16 << ANCHOR_LOG2, reset=False)
    mask = (1 << ANCHOR_LOG2) - 1
    return (
        "{ .reg .pred p; .reg .u32 a0, a1, a2, a3, a4, a5, s; .reg .u64 t, g, ad, k; "
        "mov.u32 a0, %tid.x; mov.u32 a1, %tid.y; mov.u32 a2, %tid.z; "
        "mov.u32 a3, %ctaid.x; mov.u32 a4, %ctaid.y; mov.u32 a5, %ctaid.z; "
        "or.b32 s, a0, a1; or.b32 s, s, a2; or.b32 s, s, a3; or.b32 s, s, a4; "
        "or.b32 s, s, a5; setp.eq.u32 p, s, 0; "
        "mov.u64 t, %globaltimer; mov.u64 g, %gridid; "
        f"and.b64 ad, g, {mask}; shl.b64 ad, ad, 4; add.u64 ad, ad, {base}; "
        f"shl.b64 k, g, 24; shr.b64 k, k, 24; or.b64 k, k, {site << 40}; "
        "@p st.global.v2.u64 [ad], {t, k}; }"
    )


@asm()
def _anchor(site: int):
    """CTA (0,0,0) thread 0: {%globaltimer, site << 40 | gridid[0:40]} into the
    anchor ring slot gridid % 2^16 (one plain store, no return, no wait)."""
    return _anchor_code(site), "~{memory}", ()


_ORIG_WAIT = cute.arch.griddepcontrol_wait


def _wait_anchored(*args, **kwargs):
    """``cute.arch.griddepcontrol_wait`` + the anchor stamp, for kernels traced from
    the specialized-decode sources (other DSL users keep the original: their cubins
    may live in caches shared across processes, and the anchor address is baked)."""
    result = _ORIG_WAIT(*args, **kwargs)
    frame = sys._getframe(1)
    if "specialized_decode" in frame.f_code.co_filename:
        _anchor(_anchor_site(frame))
    return result


if os.environ.get("STAMP_ANCHOR", "1") == "1":
    cute.arch.griddepcontrol_wait = _wait_anchored


def dump_raw(out_path):
    """Raw buffers (accumulators zeroed after the copy) + the anchor site table."""
    torch.cuda.synchronize()
    data = {f"{k[0]}:{k[1]}": b.cpu() for k, (b, _) in _RAW.items()}
    torch.save(data, out_path)
    for b, reset in _RAW.values():
        if reset:
            b.zero_()
    with open(out_path + ".sites.json", "w") as f:
        json.dump({str(k): v for k, v in _SITES.items()}, f)
