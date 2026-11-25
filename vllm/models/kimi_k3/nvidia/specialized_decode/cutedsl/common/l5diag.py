# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DIAGNOSTIC ONLY (never merge): %globaltimer + %clock64 phase stamps of the KDA
decode chain (KdaHeadVerify, KdaVerifyCols, KdaVerifyLoop).

Each compiled kernel owns a ring buffer [RING, ctas, 32] int64 and a per-CTA launch
counter (both allocated at compile time; their addresses are compile-time constants).
CTA b of launch n writes row (n % RING, b): slots 0..15 %globaltimer, 16..31 the
matching %clock64. The counter is read at entry and bumped once per launch at exit, so
the last RING launches of every kernel are kept. ``dump(case)`` (forward bench, after
each case) writes every ring of this process with its counters to
$K/sol16/L5diag/<job>.<step>/<case>.rank<R>.pt and clears them."""

import os

import cutlass
import torch
from cutlass import Int32, Int64, Uint32

from .primitives import asm

ROOT = "/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/sol16/L5diag"
SLOTS = 32
RING = 256
_RINGS: list = []  # (kind, key, ctas, buffer, counter)


class Ring:
    """A kernel's stamp ring; ``base`` / ``count`` are Python ints (device addresses)."""

    def __init__(self, kind, key, ctas):
        device = torch.cuda.current_device()
        self.ctas = ctas
        self.buffer = torch.zeros(
            RING, ctas, SLOTS, dtype=torch.int64, device=f"cuda:{device}"
        )
        self.counter = torch.zeros(ctas, dtype=torch.int32, device=f"cuda:{device}")
        self.base, self.count = self.buffer.data_ptr(), self.counter.data_ptr()
        _RINGS.append((kind, key, ctas, self.buffer, self.counter))

    def row(self, block):
        """(trace time) this CTA's row address for the current launch."""
        seq = _volatile_load(Int64(self.count) + Int64(block) * 4)
        index = Int64(Int32(seq & Uint32(RING - 1)))
        return Int64(self.base) + (index * self.ctas + Int64(block)) * (SLOTS * 8)

    def bump(self, block, when):
        _bump(Int64(self.count) + Int64(block) * 4, Int32(when))


@asm(Uint32)
def _volatile_load(address):
    return "ld.volatile.global.u32 $0,[$1];", "=r,l,~{memory}", (Int64(address),)


@asm()
def _bump(address, when):
    code = "{ .reg .pred p; setp.ne.b32 p, $1, 0; @p red.relaxed.gpu.global.add.u32 [$0],1; }"
    return code, "l,r,~{memory}", (Int64(address), Int32(when))


@asm()
def _stamp(address, when):
    code = "{ .reg .u64 t, c; .reg .pred p; setp.ne.b32 p, $1, 0; "
    code += "mov.u64 t, %globaltimer; mov.u64 c, %clock64; "
    code += "@p st.global.u64 [$0], t; @p st.global.u64 [$0+128], c; }"
    return code, "l,r,~{memory}", (Int64(address), Int32(when))


@asm()
def _smid(address, when):
    code = "{ .reg .u32 s; .reg .u64 w; .reg .pred p; setp.ne.b32 p, $1, 0; "
    code += "mov.u32 s, %smid; cvt.u64.u32 w, s; @p st.global.u64 [$0], w; }"
    return code, "l,r,~{memory}", (Int64(address), Int32(when))


def stamp(row, slot, when):
    """(trace time) %globaltimer / %clock64 into ``slot`` of ``row`` where ``when``."""
    _stamp(row + slot * 8, when)


def smid(row, when):
    """(trace time) %smid into slot 15."""
    _smid(row + 15 * 8, when)


def dump(case):
    if not _RINGS:
        return
    run = os.environ.get("SLURM_JOB_ID", "j") + "." + os.environ.get("SLURM_STEP_ID", "s")
    out = f"{ROOT}/{run}"
    os.makedirs(out, exist_ok=True)
    rank = os.environ.get("SLURM_PROCID", "0")
    torch.cuda.synchronize()
    data = []
    for kind, key, ctas, buffer, counter in _RINGS:
        if bool(counter.any()):
            data.append((kind, key, ctas, buffer.cpu(), counter.cpu()))
    torch.save(data, f"{out}/{case}.rank{rank}.pt")
    for *_, buffer, counter in _RINGS:
        buffer.zero_()
        counter.zero_()
