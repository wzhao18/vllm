# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DIAGNOSTIC ONLY (never merge): %globaltimer phase stamps of the MoE tail
(WaveMoeTail) and the post-tail AttnRes (ClusterAttnRes ring codes).

Kernels store int64 %globaltimer values into per-CTA rows of 16 slots of a
caller-owned buffer (one buffer per layer / site and batch). ``dump(case)`` (called by
the forward bench after each case's replays) writes every buffer of this process to
$K/sol16/S1diag/<job>.<step>/<case>.rank<R>.pt."""

import os

import torch
from cutlass import Int32, Int64

from .primitives import asm

ROOT = "/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/sol16/S1diag"
SLOTS = 48  # 0-15 CTA phases (tid 0), 16-23 / 24-31 / 32-39 per warp (lane 0): polled, cluster wait, region end
AR_CTAS = 7 * 128  # AttnRes stamps: rows <= 128 (B16)

_BUFS: list = []  # (kind, key, rows, tensor, rows dumped)


@asm()
def stamp_store(address, when):
    code = "{ .reg .u64 t; .reg .pred p; setp.ne.b32 p, $1, 0; "
    code += "mov.u64 t, %globaltimer; @p st.global.u64 [$0], t; }"
    return code, "l,r,~{memory}", (Int64(address), Int32(when))


@asm()
def smid_store(address, when):
    code = "{ .reg .u32 s; .reg .u64 w; .reg .pred p; setp.ne.b32 p, $1, 0; "
    code += "mov.u32 s, %smid; cvt.u64.u32 w, s; @p st.global.u64 [$0], w; }"
    return code, "l,r,~{memory}", (Int64(address), Int32(when))


@asm()
def clock_store(address, when):
    code = "{ .reg .u64 t; .reg .pred p; setp.ne.b32 p, $1, 0; "
    code += "mov.u64 t, %clock64; @p st.global.u64 [$0], t; }"
    return code, "l,r,~{memory}", (Int64(address), Int32(when))


def buffer(kind, key, rows, ctas, device, slots=SLOTS, used=None):
    t = torch.zeros(ctas, slots, dtype=torch.int64, device=device)
    _BUFS.append((kind, key, rows, t, used or ctas))
    return t


def dump(case):
    if not _BUFS:
        return
    run = os.environ.get("SLURM_JOB_ID", "j") + "." + os.environ.get("SLURM_STEP_ID", "s")
    out = f"{ROOT}/{run}"
    os.makedirs(out, exist_ok=True)
    rank = os.environ.get("SLURM_PROCID", "0")
    torch.cuda.synchronize()
    data = []
    for kind, key, rows, t, used in _BUFS:
        if bool(t.any()):
            data.append((kind, key, rows, t[:used].cpu()))
    torch.save(data, f"{out}/{case}.rank{rank}.pt")
    for _, _, _, t, _ in _BUFS:
        t.zero_()
