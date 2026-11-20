# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""DIAGNOSTIC ONLY (D-MOE-STREAM diag branch, never lands): real-kernel ablation switches and a per-unit
%globaltimer timeline, all read at trace time.

DMS_TL_BUF=<device address of an int64 [256 CTAs][TL_ROLES][TL_UNITS][2] zeroed buffer>: lane 0 of the calling warp
writes (globaltimer ns, info) at buf[cta][role][unit % TL_UNITS]. The address is baked into the kernel, so a run that
sets it needs a private, fresh CuTe DSL cache.
DMS_NOMMA=1: the MMA warp skips the scale-factor S2T copies and the MMAs (stage release kept).
DMS_NODEP=1: FC2 TMA-B skips its fc1_done spin (routed and shared); 2 also skips the FC1 refined-SF spin.
DMS_PLAN=1: print the AB stage plan at trace time.
"""

import os

import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import Int32, Int64, T

TL_BUF = int(os.environ.get("DMS_TL_BUF", "0"))
NOMMA = os.environ.get("DMS_NOMMA") == "1"
NODEP = int(os.environ.get("DMS_NODEP", "0"))
PLAN = os.environ.get("DMS_PLAN") == "1"
TL_ROLES = 8
TL_UNITS = 64

# roles
TL_A_START = 0  # TMA-A took unit u (info = tile)
TL_MMA_START = 1  # leader MMA warp got the accumulator for unit u
TL_MMA_END = 2  # leader MMA warp issued the last k tile of unit u
TL_B_READY = 3  # TMA-B finished its dependency wait for unit u
TL_A_END = 4  # TMA-A issued the last weight load of unit u
TL_MISC = 5  # 0 scheduler entry, 1 sizes_ready seen, 2 scheduler done, 3 TMA-A done, 4 MMA done, 5 TMA-B done,
#              6 all roles joined (sync_threads), 7 tail.arrive passed, 8 last arriver through the NVLink barrier
TL_EPI_END = 6  # epilogue finished unit u (fc1_epi / fc2_epi returned)
TL_EPI_FETCH = 7  # epilogue got the next tile after unit u


def _globaltimer():
    return Int64(
        llvm.inline_asm(T.i64(), [], "mov.u64 $0, %globaltimer;", "=l", has_side_effects=True, asm_dialect=0)
    )


def tile_info(work_tile):
    """phase | feature tile << 4 | expert << 12 | valid tokens << 24 | token block << 32."""
    return (
        Int64(work_tile.phase)
        + Int64(work_tile.tile_n_idx) * Int64(16)
        + Int64(work_tile.expert_idx) * Int64(4096)
        + Int64(work_tile.valid_tokens_in_cta_tile) * Int64(16777216)
        + Int64(work_tile.tile_m_idx) * Int64(4294967296)
    )


def tl_stamp(role, unit, info):
    if TL_BUF == 0:
        return
    bx, by, bz = cute.arch.block_idx()
    gx, gy, _ = cute.arch.grid_dim()
    cta = bx + by * gx + bz * gx * gy
    tid = cute.arch.thread_idx()[0]
    lane = tid % Int32(32)
    u = Int32(unit) % Int32(TL_UNITS)
    slot = ((cta * Int32(TL_ROLES) + Int32(role)) * Int32(TL_UNITS) + u) * Int32(2)
    addr = Int64(TL_BUF) + Int64(slot) * Int64(8)
    t = _globaltimer()
    pred = Int32(lane == Int32(0))
    llvm.inline_asm(
        None,
        [addr.ir_value(), t.ir_value(), Int64(info).ir_value(), pred.ir_value()],
        "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $3, 0;\n\t@p st.global.v2.u64 [$0], {$1, $2};\n\t}",
        "l,l,l,r",
        has_side_effects=True,
        asm_dialect=0,
    )
