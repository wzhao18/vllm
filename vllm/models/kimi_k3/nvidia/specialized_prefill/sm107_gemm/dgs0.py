# SPDX-License-Identifier: Apache-2.0
"""DG-SOL probe S0 (EXPERIMENT ONLY, never merge): per-CTA stamps in the KDA
prefill q|k|v conv GEMM (``kda_conv_gemm``) inside the real step.

The probe overlay is its own image: the stamps are compiled in (no switch).
A layer stamps when its record pointer is non-zero (``kda_conv_in_proj``
passes it for the selected KDA layers only); the pointer travels in the
kernel's ``sched_die_map`` argument, which the conv tactic (no die_sched)
never reads. The kernel writes u32 words only (no atomics, no reads).

Record per CTA (``REC`` u32 words, CTA linear index = block x + y gx + z gx gy):
  0 entry %globaltimer lo (MMA warp)    1 entry %clock64 lo (MMA warp)
  2 %smid                               3 is_leader_cta (MMA issuer)
  4 MMA tiles                           5 k tiles per tile
  6 epilogue tiles                      7 exit %globaltimer lo (epilogue warp 0)
  8 exit %clock64 lo                    9 epilogue loop end %clock64 lo
 10 kernel entry %clock64 lo (epilogue warp 0, before TMEM alloc)
 16 + 4 t (t < TILES, later tiles clamp to the last slot), MMA warp lane 0:
    [acc acquired, AB stage 0 full, AB stage K-1 full, acc committed] %clock64 lo
 80 + 3 t, epilogue thread 0: [acc full, TMEM released, column pass end]
"""

from __future__ import annotations

import os

import cutlass
from cutlass import Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

REC = 128
TILES = 16
W_ENTRY_GT, W_ENTRY_CLK, W_SMID, W_LEADER = 0, 1, 2, 3
W_MMA_TILES, W_KTILES, W_EPI_TILES = 4, 5, 6
W_EXIT_GT, W_EXIT_CLK, W_EPI_END_CLK, W_EPI_ENTRY_CLK = 7, 8, 9, 10
W_MMA = 16
W_EPI = 80
MAXCTA = 256
#: KDA instances (0-based, model construction order) that stamp.
LAYERS = (2, 60)

_ASM = dict(is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def clk(*, loc=None, ip=None) -> Int64:
    return Int64(llvm.inline_asm(T.i64(), [], "mov.u64 $0, %clock64;", "=l",
                                 has_side_effects=True, loc=loc, ip=ip, **_ASM))


@dsl_user_op
def gtime(*, loc=None, ip=None) -> Int64:
    return Int64(llvm.inline_asm(T.i64(), [], "mov.u64 $0, %globaltimer;", "=l",
                                 has_side_effects=True, loc=loc, ip=ip, **_ASM))


@dsl_user_op
def smid(*, loc=None, ip=None) -> Int32:
    return Int32(llvm.inline_asm(T.i32(), [], "mov.u32 $0, %smid;", "=r",
                                 has_side_effects=True, loc=loc, ip=ip, **_ASM))


@dsl_user_op
def st64lo(addr: Int64, val: Int64, pred: Int32, *, loc=None, ip=None) -> None:
    """``*(u32*)addr = lo32(val)`` when ``pred`` != 0."""
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Int64(val).ir_value(loc=loc, ip=ip),
         Int32(pred).ir_value(loc=loc, ip=ip)],
        "{ .reg .pred p; .reg .b32 lo; setp.ne.s32 p, $2, 0; cvt.u32.u64 lo, $1; "
        "@p st.global.u32 [$0], lo; }",
        "l,l,r",
        has_side_effects=True, loc=loc, ip=ip, **_ASM,
    )


@dsl_user_op
def st32(addr: Int64, val: Int32, pred: Int32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Int32(val).ir_value(loc=loc, ip=ip),
         Int32(pred).ir_value(loc=loc, ip=ip)],
        "{ .reg .pred p; setp.ne.s32 p, $2, 0; @p st.global.u32 [$0], $1; }",
        "l,r,r",
        has_side_effects=True, loc=loc, ip=ip, **_ASM,
    )


def rec_addr(base: Int64, bidx, bidy, bidz, gx, gy) -> Int64:
    lin = Int64(bidx) + Int64(bidy) * Int64(gx) + Int64(bidz) * Int64(gx) * Int64(gy)
    return base + lin * Int64(REC * 4)


def tile_slot(t) -> Int32:
    """min(t, TILES - 1)."""
    t = Int32(t)
    return t - (t - (TILES - 1)) * Int32(t > (TILES - 1))


# ---- host side ---------------------------------------------------------------------------------------------
_BUFS: dict = {}  # KDA ordinal -> torch int32 [MAXCTA, REC]


def buffer(ordinal: int, device):
    """The stamp record buffer of a stamped KDA instance (allocated once at
    prepare, before the memory profile), else None."""
    if ordinal not in LAYERS:
        return None
    import torch

    b = _BUFS.get(ordinal)
    if b is None:
        b = torch.zeros((MAXCTA, REC), dtype=torch.int32, device=device)
        _BUFS[ordinal] = b
    return b


def dump(tag: str) -> None:
    """Bench only: after a case, write every stamped layer's records
    (DGS0_DIR, one npz per rank)."""
    out_dir = os.environ.get("DGS0_DIR")
    if not out_dir or not _BUFS:
        return
    import numpy as np
    import torch

    os.makedirs(out_dir, exist_ok=True)
    torch.cuda.synchronize()
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    arrays = {f"L{k}": v.cpu().numpy() for k, v in _BUFS.items()}
    np.savez_compressed(os.path.join(out_dir, f"{tag}.rank{rank}.npz"), **arrays)
    for v in _BUFS.values():
        v.zero_()
    torch.cuda.synchronize()
