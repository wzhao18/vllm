# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lamport check of the MXFP8 activation stages of a ``check_x`` GEMM
(``dense_blockscaled_gemm_sm107``).

The SP all-gather (``attn_res.sp_fused`` publish-only rows) multicasts each MXFP8 row
into every rank's box and exits without waiting for the peers' rows. Words of the box
that are not published yet hold the sentinels (data 0x80000000, scales 0xFFFFFFFF);
a published word is never a sentinel (``common.mx_rows``). The GEMM's epilogue warps,
idle during the mainloop, check each landed activation stage before the MMA reads it:
a word of a row < R that is still a sentinel is re-read from global memory until it
is published and written into the stage; then the stage is released to the MMA warp
(``checked`` barrier). Words land single-copy atomically, so a word that is not a
sentinel is final: no fence or flag orders the all-gather and the GEMM.

Stage layout (asserted by the kernel): the CTA's activation tile of a stage is
``rows`` x 128 FP8 (one K block), 128-byte rows with the SW128 swizzle (16-byte chunk
c of row r at r 128 + (c ^ (r % 8)) 16), and its scales one F8_128x4 atom of 512
bytes (the 4 scales of row r at word (r % 32) 4 + r // 32), as in global memory.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint32

from ...common.mx_rows import SCALE_SENTINEL
from ...common.primitives import asm, published

# DIAG (bench): barrier forms under test
CTA_SCOPE = True
WAIT_OP = "test_wait"

# aux buffer words (``prewait.check_args``): int32 [2] rows, [3] K; int64 [2] data,
# [3] scales
ROWS_WORD, K_WORD, DATA_WORD64, SCALE_WORD64 = 2, 3, 2, 3


@asm()
def mbar_init(bar, count):
    return "mbarrier.init.shared::cta.b64 [$0], $1;", "r,r,~{memory}", (bar, Int32(count))


@asm()
def mbar_init_fence():
    return "fence.mbarrier_init.release.cluster;", "~{memory}", ()


@asm(Int32)
def _try_wait(bar, parity, *, cluster=False):
    sem = ".acquire.cluster" if cluster else ""
    sem = "" if CTA_SCOPE else sem
    code = (
        "{ .reg .pred p; mbarrier." + WAIT_OP + ".parity" + sem + ".shared::cta.b64 p, [$1], $2; "
        "selp.s32 $0, 1, 0, p; }"
    )
    return code, "=r,r,r,~{memory}", (bar, Int32(parity))


@cute.jit
def mbar_wait(bar, parity, cluster: cutlass.Constexpr[bool] = False):
    """Wait until the phase of ``bar`` with ``parity`` completed (acquire at cluster
    scope when ``cluster``: arrivals from the peer CTA)."""
    done = _try_wait(bar, parity, cluster=cluster)
    while done == 0:
        done = _try_wait(bar, parity, cluster=cluster)


@asm()
def mbar_arrive(bar):
    return "mbarrier.arrive.release.cta.shared::cta.b64 _, [$0];", "r,~{memory}", (bar,)


@asm()
def mbar_arrive_peer(bar, peer):
    """Arrive on the barrier at ``bar``'s offset in cluster CTA ``peer``."""
    code = (
        "{ .reg .b32 r; mapa.shared::cluster.u32 r, $0, $1; "
        "mbarrier.arrive" + ("" if CTA_SCOPE else ".release.cluster") + ".shared::cluster.b64 _, [r]; }"
    )
    return code, "r,r,~{memory}", (bar, Int32(peer))


@asm()
def fence_proxy_async_smem():
    return "fence.proxy.async.shared::cta;", "~{memory}", ()


@asm()
def _global4(address):
    code = "ld.volatile.global.v4.b32 {$0,$1,$2,$3}, [$4];"
    return code, "=r,=r,=r,=r,l,~{memory}", (Int64(address),), (Uint32,) * 4


@asm()
def _global1(address):
    return "ld.volatile.global.b32 $0, [$1];", "=r,l,~{memory}", (Int64(address),), Uint32


@asm()
def _shared4(address):
    code = "ld.shared.v4.b32 {$0,$1,$2,$3}, [$4];"
    return code, "=r,=r,=r,=r,r,~{memory}", (Int32(address),), (Uint32,) * 4


@asm()
def _shared1(address):
    return "ld.shared.b32 $0, [$1];", "=r,r,~{memory}", (Int32(address),), Uint32


@asm()
def _store_shared4(address, words):
    code = "st.shared.v4.b32 [$0], {$1,$2,$3,$4};"
    return code, "r,r,r,r,r,~{memory}", (Int32(address), *(Uint32(w) for w in words))


@asm()
def _store_shared1(address, word):
    return "st.shared.b32 [$0], $1;", "r,r,~{memory}", (Int32(address), Uint32(word))


@cute.jit
def check_stage(stage, sf_stage, t, row_ok, data, sf_ok, scale):
    """Thread ``t`` of 128 checks smem row ``t`` of the stage's activation tile at
    shared address ``stage`` (``row_ok``: the row is < R; its 128 global bytes at
    ``data``) and scale word ``t`` of the atom at ``sf_stage`` (``sf_ok``: its row is
    < R; global word at ``scale``). Returns 1 when it wrote the stage."""
    swz = t & 7
    row = stage + t * 128
    pending = Uint32(0)
    if row_ok:
        words = [_shared4(row + ((c ^ swz) << 4)) for c in range(8)]
        for c in cutlass.range_constexpr(8):
            if not published(words[c]):
                pending = pending | Uint32(1 << c)
    sf_word = sf_stage + t * 4
    if sf_ok:
        if _shared1(sf_word) == Uint32(SCALE_SENTINEL):
            pending = pending | Uint32(1 << 8)
    wrote = Int32(0)
    if pending != 0:
        wrote = Int32(1)
    while pending != 0:
        again = Uint32(0)
        if (pending & Uint32(0xFF)) != 0:
            values = [_global4(data + Int64(c * 16)) for c in range(8)]
            for c in cutlass.range_constexpr(8):
                if (pending >> c) & 1:
                    if published(values[c]):
                        _store_shared4(row + ((c ^ swz) << 4), values[c])
                    else:
                        again = again | Uint32(1 << c)
        if (pending >> 8) & 1:
            sfv = _global1(scale)
            if sfv != Uint32(SCALE_SENTINEL):
                _store_shared1(sf_word, sfv)
            else:
                again = again | Uint32(1 << 8)
        pending = again
    return wrote

