# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 DSA prefill sparse MLA on Rubin (sm_107): latent Q resident in TMEM.

One CTA per SM and per token (every top-k KV row is delivered into one SM once),
QK heads on M with all 512 latent Q dims as TMEM A operands (only the RoPE MMA is
SS), PV keys on M with P^T in the Rubin B collector. TMEM: Q 128 + S 64 + O^T 384 =
576. The single S buffer is released by the owning softmax warpgroup right after its
tcgen05.ld, so QK(u + 1) waits only for that load. Shared-memory operand reads per
64-key subtile: K 36 KB + RoPE Q 8 KB + V 32 KB + P 6 KB = 82 KB (sparse_mla_h6: 114
KB with Q dims 256-511 as SS A operands), the bound of the h6 form (P-MLA run89).

Pipeline (P-MLA runs 45-61):
* QK and PV are issued by two warps on different sub-partitions (one MMA warp was the
  serializer: ~10 dependent SASS per tcgen05.mma issue plus all waits and commits).
* Softmax ping-pong: warpgroup u % 2 owns 64-key subtile u. A thread (= head, TMEM
  lane) loads its whole 64-key row with one tcgen05.ld (S released at once), takes
  the row max, receives the lazy-max m of subtile u - 1 from the other warpgroup
  (mailbox + producer bar.arrive / consumer bar.sync), and writes E4M3 P. The two
  warps of a sub-partition therefore work on different subtiles. A 64-value row needs
  more than 80 registers: the CTA has 20 warps (ptxas only honours setmaxnreg region
  budgets above the launch count at <= 640 threads), softmax warpgroups run at 144.
* Each softmax warpgroup keeps l relative to the last m it used; both are combined
  at the token end. The lazy-max O^T rescale (rare) and, for the warpgroup that owns
  a token's last subtile, the epilogue (16x256b TMEM loads -> stmatrix.trans -> SW128
  staging -> TMA store) run in the softmax warpgroups: there is no separate output
  warpgroup. The QK warp copies each token's latent Q from the staging into TMEM
  (tcgen05.cp) between the tokens' QK MMAs.
* 8 gather warps fill the 64-key KV stages by per-lane cp.async (LDGSTS is not starved
  by the SS MMA operand reads, TMA gather4 is); the ring sits below the 227 KiB
  standard shared-memory limit (cp.async above it runs ~6x slower). Logical mode: the Q
  warp remaps the top-k through the block table from a shared-memory copy of each
  token's top-k row (block-table rows four stages ahead).
"""

import math
from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint8, Uint32, Uint64

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common import tensormap
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.primitives import asm
from vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla import (
    D_NOPE,
    H,
    LOG2E,
    ROW,
    SDESC_SW64,
    SDESC_SW128,
    cvt_e4m3x2,
    ex2,
    fence_async_smem,
    gather4,
    cp_async_arrive,
    ld_global_s32_pred,
    ld_global_v4_s32,
    ld_shared_f32,
    ld_shared_u32,
    mbar_arrive,
    mbar_arrive_tx,
    mbar_wait,
    mbar_init,
    named_bar,
    st_shared_f32,
    st_shared_u32,
    st_shared_v4,
    tc_fence_after,
    tc_fence_before,
    tma_tile2d,
    tmem_alloc_exclusive,
    tmem_free_exclusive,
    tmem_ld,
)
from vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla_h2 import (
    bulk_g2s,
    ld_shared_v4u,
    tmem_st_u32,
)
from vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla_h5 import (
    _tree_max,
    bf16x2,
    fadd2,
    ffma2,
    ld_shared_v2f,
    named_bar_arrive,
    st_global_u32,
    sstamp_clock,
    stmatrix_t4,
    tmem_ld16x256,
)
from vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla_sol import (
    bulk_wait_read0,
    tma_store2d,
)

P_PRE, THR = 4, 4.5  # P = 2^(s - m + P_PRE) <= 2^(P_PRE + THR) = 362 < 448 (E4M3 max)
NS = 6  # 64-key KV stages
# cpa 2: a gather warp signals stage g full once its stage g - CPA_LAG + 1 copies are done
# (one arrive per warp; per-thread cp.async.mbarrier arrives serialize at ~0.9 us per stage)
CPA_LAG = 4
# cpa >= 3: at most CP_CAP stages of cp.async in flight per SM (the producer waits for stage
# g - CP_CAP to land before issuing g): > ~150-250 KB of LDGSTS in flight collapses the
# random-row gather rate (pmla10 probe; h8 cpa 3 at NS 6 / 5 / 4: 3284 / 2270 / 1491 us)
CP_CAP = 4
NP = 3  # P buffers
NLD = 9  # gather warps
GATHER_PARTS = 1  # issuing lanes per quad (P-MLA run95)
STAGE64 = 4 * 64 * 128 + 64 * 64  # 36864: 4 latent SW128 atoms + RoPE SW64
# warps: 0-3 / 4-7 softmax warpgroups (thread = head; warps 3 and 7 hold padding rows
# 96-127 and only take part in barriers, rescales, Q copies and epilogues), 8 PV,
# 9 Q loader, 10 and 12-19 gather, 11 QK
W_SM, W_PV, W_Q, W_QK = 8, 8, 9, 11
THREADS = 32 * 20
REG_LAUNCH = 65536 // THREADS // 8 * 8  # 96
REG_SM, REG_LOW = 144, 56
assert 256 * REG_SM + 384 * REG_LOW <= THREADS * REG_LAUNCH

# shared memory (bytes). SS A tiles have M = 128 rows; only 96 are loaded and rows
# 96-127 read whatever follows (valid shared memory, padding rows of S only).
ATOM = H * 128  # one Q latent SW128 atom (96 rows x 128 B)
QS_BYTES = 128 * 64  # per token: RoPE [128 x 64 B] SW64 (A of the RoPE QK MMA)
P_BYTES = H * 64
# KV ring first, entirely below the 227 KiB standard shared-memory limit: cp.async
# (LDGSTS) into the oversized region above it runs ~6x slower (7 vs 42-60 B/clk/SM; TMA
# and st/ld.shared are unaffected: P-MLA-2CTA pmla10 g10c/g10d)
OFF_KV = 0
# Q latent atoms 0-3 staging (copied into TMEM per token; epilogue staging)
OFF_QN = OFF_KV + NS * STAGE64
OFF_QS = OFF_QN + 4 * ATOM  # 2 x QS_BYTES (token parity): A of the RoPE QK MMA
OFF_P = OFF_QS + 2 * QS_BYTES  # 3 x P [96 heads x 64 keys] E4M3, K-major SW64
OFF_AL = OFF_P + NP * P_BYTES  # [2 WG][2 (u // 2 parity)][96] alpha (lazy-max rescale)
OFF_FLAG = OFF_AL + 4 * H * 4  # [3] u32 (unused slots kept for alignment)
OFF_INV = OFF_FLAG + 16  # [96] out_scale / l of the token in its epilogue
OFF_TOK0 = OFF_INV + H * 4  # token parity 0: [2 WG][2 (l, m)][96]
NSL = 2 * NS  # logical mode: physical-row ring (the producer runs up to NSL stages ahead)
NCR = 32  # logical mode: token-count ring (> NSL + NS: the producer's max token lead)
NBAR = 112
OFF_BAR = OFF_TOK0 + 4 * H * 4
OFF_TMEM = OFF_BAR + NBAR * 8
OFF_GMASK = OFF_TMEM + 16  # [2 WG][2 (u // 2 parity)][4] u32: heads that grew (ballots)
# logical mode: [NCR] valid top-k count of token qi % NCR; [NSL][64] physical rows of stage g % NSL
OFF_NCNT = OFF_GMASK + 64
OFF_SLOT = OFF_NCNT + 4 * NCR
assert OFF_QN <= 232448 and OFF_QN % 1024 == 0
# logical mode: each token's top-k row in shared memory (2 buffers, one token ahead, B_TKI)
TKB = 2048 * 4
OFF_TKI = (OFF_SLOT + NSL * 256 + 127) // 128 * 128
SMEM_BYTES = (OFF_TKI + 2 * TKB + 1023) // 1024 * 1024
BTD = 4  # logical mode: block-table rows loaded BTD stages ahead of their stage
# RoPE A tile rows 96-127 of each QS buffer (read by the MMA, never loaded): the
# lazy-max mailbox ([2][96] f32, buffer 0) and token parity 1 of the l/m exchange
OFF_MBOX = OFF_QS + H * 64
OFF_TOK1 = OFF_QS + QS_BYTES + H * 64
assert 2 * H * 4 <= 32 * 64 and 4 * H * 4 <= 32 * 64
assert OFF_QS % 1024 == 0 and QS_BYTES % 1024 == 0 and ATOM % 1024 == 0
assert OFF_KV % 1024 == 0 and SMEM_BYTES <= 334848 - 1024, SMEM_BYTES

# barriers (count 1 = one arrive / commit per phase, 4 = one arrive per warp of a WG)
B_SFULL, B_SFREE = 0, 2  # x2 (subtile parity = owning softmax WG; one S buffer)
B_QTM, B_QLFREE, B_OFULL, B_EPIDONE = 4, 5, 6, 7
B_QFULL, B_QREMPTY = 8, 10  # x2 (token parity)
B_OFREE = 12  # x4 (O^T block)
B_OREADY, B_PEMPTY = 16, 19  # x3 (P buffer u % 3)
B_KVFULL, B_KVEMPTY = 22, 28  # x NS
B_NFULL = B_KVEMPTY + NS  # x NCR (logical mode: count of token qi % NCR written)
B_SLFULL = B_NFULL + NCR  # x NSL (logical mode: physical rows of stage g % NSL written)
B_SLEMPTY = B_SLFULL + NSL  # x NSL (... read by every gather warp)
B_SFULL1, B_SFREE1 = B_SLEMPTY + NSL, B_SLEMPTY + NSL + 2  # x2: S keys 32-63 (qkh knob)
B_TKI = B_SFREE1 + 2  # x2 (token parity): top-k row copy landed
B_QKI = B_TKI + 2  # x8 (u % 8): QK(u) issued (pvd knob: PV(u - pvd) waits for it)
B_TKCONV, B_TKFREE = B_QKI + 8, B_QKI + 10  # x2 (token parity, tconv knob): top-k row
# converted to physical rows (Q warp) / read by every copying gather warp
# x2 (staging half, epw knob): the owner warpgroup wrote / the store warp's TMA stores read
B_EPIFULL, B_EPIFREE = B_TKFREE + 2, B_TKFREE + 4
assert B_EPIFREE + 2 <= NBAR
# named barriers: 1-3 / 4-6 m handoff WG0 -> WG1 / WG1 -> WG0 (per sub-partition),
# 7-9 / 12-14 token-end l/m exchange (per sub-partition, token parity), 10 + wg
# warpgroup-wide, 15 S free (nbs knob: owning softmax WG -> QK warp)
NB_SFREE = 15
C_Q, C_S, C_O = 0, 128, 192  # Q latent dims 0-511; S; O^T

# instruction descriptors (dense kind::f8f6f4, E4M3 x E4M3 -> FP32, Rubin K64 bit 29)
IDESC_QK = (1 << 4) | (64 >> 3 << 17) | (128 >> 4 << 24) | (1 << 29)  # M128 N64
IDESC_PV = (1 << 4) | (H >> 3 << 17) | (128 >> 4 << 24) | (1 << 29) | (1 << 15)  # A MN


def _bar(base, i):
    return base + Uint32(OFF_BAR + 8 * i)


@asm()
def clock_read():
    return "{ .reg .b32 c; mov.u32 c, %clock; }", "~{memory}", ()


@asm()
def compiler_fence():
    return "", "~{memory}", ()


@asm()
def ld_global_v2_s32(ptr):
    return "ld.global.nc.v2.b32 {$0,$1}, [$2];", "=r,=r,l", (Int64(ptr),), (Int32,) * 2


@asm()
def st_shared_v2(addr, w0, w1):
    return ("st.shared.v2.b32 [$0], {$1,$2};", "r,r,r,~{memory}",
            (Uint32(addr), Int32(w0), Int32(w1)))


@asm(Int32, pure=False)
def redux_add_s32(v):
    return "redux.sync.add.s32 $0, $1, 0xffffffff;", "=r,r", (Int32(v),)


@asm(Uint32, pure=False)
def bar_red_or(bid, n, pred):
    """bar.red.or over the threads of a named barrier: 1 if any thread passed pred != 0."""
    code = ("{ .reg .pred p, q; setp.ne.u32 p, $3, 0; bar.red.or.pred q, $1, $2, p; "
            "selp.u32 $0, 1, 0, q; }")
    return code, "=r,r,r,r,~{memory}", (Uint32(bid), Uint32(n), Uint32(pred))


# MMA issue by the whole warp with one elected thread: the operands stay warp-uniform
# (uniform registers) instead of an ELECT + R2UR.BROADCAST waterfall per MMA under a
# lane-0 branch (~100 clk per MMA issue, P-MLA run83 stamps)
_ELECT = "elect.sync _|e, 0xffffffff; "


@asm()
def mma_ts_e(d, a_tmem, b, idesc, acc):
    code = ("{ .reg .pred p, e; " + _ELECT + "setp.ne.b32 p, $4, 0; "
            "@e tcgen05.mma.cta_group::1.kind::f8f6f4 [$0], [$1], $2, $3, p; }")
    return code, "r,r,l,r,r,~{memory}", (Uint32(d), Uint32(a_tmem), Uint64(b),
                                          Uint32(idesc), Uint32(acc))


@asm()
def mma_ss_e(d, a, b, idesc, acc):
    code = ("{ .reg .pred p, e; " + _ELECT + "setp.ne.b32 p, $4, 0; "
            "@e tcgen05.mma.cta_group::1.kind::f8f6f4 [$0], $1, $2, $3, p; }")
    return code, "r,l,l,r,r,~{memory}", (Uint32(d), Uint64(a), Uint64(b), Uint32(idesc),
                                          Uint32(acc))


@asm()
def mma_cb_e(d, a, b, idesc, acc, op: str):
    code = ("{ .reg .pred p, e; " + _ELECT + "setp.ne.b32 p, $4, 0; "
            f"@e tcgen05.mma.cta_group::1.kind::f8f6f4.collector::b::{op} [$0], $1, $2, $3, p; }}")
    return code, "r,l,l,r,r,~{memory}", (Uint32(d), Uint64(a), Uint64(b), Uint32(idesc),
                                          Uint32(acc))


@asm()
def bulk_wait_read2():
    return "cp.async.bulk.wait_group.read 2;", "~{memory}", ()


@asm()
def cp16(dst, src, n):
    """16 bytes global -> shared (cp.async; n = 0 zero-fills)."""
    return ("cp.async.cg.shared.global [$0], [$1], 16, $2;", "r,l,r,~{memory}",
            (Uint32(dst), Int64(src), Uint32(n)))


@asm()
def cp_commit():
    return ("cp.async.commit_group;", "~{memory}", ())


@asm()
def nanosleep(ns):
    return ("nanosleep.u32 $0;", "r,~{memory}", (Uint32(ns),))


@asm()
def cp_wait_lag():
    """Wait until at most CPA_LAG - 1 cp.async groups of this thread are pending."""
    return (f"cp.async.wait_group {CPA_LAG - 1};", "~{memory}", ())


@asm()
def cp_wait_all():
    return ("cp.async.wait_group 0;", "~{memory}", ())


@asm()
def cp_e(t, sdesc):
    """128 rows x 32 bytes from shared memory (descriptor) to tensor memory."""
    code = "{ .reg .pred e; " + _ELECT + "@e tcgen05.cp.cta_group::1.128x256b [$0], $1; }"
    return code, "r,l,~{memory}", (Uint32(t), Uint64(sdesc))


@asm()
def commit_e(bar):
    code = ("{ .reg .pred e; " + _ELECT + "@e tcgen05.commit.cta_group::1.mbarrier::arrive::one"
            ".shared::cluster.b64 [$0]; }")
    return code, "r,~{memory}", (Uint32(bar),)


@asm(Uint32)
def ballot(pred):
    code = "{ .reg .pred p; setp.ne.u32 p, $1, 0; vote.sync.ballot.b32 $0, p, 0xffffffff; }"
    return code, "=r,r", (Uint32(pred),)


@asm(Uint32)
def bfind(v):
    return "bfind.u32 $0, $1;", "=r,r", (Uint32(v),)


@asm()
def tmem_ld_o4(addr):
    """Column h of the four O^T blocks (columns addr + 96 b)."""
    code = ("tcgen05.ld.sync.aligned.32x32b.x1.b32 {$0}, [$4]; "
            "tcgen05.ld.sync.aligned.32x32b.x1.b32 {$1}, [$5]; "
            "tcgen05.ld.sync.aligned.32x32b.x1.b32 {$2}, [$6]; "
            "tcgen05.ld.sync.aligned.32x32b.x1.b32 {$3}, [$7]; "
            "tcgen05.wait::ld.sync.aligned;")
    a = Uint32(addr)
    return (code, "=f,=f,=f,=f,r,r,r,r,~{memory}",
            (a, a + Uint32(96), a + Uint32(192), a + Uint32(288)), (Float32,) * 4)


@asm()
def tmem_st_o4(addr, v0, v1, v2, v3):
    code = ("tcgen05.st.sync.aligned.32x32b.x1.b32 [$0], {$4}; "
            "tcgen05.st.sync.aligned.32x32b.x1.b32 [$1], {$5}; "
            "tcgen05.st.sync.aligned.32x32b.x1.b32 [$2], {$6}; "
            "tcgen05.st.sync.aligned.32x32b.x1.b32 [$3], {$7}; "
            "tcgen05.wait::st.sync.aligned;")
    a = Uint32(addr)
    return code, "r,r,r,r,f,f,f,f,~{memory}", (a, a + Uint32(96), a + Uint32(192),
                                               a + Uint32(288), Float32(v0), Float32(v1),
                                               Float32(v2), Float32(v3))


@asm()
def tmem_ld16x256x3(a0, a1, a2):
    """Three tcgen05.ld 16x256b.x4 (see sparse_mla_h5.tmem_ld16x256) under one wait."""
    regs = [",".join(f"${16 * k + i}" for i in range(16)) for k in range(3)]
    code = "".join(f"tcgen05.ld.sync.aligned.16x256b.x4.b32 {{{regs[k]}}}, [${48 + k}]; "
                   for k in range(3)) + "tcgen05.wait::ld.sync.aligned;"
    return (code, ",".join(["=f"] * 48 + ["r", "r", "r", "~{memory}"]),
            (Uint32(a0), Uint32(a1), Uint32(a2)), (Float32,) * 48)


@asm()
def tmem_zero128(addr):
    """Zero 128 consecutive 32-bit TMEM columns of the calling warp's 32 lanes (16 x
    32x32b.x8 stores of one zero register) and wait for them."""
    z = ",".join(["$1"] * 8)
    code = "".join(f"tcgen05.st.sync.aligned.32x32b.x8.b32 [$0 + {8 * c}], {{{z}}}; " for c in range(16))
    code += "tcgen05.wait::st.sync.aligned;"
    return code, "r,r,~{memory}", (Uint32(addr), Uint32(0))


@asm()
def tmem_ld_max64(addr):
    """64 columns of this thread's TMEM lane and their maximum (tcgen05.ld.red)."""
    regs = ",".join(f"${i}" for i in range(64))
    code = (f"tcgen05.ld.red.sync.aligned.32x32b.x64.max.f32 {{{regs}}}, $64, [$65]; "
            "tcgen05.wait::ld.sync.aligned;")
    return (code, ",".join(["=f"] * 65 + ["r", "~{memory}"]), (Uint32(addr),),
            (Float32,) * 65)


@asm()
def tmem_ld_max32(addr):
    regs = ",".join(f"${i}" for i in range(32))
    code = (f"tcgen05.ld.red.sync.aligned.32x32b.x32.max.f32 {{{regs}}}, $32, [$33]; "
            "tcgen05.wait::ld.sync.aligned;")
    return (code, ",".join(["=f"] * 33 + ["r", "~{memory}"]), (Uint32(addr),),
            (Float32,) * 33)


@cute.jit
def _subs(n):
    """64-key subtiles of a row (at least one: empty rows still write zeros)."""
    t = (n + 63) // 64
    if t < 1:
        t = Int32(1)
    return t


class SparseMlaPrefillH8:
    def __init__(self, grid: int, knobs: tuple = (), topk: int = 0):
        self.grid = grid
        # topk > 0: logical mode. ``slots`` holds the indexer's request-relative top-k
        # (width topk, < 0 invalid); the gather warps map them through the block table
        # (row = block_table[req][idx >> bs_shift] * unit_rows + idx % block size) and
        # the Q warp counts each token's valid entries (``lengths`` is unused).
        self.logical = topk > 0
        self.topk = topk
        assert topk == 0 or (topk % 64 == 0 and 128 <= topk <= 2048), topk  # lane per chunk
        # diagnostics (timing only): no_store (no TMA stores), no_gather (stages are
        # signalled without gathers: stale data), no_math (softmax warps skip all math),
        # no_mma (MMA warps only commit), sstamp (CTA 0 shared-memory clock stamps)
        # pad: 1 KiB of extra dynamic shared memory (no stamps)
        # ssmask: roles whose sstamp sites are compiled (1 softmax, 2 QK, 4 PV, 8 gather)
        # pin: loader code-layout experiment (1: clock read only at the two gather stamp
        # sites, 2: the guarded stamp store to a scratch word)
        # parts: issuing lanes per 4-row quad (its 5 gather4 ops are spread over them)
        # gskip (timing only): skip the last gskip gather4 ops of every quad (RoPE, then
        # latent atoms 3, 2, ...): stale stage data
        # cpa (KV gather + logical-mode slot producer): 0 = the production form = 3: every
        # 16 B of every row by per-lane cp.async (not starved by the SS MMA operand reads as
        # TMA gather4 is) + the shared-memory top-k slot producer; diagnostics: 9 = TMA
        # gather4 + global-load slot producer (the pre-2026-11-20 form), 1 / 2 = RoPE / all
        # rows by cp.async with per-thread / lagged arrives (physical mode), 4 = 3 without
        # the consumer proxy fence, 5 = 3 with lagged producer-side fences, 6 = cp.async
        # rows + global-load slot producer, 7 = TMA gather4 + shared-memory slot producer
        # qkh: QK in two 32-key halves, each with its own S-full / S-free barriers, so
        # QK(u + 1) keys 0-31 only wait for the softmax load of S(u) keys 0-31
        # lglob (logical mode, timing only): every role reads token counts from
        # ``lengths`` instead of the producer's shared-memory ring
        # xphys (logical mode, timing only): the gather warps run the physical-slot loop on
        # the slots passed as ``stamp`` (counts from ``lengths``); the producer still runs,
        # gated by B_KVEMPTY
        # ssu (sstamp diagnostic): first stamped subtile (0 -> 16); 28 straddles the
        # token 0 / token 1 boundary
        # epb: the epilogue loads each 16-lane half of an O^T block (three 16x256b loads)
        # under one tcgen05.wait::ld (production: knob 0; 9 = one wait per load)
        # epw: warp 19 (idle gather warp pw 8) issues the epilogue's TMA stores and waits
        # for them, so the owner softmax warpgroup only writes the staging and returns to
        # its next subtile (production: knob 0; 9 = stores issued and awaited in the WG)
        # msk: a token's last subtile takes the masked softmax path only when it is
        # partial (n % 64 != 0); a full one keeps the tcgen05.ld.red max (exact either
        # way) (production: knob 0; 9 = always masked)
        # zf (timing only): every cp.async zero-fills (src-size 0: no L2 request), so the
        # gather warps keep their instruction stream without delivering any data
        # ewslow (timing only, tests): the epw store warp sleeps ewslow ns before each
        # block of epi_store, so the softmax producer runs ahead of it (production: 0)
        # pace (timing only): every gather lane sleeps pace ns after each of its first
        # 8 cp.async of a row, so a stage's 72 LDGSTS leave the SM in 8-instruction slices
        # nbs: S-free as named barrier NB_SFREE (the owning softmax warpgroup bar.arrives
        # after its S load, the QK warp bar.syncs before QK(u + 1)): one generation per
        # subtile (WG u % 2 can only arrive for u once QK(u) was issued, i.e. after the
        # QK warp consumed u - 1), no shared-memory mbarrier round trip on the S chain
        # nex (timing only): the P math uses FMUL2 instead of MUFU ex2 (wrong values)
        # sw: how a cp.async gather warp waits for a free (KVEMPTY) and capped (KVFULL of
        # stage g - CP_CAP) KV stage: 0 = lane 0 waits both, then a warp sync (production);
        # 1 = every lane waits both; 2 = lane 0 KVEMPTY + every lane the cap; 3 = every lane
        # KVEMPTY + lane 0 the cap (P-MLA pmla12: in the bare producer a lane-0 wait + warp
        # sync costs ~216 clk per stage; in the kernel every-lane waits on KVEMPTY are slower)
        # cpb (timing only): every cp.async gather lane copies only the first cpb of its 9
        # chunks per row (the rest of the stage stays stale): KV bytes vs ring protocol
        # qz: the QK warp (TMEM lanes 96-127) zeroes the 32 padding-head lanes of the latent
        # Q columns after each token's Q copy (the copy fills them with the neighbouring Q
        # atom's rows), so the QK MMAs multiply zeros there. The kernel is power-bound at the
        # 2.3 kW cap (P-MLA run184-185: sustained -2.5%, bitwise equal): operand energy is
        # time. Production: knob 0; 9 = off (diagnostic).
        knobs = tuple(knobs) + (0,) * (28 - len(knobs))
        (self.no_store, self.no_gather, self.no_math, self.no_mma, self.sstamp,
         self.pad, self.ssmask, self.pin, self.parts, self.gskip, self.cpa,
         self.qkh, self.lglob, self.xphys, self.pvd, self.tconv, self.ssu,
         self.epb, self.epw, self.msk, self.zf, self.pace, self.nbs, self.nex,
         self.sw, self.cpb, self.qz, self.ewslow) = knobs[:28]
        self.msk = self.msk != 9
        self.ssu = self.ssu or 16
        self.epb = self.epb != 9
        self.qz = self.qz != 9
        if self.cpa == 0:
            self.cpa = 3
        elif self.cpa == 9:
            self.cpa = 0
        if self.xphys:
            self.lglob = 1
        if not self.parts:
            self.parts = GATHER_PARTS
        if self.sstamp and not self.ssmask:
            self.ssmask = 15
        assert not (self.logical and self.cpa in (1, 2)), "cpa 1-2: physical-slot diagnostics"
        self.smem = SMEM_BYTES + (1024 if self.sstamp or self.pad else 0)
        self.sm_arrivals = 4  # S-free / O-ready arrivals per subtile (one per softmax warp)
        # cpa 3-6: KV rows by per-lane cp.async (6: with the old slot producer); cpa 3-5 and 7
        # (TMA gather4): logical mode with the shared-memory top-k slot producer
        self.ldgsts = self.cpa in (3, 4, 5, 6)
        self.prod2 = self.cpa in (3, 4, 5, 7)
        # pvd: PV(u) is issued only after QK(u + pvd) was issued (within a token), so the
        # tensor pipe runs QK(u + pvd), PV(u), QK(u + pvd + 1): with pvd 2 the PV of an
        # older subtile covers the single-S round trip (S(u) load -> QK(u + 1) issue)
        # production pvd 2 (knob 0); 9 = off (diagnostic); qkh issues QK in halves: no pvd
        self.pvd = 0 if self.qkh or self.pvd == 9 else (self.pvd or 2)
        assert self.pvd in (0, 1, 2, 3), self.pvd
        # tconv (logical mode, cp.async gather): the Q warp converts each token's top-k
        # copy in shared memory into physical rows one token ahead (all of a batch's
        # block-table loads in flight) and the gather warps read rows from it: no
        # per-stage slot ring (its register-rotated loads cost ~1 L2 round trip/stage)
        # (production: knob 0; 9 = the per-stage slot producer, diagnostic)
        self.tconv = self.tconv != 9 and self.logical and self.cpa in (3, 4) and not self.xphys
        self.epw = (self.epw != 9 and self.cpa in (3, 4) and (self.tconv or not self.logical))
        # KVFULL arrivals per stage: one noinc async arrive per copying lane (3, 4, 6), one
        # per gather warp (5, 2 + the expect-tx arrive), per-lane RoPE copies (1), TMA (0)
        self.kv_full_arrivals = (256 if self.ldgsts and self.cpa != 5 else 8 if self.cpa == 5
                                 else 1 + (8 if self.cpa == 2 else 256 if self.cpa == 1 else 0))

    @cute.jit
    def __call__(self, q8: Int64, rows: Int32, kv: Int64, unit_rows: Int32, slots: Int64,
                 slot_stride: Int32, lengths: Int64, out: Int64, scale_log2: Float32,
                 out_scale: Float32, kv_count: Int32, stamp: Int64, req_ids: Int64,
                 btab: Int64, bt_stride: Int32, bt_cols: Int32, bs_shift: Int32,
                 stream: cuda.CUstream):
        u8 = cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8
        bf16 = cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16
        l2 = cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B
        sw = cuda.CUtensorMapSwizzle
        qn = tensormap.encode_map(q8, u8, (D_NOPE, rows * H), (ROW,), (128, H),
                                  sw.CU_TENSOR_MAP_SWIZZLE_128B, l2)
        qr = tensormap.encode_map(q8 + D_NOPE, u8, (64, rows * H), (ROW,), (64, H),
                                  sw.CU_TENSOR_MAP_SWIZZLE_64B, l2)
        kvm = tensormap.encode_map(kv, u8, (ROW, kv_count), (ROW,), (128, 1),
                                   sw.CU_TENSOR_MAP_SWIZZLE_128B, l2)
        kvr = tensormap.encode_map(kv, u8, (ROW, kv_count), (ROW,), (64, 1),
                                   sw.CU_TENSOR_MAP_SWIZZLE_64B, l2)
        om = tensormap.encode_map(out, bf16, (D_NOPE, rows * H), (D_NOPE * 2,), (64, H),
                                  sw.CU_TENSOR_MAP_SWIZZLE_128B, l2)
        self.kernel(qn, qr, kvm, kvr, om, rows, unit_rows, slots, slot_stride, lengths,
                    scale_log2, out_scale, stamp, kv, req_ids, btab, bt_stride, bt_cols,
                    bs_shift).launch(
            grid=(self.grid, 1, 1), block=(THREADS, 1, 1), smem=self.smem,
            stream=stream, min_blocks_per_mp=1)

    @cute.kernel
    def kernel(self, qn: tensormap.GridMap, qr: tensormap.GridMap,
               kvm: tensormap.GridMap, kvr: tensormap.GridMap, om: tensormap.GridMap,
               rows: Int32, unit_rows: Int32, slots: Int64, slot_stride: Int32,
               lengths: Int64, scale_log2: Float32, out_scale: Float32, stamp: Int64,
               kvp: Int64, req_ids: Int64, btab: Int64, bt_stride: Int32, bt_cols: Int32,
               bs_shift: Int32):
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(tid // 32)
        lane = tid % 32
        smem = cutlass.utils.SmemAllocator()
        raw = smem.allocate_array(Uint8, self.smem, byte_alignment=1024)
        base = Uint32(raw.toint())
        if tid == 0:
            for i, cnt in ((B_QTM, 4), (B_QLFREE, 1), (B_OFULL, 1), (B_EPIDONE, 1)):
                mbar_init(_bar(base, i), Uint32(cnt))
            for j in cutlass.range_constexpr(2):
                mbar_init(_bar(base, B_SFULL + j), Uint32(1))
                mbar_init(_bar(base, B_SFREE + j), Uint32(self.sm_arrivals))
                mbar_init(_bar(base, B_SFULL1 + j), Uint32(1))
                mbar_init(_bar(base, B_SFREE1 + j), Uint32(self.sm_arrivals))
                mbar_init(_bar(base, B_QFULL + j), Uint32(1))
                mbar_init(_bar(base, B_QREMPTY + j), Uint32(1))
                mbar_init(_bar(base, B_TKI + j), Uint32(1))
            for j in cutlass.range_constexpr(8):
                mbar_init(_bar(base, B_QKI + j), Uint32(1))
            for j in cutlass.range_constexpr(2):
                mbar_init(_bar(base, B_TKCONV + j), Uint32(1))
                mbar_init(_bar(base, B_TKFREE + j), Uint32(8))
                mbar_init(_bar(base, B_EPIFULL + j), Uint32(1))
                mbar_init(_bar(base, B_EPIFREE + j), Uint32(1))
            for j in cutlass.range_constexpr(4):
                mbar_init(_bar(base, B_OFREE + j), Uint32(4))
            for j in cutlass.range_constexpr(NP):
                mbar_init(_bar(base, B_OREADY + j), Uint32(self.sm_arrivals))
                mbar_init(_bar(base, B_PEMPTY + j), Uint32(1))
            for s in cutlass.range_constexpr(NS):
                mbar_init(_bar(base, B_KVFULL + s), Uint32(self.kv_full_arrivals))
                mbar_init(_bar(base, B_KVEMPTY + s), Uint32(1))
            for j in cutlass.range_constexpr(NCR):
                mbar_init(_bar(base, B_NFULL + j), Uint32(1))
            for j in cutlass.range_constexpr(NSL):
                mbar_init(_bar(base, B_SLFULL + j), Uint32(1))
                mbar_init(_bar(base, B_SLEMPTY + j), Uint32(NLD))
            cute.arch.mbarrier_init_fence()
        if warp == 0:
            tmem_alloc_exclusive(base + Uint32(OFF_TMEM))
        tc_fence_before()
        cute.arch.sync_threads()
        tc_fence_after()
        taddr = ld_shared_u32(base + Uint32(OFF_TMEM))
        n_mine = (rows - cta + self.grid - 1) // self.grid
        if warp < W_SM:
            cute.arch.warpgroup_reg_alloc(REG_SM)
            self.softmax(base, tid, warp, lane, taddr, n_mine, lengths, cta, scale_log2,
                         out_scale, tensormap.map_address(om))
            cute.arch.warpgroup_reg_dealloc(REG_LAUNCH)
        else:
            cute.arch.warpgroup_reg_dealloc(REG_LOW)
            taddr = Uint32(cute.arch.make_warp_uniform(taddr))
            if warp == W_PV:
                self.pv_loop(base, lane, taddr, n_mine, lengths, cta)
            elif warp == W_QK:
                self.qk_loop(base, lane, taddr, n_mine, lengths, cta)
            else:
                pw = warp - (W_Q + 1)  # gather 10 -> 0, 12-19 -> 1-8; Q warp 9 -> NLD
                if warp == W_Q:
                    pw = Int32(NLD)
                elif warp > W_QK:
                    pw = warp - (W_Q + 2)
                if cutlass.const_expr(self.epw):
                    if warp == THREADS // 32 - 1:
                        self.epi_store(base, lane, cta, n_mine, tensormap.map_address(om))
                self.loader(base, pw, lane, cta, n_mine, tensormap.map_address(qn),
                            tensormap.map_address(qr), tensormap.map_address(kvm),
                            tensormap.map_address(kvr), unit_rows, slots, slot_stride,
                            lengths, kvp, req_ids, btab, bt_stride, bt_cols, bs_shift, stamp)
            cute.arch.warpgroup_reg_alloc(REG_LAUNCH)
        tc_fence_before()
        cute.arch.sync_threads()
        if cutlass.const_expr(self.sstamp):
            if cta == 0 and tid < 256 and stamp != Int64(0):
                st_global_u32(stamp + Int64(4 * tid),
                              ld_shared_u32(base + Uint32(SMEM_BYTES + 4 * tid)))
        if warp == 0:
            tc_fence_after()
            tmem_free_exclusive(taddr)

    @cute.jit
    def ss(self, base, cta, lane, u, slot, role: cutlass.Constexpr = 15):
        """sstamp: CTA 0, subtiles 16-23, [8][32] u32 clock slots (lane 0 of the caller)."""
        if cutlass.const_expr(self.sstamp and (self.ssmask & role)):
            if cta == 0 and lane == 0 and u >= self.ssu and u < self.ssu + 8:
                sstamp_clock(base + Uint32(SMEM_BYTES)
                             + Uint32(((u - self.ssu) * 32 + Int32(slot)) * 4))

    def _tma_ops(self):
        if self.cpa == 2:
            return 0
        return 4 if self.cpa == 1 else 5 - self.gskip

    def _skipped(self):
        return sum((4096 if a == 4 else 8192) for a in range(self._tma_ops(), 5))

    @cute.jit
    def lpin(self, base, cta, lane, g):
        if cutlass.const_expr(self.pin == 1):
            clock_read()
        if cutlass.const_expr(self.pin == 2):
            if cta == 0 and lane == 0 and g >= 16 and g < 24:
                sstamp_clock(base + Uint32(OFF_FLAG + 12))

    # ------------------------------------------------------------------ gather
    @cute.jit
    def loader(self, base, pw, lane, cta, n_mine, tm_qn, tm_qr, tm_kv, tm_kr, unit_rows,
               slots, slot_stride, lengths, kvp, req_ids, btab, bt_stride, bt_cols,
               bs_shift, stamp):
        quads = (16 - pw + NLD - 1) // NLD
        lptr = cute.make_ptr(Int32, lengths, cute.AddressSpace.gmem, assumed_align=4)
        lens = cute.make_tensor(lptr, cute.make_layout(1 << 30))
        if pw == NLD:
            # Q warp: atoms 0-1 staging (single) is free once token qi - 1's Q is in TMEM
            # and token qi - 2's epilogue has stored its staging; SS buffer qi % 2 once
            # token qi - 2's last QK is done
            if cutlass.const_expr(self.tconv):
                self.slot_producer3(base, lane, cta, n_mine, tm_qn, tm_qr, unit_rows, slots,
                                    slot_stride, req_ids, btab, bt_stride, bt_cols, bs_shift)
            elif cutlass.const_expr(self.logical and self.prod2):
                self.slot_producer2(base, lane, cta, n_mine, tm_qn, tm_qr, unit_rows, slots,
                                    slot_stride, req_ids, btab, bt_stride, bt_cols, bs_shift)
            elif cutlass.const_expr(self.logical):
                self.slot_producer(base, lane, cta, n_mine, tm_qn, tm_qr, unit_rows, slots,
                                   slot_stride, req_ids, btab, bt_stride, bt_cols, bs_shift)
            for qi in range(0 if self.logical else n_mine):
                row = cta + qi * self.grid
                if lane == 0:
                    if qi >= 1:
                        mbar_wait(_bar(base, B_QLFREE), Uint32((qi - 1) & 1))
                    if qi >= 2:
                        mbar_wait(_bar(base, B_QREMPTY + qi % 2), Uint32((qi // 2 - 1) & 1))
                        mbar_wait(_bar(base, B_EPIDONE), Uint32((qi - 2) & 1))
                    full = _bar(base, B_QFULL + qi % 2)
                    qs = base + Uint32(OFF_QS) + Uint32(qi % 2) * Uint32(QS_BYTES)
                    mbar_arrive_tx(full, Uint32(4 * H * 128 + H * 64))
                    for j in cutlass.range_constexpr(4):
                        tma_tile2d(base + Uint32(OFF_QN + j * ATOM), tm_qn, full,
                                   Int32(128 * j), row * H)
                    tma_tile2d(qs, tm_qr, full, Int32(0), row * H)
                cute.arch.sync_warp()
        else:
            # lanes P j .. P j + P - 1 gather quad q = pw + NLD j of every 64-key stage
            # (P = parts; lane part p issues ops p, p + P, ... of the quad's 4 latent atoms
            # + RoPE): TMA gather4 throughput scales with issuing threads, not warps
            # (hwdb 17: ~32 issuers per SM); the 4 slot indices are loaded one stage ahead
            npart = self.parts
            part = lane % npart
            q = pw + NLD * (lane // npart)
            act = lane < quads * npart
            if not act:
                q = pw  # idle lanes read a valid slot of the stage (unused)
            if cutlass.const_expr(self.xphys):
                slots = stamp
            if cutlass.const_expr(self.tconv):
                if pw < 8:
                    self.gather_cp_tconv(base, pw, lane, cta, n_mine, lens, kvp)
            elif cutlass.const_expr(self.logical and not self.xphys and self.ldgsts):
                self.gather_cp_logical(base, pw, lane, cta, n_mine, lens, kvp)
            elif cutlass.const_expr(self.logical and not self.xphys):
                self.gather_logical(base, pw, lane, cta, n_mine, q, act, part, tm_kv, tm_kr,
                                    lens)
            elif cutlass.const_expr(self.ldgsts):
                self.gather_cp(base, pw, lane, cta, n_mine, slots, slot_stride, lens, kvp,
                               unit_rows)
            else:
                g = Int32(0)
                for qi in range(n_mine):
                    row = cta + qi * self.grid
                    n = lens[row]
                    subs = _subs(n)
                    srow = slots + Int64(row) * Int64(slot_stride) * Int64(4) + Int64(16 * q)
                    c0 = c1 = c2 = c3 = Int32(-1)
                    if cutlass.const_expr(self.cpa != 2):
                        c0, c1, c2, c3 = ld_global_v4_s32(srow)
                    # cpa: lane id = 32 pw + lane < 256 copies RoPE chunk id % 4 of row id // 4
                    rid = pw * 32 + lane
                    rrow = rid // 4
                    rch = rid % 4
                    rop = rid < (256 if self.cpa in (1, 2) else 0)
                    rsl = slots + Int64(row) * Int64(slot_stride) * Int64(4) + Int64(4 * rrow)
                    cr = ld_global_s32_pred(rsl, Int32(rop))
                    # cpa 2: every 16 B piece by cp.async (LDGSTS reaches the 30.8 TB/s random
                    # 576 B row ceiling with 8 warps once indices are >= 2 stages ahead;
                    # P-MLA-2CTA gather6/probe5): slot index prefetched two stages ahead
                    cr2 = cr
                    if cutlass.const_expr(self.cpa == 2):
                        cr2 = ld_global_s32_pred(rsl + Int64(256), Int32(rop and subs > 1))
                    # unroll=1: the 2x LLVM unroll (register stage addresses, BRA.DIV gather
                    # blocks) ran the kernel at 3.63 vs 2.02 ms/layer (P-MLA runs 64-71)
                    for t in cutlass.range(subs, unroll=1):
                        s = g % NS
                        st = base + Uint32(OFF_KV) + Uint32(s) * Uint32(STAGE64)
                        full = _bar(base, B_KVFULL + s)
                        key = t * 64 + 4 * q
                        r = (c0, c1, c2, c3)
                        rc = cr
                        if t + 1 < subs:
                            if cutlass.const_expr(self.cpa != 2):
                                c0, c1, c2, c3 = ld_global_v4_s32(srow + Int64((t + 1) * 256))
                            if cutlass.const_expr(self.cpa == 2):
                                cr = cr2
                                cr2 = ld_global_s32_pred(rsl + Int64((t + 2) * 256),
                                                         Int32(rop and t + 2 < subs))
                            elif cutlass.const_expr(self.cpa == 1):
                                cr = ld_global_s32_pred(rsl + Int64((t + 1) * 256), Int32(rop))
                        # keep the prefetch here (issued before the stage-free wait): without
                        # a barrier the compiler sinks it to its use, exposing an L2 round trip
                        # per stage (2.0 vs 3.6 ms, P-MLA run66)
                        compiler_fence()
                        if lane == 0:
                            mbar_wait(_bar(base, B_KVEMPTY + s), Uint32(((g // NS) & 1) ^ 1))
                            if pw == 0:
                                self.ss(base, cta, lane, g, 22, 8)
                                self.lpin(base, cta, lane, g)
                                if cutlass.const_expr(self.no_gather):
                                    mbar_arrive(full)
                                else:
                                    mbar_arrive_tx(full, Uint32(STAGE64 - self._skipped()))
                        cute.arch.sync_warp()
                        if cutlass.const_expr(self.cpa in (1, 2) and not self.no_gather):
                            if rop:
                                src = kvp
                                nb = Uint32(0)
                                if rc >= 0 and t * 64 + rrow < n:
                                    src = kvp + Int64((rc >> 6) * unit_rows + (rc & 63)) * Int64(
                                        ROW) + Int64(D_NOPE + 16 * rch)
                                    nb = Uint32(16)
                                cp16(st + Uint32(4 * 8192) + Uint32(rrow * 64) + (
                                    (Uint32(rch) ^ ((Uint32(rrow) >> Uint32(1)) & Uint32(3)))
                                    << Uint32(4)), src, nb)
                                if cutlass.const_expr(self.cpa == 2):
                                    # latent chunk c = rch + 4 k (16 B) of row rrow -> SW128 atom
                                    # c // 8, 16 B column (c % 8) ^ (rrow % 8)
                                    lsrc = kvp
                                    if nb > Uint32(0):
                                        lsrc = src - Int64(D_NOPE)
                                    for k in cutlass.range_constexpr(8):
                                        cp16(st + Uint32((k // 2) * 8192) + Uint32(rrow * 128) + (
                                            ((Uint32(rch) + Uint32(4 * (k % 2))) ^ (Uint32(rrow) & Uint32(7)))
                                            << Uint32(4)),
                                            lsrc + Int64(64 * k), nb)
                                    cp_commit()
                                    cp_wait_lag()
                                    cute.arch.sync_warp()
                                    if lane == 0 and g >= CPA_LAG - 1:
                                        mbar_arrive(_bar(base, B_KVFULL + (g - (CPA_LAG - 1)) % NS))
                                else:
                                    cp_async_arrive(full)
                        if cutlass.const_expr(not self.no_gather and self.cpa != 2):
                            if act:
                                r0 = (r[0] >> 6) * unit_rows + (r[0] & 63)
                                r1 = (r[1] >> 6) * unit_rows + (r[1] & 63)
                                r2 = (r[2] >> 6) * unit_rows + (r[2] & 63)
                                r3 = (r[3] >> 6) * unit_rows + (r[3] & 63)
                                if r[0] < 0 or key >= n:
                                    r0 = Int32(-1)
                                if r[1] < 0 or key + 1 >= n:
                                    r1 = Int32(-1)
                                if r[2] < 0 or key + 2 >= n:
                                    r2 = Int32(-1)
                                if r[3] < 0 or key + 3 >= n:
                                    r3 = Int32(-1)
                                for k in cutlass.range_constexpr((5 + npart - 1) // npart):
                                    a = part + npart * k
                                    if a < self._tma_ops():
                                        dst = st + Uint32(a * 8192) + Uint32(q) * Uint32(512)
                                        desc = tm_kv
                                        col = 128 * a
                                        if a == 4:
                                            dst = st + Uint32(4 * 8192) + Uint32(q) * Uint32(256)
                                            desc = tm_kr
                                            col = Int32(D_NOPE)
                                        gather4(dst, desc, full, col, r0, r1, r2, r3)
                        if pw == 0:
                            self.ss(base, cta, lane, g, 23, 8)
                            self.lpin(base, cta, lane, g)
                        g = g + 1
                if cutlass.const_expr(self.cpa == 2 and not self.no_gather):
                    # drain: the last CPA_LAG - 1 stages are signalled here
                    if pw * 32 + lane < 256:
                        cp_wait_all()
                        cute.arch.sync_warp()
                        if lane == 0:
                            for j in cutlass.range_constexpr(CPA_LAG - 1):
                                if g - (CPA_LAG - 1) + j >= 0:
                                    mbar_arrive(_bar(base, B_KVFULL + (g - (CPA_LAG - 1) + j) % NS))

    @cute.jit
    def gather_cp(self, base, pw, lane, cta, n_mine, slots, slot_stride, lens, kvp,
                  unit_rows):
        """cpa 3: every 16 B of every row by cp.async (LDGSTS is not starved by the SS
        MMA operand reads, TMA gather4 is: hwdb 24/32 vs P-MLA-2CTA probe5). Warps
        pw 0-7 own rows 8 pw .. 8 pw + 7 of every 64-key stage; lane = (row, 16 B column
        ch + 4 j of the latent, ch of the RoPE). Each lane's noinc async arrive on
        KVFULL tracks its own copies, so no gather warp ever waits for its loads (a
        CTA-wide barrier with 256 such arrivals per phase costs nothing: pmla10 M2)."""
        if pw < 8:
            r = pw * 8 + lane // 4
            ch = lane % 4
            rx = (lane // 4) % 8
            d0 = Uint32(r * 128) + (Uint32(ch ^ rx) << Uint32(4))
            d1 = Uint32(r * 128) + (Uint32((ch + 4) ^ rx) << Uint32(4))
            dr = Uint32(4 * 8192 + r * 64) + (Uint32(ch ^ ((r >> 1) & 3)) << Uint32(4))
            g = Int32(0)
            for qi in range(n_mine):
                row = cta + qi * self.grid
                n = lens[row]
                subs = _subs(n)
                rsl = slots + Int64(row) * Int64(slot_stride) * Int64(4) + Int64(4 * r)
                c1 = ld_global_s32_pred(rsl, Int32(r < n))
                c2 = ld_global_s32_pred(rsl + Int64(256), Int32(subs > 1 and 64 + r < n))
                for t in cutlass.range(subs, unroll=1):
                    s = g % NS
                    st = base + Uint32(OFF_KV) + Uint32(s) * Uint32(STAGE64)
                    rc = c1
                    c1 = c2
                    c2 = ld_global_s32_pred(rsl + Int64((t + 2) * 256),
                                            Int32(t + 2 < subs and (t + 2) * 64 + r < n))
                    compiler_fence()
                    self.stage_wait(base, lane, g, s)
                    src = kvp
                    nb = Uint32(0)
                    if rc >= 0:
                        src = kvp + Int64((rc >> 6) * unit_rows + (rc & 63)) * Int64(
                            ROW) + Int64(16 * ch)
                        nb = Uint32(16)
                    self.cp_row(st, d0, d1, dr, src, nb)
                    self.cp_signal(base, lane, g, s)
                    g = g + 1
            self.cp_drain(lane, base, g)

    @cute.jit
    def stage_wait(self, base, lane, g, s):
        """cp.async gather warp: KV stage g (ring slot s) is free (KVEMPTY) and within the
        in-flight cap (form: knob sw)."""
        if cutlass.const_expr(self.sw in (0, 2)):
            if lane == 0:
                mbar_wait(_bar(base, B_KVEMPTY + s), Uint32(((g // NS) & 1) ^ 1))
                if cutlass.const_expr(self.sw == 0):
                    self.cp_cap(base, g)
            cute.arch.sync_warp()
            if cutlass.const_expr(self.sw == 2):
                self.cp_cap(base, g)
        else:
            mbar_wait(_bar(base, B_KVEMPTY + s), Uint32(((g // NS) & 1) ^ 1))
            if cutlass.const_expr(self.sw == 1):
                self.cp_cap(base, g)
            else:
                if lane == 0:
                    self.cp_cap(base, g)
                cute.arch.sync_warp()

    @cute.jit
    def cp_row(self, st, d0, d1, dr, src, nb):
        """This lane's 9 x 16 B of its row (latent chunks ch + 4 j into SW128 atoms j // 2,
        RoPE chunk ch into the SW64 atom); nb = 0 zero-fills (invalid key)."""
        if cutlass.const_expr(self.zf):
            nb = Uint32(0)
        if cutlass.const_expr(not self.no_gather):
            nc = self.cpb if self.cpb else 9
            for a in cutlass.range_constexpr(4):
                if cutlass.const_expr(2 * a < nc):
                    cp16(st + Uint32(a * 8192) + d0, src + Int64(128 * a), nb)
                    if cutlass.const_expr(self.pace):
                        nanosleep(self.pace)
                if cutlass.const_expr(2 * a + 1 < nc):
                    cp16(st + Uint32(a * 8192) + d1, src + Int64(128 * a + 64), nb)
                    if cutlass.const_expr(self.pace):
                        nanosleep(self.pace)
            if cutlass.const_expr(nc == 9):
                cp16(st + dr, src + Int64(D_NOPE), nb)

    @cute.jit
    def cp_cap(self, base, g):
        """Lane 0, cpa 3/4: stage g is issued only once stage g - CP_CAP has landed (its
        KVFULL phase completed), bounding the cp.async bytes in flight."""
        if cutlass.const_expr(self.cpa in (3, 4, 6) and CP_CAP < NS):
            if g >= CP_CAP:
                mbar_wait(_bar(base, B_KVFULL + (g - CP_CAP) % NS),
                          Uint32(((g - CP_CAP) // NS) & 1))

    @cute.jit
    def cp_signal(self, base, lane, g, s):
        """Stage g (ring slot s) copies issued by this lane. cpa 3/4: noinc async arrive on
        KVFULL (the consumer fences the proxies; cpa 4 skips that fence: timing only).
        cpa 5: wait for this lane's stage g - CPA_LAG + 1 copies, fence them into the async
        proxy here (a consumer-side fence.proxy.async waits for every cp.async in flight
        on the SM), one arrive per warp."""
        if cutlass.const_expr(self.cpa == 5):
            cp_commit()
            cp_wait_lag()
            fence_async_smem()
            cute.arch.sync_warp()
            if lane == 0 and g >= CPA_LAG - 1:
                mbar_arrive(_bar(base, B_KVFULL + (g - (CPA_LAG - 1)) % NS))
        else:
            cp_async_arrive(_bar(base, B_KVFULL + s))

    @cute.jit
    def cp_drain(self, lane, base, g):
        """cpa 5: signal the last CPA_LAG - 1 stages of this warp."""
        if cutlass.const_expr(self.cpa == 5):
            cp_wait_all()
            fence_async_smem()
            cute.arch.sync_warp()
            if lane == 0:
                for j in cutlass.range_constexpr(CPA_LAG - 1):
                    if g - (CPA_LAG - 1) + j >= 0:
                        mbar_arrive(_bar(base, B_KVFULL + (g - (CPA_LAG - 1) + j) % NS))

    @cute.jit
    def gather_cp_logical(self, base, pw, lane, cta, n_mine, lens, kvp):
        """cpa 3, logical mode: as gather_cp, physical rows from the slot producer's ring
        (each warp, pw 8 included, releases the entry once its lanes hold their rows)."""
        r = pw * 8 + lane // 4
        ch = lane % 4
        rx = (lane // 4) % 8
        d0 = Uint32(r * 128) + (Uint32(ch ^ rx) << Uint32(4))
        d1 = Uint32(r * 128) + (Uint32((ch + 4) ^ rx) << Uint32(4))
        dr = Uint32(4 * 8192 + r * 64) + (Uint32(ch ^ ((r >> 1) & 3)) << Uint32(4))
        g = Int32(0)
        for qi in range(n_mine):
            subs = _subs(self.ntok(base, lens, cta + qi * self.grid, qi))
            for t in cutlass.range(subs, unroll=1):
                s = g % NS
                st = base + Uint32(OFF_KV) + Uint32(s) * Uint32(STAGE64)
                sl = g % NSL
                mbar_wait(_bar(base, B_SLFULL + sl), Uint32((g // NSL) & 1))  # every lane
                rc = Int32(-1)
                if pw < 8:
                    rc = Int32(ld_shared_u32(base + Uint32(OFF_SLOT) + Uint32(sl) * Uint32(256)
                                             + Uint32(4 * r)))
                cute.arch.sync_warp()
                if lane == 0:
                    mbar_arrive(_bar(base, B_SLEMPTY + sl))
                self.stage_wait(base, lane, g, s)
                if pw < 8:
                    src = kvp
                    nb = Uint32(0)
                    if rc >= 0:
                        src = kvp + Int64(rc) * Int64(ROW) + Int64(16 * ch)
                        nb = Uint32(16)
                    self.cp_row(st, d0, d1, dr, src, nb)
                    self.cp_signal(base, lane, g, s)
                g = g + 1
        if pw < 8:
            self.cp_drain(lane, base, g)

    @cute.jit
    def q_issue(self, base, row, qi, tm_qn, tm_qr):
        """Lane 0 of the Q warp: token qi's Q into the staging (atoms 0-1 free once token
        qi - 1's Q is in TMEM and token qi - 2's epilogue stored its staging; SS buffer
        qi % 2 once token qi - 2's last QK is done)."""
        if qi >= 1:
            mbar_wait(_bar(base, B_QLFREE), Uint32((qi - 1) & 1))
        if qi >= 2:
            mbar_wait(_bar(base, B_QREMPTY + qi % 2), Uint32((qi // 2 - 1) & 1))
            mbar_wait(_bar(base, B_EPIDONE), Uint32((qi - 2) & 1))
        full = _bar(base, B_QFULL + qi % 2)
        qs = base + Uint32(OFF_QS) + Uint32(qi % 2) * Uint32(QS_BYTES)
        mbar_arrive_tx(full, Uint32(4 * H * 128 + H * 64))
        for j in cutlass.range_constexpr(4):
            tma_tile2d(base + Uint32(OFF_QN + j * ATOM), tm_qn, full, Int32(128 * j), row * H)
        tma_tile2d(qs, tm_qr, full, Int32(0), row * H)

    @cute.jit
    def bt2(self, btr, i0, i1, bt_cols, bs_shift):
        """Block-table entries of 2 request-relative indices (-1: invalid index)."""
        k0 = i0 >> bs_shift
        k1 = i1 >> bs_shift
        ok0 = Int32(0)
        ok1 = Int32(0)
        if i0 >= 0:
            if k0 < bt_cols:
                ok0 = Int32(1)
        if i1 >= 0:
            if k1 < bt_cols:
                ok1 = Int32(1)
        return (ld_global_s32_pred(btr + Int64(k0) * Int64(4), ok0),
                ld_global_s32_pred(btr + Int64(k1) * Int64(4), ok1))

    @cute.jit
    def nvalid2(self, a, b, bt_cols, bs_shift):
        c = Int32(0)
        if a >= 0:
            if (a >> bs_shift) < bt_cols:
                c = c + 1
        if b >= 0:
            if (b >> bs_shift) < bt_cols:
                c = c + 1
        return c

    @cute.jit
    def prefix_count(self, lane, src, bt_cols, bs_shift):
        """Valid entries of one top-k row (valid entries first, the stock contract of
        the valid count: kernels attend the first count entries) from two loads per
        lane: the last entry of every 64-entry chunk, then the first chunk that ends
        invalid."""
        nchunk = self.topk // 64
        ok = Int32(0)
        if lane < nchunk:
            ok = Int32(1)
        e = ld_global_s32_pred(src + Int64(4 * (64 * lane + 63)), ok)
        bad = Int32(0)
        if ok != 0:
            if e < 0 or (e >> bs_shift) >= bt_cols:
                bad = Int32(1)
        m = ballot(Uint32(bad))
        n = Int32(self.topk)
        if m != Uint32(0):
            j = Int32(bfind(m & (Uint32(0) - m)))  # first chunk that ends invalid
            a, b = ld_global_v2_s32(src + Int64(256) * Int64(j) + Int64(8 * lane))
            n = 64 * j + redux_add_s32(self.nvalid2(a, b, bt_cols, bs_shift))
        return n

    @cute.jit
    def tki_count(self, buf, lane, bt_cols, bs_shift):
        """prefix_count on the shared-memory copy of a top-k row."""
        nchunk = self.topk // 64
        bad = Int32(0)
        if lane < nchunk:
            e = Int32(ld_shared_u32(buf + Uint32(4 * (64 * lane + 63))))
            if e < 0 or (e >> bs_shift) >= bt_cols:
                bad = Int32(1)
        m = ballot(Uint32(bad))
        n = Int32(self.topk)
        if m != Uint32(0):
            j = Int32(bfind(m & (Uint32(0) - m)))
            ea = Int32(ld_shared_u32(buf + Uint32(256) * Uint32(j) + Uint32(8 * lane)))
            eb = Int32(ld_shared_u32(buf + Uint32(256) * Uint32(j) + Uint32(8 * lane + 4)))
            n = 64 * j + redux_add_s32(self.nvalid2(ea, eb, bt_cols, bs_shift))
        return n

    @cute.jit
    def tki_issue(self, base, slots, sstride, row, j):
        """Lane 0: bulk-copy token ``row``'s top-k row into TKI buffer j."""
        kb = self.topk * 4
        mbar_arrive_tx(_bar(base, B_TKI + j), Uint32(kb))
        bulk_g2s(base + Uint32(OFF_TKI) + Uint32(j) * Uint32(TKB),
                 slots + Int64(row) * sstride, Uint32(kb), _bar(base, B_TKI + j))

    @cute.jit
    def slot_producer2(self, base, lane, cta, n_mine, tm_qn, tm_qr, unit_rows, slots,
                       slot_stride, req_ids, btab, bt_stride, bt_cols, bs_shift):
        """Q warp, logical mode (cpa >= 3): the index remap with no dependent global load
        on its per-stage path. Each token's top-k row is bulk-copied into shared memory
        (OFF_TKI, token parity) while the previous token is produced, its valid count is
        taken from that copy, and the block-table rows of stage g + BTD are loaded at
        stage g (ring b0..b3; -2: not prefetched, loaded on demand: a stage beyond the
        next token of a short one). The slot_producer chain (indices two stages, block
        table one stage ahead) bounded the logical-mode step at ~1 L2 round trip per
        stage."""
        bmask = (Int32(1) << bs_shift) - Int32(1)
        sstride = Int64(slot_stride) * Int64(4)
        tki = base + Uint32(OFF_TKI)
        lo = Uint32(8 * lane)
        if lane == 0:
            self.tki_issue(base, slots, sstride, cta, 0)
            if n_mine > 1:
                self.tki_issue(base, slots, sstride, cta + self.grid, 1)
        cute.arch.sync_warp()
        mbar_wait(_bar(base, B_TKI), Uint32(0))
        n = self.tki_count(tki, lane, bt_cols, bs_shift)
        self.publish_count(base, lane, 0, n)
        if lane == 0:
            self.q_issue(base, cta, Int32(0), tm_qn, tm_qr)
        cute.arch.sync_warp()
        rq = ld_global_s32_pred(req_ids + Int64(cta) * Int64(4), Int32(1))
        btr = btab + Int64(rq) * Int64(bt_stride) * Int64(4)
        b0a, b0b, b1a, b1b = Int32(-2), Int32(-2), Int32(-2), Int32(-2)
        b2a, b2b, b3a, b3b = Int32(-2), Int32(-2), Int32(-2), Int32(-2)
        subs0 = _subs(n)
        if 0 < subs0:
            b0a, b0b = self.bt2(btr, Int32(ld_shared_u32(tki + lo)),
                                Int32(ld_shared_u32(tki + lo + Uint32(4))), bt_cols, bs_shift)
        if 1 < subs0:
            b1a, b1b = self.bt2(btr, Int32(ld_shared_u32(tki + Uint32(256) + lo)),
                                Int32(ld_shared_u32(tki + Uint32(260) + lo)), bt_cols, bs_shift)
        if 2 < subs0:
            b2a, b2b = self.bt2(btr, Int32(ld_shared_u32(tki + Uint32(512) + lo)),
                                Int32(ld_shared_u32(tki + Uint32(516) + lo)), bt_cols, bs_shift)
        if 3 < subs0:
            b3a, b3b = self.bt2(btr, Int32(ld_shared_u32(tki + Uint32(768) + lo)),
                                Int32(ld_shared_u32(tki + Uint32(772) + lo)), bt_cols, bs_shift)
        g = Int32(0)
        for qi in range(n_mine):
            row = cta + qi * self.grid
            subs = _subs(n)
            buf = tki + Uint32(qi % 2) * Uint32(TKB)
            nbuf = tki + Uint32((qi + 1) % 2) * Uint32(TKB)
            nrow = row
            if qi + 1 < n_mine:
                nrow = row + self.grid
            rqn = ld_global_s32_pred(req_ids + Int64(nrow) * Int64(4), Int32(1))
            btrn = btab + Int64(rqn) * Int64(bt_stride) * Int64(4)
            nn = Int32(-1)  # next token's count, once its copy is needed
            for t in cutlass.range(subs, unroll=1):
                s = g % NSL
                tt = t + BTD
                na, nb = Int32(-2), Int32(-2)
                if tt < subs:
                    pa = buf + Uint32(256) * Uint32(tt) + lo
                    na, nb = self.bt2(btr, Int32(ld_shared_u32(pa)),
                                      Int32(ld_shared_u32(pa + Uint32(4))), bt_cols, bs_shift)
                elif qi + 1 < n_mine:
                    if nn < 0:
                        mbar_wait(_bar(base, B_TKI + (qi + 1) % 2), Uint32(((qi + 1) // 2) & 1))
                        nn = self.tki_count(nbuf, lane, bt_cols, bs_shift)
                    tn = tt - subs
                    if tn < _subs(nn):
                        pa = nbuf + Uint32(256) * Uint32(tn) + lo
                        na, nb = self.bt2(btrn, Int32(ld_shared_u32(pa)),
                                          Int32(ld_shared_u32(pa + Uint32(4))), bt_cols, bs_shift)
                pc = buf + Uint32(256) * Uint32(t) + lo
                ia = Int32(ld_shared_u32(pc))
                ib = Int32(ld_shared_u32(pc + Uint32(4)))
                ca, cb = b0a, b0b
                if ca == Int32(-2):
                    ca, cb = self.bt2(btr, ia, ib, bt_cols, bs_shift)
                b0a, b0b, b1a, b1b, b2a, b2b = b1a, b1b, b2a, b2b, b3a, b3b
                b3a, b3b = na, nb
                key = t * 64 + 2 * lane
                p0 = ca * unit_rows + (ia & bmask)
                p1 = cb * unit_rows + (ib & bmask)
                if ca < 0 or key >= n:
                    p0 = Int32(-1)
                if cb < 0 or key + 1 >= n:
                    p1 = Int32(-1)
                compiler_fence()
                if lane == 0:
                    if cutlass.const_expr(self.xphys):
                        mbar_wait(_bar(base, B_KVEMPTY + g % NS), Uint32(((g // NS) & 1) ^ 1))
                    else:
                        mbar_wait(_bar(base, B_SLEMPTY + s), Uint32(((g // NSL) & 1) ^ 1))
                cute.arch.sync_warp()
                st_shared_v2(base + Uint32(OFF_SLOT) + Uint32(s) * Uint32(256) + Uint32(8 * lane),
                             p0, p1)
                cute.arch.sync_warp()
                if lane == 0:
                    mbar_arrive(_bar(base, B_SLFULL + s))
                g = g + 1
            if qi + 1 < n_mine:
                if nn < 0:
                    mbar_wait(_bar(base, B_TKI + (qi + 1) % 2), Uint32(((qi + 1) // 2) & 1))
                    nn = self.tki_count(nbuf, lane, bt_cols, bs_shift)
                self.publish_count(base, lane, qi + 1, nn)
                fence_async_smem()  # generic reads of buffer qi % 2 before its next bulk copy
                cute.arch.sync_warp()
                if lane == 0:
                    if qi + 2 < n_mine:
                        self.tki_issue(base, slots, sstride, row + 2 * self.grid, qi % 2)
                    self.q_issue(base, nrow, qi + 1, tm_qn, tm_qr)
                cute.arch.sync_warp()
            n = nn
            btr = btrn

    @cute.jit
    def gather_cp_tconv(self, base, pw, lane, cta, n_mine, lens, kvp):
        """tconv: as gather_cp_logical, each stage's physical rows read straight from the
        Q warp's converted top-k row (one barrier per token instead of a slot ring)."""
        r = pw * 8 + lane // 4
        ch = lane % 4
        rx = (lane // 4) % 8
        d0 = Uint32(r * 128) + (Uint32(ch ^ rx) << Uint32(4))
        d1 = Uint32(r * 128) + (Uint32((ch + 4) ^ rx) << Uint32(4))
        dr = Uint32(4 * 8192 + r * 64) + (Uint32(ch ^ ((r >> 1) & 3)) << Uint32(4))
        g = Int32(0)
        for qi in range(n_mine):
            subs = _subs(self.ntok(base, lens, cta + qi * self.grid, qi))
            buf = base + Uint32(OFF_TKI) + Uint32(qi % 2) * Uint32(TKB) + Uint32(4 * r)
            mbar_wait(_bar(base, B_TKCONV + qi % 2), Uint32((qi // 2) & 1))
            for t in cutlass.range(subs, unroll=1):
                s = g % NS
                st = base + Uint32(OFF_KV) + Uint32(s) * Uint32(STAGE64)
                rc = Int32(ld_shared_u32(buf + Uint32(256) * Uint32(t)))
                self.stage_wait(base, lane, g, s)
                src = kvp
                nb = Uint32(0)
                if rc >= 0:
                    src = kvp + Int64(rc) * Int64(ROW) + Int64(16 * ch)
                    nb = Uint32(16)
                self.cp_row(st, d0, d1, dr, src, nb)
                self.cp_signal(base, lane, g, s)
                g = g + 1
            cute.arch.sync_warp()
            if lane == 0:
                mbar_arrive(_bar(base, B_TKFREE + qi % 2))
        self.cp_drain(lane, base, g)

    @cute.jit
    def tconv_row(self, buf, lane, row, n, req_ids, btab, bt_stride, bt_cols, bs_shift,
                  unit_rows):
        """Q warp: token ``row``'s top-k copy in ``buf`` -> physical rows in place (-1 for
        invalid entries), 4 stages (8 block-table loads per lane) in flight per batch."""
        bmask = (Int32(1) << bs_shift) - Int32(1)
        rq = ld_global_s32_pred(req_ids + Int64(row) * Int64(4), Int32(1))
        btr = btab + Int64(rq) * Int64(bt_stride) * Int64(4)
        lo = Uint32(8 * lane)
        nst = _subs(n)
        for t0 in cutlass.range(0, nst, 4, unroll=1):
            ii = []
            bb = []
            for k in cutlass.range_constexpr(4):
                pa = buf + Uint32(256) * Uint32(t0 + k) + lo
                i0 = Int32(ld_shared_u32(pa))
                i1 = Int32(ld_shared_u32(pa + Uint32(4)))
                if t0 + k >= nst:
                    i0 = Int32(-1)
                    i1 = Int32(-1)
                b0, b1 = self.bt2(btr, i0, i1, bt_cols, bs_shift)
                ii.append((i0, i1))
                bb.append((b0, b1))
            for k in cutlass.range_constexpr(4):
                i0, i1 = ii[k]
                b0, b1 = bb[k]
                key = (t0 + k) * 64 + 2 * lane
                p0 = b0 * unit_rows + (i0 & bmask)
                p1 = b1 * unit_rows + (i1 & bmask)
                if b0 < 0 or key >= n:
                    p0 = Int32(-1)
                if b1 < 0 or key + 1 >= n:
                    p1 = Int32(-1)
                if t0 + k < nst:
                    st_shared_v2(buf + Uint32(256) * Uint32(t0 + k) + lo, p0, p1)

    @cute.jit
    def slot_producer3(self, base, lane, cta, n_mine, tm_qn, tm_qr, unit_rows, slots,
                       slot_stride, req_ids, btab, bt_stride, bt_cols, bs_shift):
        """Q warp, tconv: each token's top-k row is bulk-copied into shared memory and
        converted there into physical rows one token ahead; the copy of token qi + 2 reuses
        token qi's buffer once every copying gather warp has read it (B_TKFREE)."""
        sstride = Int64(slot_stride) * Int64(4)
        tki = base + Uint32(OFF_TKI)
        if lane == 0:
            self.tki_issue(base, slots, sstride, cta, 0)
            if n_mine > 1:
                self.tki_issue(base, slots, sstride, cta + self.grid, 1)
        cute.arch.sync_warp()
        mbar_wait(_bar(base, B_TKI), Uint32(0))
        n = self.tki_count(tki, lane, bt_cols, bs_shift)
        self.publish_count(base, lane, 0, n)
        if lane == 0:
            self.q_issue(base, cta, Int32(0), tm_qn, tm_qr)
        cute.arch.sync_warp()
        self.tconv_row(tki, lane, cta, n, req_ids, btab, bt_stride, bt_cols, bs_shift,
                       unit_rows)
        cute.arch.sync_warp()
        if lane == 0:
            mbar_arrive(_bar(base, B_TKCONV))
        for qi in range(n_mine):
            row = cta + qi * self.grid
            if qi + 1 < n_mine:
                nbuf = tki + Uint32((qi + 1) % 2) * Uint32(TKB)
                mbar_wait(_bar(base, B_TKI + (qi + 1) % 2), Uint32(((qi + 1) // 2) & 1))
                nn = self.tki_count(nbuf, lane, bt_cols, bs_shift)
                self.publish_count(base, lane, qi + 1, nn)
                self.tconv_row(nbuf, lane, row + self.grid, nn, req_ids, btab, bt_stride,
                               bt_cols, bs_shift, unit_rows)
                cute.arch.sync_warp()
                if lane == 0:
                    mbar_arrive(_bar(base, B_TKCONV + (qi + 1) % 2))
                    self.q_issue(base, row + self.grid, qi + 1, tm_qn, tm_qr)
                cute.arch.sync_warp()
            if qi + 2 < n_mine:
                mbar_wait(_bar(base, B_TKFREE + qi % 2), Uint32((qi // 2) & 1))
                fence_async_smem()  # gather warps' generic reads before the next bulk copy
                cute.arch.sync_warp()
                if lane == 0:
                    self.tki_issue(base, slots, sstride, row + 2 * self.grid, qi % 2)
                cute.arch.sync_warp()

    @cute.jit
    def publish_count(self, base, lane, qi, c):
        if lane == 0:
            st_shared_u32(base + Uint32(OFF_NCNT) + Uint32(4 * (qi % NCR)), Uint32(c))
            mbar_arrive(_bar(base, B_NFULL + qi % NCR))
        cute.arch.sync_warp()
        return c

    @cute.jit
    def slot_producer(self, base, lane, cta, n_mine, tm_qn, tm_qr, unit_rows, slots,
                      slot_stride, req_ids, btab, bt_stride, bt_cols, bs_shift):
        """Q warp, logical mode: the index remap. Lane l maps keys 2l, 2l + 1 of every
        64-key stage to physical rows (block_table[req][idx >> bs_shift] * unit_rows +
        idx % block size; -1 for invalid indices and keys >= the token's count) into
        OFF_SLOT[g % NSL] once every gather warp has read the entry's previous rows
        (B_SLEMPTY), then B_SLFULL. Over the stage
        sequence of the CTA's tokens, index loads run two stages and block-table loads
        one stage ahead, so the gather warps never wait on global memory; the next
        token's valid count (prefix_count) and Q follow the current token's last stage
        (B_NFULL ring of NCR: the producer is at most NSL + NS stages, so tokens, ahead of
        every reader)."""
        bmask = (Int32(1) << bs_shift) - Int32(1)
        sstride = Int64(slot_stride) * Int64(4)
        lo = Int64(8 * lane)
        # token 0: count, Q, first indices
        n = self.prefix_count(lane, slots + Int64(cta) * sstride, bt_cols, bs_shift)
        self.publish_count(base, lane, 0, n)
        if lane == 0:
            self.q_issue(base, cta, Int32(0), tm_qn, tm_qr)
        cute.arch.sync_warp()
        rq = ld_global_s32_pred(req_ids + Int64(cta) * Int64(4), Int32(1))
        btr = btab + Int64(rq) * Int64(bt_stride) * Int64(4)
        i0a, i0b = ld_global_v2_s32(slots + Int64(cta) * sstride + lo)
        i1a, i1b = ld_global_v2_s32(slots + Int64(cta) * sstride + lo + Int64(256))
        b0a, b0b = self.bt2(btr, i0a, i0b, bt_cols, bs_shift)
        g = Int32(0)
        for qi in range(n_mine):
            row = cta + qi * self.grid
            subs = _subs(n)
            srow = slots + Int64(row) * sstride + lo
            nrow = row
            if qi + 1 < n_mine:
                nrow = row + self.grid
            nsrow = slots + Int64(nrow) * sstride + lo
            rqn = ld_global_s32_pred(req_ids + Int64(nrow) * Int64(4), Int32(1))
            btrn = btab + Int64(rqn) * Int64(bt_stride) * Int64(4)
            if subs == 1:
                # the stage after this token's only one is the next token's first
                i1a, i1b = ld_global_v2_s32(nsrow)
            for t in cutlass.range(subs, unroll=1):
                s = g % NSL
                key = t * 64 + 2 * lane
                p0 = b0a * unit_rows + (i0a & bmask)
                p1 = b0b * unit_rows + (i0b & bmask)
                if b0a < 0 or key >= n:
                    p0 = Int32(-1)
                if b0b < 0 or key + 1 >= n:
                    p1 = Int32(-1)
                # stage t + 1: block-table rows (this token's or, after its last stage,
                # the next token's request); stage t + 2: indices
                if t + 1 < subs:
                    b0a, b0b = self.bt2(btr, i1a, i1b, bt_cols, bs_shift)
                else:
                    b0a, b0b = self.bt2(btrn, i1a, i1b, bt_cols, bs_shift)
                i0a, i0b = i1a, i1b
                if t + 2 < subs:
                    i1a, i1b = ld_global_v2_s32(srow + Int64((t + 2) * 256))
                else:
                    i1a, i1b = ld_global_v2_s32(nsrow + Int64((t + 2 - subs) * 256))
                compiler_fence()
                if lane == 0:
                    if cutlass.const_expr(self.xphys):
                        mbar_wait(_bar(base, B_KVEMPTY + g % NS), Uint32(((g // NS) & 1) ^ 1))
                    else:
                        mbar_wait(_bar(base, B_SLEMPTY + s), Uint32(((g // NSL) & 1) ^ 1))
                cute.arch.sync_warp()
                st_shared_v2(base + Uint32(OFF_SLOT) + Uint32(s) * Uint32(256) + Uint32(8 * lane),
                             p0, p1)
                cute.arch.sync_warp()
                if lane == 0:
                    mbar_arrive(_bar(base, B_SLFULL + s))
                g = g + 1
            if qi + 1 < n_mine:
                n = self.prefix_count(lane, slots + Int64(nrow) * sstride, bt_cols,
                                      bs_shift)
                self.publish_count(base, lane, qi + 1, n)
                if lane == 0:
                    self.q_issue(base, nrow, qi + 1, tm_qn, tm_qr)
                cute.arch.sync_warp()
            btr = btrn

    @cute.jit
    def gather_logical(self, base, pw, lane, cta, n_mine, q, act, part, tm_kv, tm_kr, lens):
        """Gather warps, logical mode: the stage's physical rows come from the slot
        producer's ring (OFF_SLOT, B_SLFULL); each warp releases the entry (B_SLEMPTY)
        once its lanes hold their rows."""
        npart = self.parts
        g = Int32(0)
        for qi in range(n_mine):
            subs = _subs(self.ntok(base, lens, cta + qi * self.grid, qi))
            for t in cutlass.range(subs, unroll=1):
                s = g % NS
                st = base + Uint32(OFF_KV) + Uint32(s) * Uint32(STAGE64)
                full = _bar(base, B_KVFULL + s)
                sl = g % NSL
                mbar_wait(_bar(base, B_SLFULL + sl), Uint32((g // NSL) & 1))  # every lane
                if lane == 0:
                    mbar_wait(_bar(base, B_KVEMPTY + s), Uint32(((g // NS) & 1) ^ 1))
                    if pw == 0:
                        self.ss(base, cta, lane, g, 22, 8)
                        if cutlass.const_expr(self.no_gather):
                            mbar_arrive(full)
                        else:
                            mbar_arrive_tx(full, Uint32(STAGE64 - self._skipped()))
                cute.arch.sync_warp()
                v = ld_shared_v4u(base + Uint32(OFF_SLOT) + Uint32(sl) * Uint32(256)
                                  + Uint32(q) * Uint32(16))
                r0, r1, r2, r3 = Int32(v[0]), Int32(v[1]), Int32(v[2]), Int32(v[3])
                cute.arch.sync_warp()
                if lane == 0:  # the rows are in registers: the producer may refill the entry
                    mbar_arrive(_bar(base, B_SLEMPTY + sl))
                if cutlass.const_expr(not self.no_gather):
                    if act:
                        for k in cutlass.range_constexpr((5 + npart - 1) // npart):
                            a = part + npart * k
                            if a < self._tma_ops():
                                dst = st + Uint32(a * 8192) + Uint32(q) * Uint32(512)
                                desc = tm_kv
                                col = 128 * a
                                if a == 4:
                                    dst = st + Uint32(4 * 8192) + Uint32(q) * Uint32(256)
                                    desc = tm_kr
                                    col = Int32(D_NOPE)
                                gather4(dst, desc, full, col, r0, r1, r2, r3)
                if pw == 0:
                    self.ss(base, cta, lane, g, 23, 8)
                g = g + 1

    @cute.jit
    def ntok(self, base, lens, row, qi):
        """Valid top-k count of token ``row`` (the CTA's token qi): from ``lengths``,
        or in logical mode from the Q warp's count in shared memory."""
        n = Int32(0)
        if cutlass.const_expr(self.logical and not self.lglob):
            mbar_wait(_bar(base, B_NFULL + qi % NCR), Uint32((qi // NCR) & 1))
            n = Int32(ld_shared_u32(base + Uint32(OFF_NCNT) + Uint32(4 * (qi % NCR))))
        else:
            n = lens[row]
        return n

    # ------------------------------------------------------------------ MMA
    @cute.jit
    def qk_loop(self, base, lane, taddr, n_mine, lengths, cta):
        """QK(u) as soon as S(u - 1) has been loaded and stage u has landed."""
        lptr = cute.make_ptr(Int32, lengths, cute.AddressSpace.gmem, assumed_align=4)
        lens = cute.make_tensor(lptr, cute.make_layout(1 << 30))
        u = Int32(0)
        for qi in range(n_mine):
            subs = _subs(self.ntok(base, lens, cta + qi * self.grid, qi))
            # the next token's latent Q: staging -> TMEM by tcgen05.cp from this warp, once
            # the previous token's QK MMAs (which read C_Q) have completed; cp -> mma
            # is ordered in the tensor pipe
            mbar_wait(_bar(base, B_QFULL + qi % 2), Uint32((qi // 2) & 1))
            if qi >= 1:
                mbar_wait(_bar(base, B_QREMPTY + (qi - 1) % 2), Uint32(((qi - 1) // 2) & 1))
            tc_fence_after()
            qd = Uint64(SDESC_SW128) | Uint64((base + Uint32(OFF_QN)) >> Uint32(4))
            for j in cutlass.range_constexpr(4):
                for c in cutlass.range_constexpr(4):
                    cp_e(taddr + Uint32(C_Q + 32 * j + 8 * c), qd + Uint64(j * ATOM // 16 + 2 * c))
            commit_e(_bar(base, B_QLFREE))
            if cutlass.const_expr(self.qz):
                # the copy has landed (QLFREE), then zero this warp's lanes 96-127 of Q
                mbar_wait(_bar(base, B_QLFREE), Uint32(qi & 1))
                tc_fence_after()
                tmem_zero128(taddr + Uint32(96 << 16) + Uint32(C_Q))
                tc_fence_before()
                cute.arch.sync_warp()
                tc_fence_after()
            for i in range(subs):
                if not self.qkh:
                    if u >= 1:
                        if cutlass.const_expr(self.nbs):
                            named_bar(Uint32(NB_SFREE), Uint32(32 + 128))
                        else:
                            mbar_wait(_bar(base, B_SFREE + (u - 1) % 2),
                                      Uint32(((u - 1) // 2) & 1))
                        tc_fence_after()
                self.ss(base, cta, lane, u, 24, 2)
                self.qk(base, lane, taddr, u, qi, i == subs - 1, cta)
                u = u + 1
        if cutlass.const_expr(self.nbs and not self.qkh):
            if u >= 1:  # the last subtile's S-free generation
                named_bar(Uint32(NB_SFREE), Uint32(32 + 128))

    @cute.jit
    def qk(self, base, lane, taddr, u, qi, last, cta):
        s = u % NS
        st = base + Uint32(OFF_KV) + Uint32(s) * Uint32(STAGE64)
        self.ss(base, cta, lane, u, 16, 2)
        mbar_wait(_bar(base, B_KVFULL + s), Uint32((u // NS) & 1))
        self.ss(base, cta, lane, u, 20, 2)
        if cutlass.const_expr(self.cpa in (1, 2, 3, 6)):
            fence_async_smem()  # RoPE chunks written by cp.async (generic proxy)
        tc_fence_after()
        sc = taddr + Uint32(C_S)
        qs = base + Uint32(OFF_QS) + Uint32(qi % 2) * Uint32(QS_BYTES)
        bd0 = Uint64(SDESC_SW128) | Uint64(st >> Uint32(4))
        if cutlass.const_expr(self.qkh):
            self.qk_halves(base, taddr, u, qi, last, sc, qs, bd0)
        else:
            for j in cutlass.range_constexpr(4):
                bd = bd0 + Uint64(j * 512)
                for k2 in cutlass.range_constexpr(2):
                    if cutlass.const_expr(not self.no_mma):
                        mma_ts_e(sc, taddr + Uint32(C_Q + 32 * j + 16 * k2),
                                 bd + Uint64(k2 * 4), Uint32(IDESC_QK),
                                 Uint32(1 if (j + k2) > 0 else 0))
            ad = Uint64(SDESC_SW64) | Uint64(qs >> Uint32(4))
            bd = (bd0 + Uint64(4 * 512)) ^ Uint64(SDESC_SW128 ^ SDESC_SW64)
            if cutlass.const_expr(not self.no_mma):
                mma_ss_e(sc, ad, bd, Uint32(IDESC_QK), Uint32(1))
            self.ss(base, cta, lane, u, 17, 2)
            commit_e(_bar(base, B_SFULL + u % 2))
            if last:
                commit_e(_bar(base, B_QREMPTY + qi % 2))
            if cutlass.const_expr(self.pvd):
                if lane == 0:
                    mbar_arrive(_bar(base, B_QKI + u % 8))

    @cute.jit
    def qk_halves(self, base, taddr, u, qi, last, sc, qs, bd0):
        """QK(u) as two N = 32 halves (keys 0-31 -> S columns 0-31, keys 32-63 -> 32-63),
        half h issued once the softmax has loaded half h of S(u - 1)."""
        idesc = (1 << 4) | (32 >> 3 << 17) | (128 >> 4 << 24) | (1 << 29)
        for hf in cutlass.range_constexpr(2):
            if u >= 1:
                fb = B_SFREE if hf == 0 else B_SFREE1
                mbar_wait(_bar(base, fb + (u - 1) % 2), Uint32(((u - 1) // 2) & 1))
                tc_fence_after()
            sch = sc + Uint32(32 * hf)
            for j in cutlass.range_constexpr(4):
                bd = bd0 + Uint64(j * 512 + 256 * hf)
                for k2 in cutlass.range_constexpr(2):
                    if cutlass.const_expr(not self.no_mma):
                        mma_ts_e(sch, taddr + Uint32(C_Q + 32 * j + 16 * k2),
                                 bd + Uint64(k2 * 4), Uint32(idesc),
                                 Uint32(1 if (j + k2) > 0 else 0))
            ad = Uint64(SDESC_SW64) | Uint64(qs >> Uint32(4))
            bd = (bd0 + Uint64(4 * 512 + 128 * hf)) ^ Uint64(SDESC_SW128 ^ SDESC_SW64)
            if cutlass.const_expr(not self.no_mma):
                mma_ss_e(sch, ad, bd, Uint32(idesc), Uint32(1))
            commit_e(_bar(base, (B_SFULL if hf == 0 else B_SFULL1) + u % 2))
        if last:
            commit_e(_bar(base, B_QREMPTY + qi % 2))

    @cute.jit
    def pv_loop(self, base, lane, taddr, n_mine, lengths, cta):
        """PV(u) once the owning softmax warpgroup released it (P written, O^T
        rescaled if needed); P^T is fetched once into the B collector."""
        lptr = cute.make_ptr(Int32, lengths, cute.AddressSpace.gmem, assumed_align=4)
        lens = cute.make_tensor(lptr, cute.make_layout(1 << 30))
        u = Int32(0)
        for qi in range(n_mine):
            subs = _subs(self.ntok(base, lens, cta + qi * self.grid, qi))
            for i in range(subs):
                s = u % NS
                st = base + Uint32(OFF_KV) + Uint32(s) * Uint32(STAGE64)
                self.ss(base, cta, lane, u, 7, 4)
                if cutlass.const_expr(self.pvd):
                    if i + self.pvd < subs:
                        v = u + self.pvd
                        mbar_wait(_bar(base, B_QKI + v % 8), Uint32((v // 8) & 1))
                self.ss(base, cta, lane, u, 15, 4)
                mbar_wait(_bar(base, B_OREADY + u % NP), Uint32((u // NP) & 1))
                self.ss(base, cta, lane, u, 18, 4)
                if cutlass.const_expr(self.cpa in (2, 3, 6)):
                    fence_async_smem()  # V chunks written by cp.async (generic proxy)
                tc_fence_after()
                pbuf = base + Uint32(OFF_P) + Uint32(u % NP) * Uint32(P_BYTES)
                bd = Uint64(SDESC_SW64) | Uint64(pbuf >> Uint32(4))
                ad0 = Uint64(SDESC_SW128) | Uint64(st >> Uint32(4))
                for b in cutlass.range_constexpr(4):
                    if i == 0:
                        mbar_wait(_bar(base, B_OFREE + b), Uint32((qi & 1) ^ 1))
                        tc_fence_after()
                    acc = Uint32(1)
                    if i == 0:
                        acc = Uint32(0)
                    op = "fill" if b == 0 else ("lastuse" if b == 3 else "use")
                    if cutlass.const_expr(not self.no_mma):
                        mma_cb_e(taddr + Uint32(C_O + 96 * b), ad0 + Uint64(b * 512), bd,
                                 Uint32(IDESC_PV), acc, op)
                self.ss(base, cta, lane, u, 19, 4)
                commit_e(_bar(base, B_KVEMPTY + s))
                commit_e(_bar(base, B_PEMPTY + u % NP))
                if i == subs - 1:
                    commit_e(_bar(base, B_OFULL))
                u = u + 1

    # ------------------------------------------------------------------ softmax
    @cute.jit
    def softmax(self, base, tid, warp, lane, taddr, n_mine, lengths, cta, scale_log2,
                out_scale, tm_o):
        lptr = cute.make_ptr(Int32, lengths, cute.AddressSpace.gmem, assumed_align=4)
        lens = cute.make_tensor(lptr, cute.make_layout(1 << 30))
        wg = warp // 4
        sp = warp % 4
        h = tid % 128  # TMEM lane: head (S, Q) / dim (O^T)
        lane_addr = taddr + (Uint32(sp * 32) << Uint32(16))
        real = h < H
        real_m = h < (0 if self.no_math else H)
        xv = cute.make_rmem_tensor((64,), Float32)
        u = Int32(0)
        for qi in range(n_mine):
            row = cta + qi * self.grid
            n = self.ntok(base, lens, row, qi)
            more_tok = Int32(0)
            if qi + 1 < n_mine:
                more_tok = Int32(1)
            subs = _subs(n)
            mp_last = Float32(-1e30)  # m - P_PRE that this WG's l is relative to
            lw = Float32(0.0)
            for i in range(subs):
                if u % 2 == wg:
                    full_last = Int32(0)
                    if cutlass.const_expr(self.msk):
                        if i == subs - 1 and n % 64 == 0:
                            full_last = Int32(1)
                    if full_last != 0:
                        mp_last, lw = self.sub(base, warp, lane, wg, sp, h, lane_addr, real_m,
                                               xv, scale_log2, u, i, n, more_tok, mp_last,
                                               lw, cta, False)
                    elif i == subs - 1:
                        mp_last, lw = self.sub(base, warp, lane, wg, sp, h, lane_addr, real_m,
                                               xv, scale_log2, u, i, n, more_tok, mp_last,
                                               lw, cta, True)
                    else:
                        mp_last, lw = self.sub(base, warp, lane, wg, sp, h, lane_addr, real_m,
                                               xv, scale_log2, u, i, n, Int32(1), mp_last,
                                               lw, cta, False)
                u = u + 1
            # ---- token end. The WG owning the last subtile (u - 1) combines
            # l = l0 2^(m0 - m) + l1 2^(m1 - m), m = max(m0, m1) and runs the epilogue (in
            # the Q staging, once the next token's Q is in TMEM); the other WG hands over
            # (l, m) and goes on.
            owner = (u - 1) % 2 == wg
            tok = base + Uint32(OFF_TOK0)
            if qi % 2 == 1:
                tok = base + Uint32(OFF_TOK1)
            # token-parity barriers (7-9 / 12-14) like the l/m buffers: with one set, the
            # non-owner's arrive for token qi and its own sync as owner of a short token
            # qi + 1 could complete one generation alone, leaving the token-qi owner
            # waiting forever (two 1-subtile tokens per CTA hung, P-MLA runs 121-123)
            xbar = Uint32(7 + sp) + Uint32(5) * Uint32(qi % 2)
            if owner:
                self.ss(base, cta, lane, u - 1, 21, 1)
            if real:
                st_shared_f32(tok + Uint32((wg * 2 * H + h) * 4), lw)
                st_shared_f32(tok + Uint32((wg * 2 * H + H + h) * 4), mp_last)
                if owner:
                    named_bar(xbar, 64)
                    self.ss(base, cta, lane, u - 1, 22, 1)
                    o = 1 - wg
                    l1 = ld_shared_f32(tok + Uint32((o * 2 * H + h) * 4))
                    m1 = ld_shared_f32(tok + Uint32((o * 2 * H + H + h) * 4))
                    mf = cute.arch.fmax(mp_last, m1)
                    lt = lw * ex2(mp_last - mf) + l1 * ex2(m1 - mf)
                    iv = Float32(0.0)
                    if lt > Float32(0.0):
                        iv = out_scale / lt
                    st_shared_f32(base + Uint32(OFF_INV + h * 4), iv)
                else:
                    named_bar_arrive(xbar, 64)
            if owner:
                if qi + 1 < n_mine:
                    mbar_wait(_bar(base, B_QLFREE), Uint32((qi + 1) & 1))
                elif cutlass.const_expr(not self.epw):
                    # Last token (non-epw): no QLFREE(qi + 1) orders this epilogue after the
                    # previous token's stores (the other WG may own it, and bulk_wait_read is
                    # per issuing thread), so wait for its B_EPIDONE: completion qi - 1, parity
                    # (qi - 1) & 1 (completion qi - 2 is behind the Q warp's wait; qi is ours).
                    # epw orders every staging half through B_EPIFREE instead.
                    if qi > 0:
                        mbar_wait(_bar(base, B_EPIDONE), Uint32((qi - 1) & 1))
                self.ss(base, cta, lane, u - 1, 23, 1)
                self.epilogue(base, wg, sp, lane, lane_addr, qi, cta, tm_o, u - 1)

    @cute.jit
    def sub(self, base, warp, lane, wg, sp, h, lane_addr, real_m, xv, scale_log2, u, i, n,
            more, mp_last, lw, cta, masked: cutlass.Constexpr):
        """Subtile u (subtile i of its token) on WG u % 2. Returns (m, l) of this WG."""
        self.ss(base, cta, lane, u, 8 * wg + 0, 1)
        s_addr = lane_addr + Uint32(C_S)
        smax = Float32(0.0)
        if cutlass.const_expr(self.qkh):
            for hf in cutlass.range_constexpr(2):
                mbar_wait(_bar(base, (B_SFULL if hf == 0 else B_SFULL1) + u % 2),
                          Uint32((u // 2) & 1))
                tc_fence_after()
                sm = tmem_ld_max32(s_addr + Uint32(32 * hf))
                for k in cutlass.range_constexpr(32):
                    xv[32 * hf + k] = sm[k]
                if hf == 0:
                    smax = sm[32]
                else:
                    smax = cute.arch.fmax(smax, sm[32])
                tc_fence_before()
                cute.arch.sync_warp()
                if lane == 0:
                    mbar_arrive(_bar(base, (B_SFREE if hf == 0 else B_SFREE1) + u % 2))
        else:
            mbar_wait(_bar(base, B_SFULL + u % 2), Uint32((u // 2) & 1))
            self.ss(base, cta, lane, u, 8 * wg + 1, 1)
            tc_fence_after()
            if cutlass.const_expr(masked):
                sa = tmem_ld(s_addr, 64)
            else:
                sm = tmem_ld_max64(s_addr)
                sa = sm[:64]
                smax = sm[64]
            for k in cutlass.range_constexpr(64):
                xv[k] = sa[k]
            self.ss(base, cta, lane, u, 25, 1)
            tc_fence_before()
            if cutlass.const_expr(self.nbs):
                named_bar_arrive(Uint32(NB_SFREE), Uint32(32 + 128))
            else:
                cute.arch.sync_warp()
                if lane == 0:
                    mbar_arrive(_bar(base, B_SFREE + u % 2))
        self.ss(base, cta, lane, u, 8 * wg + 2, 1)
        mbx = base + Uint32(OFF_MBOX)
        grow = Int32(0)
        alpha = Float32(1.0)
        mp = Float32(0.0)
        if real_m:
            if cutlass.const_expr(masked):
                for k in cutlass.range_constexpr(64):
                    if 64 * i + k >= n:
                        xv[k] = Float32(-math.inf)
                xm = _tree_max([xv[k] for k in range(64)]) * scale_log2
            else:
                xm = smax * scale_log2
            # lazy-max decision after the previous subtile's (other WG) decision
            if u >= 1:
                named_bar(Uint32(1 + sp + 3 * (1 - wg)), 64)
            mprev = ld_shared_f32(mbx + Uint32(((u + 1) % 2) * H * 4 + h * 4))
            if i == 0:
                if xm > Float32(-1e30):
                    mp = xm - Float32(P_PRE)
            else:
                mp = mprev
                if xm - mprev > Float32(P_PRE + THR):
                    grow = Int32(1)
                    mp = xm - Float32(P_PRE)
                    alpha = ex2(mprev - mp)
            st_shared_f32(mbx + Uint32((u % 2) * H * 4 + h * 4), mp)
            if more != 0:  # a subtile follows (this token's or the next token's)
                named_bar_arrive(Uint32(1 + sp + 3 * wg), 64)
            if mp != mp_last:
                lw = lw * ex2(mp_last - mp)
                mp_last = mp
        # publish the heads that grew (ballot masks, alphas) and OR them over the WG
        # here, where the warps are already in step; the rescale runs after P
        par = Uint32(wg * 2) + Uint32((u // 2) % 2)
        gm = base + Uint32(OFF_GMASK) + par * Uint32(16)
        alb = base + Uint32(OFF_AL) + par * Uint32(H * 4)
        gw = ballot(Uint32(grow))
        if lane == 0 and sp < 3:
            st_shared_u32(gm + Uint32(4 * sp), gw)
        if grow != 0:
            st_shared_f32(alb + Uint32(h * 4), alpha)
        anyg = bar_red_or(Uint32(10 + wg), Uint32(128), Uint32(grow))
        self.ss(base, cta, lane, u, 8 * wg + 3, 1)
        if u >= NP:
            mbar_wait(_bar(base, B_PEMPTY + u % NP), Uint32((u // NP - 1) & 1))
        self.ss(base, cta, lane, u, 8 * wg + 4, 1)
        if real_m:
            pbuf = base + Uint32(OFF_P) + Uint32(u % NP) * Uint32(P_BYTES)
            a0, a1 = Float32(0.0), Float32(0.0)
            b0, b1 = Float32(0.0), Float32(0.0)
            nmp = Float32(0.0) - mp
            for k in cutlass.range_constexpr(32):
                xv[2 * k], xv[2 * k + 1] = ffma2(xv[2 * k], xv[2 * k + 1], scale_log2,
                                                 scale_log2, nmp, nmp)
            for ch in cutlass.range_constexpr(4):
                w = []
                for q in cutlass.range_constexpr(4):
                    k = 16 * ch + 4 * q
                    if cutlass.const_expr(self.nex):  # timing only: FMUL instead of MUFU
                        h0, z0 = Float32(0.5), Float32(-0.0)
                        p0, p1 = ffma2(xv[k], xv[k + 1], h0, h0, z0, z0)
                        p2, p3 = ffma2(xv[k + 2], xv[k + 3], h0, h0, z0, z0)
                    else:
                        p0 = ex2(xv[k])
                        p1 = ex2(xv[k + 1])
                        p2 = ex2(xv[k + 2])
                        p3 = ex2(xv[k + 3])
                    a0, a1 = fadd2(a0, a1, p0, p1)
                    b0, b1 = fadd2(b0, b1, p2, p3)
                    w.append(cvt_e4m3x2(p0, p1) | (cvt_e4m3x2(p2, p3) << Uint32(16)))
                st_shared_v4(pbuf + Uint32(h * 64) + (
                    (Uint32(ch) ^ ((Uint32(h) >> Uint32(1)) & Uint32(3))) << Uint32(4)),
                    w[0], w[1], w[2], w[3])
            a0, a1 = fadd2(a0, a1, b0, b1)
            lw = lw + (a0 + a1)
        self.ss(base, cta, lane, u, 8 * wg + 5, 1)
        # heads of this subtile that grew: rescale their O^T columns (one per block)
        # after PV(u - 1), before PV(u); every other column has alpha = 1
        if anyg != Uint32(0):
            mbar_wait(_bar(base, B_PEMPTY + (u - 1) % NP), Uint32(((u - 1) // NP) & 1))
            tc_fence_after()
            for w in cutlass.range_constexpr(3):
                m = ld_shared_u32(gm + Uint32(4 * w))
                while m != Uint32(0):
                    j = bfind(m)
                    m = m ^ (Uint32(1) << j)
                    hh = Uint32(32 * w) + j
                    al = ld_shared_f32(alb + hh * Uint32(4))
                    col = lane_addr + Uint32(C_O) + hh
                    o0, o1, o2, o3 = tmem_ld_o4(col)
                    tmem_st_o4(col, o0 * al, o1 * al, o2 * al, o3 * al)
            tc_fence_before()
        fence_async_smem()
        cute.arch.sync_warp()
        self.ss(base, cta, lane, u, 8 * wg + 6, 1)
        if lane == 0:
            mbar_arrive(_bar(base, B_OREADY + u % NP))
        return mp_last, lw

    @cute.jit
    def epilogue(self, base, wg, sp, lane, lane_addr, qi, cta, tm_o, ul):
        """O^T block b (lanes = dims, columns = heads) -> BF16 [heads x dims]: 16x256b
        TMEM loads give each thread two adjacent heads of one dim (an m8n8 b16
        fragment with dims as rows), scaled by s / l and packed; stmatrix.trans writes
        head rows into two SW128 [96 x 64 dims] staging halves (the Q latent atoms 0-1
        staging), one TMA store each. Block b is released once read."""
        row = cta + qi * self.grid
        mbar_wait(_bar(base, B_OFULL), Uint32(qi & 1))
        self.ss(base, cta, lane, ul, 26, 1)
        tc_fence_after()
        bid = Uint32(10 + wg)
        named_bar(bid, 128)  # 1/l of every head written
        self.ss(base, cta, lane, ul, 27, 1)
        t4 = lane % 4
        j = lane % 8
        mk = lane // 8  # matrix this lane addresses in stmatrix.x4
        inv_base = base + Uint32(OFF_INV)
        # epb: 1/l of this thread's 24 heads loaded once (the same for every block)
        invs = []
        if cutlass.const_expr(self.epb):
            for c in cutlass.range_constexpr(3):
                for g in cutlass.range_constexpr(4):
                    invs.append(ld_shared_v2f(inv_base + Uint32((32 * c + 8 * g + 2 * t4) * 4)))
        for b in cutlass.range_constexpr(4):
            # staging halves alternate between Q atoms 0-1 and 2-3 (block b % 2): block
            # b reuses block b - 2's buffer once its two stores have read it
            if cutlass.const_expr(self.epw):
                # The store warp releases each use of a half (B_EPIFREE, 2 completions per
                # token): block b waits for the previous use's release, completion 2 qi
                # (block b - 2, parity 0) or, for b < 2, 2 qi - 1 (block b + 2 of token
                # qi - 1, parity 1; none for the CTA's first token). Every use waits, so
                # the producer is never two completions ahead on B_EPIFULL / B_EPIFREE and
                # both parity waits are exact. (Waiting only for b >= 2 let a token's b < 2
                # blocks overwrite a half before its stores read it and, two completions
                # ahead, hang epi_store on an aliased parity: the startup stall.)
                if cutlass.const_expr(b >= 2):
                    mbar_wait(_bar(base, B_EPIFREE + b % 2), Uint32(0))
                elif qi > 0:
                    mbar_wait(_bar(base, B_EPIFREE + b % 2), Uint32(1))
            elif cutlass.const_expr(b >= 2):
                if sp == 0 and lane == 0:
                    bulk_wait_read2()
                named_bar(bid, 128)
            for hh in cutlass.range_constexpr(2):
                d0 = 32 * sp + 16 * hh  # first dim (lane) of this 16-lane half
                ha = lane_addr + Uint32(16 * hh << 16) + Uint32(C_O + 96 * b)
                if cutlass.const_expr(self.epb):
                    vh = tmem_ld16x256x3(ha, ha + Uint32(32), ha + Uint32(64))
                for c in cutlass.range_constexpr(3):
                    if cutlass.const_expr(self.epb):
                        v = vh[16 * c:16 * c + 16]
                    else:
                        v = tmem_ld16x256(ha + Uint32(32 * c))
                    if cutlass.const_expr(hh == 1 and c == 2):
                        tc_fence_before()
                        cute.arch.sync_warp()
                        if lane == 0:
                            mbar_arrive(_bar(base, B_OFREE + b))
                    frag = []
                    for g in cutlass.range_constexpr(4):
                        if cutlass.const_expr(self.epb):
                            # packed products (+ -0.0: exactly v * i, signed zeros included)
                            i0, i1 = invs[4 * c + g]
                            p0, p1 = ffma2(v[4 * g], v[4 * g + 1], i0, i1, Float32(-0.0),
                                           Float32(-0.0))
                            p2, p3 = ffma2(v[4 * g + 2], v[4 * g + 3], i0, i1, Float32(-0.0),
                                           Float32(-0.0))
                            frag.append(bf16x2(p0, p1))
                            frag.append(bf16x2(p2, p3))
                        else:
                            i0, i1 = ld_shared_v2f(inv_base + Uint32((32 * c + 8 * g + 2 * t4) * 4))
                            frag.append(bf16x2(v[4 * g] * i0, v[4 * g + 1] * i1))
                            frag.append(bf16x2(v[4 * g + 2] * i0, v[4 * g + 3] * i1))
                    for sx in cutlass.range_constexpr(2):
                        g = 2 * sx + mk // 2
                        dd = d0 + 8 * (mk % 2)  # first dim of the matrix (0-127)
                        hrow = 32 * c + 8 * g + j  # stored row = head
                        half = base + Uint32(OFF_QN + (b % 2) * 2 * ATOM) + Uint32(
                            dd // 64) * Uint32(H * 128)
                        addr = half + Uint32(hrow * 128) + (
                            ((Uint32((dd % 64) // 8)) ^ (Uint32(hrow) & Uint32(7))) << Uint32(4))
                        stmatrix_t4(addr, frag[4 * sx], frag[4 * sx + 1], frag[4 * sx + 2],
                                    frag[4 * sx + 3])
            fence_async_smem()
            named_bar(bid, 128)
            self.ss(base, cta, lane, ul, 28 + b, 1)
            if sp == 0 and lane == 0:
                if cutlass.const_expr(self.epw):
                    mbar_arrive(_bar(base, B_EPIFULL + b % 2))
                elif cutlass.const_expr(not self.no_store):
                    for hf in cutlass.range_constexpr(2):
                        tma_store2d(tm_o, base + Uint32(OFF_QN + (b % 2) * 2 * ATOM + hf * H * 128),
                                    Int32(128 * b + 64 * hf), row * H)
        if cutlass.const_expr(not self.epw):
            if sp == 0 and lane == 0:
                bulk_wait_read0()
                mbar_arrive(_bar(base, B_EPIDONE))

    @cute.jit
    def epi_store(self, base, lane, cta, n_mine, tm_o):
        """epw: per token, block b's staging half b % 2 -> two TMA stores once the owner
        warpgroup wrote it (B_EPIFULL); the half is released (B_EPIFREE) and, after block 3,
        the staging (B_EPIDONE: the Q warp's next Q load) once the stores have read it."""
        for qi in range(n_mine):
            row = cta + qi * self.grid
            for b in cutlass.range_constexpr(4):
                if cutlass.const_expr(self.ewslow):
                    nanosleep(self.ewslow)
                mbar_wait(_bar(base, B_EPIFULL + b % 2), Uint32(b // 2))
                if lane == 0:
                    if cutlass.const_expr(not self.no_store):
                        for hf in cutlass.range_constexpr(2):
                            tma_store2d(tm_o, base + Uint32(OFF_QN + (b % 2) * 2 * ATOM
                                                            + hf * H * 128),
                                        Int32(128 * b + 64 * hf), row * H)
                    bulk_wait_read0()
                    mbar_arrive(_bar(base, B_EPIFREE + b % 2))
                    if cutlass.const_expr(b == 3):
                        mbar_arrive(_bar(base, B_EPIDONE))
                cute.arch.sync_warp()


@cache
def _compiled(grid: int, knobs: tuple = (), topk: int = 0):
    kernel = SparseMlaPrefillH8(grid, knobs, topk)
    args = (Int64(0), Int32(1), Int64(0), Int32(64), Int64(0), Int32(1), Int64(0), Int64(0),
            Float32(1.0), Float32(1.0), Int32(64), Int64(0), Int64(0), Int64(0), Int32(1),
            Int32(1), Int32(6), cuda.CUstream(0))
    # register budgets come from setmaxnreg (.reqntid 640 -> 96 at launch)
    return cute.compile(kernel, *args)


def sparse_mla_prefill_h8(q8: torch.Tensor, kv_rows: torch.Tensor, slots: torch.Tensor,
                          lengths: torch.Tensor, scale: float, out_scale: float,
                          unit_rows: int = 64, out: torch.Tensor | None = None,
                          stamps: torch.Tensor | None = None,
                          knobs: tuple = ()) -> torch.Tensor:
    """Same contract as ``sparse_mla_sol.sparse_mla_prefill_sol``: q8 [T, 96, 576]
    E4M3 (latent 512 + RoPE 64), kv_rows [rows, 576] E4M3, slots [T, >= top-k] int32
    (paged slot = page * unit_rows + offset, < 0 or >= lengths[t] masked), out [T, 96,
    512] BF16 = softmax(scale q k^T) v * out_scale. stamps (int32 [256]) only with the
    sstamp knob."""
    t = q8.shape[0]
    if q8.shape[1:] != (H, ROW) or not q8.is_contiguous():
        raise ValueError(f"sparse_mla_prefill_h8: q8 must be contiguous [T, {H}, {ROW}]")
    if out is None:
        out = torch.empty((t, H, D_NOPE), dtype=torch.bfloat16, device=q8.device)
    if not out.is_contiguous() or out.shape != (t, H, D_NOPE):
        raise ValueError("sparse_mla_prefill_h8: out must be contiguous [T, 96, 512]")
    if t == 0:
        return out
    sms = torch.cuda.get_device_properties(q8.device).multi_processor_count
    grid = min(sms, t)
    fn = _compiled(grid, tuple(knobs))
    stream = cuda.CUstream(torch.cuda.current_stream(q8.device).cuda_stream)
    fn(Int64(q8.data_ptr()), Int32(t), Int64(kv_rows.data_ptr()), Int32(unit_rows),
       Int64(slots.data_ptr()), Int32(slots.stride(0)), Int64(lengths.data_ptr()),
       Int64(out.data_ptr()), Float32(scale * LOG2E), Float32(out_scale),
       Int32(kv_rows.shape[0]), Int64(0 if stamps is None else stamps.data_ptr()),
       Int64(0), Int64(0), Int32(1), Int32(1), Int32(6), stream)
    return out


def sparse_mla_prefill_h8_logical(q8: torch.Tensor, kv_rows: torch.Tensor,
                                  topk_indices: torch.Tensor, req_ids: torch.Tensor,
                                  block_table: torch.Tensor, block_size: int,
                                  block_stride_rows: int, scale: float, out_scale: float,
                                  out: torch.Tensor | None = None, knobs: tuple = (),
                                  lengths: torch.Tensor | None = None,
                                  phys_dbg: torch.Tensor | None = None) -> torch.Tensor:
    """``sparse_mla_prefill_h8`` on the indexer's request-relative top-k, with the
    index remap (``triton_convert_req_index_to_global_index``) fused into the gather:
    topk_indices [T, K] int32 (K % 64 == 0, 128 <= K <= 2048, < 0 invalid, valid
    entries first),
    req_ids [T] int32 (block-table row of each token), block_table [R, B] int32,
    kv_rows [rows, 576] with block b at rows [b * block_stride_rows, + block_size)."""
    t = q8.shape[0]
    if q8.shape[1:] != (H, ROW) or not q8.is_contiguous():
        raise ValueError(f"sparse_mla_prefill_h8: q8 must be contiguous [T, {H}, {ROW}]")
    k = topk_indices.shape[1]
    if (topk_indices.dtype != torch.int32 or topk_indices.stride(1) != 1 or k % 64
            or topk_indices.stride(0) % 4 or topk_indices.data_ptr() % 16
            or not 128 <= k <= 2048
            or topk_indices.shape[0] < t):
        raise ValueError(f"sparse_mla_prefill_h8: bad top-k {tuple(topk_indices.shape)}")
    if req_ids.dtype != torch.int32 or not req_ids.is_contiguous() or req_ids.shape[0] < t:
        raise ValueError("sparse_mla_prefill_h8: req_ids must be contiguous int32 [>= T]")
    if block_table.dtype != torch.int32 or block_table.stride(1) != 1:
        raise ValueError("sparse_mla_prefill_h8: block_table must be int32, unit col stride")
    if block_size & (block_size - 1) or block_size > block_stride_rows:
        raise ValueError(f"sparse_mla_prefill_h8: block size {block_size} (stride "
                         f"{block_stride_rows}) must be a power of two within the stride")
    if out is None:
        out = torch.empty((t, H, D_NOPE), dtype=torch.bfloat16, device=q8.device)
    if not out.is_contiguous() or out.shape != (t, H, D_NOPE):
        raise ValueError("sparse_mla_prefill_h8: out must be contiguous [T, 96, 512]")
    if t == 0:
        return out
    sms = torch.cuda.get_device_properties(q8.device).multi_processor_count
    grid = min(sms, t)
    fn = _compiled(grid, tuple(knobs), k)
    stream = cuda.CUstream(torch.cuda.current_stream(q8.device).cuda_stream)
    fn(Int64(q8.data_ptr()), Int32(t), Int64(kv_rows.data_ptr()), Int32(block_stride_rows),
       Int64(topk_indices.data_ptr()), Int32(topk_indices.stride(0)),
       Int64(0 if lengths is None else lengths.data_ptr()),  # lglob diagnostic only
       Int64(out.data_ptr()), Float32(scale * LOG2E), Float32(out_scale),
       Int32(kv_rows.shape[0]), Int64(0 if phys_dbg is None else phys_dbg.data_ptr()),
       Int64(req_ids.data_ptr()),
       Int64(block_table.data_ptr()), Int32(block_table.stride(0)),
       Int32(block_table.shape[1]), Int32(block_size.bit_length() - 1), stream)
    return out
