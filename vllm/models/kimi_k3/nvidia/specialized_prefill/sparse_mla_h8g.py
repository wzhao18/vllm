# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 DSA prefill sparse MLA h8 with an LDGSTS row producer (sm_107).

Same math, pipeline, TMEM plan, barriers and shared-memory layout as ``sparse_mla_h8``
(physical-slot mode); only the KV ring producer differs. h8 fills a 64-key stage with
80 TMA tile::gather4 ops issued by 9 warps (per-lane operands), which caps the stage
rate at the per-SM gather4 issue rate. Here ``nlw`` warps fill it with 16 B
``cp.async.cg`` (LDGSTS) copies written to the TMA SWIZZLE_128B (latent atoms) /
SWIZZLE_64B (RoPE) positions. Copies are row-contiguous (a latent instruction moves 4
rows x 128 B, a RoPE instruction 8 rows x 64 B). The stage barrier keeps h8's single
``arrive.expect_tx(STAGE64)`` (warp 0, at stage start); every producer warp completes
its bytes with ``mbarrier.complete_tx`` once ``cp.async.wait_group`` shows its copies of
that stage done (``lag`` stages later) and its lanes fenced them into the async proxy.
Masked keys are zero-filled like the TMA out-of-bound fill: stage bytes equal h8's.
"""

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.primitives import asm
from vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla import (
    D_NOPE,
    H,
    LOG2E,
    ROW,
    fence_async_smem,
    ld_global_s32_pred,
    mbar_arrive,
    mbar_arrive_tx,
    mbar_wait,
)
from vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla_h8 import (
    B_KVEMPTY,
    B_KVFULL,
    B_NFULL,
    NCR,
    NLD,
    NS,
    OFF_KV,
    STAGE64,
    SparseMlaPrefillH8,
    _bar,
    _subs,
    compiler_fence,
    cp16,
)

ROPE_OFF = 4 * 8192  # RoPE SW64 block of a stage (after the 4 latent SW128 atoms)


@asm(Int32)
def shfl_idx(v, src):
    return "shfl.sync.idx.b32 $0, $1, $2, 31, -1;", "=r,r,r", (Int32(v), Int32(src))


@asm()
def cp_async_arrive_inc(bar):
    return "cp.async.mbarrier.arrive.shared::cta.b64 [$0];", "r,~{memory}", (Uint32(bar),)


@asm()
def cp_commit():
    return "cp.async.commit_group;", "~{memory}", ()


def _wait_group(n):
    @asm()
    def w():
        return f"cp.async.wait_group {n};", "~{memory}", ()

    return w


_WAITS = [_wait_group(n) for n in range(NS)]


@asm()
def complete_tx_release(bar, nbytes):
    """Release fence, then retire nbytes of the barrier's transaction count."""
    return ("{ fence.acq_rel.cta; "
            "mbarrier.complete_tx.relaxed.cta.shared::cta.b64 [$0], $1; }",
            "r,r,~{memory}", (Uint32(bar), Uint32(nbytes)))


class SparseMlaPrefillH8G(SparseMlaPrefillH8):
    def __init__(self, grid: int, knobs: tuple = (), nlw: int = 4, lag: int = 3):
        super().__init__(grid, knobs)
        if nlw not in (2, 4, 8) or not 0 <= lag < NS or (lag == 0 and nlw * NS > NCR):
            raise ValueError(
                f"SparseMlaPrefillH8G: nlw 2|4|8, lag < {NS} (lag 0: nlw <= 4); got {nlw}, {lag}")
        if self.sstamp or self.cpa or self.gskip:
            raise ValueError("SparseMlaPrefillH8G: sstamp/cpa/gskip are h8 gather4 knobs")
        self.nlw = nlw
        self.lag = lag
        # lag 0: per-warp stage barriers (the count-1 B_NFULL slots, unused in physical
        # mode): every lane's cp.async.mbarrier.arrive (no .noinc: self-balancing) then
        # one lane's arrive after __syncwarp; QK waits them, fences the generic-proxy
        # writes and completes h8's KV-full barrier itself.

    @cute.jit
    def loader(self, base, pw, lane, cta, n_mine, tm_qn, tm_qr, tm_kv, tm_kr, unit_rows,
               slots, slot_stride, lengths, kvp, req_ids, btab, bt_stride, bt_cols,
               bs_shift, stamp):
        if pw == NLD:
            SparseMlaPrefillH8.loader(self, base, pw, lane, cta, n_mine, tm_qn, tm_qr,
                                      tm_kv, tm_kr, unit_rows, slots, slot_stride, lengths,
                                      kvp, req_ids, btab, bt_stride, bt_cols, bs_shift,
                                      stamp)
        elif pw < self.nlw:
            lptr = cute.make_ptr(Int32, lengths, cute.AddressSpace.gmem, assumed_align=4)
            lens = cute.make_tensor(lptr, cute.make_layout(1 << 30))
            self.ldgsts_rows(base, pw, lane, cta, n_mine, lens, unit_rows, slots,
                             slot_stride, kvp)

    @cute.jit
    def ldgsts_rows(self, base, pw, lane, cta, n_mine, lens, unit_rows, slots,
                    slot_stride, kvp):
        # Warp pw owns rows [rpw pw, rpw (pw + 1)) of every stage (rpw = 64 / nlw). Lane
        # l < rpw loads the slot of row rpw pw + l two stages ahead and turns it into a
        # physical row (-1 masked); the copy lanes read it with shfl.
        rpw = 64 // self.nlw
        lag = self.lag
        own = pw * rpw + lane
        okl = Int32(lane < rpw)
        g = Int32(0)
        for qi in range(n_mine):
            row = cta + qi * self.grid
            n = lens[row]
            subs = _subs(n)
            sl = slots + Int64(row) * Int64(slot_stride) * Int64(4) + Int64(4 * own)
            a0 = ld_global_s32_pred(sl, okl)
            a1 = ld_global_s32_pred(sl + Int64(256), okl * Int32(subs > 1))
            for t in cutlass.range(subs, unroll=1):
                s = g % NS
                st = base + Uint32(OFF_KV) + Uint32(s) * Uint32(STAGE64)
                idx = a0
                a0 = a1
                a1 = ld_global_s32_pred(sl + Int64((t + 2) * 256),
                                        okl * Int32(t + 2 < subs))
                prow = Int32(-1)
                if idx >= 0 and t * 64 + own < n:
                    prow = (idx >> 6) * unit_rows + (idx & 63)
                compiler_fence()
                if lane == 0:
                    mbar_wait(_bar(base, B_KVEMPTY + s), Uint32(((g // NS) & 1) ^ 1))
                    if cutlass.const_expr(lag > 0):
                        if pw == 0:
                            mbar_arrive_tx(_bar(base, B_KVFULL + s), Uint32(STAGE64))
                cute.arch.sync_warp()
                r0 = pw * rpw
                c8 = Uint32(lane % 8)
                for i in cutlass.range_constexpr(rpw // 4):
                    rl = 4 * i + lane // 8
                    pr = shfl_idx(prow, rl)
                    src = kvp
                    nb = Uint32(0)
                    if pr >= 0:
                        src = kvp + Int64(pr) * Int64(ROW)
                        nb = Uint32(16)
                    r = Uint32(r0 + rl)
                    dl = st + r * Uint32(128) + ((c8 ^ (r & Uint32(7))) << Uint32(4))
                    sc = src + Int64(c8 * Uint32(16))
                    for a in cutlass.range_constexpr(4):
                        cp16(dl + Uint32(a * 8192), sc + Int64(a * 128), nb)
                c4 = Uint32(lane % 4)
                for i in cutlass.range_constexpr(rpw // 8):
                    rl = 8 * i + lane // 4
                    pr = shfl_idx(prow, rl)
                    src = kvp
                    nb = Uint32(0)
                    if pr >= 0:
                        src = kvp + Int64(pr) * Int64(ROW)
                        nb = Uint32(16)
                    r = Uint32(r0 + rl)
                    cp16(st + Uint32(ROPE_OFF) + r * Uint32(64)
                         + ((c4 ^ ((r >> Uint32(1)) & Uint32(3))) << Uint32(4)),
                         src + Int64(D_NOPE) + Int64(c4 * Uint32(16)), nb)
                if cutlass.const_expr(lag == 0):
                    wb = _bar(base, B_NFULL + s * self.nlw + pw)
                    cp_async_arrive_inc(wb)
                    cute.arch.sync_warp()
                    if lane == 0:
                        mbar_arrive(wb)
                else:
                    cp_commit()
                    _WAITS[lag - 1]()
                    fence_async_smem()
                    cute.arch.sync_warp()
                    if lane == 0 and g >= lag - 1:
                        complete_tx_release(_bar(base, B_KVFULL + (g - (lag - 1)) % NS),
                                            Uint32(STAGE64 // self.nlw))
                g = g + 1
        if cutlass.const_expr(lag > 0):
            self.drain(base, lane, g)

    @cute.jit
    def qk(self, base, lane, taddr, u, qi, last, cta):
        if cutlass.const_expr(self.lag == 0):
            s = u % NS
            for w in cutlass.range_constexpr(self.nlw):
                mbar_wait(_bar(base, B_NFULL + s * self.nlw + w), Uint32((u // NS) & 1))
            fence_async_smem()
            cute.arch.sync_warp()
            if lane == 0:
                mbar_arrive(_bar(base, B_KVFULL + s))
            cute.arch.sync_warp()
        SparseMlaPrefillH8.qk(self, base, lane, taddr, u, qi, last, cta)

    @cute.jit
    def drain(self, base, lane, g):
        lag = self.lag
        _WAITS[0]()
        fence_async_smem()
        cute.arch.sync_warp()
        if lane == 0:
            for j in cutlass.range_constexpr(lag - 1):
                if g - (lag - 1) + j >= 0:
                    complete_tx_release(_bar(base, B_KVFULL + (g - (lag - 1) + j) % NS),
                                        Uint32(STAGE64 // self.nlw))


@cache
def _compiled(grid: int, knobs: tuple = (), nlw: int = 4, lag: int = 3):
    kernel = SparseMlaPrefillH8G(grid, knobs, nlw, lag)
    args = (Int64(0), Int32(1), Int64(0), Int32(64), Int64(0), Int32(1), Int64(0), Int64(0),
            Float32(1.0), Float32(1.0), Int32(64), Int64(0), Int64(0), Int64(0), Int32(1),
            Int32(1), Int32(6), cuda.CUstream(0))
    return cute.compile(kernel, *args)


def sparse_mla_prefill_h8g(q8: torch.Tensor, kv_rows: torch.Tensor, slots: torch.Tensor,
                           lengths: torch.Tensor, scale: float, out_scale: float,
                           unit_rows: int = 64, out: torch.Tensor | None = None,
                           knobs: tuple = (), nlw: int = 4, lag: int = 3) -> torch.Tensor:
    """``sparse_mla_prefill_h8`` (physical slots) contract with the LDGSTS row producer."""
    t = q8.shape[0]
    if q8.shape[1:] != (H, ROW) or not q8.is_contiguous():
        raise ValueError(f"sparse_mla_prefill_h8g: q8 must be contiguous [T, {H}, {ROW}]")
    if out is None:
        out = torch.empty((t, H, D_NOPE), dtype=torch.bfloat16, device=q8.device)
    if not out.is_contiguous() or out.shape != (t, H, D_NOPE):
        raise ValueError("sparse_mla_prefill_h8g: out must be contiguous [T, 96, 512]")
    if t == 0:
        return out
    sms = torch.cuda.get_device_properties(q8.device).multi_processor_count
    grid = min(sms, t)
    fn = _compiled(grid, tuple(knobs), nlw, lag)
    stream = cuda.CUstream(torch.cuda.current_stream(q8.device).cuda_stream)
    fn(Int64(q8.data_ptr()), Int32(t), Int64(kv_rows.data_ptr()), Int32(unit_rows),
       Int64(slots.data_ptr()), Int32(slots.stride(0)), Int64(lengths.data_ptr()),
       Int64(out.data_ptr()), Float32(scale * LOG2E), Float32(out_scale),
       Int32(kv_rows.shape[0]), Int64(0), Int64(0), Int64(0), Int32(1), Int32(1), Int32(6),
       stream)
    return out
