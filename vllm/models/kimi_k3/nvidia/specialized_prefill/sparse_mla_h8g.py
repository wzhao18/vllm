# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 DSA prefill sparse MLA h8 with an LDGSTS row producer (sm_107).

Same math, pipeline, TMEM plan and shared-memory layout as ``sparse_mla_h8``; only the
KV ring producer differs. h8 fills a 64-key stage with 80 TMA tile::gather4 ops issued
by 9 warps (per-lane operands), which caps the stage rate at the per-SM gather4 issue
rate. Here ``nlw`` warps fill it with 16 B ``cp.async.cg`` (LDGSTS) copies: every lane
owns whole rows (36 chunks of 576 B) or half rows, writes each chunk to its TMA
SWIZZLE_128B (latent atoms) / SWIZZLE_64B (RoPE) position, and signals the stage with
one ``cp.async.mbarrier.arrive.noinc`` per lane, so no lane waits for its own copies.
Invalid keys (slot < 0 or past the row length) are zero-filled like the TMA out-of-bound
fill, so the stage bytes are identical to h8's.
"""

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common import tensormap
from vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla import (
    D_NOPE,
    H,
    LOG2E,
    ROW,
    cp_async_arrive,
    ld_global_s32_pred,
    ld_shared_u32,
    mbar_arrive_tx,
    mbar_init,
    mbar_wait,
    tc_fence_after,
    tc_fence_before,
    tma_tile2d,
    tmem_alloc_exclusive,
    tmem_free_exclusive,
)
from vllm.models.kimi_k3.nvidia.specialized_prefill.sparse_mla_h8 import (
    ATOM,
    B_EPIDONE,
    B_KVEMPTY,
    B_KVFULL,
    B_OFREE,
    B_OFULL,
    B_OREADY,
    B_PEMPTY,
    B_QFULL,
    B_QLFREE,
    B_QREMPTY,
    B_QTM,
    B_SFREE,
    B_SFULL,
    NLD,
    NP,
    NS,
    OFF_KV,
    OFF_QN,
    OFF_QS,
    OFF_TMEM,
    QS_BYTES,
    REG_LAUNCH,
    REG_LOW,
    REG_SM,
    STAGE64,
    THREADS,
    W_PV,
    W_Q,
    W_QK,
    W_SM,
    SparseMlaPrefillH8,
    _bar,
    _subs,
    compiler_fence,
    cp16,
)

ROPE_OFF = 4 * 8192  # RoPE SW64 block of a stage (after the 4 latent SW128 atoms)


class SparseMlaPrefillH8G(SparseMlaPrefillH8):
    def __init__(self, grid: int, knobs: tuple = (), nlw: int = 2):
        super().__init__(grid, knobs)
        if nlw not in (1, 2):
            raise ValueError(f"SparseMlaPrefillH8G: nlw must be 1 or 2, got {nlw}")
        if self.sstamp or self.cpa or self.gskip:
            raise ValueError("SparseMlaPrefillH8G: sstamp/cpa/gskip are h8-only knobs")
        self.nlw = nlw
        # QK fences the generic-proxy (cp.async) stage writes before its MMAs (h8 cpa
        # path); every other use of cpa is in the kernel/loader overridden here.
        self.cpa = 1

    @cute.kernel
    def kernel(self, qn: tensormap.GridMap, qr: tensormap.GridMap,
               kvm: tensormap.GridMap, kvr: tensormap.GridMap, om: tensormap.GridMap,
               rows: Int32, unit_rows: Int32,
               slots: Int64, slot_stride: Int32, lengths: Int64, scale_log2: Float32,
               out_scale: Float32, stamp: Int64, kvp: Int64):
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(tid // 32)
        lane = tid % 32
        smem = cutlass.utils.SmemAllocator()
        raw = smem.allocate_array(cutlass.Uint8, self.smem, byte_alignment=1024)
        base = Uint32(raw.toint())
        if tid == 0:
            for i, cnt in ((B_QTM, 4), (B_QLFREE, 1), (B_OFULL, 1), (B_EPIDONE, 1)):
                mbar_init(_bar(base, i), Uint32(cnt))
            for j in cutlass.range_constexpr(2):
                mbar_init(_bar(base, B_SFULL + j), Uint32(1))
                mbar_init(_bar(base, B_SFREE + j), Uint32(self.sm_arrivals))
                mbar_init(_bar(base, B_QFULL + j), Uint32(1))
                mbar_init(_bar(base, B_QREMPTY + j), Uint32(1))
            for j in cutlass.range_constexpr(4):
                mbar_init(_bar(base, B_OFREE + j), Uint32(4))
            for j in cutlass.range_constexpr(NP):
                mbar_init(_bar(base, B_OREADY + j), Uint32(self.sm_arrivals))
                mbar_init(_bar(base, B_PEMPTY + j), Uint32(1))
            for s in cutlass.range_constexpr(NS):
                # one cp.async arrive per producer lane and stage
                mbar_init(_bar(base, B_KVFULL + s), Uint32(32 * self.nlw))
                mbar_init(_bar(base, B_KVEMPTY + s), Uint32(1))
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
                pw = warp - (W_Q + 1)
                if warp == W_Q:
                    pw = Int32(NLD)
                elif warp > W_QK:
                    pw = warp - (W_Q + 2)
                self.loader(base, pw, lane, cta, n_mine, tensormap.map_address(qn),
                            tensormap.map_address(qr), unit_rows, slots, slot_stride,
                            lengths, kvp)
            cute.arch.warpgroup_reg_alloc(REG_LAUNCH)
        tc_fence_before()
        cute.arch.sync_threads()
        if warp == 0:
            tc_fence_after()
            tmem_free_exclusive(taddr)

    @cute.jit
    def loader(self, base, pw, lane, cta, n_mine, tm_qn, tm_qr, unit_rows, slots,
               slot_stride, lengths, kvp):
        lptr = cute.make_ptr(Int32, lengths, cute.AddressSpace.gmem, assumed_align=4)
        lens = cute.make_tensor(lptr, cute.make_layout(1 << 30))
        if pw == NLD:
            # Q warp: unchanged from h8
            if lane == 0:
                for qi in range(n_mine):
                    row = cta + qi * self.grid
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
        elif pw < self.nlw:
            self.ldgsts_rows(base, pw, lane, cta, n_mine, lens, unit_rows, slots,
                             slot_stride, kvp)

    @cute.jit
    def ldgsts_rows(self, base, pw, lane, cta, n_mine, lens, unit_rows, slots,
                    slot_stride, kvp):
        # nlw 1: lane owns rows lane and lane + 32; nlw 2: row 32 pw + lane
        rpl = 2 if self.nlw == 1 else 1
        r0 = pw * 32 + lane
        g = Int32(0)
        for qi in range(n_mine):
            row = cta + qi * self.grid
            n = lens[row]
            subs = _subs(n)
            sl = slots + Int64(row) * Int64(slot_stride) * Int64(4) + Int64(4 * r0)
            # slot indices two stages ahead (one register pair per owned row)
            a0 = ld_global_s32_pred(sl, Int32(1))
            a1 = ld_global_s32_pred(sl + Int64(256), Int32(subs > 1))
            b0 = Int32(-1)
            b1 = Int32(-1)
            if cutlass.const_expr(rpl == 2):
                b0 = ld_global_s32_pred(sl + Int64(128), Int32(1))
                b1 = ld_global_s32_pred(sl + Int64(384), Int32(subs > 1))
            for t in cutlass.range(subs, unroll=1):
                s = g % NS
                st = base + Uint32(OFF_KV) + Uint32(s) * Uint32(STAGE64)
                ia = a0
                ib = b0
                a0 = a1
                b0 = b1
                a1 = ld_global_s32_pred(sl + Int64((t + 2) * 256), Int32(t + 2 < subs))
                if cutlass.const_expr(rpl == 2):
                    b1 = ld_global_s32_pred(sl + Int64((t + 2) * 256 + 128),
                                            Int32(t + 2 < subs))
                compiler_fence()
                if lane == 0:
                    mbar_wait(_bar(base, B_KVEMPTY + s), Uint32(((g // NS) & 1) ^ 1))
                cute.arch.sync_warp()
                self.copy_row(st, r0, ia, t * 64 + r0, n, unit_rows, kvp)
                if cutlass.const_expr(rpl == 2):
                    self.copy_row(st, r0 + 32, ib, t * 64 + r0 + 32, n, unit_rows, kvp)
                cp_async_arrive(_bar(base, B_KVFULL + s))
                g = g + 1

    @cute.jit
    def copy_row(self, st, r, idx, key, n, unit_rows, kvp):
        """Row r of the stage: chunk k (16 B at byte 16 k of the 576 B KV row) to its
        SW128 latent-atom / SW64 RoPE position; zero fill for masked keys."""
        src = kvp
        nb = Uint32(0)
        if idx >= 0 and key < n:
            src = kvp + Int64((idx >> 6) * unit_rows + (idx & 63)) * Int64(ROW)
            nb = Uint32(16)
        lat = st + Uint32(r) * Uint32(128)
        xl = (Uint32(r) & Uint32(7)) << Uint32(4)
        rope = st + Uint32(ROPE_OFF) + Uint32(r) * Uint32(64)
        xr = ((Uint32(r) >> Uint32(1)) & Uint32(3)) << Uint32(4)
        for k in cutlass.range_constexpr(36):
            if cutlass.const_expr(k < 32):
                dst = lat + Uint32((k // 8) * 8192) + (Uint32((k % 8) << 4) ^ xl)
            else:
                dst = rope + (Uint32((k - 32) << 4) ^ xr)
            cp16(dst, src + Int64(16 * k), nb)


@cache
def _compiled(grid: int, knobs: tuple = (), nlw: int = 2):
    kernel = SparseMlaPrefillH8G(grid, knobs, nlw)
    args = (Int64(0), Int32(1), Int64(0), Int32(64), Int64(0), Int32(1), Int64(0), Int64(0),
            Float32(1.0), Float32(1.0), Int32(64), Int64(0), cuda.CUstream(0))
    return cute.compile(kernel, *args)


def sparse_mla_prefill_h8g(q8: torch.Tensor, kv_rows: torch.Tensor, slots: torch.Tensor,
                           lengths: torch.Tensor, scale: float, out_scale: float,
                           unit_rows: int = 64, out: torch.Tensor | None = None,
                           knobs: tuple = (), nlw: int = 2) -> torch.Tensor:
    """``sparse_mla_prefill_h8`` contract with the LDGSTS row producer."""
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
    fn = _compiled(grid, tuple(knobs), nlw)
    stream = cuda.CUstream(torch.cuda.current_stream(q8.device).cuda_stream)
    fn(Int64(q8.data_ptr()), Int32(t), Int64(kv_rows.data_ptr()), Int32(unit_rows),
       Int64(slots.data_ptr()), Int32(slots.stride(0)), Int64(lengths.data_ptr()),
       Int64(out.data_ptr()), Float32(scale * LOG2E), Float32(out_scale),
       Int32(kv_rows.shape[0]), Int64(0), stream)
    return out
