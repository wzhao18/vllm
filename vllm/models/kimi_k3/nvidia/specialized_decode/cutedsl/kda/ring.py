# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KdaRing: KdaFlow's deferred-commit record step as a page stream that never waits
on the item math (one persistent CTA per SM, 384 threads, KdaFlow's item ranges).

Roles:

- **scan** (warps 4-11, 256 threads; thread (b, l), b = c // 8, l = c % 8: rows
  4 b + [0, 4), KdaFlow's keys 4 (l + 8 j) + [0, 4), j < 4) runs the CTA's items in
  order: it copies the item's FP32 page from its ring slot into registers and at once
  refills its warp's eighth of the slot (rows 16 w + [0, 16)) with item j + SLOTS's,
  folds the pending commit (the commit's math), stores the committed checkpoint from
  registers, accumulates the 2q dot products and reduces them over the 8 lanes of a
  row (KdaFlow's butterfly) into the item's row sums (shared memory).
- **prep** (warps 0-3, thread c = channel / row c) alternates item j + 1's prep
  (convolution + SiLU, decays, q / k norms, token couplings, fold weights; the new key
  / raw-gate records and the rolled window straight to global memory) into operand
  block (j + 1) % NOPS with item j's tail (solve, gated RMSNorm, corrections and
  outputs). Its inputs (projection rows, output gates, window, pending records) are
  bulk-copied NSTAGE - 1 items ahead, one copy per issuing lane (a thread's bulk
  copies complete one at a time).

Prologue: the item table, head constants, F_B (in the last page slot), F_A and the
betas; the F_B gate projection of every item on all 12 warps; the first pages.

The math is KdaFlow's operation for operation, in its order (per-thread key order,
butterfly pairing, block-reduction trees), so outputs, checkpoints, windows and
records are bitwise KdaFlow's.
"""

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Uint32

from ..common.primitives import alloc, asm, at, copy_g2s, fma, launch, mma_into, zeros
from .flow import (
    A_EXP,
    CHUNKS,
    CONSTS,
    CONV_OFFSET,
    CONV_SLOT,
    DT,
    HEAD,
    HSEL,
    LANES,
    MAX_ITEMS,
    META,
    PENDING,
    RECORD,
    REQUEST,
    SC_HL,
    SC_MINE,
    SLOT,
    SUMS,
    K,
    KdaFlow,
    clock64,
    couplings,
    load_global_u32x4,
    store_shared_f32x4,
    store_shared_u32x4,
)
from .recover import mul_rn, reciprocal, sqrt_approx, triton_exp, triton_sigmoid
from .specialized import (
    PAGE_BYTES,
    fence_async_shared,
    load_shared_f32x4,
    load_shared_u32x2,
    set_max_registers,
    sync_threads,
)
from .wide import rsqrt_ftz

PREP_THREADS = TAIL_THREADS = 128
SCAN_THREADS = 256
SCAN_BASE = PREP_THREADS + TAIL_THREADS  # the first scan thread
THREADS = SCAN_BASE + SCAN_THREADS
SCAN_WARPS = SCAN_THREADS // 32
QROWS = 4  # state rows per scan thread: 4 (c // 8) + [0, 4)
EIGHTH = PAGE_BYTES // SCAN_WARPS  # a scan warp's rows 16 w + [0, 16) of a page
SLOTS = 3  # the CTA's state page ring (item j: slot j % SLOTS)
NOPS = 3  # operand blocks (item j: block j % NOPS): prep j + 1, scan j, tail j - 1
NSTAGE = 3  # prep input staging buffers (item j: buffer j % NSTAGE)
PREP_BAR, TAIL_BAR = 1, 2  # named barriers (0: the CTA's)
# setmaxnreg per warpgroup: 104 x 128 + 88 x 128 + 160 x 256 = 128 x 512.
PREP_REGISTERS, TAIL_REGISTERS, SCAN_REGISTERS = 104, 88, 160


def shared_bytes(tokens, window, conv_size):
    """Shared memory of one KdaRing CTA (bytes)."""
    pages = SLOTS * PAGE_BYTES
    heads = 2 * CONSTS * K * 4
    tables = MAX_ITEMS * (tokens * K * 2 + (META + 8) * 4)  # raw gates, table, betas
    block = 5 * tokens * K * 4 + K * 4 + tokens * tokens * 4 + tokens * K * 2
    staging = 3 * K * window * conv_size + tokens * K * (3 * 2 + 2 + 4 + 4)
    sums = SUMS * 16 + 2 * 64 + 2 * K * 2 * tokens * 4
    return pages + heads + tables + NOPS * block + NSTAGE * staging + sums + 4096


@asm()
def store_global_cs(pointer, values):
    """16 B register -> global store, streaming (evict-first) cache policy."""
    return "st.global.cs.v4.f32 [$0], {$1, $2, $3, $4};", "l,f,f,f,f", (
        pointer,
        *values,
    )


class KdaRing(KdaFlow):
    """KdaFlow's operands and semantics (deferred records; ``dies`` / ``mx`` as
    KdaFlow's)."""

    def __init__(self, heads, tokens, gate_mode, has_bias, conv_dim_first, window,
                 ctas, dies=False, mx=False):  # fmt: skip
        super().__init__(heads, tokens, gate_mode, has_bias, conv_dim_first, window,
                         ctas, 1, dies, mx)  # fmt: skip

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, self.ctas, THREADS, min_blocks_per_mp=1)

    @cute.kernel
    def kernel(self, args):
        self.ring_body(args)

    # ---------------------------------------------------------------- prologue
    @cute.jit
    def ring_body(self, args):
        k0 = Int64(0)
        ks = zeros(6, Int64)  # TRACE: prologue stamps
        if cutlass.const_expr(self.trace):
            k0 = clock64()
        conv_state, output = args[3], args[10]
        f_a, beta, f_b = args[4], args[5], args[16]
        T: cutlass.Constexpr = self.tokens
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        requests: cutlass.Constexpr = output.shape[0] // T
        total_items: cutlass.Constexpr = self.heads * requests
        first = Int32(total_items * cta // self.ctas)
        count = Int32(total_items * (cta + 1) // self.ctas) - first
        conv_type: cutlass.Constexpr = conv_state.element_type

        smem = cutlass.utils.SmemAllocator()
        pages = alloc(smem, Float32, (SLOTS, K * K), align=1024)
        scratch = cute.make_tensor(
            cute.recast_ptr(at(pages, 0, 0), dtype=Int32), cute.make_layout(1024)
        )
        if cutlass.const_expr(self.dies):  # die-matched items
            count = self.die_claim(args, tid, cta, scratch)
        hcs = alloc(smem, Float32, (2, CONSTS, K))
        raw = alloc(smem, BFloat16, (MAX_ITEMS, T, K), align=16)
        table = alloc(smem, Int32, (MAX_ITEMS, META))
        betas = alloc(smem, Float32, (MAX_ITEMS, 8))
        # Operand blocks: the 2q scan vectors (2 blocks), v, the fold weights w_t.
        qkvd = alloc(smem, Float32, (NOPS, 4, T, K), align=128)
        xv = cute.make_tensor(
            at(qkvd, 0, 0, 0, 0),
            cute.make_layout((NOPS, 2 * T * K), stride=(4 * T * K, 1)),
        )
        v_s, fw_s = qkvd[None, 2, None, None], qkvd[None, 3, None, None]
        fu_s = alloc(smem, Float32, (NOPS, T, K))  # the old corrections u_t
        fd_s = alloc(smem, Float32, (NOPS, K))  # D_n
        ce_s = alloc(smem, Float32, (NOPS, T * T))  # the token couplings
        g_s = alloc(smem, BFloat16, (NOPS, T, K))  # the output gates
        convs = alloc(smem, conv_type, (NSTAGE, 3, K * self.window), align=128)
        xs = alloc(smem, BFloat16, (NSTAGE, T, 3, K), align=128)
        gs = alloc(smem, BFloat16, (NSTAGE, T, K), align=128)
        kgs = alloc(smem, BFloat16, (NSTAGE, T, 2 * K), align=128)
        uss = alloc(smem, Float32, (NSTAGE, T, K), align=128)
        psums = alloc(smem, Float32, (SUMS, 4))
        psum2 = alloc(smem, Float32, (2, 4, 4))  # parity, token, warp
        rsum = alloc(smem, Float32, (2, K, 2 * T), align=16)  # parity, row, 2q sums
        bars = smem.allocate(cutlass.Uint64, SLOTS + 2 * NOPS + 4 + NSTAGE + 1)
        state_full, ops_full, ops_empty = bars, bars + SLOTS, bars + SLOTS + NOPS
        dots_full = bars + SLOTS + 2 * NOPS
        dots_empty = dots_full + 2
        stage_full = dots_full + 4
        fb_full = stage_full + NSTAGE
        # Prologue scratch: the F_B rows of the CTA's heads in the last page slot
        # (its page loads after the gate projection), the items' F_A rows over the
        # operand blocks (unused until the first prep).
        fbs = cute.make_tensor(
            cute.recast_ptr(at(pages, SLOTS - 1, 0), dtype=BFloat16),
            cute.make_layout((2, K, K), stride=(K * K, K, 1)),
        )
        fas = cute.make_tensor(
            cute.recast_ptr(at(qkvd, 0, 0, 0, 0), dtype=BFloat16),
            cute.make_layout((MAX_ITEMS, T, K), stride=(T * K, K, 1)),
        )

        if tid == 0:
            for s in cutlass.range_constexpr(SLOTS):  # one arrival per scan warp
                cute.arch.mbarrier_init(state_full + s, SCAN_WARPS)
            for b in cutlass.range_constexpr(NOPS):
                cute.arch.mbarrier_init(ops_full + b, 1)
                cute.arch.mbarrier_init(ops_empty + b, 1)
            for p in cutlass.range_constexpr(2):
                cute.arch.mbarrier_init(dots_full + p, SCAN_WARPS)
                cute.arch.mbarrier_init(dots_empty + p, 1)
            for i in cutlass.range_constexpr(NSTAGE):
                cute.arch.mbarrier_init(stage_full + i, 1)
            cute.arch.mbarrier_init(fb_full, 1)
            cute.arch.mbarrier_init_fence()
        h0 = first // requests
        h1 = (first + count - 1) // requests
        if cutlass.const_expr(self.dies):
            h0 = scratch[SC_MINE] // requests
        if tid < count:
            item = first + tid
            if cutlass.const_expr(self.dies):
                item = scratch[SC_MINE + tid]
            head, request, slot, conv_slot, conv_offset, pending, rec = self.item_info(
                args, item
            )
            table[tid, HSEL] = Int32(head != h0)
            if cutlass.const_expr(self.dies):
                if head != h0:  # the second head (claims span at most two)
                    scratch[SC_HL] = head
            table[tid, SLOT] = slot
            table[tid, HEAD] = head
            table[tid, REQUEST] = request
            table[tid, CONV_SLOT] = conv_slot
            table[tid, CONV_OFFSET] = conv_offset
            table[tid, PENDING] = pending
            table[tid, RECORD] = rec
        cute.arch.sync_warp()
        cute.arch.barrier()
        if cutlass.const_expr(self.dies):
            h1 = scratch[SC_HL]
            if h1 < 0:
                h1 = h0
        heads = Int32(1) + Int32(h1 != h0)
        if cutlass.const_expr(self.trace):
            ks[0] = clock64() - k0
        # Before the PDL wait: the head constants, then F_B (two issuing threads).
        hslot = tid // K
        BULK_HC: cutlass.Constexpr = self.bulk_head_constants(args)
        if cutlass.const_expr(BULK_HC):  # conv weight and dt rows: 13 copies a head
            lane, hw = tid % 32, tid // 32
            if hw < heads:
                rows = hcs[hw, None, None]
                if lane < 12:
                    copy_g2s(at(rows, lane, 0),
                             at(args[1], lane // 4, lane % 4, (h0 + (h1 - h0) * hw) * K), K * 4,
                             fb_full)  # fmt: skip
                if lane == 12:
                    copy_g2s(at(rows, DT, 0), at(args[7], (h0 + (h1 - h0) * hw) * K), K * 4, fb_full)
        elif hslot < heads:
            self.head_constants(hcs[hslot, None, None], args, h0 + (h1 - h0) * hslot, tid % K)
        if tid == 0:
            hc_bytes = 0
            if cutlass.const_expr(BULK_HC):
                hc_bytes = heads * 13 * K * 4
            cute.arch.mbarrier_arrive_and_expect_tx(fb_full,
                                                    heads * K * K * 2 + hc_bytes)  # fmt: skip
        for h in cutlass.range_constexpr(2):
            if tid == 64 + 32 * h:  # noqa: SIM102
                if h < heads:
                    copy_g2s(at(fbs, h, 0, 0), at(f_b, (h1 if h > 0 else h0) * K, 0),
                             K * K * 2, fb_full)  # fmt: skip
        cute.arch.sync_warp()
        if cutlass.const_expr(self.trace):
            ks[5] = clock64() - k0
        cute.arch.griddepcontrol_wait()
        if cutlass.const_expr(not self.mx):
            cute.arch.griddepcontrol_launch_dependents()
        if cutlass.const_expr(self.trace):
            ks[1] = clock64() - k0
        # F_A rows (16 B per thread, all in flight at once) and the betas.
        FA_CHUNKS: cutlass.Constexpr = K * 2 // 16
        FA_ROUNDS: cutlass.Constexpr = -(-MAX_ITEMS * T * FA_CHUNKS // THREADS)
        fav = cute.make_rmem_tensor((FA_ROUNDS, 4), Uint32)
        for k in cutlass.range_constexpr(FA_ROUNDS):
            idx = tid + k * THREADS
            item = idx // (T * FA_CHUNKS)
            if item < count:
                t, part = (idx // FA_CHUNKS) % T, idx % FA_CHUNKS
                row = table[item, REQUEST] * T + t
                v = load_global_u32x4(at(f_a, row, part * 8))
                for e in cutlass.range_constexpr(4):
                    fav[k, e] = v[e]
        if tid < count * T:
            item, t = tid // T, tid % T
            row = table[item, REQUEST] * T + t
            betas[item, t] = triton_sigmoid(Float32(beta[row, table[item, HEAD]]))
        if cutlass.const_expr(BULK_HC):  # noqa: SIM102
            if hslot < heads:
                hcs[hslot, A_EXP, tid % K] = triton_exp(Float32(args[6][h0 + (h1 - h0) * hslot]))
        for k in cutlass.range_constexpr(FA_ROUNDS):
            idx = tid + k * THREADS
            item = idx // (T * FA_CHUNKS)
            if item < count:
                t, part = (idx // FA_CHUNKS) % T, idx % FA_CHUNKS
                store_shared_u32x4(at(fas, item, t, part * 8),
                                   [fav[k, e] for e in range(4)])  # fmt: skip
        staging = (convs, xs, gs, kgs, uss)
        if tid < 128 * (NSTAGE - 1):  # the first items' prep inputs (warpgroup i: item i)
            i = tid // 128
            if i < count:
                self.stage(args, table, i, staging, stage_full + i, i, tid % 128)
        cute.arch.sync_warp()
        if cutlass.const_expr(self.trace):
            ks[2] = clock64() - k0
        self.bwait(fb_full, 0, "fb_full")
        if cutlass.const_expr(self.trace):
            ks[3] = clock64() - k0
        # The first pages (eighth w of slot s from lane s of scan warp w), then the
        # gate projection of every item, then the last slot's page.
        fence_async_shared()
        lane, warp = tid % 32, tid // 32
        if warp >= SCAN_BASE // 32:  # noqa: SIM102
            if lane < SLOTS - 1:  # noqa: SIM102
                if lane < count:
                    self.load_eighth(args, pages, lane, table, lane, state_full,
                                     warp - SCAN_BASE // 32)  # fmt: skip
        cute.arch.sync_warp()
        cute.arch.barrier()  # F_A
        self.gate_projection3(tid, count, table, fbs, fas, raw)
        cute.arch.sync_warp()
        cute.arch.barrier()  # raw gates; the last page slot and the blocks are free
        if cutlass.const_expr(self.trace):
            ks[4] = clock64() - k0
        fence_async_shared()
        if warp >= SCAN_BASE // 32:  # noqa: SIM102
            if lane == SLOTS - 1:  # noqa: SIM102
                if lane < count:
                    self.load_eighth(args, pages, lane, table, lane, state_full,
                                     warp - SCAN_BASE // 32)  # fmt: skip
        cute.arch.sync_warp()
        if cutlass.const_expr(self.trace):
            k0 = clock64() - k0  # the prologue's cycles (printed by the prep trace)
        ops = (xv, v_s, g_s, fw_s, fu_s, fd_s, ce_s)
        barriers = (state_full, ops_full, ops_empty, dots_full, dots_empty, stage_full)
        if tid < PREP_THREADS:
            set_max_registers(PREP_REGISTERS, increase=False)
            self.prep_items(args, tid, count, table, raw, hcs, staging, ops, psums,
                            barriers)  # fmt: skip
        elif tid < SCAN_BASE:
            set_max_registers(TAIL_REGISTERS, increase=False)
            self.tail_items(args, tid - PREP_THREADS, count, table, betas, ops, psum2,
                            rsum, barriers, k0, ks)  # fmt: skip
        else:
            set_max_registers(SCAN_REGISTERS, increase=True)
            self.scan_side(args, tid - SCAN_BASE, count, pages, table, ops, rsum,
                           barriers)  # fmt: skip

    @cute.jit
    def load_eighth(self, args, pages, s, table, j, state_full, w):
        """Bulk-copy rows 16 w + [0, 16) (scan warp w's) of table item j's state page
        into page slot ``s``: one of the slot barrier's eight arrivals (a padded
        item's carries no bytes)."""
        slot, head = table[j, SLOT], table[j, HEAD]
        barrier = state_full + s
        if slot > 0:
            rows: cutlass.Constexpr = EIGHTH // (K * 4)
            cute.arch.mbarrier_arrive_and_expect_tx(barrier, EIGHTH)
            source = args[9][Int64(slot), None, None, None]
            copy_g2s(at(pages, s, w * rows * K), at(source, head, w * rows, 0),
                     EIGHTH, barrier)  # fmt: skip
        else:
            cute.arch.mbarrier_arrive_and_expect_tx(barrier, 0)

    @cute.jit
    def gate_projection3(self, tid, count, table, fbs, fas, raw):
        """raw[j, t, f] = BF16(F_B[head_j] @ F_A[j, t]) for the CTA's items (warp w:
        features 32 (w % 4) + [0, 32), items w // 4 + 4 i; KdaFlow's MMA chain per
        item). Results are held until every warp has read F_A (which overlays the
        operand blocks)."""
        T: cutlass.Constexpr = self.tokens
        lane, warp = tid % 32, tid // 32
        group_id, quarter = lane >> 2, lane & 3
        wq, third = warp % 4, warp // 4
        PARTS: cutlass.Constexpr = THREADS // 128
        PER: cutlass.Constexpr = -(-MAX_ITEMS // PARTS)
        acc = cute.make_rmem_tensor((PER, 2, 4), Float32)
        wr = cute.make_rmem_tensor((2, 8, 4), Uint32)
        cached = Int32(-1)
        for i in cutlass.range_constexpr(PER):
            j = third + PARTS * i
            if j < count:
                head = table[j, HEAD]
                if head != cached:
                    fb = fbs[table[j, HSEL], None, None]
                    for m in cutlass.range_constexpr(2):
                        for chunk in cutlass.range_constexpr(8):
                            feature = 32 * wq + 16 * m + group_id
                            col = 16 * chunk + 4 * quarter
                            low = load_shared_u32x2(at(fb, feature, col))
                            high = load_shared_u32x2(at(fb, feature + 8, col))
                            wr[m, chunk, 0], wr[m, chunk, 2] = low
                            wr[m, chunk, 1], wr[m, chunk, 3] = high
                    cached = head
                cute.arch.sync_warp()  # converged for mma.sync
                for m in cutlass.range_constexpr(2):
                    chains: cutlass.Constexpr = 4 if self.gate_mode == 1 else 1
                    part = zeros((chains, 4), Float32)
                    for chunk in cutlass.range_constexpr(8):
                        chain: cutlass.Constexpr = (
                            chunk // 2 if self.gate_mode == 1 else 0
                        )
                        wa = tuple(wr[m, chunk, e] for e in range(4))
                        ab = load_shared_u32x2(
                            at(fas, j, group_id % T, 16 * chunk + 4 * quarter)
                        )
                        mma_into(part[chain, None], wa, ab, bf16=True)
                    for e in cutlass.range_constexpr(4):
                        value = part[0, e]
                        if cutlass.const_expr(self.gate_mode == 1):
                            value = Float32(0) + (
                                ((value + part[1, e]) + part[2, e]) + part[3, e]
                            )
                        acc[i, m, e] = value
        cute.arch.sync_warp()
        cute.arch.barrier()  # every warp has read F_A
        for i in cutlass.range_constexpr(PER):
            j = third + PARTS * i
            if j < count:
                for m in cutlass.range_constexpr(2):
                    for e in cutlass.range_constexpr(4):
                        t = 2 * quarter + e % 2
                        feature = 32 * wq + 16 * m + group_id + 8 * (e // 2)
                        if t < T:
                            raw[j, t, feature] = BFloat16(acc[i, m, e])

    # ------------------------------------------------------------------- prep
    @cute.jit
    def prep_sync(self):
        cute.arch.sync_warp()  # converged before the aligned barrier
        sync_threads(PREP_BAR, PREP_THREADS)

    @cute.jit
    def stage(self, args, table, j, staging, barrier, st, r):
        """Bulk-copy table item j's prep inputs into staging buffer ``st``: thread r
        of a warpgroup issues copy 4 (r % 32) + r // 32 (each warp's TMA queue takes a
        quarter): output gate rows, q / k / v rows, the window and, with a pending
        commit, its records."""
        x, conv_state, gate = args[0], args[3], args[11]
        convs, xs, gs, kgs, uss = staging
        T: cutlass.Constexpr = self.tokens
        dim: cutlass.Constexpr = self.heads * K
        size: cutlass.Constexpr = conv_state.element_type.width // 8
        part_bytes: cutlass.Constexpr = K * self.window * size
        slot, head, pending = table[j, SLOT], table[j, HEAD], table[j, PENDING]
        if slot > 0:
            rows: cutlass.Constexpr = 4 * T * K * 2 + 3 * part_bytes
            records: cutlass.Constexpr = T * 2 * K * 2 + T * K * 4
            total = Int32(rows)
            if pending > 0:
                total = total + records
            token = table[j, REQUEST] * T
            rec = table[j, RECORD]
            lane = 4 * (r % 32) + r // 32  # this thread's copy
            if r == 0:
                cute.arch.mbarrier_arrive_and_expect_tx(barrier, total)
            if lane < T:
                copy_g2s(at(gs, st, lane, 0), at(gate, token + lane, head * K), K * 2,
                         barrier)  # fmt: skip
            if lane >= T:  # noqa: SIM102
                if lane < 4 * T:
                    part, t = (lane - T) // T, (lane - T) % T
                    copy_g2s(at(xs, st, t, part, 0),
                             at(x, token + t, part * dim + head * K), K * 2,
                             barrier)  # fmt: skip
            if lane >= 4 * T:  # noqa: SIM102
                if lane < 4 * T + 3:
                    part = lane - 4 * T
                    page = conv_state[Int64(table[j, CONV_SLOT]), None, None]
                    copy_g2s(at(convs, st, part, 0), at(page, part * dim + head * K, 0),
                             part_bytes, barrier)  # fmt: skip
            if pending > 0:
                if lane == 4 * T + 3:
                    copy_g2s(at(kgs, st, 0, 0), at(args[19], rec, head, 0, 0),
                             T * 2 * K * 2, barrier)  # fmt: skip
                if lane == 4 * T + 4:
                    copy_g2s(at(uss, st, 0, 0), at(args[18], rec, head, 0, 0),
                             T * K * 4, barrier)  # fmt: skip
        else:
            if r == 0:
                cute.arch.mbarrier_arrive_and_expect_tx(barrier, 0)

    @cute.jit
    def prep_items(self, args, c, count, table, raw, hcs, staging, ops, psums,
                   barriers):  # fmt: skip
        """The CTA's items in order: inputs staged NSTAGE - 1 items ahead, then the
        operand block once the tail has released it."""
        _, ops_full, ops_empty, _, _, stage_full = barriers
        pt = zeros(8, Int64)  # TRACE: stage issue, stage wait, block wait, conv..terms,
        # block sums, sync, post, total
        if cutlass.const_expr(self.trace):
            pt[7] = clock64()
        for j in cutlass.range(count, unroll=1):
            st, b = j % NSTAGE, j % NOPS
            ahead = j + NSTAGE - 1
            q0 = Int64(0)
            if cutlass.const_expr(self.trace):
                q0 = clock64()
            if ahead < count:  # (its buffer's last reader, item j - 1, is done)
                self.stage(args, table, ahead, staging, stage_full + ahead % NSTAGE,
                           ahead % NSTAGE, c)  # fmt: skip
            cute.arch.sync_warp()
            if cutlass.const_expr(self.trace):
                q1 = clock64()
                pt[0] = pt[0] + (q1 - q0)
                q0 = q1
            self.bwait(stage_full + st, (j // NSTAGE) % 2, "stage_full")
            if cutlass.const_expr(self.trace):
                q1 = clock64()
                pt[1] = pt[1] + (q1 - q0)
                q0 = q1
            if j >= NOPS:
                self.bwait(ops_empty + b, (j // NOPS - 1) % 2, "ops_empty")
            if cutlass.const_expr(self.trace):
                pt[2] = pt[2] + (clock64() - q0)
            if cutlass.const_expr("prep" not in self.skip):  # noqa: SIM102
                if table[j, SLOT] > 0:
                    self.prep_ops(args, c, j, b, st, table, raw,
                                  hcs[table[j, HSEL], None, None], staging, ops,
                                  psums, pt)  # fmt: skip
            self.prep_sync()  # the block is complete
            if c == 0:
                cute.arch.mbarrier_arrive(ops_full + b)
        if cutlass.const_expr(self.trace):  # noqa: SIM102
            if c == 0:
                cta, _, _ = cute.arch.block_idx()
                if cta % 53 == 0:
                    cute.printf("RTRACE prep cta {} items {} total {} | stage-issue {} stage-wait {} block-wait {} conv..terms {} bsum {} sync {} post {}", cta, count, clock64() - pt[7], pt[0], pt[1], pt[2], pt[3], pt[4], pt[5], pt[6])  # noqa: E501

    @cute.jit
    def tail_sync(self):
        cute.arch.sync_warp()  # converged before the aligned barrier
        sync_threads(TAIL_BAR, TAIL_THREADS)

    @cute.jit
    def tail_items(self, args, c, count, table, betas, ops, psum2, rsum, barriers,
                   prologue, ks):  # fmt: skip
        """Each item's solve, gated RMSNorm and outputs once its row sums are in."""
        tr = zeros(3, Int64)  # TRACE: row-sum wait, work, start
        if cutlass.const_expr(self.trace):
            tr[2] = clock64()
        for k in cutlass.range(count, unroll=1):
            t0 = Int64(0)
            if cutlass.const_expr(self.trace):
                t0 = clock64()
            self.tail(args, c, k, table, betas, ops, psum2, rsum, barriers, tr)
            if cutlass.const_expr(self.trace):
                tr[1] = tr[1] + (clock64() - t0)
        if cutlass.const_expr(self.trace):  # noqa: SIM102
            if c == 0:
                cta, _, _ = cute.arch.block_idx()
                if cta % 53 == 0:
                    cute.printf("RTRACE pro cta {} table {} hc-issue {} pdl {} fa+stage {} fb {} gate {} total {}", cta, ks[0], ks[5], ks[1], ks[2], ks[3], ks[4], prologue)  # noqa: E501
                    cute.printf("RTRACE tail cta {} items {} total {} | dots-wait {} all {}", cta, count, clock64() - tr[2], tr[0], tr[1])  # noqa: E501

    @cute.jit
    def prep_ops(self, args, c, j, b, st, table, raw, hc, staging, ops, psums, pt):
        """KdaFlow.prep_item for item j (thread c = channel / key c) from staging
        buffer ``st`` into operand block ``b``; the new key / raw-gate records and the
        rolled window are stored straight to global memory."""
        q0 = Int64(0)
        if cutlass.const_expr(self.trace):
            q0 = clock64()
        lower_bound, scale = args[13], args[14]
        xv, v_s, g_s, fw_s, fu_s, fd_s, ce_s = ops
        convs_all, xs_all, gs_all, kgs_all, uss_all = staging
        convs = convs_all[st, None, None]
        T: cutlass.Constexpr = self.tokens
        NV: cutlass.Constexpr = 2 * T
        dim: cutlass.Constexpr = self.heads * K
        conv_type: cutlass.Constexpr = convs.element_type
        consts = hc[None, c]
        index = couplings(T)
        head, rec, pending = table[j, HEAD], table[j, RECORD], table[j, PENDING]
        offset = table[j, CONV_OFFSET]
        kg = args[19]
        masked = Float32(0)
        if pending == 0:
            masked = Float32(-0.0)
        old_keys = zeros(T, Float32)
        old_gates = zeros(T, Float32)
        for t in cutlass.range_constexpr(T):
            fu_s[b, t, c] = masked
            if t < pending:
                old_keys[t] = Float32(kgs_all[st, t, c])
                old_gates[t] = Float32(kgs_all[st, t, K + c])
                fu_s[b, t, c] = uss_all[st, t, c]
        a = consts[A_EXP]
        dt = consts[DT]

        # Convolution + SiLU of q, k, v; the rolled window back to the cache.
        page = args[3][Int64(table[j, CONV_SLOT]), None, None]
        values = cute.make_rmem_tensor((3, T), Float32)
        for part in cutlass.range_constexpr(3):
            taps = cute.make_rmem_tensor(3, conv_type)
            for t in cutlass.range_constexpr(3):
                taps[t] = convs[part, c * self.window + offset + t]
            inputs = [xs_all[st, t, part, c] for t in range(T)]
            for t in cutlass.range_constexpr(T):
                accum = Float32(0)
                if cutlass.const_expr(self.has_bias):
                    accum = consts[12 + part]
                for i in cutlass.range_constexpr(4):
                    tap = t + i
                    term = taps[tap] if tap < 3 else inputs[tap - 3]
                    accum = fma(Float32(term), consts[4 * part + i], accum)
                values[part, t] = self.silu(accum)
            row = part * dim + head * K + c
            page[row, 0] = conv_type(taps[1])
            page[row, 1] = conv_type(taps[2])
            for t in cutlass.range_constexpr(T):
                page[row, 2 + t] = conv_type(inputs[t])
        decays = zeros(T, Float32)
        for t in cutlass.range_constexpr(T):
            v_s[b, t, c] = values[2, t]
            g_s[b, t, c] = gs_all[st, t, c]
            kg[rec, head, t, c] = BFloat16(values[1, t])
            raw_g = raw[j, t, c]
            kg[rec, head, t, K + c] = raw_g
            decays[t] = self.decay(a, Float32(raw_g), dt, lower_bound)
        cumulative = [decays[0]]
        for t in cutlass.range_constexpr(1, T):
            cumulative.append(mul_rn(cumulative[t - 1], decays[t]))
        ratio = {}
        for t in cutlass.range_constexpr(1, T):
            ratio[(t, t - 1)] = decays[t]
            for s in cutlass.range_constexpr(t - 1):
                ratio[(t, s)] = mul_rn(ratio[(t - 1, s)], decays[t])
        old_decays = zeros(T, Float32)
        if pending > 0:
            for t in cutlass.range_constexpr(T):
                old_decays[t] = self.decay(a, old_gates[t], dt, lower_bound)

        # One block reduction: q / k squares, the couplings, the recorded-key squares.
        terms = [Float32(0) for _ in range(SUMS)]
        for t in cutlass.range_constexpr(T):
            terms[2 * t] = values[0, t] * values[0, t]
            terms[2 * t + 1] = values[1, t] * values[1, t]
            terms[NV + T * T + t] = old_keys[t] * old_keys[t]
        for i in cutlass.range_constexpr(len(index)):
            key = list(index)[i]
            kind, t, s = key
            weighted = values[1, s]
            if cutlass.const_expr(s < t):
                weighted = mul_rn(weighted, ratio[(t, s)])
            other = values[1, t] if kind == "c" else values[0, t]
            terms[NV + i] = mul_rn(weighted, other)
        if cutlass.const_expr(self.trace):
            q1 = clock64()
            pt[3] = pt[3] + (q1 - q0)
            q0 = q1
        self.block_sums(terms, psums, c, SUMS)
        if cutlass.const_expr(self.trace):
            q1 = clock64()
            pt[4] = pt[4] + (q1 - q0)
            q0 = q1
        self.prep_sync()  # the partial sums
        if cutlass.const_expr(self.trace):
            q1 = clock64()
            pt[5] = pt[5] + (q1 - q0)
            q0 = q1
        fq = [Float32(0) for _ in range(T)]
        fk = [Float32(0) for _ in range(T)]
        for t in cutlass.range_constexpr(T):
            qn = self.quad_sum(psums[2 * t, None]) + 1.0e-6
            kn = self.quad_sum(psums[2 * t + 1, None]) + 1.0e-6
            fq[t] = reciprocal(sqrt_approx(qn))
            fk[t] = reciprocal(sqrt_approx(kn))
        vec = [Float32(0) for _ in range(NV)]
        for t in cutlass.range_constexpr(T):
            q_t = mul_rn(mul_rn(values[0, t], fq[t]), scale)
            k_t = mul_rn(values[1, t], fk[t])
            vec[t] = mul_rn(cumulative[t], k_t)
            vec[T + t] = mul_rn(cumulative[t], q_t)
        chunk, lane_in, e = c // (4 * LANES), (c // 4) % LANES, c % 4
        for h in cutlass.range_constexpr(NV // 4):
            pos = (((chunk * 4 + e) * (NV // 4) + h) * LANES + lane_in) * 4
            store_shared_f32x4(at(xv, b, pos), [vec[4 * h + i] for i in range(4)])
        norm = Float32(0)
        for i in cutlass.range_constexpr(len(index)):
            key = list(index)[i]
            kind, t, s = key
            norm_i = mul_rn(fk[s], fk[t])
            if cutlass.const_expr(kind == "e"):
                norm_i = mul_rn(fk[s], mul_rn(fq[t], scale))
            norm = Float32(cutlass.select_(c == i, norm_i, norm))
        if c < len(index):
            ce_s[b, c] = mul_rn(self.quad_sum(psums[NV + c, None]), norm)
        final_decay = Float32(1)
        for i in cutlass.range_constexpr(T):
            t = T - 1 - i
            w_t = Float32(0)
            if t < pending:
                total = self.quad_sum(psums[NV + T * T + t, None]) + 1.0e-6
                kn_t = mul_rn(rsqrt_ftz(total), old_keys[t])
                w_t = mul_rn(kn_t, final_decay)
                final_decay = mul_rn(final_decay, old_decays[t])
            fw_s[b, t, c] = w_t
        fd_s[b, c] = final_decay
        if cutlass.const_expr(self.trace):
            pt[6] = pt[6] + (clock64() - q0)

    @cute.jit
    def tail(self, args, c, k, table, betas, ops, psum2, rsum, barriers, tr):
        """Item k's solve, gated RMSNorm and outputs (thread c = row / channel c) from
        its row sums: KdaFlow.solve and finish_alone."""
        _, _, ops_empty, dots_full, dots_empty, _ = barriers
        xv, v_s, g_s, fw_s, fu_s, fd_s, ce_s = ops
        eps = args[15]
        T: cutlass.Constexpr = self.tokens
        NV: cutlass.Constexpr = 2 * T
        p, b = k % 2, k % NOPS
        slot, head, rec = table[k, SLOT], table[k, HEAD], table[k, RECORD]
        token = table[k, REQUEST] * T
        nw = args[12][c]
        t0 = Int64(0)
        if cutlass.const_expr(self.trace):
            t0 = clock64()
        self.bwait(dots_full + p, (k // 2) % 2, "dots_full")
        if cutlass.const_expr(self.trace):
            tr[0] = tr[0] + (clock64() - t0)
        if slot > 0:
            values = []
            for h in cutlass.range_constexpr(NV // 4):
                v4 = load_shared_f32x4(at(rsum, p, c, 4 * h))
                values.extend([v4[i] for i in range(4)])
            po = cute.make_rmem_tensor(T, Float32)
            for t in cutlass.range_constexpr(T):
                po[t] = Float32(0)
            if cutlass.const_expr("solve" not in self.skip):
                self.solve(values, b, betas, k, c, v_s, ce_s,
                           args[18][rec, head, None, None], po)  # fmt: skip
            pg = [Float32(g_s[b, t, c]) for t in range(T)]
            sums = psum2[p, None, None]
            terms = [Float32(0) for _ in range(4)]
            for t in cutlass.range_constexpr(T):
                terms[t] = po[t] * po[t]
            self.block_sums(terms, sums, c, 4)
            self.tail_sync()  # the squares' partials; every row sum and block read
            if c == 0:
                cute.arch.mbarrier_arrive(dots_empty + p)
                cute.arch.mbarrier_arrive(ops_empty + b)
            for t in cutlass.range_constexpr(T):
                sumsq = self.quad_sum(sums[t, None])
                variance = sumsq * (1.0 / 128) + eps
                rstd = reciprocal(sqrt_approx(variance))
                sigmoid_gate = triton_sigmoid(pg[t])
                result = po[t] * rstd * nw * sigmoid_gate
                self.store_output(args, token + t, head, c, BFloat16(result))
        else:
            if c == 0:
                cute.arch.mbarrier_arrive(dots_empty + p)
                cute.arch.mbarrier_arrive(ops_empty + b)
            for t in cutlass.range_constexpr(T):
                self.store_output(args, token + t, head, c, BFloat16(0))

    # ------------------------------------------------------------------- scan
    @cute.jit
    def scan_side(self, args, c, count, pages, table, ops, rsum, barriers):
        """The CTA's items in order (thread c of the scan warps; module docstring)."""
        state_full, ops_full, _, dots_full, dots_empty, _ = barriers
        xv, v_s, g_s, fw_s, fu_s, fd_s, ce_s = ops
        state = args[9]
        T: cutlass.Constexpr = self.tokens
        NV: cutlass.Constexpr = 2 * T
        lane, warp = c % 32, c // 32
        lane_in, base = c % LANES, (c // LANES) * QROWS  # rows base + [0, 4)
        tr = zeros(8, Int64)  # TRACE: ops wait, state wait, copy, math, empty wait, reduce
        if cutlass.const_expr(self.trace):
            tr[7] = clock64()
        for j in cutlass.range(count, unroll=1):
            s, b, p = j % SLOTS, j % NOPS, j % 2
            slot, head, pending = table[j, SLOT], table[j, HEAD], table[j, PENDING]
            ta = Int64(0)
            if cutlass.const_expr(self.trace):
                ta = clock64()
            self.bwait(ops_full + b, (j // NOPS) % 2, "ops_full")
            if cutlass.const_expr(self.trace):
                tb = clock64()
                tr[0] = tr[0] + (tb - ta)
                ta = tb
            self.bwait(state_full + s, (j // SLOTS) % 2, "state_full")
            if cutlass.const_expr(self.trace):
                tb = clock64()
                tr[1] = tr[1] + (tb - ta)
                ta = tb
            acc = cute.make_rmem_tensor((QROWS, NV), Float32)
            for rr in cutlass.range_constexpr(QROWS):
                for m in cutlass.range_constexpr(NV):
                    acc[rr, m] = Float32(0)
            if slot > 0:
                stored = pending > 0
                if cutlass.const_expr("store" in self.skip):
                    stored = pending < 0
                sts = cute.make_rmem_tensor((CHUNKS, QROWS, 4), Float32)
                for ch in cutlass.range_constexpr(CHUNKS):
                    key = 4 * (lane_in + LANES * ch)
                    for rr in cutlass.range_constexpr(QROWS):
                        sv = load_shared_f32x4(at(pages, s, (base + rr) * K + key))
                        for i in cutlass.range_constexpr(4):
                            sts[ch, rr, i] = sv[i]
                # The warp's eighth of the slot is in registers: refill it at once.
                fence_async_shared()
                cute.arch.sync_warp()
                if lane == 0:  # noqa: SIM102
                    if j + SLOTS < count:
                        self.load_eighth(args, pages, s, table, j + SLOTS, state_full,
                                         warp)  # fmt: skip
                cute.arch.sync_warp()
                if cutlass.const_expr(self.trace):
                    tb = clock64()
                    tr[2] = tr[2] + (tb - ta)
                    ta = tb
                if cutlass.const_expr("scan" not in self.skip):
                    self.fold_dots(acc, sts, b, pending, lane_in, base, xv, fw_s, fu_s,
                                   fd_s, state[Int64(slot), head, None, None],
                                   stored)  # fmt: skip
                if cutlass.const_expr(self.trace):
                    tb = clock64()
                    tr[3] = tr[3] + (tb - ta)
                    ta = tb
            else:
                if lane == 0:  # noqa: SIM102
                    if j + SLOTS < count:
                        self.load_eighth(args, pages, s, table, j + SLOTS, state_full,
                                         warp)  # fmt: skip
                cute.arch.sync_warp()
            if cutlass.const_expr(self.trace):
                ta = clock64()
            if j >= 2:
                self.bwait(dots_empty + p, (j // 2 - 1) % 2, "dots_empty")
            if cutlass.const_expr(self.trace):
                tb = clock64()
                tr[4] = tr[4] + (tb - ta)
                ta = tb
            if slot > 0:
                values = self.reduce4(acc, lane_in)
                if lane_in % 2 == 0:
                    row = base + lane_in // 2
                    for h in cutlass.range_constexpr(NV // 4):
                        store_shared_f32x4(at(rsum, p, row, 4 * h),
                                           [values[4 * h + i] for i in range(4)])  # fmt: skip
            cute.arch.sync_warp()
            if lane == 0:
                cute.arch.mbarrier_arrive(dots_full + p)
            if cutlass.const_expr(self.trace):
                tr[5] = tr[5] + (clock64() - ta)
        if cutlass.const_expr(self.trace):  # noqa: SIM102
            if c == 0:
                cta, _, _ = cute.arch.block_idx()
                if cta % 53 == 0:
                    cute.printf("RTRACE scan cta {} items {} ops-wait {} state-wait {} copy {} math {} empty-wait {} reduce {} total {}", cta, count, tr[0], tr[1], tr[2], tr[3], tr[4], tr[5], clock64() - tr[7])  # noqa: E501

    @cute.jit
    def fold_dots(self, acc, sts, b, pending, lane_in, base, xv, fw_s, fu_s, fd_s,
                  destination, stored):  # fmt: skip
        """KdaFlow.scan from registers over rows base + [0, 4) (keys 4 (lane_in + 8 j)
        + [0, 4)): S = fma(S_0, D_n, fc), fc = sum_t fma(u_t, w_t, fc) over
        t = q - 1 .. 0; the committed element leaves for ``destination``; acc[r, m] +=
        S[r, key] x_m[key] over the 2q scan vectors (per row: chunk, then key)."""
        T: cutlass.Constexpr = self.tokens
        NV: cutlass.Constexpr = 2 * T
        start = Float32(0)
        if pending == 0:
            start = Float32(-0.0)  # S + (-0) == S for every S, -0 included
        us = cute.make_rmem_tensor((QROWS, T), Float32)
        for t in cutlass.range_constexpr(T):
            u4 = load_shared_f32x4(at(fu_s, b, t, base))
            for rr in cutlass.range_constexpr(QROWS):
                us[rr, t] = u4[rr]
        for ch in cutlass.range_constexpr(CHUNKS):
            key = 4 * (lane_in + LANES * ch)
            d4 = load_shared_f32x4(at(fd_s, b, key))
            w4 = [load_shared_f32x4(at(fw_s, b, t, key)) for t in range(T)]
            for rr in cutlass.range_constexpr(QROWS):
                for p in cutlass.range_constexpr(2):
                    fc = (start, start)
                    for i in cutlass.range_constexpr(T):
                        t = T - 1 - i
                        fc = self.pfma((us[rr, t], us[rr, t]),
                                       (w4[t][2 * p], w4[t][2 * p + 1]), fc)  # fmt: skip
                    pair = self.pfma((sts[ch, rr, 2 * p], sts[ch, rr, 2 * p + 1]),
                                     (d4[2 * p], d4[2 * p + 1]), fc)  # fmt: skip
                    sts[ch, rr, 2 * p], sts[ch, rr, 2 * p + 1] = pair
                if stored:
                    store_global_cs(at(destination, base + rr, key),
                                    [sts[ch, rr, i] for i in range(4)])  # fmt: skip
            for e in cutlass.range_constexpr(4):
                xs = cute.make_rmem_tensor(NV, Float32)
                for h in cutlass.range_constexpr(NV // 4):
                    # swizzled: [chunk][e][h][lane][4] (lanes read 128 B in a row)
                    pos = (((ch * 4 + e) * (NV // 4) + h) * LANES + lane_in) * 4
                    x4 = load_shared_f32x4(at(xv, b, pos))
                    for i in cutlass.range_constexpr(4):
                        xs[4 * h + i] = x4[i]
                for rr in cutlass.range_constexpr(QROWS):
                    for p in cutlass.range_constexpr(NV // 2):
                        pair = (acc[rr, 2 * p], acc[rr, 2 * p + 1])
                        pair = self.pfma((sts[ch, rr, e], sts[ch, rr, e]),
                                         (xs[2 * p], xs[2 * p + 1]), pair)  # fmt: skip
                        acc[rr, 2 * p], acc[rr, 2 * p + 1] = pair

    @cute.jit
    def reduce4(self, acc, lane_in):
        """The 8 lanes' partials of rows base + [0, 4) -> lanes 2 r, 2 r + 1 hold the
        2q full sums of row base + r (KdaFlow's pairing: ^4, ^2, then ^1)."""
        NV: cutlass.Constexpr = 2 * self.tokens
        rows = [[acc[rr, m] for m in range(NV)] for rr in range(QROWS)]
        for stage in cutlass.range_constexpr(2):
            offset: cutlass.Constexpr = 4 >> stage
            half: cutlass.Constexpr = len(rows) // 2
            upper = (lane_in & offset) != 0
            merged = []
            for i in cutlass.range_constexpr(half):
                row = []
                for m in cutlass.range_constexpr(NV):
                    send = rows[i][m] if upper else rows[i + half][m]
                    keep = rows[i + half][m] if upper else rows[i][m]
                    row.append(keep + cute.arch.shuffle_sync_bfly(send, offset))
                merged.append(row)
            rows = merged
        return [v + cute.arch.shuffle_sync_bfly(v, 1) for v in rows[0]]
