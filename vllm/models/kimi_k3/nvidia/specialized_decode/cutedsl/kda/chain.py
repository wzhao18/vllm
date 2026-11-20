# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KdaFlow's record core as two kernels: a wide prep and a state-streaming scan
(KdaChain).

KdaFlow runs each item's prep (convolution, gates, decays, the fold's operands) on
the same latency-bound warp chain as its scan, so per SM two chains of prep + scan
(~11.6K cycles an item) outlast the state stream. Here the prep runs first as its
own grid at full occupancy, one CTA (one group of 128 threads) per item: its
operand block (the scan vectors, v, the fold operands, the couplings and the output
gates) goes to a global scratch row, its records and window straight to their
caches. The scan grid (KdaFlow's persistent CTAs, two groups) starts its state
pages before the PDL wait (the state does not depend on the prep), then per item
bulk-copies the operand block one item ahead and runs KdaFlow's scan, solve and
gated norm only.

The math is KdaFlow's, op for op: the same prep (its code), the same gate
projection MMAs, the same scan / solve / gated norm; only the gated norm's block
reduction is its own (``finish_alone``, bit-identical sums).
"""

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64

from ..common.primitives import (
    alloc,
    asm,
    at,
    copy_g2s,
    launch,
    load_words,
    mma_into,
    zeros,
)
from .flow import (
    CONSTS,
    CONV_OFFSET,
    CONV_SLOT,
    GROUP_THREADS,
    GROUPS,
    HEAD,
    LANES,
    MAX_ITEMS,
    META,
    PENDING,
    RECORD,
    REQUEST,
    ROWS,
    SLOT,
    SUMS,
    THREADS,
    K,
    KdaFlow,
    bulk_commit,
    store_bulk,
)
from .recover import triton_sigmoid
from .specialized import (
    COPY_BYTES,
    PAGE_BYTES,
    fence_async_shared,
    load_shared_u32x2,
    wait_bulk_reads,
)

# Operand block (FP32 words; offsets for up to 4 tokens): the 2q scan vectors, v,
# the fold weights w_t, the old corrections u_t, D_n, the couplings, the BF16 output
# gates. KdaFlow's shared-memory image, one global scratch row per item.
XV, V, FW, FU, FD, CE, GATES = 0, 1024, 1536, 2048, 2560, 2688, 2704
BLOCK_BYTES = (GATES + 4 * K // 2) * 4
BLOCK = 3072  # words per scratch row
CHAIN_SLOTS = 4  # the scan CTA's state page ring (no prep staging: one more slot)


def scratch_shape(items):
    return (items, BLOCK)


@asm()
def store_global_f32x4(pointer, values):
    """16 B register -> global store (default cache policy: the scan reads it)."""
    return "st.global.v4.f32 [$0], {$1, $2, $3, $4};", "l,f,f,f,f,~{memory}", (
        pointer,
        *values,
    )


def block_views(base, blocks, stride, T):
    """KdaFlow's operand tensors (xv, v, g, w, u, D, couplings) over ``blocks``
    operand blocks at FP32 pointer ``base``, ``stride`` words apart."""

    def f32(offset, shape, inner):
        layout = cute.make_layout((blocks, *shape), stride=(stride, *inner))
        return cute.make_tensor(base + offset, layout)

    xv = f32(XV, (2 * T * K,), (1,))
    v_s = f32(V, (T, K), (K, 1))
    fw_s = f32(FW, (T, K), (K, 1))
    fu_s = f32(FU, (T, K), (K, 1))
    fd_s = f32(FD, (K,), (1,))
    ce_s = f32(CE, (T * T,), (1,))
    g_s = cute.make_tensor(
        cute.recast_ptr(base + GATES, dtype=BFloat16),
        cute.make_layout((blocks, T, K), stride=(2 * stride, K, 1)),
    )
    return xv, v_s, g_s, fw_s, fu_s, fd_s, ce_s


class KdaChain(KdaFlow):
    """KdaFlow's operands plus a ``scratch_shape(items)`` FP32 operand-block scratch
    (the last argument); deferred records only."""

    def __init__(self, heads, tokens, gate_mode, has_bias, conv_dim_first, window,
                 ctas, items, stages=1):  # fmt: skip
        super().__init__(heads, tokens, gate_mode, has_bias, conv_dim_first, window,
                         ctas, stages)  # fmt: skip
        self.items = items
        self.vec_store = None

    def store_vec(self, pointer, values):
        if self.vec_store is None:
            super().store_vec(pointer, values)
        else:
            self.vec_store(pointer, values)

    @cute.jit
    def __call__(self, *args):
        if cutlass.const_expr("noprep" not in self.skip):
            launch(self.prep_kernel, args, self.items, GROUP_THREADS)
        if cutlass.const_expr("noscan" not in self.skip):
            launch(self.kernel, args, self.ctas, THREADS, min_blocks_per_mp=1)

    # ------------------------------------------------------------------- prep
    @cute.kernel
    def prep_kernel(self, args):
        self.prep_body(args)

    @cute.jit
    def prep_body(self, args):
        conv_state, f_a, f_b, scratch = args[3], args[4], args[16], args[22]
        T: cutlass.Constexpr = self.tokens
        tid, _, _ = cute.arch.thread_idx()
        item, _, _ = cute.arch.block_idx()
        conv_type: cutlass.Constexpr = conv_state.element_type
        smem = cutlass.utils.SmemAllocator()
        hc = alloc(smem, Float32, (CONSTS, K))
        raw = alloc(smem, BFloat16, (1, T, K), align=16)
        fas = alloc(smem, BFloat16, (T, K), align=128)
        table = alloc(smem, Int32, (1, META))
        convs = alloc(smem, conv_type, (1, 3, K * self.window), align=128)
        xs = alloc(smem, BFloat16, (T, 3, K), align=128)
        gs = alloc(smem, BFloat16, (T, K), align=128)
        kgs = alloc(smem, BFloat16, (T, 2 * K), align=128)
        uss = alloc(smem, Float32, (T, K), align=128)
        sums = alloc(smem, Float32, (SUMS, 4))
        bars = smem.allocate(cutlass.Uint64, 2)
        stage_full, fa_full = bars, bars + 1
        head, request, slot, conv_slot, conv_offset, pending, rec = self.item_info(
            args, item
        )
        if tid == 0:
            cute.arch.mbarrier_init(stage_full, 1)
            cute.arch.mbarrier_init(fa_full, 1)
            cute.arch.mbarrier_init_fence()
            table[0, SLOT] = slot
            table[0, HEAD] = head
            table[0, REQUEST] = request
            table[0, CONV_SLOT] = conv_slot
            table[0, CONV_OFFSET] = conv_offset
            table[0, PENDING] = pending
            table[0, RECORD] = rec
        self.head_constants(hc, args, head, tid)
        smem_tensors = (hc, raw, fas, table, convs, xs, gs, kgs, uss, sums, stage_full,
                        fa_full)  # fmt: skip
        cute.arch.sync_warp()
        cute.arch.barrier()
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        self.vec_store = store_global_f32x4  # trace time: the prep's xv stores
        self.prep_items(args, tid, item, slot, conv_slot, conv_offset, pending, rec,
                        head, request, smem_tensors)  # fmt: skip
        self.vec_store = None

    @cute.jit
    def prep_items(self, args, tid, item, slot, conv_slot, conv_offset, pending, rec,
                   head, request, smem_tensors):  # fmt: skip
        conv_state, f_a, f_b, scratch = args[3], args[4], args[16], args[22]
        T: cutlass.Constexpr = self.tokens
        conv_type: cutlass.Constexpr = conv_state.element_type
        hc, raw, fas, table, convs, xs, gs, kgs, uss, sums, stage_full, fa_full = (
            smem_tensors
        )
        if slot > 0:
            if tid == 0:
                cute.arch.mbarrier_arrive_and_expect_tx(fa_full, T * K * 2)
                for t in cutlass.range_constexpr(T):
                    copy_g2s(at(fas, t, 0), at(f_a, request * T + t, 0), K * 2,
                             fa_full)  # fmt: skip
            buffers = (convs, xs, gs, kgs, uss)
            self.stage_item(args, table, 0, buffers, stage_full, 0, tid)
            cute.arch.sync_warp()
            self.bwait(fa_full, 0, "fa_full")
            self.item_gates(tid, head, f_b, fas, raw)
            self.group_sync(0)  # the raw gates
            self.bwait(stage_full, 0, "stage_full")
            ops = block_views(at(scratch, item, 0), 1, BLOCK, T)
            kgo = cute.make_tensor(
                at(args[19], rec, head, 0, 0),
                cute.make_layout((T, 2 * K), stride=(2 * K, 1)),
            )
            po = zeros(T, Float32)
            pt = zeros(6, Int64)
            self.prep_item(args, 0, tid, 0, slot, pending, conv_offset, raw, hc,
                           (convs[0, None, None], xs, gs, kgs, uss), ops, (sums,),
                           kgo, pt, po, 0)  # fmt: skip
            if tid == 0:  # the window (fenced and synced by the prep)
                dim: cutlass.Constexpr = self.heads * K
                part_bytes: cutlass.Constexpr = K * self.window * conv_type.width // 8
                page = conv_state[Int64(conv_slot), None, None]
                for part in cutlass.range_constexpr(3):
                    store_bulk(at(page, part * dim + head * K, 0),
                               at(convs, 0, part, 0), part_bytes)  # fmt: skip
                bulk_commit()
                wait_bulk_reads()

    @cute.jit
    def item_gates(self, tid, head, f_b, fas, raw):
        """raw[0, t, f] = BF16(F_B[head] @ F_A[t]): ``gate_projection``'s MMA chain
        for one item (warp w: features 32 w + [0, 32)), F_B fragments from global."""
        T: cutlass.Constexpr = self.tokens
        lane, wq = tid % 32, tid // 32
        group_id, quarter = lane >> 2, lane & 3
        wr = cute.make_rmem_tensor((2, 8, 4), cutlass.Uint32)
        for m in cutlass.range_constexpr(2):
            for chunk in cutlass.range_constexpr(8):
                feature = head * K + 32 * wq + 16 * m + group_id
                col = 16 * chunk + 4 * quarter
                low = load_words(at(f_b, feature, col), count=2, word_type="u32")
                high = load_words(at(f_b, feature + 8, col), count=2, word_type="u32")
                wr[m, chunk, 0], wr[m, chunk, 2] = low
                wr[m, chunk, 1], wr[m, chunk, 3] = high
        cute.arch.sync_warp()  # converged for mma.sync
        for m in cutlass.range_constexpr(2):
            chains: cutlass.Constexpr = 4 if self.gate_mode == 1 else 1
            part = zeros((chains, 4), Float32)
            for chunk in cutlass.range_constexpr(8):
                chain: cutlass.Constexpr = chunk // 2 if self.gate_mode == 1 else 0
                wa = tuple(wr[m, chunk, e] for e in range(4))
                ab = load_shared_u32x2(at(fas, group_id % T, 16 * chunk + 4 * quarter))
                mma_into(part[chain, None], wa, ab, bf16=True)
            for e in cutlass.range_constexpr(4):
                value = part[0, e]
                if cutlass.const_expr(self.gate_mode == 1):
                    value = Float32(0) + (
                        ((value + part[1, e]) + part[2, e]) + part[3, e]
                    )
                t = 2 * quarter + e % 2
                feature = 32 * wq + 16 * m + group_id + 8 * (e // 2)
                if t < T:
                    raw[0, t, feature] = BFloat16(value)

    # ------------------------------------------------------------------- scan
    @cute.jit
    def body(self, args):
        beta, scratch = args[5], args[22]
        T: cutlass.Constexpr = self.tokens
        H: cutlass.Constexpr = self.heads
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        requests: cutlass.Constexpr = args[10].shape[0] // T
        total_items: cutlass.Constexpr = H * requests
        first = Int32(total_items * cta // self.ctas)
        count = Int32(total_items * (cta + 1) // self.ctas) - first
        smem = cutlass.utils.SmemAllocator()
        pages = alloc(smem, Float32, (CHAIN_SLOTS, K * K), align=1024)
        blocks = alloc(smem, Float32, (2 * GROUPS, BLOCK), align=128)
        table = alloc(smem, Int32, (MAX_ITEMS, META))
        betas = alloc(smem, Float32, (MAX_ITEMS, 8))
        sums = alloc(smem, Float32, (GROUPS, SUMS, 4))
        bars = smem.allocate(cutlass.Uint64, CHAIN_SLOTS + 2 * GROUPS)
        state_full, ops_full = bars, bars + CHAIN_SLOTS
        if tid == 0:
            for s in cutlass.range_constexpr(CHAIN_SLOTS):  # one arrival per warp quarter
                cute.arch.mbarrier_init(state_full + s, 1)
            for b in cutlass.range_constexpr(2 * GROUPS):
                cute.arch.mbarrier_init(ops_full + b, 1)
            cute.arch.mbarrier_init_fence()
        if tid < count:
            head, request, slot, conv_slot, conv_offset, pending, rec = self.item_info(
                args, first + tid
            )
            table[tid, SLOT] = slot
            table[tid, HEAD] = head
            table[tid, REQUEST] = request
            table[tid, PENDING] = pending
            table[tid, RECORD] = rec
        cute.arch.sync_warp()
        cute.arch.barrier()
        if tid == 0:  # the state does not depend on the prep: stream before the wait
            for s in cutlass.range_constexpr(CHAIN_SLOTS):
                if s < count:
                    self.load_page(args, pages, s, table, s, state_full)
        cute.arch.sync_warp()
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        if tid < count * T:
            item, t = tid // T, tid % T
            row = table[item, REQUEST] * T + t
            betas[item, t] = triton_sigmoid(Float32(beta[row, table[item, HEAD]]))
        g = tid // GROUP_THREADS
        c = tid % GROUP_THREADS
        if g < count:
            self.load_block(scratch, blocks, table, first, g, 2 * g, ops_full, c)
        cute.arch.sync_warp()
        cute.arch.barrier()  # betas
        ops = block_views(at(blocks, 0, 0), 2 * GROUPS, BLOCK, T)
        self.scan_items(args, g, c, count, pages, table, betas, blocks, ops,
                        sums[g, None, None], state_full, ops_full, first)  # fmt: skip

    @cute.jit
    def load_page(self, args, pages, s, table, j, state_full):
        """Table item j's whole state page into page slot s: ONE 64 KiB bulk copy
        (a single TMA op of this size streams ~3x a 16 KiB op per issuing thread)."""
        slot, head = table[j, SLOT], table[j, HEAD]
        barrier = state_full + s
        if slot > 0:
            cute.arch.mbarrier_arrive_and_expect_tx(barrier, PAGE_BYTES)
            source = args[9][Int64(slot), None, None, None]
            copy_g2s(at(pages, s, 0), at(source, head, 0, 0), PAGE_BYTES, barrier)
        else:
            cute.arch.mbarrier_arrive_and_expect_tx(barrier, 0)

    @cute.jit
    def release_slot(self, args, c, j, count, pages, table, state_full, stored):
        if c == 0:
            if j + CHAIN_SLOTS < count:
                self.load_page(args, pages, j % CHAIN_SLOTS, table, j + CHAIN_SLOTS,
                               state_full)  # fmt: skip
        cute.arch.sync_warp()

    @cute.jit
    def load_block(self, scratch, blocks, table, first, j, b, ops_full, c):
        """Item j's operand block into block buffer ``b`` (group thread 0)."""
        if c == 0:
            barrier = ops_full + b
            if table[j, SLOT] > 0:
                cute.arch.mbarrier_arrive_and_expect_tx(barrier, BLOCK_BYTES)
                copy_g2s(at(blocks, b, 0), at(scratch, first + j, 0), BLOCK_BYTES,
                         barrier)  # fmt: skip
            else:
                cute.arch.mbarrier_arrive_and_expect_tx(barrier, 0)
        cute.arch.sync_warp()

    @cute.jit
    def scan_items(self, args, g, c, count, pages, table, betas, blocks, ops, sums_g,
                   state_full, ops_full, first):  # fmt: skip
        xv, v_s, g_s, fw_s, fu_s, fd_s, ce_s = ops
        state, output = args[9], args[10]
        T: cutlass.Constexpr = self.tokens
        NV: cutlass.Constexpr = 2 * T
        lane, warp = c % 32, c // 32
        lane_in, base = c % LANES, (c // LANES) * ROWS
        own = base + lane_in
        nw = args[12][c]
        mine = (count - g + 1) // 2  # this group's items: g, g + 2, ...
        po = cute.make_rmem_tensor(T, Float32)
        pg = cute.make_rmem_tensor(T, Float32)
        pm = cute.make_rmem_tensor(3, Int32)
        for jj in cutlass.range(mine, unroll=1):
            j = g + GROUPS * jj
            b = 2 * g + jj % 2
            slot, head = table[j, SLOT], table[j, HEAD]
            pending, rec = table[j, PENDING], table[j, RECORD]
            token = table[j, REQUEST] * T
            if jj + 1 < mine:  # the other buffer's item finished (group-synced)
                if c == 0:
                    fence_async_shared()
                self.load_block(args[22], blocks, table, first, j + GROUPS,
                                2 * g + (jj + 1) % 2, ops_full, c)  # fmt: skip
            self.bwait(ops_full + b, (jj // 2) % 2, "ops_full")
            s = j % CHAIN_SLOTS
            if slot > 0:
                for t in cutlass.range_constexpr(T):
                    pg[t] = Float32(g_s[b, t, c])
                self.bwait(state_full + s, (j // CHAIN_SLOTS) % 2, "state_full")
                acc = cute.make_rmem_tensor((ROWS, NV), Float32)
                destination = state[Int64(slot), head, None, None]
                self.scan(acc, b, s, pages, pending, lane_in, base, xv, fw_s, fu_s,
                          fd_s, destination, pending > 0)  # fmt: skip
                fence_async_shared()  # the slot's generic reads before its refill
                values = self.rows_reduce(acc, lane_in)
                self.solve(values, b, betas, j, own, v_s, ce_s,
                           args[18][rec, head, None, None], po)  # fmt: skip
                pm[0], pm[1], pm[2] = Int32(1), token, head
                self.finish_alone(args, g, c, sums_g, po, pg, pm, nw)
                if c == 0:  # the group's scan reads of slot s are done (group-synced)
                    if j + CHAIN_SLOTS < count:
                        self.load_page(args, pages, s, table, j + CHAIN_SLOTS,
                                       state_full)  # fmt: skip
                cute.arch.sync_warp()
            else:
                self.release_slot(args, c, j, count, pages, table, state_full, False)
                for t in cutlass.range_constexpr(T):
                    output[token + t, head * K + c] = BFloat16(0)
