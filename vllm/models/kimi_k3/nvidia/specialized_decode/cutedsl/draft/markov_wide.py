# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark greedy Markov drafting for wide batches (more than ``markov.MAX_WIDTH``
requests, up to ``MAX_WIDTH``): ``markov.MarkovGreedy``'s keys with a two-level
exchange.

Per position every CTA computes the keys of its 256 vocabulary rows for every request
(N tiles of 8 requests: the markov_w1 rows of the requests' previous tokens staged in
shared memory, the CTA's markov_w2 A fragments held in registers; the same MMAs,
roundings and keys as MarkovGreedy), reduces them over its warps and folds them into
this rank's maxima (``red.max`` on rank-local words). The CTA that arrives last (a
rank-local counter) multicasts the rank's keys into entry [rank, request] of every
rank's mailbox; every CTA polls the tp x width keys and takes the global maxima, the
next position's previous tokens. The drafts are bitwise MarkovGreedy's (and stock's).

``sampled``: probabilistic drafting as ``markov.MarkovGreedy`` ``sampled`` (the stock
Gumbel draw per logit, FP32-score keys, and the CTA's rows of each live request's
logits written to the draft-logits cache while the keys travel).

Buffers per parity (launch ``n`` uses parity ``n % 2`` from a per-rank ordinal and
first re-arms the other, as MarkovGreedy): the mailbox [position, rank, request] u64
keys (0 = unpublished) and, rank-local, the maxima [position, request] and one counter
per position.
"""

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
import torch.distributed._symmetric_memory as symm_mem
from cutlass import Float32, Int32, Int64, Uint16, Uint32, Uint64
from cutlass.cute.runtime import make_fake_stream

from ..common.gumbel import (
    DRAFT_NOISE_SALT,
    drafted_value,
    request_prefix,
    sampled_key,
    sampled_token,
)
from ..common.host import fake
from ..common.primitives import (
    add_rn,
    alloc,
    asm,
    bf16,
    bf16x2,
    bits_float,
    copy_g2s,
    launch,
    ldmatrix,
    load_words,
    mma,
    multicast_words,
    store_relaxed,
    store_words,
    zeros,
)
from .markov import (
    MAX_STEPS,
    PITCH,
    RANK,
    ROWS,
    THREADS,
    W1_PITCH,
    load_bf16_bits,
    make_key,
    sampled_fakes,
    shuffle_max,
)

MAX_WIDTH = 64  # requests (keys per mailbox entry), in N tiles of 8
# Rank-local u32 words per parity: the maxima, then one 16-byte counter per position.
LOCAL_WORDS = MAX_STEPS * MAX_WIDTH * 2 + MAX_STEPS * 4


def parity_bytes(tp):
    return MAX_STEPS * tp * MAX_WIDTH * 8


@asm()
def red_max(pointer, key: Uint64):
    return "red.relaxed.gpu.global.max.u64 [$0],$1;", "l,l,~{memory}", (pointer, key)


@asm()
def fence_gpu():
    return "fence.acq_rel.gpu;", "~{memory}", ()


@asm(Uint32)
def arrive(pointer):
    """Relaxed GPU-scope ``atom.add.u32`` of 1: the count before."""
    return "atom.relaxed.gpu.global.add.u32 $0,[$1],1;", "=r,l,~{memory}", (pointer,)


@asm(Uint64)
def load_key(pointer):
    return "ld.relaxed.gpu.global.u64 $0,[$1];", "=l,l,~{memory}", (pointer,)


class MarkovWide:
    """``width`` request slots (a multiple of 8; the batch is the anchors' length),
    ``steps`` draft positions, ``tp`` vocabulary shards of ``vocab_rank`` rows. Args:
    markov_w2 rows of this rank [vocab_rank, 256], markov_w1 [V, 256], base logits
    [batch * steps, vocab_rank] (row = request * steps + step), anchors [batch] (int32,
    any stride), draft [batch, steps] int64 (any row stride), ordinal int32 [1], mailbox
    (u32 view of both parities), multicast address, rank-local words (u32, both
    parities), rank, stats."""

    def __init__(self, width, steps, tp, vocab_rank, spin_limit=0, sampled=False):
        assert width % 8 == 0 and 0 < width <= MAX_WIDTH and steps <= MAX_STEPS
        assert vocab_rank % ROWS == 0
        # ``spin_limit`` > 0 (tests only): a poll gives up after that many rounds and
        # records the unpublished chunk mask in ``stats`` instead of hanging.
        self.spin_limit = spin_limit
        self.sampled = sampled
        self.width, self.steps, self.tp, self.vocab_rank = width, steps, tp, vocab_rank
        self.tiles = width // 8
        self.ctas = vocab_rank // ROWS
        self.box_words = parity_bytes(tp) // 4
        self.chunks = tp * width // 2  # 16-byte poll chunks (two keys) per position
        self.per_thread = (self.chunks + THREADS - 1) // THREADS

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, self.ctas, THREADS)

    @cute.kernel
    def kernel(self, args):
        w2, w1, base, anchors, draft, ordinal, box, mc, local, rank, stats = args[:11]
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        lane, warp = tid % 32, tid // 32
        group, quarter = lane // 4, lane % 4
        W = self.width
        batch = Int32(anchors.shape[0])
        smem = cutlass.utils.SmemAllocator()
        stage = alloc(smem, cutlass.Uint8, ROWS * PITCH, 128)
        w1s = alloc(smem, cutlass.Uint8, W * W1_PITCH, 128)
        best = alloc(smem, Uint64, (16, W), 16)  # this CTA's keys per warp
        keys_all = alloc(smem, Uint64, self.tp * W, 16)  # every rank's keys
        previous = alloc(smem, Int32, W, 16)
        last = alloc(smem, Int32, 4, 16)
        cached = slots_s = temps_s = seeds_s = prefixes_s = None
        if cutlass.const_expr(self.sampled):
            # This position's pre-temperature logits of the CTA's rows per request;
            # per request: slot (-1: padding), temperature, seed, this position's
            # murmur3 prefix.
            cached = alloc(smem, Uint16, (W, ROWS), 16)
            slots_s = alloc(smem, Int32, W, 16)
            temps_s = alloc(smem, Float32, W, 16)
            seeds_s = alloc(smem, Int64, W, 16)
            prefixes_s = alloc(smem, Uint32, W, 16)
        first = cta * ROWS
        # Static weights before the PDL wait: this CTA's markov_w2 rows.
        for c in range(tid, ROWS * 32, THREADS):
            r, j = c // 32, c % 32
            copy_g2s(
                stage.iterator + r * PITCH + j * 16,
                w2.iterator + (first + r) * RANK + j * 8,
                evict_first=True,
            )
        cute.arch.cp_async_commit_group()
        # Padding requests' markov_w1 rows stay zero (live rows are rewritten).
        for c in range(tid, W * W1_PITCH // 16, THREADS):
            store_words(w1s.iterator + 16 * c, (Uint32(0),) * 4, shared=True)
        launch_ordinal = ordinal[0]
        parity = launch_ordinal & 1
        # Re-arm the other parity (consumed by the previous launch), here and local.
        other = box.iterator + (1 - parity) * self.box_words
        for c in range(cta * THREADS + tid, self.box_words // 4, self.ctas * THREADS):
            store_relaxed(other + c * 4, (Uint32(0),) * 4)
        other_local = local.iterator + (1 - parity) * LOCAL_WORDS
        for c in range(cta * THREADS + tid, LOCAL_WORDS // 4, self.ctas * THREADS):
            store_words(other_local + c * 4, (Uint32(0),) * 4, scope="gpu")
        # No early trigger: a dependent launch of this kernel would re-arm and read the
        # ordinal before this launch has consumed its parity.
        cute.arch.griddepcontrol_wait()
        if tid < W:
            anchor = Int32(0)
            if tid < batch:
                anchor = Int32(anchors[tid])
            previous[tid] = anchor
            if cutlass.const_expr(self.sampled):
                temperature, seeds, slots = args[11], args[12], args[13]
                slot = Int32(-1)
                if tid < batch:
                    slot = Int32(slots[tid * self.steps])
                temp, seed = Float32(0.0), Int64(0)
                if slot >= Int32(0):
                    temp, seed = Float32(temperature[slot]), Int64(seeds[slot])
                slots_s[tid], temps_s[tid], seeds_s[tid] = slot, temp, seed
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        # The CTA's markov_w2 A fragments, every position's.
        tile = stage.iterator + (warp * 16 + lane % 8 + 8 * ((lane // 8) % 2)) * PITCH
        tile = tile + 16 * (lane // 16)
        a = cute.make_rmem_tensor((16, 4), Uint32)
        for s in cutlass.range_constexpr(16):
            frag = ldmatrix(tile + 32 * s)
            for k in cutlass.range_constexpr(4):
                a[s, k] = frag[k]
        rows = (first + warp * 16 + group, first + warp * 16 + group + 8)
        maxima = local.iterator + parity * LOCAL_WORDS
        counters = maxima + MAX_STEPS * MAX_WIDTH * 2
        region0 = box.iterator + parity * self.box_words
        for i in cutlass.range(self.steps, unroll=1):  # one copy of the position body
            if cutlass.const_expr(self.sampled):  # noqa: SIM102
                if tid < W:
                    position = Int64(0)
                    if tid < batch:
                        position = Int64(args[14][tid * self.steps + i])
                    # gumbel_sample's position: the predicted token's - 1, salted.
                    position = position - Int64(1) + Int64(DRAFT_NOISE_SALT)
                    prefixes_s[tid] = request_prefix(seeds_s[tid], position)
            # This position's base logits: rows (group, group + 8) of the warp tile,
            # requests 8 t + 2 quarter + (0, 1) of every N tile t.
            logits0 = cute.make_rmem_tensor((self.tiles, 4), Uint16)
            logits0.fill(0)
            for t in cutlass.range_constexpr(self.tiles):
                for n in cutlass.range_constexpr(2):
                    request = 8 * t + 2 * quarter + n
                    if request < batch:
                        row = request * self.steps + i
                        for r in cutlass.range_constexpr(2):
                            pointer = base.iterator + row * self.vocab_rank + rows[r]
                            logits0[t, 2 * r + n] = load_bf16_bits(pointer)
            # markov_w1 rows of the live requests' previous tokens.
            for c in range(tid, W * 32, THREADS):
                r, j = c // 32, c % 32
                if r < batch:
                    source = w1.iterator + Int64(previous[r]) * RANK + j * 8
                    copy_g2s(w1s.iterator + r * W1_PITCH + j * 16, source)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
            for t in cutlass.range_constexpr(self.tiles):
                b = cute.make_rmem_tensor((16, 2), Uint32)
                matrix_rows = w1s.iterator + (8 * t + lane % 8) * W1_PITCH
                matrix_rows = matrix_rows + 16 * (lane // 8)
                for q in cutlass.range_constexpr(8):
                    frag = ldmatrix(matrix_rows + 64 * q)
                    b[2 * q, 0], b[2 * q, 1] = frag[0], frag[1]
                    b[2 * q + 1, 0], b[2 * q + 1, 1] = frag[2], frag[3]
                acc = zeros(4, Float32)
                for s in cutlass.range_constexpr(16):
                    result = mma(
                        tuple(a[s, k] for k in range(4)),
                        (b[s, 0], b[s, 1]),
                        tuple(acc[k] for k in range(4)),
                        bf16=True,
                    )
                    for k in cutlass.range_constexpr(4):
                        acc[k] = result[k]
                keys = cute.make_rmem_tensor(2, Uint64)
                keys.fill(0)
                for r in cutlass.range_constexpr(2):
                    bias = bf16x2(acc[2 * r], acc[2 * r + 1])
                    for n in cutlass.range_constexpr(2):
                        base_bits = Uint32(logits0[t, 2 * r + n]) << 16
                        total = add_rn(bf16(bias, n == 1), bits_float(base_bits))
                        packed = bf16x2(total, total)
                        logit = bf16(packed, False)
                        index = rank * self.vocab_rank + rows[r]
                        if cutlass.const_expr(self.sampled):
                            key = sampled_key(
                                drafted_value(
                                    logit,
                                    temps_s[8 * t + 2 * quarter + n],
                                    prefixes_s[8 * t + 2 * quarter + n],
                                    index,
                                ),
                                index,
                            )
                            cached[
                                8 * t + 2 * quarter + n, warp * 16 + group + 8 * r
                            ] = Uint16(packed & Uint32(0xFFFF))
                        else:
                            key = make_key(logit, index)
                        keys[n] = max(keys[n], key)
                for offset in (4, 8, 16):
                    for n in cutlass.range_constexpr(2):
                        keys[n] = shuffle_max(keys[n], offset)
                if group == 0:
                    best[warp, 8 * t + 2 * quarter] = keys[0]
                    best[warp, 8 * t + 2 * quarter + 1] = keys[1]
            cute.arch.sync_threads()
            # This rank's maxima; the last CTA of the rank publishes them.
            if tid < W:
                mine = best[0, tid]
                for w in cutlass.range_constexpr(1, 16):
                    mine = max(mine, best[w, tid])
                red_max(maxima + (i * MAX_WIDTH + tid) * 2, mine)
            cute.arch.sync_threads()
            if tid == 0:
                fence_gpu()
                count = arrive(counters + i * 4)
                fence_gpu()
                flag = Int32(0)
                if count == Uint32(self.ctas - 1):
                    flag = Int32(1)
                last[0] = flag
            cute.arch.sync_threads()
            region = region0 + i * self.tp * W * 2
            if (last[0] != Int32(0)) & (tid < W):
                key = load_key(maxima + (i * MAX_WIDTH + tid) * 2)
                words = parity * self.box_words + i * self.tp * W * 2
                words = words + (rank * W + tid) * 2
                multicast_words(
                    mc + Int64(words) * 4,
                    (Uint32(key & Uint64(0xFFFFFFFF)), Uint32(key >> 32)),
                    # A bit-exact store (multimem.st vectors need a float type).
                    kind="f32",
                )
            if cutlass.const_expr(self.sampled):
                # While the keys travel: the position's logits into the cache.
                cache = args[15]
                vocab = self.tp * self.vocab_rank
                for c in range(tid, batch * (ROWS // 8), THREADS):
                    request = c // (ROWS // 8)
                    chunk = c % (ROWS // 8)
                    slot = slots_s[request]
                    if slot >= Int32(0):
                        words = load_words(
                            cached.iterator + request * ROWS + 8 * chunk,
                            count=4,
                            shared=True,
                        )
                        row = Int64(slot) * self.steps + i
                        column = rank * self.vocab_rank + first + 8 * chunk
                        target = cache.iterator + (row * vocab + Int64(column))
                        store_words(target, words)
            # Every rank's keys of this position.
            got = cute.make_rmem_tensor((self.per_thread, 4), Uint32)
            got.fill(0)
            pending = Uint32(0)
            for j in cutlass.range_constexpr(self.per_thread):
                if tid + j * THREADS < self.chunks:
                    pending |= Uint32(1 << j)
            spins = Int32(0)
            while pending != Uint32(0):
                spins += 1
                if cutlass.const_expr(self.spin_limit > 0):  # noqa: SIM102
                    if spins > self.spin_limit:
                        stats[cta, i] = Int64(0x7FFF0000) | Int64(pending)
                        pending = Uint32(0)
                loads = [
                    load_words(
                        region + min(tid + j * THREADS, self.chunks - 1) * 4,
                        count=4,
                        volatile=True,
                    )
                    for j in range(self.per_thread)
                ]
                for j in cutlass.range_constexpr(self.per_thread):
                    if (pending >> j) & Uint32(1):
                        values = loads[j]
                        ready = True
                        for k in cutlass.range_constexpr(4):
                            got[j, k] = values[k]
                            ready &= values[k] != Uint32(0)
                        if ready:
                            pending &= ~Uint32(1 << j)
            for j in cutlass.range_constexpr(self.per_thread):
                c = tid + j * THREADS
                if c < self.chunks:
                    keys_all[2 * c] = (Uint64(got[j, 1]) << 32) | Uint64(got[j, 0])
                    keys_all[2 * c + 1] = (Uint64(got[j, 3]) << 32) | Uint64(got[j, 2])
            cute.arch.sync_threads()
            if tid < W:
                final = keys_all[tid]
                for s in cutlass.range_constexpr(1, self.tp):
                    final = max(final, keys_all[s * W + tid])
                token = Int32(Uint32(0xFFFFFFFF) - Uint32(final & Uint64(0xFFFFFFFF)))
                if cutlass.const_expr(self.sampled):
                    token = sampled_token(final)
                if (cta == 0) & (tid < batch):
                    draft[tid, i] = Int64(token)
                if cutlass.const_expr(self.spin_limit > 0):  # noqa: SIM102
                    if (cta == 0) & (tid == 0):
                        stats[self.ctas, i] = Int64(final)
                previous[tid] = min(max(token, Int32(0)), Int32(w1.shape[0] - 1))
            cute.arch.sync_threads()
        if (cta == 0) & (tid == 0):
            ordinal[0] = launch_ordinal + 1


class MarkovWideBox:
    """Both parities' wide key mailboxes (symmetric, multicast), the rank-local maxima
    and counters, and the ordinal."""

    def __init__(self, group, device, tp):
        words = 2 * parity_bytes(tp) // 4
        self.buffer = symm_mem.empty((words,), dtype=torch.int32, device=device)
        self.handle = handle = symm_mem.rendezvous(self.buffer, group)
        self.mc = int(handle.multicast_ptr or 0)
        if not self.mc:
            raise RuntimeError("Kimi K3 DSpark Markov drafting needs multicast memory")
        self.buffer.zero_()
        self.local = torch.zeros(2 * LOCAL_WORDS, dtype=torch.int32, device=device)
        self.ordinal = torch.zeros(1, dtype=torch.int32, device=device)
        torch.accelerator.synchronize(device)
        torch.distributed.barrier(group=group)


@cache
def compile_markov_wide(
    width,
    steps,
    tp,
    vocab_rank,
    arch,
    anchor_stride,
    draft_stride,
    spin_limit=0,
    sampled=False,
):
    kernel = MarkovWide(width, steps, tp, vocab_rank, spin_limit, sampled)
    batch = cute.sym_int()
    args: tuple = (
        fake((vocab_rank, RANK)),
        fake((cute.sym_int(), RANK)),
        fake((cute.sym_int(), vocab_rank)),
        fake((batch,), (anchor_stride,), torch.int32, 4),
        fake((cute.sym_int(), steps), (draft_stride, 1), torch.int64, 8),
        fake((1,), None, torch.int32, 4),
        fake((cute.sym_int(),), None, torch.uint32, 16),
        Int64(0),
        fake((cute.sym_int(),), None, torch.uint32, 16),
        Int32(0),
        fake((vocab_rank // ROWS + 2, 64), None, torch.int64, 8),
    )
    if sampled:
        args += sampled_fakes(cute.sym_int(), steps, tp * vocab_rank)
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    options = f"--gpu-arch={arch} --enable-tvm-ffi"
    return cute.compile(kernel, *args, stream, options=options)
