# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark greedy Markov drafting for wide batches (more than ``markov.MAX_WIDTH``
requests, up to ``MAX_WIDTH``): ``markov.MarkovGreedy``'s keys with a two-level
exchange.

Per position every CTA computes the keys of its vocabulary rows for every request (N
tiles of 8 requests: the markov_w1 rows of the requests' previous tokens staged in
shared memory; the same MMAs, roundings and keys as MarkovGreedy: at Markov rank 256
the CTA's markov_w2 A fragments held in registers, at rank 512 (128 rows per CTA,
``markov.CTA_ROWS``) read from the staged rows per N tile, warps 0-7 on the even N
tiles and warps 8-15 on the odd ones, each over all 32 k-steps in column order),
reduces them over its warps and folds them into
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
per position. Consecutive launches (``markov.PreparedMarkov.plan``: a batch above
``launch_width`` runs as launches over request slices) alternate parities the same
way on every rank.

``chained``: a launch that directly follows another MarkovWide launch on the stream
(the request slices after the first). A launch normally reads the ordinal and re-arms
the other parity before its PDL wait, which is safe only because the kernel before it
(the LM head) has itself waited on the previous MarkovWide launch. Back to back, nothing
orders launch ``n``'s ordinal increment and parity reads before launch ``n + 1``'s
pre-wait reads and re-arm (a stale ordinal would reuse launch ``n``'s parity with its
keys still published: stale drafts, or ranks on different parities: a hang), so a
chained launch reads the ordinal and re-arms after ``griddepcontrol_wait`` (launch
``n`` complete and its writes visible). Only the static markov_w2 staging still
overlaps the previous launch's tail.
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
    CTA_ROWS,
    MAX_STEPS,
    RANK,
    THREADS,
    WARPS,
    load_bf16_bits,
    make_key,
    resident,
    row_pitch,
    sampled_fakes,
    shuffle_max,
)

# requests (keys per mailbox entry), in N tiles of 8. 128 (hb192: DL3 B65-B128): the
# sampled kernel's shared memory is ~135 KB + ~1.3 KB per request slot, ~304 KB at 128
# (SM107's oversized opt-in: 334848 B); 192 would not fit. Per launch at most
# ``launch_width`` slots (the device's opt-in shared memory, the Markov rank); wider
# batches run as several launches (markov.PreparedMarkov.plan).
MAX_WIDTH = 128
# Rank-local u32 words per parity: the maxima, then one 16-byte counter per position.
LOCAL_WORDS = MAX_STEPS * MAX_WIDTH * 2 + MAX_STEPS * 4


def wide_smem(width, tp, rank=RANK, sampled=True):
    """Upper bound of MarkovWide's shared memory (bytes) for ``width`` request slots at
    Markov ``rank``: the staged markov_w2 rows, then per slot its staged markov_w1 row,
    16 warps' keys, every rank's key, the previous token (and, sampled, the CTA's
    logits row, slot, temperature, seed and murmur3 prefix)."""
    rows, pitch = CTA_ROWS[rank], row_pitch(rank)
    per_slot = pitch + WARPS * 8 + tp * 8 + 4
    if sampled:
        per_slot += rows * 2 + 4 + 4 + 8 + 4
    return rows * pitch + width * per_slot + 16 + 8 * 128  # (+ the alignments)


def launch_width(tp, rank, sampled, limits, vocab_rank):
    """Request slots of one MarkovWide launch: the largest multiple of 8 up to MAX_WIDTH
    whose shared memory fits the device and whose CTAs (``vocab_rank`` rows,
    markov.CTA_ROWS per CTA) are co-resident; 0 if none (``limits``:
    markov.device_limits). Rank 256 at 227 KiB: 72 (sampled); SM107's 334848 B: 128.
    NQ-TUNE: rank 512 per-launch width vs launches (one launch of 120 vs two of 64)."""
    ctas = vocab_rank // CTA_ROWS[rank]
    for width in range(MAX_WIDTH, 0, -8):
        if resident(ctas, wide_smem(width, tp, rank, sampled), limits):
            return width
    return 0


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
    ``steps`` draft positions, ``tp`` vocabulary shards of ``vocab_rank`` rows, Markov
    ``rank``. Args: markov_w2 rows of this rank [vocab_rank, rank], markov_w1 [V,
    rank], base logits
    [batch * steps, vocab_rank] (row = request * steps + step), anchors [batch] (int32,
    any stride), draft [batch, steps] int64 (any row stride), ordinal int32 [1], mailbox
    (u32 view of both parities), multicast address, rank-local words (u32, both
    parities), rank, stats."""

    def __init__(
        self,
        width,
        steps,
        tp,
        vocab_rank,
        spin_limit=0,
        sampled=False,
        rank=RANK,
        chained=False,
    ):
        assert width % 8 == 0 and 0 < width <= MAX_WIDTH and steps <= MAX_STEPS
        # Markov rank: CTA rows, staged row pitch; rank 256 holds the CTA's A fragments
        # (16 k-steps) in registers, rank 512 (32 k-steps) reads them per N tile, its
        # 8 tile warps on N tiles t with t % 2 == warp // 8 (``halves``).
        self.markov_rank, self.rows = rank, CTA_ROWS[rank]
        self.pitch, self.k_halves = row_pitch(rank), rank // 256
        self.row_warps = self.rows // 16
        self.halves = WARPS // self.row_warps
        self.a_registers = rank == 256
        assert vocab_rank % self.rows == 0 and rank % 256 == 0
        # ``spin_limit`` > 0 (tests only): a poll gives up after that many rounds and
        # records the unpublished chunk mask in ``stats`` instead of hanging.
        self.spin_limit = spin_limit
        self.sampled = sampled
        self.chained = chained  # (follows a MarkovWide launch: see the module notes)
        self.width, self.steps, self.tp, self.vocab_rank = width, steps, tp, vocab_rank
        self.tiles = width // 8
        self.ctas = vocab_rank // self.rows
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
        R, P = self.markov_rank, self.pitch  # Markov rank, staged row pitch (bytes)
        batch = Int32(anchors.shape[0])
        smem = cutlass.utils.SmemAllocator()
        stage = alloc(smem, cutlass.Uint8, self.rows * P, 128)
        w1s = alloc(smem, cutlass.Uint8, W * P, 128)
        best = alloc(smem, Uint64, (16, W), 16)  # this CTA's keys per warp
        keys_all = alloc(smem, Uint64, self.tp * W, 16)  # every rank's keys
        previous = alloc(smem, Int32, W, 16)
        last = alloc(smem, Int32, 4, 16)
        cached = slots_s = temps_s = seeds_s = prefixes_s = None
        if cutlass.const_expr(self.sampled):
            # This position's pre-temperature logits of the CTA's rows per request;
            # per request: slot (-1: padding), temperature, seed, this position's
            # murmur3 prefix.
            cached = alloc(smem, Uint16, (W, self.rows), 16)
            slots_s = alloc(smem, Int32, W, 16)
            temps_s = alloc(smem, Float32, W, 16)
            seeds_s = alloc(smem, Int64, W, 16)
            prefixes_s = alloc(smem, Uint32, W, 16)
        first = cta * self.rows
        # Static weights before the PDL wait: this CTA's markov_w2 rows.
        pieces = R // 8  # 16 B chunks per row
        for c in range(tid, self.rows * pieces, THREADS):
            r, j = c // pieces, c % pieces
            copy_g2s(
                stage.iterator + r * P + j * 16,
                w2.iterator + (first + r) * R + j * 8,
                evict_first=True,
            )
        cute.arch.cp_async_commit_group()
        # Padding requests' markov_w1 rows stay zero (live rows are rewritten).
        for c in range(tid, W * P // 16, THREADS):
            store_words(w1s.iterator + 16 * c, (Uint32(0),) * 4, shared=True)
        launch_ordinal = Int32(0)
        if cutlass.const_expr(not self.chained):
            launch_ordinal = self.rearm(ordinal, box, local, cta, tid)
        # No early trigger: a dependent launch of this kernel would re-arm and read the
        # ordinal before this launch has consumed its parity.
        cute.arch.griddepcontrol_wait()
        if cutlass.const_expr(self.chained):
            # (the previous MarkovWide launch is complete and its ordinal visible)
            launch_ordinal = self.rearm(ordinal, box, local, cta, tid)
        parity = launch_ordinal & 1
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
        # The warp's 16-row tile of the staged rows (rank 512: warps w and w + 8 share
        # tile w, on alternate N tiles).
        tile_warp = warp % self.row_warps
        tile = stage.iterator + (tile_warp * 16 + lane % 8 + 8 * ((lane // 8) % 2)) * P
        tile = tile + 16 * (lane // 16)
        a = None
        if cutlass.const_expr(self.a_registers):
            # The CTA's markov_w2 A fragments, every position's.
            a = cute.make_rmem_tensor((16, 4), Uint32)
            for s in cutlass.range_constexpr(16):
                frag = ldmatrix(tile + 32 * s)
                for k in cutlass.range_constexpr(4):
                    a[s, k] = frag[k]
        rows = (first + tile_warp * 16 + group, first + tile_warp * 16 + group + 8)
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
                    # spec_step_seed(seed, i): gumbel_sample's seed for cache
                    # column i (the draft position).
                    step_seed = seeds_s[tid] ^ (Int64(i) << 32)
                    prefixes_s[tid] = request_prefix(step_seed, position)
            # This position's base logits: rows (group, group + 8) of the warp tile,
            # requests 8 t + 2 quarter + (0, 1) of every N tile t.
            logits0 = cute.make_rmem_tensor((self.tiles, 4), Uint16)
            logits0.fill(0)
            for t in cutlass.range_constexpr(self.tiles):
                for n in cutlass.range_constexpr(2):
                    request = 8 * t + 2 * quarter + n
                    live = request < batch
                    if cutlass.const_expr(self.halves > 1):  # (this warp's N tiles)
                        live = live & (warp // self.row_warps == t % self.halves)
                    if live:
                        row = request * self.steps + i
                        for r in cutlass.range_constexpr(2):
                            pointer = base.iterator + row * self.vocab_rank + rows[r]
                            logits0[t, 2 * r + n] = load_bf16_bits(pointer)
            # markov_w1 rows of the live requests' previous tokens.
            for c in range(tid, W * pieces, THREADS):
                r, j = c // pieces, c % pieces
                if r < batch:
                    source = w1.iterator + Int64(previous[r]) * R + j * 8
                    copy_g2s(w1s.iterator + r * P + j * 16, source)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
            # Placeholders for the arguments a configuration does not read.
            a_in = logits0
            if cutlass.const_expr(self.a_registers):
                a_in = a
            sampled_in = (best, best, best)
            if cutlass.const_expr(self.sampled):
                sampled_in = (cached, temps_s, prefixes_s)
            for t in cutlass.range_constexpr(self.tiles):
                if cutlass.const_expr(self.halves == 1):
                    self.tile_keys(
                        t, a_in, tile, w1s, logits0, lane, warp, rows, rank, best,
                        *sampled_in,
                    )  # fmt: skip
                else:
                    if warp // self.row_warps == t % self.halves:
                        self.tile_keys(
                            t, a_in, tile, w1s, logits0, lane, warp, rows, rank, best,
                            *sampled_in,
                        )  # fmt: skip
            cute.arch.sync_threads()
            # This rank's maxima; the last CTA of the rank publishes them.
            if tid < W:
                # The warps of the request's N tile (rank 512: tile t's half).
                owner = ((tid // 8) % self.halves) * self.row_warps
                mine = best[owner, tid]
                for w in cutlass.range_constexpr(1, self.row_warps):
                    mine = max(mine, best[owner + w, tid])
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
                chunks = self.rows // 8
                for c in range(tid, batch * chunks, THREADS):
                    request = c // chunks
                    chunk = c % chunks
                    slot = slots_s[request]
                    if slot >= Int32(0):
                        words = load_words(
                            cached.iterator + request * self.rows + 8 * chunk,
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

    @cute.jit
    def rearm(self, ordinal, box, local, cta, tid):
        """The launch ordinal; re-arms the other parity (consumed by the previous
        launch), here and rank-local."""
        launch_ordinal = Int32(ordinal[0])
        parity = launch_ordinal & 1
        other = box.iterator + (1 - parity) * self.box_words
        for c in range(cta * THREADS + tid, self.box_words // 4, self.ctas * THREADS):
            store_relaxed(other + c * 4, (Uint32(0),) * 4)
        other_local = local.iterator + (1 - parity) * LOCAL_WORDS
        for c in range(cta * THREADS + tid, LOCAL_WORDS // 4, self.ctas * THREADS):
            store_words(other_local + c * 4, (Uint32(0),) * 4, scope="gpu")
        return launch_ordinal

    @cute.jit
    def tile_keys(
        self, t: cutlass.Constexpr[int], a, tile, w1s, logits0, lane, warp, rows, rank,
        best, cached, temps_s, prefixes_s,
    ):  # fmt: skip
        """N tile ``t``'s keys of the warp's 16 rows into ``best[warp]``: the bias
        (FP32 accumulation over the Markov columns in order: 256-column K halves of 16
        MMA k-steps, A from ``a`` (registers, rank 256) or the staged ``tile``), the
        logits and their keys (``cached``, ``temps_s``, ``prefixes_s``: ``sampled``
        only, placeholders otherwise)."""
        group, quarter = lane // 4, lane % 4
        tile_warp = warp % self.row_warps
        matrix_rows = w1s.iterator + (8 * t + lane % 8) * self.pitch
        matrix_rows = matrix_rows + 16 * (lane // 8)
        acc = zeros(4, Float32)
        for h in cutlass.range_constexpr(self.k_halves):
            b = cute.make_rmem_tensor((16, 2), Uint32)
            for q in cutlass.range_constexpr(8):
                frag = ldmatrix(matrix_rows + 512 * h + 64 * q)
                b[2 * q, 0], b[2 * q, 1] = frag[0], frag[1]
                b[2 * q + 1, 0], b[2 * q + 1, 1] = frag[2], frag[3]
            for s in cutlass.range_constexpr(16):
                a_frag = None
                if cutlass.const_expr(self.a_registers):
                    a_frag = tuple(a[s, k] for k in range(4))
                else:
                    a_frag = ldmatrix(tile + 512 * h + 32 * s)
                result = mma(
                    a_frag,
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
                    cached[8 * t + 2 * quarter + n, tile_warp * 16 + group + 8 * r] = (
                        Uint16(packed & Uint32(0xFFFF))
                    )
                else:
                    key = make_key(logit, index)
                keys[n] = max(keys[n], key)
        for offset in (4, 8, 16):
            for n in cutlass.range_constexpr(2):
                keys[n] = shuffle_max(keys[n], offset)
        if group == 0:
            best[warp, 8 * t + 2 * quarter] = keys[0]
            best[warp, 8 * t + 2 * quarter + 1] = keys[1]


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
    markov_rank=RANK,
    chained=False,
):
    kernel = MarkovWide(
        width, steps, tp, vocab_rank, spin_limit, sampled, markov_rank, chained
    )
    batch = cute.sym_int()
    args: tuple = (
        fake((vocab_rank, markov_rank)),
        fake((cute.sym_int(), markov_rank)),
        fake((cute.sym_int(), vocab_rank)),
        fake((batch,), (anchor_stride,), torch.int32, 4),
        fake((cute.sym_int(), steps), (draft_stride, 1), torch.int64, 8),
        fake((1,), None, torch.int32, 4),
        fake((cute.sym_int(),), None, torch.uint32, 16),
        Int64(0),
        fake((cute.sym_int(),), None, torch.uint32, 16),
        Int32(0),
        fake((vocab_rank // CTA_ROWS[markov_rank] + 2, 64), None, torch.int64, 8),
    )
    if sampled:
        args += sampled_fakes(cute.sym_int(), steps, tp * vocab_rank)
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    options = f"--gpu-arch={arch} --enable-tvm-ffi"
    return cute.compile(kernel, *args, stream, options=options)
