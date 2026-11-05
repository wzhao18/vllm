# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark greedy Markov drafting over a vocabulary-sharded LM head, one kernel for all
draft positions.

Stock DSpark gathers the full-vocabulary base logits of every draft position to every
rank and then, per position, adds the Markov bias ``markov_w2 @ markov_w1[prev]`` over
the full vocabulary (a replicated [V, rank] weight read per position) and takes the
argmax. Here each rank keeps its own LM-head shard's base logits and its rows of
``markov_w2`` (staged in shared memory once): per position every CTA computes the bias
and logits of its vocabulary rows (BF16 MMA, FP32 accumulation over the Markov columns
in order, rounded to BF16 and added to the BF16 base logit with one BF16 rounding, as
the stock GEMM epilogue and add do), takes its argmax and multicasts the
(value, index) key into a Lamport mailbox on every rank; every CTA then polls all
ranks' keys and takes the global argmax (the lowest index among equal maxima, as
``torch.argmax``), which is the next position's previous token.

Mailbox: per parity, [position, entry = rank x CTA, request] u64 keys (0 = unpublished;
a published key has no zero word). Launch ``n`` uses parity ``n % 2`` from a per-rank
device ordinal (CTA 0 increments it after its last poll, by which time every CTA of the
rank has read it) and first re-arms the other parity, which launch ``n - 1`` consumed
and no rank writes before launch ``n + 1``.

``gate``: per position, lane s of warp 0 first waits for source rank s's chunk holding
its last CTA's key, then every thread sweeps its chunks (each still checked): at 16
ranks, many outstanding polls on the lines the remote keys land in delay them.

``staged``: per position, warp r < batch copies request r's markov_w1 row to shared
memory and every warp takes its B fragments with ldmatrix. The direct per-lane
fragment loads touch one line per live request per load (16 warps x 32 loads per
CTA), which bounds the load phase from about three requests on.

``settle``: a CTA barrier between the key poll and its shuffle maxima, so warps whose
chunks landed early do not issue shuffles while sibling warps still poll (the
shuffles queue behind those loads).

``sampled`` (probabilistic drafting, ``draft_sample_method="probabilistic"``): the
stock loop's ``gumbel_sample`` instead of the argmax. Each logit (bitwise the stock
BF16 one) becomes ``logit / T + noise`` with the stock Gumbel noise of (request seed,
position + the draft salt, vocabulary index) (``common.gumbel``, bitwise the Triton
code), or stays the logit at temperature 0 / for padding requests; the keys carry the
FP32 score (``gumbel.sampled_key``), so every rank takes the stock draw. Each CTA also
stores its rows of every live request's pre-temperature logits into the draft-logits
cache ``[slot, position, V]`` (this rank's columns only; staged in shared memory and
written as 16-byte chunks while the keys travel), as ``gumbel_sample``'s
``logits_cache`` does for the whole vocabulary.

Markov rank (``CTA_ROWS``): at rank 256 a CTA owns 256 vocabulary rows (16 warps x one
16-row MMA tile, 16 k-steps). At rank 512 (kimi-k3.1-pro-dspark-preview) 256 staged
rows would take 256 x 1040 B = 266 KB of shared memory, so a CTA owns 128 rows (the
same ~133 KB of staging), warps 0-7 one 16-row tile each over 32 k-steps in column
order (the B fragments in two 256-column halves, so the registers stay those of rank
256), warps 8-15 stage, poll and reduce only: the same FP32 accumulation order and
roundings, twice the CTAs and mailbox entries per rank.
"""

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
import torch.distributed._symmetric_memory as symm_mem
from cutlass import Float32, Int32, Int64, Uint16, Uint32, Uint64  # noqa: F401
from cutlass.cute.runtime import make_fake_stream

from ..common.gumbel import (
    DRAFT_NOISE_SALT,
    drafted_value,
    request_prefix,
    sampled_key,
    sampled_token,
)
from ..common.host import architecture, fake
from ..common.primitives import (
    add_rn,
    alloc,
    asm,
    bf16,
    bf16x2,
    bits_float,
    copy_g2s,
    float_bits,
    launch,
    ldmatrix,
    load_words,
    mma,
    multicast_words,
    store_relaxed,
    store_words,
    zeros,
)

THREADS = 512
WARPS = THREADS // 32
MAX_STEPS = 7  # draft positions (the specialized decode verifies at most 7 drafts)
MAX_WIDTH = 8  # keys per mailbox entry (requests, padded to a power of two)
# Local vocabulary rows per CTA by Markov rank (columns of markov_w1 / markov_w2):
# 16-row MMA tiles of one warp each, the CTA's markov_w2 rows staged in shared memory
# (rows x row_pitch(rank): 135168 B at rank 256, 133120 B at rank 512).
CTA_ROWS = {256: 256, 512: 128}
# The specialized decode's Rubin shared memory per CTA when the device does not report
# its opt-in maximum (227 KiB).
SMEM_LIMIT = 232448


def row_pitch(rank):
    """Bytes per staged markov_w2 / markov_w1 row: 2 x rank + 16 (the 16 B pad puts the
    8 rows of an ldmatrix 8x8 tile in distinct banks)."""
    return rank * 2 + 16


ROWS = CTA_ROWS[256]  # (rank 256's; the kernels take theirs from CTA_ROWS)
RANK = 256
PITCH = row_pitch(RANK)  # 528
W1_PITCH = PITCH  # ``staged``: bytes per markov_w1 row in shared memory


def greedy_smem(rank, staged=True, sampled=True):
    """Upper bound of MarkovGreedy's shared memory (bytes) at Markov ``rank``."""
    rows, pitch = CTA_ROWS[rank], row_pitch(rank)
    total = rows * pitch + 2 * WARPS * MAX_WIDTH * 8 + MAX_WIDTH * 4
    if staged:
        total += MAX_WIDTH * pitch
    if sampled:
        total += MAX_WIDTH * rows * 2 + MAX_WIDTH * 4
    return total + 1024  # the allocations' alignment


def device_limits(device):
    """(SMs, opt-in shared memory per CTA, shared memory per SM) of ``device``."""
    props = torch.cuda.get_device_properties(device)
    block = getattr(props, "shared_memory_per_block_optin", 0) or SMEM_LIMIT
    sm = getattr(props, "shared_memory_per_multiprocessor", 0) or block + 1024
    return props.multi_processor_count, block, sm


def resident(ctas, smem, limits):
    """Whether ``ctas`` CTAs of ``THREADS`` threads and ``smem`` bytes of shared memory
    can be co-resident (every CTA polls every other CTA's keys); ``limits``:
    ``device_limits``. Registers are not checked (one CTA per SM fits at 128)."""
    sms, block, per_sm = limits
    if smem > block:
        return False
    per = min(2048 // THREADS, per_sm // (smem + 1024))
    return ctas <= sms * max(per, 0)


def supports(w1_shape, w2_shape, tp, limits=None):
    """Whether the fused Markov kernels cover a Markov head of markov_w1 / markov_w2
    shapes [V, rank] at ``tp`` vocabulary shards (a pure function of the shapes and the
    device, the same on every rank): a rank of ``CTA_ROWS``, the shards in whole CTAs,
    and every CTA of a rank co-resident (``limits``: ``device_limits``; None: not
    checked)."""
    if len(w2_shape) != 2 or tuple(w1_shape) != tuple(w2_shape):
        return False
    vocab, rank = w2_shape
    rows = CTA_ROWS.get(rank)
    if rows is None or vocab % (tp * rows):
        return False
    return limits is None or resident(vocab // tp // rows, greedy_smem(rank), limits)


@asm()
def fence_sys():
    return "fence.acq_rel.sys;", "~{memory}", ()


@asm(Int64)
def globaltimer():
    return "mov.u64 $0, %globaltimer;", "=l,~{memory}", ()


@asm(Uint16)
def load_bf16_bits(pointer):
    """One BF16 written by an earlier kernel of the step (side-effecting load)."""
    return "ld.global.cg.u16 $0,[$1];", "=h,l", (pointer,)


@cute.jit
def ordered(bits: Uint32):
    """Monotone image of BF16-valued FP32 ``bits`` in [1, 2^31) (NaN above everything,
    as torch.argmax): the usual sign flip, shifted right once, which drops a bit that
    is constant for BF16 values (0 for positive, 1 for negative ones)."""
    if bits == Uint32(0x80000000):  # -0.0 ranks as +0.0 (equal under torch.argmax)
        bits = Uint32(0)
    result = bits | Uint32(0x80000000)
    if (bits & Uint32(0x80000000)) != Uint32(0):
        result = ~bits
    if (bits & Uint32(0x7FFFFFFF)) > Uint32(0x7F800000):
        result = Uint32(0xFFFFFFFF)
    return result >> 1


@cute.jit
def make_key(value: Float32, index):
    """(value, lowest index wins) as one u64 below 2^63 (signed and unsigned order
    agree); never has a zero word."""
    high = ordered(Uint32(float_bits(value)))
    return (Uint64(high) << 32) | Uint64(Uint32(0xFFFFFFFF) - Uint32(index))


@cute.jit
def shuffle_max(key: Uint64, offset):
    low = cute.arch.shuffle_sync_bfly(Uint32(key & Uint64(0xFFFFFFFF)), offset)
    high = cute.arch.shuffle_sync_bfly(Uint32(key >> 32), offset)
    other = (Uint64(high) << 32) | Uint64(low)
    return max(key, other)


def parity_bytes(entries):
    return MAX_STEPS * entries * MAX_WIDTH * 8


class MarkovGreedy:
    """``batch`` requests, ``steps`` draft positions, ``tp`` vocabulary shards of
    ``vocab_rank`` rows, Markov ``rank`` (``CTA_ROWS``). Args: markov_w2 rows of this
    rank [vocab_rank, rank], markov_w1 [V, rank], base logits [batch * steps,
    vocab_rank] (row = request * steps + step), anchors [batch] (int32, any stride),
    draft [batch, steps] int64 (any row stride), ordinal int32 [1], mailbox (u32 view
    of both parities), multicast address, rank."""

    def __init__(
        self,
        batch,
        steps,
        tp,
        vocab_rank,
        spin_limit=0,
        stamps=False,
        gate=False,
        staged=False,
        settle=False,
        sampled=False,
        rank=RANK,
    ):
        self.stamps = stamps  # benchmarks: CTA 0's %globaltimer phase stamps in stats
        self.gate = gate
        self.staged = staged
        self.settle = settle
        self.sampled = sampled
        # ``spin_limit`` > 0 (tests only): a poll gives up after that many rounds and
        # records the unpublished chunk mask in ``stats`` instead of hanging.
        self.spin_limit = spin_limit
        # Markov rank: CTA rows, staged row pitch, 256-column K halves, tile warps.
        self.markov_rank, self.rows = rank, CTA_ROWS[rank]
        self.pitch, self.k_halves = row_pitch(rank), rank // 256
        self.row_warps = self.rows // 16
        assert vocab_rank % self.rows == 0 and rank % 256 == 0
        assert steps <= MAX_STEPS and batch <= MAX_WIDTH
        self.batch, self.steps, self.tp, self.vocab_rank = batch, steps, tp, vocab_rank
        self.width = 1 << (batch - 1).bit_length()
        self.ctas = vocab_rank // self.rows
        self.entries = tp * self.ctas
        self.parity_bytes = parity_bytes(self.entries)
        # 16-byte poll chunks (two keys) of one position, and per thread.
        self.chunks = self.entries * self.width // 2
        self.per_thread = (self.chunks + THREADS - 1) // THREADS

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, self.ctas, THREADS)

    @cute.jit
    def stamp(self, stats, cta, tid, slot: cutlass.Constexpr[int]):
        if cutlass.const_expr(self.stamps):  # noqa: SIM102
            if (cta == 0) & (tid == 0):
                stats[self.ctas + 1, slot] = globaltimer()

    @cute.kernel
    def kernel(self, args):
        w2, w1, base, anchors, draft, ordinal, box, mc, rank, stats = args[:10]
        R, P = self.markov_rank, self.pitch  # Markov rank, staged row pitch (bytes)
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        lane, warp = tid % 32, tid // 32
        group, quarter = lane // 4, lane % 4
        # The warp's 16-row MMA tile (rank 512: warps 8-15 have none, ``row_warps``).
        tile_warp = warp % self.row_warps
        smem = cutlass.utils.SmemAllocator()
        stage = alloc(smem, cutlass.Uint8, self.rows * P, 128)
        best = alloc(smem, Uint64, (16, MAX_WIDTH), 16)  # this CTA's keys per warp
        polled = alloc(smem, Uint64, (16, MAX_WIDTH), 16)  # every rank's, per warp
        winners = alloc(smem, Int32, MAX_WIDTH, 16)
        w1s = None
        if cutlass.const_expr(self.staged):
            w1s = alloc(smem, cutlass.Uint8, MAX_WIDTH * P, 128)
        cached = slots_s = None
        if cutlass.const_expr(self.sampled):
            # This position's pre-temperature logits of the CTA's rows per request,
            # and each request's slot (request-state index; -1: padding).
            cached = alloc(smem, Uint16, (MAX_WIDTH, self.rows), 16)
            slots_s = alloc(smem, Int32, MAX_WIDTH, 16)
        first = cta * self.rows
        self.stamp(stats, cta, tid, 0)
        # Static weights before the PDL wait: this CTA's markov_w2 rows (L2 evict
        # first: the drafter's weight streams leave the target's code and data in L2).
        pieces = R // 8  # 16 B chunks per row
        for c in range(tid, self.rows * pieces, THREADS):
            r, j = c // pieces, c % pieces
            copy_g2s(
                stage.iterator + r * P + j * 16,
                w2.iterator + (first + r) * R + j * 8,
                evict_first=True,
            )
        cute.arch.cp_async_commit_group()
        if cutlass.const_expr(self.staged):
            # Padding requests' rows stay zero (live rows are rewritten every position).
            for c in range(tid, MAX_WIDTH * P // 16, THREADS):
                store_words(w1s.iterator + 16 * c, (Uint32(0),) * 4, shared=True)
        launch_ordinal = ordinal[0]
        parity = launch_ordinal & 1
        # Re-arm the other parity (consumed by the previous launch).
        words = self.parity_bytes // 4
        other = box.iterator + (1 - parity) * words
        for c in range(cta * THREADS + tid, words // 4, self.ctas * THREADS):
            store_relaxed(other + c * 4, (Uint32(0),) * 4)
        # Peers write the re-armed parity only in their next launch, after seeing all of
        # this launch's publications, which follow these stores (no fence: a
        # system-scope fence here costs ~5 us).
        self.stamp(stats, cta, tid, 1)
        # No early trigger: a dependent launch of this kernel would re-arm and read the
        # ordinal before this launch has consumed its parity.
        cute.arch.griddepcontrol_wait()
        self.stamp(stats, cta, tid, 2)
        # This lane's base logits: vocabulary rows (group, group + 8) of its warp tile,
        # requests (2 * quarter, 2 * quarter + 1), every position.
        rows = (first + tile_warp * 16 + group, first + tile_warp * 16 + group + 8)
        requests = (2 * quarter, 2 * quarter + 1)
        logits0 = cute.make_rmem_tensor((MAX_STEPS, 4), Uint16)
        logits0.fill(0)
        for i in cutlass.range_constexpr(self.steps):
            for r in cutlass.range_constexpr(2):
                for n in cutlass.range_constexpr(2):
                    if (requests[n] < self.batch) & (warp < self.row_warps):
                        row = requests[n] * self.steps + i
                        pointer = base.iterator + row * self.vocab_rank + rows[r]
                        logits0[i, 2 * r + n] = load_bf16_bits(pointer)
        previous = Int32(0)
        if group < self.batch:
            previous = Int32(anchors[group])
        temps = seeds_r = positions_r = None
        if cutlass.const_expr(self.sampled):
            temps, seeds_r, positions_r = self.request_state(
                args, requests, tid, slots_s
            )
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        self.stamp(stats, cta, tid, 3)
        tile = stage.iterator + (tile_warp * 16 + lane % 8 + 8 * ((lane // 8) % 2)) * P
        tile = tile + 16 * (lane // 16)
        pair = (2 * tid) % self.width
        step_words = self.entries * self.width * 2  # u32 words per position
        own = (rank * self.ctas + cta) * self.width
        for i in cutlass.range_constexpr(self.steps):
            if cutlass.const_expr(self.staged):
                # Warp r < batch copies request r's row (2 x rank B: one 16 B load per
                # lane per 256 columns) to shared memory; then every tile warp takes
                # its fragments with ldmatrix (matrix j of x4 p: k-step 2 p + j // 2,
                # half j % 2).
                if warp < self.batch:
                    prev_token = cute.arch.shuffle_sync(previous, 4 * warp)
                    for piece in cutlass.range_constexpr(self.k_halves):
                        row_source = w1.iterator + Int64(prev_token) * R + 8 * lane
                        row_words = load_words(
                            row_source + 256 * piece, count=4, cache_policy="nc"
                        )
                        row_target = w1s.iterator + warp * P + 16 * lane + 512 * piece
                        store_words(row_target, row_words, shared=True)
                cute.arch.sync_threads()
            # (rank 512: warps 8-15 own no rows; their keys stay out of ``best``)
            if warp < self.row_warps:
                # FP32 accumulation over the Markov columns in order: 256-column K
                # halves, each with its B fragments (markov_w1 row of each request's
                # previous token, lane group) and 16 MMA k-steps.
                acc = zeros(4, Float32)
                for h in cutlass.range_constexpr(self.k_halves):
                    b = cute.make_rmem_tensor((16, 2), Uint32)
                    b.fill(0)
                    if cutlass.const_expr(self.staged):
                        matrix_rows = w1s.iterator + (lane % 8) * P + 16 * (lane // 8)
                        matrix_rows = matrix_rows + 512 * h
                        for q in cutlass.range_constexpr(8):
                            frag = ldmatrix(matrix_rows + 64 * q)
                            b[2 * q, 0], b[2 * q, 1] = frag[0], frag[1]
                            b[2 * q + 1, 0], b[2 * q + 1, 1] = frag[2], frag[3]
                    else:
                        # Read-only path (L1-cached): every warp of the CTA reads the
                        # same rows; only live requests' rows are read (padding
                        # columns stay zero).
                        source = w1.iterator + Int64(previous) * R + 2 * quarter
                        source = source + 256 * h
                        if group < self.batch:
                            for s in cutlass.range_constexpr(16):
                                for hh in cutlass.range_constexpr(2):
                                    pointer = source + 16 * s + 8 * hh
                                    b[s, hh] = load_words(
                                        pointer, count=1, cache_policy="nc"
                                    )[0]
                    if cutlass.const_expr(self.stamps and h == 0):  # noqa: SIM102
                        if (b[15, 1] | b[0, 0]) != Uint32(0x12345679):
                            self.stamp(stats, cta, tid, 4 + 6 * i)
                    for s in cutlass.range_constexpr(16):
                        a = ldmatrix(tile + 512 * h + 32 * s)
                        result = mma(
                            a,
                            (b[s, 0], b[s, 1]),
                            tuple(acc[k] for k in range(4)),
                            bf16=True,
                        )
                        for k in cutlass.range_constexpr(4):
                            acc[k] = result[k]
                if cutlass.const_expr(self.stamps):  # noqa: SIM102
                    if float_bits(acc[0] + acc[3]) != Int32(0x12345679):
                        self.stamp(stats, cta, tid, 5 + 6 * i)
                keys = cute.make_rmem_tensor(2, Uint64)
                keys.fill(0)
                prefixes = None
                if cutlass.const_expr(self.sampled):
                    prefixes = cute.make_rmem_tensor(2, Uint32)
                    for n in cutlass.range_constexpr(2):
                        prefixes[n] = request_prefix(seeds_r[n], positions_r[i, n])
                for r in cutlass.range_constexpr(2):
                    bias = bf16x2(acc[2 * r], acc[2 * r + 1])
                    for n in cutlass.range_constexpr(2):
                        base_bits = Uint32(logits0[i, 2 * r + n]) << 16
                        total = add_rn(bf16(bias, n == 1), bits_float(base_bits))
                        packed = bf16x2(total, total)
                        logit = bf16(packed, False)
                        index = rank * self.vocab_rank + rows[r]
                        if cutlass.const_expr(self.sampled):
                            key = sampled_key(
                                drafted_value(logit, temps[n], prefixes[n], index),
                                index,
                            )
                            cached[requests[n], tile_warp * 16 + group + 8 * r] = (
                                Uint16(packed & Uint32(0xFFFF))
                            )
                        else:
                            key = make_key(logit, index)
                        keys[n] = max(keys[n], key)
                for offset in (4, 8, 16):
                    for n in cutlass.range_constexpr(2):
                        keys[n] = shuffle_max(keys[n], offset)
                if group == 0:
                    best[warp, requests[0]] = keys[0]
                    best[warp, requests[1]] = keys[1]
            self.stamp(stats, cta, tid, 6 + 6 * i)
            cute.arch.sync_threads()
            if tid < self.width:
                mine = best[0, tid]
                for w in cutlass.range_constexpr(1, self.row_warps):
                    mine = max(mine, best[w, tid])
                offset_bytes = (parity * (self.parity_bytes // 4) + i * step_words) * 4
                offset_bytes = offset_bytes + (own + tid) * 8
                multicast_words(
                    mc + Int64(offset_bytes),
                    (Uint32(mine & Uint64(0xFFFFFFFF)), Uint32(mine >> 32)),
                    # A bit-exact store (multimem.st vectors need a float type).
                    kind="f32",
                )
                self.stamp(stats, cta, tid, 7 + 6 * i)
            if cutlass.const_expr(self.sampled):
                # While the keys travel: the position's logits into the cache.
                self.store_cache(args, cached, slots_s, tid, i, first, rank)
            # Every rank's keys of this position.
            region = box.iterator + parity * (self.parity_bytes // 4) + i * step_words
            if cutlass.const_expr(self.gate):
                if tid < self.tp:
                    last = ((tid + 1) * self.ctas - 1) * self.width // 2
                    self.wait_chunk(region + last * 4)
                cute.arch.sync_threads()
            got = cute.make_rmem_tensor((self.per_thread, 4), Uint32)
            got.fill(0)
            pending = Uint32(0)
            for j in cutlass.range_constexpr(self.per_thread):
                c = tid + j * THREADS
                if c < self.chunks:
                    pending |= Uint32(1 << j)
            spins = Int32(0)
            while pending != Uint32(0):
                spins += 1
                if cutlass.const_expr(self.spin_limit > 0):  # noqa: SIM102
                    if spins > self.spin_limit:
                        stats[cta, i] = Int64(0x7FFF0000) | Int64(pending)
                        pending = Uint32(0)
                # Every chunk's load first (one round trip per pass), then the checks;
                # threads past the end re-load the last chunk.
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
            self.stamp(stats, cta, tid, 8 + 6 * i)
            if cutlass.const_expr(self.settle):
                cute.arch.sync_threads()
            lo_key, hi_key = Uint64(0), Uint64(0)
            for j in cutlass.range_constexpr(self.per_thread):
                k0 = (Uint64(got[j, 1]) << 32) | Uint64(got[j, 0])
                k1 = (Uint64(got[j, 3]) << 32) | Uint64(got[j, 2])
                lo_key, hi_key = max(lo_key, k0), max(hi_key, k1)
            if cutlass.const_expr(self.width == 1):
                lo_key = max(lo_key, hi_key)
                for offset in (1, 2, 4, 8, 16):
                    lo_key = shuffle_max(lo_key, offset)
            else:
                for offset in (1, 2, 4, 8, 16):
                    if cutlass.const_expr(offset >= self.width // 2):
                        lo_key = shuffle_max(lo_key, offset)
                        hi_key = shuffle_max(hi_key, offset)
            if lane < max(self.width // 2, 1):
                polled[warp, pair] = lo_key
                if cutlass.const_expr(self.width > 1):
                    polled[warp, pair + 1] = hi_key
            cute.arch.sync_threads()
            if tid < self.width:
                final = polled[0, tid]
                for w in cutlass.range_constexpr(1, 16):
                    final = max(final, polled[w, tid])
                token = Int32(Uint32(0xFFFFFFFF) - Uint32(final & Uint64(0xFFFFFFFF)))
                if cutlass.const_expr(self.sampled):
                    token = sampled_token(final)
                winners[tid] = token
                if (cta == 0) & (tid < self.batch):
                    draft[tid, i] = Int64(token)
                if cutlass.const_expr(self.spin_limit > 0):  # noqa: SIM102
                    if (cta == 0) & (tid == 0):
                        stats[self.ctas, i] = Int64(final)
            cute.arch.sync_threads()
            self.stamp(stats, cta, tid, 9 + 6 * i)
            if group < self.width:
                previous = min(max(winners[group], Int32(0)), Int32(w1.shape[0] - 1))
        if (cta == 0) & (tid == 0):
            ordinal[0] = launch_ordinal + 1

    @cute.jit
    def request_state(self, args, requests, tid, slots_s):
        """``sampled``: the lane's two requests' temperature (0 for padding), seed and
        salted positions per draft position; ``slots_s`` filled (read after the PDL
        wait: the step's input preparation writes them)."""
        temperature, seeds, slots, positions = args[10], args[11], args[12], args[13]
        temps = cute.make_rmem_tensor(2, Float32)
        seeds_r = cute.make_rmem_tensor(2, Int64)
        positions_r = cute.make_rmem_tensor((MAX_STEPS, 2), Int64)
        positions_r.fill(0)
        for n in cutlass.range_constexpr(2):
            temps[n] = Float32(0.0)
            seeds_r[n] = Int64(0)
            if requests[n] < self.batch:
                slot = Int64(slots[requests[n] * self.steps])
                if slot >= Int64(0):
                    temps[n] = Float32(temperature[slot])
                    seeds_r[n] = Int64(seeds[slot])
                for i in cutlass.range_constexpr(self.steps):
                    # gumbel_sample's position: the predicted token's - 1, salted.
                    position = Int64(positions[requests[n] * self.steps + i])
                    positions_r[i, n] = position - Int64(1) + Int64(DRAFT_NOISE_SALT)
        if tid < MAX_WIDTH:
            slot_value = Int32(-1)
            if tid < self.batch:
                slot_value = Int32(slots[tid * self.steps])
            slots_s[tid] = slot_value
        return temps, seeds_r, positions_r

    @cute.jit
    def store_cache(self, args, cached, slots_s, tid, i, first, rank):
        """The CTA's rows of position ``i`` of every live request into the cache
        [slot, position, V] (16-byte chunks; padding requests are skipped)."""
        cache = args[14]
        vocab = self.tp * self.vocab_rank
        chunks = self.rows // 8
        if tid < self.batch * chunks:
            request = tid // chunks
            chunk = tid % chunks
            slot = slots_s[request]
            if slot >= Int32(0):
                words = load_words(
                    cached.iterator + request * self.rows + 8 * chunk,
                    count=4,
                    shared=True,
                )
                row = Int64(slot) * self.steps + i
                column = rank * self.vocab_rank + first + 8 * chunk
                store_words(cache.iterator + (row * vocab + Int64(column)), words)

    @cute.jit
    def wait_chunk(self, pointer):
        """Spin until the 16-byte key chunk at ``pointer`` has no zero word (gives up
        after ``spin_limit`` rounds when set; the sweep then records it)."""
        spins = Int32(0)
        values = load_words(pointer, count=4, volatile=True)
        ready = (values[0] != Uint32(0)) & (values[1] != Uint32(0))
        ready = ready & (values[2] != Uint32(0)) & (values[3] != Uint32(0))
        while not ready:
            spins += 1
            values = load_words(pointer, count=4, volatile=True)
            ready = (values[0] != Uint32(0)) & (values[1] != Uint32(0))
            ready = ready & (values[2] != Uint32(0)) & (values[3] != Uint32(0))
            if cutlass.const_expr(self.spin_limit > 0):  # noqa: SIM102
                if spins > self.spin_limit:
                    ready = True


class MarkovBox:
    """Both parities' Lamport key mailboxes (symmetric, multicast) and the ordinal."""

    def __init__(self, group, device, entries):
        words = 2 * parity_bytes(entries) // 4
        self.buffer = symm_mem.empty((words,), dtype=torch.int32, device=device)
        self.handle = handle = symm_mem.rendezvous(self.buffer, group)
        self.mc = int(handle.multicast_ptr or 0)
        if not self.mc:
            raise RuntimeError("Kimi K3 DSpark Markov drafting needs multicast memory")
        self.buffer.zero_()
        self.ordinal = torch.zeros(1, dtype=torch.int32, device=device)
        torch.accelerator.synchronize(device)
        torch.distributed.barrier(group=group)


@cache
def compile_markov(
    batch,
    steps,
    tp,
    vocab_rank,
    arch,
    anchor_stride,
    draft_stride,
    spin_limit=0,
    stamps=False,
    gate=False,
    staged=False,
    settle=False,
    sampled=False,
    markov_rank=RANK,
):
    kernel = MarkovGreedy(
        batch, steps, tp, vocab_rank, spin_limit, stamps, gate, staged, settle, sampled,
        markov_rank,
    )  # fmt: skip
    rows = vocab_rank
    args: tuple = (
        fake((rows, markov_rank)),
        fake((cute.sym_int(), markov_rank)),
        fake((batch * steps, rows)),
        fake((batch,), (anchor_stride,), torch.int32, 4),
        fake((batch, steps), (draft_stride, 1), torch.int64, 8),
        fake((1,), None, torch.int32, 4),
        fake((cute.sym_int(),), None, torch.uint32, 16),
        Int64(0),
        Int32(0),
        fake((vocab_rank // CTA_ROWS[markov_rank] + 2, 64), None, torch.int64, 8),
    )
    if sampled:
        args += sampled_fakes(batch * steps, steps, tp * vocab_rank)
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    options = f"--gpu-arch={arch} --enable-tvm-ffi"
    return cute.compile(kernel, *args, stream, options=options)


def sampled_fakes(rows, steps, vocab):
    """``sampled`` kernels' extra arguments: temperature [max_reqs] FP32, seeds
    [max_reqs] int64, slots [rows] int64 (request-state index per sample row),
    positions [rows] int64 (predicted-token positions), draft-logits cache [max_reqs,
    steps, V] BF16."""
    slots_count = cute.sym_int()
    return (
        fake((slots_count,), None, torch.float32, 4),
        fake((slots_count,), None, torch.int64, 8),
        fake((rows,), None, torch.int64, 8),
        fake((rows,), None, torch.int64, 8),
        fake((cute.sym_int(), steps, vocab), None, torch.bfloat16, 16),
    )


class PreparedMarkov:
    """The fused Markov drafter of one rank: compiles per (launch, strides, sampled).
    ``wide``: batches above MAX_WIDTH run ``markov_wide.MarkovWide`` (its mailboxes
    are allocated here, on every rank together), in launches of at most
    ``launch_width`` request slots (``plan``: every request's draft is its own chain,
    so a batch above one launch runs as consecutive launches over request slices; the
    same drafts; the launches after the first are compiled ``chained``: they read the
    ordinal and re-arm after their PDL wait, see markov_wide). ``sampled`` kernels draw
    as the stock probabilistic loop (``MarkovGreedy`` ``sampled``).
    NQ-TUNE: the chained launches' re-arm after the wait (no longer overlapping the
    previous launch's tail) at B65-B192, vs one wider launch where it fits."""

    def __init__(
        self,
        w1,
        w2,
        tp,
        rank,
        group,
        steps,
        spin_limit=0,
        stamps=False,
        gated=(),
        staged=(),
        settled=(),
        wide=False,
    ):
        self.spin_limit, self.stamps = spin_limit, stamps
        self.gated = frozenset(gated)  # batches whose polls are gated
        self.staged = frozenset(staged)  # batches with markov_w1 rows staged in smem
        self.settled = frozenset(settled)  # batches with a barrier after the poll
        vocab = w2.shape[0]
        if not supports(w1.shape, w2.shape, tp):
            raise ValueError(f"unsupported Markov head shapes {tuple(w1.shape)}")
        self.tp, self.rank, self.steps = tp, rank, steps
        self.markov_rank = w2.shape[1]
        self.rows = CTA_ROWS[self.markov_rank]
        self.vocab = vocab
        self.vocab_rank = vocab // tp
        self.w1 = w1.detach()
        self.w2 = w2.detach()[rank * self.vocab_rank : (rank + 1) * self.vocab_rank]
        self.device = w2.device
        self.arch = architecture(self.device)
        self.limits = device_limits(self.device)
        self.box = MarkovBox(group, self.device, vocab // self.rows)
        self.wide_box = None
        if wide:
            from .markov_wide import MarkovWideBox

            self.wide_box = MarkovWideBox(group, self.device, tp)
        ctas = self.vocab_rank // self.rows
        self.stats = torch.zeros((ctas + 2, 64), dtype=torch.int64, device=self.device)
        self.kernels = {}
        self.plans = {}

    def plan(self, batch, sampled=False):
        """The launches of ``batch`` requests, [(first, end, width)]: width 0, one
        MarkovGreedy launch (at most MAX_WIDTH requests); else MarkovWide launches of
        ``width`` request slots over balanced request slices. [] when not covered
        (no wide mailbox, or not even 8 slots fit the device's shared memory). A pure
        function of the deployment, the same on every rank."""
        key = batch, sampled
        if key not in self.plans:
            launches = []
            if batch <= MAX_WIDTH:
                launches = [(0, batch, 0)]
            elif self.wide_box is not None:
                from .markov_wide import launch_width

                limit = launch_width(
                    self.tp, self.markov_rank, sampled, self.limits, self.vocab_rank
                )
                if limit:
                    count = -(-batch // limit)  # launches
                    per = -(-batch // count)  # requests per launch (balanced)
                    width = -(-per // 8) * 8
                    launches = [
                        (lo, min(lo + width, batch), width)
                        for lo in range(0, batch, width)
                    ]
            self.plans[key] = launches
        return self.plans[key]

    def covers(self, batch, sampled=False):
        return bool(self.plan(batch, sampled))

    @staticmethod
    def launch_key(index, batch, width):
        """A plan launch's compile key: MarkovGreedy by its batch, MarkovWide by its
        width and whether it follows another MarkovWide launch (``chained``)."""
        return ("wide", width, index > 0) if width else batch

    def kernel(
        self, batch, anchor_stride, draft_stride, sampled=False, width=0, chained=False
    ):
        """The compiled MarkovGreedy launch of ``batch`` requests (``width`` 0), or
        the MarkovWide launch of ``width`` request slots (any batch up to it;
        ``chained``: right after another MarkovWide launch)."""
        launch = ("wide", width, chained) if width else batch
        key = (launch, anchor_stride, draft_stride, sampled)
        if key not in self.kernels and width:
            from .markov_wide import compile_markov_wide

            with torch.accelerator.device_index(self.device.index):
                self.kernels[key] = compile_markov_wide(
                    width,
                    self.steps,
                    self.tp,
                    self.vocab_rank,
                    self.arch,
                    anchor_stride,
                    draft_stride,
                    self.spin_limit,
                    sampled,
                    self.markov_rank,
                    chained,
                )
        if key not in self.kernels:
            with torch.accelerator.device_index(self.device.index):
                self.kernels[key] = compile_markov(
                    batch,
                    self.steps,
                    self.tp,
                    self.vocab_rank,
                    self.arch,
                    anchor_stride,
                    draft_stride,
                    self.spin_limit,
                    self.stamps,
                    batch in self.gated,
                    batch in self.staged,
                    batch in self.settled,
                    sampled,
                    self.markov_rank,
                )
        return self.kernels[key]

    def keys(self, batch, anchor_stride, draft_stride, sampled=False):
        strides = anchor_stride, draft_stride, sampled
        return [
            (self.launch_key(index, hi - lo, width), *strides)
            for index, (lo, hi, width) in enumerate(self.plan(batch, sampled))
        ]

    def ready(self, batch, anchor_stride, draft_stride, sampled=False):
        keys = self.keys(batch, anchor_stride, draft_stride, sampled)
        return bool(keys) and all(key in self.kernels for key in keys)

    def prepare(self, batch, anchor_stride, draft_stride, sampled=False):
        """Compile and run once (every rank together: the launch exchanges keys), so
        nothing compiles or loads during CUDA graph capture."""
        launches = self.plan(batch, sampled)
        if not launches:
            raise ValueError(f"Markov drafting: {batch} requests are not covered")
        for index, (lo, hi, width) in enumerate(launches):
            chained = bool(width) and index > 0
            self.kernel(hi - lo, anchor_stride, draft_stride, sampled, width, chained)
        rows = batch * self.steps
        base = torch.zeros(
            (rows, self.vocab_rank), dtype=torch.bfloat16, device=self.device
        )
        ids = torch.zeros(rows * anchor_stride, dtype=torch.int32, device=self.device)
        out = torch.zeros(
            (batch * draft_stride,), dtype=torch.int64, device=self.device
        )
        draft = out.as_strided((batch, self.steps), (draft_stride, 1))
        if sampled:
            # Padding slots only: no cache row is written, every draw is an argmax.
            sampling = self.sampling_args(
                torch.zeros(1, dtype=torch.float32, device=self.device),
                torch.zeros(1, dtype=torch.int64, device=self.device),
                torch.full((rows,), -1, dtype=torch.int64, device=self.device),
                torch.zeros(rows, dtype=torch.int64, device=self.device),
                torch.zeros(
                    (1, self.steps, self.vocab),
                    dtype=torch.bfloat16,
                    device=self.device,
                ),
            )
            self.run(base, ids[::anchor_stride][:batch], draft, sampling)
        else:
            self.run(base, ids[::anchor_stride][:batch], draft)
        torch.accelerator.synchronize(self.device)

    def sampling_args(self, temperature, seeds, slots, positions, cache):
        """The ``sampled`` kernels' extra arguments (``sampled_fakes``): temperature
        [max_reqs] FP32 and seeds [max_reqs] int64 per request slot, slots and
        positions [rows] int64 per sample row (``slots[r * steps]`` is request r's slot,
        -1 for padding; positions are the predicted tokens'), cache [max_reqs, steps,
        V] BF16 (this rank's columns are written)."""
        if (
            temperature.dtype != torch.float32
            or seeds.dtype != torch.int64
            or slots.dtype != torch.int64
            or positions.dtype != torch.int64
            or cache.dtype != torch.bfloat16
            or tuple(cache.shape[1:]) != (self.steps, self.vocab)
            or not cache.is_contiguous()
            or not (slots.is_contiguous() and positions.is_contiguous())
        ):
            raise ValueError("Markov drafting: unexpected sampling arguments")
        return (temperature, seeds, slots, positions, cache)

    def run(self, base, anchors, draft, sampling=None):
        """Base [batch * steps, vocab_rank] BF16 (this rank's LM-head logits), anchors
        [batch] int32, draft [batch, steps] int64 (written); ``sampling``: a
        ``sampling_args`` tuple for probabilistic drafting (else greedy)."""
        batch = anchors.shape[0]
        if (
            base.shape != (batch * self.steps, self.vocab_rank)
            or not base.is_contiguous()
        ):
            raise ValueError(
                f"Markov drafting: unexpected base logits {tuple(base.shape)}"
            )
        sampled = sampling is not None
        if sampled and sampling[2].shape[0] != batch * self.steps:
            raise ValueError("Markov drafting: one slot / position per sample row")
        launches = self.plan(batch, sampled)
        if not launches:
            raise ValueError(f"Markov drafting: {batch} requests are not covered")
        steps = self.steps
        for index, (lo, hi, width) in enumerate(launches):
            # (a MarkovWide launch after the first reads its ordinal after the PDL
            # wait: back to back, nothing else orders it after the previous launch)
            kernel = self.kernel(
                hi - lo,
                anchors.stride(0),
                draft.stride(0),
                sampled,
                width,
                chained=bool(width) and index > 0,
            )
            rows = slice(lo * steps, hi * steps)
            # (one launch: the caller's tensors themselves)
            whole = len(launches) == 1
            part = (base, anchors, draft) if whole else (
                base[rows], anchors[lo:hi], draft[lo:hi]
            )  # fmt: skip
            extra = ()
            if sampled:
                temperature, seeds, slots, positions, cache = sampling
                if not whole:
                    slots, positions = slots[rows], positions[rows]
                extra = (temperature, seeds, slots, positions, cache)
            if width:
                wide = self.wide_box
                assert wide is not None
                kernel(
                    self.w2,
                    self.w1,
                    *part,
                    wide.ordinal,
                    wide.buffer.view(torch.uint32),
                    wide.mc,
                    wide.local.view(torch.uint32),
                    self.rank,
                    self.stats,
                    *extra,
                )
                continue
            box = self.box
            kernel(
                self.w2,
                self.w1,
                *part,
                box.ordinal,
                box.buffer.view(torch.uint32),
                box.mc,
                self.rank,
                self.stats,
                *extra,
            )
