# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The target's LM-head shard and the greedy argmax of its rows in one weight stream,
for spec-decode verification of up to 8 rows (B1: one request, 8 verify tokens).

``LmHeadKeys`` streams this rank's LM-head shard [vocab_rank, hidden] (BF16,
row-major, read in place) straight from HBM into ``mma.sync`` A fragments and
multiplies it with the rows' hidden states (the B fragments, 8 rows = the MMA's n). A
CTA owns 16 vocabulary rows; its 8 warps take interleaved 64-element hidden chunks
(chunk c of warp w is elements 64 (8c + w)), so the CTA reads each row in adjacent
128-byte lines. Per chunk, lane (g, t) loads hidden elements 16t..16t+15 of rows g and
g + 8 with one 32-byte load each and uses them as the k slots {2t, 2t+1, 2t+8, 2t+9}
of four MMAs; the B fragments take the same k permutation, so every product pairs the
right elements. The warps' FP32 partial sums are added in a fixed order and rounded
to BF16 (as the stock GEMM's epilogue rounds). Each CTA then writes, per row, the
(value, lowest index) key of its vocabulary rows (``argmax.local_key``), and
``KeyArgmax`` reduces the CTAs' keys of a row and exchanges them across ranks exactly
as ``argmax.VocabArgmax`` does with its scanned keys (same mailbox, parities and
ordinals, so the two kernels share an ``ArgmaxBox``).

``LmHeadLogits`` is the same weight stream with a BF16 logits epilogue (the drafter's
LM head, whose sampling needs the logits).

Numerics: the logits are the stock BF16 logits up to the FP32 accumulation order
(a stated reorder: MMA k blocks in a fixed permuted order, hidden splits added last),
so a row's argmax can differ from the stock cuBLAS GEMM's only where the top two
logits round to adjacent or equal BF16 values differently; ties keep the lowest index.
"""

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32, Uint64
from cutlass.cute.runtime import make_fake_stream

from ..common.host import architecture, fake
from ..common.primitives import (
    alloc,
    asm,
    bf16x2,
    launch,
    load_words,
    mma,
    multicast_words,
    store_relaxed,
    thread,
)
from .argmax import MAX_ROWS, box_words, global_key, local_key, shuffle_max, umax

TOKENS = 8  # rows per launch (the MMA's n)
WARPS = 8  # hidden splits of a CTA's 16 vocabulary rows
UNROLL = 2  # 64-element hidden chunks per lane in flight


@asm()
def store_bf16(pointer, value):
    code = "{ .reg .b16 h; cvt.rn.bf16.f32 h,$1; st.global.b16 [$0],h; }"
    return code, "l,f,~{memory}", (pointer, Float32(value))


class LmHeadKeys:
    """Args: weight [vocab_rank, hidden] BF16, hidden_states [rows, hidden] BF16 (rows
    <= 8; both row-major, 32-byte aligned), keys [8, ctas] u32 (written), rows. Grid:
    vocab_rank / 16 CTAs of 8 warps."""

    def __init__(self, vocab_rank, hidden):
        assert vocab_rank % 16 == 0 and vocab_rank <= 1 << 16
        self.vocab_rank, self.hidden = vocab_rank, hidden
        self.ctas = vocab_rank // 16
        self.threads = 32 * WARPS
        assert hidden % (WARPS * 64 * UNROLL) == 0
        self.iterations = hidden // (WARPS * 64 * UNROLL)

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, self.ctas, self.threads)

    @cute.kernel
    def kernel(self, args):
        weight, hidden_states, keys, rows = args
        smem = cutlass.utils.SmemAllocator()
        partial = self.partials(smem, weight, hidden_states, rows)
        tid, split, lane, cta = thread()
        best = alloc(smem, Uint32, (WARPS, TOKENS))
        # (row, token) items with the hidden splits added in order; a thread's items
        # all have token tid % 8.
        key = Uint32(0)
        for item in range(tid, 16 * TOKENS, self.threads):
            row, token = item // TOKENS, item % TOKENS
            bits = bf16x2(self.total(partial, row, token), Float32(0)) & Uint32(0xFFFF)
            key = umax(key, local_key(bits, cta * 16 + row))
        # Lanes with the same token (lane % 8) hold the same column: reduce them.
        for offset in (8, 16):
            key = umax(key, cute.arch.shuffle_sync_bfly(key, offset))
        if lane < TOKENS:
            best[split, lane] = key
        cute.arch.sync_threads()
        if tid < TOKENS:
            mine = best[0, tid]
            for w in cutlass.range_constexpr(1, WARPS):
                mine = umax(mine, best[w, tid])
            keys[tid, cta] = mine

    @cute.jit
    def total(self, partial, row, token):
        """A logit in FP32: the hidden splits' partial sums added in order."""
        total = partial[0, row, token]
        for s in cutlass.range_constexpr(1, WARPS):
            total = total + partial[s, row, token]
        return total

    @cute.jit
    def partials(self, smem, weight, hidden_states, rows):
        """The CTA's [split, row, token] FP32 partial logits in SMEM (synced)."""
        tid, split, lane, cta = thread()
        group, quarter = lane // 4, lane % 4
        partial = alloc(smem, cutlass.Float32, (WARPS, 16, TOKENS))
        w_lo = weight.iterator + (cta * 16 + group) * self.hidden + 16 * quarter
        w_hi = w_lo + 8 * self.hidden
        h_row = hidden_states.iterator + group * self.hidden + 16 * quarter
        acc = cute.make_rmem_tensor(4, cutlass.Float32)
        acc.fill(0.0)
        lo = cute.make_rmem_tensor((UNROLL, 8), Uint32)
        hi = cute.make_rmem_tensor((UNROLL, 8), Uint32)
        h = cute.make_rmem_tensor((UNROLL, 8), Uint32)
        h.fill(0)
        for it in range(self.iterations):
            for j in cutlass.range_constexpr(UNROLL):
                # Static weights, streamed once and evicted first, so the 147 MB
                # stream leaves the L2 (kernel code, other weights) to the step.
                k = 64 * (WARPS * (it * UNROLL + j) + split)
                words = load_words(w_lo + k, count=8, evict_first=True)
                for q in cutlass.range_constexpr(8):
                    lo[j, q] = words[q]
                words = load_words(w_hi + k, count=8, evict_first=True)
                for q in cutlass.range_constexpr(8):
                    hi[j, q] = words[q]
            if it == 0:
                # The hidden states are the step's activations: read after the wait
                # (the first weight chunks are already in flight).
                cute.arch.griddepcontrol_wait()
            if group < rows:
                for j in cutlass.range_constexpr(UNROLL):
                    k = 64 * (WARPS * (it * UNROLL + j) + split)
                    words = load_words(
                        h_row + k, count=8, cache_policy="ca", mutable=True
                    )
                    for q in cutlass.range_constexpr(8):
                        h[j, q] = words[q]
            for j in cutlass.range_constexpr(UNROLL):
                for p in cutlass.range_constexpr(4):
                    a = (lo[j, 2 * p], hi[j, 2 * p], lo[j, 2 * p + 1], hi[j, 2 * p + 1])
                    b = (h[j, 2 * p], h[j, 2 * p + 1])
                    result = mma(a, b, tuple(acc[c] for c in range(4)), bf16=True)
                    for c in cutlass.range_constexpr(4):
                        acc[c] = result[c]
        # The argmax exchange may be scheduled now: it waits for this grid to finish.
        cute.arch.griddepcontrol_launch_dependents()
        # acc: (row group, tokens 2q, 2q + 1), (row group + 8, tokens 2q, 2q + 1).
        for r in cutlass.range_constexpr(2):
            for n in cutlass.range_constexpr(2):
                partial[split, group + 8 * r, 2 * quarter + n] = acc[2 * r + n]
        cute.arch.sync_threads()
        return partial


class LmHeadLogits(LmHeadKeys):
    """``LmHeadKeys``' weight stream with a BF16 logits epilogue. Args: weight,
    hidden_states [rows, hidden], logits [rows, vocab_rank] BF16 (written, row-major),
    rows."""

    @cute.kernel
    def kernel(self, args):
        weight, hidden_states, logits, rows = args
        smem = cutlass.utils.SmemAllocator()
        partial = self.partials(smem, weight, hidden_states, rows)
        tid, _, _, cta = thread()
        # Token-major items: a half-warp writes 16 consecutive logits of a row.
        for item in range(tid, 16 * TOKENS, self.threads):
            token, row = item // 16, item % 16
            if token < rows:
                pointer = logits.iterator + token * self.vocab_rank + cta * 16 + row
                store_bf16(pointer, self.total(partial, row, token))


class KeyArgmax:
    """``argmax.VocabArgmax`` over the CTAs' keys of ``LmHeadKeys`` instead of the
    logits: one CTA (256 threads) per row reduces its ``ctas`` keys, multicasts the
    rank's key and polls every rank's (the same mailbox, parities and ordinals).
    Args: keys [8, ctas] u32, out [rows] int64 (written), ordinals, mailbox (u32
    view), multicast address, rank."""

    def __init__(self, tp, vocab_rank, ctas):
        assert 1 < tp <= 32 and ctas % 4 == 0
        self.tp, self.vocab_rank, self.ctas = tp, vocab_rank, ctas

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, cute.size(args[1], mode=[0]), 256)

    @cute.kernel
    def kernel(self, args):
        keys, out, ordinals, box, mc, rank = args
        tid, warp, lane, row = thread()
        smem = cutlass.utils.SmemAllocator()
        best = alloc(smem, Uint32, 8, 16)
        cute.arch.griddepcontrol_wait()
        launch_ordinal = ordinals[row]
        parity = launch_ordinal & 1
        key = Uint32(0)
        # The row's keys in one round of 16-byte loads.
        for j in cutlass.range_constexpr((self.ctas // 4 + 255) // 256):
            quad = j * 256 + tid
            if quad < self.ctas // 4:
                words = load_words(
                    keys.iterator + row * self.ctas + 4 * quad, count=4, mutable=True
                )
                for q in cutlass.range_constexpr(4):
                    key = umax(key, words[q])
        for offset in (1, 2, 4, 8, 16):
            key = umax(key, cute.arch.shuffle_sync_bfly(key, offset))
        if lane == 0:
            best[warp] = key
        cute.arch.sync_threads()
        entries = (parity * MAX_ROWS + row) * self.tp
        if tid == 0:
            for w in cutlass.range_constexpr(1, 8):
                key = umax(key, best[w])
            mine = global_key(key, rank * self.vocab_rank)
            multicast_words(
                mc + Int64((entries + rank) * 8),
                (Uint32(mine & Uint64(0xFFFFFFFF)), Uint32(mine >> 32)),
                kind="f32",
            )
        if warp == 0:
            got = Uint64(0)
            if lane < self.tp:
                entry = box.iterator + (entries + lane) * 2
                pending = Int32(1)
                while pending != Int32(0):
                    pair = load_words(entry, count=2, volatile=True)
                    got = (Uint64(pair[1]) << 32) | Uint64(pair[0])
                    if (pair[0] != Uint32(0)) & (pair[1] != Uint32(0)):
                        pending = Int32(0)
                store_relaxed(entry, (Uint32(0), Uint32(0)))
            for offset in (1, 2, 4, 8, 16):
                got = shuffle_max(got, offset)
            if lane == 0:
                token = Uint32(0xFFFFFFFF) - Uint32(got & Uint64(0xFFFFFFFF))
                out[row] = Int64(token)
                ordinals[row] = launch_ordinal + 1


@cache
def compile_lmhead(vocab_rank, hidden, arch):
    kernel = LmHeadKeys(vocab_rank, hidden)
    rows = cute.sym_int()
    args = (
        fake((vocab_rank, hidden)),
        fake((rows, hidden)),
        fake((TOKENS, kernel.ctas), None, torch.uint32, 16),
        Int32(0),
    )
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    options = f"--gpu-arch={arch} --enable-tvm-ffi"
    return kernel, cute.compile(kernel, *args, stream, options=options)


@cache
def compile_lmhead_logits(vocab_rank, hidden, arch):
    kernel = LmHeadLogits(vocab_rank, hidden)
    rows = cute.sym_int()
    args = (
        fake((vocab_rank, hidden)),
        fake((rows, hidden)),
        fake((rows, vocab_rank), None, torch.bfloat16, 2),
        Int32(0),
    )
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    options = f"--gpu-arch={arch} --enable-tvm-ffi"
    return kernel, cute.compile(kernel, *args, stream, options=options)


@cache
def compile_key_argmax(tp, vocab_rank, ctas, arch):
    kernel = KeyArgmax(tp, vocab_rank, ctas)
    rows = cute.sym_int()
    args = (
        fake((TOKENS, ctas), None, torch.uint32, 16),
        fake((rows,), None, torch.int64, 8),
        fake((MAX_ROWS,), None, torch.int32, 4),
        fake((box_words(tp),), None, torch.uint32, 16),
        Int64(0),
        Int32(0),
    )
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    options = f"--gpu-arch={arch} --enable-tvm-ffi"
    return cute.compile(kernel, *args, stream, options=options)


def supports(weight):
    """Whether ``PreparedLmHeadArgmax`` / ``PreparedLmHeadLogits`` take this LM-head
    shard [vocab_rank, hidden]."""
    vocab_rank, hidden = weight.shape
    return (
        weight.dtype == torch.bfloat16
        and weight.is_contiguous()
        and weight.data_ptr() % 32 == 0
        and vocab_rank % 16 == 0
        and vocab_rank <= 1 << 16
        and hidden % (WARPS * 64 * UNROLL) == 0
    )


def takes(hidden_states, hidden):
    """Whether the kernels take these hidden rows (<= 8, row-major, 32-byte aligned)."""
    return (
        0 < hidden_states.shape[0] <= TOKENS
        and hidden_states.shape[1] == hidden
        and hidden_states.dtype == torch.bfloat16
        and hidden_states.is_contiguous()
        and hidden_states.data_ptr() % 32 == 0
    )


class PreparedLmHeadArgmax:
    """The fused LM-head argmax of one rank for up to 8 rows (every rank calls it
    together). ``weight``: this rank's LM-head shard [vocab_rank, hidden] BF16 (read in
    place); ``box``: the ``ArgmaxBox`` it shares with ``argmax.PreparedArgmax``."""

    rows = TOKENS

    def __init__(self, tp, rank, box, weight, device):
        self.rank, self.box = rank, box
        vocab_rank, self.hidden = weight.shape
        arch = architecture(device)
        with torch.accelerator.device_index(device.index):
            self.lmhead, self.lmhead_kernel = compile_lmhead(
                vocab_rank, self.hidden, arch
            )
            self.argmax_kernel = compile_key_argmax(
                tp, vocab_rank, self.lmhead.ctas, arch
            )
        self.weight = weight
        self.keys = torch.empty(
            (TOKENS, self.lmhead.ctas), dtype=torch.uint32, device=device
        )

    def takes(self, hidden_states):
        return takes(hidden_states, self.hidden)

    def run(self, hidden_states, out=None):
        """Global argmax [rows] int64 of ``hidden_states @ weight.T`` per row."""
        rows = hidden_states.shape[0]
        if not self.takes(hidden_states):
            raise ValueError(f"fused LM-head argmax: rows {tuple(hidden_states.shape)}")
        if out is None:
            out = torch.empty(rows, dtype=torch.int64, device=hidden_states.device)
        self.lmhead_kernel(self.weight, hidden_states, self.keys, rows)
        box = self.box
        self.argmax_kernel(
            self.keys,
            out,
            box.ordinals,
            box.buffer.view(torch.uint32),
            box.mc,
            self.rank,
        )
        return out


class PreparedLmHeadLogits:
    """BF16 logits [rows, vocab_rank] of this rank's LM-head shard for up to 8 rows
    (``LmHeadLogits``; the shard is read in place)."""

    rows = TOKENS

    def __init__(self, weight, device):
        self.vocab_rank, self.hidden = weight.shape
        with torch.accelerator.device_index(device.index):
            self.lmhead, self.kernel = compile_lmhead_logits(
                self.vocab_rank, self.hidden, architecture(device)
            )
        self.weight = weight

    def takes(self, hidden_states):
        return takes(hidden_states, self.hidden)

    def run(self, hidden_states, out=None):
        rows = hidden_states.shape[0]
        if not self.takes(hidden_states):
            raise ValueError(f"LM-head logits: rows {tuple(hidden_states.shape)}")
        if out is None:
            out = torch.empty(
                (rows, self.vocab_rank),
                dtype=torch.bfloat16,
                device=hidden_states.device,
            )
        self.kernel(self.weight, hidden_states, out, rows)
        return out


# The LM-head shard GEMM of 10-64 rows (verify B2-B8, the drafter's 14-56 rows): the
# split-K projection core (cutedsl.gemm.projection, TMA weight loads with the L2
# evict_first policy, so the 147 MB stream leaves the L2 to the step) without a K
# split: the FP32 accumulation runs in K order, as cuBLAS's unsplit kernels do. 1 GPU,
# [10240, 7168], cold L2, us vs torch.mm: 64 rows 17.2 vs 16.9, 56 rows 17.4 vs 16.8,
# 16 rows 16.7 vs 16.3, 32 rows 16.8 vs 19.1 (cuBLAS splits K at 17-32 rows). BF16
# logits bitwise equal to cuBLAS wherever cuBLAS does not split K (9 rows: 35 us).
WIDE_TACTIC = (64, 64, 1, 7)
WIDE_ROWS = range(10, 65)


class PreparedWideLmHead:
    """BF16 logits [rows, vocab_rank] of this rank's LM-head shard for 10-64 rows,
    into a persistent buffer (so a capture records fixed addresses). ``weight``: the
    shard [vocab_rank, hidden] BF16, read in place. Prepare outside captures."""

    def __init__(self, weight, device):
        from ..gemm.projection import PreparedProjection

        self.vocab_rank, self.hidden = weight.shape
        policies = dict.fromkeys(WIDE_ROWS, WIDE_TACTIC)
        with torch.accelerator.device_index(device.index):
            self.projection = PreparedProjection(weight, policies).prepare()
        self.logits = torch.empty(
            (WIDE_ROWS[-1], self.vocab_rank), dtype=torch.bfloat16, device=device
        )

    def takes(self, hidden_states):
        return (
            hidden_states.shape[0] in WIDE_ROWS
            and hidden_states.shape[1] == self.hidden
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.stride(1) == 1
        )

    def run(self, hidden_states):
        rows = hidden_states.shape[0]
        if not self.takes(hidden_states):
            raise ValueError(f"wide LM head: rows {tuple(hidden_states.shape)}")
        return self.projection.run(hidden_states, self.logits[:rows])
