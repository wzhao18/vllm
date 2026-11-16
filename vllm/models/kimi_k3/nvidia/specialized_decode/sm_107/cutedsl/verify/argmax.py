# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Greedy argmax of the target's logits rows over a vocabulary-sharded LM head, without
gathering the full-vocabulary logits (spec-decode verification at temperature 0).

Each rank keeps its LM-head shard's logits [rows, vocab_rank] (the stock GEMM, so the
values are bitwise stock). One CTA per row takes the shard's (value, lowest index) as a
key, multicasts it into a Lamport mailbox on every rank, polls the row's ``tp``
keys and writes the global argmax: the lowest index among equal maxima, as
``torch.argmax`` and the stock greedy rejection sampler pick it.

Mailbox: [parity, row, rank] u64 keys (0 = unpublished; a published key has no zero
word). A row uses parity ``n % 2`` on its ``n``-th launch (a per-row device ordinal,
which only that row's CTA reads and writes; every rank runs the same rows each launch)
and clears the keys it consumed before the launch ends. A peer writes that parity again
two launches of the row later, after this rank's next launch of it has published, so
after the kernel boundary that retired the clear.
"""

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
import torch.distributed._symmetric_memory as symm_mem
from cutlass import Int32, Int64, Uint32, Uint64
from cutlass.cute.runtime import make_fake_stream

from ..common.host import architecture, fake
from ..common.primitives import (
    alloc,
    asm,
    launch,
    load_words,
    multicast_words,
    redux,
    store_relaxed,
)

THREADS = 256
WARPS = THREADS // 32
VECTOR = 8  # BF16 logits per 16-byte load
MAX_ROWS = 512  # logits rows per launch (the widest verify graph: one launch, so one
# cross-rank key exchange, per step)


@asm(Uint32, pure=True)
def umax(a, b):
    """Unsigned max (the local keys use all 32 bits)."""
    return "max.u32 $0,$1,$2;", "=r,r,r", (Uint32(a), Uint32(b))


@cute.jit
def local_key(bits: Uint32, index):
    """(value, lowest index wins) of one BF16 logit of this rank's shard as a u32: a
    monotone 16-bit image of the BF16 ``bits`` (sign flip; -0 as +0, which an FP32
    compare finds equal; NaN above everything, as torch.argmax), then 0xFFFF - the
    shard-local ``index`` (< 2^16). Never zero."""
    if bits == Uint32(0x8000):
        bits = Uint32(0)
    flip = Uint32(0x8000)
    if (bits & Uint32(0x8000)) != Uint32(0):
        flip = Uint32(0xFFFF)
    image = bits ^ flip
    if (bits & Uint32(0x7FFF)) > Uint32(0x7F80):
        image = Uint32(0xFFFF)
    return (image << Uint32(16)) | (Uint32(0xFFFF) - Uint32(index))


@cute.jit
def global_key(key: Uint32, first):
    """The rank's best ``local_key`` as a u64 that orders across ranks (value, then the
    lowest vocabulary index; ``first``: the shard's first index). Below 2^63 (signed and
    unsigned order agree), no zero word."""
    index = Uint32(first) + (Uint32(0xFFFF) - (key & Uint32(0xFFFF)))
    high = (key >> Uint32(16)) + Uint32(1)
    return (Uint64(high) << 32) | Uint64(Uint32(0xFFFFFFFF) - index)


@cute.jit
def shuffle_max(key: Uint64, offset):
    low = cute.arch.shuffle_sync_bfly(Uint32(key & Uint64(0xFFFFFFFF)), offset)
    high = cute.arch.shuffle_sync_bfly(Uint32(key >> 32), offset)
    return max(key, (Uint64(high) << 32) | Uint64(low))


def box_words(tp):
    """u32 words of both parities' keys."""
    return 2 * MAX_ROWS * tp * 2


class VocabArgmax:
    """Args: logits [rows, vocab_rank] BF16 (this rank's shard, row-major), out [rows]
    int64 (written), ordinals [MAX_ROWS] int32, mailbox (u32 view), multicast address,
    rank. One CTA per row; ``rows`` is dynamic (at most MAX_ROWS)."""

    def __init__(self, tp, vocab_rank):
        assert vocab_rank % (THREADS * VECTOR) == 0 and vocab_rank <= 1 << 16
        assert 1 < tp <= 32
        self.tp, self.vocab_rank = tp, vocab_rank
        self.per_thread = vocab_rank // (THREADS * VECTOR)

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, cute.size(args[0], mode=[0]), THREADS)

    @cute.kernel
    def kernel(self, args):
        logits, out, ordinals, box, mc, rank = args
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        lane, warp = tid % 32, tid // 32
        smem = cutlass.utils.SmemAllocator()
        best = alloc(smem, Uint32, WARPS, 16)
        # Logits and ordinals are read after the wait (written by the step's kernels).
        cute.arch.griddepcontrol_wait()
        launch_ordinal = ordinals[row]
        parity = launch_ordinal & 1
        # This rank's (value, lowest index) of the row.
        source = logits.iterator + row * self.vocab_rank
        key = Uint32(0)
        for j in cutlass.range_constexpr(self.per_thread):
            column = (j * THREADS + tid) * VECTOR
            words = load_words(source + column, count=4, mutable=True)
            for w in cutlass.range_constexpr(4):
                low = local_key(words[w] & Uint32(0xFFFF), column + 2 * w)
                high = local_key(words[w] >> Uint32(16), column + 2 * w + 1)
                key = umax(key, umax(low, high))
        key = redux(key, "max")
        if lane == 0:
            best[warp] = key
        cute.arch.sync_threads()
        entries = (parity * MAX_ROWS + row) * self.tp  # this row's keys, in u64
        if tid == 0:
            for w in cutlass.range_constexpr(1, WARPS):
                key = umax(key, best[w])
            mine = global_key(key, rank * self.vocab_rank)
            multicast_words(
                mc + Int64((entries + rank) * 8),
                (Uint32(mine & Uint64(0xFFFFFFFF)), Uint32(mine >> 32)),
                # A bit-exact store (multimem.st vectors need a float type).
                kind="f32",
            )
        # Every rank's key of this row: lane r of warp 0 polls rank r's entry.
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
                # Consumed: clear it for this row's launch after next.
                store_relaxed(entry, (Uint32(0), Uint32(0)))
            for offset in (1, 2, 4, 8, 16):
                got = shuffle_max(got, offset)
            if lane == 0:
                token = Uint32(0xFFFFFFFF) - Uint32(got & Uint64(0xFFFFFFFF))
                out[row] = Int64(token)
                ordinals[row] = launch_ordinal + 1


class ArgmaxBox:
    """Both parities' key mailboxes (symmetric, multicast) and the per-row ordinals."""

    def __init__(self, group, device, tp):
        self.buffer = symm_mem.empty((box_words(tp),), dtype=torch.int32, device=device)
        self.handle = handle = symm_mem.rendezvous(self.buffer, group)
        self.mc = int(handle.multicast_ptr or 0)
        if not self.mc:
            raise RuntimeError("K3 vocabulary-sharded argmax needs multicast memory")
        self.buffer.zero_()
        self.ordinals = torch.zeros(MAX_ROWS, dtype=torch.int32, device=device)
        torch.accelerator.synchronize(device)
        torch.distributed.barrier(group=group)


@cache
def compile_argmax(tp, vocab_rank, arch):
    kernel = VocabArgmax(tp, vocab_rank)
    rows = cute.sym_int()
    args = (
        fake((rows, vocab_rank)),
        fake((rows,), None, torch.int64, 8),
        fake((MAX_ROWS,), None, torch.int32, 4),
        fake((box_words(tp),), None, torch.uint32, 16),
        Int64(0),
        Int32(0),
    )
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    options = f"--gpu-arch={arch} --enable-tvm-ffi"
    return cute.compile(kernel, *args, stream, options=options)


class PreparedArgmax:
    """The vocabulary-sharded greedy argmax of one rank (every rank calls it together:
    each launch exchanges keys with every rank)."""

    def __init__(self, tp, rank, group, vocab_rank, device):
        self.tp, self.rank, self.vocab_rank = tp, rank, vocab_rank
        self.device = device
        with torch.accelerator.device_index(device.index):
            self.kernel = compile_argmax(tp, vocab_rank, architecture(device))
        self.box = ArgmaxBox(group, device, tp)

    def prepare(self):
        """Run once on every rank together (module load outside any capture)."""
        logits = torch.zeros(
            (8, self.vocab_rank), dtype=torch.bfloat16, device=self.device
        )
        self.run(logits)
        torch.accelerator.synchronize(self.device)

    def run(self, logits, out=None):
        """``logits`` [rows, vocab_rank] BF16, row-major: this rank's shard. Returns
        the global argmax of each row [rows] int64."""
        rows = logits.shape[0]
        if (
            not 0 < rows <= MAX_ROWS
            or logits.shape[1] != self.vocab_rank
            or logits.dtype != torch.bfloat16
            or not logits.is_contiguous()
        ):
            raise ValueError(f"vocabulary-sharded argmax: logits {tuple(logits.shape)}")
        if out is None:
            out = torch.empty(rows, dtype=torch.int64, device=logits.device)
        box = self.box
        self.kernel(
            logits, out, box.ordinals, box.buffer.view(torch.uint32), box.mc, self.rank
        )
        return out
