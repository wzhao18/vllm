# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""All-gather of small per-row entries across the TP ranks through a Lamport mailbox
(``runtime.sampled_verify``'s two exchange rounds).

Each rank holds ``entries`` [rows, WORDS] u32 (one entry per logits row; every word
nonzero: the writer encodes 0 as another word). One CTA per row multicasts the row's
entry into slot [parity, row, rank] of every rank's mailbox (16-byte chunks: a chunk is
read only once all four of its words are nonzero), polls the row's ``tp`` slots (one
thread per chunk), writes them to ``gathered`` [rows, tp, WORDS] and clears them
(consume-then-clear).

Protocol as ``argmax.VocabArgmax``: a row uses parity ``n % 2`` on its ``n``-th launch
(a per-row device ordinal, read and bumped only by that row's CTA; every rank runs the
same rows each launch). A peer writes that parity again two launches of the row later,
after this rank's next launch of it has published, so after the kernel boundary that
retired the clear. PDL: the entries and ordinals are read after
``griddepcontrol.wait``; no early trigger.
"""

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
import torch.distributed._symmetric_memory as symm_mem
from cutlass import Int32, Int64, Uint32
from cutlass.cute.runtime import make_fake_stream

from ..common.host import architecture, fake
from ..common.primitives import launch, load_words, multicast_words, store_relaxed

MAX_ROWS = 512  # logits rows per launch (the widest verify graph)


def box_words(tp, words):
    """u32 words of both parities' slots."""
    return 2 * MAX_ROWS * tp * words


class RowExchange:
    """Args: entries [rows, words] u32 (row-major), gathered [rows, tp, words] u32
    (written), ordinals [MAX_ROWS] int32, mailbox (u32), multicast address, rank. One
    CTA per row (a thread per polled chunk); ``rows`` is dynamic (at most MAX_ROWS)."""

    def __init__(self, tp, words):
        assert words % 4 == 0 and 1 < tp <= 32
        self.tp, self.words = tp, words
        self.chunks = words // 4  # 16-byte chunks per entry
        self.threads = -(-tp * self.chunks // 32) * 32
        assert self.threads <= 1024

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, cute.size(args[0], mode=[0]), self.threads)

    @cute.kernel
    def kernel(self, args):
        entries, gathered, ordinals, box, mc, rank = args
        lane, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        cute.arch.griddepcontrol_wait()
        launch_ordinal = ordinals[row]
        parity = launch_ordinal & 1
        # This row's slots (u32 words): [parity, row, rank, word].
        first = ((parity * MAX_ROWS + row) * self.tp) * self.words
        if lane < self.chunks:
            words = load_words(
                entries.iterator + row * self.words + 4 * lane, count=4, mutable=True
            )
            offset = first + rank * self.words + 4 * lane
            # A bit-exact store (multimem.st vectors need a float type).
            multicast_words(mc + Int64(offset) * 4, words, kind="f32")
        # Lane (source, chunk) polls that chunk of that rank's slot.
        if lane < self.tp * self.chunks:
            source, chunk = lane // self.chunks, lane % self.chunks
            slot = box.iterator + first + source * self.words + 4 * chunk
            got = load_words(slot, count=4, volatile=True)
            ready = (got[0] != Uint32(0)) & (got[1] != Uint32(0))
            ready = ready & (got[2] != Uint32(0)) & (got[3] != Uint32(0))
            while not ready:
                got = load_words(slot, count=4, volatile=True)
                ready = (got[0] != Uint32(0)) & (got[1] != Uint32(0))
                ready = ready & (got[2] != Uint32(0)) & (got[3] != Uint32(0))
            target = (row * self.tp + source) * self.words + 4 * chunk
            for k in cutlass.range_constexpr(4):
                gathered[target + k] = got[k]
            # Consumed: clear it for this row's launch after next.
            store_relaxed(slot, (Uint32(0),) * 4)
        cute.arch.sync_threads()
        if lane == 0:
            ordinals[row] = launch_ordinal + 1


class ExchangeBox:
    """Both parities' slots (symmetric, multicast) and the per-row ordinals."""

    def __init__(self, group, device, tp, words):
        self.buffer = symm_mem.empty(
            (box_words(tp, words),), dtype=torch.int32, device=device
        )
        self.handle = handle = symm_mem.rendezvous(self.buffer, group)
        self.mc = int(handle.multicast_ptr or 0)
        if not self.mc:
            raise RuntimeError("K3 sampled verification needs multicast memory")
        self.buffer.zero_()
        self.ordinals = torch.zeros(MAX_ROWS, dtype=torch.int32, device=device)
        torch.accelerator.synchronize(device)
        torch.distributed.barrier(group=group)


@cache
def compile_exchange(tp, words, arch):
    kernel = RowExchange(tp, words)
    rows = cute.sym_int()
    args = (
        fake((rows, words), None, torch.uint32, 16),
        fake((cute.sym_int(),), None, torch.uint32, 16),
        fake((MAX_ROWS,), None, torch.int32, 4),
        fake((box_words(tp, words),), None, torch.uint32, 16),
        Int64(0),
        Int32(0),
    )
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    options = f"--gpu-arch={arch} --enable-tvm-ffi"
    return cute.compile(kernel, *args, stream, options=options)


class PreparedExchange:
    """One rank's row exchange of ``words``-word entries (every rank calls it together:
    each launch exchanges with every rank)."""

    def __init__(self, tp, rank, group, words, device):
        self.tp, self.rank, self.words, self.device = tp, rank, words, device
        with torch.accelerator.device_index(device.index):
            self.kernel = compile_exchange(tp, words, architecture(device))
        self.box = ExchangeBox(group, device, tp, words)

    def prepare(self):
        """Run once on every rank together (module load outside any capture)."""
        entries = torch.ones((4, self.words), dtype=torch.int32, device=self.device)
        self.run(entries.view(torch.uint32))
        torch.accelerator.synchronize(self.device)

    def run(self, entries, gathered=None):
        """``entries`` [rows, words] u32 (no zero word). Returns every rank's entries
        [rows, tp, words] u32 (rank order)."""
        rows = entries.shape[0]
        if (
            not 0 < rows <= MAX_ROWS
            or entries.shape[1] != self.words
            or entries.dtype != torch.uint32
            or not entries.is_contiguous()
        ):
            raise ValueError(f"row exchange: entries {tuple(entries.shape)}")
        if gathered is None:
            gathered = torch.empty(
                (rows, self.tp, self.words), dtype=torch.uint32, device=entries.device
            )
        box = self.box
        self.kernel(
            entries,
            gathered.view(-1),
            box.ordinals,
            box.buffer.view(torch.uint32),
            box.mc,
            self.rank,
        )
        return gathered
