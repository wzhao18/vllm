# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The exchange of the FP32 DCP top-k (dcp_topk.DcpTopk): peer stores of each rank's
local top-N instead of an NCCL all-gather of its full local top-K, with an exact
second look at the ranks that may hold more. Stock's selection is unchanged: the
exact top-K of the FP32 logits by (score desc, global id asc) over every rank's
keys (sparse_attn_indexer._merge_dcp_topk_global's FP32 rule, 32 score bits); only
how the ranks find it differs.

Words: as IndexerSelect's "candidates", per key a 64-bit word, low half
0xFFFFFFFF - its global id, high half its FP32 score's order key (0: empty).
Every global id is on one rank only, so the words are distinct, and the top-K is
the K largest words.

  DcpPack    (one CTA per row, never waits on a peer) this rank's local top-K
             words (IndexerSelect "candidates") -> a copy in this rank's
             symmetric ``copies`` row, and its ``count`` <= N largest words (N ~
             1.5 x K / W, round1_words) pushed by peer stores into slot ``rank``
             of the row in every rank's symmetric ``box``, then the slot's count
             word (``count`` and whether this rank holds more than N words). No
             fence, no header: the box is Lamport-armed (below).
  DcpSelect  (one CTA per row) polls its row's W count words, then the counted
             words (each re-armed once read), radix-selects the K-th largest of
             the W x N round-1 words (tau). A rank whose N words are all selected
             while it holds more ("saturated") may hold a top-K word it did not
             send; for those (rare: at most K / N per row) every rank reads the
             saturated rank's whole local top-K from its ``copies`` (peer loads,
             after its release flag in ``ready``) and selects the K-th largest
             again over the union of the other ranks' round-1 words and those
             lists. Then this rank's selected keys -> its physical KV rows, as
             DcpMerge.

Exactness: a word of the global top-K on rank r is in r's local top-K. If r did
not send it, it is below r's N sent words and (being in the top-K) at or above the
K-th of the round-1 words, which bounds the true K-th from below; so all N of r's
words are selected in round 1 and r has more than N: r is saturated, and the union
holds the word. The union holds every top-K word and nothing outside the keys, so
its K-th largest is the true K-th: every rank selects the same exact set. A
saturated rank's N words are selected, so at most K / N ranks are (each row).

Lamport box: an empty box word is (low 1, high 0) (LOW_EMPTY, HIGH_EMPTY). No
pushed word has either half: a key word's low half is >= 2^31 (global ids < 2^31)
and its score key is not 0 (IndexerSelect's empty words are never pushed); a count
word is (2 + more, count + 1). The reader takes a word once neither 32-bit half is
empty, so a word seen half-written (one half new, the other still armed) is not
taken (no 64-bit atomicity is assumed), and re-arms it. That needs every word a
reader may poll to be armed whenever no push of it is in flight (the invariant):
  - the box starts armed (arm_box);
  - DcpSelect re-arms every word it takes: the W count words and each slot's
    first ``count`` words;
  - DcpPack pushes 16-byte fragments of two words, so an odd count's last
    fragment carries one word past the count, which no reader takes: that word is
    the ARMED pair (a zero word there would sit unread and un-re-armed, and a later
    larger count on the same parity could be torn-read as (old low 0, new high),
    which passes ``taken``: global id -1);
  - nothing else writes the box (the 8 bytes after each count word are never
    written nor read).
So after DcpSelect(n) of a row exits, parity n % 2 of the row is armed throughout
(test_dcp_round_exchange_exact checks every word after every call).
The count word comes with no ordering against the key words: each is polled on its
own. The ``copies`` of a saturated rank are released explicitly: that rank's
DcpSelect waits for its DcpPack grid (PDL wait), fences at system scope and stores
the call's tag into every rank's ``ready`` [row, rank]; readers poll it. Round 1
(the critical path) needs no wait on the DcpPack grid: DcpSelect's PDL wait comes
after its first selection, before it reads its own candidates.

Tags: the ordinal and the ``ready`` flags are 64-bit (two 32-bit words, low
first; the carry is explicit), call n's tag n + 1: they never wrap (2^64 calls),
so a flag a rank stored calls ago never equals a later call's tag (a 30-bit tag
wrapped after 2^30 calls of a row, ~2.6 days at 4.8K calls/s). A flag is stored and
polled as one 8-byte vector access at system scope, not atomic as a whole: a poll
may see one half new and the other old. It matches the awaited tag T only if both
halves are T's; the old value is an earlier tag (or 0), never T, so at least one
half came from the store of T, which follows the releasing fence: the match
synchronizes. (No flag moves past T while a peer waits for it: d.)

Ordering: each row has an ordinal (``ordinal``, per rank, bumped by DcpSelect;
every rank runs the same calls, so the ordinals agree); call ``n`` uses parity
``n % 2`` of ``box`` and ``copies`` and tags its ``ready`` flags ``n + 1``. No rank
overwrites a word or copy a peer still reads, and no rank takes a stale word,
whatever the kernels between two calls do (none of them needs a PDL wait for this):

  (a) DcpSelect never triggers its dependents (no griddepcontrol.launch_dependents:
      the trigger is the exit of its last CTA). A kernel starts only after its
      stream (graph) predecessor started and triggered, so every kernel downstream
      of DcpSelect(n), DcpPack(n + 1) included, starts after DcpSelect(n) exited:
      all its loads (each value is consumed), its re-arms and its ordinal bump are
      done.
  (b) Rank A's DcpPack(n + 2) (parity n % 2 again: B's slot A and A's copies)
      starts after A's DcpSelect(n + 1) exited (a), which took B's words of call
      n + 1, stored by B's DcpPack(n + 1) after it started, i.e. (a) after B's
      DcpSelect(n) (the last reader of that parity on B, which re-armed every word
      of it that it read) exited.
  (c) DcpSelect(n) of a row reads and re-arms only parity n % 2 of that row;
      DcpPack(n + 1) writes only parity (n + 1) % 2. Every word DcpSelect does not
      read is armed (the invariant above).
  (d) ``ready`` tags only grow: B stores tag n + 1 after it took A's call n + 1
      words, which A pushed after its DcpSelect(n) (its poll for tag n) exited.

Visibility: (a)-(d) order the kernels' execution; the memory model needs the
peers' accesses ordered too (B's re-arm of a word in its box must precede, in that
word's coherence order, A's push of call n + 2 into it; B's loads of A's copies
must not see A's DcpPack(n + 2)). Explicitly, per thread:
  - every DcpSelect thread ends with fence.acq_rel.sys after its last access of
    the box, the copies and the flags (acquire: it read peers' pushes; release:
    its re-arms);
  - every DcpPack thread starts with fence.acq_rel.sys after its PDL wait, before
    any store (release of everything before it on this rank: the grid dependency
    makes the earlier grids' memory operations visible to this grid first);
  - box words, re-arms and flags are relaxed.sys accesses (morally strong with
    the peers' accesses of the same words); the ordinal is read before DcpSelect's
    PDL wait with a relaxed.gpu load (the previous call's DcpSelect, a completed
    grid upstream, stored it: no stale L1 line).
So B's re-arm -> B's DcpPack(n + 1) fence -> its push, read by A's DcpSelect(n + 1)
-> that thread's fence -> A's DcpPack(n + 2) fence -> its push: a release /
acquire chain at system scope (fence-fence synchronization through the push), and
likewise from B's copies loads to A's next copies stores. Neither fence is behind
outstanding peer stores (DcpPack's comes before its first store; DcpSelect stores
to peers only the rare release flag, itself fenced), so each drains only local
traffic (cost: sm107gap NOTES, gapfix-5).

So the exchange needs every DcpPack(n + 1) downstream of DcpSelect(n) on its rank
(one stream, or a joined fork) and the same calls on every rank (the collective's
own contract); test_dcp_round_stress replays graphs of three calls under random
per-rank skew, with short, odd, empty and growing rows. DcpPack never waits on a
peer, and DcpSelect waits only on the DcpPacks (and a saturated rank's DcpSelect,
which waits on nothing but its own DcpPack), so the grids complete at any row
count (no co-residency assumption)."""

import functools

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass import Int32, Int64, Uint32
from cutlass.cute.runtime import make_fake_stream

from ..common.host import fake
from ..common.primitives import asm, load_words, store_relaxed, store_words
from .dcp_topk import DIGITS, require_below
from .indexer import (
    SELECT_THREADS,
    SELECT_WARPS,
    block_exclusive,
    shared_add,
    table_entry,
    topdown,
)

_WARP = 32
LOW_EMPTY, HIGH_EMPTY = 1, 0  # an armed box word (no pushed word has either half)


def round1_words(topk: int, world: int) -> int:
    """N, the words each rank pushes per row: 1.5 x the fair share K / W, a multiple
    of 32 (#63's rule), at most K."""
    fair = -(-topk // world)
    return min(-(-(3 * fair) // (2 * _WARP)) * _WARP, topk)


def slot_words(local_topk: int) -> int:
    """int32 words of one rank's slot in a row: N key words (2 int32 each), then
    the count word (low 2 + more, high count + 1) and 8 bytes of padding (the
    slots stay 16-byte aligned for the pushes' 16-byte stores)."""
    return 2 * local_topk + 4


def arm_box(box: torch.Tensor) -> torch.Tensor:
    """Every 64-bit word of an int32 ``box`` -> empty (low LOW_EMPTY, high
    HIGH_EMPTY)."""
    pairs = box.view(-1, 2)
    pairs[:, 0] = LOW_EMPTY
    pairs[:, 1] = HIGH_EMPTY
    return box


def rescue_ranks(topk: int, local_topk: int) -> int:
    """The most saturated ranks of a row: their N words are all selected."""
    return topk // local_topk


@asm()
def fence_sys():
    return "fence.acq_rel.sys;", "~{memory}", ()


@asm()
def load_acquire(pointer):
    """One 64-bit box word as two 32-bit halves, an acquire load at system scope
    (the acquire of the push it reads: module doc, Visibility)."""
    code = "ld.acquire.sys.global.v2.u32 {$0,$1}, [$2];"
    return code, "=r,=r,l,~{memory}", (pointer,), (Uint32, Uint32), False


@cute.jit
def taken(lo, hi):
    """A pushed box word: neither half is the armed value."""
    ready = lo != Uint32(LOW_EMPTY)
    ready &= hi != Uint32(HIGH_EMPTY)
    return ready


@cute.jit
def poll_word(at, acquire: cutlass.Constexpr[bool] = False):
    """Spin until the box word at ``at`` is pushed; re-arm it -> (lo, hi).
    ``acquire``: acquire loads at system scope (the count words: module doc,
    Visibility), else relaxed."""
    if cutlass.const_expr(acquire):
        lo, hi = load_acquire(at)
        while not taken(lo, hi):
            lo, hi = load_acquire(at)
    else:
        lo, hi = load_words(at, count=2, scope="sys")
        while not taken(lo, hi):
            lo, hi = load_words(at, count=2, scope="sys")
    store_words(at, (Uint32(LOW_EMPTY), Uint32(HIGH_EMPTY)), scope="sys")
    return Uint32(lo), Uint32(hi)


@cute.jit
def load_ordinal(ordinal, row):
    """Row ``row``'s 64-bit ordinal (int64 [max_rows] at address ``ordinal``) ->
    (low, high) words: a relaxed.gpu load (the previous call's DcpSelect, an earlier
    grid of this device, stored it; DcpSelect reads it before its PDL wait)."""
    lo, hi = load_words(ordinal + Int64(8) * Int64(row), count=2, scope="gpu")
    return Uint32(lo), Uint32(hi)


@cute.jit
def next_tag(lo, hi):
    """Ordinal (lo, hi) + 1 as (low, high) words (the carry explicit): the call's
    64-bit tag and the row's next ordinal."""
    lo1 = lo + Uint32(1)
    hi1 = hi
    if lo1 == Uint32(0):
        hi1 = hi + Uint32(1)
    return lo1, hi1


@cute.jit
def selected(hi, lo, threshold, low) -> Uint32:
    """1 if a word (score key ``hi``, low half ``lo``) is at or above the selection
    (``hi`` > ``threshold``, or equal with ``lo`` >= ``low``), else 0."""
    take = Uint32(0)
    if hi != Uint32(0):
        if hi > threshold:
            take = Uint32(1)
        if hi == threshold and lo >= low:
            take = Uint32(1)
    return take


@cute.jit
def tally(
    hist, phase: cutlass.Constexpr[int], d: cutlass.Constexpr[int], prefix, T, hi, lo
):
    """Add one word to digit ``d``'s histogram: phase 0 its score key, phase 1 (only
    a word whose score key is ``T``) its low half; past the first digit only a word
    whose higher digits are ``prefix``. Empty words (score key 0) never count."""
    shift, width = DIGITS[d]
    key = hi
    if cutlass.const_expr(phase == 1):
        key = lo
    ok = Int32(0)
    if hi != Uint32(0):
        ok = Int32(1)
        if cutlass.const_expr(phase == 1):
            if hi != T:
                ok = Int32(0)
        if cutlass.const_expr(d > 0):
            if (key >> Uint32(shift + width)) != prefix:
                ok = Int32(0)
    if ok != Int32(0):
        bin_ = (key >> Uint32(shift)) & Uint32((1 << width) - 1)
        shared_add(hist.iterator + bin_, 1)


class _Kth:
    """The K-th largest of a CTA's 64-bit words (``visit`` adds this thread's
    words to a histogram): a radix select of the score keys (11, 11, 10 bits),
    then, only when the ties at the threshold outnumber the ones to take, one of
    their low halves. -> (T, L): a word is selected iff its score key > T, or == T
    with low half >= L."""

    @cute.jit
    def radix(self, phase: cutlass.Constexpr[int], need, T, sm, items, tid, warp, lane):
        hist, scan, info = sm
        prefix = Uint32(0)
        for d in cutlass.range_constexpr(3):
            shift, width = DIGITS[d]
            for b in cutlass.range(tid, 1 << width, SELECT_THREADS, unroll=1):
                hist[b] = Uint32(0)
            cute.arch.sync_threads()
            self.visit(phase, d, prefix, T, hist, items)
            cute.arch.sync_threads()
            topdown(hist, 1 << width, need, info, 2 + 2 * d, scan, tid, warp, lane)
            prefix = (prefix << Uint32(width)) | Uint32(info[2 + 2 * d])
            need = need - Uint32(info[3 + 2 * d])
        return prefix, need

    @cute.jit
    def kth(self, need, sm, items, tid, warp, lane):
        hist = sm[0]
        threshold, take = self.radix(0, need, Uint32(0), sm, items, tid, warp, lane)
        ties = hist[Int32(threshold & Uint32(0x3FF))]
        cute.arch.sync_threads()
        low = Uint32(0)
        if ties != take:  # the largest low halves (lowest global ids) among the ties
            low, _ = self.radix(1, take, threshold, sm, items, tid, warp, lane)
        return threshold, low


class DcpPack(_Kth):
    """Per row (one CTA): this rank's candidate words [rows, 2 K] int32 ->
    ``copies`` [parity, row] and its ``count`` largest (N when it holds more, else
    all of them; in candidate order) pushed into slot ``rank`` of the row in every
    rank's ``box``, then the slot's count word. Launches the next kernel
    (DcpSelect) at once."""

    def __init__(self, rows, world, topk, local_topk, max_rows):
        if topk % SELECT_THREADS or local_topk % 2 or not 0 < local_topk <= topk:
            raise ValueError(f"DcpPack: top-k {topk}, N {local_topk}")
        self.rows, self.world, self.topk, self.n = rows, world, topk, local_topk
        self.max_rows, self.slot = max_rows, slot_words(local_topk)

    @cute.jit
    def __call__(
        self,
        candidates: Int64,  # int32 [>= rows, 2 * topk]
        ordinal: Int64,  # int64 [max_rows]
        box_peers: cute.Tensor,  # int64 [world]: every rank's box
        copies: Int64,  # int32 [2, max_rows, 2 * topk]: this rank's
        rank: Int32,
        s,
    ):
        args = (candidates, ordinal, box_peers, copies, rank)
        self.kernel(args).launch(
            grid=(self.rows, 1, 1),
            block=(SELECT_THREADS, 1, 1),
            stream=s,
            use_pdl=True,
        )

    @cute.jit
    def visit(
        self,
        phase: cutlass.Constexpr[int],
        d: cutlass.Constexpr[int],
        prefix,
        T,
        hist,
        items,
    ):
        his, los = items
        for i in cutlass.range_constexpr(len(his)):
            tally(hist, phase, d, prefix, T, his[i], los[i])

    @cute.kernel
    def kernel(self, args):
        candidates, ordinal, box_peers, copies, rank = args
        K, N, W = self.topk, self.n, self.world
        per: cutlass.Constexpr = K // SELECT_THREADS
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(tid // 32)
        lane = tid % 32

        smem = utils.SmemAllocator()
        hist = smem.allocate_tensor(Uint32, cute.make_layout(2048), 16)
        scan = smem.allocate_tensor(Uint32, cute.make_layout(SELECT_WARPS), 16)
        info = smem.allocate_tensor(Int32, cute.make_layout(16), 16)
        out = smem.allocate_tensor(Uint32, cute.make_layout(2 * N), 16)
        sm = hist, scan, info
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        # Release (module doc, Visibility): the threads that store the count words
        # fence before any store of theirs (nothing of theirs outstanding: cheap),
        # ordering everything before this grid on this rank (the last DcpSelect's
        # re-arms and peer loads) before the count words peers acquire.
        if tid < W:
            fence_sys()
        o, _ = load_ordinal(ordinal, row)
        parity = o & Uint32(1)
        width: cutlass.Constexpr = 2 * K * 4  # bytes per row
        src = candidates + Int64(row) * Int64(width)
        mirror = Int64(parity) * Int64(self.max_rows) + Int64(row)
        dst = copies + mirror * Int64(width)
        # Item i of this thread: candidate i x 1024 + tid (coalesced).
        his = []
        los = []
        mine = Uint32(0)
        for i in cutlass.range_constexpr(per):
            j = i * SELECT_THREADS + tid
            lo, hi = load_words(src + Int64(8 * j), count=2, mutable=True)
            store_words(dst + Int64(8 * j), (lo, hi))
            his.append(Uint32(hi))
            los.append(Uint32(lo))
            if Uint32(hi) != Uint32(0):
                mine += Uint32(1)
        _, total = block_exclusive(mine, scan, warp, lane)
        threshold = Uint32(0)
        low = Uint32(0)
        more = Uint32(0)
        if total > Uint32(N):
            more = Uint32(1)
            threshold, low = self.kth(Uint32(N), sm, (his, los), tid, warp, lane)
        # The ``count`` largest in candidate order (exactly N when there are more:
        # the words are distinct), ARMED words after (an odd count's last fragment
        # pushes one: it must stay armed, module doc, Lamport box).
        base = Uint32(0)
        for i in cutlass.range_constexpr(per):
            take = selected(his[i], los[i], threshold, low)
            pos, n = block_exclusive(take, scan, warp, lane)
            if take != Uint32(0) and base + pos < Uint32(N):
                place = Int32(base + pos)
                out[2 * place] = los[i]
                out[2 * place + 1] = his[i]
            base += n
        count = cutlass.min(base, Uint32(N))
        for e in cutlass.range(2 * Int32(count) + tid, 2 * N, SELECT_THREADS, unroll=1):
            fill = Uint32(HIGH_EMPTY)
            if e % 2 == 0:
                fill = Uint32(LOW_EMPTY)
            out[e] = fill
        cute.arch.sync_threads()
        # The counted words (16-byte fragments of two; after an odd count the
        # fragment's second word is the armed pair, never taken), then the count
        # word, into every rank's slot.
        frags = (Int32(count) + 1) // 2
        slot: cutlass.Constexpr = self.slot * 4  # bytes
        offset = (mirror * Int64(W) + Int64(rank)) * Int64(slot)
        for f in cutlass.range(tid, frags * W, SELECT_THREADS, unroll=1):
            peer = f // frags
            k = f - peer * frags
            words = (out[4 * k], out[4 * k + 1], out[4 * k + 2], out[4 * k + 3])
            store_relaxed(Int64(box_peers[peer]) + offset + Int64(16 * k), words)
        if tid < W:
            word = (Uint32(2) + more, count + Uint32(1))
            store_relaxed(Int64(box_peers[tid]) + offset + Int64(8 * N), word)


@functools.cache
def compile_pack(rows, world, topk, local_topk, max_rows, device):
    pack = DcpPack(rows, world, topk, local_topk, max_rows)
    with torch.accelerator.device_index(device):
        peers = fake((world,), (1,), torch.int64, 8)
        s = make_fake_stream()
        args = Int64(0), Int64(0), peers, Int64(0), Int32(0), s
        return cute.compile(pack, *args)


class DcpSelect(_Kth):
    """Per row (one CTA): the exact global top-K from this rank's ``box`` (the
    ranks' round-1 words) and, for saturated ranks, their ``copies`` -> this rank's
    physical KV rows ``rows_out`` [rows, >= K] (compacted in its local key order,
    -1 after), their counts ``valid_out`` and, with ``emit_ids``, the selected
    global ids ``ids_out`` [rows, K] (rank-major, each rank's in its candidate order;
    -1 past the count; tests: every rank then releases its copies). Bumps the row's
    ordinal.

    It never launches the next kernel early (PDL), as DcpMerge: every kernel after
    it reads its rows or counts before its own PDL wait, and the box's reuse
    relies on it (module doc, Ordering (a)). Its own PDL wait (for the DcpPack
    grid, so for this rank's candidates) comes after round 1. A selected key past
    the block table traps (fast or raise: never a silent -1 row)."""

    def __init__(
        self, rows, world, topk, local_topk, max_rows, interleave=1, emit_ids=False
    ):
        if topk % SELECT_THREADS or not 0 < local_topk <= topk:
            raise ValueError(f"DcpSelect: top-k {topk}, N {local_topk}")
        self.rows, self.world, self.topk, self.n = rows, world, topk, local_topk
        self.max_rows, self.slot = max_rows, slot_words(local_topk)
        self.interleave, self.emit_ids = interleave, emit_ids
        self.items = -(-(world * local_topk) // SELECT_THREADS)
        self.spare = rescue_ranks(topk, local_topk)

    @cute.jit
    def __call__(
        self,
        box: Int64,  # int32 [2, max_rows, world, slot]: this rank's
        copy_peers: cute.Tensor,  # int64 [world]: every rank's copies
        ready_peers: cute.Tensor,  # int64 [world]: every rank's ready flags
        ordinal: Int64,  # int64 [max_rows]
        candidates: Int64,  # int32 [>= rows, 2 * topk]: this rank's
        req_id: cute.Tensor,  # int32 [>= rows]
        table_address: Int64,
        table_pitch: Int32,
        table_width: Int32,
        block_size: Int32,
        stride_rows: Int32,
        rank: Int32,
        rows_out: cute.Tensor,
        valid_out: cute.Tensor,
        ids_out: cute.Tensor,
        s,
    ):
        table = (table_address, table_pitch, table_width)
        meta = (req_id, table, block_size, stride_rows, rank)
        args = (box, copy_peers, ready_peers, ordinal, candidates, meta)
        self.kernel(args, rows_out, valid_out, ids_out).launch(
            grid=(self.rows, 1, 1),
            block=(SELECT_THREADS, 1, 1),
            stream=s,
            use_pdl=True,
        )

    @cute.jit
    def visit(
        self,
        phase: cutlass.Constexpr[int],
        d: cutlass.Constexpr[int],
        prefix,
        T,
        hist,
        items,
    ):
        """Round 1: the W x N words (``items``: (his, los, srcs, None)); the rescue
        (``items``: (his, los, srcs, rescue)): those of the unsaturated ranks and
        every saturated rank's whole local top-K (peer loads, again each pass)."""
        his, los, srcs, rescue = items
        for i in cutlass.range_constexpr(len(his)):
            if cutlass.const_expr(rescue is None):
                tally(hist, phase, d, prefix, T, his[i], los[i])
            else:
                hi = his[i]
                if rescue[0][srcs[i]] != Int32(0):
                    hi = Uint32(0)
                tally(hist, phase, d, prefix, T, hi, los[i])
        if cutlass.const_expr(rescue is not None):
            _, sat, nsat, copy_peers, mirror, tid = rescue
            per: cutlass.Constexpr = self.topk // SELECT_THREADS
            for s in cutlass.range_constexpr(self.spare):
                if Int32(s) < nsat:
                    base = Int64(copy_peers[sat[s]]) + mirror
                    for j in cutlass.range_constexpr(per):
                        word_at = base + Int64(8 * (j * SELECT_THREADS + tid))
                        lo, hi = load_words(word_at, count=2, scope="sys")
                        tally(hist, phase, d, prefix, T, Uint32(hi), Uint32(lo))

    @cute.jit
    def count(self, items):
        """This thread's words of a rescue's union."""
        his, los, srcs, rescue = items
        satflag, sat, nsat, copy_peers, mirror, tid = rescue
        mine = Uint32(0)
        for i in cutlass.range_constexpr(len(his)):
            if his[i] != Uint32(0) and satflag[srcs[i]] == Int32(0):
                mine += Uint32(1)
        per: cutlass.Constexpr = self.topk // SELECT_THREADS
        for s in cutlass.range_constexpr(self.spare):
            if Int32(s) < nsat:
                base = Int64(copy_peers[sat[s]]) + mirror
                for j in cutlass.range_constexpr(per):
                    word_at = base + Int64(8 * (j * SELECT_THREADS + tid))
                    _, hi = load_words(word_at, count=2, scope="sys")
                    if Uint32(hi) != Uint32(0):
                        mine += Uint32(1)
        return mine

    @cute.jit
    def release(self, ready_peers, row, rank, tag, tid):
        """This rank's copies row (written by its DcpPack grid) -> every rank's
        ``ready`` [row, rank] = ``tag`` (64-bit: (low, high)), after the DcpPack grid
        and a system-scope fence (the caller synchronized the CTA after the PDL
        wait)."""
        W = self.world
        if tid < W:
            fence_sys()
            at = Int64(ready_peers[tid]) + (Int64(row) * Int64(W) + Int64(rank)) * 8
            store_words(at, tag, scope="sys")

    @cute.jit
    def acquire(self, ready, row, source, tag):
        """Spin until rank ``source`` released its copies row of this call (both
        halves of the 64-bit flag equal to ``tag``: module doc, Tags)."""
        at = Int64(ready[0]) + (Int64(row) * Int64(self.world) + Int64(source)) * 8
        lo, hi = load_words(at, count=2, scope="sys")
        pending = (Uint32(lo) ^ tag[0]) | (Uint32(hi) ^ tag[1])
        while pending != Uint32(0):
            lo, hi = load_words(at, count=2, scope="sys")
            pending = (Uint32(lo) ^ tag[0]) | (Uint32(hi) ^ tag[1])
        fence_sys()

    @cute.kernel
    def kernel(self, args, rows_out, valid_out, ids_out):
        box, copy_peers, ready_peers, ordinal, candidates, meta = args
        req_id, table, block_size, stride_rows, rank = meta
        K, N, W = self.topk, self.n, self.world
        per: cutlass.Constexpr = K // SELECT_THREADS
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(tid // 32)
        lane = tid % 32

        smem = utils.SmemAllocator()
        hist = smem.allocate_tensor(Uint32, cute.make_layout(2048), 16)
        scan = smem.allocate_tensor(Uint32, cute.make_layout(SELECT_WARPS), 16)
        info = smem.allocate_tensor(Int32, cute.make_layout(16), 16)
        more = smem.allocate_tensor(Int32, cute.make_layout(W), 16)
        counts = smem.allocate_tensor(Int32, cute.make_layout(W), 16)
        hits = smem.allocate_tensor(Uint32, cute.make_layout(W), 16)
        satflag = smem.allocate_tensor(Int32, cute.make_layout(W), 16)
        sat = smem.allocate_tensor(Int32, cute.make_layout(self.spare + 1), 16)
        sm = hist, scan, info
        # (No PDL wait yet: round 1 reads only the Lamport box and the ordinal,
        # which the previous call's DcpSelect bumped before it exited.)
        o, o_hi = load_ordinal(ordinal, row)
        parity = o & Uint32(1)
        tag = next_tag(o, o_hi)  # 64-bit (module doc, Tags): never wraps
        slot: cutlass.Constexpr = self.slot * 4  # bytes
        width: cutlass.Constexpr = 2 * K * 4  # bytes of a copies row
        mirror = Int64(parity) * Int64(self.max_rows) + Int64(row)
        row_box = box + mirror * Int64(W * slot)
        # The W count words of this row: thread r polls (and re-arms) rank r's.
        if tid < W:
            count_at = row_box + Int64(tid) * Int64(slot) + Int64(8 * N)
            lo, hi = poll_word(count_at, acquire=True)
            more[tid] = Int32(lo & Uint32(1))
            counts[tid] = Int32(hi) - Int32(1)
            hits[tid] = Uint32(0)
            satflag[tid] = Int32(0)
        cute.arch.sync_threads()
        # Round 1: item i of this thread is word g = i x 1024 + tid of the W x N;
        # a rank's words past its count are empty (not read).
        his = []
        los = []
        srcs = []
        mine = Uint32(0)
        for i in cutlass.range_constexpr(self.items):
            g = i * SELECT_THREADS + tid
            src = cutlass.min(g // N, Int32(W - 1))
            hi = Uint32(0)
            lo = Uint32(0)
            if g < W * N:
                if g - src * N < counts[src]:
                    word_at = row_box + Int64(src) * Int64(slot) + Int64(8 * (g - src * N))
                    lo, hi = poll_word(word_at)
            his.append(hi)
            los.append(lo)
            srcs.append(src)
            if hi != Uint32(0):
                mine += Uint32(1)
        _, total = block_exclusive(mine, scan, warp, lane)
        threshold = Uint32(0)
        low = Uint32(0)
        items = (his, los, srcs, None)
        if total > Uint32(K):
            threshold, low = self.kth(Uint32(K), sm, items, tid, warp, lane)
        # Saturated: a rank holding more than N whose N words are all selected.
        for i in cutlass.range_constexpr(self.items):
            if selected(his[i], los[i], threshold, low) != Uint32(0):
                shared_add(hits.iterator + srcs[i], 1)
        cute.arch.sync_threads()
        if tid == 0:
            nsat = Int32(0)
            for r in cutlass.range_constexpr(W):
                if more[r] != Int32(0) and hits[r] == Uint32(N):
                    require_below(nsat, Int32(self.spare))  # (N x saturated <= K)
                    sat[nsat] = Int32(r)
                    satflag[r] = Int32(1)
                    nsat += Int32(1)
            info[0] = nsat
        # The DcpPack grid (this rank's candidates and copies) before anything else.
        cute.arch.griddepcontrol_wait()
        cute.arch.sync_threads()
        nsat = info[0]
        ready = (Int64(ready_peers[rank]),)
        if cutlass.const_expr(self.emit_ids):
            self.release(ready_peers, row, rank, tag, tid)  # (every rank: ids_out)
        else:
            if satflag[rank] != Int32(0):
                self.release(ready_peers, row, rank, tag, tid)
        if nsat != Int32(0):
            if tid < nsat:
                self.acquire(ready, row, sat[tid], tag)
            cute.arch.sync_threads()
            # The union: the other ranks' round-1 words and the saturated ranks'
            # whole local top-K.
            rescue = (satflag, sat, nsat, copy_peers, mirror * Int64(width), tid)
            union = (his, los, srcs, rescue)
            _, total2 = block_exclusive(self.count(union), scan, warp, lane)
            threshold = Uint32(0)
            low = Uint32(0)
            if total2 > Uint32(K):
                threshold, low = self.kth(Uint32(K), sm, union, tid, warp, lane)
        # This rank's selected keys (its own candidates), in its local key order.
        own = candidates + Int64(row) * Int64(width)
        base = Uint32(0)
        for j0 in cutlass.range_constexpr(per):
            j = j0 * SELECT_THREADS + tid
            lo, hi = load_words(own + Int64(8 * j), count=2, mutable=True)
            take = selected(Uint32(hi), Uint32(lo), threshold, low)
            pos, n = block_exclusive(take, scan, warp, lane)
            # (at most K: the words are distinct, every global id on one rank)
            if take != Uint32(0) and base + pos < Uint32(K):
                gid = Int32(Uint32(0xFFFFFFFF) - Uint32(lo))
                span: cutlass.Constexpr = self.interleave
                local = (gid // (span * W)) * span + gid % span
                block = local // block_size
                require_below(block, table[2])  # a key past the block table
                page = table_entry(table, req_id[row], block)
                slot_row = page * stride_rows + local % block_size
                rows_out[row, Int32(base + pos)] = slot_row
            base += n
        base = cutlass.min(base, Uint32(K))
        if tid == 0:
            valid_out[row] = Int32(base)
        for e in cutlass.range(Int32(base) + tid, K, SELECT_THREADS, unroll=1):
            rows_out[row, e] = Int32(-1)
        if cutlass.const_expr(self.emit_ids):
            # Deterministic: rank-major, each rank's candidates in order (peer loads
            # of every rank's copies, each released).
            if tid < W:
                self.acquire(ready, row, tid, tag)
            cute.arch.sync_threads()
            filled = Uint32(0)
            for r in cutlass.range_constexpr(W):
                peer = Int64(copy_peers[r]) + mirror * Int64(width)
                for j0 in cutlass.range_constexpr(per):
                    j = j0 * SELECT_THREADS + tid
                    lo, hi = load_words(peer + Int64(8 * j), count=2, scope="sys")
                    take = selected(Uint32(hi), Uint32(lo), threshold, low)
                    pos, n = block_exclusive(take, scan, warp, lane)
                    if take != Uint32(0) and filled + pos < Uint32(K):
                        gid = Int32(Uint32(0xFFFFFFFF) - Uint32(lo))
                        ids_out[row, Int32(filled + pos)] = gid
                    filled += n
            filled = cutlass.min(filled, Uint32(K))
            for e in cutlass.range(Int32(filled) + tid, K, SELECT_THREADS, unroll=1):
                ids_out[row, e] = Int32(-1)
        cute.arch.sync_threads()
        if tid == 0:
            at = ordinal + Int64(8) * Int64(row)
            store_words(at, tag, scope="gpu")


@functools.cache
def compile_round_select(
    rows, world, topk, local_topk, max_rows, interleave, emit_ids, device
):
    select = DcpSelect(rows, world, topk, local_topk, max_rows, interleave, emit_ids)
    with torch.accelerator.device_index(device):
        peers = fake((world,), (1,), torch.int64, 8)
        ready = fake((world,), (1,), torch.int64, 8)
        req = fake((rows,), (1,), torch.int32, 4)
        out = fake((rows, topk), (topk, 1), torch.int32, 4)
        valid = fake((rows,), (1,), torch.int32, 4)
        ids = fake((rows, topk), (topk, 1), torch.int32, 4)
        scalars = Int64(0), Int32(0), Int32(0), Int32(0), Int32(0), Int32(0)
        s = make_fake_stream()
        args = Int64(0), peers, ready, Int64(0), Int64(0), req, *scalars
        return cute.compile(select, *args, out, valid, ids, s)


class DcpRoundBox:
    """The symmetric state of the exchange over one DCP group, shared by every DSA
    layer of a device with the same (rows, top-k): ``box`` [2, max_rows, W, slot]
    (armed) and ``copies`` [2, max_rows, 2 K] int32, ``ready`` [max_rows, W]
    int64 (peer stores and loads only: no multicast object), every rank's
    addresses, and the per-row int64 ordinals. Collective over ``group``: every
    rank builds it in the same order, outside any CUDA graph capture."""

    def __init__(self, device, max_rows, topk, world, rank, group):
        import torch.distributed as dist
        import torch.distributed._symmetric_memory as symm_mem

        from ...symm import peer_rendezvous

        self.world, self.rank, self.topk, self.max_rows = world, rank, topk, max_rows
        self.n = round1_words(topk, world)
        i32 = dict(dtype=torch.int32, device=device)
        self.box = symm_mem.empty((2, max_rows, world, slot_words(self.n)), **i32)
        self.copies = symm_mem.empty((2, max_rows, 2 * topk), **i32)
        i64 = dict(dtype=torch.int64, device=device)
        self.ready = symm_mem.empty((max_rows, world), **i64)  # 64-bit tags
        tensors = (self.box, self.copies, self.ready)
        handles = [peer_rendezvous(t, group) for t in tensors]
        arm_box(self.box)  # no call yet
        self.copies.zero_()
        self.ready.zero_()  # (tags start at 1)
        self.box_peers = torch.tensor(handles[0].buffer_ptrs, **i64)
        self.copy_peers = torch.tensor(handles[1].buffer_ptrs, **i64)
        self.ready_peers = torch.tensor(handles[2].buffer_ptrs, **i64)
        self.ordinal = torch.zeros(max_rows, **i64)  # 64-bit: tags never wrap
        torch.accelerator.synchronize(device)
        dist.barrier(group=group)


_BOXES: dict = {}


def round_box(device, max_rows, topk, world, rank, group):
    """The DcpRoundBox of (device, rows, top-k, group), built on first use
    (collective)."""
    key = (device.index, max_rows, topk, world, id(group))
    if key not in _BOXES:
        _BOXES[key] = DcpRoundBox(device, max_rows, topk, world, rank, group)
    return _BOXES[key]
