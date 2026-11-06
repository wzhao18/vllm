# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DSpark drafter's DCP attention exchanges on the specialized decode's Lamport
mailboxes (mla.dcp), replacing the stock MLADCPManager query all-gather and LSE
combine (and the separate W_UV) of the drafter's MLA layers.

Per layer, after the stock fused query concat (RoPE, FP8, KV insert):
  QGather      this rank's FP8 query rows [R, H, 576] -> every rank's q mailbox
               [R, DCP*H, 576] (multicast), then the grid polls the gathered mailbox
  tokenspeed   the stock decode over the local KV shard for every rank's heads
               (draft rows, masked as stock) -> BF16 rows + LSE (log2); for FAST_CORE
               rows without its reducer (mla.dcp split_partials_q): FP32 split
               partials + LSEs (packed rows t * DCP * H + g)
  DcpValue     (mla.dcp) re-arms the q mailbox, pushes the rows (FAST_CORE: LSE-folds
               the split partials, as the target's verify fold) to their head owners,
               LSE-merges, W_UV; the gate input is a constant +inf row (sigmoid 1:
               the drafter has no output gate), so the output is W_UV's BF16 rows
Parity alternates by drafter layer, as the target's MLA layers. Without the gather
(FORK_ROWS, POLLED_CORE) the core polls the q rows it reads instead.

Causality (``DraftDcp.causal``): the drafter's MLA layers' own flag (the K3 DSpark
layers are built non_causal_multi_token_decode, so both kimi-k3-dspark and
kimi-k3.1-pro-dspark-preview run non-causal, whatever dflash_config says), which
stock's MLA metadata builder puts in the attention metadata that tokenspeed's
causal_mask reads; the cores here compile with the same flag (causal: the target
verify's global-position mask, rows mode's per-row lengths from VerifyRows). A
captured call whose metadata says otherwise raises on the target topology
(NQ-NOFALLBACK; elsewhere the stock path runs it). The "non-causal" notes below
describe the 64-head drafter the tables were measured on.

The drafter's row-keyed tables here (FAST_CORE .. DRAFT_WLIST) are its own (drafter
rows: multiples of the draft length). mla.dcp's split tables (LONG_SPLIT,
VERIFY_FOLD_SPLITS) are keyed by (rows, gathered heads), not by role or q, and shared
with the target's MLA verify: the (rows, 64) keys are the 64-head drafter's (measured
on it, q 3 / 7), but at 96 gathered heads (kimi-k3.1-pro-dspark-preview, 6 per rank at
DCP16) they are the K3 target's, measured at its verify q (4 / 8: other packed tiles
and requests at the same rows). The drafter's split (``draft_fold_splits``) therefore
takes those tables only at the gathered head counts measured on a drafter
(DRAFT_RULE_HEADS), else tokenspeed's own split (the stock drafter's). A rule the
cores and DcpValue still read where that split equals a rule's long split keeps the
partition consistent (both read it), and a gathered fold row it fires on moves to the
polled core (``long_polled``: the gathered fold core has no long partition). Rows mode
(q = 1) keeps the (padded rows, heads) q = 1 rules: the same shape whichever role
measured them. NQ-TUNE: a drafter key in mla.dcp's split tables (rows, heads, role)
and the 96-head drafter's LONG_SPLIT / VERIFY_FOLD_SPLITS / POLLED_CORE / VALID_CORE /
DRAFT_WLIST / QUERY_TOKENS / DRAFT_CORE_PUSH, DL3 B1-B166 and DL7 B1-B36 graphs
(scripts/dsdraft/layer_test.py at 6 heads x DCP16).
"""

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64
from cutlass.cute.runtime import make_fake_stream

from ..common.host import (
    dynamic_fake,
    dynamic_view,
    fake,
    layout,
    prepare_shared_carveout,
    stream,
    to_cute,
)
from ...shapes import Shape, full_capture, refuse
from ..common.policy import batch_table
from ..common.primitives import (
    launch,
    load_volatile,
    load_words,
    multicast_words,
    published,
)
from ..mla.core_q import (
    compile_polled_q,
    polled_partials_q,
    push_table,
    verify_table,
    wide_persistent,
)
from ..mla.dcp import (
    FORK_BALANCE,
    FORK_CLUSTER,
    FORK_REPOLL,
    FORK_VALID,
    FORK_VALID_DELAY,
    GB300_SMS,
    LONG_SPLIT,
    LSE_FOLD,
    Q_SENT,
    DcpBox,
    compile_fork,
    compile_split_partials,
    compile_split_partials_q,
    compile_value,
    fill_splits,
    fork_partials,
    fork_table,
    local_max,
    long_split,
    split_partials,
    split_partials_q,
    split_workspace_ts,
    verify_fold_splits,
)
from ..mla.rows import compile_rows
from ..mla.wlist import shared_list

# Drafter query rows (requests x draft length; Rubin, DCP) whose tokenspeed core skips
# its reducer and folds its FP32 split partials in DcpValue (mla.dcp's q > 1 fold).
FAST_CORE = batch_table("7-64:1")
# Drafter query rows run as single-query rows on the q = 1 path (mla.rows, non-causal:
# every row attends to its request's whole local context), padded to the q = 1 batch
# (value) of the target's tables: the q = 1 tokenspeed core without its reducer
# (mla.dcp split_partials) and DcpValue's LSE fold of its LSE_FOLD splits, instead of
# the packed q > 1 core + fold. rows -> padded rows.
# DL3 (3 draft rows per request): B1 / B2 (3 / 6 rows) take the same rows mode, padded
# to the same 8-row q = 1 batch (its fork options and LSE_FOLD / LONG_SPLIT rows).
ROWS_MODE = batch_table("3:8,6:8,7:8")
# Drafter query rows whose MlaQuery(rope) replaces the stock query ops + QGather.
# (Drafter rows are multiples of the draft length: DL7 7, 14, ..; DL3 3, 6, 9, ..)
FAST_QUERY = batch_table("3-64:1")
# FAST_QUERY rows whose gathered q poll is gated on one chunk per source rank
# (``MlaQuery.gate``, as the target's query.GATED_GATHER). 16 ranks, the drafter forward
# (scripts/dsdraft/layer_test.py), us ungated -> gated: B1 210.1 -> 208.4, B8 416.1 ->
# 399.5; outputs bitwise identical.
GATED_QUERY = batch_table("3-64:1")
# FAST_QUERY rows -> MlaQuery's tiled request tile (not listed: its rule, 32 above 16
# rows at 4 heads): 16-request tiles put 2-4x the CTAs on the drafter's B3-B8 rows.
# 16 ranks, the drafter forward (scripts/dsdraft/layer_test.py), us rule -> 16: B3
# 327.1 -> 320.8, B4 336.3 -> 328.1, B5 344.6 -> 330.7, B8 397.1 -> 384.3; bitwise.
QUERY_TOKENS = batch_table("21:16,28:16,35:16,42:16,49:16,56:16,63:16")
# ROWS_MODE rows whose core is the target's polled tokenspeed fork (mla.core_ts, as
# dcp.POLLED_TS): MlaQuery publishes q without gathering it, and each fork CTA polls
# its own q rows of the mailbox before its Q TMA (the K / V loads do not wait for q).
# 16 ranks, the B1 drafter forward (scripts/dsdraft/layer_test.py), us gathered ->
# fork: 203.0 -> 198.0 at local 560, 221.4 -> 214.5 at local 4096; bitwise identical.
# On Rubin the fork takes the target's per-batch fork options of the padded rows
# (mla.dcp FORK_CLUSTER, FORK_VALID): with FORK_VALID it polls q with
# validity-reporting copies from DcpBox.qv (as VALID_CORE below).
# The fork re-polls its late q chunks together where mla.dcp.FORK_REPOLL lists the
# padded rows: 194.2 -> 193.4 (195.1 -> 193.6) at local 560, 208.5 -> 208.7 at 4096.
FORK_ROWS = batch_table("3:1,6:1,7:1")
# FAST_CORE rows (not rows mode) whose folded core polls its own q rows of the mailbox
# (mla.core_q, the target's polled verify core, non-causal): MlaQuery publishes q
# without gathering it, and each core CTA polls its packed q rows before its Q TMA
# (the K and V loads do not wait for q). 16 ranks, the B8 drafter forward
# (scripts/rt/draft_layer_test.py --ab-polledq), us gathered -> polled: 359.5 -> 347.5
# at local 560, 406.1 -> 395.5 at local 4096; bitwise identical. B2-B5 (14-35 rows),
# 16 ranks, the drafter forward with its serial context step (scripts/dsdraft/
# overlap_test.py), us gathered -> polled: B2 272.9 -> 268.6, B3 282.7 -> 278.2, B4
# 287.5 -> 282.9, B5 311.3 -> 304.6; MlaQuery then does not poll, so their context step
# can also overlap the forward (runtime.draft CONTEXT_OVERLAP).
# DL3 B3 / B4 (9 / 12 rows) as DL7's B2-B5. DL7 B6 / B7 / B9 (42 / 49 / 63 rows) as B8
# (they ran the gathered fold core: serving drafter B6 / B7 0.587 / 0.678 ms vs B8
# 0.547, B9 0.843; C48 iterlogs).
POLLED_CORE = batch_table("9-35:1,42:1,49:1,56:1,63:1")
# POLLED_CORE rows whose core reads q with validity-reporting copies (mla.qvalid,
# core_q's ``valid`` mode: 1 copies then loads Q, 2 the copy is the Q load) from the
# validity-polled mailbox (DcpBox.qv; never the LSU-polled DcpBox.q): MlaQuery
# publishes with ``valid`` (one 16 B store per chunk) and DcpValue re-arms qv
# (``valid_box``). 16 ranks, the B8 drafter forward
# (scripts/rt/draft_layer_test.py --ab-valid), us LSU-polled -> validity copies (mode
# 1): 341.0 / 338.9 -> 335.2 at local 560, 384.4 / 383.6 -> 381.8 at local 4096;
# bitwise.
VALID_CORE = batch_table("28:2,35:2,42:2,49:2,56:2,63:2")
# Rubin wide rows whose persistent split-1 core pushes its BF16 rows and LSEs into the
# owners' widened DcpBox itself, DcpValue only merging (the target's core push,
# mla.prepared.CORE_PUSH; value: the early cap). 16 ranks, 4 heads per rank, 224 rows
# (DL7 B32), us per MLA layer, DcpValue push -> core push: 66.14 -> 58.63
# (mlasplit-dft16, the target's verify core at the drafter's grids). Bitwise.
DRAFT_CORE_PUSH = batch_table("224:1")
# Rows whose persistent core runs the per-step work list (mla.wlist) where mla.dcp's
# LONG_SPLIT launches its long split: the polled fold rows (valid 2) and the wide rows.
# The first drafter layer builds the lists; the core push (split 1) is off for them.
# (225-576: DL3 B65-B192 drafter rows, hb192, with mla.dcp's LONG_SPLIT rule)
DRAFT_WLIST = batch_table("28:1,35:1,42:1,49:1,56:1,63-576:1")
# Gathered head counts (DCP x heads per rank) whose (rows, heads) keys in mla.dcp's
# split tables were measured on the drafter (the 64-head kimi-k3-dspark at DCP16; the
# K3 target verifies 96). See the module notes and ``draft_fold_splits``.
DRAFT_RULE_HEADS = frozenset({64})

THREADS = 256
POLLS = 8  # 16 B gathered-mailbox chunks polled per thread


class QGather:
    """Publish this rank's query rows into every rank's q mailbox, then poll the
    gathered rows (``rows`` x ``size`` x ``heads`` x 576 FP8 bytes) until landed."""

    def __init__(self, rows, heads, size):
        self.rows, self.heads, self.size = rows, heads, size
        self.publish_chunks = rows * heads * 36
        self.poll_chunks = rows * size * heads * 36
        self.ctas = max(1, -(-self.poll_chunks // (THREADS * POLLS)))

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, self.ctas, THREADS)

    @cute.kernel
    def kernel(self, args):
        query, mailbox, mc, rank = args
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        H, D = self.heads, self.size
        threads: cutlass.Constexpr = self.ctas * THREADS
        me = cta * THREADS + tid
        cute.arch.griddepcontrol_wait()
        for c in range(me, self.publish_chunks, threads):
            row, rest = c // (H * 36), c % (H * 36)
            h, w = rest // 36, rest % 36
            words = load_words(
                query.iterator + (row * H + h) * 576 + 16 * w, count=4, mutable=True
            )
            index = (row * D + rank) * H + h
            # FP8 bytes are never 0xFF (satfinite, canonical NaN): no sentinel word.
            multicast_words(mc + Int64(index) * 576 + 16 * w, words)
        last = self.poll_chunks - 1  # threads past the end re-poll it
        per: cutlass.Constexpr = -(-self.poll_chunks // threads)
        for i0 in cutlass.range_constexpr(0, per, POLLS):
            n: cutlass.Constexpr = min(POLLS, per - i0)
            sources = [
                mailbox.iterator + min(me + (i0 + i) * threads, last) * 16
                for i in range(n)
            ]
            loads = [load_volatile(source, count=4) for source in sources]
            for i in cutlass.range_constexpr(n):
                words = loads[i]
                while not published(words, sentinel=Q_SENT):
                    words = load_volatile(sources[i], count=4)
        cute.arch.griddepcontrol_launch_dependents()


@cache
def compile_gather(rows, heads, size, cap, device):
    kernel = QGather(rows, heads, size)
    args = (
        fake((rows, heads, 576), None, torch.uint8),
        fake((cap, size * heads, 576), None, torch.uint8),
        Int64(0),
        Int32(0),
    )
    with torch.accelerator.device_index(device):
        return cute.compile(kernel, *args, make_fake_stream())


def query_heads(heads):
    """Whether MlaQuery(rope) has a drafter tile for ``heads`` heads per rank
    (mla.query DRAFT_TILED_FROM): the fast DCP attention needs it."""
    from ..mla.query import DRAFT_TILED_FROM

    return heads in DRAFT_TILED_FROM


def uncovered_rows(rows, steps, cap, size, sms):
    """The drafter ``rows`` DraftDcp cannot serve by shape alone -> why (the wide rows
    whose DcpValue is not co-resident are found when it compiles)."""
    dropped = {}
    for r in rows:
        if r % steps:
            dropped[r] = "not a whole draft length"
        elif r > cap and size == 1:
            dropped[r] = f"above the {cap}-row box without DCP"
        elif r > cap and sms <= GB300_SMS:
            dropped[r] = f"above the {cap}-row box without the Rubin wide path"
    return dropped


def draft_fold_splits(rows, steps, heads, max_seq, device):
    """(split, packed rows) of the drafter's packed (q = ``steps``) tokenspeed core for
    ``rows`` drafter rows of ``heads`` gathered heads: mla.dcp.verify_fold_splits
    (its Rubin tables VERIFY_FOLD_SPLITS / LONG_SPLIT) at DRAFT_RULE_HEADS, else its
    tokenspeed rule alone (the target's (rows, heads) keys were measured at another
    q). The cores and DcpValue read LONG_SPLIT with the split this returns."""
    if heads in DRAFT_RULE_HEADS:
        return verify_fold_splits(rows, steps, heads, max_seq, device)
    from tokenspeed_mla import mla_decode as md

    # (verify_fold_splits without its Rubin tables)
    tiles = md.compute_q_tile_layout(heads, steps, 128)[1]
    sms = md.get_num_sm(device)
    args = rows // steps, tiles, 128, 512, sms, max_seq, torch.float8_e4m3fn
    args += ((128, 128),)
    split = md._get_split_kv_and_workspace_size(*args, fill_splits(rows, device))[0]
    return split, tiles * 128


@cache
def compile_query(
    rows,
    heads,
    size,
    cap,
    specs,
    device,
    sms,
    max_position,
    rope=True,
    kv_shape=None,
    gate=False,
    gather=True,
    tokens=0,
    valid=False,
):
    """MlaQuery (lean, RoPE) of ``rows`` drafter rows: norms, Q_B, W_UK, RoPE, FP8,
    the KV insert of the owned slots, the q publish and the gathered q poll. The
    cache is dynamically shaped [blocks, page, 576]. ``valid``: the publish keeps
    mla.qvalid's contract (into DcpBox.qv)."""
    from ..mla.query import MlaQuery

    qnorm, kvnorm, qb, uk, qinv, kinv = (
        fake(shape, stride, dtype, 4 if dtype == torch.float32 else 16)
        for shape, stride, dtype in specs
    )
    # The paged cache is bound after this compile: a dynamic layout (pass it as a
    # ``dynamic_view``).
    kv = (
        dynamic_fake(3, torch.uint8)
        if kv_shape is None
        else fake(kv_shape, None, torch.uint8)
    )
    query = fake((rows, heads, 576), (heads * 576, 576, 1), torch.uint8)
    args: tuple = (fake((rows, 2112)), qnorm, kvnorm, qb, uk, kv)
    args += (fake((rows,), (1,), torch.int64, 8), qinv, kinv, Float32(1e-6), query)
    args += (fake((cap, size * heads, 576), None, torch.uint8), Int64(0), Int32(0))
    if rope:
        args += (fake((rows,), (1,), torch.int64, 8),)
        args += (fake((max_position, 64), None, torch.float32, 4),)
    kernel = MlaQuery(
        heads, rows, dcp=size, gather=gather, sms=sms, rope=rope, gate=gate,
        tokens=tokens, valid=valid,
    )  # fmt: skip
    with torch.accelerator.device_index(device):
        compiled = cute.compile(kernel, *args, make_fake_stream())
        if rows > 1:
            compiled = prepare_shared_carveout(compiled, device, 100)
    return compiled


class DraftDcp:
    """The drafter layers' shared DcpBox and per-(rows) kernels; ``rows``: the drafter
    graphs' query rows (requests x draft length); ``max_context`` bounds the local KV
    shard (the folded core's split count), ``interleave``: the KV cache's DCP
    interleave."""

    def __init__(
        self, attentions, group, size, rank, rows, steps, max_context=None,
        interleave=1, causal=False,
    ):  # fmt: skip
        first = attentions[0]
        # NQ-NOFALLBACK: set by runtime.draft on the target topology: a captured
        # drafter graph of a served row raises instead of running stock attention.
        self.strict = False
        # The drafter's attention mask, as stock resolves it from the drafter config
        # (False: every draft row sees its request's whole context and draft block).
        self.causal = bool(causal)
        self.heads = first.num_local_heads
        self.size, self.rank, self.steps = size, rank, steps
        self.device = first.W_UV.device
        self.cap = 64
        self.box = DcpBox(
            group, self.device, size, rank, self.heads, self.cap, interleave
        )
        sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        self.rows = tuple(r for r in rows if r <= self.cap and r % steps == 0)
        # Rows above the box (Rubin): the target's wide path (mla.prepared.choose) on
        # this box widened to the largest of them (DcpBox.widened, its own allocation):
        # MlaQuery publishes q under the validity contract without gathering it, the
        # polled packed core (the drafter's mask, validity Q TMA, its split; persistent
        # by core_q.wide_persistent) and DcpValue (fold when split > 1, spread merge).
        # Rows whose compiled DcpValue cannot be co-resident (mla.dcp.compile_value's
        # check_residency) stay on the stock path. ``dropped``: rows -> why they stay
        # on the stock path (logged per cause by runtime.draft).
        self.dropped = uncovered_rows(rows, steps, self.cap, size, sms)
        self.wide = tuple(r for r in rows if r > self.cap and r not in self.dropped)
        self.wide_box = self.box.widened(max(self.wide)) if self.wide else None
        # Constant gate rows: sigmoid(+inf) = 1 exactly (the drafter has no gate).
        width = self.heads * 128
        self.gate = torch.full(
            (max((self.cap, *self.wide)), width),
            float("inf"),
            dtype=torch.bfloat16,
            device=self.device,
        )
        self.gathers, self.values = {}, {}
        index = self.device.index
        uv = layout(first.W_UV)
        for r in self.rows:
            self.gathers[r] = compile_gather(r, self.heads, size, self.cap, index)
            self.values[r] = compile_value(
                self.heads, size, r, self.cap, uv, width, index, sms, steps
            )
        # Rows-mode rows whose core is the polled fork (FORK_ROWS): their MlaQuery does
        # not gather. The fork re-arms "x" (the target's polled in-projection rows):
        # here a scratch buffer no kernel reads.
        self.fork_rows = {
            r
            for r in self.rows
            if FORK_ROWS.get(r) and r <= ROWS_MODE.get(r, 0) <= self.cap
        }
        self.fork_x = torch.empty(
            (self.cap, 1536), dtype=torch.bfloat16, device=self.device
        )
        # Rows whose tokenspeed core skips its reducer: DcpValue LSE-folds the FP32
        # split partials (mla.dcp split_partials_q, non-causal): rows -> split count.
        # The cores compile once the cache (its page size) is bound (``bind``).
        self.folds, self.fold_values, self.fold_cores = {}, {}, set()
        # NQ-NOFALLBACK: rows left unfolded -> why (the coverage log's ``core_of``).
        self.unfolded: dict[int, str] = {}
        self.max_seq = local_max(max_context or 1 << 17, size, interleave)
        # Packed rows (tiles x 128) of the folded and wide rows (``draft_fold_splits``).
        self.packed: dict[int, int] = {}
        self.scales = (
            first.scale * first._q_scale_float * first._k_scale_float,
            first._k_scale_float,
        )
        heads = size * self.heads
        # Rows whose q mailbox is DcpBox.qv -> their core's validity mode (VALID_CORE;
        # the polled core below; the fork rows' FORK_VALID mode).
        self.valid = {
            r: VALID_CORE[r]
            for r in self.rows
            if VALID_CORE.get(r) and POLLED_CORE.get(r) and not ROWS_MODE.get(r)
        }
        for r in self.rows:
            if r not in FAST_CORE:
                self.valid.pop(r, None)
                self.unfolded[r] = "not in FAST_CORE"
                continue
            split, packed = draft_fold_splits(
                r, steps, heads, self.max_seq, self.device
            )
            if split <= 1:
                self.valid.pop(r, None)
                self.unfolded[r] = f"draft_fold_splits split {split}"
                continue
            self.folds[r] = split
            self.packed[r] = packed
            self.fold_values[r] = compile_value(
                self.heads, size, r, self.cap, uv, width, index, sms, steps, split,
                lse_split=True, out_scale=self.scales[1], valid_box=r in self.valid,
            )  # fmt: skip
            split_workspace_ts(self.device, packed, r // steps, split)
        # Rows mode (ROWS_MODE): rows -> (padded rows, splits, DcpValue, row metadata
        # kernel); the per-row metadata (block table rows, local and global lengths)
        # is written once per forward by the first layer (``row_meta``).
        self.row_modes, self.row_meta, self.forks = {}, {}, {}
        # Rows-mode metadata buffers by (rows, block-table width) (``rows_tables``).
        self.row_tables: dict = {}
        for r in self.rows:
            padded = ROWS_MODE.get(r)
            if not padded or padded > self.cap or padded < r:
                continue
            split = LSE_FOLD.get(padded, 8)
            # (mla.dcp: length-keyed splits; a q = 1 rule: the same shape whichever
            # role measured it)
            rule = LONG_SPLIT.get((padded, heads))
            if rule and sms > GB300_SMS and split == rule[0]:
                split = rule[1]
            valid = 0  # the fork's FORK_VALID mode (Rubin: DcpBox.qv exists)
            if r in self.fork_rows and self.box.qv is not None:
                valid = FORK_VALID.get(padded, 0)
            if valid:
                self.valid[r] = valid
            value = compile_value(
                self.heads, size, padded, self.cap, uv, width, index, sms, 1, split,
                lse_split=True, out_scale=self.scales[1], valid_box=bool(valid),
            )  # fmt: skip
            split_workspace_ts(self.device, heads, padded, split)
            meta = compile_rows(steps, padded, size, rank, interleave, self.causal)
            self.row_modes[r] = (padded, split, value, meta)
            if r in self.fork_rows:  # per parity: the fork's q mailbox / x table
                qbox = self.box.mailbox(bool(valid))
                tables = [
                    fork_table(qbox[p], self.fork_x, padded, self.device)
                    for p in (0, 1)
                ]
                options: tuple = (padded in FORK_BALANCE, padded in FORK_REPOLL)
                options += (padded in FORK_CLUSTER, valid)
                options += (FORK_VALID_DELAY.get(padded, 0),)
                self.forks[r] = (tables, *options)
        # Wide rows: rows -> (split, persistent, DcpValue, per-parity core tables of
        # the widened box's validity-polled q mailbox, the core push's early cap or
        # None: DRAFT_CORE_PUSH).
        self.wide_cores: dict = {}
        self.wide_flags: dict = {}  # (the core push's per-parity block ready flags)
        # Rows on the per-step work list (DRAFT_WLIST): rows -> mla.wlist.WorkList.
        self.wlists: dict = {}
        kept = []
        for r in self.wide:
            reqs, cap = r // steps, self.wide_box.cap
            split, packed = draft_fold_splits(
                r, steps, heads, self.max_seq, self.device
            )
            persistent = wide_persistent(reqs, packed // 128, split, sms)
            work = self.work_list(r, heads, packed // 128, split, sms)
            persistent = persistent or work is not None
            fold = split if split > 1 else 0
            push = None
            if r in DRAFT_CORE_PUSH and split == 1 and persistent and work is None:
                push = DRAFT_CORE_PUSH[r]
            try:
                value = compile_value(
                    self.heads, size, r, cap, uv, width, index, sms, steps, fold,
                    lse_split=fold > 0, out_scale=self.scales[1] if fold else 1.0,
                    valid_box=True, pushed=push is not None,
                )  # fmt: skip
            except RuntimeError as error:  # (check_residency: not co-resident)
                if "co-resident" not in str(error):
                    raise
                self.dropped[r] = "DcpValue grid not co-resident"
                continue
            kept.append(r)
            self.packed[r] = packed
            if fold:
                split_workspace_ts(self.device, packed, reqs, split)
            qbox = self.wide_box.mailbox(True)
            if push is not None:  # q mailbox, owners' boxes, block flags per parity
                blocks = 2 * (packed // 128) * reqs
                i32 = dict(dtype=torch.int32, device=self.device)
                flags = [torch.zeros(blocks, **i32) for _ in (0, 1)]
                self.wide_flags[r] = flags
                tables = [
                    push_table(qbox[p], reqs, self.wide_box, p, self.device, flags[p])
                    for p in (0, 1)
                ]
            else:
                tables = [verify_table(qbox[p], reqs, self.device) for p in (0, 1)]
                if work is not None:
                    tables = [work.table(t) for t in tables]
                    self.wlists[r] = work
            self.wide_cores[r] = (split, persistent, value, tables, push)
        self.wide = tuple(kept)
        # The context step's KV, when the caller runs it on a side stream: the event
        # the first layer's attention waits for before any core (``join_context``).
        self.context_ready = None
        # Folded rows whose core polls its q rows (POLLED_CORE): per parity, the core's
        # table of the q mailbox. Also the gathered fold rows where a LONG_SPLIT rule
        # launches its long split (``long_polled``: the gathered fold core has no long
        # partition, the polled one has; at gathered heads outside DRAFT_RULE_HEADS
        # only where tokenspeed's own split equals a rule's long split).
        self.long_polled = {
            r
            for r, split in self.folds.items()
            if not POLLED_CORE.get(r)
            and r not in self.row_modes
            and r not in self.forks
            and long_split(r, heads, split, sms)
        }
        self.polled = {}
        for r in self.folds:
            polled = POLLED_CORE.get(r) or r in self.long_polled
            if polled and r not in self.row_modes:
                reqs = r // steps
                qbox = self.box.mailbox(r in self.valid)
                tables = [verify_table(qbox[p], reqs, self.device) for p in (0, 1)]
                tiles = -(-steps * size * self.heads // 128)
                work = self.work_list(r, size * self.heads, tiles, self.folds[r], sms)
                if work is not None and self.valid.get(r) == 2:
                    tables = [work.table(t) for t in tables]
                    self.wlists[r] = work
                self.polled[r] = tables
        assert set(self.valid) <= set(self.polled) | set(self.forks)
        # A LONG_SPLIT rule partitions each request's KV by its length (long_per) and
        # DcpValue folds by that partition; the gathered fold core (split_partials_q)
        # splits uniformly, so a long-ruled row on it would merge the wrong partials at
        # short contexts. Such a table entry is a startup error.
        for r, split in self.folds.items():
            gathered = r not in self.polled and r not in self.row_modes
            if gathered and r not in self.forks and long_split(r, heads, split, sms):
                raise ValueError(
                    f"drafter rows {r} (q {steps}): LONG_SPLIT {LONG_SPLIT[r, heads]} "
                    "fires on the gathered fold core, which has no long partition: "
                    "add the rows to POLLED_CORE or drop the rule"
                )
        ungathered = self.fork_rows | set(self.polled)
        # Rows the fused query kernel (MlaQuery, RoPE) serves (FAST_QUERY: the lean
        # tile up to 8 rows, the tiled kernel above).
        self.queries = {}
        rotary = first.rotary_emb
        if rotary is not None:
            weights = (first.q_a_layernorm.weight, first.kv_a_layernorm.weight)
            weights += (first.q_b_proj.weight, first.W_UK_T)
            weights += (first._q_scale_inv, first._k_scale_inv)
            specs = tuple(layout(w) for w in weights)
            positions = rotary.cos_sin_cache.shape[0]
            for r in self.rows:
                if r in FAST_QUERY:
                    self.queries[r] = compile_query(
                        r, self.heads, size, self.cap, specs, index, sms, positions,
                        gate=bool(GATED_QUERY.get(r)), gather=r not in ungathered,
                        tokens=QUERY_TOKENS.get(r, 0), valid=r in self.valid,
                    )  # fmt: skip
            for r in self.wide:  # (no gather: the polled core reads its q rows)
                self.queries[r] = compile_query(
                    r, self.heads, size, self.wide_box.cap, specs, index, sms,
                    positions, gather=False, valid=True,
                )  # fmt: skip
        box = self.box
        self.peers = to_cute(box.peers, 8), to_cute(box.lse_peers, 8)
        self.boxes = [(to_cute(box.partial[p]), to_cute(box.lse[p], 4)) for p in (0, 1)]
        self.mailboxes = [to_cute(box.q[p]) for p in (0, 1)]
        self.valid_mailboxes = (
            [to_cute(box.qv[p]) for p in (0, 1)] if self.valid else None
        )
        if self.wide:
            wide_box = self.wide_box
            self.wide_peers = to_cute(wide_box.peers, 8), to_cute(wide_box.lse_peers, 8)
            self.wide_boxes = [
                (to_cute(wide_box.partial[p]), to_cute(wide_box.lse[p], 4))
                for p in (0, 1)
            ]
            self.wide_mailboxes = [to_cute(wide_box.qv[p]) for p in (0, 1)]
        for index, attention in enumerate(attentions):
            attention.draft_dcp = DraftLayerDcp(self, attention, index % 2, index == 0)

    def core_of(self, rows):
        """NQ-NOFALLBACK: the attention core a served drafter row runs, for the
        coverage log."""
        if rows in self.wide_cores:
            return f"wide polled core split {self.wide_cores[rows][0]}"
        if rows in self.row_modes:
            return "rows mode" + (" fork" if rows in self.forks else "")
        if rows in self.folds:
            kind = "polled" if rows in self.polled else "gathered"
            return f"fold split {self.folds[rows]} {kind}"
        # (``merge``: the stock tokenspeed forward_mqa inside the fast path)
        why = self.unfolded.get(rows, "not prepared as a fold, wide or rows-mode row")
        return f"stock tokenspeed forward_mqa + DcpValue ({why}; NQ-TUNE)"

    def coverage(self, rows):
        """Served rows grouped by their attention core (the coverage log line)."""
        kinds: dict[str, list[int]] = {}
        for r in rows:
            kinds.setdefault(self.core_of(r), []).append(r)
        return "; ".join(f"{kind}: {r}" for kind, r in kinds.items())

    def served(self, rows):
        """Whether a captured drafter graph of ``rows`` must run this fast path."""
        return rows in self.rows or rows in self.wide

    def work_list(self, rows, heads, tiles, split, sms):
        """The shared per-step work list (mla.wlist) of DRAFT_WLIST ``rows`` launched
        at their LONG_SPLIT split, or None."""
        long = long_split(rows, heads, split, sms)
        if rows not in DRAFT_WLIST or long is None:
            return None
        # (None past mla.wlist.fits: the long split without the list)
        return shared_list(
            rows // self.steps, tiles, split, long, sms // 2, self.device
        )

    def mailbox(self, rows, parity):
        """The q mailbox of ``rows`` (DcpBox.qv for VALID_CORE and FORK_VALID rows; the
        widened box's qv for wide rows) at ``parity``, and its multicast address."""
        if rows in self.wide_cores:
            return self.wide_mailboxes[parity], self.wide_box.mc(parity, True)
        valid = rows in self.valid
        boxes = self.valid_mailboxes if valid else self.mailboxes
        return boxes[parity], self.box.mc(parity, valid)

    def cp(self):
        """The local KV shard's (world, rank, interleave), as tokenspeed takes it."""
        return self.size, self.rank, self.box.interleave

    def bind(self, cache):
        """Compile the folded cores for a bound cache [blocks, page, 576]."""
        if cache.dim() != 3:
            return
        heads, page = self.size * self.heads, cache.shape[1]
        for rows, split in self.folds.items():
            if (rows, page) in self.fold_cores:
                continue
            reqs, index = rows // self.steps, self.device.index
            args = heads, page, split, reqs, self.steps, self.cp(), index
            args += (self.causal,)
            if rows in self.polled:
                wl = rows in self.wlists
                valid = self.valid.get(rows, 0)
                compile_polled_q(*args, valid=valid, persistent=wl, wlist=wl)
            else:
                compile_split_partials_q(*args)
            self.fold_cores.add((rows, page))
        for rows, (split, persistent, _, _, push) in self.wide_cores.items():
            if ("wide", rows, page) not in self.fold_cores:
                reqs, index = rows // self.steps, self.device.index
                args = heads, page, split, reqs, self.steps, self.cp(), index
                args += (self.causal,)
                wl = rows in self.wlists
                compile_polled_q(
                    *args, valid=2, persistent=persistent, push=push, wlist=wl
                )
                self.fold_cores.add(("wide", rows, page))
        for rows, (padded, split, _, _) in self.row_modes.items():
            if ("rows", rows, page) not in self.fold_cores:
                if rows in self.forks:
                    options = self.forks[rows][1:]
                    compile_fork(
                        heads, page, split, padded, self.device.index, *options
                    )
                else:
                    compile_split_partials(heads, page, split, self.device.index)
                self.fold_cores.add(("rows", rows, page))


class DraftLayerDcp:
    """One drafter MLA layer's fast DCP attention (``attend``, or ``attend_projected``
    from the in-projection rows)."""

    def __init__(self, shared, attention, parity, first=False):
        self.shared, self.parity = shared, parity
        # The first layer writes the rows mode's per-row metadata of each forward.
        self.first = first
        self.uv = to_cute(attention.W_UV)
        self.query_weights = None
        if shared.queries:
            weights = (attention.q_a_layernorm.weight, attention.kv_a_layernorm.weight)
            weights += (attention.q_b_proj.weight, attention.W_UK_T)
            weights += (attention._q_scale_inv, attention._k_scale_inv)
            self.query_weights = tuple(
                to_cute(w, 4 if w.dtype == torch.float32 else 16) for w in weights
            )
            self.eps = Float32(attention.rms_norm_eps)
            self.cos_sin = to_cute(attention.rotary_emb.cos_sin_cache, 4)
            rows = max((shared.cap, *shared.wide))
            self.scratch = torch.empty(
                (rows, shared.heads, 576), dtype=torch.uint8, device=shared.device
            )

    def attend_projected(self, attention, positions, qkv):
        """The attention output [R, H*128] (W_UV applied) of the in-projection rows
        ``qkv`` [R, 2112] (q_a | kv_c | k_pe) at ``positions``: MlaQuery (RoPE, KV
        insert, q publish + gather), the stock tokenspeed core, DcpValue. None when
        not covered (nothing launched)."""
        from vllm.forward_context import (
            get_forward_context,
            is_forward_context_available,
        )

        shared, p = self.shared, self.parity
        rows = qkv.shape[0]
        capturing = torch.cuda.is_current_stream_capturing()
        warm = self.first and rows in shared.row_modes
        if not capturing and warm and is_forward_context_available():
            # An eager (warm-up) forward sizes the rows-mode buffers, so that the
            # captured one has no memset.
            metadata = get_forward_context().attn_metadata
            if isinstance(metadata, dict):
                decode = getattr(metadata.get(attention.layer_name), "decode", None)
                if decode is not None:
                    self.rows_tables(rows, decode.block_table.shape[1])
        # NQ-NOFALLBACK: a captured graph of a row the fused query serves either runs
        # it or raises (``decline``); eager calls take the stock path (no graph).
        strict = shared.strict and rows in shared.queries and full_capture()

        def decline(cause):
            if strict:
                refuse("DSpark drafter attention (fused query)", cause, Shape(rows))
            return self.join_context()

        if self.query_weights is None or rows not in shared.queries or not capturing:
            return self.join_context()
        if qkv.shape[1] != 2112 or not qkv.is_contiguous():
            return decline(f"in-projection rows {tuple(qkv.shape)} {qkv.stride()}")
        if positions.dtype != torch.int64:
            return decline(f"positions {positions.dtype}")
        context = get_forward_context()
        metadata = context.attn_metadata
        slots = context.slot_mapping
        if not isinstance(metadata, dict) or not isinstance(slots, dict):
            return decline("no per-layer metadata / slot mapping")
        metadata = metadata.get(attention.layer_name)
        slots = slots.get(attention.layer_name)
        decode = getattr(metadata, "decode", None)
        checks = (
            (decode is None, "no decode metadata"),
            (
                decode is not None
                and bool(getattr(metadata, "causal", True)) != shared.causal,
                f"metadata mask is not the layers' (causal={shared.causal})",
            ),
            (
                decode is not None and metadata.num_decode_tokens != rows,
                "decode tokens != rows",
            ),
            (decode is not None and metadata.num_prefills, "prefill rows"),
            (
                decode is not None and decode.seq_lens.shape[0] * shared.steps != rows,
                "requests x draft length != rows",
            ),
            (
                getattr(decode, "dcp_tot_seq_lens", None) is None,
                "no DCP total lengths",
            ),
            (slots is None or slots.dtype != torch.int64, "no int64 slot mapping"),
        )
        reason = next((why for failed, why in checks if failed), None)
        if reason is not None:
            return decline(reason)
        s = stream(shared.device)
        if rows in shared.forks and not self.rows_ready(attention, rows):
            # The fork rows' MlaQuery does not gather: no other core.
            return decline("fork core not compiled for the bound cache page")
        if rows in shared.polled and not self.fold_ready(attention, rows):
            # nor does the polled core rows' MlaQuery
            return decline("polled core not compiled for the bound cache page")
        if rows in shared.wide_cores and not self.wide_ready(attention, rows):
            # nor does the wide rows' MlaQuery
            return decline("wide core not compiled for the bound cache page")
        cache = attention.kv_cache
        qnorm, kvnorm, qb, uk, qinv, kinv = self.query_weights
        shared.queries[rows](
            to_cute(qkv),
            qnorm,
            kvnorm,
            qb,
            uk,
            dynamic_view(cache.view(torch.uint8)),
            to_cute(slots[:rows], 8),
            qinv,
            kinv,
            self.eps,
            to_cute(self.scratch[:rows]),
            *shared.mailbox(rows, p),
            Int32(shared.rank),
            to_cute(positions[:rows], 8),
            self.cos_sin,
            s,
        )
        padded = shared.row_modes.get(rows, (rows,))[0]
        out = qkv.new_empty((padded, shared.heads * 128))
        self.merge(attention, metadata, rows, out, rows_mode=True)
        return out[:rows]

    def rows_tables(self, rows, width):
        """Rows-mode metadata buffers (block-table rows, local and global lengths) of
        ``rows`` at block-table ``width``, kept across forwards: the table is zeroed
        once (VerifyRows writes only the entries a row can reach; the rest stay
        valid pages)."""
        shared = self.shared
        key = rows, width
        if key not in shared.row_tables:
            padded = shared.row_modes[rows][0]
            i32 = dict(dtype=torch.int32, device=shared.device)
            shared.row_tables[key] = (
                torch.zeros((padded, width), **i32),
                torch.empty((padded,), **i32),
                torch.empty((padded,), **i32),
            )
        return shared.row_tables[key]

    def join_context(self):
        """The first layer: before any attention core, the current stream waits for
        the context step's KV (``DraftDcp.context_ready``). Returns None."""
        shared = self.shared
        if self.first and shared.context_ready is not None:
            torch.cuda.current_stream(shared.device).wait_event(shared.context_ready)
            shared.context_ready = None

    def bind(self, cache):
        """The layer's cache is bound (bind_kv_cache): compile the folded cores."""
        self.shared.bind(cache)

    def fold_ready(self, attention, rows):
        """A folded core covers ``rows`` with the bound cache (compiled)."""
        cache = attention._attn_read_kv_cache()
        if rows not in self.shared.folds or cache.dim() != 3:
            return False
        return (rows, cache.shape[1]) in self.shared.fold_cores

    def fold_core(self, attention, metadata, rows):
        """The reducer-free tokenspeed core over the q mailbox (gathered, or polled by
        the core for POLLED_CORE rows): FP32 split partials + LSEs in the shared
        workspace; False when it does not cover the call (no fold for ``rows``, or no
        core compiled for the bound cache)."""
        shared, p = self.shared, self.parity
        if not self.fold_ready(attention, rows):
            return False
        split = shared.folds[rows]
        cache = attention._attn_read_kv_cache()
        reqs, steps = rows // shared.steps, shared.steps
        heads = shared.size * shared.heads
        box, decode = shared.box, metadata.decode
        valid = shared.valid.get(rows, 0)  # (nonzero: the q mailbox is DcpBox.qv)
        query = box.mailbox(bool(valid))[p, :rows].view(torch.float8_e4m3fn)
        query = query.view(reqs, steps, heads, 576)
        out = box.rows[:rows].view(reqs, steps, heads, 512)
        lse = box.rows_lse[:rows].view(reqs, steps, heads)
        tables = decode.block_table[:reqs], decode.seq_lens[:reqs]
        tables += (decode.dcp_tot_seq_lens[:reqs],)
        if rows in shared.polled:
            work = shared.wlists.get(rows)
            if work is not None and self.first:  # this step's list, every layer
                work.build(tables[1])
            polled_partials_q(
                query, cache, *tables, shared.scales, out, lse, split, shared.cp(),
                shared.polled[rows][p], causal=shared.causal, valid=valid,
                persistent=work is not None, wlist=work is not None,
            )  # fmt: skip
        else:
            split_partials_q(
                query, cache, *tables, shared.scales, out, lse, split, shared.cp(),
                shared.causal,
            )  # fmt: skip
        return True

    def wide_ready(self, attention, rows):
        """The wide path covers ``rows`` with the bound cache (its core compiled)."""
        cache = attention._attn_read_kv_cache()
        if rows not in self.shared.wide_cores or cache.dim() != 3:
            return False
        return ("wide", rows, cache.shape[1]) in self.shared.fold_cores

    def wide_merge(self, attention, metadata, rows, out):
        """Wide rows: the polled packed core over the widened box's q mailbox (split
        1: BF16 rows + LSEs; more: FP32 split partials + LSEs in the workspace), then
        DcpValue into ``out`` [rows, H*128]."""
        shared, p, s = self.shared, self.parity, stream(self.shared.device)
        split, persistent, value, tables, push = shared.wide_cores[rows]
        cache = attention._attn_read_kv_cache()
        reqs, steps = rows // shared.steps, shared.steps
        heads = shared.size * shared.heads
        box, decode = shared.wide_box, metadata.decode
        query = box.mailbox(True)[p, :rows].view(torch.float8_e4m3fn)
        query = query.view(reqs, steps, heads, 576)
        out_rows = box.rows[:rows].view(reqs, steps, heads, 512)
        lse = box.rows_lse[:rows].view(reqs, steps, heads)
        meta = decode.block_table[:reqs], decode.seq_lens[:reqs]
        meta += (decode.dcp_tot_seq_lens[:reqs],)
        work = shared.wlists.get(rows)
        if work is not None and self.first:  # this step's list, every layer
            work.build(meta[1])
        polled_partials_q(
            query, cache, *meta, shared.scales, out_rows, lse, split, shared.cp(),
            tables[p], causal=shared.causal, valid=2, persistent=persistent, push=push,
            wlist=work is not None,
        )  # fmt: skip
        if split > 1:  # DcpValue folds the split partials
            packed = shared.packed[rows]
            _, part, stats = split_workspace_ts(shared.device, packed, reqs, split)
            rows_in = to_cute(part), to_cute(stats, 4)
        else:
            rows_in = to_cute(box.rows[:rows]), to_cute(box.rows_lse[:rows], 4)
        value(
            *rows_in,
            to_cute(decode.seq_lens, 4),
            *shared.wide_boxes[p],
            *shared.wide_peers,
            shared.wide_mailboxes[p],  # (re-armed by DcpValue)
            self.uv,
            to_cute(shared.gate[:rows]),
            to_cute(decode.dcp_tot_seq_lens, 4),
            to_cute(out),
            Int32(p),
            Int32(shared.rank),
            s,
        )

    def rows_ready(self, attention, rows):
        """Rows mode covers ``rows`` with the bound cache (its core compiled)."""
        cache = attention._attn_read_kv_cache()
        if rows not in self.shared.row_modes or cache.dim() != 3:
            return False
        return ("rows", rows, cache.shape[1]) in self.shared.fold_cores

    def rows_core(self, attention, metadata, rows, out):
        """Rows mode (ROWS_MODE): the q = 1 core over the gathered q rows (padded with
        empty rows) and DcpValue's LSE fold into ``out`` [padded, H*128]; False when
        it does not cover the call (no mode for ``rows``, or no compiled core)."""
        shared, p = self.shared, self.parity
        if not self.rows_ready(attention, rows):
            return False
        cache = attention._attn_read_kv_cache()
        padded, split, value, meta = shared.row_modes[rows]
        if out.shape[0] != padded:
            return False
        decode, s, box = metadata.decode, stream(shared.device), shared.box
        if self.first or rows not in shared.row_meta:
            table = decode.block_table
            rows_meta = self.rows_tables(rows, table.shape[1])
            meta(
                dynamic_view(table, 4),
                dynamic_view(decode.dcp_tot_seq_lens, 4),
                dynamic_view(rows_meta[0], 4),
                *(to_cute(t, 4) for t in rows_meta[1:]),
                Int32(rows // shared.steps),
                s,
            )
            shared.row_meta[rows] = rows_meta
        table, lengths, totals = shared.row_meta[rows]
        self.join_context()  # (after VerifyRows: it reads no context-step output)
        heads = shared.size * shared.heads
        valid = rows in shared.valid  # (the fork's q mailbox is DcpBox.qv)
        query = box.mailbox(valid)[p, :padded].view(torch.float8_e4m3fn)
        query = query.view(padded, 1, heads, 576)
        rows_out = box.rows[:padded].view(padded, 1, heads, 512)
        rows_lse = box.rows_lse[:padded].view(padded, 1, heads)
        if rows in shared.forks:
            tables, *options = shared.forks[rows]
            fork_partials(
                query, cache, table, lengths, shared.scales, rows_out, rows_lse, split,
                tables[p], *options,
            )  # fmt: skip
        else:
            split_partials(
                query, cache, table, lengths, shared.scales, rows_out, rows_lse, split
            )
        _, part, stats = split_workspace_ts(shared.device, heads, padded, split)
        value(
            to_cute(part),
            to_cute(stats, 4),
            to_cute(lengths, 4),
            *shared.boxes[p],
            *shared.peers,
            shared.mailbox(rows, p)[0],
            self.uv,
            to_cute(shared.gate[:padded]),
            to_cute(totals, 4),
            to_cute(out),
            Int32(p),
            Int32(shared.rank),
            s,
        )
        return True

    def merge(self, attention, metadata, rows, out, rows_mode=False):
        """Tokenspeed over the gathered q mailbox, then DcpValue into ``out``
        (``rows_mode``: ``out`` has the rows mode's padded rows)."""
        shared, p = self.shared, self.parity
        box, s = shared.box, stream(shared.device)
        decode = metadata.decode
        if rows_mode and self.rows_core(attention, metadata, rows, out):
            return
        self.join_context()
        out = out[:rows]
        if rows in shared.wide_cores:
            self.wide_merge(attention, metadata, rows, out)
            return
        if self.fold_core(attention, metadata, rows):
            packed = shared.packed[rows]
            split = shared.folds[rows]
            _, part, stats = split_workspace_ts(
                shared.device, packed, rows // shared.steps, split
            )
            shared.fold_values[rows](
                to_cute(part),
                to_cute(stats, 4),
                to_cute(decode.seq_lens, 4),
                *shared.boxes[p],
                *shared.peers,
                shared.mailbox(rows, p)[0],  # (re-armed by DcpValue)
                self.uv,
                to_cute(shared.gate[:rows]),
                to_cute(decode.dcp_tot_seq_lens, 4),
                to_cute(out),
                Int32(p),
                Int32(shared.rank),
                s,
            )
            return
        query = box.q[p, :rows].view(torch.float8_e4m3fn)
        cache = attention._attn_read_kv_cache()
        latent, lse = attention.impl.forward_mqa(query, cache, metadata, attention)
        heads = shared.size * shared.heads
        shared.values[rows](
            to_cute(latent.view(rows, 1, heads, 512)),
            to_cute(lse.view(rows, 1, heads), 4),
            to_cute(decode.seq_lens, 4),
            *shared.boxes[p],
            *shared.peers,
            shared.mailboxes[p],
            self.uv,
            to_cute(shared.gate[:rows]),
            to_cute(decode.dcp_tot_seq_lens, 4),
            to_cute(out),
            Int32(p),
            Int32(shared.rank),
            s,
        )

    def attend(self, attention, mqa_q, metadata, out):
        """The decode rows' attention output (W_UV applied) into ``out`` [R, H*128]
        from the stock fused query rows ``mqa_q`` [R, H, 576] FP8; False when this
        call is not covered (nothing launched)."""
        shared, p = self.shared, self.parity
        rows = mqa_q.shape[0]
        decode = metadata.decode
        self.join_context()
        if not torch.cuda.is_current_stream_capturing():
            return False  # eager: no graph, the stock attention
        checks = (
            (rows not in shared.gathers, "no QGather for the rows"),
            (decode is None, "no decode metadata"),
            (
                bool(getattr(metadata, "causal", True)) != shared.causal,
                f"metadata mask is not the layers' (causal={shared.causal})",
            ),
            (mqa_q.dtype != torch.float8_e4m3fn, f"query {mqa_q.dtype}"),
            (not mqa_q.is_contiguous(), "a strided query"),
            (out.shape != (rows, shared.heads * 128), f"out {tuple(out.shape)}"),
            (not out.is_contiguous(), "a strided output"),
            (
                decode is not None and decode.seq_lens.shape[0] * shared.steps != rows,
                "requests x draft length != rows",
            ),
            (
                getattr(decode, "dcp_tot_seq_lens", None) is None,
                "no DCP total lengths",
            ),
            # (QGather publishes into DcpBox.q)
            (rows in shared.valid, "validity rows without a fused query"),
        )
        reason = next((why for failed, why in checks if failed), None)
        if reason is not None:
            # NQ-NOFALLBACK: a captured drafter graph never captures the stock query
            # gather + forward_mqa + combine.
            if shared.strict and full_capture():
                refuse("DSpark drafter attention", reason, Shape(rows))
            return False
        box, s = shared.box, stream(shared.device)
        shared.gathers[rows](
            to_cute(mqa_q.view(torch.uint8)),
            shared.mailboxes[p],
            box.mc(p),
            Int32(shared.rank),
            s,
        )
        self.merge(attention, metadata, rows, out)
        return True
