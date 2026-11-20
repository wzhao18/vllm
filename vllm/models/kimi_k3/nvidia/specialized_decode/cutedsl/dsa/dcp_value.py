# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DcpValue of the K3.1 DSA layers under decode context parallelism (DCP = TP, rank d
owns head block d; each rank's KV cache holds 1 / DCP of every context): re-arms the
q mailbox, pushes this rank's partial rows (BF16 latent rows, or "proj": BF16 value
rows through W_UV) and LSE of the gathered heads to their owner's Lamport mailbox
(peer stores; a source with nothing to merge sends only its LSE); the owner
LSE-merges its H heads over the DCP sources, then W_UV and the gate (padding from the
global lengths). The mailboxes are the model's DSA DcpBox (mla.dcp) and, for "proj"
rows, a DcpMailbox beside it (r1.DsaBox)."""

import math
from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
import torch.distributed._symmetric_memory as symm_mem
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, Uint32, Uint64
from cutlass.cute.runtime import make_fake_stream

from ...symm import peer_rendezvous
from ..common import tensormap
from ..common.host import fake
from ..common.policy import batch_table
from ..common.primitives import (
    alloc,
    asm,
    at,
    bf16x2,
    bits_float,
    div_rn,
    float_bits,
    launch,
    load_volatile,
    load_words,
    mma_into,
    no_neg_zero,
    pack_pair,
    published,
    sigmoid,
    store_relaxed,
    store_shared_words,
    store_words,
    unpack_bf16,
    warp_sum,
    zeros,
)
from .sparse_mla import (
    SDESC_SW128,
    fence_mbar_init,
    globaltimer,
    ld_shared_u32,
    mbar_arrive,
    mbar_arrive_tx,
    mbar_init,
    mbar_wait,
    st_global_u64,
    st_shared_u32,
    tc_commit,
    tc_fence_after,
    tc_fence_before,
)

# Bench probe (benchmarks/kernels/k31_vr/bench_dattn_dcpv.py): the address of a
# [CTAs, 16] u64 globaltimer buffer the "proj" tcgen05 DcpValue stamps its phases
# into when compiled while it is nonzero (0, the default: no stamping code).
STAMP_ADDR = 0
STAMP_SLOTS = 32
# Bench probe (KNOBS 4096): this rank's view of a [2] u32 symmetric counter per
# rank (BAR_PEERS[r] = rank r's counter, BAR_OWN this rank's, BAR_DONE a local
# [2] u32): every CTA
# adds 1 to each rank's counter after its pushes drained, then the merge waits
# for all ranks' CTAs (the merge over resident data).
BAR_PEERS = (0, 0, 0, 0)
BAR_OWN = 0
BAR_DONE = 0
# Bench probe ablation bits (0, the default: production code): 1 no projection
# (no TMA loads or MMAs; stale value rows pushed), 2 no push and no merge,
# 4 no merge, 8 merge without value polls, 16 merge without LSE polls,
# 32 every LSE read as 0, 128 (+256) a 64 (256) ns sleep between polls.
KNOBS = 0

LSE_SENT = 0xFFFFFFFF  # a NaN pattern the merge never sends
# The LSE of an empty or invalid source: finite, so the merge never forms inf - inf.
EMPTY_LSE = -3.0e38
Q_SENT = 0xFFFFFFFF  # four FP8 0xFF bytes: never written (satfinite, canonical NaN)
PAIR_SENT = 0x80000000  # BF16 pair (+0, -0); sent pairs have no negative zero
# "proj": every source applies W_UV of each gathered head to its own partial row before
# the exchange (the merge is linear in the rows: sum_s w_s O_s W = sum_s w_s (O_s W)),
# so the owners receive BF16 [H, 128] value rows instead of [H, 512] latents: 4x fewer
# NVLink bytes, for a replicated W_UV (all DCP * H heads) per layer.
PROJECTED = "proj"
# Latent rows a "proj" projecting CTA stages per pipelined step (at most; DCP8
# GB300 DcpValue at 320 rows: 16 -> 64.7 us, 32 -> 55.7, 64 -> 54.6, but 64 loses
# 1-2 us at 80-160 rows).
PROJ_ROWS = 32
# "proj" above PROJ_ROWS rows: the projection on tcgen05 (kind::f16, BF16 in, FP32
# accumulation; the legacy mma.sync path runs at a quarter of its rate): units of
# (gathered head, 128 rows) [128 x 512] x [512 x 128] spread over every CTA, K in
# 64-element blocks through a TMA ring (A: the latent rows, B: the head's W_UV).
TC_STAGES = 4
TC_STAGE = 2 * 128 * 128  # A + B: [128 x 64] BF16 each, SW128
# A unit's BF16 value rows [128 x 128], staged for its one bulk push (two buffers:
# a unit's push is in flight while the next one's epilogue writes).
TC_OUT = 128 * 128 * 2
# Its epilogue warps (warp % 4 = TMEM lane quadrant), clear of the TMA and MMA
# issuers (warps 0 and 1), which run ahead into the next unit.
TC_EPI = 4
# kind::f16: FP32 D, BF16 A and B (K-major), N = 128, M = 128
TC_IDESC = (1 << 4) | (1 << 7) | (1 << 10) | ((128 >> 3) << 17) | ((128 >> 4) << 24)
# Overlapped with the persistent DSA sparse core (sparse_mla_ts, grid = SMs, row = CTA
# + i x grid), which publishes every finished row into a per-block counter: row blocks
# of at most 128 rows that never straddle a wave (wave w holds rows [w G, (w + 1) G)),
# so a wave's blocks complete before the next wave's. The sync words per mailbox
# parity: unit claims, merge claims, CTAs done, spare, then one counter per block.
SYNC_WORDS = 32
SYNC_READY = 4
# Item ring depth between the claiming TMA thread and the MMA / epilogue roles.
OV_ITEMS = 4
# kernel_ov: rows per chunk (the core's publish block: HS_ROWS rows of one wave), the
# chunk ring (each chunk's 8 [HS_ROWS x 64] SW128 latent tiles), the staged value rows
# per chunk and their buffers, the accumulators' TMEM column (after W_UV's 256), and
# kind::f16 M128 (value dims) x N HS_ROWS (rows) with A (W_UV^T) in TMEM.
HS_ROWS = 32
HS_RING = 6  # 192 KiB of 32 KiB slots: a chunk, or two W_UV tiles of a new stream
HS_MASKS = 8
HS_PART = 4  # chunks per stream part
HS_ACCS = 4  # accumulators (32 TMEM columns each, after W_UV)
HS_ISS = 4  # TMA issuing lanes
HS_CHUNK = 8 * HS_ROWS * 128
HS_OUT = HS_ROWS * 256
HS_OUTS = 4
HS_ACC = 256
HS_IDESC = (1 << 4) | (1 << 7) | (1 << 10) | ((HS_ROWS >> 3) << 17) | ((128 >> 4) << 24)


def wave_blocks(rows, grid):
    """The (first row, rows) publish blocks of ``rows`` rows under a persistent
    ``grid`` (row = CTA + i x grid): block w ceil(grid / HS_ROWS) + k holds rows
    w grid + k HS_ROWS .. of wave w."""
    per = -(-grid // HS_ROWS)
    blocks = {}
    for w in range(-(-rows // grid)):
        for k in range(per):
            t0 = w * grid + k * HS_ROWS
            t1 = min(w * grid + min((k + 1) * HS_ROWS, grid), rows)
            if t1 > t0:
                blocks[w * per + k] = (t0, t1 - t0)
    return blocks


def projects_on_tc(batch):
    """Whether a DSA "proj" DcpValue of ``batch`` rows projects on tcgen05 itself
    (dcp_stage then passes the gathered W_UV, not the bmm's value rows)."""
    return batch > PROJ_ROWS


@asm()
def _tma3d(dst, tmap, bar, c0, c1, c2):
    code = (
        "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx"
        "::bytes [$0], [$1, {$3, $4, $5}], [$2];"
    )
    return code, "r,l,r,r,r,r,~{memory}", (
        Uint32(dst), Int64(tmap), Uint32(bar), Int32(c0), Int32(c1), Int32(c2))


@asm()
def _mma_f16(d, a, b, idesc, acc):
    code = (
        "{ .reg .pred p; setp.ne.b32 p, $4, 0; "
        "tcgen05.mma.cta_group::1.kind::f16 [$0], $1, $2, $3, p; }"
    )
    return code, "r,l,l,r,r,~{memory}", (
        Uint32(d), Uint64(a), Uint64(b), Uint32(idesc), Uint32(acc))


@asm()
def _tmem_alloc256(dst):
    code = (
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [$0], 256; "
        "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"
    )
    return code, "r,~{memory}", (Uint32(dst),)


@asm()
def _tmem_alloc512(dst):
    code = (
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [$0], 512; "
        "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"
    )
    return code, "r,~{memory}", (Uint32(dst),)


@asm()
def _tmem_free512(taddr):
    return ("tcgen05.dealloc.cta_group::1.sync.aligned.b32 $0, 512;", "r,~{memory}",
            (Uint32(taddr),))


@asm()
def _tmem_cp(t, sdesc):
    """128 rows x 32 bytes from shared memory (descriptor) to tensor memory."""
    return ("tcgen05.cp.cta_group::1.128x256b [$0], $1;", "r,l,~{memory}",
            (Uint32(t), Uint64(sdesc)))


@asm()
def _mma_f16_ts(d, a_tmem, b, idesc, acc):
    """kind::f16 with A from tensor memory."""
    code = (
        "{ .reg .pred p; setp.ne.b32 p, $4, 0; "
        "tcgen05.mma.cta_group::1.kind::f16 [$0], [$1], $2, $3, p; }"
    )
    return code, "r,r,l,r,r,~{memory}", (
        Uint32(d), Uint32(a_tmem), Uint64(b), Uint32(idesc), Uint32(acc))


@asm()
def _tmem_ld16x256(addr):
    """tcgen05.ld 16x256b.x4: thread t gets, for column groups g = 0..3 (8 columns
    each), (lane t / 4, cols 8 g + 2 (t % 4) + {0, 1}) and (lane 8 + t / 4, same)."""
    regs = ",".join(f"${i}" for i in range(16))
    code = (f"tcgen05.ld.sync.aligned.16x256b.x4.b32 {{{regs}}}, [$16]; "
            "tcgen05.wait::ld.sync.aligned;")
    return code, ",".join(["=f"] * 16 + ["r", "~{memory}"]), (Uint32(addr),), (
        Float32,) * 16


@asm()
def stmatrix_t4(addr, r0, r1, r2, r3):
    """stmatrix m8n8.x4.trans b16: matrix k's stored row j at the address of thread
    8 k + j holds column j of the fragment matrix k."""
    return ("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [$0], {$1, $2, $3, $4};",
            "r,r,r,r,r,~{memory}", (Uint32(addr), Uint32(r0), Uint32(r1), Uint32(r2),
                                     Uint32(r3)))


@asm()
def _bulk_read_wait3():
    return "cp.async.bulk.wait_group.read 3;", "~{memory}", ()


@asm()
def _tmem_free256(taddr):
    return ("tcgen05.dealloc.cta_group::1.sync.aligned.b32 $0, 256;", "r,~{memory}",
            (Uint32(taddr),))


@asm()
def _bulk_push(dst, src, nbytes):
    """One bulk copy of ``nbytes`` from shared memory to (peer) global memory."""
    return ("cp.async.bulk.global.shared::cta.bulk_group [$0], [$1], $2; "
            "cp.async.bulk.commit_group;", "l,r,r,~{memory}",
            (Int64(dst), Uint32(src), Uint32(nbytes)))


@asm()
def _bulk_read_wait1():
    return "cp.async.bulk.wait_group.read 1;", "~{memory}", ()


@asm()
def _bulk_wait0():
    return "cp.async.bulk.wait_group 0;", "~{memory}", ()


@asm()
def _nanosleep64():
    return "nanosleep.u32 64;", "~{memory}", ()


@asm()
def _nanosleep256():
    return "nanosleep.u32 256;", "~{memory}", ()


@asm()
def _red_max_u64(ptr, value):
    return "red.relaxed.gpu.global.max.u64 [$0], $1;", "l,l,~{memory}", (ptr, value)


@asm()
def _bar_arrive_sys(ptr):
    return "red.release.sys.global.add.u32 [$0], 1;", "l,~{memory}", (ptr,)


@asm()
def _ld_acquire_sys(ptr):
    return "ld.acquire.sys.global.u32 $0, [$1];", "=r,l,~{memory}", (ptr,), (Uint32,)


@asm()
def _atom_add_gpu(ptr):
    return "atom.relaxed.gpu.global.add.u32 $0, [$1], 1;", "=r,l,~{memory}", (ptr,), (
        Uint32,)


@asm()
def _store_u32(ptr, value):
    return "st.relaxed.gpu.global.u32 [$0], $1;", "l,r,~{memory}", (ptr, value)


@asm()
def _ld_acquire_gpu(ptr):
    return "ld.acquire.gpu.global.u32 $0, [$1];", "=r,l,~{memory}", (ptr,), (Uint32,)


@asm()
def _ld_relaxed_gpu(ptr):
    return "ld.relaxed.gpu.global.u32 $0, [$1];", "=r,l,~{memory}", (ptr,), (Uint32,)


@asm()
def _fence_proxy_async_global():
    return "fence.proxy.async.global;", "~{memory}", ()


@asm()
def _fence_gpu():
    return "fence.acq_rel.gpu;", "~{memory}", ()


@asm()
def _fence_sys():
    return "fence.acq_rel.sys;", "~{memory}", ()


@asm()
def _fence_async_shared():
    return "fence.proxy.async.shared::cta;", "~{memory}", ()


@asm()
def _epi_sync():
    """Barrier 1 over the 128 epilogue threads."""
    return "bar.sync 1, 128;", "~{memory}", ()


@asm()
def _st_shared4(addr, w0, w1, w2, w3):
    return "st.shared.v4.b32 [$0], {$1, $2, $3, $4};", "r,r,r,r,r,~{memory}", (
        Uint32(addr), Uint32(w0), Uint32(w1), Uint32(w2), Uint32(w3))


@asm()
def _tmem_ld32(addr):
    regs = ",".join(f"${i}" for i in range(32))
    code = (f"tcgen05.ld.sync.aligned.32x32b.x32.b32 {{{regs}}}, [$32]; "
            "tcgen05.wait::ld.sync.aligned;")
    return code, ",".join(["=f"] * 32 + ["r", "~{memory}"]), (Uint32(addr),), (
        Float32,) * 32
# Both formats are exact up to BF16 rounding (no lossy 8-bit partials).
FORMATS = ("bf16", PROJECTED)
# The default exchange per layer kind. K3.1 DSA: the one automatic policy
# (``DSA_PARTIALS``, also r1.AUTO_PARTIALS on FlashInfer sparse): "bf16" at 1-16 rows
# (the DSA sparse core pushes BF16 partials there; "proj" ties BF16 at few rows) and
# "proj" from 17 rows on, open-ended (r1.auto_partials: "proj" at every row of a q
# the core does not serve). "proj" numerics are what production already served (the
# C48-C256 recipes ran "proj", C1 / C8 "1-16:bf16,17-512:proj"). DCP8 GB300 attention layer (q5, slowest rank)
# measured "proj" from 160 rows (-6 to -7 us at 160, -17 to -19 at 320, a tie with
# BF16 at 5-80 rows, where its serial per-head projection costs what the 4x fewer
# NVLink bytes save). MLA ("proj" from 40 rows: -1 at 40, -2 to -3 at 60-80, -9 at
# 160, -17 to -23 at 320; +1 to +3.5 at 5-20 rows) starts at 65 rows for now: the
# step serves at most 64 rows today, so K3's served numerics stay those of the BF16
# exchange until a gsm8k run gates "proj" (then lower it to 40). The DcpBox allocates
# the "proj" mailbox and the layers gather W_UV only when a batch up to its cap uses
# "proj".
DSA_PARTIALS = "1-16:bf16,17-:proj"
PARTIALS_DEFAULT = {"mla": "1-64:bf16,65-320:proj", "dsa": DSA_PARTIALS}


def resolve_partials(spec, kind):
    """The DcpBox policy of a ``spec`` ("default": PARTIALS_DEFAULT of the layer
    ``kind``, "mla" or "dsa"; anything else as given)."""
    return PARTIALS_DEFAULT[kind] if spec in (None, "default") else spec


def partials_for(spec, batch):
    """The partials format of a ``batch`` of rows under the ``spec`` policy: a format
    ("bf16", "proj") or a batch table of formats."""
    if spec in FORMATS:
        return spec
    table = batch_table(spec)
    fmt = table.get(batch, "bf16")
    assert fmt in FORMATS, (spec, fmt)
    return fmt


class DcpMailbox:
    """One partials format's peer-store mailboxes: partial rows [2, DCP, CAP, H, 512]
    BF16 and LSE [2, DCP, CAP, H] FP32 ("proj": head-major value rows [2, DCP, H,
    CAP, 128] BF16, the 16 B chunks of row t XOR-permuted by t % 8, and LSE [2, DCP,
    H, CAP]), sentinel-filled, and their peers' addresses."""

    def __init__(self, group, device, size, cap, heads, fmt):
        assert fmt in FORMATS, fmt
        shape, empty = (2, size, cap, heads), symm_mem.empty
        if fmt == PROJECTED:  # head-major: a unit's rows are contiguous
            shape = (2, size, heads, cap)
        self.format = fmt
        width = 128 if fmt == PROJECTED else 512
        self.partial = empty((*shape, width), dtype=torch.bfloat16, device=device)
        self.lse = empty(shape, dtype=torch.float32, device=device)
        # Peer stores only: no multicast object.
        handles = [peer_rendezvous(t, group) for t in (self.partial, self.lse)]
        i64 = dict(dtype=torch.int64, device=device)
        self.peers = torch.tensor(handles[0].buffer_ptrs, **i64)
        self.lse_peers = torch.tensor(handles[1].buffer_ptrs, **i64)
        self.partial.view(torch.int32).fill_(PAIR_SENT - (1 << 32))
        self.lse.view(torch.int32).fill_(-1)
        # Per parity: kernel_ov's claim / done counters and sparse_mla_ts's
        # published rows per wave block (zero between uses).
        self.sync = torch.zeros(2, SYNC_WORDS, dtype=torch.int32, device=device)


def gather_heads(w_uv, size, group):
    """Every rank's W_UV [H, 512, 128] (latent-contiguous) -> [DCP * H, 512, 128] in
    the same layout, gathered head block d = rank d's heads ("proj" sources project
    the rows of every gathered head). Collective: call on every rank of ``group``
    (``size`` ranks)."""
    heads = w_uv.transpose(1, 2).contiguous()  # [H, 128, 512]
    gathered = heads.new_empty((size * heads.shape[0], *heads.shape[1:]))
    torch.distributed.all_gather_into_tensor(gathered, heads, group=group)
    return gathered.transpose(1, 2)


class DcpValue:
    """The DCP tail of one MLA layer, one 512-thread CTA per (head, 16 requests).
    Every rank re-arms this parity's q mailbox (tokenspeed has read it) and pushes its
    local tokenspeed rows (+ LSE) to their head owner's Lamport mailbox; a row of an
    empty local shard (or an invalid LSE) sends only the LSE EMPTY_LSE, and the owner
    neither polls nor re-arms its partial. Warp r then LSE-weights request t0 + r of
    this head over the DCP sources (exp2(lse - max) / sum in FP32), the CTA merges the
    polled rows in FP32 (one BF16 rounding, re-arming what it read) into shared
    memory, and W_UV (BF16 MMA, FP32 accumulation) and the sigmoid gate follow as in
    MlaValue: warp w owns outputs 32 (w // 4) + 8 (w % 4) .. + 7. Rows are tokens,
    ``q`` per request. Above 64 rows a merging CTA owns every ``grid[1]``-th 16-row
    tile: at most one CTA per SM, so every CTA's pushes land while the merging CTAs
    poll.

    ``per_row``: the emptiness test reads ``lens`` per row (a sparse decode's valid
    counts of the local shard), else per request (the local lengths).

    ``partials`` "proj" (``w_uv``: the gathered [DCP * H, 512, 128] W_UV, see
    DcpBox.gather_heads): CTA (h, d < DCP) projects the local rows of gathered head
    d * H + h through its W_UV (steps of up to PROJ_ROWS rows staged in shared
    memory, the BF16 MMAs of the merge; the next step's latents and LSEs load while
    one projects) and pushes BF16 [rows, 128] value rows; CTA (h, DCP + m) merges
    request tiles m, m + M, .. of head h over the DCP sources (FP32, one BF16
    rounding: the merge is linear, sum_s w_s (O_s W) = (sum_s w_s O_s) W, so only
    the rounding points differ from the latent merge) and applies the gate.
    No CTA waits on a CTA that may not be dispatched: projecting CTAs wait on
    nothing but the sparse decode before them (griddepcontrol_wait); merging CTAs
    poll only values and LSEs the projecting CTAs of every rank push. Those have
    the lowest linear block indices, so the scheduler dispatches them before any
    merging CTA, and independently the grid is at most one CTA per SM, while
    everything else that can hold SMs during DcpValue (the PDL-overlapped sparse
    decode, the prefetch stream) finishes without waiting on it, and the next
    kernel only launches once every CTA here passed its PDL trigger, which the
    projecting CTAs reach after their projections. M fills the SMs left by the
    DCP * H projecting CTAs."""

    def __init__(
        self,
        heads,
        size,
        batch,
        sms=148,
        q=1,
        q_chunks=36,
        q_heads=0,
        lse_scale=1.0,
        per_row=False,
        partials="bf16",
        preprojected=False,
        rearm=True,
        ts_grid=0,
    ):
        """``ts_grid`` > 0 ("proj" on tcgen05 only): the sparse decode before this is
        sparse_mla_ts on that persistent grid, publishing its rows into the
        mailbox's sync counters (kernel_ov). ``rearm`` False: no reader polls this parity's q mailbox (the query of
        every gathered head is computed on each rank, dsa.local_query), so it is
        never re-armed. ``q_chunks``: 16 B chunks per q mailbox head row (36 FP8, 72 BF16);
        ``q_heads``: q mailbox rows per token (default DCP * H); ``lse_scale``: to
        base 2 (LOG2E for a natural-log LSE). ``preprojected`` ("proj" only): the
        W_UV argument holds the value rows [DCP * H, B, 128] BF16 a batched GEMM
        already projected (the stock W_UV bmm over the gathered heads), so the
        projecting CTAs only push them."""
        self.heads, self.size, self.batch = heads, size, batch
        self.stamp_addr = STAMP_ADDR
        self.bar_peers, self.bar_own, self.bar_done = BAR_PEERS, BAR_OWN, BAR_DONE
        self.knobs = KNOBS
        assert partials in FORMATS, partials
        self.q, self.per_row, self.partials = q, per_row, partials
        self.projected = partials == PROJECTED
        self.tc = self.ov = False
        self.preprojected = preprojected and self.projected
        self.rearm_q = rearm
        # Rows with nothing to merge send only their LSE (neither polled nor
        # re-armed): "proj", and BF16 for a sparse decode's per-row counts. The MLA
        # BF16 layout (rarely empty: padded requests, short contexts) sends zeros
        # and polls every row, which keeps its merge loop branch-free.
        self.skip_empty = per_row or self.projected
        self.q_chunks, self.q_heads = q_chunks, q_heads or size * heads
        self.lse_scale = lse_scale
        # Push-only CTAs (z > 0) spread the NVLink stores over more SMs: up to one
        # wave, at least 4 16 B chunks per thread.
        self.tiles = tiles = (batch + 15) // 16
        if self.projected:
            # Rows each projecting step stages and pushes (16-row MMA blocks).
            self.prows = min(PROJ_ROWS, tiles * 16)
            # Above one step the projection runs on tcgen05 (TC_IDESC) in units of
            # (gathered head, 128 rows) over every CTA (grid <= SMs), which then
            # merge; no CTA waits before its units (see the class docstring).
            self.tc = not self.preprojected and per_row and projects_on_tc(batch)
            self.ov = self.tc and ts_grid > 0
            if self.ov:
                # Overlapped with the persistent sparse core (kernel_ov): units of
                # (gathered head, wave block) claimed in block order by one CTA per
                # SM as the core vacates them, then 32-row merge items.
                assert not rearm, "kernel_ov: the q mailbox must not need a re-arm"
                assert 1 <= ts_grid <= sms, (ts_grid, sms)
                self.ts_grid, self.per = ts_grid, -(-ts_grid // HS_ROWS)
                self.nblocks = max(wave_blocks(batch, ts_grid)) + 1
                assert SYNC_READY + self.nblocks <= SYNC_WORDS, self.nblocks
                # streams (wave, part, gathered head): a wave's chunks split into
                # parts of at most HS_PART chunks (all SMs busy after the core)
                self.parts = []
                for w in range(-(-batch // ts_grid)):
                    nk = -(-min(ts_grid, batch - w * ts_grid) // HS_ROWS)
                    self.parts.append((nk, -(-nk // HS_PART)))
                self.units = size * heads * sum(p for _, p in self.parts)
                self.merge_items = heads * -(-batch // 32)
                self.grid = (sms, 1, 1)
                self.merge_tiles = 1
                return
            if self.tc:
                self.mtiles = -(-batch // 128)
                self.units = size * heads * self.mtiles
                self.grid = (heads, max(1, sms // heads), 1)
                self.merge_tiles = -(-tiles // self.grid[1])
                return
            merge = min(tiles, max(1, (sms - size * heads) // heads))
            self.grid = (heads, size + merge, 1)
            # One CTA per SM at most (the progress argument above).
            assert heads * (size + merge) <= sms, (self.grid, sms)
            return
        if batch > 64:
            tiles = min(tiles, max(1, sms // heads))
        merge = heads * tiles
        chunks = batch * size * heads * 64
        pushers = min(max(1, sms // merge), -(-chunks // (merge * 512 * 4)))
        self.grid = (heads, tiles, pushers)

    @cute.jit
    def __call__(self, *args):
        if cutlass.const_expr(self.projected and self.tc):
            rows, w_uv = args[0], args[8]
            bf = cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16
            sw = cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B
            l2 = cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B
            # A: latent rows of one gathered head [128 rows x 64]; B: its W_UV
            # [128 outputs x 64] (latent-contiguous, gather_heads' layout)
            # (bench probe KNOBS 8192: rows 1 KiB apart, the head-major access pattern)
            row_stride = 1024 if self.knobs & 8192 else self.q_heads * 1024
            am = tensormap.encode_map(
                rows.iterator.toint(), bf, (512, self.q_heads, self.batch),
                (1024, row_stride), (64, 1, HS_ROWS if self.ov else 128), sw, l2)
            bm = tensormap.encode_map(
                w_uv.iterator.toint(), bf, (512, 128, self.size * self.heads),
                (1024, 128 * 1024), (64, 128, 1), sw, l2)
            # The descriptors are grid-constant parameters (not tuple members, which
            # the kernel would copy to local memory, an invalid TMA source).
            kernel = self.kernel_tc
            if cutlass.const_expr(self.ov):
                kernel = self.kernel_ov
            kernel(args[:-1], am, bm).launch(
                grid=self.grid, block=(512, 1, 1), stream=args[-1], use_pdl=True)
        else:
            kernel = self.kernel_projected if self.projected else self.kernel
            launch(kernel, args, self.grid, 512)

    @cute.jit
    def rearm(self, mailbox, me, threads):
        """Re-arm this parity's q mailbox (the sparse decode has read it): thread
        ``me`` of ``threads`` stores the sentinel into every threads-th 16 B chunk of
        the published rows."""
        H, D, B = self.heads, self.size, self.batch
        C, P = self.q_chunks, self.q_heads
        if cutlass.const_expr(self.rearm_q):
            for c in range(me, B * D * H * C, threads):  # 16 B q mailbox chunks
                row = c // (D * H * C) * P * C + c % (D * H * C)
                store_words(mailbox.iterator + row * 16, (Uint32(Q_SENT),) * 4)

    @cute.kernel
    def kernel(self, args):
        rows, rows_lse, lens, box, lbox, peers, lpeers, mailbox = args[0:8]
        w_uv, gate, totals, output, parity, rank = args[8:14]
        H, D, B, cap = self.heads, self.size, self.batch, box.shape[1]
        smem = cutlass.utils.SmemAllocator()
        latent = alloc(smem, BFloat16, (16, 544), 128)
        scales = alloc(smem, Float32, (16, D))
        tid, _, _ = cute.arch.thread_idx()
        head, tile, pusher = cute.arch.block_idx()
        lane, warp = tid % 32, tid // 32
        group, quarter = lane // 4, lane % 4
        v = 32 * (warp // 4) + 8 * (warp % 4) + group
        # B fragments: 8 latent values per 32-wide chunk; A uses the same permutation.
        wb = [
            load_words(at(w_uv, head, 32 * c + 8 * quarter, v), count=4)
            for c in range(16)
        ]
        cute.arch.griddepcontrol_wait()
        threads = H * self.grid[1] * self.grid[2] * 512
        me = ((pusher * self.grid[1] + tile) * H + head) * 512 + tid
        for i in range(me, B * D * H * 64, threads):
            r, k = i // 64, i % 64  # r = b * D * H + global head
            b, gh = r // (D * H), r % (D * H)
            sent = self.sent_lse(rows_lse, lens, b, gh) > Float32(EMPTY_LSE)
            slot = ((parity * D + rank) * cap + b) * H + gh % H
            address = peers[gh // H] + (Int64(slot) * 512 + 8 * k) * 2
            if cutlass.const_expr(self.skip_empty):
                if sent:
                    words = load_words(at(rows, b, 0, gh, 8 * k), count=4, mutable=True)
                    store_relaxed(address, tuple(no_neg_zero(w) for w in words))
            else:  # zeros for an empty row: every row is polled
                words = (Uint32(0),) * 4
                if sent:
                    words = load_words(at(rows, b, 0, gh, 8 * k), count=4, mutable=True)
                store_relaxed(address, tuple(no_neg_zero(w) for w in words))
        for i in range(me, B * D * H, threads):
            b, gh = i // (D * H), i % (D * H)
            value = self.sent_lse(rows_lse, lens, b, gh)
            slot = ((parity * D + rank) * cap + b) * H + gh % H
            store_relaxed(lpeers[gh // H] + Int64(slot) * 4, (float_bits(value),))
        # The peer stores leave first; the local re-arm overlaps their flight.
        self.rearm(mailbox, me, threads)
        cute.arch.griddepcontrol_launch_dependents()
        if pusher == 0:
            if cutlass.const_expr(self.grid[1] == self.tiles):
                self.merge(args, latent, scales, wb, tid, tile)
            else:
                for t in range(tile, self.tiles, self.grid[1]):
                    self.merge(args, latent, scales, wb, tid, t)

    @cute.kernel
    def kernel_projected(self, args):
        """``partials`` "proj": the projecting CTAs (y < DCP) and the merging CTAs
        (y >= DCP). The projecting CTAs come first in launch order and never wait
        on another CTA (only griddepcontrol_wait on the sparse decode before them),
        so every rank's projections finish however many merging CTAs are
        resident; see the class docstring."""
        mailbox, w_uv = args[7], args[8]
        H, D, R = self.heads, self.size, self.prows
        merges: cutlass.Constexpr = self.grid[1] - D
        smem = cutlass.utils.SmemAllocator()
        latent = alloc(smem, BFloat16, (R, 544), 128)
        values = alloc(smem, BFloat16, (R, 136), 16)
        scales = alloc(smem, Float32, (16, D))
        tid, _, _ = cute.arch.thread_idx()
        head, y, _ = cute.arch.block_idx()
        lane, warp = tid % 32, tid // 32
        group, quarter = lane // 4, lane % 4
        threads: cutlass.Constexpr = H * self.grid[1] * 512
        me = (y * H + head) * 512 + tid
        if y < D:
            gh = y * H + head
            if cutlass.const_expr(self.preprojected):
                cute.arch.griddepcontrol_wait()
                self.push_values(args, gh, tid)
            else:
                v = 32 * (warp // 4) + 8 * (warp % 4) + group
                wb = [
                    load_words(at(w_uv, gh, 32 * c + 8 * quarter, v), count=4)
                    for c in range(16)
                ]
                cute.arch.griddepcontrol_wait()
                # Software pipeline: step t + 1's latents and LSEs load while
                # step t projects and pushes.
                ahead = self.load_tile(args, gh, 0, tid)
                for t0 in range(0, self.batch, R):
                    ahead = self.project(args, latent, values, wb, gh, t0, tid, ahead)
            self.rearm(mailbox, me, threads)
            cute.arch.griddepcontrol_launch_dependents()
        else:
            cute.arch.griddepcontrol_wait()
            self.rearm(mailbox, me, threads)
            cute.arch.griddepcontrol_launch_dependents()
            for t in range(y - D, self.tiles, merges):
                self.merge_projected(args, scales, tid, t)

    @cute.kernel
    def kernel_tc(self, args, am: tensormap.GridMap, bm: tensormap.GridMap):
        """"proj" above PROJ_ROWS rows: CTA c = y H + head projects units c, c + G,
        .. (unit u: gathered head u % (DCP H), rows 128 (u // (DCP H)) ..) on
        tcgen05 and pushes each unit's value rows to the head's owner in one bulk
        copy (the rows are contiguous there, rows with nothing to merge carry the
        sentinel) and its LSEs, then merges tiles y, y + grid[1], .. of head
        ``head``. Warp 0 issues the TMA ring and warp 1 the MMAs into two TMEM
        accumulators, both running ahead of the epilogue warps TC_EPI.. TC_EPI + 3
        (thread = row), which stage a unit's BF16 rows in one of two buffers while
        the previous unit's push is in flight."""
        rows_lse, lens, box, _, peers, lpeers, mailbox = args[1:8]
        parity, rank = args[12], args[13]
        H, D, B = self.heads, self.size, self.batch
        G: cutlass.Constexpr = H * self.grid[1]
        DH: cutlass.Constexpr = D * H
        cap = box.shape[2]
        smem = cutlass.utils.SmemAllocator()
        scales = alloc(smem, Float32, (16 * self.merge_tiles, D))
        ring = Uint32(smem.allocate_array(
            Uint8, TC_STAGES * TC_STAGE + 2 * TC_OUT + 128, byte_alignment=1024
        ).toint())
        outs = ring + Uint32(TC_STAGES * TC_STAGE)
        bars = outs + Uint32(2 * TC_OUT)  # full[st], empty[st]
        acc_full = bars + Uint32(16 * TC_STAGES)  # [2]
        acc_free = acc_full + Uint32(16)  # [2]
        slot_addr = acc_free + Uint32(16)
        tid, _, _ = cute.arch.thread_idx()
        head, y, _ = cute.arch.block_idx()
        lane = tid % 32
        warp = cute.arch.make_warp_uniform(tid // 32)
        c = y * H + head
        if tid == 0:
            for st in cutlass.range_constexpr(TC_STAGES):
                mbar_init(bars + Uint32(8 * st), Uint32(1))
                mbar_init(bars + Uint32(8 * (TC_STAGES + st)), Uint32(1))
            for b in cutlass.range_constexpr(2):
                mbar_init(acc_full + Uint32(8 * b), Uint32(1))
                mbar_init(acc_free + Uint32(8 * b), Uint32(128))
            fence_mbar_init()
        tm_a, tm_b = tensormap.map_address(am), tensormap.map_address(bm)
        self.mark(c, 0, tid)
        # W_UV does not depend on the predecessor: the first unit's B halves of the
        # first TC_STAGES stages load before the PDL wait (their A halves after).
        if cutlass.const_expr(self.knobs & 1 == 0):
            if tid == 0:
                if c < self.units:
                    for k in cutlass.range_constexpr(TC_STAGES):
                        full = bars + Uint32(8 * k)
                        mbar_arrive_tx(full, Uint32(TC_STAGE))
                        dst = ring + Uint32(TC_STAGE * k + TC_STAGE // 2)
                        _tma3d(dst, tm_b, full, 64 * k, 0, c % DH)
        # Allocate TMEM only once the predecessor grid is complete: under PDL its
        # CTAs may still be resident here and allocate TMEM after this CTA would.
        cute.arch.griddepcontrol_wait()
        if warp == 0:
            _tmem_alloc256(slot_addr)
        tc_fence_before()
        cute.arch.sync_threads()
        tc_fence_after()
        taddr = ld_shared_u32(slot_addr)
        self.mark(c, 1, tid)
        if cutlass.const_expr(self.knobs & 1 == 0):
            if warp == 0:
                if lane == 0:
                    kc = Int32(0)
                    for u in range(c, self.units, G):
                        gh = u % DH
                        t0 = (u // DH) * 128
                        for k in cutlass.range_constexpr(8):
                            st = (kc + k) % TC_STAGES
                            ph = ((kc + k) // TC_STAGES) & 1
                            full = bars + Uint32(8) * Uint32(st)
                            dst = ring + Uint32(TC_STAGE) * Uint32(st)
                            if kc + k >= TC_STAGES:  # (else B issued before the wait)
                                mbar_wait(
                                    bars + Uint32(8) * Uint32(TC_STAGES + st), ph ^ 1)
                                mbar_arrive_tx(full, Uint32(TC_STAGE))
                                _tma3d(dst + Uint32(TC_STAGE // 2), tm_b, full, 64 * k,
                                       0, gh)
                            _tma3d(dst, tm_a, full, 64 * k, gh, t0)
                        kc = kc + 8
        if warp == 1:
            if lane == 0:
                kc = Int32(0)
                j = Int32(0)
                for u in range(c, self.units, G):
                    b = j & 1
                    mbar_wait(acc_free + Uint32(8) * Uint32(b), ((j >> 1) & 1) ^ 1)
                    tc_fence_after()
                    acc = taddr + Uint32(128) * Uint32(b)
                    if cutlass.const_expr(self.knobs & 1 == 0):
                        for k in cutlass.range_constexpr(8):
                            st = (kc + k) % TC_STAGES
                            ph = ((kc + k) // TC_STAGES) & 1
                            mbar_wait(bars + Uint32(8) * Uint32(st), ph)
                            tc_fence_after()
                            if j == 0:
                                if cutlass.const_expr(k == 0):
                                    self.mark(c, 6, lane)
                                if cutlass.const_expr(k == 7):
                                    self.mark(c, 7, lane)
                            src = ring + Uint32(TC_STAGE) * Uint32(st)
                            for kk in cutlass.range_constexpr(4):
                                ad = Uint64(SDESC_SW128) | Uint64((src + Uint32(32 * kk)) >> 4)
                                bd = Uint64(SDESC_SW128) | Uint64(
                                    (src + Uint32(TC_STAGE // 2 + 32 * kk)) >> 4)
                                _mma_f16(acc, ad, bd, Uint32(TC_IDESC),
                                         Uint32(1 if k + kk > 0 else 0))
                            tc_commit(bars + Uint32(8) * Uint32(TC_STAGES + st))
                    tc_commit(acc_full + Uint32(8) * Uint32(b))
                    kc = kc + 8
                    j = j + 1
        if (warp >= TC_EPI) & (warp < TC_EPI + 4):
            row = tid - 32 * TC_EPI
            quad = Uint32(32 * (warp - TC_EPI)) << Uint32(16)
            # Every unit's LSEs first (they do not wait for the projection): the
            # owners' weights are ready before the value rows land.
            if cutlass.const_expr(self.knobs & 2 == 0):
                for u in range(c, self.units, G):
                    gh = u % DH
                    t = (u // DH) * 128 + row
                    if t < B:
                        base = ((parity * D + rank) * H + gh % H) * cap
                        lse_at = lpeers[gh // H] + Int64(base + t) * 4
                        lse = self.sent_lse(rows_lse, lens, t, gh)
                        store_relaxed(lse_at, (float_bits(lse),))
            j = Int32(0)
            for u in range(c, self.units, G):
                b = j & 1
                gh = u % DH
                t0 = (u // DH) * 128
                t = t0 + row
                live = t < B
                lse = Float32(EMPTY_LSE)
                if live:
                    lse = self.sent_lse(rows_lse, lens, t, gh)
                send = live & (lse > Float32(EMPTY_LSE))
                owner = gh // H
                base = ((parity * D + rank) * H + gh % H) * cap
                if row == 0:  # this buffer's push (two units ago) has read it
                    _bulk_read_wait1()
                _epi_sync()
                mbar_wait(acc_full + Uint32(8) * Uint32(b), (j >> 1) & 1)
                tc_fence_after()
                self.mark(c, 2 + 2 * j, row)
                out = outs + Uint32(TC_OUT) * Uint32(b) + Uint32(256) * Uint32(row)
                swz = row % 8  # chunk ^ (t % 8): conflict-free rows, as the mailbox
                acc = taddr + Uint32(128) * Uint32(b) + quad
                for c4 in cutlass.range_constexpr(4):  # (warp-collective TMEM loads)
                    v = _tmem_ld32(acc + Uint32(32 * c4))
                    for q4 in cutlass.range_constexpr(4):
                        words = (Uint32(PAIR_SENT),) * 4
                        if send:
                            words = tuple(
                                pack_pair(v[8 * q4 + 2 * i], v[8 * q4 + 2 * i + 1],
                                          policy="latent")
                                for i in range(4))
                        chunk = Uint32(swz ^ Int32(4 * c4 + q4))
                        _st_shared4(out + chunk * Uint32(16), *words)
                tc_fence_before()
                mbar_arrive(acc_free + Uint32(8) * Uint32(b))  # TMEM read
                _fence_async_shared()
                _epi_sync()
                if cutlass.const_expr(self.knobs & 2 == 0):
                    if row == 0:
                        n = B - t0
                        if n > 128:
                            n = 128
                        _bulk_push(peers[owner] + Int64(base + t0) * 256,
                                   outs + Uint32(TC_OUT) * Uint32(b), Uint32(256 * n))
                self.mark(c, 3 + 2 * j, row)
                j = j + 1
            if row == 0:
                _bulk_wait0()
            self.mark(c, 8, row)
        cute.arch.sync_threads()
        # Free TMEM before the dependents launch: a dependent CTA placed on this SM
        # while the merge spins may require TMEM column 0 (exclusive allocators).
        if warp == 0:
            tc_fence_after()
            _tmem_free256(taddr)
        self.rearm(mailbox, c * 512 + tid, G * 512)
        cute.arch.griddepcontrol_launch_dependents()
        self.resident_barrier(c, tid, parity)
        if cutlass.const_expr(self.knobs & 6 == 0):
            self.merge_rows(args, scales, tid, y)
        if cutlass.const_expr(self.stamp_addr != 0):
            cute.arch.sync_threads()
        self.mark(c, 10, tid)

    @cute.jit
    def chunk_rows(self, w, k):
        """(first row, rows) of chunk k of wave w (HS_ROWS-row TS publish blocks)."""
        G: cutlass.Constexpr = self.ts_grid
        rw = self.batch - w * G
        if rw > G:
            rw = Int32(G)
        n = rw - k * HS_ROWS
        if n > HS_ROWS:
            n = Int32(HS_ROWS)
        return w * G + k * HS_ROWS, n

    @cute.jit
    def stream_head(self, u):
        """Stream u's (wave, first chunk, chunks, gathered head): consecutive
        streams go to different owners."""
        DH: cutlass.Constexpr = self.size * self.heads
        q = u // DH
        i = u % DH
        w, k0, nk = Int32(0), Int32(0), Int32(0)
        before = 0
        for ww, (nkw, parts) in enumerate(self.parts):
            if (q >= before) & (q < before + parts):
                p = q - before
                w = Int32(ww)
                k0 = p * nkw // parts
                nk = (p + 1) * nkw // parts - k0
            before += parts
        return w, k0, nk, (i % self.size) * self.heads + i // self.size

    @cute.kernel
    def kernel_ov(self, args, am: tensormap.GridMap, bm: tensormap.GridMap):
        """"proj" overlapped with the persistent sparse core (sparse_mla_ts): one CTA
        per SM, placed as the core's CTAs exit (it triggers at entry). Work is
        claimed in streams (TS wave w, gathered head gh): the stream's W_UV goes to
        tensor memory once (TMA through the chunk ring, tcgen05.cp) and its rows
        follow in HS_ROWS-row chunks as the core publishes them (ld.acquire of the
        chunk's counter), each projected by MMAs with W_UV^T as the TMEM A operand
        and the chunk's latent rows as B (D = [128 value dims, rows]): a chunk costs
        its 32 KiB of latent rows. Warp 0 claims and issues the TMA ring
        (HS_RING slots), warp 1 the copies and MMAs (two accumulators), warp 2 the
        chunks' LSEs (pushes them, hands the epilogue the rows that send), the
        epilogue warps (thread = value dim) pack each chunk's rows (pairs by a lane
        shuffle, the mailbox's chunk permutation) and bulk-push them to the head
        owner. Once every stream is claimed the CTA merges claimed 32-row items
        (merge_item); the last CTA through re-arms the counters. No CTA waits on one
        that may not be dispatched: a CTA merges only after every stream of this
        rank was claimed, and streams wait only on the core, which waits on nothing
        here."""
        rows_lse, lens, box, _, peers, lpeers, _ = args[1:8]
        parity, rank, sync = args[12], args[13], args[14]
        H, D = self.heads, self.size
        G: cutlass.Constexpr = self.grid[0]
        NI: cutlass.Constexpr = OV_ITEMS
        NM: cutlass.Constexpr = HS_MASKS
        cap = box.shape[2]
        sbase = Int64(sync.iterator.toint())
        rbase = sbase + Int64(4 * SYNC_READY)
        smem = cutlass.utils.SmemAllocator()
        ring = Uint32(smem.allocate_array(
            Uint8, HS_RING * HS_CHUNK + HS_OUTS * HS_OUT + 512, byte_alignment=1024,
        ).toint())
        outs = ring + Uint32(HS_RING * HS_CHUNK)
        bars = outs + Uint32(HS_OUTS * HS_OUT)  # full[HS_RING], empty[HS_RING]
        acc_full = bars + Uint32(16 * HS_RING)  # [HS_ACCS]
        acc_free = acc_full + Uint32(8 * HS_ACCS)  # [HS_ACCS]
        item_full = acc_free + Uint32(8 * HS_ACCS)  # [NI]
        item_empty = item_full + Uint32(8 * NI)  # [NI]
        mask_full = item_empty + Uint32(8 * NI)  # [NM]
        mask_empty = mask_full + Uint32(8 * NM)  # [NM]
        items = mask_empty + Uint32(8 * NM)  # u32 [NI]
        masks = items + Uint32(4 * NI)  # u32 [NM]
        mslot = masks + Uint32(4 * NM)  # u32 [2]
        slot_addr = mslot + Uint32(8)
        tid, _, _ = cute.arch.thread_idx()
        c, _, _ = cute.arch.block_idx()
        lane = tid % 32
        warp = cute.arch.make_warp_uniform(tid // 32)
        if tid == 0:
            for st in cutlass.range_constexpr(HS_RING):
                mbar_init(bars + Uint32(8 * st), Uint32(1))
                mbar_init(bars + Uint32(8 * (HS_RING + st)), Uint32(1))
            for b in cutlass.range_constexpr(HS_ACCS):
                mbar_init(acc_full + Uint32(8 * b), Uint32(1))
                mbar_init(acc_free + Uint32(8 * b), Uint32(128))
            for i in cutlass.range_constexpr(NI):
                mbar_init(item_full + Uint32(8 * i), Uint32(1))
                mbar_init(item_empty + Uint32(8 * i), Uint32(130))
            for i in cutlass.range_constexpr(NM):
                mbar_init(mask_full + Uint32(8 * i), Uint32(1))
                mbar_init(mask_empty + Uint32(8 * i), Uint32(1))
            fence_mbar_init()
        tm_a, tm_b = tensormap.map_address(am), tensormap.map_address(bm)
        self.mark(c, 0, tid)
        # No CTA of the core shares this SM (its registers and shared memory leave
        # no room): tensor memory is free without griddepcontrol_wait.
        if warp == 0:
            _tmem_alloc512(slot_addr)
        tc_fence_before()
        cute.arch.sync_threads()
        tc_fence_after()
        taddr = ld_shared_u32(slot_addr)
        self.mark(c, 1, tid)
        if warp == 0:
            # lane 0 claims; lane (stage % HS_ISS) issues every load of a ring stage:
            # one thread completes one barrier stage per ~0.25-0.5 us whatever its size
            # (hwdb INDEX 39), HS_ISS issuing lanes keep HS_ISS stages in flight
            kc = Int32(0)
            jc = Int32(0)
            i = Int32(0)
            go = Int32(1)
            while go != 0:
                uu = Int32(0)
                if lane == 0:
                    uu = Int32(_atom_add_gpu(sbase)[0])
                u = cute.arch.shuffle_sync(uu, 0)
                if i == 0:
                    self.mark(c, 4, lane)
                if lane == 0:
                    s = i % NI
                    mbar_wait(item_empty + Uint32(8) * Uint32(s), ((i // NI) & 1) ^ 1)
                    st_shared_u32(items + Uint32(4) * Uint32(s), Uint32(u))
                    mbar_arrive(item_full + Uint32(8) * Uint32(s))
                if u < self.units:
                    w, k0, nk, gh = self.stream_head(u)
                    # W_UV (2 tiles per ring slot) does not wait for the core
                    # each issuing lane runs its own stages (independent progress)
                    if lane < 4:
                        st = (kc + lane) % HS_RING
                        ph = ((kc + lane) // HS_RING) & 1
                        full = bars + Uint32(8) * Uint32(st)
                        dst = ring + Uint32(HS_CHUNK) * Uint32(st)
                        mbar_wait(bars + Uint32(8) * Uint32(HS_RING + st), ph ^ 1)
                        mbar_arrive_tx(full, Uint32(HS_CHUNK))
                        for e in cutlass.range_constexpr(2):
                            _tma3d(dst + Uint32(e * HS_CHUNK // 2), tm_b, full,
                                   64 * (2 * lane + e), 0, gh)
                    kc = kc + 4
                    # the readiness warp saw every chunk of the part published before
                    # it hands over the first one
                    if lane == 0:
                        mbar_wait(mask_full + Uint32(8) * Uint32(jc % NM), (jc // NM) & 1)
                    cute.arch.sync_warp()
                    if lane < HS_ISS:
                        _fence_proxy_async_global()
                    jc = jc + nk
                    if i == 0:
                        self.mark(c, 5, lane)
                    if lane < HS_ISS:
                        k = (lane - kc % HS_ISS + HS_ISS) % HS_ISS
                        while k < nk:
                            t0, n = self.chunk_rows(w, k0 + k)
                            st = (kc + k) % HS_RING
                            ph = ((kc + k) // HS_RING) & 1
                            full = bars + Uint32(8) * Uint32(st)
                            dst = ring + Uint32(HS_CHUNK) * Uint32(st)
                            mbar_wait(bars + Uint32(8) * Uint32(HS_RING + st), ph ^ 1)
                            mbar_arrive_tx(full, Uint32(HS_CHUNK))
                            for jj in cutlass.range_constexpr(8):
                                _tma3d(dst + Uint32(jj * HS_CHUNK // 8), tm_a, full,
                                       64 * jj, gh, t0)
                            if (i == 0) & (k < 4):
                                self.mark(c, 12 + k, 0)
                            k = k + HS_ISS
                    cute.arch.sync_warp()
                    kc = kc + nk
                    self.mark(c, 7, lane)
                else:
                    go = Int32(0)
                i = i + 1
        if warp == 1:
            if lane == 0:
                kc = Int32(0)
                j = Int32(0)
                i = Int32(0)
                go = Int32(1)
                while go != 0:
                    s = i % NI
                    mbar_wait(item_full + Uint32(8) * Uint32(s), (i // NI) & 1)
                    u = Int32(ld_shared_u32(items + Uint32(4) * Uint32(s)))
                    if u < self.units:
                        w, k0, nk, gh = self.stream_head(u)
                        if i > 0:  # the previous stream's MMAs read W_UV in TMEM
                            jl = j - 1
                            mbar_wait(acc_full + Uint32(8) * Uint32(jl % HS_ACCS), (jl // HS_ACCS) & 1)
                        for h2 in cutlass.range_constexpr(4):
                            st = kc % HS_RING
                            ph = (kc // HS_RING) & 1
                            mbar_wait(bars + Uint32(8) * Uint32(st), ph)
                            tc_fence_after()
                            uvd = Uint64(SDESC_SW128) | Uint64(
                                (ring + Uint32(HS_CHUNK) * Uint32(st)) >> Uint32(4))
                            for e in cutlass.range_constexpr(2 if self.knobs & 32768 == 0
                                                             else 0):
                                for q4 in cutlass.range_constexpr(4):
                                    _tmem_cp(taddr + Uint32(32 * (2 * h2 + e) + 8 * q4),
                                             uvd + Uint64(e * HS_CHUNK // 2 // 16 + 2 * q4))
                            tc_commit(bars + Uint32(8) * Uint32(HS_RING + st))
                            kc = kc + 1
                        for k in range(nk):
                            st = kc % HS_RING
                            ph = (kc // HS_RING) & 1
                            b = j % HS_ACCS
                            mbar_wait(acc_free + Uint32(8) * Uint32(b), ((j // HS_ACCS) & 1) ^ 1)
                            mbar_wait(bars + Uint32(8) * Uint32(st), ph)
                            tc_fence_after()
                            if (i == 0) & (k < 4):
                                self.mark(c, 28 + k, lane)
                            acc = taddr + Uint32(HS_ACC) + Uint32(HS_ROWS) * Uint32(b)
                            src = ring + Uint32(HS_CHUNK) * Uint32(st)
                            for kk in cutlass.range_constexpr(32 if self.knobs & 32768 == 0
                                                              else 0):
                                bd = Uint64(SDESC_SW128) | Uint64(
                                    (src + Uint32((kk // 4) * HS_CHUNK // 8 + 32 * (kk % 4)))
                                    >> 4)
                                _mma_f16_ts(acc, taddr + Uint32(8 * kk), bd,
                                            Uint32(HS_IDESC), Uint32(1 if kk > 0 else 0))
                            tc_commit(bars + Uint32(8) * Uint32(HS_RING + st))
                            tc_commit(acc_full + Uint32(8) * Uint32(b))
                            if (i == 0) & (k < 4):
                                self.mark(c, 16 + k, lane)
                            kc = kc + 1
                            j = j + 1
                        mbar_arrive(item_empty + Uint32(8) * Uint32(s))
                    else:
                        go = Int32(0)
                    i = i + 1
        if warp == 2:  # the chunks' LSEs (lane = row), ahead of the epilogue
            j = Int32(0)
            i = Int32(0)
            go = Int32(1)
            while go != 0:
                s = i % NI
                mbar_wait(item_full + Uint32(8) * Uint32(s), (i // NI) & 1)
                u = Int32(ld_shared_u32(items + Uint32(4) * Uint32(s)))
                cute.arch.sync_warp()
                if lane == 0:
                    mbar_arrive(item_empty + Uint32(8) * Uint32(s))
                if u < self.units:
                    w, k0, nk, gh = self.stream_head(u)
                    owner = gh // H
                    base = ((parity * D + rank) * H + gh % H) * cap
                    # lane k waits for chunk k0 + k of the part to be published, then
                    # one acquire fence covers them all
                    if lane < nk:
                        tk, nkk = self.chunk_rows(w, k0 + lane)
                        ready = rbase + Int64(4) * Int64(w * self.per + k0 + lane)
                        while _ld_relaxed_gpu(ready)[0] < Uint32(nkk):
                            _nanosleep256()
                    cute.arch.sync_warp()
                    if lane == 0:
                        _fence_gpu()
                    cute.arch.sync_warp()
                    # every chunk's LSE loads at once (lane = row; rows past a chunk
                    # read row 0 and are dropped)
                    lses, lives, rws = [], [], []
                    for kk in cutlass.range_constexpr(HS_PART):
                        t0, n = self.chunk_rows(w, k0 + kk)
                        live = (kk < nk) & (lane < n)
                        t = t0 + lane
                        if not live:
                            t = Int32(0)
                        if cutlass.const_expr(self.knobs & 16384 == 0):
                            lses.append(self.sent_lse(rows_lse, lens, t, gh))
                        else:
                            lses.append(Float32(EMPTY_LSE))
                        lives.append(live)
                        rws.append(t)
                    for kk in cutlass.range_constexpr(HS_PART):
                        if kk < nk:
                            lse = lses[kk]
                            if not lives[kk]:
                                lse = Float32(EMPTY_LSE)
                            send = lives[kk] & (lse > Float32(EMPTY_LSE))
                            mask = cute.arch.vote_ballot_sync(send)
                            if cutlass.const_expr(self.knobs & 2 == 0):
                                if lives[kk]:
                                    store_relaxed(lpeers[owner] + Int64(base + rws[kk]) * 4,
                                                  (float_bits(lse),))
                            m = j % NM
                            if lane == 0:
                                mbar_wait(mask_empty + Uint32(8) * Uint32(m),
                                          ((j // NM) & 1) ^ 1)
                                st_shared_u32(masks + Uint32(4) * Uint32(m), Uint32(mask))
                                mbar_arrive(mask_full + Uint32(8) * Uint32(m))
                            j = j + 1
                else:
                    go = Int32(0)
                i = i + 1
        if (warp >= TC_EPI) & (warp < TC_EPI + 4):
            v = tid - 32 * TC_EPI
            j = Int32(0)
            i = Int32(0)
            go = Int32(1)
            while go != 0:
                s = i % NI
                mbar_wait(item_full + Uint32(8) * Uint32(s), (i // NI) & 1)
                u = Int32(ld_shared_u32(items + Uint32(4) * Uint32(s)))
                mbar_arrive(item_empty + Uint32(8) * Uint32(s))
                if u < self.units:
                    w, k0, nk, gh = self.stream_head(u)
                    owner = gh // H
                    base = ((parity * D + rank) * H + gh % H) * cap
                    for k in range(nk):
                        t0, n = self.chunk_rows(w, k0 + k)
                        b = j % HS_ACCS
                        out = outs + Uint32(HS_OUT) * Uint32(j % HS_OUTS)
                        if v == 0:  # this buffer's push (HS_OUTS chunks ago) read it
                            _bulk_read_wait3()
                        m = j % NM
                        mbar_wait(mask_full + Uint32(8) * Uint32(m), (j // NM) & 1)
                        mask = ld_shared_u32(masks + Uint32(4) * Uint32(m))
                        mbar_wait(acc_full + Uint32(8) * Uint32(b), (j // HS_ACCS) & 1)
                        tc_fence_after()
                        if j == 0:
                            self.mark(c, 2, v)
                        if j < 4:
                            self.mark(c, 20 + j, v)
                        # 16x256b loads: lane l gets dims (l / 4, 8 + l / 4) of its 16-lane
                        # half x rows 8 g + 2 (l % 4) + {0, 1}: m8n8 fragments (dims as
                        # rows) that stmatrix.trans stores as 16 B value-row chunks
                        accq = (taddr + Uint32(HS_ACC) + Uint32(HS_ROWS) * Uint32(b)
                                + (Uint32(32 * (warp - TC_EPI)) << Uint32(16)))
                        halves = [_tmem_ld16x256(accq + Uint32(16 * hh << 16))
                                  for hh in range(2 if self.knobs & 16384 == 0 else 0)]
                        tc_fence_before()
                        mbar_arrive(acc_free + Uint32(8) * Uint32(b))  # TMEM read
                        _epi_sync()  # every thread read the mask; the buffer is free
                        if v == 0:
                            mbar_arrive(mask_empty + Uint32(8) * Uint32(m))
                        r0 = 2 * (lane % 4)
                        sent = Uint32(0)  # the unsent rows' sentinel halves of this dim
                        if (lane // 4) % 2 == 1:
                            sent = Uint32(0x80008000)
                        mk = lane // 8
                        for hh in cutlass.range_constexpr(2 if self.knobs & 16384 == 0 else 0):
                            frag = []
                            for g in cutlass.range_constexpr(4):
                                rows2 = (mask >> Uint32(8 * g + r0)) & Uint32(3)
                                keep = (rows2 & Uint32(1)) * Uint32(0xFFFF) | (
                                    rows2 >> Uint32(1)) * Uint32(0xFFFF0000)
                                for vb in cutlass.range_constexpr(2):
                                    bits = pack_pair(halves[hh][4 * g + 2 * vb],
                                                     halves[hh][4 * g + 2 * vb + 1],
                                                     policy="latent")
                                    frag.append((bits & keep) | (sent & (keep ^ Uint32(
                                        0xFFFFFFFF))))
                            for sx in cutlass.range_constexpr(2):
                                row = 8 * (2 * sx + mk // 2) + lane % 8
                                chunk = Uint32(4 * (warp - TC_EPI) + 2 * hh + mk % 2) ^ Uint32(
                                    (t0 + row) % 8)
                                stmatrix_t4(out + Uint32(256) * Uint32(row) + chunk * Uint32(16),
                                            frag[4 * sx], frag[4 * sx + 1], frag[4 * sx + 2],
                                            frag[4 * sx + 3])
                        _fence_async_shared()
                        _epi_sync()
                        if cutlass.const_expr(self.knobs & (2 | 16384) == 0):
                            if v == 0:
                                _bulk_push(peers[owner] + Int64(base + t0) * 256, out,
                                           Uint32(256 * n))
                        if j == 0:
                            self.mark(c, 3, v)
                        if j < 4:
                            self.mark(c, 24 + j, v)
                        j = j + 1
                else:
                    go = Int32(0)
                i = i + 1
            if v == 0:
                _bulk_wait0()
            self.mark(c, 8, v)
            if cutlass.const_expr(self.stamp_addr != 0):
                if v == 0:  # chunks this CTA projected (a count, not a time)
                    st_global_u64(Int64(self.stamp_addr) + (Int64(c) * STAMP_SLOTS
                                                            + Int64(11)) * 8, Uint64(j))
        cute.arch.sync_threads()
        if warp == 0:
            tc_fence_after()
            _tmem_free512(taddr)
        cute.arch.griddepcontrol_launch_dependents()
        if cutlass.const_expr(self.knobs & 6 == 0):
            k = Int32(0)
            go = Int32(1)
            while go != 0:
                slot = mslot + Uint32(4) * Uint32(k % 2)
                if tid == 0:
                    st_shared_u32(slot, _atom_add_gpu(sbase + Int64(4))[0])
                cute.arch.sync_threads()
                m = Int32(ld_shared_u32(slot))
                if m < self.merge_items:
                    self.merge_item(args, m, tid)
                else:
                    go = Int32(0)
                k = k + 1
        self.mark(c, 9, tid)
        cute.arch.sync_threads()
        if tid == 0:
            _fence_gpu()  # this CTA's claims before its done count
            if _atom_add_gpu(sbase + Int64(8))[0] == Uint32(G - 1):
                _fence_gpu()
                for q in cutlass.range_constexpr(SYNC_READY + self.nblocks):
                    _store_u32(sbase + Int64(4 * q), Uint32(0))
        self.mark(c, 10, tid)

    @cute.jit
    def merge_item(self, args, m, tid):
        """Merge item m (32-row group m // H, head m % H): thread tid merges 16 B
        chunk tid % 16 of row 32 (m // H) + tid // 16 over the DCP sources in the
        merge_rows form (every load first, re-polls of what is still unpublished,
        the chunk-0 thread re-arms the row's LSEs after a warp sync; the same
        arithmetic)."""
        lbox, gate, totals, output = args[4], args[9], args[10], args[11]
        box = args[3]
        D, B = self.size, self.batch
        head = m % self.heads
        chunk = tid % 16
        t = (m // self.heads) * 32 + tid // 16
        live = t < B
        if not live:
            t = Int32(0)
        col = head * 128 + 8 * chunk
        gates = load_words(at(gate, t, col), count=4, mutable=True)
        active = totals[t // self.q] > 0
        swz = chunk ^ (t % 8)
        srcs = [at(box, s, head, t, 8 * swz) for s in range(D)]
        lsrcs = [at(lbox, s, head, t) for s in range(D)]
        lses = cute.make_rmem_tensor((D,), Uint32)
        words = cute.make_rmem_tensor((D * 4,), Uint32)
        for s in cutlass.range_constexpr(D):
            lses[s] = load_volatile(lsrcs[s])[0]
            loaded = load_volatile(srcs[s], count=4)
            for e in cutlass.range_constexpr(4):
                words[s * 4 + e] = loaded[e]
        lives = [live]
        pending = self.unmerged(lses, words, lives)
        while pending:
            _nanosleep256()  # the sources' pushes are in flight: poll gently
            for s in cutlass.range_constexpr(D):
                if live & (lses[s] == Uint32(LSE_SENT)):
                    lses[s] = load_volatile(lsrcs[s])[0]
                if self.value_due(lses, words, lives, 0, s):
                    loaded = load_volatile(srcs[s], count=4)
                    for e in cutlass.range_constexpr(4):
                        words[4 * s + e] = loaded[e]
            pending = self.unmerged(lses, words, lives)
        cute.arch.sync_warp()  # every thread of a row read its LSEs
        if live:
            values = [bits_float(lses[s]) for s in range(D)]
            if chunk == 0:
                for s in cutlass.range_constexpr(D):
                    store_words(lsrcs[s], (Uint32(LSE_SENT),))
            top = values[0]
            for s in cutlass.range_constexpr(1, D):
                top = max(top, values[s])
            top = max(top, Float32(EMPTY_LSE))
            weights = []
            for s in cutlass.range_constexpr(D):
                w = Float32(0)
                if values[s] > Float32(EMPTY_LSE):
                    w = cute.math.exp2(values[s] - top, fastmath=True)
                weights.append(w)
            sums = weights + [Float32(0)] * ((1 << (D - 1).bit_length()) - D)
            for _ in cutlass.range_constexpr((len(sums) - 1).bit_length()):
                half = len(sums) // 2
                sums = [sums[k] + sums[k + half] for k in range(half)]
            total = sums[0]
            inverse = Float32(0)
            if total > 0:
                inverse = div_rn(Float32(1), total)
            merged = zeros(8, Float32)
            for s in cutlass.range_constexpr(D):
                if values[s] > Float32(EMPTY_LSE):
                    w = weights[s] * inverse
                    store_words(srcs[s], (Uint32(PAIR_SENT),) * 4)
                    for e in cutlass.range_constexpr(4):
                        lo, hi = unpack_bf16(words[4 * s + e])
                        merged[2 * e] += w * lo
                        merged[2 * e + 1] += w * hi
            out_words = []
            for e in cutlass.range_constexpr(4):
                # Padded rows may hold a NaN gate; keep inactive rows zero.
                lo, hi = Float32(0), Float32(0)
                if active:
                    g = unpack_bf16(gates[e])
                    lo, hi = (
                        Float32(BFloat16(merged[2 * e + k])) * sigmoid(g[k], False)
                        for k in range(2)
                    )
                out_words.append(bf16x2(lo, hi))
            store_words(at(output, t, col), tuple(out_words))

    @staticmethod
    def stale_chunk(words, k):
        """Whether the 16 B chunk words[k .. k + 3] still holds the sentinel."""
        stale = words[k] == Uint32(PAIR_SENT)
        for e in range(1, 4):
            stale |= words[k + e] == Uint32(PAIR_SENT)
        return stale

    def stale(self, words, need):
        """Whether the chunk of a source in ``need`` is still unpublished."""
        pending = need[0] & self.stale_chunk(words, 0)
        for k in range(1, len(need)):
            pending |= need[k] & self.stale_chunk(words, 4 * k)
        return pending

    @cute.jit
    def backoff(self):
        """Bench probe (KNOBS 128 / 384): sleep between polls."""
        if cutlass.const_expr(self.knobs & 128 != 0):
            if cutlass.const_expr(self.knobs & 256 != 0):
                _nanosleep256()
            else:
                _nanosleep64()

    @cute.jit
    def mark(self, c, slot, who):
        """Bench probe: thread ``who`` == 0 stamps globaltimer into slot ``slot`` of
        CTA ``c`` (STAMP_ADDR; slots past STAMP_SLOTS are dropped)."""
        if cutlass.const_expr(self.stamp_addr != 0):
            if (who == 0) & (slot < STAMP_SLOTS):
                at_ = Int64(self.stamp_addr) + (
                    Int64(c) * STAMP_SLOTS + Int64(slot)) * 8
                st_global_u64(at_, globaltimer())

    @cute.jit
    def mark_max(self, c, slot):
        """Bench probe: every caller folds globaltimer into slot ``slot`` (max)."""
        if cutlass.const_expr(self.stamp_addr != 0):
            at_ = Int64(self.stamp_addr) + (Int64(c) * STAMP_SLOTS + Int64(slot)) * 8
            _red_max_u64(at_, globaltimer())

    @cute.jit
    def resident_barrier(self, c, tid, parity):
        """Bench probe (KNOBS 4096): after this CTA's pushes drained, wait until every
        rank's CTAs drained theirs (stamp slot 24), then reset for the parity's
        next use (the last local CTA through)."""
        if cutlass.const_expr(self.knobs & 4096 != 0):
            if tid == 0:
                _fence_sys()
                for r in cutlass.range_constexpr(self.size):
                    _bar_arrive_sys(Int64(self.bar_peers[r]) + Int64(parity) * 4)
                mine = Int64(self.bar_done)
                target = Uint32(self.size * self.grid[0] * self.grid[1])
                count = Int64(self.bar_own) + Int64(parity) * 4
                while _ld_acquire_sys(count)[0] < target:
                    pass
                if _atom_add_gpu(mine + Int64(parity) * 4)[0] == Uint32(
                        self.grid[0] * self.grid[1] - 1):
                    _store_u32(count, Uint32(0))
                    _store_u32(mine + Int64(parity) * 4, Uint32(0))
            cute.arch.sync_threads()
            self.mark(c, 24, tid)

    @cute.jit
    def merge_rows(self, args, scales, tid, y):
        """merge_projected over every row of tiles y, y + grid[1], .. at once: a
        thread merges chunk i % 16 of slot row i // 16 for i = tid, tid + 512, ..
        (slot r = 16 j + i' is row 16 (y + j grid[1]) + i'), and issues every load
        of its rows first (the DCP sources' LSEs and value chunks, the gate): one
        round trip once they landed, then rounds that reload only what is still
        unpublished. The 16 threads of a row are one half warp; its chunk-0 thread
        re-arms the row's LSEs after a warp sync. Same arithmetic: the source
        weights exp2(lse - max) times the IEEE reciprocal of their butterfly sum,
        the FP32 merge in source order, one BF16 rounding, the gate."""
        lbox, gate, totals, output = args[4], args[9], args[10], args[11]
        box = args[3]
        D, B = self.size, self.batch
        head, _, _ = cute.arch.block_idx()
        R: cutlass.Constexpr = 16 * self.merge_tiles
        step: cutlass.Constexpr = self.grid[1]
        N: cutlass.Constexpr = -(-R * 16 // 512)  # rows chunks per thread
        chunk = tid % 16
        rows, lives, gates, actives, srcs, lsrcs = [], [], [], [], [], []
        lses = cute.make_rmem_tensor((N * D,), Uint32)
        words = cute.make_rmem_tensor((N * D * 4,), Uint32)
        for n in cutlass.range_constexpr(N):
            r = (tid + 512 * n) // 16
            t = (y + (r // 16) * step) * 16 + r % 16
            live = (r < R) & (t < B)
            if not live:
                t = Int32(0)
            col = head * 128 + 8 * chunk
            gates.append(load_words(at(gate, t, col), count=4, mutable=True))
            actives.append(totals[t // self.q] > 0)
            swz = chunk ^ (t % 8)
            srcs.append([at(box, s, head, t, 8 * swz) for s in range(D)])
            lsrcs.append([at(lbox, s, head, t) for s in range(D)])
            for s in cutlass.range_constexpr(D):
                lses[n * D + s] = load_volatile(lsrcs[n][s])[0]
                loaded = load_volatile(srcs[n][s], count=4)
                for e in cutlass.range_constexpr(4):
                    words[(n * D + s) * 4 + e] = loaded[e]
            rows.append(t)
            lives.append(live)
        pending = self.unmerged(lses, words, lives)
        while pending:
            self.backoff()
            for n in cutlass.range_constexpr(N):
                for s in cutlass.range_constexpr(D):
                    k = n * D + s
                    if lives[n] & (lses[k] == Uint32(LSE_SENT)):
                        lses[k] = load_volatile(lsrcs[n][s])[0]
                    if self.value_due(lses, words, lives, n, s):
                        loaded = load_volatile(srcs[n][s], count=4)
                        for e in cutlass.range_constexpr(4):
                            words[4 * k + e] = loaded[e]
            pending = self.unmerged(lses, words, lives)
        self.mark(y * self.heads + head, 9, tid)
        cute.arch.sync_warp()  # every thread of a row read its LSEs
        for n in cutlass.range_constexpr(N):
            if lives[n]:
                t = rows[n]
                values = [bits_float(lses[n * D + s]) for s in range(D)]
                if chunk == 0:
                    for s in cutlass.range_constexpr(D):
                        store_words(lsrcs[n][s], (Uint32(LSE_SENT),))
                top = values[0]
                for s in cutlass.range_constexpr(1, D):
                    top = max(top, values[s])
                top = max(top, Float32(EMPTY_LSE))
                weights = []
                for s in cutlass.range_constexpr(D):
                    w = Float32(0)
                    if values[s] > Float32(EMPTY_LSE):
                        w = cute.math.exp2(values[s] - top, fastmath=True)
                    weights.append(w)
                sums = weights + [Float32(0)] * ((1 << (D - 1).bit_length()) - D)
                for _ in cutlass.range_constexpr((len(sums) - 1).bit_length()):
                    half = len(sums) // 2
                    sums = [sums[k] + sums[k + half] for k in range(half)]
                total = sums[0]
                inverse = Float32(0)
                if total > 0:
                    inverse = div_rn(Float32(1), total)
                merged = zeros(8, Float32)
                for s in cutlass.range_constexpr(D):
                    if values[s] > Float32(EMPTY_LSE):
                        w = weights[s] * inverse
                        k = (n * D + s) * 4
                        store_words(srcs[n][s], (Uint32(PAIR_SENT),) * 4)
                        for e in cutlass.range_constexpr(4):
                            lo, hi = unpack_bf16(words[k + e])
                            merged[2 * e] += w * lo
                            merged[2 * e + 1] += w * hi
                col = head * 128 + 8 * chunk
                out_words = []
                for e in cutlass.range_constexpr(4):
                    # Padded rows may hold a NaN gate; keep inactive rows zero.
                    lo, hi = Float32(0), Float32(0)
                    if actives[n]:
                        g = unpack_bf16(gates[n][e])
                        lo, hi = (
                            Float32(BFloat16(merged[2 * e + k])) * sigmoid(g[k], False)
                            for k in range(2)
                        )
                    out_words.append(bf16x2(lo, hi))
                store_words(at(output, t, col), tuple(out_words))
                self.mark_max(y * self.heads + head, 17 + 2 * n)
            self.mark(y * self.heads + head, 11 + n, tid)

    def value_due(self, lses, words, lives, n, s):
        """Whether row chunk n's source s value chunk must still be (re)loaded: its
        LSE arrived, names a live source, and the chunk holds the sentinel."""
        i = n * self.size + s
        word = lses[i]
        due = lives[n] & (word != Uint32(LSE_SENT))
        due &= bits_float(word) > Float32(EMPTY_LSE)
        stale = words[4 * i] == Uint32(PAIR_SENT)
        for e in range(1, 4):
            stale |= words[4 * i + e] == Uint32(PAIR_SENT)
        return due & stale

    def unmerged(self, lses, words, lives):
        """Whether a live row chunk still waits for an LSE or a value chunk."""
        d = self.size
        pending = None
        for n in range(len(lives)):
            for s in range(d):
                waiting = lives[n] & (lses[n * d + s] == Uint32(LSE_SENT))
                waiting |= self.value_due(lses, words, lives, n, s)
                pending = waiting if pending is None else pending | waiting
        return pending

    @cute.jit
    def push_values(self, args, gh, tid):
        """``preprojected``: push gathered head ``gh``'s value rows (the W_UV
        argument, [DCP * H, B, 128] BF16) and every row's LSE to the head's owner,
        a row with nothing to merge only its LSE (as ``project``). Thread tid
        takes 16 B chunk tid % 16 of rows tid // 16, + 32, ..: four rows' loads in
        flight per thread."""
        rows_lse, lens, box, _, peers, lpeers = args[1:7]
        values, parity, rank = args[8], args[12], args[13]
        H, D, B = self.heads, self.size, self.batch
        cap = box.shape[2]
        chunk, owner = tid % 16, gh // H
        per: cutlass.Constexpr = 4  # rows per thread per round
        for r0 in range(tid // 16, B, 32 * per):
            loads = []
            for i in cutlass.range_constexpr(per):
                t = r0 + 32 * i
                lse = Float32(EMPTY_LSE)
                words = (Uint32(0),) * 4
                if t < B:
                    lse = self.sent_lse(rows_lse, lens, t, gh)
                    if lse > Float32(EMPTY_LSE):
                        words = load_words(
                            at(values, gh, t, 8 * chunk), count=4, mutable=True
                        )
                loads.append((t, lse, words))
            for t, lse, words in loads:
                if t < B:
                    slot = ((parity * D + rank) * H + gh % H) * cap + t
                    if lse > Float32(EMPTY_LSE):
                        chunk_at = Int64(slot) * 128 + 8 * (chunk ^ (t % 8))
                        target = peers[owner] + chunk_at * 2
                        store_relaxed(target, tuple(no_neg_zero(w) for w in words))
                    if chunk == 0:
                        address = lpeers[owner] + Int64(slot) * 4
                        store_relaxed(address, (float_bits(lse),))

    @cute.jit
    def load_tile(self, args, gh, t0, tid):
        """This thread's 16 B chunks of the BF16 latent rows t0.. t0 + prows - 1 of
        gathered head ``gh`` (chunk tid + 512 j: row // 64, 8 (chunk % 64)); zeros
        beyond the batch), led by the LSEs of the rows it pushes ((tid + 512 j)
        // 16; sent_lse, EMPTY_LSE beyond the batch)."""
        rows, rows_lse, lens = args[0:3]
        R = self.prows
        chunks = []
        for j in cutlass.range_constexpr((16 * R + 511) // 512):  # pushed rows
            lse = Float32(EMPTY_LSE)
            row = (tid + 512 * j) // 16
            if (row < R) & (t0 + row < self.batch):
                lse = self.sent_lse(rows_lse, lens, t0 + row, gh)
            chunks.append(lse)
        for j in cutlass.range_constexpr(R // 8):
            r, k = (tid + 512 * j) // 64, (tid + 512 * j) % 64
            words = (Uint32(0),) * 4
            if t0 + r < self.batch:
                words = load_words(
                    at(rows, t0 + r, 0, gh, 8 * k), count=4, mutable=True
                )
            chunks.append(words)
        return tuple(chunks)

    @cute.jit
    def project(self, args, latent, values, wb, gh, t0, tid, tile):
        """Rows t0.. t0 + prows - 1 of gathered head ``gh`` (``tile``: this
        thread's latent chunks and LSE of them, load_tile): stage the BF16 latents,
        issue the next step's loads, W_UV (BF16 MMA, FP32 accumulation, rounded to
        BF16; one 16-row block per accumulator set) and push the [prows, 128] value
        rows (and every row's LSE; a row with nothing to merge sends only its LSE).
        Returns the next step's load_tile."""
        box, _, peers, lpeers = args[3:7]
        parity, rank = args[12], args[13]
        H, D, B, R = self.heads, self.size, self.batch, self.prows
        cap = box.shape[2]
        lane, warp = tid % 32, tid // 32
        group, quarter = lane // 4, lane % 4
        cute.arch.sync_threads()  # the previous step's readers are done
        pushes = (16 * R + 511) // 512  # the LSEs lead the tile (load_tile)
        for j in cutlass.range_constexpr(R // 8):
            r, k = (tid + 512 * j) // 64, (tid + 512 * j) % 64
            store_shared_words(latent.iterator + r * 544 + 8 * k, tile[pushes + j])
        cute.arch.sync_threads()
        ahead = self.load_tile(args, gh, t0 + R, tid)
        col = 32 * (warp // 4) + 8 * (warp % 4) + 2 * quarter
        for g in cutlass.range_constexpr(R // 16):
            parts = zeros((2, 4), Float32)  # two accumulation chains
            for c in cutlass.range_constexpr(16):
                x, y = (
                    load_words(
                        latent.iterator + (16 * g + r) * 544 + 32 * c + 8 * quarter,
                        count=4,
                        shared=True,
                    )
                    for r in (group, group + 8)
                )
                for s in cutlass.range_constexpr(2):
                    a = (x[2 * s], y[2 * s], x[2 * s + 1], y[2 * s + 1])
                    b = (wb[c][2 * s], wb[c][2 * s + 1])
                    mma_into(parts[s, None], a, b, bf16=True)
            for r in cutlass.range_constexpr(2):
                pair = bf16x2(
                    parts[0, 2 * r] + parts[1, 2 * r],
                    parts[0, 2 * r + 1] + parts[1, 2 * r + 1],
                )
                address = values.iterator + (16 * g + group + 8 * r) * 136 + col
                store_shared_words(address, (no_neg_zero(pair),))
        cute.arch.sync_threads()
        for j in cutlass.range_constexpr(pushes):
            row, chunk = (tid + 512 * j) // 16, tid % 16
            t = t0 + row
            if (row < R) & (t < B):
                lse = tile[j]
                slot = ((parity * D + rank) * H + gh % H) * cap + t
                owner = gh // H
                if lse > Float32(EMPTY_LSE):
                    sent = load_words(
                        values.iterator + row * 136 + 8 * chunk, count=4, shared=True
                    )
                    chunk_at = Int64(slot) * 128 + 8 * (chunk ^ (t % 8))
                    target = peers[owner] + chunk_at * 2
                    store_relaxed(target, sent)
                if chunk == 0:
                    store_relaxed(lpeers[owner] + Int64(slot) * 4, (float_bits(lse),))
        return ahead

    @cute.jit
    def merge_projected(self, args, scales, tid, tile):
        """The owner's merge of the value rows of request rows t0.. of this head over
        the DCP sources (FP32, polled 8 at a time, re-armed as read), rounded to
        BF16 as the latent merge's W_UV output, then the gate."""
        box, lbox, gate, totals, output = args[3], args[4], args[9], args[10], args[11]
        D, B = self.size, self.batch
        head, _, _ = cute.arch.block_idx()
        lane, warp = tid % 32, tid // 32
        t0 = tile * 16
        if t0 + warp < B:  # warp r: the source weights of request t0 + r
            lse = Float32(EMPTY_LSE)
            if lane < D:
                source = at(lbox, lane, head, t0 + warp)
                word = load_volatile(source)[0]
                while word == Uint32(LSE_SENT):
                    word = load_volatile(source)[0]
                store_words(source, (Uint32(LSE_SENT),))
                lse = bits_float(word)
            top = lse
            for shift in cutlass.range_constexpr(4, -1, -1):
                top = max(top, cute.arch.shuffle_sync_bfly(top, 1 << shift))
            weight = Float32(0)
            if lse > Float32(EMPTY_LSE):
                weight = cute.math.exp2(lse - top, fastmath=True)
            total = warp_sum(weight)
            # As dcp_direct_a2a_lse_reduce: weight * (1 / sum) (IEEE reciprocal).
            inverse = Float32(0)
            if total > 0:
                inverse = div_rn(Float32(1), total)
            scale = weight * inverse
            if lse <= Float32(EMPTY_LSE):
                scale = Float32(-1)  # no value row: neither polled nor re-armed
            if lane < D:
                scales[warp, lane] = scale
        cute.arch.sync_threads()
        if tid < 256:
            r, chunk = tid // 16, tid % 16
            t = t0 + r
            if t < B:
                merged = zeros(8, Float32)
                for s0 in cutlass.range_constexpr(0, D, 8):  # 8 polls in flight
                    sources = [
                        at(box, s, head, t, 8 * (chunk ^ (t % 8)))
                        for s in range(s0, min(D, s0 + 8))
                    ]
                    loads = [load_volatile(source, count=4) for source in sources]
                    for i in cutlass.range_constexpr(len(sources)):
                        w = scales[r, s0 + i]
                        if w >= Float32(0):
                            self.poll_pair(sources[i], loads[i], w, merged)
                col = head * 128 + 8 * chunk
                gates = load_words(at(gate, t, col), count=4, mutable=True)
                words = []
                for e in cutlass.range_constexpr(4):
                    # Padded rows may hold a NaN gate; keep inactive rows zero.
                    lo, hi = Float32(0), Float32(0)
                    if totals[t // self.q] > 0:
                        g = unpack_bf16(gates[e])
                        lo, hi = (
                            Float32(BFloat16(merged[2 * e + k])) * sigmoid(g[k], False)
                            for k in range(2)
                        )
                    words.append(bf16x2(lo, hi))
                store_words(at(output, t, col), tuple(words))
        cute.arch.sync_threads()  # the next tile rewrites the weights

    @cute.jit
    def sent_lse(self, rows_lse, lens, b, gh):
        """The base-2 LSE row (b, gh) sends: EMPTY_LSE for an empty local shard or a
        NaN / infinite LSE (then its partial row is not sent)."""
        value = rows_lse[b, 0, gh] * Float32(self.lse_scale)
        n = lens[b] if cutlass.const_expr(self.per_row) else lens[b // self.q]
        bad = (value != value) | (value <= Float32(EMPTY_LSE))  # NaN, -inf
        if (n <= 0) | bad | (value >= Float32(-EMPTY_LSE)):  # or +inf
            value = Float32(EMPTY_LSE)
        return value

    @cute.jit
    def merge(self, args, latent, scales, wb, tid, tile):
        """The owner's merge of request rows t0.. of this head, then W_UV + gate."""
        box, lbox, gate, totals, output = args[3], args[4], args[9], args[10], args[11]
        D, B = self.size, self.batch
        head, _, _ = cute.arch.block_idx()
        lane, warp = tid % 32, tid // 32
        group, quarter = lane // 4, lane % 4
        t0 = tile * 16
        col = head * 128 + 32 * (warp // 4) + 8 * (warp % 4) + 2 * quarter
        gates = []
        for r in cutlass.range_constexpr(2):
            word = Uint32(0)
            if t0 + group + 8 * r < B:
                word = load_words(
                    at(gate, t0 + group + 8 * r, col), count=1, mutable=True
                )[0]
            gates.append(unpack_bf16(word))
        if t0 + warp < B:  # warp r: the source weights of request t0 + r
            lse = Float32(EMPTY_LSE)
            if lane < D:
                source = at(lbox, lane, t0 + warp, head)
                word = load_volatile(source)[0]
                while word == Uint32(LSE_SENT):
                    word = load_volatile(source)[0]
                store_words(source, (Uint32(LSE_SENT),))
                lse = bits_float(word)
            top = lse
            for shift in cutlass.range_constexpr(4, -1, -1):
                top = max(top, cute.arch.shuffle_sync_bfly(top, 1 << shift))
            weight = Float32(0)
            if lse > Float32(EMPTY_LSE):
                weight = cute.math.exp2(lse - top, fastmath=True)
            total = warp_sum(weight)
            # As dcp_direct_a2a_lse_reduce: weight * (1 / sum) (IEEE reciprocal).
            inverse = Float32(0)
            if total > 0:
                inverse = div_rn(Float32(1), total)
            scale = weight * inverse
            if cutlass.const_expr(self.skip_empty):  # noqa: SIM102 (compile time)
                if lse <= Float32(EMPTY_LSE):
                    scale = Float32(-1)  # no partial row: neither polled nor re-armed
            if lane < D:
                scales[warp, lane] = scale
        cute.arch.sync_threads()
        chunk = tid % 64
        for half in cutlass.range_constexpr(2):  # rows r and r + 8, 16 B chunk
            r = tid // 64 + 8 * half
            merged = zeros(8, Float32)
            if t0 + r < B:
                for s0 in cutlass.range_constexpr(0, D, 8):  # 8 polls in flight
                    sources = [
                        at(box, s, t0 + r, head, 8 * chunk)
                        for s in range(s0, min(D, s0 + 8))
                    ]
                    loads = [load_volatile(source, count=4) for source in sources]
                    for i in cutlass.range_constexpr(len(sources)):
                        w = scales[r, s0 + i]
                        if cutlass.const_expr(self.skip_empty):
                            if w >= Float32(0):
                                self.poll_pair(sources[i], loads[i], w, merged)
                        else:
                            self.poll_pair(sources[i], loads[i], w, merged)
            pairs = tuple(
                pack_pair(merged[2 * e], merged[2 * e + 1], policy=None)
                for e in range(4)
            )
            store_shared_words(latent.iterator + r * 544 + 8 * chunk, pairs)
        cute.arch.sync_threads()
        parts = zeros((2, 4), Float32)  # two accumulation chains
        for c in cutlass.range_constexpr(16):
            x, y = (
                load_words(
                    latent.iterator + r * 544 + 32 * c + 8 * quarter,
                    count=4,
                    shared=True,
                )
                for r in (group, group + 8)
            )
            for s in cutlass.range_constexpr(2):
                a = (x[2 * s], y[2 * s], x[2 * s + 1], y[2 * s + 1])
                b = (wb[c][2 * s], wb[c][2 * s + 1])
                mma_into(parts[s, None], a, b, bf16=True)
        acc = [parts[0, k] + parts[1, k] for k in range(4)]
        for r in cutlass.range_constexpr(2):
            t = t0 + group + 8 * r
            if t < B:
                # Padded rows may hold a NaN gate; keep inactive rows zero.
                lo, hi = Float32(0), Float32(0)
                if totals[t // self.q] > 0:
                    lo, hi = (
                        Float32(BFloat16(acc[2 * r + e])) * sigmoid(gates[r][e], False)
                        for e in range(2)
                    )
                store_words(at(output, t, col), (bf16x2(lo, hi),))

    @cute.jit
    def poll_pair(self, source, words, w, merged):
        """Polls a BF16 partial chunk (8 values) until published, re-arms it and adds
        it, weighted by ``w``, to ``merged``."""
        while not published(words):
            self.backoff()
            words = load_volatile(source, count=4)
        store_words(source, (Uint32(PAIR_SENT),) * 4)
        for e in cutlass.range_constexpr(4):
            lo, hi = unpack_bf16(words[e])
            merged[2 * e] += w * lo
            merged[2 * e + 1] += w * hi


@cache
def compile_value(
    heads,
    size,
    batch,
    cap,
    uv,
    pitch,
    device,
    sms=148,
    q=1,
    dsa_heads=0,
    dsa_bf16=True,
    lse_base_e=True,
    partials="bf16",
    preprojected=False,
    rearm=True,
    ts_grid=0,
):
    """DcpValue of one batch (``q`` rows per request); ``uv``: the (shape, stride,
    dtype) of W_UV, ``pitch``: the projected rows' (the gate follows the 2112 query /
    KV columns). ``dsa_heads``: DSA layout, its padded head count: q mailbox rows
    (BF16 for FlashMLA sparse, else FP8), rows / LSE as the sparse decode returns
    them (``lse_base_e``: natural-log, else base 2) and ``lens`` the sparse decode's
    per-row valid counts. ``partials``: the format of its DcpMailbox. ``ts_grid``:
    see DcpValue (kernel_ov after sparse_mla_ts)."""
    f32, i32 = torch.float32, torch.int32
    w_uv, gate = fake(*uv), fake((batch, heads * 128), (pitch, 1))
    rows = fake((batch, 1, size * heads, 512))
    rows_lse = fake((batch, 1, size * heads), None, f32, 4)
    mailbox = fake((cap, size * heads, 576), None, torch.uint8)
    lens = fake((batch // q,), None, i32, 4)
    value = DcpValue(heads, size, batch, sms, q, partials=partials)
    if dsa_heads:
        rows = fake((batch, 1, size * heads, 512), (dsa_heads * 512, 0, 512, 1))
        rows_lse = fake((batch, 1, size * heads), (dsa_heads, 0, 1), f32, 4)
        row_bytes = 1152 if dsa_bf16 else 576
        mailbox = fake((cap, dsa_heads, row_bytes), None, torch.uint8)
        lens = fake((batch,), None, i32, 4)
        scale = math.log2(math.e) if lse_base_e else 1.0
        chunks = row_bytes // 16
        value = DcpValue(
            heads,
            size,
            batch,
            sms,
            q,
            chunks,
            dsa_heads,
            scale,
            True,
            partials,
            preprojected,
            rearm,
            ts_grid,
        )
    totals = fake((batch // q,), None, i32, 4)
    width = 512
    if partials == PROJECTED:  # the gathered W_UV (DcpBox.gather_heads) and values
        gathered = size * heads
        w_uv, width = fake((gathered, 512, 128), (128 * 512, 1, 512)), 128
        if preprojected:  # the bmm's value rows instead
            w_uv = fake((gathered, batch, 128), (batch * 128, 128, 1))
    lead = (size, heads, cap) if partials == PROJECTED else (size, cap, heads)
    box = fake((*lead, width), None, torch.bfloat16)
    lbox = fake(lead, None, f32, 4)
    peers = fake((size,), None, torch.int64, 8)
    output = fake((batch, heads * 128), (heads * 128, 1))
    args: tuple = (rows, rows_lse, lens, box, lbox, peers, peers, mailbox, w_uv)
    sync = fake((SYNC_WORDS,), None, i32, 4)  # the mailbox parity's (kernel_ov)
    args += (gate, totals, output, Int32(0), Int32(0), sync, make_fake_stream())
    with torch.accelerator.device_index(device):
        return cute.compile(value, *args)
