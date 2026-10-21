# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-rank MLA work sharing for the DEP decode.

DP ranks step in lockstep (every layer's MegaMoE synchronizes them), so each MLA
layer runs at the pace of the rank with the most context. A request's KV prefix is
immutable during decode, so a lightly loaded rank (helper) can keep a replica of a
heavy rank's (owner's) prefix range [0, b) and compute that range's attention; the
owner computes [b, len) and merges the two partial outputs by their LSE, exactly as
a split-KV reduction merges splits.

Everything runs on the device, inside the step's CUDA graph, over symmetric memory
on the EP group:

* each rank publishes its requests (committed length, key = first page id + 1) to
  rank 0 under a per-rank board counter (negated on steps that run the stock
  forward); rank 0 waits a bounded time for boards newer than the ones it last
  planned with, plans the assignments (one new assignment per step, kept while the
  request lives) and pushes the table, tagged with the board counter of every rank
  it planned with, to every rank. A rank takes part only in a table planned with
  its current board (re-checked after copying the table), so a late or stock-path
  rank never desynchronizes the others: its assignments just idle that step;
* owners copy the prefix pages of filling assignments into the helper's replica pool
  (a budget of pages per step) and acknowledge them; an assignment becomes active
  once every page is acknowledged;
* per MLA layer the owner pushes its query rows of active assignments to the helper
  before its own attention (over [b, len)); the helper, after its own attention,
  computes the prefix range (non-causal) and returns O and LSE; the owner merges.
  If a helper's partial does not arrive in time, the owner computes the prefix
  range itself in the merge kernel (slow, but exact), and counts the error.

Flags carry a (table epoch, layer) tag, so stale writes never match.
"""

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from vllm.triton_utils import tl, triton

from .mla_decode import WorkListMla

LATENT, ROPE = 512, 64
# Table fields (16 int64 per assignment): state, owner, helper, replica slot, key,
# prefix tokens b, uid, this step's fill range [from, to), table epoch.
FIELDS = (
    "state",
    "owner",
    "helper",
    "slot",
    "key",
    "b",
    "uid",
    "fill_from",
    "fill_to",
    "epoch",
)
F_STATE = tl.constexpr(0)
F_OWNER = tl.constexpr(1)
F_HELPER = tl.constexpr(2)
F_SLOT = tl.constexpr(3)
F_KEY = tl.constexpr(4)
F_B = tl.constexpr(5)
F_UID = tl.constexpr(6)
F_FROM = tl.constexpr(7)
F_TO = tl.constexpr(8)
F_EPOCH = tl.constexpr(9)
FILLING = tl.constexpr(1)
ACTIVE = tl.constexpr(2)


@triton.jit
def _now():
    return tl.inline_asm_elementwise(
        "mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64, is_pure=False, pack=1
    )


@triton.jit
def _ptr(table, rank, offset, DTYPE: tl.constexpr):
    """``DTYPE*`` to element ``offset`` of ``rank``'s buffer (base addresses in
    ``table``)."""
    base = tl.load(table + rank)
    return (base + offset * DTYPE.primitive_bitwidth // 8).to(tl.pointer_type(DTYPE))


@triton.jit
def _acquire(p):
    return tl.atomic_add(p, 0, sem="acquire", scope="sys")


@triton.jit(do_not_specialize=["bt_stride", "num_reqs", "q", "rank", "absent"])
def _publish_kernel(
    lens_ptr,
    bt_ptr,
    bt_stride,
    num_reqs,
    q,
    ctrl_ptrs,
    pub_ptr,
    rank,
    absent,
    BOARD: tl.constexpr,
    BOARD_EPOCH: tl.constexpr,
    MAXR: tl.constexpr,
):
    """My requests to rank 0's board under my next board counter c (-c when this
    step runs the stock forward: I take no part)."""
    r = tl.arange(0, MAXR)
    m = (r < num_reqs) & (absent == 0)
    lens = tl.load(lens_ptr + r, m, other=0)
    key = tl.load(bt_ptr + r * bt_stride, m & (lens > 0), other=-1) + 1
    committed = tl.maximum(lens - q, 0).to(tl.int64)
    dst = _ptr(ctrl_ptrs, 0, BOARD + rank * MAXR * 2, tl.int64)
    tl.store(dst + 2 * r, committed)
    tl.store(dst + 2 * r + 1, key.to(tl.int64))
    c = tl.load(pub_ptr) + 1
    tl.store(pub_ptr, c)
    tag = tl.where(absent != 0, -c, c)
    tl.debug_barrier()
    tl.atomic_xchg(
        _ptr(ctrl_ptrs, 0, BOARD_EPOCH + rank, tl.int64),
        tag,
        sem="release",
        scope="sys",
    )


@triton.jit(
    do_not_specialize=[
        "world",
        "wait_ns",
        "slots",
        "slot_pages",
        "page",
        "fill_pages",
        "min_gain",
        "min_b",
    ]
)
def _plan_kernel(
    ctrl_ptr,
    ctrl_ptrs,
    ent_ptr,
    seen_ptr,  # [W] last board counter planned with, per rank
    state_ptr,  # [0] next uid, [1] last table epoch
    world,
    wait_ns,
    slots,
    slot_pages,
    page,
    fill_pages,
    min_gain,
    min_b,
    BOARD: tl.constexpr,
    BOARD_EPOCH: tl.constexpr,
    ACKS: tl.constexpr,
    TAB: tl.constexpr,
    TAB_EPOCH: tl.constexpr,
    TAB_PART: tl.constexpr,
    MAXR: tl.constexpr,
    W: tl.constexpr,
    A: tl.constexpr,
):
    """Rank 0: the next table epoch from every board newer than the one last
    planned with (waiting a bounded time for all ranks), pushed to every rank with
    each rank's board counter (TAB_PART; 0: not planned with, negative: stock step).
    A rank takes part only in a table planned with its current board."""
    w = tl.arange(0, W)
    wm = w < world
    seen = tl.load(seen_ptr + w, wm, other=0)
    start = _now()
    tags = tl.atomic_add(
        ctrl_ptr + BOARD_EPOCH + w, 0, mask=wm, sem="acquire", scope="sys"
    )
    got = wm & (tl.abs(tags) > seen)
    arrived = tl.sum(got.to(tl.int32), 0)
    while (arrived < world) & (_now() - start < wait_ns):
        tags = tl.atomic_add(
            ctrl_ptr + BOARD_EPOCH + w, 0, mask=wm, sem="acquire", scope="sys"
        )
        got = wm & (tl.abs(tags) > seen)
        arrived = tl.sum(got.to(tl.int32), 0)
    fresh = got & (tags > 0)
    tl.store(seen_ptr + w, tl.where(got, tl.abs(tags), seen), wm)
    epoch = tl.load(state_ptr + 1) + 1
    tl.store(state_ptr + 1, epoch)
    tl.debug_barrier()

    # Board: [W, MAXR, 2] (committed length, key).
    r = tl.arange(0, MAXR)
    lens2 = tl.load(
        ctrl_ptr + BOARD + (w[:, None] * MAXR + r[None, :]) * 2,
        wm[:, None],
        other=0,
        volatile=True,
    )
    keys2 = tl.load(
        ctrl_ptr + BOARD + (w[:, None] * MAXR + r[None, :]) * 2 + 1,
        wm[:, None],
        other=0,
        volatile=True,
    )
    load = tl.sum(lens2, 1)  # per rank context tokens
    used_slots = tl.zeros([W], tl.int32)

    # Existing assignments: drop finished requests, advance fills.
    for e in range(A):
        st = tl.load(ent_ptr + e * 8 + 0)
        if st != 0:
            owner = tl.load(ent_ptr + e * 8 + 1)
            helper = tl.load(ent_ptr + e * 8 + 2)
            key = tl.load(ent_ptr + e * 8 + 4)
            b = tl.load(ent_ptr + e * 8 + 5)
            uid = tl.load(ent_ptr + e * 8 + 6)
            done = tl.load(ent_ptr + e * 8 + 7)
            owner_fresh = tl.sum(tl.where(w == owner, fresh.to(tl.int32), 0), 0) > 0
            okeys = tl.sum(tl.where(w[:, None] == owner, keys2, 0), 0)
            found = tl.sum((okeys == key).to(tl.int32), 0) > 0
            if owner_fresh & (found == 0):
                tl.store(ent_ptr + e * 8 + 0, 0)
            else:
                ack = tl.atomic_add(ctrl_ptr + ACKS + e, 0, sem="acquire", scope="sys")
                if (ack >> 32) == uid:
                    done = tl.maximum(done, ack & 0xFFFFFFFF)
                    tl.store(ent_ptr + e * 8 + 7, done)
                need = b // page
                if done >= need:
                    tl.store(ent_ptr + e * 8 + 0, ACTIVE)
                load = tl.where(w == owner, load - b, load)
                load = tl.where(w == helper, load + b, load)
                used_slots = tl.where(w == helper, used_slots + 1, used_slots)

    # One new assignment per step: heaviest fresh rank -> lightest fresh rank.
    neg = -(1 << 60)
    heavy_load = tl.where(fresh, load, neg)
    owner = tl.argmax(heavy_load, 0)
    light_load = tl.where(fresh & (used_slots < slots), load, 1 << 60)
    helper = tl.argmin(light_load, 0)
    lo = tl.sum(tl.where(w == owner, load, 0), 0)
    lh = tl.sum(tl.where(w == helper, load, 0), 0)
    free_e = -1
    for e in range(A):
        if (free_e < 0) & (tl.load(ent_ptr + e * 8 + 0) == 0):
            free_e = e
    can = (free_e >= 0) & (owner != helper) & (lo - lh > min_gain) & (lh < (1 << 59))
    if can:
        # Longest request of the owner not assigned yet.
        okeys = tl.sum(tl.where(w[:, None] == owner, keys2, 0), 0)
        olens = tl.sum(tl.where(w[:, None] == owner, lens2, 0), 0)
        taken = tl.zeros([MAXR], tl.int32)
        for e in range(A):
            st = tl.load(ent_ptr + e * 8 + 0)
            eo = tl.load(ent_ptr + e * 8 + 1)
            ek = tl.load(ent_ptr + e * 8 + 4)
            taken = taken | ((st != 0) & (eo == owner) & (okeys == ek)).to(tl.int32)
        cand = tl.where((taken == 0) & (okeys > 0), olens, 0)
        row = tl.argmax(cand, 0)
        clen = tl.sum(tl.where(r == row, cand, 0), 0)
        key = tl.sum(tl.where(r == row, okeys, 0), 0)
        b = tl.minimum(tl.minimum(clen - page, (lo - lh) // 2), slot_pages * page)
        b = (b // page) * page
        if b >= min_b:
            # First free slot of the helper.
            slot_used = tl.zeros([8], tl.int32)
            sidx = tl.arange(0, 8)
            for e in range(A):
                st = tl.load(ent_ptr + e * 8 + 0)
                eh = tl.load(ent_ptr + e * 8 + 2)
                es = tl.load(ent_ptr + e * 8 + 3)
                slot_used = slot_used | ((st != 0) & (eh == helper) & (sidx == es)).to(
                    tl.int32
                )
            slot = tl.argmin(tl.where(sidx < slots, slot_used, 2), 0)
            uid = tl.load(state_ptr + 0) + 1
            tl.store(state_ptr + 0, uid)
            tl.store(ent_ptr + free_e * 8 + 0, FILLING)
            tl.store(ent_ptr + free_e * 8 + 1, owner.to(tl.int64))
            tl.store(ent_ptr + free_e * 8 + 2, helper.to(tl.int64))
            tl.store(ent_ptr + free_e * 8 + 3, slot.to(tl.int64))
            tl.store(ent_ptr + free_e * 8 + 4, key)
            tl.store(ent_ptr + free_e * 8 + 5, b)
            tl.store(ent_ptr + free_e * 8 + 6, uid)
            tl.store(ent_ptr + free_e * 8 + 7, 0)
    tl.debug_barrier()

    # Push the table: an assignment takes part this step only if both ranks do.
    for e in range(A):
        st = tl.load(ent_ptr + e * 8 + 0)
        eo = tl.load(ent_ptr + e * 8 + 1)
        eh = tl.load(ent_ptr + e * 8 + 2)
        ok = st != 0
        ok = ok & (tl.sum(tl.where(w == eo, fresh.to(tl.int32), 0), 0) > 0)
        ok = ok & (tl.sum(tl.where(w == eh, fresh.to(tl.int32), 0), 0) > 0)
        done = tl.load(ent_ptr + e * 8 + 7)
        eb = tl.load(ent_ptr + e * 8 + 5)
        fill_to = tl.minimum(done + fill_pages, eb // page)
        f = tl.arange(0, 16)
        vals = tl.zeros([16], tl.int64)
        vals = tl.where(f == F_STATE, tl.where(ok, st, 0), vals)
        vals = tl.where(f == F_OWNER, eo, vals)
        vals = tl.where(f == F_HELPER, eh, vals)
        vals = tl.where(f == F_SLOT, tl.load(ent_ptr + e * 8 + 3), vals)
        vals = tl.where(f == F_KEY, tl.load(ent_ptr + e * 8 + 4), vals)
        vals = tl.where(f == F_B, eb, vals)
        vals = tl.where(f == F_UID, tl.load(ent_ptr + e * 8 + 6), vals)
        vals = tl.where(f == F_FROM, done, vals)
        vals = tl.where(f == F_TO, fill_to, vals)
        vals = tl.where(f == F_EPOCH, epoch, vals)
        for dst in range(world):
            tl.store(_ptr(ctrl_ptrs, dst, TAB + e * 16, tl.int64) + f, vals, f < 16)
    part = tl.where(got, tags, 0)
    for dst in range(world):
        tl.store(_ptr(ctrl_ptrs, dst, TAB_PART, tl.int64) + w, part, wm)
    tl.debug_barrier()
    for dst in range(world):
        tl.atomic_xchg(
            _ptr(ctrl_ptrs, dst, TAB_EPOCH, tl.int64), epoch, sem="release", scope="sys"
        )


@triton.jit(do_not_specialize=["rank", "wait_ns"])
def _snapshot_kernel(
    ctrl_ptr,
    snap_ptr,
    last_ptr,
    pub_ptr,
    stats_ptr,
    rank,
    wait_ns,
    TAB: tl.constexpr,
    TAB_EPOCH: tl.constexpr,
    TAB_PART: tl.constexpr,
    A: tl.constexpr,
):
    """This step's table: the first epoch newer than the last one I took that was
    planned with my current board (bounded wait). I take part if it was planned with
    my board as a participant and was not replaced while I copied it; otherwise this
    step has no assignments for me."""
    c = tl.load(pub_ptr)
    last = tl.load(last_ptr)
    start = _now()
    epoch = _acquire(ctrl_ptr + TAB_EPOCH)
    part = tl.load(ctrl_ptr + TAB_PART + rank, volatile=True)
    done = (epoch > last) & (tl.abs(part) == c)
    while (done == 0) & (_now() - start < wait_ns):
        epoch = _acquire(ctrl_ptr + TAB_EPOCH)
        part = tl.load(ctrl_ptr + TAB_PART + rank, volatile=True)
        done = (epoch > last) & (tl.abs(part) == c)
    i = tl.arange(0, A * 16)
    tab = tl.load(ctrl_ptr + TAB + i, volatile=True)
    tl.debug_barrier()
    again = _acquire(ctrl_ptr + TAB_EPOCH)
    use = done & (part == c) & (again == epoch)
    tl.store(last_ptr, tl.where(done, epoch, last))
    tl.atomic_add(stats_ptr + 0, 1)
    tl.atomic_add(stats_ptr + 1, use.to(tl.int64))
    keep = use | ((i % 16) != F_STATE)
    tl.store(snap_ptr + i, tl.where(keep, tab, 0))


@triton.jit(do_not_specialize=["rank", "slot_pages", "page"])
def _entry_meta_kernel(
    snap_ptr,
    rank,
    row_of_ptr,
    helper_lens_ptr,
    helper_bt_ptr,
    slot_pages,
    page,
    A: tl.constexpr,
    P: tl.constexpr,
):
    """Helper-side lengths and replica page tables; reset the owner row map."""
    e = tl.program_id(0)
    st = tl.load(snap_ptr + e * 16 + F_STATE)
    helper = tl.load(snap_ptr + e * 16 + F_HELPER)
    slot = tl.load(snap_ptr + e * 16 + F_SLOT)
    b = tl.load(snap_ptr + e * 16 + F_B)
    mine = (st == ACTIVE) & (helper == rank)
    tl.store(helper_lens_ptr + e, tl.where(mine, b, 0).to(tl.int32))
    p = tl.arange(0, P)
    tl.store(
        helper_bt_ptr + e * P + p, (slot * slot_pages + p).to(tl.int32), p < slot_pages
    )
    tl.store(row_of_ptr + e, -1)


@triton.jit(do_not_specialize=["bt_stride", "rank", "page", "pages"])
def _own_meta_kernel(
    lens_ptr,
    bt_ptr,
    bt_stride,
    own_lens_ptr,
    own_bt_ptr,
    snap_ptr,
    row_of_ptr,
    rank,
    page,
    pages,
    A: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Owner-side lengths and page tables: active assignments drop their prefix."""
    r = tl.program_id(0)
    n = tl.load(lens_ptr + r)
    key = tl.load(bt_ptr + r * bt_stride) + 1
    shift = 0
    for e in range(A):
        st = tl.load(snap_ptr + e * 16 + F_STATE)
        owner = tl.load(snap_ptr + e * 16 + F_OWNER)
        ekey = tl.load(snap_ptr + e * 16 + F_KEY)
        b = tl.load(snap_ptr + e * 16 + F_B)
        match = (st != 0) & (owner == rank) & (ekey == key) & (n > 0)
        if match:
            tl.store(row_of_ptr + e, r)
            if st == ACTIVE:
                shift = (b // page).to(tl.int32)
    tl.store(own_lens_ptr + r, n - shift * page)
    for c in range(0, pages, BLOCK):
        j = c + tl.arange(0, BLOCK)
        src = tl.load(bt_ptr + r * bt_stride + j + shift, (j + shift) < pages, other=0)
        tl.store(own_bt_ptr + r * pages + j, src, j < pages)


@triton.jit(
    do_not_specialize=[
        "bt_stride",
        "rank",
        "num_layers",
        "layer_pages",
        "slot_pages",
        "page_bytes",
    ]
)
def _fill_kernel(
    snap_ptr,
    row_of_ptr,
    bt_ptr,
    bt_stride,
    cache_ptrs,
    cache_strides,
    pool_ptrs,
    ctrl_ptrs,
    counter_ptr,
    rank,
    num_layers,
    layer_pages,  # replica pages per layer (slots * slot_pages)
    slot_pages,
    page_bytes,
    FILL: tl.constexpr,
    A: tl.constexpr,
    ACKS: tl.constexpr,
    CHUNK: tl.constexpr,
):
    """Copy this step's prefix pages of my filling assignments to the helpers'
    replica pools (all MLA layers); the last program acknowledges to rank 0."""
    e = tl.program_id(0)
    j = tl.program_id(1)
    st = tl.load(snap_ptr + e * 16 + F_STATE)
    owner = tl.load(snap_ptr + e * 16 + F_OWNER)
    row = tl.load(row_of_ptr + e)
    lo = tl.load(snap_ptr + e * 16 + F_FROM)
    hi = tl.load(snap_ptr + e * 16 + F_TO)
    p = lo + j
    if (st == FILLING) & (owner == rank) & (row >= 0) & (p < hi):
        helper = tl.load(snap_ptr + e * 16 + F_HELPER)
        slot = tl.load(snap_ptr + e * 16 + F_SLOT)
        src_page = tl.load(bt_ptr + row * bt_stride + p).to(tl.int64)
        dst_page = slot * slot_pages + p
        o = tl.arange(0, CHUNK)
        for layer in range(num_layers):
            src = (
                tl.load(cache_ptrs + layer) + src_page * tl.load(cache_strides + layer)
            ).to(tl.pointer_type(tl.int32))
            dst = _ptr(
                pool_ptrs,
                helper,
                (layer * layer_pages + dst_page) * page_bytes // 4,
                tl.int32,
            )
            for c in range(0, page_bytes // 4, CHUNK):
                v = tl.load(src + c + o, (c + o) < page_bytes // 4)
                tl.store(dst + c + o, v, (c + o) < page_bytes // 4)
    tl.debug_barrier()
    total = tl.num_programs(0) * tl.num_programs(1)
    done = tl.atomic_add(counter_ptr, 1, sem="acq_rel", scope="gpu")
    if done == total - 1:
        tl.store(counter_ptr, 0)
        for k in range(A):
            kst = tl.load(snap_ptr + k * 16 + F_STATE)
            kown = tl.load(snap_ptr + k * 16 + F_OWNER)
            krow = tl.load(row_of_ptr + k)
            if (kst == FILLING) & (kown == rank) & (krow >= 0):
                uid = tl.load(snap_ptr + k * 16 + F_UID)
                to = tl.load(snap_ptr + k * 16 + F_TO)
                tl.atomic_xchg(
                    _ptr(ctrl_ptrs, 0, ACKS + k, tl.int64),
                    (uid << 32) | to,
                    sem="release",
                    scope="sys",
                )


@triton.jit(
    do_not_specialize=[
        "src_row_bytes",
        "lse_row_bytes",
        "dst_entry_bytes",
        "rank",
        "layer",
        "nchunks",
    ]
)
def _send_kernel(
    snap_ptr,
    row_of_ptr,
    ok_ptr,
    src_ptr,
    src_row_bytes,
    lse_ptr,
    lse_row_bytes,
    dst_ptrs,
    dst_entry_bytes,
    ctrl_ptrs,
    rank,
    layer,
    nchunks,
    SENDER: tl.constexpr,  # 1: owner sends queries, 2: helper sends results
    FLAG: tl.constexpr,
    A: tl.constexpr,
    CHUNK: tl.constexpr,
    NPROG: tl.constexpr,
):
    """Copy each entry's rows to its peer's slot in CHUNK-word pieces, each raising
    its own flag at the peer (persistent grid over (entry, chunk) items)."""
    tag = tl.load(snap_ptr + F_EPOCH) * 64 + layer
    words = src_row_bytes // 4
    o = tl.arange(0, CHUNK)
    for it in range(tl.program_id(0), A * nchunks, NPROG):
        e = it // nchunks
        c = it - e * nchunks
        st = tl.load(snap_ptr + e * 16 + F_STATE)
        owner = tl.load(snap_ptr + e * 16 + F_OWNER)
        helper = tl.load(snap_ptr + e * 16 + F_HELPER)
        row = tl.load(row_of_ptr + e)
        if SENDER == 1:
            go = (st == ACTIVE) & (owner == rank) & (row >= 0)
            peer = helper
            src_row = row
        else:
            go = (st == ACTIVE) & (helper == rank) & (tl.load(ok_ptr + e) != 0)
            peer = owner
            src_row = e
        if go:
            src = (
                src_ptr.to(tl.pointer_type(tl.int8))
                + src_row.to(tl.int64) * src_row_bytes
            ).to(tl.pointer_type(tl.int32))
            dst = _ptr(dst_ptrs, peer, e * dst_entry_bytes // 4, tl.int32)
            k = c * CHUNK + o
            tl.store(dst + k, tl.load(src + k, k < words), k < words)
            if SENDER == 2:
                if c == 0:
                    lw = lse_row_bytes // 4
                    lo = tl.arange(0, 1024)
                    lsrc = (
                        lse_ptr.to(tl.pointer_type(tl.int8))
                        + src_row.to(tl.int64) * lse_row_bytes
                    ).to(tl.pointer_type(tl.int32))
                    for m in range(0, lw, 1024):
                        lv = tl.load(lsrc + m + lo, (m + lo) < lw)
                        tl.store(dst + words + m + lo, lv, (m + lo) < lw)
            tl.debug_barrier()
            tl.atomic_xchg(
                _ptr(ctrl_ptrs, peer, FLAG + e * 32 + c, tl.int64),
                tag,
                sem="release",
                scope="sys",
            )


@triton.jit
def _all_flags(ctrl_ptr, base, nchunks, tag, start, wait_ns):
    """Spin (bounded) until all ``nchunks`` flags at ``base`` hold ``tag``."""
    c = tl.arange(0, 32)
    cm = c < nchunks
    v = tl.atomic_add(ctrl_ptr + base + c, 0, mask=cm, sem="acquire", scope="sys")
    n = tl.sum((cm & (v == tag)).to(tl.int32), 0)
    while (n < nchunks) & (_now() - start < wait_ns):
        v = tl.atomic_add(ctrl_ptr + base + c, 0, mask=cm, sem="acquire", scope="sys")
        n = tl.sum((cm & (v == tag)).to(tl.int32), 0)
    return n == nchunks


@triton.jit(do_not_specialize=["rank", "layer", "wait_ns", "nchunks"])
def _wait_queries_kernel(
    stats_ptr,
    snap_ptr,
    ctrl_ptr,
    ok_ptr,
    rank,
    layer,
    wait_ns,
    nchunks,
    FLAG: tl.constexpr,
):
    tag = tl.load(snap_ptr + F_EPOCH) * 64 + layer
    e = tl.program_id(0)
    st = tl.load(snap_ptr + e * 16 + F_STATE)
    helper = tl.load(snap_ptr + e * 16 + F_HELPER)
    ok = 0
    if (st == ACTIVE) & (helper == rank):
        ok = _all_flags(ctrl_ptr, FLAG + e * 32, nchunks, tag, _now(), wait_ns).to(
            tl.int32
        )
        tl.atomic_add(stats_ptr + 2, ok.to(tl.int64))
    tl.store(ok_ptr + e, ok)


@triton.jit
def _prefix_attention(
    query_ptr,
    qrow,
    i,
    im,
    bt_ptr,
    bt_stride,
    cache_ptr,
    page_stride,
    b,
    scale_log2,
    output_scale,
    LAT: tl.constexpr,
    ROPE_D: tl.constexpr,
    PAGE: tl.constexpr,
    ROWS: tl.constexpr,
    TOK: tl.constexpr,
):
    """Partial attention (output, log2-domain LSE) of query rows ``qrow * rows + i``
    over the owner's own prefix tokens [0, b): the fallback when a helper's partial
    does not arrive (slow, only on that error path)."""
    d = tl.arange(0, LAT)
    rd = tl.arange(0, ROPE_D)
    qbase = query_ptr + (qrow + i).to(tl.int64) * (LAT + ROPE_D)
    q_lat = tl.load(qbase[:, None] + d[None, :], im[:, None], other=0.0).to(tl.bfloat16)
    q_rope = tl.load(qbase[:, None] + LAT + rd[None, :], im[:, None], other=0.0).to(
        tl.bfloat16
    )
    m = tl.full([ROWS], -float("inf"), tl.float32)
    total = tl.zeros([ROWS], tl.float32)
    acc = tl.zeros([ROWS, LAT], tl.float32)
    for t0 in range(0, b, TOK):
        t = t0 + tl.arange(0, TOK)
        tm = t < b
        pg = tl.load(bt_ptr + (t // PAGE), tm, other=0).to(tl.int64)
        kbase = cache_ptr + pg * page_stride + (t % PAGE).to(tl.int64) * (LAT + ROPE_D)
        k_lat = tl.load(kbase[:, None] + d[None, :], tm[:, None], other=0.0).to(
            tl.bfloat16
        )
        k_rope = tl.load(kbase[:, None] + LAT + rd[None, :], tm[:, None], other=0.0).to(
            tl.bfloat16
        )
        s = tl.dot(q_lat, tl.trans(k_lat)) + tl.dot(q_rope, tl.trans(k_rope))
        s = tl.where(tm[None, :], s, -float("inf"))
        m_new = tl.maximum(m, tl.max(s, 1))
        p = tl.exp2((s - m_new[:, None]) * scale_log2)
        alpha = tl.exp2((m - m_new) * scale_log2)
        total = total * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), k_lat)
        m = m_new
    out = acc / total[:, None] * output_scale
    lse = tl.log2(total) + scale_log2 * m
    return out, lse


@triton.jit(
    do_not_specialize=[
        "entry_bytes",
        "rank",
        "layer",
        "wait_ns",
        "rows",
        "nchunks",
        "bt_stride",
        "page_stride",
        "scale_log2",
        "output_scale",
    ]
)
def _merge_kernel(
    stats_ptr,
    snap_ptr,
    row_of_ptr,
    ctrl_ptr,
    results_ptr,
    entry_bytes,
    out_ptr,
    lse_ptr,
    err_ptr,
    query_ptr,
    bt_ptr,
    bt_stride,
    cache_ptr,
    page_stride,
    scale_log2,
    output_scale,
    rank,
    layer,
    wait_ns,
    rows,  # q * heads
    nchunks,
    FLAG: tl.constexpr,
    LAT: tl.constexpr,
    ROPE_D: tl.constexpr,
    PAGE: tl.constexpr,
    ROWS: tl.constexpr,
):
    """Merge each active assignment's helper partial into my rows (by LSE). If the
    partial does not arrive in time, compute the prefix range here instead, so the
    output stays exact (counted in ``err_ptr``)."""
    tag = tl.load(snap_ptr + F_EPOCH) * 64 + layer
    e = tl.program_id(0)
    chunk = tl.program_id(1)
    st = tl.load(snap_ptr + e * 16 + F_STATE)
    owner = tl.load(snap_ptr + e * 16 + F_OWNER)
    row = tl.load(row_of_ptr + e)
    if (st == ACTIVE) & (owner == rank) & (row >= 0):
        i = chunk * ROWS + tl.arange(0, ROWS)
        im = i < rows
        d = tl.arange(0, LAT)
        if _all_flags(ctrl_ptr, FLAG + e * 32, nchunks, tag, _now(), wait_ns):
            if chunk == 0:
                tl.atomic_add(stats_ptr + 3, 1)
            base = results_ptr + e.to(tl.int64) * entry_bytes
            ox = tl.load(
                base.to(tl.pointer_type(tl.bfloat16)) + i[:, None] * LAT + d[None, :],
                im[:, None],
                volatile=True,
            ).to(tl.float32)
            lx = tl.load(
                (base + rows * LAT * 2).to(tl.pointer_type(tl.float32)) + i,
                im,
                volatile=True,
            )
        else:
            if chunk == 0:
                tl.atomic_add(err_ptr, 1)
            b = tl.load(snap_ptr + e * 16 + F_B).to(tl.int32)
            ox, lx = _prefix_attention(
                query_ptr,
                row.to(tl.int64) * rows,
                i,
                im,
                bt_ptr + row.to(tl.int64) * bt_stride,
                bt_stride,
                cache_ptr,
                page_stride,
                b,
                scale_log2,
                output_scale,
                LAT=LAT,
                ROPE_D=ROPE_D,
                PAGE=PAGE,
                ROWS=ROWS,
                TOK=64,
            )
            ox = ox.to(tl.bfloat16).to(tl.float32)
        orow = row.to(tl.int64) * rows + i
        oo = tl.load(out_ptr + orow[:, None] * LAT + d[None, :], im[:, None]).to(
            tl.float32
        )
        lo = tl.load(lse_ptr + orow, im)
        mx = tl.maximum(lo, lx)
        wo = tl.exp2(lo - mx)
        wx = tl.exp2(lx - mx)
        merged = (oo * wo[:, None] + ox * wx[:, None]) / (wo + wx)[:, None]
        tl.store(
            out_ptr + orow[:, None] * LAT + d[None, :],
            merged.to(tl.bfloat16),
            im[:, None],
        )


class MlaShare:
    """Cross-rank MLA work sharing over the EP ``group`` (see the module docstring).

    ``core`` is the rank's own ``WorkListMla``; this object owns the helper core, the
    symmetric buffers and the replica pool."""

    def __init__(
        self,
        group,
        heads: int,
        q: int,
        max_tokens: int,
        num_layers: int,
        page_size: int,
        device: torch.device,
        *,
        slots: int = 2,
        slot_tokens: int = 65536,
        entries: int = 16,
        fill_pages: int = 64,
        min_gain_tokens: int = 16384,
        min_prefix_tokens: int = 4096,
        board_wait_us: float = 20000.0,
        table_wait_us: float = 30000.0,
        flag_wait_us: float = 20000.0,
    ):
        self.group = group
        self.rank, self.world = dist.get_rank(group), dist.get_world_size(group)
        self.heads, self.q, self.layers, self.page = heads, q, num_layers, page_size
        self.device = device
        self.slots, self.slot_pages = slots, slot_tokens // page_size
        self.entries, self.fill_pages = entries, fill_pages
        self.min_gain, self.min_b = min_gain_tokens, min_prefix_tokens
        self.board_wait = int(board_wait_us * 1000)
        self.table_wait = int(table_wait_us * 1000)
        self.flag_wait = int(flag_wait_us * 1000)
        self.maxr = triton.next_power_of_2(max(16, -(-max_tokens // q)))
        self.W = triton.next_power_of_2(max(2, self.world))
        assert entries <= 64 and slots <= 8 and entries & (entries - 1) == 0
        A = entries
        # Control block (int64): board, board epochs, acks, table, table epoch,
        # query flags, result flags.
        self.BOARD = 0
        self.BOARD_EPOCH = self.BOARD + self.W * self.maxr * 2
        self.ACKS = self.BOARD_EPOCH + self.W
        self.TAB = self.ACKS + A
        self.TAB_EPOCH = self.TAB + A * 16
        self.TAB_PART = self.TAB_EPOCH + 8
        self.FLAG_Q = self.TAB_PART + self.W
        self.FLAG_O = self.FLAG_Q + A * 32
        ctrl_len = self.FLAG_O + A * 32
        name = group.group_name
        self.ctrl = symm_mem.empty(ctrl_len, dtype=torch.int64, device=device)
        self.ctrl.zero_()
        self.ctrl_h = symm_mem.rendezvous(self.ctrl, name)
        self.qbytes = q * heads * (LATENT + ROPE)
        self.obytes = q * heads * LATENT * 2 + q * heads * 4
        self.inbox = symm_mem.empty(A * self.qbytes, dtype=torch.uint8, device=device)
        self.inbox_h = symm_mem.rendezvous(self.inbox, name)
        self.results = symm_mem.empty(A * self.obytes, dtype=torch.uint8, device=device)
        self.results_h = symm_mem.rendezvous(self.results, name)
        self.page_bytes = page_size * (LATENT + ROPE)
        self.layer_pages = slots * self.slot_pages
        self.pool = symm_mem.empty(
            num_layers * self.layer_pages * self.page_bytes,
            dtype=torch.uint8,
            device=device,
        )
        self.pool_h = symm_mem.rendezvous(self.pool, name)

        def ptrs(handle):
            return torch.tensor(handle.buffer_ptrs, dtype=torch.int64, device=device)

        self.ctrl_ptrs = ptrs(self.ctrl_h)
        self.inbox_ptrs = ptrs(self.inbox_h)
        self.results_ptrs = ptrs(self.results_h)
        self.pool_ptrs = ptrs(self.pool_h)
        dist.barrier(group)
        # Local state.
        i64 = dict(dtype=torch.int64, device=device)
        i32 = dict(dtype=torch.int32, device=device)
        self.ent = torch.zeros(A * 8, **i64)
        self.plan_state = torch.zeros(2, **i64)
        self.last_tab = torch.zeros(1, **i64)  # last table epoch taken
        self.pub = torch.zeros(1, **i64)  # my board counter
        self.seen = torch.zeros(self.W, **i64)  # rank 0: board counters planned with
        self.snap = torch.zeros(A * 16, **i64)
        self.row_of = torch.full((A,), -1, **i32)
        self.ok = torch.zeros(A, **i32)
        self.errors = torch.zeros(1, **i32)
        # steps, steps taking part, entries served (helper), entries merged (owner)
        self.stats = torch.zeros(4, **i64)
        self._none_lens = torch.zeros(1, **i32)
        self._none_bt = torch.zeros(1, 1, **i32)
        self._debug_report()
        self.counters = torch.zeros(8, **i32)
        self.P = triton.next_power_of_2(self.slot_pages)
        self.helper_lens = torch.zeros(A, **i32)
        self.helper_bt = torch.zeros(A, self.P, **i32)
        self.cache_ptrs = torch.zeros(num_layers, **i64)
        self.cache_strides = torch.zeros(num_layers, **i64)
        # Helper attention over the replica pool (non-causal prefix ranges).
        self.helper_core = WorkListMla(
            heads, q, A * q, device, causal=False, skip_empty=True, page_size=page_size
        )
        self.helper_out = torch.empty(
            A * q, heads, LATENT, dtype=torch.bfloat16, device=device
        )
        self.helper_lse = torch.empty(A, q, heads, dtype=torch.float32, device=device)
        self.own_lse = torch.empty(
            -(-max_tokens // q), q, heads, dtype=torch.float32, device=device
        )
        self.own_lens = None
        self.own_bt = None

    def _debug_report(self) -> None:
        """DEP_MLA_SHARE_DEBUG=1: log the counters every 10 s (copied to pinned host
        memory inside the step, so the log thread never touches the device)."""
        import os

        self.stats_host = None
        if not os.environ.get("DEP_MLA_SHARE_DEBUG"):
            return
        import threading
        import time

        from vllm.logger import init_logger

        logger = init_logger(__name__)
        self.stats_host = torch.zeros(8, dtype=torch.int64, pin_memory=True)

        def report():
            last = None
            while True:
                time.sleep(10)
                values = self.stats_host.tolist()
                if values != last:
                    logger.info(
                        "DEP MLA share rank %d: steps %d, taking part %d, served %d, "
                        "merged %d, errors %d",
                        self.rank,
                        *values[:5],
                    )
                    last = values

        threading.Thread(target=report, daemon=True).start()

    def bind_caches(self, caches: list[torch.Tensor]) -> None:
        """The MLA layers' paged FP8 caches [pages, page, 576] in layer order (each
        page contiguous; pages may be strided)."""
        assert len(caches) == self.layers
        for c in caches:
            if c.shape[1] != self.page or c.element_size() != 1:
                raise ValueError(
                    f"MLA sharing expects {self.page}-token FP8 pages, "
                    f"got {tuple(c.shape)}"
                )
            assert c.stride(2) == 1 and c.stride(1) == LATENT + ROPE
        self.cache_ptrs.copy_(
            torch.tensor([c.data_ptr() for c in caches], dtype=torch.int64)
        )
        self.cache_strides.copy_(
            torch.tensor([c.stride(0) for c in caches], dtype=torch.int64)
        )

    def _sync(self, seq_lens, block_table, absent: int) -> None:
        """Publish my board, plan (rank 0) and take this step's table."""
        A = self.entries
        _publish_kernel[(1,)](
            seq_lens,
            block_table,
            block_table.stride(0),
            block_table.shape[0],
            self.q,
            self.ctrl_ptrs,
            self.pub,
            self.rank,
            absent,
            BOARD=self.BOARD,
            BOARD_EPOCH=self.BOARD_EPOCH,
            MAXR=self.maxr,
        )
        if self.rank == 0:
            _plan_kernel[(1,)](
                self.ctrl,
                self.ctrl_ptrs,
                self.ent,
                self.seen,
                self.plan_state,
                self.world,
                self.board_wait,
                self.slots,
                self.slot_pages,
                self.page,
                self.fill_pages,
                self.min_gain,
                self.min_b,
                BOARD=self.BOARD,
                BOARD_EPOCH=self.BOARD_EPOCH,
                ACKS=self.ACKS,
                TAB=self.TAB,
                TAB_EPOCH=self.TAB_EPOCH,
                TAB_PART=self.TAB_PART,
                MAXR=self.maxr,
                W=self.W,
                A=A,
            )
        _snapshot_kernel[(1,)](
            self.ctrl,
            self.snap,
            self.last_tab,
            self.pub,
            self.stats,
            self.rank,
            self.table_wait,
            TAB=self.TAB,
            TAB_EPOCH=self.TAB_EPOCH,
            TAB_PART=self.TAB_PART,
            A=A,
        )

    def absent_step(self) -> None:
        """A forward that runs the stock path: keep the protocol in step (rank 0
        still plans for the others) without taking part."""
        self._sync(self._none_lens, self._none_bt, 1)

    def begin_step(self, seq_lens: torch.Tensor, block_table: torch.Tensor):
        """Publish, plan (rank 0), take the table, fill replicas; returns this step's
        own (lengths, page table) for the rank's own attention."""
        B, pages = block_table.shape
        A = self.entries
        self.full_bt = block_table  # the owner's whole page table (merge fallback)
        self._sync(seq_lens, block_table, 0)
        _entry_meta_kernel[(A,)](
            self.snap,
            self.rank,
            self.row_of,
            self.helper_lens,
            self.helper_bt,
            self.slot_pages,
            self.page,
            A=A,
            P=self.P,
        )
        own_lens = torch.empty_like(seq_lens)
        own_bt = torch.empty((B, pages), dtype=block_table.dtype, device=self.device)
        _own_meta_kernel[(B,)](
            seq_lens,
            block_table,
            block_table.stride(0),
            own_lens,
            own_bt,
            self.snap,
            self.row_of,
            self.rank,
            self.page,
            pages,
            A=A,
            BLOCK=1024,
        )
        _fill_kernel[(A, self.fill_pages)](
            self.snap,
            self.row_of,
            block_table,
            block_table.stride(0),
            self.cache_ptrs,
            self.cache_strides,
            self.pool_ptrs,
            self.ctrl_ptrs,
            self.counters[0:],
            self.rank,
            self.layers,
            self.layer_pages,
            self.slot_pages,
            self.page_bytes,
            FILL=self.fill_pages,
            A=A,
            ACKS=self.ACKS,
            CHUNK=1024,
        )
        self.helper_core.plan(self.helper_lens)
        self.own_lens, self.own_bt = own_lens, own_bt
        if self.stats_host is not None:
            self.stats_host[:4].copy_(self.stats, non_blocking=True)
            self.stats_host[4:5].copy_(self.errors, non_blocking=True)
        return own_lens, own_bt

    def attend(self, layer, core, query, cache, softmax_scale, output_scale):
        """This layer's latent attention [T, H, L] with cross-rank sharing; ``core``
        is planned with ``begin_step``'s own lengths."""
        A = self.entries
        T = query.shape[0]
        # Owner: queries of active assignments to their helpers.
        _send_dyn(
            self,
            1,
            query,
            self.qbytes,
            None,
            0,
            self.inbox_ptrs,
            self.qbytes,
            self.FLAG_Q,
            layer,
        )
        B = T // self.q
        lse_own = self.own_lse[:B]
        out = core.decode(
            query,
            cache,
            self.own_bt,
            self.own_lens,
            softmax_scale,
            output_scale,
            lse=lse_own,
        )
        # Helper: wait for my assignments' queries, compute their prefix ranges.
        _wait_dyn(self, layer)
        replica = self.pool.view(
            self.layers, self.layer_pages, self.page, LATENT + ROPE
        )[layer]
        self.helper_core.decode(
            self.inbox.view(torch.float8_e4m3fn).view(
                A * self.q, self.heads, LATENT + ROPE
            ),
            replica,
            self.helper_bt,
            self.helper_lens,
            softmax_scale,
            output_scale,
            out=self.helper_out,
            lse=self.helper_lse,
        )
        _send_dyn(
            self,
            2,
            self.helper_out,
            self.q * self.heads * LATENT * 2,
            self.helper_lse,
            self.q * self.heads * 4,
            self.results_ptrs,
            self.obytes,
            self.FLAG_O,
            layer,
        )
        # Owner: merge the helpers' partials.
        rows = self.q * self.heads
        _merge_kernel[(A, triton.cdiv(rows, 16))](
            self.stats,
            self.snap,
            self.row_of,
            self.ctrl,
            self.results,
            self.obytes,
            out,
            lse_own,
            self.errors,
            query.view(torch.float8_e4m3fn),
            self.full_bt,
            self.full_bt.stride(0),
            cache.view(torch.float8_e4m3fn),
            cache.stride(0),
            softmax_scale * 1.4426950408889634,
            output_scale,
            self.rank,
            layer,
            self.flag_wait,
            rows,
            _chunks(self.q * self.heads * LATENT * 2),
            FLAG=self.FLAG_O,
            LAT=LATENT,
            ROPE_D=ROPE,
            PAGE=self.page,
            ROWS=16,
        )
        return out


CHUNK_WORDS = 4096  # 16 KiB pieces, one flag each (at most 32 per entry)


def _chunks(nbytes: int) -> int:
    n = -(-nbytes // (CHUNK_WORDS * 4))
    assert n <= 32
    return n


def _send_dyn(
    share,
    sender,
    src,
    row_bytes,
    lse,
    lse_row_bytes,
    dst_ptrs,
    entry_bytes,
    flag,
    layer,
):
    _send_kernel[(128,)](
        share.snap,
        share.row_of,
        share.ok,
        src,
        row_bytes,
        lse if lse is not None else src,
        lse_row_bytes,
        dst_ptrs,
        entry_bytes,
        share.ctrl_ptrs,
        share.rank,
        layer,
        _chunks(row_bytes),
        SENDER=sender,
        FLAG=flag,
        A=share.entries,
        CHUNK=CHUNK_WORDS,
        NPROG=128,
    )


def _wait_dyn(share, layer):
    _wait_queries_kernel[(share.entries,)](
        share.stats,
        share.snap,
        share.ctrl,
        share.ok,
        share.rank,
        layer,
        share.flag_wait,
        _chunks(share.qbytes),
        FLAG=share.FLAG_Q,
    )
