# Copyright (c) 2026 KDA Team
# SPDX-License-Identifier: MIT
# Adapted from humanfia/kda-for-kda-release; see licenses/LICENSE.kda-for-kda.

"""Persistent KDA scheduling and compilation, with exact FP32 state handoffs."""

import heapq
import hashlib
from pathlib import Path

import cuda.bindings.driver as cuda_driver
import cutlass
import torch
from cutlass import cute
from cutlass.cute.runtime import from_dlpack

from . import kernel as _k


def build_and_load_cute_dsl_kernel(module_name, kernel_name, compile_fn, extra_key_files=()):
    return compile_fn()

D = 128
C32 = 32
LOG2E = 1.4426950408889634

# Device compilation controls and caches.
_TPROBE = 0
# Dev A/B constexpr overrides (benchmarks only; empty in production).
_DEV: dict = {}
_COMPILED: dict = {}
_DUMMY_EXP: dict = {}
_DUMMY_MID: dict = {}
_DUMMY_TPR: dict = {}
LAST_TPROBE: list = [None]

_SMS: dict = {}


def _sm_count(dev: torch.device) -> int:
    n = _SMS.get(dev.index)
    if n is None:
        n = torch.cuda.get_device_properties(dev).multi_processor_count
        _SMS[dev.index] = n
    return n


def _get_compiled(
    H: int,
    dev: torch.device,
    gate2: int = 0,
    mid_bf16: bool = False,
    has_split: bool = False,
    do_export: bool = False,
    have_state: bool = True,
    beta_tma: bool = False,
    bf16_dv: bool = False,
    beta_bf16: bool = False,
    qk_rowpair: bool = False,
    rcp_decor: bool = True,
    rf_hoist: bool = True,
    restore_tail: bool = False,
    gram_w9: int = 0,
    reg_mode: int = 0,
    store_final_state: bool = False,
    full_chunks: bool = False,
    lower_bound_m5: bool = False,
    approximate_split: bool = False,
    prep_peel: bool = False,
    gate_roll: bool = False,
    cluster_size: int = 1,
    partial_tma: bool = True,
    state_rows: bool = False,
    gswz: bool = False,
):
    key = (
        state_rows,
        gswz,
        dev.index,
        torch.cuda.get_device_capability(dev),
        H,
        gate2,
        mid_bf16,
        store_final_state,
        has_split,
        do_export,
        have_state,
        beta_tma,
        bf16_dv,
        beta_bf16,
        qk_rowpair,
        rcp_decor,
        rf_hoist,
        restore_tail,
        gram_w9,
        reg_mode,
        full_chunks,
        lower_bound_m5,
        approximate_split,
        prep_peel,
        cluster_size,
        gate_roll,
        partial_tma,
        tuple(sorted(_DEV.items())),
    )
    entry = _COMPILED.get(key)
    if entry is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Warm the persistent KDA kernel before CUDA graph capture"
            )
        N = 2
        T = 8 * C32
        bf = torch.bfloat16
        mk = lambda t, a=16: from_dlpack(t, assumed_align=a).mark_compact_shape_dynamic(
            mode=0
        )

        def beta_rows_fake():
            return cute.runtime.make_fake_tensor(
                cutlass.BFloat16,
                shape=(cute.sym_int(), H),
                stride=(cute.sym_int64(), 1),
                assumed_align=2,
            )

        def qkv_fake():
            return cute.runtime.make_fake_tensor(
                cutlass.BFloat16,
                shape=(cute.sym_int(), H, D),
                stride=(cute.sym_int64(divisibility=8), D, 1),
                assumed_align=16,
            )

        g = torch.empty(T, H, D, dtype=bf, device=dev)
        beta = torch.empty(T, H, dtype=bf, device=dev)
        alog = torch.empty(H, dtype=torch.float32, device=dev)
        dtb = torch.empty(H, D, dtype=torch.float32, device=dev)
        st = torch.empty(N * H, D, D, dtype=torch.float32, device=dev)
        ns = torch.empty(N * H, D, D, dtype=torch.float32, device=dev)
        cu_t = torch.zeros(N + 1, dtype=torch.int32, device=dev)
        cu_t[1] = T // 2
        cu_t[2] = T
        G = 2 * H
        soff = torch.arange(G + 1, dtype=torch.int32, device=dev)
        schain = torch.arange(G, dtype=torch.int32, device=dev)
        spt0 = torch.zeros(G, dtype=torch.int32, device=dev)
        sptn = torch.full((G,), T // 2, dtype=torch.int32, device=dev)
        ssrc = torch.full((G,), -1, dtype=torch.int32, device=dev)
        sdst = torch.full((G,), -1, dtype=torch.int32, device=dev)
        srow_d = torch.zeros(2 * G, dtype=torch.int64, device=dev)
        mid = torch.empty(
            2, D, D, dtype=torch.bfloat16 if mid_bf16 else torch.float32, device=dev
        )
        mfl = torch.zeros(2, dtype=torch.int32, device=dev)
        tpr = torch.zeros(G * 64, dtype=torch.int64, device=dev)
        out = torch.empty(T, H, D, dtype=bf, device=dev)
        nc2 = (T // 2 + C32 - 1) // C32
        exp_ws = torch.empty(H * nc2 * 64, 1, 256, dtype=bf, device=dev)
        expt = torch.empty(H * nc2, 160, dtype=torch.float32, device=dev)
        stream = cuda_driver.CUstream(torch.cuda.current_stream(dev).cuda_stream)

        def compile_kernel():
            kw = dict(
                H_=H,
                stream=stream,
                TPROBE_=_TPROBE,
                GATE2_=gate2,
                FINAL_=1 if store_final_state else 0,
                SPLIT_=1 if has_split and not approximate_split else 0,
                WARM_SPLIT_=1 if approximate_split else 0,
                EXPORT_=1 if do_export else 0,
                STATE_=1 if have_state else 0,
                BETA_TMA_=1 if beta_tma else 0,
                BF16_DV_=1 if bf16_dv else 0,
                BETA_BF16_=1 if beta_bf16 else 0,
                QK_ROWPAIR_=int(qk_rowpair),
                RCP_DECOR_=1 if rcp_decor else 0,
                RF_HOIST_=1 if rf_hoist else 0,
                RESTORE_TAIL_=1 if restore_tail else 0,
                GRAM_W9_=int(gram_w9),
                REG_MODE_=int(reg_mode),
                FULL_CHUNKS_=1 if full_chunks else 0,
                LOWER_BOUND_M5_=1 if lower_bound_m5 else 0,
                CLUSTER_=int(cluster_size),
                SROW_=1 if state_rows else 0,
                PREP_PEEL_=1 if prep_peel else 0,
                PARTIAL_TMA_=1 if partial_tma else 0,
                GATE_ROLL_=1 if gate_roll else 0,
                GSWZ_=1 if gswz else 0,
            )
            kw.update(_DEV)
            return cute.compile(
                _k._launch_pkd,
                qkv_fake(),
                qkv_fake(),
                qkv_fake(),
                mk(g),
                # The per-element beta loads (no TMA) take any row stride: a
                # column view of the KDA in_proj output goes in uncopied.
                mk(beta, 4) if beta_tma else beta_rows_fake(),
                from_dlpack(alog, assumed_align=16),
                from_dlpack(dtb, assumed_align=16),
                mk(st),
                mk(ns),
                mk(srow_d, 8),
                mk(out),
                mk(cu_t, 4),
                mk(soff, 4),
                mk(schain, 4),
                mk(spt0, 4),
                mk(sptn, 4),
                mk(ssrc, 4),
                mk(sdst, 4),
                mk(mid),
                mk(mfl, 4),
                mk(exp_ws),
                mk(expt),
                mk(tpr, 8),
                cutlass.Int64(0),
                cutlass.Int32(1),
                cutlass.Int32(G),
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int32(nc2),
                cutlass.Int32(1),
                cutlass.Float32(1.0),
                cutlass.Float32(-1.0),
                **kw,
                options="--enable-tvm-ffi --opt-level 3",
            )

        tfn = build_and_load_cute_dsl_kernel(
            "kda_persistent",
            "kda_" + hashlib.sha256(repr(key).encode()).hexdigest()[:24],
            compile_kernel,
            extra_key_files=[str(Path(__file__)), str(Path(_k.__file__))],
        )
        _COMPILED[key] = tfn
        entry = tfn
    return entry


def _dummy_exp(dev: torch.device):
    key = dev.index
    tensors = _DUMMY_EXP.get(key)
    if tensors is None:
        tensors = (
            torch.empty(64, 1, 256, dtype=torch.bfloat16, device=dev),
            torch.empty(1, 160, dtype=torch.float32, device=dev),
        )
        _DUMMY_EXP[key] = tensors
    return tensors


def _dummy_mid(dev: torch.device):
    key = dev.index
    tensors = _DUMMY_MID.get(key)
    if tensors is None:
        tensors = (
            torch.empty(1, D, D, dtype=torch.float32, device=dev),
            torch.zeros(1, dtype=torch.int32, device=dev),
        )
        _DUMMY_MID[key] = tensors
    return tensors


_DUMMY_SROW: dict = {}


def _dummy_srow(dev: torch.device):
    tensor = _DUMMY_SROW.get(dev.index)
    if tensor is None:
        tensor = torch.zeros(2, dtype=torch.int64, device=dev)
        _DUMMY_SROW[dev.index] = tensor
    return tensor


def state_cache_view(cache, H: int):
    """The (1, D, D) state view of ``state_cache_rows`` alone (no index work),
    for a layer that reuses the step's ``srow`` table."""
    assert cache.dtype == torch.float32 and cache.shape[1:] == (H, D, D)
    assert cache.stride()[1:] == (D * D, D, 1)
    assert cache.stride(0) % 8 == 0, "state slots must be 32-byte aligned"
    return cache.as_strided((1, D, D), (D * D, D, 1))


def state_cache_rows(cache, indices, has_initial_state, H: int):
    """In-place state I/O on the recurrent-state cache ``cache`` [slots, H, D, D]
    (FP32; each (D, D) block dense, any slot stride, e.g. the hybrid cache's
    shared per-slot page).

    Returns ``(rows, srow)``: ``rows`` is a (1, D, D) view at the cache storage
    base and ``srow`` the int64 [2 * N * H] chain -> element offset table (load
    offset, or -1 for a zero initial state; then store offset) for
    ``fwd(..., srow=srow)``. Same result as gather_initial_states + fwd +
    write_state_rows.
    """
    assert cache.dtype == torch.float32 and cache.shape[1:] == (H, D, D)
    assert cache.stride()[1:] == (D * D, D, 1)
    assert cache.stride(0) % 8 == 0, "state slots must be 32-byte aligned"
    rows = cache.as_strided((1, D, D), (D * D, D, 1))
    n = indices.numel()
    base = indices.to(torch.int64).view(n, 1) * cache.stride(0) + torch.arange(
        H, dtype=torch.int64, device=cache.device
    ) * (D * D)
    load = torch.where(has_initial_state[:n].view(n, 1).bool(), base, -1)
    srow = torch.cat((load.reshape(-1), base.reshape(-1)))
    return rows, srow


def _tprobe_buf(G: int, dev: torch.device):
    if _DEV.get("TLINE_"):
        tensor = torch.zeros(max(G * 64, 48 * 32), dtype=torch.int64, device=dev)
        LAST_TPROBE[0] = tensor
        return tensor
    if _TPROBE:
        tensor = torch.zeros(G * 64, dtype=torch.int64, device=dev)
        LAST_TPROBE[0] = tensor
        return tensor
    key = dev.index
    tensor = _DUMMY_TPR.get(key)
    if tensor is None:
        tensor = torch.zeros(64, dtype=torch.int64, device=dev)
        _DUMMY_TPR[key] = tensor
    return tensor


@torch.no_grad()
def _launch_forward(
    q,
    k,
    v,
    g,
    beta,
    scale,
    out,
    A_log,
    dt_bias,
    lower_bound,
    initial_state=None,
    final_state=None,
    cu_seqlens=None,
    sched=None,
    export=None,
    gate2=0,
    has_split=False,
    beta_tma=False,
    bf16_dv=True,
    beta_bf16=True,
    qk_rowpair=False,
    rcp_decor=True,
    rf_hoist=True,
    restore_tail=False,
    gram_w9=0,
    reg_mode=0,
    full_chunks=False,
    lower_bound_m5=False,
    approximate_split=False,
    prep_peel=False,
    cluster_size=1,
    gate_roll=False,
    partial_tma=True,
    srow=None,
    compile_only=False,
    gswz=False,
):
    dev = q.device
    B, T, H, K = q.shape
    assert K == D
    qf = q.view(T * B, H, D)
    kf = k.view(T * B, H, D)
    vf = v.view(T * B, H, D)
    gf = g.view(T * B, H, D)
    bf_ = beta.view(T * B, H)
    of = out.view(T * B, H, D)

    if cu_seqlens is None:
        cu_t = torch.tensor([0, B * T], dtype=torch.int32, device=dev)
        nseqs = 1
    else:
        cu_t = cu_seqlens.to(torch.int32)
        nseqs = int(cu_seqlens.numel()) - 1

    store_final_state = final_state is not None
    if final_state is None:
        final_state = torch.empty(nseqs, H, D, D, dtype=torch.float32, device=dev)
    have_state = 1 if initial_state is not None else 0
    if srow is not None:
        # initial_state / final_state are the same flat (rows, D, D) view of
        # the recurrent-state cache; srow maps chains to its rows.
        assert initial_state is not None and final_state is not None
        assert srow.numel() == 2 * nseqs * H and srow.dtype == torch.int64
    st = initial_state if initial_state is not None else final_state
    if st.dtype != torch.float32:
        st = st.float()

    if sched is None:
        chains = nseqs * H
        soff = torch.arange(chains + 1, dtype=torch.int32, device=dev)
        schain = torch.arange(chains, dtype=torch.int32, device=dev)
        spt0 = torch.zeros(chains, dtype=torch.int32, device=dev)
        sptn = cu_t.diff().to(torch.int32).repeat_interleave(H)
        ssrc = torch.full((chains,), -1, dtype=torch.int32, device=dev)
        sdst = ssrc
        mid, mfl = _dummy_mid(dev)
        fepoch = 1
        G = chains
    else:
        (soff, schain, spt0, sptn, ssrc, sdst, G, mid, mfl, fepoch) = sched
    if export is None:
        exp_ws, expt = _dummy_exp(dev)
        do_export, export_seq, nc2 = 0, 0, 1
    else:
        exp_ws, expt, export_seq, nc2 = export
        do_export = 1

    tfn = _get_compiled(
        H,
        dev,
        gate2,
        mid.dtype == torch.bfloat16,
        has_split=has_split,
        do_export=bool(do_export),
        have_state=bool(have_state),
        beta_tma=beta_tma,
        bf16_dv=bf16_dv,
        beta_bf16=beta_bf16,
        qk_rowpair=qk_rowpair,
        rcp_decor=rcp_decor,
        rf_hoist=rf_hoist,
        restore_tail=restore_tail,
        gram_w9=gram_w9,
        reg_mode=reg_mode,
        store_final_state=store_final_state,
        full_chunks=full_chunks,
        lower_bound_m5=lower_bound_m5,
        approximate_split=approximate_split,
        prep_peel=prep_peel,
        cluster_size=cluster_size,
        gate_roll=gate_roll,
        state_rows=srow is not None,
        partial_tma=partial_tma,
        gswz=gswz,
    )
    if compile_only:
        return None
    stream = cuda_driver.CUstream(torch.cuda.current_stream(dev).cuda_stream)
    tfn(
        qf,
        kf,
        vf,
        gf,
        bf_,
        A_log,
        dt_bias.view(H, D),
        st.view(-1, D, D) if srow is not None else st.view(nseqs * H, D, D),
        final_state.view(-1, D, D)
        if srow is not None
        else final_state.view(nseqs * H, D, D),
        srow if srow is not None else _dummy_srow(dev),
        of,
        cu_t,
        soff,
        schain,
        spt0,
        sptn,
        ssrc,
        sdst,
        mid,
        mfl,
        exp_ws,
        expt,
        _tprobe_buf(G, dev),
        0,
        have_state,
        G,
        do_export,
        export_seq,
        nc2,
        fepoch,
        float(scale),
        float(lower_bound) * LOG2E,
        stream,
    )
    return out, final_state


C_TOK = 32  # chunk tokens; split points are chunk-aligned
# Modelled per-piece cost (state seed/export and pipeline refill), in chunks;
# measured B300 chain starts cost about three chunks.
PIECE_OVERHEAD = 3
# Smallest producer or consumer piece, in chunks.
MIN_PIECE = 2
# Split only for this modelled gain: the handoff-capable kernel variant costs
# up to 3% on H96 routes.
MIN_SPLIT_GAIN = 0.05


def _lpt(costs, nslots):
    """Longest-processing-time packing of whole chains; returns (slots, loads)."""
    heap = [(0, slot) for slot in range(nslots)]
    heapq.heapify(heap)
    slots: list[list[int]] = [[] for _ in range(nslots)]
    loads = [0] * nslots
    for chain in sorted(range(len(costs)), key=lambda c: (-costs[c], c)):
        load, slot = heapq.heappop(heap)
        slots[slot].append(chain)
        loads[slot] = load + costs[chain]
        heapq.heappush(heap, (loads[slot], slot))
    return slots, loads


def _wrap_around(chunks, full, span, nslots):
    """McNaughton wrap-around packing at makespan ``span`` (chunks).

    Chains fill slots in order. The chain crossing a slot boundary is cut at a
    chunk boundary: its first part (a producer, full chunks only) starts the
    next slot and its remainder (the consumer) ends the current one. Producers
    lead their slots and never wait, so every consumer's producer runs on a
    resident CTA and the handoff graph cannot deadlock. Returns None when the
    chains do not fit in ``nslots`` slots.
    """
    slots = [[]]
    load = 0
    # Alternate long and short chains so every slot gets a similar mix of
    # chain starts; short chains cost more per chunk than the model assumes.
    ranked = sorted(range(len(chunks)), key=lambda c: (-chunks[c], c))
    order = [
        ranked[i // 2] if i % 2 == 0 else ranked[-1 - i // 2]
        for i in range(len(ranked))
    ]
    for chain in order:
        count = chunks[chain]
        if load + count + PIECE_OVERHEAD <= span:
            slots[-1].append((chain, 0, count))
            load += count + PIECE_OVERHEAD
            continue
        tail = span - load - 2 * PIECE_OVERHEAD
        head = count - tail
        if tail >= MIN_PIECE and MIN_PIECE <= head <= full[chain]:
            slots[-1].append((chain, head, count))
            slots.append([(chain, 0, head)])
            load = head + PIECE_OVERHEAD
        else:
            slots.append([(chain, 0, count)])
            load = count + PIECE_OVERHEAD
        if len(slots) > nslots:
            return None
    return slots


def _split_heads(chunks, full, nslots, span_limit):
    """LPT after cutting a short head off the longest chains.

    Each head is a producer placed before every other piece of its slot, so
    producers never wait; the rest of the chain is a consumer that ends its
    slot. Searches a coarse grid of split counts and head lengths with a
    waiting-aware simulation and returns the best (span, slots) below
    ``span_limit``, where slots hold (chain, first chunk, end chunk), or None.
    """
    order = sorted(range(len(chunks)), key=lambda c: (-chunks[c], c))
    best = None
    for n_split in (8, 16, 24, 32, 48, 64, 96, 128, 192, 256):
        if n_split > len(order):
            break
        cut = order[:n_split]
        room = min(full[c] for c in cut)
        for head in (2, 4, 6, 8, 12, 16, 24, 32, 48, 64):
            if head > room or chunks[cut[-1]] - head < MIN_PIECE:
                break
            # (cost, kind, chain, first chunk, end chunk); kind 0 = producer,
            # 1 = whole chain, 2 = consumer.
            jobs = [
                (chunks[c] + PIECE_OVERHEAD, 1, c, 0, chunks[c])
                for c in order[n_split:]
            ]
            for c in cut:
                jobs.append((head + PIECE_OVERHEAD, 0, c, 0, head))
                jobs.append((chunks[c] - head + PIECE_OVERHEAD, 2, c, head, chunks[c]))
            heap = [(0, slot) for slot in range(nslots)]
            slots = [[] for _ in range(nslots)]
            for job in sorted(jobs, key=lambda j: (-j[0], j[1], j[2])):
                load, slot = heapq.heappop(heap)
                slots[slot].append(job)
                heapq.heappush(heap, (load + job[0], slot))
            ready = {}
            for slot in slots:
                slot.sort(key=lambda j: (j[1], -j[0]))
                t = 0
                for cost, kind, c, _, _ in slot:
                    if kind == 0:
                        t += cost
                        ready[c] = t
            span = 0
            for slot in slots:
                t = 0
                for cost, kind, c, _, _ in slot:
                    if kind == 2:
                        t = max(t, ready[c])
                    t += cost
                span = max(span, t)
            if span < span_limit and (best is None or span < best[0]):
                best = (span, [[j[2:] for j in slot] for slot in slots])
    return best


def _piece_items(lens, H: int, nslots: int):
    """Pack the (sequence, head) chains onto at most ``nslots`` persistent CTAs.

    Whole-chain LPT is used unless cutting chains lowers the modelled
    makespan. Split schedules use wrap-around packing, with exact FP32 state
    handoffs between the pieces of a chain.

    Returns ``(items, nbuf)``: per slot, its pieces ``(chain, t0, tn, src,
    dst)`` in execution order (token window [t0, t0 + tn) of the chain;
    ``src``/``dst`` the handoff buffer it seeds from / exports to, or -1).
    """
    nchains = len(lens) * H
    chunks = [(lens[c // H] + C_TOK - 1) // C_TOK for c in range(nchains)]
    full = [lens[c // H] // C_TOK for c in range(nchains)]
    G = min(nchains, nslots)
    costs = [count + PIECE_OVERHEAD for count in chunks]
    lpt_slots, loads = _lpt(costs, G)
    items = [[(c, 0, lens[c // H], -1, -1) for c in slot] for slot in lpt_slots]
    nbuf = 0
    span = max(max(costs), -(-sum(costs) // G))
    limit = max(loads) * (1 - MIN_SPLIT_GAIN)
    if nchains > G and span <= limit:
        packed = None
        while packed is None and span <= limit:
            packed = _wrap_around(chunks, full, span, G)
            span += 1
        heads = _split_heads(chunks, full, G, span if packed else limit + 1)
        if heads is not None:
            packed = heads[1]
        if packed is not None:
            split = sorted({chain for slot in packed for chain, c0, _ in slot if c0})
            buffer_of = {chain: index for index, chain in enumerate(split)}
            items = [
                [
                    (
                        chain,
                        c0 * C_TOK,
                        min(c1 * C_TOK, lens[chain // H]) - c0 * C_TOK,
                        buffer_of[chain] if c0 else -1,
                        buffer_of[chain] if c1 < chunks[chain] else -1,
                    )
                    for chain, c0, c1 in slot
                ]
                for slot in packed
            ]
            nbuf = len(buffer_of)
    return items, nbuf


def _tensorize(items, dev: torch.device):
    fc, ft0, ftn, fsrc, fdst, off = [], [], [], [], [], [0]
    for slot in items:
        for c, t0, tn, src, dst in slot:
            fc.append(c)
            ft0.append(t0)
            ftn.append(tn)
            fsrc.append(src)
            fdst.append(dst)
        off.append(len(fc))
    i32 = lambda x: torch.tensor(x, dtype=torch.int32, device=dev)
    return (
        i32(off),
        i32(fc),
        i32(ft0),
        i32(ftn),
        i32(fsrc),
        i32(fdst),
        len(items),
    )


def _piece_schedule(lens, H: int, nslots: int, dev: torch.device):
    items, nbuf = _piece_items(lens, H, nslots)
    return (*_tensorize(items, dev), nbuf)


def _cut_items(items, lens, cuts, H: int, nbuf: int):
    """Re-express a schedule with checkpoint cuts as sequence boundaries.

    ``cuts[s]`` (0 = none, else 0 < cut < lens[s]) splits sequence ``s`` into a
    prefix pseudo-sequence [0, cut) and a suffix pseudo-sequence [cut, len).
    The prefix ends like any sequence (a partial last chunk is allowed) and
    exports its FP32 state to buffer ``s * H + h`` instead of storing a final
    state; the suffix seeds from that buffer. Both keep their own 32-token
    chunk grid, so the prefix computes exactly as a run of [0, cut) alone and
    the suffix exactly as a run of [cut, len) seeded with that state.

    A chain is split by the planner at most once (a producer head and a
    consumer remainder). A planner boundary inside the suffix moves up to the
    suffix's chunk grid (or vanishes past its end), and the piece covering the
    cut becomes two consecutive pieces of the same slot, so every new wait is
    on the piece just before it in the same CTA: the handoff graph keeps the
    planner's deadlock-free shape. The planner's own buffers move to
    ``nseq * H + b``.

    Returns ``(items, pseudo_lens, pseudo_of, nbuf_total)``: pieces in
    pseudo-sequence chains and coordinates, the pseudo-sequence lengths, and
    per original sequence its (prefix, suffix) pseudo indices (suffix -1 when
    uncut)."""
    nseq = len(lens)
    shift = nseq * H
    pseudo_of, pseudo_lens = [], []
    for s, length in enumerate(lens):
        x = cuts[s]
        if x:
            if not 0 < x < length:
                raise ValueError(
                    f"checkpoint offset {x} outside sequence {s} of length {length}"
                )
            pseudo_of.append((len(pseudo_lens), len(pseudo_lens) + 1))
            pseudo_lens += [x, length - x]
        else:
            pseudo_of.append((len(pseudo_lens), -1))
            pseudo_lens.append(length)

    def sh(b):
        return b + shift if b >= 0 else -1

    # Per-chain planner pieces: chain -> [(slot, index)] in token order.
    where: dict = {}
    for si, slot in enumerate(items):
        for ii, (c, t0, _tn, _src, _dst) in enumerate(slot):
            where.setdefault(c, []).append((t0, si, ii))
    # Planner buffers move past the cut buffers first, so the two never alias.
    new = [[(c, t0, tn, sh(src), sh(dst)) for c, t0, tn, src, dst in slot]
           for slot in items]
    drop = set()
    insert: dict = {}
    for c, locs in where.items():
        s, h = divmod(c, H)
        x = cuts[s]
        if not x:
            continue
        length = lens[s]
        locs.sort()
        if len(locs) > 2:
            raise RuntimeError(f"chain {c} has {len(locs)} planner pieces (max 2)")
        cb = s * H + h
        if len(locs) == 2:
            (_, sa, ia), (_, sb, ib) = locs
            ca, a0, an, asrc, adst = new[sa][ia]
            cbb, b0, bn, bsrc, bdst = new[sb][ib]
            p = b0
            if p > x:
                p = x + -(-(p - x) // C_TOK) * C_TOK
                if p >= length:
                    # boundary vanishes: the head covers the whole chain
                    new[sa][ia] = (ca, a0, length - a0, asrc, -1)
                    drop.add((sb, ib))
                    locs = [(a0, sa, ia)]
                else:
                    new[sa][ia] = (ca, a0, p - a0, asrc, adst)
                    new[sb][ib] = (cbb, p, length - p, bsrc, bdst)
            if len(locs) == 2 and p == x:
                new[sa][ia] = (ca, a0, x - a0, asrc, cb)
                new[sb][ib] = (cbb, x, length - x, cb, bdst)
                continue
        for _, si, ii in locs:
            cc, t0, tn, src, dst = new[si][ii]
            if t0 < x < t0 + tn:
                new[si][ii] = (cc, t0, x - t0, src, cb)
                insert[(si, ii)] = (cc, x, t0 + tn - x, cb, dst)
                break
    out = []
    for si, slot in enumerate(new):
        row = []
        for ii, (c, t0, tn, src, dst) in enumerate(slot):
            if (si, ii) in drop:
                continue
            pieces = [(c, t0, tn, src, dst)]
            if (si, ii) in insert:
                pieces.append(insert[(si, ii)])
            for c2, t2, n2, src2, dst2 in pieces:
                s, h = divmod(c2, H)
                pre, suf = pseudo_of[s]
                if suf >= 0 and t2 >= cuts[s]:
                    row.append((suf * H + h, t2 - cuts[s], n2, src2, dst2))
                else:
                    row.append((pre * H + h, t2, n2, src2, dst2))
        out.append(row)
    return out, pseudo_lens, pseudo_of, shift + nbuf


def beta_tma_supported(beta) -> bool:
    """The beta TMA path loads (C tokens x 8 heads) BF16 boxes from the
    (tokens, H) beta rows: the row stride (2H bytes) and the base must be
    16-byte multiples, i.e. H % 8 == 0. Other head counts (the TP8 shard of the
    96-head KDA has 12; 4, 6, 20 likewise) take the per-element beta loads:
    with TMA their output and final state were wrong (rel. RMS 0.4-0.6 vs
    FlashKDA, BUG-DCP8-3)."""
    H = beta.shape[-1]
    return (
        beta.dtype == torch.bfloat16
        and beta.stride(-1) == 1
        and (H * beta.element_size()) % 16 == 0
        and all((s * beta.element_size()) % 16 == 0 for s in beta.stride()[:-1])
        and beta.data_ptr() % 16 == 0
    )


def make_plan(offsets, H, dev):
    """Build private scheduling and handoff storage from validated host offsets."""
    lens = [end - start for start, end in zip(offsets, offsets[1:], strict=False)]
    cu32 = torch.tensor(offsets, dtype=torch.int32, device=dev)
    soff, schain, spt0, sptn, ssrc, sdst, G, nbuf = _piece_schedule(
        lens, H, _sm_count(dev), dev
    )
    # Every inter-CTA handoff carries FP32 state.
    mid = torch.empty(max(nbuf, 1), D, D, dtype=torch.float32, device=dev)
    flags = torch.zeros(max(nbuf, 1), dtype=torch.int32, device=dev)
    return (
        cu32,
        soff,
        schain,
        spt0,
        sptn,
        ssrc,
        sdst,
        G,
        mid,
        flags,
        nbuf > 0,
        all(length % C_TOK == 0 for length in lens),
    )


def make_cut_plan(offsets, cuts, H, dev):
    """``make_plan`` with prefill checkpoints: ``cuts[s]`` (0 = none) is the
    token offset into sequence ``s`` whose FP32 state is exported.

    The schedule is planned on the uncut sequences and the cuts become
    sequence boundaries (``_cut_items``). Returns ``(plan, meta)``: the plan's
    handoff storage slot is None; the caller passes ``mid`` (at least
    ``meta["nbuf"]`` (D, D) FP32 rows) whose first ``nseq * H`` rows receive
    the checkpoint states, row ``s * H + h`` for sequence ``s``, head ``h``.
    ``meta["pseudo_of"]`` maps each sequence to its (prefix, suffix) pseudo
    sequences (suffix -1 when uncut) and ``meta["cu"]`` holds the
    pseudo-sequence offsets."""
    lens = [end - start for start, end in zip(offsets, offsets[1:], strict=False)]
    items, nbuf = _piece_items(lens, H, _sm_count(dev))
    items, pseudo_lens, pseudo_of, nbuf_total = _cut_items(items, lens, cuts, H, nbuf)
    pcu = [offsets[0]]
    for length in pseudo_lens:
        pcu.append(pcu[-1] + length)
    soff, schain, spt0, sptn, ssrc, sdst, G = _tensorize(items, dev)
    flags = torch.zeros(nbuf_total, dtype=torch.int32, device=dev)
    plan = (
        torch.tensor(pcu, dtype=torch.int32, device=dev),
        soff,
        schain,
        spt0,
        sptn,
        ssrc,
        sdst,
        G,
        None,
        flags,
        True,
        all(length % C_TOK == 0 for length in pseudo_lens),
    )
    meta = {"nbuf": nbuf_total, "pseudo_of": pseudo_of, "cu": pcu, "nseq": len(lens)}
    return plan, meta


@torch.no_grad()
def fwd(
    q,
    k,
    v,
    g,
    beta,
    scale,
    out,
    A_log,
    dt_bias,
    lower_bound,
    initial_state=None,
    final_state=None,
    cu_seqlens=None,
    plan=None,
    srow=None,
    compile_only=False,
):
    fixed = cu_seqlens is None or int(cu_seqlens.numel()) == 2
    (
        cu32,
        soff,
        schain,
        spt0,
        sptn,
        ssrc,
        sdst,
        G,
        mid,
        mfl,
        has_split,
        full_chunks,
    ) = plan
    # The beta TMA choice depends on the head count only (H % 8 == 0): a beta
    # view whose base or row stride is misaligned is copied, so serving launches
    # the variant warm-up compiled for H and never JITs.
    if (
        beta.dtype == torch.bfloat16
        and beta.shape[-1] % 8 == 0
        and not beta_tma_supported(beta)
    ):
        beta = beta.clone(memory_format=torch.contiguous_format)
    # Reset before every launch, including captured launches and graph replays.
    if has_split:
        mfl.zero_()
    # Route: single sequence versus packed sequences.  Single sequences use the
    # fused-gate kernel with row-paired Q/K prep and a per-element partial
    # tail; packed sequences roll the gate scan in eight-row batches and
    # TMA-load partial chunks with row masks.  Packed sequences that end in a
    # partial chunk also run the GATE2 recurrence (wide V residual, two-half
    # V_new handoff, FIN-gated epilogue image): bitwise identical, 2-3.5%
    # faster at H96 (f16k, 3x5456, mixed, 2x8k with partial tails) but 9-19%
    # slower when every chunk is full (4x4k, 2x8k), so all-full packed shapes
    # keep the plain recurrence.  Whether every chunk is full is a static
    # property of the lengths, and a -5 lower bound folds into the gate.
    # Per-shape dispatch for the TP8 head shard (H = 12, VR200, 1-GPU A/B at
    # 8 / 16 x 2048 packed full chunks; PR-5 loop 1): the prep warps' gate
    # cumsum reads use the bank-swizzled layout (GSWZ), and all-full packed
    # sequences take the row-paired GATE2 route without the gate roll or the
    # partial-chunk TMA (-4.2% / -4.5%). Both are data movement only: output and
    # final state are bitwise identical to the other routes.
    tp8 = q.shape[2] == 12
    rowpair_full = tp8 and full_chunks and not fixed
    return _launch_forward(
        q,
        k,
        v,
        g,
        beta,
        scale,
        out,
        A_log,
        dt_bias,
        lower_bound,
        initial_state=initial_state,
        final_state=final_state,
        cu_seqlens=cu32,
        sched=(soff, schain, spt0, sptn, ssrc, sdst, G, mid, mfl, 1),
        gate2=1 if (fixed or not full_chunks or rowpair_full) else 0,
        has_split=has_split,
        beta_tma=beta_tma_supported(beta),
        qk_rowpair=4 if (fixed or rowpair_full) else 8,
        restore_tail=False,
        reg_mode=2,
        full_chunks=full_chunks,
        lower_bound_m5=lower_bound == -5.0,
        approximate_split=False,
        gate_roll=not (fixed or rowpair_full),
        partial_tma=not (fixed or rowpair_full),
        srow=srow,
        compile_only=compile_only,
        gswz=tp8,
    )


def warm_compile(H: int, dev: torch.device, lower_bound: float) -> None:
    """Compile (or load from the CuTe cache) every kernel variant fwd() can
    launch for H heads: single or packed sequences, full or partial chunks,
    with or without split chain pieces, in-place state rows or gathered
    states. Runs fwd()'s own flag derivation on small dummies and stops
    before the launch, so serving steps never compile."""
    T = 2 * C_TOK
    bf = torch.bfloat16
    q = torch.zeros(1, T, H, D, dtype=bf, device=dev)
    beta = torch.zeros(1, T, H, dtype=bf, device=dev)
    A_log = torch.zeros(H, dtype=torch.float32, device=dev)
    dt_bias = torch.zeros(H, D, dtype=torch.float32, device=dev)
    for nseq in (1, 2):
        offs = [0, T] if nseq == 1 else [0, C_TOK, T]
        cu32 = torch.tensor(offs, dtype=torch.int32, device=dev)
        state = torch.zeros(nseq, H, D, D, dtype=torch.float32, device=dev)
        srow = torch.zeros(2 * nseq * H, dtype=torch.int64, device=dev)
        idx = torch.zeros(nseq * H + 1, dtype=torch.int32, device=dev)
        mid = torch.zeros(1, D, D, dtype=torch.float32, device=dev)
        mfl = torch.zeros(1, dtype=torch.int32, device=dev)
        for full_chunks in (False, True):
            for has_split in (False, True):
                plan = (cu32, idx, idx, idx, idx, idx, idx, nseq * H, mid, mfl)
                plan += (has_split, full_chunks)
                for rows in (None, srow):
                    fwd(q, q, q, q, beta, D**-0.5, q, A_log, dt_bias,
                        lower_bound, initial_state=state, final_state=state,
                        cu_seqlens=cu32, plan=plan, srow=rows,
                        compile_only=True)

