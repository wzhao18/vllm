# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 prefill indexer on sm_107: one layer's exact prefill top-k through the
selective path of ``indexer`` (sampled thresholds, candidate-publishing MXFP4
score, exact candidate select, on-device recovery of unsettled rows), called by
the stock sparse_attn_indexer for single-rank (no DCP / PCP / ShardQ) MXFP4
prefills in place of its per-chunk logits + top-k loop.

The stock loop splits the rows into chunks that bound the materialized logits;
this path writes no logits on its common path, so it regroups consecutive chunks
into groups of about GROUP_ROWS rows (fewer, longer launches: the persistent
score kernel's tail stays small) and bounds only the recovery scratch.

Per step (cached on the first chunk's ``logits_inputs`` by the step's first DSA
layer, outside graph capture): one host read of the row bounds of all chunks ->
segments (one request's rows in one chunk), their 128-aligned key layout and a
SelectivePlan per group. Per layer, with no host reads: each segment's keys
gathered to its aligned base and the selective top-k of each group.
"""

import torch

from vllm import _custom_ops as ops
from vllm.models.kimi_k3.nvidia.specialized_prefill import indexer as pidx
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import mqa_logits_mx_inputs

_KEY = "kimi_k3_pidx"
GROUP_ROWS = 16384
# Transient device memory one row group may hold (candidate regions, sampled
# logits, the recovery's logits scratch, bitmap and bound-bin keys): the memory
# profile reserves it (reserved_bytes), so groups at long contexts get fewer rows.
SCRATCH_BUDGET = 4 << 30


def group_bytes(rows: int, pitch: int) -> int:
    """Transient bytes of a group of ``rows`` rows whose widest request has
    ``pitch`` (128-aligned) keys."""
    per_row = (
        pidx.REGION * 4 + pidx.SAMPLES * 2 + pitch * 2 + pitch // 8 + pidx.CAND * 4
    )
    return (rows + 1) * per_row + pitch * 2


def wave_cap_items(pitch: int, sms: int) -> int:
    """Most work items a group of widest request ``pitch`` may hold: whole waves of
    ``sms`` items (one CTA per SM; an item is BQ rows walking its request's whole
    context, so a group costs ceil(items / sms) item latencies), as many as the
    scratch budget and GROUP_ROWS allow; 0 when not even one wave fits (the budget
    alone then bounds the group). Grouping never changes a row's selection."""
    per_row = group_bytes(1, pitch) - group_bytes(0, pitch)
    fit = min(GROUP_ROWS, (SCRATCH_BUDGET - group_bytes(0, pitch)) // per_row)
    return (fit // pidx.BQ) // sms * sms


def _seg_pitch(seg) -> int:
    return -(-seg[4] // pidx.TILE) * pidx.TILE


def pack_groups(segs, sms: int):
    """Row groups (lists of segments, in token order) from the step's segments
    (chunk, request in chunk, rows, last row's keys, key count, request).

    Each group holds whole waves of work items (wave_cap_items at its widest
    request): a segment that would pass the group's last whole wave is split at
    that row (its first part ends ``rows - k`` keys earlier), so every group but
    the last costs exactly its waves. When not even one wave fits the scratch
    budget, groups close on segment boundaries by the budget alone. A row's keys
    and selection do not depend on its group: results are identical."""
    groups, cur, rows, items, pitch = [], [], 0, 0, 0

    def close():
        nonlocal cur, rows, items, pitch
        if cur:
            groups.append(cur)
        cur, rows, items, pitch = [], 0, 0, 0

    for seg in segs:
        while seg is not None:
            p = max(pitch, _seg_pitch(seg))
            cap = wave_cap_items(p, sms)
            s_items = -(-seg[2] // pidx.BQ)
            if cap == 0:
                if cur and group_bytes(rows + seg[2], p) > SCRATCH_BUDGET:
                    close()
                    continue
                cur.append(seg)
                rows, items, pitch = rows + seg[2], items + s_items, p
                seg = None
                if rows >= GROUP_ROWS:
                    close()
                continue
            room = cap - items
            if room <= 0:
                close()
                continue
            if s_items <= room:
                cur.append(seg)
                rows, items, pitch = rows + seg[2], items + s_items, p
                seg = None
                if items == cap:
                    close()
                continue
            k = room * pidx.BQ  # rows of the head part (whole items)
            head = (seg[0], seg[1], k, seg[3] - (seg[2] - k), seg[4], seg[5])
            seg = (seg[0], seg[1], seg[2] - k, seg[3], seg[4], seg[5])
            cur.append(head)
            rows, items, pitch = rows + k, items + room, p
            close()
    close()
    return groups


def reserved_bytes() -> int:
    """Bytes the memory profile reserves for this path's per-layer scratch."""
    return SCRATCH_BUDGET


def applicable(
    use_fp4_cache: bool,
    topk_tokens: int,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    dcp_world_size: int,
    has_shardq: bool,
    has_candidate_blocks: bool,
    chunks,
) -> bool:
    from vllm.logger import init_logger

    ok = (
        use_fp4_cache
        and topk_tokens == 2048
        and q_scale is not None
        and q_quant.shape[1] == pidx.HEADS
        and dcp_world_size == 1
        and not has_shardq
        and not has_candidate_blocks
        and len(chunks) > 0
        and all(c.pcp_deinterleave_idx is None for c in chunks)
        and current_platform.is_device_capability((10, 7))
    )
    init_logger(__name__).info_once(
        "Kimi K3.1 sm_107 selective prefill indexer: %s (fp4=%s topk=%d heads=%d "
        "dcp=%d shardq=%s)", "on" if ok else "off", use_fp4_cache, topk_tokens,
        q_quant.shape[1], dcp_world_size, has_shardq,
    )
    return ok


class _Group:
    """Consecutive rows [r0, r1) of the layer: their segments (one request's rows
    in one chunk), the key gathers (one per request: a request split over chunks
    shares its keys) and the SelectivePlan."""

    def __init__(self, segs, r0, r1, device):
        self.r0, self.r1 = r0, r1
        # segs: (chunk, request in chunk, rows, last row's keys, key count, request)
        widest = {}
        for c, b, _, _, n, req in segs:
            if req not in widest or n > widest[req][2]:
                widest[req] = (c, b, n)
        base_of, self.gathers, nk = {}, [], 0
        for req, (c, b, n) in widest.items():
            base_of[req] = nk
            self.gathers.append(
                (c, b, n, nk, torch.tensor([0, n], dtype=torch.int32, device=device))
            )
            nk += (n + pidx.TILE - 1) // pidx.TILE * pidx.TILE
        self.nk = max(nk, pidx.TILE)
        self.sp = pidx.SelectivePlan(
            [s[2] for s in segs], [s[3] for s in segs], [base_of[s[5]] for s in segs],
            device,
        )
        assert self.sp.t == r1 - r0


_HOST_KEY = "kimi_k3_pidx_host"


def _host_copies(chunks):
    """Pinned host copies of the chunks' row bounds and block tables plus an
    event recorded after the copies, enqueued on the current stream."""
    host = []
    for c in chunks:
        row = []
        for t in (c.cu_seqlen_ks, c.cu_seqlen_ke, c.local_cu_seq_lens, c.block_table):
            h = torch.empty(t.shape, dtype=t.dtype, device="cpu", pin_memory=True)
            h.copy_(t, non_blocking=True)
            row.append(h)
        host.append(tuple(row))
    event = torch.cuda.Event()
    event.record()
    return host, event


def prefetch_host(attn_metadata) -> None:
    """Enqueue the layout's host copies at the start of the step's forward.

    The layout of the selective prefill indexer needs the row bounds on the
    host. Copied at the first DSA layer, the host had to wait for every layer
    enqueued before it (stream synchronize), and the GPU then idled while the
    host launched the rest of the step. Enqueued before layer 0, the copies only
    wait for the step's metadata build; the first DSA layer waits on their
    event. Same data, same layout: the copies are only issued earlier."""
    if attn_metadata is None or torch.cuda.is_current_stream_capturing():
        return
    metas = attn_metadata if isinstance(attn_metadata, list) else [attn_metadata]
    seen = set()
    for meta in metas:
        if not isinstance(meta, dict):
            continue
        for m in meta.values():
            prefill = getattr(m, "prefill", None)
            chunks = getattr(prefill, "chunks", None)
            if not chunks or id(chunks[0]) in seen:
                continue
            c0 = chunks[0]
            if not hasattr(c0, "cu_seqlen_ks") or not hasattr(c0, "logits_inputs"):
                continue
            seen.add(id(c0))
            if any(
                c.local_cu_seq_lens is None or c.pcp_deinterleave_idx is not None
                for c in chunks
            ):
                continue
            if _HOST_KEY not in c0.logits_inputs and _KEY not in c0.logits_inputs:
                c0.logits_inputs[_HOST_KEY] = _host_copies(chunks)


def _layout(chunks, device):
    cached = chunks[0].logits_inputs.get(_KEY)
    if cached is not None:
        return cached
    assert not torch.cuda.is_current_stream_capturing(), (
        "the selective prefill indexer layout is built by the step's first "
        "(eager) forward"
    )
    pre = chunks[0].logits_inputs.pop(_HOST_KEY, None)
    if pre is not None and len(pre[0]) == len(chunks):
        host, event = pre
        event.synchronize()
    else:
        host, event = _host_copies(chunks)
        event.synchronize()
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    all_segs, r0 = [], chunks[0].token_start
    next_row = r0
    for ci, (c, (ks, ke, cu, bt)) in enumerate(zip(chunks, host)):
        ks, ke, cu = ks.tolist(), ke.tolist(), cu.tolist()
        assert c.token_start == next_row and len(ks) == c.token_end - c.token_start, (
            "prefill chunks must cover consecutive rows"
        )
        next_row = c.token_end
        starts = {s: b for b, s in enumerate(cu[:-1])}
        i = 0
        while i < len(ks):
            j = i
            while j + 1 < len(ks) and ks[j + 1] == ks[i]:
                j += 1
            b = starts[ks[i]]
            n = cu[b + 1] - cu[b]
            # the request's identity: its blocks (a request split over chunks has
            # the same block table row in each)
            req = bt[b].numpy().tobytes()
            all_segs.append((ci, b, j - i + 1, ke[j] - ks[i], n, req))
            i = j + 1
    groups = []
    for segs in pack_groups(all_segs, sms):
        rows = sum(s[2] for s in segs)
        groups.append(_Group(segs, r0, r0 + rows, device))
        r0 += rows
    chunks[0].logits_inputs[_KEY] = groups
    return groups


def layer_topk(
    chunks,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor,
    weights: torch.Tensor,
    topk_buffer: torch.Tensor,
) -> None:
    """Exact top-k of every prefill row of the layer into ``topk_buffer[rows,
    :2048]`` (request-local key indices, unordered, -1 padded)."""
    dev = q_quant.device
    for g in _layout(chunks, dev):
        sp = g.sp
        keys = torch.empty((g.nk, 64), dtype=torch.uint8, device=dev)
        scales = torch.empty((g.nk, 4), dtype=torch.uint8, device=dev)
        for ci, b, n, base, cu in g.gathers:
            ops.cp_gather_indexer_k_quant_cache(
                kv_cache,
                keys[base : base + n],
                scales[base : base + n],
                chunks[ci].block_table[b : b + 1],
                cu,
            )
        rows = slice(g.r0, g.r1)
        _, w16 = mqa_logits_mx_inputs(
            q_quant[rows].view(torch.int8), q_scale[rows], weights[rows]
        )
        dst = topk_buffer[rows, :2048]
        out = dst if dst.is_contiguous() else torch.empty(
            (sp.t, 2048), dtype=torch.int32, device=dev
        )
        pidx.selective_topk(
            sp,
            q_quant[rows].contiguous().view(torch.uint8),
            q_scale[rows].contiguous(),
            keys,
            scales.view(torch.int32).view(g.nk),
            w16.contiguous(),
            out,
        )
        if out is not dst:
            dst.copy_(out)
