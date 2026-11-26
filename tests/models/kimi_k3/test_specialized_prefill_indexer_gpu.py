# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 sm_107 selective prefill indexer: the top-2048 keys of every row
equal the exact top-k of the stock (DeepGEMM) MXFP4 logits, ties to the lowest
key index, also for the rows the thresholds cannot settle (recomputed on device
by the full path)."""

import pytest
import torch

pytest.importorskip("cutlass")
if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
    pytest.skip("needs a Rubin GPU (compute capability 10.7)", allow_module_level=True)

from vllm.models.kimi_k3.nvidia.ops.dsa_indexer_mxfp4 import (  # noqa: E402
    mxfp4_q_quant,
)
from vllm.models.kimi_k3.nvidia.specialized_prefill import (  # noqa: E402
    indexer as pidx,
)
from vllm.utils.deep_gemm import (  # noqa: E402
    fp8_fp4_mqa_logits,
    get_mqa_logits_metadata,
    mqa_logits_mx_inputs,
)

K = 2048


def _order16(bits):
    b = bits.to(torch.int64) & 0xFFFF
    b = torch.where(b == 0x8000, torch.zeros_like(b), b)
    return torch.where(b & 0x8000 != 0, b ^ 0xFFFF, b ^ 0x8000)


def _inputs(reqs, zero_rows=(), seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    t = sum(r for r, _ in reqs)
    index_q = torch.randn(t, 32, 128, device="cuda", dtype=torch.bfloat16, generator=g)
    index_w = torch.randn(t, 32, device="cuda", generator=g)
    q, q_scale, w = mxfp4_q_quant(index_q, index_w, (128**-0.5) * (32**-0.5), 32)
    w[list(zero_rows)] = 0
    _, w16 = mqa_logits_mx_inputs(q.view(torch.int8), q_scale, w)
    pads = [(kv + 127) // 128 * 128 for _, kv in reqs]
    nk = sum(pads)
    k8 = torch.randint(0, 256, (nk, 64), device="cuda", dtype=torch.uint8, generator=g)
    ks = torch.randint(122, 130, (nk, 4), device="cuda", dtype=torch.uint8, generator=g)
    bases = [sum(pads[:i]) for i in range(len(reqs))]
    return q, q_scale, w16, k8, ks.view(torch.int32).view(nk), bases


def _stock_topk(q, q_scale, w16, k8, k_sf, t0, rows, kv, base):
    ks = torch.zeros(rows, dtype=torch.int32, device="cuda")
    ke = kv - rows + 1 + torch.arange(rows, dtype=torch.int32, device="cuda")
    logits = fp8_fp4_mqa_logits(
        (q[t0 : t0 + rows].view(torch.int8), q_scale[t0 : t0 + rows]),
        (k8[base : base + kv].view(torch.int8), k_sf[base : base + kv]),
        w16[t0 : t0 + rows], ks, ke, kv,
        schedule_meta=get_mqa_logits_metadata(ks, ke, kv, 32),
    )[:, :kv]
    idx = torch.arange(kv, device="cuda")
    comp = torch.where(
        idx[None, :] < ke[:, None],
        _order16(logits.view(torch.int16)) * (1 << 20) + ((1 << 20) - 1 - idx),
        -1,
    )
    top = torch.topk(comp, min(K, kv), dim=1)
    return torch.where(top.values >= 0, top.indices, 1 << 40).sort(dim=1).values


@pytest.mark.parametrize(
    "reqs,zero_rows,recover_all",
    [
        # short rows (all keys kept), rows across the CAP boundary, long rows
        ([(37, 1000), (300, 9000), (9, 2100), (40, 40000), (3, 140000)], (), False),
        # all-tied rows (zero head weights): flagged, recomputed by the full path
        ([(64, 20000)], tuple(range(10, 20)), False),
        # every row through the full path (few items: key-split scoring)
        ([(37, 1000), (21, 9000), (40, 40000)], (), True),
        # more items than SMs: whole items, row bounds in the score kernel
        ([(1800, 3000)], (), True),
    ],
)
def test_selective_topk_matches_stock(reqs, zero_rows, recover_all):
    q, q_scale, w16, k8, k_sf, bases = _inputs(reqs, zero_rows)
    sp = pidx.SelectivePlan(
        [r for r, _ in reqs], [kv for _, kv in reqs], bases, "cuda"
    )
    out = torch.empty(sp.t, K, dtype=torch.int32, device="cuda")
    row_bin, _ = pidx.selective_topk(
        sp, q, q_scale, k8, k_sf, w16, out, recover_all=recover_all
    )
    flagged = row_bin[1 : 3 * sp.t : 3] != 0
    assert not flagged[[i for i in range(sp.t) if i not in zero_rows]].any()
    if zero_rows:
        assert flagged[list(zero_rows)].all()
    t0 = 0
    for (rows, kv), base in zip(reqs, bases):
        ref = _stock_topk(q, q_scale, w16, k8, k_sf, t0, rows, kv, base)
        got = out[t0 : t0 + rows].to(torch.int64)
        got = torch.where(got >= 0, got, 1 << 40).sort(dim=1).values[:, : ref.shape[1]]
        ok = (got == ref).all(1)
        assert ok.all(), f"request ({rows}, {kv}): {int((~ok).sum())} rows differ"
        t0 += rows


@pytest.mark.parametrize(
    "reqs,zero_keys",
    [
        # fresh prefills (row i sees i + 1 keys): kv_len < top-k, = top-k, top-k + 1
        ([(2047, 2047)], 0),
        ([(2048, 2048)], 0),
        ([(2049, 2049)], 0),
        # prefix hits (1024 cached + new rows): kv_len < top-k, = top-k, top-k + 1
        ([(1023, 2047)], 0),
        ([(1024, 2048)], 0),
        ([(1025, 2049)], 0),
        # a zero-key prefix (zero KV, as a bench's restored cache): ties at logit 0
        # in the keep-all rows, and past top-k more threshold ties than TIE_RANK
        # (the tile path)
        ([(1024, 2048)], 1024),
        ([(256, 4096)], 1800),
        ([(37, 1000), (300, 9000), (9, 2100), (40, 40000)], 0),
    ],
)
def test_selective_topk_reproducible(reqs, zero_keys):
    """Bit-identical output (including the order) on every call: the sparse MLA's
    online softmax depends on the key order, so an order that varies between calls
    makes the prefill forward non-reproducible (BUG-BENCH-3)."""
    q, q_scale, w16, k8, k_sf, bases = _inputs(reqs)
    for base in bases:
        k8[base : base + zero_keys] = 0
    sp = pidx.SelectivePlan(
        [r for r, _ in reqs], [kv for _, kv in reqs], bases, "cuda"
    )
    outs = []
    for _ in range(4):
        out = torch.full((sp.t, K), -7, dtype=torch.int32, device="cuda")
        pidx.selective_topk(sp, q, q_scale, k8, k_sf, w16, out)
        outs.append(out)
    for i, out in enumerate(outs[1:], 1):
        same = (out == outs[0]).all(1)
        assert same.all(), f"call {i}: {int((~same).sum())} of {sp.t} rows differ in order"
    t0 = 0
    for (rows, kv), base in zip(reqs, bases):
        ref = _stock_topk(q, q_scale, w16, k8, k_sf, t0, rows, kv, base)
        got = outs[0][t0 : t0 + rows].to(torch.int64)
        assert (got >= -1).all()
        got = torch.where(got >= 0, got, 1 << 40).sort(dim=1).values
        assert (got[:, ref.shape[1] :] == 1 << 40).all()
        got = got[:, : ref.shape[1]]
        ok = (got == ref).all(1)
        assert ok.all(), f"request ({rows}, {kv}): {int((~ok).sum())} rows differ"
        t0 += rows


def _tie_inputs(reqs, seed=0, counts=(1500, 1000, 4500)):
    """Keys c * e_0 (MXFP4 c in {0, 3, 4, 6}) and one positive head: every row's
    logit is a function of c only. Per request: 1500 keys c = 6, 1000 keys c = 4 (the
    top-k boundary value: 548 of its 1000 ties are taken), 4500 keys c = 3 (so the
    sampled PUBLISH threshold lands at c = 3), the rest c = 0, at random positions
    over the whole row (ties in older, boundary and newer position bins).
    ``counts`` overrides the (above, boundary, below) key counts."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    t = sum(r for r, _ in reqs)
    index_q = torch.randn(t, 32, 128, device="cuda", dtype=torch.bfloat16, generator=g)
    index_q[:, 0, :2] = 2.0
    index_w = torch.zeros(t, 32, device="cuda")
    index_w[:, 0] = 1.0
    q, q_scale, w = mxfp4_q_quant(index_q, index_w, (128**-0.5) * (32**-0.5), 32)
    _, w16 = mqa_logits_mx_inputs(q.view(torch.int8), q_scale, w)
    pads = [(kv + 127) // 128 * 128 for _, kv in reqs]
    nk = sum(pads)
    k8 = torch.zeros(nk, 64, device="cuda", dtype=torch.uint8)
    ks = torch.full((nk, 4), 127, device="cuda", dtype=torch.uint8)
    bases = [sum(pads[:i]) for i in range(len(reqs))]
    for (_, kv), base in zip(reqs, bases):
        perm = torch.randperm(kv, device="cuda", generator=g)
        a, b, c = counts[0], counts[0] + counts[1], sum(counts)
        for code, (lo, hi) in ((7, (0, a)), (6, (a, b)), (5, (b, c))):
            # FP4 e2m1 code in both nibbles of byte 0 (elements 0 and 1)
            k8[base + perm[lo:hi], 0] = code | (code << 4)
    return q, q_scale, w16, k8, ks.view(torch.int32).view(nk), bases


@pytest.mark.parametrize(
    "reqs,counts",
    [
        # rows longer than CS2_BINS 128-key tiles (262,144 keys): 256-key position bins
        ([(48, 300000)], (1500, 1000, 4500)),
        ([(16, 262144), (16, 262145), (16, 319488)], (1500, 1000, 4500)),
        # past the 256-key bitmap: still flagged, recomputed by the full path
        # (PrefillFinish: 1000 boundary ties > TIE_BUF -> radix tie select)
        ([(16, 330000)], (1500, 1000, 4500)),
        # 600K rows: 3000 boundary ties, 1048 taken (radix tie select)
        ([(8, 600000)], (1000, 3000, 1000)),
        # bound bin > CAND: PrefillExact (chunk-parallel threshold + key-order ties)
        ([(8, 600000)], (1500, 6000, 1500)),
        # boundary ties <= TIE_BUF: shared-memory tie rank
        ([(8, 600000)], (1900, 300, 1500)),
    ],
)
def test_selective_topk_long_row_ties(reqs, counts):
    q, q_scale, w16, k8, k_sf, bases = _tie_inputs(reqs, counts=counts)
    sp = pidx.SelectivePlan(
        [r for r, _ in reqs], [kv for _, kv in reqs], bases, "cuda"
    )
    outs = []
    for _ in range(3):
        out = torch.full((sp.t, K), -7, dtype=torch.int32, device="cuda")
        row_bin, _ = pidx.selective_topk(sp, q, q_scale, k8, k_sf, w16, out)
        outs.append(out)
        flagged = (row_bin[1 : 3 * sp.t : 3] != 0).cpu()
    for i, out in enumerate(outs[1:], 1):
        assert (out == outs[0]).all(), f"call {i}: output order differs"
    t0 = 0
    for (rows, kv), base in zip(reqs, bases):
        ref = _stock_topk(q, q_scale, w16, k8, k_sf, t0, rows, kv, base)
        got = outs[0][t0 : t0 + rows].to(torch.int64)
        got = torch.where(got >= 0, got, 1 << 40).sort(dim=1).values[:, : ref.shape[1]]
        ok = (got == ref).all(1)
        assert ok.all(), f"request ({rows}, {kv}): {int((~ok).sum())} rows differ"
        lens = kv - rows + 1 + torch.arange(rows)
        settled = lens <= pidx.TIE_LONG_KEYS
        assert not flagged[t0 : t0 + rows][settled].any(), (
            f"request ({rows}, {kv}): rows <= TIE_LONG_KEYS flagged"
        )
        assert flagged[t0 : t0 + rows][~settled].all()
        t0 += rows


@pytest.mark.parametrize("recover_all", [False, True])
def test_selective_topk_long_rows_random(recover_all):
    """600K-key rows of random keys: the flagged rows' bound bin overflows CAND
    (PrefillExact); every row matches stock, run to run identical."""
    reqs = [(16, 600000)]
    q, q_scale, w16, k8, k_sf, bases = _inputs(reqs, seed=1)
    sp = pidx.SelectivePlan(
        [r for r, _ in reqs], [kv for _, kv in reqs], bases, "cuda"
    )
    outs = []
    for _ in range(2):
        out = torch.full((sp.t, K), -7, dtype=torch.int32, device="cuda")
        pidx.selective_topk(sp, q, q_scale, k8, k_sf, w16, out, recover_all=recover_all)
        outs.append(out)
    assert (outs[1] == outs[0]).all()
    ref = _stock_topk(q, q_scale, w16, k8, k_sf, 0, 16, 600000, bases[0])
    got = outs[0].to(torch.int64)
    got = torch.where(got >= 0, got, 1 << 40).sort(dim=1).values[:, : ref.shape[1]]
    ok = (got == ref).all(1)
    assert ok.all(), f"{int((~ok).sum())} rows differ"


@pytest.mark.parametrize(
    "reqs,zero_rows",
    [
        # ShardQ-like: two 128-row pieces per request over a long cached prefix
        ([(128, 30000), (128, 31792)] * 2, ()),
        # mixed lengths, a short (keep-all) request and all-tied rows (recovered)
        ([(37, 1000), (300, 9000), (64, 20000)], (5, 6, 7)),
    ],
)
def test_split_publish_matches_unsplit(monkeypatch, reqs, zero_rows):
    """Key-tile split PUBLISH units (pub_split 2 / 3: each item's tiles in that many
    ranges, each into its own candidate region copy, staged per segment in range
    order) give the unsplit top-k bit for bit, order included, and the same
    recomputed rows."""
    q, q_scale, w16, k8, k_sf, bases = _inputs(reqs, zero_rows)
    outs, flags = [], []
    for cap in (1, 2, 3):
        monkeypatch.setattr(pidx, "PUB_SPLIT_MAX", cap)
        sp = pidx.SelectivePlan(
            [r for r, _ in reqs], [kv for _, kv in reqs], bases, "cuda"
        )
        assert sp.pub_split <= cap and (cap == 1) == (sp.pub_split == 1)
        out = torch.full((sp.t, K), -7, dtype=torch.int32, device="cuda")
        row_bin, _ = pidx.selective_topk(sp, q, q_scale, k8, k_sf, w16, out)
        outs.append(out)
        flags.append(row_bin[1 : 3 * sp.t : 3].clone())
    for i in (1, 2):
        assert torch.equal(flags[0], flags[i])
        same = (outs[0] == outs[i]).all(1)
        assert same.all(), f"split {(1, 2, 3)[i]}: {int((~same).sum())} rows differ"
    t0 = 0
    for (rows, kv), base in zip(reqs, bases):
        ref = _stock_topk(q, q_scale, w16, k8, k_sf, t0, rows, kv, base)
        got = outs[1][t0 : t0 + rows].to(torch.int64)
        got = torch.where(got >= 0, got, 1 << 40).sort(dim=1).values[:, : ref.shape[1]]
        assert (got == ref).all(1).all(), f"request ({rows}, {kv})"
        t0 += rows


def test_segment_overflow_recomputed(monkeypatch):
    """A segment that reaches its capacity (shrunk here to 256 entries) has the
    row recomputed by the exact full path: those rows still equal stock, the output
    is identical across calls and between split and unsplit PUBLISH, and rows whose
    segments stay below the capacity are settled from their candidates."""
    reqs = [(64, 20000), (37, 1000), (128, 30000)]
    q, q_scale, w16, k8, k_sf, bases = _inputs(reqs)
    monkeypatch.setattr(pidx, "SEG_CAP", 256)
    outs, flags = [], []
    for cap, reps in ((1, 2), (3, 1)):
        monkeypatch.setattr(pidx, "PUB_SPLIT_MAX", cap)
        sp = pidx.SelectivePlan(
            [r for r, _ in reqs], [kv for _, kv in reqs], bases, "cuda"
        )
        for _ in range(reps):
            out = torch.full((sp.t, K), -7, dtype=torch.int32, device="cuda")
            row_bin, _ = pidx.selective_topk(sp, q, q_scale, k8, k_sf, w16, out)
            outs.append(out)
            flags.append((row_bin[1 : 3 * sp.t : 3] != 0).cpu())
    for i in (1, 2):
        assert torch.equal(flags[0], flags[i])
        assert torch.equal(outs[0], outs[i]), f"output {i} differs from output 0"
    # 20000 / 30000-key rows: ~4096 candidates, ~1024 per segment > 256
    assert flags[0][:64].all() and flags[0][101:].all()
    # 1000-key rows keep every key: 250 per segment
    assert not flags[0][64:101].any()
    t0 = 0
    for (rows, kv), base in zip(reqs, bases):
        ref = _stock_topk(q, q_scale, w16, k8, k_sf, t0, rows, kv, base)
        got = outs[0][t0 : t0 + rows].to(torch.int64)
        got = torch.where(got >= 0, got, 1 << 40).sort(dim=1).values[:, : ref.shape[1]]
        ok = (got == ref).all(1)
        assert ok.all(), f"request ({rows}, {kv}): {int((~ok).sum())} rows differ"
        t0 += rows
