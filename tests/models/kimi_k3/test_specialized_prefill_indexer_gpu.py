# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 sm_107 selective prefill indexer. The head sum runs on the tensor
core (BF16 products, FP32 accumulation), so the logits are at least as precise as
the stock (DeepGEMM) BF16-chain logits and may differ in the last BF16 bit. Checks:
the selective top-2048 of every row equals the exact top-k of the kernel's own
logits (the full path, also for the rows the thresholds cannot settle), and it
agrees with the stock top-k up to keys at the boundary."""

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
        # every row through the full path
        ([(37, 1000), (21, 9000), (40, 40000)], (), True),
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
    # exact: the selective result equals the full path over the same logits
    full = torch.empty_like(out)
    pidx.selective_topk(sp, q, q_scale, k8, k_sf, w16, full, recover_all=True)

    def rows_of(x):
        x = x.to(torch.int64)
        return torch.where(x >= 0, x, 1 << 40).sort(dim=1).values

    assert torch.equal(rows_of(out), rows_of(full))
    t0 = 0
    for (rows, kv), base in zip(reqs, bases):
        ref = _stock_topk(q, q_scale, w16, k8, k_sf, t0, rows, kv, base)
        got = rows_of(out[t0 : t0 + rows])[:, : ref.shape[1]]
        # same set as stock up to boundary keys whose BF16 logit moved by the
        # more precise head sum
        n = ref.shape[1]
        valid = [set(x for x in ref[i].tolist() if x < (1 << 40)) for i in range(rows)]
        miss = torch.tensor(
            [len(valid[i] - set(got[i].tolist())) for i in range(rows)]
        )
        assert (miss <= max(16, n // 50)).all(), (
            f"request ({rows}, {kv}): up to {int(miss.max())} of {n} keys differ"
        )
        t0 += rows


def test_logits_vs_stock():
    """The kernel logits (full path, STORE mode) against the stock DeepGEMM BF16
    logits: equal up to the last BF16 bit (the head sum is more precise)."""
    rows, kv = 64, 3000
    q, q_scale, w16, k8, k_sf, bases = _inputs([(rows, kv)])
    items, row_len, row_base, scratch = pidx.plan([rows], [kv], bases, "cuda")
    logits = torch.zeros(scratch, dtype=torch.bfloat16, device="cuda")
    row_bin = torch.zeros(3 * (rows + 1), dtype=torch.int32, device="cuda")
    pidx.score(q, q_scale, k8, k_sf, w16, row_len, items, row_base, logits, row_bin,
               mode=pidx.STORE)
    ks = torch.zeros(rows, dtype=torch.int32, device="cuda")
    ke = kv - rows + 1 + torch.arange(rows, dtype=torch.int32, device="cuda")
    ref = fp8_fp4_mqa_logits(
        (q.view(torch.int8), q_scale), (k8[:kv].view(torch.int8), k_sf[:kv]),
        w16, ks, ke, kv, schedule_meta=get_mqa_logits_metadata(ks, ke, kv, 32),
    )[:, :kv].float()
    pitch = (kv + 127) // 128 * 128
    got = logits[: rows * pitch].view(rows, pitch // 128, 128)
    p = torch.arange(128, device="cuda")
    perm = 4 * (p % 32) + p // 32  # MMA-row order: position p holds key perm[p]
    out = torch.empty_like(got)
    out[:, :, perm] = got
    out = out.view(rows, pitch)[:, :kv]
    mask = torch.arange(kv, device="cuda")[None, :] < ke[:, None]
    ref16 = ref.to(torch.bfloat16)
    ulp = (_order16(out.view(torch.int16)) - _order16(ref16.view(torch.int16))).abs()[mask]
    hist = torch.bincount(ulp.clamp(max=8)).tolist()
    print(f"ulp distance to stock (0..8+): {hist}")
    # the stock BF16 chain rounds after every head, the tensor-core sum once: the
    # two agree to a few BF16 ulps
    assert float((ulp == 0).float().mean()) > 0.3
    # absolute error against the row's largest logit (near-zero sums cancel, so
    # their ulp distance is not meaningful)
    refm = torch.where(mask, ref, 0.0)
    rowmax = refm.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
    rel = ((out.float() - refm).abs() / rowmax)[mask]
    print(f"max error / row max {float(rel.max()):.3e}")
    assert float(rel.max()) < 2**-6
