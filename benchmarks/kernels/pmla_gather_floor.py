# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gather floor of prefill sparse MLA: every token reads its 2048 top-k KV rows
(576 B FP8) once, no math (XOR-reduce to keep the loads live). This is the
L2 -> SM traffic any per-token sparse MLA kernel must move (no cross-token reuse
of KV rows). Reports the achieved L2->SM bandwidth and time per F16K layer.
Also times a contiguous-row streaming read of the same byte count (no gather).
"""

import argparse
import json
import statistics
import sys

import torch
import triton
import triton.language as tl

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from benchmark_kimi_k3_prefill_sparse_mla import make_layer  # noqa: E402


@triton.jit
def _gather_kernel(kv_ptr, idx_ptr, out_ptr, TOPK: tl.constexpr, ROWS: tl.constexpr,
                   WORDS: tl.constexpr, WPAD: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    acc = tl.zeros((ROWS, WPAD), dtype=tl.int64)
    w = tl.arange(0, WPAD)
    for k0 in range(0, TOPK, ROWS):
        r = tl.load(idx_ptr + t * TOPK + k0 + tl.arange(0, ROWS)).to(tl.int64)
        v = tl.load(kv_ptr + r[:, None] * WORDS + w[None, :], mask=w[None, :] < WORDS, other=0)
        acc = acc ^ v
    tl.store(out_ptr + t, tl.reduce(tl.reshape(acc, (ROWS * WPAD,)), 0, _xor))


@triton.jit
def _xor(a, b):
    return a ^ b


def med(fn, iters=7):
    s = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    e = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        s[i].record()
        fn(i)
        e[i].record()
    torch.cuda.synchronize()
    return statistics.median(a.elapsed_time(b) * 1e3 for a, b in zip(s, e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=16384)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    dev = torch.device("cuda")
    F = a.tokens
    layers = [make_layer(F, 72900, 3, dev, 17 + i) for i in range(3)]
    out = torch.empty(F, dtype=torch.int64, device=dev)
    res = {"tokens": F}
    for rows in (32, 64):
        for warps in (4, 8):
            def f(i, rows=rows, warps=warps):
                kv, idx = layers[i % 3]
                _gather_kernel[(F,)](kv.view(torch.int64), idx, out, TOPK=2048, ROWS=rows,
                                     WORDS=72, WPAD=128, num_warps=warps)
            f(0)
            t = med(f)
            res[f"gather_r{rows}_w{warps}_us"] = round(t, 1)
    best = min(v for k, v in res.items() if k.startswith("gather"))
    nbytes = F * 2048 * 576
    res["bytes"] = nbytes
    res["best_us"] = best
    res["best_TBps"] = round(nbytes / best / 1e6, 2)
    # contiguous stream of the same bytes from a pool that does not fit L2 well
    big = torch.empty(nbytes // 8, dtype=torch.int64, device=dev) if nbytes < 24e9 else None
    if big is not None:
        t = med(lambda i: big.sum())
        res["dram_stream_us"] = round(t, 1)
        res["dram_stream_TBps"] = round(nbytes / t / 1e6, 2)
    print(json.dumps(res, indent=1))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
