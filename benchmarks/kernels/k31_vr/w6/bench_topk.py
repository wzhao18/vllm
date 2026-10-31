# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""W6 M1: DEP decode indexer top-k at AgentX per-rank shapes (one GPU).

Arms (us per call, one DSA layer):
- ``stock``: BF16 -> FP32 upcast (as_fp32_logits) + cooperative_topk, the
  stock SM10x DCP1 decode path under topk backend auto (<= 64 rows).
- ``persistent`` / ``per_row``: upcast + the other stock kernels (> 64 rows
  goes to persistent under auto).
- ``balanced``: balanced_topk on the BF16 logits.

Each arm is captured as one CUDA graph of --reps calls (logits stay L2-warm,
as after the logits kernel), median over --iters replays. Also checks that
balanced selects the same values as torch.topk on every case.
"""

import argparse
import json

import torch

from vllm.model_executor.layers.indexer_topk import RADIX_TOPK_WORKSPACE_SIZE
from vllm.model_executor.layers.sparse_attn_indexer import as_fp32_logits
from vllm.v1.attention.ops.balanced_topk import balanced_topk
from vllm.v1.attention.ops.balanced_topk_cuda import balanced_topk_cuda

AGX = {
    "r4_agx": [48128, 114688, 187392, 346112],
    "r8_agx": [20480, 66560, 98304, 132096, 166912, 211968, 281600, 516096],
    "r16_agx": [4096, 37888, 57344, 74752, 90112, 106496, 122880, 140288, 157696,
                176128, 198656, 226304, 261120, 307200, 403456, 659456],
}
AGX["r16_agx70"] = [int(x * 0.7) // 1024 * 1024 for x in AGX["r16_agx"]]
UNI = {"r1_u128k": [131072], "r8_u32k": [32768] * 8, "r8_u128k": [131072] * 8,
       "r16_u64k": [65536] * 16, "r16_u128k": [131072] * 16, "r8_u256k": [262144] * 8,
       "r8_tail800k": [65536] * 7 + [819200]}


def indexer_logits(req, next_n, stride, dev, heads=64, dim=128):
    """Logits shaped like the DSA indexer's: sum_h w_h * relu(q_h . k_t)
    over random FP8 keys (the W0 bench fill: N(0, 0.5) in E4M3), random
    per-row queries and head weights, rounded to BF16 like DeepGEMM's."""
    rows = sum(1 for _ in req) * next_n
    out = torch.zeros(rows, stride, dtype=torch.bfloat16, device=dev)
    for b, L in enumerate(req):
        k = (torch.randn(L, dim, device=dev) * 0.5).to(torch.float8_e4m3fn).float()
        for j in range(next_n):
            q = torch.randn(heads, dim, device=dev) / dim**0.5
            w = torch.randn(heads, device=dev) / heads**0.5
            for s in range(0, L, 1 << 17):
                e = min(L, s + (1 << 17))
                out[b * next_n + j, s:e] = (torch.relu(k[s:e] @ q.T) @ w).to(torch.bfloat16)
    return out


def row_lens(req, next_n):
    return [max(0, L - next_n + 1 + j) for L in req for j in range(next_n)]


def time_graph(fn, reps, iters):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); g.replay(); e.record(); e.synchronize()
        ts.append(s.elapsed_time(e) * 1000 / reps)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--next-n", type=int, nargs="+", default=[4, 8])
    p.add_argument("--cases", nargs="*", default=None)
    p.add_argument("--arms", nargs="+", default=["stock", "persistent", "balanced"])
    p.add_argument("--topk", type=int, default=2048)
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--iters", type=int, default=15)
    p.add_argument("--programs-per-sm", type=int, nargs="+", default=[8])
    p.add_argument("--out", default=None)
    p.add_argument("--dist", choices=["randn", "indexer"], default="indexer")
    a = p.parse_args()
    cases = {**AGX, **UNI}
    if a.cases:
        cases = {k: cases[k] for k in a.cases}
    dev = torch.device("cuda")
    ws = torch.empty(RADIX_TOPK_WORKSPACE_SIZE, dtype=torch.uint8, device=dev)
    res = []
    for next_n in a.next_n:
        for name, req in cases.items():
            torch.manual_seed(0)
            lens = row_lens(req, next_n)
            rows = len(lens)
            width = (max(req) + 1023) // 1024 * 1024
            # DeepGEMM's padded row stride (1024 B aligned + a split's slack).
            stride = width + 512
            if a.dist == "indexer":
                buf = indexer_logits(req, next_n, stride, dev)
            else:
                buf = torch.randn(rows, stride, device=dev).mul_(4).to(torch.bfloat16)
            logits = buf[:, :width]
            seq_lens = torch.tensor(lens, dtype=torch.int32, device=dev).view(-1, next_n)
            row_ends = seq_lens.view(-1).contiguous()
            out = torch.empty(rows, a.topk, dtype=torch.int32, device=dev)
            maxlen = max(req)
            r = {"case": name, "next_n": next_n, "rows": rows, "mtok": sum(req) / 1e6}

            def stock(kernel):
                def f():
                    f32 = as_fp32_logits(logits, row_ends)
                    getattr(torch.ops._C, kernel)(f32, seq_lens, out, ws, a.topk, maxlen)
                return f

            for arm in a.arms:
                if arm == "stock":
                    if rows > 64:
                        continue
                    r[arm] = time_graph(stock("cooperative_topk"), a.reps, a.iters)
                elif arm == "persistent":
                    r[arm] = time_graph(stock("persistent_topk"), a.reps, a.iters)
                elif arm == "gvr2":
                    from flashinfer.topk_varlen.kernels.gvr2_topk_host import run_varlen, warmup_varlen
                    f32 = as_fp32_logits(logits, row_ends)
                    warmup_varlen(top_k=a.topk, max_seq_len=f32.shape[1], compress_ratio=1, next_n=1,
                                  num_rows_list=[rows], row_stride=f32.stride(0))
                    def g():
                        f = as_fp32_logits(logits, row_ends)
                        run_varlen(f, None, row_ends, out, next_n=1, compress_ratio=1,
                                   max_seq_len=f.shape[1], top_k=a.topk)
                    g(); torch.cuda.synchronize()
                    ts = []
                    for _ in range(a.iters):
                        s0, e0 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        s0.record()
                        for _ in range(a.reps):
                            g()
                        e0.record(); e0.synchronize()
                        ts.append(s0.elapsed_time(e0) * 1000 / a.reps)
                    r[arm + "_eager"] = sorted(ts)[len(ts) // 2]
                elif arm == "bcuda":
                    for pps in a.programs_per_sm:
                        r[f"bcuda_c{pps}"] = time_graph(
                            lambda: balanced_topk_cuda(logits, seq_lens, next_n, out, a.topk, pps),
                            a.reps, a.iters)
                elif arm == "upcast":
                    r[arm] = time_graph(lambda: as_fp32_logits(logits, row_ends), a.reps, a.iters)
                elif arm == "balanced":
                    for pps in a.programs_per_sm:
                        r[f"balanced_p{pps}"] = time_graph(
                            lambda: balanced_topk(logits, seq_lens, next_n, out, a.topk, pps),
                            a.reps, a.iters)
            # correctness of the balanced output on this case
            out.fill_(-5)
            (balanced_topk_cuda if "bcuda" in a.arms else balanced_topk)(
                logits, seq_lens, next_n, out, a.topk)
            ok = True
            for i in range(rows):
                n = lens[i]
                k = min(n, a.topk)
                sel = out[i, :k].long()
                if (out[i, k:] != -1).any() or sel.unique().numel() != k or (sel < 0).any() or (sel >= n).any():
                    ok = False; break
                if n > a.topk:
                    ref = logits[i, :n].float().topk(k).values.sort().values
                    if not torch.equal(logits[i, sel].float().sort().values, ref):
                        ok = False; break
            r["balanced_exact"] = ok
            print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
            res.append(r)
            del buf, logits
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
