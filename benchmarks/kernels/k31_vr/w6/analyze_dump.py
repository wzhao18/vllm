# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay dumped decode logits (VLLM_W6_DUMP_LOGITS) through balanced_topk:
per-row path taken (fast / ties / exact fallback), value distribution, and
timing vs the stock upcast + cooperative top-k on the same real logits."""
import glob
import sys

import torch

import vllm.v1.attention.ops.balanced_topk as bt
from vllm.model_executor.layers.indexer_topk import RADIX_TOPK_WORKSPACE_SIZE
from vllm.model_executor.layers.sparse_attn_indexer import as_fp32_logits


def keys_of(x):
    u = x.view(torch.int16).to(torch.int32) & 0xFFFF
    return torch.where(u >= 0x8000, u ^ 0xFFFF, u | 0x8000)


def time_it(fn, reps=20, iters=10):
    fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); g.replay(); e.record(); e.synchronize()
        ts.append(s.elapsed_time(e) * 1000 / reps)
    return sorted(ts)[len(ts) // 2]


def sources():
    if sys.argv[1] == "synthetic":
        sys.path.insert(0, __file__.rsplit("/", 1)[0])
        from bench_topk import AGX, indexer_logits
        torch.manual_seed(0)
        for name in sys.argv[2:] or ["r16_agx"]:
            req = AGX[name]
            width = (max(req) + 1023) // 1024 * 1024
            logits = indexer_logits(req, 4, width + 512, "cuda")[:, :width]
            seq = torch.tensor(req, dtype=torch.int32, device="cuda").view(-1, 1)
            yield name, {"logits": logits, "seq_lens": seq, "next_n": 4}
    else:
        for f in sorted(glob.glob(sys.argv[1] + "/logits_*.pt")):
            yield f, torch.load(f)


topk = 2048
for f, d in sources():
    logits = d["logits"].cuda()
    seq = d["seq_lens"].cuda().to(torch.int32)
    next_n = d["next_n"]
    rows, width = logits.shape
    per_row = seq.numel() == rows
    if per_row:
        lens = seq.view(-1).tolist()
    else:
        lens = [max(0, int(seq.view(-1)[r // next_n]) - next_n + 1 + r % next_n) for r in range(rows)]
    print(f"== {f}: rows {rows} width {width} seq {tuple(seq.shape)} next_n {next_n} max_len {max(lens)}")
    paths = {}
    for r in range(rows):
        n = min(lens[r], width)
        if n <= topk:
            paths["short"] = paths.get("short", 0) + 1
            continue
        k = keys_of(logits[r, :n])
        if n <= bt.SAMPLES:
            t0 = int(k.sort(descending=True).values[topk - 1])
        else:
            i = torch.arange(bt.SAMPLES, device="cuda")
            pos = (2 * i + 1) * n // (2 * bt.SAMPLES)
            need = min(max(-(-bt.TARGET * bt.SAMPLES // n), 1), bt.SAMPLES)
            t0 = int(k[pos].sort(descending=True).values[need - 1])
        gt = int((k > t0).sum()); eq = int((k == t0).sum())
        if gt >= topk and (n <= bt.CAP or gt <= bt.CAP):
            kth = int(k.sort(descending=True).values[topk - 1])
            p = "fast" if kth - t0 - 1 < 255 else "gap"
        elif gt + eq >= topk:
            p = "ties"
        else:
            p = "exact"
        paths[p] = paths.get(p, 0) + 1
        if r < 8 or p in ("exact",):
            x = logits[r, :n].float()
            print(f"  row {r} n {n} t0 {t0:#06x} gt {gt} eq {eq} path {p} | min {x.min():.3g} "
                  f"med {x.median():.3g} max {x.max():.3g} zeros {(x == 0).sum().item()} "
                  f"distinct {torch.unique(logits[r,:n]).numel()}")
    print("  paths:", paths)
    out = torch.empty(rows, topk, dtype=torch.int32, device="cuda")
    bt.balanced_topk(logits, seq, next_n, out, topk)
    ok = True
    for r in range(rows):
        n = min(lens[r], width); kk = min(n, topk)
        sel = out[r, :kk].long()
        if n > topk and not torch.equal(logits[r, sel].float().sort().values,
                                        logits[r, :n].float().topk(kk).values.sort().values):
            ok = False
    print("  balanced exact:", ok)
    ws = torch.empty(RADIX_TOPK_WORKSPACE_SIZE, dtype=torch.uint8, device="cuda")
    row_ends = torch.tensor(lens, dtype=torch.int32, device="cuda")
    seq2 = row_ends.view(-1, next_n) if not per_row else seq
    def stock():
        f32 = as_fp32_logits(logits, row_ends)
        torch.ops._C.cooperative_topk(f32, seq2, out, ws, topk, max(lens))
    print(f"  stock(upcast+coop) {time_it(stock):.1f} us  balanced {time_it(lambda: bt.balanced_topk(logits, seq, next_n, out, topk)):.1f} us")
