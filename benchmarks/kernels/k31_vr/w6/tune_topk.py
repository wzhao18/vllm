# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sweep balanced_topk launch parameters (per-kernel CUPTI us)."""
import itertools
import sys

import torch
from torch.profiler import ProfilerActivity, profile

from vllm.v1.attention.ops.balanced_topk import balanced_topk

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from bench_topk import AGX, UNI, row_lens  # noqa: E402

cases = {**AGX, **UNI}
data = {}
for name in ["r1_u128k", "r16_u128k", "r16_agx"]:
    req = cases[name]
    lens = row_lens(req, 4)
    width = (max(req) + 1023) // 1024 * 1024
    logits = torch.randn(len(lens), width + 512, device="cuda").mul_(4).to(torch.bfloat16)[:, :width]
    seq = torch.tensor(lens, dtype=torch.int32, device="cuda").view(-1, 4)
    out = torch.empty(len(lens), 2048, dtype=torch.int32, device="cuda")
    data[name] = (logits, seq, out)
grid = list(itertools.product([4, 8, 16], [1024, 2048, 4096], [4, 8], [(16384, 32), (16384, 16), (4096, 8)]))
for pps, block, fw, (sub, sw) in grid:
    if block // (32 * fw) > 16:
        continue
    row = []
    for name, (logits, seq, out) in data.items():
        kw = dict(programs_per_sm=pps, block=block, filter_warps=fw, sub=sub, select_warps=sw)
        for _ in range(2):
            balanced_topk(logits, seq, 4, out, 2048, **kw)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            for _ in range(10):
                balanced_topk(logits, seq, 4, out, 2048, **kw)
            torch.cuda.synchronize()
        t = {e.key: e.self_device_time_total / e.count for e in p.key_averages() if e.self_device_time_total}
        row.append(f"{name}: s={t.get('_sample_threshold',0):.1f} f={t.get('_filter',0):.1f} sel={t.get('_select',0):.1f}")
    print(f"pps={pps} block={block} fw={fw} sub={sub} sw={sw} | " + " | ".join(row), flush=True)
