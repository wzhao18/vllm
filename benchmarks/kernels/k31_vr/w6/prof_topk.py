# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-kernel CUPTI times of balanced_topk on a few AgentX cases."""
import sys

import torch
from torch.profiler import ProfilerActivity, profile

from vllm.v1.attention.ops.balanced_topk import balanced_topk

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from bench_topk import AGX, UNI, indexer_logits, row_lens  # noqa: E402

cases = {**AGX, **UNI}
for name in sys.argv[1:] or ["r1_u128k", "r16_u128k", "r16_agx"]:
    req = cases[name]
    next_n = 4
    lens = row_lens(req, next_n)
    width = (max(req) + 1023) // 1024 * 1024
    logits = indexer_logits(req, next_n, width + 512, "cuda")[:, :width]
    seq = torch.tensor(lens, dtype=torch.int32, device="cuda").view(-1, next_n)
    out = torch.empty(len(lens), 2048, dtype=torch.int32, device="cuda")
    for _ in range(3):
        balanced_topk(logits, seq, next_n, out, 2048)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(20):
            balanced_topk(logits, seq, next_n, out, 2048)
        torch.cuda.synchronize()
    print("==", name)
    for e in p.key_averages():
        if e.device_type.name == "CUDA" or getattr(e, "self_device_time_total", 0):
            t = getattr(e, "device_time", None) or e.self_device_time_total / max(e.count, 1)
            print(f"  {e.key[:60]:60s} n={e.count:4d} avg_us={t:8.2f}")
