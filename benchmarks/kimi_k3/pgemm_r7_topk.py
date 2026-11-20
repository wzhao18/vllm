# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P-GEMM R7: prefill router top-k (prefill_glue.router_topk, sigmoid + bias, top-16
of 896, renormalize) standalone at the prefill row counts, warm and THRASH (512 MB L2
flush before every call).

    python3 pgemm_r7_topk.py [--rows 16384,8192] [--iters 50]
"""

import argparse
import json

import torch

from vllm.models.kimi_k3.nvidia.ops import prefill_glue


def timed(fn, iters, flush=None):
    start = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    for i in range(iters):
        if flush is not None:
            flush.zero_()
        start[i].record()
        fn()
        end[i].record()
    torch.cuda.synchronize()
    t = sorted(s.elapsed_time(e) * 1e3 for s, e in zip(start, end))
    return t[len(t) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="16384,8192")
    ap.add_argument("--experts", type=int, default=896)
    ap.add_argument("--hidden", type=int, default=7168)
    ap.add_argument("--topk", type=int, default=16)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(3)
    flush = torch.empty(512 << 20, dtype=torch.uint8, device=dev)
    e = args.experts
    bias = (0.01 * torch.randn(e, device=dev, generator=g)).float()
    w = (0.02 * torch.randn(e, args.hidden, device=dev, generator=g)).to(torch.bfloat16)
    for rows in (int(r) for r in args.rows.split(",")):
        x = torch.randn(rows, args.hidden, device=dev, generator=g).to(torch.bfloat16)
        logits = torch.empty(rows, e, device=dev, dtype=torch.float32)

        logits.copy_(torch.nn.functional.linear(x.float(), w.float()))
        assert prefill_glue.use_router_topk(logits, bias, args.topk, 1, 1, "sigmoid")

        def topk():
            prefill_glue.router_topk(logits, bias, args.topk, True, 2.5)

        res = {"rows": rows, "experts": e, "topk": args.topk}
        res["topk_warm_us"] = round(timed(topk, args.iters), 2)
        res["topk_thrash_us"] = round(timed(topk, args.iters, flush), 2)
        res["logits_MB"] = round(logits.numel() * 4 / 1e6, 1)
        print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
