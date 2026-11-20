# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbench: sdec route_stage (router top-k + MegaMoE staging, one launch) vs
stock fused_grouped_topk + stage_inputs, at the SP decode shard rows.

Timing: CUDA graph of REP back-to-back launches per arm, replayed ITERS times,
cuda events; reports us per launch (median of SAMPLES)."""

import argparse
import statistics

import torch

E, TOPK, HIDDEN = 896, 16, 3584


def bufs(capacity, nvfp4, device):
    vec = 16 if nvfp4 else 32
    x = torch.zeros((capacity, HIDDEN // 2 if nvfp4 else HIDDEN), dtype=torch.uint8, device=device)
    sf = torch.zeros((capacity, HIDDEN // vec + 4), dtype=torch.uint8, device=device)
    idx = torch.zeros((capacity, TOPK), dtype=torch.int32, device=device)
    sw = torch.zeros((capacity, TOPK), dtype=torch.float32, device=device)
    err = torch.zeros(2, dtype=torch.int64, device=device)
    return [x, sf, idx, sw, err]


def timeit(fn, rep, iters, samples):
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        for _ in range(rep):
            fn()
    out = []
    for _ in range(samples):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(iters):
            g.replay()
        b.record()
        torch.cuda.synchronize()
        out.append(a.elapsed_time(b) * 1e3 / (iters * rep))
    return statistics.median(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", default="32,64,96")
    p.add_argument("--capacity", type=int, default=384)
    p.add_argument("--rep", type=int, default=24)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--samples", type=int, default=7)
    p.add_argument("--quant", default="nvfp4")
    args = p.parse_args()

    from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
        fused_grouped_topk,
    )
    from vllm.models.kimi_k3.nvidia.megamoe_rubin.staging import stage_inputs
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe.route_stage import (
        _module,
    )

    dev = torch.device("cuda")
    nvfp4 = args.quant == "nvfp4"
    mod = _module()
    print(f"{'n':>4} {'route_stage split':>18} {'concat':>8} {'stock topk':>11} {'stock stage':>12} {'stock sum':>10}")
    for n in [int(r) for r in args.rows.split(",")]:
        torch.manual_seed(n)
        logits = torch.randn(n, E, device=dev) * 3
        bias = torch.randn(E, device=dev) * 0.1
        latent = (torch.randn(n, HIDDEN, device=dev) * 4).to(torch.bfloat16)
        y = torch.cat([logits, latent.float()], dim=1).contiguous()
        o = bufs(args.capacity, nvfp4, dev)
        r = bufs(args.capacity, nvfp4, dev)
        st = {}

        def split():
            mod.route_stage(logits, E, bias, o[0], o[1], o[2], o[3], o[4], None, E,
                            True, 2.827, nvfp4, 2.5, latent)

        def concat():
            mod.route_stage(y, E, bias, o[0], o[1], o[2], o[3], o[4], None, E,
                            True, 2.827, nvfp4, 2.5, None)

        def topk():
            st["w"], st["i"] = fused_grouped_topk(latent, logits, TOPK, True, bias, 1, 1,
                                                  "sigmoid", 2.827)

        topk()

        def stage():
            stage_inputs(latent, st["i"], st["w"], r[0], r[1], r[2], r[3], route_error=r[4],
                         quant_kind=args.quant, num_experts=E, input_norm_const=2.5)

        t = [timeit(f, args.rep, args.iters, args.samples) for f in (split, concat, topk, stage)]
        print(f"{n:>4} {t[0]:>18.2f} {t[1]:>8.2f} {t[2]:>11.2f} {t[3]:>12.2f} {t[2] + t[3]:>10.2f}")


if __name__ == "__main__":
    main()
