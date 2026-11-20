#!/usr/bin/env python3
"""Persistent KDA prefill core: in-kernel q/k L2 norm vs producer-normalized
q/k (QKN route), same GPU, interleaved (A B B A per round), F16K / F8K shapes.

Prints one JSON line per shape: median kernel time per arm (us) per round and
the relative RMS of out / state between the arms (equal up to the norm's
reduction order and the BF16 rounding of the normalized q/k).
"""

import argparse
import json
import statistics

import torch
from pgemm_conv_bench import kernel_times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", default="5461,5461,5461;2731,2731,2731")
    ap.add_argument("--heads", type=int, default=96)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()

    from vllm.models.kimi_k3.nvidia.specialized_prefill import kda as sp_kda

    D, H, bf = 128, args.heads, torch.bfloat16
    for spec in args.lens.split(";"):
        lens = [int(v) for v in spec.split(",")]
        torch.manual_seed(0)
        T, N = sum(lens), len(lens)
        cu_cpu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32)
        cu = cu_cpu.cuda()
        q, k, g = (torch.randn(1, T, H, D, device="cuda").to(bf) for _ in range(3))
        v = (torch.randn(1, T, H, D, device="cuda") * 0.5).to(bf)
        beta = torch.randn(1, T, H, device="cuda").to(bf)
        A_log = torch.randn(H, device="cuda") * 0.5
        dt_bias = torch.randn(H * D, device="cuda") * 0.1
        h0 = torch.randn(N, H, D, D, device="cuda") * 0.1
        qn, kn = (
            (x.float() * torch.rsqrt(x.float().square().sum(-1, keepdim=True) + 1e-6)).to(bf)
            for x in (q, k)
        )

        def run(norm):
            return sp_kda.persistent_prefill(
                qn if norm else q, kn if norm else k, v, g, beta, A_log, dt_bias,
                -5.0, h0, cu, cu_cpu, qk_normalized=norm,
            )  # fmt: skip

        oa, sa = run(False)
        ob, sb = run(True)
        torch.cuda.synchronize()

        def rel(a, b):
            return float((a.float() - b.float()).square().mean().sqrt()
                         / b.float().square().mean().sqrt())  # fmt: skip

        res = {"T": T, "lens": lens, "H": H, "rel_out": rel(ob, oa), "rel_state": rel(sb, sa)}
        ta, tb = [], []
        for _ in range(args.rounds):
            for norm in (False, True, True, False):
                t = kernel_times(lambda n=norm: run(n), iters=args.iters)[0]
                (tb if norm else ta).append(round(t, 1))
        res["in_kernel_norm_us"] = ta
        res["qkn_us"] = tb
        res["delta_us"] = round(statistics.median(tb) - statistics.median(ta), 1)
        print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
