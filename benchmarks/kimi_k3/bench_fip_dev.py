# SPDX-License-Identifier: Apache-2.0
"""In-process A/B of persistent-KDA prefill constexpr variants (dev knobs).

Every variant is a dict of kernel constexpr overrides (forward._DEV). The
first variant is the reference: each other variant is checked bitwise against
it (output and final state), then all variants are timed with CUDA graphs in
alternating rounds on the same inputs.

  python3 benchmarks/kimi_k3/bench_fip_dev.py --lens 5461,5461,5462 \
      --variants '{}' '{"VLDG_": 1}' --rounds 4
"""
import argparse
import json

import torch

import vllm.models.kimi_k3.nvidia.specialized_prefill.kda_persistent.forward as fwd_mod
from vllm.models.kimi_k3.nvidia.specialized_prefill import kda as sp_kda

D = 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", nargs="+", default=["5461,5461,5462"])
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--variants", nargs="+", default=["{}"])
    a = ap.parse_args()
    variants = [json.loads(v) for v in a.variants]
    for lens_s in a.lens:
        one_shape(a, lens_s, variants)


def one_shape(a, lens_s, variants):
    lens = [int(x) for x in lens_s.split(",")]
    torch.manual_seed(0)
    T, N, H = sum(lens), len(lens), 96
    bf = torch.bfloat16
    offs = [0, *torch.tensor(lens).cumsum(0).tolist()]
    cu_cpu = torch.tensor(offs, dtype=torch.int32)
    cu = cu_cpu.cuda()
    q, k, g = (torch.randn(1, T, H, D, device="cuda").to(bf) for _ in range(3))
    v = (torch.randn(1, T, H, D, device="cuda") * 0.5).to(bf)
    beta = torch.randn(1, T, H, device="cuda").to(bf)
    A_log = torch.randn(H, device="cuda") * 0.5
    dt_bias = torch.randn(H * D, device="cuda") * 0.1
    cache0 = torch.randn(64, H, D, D, device="cuda") * 0.1
    idx = torch.tensor([5, 17, 40, 41, 50, 60], dtype=torch.int32, device="cuda")[:N]
    has = torch.ones(N, dtype=torch.bool, device="cuda")

    def run(cache):
        return sp_kda.persistent_prefill(
            q, k, v, g, beta, A_log, dt_bias, -5.0, None, cu, cu_cpu,
            state_cache=cache, state_indices=idx, has_initial_state=has,
        )

    ref = None
    for i, var in enumerate(variants):
        fwd_mod._DEV = dict(var)
        cache = cache0.clone()
        o, _ = run(cache)
        torch.cuda.synchronize()
        res = (o.clone(), cache[idx.long()].clone())
        if ref is None:
            ref = res
            continue
        ob = torch.equal(res[0], ref[0])
        sb = torch.equal(res[1], ref[1])
        md = (res[0].float() - ref[0].float()).abs().max().item()
        sd = (res[1] - ref[1]).abs().max().item()
        print(f"lens={lens_s} v{i}={var} bitequal out={ob} state={sb} "
              f"out_maxdiff={md:.3e} state_maxdiff={sd:.3e}", flush=True)

    cache = cache0.clone()
    graphs = []
    for var in variants:
        fwd_mod._DEV = dict(var)
        for _ in range(3):
            run(cache)
        torch.cuda.synchronize()
        gph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gph):
            run(cache)
        graphs.append(gph)
    med = [[] for _ in variants]
    for r in range(a.rounds):
        order = range(len(variants)) if r % 2 == 0 else reversed(range(len(variants)))
        for i in order:
            ts = []
            for _ in range(a.iters):
                e0, e1 = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                e0.record()
                graphs[i].replay()
                e1.record()
                e1.synchronize()
                ts.append(e0.elapsed_time(e1) * 1e3)
            ts.sort()
            med[i].append(ts[len(ts) // 2])
    for i, var in enumerate(variants):
        m = sorted(med[i])
        print(f"lens={lens_s} v{i}={json.dumps(var)} graph us median-of-rounds "
              f"{m[len(m) // 2]:.1f} rounds {' '.join(f'{x:.1f}' for x in med[i])}",
              flush=True)


if __name__ == "__main__":
    main()
