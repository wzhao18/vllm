# SPDX-License-Identifier: Apache-2.0
"""In-process A/B of the persistent prefill KDA kernel: reference copy
(benchmarks/kimi_k3/fip_ref, kda_persistent at 22d9c40693) vs the current
tree, same inputs, in-place state I/O; checks bitwise equality of out and
final state, then ABAB CUDA-graph timing. Usage: run from the repo root."""

import argparse
import importlib.util
import pathlib
import sys

import torch

import vllm.models.kimi_k3.nvidia.specialized_prefill.kda_persistent as pkg
from vllm.models.kimi_k3.nvidia.specialized_prefill import kda as sp_kda

D = 128


def load_ref():
    d = pathlib.Path(__file__).resolve().parent / "fip_ref"
    spec = importlib.util.spec_from_file_location(
        "fip_ref", d / "__init__.py", submodule_search_locations=[str(d)]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fip_ref"] = mod
    spec.loader.exec_module(mod)
    import fip_ref.forward as ref_fwd  # noqa: E402

    return ref_fwd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", default="5461,5461,5462")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--rounds", type=int, default=2)
    a = ap.parse_args()
    lens = [int(x) for x in a.lens.split(",")]
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
    arms = {"new": pkg.forward, "ref": load_ref()}
    res = {}
    for name, mod in arms.items():
        pkg.forward = mod
        cache = cache0.clone()
        o, _ = sp_kda.persistent_prefill(
            q, k, v, g, beta, A_log, dt_bias, -5.0, None, cu, cu_cpu,
            state_cache=cache, state_indices=idx, has_initial_state=has,
        )
        torch.cuda.synchronize()
        res[name] = (o.clone(), cache[idx.long()].clone())
    ob = torch.equal(res["new"][0], res["ref"][0])
    sb = torch.equal(res["new"][1], res["ref"][1])
    md = (res["new"][0].float() - res["ref"][0].float()).abs().max().item()
    print(f"lens={a.lens} bitequal out={ob} state={sb} out_maxdiff={md:.3e}")
    cache = cache0.clone()
    for r in range(a.rounds):
        for name in (("ref", "new") if r % 2 == 0 else ("new", "ref")):
            pkg.forward = arms[name]

            def fn():
                sp_kda.persistent_prefill(
                    q, k, v, g, beta, A_log, dt_bias, -5.0, None, cu, cu_cpu,
                    state_cache=cache, state_indices=idx, has_initial_state=has,
                )

            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            gph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gph):
                fn()
            ts = []
            for _ in range(a.iters):
                e0, e1 = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                e0.record()
                gph.replay()
                e1.record()
                e1.synchronize()
                ts.append(e0.elapsed_time(e1) * 1e3)
            ts.sort()
            print(f"lens={a.lens} r{r} {name} graph us median {ts[len(ts) // 2]:.1f} min {ts[0]:.1f}")


if __name__ == "__main__":
    main()
