"""Persistent prefill KDA (fip) bitwise reference check + timing.

--save PATH stores out/state for a fixed set of shapes; --cmp PATH compares a
build against that file bit for bit.  Each shape is timed as a CUDA graph."""

import argparse

import torch

from vllm.models.kimi_k3.nvidia.specialized_prefill import kda as sp_kda

D = 128
SHAPES = {
    "f16k": [5461, 5461, 5462],
    "2x8k": [8192, 8192],
    "1x8k": [8192],
    "mixed": [1000, 3000, 7000, 5384],
    "small": [37, 100, 333],
}


def run_shape(lens, iters, time_it):
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
    idx = torch.arange(N, dtype=torch.int32, device="cuda") * 3 + 5
    has = torch.ones(N, dtype=torch.bool, device="cuda")
    has[-1] = N == 1
    cache = cache0.clone()

    def fn():
        return sp_kda.persistent_prefill(
            q, k, v, g, beta, A_log, dt_bias, -5.0, None, cu, cu_cpu,
            state_cache=cache, state_indices=idx, has_initial_state=has,
        )

    o, _ = fn()
    torch.cuda.synchronize()
    res = {"out": o.clone(), "state": cache[idx.long()].clone()}
    t = None
    if time_it:
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        gph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gph):
            fn()
        ts = []
        for _ in range(iters):
            e0, e1 = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            e0.record()
            gph.replay()
            e1.record()
            e1.synchronize()
            ts.append(e0.elapsed_time(e1) * 1e3)
        ts.sort()
        t = (ts[len(ts) // 2], ts[0])
    return res, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save")
    ap.add_argument("--cmp")
    ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument("--time", default="f16k,2x8k,1x8k")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--phantom", type=int, default=0)
    a = ap.parse_args()
    if a.phantom:
        from vllm.models.kimi_k3.nvidia.specialized_prefill.kda_persistent import (
            forward as fw,
        )

        fw._PHANTOM = a.phantom
    ref = torch.load(a.cmp) if a.cmp else None
    saved = {}
    ok_all = True
    for name in a.shapes.split(","):
        res, t = run_shape(SHAPES[name], a.iters, name in a.time.split(","))
        saved[name] = {k: v.cpu() for k, v in res.items()}
        msg = f"{name:6s}"
        if t:
            msg += f" graph us median {t[0]:.1f} min {t[1]:.1f}"
        if ref is not None:
            for key in ("out", "state"):
                r, x = ref[name][key], saved[name][key]
                eq = torch.equal(r, x)
                ok_all &= eq
                d = (r.float() - x.float()).abs().max().item()
                msg += f" {key} {'BITWISE' if eq else 'DIFF'} maxabs {d:.3e}"
        print(msg, flush=True)
    if a.save:
        torch.save(saved, a.save)
        print("saved", a.save)
    if ref is not None:
        print("CHECK", "PASS" if ok_all else "FAIL")


if __name__ == "__main__":
    main()
