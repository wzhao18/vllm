"""fip (persistent prefill KDA) A/B arm: bitwise outputs + CUDA-graph timing.

Works with both persistent_prefill signatures (initial_state or state_cache).
--save PATH stores out/final state; --cmp PATH compares bit for bit."""
import argparse
import inspect

import torch

from vllm.models.kimi_k3.nvidia.specialized_prefill import kda as sp_kda

D, H = 128, 96
SHAPES = {
    "f16k": [5461, 5461, 5462],
    "2x8k": [8192, 8192],
    "1x16k": [16384],
    "mixed": [1000, 3000, 7000, 5384],
    "small": [37, 100, 333],
}


def make(lens):
    torch.manual_seed(0)
    T, N = sum(lens), len(lens)
    bf = torch.bfloat16
    offs = [0, *torch.tensor(lens).cumsum(0).tolist()]
    cu_cpu = torch.tensor(offs, dtype=torch.int32)
    q, k, g = (torch.randn(1, T, H, D, device="cuda").to(bf) for _ in range(3))
    v = (torch.randn(1, T, H, D, device="cuda") * 0.5).to(bf)
    beta = torch.randn(1, T, H, device="cuda").to(bf)
    a_log = torch.randn(H, device="cuda") * 0.5
    dt_bias = torch.randn(H * D, device="cuda") * 0.1
    h0 = torch.randn(N, H, D, D, device="cuda") * 0.1
    return q, k, v, g, beta, a_log, dt_bias, cu_cpu.cuda(), cu_cpu, h0


def runner(lens):
    q, k, v, g, beta, a_log, dt_bias, cu, cu_cpu, h0 = make(lens)
    params = inspect.signature(sp_kda.persistent_prefill).parameters
    if "state_cache" in params:
        n = len(lens)
        cache = torch.zeros(3 * n + 8, H, D, D, device="cuda")
        idx = torch.arange(n, dtype=torch.int32, device="cuda") * 3 + 5
        cache[idx.long()] = h0
        has = torch.ones(n, dtype=torch.bool, device="cuda")

        def fn():
            o, _ = sp_kda.persistent_prefill(
                q, k, v, g, beta, a_log, dt_bias, -5.0, None, cu, cu_cpu,
                state_cache=cache, state_indices=idx, has_initial_state=has)
            return o, cache[idx.long()]

        def reset():
            cache[idx.long()] = h0
    else:

        def fn():
            return sp_kda.persistent_prefill(
                q, k, v, g, beta, a_log, dt_bias, -5.0, h0, cu, cu_cpu)

        def reset():
            pass
    return fn, reset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save")
    ap.add_argument("--cmp")
    ap.add_argument("--shapes", default="small,mixed,f16k")
    ap.add_argument("--time", default="f16k")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    ref = torch.load(a.cmp) if a.cmp else None
    saved, ok = {}, True
    for name in a.shapes.split(","):
        fn, reset = runner(SHAPES[name])
        reset()
        o, s = fn()
        torch.cuda.synchronize()
        saved[name] = {"out": o.cpu(), "state": s.cpu()}
        msg = f"{a.tag} {name:6s}"
        if ref is not None:
            r = ref[name]
            eq = torch.equal(r["out"], saved[name]["out"]) and torch.equal(
                r["state"], saved[name]["state"])
            ok &= eq
            msg += " BITWISE" if eq else " DIFF max|dout| %.3e" % (
                (r["out"].float() - saved[name]["out"].float()).abs().max())
        if name in a.time.split(","):
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
            msg += f" graph_us med {ts[len(ts) // 2]:.1f} min {ts[0]:.1f}"
        print(msg, flush=True)
    if a.save:
        torch.save(saved, a.save)
    print(f"{a.tag} RESULT {'OK' if ok else 'MISMATCH'}")


if __name__ == "__main__":
    main()
