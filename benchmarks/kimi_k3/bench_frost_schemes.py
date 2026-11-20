"""FROST KDA scheme sweep at the deployed prefill shape (dev tool).

cuDNN frontend FROST picks one scheme per plan: uncut (+ tail), dv_split
(with or without the chunk-parallel prep path) or chain (pieces + summary).
This forces each scheme (monkeypatching the engine's choosers before the plan
is built) and reports kernel time and rel-rms vs the default scheme.

usage: python bench_frost_schemes.py [--tokens 16384] [--seqs 3] [--heads 96]
       [--variants default,dv_prep,dv_noprep,chain2,chain3]
"""

import argparse
import time
from collections import defaultdict

import torch

D = 128


def rel_rms(a, b):
    a, b = a.float(), b.float()
    return ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt().clamp_min(1e-30)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=16384)
    ap.add_argument("--seqs", type=int, default=3)
    ap.add_argument("--heads", type=int, default=96)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--variants", default="default,dv_prep,dv_noprep,chain2,chain3")
    args = ap.parse_args()

    from cudnn.linear_attention.frost import kda_engine
    from cudnn.linear_attention.frost.common import piece_chain, tail
    from cudnn.linear_attention.ops import kimi_delta_attention

    orig = dict(cp=piece_chain.choose_pieces, dv=piece_chain.is_dv_split, tail=tail.choose_tail,
                frac=kda_engine.PREP_TILE_FRACTION)

    def setv(name):
        piece_chain.choose_pieces = orig["cp"]
        piece_chain.is_dv_split = orig["dv"]
        tail.choose_tail = orig["tail"]
        kda_engine.PREP_TILE_FRACTION = orig["frac"]
        if name.startswith("dv_"):
            piece_chain.choose_pieces = lambda **kw: (0, orig["cp"](**kw)[1])
            piece_chain.is_dv_split = lambda **kw: True
            kda_engine.PREP_TILE_FRACTION = 100.0 if name == "dv_prep" else 0.0
        elif name.startswith("chain"):
            n = int(name[5:])
            piece_chain.choose_pieces = lambda **kw: (n, orig["cp"](**kw)[1] or 1)

    dev = torch.device("cuda")
    torch.manual_seed(0)
    T, N, H = args.tokens, args.seqs, args.heads
    lens = [T // N] * N
    lens[-1] += T - sum(lens)
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device=dev)
    bf = torch.bfloat16
    q = torch.randn(T, H, D, dtype=bf, device=dev)
    k = torch.randn(T, H, D, dtype=bf, device=dev)
    v = torch.randn(T, H, D, dtype=bf, device=dev) * 0.5
    g = torch.randn(T, H, D, dtype=bf, device=dev) * 0.5
    beta = torch.randn(T, H, dtype=bf, device=dev)
    A_log = torch.randn(H, dtype=torch.float32, device=dev) * 0.5
    dt_bias = torch.randn(H, D, dtype=torch.float32, device=dev) * 0.1
    init_state = torch.zeros(N, H, D, D, dtype=torch.float32, device=dev)
    print("orig choose_pieces:", orig["cp"](num_seqs=N, heads_out=H, num_sm=torch.cuda.get_device_properties(0).multi_processor_count,
          total_tokens=T, b_t=16, cadence_tokens=0, batch_invariant=False, expand_num=1), flush=True)

    res = {}
    import importlib
    kops = importlib.import_module("cudnn.linear_attention.ops.kda")
    for name in args.variants.split(","):
        setv(name)
        for attr in dir(kops):
            if attr.endswith("_cache") and isinstance(getattr(kops, attr), dict):
                getattr(kops, attr).clear()

        def fn(name=name):
            return kimi_delta_attention(
                q, k, v, g, beta, cu, scale=D**-0.5, initial_state=init_state,
                output_final_state=True, use_qk_l2norm_in_kernel=True,
                use_beta_sigmoid_in_kernel=True, safe_gate=True, gate_lower_bound=-5.0,
                a_log=A_log, dt_bias=dt_bias, plan_name="kda_frost",
            )[:2]

        try:
            t0 = time.time()
            o, s = fn()
            torch.cuda.synchronize()
            print(f"{name}: first call {time.time() - t0:.1f} s", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"{name}: FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
            continue
        res[name] = (o.clone(), s.clone())
        if "default" in res and name != "default":
            o0, s0 = res["default"]
            print(f"  check {name} vs default: out {rel_rms(o, o0):.3e} state {rel_rms(s, s0):.3e} "
                  f"nan {torch.isnan(o).any().item()}", flush=True)
        from torch.profiler import ProfilerActivity, profile

        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(args.iters):
                fn()
            torch.cuda.synchronize()
        acc, cnt = defaultdict(float), defaultdict(int)
        for e in prof.events():
            if e.device_type.name == "CUDA" and e.device_time_total > 0:
                acc[e.name] += e.device_time_total
                cnt[e.name] += 1
        rows = sorted(((t / args.iters, cnt[n] // args.iters, n) for n, t in acc.items()), reverse=True)
        print(f"variant={name} T={T} N={N} H={H}: kernels {sum(r[0] for r in rows):.1f} us", flush=True)
        for us, c, n in rows[:5]:
            print(f"  {us:8.1f} us x{c} {n[:100]}", flush=True)


if __name__ == "__main__":
    main()
