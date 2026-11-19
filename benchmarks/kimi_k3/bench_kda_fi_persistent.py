"""FROST vs FlashInfer persistent KDA (CuTe DSL, FP32 state in TMEM, LPT pieces
with exact FP32 state handoffs) at the deployed prefill shape (dev tool).

usage: python bench_kda_fi_persistent.py [--tokens 16384] [--seqs 3] [--heads 96]
       [--iters 10] [--lower-bound -5.0]
"""

import argparse
import os
import sys
import time
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

D = 128


def rel_rms(a, b):
    a, b = a.float(), b.float()
    return ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt().clamp_min(1e-30)).item()


def prof_us(fn, iters):
    from torch.profiler import ProfilerActivity, profile

    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    acc, cnt = defaultdict(float), defaultdict(int)
    for e in prof.events():
        if e.device_type.name == "CUDA" and e.device_time_total > 0:
            acc[e.name] += e.device_time_total
            cnt[e.name] += 1
    rows = sorted(((t / iters, cnt[n] // iters, n) for n, t in acc.items()), reverse=True)
    return sum(r[0] for r in rows), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=16384)
    ap.add_argument("--seqs", type=int, default=3)
    ap.add_argument("--heads", type=int, default=96)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--lower-bound", type=float, default=-5.0)
    ap.add_argument("--impls", default="frost,fip")
    ap.add_argument("--lens", default="")
    args = ap.parse_args()

    dev = torch.device("cuda")
    print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0),
          torch.cuda.get_device_properties(0).multi_processor_count, flush=True)
    try:
        import cutlass

        print("cutlass-dsl", getattr(cutlass, "__version__", "?"), flush=True)
    except Exception as e:  # noqa: BLE001
        print("cutlass import failed", e, flush=True)
    torch.manual_seed(0)
    T, N, H = args.tokens, args.seqs, args.heads
    lens = [T // N] * N
    lens[-1] += T - sum(lens)
    if args.lens:
        lens = [int(x) for x in args.lens.split(",")]
        T, N = sum(lens), len(lens)
    offs = [0] + torch.tensor(lens).cumsum(0).tolist()
    cu = torch.tensor(offs, dtype=torch.int32, device=dev)
    bf = torch.bfloat16
    q = torch.randn(1, T, H, D, dtype=bf, device=dev)
    k = torch.randn(1, T, H, D, dtype=bf, device=dev)
    v = torch.randn(1, T, H, D, dtype=bf, device=dev) * 0.5
    g = torch.randn(1, T, H, D, dtype=bf, device=dev) * 0.5
    beta = torch.randn(1, T, H, dtype=bf, device=dev)
    A_log = torch.randn(H, dtype=torch.float32, device=dev) * 0.5
    dt_bias = torch.randn(H, D, dtype=torch.float32, device=dev) * 0.1
    init_state = torch.randn(N, H, D, D, dtype=torch.float32, device=dev) * 0.05
    lb = args.lower_bound
    res = {}

    for impl in args.impls.split(","):
        if impl == "frost":
            from cudnn.linear_attention.ops import kimi_delta_attention

            def fn():
                return kimi_delta_attention(
                    q.view(T, H, D), k.view(T, H, D), v.view(T, H, D), g.view(T, H, D),
                    beta.view(T, H), cu, scale=D**-0.5, initial_state=init_state,
                    output_final_state=True, use_qk_l2norm_in_kernel=True,
                    use_beta_sigmoid_in_kernel=True, safe_gate=True, gate_lower_bound=lb,
                    a_log=A_log, dt_bias=dt_bias, plan_name="kda_frost",
                )[:2]
        elif impl in ("fip", "fipprobe", "fipprobe2", "fipprobe3"):
            if impl == "fip":
                from fi_persistent import forward as fip
            elif impl == "fipprobe":
                from fi_probe import forward as fip
            elif impl == "fipprobe2":
                from fi_probe2 import forward as fip
            else:
                from fi_probe3 import forward as fip

            out = torch.empty_like(q)
            fstate = torch.empty_like(init_state)
            t0 = time.time()
            plan = fip.make_plan(offs, H, dev)
            print(f"fip plan {time.time() - t0:.1f} s: G={plan[7]} split={plan[10]} "
                  f"full_chunks={plan[11]}", flush=True)

            def fn():
                fip.fwd(q, k, v, g, beta, D**-0.5, out, A_log, dt_bias, lb,
                        initial_state=init_state, final_state=fstate, cu_seqlens=cu, plan=plan)
                return out.view(T, H, D), fstate
        else:
            raise ValueError(impl)
        try:
            t0 = time.time()
            o, s = fn()
            torch.cuda.synchronize()
            print(f"{impl}: first call {time.time() - t0:.1f} s", flush=True)
        except Exception as e:  # noqa: BLE001
            import traceback

            tb = traceback.format_exc().splitlines()
            print(f"{impl}: FAILED {type(e).__name__}: {str(e)[:600]}", flush=True)
            print("\n".join(tb[-12:])[:2500], flush=True)
            continue
        res[impl] = (o.reshape(T, H, D).clone(), s.clone())
        tot, rows = prof_us(fn, args.iters)
        print(f"impl={impl} T={T} N={N} H={H}: kernels {tot:.1f} us", flush=True)
        for us, c, n in rows[:5]:
            print(f"  {us:8.1f} us x{c} {n[:100]}", flush=True)

    if "frost" in res and "fip" in res:
        (o0, s0), (o1, s1) = res["frost"], res["fip"]
        print(f"check fip vs frost: out {rel_rms(o1, o0):.3e} state {rel_rms(s1, s0):.3e} "
              f"state^T {rel_rms(s1.transpose(-1, -2), s0):.3e} nan {torch.isnan(o1).any().item()}",
              flush=True)
        for n in range(N):
            a, b = offs[n], offs[n + 1]
            print(f"  seq{n} out {rel_rms(o1[a:b], o0[a:b]):.3e} state {rel_rms(s1[n], s0[n]):.3e}",
                  flush=True)


if __name__ == "__main__":
    main()
