# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 specialized decode: the die-local stream-K token-wide BF16 GEMM
(``cutedsl.gemm.wide_sk``) vs cuBLAS (F.linear, the pinned cuBLASLt algorithm) at the
release DSpark drafter's TP4 shapes on one GPU, cold weights (one weight copy per call
in a CUDA graph), us per call, plus the max |d| / max |ref| error vs F.linear.

python benchmarks/kernels/bench_kimi_k3_sdec_wide_sk.py [--rows 96,192,288] [--shape lm]
    [--arms lt,sk,striped,nomma] [--issuers 3:2]
"""

import argparse
import statistics

import torch
import torch.nn.functional as F

SHAPES = {"lm": (40960, 7168), "gu": (7168, 7168), "qkv": (2048, 7168), "down": (7168, 3584),
          "o": (7168, 1536), "markov": (40960, 512)}


def time_graph(fn, ncopy, reps=10, rounds=5):
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for i in range(ncopy):
            fn(i)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        for i in range(ncopy):
            fn(i)
    ts = []
    for _ in range(rounds):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(reps):
            g.replay()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1e3 / reps / ncopy)
    del g
    return statistics.median(ts)


def stamp_summary(st, ks, total, pairs):
    """One line: kernel span and per-phase medians / maxima (us) over CTAs."""
    slots = (st[:, 14] & 0xFFFFFFFF).tolist()
    st = st.cpu().double() / 1e3
    t0 = st[:, 0].min()
    span = (st[:, 15].max() - t0).item()
    entry = st[:, 0] - t0
    first = st[:, 2] - st[:, 0]
    nseg = []
    for g in slots:
        d, j = (0, g) if g < pairs[0] else (1, g - pairs[0])
        n = pairs[d]
        s0, s1 = j * total // n, (j + 1) * total // n
        nseg.append((s1 - 1) // ks - s0 // ks + 1)
    epi, main = [], []
    kinds = {"part": [], "full": [], "own": []}
    per_step = []
    for c, (n, g) in enumerate(zip(nseg, slots)):
        d, j = (0, g) if g < pairs[0] else (1, g - pairs[0])
        s0, s1 = j * total // pairs[d], (j + 1) * total // pairs[d]
        for i in range(min(n, 6)):
            t = s0 // ks + i
            lo, hi = max(s0, t * ks), min(s1, (t + 1) * ks)
            kind = "part" if lo != t * ks else ("full" if hi == (t + 1) * ks else "own")
            e = (st[c, 3 + 2 * i] - st[c, 2 + 2 * i]).item()
            epi.append(e)
            kinds[kind].append(e)
            if i:
                w = (st[c, 2 + 2 * i] - st[c, 1 + 2 * i]).item()
                main.append(w)
                per_step.append(w / (hi - lo))
    tail = st[:, 15] - st[:, 3 + 2 * (torch.tensor(nseg).clamp(max=6) - 1)]
    med = lambda v: statistics.median(v) if len(v) else float('nan')
    return (f'span {span:.1f} entry max {entry.max():.1f} | seg0 ready med {first.median():.1f} '
            f'max {first.max():.1f} | epilogue med {med(epi):.2f} max {max(epi):.2f} | '
            f'next-seg wait med {med(main):.2f} max {max(main) if main else 0:.2f} | '
            f'tail max {tail.max():.2f} | end-entry med {(st[:, 15] - st[:, 0]).median():.1f} | '
            f'slot med {(st[:, 1] - st[:, 0]).median():.2f} | epi by kind ' + ' '.join(
                f'{kk} n{len(v)} med {med(v):.2f} max {max(v) if v else 0:.2f}'
                for kk, v in kinds.items()) + f' | us/step (seg>0) med {med(per_step):.3f}'
            + f' | tmem-release seg0 med {(st[:, 12] - st[:, 2]).median():.2f}')


def _with_pairs(pk, pairs):
    pk.pairs = pairs
    return pk


def spread_arms(args, arms, m, n, k, x, ws, pairs, ncopy, res, run, configs):
    """``spread`` arms: sp (BF16 out), spst (+ stamps), spsilu (gate/up + silu_and_mul vs
    F.linear + vLLM silu_and_mul)."""
    import vllm._custom_ops  # noqa: F401  (torch.ops._C.silu_and_mul)
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm import wide_sk

    for nw, nx, xs in configs:
        if not wide_sk.supports(m, n, k, pairs, spread=True, x_stages=xs):
            res[f"sp{nw}{nx}x{xs}"] = (float("nan"), "unsupported")
            continue
        for arm, dbg, silu in (("sp", 0, False), ("spst", 8, False), ("spsilu", 0, True)):
            if arm not in arms:
                continue
            tag = f"{arm}{nw}{nx}x{xs}"
            pk = [_with_pairs(wide_sk.PreparedWideSK(w, [m], w_issuers=nw, x_issuers=nx, debug=dbg,
                                         x_stages=xs, spread=True, silu=silu,
                                         **args.extra), pairs).prepare()
                  for w in ws]  # fmt: skip
            if silu:
                h = torch.empty(m, n, device=x.device, dtype=torch.bfloat16)
                act = torch.empty(m, n // 2, device=x.device, dtype=torch.bfloat16)

                def stock(i):
                    torch.matmul(x, ws[i].t(), out=h)
                    torch.ops._C.silu_and_mul(act, h)

                stock(0)
                ref_act = act.float().clone()
                res["F.linear+silu"] = (time_graph(stock, ncopy), 0.0)
                o = torch.empty(m, n // 2, device=x.device, dtype=torch.bfloat16)
                pk[0].run(x, o)
                torch.cuda.synchronize()
                err = ((o.float() - ref_act).abs().max() / ref_act.abs().max()).item()
                ulp = (o.float() != ref_act).float().mean().item()
                res[tag] = (time_graph(lambda i, pk=pk: pk[i].run(x, o), ncopy), err)
                pk[0].run(x, o)
                o1 = o.clone()
                pk[0].run(x, o)
                torch.cuda.synchronize()
                res[tag + ".det"] = (0.0, float((o1 != o).sum().item()))
                res[tag + ".neq%"] = (0.0, 100.0 * ulp)
            else:
                run(tag, lambda i, pk=pk: pk[i].run(x, run.out), check=dbg == 0)
                if dbg & 8:
                    pk[0].run(x, run.out)
                    torch.cuda.synchronize()
                    st = pk[0].stamps(m).cpu()
                    t0 = st[:, 0].min().item()
                    rel = lambda c: (st[:, c] - t0).float() / 1e3  # noqa: E731
                    print(f"R{m} {tag} stamps us (median/max over CTAs): "
                          f"seg0 acc {rel(2).median():.1f}/{rel(2).max():.1f} "
                          f"pub0 {rel(3).median():.1f}/{rel(3).max():.1f} "
                          f"red-ready {rel(10).median():.1f}/{rel(10).max():.1f} "
                          f"loaded {rel(11).median():.1f}/{rel(11).max():.1f} "
                          f"end {rel(12).median():.1f}/{rel(12).max():.1f}", flush=True)
                else:
                    pk[0].run(x, run.out)
                    o1 = run.out.clone()
                    pk[0].run(x, run.out)
                    torch.cuda.synchronize()
                    res[tag + ".det"] = (0.0, float((o1 != run.out).sum().item()))
            del pk
            torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="96,192,288")
    ap.add_argument("--shape", default="lm")
    ap.add_argument("--arms", default="lt,sk,striped,nomma")
    # weight warps : activation warps per chunk : activation ring stages, several
    ap.add_argument("--issuers", default="3:2:4")
    ap.add_argument("--smem", type=int, default=0, help="KiB budget (0: smem_budget)")
    ap.add_argument("--packed", type=int, default=0)
    ap.add_argument("--keep-w", type=int, default=0)
    ap.add_argument("--warm", type=int, default=0, help="one weight copy (L2-warm)")
    ap.add_argument("--nocopy", type=int, default=1, help="two TMEM accumulators (0: copy form)")
    ap.add_argument("--pairs", default="", help="p0,p1 CTA pairs per die (default: all)")
    args = ap.parse_args()
    args.extra = dict(smem=args.smem * 1024 or None, packed=bool(args.packed),
                      keep_w=bool(args.keep_w), nocopy=bool(args.nocopy))
    torch.manual_seed(0)
    dev = torch.device("cuda")
    print(torch.cuda.get_device_name(), flush=True)
    from vllm.models.kimi_k3.nvidia.specialized_decode import lt_pin
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common import dies
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm import wide_sk

    n, k = SHAPES[args.shape]
    configs = [tuple(int(v) for v in c.split(":")) for c in args.issuers.split(",")]
    arms = args.arms.split(",")
    _, pairs = wide_sk.die_pairs(dev)
    if args.pairs:  # a subset of the SM pairs per die (static SM-split calibration)
        pairs = tuple(int(v) for v in args.pairs.split(","))
    print(f"shape {args.shape} N{n} K{k} pairs/die {pairs} configs {configs}", flush=True)
    ncopy = 1 if args.warm else max(2, min(16, int(1.2e9 // (n * k * 2)) + 1))
    ws = [torch.randn(n, k, device=dev, dtype=torch.bfloat16) * 0.02 for _ in range(ncopy)]
    for m in [int(r) for r in args.rows.split(",")]:
        x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
        out = torch.empty(m, n, device=dev, dtype=torch.bfloat16)
        ref = F.linear(x, ws[0]).float()
        res = {}

        def run(tag, fn, check=True):
            try:
                out.zero_()
                torch.cuda.nvtx.range_push("wsk_" + tag)  # ncu --nvtx-include wsk_<arm>/
                fn(0)
                torch.cuda.nvtx.range_pop()
                torch.cuda.synchronize()
                err = ((out.float() - ref).abs().max() / ref.abs().max()).item()
                if err > 0.01 and check:
                    bad = (out.float() - ref).abs() > 0.05 * ref.abs().max()
                    tok = bad.any(1).nonzero().flatten().tolist()
                    col = bad.any(0).view(-1, 128).any(1).nonzero().flatten().tolist()
                    print(f"{tag}: bad tokens {len(tok)} {tok[:6]}..{tok[-6:]} bad 128-col "
                          f"blocks {len(col)} {col[:8]}..{col[-4:]}", flush=True)
                    o, rf = out.float(), ref
                    src = [int((rf - o[t]).abs().amax(1).argmin()) for t in range(o.shape[0])]
                    runs, st0 = [], 0
                    for t in range(1, len(src) + 1):
                        if t == len(src) or src[t] != src[t - 1] + 1:
                            runs.append(f"{st0}-{t - 1}<-{src[st0]}")
                            st0 = t
                    print("  out rows <- ref rows: " + " ".join(runs), flush=True)
                    for t in (tok[0], tok[len(tok) // 2], tok[-1]):
                        d = (rf - o[t]).abs().amax(1)
                        print(f"  tok {t}: zeros {int((o[t] == 0).sum())}/{o.shape[1]} "
                              f"closest ref row {int(d.argmin())} (maxdiff {d.min():.3g}, "
                              f"own {d[t]:.3g})", flush=True)
                res[tag] = (time_graph(fn, ncopy), err if check else 0.0)
            except Exception as e:  # noqa: BLE001
                res[tag] = (float("nan"), repr(e)[:160])

        run.out = out
        run("F.linear", lambda i: torch.matmul(x, ws[i].t(), out=out))
        if "lt" in arms and m in lt_pin.pins_for((n, k)):
            pls = [lt_pin.pinned(w, [m]) for w in ws]
            if all(p is not None for p in pls):
                run("lt_pin", lambda i: pls[i].run(x, out))
            del pls
        todo = [(a_, p_, d_, cf) for cf in configs
                for a_, p_, d_ in (("sk", True, 0), ("striped", False, 0), ("nomma", True, 1),
                                   ("xl", True, 0), ("xlnomma", True, 1),
                                   ("wonly", True, 3), ("xonly", True, 5),
                                   ("stamp", True, 8), ("nommastamp", True, 9),
                                   ("wonlystamp", True, 11), ("xonlystamp", True, 13))
                if a_ in arms]  # fmt: skip
        if any(a.startswith("sp") for a in arms):
            spread_arms(args, arms, m, n, k, x, ws, pairs, ncopy, res, run, configs)
        for arm, place, dbg, (nw, nx, xs) in todo:
            arm = f"{arm}{nw}{nx}x{xs}"
            if not wide_sk.supports(m, n, k, pairs):
                res[arm] = (float("nan"), "unsupported")
                continue
            pk = [
                _with_pairs(wide_sk.PreparedWideSK(w, [m], place=place, w_issuers=nw, x_issuers=nx,
                                       debug=dbg, x_stages=xs, **args.extra), pairs).prepare()
                for w in ws
            ]  # fmt: skip
            if arm.startswith("xl"):  # activations copied per die
                xd = [dies.localized_like(x, d) for d in (0, 1)]
                run(arm, lambda i, pk=pk, xd=xd: pk[i].run(xd[0], out, xd[1]), check=dbg == 0)
            else:
                run(arm, lambda i, pk=pk: pk[i].run(x, out), check=(dbg & ~8) == 0)
            if dbg & 8:
                pk[0].run(x, out)
                torch.cuda.synchronize()
                ks, total = k // 64, n // 512 * (k // 64)
                print(f"R{m} {arm}: " + stamp_summary(pk[0].stamps(m), ks, total, pairs),
                      flush=True)  # fmt: skip
            if dbg == 0:
                # bitwise repeatability (deterministic reduction order)
                out.zero_()
                pk[0].run(x, out)
                o1 = out.clone()
                pk[0].run(x, out)
                torch.cuda.synchronize()
                res[arm + ".det"] = (0.0, float((o1 != out).sum().item()))
            del pk
            torch.cuda.empty_cache()
        base = res["F.linear"][0]
        line = " ".join(
            f"{t}={v[1]:.2g}" if t.endswith((".det", ".neq%")) else
            f"{t}={v[0]:.2f}" + (f"(x{base / v[0]:.2f} e{v[1]:.1e})" if isinstance(v[1], float)
                                 else f"[{v[1]}]")
            for t, v in res.items()
        )  # fmt: skip
        print(f"R{m} {args.shape} M{m} N{n} K{k} ncopy{ncopy}: {line}", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
