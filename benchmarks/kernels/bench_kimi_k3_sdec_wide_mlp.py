# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 specialized decode: the DSpark drafter's MLP (gate_up -> silu_and_mul ->
down) as ONE die-local persistent stream-K grid (``cutedsl.gemm.wide_mlp``) vs the
stock path (F.linear gate_up, vLLM silu_and_mul, F.linear down; the pinned cuBLASLt
algorithms where pinned) at the release drafter's TP4 shape ([7168, 7168] gate_up,
[7168, 3584] down) on one GPU, cold weights (one weight copy per call in a CUDA
graph), us per call, plus the max |d| / max |ref| error vs the stock path and the
bitwise repeatability of the fused kernel.

python benchmarks/kernels/bench_kimi_k3_sdec_wide_mlp.py [--rows 288]
    [--arms stock,lt,mlp,nomma,stamp] [--issuers 3:2:4]
"""

import argparse
import statistics

import torch
import torch.nn.functional as F

H, I = 7168, 3584
NAMES = ("entry", "slot", "gu acc0", "gu pub", "gu cnt0", "gu red", "dn acc0", "dn pub",
         "dn cnt0", "dn red", "dn x0", "gu mma", "gu W0", "-", "dn mma", "end", "gu sum0",
         "gu st0", "dn sum0", "dn st0", "dn act0", "dn W0", "-", "-")  # fmt: skip


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


def stamp_line(st, wbytes=None):
    """Per stamp: median / max over CTAs (us after the earliest entry)."""
    st = st.cpu().double() / 1e3
    t0 = st[:, 0].min()
    out = []
    for c, name in enumerate(NAMES):
        if name == "-" or c == 13:
            continue
        col = st[:, c]
        col = col[col > 0]
        if len(col):
            col = col - t0
            out.append(f"{name} {col.median().item():.2f}/{col.max().item():.2f}")
    if wbytes:  # aggregate weight rate of each phase (first W landed -> last MMA issued)
        for name, a, b, nb in (("gu", 12, 11, wbytes[0]), ("dn", 21, 14, wbytes[1])):
            ca, cb = st[:, a][st[:, a] > 0], st[:, b][st[:, b] > 0]
            if len(ca) and len(cb):
                span = (cb.max() - ca.min()).item()
                out.append(f"{name} W {nb / span / 1e6:.1f} TB/s over {span:.2f} us")
    return " | ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="288")
    ap.add_argument("--arms", default="stock,lt,mlp,nomma,stamp")
    # weight issuers : activation issuers : activation ring stages, several
    ap.add_argument("--issuers", default="3:2:4")
    ap.add_argument("--ncopy", type=int, default=0)
    ap.add_argument("--ahead", default="2")
    ap.add_argument("--debugs", default="", help="extra skeleton arms: debug bits, e.g. 3,5,7")
    ap.add_argument("--dump", default="", help="save raw stamps to <dump>_<tag>_R<rows>.pt")
    args = ap.parse_args()
    torch.manual_seed(0)
    dev = torch.device("cuda")
    print(torch.cuda.get_device_name(), flush=True)
    import vllm._custom_ops  # noqa: F401  (torch.ops._C.silu_and_mul)
    from vllm.models.kimi_k3.nvidia.specialized_decode import lt_pin
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm import (
        wide_mlp,
        wide_sk,
    )

    configs = [tuple(int(v) for v in c.split(":")) for c in args.issuers.split(",")]
    arms = args.arms.split(",")
    _, pairs = wide_sk.die_pairs(dev)
    ncopy = args.ncopy or max(2, min(8, int(1.2e9 // (3 * H * I * 2)) + 1))
    print(f"H{H} I{I} pairs/die {pairs} configs {configs} ncopy {ncopy}", flush=True)
    wgu = [torch.randn(2 * I, H, device=dev, dtype=torch.bfloat16) * 0.02
           for _ in range(ncopy)]  # fmt: skip
    wdn = [torch.randn(H, I, device=dev, dtype=torch.bfloat16) * 0.02 for _ in range(ncopy)]
    for m in [int(r) for r in args.rows.split(",")]:
        x = torch.randn(m, H, device=dev, dtype=torch.bfloat16)
        out = torch.empty(m, H, device=dev, dtype=torch.bfloat16)
        gu = torch.empty(m, 2 * I, device=dev, dtype=torch.bfloat16)
        act = torch.empty(m, I, device=dev, dtype=torch.bfloat16)

        def stock(i, gu_fn=None, dn_fn=None):
            if gu_fn is None:
                torch.matmul(x, wgu[i].t(), out=gu)
            else:
                gu_fn(i)
            torch.ops._C.silu_and_mul(act, gu)
            if dn_fn is None:
                torch.matmul(act, wdn[i].t(), out=out)
            else:
                dn_fn(i)

        stock(0)
        torch.cuda.synchronize()
        ref = out.float().clone()
        res = {}
        res["stock"] = (time_graph(stock, ncopy), 0.0)
        if "lt" in arms:
            pg = [lt_pin.pinned(w, [m]) for w in wgu] if m in lt_pin.pins_for((2 * I, H)) else []
            pd = [lt_pin.pinned(w, [m]) for w in wdn] if m in lt_pin.pins_for((H, I)) else []
            if pg and all(p is not None for p in pg):
                gfn = lambda i: pg[i].run(x, gu)  # noqa: E731
                dfn = (lambda i: pd[i].run(act, out)) if pd and all(  # noqa: E731
                    p is not None for p in pd) else None  # fmt: skip
                res["lt"] = (time_graph(lambda i: stock(i, gfn, dfn), ncopy), 0.0)
            del pg, pd
        share = None
        extra = [(f"d{d}", int(d)) for d in args.debugs.split(",") if d]
        for (nw, nx, xs), ah in [(c, int(a)) for c in configs for a in args.ahead.split(",")]:
            for arm, dbg in ((("mlp", 0), ("nomma", 1), ("stamp", 8)) + tuple(extra)):
                if arm not in arms and not arm.startswith("d"):
                    continue
                tag = f"{arm}{nw}{nx}x{xs}a{ah}"
                try:
                    if not wide_mlp.supports(m, H, I, pairs, xs):
                        res[tag] = (float("nan"), "unsupported")
                        continue
                    pk = [
                        wide_mlp.PreparedWideMlp(wgu[c], wdn[c], [m], nw, nx, dbg, xs,
                                                 share=None, ahead=ah)
                        for c in range(ncopy)
                    ]  # fmt: skip
                    for c, p in enumerate(pk):
                        if c:
                            p.share = pk[0]
                        p.prepare()
                    out.zero_()
                    torch.cuda.nvtx.range_push(f"mlp_{arm}")  # ncu --nvtx-include mlp_<arm>/
                    pk[0].run(x, out)
                    torch.cuda.nvtx.range_pop()
                    torch.cuda.synchronize()
                    err = ((out.float() - ref).abs().max() / ref.abs().max()).item()
                    if dbg == 0:
                        o1 = out.clone()
                        pk[0].run(x, out)
                        torch.cuda.synchronize()
                        res[tag + ".det"] = (0.0, float((o1 != out).sum().item()))
                    res[tag] = (time_graph(lambda i, pk=pk: pk[i].run(x, out), ncopy),
                                err if dbg == 0 else 0.0)  # fmt: skip
                    if dbg & 8:
                        pk[0].run(x, out)
                        torch.cuda.synchronize()
                        print(f"R{m} {tag}: " + stamp_line(pk[0].stamps(m), (2 * I * H * 2, H * I * 2)),
                              flush=True)
                        if args.dump:
                            torch.save(pk[0].stamps(m).cpu(), f"{args.dump}_{tag}_R{m}.pt")
                    del pk
                except Exception as e:  # noqa: BLE001
                    res[tag] = (float("nan"), repr(e)[:300])
                torch.cuda.empty_cache()
        base = res["stock"][0]
        line = " ".join(
            f"{t}={v[1]:.2g}" if t.endswith(".det") else
            f"{t}={v[0]:.2f}" + (f"(x{base / v[0]:.2f} e{v[1]:.1e})" if isinstance(v[1], float)
                                 else f"[{v[1]}]")
            for t, v in res.items()
        )  # fmt: skip
        print(f"R{m} mlp H{H} I{I} ncopy{ncopy}: {line}", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
