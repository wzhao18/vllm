"""D-GEMM microbench: weight-stationary pre-wait BF16 GEMM (cutedsl/gemm/wstat.py) vs
torch.mm (nvjet) and GEMM-EX2's prewait_gemm, chained in a CUDA graph on rotating cold
weight copies. Optional host kernel before every GEMM (--prod k spin iterations; it
triggers launch_dependents at entry, like the SP all-gather / reduce): exposed = chain
minus the host chain.

Arms: mm | pw (EX2 prewait_gemm, its CONFIGS) | ws:C:NWS:NXS:NW:XL:KB.
"""

import argparse
import json
import statistics
import time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=5344)
ap.add_argument("--k", type=int, default=7168)
ap.add_argument("--rows", default="128")
ap.add_argument("--arms", default="mm,ws:4:12:4:2:2:1")
ap.add_argument("--chain", type=int, default=16)
ap.add_argument("--reps", type=int, default=15)
ap.add_argument("--prod", type=float, default=0.0)
ap.add_argument("--prod_ctas", type=int, default=224)
ap.add_argument("--prod_smem", type=int, default=1816)
ap.add_argument("--out", default="")
args = ap.parse_args()

from vllm.models.kimi_k3.nvidia.specialized_decode import prewait_gemm  # noqa: E402
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm import wstat  # noqa: E402
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.sm107 import (  # noqa: E402
    prewait as pw,
)

dev = torch.device("cuda")
torch.manual_seed(0)
N, K = args.n, args.k
ncopy = max(args.chain, -(-1_600_000_000 // (N * K * 2)))
W = [torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.05 for _ in range(ncopy)]
PK = {}
print(f"N {N} K {K} copies {ncopy} optin {wstat._module().wstat_smem_optin()}", flush=True)
_sink = torch.zeros(args.prod_ctas, device=dev, dtype=torch.float32)
prod = ((lambda: wstat._module().wstat_host(int(args.prod * 1000), args.prod_ctas, _sink))
        if args.prod > 0 else None)  # gather-like host, carve-out 100


def mk_arm(arm, x):
    if arm == "mm":
        return lambda i, out: torch.mm(x, W[i].t(), out=out)
    if arm == "pw":
        if not prewait_gemm.supported(x.shape[0], K):
            raise ValueError("prewait_gemm: no config")
        if "pk" not in PK:
            PK["pk"] = [prewait_gemm.pack_weight(w) for w in W]
        return lambda i, out: prewait_gemm.gemm(out, x, PK["pk"][i], N)
    cfg = tuple(int(v) for v in arm.split(":")[1:])
    if arm.startswith("wsp"):  # the packed tile image (one bulk copy per stage)
        if "pk" not in PK:
            PK["pk"] = [prewait_gemm.pack_weight(w) for w in W]
        return lambda i, out: wstat.gemm(out, x, PK["pk"][i], cfg)
    return lambda i, out: wstat.gemm(out, x, W[i], cfg)


def time_chain(f, R, with_prod):
    outs = [torch.empty(R, N, device=dev, dtype=torch.bfloat16) for _ in range(2)]
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for i in range(3):
            if with_prod and prod is not None:
                prod()
            if f is not None:
                f(i % ncopy, outs[i % 2])
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        for i in range(args.chain):
            if with_prod and prod is not None:
                prod()
            if f is not None:
                f(i % ncopy, outs[i % 2])
    ts = []
    for _ in range(args.reps + 2):
        flush.zero_()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000 / args.chain)
    return statistics.median(ts[2:])


flush = torch.empty(512 * 1024 * 1024, device=dev, dtype=torch.uint8)
prod_us = time_chain(None, 128, True) if prod is not None else None
if prod is not None:
    print(f"host alone {prod_us:.2f} us/call", flush=True)
res = []
for R in [int(r) for r in args.rows.split(",")]:
    x = torch.randn(R, K, device=dev, dtype=torch.bfloat16)
    ref = (x.float() @ W[0].float().t())
    for arm in args.arms.split(","):
        try:
            t0 = time.time()
            f = mk_arm(arm, x)
            out = torch.full((R, N), float("nan"), device=dev, dtype=torch.bfloat16)
            f(0, out)
            torch.cuda.synchronize()
            err = ((out.float() - ref).abs().max() / ref.abs().max()).item()
            row = dict(R=R, arm=arm, rel=round(err, 6), build_s=round(time.time() - t0, 1))
            row["us"] = round(time_chain(f, R, False), 2)
            if prod is not None:
                tot = time_chain(f, R, True)
                row.update(with_host_us=round(tot, 2), exposed_us=round(tot - prod_us, 2))
        except Exception as e:  # report and continue
            row = dict(R=R, arm=arm, error=f"{type(e).__name__}: {str(e)[:400]}")
        print(json.dumps(row), flush=True)
        res.append(row)
if args.out:
    json.dump(res, open(args.out, "w"), indent=1)
