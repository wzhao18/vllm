"""D-GEMM microbench: decode MXFP8 o_proj-class GEMM (KDA o_proj 7168 x 3072) on the
vendored sm_107 block-scaled kernel, tactics incl. swap_ab (W = the MMA's M operand,
tokens split into N tiles: no split-K, all SMs), vs scaled_mm. Cold weights (rotating
copies + L2 flush between replays) or --warm (one copy). Optional producer (--prod k
spin iterations, --prod_smem bytes: a smem-hogging predecessor leaves no co-residency).

Arms: scaled_mm | own:<tile MxN>:<cluster>:<swap 0/1>:<inst_m>:<pdl 0/1>:<prewait>.
"""

import argparse
import json
import statistics
import time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=7168)
ap.add_argument("--k", type=int, default=3072)
ap.add_argument("--rows", default="128,256,384")
ap.add_argument("--arms", default="scaled_mm")
ap.add_argument("--chain", type=int, default=16)
ap.add_argument("--reps", type=int, default=15)
ap.add_argument("--prod", type=float, default=0.0)
ap.add_argument("--prod_smem", type=int, default=0)
ap.add_argument("--warm", action="store_true")
ap.add_argument("--out", default="")
args = ap.parse_args()

from flashinfer import mxfp8_quantize  # noqa: E402

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.sm107 import (  # noqa: E402
    prewait as pw,
)

dev = torch.device("cuda")
torch.manual_seed(0)
N, K = args.n, args.k
nsm = torch.cuda.get_device_properties(0).multi_processor_count
wbytes = N * K
ncopy = max(args.chain, -(-1_600_000_000 // wbytes))
w_bf = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.05
w_q, w_s = mxfp8_quantize(w_bf, True)
W = [w_q] * ncopy if args.warm else [w_q.clone() for _ in range(ncopy)]
WS = [w_s] * ncopy if args.warm else [w_s.clone() for _ in range(ncopy)]
del w_bf
print(f"sms {nsm} N {N} K {K} copies {ncopy} warm {args.warm}", flush=True)
prod = pw.Spin(int(args.prod * 1000), nsm, args.prod_smem) if args.prod > 0 else None


def mk_arm(arm, R, xq, xs):
    if arm == "scaled_mm":
        def f(i, out):
            torch._scaled_mm(xq, W[i].t(), scale_a=xs.view(torch.float8_e8m0fnu),
                             scale_b=WS[i].view(torch.float8_e8m0fnu),
                             out_dtype=torch.bfloat16, out=out)
        return f, "scaled_mm"
    _, tile, cl, swap, instm, pdl, pre = arm.split(":")
    tac = (tuple(int(v) for v in tile.split("x")), tuple(int(v) for v in cl.split("x")),
           swap == "1", int(instm))
    g = pw.Mxfp8PreGemm(N, K, R, tac, use_pdl=pdl == "1", prewait=int(pre))

    def f(i, out):
        g(xq, xs, W[i], WS[i], out)
    return f, f"stages {g.stages}"


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
        if not args.warm:
            flush.zero_()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000 / args.chain)
    return statistics.median(ts[2:])


flush = torch.empty(512 * 1024 * 1024, device=dev, dtype=torch.uint8)
res = []
prod_us = time_chain(None, 128, True) if prod is not None else None
if prod is not None:
    print(f"producer alone {prod_us:.2f} us/call", flush=True)
for R in [int(r) for r in args.rows.split(",")]:
    x = torch.randn(R, K, device=dev, dtype=torch.bfloat16)
    xq, xs = mxfp8_quantize(x, True)
    ref = torch.empty(R, N, device=dev, dtype=torch.bfloat16)
    torch._scaled_mm(xq, W[0].t(), scale_a=xs.view(torch.float8_e8m0fnu),
                     scale_b=WS[0].view(torch.float8_e8m0fnu), out_dtype=torch.bfloat16,
                     out=ref)
    for arm in args.arms.split(","):
        try:
            t0 = time.time()
            f, info = mk_arm(arm, R, xq, xs)
            out = torch.full((R, N), float("nan"), device=dev, dtype=torch.bfloat16)
            f(0, out)
            torch.cuda.synchronize()
            err = ((out.float() - ref.float()).abs().max() / ref.float().abs().max()).item()
            row = dict(R=R, arm=arm, info=info, rel=err, bitwise=bool(torch.equal(out, ref)),
                       build_s=round(time.time() - t0, 1))
            row["us"] = round(time_chain(f, R, False), 2)
            if prod is not None:
                tot = time_chain(f, R, True)
                row.update(with_prod_us=round(tot, 2), exposed_us=round(tot - prod_us, 2))
        except Exception as e:  # report and continue
            row = dict(R=R, arm=arm, error=f"{type(e).__name__}: {str(e)[:300]}")
        print(json.dumps(row), flush=True)
        res.append(row)
if args.out:
    json.dump(res, open(args.out, "w"), indent=1)
