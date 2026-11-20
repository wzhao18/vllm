"""GEMM-EX1 microbench: MXFP8 decode GEMM (kda_in etc.), cold weights, CUDA-graph chain.

Arms: scaled_mm (cuBLASLt nvjet heuristic), fi (flashinfer mm_mxfp8 cute-dsl, image),
own:<tile>:<cluster>:<swap>:<instm>:<pdl>:<pre> (vendored sm107 kernel, PDL + pre-wait
weight prologue). Optional producer (--prod US): a spin kernel on every SM that triggers
launch_dependents at entry, run before every GEMM (exposed = chain - producer chain).
"""
import argparse, json, statistics, sys, time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=12448)
ap.add_argument("--k", type=int, default=7168)
ap.add_argument("--rows", default="128,256,384")
ap.add_argument("--arms", default="scaled_mm,fi")
ap.add_argument("--chain", type=int, default=16)
ap.add_argument("--reps", type=int, default=15)
ap.add_argument("--prod", type=float, default=0.0, help="producer spin iterations (k)")
ap.add_argument("--prod_ctas", type=int, default=1, help="producer CTAs per SM")
ap.add_argument("--prod_smem", type=int, default=0, help="producer dynamic smem bytes")
ap.add_argument("--prod_carveout", type=int, default=-1, help="producer preferred smem carve-out (-1 default)")
ap.add_argument("--out", default="")
ap.add_argument("--nvml", action="store_true")
ap.add_argument("--warm", action="store_true", help="one weight copy, no L2 flush")
args = ap.parse_args()

import flashinfer
from flashinfer import mxfp8_quantize

dev = torch.device("cuda")
torch.manual_seed(0)
N, K = args.n, args.k
nsm = torch.cuda.get_device_properties(0).multi_processor_count
print(f"flashinfer {flashinfer.__version__} sms {nsm} N {N} K {K}", flush=True)

wbytes = N * K
ncopy = max(args.chain, -(-1_600_000_000 // wbytes))
w_bf = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.05
w_q, w_s = mxfp8_quantize(w_bf, True)
W = [w_q.clone() for _ in range(1 if args.warm else ncopy)] * (ncopy if args.warm else 1)
WS = [w_s.clone() for _ in range(ncopy)]
del w_bf
print(f"weights: {ncopy} copies x {wbytes/1e6:.1f} MB, scale {w_s.numel()} B {w_s.dtype}", flush=True)

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.sm107 import prewait as pw

prod = None
if args.prod > 0:
    prod = pw.Spin(int(args.prod * 1000), nsm * args.prod_ctas, args.prod_smem)
    if args.prod_carveout >= 0:
        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import prepare_shared_carveout
        prod.compiled = prepare_shared_carveout(prod.compiled, 0, args.prod_carveout)


WDL = {}
LAST_G = None


def fi_tactic(R):
    from flashinfer.gemm.kernels.utils import _select_sm107_mm_mxfp8_cute_dsl_tactic as sel
    return sel(R, N, K, nsm)


def mk_arm(arm, R, xq, xs):
    if arm == "scaled_mm":
        def f(i, out):
            torch._scaled_mm(xq, W[i].t(), scale_a=xs.view(torch.float8_e8m0fnu),
                             scale_b=WS[i].view(torch.float8_e8m0fnu),
                             out_dtype=torch.bfloat16, out=out)
        return f, "rn"
    if arm == "ltpin":
        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.ltpin import LtPin
        lp = LtPin("bench", N, K, [R], dtype="mxfp8", device=dev)
        def f(i, out):
            lp.run(W[i], xq, out, WS[i], xs)
        return f, "rn"
    if arm == "fi":
        def f(i, out):
            flashinfer.mm_mxfp8(xq, W[i].t(), xs, WS[i], out=out, out_dtype=torch.bfloat16,
                                backend="cute-dsl")
        return f, "rn"
    if arm.startswith("own"):
        parts = arm.split(":")
        flags = [t for t in parts if t in ("q", "dl", "dc", "x2") or t[:1] in ("p", "f", "k", "s", "r", "d", "o") and t[1:].isdigit()]
        parts = [t for t in parts if t not in flags]
        q = "q" in flags
        dbg = ([int(t[1:]) for t in flags if t[:1] == "d" and t[1:].isdigit()] or [0])[0]
        prod_w = ([int(t[1:]) for t in flags if t.startswith("p")] or [1])[0]
        pf = ([int(t[1:]) for t in flags if t.startswith("f")] or [0])[0]
        tk = ([int(t[1:]) for t in flags if t.startswith("k")] or [128])[0]
        ms = ([int(t[1:]) for t in flags if t.startswith("s")] or [0])[0]
        kr = ([int(t[1:]) for t in flags if t.startswith("r")] or [0])[0]
        ro = ([int(t[1:]) for t in flags if t.startswith("o")] or [0])[0]
        if parts[1] == "auto":
            _, _, pdl, pre = parts
            tac = fi_tactic(R)
        else:
            _, tile, cl, swap, instm, pdl, pre = parts
            tac = (tuple(int(v) for v in tile.split("x")), tuple(int(v) for v in cl.split("x")),
                   swap == "1", int(instm))
        split, WW = None, W
        if "dl" in flags or "dc" in flags:
            key = tac[0][1]
            if key not in WDL:
                WDL[key] = [pw.die_weight(w, key) for w in W]
            WW = [w for w, _ in WDL[key]] if "dl" in flags else W
            split = WDL[key][0][1]
        g = pw.Mxfp8PreGemm(N, K, R, tac, use_pdl=pdl == "1", prewait=int(pre), quiet=q, dbg=dbg, producers=prod_w, prefetch=pf, tile_k=tk, die_split=split, max_stages=ms)
        print(f"  {arm}: tactic {tac} stages {g.stages}", flush=True)
        global LAST_G
        LAST_G = g
        def f(i, out):
            g(xq, xs, WW[i], WS[i], out)
        return f, g.out_layout
    raise ValueError(arm)


def view_out(out, layout, R):
    return out if layout == "rn" else torch.as_strided(out, (R, N), (1, R))


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
    for r in range(args.reps + 2):
        # rotate the weight copies between replays: graph uses fixed copies; flush L2 with a
        # big memset of an unrelated buffer instead
        if not args.warm:
            flush.zero_()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000 / args.chain)
    return statistics.median(ts[2:])


flush = torch.empty(512 * 1024 * 1024, device=dev, dtype=torch.uint8)
res = []
prod_us = None
if prod is not None:
    prod_us = time_chain(None, 128, True)
    print(f"producer alone {prod_us:.2f} us/call", flush=True)
for R in [int(r) for r in args.rows.split(",")]:
    try:
        print(f"R {R}: FI default tactic {fi_tactic(R)}", flush=True)
    except Exception as e:
        print(f"R {R}: FI tactic lookup failed {e}", flush=True)
    x = torch.randn(R, K, device=dev, dtype=torch.bfloat16)
    xq, xs = mxfp8_quantize(x, True)
    ref = torch.empty(R, N, device=dev, dtype=torch.bfloat16)
    torch._scaled_mm(xq, W[0].t(), scale_a=xs.view(torch.float8_e8m0fnu),
                     scale_b=WS[0].view(torch.float8_e8m0fnu), out_dtype=torch.bfloat16, out=ref)
    for arm in args.arms.split(","):
        try:
            t0 = time.time()
            f, layout = mk_arm(arm, R, xq, xs)
            out = torch.empty(R, N, device=dev, dtype=torch.bfloat16)
            f(0, out); torch.cuda.synchronize()
            got = view_out(out, layout, R).float()
            err = ((got - ref.float()).abs().max() / ref.float().abs().max()).item()
            bit = bool(torch.equal(view_out(out, layout, R), ref))
            tb = time.time() - t0
            us = time_chain(f, R, False)
            try:
                import pynvml as nv
                nv.nvmlInit(); hh = nv.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
                clk = (nv.nvmlDeviceGetClockInfo(hh, nv.NVML_CLOCK_SM), round(nv.nvmlDeviceGetPowerUsage(hh) / 1000), nv.nvmlDeviceGetCurrentClocksEventReasons(hh))
            except Exception:
                clk = None
            row = dict(R=R, arm=arm, us=round(us, 2), rel=err, bitwise_vs_scaled_mm=bit,
                       build_s=round(tb, 1), nvml=clk)
            if LAST_G is not None and getattr(LAST_G, "alpha", None) is not None and LAST_G.alpha.numel() > 2048:
                from ex1_st2_ana import summarize
                torch.cuda.synchronize()
                row["st"] = {k: round(v, 2) for k, v in summarize(LAST_G.alpha).items()}
            LAST_G = None
            if prod is not None:
                tot = time_chain(f, R, True)
                row.update(prod_us=round(prod_us, 2), with_prod_us=round(tot, 2),
                           exposed_us=round(tot - prod_us, 2))
        except Exception as e:  # report and continue with the next arm
            row = dict(R=R, arm=arm, error=f"{type(e).__name__}: {str(e)[:300]}")
        print(json.dumps(row), flush=True)
        res.append(row)
if args.out:
    json.dump(res, open(args.out, "w"), indent=1)
if args.nvml:
    try:
        import pynvml as nv
        nv.nvmlInit(); h = nv.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
        print(json.dumps(dict(nvml_sm_mhz=nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM), nvml_mem_mhz=nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_MEM), power_w=nv.nvmlDeviceGetPowerUsage(h) / 1000, cap_w=nv.nvmlDeviceGetEnforcedPowerLimit(h) / 1000, throttle=hex(nv.nvmlDeviceGetCurrentClocksEventReasons(h)))), flush=True)
    except Exception as e:
        print("nvml failed", e, flush=True)
