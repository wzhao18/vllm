#!/usr/bin/env python3
"""M9a ladder: KDA in_proj Q|K|V GEMM with the column-wise conv + SiLU epilogue.

Arms per M (3 sequences, F16K = 3 x 5461, F8K = 3 x 2731), N = 3 x 12288,
K = 7168, MXFP8 operands; burst (synchronized per call) and sustained
(back-to-back, with mean SM clock and board power) kernel times:
  nvjet          cuBLASLt in_proj (production)
  qkv_conv       production conv + SiLU pass
  ref            nvjet + qkv_conv
  a  cskel4      CuTe GEMM, epilogue releases TMEM only (mainloop)
  b  cskel1      + drain, BF16 shared staging
  c  cskel2      + conv/SiLU column pass, global stores predicated off
  c' cskel3      c with the math only on the SMSP 2/3 warps
  d1 full        fused conv epilogue (conv_t) + fix-up kernel (bitwise vs ref)
  e  full t2     branch-free column pass (conv_t2) + fix-up kernel (bitwise)
"""

import argparse
import json
import statistics
import threading
import time

import torch
from pgemm_conv_bench import cublas, kernel_times, operands

ARMS = [
    ("a", "cskel4"),
    ("b", "cskel1"),
    ("c", "cskel2"),
    ("cp", "cskel3"),
    ("d1", "ct1"),
    ("e", ""),
    ("en", "nrs"),
    ("es", "sc"),
    ("e23", "c23"),
    ("enl", "nolds"),
    ("ens", "nost"),
    ("l1", "lp1"),
    ("l2", "lp2"),
    ("l4", "lp4"),
    # Probe: e without the d > 2^126 tracking (2 VIMNMX per column pair-row).
    ("eg", "nogm"),
]
#: cluster -> (swizzle, raster) as in the B3 tactics.
RASTER = {(2, 1): (8, "n"), (4, 1): (8, "n"), (2, 4): (4, "m"), (2, 2): (8, "n")}


def sustained(fn, seconds=1.5):
    """Back-to-back mean time per call (us), mean SM clock (MHz), power (W)."""
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    one = max(time.perf_counter() - t0, 1e-5)
    iters = max(20, int(seconds / one))
    samples, stop = [], threading.Event()

    def sample():
        while not stop.is_set():
            try:
                samples.append((torch.cuda.clock_rate(), torch.cuda.power_draw() / 1000))
            except Exception:  # noqa: BLE001
                return
            time.sleep(0.005)

    th = threading.Thread(target=sample)
    th.start()
    for _ in range(iters // 4):
        fn()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    stop.set()
    th.join()
    tail = samples[len(samples) // 3 :] or samples or [(0, 0)]
    return {
        "us": round(s.elapsed_time(e) * 1000 / iters, 1),
        "mhz": round(statistics.mean(c for c, _ in tail)),
        "w": round(statistics.mean(w for _, w in tail)),
        "iters": iters,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", default="5461,2731")
    ap.add_argument("--p", type=int, default=12288)
    ap.add_argument("--k", type=int, default=7168)
    ap.add_argument("--clusters", default="2x1,4x1")
    ap.add_argument("--arms", default="a,b,c,e")
    ap.add_argument("--sus", default="nvjet,ref,a,e")
    ap.add_argument("--cool", type=float, default=2.0, help="idle s before bursts")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--ncu", action="store_true", help="one profiled call per arm")
    args = ap.parse_args()

    from vllm.models.kimi_k3.nvidia.ops import kda_conv
    from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm.kda_conv_gemm import (  # noqa: E501
        KdaConvGemm,
    )

    p, k = args.p, args.k
    n = 3 * p
    clusters = [tuple(int(v) for v in c.split("x")) for c in args.clusters.split(",")]
    arms = [a for a in ARMS if a[0] in args.arms.split(",")]
    sus = set(args.sus.split(","))
    for sl in [int(v) for v in args.seqs.split(",")]:
        lens = [sl] * 3
        m = sum(lens)
        xq, xs, wq, ws = operands(m, n, k)
        cw = (torch.randn(n, 4, device="cuda") * 0.5).contiguous()
        qsl = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device="cuda")
        lines = torch.arange(1, 4, dtype=torch.int32, device="cuda")
        has_init = torch.tensor([False, True, False], device="cuda")
        state0 = torch.randn(5, n, 4, device="cuda").to(torch.bfloat16).float()[:, :, :3]
        proj = cublas(xq, xs, wq, ws)
        st_ref = state0.clone()
        ref = kda_conv.qkv_conv(proj, cw, st_ref, qsl, lines, has_init, p)
        st_t = state0.clone()

        def f_nvjet():
            cublas(xq, xs, wq, ws)

        def f_conv():
            kda_conv.qkv_conv(proj, cw, st_t, qsl, lines, has_init, p)

        def f_ref():
            o = cublas(xq, xs, wq, ws)
            kda_conv.qkv_conv(o, cw, st_t, qsl, lines, has_init, p)

        res = {"m": m, "n": n, "k": k}
        refs = () if args.ncu else (("nvjet", f_nvjet), ("qkv_conv", f_conv), ("ref", f_ref))
        for name, fn in refs:
            time.sleep(args.cool)
            res[name] = kernel_times(fn, iters=args.iters)
            if name in sus:
                res[name + "_sus"] = sustained(fn)
        print(json.dumps(res), flush=True)
        for cl in clusters:
            for arm, flag in arms:
                tac = ((512, 256), cl, 256, *RASTER[cl], 0, 0, flag)
                row = {"m": m, "cluster": "%dx%d" % cl, "arm": arm}
                try:
                    g = KdaConvGemm(wq, ws, cw, p, tactic=tac)
                    st = state0.clone()
                    t0 = time.perf_counter()
                    y, _ = g(xq, xs, st, qsl, lines, has_init)
                    torch.cuda.synchronize()
                    row["first_call_s"] = round(time.perf_counter() - t0, 1)
                    if arm in ("d1", "e", "en", "es", "e23", "l1", "l2", "l4"):
                        row["bitwise_out"] = bool(torch.equal(y, ref))
                        row["bitwise_state"] = bool(torch.equal(st, st_ref))
                        if not row["bitwise_out"]:
                            row["mismatch"] = int((y != ref).sum())

                    def f_arm():
                        g(xq, xs, st, qsl, lines, has_init)

                    if args.ncu:
                        f_arm()
                        torch.cuda.synchronize()
                        torch.cuda.cudart().cudaProfilerStart()
                        f_arm()
                        torch.cuda.synchronize()
                        torch.cuda.cudart().cudaProfilerStop()
                        print(json.dumps(row), flush=True)
                        continue
                    time.sleep(args.cool)
                    row["burst"] = kernel_times(f_arm, iters=args.iters)
                    if arm in sus:
                        row["sus"] = sustained(f_arm)
                except Exception as e:  # noqa: BLE001
                    row["error"] = repr(e)[:400]
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
