#!/usr/bin/env python3
"""P-GEMM r6: BF16 GEMM energy ladder at the power cap (1 GPU).

Question (07:25 audit, coordinator 08:48): is the BF16 body's 1.14-1.21x of the
compulsory dynamic J an operand-data effect or a form term? Per shape and data
set, arms run back to back for --secs per round (rounds interleave the arms):
  nvjet       F.linear (cuBLASLt heuristic)
  base        sm_107 BF16 persistent base (Bf16Gemm, 256x256 2-CTA, TMA store)
  base_noepi  mainloop only (epilogue waits for the accumulator and releases it)
  base_nomma  loads only (no tcgen05.mma, no epilogue)
Per arm: median us/call, J/call (NVML total-energy counter of THIS GPU), SM clock,
board power; plus a THRASH time (512 MB L2 flush before each timed call).
Data: real = randn activations, randn * 0.02 weights (BF16-like: random sign and
mantissa, small exponent spread); uniform = U(-1, 1) activations, U(-0.035,
0.035) weights; zero = all-zero operands (MMA + delivery floor of the datapath).
"""

import argparse
import json
import statistics
import threading
import time

import torch

SHAPES = {"lat_up": (16384, 7168, 3584), "g_proj": (16384, 12288, 7168)}


def operands(kind, m, n, k, dev):
    g = torch.Generator(device=dev).manual_seed(n * 7 + k)
    if kind == "real":
        x = torch.randn(m, k, device=dev, dtype=torch.bfloat16, generator=g)
        w = torch.randn(n, k, device=dev, dtype=torch.bfloat16, generator=g) * 0.02
    elif kind == "uniform":
        x = torch.empty(m, k, device=dev, dtype=torch.bfloat16).uniform_(-1, 1, generator=g)
        w = torch.empty(n, k, device=dev, dtype=torch.bfloat16).uniform_(-0.035, 0.035, generator=g)
    else:
        x = torch.zeros(m, k, device=dev, dtype=torch.bfloat16)
        w = torch.zeros(n, k, device=dev, dtype=torch.bfloat16)
    return x, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="lat_up,g_proj")
    ap.add_argument("--data", default="real,uniform,zero")
    ap.add_argument("--arms", default="nvjet,base,base_noepi,base_nomma")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--secs", type=float, default=2.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import pynvml as N

    from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm import bf16_gemm
    from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm.dense_gemm_bf16_sm107 import (
        PersistentDenseGemmKernel,
    )

    class _NoEpi(PersistentDenseGemmKernel):
        skel_noepi = True

    class _NoMma(PersistentDenseGemmKernel):
        skel_noepi = True
        skel_nomma = True

    class Bf16NoEpi(bf16_gemm.Bf16Gemm):
        kernel_cls = _NoEpi

    class Bf16NoMma(bf16_gemm.Bf16Gemm):
        kernel_cls = _NoMma

    dev = torch.device("cuda")
    N.nvmlInit()
    h = N.nvmlDeviceGetHandleByUUID("GPU-" + str(torch.cuda.get_device_properties(0).uuid))
    samples, stop = [], [False]

    def poll():
        while not stop[0]:
            try:
                samples.append((
                    time.perf_counter(),
                    N.nvmlDeviceGetClockInfo(h, N.NVML_CLOCK_SM),
                    N.nvmlDeviceGetPowerUsage(h) / 1e3,
                    N.nvmlDeviceGetTotalEnergyConsumption(h),
                ))  # fmt: skip
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.004)

    threading.Thread(target=poll, daemon=True).start()
    print(json.dumps({"power_limit_w": N.nvmlDeviceGetEnforcedPowerLimit(h) / 1e3}), flush=True)
    flush = torch.empty(512 << 20, dtype=torch.uint8, device=dev)
    kern = {"base": bf16_gemm.Bf16Gemm(), "base_noepi": Bf16NoEpi(), "base_nomma": Bf16NoMma()}
    rows = []
    for shp in args.shapes.split(","):
        m, n, k = SHAPES[shp]
        for kind in args.data.split(","):
            x, w = operands(kind, m, n, k, dev)
            out = torch.empty(m, n, device=dev, dtype=torch.bfloat16)
            ref = torch.nn.functional.linear(x, w)
            fns = {}
            for arm in args.arms.split(","):
                if arm == "nvjet":
                    fns[arm] = lambda: torch.mm(x, w.t(), out=out)  # noqa: B023
                    continue
                g = kern[arm]
                try:
                    g(x, w, out)
                    torch.cuda.synchronize()
                except Exception as e:  # noqa: BLE001
                    print(json.dumps({"shape": shp, "data": kind, "arm": arm, "error": repr(e)[:300]}), flush=True)
                    continue
                if arm == "base":
                    print(json.dumps({"shape": shp, "data": kind, "base_bitwise": bool(torch.equal(out, ref))}), flush=True)
                fns[arm] = lambda g=g: g(x, w, out)  # noqa: B023
            res = {a: {"us": [], "j": [], "clk": [], "w": []} for a in fns}
            for _ in range(args.rounds):
                for arm, fn in fns.items():
                    fn()
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    calls = 0
                    while time.perf_counter() - t0 < args.secs:
                        for _ in range(4):
                            fn()
                        calls += 4
                        torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    # last second of the round: steady state at the cap
                    ss = [s for s in samples if t1 - 1.0 <= s[0] <= t1]
                    if len(ss) < 2:
                        continue
                    us = (t1 - t0) / calls * 1e6
                    per_s = calls / (t1 - t0)
                    de = (ss[-1][3] - ss[0][3]) / 1e3  # J
                    dt = ss[-1][0] - ss[0][0]
                    res[arm]["us"].append(us)
                    res[arm]["j"].append(de / (dt * per_s))
                    res[arm]["clk"].append(statistics.median(s[1] for s in ss))
                    res[arm]["w"].append(statistics.median(s[2] for s in ss))
            for arm, fn in fns.items():
                thr = []
                for _ in range(10):
                    flush.add_(1)
                    s1, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    s1.record()
                    fn()
                    e1.record()
                    e1.synchronize()
                    thr.append(s1.elapsed_time(e1) * 1e3)
                r = res[arm]
                row = {
                    "shape": shp, "data": kind, "arm": arm,
                    "us": round(statistics.median(r["us"]), 1) if r["us"] else None,
                    "j_call": round(statistics.median(r["j"]), 4) if r["j"] else None,
                    "j_rounds": [round(v, 4) for v in r["j"]],
                    "clk_mhz": statistics.median(r["clk"]) if r["clk"] else None,
                    "power_w": round(statistics.median(r["w"]), 0) if r["w"] else None,
                    "thrash_us": round(statistics.median(thr), 1),
                    "tflops": round(2 * m * n * k / (statistics.median(r["us"]) * 1e6), 1) if r["us"] else None,
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
            del x, w, out, ref
            torch.cuda.empty_cache()
    stop[0] = True
    if args.out:
        with open(args.out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
