#!/usr/bin/env python3
"""P-GEMM round 4: ONE sustained (power-capped) harness for the sm_107 GEMM
forks vs nvjet at the prefill shapes (M16384 K7168, N in --ns).

Every arm runs back to back for ``--secs`` per round, rounds interleave the
arms (the F16K prefill step runs at the VR200 power cap, so a GEMM's in-step
time follows its energy). Operands: randn -> MXFP8 (real E4M3 codes, real E8M0
block scales). Per arm: median us/call over the last second of each round,
NVML SM clock, board power, energy per call (total-energy counter of THIS GPU,
looked up by UUID). ``--ncu``: one profiled call per arm (cudaProfilerStart/
Stop around it) for the cold-cache DRAM / L2 / cycle counters.

Arms (all 512x256 c2x4 m-raster swizzle 4 unless named):
  nvjet        torch._scaled_mm, BF16 out (cuBLASLt heuristic)
  base         PlainGemm, BF16 out, TMA store after the full TMEM drain
  base_drain   PlainGemm "drain" epilogue (BF16 out)
  base_early   PlainGemm "early" TMEM release (BF16 out)
  base_noepi   PlainGemm mainloop only (epilogue releases TMEM, no stores)
  t256         PlainGemm 256x256 c2x4 m4 (two accumulator stages), BF16 out
  b3           SharedSituGemm DEFAULT_TACTIC (SiTU + MXFP8 out, N/2 act cols)
  b3_s6        SharedSituGemm swizzle 6
  kda_a        KdaConvGemm arm a (cskel4: mainloop, epilogue releases TMEM)
"""

import argparse
import json
import statistics
import threading
import time

import torch

T512 = ((512, 256), (2, 4), 256, 4, "m", 0, 0)
ARMS = {
    "base": T512 + ("",),
    "base_drain": T512 + ("drain",),
    "base_early": T512 + ("early",),
    "base_noepi": T512 + ("noepi",),
    "t256": ((256, 256), (2, 4), 256, 4, "m", 0, 0, ""),
    "c2x2": ((512, 256), (2, 2), 256, 8, "n", 0, 0, ""),
    "c2x1": ((512, 256), (2, 1), 256, 8, "n", 0, 0, ""),
    "c2x2_drain": ((512, 256), (2, 2), 256, 8, "n", 0, 0, "drain"),
    "c2x1_drain": ((512, 256), (2, 1), 256, 8, "n", 0, 0, "drain"),
}
MIXED = {"mix24_21": ((2, 4), (2, 1)), "mix22_21": ((2, 2), (2, 1))}


def real_operands(m, n, k, dev):
    from flashinfer import mxfp8_quantize

    g = torch.Generator(device=dev).manual_seed(n)
    x = torch.randn(m, k, device=dev, dtype=torch.bfloat16, generator=g)
    w = torch.randn(n, k, device=dev, dtype=torch.bfloat16, generator=g) * 0.02
    xq, xs = mxfp8_quantize(x, is_sf_swizzled_layout=True)
    wq, ws = mxfp8_quantize(w, is_sf_swizzled_layout=True)
    return xq, xs.view(torch.uint8).reshape(-1), wq, ws.view(torch.uint8).reshape(-1)


def scaled_mm(x, sx, w, sw, out=None):  # noqa: ARG001
    return torch._scaled_mm(
        x, w.t(),
        scale_a=sx.view(torch.float8_e8m0fnu), scale_b=sw.view(torch.float8_e8m0fnu),
        out_dtype=torch.bfloat16,
    )  # fmt: skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=16384)
    ap.add_argument("--k", type=int, default=7168)
    ap.add_argument("--shapes", default="12288:7168,36864:7168,49376:7168,7168:12288,7168:6144,12512:7168")
    ap.add_argument("--arms", default="nvjet,base,base_drain,base_noepi")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--secs", type=float, default=2.0)
    ap.add_argument("--ncu", action="store_true")
    ap.add_argument("--nonvml", action="store_true")
    args = ap.parse_args()

    import pynvml as N

    from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm import (
        kda_conv_gemm as kcg,
    )
    from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm import (
        shared_situ_gemm as ssg,
    )

    dev = torch.device("cuda")
    if not args.nonvml:
        N.nvmlInit()
    h = None if args.nonvml else N.nvmlDeviceGetHandleByUUID("GPU-" + str(torch.cuda.get_device_properties(0).uuid))
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

    want = args.arms.split(",")
    compiled = {}
    if any(a.endswith("_die") for a in want):
        # Green contexts first: creating them while the NVML poller runs crashed (segfault in NVML).
        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common import dies

        torch.zeros(1, device=dev)
        compiled["die_streams"] = [dies.domain_stream(dev.index or 0, d, coscheduled=8) for d in range(2)]
        torch.cuda.synchronize()
    if not args.ncu and not args.nonvml:
        threading.Thread(target=poll, daemon=True).start()
    import cutlass.utils as cu

    torch.zeros(1, device="cuda")
    try:
        hw = cu.HardwareInfo()
        print(json.dumps({"max_active_clusters": {c: hw.get_max_active_clusters(c) for c in (2, 4, 8)}}), flush=True)
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"max_active_clusters": repr(e)[:100]}), flush=True)
    m = args.m
    for shp in args.shapes.split(","):
        bf16 = shp.startswith("b")
        n, k = (int(v) for v in shp.lstrip("b").split(":"))
        if bf16:
            g0 = torch.Generator(device=dev).manual_seed(n)
            x = torch.randn(m, k, device=dev, dtype=torch.bfloat16, generator=g0)
            w = torch.randn(n, k, device=dev, dtype=torch.bfloat16, generator=g0) * 0.02
            sx = sw = None
            ref = torch.nn.functional.linear(x, w)
        else:
            x, sx, w, sw = real_operands(m, n, k, dev)
            ref = scaled_mm(x, sx, w, sw)
        out = torch.empty(m, n, device=dev, dtype=torch.bfloat16)
        fns, info = {}, {}
        if any(a.endswith(("_die", "_split")) for a in want):
            from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common import dies

            h, sk = n // 2, k // 32
            w_h = [w[:h], w[h:]]
            s_h = [None, None] if bf16 else [sw[: h * sk], sw[h * sk :]]
            w_loc = [dies.localized_like(w_h[d].contiguous(), [d]) for d in range(2)]
            s_loc = [None, None] if bf16 else [dies.localized_like(s_h[d].contiguous(), [d]) for d in range(2)]
            if not bf16:
                w_loc = [t.view(torch.float8_e4m3fn).view(h, k) for t in w_loc]
                s_loc = [t.view(torch.uint8).reshape(-1) for t in s_loc]
            else:
                w_loc = [t.view(torch.bfloat16).view(h, k) for t in w_loc]
            if "die_streams" not in compiled:
                compiled["die_streams"] = [dies.domain_stream(dev.index or 0, d, coscheduled=8) for d in range(2)]
            streams = compiled["die_streams"]

            def half(d, wt, st):
                return torch.nn.functional.linear(x, wt) if bf16 else scaled_mm(x, sx, wt, st)

            def split_seq(wt=w_h, st=s_h):
                return [half(d, wt[d], st[d]) for d in range(2)]

            def die_conc():
                ev = torch.cuda.current_stream().record_event()
                outs, evs = [None, None], []
                for d in range(2):
                    streams[d].wait_event(ev)
                    with torch.cuda.stream(streams[d]):
                        outs[d] = half(d, w_loc[d], s_loc[d])
                        evs.append(streams[d].record_event())
                for e in evs:
                    torch.cuda.current_stream().wait_event(e)
                return outs

            pre = "bf16" if bf16 else "nvjet"
            for name, fn in ((pre + "_split", split_seq), (pre + "_die", die_conc)):
                if name in want:
                    o = fn()
                    torch.cuda.synchronize()
                    info[name] = {"bitwise": bool(torch.equal(torch.cat(o, 1), ref))}
                    fns[name] = fn
        if "bf16" in want and bf16:
            fns["bf16"] = lambda: torch.nn.functional.linear(x, w)
        if bf16:
            pass
        elif "nvjet" in want:
            fns["nvjet"] = lambda: scaled_mm(x, sx, w, sw, out)
        for name, tac in ARMS.items():
            if name not in want or bf16:
                continue
            if compiled.get(name, 0) is None:
                continue
            g = kcg.PlainGemm(w, sw, tac)
            if name in compiled:  # problem / ld are runtime args: reuse across N
                g._compiled = compiled[name]
            out.zero_()
            t0 = time.perf_counter()
            try:
                g(x, sx, out)
                torch.cuda.synchronize()
            except Exception as e:  # noqa: BLE001
                compiled[name] = None
                info[name] = {"error": repr(e)[:200]}
                continue
            compiled[name] = g._compiled
            info[name] = {"first_s": round(time.perf_counter() - t0, 1)}
            if "noepi" not in tac[7]:
                info[name]["bitwise"] = bool(torch.equal(out, ref))
            fns[name] = lambda g=g: g(x, sx, out)
        for name, (pref, fb) in MIXED.items():
            if name not in want or bf16:
                continue
            try:
                g = kcg.MixedGemm(w, sw, (512, 256), pref, fb)
                out.zero_()
                g(x, sx, out)
                torch.cuda.synchronize()
                info[name] = {"bitwise": bool(torch.equal(out, ref)), "pmac": g.pmac, "fmac": g.fmac}
                fns[name] = lambda g=g: g(x, sx, out)
            except Exception as e:  # noqa: BLE001
                info[name] = {"error": repr(e)[:200]}
        if any(a.startswith("b3") for a in want) and n % 512 == 0 and not bf16:
            from vllm.models.kimi_k3.nvidia.ops import mxfp8_act

            inter = n // 2
            wi, swi = w.clone(), sw.clone()
            ssg.interleave_gate_up_rows(wi, swi)
            act = mxfp8_act.Mxfp8Buffer(m, inter, dev)
            for name, sw_ in (("b3", 4), ("b3_s6", 6)):
                if name not in want:
                    continue
                t = list(ssg.DEFAULT_TACTIC)
                t[3] = sw_
                g = ssg.SharedSituGemm(wi, swi, 4.0, 25.0, tactic=tuple(t))
                t0 = time.perf_counter()
                g(x, sx, act)
                torch.cuda.synchronize()
                info[name] = {"first_s": round(time.perf_counter() - t0, 1), "tactic": str(t)}
                fns[name] = lambda g=g, act=act: g(x, sx, act)
        if "kda_a" in want and n % 3 == 0 and not bf16:
            p = n // 3
            lens = [m // 3, m // 3, m - 2 * (m // 3)]
            cw = (torch.randn(n, 4, device=dev) * 0.5).contiguous()
            qsl = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device=dev)
            lines = torch.arange(1, 4, dtype=torch.int32, device=dev)
            has_init = torch.tensor([False, True, False], device=dev)
            st = torch.randn(5, n, 4, device=dev).to(torch.bfloat16).float()[:, :, :3]
            tac = ((512, 256), (2, 4), 256, 4, "m", 0, 0, "cskel4")
            try:
                g = kcg.KdaConvGemm(w, sw, cw, p, tactic=tac)
                t0 = time.perf_counter()
                g(x, sx, st, qsl, lines, has_init)
                torch.cuda.synchronize()
                info["kda_a"] = {"first_s": round(time.perf_counter() - t0, 1)}
                fns["kda_a"] = lambda g=g, st=st: g(x, sx, st, qsl, lines, has_init)
            except Exception as e:  # noqa: BLE001
                info["kda_a"] = {"error": repr(e)[:300]}
        print(json.dumps({"n": n, "k": k, "info": info}), flush=True)
        if args.ncu:
            for name, fn in fns.items():
                fn()
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStart()
                fn()
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStop()
                print(json.dumps({"n": n, "k": k, "ncu_arm": name}), flush=True)
            continue
        res = {name: [] for name in fns}
        for _ in range(args.rounds):
            for name, fn in fns.items():
                ev = []
                t0 = time.perf_counter()
                while time.perf_counter() - t0 < args.secs:
                    a = torch.cuda.Event(enable_timing=True)
                    b = torch.cuda.Event(enable_timing=True)
                    a.record()
                    fn()
                    b.record()
                    ev.append((a, b))
                    if len(ev) % 8 == 0:
                        torch.cuda.synchronize()
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                ms = [a.elapsed_time(b) for a, b in ev]
                last, cum = [], 0.0
                for v in reversed(ms):
                    if cum > 1000.0:
                        break
                    last.append(v)
                    cum += v
                win = [s for s in samples if t1 - cum * 1e-3 <= s[0] <= t1]
                if len(win) >= 2:
                    de = (win[-1][3] - win[0][3]) / 1e3
                    dt = win[-1][0] - win[0][0]
                    calls = len(last) * dt / (cum * 1e-3)
                    jpc = de / calls if calls else 0.0
                    clk = statistics.median(s[1] for s in win)
                    pw = statistics.median(s[2] for s in win)
                else:
                    jpc, clk, pw = 0.0, 0, 0.0
                res[name].append((statistics.median(last) * 1e3, clk, pw, jpc))
        base = statistics.median(r[0] for r in res[next(iter(res))])
        for name, rs in res.items():
            us = statistics.median(r[0] for r in rs)
            print(json.dumps({
                "n": n, "k": k, "v": name, "us": round(us, 1), "rounds": [round(r[0], 1) for r in rs],
                "x_first": round(us / base, 3),
                "mhz": round(statistics.median(r[1] for r in rs)),
                "w": round(statistics.median(r[2] for r in rs)),
                "mj_per_call": round(1e3 * statistics.median(r[3] for r in rs), 2),
                "mj_rounds": [round(1e3 * r[3], 1) for r in rs],
            }), flush=True)  # fmt: skip
        del x, sx, w, sw, ref, out, fns
        torch.cuda.empty_cache()
    stop[0] = True


if __name__ == "__main__":
    main()
