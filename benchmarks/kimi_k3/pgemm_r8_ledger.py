#!/usr/bin/env python3
"""P-GEMM R8: per-row energy ledger of the prefill GEMM family on the DEPLOYED kernels.

R7 (pgemm_r7_ledger.py) priced KDA in_proj as one N49376 nvjet GEMM, the shared gate_up as a
plain MXFP8 GEMM and w_uk as the stock bmm, and ran the four pinned BF16 shapes on cuBLAS's
heuristic. R8 runs what the prefill forward launches (int46): KDA q|k|v = KdaConvGemm (conv +
SiLU epilogue, production tactic) + the g|f_a|beta MXFP8 rest; shared gate_up = SharedSituGemm
on the die-split tactic; the fused UkQuery; the pinned cuBLASLt tactics (gemm_tactics) for
dense gate_up / dense down / DSA o_proj; cuBLAS (F.linear) for the rest, incl. latent down,
whose pin the low-latency projection wrapper hides from gemm_tactics.enable (arm
moe_latent_down_pin prices that pin). Both prefill loads: M = 16384 (F16K) and 8192 (F8K).

Per row and M: median us/call, total J/call from the NVML energy counter of THIS GPU over
the last second of each sustained round, dynamic J/call = total - P_idle x t, SM clock,
board power, calls per forward, and the launched kernel names (torch.profiler, one call).

    python3 pgemm_r8_ledger.py [--m 16384,8192] [--rows all|a,b] [--secs 1.5] [--rounds 2]
"""

import argparse
import json
import statistics
import threading
import time

import torch
import torch.nn.functional as F

# name: (kind, N, K, calls per forward); identical call counts at F16K and F8K
ROWS = {
    "dsa_qkv_a": ("bf16", 2272, 7168, 6),
    "dsa_qb_wq": ("bf16", 22528, 1536, 6),
    "dsa_g_proj": ("bf16", 12288, 7168, 6),
    "dsa_o_proj": ("bf16_pin", 7168, 12288, 6),
    "dsa_ukq": ("ukq", 512, 128, 6),
    "dsa_w_uv": ("bmm", 128, 512, 6),
    "moe_router": ("bf16_f32", 896, 7168, 23),
    "moe_latent_down": ("bf16", 3584, 7168, 23),
    "moe_latent_up": ("bf16_add", 7168, 3584, 23),
    "dense_gate_up": ("bf16_pin", 67584, 7168, 1),
    "dense_down": ("bf16_pin", 7168, 33792, 1),
    "kda_f_b": ("bf16", 12288, 128, 18),
    "kda_qkv_conv": ("kda_conv", 36864, 7168, 18),
    "kda_rest": ("mxfp8", 12512, 7168, 18),
    "kda_o": ("mxfp8", 7168, 12288, 18),
    "shared_gate_up_situ": ("b3", 12288, 7168, 23),
    "shared_down": ("mxfp8", 7168, 6144, 23),
    "glue_latent_norm": ("norm", 3584, 0, 23),
    "glue_router_topk": ("topk", 896, 0, 23),
    "glue_dsa_gate": ("gate", 12288, 0, 6),
    # reference arms (not deployed): the latent-down pin and the heuristic of the pinned shapes
    "moe_latent_down_pin": ("bf16_pin", 3584, 7168, 0),
    "dsa_o_proj_heur": ("bf16", 7168, 12288, 0),
    "dense_gate_up_heur": ("bf16", 67584, 7168, 0),
    "dense_down_heur": ("bf16", 7168, 33792, 0),
}


def _mx(m, k, dev, g, wscale=0.02, n=None):
    from flashinfer import mxfp8_quantize

    x = torch.randn(m, k, device=dev, dtype=torch.bfloat16, generator=g)
    xq, xs = mxfp8_quantize(x, is_sf_swizzled_layout=True)
    del x
    if n is None:
        return xq, xs
    w = torch.randn(n, k, device=dev, dtype=torch.bfloat16, generator=g) * wscale
    wq, ws = mxfp8_quantize(w, is_sf_swizzled_layout=True)
    del w
    return xq, xs, wq, ws


def make(name, m, dev):
    kind, n, k, _ = ROWS[name]
    g = torch.Generator(device=dev).manual_seed(n * 13 + k)
    if kind in ("bf16", "bf16_f32", "bf16_add", "bf16_pin"):
        x = torch.randn(m, k, device=dev, dtype=torch.bfloat16, generator=g)
        w = torch.randn(n, k, device=dev, dtype=torch.bfloat16, generator=g) * 0.02
        if kind == "bf16":
            return lambda: F.linear(x, w)
        if kind == "bf16_pin":
            from vllm.models.kimi_k3.nvidia.specialized_prefill import gemm_tactics

            plan = gemm_tactics.plan("bf16", n, k)
            if plan is None:
                raise RuntimeError("no pin for [%d, %d] in this process" % (n, k))
            return lambda: gemm_tactics.mm(plan, x, w)
        if kind == "bf16_f32":
            return lambda: torch.mm(x, w.t(), out_dtype=torch.float32)
        res = torch.randn(m, n, device=dev, dtype=torch.bfloat16, generator=g)
        return lambda: res.addmm_(x, w.t())
    if kind == "bmm":
        heads = 96
        x = torch.randn(heads, m, k, device=dev, dtype=torch.bfloat16, generator=g)
        w = torch.randn(heads, k, n, device=dev, dtype=torch.bfloat16, generator=g) * 0.05
        out = torch.empty(heads, m, n, device=dev, dtype=torch.bfloat16)
        return lambda: torch.bmm(x, w, out=out)
    if kind == "ukq":
        from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm import uk_query as U

        heads, p, r, lat = 96, 128, 64, 512
        q = (torch.randn(m, heads, p + r, device=dev, generator=g) * 2.0).to(torch.bfloat16)
        w = (torch.randn(heads, p, lat, device=dev, generator=g) * 0.5).to(torch.bfloat16)
        inv = torch.ones(1, device=dev, dtype=torch.float32)
        out = torch.empty((m, heads, lat + r), device=dev, dtype=torch.float8_e4m3fn)
        kern = U._KERNEL
        return lambda: kern(q, w, inv, out)
    if kind == "mxfp8":
        xq, xs, wq, ws = _mx(m, k, dev, g, n=n)
        sa = xs.view(torch.uint8).reshape(-1).view(torch.float8_e8m0fnu)
        sb = ws.view(torch.uint8).reshape(-1).view(torch.float8_e8m0fnu)
        return lambda: torch._scaled_mm(
            xq, wq.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16
        )
    if kind == "kda_conv":
        from vllm.models.kimi_k3.nvidia.specialized_prefill.kda_conv_in_proj import TACTIC
        from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm.kda_conv_gemm import (  # noqa: E501
            KdaConvGemm,
        )

        p = n // 3
        xq, xs, wq, ws = _mx(m, k, dev, g, wscale=0.05, n=n)
        cw = (torch.randn(n, 4, device=dev, generator=g) * 0.5).contiguous()
        lens = [m // 3, m // 3, m - 2 * (m // 3)]
        qsl = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device=dev)
        lines = torch.arange(1, 4, dtype=torch.int32, device=dev)
        has_init = torch.tensor([True, True, True], device=dev)
        st = torch.randn(5, n, 4, device=dev, generator=g).to(torch.bfloat16).float()[:, :, :3]
        gemm = KdaConvGemm(wq, ws, cw, p, tactic=TACTIC)
        return lambda: gemm(xq, xs, st, qsl, lines, has_init)
    if kind == "b3":
        from vllm.models.kimi_k3.nvidia.ops import mxfp8_act
        from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm import (
            die_sched,
        )
        from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm import (
            shared_situ_gemm as ssg,
        )

        inter = n // 2
        xq, xs, wq, ws = _mx(m, k, dev, g, n=n)
        sx = xs.view(torch.uint8).reshape(-1)
        sw = ws.view(torch.uint8).reshape(-1).clone()
        wq = wq.contiguous()
        ssg.interleave_gate_up_rows(wq, sw)
        tac = ssg.die_tactic(wq)
        if tac is None:
            raise RuntimeError("die_tactic: no die split on this device")
        wl = die_sched.localize_rows(wq)
        gemm = ssg.SharedSituGemm(wl, sw, 4.0, 25.0, tactic=tac)
        act = mxfp8_act.Mxfp8Buffer(m, inter, dev)
        return lambda: gemm(xq, sx, act)
    from vllm.models.kimi_k3.nvidia.ops import prefill_glue

    if kind == "norm":
        x = torch.randn(m, n, device=dev, dtype=torch.bfloat16, generator=g)
        wt = (1 + 0.1 * torch.randn(n, device=dev, generator=g)).to(torch.bfloat16)
        out = torch.empty_like(x)
        return lambda: prefill_glue.rms_norm(x, wt, 1e-6, out=out)
    if kind == "topk":
        logits = torch.randn(m, n, device=dev, dtype=torch.float32, generator=g) * 1.7
        bias = (0.01 * torch.randn(n, device=dev, generator=g)).float()
        return lambda: prefill_glue.router_topk(logits, bias, 16, True, 2.5)
    a = torch.randn(m, n, device=dev, dtype=torch.bfloat16, generator=g)
    gt = torch.randn(m, n, device=dev, dtype=torch.bfloat16, generator=g)
    return lambda: prefill_glue.gate_sigmoid_mul(a, gt)


def kernel_names(fn):
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    names = {}
    for e in prof.events():
        if e.device_type.name == "CUDA":
            names[e.name] = names.get(e.name, 0) + 1
    return [nm[:90] for nm in names][:3]


def main():
    import pynvml as N

    ap = argparse.ArgumentParser()
    ap.add_argument("--m", default="16384,8192")
    ap.add_argument("--rows", default="all")
    ap.add_argument("--secs", type=float, default=1.5)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    names = list(ROWS) if args.rows == "all" else args.rows.split(",")
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
    z = torch.zeros(1 << 20, device=dev)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 2.0:
        z.add_(1)
        torch.cuda.synchronize()
        time.sleep(0.01)
    t1 = time.perf_counter()
    ss = [s for s in samples if t1 - 1.0 <= s[0] <= t1]
    p_idle = (ss[-1][3] - ss[0][3]) / 1e3 / (ss[-1][0] - ss[0][0])
    head = {"power_limit_w": N.nvmlDeviceGetEnforcedPowerLimit(h) / 1e3, "p_idle_w": round(p_idle, 1)}
    print(json.dumps(head), flush=True)
    rows = []
    for m in [int(v) for v in args.m.split(",")]:
        for name in names:
            try:
                fn = make(name, m, dev)
                fn()
                torch.cuda.synchronize()
                kn = kernel_names(fn)
            except Exception as e:  # noqa: BLE001
                print(json.dumps({"row": name, "m": m, "error": repr(e)[:300]}), flush=True)
                torch.cuda.empty_cache()
                continue
            us, jt, clk, pw = [], [], [], []
            for _ in range(args.rounds):
                t0 = time.perf_counter()
                calls = 0
                while time.perf_counter() - t0 < args.secs:
                    for _ in range(4):
                        fn()
                    calls += 4
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                ss = [s for s in samples if t1 - 1.0 <= s[0] <= t1]
                if len(ss) < 2:
                    continue
                per_s = calls / (t1 - t0)
                us.append(1e6 / per_s)
                jt.append((ss[-1][3] - ss[0][3]) / 1e3 / ((ss[-1][0] - ss[0][0]) * per_s))
                clk.append(statistics.median(s[1] for s in ss))
                pw.append(statistics.median(s[2] for s in ss))
            u = statistics.median(us)
            j = statistics.median(jt)
            row = {
                "row": name, "m": m, "kind": ROWS[name][0], "n": ROWS[name][1], "k": ROWS[name][2],
                "calls": ROWS[name][3], "us": round(u, 1), "j_total": round(j, 4),
                "j_dyn": round(j - p_idle * u * 1e-6, 4), "clk_mhz": statistics.median(clk),
                "power_w": round(statistics.median(pw), 0), "kernels": kn,
            }  # fmt: skip
            rows.append(row)
            print(json.dumps(row), flush=True)
            del fn
            torch.cuda.empty_cache()
    stop[0] = True
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"head": head, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
