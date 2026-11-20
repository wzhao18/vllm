#!/usr/bin/env python3
"""P-GEMM R7: per-row energy ledger of the prefill GEMM family at the power cap (1 GPU).

Every GEMM shape of the family (F16K per-GPU rows, M = 16384) and its glue kernels run
back to back for --secs per round on realistic data (randn activations, randn x 0.02
weights; MXFP8 rows quantized with real E8M0 block scales). Per row: median us/call,
total J/call (NVML total-energy counter of THIS GPU over the last second of each round),
dynamic J/call = total - P_idle x t (P_idle: awake-idle board power measured first),
SM clock, board power, calls per forward. The compulsory J is computed off-line.

    python3 pgemm_r7_ledger.py [--rows all|name,...] [--secs 1.5] [--rounds 2] [--out f]
"""

import argparse
import json
import statistics
import threading
import time

import torch

M = 16384
# name: (kind, N, K, calls per F16K forward)
ROWS = {
    "dsa_qkv_a": ("bf16", 2272, 7168, 6),
    "dsa_qb_wq": ("bf16", 22528, 1536, 6),
    "dsa_g_proj": ("bf16", 12288, 7168, 6),
    "dsa_o_proj": ("bf16", 7168, 12288, 6),
    "dsa_w_uk": ("bmm", 512, 128, 6),
    "dsa_w_uv": ("bmm", 128, 512, 6),
    "moe_router": ("bf16_f32", 896, 7168, 23),
    "moe_latent_down": ("bf16", 3584, 7168, 23),
    "moe_latent_up": ("bf16_add", 7168, 3584, 23),
    "dense_gate_up": ("bf16", 67584, 7168, 1),
    "dense_down": ("bf16", 7168, 33792, 1),
    "kda_f_b": ("bf16", 12288, 128, 18),
    "kda_in": ("mxfp8", 49376, 7168, 18),
    "kda_o": ("mxfp8", 7168, 12288, 18),
    "shared_gate_up": ("mxfp8", 12288, 7168, 23),
    "shared_down": ("mxfp8", 7168, 6144, 23),
    "glue_latent_norm": ("norm", 3584, 0, 23),
    "glue_router_topk": ("topk", 896, 0, 23),
    "glue_dsa_gate": ("gate", 12288, 0, 6),
}


def make(name, dev):
    kind, n, k, _ = ROWS[name]
    g = torch.Generator(device=dev).manual_seed(n * 13 + k)
    if kind in ("bf16", "bf16_f32", "bf16_add"):
        x = torch.randn(M, k, device=dev, dtype=torch.bfloat16, generator=g)
        w = torch.randn(n, k, device=dev, dtype=torch.bfloat16, generator=g) * 0.02
        if kind == "bf16":
            out = torch.empty(M, n, device=dev, dtype=torch.bfloat16)
            return lambda: torch.mm(x, w.t(), out=out)
        if kind == "bf16_f32":
            return lambda: torch.mm(x, w.t(), out_dtype=torch.float32)
        res = torch.randn(M, n, device=dev, dtype=torch.bfloat16, generator=g)
        return lambda: res.addmm_(x, w.t())
    if kind == "bmm":
        heads = 96
        x = torch.randn(heads, M, k, device=dev, dtype=torch.bfloat16, generator=g)
        w = torch.randn(heads, k, n, device=dev, dtype=torch.bfloat16, generator=g) * 0.05
        out = torch.empty(heads, M, n, device=dev, dtype=torch.bfloat16)
        return lambda: torch.bmm(x, w, out=out)
    if kind == "mxfp8":
        from flashinfer import mxfp8_quantize

        x = torch.randn(M, k, device=dev, dtype=torch.bfloat16, generator=g)
        w = torch.randn(n, k, device=dev, dtype=torch.bfloat16, generator=g) * 0.02
        xq, xs = mxfp8_quantize(x, is_sf_swizzled_layout=True)
        wq, ws = mxfp8_quantize(w, is_sf_swizzled_layout=True)
        sa = xs.view(torch.uint8).reshape(-1).view(torch.float8_e8m0fnu)
        sb = ws.view(torch.uint8).reshape(-1).view(torch.float8_e8m0fnu)
        del x, w
        return lambda: torch._scaled_mm(
            xq, wq.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16
        )
    from vllm.models.kimi_k3.nvidia.ops import prefill_glue

    if kind == "norm":
        x = torch.randn(M, n, device=dev, dtype=torch.bfloat16, generator=g)
        wt = (1 + 0.1 * torch.randn(n, device=dev, generator=g)).to(torch.bfloat16)
        out = torch.empty_like(x)
        return lambda: prefill_glue.rms_norm(x, wt, 1e-6, out=out)
    if kind == "topk":
        logits = torch.randn(M, n, device=dev, dtype=torch.float32, generator=g) * 1.7
        bias = (0.01 * torch.randn(n, device=dev, generator=g)).float()
        return lambda: prefill_glue.router_topk(logits, bias, 16, True, 2.5)
    a = torch.randn(M, n, device=dev, dtype=torch.bfloat16, generator=g)
    gt = torch.randn(M, n, device=dev, dtype=torch.bfloat16, generator=g)
    return lambda: prefill_glue.gate_sigmoid_mul(a, gt)


def main():
    import pynvml as N

    ap = argparse.ArgumentParser()
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
    # awake idle: a short kernel every 10 ms for 2 s (clocks stay up, SMs idle)
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
    for name in names:
        fn = make(name, dev)
        fn()
        torch.cuda.synchronize()
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
            "row": name, "kind": ROWS[name][0], "n": ROWS[name][1], "k": ROWS[name][2],
            "calls": ROWS[name][3], "us": round(u, 1), "j_total": round(j, 4),
            "j_dyn": round(j - p_idle * u * 1e-6, 4), "clk_mhz": statistics.median(clk),
            "power_w": round(statistics.median(pw), 0),
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
