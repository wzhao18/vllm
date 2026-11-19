# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill MoE dense front at the DEP16 per-rank shape (M tokens, hidden 7168, latent 3584, 896 experts):
isolated cuBLAS kernel time of router (BF16 -> FP32), latent_down, latent_up (+ residual), the N-concatenated
[router | latent_down] GEMM (BF16 and FP32 out), and the same kernels after a sustained back-to-back loop
(clock/power state of a long prefill). Prints one JSON line per M."""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pgemm_bench import kernel_times  # noqa: E402

H, LAT, NE = 7168, 3584, 896
PEAK_BF16 = 4.0e15  # model SOL rate (pf_model_rows: 842 GF latent_down in 210.5 us)


def clk():
    try:
        return torch.cuda.clock_rate()
    except Exception:  # noqa: BLE001 (no NVML in the image)
        return None


def sol_us(m, n, k):
    return round(2 * m * n * k / PEAK_BF16 * 1e6, 1)


def main():
    sustain = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    for m in (16384, 8192):
        g = torch.Generator(device="cuda").manual_seed(0)
        x = torch.randn(m, H, device="cuda", generator=g).bfloat16()
        lat = torch.randn(m, LAT, device="cuda", generator=g).bfloat16()
        res = torch.randn(m, H, device="cuda", generator=g).bfloat16()
        wr = (torch.randn(NE, H, device="cuda", generator=g) * 0.02).bfloat16()
        wd = (torch.randn(LAT, H, device="cuda", generator=g) * 0.02).bfloat16()
        wu = (torch.randn(H, LAT, device="cuda", generator=g) * 0.02).bfloat16()
        wc = torch.cat([wr, wd], 0).contiguous()
        ops = {
            "router": lambda: torch.mm(x, wr.T, out_dtype=torch.float32),
            "latent_down": lambda: torch.mm(x, wd.T),
            "latent_up": lambda: res.addmm_(lat, wu.T),
            "cat_bf16": lambda: torch.mm(x, wc.T),
            "cat_fp32": lambda: torch.mm(x, wc.T, out_dtype=torch.float32),
        }
        r = {"M": m, "sol_us": {"router": sol_us(m, NE, H), "latent_down": sol_us(m, LAT, H),
                                 "latent_up": sol_us(m, H, LAT), "cat": sol_us(m, NE + LAT, H)}}
        r["iso"] = {k: round(kernel_times(f)[0], 1) for k, f in ops.items()}
        r["clk_idle_mhz"] = clk()

        def layer():
            for k in ("router", "latent_down", "latent_up"):
                ops[k]()

        for _ in range(sustain):
            layer()
        r["clk_sustained_mhz"] = clk()
        r["sustained"] = {}
        for k in ("router", "latent_down", "latent_up", "cat_bf16"):
            for _ in range(sustain // 3):
                layer()
            r["sustained"][k] = round(kernel_times(ops[k], warm=0)[0], 1)
        print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
