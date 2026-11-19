# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MoE router path at the DEP16 prefill shape: router GEMM (BF16 -> FP32),
prefill_glue.router_topk, latent RMSNorm, and router || latent_down on two
streams vs serial (CUDA-event wall time of the pair)."""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pgemm_bench import kernel_times  # noqa: E402


def wall_us(fn, iters=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1e3 / iters


def main():
    from vllm.models.kimi_k3.nvidia.ops import prefill_glue

    for m in (16384, 8192):
        x = torch.randn(m, 7168, device="cuda").bfloat16()
        wr = (torch.randn(896, 7168, device="cuda") * 0.02).bfloat16()
        wd = (torch.randn(3584, 7168, device="cuda") * 0.02).bfloat16()
        bias = torch.randn(896, device="cuda") * 0.01
        r = {"M": m}
        r["router_gemm"] = kernel_times(lambda: torch.mm(x, wr.T, out_dtype=torch.float32))[0]
        logits = torch.mm(x, wr.T, out_dtype=torch.float32)
        r["router_topk"] = kernel_times(
            lambda: prefill_glue.router_topk(logits, bias, 16, True, 2.5))[0]
        r["router_topk_floor_us"] = round(m * 896 * 4 / 13.9e12 * 1e6, 1)
        r["latent_down"] = kernel_times(lambda: torch.mm(x, wd.T))[0]
        side = torch.cuda.Stream()

        def serial():
            torch.mm(x, wr.T, out_dtype=torch.float32)
            torch.mm(x, wd.T)

        def overlap():
            main_s = torch.cuda.current_stream()
            side.wait_stream(main_s)
            torch.mm(x, wr.T, out_dtype=torch.float32)
            with torch.cuda.stream(side):
                torch.mm(x, wd.T)
            main_s.wait_stream(side)

        r["pair_serial_wall"] = wall_us(serial)
        r["pair_overlap_wall"] = wall_us(overlap)
        print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
