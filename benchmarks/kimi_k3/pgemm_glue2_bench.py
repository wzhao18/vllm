# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P-GEMM glue kernels at the DEP16 prefill shape (M = 16384 / 8192): router
top-k (vs stock grouped_topk, bitwise check), DSA gate sigmoid-mul and the latent
RMSNorm, CUPTI kernel time vs the HBM floor."""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pgemm_bench import ACH_HBM, kernel_times  # noqa: E402


def main():
    from vllm import _custom_ops as ops
    from vllm.models.kimi_k3.nvidia.ops import prefill_glue

    prefill_glue.warmup()
    out = []
    for m in (16384, 8192, 4097):
        x = torch.randn(m, 896, device="cuda") * 2
        bias = torch.randn(896, device="cuda") * 0.05
        v0, i0 = ops.grouped_topk(x, 1, 1, 16, True, 2.5, bias, 1)
        v1, i1 = prefill_glue.router_topk(x, bias, 16, True, 2.5)
        ok = bool(torch.equal(i0, i1) and torch.equal(v0.view(torch.int32), v1.view(torch.int32)))
        r = {"case": "router_topk", "M": m, "bitwise": ok,
             "floor_us": round(m * 896 * 4 / ACH_HBM * 1e6, 1),
             "glue_us": kernel_times(lambda: prefill_glue.router_topk(x, bias, 16, True, 2.5))[0],
             "stock_us": kernel_times(lambda: ops.grouped_topk(x, 1, 1, 16, True, 2.5, bias, 1))[0]}
        out.append(r)
        print(json.dumps(r), flush=True)

        d = 12288
        a = torch.randn(m, d, device="cuda").bfloat16()
        g = torch.randn(m, d, device="cuda").bfloat16() * 3
        ref = a * g.sigmoid()
        got = prefill_glue.gate_sigmoid_mul(a, g)
        r = {"case": "gate_sigmoid_mul", "M": m, "d": d,
             "bitwise": bool(torch.equal(ref.view(torch.int16), got.view(torch.int16))),
             "floor_us": round(3 * m * d * 2 / ACH_HBM * 1e6, 1),
             "glue_us": kernel_times(lambda: prefill_glue.gate_sigmoid_mul(a, g))[0]}
        out.append(r)
        print(json.dumps(r), flush=True)

        h = 3584
        xs = torch.randn(m, h, device="cuda").bfloat16()
        w = torch.randn(h, device="cuda").bfloat16()
        ref = torch.empty_like(xs)
        ops.rms_norm(ref, xs, w, 1e-6)
        got = prefill_glue.rms_norm(xs, w, 1e-6)
        r = {"case": "latent_rms_norm", "M": m, "h": h,
             "bitwise": bool(torch.equal(ref.view(torch.int16), got.view(torch.int16))),
             "floor_us": round(2 * m * h * 2 / ACH_HBM * 1e6, 1),
             "glue_us": kernel_times(lambda: prefill_glue.rms_norm(xs, w, 1e-6))[0],
             "stock_us": kernel_times(lambda: ops.rms_norm(ref, xs, w, 1e-6))[0]}
        out.append(r)
        print(json.dumps(r), flush=True)
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
