# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P-GEMM glue kernels at the DEP16 prefill shape (M = 16384 / 8192), CUPTI
kernel time vs the HBM floor: shared SiTU + MXFP8 quant (table form), router
top-k (one warp per token) vs stock grouped_topk, latent RMSNorm."""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pgemm_bench import ACH_HBM, kernel_times  # noqa: E402


def main():
    from vllm import _custom_ops as ops
    from vllm.models.kimi_k3.nvidia.ops import mxfp8_act, prefill_glue

    for m in (16384, 8192):
        d = 6144
        x = torch.randn(m, 2 * d, device="cuda").bfloat16() * 2
        floor = (m * 2 * d * 2 + m * d + m * d // 32) / ACH_HBM * 1e6
        r = {"case": "situ_quant_shared", "M": m, "floor_us": round(floor, 1)}
        r["situ_lut"] = kernel_times(lambda: mxfp8_act.situ_and_mul_quantize(x, 1.0, None))[0]
        r["situ_lut_clamp"] = kernel_times(
            lambda: mxfp8_act.situ_and_mul_quantize(x, 1.0, 10.0))[0]
        ref = torch.empty(m, d, device="cuda", dtype=torch.bfloat16)
        r["stock_situ_bf16_only"] = kernel_times(
            lambda: torch.ops._C.situ_and_mul(ref, x, 1.0, -1.0))[0]
        print(json.dumps(r), flush=True)

        logits = torch.randn(m, 896, device="cuda") * 2
        for kind in ("rand", "ep_bias"):
            bias = torch.randn(896, device="cuda") * 0.01
            if kind == "ep_bias":
                bias[torch.arange(896, device="cuda") % 224 >= 56] = -100.0
            r = {"case": "router_topk_" + kind, "M": m,
                 "floor_us": round(m * 896 * 4 / ACH_HBM * 1e6, 1)}
            r["k31"] = kernel_times(lambda: prefill_glue.router_topk(logits, bias, 16, True, 2.5))[0]
            r["stock"] = kernel_times(lambda: ops.grouped_topk(logits, 1, 1, 16, True, 2.5, bias, 1))[0]
            print(json.dumps(r), flush=True)

        lat = torch.randn(m, 3584, device="cuda").bfloat16()
        w = torch.randn(3584, device="cuda").bfloat16()
        r = {"case": "latent_rms_norm", "M": m, "floor_us": round(m * 3584 * 4 / ACH_HBM * 1e6, 1)}
        r["rms_norm_rows"] = kernel_times(lambda: prefill_glue.rms_norm(lat, w, 1e-6))[0]
        print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
