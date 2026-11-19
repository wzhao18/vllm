# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP8 activation producers (ops/mxfp8_act.py) at DEP16 prefill shapes vs
FlashInfer mxfp8_quantize and the HBM floor. CUPTI kernel time."""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pgemm_bench import ACH_HBM, kernel_times  # noqa: E402


def main():
    from flashinfer import mxfp8_quantize

    from vllm.models.kimi_k3.nvidia.ops import mxfp8_act

    for m in (16384, 8192):
        for name, k in (("quant7168", 7168), ("quant12288", 12288)):
            x = torch.randn(m, k, device="cuda").bfloat16()
            floor = (m * k * 2 + m * k + m * k // 32) / ACH_HBM * 1e6
            r = {"case": name, "M": m, "floor_us": round(floor, 1)}
            r["fi"] = kernel_times(lambda: mxfp8_quantize(x, is_sf_swizzled_layout=True))[0]
            r["vllm_act"] = kernel_times(lambda: mxfp8_act.quantize(x))[0]
            print(json.dumps(r), flush=True)
        d = 6144
        x = torch.randn(m, 2 * d, device="cuda").bfloat16()
        floor = (m * 2 * d * 2 + m * d + m * d // 32) / ACH_HBM * 1e6
        r = {"case": "situ_quant_shared", "M": m, "floor_us": round(floor, 1)}
        r["situ_and_mul_quantize"] = kernel_times(
            lambda: mxfp8_act.situ_and_mul_quantize(x, 1.0, None))[0]
        r["silu_and_mul_quantize"] = kernel_times(
            lambda: mxfp8_act.silu_and_mul_quantize(x))[0]
        print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
