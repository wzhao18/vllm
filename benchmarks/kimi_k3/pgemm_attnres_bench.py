# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AttnRes (+ MXFP8 quant of its output) at the DEP16 prefill shape:
native kimi_k3_attn_res + standalone quant vs the fused specialized_prefill
kernel (BF16+MXFP8 / MXFP8 only), per ring depth. CUPTI kernel time."""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pgemm_bench import kernel_times  # noqa: E402

HIDDEN = 7168
EPS = 1e-6


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", default="16384,8192")
    p.add_argument("--depths", default="2,3")
    args = p.parse_args()
    from flashinfer import mxfp8_quantize

    from vllm import _custom_ops as ops
    from vllm.models.kimi_k3.nvidia.ops import mxfp8_act
    from vllm.models.kimi_k3.nvidia.specialized_prefill.attn_res_mxfp8 import (
        attn_res_mxfp8,
    )

    for m in [int(r) for r in args.rows.split(",")]:
        for nb, has_delta in ((1, True), (4, True), (8, True), (4, False)):
            prefix = torch.randn(m, HIDDEN, device="cuda").bfloat16()
            delta = torch.randn(m, HIDDEN, device="cuda").bfloat16() if has_delta else None
            blocks = torch.randn(m, 9, HIDDEN, device="cuda").bfloat16()
            w = torch.ones(HIDDEN, device="cuda").bfloat16()
            qk = (torch.randn(HIDDEN, device="cuda") * HIDDEN**-0.5).bfloat16()
            # model bytes: (nb sources + prefix + delta) read + out written
            nbytes = m * HIDDEN * 2 * (nb + 1 + (1 if has_delta else 0) + 1)
            row = {"M": m, "num_blocks": nb, "delta": has_delta,
                   "min_bytes_us_at_13.9TBs": round(nbytes / 13.9e12 * 1e6, 1)}

            def native():
                return ops.kimi_k3_attn_res(
                    prefix, delta, blocks, w, qk, w, nb, -1, EPS, EPS)

            row["native"] = kernel_times(native)[0]
            ref = native()
            row["quant.fi"] = kernel_times(
                lambda: mxfp8_quantize(ref, is_sf_swizzled_layout=True))[0]
            row["quant.vllm_act"] = kernel_times(lambda: mxfp8_act.quantize(ref))[0]
            for d in [int(x) for x in args.depths.split(",")]:
                for bf in (True, False):
                    key = f"fused.d{d}.{'bf16+q' if bf else 'q'}"
                    try:
                        row[key] = kernel_times(lambda: attn_res_mxfp8(
                            prefix, delta, blocks, w, qk, w, nb, -1, EPS, EPS,
                            bf16_out=bf, chunk_depth=d))[0]
                    except Exception as e:  # noqa: BLE001
                        row[key] = repr(e)[:120]
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
