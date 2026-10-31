# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP8 activation quantization for the K3.1 projections on VR: standalone
FlashInfer mxfp8_quantize (what every online-MXFP8 linear launches today) vs
the fused producers of vllm/models/kimi_k3/nvidia/ops/mxfp8_act.py and
kda_norm.rmsnorm_gated_mxfp8, at the per-rank rows of DEP / DCP4 / DCP8.

Each case times "BF16 producer + FlashInfer quant" (unfused, two launches) vs
the fused producer (one launch), in a CUDA graph of 64 calls (us per call).
Usage: python bench_act_quant.py [--rows 16,64,...] [--out f.json]
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import time_graph  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="16,64,96,128,256,384,512,768")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
    )
    from vllm.models.kimi_k3.nvidia.mla import _gate_sigmoid_mul
    from vllm.models.kimi_k3.nvidia.ops import kda_norm, mxfp8_act

    from vllm.config import VllmConfig, set_current_vllm_config

    set_current_vllm_config(VllmConfig()).__enter__()  # RMSNorm is a CustomOp
    mxfp8_act.warmup()
    dev = "cuda"
    fi = lambda x: mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)  # noqa: E731
    rows_list = [int(r) for r in args.rows.split(",")]
    res = []
    NC = 4  # rotating inputs

    def emit(row):
        print(json.dumps(row), flush=True)
        res.append(row)

    for m in rows_list:
        # plain quant of the hidden state / o_proj input widths
        for k in (7168, 12288, 3072, 1536):
            xs = [torch.randn(m, k, device=dev, dtype=torch.bfloat16) for _ in range(NC)]
            emit(dict(case="quant", m=m, k=k,
                      fi=round(time_graph([lambda x=x: fi(x) for x in xs]), 2),
                      fused=round(time_graph(
                          [lambda x=x: mxfp8_act.quantize(x) for x in xs]), 2)))
        # shared-expert activation (d = 6144 replicated / 1536 TP4 / 768 TP8)
        for d in (6144, 1536, 768):
            xs = [torch.randn(m, 2 * d, device=dev, dtype=torch.bfloat16) for _ in range(NC)]

            def unf(x):
                o = torch.empty(m, d, device=dev, dtype=torch.bfloat16)
                torch.ops._C.situ_and_mul(o, x, 1.0, -1.0)
                return fi(o)

            emit(dict(case="situ", m=m, k=d,
                      fi=round(time_graph([lambda x=x: unf(x) for x in xs]), 2),
                      fused=round(time_graph([
                          lambda x=x: mxfp8_act.situ_and_mul_quantize(x, 1.0, None)
                          for x in xs]), 2)))
        # latent norm (3584) -> latent up
        from vllm.model_executor.layers.layernorm import RMSNorm

        norm = RMSNorm(3584, eps=1e-6).to(dev).to(torch.bfloat16)
        xs = [torch.randn(m, 3584, device=dev, dtype=torch.bfloat16) for _ in range(NC)]
        emit(dict(case="rmsnorm", m=m, k=3584,
                  fi=round(time_graph([lambda x=x: fi(norm(x)) for x in xs]), 2),
                  fused=round(time_graph([
                      lambda x=x: mxfp8_act.rms_norm_quantize(x, norm.weight, 1e-6)
                      for x in xs]), 2)))
        # DSA output gate: o_proj input, 96 / 24 / 12 heads x 128
        for k in (12288, 3072, 1536):
            ab = [(torch.randn(m, k, device=dev, dtype=torch.bfloat16),
                   torch.randn(m, k, device=dev, dtype=torch.bfloat16)) for _ in range(NC)]
            emit(dict(case="gate", m=m, k=k,
                      fi=round(time_graph(
                          [lambda a=a, b=b: fi(_gate_sigmoid_mul(a, b)) for a, b in ab]), 2),
                      fused=round(time_graph([
                          lambda a=a, b=b: mxfp8_act.gate_sigmoid_mul_quantize(a, b)
                          for a, b in ab]), 2)))
        # KDA output gate norm: 96 / 24 / 12 heads
        for h in (96, 24, 12):
            w = torch.ones(128, device=dev, dtype=torch.bfloat16)
            cases = []
            for _ in range(NC):
                x = torch.randn(m * h, 128, device=dev, dtype=torch.bfloat16)
                proj = torch.randn(m, 4 * h * 128 + 256, device=dev, dtype=torch.bfloat16)
                g = proj[:, 3 * h * 128 : 4 * h * 128].view(m, h, 128)
                cases.append((x, g))

            def unf_kda(x, g):
                kda_norm.rmsnorm_gated_(x, g, w, 1e-5)
                return fi(x.view(m, h * 128))

            emit(dict(case="kda_norm", m=m, k=h * 128,
                      fi=round(time_graph([lambda x=x, g=g: unf_kda(x, g) for x, g in cases]), 2),
                      fused=round(time_graph([
                          lambda x=x, g=g: kda_norm.rmsnorm_gated_mxfp8(x, g, w, 1e-5)
                          for x, g in cases]), 2)))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
