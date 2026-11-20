# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbench: experimental 4-lanes-per-row gated o_norm + MXFP8 variants
(onorm4.cu) vs the production o_norm_mx kernel, bitwise and interleaved time."""

import argparse
import json
import os

import torch
from torch.utils.cpp_extension import load

from vllm.models.kimi_k3.nvidia.ops.mxfp8_act import Mxfp8Buffer
from vllm.models.kimi_k3.nvidia.specialized_prefill.kda import o_norm_mxfp8


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, nargs="+", default=[16383, 8192])
    ap.add_argument("--heads", type=int, default=96)
    ap.add_argument("--row-stride", type=int, default=49376)
    ap.add_argument("--variants", default="0,1,2,3,4")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--rounds", type=int, default=3)
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    ext = load(
        name="onorm4_exp",
        sources=[os.path.join(here, "onorm4.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo",
                           "-gencode", "arch=compute_100f,code=sm_100f"],
        verbose=False,
    )
    H, D = a.heads, 128
    P = H * D
    vs = [int(v) for v in a.variants.split(",")]
    for T in a.tokens:
        torch.manual_seed(0)
        x = (torch.randn(1, T, H, D, device="cuda") * 3).to(torch.bfloat16)
        proj = (torch.randn(T, a.row_stride, device="cuda") * 2).to(torch.bfloat16)
        g = proj[:, 3 * P : 4 * P]
        w = torch.rand(D, device="cuda") + 0.5
        bufs = {k: Mxfp8Buffer(T, P, "cuda", torch.bfloat16) for k in ["ref"] + vs}
        for b in bufs.values():
            b.data.view(torch.uint8).zero_()
            b.scale.zero_()

        def run(k):
            if k == "ref":
                o_norm_mxfp8(x, g, w, 1e-6, bufs["ref"])
            else:
                ext.run(x, g, w, 1e-6, bufs[k].data.view(torch.uint8), bufs[k].scale, k)

        for k in bufs:
            run(k)
        torch.cuda.synchronize()
        res = {"T": T}
        rd = bufs["ref"].data.view(torch.uint8)
        for k in vs:
            res[f"v{k}_bitwise"] = bool(
                torch.equal(bufs[k].data.view(torch.uint8), rd)
                and torch.equal(bufs[k].scale, bufs["ref"].scale)
            )
        print(json.dumps(res), flush=True)
        keys = ["ref"] + vs
        tm = {k: [] for k in keys}
        for r in range(a.rounds):
            for k in keys if r % 2 == 0 else keys[::-1]:
                for _ in range(3):
                    run(k)
                ts = []
                for _ in range(a.iters):
                    e0, e1 = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                    e0.record()
                    run(k)
                    e1.record()
                    e1.synchronize()
                    ts.append(e0.elapsed_time(e1) * 1e3)
                ts.sort()
                tm[k].append(ts[len(ts) // 2])
        gb = T * P * 5 / 1e9
        med = {str(k): round(sorted(v)[len(v) // 2], 1) for k, v in tm.items()}
        print(json.dumps({"T": T, "med_us": med,
                          "tbps": {k: round(gb / v * 1e3, 2) for k, v in med.items()},
                          "rounds": {str(k): [round(x, 1) for x in v] for k, v in tm.items()}}),
              flush=True)


if __name__ == "__main__":
    main()
