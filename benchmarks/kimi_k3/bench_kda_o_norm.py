# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbench of the Kimi K3 prefill gated o_norm + MXFP8 (o_proj input) at the
DEP16 prefill shapes: one JSON line per token count with the kernel time, the
effective HBM rate and a checksum of the E4M3 output and the UE8M0 scales for
cross-process bitwise comparison of kernel variants."""

import argparse
import json

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, nargs="+", default=[16383, 8192])
    ap.add_argument("--heads", type=int, default=96)
    ap.add_argument("--row-stride", type=int, default=49376)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    from vllm.models.kimi_k3.nvidia.ops.mxfp8_act import Mxfp8Buffer
    from vllm.models.kimi_k3.nvidia.specialized_prefill.kda import o_norm_mxfp8

    H, D = args.heads, 128
    P = H * D
    for T in args.tokens:
        torch.manual_seed(0)
        x = (torch.randn(1, T, H, D, device="cuda") * 3).to(torch.bfloat16)
        proj = (torch.randn(T, args.row_stride, device="cuda") * 2).to(torch.bfloat16)
        g = proj[:, 3 * P : 4 * P]
        w = torch.rand(D, device="cuda") + 0.5
        out = Mxfp8Buffer(T, P, "cuda", torch.bfloat16)
        out.data.view(torch.uint8).zero_()
        out.scale.view(torch.uint8).zero_()
        o_norm_mxfp8(x, g, w, 1e-6, out)
        torch.cuda.synchronize()
        ck = []
        for t in (out.data, out.scale):
            u = t.contiguous().view(torch.uint8).flatten().to(torch.int64)
            wgt = torch.arange(u.numel(), device="cuda", dtype=torch.int64) % 1009 + 1
            ck.append(int((u * wgt).sum()))
        for _ in range(3):
            o_norm_mxfp8(x, g, w, 1e-6, out)
        torch.cuda.synchronize()
        times = []
        for _ in range(args.iters):
            a = torch.cuda.Event(enable_timing=True)
            b = torch.cuda.Event(enable_timing=True)
            a.record()
            o_norm_mxfp8(x, g, w, 1e-6, out)
            b.record()
            b.synchronize()
            times.append(a.elapsed_time(b) * 1e3)
        times.sort()
        med = times[len(times) // 2]
        gb = T * P * (2 + 2 + 1) / 1e9
        print(json.dumps({"tag": args.tag, "tokens": T, "us_med": round(med, 1),
                          "us_min": round(times[0], 1), "tbps": round(gb / med * 1e3, 2),
                          "checksum": ck}), flush=True)


if __name__ == "__main__":
    main()
