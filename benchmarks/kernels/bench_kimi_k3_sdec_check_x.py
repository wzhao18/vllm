# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 sdec kda_in "pre" GEMM with the ``check_x`` activation check: us per call
of the unchecked kernel, the checked kernel on a complete activation (scan only) and
on an all-sentinel TMA source (every word patched from global), CUDA graph of
back-to-back calls on cold weight copies, one GPU.
usage: python bench_kimi_k3_sdec_check_x.py [--rows 128,256,384] [--calls 16]"""

import argparse

import torch

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    mxfp8_e4m3_quantize,
    swizzle_mxfp8_scale,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.mxfp8 import pre_tactic
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.sm107.prewait import (
    Mxfp8PreGemm,
    check_args,
)

N, K = 12448, 7168


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", default="128,256,384")
    p.add_argument("--calls", type=int, default=16)
    p.add_argument("--replays", type=int, default=20)
    p.add_argument("--form", default="", help="DIAG check_x barrier form: cta|cluster,wait_op")
    args = p.parse_args()
    if args.form:
        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.sm107 import check_x

        scope, op = args.form.split(",")
        check_x.CTA_SCOPE, check_x.WAIT_OP = scope == "cta", op
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    ws = []
    for i in range(args.calls):
        w = (torch.randn(N, K, device=dev) * 0.02).bfloat16()
        wq, sw = mxfp8_e4m3_quantize(w, is_sf_swizzled_layout=False)
        sw = sw.view(torch.uint8).reshape(N, K // 32)
        ws.append((wq.contiguous(), swizzle_mxfp8_scale(sw, M=N, K=K).contiguous()))
        del w
    for r in [int(x) for x in args.rows.split(",")]:
        tactic, prewait, cap = pre_tactic(N, K, r)
        x = torch.randn(r, K, device=dev).bfloat16()
        xq, xs = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
        words = xq.view(torch.int32)
        words[words == -0x80000000] = 0
        sq = torch.empty_like(xq)
        sq.view(torch.int32).fill_(-0x80000000)
        ss = torch.full_like(xs.view(torch.uint8), 0xFF)
        out = torch.empty(r, N, device=dev, dtype=torch.bfloat16)
        plain = Mxfp8PreGemm(N, K, r, tactic, prewait=prewait, max_stages=cap,
                             device=dev)  # fmt: skip
        checked = Mxfp8PreGemm(N, K, r, tactic, prewait=prewait, max_stages=cap,
                               check=True, device=dev)  # fmt: skip
        full = check_args(xq.data_ptr(), xs.data_ptr(), r, K, dev)
        arms = {
            "plain": lambda w: plain(xq, xs, w[0], w[1], out),
            "checked_none": lambda w: checked(xq, xs, w[0], w[1], out),
            "checked_scan": lambda w: checked(xq, xs, w[0], w[1], out, check=full),
            "checked_patch": lambda w: checked(sq, ss, w[0], w[1], out, check=full),
        }
        res = {}
        for name, fn in arms.items():
            for w in ws:
                fn(w)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for w in ws:
                    fn(w)
            times = []
            for _ in range(args.replays):
                a = torch.cuda.Event(enable_timing=True)
                b = torch.cuda.Event(enable_timing=True)
                a.record()
                g.replay()
                b.record()
                b.synchronize()
                times.append(a.elapsed_time(b) * 1e3 / len(ws))
            times.sort()
            res[name] = times[len(times) // 2]
        print(f"{args.form} R{r} {tactic}: " + "  ".join(f"{k} {v:.2f}" for k, v in res.items()),
              flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
