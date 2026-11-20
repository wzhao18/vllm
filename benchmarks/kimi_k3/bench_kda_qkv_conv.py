# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbench of the Kimi K3 prefill Q|K|V conv (ops/kda_qkv_conv) at the
DEP16 prefill shape: one JSON line with the kernel time, the effective HBM
rate (x read + out written) and an output/state checksum for cross-process
bitwise comparison of kernel variants.

--sub OLD=>NEW rewrites a temp copy of the kernel source before the JIT build
(exploration only; the shipped kernel is never modified)."""

import argparse
import json
import os
import tempfile

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=16383)
    ap.add_argument("--seqs", type=int, default=3)
    ap.add_argument("--heads", type=int, default=96)
    ap.add_argument("--row-stride", type=int, default=49376)
    ap.add_argument("--state", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--sub", action="append", default=[])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    from vllm.models.kimi_k3.nvidia.ops import kda_conv

    if args.sub:
        src = open(kda_conv._QKV_SRC).read()
        for s in args.sub:
            old, new = s.split("=>")
            assert old in src, f"--sub pattern not found: {old!r}"
            src = src.replace(old, new)
        d = tempfile.mkdtemp(prefix="qkvconv_")
        kda_conv._QKV_SRC = os.path.join(d, "kda_qkv_conv.cu")
        with open(kda_conv._QKV_SRC, "w") as f:
            f.write(src)

    dev = torch.device("cuda")
    torch.manual_seed(0)
    T, N, P = args.tokens, args.seqs, args.heads * 128
    lens = [T // N] * N
    lens[-1] += T - sum(lens)
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32)
    cu = cu.to(dev)
    bf = torch.bfloat16
    proj = torch.randn(T, args.row_stride, dtype=bf, device=dev) * 0.5
    x = proj[:, : 3 * P]
    w = torch.randn(3 * P, 4, dtype=torch.float32, device=dev) * 0.3
    sdt = bf if args.state == "bf16" else torch.float32
    state0 = torch.randn(N + 2, 3 * P, 3, device=dev).to(sdt)
    sidx = torch.arange(1, N + 1, dtype=torch.int32, device=dev)
    has_init = torch.ones(N, dtype=torch.bool, device=dev)

    state = state0.clone()
    out = kda_conv.qkv_conv(x, w, state, cu, sidx, has_init, P)
    o16 = out.view(torch.int16).to(torch.int64)
    wgt = torch.arange(o16.shape[-1], device=dev, dtype=torch.int64) % 1009 + 1
    ck_out = int((o16 * wgt).sum())
    ck_st = int((state.view(torch.int16 if sdt == bf else torch.int32)
                 .to(torch.int64)).sum())

    for _ in range(3):
        kda_conv.qkv_conv(x, w, state, cu, sidx, has_init, P)
    torch.cuda.synchronize()
    times = []
    for _ in range(args.iters):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        kda_conv.qkv_conv(x, w, state, cu, sidx, has_init, P)
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) * 1e3)
    times.sort()
    med = times[len(times) // 2]
    gb = 2 * T * 3 * P * 2 / 1e9
    print(json.dumps({"tag": args.tag, "state": args.state, "tokens": T,
                      "us_med": round(med, 1), "us_min": round(times[0], 1),
                      "TBps": round(gb / med * 1e3, 2),
                      "ck_out": ck_out, "ck_state": ck_st}), flush=True)


if __name__ == "__main__":
    main()
