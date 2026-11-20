# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DEBUG (INFRA, never land): does dsa.query_uk give the same bytes on identical inputs
when other kernels run between calls? Reference = the first call after a warm-up;
each trial runs a disturbance (big BF16 GEMMs that fill SMEM/L2 with other data,
optionally an idle gap), then query_uk, and compares the whole FP8 query, reporting
the differing (row tile, head, byte range) per trial."""

import argparse

import torch

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.query_uk import query_uk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=512)
    ap.add_argument("--heads", type=int, default=96)
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--sleep-cycles", type=int, default=2_000_000)
    args = ap.parse_args()
    torch.manual_seed(0)
    dev = "cuda"
    r, g = args.rows, args.heads
    q = (torch.randn(r, g * 192, device=dev) * 2).to(torch.bfloat16)
    uk_t = (torch.randn(g, 512, 128, device=dev) * 0.05).to(torch.bfloat16)
    scale = torch.tensor([0.02], device=dev, dtype=torch.float32)
    out = torch.zeros(r, 128, 576, dtype=torch.float8_e4m3fn, device=dev)
    a = torch.randn(8192, 8192, device=dev, dtype=torch.bfloat16)
    for _ in range(5):
        query_uk(q, uk_t, scale, out)
    torch.cuda.synchronize()
    ref = out.view(torch.uint8).clone()
    bad = 0
    for t in range(args.trials):
        mode = t % 3
        if mode == 0:
            (a @ a).sum()  # SMEM / L2 full of other data
        elif mode == 1:
            torch.cuda._sleep(args.sleep_cycles)  # idle gap
        else:
            (a @ a).sum()
            torch.cuda._sleep(args.sleep_cycles)
        out.zero_()
        query_uk(q, uk_t, scale, out)
        torch.cuda.synchronize()
        cur = out.view(torch.uint8)
        if not torch.equal(cur, ref):
            bad += 1
            idx = (cur != ref).nonzero()
            tiles = sorted(set((idx[:, 0] // 128).tolist()))
            heads = sorted(set(idx[:, 1].tolist()))
            lo, hi = int(idx[:, 2].min()), int(idx[:, 2].max())
            print(f"trial {t} mode {mode}: {idx.shape[0]} bytes differ, row tiles {tiles}, "
                  f"heads {heads[:10]} ({len(heads)}), bytes {lo}..{hi}", flush=True)
    print(f"QUERY_UK_RACE rows {r} heads {g}: {bad} of {args.trials} trials differ")


if __name__ == "__main__":
    main()
