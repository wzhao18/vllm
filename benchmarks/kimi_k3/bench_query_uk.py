# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""dsa.query_uk microbench + check vs the stock bmm + pack (1 GPU, LocalQuery shapes:
96 gathered heads, 128 padded query heads, R = verify rows of the TP group).

Prints per R: us per call (cold L2: a 1 GiB write between calls; and warm, back to
back in a CUDA graph), and the mismatch count vs the stock arithmetic (BF16 bmm
output, x * (1 / q_scale), clamp +-448, E4M3 RN)."""

import argparse
import json

import torch

G, P, QK, NOPE, LAT, ROW = 96, 128, 192, 128, 512, 576


def stock(q, uk_t, scale, rows):
    qv = q.view(rows, G, QK)
    lat = torch.bmm(qv[:, :, :NOPE].transpose(0, 1), uk_t.transpose(1, 2))  # [G,R,512]
    full = torch.cat([lat.transpose(0, 1), qv[:, :, NOPE:]], dim=-1).float()
    inv = 1.0 / scale.float()
    y = (full * inv).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    out = torch.zeros((rows, P, ROW), dtype=torch.float8_e4m3fn, device=q.device)
    out[:, :G] = y
    return out


def cold(fn, flush, iters):
    ts = []
    for i in range(iters + 3):
        flush.zero_()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        if i >= 3:
            ts.append(a.elapsed_time(b) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


def warm(fn, n=20, reps=10):
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(n):
                fn()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1e3 / n)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="128,256,384")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--splits", default="auto", help="row-tile splits, e.g. auto,1")
    ap.add_argument("--ncu", action="store_true", help="one call per R (for ncu)")
    args = ap.parse_args()
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.query_uk import query_uk

    torch.manual_seed(0)
    dev = torch.device("cuda")
    flush = torch.empty(1 << 30, dtype=torch.uint8, device=dev)
    res = {}
    for rows, sp in [(int(r), x) for r in args.rows.split(",")
                     for x in args.splits.split(",")]:
        split = None if sp == "auto" else int(sp)
        if split and -(-rows // 128) % split:
            continue
        q = (torch.randn(rows, G * QK, device=dev) * 2).to(torch.bfloat16)
        uk_t = (torch.randn(G, LAT, NOPE, device=dev) * 0.1).to(torch.bfloat16)
        scale = torch.tensor([0.02], device=dev)
        out = torch.zeros((rows, P, ROW), dtype=torch.float8_e4m3fn, device=dev)
        fn = lambda: query_uk(q, uk_t, scale, out, split)  # noqa: E731
        fn()
        torch.cuda.synchronize()
        if args.ncu:
            continue
        ref = stock(q, uk_t, scale, rows)
        a, b = out.view(torch.uint8), ref.view(torch.uint8)
        bad = (a != b)
        # K-order differences may flip a BF16 rounding: allow +-1 E4M3 code on the latent
        lat_bad = bad[:, :, :LAT].sum().item()
        diff = (a[:, :G].int() - b[:, :G].int()).abs()
        big = (diff[:, :, :LAT] > 1).sum().item()
        pe_bad = bad[:, :, LAT:].sum().item()
        pad_bad = (a[:, G:] != 0).sum().item()
        res[f"{rows}/{sp}"] = dict(cold_us=round(cold(fn, flush, args.iters), 2),
                         warm_us=round(warm(fn), 2), latent_mismatch=lat_bad,
                         latent_gt1code=big, qpe_mismatch=pe_bad, pad_nonzero=pad_bad,
                         n=rows * G * ROW)
        print(rows, sp, json.dumps(res[f"{rows}/{sp}"]), flush=True)
    print("RESULT", json.dumps(res))


if __name__ == "__main__":
    main()
