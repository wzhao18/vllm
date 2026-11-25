# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbench of the #63 DCP top-k select kernel (DCPTopkSelectKernel) on one GPU at
DCP decode shapes: ``--shapes`` of W ranks x N packed candidates per row -> top-K
(default: the DSA chain microbench's W4 x 192 -> 512 and a DCP16 rank's 16 x 192 ->
2048), this rank's physical rows out (block 64), the non-mailbox (all-gathered)
form, every source's rows pre-landed (no wait). CUDA graph of --calls launches:
us per call (median of --reps replays) and the CUPTI kernel duration (mean).

    python3 benchmarks/kernels/bench_k31_dcp_topk_select.py --rows 8,16,64
"""

import argparse

import torch

BLOCK = 64


def _ord16(x: torch.Tensor) -> torch.Tensor:
    bits = x.view(torch.int16).to(torch.int64) & 0xFFFF
    neg = (bits & 0x8000) != 0
    return torch.where(neg, (~bits) & 0xFFFF, bits | 0x8000)


def case(w, n, rows, levels, seed):
    """Gathered [W, rows, N + N/32] int32 B' words of distinct local ids."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    score = torch.randn((w, rows, n), generator=gen, device="cuda")
    if levels:
        score = torch.floor(score * levels) / levels
    score = torch.relu(score + 0.3).to(torch.bfloat16)
    base = torch.randint(0, 1 << 17, (w, rows, 1), generator=gen, device="cuda")
    local = (base + torch.arange(n, device="cuda") * 641) % (1 << 17)
    words = (_ord16(score) << 16) | (0xFFFF - (local & 0xFFFF))
    plane = torch.zeros((w, rows, n // 32), dtype=torch.int64, device="cuda")
    hi = (local >> 16) & 1
    for j in range(n):
        plane[:, :, j // 32] |= hi[:, :, j] << (j % 32)
    packed = torch.cat([words, plane], dim=2)
    packed = torch.where(packed >= 1 << 31, packed - (1 << 32), packed)
    return packed.to(torch.int32).contiguous()


def cupti_us(fn, name="Select"):
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    durs = [
        e.device_time
        for e in prof.events()
        if e.device_type.name == "CUDA" and name.lower() in e.name.lower()
    ]
    return sum(durs) / max(1, len(durs)), len(durs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="8,16,32,64,128,256,512")
    ap.add_argument("--shapes", default="4x192x512,16x192x2048", help="WxNxK,...")
    ap.add_argument("--levels", default="0,4", help="score quantization (0: none)")
    ap.add_argument("--calls", type=int, default=20)
    ap.add_argument("--reps", type=int, default=50)
    args = ap.parse_args()
    from vllm.v1.attention.ops.dcp_topk_merge.kernels import (
        _DCP_TOPK_SELECT_KERNEL,
        _RowsOutput,
    )

    for shape in args.shapes.split(","):
        w, n, k = map(int, shape.split("x"))
        for levels in map(int, args.levels.split(",")):
            for rows in map(int, args.rows.split(",")):
                packed = case(w, n, rows, levels, seed=rows + levels)
                blocks = (1 << 17) // BLOCK + 1
                table = torch.randint(
                    0, 1 << 20, (rows, blocks), device="cuda", dtype=torch.int32
                )
                rows_out = torch.empty((rows, k), dtype=torch.int32, device="cuda")
                valid = torch.empty((rows,), dtype=torch.int32, device="cuda")
                req = torch.arange(rows, dtype=torch.int32, device="cuda")
                ro = _RowsOutput(req, table, rows_out, valid, w - 1, BLOCK, 64)

                def call(packed=packed, ro=ro, n=n, k=k):
                    _DCP_TOPK_SELECT_KERNEL(packed, None, local_topk=n, topk=k, rows=ro)

                for _ in range(3):
                    call()
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                s = torch.cuda.Stream()
                with torch.cuda.stream(s):
                    call()
                    torch.cuda.synchronize()
                    with torch.cuda.graph(g, stream=s):
                        for _ in range(args.calls):
                            call()
                ts = []
                for _ in range(args.reps):
                    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
                    a.record()
                    g.replay()
                    b.record()
                    b.synchronize()
                    ts.append(a.elapsed_time(b) * 1000 / args.calls)
                ts.sort()
                med = ts[len(ts) // 2]
                dur, cnt = cupti_us(lambda g=g: [g.replay() for _ in range(3)])
                print(
                    f"REC select W{w} N{n} K{k} rows {rows:4d} levels {levels}: "
                    f"{med:7.2f} us/call (min {ts[0]:.2f}) kernel {dur:6.2f} us "
                    f"({cnt} kernels)",
                    flush=True,
                )


if __name__ == "__main__":
    main()
