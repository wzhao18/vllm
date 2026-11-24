# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbench of the #63 DCP top-k select kernel (DCPTopkSelectKernel) on one GPU at
DCP16 decode shapes: W = 16 ranks x N = 192 packed candidates per row, K = 2048, this
rank's physical rows out (block 64), the non-mailbox (all-gathered) form. CUDA graph
of --calls launches, us per call (median of --reps replays).

    python3 benchmarks/kernels/bench_k31_dcp_topk_select.py --rows 8,64,512
"""

import argparse
import os
import sys

import torch

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../tests/models/kimi_k3"
    ),
)
from test_k31_dcp_topk_select import BLOCK, K, N, _case  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="8,16,32,64,128,256,512")
    ap.add_argument("--levels", default="0,4", help="score quantization (0: none)")
    ap.add_argument("--calls", type=int, default=20)
    ap.add_argument("--reps", type=int, default=50)
    args = ap.parse_args()
    from vllm.v1.attention.ops.dcp_topk_merge.kernels import (
        _DCP_TOPK_SELECT_KERNEL,
        _RowsOutput,
    )

    for levels in map(int, args.levels.split(",")):
        for rows in map(int, args.rows.split(",")):
            packed, _, _ = _case(rows, levels, seed=rows + levels)
            blocks = (1 << 17) // BLOCK + 1
            table = torch.randint(
                0, 1 << 20, (rows, blocks), device="cuda", dtype=torch.int32
            )
            out = torch.empty((rows, K), dtype=torch.int32, device="cuda")
            rows_out = torch.empty_like(out)
            valid = torch.empty((rows,), dtype=torch.int32, device="cuda")
            req = torch.arange(rows, dtype=torch.int32, device="cuda")
            ro = _RowsOutput(req, table, rows_out, valid, 5, BLOCK, 64)

            def call(packed=packed, ro=ro):
                _DCP_TOPK_SELECT_KERNEL(packed, None, local_topk=N, topk=K, rows=ro)

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
            print(
                f"select rows {rows:4d} levels {levels}: {med:7.2f} us/call "
                f"(min {ts[0]:.2f})",
                flush=True,
            )


if __name__ == "__main__":
    main()
