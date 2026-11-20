"""D-GEMM ncu driver: eager calls of one vendored sm107 MXFP8 tactic on rotating cold
weight copies (for ncu --launch-skip/--launch-count). Arm syntax as bench_oproj_mx.py:
<tile MxN>:<cluster>:<swap 0/1>:<inst_m>:<pdl 0/1>:<prewait>[:<producers>[:<tile_k>[:<dbg>]]]
"""

import argparse

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=12448)
ap.add_argument("--k", type=int, default=7168)
ap.add_argument("--rows", type=int, default=384)
ap.add_argument("--arm", required=True)
ap.add_argument("--calls", type=int, default=12)
args = ap.parse_args()

from flashinfer import mxfp8_quantize  # noqa: E402

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.sm107 import (  # noqa: E402
    prewait as pw,
)

dev = torch.device("cuda")
N, K, R = args.n, args.k, args.rows
w_q, w_s = mxfp8_quantize(torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.05, True)
ncopy = max(4, -(-1_600_000_000 // (N * K)))
W = [w_q.clone() for _ in range(ncopy)]
WS = [w_s.clone() for _ in range(ncopy)]
xq, xs = mxfp8_quantize(torch.randn(R, K, device=dev, dtype=torch.bfloat16), True)
tile, cl, swap, instm, pdl, pre, *extra = args.arm.split(":")
prods, tile_k, dbg = ([int(v) for v in extra] + [1, 128, 0][len(extra):])[:3]
tac = (tuple(int(v) for v in tile.split("x")), tuple(int(v) for v in cl.split("x")),
       swap == "1", int(instm))
g = pw.Mxfp8PreGemm(N, K, R, tac, use_pdl=pdl == "1", prewait=int(pre),
                    producers=prods, tile_k=tile_k, dbg=dbg)
out = torch.empty(R, N, device=dev, dtype=torch.bfloat16)
flush = torch.empty(512 * 1024 * 1024, device=dev, dtype=torch.uint8)
for i in range(args.calls):
    flush.zero_()
    g(xq, xs, W[i % ncopy], WS[i % ncopy], out)
torch.cuda.synchronize()
print("done", args.arm, "stages", g.stages, flush=True)
