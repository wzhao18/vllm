"""D-GEMM R4 design probe: kda_in-class MXFP8 GEMM as an L=2 K-split (the two K halves
are the kernel's batch: W and X stored [2][rows][K/2], scales per half) with FP32
partial planes [2][R][N] out, vs today's one-plane BF16 call. Measures the GEMM side of
the 'partials consumed by the next op' form (lu2 pattern); cold weights, CUDA-graph
chain, us per call. Arms: <tile MxN>:<cluster>:<swap 0/1>:<inst_m>:<L 1|2>:<c bf16|f32>.
"""

import argparse
import json
import statistics

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_stream, make_ptr

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=12448)
ap.add_argument("--k", type=int, default=7168)
ap.add_argument("--rows", default="128,256")
ap.add_argument("--arms", required=True)
ap.add_argument("--chain", type=int, default=16)
ap.add_argument("--reps", type=int, default=15)
args = ap.parse_args()

from flashinfer import mxfp8_quantize  # noqa: E402

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.sm107 import (  # noqa: E402
    prewait as pw,
)

dev = torch.device("cuda")
N, K = args.n, args.k
torch.manual_seed(0)
w_bf = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.05


def quant_split(t, L):
    """[L][rows][K/L] fp8 + per-half swizzled scales concatenated (batch-compact)."""
    kk = t.shape[1] // L
    qs, ss = [], []
    for i in range(L):
        q, s = mxfp8_quantize(t[:, i * kk:(i + 1) * kk].contiguous(), True)
        qs.append(q)
        ss.append(s.reshape(-1))
    return torch.stack(qs).contiguous(), torch.cat(ss).contiguous()


def compile_arm(tile, cluster, swap, inst_m, L, cdt):
    gemm = pw.Sm107BlockScaledPersistentDenseGemmKernel(
        pw.SF_VEC, (inst_m, tile[1], pw.INST_K), (tile[0], tile[1], pw.TILER_K), cluster,
        prefetch_dist=0, use_pdl=True, prewait_stages=0, weight_is_a=swap, quiet_wait=False,
        producers=1, die_claim=False,
    )
    gemm.dbg_skip_mma = False
    gemm.max_ab_stages = 0
    sm, sk, sn = cute.sym_int(), cute.sym_int(), cute.sym_int()
    f8 = cutlass.Float8E4M3FN
    a = cute.runtime.make_fake_compact_tensor(f8, (sm, sk), stride_order=(1, 0), assumed_align=16)
    b = cute.runtime.make_fake_compact_tensor(f8, (sn, sk), stride_order=(1, 0), assumed_align=16)
    ct = cutlass.Float32 if cdt == "f32" else cutlass.BFloat16
    if swap:
        c = cute.runtime.make_fake_compact_tensor(ct, (sn, sm), stride_order=(0, 1), assumed_align=16)
    else:
        c = cute.runtime.make_fake_compact_tensor(ct, (sm, sn), stride_order=(1, 0), assumed_align=16)
    sfa = make_ptr(cutlass.Float8E8M0FNU, 16, cute.AddressSpace.gmem, 16)
    sfb = make_ptr(cutlass.Float8E8M0FNU, 16, cute.AddressSpace.gmem, 16)
    alpha = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (1,), assumed_align=4)
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    mac = pw._max_active_clusters(cluster[0] * cluster[1])
    return cute.compile(
        gemm.wrapper, a, b, c, cutlass.Int64(1), cutlass.Int64(1), cutlass.Int64(1), L,
        sfa, sfb, alpha, mac, stream, swap,
        options=f"--opt-level 2 --gpu-arch={pw._arch(0)} --enable-tvm-ffi",
    ), gemm


flush = torch.empty(512 * 1024 * 1024, device=dev, dtype=torch.uint8)
alpha_t = torch.ones(1, device=dev, dtype=torch.float32)
res = []
for R in [int(r) for r in args.rows.split(",")]:
    x = torch.randn(R, K, device=dev, dtype=torch.bfloat16)
    xq1, xs1 = mxfp8_quantize(x, True)
    wq1, ws1 = mxfp8_quantize(w_bf, True)
    ref = torch._scaled_mm(xq1, wq1.t(), scale_a=xs1.view(torch.float8_e8m0fnu),
                           scale_b=ws1.view(torch.float8_e8m0fnu), out_dtype=torch.float32)
    for arm in args.arms.split(","):
        try:
            tile, cl, swap, instm, L, cdt = arm.split(":")
            tile = tuple(int(v) for v in tile.split("x"))
            cl = tuple(int(v) for v in cl.split("x"))
            swap, L = swap == "1", int(L)
            comp, gemm = compile_arm(tile, cl, swap, int(instm), L, cdt)
            xq, xs = quant_split(x, L)
            wq, ws = quant_split(w_bf, L)
            ncopy = max(args.chain, -(-1_600_000_000 // wq.numel()))
            W = [(wq.clone(), ws.clone()) for _ in range(ncopy)]
            kh = K // L
            km, kn = (N, R) if swap else (R, N)
            sf = ((km + 127) // 128, (kn + 127) // 128, (kh // 32 + 3) // 4)
            odt = torch.float32 if cdt == "f32" else torch.bfloat16
            outs = [torch.empty(L, R, N, device=dev, dtype=odt) for _ in range(2)]

            def call(i, out):
                w, s = W[i % ncopy]
                if swap:
                    a_, b_, asf, bsf = w[0], xq[0], s, xs
                    c_ = out[0].as_strided((R, N), (1, R))  # stride order only; pointer used
                else:
                    a_, b_, asf, bsf = xq[0], w[0], xs, s
                    c_ = out[0]
                comp(a_, b_, c_, *sf, asf.data_ptr(), bsf.data_ptr(), alpha_t)

            out = outs[0]
            out.fill_(float("nan"))
            call(0, out)
            torch.cuda.synchronize()
            tot = out.float().sum(0) if L > 1 else out[0].float()
            rel = ((tot - ref).abs().max() / ref.abs().max()).item()
            s = torch.cuda.Stream()
            with torch.cuda.stream(s):
                for i in range(3):
                    call(i, outs[i % 2])
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=s):
                for i in range(args.chain):
                    call(i, outs[i % 2])
            ts = []
            for _ in range(args.reps + 2):
                flush.zero_()
                e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                e0.record()
                g.replay()
                e1.record()
                torch.cuda.synchronize()
                ts.append(e0.elapsed_time(e1) * 1000 / args.chain)
            row = dict(R=R, arm=arm, rel=rel, stages=getattr(gemm, "num_ab_stage", None),
                       us=round(statistics.median(ts[2:]), 2))
        except Exception as e:  # report and continue
            row = dict(R=R, arm=arm, error=f"{type(e).__name__}: {str(e)[:300]}")
        print(json.dumps(row), flush=True)
        res.append(row)
