# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Drafter BF16 GEMMs at the r8/r16/r24 query rows (96/192/288) on TP4: cuBLAS vs a
swap-AB persistent tcgen05 GEMM (FlashInfer's public ``PersistentDenseGemmKernel``:
weights on MMA M, tokens on N, 1- or 2-CTA MMA, TMA multicast clusters) with split-K
as the batch mode (FP32 partials, then a sum). Cold weights (rotating copies in one
CUDA graph). Prints one CSV line per (shape, rows, config): us per call.

    python3 bench_wide_gemm.py [--rows 96,192,288] [--splits 1,2,4]
"""

import argparse
import math
import sys
import traceback

import torch

SHAPES = {  # (N_w, K) per TP4 rank, release DSpark drafter
    "qkv": (2048, 7168),
    "o_proj": (7168, 1536),
    "gate_up": (7168, 7168),
    "down": (7168, 3584),
}
CONFIGS = (  # (2-CTA, mma tiler, cluster)
    (True, (256, 96), (2, 1)),
    (True, (256, 144), (2, 1)),
    (True, (256, 144), (2, 2)),
    (True, (256, 96), (2, 2)),
    (False, (128, 96), (1, 1)),
    (False, (128, 144), (1, 1)),
    (False, (128, 144), (1, 2)),
    (True, (256, 128), (2, 1)),
    (True, (256, 192), (2, 1)),
    (True, (256, 128), (2, 2)),
    (False, (128, 128), (1, 1)),
)


def time_graph(fns, calls=64, reps=5):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for f in fns:
            f()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    loops = max(1, calls // len(fns))
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(loops):
            for f in fns:
                f()
    g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(4):
            g.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1e3 / (4 * loops * len(fns)))
    ts.sort()
    del g
    return ts[len(ts) // 2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", default="96,192,288")
    p.add_argument("--splits", default="1,2,4")
    p.add_argument("--shapes", default=",".join(SHAPES))
    p.add_argument("--configs", default="")
    args = p.parse_args()
    rows_list = [int(r) for r in args.rows.split(",")]
    splits = [int(s) for s in args.splits.split(",")]

    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as utils
    from cutlass.cute.runtime import from_dlpack
    from flashinfer.cute_dsl.gemm_allreduce_two_shot import PersistentDenseGemmKernel

    torch.manual_seed(0)
    dev = torch.device("cuda")
    hw = utils.HardwareInfo()
    configs = CONFIGS
    if args.configs == "none":
        configs = []
    elif args.configs:
        configs = [CONFIGS[int(i)] for i in args.configs.split(",")]

    def ct(t, lead):
        return from_dlpack(t, assumed_align=16).mark_layout_dynamic(leading_dim=lead)

    compiled = {}
    print("shape,rows,impl,two_cta,mma,cluster,split,us,us_with_sum,rel_err", flush=True)
    for name in args.shapes.split(","):
        n_w, k = SHAPES[name]
        nbytes = n_w * k * 2
        copies = max(2, min(64, math.ceil((640 << 20) / nbytes)))
        ws = [torch.randn(n_w, k, device=dev, dtype=torch.bfloat16) * 0.02
              for _ in range(copies)]
        for rows in rows_list:
            x = torch.randn(rows, k, device=dev, dtype=torch.bfloat16)
            ref = (x.float() @ ws[0].float().t())
            outs = [torch.empty(rows, n_w, device=dev, dtype=torch.bfloat16)
                    for _ in ws]
            fns = [lambda w=w, o=o: torch.mm(x, w.t(), out=o) for w, o in zip(ws, outs)]
            t = time_graph(fns)
            if args.configs in ("", "none"):
                print(f"{name},{rows},cublas,,,,1,{t:.2f},{t:.2f},0", flush=True)
            for two, mma, cluster in configs:
                for s in splits:
                    if k % (s * 64):
                        continue
                    kc = k // s
                    try:
                        cdt = torch.bfloat16 if s == 1 else torch.float32
                        key = (two, mma, cluster, s == 1)
                        a_views = [w.as_strided((n_w, kc, s), (k, 1, kc)) for w in ws]
                        b_view = x.as_strided((rows, kc, s), (k, 1, kc))
                        cs = [torch.empty(s, rows, n_w, device=dev, dtype=cdt)
                              for _ in ws]
                        c_views = [c.permute(2, 1, 0) for c in cs]  # (N_w, rows, s)
                        a_ct = [ct(a, 1) for a in a_views]
                        b_ct = ct(b_view, 1)
                        c_ct = [ct(c, 0) for c in c_views]
                        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
                        if key not in compiled:
                            gemm = PersistentDenseGemmKernel(
                                cutlass.Float32, two, mma, cluster, True)
                            mac = hw.get_max_active_clusters(cluster[0] * cluster[1])
                            compiled[key] = cute.compile(
                                gemm, a_ct[0], b_ct, c_ct[0], mac, stream)
                        fn = compiled[key]

                        def call(i, fn=fn, b_ct=b_ct):
                            st = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
                            fn(a_ct[i], b_ct, c_ct[i], st)

                        call(0)
                        torch.cuda.synchronize()
                        got = cs[0].float().sum(0)
                        err = ((got - ref).norm() / ref.norm()).item()
                        g_fns = [lambda i=i: call(i) for i in range(len(ws))]
                        tg = time_graph(g_fns)
                        if s > 1:
                            red = [torch.empty(rows, n_w, device=dev,
                                               dtype=torch.bfloat16) for _ in ws]
                            r_fns = [lambda i=i: (call(i), red[i].copy_(cs[i].sum(0)))
                                     for i in range(len(ws))]
                            tr = time_graph(r_fns)
                        else:
                            tr = tg
                        print(f"{name},{rows},persist,{int(two)},{mma[0]}x{mma[1]},"
                              f"{cluster[0]}x{cluster[1]},{s},{tg:.2f},{tr:.2f},"
                              f"{err:.2e}", flush=True)
                    except Exception as e:  # noqa: BLE001
                        print(f"{name},{rows},persist,{int(two)},{mma[0]}x{mma[1]},"
                              f"{cluster[0]}x{cluster[1]},{s},FAIL,,"
                              f"{type(e).__name__}: {str(e)[:120]!r}", flush=True)
                        traceback.print_exc(limit=2, file=sys.stderr)
                        torch.cuda.synchronize()
            del outs
        del ws
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
