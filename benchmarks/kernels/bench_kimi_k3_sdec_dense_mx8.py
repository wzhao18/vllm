# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DenseMx8 (specialized decode MXFP8 projection) vs the stock scaled_mm path.

Correctness against torch._scaled_mm on the same MXFP8 buffers, then timing in a
CUDA graph: back to back, and behind a predecessor that holds the SMs like the SP
gather (``--pred-us``: 672 128-thread CTAs that trigger PDL at entry and spin).

  python bench_kimi_k3_sdec_dense_mx8.py --rows 384 --n 12448 --k 7168
"""

import argparse
import re
import sys
import time
import traceback

import torch


def mx(x):
    from vllm.models.kimi_k3.nvidia.ops import mxfp8_act

    buf = mxfp8_act.Mxfp8Buffer(x.shape[0], x.shape[1], x.device)
    mxfp8_act.quantize(x, out=buf)
    return buf


def ptrs(w, x, out, part, tick):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_ptr

    g = cute.AddressSpace.gmem
    f8, e8 = cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU
    return (
        make_ptr(f8, w.data.data_ptr(), g, assumed_align=16),
        make_ptr(e8, w.scale.data_ptr(), g, assumed_align=16),
        make_ptr(f8, x.data.data_ptr(), g, assumed_align=16),
        make_ptr(e8, x.scale.data_ptr(), g, assumed_align=16),
        make_ptr(cutlass.BFloat16, out.data_ptr(), g, assumed_align=16),
        make_ptr(cutlass.Float32, part.data_ptr(), g, assumed_align=16),
        make_ptr(cutlass.Int32, tick.data_ptr(), g, assumed_align=16),
    )


def graph_time(fn, reps=50, iters=5):
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        for _ in range(reps):
            fn()
    torch.cuda.synchronize()
    best = 1e9
    for _ in range(iters):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        best = min(best, e0.elapsed_time(e1) * 1e3 / reps)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="+", default=[384])
    ap.add_argument("--n", type=int, default=12448)
    ap.add_argument("--k", type=int, default=7168)
    ap.add_argument("--variants", default="c1s4k2,c0s4k2,c1s5k2,c1s4k1")
    ap.add_argument("--copies", type=int, default=4, help="weight copies (> L2)")
    args = ap.parse_args()
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_ptr  # noqa: F401

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import stream
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.dense_mx8 import (
        DenseMx8,
    )

    dev = torch.device("cuda", 0)
    torch.manual_seed(0)
    n, k = args.n, args.k
    ws = [mx(torch.randn(n, k, device=dev, dtype=torch.bfloat16) * 0.05)
          for _ in range(args.copies)]  # fmt: skip
    print(f"VR200 {torch.cuda.get_device_name(0)} sms "
          f"{torch.cuda.get_device_properties(0).multi_processor_count}", flush=True)  # fmt: skip
    for rows in args.rows:
        x = mx(torch.randn(rows, k, device=dev, dtype=torch.bfloat16))
        ref = torch._scaled_mm(
            x.data, ws[0].data.t(), scale_a=x.scale.view(torch.float8_e8m0fnu),
            scale_b=ws[0].scale.view(torch.float8_e8m0fnu), out_dtype=torch.bfloat16,
        )  # fmt: skip
        outs = [torch.empty(rows, n, device=dev, dtype=torch.bfloat16) for _ in ws]

        def stock(i=[0]):
            j = i[0] % len(ws)
            i[0] += 1
            torch._scaled_mm(
                x.data, ws[j].data.t(), scale_a=x.scale.view(torch.float8_e8m0fnu),
                scale_b=ws[j].scale.view(torch.float8_e8m0fnu), out_dtype=torch.bfloat16,
                out=outs[j],
            )  # fmt: skip

        t_stock = graph_time(stock)
        print(f"R{rows} scaled_mm {t_stock:.2f} us/call (graph, {len(ws)} weight copies)",
              flush=True)  # fmt: skip
        for var in args.variants.split(","):
            f = dict(re.findall(r"([a-z])(\d+)", var))  # c<col>s<stages>k<ksplit>[m<mc>][d<debug>]
            col, stg, ks = int(f["c"]), int(f["s"]), int(f["k"])
            mc, dbg = int(f.get("m", 1)), int(f.get("d", 0))
            try:
                g = DenseMx8(n, k, rows, n, ksplit=ks, stages=stg, collector=bool(col), mc=mc,
                             debug=dbg)
                pw, tw = g.scratch_words()
                part = torch.zeros(pw, device=dev, dtype=torch.float32)
                tick = torch.zeros(tw, device=dev, dtype=torch.int32)
                t0 = time.time()
                args0 = ptrs(ws[0], x, outs[0], part, tick)
                fn = cute.compile(g, *args0, stream(dev))
                print(f"  {var}: compiled in {time.time() - t0:.1f}s grid {g.grid} "
                      f"sfa/sfb cols {g.sfa_cols}/{g.sfb_cols}", flush=True)  # fmt: skip
                all_args = [ptrs(w, x, o, part, tick) for w, o in zip(ws, outs)]
                fn(*all_args[0], stream(dev))
                torch.cuda.synchronize()
                d = (outs[0].float() - ref.float())
                rel = (d.norm() / ref.float().norm()).item()
                mism = (d.abs() > 1e-2 * ref.float().abs().max()).sum().item()
                bad_rows = (d.abs().amax(1) > 1e-2 * ref.float().abs().max()).nonzero()
                print(f"  {var}: rel {rel:.3e} maxabs {d.abs().max().item():.3e} "
                      f"mismatch {mism} rows {bad_rows[:6].flatten().tolist()}", flush=True)  # fmt: skip
                if mism and not dbg:
                    bc = (d.abs().amax(0) > 1e-2 * ref.float().abs().max()).nonzero().flatten()
                    print(f"     bad cols {bc.numel()} first {bc[:8].tolist()}", flush=True)
                    continue

                def ours(i=[0]):
                    j = i[0] % len(ws)
                    i[0] += 1
                    fn(*all_args[j], stream(dev))

                t = graph_time(ours)
                print(f"  {var}: {t:.2f} us/call  ({t_stock / t:.2f}x scaled_mm)", flush=True)
            except Exception as e:  # report and go on with the next variant
                tb = traceback.format_exc().splitlines()
                print(f"  {var}: FAILED {type(e).__name__}: {str(e)[:1500]}", flush=True)
                for line in [l for l in tb if "dense_mx8" in l or "Error" in l][-8:]:
                    print("     " + line.strip()[:300], flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
