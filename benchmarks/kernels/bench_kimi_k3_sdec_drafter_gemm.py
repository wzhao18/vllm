# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 specialized decode: release DSpark drafter BF16 GEMMs (TP4, D3-32) on one
GPU, cold weights (one weight copy per call in a CUDA graph, copies >> L2), us per call:
F.linear (cuBLAS heuristic), the lt_pin pinned cuBLASLt algorithm, and FlashInfer's
public ``mm_bf16`` backends. Correctness vs F.linear (max |d| / max |ref|).

python benchmarks/kernels/bench_kimi_k3_sdec_drafter_gemm.py [--rows 288] [--only gu,lm]
"""

import argparse
import statistics

import torch
import torch.nn.functional as F

SHAPES = {  # name: (N, K, row kind)
    "qkv": (2048, 7168, "q"),
    "o": (7168, 1536, "q"),
    "gu": (7168, 7168, "q"),
    "down": (7168, 3584, "q"),
    "lm": (40960, 7168, "q"),
    "ctx_kv": (2560, 7168, "c"),
    "fc": (1792, 35840, "c"),
}
FI_BACKENDS = ("cudnn", "cutlass", "tgv", "cublaslt", "cute-dsl", "auto")


def time_graph(fn, copies, reps=10, rounds=5):
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for i in range(len(copies)):
            fn(i)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        for i in range(len(copies)):
            fn(i)
    ts = []
    for _ in range(rounds):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(reps):
            g.replay()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1e3 / reps / len(copies))
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="288")  # query rows (B x k3); context rows = B x 4
    ap.add_argument("--only", default="")
    ap.add_argument("--fi", default=",".join(FI_BACKENDS))
    ap.add_argument("--wide", default="1,2,3,4,6,8")
    args = ap.parse_args()
    torch.manual_seed(0)
    dev = torch.device("cuda")
    print(torch.cuda.get_device_name(), flush=True)
    from vllm.models.kimi_k3.nvidia.specialized_decode import lt_pin

    try:
        import flashinfer
        from flashinfer.gemm import mm_bf16
    except Exception as e:  # noqa: BLE001
        mm_bf16 = None
        print("flashinfer mm_bf16 unavailable:", repr(e)[:200])
    else:
        print("flashinfer", flashinfer.__version__)
    only = set(args.only.split(",")) - {""}
    for q in [int(r) for r in args.rows.split(",")]:
        for name, (n, k, kind) in SHAPES.items():
            if only and name not in only:
                continue
            m = q if kind == "q" else q // 3 * 4
            wbytes = n * k * 2
            ncopy = max(2, min(64, int(1.5e9 // wbytes) + 1))
            ws = [torch.randn(n, k, device=dev, dtype=torch.bfloat16) * 0.02 for _ in range(ncopy)]
            x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
            out = torch.empty(m, n, device=dev, dtype=torch.bfloat16)
            ref = F.linear(x, ws[0]).float()
            res = {}

            def run(tag, fn):
                try:
                    fn(0)
                    err = ((out.float() - ref).abs().max() / ref.abs().max()).item()
                    res[tag] = (time_graph(fn, ws), err)
                except Exception as e:  # noqa: BLE001
                    res[tag] = (float("nan"), repr(e)[:90])

            run("F.linear", lambda i: torch.matmul(x, ws[i].t(), out=out))
            pins = lt_pin.pins_for((n, k))
            if m in pins:
                pls = [lt_pin.pinned(w, [m]) for w in ws]
                if all(p is not None for p in pls):
                    run("lt_pin", lambda i: pls[i].run(x, out))
            if args.wide:  # owned token-wide 2-CTA split-K GEMM (cutedsl/gemm/wide.py)
                from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm import wide

                for split in [int(s) for s in args.wide.split(",")]:
                    if not wide.supports(m, n, k, split) or (
                        split > 1 and 2 * (n // 256) * split > 212
                    ):
                        continue
                    try:
                        pws = [wide.PreparedWide(w, {m: split}).prepare() for w in ws]
                        run(f"wide{split}", lambda i, p=pws: p[i].run(x, out))
                    except Exception as e:  # noqa: BLE001
                        res[f"wide{split}"] = (float("nan"), repr(e)[:90])
            if mm_bf16 is not None:
                for be in args.fi.split(","):
                    run("fi:" + be, lambda i, be=be: mm_bf16(x, ws[i].t(), out=out, backend=be))
                try:  # FlashInfer's autotuner picks tactics per backend (cached after)
                    from flashinfer.autotuner import autotune

                    for be in ("cutlass", "cudnn", "auto"):
                        with autotune(True):
                            mm_bf16(x, ws[0].t(), out=out, backend=be)
                        run("fiT:" + be,
                            lambda i, be=be: mm_bf16(x, ws[i].t(), out=out, backend=be))
                except Exception as e:  # noqa: BLE001
                    res["fiT"] = (float("nan"), repr(e)[:90])
            base = res["F.linear"][0]
            line = " ".join(
                f"{t}={v[0]:.2f}" + (f"(x{base / v[0]:.2f} e{v[1]:.1e})" if isinstance(v[1], float)
                                     else f"[{v[1]}]")
                for t, v in res.items())
            print(f"R{q} {name:6s} M{m} N{n} K{k} ncopy{ncopy}: {line}", flush=True)
            del ws
            torch.cuda.empty_cache()
    print("DONE")


if __name__ == "__main__":
    main()
