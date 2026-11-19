# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Drafter ``WideGemm`` (cutedsl/gemm/wide.py) vs cuBLAS at the TP4 drafter shapes:
correctness (BF16 ulps vs an FP32 reference rounded once) and us per call on cold
weights (rotating copies replayed in one CUDA graph).

    python3 bench_wide_own.py [--rows 96,192,288] [--cases qkv:12,o_proj:3,...]
"""

import argparse
import math
import sys
import traceback

import torch

sys.path.insert(0, __import__("os").path.dirname(__file__))
from bench_wide_gemm import SHAPES, time_graph  # noqa: E402

DEFAULT_CASES = "qkv:8,qkv:4,o_proj:3,o_proj:2,gate_up:3,gate_up:silu3,down:3,down:2"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", default="96,192,288")
    p.add_argument("--cases", default=DEFAULT_CASES)
    p.add_argument("--no-time", action="store_true")
    args = p.parse_args()
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.wide import (
        PreparedWide,
        supports,
    )

    torch.manual_seed(0)
    dev = torch.device("cuda")
    print("shape,rows,impl,split,us,max_ulp,rel_err", flush=True)
    for case in args.cases.split(","):
        name, spec = case.split(":")
        silu = spec.startswith("silu")
        split = int(spec[4:] if silu else spec)
        n, k = SHAPES[name]
        copies = max(2, min(48, math.ceil((640 << 20) / (n * k * 2))))
        ws = [torch.randn(n, k, device=dev, dtype=torch.bfloat16) * 0.02
              for _ in range(copies)]
        for rows in [int(r) for r in args.rows.split(",")]:
            if not supports(rows, n, k, split, silu):
                print(f"{name},{rows},wide{'_silu' if silu else ''},{split},SKIP,,")
                continue
            x = torch.randn(rows, k, device=dev, dtype=torch.bfloat16)
            try:
                gemms = [PreparedWide(w, {rows: split}, silu=silu).prepare()
                         for w in ws]
                cols = n // 2 if silu else n
                outs = [torch.empty(rows, cols, device=dev, dtype=torch.bfloat16)
                        for _ in ws]
                for _ in range(3):  # counters must re-arm across launches
                    gemms[0].run(x, outs[0])
                torch.cuda.synchronize()
                full = x.float() @ ws[0].float().t()
                if silu:
                    g, u = full[:, : n // 2].bfloat16(), full[:, n // 2:].bfloat16()
                    ref = torch.nn.functional.silu(g.float()).bfloat16().float()
                    ref = (ref * u.float()).bfloat16()
                else:
                    ref = full.bfloat16()
                got = outs[0]
                diff = (got.float() - ref.float()).abs()
                ulp = (ref.float().abs().clamp_min(1e-30).log2().floor() - 7).exp2()
                max_ulp = ((diff - 1e-4 * ref.float().abs().amax()).clamp_min(0) / ulp).max().item()
                rel = (diff.norm() / ref.float().norm()).item()
                us = float("nan")
                if not args.no_time:
                    fns = [lambda gm=gm, o=o: gm.run(x, o) for gm, o in zip(gemms, outs)]
                    us = time_graph(fns)
                print(f"{name},{rows},wide{'_silu' if silu else ''},{split},{us:.2f},"
                      f"{max_ulp:.1f},{rel:.2e}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"{name},{rows},wide,{split},FAIL,,{type(e).__name__}: "
                      f"{str(e)[:200]!r}", flush=True)
                traceback.print_exc(limit=4, file=sys.stderr)
                torch.cuda.synchronize()
        del ws
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
