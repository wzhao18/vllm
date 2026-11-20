#!/usr/bin/env python3
"""P-GEMM: latent RMSNorm + MXFP8 quant (mxfp8_act.rms_norm_quantize) bitwise dump + timing.

usage: pgemm_rmsq_bench.py <out.pt>  [run from the overlay root to test]
       pgemm_rmsq_bench.py --cmp a.pt b.pt
Cases: the packed-MoE form (latent 3584 into columns [6144, 9728) of the act buffer) at the
deployed prefill rows, plus standalone k = 1536 / 7168 and odd row counts."""
import json
import sys

import torch


def main():
    if sys.argv[1] == "--cmp":
        a, b = torch.load(sys.argv[2]), torch.load(sys.argv[3])
        ok = True
        for key in a:
            same = all(torch.equal(x, y) for x, y in zip(a[key], b[key]))
            ok &= same
            print(json.dumps({"case": key, "bitwise": same}))
        print(json.dumps({"all_bitwise": ok}))
        return
    from vllm.models.kimi_k3.nvidia.ops import mxfp8_act

    dev = "cuda"
    res, dump = [], {}
    cases = [(16384, 3584, 6144), (8192, 3584, 6144), (4100, 3584, 6144), (16384, 7168, 0),
             (1000, 1536, 0), (37, 3584, 6144)]
    for rows, k, col in cases:
        torch.manual_seed(rows + k)
        x = (torch.randn(rows, k, device=dev) * 2).to(torch.bfloat16)
        w = (torch.randn(k, device=dev) * 0.2 + 1.0).to(torch.bfloat16)
        y = torch.empty_like(x)
        buf = mxfp8_act.Mxfp8Buffer(rows, col + k, x.device, x.dtype)
        buf.data.view(torch.uint8).zero_()
        buf.scale.zero_()

        def f():
            mxfp8_act.rms_norm_quantize(x, w, 1e-6, out=buf, col=col, bf16_out=y)

        f()
        torch.cuda.synchronize()
        qa = buf.activation()
        dump[f"{rows}x{k}@{col}"] = [qa.data.view(torch.uint8).clone().cpu(),
                                    qa.scale.view(torch.uint8).clone().cpu(), y.view(torch.int16).cpu()]
        # L2-cold-ish timing: rotate over 4 inputs bigger than L2 when large
        ts = []
        for _ in range(5):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(20):
                f()
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e) * 1000 / 20)
        us = sorted(ts)[2]
        gb = rows * k * (2 + 2 + 1) / 1e9
        res.append({"rows": rows, "k": k, "col": col, "us": round(us, 2), "TBps": round(gb / us * 1e3, 2)})
        print(json.dumps(res[-1]), flush=True)
    torch.save(dump, sys.argv[1])


if __name__ == "__main__":
    main()
