#!/usr/bin/env python3
"""P-GEMM: shared-expert SiTU + MXFP8 quant (mxfp8_act mode 4, situ4_row_kernel) bitwise dump + timing.

usage: pgemm_situ_bench.py <out.pt>   [run from the overlay root to test]
       pgemm_situ_bench.py --cmp a.pt b.pt
Cases: contiguous gate_up [M, 2*6144] (non-packed prefill path) at F16K/F8K rows, the packed
form (gate_up as a column window of [M, 3584 + 12288]), odd row counts, and the precise-exp form
(beta without an integer half -> A_POW 0) and no clamp."""
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
    inter = 6144
    res, dump = [], {}
    # rows, column offset of gate_up in the source row, beta, linear_beta
    cases = [(16384, 0, 4.0, 25.0), (8192, 0, 4.0, 25.0), (16384, 3584, 4.0, 25.0),
             (4100, 0, 4.0, 25.0), (37, 0, 4.0, 25.0), (3000, 0, 3.0, 25.0),
             (3000, 0, 4.0, None)]
    for rows, off, beta, lb in cases:
        torch.manual_seed(rows + off)
        src = (torch.randn(rows, off + 2 * inter, device=dev) * 2).to(torch.bfloat16)
        x = src[:, off:]
        buf = mxfp8_act.Mxfp8Buffer(rows, inter, x.device, x.dtype)
        buf.data.view(torch.uint8).fill_(0x5A)
        buf.scale.fill_(0x5A)

        def f():
            mxfp8_act.situ_and_mul_quantize_fast(x, beta, lb, out=buf, col=0)

        f()
        torch.cuda.synchronize()
        qa = buf.activation()
        dump[f"{rows}@{off}b{beta}l{lb}"] = [qa.data.view(torch.uint8).clone().cpu(),
                                            qa.scale.view(torch.uint8).clone().cpu()]
        ts = []
        for _ in range(7):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(20):
                f()
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e) * 1000 / 20)
        us = sorted(ts)[3]
        gb = rows * inter * (2 + 2 + 1) / 1e9
        res.append({"rows": rows, "off": off, "beta": beta, "lb": lb, "us": round(us, 2),
                    "TBps": round(gb / us * 1e3, 2)})
        print(json.dumps(res[-1]), flush=True)
    # generic act_quant_kernel: gate_sigmoid (mode 3), plain quant (mode 0), precise SiTU (mode 1)
    for mode, rows, d in [(3, 16384, 8192), (3, 8192, 8192), (3, 4100, 4096), (0, 16384, 7168),
                          (0, 37, 7168), (1, 16384, 6144), (1, 300, 6144)]:
        torch.manual_seed(rows + d + mode)
        xx = (torch.randn(rows, 2 * d if mode == 1 else d, device=dev) * 2).to(torch.bfloat16)
        gg = (torch.randn(rows, d, device=dev) * 2).to(torch.bfloat16)
        buf = mxfp8_act.Mxfp8Buffer(rows, d, xx.device, xx.dtype)
        buf.data.view(torch.uint8).fill_(0x5A)
        buf.scale.fill_(0x5A)

        def f():
            if mode == 3:
                mxfp8_act.gate_sigmoid_mul_quantize(xx, gg, out=buf, col=0)
            elif mode == 0:
                mxfp8_act.quantize(xx, out=buf, col=0)
            else:
                mxfp8_act.situ_and_mul_quantize(xx, 4.0, 25.0, out=buf, col=0)

        f()
        torch.cuda.synchronize()
        qa = buf.activation()
        dump[f"m{mode}:{rows}x{d}"] = [qa.data.view(torch.uint8).clone().cpu(),
                                       qa.scale.view(torch.uint8).clone().cpu()]
        ts = []
        for _ in range(7):
            s0, e0 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s0.record()
            for _ in range(20):
                f()
            e0.record()
            torch.cuda.synchronize()
            ts.append(s0.elapsed_time(e0) * 1000 / 20)
        us = sorted(ts)[3]
        gb = rows * d * ((2 if mode == 0 else 4) + 1) / 1e9
        print(json.dumps({"mode": mode, "rows": rows, "d": d, "us": round(us, 2),
                          "TBps": round(gb / us * 1e3, 2)}), flush=True)
    torch.save(dump, sys.argv[1])


if __name__ == "__main__":
    main()
