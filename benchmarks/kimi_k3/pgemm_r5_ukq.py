"""R5-M3 bench: fused W_UK + RoPE concat + FP8 query (UkQueryFp8) vs the stock 3 passes.

python3 pgemm_r5_ukq.py [--t 16383] [--tactics 256,256,2,1;256,256,2,2] [--scale 1.0] [--iters 50]
Prints correctness (RoPE part bitwise, NoPE part vs stock) and us/call per arm.
"""
import argparse
import json
import time

import torch

from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm import uk_query as U


def stock_parts(q, w, scale, cat, out):
    from vllm import _custom_ops as ops

    p = w.shape[1]
    lat = w.shape[2]

    def bmm_copy():
        torch.bmm(q[..., :p].transpose(0, 1), w, out=cat[..., :lat].transpose(0, 1))
        cat[..., lat:].copy_(q[..., p:])

    def quant():
        ops.scaled_fp8_quant(cat.view(cat.shape[0], -1), scale, output=out.view(out.shape[0], -1))

    return bmm_copy, quant


def timeit(fn, iters):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000.0 / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t", type=int, default=16383)
    ap.add_argument("--h", type=int, default=96)
    ap.add_argument("--tactics", default="256,256,2,1")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--qmul", type=float, default=1.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.manual_seed(0)
    dev = "cuda"
    T, H, P, R, L = a.t, a.h, 128, 64, 512
    q = (torch.randn(T, H, P + R, device=dev) * 2.0 * a.qmul).to(torch.bfloat16)
    w = (torch.randn(H, P, L, device=dev) * 0.5).to(torch.bfloat16)
    scale = torch.tensor([a.scale], device=dev, dtype=torch.float32)
    inv = scale.reciprocal()
    ref = U.reference(q, w, scale)
    res = {"t": T, "h": H, "scale": a.scale, "qmul": a.qmul}
    cat = q.new_empty((T, H, L + R))
    out_s = torch.empty((T, H, L + R), device=dev, dtype=torch.float8_e4m3fn)
    bc, qu = stock_parts(q, w, scale, cat, out_s)
    bc()
    qu()
    torch.cuda.synchronize()
    res["stock_vs_ref_mismatch"] = int((out_s.view(torch.uint8) != ref.view(torch.uint8)).sum())
    res["stock_bmm_copy_us"] = timeit(bc, a.iters)
    res["stock_quant_us"] = timeit(qu, a.iters)
    res["stock_total_us"] = res["stock_bmm_copy_us"] + res["stock_quant_us"]
    for tac in a.tactics.split(";"):
        f = [int(x) for x in tac.split(",")]
        tm, tn, cm, cn = f[:4]
        tma = bool(f[4]) if len(f) > 4 else True
        epn = f[5] if len(f) > 5 else 64
        mab = f[6] if len(f) > 6 and f[6] else None
        plain = len(f) > 7 and bool(f[7])
        skip = len(f) > 8 and bool(f[8])
        unit = len(f) > 9 and bool(f[9])
        clamp = not (len(f) > 10 and f[10] == 0)
        k = U.UkQueryFp8(
            ((tm, tn), (cm, cn), tma, epn, mab), plain=plain, skip_rope=skip,
            unit_scale=unit, clamp=clamp,
        )
        out = torch.full((T, H, L + R), 0x55, device=dev, dtype=torch.uint8).view(
            torch.float8_e4m3fn
        )
        t0 = time.time()
        k(q, w, inv, out)
        torch.cuda.synchronize()
        comp_s = time.time() - t0
        ob, rb = out.view(torch.uint8), ref.view(torch.uint8)
        pe_eq = int((ob[..., L:] != rb[..., L:]).sum())
        np_eq = int((ob[..., :L] != rb[..., :L]).sum())
        d = (out[..., :L].float() - ref[..., :L].float()).abs()
        rel = d / ref[..., :L].float().abs().clamp(min=2.0 ** -6)
        us = timeit(lambda: k(q, w, inv, out), a.iters)
        res[tac] = {
            "compile_s": round(comp_s, 1),
            "rope_mismatch": pe_eq,
            "nope_mismatch": np_eq,
            "nope_max_rel": float(rel.max()),
            "us": us,
            "stages": [getattr(k.gemm, n, None) for n in ("num_acc_stage", "num_ab_stage", "num_c_stage")],
        }
        print(tac, json.dumps(res[tac]), flush=True)
    print(json.dumps(res), flush=True)
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
