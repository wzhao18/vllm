#!/usr/bin/env python3
"""P-GEMM r6: prefill DSA W_UV up-projection + output gate, fused (uv_gate) vs
stock (torch.bmm into (T, H*V) + prefill_glue gate kernel), 1 GPU.

Per T: warm (back-to-back, CUDA events over --iters) and THRASH (512 MB L2 flush
+ a straight-line code kernel before every timed call) times in us, bitwise check.
"""

import argparse
import json
import statistics

import torch

H, P, L, V = 96, 128, 512, 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", default="16384,8192")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--tactics", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from vllm.models.kimi_k3.nvidia.ops import prefill_glue
    from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm import uv_gate

    dev = torch.device("cuda")
    flush = torch.empty(512 << 20, dtype=torch.uint8, device=dev)
    w = torch.randn((L, H, P + V), device=dev, dtype=torch.bfloat16) * 0.05
    w_uv = w[..., P:].transpose(0, 1)  # the MLA layer's view
    kernels = {"fused": uv_gate.UvGate(), "fused_subtile": uv_gate.UvGate(tile_ahead=False)}
    for tac in filter(None, args.tactics.split(";")):
        tm, tn, cm, st = (int(x) for x in tac.split(","))
        kernels["fused_%s" % tac.replace(",", "x")] = uv_gate.UvGate(
            ((tm, tn), (cm, 1), st or None)
        )
    rows = []
    for t in (int(x) for x in args.ts.split(",")):
        latent = torch.randn((t, H, L), device=dev, dtype=torch.bfloat16) * 0.5
        g = torch.randn((t, H * V), device=dev, dtype=torch.bfloat16) * 2
        attn = torch.empty((t, H * V), device=dev, dtype=torch.bfloat16)
        out = torch.empty((t, H * V), device=dev, dtype=torch.bfloat16)

        def stock():
            torch.bmm(latent.transpose(0, 1), w_uv, out=attn.view(t, H, V).transpose(0, 1))
            return prefill_glue.eager_gate_sigmoid_mul(attn, g)

        arms = {"stock": stock}
        for name, k in kernels.items():
            arms[name] = (lambda k=k: k(latent, w_uv, g, out))
        ref = stock()
        for name, fn in list(arms.items()):
            if name == "stock":
                continue
            try:
                fn()
            except Exception as e:  # tactic outside the kernel's checks
                print("t=%d %s FAILED %s" % (t, name, str(e)[:200]), flush=True)
                arms[name] = None
                continue
            ok = torch.equal(out.view(torch.int16), ref.view(torch.int16))
            nd = int((out != ref).sum())
            print("t=%d %s bitwise=%s ndiff=%d" % (t, name, ok, nd), flush=True)
        arms = {k: v for k, v in arms.items() if v is not None}
        for _ in range(3):
            for fn in arms.values():
                fn()
        torch.cuda.synchronize()
        for name, fn in arms.items():
            st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            warm = []
            for _ in range(3):
                st.record()
                for _ in range(args.iters):
                    fn()
                en.record()
                en.synchronize()
                warm.append(st.elapsed_time(en) * 1e3 / args.iters)
            thr = []
            for _ in range(args.iters):
                flush.add_(1)
                s1, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s1.record()
                fn()
                e1.record()
                e1.synchronize()
                thr.append(s1.elapsed_time(e1) * 1e3)
            r = {
                "t": t, "arm": name, "warm_us": round(min(warm), 1),
                "thrash_us": round(statistics.median(thr), 1),
            }
            rows.append(r)
            print(json.dumps(r), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
