# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 prefill glue microbench at the DEP16 F16K per-GPU shapes: router
top-k (sigmoid, top-16 of 896), routed latent RMSNorm [T, 3584] and the
shared-expert SiTU-and-mul + MXFP8 quant [T, 12288] -> [T, 6144].

    python benchmarks/kernels/bench_kimi_k3_prefill_elem.py --rows 16384
"""

import argparse
import json
import os

import torch

from vllm.models.kimi_k3.nvidia.ops import mxfp8_act, prefill_glue

CKPT = "/lustre/share/coreai_comparch_aarwlt/models/nvidia_kimi-k31-pro-nvfp4/hf-1d82032"


def bench(fn, iters=50, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()
    g.replay()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(5):
        s.record()
        g.replay()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1e3 / iters)
    return sorted(ts)[len(ts) // 2]


def bench_eager(fn, iters=50, warm=5):
    """Event timing of back-to-back eager launches (CuTe calls bind their own
    stream, so they are not graph-captured)."""
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(5):
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1e3 / iters)
    return sorted(ts)[len(ts) // 2]


def cfg():
    p = os.path.join(CKPT, "config.json")
    c = json.load(open(p)) if os.path.exists(p) else {}
    c = c.get("text_config", c)
    return {
        "beta": c.get("activation_situ_beta") or 1.0,
        "linear_beta": c.get("activation_situ_linear_beta"),
        "scale": c.get("routed_scaling_factor", 1.0),
        "renorm": c.get("moe_renormalize", True),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=16384)
    ap.add_argument("--only", default="topk,norm,act,dedup")
    a = ap.parse_args()
    T, dev = a.rows, "cuda"
    c = cfg()
    print("cfg", c)
    torch.manual_seed(0)
    only = a.only.split(",")
    res = {}
    if "topk" in only:
        E = 896
        for name, std, bstd in [("n1", 1.0, 0.0), ("n3b", 3.0, 0.02),
                                ("n12sat", 12.0, 0.0), ("emu", 3.0, None), ("emu4", 3.0, "emu4")]:
            sc = torch.randn(T, E, device=dev) * std
            if bstd == "emu4":  # skewed: 4 of every 56 active (> 32 candidates)
                bias = torch.full((E,), -100.0, device=dev)
                bias[torch.arange(E, device=dev) % 56 < 4] = 0.0
            elif bstd is None:
                bias = torch.full((E,), -100.0, device=dev)
                bias[torch.arange(E, device=dev) % 224 < 56] = 0.0  # step-bench EP16 emulation
            else:
                bias = torch.randn(E, device=dev) * bstd
            t = bench(lambda: prefill_glue.router_topk(sc, bias, 16, c["renorm"],
                                                       c["scale"]))
            res[f"topk_{name}"] = t
            print(f"router_topk {name:7s} T={T} {t:8.2f} us  "
                  f"{T * E * 4 / t / 1e6:6.2f} TB/s")
    if "norm" in only:
        x = torch.randn(T, 3584, device=dev, dtype=torch.bfloat16)
        w = torch.randn(3584, device=dev, dtype=torch.bfloat16)
        t = bench(lambda: prefill_glue.rms_norm(x, w, 1e-6))
        res["rms_norm"] = t
        print(f"rms_norm [T,3584] {t:8.2f} us {2 * x.numel() * 2 / t / 1e6:6.2f} TB/s")
    if "act" in only:
        x = torch.randn(T, 12288, device=dev, dtype=torch.bfloat16) * 2
        t = bench(lambda: mxfp8_act.situ_and_mul_quantize_auto(
            x, c["beta"], c["linear_beta"]))
        res["situ_act"] = t
        by = x.numel() * 2 + x.numel() // 2 + x.numel() // 64
        print(f"situ_act [T,12288] {t:8.2f} us {by / t / 1e6:6.2f} TB/s")
    if "dedup" in only:
        res.update(bench_dedup(T))
    print("RESULT", json.dumps(res))


def _cute(t):
    import cutlass.torch as cutlass_torch

    return cutlass_torch.from_dlpack(t, assumed_align=16).mark_layout_dynamic(
        leading_dim=t.dim() - 1
    )


def bench_dedup(T, hidden=3584, world=16):
    """MegaMoE dedup combine reduce at DEP16: plain + separate rms_norm vs the
    reduce with the fused RMSNorm."""
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    from vllm.models.kimi_k3.nvidia.megamoe_rubin.src_gp.sources.kernel_src.rubin.inference.mega.topk_reduce_dedup import (  # noqa: E501
        TopkReduceDedup,
    )
    from vllm.models.kimi_k3.nvidia.megamoe_rubin.src_gp.sources.quant_def import (
        CombineFormat,
    )

    dev = "cuda"
    slots = torch.randn((T, world, hidden), device=dev).to(torch.bfloat16)
    ids = torch.randint(0, world, (T, 16), device=dev)
    mask = torch.zeros(T, dtype=torch.int32, device=dev)
    for k in range(16):
        mask |= (1 << ids[:, k]).to(torch.int32)
    w = torch.randn(hidden, device=dev).to(torch.bfloat16)
    out = torch.empty((T, hidden), dtype=torch.bfloat16, device=dev)
    op = TopkReduceDedup(hidden, world, CombineFormat.parse("bf16"))
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    a = [_cute(slots), _cute(mask), _cute(out), stream]
    plain = cute.compile(op, *a)
    nk = dict(rmsnorm_weight=_cute(w), rmsnorm_eps=cutlass.Float32(1e-6))
    fused = cute.compile(op, *a, **nk)
    r = {}
    r["dedup_reduce"] = bench_eager(lambda: plain(*a))
    r["dedup_reduce_rmsnorm"] = bench_eager(lambda: fused(*a, **nk))
    r["glue_rms_norm"] = bench_eager(lambda: prefill_glue.rms_norm(out, w, 1e-6))
    print("dedup T=%d world=%d ranks/token=%.1f: reduce %.2f + rms_norm %.2f = %.2f us;"
          " fused %.2f us" % (T, world, sum(((mask >> i) & 1).float().mean().item() for i in range(world)), r["dedup_reduce"],
          r["glue_rms_norm"], r["dedup_reduce"] + r["glue_rms_norm"],
          r["dedup_reduce_rmsnorm"]))
    return r


if __name__ == "__main__":
    main()
