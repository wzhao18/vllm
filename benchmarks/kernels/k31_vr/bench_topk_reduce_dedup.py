# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbench + exactness check of the prefill MegaMoE dedup combine reduce
(TopkReduceDedup) at the deployed EP16 shape: per-sub-call tokens x world_size rank
slots x latent hidden BF16, ~10.35 valid ranks per token. Reports us and achieved
HBM GB/s (bytes read of valid slots + bytes written) per variant.

Usage: python bench_topk_reduce_dedup.py [--tokens 8192] [--world 16] [--hidden 3584]
"""

import argparse
import importlib
import json

import torch

MOD = "vllm.models.kimi_k3.nvidia.megamoe_rubin.src_gp.sources"


def _mask(tokens: int, world: int, mean_valid: float, gen: torch.Generator) -> torch.Tensor:
    # Each token hits each rank independently with probability mean_valid / world (>= 1 rank).
    p = mean_valid / world
    bits = torch.rand(tokens, world, generator=gen) < p
    bits[bits.sum(1) == 0, 0] = True
    w = (1 << torch.arange(world, dtype=torch.int64))
    return (bits.long() * w).sum(1).to(torch.int32)


def _reference(slots: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    world = slots.shape[1]
    acc = torch.zeros(slots.shape[0], slots.shape[2], dtype=torch.float32, device=slots.device)
    for r in range(world):
        sel = ((mask >> r) & 1).bool()
        acc[sel] += slots[sel, r].float()
    return acc.to(torch.bfloat16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--world", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=3584)
    ap.add_argument("--mean-valid", type=float, default=10.35)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--variants", default="base")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    import cuda.bindings.driver as cuda

    qd = importlib.import_module(f"{MOD}.quant_def")
    gen = torch.Generator().manual_seed(0)
    dev = torch.device("cuda")
    slots = (torch.randn(a.tokens, a.world, a.hidden, generator=gen) * 0.1).to(torch.bfloat16).to(dev)
    mask = _mask(a.tokens, a.world, a.mean_valid, gen).to(dev)
    ref = _reference(slots, mask)
    valid = int(sum(bin(int(m) & 0xFFFFFFFF).count("1") for m in mask.cpu()))
    nbytes = valid * a.hidden * 2 + a.tokens * a.hidden * 2 + a.tokens * 4
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    results = []
    for v in a.variants.split(","):
        mod = importlib.import_module(f"{MOD}.kernel_src.rubin.inference.mega.topk_reduce_dedup")
        cls = mod.TopkReduceDedup
        if v != "base":  # "h<elems per thread>t<threads per CTA>", same per-element summation order
            hpt, thr = (int(x) for x in v[1:].split("t"))
            cls = type(f"TopkReduceDedup_{v}", (cls,), {"_hidden_per_thread": hpt, "_threads": thr})
        red = cls(a.hidden, a.world, qd.CombineFormat.parse("bf16"))
        out = torch.empty(a.tokens, a.hidden, dtype=torch.bfloat16, device=dev)
        args = [from_dlpack(slots, assumed_align=32), from_dlpack(mask), from_dlpack(out, assumed_align=32)]
        fn = cute.compile(red, *args, stream)
        fn(*args, stream)
        torch.cuda.synchronize()
        exact = bool(torch.equal(out, ref))
        maxdiff = float((out.float() - ref.float()).abs().max())
        # cold-ish timing: flush L2 between iters with a 512 MiB write
        flush = torch.empty(512 << 20, dtype=torch.uint8, device=dev)
        ts = []
        for _ in range(a.iters):
            flush.zero_()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            fn(*args, stream)
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e) * 1e3)
        ts.sort()
        p50 = ts[len(ts) // 2]
        r = {"variant": v, "p50_us": round(p50, 2), "min_us": round(ts[0], 2), "GBps": round(nbytes / p50 / 1e3, 1),
             "bytes": nbytes, "valid_per_token": round(valid / a.tokens, 2), "exact": exact, "maxdiff": maxdiff}
        print(json.dumps(r), flush=True)
        results.append(r)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(results, f)


if __name__ == "__main__":
    main()
