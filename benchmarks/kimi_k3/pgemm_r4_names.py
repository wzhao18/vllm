#!/usr/bin/env python3
"""Kernel names (tile form) of the pinned prefill BF16 / MXFP8 cuBLASLt tactics
(specialized_prefill.gemm_tactics PINS) and of cuBLASLt's heuristic pick."""

import json

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile


def names(fn):
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return sorted({e.name[:110] for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA})


def main():
    from vllm.models.kimi_k3.nvidia.specialized_prefill import gemm_tactics as gt

    m = 16384
    for (dtype, n, k) in sorted(gt.PINS):
        row = {"dtype": dtype, "n": n, "k": k}
        if dtype != "bf16":
            row["skip"] = "mxfp8"
            print(json.dumps(row), flush=True)
            continue
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
        p = gt.plan(dtype, n, k)
        row["pinned"] = names(lambda: gt.mm(p, x, w)) if p is not None else None
        row["heuristic"] = names(lambda: F.linear(x, w))
        for fn in ((lambda: gt.mm(p, x, w)) if p is not None else None, lambda: F.linear(x, w)):
            if fn is None:
                continue
            torch.cuda.cudart().cudaProfilerStart()
            fn()
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
        print(json.dumps(row), flush=True)
        del x, w
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
