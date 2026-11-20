# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MoE latent up (7168 x 3584 BF16, beta-add of the shared output) on one Rubin GPU:
stock ``residual.addmm_(x, W^T)`` (cuBLAS) vs the sdec token-wide 2-CTA GEMM
(``cutedsl.gemm.wide``) writing FP32 split partials (``partials``: no split
exchange; the consumer sums them in split order and adds the residual), each after a
poll-like predecessor (one small CTA per SM, ``griddepcontrol.launch_dependents`` at
entry, carve-out 100) that spins ``window`` us.

Arms: ``addmm``; ``wide.s<S>`` (the GEMM alone: the consumer-fused form);
``wide.s<S>+sum`` (plus a torch sum of the partials and the residual: a separate
consumer kernel). Chain in a CUDA graph over rotating cold weight copies; reported =
(chain) - (host-only chain) per link, us. Checks the partial sums (+ residual)
against an FP64 reference."""

import argparse
import json
import math
import statistics

import torch
from torch.utils.cpp_extension import load_inline

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.wide import (
    PreparedWide,
    supports,
)

CUDA = r"""
#include <cstdint>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

__global__ void host_kernel(long long window_ns) {
  __shared__ int pad[256];
  if (threadIdx.x == 0) pad[0] = 0;
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  if (threadIdx.x == 0) {
    uint64_t t0, t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    do {
      __nanosleep(64);
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    } while (t < t0 + (uint64_t)window_ns);
  }
}

void host(int64_t window_ns, int64_t grid) {
  static bool set = false;
  if (!set) {
    cudaFuncSetAttribute(host_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    set = true;
  }
  host_kernel<<<(int)grid, 128, 0, c10::cuda::getCurrentCUDAStream()>>>(window_ns);
}
"""
CPP = "void host(int64_t window_ns, int64_t grid);"


def graph_time(fn, reps=8, samples=7):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(samples):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(reps):
            g.replay()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1e3 / reps)
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="32,64,96")
    ap.add_argument("--splits", default="2,3")
    ap.add_argument("--windows", default="0,4,8")
    ap.add_argument("--n", type=int, default=7168)
    ap.add_argument("--k", type=int, default=3584)
    args = ap.parse_args()
    ext = load_inline("dgemm_lup_host", CPP, cuda_sources=CUDA, functions=["host"],
                      extra_cuda_cflags=["-O3", "-gencode=arch=compute_100,code=compute_100"],
                      verbose=False)  # fmt: skip
    dev = torch.device("cuda")
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    torch.manual_seed(0)
    n, k = args.n, args.k
    copies = max(6, math.ceil(1.6e9 / (n * k * 2)))
    ws = [torch.randn(n, k, device=dev).mul_(0.02).to(torch.bfloat16)
          for _ in range(copies)]  # fmt: skip
    for rows in (int(r) for r in args.rows.split(",")):
        x = torch.randn(rows, k, device=dev).to(torch.bfloat16)
        res0 = torch.randn(rows, n, device=dev).to(torch.bfloat16)
        outs = [res0.clone() for _ in range(copies)]
        ref = res0.double() + x.double() @ ws[0].double().t()
        scale = ref.abs().max().item()
        chk = res0.clone().addmm_(x, ws[0].t())
        out = {"rows": rows, "addmm_err": (chk.double() - ref).abs().max().item() / scale}
        preps, parts = {}, {}
        for s in (int(v) for v in args.splits.split(",")):
            if not supports(rows, n, k, s, out_f32=True) or (n // 128) * s > sms:
                out[f"s{s}"] = "unsupported"
                continue
            preps[s] = [PreparedWide(w, {rows: s}, partials=True).prepare() for w in ws]
            parts[s] = [torch.full((s * rows, n), float("nan"), device=dev)
                        for _ in range(copies)]  # fmt: skip
            preps[s][0].run(x, parts[s][0])
            torch.cuda.synchronize()
            p = parts[s][0].view(s, rows, n)
            total = p[0].clone()
            for i in range(1, s):
                total += p[i]
            got = (total + res0.float()).bfloat16()
            out[f"s{s}_err"] = (got.double() - ref).abs().max().item() / scale
        for win in (float(w) for w in args.windows.split(",")):
            ns = int(win * 1000)

            def chain(kind, s=None, ns=ns):
                def fn():
                    for i in range(copies):
                        ext.host(ns, sms)
                        if kind == "addmm":
                            outs[i].addmm_(x, ws[i].t())
                        elif kind == "wide":
                            preps[s][i].run(x, parts[s][i])
                        elif kind == "wide+sum":
                            preps[s][i].run(x, parts[s][i])
                            torch.add(parts[s][i].view(s, rows, n).sum(0), outs[i],
                                      out=parts[s][i][:rows])  # fmt: skip
                return fn

            host = graph_time(chain(None)) / copies
            tag = f"W{win:g}"
            out[f"{tag}.host"] = round(host, 2)
            out[f"{tag}.addmm"] = round(graph_time(chain("addmm")) / copies - host, 2)
            for s in preps:
                out[f"{tag}.wide.s{s}"] = round(graph_time(chain("wide", s)) / copies - host, 2)
                out[f"{tag}.wide.s{s}+sum"] = round(
                    graph_time(chain("wide+sum", s)) / copies - host, 2
                )
        print(json.dumps(out), flush=True)
        del preps, parts
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
