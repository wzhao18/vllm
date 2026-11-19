# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Probe: time for one CTA (or a few) to pull one 67.5 KB logits row into registers / SMEM
(1 GPU). Rows of 33750 bf16 keys at a 512 KB row stride, L2-hot (re-read in a loop)."""

import torch
from torch.utils.cpp_extension import load_inline

SRC = r"""
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>
template <int NU>
__global__ void __launch_bounds__(256) ld_regs(const uint4* x, int64_t stride16, int n16, int split,
                                              unsigned* out, long long* t) {
  const int row = blockIdx.x / split, part = blockIdx.x % split;
  const uint4* r = x + row * stride16;
  const int per = (n16 + split - 1) / split, b = part * per, e = min(n16, b + per);
  long long t0 = clock64();
  uint4 v[NU];
#pragma unroll
  for (int u = 0; u < NU; ++u) {
    const int g = b + u * 256 + threadIdx.x;
    v[u] = g < e ? __ldg(r + g) : make_uint4(0, 0, 0, 0);
  }
  unsigned s = 0;
#pragma unroll
  for (int u = 0; u < NU; ++u) s += v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
  __syncthreads();
  long long t1 = clock64();
  if (s == 0x12345678u) out[0] = s;
  if (threadIdx.x == 0) t[blockIdx.x] = t1 - t0;
}
__global__ void __launch_bounds__(256) ld_tma(const uint4* x, int64_t stride16, int n16, int split,
                                             unsigned* out, long long* t) {
  extern __shared__ __align__(128) uint4 sm[];
  __shared__ __align__(8) uint64_t bar;
  const int row = blockIdx.x / split, part = blockIdx.x % split;
  const uint4* r = x + row * stride16;
  const int per = (n16 + split - 1) / split, b = part * per, e = min(n16, b + per);
  long long t0 = clock64();
  uint32_t ba = (uint32_t)__cvta_generic_to_shared(&bar);
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(ba));
    asm volatile("fence.mbarrier_init.release.cluster;");
    asm volatile("fence.proxy.async.shared::cta;");
    const uint32_t bytes = (uint32_t)(e - b) * 16u;
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(ba), "r"(bytes));
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 ::"r"((uint32_t)__cvta_generic_to_shared(sm)), "l"(r + b), "r"(bytes), "r"(ba) : "memory");
  }
  __syncthreads();
  asm volatile("{\n .reg .pred P1;\n W: mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], 0;\n @P1 bra D;\n bra W;\n D:\n}" ::"r"(ba));
  unsigned s = 0;
  for (int g = threadIdx.x; g < e - b; g += 256) { uint4 v = sm[g]; s += v.x ^ v.w; }
  __syncthreads();
  long long t1 = clock64();
  if (s == 0x12345678u) out[0] = s;
  if (threadIdx.x == 0) t[blockIdx.x] = t1 - t0;
}
void run(torch::Tensor x, int64_t rows, int64_t n16, int64_t split, int64_t kind, torch::Tensor out,
         torch::Tensor t) {
  const uint4* p = reinterpret_cast<const uint4*>(x.data_ptr());
  const int64_t s16 = x.stride(0) * 2 / 16;
  dim3 g((unsigned)(rows * split));
  if (kind == 0) {
    if (split == 1) ld_regs<20><<<g, 256>>>(p, s16, (int)n16, (int)split, (unsigned*)out.data_ptr(), (long long*)t.data_ptr());
    else ld_regs<5><<<g, 256>>>(p, s16, (int)n16, (int)split, (unsigned*)out.data_ptr(), (long long*)t.data_ptr());
  } else {
    const size_t smem = (size_t)((n16 + split - 1) / split) * 16;
    cudaFuncSetAttribute(ld_tma, cudaFuncAttributeMaxDynamicSharedMemorySize, 200 * 1024);
    ld_tma<<<g, 256, smem>>>(p, s16, (int)n16, (int)split, (unsigned*)out.data_ptr(), (long long*)t.data_ptr());
  }
}
"""

ext = load_inline("probe_k31_row_load", cpp_sources="void run(torch::Tensor, int64_t, int64_t, int64_t, int64_t, torch::Tensor, torch::Tensor);",
                  cuda_sources=SRC, functions=["run"],
                  extra_cuda_cflags=["-O3", "-gencode=arch=compute_107a,code=sm_107a"])
dev = torch.device("cuda", 0)
L = 33750
n16 = L * 2 // 16
for rows in (16, 128, 384):
    x = torch.randn(rows, 262144, device=dev).to(torch.bfloat16)
    out = torch.zeros(4, dtype=torch.int32, device=dev)
    for kind, split in ((0, 1), (0, 4), (1, 1), (1, 4)):
        t = torch.zeros(rows * split, dtype=torch.int64, device=dev)
        for _ in range(3):
            ext.run(x, rows, n16, split, kind, out, t)
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(20):
            ext.run(x, rows, n16, split, kind, out, t)
        e1.record()
        torch.cuda.synchronize()
        cyc = t.double()
        print(f"rows {rows} {'regs' if kind == 0 else 'tma '} split {split}: kernel {e0.elapsed_time(e1) * 50:.2f} us/launch, "
              f"in-CTA load {float(cyc.median()) / 2.4e3:.2f} us median, {float(cyc.max()) / 2.4e3:.2f} max", flush=True)
