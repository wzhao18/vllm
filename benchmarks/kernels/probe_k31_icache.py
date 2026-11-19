# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Probe: cost of executing straight-line code once (cold instruction fetch) vs again
(warm) per CTA, 512 threads, 1 or 2 CTAs per SM, for blocks of N independent integer ops.
Tells whether a kernel whose per-row chain runs long code once is instruction-fetch bound."""

import torch
from torch.utils.cpp_extension import load_inline

N_LIST = (256, 1024, 4096)

SRC = r"""
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>
#define MAD(x, y, i) asm volatile("mad.lo.u32 %0, %0, %1, %2;" : "+r"(x) : "r"(y), "r"((uint32_t)(i)));
#define OP4(i) MAD(a, b, i) MAD(b, c, i) MAD(c, d, i) MAD(d, a, i)
#define OP16(i) OP4(i) OP4(i + 1) OP4(i + 2) OP4(i + 3)
#define OP64(i) OP16(i) OP16(i + 4) OP16(i + 8) OP16(i + 12)
#define OP256(i) OP64(i) OP64(i + 16) OP64(i + 32) OP64(i + 48)
template <int N>
__device__ __forceinline__ void block(uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d) {
#pragma unroll
  for (int r = 0; r < N / 256; ++r) {
    asm volatile("" ::: "memory");
    OP256(r * 64 + 1)
  }
}
template <int N>
__global__ void __launch_bounds__(512, 2) icache(unsigned* out, long long* t) {
  uint32_t a = threadIdx.x, b = a + 1, c = a + 2, d = a + 3;
  __syncthreads();
  long long t0 = clock64();
  block<N>(a, b, c, d);          // first time: cold
  __syncthreads();
  long long t1 = clock64();
#pragma unroll 1
  for (int k = 0; k < 2; ++k) block<N>(a, b, c, d);  // a rolled loop over the same code: warm on k = 1
  __syncthreads();
  long long t2 = clock64();
  if ((a ^ b ^ c ^ d) == 0x12345u) out[0] = a;
  if (threadIdx.x == 0) { t[2 * blockIdx.x] = t1 - t0; t[2 * blockIdx.x + 1] = t2 - t1; }
}
__global__ void __launch_bounds__(512, 2) barriers(unsigned* out, long long* t, int n) {
  __shared__ uint32_t sm[1024];
  sm[threadIdx.x] = threadIdx.x;
  __syncthreads();
  long long t0 = clock64();
  for (int k = 0; k < n; ++k) __syncthreads();
  long long t1 = clock64();
  uint32_t a = 0;
  for (int k = 0; k < n; ++k) {  // write, barrier, dependent read of another warp's write
    sm[(threadIdx.x + k) & 511] = a + k;
    __syncthreads();
    a += sm[(threadIdx.x + 37 * k + 1) & 511];
  }
  long long t2 = clock64();
  for (int k = 0; k < n; ++k) a += atomicAdd(&sm[512 + (k & 7)], 1u);  // same-address SMEM atomics
  long long t3 = clock64();
  if (a == 0x12345u) out[0] = a;
  if (threadIdx.x == 0) { t[3 * blockIdx.x] = t1 - t0; t[3 * blockIdx.x + 1] = t2 - t1; t[3 * blockIdx.x + 2] = t3 - t2; }
}
void run_bar(int64_t n, int64_t ctas, torch::Tensor out, torch::Tensor t) {
  barriers<<<(unsigned)ctas, 512>>>((unsigned*)out.data_ptr(), (long long*)t.data_ptr(), (int)n);
}
void run(int64_t n, int64_t ctas, torch::Tensor out, torch::Tensor t) {
  auto o = (unsigned*)out.data_ptr();
  auto tt = (long long*)t.data_ptr();
  if (n == 256) icache<256><<<(unsigned)ctas, 512>>>(o, tt);
  else if (n == 1024) icache<1024><<<(unsigned)ctas, 512>>>(o, tt);
  else icache<4096><<<(unsigned)ctas, 512>>>(o, tt);
}
"""
ext = load_inline("probe_k31_icache", cpp_sources="void run(int64_t, int64_t, torch::Tensor, torch::Tensor); void run_bar(int64_t, int64_t, torch::Tensor, torch::Tensor);",
                  cuda_sources=SRC, functions=["run", "run_bar"],
                  extra_cuda_cflags=["-O3", "-gencode=arch=compute_107a,code=sm_107a"])
dev = torch.device("cuda", 0)
out = torch.zeros(8, dtype=torch.int32, device=dev)
for ctas in (16, 424):
    for n in N_LIST:
        t = torch.zeros(2 * ctas, dtype=torch.int64, device=dev)
        for _ in range(3):
            ext.run(n, ctas, out, t)
        torch.cuda.synchronize()
        cold = t[0::2].double().median().item()
        warm2 = t[1::2].double().median().item()
        # 4 ops per OP4 per thread -> n ops per thread, 16 warps per CTA
        print(f"ctas {ctas} n {n}: first pass {cold / 2.4e3:.2f} us ({cold / n:.1f} clk/op), "
              f"two more passes {warm2 / 2.4e3:.2f} us ({warm2 / (2 * n):.1f} clk/op)", flush=True)
for ctas in (16, 424):
    t = torch.zeros(3 * ctas, dtype=torch.int64, device=dev)
    for _ in range(3):
        ext.run_bar(200, ctas, out, t)
    torch.cuda.synchronize()
    b, bs, at = (t[i::3].double().median().item() / 200 for i in range(3))
    print(f"ctas {ctas}: __syncthreads {b:.0f} clk, write+barrier+read {bs:.0f} clk, "
          f"smem atomic same addr x512 thr {at:.0f} clk", flush=True)
