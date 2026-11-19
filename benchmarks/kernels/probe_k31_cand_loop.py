# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Probe: cost of topk_v2's candidate pass per row (1 GPU, 512 threads per row), by stage:
0 loads only, 1 + packed bound / NaN test, 2 + per-key count, 3 + warp scan, atomic, writes."""

import torch
from torch.utils.cpp_extension import load_inline

SRC = r"""
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>
__device__ __forceinline__ uint32_t keys2(uint32_t w) {
  const uint32_t neg = (w >> 15) & 0x00010001u;
  const uint32_t k = w ^ (neg * 0x7FFFu + 0x80008000u);
  const uint32_t nan = (((w & 0x7FFF7FFFu) + 0x007F007Fu) >> 15) & 0x00010001u;
  return k & ~(nan * 0xFFFFu);
}
__device__ __forceinline__ uint32_t mkb(const uint4& r) {
  uint32_t a, b, m;
  asm("max.bf16x2 %0, %1, %2;" : "=r"(a) : "r"(r.x), "r"(r.y));
  asm("max.bf16x2 %0, %1, %2;" : "=r"(b) : "r"(r.z), "r"(r.w));
  asm("max.bf16x2 %0, %1, %2;" : "=r"(m) : "r"(a), "r"(b));
  const uint32_t k = keys2(m);
  return ::max(k & 0xFFFFu, k >> 16) + 1u;
}
__device__ __forceinline__ bool hasnan(const uint4& r) {
  uint32_t a, b, m;
  asm("max.NaN.bf16x2 %0, %1, %2;" : "=r"(a) : "r"(r.x), "r"(r.y));
  asm("max.NaN.bf16x2 %0, %1, %2;" : "=r"(b) : "r"(r.z), "r"(r.w));
  asm("max.NaN.bf16x2 %0, %1, %2;" : "=r"(m) : "r"(a), "r"(b));
  return (((m & 0x7FFF7FFFu) + 0x007F007Fu) & 0x80008000u) != 0u;
}
template <int STAGE, int RG>
__global__ void __launch_bounds__(512, 2) cand(const uint4* x, int64_t s16, int ng, int th,
                                             unsigned* out, long long* t) {
  __shared__ uint32_t cpos[3072];
  __shared__ uint16_t ckey[3072];
  __shared__ int s_cnt;
  const int tid = threadIdx.x;
  const uint4* r = x + blockIdx.x * s16;
  if (tid == 0) s_cnt = 0;
  __syncthreads();
  long long t0 = clock64();
  uint32_t acc = 0, nv = 0;
  for (int g0 = 0; g0 < ng; g0 += RG * 512) {
    uint4 rb[RG];
#pragma unroll
    for (int u = 0; u < RG; ++u) {
      const int g = g0 + u * 512 + tid;
      rb[u] = g < ng ? __ldg(r + g) : make_uint4(~0u, ~0u, ~0u, ~0u);
    }
    if (STAGE == 0) {
#pragma unroll
      for (int u = 0; u < RG; ++u) acc ^= rb[u].x ^ rb[u].w;
      continue;
    }
    uint32_t cnt = 0, hit = 0;
#pragma unroll
    for (int u = 0; u < RG; ++u) {
      nv += hasnan(rb[u]) ? 0 : 8;
      if ((int)mkb(rb[u]) > th) {
        hit |= 1u << u;
        if (STAGE >= 2) {
          uint32_t w[4] = {keys2(rb[u].x), keys2(rb[u].y), keys2(rb[u].z), keys2(rb[u].w)};
#pragma unroll
          for (int j = 0; j < 8; ++j) cnt += ((j & 1) ? (w[j >> 1] >> 16) : (w[j >> 1] & 0xFFFFu)) >= (uint32_t)th;
        }
      }
    }
    acc += hit + cnt;
    if (STAGE < 3) continue;
    uint32_t incl = cnt;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const uint32_t y = __shfl_up_sync(0xffffffffu, incl, o);
      if ((tid & 31) >= o) incl += y;
    }
    int base = 0;
    if ((tid & 31) == 31 && incl) base = atomicAdd(&s_cnt, (int)incl);
    int c = __shfl_sync(0xffffffffu, base, 31) + (int)(incl - cnt);
    if (cnt) {
#pragma unroll
      for (int u = 0; u < RG; ++u) {
        if (!((hit >> u) & 1u)) continue;
        uint32_t w[4] = {keys2(rb[u].x), keys2(rb[u].y), keys2(rb[u].z), keys2(rb[u].w)};
        const int p0 = 8 * (g0 + u * 512 + tid);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const uint32_t k = (j & 1) ? (w[j >> 1] >> 16) : (w[j >> 1] & 0xFFFFu);
          if (k >= (uint32_t)th) {
            if (c < 3072) { cpos[c] = p0 + j; ckey[c] = (uint16_t)k; }
            ++c;
          }
        }
      }
    }
  }
  __syncthreads();
  long long t1 = clock64();
  if (acc == 0x12345u) out[0] = acc + nv;
  if (tid == 0) { t[blockIdx.x] = t1 - t0; out[1 + blockIdx.x % 4] = s_cnt; }
}
void run(torch::Tensor x, int64_t rows, int64_t ng, int64_t th, int64_t stage, int64_t rg,
         torch::Tensor out, torch::Tensor t) {
  const uint4* p = reinterpret_cast<const uint4*>(x.data_ptr());
  const int64_t s16 = x.stride(0) * 2 / 16;
  auto o = (unsigned*)out.data_ptr();
  auto tt = (long long*)t.data_ptr();
#define L(S, R) cand<S, R><<<(unsigned)rows, 512>>>(p, s16, (int)ng, (int)th, o, tt)
  if (rg == 4) { if (stage == 0) L(0, 4); else if (stage == 1) L(1, 4); else if (stage == 2) L(2, 4); else L(3, 4); }
  else { if (stage == 0) L(0, 8); else if (stage == 1) L(1, 8); else if (stage == 2) L(2, 8); else L(3, 8); }
}
"""

ext = load_inline("probe_k31_cand_loop", cpp_sources="void run(torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, torch::Tensor, torch::Tensor);",
                  cuda_sources=SRC, functions=["run"],
                  extra_cuda_cflags=["-O3", "-gencode=arch=compute_107a,code=sm_107a"])
dev = torch.device("cuda", 0)
L = 33750
ng = L // 8
for rows in (16, 384):
    x = torch.relu(torch.randn(rows, 262144, device=dev) + 0.3).mul(2).to(torch.bfloat16)
    keys = x[:, :L].float()
    thv = keys.sort(dim=1, descending=True).values[:, 1500].min()  # ~1500 candidates per row
    thk = int((torch.tensor([float(thv)]).to(torch.bfloat16).view(torch.int16).int() & 0xFFFF) | 0x8000)
    out = torch.zeros(8, dtype=torch.int32, device=dev)
    for rg in (4, 8):
        for stage in range(4):
            t = torch.zeros(rows, dtype=torch.int64, device=dev)
            for _ in range(3):
                ext.run(x, rows, ng, thk, stage, rg, out, t)
            torch.cuda.synchronize()
            cyc = t.double()
            print(f"rows {rows} RG {rg} stage {stage}: {float(cyc.median()) / 2.4e3:.2f} us median "
                  f"{float(cyc.max()) / 2.4e3:.2f} max, cand {int(out[1])}", flush=True)
