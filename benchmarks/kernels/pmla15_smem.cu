// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// P-MLA-2CTA round 5: does shared-memory WRITE traffic slow the h8 tensor-pipe mix?
// All SMs, 1 CTA/SM. Warps 0-1 issue the EXACT h8 subtile mix (pmla11 mode 0 from one warp,
// or mode 8: QK on warp 0 + PV on warp 1) on 2 KV stages; warps 4-11 (8 warps, as h8's
// copiers) write a separate 3-stage ring (below the 227 KiB line) until the MMA warps finish:
//   w 0 none | 1 cp.async 16 B random 576 B rows, h8 mapping (4 lanes/row, 64 B per request)
//   | 2 cp.async line-aligned (8 lanes per 128 B line, 5 requests/row) | 3 st.shared.v4 from
//   registers (same bytes/addresses, no L2) | 4 cp.async random rows from 64 L2-hot rows.
// Per-warp cp.async in flight bounded by cp.async.wait_group (depth D stages of 8 rows).
// Prints MMA clk/subtile and the writers' B/clk/SM (median over CTAs).
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("CUDA %s at %d\n", cudaGetErrorString(e), __LINE__); return 1; } } while (0)

static constexpr uint64_t SW128 = (uint64_t(1024 >> 4) << 32) | (1ull << 46) | (2ull << 61);
static constexpr uint64_t SW64 = (uint64_t(512 >> 4) << 32) | (1ull << 46) | (4ull << 61);
static constexpr uint32_t STAGE = 36864, NSTG = 2, OFF_QS = NSTG * STAGE, OFF_P = OFF_QS + 8192;
static constexpr uint32_t OFF_W = OFF_P + 2 * 6144, NWS = 3;
static constexpr uint32_t SMEM = OFF_W + NWS * STAGE + 1024;
static constexpr uint32_t C_Q = 0, C_S = 128, C_O = 192;
__device__ __forceinline__ uint32_t sa(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void wait_par(uint32_t bar, uint32_t par) {
  asm volatile("{.reg .pred P; W%=: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1; "
               "@!P bra W%=;}" :: "r"(bar), "r"(par) : "memory");
}
__device__ __forceinline__ void commit(uint32_t bar) {
  asm volatile("{.reg .pred e; elect.sync _|e, 0xffffffff; @e tcgen05.commit.cta_group::1."
               "mbarrier::arrive::one.shared::cluster.b64 [%0];}" :: "r"(bar) : "memory");
}
#define MMA_TS(d, a, b, i, acc) asm volatile("{.reg .pred p, e; elect.sync _|e, 0xffffffff; " \
  "setp.ne.b32 p, %4, 0; @e tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [%1], %2, %3, p;}" \
  :: "r"(d), "r"(a), "l"(b), "r"(i), "r"(acc) : "memory")
#define MMA_SS(d, a, b, i, acc) asm volatile("{.reg .pred p, e; elect.sync _|e, 0xffffffff; " \
  "setp.ne.b32 p, %4, 0; @e tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;}" \
  :: "r"(d), "l"(a), "l"(b), "r"(i), "r"(acc) : "memory")
#define MMA_CB(op, d, a, b, i, acc) asm volatile("{.reg .pred p, e; elect.sync _|e, 0xffffffff; " \
  "setp.ne.b32 p, %4, 0; @e tcgen05.mma.cta_group::1.kind::f8f6f4.collector::b::" op \
  " [%0], %1, %2, %3, p;}" :: "r"(d), "l"(a), "l"(b), "r"(i), "r"(acc) : "memory")
__device__ __forceinline__ void qk(uint32_t t, uint64_t bd0, uint64_t aq, uint32_t iqk) {
#pragma unroll
  for (int j = 0; j < 4; ++j)
#pragma unroll
    for (int k2 = 0; k2 < 2; ++k2)
      MMA_TS(t + C_S, t + C_Q + 32 * j + 16 * k2, bd0 + j * 512 + k2 * 4, iqk, j + k2 > 0 ? 1u : 0u);
  MMA_SS(t + C_S, aq, (bd0 + 4 * 512) ^ (SW128 ^ SW64), iqk, 1);
}
__device__ __forceinline__ void pv(uint32_t t, uint64_t ad0, uint64_t pd, uint32_t ipv) {
  MMA_CB("fill", t + C_O, ad0, pd, ipv, 1);
  MMA_CB("use", t + C_O + 96, ad0 + 512, pd, ipv, 1);
  MMA_CB("use", t + C_O + 192, ad0 + 1024, pd, ipv, 1);
  MMA_CB("lastuse", t + C_O + 288, ad0 + 1536, pd, ipv, 1);
}
__device__ __forceinline__ void cp16(uint32_t dst, const void* src, uint32_t n) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst), "l"(src), "r"(n)
               : "memory");
}
__device__ __forceinline__ void sts16(uint32_t dst, uint32_t v) {
  asm volatile("st.shared.v4.b32 [%0], {%1, %1, %1, %1};" :: "r"(dst), "r"(v) : "memory");
}
// 16 B chunk ch (0..35) of a row into stage row rs (SW128 latent atoms + SW64 RoPE atom)
__device__ __forceinline__ uint32_t dst_of(uint32_t st, uint32_t rs, uint32_t ch) {
  if (ch >= 32) return st + 4 * 8192 + rs * 64 + (((ch - 32) ^ ((rs >> 1) & 3)) << 4);
  return st + ((ch >> 3) << 13) + rs * 128 + (((ch & 7) ^ (rs & 7)) << 4);
}
__device__ __forceinline__ void wait_group(int d) {
  switch (d) {
    case 0: asm volatile("cp.async.wait_group 0;" ::: "memory"); break;
    case 1: asm volatile("cp.async.wait_group 1;" ::: "memory"); break;
    case 2: asm volatile("cp.async.wait_group 2;" ::: "memory"); break;
    case 3: asm volatile("cp.async.wait_group 3;" ::: "memory"); break;
    default: asm volatile("cp.async.wait_group 4;" ::: "memory"); break;
  }
}
extern "C" __global__ void __launch_bounds__(384, 1)
k(int mmode, int nsub, int wmode, int depth, const uint8_t* __restrict__ kv,
  const int* __restrict__ idx, int nidx, long long* out) {
  extern __shared__ __align__(1024) unsigned char dyn[];
  __shared__ uint64_t bars[8];
  __shared__ uint32_t tslot;
  __shared__ volatile int stop;
  unsigned char* sm = (unsigned char*)(((uintptr_t)dyn + 1023) & ~(uintptr_t)1023);
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  uint32_t* w = (uint32_t*)sm;
  for (uint32_t i = tid; i < OFF_W / 4; i += blockDim.x) {
    uint32_t x = (i + 1) * 2654435761u ^ (blockIdx.x * 40503u);
    w[i] = x & 0x3f3f3f3fu;  // finite E4M3 bytes
  }
  if (warp == 0) {
    asm volatile("tcgen05.alloc.exclusive.cta_group::1.sync.aligned.shared::cta.b32 [%0], 576; "
                 "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"
                 :: "r"(sa(&tslot)) : "memory");
  }
  if (tid == 0) {
    stop = 0;
    for (int i = 0; i < 8; ++i)
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(sa(&bars[i])) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  const uint32_t t = tslot, b0 = sa(sm);
  const uint32_t iqk = (1u << 4) | (8u << 17) | (8u << 24) | (1u << 29);
  const uint32_t ipv = (1u << 4) | (12u << 17) | (8u << 24) | (1u << 29) | (1u << 15);
  const uint64_t aq = SW64 | ((b0 + OFF_QS) >> 4);
  long long c0 = clock64();
  long long wbytes = 0;
  if (warp < 2) {
    if (mmode >= 0 && (warp == 0 || mmode == 8)) {
      const uint32_t mb = sa(&bars[2 * warp]);
      for (int i = 0; i < nsub; ++i) {
        if (i >= 2) wait_par(mb + 8 * (i & 1), ((i >> 1) - 1) & 1);
        const uint32_t st = b0 + (i % NSTG) * STAGE;
        const uint64_t bd0 = SW128 | (st >> 4);
        const uint64_t pd = SW64 | ((b0 + OFF_P + (i & 1) * 6144) >> 4);
        if (warp == 0) qk(t, bd0, aq, iqk);
        if (mmode != 8 || warp == 1) pv(t, bd0, pd, ipv);
        commit(mb + 8 * (i & 1));
      }
      for (int i = nsub > 2 ? nsub - 2 : 0; i < nsub; ++i) wait_par(mb + 8 * (i & 1), (i >> 1) & 1);
    }
  } else if (warp >= 4 && wmode > 0) {
    const int pw = warp - 4;
    const long long cap = mmode >= 0 ? (1ll << 40) : (long long)nsub;  // writer-only: nsub stages
    const uint32_t r = pw * 8 + lane / 4, ch = lane % 4;
    for (long long g = 0; g < cap; ++g) {
      if (mmode >= 0 && stop) break;
      const uint32_t stg = b0 + OFF_W + (uint32_t)(g % NWS) * STAGE;
      const long long base = (long long)blockIdx.x * 7919 + g * 64;
      if (wmode == 2) {
        const uint32_t a = lane / 8, c8 = lane % 8;
        for (int j = 0; j < 8; ++j) {
          const uint32_t rr = pw * 8 + 4 * (j & 1) + a;
          const int rc = idx[(base + rr) % nidx];
          const uint32_t c = ((uint32_t)(rc & 1) << 2) + 8 * (j >> 1) + c8;
          cp16(dst_of(stg, rr, c), kv + (long long)rc * 576 + c * 16, 16);
        }
        const int rc = idx[(base + r) % nidx];
        const uint32_t c = ((1u - (uint32_t)(rc & 1)) << 5) + ch;
        cp16(dst_of(stg, r, c), kv + (long long)rc * 576 + c * 16, 16);
      } else {
        int rc = idx[(base + r) % nidx];
        if (wmode == 4) rc &= 63;
        const uint8_t* src = kv + (long long)rc * 576;
        for (int j = 0; j < 9; ++j) {
          const uint32_t c = j < 8 ? ch + 4 * j : 32 + ch;
          if (wmode == 3) sts16(dst_of(stg, r, c), (uint32_t)rc + j);
          else cp16(dst_of(stg, r, c), src + c * 16, 16);
        }
      }
      if (wmode != 3) {
        asm volatile("cp.async.commit_group;" ::: "memory");
        wait_group(depth - 1);
      }
      wbytes += 4608;
    }
    if (wmode != 3) asm volatile("cp.async.wait_group 0;" ::: "memory");
  }
  long long c1 = clock64();
  if (warp == 0 && lane == 0) stop = 1;
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  if (warp == 0 && lane == 0) out[blockIdx.x * 3] = c1 - c0;
  if (warp >= 4 && lane == 0) atomicAdd((unsigned long long*)&out[blockIdx.x * 3 + 1], wbytes);
  if (warp == 4 && lane == 0) out[blockIdx.x * 3 + 2] = c1 - c0;
  if (warp == 0)
    asm volatile("tcgen05.dealloc.exclusive.cta_group::1.sync.aligned.b32 %0, 576;" :: "r"(t) : "memory");
}

int main(int argc, char** argv) {
  int sms = 0;
  CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
  CK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM + 1024));
  const int nsub = argc > 1 ? atoi(argv[1]) : 2048;
  const long long nrows = 78 * 1024;
  uint8_t* kv;
  int* idx;
  long long* d;
  const int nidx = 1 << 20;
  CK(cudaMalloc(&kv, nrows * 576));
  CK(cudaMemset(kv, 0x31, nrows * 576));
  std::vector<int> hi(nidx);
  srand(5);
  for (int i = 0; i < nidx; ++i) hi[i] = (int)(((long long)rand() * 7919 + rand()) % nrows);
  CK(cudaMalloc(&idx, nidx * sizeof(int)));
  CK(cudaMemcpy(idx, hi.data(), nidx * sizeof(int), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&d, sms * 3 * sizeof(long long)));
  printf("sms %d nsub %d SMEM %u\n", sms, nsub, SMEM);
  const char* wn[] = {"none", "cp.async h8 map", "cp.async line-aligned", "st.shared.v4",
                      "cp.async L2-hot rows"};
  struct Cfg { int mm, w, dep; };
  std::vector<Cfg> cfgs;
  for (int mm : {0, 8}) {
    cfgs.push_back({mm, 0, 1});
    for (int w : {1, 2, 3, 4}) for (int dep : {2, 4}) cfgs.push_back({mm, w, dep});
  }
  for (int w : {1, 2, 4}) for (int dep : {2, 4}) cfgs.push_back({-1, w, dep});
  cfgs.push_back({-1, 3, 1});
  for (int rep = 0; rep < 2; ++rep)
    for (const Cfg& c : cfgs) {
      CK(cudaMemset(d, 0, sms * 3 * sizeof(long long)));
      k<<<sms, 384, SMEM + 1024>>>(c.mm, c.mm >= 0 ? nsub : nsub / 4, c.w, c.dep, kv, idx, nidx, d);
      CK(cudaDeviceSynchronize());
      std::vector<long long> h(sms * 3);
      CK(cudaMemcpy(h.data(), d, h.size() * sizeof(long long), cudaMemcpyDeviceToHost));
      std::vector<double> mclk, wbpc;
      for (int s = 0; s < sms; ++s) {
        mclk.push_back((double)h[3 * s] / nsub);
        const long long wc = c.mm >= 0 ? h[3 * s] : h[3 * s + 2];
        wbpc.push_back(wc > 0 ? (double)h[3 * s + 1] / wc : 0.0);
      }
      std::sort(mclk.begin(), mclk.end());
      std::sort(wbpc.begin(), wbpc.end());
      printf("rep %d mma %2d writers %-22s depth %d: MMA clk/subtile %7.1f | writers %5.1f B/clk/SM\n",
             rep, c.mm, wn[c.w], c.dep, c.mm >= 0 ? mclk[sms / 2] : 0.0, wbpc[sms / 2]);
    }
  return 0;
}
