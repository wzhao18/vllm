// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// P-MLA-2CTA round 3: the EXACT h8 tensor-pipe mix per 64-key subtile, standalone (no
// gather, no softmax), every SM at once: QK = 8 x TS f8f6f4 M128 N64 K64 (Q in TMEM, K
// SW128 B 4 KiB) + 1 x SS RoPE (A 8 KiB SW64, B 4 KiB); PV = 4 x M128 N96 K64 (A = V^T
// MN-major SW128 8 KiB from the KV stage, B = P^T SW64 6 KiB, collector::b fill/use/lastuse).
// Modes: 0 exact | 1 PV without collector | 2 PV A K-major | 3 QK only | 4 PV only
// | 5 QK latent only (no RoPE) | 6 PV only, no collector | 7 PV only, A K-major
// | 8 QK (warp 0) and PV (warp 1) issued by two free-running warps (h8 issue split)
// | 9 exact, but the RoPE MMA before the latent ones.
// Prints clk per subtile (median over CTAs) and us per F16K DSA layer (2473 subtiles/SM).
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("CUDA %s at %d\n", cudaGetErrorString(e), __LINE__); return 1; } } while (0)

static constexpr uint64_t SW128 = (uint64_t(1024 >> 4) << 32) | (1ull << 46) | (2ull << 61);
static constexpr uint64_t SW64 = (uint64_t(512 >> 4) << 32) | (1ull << 46) | (4ull << 61);
static constexpr uint32_t STAGE = 36864, NSTG = 2, OFF_QS = NSTG * STAGE, OFF_P = OFF_QS + 8192;
static constexpr uint32_t SMEM = OFF_P + 2 * 6144 + 1024;
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

__device__ __forceinline__ void qk(int mode, uint32_t t, uint64_t bd0, uint64_t aq, uint32_t iqk) {
  if (mode == 9) MMA_SS(t + C_S, aq, (bd0 + 4 * 512) ^ (SW128 ^ SW64), iqk, 0);
#pragma unroll
  for (int j = 0; j < 4; ++j)
#pragma unroll
    for (int k2 = 0; k2 < 2; ++k2)
      MMA_TS(t + C_S, t + C_Q + 32 * j + 16 * k2, bd0 + j * 512 + k2 * 4, iqk,
             (mode == 9 || j + k2 > 0) ? 1u : 0u);
  if (mode != 5 && mode != 9) MMA_SS(t + C_S, aq, (bd0 + 4 * 512) ^ (SW128 ^ SW64), iqk, 1);
}

__device__ __forceinline__ void pv(int mode, uint32_t t, uint64_t ad0, uint64_t pd, uint32_t ipv) {
  if (mode == 2 || mode == 7) ipv &= ~(1u << 15);
  if (mode == 1 || mode == 6) {
#pragma unroll
    for (int b = 0; b < 4; ++b) MMA_SS(t + C_O + 96 * b, ad0 + b * 512, pd, ipv, 1);
  } else {
    MMA_CB("fill", t + C_O, ad0, pd, ipv, 1);
    MMA_CB("use", t + C_O + 96, ad0 + 512, pd, ipv, 1);
    MMA_CB("use", t + C_O + 192, ad0 + 1024, pd, ipv, 1);
    MMA_CB("lastuse", t + C_O + 288, ad0 + 1536, pd, ipv, 1);
  }
}

// 10: 8 TS N64 into 4 independent accumulators (2 chained each) | 11: 8 TS N64, 8 independent
// accumulators (alias cols) | 12: interleave latent QK j with PV block j, then RoPE
// | 13: 4 x TS N128 (= 64 keys of N64 work: half the MMAs) chained | 14: 2 x TS N256 chained
// | 15: 8 TS N64 chained, but every other MMA has acc = 0 (no RAW on D)
__device__ __forceinline__ void lat(int mode, uint32_t t, uint64_t bd0, uint64_t aq, uint64_t pd,
                                    uint32_t iqk, uint32_t ipv) {
  if (mode == 10 || mode == 11 || mode == 15) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      uint32_t d = mode == 10 ? C_S + 64 * (j & 3) : mode == 11 ? C_S + 64 * (j % 7) : C_S;
      MMA_TS(t + d, t + C_Q + 16 * j, bd0 + (j >> 1) * 512 + (j & 1) * 4, iqk,
             mode == 15 ? (j & 1) : (mode == 10 ? (j >= 4) : 0));
    }
  } else if (mode == 12) {
    MMA_CB("fill", t + C_O, bd0, pd, ipv, 1);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      MMA_TS(t + C_S, t + C_Q + 16 * j, bd0 + (j >> 1) * 512 + (j & 1) * 4, iqk, j > 0 ? 1u : 0u);
      if (j == 1) MMA_CB("use", t + C_O + 96, bd0 + 512, pd, ipv, 1);
      if (j == 3) MMA_CB("use", t + C_O + 192, bd0 + 1024, pd, ipv, 1);
      if (j == 5) MMA_CB("lastuse", t + C_O + 288, bd0 + 1536, pd, ipv, 1);
    }
    MMA_SS(t + C_S, aq, (bd0 + 4 * 512) ^ (SW128 ^ SW64), iqk, 1);
  } else if (mode == 13) {
    const uint32_t i128 = (iqk & ~(63u << 17)) | (16u << 17);
#pragma unroll
    for (int j = 0; j < 4; ++j) MMA_TS(t + C_S, t + C_Q + 16 * j, bd0 + j * 512, i128, j > 0 ? 1u : 0u);
  } else if (mode == 14) {
    const uint32_t i256 = (iqk & ~(63u << 17)) | (32u << 17);
#pragma unroll
    for (int j = 0; j < 2; ++j) MMA_TS(t + C_S, t + C_Q + 16 * j, bd0 + j * 512, i256, j > 0 ? 1u : 0u);
  }
}

extern "C" __global__ void __launch_bounds__(128, 1) k(int mode, int nsub, long long* out, int wn) {
  extern __shared__ __align__(1024) unsigned char dyn[];
  __shared__ uint64_t bars[8];
  __shared__ uint32_t tslot;
  unsigned char* sm = (unsigned char*)(((uintptr_t)dyn + 1023) & ~(uintptr_t)1023);
  const int tid = threadIdx.x, warp = tid / 32;
  uint32_t* w = (uint32_t*)sm;
  for (uint32_t i = tid; i < (SMEM - 1024) / 4; i += blockDim.x) {
    uint32_t x = (i + 1) * 2654435761u ^ (blockIdx.x * 40503u);
    w[i] = x & 0x3f3f3f3fu;  // finite E4M3 bytes
  }
  if (warp == 0) {
    asm volatile("tcgen05.alloc.exclusive.cta_group::1.sync.aligned.shared::cta.b32 [%0], 576; "
                 "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"
                 :: "r"(sa(&tslot)) : "memory");
  }
  if (tid == 0) {
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
  const bool two = mode == 8;
  if (warp < wn || (two && warp == 1)) {
    const uint32_t mb = sa(&bars[2 * warp]);
    const int my = two ? nsub : nsub / wn;
    for (int i = 0; i < my; ++i) {
      if (i >= 2) wait_par(mb + 8 * (i & 1), ((i >> 1) - 1) & 1);
      const uint32_t st = b0 + (i % NSTG) * STAGE;
      const uint64_t bd0 = SW128 | (st >> 4);
      const uint64_t pd = SW64 | ((b0 + OFF_P + (i & 1) * 6144) >> 4);
      if (mode >= 10) {
        lat(mode, t, bd0, aq, pd, iqk, ipv);
      } else {
        if (mode != 4 && mode != 6 && mode != 7 && !(two && warp == 1)) qk(mode, t, bd0, aq, iqk);
        if (mode != 3 && mode != 5 && !(two && warp == 0)) pv(mode, t, bd0, pd, ipv);
      }
      commit(mb + 8 * (i & 1));
    }
    for (int i = my > 2 ? my - 2 : 0; i < my; ++i) wait_par(mb + 8 * (i & 1), (i >> 1) & 1);
  }
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  if (tid == 0) out[blockIdx.x] = clock64() - c0;
  if (warp == 0)
    asm volatile("tcgen05.dealloc.exclusive.cta_group::1.sync.aligned.b32 %0, 576;" :: "r"(t) : "memory");
}

int main(int argc, char** argv) {
  int sms = 0, clk = 0;
  CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
  CK(cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0));
  CK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM + 1024));
  long long* d;
  CK(cudaMalloc(&d, sms * sizeof(long long)));
  const int nsub = argc > 1 ? atoi(argv[1]) : 4096;
  const char* names[] = {"exact h8", "PV no collector", "PV A K-major", "QK only", "PV only",
                         "QK latent only", "PV only no collector", "PV only A K-major",
                         "QK warp0 + PV warp1", "exact, RoPE first", "8 TS N64 4 indep acc",
                         "8 TS N64 8 indep acc", "QK/PV interleaved", "4 TS N128 chained (=64k)",
                         "2 TS N256 chained (=64k)", "8 TS N64 acc=0 every 2nd"};
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  printf("sms %d nsub %d\n", sms, nsub);
  const int sel[] = {0, 3, 4, 5, 10, 13, 14, 9, 12};
  for (int rep = 0; rep < 2; ++rep)
    for (int wn = 1; wn <= 4; wn *= 2)
    for (int mode : sel) {
      k<<<sms, 128, SMEM + 1024>>>(mode, 64, d, wn);
      CK(cudaEventRecord(e0));
      k<<<sms, 128, SMEM + 1024>>>(mode, nsub, d, wn);
      CK(cudaEventRecord(e1));
      CK(cudaDeviceSynchronize());
      float ms = 0;
      CK(cudaEventElapsedTime(&ms, e0, e1));
      std::vector<long long> h(sms);
      CK(cudaMemcpy(h.data(), d, sms * sizeof(long long), cudaMemcpyDeviceToHost));
      std::sort(h.begin(), h.end());
      double cps = (double)h[sms / 2] / nsub, mhz = (double)h[sms / 2] / (ms * 1e3);
      printf("rep %d W%d mode %2d %-22s clk/subtile %7.1f (min %7.1f max %7.1f) eff %5.0f MHz -> "
             "%6.1f us/layer at that clock\n", rep, wn, mode, names[mode], cps,
             (double)h[0] / nsub, (double)h[sms - 1] / nsub, mhz, cps * 2473 / mhz);
    }
  return 0;
}
