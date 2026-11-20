// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// pmla16_rt.cu (P-MLA round 6): tensor-pipe floors of the TMEM-feasible heads-on-M layouts on sm_107a, all SMs.
// Layouts (TMEM 576 cols): L0 = h8 (Q latent 128 + S 64 + O^T 384; RoPE QK SS), L1 = Q latent 128 + RoPE 16 +
// ONE 48-key S + O^T 384 (RoPE QK TS), L2 = the same with ONE 32-key S. Per round trip (N keys): the QK warp waits
// S-free, issues QK (8 latent TS K64 + RoPE, N columns) and commits S-full; 4 softmax warps (one per TMEM lane
// quarter) wait S-full, load the N S columns, fence and arrive S-free; with PV on, a PV warp issues N/16 PV MMAs
// (M128 N96 K64, A = V^T MN-major SW128, B = P^T SW64 with collector::b; the h8 PV form, 4 per 64 keys) after each
// S-free. Free-running mode: QK and PV warps issue back to back (2-deep commit rings), no softmax: the TC mix
// throughput of the layout. Calibration vs P-MLA-2CTA pmla13 (h8 chain 595 clk) and pmla11 (h8 mix 778-790).
// Prints clk per round trip and per 64 keys (median over SMs) and us/layer at the measured clock.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

#define CK(x)                                                         \
  do {                                                                \
    cudaError_t e = (x);                                              \
    if (e != cudaSuccess) {                                           \
      printf("CUDA %s at %d\n", cudaGetErrorString(e), __LINE__);     \
      return 1;                                                       \
    }                                                                 \
  } while (0)

static constexpr uint64_t SW128 = (uint64_t(1024 >> 4) << 32) | (1ull << 46) | (2ull << 61);
static constexpr uint64_t SW64 = (uint64_t(512 >> 4) << 32) | (1ull << 46) | (4ull << 61);
static constexpr uint32_t STAGE = 36864, NSTG = 2, OFF_QS = NSTG * STAGE, OFF_P = OFF_QS + 8192;
static constexpr uint32_t SMEM = OFF_P + 2 * 6144 + 1024;
static constexpr uint32_t C_Q = 0, C_R = 128, C_O = 192;

__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void wait_par(uint32_t bar, uint32_t par) {
  asm volatile(
      "{.reg .pred P; W%=: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1; "
      "@!P bra W%=;}" ::"r"(bar),
      "r"(par)
      : "memory");
}
__device__ __forceinline__ void commit(uint32_t bar) {
  asm volatile(
      "{.reg .pred e; elect.sync _|e, 0xffffffff; @e tcgen05.commit.cta_group::1."
      "mbarrier::arrive::one.shared::cluster.b64 [%0];}" ::"r"(bar)
      : "memory");
}
#define MMA_TS(d, a, b, i, acc)                                                                      \
  asm volatile("{.reg .pred p, e; elect.sync _|e, 0xffffffff; "                                      \
               "setp.ne.b32 p, %4, 0; @e tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [%1], %2, %3, p;}" \
               ::"r"(d), "r"(a), "l"(b), "r"(i), "r"(acc)                                            \
               : "memory")
#define MMA_SS(d, a, b, i, acc)                                                                    \
  asm volatile("{.reg .pred p, e; elect.sync _|e, 0xffffffff; "                                    \
               "setp.ne.b32 p, %4, 0; @e tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;}" \
               ::"r"(d), "l"(a), "l"(b), "r"(i), "r"(acc)                                          \
               : "memory")
#define MMA_CB(op, d, a, b, i, acc)                                                         \
  asm volatile("{.reg .pred p, e; elect.sync _|e, 0xffffffff; "                             \
               "setp.ne.b32 p, %4, 0; @e tcgen05.mma.cta_group::1.kind::f8f6f4.collector::b::" op \
               " [%0], %1, %2, %3, p;}" ::"r"(d),                                           \
               "l"(a), "l"(b), "r"(i), "r"(acc)                                             \
               : "memory")

__device__ __forceinline__ uint32_t cs(int lay) { return lay == 0 ? 128u : 144u; }
__device__ __forceinline__ int nk(int lay) { return lay == 0 ? 64 : lay == 1 ? 48 : 32; }

template <int lay>
__device__ __forceinline__ void qk(uint32_t t, uint64_t bd0, uint64_t aq) {
  constexpr uint32_t n = lay == 0 ? 64u : lay == 1 ? 48u : 32u;
  constexpr uint32_t iqk = (1u << 4) | ((n >> 3) << 17) | (8u << 24) | (1u << 29);
  const uint32_t d = t + cs(lay);
#pragma unroll
  for (int j = 0; j < 4; ++j)
#pragma unroll
    for (int k2 = 0; k2 < 2; ++k2) MMA_TS(d, t + C_Q + 32 * j + 16 * k2, bd0 + j * 512 + k2 * 4, iqk, (j + k2 > 0) ? 1u : 0u);
  const uint64_t br = (bd0 + 4 * 512) ^ (SW128 ^ SW64);
  if constexpr (lay == 0)
    MMA_SS(d, aq, br, iqk, 1);
  else
    MMA_TS(d, t + C_R, br, iqk, 1);
}

__device__ __forceinline__ void pv(int npv, uint32_t t, uint64_t bd0, uint64_t pd) {
  const uint32_t ipv = (1u << 4) | (12u << 17) | (8u << 24) | (1u << 29) | (1u << 15);
  if (npv == 4) {
    MMA_CB("fill", t + C_O, bd0, pd, ipv, 1);
    MMA_CB("use", t + C_O + 96, bd0 + 512, pd, ipv, 1);
    MMA_CB("use", t + C_O + 192, bd0 + 1024, pd, ipv, 1);
    MMA_CB("lastuse", t + C_O + 288, bd0 + 1536, pd, ipv, 1);
  } else if (npv == 3) {
    MMA_CB("fill", t + C_O, bd0, pd, ipv, 1);
    MMA_CB("use", t + C_O + 96, bd0 + 512, pd, ipv, 1);
    MMA_CB("lastuse", t + C_O + 192, bd0 + 1024, pd, ipv, 1);
  } else {
    MMA_CB("fill", t + C_O, bd0, pd, ipv, 1);
    MMA_CB("lastuse", t + C_O + 96, bd0 + 512, pd, ipv, 1);
  }
}

__device__ __forceinline__ float ld16(uint32_t a) {
  float v[16];
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
      : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]), "=f"(v[4]), "=f"(v[5]), "=f"(v[6]), "=f"(v[7]), "=f"(v[8]),
        "=f"(v[9]), "=f"(v[10]), "=f"(v[11]), "=f"(v[12]), "=f"(v[13]), "=f"(v[14]), "=f"(v[15])
      : "r"(a)
      : "memory");
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < 16; ++i) s = fmaxf(s, v[i]);
  return s;
}

#define R8(b) "=f"(v[b]), "=f"(v[b + 1]), "=f"(v[b + 2]), "=f"(v[b + 3]), "=f"(v[b + 4]), "=f"(v[b + 5]), "=f"(v[b + 6]), "=f"(v[b + 7])
__device__ __forceinline__ float ldr16(uint32_t a) {
  float v[16], m;
  asm volatile("tcgen05.ld.red.sync.aligned.32x32b.x16.max.f32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, %16, [%17];"
               : R8(0), R8(8), "=f"(m) : "r"(a) : "memory");
  return m + v[0] * 0.f + v[15] * 0.f;
}
__device__ __forceinline__ float ldr32(uint32_t a) {
  float v[32], m;
  asm volatile("tcgen05.ld.red.sync.aligned.32x32b.x32.max.f32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
               "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, [%33];"
               : R8(0), R8(8), R8(16), R8(24), "=f"(m) : "r"(a) : "memory");
  return m + v[0] * 0.f + v[31] * 0.f;
}
__device__ __forceinline__ float ldr64(uint32_t a) {
  float v[64], m;
  asm volatile("tcgen05.ld.red.sync.aligned.32x32b.x64.max.f32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
               "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,"
               "%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %64, [%65];"
               : R8(0), R8(8), R8(16), R8(24), R8(32), R8(40), R8(48), R8(56), "=f"(m) : "r"(a) : "memory");
  return m + v[0] * 0.f + v[63] * 0.f;
}

// lay: 0 h8 / 1 L1 / 2 L2; pvon: PV MMAs after each S-free; chain: 0 = free-running TC mix (no softmax)
template <int lay>
__global__ void __launch_bounds__(320, 1) rt(int pvon, int chain, int nsub, long long* out, int red) {
  extern __shared__ __align__(1024) unsigned char dyn[];
  __shared__ uint64_t bars[10];
  __shared__ uint32_t tslot;
  unsigned char* sm = (unsigned char*)(((uintptr_t)dyn + 1023) & ~(uintptr_t)1023);
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  uint32_t* w = (uint32_t*)sm;
  for (uint32_t i = tid; i < (SMEM - 1024) / 4; i += blockDim.x) w[i] = ((i + 1) * 2654435761u) & 0x3f3f3f3fu;
  if (warp == 0)
    asm volatile(
        "tcgen05.alloc.exclusive.cta_group::1.sync.aligned.shared::cta.b32 [%0], 576; "
        "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::"r"(sa(&tslot))
        : "memory");
  if (tid == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(sa(&bars[0])) : "memory");  // S-full
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 4;" ::"r"(sa(&bars[1])) : "memory");  // S-free
    for (int j = 2; j < 6; ++j) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(sa(&bars[j])) : "memory");
    for (int j = 6; j < 10; ++j) asm volatile("mbarrier.init.shared::cta.b64 [%0], 4;" ::"r"(sa(&bars[j])) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  const uint32_t t = tslot, b0 = sa(sm);
  const uint32_t sfull = sa(&bars[0]), sfree = sa(&bars[1]), qkr = sa(&bars[2]), pvr = sa(&bars[4]);
  const uint32_t go = sa(&bars[6]);  // 4-deep ring: softmax -> PV warp (P(i) ready stand-in)
  const uint64_t aq = SW64 | ((b0 + OFF_QS) >> 4);
  const int n = nk(lay), npv = n / 16;
  long long c0 = clock64();
  float acc = 0.f;
  if (warp == 8) {  // QK issuer
    for (int i = 0; i < nsub; ++i) {
      if (chain) {
        if (i >= 1) wait_par(sfree, (i - 1) & 1);
        if (pvon && i >= 2) wait_par(pvr + 8 * (i & 1), ((i >> 1) - 1) & 1);  // bounds the PV lag to 2
      } else if (i >= 2) {
        wait_par(qkr + 8 * (i & 1), ((i >> 1) - 1) & 1);
      }
      asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
      qk<lay>(t, SW128 | ((b0 + (i % NSTG) * STAGE) >> 4), aq);
      commit(chain ? sfull : qkr + 8 * (i & 1));
    }
    if (chain)
      wait_par(sfree, (nsub - 1) & 1);
    else
      for (int i = nsub > 2 ? nsub - 2 : 0; i < nsub; ++i) wait_par(qkr + 8 * (i & 1), (i >> 1) & 1);
  } else if (warp == 9 && pvon) {  // PV issuer
    for (int i = 0; i < nsub; ++i) {
      if (chain) wait_par(go + 8 * (i & 3), (i >> 2) & 1);
      if (i >= 2) wait_par(pvr + 8 * (i & 1), ((i >> 1) - 1) & 1);
      asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
      pv(npv, t, SW128 | ((b0 + (i % NSTG) * STAGE) >> 4), SW64 | ((b0 + OFF_P + (i & 1) * 6144) >> 4));
      commit(pvr + 8 * (i & 1));
    }
    for (int i = nsub > 2 ? nsub - 2 : 0; i < nsub; ++i) wait_par(pvr + 8 * (i & 1), (i >> 1) & 1);
  } else if (warp < 4 && chain) {  // softmax stand-in: S load of the N columns, fence, S-free
    const uint32_t la = t + ((uint32_t)(warp * 32) << 16) + cs(lay);
    for (int i = 0; i < nsub; ++i) {
      wait_par(sfull, i & 1);
      asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
      float m;
      if (red) {  // rt13a form: one reduction load per S (x64 | x32 + x16 | x32)
        m = n == 64 ? ldr64(la) : n == 48 ? fmaxf(ldr32(la), ldr16(la + 32)) : ldr32(la);
      } else {
        m = ld16(la);
        if (n >= 32) m = fmaxf(m, ld16(la + 16));
        if (n >= 48) m = fmaxf(m, ld16(la + 32));
        if (n >= 64) m = fmaxf(m, ld16(la + 48));
      }
      asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
      asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
      __syncwarp();
      if (lane == 0) {
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(sfree) : "memory");
        if (pvon) asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(go + 8 * (i & 3)) : "memory");
      }
      acc += m;
    }
  }
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  if (tid == 0) out[blockIdx.x] = clock64() - c0;
  if (acc == 12345.f) out[0] = 0;
  if (warp == 0) asm volatile("tcgen05.dealloc.exclusive.cta_group::1.sync.aligned.b32 %0, 576;" ::"r"(t) : "memory");
}

int main(int argc, char** argv) {
  int sms = 0;
  CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
  CK(cudaFuncSetAttribute(rt<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM + 1024));
  CK(cudaFuncSetAttribute(rt<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM + 1024));
  CK(cudaFuncSetAttribute(rt<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM + 1024));
  long long* d;
  CK(cudaMalloc(&d, sms * sizeof(long long)));
  const int nsub = argc > 1 ? atoi(argv[1]) : 4096;
  const char* ln[] = {"L0 h8 (S64, RoPE SS)", "L1 (S48, RoPE TS)", "L2 (S32, RoPE TS)"};
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  struct M {
    int lay, pvon, chain, red;
    const char* what;
  };
  const M ms[] = {{0, 0, 1, 1, "chain, ld.red (rt13a)"}, {1, 0, 1, 1, "chain, ld.red"}, {2, 0, 1, 1, "chain, ld.red"},
                  {0, 1, 1, 1, "chain + PV, ld.red"}, {1, 1, 1, 1, "chain + PV, ld.red"},
                  {0, 1, 0, 1, "free-running QK+PV"}, {1, 1, 0, 1, "free-running QK+PV"},
                  {0, 0, 0, 1, "free-running QK only"}, {1, 0, 0, 1, "free-running QK only"}};
  for (int rep = 0; rep < 2; ++rep)
    for (const M& m : ms) {
      auto launch = [&](int ns) {
        if (m.lay == 0) rt<0><<<sms, 320, SMEM + 1024>>>(m.pvon, m.chain, ns, d, m.red);
        else if (m.lay == 1) rt<1><<<sms, 320, SMEM + 1024>>>(m.pvon, m.chain, ns, d, m.red);
        else rt<2><<<sms, 320, SMEM + 1024>>>(m.pvon, m.chain, ns, d, m.red);
      };
      launch(64);
      CK(cudaEventRecord(e0));
      launch(nsub);
      CK(cudaEventRecord(e1));
      CK(cudaDeviceSynchronize());
      float msec = 0;
      CK(cudaEventElapsedTime(&msec, e0, e1));
      std::vector<long long> h(sms);
      CK(cudaMemcpy(h.data(), d, sms * sizeof(long long), cudaMemcpyDeviceToHost));
      std::sort(h.begin(), h.end());
      const int n = m.lay == 0 ? 64 : m.lay == 1 ? 48 : 32;
      const double cps = (double)h[sms / 2] / nsub, mhz = (double)h[sms / 2] / (msec * 1e3);
      const double c64 = cps * 64.0 / n, usl = msec * 1e3 * 2473.0 / nsub * 64.0 / n;
      printf("rep %d %-22s %-24s clk/rt %6.1f -> %6.1f clk/64 keys at %5.0f MHz | event %7.1f us/layer\n", rep,
             ln[m.lay], m.what, cps, c64, mhz, usl);
    }
  return 0;
}
