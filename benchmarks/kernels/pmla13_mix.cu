// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// pmla13_mix.cu (P-MLA round 5): does the h8 tensor-pipe mix slow the random-row LDGSTS KV gather (and vice versa)
// on sm_107a? One CTA per SM, all SMs, one F16K DSA layer per run (2473 64-key subtiles per SM):
//   gather: 8 warps, the lean h8 producer (h8 lane map, SW128 latent atoms + SW64 RoPE atom, zero-fill form,
//           per-warp in-flight cap 4, token index rows in smem from an index warp) into a 6-stage ring at 0-216 KB
//           (pmla12 L1: 58.8 B/clk/SM alone);
//   MMA:    the exact h8 subtile mix from pmla11 (QK = 8 TS M128 N64 K64 + 1 SS RoPE; PV = 4 x M128 N96 K64 SS with
//           collector::b) issued by a QK warp and a PV warp, free-running (2-deep commit rings), reading a fixed
//           stage above the ring (operand values are irrelevant for timing).
// Modes: 1 gather only | 2 MMA only | 3 both | 4 both, MMA reads the ring stages the gather writes (stage i % 6).
// Reports each role's own clk per subtile (CTA 0 stamps) and the kernel time.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <dlfcn.h>
#include <unistd.h>
// NVML through dlopen (no nvml.h in the image): init, device handle, total energy counter (mJ)
typedef int (*nvmlInit_t)();
typedef int (*nvmlHandle_t)(unsigned, void**);
typedef int (*nvmlEnergy_t)(void*, unsigned long long*);
static nvmlEnergy_t g_energy = nullptr;
static void* g_dev = nullptr;
static void nvml_open() {
  void* h = dlopen("libnvidia-ml.so.1", RTLD_NOW);
  if (!h) return;
  auto init = (nvmlInit_t)dlsym(h, "nvmlInit_v2");
  auto get = (nvmlHandle_t)dlsym(h, "nvmlDeviceGetHandleByIndex_v2");
  g_energy = (nvmlEnergy_t)dlsym(h, "nvmlDeviceGetTotalEnergyConsumption");
  if (!init || !get || !g_energy || init() != 0 || get(0, &g_dev) != 0) g_energy = nullptr;
}
static unsigned long long energy_mj() {
  unsigned long long e = 0;
  if (g_energy) g_energy(g_dev, &e);
  return e;
}

constexpr int NS = 6, STAGE = 36864, ROWB = 576, KEYS = 2048, SUBS = KEYS / 64, NTOK = 16384;
constexpr uint32_t OFF_BAR = NS * STAGE;      // 221184
constexpr uint32_t OFF_IDX = OFF_BAR + 1024;  // 2 x 8 KB token index rows
constexpr uint32_t OFF_MST = 239616;          // MMA stage (36 KB), 1024-aligned
constexpr uint32_t OFF_QS = OFF_MST + STAGE;  // RoPE Q A tile (8 KB)
constexpr uint32_t OFF_P = OFF_QS + 8192;     // 2 x P^T (6 KB)
constexpr uint32_t SMEM_USED = OFF_P + 2 * 6144;
// barriers: per-warp FW[8][NS] at 0, TKI[2], TKF[2], MMA commit rings QK[2], PV[2], done flag word after
constexpr int B_FW = 0, B_TKI = 8 * NS, B_TKF = B_TKI + 2, B_MQK = B_TKF + 2, B_MPV = B_MQK + 2;
static constexpr uint64_t SW128 = (uint64_t(1024 >> 4) << 32) | (1ull << 46) | (2ull << 61);
static constexpr uint64_t SW64 = (uint64_t(512 >> 4) << 32) | (1ull << 46) | (4ull << 61);
static constexpr uint32_t C_Q = 0, C_S = 128, C_O = 192;

__device__ __forceinline__ uint32_t bar(uint32_t base, int i) { return base + OFF_BAR + 8 * i; }
__device__ __forceinline__ void mbar_init(uint32_t b, uint32_t n) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(b), "r"(n) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t b, uint32_t parity) {
  asm volatile(
      "{ .reg .pred p; W%=: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1; @!p bra W%=; }" ::"r"(b),
      "r"(parity)
      : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(b) : "memory");
}
__device__ __forceinline__ void cp_arrive(uint32_t b) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(b) : "memory");
}
__device__ __forceinline__ void cp16(uint32_t dst, const uint8_t* src, uint32_t nb) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(nb) : "memory");
}
__device__ __forceinline__ uint32_t lds32(uint32_t a) {
  uint32_t v;
  asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(a) : "memory");
  return v;
}
__device__ __forceinline__ void commit(uint32_t b) {
  asm volatile("{.reg .pred e; elect.sync _|e, 0xffffffff; @e tcgen05.commit.cta_group::1."
               "mbarrier::arrive::one.shared::cluster.b64 [%0];}" ::"r"(b)
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

template <int PADZ>
__global__ void __launch_bounds__(384, 1) mk(int mode, const uint8_t* __restrict__ kv, const int* __restrict__ idx,
                                              unsigned long long* clk) {
  extern __shared__ __align__(1024) unsigned char sm[];
  __shared__ uint32_t tslot;
  const uint32_t base = (uint32_t)__cvta_generic_to_shared(sm);
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int grid = gridDim.x, cta = blockIdx.x;
  const int n_mine = (NTOK - cta + grid - 1) / grid;
  const int nsub = n_mine * SUBS;
  const bool G = mode != 2, M = mode != 1;
  // finite E4M3 operand bytes for the MMA region
  for (uint32_t i = threadIdx.x; i < (SMEM_USED - OFF_MST) / 4; i += blockDim.x)
    ((uint32_t*)(sm + OFF_MST))[i] = ((i + 1) * 2654435761u ^ (cta * 40503u)) & 0xbfbfbfbfu;
  if (threadIdx.x == 0) {
    for (int w = 0; w < 8; ++w)
      for (int s = 0; s < NS; ++s) mbar_init(bar(base, B_FW + w * NS + s), 32);
    for (int p = 0; p < 2; ++p) {
      mbar_init(bar(base, B_TKI + p), 32);
      mbar_init(bar(base, B_TKF + p), 8);
      mbar_init(bar(base, B_MQK + p), 1);
      mbar_init(bar(base, B_MPV + p), 1);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (warp == 10)
    asm volatile("tcgen05.alloc.exclusive.cta_group::1.sync.aligned.shared::cta.b32 [%0], 576; "
                 "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::"r"(
                     (uint32_t)__cvta_generic_to_shared(&tslot))
                 : "memory");
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  const uint32_t t = tslot;
  // Q latent in TMEM (cols C_Q .. C_Q + 127): every warp of the first warpgroup writes its 32 lanes with random
  // E4M3 words (lanes 96-127: zero when PADZ, i.e. h8 with zeroed padding heads)
  if (warp < 4) {
    uint32_t x = 0x9e3779b9u * (threadIdx.x + 1) ^ (cta * 7919u);
    for (int c = 0; c < 128; c += 8) {
      uint32_t v[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        x = x * 1664525u + 1013904223u;
        v[j] = (PADZ && warp == 3) ? 0u : (x & 0xbfbfbfbfu);
      }
      asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};" ::"r"(
                       t + ((uint32_t)(warp * 32) << 16) + C_Q + c),
                   "r"(v[0]), "r"(v[1]), "r"(v[2]), "r"(v[3]), "r"(v[4]), "r"(v[5]), "r"(v[6]), "r"(v[7])
                   : "memory");
    }
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
  }
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  const unsigned long long c0 = clock64();
  if (cta == 0 && threadIdx.x == 0) {
    unsigned long long g;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g));
    clk[0] = c0;
    clk[1] = g;
  }
  if (warp < 8 && G) {
    const int pw = warp, r = pw * 8 + lane / 4, ch = lane % 4, rx = r % 8;
    const uint32_t d0 = r * 128 + ((ch ^ rx) << 4), d1 = r * 128 + (((ch + 4) ^ rx) << 4);
    const uint32_t dr = 4 * 8192 + r * 64 + ((ch ^ ((r >> 1) & 3)) << 4);
    int g = 0;
    for (int qi = 0; qi < n_mine; ++qi) {
      const uint32_t ib = base + OFF_IDX + (qi % 2) * KEYS * 4 + 4 * r;
      mbar_wait(bar(base, B_TKI + qi % 2), (qi / 2) & 1);
      for (int tt = 0; tt < SUBS; ++tt) {
        const int s = g % NS;
        const uint32_t st = base + s * STAGE;
        const int rc = (int)lds32(ib + 256 * tt);
        if (g >= 4) mbar_wait(bar(base, B_FW + pw * NS + (g - 4) % NS), ((g - 4) / NS) & 1);
        const uint8_t* src = kv + (long)rc * ROWB + 16 * ch;
#pragma unroll
        for (int a = 0; a < 4; ++a) {
          cp16(st + a * 8192 + d0, src + 128 * a, 16u);
          cp16(st + a * 8192 + d1, src + 128 * a + 64, 16u);
        }
        cp16(st + dr, src + 512, 16u);
        cp_arrive(bar(base, B_FW + pw * NS + s));
        ++g;
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(bar(base, B_TKF + qi % 2));
    }
    for (int m = g > NS ? g - NS : 0; m < g; ++m) mbar_wait(bar(base, B_FW + pw * NS + m % NS), (m / NS) & 1);
    if (cta == 0 && lane == 0) clk[8 + pw] = clock64() - c0;
  } else if (warp == 9 && G) {
    for (int qi = 0; qi < n_mine; ++qi) {
      const int tok = cta + qi * grid;
      if (qi >= 2) mbar_wait(bar(base, B_TKF + qi % 2), ((qi - 2) / 2) & 1);
      const uint32_t ib = base + OFF_IDX + (qi % 2) * KEYS * 4;
      const int4* src = (const int4*)(idx + (long)tok * KEYS);
      for (int i = lane; i < KEYS / 4; i += 32) {
        const int4 v = __ldg(src + i);
        asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" ::"r"(ib + 16 * i), "r"(v.x), "r"(v.y), "r"(v.z),
                     "r"(v.w)
                     : "memory");
      }
      mbar_arrive(bar(base, B_TKI + qi % 2));
    }
  } else if ((warp == 10 || warp == 11) && M) {
    const uint32_t iqk = (1u << 4) | (8u << 17) | (8u << 24) | (1u << 29);
    const uint32_t ipv = (1u << 4) | (12u << 17) | (8u << 24) | (1u << 29) | (1u << 15);
    const uint64_t aq = SW64 | ((base + OFF_QS) >> 4);
    const uint32_t mb = bar(base, warp == 10 ? B_MQK : B_MPV);
    for (int i = 0; i < nsub; ++i) {
      if (i >= 2) mbar_wait(mb + 8 * (i & 1), ((i >> 1) - 1) & 1);
      const uint32_t st = mode == 4 ? base + (i % NS) * STAGE : base + OFF_MST;
      const uint64_t bd0 = SW128 | (st >> 4);
      const uint64_t pd = SW64 | ((base + OFF_P + (i & 1) * 6144) >> 4);
      asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
      if (warp == 10) {
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
          for (int k2 = 0; k2 < 2; ++k2)
            MMA_TS(t + C_S, t + C_Q + 32 * j + 16 * k2, bd0 + j * 512 + k2 * 4, iqk, (j + k2 > 0) ? 1u : 0u);
        MMA_SS(t + C_S, aq, (bd0 + 4 * 512) ^ (SW128 ^ SW64), iqk, 1);
      } else {
        MMA_CB("fill", t + C_O, bd0, pd, ipv, 1);
        MMA_CB("use", t + C_O + 96, bd0 + 512, pd, ipv, 1);
        MMA_CB("use", t + C_O + 192, bd0 + 1024, pd, ipv, 1);
        MMA_CB("lastuse", t + C_O + 288, bd0 + 1536, pd, ipv, 1);
      }
      commit(mb + 8 * (i & 1));
    }
    for (int i = nsub > 2 ? nsub - 2 : 0; i < nsub; ++i) mbar_wait(mb + 8 * (i & 1), (i >> 1) & 1);
    if (cta == 0 && lane == 0) clk[16 + warp - 10] = clock64() - c0;
  }
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  if (cta == 0 && threadIdx.x == 0) {
    unsigned long long g;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g));
    clk[2] = clock64();
    clk[3] = g;
  }
  if (warp == 10)
    asm volatile("tcgen05.dealloc.exclusive.cta_group::1.sync.aligned.b32 %0, 576;" ::"r"(t) : "memory");
}

int main() {
  int nsm = 0;
  cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
  const int nrows = 78 * 1024;
  const long nkeys = (long)NTOK * KEYS;
  uint8_t* kv;
  int* idx;
  unsigned long long* dclk;
  cudaMalloc(&kv, (size_t)nrows * ROWB);
  {
    std::vector<uint32_t> hk((size_t)nrows * ROWB / 4);
    uint32_t x = 12345u;
    for (auto& v : hk) {
      x = x * 1664525u + 1013904223u;
      v = x & 0xbfbfbfbfu;  // finite E4M3 (|v| < 2, both signs), randn-like bit activity
    }
    cudaMemcpy(kv, hk.data(), hk.size() * 4, cudaMemcpyHostToDevice);
  }
  std::vector<int> hi(nkeys);
  srand(1);
  for (long t = 0; t < NTOK; ++t) {
    const int ctx = nrows - NTOK + (int)t + 1;
    for (int k = 0; k < KEYS; ++k) hi[t * KEYS + k] = rand() % ctx;
  }
  cudaMalloc(&idx, nkeys * 4);
  cudaMemcpy(idx, hi.data(), nkeys * 4, cudaMemcpyHostToDevice);
  cudaMalloc(&dclk, 256);
  cuInit(0);
  CUfunction f;
  cudaGetFuncBySymbol((cudaFunction_t*)&f, (const void*)mk<0>);
  CUfunction fz;
  cudaGetFuncBySymbol((cudaFunction_t*)&fz, (const void*)mk<1>);
  cuFuncSetAttribute(fz, CU_FUNC_ATTRIBUTE_SHARED_MEMORY_MODE, CU_SHARED_MEMORY_MODE_ALLOW_OVERSIZED_SHARED_MEMORY);
  cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_SHARED_MEMORY_MODE, CU_SHARED_MEMORY_MODE_ALLOW_OVERSIZED_SHARED_MEMORY);
  CUlaunchConfig cfg = {};
  cfg.gridDimX = nsm;
  cfg.gridDimY = cfg.gridDimZ = 1;
  cfg.blockDimX = 384;
  cfg.blockDimY = cfg.blockDimZ = 1;
  cfg.sharedMemBytes = 300 * 1024;  // + static tslot: stay below the 327 KiB per-CTA maximum
  CUlaunchAttribute la[1];
  la[0].id = CU_LAUNCH_ATTRIBUTE_SHARED_MEMORY_MODE;
  la[0].value.sharedMemoryMode = CU_SHARED_MEMORY_MODE_ALLOW_OVERSIZED_SHARED_MEMORY;
  cfg.attrs = la;
  cfg.numAttrs = 1;
  printf("SMs %d, F16K layer: 2473 subtiles/SM, 36864 B gathered per subtile\n", nsm);
  const char* names[] = {"", "gather only", "MMA only", "gather + MMA (fixed stage)", "gather + MMA (ring stages)",
                         "MMA only, pad lanes 0", "gather + MMA fixed, pad 0", "gather + MMA ring, pad 0"};
  nvml_open();
  {
    const unsigned long long e0 = energy_mj();
    usleep(1000000);
    printf("idle power %.0f W (nvml %s)\n", (double)(energy_mj() - e0) / 1000.0, g_energy ? "ok" : "missing");
  }
  for (int rep = 0; rep < 2; ++rep)
    for (int mode = 1; mode <= 7; ++mode) {
      int md = mode;
      void* args[] = {(void*)&md, (void*)&kv, (void*)&idx, (void*)&dclk};
      cuLaunchKernelEx(&cfg, f, args, nullptr);
      cudaMemset(dclk, 0, 256);
      CUfunction fm = (mode >= 5) ? fz : f;
      if (mode >= 5) md = mode - 3;  // 5 / 6 / 7 = modes 2 / 3 / 4 with zero padding lanes
      cudaEvent_t a, b;
      cudaEventCreate(&a);
      cudaEventCreate(&b);
      cuLaunchKernelEx(&cfg, fm, args, nullptr);
      cudaDeviceSynchronize();
      const unsigned long long e0 = energy_mj();
      cudaEventRecord(a);
      CUresult r0 = CUDA_SUCCESS;
      const int NREP = 2500;  // ~3 s sustained per arm (power-cap steady state)
      for (int i = 0; i < NREP; ++i) r0 = cuLaunchKernelEx(&cfg, fm, args, nullptr);
      cudaEventRecord(b);
      cudaEventSynchronize(b);
      const unsigned long long e1 = energy_mj();
      float ms;
      cudaEventElapsedTime(&ms, a, b);
      ms /= NREP;
      const double jpl = (double)(e1 - e0) / 1000.0 / NREP;
      cudaError_t e = cudaGetLastError();
      unsigned long long h[32];
      cudaMemcpy(h, dclk, sizeof(h), cudaMemcpyDeviceToHost);
      const double mhz = (h[3] > h[1]) ? (double)(h[2] - h[0]) / (double)(h[3] - h[1]) * 1000.0 : 0.0;
      const int nsub0 = ((NTOK + nsm - 1) / nsm) * SUBS;
      unsigned long long gmax = 0;
      for (int w = 0; w < 8; ++w) gmax = h[8 + w] > gmax ? h[8 + w] : gmax;
      printf("%-28s: %7.1f us %6.3f J/layer %5.0f W clk %4.0f MHz | gather %6.1f clk/subtile = %5.1f B/clk/SM | QK warp %6.1f PV warp %6.1f "
             "clk/subtile %s\n",
             names[mode], ms * 1000, jpl, jpl / (ms * 1e-3), mhz, gmax ? (double)gmax / nsub0 : 0.0,
             gmax ? 36864.0 * nsub0 / (double)gmax : 0.0, h[16] ? (double)h[16] / nsub0 : 0.0,
             h[17] ? (double)h[17] / nsub0 : 0.0,
             (r0 == CUDA_SUCCESS && e == cudaSuccess) ? "" : cudaGetErrorString(e));
      if (r0 != CUDA_SUCCESS) printf("  launch result %d\n", (int)r0);
    }
  return 0;
}
