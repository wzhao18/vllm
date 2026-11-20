// pmla12_ladder.cu (P-MLA round 5): delta ladder from the lean LDGSTS probe to the h8 in-kernel KV producer.
// One F16K DSA layer = 16384 tokens x 2048 random rows of 576 B from a 78K-row KV (45 MB, L2-resident); persistent
// grid = #SMs, one CTA per SM, each CTA walks its tokens in 64-key stages through a 6-stage ring at smem 0-216 KB
// (below the 227 KiB LDGSTS line). 8 producer warps own rows 8 pw .. 8 pw + 7 of every stage, 4 lanes per row,
// 9 x cp.async 16 B per lane per stage (h8 gather_cp lane map). Rungs (template knobs):
//   LAYOUT 0 linear rows (r * 576) | 1 h8 SW128 latent atoms + SW64 RoPE atom
//   Z      0 plain cp.async        | 1 src-size operand (zero-fill form, as h8)
//   SYNC   0 per-warp completion barriers, the warp waits only for its own stage g - CAP
//          1 h8: CTA-wide KVFULL (256 noinc arrivals), lane 0 waits KVEMPTY[s] (consumer) + KVFULL[g - CAP], syncwarp;
//            consumer warp 8 waits KVFULL[s] (+ fence.proxy.async if FENCE) and arrives KVEMPTY[s]
//   IDX    0 indices from global, register prefetch 2 stages ahead (h8 physical) | 1 one ld.shared per stage from a
//            per-token index row in smem (h8 tconv), filled by warp 9 one token ahead
//   XW     extra idle warps (h8 has 20 warps)
// Reports us/layer, TB/s, B/clk/SM at the SM clock measured in CTA 0 (clock64 / globaltimer).
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

constexpr int NS = 6, STAGE = 36864, ROWB = 576, KEYS = 2048, SUBS = KEYS / 64, NTOK = 16384;
constexpr int OFF_BAR = NS * STAGE;                // 221184
constexpr int OFF_IDX = OFF_BAR + 1024;            // 2 x 8 KB token index rows (IDX 1)
constexpr int SMEM = OFF_IDX + 2 * KEYS * 4 + 1024;
// barriers (8 B each): KVFULL[NS] at 0, KVEMPTY[NS] at NS, per-warp FULLW[8][NS] at 2 NS, TKI[2] / TKF[2] after
constexpr int B_FULL = 0, B_EMPTY = NS, B_FW = 2 * NS, B_TKI = 2 * NS + 8 * NS, B_TKF = B_TKI + 2;

__device__ __forceinline__ uint32_t bar(uint32_t base, int i) { return base + OFF_BAR + 8 * i; }
__device__ __forceinline__ void mbar_init(uint32_t b, uint32_t n) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(b), "r"(n) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t b, uint32_t parity) {
  asm volatile(
      "{ .reg .pred p; W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1; @!p bra W; }" ::"r"(b), "r"(parity)
      : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(b) : "memory");
}
__device__ __forceinline__ void cp_arrive(uint32_t b) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(b) : "memory");
}
template <int Z>
__device__ __forceinline__ void cp16(uint32_t dst, const uint8_t* src, uint32_t nb) {
  if (Z)
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(nb) : "memory");
  else
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ uint32_t lds32(uint32_t a) {
  uint32_t v;
  asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(a) : "memory");
  return v;
}

template <int LAYOUT, int Z, int SYNC, int CAP, int FENCE, int IDX>
__global__ void __launch_bounds__(640, 1) lk(const uint8_t* __restrict__ kv, const int* __restrict__ idx,
                                              unsigned long long* clk) {
  extern __shared__ __align__(1024) unsigned char sm[];
  const uint32_t base = (uint32_t)__cvta_generic_to_shared(sm);
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int grid = gridDim.x, cta = blockIdx.x;
  const int n_mine = (NTOK - cta + grid - 1) / grid;
  if (threadIdx.x == 0) {
    for (int s = 0; s < NS; ++s) {
      mbar_init(bar(base, B_FULL + s), 256);
      mbar_init(bar(base, B_EMPTY + s), 1);
      for (int w = 0; w < 8; ++w) mbar_init(bar(base, B_FW + w * NS + s), 32);
    }
    for (int p = 0; p < 2; ++p) {
      mbar_init(bar(base, B_TKI + p), 32);
      mbar_init(bar(base, B_TKF + p), 8);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (cta == 0 && threadIdx.x == 0) {
    unsigned long long g;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g));
    clk[0] = clock64();
    clk[1] = g;
  }
  if (warp < 8) {
    const int pw = warp, r = pw * 8 + lane / 4, ch = lane % 4, rx = r % 8;
    uint32_t d0, d1, dr;
    if (LAYOUT == 1) {
      d0 = r * 128 + ((ch ^ rx) << 4);
      d1 = r * 128 + (((ch + 4) ^ rx) << 4);
      dr = 4 * 8192 + r * 64 + ((ch ^ ((r >> 1) & 3)) << 4);
    } else {
      d0 = r * ROWB + 16 * ch;
      d1 = d0 + 64;
      dr = r * ROWB + 512 + 16 * ch;
    }
    const uint32_t astride = LAYOUT == 1 ? 8192u : 128u;
    int g = 0;
    for (int qi = 0; qi < n_mine; ++qi) {
      const int tok = cta + qi * grid;
      const int* ti = idx + (long)tok * KEYS + r;
      int c1 = 0, c2 = 0;
      uint32_t ib = base + OFF_IDX + (qi % 2) * KEYS * 4 + 4 * r;
      if (IDX == 0) {
        c1 = __ldg(ti);
        c2 = __ldg(ti + 64);
      } else {
        mbar_wait(bar(base, B_TKI + qi % 2), (qi / 2) & 1);
      }
      for (int t = 0; t < SUBS; ++t) {
        const int s = g % NS;
        const uint32_t st = base + s * STAGE;
        int rc;
        if (IDX == 0) {
          rc = c1;
          c1 = c2;
          c2 = t + 2 < SUBS ? __ldg(ti + (t + 2) * 64) : 0;
        } else {
          rc = (int)lds32(ib + 256 * t);
        }
        if (SYNC == 1) {
          if (lane == 0) {
            mbar_wait(bar(base, B_EMPTY + s), ((g / NS) & 1) ^ 1);
            if (g >= CAP) mbar_wait(bar(base, B_FULL + (g - CAP) % NS), ((g - CAP) / NS) & 1);
          }
          __syncwarp();
        } else if (SYNC == 2) {
          // per-warp in-flight cap on the warp's own copies + ring safety by the consumer's release
          if (lane == 0) mbar_wait(bar(base, B_EMPTY + s), ((g / NS) & 1) ^ 1);
          if (g >= CAP) mbar_wait(bar(base, B_FW + pw * NS + (g - CAP) % NS), ((g - CAP) / NS) & 1);
          __syncwarp();
        } else if (SYNC == 3) {
          // CTA-wide cap only (no consumer): lane 0 waits for stage g - CAP landed by every warp
          if (lane == 0 && g >= CAP) mbar_wait(bar(base, B_FULL + (g - CAP) % NS), ((g - CAP) / NS) & 1);
          __syncwarp();
        } else if (SYNC == 4) {
          // CTA-wide cap only, every lane waits
          if (g >= CAP) mbar_wait(bar(base, B_FULL + (g - CAP) % NS), ((g - CAP) / NS) & 1);
        } else {
          if (g >= CAP) mbar_wait(bar(base, B_FW + pw * NS + (g - CAP) % NS), ((g - CAP) / NS) & 1);
        }
        const uint8_t* src = kv + (long)rc * ROWB + 16 * ch;
#pragma unroll
        for (int a = 0; a < 4; ++a) {
          cp16<Z>(st + a * astride + d0, src + 128 * a, 16u);
          cp16<Z>(st + a * astride + d1, src + 128 * a + 64, 16u);
        }
        cp16<Z>(st + dr, src + 512, 16u);
        if (SYNC != 0) cp_arrive(bar(base, B_FULL + s));
        if (SYNC == 0 || SYNC == 2) cp_arrive(bar(base, B_FW + pw * NS + s));
        ++g;
      }
      if (IDX == 1) {
        __syncwarp();
        if (lane == 0) mbar_arrive(bar(base, B_TKF + qi % 2));
      }
    }
    // drain: wait for this warp's last stages
    for (int m = g > NS ? g - NS : 0; m < g; ++m)
      mbar_wait(bar(base, ((SYNC == 0 || SYNC == 2) ? B_FW + pw * NS : B_FULL) + m % NS), (m / NS) & 1);
    if (cta == 0 && lane == 0) clk[8 + pw] = clock64();
  } else if (warp == 8 && (SYNC == 1 || SYNC == 2)) {
    // consumer: free each stage as soon as it has landed
    const int total = n_mine * SUBS;
    for (int g = 0; g < total; ++g) {
      const int s = g % NS;
      if (FENCE == 2) {
        if (lane == 0) {
          uint32_t ok = 0;
          while (true) {
            asm volatile("{ .reg .pred p; mbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
                         : "=r"(ok) : "r"(bar(base, B_FULL + s)), "r"((uint32_t)((g / NS) & 1)) : "memory");
            if (ok) break;
            __nanosleep(64);
          }
          mbar_arrive(bar(base, B_EMPTY + s));
        }
      } else {
        mbar_wait(bar(base, B_FULL + s), (g / NS) & 1);
        if (FENCE) asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncwarp();
        if (lane == 0) mbar_arrive(bar(base, B_EMPTY + s));
      }
    }
  } else if (warp == 9 && IDX == 1) {
    // index warp: copy each token's 2048 indices into smem one token ahead
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
  }
  __syncthreads();
  if (cta == 0 && threadIdx.x == 0) {
    unsigned long long g;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g));
    clk[2] = clock64();
    clk[3] = g;
  }
}

template <int LAYOUT, int Z, int SYNC, int CAP, int FENCE, int IDX>
void run(const char* name, int nsm, int warps, const uint8_t* kv, const int* idx, unsigned long long* dclk) {
  // oversized-mode footprint like h8 (ring at 0-216 KB stays below the 227 KiB LDGSTS line)
  CUfunction f;
  cudaGetFuncBySymbol((cudaFunction_t*)&f, (const void*)lk<LAYOUT, Z, SYNC, CAP, FENCE, IDX>);
  cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_SHARED_MEMORY_MODE, CU_SHARED_MEMORY_MODE_ALLOW_OVERSIZED_SHARED_MEMORY);
  CUlaunchConfig cfg = {};
  cfg.gridDimX = nsm;
  cfg.gridDimY = cfg.gridDimZ = 1;
  cfg.blockDimX = 32 * warps;
  cfg.blockDimY = cfg.blockDimZ = 1;
  cfg.sharedMemBytes = 327 * 1024;
  CUlaunchAttribute la[1];
  la[0].id = CU_LAUNCH_ATTRIBUTE_SHARED_MEMORY_MODE;
  la[0].value.sharedMemoryMode = CU_SHARED_MEMORY_MODE_ALLOW_OVERSIZED_SHARED_MEMORY;
  cfg.attrs = la;
  cfg.numAttrs = 1;
  void* args[] = {(void*)&kv, (void*)&idx, (void*)&dclk};
  CUresult r0 = cuLaunchKernelEx(&cfg, f, args, nullptr);
  cudaEvent_t a, b;
  cudaEventCreate(&a);
  cudaEventCreate(&b);
  cudaEventRecord(a);
  const int reps = 5;
  for (int i = 0; i < reps; ++i) cuLaunchKernelEx(&cfg, f, args, nullptr);
  cudaEventRecord(b);
  cudaEventSynchronize(b);
  float ms;
  cudaEventElapsedTime(&ms, a, b);
  cudaError_t e = cudaGetLastError();
  if (r0 != CUDA_SUCCESS) e = cudaErrorLaunchFailure;
  unsigned long long h[16];
  cudaMemcpy(h, dclk, sizeof(h), cudaMemcpyDeviceToHost);
  unsigned long long wmin = ~0ull, wmax = 0;
  for (int w = 0; w < 8; ++w) {
    if (h[8 + w] < wmin) wmin = h[8 + w];
    if (h[8 + w] > wmax) wmax = h[8 + w];
  }
  const double spread = h[2] > h[0] ? 100.0 * (double)(wmax - wmin) / (double)(h[2] - h[0]) : 0.0;
  const double mhz = (h[3] > h[1]) ? (double)(h[2] - h[0]) / (double)(h[3] - h[1]) * 1000.0 : 0.0;
  const double us = ms * 1000 / reps, bytes = (double)NTOK * KEYS * ROWB;
  printf("%-34s W%2d: %7.1f us/layer %5.1f TB/s %5.1f B/clk/SM (clk %4.0f MHz) warp-finish spread %4.1f%% %s\n", name,
         warps, us, bytes / us / 1e6, bytes / (us * 1e-6) / nsm / (mhz * 1e6), mhz, spread,
         e == cudaSuccess ? "" : cudaGetErrorString(e));
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
  cudaMemset(kv, 0x11, (size_t)nrows * ROWB);
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
  printf("SMs %d, KV %d rows (%.1f MB), %ld token-keys x 576 B; smem %d B\n", nsm, nrows, nrows * ROWB / 1e6, nkeys,
         SMEM);
  for (int rep = 0; rep < 2; ++rep) {
    run<1, 1, 0, 4, 0, 1>("L1 lean h8 layout, smem idx", nsm, 10, kv, idx, dclk);
    run<1, 1, 3, 4, 0, 1>("C1 CTA cap 4 lane0, no consumer", nsm, 10, kv, idx, dclk);
    run<1, 1, 3, 6, 0, 1>("C2 CTA cap 6 lane0, no consumer", nsm, 10, kv, idx, dclk);
    run<1, 1, 4, 4, 0, 1>("C3 CTA cap 4 all lanes", nsm, 10, kv, idx, dclk);
    run<1, 1, 4, 2, 0, 1>("C4 CTA cap 2 all lanes", nsm, 10, kv, idx, dclk);
    run<1, 1, 1, 4, 1, 1>("R5 h8 producer (CTA cap 4)", nsm, 10, kv, idx, dclk);
  }
  return 0;
}
