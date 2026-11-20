// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// pmla15_deliv.cu (P-MLA round 6): random 576 B KV-row L2 -> SMEM delivery ceiling per mechanism MIX on sm_107a
// (coordinator 05:22 step 2: do LDGSTS + TMA gather4 / LDG / bulk add, or is every random-row form capped ~60 B/clk?).
// One F16K DSA layer = 16384 tokens x 2048 random rows of 576 B from a 78K-row KV (45 MB, L2-resident); persistent
// grid = #SMs, one CTA per SM. Stages of 64 rows (the h8 SW128 latent atoms + SW64 RoPE atom layout) are split
// between mechanisms by stage: X takes stages g % XP == XP - 1 (XP 1: all, 0: none), LDGSTS warps take the rest.
//   L: lean LDGSTS (pmla12 rung 0: 8 warps x 8 rows, 4 lanes/row, 9 x 16 B zero-fill cp.async per lane, per-warp
//      completion barriers, smem index row); nLg groups of 8 warps take alternate LDGSTS stages; ring 0-216 KB.
//   X: 1 TMA tile::gather4 (16 quads x 5 boxes per stage over nX warps) | 2 LDG.128 -> st.shared (8 warps x 8 rows)
//      | 3 1-D bulk copies of 8 contiguous rows (4608 B) | 4 1-D bulk copies of single 576 B rows; ring above 227 KiB
//      (TMA / st.shared are not slowed there, P-MLA-2CTA g10c).
// No consumer: each warp only waits for its own copies before reusing a slot. Reports us/layer, TB/s, B/clk/SM at the
// SM clock measured in CTA 0 (clock64 / globaltimer), and each group's finish time.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

constexpr int NS = 6, STAGE = 36864, KEYS = 2048, SUBS = KEYS / 64, NTOK = 16384, NSX = 2;
constexpr uint32_t OFF_BAR = NS * STAGE;      // 221184
constexpr uint32_t OFF_IDX = OFF_BAR + 2048;  // 2 x 8 KB token index rows
constexpr uint32_t OFF_X = 240640;            // X ring, 1024-aligned, above the 227 KiB LDGSTS line
constexpr uint32_t SMEM = OFF_X + NSX * STAGE;
constexpr int B_FW = 0, B_XB = 96, B_TKI = 112, B_TKF = 114;

struct Cfg {
  int nL, nLg, xk, nX, XP, pitch;
};

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
__device__ __forceinline__ void mbar_expect(uint32_t b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(b), "r"(bytes) : "memory");
}
__device__ __forceinline__ void cp_arrive(uint32_t b) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(b) : "memory");
}
__device__ __forceinline__ void cp16(uint32_t dst, const uint8_t* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(16u) : "memory");
}
__device__ __forceinline__ uint32_t lds32(uint32_t a) {
  uint32_t v;
  asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(a) : "memory");
  return v;
}
__device__ __forceinline__ int4 lds128(uint32_t a) {
  int4 v;
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a)
               : "memory");
  return v;
}
__device__ __forceinline__ uint4 ldg128(const uint8_t* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p)
               : "memory");
  return v;
}
__device__ __forceinline__ void sts128(uint32_t a, uint4 v) {
  asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" ::"r"(a), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
               : "memory");
}
__device__ __forceinline__ void gather4(uint32_t dst, const CUtensorMap* m, int x, int4 r, uint32_t mb) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];" ::"r"(dst),
      "l"(m), "r"(x), "r"(r.x), "r"(r.y), "r"(r.z), "r"(r.w), "r"(mb)
      : "memory");
}
__device__ __forceinline__ void bulk(uint32_t dst, const uint8_t* src, uint32_t n, uint32_t mb) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst),
               "l"(src), "r"(n), "r"(mb)
               : "memory");
}

__global__ void __launch_bounds__(800, 1) dk(Cfg c, const uint8_t* __restrict__ kv, const int* __restrict__ idx,
                                              int nrows, const __grid_constant__ CUtensorMap tmM,
                                              const __grid_constant__ CUtensorMap tmR, unsigned long long* clk) {
  extern __shared__ __align__(1024) unsigned char sm[];
  const uint32_t base = (uint32_t)__cvta_generic_to_shared(sm);
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int grid = gridDim.x, cta = blockIdx.x;
  const int n_mine = (NTOK - cta + grid - 1) / grid;
  const int nprod = c.nL + c.nX;
  if (threadIdx.x == 0) {
    for (int i = 0; i < 96; ++i) mbar_init(bar(base, B_FW + i), 32);
    for (int i = 0; i < 16; ++i) mbar_init(bar(base, B_XB + i), 1);
    for (int p = 0; p < 2; ++p) {
      mbar_init(bar(base, B_TKI + p), 32);
      mbar_init(bar(base, B_TKF + p), nprod);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  unsigned long long c0 = clock64();
  if (cta == 0 && threadIdx.x == 0) {
    unsigned long long g;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g));
    clk[0] = c0;
    clk[1] = g;
  }
  auto is_x = [&](int g) { return c.XP > 0 && (g % c.XP) == c.XP - 1; };
  if (warp < c.nL) {
    const int h = warp / 8, pw = warp % 8, SPG = NS / c.nLg;
    const int r = pw * 8 + lane / 4, ch = lane % 4, rx = r % 8;
    const uint32_t d0 = r * 128 + ((ch ^ rx) << 4), d1 = r * 128 + (((ch + 4) ^ rx) << 4);
    const uint32_t dr = 4 * 8192 + r * 64 + ((ch ^ ((r >> 1) & 3)) << 4);
    int m = 0, lidx = 0;
    for (int qi = 0; qi < n_mine; ++qi) {
      mbar_wait(bar(base, B_TKI + qi % 2), (qi / 2) & 1);
      const uint32_t ib = base + OFF_IDX + (qi % 2) * KEYS * 4 + 4 * r;
      for (int tt = 0; tt < SUBS; ++tt) {
        const int g = qi * SUBS + tt;
        if (is_x(g)) continue;
        const bool mine = (lidx % c.nLg) == h;
        ++lidx;
        if (!mine) continue;
        const int slot = h + c.nLg * (m % SPG);
        const uint32_t st = base + slot * STAGE;
        const int rc = (int)lds32(ib + 256 * tt);
        if (m >= SPG) mbar_wait(bar(base, B_FW + warp * NS + slot), ((m - SPG) / SPG) & 1);
        const uint8_t* src = kv + (long)rc * c.pitch + 16 * ch;
#pragma unroll
        for (int a = 0; a < 4; ++a) {
          cp16(st + a * 8192 + d0, src + 128 * a);
          cp16(st + a * 8192 + d1, src + 128 * a + 64);
        }
        cp16(st + dr, src + 512);
        cp_arrive(bar(base, B_FW + warp * NS + slot));
        ++m;
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(bar(base, B_TKF + qi % 2));
    }
    for (int k = m > SPG ? m - SPG : 0; k < m; ++k) {
      const int slot = h + c.nLg * (k % SPG);
      mbar_wait(bar(base, B_FW + warp * NS + slot), (k / SPG) & 1);
    }
    if (cta == 0 && lane == 0) clk[8 + warp] = clock64() - c0;
  } else if (warp < nprod) {
    const int xw = warp - c.nL;
    int xm = 0;
    for (int qi = 0; qi < n_mine; ++qi) {
      mbar_wait(bar(base, B_TKI + qi % 2), (qi / 2) & 1);
      const uint32_t ib = base + OFF_IDX + (qi % 2) * KEYS * 4;
      for (int tt = 0; tt < SUBS; ++tt) {
        const int g = qi * SUBS + tt;
        if (!is_x(g)) continue;
        const int slot = xm % NSX;
        const uint32_t st = base + OFF_X + slot * STAGE;
        const uint32_t mb = bar(base, B_XB + xw * NSX + slot);
        if (c.xk == 2) {
          // LDG.128 -> swizzled st.shared, 8 warps x 8 rows, all 9 loads of a lane before its stores
          const int r = xw * 8 + lane / 4, ch = lane % 4, rx = r % 8;
          const uint32_t d0 = r * 128 + ((ch ^ rx) << 4), d1 = r * 128 + (((ch + 4) ^ rx) << 4);
          const uint32_t dr = 4 * 8192 + r * 64 + ((ch ^ ((r >> 1) & 3)) << 4);
          const int rc = (int)lds32(ib + 256 * tt + 4 * r);
          const uint8_t* src = kv + (long)rc * c.pitch + 16 * ch;
          uint4 v[9];
#pragma unroll
          for (int a = 0; a < 4; ++a) {
            v[2 * a] = ldg128(src + 128 * a);
            v[2 * a + 1] = ldg128(src + 128 * a + 64);
          }
          v[8] = ldg128(src + 512);
#pragma unroll
          for (int a = 0; a < 4; ++a) {
            sts128(st + a * 8192 + d0, v[2 * a]);
            sts128(st + a * 8192 + d1, v[2 * a + 1]);
          }
          sts128(st + dr, v[8]);
        } else {
          if (xm >= NSX) mbar_wait(mb, ((xm - NSX) / NSX) & 1);
          __syncwarp();
          if (c.xk == 1) {
            const int qpw = 16 / c.nX;  // quads per warp
            if (lane == 0) mbar_expect(mb, qpw * 4 * 576);
            __syncwarp();
            if (lane < 5 * qpw) {
              const int q = xw * qpw + lane / 5, a = lane % 5;
              const int4 rr = lds128(ib + 256 * tt + 16 * q);
              if (a < 4)
                gather4(st + a * 8192 + q * 512, &tmM, 128 * a, rr, mb);
              else
                gather4(st + 4 * 8192 + q * 256, &tmR, 512, rr, mb);
            }
          } else if (c.xk == 3) {
            const int cpw = 8 / c.nX;  // 4608 B chunks per warp
            if (lane == 0) mbar_expect(mb, cpw * 4608);
            __syncwarp();
            if (lane < cpw) {
              const int j = xw * cpw + lane;
              int rc = (int)lds32(ib + 256 * tt + 4 * 8 * j);
              rc = rc > nrows - 8 ? nrows - 8 : rc;
              bulk(st + j * 4608, kv + (long)rc * c.pitch, 4608, mb);
            }
          } else {
            const int rpw = 64 / c.nX;  // single 576 B rows per warp
            if (lane == 0) mbar_expect(mb, rpw * 576);
            __syncwarp();
            for (int k = lane; k < rpw; k += 32) {
              const int j = xw * rpw + k;
              const int rc = (int)lds32(ib + 256 * tt + 4 * j);
              bulk(st + j * 576, kv + (long)rc * c.pitch, 576, mb);
            }
          }
        }
        ++xm;
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(bar(base, B_TKF + qi % 2));
    }
    if (c.xk != 2)
      for (int k = xm > NSX ? xm - NSX : 0; k < xm; ++k) mbar_wait(bar(base, B_XB + xw * NSX + k % NSX), (k / NSX) & 1);
    if (cta == 0 && lane == 0) clk[32 + xw] = clock64() - c0;
  } else if (warp == nprod) {
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

static CUtensorMap make_map(const void* kv, int nrows, int pitch, int box, CUtensorMapSwizzle sw) {
  CUtensorMap m;
  cuuint64_t dims[2] = {576, (cuuint64_t)nrows};
  cuuint64_t strides[1] = {(cuuint64_t)pitch};
  cuuint32_t boxd[2] = {(cuuint32_t)box, 1};
  cuuint32_t es[2] = {1, 1};
  CUresult r = cuTensorMapEncodeTiled(&m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, (void*)kv, dims, strides, boxd, es,
                                      CU_TENSOR_MAP_INTERLEAVE_NONE, sw, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                                      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) printf("cuTensorMapEncodeTiled failed %d (box %d pitch %d)\n", (int)r, box, pitch);
  return m;
}

static void run(const char* name, Cfg c, int nsm, const uint8_t* kv, const int* idx, int nrows,
                unsigned long long* dclk) {
  const CUtensorMap tmM = make_map(kv, nrows, c.pitch, 128, CU_TENSOR_MAP_SWIZZLE_128B);
  const CUtensorMap tmR = make_map(kv, nrows, c.pitch, 64, CU_TENSOR_MAP_SWIZZLE_64B);
  const int warps = c.nL + c.nX + 1;
  // oversized-mode launch (as pmla12): the X ring sits above the 227 KiB standard limit
  CUfunction f;
  cudaGetFuncBySymbol((cudaFunction_t*)&f, (const void*)dk);
  cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_SHARED_MEMORY_MODE, CU_SHARED_MEMORY_MODE_ALLOW_OVERSIZED_SHARED_MEMORY);
  CUlaunchConfig cfg = {};
  cfg.gridDimX = nsm;
  cfg.gridDimY = cfg.gridDimZ = 1;
  cfg.blockDimX = 32 * warps;
  cfg.blockDimY = cfg.blockDimZ = 1;
  cfg.sharedMemBytes = SMEM;
  CUlaunchAttribute la[1];
  la[0].id = CU_LAUNCH_ATTRIBUTE_SHARED_MEMORY_MODE;
  la[0].value.sharedMemoryMode = CU_SHARED_MEMORY_MODE_ALLOW_OVERSIZED_SHARED_MEMORY;
  cfg.attrs = la;
  cfg.numAttrs = 1;
  void* args[] = {(void*)&c, (void*)&kv, (void*)&idx, (void*)&nrows, (void*)&tmM, (void*)&tmR, (void*)&dclk};
  cudaMemset(dclk, 0, 512);
  CUresult r0 = cuLaunchKernelEx(&cfg, f, args, nullptr);
  cudaError_t e = cudaDeviceSynchronize();
  if (r0 != CUDA_SUCCESS || e != cudaSuccess) {
    printf("%-30s FAILED launch %d %s\n", name, (int)r0, cudaGetErrorString(e));
    exit(1);
  }
  cudaEvent_t a, b;
  cudaEventCreate(&a);
  cudaEventCreate(&b);
  const int reps = 5;
  cudaEventRecord(a);
  for (int i = 0; i < reps; ++i) cuLaunchKernelEx(&cfg, f, args, nullptr);
  cudaEventRecord(b);
  cudaEventSynchronize(b);
  float ms;
  cudaEventElapsedTime(&ms, a, b);
  unsigned long long hc[48];
  cudaMemcpy(hc, dclk, sizeof(hc), cudaMemcpyDeviceToHost);
  const double mhz = (hc[3] > hc[1]) ? (double)(hc[2] - hc[0]) / (double)(hc[3] - hc[1]) * 1000.0 : 0.0;
  const double us = ms * 1000 / reps, bytes = (double)NTOK * KEYS * 576;
  unsigned long long lmax = 0, xmax = 0;
  for (int w = 0; w < c.nL; ++w) lmax = hc[8 + w] > lmax ? hc[8 + w] : lmax;
  for (int w = 0; w < c.nX; ++w) xmax = hc[32 + w] > xmax ? hc[32 + w] : xmax;
  const double tot = (double)(hc[2] - hc[0]);
  printf("%-30s %7.1f us/layer %5.1f TB/s %5.1f B/clk/SM @%4.0f MHz | L done %5.1f%% X done %5.1f%% of CTA0\n", name,
         us, bytes / us / 1e6, bytes / (us * 1e-6) / nsm / (mhz * 1e6), mhz, tot > 0 ? 100.0 * lmax / tot : 0.0,
         tot > 0 ? 100.0 * xmax / tot : 0.0);
}

int main(int argc, char** argv) {
  int nsm = 0;
  cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
  cuInit(0);
  const int nrows = 78 * 1024;
  const long nkeys = (long)NTOK * KEYS;
  uint8_t *kv, *kv640;
  int* idx;
  unsigned long long* dclk;
  cudaMalloc(&kv, (size_t)nrows * 576);
  cudaMalloc(&kv640, (size_t)nrows * 640);
  {
    std::vector<uint32_t> hk((size_t)nrows * 640 / 4);
    uint32_t x = 12345u;
    for (auto& v : hk) {
      x = x * 1664525u + 1013904223u;
      v = x & 0xbfbfbfbfu;
    }
    cudaMemcpy(kv, hk.data(), (size_t)nrows * 576, cudaMemcpyHostToDevice);
    cudaMemcpy(kv640, hk.data(), (size_t)nrows * 640, cudaMemcpyHostToDevice);
  }
  std::vector<int> hi(nkeys);
  srand(1);
  for (long t = 0; t < NTOK; ++t) {
    const int ctx = nrows - NTOK + (int)t + 1;
    for (int k = 0; k < KEYS; ++k) hi[t * KEYS + k] = rand() % ctx;
  }
  cudaMalloc(&idx, nkeys * 4);
  cudaMemcpy(idx, hi.data(), nkeys * 4, cudaMemcpyHostToDevice);
  cudaMalloc(&dclk, 512);
  printf("SMs %d, KV %d rows (%.1f MB), %ld token-keys x 576 B, smem %u B\n", nsm, nrows, nrows * 576 / 1e6, nkeys,
         SMEM);
  struct Arm {
    const char* n;
    Cfg c;
    bool p640;
  };
  const Arm arms[] = {
      {"L8 (lean LDGSTS)", {8, 1, 0, 0, 0, 576}, false},
      {"L16 (2 groups)", {16, 2, 0, 0, 0, 576}, false},
      {"G4 gather4 only", {0, 1, 1, 4, 1, 576}, false},
      {"G8 gather4 only", {0, 1, 1, 8, 1, 576}, false},
      {"T8 LDG->STS only", {0, 1, 2, 8, 1, 576}, false},
      {"R4 bulk 576B rows only", {0, 1, 4, 4, 1, 576}, false},
      {"R8 bulk 576B rows only", {0, 1, 4, 8, 1, 576}, false},
      {"C4 bulk 4608B chunks only", {0, 1, 3, 4, 1, 576}, false},
      {"L8 + G4 1/4", {8, 1, 1, 4, 4, 576}, false},
      {"L8 + G4 1/3", {8, 1, 1, 4, 3, 576}, false},
      {"L8 + G4 1/2", {8, 1, 1, 4, 2, 576}, false},
      {"L8 + T8 1/4", {8, 1, 2, 8, 4, 576}, false},
      {"L8 + T8 1/3", {8, 1, 2, 8, 3, 576}, false},
      {"L8 + R8 1/4", {8, 1, 4, 8, 4, 576}, false},
      {"L8 + R8 1/3", {8, 1, 4, 8, 3, 576}, false},
      {"L16 + G4 1/3", {16, 2, 1, 4, 3, 576}, false},
      {"L8 pitch 640", {8, 1, 0, 0, 0, 640}, true},
  };
  const int only = argc > 1 ? atoi(argv[1]) : -1;
  for (int rep = 0; rep < 2; ++rep)
    for (int i = 0; i < (int)(sizeof(arms) / sizeof(arms[0])); ++i) {
      if (only >= 0 && i != only) continue;
      run(arms[i].n, arms[i].c, nsm, arms[i].p640 ? kv640 : kv, idx, nrows, dclk);
    }
  return 0;
}
