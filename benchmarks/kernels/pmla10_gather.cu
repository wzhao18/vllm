// pmla10_gather.cu (P-MLA-2CTA round 2): LDGSTS (cp.async 16 B) random 576 B KV-row gather rate laws on sm_107a.
// One F16K DSA layer = 16384 tokens x 2048 rows of 576 B from a 78K-row KV (45 MB); persistent grid = #SMs, 1 CTA/SM.
// Variants: L lanes per row (4/8/16/32), completion by cp.async.wait_group (D groups of SPG 8-row sets in flight per
// warp) or per-slot mbarriers (cp.async.mbarrier.arrive.noinc, D sets in flight), dynamic smem footprint (oversized
// mode above 227 KiB: L1 carve-out effect), and the SM clock measured in CTA 0 (clock64 / globaltimer).
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

constexpr int SET = 4608;  // 8 rows x 576 B

__device__ __forceinline__ void wait_group(int n) {
  switch (n) {
    case 0: asm volatile("cp.async.wait_group 0;" ::: "memory"); break;
    case 1: asm volatile("cp.async.wait_group 1;" ::: "memory"); break;
    case 2: asm volatile("cp.async.wait_group 2;" ::: "memory"); break;
    case 3: asm volatile("cp.async.wait_group 3;" ::: "memory"); break;
    case 4: asm volatile("cp.async.wait_group 4;" ::: "memory"); break;
    case 5: asm volatile("cp.async.wait_group 5;" ::: "memory"); break;
    case 6: asm volatile("cp.async.wait_group 6;" ::: "memory"); break;
    default: asm volatile("cp.async.wait_group 7;" ::: "memory"); break;
  }
}

__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{ .reg .pred p; W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1; @!p bra W; }" ::"r"(bar), "r"(parity)
      : "memory");
}

template <int L, int MODE>
__global__ void __launch_bounds__(1024, 1) gk(const uint8_t* __restrict__ kv, const int* __restrict__ idx, long nsets,
                                               int D, int SPG, int off, unsigned long long* clk) {
  extern __shared__ __align__(128) unsigned char sm[];
  constexpr int R = 32 / L, NJ = (36 + L - 1) / L, RP = L / 4, PF = 4;
  const int lane = threadIdx.x % 32, w = threadIdx.x / 32, W = blockDim.x / 32;
  const long gw = (long)blockIdx.x * W + w, nw = (long)gridDim.x * W;
  const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(sm);
  const uint32_t ring = sbase + off + w * D * SPG * SET;
  // MODE 2: one CTA-wide barrier ring (32 W arrivals per phase) shared by all warps
  const uint32_t bars = sbase + off + W * D * SPG * SET + (MODE == 2 ? 0 : w * D * 8);
  if (MODE == 1 && lane == 0)
    for (int k = 0; k < D; ++k) asm volatile("mbarrier.init.shared::cta.b64 [%0], 32;" ::"r"(bars + 8 * k));
  if (MODE == 2 && threadIdx.x == 0)
    for (int k = 0; k < D; ++k) asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bars + 8 * k), "r"(32 * W));
  const long npad = MODE == 2 ? (nsets + nw * PF - 1) / (nw * PF) * (nw * PF) : nsets;
  __syncthreads();
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    unsigned long long g;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g));
    clk[0] = clock64();
    clk[1] = g;
  }
  int ix[PF][RP];
#pragma unroll
  for (int k = 0; k < PF; ++k) {
    const long t = gw + k * nw;
#pragma unroll
    for (int i = 0; i < RP; ++i) ix[k][i] = t < nsets ? __ldg(idx + t * 8 + lane / L + R * i) : 0;
  }
  long n = 0;  // sets issued by this warp
  long long isum = 0;
  for (long s0 = gw; s0 < npad; s0 += PF * nw) {
#pragma unroll
    for (int k = 0; k < PF; ++k) {
      const long s = s0 + k * nw;
      if (s < nsets || MODE == 2) {
        const int slot = MODE >= 1 ? (int)(n % D) : (int)(n % (D * SPG));
        if (MODE >= 1 && n >= D) mbar_wait(bars + 8 * slot, (uint32_t)((n / D - 1) & 1));
        const uint32_t dst = ring + slot * SET;
        const long long ci = clock64();
#pragma unroll
        for (int i = 0; i < RP; ++i) {
          const int r = lane / L + R * i;
          const uint8_t* src = kv + (size_t)ix[k][i] * 576;
#pragma unroll
          for (int j = 0; j < NJ; ++j) {
            const int c = lane % L + L * j;
            if (c < 36)
              asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst + r * 576 + 16 * c), "l"(src + 16 * c)
                           : "memory");
          }
        }
        if (blockIdx.x == 0 && w == 0 && lane == 0) isum += clock64() - ci;  // warp issue time of one set
        if (MODE >= 1) {
          asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(bars + 8 * slot) : "memory");
        } else if (n % SPG == SPG - 1) {
          asm volatile("cp.async.commit_group;" ::: "memory");
          wait_group(D - 1);
        }
        ++n;
      }
      const long t = s + PF * nw;
#pragma unroll
      for (int i = 0; i < RP; ++i) ix[k][i] = t < nsets ? __ldg(idx + t * 8 + lane / L + R * i) : 0;
    }
  }
  if (MODE >= 1) {
    for (long m = n > D ? n - D : 0; m < n; ++m) mbar_wait(bars + 8 * (int)(m % D), (uint32_t)((m / D) & 1));
  } else {
    asm volatile("cp.async.commit_group;" ::: "memory");
    asm volatile("cp.async.wait_group 0;" ::: "memory");
  }
  __syncthreads();
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    unsigned long long g;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g));
    clk[2] = clock64();
    clk[3] = g;
    clk[4] = isum;
    clk[5] = n;
  }
}

// SIMT shared-memory rate at a byte offset: 8 warps, st.shared.v4 / ld.shared.v4 over a 32 KB window
template <int ST>
__global__ void __launch_bounds__(256, 1) smk(int off, int iters, unsigned long long* clk) {
  extern __shared__ __align__(128) unsigned char sm[];
  const uint32_t b = (uint32_t)__cvta_generic_to_shared(sm) + off + 16 * threadIdx.x;
  uint32_t x = threadIdx.x, acc = 0;
  __syncthreads();
  const long long c0 = clock64();
  for (int i = 0; i < iters; ++i) {
    const uint32_t a = b + ((i * 4096) & 32767);
    if (ST) {
      asm volatile("st.shared.v4.u32 [%0], {%1,%1,%1,%1};" ::"r"(a), "r"(x + i) : "memory");
    } else {
      uint32_t v0, v1, v2, v3;
      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v0), "=r"(v1), "=r"(v2), "=r"(v3) : "r"(a) : "memory");
      acc += v0 ^ v3;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0 && blockIdx.x == 0) clk[6] = clock64() - c0;
  if (acc == 0x12345u) clk[7] = acc;
}

template <int ST>
void runs(int nsm, int off, unsigned long long* dclk) {
  CUfunction f;
  cudaGetFuncBySymbol((cudaFunction_t*)&f, (const void*)smk<ST>);
  cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_SHARED_MEMORY_MODE, CU_SHARED_MEMORY_MODE_ALLOW_OVERSIZED_SHARED_MEMORY);
  CUlaunchConfig cfg = {};
  cfg.gridDimX = nsm;
  cfg.gridDimY = cfg.gridDimZ = 1;
  cfg.blockDimX = 256;
  cfg.blockDimY = cfg.blockDimZ = 1;
  cfg.sharedMemBytes = 327 * 1024;
  CUlaunchAttribute la[1];
  la[0].id = CU_LAUNCH_ATTRIBUTE_SHARED_MEMORY_MODE;
  la[0].value.sharedMemoryMode = CU_SHARED_MEMORY_MODE_ALLOW_OVERSIZED_SHARED_MEMORY;
  cfg.attrs = la;
  cfg.numAttrs = 1;
  int iters = 4096;
  void* args[] = {(void*)&off, (void*)&iters, (void*)&dclk};
  cuLaunchKernelEx(&cfg, f, args, nullptr);
  cudaDeviceSynchronize();
  unsigned long long h[8];
  cudaMemcpy(h, dclk, sizeof(h), cudaMemcpyDeviceToHost);
  printf("%s.shared.v4 8 warps window at %3d KB: %6.1f B/clk/SM\n", ST ? "st" : "ld", off / 1024,
         256.0 * 16 * iters / (double)h[6]);
}

template <int L, int MODE>
void run(int nsm, int W, int D, int SPG, size_t smem_min, int off, const uint8_t* kv, const int* idx, long nsets,
         unsigned long long* dclk) {
  const size_t need = (size_t)off + (size_t)W * D * SPG * SET + (size_t)W * D * 8;
  const size_t smb = need > smem_min ? need : smem_min;
  if (smb > 334848) {
    printf("L%-2d M%d W%2d D%d SPG%d: smem %zu too big\n", L, MODE, W, D, SPG, smb);
    return;
  }
  CUfunction f;
  cudaGetFuncBySymbol((cudaFunction_t*)&f, (const void*)gk<L, MODE>);
  const bool over = smb > 232448;
  if (over)
    cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_SHARED_MEMORY_MODE, CU_SHARED_MEMORY_MODE_ALLOW_OVERSIZED_SHARED_MEMORY);
  else
    cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)smb);
  CUlaunchConfig cfg = {};
  cfg.gridDimX = nsm;
  cfg.gridDimY = cfg.gridDimZ = 1;
  cfg.blockDimX = 32 * W;
  cfg.blockDimY = cfg.blockDimZ = 1;
  cfg.sharedMemBytes = (unsigned)smb;
  CUlaunchAttribute la[1];
  cfg.attrs = la;
  cfg.numAttrs = 0;
  if (over) {
    la[0].id = CU_LAUNCH_ATTRIBUTE_SHARED_MEMORY_MODE;
    la[0].value.sharedMemoryMode = CU_SHARED_MEMORY_MODE_ALLOW_OVERSIZED_SHARED_MEMORY;
    cfg.numAttrs = 1;
  }
  void* args[] = {(void*)&kv, (void*)&idx, (void*)&nsets, (void*)&D, (void*)&SPG, (void*)&off, (void*)&dclk};
  cudaEvent_t a, b;
  cudaEventCreate(&a);
  cudaEventCreate(&b);
  CUresult r0 = cuLaunchKernelEx(&cfg, f, args, nullptr);
  cudaEventRecord(a);
  for (int r = 0; r < 5; ++r) cuLaunchKernelEx(&cfg, f, args, nullptr);
  cudaEventRecord(b);
  cudaEventSynchronize(b);
  float ms;
  cudaEventElapsedTime(&ms, a, b);
  cudaError_t e = cudaGetLastError();
  unsigned long long h[6];
  cudaMemcpy(h, dclk, sizeof(h), cudaMemcpyDeviceToHost);
  const double mhz = (h[3] > h[1]) ? (double)(h[2] - h[0]) / (double)(h[3] - h[1]) * 1000.0 : 0.0;
  const double us = ms * 1000 / 5, bytes = (double)nsets * SET;
  printf("L%-2d M%d W%2d D%d SPG%d off %3d KB smem %3zu KB%s: %7.1f us/layer %5.1f TB/s %5.1f B/clk/SM (clk %4.0f MHz) issue %5.0f clk/set %s%s\n", L,
         MODE, W, D, SPG, off / 1024, smb / 1024, over ? "*" : " ", us, bytes / us / 1e6, bytes / (us * 1e-6) / nsm / (mhz * 1e6),
         mhz, h[5] ? (double)h[4] / h[5] : 0.0, r0 == CUDA_SUCCESS ? "" : "launch-fail ", e == cudaSuccess ? "" : cudaGetErrorString(e));
}

int main(int argc, char** argv) {
  int nsm = 0;
  cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
  const int nrows = 78 * 1024;
  const long nkeys = 16384L * 2048, nsets = nkeys / 8;
  uint8_t* kv;
  int* idx;
  unsigned long long* dclk;
  cudaMalloc(&kv, (size_t)nrows * 576);
  cudaMemset(kv, 0x11, (size_t)nrows * 576);
  std::vector<int> hi(nkeys);
  srand(1);
  for (long t = 0; t < 16384; ++t) {
    const int ctx = nrows - 16384 + (int)t + 1;
    for (int k = 0; k < 2048; ++k) hi[t * 2048 + k] = rand() % ctx;
  }
  cudaMalloc(&idx, nkeys * 4);
  cudaMemcpy(idx, hi.data(), nkeys * 4, cudaMemcpyHostToDevice);
  cudaMalloc(&dclk, 64);
  cuInit(0);
  printf("SMs %d, KV %d rows (%.1f MB), %ld token-keys (576 B each); SOL 643, target 707 us/layer\n", nsm, nrows,
         nrows * 576 / 1e6, nkeys);
  const size_t O = 327 * 1024;  // oversized footprint for every arm
  // LDGSTS destination below vs above the 227 KiB standard shared-memory limit (232448 B)
  for (int off : {0, 100, 190, 236, 280}) {
    runs<1>(nsm, off * 1024, dclk);
    runs<0>(nsm, off * 1024, dclk);
  }
  run<4, 2>(nsm, 8, 2, 1, O, 152 * 1024, kv, idx, nsets, dclk);  // ring 152-226 KB (standard)
  run<4, 2>(nsm, 8, 2, 1, O, 200 * 1024, kv, idx, nsets, dclk);  // ring 200-274 KB (half oversized)
  return 0;
}
