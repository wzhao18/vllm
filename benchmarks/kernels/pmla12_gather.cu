// pmla12_gather.cu (P-MLA-2CTA round 4): per-warp cp.async (LDGSTS) issue cap at the h8 operating point.
// pmla10 (g10a): with 8 issuing warps the random 576 B row gather caps at ~40 B/clk/SM whatever the depth (D1-D6),
// 16-32 warps reach the ~60 B/clk/SM L2 ceiling. h8 has 8 copying warps (640-thread cap). Questions:
//   (1) is the cap specific to cp.async.cg (L1 bypass)? -> .ca arm
//   (2) does the h8 zero-fill form (src-size register operand) cost extra? -> zfill arm
//   (3) rate vs issuing warps 8..16 at the h8 depth (CTA-wide barrier ring, 3 stages in flight)
//   (4) helpers: 8 main warps carry 8/9 of every set, H helper warps the 9th 64 B segment (h8: RoPE by other warps)
// One F16K DSA layer = 16384 tokens x 2048 rows of 576 B from a 78K-row KV (46 MB); grid = #SMs, 1 CTA/SM.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

constexpr int SET = 4608;  // 8 rows x 576 B

__device__ __forceinline__ int hrow(long k, int nrows) {
  unsigned long long x = (unsigned long long)k * 0x9E3779B97F4A7C15ull;
  x ^= x >> 29;
  return (int)(x % (unsigned long long)nrows);
}

__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{ .reg .pred p; W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1; @!p bra W; }" ::"r"(bar), "r"(parity)
      : "memory");
}

template <int CA, int ZF>
__device__ __forceinline__ void cp16(uint32_t dst, const uint8_t* src, uint32_t nb) {
  if (CA) {
    if (ZF)
      asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(nb) : "memory");
    else
      asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
  } else {
    if (ZF)
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(nb) : "memory");
    else
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
  }
}

// Warps 0..W-1 issue full sets (8 rows x 9 segments of 64 B, 4 lanes per row: 9 cp.async per lane per set).
// HELP > 0: warps 0..7 issue segments 0-7 of their sets, warps 8..8+HELP-1 issue segment 8 of every main warp's set
// (each helper covers 8 / HELP main warps). Completion: one CTA-wide mbarrier per ring slot (all issuing lanes
// arrive with cp.async.mbarrier.arrive.noinc), D slots per main warp, like the h8 KVFULL ring.
template <int CA, int ZF>
__global__ void __launch_bounds__(1024, 1) gk(const uint8_t* __restrict__ kv, const int* __restrict__ idx, long nsets,
                                               int W, int D, int HELP, int nrows, unsigned long long* clk) {
  extern __shared__ __align__(128) unsigned char sm[];
  const int lane = threadIdx.x % 32, w = threadIdx.x / 32;
  const int MW = HELP ? 8 : W;  // main warps (own sets)
  const long nw = (long)gridDim.x * MW;
  const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(sm);
  const uint32_t bars = sbase + MW * D * SET;
  const int arrivals = 32 * (MW + HELP);
  if (threadIdx.x == 0)
    for (int k = 0; k < D; ++k) asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bars + 8 * k), "r"(arrivals));
  __syncthreads();
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    unsigned long long g;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g));
    clk[0] = clock64();
    clk[1] = g;
  }
  const bool helper = HELP && w >= 8;
  if (w < MW || helper) {
    const int r = lane / 4, ch = lane % 4;
    // padded so that every warp runs the same number of rounds (one barrier phase per round and slot)
    const long rounds = (nsets + nw - 1) / nw;
    for (long n = 0; n < rounds; ++n) {
      const int slot = (int)(n % D);
      if (n >= D) mbar_wait(bars + 8 * slot, (uint32_t)((n / D - 1) & 1));
      if (!helper) {
        const long s = (long)blockIdx.x * MW + w + n * nw;
        const int row = hrow(s * 8 + r, nrows);
        const uint8_t* src = kv + (size_t)row * 576 + 16 * ch;
        const uint32_t dst = sbase + (w * D + slot) * SET + r * 576 + 16 * ch;
        const uint32_t nb = s < nsets ? 16u : 0u;
        const int nseg = HELP ? 8 : 9;
#pragma unroll
        for (int j = 0; j < 9; ++j)
          if (j < nseg) cp16<CA, ZF>(dst + 64 * j, src + 64 * j, nb);
      } else {
        const int per = 8 / HELP;  // main warps served by this helper
        for (int m = (w - 8) * per; m < (w - 8 + 1) * per; ++m) {
          const long s = (long)blockIdx.x * MW + m + n * nw;
          const int row = hrow(s * 8 + r, nrows);
          const uint8_t* src = kv + (size_t)row * 576 + 16 * ch;
          const uint32_t dst = sbase + (m * D + slot) * SET + r * 576 + 16 * ch;
          cp16<CA, ZF>(dst + 512, src + 512, s < nsets ? 16u : 0u);
        }
      }
      asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(bars + 8 * slot) : "memory");
    }
    for (long m = rounds > D ? rounds - D : 0; m < rounds; ++m)
      mbar_wait(bars + 8 * (int)(m % D), (uint32_t)((m / D) & 1));
  }
  __syncthreads();
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    unsigned long long g;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g));
    clk[2] = clock64();
    clk[3] = g;
  }
}

template <int CA, int ZF>
void run(const char* tag, int nsm, int W, int D, int HELP, const uint8_t* kv, const int* idx, long nsets,
         unsigned long long* dclk) {
  const int MW = HELP ? 8 : W, T = HELP ? 8 + HELP : W;
  const size_t smb = (size_t)MW * D * SET + 8 * D + 64;
  CUfunction f;
  cudaGetFuncBySymbol((cudaFunction_t*)&f, (const void*)gk<CA, ZF>);
  cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)smb);
  CUlaunchConfig cfg = {};
  cfg.gridDimX = nsm;
  cfg.gridDimY = cfg.gridDimZ = 1;
  cfg.blockDimX = 32 * T;
  cfg.blockDimY = cfg.blockDimZ = 1;
  cfg.sharedMemBytes = (unsigned)smb;
  int nrows = 78 * 1024;
  void* args[] = {(void*)&kv, (void*)&idx, (void*)&nsets, (void*)&W, (void*)&D, (void*)&HELP, (void*)&nrows, (void*)&dclk};
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
  unsigned long long h[4];
  cudaMemcpy(h, dclk, sizeof(h), cudaMemcpyDeviceToHost);
  const double mhz = (h[3] > h[1]) ? (double)(h[2] - h[0]) / (double)(h[3] - h[1]) * 1000.0 : 0.0;
  const double us = ms * 1000 / 5, bytes = (double)nsets * SET;
  printf("%-10s W%2d D%d help%d smem %3zu KB: %7.1f us/layer %5.1f TB/s %5.1f B/clk/SM (clk %4.0f MHz)%s%s\n", tag, W, D,
         HELP, smb / 1024, us, bytes / us / 1e6, bytes / (us * 1e-6) / nsm / (mhz * 1e6), mhz,
         r0 == CUDA_SUCCESS ? "" : " launch-fail", e == cudaSuccess ? "" : cudaGetErrorString(e));
}

int main() {
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
  printf("SMs %d, KV %d rows (%.1f MB), %ld token-keys (576 B each)\n", nsm, nrows, nrows * 576 / 1e6, nkeys);
  for (int rep = 0; rep < 2; ++rep) {
    run<0, 1>("cg-zfill", nsm, 8, 3, 0, kv, idx, nsets, dclk);
    run<0, 1>("cg-zfill", nsm, 8, 5, 0, kv, idx, nsets, dclk);
    run<1, 1>("ca-zfill", nsm, 8, 3, 0, kv, idx, nsets, dclk);
    for (int W : {9, 10, 11, 12, 14, 16, 20, 24}) run<0, 1>("cg-zfill", nsm, W, W > 16 ? 2 : 3, 0, kv, idx, nsets, dclk);
    for (int H : {2, 4, 8}) run<0, 1>("cg-zf-help", nsm, 8, 3, H, kv, idx, nsets, dclk);
  }
  return 0;
}
