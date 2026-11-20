// I-cache capacity probe for VR200 (P-KDA2-S6): cycles per warp-instruction of
// straight-line FADD blocks vs code footprint, 1 CTA on 1 SM.
//  - mode 0: W warps (W/4 per SMSP) all run the SAME block of N instructions
//  - mode 1: W warps, every warp runs its OWN copy (footprint W x N)
// Each block = N/8 groups of 8 independent FADD (8 accumulators), so the
// issue-limited CPI is ~1 per warp when the block fits in the cache.
// Build: nvcc -O3 -arch=sm_107a -o icache icache_probe.cu ; run: ./icache
#include <cstdio>
#include <cuda_runtime.h>

#define F8                                                   \
  asm volatile(                                              \
      "add.f32 %0,%0,%8;add.f32 %1,%1,%8;add.f32 %2,%2,%8;"  \
      "add.f32 %3,%3,%8;add.f32 %4,%4,%8;add.f32 %5,%5,%8;"  \
      "add.f32 %6,%6,%8;add.f32 %7,%7,%8;"                   \
      : "+f"(a0), "+f"(a1), "+f"(a2), "+f"(a3), "+f"(a4), "+f"(a5), "+f"(a6), \
        "+f"(a7)                                             \
      : "f"(y));
#define F64 F8 F8 F8 F8 F8 F8 F8 F8
#define F512 F64 F64 F64 F64 F64 F64 F64 F64

template <int ID, int G512>
__device__ __noinline__ float blk(float y, int iters) {
  float a0 = ID, a1 = 1, a2 = 2, a3 = 3, a4 = 4, a5 = 5, a6 = 6, a7 = 7;
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int g = 0; g < G512; ++g) { F512 }
  }
  return a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}

template <int G512>
__device__ float run(int id, float y, int iters) {
  switch (id) {
    case 0: return blk<0, G512>(y, iters);
    case 1: return blk<1, G512>(y, iters);
    case 2: return blk<2, G512>(y, iters);
    case 3: return blk<3, G512>(y, iters);
    case 4: return blk<4, G512>(y, iters);
    case 5: return blk<5, G512>(y, iters);
    case 6: return blk<6, G512>(y, iters);
    case 7: return blk<7, G512>(y, iters);
    default: return blk<8, G512>(y, iters);
  }
}

template <int G512>
__global__ void k(float* out, long long* cyc, int mode, int nw, int iters) {
  int w = threadIdx.x >> 5;
  if (w >= nw) return;
  int id = mode == 0 ? 0 : (w % 9);
  long long t0 = clock64();
  float r = run<G512>(id, out[0], iters);
  long long t1 = clock64();
  if ((threadIdx.x & 31) == 0) cyc[w] = t1 - t0;
  if (r == 12345.f) out[1] = r;
}

template <int G512>
void meas(float* d, long long* c, int mode, int nw) {
  int iters = 4096 / G512;
  k<G512><<<1, nw * 32>>>(d, c, mode, nw, iters);  // warm
  k<G512><<<1, nw * 32>>>(d, c, mode, nw, iters);
  cudaDeviceSynchronize();
  long long h[64];
  cudaMemcpy(h, c, nw * sizeof(long long), cudaMemcpyDeviceToHost);
  double s = 0;
  for (int i = 0; i < nw; ++i) s += h[i];
  double inst = (double)iters * G512 * 512;
  printf("mode %d warps %2d block %5d KiB/warp footprint %6d KiB  CPI/warp %.2f  SMSP issue %.2f\n",
         mode, nw, G512 * 8, mode == 0 ? G512 * 8 : G512 * 8 * (nw < 9 ? nw : 9),
         s / nw / inst, inst * nw / (s / nw) / (nw < 4 ? nw : 4));
}

int main() {
  float* d; long long* c;
  cudaMalloc(&d, 64); cudaMalloc(&c, 64 * sizeof(long long));
  cudaMemset(d, 0, 64);
  int ws[] = {1, 4, 8, 32};
  for (int m = 0; m < 2; ++m)
    for (int wi = 0; wi < 4; ++wi) {
      int nw = ws[wi];
      if (m == 1 && nw == 1) continue;
      meas<1>(d, c, m, nw); meas<2>(d, c, m, nw); meas<4>(d, c, m, nw);
      meas<8>(d, c, m, nw);
    }
  printf("err %s\n", cudaGetErrorString(cudaGetLastError()));
  return 0;
}
