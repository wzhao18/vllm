// I-cache sharing probe for VR200 (P-KDA2-S6 r7): SM issue rate of 32 warps
// (8 groups of 4 warps, one warp per SMSP per group, like the fip roles) when
// the groups run straight-line blocks larger than the fast I-cache and share
// them according to a group -> block map.  Each block = groups of 8
// independent FADD, so the issue-limited CPI is ~1 per warp when fetch keeps up.
// Build: nvcc -O3 -arch=sm_107a -o icache2 icache_probe2.cu ; run: ./icache2
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
    default: return blk<7, G512>(y, iters);
  }
}

struct Map { int g[8]; };

template <int G512>
__global__ void k(float* out, long long* cyc, Map m, int iters) {
  int w = threadIdx.x >> 5;
  int id = m.g[w >> 2];
  long long t0 = clock64();
  float r = 0.f;
  if (id >= 0) r = run<G512>(id, out[0], iters);
  long long t1 = clock64();
  if ((threadIdx.x & 31) == 0) cyc[w] = id >= 0 ? t1 - t0 : -1;
  if (r == 12345.f) out[1] = r;
}

template <int G512>
void meas(float* d, long long* c, const char* name, Map m) {
  int iters = 8192 / G512;
  k<G512><<<1, 1024>>>(d, c, m, iters);  // warm
  k<G512><<<1, 1024>>>(d, c, m, iters);
  cudaDeviceSynchronize();
  long long h[32];
  cudaMemcpy(h, c, 32 * sizeof(long long), cudaMemcpyDeviceToHost);
  double inst = (double)iters * G512 * 512, tot = 0, mx = 0;
  int act = 0;
  for (int i = 0; i < 32; ++i)
    if (h[i] > 0) { tot += inst; act++; if (h[i] > mx) mx = h[i]; }
  printf("%-34s block %3d KiB warps %2d  SMSP issue %.3f  (max cyc %.0f)\n", name,
         G512 * 8, act, tot / mx / 4.0, mx);
}

int main() {
  float* d; long long* c;
  cudaMalloc(&d, 64); cudaMalloc(&c, 32 * sizeof(long long));
  cudaMemset(d, 0, 64);
  // group g = warps 4g..4g+3 (one per SMSP); value = block id, -1 idle
  const char* nm[] = {
    "K1 all groups same block", "K2 two blocks x4 groups", "K3 3/3/2 groups",
    "K4 four blocks x2 groups", "K5 DP: 3 chain + E(4 grp) + S",
    "K5 eq: 3 chain + 5 prep-ish", "K8 today: every group own block",
    "K6 DP2: 3 chain + E(4) + S0 + S1", "K4 chain-shared + E + S", "K2 chain(4) + E(4)"};
  Map ms[] = {
    {{0, 0, 0, 0, 0, 0, 0, 0}}, {{0, 1, 0, 1, 0, 1, 0, 1}},
    {{0, 1, 2, 0, 1, 2, 0, 1}}, {{0, 1, 2, 3, 0, 1, 2, 3}},
    {{0, 1, 2, 3, 3, 3, 3, 4}}, {{0, 1, 2, 3, 4, 3, 4, 3}},
    {{0, 1, 2, 3, 4, 5, 6, 7}}, {{0, 1, 2, 3, 3, 3, 4, 5}},
    {{0, 0, 0, 3, 3, 3, 3, 4}}, {{0, 0, 0, 0, 3, 3, 3, 3}}};
  for (int i = 0; i < 10; ++i) {
    meas<1>(d, c, nm[i], ms[i]);
    meas<2>(d, c, nm[i], ms[i]);
    meas<4>(d, c, nm[i], ms[i]);
  }
  printf("err %s\n", cudaGetErrorString(cudaGetLastError()));
  return 0;
}
