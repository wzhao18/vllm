// I-cache probe (D-TOPK-FP): a straight-line block of N independent IADD3 (16 B each) run
// twice in one launch; cycles of pass 0 (cold: fetched from L2/HBM) and pass 1 (warm if the
// block fits the SM's instruction cache). 1 warp/SM and 16 warps/SM.
#include <cstdio>
#include <cuda_runtime.h>
#define A1 asm volatile("mad.lo.u32 %0, %0, %1, %2;" : "+r"(x0) : "r"(y), "r"(z)); \
           asm volatile("mad.lo.u32 %0, %0, %1, %2;" : "+r"(x1) : "r"(y), "r"(z)); \
           asm volatile("mad.lo.u32 %0, %0, %1, %2;" : "+r"(x2) : "r"(y), "r"(z)); \
           asm volatile("mad.lo.u32 %0, %0, %1, %2;" : "+r"(x3) : "r"(y), "r"(z));
#define A16 A1 A1 A1 A1
#define A64 A16 A16 A16 A16
#define A256 A64 A64 A64 A64
#define A1K A256 A256 A256 A256
template <int NK>  // NK x 1024 instructions
__global__ void probe(long long* out) {
  unsigned x0 = threadIdx.x, x1 = 1, x2 = 2, x3 = 3, y = blockDim.x | 1u, z = threadIdx.x ^ 5u;
  long long t[3];
#pragma unroll 1
  for (int r = 0; r < 2; ++r) {
    t[r] = clock64();
    if constexpr (NK >= 1) { A1K }
    if constexpr (NK >= 2) { A1K }
    if constexpr (NK >= 4) { A1K A1K }
    if constexpr (NK >= 8) { A1K A1K A1K A1K }
    if constexpr (NK >= 16) { A1K A1K A1K A1K A1K A1K A1K A1K }
    __syncwarp();
  }
  t[2] = clock64();
  if ((threadIdx.x & 31) == 0)
    out[(blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32) * 3 + 0] = t[1] - t[0],
    out[(blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32) * 3 + 1] = t[2] - t[1],
    out[(blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32) * 3 + 2] = x0 + x1 + x2 + x3;
}
template <int NK>
void run(int sms, int warps, long long* d, long long* h) {
  for (int it = 0; it < 3; ++it) {  // it 0: code cold in L2/HBM; it 1,2: code in L2
    probe<NK><<<sms, 32 * warps>>>(d);
    cudaDeviceSynchronize();
  }
  cudaMemcpy(h, d, sizeof(long long) * 3 * sms * warps, cudaMemcpyDeviceToHost);
  double c0 = 0, c1 = 0;
  for (int i = 0; i < sms * warps; ++i) { c0 += h[3 * i]; c1 += h[3 * i + 1]; }
  c0 /= sms * warps; c1 /= sms * warps;
  printf("ICPROBE N=%5d instr (%4d KB) warps/SM=%2d pass0 %8.0f cyc pass1 %8.0f cyc | cyc/instr %.2f / %.2f | extra/line(128B) %.1f cyc\n",
         NK * 1024, NK * 16, warps, c0, c1, c0 / (NK * 1024.0), c1 / (NK * 1024.0), (c0 - c1) / (NK * 128.0));
}
int main() {
  int sms; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  long long *d, *h = new long long[3 * 212 * 32];
  cudaMalloc(&d, sizeof(long long) * 3 * 212 * 32);
  for (int w : {1, 16}) {
    run<1>(sms, w, d, h); run<2>(sms, w, d, h); run<4>(sms, w, d, h); run<8>(sms, w, d, h); run<16>(sms, w, d, h);
  }
  printf("ICPROBE done sms=%d\n", sms);
  return 0;
}
