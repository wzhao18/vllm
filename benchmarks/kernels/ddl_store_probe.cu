// ddl_store_probe.cu: all-SM FP32 partial store / read rates for an L2-resident
// split-K exchange (D-DENSE-DL). 1 CTA per SM (large dynamic smem), 256 threads,
// each CTA stores or loads P bytes of its own slot with 16-byte accesses (a warp
// covers 512 contiguous bytes); %globaltimer stamps per CTA give the span.
#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

__device__ __forceinline__ uint64_t gtime() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// mode 0: store, 1: load (+sum), 2: load 4 quarter slices from 4 other CTAs' slots
__global__ void __launch_bounds__(256, 1) probe(float4* buf, uint64_t* st, int words, int mode,
                                               float* sink) {
  extern __shared__ char smem[];
  int b = blockIdx.x, nb = gridDim.x;
  __syncthreads();
  uint64_t t0 = gtime();
  float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
  if (mode == 0) {
    float4* p = buf + (size_t)b * words;
    float4 v = make_float4(b, 1.f, 2.f, 3.f);
    for (int i = threadIdx.x; i < words; i += 256) { v.x += 1.f; p[i] = v; }
  } else if (mode == 1) {
    const float4* p = buf + (size_t)((b + nb / 4) % nb) * words;
    for (int i = threadIdx.x; i < words; i += 256) {
      float4 v = p[i]; acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }
  } else {
    int q = words / 4;
    for (int k = 0; k < 4; ++k) {
      const float4* p = buf + (size_t)((b + 1 + k) % nb) * words + (size_t)k * q;
      for (int i = threadIdx.x; i < q; i += 256) {
        float4 v = p[i]; acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
      }
    }
  }
  __syncthreads();
  uint64_t t1 = gtime();
  if (threadIdx.x == 0) { st[2 * b] = t0; st[2 * b + 1] = t1; }
  if (acc.x + acc.y + acc.z + acc.w == 12345.f) sink[b] = acc.x;
}

int main() {
  int dev = 0, sms = 0;
  cudaSetDevice(dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  int smem = 160 * 1024;
  cudaFuncSetAttribute(probe, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  size_t maxp = 4 * 147456;
  float4* buf; uint64_t* st; float* sink;
  cudaMalloc(&buf, maxp * sms);
  cudaMalloc(&st, 16 * sms);
  cudaMalloc(&sink, 4 * sms);
  cudaMemset(buf, 0, maxp * sms);
  std::vector<uint64_t> h(2 * sms);
  int sizes[] = {36864, 73728, 147456, 294912};
  const char* names[] = {"store", "load", "load4x"};
  printf("SMs %d, 1 CTA/SM, 256 thr, 16B accesses, L2-resident slots\n", sms);
  for (int mode = 0; mode < 3; ++mode) {
    for (int si = 0; si < 4; ++si) {
      int P = sizes[si], words = P / 16;
      std::vector<double> spans, avg;
      for (int rep = 0; rep < 12; ++rep) {
        probe<<<sms, 256, smem>>>(buf, st, words, mode, sink);
        cudaDeviceSynchronize();
        cudaMemcpy(h.data(), st, 16 * sms, cudaMemcpyDeviceToHost);
        uint64_t lo = ~0ull, hi = 0; double s = 0;
        for (int b = 0; b < sms; ++b) {
          lo = std::min(lo, h[2 * b]); hi = std::max(hi, h[2 * b + 1]);
          s += double(h[2 * b + 1] - h[2 * b]);
        }
        if (rep >= 2) { spans.push_back((hi - lo) / 1e3); avg.push_back(s / sms / 1e3); }
      }
      std::sort(spans.begin(), spans.end()); std::sort(avg.begin(), avg.end());
      double sp = spans[spans.size() / 2], av = avg[avg.size() / 2];
      printf("%-7s P=%7d B/CTA total %6.1f MB: span %6.2f us (%5.1f GB/s/SM, %5.1f TB/s), "
             "mean CTA %6.2f us\n", names[mode], P, P * double(sms) / 1e6, sp, P / sp / 1e3,
             P * double(sms) / sp / 1e6, av);
    }
  }
  cudaError_t e = cudaGetLastError();
  printf("err %s\n", cudaGetErrorString(e));
  return 0;
}
