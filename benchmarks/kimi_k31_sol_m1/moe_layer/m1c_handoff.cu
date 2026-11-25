// M1 probes (c) and (b) for moe-layer.md. Not production code.
// (c) in-kernel cross-CTA hand-off, publish -> observe, by ping-pong between two CTAs on different SMs (one-way = round
//     trip / 2; globaltimer span over ITERS round trips, int64 math). Forms:
//     0 data-as-flag: every 16 B of the payload carries the epoch (Lamport form, no separate flag, no fence); the reader
//       polls its own 16 B words until all carry the epoch (all 128 threads), then __syncthreads.
//     1 fenced flag: writers st payload, __syncthreads, thread 0 fence.acq_rel.gpu + st.release.gpu flag; reader thread 0
//       ld.acquire.gpu spin, __syncthreads, all threads read the payload (fenced-flag form).
//     2 counter release: every writer thread red.release.gpu.add after its own stores; reader waits count (KDA X1-X8 form).
// (b) post-wait slowdown: a consumer CTA (all warps polling a flag, as WaveMoeTail) waits W us for a producer CTA on
//     another SM, then runs a fixed post-landing chain (8 rounds: STS, bar, LDS, 64 FMA, shuffle-reduce, bar). The chain is
//     timed in clock64 cycles only (no %globaltimer anywhere in the consumer unless the arm reads it). Arms:
//     P0 tight poll; P1 tight poll + the K31 spin-guard tick (1024 failed polls then a %globaltimer read, timeout path to
//     printf/trap compiled in, never taken); P2 nanosleep(64) backoff; P3 backoff + guard tick.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#define RK(x) do { cudaError_t r_ = (x); if (r_ != cudaSuccess) { printf("CUDA error %s line %d\n", cudaGetErrorString(r_), __LINE__); exit(1); } } while (0)
__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ unsigned smid() { unsigned r; asm volatile("mov.u32 %0, %%smid;" : "=r"(r)); return r; }
__device__ __forceinline__ void st4(uint4* p, uint4 v) {
  asm volatile("st.relaxed.gpu.global.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
__device__ __forceinline__ uint4 ld4(const uint4* p) {
  uint4 v; asm volatile("ld.relaxed.gpu.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ unsigned ld_acq(const unsigned* p) { unsigned v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void st_rel(unsigned* p, unsigned v) { asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }

// payload layout: box[side] = n16 x uint4, flag[side], cnt[side]
__device__ void publish(int form, uint4* box, unsigned* flag, unsigned* cnt, int n16, unsigned e) {
  const uint4 v = make_uint4(e, e ^ 0x5a5a5a5au, e + 7u, e * 3u);
  if (form == 0) { for (int i = threadIdx.x; i < n16; i += blockDim.x) { uint4 w = v; w.x = e; st4(box + i, make_uint4(e, e, e, e)); } return; }
  for (int i = threadIdx.x; i < n16; i += blockDim.x) box[i] = v;
  if (form == 1) {
    __syncthreads();
    if (threadIdx.x == 0) { asm volatile("fence.acq_rel.gpu;" ::: "memory"); st_rel(flag, e); }
  } else {
    asm volatile("red.release.gpu.global.add.u32 [%0], 1;" :: "l"(cnt) : "memory");
  }
}
__device__ unsigned observe(int form, const uint4* box, const unsigned* flag, const unsigned* cnt, int n16, unsigned e) {
  unsigned acc = 0;
  if (form == 0) {
    // all of this thread's 16 B words in flight per pass (<= 16 per thread at 32 KiB), re-poll until every tag matches
    bool ok;
    do {
      ok = true; uint4 w[16]; const int nmine = (n16 - (int)threadIdx.x + (int)blockDim.x - 1) / (int)blockDim.x;
#pragma unroll
      for (int k = 0; k < 16; ++k) if (k < nmine) w[k] = ld4(box + threadIdx.x + k * blockDim.x);
#pragma unroll
      for (int k = 0; k < 16; ++k) if (k < nmine) { ok &= (w[k].x == e) & (w[k].y == e) & (w[k].z == e) & (w[k].w == e); acc += w[k].x; }
    } while (!ok);
    __syncthreads();
    return acc;
  }
  if (threadIdx.x == 0) {
    if (form == 1) { while (ld_acq(flag) != e) {} }
    else { while (ld_acq(cnt) != e * blockDim.x) {} }
  }
  __syncthreads();
  for (int i = threadIdx.x; i < n16; i += blockDim.x) acc += ld4(box + i).y;
  return acc;
}
__global__ void __launch_bounds__(128, 1) pingpong(int form, int n16, int iters, uint4* boxA, uint4* boxB, unsigned* flags, unsigned* cnts,
                                                   unsigned long long* out, unsigned base) {
  extern __shared__ unsigned char dsm[];
  if (blockIdx.x > 1) return;
  if (threadIdx.x == 0) out[4 + blockIdx.x] = smid();
  unsigned acc = 0; dsm[threadIdx.x] = 0;
  const unsigned long long t0 = gtime();
  for (int it = 0; it < iters; ++it) {
    const unsigned e = base + (unsigned)it;
    if (blockIdx.x == 0) {
      publish(form, boxA, flags + 0, cnts + 0, n16, e);
      acc += observe(form, boxB, flags + 32, cnts + 32, n16, e);
    } else {
      acc += observe(form, boxA, flags + 0, cnts + 0, n16, e);
      publish(form, boxB, flags + 32, cnts + 32, n16, e);
    }
  }
  const unsigned long long t1 = gtime();
  if (blockIdx.x == 0 && threadIdx.x == 0) { out[0] = t1 - t0; out[1] = acc; }
}

// ---- (b) post-wait slowdown
__device__ __noinline__ void guard_timeout() { printf("K31_SPIN_TIMEOUT probe\n"); asm volatile("trap;"); }
template <int arm> __global__ void __launch_bounds__(128, 1) postwait(long long wait_cyc, unsigned* flag, unsigned e, unsigned long long* out, float* sink) {
  __shared__ float sh[128 * 4];
  if (blockIdx.x > 1) return;
  if (blockIdx.x == 1) {  // producer on another SM: spin wait_cyc then publish
    if (threadIdx.x == 0) { long long c0 = clock64(); while (clock64() - c0 < wait_cyc) {} st_rel(flag, e); }
    return;
  }
  const bool guard = arm == 1 || arm == 3, backoff = arm == 2 || arm == 3;
  unsigned n = 0; unsigned long long gt0 = 0, polls = 0;
  const long long w0 = clock64();
  while (ld_acq(flag) != e) {
    ++polls;
    if (backoff) asm volatile("nanosleep.u32 64;");
    if (guard) {
      ++n;
      if ((n & 1023u) == 0) {
        unsigned long long t = gtime();
        if (gt0 == 0) gt0 = t; else if (t - gt0 > 600000000000ull) guard_timeout();
      }
    }
  }
  const long long w1 = clock64();
  // fixed post-landing chain
  float x = (float)threadIdx.x * 1e-3f + 1.f, s = 0.f;
  for (int r = 0; r < 8; ++r) {
    sh[threadIdx.x * 4 + (r & 3)] = x;
    __syncthreads();
    float y = sh[((threadIdx.x + 33 * (r + 1)) & 127) * 4 + (r & 3)];
#pragma unroll
    for (int k = 0; k < 64; ++k) y = fmaf(y, 0.999f, 1e-3f);
    for (int o = 16; o; o >>= 1) y += __shfl_xor_sync(0xffffffffu, y, o);
    s += y; x = y * 1e-2f;
    __syncthreads();
  }
  const long long w2 = clock64();
  if (threadIdx.x == 0) { out[0] = (unsigned long long)(w1 - w0); out[1] = (unsigned long long)(w2 - w1); out[2] = polls; }
  if (threadIdx.x == 64) out[3] = (unsigned long long)(w2 - w1);
  if (s == 12345.f) sink[0] = s;
}

static double pct(std::vector<double> v, double p) { std::sort(v.begin(), v.end()); return v.empty() ? 0 : v[(size_t)(p * (v.size() - 1))]; }
int main(int argc, char** argv) {
  RK(cudaSetDevice(0));
  int nsm; cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
  int clk_khz; cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, 0);
  uint4 *bA, *bB; unsigned *fl, *cn; unsigned long long* out; float* sink;
  RK(cudaMalloc(&bA, 1 << 20)); RK(cudaMalloc(&bB, 1 << 20)); RK(cudaMalloc(&fl, 4096)); RK(cudaMalloc(&cn, 4096));
  RK(cudaMalloc(&out, 64)); RK(cudaMalloc(&sink, 64));
  RK(cudaFuncSetAttribute(pingpong, cudaFuncAttributeMaxDynamicSharedMemorySize, 150 * 1024));
  const char* fn[] = {"data-as-flag (Lamport 16B tags)", "fenced flag (fence.acq_rel + st.release)", "counter release (red.release per thread)"};
  printf("== m1c (c) hand-off: one-way publish->observe us = ping-pong round trip / 2; 128 threads/CTA; 1 CTA/SM; %d SMs\n", nsm);
  unsigned base = 1000;
  for (int form = 0; form < 3; ++form)
    for (int bytes : {16, 1024, 4096, 16384, 32768}) {
      std::vector<double> v; unsigned long long h[8];
      for (int r = 0; r < 7; ++r) {
        RK(cudaMemset(bA, 0, 1 << 20)); RK(cudaMemset(bB, 0, 1 << 20)); RK(cudaMemset(fl, 0, 4096)); RK(cudaMemset(cn, 0, 4096));
        const int iters = 2000;
        // counter form: counts accumulate 128 per epoch from 0: start epochs at 1
        unsigned b = form == 2 ? 1 : base; base += iters + 1;
        pingpong<<<2, 128, 150 * 1024>>>(form, bytes / 16, iters, bA, bB, fl, cn, out, b);
        RK(cudaDeviceSynchronize());
        RK(cudaMemcpy(h, out, 64, cudaMemcpyDeviceToHost));
        if (r) v.push_back((double)h[0] / iters / 2 / 1e3);
      }
      printf("hand-off | %-42s | %6d B | one-way us p50 %.3f min %.3f max %.3f (6 reps x 2000 round trips) | SMs %llu,%llu\n", fn[form], bytes,
             pct(v, .5), pct(v, 0), pct(v, 1), h[4], h[5]);
      fflush(stdout);
    }
  RK(cudaFuncSetAttribute(postwait<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, 150 * 1024));
  RK(cudaFuncSetAttribute(postwait<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, 150 * 1024));
  RK(cudaFuncSetAttribute(postwait<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, 150 * 1024));
  RK(cudaFuncSetAttribute(postwait<3>, cudaFuncAttributeMaxDynamicSharedMemorySize, 150 * 1024));
  const char* pn[] = {"P0 tight poll", "P1 tight + guard tick", "P2 nanosleep64 backoff", "P3 backoff + guard tick"};
  printf("== m1c (b) post-wait: consumer chain cycles after a wait of W us (producer on another SM); 101 launches, first dropped; SM clock attr %d MHz\n", clk_khz / 1000);
  for (int arm = 0; arm < 4; ++arm)
    for (double wus : {0.0, 1.0, 3.0, 6.0, 16.0}) {
      std::vector<double> wc, cc, pl;
      for (int r = 0; r < 101; ++r) {
        RK(cudaMemset(fl, 0, 4096));
        const long long wc_ = (long long)(wus * 2.36e3);
        if (arm == 0) postwait<0><<<2, 128, 150 * 1024>>>(wc_, fl, 77u + r, out, sink);
        else if (arm == 1) postwait<1><<<2, 128, 150 * 1024>>>(wc_, fl, 77u + r, out, sink);
        else if (arm == 2) postwait<2><<<2, 128, 150 * 1024>>>(wc_, fl, 77u + r, out, sink);
        else postwait<3><<<2, 128, 150 * 1024>>>(wc_, fl, 77u + r, out, sink);
        RK(cudaDeviceSynchronize());
        unsigned long long h[4]; RK(cudaMemcpy(h, out, 32, cudaMemcpyDeviceToHost));
        if (r) { wc.push_back(h[0] / 2.36e3); cc.push_back((double)h[1]); pl.push_back((double)h[2]); }
      }
      printf("post-wait | %-26s | W %4.1f us | measured wait us p50 %.2f | chain cycles p10 %.0f p50 %.0f p90 %.0f max %.0f | polls p50 %.0f\n", pn[arm], wus,
             pct(wc, .5), pct(cc, .1), pct(cc, .5), pct(cc, .9), pct(cc, 1), pct(pl, .5));
      fflush(stdout);
    }
  printf("DONE\n");
  return 0;
}
