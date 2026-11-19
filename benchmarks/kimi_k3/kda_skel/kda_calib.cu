// SPDX-License-Identifier: Apache-2.0
// CALIBRATION skeleton of TODAY's KDA prefill layer design (SOL protocol, milestone M1), with math.
// Today's layer at F16K = three passes over striped (cudaMalloc) token-major tensors:
//   1. qkv_conv: width-4 causal depthwise conv (FP32 weights) + SiLU over raw q|k|v [T][3*12288] BF16 -> BF16
//   2. core (FlashInfer persistent KDA, 1 chain per SM, BT32): priced by P-KDA2's skel2 1-chain mode (same job)
//   3. o_norm: gated RMSNorm (sigmoid gate, per-dim weight) + MXFP8 (E8M0 per 32) of O [T][12288] BF16
// Passes 1 and 3 here do the full per-element math (conv 4 FMA + SiLU ex2/rcp; norm sum of squares, sigmoid
// ex2/rcp, scale, amax, e4m3 cvt) with plain vector LDG/STG, like the deployed kernels; they must land within ~10%
// of the deployed kernels' microbench (qkv_conv 222 us, o_norm 101 us) for the calibration to hold.
//   kda_calib <nseq> <seqlen> [iters]
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <algorithm>

#define CR(x) do { cudaError_t r_ = (x); if (r_ != cudaSuccess) { printf("CUDA err %s line %d\n", cudaGetErrorString(r_), __LINE__); exit(1); } } while (0)

constexpr int H = 96, D = 128, P = H * D, P3 = 3 * P;
constexpr int CT = 64;   // conv token tile

__device__ __forceinline__ void unpack8(uint4 u, float* f) {
  unsigned w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
  for (int e = 0; e < 4; ++e) { f[2 * e] = __uint_as_float(w[e] << 16); f[2 * e + 1] = __uint_as_float(w[e] & 0xffff0000u); }
}
__device__ __forceinline__ unsigned pk2(float a, float b) {
  __nv_bfloat162 v = __floats2bfloat162_rn(a, b); return *(unsigned*)&v;
}
__device__ __forceinline__ float silu(float x) {
  float e, r;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e) : "f"(x * -1.4426950409f));
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(1.f + e));
  return x * r;
}

// thread = 8 channels x CT tokens, 3-row halo from the same sequence (zero at the sequence start: conv state 0)
__global__ void __launch_bounds__(256) conv_k(const uint4* __restrict__ in, uint4* __restrict__ out,
                                              const float4* __restrict__ w, int T, int seqlen) {
  const int cg = blockIdx.x * blockDim.x + threadIdx.x;   // channel group (8 ch)
  if (cg >= P3 / 8) return;
  const int t0 = blockIdx.y * CT, seq0 = (t0 / seqlen) * seqlen;
  float wt[8][4];
#pragma unroll
  for (int c = 0; c < 8; ++c) { float4 q = w[cg * 8 + c]; wt[c][0] = q.x; wt[c][1] = q.y; wt[c][2] = q.z; wt[c][3] = q.w; }
  float h0[8], h1[8], h2[8];
#pragma unroll
  for (int c = 0; c < 8; ++c) h0[c] = h1[c] = h2[c] = 0.f;
#pragma unroll
  for (int r = -3; r < 0; ++r) {
    const int t = t0 + r;
    float x[8];
    if (t >= 0 && t >= seq0 && t < T) unpack8(in[(size_t)t * (P3 / 8) + cg], x);
    else {
#pragma unroll
      for (int c = 0; c < 8; ++c) x[c] = 0.f;
    }
#pragma unroll
    for (int c = 0; c < 8; ++c) { h0[c] = h1[c]; h1[c] = h2[c]; h2[c] = x[c]; }
  }
  for (int r0 = 0; r0 < CT; r0 += 8) {   // 8 rows of loads in flight per thread (memory-level parallelism)
    uint4 xr[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) { const int t = t0 + r0 + j; xr[j] = t < T ? in[(size_t)t * (P3 / 8) + cg] : make_uint4(0, 0, 0, 0); }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int t = t0 + r0 + j;
      if (t >= T) break;
      if (t == ((t / seqlen) * seqlen) && (r0 + j)) {   // new sequence inside the tile: reset the halo
#pragma unroll
        for (int c = 0; c < 8; ++c) h0[c] = h1[c] = h2[c] = 0.f;
      }
      float x[8], y[8];
      unpack8(xr[j], x);
#pragma unroll
      for (int c = 0; c < 8; ++c) {
        float a = wt[c][3] * x[c];
        a = fmaf(wt[c][2], h2[c], a); a = fmaf(wt[c][1], h1[c], a); a = fmaf(wt[c][0], h0[c], a);
        y[c] = silu(a);
        h0[c] = h1[c]; h1[c] = h2[c]; h2[c] = x[c];
      }
      out[(size_t)t * (P3 / 8) + cg] = make_uint4(pk2(y[0], y[1]), pk2(y[2], y[3]), pk2(y[4], y[5]), pk2(y[6], y[7]));
    }
  }
}

// thread = 16 channels of one (token, head) row; 8 lanes per row; 2 lanes per MX block of 32
__global__ void __launch_bounds__(256) onorm_k(const uint4* __restrict__ o, const uint4* __restrict__ g,
                                               const float* __restrict__ w, uint4* __restrict__ out,
                                               unsigned char* __restrict__ sc, long long n16) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;   // 16-channel piece
  if (i >= n16) return;
  const int part = (int)(i & 7);
  float y[16], gt[16];
  unpack8(o[2 * i], y); unpack8(o[2 * i + 1], y + 8);
  unpack8(g[2 * i], gt); unpack8(g[2 * i + 1], gt + 8);
  float ss = 0.f;
#pragma unroll
  for (int c = 0; c < 16; ++c) ss = fmaf(y[c], y[c], ss);
  ss += __shfl_xor_sync(0xffffffffu, ss, 1); ss += __shfl_xor_sync(0xffffffffu, ss, 2); ss += __shfl_xor_sync(0xffffffffu, ss, 4);
  const float rstd = rsqrtf(ss * (1.f / 128.f) + 1e-6f);
  float amax = 0.f;
#pragma unroll
  for (int c = 0; c < 16; ++c) {
    float e, s;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e) : "f"(gt[c] * -1.4426950409f));
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(s) : "f"(1.f + e));
    y[c] = y[c] * rstd * __ldg(w + part * 16 + c) * s;
    amax = fmaxf(amax, fabsf(y[c]));
  }
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
  int ex8 = (int)((__float_as_uint(amax) >> 23) & 0xff) - 8; ex8 = ex8 < 1 ? 1 : ex8;
  const float inv = __uint_as_float((unsigned)(254 - ex8) << 23);
  unsigned pk[4];
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    unsigned short a01, a23;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(a01) : "f"(y[4 * q + 1] * inv), "f"(y[4 * q] * inv));
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(a23) : "f"(y[4 * q + 3] * inv), "f"(y[4 * q + 2] * inv));
    pk[q] = (unsigned)a01 | ((unsigned)a23 << 16);
  }
  out[i] = make_uint4(pk[0], pk[1], pk[2], pk[3]);
  if (!(part & 1)) sc[i >> 1] = (unsigned char)ex8;
}

int main(int argc, char** argv) {
  int nseq = argc > 1 ? atoi(argv[1]) : 3, seqlen = argc > 2 ? atoi(argv[2]) : 5461, iters = argc > 3 ? atoi(argv[3]) : 15;
  const long long T = (long long)nseq * seqlen;
  printf("CALIB T=%lld nseq=%d seqlen=%d\n", T, nseq, seqlen);
  void *qkv, *qkvo, *o, *g, *out, *flush; unsigned char* sc; float *w, *ow;
  CR(cudaMalloc(&qkv, T * P3 * 2)); CR(cudaMalloc(&qkvo, T * P3 * 2)); CR(cudaMalloc(&o, T * P * 2)); CR(cudaMalloc(&g, T * P * 2));
  CR(cudaMalloc(&out, T * P)); CR(cudaMalloc(&sc, T * P / 32)); CR(cudaMalloc(&w, P3 * 16)); CR(cudaMalloc(&ow, D * 4));
  std::vector<float> hw(P3 * 4); for (size_t i = 0; i < hw.size(); ++i) hw[i] = 0.1f * (float)((i * 2654435761u) % 17) - 0.8f;
  CR(cudaMemcpy(w, hw.data(), P3 * 16, cudaMemcpyHostToDevice)); CR(cudaMemcpy(ow, hw.data(), D * 4, cudaMemcpyHostToDevice));
  CR(cudaMemset(qkv, 0x3c, T * P3 * 2)); CR(cudaMemset(o, 0x3c, T * P * 2)); CR(cudaMemset(g, 0x3b, T * P * 2));
  size_t FL = 512ull << 20; CR(cudaMalloc(&flush, FL));
  cudaEvent_t e0, e1; CR(cudaEventCreate(&e0)); CR(cudaEventCreate(&e1));
  auto timeit = [&](const char* name, double bytes, auto launch) {
    std::vector<float> ts;
    for (int it = 0; it < iters + 3; ++it) {
      CR(cudaMemsetAsync(flush, it, FL));
      CR(cudaEventRecord(e0)); launch(); CR(cudaEventRecord(e1)); CR(cudaEventSynchronize(e1)); CR(cudaGetLastError());
      float ms; CR(cudaEventElapsedTime(&ms, e0, e1)); if (it >= 3) ts.push_back(ms * 1000.f);
    }
    std::sort(ts.begin(), ts.end());
    float med = ts[ts.size() / 2];
    printf("CALIB %-10s med %7.1f us min %7.1f | bytes %.3f GB -> %5.2f TB/s | SOL@19.4 %6.1f us\n", name, med, ts[0],
           bytes / 1e9, bytes / (med * 1e-6) / 1e12, bytes / 19.4e12 * 1e6);
    return med;
  };
  dim3 cgrid((P3 / 8 + 255) / 256, (unsigned)((T + CT - 1) / CT));
  double cb = (double)T * P3 * 4;
  float tc = timeit("qkv_conv", cb, [&] { conv_k<<<cgrid, 256>>>((const uint4*)qkv, (uint4*)qkvo, (const float4*)w, (int)T, seqlen); });
  long long n16 = T * P / 16;
  double ob = (double)T * P * (2 + 2 + 1) + (double)T * P / 32;
  float to = timeit("o_norm", ob, [&] { onorm_k<<<(unsigned)((n16 + 255) / 256), 256>>>((const uint4*)o, (const uint4*)g, ow, (uint4*)out, sc, n16); });
  printf("CALIB_SUM conv %.1f + o_norm %.1f = %.1f us (add the skel2 1-chain core below)\n", tc, to, tc + to);
  return 0;
}
