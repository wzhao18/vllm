// pmla9_probe.cu (P-MLA-2CTA M9 projection probe): resource rates of the h9 form (2-CTA keys-on-M sparse MLA
// prefill) on sm_107a, all SMs, one 2-CTA cluster per SM pair. Per 256-key subtile per pair (128 own keys per CTA):
//   MMA  : QK 9 x M256 N96 K64 (A = K rows from TMEM, B = Q^T smem) + PV 8 x M256 N96 K64 (A = V^T TMEM, B = P^T smem)
//   KPROD: own 128 keys x 576 B by LDG.128 (4 lanes per row) -> tcgen05.st.16x128b.x4 (K-major A, permuted K order)
//   VPROD: 256 keys x own 256 dims by LDG.128 -> prmt 2x2 interleave -> movmatrix.trans -> tcgen05.st.16x128b.x4
//   SMX  : softmax-like work on S^T: tcgen05.ld 96 cols, redux max per head, ex2 x96, e4m3 cvt, 96 B st.shared
// Roles run free (no data dependencies; garbage values), clk per subtile per role = role span / NSUB.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>

#define NSUB 256
__device__ __forceinline__ uint64_t sdesc(uint32_t saddr) {  // SW128 K-major, SBO 1024
  return (uint64_t)((saddr >> 4) & 0x3FFF) | ((uint64_t)1 << 16) | ((uint64_t)64 << 32) | ((uint64_t)1 << 46) |
         ((uint64_t)2 << 61);
}
__device__ __forceinline__ uint4 ldg16(const uint8_t* p) {
  uint4 v;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ uint32_t movt(uint32_t a) {
  uint32_t d;
  asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(d) : "r"(a));
  return d;
}
__device__ __forceinline__ void st16x128x4(uint32_t ta, const uint32_t* r) {
  asm volatile("tcgen05.st.sync.aligned.16x128b.x4.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};" ::"r"(ta), "r"(r[0]),
               "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7])
               : "memory");
}

template <int MODE>  // bit0 MMA, bit1 KPROD, bit2 VPROD, bit3 SMX
__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(416, 1)
    h9probe(const uint8_t* __restrict__ kv, const int* __restrict__ idx, int nrows, unsigned long long* out) {
  extern __shared__ __align__(1024) unsigned char smem[];
  __shared__ uint32_t tmem_base;
  __shared__ __align__(8) uint64_t mbar;
  uint32_t s0 = (uint32_t)__cvta_generic_to_shared(smem);
  uint32_t sq = (s0 + 1023) & ~1023u, sp = sq + 65536;  // Q^T 48 heads x 576 (pad 64K), P^T 2 x 48 x 256
  uint32_t tb = (uint32_t)__cvta_generic_to_shared(&tmem_base), mb = (uint32_t)__cvta_generic_to_shared(&mbar);
  unsigned rank;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, p = warp % 4;
  if (warp == 0) {
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], 512;" ::"r"(tb) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
  }
  if (threadIdx.x == 32) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(mb) : "memory");
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  asm volatile("tcgen05.fence::before_thread_sync;");
  asm volatile("barrier.cluster.arrive.release.aligned;");
  asm volatile("barrier.cluster.wait.acquire.aligned;");
  asm volatile("tcgen05.fence::after_thread_sync;");
  const uint32_t t = tmem_base, TO = t, TS = t + 192, TK = t + 288, TV = t + 384;  // V^T overlaps (probe only)
  const uint32_t lofs = (uint32_t)(32 * p) << 16;
  const int pair = blockIdx.x / 2;
  const int* myidx = idx + (size_t)pair * NSUB * 256;
  long long c0 = clock64(), c1 = c0;
  uint32_t sink = 0;
  if (warp == 12) {  // MMA issuer (leader CTA)
    if ((MODE & 1) && rank == 0) {
      const unsigned iqk = (1u << 4) | (12u << 17) | (16u << 24) | (1u << 29);  // F32 D, E4M3, N96, M256, K64
      const uint64_t bq = sdesc(sq), bp = sdesc(sp);
      for (int u = 0; u < NSUB; ++u) {
        if (MODE & 16) {  // SS: A (keys / V^T) from SMEM instead of TMEM; same shapes
          const uint64_t ba = sdesc(sp + 16384);
          for (int k = 0; k < 17; ++k)
            asm volatile(
                "{.reg .pred P, Q; elect.sync _|P, 0xffffffff; setp.ne.b32 Q, %4, 0; @P tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], %1, %2, %3, Q;}" ::"r"(k < 9 ? TS : TO + 96 * (k & 1)),
                "l"(ba + 4 * (k % 8)), "l"(bq + 4 * (k % 8)), "r"(iqk), "r"(k)
                : "memory");
          continue;
        }
        for (int k = 0; k < 9; ++k)
          asm volatile(
              "{.reg .pred P, Q; elect.sync _|P, 0xffffffff; setp.ne.b32 Q, %4, 0; @P tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [%1], %2, %3, Q;}" ::"r"(TS),
              "r"(TK + 16 * k), "l"(bq + 4 * k), "r"(iqk), "r"(k)
              : "memory");
        for (int b = 0; b < 2; ++b)
          for (int k = 0; k < 4; ++k)
            asm volatile(
                "{.reg .pred P, Q; elect.sync _|P, 0xffffffff; setp.ne.b32 Q, %4, 0; @P tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [%1], %2, %3, Q;}" ::"r"(TO + 96 * b),
                "r"(TV + 64 * b + 16 * k), "l"(bp + 64 * k), "r"(iqk), "r"(u | k)
                : "memory");
      }
      unsigned short mask = 3;
      asm volatile(
          "{.reg .pred P; elect.sync _|P, 0xffffffff; @P tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;}" ::"r"(mb),
          "h"(mask)
          : "memory");
    }
    if (MODE & 1)
      asm volatile("{.reg .pred P; W: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], 0; @!P bra W;}" ::"r"(mb)
                   : "memory");
    c1 = clock64();
  } else if (warp >= 4 && warp < 12) {  // 8 producer warps, two per sub-partition
    if (MODE & 32) {  // LDGSTS gather of own 128 keys x 576 B per subtile: 8 warps, idx prefetch 4, ring D 3
      const int w8 = warp - 4, nset = NSUB * 16;
      const uint32_t rb = sq + 102400 + w8 * 3 * 4608 + (lane / 4) * 576 + 16 * (lane % 4);
      const int* bi = idx + (size_t)pair * NSUB * 256 + 128 * rank;
      auto sidx = [&](int st) { return bi + (st / 16) * 256 + (st % 16) * 8 + lane / 4; };
      int ix[4];
#pragma unroll
      for (int k = 0; k < 4; ++k) ix[k] = __ldg(sidx(w8 + 8 * k));
      int slot = 0;
      uint32_t acc = 0;
      for (int s0 = w8; s0 < nset + 24; s0 += 32) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const int st = s0 + 8 * k;
          if (st < nset) {
            const uint8_t* r = kv + (size_t)ix[k] * 576 + 16 * (lane % 4);
#pragma unroll
            for (int j = 0; j < 9; ++j)
              asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(rb + slot * 4608 + 64 * j), "l"(r + 64 * j) : "memory");
          }
          if (st + 32 < nset) ix[k] = __ldg(sidx(st + 32));
          asm volatile("cp.async.commit_group;" ::: "memory");
          asm volatile("cp.async.wait_group 2;" ::: "memory");
          slot = slot == 2 ? 0 : slot + 1;
          uint32_t x;
          asm volatile("ld.shared.u32 %0, [%1];" : "=r"(x) : "r"(rb + slot * 4608));
          acc ^= x;
        }
      }
      sink = acc;
    }
    const int h2 = (warp - 4) / 4;      // which of the two warps of sub-partition p
    // batched producer: per subtile load all K pieces (18 x LDG.128) and all V^T pieces (16 x LDG.128) before the
    // stores; K(u+1) loads are issued before V^T(u) is transposed and stored (register double buffering).
    const int hh = h2;
    uint4 ka[9], kb[9];
    auto loadk = [&](int u) {
      const int* si = myidx + u * 256;
      const int k0 = 128 * rank + 32 * p + 16 * hh + lane / 4;
      const uint8_t* ra = kv + (size_t)__ldg(si + k0) * 576 + 16 * (lane % 4);
      const uint8_t* rb = kv + (size_t)__ldg(si + k0 + 8) * 576 + 16 * (lane % 4);
#pragma unroll
      for (int j = 0; j < 9; ++j) { ka[j] = ldg16(ra + 64 * j); kb[j] = ldg16(rb + 64 * j); }
    };
    if (MODE & 2) loadk(0);
    for (int u = 0; u < NSUB; ++u) {
      const int* si = myidx + u * 256;
      uint4 va[8], vb[8];
      if (MODE & 4) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const int g = h2 + 2 * i, kr = 16 * g + 2 * (lane / 4);
          const int off = 256 * rank + 64 * p + 16 * (lane % 4);
          va[i] = ldg16(kv + (size_t)__ldg(si + kr) * 576 + off);
          vb[i] = ldg16(kv + (size_t)__ldg(si + kr + 1) * 576 + off);
        }
      }
      if (MODE & 2) {
#pragma unroll
        for (int j = 0; j < 9; ++j) {
          uint32_t r[8] = {ka[j].x, kb[j].x, ka[j].y, kb[j].y, ka[j].z, kb[j].z, ka[j].w, kb[j].w};
          st16x128x4(TK + lofs + ((uint32_t)(16 * hh) << 16) + 16 * j, r);
        }
        if (u + 1 < NSUB) loadk(u + 1);
      }
      if (MODE & 4) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const int g = h2 + 2 * i;
          const uint32_t aw[4] = {va[i].x, va[i].y, va[i].z, va[i].w}, bw[4] = {vb[i].x, vb[i].y, vb[i].z, vb[i].w};
          uint32_t r[8];
#pragma unroll
          for (int w = 0; w < 4; ++w) {
            r[2 * w] = movt(__byte_perm(aw[w], bw[w], 0x6240));
            r[2 * w + 1] = movt(__byte_perm(aw[w], bw[w], 0x7351));
          }
          st16x128x4(TV + lofs + 64 * (g / 8) + 4 * (g % 8), r);
        }
      }
      if (MODE & 6) asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    }
    c1 = clock64();
  } else if (warp < 4) {  // softmax-like role on S^T (thread = key, 96 heads)
    if (MODE & 8) {
      float acc = 0.f;
      for (int u = 0; u < NSUB; ++u) {
        uint32_t v[96];
#pragma unroll
        for (int c = 0; c < 3; ++c)
          asm volatile(
              "tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
              : "=r"(v[32 * c + 0]), "=r"(v[32 * c + 1]), "=r"(v[32 * c + 2]), "=r"(v[32 * c + 3]),
                "=r"(v[32 * c + 4]), "=r"(v[32 * c + 5]), "=r"(v[32 * c + 6]), "=r"(v[32 * c + 7]),
                "=r"(v[32 * c + 8]), "=r"(v[32 * c + 9]), "=r"(v[32 * c + 10]), "=r"(v[32 * c + 11]),
                "=r"(v[32 * c + 12]), "=r"(v[32 * c + 13]), "=r"(v[32 * c + 14]), "=r"(v[32 * c + 15]),
                "=r"(v[32 * c + 16]), "=r"(v[32 * c + 17]), "=r"(v[32 * c + 18]), "=r"(v[32 * c + 19]),
                "=r"(v[32 * c + 20]), "=r"(v[32 * c + 21]), "=r"(v[32 * c + 22]), "=r"(v[32 * c + 23]),
                "=r"(v[32 * c + 24]), "=r"(v[32 * c + 25]), "=r"(v[32 * c + 26]), "=r"(v[32 * c + 27]),
                "=r"(v[32 * c + 28]), "=r"(v[32 * c + 29]), "=r"(v[32 * c + 30]), "=r"(v[32 * c + 31])
              : "r"(TS + lofs + 32 * c));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        uint32_t pk[24];
#pragma unroll
        for (int h = 0; h < 96; h += 4) {
          float f[4];
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            float s = __uint_as_float(v[h + e]) * 0.0625f;
            float m;
            asm volatile("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(m) : "f"(s));
            float x;
            asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(x) : "f"(s - m));
            f[e] = x;
            acc += x;
          }
          uint16_t lo, hi;
          asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(lo) : "f"(f[1]), "f"(f[0]));
          asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(hi) : "f"(f[3]), "f"(f[2]));
          pk[h / 4] = (uint32_t)lo | ((uint32_t)hi << 16);
        }
        uint32_t a = sp + 32768 + (u & 1) * 12288 + threadIdx.x * 96;
#pragma unroll
        for (int i = 0; i < 24; i += 4)
          asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" ::"r"(a + 4 * i), "r"(pk[i]), "r"(pk[i + 1]),
                       "r"(pk[i + 2]), "r"(pk[i + 3])
                       : "memory");
      }
      sink = __float_as_uint(acc);
    }
    c1 = clock64();
  }
  __shared__ long long span[13];
  if (lane == 0 && warp < 13) span[warp] = c1 - c0;
  asm volatile("tcgen05.fence::before_thread_sync;");
  __syncthreads();
  asm volatile("barrier.cluster.arrive.release.aligned;");
  asm volatile("barrier.cluster.wait.acquire.aligned;");
  if (threadIdx.x == 0) {
    long long mm = span[12], kp = 0, sm = 0;
    for (int w = 4; w < 12; ++w) kp = kp > span[w] ? kp : span[w];
    for (int w = 0; w < 4; ++w) sm = sm > span[w] ? sm : span[w];
    out[blockIdx.x * 4 + 0] = mm;
    out[blockIdx.x * 4 + 1] = kp;
    out[blockIdx.x * 4 + 2] = sm;
    out[blockIdx.x * 4 + 3] = sink;
  }
  if (warp == 0) asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, 512;" ::"r"(t) : "memory");
}

template <int MODE>
void run(const char* name, int grid, const uint8_t* kv, const int* idx, int nrows, unsigned long long* out) {
  cudaFuncSetAttribute(h9probe<MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, 220 * 1024);
  cudaEvent_t ea, eb;
  cudaEventCreate(&ea);
  cudaEventCreate(&eb);
  float ms = 0;
  for (int rep = 0; rep < 3; ++rep) {
    cudaEventRecord(ea);
    h9probe<MODE><<<grid, 416, 220 * 1024>>>(kv, idx, nrows, out);
    cudaEventRecord(eb);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
      printf("%s: %s\n", name, cudaGetErrorString(e));
      exit(1);
    }
    cudaEventElapsedTime(&ms, ea, eb);
  }
  std::vector<unsigned long long> h(grid * 4);
  cudaMemcpy(h.data(), out, grid * 4 * 8, cudaMemcpyDeviceToHost);
  double s[3] = {0, 0, 0}, mx[3] = {0, 0, 0};
  for (int b = 0; b < grid; ++b)
    for (int r = 0; r < 3; ++r) {
      double v = (double)h[b * 4 + r] / NSUB;
      s[r] += v;
      mx[r] = mx[r] > v ? mx[r] : v;
    }
  const double nsps = ms * 1e6 / NSUB, mxc = mx[0] > mx[1] ? (mx[0] > mx[2] ? mx[0] : mx[2]) : (mx[1] > mx[2] ? mx[1] : mx[2]);
  printf("%-14s %6.1f ns/subtile = %5.1f us/layer, eff clk %4.2f GHz | ", name, nsps, nsps * 1236.5 / 1000, mxc / nsps);
  printf("%-18s clk/subtile(256 keys/pair)  MMA avg %6.0f max %6.0f | PROD avg %6.0f max %6.0f | SMX avg %6.0f max %6.0f\n",
         name, s[0] / grid, mx[0], s[1] / grid, mx[1], s[2] / grid, mx[2]);
}

int main() {
  int dev = 0, nsm = 0;
  cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
  const int grid = nsm & ~1, npair = grid / 2, nrows = 78 * 1024;
  uint8_t* kv;
  int* idx;
  unsigned long long* out;
  cudaMalloc(&kv, (size_t)nrows * 576);
  cudaMemset(kv, 0x11, (size_t)nrows * 576);
  std::vector<int> hi((size_t)npair * NSUB * 256);
  srand(1);
  for (auto& x : hi) x = rand() % nrows;
  cudaMalloc(&idx, hi.size() * 4);
  cudaMemcpy(idx, hi.data(), hi.size() * 4, cudaMemcpyHostToDevice);
  cudaMalloc(&out, grid * 4 * 8);
  printf("SMs %d grid %d (pairs %d), KV %d rows x 576 B; SOL budget 1008 clk/subtile (7.9 clk/own key), target 1109\n",
         nsm, grid, npair, nrows);
  run<1>("MMA-TS", grid, kv, idx, nrows, out);
  run<17>("MMA-SS", grid, kv, idx, nrows, out);
  run<32>("LDGSTS", grid, kv, idx, nrows, out);
  run<33>("TS+LDGSTS", grid, kv, idx, nrows, out);
  run<49>("SS+LDGSTS", grid, kv, idx, nrows, out);
  run<41>("TS+LDGSTS+SMX", grid, kv, idx, nrows, out);
  return 0;
}
