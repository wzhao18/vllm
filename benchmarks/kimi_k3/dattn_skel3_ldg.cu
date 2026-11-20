// dattn_skel3_ldg.cu (D-ATTN M6-1a): MMA-consumable KV tile landing through the LSU path instead of TMA gather4.
// M6-0b measured gather4 in the SW128 form (4 x 128 B + 1 x 64 B boxes per quad) at ~1 op / 33 clk / SM regardless of
// issuers (5.9 TB/s chip: union 8.8-9.4 us, per-row 18.5 us at R384). Here loader warps read whole 576 B rows with
// coalesced 128-bit LDG (one warp instruction = one row's 512 B latent, lanes 0-3 the 64 B RoPE) and st.shared them into
// the K-major SW128 latent chunks [128 keys x 128 B] x 4 + SW64 RoPE [128 x 64 B] (the layout the tcgen05 B/A
// descriptors of km / sparse_mla read), then fence.proxy.async + one mbarrier arrive per warp. A consumer warp waits each
// stage and releases it (no math). Same R384/R256/R128 lists as dattn_skel2_mla.cu.
// Usage: skel3 MODE NW RPW ST [cfrac] [iters] [NREQ]
//   MODE bit0: request-union list, bit2: L2-warm (no flush); NW loader warps, RPW rows in flight per warp per batch
//   (two batches in flight), ST stages of 128 rows.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define RK(x) do { cudaError_t r_ = (x); if (r_ != cudaSuccess) { printf("CUDA error %s at %s:%d\n", \
  cudaGetErrorString(r_), __FILE__, __LINE__); exit(1); } } while (0)
constexpr int ROWB = 576, TK = 128, STAGEB = 4 * TK * 128 + TK * 64;  // 73728 B
__device__ __forceinline__ unsigned sa(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ unsigned long long gt() { unsigned long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ void wait_par(unsigned b, unsigned ph) {
  asm volatile("{.reg .pred P; W: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1; @!P bra W;}" :: "r"(b), "r"(ph) : "memory");
}
__device__ __forceinline__ void sts16(unsigned d, uint4 x) {
  asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" :: "r"(d), "r"(x.x), "r"(x.y), "r"(x.z), "r"(x.w) : "memory");
}
__device__ __forceinline__ void arrive(unsigned b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(b) : "memory"); }
// SW128: 16 B unit j of row r in a [rows x 128 B] chunk lands at r*128 + ((j ^ (r & 7)) << 4); SW64 RoPE: j ^ ((r >> 1) & 3)
__device__ __forceinline__ unsigned dst_off(int r, int g) {
  return g < 32 ? (unsigned)((g >> 3) * (TK * 128) + r * 128 + ((((g & 7) ^ (r & 7))) << 4))
                : (unsigned)(4 * TK * 128 + r * 64 + ((((g - 32) ^ ((r >> 1) & 3))) << 4));
}
template <int RPW>
__global__ void __launch_bounds__(544, 1) k3(const unsigned char* __restrict__ kv, const unsigned* __restrict__ rows,
                                              const int* __restrict__ off, int ST, unsigned long long* tspan) {
  extern __shared__ __align__(1024) unsigned char smem[];
  __shared__ __align__(8) unsigned long long full[8], empty[8];
  __shared__ unsigned sidx[2048];
  const int NW = blockDim.x / 32 - 1, w = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int r0 = off[blockIdx.x], nr = off[blockIdx.x + 1] - r0, ntile = (nr + TK - 1) / TK;
  if (threadIdx.x == 0) {
    for (int i = 0; i < ST; ++i) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(sa(&full[i])), "r"(NW));
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(sa(&empty[i])));
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    atomicMin(&tspan[0], gt());
  }
  for (int i = threadIdx.x; i < nr; i += blockDim.x) sidx[i] = rows[r0 + i];
  __syncthreads();
  const unsigned sbase = sa(smem);
  if (w < NW) {
    // NW = 16: warp w owns rows w + 16 k (k < 8) of every tile; one tile of loads in flight ahead of the stores
    // (2 x 8 rows x 576 B per warp); latent: lane = 16 B unit of the row's 512 B; RoPE: lane -> row k = lane / 4.
    const int per = TK / NW;
    uint4 xa[8], xb[8], qa = make_uint4(0, 0, 0, 0), qb = qa;
#define LD(t, X, Q) { _Pragma("unroll") for (int k = 0; k < 8; ++k) { const int gi = (t) * TK + w + NW * k; \
      if (k < per && gi < nr) X[k] = __ldg((const uint4*)(kv + (size_t)sidx[gi] * ROWB) + lane); } \
    { const int k = lane >> 2, gi = (t) * TK + w + NW * k; \
      if (k < per && gi < nr) Q = __ldg((const uint4*)(kv + (size_t)sidx[gi] * ROWB) + 32 + (lane & 3)); } }
#define STO(t, X, Q, st) { _Pragma("unroll") for (int k = 0; k < 8; ++k) { const int rr = w + NW * k, gi = (t) * TK + rr; \
      if (k < per && gi < nr) sts16((st) + dst_off(rr, lane), X[k]); } \
    { const int k = lane >> 2, rr = w + NW * k, gi = (t) * TK + rr; \
      if (k < per && gi < nr) sts16((st) + dst_off(rr, 32 + (lane & 3)), Q); } }
    if (ntile > 0) LD(0, xa, qa);
    for (int t = 0; t < ntile; ++t) {
      const int stg = t % ST;
      if (t + 1 < ntile) { if (t & 1) LD(t + 1, xa, qa) else LD(t + 1, xb, qb) }
      if (t >= ST) wait_par(sa(&empty[stg]), ((t / ST) - 1) & 1);
      const unsigned st = sbase + stg * STAGEB;
      if (t & 1) STO(t, xb, qb, st) else STO(t, xa, qa, st)
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      __syncwarp();
      if (lane == 0) arrive(sa(&full[stg]));
    }
  } else if (lane == 0) {
    unsigned acc = 0;
    for (int t = 0; t < ntile; ++t) {
      const int stg = t % ST;
      wait_par(sa(&full[stg]), (t / ST) & 1);
      acc += *(volatile unsigned*)(smem + stg * STAGEB + (t & 1023) * 64);
      arrive(sa(&empty[stg]));
    }
    if (acc == 0xdeadbeef) tspan[2] = acc;
    atomicMax(&tspan[1], gt());
  }
}
int main(int argc, char** argv) {
  if (argc < 5) { printf("usage: %s MODE NW RPW ST [cfrac] [iters] [NREQ]\n", argv[0]); return 2; }
  const int MODE = atoi(argv[1]), NW = atoi(argv[2]), RPW = atoi(argv[3]), ST = atoi(argv[4]);
  const double cfrac = argc > 5 ? atof(argv[5]) : 0.8;
  const int iters = argc > 6 ? atoi(argv[6]) : 10, NREQ = argc > 7 ? atoi(argv[7]) : 96, nmin = 448, nmax = 576;
  const size_t NKV = (size_t)4 << 20;
  if (NW != 16 || ST < 1 || ST > 4 || !(RPW == 2 || RPW == 4 || RPW == 8)) { printf("bad cfg\n"); return 2; }
  int nsm; RK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
  unsigned rs = 777;  // same generator as dattn_skel2_mla.cu
  auto rnd = [&]() { rs = rs * 1664525u + 1013904223u; return rs >> 2; };
  std::vector<unsigned> rows;
  for (int qi = 0; qi < NREQ; ++qi) {
    int nr[4], mn = 1 << 30;
    for (int r = 0; r < 4; ++r) { nr[r] = nmin + rnd() % (nmax - nmin + 1); mn = std::min(mn, nr[r]); }
    int C = (int)(cfrac * mn);
    std::vector<unsigned> com(C);
    for (auto& v : com) v = rnd() % NKV;
    if (MODE & 1) rows.insert(rows.end(), com.begin(), com.end());
    for (int r = 0; r < 4; ++r) {
      std::vector<unsigned> s(com);
      for (int i = C; i < nr[r]; ++i) s.push_back(rnd() % NKV);
      if (MODE & 1) rows.insert(rows.end(), s.begin() + C, s.end());
      else rows.insert(rows.end(), s.begin(), s.end());
    }
  }
  // split-K: contiguous row ranges, tile-aligned (128) so tiles never straddle CTAs
  const int NT = (int)((rows.size() + TK - 1) / TK);
  if ((size_t)(NT + nsm - 1) / nsm * TK > 2048) { printf("rows per CTA > 2048\n"); return 2; }
  std::vector<int> off(nsm + 1);
  for (int c = 0; c <= nsm; ++c) off[c] = std::min((int)rows.size(), (int)((long long)NT * c / nsm) * TK);
  unsigned* drows; int* doff; unsigned long long* dts; unsigned char* kv;
  RK(cudaMalloc(&drows, rows.size() * 4)); RK(cudaMemcpy(drows, rows.data(), rows.size() * 4, cudaMemcpyHostToDevice));
  RK(cudaMalloc(&doff, off.size() * 4)); RK(cudaMemcpy(doff, off.data(), off.size() * 4, cudaMemcpyHostToDevice));
  RK(cudaMalloc(&dts, 8 * 16)); RK(cudaMalloc(&kv, NKV * ROWB)); RK(cudaMemset(kv, 1, NKV * ROWB));
  size_t FL = 1024ull << 20; unsigned char* flush; RK(cudaMalloc(&flush, FL));
  const size_t smem = (size_t)ST * STAGEB;
  auto kf = RPW == 2 ? k3<2> : RPW == 4 ? k3<4> : k3<8>;
  RK(cudaFuncSetAttribute(kf, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  std::vector<double> sp;
  for (int it = 0; it < iters + 2; ++it) {
    if (!(MODE & 4) || it == 0) RK(cudaMemset(flush, it, FL));
    unsigned long long init[2] = {~0ull, 0};
    RK(cudaMemcpy(dts, init, 16, cudaMemcpyHostToDevice));
    kf<<<nsm, 32 * (NW + 1), smem>>>(kv, drows, doff, ST, dts);
    RK(cudaGetLastError()); RK(cudaDeviceSynchronize());
    unsigned long long h[2]; RK(cudaMemcpy(h, dts, 16, cudaMemcpyDeviceToHost));
    if (it >= 2) sp.push_back((h[1] - h[0]) * 1e-3);
  }
  std::sort(sp.begin(), sp.end());
  const double mb = rows.size() * (double)ROWB / 1e6, med = sp[sp.size() / 2];
  printf("SKEL3 MODE=%d NW=%d RPW=%d ST=%d %s %s NREQ=%d | %.1f MB smem %zu | span med %.2f us (min %.2f) | %.2f TB/s %.0f GB/s/SM\n",
         MODE, NW, RPW, ST, MODE & 1 ? "union" : "per-row", MODE & 4 ? "warm" : "cold", NREQ, mb, smem, med, sp[0],
         mb / med, mb / med * 1e3 / nsm);
  return 0;
}
