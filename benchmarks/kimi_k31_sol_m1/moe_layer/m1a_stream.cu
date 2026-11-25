#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char* s; cuGetErrorString(r_, &s); \
  printf("CU error %s at %s:%d\n", s, __FILE__, __LINE__); exit(1); } } while (0)
#define RK(x) do { cudaError_t r_ = (x); if (r_ != cudaSuccess) { printf("CUDA error %s at %s:%d\n", \
  cudaGetErrorString(r_), __FILE__, __LINE__); exit(1); } } while (0)

__device__ unsigned char g_dom[1024];
__device__ int g_rank[1024];
__device__ __forceinline__ unsigned smid() { unsigned r; asm volatile("mov.u32 %0, %%smid;" : "=r"(r)); return r; }
__device__ __forceinline__ unsigned sa(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void wait_par(unsigned b, unsigned ph) {
  asm volatile("{.reg .pred P; W: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1; @!P bra W;}" :: "r"(b), "r"(ph) : "memory");
}
__device__ __forceinline__ void wait_test(unsigned b, unsigned ph) {  // non-suspending poll
  asm volatile("{.reg .pred P; W: mbarrier.test_wait.parity.shared::cta.b64 P, [%0], %1; @!P bra W;}" :: "r"(b), "r"(ph) : "memory");
}
__device__ __forceinline__ void bulk(unsigned dst, const void* src, unsigned n, unsigned b) {
  asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" :: "r"(dst), "l"(src), "r"(n), "r"(b) : "memory");
}
__device__ __forceinline__ void tile2d(unsigned dst, const CUtensorMap* m, int row, unsigned b) {
  int col = 0;
  asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
               :: "r"(dst), "l"(m), "r"(col), "r"(row), "r"(b) : "memory");
}
__global__ void lat_k(const unsigned long long* b0, const unsigned long long* b1, size_t words, unsigned* out) {
  if (threadIdx.x) return;
  const unsigned long long* bs[2] = {b0, b1};
  for (int d = 0; d < 2; ++d) {
    size_t idx = (size_t)(blockIdx.x * 7919ull * 1031ull + d * 977) % words;
    unsigned long long acc = 0; long long t0 = clock64();
    for (int i = 0; i < 32; ++i) {
      unsigned long long v; asm volatile("ld.global.cv.u64 %0, [%1];" : "=l"(v) : "l"(bs[d] + idx));
      acc += v; idx = (idx + 131088ull + (v & 1)) % words;
    }
    out[4 * blockIdx.x + 1 + d] = (unsigned)((clock64() - t0) / 32);
    if (acc == 12345) out[0] = 1;
  }
  out[4 * blockIdx.x] = smid();
}
static void* alloc_dom(int dev, int dom, size_t bytes) {
  CUmemAllocationProp p = {}; p.type = CU_MEM_ALLOCATION_TYPE_PINNED; p.location.type = CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN;
  p.location.localized.deviceId = (unsigned char)dev; p.location.localized.localityDomainId = (unsigned char)dom;
  size_t g = 0; CK(cuMemGetAllocationGranularity(&g, &p, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED)); bytes = (bytes + g - 1) / g * g;
  CUmemGenericAllocationHandle h; CK(cuMemCreate(&h, bytes, &p, 0)); CUdeviceptr va; CK(cuMemAddressReserve(&va, bytes, g, 0, 0));
  CK(cuMemMap(va, bytes, 0, h, 0)); CUmemAccessDesc a = {}; a.location.type = CU_MEM_LOCATION_TYPE_DEVICE; a.location.id = dev;
  a.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE; CK(cuMemSetAccess(va, bytes, &a, 1)); return (void*)va;
}
static cudaEvent_t e0, e1;

__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
#include <cstdint>

__device__ __forceinline__ uint64_t sdesc(uint32_t saddr) {
  return (uint64_t)((saddr >> 4) & 0x3FFF) | ((uint64_t)1 << 16) | ((uint64_t)64 << 32) | ((uint64_t)1 << 46) | ((uint64_t)2 << 61);
}
static unsigned mk(int dfmt_or_bsf, int afmt, int bfmt, int N, int M, int sf_ue8m0, int bit31, int bit3) {
  return ((unsigned)dfmt_or_bsf << 4) | ((unsigned)afmt << 7) | ((unsigned)bfmt << 10) | ((unsigned)(N >> 3) << 17) |
         ((unsigned)sf_ue8m0 << 23) | ((unsigned)(M >> 4) << 24) | ((unsigned)bit31 << 31) | ((unsigned)bit3 << 3);
}
__device__ __forceinline__ void mma_mx(uint32_t d, uint64_t as, uint64_t b, unsigned id, uint32_t sfa, uint32_t sfb) {
  asm volatile("{.reg .pred P, Q; elect.sync _|P, 0xffffffff; setp.ne.b32 Q, %6, 0; @P "
               "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%4], [%5], Q;}"
               :: "r"(d), "l"(as), "l"(b), "r"(id), "r"(sfa), "r"(sfb), "r"(1u) : "memory");
}
__device__ __forceinline__ void commit_el(uint32_t mb) {
  asm volatile("{.reg .pred P; elect.sync _|P, 0xffffffff; @P tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];}"
               :: "r"(mb) : "memory");
}
// NM MMAs with compile-time descriptor offsets (hw28c uniform-issue rule): A 4 KiB apart, B alternating 2 x 16 KiB
template <int NM> __device__ __forceinline__ void mma_n(uint32_t t, uint64_t ad, uint64_t bd, unsigned idesc) {
#pragma unroll
  for (int m = 0; m < NM; ++m) mma_mx(t, ad + (uint64_t)(m * 256), bd + (uint64_t)((m & 1) * 1024), idesc, t + 256, t + 320);
}
// ===== M1 probe (a), moe-layer.md: die-local routed-expert weight stream from a persistent 1-CTA/SM kernel at the
// per-B active-set byte counts (served census rev 7: A x 1.0967 MB per GPU per layer), 36 KiB stages, mixed MXFP8xE2M1
// SS MMA consumer (N tokens), 10-layer PDL graph, cold weights (each layer its own region). Stamps per CTA: entry,
// first FULL stage (first byte), last commit landed (end). Derived from an internal ring probe (same kernel body + entry stamp +
// NOPS bulk ops per stage). Not production code.
struct Arm { int S, SB, NOPS, N; const char* name; };
__global__ void __launch_bounds__(96, 1) s52(const unsigned char* d0, const unsigned char* d1, long long nstage, Arm a, int n0, int n1,
                                             unsigned idesc, int layer, unsigned long long* st, unsigned long long* sink) {
  extern __shared__ __align__(1024) unsigned char dsm[];
  __shared__ __align__(8) unsigned long long full[8], empty[8];
  __shared__ uint32_t tmem_base;
  const unsigned long long tent = gtime();
  asm volatile("griddepcontrol.launch_dependents;");
  unsigned s = smid() & 1023; unsigned dom = g_dom[s] & 1; int rk = g_rank[s];
  unsigned char* ring = (unsigned char*)(((size_t)dsm + 1023) & ~(size_t)1023);
  const int w = threadIdx.x / 32, lane = threadIdx.x % 32;
  if (threadIdx.x == 0) for (int i = 0; i < a.S; ++i) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(sa(&full[i])));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(sa(&empty[i])));
  }
  if (w == 1) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;" :: "r"(sa(&tmem_base)) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  asm volatile("tcgen05.fence::before_thread_sync;");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;");
  const bool active = rk < (1 << 19);
  const unsigned char* buf = dom ? d1 : d0; long long ncta = dom ? n1 : n0;
  const int SB = a.SB, OP = SB / a.NOPS;
  const long long n = active ? nstage / ncta + (rk < nstage % ncta) : 0;
  unsigned long long* my = st + ((size_t)layer * 1024 + s) * 4;
  if (active && w == 0 && lane == 0) {
    const int S = a.S; const int ni = (int)n; int sl = 0, round = 0;
    for (int i = 0; i < ni; ++i) {
      if (round > 0) wait_par(sa(&empty[sl]), (unsigned)((round - 1) & 1));
      const unsigned b = sa(&full[sl]); const size_t off = (size_t)(rk + (long long)i * ncta) * SB;
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(SB) : "memory");
      for (int o = 0; o < a.NOPS; ++o) bulk(sa(ring + (size_t)sl * SB + o * OP), buf + off + o * OP, OP, b);
      if (++sl == S) { sl = 0; ++round; }
    }
  } else if (active && w == 1) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
    const uint32_t t = tmem_base;
    const uint32_t bsm = sa(ring + (size_t)a.S * SB);
    const uint64_t bd = sdesc(bsm);
    unsigned long long t0 = 0;
    const int nm = SB / 4096;
    const int ni = (int)n; int sl = 0; unsigned ph = 0;
    for (int i = 0; i < ni; ++i) {
      wait_par(sa(&full[sl]), ph);
      if (i == 0) t0 = gtime();
      asm volatile("tcgen05.fence::after_thread_sync;");
      const uint64_t ad = sdesc(sa(ring + (size_t)sl * SB));
      if (nm == 9) mma_n<9>(t, ad, bd, idesc); else if (nm == 16) mma_n<16>(t, ad, bd, idesc); else mma_n<4>(t, ad, bd, idesc);
      commit_el(sa(&empty[sl]));
      if (++sl == a.S) { sl = 0; ph ^= 1u; }
    }
    if (n) { const long long i = n - 1; const int sl2 = (int)(i % a.S); wait_par(sa(&empty[sl2]), (unsigned)((i / a.S) & 1)); }
    if (lane == 0) { my[0] = tent; my[1] = t0; my[2] = gtime(); my[3] = (unsigned long long)n; }
  }
  asm volatile("tcgen05.fence::before_thread_sync;");
  __syncthreads();
  if (w == 1) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" :: "r"(tmem_base) : "memory");
}
static double pct(std::vector<double> v, double p) { std::sort(v.begin(), v.end()); return v.empty() ? 0 : v[(size_t)(p * (v.size() - 1))]; }

int main(int argc, char** argv) {
  RK(cudaSetDevice(0)); RK(cudaFree(0)); cudaEventCreate(&e0); cudaEventCreate(&e1);
  int nsm; cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
  const int NL = 10;
  const double SLICE = 33030144.0 * 0.53125 / 16;  // moe-layer.predict.py SLICE
  const int Bs[] = {1, 2, 4, 8, 16, 32, 64};
  const double A[] = {75.1, 133.1, 217.3, 330.1, 460.1, 585.6, 680.8};
  const double MAXB = 680.8 * SLICE;
  size_t per_die = (((size_t)(MAXB / 2) + (4 << 20)) + (2 << 20) - 1) / (2 << 20) * (2 << 20), B = per_die * NL + (64 << 20);
  const bool il = argc > 2 && argv[2][0] == 'i';  // "il": default cudaMalloc (4 KiB die-interleaved) weights, as today's checkpoint buffers
  unsigned char *d0, *d1;
  if (il) { RK(cudaMalloc(&d0, B)); RK(cudaMalloc(&d1, B)); printf("== weights: cudaMalloc (die-interleaved)\n"); }
  else { d0 = (unsigned char*)alloc_dom(0, 0, B); d1 = (unsigned char*)alloc_dom(0, 1, B); printf("== weights: die-local VMM pools\n"); }
  RK(cudaMemset(d0, 0x22, B)); RK(cudaMemset(d1, 0x22, B));
  int nb = 8 * nsm; unsigned* lo; RK(cudaMalloc(&lo, nb * 16)); RK(cudaMemset(lo, 0, nb * 16));
  lat_k<<<nb, 32>>>((const unsigned long long*)d0, (const unsigned long long*)d1, (1ull << 30) / 8, lo); RK(cudaDeviceSynchronize());
  std::vector<unsigned> hl(nb * 4); RK(cudaMemcpy(hl.data(), lo, nb * 16, cudaMemcpyDeviceToHost));
  std::vector<double> a0(1024, 0), a1(1024, 0); std::vector<int> cnt(1024, 0);
  for (int b = 0; b < nb; ++b) { unsigned s = hl[4 * b]; if (s >= 1024) continue; a0[s] += hl[4 * b + 1]; a1[s] += hl[4 * b + 2]; cnt[s]++; }
  std::vector<unsigned char> dom(1024, 0); std::vector<int> rank(1024, 1 << 20); int per[2] = {0, 0};
  {  // split at the median of (lat die0 - lat die1): 2 dies x nsm/2 SMs ; noisy per-SM classification
    std::vector<std::pair<double, int>> v;
    for (int s = 0; s < 1024; ++s) if (cnt[s]) v.push_back({a0[s] / cnt[s] - a1[s] / cnt[s], s});
    std::sort(v.begin(), v.end());
    int nraw0 = 0; for (auto& x : v) nraw0 += x.first < 0;
    for (size_t i = 0; i < v.size(); ++i) { int d = i < v.size() / 2 ? 0 : 1; dom[v[i].second] = d; rank[v[i].second] = per[d]++; }
    printf("== die map: raw sign classification die0 %d of %zu; median split used\n", nraw0, v.size());
  }
  printf("== m1a stream probe; die0 %d SMs, die1 %d SMs\n", per[0], per[1]);
  RK(cudaMemcpyToSymbol(g_dom, dom.data(), 1024)); RK(cudaMemcpyToSymbol(g_rank, rank.data(), 1024 * sizeof(int)));
  unsigned long long *sink, *st; RK(cudaMalloc(&sink, 8));
  size_t stn = (size_t)NL * 1024 * 4; RK(cudaMalloc(&st, stn * 8));
  std::vector<unsigned long long> hs(stn);
  cudaStream_t sm; RK(cudaStreamCreate(&sm));
  unsigned char* flush; RK(cudaMalloc(&flush, 256 << 20));
  Arm arms[] = {
      {5, 36864, 2, 32, "5x36K 2op N32 (design ring)"},
      {5, 36864, 3, 32, "5x36K 3x12K-op N32 (FC2 13K tiles)"},
      {4, 36864, 2, 32, "4x36K 2op N32 (<=192KB co-resident)"},
      {6, 18432, 1, 32, "6x18K 1op N32 (today's ring)"},
      {5, 36864, 2, 64, "5x36K 2op N64"},
  };
  int only = argc > 1 ? atoi(argv[1]) : -1;
  printf("arm | B | MB/layer | graph us/layer | kernel-time TB/s | span us (first full..last end) | span TB/s | first-byte us p50/p90 (entry->first full) | end spread us (p50 end..max end) | stages/CTA\n");
  for (int ai = 0; ai < (int)(sizeof(arms) / sizeof(arms[0])); ++ai) {
    if (only >= 0 && ai != only) continue;
    const Arm a = arms[ai];
    size_t smem = (size_t)a.S * a.SB + 32768 + 1024;
    unsigned idesc = mk(0, 5, 0, a.N, 128, 1, 1, 0);
    RK(cudaFuncSetAttribute(s52, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    RK(cudaFuncSetAttribute(s52, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    for (int bi = 0; bi < 7; ++bi) {
      const double LB = A[bi] * SLICE;
      long long nstage = (long long)(LB / 2 / a.SB);
      cudaGraph_t g; cudaGraphExec_t ge;
      RK(cudaStreamBeginCapture(sm, cudaStreamCaptureModeGlobal));
      for (int l = 0; l < NL; ++l) {
        cudaLaunchConfig_t cfg = {}; cfg.gridDim = nsm; cfg.blockDim = 96; cfg.dynamicSmemBytes = smem; cfg.stream = sm;
        cudaLaunchAttribute at[1]; at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        at[0].val.programmaticStreamSerializationAllowed = 1; cfg.attrs = at; cfg.numAttrs = 1;
        RK(cudaLaunchKernelEx(&cfg, s52, (const unsigned char*)(d0 + (size_t)l * per_die), (const unsigned char*)(d1 + (size_t)l * per_die),
                              nstage, a, per[0], per[1], idesc, l, st, sink));
      }
      RK(cudaStreamEndCapture(sm, &g)); RK(cudaGraphInstantiate(&ge, g, 0));
      // L2 flush between replays: weights are cold in-step (layer stride >> L2); still flush the tail of the last replay
      float best = 1e9; std::vector<double> spans, fb50, fb90, endsp, rates;
      for (int r = 0; r < 8; ++r) {
        RK(cudaMemsetAsync(flush, r, 256 << 20, sm));
        RK(cudaEventRecord(e0, sm)); RK(cudaGraphLaunch(ge, sm)); RK(cudaEventRecord(e1, sm)); RK(cudaEventSynchronize(e1));
        float ms; cudaEventElapsedTime(&ms, e0, e1); if (r == 0) continue; best = std::min(best, ms);
        RK(cudaMemcpy(hs.data(), st, stn * 8, cudaMemcpyDeviceToHost));
        for (int l = 1; l < NL; ++l) {  // skip layer 0 (graph head)
          std::vector<double> fb, en; unsigned long long t0min = ~0ull, endmax = 0; std::vector<unsigned long long> t0s;
          for (int s = 0; s < 1024; ++s) if (cnt[s]) {
            unsigned long long* m = &hs[((size_t)l * 1024 + s) * 4]; if (!m[3]) continue;
            fb.push_back((double)(m[1] - m[0]) / 1e3); t0s.push_back(m[1]); en.push_back((double)m[2]);
            t0min = std::min(t0min, m[1]); endmax = std::max(endmax, m[2]);
          }
          std::sort(t0s.begin(), t0s.end()); unsigned long long t0med = t0s[t0s.size() / 2];
          double span = (double)(endmax - t0med) / 1e3; spans.push_back(span); rates.push_back(2.0 * nstage * a.SB / (span * 1e-6) / 1e12);
          fb50.push_back(pct(fb, .5)); fb90.push_back(pct(fb, .9));
          endsp.push_back(((double)endmax - pct(en, .5)) / 1e3);
        }
      }
      double us = best * 1e3 / NL, bytes = 2.0 * nstage * a.SB;
      printf("%s | B%d | %.1f | %.2f | %.2f | %.2f (p10 %.2f p90 %.2f) | %.2f | %.2f/%.2f | %.2f | %.1f\n", a.name, Bs[bi], bytes / 1e6, us,
             bytes / (us * 1e-6) / 1e12, pct(spans, .5), pct(spans, .1), pct(spans, .9), pct(rates, .5), pct(fb50, .5), pct(fb90, .5),
             pct(endsp, .5), (double)nstage / std::min(per[0], per[1]));
      fflush(stdout);
      cudaGraphExecDestroy(ge); cudaGraphDestroy(g);
    }
  }
  printf("DONE\n");
  return 0;
}
