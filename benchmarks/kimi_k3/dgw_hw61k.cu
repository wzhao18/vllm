#include <cmath>
// hw61_ring_law.cu: hw60 chip HBM ring probe, round 2 (verifier fixes); kernel s60 = hw60 + nact
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

// ===== hw52: does the MMA consumer (mixed MXFP8 x E2M1 tcgen05, weights = A from smem, tokens = B N) slow a die-local
// MegaMoE weight stream (656.5 MB/layer, static interleaved split, 1 CTA/SM, graph + PDL as hw45), and does a 2nd
// weight-issuing warp help? Per stage: SB bytes = 2 bulk ops into one barrier (S-PAIR); MMA warp issues SB/4096
// mixed K64 MMAs (4 KiB of packed E2M1 A each) at N tokens, then tcgen05.commit -> EMPTY. N = 0: touch-only consumer
// (hw45 arm, the calibration point). B tokens: static 32 KiB smem region (L2-hot rows already landed).
// Model: layer = max(weight bytes / die-local BW (hw45 37.7 us), stages/SM x MMAs x max(31, N, (4096 + 64 N) / 128) clk).
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
// ===== hw60 (coordinator 21:55 EST, top priority): sustainable chip-level HBM read rate on one VR200.
// TMA bulk-load streams from all SMs (1 CTA/SM), placement: die-local (SM reads its own die), cross (other die),
// striped (one default cudaMalloc buffer, all SMs one static interleaved split). Ops of OB bytes, OPS ops per barrier
// stage, S stages per issuer ring, I issuing warps (one lane each), bytes in flight per SM = S x OPS x OB.
// Consumer: N = 0 touch-only, N > 0 light mixed MXFP8xE2M1 MMA consumer (hw52 form, SB/4096 K64 MMAs per stage).
// dmask: 3 = both dies active, 1 / 2 = only die0 / die1 SMs active (per-die cap). GB = bytes per kernel ("layer").
template <int NM> __device__ __forceinline__ void mma_k(uint32_t t, uint64_t ad, uint64_t bd, unsigned idesc) {
#pragma unroll
  for (int m = 0; m < NM; ++m) mma_mx(t, ad + (uint64_t)(m * 256), bd + (uint64_t)((m & 1) * 1024), idesc, t + 256, t + 320);
}
struct A60 { int pl, OB, OPS, S, I, N, dmask, nact; double GB; const char* name; };
__global__ void __launch_bounds__(192, 1) s60(const unsigned char* d0, const unsigned char* d1, long long nstage, A60 a, int n0, int n1,
                                              unsigned idesc, int layer, unsigned long long* st, unsigned long long* sink) {
  extern __shared__ __align__(1024) unsigned char dsm[];
  __shared__ __align__(8) unsigned long long full[16], empty[16];
  __shared__ uint32_t tmem_base;
  asm volatile("griddepcontrol.launch_dependents;");
  unsigned s = smid() & 1023; unsigned dom = g_dom[s] & 1; int rk = g_rank[s];
  unsigned char* ring = (unsigned char*)(((size_t)dsm + 1023) & ~(size_t)1023);
  const int w = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int S = a.S;
  if (threadIdx.x == 0) for (int i = 0; i < S; ++i) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(sa(&full[i])));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(sa(&empty[i])));
  }
  if (w == 1 && a.N) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;" :: "r"(sa(&tmem_base)) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  asm volatile("tcgen05.fence::before_thread_sync;");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;");
  bool active = rk < (1 << 19) && ((a.dmask >> dom) & 1);
  const unsigned char* buf; long long ncta;
  if (a.pl == 2) { buf = d0; ncta = (long long)n0 + n1; rk = dom ? n0 + rk : rk; }          // striped: one split over all SMs
  else { unsigned src = a.pl == 1 ? dom ^ 1 : dom; buf = src ? d1 : d0; ncta = dom ? n1 : n0; }  // die-local / cross
  if (a.nact > 0 && a.pl != 2) { if (rk >= a.nact) active = false; ncta = ncta < a.nact ? ncta : a.nact; }  // hw61 low-load arms
  const long long SB = (long long)a.OB * a.OPS;
  const long long n = active ? nstage / ncta + (rk < nstage % ncta) : 0;
  unsigned long long* my = st + ((size_t)layer * 1024 + s) * 4;
  const int nis = a.I;
  const int p = w == 0 ? 0 : (w >= 2 ? w - 1 : -1);
  if (active && p >= 0 && p < nis && lane == 0) {
    for (long long i = p; i < n; i += nis) {
      const long long round = i / S; const int sl = (int)(i - round * S);
      if (round > 0) wait_par(sa(&empty[sl]), (unsigned)((round - 1) & 1));
      const unsigned b = sa(&full[sl]); const size_t off = (size_t)(rk + i * ncta) * SB;
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"((unsigned)SB) : "memory");
      for (int o = 0; o < a.OPS; ++o) bulk(sa(ring + (size_t)sl * SB + (size_t)o * a.OB), buf + off + (size_t)o * a.OB, a.OB, b);
    }
  } else if (active && w == 1) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
    const uint32_t t = tmem_base;
    const uint64_t bd = sdesc(sa(ring + (size_t)S * SB));
    unsigned acc = 0; unsigned long long t0 = 0, wcyc = 0;
    const int nm = (int)(SB / 4096);
    int sl = 0; unsigned ph = 0;
    for (long long i = 0; i < n; ++i) {
      const long long c0 = clock64();
      wait_par(sa(&full[sl]), ph);
      if (i == 0) t0 = gtime(); else wcyc += clock64() - c0;
      if (a.N) {
        asm volatile("tcgen05.fence::after_thread_sync;");
        const uint64_t ad = sdesc(sa(ring + (size_t)sl * SB));
        if (nm == 8) mma_k<8>(t, ad, bd, idesc); else if (nm == 16) mma_k<16>(t, ad, bd, idesc);
        else if (nm == 9) mma_k<9>(t, ad, bd, idesc); else mma_k<4>(t, ad, bd, idesc);
        commit_el(sa(&empty[sl]));
      } else {
        if (lane == 0) {
          acc += ring[(size_t)sl * SB + 64];
          asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(sa(&empty[sl])) : "memory");
        }
        __syncwarp();
      }
      if (++sl == S) { sl = 0; ph ^= 1u; }
    }
    if (a.N && n) { const long long i = n - 1; wait_par(sa(&empty[(int)(i % S)]), (unsigned)((i / S) & 1)); }
    if (lane == 0) { my[0] = t0; my[1] = gtime(); my[2] = (unsigned long long)n; my[3] = wcyc; if (acc == 0x12345) sink[0] = acc; }
  }
  asm volatile("tcgen05.fence::before_thread_sync;");
  __syncthreads();
  if (w == 1 && a.N) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" :: "r"(tmem_base) : "memory");
}

// ===== hw61 main (HW lane round 2, 2026-11-20 00:45 EST; verifier points 1-6 on hw60):
// (1) PDL really set on both-die arms (hw60 line 267 kept cfg.attrs inside a comment -> hw60f/g ran without PDL);
// (2) oversized shared-memory mode (driver API) for rings above 227 KiB -> 200-256 KiB in flight per SM;
// (3) nact = active CTAs per die (1-SM arms reconcile the hw51 per-issuer op law; 92/die = 184 CTAs);
// (4) one-op-per-barrier ladder 18K x6/x10/x14, 16K x6/x11, I1/I2 separates depth from bytes per barrier;
// (5) per-SM stage interval (consumer stamps, ns per barrier stage) printed next to the chip rate.
static double pct(std::vector<double> v, double p) { std::sort(v.begin(), v.end()); return v.empty() ? 0 : v[(size_t)(p * (v.size() - 1))]; }

int main(int argc, char** argv) {
  RK(cudaSetDevice(0)); RK(cudaFree(0)); cudaEventCreate(&e0); cudaEventCreate(&e1);
  int nsm, optin, clkkhz; cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
  cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
  cudaDeviceGetAttribute(&clkkhz, cudaDevAttrClockRate, 0);
  const int NL = 6;
  const double MAXGB = 2.6256;  // 4 x 656.4 MB
  size_t per_die = (size_t)(MAXGB * 1e9 / 2) + (4 << 20), B = per_die * NL + (64 << 20);
  const double L = 0.6565, X = 2.6256;
  // pl OB OPS S I N dmask nact GB name
  A60 arms[] = {
      {0, 32768, 1, 6, 2, 0, 3, 0, 0.0461, "K05 local 32K x6 I2 46MB"},
      {0, 32768, 1, 6, 2, 0, 3, 0, 0.0921, "K00 local 32K x6 I2 92MB"},
      {0, 32768, 1, 6, 2, 0, 3, 0, 0.1842, "K06 local 32K x6 I2 184MB"},
      {2, 32768, 1, 6, 2, 0, 3, 0, 0.0921, "K01 striped 32K x6 I2 92MB"},
      {0, 16384, 1, 12, 1, 0, 3, 0, 0.0921, "K02 local 16K x12 I1 92MB"},
      {2, 16384, 1, 12, 1, 0, 3, 0, 0.0921, "K03 striped 16K x12 I1 92MB"},
      {0, 32768, 1, 6, 2, 0, 3, 0, X, "K04 local 32K x6 I2 2.6GB (W08 calibration)"},
  };
  const int narm = sizeof(arms) / sizeof(arms[0]);
  if (argc > 1 && atoi(argv[1]) < 0) { printf("%d\n", narm); return 0; }
  unsigned char* d0 = (unsigned char*)alloc_dom(0, 0, B); unsigned char* d1 = (unsigned char*)alloc_dom(0, 1, B);
  unsigned char* ds; size_t BS = 2 * per_die * NL + (64 << 20); RK(cudaMalloc(&ds, BS));
  RK(cudaMemset(d0, 0x22, B)); RK(cudaMemset(d1, 0x22, B)); RK(cudaMemset(ds, 0x22, BS));
  int nb = 8 * nsm; unsigned* lo; RK(cudaMalloc(&lo, nb * 16)); RK(cudaMemset(lo, 0, nb * 16));
  lat_k<<<nb, 32>>>((const unsigned long long*)d0, (const unsigned long long*)d1, (1ull << 30) / 8, lo); RK(cudaDeviceSynchronize());
  std::vector<unsigned> hl(nb * 4); RK(cudaMemcpy(hl.data(), lo, nb * 16, cudaMemcpyDeviceToHost));
  std::vector<double> a0(1024, 0), a1(1024, 0); std::vector<int> cnt(1024, 0);
  for (int b = 0; b < nb; ++b) { unsigned s = hl[4 * b]; if (s >= 1024) continue; a0[s] += hl[4 * b + 1]; a1[s] += hl[4 * b + 2]; cnt[s]++; }
  std::vector<unsigned char> dom(1024, 0); std::vector<int> rank(1024, 1 << 20); int per[2] = {0, 0};
  for (int s = 0; s < 1024; ++s) if (cnt[s]) { int d = a0[s] / cnt[s] < a1[s] / cnt[s] ? 0 : 1; dom[s] = d; rank[s] = per[d]++; }
  RK(cudaMemcpyToSymbol(g_dom, dom.data(), 1024)); RK(cudaMemcpyToSymbol(g_rank, rank.data(), 1024 * sizeof(int)));
  unsigned long long *sink, *st; RK(cudaMalloc(&sink, 8));
  size_t stn = (size_t)NL * 1024 * 4; RK(cudaMalloc(&st, stn * 8));
  std::vector<unsigned long long> hs(stn);
  cudaStream_t sm; RK(cudaStreamCreate(&sm));
  CUfunction fn; RK(cudaGetFuncBySymbol((cudaFunction_t*)&fn, (const void*)s60));
  int maxov = 0; cuDeviceGetAttribute(&maxov, CU_DEVICE_ATTRIBUTE_MAX_OVERSIZED_SHARED_MEMORY_PER_BLOCK, 0);
  int only = argc > 1 ? atoi(argv[1]) : -1;
  int rep = argc > 2 ? atoi(argv[2]) : 0;
  for (int ai = 0; ai < narm; ++ai) {
    if (only >= 0 && ai != only) continue;
    A60 a = arms[ai];
    const long long SB = (long long)a.OB * a.OPS;
    const int ndie = (a.dmask == 3) ? 2 : 1;
    const double per_side = a.GB * 1e9 / 2;
    long long nstage = a.pl == 2 ? (long long)(a.GB * 1e9 / SB) : (long long)(per_side / SB);
    if (a.nact > 0) nstage = std::min(nstage, (long long)a.nact * (a.GB == L ? 400 : 1600));  // low-load arms: bounded work
    size_t smem = (size_t)a.S * SB + (a.N ? 32768 : 0) + 1024;
    const bool over = (int)smem > optin;
    if (over && (int)smem > std::min(maxov, 320 * 1024)) { printf("== %-44s SKIP smem %zu > %d\n", a.name, smem, maxov); continue; }
    unsigned idesc = mk(0, 5, 0, a.N ? a.N : 16, 128, 1, 1, 0);
    if (over) CK(cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_SHARED_MEMORY_MODE, CU_SHARED_MEMORY_MODE_ALLOW_OVERSIZED_SHARED_MEMORY));
    else RK(cudaFuncSetAttribute(s60, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    RK(cudaFuncSetAttribute(s60, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    const unsigned char *b0 = a.pl == 2 ? ds : d0, *b1 = a.pl == 2 ? ds : d1;
    const size_t stride = a.pl == 2 ? 2 * per_die : per_die;
    const bool pdl = a.dmask == 3 && a.nact == 0;  // nact arms: no PDL (else next-layer CTAs land on idle SMs and skip work)
    cudaGraph_t g; cudaGraphExec_t ge;
    RK(cudaStreamBeginCapture(sm, cudaStreamCaptureModeGlobal));
    for (int l = 0; l < NL; ++l) {
      CUlaunchConfig cfg = {};
      cfg.gridDimX = nsm; cfg.gridDimY = 1; cfg.gridDimZ = 1; cfg.blockDimX = 32 * (1 + std::max(1, a.I)); cfg.blockDimY = 1;
      cfg.blockDimZ = 1; cfg.sharedMemBytes = (unsigned)smem; cfg.hStream = sm;
      CUlaunchAttribute at[2]; int na = 0;
      if (pdl) { at[na].id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION; at[na].value.programmaticStreamSerializationAllowed = 1; ++na; }
      if (over) { at[na].id = CU_LAUNCH_ATTRIBUTE_SHARED_MEMORY_MODE; at[na].value.sharedMemoryMode = CU_SHARED_MEMORY_MODE_ALLOW_OVERSIZED_SHARED_MEMORY; ++na; }
      cfg.attrs = at; cfg.numAttrs = na;
      const unsigned char* p0 = b0 + (size_t)l * stride; const unsigned char* p1 = b1 + (size_t)l * stride;
      int n0 = per[0], n1 = per[1]; int lay = l;
      void* args[] = {&p0, &p1, &nstage, &a, &n0, &n1, &idesc, &lay, &st, &sink};
      CK(cuLaunchKernelEx(&cfg, fn, args, nullptr));
    }
    RK(cudaStreamEndCapture(sm, &g)); RK(cudaGraphInstantiate(&ge, g, 0));
    float best = 1e9;
    for (int r = 0; r < 5; ++r) {
      RK(cudaEventRecord(e0, sm)); RK(cudaGraphLaunch(ge, sm)); RK(cudaEventRecord(e1, sm)); RK(cudaEventSynchronize(e1));
      float ms; cudaEventElapsedTime(&ms, e0, e1); if (r) best = std::min(best, ms);
    }
    RK(cudaMemcpy(hs.data(), st, stn * 8, cudaMemcpyDeviceToHost));
    const double us = best * 1e3 / NL;
    // bytes: active CTAs only (nact arms do bounded work)
    double dt[2] = {0, 0}, db[2] = {0, 0}, f0[2] = {1e30, 1e30}, f1[2] = {0, 0}; std::vector<double> rate[2], sti[2];
    for (int s2 = 0; s2 < 1024; ++s2) {
      if (!cnt[s2]) continue; unsigned long long* m = &hs[((size_t)3 * 1024 + s2) * 4]; if (!m[2]) continue;
      int d = dom[s2]; db[d] += m[2] * (double)SB; f0[d] = std::min(f0[d], (double)m[0]); f1[d] = std::max(f1[d], (double)m[1]);
      rate[d].push_back(m[2] * (double)SB / std::max(1.0, (double)(m[1] - m[0])));
      if (m[2] > 1) sti[d].push_back((double)(m[1] - m[0]) / (double)(m[2] - 1));
    }
    for (int d = 0; d < 2; ++d) dt[d] = f1[d] > f0[d] ? db[d] / (f1[d] - f0[d]) / 1e3 : 0;  // TB/s
    const double bytes = db[0] + db[1];
    printf("== %-44s rep%d %8.2f us/kernel %6.2f TB/s chip | span die0 %5.2f die1 %5.2f | per-SM GB/s p10/p50/p90 d0 %.0f/%.0f/%.0f d1 %.0f/%.0f/%.0f"
           " | stage ns p50 d0 %.0f d1 %.0f | inflight/SM %lldK smem %zu%s pdl%d\n",
           a.name, rep, us, bytes / (us * 1e-6) / 1e12, dt[0], dt[1], pct(rate[0], .1), pct(rate[0], .5), pct(rate[0], .9),
           pct(rate[1], .1), pct(rate[1], .5), pct(rate[1], .9), pct(sti[0], .5), pct(sti[1], .5), (long long)a.S * SB / 1024, smem,
           over ? " OVERSIZED" : "", (int)pdl);
    fflush(stdout);
    cudaGraphExecDestroy(ge); cudaGraphDestroy(g);
  }
  if (only < 0 || only == 0) printf("== map: die0 %d SMs, die1 %d SMs; smem optin %d B, oversized max %d B; clock %d MHz; NL %d\n",
                                    per[0], per[1], optin, maxov, clkkhz / 1000, NL);
  return 0;
}
