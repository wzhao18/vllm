// dattn_skel_cc.cu (D-ATTN M6a step 1): compute-chain MMA skeleton for the sparse MLA decode core on VR200 (sm_107a).
// Garbage FP8 operands, no gather, no softmax: one CTA per SM (or 2-CTA pairs) issues the per-128-key-tile MMA list of a
// candidate form NT times back to back, clock64 span / NT = MMA cycles per tile. Verifies the smem-operand model
//   t_MMA = sum over MMAs of max(FLOP_per_CTA / 32768, (A_smem_bytes + B_smem_bytes_per_CTA) / 128 B/clk)
// (hwdb INDEX 12 + topic 01) for today's km form and the 2-CTA candidates before any kernel is built.
// Forms (per CTA, per 128 keys of one verify row unless noted):
//   km   keys-on-M SS: QK 9 x M128 N96 K64 (A = K tile, B = Q^T), PV 8 x M128 N96 K64 (A = V^T, B = P^T)
//   tsr  same 17 MMAs with A from TMEM (math + B only reference)
//   qk2  keys-on-M 2-CTA QK: 9 x M256 N96 K64 (256 keys of one row per pair, B = Q^T split 48/48)
//   hm2  heads-on-M 2-CTA (2 rows of one request, union keys), per 64-key subtile: QK 9 x M256 N64 K64 SS (A = padded Q),
//        PV 2 x M256 N256 K64 SS (A = P smem); x2 subtiles per 128 keys
//   hm2t hm2 with PV TS (P in TMEM)
// Usage: dattn_skel_cc [NT=256]
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
__device__ __forceinline__ uint64_t sdesc(uint32_t saddr) {
  return (uint64_t)((saddr >> 4) & 0x3FFF) | ((uint64_t)1 << 16) | ((uint64_t)64 << 32) | ((uint64_t)1 << 46) | ((uint64_t)2 << 61);
}
struct Op { unsigned idesc; int ts; int boff; int doff; };
__constant__ Op c_ops[64];
constexpr unsigned mkc(int N, int M) { return (1u << 4) | ((unsigned)(N >> 3) << 17) | ((unsigned)(M >> 4) << 24) | (1u << 29); }
template <int CG, int TS>
__device__ __forceinline__ void mma(uint32_t d, uint32_t ta, uint64_t ad, uint64_t bd, uint32_t idesc) {
  if (CG == 2) {
    if (TS) asm volatile("tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [%1], %2, %3, 1;" :: "r"(d), "r"(ta), "l"(bd), "r"(idesc));
    else asm volatile("tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], %1, %2, %3, 1;" :: "r"(d), "l"(ad), "l"(bd), "r"(idesc));
  } else {
    if (TS) asm volatile("tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [%1], %2, %3, 1;" :: "r"(d), "r"(ta), "l"(bd), "r"(idesc));
    else asm volatile("tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, 1;" :: "r"(d), "l"(ad), "l"(bd), "r"(idesc));
  }
}
// form: REP x (C1 MMAs of M x N1 (TS1) then C2 MMAs of M x N2 (TS2)), K64 FP8
template <int M_, int N1_, int C1_, int TS1_, int N2_, int C2_, int TS2_, int REP_>
struct Fm { static constexpr int M = M_, N1 = N1_, C1 = C1_, TS1 = TS1_, N2 = N2_, C2 = C2_, TS2 = TS2_, REP = REP_;
  static constexpr unsigned I1 = mkc(N1_, M_), I2 = mkc(N2_ ? N2_ : 64, M_); };
template <int CG, class F>
__global__ void __launch_bounds__(128, 1) cc_k(int nops, int nt, unsigned long long* out) {
  extern __shared__ __align__(1024) unsigned char smem[];
  __shared__ uint32_t tmem_base;
  __shared__ __align__(8) uint64_t mbar;
  uint32_t s0 = (uint32_t)__cvta_generic_to_shared(smem);
  uint32_t sa = (s0 + 1023) & ~1023u, sb = sa + 98304;
  uint32_t tb = (uint32_t)__cvta_generic_to_shared(&tmem_base), mb = (uint32_t)__cvta_generic_to_shared(&mbar);
  unsigned rank = 0;
  if (CG == 2) asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
  if (threadIdx.x < 32) {
    if (CG == 2) {
      asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], 512;" :: "r"(tb) : "memory");
      asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    } else {
      asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;" :: "r"(tb) : "memory");
      asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
  }
  if (threadIdx.x == 32) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(mb) : "memory");
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  asm volatile("tcgen05.fence::before_thread_sync;");
  if (CG == 2) { asm volatile("barrier.cluster.arrive.release.aligned;"); asm volatile("barrier.cluster.wait.acquire.aligned;"); }
  else __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;");
  const uint32_t ta = tmem_base + 448;
  long long c0 = clock64(); unsigned long long g0; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g0));
  if (rank == 0 && threadIdx.x == 0) {
    const uint64_t ad = sdesc(sa);
    // compile-time op list (template F): descriptors are immediates / uniform adds, so the single issuing thread is not
    // the bound (a runtime op table costs R2UR moves per MMA: 150-215 clk/MMA measured in cc2/cc3)
    const uint64_t bd0 = sdesc(sb);
    for (int t = 0; t < nt; ++t) {
#pragma unroll
      for (int r = 0; r < F::REP; ++r) {
#pragma unroll
        for (int i = 0; i < F::C1; ++i) mma<CG, F::TS1>(tmem_base, ta, ad, bd0 + (uint64_t)((i % 4) * 512), F::I1);
#pragma unroll
        for (int i = 0; i < F::C2; ++i) mma<CG, F::TS2>(tmem_base + (F::N2 <= 96 ? 96 * (i % 2) : 0), ta, ad, bd0 + (uint64_t)((i % 4) * 512), F::I2);
      }
    }
    if (CG == 2) {
      unsigned short mask = 3;
      asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;" :: "r"(mb), "h"(mask) : "memory");
    } else {
      asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(mb) : "memory");
    }
  }
  if (threadIdx.x == 0) {
    asm volatile("{.reg .pred P; W: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], 0; @!P bra W;}" :: "r"(mb) : "memory");
    long long c1 = clock64(); unsigned long long g1; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g1));
    if (rank == 0) atomicMax(out, (unsigned long long)(c1 - c0));
    if (blockIdx.x == 0) { out[1] = c1 - c0; out[2] = g1 - g0; }
  }
  asm volatile("tcgen05.fence::before_thread_sync;");
  __syncthreads();
  if (CG == 2) { asm volatile("barrier.cluster.arrive.release.aligned;"); asm volatile("barrier.cluster.wait.acquire.aligned;"); }
  if (threadIdx.x < 32) {
    if (CG == 2) asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, 512;" :: "r"(tmem_base) : "memory");
    else asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" :: "r"(tmem_base) : "memory");
  }
}
// kind::f8f6f4 E4M3 x E4M3 -> F32, K-major A/B, 2x-K (K64) = idesc bit 29
static unsigned mk(int N, int M) { return (1u << 4) | ((unsigned)(N >> 3) << 17) | ((unsigned)(M >> 4) << 24) | (1u << 29); }
// model: per CTA, per MMA max(FLOP/32768, smem bytes/128); K64 FP8: A bytes/CTA = 128*64 (SS), B bytes/CTA = N/CG*64
static double mclk(int cnt, int M, int N, int ts, int cg) {
  double fl = 2.0 * (M / cg) * N * 64 / 32768.0, by = (ts ? 0 : 128 * 64) + (double)N / cg * 64;
  return cnt * (fl > by / 128 ? fl : by / 128);
}
static int g_sms; static unsigned long long* g_out;
template <int CG, class F>
static void run(const char* name, int NT) {
  const double pred = F::REP * (mclk(F::C1, F::M, F::N1, F::TS1, CG) + (F::C2 ? mclk(F::C2, F::M, F::N2, F::TS2, CG) : 0));
  const int nm = F::REP * (F::C1 + F::C2);
  cudaFuncSetAttribute(cc_k<CG, F>, cudaFuncAttributeMaxDynamicSharedMemorySize, 200 * 1024);
  unsigned long long best = ~0ull; double mhz = 0; cudaError_t e = cudaSuccess;
  for (int rep = 0; rep < 3; ++rep) {
    cudaMemset(g_out, 0, 32);
    if (CG == 2) {
      cudaLaunchConfig_t cfg = {}; cudaLaunchAttribute at[1];
      at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = 2; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
      cfg.gridDim = g_sms / 2 * 2; cfg.blockDim = 128; cfg.dynamicSmemBytes = 200 * 1024; cfg.attrs = at; cfg.numAttrs = 1;
      e = cudaLaunchKernelEx(&cfg, cc_k<CG, F>, nm, NT, g_out);
    } else cc_k<CG, F><<<g_sms, 128, 200 * 1024>>>(nm, NT, g_out);
    if (e == cudaSuccess) e = cudaDeviceSynchronize();
    if (e != cudaSuccess) break;
    unsigned long long c[3]; cudaMemcpy(c, g_out, 24, cudaMemcpyDeviceToHost); if (c[0] < best) best = c[0]; if (c[2]) mhz = (double)c[1] / c[2] * 1e3;
  }
  if (e != cudaSuccess) { printf("CC %-46s ERROR %s\n", name, cudaGetErrorString(e)); exit(1); }
  const double ct = (double)best / NT;
  printf("CC %-46s %2d MMA | %7.1f clk/tile %5.1f clk/MMA (%.3f us @ %4.0f MHz) | model %7.1f (%+.0f%%)\n", name, nm, ct, ct / nm, ct / mhz, mhz, pred,
         100.0 * (ct - pred) / pred);
}
int main(int argc, char** argv) {
  const int NT = argc > 1 ? atoi(argv[1]) : 256;
  cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, 0); cudaMalloc(&g_out, 32);
  printf("SKEL-CC sms=%d NT=%d (per CTA, per 128-key tile of one verify row unless noted)\n", g_sms, NT);
  run<2, Fm<256, 256, 8, 0, 0, 0, 0, 1>>("ref2 2-CTA SS M256 N256 (mma2cta 134.7)", NT);
  run<1, Fm<128, 128, 8, 0, 0, 0, 0, 1>>("ref1 1-CTA SS M128 N128", NT);
  run<1, Fm<128, 128, 8, 1, 0, 0, 0, 1>>("ref1t 1-CTA TS M128 N128", NT);
  run<1, Fm<128, 96, 9, 0, 96, 8, 0, 1>>("km   keys-on-M SS QK9 + PV8 N96 (today)", NT);
  run<1, Fm<128, 96, 9, 1, 96, 8, 1, 1>>("tsr  keys-on-M TS QK9 + PV8 N96 (math+B ref)", NT);
  run<1, Fm<128, 96, 9, 0, 0, 0, 0, 1>>("qk1  keys-on-M QK9 SS N96", NT);
  run<1, Fm<128, 96, 8, 0, 0, 0, 0, 1>>("pv1  keys-on-M PV8 SS N96", NT);
  run<1, Fm<128, 192, 9, 0, 0, 0, 0, 1>>("qkN2 keys-on-M QK9 SS N192 (2 rows, /2 per row)", NT);
  run<2, Fm<256, 96, 9, 0, 0, 0, 0, 1>>("qk2  keys-on-M 2-CTA QK9 M256 N96", NT);
  run<2, Fm<256, 64, 9, 0, 256, 2, 0, 2>>("hm2  heads-on-M 2-CTA 2x(QK9 N64 + PV2 N256) SS", NT);
  run<2, Fm<256, 64, 9, 0, 256, 2, 1, 2>>("hm2t heads-on-M 2-CTA QK SS + PV TS", NT);
  printf("DONE\n");
  return 0;
}
