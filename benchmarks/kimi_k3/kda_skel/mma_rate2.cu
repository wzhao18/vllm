// mma_rate2: mma_rate + M64 (d_v-split half chains: is an M64 MMA half the clk of M128?)
// tcgen05 MMA rate probe for the BT64 KDA chain step (P-KDA2 M1): clk per MMA for kind::tf32 and kind::f16 with
// A from TMEM (TS) or smem (SS), M128 x N{32,64,128}; 1 or 2 concurrent issuers per SM (two chains sharing the
// tensor pipe); all SMs run at once. Issue is fully unrolled with loop-invariant descriptors (UR operands), so the
// number is the pipe rate, not the issue cost. LOOPY=1 variant re-derives descriptors per MMA (issue-cost check).
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <algorithm>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

__device__ __forceinline__ uint64_t sdesc(uint32_t saddr) {
  return (uint64_t)((saddr >> 4) & 0x3FFF) | ((uint64_t)1 << 16) | ((uint64_t)64 << 32) | ((uint64_t)1 << 46) | ((uint64_t)2 << 61);
}
__device__ __forceinline__ void mwait(uint32_t mb, uint32_t ph) {
  uint32_t ok = 0;
  while (true) {
    asm volatile("{.reg .pred P; mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2; selp.u32 %0, 1, 0, P;}" : "=r"(ok) : "r"(mb), "r"(ph) : "memory");
    if (ok) return;
  }
}
template <int KIND, int TS>
__device__ __forceinline__ void mma(uint32_t d, uint32_t at, uint64_t as, uint64_t b, unsigned idesc, uint32_t acc) {
  if (TS) {
    if (KIND) asm volatile("{.reg .pred P, Q; elect.sync _|P, 0xffffffff; setp.ne.b32 Q, %4, 0; @P tcgen05.mma.cta_group::1.kind::tf32 [%0], [%1], %2, %3, Q;}" :: "r"(d), "r"(at), "l"(b), "r"(idesc), "r"(acc) : "memory");
    else asm volatile("{.reg .pred P, Q; elect.sync _|P, 0xffffffff; setp.ne.b32 Q, %4, 0; @P tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, Q;}" :: "r"(d), "r"(at), "l"(b), "r"(idesc), "r"(acc) : "memory");
  } else {
    if (KIND) asm volatile("{.reg .pred P, Q; elect.sync _|P, 0xffffffff; setp.ne.b32 Q, %4, 0; @P tcgen05.mma.cta_group::1.kind::tf32 [%0], %1, %2, %3, Q;}" :: "r"(d), "l"(as), "l"(b), "r"(idesc), "r"(acc) : "memory");
    else asm volatile("{.reg .pred P, Q; elect.sync _|P, 0xffffffff; setp.ne.b32 Q, %4, 0; @P tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, Q;}" :: "r"(d), "l"(as), "l"(b), "r"(idesc), "r"(acc) : "memory");
  }
}
struct P { int reps, nissue; unsigned idesc; long long* out; };
template <int KIND, int TS, int NM>
__global__ void __launch_bounds__(128, 1) probe(P p) {
  extern __shared__ __align__(1024) unsigned char smem[];
  __shared__ uint32_t tmem_base;
  __shared__ __align__(8) uint64_t mb[2];
  const int warp = threadIdx.x >> 5;
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem);
  uint32_t s0 = (sraw + 1023) & ~1023u;
  if (warp == 0) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;" :: "r"((uint32_t)__cvta_generic_to_shared(&tmem_base)) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  if (threadIdx.x < 2) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"((uint32_t)__cvta_generic_to_shared(&mb[threadIdx.x])) : "memory");
  for (int i = threadIdx.x; i < 64 * 1024 / 4; i += 128) ((uint32_t*)(smem + (s0 - sraw)))[i] = 0;
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  asm volatile("tcgen05.fence::before_thread_sync;");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;");
  const uint32_t tb = tmem_base;
  if (warp < p.nissue) {
    const uint32_t A = tb + warp * 256, D = tb + warp * 256 + 128;
    const uint32_t bs = s0 + warp * 32768;
    const uint64_t bd = sdesc(bs), ad = sdesc(bs + 16384);
    const uint32_t mbar = (uint32_t)__cvta_generic_to_shared(&mb[warp]);
    const unsigned id = p.idesc;
    long long best = 1LL << 62;
    for (int r = 0; r < p.reps; ++r) {
      __syncwarp();
      long long t0 = clock64();
#pragma unroll
      for (int i = 0; i < NM; ++i) mma<KIND, TS>(D, A + (i & 7) * 8, ad + (i & 7) * 2, bd + (i & 7) * 2, id, i ? 1u : 0u);
      asm volatile("{.reg .pred P; elect.sync _|P, 0xffffffff; @P tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];}" :: "r"(mbar) : "memory");
      mwait(mbar, r & 1);
      long long dt = clock64() - t0;
      if (r > 0) best = dt < best ? dt : best;
    }
    if ((threadIdx.x & 31) == 0) p.out[blockIdx.x * 2 + warp] = best;
  }
  asm volatile("tcgen05.fence::before_thread_sync;");
  __syncthreads();
  if (warp == 0) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" :: "r"(tb) : "memory");
}
static unsigned mk(int dfmt, int afmt, int bfmt, int N, int M) {
  return ((unsigned)dfmt << 4) | ((unsigned)afmt << 7) | ((unsigned)bfmt << 10) | ((unsigned)(N >> 3) << 17) | ((unsigned)(M >> 4) << 24);
}
template <int KIND, int TS, int NM>
static void run(int nsm, long long* out, std::vector<long long>& h) {
  auto k = probe<KIND, TS, NM>;
  CK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, 66 * 1024 + 1024));
  for (int M : {128, 64}) for (int N : {32, 64, 128}) for (int ni = 1; ni <= 2; ++ni) {
    P p; p.reps = 6; p.nissue = ni; p.out = out;
    p.idesc = KIND ? mk(1, 2, 2, N, M) : mk(1, 1, 1, N, M);
    CK(cudaMemset(out, 0, nsm * 2 * 8));
    k<<<nsm, 128, 66 * 1024 + 1024>>>(p);
    CK(cudaDeviceSynchronize()); if (cudaGetLastError() != cudaSuccess) { printf("RATE M%d N%d: launch error\n", M, N); continue; }
    CK(cudaMemcpy(h.data(), out, h.size() * 8, cudaMemcpyDeviceToHost));
    std::vector<double> v; for (int b = 0; b < nsm; ++b) for (int w = 0; w < ni; ++w) v.push_back((double)h[b * 2 + w]);
    std::sort(v.begin(), v.end());
    const int K = KIND ? 8 : 16;
    double flop = 2.0 * M * N * K * NM * ni;
    printf("RATE kind %s %s M%d N%d K%d n_mma %d issuers/SM %d: chain clk p50 %.0f max %.0f | per-SM clk/MMA %.1f | FLOP/clk/SM %.0f\n",
           KIND ? "tf32" : "f16 ", TS ? "TS" : "SS", M, N, K, NM, ni, v[v.size() / 2], v.back(), v[v.size() / 2] / (NM * ni), flop / v[v.size() / 2]);
  }
}
int main() {
  int nsm; CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
  long long* out; CK(cudaMalloc(&out, nsm * 2 * 8));
  std::vector<long long> h(nsm * 2);
  run<1, 1, 16>(nsm, out, h); run<1, 1, 64>(nsm, out, h);
  run<1, 0, 64>(nsm, out, h);
  run<0, 1, 16>(nsm, out, h); run<0, 1, 64>(nsm, out, h);
  run<0, 0, 64>(nsm, out, h);
  return 0;
}
