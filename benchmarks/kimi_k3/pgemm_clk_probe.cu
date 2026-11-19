// mma2cta.cu: tcgen05.mma cta_group::2 (M256 across a CTA pair) issue throughput on sm_107a vs the 1-CTA SS cap
// (mma_rate.cu: 1-CTA SS capped at 2/3 of the 2x-K peak by smem operand fetch). Garbage operands; the leader CTA's
// thread 0 issues ITERS MMAs into one TMEM accumulator, commits with multicast to both CTAs' mbarriers, both wait.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <chrono>
__device__ __forceinline__ uint64_t sdesc(uint32_t saddr) {
  return (uint64_t)((saddr >> 4) & 0x3FFF) | ((uint64_t)1 << 16) | ((uint64_t)64 << 32) | ((uint64_t)1 << 46) | ((uint64_t)2 << 61);
}
template <int KIND>
__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(128, 1) mma2(unsigned idesc, int iters, unsigned long long* out, int rnd) {
  extern __shared__ __align__(1024) unsigned char smem[];
  __shared__ uint32_t tmem_base;
  __shared__ __align__(8) uint64_t mbar;
  uint32_t s0 = (uint32_t)__cvta_generic_to_shared(smem);
  uint32_t sa = (s0 + 1023) & ~1023u, sb = sa + 32768;
  uint32_t tb = (uint32_t)__cvta_generic_to_shared(&tmem_base), mb = (uint32_t)__cvta_generic_to_shared(&mbar);
  unsigned rank; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
  int warp = threadIdx.x / 32;
  if (rnd) {  // random operand bytes (realistic datapath toggling); generic writes -> async proxy fence below
    uint32_t x = 2463534242u ^ (blockIdx.x * 977u + threadIdx.x * 131u);
    unsigned char* base = smem + (sa - s0);
    for (int i = threadIdx.x; i < 16384; i += blockDim.x) {
      x ^= x << 13; x ^= x >> 17; x ^= x << 5;
      reinterpret_cast<uint32_t*>(base)[i] = x & 0x7e7e7e7eu;  // finite E4M3 bytes, no NaN
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  if (warp == 0) {
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], 512;" :: "r"(tb) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
  }
  if (threadIdx.x == 32) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(mb) : "memory");
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  asm volatile("tcgen05.fence::before_thread_sync;");
  asm volatile("barrier.cluster.arrive.release.aligned;"); asm volatile("barrier.cluster.wait.acquire.aligned;");
  asm volatile("tcgen05.fence::after_thread_sync;");
  uint32_t t = tmem_base, d = t, sfa = t + 256, sfb = t + 320;
  long long c0 = clock64();
  if (rank == 0 && threadIdx.x == 0) {
    uint64_t ad = sdesc(sa), bd = sdesc(sb);
    for (int i = 0; i < iters; ++i) {
      if (KIND == 0)
        asm volatile("{.reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;}" :: "r"(d), "l"(ad), "l"(bd), "r"(idesc), "r"(i) : "memory");
      else if (KIND == 1)
        asm volatile("{.reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], %1, %2, %3, p;}" :: "r"(d), "l"(ad), "l"(bd), "r"(idesc), "r"(i) : "memory");
      else if (KIND == 2)
        asm volatile("{.reg .pred p; setp.ne.b32 p, %6, 0; tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%4], [%5], p;}" :: "r"(d), "l"(ad), "l"(bd), "r"(idesc), "r"(sfa), "r"(sfb), "r"(i) : "memory");
      else
        asm volatile("{.reg .pred p; setp.ne.b32 p, %6, 0; tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%4], [%5], p;}" :: "r"(d), "l"(ad), "l"(bd), "r"(idesc), "r"(sfa), "r"(sfb), "r"(i) : "memory");
    }
    unsigned short mask = 3;
    asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;" :: "r"(mb), "h"(mask) : "memory");
  }
  if (threadIdx.x == 0) {
    asm volatile("{.reg .pred P; W: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], 0; @!P bra W;}" :: "r"(mb) : "memory");
    long long c1 = clock64();
    if (blockIdx.x == 0) out[0] = c1 - c0;
  }
  asm volatile("tcgen05.fence::before_thread_sync;");
  __syncthreads();
  asm volatile("barrier.cluster.arrive.release.aligned;"); asm volatile("barrier.cluster.wait.acquire.aligned;");
  if (warp == 0) asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, 512;" :: "r"(t) : "memory");
}
static unsigned mk(int dfmt_or_bsf, int afmt, int bfmt, int N, int M, int sf_ue8m0, int bit31, int bit3) {
  return ((unsigned)dfmt_or_bsf << 4) | ((unsigned)afmt << 7) | ((unsigned)bfmt << 10) | ((unsigned)(N >> 3) << 17) |
         ((unsigned)sf_ue8m0 << 23) | ((unsigned)(M >> 4) << 24) | ((unsigned)bit31 << 31) | ((unsigned)bit3 << 3);
}
struct Case { const char* name; int kind; int M, N, K; unsigned idesc; };

// --- P-GEMM HW-CLK probe host (2026-11-19): sustained 2-CTA mxf8f6f4 M256 N256 K64 loop; per launch the leader's
// clock64 delta / event wall time = effective SM clock; NVML (dlopen) sampler at 10 ms: SM clock, power, limit, reasons.
#include <dlfcn.h>
#include <atomic>
#include <thread>
#include <vector>
#include <algorithm>
typedef int (*f_init)();
typedef int (*f_hbybus)(const char*, void**);
typedef int (*f_clk)(void*, int, unsigned*);
typedef int (*f_u)(void*, unsigned*);
typedef int (*f_ull)(void*, unsigned long long*);
typedef int (*f_temp)(void*, int, unsigned*);
struct Smp { double t; unsigned mhz, mw; unsigned long long rs; unsigned temp; };
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int main(int argc, char** argv) {
  int rnd = argc > 1 ? atoi(argv[1]) : 1; double secs = argc > 2 ? atof(argv[2]) : 10.0;
  int sms; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  char bus[32]; cudaDeviceGetPCIBusId(bus, 32, 0);
  void* h = dlopen("libnvidia-ml.so.1", RTLD_NOW); void* dev = nullptr; unsigned limit = 0;
  f_clk gclk = nullptr; f_u gpow = nullptr; f_ull grs = nullptr; f_temp gtmp = nullptr;
  if (h) {
    ((f_init)dlsym(h, "nvmlInit_v2"))();
    f_hbybus hb = (f_hbybus)dlsym(h, "nvmlDeviceGetHandleByPciBusId_v2");
    if (hb(bus, &dev) != 0) dev = nullptr;
    gclk = (f_clk)dlsym(h, "nvmlDeviceGetClockInfo"); gpow = (f_u)dlsym(h, "nvmlDeviceGetPowerUsage");
    grs = (f_ull)dlsym(h, "nvmlDeviceGetCurrentClocksEventReasons");
    if (!grs) grs = (f_ull)dlsym(h, "nvmlDeviceGetCurrentClocksThrottleReasons");
    gtmp = (f_temp)dlsym(h, "nvmlDeviceGetTemperature");
    if (dev) ((f_u)dlsym(h, "nvmlDeviceGetEnforcedPowerLimit"))(dev, &limit);
  }
  printf("bus %s nvml %s power_limit_w %.0f sms %d rnd %d\n", bus, dev ? "ok" : "NO", limit / 1000.0, sms, rnd);
  std::vector<Smp> smp; std::atomic<bool> stop{false};
  std::thread th([&] { while (!stop) { Smp s{now(), 0, 0, 0, 0}; if (dev) { gclk(dev, 1, &s.mhz); gpow(dev, &s.mw); if (grs) grs(dev, &s.rs); gtmp(dev, 0, &s.temp); }
                                       smp.push_back(s); std::this_thread::sleep_for(std::chrono::milliseconds(10)); } });
  cudaFuncSetAttribute(mma2<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, 100 * 1024);
  unsigned idesc = mk(0, 0, 0, 256, 256, 1, 1, 0);
  unsigned long long* out; cudaMalloc(&out, 64);
  const int ITERS = 200000;  // ~11 ms at 2.4 GHz (135 clk/MMA)
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  struct L { double t, ms; unsigned long long cyc; }; std::vector<L> ls;
  double t0 = now();
  while (now() - t0 < secs) {
    cudaEventRecord(a); mma2<2><<<sms / 2 * 2, 128, 100 * 1024>>>(idesc, ITERS, out, rnd); cudaEventRecord(b);
    cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b);
    unsigned long long cyc; cudaMemcpy(&cyc, out, 8, cudaMemcpyDeviceToHost); ls.push_back({now() - t0, ms, cyc});
  }
  stop = true; th.join();
  cudaError_t e = cudaGetLastError(); printf("cuda %s launches %zu\n", cudaGetErrorString(e), ls.size());
  for (double w = 0; w < secs; w += 1.0) {  // 1 s windows
    std::vector<double> fe, cpm; std::vector<unsigned> mh, pw; unsigned long long rs = 0; unsigned tp = 0;
    for (auto& l : ls) if (l.t >= w && l.t < w + 1) { fe.push_back(l.cyc / (l.ms * 1e3)); cpm.push_back((double)l.cyc / ITERS); }
    for (auto& s : smp) if (s.t - t0 >= w && s.t - t0 < w + 1) { mh.push_back(s.mhz); pw.push_back(s.mw); rs |= s.rs; tp = s.temp; }
    if (fe.empty()) continue;
    std::sort(fe.begin(), fe.end()); std::sort(cpm.begin(), cpm.end()); std::sort(mh.begin(), mh.end()); std::sort(pw.begin(), pw.end());
    printf("{\"probe\":\"mma2cta_mxf8\",\"rnd\":%d,\"t\":%.0f,\"f_eff_mhz\":%.0f,\"clk_per_mma\":%.1f,\"nvml_mhz\":%u,\"power_w\":%.0f,\"limit_w\":%.0f,\"reasons\":\"0x%llx\",\"temp_c\":%u}\n",
           rnd, w, fe[fe.size() / 2], cpm[cpm.size() / 2], mh.empty() ? 0 : mh[mh.size() / 2], pw.empty() ? 0 : pw[pw.size() / 2] / 1000.0, limit / 1000.0, rs, tp);
  }
  return 0;
}
