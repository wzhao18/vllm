// M1 probe (d)/(i) for moe-layer.md: front-weight landing after the MLK entry, co-resident (<= ~192 KB smem) vs not
// (255 KB), behind an o-proj-consumer stand-in with PDL. Derived from an internal probe. Not production code.
// P (predecessor stand-in): 1 CTA/SM x 256 thr, PS KiB smem, carveout 100, launch_dependents at entry, busy D us, writes x.
// G (MLK entry stand-in): 1 CTA/SM x 128 thr, GS KiB dynamic smem (oversized mode above 227 KiB). Thread 0 TMA-loads its
//   front-weight share FW bytes (4.01 MB / #SMs, one bulk op per 18 KiB) into smem, BEFORE griddepcontrol.wait (pre=1) or
//   after it (pre=0), then waits the weights, reads x. Stamps (globaltimer, int64): P end (max over CTAs), G entry
//   (min/max), G wait return (max), G weights landed (max). A graph of NP (P,G) pairs, each pair its own cold weight region.
// Reported: front-ready after P end = landed_max - P_end (the MLK front's weight critical path beyond the predecessor),
//   pair time - P alone (what G adds to the chain), G entry lead (P_end - G first entry: co-residency).
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#define RK(x) do { cudaError_t r_ = (x); if (r_ != cudaSuccess) { printf("CUDA error %s line %d\n", cudaGetErrorString(r_), __LINE__); exit(1); } } while (0)
__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__global__ void __launch_bounds__(256, 1) kP(int dur_ns, float* act, unsigned long long* st) {
  extern __shared__ __align__(16) unsigned char sm[];
  asm volatile("griddepcontrol.launch_dependents;");
  unsigned long long t0 = gtime();
  if (threadIdx.x == 0) sm[0] = 1;
  while (gtime() - t0 < (unsigned long long)dur_ns) {}
  __syncthreads();
  if (threadIdx.x == 0) { act[blockIdx.x] = (float)sm[0]; atomicMax(&st[0], gtime()); }
}
__device__ __forceinline__ void bulk(uint32_t dst, const void* src, uint32_t bytes, uint32_t mb) {
  asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" :: "r"(dst), "l"(src), "r"(bytes), "r"(mb) : "memory");
}
__device__ __forceinline__ void mwait(uint32_t mb, uint32_t ph) {
  asm volatile("{.reg .pred P; W: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1; @!P bra W;}" :: "r"(mb), "r"(ph) : "memory");
}
__global__ void __launch_bounds__(128, 1) kG(const unsigned char* w, int fw, int pre, int pdl, const float* act, float* out, unsigned long long* st) {
  extern __shared__ __align__(1024) unsigned char sm[];
  __shared__ __align__(8) uint64_t full;
  const unsigned long long te = gtime();
  uint32_t s0 = ((uint32_t)__cvta_generic_to_shared(sm) + 1023) & ~1023u;
  const unsigned char* my = w + (long long)blockIdx.x * fw;
  const uint32_t mb = (uint32_t)__cvta_generic_to_shared(&full);
  float a = 0.f;
  if (threadIdx.x == 0) {
    atomicMin(&st[1], te); atomicMax(&st[4], te);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(mb) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    auto issue = [&]() {
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(mb), "r"(fw) : "memory");
      for (int o = 0; o < fw; o += 18432) bulk(s0 + o, my + o, (uint32_t)min(18432, fw - o), mb);
    };
    if (pre) issue();
    if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");
    atomicMax(&st[2], gtime());
    a = act[blockIdx.x];
    if (!pre) issue();
    mwait(mb, 0);
    atomicMax(&st[3], gtime());
    a += (float)sm[(s0 - (uint32_t)__cvta_generic_to_shared(sm)) + 77];
    out[blockIdx.x] = a;
  }
}
static double pct(std::vector<double> v, double p) { std::sort(v.begin(), v.end()); return v.empty() ? 0 : v[(size_t)(p * (v.size() - 1))]; }
int main() {
  int sms; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  const int NP = 48;
  const int FW = (int)((4.01e6 / sms + 15) / 16) * 16;  // front weights per CTA
  const size_t stride = 16 << 20, pool = (size_t)NP * stride + (64 << 20);
  unsigned char* w; float *act, *out; unsigned long long* st;
  RK(cudaMalloc(&w, pool)); RK(cudaMemset(w, 1, pool));
  RK(cudaMalloc(&act, sms * 4)); RK(cudaMalloc(&out, sms * 4)); RK(cudaMalloc(&st, NP * 8 * 8));
  RK(cudaFuncSetAttribute(kP, cudaFuncAttributeMaxDynamicSharedMemorySize, 200 * 1024));
  RK(cudaFuncSetAttribute(kP, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
  RK(cudaFuncSetAttribute(kG, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
  cudaStream_t s; RK(cudaStreamCreate(&s));
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  printf("== m1d: %d SMs, front weights %d B/CTA (%.2f MB), NP %d pairs/graph, 10 graph replays x 3 timing reps\n", sms, FW, (double)FW * sms / 1e6, NP);
  // what: 0 pair, 1 P only
  auto run = [&](int D_us, int PS_kb, int GS_kb, int pre, int pdl, int what, double* stm) -> double {
    const size_t gsm = (size_t)GS_kb * 1024;
    const bool over = gsm > 227 * 1024;
    if (!over) RK(cudaFuncSetAttribute(kG, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)gsm));
    else RK(cudaFuncSetAttribute(kG, cudaFuncAttributeSharedMemoryMode, (int)cudaSharedMemoryModeAllowOversizedSharedMemory));
    cudaGraph_t g; cudaGraphExec_t ge;
    RK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    for (int i = 0; i < NP; ++i) {
      unsigned long long* sti = st + i * 8;
      kP<<<sms, 256, PS_kb * 1024, s>>>(D_us * 1000, act, sti);
      if (what == 0) {
        cudaLaunchConfig_t cfg = {}; cfg.gridDim = sms; cfg.blockDim = 128; cfg.dynamicSmemBytes = gsm; cfg.stream = s;
        cudaLaunchAttribute at[2]; at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[0].val.programmaticStreamSerializationAllowed = pdl;
        at[1].id = cudaLaunchAttributeSharedMemoryMode; at[1].val.sharedMemoryMode = cudaSharedMemoryModeAllowOversizedSharedMemory;
        cfg.attrs = at; cfg.numAttrs = over ? 2 : 1;
        RK(cudaLaunchKernelEx(&cfg, kG, (const unsigned char*)(w + (size_t)i * stride), FW, pre, pdl, (const float*)act, out, sti));
      }
    }
    RK(cudaStreamEndCapture(s, &g)); RK(cudaGraphInstantiate(&ge, g, 0));
    for (int r = 0; r < 2; ++r) RK(cudaGraphLaunch(ge, s));
    std::vector<double> times;
    for (int k = 0; k < 3; ++k) {
      RK(cudaEventRecord(e0, s)); for (int r = 0; r < 10; ++r) RK(cudaGraphLaunch(ge, s)); RK(cudaEventRecord(e1, s));
      RK(cudaStreamSynchronize(s)); float ms; cudaEventElapsedTime(&ms, e0, e1); times.push_back(ms * 1000.0 / 10 / NP);
    }
    if (what == 0 && stm) {
      std::vector<unsigned long long> init(NP * 8, 0); for (int i = 0; i < NP; ++i) init[i * 8 + 1] = ~0ull;
      std::vector<double> ready, lead, wret, elast;
      for (int k = 0; k < 5; ++k) {
        RK(cudaMemcpyAsync(st, init.data(), NP * 64, cudaMemcpyHostToDevice, s)); RK(cudaGraphLaunch(ge, s)); RK(cudaStreamSynchronize(s));
        std::vector<unsigned long long> h(NP * 8); RK(cudaMemcpy(h.data(), st, NP * 64, cudaMemcpyDeviceToHost));
        for (int i = 4; i < NP; ++i) {
          const long long pe = (long long)h[i * 8];
          ready.push_back(((long long)h[i * 8 + 3] - pe) / 1e3); lead.push_back((pe - (long long)h[i * 8 + 1]) / 1e3);
          wret.push_back(((long long)h[i * 8 + 2] - pe) / 1e3); elast.push_back(((long long)h[i * 8 + 4] - pe) / 1e3);
        }
      }
      stm[0] = pct(ready, .5); stm[1] = pct(ready, .1); stm[2] = pct(ready, .9); stm[3] = pct(lead, .5); stm[4] = pct(wret, .5); stm[5] = pct(elast, .5);
    }
    RK(cudaGraphExecDestroy(ge)); RK(cudaGraphDestroy(g));
    std::sort(times.begin(), times.end()); return times[1];
  };
  printf("D us | P smem KiB | G smem KiB | form | pair us | P alone us | G adds us | front ready after P end us p50 (p10..p90) | G first-entry lead us | last G entry after P end us | wait-return after P end us\n");
  for (int D : {3, 6})
    for (int PS : {16, 32, 64}) {
      double p = run(D, PS, 160, 0, 1, 1, nullptr);
      for (int GS : {255, 224, 192, 160}) {
        for (int pre : {1, 0}) {
          double stm[6]; double pr = run(D, PS, GS, pre, 1, 0, stm);
          printf("%d | %d | %d | PDL %s | %.2f | %.2f | %.2f | %.2f (%.2f..%.2f) | %.2f | %.2f | %.2f\n", D, PS, GS, pre ? "weights before wait" : "weights after wait", pr, p,
                 pr - p, stm[0], stm[1], stm[2], stm[3], stm[5], stm[4]);
          fflush(stdout);
        }
      }
    }
  printf("DONE\n");
  return 0;
}
