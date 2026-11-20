#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("CUDA %s at %d\n", cudaGetErrorString(e), __LINE__); return 1; } } while (0)

static constexpr uint64_t SW128 = (uint64_t(1024 >> 4) << 32) | (1ull << 46) | (2ull << 61);
static constexpr uint64_t SW64 = (uint64_t(512 >> 4) << 32) | (1ull << 46) | (4ull << 61);
static constexpr uint32_t STAGE = 36864, NSTG = 2, OFF_QS = NSTG * STAGE, OFF_P = OFF_QS + 8192;
static constexpr uint32_t SMEM = OFF_P + 2 * 6144 + 1024;
static constexpr uint32_t C_Q = 0, C_S = 128, C_O = 192;

__device__ __forceinline__ uint32_t sa(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void wait_par(uint32_t bar, uint32_t par) {
  asm volatile("{.reg .pred P; W%=: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1; "
               "@!P bra W%=;}" :: "r"(bar), "r"(par) : "memory");
}
__device__ __forceinline__ void commit(uint32_t bar) {
  asm volatile("{.reg .pred e; elect.sync _|e, 0xffffffff; @e tcgen05.commit.cta_group::1."
               "mbarrier::arrive::one.shared::cluster.b64 [%0];}" :: "r"(bar) : "memory");
}
#define MMA_TS(d, a, b, i, acc) asm volatile("{.reg .pred p, e; elect.sync _|e, 0xffffffff; " \
  "setp.ne.b32 p, %4, 0; @e tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [%1], %2, %3, p;}" \
  :: "r"(d), "r"(a), "l"(b), "r"(i), "r"(acc) : "memory")
#define MMA_SS(d, a, b, i, acc) asm volatile("{.reg .pred p, e; elect.sync _|e, 0xffffffff; " \
  "setp.ne.b32 p, %4, 0; @e tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;}" \
  :: "r"(d), "l"(a), "l"(b), "r"(i), "r"(acc) : "memory")
#define MMA_CB(op, d, a, b, i, acc) asm volatile("{.reg .pred p, e; elect.sync _|e, 0xffffffff; " \
  "setp.ne.b32 p, %4, 0; @e tcgen05.mma.cta_group::1.kind::f8f6f4.collector::b::" op \
  " [%0], %1, %2, %3, p;}" :: "r"(d), "l"(a), "l"(b), "r"(i), "r"(acc) : "memory")

__device__ __forceinline__ void qk(int mode, uint32_t t, uint64_t bd0, uint64_t aq, uint32_t iqk) {
  if (mode == 9) MMA_SS(t + C_S, aq, (bd0 + 4 * 512) ^ (SW128 ^ SW64), iqk, 0);
#pragma unroll
  for (int j = 0; j < 4; ++j)
#pragma unroll
    for (int k2 = 0; k2 < 2; ++k2)
      MMA_TS(t + C_S, t + C_Q + 32 * j + 16 * k2, bd0 + j * 512 + k2 * 4, iqk,
             (mode == 9 || j + k2 > 0) ? 1u : 0u);
  if (mode != 5 && mode != 9) MMA_SS(t + C_S, aq, (bd0 + 4 * 512) ^ (SW128 ^ SW64), iqk, 1);
}


// P-MLA-2CTA round 4 (pmla13): the h8 single-S round trip standalone, all SMs: warp 4 issues
// QK(u) (mode 1: the exact 8 TS + 1 SS RoPE QK; mode 0: no MMA) and commits S-full; warps 0-3
// (one per TMEM lane quarter) wait S-full, load the 64-column S row (tcgen05.ld.red max, or a
// plain load in modes 2/3), fence and arrive S-free; QK(u + 1) waits for S-free. Prints clk per
// round trip (median over SMs). In-kernel (pmla12_stamps): empty skeleton 447, MMA-only 741-750.
extern "C" __global__ void __launch_bounds__(288, 1) rt(int mode, int nsub, long long* out) {
  extern __shared__ __align__(1024) unsigned char dyn[];
  __shared__ uint64_t bars[4];
  __shared__ uint32_t tslot;
  unsigned char* sm = (unsigned char*)(((uintptr_t)dyn + 1023) & ~(uintptr_t)1023);
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  uint32_t* w = (uint32_t*)sm;
  for (uint32_t i = tid; i < (SMEM - 1024) / 4; i += blockDim.x) w[i] = ((i + 1) * 2654435761u) & 0x3f3f3f3fu;
  if (warp == 0)
    asm volatile("tcgen05.alloc.exclusive.cta_group::1.sync.aligned.shared::cta.b32 [%0], 576; "
                 "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" :: "r"(sa(&tslot)) : "memory");
  if (tid == 0) {
    for (int j = 0; j < 2; ++j) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(sa(&bars[j])) : "memory");  // S-full
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 4;" :: "r"(sa(&bars[2 + j])) : "memory");  // S-free
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  const uint32_t t = tslot, b0 = sa(sm), full0 = sa(&bars[0]), free0 = sa(&bars[2]);
  const bool two = mode >= 6;
  const uint32_t iqk = (1u << 4) | (8u << 17) | (8u << 24) | (1u << 29);
  const uint64_t aq = SW64 | ((b0 + OFF_QS) >> 4);
  long long c0 = clock64();
  float acc = 0.f;
  if (warp == 8) {
    for (int i = 0; i < nsub; ++i) {
      const int pi = two ? (i - 1) & 1 : 0, ph = two ? ((i - 1) >> 1) & 1 : (i - 1) & 1;
      if (i >= 1) {
        wait_par(free0 + 8 * pi, ph);
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
      }
      if (mode == 4 || mode == 5) asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      if (mode == 1 || mode == 3 || mode == 5 || mode == 7)
        qk(0, t, SW128 | ((b0 + (i % NSTG) * STAGE) >> 4), aq, iqk);
      commit(full0 + 8 * (two ? i & 1 : 0));
    }
    wait_par(free0 + 8 * (two ? (nsub - 1) & 1 : 0), two ? ((nsub - 1) >> 1) & 1 : (nsub - 1) & 1);
  } else if (warp < 4 || (two && warp < 8)) {
    const int g = warp / 4;
    const uint32_t la = t + ((uint32_t)((warp % 4) * 32) << 16) + C_S;
    for (int i = g; i < nsub; i += two ? 2 : 1) {
      const uint32_t full = full0 + 8 * (two ? i & 1 : 0), freeb = free0 + 8 * (two ? i & 1 : 0);
      wait_par(full, two ? (i >> 1) & 1 : i & 1);
      asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
      float v[65];
      if (mode != 2 && mode != 3) {
        asm volatile("tcgen05.ld.red.sync.aligned.32x32b.x64.max.f32 {"
                     "%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,"
                     "%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,"
                     "%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},"
                     " %64, [%65]; tcgen05.wait::ld.sync.aligned;"
                     : "=f"(v[0]),"=f"(v[1]),"=f"(v[2]),"=f"(v[3]),"=f"(v[4]),"=f"(v[5]),"=f"(v[6]),"=f"(v[7]),
                       "=f"(v[8]),"=f"(v[9]),"=f"(v[10]),"=f"(v[11]),"=f"(v[12]),"=f"(v[13]),"=f"(v[14]),"=f"(v[15]),
                       "=f"(v[16]),"=f"(v[17]),"=f"(v[18]),"=f"(v[19]),"=f"(v[20]),"=f"(v[21]),"=f"(v[22]),"=f"(v[23]),
                       "=f"(v[24]),"=f"(v[25]),"=f"(v[26]),"=f"(v[27]),"=f"(v[28]),"=f"(v[29]),"=f"(v[30]),"=f"(v[31]),
                       "=f"(v[32]),"=f"(v[33]),"=f"(v[34]),"=f"(v[35]),"=f"(v[36]),"=f"(v[37]),"=f"(v[38]),"=f"(v[39]),
                       "=f"(v[40]),"=f"(v[41]),"=f"(v[42]),"=f"(v[43]),"=f"(v[44]),"=f"(v[45]),"=f"(v[46]),"=f"(v[47]),
                       "=f"(v[48]),"=f"(v[49]),"=f"(v[50]),"=f"(v[51]),"=f"(v[52]),"=f"(v[53]),"=f"(v[54]),"=f"(v[55]),
                       "=f"(v[56]),"=f"(v[57]),"=f"(v[58]),"=f"(v[59]),"=f"(v[60]),"=f"(v[61]),"=f"(v[62]),"=f"(v[63]),
                       "=f"(v[64]) : "r"(la) : "memory");
      } else {
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x64.b32 {"
                     "%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,"
                     "%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,"
                     "%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},"
                     " [%64]; tcgen05.wait::ld.sync.aligned;"
                     : "=f"(v[0]),"=f"(v[1]),"=f"(v[2]),"=f"(v[3]),"=f"(v[4]),"=f"(v[5]),"=f"(v[6]),"=f"(v[7]),
                       "=f"(v[8]),"=f"(v[9]),"=f"(v[10]),"=f"(v[11]),"=f"(v[12]),"=f"(v[13]),"=f"(v[14]),"=f"(v[15]),
                       "=f"(v[16]),"=f"(v[17]),"=f"(v[18]),"=f"(v[19]),"=f"(v[20]),"=f"(v[21]),"=f"(v[22]),"=f"(v[23]),
                       "=f"(v[24]),"=f"(v[25]),"=f"(v[26]),"=f"(v[27]),"=f"(v[28]),"=f"(v[29]),"=f"(v[30]),"=f"(v[31]),
                       "=f"(v[32]),"=f"(v[33]),"=f"(v[34]),"=f"(v[35]),"=f"(v[36]),"=f"(v[37]),"=f"(v[38]),"=f"(v[39]),
                       "=f"(v[40]),"=f"(v[41]),"=f"(v[42]),"=f"(v[43]),"=f"(v[44]),"=f"(v[45]),"=f"(v[46]),"=f"(v[47]),
                       "=f"(v[48]),"=f"(v[49]),"=f"(v[50]),"=f"(v[51]),"=f"(v[52]),"=f"(v[53]),"=f"(v[54]),"=f"(v[55]),
                       "=f"(v[56]),"=f"(v[57]),"=f"(v[58]),"=f"(v[59]),"=f"(v[60]),"=f"(v[61]),"=f"(v[62]),"=f"(v[63])
                     : "r"(la) : "memory");
        v[64] = v[0];
      }
      asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
      __syncwarp();
      if (lane == 0) asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(freeb) : "memory");
      acc += v[64] + v[0] + v[63];
    }
  }
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  if (tid == 0) out[blockIdx.x] = clock64() - c0;
  if (acc == 12345.f) out[0] = 0;
  if (warp == 0)
    asm volatile("tcgen05.dealloc.exclusive.cta_group::1.sync.aligned.b32 %0, 576;" :: "r"(t) : "memory");
}

int main(int argc, char** argv) {
  int sms = 0;
  CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
  CK(cudaFuncSetAttribute(rt, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM + 1024));
  long long* d;
  CK(cudaMalloc(&d, sms * sizeof(long long)));
  const int nsub = argc > 1 ? atoi(argv[1]) : 4096;
  const char* names[] = {"no MMA, ld.red", "QK 9 MMA, ld.red", "no MMA, plain ld", "QK 9 MMA, plain ld",
                         "no MMA + proxy fence", "QK + proxy fence", "no MMA, 2 WG ping-pong", "QK, 2 WG ping-pong"};
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  for (int rep = 0; rep < 2; ++rep)
    for (int mode : {0, 1, 4, 5, 6, 7}) {
      rt<<<sms, 288, SMEM + 1024>>>(mode, 64, d);
      CK(cudaEventRecord(e0));
      rt<<<sms, 288, SMEM + 1024>>>(mode, nsub, d);
      CK(cudaEventRecord(e1));
      CK(cudaDeviceSynchronize());
      float ms = 0;
      CK(cudaEventElapsedTime(&ms, e0, e1));
      std::vector<long long> h(sms);
      CK(cudaMemcpy(h.data(), d, sms * sizeof(long long), cudaMemcpyDeviceToHost));
      std::sort(h.begin(), h.end());
      double cps = (double)h[sms / 2] / nsub, mhz = (double)h[sms / 2] / (ms * 1e3);
      printf("rep %d mode %d %-20s clk/round trip %7.1f (min %7.1f max %7.1f) at %5.0f MHz -> %6.1f us/layer\n",
             rep, mode, names[mode], cps, (double)h[0] / nsub, (double)h[sms - 1] / nsub, mhz, cps * 2473 / mhz);
    }
  return 0;
}
