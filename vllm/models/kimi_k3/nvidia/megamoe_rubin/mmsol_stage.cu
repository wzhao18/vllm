// SPDX-License-Identifier: Apache-2.0
// EXPERIMENT ONLY (hwdb candidate microbench; extends hwdb bench/mma2cta.cu): what sets the per-stage floor of a
// 2-CTA K256 block-scaled NVFP4 MMA stage loop on VR200? One elected MMA thread per pair (leader) runs S stages; each
// stage = [wait FULL] + [NCP SF tcgen05.cp 32x128b.warpx4] + MMAs (pieces x 2 x K128) + [commit -> EMPTY (multicast)].
// A producer thread per CTA models the ring without data: waits EMPTY, arrives on the leader's FULL (remote for the peer).
// Operands are static smem garbage (no TMA): the floor of the issue / barrier / tcgen05 pipe alone.
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_107a,code=sm_107a mmsol_stage.cu -o stage ; run: ./stage
#include "mmsol_core.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
using namespace mm;

struct Arm {
  const char* name;
  int cg2;         // 1 = cta_group::2 M256, 0 = cta_group::1 M128 (single CTA)
  int np; int n[2];  // pieces and their N
  int ncp;         // SF cps per stage
  int ring;        // 1 = producer/consumer FULL/EMPTY ring, 0 = no waits
  int cevery;      // commit to EMPTY every cevery stages (0 = only at the end)
  int depth;       // ring depth
  int peer;        // ring: the peer CTA's producer also arrives on the leader FULL (2 arrivals per stage)
  int kind;        // 0 = mxf4nvf4 K128 (x2 per K256), 1 = MXFP8 K64 (x2 per K128 stage)
  int cpform;      // 0 = converged warp + elect.sync 32x128b.warpx4 (production), 1 = diverged thread 0, 2 = elect 128x256b
};

template <int CG2>
__global__ void __launch_bounds__(128, 1) stage_k(Arm a, int S, unsigned long long* out) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  __shared__ __align__(8) uint64_t full[16], empty[16], done;
  __shared__ uint32_t tmem_base;
  uint8_t* sm = (uint8_t*)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);
  const uint32_t cr = CG2 ? cta_rank() : 0;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  if (threadIdx.x == 0) {
    for (int i = 0; i < 16; ++i) { bar_init(sa(&full[i]), a.peer ? 2 : 1); bar_init(sa(&empty[i]), 1); }
    bar_init(sa(&done), 1); bar_fence_init();
  }
  if (warp == 1) {
    if (CG2) tmem_alloc576(sa(&tmem_base));
    else {
      asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;" :: "r"(sa(&tmem_base)) : "memory");
      asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    }
  }
  tc_fence_before();
  if (CG2) cluster_sync(); else __syncthreads();
  tc_fence_after();
  const uint32_t tb = tmem_base;
  const int D = a.depth;
  if (warp == 0 && lane == 0 && a.ring && (cr == 0 || a.peer)) {  // producer: wait EMPTY, arrive FULL (leader's)
    for (int s = 0; s < S; ++s) {
      const int slot = s % D; const int r = s / D;
      if (r > 0) bar_wait(sa(&empty[slot]), (r - 1) & 1);
      if (CG2) bar_arrive_cluster(mapa(sa(&full[slot]), 0)); else bar_arrive(sa(&full[slot]));
      if (!a.peer && CG2 && cr == 1) {}  // (peer arrives only when a.peer; the leader's count is 1 otherwise)
    }
  }
  if (warp == 1 && cr == 0) {
    uint64_t c0 = clk64(), t0 = now_ns();
    const uint64_t da = desc_sw128(sa(sm)), db = desc_sw128(sa(sm + 32768));
    uint32_t id[2];
    for (int p = 0; p < 2; ++p)
      id[p] = a.kind ? (CG2 ? idesc_mx8(a.n[p]) : ((1u << 23) | ((uint32_t)(a.n[p] >> 3) << 17) | ((128u >> 4) << 24) | (1u << 31)))
                     : (CG2 ? idesc_nvf4(a.n[p]) : ((1u << 3) | (1u << 7) | (1u << 10) | ((uint32_t)(a.n[p] >> 3) << 17) | ((128u >> 4) << 24)));
    const uint32_t col[2] = {0, 256};
    for (int s = 0; s < S; ++s) {
      const int slot = s % D;
      if (a.ring) bar_wait(sa(&full[slot]), (s / D) & 1);
      tc_fence_after();
      for (int j = 0; j < a.ncp; ++j) {
        const uint32_t ta = tb + 512 + 4 * (j % 16);
        const uint64_t sd = desc_sf(sa(sm + 65536 + 512 * j));
        if (CG2) {
          if (a.cpform == 0) sf_cp(ta, sd);
          else if (a.cpform == 1) { if (lane == 0) asm volatile("tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;" :: "r"(ta), "l"(sd) : "memory"); __syncwarp(); }
          else asm volatile("{.reg .pred P; elect.sync _|P, 0xffffffff; @P tcgen05.cp.cta_group::2.128x256b [%0], %1;}" :: "r"(tb + 512 + 8 * (j % 8)), "l"(sd) : "memory");
        }
        else asm volatile("{.reg .pred P; elect.sync _|P, 0xffffffff; @P tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;}" :: "r"(ta), "l"(sd) : "memory");
      }
      for (int p = 0; p < a.np; ++p)
        for (int i = 0; i < 2; ++i) {
          const uint32_t acc = (s > 0 || i > 0);
          if (CG2) {
            if (a.kind) mma_mx8(tb + col[p], da + 4 * i, db + 4 * i, id[p], tb + 512, tb + 528, acc);
            else mma_nvf4(tb + col[p], da + 4 * i, db + 4 * i, id[p], tb + 512, tb + 528, acc);
          } else {
            if (a.kind)
              asm volatile("{.reg .pred P, Q; elect.sync _|P, 0xffffffff; setp.ne.b32 Q, %6, 0;\n@P tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%4], [%5], Q;}"
                           :: "r"(tb + col[p]), "l"(da + 4 * i), "l"(db + 4 * i), "r"(id[p]), "r"(tb + 512), "r"(tb + 528), "r"(acc) : "memory");
            else
              asm volatile("{.reg .pred P, Q; elect.sync _|P, 0xffffffff; setp.ne.b32 Q, %6, 0;\n@P tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%4], [%5], Q;}"
                           :: "r"(tb + col[p]), "l"(da + 4 * i), "l"(db + 4 * i), "r"(id[p]), "r"(tb + 512), "r"(tb + 528), "r"(acc) : "memory");
          }
        }
      if (a.cevery && ((s + 1) % a.cevery == 0)) {
        if (CG2) commit_mc(sa(&empty[slot]), 3);
        else asm volatile("{.reg .pred P; elect.sync _|P, 0xffffffff; @P tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];}" :: "r"(sa(&empty[slot])) : "memory");
      }
    }
    if (CG2) commit_mc(sa(&done), 3);
    else asm volatile("{.reg .pred P; elect.sync _|P, 0xffffffff; @P tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];}" :: "r"(sa(&done)) : "memory");
    if (lane == 0) {
      bar_wait(sa(&done), 0);
      out[blockIdx.x * 2] = clk64() - c0; out[blockIdx.x * 2 + 1] = now_ns() - t0;
    }
  }
  // peer producer when a.peer: arrives on the leader FULL too (inside the producer loop above for cr == 1 as well)
  if (CG2 && cr == 1 && warp == 1 && lane == 0) bar_wait(sa(&done), 0);
  tc_fence_before();
  __syncthreads();
  if (CG2) cluster_sync();
  if (warp == 1) {
    tc_fence_after();
    if (CG2) tmem_dealloc576(tb);
    else asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" :: "r"(tb) : "memory");
  }
}


// Tight, compile-time-specialized issue loop (hwdb mma2cta / hw28c form): no runtime branches between tcgen05 ops.
// NP pieces x 2 K128 MMAs per stage (N0 then N1), NCP SF cps per stage, COMMIT = 1: commit (multicast) per stage.
template <int N0, int N1, int NP, int NCP, int COMMIT>
__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(128, 1) tight_k(int S, unsigned long long* out) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  __shared__ __align__(8) uint64_t empty, done;
  __shared__ uint32_t tmem_base;
  uint8_t* sm = (uint8_t*)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);
  const uint32_t cr = cta_rank();
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  if (threadIdx.x == 0) { bar_init(sa(&empty), 1); bar_init(sa(&done), 1); bar_fence_init(); }
  if (warp == 1) tmem_alloc576(sa(&tmem_base));
  tc_fence_before(); cluster_sync(); tc_fence_after();
  const uint32_t tb = tmem_base;
  if (warp == 1 && cr == 0) {
    const uint64_t c0 = clk64(), t0 = now_ns();
    const uint64_t da = desc_sw128(sa(sm)), db = desc_sw128(sa(sm + 32768)), ds = desc_sf(sa(sm + 65536));
    constexpr uint32_t id0 = (1u << 3) | (1u << 7) | (1u << 10) | ((uint32_t)(N0 >> 3) << 17) | ((256u >> 4) << 24);
    constexpr uint32_t id1 = (1u << 3) | (1u << 7) | (1u << 10) | ((uint32_t)(N1 >> 3) << 17) | ((256u >> 4) << 24);
    #pragma unroll 4
    for (int s = 0; s < S; ++s) {
      #pragma unroll
      for (int j = 0; j < NCP; ++j) sf_cp(tb + 512 + 4 * (j % 16), ds + 32 * j);
      #pragma unroll
      for (int i = 0; i < 2; ++i) {
        mma_nvf4(tb, da + 4 * i, db + 4 * i, id0, tb + 512, tb + 528, 1u);
        if (NP > 1) mma_nvf4(tb + 256, da + 4 * i, db + 4 * i, id1, tb + 512, tb + 528, 1u);
      }
      if (COMMIT) commit_mc(sa(&empty), 3);
    }
    commit_mc(sa(&done), 3);
    if (lane == 0) { bar_wait(sa(&done), 0); out[blockIdx.x * 2] = clk64() - c0; out[blockIdx.x * 2 + 1] = now_ns() - t0; }
  }
  if (cr == 1 && warp == 1 && lane == 0) bar_wait(sa(&done), 0);
  tc_fence_before(); __syncthreads(); cluster_sync();
  if (warp == 1) { tc_fence_after(); tmem_dealloc576(tb); }
}
template <int N0, int N1, int NP, int NCP, int COMMIT>
static void run_tight(const char* name, int sms, int S, unsigned long long* out) {
  auto k = tight_k<N0, N1, NP, NCP, COMMIT>;
  cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);
  float bc = 1e30f, bn = 1e30f;
  for (int rep = 0; rep < 4; ++rep) {
    cudaMemset(out, 0, 16 * sms);
    k<<<sms / 2 * 2, 128, 96 * 1024>>>(S, out);
    if (cudaDeviceSynchronize() != cudaSuccess) { printf("%-52s ERROR\n", name); exit(1); }
    std::vector<unsigned long long> h(2 * sms); cudaMemcpy(h.data(), out, 16 * sms, cudaMemcpyDeviceToHost);
    std::vector<double> c, n;
    for (int b = 0; b < sms; ++b) if (h[2 * b]) { c.push_back((double)h[2 * b] / S); n.push_back((double)h[2 * b + 1] / S); }
    std::sort(c.begin(), c.end()); std::sort(n.begin(), n.end());
    if (rep) { bc = std::min(bc, (float)c[c.size() / 2]); bn = std::min(bn, (float)n[n.size() / 2]); }
  }
  printf("%-52s %7.1f clk/stage %7.1f ns/stage (%.2f GHz)\n", name, bc, bn, bc / bn); fflush(stdout);
}

int main(int argc, char** argv) {
  int sms; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  const int S = argc > 1 ? atoi(argv[1]) : 2048;
  unsigned long long* out; cudaMalloc(&out, 8 * 2 * sms);
  const int smem = 96 * 1024;
  cudaFuncSetAttribute(stage_k<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaFuncSetAttribute(stage_k<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  // name, cg2, np, n, ncp, ring, cevery, depth, peer, kind
  std::vector<Arm> arms = {
    {"A0 issue-loop only: ring+peer, no MMA no cp", 1, 0, {208, 96}, 0, 1, 1, 8, 1, 0},
    {"A1 2CTA skel {208,96} MMA only, commit at end", 1, 2, {208, 96}, 0, 0, 0, 8, 0, 0},
    {"A2 + commit every stage (no waits)", 1, 2, {208, 96}, 0, 0, 1, 8, 0, 0},
    {"A3 + ring wait full / commit empty, depth 8", 1, 2, {208, 96}, 0, 1, 1, 8, 0, 0},
    {"A4 + ring with peer arrivals (2 per stage)", 1, 2, {208, 96}, 0, 1, 1, 8, 1, 0},
    {"A5 ring depth 3", 1, 2, {208, 96}, 0, 1, 1, 3, 1, 0},
    {"A6 commit every 4 stages, ring depth 8", 1, 2, {208, 96}, 0, 1, 4, 8, 1, 0},
    {"A7 A4 + 4 SF cps", 1, 2, {208, 96}, 4, 1, 1, 8, 1, 0},
    {"A8 A4 + 12 SF cps (production count)", 1, 2, {208, 96}, 12, 1, 1, 8, 1, 0},
    {"A9 A4 + 16 SF cps (skel count)", 1, 2, {208, 96}, 16, 1, 1, 8, 1, 0},
    {"A10 16 cps only, no MMA", 1, 0, {208, 96}, 16, 1, 1, 8, 1, 0},
    {"A11 16 cps only, no ring", 1, 0, {208, 96}, 16, 0, 0, 8, 0, 0},
    {"A12 16 cps only, no ring, DIVERGED thread-0 issue", 1, 0, {208, 96}, 16, 0, 0, 8, 0, 0, 1},
    {"A13 4 cps 128x256b (4 KiB each = same 16 KiB of TMEM SF writes as 16 x warpx4? no: 4x data) only, no ring", 1, 0, {208, 96}, 4, 0, 0, 8, 0, 0, 2},
    {"A14 A4 + 4 cps 128x256b", 1, 2, {208, 96}, 4, 1, 1, 8, 1, 0, 2},
    {"B1 2CTA production {256} MMA only", 1, 1, {256, 0}, 0, 0, 0, 8, 0, 0},
    {"B2 2CTA production {256} ring+peer", 1, 1, {256, 0}, 0, 1, 1, 8, 1, 0},
    {"B3 2CTA production {256} ring+peer + 12 cps", 1, 1, {256, 0}, 12, 1, 1, 8, 1, 0},
    {"B4 2CTA {256,256} MMA only", 1, 2, {256, 256}, 0, 0, 0, 8, 0, 0},
    {"B5 2CTA {128} MMA only", 1, 1, {128, 0}, 0, 0, 0, 8, 0, 0},
    {"B6 2CTA {96} MMA only", 1, 1, {96, 0}, 0, 0, 0, 8, 0, 0},
    {"B7 2CTA {208} MMA only", 1, 1, {208, 0}, 0, 0, 0, 8, 0, 0},
    {"C1 1CTA M128 {208,96} MMA only", 0, 2, {208, 96}, 0, 0, 0, 8, 0, 0},
    {"C2 1CTA M128 {256} MMA only", 0, 1, {256, 0}, 0, 0, 0, 8, 0, 0},
    {"D1 2CTA MXFP8 K64 {256,256} MMA only (SE gate_up)", 1, 2, {256, 256}, 0, 0, 0, 8, 0, 1},
    {"D2 2CTA MXFP8 K64 {256} MMA only (SE down)", 1, 1, {256, 0}, 0, 0, 0, 8, 0, 1},
  };
  const int only = argc > 2 ? atoi(argv[2]) : -1;
  if (only == 100) {
    run_tight<256, 256, 1, 0, 0>("T1 tight {256} 2 MMA/stage, no commit", sms, S, out);
    run_tight<96, 96, 1, 0, 0>("T2 tight {96}", sms, S, out);
    run_tight<208, 96, 2, 0, 0>("T3 tight {208,96} 4 MMA/stage", sms, S, out);
    run_tight<256, 256, 2, 0, 0>("T4 tight {256,256}", sms, S, out);
    run_tight<256, 256, 1, 0, 1>("T5 tight {256} + commit/stage", sms, S, out);
    run_tight<208, 96, 2, 0, 1>("T6 tight {208,96} + commit/stage", sms, S, out);
    run_tight<256, 256, 1, 12, 1>("T7 tight {256} + 12 cps + commit (production)", sms, S, out);
    run_tight<208, 96, 2, 16, 1>("T8 tight {208,96} + 16 cps + commit (skel)", sms, S, out);
    run_tight<256, 256, 0, 16, 0>("T9 tight 16 cps only", sms, S, out);
    run_tight<256, 256, 0, 4, 0>("T10 tight 4 cps only", sms, S, out);
    return 0;
  }
  if (only >= (int)arms.size()) return 0;
  if (only <= 0) printf("== mmsol_stage: per-stage floor, %d stages, grid %d (all SMs, 1 CTA/SM)\n", S, sms / 2 * 2);
  int ai = -1;
  for (auto& a : arms) {
    if (++ai, only >= 0 && ai != only) continue;
    float best_c = 1e30f, best_n = 1e30f;
    for (int rep = 0; rep < 3; ++rep) {
      cudaMemset(out, 0, 8 * 2 * sms);
      if (a.cg2) {
        cudaLaunchConfig_t cfg = {}; cfg.gridDim = sms / 2 * 2; cfg.blockDim = 128; cfg.dynamicSmemBytes = smem;
        cudaLaunchAttribute at[1]; at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = 2;
        at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1; cfg.attrs = at; cfg.numAttrs = 1;
        cudaLaunchKernelEx(&cfg, stage_k<1>, a, S, out);
      } else stage_k<0><<<sms, 128, smem>>>(a, S, out);
      cudaError_t e = cudaDeviceSynchronize();
      if (e != cudaSuccess) { printf("%-52s ERROR %s\n", a.name, cudaGetErrorString(e)); return 1; }
      std::vector<unsigned long long> h(2 * sms); cudaMemcpy(h.data(), out, 16 * sms, cudaMemcpyDeviceToHost);
      std::vector<double> c, n;
      for (int b = 0; b < sms; ++b) if (h[2 * b]) { c.push_back((double)h[2 * b] / S); n.push_back((double)h[2 * b + 1] / S); }
      std::sort(c.begin(), c.end()); std::sort(n.begin(), n.end());
      if (rep) { best_c = std::min(best_c, (float)c[c.size() / 2]); best_n = std::min(best_n, (float)n[n.size() / 2]); }
    }
    printf("%-52s %7.1f clk/stage %7.1f ns/stage (%.2f GHz)\n", a.name, best_c, best_n, best_c / best_n);
    fflush(stdout);
  }
  if (only < 0) printf("DONE\n");
  return 0;
}
