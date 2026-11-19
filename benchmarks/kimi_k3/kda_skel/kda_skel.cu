// SPDX-License-Identifier: Apache-2.0
// Data-movement-only skeleton of the single-pass KDA prefill layer (SOL protocol step 2).
//
// One persistent CTA per SM. Per chain slot (one (sequence, head) chain): one producer warp streams 32-token stages of
// raw q|k|v (in_proj rows), the KDA gate g, beta, and the o_norm gate through an S-deep TMA ring; four consumer warps
// release each stage, write a 32x128 MXFP8 output tile + E8M0 scales (TMA store) and, at the chain ends, read and write
// the 64 KiB FP32 state. Same bytes and barriers as the fused kernel, no math.
//
//   kda_skel <nseq> <seqlen> [iters]
// Modes (all run): chain-faithful (2 chain slots on the first NCH-212 SMs) with several stage depths, chunk round-robin
// (perfect balance, isolates the chain imbalance), issuer variants (1 lane vs 6 lanes).
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <algorithm>

#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { printf("CU err %d line %d\n", (int)r_, __LINE__); exit(1); } } while (0)
#define CR(x) do { cudaError_t r_ = (x); if (r_ != cudaSuccess) { printf("CUDA err %s line %d\n", cudaGetErrorString(r_), __LINE__); exit(1); } } while (0)

constexpr int H = 96, D = 128, C = 32, P = H * D;     // P = 12288 channels per tensor
constexpr int TILE = C * D * 2;                        // 8 KiB BF16 tile
constexpr int BETA_BYTES = C * 8 * 2;                  // 32 rows x 8 heads BF16 box (512 B)
constexpr int STAGE = 5 * TILE + 1024;                 // q k v g og + beta, 1 KiB aligned = 41984 B
constexpr int OUT_TILE = C * D;                        // MXFP8 32 x 128 B
constexpr int STATE_HALF = 32768;                      // FP32 state 64 KiB as two halves
constexpr int NWARP_SLOT = 5;

struct Maps { CUtensorMap qkv, g, og, beta, out; };

__device__ __forceinline__ unsigned sa(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mb_wait(unsigned b, unsigned ph) {
  unsigned ok = 0; long long n = 0;
  while (true) {
    asm volatile("{.reg .pred P; mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2; selp.u32 %0, 1, 0, P;}"
                 : "=r"(ok) : "r"(b), "r"(ph) : "memory");
    if (ok) return;
    if (++n > 400000000LL) { printf("WATCHDOG blk %d tid %d\n", blockIdx.x, threadIdx.x); __trap(); }
  }
}
__device__ __forceinline__ void tma2(unsigned dst, const CUtensorMap* m, int c0, int c1, unsigned bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
               :: "r"(dst), "l"(m), "r"(c0), "r"(c1), "r"(bar) : "memory");
}
__device__ __forceinline__ void bulk_ld(unsigned dst, const void* src, int n, unsigned bar) {
  asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(n), "r"(bar) : "memory");
}
__device__ __forceinline__ void bulk_st(void* dst, unsigned src, int n) {
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" :: "l"(dst), "r"(src), "r"(n) : "memory");
}
__device__ __forceinline__ void named_bar(int id, int n) { asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory"); }

struct Args {
  int nseq, seqlen, nchunk, nchain, mode, S_total, iss6, fma, mufu, onorm;
  const float* w;          // [H*128] o_norm weight
  float* state;            // [nchain][128*128] f32
  unsigned char* scales;   // [T][H*4]
  unsigned long long* sink;
};

// item -> (chain, k) ; k < 0 : state-in half (-1, -2)
__global__ void __launch_bounds__(2 * NWARP_SLOT * 32, 1) skel_k(const __grid_constant__ Maps M, Args a) {
  extern __shared__ __align__(1024) unsigned char sm[];
  __shared__ __align__(8) unsigned long long full[2][16], empty[2][16];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int slot = warp / NWARP_SLOT, role = warp % NWARP_SLOT;   // role 0 producer, 1-4 consumers
  // which chains / items this slot owns
  int chain = -1, nslots_used = 2;
  if (a.mode == 0) {
    chain = blockIdx.x + slot * gridDim.x;
    if (blockIdx.x + gridDim.x >= a.nchain) nslots_used = 1;
    if (chain >= a.nchain) chain = -1;
  }
  const int S = a.S_total / nslots_used;
  unsigned char* ring = sm + (size_t)slot * S * STAGE;
  unsigned char* ob = sm + (size_t)a.S_total * STAGE + (size_t)slot * 2 * OUT_TILE;
  if (threadIdx.x == 0) {
    for (int s = 0; s < 2; ++s)
      for (int i = 0; i < 16; ++i) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(sa(&full[s][i])));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(sa(&empty[s][i])));
      }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (a.mode == 0 && chain < 0) return;
  // item enumeration
  long long n_items, it0 = 0, it_step = 1;
  if (a.mode == 0) n_items = 2 + a.nchunk;
  else { it0 = blockIdx.x * 2 + slot; it_step = 2 * gridDim.x; n_items = (long long)a.nchain * a.nchunk; }
  auto decode = [&](long long j, int& ch, int& k) {
    if (a.mode == 0) { ch = chain; k = (int)j - 2; }
    else { ch = (int)(j / a.nchunk); k = (int)(j % a.nchunk); }
  };
  if (role == 0) {
    long long i = 0;
    for (long long j = it0; j < n_items; j += it_step, ++i) {
      int ch, k; decode(j, ch, k);
      int sl = (int)(i % S);
      unsigned fb = sa(&full[slot][sl]), eb = sa(&empty[slot][sl]);
      if (i >= S) mb_wait(eb, (unsigned)(((i / S) - 1) & 1));
      unsigned dst = sa(ring + (size_t)sl * STAGE);
      if (k < 0) {
        if (lane == 0) {
          asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(fb), "r"(STATE_HALF) : "memory");
          bulk_ld(dst, (const unsigned char*)a.state + (size_t)ch * 65536 + (size_t)(k + 2) * STATE_HALF, STATE_HALF, fb);
        }
        __syncwarp();
        continue;
      }
      int seq = ch / H, head = ch % H, tok = seq * a.seqlen + k * C;
      if (lane == 0)
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(fb), "r"(5 * TILE + BETA_BYTES) : "memory");
      __syncwarp();
      int who = a.iss6 ? lane : 0;
      if (a.iss6 ? lane < 6 : lane == 0) {
        for (int t = (a.iss6 ? lane : 0); t < (a.iss6 ? lane + 1 : 6); ++t) {
          unsigned d = dst + t * TILE;
          if (t < 3) tma2(d, &M.qkv, t * P + head * D, tok, fb);
          else if (t == 3) tma2(d, &M.g, head * D, tok, fb);
          else if (t == 4) tma2(d, &M.og, head * D, tok, fb);
          else tma2(d, &M.beta, (head / 8) * 8, tok, fb);
        }
      }
      (void)who;
      __syncwarp();
    }
  } else {
    const int ctid = threadIdx.x - slot * NWARP_SLOT * 32 - 32;   // 0..127
    const int bar_id = 1 + slot;
    unsigned acc = 0;
    long long i = 0, nout = 0;
    for (long long j = it0; j < n_items; j += it_step, ++i) {
      int ch, k; decode(j, ch, k);
      int sl = (int)(i % S);
      unsigned char* st = ring + (size_t)sl * STAGE;
      mb_wait(sa(&full[slot][sl]), (unsigned)((i / S) & 1));
      if (k < 0) {
        acc += st[ctid * 256];
        named_bar(bar_id, 128);
        if (ctid == 0) asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(sa(&empty[slot][sl])) : "memory");
        continue;
      }
      int seq = ch / H, head = ch % H, tok = seq * a.seqlen + k * C;
      acc += *(const unsigned*)(st + ctid * 320);
      if (a.fma | a.mufu) {   // SIMT pipe load of the fused math (conv+SiLU, gate, o_norm): FFMA2 + MUFU ex2
        unsigned long long x[8];
        float f0 = __uint_as_float((acc & 0x007fffffu) | 0x3f000000u);
#pragma unroll
        for (int j = 0; j < 8; ++j) { float2 t = make_float2(f0 + j, f0 - j); x[j] = *(unsigned long long*)&t; }
        float2 tm = make_float2(0.999f, 0.998f), td = make_float2(1e-3f, 2e-3f);
        unsigned long long cm = *(unsigned long long*)&tm, cd = *(unsigned long long*)&td;
        for (int r = 0; r < a.fma; r += 16) {
#pragma unroll
          for (int j = 0; j < 8; ++j) asm volatile("fma.rn.f32x2 %0, %0, %1, %2;" : "+l"(x[j]) : "l"(cm), "l"(cd));
        }
        float y[4] = {f0 * -1.f, f0 * -0.5f, f0 * -0.25f, f0 * -0.125f};
        for (int r = 0; r < a.mufu; r += 4) {
#pragma unroll
          for (int j = 0; j < 4; ++j) asm volatile("ex2.approx.ftz.f32 %0, %0;" : "+f"(y[j]));
        }
        unsigned sx = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) sx ^= (unsigned)x[j] ^ (unsigned)(x[j] >> 32);
#pragma unroll
        for (int j = 0; j < 4; ++j) sx ^= __float_as_uint(y[j]);
        acc += sx;
      }
      unsigned char* o = ob + (nout & 1) * OUT_TILE;
      if (ctid == 0) asm volatile("cp.async.bulk.wait_group.read 1;" ::: "memory");
      named_bar(bar_id, 128);
      unsigned char scale_b = (unsigned char)acc;
      if (a.onorm) {   // gated RMSNorm + MXFP8 of a 32x128 BF16 O tile (pricing: O = the q tile, gate = the og tile)
        const int tk = ctid >> 2, part = ctid & 3;
        const uint4* orow = (const uint4*)(st + tk * 256 + part * 64);
        const uint4* grow = (const uint4*)(st + 4 * TILE + tk * 256 + part * 64);
        const float* wr = a.w + head * D + part * 32;
        float y[32]; float ss = 0.f;
#pragma unroll
        for (int q4 = 0; q4 < 4; ++q4) {
          uint4 u = orow[q4]; unsigned uu[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            float lo = __uint_as_float(uu[e] << 16), hi = __uint_as_float(uu[e] & 0xffff0000u);
            y[q4 * 8 + 2 * e] = lo; y[q4 * 8 + 2 * e + 1] = hi; ss = fmaf(lo, lo, ss); ss = fmaf(hi, hi, ss);
          }
        }
        ss += __shfl_xor_sync(0xffffffffu, ss, 1); ss += __shfl_xor_sync(0xffffffffu, ss, 2);
        const float rstd = rsqrtf(ss * (1.f / 128.f) + 1e-6f);
        float amax = 0.f;
#pragma unroll
        for (int q4 = 0; q4 < 4; ++q4) {
          uint4 u = grow[q4]; unsigned uu[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            float gt = e & 1 ? __uint_as_float(uu[e >> 1] & 0xffff0000u) : __uint_as_float(uu[e >> 1] << 16);
            float ex, sg;
            asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(ex) : "f"(gt * -1.4426950409f));
            asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(sg) : "f"(1.f + ex));
            int i = q4 * 8 + e;
            y[i] = y[i] * rstd * __ldg(wr + i) * sg;
            amax = fmaxf(amax, fabsf(y[i]));
          }
        }
        int ex8 = (int)((__float_as_uint(amax) >> 23) & 0xff) - 8;   // E8M0 block scale (amax / 2^8 <= 448)
        ex8 = ex8 < 1 ? 1 : ex8;
        const float inv = __uint_as_float((unsigned)(254 - ex8) << 23);
        unsigned pk[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          unsigned short a01, a23;
          asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(a01) : "f"(y[4 * i + 1] * inv), "f"(y[4 * i] * inv));
          asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(a23) : "f"(y[4 * i + 3] * inv), "f"(y[4 * i + 2] * inv));
          pk[i] = (unsigned)a01 | ((unsigned)a23 << 16);
        }
        uint4* od = (uint4*)(o + tk * D + part * 32);
        od[0] = make_uint4(pk[0], pk[1], pk[2], pk[3]); od[1] = make_uint4(pk[4], pk[5], pk[6], pk[7]);
        scale_b = (unsigned char)ex8;
        if (tok + tk < (seq + 1) * a.seqlen) a.scales[(size_t)(tok + tk) * (H * 4) + head * 4 + part] = scale_b;
      } else {
        ((uint4*)o)[ctid * 2] = make_uint4(acc, ctid, k, ch);
        ((uint4*)o)[ctid * 2 + 1] = make_uint4(ch, k, ctid, acc);
      }
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      named_bar(bar_id, 128);
      if (ctid == 0) {
        asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
                     :: "l"(&M.out), "r"(head * D), "r"(tok), "r"(sa(o)) : "memory");
        asm volatile("cp.async.bulk.commit_group;" ::: "memory");
      }
      if (!a.onorm && ctid < C && tok + ctid < (seq + 1) * a.seqlen)
        *(unsigned*)(a.scales + (size_t)(tok + ctid) * (H * 4) + head * 4) = acc ^ ctid;
      ++nout;
      if (a.mode == 0 && k == a.nchunk - 1 && ctid == 0) {   // state out, read from this stage before releasing it
        bulk_st((unsigned char*)a.state + (size_t)ch * 65536, sa(st), STATE_HALF);
        bulk_st((unsigned char*)a.state + (size_t)ch * 65536 + STATE_HALF, sa(st), STATE_HALF);
        asm volatile("cp.async.bulk.commit_group;" ::: "memory");
        asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
      }
      if (ctid == 0) asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(sa(&empty[slot][sl])) : "memory");
    }
    if (ctid == 0) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
    if (acc == 0x9e3779b9u) a.sink[0] = acc;
  }
}

static void enc(CUtensorMap* m, CUtensorMapDataType dt, int esz, void* p, uint64_t cols, uint64_t rows, uint64_t rowbytes,
                uint32_t bc, uint32_t br) {
  cuuint64_t dims[2] = {cols, rows}, strides[1] = {rowbytes};
  cuuint32_t box[2] = {bc, br}, es[2] = {1, 1};
  (void)esz;
  CK(cuTensorMapEncodeTiled(m, dt, 2, p, dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                            CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

int main(int argc, char** argv) {
  int nseq = argc > 1 ? atoi(argv[1]) : 3, seqlen = argc > 2 ? atoi(argv[2]) : 5461, iters = argc > 3 ? atoi(argv[3]) : 20;
  const long long T = (long long)nseq * seqlen;
  const int nchunk = (seqlen + C - 1) / C, nchain = nseq * H;
  int nsm; CR(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
  int clk; CR(cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0));
  printf("T=%lld nseq=%d seqlen=%d chunks/chain=%d chains=%d SMs=%d\n", T, nseq, seqlen, nchunk, nchain, nsm);
  void *qkv, *g, *og, *beta, *out; float* state; unsigned char* scales; unsigned long long* sink;
  CR(cudaMalloc(&qkv, T * 3 * P * 2)); CR(cudaMalloc(&g, T * P * 2)); CR(cudaMalloc(&og, T * P * 2));
  CR(cudaMalloc(&beta, T * H * 2)); CR(cudaMalloc(&out, T * P)); CR(cudaMalloc(&scales, T * H * 4));
  CR(cudaMalloc(&state, (size_t)nchain * 65536)); float* wgt; CR(cudaMalloc(&wgt, H * D * 4)); CR(cudaMemset(wgt, 0, H * D * 4)); CR(cudaMalloc(&sink, 64));
  CR(cudaMemset(qkv, 1, T * 3 * P * 2)); CR(cudaMemset(g, 1, T * P * 2)); CR(cudaMemset(og, 1, T * P * 2));
  void* flush; size_t FL = 512ull << 20; CR(cudaMalloc(&flush, FL));
  Maps M;
  enc(&M.qkv, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, qkv, 3 * P, T, 3 * P * 2, D, C);
  enc(&M.g, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, g, P, T, P * 2, D, C);
  enc(&M.og, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, og, P, T, P * 2, D, C);
  enc(&M.beta, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, beta, H, T, H * 2, 8, C);
  enc(&M.out, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, out, P, T, P, D, C);
  // SOL bytes (exact tensors, no over-fetch)
  double bytes = (double)T * (5.0 * P * 2 + H * 2 + P + H * 4) + (double)nchain * 65536 * 2;
  printf("SOL bytes %.3f GB -> %.1f us @19.4 TB/s, %.1f us @17.7 TB/s\n", bytes / 1e9, bytes / 19.4e12 * 1e6, bytes / 17.7e12 * 1e6);
  struct V { const char* name; int mode, S_total, iss6, fma, mufu, onorm; };
  std::vector<V> vs = {
      {"chain  S=4 (2x2)  iss1", 0, 4, 0}, {"chain  S=4 (2x2)  iss6", 0, 4, 1}, {"chain  S=6 (2x3)  iss6", 0, 6, 1},
      {"chain  S=7 (2x3)  iss1", 0, 7, 0}, {"chain  S=7 (2x3)  iss6", 0, 7, 1}, {"rrbal  S=4 (2x2)  iss6", 1, 4, 1},
      {"rrbal  S=6 (2x3)  iss6", 1, 6, 1}, {"rrbal  S=7 (2x3)  iss6", 1, 7, 1},
      {"chain  S=7 iss6 +FMA1056", 0, 7, 1, 1056, 0}, {"chain  S=7 iss6 +MUFU320", 0, 7, 1, 0, 320},
      {"chain  S=7 iss6 +both", 0, 7, 1, 1056, 320}, {"rrbal  S=7 iss6 +both", 1, 7, 1, 1056, 320},
      {"chain  S=7 iss6 +onorm", 0, 7, 1, 0, 0, 1}, {"chain  S=4 iss6 +onorm", 0, 4, 1, 0, 0, 1}};
  cudaEvent_t e0, e1; CR(cudaEventCreate(&e0)); CR(cudaEventCreate(&e1));
  printf("attr maxdyn: %s\n", cudaGetErrorString(cudaFuncSetAttribute(skel_k, cudaFuncAttributeMaxDynamicSharedMemorySize, 232448)));
  printf("attr oversized: %s\n", cudaGetErrorString(cudaFuncSetAttribute(skel_k, cudaFuncAttributeSharedMemoryMode, cudaSharedMemoryModeAllowOversizedSharedMemory)));
  cudaGetLastError();
  for (auto& v : vs) {
    Args a{nseq, seqlen, nchunk, nchain, v.mode, v.S_total, v.iss6, v.fma, v.mufu, v.onorm, wgt, state, scales, sink};
    size_t smem = (size_t)v.S_total * STAGE + 4 * OUT_TILE;
    std::vector<float> ts;
    for (int it = 0; it < iters + 3; ++it) {
      CR(cudaMemsetAsync(flush, it, FL));   // evict L2 between iterations
      CR(cudaEventRecord(e0));
      cudaLaunchConfig_t cfg = {}; cudaLaunchAttribute la[1];
      cfg.gridDim = dim3(nsm); cfg.blockDim = dim3(2 * NWARP_SLOT * 32); cfg.dynamicSmemBytes = smem;
      la[0].id = cudaLaunchAttributeSharedMemoryMode; la[0].val.sharedMemoryMode = cudaSharedMemoryModeAllowOversizedSharedMemory;
      cfg.attrs = la; cfg.numAttrs = 1;
      cudaError_t le = cudaLaunchKernelEx(&cfg, skel_k, M, a);
      if (le != cudaSuccess) { printf("SKEL %-24s launch failed: %s\n", v.name, cudaGetErrorString(le)); cudaGetLastError(); break; }
      CR(cudaEventRecord(e1));
      CR(cudaEventSynchronize(e1));
      CR(cudaGetLastError());
      float ms; CR(cudaEventElapsedTime(&ms, e0, e1));
      if (it >= 3) ts.push_back(ms * 1000.f);
    }
    if (ts.empty()) continue;
    std::sort(ts.begin(), ts.end());
    float med = ts[ts.size() / 2];
    printf("SKEL %-24s smem %6zu B  med %7.1f us  min %7.1f  -> %5.2f TB/s  = %5.1f%% of SOL@19.4  %5.1f%% of 130us-target\n",
           v.name, smem, med, ts[0], bytes / (med * 1e-6) / 1e12, bytes / 19.4e12 * 1e6 / med * 100, 130.0 / med * 100);
  }
  printf("DONE\n");
  return 0;
}
