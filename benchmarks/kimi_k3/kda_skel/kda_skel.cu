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

#ifndef CHUNK
#define CHUNK 32
#endif
#ifndef NSLOT
#define NSLOT 2
#endif
constexpr int H = 96, D = 128, C = CHUNK, P = H * D;     // P = 12288 channels per tensor
constexpr int TILE = C * D * 2;                        // 8 KiB BF16 tile
constexpr int BETA_BYTES = C * 8 * 2;                  // 32 rows x 8 heads BF16 box (512 B)
constexpr int STAGE = 5 * TILE + 1024;                 // q k v g og + beta, 1 KiB aligned = 41984 B
constexpr int OUT_TILE = C * D;                        // MXFP8 32 x 128 B
constexpr int STATE_HALF = 32768;                      // FP32 state 64 KiB as two halves
#ifndef NCW
#define NCW 4
#endif
constexpr int NCT = NCW * 32;                  // consumer threads per chain slot
constexpr int NWARP_SLOT = 1 + NCW;
#ifndef EARLYREL
#define EARLYREL 0
#endif
#ifndef NGRP
#define NGRP 1
#endif
constexpr int GT = NCT / NGRP;                 // consumer threads per item group (groups ping-pong items)
constexpr int TPT = GT / C, CPT = D / TPT;    // o_norm: threads per token, channels per thread

struct Maps { CUtensorMap qkv, g, og, beta, out, qkv_h, g_h, og_h, beta_h, out_h, qd, kd, vd, gd, ogd, outd; };

__device__ __forceinline__ unsigned sa(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ volatile unsigned* g_diag;
__device__ __forceinline__ void mb_wait(unsigned b, unsigned ph) {
  unsigned ok = 0; long long n = 0;
  while (true) {
    asm volatile("{.reg .pred P; mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2; selp.u32 %0, 1, 0, P;}"
                 : "=r"(ok) : "r"(b), "r"(ph) : "memory");
    if (ok) return;
    if (++n > 400000000LL) { g_diag[0] = 1; g_diag[1] = blockIdx.x; g_diag[2] = threadIdx.x; g_diag[3] = b; __threadfence_system(); printf("WATCHDOG blk %d tid %d\n", blockIdx.x, threadIdx.x); __trap(); }
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
  int nseq, seqlen, nchunk, nchain, mode, S_total, iss6, fma, mufu, onorm, hm, noout, T;
  int dl, Tpad, nd0, nd1;    // die-local: heads [0,48) homed on die 0, [48,96) on die 1; chains die-matched
  const int* sm_die; const int* sm_rank;
  const float* w;          // [H*128] o_norm weight
  float* state;            // [nchain][128*128] f32
  unsigned char* scales;   // [T][H*4]
  unsigned long long* sink;
};

// item -> (chain, k) ; k < 0 : state-in half (-1, -2)
__global__ void __launch_bounds__(NSLOT * NWARP_SLOT * 32, 1) skel_k(const __grid_constant__ Maps M, Args a) {
  extern __shared__ __align__(1024) unsigned char sm[];
  __shared__ __align__(8) unsigned long long full[2][16], empty[2][16];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int slot = warp / NWARP_SLOT, role = warp % NWARP_SLOT;   // role 0 producer, 1-4 consumers
  // which chains / items this slot owns
  int chain = -1, nslots_used = NSLOT;
  unsigned smid_; asm volatile("mov.u32 %0, %%smid;" : "=r"(smid_));
  const int die = a.dl ? a.sm_die[smid_ & 1023] : 0, drank = a.dl ? a.sm_rank[smid_ & 1023] : 0;
  const int nd = die ? a.nd1 : a.nd0, cpd = a.nchain / 2;   // chains per die
  auto dchain = [&](int cl) { return (cl / (H / 2)) * H + die * (H / 2) + cl % (H / 2); };
  if (a.mode == 0 && !a.dl) {
    chain = blockIdx.x + slot * gridDim.x;
    if (blockIdx.x + gridDim.x >= a.nchain) nslots_used = 1;
    if (chain >= a.nchain) chain = -1;
  } else if (a.mode == 0) {
    int cl = drank + slot * nd;
    if (drank + nd >= cpd) nslots_used = 1;
    chain = cl < cpd ? dchain(cl) : -1;
  }
  const int S = a.S_total / nslots_used;
  if (S % NGRP) { if (threadIdx.x == 0 && blockIdx.x == 0) printf("S %d not a multiple of NGRP\n", S); return; }
  unsigned char* ring = sm + (size_t)slot * S * STAGE;
  unsigned char* ob = sm + (size_t)a.S_total * STAGE + (size_t)slot * NGRP * 2 * OUT_TILE;
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
  long long n_items, it0 = 0, it_step = 1, it_end;
  if (a.mode == 0) n_items = 2 + a.nchunk;
  else if (!a.dl) { it0 = blockIdx.x * NSLOT + slot; it_step = NSLOT * gridDim.x; n_items = (long long)a.nchain * a.nchunk; }
  else { it0 = drank * NSLOT + slot; it_step = NSLOT * nd; n_items = (long long)cpd * a.nchunk; }
  it_end = n_items;
  if (a.mode == 2) {   // balanced contiguous pieces: each slot owns a contiguous run of chunk-items (piece-split chains)
    const long long ns = a.dl ? (long long)NSLOT * nd : (long long)NSLOT * gridDim.x, g = a.dl ? drank * NSLOT + slot : blockIdx.x * NSLOT + slot;
    it0 = g * n_items / ns; it_end = (g + 1) * n_items / ns; it_step = 1;
  }
  auto decode = [&](long long j, int& ch, int& k) {
    if (a.mode == 0) { ch = chain; k = (int)j - 2; }
    else if (!a.dl) { ch = (int)(j / a.nchunk); k = (int)(j % a.nchunk); }
    else { ch = dchain((int)(j / a.nchunk)); k = (int)(j % a.nchunk); }
  };
  if (role == 0) {
    long long i = 0;
    for (long long j = it0; j < it_end; j += it_step, ++i) {
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
          if (a.dl) {   // die-local head-major, rows head * Tpad + tok
            const CUtensorMap* mm = t == 0 ? &M.qd : t == 1 ? &M.kd : t == 2 ? &M.vd : t == 3 ? &M.gd : &M.ogd;
            if (t < 5) tma2(d, mm, 0, head * a.Tpad + tok, fb);
            else tma2(d, &M.beta, (head / 8) * 8, tok, fb);
          } else if (a.hm) {   // head-major [tensor][H][T][128]: every tile is one contiguous 8 KiB block
            if (t < 3) tma2(d, &M.qkv_h, 0, (t * H + head) * a.T + tok, fb);
            else if (t == 3) tma2(d, &M.g_h, 0, head * a.T + tok, fb);
            else if (t == 4) tma2(d, &M.og_h, 0, head * a.T + tok, fb);
            else tma2(d, &M.beta, (head / 8) * 8, tok, fb);
          } else if (t < 3) tma2(d, &M.qkv, t * P + head * D, tok, fb);
          else if (t == 3) tma2(d, &M.g, head * D, tok, fb);
          else if (t == 4) tma2(d, &M.og, head * D, tok, fb);
          else tma2(d, &M.beta, (head / 8) * 8, tok, fb);
        }
      }
      (void)who;
      __syncwarp();
    }
  } else {
    const int ctid_all = threadIdx.x - slot * NWARP_SLOT * 32 - 32;   // 0..NCT-1
    const int grp = ctid_all / GT, ctid = ctid_all % GT;
    const int bar_id = 1 + slot * NGRP + grp;
    unsigned acc = 0;
    float wv[CPT]; int whead = -1;
    long long i = 0, nout = 0;
    for (long long j = it0; j < it_end; j += it_step, ++i) {
      if ((int)(i % NGRP) != grp) continue;
      int ch, k; decode(j, ch, k);
      int sl = (int)(i % S);
      unsigned char* st = ring + (size_t)sl * STAGE;
      mb_wait(sa(&full[slot][sl]), (unsigned)((i / S) & 1));
      if (k < 0) {
        acc += st[ctid * (STATE_HALF / GT)];
        named_bar(bar_id, GT);
        if (ctid == 0) asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(sa(&empty[slot][sl])) : "memory");
        continue;
      }
      int seq = ch / H, head = ch % H, tok = seq * a.seqlen + k * C;
      acc += *(const unsigned*)(st + ctid * ((5 * TILE / GT) & ~3));
      if (a.fma | a.mufu) {   // SIMT pipe load of the fused math (conv+SiLU, gate, o_norm): FFMA2 + MUFU ex2
        unsigned long long x[8];
        float f0 = __uint_as_float((acc & 0x007fffffu) | 0x3f000000u);
#pragma unroll
        for (int j = 0; j < 8; ++j) { float2 t = make_float2(f0 + j, f0 - j); x[j] = *(unsigned long long*)&t; }
        float2 tm = make_float2(0.999f, 0.998f), td = make_float2(1e-3f, 2e-3f);
        unsigned long long cm = *(unsigned long long*)&tm, cd = *(unsigned long long*)&td;
        for (int r = 0; r < a.fma / NCT; r += 16) {
#pragma unroll
          for (int j = 0; j < 8; ++j) asm volatile("fma.rn.f32x2 %0, %0, %1, %2;" : "+l"(x[j]) : "l"(cm), "l"(cd));
        }
        float y[4] = {f0 * -1.f, f0 * -0.5f, f0 * -0.25f, f0 * -0.125f};
        for (int r = 0; r < a.mufu / NCT; r += 4) {
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
      unsigned char* o = ob + (size_t)(grp * 2 + (nout & 1)) * OUT_TILE;
      if (ctid == 0) asm volatile("cp.async.bulk.wait_group.read 1;" ::: "memory");
      named_bar(bar_id, GT);
      unsigned char scale_b = (unsigned char)acc;
      bool released = false;
      if (a.onorm) {   // gated RMSNorm + MXFP8 of a 32x128 BF16 O tile (pricing: O = the q tile, gate = the og tile)
        const int tk = ctid / TPT, part = ctid % TPT;
        const uint4* orow = (const uint4*)(st + tk * 256 + part * CPT * 2);
        const uint4* grow = (const uint4*)(st + 4 * TILE + tk * 256 + part * CPT * 2);
        if (head != whead) {
          whead = head;
#pragma unroll
          for (int i = 0; i < CPT; ++i) wv[i] = a.w[head * D + part * CPT + i];
        }
        float y[CPT]; float ss = 0.f;
#pragma unroll
        for (int q4 = 0; q4 < CPT / 8; ++q4) {
          uint4 u = orow[q4]; unsigned uu[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            float lo = __uint_as_float(uu[e] << 16), hi = __uint_as_float(uu[e] & 0xffff0000u);
            y[q4 * 8 + 2 * e] = lo; y[q4 * 8 + 2 * e + 1] = hi; ss = fmaf(lo, lo, ss); ss = fmaf(hi, hi, ss);
          }
        }
        uint4 gw[CPT / 8];
#pragma unroll
        for (int q4 = 0; q4 < CPT / 8; ++q4) gw[q4] = grow[q4];
        if (EARLYREL && !(a.mode == 0 && k == a.nchunk - 1)) {   // stage fully read into registers: release it before the math
          named_bar(bar_id, GT);
          if (ctid == 0) asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(sa(&empty[slot][sl])) : "memory");
          released = true;
        }
#pragma unroll
        for (int m = 1; m < TPT; m <<= 1) ss += __shfl_xor_sync(0xffffffffu, ss, m);
        const float rstd = rsqrtf(ss * (1.f / 128.f) + 1e-6f);
        float amax = 0.f;
#pragma unroll
        for (int q4 = 0; q4 < CPT / 8; ++q4) {
          uint4 u = gw[q4]; unsigned uu[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            float gt = e & 1 ? __uint_as_float(uu[e >> 1] & 0xffff0000u) : __uint_as_float(uu[e >> 1] << 16);
            float ex, sg;
            asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(ex) : "f"(gt * -1.4426950409f));
            asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(sg) : "f"(1.f + ex));
            int i = q4 * 8 + e;
            y[i] = y[i] * rstd * wv[i] * sg;
            amax = fmaxf(amax, fabsf(y[i]));
          }
        }
#pragma unroll
        for (int m = 1; m < 32 / CPT; m <<= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, m));
        int ex8 = (int)((__float_as_uint(amax) >> 23) & 0xff) - 8;   // E8M0 block scale
        ex8 = ex8 < 1 ? 1 : ex8;
        const float inv = __uint_as_float((unsigned)(254 - ex8) << 23);
        unsigned pk[CPT / 4];
#pragma unroll
        for (int i = 0; i < CPT / 4; ++i) {
          unsigned short a01, a23;
          asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(a01) : "f"(y[4 * i + 1] * inv), "f"(y[4 * i] * inv));
          asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(a23) : "f"(y[4 * i + 3] * inv), "f"(y[4 * i + 2] * inv));
          pk[i] = (unsigned)a01 | ((unsigned)a23 << 16);
        }
        uint4* od = (uint4*)(o + tk * D + part * CPT);
#pragma unroll
        for (int i = 0; i < CPT / 16; ++i) od[i] = make_uint4(pk[4 * i], pk[4 * i + 1], pk[4 * i + 2], pk[4 * i + 3]);
        scale_b = (unsigned char)ex8;
        if ((part * CPT) % 32 == 0 && tok + tk < (seq + 1) * a.seqlen)
          a.scales[(size_t)(tok + tk) * (H * 4) + head * 4 + part * CPT / 32] = scale_b;
      } else {
        for (int q = ctid; q < OUT_TILE / 16; q += GT) ((uint4*)o)[q] = make_uint4(acc, q, k, ch);
      }
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      named_bar(bar_id, GT);
      const bool late_rel = a.mode == 0 && k == a.nchunk - 1;   // state-out reads the stage
      if (!late_rel && !released && ctid == 0) asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(sa(&empty[slot][sl])) : "memory");
      if (ctid == 0 && !a.noout) {
        asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
                     :: "l"(a.dl ? &M.outd : a.hm ? &M.out_h : &M.out), "r"(a.hm | a.dl ? 0 : head * D),
                        "r"(a.dl ? head * a.Tpad + tok : a.hm ? head * a.T + tok : tok), "r"(sa(o)) : "memory");
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
      if (late_rel && ctid == 0) asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(sa(&empty[slot][sl])) : "memory");
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

// smid -> die by cold-miss latency: each CTA chases fresh lines in a die-0 and a die-1 buffer
__global__ void die_probe_k(const unsigned* b0, const unsigned* b1, long long* out) {
  if (threadIdx.x) return;
  unsigned s; asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
  long long t[2];
  for (int d = 0; d < 2; ++d) {
    const unsigned* b = (d ? b1 : b0) + (size_t)blockIdx.x * 65536;   // 256 KiB fresh region per CTA
    unsigned idx = 0, acc = 0;
    long long c0 = clock64();
    for (int i = 0; i < 24; ++i) { unsigned v; asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(b + idx + i * 2048)); acc += v; idx = acc & 1; }
    t[d] = clock64() - c0 + (acc == 0xdeadbeef);
  }
  out[blockIdx.x * 3] = s; out[blockIdx.x * 3 + 1] = t[0]; out[blockIdx.x * 3 + 2] = t[1];
}

static void* die_alloc_split(size_t half, int dev) {   // [half on die 0 | half on die 1], half % 2 MiB == 0
  CUmemAllocationProp p = {};
  p.type = CU_MEM_ALLOCATION_TYPE_PINNED; p.location.type = CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN;
  p.location.localized.deviceId = dev;
  size_t g; CK(cuMemGetAllocationGranularity(&g, &p, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
  if (half % g) { printf("half %zu not a multiple of %zu\n", half, g); exit(1); }
  CUdeviceptr va; CK(cuMemAddressReserve(&va, 2 * half, g, 0, 0));
  for (int d = 0; d < 2; ++d) {
    p.location.localized.localityDomainId = d;
    CUmemGenericAllocationHandle h; CK(cuMemCreate(&h, half, &p, 0));
    CK(cuMemMap(va + d * half, half, 0, h, 0));
  }
  CUmemAccessDesc ad = {}; ad.location.type = CU_MEM_LOCATION_TYPE_DEVICE; ad.location.id = dev;
  ad.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE; CK(cuMemSetAccess(va, 2 * half, &ad, 1));
  return (void*)va;
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
  CR(cudaMalloc(&beta, (T + 8) * H * 2)); CR(cudaMalloc(&out, T * P)); CR(cudaMalloc(&scales, T * H * 4));
  CR(cudaMalloc(&state, (size_t)nchain * 65536)); float* wgt; CR(cudaMalloc(&wgt, H * D * 4)); CR(cudaMemset(wgt, 0, H * D * 4)); CR(cudaMalloc(&sink, 64));
  CR(cudaMemset(qkv, 1, T * 3 * P * 2)); CR(cudaMemset(g, 1, T * P * 2)); CR(cudaMemset(og, 1, T * P * 2));
  void* flush; size_t FL = 512ull << 20; CR(cudaMalloc(&flush, FL));
  Maps M;
  enc(&M.qkv, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, qkv, 3 * P, T, 3 * P * 2, D, C);
  enc(&M.g, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, g, P, T, P * 2, D, C);
  enc(&M.og, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, og, P, T, P * 2, D, C);
  enc(&M.beta, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, beta, H, T, H * 2, 8, C);
  enc(&M.out, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, out, P, T, P, D, C);
  enc(&M.qkv_h, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, qkv, D, 3 * H * T, D * 2, D, C);
  enc(&M.g_h, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, g, D, H * T, D * 2, D, C);
  enc(&M.og_h, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, og, D, H * T, D * 2, D, C);
  enc(&M.beta_h, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, beta, T, H, (T + 7) / 8 * 16, C, 1);
  enc(&M.out_h, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, out, D, H * T, D, D, C);
  // die-local head-major tensors: rows per head Tpad (48 x Tpad x 256 B is a multiple of 2 MiB)
  CR(cudaFree(0));
  const long long Tpad = (T + 1023) / 1024 * 1024;
  const size_t half_bf = (size_t)(H / 2) * Tpad * D * 2, half_u8 = (size_t)(H / 2) * Tpad * D;
  void* dq = die_alloc_split(half_bf, 0); void* dk = die_alloc_split(half_bf, 0); void* dv = die_alloc_split(half_bf, 0);
  void* dg = die_alloc_split(half_bf, 0); void* dog = die_alloc_split(half_bf, 0); void* dout = die_alloc_split(half_u8, 0);
  for (void* x : {dq, dk, dv, dg, dog}) CR(cudaMemset(x, 1, 2 * half_bf));
  enc(&M.qd, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, dq, D, H * Tpad, D * 2, D, C);
  enc(&M.kd, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, dk, D, H * Tpad, D * 2, D, C);
  enc(&M.vd, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, dv, D, H * Tpad, D * 2, D, C);
  enc(&M.gd, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, dg, D, H * Tpad, D * 2, D, C);
  enc(&M.ogd, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, dog, D, H * Tpad, D * 2, D, C);
  enc(&M.outd, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, dout, D, H * Tpad, D, D, C);
  // smid -> die map (probe on two fresh die-local buffers)
  size_t pb = (size_t)nsm * 65536 * 4 * 2;   // 2 x ... rounded below
  pb = (pb + (2u << 20) - 1) / (2u << 20) * (2u << 20);
  unsigned* probe = (unsigned*)die_alloc_split(pb, 0);

  long long* pr; CR(cudaMalloc(&pr, nsm * 3 * 8));
  die_probe_k<<<nsm, 32>>>(probe, probe + pb / 4, pr); CR(cudaDeviceSynchronize());
  std::vector<long long> hp(nsm * 3); CR(cudaMemcpy(hp.data(), pr, nsm * 24, cudaMemcpyDeviceToHost));
  std::vector<int> hdie(1024, 0), hrank(1024, 0); int nd[2] = {0, 0}; int amb = 0;
  std::vector<int> order(nsm); for (int i = 0; i < nsm; ++i) order[i] = i;
  std::sort(order.begin(), order.end(), [&](int x, int y) { return hp[x * 3] < hp[y * 3]; });
  for (int i : order) {
    int s = (int)hp[i * 3]; long long t0 = hp[i * 3 + 1], t1 = hp[i * 3 + 2];
    int d = t1 < t0 ? 1 : 0; if (std::min(t0, t1) * 1.15 > std::max(t0, t1)) ++amb;
    hdie[s & 1023] = d; hrank[s & 1023] = nd[d]++;
  }
  int pair_ok = 0; for (int i = 0; i < nsm; ++i) { int s = (int)hp[i * 3]; pair_ok += hdie[s] == ((s >> 1) & 1); }
  printf("die map: %d / %d SMs, ambiguous %d, matches (smid>>1)&1 for %d\n", nd[0], nd[1], amb, pair_ok);
  int *dsm_die, *dsm_rank; CR(cudaMalloc(&dsm_die, 4096)); CR(cudaMalloc(&dsm_rank, 4096));
  CR(cudaMemcpy(dsm_die, hdie.data(), 4096, cudaMemcpyHostToDevice)); CR(cudaMemcpy(dsm_rank, hrank.data(), 4096, cudaMemcpyHostToDevice));
  // SOL bytes (exact tensors, no over-fetch)
  double bytes = (double)T * (5.0 * P * 2 + H * 2 + P + H * 4) + (double)nchain * 65536 * 2;
  printf("SOL bytes %.3f GB -> %.1f us @19.4 TB/s, %.1f us @17.7 TB/s\n", bytes / 1e9, bytes / 19.4e12 * 1e6, bytes / 17.7e12 * 1e6);
  struct V { const char* name; int mode, S_total, iss6, fma, mufu, onorm, hm, noout, dl; };
  // fused-math pipe load per 32-token chunk-item (lane-ops): FMA 135K (conv 49K + SiLU poly part + o_norm 33K + prep),
  // MUFU 41K (SiLU 24.6K + gate/decay 8.2K + o_norm sigmoid 8.2K); KDA pass without conv+SiLU: FMA 70K, MUFU 16K
  std::vector<V> vs = NSLOT == 2 ? std::vector<V>{
      {"DL rrbal S=6", 1, 6, 1, 0, 0, 0, 0, 0, 1}, {"DL piece S=6", 2, 6, 1, 0, 0, 0, 0, 0, 1},
      {"DL piece S=4", 2, 4, 1, 0, 0, 0, 0, 0, 1},
      {"DL piece S=6 +onorm", 2, 6, 1, 0, 0, 1, 0, 0, 1}, {"DL piece S=4 +onorm", 2, 4, 1, 0, 0, 1, 0, 0, 1}}
    : std::vector<V>{
      {"DL piece 1slot S=3", 2, 3, 1, 0, 0, 0, 0, 0, 1}, {"DL piece 1slot S=3 +onorm", 2, 3, 1, 0, 0, 1, 0, 0, 1}};
  unsigned* hdiag; CR(cudaHostAlloc(&hdiag, 64, cudaHostAllocMapped)); for (int i = 0; i < 16; ++i) hdiag[i] = 0;
  { unsigned* dd; CR(cudaHostGetDevicePointer(&dd, hdiag, 0)); CR(cudaMemcpyToSymbol(g_diag, &dd, sizeof(dd))); }
  cudaEvent_t e0, e1; CR(cudaEventCreate(&e0)); CR(cudaEventCreate(&e1));
  printf("attr maxdyn: %s\n", cudaGetErrorString(cudaFuncSetAttribute(skel_k, cudaFuncAttributeMaxDynamicSharedMemorySize, 232448)));
  printf("attr oversized: %s\n", cudaGetErrorString(cudaFuncSetAttribute(skel_k, cudaFuncAttributeSharedMemoryMode, cudaSharedMemoryModeAllowOversizedSharedMemory)));
  cudaGetLastError();
  for (auto& v : vs) {
    Args a{nseq, seqlen, nchunk, nchain, v.mode, v.S_total, v.iss6, v.fma, v.mufu, v.onorm, v.hm, v.noout, (int)T,
           v.dl, (int)Tpad, nd[0], nd[1], dsm_die, dsm_rank, wgt, state, scales, sink};
    size_t smem = (size_t)v.S_total * STAGE + 2 * NSLOT * NGRP * OUT_TILE;
    std::vector<float> ts;
    for (int it = 0; it < iters + 3; ++it) {
      CR(cudaMemsetAsync(flush, it, FL));   // evict L2 between iterations
      CR(cudaEventRecord(e0));
      cudaLaunchConfig_t cfg = {}; cudaLaunchAttribute la[1];
      cfg.gridDim = dim3(nsm); cfg.blockDim = dim3(NSLOT * NWARP_SLOT * 32); cfg.dynamicSmemBytes = smem;
      la[0].id = cudaLaunchAttributeSharedMemoryMode; la[0].val.sharedMemoryMode = cudaSharedMemoryModeAllowOversizedSharedMemory;
      cfg.attrs = la; cfg.numAttrs = 1;
      cudaError_t le = cudaLaunchKernelEx(&cfg, skel_k, M, a);
      if (le != cudaSuccess) { printf("SKEL %-24s launch failed: %s\n", v.name, cudaGetErrorString(le)); cudaGetLastError(); break; }
      CR(cudaEventRecord(e1));
      { cudaError_t se = cudaEventSynchronize(e1);
        if (se != cudaSuccess) { printf("SKEL %-24s FAILED %s diag %u blk %u tid %u bar %u\n", v.name, cudaGetErrorString(se), hdiag[0], hdiag[1], hdiag[2], hdiag[3]); exit(3); } }
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
