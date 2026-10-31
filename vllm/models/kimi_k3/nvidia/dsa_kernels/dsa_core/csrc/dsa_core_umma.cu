// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Kimi K3.1 DSA sparse core under DCP, tcgen05 (UMMA) variant: CTA (latent half d,
// row t), 256 threads, all gathered heads of the row as the MMA's M = 128 (96 heads
// + zero rows), this rank's selected KV rows in chunks of 64 keys:
//
//   QK  tcgen05.mma kind::f8f6f4 (E4M3 x E4M3, FP32 in TMEM), M 128 x N 64 x K 576
//       (18 K-steps of 32), Q and K both K-major from shared memory;
//   softmax  warps 0-3, one head row per thread (TMEM lane), online in base 2 with P
//       scaled by 2^8 (P = exp2(s - m + 8) <= 256) and stored as E4M3 (as the stock
//       FP8 MLA decode kernels), the running O rescaled in TMEM when a row's max grows;
//   PV  tcgen05.mma kind::f8f6f4, M 128 x N 256 (this CTA's latent half) x K 64
//       (2 K-steps of 32), P K-major, V (the same staged KV rows) MN-major.
//
// Shared layouts are the canonical no-swizzle core-matrix layouts (8 rows x 16 B
// blocks): one KV chunk is staged once and read as K (K-major, N = keys) by QK and
// as V (MN-major, N = latent dims) by PV.
//
// The query tile is polled out of the gathered q mailbox before the PDL wait (as the
// mma.sync core, dsa_core.cu); the normalized partial O (bf16, no negative zero) and
// the base-2 LSE go to the head owners' DcpBox for DcpValue(pushed=True).

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include <algorithm>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

namespace k31dsau {

constexpr int kThreads = 256;
constexpr uint32_t kSent = 0xFFFFFFFFu;
constexpr int kMaxK = 2048;
constexpr int kM = 128;        // MMA rows (gathered heads, zero padded)
constexpr int kChunk = 64;     // keys per chunk (QK N, PV K)
constexpr int kRowB = 576;     // FP8 Q / KV row bytes
constexpr int kKc = kRowB / 16;  // 36 16-byte chunks per row
constexpr int kGrp = kKc * 128;  // bytes per 8-row group (36 core matrices)
constexpr int kHalf = 256;     // latent dims per CTA (PV N)
constexpr float kEmptyLse = -3.0e38f;
constexpr int kTmemCols = 512;  // O [0, 256) | S [256, 320)
constexpr int kOCol = 0, kSCol = 256;

// shared memory plan (bytes)
constexpr int kQ = 0;                              // [16 groups][36][128] Q, K-major
constexpr int kKV = kQ + (kM / 8) * kGrp;          // 2 x [8 groups][36][128] KV rows
constexpr int kP = kKV + 2 * (kChunk / 8) * kGrp;  // [16 groups][4][128] P, K-major
constexpr int kIdx = kP + (kM / 8) * 4 * 128;      // [kMaxK] slots
constexpr int kRed = kIdx + kMaxK * 4;             // [2][128] f32 row max / sum halves
constexpr int kBar = kRed + 2 * kM * 4;            // mbarriers (MMA done) + TMEM address
constexpr int kSmem = kBar + 32;
constexpr int kOStr = 264;  // bf16 O staging stride (halves)
static_assert(kM * kOStr * 2 <= 2 * (kChunk / 8) * kGrp, "O staging over the KV buffers");
// QK in the 2x FP8 mode (K = 64 per instruction; idesc bit 29, 1-CTA M = 128)
#ifndef K31_DSAU_QK1X
constexpr bool QK2X = true;
#else
constexpr bool QK2X = false;
#endif
static_assert(kSmem <= 225 * 1024, "dsa_core_umma shared memory");

struct Params {
  const uint8_t* q;
  int64_t q_row_stride;
  const int* sel;
  int64_t sel_stride;
  int width;
  const int* cnt;
  const uint8_t* kv;
  float s2scale, vscale;
  const uint64_t* part_peers;
  const uint64_t* lse_peers;
  int box_cap, hpr, parity, rank, W, T, heads;
  int* err;
  long long spin_ns;
  long long* dbg;  // optional [T * 2, 24]
};

// ------------------------------------------------------------------ primitives
__device__ __forceinline__ long long now_ns() {
  long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __noinline__ void fail(int* err, int code) {
  atomicCAS(err, 0, code);
  __threadfence_system();
  __trap();
}
__device__ __forceinline__ uint4 ld_sys_v4(const void* p) {
  uint4 v;
  asm volatile("ld.relaxed.sys.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p)
               : "memory");
  return v;
}
__device__ __forceinline__ void st_v4(uint32_t* p, uint4 v) {
  asm volatile("st.relaxed.sys.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(p), "r"(v.x),
               "r"(v.y), "r"(v.z), "r"(v.w)
               : "memory");
}
__device__ __forceinline__ void st_u32(uint32_t* p, uint32_t v) {
  asm volatile("st.relaxed.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ int ld_cg(const int* p) {
  int v;
  asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}
__device__ __forceinline__ void cp_async16(uint32_t saddr, const void* gmem) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(saddr), "l"(gmem) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
}
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_launch() {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void fence_async_smem() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
__device__ __forceinline__ void mbar_init(uint32_t bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count));
}
__device__ __forceinline__ bool mbar_try(uint32_t bar, uint32_t parity) {
  uint32_t ok;
  asm volatile(
      "{\n\t.reg .pred P;\n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2;\n\t"
      "selp.u32 %0, 1, 0, P;\n\t}"
      : "=r"(ok)
      : "r"(bar), "r"(parity)
      : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity, const Params& p,
                                          int code) {
  if (mbar_try(bar, parity)) return;
  const long long ts = now_ns();
  while (!mbar_try(bar, parity)) {
    if (now_ns() - ts > p.spin_ns) fail(p.err, code);
  }
}
__device__ __forceinline__ void tc_fence_before() {
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}
__device__ __forceinline__ void tc_fence_after() {
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}
// no-swizzle canonical shared memory descriptor (SM100 version 1)
__device__ __forceinline__ uint64_t sdesc(uint32_t addr, uint32_t lbo, uint32_t sbo) {
  uint64_t d = 0;
  d |= (uint64_t)((addr >> 4) & 0x3FFF);
  d |= (uint64_t)((lbo >> 4) & 0x3FFF) << 16;
  d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
  d |= (uint64_t)1 << 46;  // version
  return d;                // base offset 0, lbo mode 0, layout SWIZZLE_NONE (0)
}
// kind::f8f6f4 instruction descriptor: E4M3 x E4M3 -> F32, M x N
__host__ __device__ constexpr uint32_t idesc(int m, int n, bool b_mn_major) {
  return (1u << 4)                         // D format F32
         | (0u << 7) | (0u << 10)          // A, B E4M3
         | (0u << 15)                      // A K-major
         | ((b_mn_major ? 1u : 0u) << 16)  // B major
         | ((uint32_t)(n >> 3) << 17) | ((uint32_t)(m >> 4) << 24);
}
__device__ __forceinline__ void umma(uint32_t d_tmem, uint64_t a, uint64_t b, uint32_t id,
                                     uint32_t acc) {
  asm volatile(
      "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t}" ::"r"(d_tmem),
      "l"(a), "l"(b), "r"(id), "r"(acc)
      : "memory");
}
__device__ __forceinline__ void umma_commit(uint32_t bar) {
  asm volatile(
      "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(bar)
      : "memory");
}
// 32 lanes x 32 bit, 32 consecutive columns per thread
__device__ __forceinline__ void tmem_ld32(uint32_t taddr, float* v) {
  uint32_t r[32];
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,"
      "%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]),
        "=r"(r[7]), "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]),
        "=r"(r[14]), "=r"(r[15]), "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]),
        "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]), "=r"(r[25]),
        "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
      : "r"(taddr));
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
  for (int i = 0; i < 32; ++i) v[i] = __uint_as_float(r[i]);
}
__device__ __forceinline__ void tmem_st32(uint32_t taddr, const float* v) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,"
      "%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32};" ::"r"(taddr),
      "r"(__float_as_uint(v[0])), "r"(__float_as_uint(v[1])), "r"(__float_as_uint(v[2])),
      "r"(__float_as_uint(v[3])), "r"(__float_as_uint(v[4])), "r"(__float_as_uint(v[5])),
      "r"(__float_as_uint(v[6])), "r"(__float_as_uint(v[7])), "r"(__float_as_uint(v[8])),
      "r"(__float_as_uint(v[9])), "r"(__float_as_uint(v[10])), "r"(__float_as_uint(v[11])),
      "r"(__float_as_uint(v[12])), "r"(__float_as_uint(v[13])), "r"(__float_as_uint(v[14])),
      "r"(__float_as_uint(v[15])), "r"(__float_as_uint(v[16])), "r"(__float_as_uint(v[17])),
      "r"(__float_as_uint(v[18])), "r"(__float_as_uint(v[19])), "r"(__float_as_uint(v[20])),
      "r"(__float_as_uint(v[21])), "r"(__float_as_uint(v[22])), "r"(__float_as_uint(v[23])),
      "r"(__float_as_uint(v[24])), "r"(__float_as_uint(v[25])), "r"(__float_as_uint(v[26])),
      "r"(__float_as_uint(v[27])), "r"(__float_as_uint(v[28])), "r"(__float_as_uint(v[29])),
      "r"(__float_as_uint(v[30])), "r"(__float_as_uint(v[31]))
      : "memory");
  asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ float ex2(float x) {  // 2^x (approx, -inf -> 0)
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ bool armed(uint4 v) {
  return v.x == kSent || v.y == kSent || v.z == kSent || v.w == kSent;
}
__device__ __forceinline__ uint32_t no_neg_zero(uint32_t w) {
  if ((w & 0xFFFFu) == 0x8000u) w &= 0xFFFF0000u;
  if ((w >> 16) == 0x8000u) w &= 0xFFFFu;
  return w;
}
__device__ __forceinline__ uint32_t bf2(float a, float b) {
  __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
  return no_neg_zero(*reinterpret_cast<uint32_t*>(&v));
}
// offset of (row r, 16 B chunk c) in a K-major no-swizzle layout with K extent kc chunks
__device__ __forceinline__ int cm_off(int r, int c, int kc) {
  return (r >> 3) * (kc * 128) + c * 128 + (r & 7) * 16;
}

// ------------------------------------------------------------------ kernel
__global__ void __launch_bounds__(kThreads, 1) dsau_kernel(const __grid_constant__ Params p) {
  extern __shared__ __align__(1024) uint8_t smem[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int half = blockIdx.x, t = blockIdx.y;
  uint8_t* sQ = smem + kQ;
  uint8_t* sKV = smem + kKV;
  uint8_t* sP = smem + kP;
  int* sIdx = reinterpret_cast<int*>(smem + kIdx);
  const uint32_t barS = smem_u32(smem + kBar), barO = barS + 8;
  uint32_t* sTmem = reinterpret_cast<uint32_t*>(smem + kBar + 16);
  __shared__ int s_n;
  long long* dbg = p.dbg ? p.dbg + ((int64_t)t * 2 + half) * 24 : nullptr;
  if (dbg && tid == 0) dbg[0] = now_ns();

  // ---- TMEM (warp 4), mbarriers
  if (warp == 4) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(
                     smem_u32(sTmem)),
                 "n"(kTmemCols)
                 : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
  }
  if (dbg && tid == 128) dbg[6] = now_ns();  // (TMEM allocated)
  if (tid == 0) {
    mbar_init(barS, 1);
    mbar_init(barO, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  // ---- query tile [128, 576] FP8 (rows >= heads zero) from the q mailbox (polled)
  {
    const uint8_t* qrow = p.q + (int64_t)t * p.q_row_stride;
    const int nq = p.heads * kKc;
    for (int i0 = tid; i0 < kM * kKc; i0 += 8 * kThreads) {
      uint4 v[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int i = i0 + j * kThreads;
        v[j] = i < nq ? ld_sys_v4(qrow + 16 * i) : make_uint4(0, 0, 0, 0);
      }
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int i = i0 + j * kThreads;
        if (i < kM * kKc) {
          if (i < nq && armed(v[j])) {
            const long long ts = now_ns();
            do {
              __nanosleep(32);
              if (now_ns() - ts > p.spin_ns) fail(p.err, 1);
              v[j] = ld_sys_v4(qrow + 16 * i);
            } while (armed(v[j]));
          }
          *reinterpret_cast<uint4*>(sQ + cm_off(i / kKc, i % kKc, kKc)) = v[j];
        }
      }
    }
  }
  if (dbg && tid == 0) dbg[1] = now_ns();
  pdl_wait();
  pdl_launch();
  if (dbg && tid == 0) dbg[5] = now_ns();
  // ---- count + slots (one round trip)
  const int* selrow = p.sel + (int64_t)t * p.sel_stride;
  {
    int v[kMaxK / kThreads];
#pragma unroll
    for (int b = 0; b < kMaxK / kThreads; ++b) {
      const int k = b * kThreads + tid;
      v[b] = k < p.width ? ld_cg(selrow + k) : -1;
    }
    if (tid == 0) s_n = ld_cg(p.cnt + t);
    __syncthreads();
    const int n0 = min(s_n, p.width);
#pragma unroll
    for (int b = 0; b < kMaxK / kThreads; ++b) {
      const int k = b * kThreads + tid;
      sIdx[k] = k < n0 ? v[b] : -1;
    }
  }
  fence_async_smem();  // (the Q tile, for the MMA)
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *sTmem;
  const int n = min(s_n, p.width);
  const int nchunks = (n + kChunk - 1) / kChunk;
  if (dbg && tid == 0) dbg[2] = now_ns();

  auto issue = [&](int c) {  // chunk c -> KV buffer c & 1, K-major core matrices
    uint8_t* buf = sKV + (c & 1) * (kChunk / 8) * kGrp;
    const uint32_t b0 = smem_u32(buf);
    for (int i = tid; i < kChunk * kKc; i += kThreads) {
      const int r = i / kKc, pc = i % kKc;
      const int slot = sIdx[c * kChunk + r];
      const int off = cm_off(r, pc, kKc);
      if (slot >= 0) {
        cp_async16(b0 + off, p.kv + (int64_t)slot * kRowB + 16 * pc);
      } else {
        *reinterpret_cast<uint4*>(buf + off) = make_uint4(0, 0, 0, 0);
      }
    }
    cp_commit();
  };
  // softmax: warp w owns TMEM lanes 32 (w & 3) .. (head rows) and S columns
  // 32 (w >> 2) .., and O columns 128 (w >> 2) .. of the CTA's half; each pair of warps
  // (w, w + 4) combines its row max / sum through shared memory
  const int row = 32 * (warp & 3) + lane, ch = warp >> 2;
  float m_run = -INFINITY, l_run = 0.f;
  const uint32_t lane_base = (uint32_t)((warp & 3) * 32) << 16;
  float* sRed = reinterpret_cast<float*>(smem + kRed);  // [2][128] max, then sum
  constexpr uint32_t kIdQK = idesc(kM, kChunk, false) | (QK2X ? (1u << 29) : 0u);
  constexpr uint32_t kIdPV = idesc(kM, kHalf, true);
  constexpr int kQkStep = QK2X ? 512 : 256;  // bytes of K per QK instruction
  // Chunks 0 and 1 load up front; chunk c + 1 (c >= 1) loads into the buffer of chunk
  // c - 1 once QK(c)'s completion is seen (its commit also covers PV(c - 1)).
  if (nchunks > 0) issue(0);
  if (nchunks > 1) issue(1);
  for (int c = 0; c < nchunks; ++c) {
    if (c == 0 && nchunks > 1) {
      cp_wait<1>();
    } else {
      cp_wait<0>();
    }
    fence_async_smem();
    __syncthreads();
    if (dbg && tid == 0 && c < 2) dbg[10 + 4 * c] = clock64();
    const uint32_t buf = smem_u32(sKV + (c & 1) * (kChunk / 8) * kGrp);
    // ---- QK: S[128, 64] = Q K^T
    if (tid == 0) {
      tc_fence_after();
      const uint32_t qa = smem_u32(sQ);
#pragma unroll
      for (int ks = 0; ks < kRowB / (kQkStep / 8); ++ks) {
        const uint64_t a = sdesc(qa + ks * kQkStep, 128, kGrp);
        const uint64_t b = sdesc(buf + ks * kQkStep, 128, kGrp);
        umma(tmem + kSCol, a, b, kIdQK, ks > 0 ? 1u : 0u);
      }
      umma_commit(barS);
    }
    mbar_wait(barS, c & 1, p, 2);  // QK(c) and PV(c - 1) complete
    tc_fence_after();
    if (dbg && tid == 0 && c < 2) dbg[11 + 4 * c] = clock64();
    if (c >= 1 && c + 1 < nchunks) issue(c + 1);  // (buffer of chunk c - 1)
    const int valid = min(kChunk, n - c * kChunk);
    float sv[32];
    tmem_ld32(tmem + lane_base + kSCol + 32 * ch, sv);
    float mx4[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
#pragma unroll
    for (int j = 0; j < 32; ++j) {
      if (32 * ch + j >= valid) sv[j] = -INFINITY;
      mx4[j & 3] = fmaxf(mx4[j & 3], sv[j]);
    }
    sRed[ch * 128 + row] = fmaxf(fmaxf(mx4[0], mx4[1]), fmaxf(mx4[2], mx4[3]));
    __syncthreads();
    const float mx = fmaxf(sRed[row], sRed[128 + row]) * p.s2scale;  // (s2scale > 0)
    const float m_new = fmaxf(m_run, mx);  // finite: valid >= 1
    const float alpha = m_run == -INFINITY ? 0.f : ex2(m_run - m_new);
    const float bias = 8.f - m_new;  // P = 2^(s * scale - m + 8) <= 256
    float sum4[4] = {0.f, 0.f, 0.f, 0.f};
    uint32_t pw[8];
#pragma unroll
    for (int j = 0; j < 32; j += 4) {
      const float p0 = ex2(fmaf(sv[j], p.s2scale, bias)), p1 = ex2(fmaf(sv[j + 1], p.s2scale, bias));
      const float p2 = ex2(fmaf(sv[j + 2], p.s2scale, bias)), p3 = ex2(fmaf(sv[j + 3], p.s2scale, bias));
      sum4[(j >> 2) & 3] += (p0 + p1) + (p2 + p3);
      const __nv_fp8x4_e4m3 q4(make_float4(p0, p1, p2, p3));
      pw[j / 4] = *reinterpret_cast<const uint32_t*>(&q4);
    }
    // P row, keys 32 ch .. 32 ch + 31 -> K-major core matrices 2 ch, 2 ch + 1
    *reinterpret_cast<uint4*>(sP + cm_off(row, 2 * ch, 4)) = make_uint4(pw[0], pw[1], pw[2], pw[3]);
    *reinterpret_cast<uint4*>(sP + cm_off(row, 2 * ch + 1, 4)) = make_uint4(pw[4], pw[5], pw[6], pw[7]);
    // rescale the running O of the warp's rows / columns (PV(c - 1) is done); tcgen05.ld
    // and st are warp-collective, so the whole warp takes the branch
    if (c > 0 && __any_sync(0xffffffffu, alpha < 1.f)) {
#pragma unroll 1
      for (int k = 0; k < kHalf / 64; ++k) {
        float o[32];
        const uint32_t col = tmem + lane_base + kOCol + 128 * ch + 32 * k;
        tmem_ld32(col, o);
#pragma unroll
        for (int j = 0; j < 32; ++j) o[j] *= alpha;
        tmem_st32(col, o);
      }
    }
    const float part = (sum4[0] + sum4[1]) + (sum4[2] + sum4[3]);
    __syncthreads();  // (both halves read the max before the sums overwrite it)
    sRed[ch * 128 + row] = part;
    fence_async_smem();
    tc_fence_before();
    __syncthreads();
    l_run = l_run * alpha + (sRed[row] + sRed[128 + row]);
    m_run = m_new;
    if (dbg && tid == 0 && c < 2) dbg[12 + 4 * c] = clock64();
    // ---- PV: O[128, 256] += P[128, 64] V[64, 256] (2 K-steps of 32 keys)
    if (tid == 0) {
      tc_fence_after();
      const uint32_t pa = smem_u32(sP);
#pragma unroll
      for (int ks = 0; ks < 2; ++ks) {
        const uint64_t a = sdesc(pa + ks * 256, 128, 4 * 128);
        // V, MN-major no swizzle ((T,1,m),(8,k)):((1,T,SBO),(1T,LBO)): SBO = 128 between
        // 16 B latent chunks, LBO = kGrp between 8-key groups
        const uint64_t b = sdesc(buf + half * (kHalf / 16) * 128 + ks * 4 * kGrp, kGrp, 128);
        umma(tmem + kOCol, a, b, kIdPV, (c > 0 || ks > 0) ? 1u : 0u);
      }
      if (c + 1 == nchunks) umma_commit(barO);
    }
    __syncthreads();  // (sRed is rewritten by the next chunk)
  }
  if (nchunks > 0) {
    mbar_wait(barO, 0, p, 3);
    tc_fence_after();
  }
  if (dbg && tid == 0) dbg[3] = now_ns();
  // ---- epilogue: O x vscale / l -> bf16, staged [heads][kOStr] over the KV buffers
  // (all copies and MMAs are done), then coalesced 16 B pushes to the owners
  __nv_bfloat16* sO = reinterpret_cast<__nv_bfloat16*>(sKV);
  {
    const bool live = n > 0 && l_run > 0.f;
    const float r = live ? p.vscale / l_run : 0.f;
#pragma unroll 1
    for (int k = 0; k < kHalf / 64; ++k) {
      float o[32];
      const int c0 = 128 * ch + 32 * k;
      if (nchunks > 0) {
        tmem_ld32(tmem + lane_base + kOCol + c0, o);
      } else {
#pragma unroll
        for (int j = 0; j < 32; ++j) o[j] = 0.f;
      }
      if (row < p.heads) {
#pragma unroll
        for (int j = 0; j < 32; j += 8)
          *reinterpret_cast<uint4*>(sO + row * kOStr + c0 + j) =
              make_uint4(bf2(o[j] * r, o[j + 1] * r), bf2(o[j + 2] * r, o[j + 3] * r),
                         bf2(o[j + 4] * r, o[j + 5] * r), bf2(o[j + 6] * r, o[j + 7] * r));
      }
    }
    if (half == 0 && ch == 0 && row < p.heads) {
      const int H = p.hpr, gh = row;
      const int64_t slot = (((int64_t)p.parity * p.W + p.rank) * p.box_cap + t) * H + gh % H;
      float lse = kEmptyLse;
      if (live && m_run != -INFINITY) {  // P carried 2^8
        lse = m_run + log2f(l_run) - 8.f;
        if (!(lse > kEmptyLse) || !(lse < -kEmptyLse)) lse = kEmptyLse;
      }
      st_u32(reinterpret_cast<uint32_t*>(p.lse_peers[gh / H]) + slot, __float_as_uint(lse));
    }
  }
  tc_fence_before();
  __syncthreads();
  {
    const int H = p.hpr;
    for (int i = tid; i < p.heads * (kHalf / 8); i += kThreads) {
      const int gh = i / (kHalf / 8), c8 = i % (kHalf / 8);
      const int64_t slot = (((int64_t)p.parity * p.W + p.rank) * p.box_cap + t) * H + gh % H;
      uint32_t* dst = reinterpret_cast<uint32_t*>(p.part_peers[gh / H]) + slot * 256 +
                      half * (kHalf / 2) + 4 * c8;
      st_v4(dst, *reinterpret_cast<const uint4*>(sO + gh * kOStr + 8 * c8));
    }
  }
  if (warp == 4) {
    tc_fence_after();
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem),
                 "n"(kTmemCols)
                 : "memory");
  }
  if (dbg && tid == 0) dbg[4] = now_ns();
}

// ------------------------------------------------------------------ host
void run(torch::Tensor q, torch::Tensor sel, torch::Tensor cnt, torch::Tensor kv, double s2scale,
         double vscale, torch::Tensor part_peers, torch::Tensor lse_peers, int64_t box_cap,
         int64_t hpr, int64_t parity, int64_t rank, int64_t T, torch::Tensor err, int64_t spin_ns,
         int64_t pdl, c10::optional<torch::Tensor> dbg) {
  if (T == 0) return;
  const int W = (int)part_peers.numel();
  const int heads = (int)(W * hpr);
  TORCH_CHECK(heads <= kM && heads % 8 == 0, "dsa_core_umma: at most 128 gathered heads");
  TORCH_CHECK(q.dim() == 3 && q.size(-1) == kRowB && q.element_size() == 1 && q.size(1) >= heads);
  TORCH_CHECK(T <= q.size(0) && T <= box_cap && T <= sel.size(0) && T <= cnt.numel());
  TORCH_CHECK(sel.dtype() == torch::kInt32 && sel.stride(1) == 1 && cnt.dtype() == torch::kInt32);
  TORCH_CHECK(kv.element_size() == 1 && kv.size(-1) == kRowB);
  const c10::cuda::CUDAGuard guard(q.device());
  Params p{};
  p.q = reinterpret_cast<const uint8_t*>(q.data_ptr());
  p.q_row_stride = q.stride(0);
  p.sel = sel.data_ptr<int>();
  p.sel_stride = sel.stride(0);
  p.width = (int)std::min<int64_t>(sel.size(1), kMaxK);
  p.cnt = cnt.data_ptr<int>();
  p.kv = reinterpret_cast<const uint8_t*>(kv.data_ptr());
  p.s2scale = (float)s2scale;
  p.vscale = (float)vscale;
  p.part_peers = reinterpret_cast<const uint64_t*>(part_peers.data_ptr<int64_t>());
  p.lse_peers = reinterpret_cast<const uint64_t*>(lse_peers.data_ptr<int64_t>());
  p.box_cap = (int)box_cap;
  p.hpr = (int)hpr;
  p.parity = (int)parity;
  p.rank = (int)rank;
  p.W = W;
  p.T = (int)T;
  p.heads = heads;
  p.err = err.data_ptr<int>();
  p.spin_ns = spin_ns;
  p.dbg = dbg.has_value() ? reinterpret_cast<long long*>(dbg->data_ptr<int64_t>()) : nullptr;
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(
        cudaFuncSetAttribute(dsau_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
    attr = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(2, (unsigned)T, 1);
  cfg.blockDim = dim3(kThreads, 1, 1);
  cfg.dynamicSmemBytes = kSmem;
  cfg.stream = c10::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
  cfg.attrs = attrs;
  cfg.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, dsau_kernel, p));
}

}  // namespace k31dsau
