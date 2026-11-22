// Kimi-K3: FlashKDA K2 (chunk-16 recurrence) on the tcgen05 tensor cores.
//
// Same math as fwd_kernel2.cuh (FlashKDA 17a037d, MIT, see LICENSE.FlashKDA),
// per 16-token chunk c of one (sequence, head), with the [K, V] state S:
//   W   = kd @ bf16(S)              Oq  = qd @ bf16(S)          (fp32 accum)
//   u   = bf16(bf16(v - bf16(W)) * beta)                        (bf16 ops)
//   U   = bf16(INV @ u)                                         (fp32 accum)
//   out = bf16(bf16(Oq) + bf16(Mqk @ U))
//   S   = fma(S, g_total, kr^T @ U)                             (fp32 state)
// The fp32 state crosses the kernel boundary through bf16 as FlashKDA (the
// initial state is read as bf16(S0), the final state written as bf16(S)).
//
// Layout: everything is transposed so V (128) is the MMA M dimension and one
// math thread owns one value row v of S^T for the whole sequence (fp32 in
// registers, 128 values). Per chunk the tensor core gets S^T as a bf16 A
// operand in tensor memory (written row-per-thread by tcgen05.st), the K1
// workspace tiles (kd, qd, INV, Mqk K-major, kr MN-major: K1's 8x8 core-matrix
// byte images) as B operands in shared memory, and accumulates into tensor
// memory, which the same thread reads back row-per-thread: no transposes, no
// shuffles. Per chunk: MMA1 (W | Oq, M128 N16 K128 x2), MMA2 (U, M128 N16 K16),
// MMA3 (Mqk@U M128 N16 K16, kr^T@U M128 N128 K16).
//
// Warps: 0-3 math (thread = value row), 4 loader (bulk copies of the K1
// workspace + v rows, beta), 5 MMA issuer + tensor-memory owner, 6-7 idle.
// Two CTAs per SM (256 tensor-memory columns each).
#pragma once

#include <cstdint>

#include "utils.cuh"
#include "k31_spin_guard.cuh"  // bounded mbarrier waits

namespace k2tc {

constexpr int kChunk = 16;
constexpr int kD = 128;
constexpr int kStages = 4;

// Per-stage shared-memory image (bytes).
// kd | qd as one N = 32 B operand: per 8-wide K group g, kd rows 0-15 at
// g * 512 and qd rows at g * 512 + 256 (K1's 8x8 core-matrix rows).
constexpr int kOffKdq = 0;
constexpr int kOffKr = 8192;
constexpr int kOffV = 12288;     // [16 t][128 v] bf16 rows
constexpr int kOffInv = 16384;   // 16x16 bf16 (K1 image)
constexpr int kOffMqk = 16896;
constexpr int kOffGt = 17408;    // float[128]
constexpr int kOffBeta = 17920;  // bf16[16]: bf16(sigmoid(beta))
constexpr int kStageBytes = 18048;
constexpr int kBarOff = kStages * kStageBytes;
// barriers: full[S], empty[S], d1, d2, d3, d4
constexpr int kNumBars = 2 * kStages + 4;
constexpr int kDbgOff = kBarOff + 8 * kNumBars + 16;
constexpr int kDbgChunk0 = 100, kDbgChunks = 32, kDbgSlots = 8;
constexpr int kSmemBytes = kDbgOff + 8 * kDbgChunks * kDbgSlots;

// Tensor-memory columns (one 256-column allocation per CTA).
constexpr uint32_t kColP = 0;     // kr^T @ U, fp32 [128 v][128 k]
constexpr uint32_t kColW = 128;   // kd @ S | qd @ S over k < 64 (N = 32)
constexpr uint32_t kColW2 = 0;    // the k >= 64 half (P's columns, free then)
constexpr uint32_t kColU = 160;   // INV @ u, then Mqk @ U (after U is read)
constexpr uint32_t kColAu = 176;  // A operand u^T, bf16 [v][16 t]
constexpr uint32_t kColAU = 184;  // A operand U^T
constexpr uint32_t kColAS = 192;  // A operand S^T, bf16 [v][128 k]

__device__ __forceinline__ uint32_t smem_u32(void const* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count)
               : "memory");
}

// bounded (k31_spin_guard.cuh): site = the wait's line
#define mbar_wait(bar, parity) k31sg::mbar_wait((bar), (parity), "flashkda_k2_tc", __LINE__)

__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::"r"(bar)
               : "memory");
}

__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile(
      "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" ::"r"(
          bar),
      "r"(bytes)
      : "memory");
}

__device__ __forceinline__ void bulk_g2s(uint32_t dst, void const* src,
                                         uint32_t bytes, uint32_t bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
      "[%0], [%1], %2, [%3];" ::"r"(dst),
      "l"(src), "r"(bytes), "r"(bar)
      : "memory");
}

__device__ __forceinline__ void tc_fence_before() {
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}
__device__ __forceinline__ void tc_fence_after() {
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}
__device__ __forceinline__ void tc_wait_ld() {
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void tc_wait_st() {
  asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tc_commit(uint32_t bar) {
  asm volatile(
      "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 "
      "[%0];" ::"r"(bar)
      : "memory");
}

// D[tmem] (+)= A[tmem] * B[smem desc], kind::f16 (bf16 in, fp32 accum).
__device__ __forceinline__ void mma_ts(uint32_t d, uint32_t a, uint64_t b_desc,
                                       uint32_t idesc, uint32_t accumulate) {
  asm volatile(
      "{\n .reg .pred p;\n setp.ne.b32 p, %4, 0;\n"
      " tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n}" ::"r"(d),
      "r"(a), "l"(b_desc), "r"(idesc), "r"(accumulate)
      : "memory");
}

// Instruction descriptor: bf16 x bf16 -> f32, M = 128, A K-major (tensor
// memory), B K-major (b_mn = 0) or MN-major (b_mn = 1).
__host__ __device__ constexpr uint32_t idesc_bf16(int n, int b_mn) {
  return (1u << 4) | (1u << 7) | (1u << 10) | (uint32_t(b_mn) << 16) |
         (uint32_t(n >> 3) << 17) | (uint32_t(128 >> 4) << 24);
}

// Shared-memory matrix descriptor, no swizzle (8x8 core matrices of 16 B rows).
__device__ __forceinline__ uint64_t sdesc(uint32_t addr, uint32_t lbo,
                                          uint32_t sbo) {
  return uint64_t((addr >> 4) & 0x3FFF) |
         (uint64_t((lbo >> 4) & 0x3FFF) << 16) |
         (uint64_t((sbo >> 4) & 0x3FFF) << 32) | (uint64_t(1) << 46);
}

#define K2TC_LD16(taddr, r)                                                    \
  asm volatile(                                                                \
      "tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9," \
      "%10,%11,%12,%13,%14,%15}, [%16];"                                       \
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]),            \
        "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]), "=r"(r[9]),            \
        "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]),       \
        "=r"(r[15])                                                            \
      : "r"(taddr))

#define K2TC_LD32(taddr, r)                                                    \
  asm volatile(                                                                \
      "tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9," \
      "%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,"   \
      "%27,%28,%29,%30,%31}, [%32];"                                           \
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]),            \
        "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]), "=r"(r[9]),            \
        "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]),       \
        "=r"(r[15]), "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]),       \
        "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]),       \
        "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]),       \
        "=r"(r[30]), "=r"(r[31])                                               \
      : "r"(taddr))

#define K2TC_ST8(taddr, r)                                                     \
  asm volatile(                                                                \
      "tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};" \
      ::"r"(taddr),                                                            \
      "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]),       \
      "r"(r[6]), "r"(r[7])                                                     \
      : "memory")

#define K2TC_ST16(taddr, r)                                                    \
  asm volatile(                                                                \
      "tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8," \
      "%9,%10,%11,%12,%13,%14,%15,%16};" ::"r"(taddr),                         \
      "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]),       \
      "r"(r[6]), "r"(r[7]), "r"(r[8]), "r"(r[9]), "r"(r[10]), "r"(r[11]),     \
      "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15])                          \
      : "memory")

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}
__device__ __forceinline__ float bf16_round(float x) {
  return float(cutlass::bfloat16_t(x));
}
__device__ __forceinline__ float bf16_lo(uint32_t p) {
  return __uint_as_float(p << 16);
}
__device__ __forceinline__ float bf16_hi(uint32_t p) {
  return __uint_as_float(p & 0xFFFF0000u);
}

#define K2TC_LD8(taddr, r)                                                     \
  asm volatile(                                                                \
      "tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];" \
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]),            \
        "=r"(r[5]), "=r"(r[6]), "=r"(r[7])                                     \
      : "r"(taddr))

#define K2TC_ST4(taddr, r)                                                     \
  asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};" ::  \
                   "r"(taddr),                                                 \
               "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3])                      \
               : "memory")

// fma.rn.ftz on two lanes (IEEE per lane: the same bits as two fmaf under
// --use_fast_math).
__device__ __forceinline__ void ffma2(float& d0, float& d1, float a0, float a1,
                                      float b0, float b1, float c0, float c1) {
  asm("{\n .reg .b64 a, b, c, d;\n mov.b64 a, {%2, %3};\n mov.b64 b, {%4, %5};\n"
      " mov.b64 c, {%6, %7};\n fma.rn.ftz.f32x2 d, a, b, c;\n"
      " mov.b64 {%0, %1}, d;\n}"
      : "=f"(d0), "=f"(d1)
      : "f"(a0), "f"(a1), "f"(b0), "f"(b1), "f"(c0), "f"(c1));
}

constexpr int kThreads = 256;

__device__ __forceinline__ void math_sync() {
  asm volatile("bar.sync 1, 128;" ::: "memory");
}

// Warps 0-3: math (thread v = value row v = tensor-memory lane v; thread 0
// also issues every MMA, so a tensor-core hand-off costs one named barrier);
// warp 4: loader; warp 5: tensor-memory owner; 6-7 idle. Per chunk c:
//   MMA1(c)  W | Oq over two independent K halves (overlaps the state update
//            of chunk c - 1: K-steps issue as each 32-column state batch lands)
//   u -> MMA2 (U = INV @ u) -> U -> MMA3 (kr^T @ U; Mqk @ U)
//   state update batches -> MMA1(c + 1) K-steps; then out(c).
template <bool HasStateIn, bool HasStateOut>
__global__ void __launch_bounds__(kThreads, 2) k2_tc_kernel(
    cutlass::bfloat16_t const* __restrict__ v_ptr,      // [T, H, 128]
    cutlass::bfloat16_t const* __restrict__ beta_t,     // [H, T] raw
    float const* __restrict__ init_state,               // [N, H, 128 v, 128 k]
    float* __restrict__ final_state,                    // [N, H, 128 v, 128 k]
    cutlass::bfloat16_t* __restrict__ out_ptr,          // [T, H, 128]
    int T_total, int H, int N, int const* __restrict__ cu_seqlens,
    int total_tiles, cutlass::bfloat16_t const* __restrict__ ws_kd,
    cutlass::bfloat16_t const* __restrict__ ws_qd,
    cutlass::bfloat16_t const* __restrict__ ws_kr,
    float const* __restrict__ ws_gt,
    cutlass::bfloat16_t const* __restrict__ ws_inv,
    cutlass::bfloat16_t const* __restrict__ ws_mqk, int flags,
    unsigned const* __restrict__ tile_flags) {
  k31sg::begin_call((long long)(T_total));  // timeout lines: call index and size
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  extern __shared__ __align__(1024) unsigned char smem[];
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int head = blockIdx.x;
  const int seq = N - 1 - int(blockIdx.y);

  const uint32_t sbase = smem_u32(smem);
  const uint32_t bar0 = sbase + kBarOff;
  auto full_bar = [&](int s) { return bar0 + 8 * s; };
  auto empty_bar = [&](int s) { return bar0 + 8 * (kStages + s); };
  const uint32_t d1_bar = bar0 + 8 * (2 * kStages + 0);
  const uint32_t d2_bar = bar0 + 8 * (2 * kStages + 1);
  const uint32_t d3_bar = bar0 + 8 * (2 * kStages + 2);
  const uint32_t d4_bar = bar0 + 8 * (2 * kStages + 3);
  uint32_t* tmem_slot = reinterpret_cast<uint32_t*>(smem + kBarOff + 8 * kNumBars);

  if (threadIdx.x == 0) {
    for (int s = 0; s < kStages; ++s) {
      mbar_init(full_bar(s), 1);
      mbar_init(empty_bar(s), 4);
    }
    mbar_init(d1_bar, 1);
    mbar_init(d2_bar, 1);
    mbar_init(d3_bar, 1);
    mbar_init(d4_bar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (warp == 5) {
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 256;" ::
            "r"(smem_u32(tmem_slot))
        : "memory");
    asm volatile(
        "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::
            : "memory");
  }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tmem_slot;
  // Dev clock stamps (flags bit 0): CTA (0, 0), chunks kDbgChunk0.., thread
  // 0; dumped over its final state.
  long long* dbg = reinterpret_cast<long long*>(smem + kDbgOff);
  const bool dbg_cta = (flags & 1) && blockIdx.x == 0 && blockIdx.y == 0;
#define K2TC_STAMP(c, slot)                                                  \
  if (dbg_cta && threadIdx.x == 0 && (c) >= kDbgChunk0 &&                     \
      (c) < kDbgChunk0 + kDbgChunks)                                          \
    dbg[((c) - kDbgChunk0) * kDbgSlots + (slot)] = clock64();

  const int bos = cu_seqlens[seq];
  const int eos = cu_seqlens[seq + 1];
  int tile_base = 0;
  for (int i = 0; i < seq; ++i) {
    tile_base += (cu_seqlens[i + 1] - cu_seqlens[i] + kChunk - 1) / kChunk;
  }
  const int seq_len = eos - bos;
  const int n_chunks = (seq_len + kChunk - 1) / kChunk;
  const int64_t ws_head = int64_t(head) * total_tiles + tile_base;

  constexpr uint32_t kI16 = idesc_bf16(16, 0);
  constexpr uint32_t kI32 = idesc_bf16(32, 0);
  constexpr uint32_t kI128MN = idesc_bf16(128, 1);

  if (warp < 4) {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 216;");
    const int v = threadIdx.x;
    const bool issuer = (threadIdx.x == 0);
    const uint32_t trow = tmem + (uint32_t(warp * 32) << 16);
    // MMA1 K-steps 2q, 2q + 1 of chunk c (state batch q of 32 columns):
    // K half 0 accumulates at kColW, half 1 at kColW2.
    auto mma1_steps = [&](int c, int q) {
      const uint32_t sst = sbase + (c % kStages) * kStageBytes;
#pragma unroll
      for (int jj = 0; jj < 2; ++jj) {
        const int j = 2 * q + jj;
        const uint32_t d = tmem + (j < 4 ? kColW : kColW2);
        mma_ts(d, tmem + kColAS + 8 * j, sdesc(sst + kOffKdq + 1024 * j, 512, 128),
               kI32, (j & 3) != 0);
      }
    };
    float S[kD];
    const int64_t state_off = (int64_t(seq) * H + head) * kD * kD + int64_t(v) * kD;
    if constexpr (HasStateIn) {
      float4 const* src = reinterpret_cast<float4 const*>(init_state + state_off);
#pragma unroll
      for (int i = 0; i < kD / 4; ++i) {
        float4 x = src[i];
        S[4 * i + 0] = bf16_round(x.x);
        S[4 * i + 1] = bf16_round(x.y);
        S[4 * i + 2] = bf16_round(x.z);
        S[4 * i + 3] = bf16_round(x.w);
      }
    } else {
#pragma unroll
      for (int i = 0; i < kD; ++i) S[i] = 0.0f;
    }
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      uint32_t a[16];
#pragma unroll
      for (int i = 0; i < 16; ++i) a[i] = pack_bf16(S[32 * q + 2 * i], S[32 * q + 2 * i + 1]);
      K2TC_ST16(trow + kColAS + 16 * q, a);
    }
    tc_wait_st();
    tc_fence_before();
    math_sync();
    if (issuer && n_chunks > 0) {
      mbar_wait(full_bar(0), 0);
      tc_fence_after();
#pragma unroll
      for (int q = 0; q < 4; ++q) mma1_steps(0, q);
      tc_commit(d1_bar);
    }

    for (int c = 0; c < n_chunks; ++c) {
      const int stage = c % kStages;
      const uint32_t ph = uint32_t(c / kStages) & 1u;
      const uint32_t cp = uint32_t(c) & 1u;
      unsigned char const* st = smem + stage * kStageBytes;
      const int valid = min(kChunk, seq_len - c * kChunk);
      mbar_wait(full_bar(stage), ph);
      K2TC_STAMP(c, 0);

      // ---- u = bf16(bf16(v - bf16(W)) * beta), W = both K halves ----
      mbar_wait(d1_bar, cp);
      tc_fence_after();
      K2TC_STAMP(c, 1);
      uint32_t oqb[8];
      {
        uint32_t wa[16], wb[16];
        K2TC_LD16(trow + kColW, wa);
        K2TC_LD16(trow + kColW2, wb);
        tc_wait_ld();
        cutlass::bfloat16_t const* vs =
            reinterpret_cast<cutlass::bfloat16_t const*>(st + kOffV);
        cutlass::bfloat16_t const* bs =
            reinterpret_cast<cutlass::bfloat16_t const*>(st + kOffBeta);
        uint32_t au[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          float uu[2];
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const int t = 2 * i + e;
            cutlass::bfloat16_t vv =
                t < valid ? vs[t * kD + v] : cutlass::bfloat16_t(0.0f);
            cutlass::bfloat16_t wbf(__uint_as_float(wa[t]) + __uint_as_float(wb[t]));
            cutlass::bfloat16_t x = (vv - wbf) * bs[t];
            uu[e] = float(x);
          }
          au[i] = pack_bf16(uu[0], uu[1]);
        }
        K2TC_ST8(trow + kColAu, au);
        tc_wait_st();
        tc_fence_before();
        math_sync();
        if (issuer) {
          tc_fence_after();
          mma_ts(tmem + kColU, tmem + kColAu,
                 sdesc(sbase + stage * kStageBytes + kOffInv, 256, 128), kI16, 0);
          tc_commit(d2_bar);
        }
        K2TC_LD16(trow + kColW + 16, wa);
        K2TC_LD16(trow + kColW2 + 16, wb);
        tc_wait_ld();
#pragma unroll
        for (int i = 0; i < 8; ++i)
          oqb[i] = pack_bf16(__uint_as_float(wa[2 * i]) + __uint_as_float(wb[2 * i]),
                             __uint_as_float(wa[2 * i + 1]) + __uint_as_float(wb[2 * i + 1]));
      }

      // ---- U = bf16(INV @ u) ----
      mbar_wait(d2_bar, cp);
      tc_fence_after();
      {
        uint32_t uacc[16], aU[8];
        K2TC_LD16(trow + kColU, uacc);
        tc_wait_ld();
#pragma unroll
        for (int i = 0; i < 8; ++i)
          aU[i] = pack_bf16(__uint_as_float(uacc[2 * i]), __uint_as_float(uacc[2 * i + 1]));
        K2TC_ST8(trow + kColAU, aU);
      }
      tc_wait_st();
      tc_fence_before();
      math_sync();
      if (issuer) {
        tc_fence_after();
        const uint32_t sst = sbase + stage * kStageBytes;
        mma_ts(tmem + kColP, tmem + kColAU, sdesc(sst + kOffKr, 128, 256), kI128MN, 0);
        tc_commit(d3_bar);
        mma_ts(tmem + kColU, tmem + kColAU, sdesc(sst + kOffMqk, 256, 128), kI16, 0);
        tc_commit(d4_bar);
      }
      K2TC_STAMP(c, 2);

      // ---- S = fma(S, g_total, kr^T @ U) per 32-column batch; each batch's
      // bf16(S^T) feeds MMA1(c + 1) as soon as it lands ----
      mbar_wait(d3_bar, cp);
      tc_fence_after();
      K2TC_STAMP(c, 3);
      const bool more = c + 1 < n_chunks;
      float const* gt = reinterpret_cast<float const*>(st + kOffGt);
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        uint32_t p[32];
        K2TC_LD32(trow + kColP + 32 * q, p);
        tc_wait_ld();
        uint32_t a[16];
#pragma unroll
        for (int i = 0; i < 32; i += 4) {
          float4 g4 = *reinterpret_cast<float4 const*>(gt + 32 * q + i);
          float* s4 = S + 32 * q + i;
          ffma2(s4[0], s4[1], s4[0], s4[1], g4.x, g4.y, __uint_as_float(p[i]),
                __uint_as_float(p[i + 1]));
          ffma2(s4[2], s4[3], s4[2], s4[3], g4.z, g4.w, __uint_as_float(p[i + 2]),
                __uint_as_float(p[i + 3]));
          a[i / 2] = pack_bf16(s4[0], s4[1]);
          a[i / 2 + 1] = pack_bf16(s4[2], s4[3]);
        }
        K2TC_ST16(trow + kColAS + 16 * q, a);
        if (more) {
          tc_wait_st();
          tc_fence_before();
          math_sync();
          if (issuer) {
            if (q == 0) mbar_wait(full_bar((c + 1) % kStages), uint32_t((c + 1) / kStages) & 1u);
            tc_fence_after();
            mma1_steps(c + 1, q);
            if (q == 3) tc_commit(d1_bar);
          }
        }
      }
      K2TC_STAMP(c, 4);

      // ---- out(c) = bf16(bf16(Oq) + bf16(Mqk @ U)), under MMA1(c + 1) ----
      mbar_wait(d4_bar, cp);
      tc_fence_after();
      {
        uint32_t o2[16];
        K2TC_LD16(trow + kColU, o2);
        tc_wait_ld();
        cutlass::bfloat16_t* dst = out_ptr + (int64_t(bos) + c * kChunk) * H * kD +
                                   int64_t(head) * kD + v;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const int t = 2 * i + e;
            float a = e ? bf16_hi(oqb[i]) : bf16_lo(oqb[i]);
            float b = bf16_round(__uint_as_float(o2[t]));
            if (t < valid) dst[int64_t(t) * H * kD] = cutlass::bfloat16_t(a + b);
          }
        }
      }
      tc_fence_before();
      __syncwarp();
      K2TC_STAMP(c, 5);
      if (lane == 0) mbar_arrive(empty_bar(stage));
    }
    if constexpr (HasStateOut) {
      float4* dst = reinterpret_cast<float4*>(final_state + state_off);
#pragma unroll
      for (int i = 0; i < kD / 4; ++i) {
        dst[i] = make_float4(bf16_round(S[4 * i]), bf16_round(S[4 * i + 1]),
                             bf16_round(S[4 * i + 2]), bf16_round(S[4 * i + 3]));
      }
    }
  } else {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
    if (warp == 4) {
      // ---------------- loader ----------------
      for (int c = 0; c < n_chunks; ++c) {
        const int stage = c % kStages;
        if (c >= kStages) mbar_wait(empty_bar(stage), uint32_t((c / kStages) - 1) & 1u);
        const int valid = min(kChunk, seq_len - c * kChunk);
        const int tok0 = bos + c * kChunk;
        unsigned char* st = smem + stage * kStageBytes;
        const uint32_t fb = full_bar(stage);
        const uint32_t sst = sbase + stage * kStageBytes;
        const int64_t wi = ws_head + c;
        if (tile_flags != nullptr) {
          if (lane == 0) k3_acquire_tile(tile_flags + wi);
          __syncwarp();
        }
        if (lane < kChunk) {
          float b = 0.0f;
          if (lane < valid) {
            b = sigmoid_tanh_approx_f32(
                float(beta_t[int64_t(head) * T_total + tok0 + lane]));
          }
          reinterpret_cast<cutlass::bfloat16_t*>(st + kOffBeta)[lane] =
              cutlass::bfloat16_t(b);
        }
        __syncwarp();
        if (lane == 0) {
          mbar_expect_tx(fb, 3 * 4096 + 3 * 512 + uint32_t(valid) * 256);
        }
        __syncwarp();
        {
          const int gk = lane & 15;
          cutlass::bfloat16_t const* src =
              (lane < 16 ? ws_kd : ws_qd) + wi * (kChunk * kD) + gk * 128;
          bulk_g2s(sst + kOffKdq + gk * 512 + (lane < 16 ? 0 : 256), src, 256, fb);
        }
        if (lane < valid) {
          bulk_g2s(sst + kOffV + lane * 256,
                   v_ptr + (int64_t(tok0) + lane) * H * kD + int64_t(head) * kD, 256, fb);
        }
        if (lane == 16) bulk_g2s(sst + kOffKr, ws_kr + wi * (kChunk * kD), 4096, fb);
        if (lane == 17) bulk_g2s(sst + kOffInv, ws_inv + wi * (kChunk * kChunk), 512, fb);
        if (lane == 18) bulk_g2s(sst + kOffMqk, ws_mqk + wi * (kChunk * kChunk), 512, fb);
        if (lane == 19) bulk_g2s(sst + kOffGt, ws_gt + wi * kD, 512, fb);
        __syncwarp();
      }
    }
  }
#undef K2TC_STAMP
  tc_fence_before();
  __syncthreads();
  if (dbg_cta && HasStateOut && threadIdx.x < kDbgChunks * kDbgSlots) {
    reinterpret_cast<long long*>(final_state + (int64_t(seq) * H + head) * kD * kD)
        [threadIdx.x] = dbg[threadIdx.x];
  }
  if (warp == 5) {
    tc_fence_after();
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 256;" ::"r"(tmem)
                 : "memory");
  }
#endif
}

}  // namespace k2tc
