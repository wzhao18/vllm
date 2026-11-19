// Kimi-K3: chunk-64 KDA recurrence on the tcgen05 tensor cores (dev stage of
// the fused KDA prefill kernel).
//
// FlashKDA's per-chunk math (fwd_kernel2.cuh, FlashKDA 17a037d, MIT, see
// LICENSE.FlashKDA) with 64-token chunks: 86 dependent state steps per 5.4K
// token sequence instead of 341. Per chunk c of one (sequence, head), with the
// [K, V] state S (fp32, one value row v per math thread, in registers):
//   W   = kd @ bf16(S)              Oq  = qd @ bf16(S)          (fp32 accum)
//   x   = bf16(bf16(v - bf16(W)) * beta)                        (bf16 ops)
//   U   = bf16(INV @ x)                                         (fp32 accum)
//   out = bf16(Oq + Mqk @ U)                                    (fp32 accum)
//   S   = fma(S, g_total, kr^T @ U)                             (fp32 state)
// kd / qd / kr / INV / Mqk / g_total / sigmoid(beta) come from the chunk-64
// prepare (kd, qd, kr relative to the chunk start; INV, Mqk 64 x 64 lower
// triangular) as K-major 8x8 core-matrix byte images:
//   kd, qd [H][C][16 k-groups][64 t][8]      kr^T [H][C][8 t-groups][128 k][8]
//   INV, Mqk [H][C][8 s-groups][64 t][8]     g_total [H][C][128] fp32
// Everything is transposed so V (128) is the MMA M dimension: S^T / x^T / U^T
// are bf16 A operands in tensor memory written row-per-thread, the images are
// B operands in shared memory, accumulators are read back row-per-thread.
//
// Tensor memory (256 columns, two CTAs per SM): regions A0 = [0, 64) and
// A1 = [64, 128) alternate per chunk between W(c) (-> U(c) -> kr^T@U low half)
// and Oq(c) (+= Mqk@U); A2 = [128, 192) takes the kr^T@U high half; [192, 256)
// holds bf16 S^T (then x^T, then U^T). out(c) is read before the state update,
// so MMA1(c + 1) can write W(c + 1) into Oq(c)'s region while the state update
// still reads kr^T@U, and its K-steps issue as each 32-column S^T batch lands.
//
// Warps: 0-3 math (thread 0 issues the MMAs), 4 loader kd|qd (double
// buffered), 5 loader v + beta and INV, 6 loader kr^T + Mqk + g_total,
// 7 tensor-memory owner.
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "k2_tc.cuh"

namespace k64tc {

using namespace k2tc;

constexpr int kC = 64;
constexpr int kD = 128;

constexpr int kOffKdq0 = 0;         // kd image (16 KB) + qd image (16 KB)
constexpr int kOffKdq1 = 32768;
constexpr int kOffV = 65536;        // [64 t][128 v] bf16
constexpr int kOffBeta = 81920;     // bf16[64]
constexpr int kOffInv = 82048;      // 8 KB
constexpr int kOffKr = 90240;       // 16 KB
constexpr int kOffMqk = 106624;     // 8 KB
constexpr int kOffGt = 114816;      // float[128]
constexpr int kOffBar = 115328;
// barriers: full_kdq[2], empty_kdq[2], full_vb, empty_vb, full_inv,
// empty_inv, full_krm, empty_krm, d1, d2, d3
constexpr int kNumBars = 13;
constexpr int kSmemBytes = kOffBar + 8 * kNumBars + 16;

constexpr uint32_t kColS = 192;  // bf16 S^T, then x^T, then U^T (A operand)
constexpr uint32_t kColHi = 128; // kr^T @ U, k >= 64

constexpr int kDbgSlots = 8;

__device__ __forceinline__ void named_sync() {
  asm volatile("bar.sync 1, 128;" ::: "memory");
}

template <bool HasStateIn>
__global__ void __launch_bounds__(256, 2) k64_tc_kernel(
    __nv_bfloat16 const* __restrict__ kd_img,
    __nv_bfloat16 const* __restrict__ qd_img,
    __nv_bfloat16 const* __restrict__ kr_img,
    __nv_bfloat16 const* __restrict__ inv_img,
    __nv_bfloat16 const* __restrict__ mqk_img,
    float const* __restrict__ gt_ptr,            // [H][C][128]
    __nv_bfloat16 const* __restrict__ beta_ptr,  // [H][T] sigmoid(beta)
    __nv_bfloat16 const* __restrict__ v_ptr,     // [T][H][128]
    float const* __restrict__ init_state,        // [N][H][128 v][128 k]
    float* __restrict__ final_state,
    __nv_bfloat16* __restrict__ out_ptr,         // [T][H][128]
    int const* __restrict__ cu_seqlens, int T_total, int H, int N, int C_total,
    int flags, long long* __restrict__ dbg) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  extern __shared__ __align__(1024) unsigned char smem[];
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int head = blockIdx.x;
  const int seq = N - 1 - int(blockIdx.y);
  const uint32_t sbase = smem_u32(smem);
  const uint32_t bar0 = sbase + kOffBar;
  const uint32_t full_kdq0 = bar0, empty_kdq0 = bar0 + 16;
  const uint32_t full_vb = bar0 + 32, empty_vb = bar0 + 40;
  const uint32_t full_inv = bar0 + 48, empty_inv = bar0 + 56;
  const uint32_t full_krm = bar0 + 64, empty_krm = bar0 + 72;
  const uint32_t d1_bar = bar0 + 80, d2_bar = bar0 + 88, d3_bar = bar0 + 96;
  uint32_t* tmem_slot = reinterpret_cast<uint32_t*>(smem + kOffBar + 8 * kNumBars);

  if (threadIdx.x == 0) {
    mbar_init(full_kdq0, 1);
    mbar_init(full_kdq0 + 8, 1);
    mbar_init(empty_kdq0, 1);
    mbar_init(empty_kdq0 + 8, 1);
    mbar_init(full_vb, 1);
    mbar_init(empty_vb, 4);
    mbar_init(full_inv, 1);
    mbar_init(empty_inv, 1);
    mbar_init(full_krm, 1);
    mbar_init(empty_krm, 5);
    mbar_init(d1_bar, 1);
    mbar_init(d2_bar, 1);
    mbar_init(d3_bar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (warp == 7) {
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 256;" ::
            "r"(smem_u32(tmem_slot))
        : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::
                     : "memory");
  }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem = *tmem_slot;

  const int bos = cu_seqlens[seq];
  const int seq_len = cu_seqlens[seq + 1] - bos;
  const int n_chunks = (seq_len + kC - 1) / kC;
  int cbase = 0;
  for (int i = 0; i < seq; ++i) cbase += (cu_seqlens[i + 1] - cu_seqlens[i] + kC - 1) / kC;
  // flags bit 1: every chunk reads chunk 0's images (L2-resident: the
  // recurrence's compute time without the workspace HBM stream).
  const bool reuse = (flags & 2) != 0;
  auto img = [&](int c) { return int64_t(head) * C_total + cbase + (reuse ? 0 : c); };
  const bool dbg_cta = (flags & 1) && blockIdx.x == 0 && blockIdx.y == 0;

  constexpr uint32_t kI64 = idesc_bf16(64, 0);

  if (warp < 4) {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 216;");
    const int v = threadIdx.x;
    const bool issuer = threadIdx.x == 0;
    const uint32_t trow = tmem + (uint32_t(warp * 32) << 16);
    auto stamp = [&](int c, int slot) {
      if (dbg_cta && issuer && c < 128) dbg[c * kDbgSlots + slot] = clock64();
    };
    // MMA1(c) K-steps j: W into region rw, Oq into region ro.
    auto mma1_w = [&](int c, int j, uint32_t rw) {
      const uint32_t b = sbase + ((c & 1) ? kOffKdq1 : kOffKdq0);
      mma_ts(tmem + rw, tmem + kColS + 8 * j, sdesc(b + 2048 * j, 1024, 128), kI64, j != 0);
    };
    auto mma1_o = [&](int c, int j, uint32_t ro) {
      const uint32_t b = sbase + ((c & 1) ? kOffKdq1 : kOffKdq0) + 16384;
      mma_ts(tmem + ro, tmem + kColS + 8 * j, sdesc(b + 2048 * j, 1024, 128), kI64, j != 0);
    };

    float S[kD];
    const int64_t state_off = (int64_t(seq) * H + head) * kD * kD + int64_t(v) * kD;
    if constexpr (HasStateIn) {
      float4 const* src = reinterpret_cast<float4 const*>(init_state + state_off);
#pragma unroll
      for (int i = 0; i < kD / 4; ++i) {
        float4 x = src[i];
        S[4 * i] = x.x;
        S[4 * i + 1] = x.y;
        S[4 * i + 2] = x.z;
        S[4 * i + 3] = x.w;
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
      K2TC_ST16(trow + kColS + 16 * q, a);
    }
    tc_wait_st();
    tc_fence_before();
    named_sync();
    if (issuer && n_chunks > 0) {
      mbar_wait(full_kdq0, 0);
      tc_fence_after();
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        mma1_w(0, j, 0);
        mma1_o(0, j, 64);
      }
      tc_commit(d1_bar);
      tc_commit(empty_kdq0);
    }

    for (int c = 0; c < n_chunks; ++c) {
      const uint32_t cp = uint32_t(c) & 1u;
      const uint32_t rw = cp ? 64 : 0;   // W(c) -> U(c) -> kr^T@U low half
      const uint32_t ro = cp ? 0 : 64;   // Oq(c)
      const int valid = min(kC, seq_len - c * kC);
      stamp(c, 0);

      // ---- x = bf16(bf16(v - bf16(W)) * beta) -> x^T ----
      mbar_wait(d1_bar, cp);
      mbar_wait(full_vb, cp);
      tc_fence_after();
      stamp(c, 1);
      {
        __nv_bfloat16 const* vs = reinterpret_cast<__nv_bfloat16 const*>(smem + kOffV);
        __nv_bfloat16 const* bs = reinterpret_cast<__nv_bfloat16 const*>(smem + kOffBeta);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          uint32_t w[32], ax[16];
          K2TC_LD32(trow + rw + 32 * h, w);
          tc_wait_ld();
#pragma unroll
          for (int i = 0; i < 16; ++i) {
            float xx[2];
#pragma unroll
            for (int e = 0; e < 2; ++e) {
              const int t = 32 * h + 2 * i + e;
              const __nv_bfloat16 vv = t < valid ? vs[t * kD + v] : __float2bfloat16(0.0f);
              const __nv_bfloat16 wb = __float2bfloat16(__uint_as_float(w[2 * i + e]));
              const __nv_bfloat16 d = __float2bfloat16(__bfloat162float(vv) - __bfloat162float(wb));
              const __nv_bfloat16 bt = t < valid ? bs[t] : __float2bfloat16(0.0f);
              xx[e] = __bfloat162float(__float2bfloat16(__bfloat162float(d) * __bfloat162float(bt)));
            }
            ax[i] = pack_bf16(xx[0], xx[1]);
          }
          K2TC_ST16(trow + kColS + 16 * h, ax);
        }
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(empty_vb);
      tc_wait_st();
      tc_fence_before();
      named_sync();
      if (issuer) {
        mbar_wait(full_inv, cp);
        tc_fence_after();
#pragma unroll
        for (int j = 0; j < 4; ++j)
          mma_ts(tmem + rw, tmem + kColS + 8 * j, sdesc(sbase + kOffInv + 2048 * j, 1024, 128),
                 kI64, j != 0);
        tc_commit(d2_bar);
        tc_commit(empty_inv);
      }
      stamp(c, 2);

      // ---- U = bf16(INV @ x) -> U^T; MMA3 (kr^T @ U) + MMA4 (Oq += Mqk @ U) ----
      mbar_wait(d2_bar, 0);  // d2 completes twice per chunk: MMA2, MMA4
      tc_fence_after();
      stamp(c, 3);
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        uint32_t u[32], au[16];
        K2TC_LD32(trow + rw + 32 * h, u);
        tc_wait_ld();
#pragma unroll
        for (int i = 0; i < 16; ++i)
          au[i] = pack_bf16(__uint_as_float(u[2 * i]), __uint_as_float(u[2 * i + 1]));
        K2TC_ST16(trow + kColS + 16 * h, au);
      }
      tc_wait_st();
      tc_fence_before();
      named_sync();
      if (issuer) {
        mbar_wait(full_krm, cp);
        tc_fence_after();
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          mma_ts(tmem + rw, tmem + kColS + 8 * j, sdesc(sbase + kOffKr + 4096 * j, 2048, 128),
                 kI64, j != 0);
          mma_ts(tmem + kColHi, tmem + kColS + 8 * j,
                 sdesc(sbase + kOffKr + 1024 + 4096 * j, 2048, 128), kI64, j != 0);
        }
        tc_commit(d3_bar);
#pragma unroll
        for (int j = 0; j < 4; ++j)
          mma_ts(tmem + ro, tmem + kColS + 8 * j, sdesc(sbase + kOffMqk + 2048 * j, 1024, 128),
                 kI64, 1);
        tc_commit(d2_bar);  // second d2 phase: MMA4 done (out(c) ready)
        tc_commit(empty_krm);
      }
      stamp(c, 4);

      // ---- out(c) = bf16(Oq + Mqk @ U): read before the state update so
      // MMA1(c + 1) may write W(c + 1) into this region ----
      mbar_wait(d2_bar, 1);
      tc_fence_after();
      {
        __nv_bfloat16* dst = out_ptr + (int64_t(bos) + c * kC) * H * kD + int64_t(head) * kD + v;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          uint32_t o[32];
          K2TC_LD32(trow + ro + 32 * h, o);
          tc_wait_ld();
#pragma unroll
          for (int i = 0; i < 32; ++i) {
            const int t = 32 * h + i;
            if (t < valid) dst[int64_t(t) * H * kD] = __float2bfloat16(__uint_as_float(o[i]));
          }
        }
      }
      stamp(c, 5);

      // ---- S = fma(S, g_total, kr^T @ U) per 32-column batch ----
      mbar_wait(d3_bar, cp);
      mbar_wait(full_krm, cp);
      tc_fence_after();
      const bool more = c + 1 < n_chunks;
      float const* gt = reinterpret_cast<float const*>(smem + kOffGt);
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        uint32_t p[32];
        K2TC_LD32(trow + (q < 2 ? rw + 32 * q : kColHi + 32 * (q - 2)), p);
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
        K2TC_ST16(trow + kColS + 16 * q, a);
        if (q == 3) {
          __syncwarp();
          if (lane == 0) mbar_arrive(empty_krm);
        }
        if (more) {
          tc_wait_st();
          tc_fence_before();
          named_sync();
          if (issuer) {
            if (q == 0) mbar_wait(full_kdq0 + 8 * ((c + 1) & 1), uint32_t((c + 1) >> 1) & 1u);
            tc_fence_after();
            // W(c + 1) -> ro (out(c) already read); Oq(c + 1) -> rw once the
            // low half of kr^T@U has been read (after batch 1).
            mma1_w(c + 1, 2 * q, ro);
            mma1_w(c + 1, 2 * q + 1, ro);
            if (q == 1) {
#pragma unroll
              for (int j = 0; j < 4; ++j) mma1_o(c + 1, j, rw);
            }
            if (q == 3) {
#pragma unroll
              for (int j = 4; j < 8; ++j) mma1_o(c + 1, j, rw);
              tc_commit(d1_bar);
              tc_commit(empty_kdq0 + 8 * ((c + 1) & 1));
            }
          }
        }
      }
      stamp(c, 6);
    }
    {
      float4* dst = reinterpret_cast<float4*>(final_state + state_off);
#pragma unroll
      for (int i = 0; i < kD / 4; ++i)
        dst[i] = make_float4(S[4 * i], S[4 * i + 1], S[4 * i + 2], S[4 * i + 3]);
    }
  } else {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
    if (warp == 4) {
      for (int c = 0; c < n_chunks; ++c) {
        const int b = c & 1;
        if (c >= 2) mbar_wait(empty_kdq0 + 8 * b, uint32_t((c >> 1) - 1) & 1u);
        if (lane == 0) {
          const uint32_t fb = full_kdq0 + 8 * b;
          const uint32_t dst = sbase + (b ? kOffKdq1 : kOffKdq0);
          mbar_expect_tx(fb, 32768);
          bulk_g2s(dst, kd_img + img(c) * (kC * kD), 16384, fb);
          bulk_g2s(dst + 16384, qd_img + img(c) * (kC * kD), 16384, fb);
        }
        __syncwarp();
      }
    } else if (warp == 5) {
      for (int c = 0; c < n_chunks; ++c) {
        const int valid = min(kC, seq_len - c * kC);
        const int tok0 = bos + c * kC;
        if (c >= 1) mbar_wait(empty_vb, uint32_t(c - 1) & 1u);
        __nv_bfloat16* bsm = reinterpret_cast<__nv_bfloat16*>(smem + kOffBeta);
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int t = lane + 32 * e;
          bsm[t] = t < valid ? beta_ptr[int64_t(head) * T_total + tok0 + t] : __float2bfloat16(0.0f);
        }
        __syncwarp();
        if (lane == 0) mbar_expect_tx(full_vb, uint32_t(valid) * 256);
        __syncwarp();
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int t = lane + 32 * e;
          if (t < valid)
            bulk_g2s(sbase + kOffV + t * 256, v_ptr + (int64_t(tok0) + t) * H * kD + int64_t(head) * kD,
                     256, full_vb);
        }
        if (c >= 1) mbar_wait(empty_inv, uint32_t(c - 1) & 1u);
        if (lane == 0) {
          mbar_expect_tx(full_inv, 8192);
          bulk_g2s(sbase + kOffInv, inv_img + img(c) * (kC * kC), 8192, full_inv);
        }
        __syncwarp();
      }
    } else if (warp == 6) {
      for (int c = 0; c < n_chunks; ++c) {
        if (c >= 1) mbar_wait(empty_krm, uint32_t(c - 1) & 1u);
        if (lane == 0) {
          mbar_expect_tx(full_krm, 16384 + 8192 + 512);
          bulk_g2s(sbase + kOffKr, kr_img + img(c) * (kC * kD), 16384, full_krm);
          bulk_g2s(sbase + kOffMqk, mqk_img + img(c) * (kC * kC), 8192, full_krm);
          bulk_g2s(sbase + kOffGt, gt_ptr + img(c) * kD, 512, full_krm);
        }
        __syncwarp();
      }
    }
  }
  tc_fence_before();
  __syncthreads();
  if (warp == 7) {
    tc_fence_after();
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 256;" ::"r"(tmem)
                 : "memory");
  }
#endif
}

}  // namespace k64tc

void launch_k64_tc(void const* kd, void const* qd, void const* kr, void const* inv,
                   void const* mqk, float const* gt, void const* beta, void const* v,
                   float const* init_state, float* final_state, void* out,
                   int const* cu_seqlens, int T_total, int H, int N, int C_total,
                   int flags, long long* dbg, cudaStream_t stream) {
  using k64tc::k64_tc_kernel;
  auto launch = [&](auto kernel) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, k64tc::kSmemBytes);
    kernel<<<dim3(H, N), dim3(256), k64tc::kSmemBytes, stream>>>(
        static_cast<__nv_bfloat16 const*>(kd), static_cast<__nv_bfloat16 const*>(qd),
        static_cast<__nv_bfloat16 const*>(kr), static_cast<__nv_bfloat16 const*>(inv),
        static_cast<__nv_bfloat16 const*>(mqk), gt, static_cast<__nv_bfloat16 const*>(beta),
        static_cast<__nv_bfloat16 const*>(v), init_state, final_state,
        static_cast<__nv_bfloat16*>(out), cu_seqlens, T_total, H, N, C_total, flags, dbg);
  };
  if (init_state != nullptr) {
    launch(k64_tc_kernel<true>);
  } else {
    launch(k64_tc_kernel<false>);
  }
}
