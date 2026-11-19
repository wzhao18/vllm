// Kimi-K3: chunk-64 KDA prepare (dev stage of the fused KDA prefill kernel).
//
// Builds k64_tc.cu's chunk images from post-conv q / k, the raw gate g and
// raw beta, in FlashKDA K1's rounding style (the torch reference is
// benchmarks/kimi_k3/k64_diag.py prep64):
//   qn, kn   = bf16(x * rsqrt(sum x^2 + 1e-6))
//   gv       = lb * log2(e) * sigmoid(exp(A_log) * (g + dt_bias))   (fp32)
//   gc       = cumsum_t gv (chunk-relative, fp32), gT = gc[last valid row]
//   kd       = bf16(kn * bf16(exp2(gc)))      qd = bf16(bf16(qn * bf16(exp2(gc))) * bf16(scale))
//   kr       = bf16(kn * bf16(exp2(gT - gc))) g_total = exp2(gT)
//   A[i, j]  = sum_d kn_i kn_j exp2(gc_i - gc_j),  Mq[i, j] = sum_d scale qn_i kn_j exp2(gc_i - gc_j)
//   INV      = bf16((I + beta_i tril(A, -1))^-1),  Mqk = bf16(tril(Mq))
// The pairwise decays are factored per 16-row block I with the reference row
// r = 16 I: exp2(gc_i - gc_r) <= 1 on the row operand, exp2(gc_r - gc_j) on
// the column operand (<= 1 below the block, bounded by 15 rows of decay inside
// it), which is the operand form the tensor-core version uses.
//
// One CTA per (chunk, head), fp32 SIMT (validation stage: the fused kernel
// moves the dot products onto the tensor cores).
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace prep64 {

constexpr int kC = 64;
constexpr int kD = 128;
constexpr int kP = kD + 1;  // padded fp32 rows

struct Smem {
  float k[kC][kP];   // kn (bf16-rounded)
  float q[kC][kP];   // qn (bf16-rounded)
  float g[kC][kP];   // gv, then gc
  float kc[kC][kP];  // column operand of the current block
  float a[kC][kC + 1];
  float m[kC][kC + 1];
  float beta[kC];
};

__device__ __forceinline__ float bfr(float x) { return __bfloat162float(__float2bfloat16(x)); }

__global__ void __launch_bounds__(256) prep64_kernel(
    __nv_bfloat16 const* __restrict__ q, __nv_bfloat16 const* __restrict__ k,
    __nv_bfloat16 const* __restrict__ g, __nv_bfloat16 const* __restrict__ beta,
    int64_t row_stride_qk, int64_t row_stride_g, int64_t row_stride_beta,
    float const* __restrict__ a_log, float const* __restrict__ dt_bias, float lower_bound,
    float scale, int const* __restrict__ cu_seqlens, int N, int H, int T_total,
    __nv_bfloat16* __restrict__ kdq_img, __nv_bfloat16* __restrict__ kr_img,
    __nv_bfloat16* __restrict__ inv_img, __nv_bfloat16* __restrict__ mqk_img,
    float* __restrict__ gt_out, __nv_bfloat16* __restrict__ beta_out) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  Smem& s = *reinterpret_cast<Smem*>(smem_raw);
  const int ci = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  // chunk -> (sequence, first token, valid rows)
  int tok0 = 0, valid = 0, base = 0;
  for (int b = 0; b < N; ++b) {
    const int bos = cu_seqlens[b], len = cu_seqlens[b + 1] - bos;
    const int nc = (len + kC - 1) / kC;
    if (ci < base + nc) {
      tok0 = bos + (ci - base) * kC;
      valid = min(kC, len - (ci - base) * kC);
      break;
    }
    base += nc;
  }
  const float gs = lower_bound * 1.4426950408889634f;
  const float amp = expf(a_log[h]);
  // ---- rows: L2 norm of q / k, gate activation ------------------------------
  for (int t = warp; t < kC; t += 8) {
    float qv[4], kv[4], gv[4], sq = 0.f, sk = 0.f;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int d = lane + 32 * e;
      qv[e] = kv[e] = gv[e] = 0.f;
      if (t < valid) {
        const int64_t row = tok0 + t;
        qv[e] = __bfloat162float(q[row * row_stride_qk + h * kD + d]);
        kv[e] = __bfloat162float(k[row * row_stride_qk + h * kD + d]);
        const float x = __bfloat162float(g[row * row_stride_g + h * kD + d]) + dt_bias[h * kD + d];
        gv[e] = gs / (1.f + expf(-amp * x));
      }
      sq += qv[e] * qv[e];
      sk += kv[e] * kv[e];
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      sq += __shfl_xor_sync(0xffffffffu, sq, o);
      sk += __shfl_xor_sync(0xffffffffu, sk, o);
    }
    const float rq = rsqrtf(sq + 1e-6f), rk = rsqrtf(sk + 1e-6f);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int d = lane + 32 * e;
      s.q[t][d] = bfr(qv[e] * rq);
      s.k[t][d] = bfr(kv[e] * rk);
      s.g[t][d] = gv[e];
    }
  }
  if (tid < kC) {
    float b = 0.f;
    if (tid < valid) {
      b = bfr(1.f / (1.f + expf(-__bfloat162float(beta[int64_t(tok0 + tid) * row_stride_beta + h]))));
      beta_out[int64_t(h) * T_total + tok0 + tid] = __float2bfloat16(b);
    }
    s.beta[tid] = b;
  }
  __syncthreads();
  // ---- gate cumsum per channel (sequential, chunk-relative) ------------------
  if (tid < kD) {
    float acc = 0.f;
    for (int t = 0; t < valid; ++t) {
      acc += s.g[t][tid];
      s.g[t][tid] = acc;
    }
  }
  __syncthreads();
  const int64_t img = int64_t(h) * gridDim.x + ci;
  // ---- decayed operands: kd | qd, kr^T, g_total -------------------------------
  {
    const float sc = bfr(scale);
    __nv_bfloat16* kdq = kdq_img + img * (2 * kC * kD);
    __nv_bfloat16* kr = kr_img + img * (kC * kD);
    for (int idx = tid; idx < kC * kD; idx += 256) {
      const int t = idx / kD, d = idx % kD;
      float kd = 0.f, qd = 0.f, krv = 0.f;
      if (t < valid) {
        const float gc = s.g[t][d], gT = s.g[valid - 1][d];
        const float e = bfr(exp2f(gc));
        kd = s.k[t][d] * e;
        qd = bfr(s.q[t][d] * e) * sc;
        krv = s.k[t][d] * bfr(exp2f(gT - gc));
      }
      kdq[((d >> 3) * 128 + t) * 8 + (d & 7)] = __float2bfloat16(kd);
      kdq[((d >> 3) * 128 + 64 + t) * 8 + (d & 7)] = __float2bfloat16(qd);
      kr[((t >> 3) * 128 + d) * 8 + (t & 7)] = __float2bfloat16(krv);
    }
    if (tid < kD) gt_out[img * kD + tid] = valid > 0 ? exp2f(s.g[valid - 1][tid]) : 1.f;
  }
  // ---- A and Mq per 16-row block with factored decays --------------------------
  for (int blk = 0; blk < 4; ++blk) {
    const int r = 16 * blk, jmax = r + 16;
    __syncthreads();
    for (int idx = tid; idx < jmax * kD; idx += 256) {
      const int j = idx / kD, d = idx % kD;
      s.kc[j][d] = j < valid && r < valid ? s.k[j][d] * exp2f(s.g[r][d] - s.g[j][d]) : 0.f;
    }
    __syncthreads();
    // 16 rows x jmax columns, 4 threads per (i, j) pair splitting d.
    for (int pair = tid >> 2; pair < 16 * jmax; pair += 64) {
      const int i = r + pair / jmax, j = pair % jmax, part = tid & 3;
      float sa = 0.f, sm = 0.f;
      if (j <= i && i < valid) {
        for (int d = part; d < kD; d += 4) {
          const float f = exp2f(s.g[i][d] - s.g[r][d]);
          sa += s.k[i][d] * f * s.kc[j][d];
          sm += s.q[i][d] * scale * f * s.kc[j][d];
        }
      }
      sa += __shfl_xor_sync(0xffffffffu, sa, 1);
      sa += __shfl_xor_sync(0xffffffffu, sa, 2);
      sm += __shfl_xor_sync(0xffffffffu, sm, 1);
      sm += __shfl_xor_sync(0xffffffffu, sm, 2);
      if (part == 0) {
        s.a[i][j] = sa;
        s.m[i][j] = sm;
      }
    }
  }
  __syncthreads();
  // ---- INV = (I + beta tril(A, -1))^-1 by forward substitution, column per thread
  __nv_bfloat16* inv = inv_img + img * (kC * kC);
  __nv_bfloat16* mqk = mqk_img + img * (kC * kC);
  if (tid < kC) {
    const int j = tid;
    float x[kC];
#pragma unroll
    for (int i = 0; i < kC; ++i) {
      float acc = i == j ? 1.f : 0.f;
      if (i > j) {
#pragma unroll
        for (int kk = 0; kk < i; ++kk)
          if (kk >= j) acc -= s.beta[i] * s.a[i][kk] * x[kk];
      }
      x[i] = i >= j ? acc : 0.f;
    }
#pragma unroll
    for (int i = 0; i < kC; ++i) {
      const bool in = i < valid && j < valid;
      inv[((j >> 3) * 64 + i) * 8 + (j & 7)] = __float2bfloat16(in ? x[i] : 0.f);
      mqk[((j >> 3) * 64 + i) * 8 + (j & 7)] =
          __float2bfloat16(in && j <= i ? s.m[i][j] : 0.f);
    }
  }
}

}  // namespace prep64

void launch_prep64(void const* q, void const* k, void const* g, void const* beta,
                   int64_t row_stride_qk, int64_t row_stride_g, int64_t row_stride_beta,
                   float const* a_log, float const* dt_bias, float lower_bound, float scale,
                   int const* cu_seqlens, int N, int H, int T_total, int C_total,
                   void* kdq, void* kr, void* inv, void* mqk, float* gt, void* beta_out,
                   cudaStream_t stream) {
  const int smem = sizeof(prep64::Smem);
  cudaFuncSetAttribute(prep64::prep64_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  prep64::prep64_kernel<<<dim3(C_total, H), 256, smem, stream>>>(
      static_cast<__nv_bfloat16 const*>(q), static_cast<__nv_bfloat16 const*>(k),
      static_cast<__nv_bfloat16 const*>(g), static_cast<__nv_bfloat16 const*>(beta),
      row_stride_qk, row_stride_g, row_stride_beta, a_log, dt_bias, lower_bound, scale,
      cu_seqlens, N, H, T_total, static_cast<__nv_bfloat16*>(kdq),
      static_cast<__nv_bfloat16*>(kr), static_cast<__nv_bfloat16*>(inv),
      static_cast<__nv_bfloat16*>(mqk), gt, static_cast<__nv_bfloat16*>(beta_out));
}
