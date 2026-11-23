// Kimi-K3: chunk-64 KDA prepare on the raw in_proj rows (short conv fused),
// tensor-core block products, for the chunk-64 tcgen05 recurrence (k64_tc.cu).
//
// One CTA per (chunk, head), 256 threads, two CTAs per SM.
//   conv     q|k|v = SiLU(causal width-4 conv) of the raw rows (2D TMA boxes with
//            3 history rows), FP32 weights and conv state: the qkv_conv kernel's
//            arithmetic (bit for bit); the chunk-0 CTA of a sequence reads and
//            rewrites its conv state
//   v image  bf16 post-conv v, [8 t-groups][128 v][8 t]
//   qn, kn   = bf16(x * rsqrt(sum x^2 + 1e-6))
//   gc       = cumsum_t gate_scale * sigmoid(exp(A_log) (g + dt_bias)) (FP32,
//            chunk-relative), gT = gc[last valid row], g_total = exp2(gT)
//   kd = kn * bf16(exp2 gc), qd = (qn * bf16(exp2 gc)) * bf16(scale),
//   kr = kn * bf16(exp2(gT - gc))                       (bf16 products, RN)
//   A, Mq    16-row blocks I with the reference row r_I = 16 I: row operands
//            kn_i * bf16(exp2(gc_i - gc_rI)) (<= 1) and (qn_i * that) * bf16(scale),
//            column operand kn_j * bf16(exp2(gc_rJ - gc_j)) of j's own block J,
//            scaled by bf16(exp2(gc_rI - gc_rJ)) (<= 1) for the tiles I > J;
//            mma.sync m16n8k16 BF16 -> FP32
//   INV      = (I + beta tril(A, -1))^-1: 16x16 diagonal blocks by FP32 forward
//            substitution, off-diagonal blocks by BF16 tensor-core block products
//            (FlashKDA's 16x16 merge, one warp per block column), bf16 image
//   Mqk      = bf16(tril(Mq))
// The images are k64_tc.cu's (rows past the sequence end zero).
#include <cstddef>
#include <cstdint>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cudaTypedefs.h>
#include "k31_spin_guard.cuh"  // bounded mbarrier waits
K31SG_DEFINE_SET_RANK(prep64_tc)  // rank of the timeout line (binding: set_spin_rank)

namespace prep64tc {

constexpr int kC = 64;
constexpr int kD = 128;
constexpr int kPitch = 136;  // bf16 per operand row: conflict-free ldmatrix
constexpr int kThreads = 256;
constexpr int kRawRows = kC + 3;
constexpr int kRawBox = kRawRows * kD * 2;
constexpr int kOffRawG = 3 * kRawBox;
constexpr int kRawBytes = 3 * kRawBox + kC * kD * 2;
constexpr int kKrPitch = 24;  // bf16 per row of the per-warp kr transpose tile (16 x 16)

struct Smem {
  union {
    unsigned char raw[kRawBytes];  // q | k | v boxes (67 rows) and the gate box (64 rows)
    struct {
      __nv_bfloat16 rk[kC * kPitch];
      __nv_bfloat16 rq[kC * kPitch];
      __nv_bfloat16 ck[kC * kPitch];  // column operand, own-block reference
      __nv_bfloat16 krs[8][16 * kKrPitch];
    } op;
    struct {
      float x[kC][kC + 4];  // INV
      float t[3][16][20];   // per-warp T tiles of the block-column chains
    } inv;
  } a;
  __nv_bfloat16 qkn[2][kC][kPitch];  // bf16 normalized post-conv q, k (padded rows)
  union {
    float g[kC][kD + 4];                // chunk-relative gate cumsum
    __nv_bfloat16 vs[8][8 * kPitch];    // per-warp v transpose tiles (8 tokens x 128)
    float l[kC][kC + 4];                // beta_i A_ij below the diagonal
  } c;
  __nv_bfloat162 pf[6][kD / 2];  // bf16 exp2(gc_rI - gc_rJ), I > J (pair I (I - 1) / 2 + J)
  float tot[16][kD];
  float wc[3][4][kD];  // conv weights of this head, [q|k|v][tap][channel]
  float dt[kD];
  float gref[4][kD];
  float gT[kD];
  float beta[kC];
  alignas(8) uint64_t bar;
};

struct Params {
  CUtensorMap tm_x;  // raw q|k|v rows: dims (3 H 128, T), box (128, 67)
  CUtensorMap tm_g;  // gate rows: dims (H 128, T), box (128, 64)
  __nv_bfloat16 const* x;  // (T, 3 H 128) raw q|k|v rows
  int64_t ldx;
  float const* conv_w;  // (3 H 128, 4) FP32
  float* conv_state;    // (lines, 3 H 128, >= 3) FP32
  int64_t st_s0, st_s1, st_s2;
  int32_t const* cache_indices;
  int64_t ci_s0;
  bool const* has_init;
  int null_block_id;  // < 0: none
  __nv_bfloat16 const* g;  // (T, H 128) raw gate, row stride ldg
  int64_t ldg;
  __nv_bfloat16 const* beta;  // (T, H) raw logits, row stride ldb
  int64_t ldb;
  float const* a_log;
  float const* dt_bias;
  float gate_scale;  // lower_bound * log2(e)
  float scale;
  int const* cu_seqlens;
  int N, H, T, C_total;
  __nv_bfloat16* kdq;
  __nv_bfloat16* kr;
  __nv_bfloat16* inv;
  __nv_bfloat16* mqk;
  float* gt;
  __nv_bfloat16* v_img;
  __nv_bfloat16* beta_out;  // (H, T) bf16 sigmoid(beta)
  long long* dbg;  // dev: per-phase globaltimer of the CTAs of heads 0, 1 (or null)
};

__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float rcp(float x) {
  float y;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float sigmoid_tanh(float x) {
  float th;
  asm("tanh.approx.f32 %0, %1;" : "=f"(th) : "f"(x * 0.5f));
  return th * 0.5f + 0.5f;
}

// qkv_conv's SiLU (Triton's acc / (1 + exp(-acc)) as lowered for FP32).
__device__ __forceinline__ float silu_div(float acc) {
  float t, e, d, r;
  asm("mul.f32 %0, %1, 0fBFB8AA3B;" : "=f"(t) : "f"(acc));
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e) : "f"(t));
  asm("add.f32 %0, %1, 0f3F800000;" : "=f"(d) : "f"(e));
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(acc), "f"(d));
  return r;
}

__device__ __forceinline__ float bfr(float x) { return __bfloat162float(__float2bfloat16(x)); }

__device__ __forceinline__ uint32_t pack2(float a, float b) {
  __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&h);
}

__device__ __forceinline__ void unpack8(uint4 v, float (&f)[8]) {
  const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    f[2 * i] = __uint_as_float(w[i] << 16);
    f[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
  }
}

__device__ __forceinline__ uint4 pack8(const float (&f)[8]) {
  return make_uint4(pack2(f[0], f[1]), pack2(f[2], f[3]), pack2(f[4], f[5]), pack2(f[6], f[7]));
}

__device__ __forceinline__ uint4 ldg16(void const* p) {
  return __ldg(reinterpret_cast<uint4 const*>(p));
}

__device__ __forceinline__ void tma2d(uint32_t dst, CUtensorMap const* tm, int c0, int c1,
                                      uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, "
      "%3}], [%4];" ::"r"(dst),
      "l"(reinterpret_cast<uint64_t>(tm)), "r"(c0), "r"(c1), "r"(bar)
      : "memory");
}

// qkv_conv's FP32-weight row: FP32 FMA chain from +0, oldest tap first, then
// SiLU (a * rcp(1 + e) for 1 + e <= 2^126, else the full-range division).
__device__ __forceinline__ void conv_row(const float (&h0)[8], const float (&h1)[8],
                                         const float (&h2)[8], const float (&cur)[8],
                                         const float (&w)[4][8], float (&o)[8]) {
  float dmax = 0.0f;
  float acc[8];
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    float2 a = __ffma2_rn(make_float2(h0[2 * q], h0[2 * q + 1]),
                          make_float2(w[0][2 * q], w[0][2 * q + 1]), make_float2(0.0f, 0.0f));
    a = __ffma2_rn(make_float2(h1[2 * q], h1[2 * q + 1]), make_float2(w[1][2 * q], w[1][2 * q + 1]), a);
    a = __ffma2_rn(make_float2(h2[2 * q], h2[2 * q + 1]), make_float2(w[2][2 * q], w[2][2 * q + 1]), a);
    a = __ffma2_rn(make_float2(cur[2 * q], cur[2 * q + 1]), make_float2(w[3][2 * q], w[3][2 * q + 1]), a);
    acc[2 * q] = a.x;
    acc[2 * q + 1] = a.y;
    const float2 t = __fmul2_rn(a, make_float2(-1.44269502162933349609375f, -1.44269502162933349609375f));
    const float2 d = __fadd2_rn(make_float2(ex2(t.x), ex2(t.y)), make_float2(1.0f, 1.0f));
    const float2 r = __fmul2_rn(a, make_float2(rcp(d.x), rcp(d.y)));
    o[2 * q] = r.x;
    o[2 * q + 1] = r.y;
    dmax = fmaxf(dmax, fmaxf(d.x, d.y));
  }
  if (dmax > 0x1p126f) {
#pragma unroll
    for (int i = 0; i < 8; ++i) o[i] = silu_div(acc[i]);
  }
}

__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t (&r)[4]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}

__device__ __forceinline__ void mma16816(float (&c)[4], const uint32_t (&a)[4], uint32_t b0,
                                         uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, "
      "{%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t smem_u32(void const* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), *reinterpret_cast<__nv_bfloat162*>(&b));
  return *reinterpret_cast<uint32_t*>(&r);
}

__device__ __forceinline__ void ldsm_x4_t(uint32_t addr, uint32_t (&r)[4]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}

__device__ __forceinline__ void stg32(void* p, uint32_t v) { *reinterpret_cast<uint32_t*>(p) = v; }

__global__ void __launch_bounds__(kThreads, 2) prep64_tc_kernel(const __grid_constant__ Params p) {
  k31sg::begin_call((long long)(p.T));  // timeout lines: call index and size
  extern __shared__ __align__(128) unsigned char smem_raw[];
  Smem& s = *reinterpret_cast<Smem*>(smem_raw);
  const int ci = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int tg = tid >> 4, cg = tid & 15;
  const int H = p.H;
  const int HD = H * kD;

  int seq = 0, tok0 = 0, valid = 0, c_in_seq = 0, seq_len = 0, bos = 0;
  {
    int base = 0;
    for (int b = 0; b < p.N; ++b) {
      const int s0 = p.cu_seqlens[b], len = p.cu_seqlens[b + 1] - s0;
      const int nc = (len + kC - 1) / kC;
      if (ci < base + nc) {
        seq = b;
        bos = s0;
        seq_len = len;
        c_in_seq = ci - base;
        tok0 = s0 + c_in_seq * kC;
        valid = min(kC, len - c_in_seq * kC);
        break;
      }
      base += nc;
    }
  }
  const int64_t img = int64_t(h) * p.C_total + ci;
  auto stamp = [&](int k) {
    if (p.dbg != nullptr && tid == 0 && h < 2) {
      unsigned long long t;
      asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
      p.dbg[(int64_t(h) * p.C_total + ci) * 16 + k] = (long long)t;
    }
  };
  stamp(0);
  const uint32_t bar = smem_u32(&s.bar);
  unsigned char* raw = s.a.raw;
  if (tid == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(bar) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar),
                 "r"(kRawBytes)
                 : "memory");
#pragma unroll
    for (int X = 0; X < 3; ++X)
      tma2d(smem_u32(raw + X * kRawBox), &p.tm_x, X * H * kD + h * kD, tok0 - 3, bar);
    tma2d(smem_u32(raw + kOffRawG), &p.tm_g, h * kD, tok0, bar);
  }
  if (tid < kC) {
    float b = 0.0f;
    if (tid < valid) {
      b = sigmoid_tanh(__bfloat162float(p.beta[int64_t(tok0 + tid) * p.ldb + h]));
      p.beta_out[int64_t(h) * p.T + tok0 + tid] = __float2bfloat16(b);
    }
    s.beta[tid] = b;
  }

  // ---- short conv (v, q, k): thread (tg, cg) = tokens 4 tg.., channels 8 cg.. ----
  const int t_first = 4 * tg;
  int64_t line = 0;
  bool load_init = false, write_state = false;
  if (c_in_seq == 0) {
    line = p.cache_indices[int64_t(seq) * p.ci_s0];
    const bool null_line = p.null_block_id >= 0 && line == p.null_block_id;
    load_init = !null_line && p.has_init[seq];
    write_state = !null_line && tg == 0;
  }
  const bool hist_from_state = c_in_seq == 0 && tg == 0;
  for (int e = tid; e < 3 * kD; e += kThreads) {
    const int X = e >> 7, c = e & (kD - 1);
    const float4 wv = __ldg(reinterpret_cast<float4 const*>(p.conv_w + (int64_t(X) * HD + h * kD + c) * 4));
    s.wc[X][0][c] = wv.x;
    s.wc[X][1][c] = wv.y;
    s.wc[X][2][c] = wv.z;
    s.wc[X][3][c] = wv.w;
  }
  if (tid < kD) s.dt[tid] = p.dt_bias[h * kD + tid];
  const float amp = expf(p.a_log[h]);
  __syncthreads();  // barrier init, weights visible
  k31sg::mbar_wait(bar, 0u, "flashkda_prep64_tc", __LINE__);
  stamp(15);
#pragma unroll 1
  for (int xi = 0; xi < 3; ++xi) {
    const int X = xi == 0 ? 2 : xi - 1;  // v (to its image), then q, k
    const int ch0 = X * HD + h * kD + 8 * cg;
    unsigned char const* rawx = raw + X * kRawBox + 16 * cg;
    float w[4][8];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const float4* wp = reinterpret_cast<const float4*>(&s.wc[X][k][8 * cg]);
      const float4 a = wp[0], b = wp[1];
      w[k][0] = a.x;
      w[k][1] = a.y;
      w[k][2] = a.z;
      w[k][3] = a.w;
      w[k][4] = b.x;
      w[k][5] = b.y;
      w[k][6] = b.z;
      w[k][7] = b.w;
    }
    float hist[3][8];
#pragma unroll
    for (int k = 0; k < 3; ++k) unpack8(*reinterpret_cast<uint4 const*>(rawx + (t_first + k) * 256), hist[k]);
    if (hist_from_state) {
      float* st = p.conv_state + line * p.st_s0 + int64_t(ch0) * p.st_s1;
#pragma unroll
      for (int k = 0; k < 3; ++k)
#pragma unroll
        for (int i = 0; i < 8; ++i) hist[k][i] = load_init ? st[i * p.st_s1 + k * p.st_s2] : 0.0f;
      if (write_state) {
        // new state: the last 3 inputs of the sequence (shifted in after the old
        // state for sequences shorter than 3); this thread read the old state above.
#pragma unroll
        for (int j = 0; j < 3; ++j) {
          float ns[8];
          const int src = seq_len - 3 + j;
          if (src >= 0) {
            unpack8(ldg16(p.x + int64_t(bos + src) * p.ldx + ch0), ns);
          } else {
            const int k = j + seq_len;  // 0..2; selects keep hist in registers
#pragma unroll
            for (int i = 0; i < 8; ++i) ns[i] = k == 0 ? hist[0][i] : (k == 1 ? hist[1][i] : hist[2][i]);
          }
#pragma unroll
          for (int i = 0; i < 8; ++i) st[i * p.st_s1 + j * p.st_s2] = ns[i];
        }
      }
    }
    uint4 ov[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float cur[8], o[8];
      unpack8(*reinterpret_cast<uint4 const*>(rawx + (t_first + j + 3) * 256), cur);
      conv_row(hist[0], hist[1], hist[2], cur, w, o);
      ov[j] = t_first + j < valid ? pack8(o) : make_uint4(0, 0, 0, 0);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        hist[0][i] = hist[1][i];
        hist[1][i] = hist[2][i];
        hist[2][i] = cur[i];
      }
    }
    if (X == 2) {
      // v image: the warp's 8 tokens (= t-group warp) x 128 channels through a
      // shared tile, transposed by ldmatrix: lane gets tokens 2 (l % 4) + {0, 1}
      // of channel l / 4 of each 8x8 tile -> 32-bit stores, 128 B per tile row.
      __nv_bfloat16* vt = s.c.vs[warp];
#pragma unroll
      for (int j = 0; j < 4; ++j)
        *reinterpret_cast<uint4*>(vt + (4 * (tg & 1) + j) * kPitch + 8 * cg) = ov[j];
      __syncwarp();
      __nv_bfloat16* vb = p.v_img + img * (kC * kD) + warp * (kD * 8) + 2 * (lane & 3);
#pragma unroll
      for (int m4 = 0; m4 < 4; ++m4) {
        uint32_t r[4];
        ldsm_x4_t(smem_u32(vt + (lane & 7) * kPitch + 8 * (4 * m4 + (lane >> 3))), r);
#pragma unroll
        for (int q = 0; q < 4; ++q) stg32(vb + (8 * (4 * m4 + q) + (lane >> 2)) * 8, r[q]);
      }
    } else {
      // L2 norm (16 lanes per token) -> bf16(x * rsqrt(sum x^2 + 1e-6))
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        float f[8];
        unpack8(ov[j], f);
        float ss = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) ss += f[i] * f[i];
#pragma unroll
        for (int o = 8; o >= 1; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        const float r = rsqrtf(ss + 1e-6f);
#pragma unroll
        for (int i = 0; i < 8; ++i) f[i] *= r;
        *reinterpret_cast<uint4*>(&s.qkn[X][t_first + j][8 * cg]) = pack8(f);
      }
    }
  }
  __syncthreads();  // raw q|k|v reads and the v tiles done
  stamp(1);

  // ---- gate activation + chunk-relative cumsum ----------------------------------
  {
    float dt[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) dt[i] = s.dt[8 * cg + i];
    float run[8] = {};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float gv[8];
      unpack8(*reinterpret_cast<uint4 const*>(raw + kOffRawG + (t_first + j) * 256 + 16 * cg), gv);
      const bool okj = t_first + j < valid;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        run[i] += okj ? p.gate_scale * sigmoid_tanh(amp * (gv[i] + dt[i])) : 0.0f;
      }
      float4* gp = reinterpret_cast<float4*>(&s.c.g[t_first + j][8 * cg]);
      gp[0] = make_float4(run[0], run[1], run[2], run[3]);
      gp[1] = make_float4(run[4], run[5], run[6], run[7]);
    }
    float4* tp = reinterpret_cast<float4*>(&s.tot[tg][8 * cg]);
    tp[0] = make_float4(run[0], run[1], run[2], run[3]);
    tp[1] = make_float4(run[4], run[5], run[6], run[7]);
  }
  __syncthreads();
  stamp(2);
  {
    float pre[8] = {};
    for (int t2 = 0; t2 < tg; ++t2) {
      const float4* tp = reinterpret_cast<const float4*>(&s.tot[t2][8 * cg]);
      const float4 a = tp[0], b = tp[1];
      pre[0] += a.x;
      pre[1] += a.y;
      pre[2] += a.z;
      pre[3] += a.w;
      pre[4] += b.x;
      pre[5] += b.y;
      pre[6] += b.z;
      pre[7] += b.w;
    }
    const int last = valid - 1;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int t = t_first + j;
      float4* gp = reinterpret_cast<float4*>(&s.c.g[t][8 * cg]);
      float4 a = gp[0], b = gp[1];
      a.x += pre[0];
      a.y += pre[1];
      a.z += pre[2];
      a.w += pre[3];
      b.x += pre[4];
      b.y += pre[5];
      b.z += pre[6];
      b.w += pre[7];
      gp[0] = a;
      gp[1] = b;
      if ((t & 15) == 0) {
        float4* rp = reinterpret_cast<float4*>(&s.gref[t >> 4][8 * cg]);
        rp[0] = a;
        rp[1] = b;
      }
      if (t == last) {
        float4* rp = reinterpret_cast<float4*>(&s.gT[8 * cg]);
        rp[0] = a;
        rp[1] = b;
      }
    }
  }
  __syncthreads();
  stamp(3);

  // ---- images and MMA operands: thread (tl, cgi) = tokens tl + 16 j, channels 8 cgi.. ----
  {
    const int tl = tid & 15, cgi = tid >> 4;
    // block-pair factors bf16(exp2(gc_rI - gc_rJ)), I > J
    for (int e = tid; e < 6 * (kD / 2); e += kThreads) {
      const int pr = e >> 6, d = 2 * (e & 63);
      const int I = pr == 0 ? 1 : (pr < 3 ? 2 : 3);
      const int J = pr - I * (I - 1) / 2;
      s.pf[pr][e & 63] = __floats2bfloat162_rn(ex2(s.gref[I][d] - s.gref[J][d]),
                                               ex2(s.gref[I][d + 1] - s.gref[J][d + 1]));
    }
    uint32_t sc2;
    {
      const __nv_bfloat162 sc = __floats2bfloat162_rn(p.scale, p.scale);
      sc2 = *reinterpret_cast<const uint32_t*>(&sc);
    }
    float gT[8];
    {
      const float4* gp = reinterpret_cast<const float4*>(&s.gT[8 * cgi]);
      const float4 a = gp[0], b = gp[1];
      gT[0] = a.x; gT[1] = a.y; gT[2] = a.z; gT[3] = a.w;
      gT[4] = b.x; gT[5] = b.y; gT[6] = b.z; gT[7] = b.w;
    }
    __nv_bfloat16* kdq = p.kdq + img * (2 * kC * kD) + (cgi * 128) * 8;
    __nv_bfloat16* krt = s.a.op.krs[warp];
    __nv_bfloat16* krb = p.kr + img * (kC * kD) + 2 * (lane & 3);
#pragma unroll 1
    for (int j = 0; j < 4; ++j) {
      const int t = tl + 16 * j;
      const bool ok = t < valid;
      const uint4 knv = *reinterpret_cast<uint4 const*>(&s.qkn[1][t][8 * cgi]);
      const uint4 qnv = *reinterpret_cast<uint4 const*>(&s.qkn[0][t][8 * cgi]);
      const uint32_t kn2[4] = {knv.x, knv.y, knv.z, knv.w};
      const uint32_t qn2[4] = {qnv.x, qnv.y, qnv.z, qnv.w};
      float gc[8], gr[8];
      {
        const float4* gp = reinterpret_cast<const float4*>(&s.c.g[t][8 * cgi]);
        const float4 a = gp[0], b = gp[1];
        gc[0] = a.x; gc[1] = a.y; gc[2] = a.z; gc[3] = a.w;
        gc[4] = b.x; gc[5] = b.y; gc[6] = b.z; gc[7] = b.w;
        const float4* rp = reinterpret_cast<const float4*>(&s.gref[j][8 * cgi]);
        const float4 c = rp[0], d = rp[1];
        gr[0] = c.x; gr[1] = c.y; gr[2] = c.z; gr[3] = c.w;
        gr[4] = d.x; gr[5] = d.y; gr[6] = d.z; gr[7] = d.w;
      }
      uint32_t kd[4], qd[4], kr[4], rk[4], rq[4], ck[4];
#pragma unroll
      for (int m = 0; m < 4; ++m) {
        const float d0 = gc[2 * m] - gr[2 * m], d1 = gc[2 * m + 1] - gr[2 * m + 1];
        const uint32_t e = pack2(ex2(gc[2 * m]), ex2(gc[2 * m + 1]));
        const uint32_t er = pack2(ex2(gT[2 * m] - gc[2 * m]), ex2(gT[2 * m + 1] - gc[2 * m + 1]));
        const uint32_t fr = pack2(ex2(d0), ex2(d1));
        const uint32_t ff = pack2(ex2(-d0), ex2(-d1));
        const uint32_t k2 = ok ? kn2[m] : 0u, q2 = ok ? qn2[m] : 0u;
        kd[m] = hmul2(k2, e);
        qd[m] = hmul2(hmul2(q2, e), sc2);
        kr[m] = hmul2(k2, er);
        rk[m] = hmul2(k2, fr);
        rq[m] = hmul2(hmul2(q2, fr), sc2);
        ck[m] = hmul2(k2, ff);
      }
      *reinterpret_cast<uint4*>(kdq + t * 8) = make_uint4(kd[0], kd[1], kd[2], kd[3]);
      *reinterpret_cast<uint4*>(kdq + (64 + t) * 8) = make_uint4(qd[0], qd[1], qd[2], qd[3]);
      *reinterpret_cast<uint4*>(&s.a.op.rk[t * kPitch + 8 * cgi]) = make_uint4(rk[0], rk[1], rk[2], rk[3]);
      *reinterpret_cast<uint4*>(&s.a.op.rq[t * kPitch + 8 * cgi]) = make_uint4(rq[0], rq[1], rq[2], rq[3]);
      *reinterpret_cast<uint4*>(&s.a.op.ck[t * kPitch + 8 * cgi]) = make_uint4(ck[0], ck[1], ck[2], ck[3]);
      // kr image: the warp's 16 tokens x 16 channels, transposed through a shared tile
      *reinterpret_cast<uint4*>(krt + tl * kKrPitch + 8 * (cgi & 1)) = make_uint4(kr[0], kr[1], kr[2], kr[3]);
      __syncwarp();
      {
        // matrices: (tokens 0-7 | 8-15) x (channels 0-7 | 8-15) of the tile
        uint32_t r[4];
        const int mi = lane >> 3;
        ldsm_x4_t(smem_u32(krt + (8 * (mi & 1) + (lane & 7)) * kKrPitch + 8 * (mi >> 1)), r);
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const int tgrp = 2 * j + (q & 1), ch = 16 * warp + 8 * (q >> 1) + (lane >> 2);
          stg32(krb + (tgrp * kD + ch) * 8, r[q]);
        }
      }
      __syncwarp();
    }
    if (tid < kD) p.gt[img * kD + tid] = valid > 0 ? exp2f(s.gT[tid]) : 1.0f;
  }
  __syncthreads();
  stamp(4);

  // ---- A, Mq block products (mma.sync) ------------------------------------------
  {
    // (row block I, first n-tile, n-tiles) per warp: 20 n-tiles of 8 columns
    const int I = warp < 3 ? 3 : (warp < 5 ? 2 : (warp < 7 ? 1 : 0));
    const int n0 = warp < 3 ? 3 * warp : (warp < 5 ? 3 * (warp - 3) : (warp < 7 ? 2 * (warp - 5) : 0));
    const int nn = (warp == 2 || warp >= 5) ? 2 : 3;
    const int g = lane >> 2, tq = lane & 3;
    const uint32_t a_row = 16 * I + (lane & 15);
    const uint32_t a_col = (lane >> 4) * 8;
    const uint32_t rk_base = smem_u32(s.a.op.rk) + (a_row * kPitch + a_col) * 2;
    const uint32_t rq_base = smem_u32(s.a.op.rq) + (a_row * kPitch + a_col) * 2;
    __nv_bfloat16* mqk = p.mqk + img * (kC * kC);
    for (int n = n0; n < n0 + nn; ++n) {
      const int J = n >> 1;
      const bool scale_b = J < I;
      uint32_t const* pf = reinterpret_cast<uint32_t const*>(s.pf[scale_b ? I * (I - 1) / 2 + J : 0]);
      float ca[4] = {}, cm[4] = {};
      const uint32_t b_base = smem_u32(s.a.op.ck) + ((8 * n + (lane & 7)) * kPitch + (lane >> 3) * 8) * 2;
#pragma unroll
      for (int ks = 0; ks < 8; ks += 2) {
        uint32_t b[4], a[4], q[4];
        ldsm_x4(b_base + ks * 32, b);
        if (scale_b) {
          b[0] = hmul2(b[0], pf[8 * ks + tq]);
          b[1] = hmul2(b[1], pf[8 * ks + 4 + tq]);
          b[2] = hmul2(b[2], pf[8 * ks + 8 + tq]);
          b[3] = hmul2(b[3], pf[8 * ks + 12 + tq]);
        }
        ldsm_x4(rk_base + ks * 32, a);
        ldsm_x4(rq_base + ks * 32, q);
        mma16816(ca, a, b[0], b[1]);
        mma16816(cm, q, b[0], b[1]);
        ldsm_x4(rk_base + (ks + 1) * 32, a);
        ldsm_x4(rq_base + (ks + 1) * 32, q);
        mma16816(ca, a, b[2], b[3]);
        mma16816(cm, q, b[2], b[3]);
      }
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int i = 16 * I + g + 8 * hh;
        const int j = 8 * n + 2 * tq;
        const float bi = s.beta[i];
        s.c.l[i][j] = j < i ? ca[2 * hh] * bi : 0.0f;
        s.c.l[i][j + 1] = j + 1 < i ? ca[2 * hh + 1] * bi : 0.0f;
        const bool iv = i < valid;
        const float m0 = (iv && j <= i && j < valid) ? cm[2 * hh] : 0.0f;
        const float m1 = (iv && j + 1 <= i && j + 1 < valid) ? cm[2 * hh + 1] : 0.0f;
        stg32(mqk + ((j >> 3) * 64 + i) * 8 + (j & 7), pack2(m0, m1));
      }
    }
  }
  __syncthreads();
  stamp(5);

  // ---- INV = (I + L)^-1 -------------------------------------------------------------
  auto& X = s.a.inv.x;
  auto& L = s.c.l;
  if (tid < kC) {
    // 16x16 diagonal blocks: FP32 forward substitution, one column per thread
    const int I = tid >> 4, jj = tid & 15, o = 16 * I;
    float x[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      float acc = i == jj ? 1.0f : 0.0f;
#pragma unroll
      for (int k = 0; k < i; ++k) acc -= L[o + i][o + k] * x[k];
      x[i] = acc;
    }
#pragma unroll
    for (int i = 0; i < 16; ++i) X[o + i][o + jj] = x[i];
  }
  __syncthreads();
  stamp(6);
  if (warp < 3) {
    // block column J = warp: X[I][J] = -X[I][I] (sum_{k=J}^{I-1} L[I][k] X[k][J]),
    // I = J+1..3, on BF16 tensor-core products with FP32 accumulation
    const int J = warp, g = lane >> 2, tq = lane & 3;
    float (*Tt)[20] = s.a.inv.t[warp];
    for (int I = J + 1; I < 4; ++I) {
      float tacc[2][4] = {};
      for (int k = J; k < I; ++k) {
        uint32_t a[4];
        const int r0 = 16 * I + g, c0 = 16 * k + 2 * tq;
        a[0] = pack2(L[r0][c0], L[r0][c0 + 1]);
        a[1] = pack2(L[r0 + 8][c0], L[r0 + 8][c0 + 1]);
        a[2] = pack2(L[r0][c0 + 8], L[r0][c0 + 9]);
        a[3] = pack2(L[r0 + 8][c0 + 8], L[r0 + 8][c0 + 9]);
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
          const int col = 16 * J + 8 * nt + g, kr0 = 16 * k + 2 * tq;
          const uint32_t b0 = pack2(X[kr0][col], X[kr0 + 1][col]);
          const uint32_t b1 = pack2(X[kr0 + 8][col], X[kr0 + 9][col]);
          mma16816(tacc[nt], a, b0, b1);
        }
      }
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        Tt[g][8 * nt + 2 * tq] = tacc[nt][0];
        Tt[g][8 * nt + 2 * tq + 1] = tacc[nt][1];
        Tt[g + 8][8 * nt + 2 * tq] = tacc[nt][2];
        Tt[g + 8][8 * nt + 2 * tq + 1] = tacc[nt][3];
      }
      __syncwarp();
      uint32_t a[4];
      {
        const int r0 = 16 * I + g, c0 = 16 * I + 2 * tq;
        a[0] = pack2(X[r0][c0], X[r0][c0 + 1]);
        a[1] = pack2(X[r0 + 8][c0], X[r0 + 8][c0 + 1]);
        a[2] = pack2(X[r0][c0 + 8], X[r0][c0 + 9]);
        a[3] = pack2(X[r0 + 8][c0 + 8], X[r0 + 8][c0 + 9]);
      }
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        float xacc[4] = {};
        const int col = 8 * nt + g;
        const uint32_t b0 = pack2(Tt[2 * tq][col], Tt[2 * tq + 1][col]);
        const uint32_t b1 = pack2(Tt[2 * tq + 8][col], Tt[2 * tq + 9][col]);
        mma16816(xacc, a, b0, b1);
        const int r0 = 16 * I + g, c0 = 16 * J + 8 * nt + 2 * tq;
        X[r0][c0] = -xacc[0];
        X[r0][c0 + 1] = -xacc[1];
        X[r0 + 8][c0] = -xacc[2];
        X[r0 + 8][c0 + 1] = -xacc[3];
      }
      __syncwarp();
    }
  }
  __syncthreads();
  stamp(12);

  // ---- INV image, Mqk above the diagonal blocks ----------------------------------
  {
    __nv_bfloat16* inv = p.inv + img * (kC * kC);
    __nv_bfloat16* mqk = p.mqk + img * (kC * kC);
    for (int uidx = tid; uidx < 512; uidx += kThreads) {
      const int jg = uidx >> 6, i = uidx & 63;
      float v8[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int j = 8 * jg + e;
        v8[e] = (j <= i && i < valid && j < valid) ? X[i][j] : 0.0f;
      }
      *reinterpret_cast<uint4*>(inv + (jg * 64 + i) * 8) = pack8(v8);
      if (8 * jg >= 16 * ((i >> 4) + 1))
        *reinterpret_cast<uint4*>(mqk + (jg * 64 + i) * 8) = make_uint4(0, 0, 0, 0);
    }
  }
  stamp(14);
}



__device__ __forceinline__ void ffma2p(float& d0, float& d1, float a0, float a1, float b0, float b1,
                                       float c0, float c1) {
  asm("{\n .reg .b64 a, b, c, d;\n mov.b64 a, {%2, %3};\n mov.b64 b, {%4, %5};\n"
      " mov.b64 c, {%6, %7};\n fma.rn.ftz.f32x2 d, a, b, c;\n mov.b64 {%0, %1}, d;\n}"
      : "=f"(d0), "=f"(d1)
      : "f"(a0), "f"(a1), "f"(b0), "f"(b1), "f"(c0), "f"(c1));
}
__device__ __forceinline__ void fadd2p(float& d0, float& d1, float a0, float a1, float b0, float b1) {
  asm("{\n .reg .b64 a, b, d;\n mov.b64 a, {%2, %3};\n mov.b64 b, {%4, %5};\n"
      " add.rn.ftz.f32x2 d, a, b;\n mov.b64 {%0, %1}, d;\n}"
      : "=f"(d0), "=f"(d1)
      : "f"(a0), "f"(a1), "f"(b0), "f"(b1));
}
__device__ __forceinline__ void fmul2p(float& d0, float& d1, float a0, float a1, float b0, float b1) {
  asm("{\n .reg .b64 a, b, d;\n mov.b64 a, {%2, %3};\n mov.b64 b, {%4, %5};\n"
      " mul.rn.ftz.f32x2 d, a, b;\n mov.b64 {%0, %1}, d;\n}"
      : "=f"(d0), "=f"(d1)
      : "f"(a0), "f"(a1), "f"(b0), "f"(b1));
}
__device__ __forceinline__ float tanh_ap(float x) {
  float th;
  asm("tanh.approx.f32 %0, %1;" : "=f"(th) : "f"(x));
  return th;
}

// kp64 (no conv, WY images for kc64 v3): post-conv rows in, kd / v as row-major
// MMA operands, INVb = bf16(INV * beta) as the A operand of [w | u].
struct SmemK {
  Smem s;
  alignas(128) __nv_bfloat16 kv[2][kC][kPitch];   // kd (bf16 kn * e) | post-conv v rows
  alignas(128) __nv_bfloat16 invb[kC][kC + 8];
};

__global__ void __launch_bounds__(kThreads, 1) kp64_kernel(const __grid_constant__ Params p) {
  k31sg::begin_call((long long)(p.T));  // timeout lines: call index and size
  extern __shared__ __align__(128) unsigned char smem_raw[];
  SmemK& sk = *reinterpret_cast<SmemK*>(smem_raw);
  Smem& s = sk.s;
  const int ci = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int tg = tid >> 4, cg = tid & 15;
  const int H = p.H;
  const int HD = H * kD;

  int seq = 0, tok0 = 0, valid = 0, c_in_seq = 0, seq_len = 0, bos = 0;
  {
    int base = 0;
    for (int b = 0; b < p.N; ++b) {
      const int s0 = p.cu_seqlens[b], len = p.cu_seqlens[b + 1] - s0;
      const int nc = (len + kC - 1) / kC;
      if (ci < base + nc) {
        seq = b;
        bos = s0;
        seq_len = len;
        c_in_seq = ci - base;
        tok0 = s0 + c_in_seq * kC;
        valid = min(kC, len - c_in_seq * kC);
        break;
      }
      base += nc;
    }
  }
  const int64_t img = int64_t(h) * p.C_total + ci;
  auto stamp = [&](int k) {
    if (p.dbg != nullptr && tid == 0 && h < 2) {
      unsigned long long t;
      asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
      p.dbg[(int64_t(h) * p.C_total + ci) * 16 + k] = (long long)t;
    }
  };
  stamp(0);
  const uint32_t bar = smem_u32(&s.bar);
  unsigned char* raw = s.a.raw;
  if (tid == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(bar) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar),
                 "r"(4 * kC * kD * 2)
                 : "memory");
#pragma unroll
    for (int X = 0; X < 3; ++X)
      tma2d(smem_u32(raw + X * kRawBox), &p.tm_x, X * H * kD + h * kD, tok0, bar);
    tma2d(smem_u32(raw + kOffRawG), &p.tm_g, h * kD, tok0, bar);
  }
  if (tid < kC) {
    float b = 0.0f;
    if (tid < valid) {
      b = bfr(sigmoid_tanh(__bfloat162float(p.beta[int64_t(tok0 + tid) * p.ldb + h])));
    }
    s.beta[tid] = b;
  }

  // ---- post-conv rows: q, k -> bf16 l2-normalized rows, v rows -> kv[1] ----------
  const int t_first = 4 * tg;
  if (tid < kD) s.dt[tid] = p.dt_bias[h * kD + tid];
  const float amp = expf(p.a_log[h]);
  __syncthreads();  // barrier init, dt visible
  k31sg::mbar_wait(bar, 0u, "flashkda_prep64_tc", __LINE__);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int t = t_first + j;
#pragma unroll
    for (int X = 0; X < 2; ++X) {
      float f[8];
      unpack8(*reinterpret_cast<uint4 const*>(raw + X * kRawBox + t * 256 + 16 * cg), f);
      float s0 = 0.0f, s1 = 0.0f;
#pragma unroll
      for (int m = 0; m < 4; ++m) ffma2p(s0, s1, f[2 * m], f[2 * m + 1], f[2 * m], f[2 * m + 1], s0, s1);
      float ss = s0 + s1;
#pragma unroll
      for (int o = 8; o >= 1; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
      const float r = rsqrtf(ss + 1e-6f);
      uint32_t o4[4];
#pragma unroll
      for (int m = 0; m < 4; ++m) {
        float a0, a1;
        fmul2p(a0, a1, f[2 * m], f[2 * m + 1], r, r);
        o4[m] = pack2(a0, a1);
      }
      *reinterpret_cast<uint4*>(&s.qkn[X][t][8 * cg]) = make_uint4(o4[0], o4[1], o4[2], o4[3]);
    }
    *reinterpret_cast<uint4*>(&sk.kv[1][t][8 * cg]) =
        *reinterpret_cast<uint4 const*>(raw + 2 * kRawBox + t * 256 + 16 * cg);
  }
  __syncthreads();  // raw q|k|v reads done
  stamp(1);

  // ---- gate activation + chunk-relative cumsum ----------------------------------
  {
    const float ha = 0.5f * amp, hg = 0.5f * p.gate_scale;
    float dta[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) dta[i] = ha * s.dt[8 * cg + i];
    float run[8] = {};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float gv[8];
      unpack8(*reinterpret_cast<uint4 const*>(raw + kOffRawG + (t_first + j) * 256 + 16 * cg), gv);
      if (t_first + j < valid) {
#pragma unroll
        for (int m = 0; m < 4; ++m) {
          float x0, x1, y0, y1;
          ffma2p(x0, x1, gv[2 * m], gv[2 * m + 1], ha, ha, dta[2 * m], dta[2 * m + 1]);
          ffma2p(y0, y1, tanh_ap(x0), tanh_ap(x1), hg, hg, hg, hg);
          fadd2p(run[2 * m], run[2 * m + 1], run[2 * m], run[2 * m + 1], y0, y1);
        }
      }
      float4* gp = reinterpret_cast<float4*>(&s.c.g[t_first + j][8 * cg]);
      gp[0] = make_float4(run[0], run[1], run[2], run[3]);
      gp[1] = make_float4(run[4], run[5], run[6], run[7]);
    }
    float4* tp = reinterpret_cast<float4*>(&s.tot[tg][8 * cg]);
    tp[0] = make_float4(run[0], run[1], run[2], run[3]);
    tp[1] = make_float4(run[4], run[5], run[6], run[7]);
  }
  __syncthreads();
  stamp(2);
  {
    float pre[8] = {};
    for (int t2 = 0; t2 < tg; ++t2) {
      const float4* tp = reinterpret_cast<const float4*>(&s.tot[t2][8 * cg]);
      const float4 a = tp[0], b = tp[1];
      fadd2p(pre[0], pre[1], pre[0], pre[1], a.x, a.y);
      fadd2p(pre[2], pre[3], pre[2], pre[3], a.z, a.w);
      fadd2p(pre[4], pre[5], pre[4], pre[5], b.x, b.y);
      fadd2p(pre[6], pre[7], pre[6], pre[7], b.z, b.w);
    }
    const int last = valid - 1;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int t = t_first + j;
      float4* gp = reinterpret_cast<float4*>(&s.c.g[t][8 * cg]);
      float4 a = gp[0], b = gp[1];
      fadd2p(a.x, a.y, a.x, a.y, pre[0], pre[1]);
      fadd2p(a.z, a.w, a.z, a.w, pre[2], pre[3]);
      fadd2p(b.x, b.y, b.x, b.y, pre[4], pre[5]);
      fadd2p(b.z, b.w, b.z, b.w, pre[6], pre[7]);
      gp[0] = a;
      gp[1] = b;
      if ((t & 15) == 0) {
        float4* rp = reinterpret_cast<float4*>(&s.gref[t >> 4][8 * cg]);
        rp[0] = a;
        rp[1] = b;
      }
      if (t == last) {
        float4* rp = reinterpret_cast<float4*>(&s.gT[8 * cg]);
        rp[0] = a;
        rp[1] = b;
      }
    }
  }
  __syncthreads();
  stamp(3);

  // ---- images and MMA operands: thread (tl, cgi) = tokens tl + 16 j, channels 8 cgi.. ----
  {
    const int tl = tid & 15, cgi = tid >> 4;
    // block-pair factors bf16(exp2(gc_rI - gc_rJ)), I > J
    for (int e = tid; e < 6 * (kD / 2); e += kThreads) {
      const int pr = e >> 6, d = 2 * (e & 63);
      const int I = pr == 0 ? 1 : (pr < 3 ? 2 : 3);
      const int J = pr - I * (I - 1) / 2;
      s.pf[pr][e & 63] = __floats2bfloat162_rn(ex2(s.gref[I][d] - s.gref[J][d]),
                                               ex2(s.gref[I][d + 1] - s.gref[J][d + 1]));
    }
    uint32_t sc2;
    {
      const __nv_bfloat162 sc = __floats2bfloat162_rn(p.scale, p.scale);
      sc2 = *reinterpret_cast<const uint32_t*>(&sc);
    }
    float gT[8];
    {
      const float4* gp = reinterpret_cast<const float4*>(&s.gT[8 * cgi]);
      const float4 a = gp[0], b = gp[1];
      gT[0] = a.x; gT[1] = a.y; gT[2] = a.z; gT[3] = a.w;
      gT[4] = b.x; gT[5] = b.y; gT[6] = b.z; gT[7] = b.w;
    }
    __nv_bfloat16* kdq = p.kdq + img * (2 * kC * kD) + (cgi * 128) * 8;
    __nv_bfloat16* krt = s.a.op.krs[warp];
    __nv_bfloat16* krb = p.kr + img * (kC * kD) + 2 * (lane & 3);
#pragma unroll 1
    for (int j = 0; j < 4; ++j) {
      const int t = tl + 16 * j;
      const bool ok = t < valid;
      const uint4 knv = *reinterpret_cast<uint4 const*>(&s.qkn[1][t][8 * cgi]);
      const uint4 qnv = *reinterpret_cast<uint4 const*>(&s.qkn[0][t][8 * cgi]);
      const uint32_t kn2[4] = {knv.x, knv.y, knv.z, knv.w};
      const uint32_t qn2[4] = {qnv.x, qnv.y, qnv.z, qnv.w};
      float gc[8], gr[8];
      {
        const float4* gp = reinterpret_cast<const float4*>(&s.c.g[t][8 * cgi]);
        const float4 a = gp[0], b = gp[1];
        gc[0] = a.x; gc[1] = a.y; gc[2] = a.z; gc[3] = a.w;
        gc[4] = b.x; gc[5] = b.y; gc[6] = b.z; gc[7] = b.w;
        const float4* rp = reinterpret_cast<const float4*>(&s.gref[j][8 * cgi]);
        const float4 c = rp[0], d = rp[1];
        gr[0] = c.x; gr[1] = c.y; gr[2] = c.z; gr[3] = c.w;
        gr[4] = d.x; gr[5] = d.y; gr[6] = d.z; gr[7] = d.w;
      }
      uint32_t kd[4], qd[4], kr[4], rk[4], rq[4], ck[4];
#pragma unroll
      for (int m = 0; m < 4; ++m) {
        const float d0 = gc[2 * m] - gr[2 * m], d1 = gc[2 * m + 1] - gr[2 * m + 1];
        const uint32_t e = pack2(ex2(gc[2 * m]), ex2(gc[2 * m + 1]));
        const uint32_t er = pack2(ex2(gT[2 * m] - gc[2 * m]), ex2(gT[2 * m + 1] - gc[2 * m + 1]));
        const uint32_t fr = pack2(ex2(d0), ex2(d1));
        const uint32_t ff = pack2(ex2(-d0), ex2(-d1));
        const uint32_t k2 = ok ? kn2[m] : 0u, q2 = ok ? qn2[m] : 0u;
        kd[m] = hmul2(k2, e);
        qd[m] = hmul2(hmul2(q2, e), sc2);
        kr[m] = hmul2(k2, er);
        rk[m] = hmul2(k2, fr);
        rq[m] = hmul2(hmul2(q2, fr), sc2);
        ck[m] = hmul2(k2, ff);
      }
      *reinterpret_cast<uint4*>(&sk.kv[0][t][8 * cgi]) = make_uint4(kd[0], kd[1], kd[2], kd[3]);
      *reinterpret_cast<uint4*>(kdq + t * 8) = make_uint4(qd[0], qd[1], qd[2], qd[3]);
      *reinterpret_cast<uint4*>(&s.a.op.rk[t * kPitch + 8 * cgi]) = make_uint4(rk[0], rk[1], rk[2], rk[3]);
      *reinterpret_cast<uint4*>(&s.a.op.rq[t * kPitch + 8 * cgi]) = make_uint4(rq[0], rq[1], rq[2], rq[3]);
      *reinterpret_cast<uint4*>(&s.a.op.ck[t * kPitch + 8 * cgi]) = make_uint4(ck[0], ck[1], ck[2], ck[3]);
      // kr image: the warp's 16 tokens x 16 channels, transposed through a shared tile
      *reinterpret_cast<uint4*>(krt + tl * kKrPitch + 8 * (cgi & 1)) = make_uint4(kr[0], kr[1], kr[2], kr[3]);
      __syncwarp();
      {
        // matrices: (tokens 0-7 | 8-15) x (channels 0-7 | 8-15) of the tile
        uint32_t r[4];
        const int mi = lane >> 3;
        ldsm_x4_t(smem_u32(krt + (8 * (mi & 1) + (lane & 7)) * kKrPitch + 8 * (mi >> 1)), r);
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const int tgrp = 2 * j + (q & 1), ch = 16 * warp + 8 * (q >> 1) + (lane >> 2);
          stg32(krb + (tgrp * kD + ch) * 8, r[q]);
        }
      }
      __syncwarp();
    }
    if (tid < kD) p.gt[img * kD + tid] = valid > 0 ? exp2f(s.gT[tid]) : 1.0f;
  }
  __syncthreads();
  stamp(4);

  // ---- A, Mq block products (mma.sync) ------------------------------------------
  {
    // (row block I, first n-tile, n-tiles) per warp: 20 n-tiles of 8 columns
    const int I = warp < 3 ? 3 : (warp < 5 ? 2 : (warp < 7 ? 1 : 0));
    const int n0 = warp < 3 ? 3 * warp : (warp < 5 ? 3 * (warp - 3) : (warp < 7 ? 2 * (warp - 5) : 0));
    const int nn = (warp == 2 || warp >= 5) ? 2 : 3;
    const int g = lane >> 2, tq = lane & 3;
    const uint32_t a_row = 16 * I + (lane & 15);
    const uint32_t a_col = (lane >> 4) * 8;
    const uint32_t rk_base = smem_u32(s.a.op.rk) + (a_row * kPitch + a_col) * 2;
    const uint32_t rq_base = smem_u32(s.a.op.rq) + (a_row * kPitch + a_col) * 2;
    __nv_bfloat16* mqk = p.mqk + img * (kC * kC);
    for (int n = n0; n < n0 + nn; ++n) {
      const int J = n >> 1;
      const bool scale_b = J < I;
      uint32_t const* pf = reinterpret_cast<uint32_t const*>(s.pf[scale_b ? I * (I - 1) / 2 + J : 0]);
      float ca[4] = {}, cm[4] = {};
      const uint32_t b_base = smem_u32(s.a.op.ck) + ((8 * n + (lane & 7)) * kPitch + (lane >> 3) * 8) * 2;
#pragma unroll
      for (int ks = 0; ks < 8; ks += 2) {
        uint32_t b[4], a[4], q[4];
        ldsm_x4(b_base + ks * 32, b);
        if (scale_b) {
          b[0] = hmul2(b[0], pf[8 * ks + tq]);
          b[1] = hmul2(b[1], pf[8 * ks + 4 + tq]);
          b[2] = hmul2(b[2], pf[8 * ks + 8 + tq]);
          b[3] = hmul2(b[3], pf[8 * ks + 12 + tq]);
        }
        ldsm_x4(rk_base + ks * 32, a);
        ldsm_x4(rq_base + ks * 32, q);
        mma16816(ca, a, b[0], b[1]);
        mma16816(cm, q, b[0], b[1]);
        ldsm_x4(rk_base + (ks + 1) * 32, a);
        ldsm_x4(rq_base + (ks + 1) * 32, q);
        mma16816(ca, a, b[2], b[3]);
        mma16816(cm, q, b[2], b[3]);
      }
      if (J < I && valid == kC) {
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const int i = 16 * I + g + 8 * hh;
          const int j = 8 * n + 2 * tq;
          const float bi = s.beta[i];
          *reinterpret_cast<float2*>(&s.c.l[i][j]) = make_float2(ca[2 * hh] * bi, ca[2 * hh + 1] * bi);
          stg32(mqk + ((j >> 3) * 64 + i) * 8 + (j & 7), pack2(cm[2 * hh], cm[2 * hh + 1]));
        }
        continue;
      }
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int i = 16 * I + g + 8 * hh;
        const int j = 8 * n + 2 * tq;
        const float bi = s.beta[i];
        s.c.l[i][j] = j < i ? ca[2 * hh] * bi : 0.0f;
        s.c.l[i][j + 1] = j + 1 < i ? ca[2 * hh + 1] * bi : 0.0f;
        const bool iv = i < valid;
        const float m0 = (iv && j <= i && j < valid) ? cm[2 * hh] : 0.0f;
        const float m1 = (iv && j + 1 <= i && j + 1 < valid) ? cm[2 * hh + 1] : 0.0f;
        stg32(mqk + ((j >> 3) * 64 + i) * 8 + (j & 7), pack2(m0, m1));
      }
    }
  }
  __syncthreads();
  stamp(5);

  // ---- INV = (I + L)^-1 -------------------------------------------------------------
  auto& X = s.a.inv.x;
  auto& L = s.c.l;
  if (tid < kC) {
    // 16x16 diagonal blocks: FP32 forward substitution, one column per thread
    const int I = tid >> 4, jj = tid & 15, o = 16 * I;
    float x[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      float acc = i == jj ? 1.0f : 0.0f;
#pragma unroll
      for (int k = 0; k < i; ++k) acc -= L[o + i][o + k] * x[k];
      x[i] = acc;
    }
#pragma unroll
    for (int i = 0; i < 16; ++i) X[o + i][o + jj] = x[i];
  }
  __syncthreads();
  stamp(6);
  if (warp < 3) {
    // block column J = warp: X[I][J] = -X[I][I] (sum_{k=J}^{I-1} L[I][k] X[k][J]),
    // I = J+1..3, on BF16 tensor-core products with FP32 accumulation
    const int J = warp, g = lane >> 2, tq = lane & 3;
    float (*Tt)[20] = s.a.inv.t[warp];
    for (int I = J + 1; I < 4; ++I) {
      float tacc[2][4] = {};
      for (int k = J; k < I; ++k) {
        uint32_t a[4];
        const int r0 = 16 * I + g, c0 = 16 * k + 2 * tq;
        a[0] = pack2(L[r0][c0], L[r0][c0 + 1]);
        a[1] = pack2(L[r0 + 8][c0], L[r0 + 8][c0 + 1]);
        a[2] = pack2(L[r0][c0 + 8], L[r0][c0 + 9]);
        a[3] = pack2(L[r0 + 8][c0 + 8], L[r0 + 8][c0 + 9]);
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
          const int col = 16 * J + 8 * nt + g, kr0 = 16 * k + 2 * tq;
          const uint32_t b0 = pack2(X[kr0][col], X[kr0 + 1][col]);
          const uint32_t b1 = pack2(X[kr0 + 8][col], X[kr0 + 9][col]);
          mma16816(tacc[nt], a, b0, b1);
        }
      }
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        Tt[g][8 * nt + 2 * tq] = tacc[nt][0];
        Tt[g][8 * nt + 2 * tq + 1] = tacc[nt][1];
        Tt[g + 8][8 * nt + 2 * tq] = tacc[nt][2];
        Tt[g + 8][8 * nt + 2 * tq + 1] = tacc[nt][3];
      }
      __syncwarp();
      uint32_t a[4];
      {
        const int r0 = 16 * I + g, c0 = 16 * I + 2 * tq;
        a[0] = pack2(X[r0][c0], X[r0][c0 + 1]);
        a[1] = pack2(X[r0 + 8][c0], X[r0 + 8][c0 + 1]);
        a[2] = pack2(X[r0][c0 + 8], X[r0][c0 + 9]);
        a[3] = pack2(X[r0 + 8][c0 + 8], X[r0 + 8][c0 + 9]);
      }
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        float xacc[4] = {};
        const int col = 8 * nt + g;
        const uint32_t b0 = pack2(Tt[2 * tq][col], Tt[2 * tq + 1][col]);
        const uint32_t b1 = pack2(Tt[2 * tq + 8][col], Tt[2 * tq + 9][col]);
        mma16816(xacc, a, b0, b1);
        const int r0 = 16 * I + g, c0 = 16 * J + 8 * nt + 2 * tq;
        X[r0][c0] = -xacc[0];
        X[r0][c0 + 1] = -xacc[1];
        X[r0 + 8][c0] = -xacc[2];
        X[r0 + 8][c0 + 1] = -xacc[3];
      }
      __syncwarp();
    }
  }
  __syncthreads();
  stamp(12);

  // ---- Mqk above the diagonal blocks; INVb = bf16(INV_ij * beta_j) -------------------
  {
    __nv_bfloat16* mqk = p.mqk + img * (kC * kC);
    for (int uidx = tid; uidx < 512; uidx += kThreads) {
      const int jg = uidx >> 6, i = uidx & 63;
      if (8 * jg >= 16 * ((i >> 4) + 1))
        *reinterpret_cast<uint4*>(mqk + (jg * 64 + i) * 8) = make_uint4(0, 0, 0, 0);
    }
    {
      // beta_j = 0 for j >= valid (s.beta), rows i >= valid zero
      const int i = tid >> 2, j0 = 16 * (tid & 3);
      uint32_t pk[8] = {};
      if (j0 <= i) {
#pragma unroll
        for (int e = 0; e < 16; e += 2) {
          const int j = j0 + e;
          float v0, v1;
          fmul2p(v0, v1, X[i][j], X[i][j + 1], s.beta[j], s.beta[j + 1]);
          pk[e / 2] = pack2(v0, v1);
        }
      }
      uint4* dst = reinterpret_cast<uint4*>(&sk.invb[i][j0]);
      dst[0] = make_uint4(pk[0], pk[1], pk[2], pk[3]);
      dst[1] = make_uint4(pk[4], pk[5], pk[6], pk[7]);
    }
  }
  __syncthreads();
  stamp(13);
  // ---- [w | u] = INVb @ [kd | v]: warp w owns 32 columns (w < 4: kd, else v) ----
  {
    const int g = lane >> 2, tq = lane & 3;
    const int bsel = warp >> 2, n0 = 32 * (warp & 3);
    __nv_bfloat16* wst = reinterpret_cast<__nv_bfloat16*>(s.a.raw);
    __nv_bfloat16* ust = wst + kC * kD;
    if (bsel == 1) {
      // u^T [v = n0.. n0 + 32][t]: A = v^T from kv[1] [t][v] (ldmatrix.trans), B = INVb rows
      float au[2][8][4] = {};
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) {
        uint32_t a[2][4];
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
          const int q = lane >> 3;
          ldsm_x4_t(smem_u32(&sk.kv[1][16 * ks + (lane & 7) + 8 * (q >> 1)][n0 + 16 * mt + 8 * (q & 1)]),
                    a[mt]);
        }
#pragma unroll
        for (int np = 0; np < 4; ++np) {
          uint32_t b[4];
          ldsm_x4(smem_u32(&sk.invb[16 * np + (lane & 7) + 8 * (lane >> 4)][16 * ks + 8 * ((lane >> 3) & 1)]),
                  b);
#pragma unroll
          for (int mt = 0; mt < 2; ++mt) {
            mma16816(au[mt][2 * np], a[mt], b[0], b[1]);
            mma16816(au[mt][2 * np + 1], a[mt], b[2], b[3]);
          }
        }
      }
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < 8; ++nt)
#pragma unroll
          for (int hh = 0; hh < 2; ++hh) {
            const int v = n0 + 16 * mt + g + 8 * hh, t = 8 * nt + 2 * tq;
            *reinterpret_cast<uint32_t*>(ust + (nt * 128 + v) * 8 + 2 * tq) =
                pack2(au[mt][nt][2 * hh], au[mt][nt][2 * hh + 1]);
          }
    } else {
    float acc[4][4][4] = {};
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      uint32_t b[2][4];
#pragma unroll
      for (int np = 0; np < 2; ++np) {
        const int kr0 = 16 * ks + (lane & 7) + 8 * ((lane >> 3) & 1);
        const int nc = n0 + 16 * np + 8 * (lane >> 4);
        ldsm_x4_t(smem_u32(&sk.kv[0][kr0][nc]), b[np]);
      }
#pragma unroll
      for (int mt = 0; mt < 4; ++mt) {
        uint32_t a[4];
        ldsm_x4(smem_u32(&sk.invb[16 * mt + (lane & 15)][16 * ks + 8 * (lane >> 4)]), a);
#pragma unroll
        for (int np = 0; np < 2; ++np) {
          mma16816(acc[mt][2 * np], a, b[np][0], b[np][1]);
          mma16816(acc[mt][2 * np + 1], a, b[np][2], b[np][3]);
        }
      }
    }
#pragma unroll
    for (int mt = 0; mt < 4; ++mt) {
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const int i = 16 * mt + g + 8 * hh, n = n0 + 8 * nt + 2 * tq;
          const float x0 = acc[mt][nt][2 * hh], x1 = acc[mt][nt][2 * hh + 1];
          *reinterpret_cast<uint32_t*>(wst + ((n >> 3) * 64 + i) * 8 + (n & 7)) = pack2(x0, x1);
        }
      }
    }
  }
    }
  __syncthreads();
  {
    uint4 const* wst = reinterpret_cast<uint4 const*>(s.a.raw);
    uint4* wimg = reinterpret_cast<uint4*>(p.kdq + img * (2 * kC * kD));
    uint4* uimg = reinterpret_cast<uint4*>(p.v_img + img * (kC * kD));
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int e = tid + kThreads * r;  // 1024 x 16 B of w rows: kg = e / 64, t = e % 64
      wimg[(e >> 6) * 128 + 64 + (e & 63)] = wst[e];
      uimg[e] = wst[1024 + e];
    }
  }
  stamp(14);
}

}  // namespace prep64tc

// ---------------------------------------------------------------------------
// prep_chunk: the chunk-64 WY prepare as a device function of 256 threads (8
// warps, named barrier 3), the producer of the fused KDA core. It reads the
// post-conv q / k / v and raw gate rows of one (chunk, head) from shared memory
// and writes the kc64 v3 image stage in shared memory:
//   wq  [16 kg][128 rows: qd 0-63, w 64-127][8]     krm [8 g][192 rows: kr^T 0-127, Mqk 128-191][8]
//   u   [8 tg][128 v][8 t]                           gt  [128] fp32
// Prep shared memory (bytes, 139 KB): the input boxes alias the MMA operand
// buffers and the INV workspace; kd is written in place over kn; v stays in
// its input box (B operand of u^T = v^T INVb^T).
namespace kf64 {

constexpr int kC = 64, kD = 128, kPitch = 136;
constexpr int kRawQ = 0, kRawK = 16384, kRawG = 32768, kRawV = 52224;   // TMA boxes (64 x 256 B)
constexpr int kOpRk = 0, kOpRq = 17408, kOpCk = 34816;                  // [64][136] bf16
constexpr int kTot = 0;                                                  // [16][128] f32 (after l2norm)
constexpr int kInvX = 0, kInvT = 17408, kInvB = 21504;                   // f32 [64][68], [3][16][20], bf16 [64][72]
constexpr int kQkn = 68608;                                              // [2][64][136] bf16 (kn -> kd)
constexpr int kCg = kQkn + 2 * kC * kPitch * 2;                         // f32 [64][132] | L f32 [64][68]
constexpr int kPf = kCg + kC * (kD + 4) * 4;                            // bf16x2 [6][64]
constexpr int kDt = kPf + 6 * 64 * 4;
constexpr int kGref = kDt + kD * 4;
constexpr int kGT = kGref + 4 * kD * 4;
constexpr int kBeta = kGT + kD * 4;
constexpr int kPrepBytes = kBeta + kC * 4;                               // 142080

using prep64tc::ex2;
using prep64tc::ffma2p;
using prep64tc::fadd2p;
using prep64tc::fmul2p;
using prep64tc::tanh_ap;
using prep64tc::pack2;
using prep64tc::unpack8;
using prep64tc::hmul2;
using prep64tc::ldsm_x4;
using prep64tc::ldsm_x4_t;
using prep64tc::mma16816;
using prep64tc::sigmoid_tanh;
using prep64tc::bfr;

__device__ __forceinline__ void pbar() { asm volatile("bar.sync 3, 256;" ::: "memory"); }

// Caller: raw boxes landed, dt (kDt) loaded once per head. tid / warp / lane are
// prep-local (0..255 / 0..7). Ends with the stage written (caller fences + syncs).
__device__ __forceinline__ void prep_chunk(prep64tc::Params const& p, unsigned char* ps, int h,
                                           int tok0, int valid, int tid, int warp, int lane,
                                           float amp, unsigned char* stage) {
  __nv_bfloat16* wq = reinterpret_cast<__nv_bfloat16*>(stage);
  __nv_bfloat16* krm = reinterpret_cast<__nv_bfloat16*>(stage + 32768);
  __nv_bfloat16* uimg = reinterpret_cast<__nv_bfloat16*>(stage + 57344);
  float* gtp = reinterpret_cast<float*>(stage + 73728);
  float* beta = reinterpret_cast<float*>(ps + kBeta);
  float* gT_s = reinterpret_cast<float*>(ps + kGT);
  float (*gref)[kD] = reinterpret_cast<float (*)[kD]>(ps + kGref);
  float (*cg)[kD + 4] = reinterpret_cast<float (*)[kD + 4]>(ps + kCg);
  float (*L)[kC + 4] = reinterpret_cast<float (*)[kC + 4]>(ps + kCg);
  float (*tot)[kD] = reinterpret_cast<float (*)[kD]>(ps + kTot);
  __nv_bfloat16 (*qkn)[kC][kPitch] = reinterpret_cast<__nv_bfloat16 (*)[kC][kPitch]>(ps + kQkn);
  const int tg = tid >> 4, cg16 = tid & 15;
  const int t_first = 4 * tg;

  if (tid < kC) {
    float b = 0.0f;
    if (tid < valid) b = bfr(sigmoid_tanh(__bfloat162float(p.beta[int64_t(tok0 + tid) * p.ldb + h])));
    beta[tid] = b;
  }
  // ---- l2 norm of q, k rows (rows past valid -> 0) ----
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int t = t_first + j;
#pragma unroll
    for (int X = 0; X < 2; ++X) {
      float f[8];
      unpack8(*reinterpret_cast<uint4 const*>(ps + (X ? kRawK : kRawQ) + t * 256 + 16 * cg16), f);
      float s0 = 0.0f, s1 = 0.0f;
#pragma unroll
      for (int m = 0; m < 4; ++m) ffma2p(s0, s1, f[2 * m], f[2 * m + 1], f[2 * m], f[2 * m + 1], s0, s1);
      float ss = s0 + s1;
#pragma unroll
      for (int o = 8; o >= 1; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
      const float r = t < valid ? rsqrtf(ss + 1e-6f) : 0.0f;
      uint32_t o4[4];
#pragma unroll
      for (int m = 0; m < 4; ++m) {
        float a0, a1;
        fmul2p(a0, a1, f[2 * m], f[2 * m + 1], r, r);
        o4[m] = pack2(a0, a1);
      }
      *reinterpret_cast<uint4*>(&qkn[X][t][8 * cg16]) = make_uint4(o4[0], o4[1], o4[2], o4[3]);
    }
  }
  pbar();
  // ---- gate activation + chunk-relative cumsum ----
  {
    const float ha = 0.5f * amp, hg = 0.5f * p.gate_scale;
    float dta[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) dta[i] = ha * reinterpret_cast<float const*>(ps + kDt)[8 * cg16 + i];
    float run[8] = {};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float gv[8];
      unpack8(*reinterpret_cast<uint4 const*>(ps + kRawG + (t_first + j) * 256 + 16 * cg16), gv);
      if (t_first + j < valid) {
#pragma unroll
        for (int m = 0; m < 4; ++m) {
          float x0, x1, y0, y1;
          ffma2p(x0, x1, gv[2 * m], gv[2 * m + 1], ha, ha, dta[2 * m], dta[2 * m + 1]);
          ffma2p(y0, y1, tanh_ap(x0), tanh_ap(x1), hg, hg, hg, hg);
          fadd2p(run[2 * m], run[2 * m + 1], run[2 * m], run[2 * m + 1], y0, y1);
        }
      }
      float4* gp = reinterpret_cast<float4*>(&cg[t_first + j][8 * cg16]);
      gp[0] = make_float4(run[0], run[1], run[2], run[3]);
      gp[1] = make_float4(run[4], run[5], run[6], run[7]);
    }
    float4* tp = reinterpret_cast<float4*>(&tot[tg][8 * cg16]);
    tp[0] = make_float4(run[0], run[1], run[2], run[3]);
    tp[1] = make_float4(run[4], run[5], run[6], run[7]);
  }
  pbar();
  {
    float pre[8] = {};
    for (int t2 = 0; t2 < tg; ++t2) {
      const float4* tp = reinterpret_cast<const float4*>(&tot[t2][8 * cg16]);
      const float4 a = tp[0], b = tp[1];
      fadd2p(pre[0], pre[1], pre[0], pre[1], a.x, a.y);
      fadd2p(pre[2], pre[3], pre[2], pre[3], a.z, a.w);
      fadd2p(pre[4], pre[5], pre[4], pre[5], b.x, b.y);
      fadd2p(pre[6], pre[7], pre[6], pre[7], b.z, b.w);
    }
    const int last = valid - 1;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int t = t_first + j;
      float4* gp = reinterpret_cast<float4*>(&cg[t][8 * cg16]);
      float4 a = gp[0], b = gp[1];
      fadd2p(a.x, a.y, a.x, a.y, pre[0], pre[1]);
      fadd2p(a.z, a.w, a.z, a.w, pre[2], pre[3]);
      fadd2p(b.x, b.y, b.x, b.y, pre[4], pre[5]);
      fadd2p(b.z, b.w, b.z, b.w, pre[6], pre[7]);
      gp[0] = a;
      gp[1] = b;
      if ((t & 15) == 0) {
        float4* rp = reinterpret_cast<float4*>(&gref[t >> 4][8 * cg16]);
        rp[0] = a;
        rp[1] = b;
      }
      if (t == last) {
        float4* rp = reinterpret_cast<float4*>(&gT_s[8 * cg16]);
        rp[0] = a;
        rp[1] = b;
      }
    }
  }
  pbar();
  // ---- images and MMA operands: thread (tl, cgi) = tokens tl + 16 j, channels 8 cgi.. ----
  {
    const int tl = tid & 15, cgi = tid >> 4;
    __nv_bfloat162* pf = reinterpret_cast<__nv_bfloat162*>(ps + kPf);
    for (int e = tid; e < 6 * (kD / 2); e += 256) {
      const int pr = e >> 6, d = 2 * (e & 63);
      const int I = pr == 0 ? 1 : (pr < 3 ? 2 : 3);
      const int J = pr - I * (I - 1) / 2;
      pf[e] = __floats2bfloat162_rn(ex2(gref[I][d] - gref[J][d]), ex2(gref[I][d + 1] - gref[J][d + 1]));
    }
    if (tid < kD) gtp[tid] = valid > 0 ? exp2f(gT_s[tid]) : 1.0f;
    uint32_t sc2;
    {
      const __nv_bfloat162 sc = __floats2bfloat162_rn(p.scale, p.scale);
      sc2 = *reinterpret_cast<const uint32_t*>(&sc);
    }
    float gT[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) gT[i] = gT_s[8 * cgi + i];
#pragma unroll 2
    for (int j = 0; j < 4; ++j) {
      const int t = tl + 16 * j;
      const uint4 knv = *reinterpret_cast<uint4 const*>(&qkn[1][t][8 * cgi]);
      const uint4 qnv = *reinterpret_cast<uint4 const*>(&qkn[0][t][8 * cgi]);
      const uint32_t kn2[4] = {knv.x, knv.y, knv.z, knv.w};
      const uint32_t qn2[4] = {qnv.x, qnv.y, qnv.z, qnv.w};
      float gc[8], gr[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        gc[i] = cg[t][8 * cgi + i];
        gr[i] = gref[j][8 * cgi + i];
      }
      uint32_t kd[4], qd[4], kr[4], rk[4], rq[4], ck[4];
#pragma unroll
      for (int m = 0; m < 4; ++m) {
        float d0, d1, r0, r1;
        fadd2p(d0, d1, gc[2 * m], gc[2 * m + 1], -gr[2 * m], -gr[2 * m + 1]);
        fadd2p(r0, r1, gT[2 * m], gT[2 * m + 1], -gc[2 * m], -gc[2 * m + 1]);
        const uint32_t e = pack2(ex2(gc[2 * m]), ex2(gc[2 * m + 1]));
        const uint32_t er = pack2(ex2(r0), ex2(r1));
        const uint32_t fr = pack2(ex2(d0), ex2(d1));
        const uint32_t ff = pack2(ex2(-d0), ex2(-d1));
        kd[m] = hmul2(kn2[m], e);
        qd[m] = hmul2(hmul2(qn2[m], e), sc2);
        kr[m] = hmul2(kn2[m], er);
        rk[m] = hmul2(kn2[m], fr);
        rq[m] = hmul2(hmul2(qn2[m], fr), sc2);
        ck[m] = hmul2(kn2[m], ff);
      }
      *reinterpret_cast<uint4*>(&qkn[1][t][8 * cgi]) = make_uint4(kd[0], kd[1], kd[2], kd[3]);
      *reinterpret_cast<uint4*>(wq + (cgi * 128 + t) * 8) = make_uint4(qd[0], qd[1], qd[2], qd[3]);
      *reinterpret_cast<uint4*>(ps + kOpRk + (t * kPitch + 8 * cgi) * 2) = make_uint4(rk[0], rk[1], rk[2], rk[3]);
      *reinterpret_cast<uint4*>(ps + kOpRq + (t * kPitch + 8 * cgi) * 2) = make_uint4(rq[0], rq[1], rq[2], rq[3]);
      *reinterpret_cast<uint4*>(ps + kOpCk + (t * kPitch + 8 * cgi) * 2) = make_uint4(ck[0], ck[1], ck[2], ck[3]);
      // kr^T image: element (k, t) at ((t >> 3) * 192 + k) * 8 + (t & 7)
      __nv_bfloat16* kb = krm + ((t >> 3) * 192 + 8 * cgi) * 8 + (t & 7);
#pragma unroll
      for (int m = 0; m < 4; ++m) {
        kb[(2 * m) * 8] = __ushort_as_bfloat16(uint16_t(kr[m] & 0xffffu));
        kb[(2 * m + 1) * 8] = __ushort_as_bfloat16(uint16_t(kr[m] >> 16));
      }
    }
  }
  pbar();
  // ---- A, Mq block products (mma.sync); Mqk -> krm rows 128 + i ----
  {
    const int I = warp < 3 ? 3 : (warp < 5 ? 2 : (warp < 7 ? 1 : 0));
    const int n0 = warp < 3 ? 3 * warp : (warp < 5 ? 3 * (warp - 3) : (warp < 7 ? 2 * (warp - 5) : 0));
    const int nn = (warp == 2 || warp >= 5) ? 2 : 3;
    const int g = lane >> 2, tq = lane & 3;
    const uint32_t a_row = 16 * I + (lane & 15);
    const uint32_t a_col = (lane >> 4) * 8;
    const uint32_t pbase = prep64tc::smem_u32(ps);
    const uint32_t rk_base = pbase + kOpRk + (a_row * kPitch + a_col) * 2;
    const uint32_t rq_base = pbase + kOpRq + (a_row * kPitch + a_col) * 2;
    for (int n = n0; n < n0 + nn; ++n) {
      const int J = n >> 1;
      const bool scale_b = J < I;
      uint32_t const* pf = reinterpret_cast<uint32_t const*>(ps + kPf) + (scale_b ? I * (I - 1) / 2 + J : 0) * 64;
      float ca[4] = {}, cm[4] = {};
      const uint32_t b_base = pbase + kOpCk + ((8 * n + (lane & 7)) * kPitch + (lane >> 3) * 8) * 2;
#pragma unroll
      for (int ks = 0; ks < 8; ks += 2) {
        uint32_t b[4], a[4], q[4];
        ldsm_x4(b_base + ks * 32, b);
        if (scale_b) {
          b[0] = hmul2(b[0], pf[8 * ks + tq]);
          b[1] = hmul2(b[1], pf[8 * ks + 4 + tq]);
          b[2] = hmul2(b[2], pf[8 * ks + 8 + tq]);
          b[3] = hmul2(b[3], pf[8 * ks + 12 + tq]);
        }
        ldsm_x4(rk_base + ks * 32, a);
        ldsm_x4(rq_base + ks * 32, q);
        mma16816(ca, a, b[0], b[1]);
        mma16816(cm, q, b[0], b[1]);
        ldsm_x4(rk_base + (ks + 1) * 32, a);
        ldsm_x4(rq_base + (ks + 1) * 32, q);
        mma16816(ca, a, b[2], b[3]);
        mma16816(cm, q, b[2], b[3]);
      }
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int i = 16 * I + g + 8 * hh;
        const int j = 8 * n + 2 * tq;
        const float bi = beta[i];
        L[i][j] = j < i ? ca[2 * hh] * bi : 0.0f;
        L[i][j + 1] = j + 1 < i ? ca[2 * hh + 1] * bi : 0.0f;
        const bool iv = i < valid;
        const float m0 = (iv && j <= i && j < valid) ? cm[2 * hh] : 0.0f;
        const float m1 = (iv && j + 1 <= i && j + 1 < valid) ? cm[2 * hh + 1] : 0.0f;
        *reinterpret_cast<uint32_t*>(krm + ((j >> 3) * 192 + 128 + i) * 8 + (j & 7)) = pack2(m0, m1);
      }
    }
  }
  pbar();
  // ---- INV = (I + L)^-1 (FP32 diagonal blocks, BF16 tensor-core off-diagonal merges) ----
  float (*X)[kC + 4] = reinterpret_cast<float (*)[kC + 4]>(ps + kInvX);
  if (tid < kC) {
    const int I = tid >> 4, jj = tid & 15, o = 16 * I;
    float x[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      float acc = i == jj ? 1.0f : 0.0f;
#pragma unroll
      for (int k = 0; k < i; ++k) acc -= L[o + i][o + k] * x[k];
      x[i] = acc;
    }
#pragma unroll
    for (int i = 0; i < 16; ++i) X[o + i][o + jj] = x[i];
  }
  pbar();
  if (warp < 3) {
    const int J = warp, g = lane >> 2, tq = lane & 3;
    float (*Tt)[20] = reinterpret_cast<float (*)[20]>(ps + kInvT + warp * 16 * 20 * 4);
    for (int I = J + 1; I < 4; ++I) {
      float tacc[2][4] = {};
      for (int k = J; k < I; ++k) {
        uint32_t a[4];
        const int r0 = 16 * I + g, c0 = 16 * k + 2 * tq;
        a[0] = pack2(L[r0][c0], L[r0][c0 + 1]);
        a[1] = pack2(L[r0 + 8][c0], L[r0 + 8][c0 + 1]);
        a[2] = pack2(L[r0][c0 + 8], L[r0][c0 + 9]);
        a[3] = pack2(L[r0 + 8][c0 + 8], L[r0 + 8][c0 + 9]);
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
          const int col = 16 * J + 8 * nt + g, kr0 = 16 * k + 2 * tq;
          const uint32_t b0 = pack2(X[kr0][col], X[kr0 + 1][col]);
          const uint32_t b1 = pack2(X[kr0 + 8][col], X[kr0 + 9][col]);
          mma16816(tacc[nt], a, b0, b1);
        }
      }
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        Tt[g][8 * nt + 2 * tq] = tacc[nt][0];
        Tt[g][8 * nt + 2 * tq + 1] = tacc[nt][1];
        Tt[g + 8][8 * nt + 2 * tq] = tacc[nt][2];
        Tt[g + 8][8 * nt + 2 * tq + 1] = tacc[nt][3];
      }
      __syncwarp();
      uint32_t a[4];
      {
        const int r0 = 16 * I + g, c0 = 16 * I + 2 * tq;
        a[0] = pack2(X[r0][c0], X[r0][c0 + 1]);
        a[1] = pack2(X[r0 + 8][c0], X[r0 + 8][c0 + 1]);
        a[2] = pack2(X[r0][c0 + 8], X[r0][c0 + 9]);
        a[3] = pack2(X[r0 + 8][c0 + 8], X[r0 + 8][c0 + 9]);
      }
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        float xacc[4] = {};
        const int col = 8 * nt + g;
        const uint32_t b0 = pack2(Tt[2 * tq][col], Tt[2 * tq + 1][col]);
        const uint32_t b1 = pack2(Tt[2 * tq + 8][col], Tt[2 * tq + 9][col]);
        mma16816(xacc, a, b0, b1);
        const int r0 = 16 * I + g, c0 = 16 * J + 8 * nt + 2 * tq;
        X[r0][c0] = -xacc[0];
        X[r0][c0 + 1] = -xacc[1];
        X[r0 + 8][c0] = -xacc[2];
        X[r0 + 8][c0 + 1] = -xacc[3];
      }
      __syncwarp();
    }
  }
  pbar();
  // ---- Mqk zeros above the diagonal blocks; INVb = bf16(INV_ij beta_j), block-masked ----
  __nv_bfloat16 (*invb)[kC + 8] = reinterpret_cast<__nv_bfloat16 (*)[kC + 8]>(ps + kInvB);
  {
    for (int uidx = tid; uidx < 512; uidx += 256) {
      const int jg = uidx >> 6, i = uidx & 63;
      if (8 * jg >= 16 * ((i >> 4) + 1))
        *reinterpret_cast<uint4*>(krm + (jg * 192 + 128 + i) * 8) = make_uint4(0, 0, 0, 0);
    }
    const int i = tid >> 2, j0 = 16 * (tid & 3);
    uint32_t pk[8] = {};
    if (j0 <= i) {
#pragma unroll
      for (int e = 0; e < 16; e += 2) {
        float v0, v1;
        fmul2p(v0, v1, X[i][j0 + e], X[i][j0 + e + 1], beta[j0 + e], beta[j0 + e + 1]);
        pk[e / 2] = pack2(v0, v1);
      }
    }
    uint4* dst = reinterpret_cast<uint4*>(&invb[i][j0]);
    dst[0] = make_uint4(pk[0], pk[1], pk[2], pk[3]);
    dst[1] = make_uint4(pk[4], pk[5], pk[6], pk[7]);
  }
  pbar();
  // ---- w = INVb kd (warps 0-3, 32 columns each) | u^T = v^T INVb^T (warps 4-7, 32 v each) ----
  {
    const int g = lane >> 2, tq = lane & 3;
    const int n0 = 32 * (warp & 3);
    if (warp >= 4) {
      float au[2][8][4] = {};
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) {
        uint32_t a[2][4];
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
          const int q = lane >> 3;
          ldsm_x4_t(prep64tc::smem_u32(ps + kRawV + (16 * ks + (lane & 7) + 8 * (q >> 1)) * 256 +
                                       (n0 + 16 * mt + 8 * (q & 1)) * 2),
                    a[mt]);
        }
#pragma unroll
        for (int np = 0; np < 4; ++np) {
          uint32_t b[4];
          ldsm_x4(prep64tc::smem_u32(&invb[16 * np + (lane & 7) + 8 * (lane >> 4)][16 * ks + 8 * ((lane >> 3) & 1)]),
                  b);
#pragma unroll
          for (int mt = 0; mt < 2; ++mt) {
            mma16816(au[mt][2 * np], a[mt], b[0], b[1]);
            mma16816(au[mt][2 * np + 1], a[mt], b[2], b[3]);
          }
        }
      }
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < 8; ++nt)
#pragma unroll
          for (int hh = 0; hh < 2; ++hh) {
            const int v = n0 + 16 * mt + g + 8 * hh;
            *reinterpret_cast<uint32_t*>(uimg + (nt * 128 + v) * 8 + 2 * tq) =
                pack2(au[mt][nt][2 * hh], au[mt][nt][2 * hh + 1]);
          }
    } else {
      float acc[4][4][4] = {};
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) {
        uint32_t b[2][4];
#pragma unroll
        for (int np = 0; np < 2; ++np) {
          const int kr0 = 16 * ks + (lane & 7) + 8 * ((lane >> 3) & 1);
          const int nc = n0 + 16 * np + 8 * (lane >> 4);
          ldsm_x4_t(prep64tc::smem_u32(&qkn[1][kr0][nc]), b[np]);
        }
#pragma unroll
        for (int mt = 0; mt < 4; ++mt) {
          uint32_t a[4];
          ldsm_x4(prep64tc::smem_u32(&invb[16 * mt + (lane & 15)][16 * ks + 8 * (lane >> 4)]), a);
#pragma unroll
          for (int np = 0; np < 2; ++np) {
            mma16816(acc[mt][2 * np], a, b[np][0], b[np][1]);
            mma16816(acc[mt][2 * np + 1], a, b[np][2], b[np][3]);
          }
        }
      }
#pragma unroll
      for (int mt = 0; mt < 4; ++mt)
#pragma unroll
        for (int nt = 0; nt < 4; ++nt)
#pragma unroll
          for (int hh = 0; hh < 2; ++hh) {
            const int i = 16 * mt + g + 8 * hh, n = n0 + 8 * nt + 2 * tq;
            *reinterpret_cast<uint32_t*>(wq + ((n >> 3) * 128 + 64 + i) * 8 + (n & 7)) =
                pack2(acc[mt][nt][2 * hh], acc[mt][nt][2 * hh + 1]);
          }
    }
  }
}

// Standalone check of prep_chunk: one CTA per (chunk, head), images copied out in
// the kp64 global formats (wq3, kr, mqk, u3, gt).
constexpr int kStage = 74752;
constexpr int kSmemV2 = kPrepBytes + kStage + 64;

__global__ void __launch_bounds__(256, 1) kp64v2_kernel(const __grid_constant__ prep64tc::Params p) {
  k31sg::begin_call((long long)(p.T));  // timeout lines: call index and size
  extern __shared__ __align__(1024) unsigned char sm[];
  unsigned char* stage = sm;
  unsigned char* ps = sm + kStage;
  const int ci = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
  int tok0 = 0, valid = 0;
  {
    int base = 0;
    for (int b = 0; b < p.N; ++b) {
      const int s0 = p.cu_seqlens[b], len = p.cu_seqlens[b + 1] - s0;
      const int nc = (len + kC - 1) / kC;
      if (ci < base + nc) {
        tok0 = s0 + (ci - base) * kC;
        valid = min(kC, len - (ci - base) * kC);
        break;
      }
      base += nc;
    }
  }
  const uint32_t bar = prep64tc::smem_u32(sm + kStage + kPrepBytes);
  if (tid == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(bar) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(4 * 16384)
                 : "memory");
    const int H = p.H;
    prep64tc::tma2d(prep64tc::smem_u32(ps + kRawQ), &p.tm_x, h * kD, tok0, bar);
    prep64tc::tma2d(prep64tc::smem_u32(ps + kRawK), &p.tm_x, H * kD + h * kD, tok0, bar);
    prep64tc::tma2d(prep64tc::smem_u32(ps + kRawV), &p.tm_x, 2 * H * kD + h * kD, tok0, bar);
    prep64tc::tma2d(prep64tc::smem_u32(ps + kRawG), &p.tm_g, h * kD, tok0, bar);
  }
  if (tid < kD) reinterpret_cast<float*>(ps + kDt)[tid] = p.dt_bias[h * kD + tid];
  const float amp = expf(p.a_log[h]);
  __syncthreads();
  k31sg::mbar_wait(bar, 0u, "flashkda_prep64_tc", __LINE__);
  prep_chunk(p, ps, h, tok0, valid, tid, tid >> 5, tid & 31, amp, stage);
  __syncthreads();
  const int64_t img = int64_t(h) * p.C_total + ci;
  uint4 const* st = reinterpret_cast<uint4 const*>(stage);
  uint4* wq = reinterpret_cast<uint4*>(p.kdq + img * (2 * kC * kD));
  uint4* kr = reinterpret_cast<uint4*>(p.kr + img * (kC * kD));
  uint4* mq = reinterpret_cast<uint4*>(p.mqk + img * (kC * kC));
  uint4* u = reinterpret_cast<uint4*>(p.v_img + img * (kC * kD));
  for (int e = tid; e < 2048; e += 256) wq[e] = st[e];
  for (int e = tid; e < 1536; e += 256) {
    const int g = e / 192, r = e % 192;
    if (r < 128) kr[g * 128 + r] = st[2048 + e];
    else mq[g * 64 + r - 128] = st[2048 + e];
  }
  for (int e = tid; e < 1024; e += 256) u[e] = st[3584 + e];
  if (tid < 32) reinterpret_cast<uint4*>(p.gt + img * kD)[tid] = st[4608 + tid];
}

}  // namespace kf64

void launch_prep64_tc(void const* x, int64_t ldx, float const* conv_w, float* conv_state,
                      int64_t st_s0, int64_t st_s1, int64_t st_s2, int32_t const* cache_indices,
                      int64_t ci_s0, bool const* has_init, int null_block_id, void const* g,
                      int64_t ldg, void const* beta, int64_t ldb, float const* a_log,
                      float const* dt_bias, float lower_bound, float scale, int const* cu_seqlens,
                      int N, int H, int T, int C_total, void* kdq, void* kr, void* inv, void* mqk,
                      float* gt, void* v_img, void* beta_out, long long* dbg, cudaStream_t stream) {
  using bf = __nv_bfloat16;
  static PFN_cuTensorMapEncodeTiled_v12000 encode = [] {
    void* fn = nullptr;
    cudaDriverEntryPointQueryResult q;
    cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &fn, cudaEnableDefault, &q);
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(fn);
  }();
  auto make_map = [&](CUtensorMap* tm, void const* base, int64_t inner, int64_t rows, int64_t ld,
                      int box_rows) {
    const cuuint64_t dims[2] = {cuuint64_t(inner), cuuint64_t(rows)};
    const cuuint64_t strides[1] = {cuuint64_t(ld * 2)};
    const cuuint32_t box[2] = {128u, cuuint32_t(box_rows)};
    const cuuint32_t es[2] = {1u, 1u};
    encode(tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(base), dims, strides, box, es,
           CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
           CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  };
  prep64tc::Params p{{}, {}, static_cast<bf const*>(x), ldx, conv_w, conv_state, st_s0, st_s1, st_s2,
                     cache_indices, ci_s0, has_init, null_block_id, static_cast<bf const*>(g),
                     ldg, static_cast<bf const*>(beta), ldb, a_log, dt_bias,
                     lower_bound * 1.4426950408889634f, scale, cu_seqlens, N, H, T, C_total,
                     static_cast<bf*>(kdq), static_cast<bf*>(kr), static_cast<bf*>(inv),
                     static_cast<bf*>(mqk), gt, static_cast<bf*>(v_img), static_cast<bf*>(beta_out), dbg};
  if (conv_w == nullptr) {
    // kp64: x = post-conv q|k|v rows; kdq -> qd|w image, v_img -> u image [tg][v][8 t]
    make_map(&p.tm_x, x, int64_t(3) * H * 128, T, ldx, prep64tc::kC);
    make_map(&p.tm_g, g, int64_t(H) * 128, T, ldg, prep64tc::kC);
    if (null_block_id == -2) {
      // kp64 v2: prep_chunk (the fused core's producer) standalone
      cudaFuncSetAttribute(kf64::kp64v2_kernel, cudaFuncAttributeSharedMemoryMode,
                           cudaSharedMemoryModeAllowOversizedSharedMemory);
      cudaFuncSetAttribute(kf64::kp64v2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           kf64::kSmemV2);
      kf64::kp64v2_kernel<<<dim3(C_total, H), 256, kf64::kSmemV2, stream>>>(p);
      return;
    }
    const int smk = sizeof(prep64tc::SmemK);
    cudaFuncSetAttribute(prep64tc::kp64_kernel, cudaFuncAttributeSharedMemoryMode,
                         cudaSharedMemoryModeAllowOversizedSharedMemory);
    cudaFuncSetAttribute(prep64tc::kp64_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smk);
    prep64tc::kp64_kernel<<<dim3(C_total, H), prep64tc::kThreads, smk, stream>>>(p);
    return;
  }
  make_map(&p.tm_x, x, int64_t(3) * H * 128, T, ldx, prep64tc::kRawRows);
  make_map(&p.tm_g, g, int64_t(H) * 128, T, ldg, prep64tc::kC);
  const int smem = sizeof(prep64tc::Smem);
  // two ~148 KB CTAs per SM need the sm_107 oversized shared-memory mode
  cudaFuncSetAttribute(prep64tc::prep64_tc_kernel, cudaFuncAttributeSharedMemoryMode,
                       cudaSharedMemoryModeAllowOversizedSharedMemory);
  cudaFuncSetAttribute(prep64tc::prep64_tc_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  prep64tc::prep64_tc_kernel<<<dim3(C_total, H), prep64tc::kThreads, smem, stream>>>(p);
}
