// Kimi-K3: chunk-64 KDA prepare on the raw in_proj rows (short conv fused),
// tensor-core block products, for the chunk-64 tcgen05 recurrence (k64_tc.cu).
//
// One CTA per (chunk, head), 256 threads; thread (tg, cg) = (tid >> 4, tid & 15)
// owns tokens 4 tg .. 4 tg + 3 and channels 8 cg .. 8 cg + 7.
//   conv     q|k|v = SiLU(causal width-4 conv) of the raw rows, FP32 weights,
//            FP32 conv state: the qkv_conv kernel's arithmetic (bit for bit),
//            the chunk-0 CTA of a sequence reads and rewrites its conv state
//   v image  bf16 post-conv v, [8 t-groups][128 v][8 t]
//   qn, kn   = bf16(x * rsqrt(sum x^2 + 1e-6))
//   gc       = cumsum_t gate_scale * sigmoid(exp(A_log) (g + dt_bias)) (FP32,
//            chunk-relative), gT = gc[last valid row], g_total = exp2(gT)
//   kd = bf16(kn * bf16(exp2 gc)), qd = bf16(bf16(qn * bf16(exp2 gc)) * bf16(scale)),
//   kr = bf16(kn * bf16(exp2(gT - gc)))
//   A, Mq    per 16-row block I with the reference row r = 16 I: row operand
//            bf16(kn_i * bf16(exp2(gc_i - gc_r))) (<= 1), column operand
//            bf16(kn_j * bf16(exp2(gc_r - gc_j))), mma.sync m16n8k16 BF16 -> FP32
//   INV      = (I + beta tril(A, -1))^-1 in FP32 (16x16 diagonal blocks by forward
//            substitution, off-diagonal blocks by block products), bf16 image
//   Mqk      = bf16(tril(Mq))
// The images are k64_tc.cu's (rows past the sequence end zero).
#include <cstddef>
#include <cstdint>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cudaTypedefs.h>

namespace prep64tc {

constexpr int kC = 64;
constexpr int kD = 128;
constexpr int kPitch = 136;  // bf16 per operand row: conflict-free ldmatrix
constexpr int kThreads = 256;

struct Smem {
  // region A: raw rows (TMA) until the gate pass, then the MMA operands
  __nv_bfloat16 rk[kC * kPitch];
  __nv_bfloat16 rq[kC * kPitch];
  union {
    __nv_bfloat16 ck[160 * kPitch];  // column operands of blocks 0..3 (16, 32, 48, 64 rows)
    struct {
      float x[kC][kC + 1];
      float t[3][16][17];
    } inv;
  } u;
  __nv_bfloat16 qkn[2][kC][kD];  // bf16 normalized post-conv q, k
  union {
    float g[kC][kD + 4];  // chunk-relative gate cumsum
    float l[kC][kC + 1];  // beta_i A_ij below the diagonal
  } w;
  float gref[4][kD];
  float gT[kD];
  float beta[kC];
  alignas(8) uint64_t bar;
};

// Raw rows staged by TMA at the start of the CTA (aliases rk / rq / u.ck, which
// are written only after the conv and the gate have read them): q, k, v boxes of
// 67 rows (3 history rows + 64) and the gate box of 64 rows, 256 B per row.
constexpr int kRawRows = kC + 3;
constexpr int kRawBox = kRawRows * kD * 2;
constexpr int kOffRawG = 3 * kRawBox;
constexpr int kRawBytes = 3 * kRawBox + kC * kD * 2;
static_assert(kRawBytes <= int(offsetof(Smem, qkn)), "raw rows must fit in rk | rq | u");

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

// Rows of column block I in s.u.ck start at kCkBase[I].
__device__ __forceinline__ int ck_base(int I) { return I * (I + 1) * 8; }

__global__ void __launch_bounds__(kThreads, 2) prep64_tc_kernel(const __grid_constant__ Params p) {
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
  unsigned char* raw = smem_raw;
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

  // ---- short conv (q, k, v) ----------------------------------------------------
  const int t_first = 4 * tg;
  const bool active = t_first < valid;
  int64_t line = 0;
  bool load_init = false, write_state = false;
  if (c_in_seq == 0) {
    line = p.cache_indices[int64_t(seq) * p.ci_s0];
    const bool null_line = p.null_block_id >= 0 && line == p.null_block_id;
    load_init = !null_line && p.has_init[seq];
    write_state = !null_line && tg == 0;
  }
  __syncthreads();  // barrier init visible
  {
    uint32_t ok = 0;
    while (!ok) {
      asm volatile(
          "{\n .reg .pred P1;\n mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], 0;\n"
          " selp.u32 %0, 1, 0, P1;\n}"
          : "=r"(ok)
          : "r"(bar)
          : "memory");
    }
  }
  stamp(15);
#pragma unroll 1
  for (int xi = 0; xi < 3; ++xi) {
    const int X = xi == 0 ? 2 : xi - 1;  // v (straight to its image), then q, k
    const int ch0 = X * HD + h * kD + 8 * cg;
    unsigned char const* rawx = raw + X * kRawBox + 16 * cg;
    float w[4][8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float4 wv = __ldg(reinterpret_cast<float4 const*>(p.conv_w + int64_t(ch0 + i) * 4));
      w[0][i] = wv.x;
      w[1][i] = wv.y;
      w[2][i] = wv.z;
      w[3][i] = wv.w;
    }
    uint4 o0 = make_uint4(0, 0, 0, 0), o1 = o0, o2 = o0, o3 = o0;
    if (active || write_state) {
      float hist[3][8];
      float* st = p.conv_state + line * p.st_s0 + int64_t(ch0) * p.st_s1;
#pragma unroll
      for (int k = 0; k < 3; ++k) {
        const int pos = t_first - 3 + k;  // chunk-relative
        if (pos >= 0 || c_in_seq > 0) {
          unpack8(*reinterpret_cast<uint4 const*>(rawx + (pos + 3) * 256), hist[k]);
        } else {
#pragma unroll
          for (int i = 0; i < 8; ++i)
            hist[k][i] = load_init ? st[i * p.st_s1 + k * p.st_s2] : 0.0f;
        }
      }
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
#pragma unroll
            for (int i = 0; i < 8; ++i) {
              const int k = j + seq_len;  // 0..2; selects keep hist in registers
              ns[i] = k == 0 ? hist[0][i] : (k == 1 ? hist[1][i] : hist[2][i]);
            }
          }
#pragma unroll
          for (int i = 0; i < 8; ++i) st[i * p.st_s1 + j * p.st_s2] = ns[i];
        }
      }
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        if (t_first + j < valid) {
          float cur[8], o[8];
          unpack8(*reinterpret_cast<uint4 const*>(rawx + (t_first + j + 3) * 256), cur);
          conv_row(hist[0], hist[1], hist[2], cur, w, o);
          const uint4 ov = pack8(o);
          if (j == 0) o0 = ov;
          if (j == 1) o1 = ov;
          if (j == 2) o2 = ov;
          if (j == 3) o3 = ov;
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            hist[0][i] = hist[1][i];
            hist[1][i] = hist[2][i];
            hist[2][i] = cur[i];
          }
        }
      }
    }
    if (X == 2) {
      // v image: channel c, tokens t_first..t_first+3 -> 8 contiguous bytes
      __nv_bfloat16* vb = p.v_img + img * (kC * kD) + ((t_first >> 3) * kD + 8 * cg) * 8 + (t_first & 7);
      const uint32_t a0[4] = {o0.x, o0.y, o0.z, o0.w}, a1[4] = {o1.x, o1.y, o1.z, o1.w};
      const uint32_t a2[4] = {o2.x, o2.y, o2.z, o2.w}, a3[4] = {o3.x, o3.y, o3.z, o3.w};
#pragma unroll
      for (int m = 0; m < 4; ++m) {
        *reinterpret_cast<uint2*>(vb + (2 * m) * 8) =
            make_uint2(__byte_perm(a0[m], a1[m], 0x5410), __byte_perm(a2[m], a3[m], 0x5410));
        *reinterpret_cast<uint2*>(vb + (2 * m + 1) * 8) =
            make_uint2(__byte_perm(a0[m], a1[m], 0x7632), __byte_perm(a2[m], a3[m], 0x7632));
      }
    } else {
      // L2 norm (16 lanes per token) -> bf16(x * rsqrt(sum x^2 + 1e-6)) in shared memory
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const uint4 ov = j == 0 ? o0 : (j == 1 ? o1 : (j == 2 ? o2 : o3));
        float f[8];
        unpack8(ov, f);
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
  __syncthreads();  // raw q|k|v rows (halo reads) done: the raw q box holds the scan totals
  stamp(1);

  // ---- gate activation + chunk-relative cumsum ----------------------------------
  float* tot = reinterpret_cast<float*>(raw);  // [16][128]
  {
    const float amp = expf(p.a_log[h]);
    float dt[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) dt[i] = p.dt_bias[h * kD + 8 * cg + i];
    float run[8] = {};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float gv[8] = {};
      if (t_first + j < valid) {
        unpack8(*reinterpret_cast<uint4 const*>(raw + kOffRawG + (t_first + j) * 256 + 16 * cg), gv);
#pragma unroll
        for (int i = 0; i < 8; ++i) gv[i] = p.gate_scale * sigmoid_tanh(amp * (gv[i] + dt[i]));
      }
#pragma unroll
      for (int i = 0; i < 8; ++i) run[i] += gv[i];
      float4* gp = reinterpret_cast<float4*>(&s.w.g[t_first + j][8 * cg]);
      gp[0] = make_float4(run[0], run[1], run[2], run[3]);
      gp[1] = make_float4(run[4], run[5], run[6], run[7]);
    }
    float4* tp = reinterpret_cast<float4*>(tot + tg * kD + 8 * cg);
    tp[0] = make_float4(run[0], run[1], run[2], run[3]);
    tp[1] = make_float4(run[4], run[5], run[6], run[7]);
  }
  __syncthreads();
  stamp(2);
  {
    float pre[8] = {};
    for (int t2 = 0; t2 < tg; ++t2) {
      const float4* tp = reinterpret_cast<const float4*>(tot + t2 * kD + 8 * cg);
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
      float* gp = &s.w.g[t][8 * cg];
      float v[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) v[i] = gp[i] + pre[i];
#pragma unroll
      for (int i = 0; i < 8; ++i) gp[i] = v[i];
      if ((t & 15) == 0) {
#pragma unroll
        for (int i = 0; i < 8; ++i) s.gref[t >> 4][8 * cg + i] = v[i];
      }
      if (t == last) {
#pragma unroll
        for (int i = 0; i < 8; ++i) s.gT[8 * cg + i] = v[i];
      }
    }
  }
  __syncthreads();
  stamp(3);

  // ---- decayed images and the factored block operands ---------------------------
  {
    const float sc = bfr(p.scale);
    const int blk = tg >> 2;
    float gT[8], gr[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      gT[i] = s.gT[8 * cg + i];
      gr[i] = s.gref[blk][8 * cg + i];
    }
    __nv_bfloat16* kdq = p.kdq + img * (2 * kC * kD) + (cg * 128) * 8;
    uint4 k0 = make_uint4(0, 0, 0, 0), k1 = k0, k2 = k0, k3 = k0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int t = t_first + j;
      const bool ok = t < valid;
      float kn[8], qn[8], gc[8];
      unpack8(*reinterpret_cast<uint4 const*>(&s.qkn[1][t][8 * cg]), kn);
      unpack8(*reinterpret_cast<uint4 const*>(&s.qkn[0][t][8 * cg]), qn);
      {
        const float4* gp = reinterpret_cast<const float4*>(&s.w.g[t][8 * cg]);
        const float4 a = gp[0], b = gp[1];
        gc[0] = a.x;
        gc[1] = a.y;
        gc[2] = a.z;
        gc[3] = a.w;
        gc[4] = b.x;
        gc[5] = b.y;
        gc[6] = b.z;
        gc[7] = b.w;
      }
      {
        float kd[8], qd[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const float e = bfr(ex2(gc[i]));
          kd[i] = ok ? kn[i] * e : 0.0f;
          qd[i] = ok ? bfr(qn[i] * e) * sc : 0.0f;
        }
        *reinterpret_cast<uint4*>(kdq + t * 8) = pack8(kd);
        *reinterpret_cast<uint4*>(kdq + (64 + t) * 8) = pack8(qd);
      }
      {
        float kr[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) kr[i] = ok ? kn[i] * bfr(ex2(gT[i] - gc[i])) : 0.0f;
        const uint4 kv = pack8(kr);
        if (j == 0) k0 = kv;
        if (j == 1) k1 = kv;
        if (j == 2) k2 = kv;
        if (j == 3) k3 = kv;
      }
      {
        float rk[8], rq[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const float fr = bfr(ex2(gc[i] - gr[i]));
          rk[i] = ok ? kn[i] * fr : 0.0f;
          rq[i] = ok ? bfr(qn[i] * fr) * sc : 0.0f;
        }
        *reinterpret_cast<uint4*>(&s.rk[t * kPitch + 8 * cg]) = pack8(rk);
        *reinterpret_cast<uint4*>(&s.rq[t * kPitch + 8 * cg]) = pack8(rq);
      }
      // column operands of blocks I >= blk
#pragma unroll
      for (int I = 0; I < 4; ++I) {
        if (I >= blk) {
          float cv[8];
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            const float f = bfr(ex2(s.gref[I][8 * cg + i] - gc[i]));
            cv[i] = ok ? kn[i] * f : 0.0f;
          }
          *reinterpret_cast<uint4*>(&s.u.ck[(ck_base(I) + t) * kPitch + 8 * cg]) = pack8(cv);
        }
      }
    }
    __nv_bfloat16* krb = p.kr + img * (kC * kD) + ((t_first >> 3) * kD + 8 * cg) * 8 + (t_first & 7);
    const uint32_t a0[4] = {k0.x, k0.y, k0.z, k0.w}, a1[4] = {k1.x, k1.y, k1.z, k1.w};
    const uint32_t a2[4] = {k2.x, k2.y, k2.z, k2.w}, a3[4] = {k3.x, k3.y, k3.z, k3.w};
#pragma unroll
    for (int m = 0; m < 4; ++m) {
      *reinterpret_cast<uint2*>(krb + (2 * m) * 8) =
          make_uint2(__byte_perm(a0[m], a1[m], 0x5410), __byte_perm(a2[m], a3[m], 0x5410));
      *reinterpret_cast<uint2*>(krb + (2 * m + 1) * 8) =
          make_uint2(__byte_perm(a0[m], a1[m], 0x7632), __byte_perm(a2[m], a3[m], 0x7632));
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
    const uint32_t rk_base = smem_u32(s.rk) + (a_row * kPitch + a_col) * 2;
    const uint32_t rq_base = smem_u32(s.rq) + (a_row * kPitch + a_col) * 2;
    __nv_bfloat16* mqk = p.mqk + img * (kC * kC);
    for (int n = n0; n < n0 + nn; ++n) {
      float ca[4] = {}, cm[4] = {};
      // B: rows (8 n .. 8 n + 7) of block I's column operand; x4 = two k-steps
      const uint32_t b_base =
          smem_u32(s.u.ck) + ((ck_base(I) + 8 * n + (lane & 7)) * kPitch + (lane >> 3) * 8) * 2;
#pragma unroll
      for (int ks = 0; ks < 8; ks += 2) {
        uint32_t b[4], a[4], q[4];
        ldsm_x4(b_base + ks * 32, b);
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
        s.w.l[i][j] = j < i ? ca[2 * hh] * bi : 0.0f;
        s.w.l[i][j + 1] = j + 1 < i ? ca[2 * hh + 1] * bi : 0.0f;
        const bool iv = i < valid;
        const float m0 = (iv && j <= i && j < valid) ? cm[2 * hh] : 0.0f;
        const float m1 = (iv && j + 1 <= i && j + 1 < valid) ? cm[2 * hh + 1] : 0.0f;
        *reinterpret_cast<uint32_t*>(mqk + ((j >> 3) * 64 + i) * 8 + (j & 7)) = pack2(m0, m1);
      }
    }
  }
  __syncthreads();
  stamp(5);

  // ---- INV = (I + L)^-1, FP32 ---------------------------------------------------
  auto& X = s.u.inv.x;
  auto& L = s.w.l;
  auto& Tm = s.u.inv.t;
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
  __syncthreads();
  stamp(6);
  // level 1: X[I+1][I] = -X[I+1][I+1] (L[I+1][I] X[I][I]), I = 0..2
  for (int e = tid; e < 768; e += kThreads) {
    const int I = e >> 8, r = (e >> 4) & 15, c = e & 15;
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < 16; ++k) acc += L[16 * (I + 1) + r][16 * I + k] * X[16 * I + k][16 * I + c];
    Tm[I][r][c] = acc;
  }
  __syncthreads();
  stamp(7);
  for (int e = tid; e < 768; e += kThreads) {
    const int I = e >> 8, r = (e >> 4) & 15, c = e & 15;
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < 16; ++k) acc += X[16 * (I + 1) + r][16 * (I + 1) + k] * Tm[I][k][c];
    X[16 * (I + 1) + r][16 * I + c] = -acc;
  }
  __syncthreads();
  stamp(8);
  // level 2: X[I+2][I] = -X[I+2][I+2] (L[I+2][I] X[I][I] + L[I+2][I+1] X[I+1][I]), I = 0, 1
  for (int e = tid; e < 512; e += kThreads) {
    const int I = e >> 8, r = (e >> 4) & 15, c = e & 15;
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < 32; ++k) acc += L[16 * (I + 2) + r][16 * I + k] * X[16 * I + k][16 * I + c];
    Tm[I][r][c] = acc;
  }
  __syncthreads();
  stamp(9);
  for (int e = tid; e < 512; e += kThreads) {
    const int I = e >> 8, r = (e >> 4) & 15, c = e & 15;
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < 16; ++k) acc += X[16 * (I + 2) + r][16 * (I + 2) + k] * Tm[I][k][c];
    X[16 * (I + 2) + r][16 * I + c] = -acc;
  }
  __syncthreads();
  stamp(10);
  // level 3: X[3][0] = -X[3][3] (L[3][0] X[0][0] + L[3][1] X[1][0] + L[3][2] X[2][0])
  {
    const int r = tid >> 4, c = tid & 15;
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < 48; ++k) acc += L[48 + r][k] * X[k][c];
    Tm[0][r][c] = acc;
  }
  __syncthreads();
  stamp(11);
  {
    const int r = tid >> 4, c = tid & 15;
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < 16; ++k) acc += X[48 + r][48 + k] * Tm[0][k][c];
    X[48 + r][c] = -acc;
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

}  // namespace prep64tc

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
  make_map(&p.tm_x, x, int64_t(3) * H * 128, T, ldx, prep64tc::kRawRows);
  make_map(&p.tm_g, g, int64_t(H) * 128, T, ldg, prep64tc::kC);
  const int smem = sizeof(prep64tc::Smem);
  // two ~148 KB CTAs per SM need the sm_107 oversized shared-memory mode
  cudaFuncSetAttribute(prep64tc::prep64_tc_kernel, cudaFuncAttributeSharedMemoryMode,
                       cudaSharedMemoryModeAllowOversizedSharedMemory);
  cudaFuncSetAttribute(prep64tc::prep64_tc_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  prep64tc::prep64_tc_kernel<<<dim3(C_total, H), prep64tc::kThreads, smem, stream>>>(p);
}
