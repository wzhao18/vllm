// Kimi K3 KDA prefill short convolution (width 4, SiLU) over the packed Q|K|V in_proj columns, one launch.
//
// Persistent, balanced streaming: CTA (column block, part) owns NC x 256 channels of an equal share of the
// batch's token rows (the varlen sequences laid end to end), so every CTA streams the same number of rows
// with one prologue. One producer warp streams the rows into a shared-memory ring with 1D bulk async copies
// (cp.async.bulk, one NC x 512 B copy per row, mbarrier completion); NC consumer warps own 256 channels each
// (8 per lane, 16 B), keep the 3 history rows in registers across rows and release each slot through an
// "empty" mbarrier. Part boundaries never fall on the 2nd or 3rd row of a sequence, so the CTA that owns a
// sequence's first row is the only reader of its old conv state, and it writes the new state.
// Arithmetic is the stock Triton kernel's (causal_conv1d_fn), so outputs and state are bitwise equal.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/all.h>
#include <torch/library.h>

namespace k3conv {

constexpr int kLanes = 32;
constexpr int kVec = 8;                    // bf16 channels per lane
constexpr int kWarpCols = kLanes * kVec;   // 256 channels per warp
constexpr int kRowBytes = kWarpCols * 2;   // 512 B per warp row
constexpr int kRowsPerSlot = 4;            // rows per ring slot: one barrier wait per slot, and the 3-row
                                           // history rotates through 4 registers with no moves

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count));
}

__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes)
               : "memory");
}

__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(bar)) : "memory");
}

__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t phase) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "WAIT_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "@!p bra WAIT_%=;\n"
      "}\n" ::"r"(smem_u32(bar)),
      "r"(phase)
      : "memory");
}

__device__ __forceinline__ void bulk_load(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
          smem_u32(dst)),
      "l"(src), "r"(bytes), "r"(smem_u32(bar))
      : "memory");
}

__device__ __forceinline__ uint4 ld_shared_v4(const void* p) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "r"(smem_u32(p)));
  return v;
}

__device__ __forceinline__ void st_global_cs_v4(void* p, uint4 v) {
  asm volatile("st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
               : "memory");
}

__device__ __forceinline__ uint4 ld_global_v4(const void* p) {
  uint4 v;
  asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}

// The stock Triton kernel's SiLU, `acc / (1 + tl.exp(-acc))`, as Triton lowers it for FP32:
// exp(x) = ex2.approx(x * log2(e)), then the approximate full-range division div.full.f32. The .ftz form of
// ex2 gives the same 1 + e for every input: a subnormal argument gives 1 either way, and a subnormal e
// vanishes in 1 + e.
__device__ __forceinline__ float silu_triton(float acc) {
  float t, e, d, r;
  asm("mul.f32 %0, %1, 0fBFB8AA3B;" : "=f"(t) : "f"(acc));
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e) : "f"(t));
  asm("add.f32 %0, %1, 0f3F800000;" : "=f"(d) : "f"(e));
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(acc), "f"(d));
  return r;
}

__device__ __forceinline__ float ex2_ftz(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float rcp_ftz(float x) {
  float y;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ uint32_t pack2(float a, float b) {
  __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&h);
}

// SiLU of 8 accumulators, packed to bf16. For 1 + e <= 2^126, div.full.f32 is exactly a * rcp.approx(d)
// (no range scaling), done here with packed FP32 pairs (each lane rounds as the scalar op). If any d is
// larger (acc < -87), the 8 values are redone with div.full itself.
__device__ __forceinline__ uint4 silu8(const float (&acc)[kVec]) {
  float o[kVec];
  float dmax = 0.0f;
#pragma unroll
  for (int q = 0; q < kVec / 2; ++q) {
    const float2 a = make_float2(acc[2 * q], acc[2 * q + 1]);
    const float2 t = __fmul2_rn(a, make_float2(-1.44269502162933349609375f, -1.44269502162933349609375f));
    const float2 d = __fadd2_rn(make_float2(ex2_ftz(t.x), ex2_ftz(t.y)), make_float2(1.0f, 1.0f));
    const float2 r = __fmul2_rn(a, make_float2(rcp_ftz(d.x), rcp_ftz(d.y)));
    o[2 * q] = r.x;
    o[2 * q + 1] = r.y;
    dmax = fmaxf(dmax, fmaxf(d.x, d.y));
  }
  if (dmax > 0x1p126f) {
#pragma unroll
    for (int i = 0; i < kVec; ++i) o[i] = silu_triton(acc[i]);
  }
  return make_uint4(pack2(o[0], o[1]), pack2(o[2], o[3]), pack2(o[4], o[5]), pack2(o[6], o[7]));
}

__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}

// acc + (float)half(p), one rounding: mixed-precision add (sm_100+).
__device__ __forceinline__ float add_lo(float acc, uint32_t p) {
  float d;
  asm("{\n.reg .b16 l, h;\nmov.b32 {l, h}, %1;\nadd.rn.f32.bf16 %0, l, %2;\n}" : "=f"(d) : "r"(p), "f"(acc));
  return d;
}
__device__ __forceinline__ float add_hi(float acc, uint32_t p) {
  float d;
  asm("{\n.reg .b16 l, h;\nmov.b32 {l, h}, %1;\nadd.rn.f32.bf16 %0, h, %2;\n}" : "=f"(d) : "r"(p), "f"(acc));
  return d;
}

__device__ __forceinline__ void unpack8(uint4 v, float (&f)[kVec]) {
  const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    f[2 * i] = __uint_as_float(w[i] << 16);
    f[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
  }
}

// bf16 conv state: each product rounded to bf16 (mul.rn.bf16x2 = Triton's bf16 x bf16), added into an FP32
// accumulator that starts at +0, oldest tap first (Triton's `acc = 0; acc += x * w` per tap).
__device__ __forceinline__ uint4 conv_row_bf16(uint4 h0, uint4 h1, uint4 h2, uint4 cur, const uint4 (&w)[4]) {
  const uint32_t a0[4] = {h0.x, h0.y, h0.z, h0.w}, a1[4] = {h1.x, h1.y, h1.z, h1.w};
  const uint32_t a2[4] = {h2.x, h2.y, h2.z, h2.w}, a3[4] = {cur.x, cur.y, cur.z, cur.w};
  const uint32_t w0[4] = {w[0].x, w[0].y, w[0].z, w[0].w}, w1[4] = {w[1].x, w[1].y, w[1].z, w[1].w};
  const uint32_t w2[4] = {w[2].x, w[2].y, w[2].z, w[2].w}, w3[4] = {w[3].x, w[3].y, w[3].z, w[3].w};
  float acc[kVec];
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const uint32_t p0 = hmul2(a0[q], w0[q]), p1 = hmul2(a1[q], w1[q]);
    const uint32_t p2 = hmul2(a2[q], w2[q]), p3 = hmul2(a3[q], w3[q]);
    float lo = add_lo(0.0f, p0), hi = add_hi(0.0f, p0);
    lo = add_lo(lo, p1);
    hi = add_hi(hi, p1);
    lo = add_lo(lo, p2);
    hi = add_hi(hi, p2);
    acc[2 * q] = add_lo(lo, p3);
    acc[2 * q + 1] = add_hi(hi, p3);
  }
  return silu8(acc);
}

// FP32 conv weights (the KDA layer keeps them in FP32): the stock kernel multiplies the bf16 (or FP32-state)
// inputs by FP32 weights into an FP32 accumulator that starts at +0, oldest tap first, and Triton contracts
// each `acc += x * w` into one FMA. Packed FP32 pairs (fma.rn.f32x2) round each lane as the scalar FMA.
__device__ __forceinline__ uint4 conv_row_fma(const float (&h0)[kVec], const float (&h1)[kVec],
                                              const float (&h2)[kVec], const float (&cur)[kVec],
                                              const float (&w)[4][kVec]) {
  float acc[kVec];
#pragma unroll
  for (int q = 0; q < kVec / 2; ++q) {
    float2 a = __ffma2_rn(make_float2(h0[2 * q], h0[2 * q + 1]), make_float2(w[0][2 * q], w[0][2 * q + 1]),
                          make_float2(0.0f, 0.0f));
    a = __ffma2_rn(make_float2(h1[2 * q], h1[2 * q + 1]), make_float2(w[1][2 * q], w[1][2 * q + 1]), a);
    a = __ffma2_rn(make_float2(h2[2 * q], h2[2 * q + 1]), make_float2(w[2][2 * q], w[2][2 * q + 1]), a);
    a = __ffma2_rn(make_float2(cur[2 * q], cur[2 * q + 1]), make_float2(w[3][2 * q], w[3][2 * q + 1]), a);
    acc[2 * q] = a.x;
    acc[2 * q + 1] = a.y;
  }
  return silu8(acc);
}

template <typename StateT>
__device__ __forceinline__ float state_ld(const StateT* p) {
  if constexpr (std::is_same_v<StateT, float>) {
    return *p;
  } else {
    return __bfloat162float(*p);
  }
}

template <typename StateT>
__device__ __forceinline__ void state_st(StateT* p, float v) {
  if constexpr (std::is_same_v<StateT, float>) {
    *p = v;
  } else {
    *p = __float2bfloat16_rn(v);  // exact: v is a bf16 value
  }
}

struct Params {
  const __nv_bfloat16* x;  // (tokens, >= 3C) row-strided
  const void* w;           // (3C, 4) contiguous: bf16 for a bf16 state, FP32 for an FP32 state
  void* state;             // (lines, 3C, >= 3)
  const int32_t* cache_indices;
  const bool* has_initial_state;
  const int32_t* qsl;
  __nv_bfloat16* out;  // (3, tokens, C)
  int64_t ldx, st_s0, st_s1, st_s2, ci_s0, out_s0, out_s1;
  int num_seqs, parts, num_col_blocks, part_dim, null_block_id, has_null_block;
};

// Last sequence s with qsl[s] <= t (the non-empty sequence holding row t when qsl[0] <= t < qsl[num_seqs]).
__device__ __forceinline__ int find_seq(const Params& p, int t) {
  const int lane = threadIdx.x % 32;
  int seq = 0;
  for (int base = 0; base < p.num_seqs; base += 32) {
    const int s = base + lane;
    const unsigned m = __ballot_sync(0xffffffffu, s < p.num_seqs && p.qsl[s] <= t);
    if (m) seq = base + 31 - __clz(m);
    if (m != 0xffffffffu) break;
  }
  return seq;
}

// First row of part r: an even split of the rows, moved past the 2nd/3rd row of a sequence.
__device__ __forceinline__ int part_start(const Params& p, int r) {
  const int first = p.qsl[0], last = p.qsl[p.num_seqs];
  if (r >= p.parts) return last;
  int t = first + static_cast<int>(static_cast<int64_t>(last - first) * r / p.parts);
  if (t >= last) return last;
  const int s = find_seq(p, t);
  const int in_seq = t - p.qsl[s];
  if (in_seq == 1 || in_seq == 2) t = min(p.qsl[s] + 3, p.qsl[s + 1]);
  return t;
}

// Starting sequence `seq` (cache line `line`) at its first row, for this lane's 8 channels from c0: returns
// the history (positions -3..-1) = old conv state, or zeros without an initial state, then, as the only
// reader of the old state, writes the new one: the last 3 inputs, shifted in after the old state for
// sequences shorter than 3. Out of line: it runs once per sequence and CTA.
template <typename StateT>
__device__ __noinline__ void begin_seq(const __nv_bfloat16* __restrict__ x, int64_t ldx, StateT* st, int64_t st_s1,
                                       int64_t st_s2, bool load_init, int seq_start, int seqlen,
                                       float (*hist)[kVec]) {
#pragma unroll
  for (int k = 0; k < 3; ++k)
#pragma unroll
    for (int i = 0; i < kVec; ++i) hist[k][i] = load_init ? state_ld(st + i * st_s1 + k * st_s2) : 0.0f;
  float ns[3][kVec];
#pragma unroll
  for (int j = 0; j < 3; ++j) {
    const int src = seqlen - 3 + j;
    if (src >= 0) {
      unpack8(ld_global_v4(x + static_cast<int64_t>(seq_start + src) * ldx), ns[j]);
    } else {
#pragma unroll
      for (int i = 0; i < kVec; ++i) ns[j][i] = hist[j + seqlen][i];
    }
  }
#pragma unroll
  for (int j = 0; j < 3; ++j)
#pragma unroll
    for (int i = 0; i < kVec; ++i) state_st(st + i * st_s1 + j * st_s2, ns[j][i]);
}

constexpr int kStages = 4;  // ring slots per CTA (4 x 4 rows x NC x 512 B)

// Registers capped at 128 per thread (3-4 CTAs per SM): measured best over 96-170 at the VR200 shapes.
template <int NC>
constexpr int min_blocks() {
  return 65536 / ((NC + 1) * 32 * 128);
}

// FMA: FP32 weights (FP32 math, the KDA layer's case); otherwise bf16 weights with a bf16 state.
template <typename StateT, int NC, bool FMA>
__global__ void __launch_bounds__((NC + 1) * 32, min_blocks<NC>()) qkv_conv_kernel(const Params p) {
  constexpr int STAGES = kStages;
  constexpr int kCtaRowBytes = NC * kRowBytes;
  constexpr int kSlotBytes = kRowsPerSlot * kCtaRowBytes;
  extern __shared__ __align__(128) uint8_t smem_raw[];
  uint64_t* full = reinterpret_cast<uint64_t*>(smem_raw);
  uint64_t* empty = full + STAGES;
  uint8_t* ring = smem_raw + 2 * STAGES * sizeof(uint64_t);
  const int warp = threadIdx.x / 32;  // warps 0..NC-1 consume, warp NC produces
  const int lane = threadIdx.x % 32;
  const int col_block = blockIdx.x % p.num_col_blocks;
  const int part = blockIdx.x / p.num_col_blocks;

  const int r_begin = part_start(p, part);
  const int r_end = part_start(p, part + 1);
  if (r_begin >= r_end) return;  // CTA-uniform
  const int nrows = r_end - r_begin;
  const int nslots = (nrows + kRowsPerSlot - 1) / kRowsPerSlot;
  const int cta_col = col_block * (NC * kWarpCols);

  if (threadIdx.x == 0) {
#pragma unroll
    for (int s = 0; s < STAGES; ++s) {
      mbar_init(&full[s], 1);
      mbar_init(&empty[s], NC);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  if (warp == NC) {  // producer
    if (lane == 0) {
      const __nv_bfloat16* src = p.x + static_cast<int64_t>(r_begin) * p.ldx + cta_col;
      for (int k = 0; k < nslots; ++k) {
        const int s = k % STAGES;
        if (k >= STAGES) mbar_wait(&empty[s], ((k / STAGES) - 1) & 1);
        const int rows = min(kRowsPerSlot, nrows - k * kRowsPerSlot);
        mbar_expect_tx(&full[s], rows * kCtaRowBytes);
#pragma unroll
        for (int i = 0; i < kRowsPerSlot; ++i)
          if (i < rows) bulk_load(ring + s * kSlotBytes + i * kCtaRowBytes, src + i * p.ldx, kCtaRowBytes, &full[s]);
        src += kRowsPerSlot * p.ldx;
      }
    }
    return;
  }

  const int col = cta_col + warp * kWarpCols;
  const int c0 = col + lane * kVec;
  const __nv_bfloat16* xcol = p.x + col + lane * kVec;  // row 0 of x, this lane's channels
  __nv_bfloat16* ocol = p.out + (col / p.part_dim) * p.out_s0 + (col % p.part_dim) + lane * kVec;

  // Current sequence.
  int seq = find_seq(p, r_begin);
  int seq_start = p.qsl[seq], seq_end = p.qsl[seq + 1];
  int64_t line = p.cache_indices[seq * p.ci_s0];
  bool skip = p.has_null_block && line == p.null_block_id;

  // Consumes the ring rows in order. A slot whose rows all lie in the current (written) sequence runs
  // fast_row(raw bf16 x 8, out row pointer) unrolled; other slots run slow_row(t, raw) per row, which moves
  // between sequences.
  auto stream_rows = [&](auto&& fast_row, auto&& slow_row) {
    const uint8_t* base = ring + warp * kRowBytes + lane * 16;
    __nv_bfloat16* optr = ocol + static_cast<int64_t>(r_begin) * p.out_s1;
    int t = r_begin;
    for (int k = 0; k < nslots; ++k) {
      const int s = k % STAGES;
      mbar_wait(&full[s], (k / STAGES) & 1);
      const uint8_t* src = base + s * kSlotBytes;
      if (t + kRowsPerSlot <= min(seq_end, r_end) && !skip) {
#pragma unroll
        for (int i = 0; i < kRowsPerSlot; ++i) {
          fast_row(ld_shared_v4(src + i * kCtaRowBytes), optr);
          optr += p.out_s1;
        }
      } else {
#pragma unroll 1
        for (int i = 0; i < kRowsPerSlot; ++i)
          if (t + i < r_end) slow_row(t + i, ld_shared_v4(src + i * kCtaRowBytes));
        optr += kRowsPerSlot * p.out_s1;
      }
      t += kRowsPerSlot;
      __syncwarp();
      if (lane == 0) mbar_arrive(&empty[s]);
    }
  };

  float hb[3][kVec];  // begin_seq's output (rare path; local memory)
  // Moves to the sequence holding row t (t >= seq_end; t is its first row).
  auto advance = [&](int t) {
    while (p.qsl[seq + 1] <= t) ++seq;
    seq_start = p.qsl[seq];
    seq_end = p.qsl[seq + 1];
    line = p.cache_indices[seq * p.ci_s0];
    skip = p.has_null_block && line == p.null_block_id;
  };

  if constexpr (FMA) {
    float w[4][kVec];
    const float4* wv = reinterpret_cast<const float4*>(reinterpret_cast<const float*>(p.w) + c0 * 4);
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      const float4 v = wv[i];
      w[0][i] = v.x;
      w[1][i] = v.y;
      w[2][i] = v.z;
      w[3][i] = v.w;
    }
    float h[3][kVec];
    auto seq_hist = [&]() {
      begin_seq<StateT>(xcol, p.ldx,
                        reinterpret_cast<StateT*>(p.state) + line * p.st_s0 + static_cast<int64_t>(c0) * p.st_s1,
                        p.st_s1, p.st_s2, p.has_initial_state[seq], seq_start, seq_end - seq_start, hb);
#pragma unroll
      for (int k = 0; k < 3; ++k)
#pragma unroll
        for (int i = 0; i < kVec; ++i) h[k][i] = hb[k][i];
    };
    if (r_begin == seq_start) {
      if (!skip) seq_hist();
    } else {  // at least 3 rows into the sequence
#pragma unroll
      for (int k = 0; k < 3; ++k) unpack8(ld_global_v4(xcol + static_cast<int64_t>(r_begin - 3 + k) * p.ldx), h[k]);
    }
    auto step = [&](const float (&cur)[kVec]) {
#pragma unroll
      for (int i = 0; i < kVec; ++i) {
        h[0][i] = h[1][i];
        h[1][i] = h[2][i];
        h[2][i] = cur[i];
      }
    };
    stream_rows(
        [&](uint4 raw, __nv_bfloat16* o) {
          float cur[kVec];
          unpack8(raw, cur);
          st_global_cs_v4(o, conv_row_fma(h[0], h[1], h[2], cur, w));
          step(cur);
        },
        [&](int t, uint4 raw) {
          if (t >= seq_end) {
            advance(t);
            if (!skip) seq_hist();
          }
          float cur[kVec];
          unpack8(raw, cur);
          if (!skip) st_global_cs_v4(ocol + static_cast<int64_t>(t) * p.out_s1, conv_row_fma(h[0], h[1], h[2], cur, w));
          step(cur);
        });
  } else {
    // bf16 weights, (channel, tap) contiguous: 64 B for this lane's 8 channels -> per-tap bf16x2 pairs
    // (channel 2q in the low half).
    uint4 w[4];
    {
      const uint4* wv = reinterpret_cast<const uint4*>(reinterpret_cast<const __nv_bfloat16*>(p.w) + c0 * 4);
      uint32_t v[4][4];
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const uint4 t = wv[q];  // {c(2q) t0|t1, c(2q) t2|t3, c(2q+1) t0|t1, c(2q+1) t2|t3}
        v[0][q] = __byte_perm(t.x, t.z, 0x5410);
        v[1][q] = __byte_perm(t.x, t.z, 0x7632);
        v[2][q] = __byte_perm(t.y, t.w, 0x5410);
        v[3][q] = __byte_perm(t.y, t.w, 0x7632);
      }
#pragma unroll
      for (int j = 0; j < 4; ++j) w[j] = make_uint4(v[j][0], v[j][1], v[j][2], v[j][3]);
    }
    // History as raw bf16 x 8 (a bf16 state holds bf16 values: the float round trip is exact).
    auto to_raw = [](const float (&f)[kVec]) {
      return make_uint4(pack2(f[0], f[1]), pack2(f[2], f[3]), pack2(f[4], f[5]), pack2(f[6], f[7]));
    };
    uint4 h[3];
    auto seq_hist = [&]() {
      begin_seq<StateT>(xcol, p.ldx,
                        reinterpret_cast<StateT*>(p.state) + line * p.st_s0 + static_cast<int64_t>(c0) * p.st_s1,
                        p.st_s1, p.st_s2, p.has_initial_state[seq], seq_start, seq_end - seq_start, hb);
#pragma unroll
      for (int k = 0; k < 3; ++k) h[k] = to_raw(hb[k]);
    };
    if (r_begin == seq_start) {
      if (!skip) seq_hist();
    } else {
#pragma unroll
      for (int k = 0; k < 3; ++k) h[k] = ld_global_v4(xcol + static_cast<int64_t>(r_begin - 3 + k) * p.ldx);
    }
    stream_rows(
        [&](uint4 cur, __nv_bfloat16* o) {
          st_global_cs_v4(o, conv_row_bf16(h[0], h[1], h[2], cur, w));
          h[0] = h[1];
          h[1] = h[2];
          h[2] = cur;
        },
        [&](int t, uint4 cur) {
          if (t >= seq_end) {
            advance(t);
            if (!skip) seq_hist();
          }
          if (!skip) st_global_cs_v4(ocol + static_cast<int64_t>(t) * p.out_s1, conv_row_bf16(h[0], h[1], h[2], cur, w));
          h[0] = h[1];
          h[1] = h[2];
          h[2] = cur;
        });
  }
}

template <typename StateT, int NC, bool FMA>
void launch(const Params& p, int64_t tokens, cudaStream_t stream) {
  constexpr int smem = kStages * (2 * sizeof(uint64_t) + kRowsPerSlot * NC * kRowBytes);
  auto k = qkv_conv_kernel<StateT, NC, FMA>;
  static const int ctas_per_sm = [&] {
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    int n = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, k, (NC + 1) * 32, smem);
    return std::max(n, 1);
  }();
  int dev = 0, sms = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  // One wave of resident CTAs, each with at least 16 rows.
  Params q = p;
  const int64_t by_rows = (tokens + 15) / 16;
  q.parts = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(by_rows, sms * ctas_per_sm / p.num_col_blocks)));
  k<<<q.parts * p.num_col_blocks, (NC + 1) * 32, smem, stream>>>(q);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename StateT, bool FMA>
void dispatch(const Params& p, int64_t tokens, int nc, cudaStream_t stream) {
  if (nc == 4) return launch<StateT, 4, FMA>(p, tokens, stream);
  if (nc == 3) return launch<StateT, 3, FMA>(p, tokens, stream);
  if (nc == 2) return launch<StateT, 2, FMA>(p, tokens, stream);
  TORCH_CHECK(false, "k3 qkv conv: unsupported warps per CTA ", nc);
}

// out = conv(x) as (3, tokens, part_dim); conv_states updated in place.
void qkv_conv(torch::Tensor const& x, torch::Tensor const& weight, torch::Tensor& conv_states,
              torch::Tensor const& query_start_loc, torch::Tensor const& cache_indices,
              torch::Tensor const& has_initial_state, torch::Tensor& out, int64_t null_block_id, int64_t warps) {
  const int64_t tokens = x.size(0);
  const int64_t dim = 3 * out.size(2);
  const int num_seqs = query_start_loc.size(0) - 1;
  if (tokens == 0 || num_seqs == 0) return;
  Params p;
  p.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  p.w = weight.data_ptr();
  p.state = conv_states.data_ptr();
  p.cache_indices = cache_indices.data_ptr<int32_t>();
  p.has_initial_state = has_initial_state.data_ptr<bool>();
  p.qsl = query_start_loc.data_ptr<int32_t>();
  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.ldx = x.stride(0);
  p.st_s0 = conv_states.stride(0);
  p.st_s1 = conv_states.stride(1);
  p.st_s2 = conv_states.stride(2);
  p.ci_s0 = cache_indices.stride(0);
  p.out_s0 = out.stride(0);
  p.out_s1 = out.stride(1);
  p.num_seqs = num_seqs;
  p.parts = 1;
  p.part_dim = static_cast<int>(out.size(2));
  TORCH_CHECK(dim % (warps * kWarpCols) == 0, "k3 qkv conv: channels not a multiple of the CTA width");
  p.num_col_blocks = static_cast<int>(dim / (warps * kWarpCols));
  p.null_block_id = static_cast<int>(null_block_id);
  p.has_null_block = null_block_id >= 0;
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(x.get_device()).stream();
  const bool st32 = conv_states.scalar_type() == at::kFloat;
  const bool w32 = weight.scalar_type() == at::kFloat;
  TORCH_CHECK(w32 || (!st32 && weight.scalar_type() == at::kBFloat16), "k3 qkv conv: weight/state dtypes");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && out.scalar_type() == at::kBFloat16, "k3 qkv conv: bf16 x/out");
  TORCH_CHECK(query_start_loc.scalar_type() == at::kInt && cache_indices.scalar_type() == at::kInt &&
                  has_initial_state.scalar_type() == at::kBool,
              "k3 qkv conv: metadata dtypes");
  if (st32) return dispatch<float, true>(p, tokens, static_cast<int>(warps), stream);
  if (w32) return dispatch<__nv_bfloat16, true>(p, tokens, static_cast<int>(warps), stream);
  return dispatch<__nv_bfloat16, false>(p, tokens, static_cast<int>(warps), stream);
}

}  // namespace k3conv

TORCH_LIBRARY_FRAGMENT(k3_kda_conv, m) {
  m.def(
      "qkv_conv(Tensor x, Tensor weight, Tensor(a!) conv_states, Tensor query_start_loc, Tensor cache_indices, "
      "Tensor has_initial_state, Tensor(b!) out, int null_block_id, int warps) -> ()");
}

TORCH_LIBRARY_IMPL(k3_kda_conv, CUDA, m) { m.impl("qkv_conv", &k3conv::qkv_conv); }
