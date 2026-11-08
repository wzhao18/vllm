// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Kimi K3.1 specialized decode: a candidate DSA sparse MLA core for the wide rows
// (DCP decode rows >= 64, per-tensor FP8 MLA KV), a persistent tcgen05 kernel with
// the interface of FlashInfer's trtllm-gen sparse decode (+ its empty-row fixup) on
// the gathered q mailbox (not wired in: trtllm-gen measured faster, see
// ../wide_core.py):
//
//   in   q   [rows, q_heads, 576] FP8 (the first `heads` rows of each are real)
//        sel [rows, >= width] int32: this rank's flat KV rows, a compacted prefix
//        cnt [rows] int32: its length; kv [N, 576] FP8 flat KV rows
//   out  latent [rows, out_heads, 512] bf16 (heads < `heads` written), and
//        lse [rows, lse_heads] fp32 in trtllm-gen's units (base 2 of the scaled
//        logits); a row without entries gets latent 0 and LSE -inf (the stock
//        _finish_dcp_partials identity).
//
// Math (the stock trtllm-gen FP8 sparse MLA decode's, unsplit rows): S = Q K^T
// (E4M3 x E4M3, FP32 accumulation) scaled by bmm1 * log2(e); online softmax in base 2
// over KV tiles of 128 keys (the running row max anchors P); P = E4M3(2^(s - m) *
// 448); O += P V (E4M3 x E4M3, FP32 in TMEM), rescaled by 2^(m_old - m_new) when the
// row max grows; out = BF16(O * bmm2 / 448 / l), l the FP32 sum of the unquantized P;
// LSE = m + log2(l).
//
// Work: item (row t, latent half h): all gathered heads of the row as the MMA's
// M = 128 (rows >= heads are don't-care), this half's 256 latent dims in PV. Every
// CTA (1 per SM) walks items blockIdx.x, + gridDim.x, ... (gridDim even: a CTA keeps
// its half, and the two halves of a row run side by side and share its Q / KV in L2).
//
// Warp roles: warps 0-15 compute (one thread issues the MMAs; all 512 run the
// softmax: thread (TMEM lane quarter w & 3, column quarter w >> 2)); warp 16 is the
// TMA producer: the item's Q (2D tile loads of the q mailbox) and its KV chunks of
// 64 keys (tile::gather4 of the selected rows; rows past the count are out of bounds,
// zero-filled) into a ring of 4 chunk buffers,
// running ahead of the consumers by mbarriers (full: TMA bytes; empty: the MMA
// commit after the last read). Shared tiles are K-major as QK's operands: per row
// 4 x 128 B latent column blocks (128 B swizzle) and one 64 B rope block (64 B
// swizzle); PV reads V MN-major from the same chunk buffers' latent blocks. The next
// live item's first QK is issued before an item's epilogue (S and O are separate
// TMEM columns), so the epilogue's stores overlap it.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

namespace k31dsaw {

constexpr int kCompute = 512;          // 16 compute warps
constexpr int kThreads = kCompute + 32;  // + the producer warp
constexpr int kM = 128;                // MMA rows (gathered heads, padded)
constexpr int kChunk = 64;             // keys per KV chunk buffer
constexpr int kTile = 128;             // keys per softmax tile (2 chunks)
constexpr int kRowB = 576;             // FP8 Q / KV row bytes
constexpr int kLat = 4;                // 128 B latent column blocks per row (SW128)
constexpr int kRope = 64;              // + one 64 B rope column block (SW64)
constexpr int kHalf = 256;             // latent dims per item (PV N)
constexpr int kNBuf = 4;               // KV chunk buffers
constexpr int kMaxHeads = 96;          // real heads (Q rows loaded)
constexpr int kTmemCols = 512;         // O [0, 256) | S [256, 384)
constexpr int kOCol = 0, kSCol = 256;
constexpr float kPScale = 448.f;       // E4M3 max: P at 448 of the running row max
// The thread that issues every MMA (and so owns every tcgen05.commit): lane 0 of warp
// 3, whose TMEM lanes 96..127 are pad rows at 96 heads, so it has no epilogue work and
// its wait for the next item's data overlaps the other warps' epilogue.
constexpr int kMmaTid = 96;

// shared memory plan (bytes; swizzled blocks 1024-aligned). Q latent block c holds
// `heads` rows x 128 B at c * heads * 128, its rope block `heads` x 64 B after them;
// the MMA's rows >= heads read the next block / P (finite don't-care rows).
constexpr int kQ = 0;
constexpr int kQBytes = kMaxHeads * (kLat * 128 + kRope);  // 55296
constexpr int kP = kQBytes;                            // [16 groups][8][128] P, no swizzle
constexpr int kPBytes = (kM / 8) * 8 * 128;            // 16384
constexpr int kRed = kP + kPBytes;                     // [4][128] f32
constexpr int kKV = kRed + 4 * kM * 4;                 // 73728
static_assert(kKV % 1024 == 0, "KV ring alignment");
constexpr int kColBytes = kChunk * 128;                // 8192: one latent block of a chunk
constexpr int kRopeOff = kLat * kColBytes;             // 32768: its rope block (64 x 64 B)
constexpr int kChunkBytes = kRopeOff + kChunk * kRope;  // 36864
constexpr int kBar = kKV + kNBuf * kChunkBytes;        // 221184
// barriers: fullQ, emptyQ, full[3], empty[3], S, O; TMEM address
constexpr int kBarFullQ = 0, kBarEmptyQ = 8, kBarFull = 16, kBarEmpty = 16 + 8 * kNBuf;
constexpr int kBarS = 16 + 16 * kNBuf, kBarO = kBarS + 8, kTmemAddr = kBarO + 8;
constexpr int kItems = kBar + 128;
constexpr int kMaxItems = 64;                          // items per CTA (counts staged)
constexpr int kSmem = kItems + kMaxItems * 4;
static_assert(kSmem <= 227 * 1024, "wide core shared memory");
constexpr uint32_t kChunkTx = kChunkBytes;             // gather bytes per chunk
constexpr uint32_t kSw128 = 2, kSw64 = 4;              // descriptor layout types

struct __align__(64) Params {
  CUtensorMap qmap;    // q mailbox rows [rows * q_heads, 576], box 128 B x heads, SW128
  CUtensorMap qrope;   // the same rows, box 64 B x heads, SW64 (columns 512 ..)
  CUtensorMap kvmap;   // KV rows [N, 576], box 128 B x 1 (gather4), SW128
  CUtensorMap kvrope;  // the same rows, box 64 B x 1 (gather4), SW64
  const int* sel;
  int64_t sel_stride;  // elements
  int width;
  const int* cnt;
  int q_heads, nkv;
  float s2scale, oscale;  // bmm1 * log2(e); bmm2 / kPScale
  __nv_bfloat16* out;
  int64_t out_row_stride;  // elements
  float* lse;
  int64_t lse_row_stride;  // elements
  int heads, T;
  long long* dbg;  // optional per-item phase timestamps of CTA dbg_cta (benches)
  int dbg_cta;
};

// ------------------------------------------------------------------ primitives
__device__ __forceinline__ long long gtime() {
  long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void fence_async_smem() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
__device__ __forceinline__ void compute_sync() {  // the 16 compute warps only
  asm volatile("bar.sync 1, %0;" ::"n"(kCompute) : "memory");
}
__device__ __forceinline__ void mbar_init(uint32_t bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count));
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  uint32_t ok;
  do {
    asm volatile(
        "{\n\t.reg .pred P;\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P;\n\t}"
        : "=r"(ok)
        : "r"(bar), "r"(parity)
        : "memory");
  } while (!ok);
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes)
               : "memory");
}
__device__ __forceinline__ void tma_load_2d(uint32_t dst, const CUtensorMap* map, int x, int y,
                                            uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3}], [%4];" ::"r"(dst),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(x), "r"(y), "r"(bar)
      : "memory");
}
__device__ __forceinline__ void tma_gather4(uint32_t dst, const CUtensorMap* map, int x, int y0,
                                            int y1, int y2, int y3, uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];" ::"r"(dst),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(x), "r"(y0), "r"(y1), "r"(y2), "r"(y3), "r"(bar)
      : "memory");
}
__device__ __forceinline__ void tc_fence_before() {
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}
__device__ __forceinline__ void tc_fence_after() {
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}
// SM100 shared memory descriptor (version 1): layout 0 = no swizzle, 2 = 128 B swizzle
__device__ __forceinline__ uint64_t sdesc(uint32_t addr, uint32_t lbo, uint32_t sbo,
                                          uint32_t layout) {
  uint64_t d = 0;
  d |= (uint64_t)((addr >> 4) & 0x3FFF);
  d |= (uint64_t)((lbo >> 4) & 0x3FFF) << 16;
  d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
  d |= (uint64_t)1 << 46;
  d |= (uint64_t)layout << 61;
  return d;
}
// kind::f8f6f4 instruction descriptor: E4M3 x E4M3 -> F32, M x N
__host__ __device__ constexpr uint32_t idesc(int m, int n, bool b_mn_major) {
  return (1u << 4) | ((b_mn_major ? 1u : 0u) << 16) | ((uint32_t)(n >> 3) << 17) |
         ((uint32_t)(m >> 4) << 24);
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
__device__ __forceinline__ void tmem_ld64(uint32_t taddr, float* v) {  // one wait for 64 columns
  uint32_t r[64];
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,"
      "%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]),
        "=r"(r[7]), "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]),
        "=r"(r[14]), "=r"(r[15]), "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]),
        "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]), "=r"(r[25]),
        "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
      : "r"(taddr));
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,"
      "%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
      : "=r"(r[32]), "=r"(r[33]), "=r"(r[34]), "=r"(r[35]), "=r"(r[36]), "=r"(r[37]),
        "=r"(r[38]), "=r"(r[39]), "=r"(r[40]), "=r"(r[41]), "=r"(r[42]), "=r"(r[43]),
        "=r"(r[44]), "=r"(r[45]), "=r"(r[46]), "=r"(r[47]), "=r"(r[48]), "=r"(r[49]),
        "=r"(r[50]), "=r"(r[51]), "=r"(r[52]), "=r"(r[53]), "=r"(r[54]), "=r"(r[55]),
        "=r"(r[56]), "=r"(r[57]), "=r"(r[58]), "=r"(r[59]), "=r"(r[60]), "=r"(r[61]),
        "=r"(r[62]), "=r"(r[63])
      : "r"(taddr + 32));
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
  for (int i = 0; i < 64; ++i) v[i] = __uint_as_float(r[i]);
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
__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ uint32_t bf2(float a, float b) {
  __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&v);
}
// offset of (row r, 16 B chunk c) in a K-major no-swizzle layout with K extent kc chunks
__device__ __forceinline__ int cm_off(int r, int c, int kc) {
  return (r >> 3) * (kc * 128) + c * 128 + (r & 7) * 16;
}

// ------------------------------------------------------------------ kernel
#define K31W_T(slot) \
  if (p.dbg && tid == 0 && (int)blockIdx.x == p.dbg_cta && k < 32) p.dbg[k * 8 + (slot)] = gtime()

__device__ __forceinline__ int item_row(int k) { return (int)(blockIdx.x + k * gridDim.x) >> 1; }

// The producer warp: Q of every live item, then its KV chunks, through the ring. The
// selected slots are loaded a group of 4 chunks ahead (lane l holds rows 2 l, 2 l + 1
// of each chunk), so their load latency overlaps the TMA issue and ring waits.
constexpr int kGroup = 4;

__device__ __forceinline__ void load_group(const Params& p, const int* sN, int k, int j0,
                                           int (&dst)[2 * kGroup]) {
  const int lane = threadIdx.x & 31;
  const int n = sN[k];
  const int* srow = p.sel + (int64_t)item_row(k) * p.sel_stride;
#pragma unroll
  for (int c = 0; c < kGroup; ++c) {
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      const int r = (j0 + c) * kChunk + 2 * lane + e;
      dst[2 * c + e] = r < n ? __ldg(srow + r) : -1;
    }
  }
}

__device__ __forceinline__ void produce(const Params& p, uint8_t* smem, const int* sN, int nitems) {
  const int lane = threadIdx.x & 31;
  const uint32_t base = smem_u32(smem);
  const uint32_t bars = base + kBar;
  const int qblock = p.heads * 128;
  auto next_live = [&](int k) {
    while (k < nitems && sN[k] == 0) ++k;
    return k;
  };
  int g = 0, qi = 0;
  int k = next_live(0), j0 = 0;
  int cur[2 * kGroup], nxt[2 * kGroup];
  if (k < nitems) load_group(p, sN, k, 0, cur);
  while (k < nitems) {
    const int n = sN[k], t = item_row(k);
    const int nchunks = (n + kChunk - 1) / kChunk;
    // the next group's slots (this item's next 4 chunks, or the next live item's first)
    int kn = k, jn = j0 + kGroup;
    if (jn >= nchunks) {
      kn = next_live(k + 1);
      jn = 0;
    }
    if (kn < nitems) load_group(p, sN, kn, jn, nxt);
    if (j0 == 0) {  // Q: free once the previous live item's last QK completed
      if (qi > 0) mbar_wait(bars + kBarEmptyQ, (qi - 1) & 1);
      if (lane == 0) {
        if (p.dbg && (int)blockIdx.x == p.dbg_cta && k < 32) p.dbg[k * 8 + 6] = gtime();
        mbar_expect_tx(bars + kBarFullQ, (uint32_t)(p.heads * (kLat * 128 + kRope)));
#pragma unroll
        for (int c = 0; c < kLat; ++c)
          tma_load_2d(base + kQ + c * qblock, &p.qmap, 128 * c, t * p.q_heads, bars + kBarFullQ);
        tma_load_2d(base + kQ + kLat * qblock, &p.qrope, 512, t * p.q_heads, bars + kBarFullQ);
      }
      ++qi;
    }
#pragma unroll
    for (int c = 0; c < kGroup; ++c) {
      if (j0 + c < nchunks) {
        const int s0 = cur[2 * c] < 0 ? p.nkv : cur[2 * c];  // (out of bounds: zero fill)
        const int s1 = cur[2 * c + 1] < 0 ? p.nkv : cur[2 * c + 1];
        const int b = g % kNBuf;
        if (g >= kNBuf) mbar_wait(bars + kBarEmpty + 8 * b, ((g / kNBuf) - 1) & 1);
        if (lane == 0) mbar_expect_tx(bars + kBarFull + 8 * b, kChunkTx);
        __syncwarp();
        // gather group gr (rows 4 gr .. 4 gr + 3): lane 2 gr issues latent blocks 0..2,
        // lane 2 gr + 1 latent block 3 and the rope block
        const int gr = lane >> 1;
        const int y0 = __shfl_sync(0xffffffffu, s0, 2 * gr);
        const int y1 = __shfl_sync(0xffffffffu, s1, 2 * gr);
        const int y2 = __shfl_sync(0xffffffffu, s0, 2 * gr + 1);
        const int y3 = __shfl_sync(0xffffffffu, s1, 2 * gr + 1);
        const uint32_t dst = base + kKV + b * kChunkBytes + gr * 4 * 128;
        const uint32_t full = bars + kBarFull + 8 * b;
        if (lane & 1) {
          tma_gather4(dst + 3 * kColBytes, &p.kvmap, 384, y0, y1, y2, y3, full);
          tma_gather4(base + kKV + b * kChunkBytes + kRopeOff + gr * 4 * kRope, &p.kvrope, 512,
                      y0, y1, y2, y3, full);
        } else {
#pragma unroll
          for (int cc = 0; cc < 3; ++cc)
            tma_gather4(dst + cc * kColBytes, &p.kvmap, 128 * cc, y0, y1, y2, y3, full);
        }
        ++g;
      }
    }
    k = kn;
    j0 = jn;
#pragma unroll
    for (int e = 0; e < 2 * kGroup; ++e) cur[e] = nxt[e];
  }
}

__global__ void __launch_bounds__(kThreads, 1) wide_kernel(const __grid_constant__ Params p) {
  extern __shared__ __align__(1024) uint8_t smem[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int half = blockIdx.x & 1;
  const int nitems = (2 * p.T - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x;
  uint8_t* sP = smem + kP;
  float* sRed = reinterpret_cast<float*>(smem + kRed);
  const uint32_t base = smem_u32(smem);
  const uint32_t bars = base + kBar;
  uint32_t* sTmem = reinterpret_cast<uint32_t*>(smem + kBar + kTmemAddr);
  int* sN = reinterpret_cast<int*>(smem + kItems);

  if (warp == 0) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(
                     smem_u32(sTmem)),
                 "n"(kTmemCols)
                 : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
  }
  if (tid == 32) {
    for (int b = 0; b < 2 + 2 * kNBuf + 2; ++b) mbar_init(bars + 8 * b, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (warp == 16 && lane == 0) {
    asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(&p.qmap)) : "memory");
    asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(&p.kvmap)) : "memory");
    asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(&p.qrope)) : "memory");
    asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(&p.kvrope)) : "memory");
  }
  // the previous kernel's q, slots and counts
  pdl_wait();
  if (tid < nitems) {  // (one load latency for every item of the CTA)
    const int c = __ldg(p.cnt + item_row(tid));
    sN[tid] = c < 0 ? 0 : (c < p.width ? c : p.width);
  }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  if (warp == 16) {
    produce(p, smem, sN, nitems);
    return;
  }
  const uint32_t tmem = *sTmem;
  const int qblock = p.heads * 128;
  const int row = 32 * (warp & 3) + lane, ch = warp >> 2;
  const uint32_t lane_base = (uint32_t)((warp & 3) * 32) << 16;
  constexpr uint32_t kIdQK = idesc(kM, kChunk, false);
  constexpr uint32_t kIdPV = idesc(kM, kHalf, true);
  uint32_t phS = 0, phO = 0;
  int g = 0, qi = 0;  // chunk / live item sequence (as the producer's)

  // QK of tile i of an item (n keys) whose chunks start at ga, the qi-th live item:
  // S[128, 128] = Q K^T (chunk a -> columns 0..63, chunk b -> 64..127). kMmaTid
  // waits for the data and issues; K steps of 32 B: 16 in the SW128 latent blocks,
  // 2 in the SW64 rope block. Its commit completes barS (and after the item's last
  // QK frees Q).
  auto issue_qk = [&](int n, int i, int ga, int q_seq, int kk) {
    const int nchunks = (n + kChunk - 1) / kChunk, ntiles = (n + kTile - 1) / kTile;
    const bool has_b = 2 * i + 1 < nchunks;
    const int ba = ga % kNBuf, bb = (ga + 1) % kNBuf;
    if (i == 0) mbar_wait(bars + kBarFullQ, q_seq & 1);
    mbar_wait(bars + kBarFull + 8 * ba, (ga / kNBuf) & 1);
    if (has_b) mbar_wait(bars + kBarFull + 8 * bb, ((ga + 1) / kNBuf) & 1);
    if (i == 0 && p.dbg && (int)blockIdx.x == p.dbg_cta && kk < 32) p.dbg[kk * 8 + 1] = gtime();
    tc_fence_after();
#pragma unroll 1
    for (int cb = 0; cb < (has_b ? 2 : 1); ++cb) {
      const uint32_t buf = base + kKV + (cb ? bb : ba) * kChunkBytes;
#pragma unroll
      for (int ks = 0; ks < kRowB / 32; ++ks) {
        uint64_t a, b;
        if (ks < 16) {
          const int c = ks >> 2, w = (ks & 3) * 32;
          a = sdesc(base + kQ + c * qblock + w, 16, 1024, kSw128);
          b = sdesc(buf + c * kColBytes + w, 16, 1024, kSw128);
        } else {
          const int w = (ks - 16) * 32;
          a = sdesc(base + kQ + kLat * qblock + w, 16, 512, kSw64);
          b = sdesc(buf + kRopeOff + w, 16, 512, kSw64);
        }
        umma(tmem + kSCol + 64 * cb, a, b, kIdQK, ks > 0 ? 1u : 0u);
      }
    }
    umma_commit(bars + kBarS);
    if (i == ntiles - 1) umma_commit(bars + kBarEmptyQ);  // (Q read for the last time)
  };
  auto next_live = [&](int k) {
    while (k < nitems && sN[k] == 0) ++k;
    return k;
  };

  {  // the first live item's first QK
    const int k0 = next_live(0);
    if (tid == kMmaTid && k0 < nitems) issue_qk(sN[k0], 0, 0, 0, k0);
  }
  for (int k = 0; k < nitems; ++k) {
    const int n = sN[k], t = item_row(k);
    K31W_T(0);
    const int nchunks = (n + kChunk - 1) / kChunk;
    const int ntiles = (n + kTile - 1) / kTile;
    float m_run = -INFINITY, l_run = 0.f;
    for (int i = 0; i < ntiles; ++i) {
      const int ga = g + 2 * i;
      const bool has_b = 2 * i + 1 < nchunks;
      const int ba = ga % kNBuf, bb = (ga + 1) % kNBuf;
      if (i > 0 && tid == kMmaTid) issue_qk(n, i, ga, qi, k);  // (tile 0's was issued before)
      mbar_wait(bars + kBarS, phS);
      if (i == 0) K31W_T(2);
      phS ^= 1;
      tc_fence_after();
      // ---- softmax of the tile: thread (row, quarter ch) holds columns 32 ch ..
      const int valid = min(kTile, n - kTile * i);
      float sv[32];
      tmem_ld32(tmem + lane_base + kSCol + 32 * ch, sv);
      float mx = -INFINITY;
#pragma unroll
      for (int j = 0; j < 32; ++j) {
        if (row >= p.heads) sv[j] = 0.f;  // (pad rows: finite don't-care)
        if (32 * ch + j >= valid) sv[j] = -INFINITY;
        mx = fmaxf(mx, sv[j]);
      }
      sRed[ch * 128 + row] = mx;
      compute_sync();
      const float tmax =
          fmaxf(fmaxf(sRed[row], sRed[128 + row]), fmaxf(sRed[256 + row], sRed[384 + row])) *
          p.s2scale;  // finite: valid >= 1
      const float m_new = fmaxf(m_run, tmax);
      const float alpha = m_run == -INFINITY ? 0.f : ex2(m_run - m_new);
      float sum = 0.f;
      uint32_t pw[8];
#pragma unroll
      for (int j = 0; j < 32; j += 4) {
        const float p0 = ex2(fmaf(sv[j], p.s2scale, -m_new));
        const float p1 = ex2(fmaf(sv[j + 1], p.s2scale, -m_new));
        const float p2 = ex2(fmaf(sv[j + 2], p.s2scale, -m_new));
        const float p3 = ex2(fmaf(sv[j + 3], p.s2scale, -m_new));
        const __nv_fp8x4_e4m3 q4(
            make_float4(p0 * kPScale, p1 * kPScale, p2 * kPScale, p3 * kPScale));
        pw[j / 4] = *reinterpret_cast<const uint32_t*>(&q4);
        sum += (p0 + p1) + (p2 + p3);
      }
      // P row, keys 32 ch .. 32 ch + 31 -> K-major core matrices 2 ch, 2 ch + 1
      *reinterpret_cast<uint4*>(sP + cm_off(row, 2 * ch, 8)) = make_uint4(pw[0], pw[1], pw[2], pw[3]);
      *reinterpret_cast<uint4*>(sP + cm_off(row, 2 * ch + 1, 8)) = make_uint4(pw[4], pw[5], pw[6], pw[7]);
      // rescale the running O (64 columns of the warp's rows; PV(i - 1) is done)
      if (i > 0 && __any_sync(0xffffffffu, alpha < 1.f)) {
#pragma unroll 1
        for (int kk = 0; kk < 2; ++kk) {
          float o[32];
          const uint32_t col = tmem + lane_base + kOCol + 64 * ch + 32 * kk;
          tmem_ld32(col, o);
#pragma unroll
          for (int j = 0; j < 32; ++j) o[j] *= alpha;
          tmem_st32(col, o);
        }
      }
      compute_sync();  // (every quarter read the max before the sums overwrite it)
      sRed[ch * 128 + row] = sum;
      fence_async_smem();
      tc_fence_before();
      compute_sync();
      l_run = l_run * alpha + ((sRed[row] + sRed[128 + row]) + (sRed[256 + row] + sRed[384 + row]));
      m_run = m_new;
      if (i == 0) K31W_T(3);
      // ---- PV: O[128, 256] += P[128, 128] V[128, 256] (K steps of 32 keys); V is
      // MN-major in the chunk's latent blocks 2 h, 2 h + 1 (atoms 8192 B apart along
      // N, 8-key groups 1024 B apart along K)
      if (tid == kMmaTid) {
        tc_fence_after();
        const uint32_t pa = base + kP;
#pragma unroll 1
        for (int ks = 0; ks < (has_b ? 4 : 2); ++ks) {
          const uint32_t buf = base + kKV + ((ks >> 1) ? bb : ba) * kChunkBytes;
          const uint64_t a = sdesc(pa + ks * 256, 128, 8 * 128, 0);
          const uint64_t b = sdesc(buf + 2 * half * kColBytes + (ks & 1) * 32 * 128, kColBytes,
                                   1024, kSw128);
          umma(tmem + kOCol, a, b, kIdPV, (i > 0 || ks > 0) ? 1u : 0u);
        }
        umma_commit(bars + kBarEmpty + 8 * ba);  // (the chunks are read for the last time)
        if (has_b) umma_commit(bars + kBarEmpty + 8 * bb);
        if (i == ntiles - 1) umma_commit(bars + kBarO);
      }
      compute_sync();  // (sRed / sP are rewritten by the next tile)
    }
    if (ntiles > 0) {
      mbar_wait(bars + kBarO, phO);
      phO ^= 1;
      tc_fence_after();
      ++qi;
    }
    g += nchunks;
    K31W_T(4);
    if (ntiles > 0) {  // the next live item's first QK runs under this epilogue
      const int kn = next_live(k + 1);
      if (tid == kMmaTid && kn < nitems) issue_qk(sN[kn], 0, g, qi, kn);
    }
    // ---- epilogue: O * bmm2 / kPScale / l -> bf16 latent; LSE
    if (row < p.heads) {
      const bool live = ntiles > 0 && l_run > 0.f;
      const float r = live ? p.oscale / l_run : 0.f;
      __nv_bfloat16* orow = p.out + (int64_t)t * p.out_row_stride + row * 512 + half * kHalf + 64 * ch;
      float o[64];
      if (ntiles > 0) {
        tmem_ld64(tmem + lane_base + kOCol + 64 * ch, o);
      } else {
#pragma unroll
        for (int j = 0; j < 64; ++j) o[j] = 0.f;
      }
#pragma unroll
      for (int j = 0; j < 64; j += 8) {
        *reinterpret_cast<uint4*>(orow + j) =
            make_uint4(bf2(o[j] * r, o[j + 1] * r), bf2(o[j + 2] * r, o[j + 3] * r),
                       bf2(o[j + 4] * r, o[j + 5] * r), bf2(o[j + 6] * r, o[j + 7] * r));
      }
      if (half == 0 && ch == 0) {
        p.lse[(int64_t)t * p.lse_row_stride + row] = live ? m_run + log2f(l_run) : -INFINITY;
      }
    }  // (heads % 32 == 0: the branch is warp-uniform, as tcgen05.ld needs)
    K31W_T(5);
    tc_fence_before();
    compute_sync();  // (O is reused by the next item's PV)
    tc_fence_after();
  }
  if (warp == 0) {
    tc_fence_after();
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem),
                 "n"(kTmemCols)
                 : "memory");
  }
}

// ------------------------------------------------------------------ host
using EncodeTiled = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*,
                                 const cuuint64_t*, const cuuint64_t*, const cuuint32_t*,
                                 const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                                 CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

EncodeTiled encode_fn() {
  static EncodeTiled fn = nullptr;
  if (fn == nullptr) {
    void* ptr = nullptr;
    cudaDriverEntryPointQueryResult status;
    C10_CUDA_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &ptr, cudaEnableDefault, &status));
    TORCH_CHECK(status == cudaDriverEntryPointSuccess && ptr != nullptr,
                "wide core: cuTensorMapEncodeTiled unavailable");
    fn = reinterpret_cast<EncodeTiled>(ptr);
  }
  return fn;
}

// 2D uint8 rows [rows, 576] (row pitch 576 B), box box_bytes (128: SW128, 64: SW64) x
// box_rows
void encode_rows(CUtensorMap* map, const void* ptr, int64_t rows, int box_rows, int box_bytes) {
  const cuuint64_t dims[2] = {(cuuint64_t)kRowB, (cuuint64_t)rows};
  const cuuint64_t strides[1] = {(cuuint64_t)kRowB};
  const cuuint32_t box[2] = {(cuuint32_t)box_bytes, (cuuint32_t)box_rows};
  const cuuint32_t estr[2] = {1, 1};
  const CUresult r = encode_fn()(map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(ptr), dims,
                                 strides, box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                 box_bytes == 128 ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_64B,
                                 CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                 CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "wide core: cuTensorMapEncodeTiled failed (", (int)r, ")");
}

void run(torch::Tensor q, torch::Tensor sel, torch::Tensor cnt, torch::Tensor kv, double bmm1,
         double bmm2, torch::Tensor out, torch::Tensor lse, int64_t heads, int64_t T, int64_t sms,
         int64_t pdl, c10::optional<torch::Tensor> dbg, int64_t dbg_cta) {
  if (T == 0) return;
  TORCH_CHECK(heads > 0 && heads <= kMaxHeads && heads % 32 == 0,
              "wide core: heads (a multiple of 32, at most 96)");
  TORCH_CHECK(q.dim() == 3 && q.size(-1) == kRowB && q.element_size() == 1 && q.size(1) >= heads &&
                  q.stride(2) == 1 && q.stride(1) == kRowB && q.stride(0) == q.size(1) * kRowB,
              "wide core: q [rows, q_heads >= heads, 576] FP8 bytes, contiguous");
  TORCH_CHECK(T <= q.size(0) && T <= sel.size(0) && T <= cnt.numel() && T <= out.size(0) &&
                  T <= lse.size(0),
              "wide core: rows");
  TORCH_CHECK(sel.dtype() == torch::kInt32 && sel.stride(1) == 1 && cnt.dtype() == torch::kInt32 &&
                  cnt.is_contiguous(),
              "wide core: sel / cnt int32");
  TORCH_CHECK(kv.element_size() == 1 && kv.size(-1) == kRowB && kv.stride(-2) == kRowB &&
                  kv.numel() / kRowB < (1LL << 31) - 1,
              "wide core: kv [N, 576] FP8 rows");
  TORCH_CHECK(out.scalar_type() == torch::kBFloat16 && out.dim() == 3 && out.size(1) >= heads &&
                  out.size(2) == 512 && out.stride(2) == 1 && out.stride(1) == 512,
              "wide core: out [rows, >= heads, 512] bf16");
  TORCH_CHECK(lse.scalar_type() == torch::kFloat32 && lse.dim() == 2 && lse.size(1) >= heads &&
                  lse.stride(1) == 1,
              "wide core: lse [rows, >= heads] fp32");
  const c10::cuda::CUDAGuard guard(q.device());
  Params p{};
  const int64_t nkv = kv.numel() / kRowB;
  encode_rows(&p.qmap, q.data_ptr(), q.size(0) * q.size(1), (int)heads, 128);
  encode_rows(&p.qrope, q.data_ptr(), q.size(0) * q.size(1), (int)heads, 64);
  encode_rows(&p.kvmap, kv.data_ptr(), nkv, 1, 128);
  encode_rows(&p.kvrope, kv.data_ptr(), nkv, 1, 64);
  p.sel = sel.data_ptr<int>();
  p.sel_stride = sel.stride(0);
  p.width = (int)sel.size(1);
  p.cnt = cnt.data_ptr<int>();
  p.q_heads = (int)q.size(1);
  p.nkv = (int)nkv;
  p.s2scale = (float)(bmm1 * 1.4426950408889634);
  p.oscale = (float)(bmm2 / (double)kPScale);
  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.out_row_stride = out.stride(0);
  p.lse = lse.data_ptr<float>();
  p.lse_row_stride = lse.stride(0);
  p.heads = (int)heads;
  p.T = (int)T;
  p.dbg = dbg.has_value() ? reinterpret_cast<long long*>(dbg->data_ptr<int64_t>()) : nullptr;
  p.dbg_cta = (int)dbg_cta;
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(
        cudaFuncSetAttribute(wide_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
    attr = true;
  }
  const int grid = (int)std::min<int64_t>(2 * T, sms & ~1LL);
  TORCH_CHECK(2 * T <= (int64_t)kMaxItems * grid, "wide core: ", T, " rows exceed ", kMaxItems,
              " items per CTA on ", grid, " CTAs");
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)grid, 1, 1);
  cfg.blockDim = dim3(kThreads, 1, 1);
  cfg.dynamicSmemBytes = kSmem;
  cfg.stream = c10::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
  cfg.attrs = attrs;
  cfg.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, wide_kernel, p));
}

void prepare() {
  C10_CUDA_CHECK(
      cudaFuncSetAttribute(wide_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
}

}  // namespace k31dsaw

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &k31dsaw::run);
  m.def("prepare", &k31dsaw::prepare);
}
