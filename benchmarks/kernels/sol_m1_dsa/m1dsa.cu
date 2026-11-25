// SOL M1 probes for the DsaLayer design (designs/dsa-layer.md). NOT production code.
// Each probe stamps %globaltimer / clock64 inside the kernel, from "inputs landed" (data already
// in shared memory, after a barrier) to "stage result ready", so the numbers are in-kernel stage
// costs with no launch edge.
//   sel   (M1-a) exact global top-K cut over W sorted per-rank lists of N BF16 scores per row
//                -> threshold + per-rank prefix length m_r (ties to the lower rank)
//   topn  (M1-b) exact local top-N (ties to the lowest key) of a request's 8 rows over its keys
//                held in shared memory by a cluster of C CTAs, 2-pass 8-bit radix with DSMEM
//                histogram merge, emission to the row owner, owner sort -> sorted candidate list
//   gqk   (F1)   gather of n candidate 576 B FP8 KV rows + FP8 QK (mma.sync e4m3, 16 heads) per CTA,
//                candidates of a (row, 16-head group) split over S CTAs
//   pre   (f)    weight prestaging: 43.6 MB ring stream (TMA bulk, 4 x 32 KiB per SM) followed by the
//                post-in-proj slices (wq_idx, q_b/W_UK, keys, W_UV) into free shared memory
//   lam   (e,g)  Lamport multicast rounds on W GPUs: scores round alone / with a concurrent q publish
#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

namespace cg = cooperative_groups;

__device__ __forceinline__ long long gtime() {
  long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ uint32_t ord16(uint16_t b) {
  // order-preserving map of BF16 bits to uint16 (larger value -> larger key)
  return (b & 0x8000) ? (uint32_t)(~b & 0xFFFF) : (uint32_t)(b | 0x8000);
}

// ============================================================== M1-a select
// scores [rows][W*N] BF16 bits, each rank's N sorted descending. out m [rows][W], thr [rows].
// st [rows][4]: clk0, clk1, g0, g1.
template <int T, int AGG>
__global__ void __launch_bounds__(T) sel_kernel(const uint16_t* __restrict__ scores, int W, int N, int K,
                                                int* __restrict__ m_out, int* __restrict__ thr_out,
                                                long long* __restrict__ st) {
  extern __shared__ uint16_t s_sc[];
  __shared__ int hist[256], hist2[256];
  __shared__ int s_bin, s_above, s_thr, s_gt;
  __shared__ int s_cgt[32], s_cge[32];
  const int row = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int WN = W * N;
  const uint4* src = reinterpret_cast<const uint4*>(scores + (size_t)row * WN);
  for (int i = tid; i < WN / 8; i += T) reinterpret_cast<uint4*>(s_sc)[i] = src[i];
  for (int i = tid; i < 256; i += T) { hist[i] = 0; hist2[i] = 0; }  // data-independent: before the scores land
  __syncthreads();
  long long c0 = clock64(), g0 = gtime();
  // pass 1: high byte (AGG: warp-aggregated shared atomics, one per distinct bin per warp)
  if (AGG == 2) {  // thread t: elements [t*E, t*E+E) of one sorted list: run-length counts, one atomic per run
    const int E = WN / T;
    int cur = -1, n = 0;
    for (int j = 0; j < E; ++j) {
      const int b = ord16(s_sc[tid * E + j]) >> 8;
      if (b != cur) { if (n) atomicAdd(&hist[cur], n); cur = b; n = 0; }
      ++n;
    }
    if (n) atomicAdd(&hist[cur], n);
  } else
  for (int i = tid; i < WN; i += T) {
    const int b = ord16(s_sc[i]) >> 8;
    if (AGG == 1) {
      const unsigned peers = __match_any_sync(__activemask(), b);
      if (lane == __ffs(peers) - 1) atomicAdd(&hist[b], __popc(peers));
    } else {
      atomicAdd(&hist[b], 1);
    }
  }
  __syncthreads();
  if (warp == 0) {
    // lane l owns bins 255-8l .. 248-8l (descending); find the bin where the count from the top reaches K
    int loc[8], s = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) { loc[j] = hist[255 - 8 * lane - j]; s += loc[j]; }
    int inc = s;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) { int v = __shfl_up_sync(~0u, inc, o); if (lane >= o) inc += v; }
    int exc = inc - s;
    unsigned b = __ballot_sync(~0u, inc >= K);
    int L = __ffs(b) - 1;
    if (lane == L) {
      int acc = exc;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        if (acc + loc[j] >= K) { s_bin = 255 - 8 * lane - j; s_above = acc; break; }
        acc += loc[j];
      }
    }
  }
  __syncthreads();
  const int bin = s_bin;
  if (AGG == 2) {
    const int E = WN / T;
    int cur = -1, n = 0;
    for (int j = 0; j < E; ++j) {
      const uint32_t u = ord16(s_sc[tid * E + j]);
      const int b = (int)(u >> 8) == bin ? (int)(u & 255) : -1;
      if (b != cur) { if (n && cur >= 0) atomicAdd(&hist2[cur], n); cur = b; n = 0; }
      ++n;
    }
    if (n && cur >= 0) atomicAdd(&hist2[cur], n);
  } else
  for (int i = tid; i < WN; i += T) {
    uint32_t u = ord16(s_sc[i]);
    const bool hit = (int)(u >> 8) == bin;
    if (AGG == 1) {
      const unsigned peers = __match_any_sync(__activemask(), hit ? (int)(u & 255) : -1);
      if (hit && lane == __ffs(peers) - 1) atomicAdd(&hist2[u & 255], __popc(peers));
    } else if (hit) {
      atomicAdd(&hist2[u & 255], 1);
    }
  }
  __syncthreads();
  if (warp == 0) {
    const int need = K - s_above;
    int loc[8], s = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) { loc[j] = hist2[255 - 8 * lane - j]; s += loc[j]; }
    int inc = s;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) { int v = __shfl_up_sync(~0u, inc, o); if (lane >= o) inc += v; }
    int exc = inc - s;
    unsigned b = __ballot_sync(~0u, inc >= need);
    int L = __ffs(b) - 1;
    if (lane == L) {
      int acc = exc;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        if (acc + loc[j] >= need) { s_thr = (bin << 8) | (255 - 8 * lane - j); s_gt = s_above + acc; break; }
        acc += loc[j];
      }
    }
  }
  __syncthreads();
  const uint32_t thr = s_thr;
  if (tid < W) {  // binary searches on the sorted (descending) list of rank tid
    const uint16_t* l = s_sc + tid * N;
    int lo = 0, hi = N;  // first index with ord <= thr  -> count gt
    while (lo < hi) { int m = (lo + hi) >> 1; if (ord16(l[m]) > thr) lo = m + 1; else hi = m; }
    int gt = lo;
    hi = N;  // first index with ord < thr -> count ge
    while (lo < hi) { int m = (lo + hi) >> 1; if (ord16(l[m]) >= thr) lo = m + 1; else hi = m; }
    s_cgt[tid] = gt; s_cge[tid] = lo;
  }
  __syncthreads();
  if (warp == 0) {  // tie quota by rank order: warp prefix over ranks
    int gt = lane < W ? s_cgt[lane] : 0, ge = lane < W ? s_cge[lane] : 0, ties = ge - gt;
    int tot = gt;
#pragma unroll
    for (int o = 16; o; o >>= 1) tot += __shfl_xor_sync(~0u, tot, o);
    int inc = ties;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) { int v = __shfl_up_sync(~0u, inc, o); if (lane >= o) inc += v; }
    int quota = K - tot, before = inc - ties;
    int take = max(0, min(ties, quota - before));
    if (lane < W) m_out[row * W + lane] = gt + take;
  }
  __syncthreads();
  long long c1 = clock64(), g1 = gtime();
  if (tid == 0) {
    thr_out[row] = (int)thr;
    st[row * 4 + 0] = c0; st[row * 4 + 1] = c1; st[row * 4 + 2] = g0; st[row * 4 + 3] = g1;
  }
}

void sel_run(torch::Tensor scores, int64_t W, int64_t N, int64_t K, torch::Tensor m, torch::Tensor thr,
             torch::Tensor st, int64_t threads, int64_t agg) {
  int rows = scores.size(0);
  size_t smem = (size_t)W * N * 2;
  auto s = at::cuda::getCurrentCUDAStream();
#define SEL_L(TT, AG)                                                                                       \
  {                                                                                                     \
    cudaFuncSetAttribute(sel_kernel<TT, AG>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);       \
    sel_kernel<TT, AG><<<rows, TT, smem, s>>>((const uint16_t*)scores.data_ptr(), W, N, K, m.data_ptr<int>(), \
                                          thr.data_ptr<int>(), (long long*)st.data_ptr<int64_t>());                 \
  }
  if (agg == 2) { if (threads == 256) SEL_L(256, 2) else if (threads == 512) SEL_L(512, 2) else SEL_L(1024, 2) }
  else if (agg == 1) { if (threads == 256) SEL_L(256, 1) else if (threads == 512) SEL_L(512, 1) else SEL_L(1024, 1) }
  else { if (threads == 256) SEL_L(256, 0) else if (threads == 512) SEL_L(512, 0) else SEL_L(1024, 0) }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ============================================================== M1-b local top-N in a cluster
// logits [Q][8][NK] BF16 bits (keys k = 0..NK-1 of this rank for request q's 8 rows).
// out_key / out_sc [Q][8][TOPN] sorted (score desc, key asc). st [Q*C][8]: c0, c_h1, c_h2, c_emit, c1, g0, g1
constexpr int TN_ROWS = 8, TN_T = 512;
template <int C>
__global__ void __launch_bounds__(TN_T, 1) topn_kernel(const uint16_t* __restrict__ logits, int NK, int TOPN,
                                                      int* __restrict__ out_key, uint16_t* __restrict__ out_sc,
                                                      long long* __restrict__ st) {
  extern __shared__ __align__(16) unsigned char smem[];
  cg::cluster_group cl = cg::this_cluster();
  const int c = (int)cl.block_rank();
  const int q = blockIdx.x / C;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int chunk = (NK + C - 1) / C;
  const int k0 = c * chunk, nk = max(0, min(NK, k0 + chunk) - k0);
  int* hist = reinterpret_cast<int*>(smem);                 // [8][256]
  int* tot = hist + TN_ROWS * 256;                           // [8][256] (merged)
  int* cnt = tot + TN_ROWS * 256;                            // [8][4]: gt, eq, emit_gt_ctr, emit_eq_ctr
  int* rowinfo = cnt + TN_ROWS * 4;                          // [8][4]: bin, above, thr, gt_total
  int* cand_key = rowinfo + TN_ROWS * 4;                     // [8][TOPN] (owner rows)
  uint32_t* cand_ord = reinterpret_cast<uint32_t*>(cand_key + TN_ROWS * TOPN);  // [8][TOPN]
  uint16_t* lg = reinterpret_cast<uint16_t*>(cand_ord + TN_ROWS * TOPN);        // [8][chunk]
  // load this CTA's logits (scorer epilogue output: already in SMEM in the design)
  for (int r = 0; r < TN_ROWS; ++r)
    for (int i = tid; i < nk; i += TN_T) lg[r * chunk + i] = logits[((size_t)q * TN_ROWS + r) * NK + k0 + i];
  for (int i = tid; i < TN_ROWS * 256; i += TN_T) hist[i] = 0;
  for (int i = tid; i < TN_ROWS * 4; i += TN_T) cnt[i] = 0;
  cl.sync();
  long long c0 = clock64(), g0 = gtime();
  // ---- pass 1: high-byte histogram per row
  for (int r = 0; r < TN_ROWS; ++r)
    for (int i = tid; i < nk; i += TN_T) {
      const int bb = ord16(lg[r * chunk + i]) >> 8;
      const unsigned peers = __match_any_sync(__activemask(), bb);
      if (lane == __ffs(peers) - 1) atomicAdd(&hist[r * 256 + bb], __popc(peers));
    }
  cl.sync();
  long long ch1 = clock64();
  for (int i = tid; i < TN_ROWS * 256; i += TN_T) {
    int s = 0;
#pragma unroll
    for (int j = 0; j < C; ++j) s += cl.map_shared_rank(hist, j)[i];
    tot[i] = s;
  }
  __syncthreads();
  if (warp < TN_ROWS) {  // warp r: crossing bin of row r
    const int r = warp, K = TOPN;
    int loc[8], s = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) { loc[j] = tot[r * 256 + 255 - 8 * lane - j]; s += loc[j]; }
    int inc = s;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) { int v = __shfl_up_sync(~0u, inc, o); if (lane >= o) inc += v; }
    int exc = inc - s;
    unsigned b = __ballot_sync(~0u, inc >= K);
    int L = __ffs(b) - 1;
    if (lane == L) {
      int acc = exc;
      for (int j = 0; j < 8; ++j) {
        if (acc + loc[j] >= K) { rowinfo[r * 4 + 0] = 255 - 8 * lane - j; rowinfo[r * 4 + 1] = acc; break; }
        acc += loc[j];
      }
    }
  }
  // own hist can be cleared only after every peer read it: use a second buffer region (tot of pass 2 -> hist2 = hist
  // reused after the next cluster barrier). Pass 2 counts into 'cnt-free' area: reuse 'tot' after this point locally.
  __syncthreads();
  // pass 2 histogram into tot (local; tot of pass 1 is consumed)
  for (int i = tid; i < TN_ROWS * 256; i += TN_T) tot[i] = 0;
  __syncthreads();
  for (int r = 0; r < TN_ROWS; ++r) {
    const int bin = rowinfo[r * 4 + 0];
    for (int i = tid; i < nk; i += TN_T) {
      uint32_t u = ord16(lg[r * chunk + i]);
      const bool hit = (int)(u >> 8) == bin;
      const unsigned peers = __match_any_sync(__activemask(), hit ? (int)(u & 255) : -1);
      if (hit && lane == __ffs(peers) - 1) atomicAdd(&tot[r * 256 + (u & 255)], __popc(peers));
    }
  }
  cl.sync();  // every peer finished reading 'hist' (pass 1) and wrote its pass-2 'tot'
  long long ch2 = clock64();
  for (int i = tid; i < TN_ROWS * 256; i += TN_T) {
    int s = 0;
#pragma unroll
    for (int j = 0; j < C; ++j) s += cl.map_shared_rank(tot, j)[i];
    hist[i] = s;  // merged pass-2 histogram
  }
  __syncthreads();
  if (warp < TN_ROWS) {
    const int r = warp, need = TOPN - rowinfo[r * 4 + 1], bin = rowinfo[r * 4 + 0];
    int loc[8], s = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) { loc[j] = hist[r * 256 + 255 - 8 * lane - j]; s += loc[j]; }
    int inc = s;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) { int v = __shfl_up_sync(~0u, inc, o); if (lane >= o) inc += v; }
    int exc = inc - s;
    unsigned b = __ballot_sync(~0u, inc >= need);
    int L = __ffs(b) - 1;
    if (lane == L) {
      int acc = exc;
      for (int j = 0; j < 8; ++j) {
        if (acc + loc[j] >= need) { rowinfo[r * 4 + 2] = (bin << 8) | (255 - 8 * lane - j); rowinfo[r * 4 + 3] = rowinfo[r * 4 + 1] + acc; break; }
        acc += loc[j];
      }
    }
  }
  __syncthreads();
  // local counts > thr and == thr per row
  for (int r = 0; r < TN_ROWS; ++r) {
    const uint32_t thr = rowinfo[r * 4 + 2];
    int g = 0, e = 0;
    for (int i = tid; i < nk; i += TN_T) { uint32_t u = ord16(lg[r * chunk + i]); g += u > thr; e += u == thr; }
#pragma unroll
    for (int o = 16; o; o >>= 1) { g += __shfl_xor_sync(~0u, g, o); e += __shfl_xor_sync(~0u, e, o); }
    if (lane == 0) { atomicAdd(&cnt[r * 4 + 0], g); atomicAdd(&cnt[r * 4 + 1], e); }
  }
  cl.sync();  // counts visible; pass-2 'tot' reads by peers done
  long long cem = clock64();
  // offsets: CTAs before me (key order) for > thr and for ties; tie quota in key order
  __shared__ int s_off_gt[TN_ROWS], s_tie_skip[TN_ROWS], s_tie_take[TN_ROWS];
  if (tid < TN_ROWS) {
    const int r = tid;
    int og = 0, oe = 0;
    for (int j = 0; j < c; ++j) { int* pc = cl.map_shared_rank(cnt, j); og += pc[r * 4 + 0]; oe += pc[r * 4 + 1]; }
    const int quota = TOPN - rowinfo[r * 4 + 3];  // ties to take in total
    const int mine = cnt[r * 4 + 1];
    s_off_gt[r] = og;
    s_tie_take[r] = max(0, min(mine, quota - oe));
    s_tie_skip[r] = min(quota, oe);  // ties of earlier CTAs taken (their slots come first)
  }
  __syncthreads();
  // emission: rows > thr via an atomic local slot (order fixed later by the owner sort); ties in key order
  // (warp r scans row r's ties with ballots when only part of them is taken).
  for (int r = 0; r < TN_ROWS; ++r) {
    const uint32_t thr = rowinfo[r * 4 + 2];
    const int owner = r % C;
    int* okey = cl.map_shared_rank(cand_key, owner) + r * TOPN;
    uint32_t* oord = cl.map_shared_rank(cand_ord, owner) + r * TOPN;
    const int gt_total = rowinfo[r * 4 + 3];
    const bool all_ties = s_tie_take[r] == cnt[r * 4 + 1];
    for (int i = tid; i < nk; i += TN_T) {
      uint32_t u = ord16(lg[r * chunk + i]);
      if (u > thr) {
        int p = s_off_gt[r] + atomicAdd(&cnt[r * 4 + 2], 1);
        okey[p] = k0 + i; oord[p] = u;
      } else if (u == thr && all_ties && s_tie_take[r] > 0) {
        int p = gt_total + s_tie_skip[r] + atomicAdd(&cnt[r * 4 + 3], 1);
        okey[p] = k0 + i; oord[p] = u;
      }
    }
    if (!all_ties && s_tie_take[r] > 0 && warp == r % (TN_T / 32)) {
      int taken = 0;
      for (int base = 0; base < nk && taken < s_tie_take[r]; base += 32) {
        int i = base + lane;
        bool t = i < nk && ord16(lg[r * chunk + i]) == thr;
        unsigned bm = __ballot_sync(~0u, t);
        int pos = taken + __popc(bm & ((1u << lane) - 1));
        if (t && pos < s_tie_take[r]) { int p = gt_total + s_tie_skip[r] + pos; okey[p] = k0 + i; oord[p] = thr; }
        taken += __popc(bm);
      }
    }
  }
  cl.sync();
  // owner sort of its rows: rank by counting (score desc, key asc)
  for (int r = c; r < TN_ROWS; r += C) {
    int* ck = cand_key + r * TOPN;
    uint32_t* co = cand_ord + r * TOPN;
    int myk = 0; uint32_t myo = 0; int rank = 0;
    const bool act = tid < TOPN;
    if (act) { myk = ck[tid]; myo = co[tid]; }
    if (act) {
      for (int j = 0; j < TOPN; ++j) {
        uint32_t oj = co[j]; int kj = ck[j];
        rank += (oj > myo) | ((oj == myo) & (kj < myk));
      }
    }
    __syncthreads();
    if (act) {
      out_key[((size_t)q * TN_ROWS + r) * TOPN + rank] = myk;
      uint32_t o = myo;
      uint16_t b = (o & 0x8000) ? (uint16_t)(o & 0x7FFF) : (uint16_t)(~o & 0xFFFF);
      out_sc[((size_t)q * TN_ROWS + r) * TOPN + rank] = b;
    }
    __syncthreads();
  }
  long long c1 = clock64(), g1 = gtime();
  if (tid == 0) {
    long long* s = st + (size_t)blockIdx.x * 8;
    s[0] = c0; s[1] = ch1; s[2] = ch2; s[3] = cem; s[4] = c1; s[5] = g0; s[6] = g1;
  }
}

void topn_run(torch::Tensor logits, int64_t C, int64_t topn, torch::Tensor out_key, torch::Tensor out_sc,
              torch::Tensor st) {
  const int Q = logits.size(0), NK = logits.size(2);
  const int chunk = (NK + C - 1) / C;
  size_t smem = (size_t)TN_ROWS * 256 * 4 * 2 + TN_ROWS * 4 * 4 * 2 + (size_t)TN_ROWS * topn * 8 + (size_t)TN_ROWS * chunk * 2 + 64;
  auto s = at::cuda::getCurrentCUDAStream();
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(Q * C);
  cfg.blockDim = dim3(TN_T);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = s;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = C; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
  cfg.attrs = at; cfg.numAttrs = 1;
#define TN_L(CC)                                                                                                  \
  {                                                                                                               \
    cudaFuncSetAttribute(topn_kernel<CC>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);                \
    if (CC > 8) cudaFuncSetAttribute(topn_kernel<CC>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);         \
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, topn_kernel<CC>, (const uint16_t*)logits.data_ptr(), NK, (int)topn,  \
                                      out_key.data_ptr<int>(), (uint16_t*)out_sc.data_ptr(), (long long*)st.data_ptr<int64_t>())); \
  }
  switch (C) {
    case 1: TN_L(1) break;
    case 2: TN_L(2) break;
    case 4: TN_L(4) break;
    case 8: TN_L(8) break;
    case 16: TN_L(16) break;
    default: TORCH_CHECK(false, "C");
  }
}

// ============================================================== F1 gather + QK
// grid (S, G, R); CTA: 16 heads of row t, candidates [s*n, s*n + n) of the row's 192.
// q [R][G*16][576] FP8, slots [R][NC] int32 (flat KV rows), kv [NKV][576] FP8. scores_out [R][G*16][NC] FP32.
// st [S*G*R][4]: c0, c_gather, c1, g0/g1 -> 6 entries
constexpr int GQ_T = 512, GQ_STR = 592;
__device__ __forceinline__ void mma_e4m3(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__global__ void __launch_bounds__(GQ_T, 1) gqk_kernel(const uint8_t* __restrict__ q, const int* __restrict__ slots,
                                                     const uint8_t* __restrict__ kv, int NC, int S, float* __restrict__ out,
                                                     long long* __restrict__ st) {
  extern __shared__ __align__(16) unsigned char sm[];
  const int s = blockIdx.x, g = blockIdx.y, t = blockIdx.z, G = gridDim.y;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int per = (NC + S - 1) / S, c0i = s * per, n = max(0, min(NC, c0i + per) - c0i);
  uint8_t* sq = sm;                        // [16][GQ_STR]
  uint8_t* skv = sm + 16 * GQ_STR;         // [n][GQ_STR]
  int* ssl = reinterpret_cast<int*>(skv + ((n + 7) & ~7) * GQ_STR);  // [n]
  const uint8_t* qs = q + ((size_t)t * G * 16 + g * 16) * 576;
  for (int i = tid; i < 16 * 36; i += GQ_T) {
    int h = i / 36, c = i % 36;
    *reinterpret_cast<uint4*>(sq + h * GQ_STR + c * 16) = *reinterpret_cast<const uint4*>(qs + h * 576 + c * 16);
  }
  for (int i = tid; i < n; i += GQ_T) ssl[i] = slots[(size_t)t * NC + c0i + i];
  // zero pad rows of the last n-tile
  for (int i = tid; i < (((n + 7) & ~7) - n) * 36; i += GQ_T) {
    int r = n + i / 36, c = i % 36;
    *reinterpret_cast<uint4*>(skv + r * GQ_STR + c * 16) = make_uint4(0, 0, 0, 0);
  }
  __syncthreads();
  long long c0 = clock64(), g0 = gtime();
  // gather: n rows x 36 chunks of 16 B, cp.async.cg (L2 only)
  for (int i = tid; i < n * 36; i += GQ_T) {
    int r = i / 36, c = i % 36;
    const uint8_t* src = kv + (size_t)ssl[r] * 576 + c * 16;
    uint32_t dst = (uint32_t)__cvta_generic_to_shared(skv + r * GQ_STR + c * 16);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src));
  }
  asm volatile("cp.async.commit_group;");
  asm volatile("cp.async.wait_group 0;" ::: "memory");
  __syncthreads();
  long long cg_ = clock64();
  // QK: warp per 8-key n-tile, 18 k-steps of 32
  const int gid = lane >> 2, tig = lane & 3;
  const int ntiles = (n + 7) / 8;
  for (int nt = warp; nt < ntiles; nt += GQ_T / 32) {
    float d[4] = {0.f, 0.f, 0.f, 0.f};
    const uint8_t* arow0 = sq + gid * GQ_STR + tig * 4;
    const uint8_t* arow1 = sq + (gid + 8) * GQ_STR + tig * 4;
    const uint8_t* brow = skv + (nt * 8 + gid) * GQ_STR + tig * 4;
#pragma unroll 6
    for (int k = 0; k < 18; ++k) {
      uint32_t a0 = *reinterpret_cast<const uint32_t*>(arow0 + k * 32);
      uint32_t a1 = *reinterpret_cast<const uint32_t*>(arow1 + k * 32);
      uint32_t a2 = *reinterpret_cast<const uint32_t*>(arow0 + k * 32 + 16);
      uint32_t a3 = *reinterpret_cast<const uint32_t*>(arow1 + k * 32 + 16);
      uint32_t b0 = *reinterpret_cast<const uint32_t*>(brow + k * 32);
      uint32_t b1 = *reinterpret_cast<const uint32_t*>(brow + k * 32 + 16);
      mma_e4m3(d, a0, a1, a2, a3, b0, b1);
    }
    float* o = out + ((size_t)t * G * 16 + g * 16) * NC;
    int col = nt * 8 + tig * 2;
    if (c0i + col < NC && col < n) { o[gid * NC + c0i + col] = d[0]; o[(gid + 8) * NC + c0i + col] = d[2]; }
    if (c0i + col + 1 < NC && col + 1 < n) { o[gid * NC + c0i + col + 1] = d[1]; o[(gid + 8) * NC + c0i + col + 1] = d[3]; }
  }
  __syncthreads();
  long long c1 = clock64(), g1 = gtime();
  if (tid == 0) {
    long long* p = st + ((size_t)(t * G + g) * S + s) * 6;
    p[0] = c0; p[1] = cg_; p[2] = c1; p[3] = g0; p[4] = g1; p[5] = n;
  }
}
void gqk_run(torch::Tensor q, torch::Tensor slots, torch::Tensor kv, int64_t S, torch::Tensor out, torch::Tensor st) {
  const int R = q.size(0), G = q.size(1) / 16, NC = slots.size(1);
  const int per = (NC + S - 1) / S;
  size_t smem = 16 * GQ_STR + (size_t)((per + 7) & ~7) * GQ_STR + per * 4 + 64;
  auto s = at::cuda::getCurrentCUDAStream();
  cudaFuncSetAttribute(gqk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  gqk_kernel<<<dim3(S, G, R), GQ_T, smem, s>>>((const uint8_t*)q.data_ptr(), slots.data_ptr<int>(),
                                               (const uint8_t*)kv.data_ptr(), NC, (int)S, out.data_ptr<float>(),
                                               (long long*)st.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ============================================================== (f) weight prestaging
// One CTA per SM. main: per-CTA contiguous slice of `main_bytes` through a ring of 4 x 32 KiB TMA bulk stages.
// extras (4 classes, per-CTA slice bytes ex[0..3]): mode 0 none; 1 issued right after the last main op is issued
// (classes 0-2 into a free region, class 3 into ring slots as they drain = the design); 2 all extras issued first.
// st [CTA][8]: g_start, g_main_done, g_ex0..g_ex3, clk_start, clk_main
constexpr int PR_STAGE = 32768, PR_RING = 4;
__device__ __forceinline__ void mb_init(uint64_t* b, int n) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"((uint32_t)__cvta_generic_to_shared(b)), "r"(n));
}
__device__ __forceinline__ void mb_expect(uint64_t* b, uint32_t tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"((uint32_t)__cvta_generic_to_shared(b)),
               "r"(tx));
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* b) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
          (uint32_t)__cvta_generic_to_shared(dst)),
      "l"(src), "r"(bytes), "r"((uint32_t)__cvta_generic_to_shared(b))
      : "memory");
}
__device__ __forceinline__ void mb_wait(uint64_t* b, uint32_t parity) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(b);
  asm volatile(
      "{\n .reg .pred p;\n W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W;\n}" ::"r"(a),
      "r"(parity)
      : "memory");
}
// issue `bytes` from src into dst in <= 32 KiB pieces on one mbarrier
__device__ __forceinline__ void bulk_issue(void* dst, const uint8_t* src, uint32_t bytes, uint64_t* b) {
  mb_expect(b, bytes);
  for (uint32_t o = 0; o < bytes; o += PR_STAGE) bulk_g2s((uint8_t*)dst + o, src + o, min((uint32_t)PR_STAGE, bytes - o), b);
}
__global__ void __launch_bounds__(128, 1) pre_kernel(const uint8_t* __restrict__ main_w, long long main_bytes,
                                                    const uint8_t* __restrict__ ex_w, int4 exb, int mode,
                                                    long long* __restrict__ st) {
  extern __shared__ __align__(1024) unsigned char sm[];
  __shared__ __align__(8) uint64_t bar[PR_RING + 4];
  uint8_t* ring = sm;                          // 4 x 32 KiB
  uint8_t* exr = sm + PR_RING * PR_STAGE;      // free region (classes 0-2)
  const int tid = threadIdx.x, cta = blockIdx.x, n = gridDim.x;
  const int ex[4] = {exb.x, exb.y, exb.z, exb.w};
  const long long per = (main_bytes / n) & ~15ll;
  const uint8_t* src = main_w + per * cta;
  long long exper = (long long)ex[0] + ex[1] + ex[2] + ex[3];
  const uint8_t* esrc = ex_w + exper * cta;
  if (tid == 0) {
    for (int i = 0; i < PR_RING + 4; ++i) mb_init(&bar[i], 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  long long gs = gtime(), cs = clock64();
  long long gm = 0, ge[4] = {0, 0, 0, 0}, cm = 0;
  if (tid == 0) {
    const int nst = (int)((per + PR_STAGE - 1) / PR_STAGE);
    auto issue_ex = [&](int k, void* dst) {
      long long off = 0;
      for (int j = 0; j < k; ++j) off += ex[j];
      if (ex[k] > 0) bulk_issue(dst, esrc + off, ex[k], &bar[PR_RING + k]);
    };
    uint8_t* exdst[3] = {exr, exr + ex[0], exr + ex[0] + ex[1]};
    if (mode == 2) { for (int k = 0; k < 3; ++k) issue_ex(k, exdst[k]); }
    int issued = 0;
    for (; issued < min(nst, PR_RING); ++issued) {
      uint32_t b = (uint32_t)min((long long)PR_STAGE, per - (long long)issued * PR_STAGE);
      mb_expect(&bar[issued], b);
      bulk_g2s(ring + issued * PR_STAGE, src + (long long)issued * PR_STAGE, b, &bar[issued]);
    }
    if (mode == 1 && issued == nst) { for (int k = 0; k < 3; ++k) issue_ex(k, exdst[k]); }
    for (int i = 0; i < nst; ++i) {
      const int slot = i % PR_RING;
      mb_wait(&bar[slot], (i / PR_RING) & 1);
      // (consumer would run its MMA on the stage here)
      if (issued < nst) {
        uint32_t b = (uint32_t)min((long long)PR_STAGE, per - (long long)issued * PR_STAGE);
        mb_expect(&bar[slot], b);
        bulk_g2s(ring + slot * PR_STAGE, src + (long long)issued * PR_STAGE, b, &bar[slot]);
        ++issued;
        if (mode == 1 && issued == nst) { for (int k = 0; k < 3; ++k) issue_ex(k, exdst[k]); }
      }
    }
    gm = gtime(); cm = clock64();
    // class 3 (W_UV) into the drained ring (mode 1 and 2)
    if (mode == 2 || mode == 1) issue_ex(3, ring);
    if (mode == 3) { for (int k = 0; k < 3; ++k) issue_ex(k, exdst[k]); issue_ex(3, ring); }
    for (int k = 0; k < 4; ++k) {
      if (mode != 0 && ex[k] > 0) mb_wait(&bar[PR_RING + k], 0);
      ge[k] = gtime();
    }
    long long* p = st + (size_t)cta * 8;
    p[0] = gs; p[1] = gm; p[2] = ge[0]; p[3] = ge[1]; p[4] = ge[2]; p[5] = ge[3]; p[6] = cs; p[7] = cm;
  }
}
void pre_run(torch::Tensor main_w, int64_t main_bytes, torch::Tensor ex_w, std::vector<int64_t> exb, int64_t mode,
             int64_t ctas, torch::Tensor st) {
  size_t smem = PR_RING * PR_STAGE + (size_t)(exb[0] + exb[1] + exb[2]);
  smem = (smem + 1023) & ~1023ull;
  TORCH_CHECK(smem <= 227 * 1024 - 1024, "pre smem");
  TORCH_CHECK((uint64_t)exb[3] <= (uint64_t)PR_RING * PR_STAGE, "W_UV slice > ring");
  auto s = at::cuda::getCurrentCUDAStream();
  cudaFuncSetAttribute(pre_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  int4 e = make_int4((int)exb[0], (int)exb[1], (int)exb[2], (int)exb[3]);
  pre_kernel<<<(int)ctas, 128, smem, s>>>((const uint8_t*)main_w.data_ptr(), main_bytes, (const uint8_t*)ex_w.data_ptr(),
                                           e, (int)mode, (long long*)st.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ============================================================== (e, g) Lamport multicast rounds
// Box layouts (symmetric, per GPU): tk [W][tk_words] u32, q [W][q_words] u32; sentinel 0xFFFFFFFF.
// CTA roles: [0, n_tk): scores publish + poll; [n_tk, n_tk + n_q): q publish + poll (if q_words).
// mode: bit0 = scores on, bit1 = q on; q_lead_ns: scores CTAs start that long after the q CTAs.
// Each launch: a 1-word Lamport barrier round (bar box [W] u32, value = epoch) aligns ranks; t0 = its exit.
// st [CTA][4]: g_t0, g_published, g_arrived, epoch
__device__ __forceinline__ void mm_st_v4(uint64_t addr, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
  asm volatile("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(addr), "r"(a), "r"(b), "r"(c),
               "r"(d)
               : "memory");
}
__device__ __forceinline__ uint4 ld_vol_v4(const uint32_t* p) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ void mm_st_u32(uint64_t addr, uint32_t v) {
  asm volatile("multimem.st.relaxed.sys.global.u32 [%0], %1;" ::"l"(addr), "r"(v) : "memory");
}
__device__ __forceinline__ uint32_t ld_vol(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}
__global__ void __launch_bounds__(512, 1) lam_kernel(uint64_t tk_mc, uint32_t* tk_loc, int tk_words, uint64_t q_mc,
                                                    uint32_t* q_loc, int q_words, uint64_t bar_mc, uint32_t* bar_loc,
                                                    int W, int rank, int n_tk, int n_q, int mode, int q_lead_ns,
                                                    uint32_t epoch, long long* st) {
  const int cta = blockIdx.x, tid = threadIdx.x;
  // ---- barrier round (every CTA polls; CTA 0 thread 0 publishes)
  if (cta == 0 && tid == 0) mm_st_u32(bar_mc + 4ull * rank, epoch);
  if (tid < W) { while (ld_vol(bar_loc + tid) != epoch) { } }
  __syncthreads();
  long long t0 = gtime(), tp = 0, ta = 0;
  const bool is_tk = cta < n_tk;
  const int idx = is_tk ? cta : cta - n_tk, nrole = is_tk ? n_tk : n_q;
  const int words = is_tk ? tk_words : q_words;
  const uint64_t mc = is_tk ? tk_mc : q_mc;
  uint32_t* loc = is_tk ? tk_loc : q_loc;
  const bool on = is_tk ? (mode & 1) : (mode & 2);
  if (on && words > 0) {
    if (is_tk && q_lead_ns > 0) { while (gtime() - t0 < q_lead_ns) { } }
    // this CTA's 16 B vectors: [lo, hi) of words/4 vectors
    const int nv = words / 4, per = (nv + nrole - 1) / nrole, lo = idx * per, hi = min(nv, lo + per);
    const uint32_t tag = epoch & 0x7FFFFFFF;  // payload words are (tag) ^ i-ish: never 0xFFFFFFFF
    const uint64_t my = mc + 4ull * ((uint64_t)rank * words);
    for (int v = lo + tid; v < hi; v += blockDim.x) mm_st_v4(my + 16ull * v, tag, tag ^ v, tag + 1, (uint32_t)v & 0x7FFFFFFF);
    __syncthreads();
    tp = gtime();
    // poll every peer's slice [lo, hi) until landed, then re-arm it
    for (int r = 0; r < W; ++r) {
      if (r == rank) continue;
      const uint32_t* base = loc + (size_t)r * words;
      for (int v = lo + tid; v < hi; v += blockDim.x) {
        uint4 x;
        do { x = ld_vol_v4(base + 4 * v); } while (x.x != tag || x.w != ((uint32_t)v & 0x7FFFFFFF));
      }
    }
    __syncthreads();
    ta = gtime();
  }
  if (tid == 0) { long long* p = st + (size_t)cta * 4; p[0] = t0; p[1] = tp; p[2] = ta; p[3] = epoch; }
}
void lam_run(int64_t tk_mc, int64_t tk_loc, int64_t tk_words, int64_t q_mc, int64_t q_loc, int64_t q_words,
             int64_t bar_mc, int64_t bar_loc, int64_t W, int64_t rank, int64_t n_tk, int64_t n_q, int64_t mode,
             int64_t q_lead_ns, int64_t epoch, torch::Tensor st) {
  auto s = at::cuda::getCurrentCUDAStream();
  lam_kernel<<<(int)(n_tk + n_q), 512, 0, s>>>((uint64_t)tk_mc, (uint32_t*)tk_loc, (int)tk_words, (uint64_t)q_mc,
                                               (uint32_t*)q_loc, (int)q_words, (uint64_t)bar_mc, (uint32_t*)bar_loc,
                                               (int)W, (int)rank, (int)n_tk, (int)n_q, (int)mode, (int)q_lead_ns,
                                               (uint32_t)epoch, (long long*)st.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#include "topn_row.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("topn_row_run", &topn_row_run);
  m.def("sel_run", &sel_run);
  m.def("topn_run", &topn_run);
  m.def("gqk_run", &gqk_run);
  m.def("pre_run", &pre_run);
  m.def("lam_run", &lam_run);
}
