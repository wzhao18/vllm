// dattn_skel2_mla.cu: sparse-MLA data-movement skeleton rebuilt from the hwdb hw15c probe (per-issuing-lane gather4
// rings, 32 issuing lanes per SM, shallow per-lane slots), fed with the R384 split-K row lists.
// Milestones (status D-ATTN): M1 = hw15c form verbatim on the per-row key lists (no dedup);
// M2 = the same on the request-union list (dedup); M3 = + tile assembly (static lane -> quad map, a consumer warp waits
// the per-lane full barriers of each CG-lane chunk and releases per-lane empty barriers) = the real kernel's sync.
// Usage: skel2 MODE Wg L S [cfrac] [iters] [CG]
//   MODE bit0: request-union dedup list; bit1: tile assembly + consumer warp (M3); bit2: L2-warm (no flush)
// One gather4 op = 4 whole 576 B KV rows (UINT32 tensor map, box 144 x 1). In-kernel span = globaltimer min start ->
// max end over CTAs.
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char* s; cuGetErrorString(r_, &s); \
  printf("CU error %s at %s:%d\n", s, __FILE__, __LINE__); exit(1); } } while (0)
#define RK(x) do { cudaError_t r_ = (x); if (r_ != cudaSuccess) { printf("CUDA error %s at %s:%d\n", \
  cudaGetErrorString(r_), __FILE__, __LINE__); exit(1); } } while (0)

constexpr int ROWB = 576, OPB = 4 * ROWB, QMAX = 1024;
__device__ __forceinline__ unsigned sa(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void wait_par(unsigned b, unsigned ph) {
  asm volatile("{.reg .pred P; W: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1; @!P bra W;}" :: "r"(b), "r"(ph) : "memory");
}
__device__ __forceinline__ unsigned long long gt() { unsigned long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }

__device__ __forceinline__ void g4(const CUtensorMap* m, unsigned dst, int col, uint4 r, unsigned b) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
               :: "r"(dst), "l"(m), "r"(col), "r"(r.x), "r"(r.y), "r"(r.z), "r"(r.w), "r"(b) : "memory");
}
// MODE bit5: MMA-consumable SW128 form = 4 x (128 B x 4 rows, SWIZZLE_128B) + 1 x (64 B x 4 rows); else one whole-row op
__device__ __forceinline__ void issue(int mode, const CUtensorMap* mw, const CUtensorMap* m128, const CUtensorMap* m64,
                                      unsigned dst, uint4 r, unsigned b) {
  if (mode & 32) {
#pragma unroll
    for (int k = 0; k < 4; ++k) g4(m128, dst + 512 * k, 128 * k, r, b);
    g4(m64, dst + 2048, 512, r, b);
  } else {
    g4(mw, dst, 0, r, b);
  }
}

__global__ void __launch_bounds__(1024, 1) g2(const __grid_constant__ CUtensorMap mg, const __grid_constant__ CUtensorMap m128, const __grid_constant__ CUtensorMap m64, const uint4* quads, const int* off,
                                              int mode, int Wg, int L, int S, int CG, int qoff, unsigned long long* tspan,
                                              unsigned long long* cnt, unsigned* sink) {
  extern __shared__ __align__(1024) unsigned char smem[];
  __shared__ __align__(8) unsigned long long full[256], empty[256];
  __shared__ unsigned qctr;
  __shared__ unsigned long long T0;
  __shared__ volatile int rel[256];
  uint4* sq = (uint4*)smem;
  unsigned char* ring = smem + qoff;  // 1024-aligned
  const int q0 = off[blockIdx.x], nq = off[blockIdx.x + 1] - q0, NL = Wg * L;
  const int w = threadIdx.x / 32, lane = threadIdx.x % 32;
  for (int i = threadIdx.x; i < nq; i += blockDim.x) sq[i] = quads[q0 + i];
  if (threadIdx.x == 0) {
    qctr = 0; T0 = gt();
    for (int i = 0; i < 256; ++i) rel[i] = 0;
    for (int i = 0; i < NL * S; ++i) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(sa(&full[i])));
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(sa(&empty[i])));
    }
  }
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  __syncthreads();
  if (threadIdx.x == 0) atomicMin(&tspan[0], gt());
  unsigned acc = 0;
  const bool tiles = mode & 2;
  if (w < Wg && lane < L) {
    const int it = w * L + lane;  // issuing-lane id; its ring = S slots of one gather4 op
    unsigned char* myring = ring + (size_t)it * S * OPB;
    unsigned long long done = 0;
    if (!tiles) {  // M1/M2: hw15c verbatim (dynamic claim, own slot reuse)
      int n = 0;
      for (;;) {
        if (n >= S) { int sl = (n - S) % S; wait_par(sa(&full[it * S + sl]), ((n - S) / S) & 1); acc += myring[(size_t)sl * OPB + 16]; }
        unsigned q = atomicAdd(&qctr, 1u);
        if ((int)q >= nq) break;
        uint4 r = sq[q];
        int sl = n % S; unsigned b = sa(&full[it * S + sl]);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(OPB) : "memory");
        issue(mode, &mg, &m128, &m64, sa(myring + (size_t)sl * OPB), r, b);
        ++n;
      }
      for (int i = (n >= S ? n - S + 1 : 0); i < n; ++i) { int sl = i % S; wait_par(sa(&full[it * S + sl]), (i / S) & 1); acc += myring[(size_t)sl * OPB + 16]; }
      done = n;
    } else {  // M3: static map, tile t = quads [NL*t, NL*t+NL), lane it owns quad NL*t+it, slot t%S
      for (int t = 0; NL * t + it < nq; ++t) {
        int sl = t % S;
        if (t >= S) {
          if (mode & 64) { wait_par(sa(&full[it * S + sl]), ((t / S) - 1) & 1); acc += myring[(size_t)sl * OPB + 16]; }
          else if (mode & 256) { while (rel[it * S + sl] < t / S) { } }
          else if (mode & 128) {
            unsigned p = 0;
            while (!p) asm volatile("{.reg .pred P; mbarrier.test_wait.parity.shared::cta.b64 P, [%1], %2; selp.u32 %0, 1, 0, P;}"
                                    : "=r"(p) : "r"(sa(&empty[it * S + sl])), "r"(((t / S) - 1) & 1) : "memory");
          } else wait_par(sa(&empty[it * S + sl]), ((t / S) - 1) & 1);
        }
        if ((mode & 16) && blockIdx.x == 0 && t < 8) tspan[130 + it * 8 + t] = gt() - T0;
        uint4 r = sq[NL * t + it];
        unsigned b = sa(&full[it * S + sl]);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(OPB) : "memory");
        issue(mode, &mg, &m128, &m64, sa(myring + (size_t)sl * OPB), r, b);
        ++done;
      }
      if (mode & 64) for (int t = (int)done > S ? (int)done - S : 0; t < (int)done; ++t) { wait_par(sa(&full[it * S + t % S]), (t / S) & 1); acc += myring[(size_t)(t % S) * OPB + 16]; }
    }
    atomicAdd(&cnt[0], done);
  } else if (tiles && !(mode & 64) && w == Wg && (mode & 8)) {
    // M3b: out-of-order chunk consumer. Chunk c = issuing lanes [c*CG, c*CG+CG) (CG <= 32 keys/4 rows each); each chunk
    // advances through its tiles independently (split-K online softmax is order-free), so a straggling op only holds
    // its own chunk, not the whole tile.
    const int NC = NL / CG, ntile = (nq + NL - 1) / NL;
    int nt[16];
    for (int c = 0; c < NC; ++c) nt[c] = 0;
    int left = NC;
    const unsigned long long tb = T0;
    int ev = 0;
    while (left) {
      for (int c = 0; c < NC; ++c) {
        if (nt[c] >= ntile) continue;
        const int t = nt[c], it = c * CG + lane;
        bool ok = true;
        if (lane < CG && NL * t + it < nq) {
          unsigned p;
          asm volatile("{.reg .pred P; mbarrier.test_wait.parity.shared::cta.b64 P, [%1], %2; selp.u32 %0, 1, 0, P;}"
                       : "=r"(p) : "r"(sa(&full[it * S + t % S])), "r"((t / S) & 1) : "memory");
          ok = p;
        }
        if (!__all_sync(0xffffffffu, ok)) continue;
        if (lane < CG && NL * t + it < nq) {
          acc += ring[((size_t)it * S + t % S) * OPB + 16];
          if (mode & 256) rel[it * S + t % S] = t / S + 1;
          else asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(sa(&empty[it * S + t % S])) : "memory");
        }
        if ((mode & 16) && blockIdx.x == 0 && lane == 0 && ev < 120) tspan[2 + ev++] = ((unsigned long long)(gt() - tb) << 16) | (c << 8) | t;
        if (++nt[c] >= ntile) --left;
      }
    }
  } else if (tiles && !(mode & 64) && w == Wg) {  // consumer warp: lane j covers issuing lanes j, j+32, ... (NL <= 64)
    const int ntile = (nq + NL - 1) / NL;
    for (int t = 0; t < ntile; ++t) {
      int sl = t % S;
      for (int c0 = 0; c0 < NL; c0 += CG) {
        for (int it = c0 + lane; it < c0 + CG; it += 32)
          if (NL * t + it < nq) { wait_par(sa(&full[it * S + sl]), (t / S) & 1); acc += ring[((size_t)it * S + sl) * OPB + 16]; }
        __syncwarp();
        for (int it = c0 + lane; it < c0 + CG; it += 32)
          if (NL * t + it < nq) { if (mode & 256) rel[it * S + sl] = t / S + 1; else asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(sa(&empty[it * S + sl])) : "memory"); }
      }
    }
  }
  if (acc == 0x12345) sink[0] = acc;
  __syncthreads();
  if (threadIdx.x == 0) atomicMax(&tspan[1], gt());
}

int main(int argc, char** argv) {
  if (argc < 5) { printf("usage: %s MODE Wg L S [cfrac] [iters] [CG]\n", argv[0]); return 2; }
  const int MODE = atoi(argv[1]), Wg = atoi(argv[2]), L = atoi(argv[3]), S = atoi(argv[4]);
  const double cfrac = argc > 5 ? atof(argv[5]) : 0.8;
  const int iters = argc > 6 ? atoi(argv[6]) : 10, CG = argc > 7 ? atoi(argv[7]) : 32;
  const int NREQ = 96, R = 4 * NREQ, nmin = 448, nmax = 576, NL = Wg * L;
  const size_t NKV = (size_t)4 << 20;
  if (NL * S > 256 || NL > 64 || CG > 64 || NL % CG || ((MODE & 8) && (CG > 32 || NL / CG > 16))) { printf("bad cfg\n"); return 2; }
  int nsm, dev = 0; RK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev));
  // same generator as dattn_skel_mla.cu: per request C = cfrac * min n_r common keys + per-row unique keys
  unsigned rs = 777;
  auto rnd = [&]() { rs = rs * 1664525u + 1013904223u; return rs >> 2; };
  std::vector<unsigned> rows;  // gather list (row ids)
  size_t keys_total = 0;
  for (int qi = 0; qi < NREQ; ++qi) {
    int nr[4], mn = 1 << 30;
    for (int r = 0; r < 4; ++r) { nr[r] = nmin + rnd() % (nmax - nmin + 1); mn = std::min(mn, nr[r]); }
    int C = (int)(cfrac * mn);
    std::vector<unsigned> com(C);
    for (auto& v : com) v = rnd() % NKV;
    if (MODE & 1) rows.insert(rows.end(), com.begin(), com.end());
    for (int r = 0; r < 4; ++r) {
      keys_total += nr[r];
      std::vector<unsigned> s(com);
      for (int i = C; i < nr[r]; ++i) s.push_back(rnd() % NKV);
      if (MODE & 1) rows.insert(rows.end(), s.begin() + C, s.end());
      else { rows.insert(rows.end(), s.begin(), s.end()); while (rows.size() % 4) rows.push_back(s.back()); }  // per-row quads
    }
    while (rows.size() % 4) rows.push_back(rows.back());
  }
  const int NQ = (int)(rows.size() / 4);
  int grid = nsm;
  std::vector<int> off(grid + 1);
  for (int c = 0; c <= grid; ++c) off[c] = (int)((long long)NQ * c / grid);
  if (off[1] - off[0] > QMAX) { printf("QMAX\n"); return 2; }
  uint4* dq; int* doff; unsigned long long *dts, *dcnt; unsigned* dsink;
  RK(cudaMalloc(&dq, NQ * 16)); RK(cudaMemcpy(dq, rows.data(), NQ * 16, cudaMemcpyHostToDevice));
  RK(cudaMalloc(&doff, (grid + 1) * 4)); RK(cudaMemcpy(doff, off.data(), (grid + 1) * 4, cudaMemcpyHostToDevice));
  RK(cudaMalloc(&dts, 8 * 1024)); RK(cudaMalloc(&dcnt, 8)); RK(cudaMalloc(&dsink, 4));
  unsigned char* kv; RK(cudaMalloc(&kv, NKV * ROWB)); RK(cudaMemset(kv, 1, NKV * ROWB));
  CUtensorMap kvw;
  cuuint64_t dw[2] = {(cuuint64_t)ROWB / 4, (cuuint64_t)NKV}, strides[1] = {(cuuint64_t)ROWB};
  cuuint32_t bw[2] = {ROWB / 4, 1}, es[2] = {1, 1};
  CK(cuTensorMapEncodeTiled(&kvw, CU_TENSOR_MAP_DATA_TYPE_UINT32, 2, kv, dw, strides, bw, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                            CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  CUtensorMap k128, k64;
  {
    cuuint64_t d8[2] = {(cuuint64_t)ROWB, (cuuint64_t)NKV}; cuuint32_t b128[2] = {128, 1}, b64[2] = {64, 1};
    CK(cuTensorMapEncodeTiled(&k128, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, kv, d8, strides, b128, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                              CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CK(cuTensorMapEncodeTiled(&k64, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, kv, d8, strides, b64, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                              CU_TENSOR_MAP_SWIZZLE_64B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  }
  size_t FL = 1024ull << 20; unsigned char* flush; RK(cudaMalloc(&flush, FL));
  const int qoff = ((off[1] - off[0] + 1) * 16 + 1023) / 1024 * 1024;
  const size_t smem = qoff + (size_t)NL * S * OPB;
  RK(cudaFuncSetAttribute(g2, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  const int threads = (Wg + ((MODE & 2) && !(MODE & 64) ? 1 : 0)) * 32;
  cudaEvent_t e0, e1; RK(cudaEventCreate(&e0)); RK(cudaEventCreate(&e1));
  std::vector<float> span, ev;
  for (int it = 0; it < iters + 2; ++it) {
    if (!(MODE & 4)) RK(cudaMemsetAsync(flush, it, FL));
    unsigned long long init[2] = {~0ull, 0ull};
    RK(cudaMemsetAsync(dts, 0, 8 * 1024)); RK(cudaMemcpyAsync(dts, init, 16, cudaMemcpyHostToDevice)); RK(cudaMemsetAsync(dcnt, 0, 8));
    RK(cudaEventRecord(e0));
    g2<<<grid, threads, smem>>>(kvw, k128, k64, dq, doff, MODE, Wg, L, S, CG, qoff, dts, dcnt, dsink);
    RK(cudaEventRecord(e1)); RK(cudaEventSynchronize(e1)); RK(cudaGetLastError());
    unsigned long long h[2], hc; RK(cudaMemcpy(h, dts, 16, cudaMemcpyDeviceToHost)); RK(cudaMemcpy(&hc, dcnt, 8, cudaMemcpyDeviceToHost));
    if (hc != (unsigned long long)NQ) { printf("COUNT MISMATCH %llu vs %d\n", hc, NQ); return 1; }
    float ms; RK(cudaEventElapsedTime(&ms, e0, e1));
    if (it >= 2) { span.push_back((h[1] - h[0]) / 1000.f); ev.push_back(ms * 1000.f); }
  }
  if (MODE & 16) {  // CTA 0 OOO consumer trace: time(us) chunk tile
    unsigned long long tr[126]; RK(cudaMemcpy(tr, dts + 2, sizeof(tr), cudaMemcpyDeviceToHost));
    printf("TRACE cta0 (us:chunk.tile)");
    for (int i = 0; i < 120 && tr[i]; ++i) printf(" %.2f:%d.%d", (tr[i] >> 16) / 1000.0, (int)((tr[i] >> 8) & 255), (int)(tr[i] & 255));
    printf("\n");
    std::vector<unsigned long long> is(512); RK(cudaMemcpy(is.data(), dts + 130, 8 * 512, cudaMemcpyDeviceToHost));
    printf("ISSUE cta0 per tile min/med/max us:");
    for (int t = 0; t < 8; ++t) { std::vector<double> v; for (int i = 0; i < NL; ++i) if (is[i * 8 + t]) v.push_back(is[i * 8 + t] / 1000.0);
      if (v.empty()) break; std::sort(v.begin(), v.end()); printf(" t%d %.2f/%.2f/%.2f", t, v[0], v[v.size() / 2], v.back()); }
    printf("\n");
  }
  std::sort(span.begin(), span.end()); std::sort(ev.begin(), ev.end());
  double MB = (double)NQ * OPB / 1e6, sp = span[span.size() / 2];
  printf("SKEL2 MODE=%d Wg=%d L=%d S=%d CG=%d cfrac=%.2f %s | rows %zu (keys %zu, %.2fx) %.1f MB smem %zu | span med %.2f us (min %.2f) ev %.2f | %.2f TB/s %.0f GB/s/SM\n",
         MODE, Wg, L, S, CG, cfrac, (MODE & 4) ? "warm" : "cold", rows.size(), keys_total, (double)rows.size() / keys_total, MB, smem,
         sp, span[0], ev[ev.size() / 2], MB / sp, MB / sp / grid * 1e3);
  return 0;
}
