// dattn_skel_mla.cu: data-movement-only skeleton of the K3.1 DCP4 sparse MLA decode core (SOL protocol rule 2).
// Same bytes and the same synchronisation as the target kernel, no math:
//   - KV rows (576 B FP8: four 128 B SW128 latent boxes + one 64 B SW64 RoPE box) gathered by TMA tile::gather4 into a
//     ring of 128-key stages (72 KiB), from W issuer warps x L issuing lanes per CTA;
//   - cluster of CL CTAs = CL verify rows of one request (CL=4) or a row pair (CL=2): the keys common to the group are
//     gathered ONCE per cluster with .multicast::cluster (granule g issued by CTA g % CL), the rest per CTA (unicast);
//   - split-K: the concatenated tile list of all groups is cut evenly over all clusters (contiguous ranges); every
//     (group, cluster) segment loads Q (96 heads x 576 B) and writes one partial O (96 x 512 BF16) + LSE;
//   - consumer: waits each stage, releases it to every CTA of the cluster (remote mbarrier arrive), drains the
//     partial O (epilogue warps) before the next segment's first tile.
// Usage: dattn_skel_mla NREQ CL W L [cfrac=0.8] [iters=10] [nmin=448] [nmax=576]
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char* s; cuGetErrorString(r_, &s); printf("CU error %s line %d\n", s, __LINE__); exit(1); } } while (0)
#define RK(x) do { cudaError_t r_ = (x); if (r_ != cudaSuccess) { printf("CUDA error %s line %d\n", cudaGetErrorString(r_), __LINE__); exit(1); } } while (0)

constexpr int ROWB = 576, TN = 128, STAGE = 4 * TN * 128 + TN * 64, QB = 96 * 576, OB = 96 * 512 * 2, MAXPOS = 1024;
constexpr int NCW = 6;  // warp 0 consumer, 1 Q loader, 2..5 epilogue, 6.. issuers

__device__ __forceinline__ unsigned sa(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void wait_cta(unsigned b, unsigned ph) {
  asm volatile("{.reg .pred P; W: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1; @!P bra W;}" :: "r"(b), "r"(ph) : "memory");
}
__device__ __forceinline__ void wait_cl(unsigned b, unsigned ph) {
  asm volatile("{.reg .pred P; W: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P, [%0], %1; @!P bra W;}" :: "r"(b), "r"(ph) : "memory");
}
__device__ __forceinline__ void arrive_tx(unsigned b, unsigned n) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(n) : "memory");
}
__device__ __forceinline__ void arrive(unsigned b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(b) : "memory"); }
__device__ __forceinline__ void arrive_remote(unsigned b, unsigned r) {
  asm volatile("{.reg .b32 ra; mapa.shared::cluster.u32 ra, %0, %1; mbarrier.arrive.release.cluster.shared::cluster.b64 _, [ra];}"
               :: "r"(b), "r"(r) : "memory");
}
__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ __forceinline__ unsigned cl_rank() { unsigned r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r; }
__device__ __forceinline__ unsigned long long gtimer() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ void g4(unsigned dst, const CUtensorMap* m, unsigned bar, int col, int4 r) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
               :: "r"(dst), "l"(m), "r"(col), "r"(r.x), "r"(r.y), "r"(r.z), "r"(r.w), "r"(bar) : "memory");
}
__device__ __forceinline__ void g4mc(unsigned dst, const CUtensorMap* m, unsigned bar, int col, int4 r, unsigned short mask) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4.mbarrier::complete_tx::bytes.multicast::cluster"
               " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
               :: "r"(dst), "l"(m), "r"(col), "r"(r.x), "r"(r.y), "r"(r.z), "r"(r.w), "r"(bar), "h"(mask) : "memory");
}

struct Seg { int gid, t0, t1, slot; };

template <int CL>
__global__ void __launch_bounds__(768, 1) skel(const __grid_constant__ CUtensorMap kvm, const __grid_constant__ CUtensorMap kvr,
                                               const int* __restrict__ seq, const int* __restrict__ cg, const Seg* __restrict__ segs,
                                               const int* __restrict__ cseg, const unsigned char* __restrict__ q,
                                               unsigned char* __restrict__ out, unsigned long long* stamps, int S, int W, int L, int LW, int LP,
                                               const unsigned char* __restrict__ kvp) {
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  unsigned base = (sa(smem_raw) + 1023) & ~1023u;
  unsigned qbuf = base + S * STAGE;
  __shared__ __align__(8) unsigned long long bars[2 * 8 + 3];
  unsigned full0 = sa(&bars[0]), empty0 = sa(&bars[8]), qfull = sa(&bars[16]), epigo = sa(&bars[17]), epidone = sa(&bars[18]);
  const int w = threadIdx.x / 32, lane = threadIdx.x % 32;
  const unsigned rank = CL > 1 ? cl_rank() : 0;
  const int cid = blockIdx.x / CL;
  if (threadIdx.x == 0) {
    for (int s = 0; s < S; ++s) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(full0 + 8 * s), "r"(1 + LW));
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(empty0 + 8 * s), "r"(CL));
    }
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(qfull));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(epigo));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 4;" :: "r"(epidone));
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (CL > 1) cluster_sync(); else __syncthreads();
  unsigned long long t0 = gtimer();
  const int sb = cseg[cid], se = cseg[cid + 1];
  if (w == 0) {
    if (lane == 0) {
      int n = 0;
      for (int j = sb; j < se; ++j) {
        if (j > sb) wait_cta(epidone, (j - sb - 1) & 1);
        wait_cta(qfull, (j - sb) & 1);
        for (int t = segs[j].t0; t < segs[j].t1; ++t, ++n) {
          int s = n % S, u = n / S;
          wait_cta(full0 + 8 * s, u & 1);
          if (CL > 1) { for (int r = 0; r < CL; ++r) arrive_remote(empty0 + 8 * s, r); }
          else arrive(empty0 + 8 * s);
        }
        arrive(epigo);
      }
    }
  } else if (w == 1) {
    if (lane == 0) {
      for (int j = sb; j < se; ++j) {
        if (j > sb) wait_cta(epigo, (j - sb - 1) & 1);
        int row = segs[j].gid * CL + rank;
        arrive_tx(qfull, QB);
        asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                     :: "r"(qbuf), "l"(q + (size_t)row * QB), "r"(QB), "r"(qfull) : "memory");
      }
    }
  } else if (w < NCW) {
    const int et = threadIdx.x - 64;  // 128 epilogue threads
    for (int j = sb; j < se; ++j) {
      wait_cta(epigo, (j - sb) & 1);
      uint4* o = (uint4*)(out + ((size_t)segs[j].slot * CL + rank) * (OB + 512));
      uint4 z = make_uint4(j, 0, 0, 0);
#pragma unroll 4
      for (int i = et; i < (OB + 512) / 16; i += 128) o[i] = z;
      __syncwarp();
      if (lane == 0) arrive(epidone);
    }
  } else if (w < NCW + W) {
    const int li = (w - NCW) * L + lane, NL = W * L;
    if (lane < L) {
      int n = 0;
      for (int j = sb; j < se; ++j) {
        const int gid = segs[j].gid, row = gid * CL + rank, cgr = CL > 1 ? cg[gid] : 0;
        const int* sq = seq + (size_t)row * MAXPOS;
        for (int t = segs[j].t0; t < segs[j].t1; ++t, ++n) {
          int s = n % S, u = n / S;
          unsigned fb = full0 + 8 * s, st = base + s * STAGE;
          if (u > 0) { if (CL > 1) wait_cl(empty0 + 8 * s, (u - 1) & 1); else wait_cta(empty0 + 8 * s, (u - 1) & 1); }
          if (li == 0) {
            int nl = 0, ku = 0;
            for (int g = 0; g < TN / 4; ++g) if (!(t * TN + 4 * g + 4 <= cgr) && LW && (ku++ % 3) < LP) ++nl;
            arrive_tx(fb, STAGE - nl * 4 * ROWB);
          }
          int ku = 0;
          // my granules: multicast ones (pos < cg) owned by g % CL == rank, then all unicast ones
          int k = 0;
          for (int g = 0; g < TN / 4; ++g) {
            int pos = t * TN + 4 * g;
            bool mc = pos + 4 <= cgr;
            if (mc && (g % CL) != (int)rank) continue;
            if (!mc && LW && (ku++ % 3) < LP) continue;
            if ((k++ % NL) != li) continue;
            int4 r = *(const int4*)(sq + pos);
            unsigned d = st + g * 512, dr = st + 4 * TN * 128 + g * 256;
            if (CL > 1 && mc) {
              const unsigned short m = (1u << CL) - 1;
#pragma unroll
              for (int b = 0; b < 4; ++b) g4mc(d + b * TN * 128, &kvm, fb, 128 * b, r, m);
              g4mc(dr, &kvr, fb, 512, r, m);
            } else {
#pragma unroll
              for (int b = 0; b < 4; ++b) g4(d + b * TN * 128, &kvm, fb, 128 * b, r);
              g4(dr, &kvr, fb, 512, r);
            }
          }
        }
      }
    }
  }
  else {  // LDG warps: a share of the unicast granules, 16 B per lane, swizzled st.shared, async-proxy fence
    const int lw = w - NCW - W;
    int n = 0;
    for (int j = sb; j < se; ++j) {
      const int gid = segs[j].gid, row = gid * CL + rank, cgr = CL > 1 ? cg[gid] : 0;
      const int* sq = seq + (size_t)row * MAXPOS;
      for (int t = segs[j].t0; t < segs[j].t1; ++t, ++n) {
        int s = n % S, u = n / S;
        unsigned st = base + s * STAGE;
        if (u > 0) { if (CL > 1) wait_cl(empty0 + 8 * s, (u - 1) & 1); else wait_cta(empty0 + 8 * s, (u - 1) & 1); }
        int ku = 0, kl = 0;
        for (int g = 0; g < TN / 4; ++g) {
          int pos = t * TN + 4 * g;
          if (pos + 4 <= cgr) continue;
          if (!((ku++ % 3) < LP)) continue;
          if ((kl++ % LW) != lw) continue;
          int4 r = *(const int4*)(sq + pos);
          uint4 v[5];
#pragma unroll
          for (int i = 0; i < 5; ++i) {
            int c = lane + 32 * i, jr = c / 36, cc = c % 36;
            int rr = jr == 0 ? r.x : jr == 1 ? r.y : jr == 2 ? r.z : r.w;
            v[i] = make_uint4(0, 0, 0, 0);
            if (c < 144 && rr >= 0) v[i] = __ldcs((const uint4*)(kvp + (size_t)rr * ROWB + cc * 16));
          }
#pragma unroll
          for (int i = 0; i < 5; ++i) {
            int c = lane + 32 * i, jr = c / 36, cc = c % 36, jj = 4 * g + jr;
            if (c >= 144) continue;
            unsigned off = cc < 32 ? (cc / 8) * TN * 128 + jj * 128 + (((cc % 8) ^ (jj & 7)) << 4)
                                   : 4 * TN * 128 + jj * 64 + ((((cc - 32)) ^ ((jj >> 1) & 3)) << 4);
            asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" :: "r"(st + off), "r"(v[i].x), "r"(v[i].y), "r"(v[i].z), "r"(v[i].w) : "memory");
          }
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncwarp();
        if (lane == 0) arrive(full0 + 8 * s);
      }
    }
  }
  if (CL > 1) cluster_sync(); else __syncthreads();
  if (threadIdx.x == 0) { stamps[2 * blockIdx.x] = t0; stamps[2 * blockIdx.x + 1] = gtimer(); }
}

int main(int argc, char** argv) {
  if (argc < 5) { printf("usage: %s NREQ CL W L [cfrac] [iters] [nmin] [nmax] [LW] [LP/3]\n", argv[0]); return 2; }
  const int NREQ = atoi(argv[1]), CL = atoi(argv[2]), W = atoi(argv[3]), L = atoi(argv[4]);
  const double cfrac = argc > 5 ? atof(argv[5]) : 0.8;
  const int iters = argc > 6 ? atoi(argv[6]) : 10, nmin = argc > 7 ? atoi(argv[7]) : 448, nmax = argc > 8 ? atoi(argv[8]) : 576;
  const int LW = argc > 9 ? atoi(argv[9]) : 0, LP = argc > 10 ? atoi(argv[10]) : 0;
  RK(cudaSetDevice(0)); RK(cudaFree(0));
  int nsm, optin; cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
  cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
  int S = std::min(4, (optin - QB - 2048) / STAGE);
  size_t smem = 1024 + (size_t)S * STAGE + QB;
  const int R = 4 * NREQ;
  const size_t NKV = 4u << 20;  // 4M rows x 576 B = 2.4 GB KV pool (HBM-resident)
  unsigned char* kv; RK(cudaMalloc(&kv, NKV * ROWB)); RK(cudaMemset(kv, 1, NKV * ROWB));
  // keys: per request C common keys + per-row unique; per row n_r in [nmin, nmax]
  unsigned rs = 777;
  auto rnd = [&]() { rs = rs * 1664525u + 1013904223u; return rs >> 2; };
  std::vector<int> nr(R), ccount(NREQ);
  std::vector<int> seq((size_t)R * MAXPOS, -1);
  double uniq_rows = 0;
  for (int qi = 0; qi < NREQ; ++qi) {
    int mn = 1 << 30;
    for (int r = 0; r < 4; ++r) { nr[4 * qi + r] = nmin + rnd() % (nmax - nmin + 1); mn = std::min(mn, nr[4 * qi + r]); }
    int C = (int)(cfrac * mn); ccount[qi] = C;
    std::vector<int> com(C);
    for (auto& v : com) v = rnd() % NKV;
    uniq_rows += C;
    for (int r = 0; r < 4; ++r) {
      int* s = &seq[(size_t)(4 * qi + r) * MAXPOS];
      for (int i = 0; i < C; ++i) s[i] = com[i];
      for (int i = C; i < nr[4 * qi + r]; ++i) s[i] = rnd() % NKV;
      uniq_rows += nr[4 * qi + r] - C;
    }
  }
  // groups of CL rows; lockstep tile count = max over the group's rows
  const int G = R / CL;
  std::vector<int> cg(G), T(G);
  long long Wt = 0;
  for (int g = 0; g < G; ++g) {
    int qi = g * CL / 4, mt = 0;
    cg[g] = CL > 1 ? (ccount[qi] / 4) * 4 : 0;
    for (int r = 0; r < CL; ++r) mt = std::max(mt, (nr[g * CL + r] + TN - 1) / TN);
    T[g] = mt; Wt += mt;
  }
  // cluster count: max co-resident
  void* kf = CL == 4 ? (void*)skel<4> : CL == 2 ? (void*)skel<2> : (void*)skel<1>;
  RK(cudaFuncSetAttribute(kf, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  if (CL > 1) RK(cudaFuncSetAttribute(kf, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
  cudaLaunchConfig_t cfg = {};
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = CL; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
  cfg.blockDim = dim3((NCW + W + LW) * 32); cfg.dynamicSmemBytes = smem; cfg.attrs = at; cfg.numAttrs = 1;
  cfg.gridDim = dim3(nsm / CL * CL);
  int NC = nsm / CL;
  if (CL > 1) { int mc = 0; RK(cudaOccupancyMaxActiveClusters(&mc, kf, &cfg)); NC = std::min(NC, mc); }
  cfg.gridDim = dim3(NC * CL);
  // split-K: cluster i gets tiles [i Wt / NC, (i+1) Wt / NC) of the concatenated group tile list
  std::vector<Seg> segs; std::vector<int> cseg(NC + 1);
  std::vector<long long> gofs(G + 1, 0);
  for (int g = 0; g < G; ++g) gofs[g + 1] = gofs[g] + T[g];
  int maxt = 0, maxsegs = 0;
  for (int i = 0; i < NC; ++i) {
    long long a = i * Wt / NC, b = (i + 1) * Wt / NC;
    cseg[i] = segs.size();
    int g = std::upper_bound(gofs.begin(), gofs.end(), a) - gofs.begin() - 1;
    for (long long x = a; x < b; ++g) {
      long long e = std::min(b, gofs[g + 1]);
      segs.push_back({g, (int)(x - gofs[g]), (int)(e - gofs[g]), (int)segs.size()});
      x = e;
    }
    maxt = std::max<int>(maxt, b - a); maxsegs = std::max<int>(maxsegs, segs.size() - cseg[i]);
  }
  cseg[NC] = segs.size();
  int *dseq, *dcg, *dcseg; Seg* dsegs; unsigned char *dq, *dout; unsigned long long* dst;
  RK(cudaMalloc(&dseq, seq.size() * 4)); RK(cudaMemcpy(dseq, seq.data(), seq.size() * 4, cudaMemcpyHostToDevice));
  RK(cudaMalloc(&dcg, G * 4)); RK(cudaMemcpy(dcg, cg.data(), G * 4, cudaMemcpyHostToDevice));
  RK(cudaMalloc(&dcseg, (NC + 1) * 4)); RK(cudaMemcpy(dcseg, cseg.data(), (NC + 1) * 4, cudaMemcpyHostToDevice));
  RK(cudaMalloc(&dsegs, segs.size() * sizeof(Seg))); RK(cudaMemcpy(dsegs, segs.data(), segs.size() * sizeof(Seg), cudaMemcpyHostToDevice));
  RK(cudaMalloc(&dq, (size_t)R * QB)); RK(cudaMemset(dq, 2, (size_t)R * QB));
  RK(cudaMalloc(&dout, segs.size() * CL * (size_t)(OB + 512)));
  RK(cudaMalloc(&dst, 2 * 8 * NC * CL));
  CUtensorMap kvm, kvr;
  cuuint64_t dims[2] = {(cuuint64_t)ROWB, (cuuint64_t)NKV}, strides[1] = {(cuuint64_t)ROWB};
  cuuint32_t box[2] = {128, 1}, boxr[2] = {64, 1}, es[2] = {1, 1};
  CK(cuTensorMapEncodeTiled(&kvm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, kv, dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                            CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  CK(cuTensorMapEncodeTiled(&kvr, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, kv, dims, strides, boxr, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                            CU_TENSOR_MAP_SWIZZLE_64B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  size_t FL = 1024ull << 20; unsigned char* flush; RK(cudaMalloc(&flush, FL));
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  std::vector<float> tc, tw; std::vector<double> span;
  std::vector<unsigned long long> hs(2 * NC * CL);
  for (int it = 0; it < 2 * iters + 1; ++it) {
    bool cold = it % 2 == 0;
    if (cold) RK(cudaMemsetAsync(flush, it, FL));
    cudaEventRecord(e0);
    RK(cudaLaunchKernelEx(&cfg, (void (*)(CUtensorMap, CUtensorMap, const int*, const int*, const Seg*, const int*, const unsigned char*,
                                         unsigned char*, unsigned long long*, int, int, int, int, int, const unsigned char*))kf,
                          kvm, kvr, (const int*)dseq, (const int*)dcg, (const Seg*)dsegs, (const int*)dcseg, (const unsigned char*)dq, dout, dst, S, W, L, LW, LP, (const unsigned char*)kv));
    cudaEventRecord(e1); RK(cudaEventSynchronize(e1));
    float ms; cudaEventElapsedTime(&ms, e0, e1);
    if (it == 0) continue;
    (cold ? tc : tw).push_back(ms * 1000);
    RK(cudaMemcpy(hs.data(), dst, hs.size() * 8, cudaMemcpyDeviceToHost));
    unsigned long long mn = ~0ull, mx = 0;
    for (int b = 0; b < NC * CL; ++b) { mn = std::min(mn, hs[2 * b]); mx = std::max(mx, hs[2 * b + 1]); }
    if (cold) span.push_back((mx - mn) / 1000.0);
  }
  std::sort(tc.begin(), tc.end()); std::sort(tw.begin(), tw.end()); std::sort(span.begin(), span.end());
  double sum_n = 0; for (int x : nr) sum_n += x;
  double deliv = (double)Wt * CL * STAGE, need = sum_n * ROWB;
  printf("SKEL R=%d CL=%d W=%d L=%d LW=%d LP=%d/3 cfrac=%.2f S=%d smem=%zu clusters=%d CTAs=%d tiles_tot=%lld max_tiles/CTA=%d max_segs/CTA=%d\n",
         R, CL, W, L, LW, LP, cfrac, S, smem, NC, NC * CL, Wt * CL, maxt, maxsegs);
  printf("  keys %.0f (KV %.1f MB), unique rows %.0f (HBM %.1f MB), smem delivered %.1f MB; SOL(18.5 TB/s on KV) %.2f us\n",
         sum_n, need / 1e6, uniq_rows, uniq_rows * ROWB / 1e6, deliv / 1e6, need / 18.5e6);
  printf("  RESULT cold median %.2f us (min %.2f) | warm median %.2f us | in-kernel span cold median %.2f us\n",
         tc[tc.size() / 2], tc[0], tw[tw.size() / 2], span[span.size() / 2]);
  return 0;
}
