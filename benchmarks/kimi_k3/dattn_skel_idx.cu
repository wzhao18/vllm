// dattn_skel_idx.cu: data-movement-only skeleton of the K3.1 DCP4 decode indexer logits (SOL protocol rule 2).
// Bytes: every request's local index-key blocks (64 tokens x 68 B = 4352 B per paged block) read ONCE for its 4
// verify rows (the 4 rows x heads go on the MMA M side), by 1D bulk copies into per-thread smem rings.
// Placement: striped cudaMalloc pool, or alternating 2 MiB die-homed chunks (VMM locality domains) with a die-matched
// claim (CTAs on die d drain the blocks homed on d, then steal). Die of an SM: (smid >> 1) & 1 (hw6 pair rule;
// skeleton only - production uses the latency-probe map).
// Usage: dattn_skel_idx NREQ NBLK THREADS_PER_CTA DEPTH [iters=10]   (runs striped and die-local)
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char* s; cuGetErrorString(r_, &s); printf("CU error %s line %d\n", s, __LINE__); exit(1); } } while (0)
#define RK(x) do { cudaError_t r_ = (x); if (r_ != cudaSuccess) { printf("CUDA error %s line %d\n", cudaGetErrorString(r_), __LINE__); exit(1); } } while (0)
constexpr int BLK = 4352, CH = 16;  // block bytes, blocks per claim

__device__ __forceinline__ unsigned sa(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void wait_cta(unsigned b, unsigned ph) {
  asm volatile("{.reg .pred P; W: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1; @!P bra W;}" :: "r"(b), "r"(ph) : "memory");
}

// items: block byte offsets, sorted die-0 list then die-1 list; cnt[0..1] claim counters; n0 = size of die-0 list
__global__ void __launch_bounds__(512, 1) skel(const unsigned char* __restrict__ pool, const long long* __restrict__ items, int n0, int nt,
                                               int* cnt, int diem, int D, unsigned long long* stamps, int LPW) {
  extern __shared__ __align__(1024) unsigned char smem[];
  __shared__ __align__(8) unsigned long long bars[512 * 4];
  unsigned sm; asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
  const int die = diem ? (sm >> 1) & 1 : 0;
  // LPW < 32: only lanes < LPW of each warp issue (issuers spread over warps: one stream per warp, hw33f/hw51)
  const int lpw = LPW < 32 ? LPW : 32, tid = (threadIdx.x >> 5) * lpw + (threadIdx.x & 31);
  const bool iss = (int)(threadIdx.x & 31) < lpw;
  if (iss) for (int j = 0; j < D; ++j) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(sa(&bars[tid * D + j])));
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  __syncthreads();
  unsigned long long t0; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  if (iss) {
  unsigned dst0 = sa(smem) + tid * D * BLK;
  int lists[2][2] = {{0, n0}, {n0, nt}};
  int n = 0;
  for (int pass = 0; pass < 2; ++pass) {
    int d = diem ? (pass == 0 ? die : 1 - die) : 0;
    if (!diem && pass == 1) break;
    int lo = diem ? lists[d][0] : 0, hi = diem ? lists[d][1] : nt;
    for (;;) {
      int c = atomicAdd(&cnt[d], CH);
      if (lo + c >= hi) break;
      for (int k = 0; k < CH && lo + c + k < hi; ++k, ++n) {
        int j = n % D; unsigned b = sa(&bars[tid * D + j]);
        if (n >= D) wait_cta(b, ((n - D) / D) & 1);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(BLK) : "memory");
        asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                     :: "r"(dst0 + j * BLK), "l"(pool + items[lo + c + k]), "r"(BLK), "r"(b) : "memory");
      }
    }
  }
  for (int x = (n > D ? n - D : 0); x < n; ++x) wait_cta(sa(&bars[tid * D + x % D]), (x / D) & 1);
  }
  __syncthreads();
  if (tid == 0) { unsigned long long t1; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1)); stamps[2 * blockIdx.x] = t0; stamps[2 * blockIdx.x + 1] = t1; }
}

int main(int argc, char** argv) {
  if (argc < 5) { printf("usage: %s NREQ NBLK THREADS DEPTH [iters]\n", argv[0]); return 2; }
  const int NREQ = atoi(argv[1]), NBLK = atoi(argv[2]), TH = atoi(argv[3]), D = atoi(argv[4]), iters = argc > 5 ? atoi(argv[5]) : 10, LPW = argc > 6 ? atoi(argv[6]) : 32;
  RK(cudaSetDevice(0)); RK(cudaFree(0));
  int nsm; cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
  const int nt = NREQ * NBLK;
  const size_t CHK = 2u << 20;
  size_t pool_bytes = ((size_t)nt * 2 * BLK + CHK - 1) / CHK * CHK;  // pool 2x the used blocks (paged, scattered)
  // striped pool
  unsigned char* ps; RK(cudaMalloc(&ps, pool_bytes)); RK(cudaMemset(ps, 1, pool_bytes));
  // die-homed pool: alternating 2 MiB chunks (chunk k on die k & 1)
  CUdevice dev; CK(cuDeviceGet(&dev, 0));
  CUmemAllocationProp p = {};
  p.type = CU_MEM_ALLOCATION_TYPE_PINNED; p.location.type = CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN;
  CUdeviceptr va; CK(cuMemAddressReserve(&va, pool_bytes, CHK, 0, 0));
  std::vector<CUmemGenericAllocationHandle> hs;
  for (size_t off = 0; off < pool_bytes; off += CHK) {
    p.location.localized.deviceId = 0; p.location.localized.localityDomainId = (int)((off / CHK) & 1);
    CUmemGenericAllocationHandle h; CK(cuMemCreate(&h, CHK, &p, 0)); CK(cuMemMap(va + off, CHK, 0, h, 0)); hs.push_back(h);
  }
  CUmemAccessDesc a = {}; a.location.type = CU_MEM_LOCATION_TYPE_DEVICE; a.location.id = 0; a.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CK(cuMemSetAccess(va, pool_bytes, &a, 1));
  RK(cudaMemset((void*)va, 1, pool_bytes));
  // block placement: random distinct slots of the pool (4352 B aligned to 16)
  size_t nslots = pool_bytes / BLK; std::vector<long long> off(nt);
  unsigned rs = 99; std::vector<char> used(nslots, 0);
  for (int i = 0; i < nt; ++i) { size_t s; do { rs = rs * 1664525u + 1013904223u; s = (rs >> 1) % nslots; } while (used[s]); used[s] = 1; off[i] = (long long)s * BLK; }
  // die of a block = die of its first byte's chunk (a block may straddle a chunk: counted by its first byte)
  std::vector<long long> l0, l1;
  for (auto o : off) ((o / CHK) & 1 ? l1 : l0).push_back(o);
  std::vector<long long> its(l0); its.insert(its.end(), l1.begin(), l1.end());
  long long* di; RK(cudaMalloc(&di, nt * 8)); RK(cudaMemcpy(di, its.data(), nt * 8, cudaMemcpyHostToDevice));
  int* cnt; RK(cudaMalloc(&cnt, 8));
  unsigned long long* st; RK(cudaMalloc(&st, 16 * nsm));
  size_t smem = (size_t)(LPW < 32 ? (TH / 32) * LPW : TH) * D * BLK;
  RK(cudaFuncSetAttribute(skel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  size_t FL = 1024ull << 20; unsigned char* flush; RK(cudaMalloc(&flush, FL));
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  printf("SKEL-IDX NREQ=%d NBLK=%d blocks=%d bytes=%.1f MB TH=%d LPW=%d D=%d smem=%zu die0/die1 blocks %zu/%zu; SOL(18.5 TB/s) %.2f us\n",
         NREQ, NBLK, nt, (double)nt * BLK / 1e6, TH, LPW, D, smem, l0.size(), l1.size(), (double)nt * BLK / 18.5e6);
  for (int mode = 0; mode < 2; ++mode) {
    const unsigned char* pool = mode ? (const unsigned char*)va : ps;
    std::vector<float> tc, tw, sp;
    std::vector<unsigned long long> hst(2 * nsm);
    for (int it = 0; it < 2 * iters + 1; ++it) {
      bool cold = it % 2 == 0;
      if (cold) RK(cudaMemsetAsync(flush, it, FL));
      RK(cudaMemsetAsync(cnt, 0, 8));
      cudaEventRecord(e0);
      skel<<<nsm, TH, smem>>>(pool, di, (int)l0.size(), nt, cnt, mode, D, st, LPW);
      cudaEventRecord(e1); RK(cudaEventSynchronize(e1)); RK(cudaGetLastError());
      float ms; cudaEventElapsedTime(&ms, e0, e1);
      if (it) (cold ? tc : tw).push_back(ms * 1000);
      if (it && cold) { RK(cudaMemcpy(hst.data(), st, 16 * nsm, cudaMemcpyDeviceToHost)); unsigned long long a = ~0ull, b = 0;
        for (int i = 0; i < nsm; ++i) { a = std::min(a, hst[2 * i]); b = std::max(b, hst[2 * i + 1]); } sp.push_back((b - a) / 1e3); }
    }
    std::sort(tc.begin(), tc.end()); std::sort(tw.begin(), tw.end());
    std::sort(sp.begin(), sp.end());
    printf("  SPAN %s: in-kernel cold median %.2f us (%.1f TB/s)\n", mode ? "die-local" : "striped", sp[sp.size() / 2], (double)nt * BLK / sp[sp.size() / 2] / 1e6);
    printf("  RESULT %s: cold median %.2f us (min %.2f, %.1f TB/s) | warm median %.2f us\n", mode ? "die-local+claim" : "striped",
           tc[tc.size() / 2], tc[0], (double)nt * BLK / tc[tc.size() / 2] / 1e6, tw[tw.size() / 2]);
  }
  return 0;
}
