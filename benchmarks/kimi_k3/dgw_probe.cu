// dgw_probe.cu (D-GEMM round 5): HBM rate of the decode-GEMM weight access pattern, W only.
// A GEMM K-major weight tile is a 2D TMA box of 128 B (one K tile of fp8) x R rows read from a row-major
// matrix with row pitch P (7168 B kda_in, 3584 B one K half). Question: does that strided pattern stream
// at the die-local/striped HBM rate of contiguous boxes (hwdb 53: ~18 / ~12 TB/s), or is it the ~11 TB/s
// every MXFP8 kda_in form shows cold?  Each CTA owns one row block and streams its boxes along K through
// an S-stage ring (1 or 2 issuing threads, consumer warp waits FULL and releases EMPTY, no MMA).
// Placement: striped cudaMalloc or die-local (cuMemCreate locality domain, CTA claims a block of its die).
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

__device__ int g_die[1024];
__device__ __forceinline__ unsigned smid() { unsigned r; asm volatile("mov.u32 %0, %%smid;" : "=r"(r)); return r; }
__device__ __forceinline__ unsigned sa(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void wait_par(unsigned b, unsigned ph) {
  asm volatile("{.reg .pred P; W: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1; @!P bra W;}" :: "r"(b), "r"(ph) : "memory");
}
__device__ __forceinline__ void tile2d(unsigned dst, const CUtensorMap* m, int col, int row, unsigned b) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
               :: "r"(dst), "l"(m), "r"(col), "r"(row), "r"(b) : "memory");
}
struct Arm { int pitch, box_rows, stages, issuers, dl, promo; const char* name; };
// blocks: row blocks of box_rows rows; per die: [first, count) in d_split; claims in ctr[die]
__global__ void __launch_bounds__(96, 1) stream_k(const __grid_constant__ CUtensorMap m0, const __grid_constant__ CUtensorMap m1,
    int blocks, int kt, int stages, int issuers, int dl, int split, int box_rows, int contig, int* ctr,
    unsigned long long* sink) {
  extern __shared__ __align__(1024) unsigned char dsm[];
  __shared__ __align__(8) unsigned long long full[32], empty[32];
  __shared__ int claim;
  unsigned char* ring = (unsigned char*)(((size_t)dsm + 1023) & ~(size_t)1023);
  const int w = threadIdx.x / 32, lane = threadIdx.x % 32;
  if (threadIdx.x == 0) {
    for (int i = 0; i < stages; ++i) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(sa(&full[i])));
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(sa(&empty[i])));
    }
    int die = dl ? g_die[smid() & 1023] : 0;
    int got;
    if (!dl) got = atomicAdd(&ctr[0], 1);
    else {
      int lo = die ? split : 0, n = die ? blocks - split : split;
      got = atomicAdd(&ctr[die], 1);
      if (got < n) got += lo;
      else { int o = 1 - die; int g2 = atomicAdd(&ctr[o], 1); int lo2 = o ? split : 0, n2 = o ? blocks - split : split;
             got = g2 < n2 ? lo2 + g2 : blocks; }
    }
    claim = got;
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  const int blk = claim;
  const CUtensorMap* m = dl ? (blk < split ? &m0 : &m1) : &m0;
  const int row0 = dl ? (blk < split ? blk : blk - split) : blk;
  if (blk < blocks) {
    const unsigned SB = 128u * (unsigned)box_rows;  // stage bytes = one box
    if (w >= 1 && w < 1 + issuers && lane == 0) {
      const int p = w - 1;
      for (int i = p; i < kt; i += issuers) {
        const int r = i / stages, sl = i - r * stages;
        if (r > 0) wait_par(sa(&empty[sl]), (unsigned)((r - 1) & 1));
        const unsigned b = sa(&full[sl]);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(SB) : "memory");
        if (contig) tile2d(sa(ring + (size_t)sl * SB), m, 0, (row0 * kt + i) * box_rows, b);
        else tile2d(sa(ring + (size_t)sl * SB), m, i * 128, row0 * box_rows, b);
      }
    } else if (w == 0) {
      unsigned acc = 0; int sl = 0; unsigned ph = 0;
      for (int i = 0; i < kt; ++i) {
        wait_par(sa(&full[sl]), ph);
        if (lane == 0) { acc += ring[(size_t)sl * SB + 64]; asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(sa(&empty[sl])) : "memory"); }
        __syncwarp();
        if (++sl == stages) { sl = 0; ph ^= 1u; }
      }
      if (lane == 0 && acc == 0x12345) sink[0] = acc;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {  // last CTA resets the claim counters
    __threadfence();
    int d = atomicAdd(&ctr[2], 1);
    if (d == (int)gridDim.x - 1) { ctr[0] = 0; ctr[1] = 0; ctr[2] = 0; __threadfence(); }
  }
}

static void* alloc_dom(int dev, int dom, size_t bytes) {
  CUmemAllocationProp p = {}; p.type = CU_MEM_ALLOCATION_TYPE_PINNED; p.location.type = CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN;
  p.location.localized.deviceId = (unsigned char)dev; p.location.localized.localityDomainId = (unsigned char)dom;
  size_t g = 0; CK(cuMemGetAllocationGranularity(&g, &p, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED)); bytes = (bytes + g - 1) / g * g;
  CUmemGenericAllocationHandle h; CK(cuMemCreate(&h, bytes, &p, 0)); CUdeviceptr va; CK(cuMemAddressReserve(&va, bytes, g, 0, 0));
  CK(cuMemMap(va, bytes, 0, h, 0)); CUmemAccessDesc a = {}; a.location.type = CU_MEM_LOCATION_TYPE_DEVICE; a.location.id = dev;
  a.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE; CK(cuMemSetAccess(va, bytes, &a, 1)); return (void*)va;
}
static CUtensorMap tmap(const void* p, size_t rows, int pitch, int box_rows, int promo) {
  CUtensorMap m; cuuint64_t dims[2] = {(cuuint64_t)pitch, (cuuint64_t)rows}; cuuint64_t strides[1] = {(cuuint64_t)pitch};
  cuuint32_t boxd[2] = {128, (cuuint32_t)box_rows}; cuuint32_t es[2] = {1, 1};
  CUtensorMapL2promotion pr[4] = {CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_64B,
                                  CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B};
  CK(cuTensorMapEncodeTiled(&m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, (void*)p, dims, strides, boxd, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                            CU_TENSOR_MAP_SWIZZLE_128B, pr[promo], CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return m;
}
// die of each SM: SM-side latency to a die-0 vs die-1 buffer (hw61 method)
__global__ void lat_k(const unsigned long long* b0, const unsigned long long* b1, size_t words, unsigned* out) {
  if (threadIdx.x) return;
  const unsigned long long* bs[2] = {b0, b1};
  for (int d = 0; d < 2; ++d) {
    size_t idx = (size_t)(blockIdx.x * 7919ull * 1031ull + d * 977) % words; unsigned long long acc = 0; long long t0 = clock64();
    for (int i = 0; i < 32; ++i) { unsigned long long v; asm volatile("ld.global.cv.u64 %0, [%1];" : "=l"(v) : "l"(bs[d] + idx));
      acc += v; idx = (idx + 131088ull + (v & 1)) % words; }
    out[4 * blockIdx.x + 1 + d] = (unsigned)((clock64() - t0) / 32); if (acc == 12345) out[0] = 1;
  }
  out[4 * blockIdx.x] = smid();
}

int main(int argc, char** argv) {
  RK(cudaSetDevice(0)); RK(cudaFree(0));
  int nsm; cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
  const size_t WB = 12448ull * 7168ull;  // kda_in MXFP8 weight bytes
  const int NC = 10;                      // cold copies per placement (920 MB > L2)
  Arm arms[] = {
      {7168, 64, 22, 1, 0, 0, "A1 striped P7168 box 128Bx64 (8K) S22 I1 (FI non-swap W)"},
      {7168, 64, 22, 1, 1, 0, "A1d die-local P7168 box 128Bx64 S22 I1"},
      {3584, 128, 12, 1, 0, 0, "A2 striped P3584 box 128Bx128 (16K) S12 I1 (FI L=2 swap W)"},
      {3584, 128, 12, 1, 1, 0, "A2d die-local P3584 box 128Bx128 S12 I1"},
      {3584, 128, 12, 2, 1, 0, "A2d2 die-local P3584 S12 I2"},
      {3584, 128, 12, 1, 1, 3, "A2dp die-local P3584 S12 I1 promo256"},
      {128, 128, 12, 1, 0, 0, "A3 striped contiguous 16K boxes S12 I1"},
      {128, 128, 12, 1, 1, 0, "A3d die-local contiguous 16K boxes S12 I1"},
      {128, 128, 12, 2, 1, 0, "A3d2 die-local contiguous 16K boxes S12 I2"},
  };
  const int narm = sizeof(arms) / sizeof(arms[0]);
  // buffers: striped NC copies, die-local NC copies per die (each die holds half of every copy)
  unsigned char* st; RK(cudaMalloc(&st, WB * NC)); RK(cudaMemset(st, 0x11, WB * NC));
  unsigned char* d0 = (unsigned char*)alloc_dom(0, 0, WB / 2 * NC + (64 << 20));
  unsigned char* d1 = (unsigned char*)alloc_dom(0, 1, WB / 2 * NC + (64 << 20));
  RK(cudaMemset(d0, 0x22, WB / 2 * NC)); RK(cudaMemset(d1, 0x22, WB / 2 * NC));
  unsigned char* fl; RK(cudaMalloc(&fl, 512 << 20));
  int nb = 8 * nsm; unsigned* lo; RK(cudaMalloc(&lo, nb * 16)); RK(cudaMemset(lo, 0, nb * 16));
  lat_k<<<nb, 32>>>((const unsigned long long*)d0, (const unsigned long long*)d1, (WB / 2) / 8, lo); RK(cudaDeviceSynchronize());
  std::vector<unsigned> hl(nb * 4); RK(cudaMemcpy(hl.data(), lo, nb * 16, cudaMemcpyDeviceToHost));
  std::vector<double> a0(1024, 0), a1(1024, 0); std::vector<int> cnt(1024, 0), die(1024, 0);
  for (int b = 0; b < nb; ++b) { unsigned s = hl[4 * b]; if (s >= 1024) continue; a0[s] += hl[4 * b + 1]; a1[s] += hl[4 * b + 2]; cnt[s]++; }
  int per[2] = {0, 0};
  for (int s = 0; s < 1024; ++s) if (cnt[s]) { die[s] = a0[s] / cnt[s] < a1[s] / cnt[s] ? 0 : 1; per[die[s]]++; }
  RK(cudaMemcpyToSymbol(g_die, die.data(), 1024 * sizeof(int)));
  printf("sms %d die0 %d die1 %d\n", nsm, per[0], per[1]);
  int* ctr; RK(cudaMalloc(&ctr, 64)); RK(cudaMemset(ctr, 0, 64));
  unsigned long long* sink; RK(cudaMalloc(&sink, 8));
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  for (int ai = 0; ai < narm; ++ai) {
    Arm a = arms[ai];
    const size_t rows = WB / a.pitch;                       // whole matrix rows
    const int sbytes = 128 * a.box_rows;
    const int blocks_all = (int)(rows / a.box_rows);        // row blocks (tail rows dropped)
    int kt = a.pitch / 128, blocks = blocks_all;
    if (a.pitch == 128) { kt = 3584 / 128; blocks = (int)(rows / ((size_t)a.box_rows * kt)); }  // contiguous: same bytes/CTA as P3584
    const int split = blocks / 2;
    const size_t smem = (size_t)a.stages * sbytes + 1024;
    RK(cudaFuncSetAttribute(stream_k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    std::vector<CUtensorMap> m0(NC), m1(NC);
    for (int c = 0; c < NC; ++c) {
      if (a.dl) {
        // die-local: die 0 holds blocks [0, split), die 1 the rest, as two matrices
        const size_t r0 = a.pitch == 128 ? (size_t)split * a.box_rows * kt : (size_t)split * a.box_rows;
        const size_t r1 = a.pitch == 128 ? (size_t)(blocks - split) * a.box_rows * kt : (size_t)(blocks - split) * a.box_rows;
        m0[c] = tmap(d0 + (size_t)c * (WB / 2), r0, a.pitch, a.box_rows, a.promo);
        m1[c] = tmap(d1 + (size_t)c * (WB / 2), r1, a.pitch, a.box_rows, a.promo);
      } else {
        m0[c] = tmap(st + (size_t)c * WB, rows, a.pitch, a.box_rows, a.promo); m1[c] = m0[c];
      }
    }
    // contiguous layout: block b covers rows [b*box*kt, +box*kt); box i = rows b*box*kt + i*box -> remap via kt
    std::vector<float> ts;
    for (int rep = 0; rep < 7; ++rep) {
      RK(cudaMemsetAsync(fl, rep, 512 << 20));
      RK(cudaEventRecord(e0));
      for (int c = 0; c < NC; ++c) {
        const int contig = a.pitch == 128;
        stream_k<<<dim3(blocks, 1, 1), dim3(32 * (1 + std::max(1, a.issuers))), smem>>>(
            m0[c], m1[c], blocks, kt, a.stages, a.issuers, a.dl, split, a.box_rows, contig, ctr, sink);
      }
      RK(cudaEventRecord(e1)); RK(cudaEventSynchronize(e1)); RK(cudaGetLastError());
      float ms; cudaEventElapsedTime(&ms, e0, e1); ts.push_back(ms * 1000.f / NC);
    }
    std::sort(ts.begin() + 1, ts.end());
    const double us = ts[1 + (ts.size() - 1) / 2];
    const double bytes = (double)blocks * kt * sbytes;
    printf("%-58s ctas %4d kt %3d  %.2f us/launch  %.2f TB/s  (%.1f MB)\n", a.name, blocks, kt, us, bytes / us / 1e6, bytes / 1e6);
  }
  return 0;
}
