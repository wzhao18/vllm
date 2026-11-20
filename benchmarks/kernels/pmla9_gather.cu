// pmla9_gather.cu (P-MLA-2CTA): all-SM random KV-row gather ceiling on sm_107a. One F16K DSA layer = 16384 tokens x
// 2048 top-k rows of 576 B from a 78K-row (45 MB) KV buffer. Each warp gathers 8 rows per set (4 lanes per row,
// 9 x LDG.128 per lane = 576 B), U sets in flight per thread, W warps per CTA, persistent grid = #SMs.
// Reports us per layer and TB/s for the minimal ingress (each token-key row delivered to an SM exactly once) and for
// the h9 pair form (832 B/key = own full row + 256 B peer half-dims, modelled as 1.444x the rows).
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>

__device__ __forceinline__ uint4 ldg16(const uint8_t* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

template <int U>
__global__ void __launch_bounds__(U == 1 ? 1024 : (U == 2 ? 512 : 256), 1) gather(const uint8_t* __restrict__ kv0, const int* __restrict__ idx, long nsets,
                                                   unsigned* out, const uint8_t* __restrict__ kv1) {
  const int lane = threadIdx.x % 32;
  unsigned smid;
  asm("mov.u32 %0, %%smid;" : "=r"(smid));
  const uint8_t* kv = ((smid >> 1) & 1) ? kv1 : kv0;  // die-local copy (kv1 == kv0 when striped)
  const long gw = (long)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, nw = (long)gridDim.x * (blockDim.x / 32);
  uint32_t acc = 0;
  for (long s0 = gw * U; s0 < nsets; s0 += nw * U) {
    uint4 v[U][9];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const long s = s0 + u < nsets ? s0 + u : nsets - 1;
      const uint8_t* r = kv + (size_t)__ldg(idx + s * 8 + lane / 4) * 576 + 16 * (lane % 4);
#pragma unroll
      for (int j = 0; j < 9; ++j) v[u][j] = ldg16(r + 64 * j);
    }
#pragma unroll
    for (int u = 0; u < U; ++u)
#pragma unroll
      for (int j = 0; j < 9; ++j) acc ^= v[u][j].x ^ v[u][j].y ^ v[u][j].z ^ v[u][j].w;
  }
  if (acc == 0x12345678u) out[0] = acc;
}

template <int U>
void run(int nsm, int W, const uint8_t* kv, const int* idx, long nsets, unsigned* out, const uint8_t* kv1, int CPS, const char* tag) {
  if (32 * W > (U == 1 ? 1024 : (U == 2 ? 512 : 256))) return;
  cudaEvent_t a, b;
  cudaEventCreate(&a);
  cudaEventCreate(&b);
  gather<U><<<CPS * nsm, 32 * W>>>(kv, idx, nsets, out, kv1);
  cudaEventRecord(a);
  for (int r = 0; r < 5; ++r) gather<U><<<CPS * nsm, 32 * W>>>(kv, idx, nsets, out, kv1);
  cudaEventRecord(b);
  cudaEventSynchronize(b);
  float ms;
  cudaEventElapsedTime(&ms, a, b);
  cudaError_t e = cudaGetLastError();
  const double us = ms * 1000 / 5, bytes = (double)nsets * 8 * 576;
  printf("%-8s CPS %d W %2d U %d inflight %6.1f KB/SM : %7.1f us/layer (576 B/key)  %5.1f TB/s  %5.1f B/clk/SM@2.42  h9(832B/key) %7.1f us %s\n",
         tag, CPS, W, U, CPS * W * 32.0 * U * 9 * 16 / 1024, us, bytes / us / 1e6, bytes / us / 1e-6 / nsm / 2.42e9 , us * 832 / 576,
         e == cudaSuccess ? "" : cudaGetErrorString(e));
}

// F2 form: line-aligned, 8 lanes per 128 B line, each LDG.128 instruction touches 4 lines (F0 touches 8)
__global__ void __launch_bounds__(1024, 1) gatherw(const uint8_t* __restrict__ kv0, const int* __restrict__ idx, long nsets,
                                                   unsigned* out, const uint8_t* __restrict__ kv1) {
  const int lane = threadIdx.x % 32, g = lane / 8, l8 = lane % 8;
  unsigned smid;
  asm("mov.u32 %0, %%smid;" : "=r"(smid));
  const uint8_t* kv = ((smid >> 1) & 1) ? kv1 : kv0;
  const long gw = (long)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, nw = (long)gridDim.x * (blockDim.x / 32);
  uint32_t acc = 0;
  for (long s = gw; s < nsets; s += nw) {
    uint4 v[10];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const size_t A = (size_t)__ldg(idx + s * 8 + g + 4 * h) * 576, base = A & ~(size_t)127;
#pragma unroll
      for (int c = 0; c < 5; ++c) {
        const size_t ad = base + 128 * c + 16 * l8;
        v[5 * h + c] = (ad >= A && ad < A + 576) ? ldg16(kv + ad) : make_uint4(0, 0, 0, 0);
      }
    }
#pragma unroll
    for (int j = 0; j < 10; ++j) acc ^= v[j].x ^ v[j].y ^ v[j].z ^ v[j].w;
  }
  if (acc == 0x12345678u) out[0] = acc;
}
void runw(int nsm, int W, const uint8_t* kv, const int* idx, long nsets, unsigned* out, const uint8_t* kv1, int CPS, const char* tag) {
  cudaEvent_t a, b;
  cudaEventCreate(&a);
  cudaEventCreate(&b);
  gatherw<<<CPS * nsm, 32 * W>>>(kv, idx, nsets, out, kv1);
  cudaEventRecord(a);
  for (int r = 0; r < 5; ++r) gatherw<<<CPS * nsm, 32 * W>>>(kv, idx, nsets, out, kv1);
  cudaEventRecord(b);
  cudaEventSynchronize(b);
  float ms;
  cudaEventElapsedTime(&ms, a, b);
  cudaError_t e = cudaGetLastError();
  const double us = ms * 1000 / 5, bytes = (double)nsets * 8 * 576;
  printf("%-8s F2 CPS %d W %2d : %7.1f us/layer (576 B/key)  %5.1f TB/s  %5.1f B/clk/SM@2.42 %s\n", tag, CPS, W, us,
         bytes / us / 1e6, bytes / us / 1e-6 / nsm / 2.42e9, e == cudaSuccess ? "" : cudaGetErrorString(e));
}

// F3 form: cp.async.cg 16 B (LDGSTS) into a per-warp SMEM ring, D sets of 8 rows in flight per warp
template <int D, int Z = 0>
__global__ void __launch_bounds__(1024, 1) gatherc(const uint8_t* __restrict__ kv0, const int* __restrict__ idx, long nsets,
                                                   unsigned* out, const uint8_t* __restrict__ kv1) {
  extern __shared__ __align__(128) unsigned char sm[];
  const int lane = threadIdx.x % 32, w = threadIdx.x / 32;
  unsigned smid;
  asm("mov.u32 %0, %%smid;" : "=r"(smid));
  const uint8_t* kv = ((smid >> 1) & 1) ? kv1 : kv0;
  const long gw = (long)blockIdx.x * (blockDim.x / 32) + w, nw = (long)gridDim.x * (blockDim.x / 32);
  const uint32_t sb = (uint32_t)__cvta_generic_to_shared(sm) + w * D * 4608 + (lane / 4) * 576 + 16 * (lane % 4);
  uint32_t acc = 0;
  int slot = 0;
  // index prefetch: PF sets ahead in registers (the probe was index-latency bound without it)
  constexpr int PF = 4;
  int ix[PF];
#pragma unroll
  for (int k = 0; k < PF; ++k) {
    const long t = gw + k * nw;
    ix[k] = t < nsets ? __ldg(idx + t * 8 + lane / 4) : 0;
  }
  for (long s0 = gw; s0 < nsets + D * nw; s0 += PF * nw) {
#pragma unroll
    for (int k = 0; k < PF; ++k) {
      const long s = s0 + k * nw;
      if (s < nsets) {
        const uint8_t* r = kv + (size_t)ix[k] * 576 + 16 * (lane % 4);
#pragma unroll
        for (int j = 0; j < 9; ++j)
          if (Z)
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(sb + slot * 4608 + 64 * j), "l"(r + 64 * j), "r"(ix[k] >= 0 ? 16 : 0) : "memory");
          else
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(sb + slot * 4608 + 64 * j), "l"(r + 64 * j) : "memory");
      }
      const long t = s + PF * nw;
      ix[k] = t < nsets ? __ldg(idx + t * 8 + lane / 4) : 0;
      asm volatile("cp.async.commit_group;" ::: "memory");
      asm volatile("cp.async.wait_group %0;" ::"n"(D - 1) : "memory");
      slot = slot + 1 == D ? 0 : slot + 1;
      uint32_t x;
      asm volatile("ld.shared.u32 %0, [%1];" : "=r"(x) : "r"(sb + slot * 4608));
      acc ^= x;
    }
  }
  if (acc == 0x12345678u) out[0] = acc;
}
template <int D, int Z = 0>
void runc(int nsm, int W, const uint8_t* kv, const int* idx, long nsets, unsigned* out, const uint8_t* kv1, const char* tag) {
  const int smb = W * D * 4608;
  cudaFuncSetAttribute(gatherc<D, Z>, cudaFuncAttributeMaxDynamicSharedMemorySize, smb);
  cudaEvent_t a, b;
  cudaEventCreate(&a);
  cudaEventCreate(&b);
  gatherc<D, Z><<<nsm, 32 * W, smb>>>(kv, idx, nsets, out, kv1);
  cudaEventRecord(a);
  for (int r = 0; r < 5; ++r) gatherc<D, Z><<<nsm, 32 * W, smb>>>(kv, idx, nsets, out, kv1);
  cudaEventRecord(b);
  cudaEventSynchronize(b);
  float ms;
  cudaEventElapsedTime(&ms, a, b);
  cudaError_t e = cudaGetLastError();
  const double us = ms * 1000 / 5, bytes = (double)nsets * 8 * 576;
  printf("%-8s F3%s W %2d D %d inflight %5.0f KB : %7.1f us/layer  %5.1f TB/s  %5.1f B/clk/SM@2.42 %s\n", tag, Z ? "z" : "", W, D, smb / 1024.0, us,
         bytes / us / 1e6, bytes / us / 1e-6 / nsm / 2.42e9, e == cudaSuccess ? "" : cudaGetErrorString(e));
}

static uint8_t* alloc_dom(size_t bytes, int die) {
  CUmemAllocationProp p = {};
  p.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  p.location.type = CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN;
  p.location.localized.deviceId = 0;
  p.location.localized.localityDomainId = die;
  size_t g;
  cuMemGetAllocationGranularity(&g, &p, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);
  size_t sz = (bytes + g - 1) / g * g;
  CUmemGenericAllocationHandle h;
  if (cuMemCreate(&h, sz, &p, 0) != CUDA_SUCCESS) { printf("cuMemCreate die %d failed\n", die); exit(1); }
  CUdeviceptr va;
  cuMemAddressReserve(&va, sz, g, 0, 0);
  cuMemMap(va, sz, 0, h, 0);
  CUmemAccessDesc a = {};
  a.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  a.location.id = 0;
  a.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  cuMemSetAccess(va, sz, &a, 1);
  int ord = -9;
  cuPointerGetAttribute(&ord, CU_POINTER_ATTRIBUTE_LOCALITY_DOMAIN_ORDINAL, va);
  printf("die %d copy: ordinal %d\n", die, ord);
  cudaMemset((void*)va, 0x11, bytes);
  return (uint8_t*)va;
}

int main(int argc, char** argv) {
  int nsm = 0;
  cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
  const int nrows = argc > 1 ? atoi(argv[1]) : 78 * 1024;
  const long nkeys = 16384L * 2048, nsets = nkeys / 8;
  uint8_t* kv;
  int* idx;
  unsigned* out;
  cudaMalloc(&kv, (size_t)nrows * 576);
  cudaMemset(kv, 0x11, (size_t)nrows * 576);
  std::vector<int> hi(nkeys);
  // per-token top-k: 2048 distinct-ish rows drawn from the token's causal prefix (uniform), tokens spread over ctx
  srand(1);
  for (long t = 0; t < 16384; ++t) {
    const int ctx = nrows - 16384 + (int)t + 1;
    for (int k = 0; k < 2048; ++k) hi[t * 2048 + k] = rand() % ctx;
  }
  cudaMalloc(&idx, nkeys * 4);
  cudaMemcpy(idx, hi.data(), nkeys * 4, cudaMemcpyHostToDevice);
  cudaMalloc(&out, 64);
  printf("SMs %d, KV %d rows (%.1f MB), %ld token-keys; SOL 643 us, target 707 us/layer\n", nsm, nrows,
         nrows * 576 / 1e6, nkeys);
  cudaFree(0);
  uint8_t* d0 = alloc_dom((size_t)nrows * 576, 0);
  uint8_t* d1 = alloc_dom((size_t)nrows * 576, 1);
  runc<6>(nsm, 8, kv, idx, nsets, out, kv, "striped");
  runc<6, 1>(nsm, 8, kv, idx, nsets, out, kv, "striped");
  runc<4, 1>(nsm, 12, kv, idx, nsets, out, kv, "striped");
  return 0;
}
