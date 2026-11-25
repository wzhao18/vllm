// m1_round.cu: boundary-chain design M1b (probe code, not production). In-kernel Lamport multicast round on one node:
// ngpu real GPUs x E emulated ranks per GPU = NR ranks (4 x 4 = 16). One persistent kernel per GPU runs R back-to-back
// rounds; each emulated rank has C CTAs. Per round every CTA multicasts its 1/C share of its rank's slot (multimem.st
// .relaxed.sys v4 into the multicast object: every GPU receives all NR slots = the 16-rank ingress), then polls its
// 1/C share of the WHOLE box (all NR slots) in local memory, named barrier, next round. No local barrier between CTAs.
// Per-round increment = per-CTA slope of round-end stamps (local %globaltimer, no cross-GPU clock needed).
// Forms: tag16 = every 16 B carries the round tag in .w (12 payload bytes per 16 wire bytes, the design's form);
//        sentinel = Lamport with a 0xFFFFFFFF "empty" sentinel in all 4 words (16 payload bytes per 16), 3 buffers,
//                   rank-0-of-GPU CTAs clear their polled share of buffer (r-1)%3 after round r (TRT-LLM style).
// One process drives all GPUs (hw49 setup).
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

__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
constexpr unsigned SENT = 0xFFFFFFFFu;

struct RB { unsigned* loc; unsigned* mc; int gpu, E, C, S16, R, form, NB, nr; unsigned long long* out; unsigned* err; };
__global__ void __launch_bounds__(512, 1) round_k(RB a) {
  const int j = blockIdx.x, tid = threadIdx.x, er = j / a.C, q = j % a.C;
  const int nr = a.nr;
  const int gr = a.gpu * a.E + er;
  const size_t slotv = (size_t)a.S16, boxv = slotv * nr;   // v4 per slot / per box
  const int w0 = (int)(slotv * q / a.C), w1 = (int)(slotv * (q + 1) / a.C);
  const size_t p0 = boxv * q / a.C, p1 = boxv * (q + 1) / a.C;
  unsigned errs = 0, acc = 0;
  if (tid == 0) a.out[(size_t)j * (a.R + 1)] = gtime();
  for (int r = 1; r <= a.R; ++r) {
    const int b = r % a.NB;
    const size_t base = (size_t)b * boxv * 4;  // u32 offset of buffer b
    for (int w = w0 + tid; w < w1; w += blockDim.x) {
      unsigned* d = a.mc + base + ((size_t)gr * slotv + w) * 4;
      unsigned x0 = (unsigned)(r * 7 + w), x1 = (unsigned)gr, x2 = (unsigned)(w ^ r), x3 = a.form == 1 ? (unsigned)(r * 13 + 1) : (unsigned)r;
      if (a.form == 1) { x0 &= 0x7fffffffu; x1 &= 0x7fffffffu; x2 &= 0x7fffffffu; x3 &= 0x7fffffffu; }  // never the sentinel
      asm volatile("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1,%2,%3,%4};" :: "l"(d), "f"(__uint_as_float(x0)), "f"(__uint_as_float(x1)),
                   "f"(__uint_as_float(x2)), "f"(__uint_as_float(x3)) : "memory");
    }
    if (a.form == 2) {  // tag16, one 16 B poll at a time (v1 form)
      for (size_t v = p0 + tid; v < p1; v += blockDim.x) {
        const unsigned* p = a.loc + base + v * 4; unsigned a0, a1, a2, a3; long long s_ = 0;
        while (true) {
          asm volatile("ld.relaxed.sys.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "l"(p) : "memory");
          if (a3 == (unsigned)r) break; if ((int)(a3 - (unsigned)r) > 0) { ++errs; break; }
          if (++s_ > (1ll << 31)) asm volatile("trap;");
        }
        acc += a0 ^ a2;
      }
    } else
    for (size_t v0 = p0 + tid; v0 < p1; v0 += 8 * blockDim.x) {  // 8 polls in flight per thread, then re-poll stragglers
      uint4 x[8];
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const size_t v = v0 + (size_t)k * blockDim.x;
        if (v < p1) asm volatile("ld.relaxed.sys.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(x[k].x), "=r"(x[k].y), "=r"(x[k].z), "=r"(x[k].w) : "l"(a.loc + base + v * 4) : "memory");
      }
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const size_t v = v0 + (size_t)k * blockDim.x;
        if (v >= p1) continue;
        const unsigned* p = a.loc + base + v * 4; long long s_ = 0;
        while (true) {
          if (a.form == 0) { if (x[k].w == (unsigned)r) break; if ((int)(x[k].w - (unsigned)r) > 0) { ++errs; break; } }
          else if (x[k].x != SENT && x[k].y != SENT && x[k].z != SENT && x[k].w != SENT) break;
          if (++s_ > (1ll << 31)) asm volatile("trap;");
          asm volatile("ld.relaxed.sys.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(x[k].x), "=r"(x[k].y), "=r"(x[k].z), "=r"(x[k].w) : "l"(p) : "memory");
        }
        acc += x[k].x ^ x[k].z;
        if (a.form && er == 0) {
          unsigned* c = a.loc + (size_t)((r - 1) % a.NB) * boxv * 4 + v * 4;
          asm volatile("st.relaxed.sys.global.v4.u32 [%0], {%1,%1,%1,%1};" :: "l"(c), "r"(SENT) : "memory");
        }
      }
    }
    __syncthreads();
    if (tid == 0) a.out[(size_t)j * (a.R + 1) + r] = gtime();
  }
  if (errs) atomicAdd(a.err, errs);
  if (acc == 0x1234567u) a.out[0] = 0;
}

static double pct(std::vector<double> v, double q) { if (v.empty()) return -1; std::sort(v.begin(), v.end()); return v[(size_t)(q * (v.size() - 1))]; }

int main(int argc, char** argv) {
  const int reps = argc > 1 ? atoi(argv[1]) : 3;
  int n = 0; RK(cudaGetDeviceCount(&n));
  if (n < 2) { printf("need >= 2 GPUs\n"); return 0; }
  for (int a = 0; a < n; ++a) { RK(cudaSetDevice(a)); RK(cudaFree(0)); }
  const int NBMAX = 4; const size_t MAXSLOT = 128 * 1024, NRMAX = 16;
  CUmulticastObjectProp mp = {}; mp.numDevices = n; mp.handleTypes = CU_MEM_HANDLE_TYPE_NONE;
  size_t gran = 0; CK(cuMulticastGetGranularity(&gran, &mp, CU_MULTICAST_GRANULARITY_RECOMMENDED));
  const size_t data_bytes = NBMAX * NRMAX * MAXSLOT * 4 / 3 + 65536;
  size_t sz = ((data_bytes + gran - 1) / gran) * gran; mp.size = sz;
  CUmemGenericAllocationHandle mc; CK(cuMulticastCreate(&mc, &mp));
  std::vector<CUdevice> devs(n);
  for (int d = 0; d < n; ++d) { CK(cuDeviceGet(&devs[d], d)); CK(cuMulticastAddDevice(mc, devs[d])); }
  std::vector<CUdeviceptr> uc(n);
  std::vector<CUmemAccessDesc> acc(n);
  for (int e = 0; e < n; ++e) { acc[e] = {}; acc[e].location.type = CU_MEM_LOCATION_TYPE_DEVICE; acc[e].location.id = e; acc[e].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE; }
  for (int d = 0; d < n; ++d) {
    RK(cudaSetDevice(d));
    CUmemAllocationProp ap = {}; ap.type = CU_MEM_ALLOCATION_TYPE_PINNED; ap.location.type = CU_MEM_LOCATION_TYPE_DEVICE; ap.location.id = d;
    CUmemGenericAllocationHandle h; CK(cuMemCreate(&h, sz, &ap, 0));
    CK(cuMulticastBindMem(mc, 0, h, 0, sz, 0));
    CK(cuMemAddressReserve(&uc[d], sz, gran, 0, 0)); CK(cuMemMap(uc[d], sz, 0, h, 0));
    CK(cuMemSetAccess(uc[d], sz, acc.data(), n));
  }
  CUdeviceptr mcva; CK(cuMemAddressReserve(&mcva, sz, gran, 0, 0)); CK(cuMemMap(mcva, sz, 0, mc, 0));
  CK(cuMemSetAccess(mcva, sz, acc.data(), n));
  const int R = 300, R0 = 50;
  std::vector<cudaStream_t> st(n); std::vector<unsigned long long*> out(n); std::vector<unsigned*> err(n);
  for (int d = 0; d < n; ++d) {
    RK(cudaSetDevice(d)); RK(cudaStreamCreateWithFlags(&st[d], cudaStreamNonBlocking));
    RK(cudaMalloc(&out[d], (size_t)128 * (R + 1) * 8)); RK(cudaMalloc(&err[d], 4));
  }
  printf("== M1b in-kernel multicast round: %d GPUs, multicast %zu MiB, R=%d rounds (slope %d..%d), %d reps\n", n, sz >> 20, R, R0, R, reps);
  printf("B form     E  NR  C  payload_B slot_B | per-round us med CTA (rep spread) max CTA | GPU0/GPU%d med | wire B per GPU in | errs\n", n - 1);
  struct Arm { int form, E, C, payload; };
  std::vector<Arm> arms;
  for (int form : {2, 0, 1})
    for (int E : {1, 4})
      for (int payload : {512, 7168, 14336, 57344, 114688})
        arms.push_back({form, E, 16, payload});
  arms.push_back({0, 4, 4, 7168}); arms.push_back({0, 4, 32, 7168}); arms.push_back({0, 4, 8, 512});
  arms.push_back({0, 4, 32, 57344}); arms.push_back({1, 4, 32, 57344}); arms.push_back({1, 4, 32, 114688}); arms.push_back({0, 4, 32, 114688});
  for (auto ar : arms) {
    const int slotB = ar.form == 1 ? ar.payload : (ar.payload * 4 / 3 + 15) / 16 * 16;
    const int S16 = slotB / 16, NR = n * ar.E, NB = ar.form == 1 ? 3 : 4;
    if ((size_t)NB * NR * slotB > sz) { printf("skip (box too large)\n"); continue; }
    std::vector<double> meds, mxs, g0, gl; unsigned errs = 0;
    for (int rep = 0; rep < reps; ++rep) {
      for (int d = 0; d < n; ++d) {
        RK(cudaSetDevice(d)); RK(cudaMemset((void*)uc[d], ar.form == 1 ? 0xFF : 0x00, (size_t)NB * NR * slotB)); RK(cudaMemset(err[d], 0, 4));
        RK(cudaDeviceSynchronize());
      }
      for (int d = 0; d < n; ++d) {
        RK(cudaSetDevice(d));
        RB a; a.loc = (unsigned*)uc[d]; a.mc = (unsigned*)mcva; a.gpu = d; a.E = ar.E; a.C = ar.C; a.S16 = S16; a.R = R; a.form = ar.form; a.NB = NB; a.nr = NR;
        a.out = out[d]; a.err = err[d];
        round_k<<<ar.E * ar.C, 512, 0, st[d]>>>(a);
        RK(cudaGetLastError());
      }
      for (int d = 0; d < n; ++d) { RK(cudaSetDevice(d)); RK(cudaStreamSynchronize(st[d])); }
      std::vector<double> all;
      for (int d = 0; d < n; ++d) {
        RK(cudaSetDevice(d));
        const int nc = ar.E * ar.C; std::vector<unsigned long long> h((size_t)nc * (R + 1));
        RK(cudaMemcpy(h.data(), out[d], h.size() * 8, cudaMemcpyDeviceToHost));
        unsigned e; RK(cudaMemcpy(&e, err[d], 4, cudaMemcpyDeviceToHost)); errs += e;
        std::vector<double> sd;
        for (int c = 0; c < nc; ++c) { const double s = (double)(long long)(h[(size_t)c * (R + 1) + R] - h[(size_t)c * (R + 1) + R0]) / (R - R0) / 1e3; sd.push_back(s); all.push_back(s); }
        if (d == 0) g0.push_back(pct(sd, .5));
        if (d == n - 1) gl.push_back(pct(sd, .5));
      }
      meds.push_back(pct(all, .5)); mxs.push_back(pct(all, 1));
    }
    printf("B %-8s %d %3d %3d %7d %7d | %6.3f (%5.3f-%5.3f) %6.3f | %6.3f / %6.3f | %8d | %u\n", ar.form == 1 ? "sentinel" : ar.form == 2 ? "tag16-1" : "tag16-b8", ar.E, NR, ar.C, ar.payload, slotB,
           pct(meds, .5), pct(meds, 0), pct(meds, 1), pct(mxs, .5), pct(g0, .5), pct(gl, .5), NR * slotB, errs);
    fflush(stdout);
  }
  printf("DONE\n");
  return 0;
}
