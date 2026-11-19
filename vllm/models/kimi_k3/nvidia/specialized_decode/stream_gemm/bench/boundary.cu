// G-PERSIST boundary microbench: cost of a kernel boundary (grid end + PDL release + next
// grid's CTA launch) vs a grid-wide flag barrier inside one persistent kernel.
// Every CTA occupies a whole SM (big dynamic smem), as the decode GEMMs do.
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

__device__ __forceinline__ unsigned long long gtime() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ unsigned ld_acq(const unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

// spin `ns` per CTA after the dependency; writes one word per CTA (a "result")
__global__ void spin_k(long long ns, int pdl, unsigned* sink, unsigned long long* st) {
  extern __shared__ uint8_t sm[];
  if (pdl && threadIdx.x == 0) asm volatile("griddepcontrol.launch_dependents;");
  if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");
  if (threadIdx.x == 0) {
    unsigned long long t0 = gtime();
    if (st) st[blockIdx.x * 2] = t0;
    while (gtime() < t0 + ns) {
    }
    sink[blockIdx.x] += 1 + sm[0];
    if (st) st[blockIdx.x * 2 + 1] = gtime();
  }
}

// one launch, `phases` phases of `ns` each separated by a grid barrier (monotone counter)
__global__ void persist_k(long long ns, int phases, unsigned* sink, unsigned* ctr,
                          unsigned base) {
  extern __shared__ uint8_t sm[];
  for (int p = 0; p < phases; ++p) {
    if (threadIdx.x == 0) {
      unsigned long long t0 = gtime();
      while (gtime() < t0 + ns) {
      }
      sink[blockIdx.x] += 1 + sm[0];
      __threadfence();
      atomicAdd(ctr, 1u);
      const unsigned target = base + (p + 1) * gridDim.x;
      while (static_cast<int>(ld_acq(ctr) - target) < 0) {
      }
    }
    __syncthreads();
  }
}

// flag barrier: every CTA publishes its epoch with st.release; warp 0 of every CTA polls
// all G flags (G / 32 per lane) with ld.acquire and votes (no atomic serialization)
__global__ void persist_flags_k(long long ns, int phases, unsigned* sink, unsigned* flags,
                                unsigned epoch0) {
  extern __shared__ uint8_t sm[];
  const int G = gridDim.x;
  for (int p = 0; p < phases; ++p) {
    const unsigned want = epoch0 + p + 1;
    if (threadIdx.x == 0) {
      unsigned long long t0 = gtime();
      while (gtime() < t0 + ns) {
      }
      sink[blockIdx.x] += 1 + sm[0];
      asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(flags + blockIdx.x), "r"(want)
                   : "memory");
    }
    if (threadIdx.x < 32) {
      bool ok;
      do {
        ok = true;
        for (int i = threadIdx.x; i < G; i += 32) ok &= static_cast<int>(ld_acq(flags + i) - want) >= 0;
      } while (!__all_sync(0xffffffffu, ok));
    }
    __syncthreads();
  }
}

void persist_flags(int64_t grid, int64_t smem, int64_t ns, int64_t phases, torch::Tensor sink,
                   torch::Tensor flags, int64_t epoch0) {
  cudaFuncSetAttribute(persist_flags_k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  persist_flags_k<<<grid, 128, smem, c10::cuda::getCurrentCUDAStream()>>>(
      ns, phases, reinterpret_cast<unsigned*>(sink.data_ptr()),
      reinterpret_cast<unsigned*>(flags.data_ptr()), (unsigned)epoch0);
}

void spin(int64_t grid, int64_t smem, int64_t ns, int64_t pdl, torch::Tensor sink,
          torch::Tensor st) {
  static int conf = 0;
  if (smem > conf) {
    cudaFuncSetAttribute(spin_k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    cudaFuncSetAttribute(persist_k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    conf = smem;
  }
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(grid);
  cfg.blockDim = dim3(128);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = c10::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  unsigned long long* sp =
      st.numel() ? reinterpret_cast<unsigned long long*>(st.data_ptr()) : nullptr;
  TORCH_CHECK(cudaLaunchKernelEx(&cfg, spin_k, (long long)ns, (int)pdl,
                                 reinterpret_cast<unsigned*>(sink.data_ptr()), sp) ==
              cudaSuccess);
}

void persist(int64_t grid, int64_t smem, int64_t ns, int64_t phases, torch::Tensor sink,
             torch::Tensor ctr, int64_t base) {
  persist_k<<<grid, 128, smem, c10::cuda::getCurrentCUDAStream()>>>(
      ns, phases, reinterpret_cast<unsigned*>(sink.data_ptr()),
      reinterpret_cast<unsigned*>(ctr.data_ptr()), (unsigned)base);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("spin", &spin);
  m.def("persist", &persist);
  m.def("persist_flags", &persist_flags);
}
