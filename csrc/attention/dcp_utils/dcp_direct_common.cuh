// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <string>

#include "../../torch_utils.h"

namespace vllm::direct_dcp {

// Peer-signal waits are bounded by wall-clock time (%globaltimer), not by a
// poll count: a poll count is a hardware-dependent time (it expired in well
// under a minute on some VR200 nodes) and a peer can legitimately lag that long
// while it compiles or loads kernels during startup warmup and graph capture.
// Only a wait longer than this bound is treated as a deadlock. Same bound as
// the CuTe DSL direct A2A combine (dcp_direct_a2a_cutedsl.py).
constexpr uint64_t kWaitTimeoutNs = 600ull * 1000000000ull;
// Read the timer once per this many polls (a power of two minus one).
constexpr uint64_t kTimerPollMask = 1023;

// Advance the invocation ID; its low bit selects one of two staging slots.
static __global__ void increment_epoch_kernel(int64_t* epoch) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    epoch[0] += 1;
  }
}

template <typename T>
__device__ __forceinline__ T* get_peer_ptr(const int64_t* peer_ptrs,
                                           int64_t peer) {
  return reinterpret_cast<T*>(static_cast<uintptr_t>(peer_ptrs[peer]));
}

// Replicate one 16-byte payload to every symmetric-buffer replica.
__device__ __forceinline__ void multimem_store_16(uint4* mc_ptr, uint4 value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1,%2,%3,%4};"
               :
               : "l"(mc_ptr), "r"(value.x), "r"(value.y), "r"(value.z),
                 "r"(value.w)
               : "memory");
#else
  asm volatile("trap;");
#endif
}

// Publish prior system-scope writes and signal every replica.
__device__ __forceinline__ void multimem_store_release_system(uint32_t* mc_ptr,
                                                              uint32_t value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("multimem.st.release.sys.global.u32 [%0], %1;"
               :
               : "l"(mc_ptr), "r"(value)
               : "memory");
#else
  asm volatile("trap;");
#endif
}

__device__ __forceinline__ void store_release_system(uint32_t* ptr,
                                                     uint32_t value) {
  uint64_t address = reinterpret_cast<uint64_t>(ptr);
  asm volatile("st.global.release.sys.u32 [%0], %1;"
               :
               : "l"(address), "r"(value)
               : "memory");
}

__device__ __forceinline__ uint32_t load_acquire_system(const uint32_t* ptr) {
  uint32_t value;
  uint64_t address = reinterpret_cast<uint64_t>(ptr);
  asm volatile("ld.global.acquire.sys.u32 %0, [%1];"
               : "=r"(value)
               : "l"(address)
               : "memory");
  return value;
}

__device__ __forceinline__ uint64_t globaltimer_ns() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__device__ __forceinline__ bool wait_for_epoch(const uint32_t* ptr,
                                               uint32_t epoch) {
  const uint64_t start = globaltimer_ns();
  for (uint64_t spins = 0;; ++spins) {
    if (load_acquire_system(ptr) == epoch) {
      return true;
    }
    if ((spins & kTimerPollMask) == 0 &&
        globaltimer_ns() - start > kWaitTimeoutNs) {
      return false;
    }
  }
}

inline void check_cuda_launch(const char* operation) {
  cudaError_t error = cudaGetLastError();
  STD_TORCH_CHECK(error == cudaSuccess,
                  std::string(operation) +
                      " kernel launch failed: " + cudaGetErrorString(error));
}

}  // namespace vllm::direct_dcp
