// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Bounded spin waits for the Kimi K3.1 custom CUDA kernels. A wait that has not
// completed K31_SPIN_GUARD_NS after it started spinning prints one line
//   K31_SPIN_TIMEOUT kernel=<name> site=<line> rank=<r> call=<n> size=<s> observed=<v>
//     target=<v> smid=<sm> block=<x>,<y> thread=<t>
// and fills record 1 of this process's HOST-MAPPED spin record (spin_record.py, shared
// with the MegaMoE guards: source pidx, site, rank, call, observed, target, smid,
// aux = size, kernel name), then traps. The host watcher logs the record as
// K31_SPIN_RECORD; pinned host memory survives the context loss, device printf output
// is only flushed if the host later synchronizes the dying context.
//
// Scope: only spins written in the Kimi K3.1 kernels are bounded and named; waits inside
// CUTLASS / CuTe library code (pipeline consumer_wait / producer_acquire / producer_tail,
// cute.arch.mbarrier_wait / cluster_wait) are not (documented residual).
//
// Cost: none on the fast path. The clock is first read only after kSpinPolls failed
// polls (a wait that resolves within them never touches it), then every kSpinPolls
// polls. The call counter is one fire-and-forget red.global.add per launch (begin_call)
// by one thread. No value any kernel computes changes: the waits poll the same
// barriers / words in the same order.
#pragma once

#include <cstdint>
#include <cstdio>

#ifndef K31_SPIN_GUARD_NS
// 15 min: far above any legitimate wait (peers compiling, loading or capturing graphs),
// far below a forever-hang. Shorter only on never-merge stall-test overlays.
#define K31_SPIN_GUARD_NS 900000000000ull
#endif

namespace k31sg {

constexpr uint32_t kSpinPolls = 1024;  // failed polls before the first clock read

__device__ __forceinline__ unsigned long long now_ns() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__device__ __forceinline__ unsigned smid() {
  unsigned s;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
  return s;
}

// Per translation unit (static: no host shadow symbol clash between the TUs of one
// extension): the rank (set once by the host: K31SG_DEFINE_SET_RANK), the launch
// counter, the size of the latest launch and the one-printer flag.
static __device__ int g_rank = -1;
static __device__ int* g_diag;  // spin_record.py record 1 (64 int32, host-mapped), or null
static __device__ unsigned long long g_calls;
static __device__ long long g_size;
static __device__ unsigned int g_printed;

// One thread of a launch counts it and records its size (rows / tokens): a stuck
// launch blocks its stream, so the latest launch is the one a timeout line reports.
// Fire-and-forget stores: no wait on the kernel's path.
__device__ __forceinline__ void begin_call(long long size) {
  if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
    asm volatile("red.relaxed.gpu.global.add.u64 [%0], 1;" ::"l"(&g_calls) : "memory");
    asm volatile("st.relaxed.gpu.global.u64 [%0], %1;" ::"l"(&g_size), "l"(size) : "memory");
  }
}

// spin_record.py layout (int32 words): 0 state (0 free, 1 filling, kRecordDone),
// 1 source, 2 site, 3 rank, 4 call, 5 observed, 6 target, 7 smid, 8 aux, 9 entries;
// PIDX records: aux = size, no entries, kernel name in words 48..63.
constexpr int kRecordDone = 0x5350494E;  // 'SPIN'
constexpr int kSourcePidx = 2;
constexpr int kNameWord = 48;

// The first timing-out thread of the TU (g_printed 0 -> 1) fills the record (if this
// process's record 1 is still free) and prints the line, then marks both complete (2)
// and traps. Every other timing-out thread waits for the mark (at most 10 s) before it
// traps: a trap from any thread ends the grid at once and drops a record / printf still
// in flight in the winner (observed: with immediate traps the line never appeared).
static __device__ __noinline__ void timeout(const char* kernel, int site,
                                            unsigned long long observed,
                                            unsigned long long target) {
  if (atomicCAS(&g_printed, 0u, 1u) == 0u) {
    const int rank = *(volatile int*)&g_rank;
    const unsigned long long calls = *(volatile unsigned long long*)&g_calls;
    const long long size = *(volatile long long*)&g_size;
    int* d = *(int* volatile*)&g_diag;
    if (d != nullptr && atomicCAS_system(d, 0, 1) == 0) {
      volatile int* v = d;
      v[1] = kSourcePidx;
      v[2] = site;
      v[3] = rank;
      v[4] = (int)calls;
      v[5] = (int)observed;
      v[6] = (int)target;
      v[7] = (int)smid();
      v[8] = (int)size;
      v[9] = 0;
      unsigned w = 0;  // kernel name, 63 chars max, words kNameWord..63
      int i = 0;
      for (; i < 63 && kernel[i] != 0; ++i) {
        w |= (unsigned)(unsigned char)kernel[i] << (8 * (i & 3));
        if ((i & 3) == 3) { v[kNameWord + (i >> 2)] = (int)w; w = 0; }
      }
      v[kNameWord + (i >> 2)] = (int)w;  // the terminator's word
      __threadfence_system();
      v[0] = kRecordDone;
      __threadfence_system();
    }
    printf("K31_SPIN_TIMEOUT kernel=%s site=%d rank=%d call=%llu size=%lld observed=%llu "
           "target=%llu smid=%u block=%d,%d thread=%d\n",
           kernel, site, rank, calls, size, observed, target, smid(), (int)blockIdx.x,
           (int)blockIdx.y, (int)threadIdx.x);
    __threadfence_system();
    atomicExch(&g_printed, 2u);
  } else {
    const unsigned long long t0 = now_ns();
    while (*(volatile unsigned int*)&g_printed != 2u && now_ns() - t0 < 10000000000ull)
      __nanosleep(1000);
  }
  __trap();
}

// One poll of an mbarrier phase (parity).
__device__ __forceinline__ bool mbar_try(uint32_t bar, uint32_t parity) {
  uint32_t ok;
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
      "selp.u32 %0, 1, 0, p;\n"
      "}\n"
      : "=r"(ok)
      : "r"(bar), "r"(parity)
      : "memory");
  return ok != 0u;
}

// Failed-poll bookkeeping: call after each failed poll.
struct Spin {
  unsigned long long t0 = 0;
  uint32_t n = 0;
  __device__ __forceinline__ void check(const char* kernel, int site,
                                        unsigned long long observed,
                                        unsigned long long target) {
    if (++n % kSpinPolls != 0u) return;
    const unsigned long long t = now_ns();
    // the clock starts once, after kSpinPolls failed polls (%globaltimer is never 0):
    // a wrap of the 32-bit poll count (~430 s of 100 ns polls) must not restart it
    if (t0 == 0) t0 = t;
    else if (t - t0 > K31_SPIN_GUARD_NS) timeout(kernel, site, observed, target);
  }
};

// mbarrier parity wait (shared::cta address), bounded.
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity, const char* kernel,
                                          int site) {
  Spin s;
  while (!mbar_try(bar, parity)) s.check(kernel, site, parity ^ 1u, parity);
}

}  // namespace k31sg

// Host side: each translation unit that includes this header defines its own setter
//   K31SG_DEFINE_SET_RANK(name)  ->  void k31sg_set_rank_name(int rank, int64 diag)
// (g_rank / g_diag are per TU; diag: device address of record 1 of spin_record.py,
// 0: none); the extension's binding calls every setter once after load
// (ops/stable_build.py spin_rank / spin_record_ptr).
#define K31SG_DEFINE_SET_RANK(name)                                             \
  void k31sg_set_rank_##name(int rank, long long diag) {                        \
    int* p = reinterpret_cast<int*>(diag);                                      \
    cudaMemcpyToSymbol(k31sg::g_rank, &rank, sizeof(int));                      \
    cudaMemcpyToSymbol(k31sg::g_diag, &p, sizeof(p));                           \
  }
