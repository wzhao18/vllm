// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Standalone diagnostic for the cuMemcpyBatchAsync ABI used by
// csrc/cache_kernels.cu. Build, for example:
//
//   g++ -std=c++17 -I/usr/local/cuda/include \
//     tests/csrc/cumemcpybatch_abi_probe.cpp \
//     -L/usr/local/cuda/compat/lib.real \
//     -Wl,-rpath,/usr/local/cuda/compat/lib.real -lcuda \
//     -o /tmp/cumemcpybatch_abi_probe
//
// The probe exits 0 when the versioned 12.8 ABI path works and the deliberate
// CUDA 13-as-12.8 mismatch fails. It exits 77 when CUDA is unavailable.

#include <cuda.h>
#include <cudaTypedefs.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace {

constexpr int kSkip = 77;

std::string cu_error_name(CUresult result) {
  const char* name = nullptr;
  if (cuGetErrorName(result, &name) == CUDA_SUCCESS && name != nullptr) {
    return name;
  }
  return "CUresult(" + std::to_string(static_cast<int>(result)) + ")";
}

bool check_cu(CUresult result, const char* expr) {
  if (result == CUDA_SUCCESS) {
    return true;
  }
  std::cerr << expr << " failed: " << cu_error_name(result) << "\n";
  return false;
}

struct CudaState {
  CUdevice device = 0;
  CUcontext context = nullptr;
  bool retained_primary_context = false;
  CUdeviceptr src = 0;
  CUdeviceptr dst = 0;
  CUstream stream = nullptr;

  ~CudaState() {
    if (stream != nullptr) {
      cuStreamDestroy(stream);
    }
    if (src != 0) {
      cuMemFree(src);
    }
    if (dst != 0) {
      cuMemFree(dst);
    }
    if (context != nullptr) {
      if (retained_primary_context) {
        cuDevicePrimaryCtxRelease(device);
      } else {
        cuCtxDestroy(context);
      }
    }
  }
};

bool init_cuda(CudaState& state) {
  CUresult result = cuInit(0);
  if (result != CUDA_SUCCESS) {
    std::cerr << "cuInit unavailable: " << cu_error_name(result) << "\n";
    return false;
  }

  CUdevice device;
  if (!check_cu(cuDeviceGet(&device, 0), "cuDeviceGet")) {
    return false;
  }
  state.device = device;
  if (!check_cu(
          cuDevicePrimaryCtxRetain(&state.context, device),
          "cuDevicePrimaryCtxRetain")) {
    return false;
  }
  state.retained_primary_context = true;
  if (!check_cu(cuCtxSetCurrent(state.context), "cuCtxSetCurrent")) {
    return false;
  }
  if (!check_cu(cuMemAlloc(&state.src, 4096), "cuMemAlloc(src)")) {
    return false;
  }
  if (!check_cu(cuMemAlloc(&state.dst, 4096), "cuMemAlloc(dst)")) {
    return false;
  }
  if (!check_cu(
          cuStreamCreate(&state.stream, CU_STREAM_NON_BLOCKING),
          "cuStreamCreate")) {
    return false;
  }
  if (!check_cu(cuMemsetD8Async(state.src, 0x5a, 4096, state.stream),
                "cuMemsetD8Async")) {
    return false;
  }
  return check_cu(cuStreamSynchronize(state.stream), "cuStreamSynchronize");
}

int run_old_12080_abi() {
  CudaState state;
  if (!init_cuda(state)) {
    return kSkip;
  }

  void* ptr = nullptr;
  CUdriverProcAddressQueryResult status;
  CUresult result = cuGetProcAddress(
      "cuMemcpyBatchAsync", &ptr, 12080, CU_GET_PROC_ADDRESS_DEFAULT, &status);
  std::cout << "old ABI cuGetProcAddress result=" << cu_error_name(result)
            << " status=" << static_cast<int>(status) << " ptr=" << ptr
            << "\n";
  if (result != CUDA_SUCCESS || ptr == nullptr) {
    return 10;
  }

  auto fn = reinterpret_cast<PFN_cuMemcpyBatchAsync_v12080>(ptr);
  CUdeviceptr dsts[1] = {state.dst};
  CUdeviceptr srcs[1] = {state.src};
  size_t sizes[1] = {4096};
  CUmemcpyAttributes attr = {};
  attr.srcAccessOrder = CU_MEMCPY_SRC_ACCESS_ORDER_STREAM;
  size_t attr_idxs[1] = {0};
  size_t fail_idx = 1234;

  result = fn(dsts, srcs, sizes, 1, &attr, attr_idxs, 1, &fail_idx,
              state.stream);
  if (!check_cu(result, "cuMemcpyBatchAsync_v12080")) {
    std::cerr << "fail_idx=" << fail_idx << "\n";
    return 11;
  }
  return check_cu(cuStreamSynchronize(state.stream), "old ABI synchronize")
             ? 0
             : 12;
}

int run_new_13000_abi() {
#if CUDA_VERSION >= 13000
  CudaState state;
  if (!init_cuda(state)) {
    return kSkip;
  }

  void* ptr = nullptr;
  CUdriverProcAddressQueryResult status;
  CUresult result = cuGetProcAddress(
      "cuMemcpyBatchAsync", &ptr, 13000, CU_GET_PROC_ADDRESS_DEFAULT, &status);
  std::cout << "new ABI cuGetProcAddress result=" << cu_error_name(result)
            << " status=" << static_cast<int>(status) << " ptr=" << ptr
            << "\n";
  if (result != CUDA_SUCCESS || ptr == nullptr) {
    return 20;
  }

  auto fn = reinterpret_cast<PFN_cuMemcpyBatchAsync_v13000>(ptr);
  CUdeviceptr dsts[1] = {state.dst};
  CUdeviceptr srcs[1] = {state.src};
  size_t sizes[1] = {4096};
  CUmemcpyAttributes attr = {};
  attr.srcAccessOrder = CU_MEMCPY_SRC_ACCESS_ORDER_STREAM;
  size_t attr_idxs[1] = {0};

  result = fn(dsts, srcs, sizes, 1, &attr, attr_idxs, 1, state.stream);
  if (!check_cu(result, "cuMemcpyBatchAsync_v13000")) {
    return 21;
  }
  return check_cu(cuStreamSynchronize(state.stream), "new ABI synchronize")
             ? 0
             : 22;
#else
  std::cerr << "CUDA headers are older than 13.0; no v13000 ABI to test.\n";
  return kSkip;
#endif
}

int run_13000_pointer_with_12080_signature() {
#if CUDA_VERSION >= 13000
  CudaState state;
  if (!init_cuda(state)) {
    return kSkip;
  }

  void* ptr = nullptr;
  CUdriverProcAddressQueryResult status;
  CUresult result = cuGetProcAddress(
      "cuMemcpyBatchAsync", &ptr, 13000, CU_GET_PROC_ADDRESS_DEFAULT, &status);
  std::cout << "mismatched ABI cuGetProcAddress result="
            << cu_error_name(result) << " status=" << static_cast<int>(status)
            << " ptr=" << ptr << "\n";
  if (result != CUDA_SUCCESS || ptr == nullptr) {
    return 30;
  }

  // This is intentionally wrong: a CUDA 13 function pointer is invoked with
  // the CUDA 12.8 signature, so the fail_idx argument is interpreted as the
  // stream by the callee. Run only in a child process.
  auto fn = reinterpret_cast<PFN_cuMemcpyBatchAsync_v12080>(ptr);
  CUdeviceptr dsts[1] = {state.dst};
  CUdeviceptr srcs[1] = {state.src};
  size_t sizes[1] = {4096};
  CUmemcpyAttributes attr = {};
  attr.srcAccessOrder = CU_MEMCPY_SRC_ACCESS_ORDER_STREAM;
  size_t attr_idxs[1] = {0};
  size_t fail_idx = 1234;

  result = fn(dsts, srcs, sizes, 1, &attr, attr_idxs, 1, &fail_idx,
              state.stream);
  if (result != CUDA_SUCCESS) {
    std::cerr << "mismatched call failed as expected: "
              << cu_error_name(result) << " fail_idx=" << fail_idx << "\n";
    return 31;
  }
  if (!check_cu(cuStreamSynchronize(state.stream), "mismatched synchronize")) {
    return 32;
  }
  std::cerr << "mismatched call unexpectedly succeeded.\n";
  return 0;
#else
  std::cerr << "CUDA headers are older than 13.0; no mismatch to test.\n";
  return kSkip;
#endif
}

int run_h2d_pinned_any_12080() {
  CudaState state;
  if (!init_cuda(state)) {
    return kSkip;
  }

  void* ptr = nullptr;
  CUdriverProcAddressQueryResult status;
  CUresult result = cuGetProcAddress(
      "cuMemcpyBatchAsync", &ptr, 12080, CU_GET_PROC_ADDRESS_DEFAULT, &status);
  std::cout << "H2D ANY cuGetProcAddress result=" << cu_error_name(result)
            << " status=" << static_cast<int>(status) << " ptr=" << ptr
            << "\n";
  if (result != CUDA_SUCCESS || ptr == nullptr) {
    return 40;
  }

  constexpr size_t count = 1024;
  constexpr size_t bytes = 4096;
  constexpr size_t total_bytes = count * bytes;

  void* host_src = nullptr;
  if (!check_cu(cuMemAllocHost(&host_src, total_bytes), "cuMemAllocHost(src)")) {
    return 41;
  }
  std::memset(host_src, 0x7c, total_bytes);

  CUdeviceptr device_dst = 0;
  if (!check_cu(cuMemAlloc(&device_dst, total_bytes), "cuMemAlloc(dst)")) {
    cuMemFreeHost(host_src);
    return 42;
  }

  std::vector<CUdeviceptr> dsts(count);
  std::vector<CUdeviceptr> srcs(count);
  std::vector<size_t> sizes(count, bytes);
  for (size_t i = 0; i < count; ++i) {
    dsts[i] = device_dst + i * bytes;
    srcs[i] =
        reinterpret_cast<CUdeviceptr>(static_cast<char*>(host_src) + i * bytes);
  }

  auto fn = reinterpret_cast<PFN_cuMemcpyBatchAsync_v12080>(ptr);
  CUmemcpyAttributes attr = {};
  attr.srcAccessOrder = CU_MEMCPY_SRC_ACCESS_ORDER_ANY;
  size_t attr_idxs[1] = {0};
  size_t fail_idx = 1234;

  result = fn(dsts.data(), srcs.data(), sizes.data(), count, &attr, attr_idxs,
              1, &fail_idx, state.stream);
  bool ok = check_cu(result, "H2D ANY cuMemcpyBatchAsync_v12080") &&
            check_cu(cuStreamSynchronize(state.stream), "H2D ANY synchronize");
  cuMemFree(device_dst);
  cuMemFreeHost(host_src);
  return ok ? 0 : 43;
}

struct ChildResult {
  bool signaled = false;
  int signal = 0;
  int exit_code = 0;
};

ChildResult run_child(int (*fn)()) {
  pid_t pid = fork();
  if (pid < 0) {
    std::perror("fork");
    std::exit(3);
  }
  if (pid == 0) {
    std::exit(fn());
  }

  int status = 0;
  if (waitpid(pid, &status, 0) < 0) {
    std::perror("waitpid");
    std::exit(4);
  }
  ChildResult result;
  if (WIFSIGNALED(status)) {
    result.signaled = true;
    result.signal = WTERMSIG(status);
  } else if (WIFEXITED(status)) {
    result.exit_code = WEXITSTATUS(status);
  } else {
    result.exit_code = 5;
  }
  return result;
}

void print_result(const char* name, const ChildResult& result) {
  std::cout << name << ": ";
  if (result.signaled) {
    std::cout << "signal " << result.signal << "\n";
  } else {
    std::cout << "exit " << result.exit_code << "\n";
  }
}

bool failed(const ChildResult& result) {
  return result.signaled || result.exit_code != 0;
}

}  // namespace

int main() {
  int driver_version = 0;
  CUresult version_result = cuDriverGetVersion(&driver_version);
  if (version_result == CUDA_SUCCESS) {
    std::cout << "CUDA headers=" << CUDA_VERSION
              << " driver=" << driver_version << "\n";
  } else {
    std::cout << "cuDriverGetVersion failed: "
              << cu_error_name(version_result) << "\n";
  }

  ChildResult old_result = run_child(run_old_12080_abi);
  ChildResult new_result = run_child(run_new_13000_abi);
  ChildResult mismatch_result = run_child(run_13000_pointer_with_12080_signature);
  ChildResult h2d_any_result = run_child(run_h2d_pinned_any_12080);
  print_result("old_12080_abi", old_result);
  print_result("new_13000_abi", new_result);
  print_result("mismatched_13000_as_12080", mismatch_result);
  print_result("h2d_pinned_any_12080", h2d_any_result);

  if (old_result.exit_code == kSkip && new_result.exit_code == kSkip &&
      mismatch_result.exit_code == kSkip && h2d_any_result.exit_code == kSkip) {
    return kSkip;
  }
  if (!failed(old_result) && !failed(new_result) && failed(mismatch_result) &&
      !failed(h2d_any_result)) {
    std::cout << "Confirmed: CUDA 13's entry point is not old-signature "
                 "compatible, but cuGetProcAddress(..., 12080) returned a "
                 "working CUDA 12.8 ABI entry point in this environment. "
                 "Pinned H2D copies with srcAccessOrder=ANY also succeeded.\n";
    return 0;
  }
  if (failed(h2d_any_result)) {
    std::cerr << "Confirmed: versioned 12.8 ABI works for simple D2D, but "
                 "failed for pinned H2D with srcAccessOrder=ANY.\n";
    return 3;
  }
  if (failed(old_result) && !failed(new_result)) {
    std::cout << "Confirmed: the old 12.8 ABI path fails while the CUDA 13 "
                 "ABI path succeeds.\n";
    return 0;
  }
  if (!failed(old_result) && !failed(new_result) && !failed(mismatch_result)) {
    std::cerr << "Inconclusive: even the deliberate ABI mismatch succeeded.\n";
    return 1;
  }

  std::cerr << "Inconclusive: the CUDA 13 ABI path failed too.\n";
  return 2;
}
