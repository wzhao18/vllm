#include <torch/all.h>
#include <torch/cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <memory>
#include <iostream>
#include <fstream>
#include <string>
#include <unistd.h>

double get_process_rss_mb() {
  long rss = 0;
  std::ifstream stat_stream("/proc/self/statm", std::ios_base::in);
  if (stat_stream.is_open()) {
      stat_stream >> rss; // The second entry in statm is the RSS in pages
      stat_stream.close();
  }
  return (rss * sysconf(_SC_PAGESIZE)) / (1024.0 * 1024.0);
}

// This function assumes that `cpu_tensor` is a CPU tensor, and that
// UVA (Unified Virtual Addressing) is enabled.
torch::Tensor get_cuda_view_from_cpu_tensor(torch::Tensor& cpu_tensor) {
  TORCH_CHECK(cpu_tensor.device().is_cpu(), "Input must be a CPU tensor");

  if (cpu_tensor.is_pinned()) {
    // For pinned memory, we can get the device pointer immediately
    
    // Get raw host pointer from CPU tensor
    void* host_ptr = const_cast<void*>(cpu_tensor.data_ptr());
    
    // Get a device pointer corresponding to the pinned host memory
    void* device_ptr = nullptr;
    cudaError_t err = cudaHostGetDevicePointer(&device_ptr, host_ptr, 0);
    TORCH_CHECK(err == cudaSuccess,
                "cudaHostGetDevicePointer failed: ", cudaGetErrorString(err));

    // Return a view that aliases the original CPU tensor's storage
    // This ensures the CPU tensor's lifetime keeps the memory alive
    return torch::from_blob(
        device_ptr,
        cpu_tensor.sizes(),
        cpu_tensor.strides(),
        [base = cpu_tensor](void*) {}, // Keep cpu_tensor alive until deleter is called
        cpu_tensor.options().device(torch::kCUDA)
    );
  }

  // If cpu_tensor is not pinned, allocate page-aligned memory and register with CUDA
  torch::Tensor contiguous_cpu = cpu_tensor.contiguous();
  
  static const size_t page_size = []() {
      long sz = sysconf(_SC_PAGESIZE);
      TORCH_CHECK(sz > 0, "sysconf(_SC_PAGESIZE) failed");
      return sz;
  }();

  size_t nbytes = contiguous_cpu.nbytes();
  // Round up to the nearest page size
  size_t aligned_size = (nbytes + page_size - 1) & ~(page_size - 1);

  void* host_ptr = nullptr;
  int res = posix_memalign(&host_ptr, page_size, aligned_size);
  TORCH_CHECK(res == 0, "Failed to allocate page-aligned memory");

  std::memcpy(host_ptr, contiguous_cpu.data_ptr(), nbytes);

  cudaError_t err = cudaHostRegister(host_ptr, aligned_size, cudaHostRegisterMapped);
  if (err != cudaSuccess) {
      std::free(host_ptr);
      AT_ERROR("cudaHostRegister failed: ", cudaGetErrorString(err));
  }

  void* device_ptr = nullptr;
  err = cudaHostGetDevicePointer(&device_ptr, host_ptr, 0);
  if (err != cudaSuccess) {
      cudaHostUnregister(host_ptr);
      std::free(host_ptr);
      AT_ERROR("cudaHostGetDevicePointer failed: ", cudaGetErrorString(err));
  }

  auto deleter = [host_ptr](void*) {
    double mem_before = get_process_rss_mb();
    
    cudaError_t err = cudaHostUnregister(host_ptr);
    std::free(host_ptr);
    
    double mem_after = get_process_rss_mb();

    std::cout << "[UVA Deleter] Unregistering " << host_ptr 
              << " | Process RSS: " << mem_before << "MB -> " 
              << mem_after << "MB" << std::endl;

    if (err != cudaSuccess) {
        std::cerr << "CRITICAL: cudaHostUnregister failed in deleter: " 
                  << cudaGetErrorString(err) << std::endl;
    }
  };

  return torch::from_blob(
      device_ptr,
      contiguous_cpu.sizes(),
      contiguous_cpu.strides(),
      deleter,
      contiguous_cpu.options().device(torch::kCUDA)
  );
}
