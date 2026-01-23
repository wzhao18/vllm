#include <torch/all.h>
#include <cuda_runtime.h>
#include <sys/mman.h> // Required for mmap
#include <malloc.h>   // Required for malloc_trim if you stay with free
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <c10/util/Logging.h>


double get_available_memory_gb() {
  std::ifstream file("/proc/meminfo");
  if (!file.is_open()) return -1.0;

  std::string line;
  unsigned long mem_available_kb = 0;
  bool found = false;

  while (std::getline(file, line)) {
      if (line.compare(0, 13, "MemAvailable:") == 0) {
          // Format is "MemAvailable:       123456 kB"
          std::stringstream ss(line.substr(13));
          ss >> mem_available_kb;
          found = true;
          break;
      }
  }

  return static_cast<double>(mem_available_kb) / 1048576.0;
}

torch::Tensor get_cuda_view_from_cpu_tensor(torch::Tensor& cpu_tensor) {
  TORCH_CHECK(cpu_tensor.device().is_cpu(), "Input must be a CPU tensor");

  // 1. Handle Pinned Memory (Short-circuit)
  if (cpu_tensor.is_pinned()) {
    void* host_ptr = const_cast<void*>(cpu_tensor.data_ptr());
    void* device_ptr = nullptr;
    cudaError_t err = cudaHostGetDevicePointer(&device_ptr, host_ptr, 0);
    TORCH_CHECK(err == cudaSuccess, "cudaHostGetDevicePointer failed: ", cudaGetErrorString(err));

    return torch::from_blob(
        device_ptr, cpu_tensor.sizes(), cpu_tensor.strides(),
        [base = cpu_tensor](void*) {}, 
        cpu_tensor.options().device(torch::kCUDA)
    );
  }

  // 2. Prepare Allocation Parameters
  
  torch::Tensor contiguous_cpu = cpu_tensor.contiguous();
  size_t nbytes = contiguous_cpu.nbytes();
  long page_size = sysconf(_SC_PAGESIZE);
  size_t aligned_size = (nbytes + page_size - 1) & ~(page_size - 1);

  LOG(WARNING) << "Allocating UVA memory for " << aligned_size << " bytes";
  
  // 3. Use mmap instead of posix_memalign
  // MAP_PRIVATE | MAP_ANONYMOUS creates a private memory buffer not backed by a file.
  void* host_ptr = mmap(nullptr, aligned_size, PROT_READ | PROT_WRITE, 
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  
  if (host_ptr == MAP_FAILED) {
      AT_ERROR("mmap failed to allocate ", aligned_size, " bytes");
  }

  std::memcpy(host_ptr, contiguous_cpu.data_ptr(), nbytes);

  // 4. Register with CUDA
  cudaError_t err = cudaHostRegister(host_ptr, aligned_size, cudaHostRegisterDefault);
  if (err != cudaSuccess) {
      munmap(host_ptr, aligned_size);
      AT_ERROR("cudaHostRegister failed: ", cudaGetErrorString(err));
  }

  void* device_ptr = nullptr;
  cudaHostGetDevicePointer(&device_ptr, host_ptr, 0);

  // 5. The Deleter (Using munmap)
  auto deleter = [host_ptr, aligned_size](void*) {
    double mem_before = get_available_memory_gb();
    
    cudaHostUnregister(host_ptr);
    
    // CRITICAL: munmap returns memory to the KERNEL immediately.
    // std::free(host_ptr) would keep it in the process heap.
    munmap(host_ptr, aligned_size);
    
    double mem_after = get_available_memory_gb();
    LOG(WARNING) << "[UVA Deleter] Released " << (aligned_size / (1024*1024)) << " MB"
              << " | OS Available: " << mem_before << "GB -> " << mem_after << "GB";
  };

  return torch::from_blob(
      device_ptr, contiguous_cpu.sizes(), contiguous_cpu.strides(),
      deleter, contiguous_cpu.options().device(torch::kCUDA)
  );
}