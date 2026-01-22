#include <torch/all.h>
#include <torch/cuda.h>
#include <cuda_runtime.h>
#include <cstdlib>

size_t round_up(size_t n, size_t alignment) {
  return (n + alignment - 1) & ~(alignment - 1);
}

// This function assumes that `cpu_tensor` is a CPU tensor, and that UVA (Unified Virtual Addressing) is enabled.
torch::Tensor get_cuda_view_from_cpu_tensor(torch::Tensor& cpu_tensor) {
  TORCH_CHECK(cpu_tensor.device().is_cpu(), "Input tensor must be on CPU");
  TORCH_CHECK(cpu_tensor.is_contiguous(), "Tensor must be contiguous for simple registration");

  // Get raw host pointer from CPU tensor
  void* host_ptr = cpu_tensor.data_ptr();

  size_t nbytes = cpu_tensor.nbytes();
  host_ptr = std::aligned_alloc(4096, round_up(nbytes, 4096));
  if (!host_ptr) {
    throw std::runtime_error("Failed to allocate page-aligned memory");
  }

  cudaError_t err = cudaHostRegister(host_ptr, nbytes, cudaHostRegisterDefault);
  TORCH_CHECK(err == cudaSuccess,
              "cudaHostRegister failed: ", cudaGetErrorString(err));
  
  // Get a device pointer corresponding to the pinned host memory
  void* device_ptr = nullptr;
  err = cudaHostGetDevicePointer(&device_ptr, host_ptr, 0);
  TORCH_CHECK(err == cudaSuccess,
              "cudaHostGetDevicePointer failed: ", cudaGetErrorString(err));

  // We'll use the same sizes, strides, and dtype as the CPU tensor.
  // TODO: check if layout is respected.
  auto sizes = cpu_tensor.sizes();
  auto strides = cpu_tensor.strides();
  auto options = cpu_tensor.options().device(torch::kCUDA);

  // use default no-op deleter, since the memory is owned by the original CPU
  // tensor
  torch::Tensor cuda_tensor =
      torch::from_blob(device_ptr, sizes, strides, options);

  TORCH_CHECK(cuda_tensor.device().is_cuda(),
              "Resulting tensor is not on CUDA device");

  return cuda_tensor;
}
