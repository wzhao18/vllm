// Kimi-K3: FlashKDA csrc/flash_kda.h @ vllm-project/FlashKDA 17a037d98da5 (MIT,
// see LICENSE.FlashKDA), with a `mode` argument (bit 0: K1 prepare, bit 1: K2
// recurrence, bit 2: exact fp32 state I/O for segment partials).
#pragma once

#include <optional>

#include <torch/csrc/stable/tensor.h>

int64_t get_workspace_size(int64_t T_total, int64_t H, int64_t N = 1);

void fwd(const torch::stable::Tensor& q, const torch::stable::Tensor& k,
         const torch::stable::Tensor& v, const torch::stable::Tensor& g,
         const torch::stable::Tensor& beta, double scale,
         const torch::stable::Tensor& out,
         const torch::stable::Tensor& workspace,
         const torch::stable::Tensor& A_log,
         const torch::stable::Tensor& dt_bias, double lower_bound,
         std::optional<torch::stable::Tensor> initial_state = std::nullopt,
         std::optional<torch::stable::Tensor> final_state = std::nullopt,
         std::optional<torch::stable::Tensor> cu_seqlens = std::nullopt,
         std::optional<torch::stable::Tensor> checkpoint_state = std::nullopt,
         std::optional<torch::stable::Tensor> checkpoint_offsets = std::nullopt,
         int64_t mode = 3);

// Kimi-K3: fwd with the Q|K|V short convolution in K1's load path.
void fwd_conv(const torch::stable::Tensor& mixed_qkv,
              const torch::stable::Tensor& conv_weight,
              const torch::stable::Tensor& conv_state,
              const torch::stable::Tensor& cache_indices,
              const torch::stable::Tensor& has_initial_state,
              int64_t null_block_id, const torch::stable::Tensor& v_out,
              const torch::stable::Tensor& g, const torch::stable::Tensor& beta,
              double scale, const torch::stable::Tensor& out,
              const torch::stable::Tensor& workspace,
              const torch::stable::Tensor& A_log,
              const torch::stable::Tensor& dt_bias, double lower_bound,
              std::optional<torch::stable::Tensor> initial_state = std::nullopt,
              std::optional<torch::stable::Tensor> final_state = std::nullopt,
              std::optional<torch::stable::Tensor> cu_seqlens = std::nullopt,
              std::optional<torch::stable::Tensor> checkpoint_state = std::nullopt,
              std::optional<torch::stable::Tensor> checkpoint_offsets = std::nullopt,
              int64_t mode = 3,
              std::optional<torch::stable::Tensor> o_gate = std::nullopt,
              std::optional<torch::stable::Tensor> o_norm_weight = std::nullopt,
              double o_norm_eps = 1e-6,
              std::optional<torch::stable::Tensor> o_q = std::nullopt,
              std::optional<torch::stable::Tensor> o_sf = std::nullopt);
