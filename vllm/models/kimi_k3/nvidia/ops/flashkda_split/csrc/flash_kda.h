// Kimi-K3: FlashKDA csrc/flash_kda.h @ vllm-project/FlashKDA 17a037d98da5 (MIT,
// see LICENSE.FlashKDA), with a `mode` argument (bit 0: K1 prepare, bit 1: K2
// recurrence).
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
