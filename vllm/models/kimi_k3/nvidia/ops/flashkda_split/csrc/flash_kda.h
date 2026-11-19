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
// Kimi-K3: gated o_norm (sigmoid, D = 128) of x (T, H, 128) with the gate rows
// g (T, >= H 128), written as the MXFP8 o_proj input q (T, H 128) e4m3 + the
// F8_128x4 swizzled UE8M0 scales sf.
void o_norm_mxfp8(const torch::stable::Tensor& x, const torch::stable::Tensor& g,
                  const torch::stable::Tensor& weight, double eps,
                  const torch::stable::Tensor& q, const torch::stable::Tensor& sf);
// Kimi-K3 (dev): chunk-64 tcgen05 KDA recurrence on prepared chunk-64 images
// (smxx/k64_tc.cu); flags bit 0: clock stamps into dbg, bit 1: every chunk
// reads chunk 0's images (compute-only timing).
void k64_rec(const torch::stable::Tensor& kdq, const torch::stable::Tensor& kr, const torch::stable::Tensor& inv,
             const torch::stable::Tensor& mqk, const torch::stable::Tensor& gt,
             const torch::stable::Tensor& beta, const torch::stable::Tensor& v,
             const torch::stable::Tensor& init_state,
             const torch::stable::Tensor& final_state,
             const torch::stable::Tensor& out,
             const torch::stable::Tensor& cu_seqlens, int64_t flags,
             const torch::stable::Tensor& dbg);

void prep64(const torch::stable::Tensor& q, const torch::stable::Tensor& k, const torch::stable::Tensor& g, const torch::stable::Tensor& beta, const torch::stable::Tensor& a_log, const torch::stable::Tensor& dt_bias,
            double lower_bound, double scale, const torch::stable::Tensor& cu_seqlens, const torch::stable::Tensor& kdq, const torch::stable::Tensor& kr,
            const torch::stable::Tensor& inv, const torch::stable::Tensor& mqk, const torch::stable::Tensor& gt, const torch::stable::Tensor& beta_out);

void prep64_tc(const torch::stable::Tensor& x, const torch::stable::Tensor& conv_w, const torch::stable::Tensor& conv_state, const torch::stable::Tensor& cache_indices, const torch::stable::Tensor& has_init,
               int64_t null_block_id, const torch::stable::Tensor& g, const torch::stable::Tensor& beta, const torch::stable::Tensor& a_log, const torch::stable::Tensor& dt_bias, double lower_bound,
               double scale, const torch::stable::Tensor& cu_seqlens, const torch::stable::Tensor& kdq, const torch::stable::Tensor& kr, const torch::stable::Tensor& inv, const torch::stable::Tensor& mqk, const torch::stable::Tensor& gt,
               const torch::stable::Tensor& v_img, const torch::stable::Tensor& beta_out, const torch::stable::Tensor& dbg);
