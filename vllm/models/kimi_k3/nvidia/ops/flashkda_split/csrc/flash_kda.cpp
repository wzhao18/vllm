// Kimi-K3: FlashKDA flash_kda.cpp @ vllm-project/FlashKDA 17a037d (MIT, see
// LICENSE.FlashKDA), changed only to pass `mode` (bit 0: K1 prepare, bit 1: K2
// recurrence, bit 2: exact fp32 state I/O for the segment-parallel prefill's
// partials; without it fp32 states round to bf16 at the kernel boundary exactly
// as FlashKDA, bits 3 / 4: K2 without outputs / with v = 0) and to dispatch
// varlen int32 only.
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor_inl.h>

#include "flash_kda.h"
#include "fwd.h"
#include "norm_args.h"

using torch::headeronly::ScalarType;
using torch::stable::Tensor;

int64_t get_workspace_size(int64_t T_total, int64_t H, int64_t N) {
  constexpr int CHUNK = 16;
  constexpr int D = 128;

  // Upper bound: each of N sequences adds at most 1 extra tile vs floor
  // division
  int64_t total_tiles = (T_total + CHUNK - 1) / CHUNK + N;

  static_assert(CHUNK * D * 2 % 128 == 0,
                "k_decayed/q_decayed/k_restored size must be 128-byte aligned");
  static_assert(D * 4 % 128 == 0, "g_total size must be 128-byte aligned");
  static_assert(CHUNK * CHUNK * 2 % 128 == 0,
                "INV/Mqk size must be 128-byte aligned");

  int64_t per_tile_bytes = 3 * CHUNK * D * 2 + D * 4 + 2 * CHUNK * CHUNK * 2;

  // Kimi-K3: + one ready flag per (head, tile) for the overlapped K1 / K2.
  return H * total_tiles * (per_tile_bytes + 4);
}

void fwd(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
         const Tensor& beta, double scale, const Tensor& out,
         const Tensor& workspace, const Tensor& A_log, const Tensor& dt_bias,
         double lower_bound, std::optional<Tensor> initial_state,
         std::optional<Tensor> final_state, std::optional<Tensor> cu_seqlens,
         std::optional<Tensor> checkpoint_state,
         std::optional<Tensor> checkpoint_offsets, int64_t mode) {
  // mode: bit 0 K1 prepare, bit 1 K2 recurrence, bit 2 exact fp32 state I/O,
  // bit 3 K2 without outputs, bit 4 K2 with v = 0 (bits 2-4 need bit 1).
  // Kimi-K3: bit 5 tcgen05 recurrence (bit 6: its kr descriptor variant).
  // Kimi-K3: bit 9 overlaps K1 with K2 (needs bits 0 and 1 and this build's
  // workspace size); bits 16-30 carry the longest sequence's chunk count.
  const int64_t mode_lo = mode & 0xFFFF;
  STD_TORCH_CHECK(mode >= 0 && mode < (int64_t(1) << 31) &&
                      (mode_lo & ~int64_t(1023)) == 0 && (mode_lo & 3) != 0 &&
                      ((mode_lo & 1020) == 0 || (mode_lo & 2) != 0) &&
                      ((mode_lo & 512) == 0 || (mode_lo & 3) == 3),
                  "k3_flashkda_split: invalid mode ", mode);
  STD_TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && g.is_cuda() &&
                      beta.is_cuda() && out.is_cuda() && workspace.is_cuda(),
                  "all tensors must be on CUDA");
  STD_TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() &&
                      g.is_contiguous() && out.is_contiguous() &&
                      workspace.is_contiguous(),
                  "q, k, v, g, out, and workspace must be contiguous");
  STD_TORCH_CHECK(q.scalar_type() == ScalarType::BFloat16,
                  "q must be bfloat16");
  STD_TORCH_CHECK(k.scalar_type() == ScalarType::BFloat16,
                  "k must be bfloat16");
  STD_TORCH_CHECK(v.scalar_type() == ScalarType::BFloat16,
                  "v must be bfloat16");
  STD_TORCH_CHECK(g.scalar_type() == ScalarType::BFloat16,
                  "g must be bfloat16");
  STD_TORCH_CHECK(beta.scalar_type() == ScalarType::BFloat16,
                  "beta must be bfloat16");
  STD_TORCH_CHECK(out.scalar_type() == ScalarType::BFloat16,
                  "out must be bfloat16");

  // Validate state tensors if present
  bool has_state_in = initial_state.has_value();
  bool has_state_out = final_state.has_value();
  bool has_checkpoint = checkpoint_state.has_value();
  STD_TORCH_CHECK(
      has_checkpoint == checkpoint_offsets.has_value(),
      "checkpoint_state and checkpoint_offsets must be provided together");
  bool state_fp32 = false;

  if (has_state_in) {
    auto& is = initial_state.value();
    STD_TORCH_CHECK(is.is_cuda() && is.is_contiguous(),
                    "initial_state must be contiguous CUDA tensor");
    STD_TORCH_CHECK(is.scalar_type() == ScalarType::BFloat16 ||
                        is.scalar_type() == ScalarType::Float,
                    "initial_state must be bfloat16 or float32");
    if (is.scalar_type() == ScalarType::Float) state_fp32 = true;
  }
  if (has_state_out) {
    auto& fs = final_state.value();
    STD_TORCH_CHECK(fs.is_cuda() && fs.is_contiguous(),
                    "final_state must be contiguous CUDA tensor");
    STD_TORCH_CHECK(fs.scalar_type() == ScalarType::BFloat16 ||
                        fs.scalar_type() == ScalarType::Float,
                    "final_state must be bfloat16 or float32");
    if (fs.scalar_type() == ScalarType::Float) state_fp32 = true;
  }
  if (has_checkpoint) {
    auto& cs = checkpoint_state.value();
    auto& co = checkpoint_offsets.value();
    STD_TORCH_CHECK(cs.is_cuda() && cs.is_contiguous(),
                    "checkpoint_state must be a contiguous CUDA tensor");
    STD_TORCH_CHECK(cs.scalar_type() == ScalarType::Float ||
                        cs.scalar_type() == ScalarType::BFloat16,
                    "checkpoint_state must be bfloat16 or float32");
    STD_TORCH_CHECK(
        (!has_state_in || cs.scalar_type() == initial_state->scalar_type()) &&
            (!has_state_out || cs.scalar_type() == final_state->scalar_type()),
        "checkpoint_state must have the same dtype as initial_state and "
        "final_state");
    // Checkpoints also select the state dtype when both other states are
    // absent.
    state_fp32 = cs.scalar_type() == ScalarType::Float;
    STD_TORCH_CHECK(co.is_cuda() && co.is_contiguous(),
                    "checkpoint_offsets must be a contiguous CUDA tensor");
    STD_TORCH_CHECK(co.scalar_type() == ScalarType::Int ||
                        co.scalar_type() == ScalarType::Long,
                    "checkpoint_offsets must be int32 or int64");
  }
  // If both present, dtypes must match
  if (has_state_in && has_state_out) {
    STD_TORCH_CHECK(initial_state->scalar_type() == final_state->scalar_type(),
                    "initial_state and final_state must have the same dtype");
  }

  STD_TORCH_CHECK(A_log.is_cuda() && A_log.is_contiguous(),
                  "A_log must be contiguous CUDA tensor");
  STD_TORCH_CHECK(A_log.scalar_type() == ScalarType::Float,
                  "A_log must be float32");
  STD_TORCH_CHECK(dt_bias.is_cuda() && dt_bias.is_contiguous(),
                  "dt_bias must be contiguous CUDA tensor");
  STD_TORCH_CHECK(dt_bias.scalar_type() == ScalarType::Float,
                  "dt_bias must be float32");

  // Accept 4D input [B, T, H, D]
  STD_TORCH_CHECK(q.dim() == 4, "q must be [B, T, H, D]");
  STD_TORCH_CHECK(k.dim() == 4, "k must be [B, T, H, D]");
  STD_TORCH_CHECK(v.dim() == 4, "v must be [B, T, H, D]");
  STD_TORCH_CHECK(g.dim() == 4, "g must be [B, T, H, D]");
  STD_TORCH_CHECK(beta.dim() == 3, "beta must be [B, T, H]");
  STD_TORCH_CHECK(out.dim() == 4, "out must be [B, T, H, D]");

  int64_t B = q.size(0);
  int64_t T_seq = q.size(1);
  int64_t H = q.size(2);
  int64_t D = q.size(3);
  int64_t T_total = B * T_seq;

  STD_TORCH_CHECK(
      k.size(0) == B && k.size(1) == T_seq && k.size(2) == H && k.size(3) == D,
      "k must match q shape");
  STD_TORCH_CHECK(
      v.size(0) == B && v.size(1) == T_seq && v.size(2) == H && v.size(3) == D,
      "v must match q shape");
  STD_TORCH_CHECK(
      g.size(0) == B && g.size(1) == T_seq && g.size(2) == H && g.size(3) == D,
      "g must match q shape");
  STD_TORCH_CHECK(out.size(0) == B && out.size(1) == T_seq &&
                      out.size(2) == H && out.size(3) == D,
                  "out must match q shape");
  STD_TORCH_CHECK(
      beta.size(0) == B && beta.size(1) == T_seq && beta.size(2) == H,
      "beta must be [B, T, H] matching q");

  STD_TORCH_CHECK(A_log.dim() == 1 && A_log.size(0) == H, "A_log must be [H]");
  STD_TORCH_CHECK(
      dt_bias.dim() == 2 && dt_bias.size(0) == H && dt_bias.size(1) == D,
      "dt_bias must be [H, D]");

  STD_TORCH_CHECK(D == 128, "currently only supports D == 128");

  auto q_ptr = reinterpret_cast<cutlass::bfloat16_t const*>(q.const_data_ptr());
  auto k_ptr = reinterpret_cast<cutlass::bfloat16_t const*>(k.const_data_ptr());
  auto v_ptr = reinterpret_cast<cutlass::bfloat16_t const*>(v.const_data_ptr());
  auto g_ptr = reinterpret_cast<cutlass::bfloat16_t const*>(g.const_data_ptr());
  float scale_f = scale;
  auto out_ptr = reinterpret_cast<cutlass::bfloat16_t*>(out.mutable_data_ptr());
  auto A_log_ptr = reinterpret_cast<float const*>(A_log.const_data_ptr());
  auto dt_bias_ptr = reinterpret_cast<float const*>(dt_bias.const_data_ptr());
  float gate_scale = float(lower_bound * 1.4426950408889634);

  // Transpose beta: [B, T, H] -> [H, B*T] in one materialization.
  Tensor beta_bht = torch::stable::transpose(beta, 1, 2);
  Tensor beta_hbt = torch::stable::transpose(beta_bht, 0, 1);
  Tensor beta_t = torch::stable::contiguous(beta_hbt);
  auto beta_t_ptr =
      reinterpret_cast<cutlass::bfloat16_t const*>(beta_t.const_data_ptr());

  auto workspace_ptr = workspace.mutable_data_ptr();

  void* stream_ptr = nullptr;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_current_cuda_stream(q.get_device_index(), &stream_ptr));
  cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

  constexpr int CHUNK = 16;

  // Get state pointers (nullptr if not present)
  void const* initial_state_raw =
      has_state_in ? initial_state->const_data_ptr() : nullptr;
  void* final_state_raw =
      has_state_out ? final_state->mutable_data_ptr() : nullptr;
  void* checkpoint_state_raw =
      has_checkpoint ? checkpoint_state->mutable_data_ptr() : nullptr;

  // Determine cu_seqlens and N
  bool is_varlen = cu_seqlens.has_value();
  int64_t N_val;
  void const* cu_seqlens_dev = nullptr;
  bool cu_seqlens_is_int32 = false;

  if (is_varlen) {
    STD_TORCH_CHECK(B == 1, "B must be 1 when cu_seqlens is provided");
    auto& cu_seqlens_t = cu_seqlens.value();
    STD_TORCH_CHECK(cu_seqlens_t.is_cuda(), "cu_seqlens must be on CUDA");
    STD_TORCH_CHECK(cu_seqlens_t.scalar_type() == ScalarType::Int ||
                        cu_seqlens_t.scalar_type() == ScalarType::Long,
                    "cu_seqlens must be int32 or int64");
    STD_TORCH_CHECK(cu_seqlens_t.is_contiguous(),
                    "cu_seqlens must be contiguous");
    STD_TORCH_CHECK(cu_seqlens_t.dim() == 1, "cu_seqlens must be 1D");
    N_val = cu_seqlens_t.numel() - 1;
    STD_TORCH_CHECK(N_val > 0, "cu_seqlens must have at least 2 elements");
    cu_seqlens_is_int32 = cu_seqlens_t.scalar_type() == ScalarType::Int;
    cu_seqlens_dev = cu_seqlens_t.const_data_ptr();
  } else {
    N_val = B;
  }

  int num_sms = 0;
  cudaError_t attr_status = cudaDeviceGetAttribute(
      &num_sms, cudaDevAttrMultiProcessorCount, q.get_device_index());
  STD_TORCH_CHECK(attr_status == cudaSuccess,
                  "failed to query CUDA multiprocessor count: ",
                  cudaGetErrorString(attr_status));

  // Validate state shapes: always [N, H, D, D]
  if (has_state_in) {
    auto& is = initial_state.value();
    STD_TORCH_CHECK(is.dim() == 4, "initial_state must be [N, H, D, D]");
    STD_TORCH_CHECK(is.size(0) == N_val && is.size(1) == H && is.size(2) == D &&
                        is.size(3) == D,
                    "initial_state must be [N, H, D, D]");
  }
  if (has_state_out) {
    auto& fs = final_state.value();
    STD_TORCH_CHECK(fs.dim() == 4, "final_state must be [N, H, D, D]");
    STD_TORCH_CHECK(fs.size(0) == N_val && fs.size(1) == H && fs.size(2) == D &&
                        fs.size(3) == D,
                    "final_state must be [N, H, D, D]");
  }
  if (has_checkpoint) {
    auto& cs = checkpoint_state.value();
    auto& co = checkpoint_offsets.value();
    STD_TORCH_CHECK(cs.dim() == 4 && cs.size(0) == N_val && cs.size(1) == H &&
                        cs.size(2) == D && cs.size(3) == D,
                    "checkpoint_state must be [N, H, D, D]");
    STD_TORCH_CHECK(co.dim() == 1 && co.size(0) == N_val,
                    "checkpoint_offsets must be [N]");
    STD_TORCH_CHECK(
        co.scalar_type() ==
            (cu_seqlens_is_int32 ? ScalarType::Int : ScalarType::Long),
        "checkpoint_offsets and cu_seqlens must have matching dtypes");
  }

  int total_tiles;
  if (is_varlen) {
    total_tiles =
        int((T_total + CHUNK - 1) / CHUNK + N_val);  // upper bound for varlen
  } else {
    total_tiles =
        int(N_val * ((T_seq + CHUNK - 1) / CHUNK));  // exact for batched
  }
  if (mode & 512) {
    STD_TORCH_CHECK(is_varlen && workspace.scalar_type() == ScalarType::Byte &&
                        workspace.numel() >= get_workspace_size(T_total, H, N_val),
                    "k3_flashkda_split: the overlapped K1 / K2 (mode bit 9) "
                    "needs varlen and k3_flashkda_split.get_workspace_size "
                    "bytes of workspace");
  }

  auto dispatch = [&](auto typed_cu_seqlens) {
    using SeqlenPtr = decltype(typed_cu_seqlens);
    SeqlenPtr typed_checkpoint_offsets =
        has_checkpoint
            ? static_cast<SeqlenPtr>(checkpoint_offsets->const_data_ptr())
            : nullptr;
#define LAUNCH(HI, HO, FP32, CKPT, VL)                                       \
  launch_fwd<128, HI, HO, FP32, CKPT, VL>(                                   \
      q_ptr, k_ptr, v_ptr, g_ptr, beta_t_ptr, initial_state_raw, scale_f,    \
      final_state_raw, checkpoint_state_raw, typed_checkpoint_offsets,       \
      out_ptr, workspace_ptr, total_tiles, int(T_total), int(H), int(N_val), \
      typed_cu_seqlens, A_log_ptr, dt_bias_ptr, gate_scale, num_sms, stream, \
      int(mode))

#define DISPATCH_STATE(CKPT, VL)                              \
  if (!has_state_in && !has_state_out && state_fp32) {        \
    LAUNCH(false, false, true, CKPT, VL);                     \
  } else if (!has_state_in && !has_state_out) {               \
    LAUNCH(false, false, false, CKPT, VL);                    \
  } else if (has_state_in && has_state_out && state_fp32) {   \
    LAUNCH(true, true, true, CKPT, VL);                       \
  } else if (has_state_in && has_state_out && !state_fp32) {  \
    LAUNCH(true, true, false, CKPT, VL);                      \
  } else if (!has_state_in && has_state_out && state_fp32) {  \
    LAUNCH(false, true, true, CKPT, VL);                      \
  } else if (!has_state_in && has_state_out && !state_fp32) { \
    LAUNCH(false, true, false, CKPT, VL);                     \
  } else if (has_state_in && !has_state_out && state_fp32) {  \
    LAUNCH(true, false, true, CKPT, VL);                      \
  } else {                                                    \
    LAUNCH(true, false, false, CKPT, VL);                     \
  }

    if (has_checkpoint) {
      DISPATCH_STATE(true, true);
    } else {
      DISPATCH_STATE(false, true);
    }

#undef DISPATCH_STATE
#undef LAUNCH
  };

  STD_TORCH_CHECK(is_varlen && cu_seqlens_is_int32,
                  "k3_flashkda_split: varlen int32 cu_seqlens only");
  dispatch(static_cast<int32_t const*>(cu_seqlens_dev));
}

// Kimi-K3: gated o_norm + MXFP8 o_proj input (smxx/o_norm_mx.cu).
void launch_o_norm_mx(const void* x, int64_t tokens, int heads,
                      NormMxArgs const& a, int num_sms, cudaStream_t stream);
void launch_o_norm_mx_ldg(const void* x, int64_t tokens, int heads,
                          NormMxArgs const& a, int num_sms, cudaStream_t stream);

static void o_norm_mxfp8_impl(const Tensor& x, const Tensor& g, const Tensor& weight,
                              double eps, const Tensor& q, const Tensor& sf, bool staged) {
  STD_TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() >= 2 &&
                      x.size(x.dim() - 1) == 128 &&
                      x.scalar_type() == ScalarType::BFloat16,
                  "o_norm_mxfp8: x must be contiguous bf16 (..., H, 128)");
  const int64_t H = x.size(x.dim() - 2);
  const int64_t P = H * 128;
  const int64_t T = x.numel() / P;
  STD_TORCH_CHECK(g.dim() == 2 && g.size(0) >= T && g.size(1) >= P &&
                      g.stride(1) == 1 && g.stride(0) % 8 == 0 &&
                      reinterpret_cast<uintptr_t>(g.const_data_ptr()) % 16 == 0 &&
                      g.scalar_type() == ScalarType::BFloat16,
                  "o_norm_mxfp8: g must be bf16 (T, >= P) unit-stride rows");
  STD_TORCH_CHECK(weight.numel() == 128 && weight.is_contiguous() &&
                      weight.scalar_type() == ScalarType::Float,
                  "o_norm_mxfp8: weight must be FP32 [128]");
  STD_TORCH_CHECK(q.dim() == 2 && q.size(0) >= T && q.size(1) == P &&
                      q.stride(1) == 1 && q.stride(0) % 16 == 0 &&
                      q.scalar_type() == ScalarType::Float8_e4m3fn,
                  "o_norm_mxfp8: q must be e4m3 (T, P) unit-stride rows");
  STD_TORCH_CHECK(sf.is_contiguous() && sf.scalar_type() == ScalarType::Byte &&
                      sf.numel() >= ((T + 127) / 128) * 128 * (P / 32),
                  "o_norm_mxfp8: sf must be uint8 with the padded scale tile");
  NormMxArgs a{};
  a.g = g.const_data_ptr();
  a.g_ld = g.stride(0);
  a.w = reinterpret_cast<const float*>(weight.const_data_ptr());
  a.eps = float(eps);
  a.q = q.mutable_data_ptr();
  a.q_ld = q.stride(0);
  a.sf = sf.mutable_data_ptr();
  a.sf_cols = int(P / 32);
  int num_sms = 0;
  cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount,
                         x.get_device_index());
  void* stream_ptr = nullptr;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_current_cuda_stream(x.get_device_index(), &stream_ptr));
  (staged ? launch_o_norm_mx : launch_o_norm_mx_ldg)(
      x.const_data_ptr(), T, int(H), a, num_sms, static_cast<cudaStream_t>(stream_ptr));
}

void o_norm_mxfp8(const Tensor& x, const Tensor& g, const Tensor& weight,
                  double eps, const Tensor& q, const Tensor& sf) {
  o_norm_mxfp8_impl(x, g, weight, eps, q, sf, true);
}

// The direct-load kernel: the parity reference of o_norm_mxfp8 in tests.
void o_norm_mxfp8_ref(const Tensor& x, const Tensor& g, const Tensor& weight,
                      double eps, const Tensor& q, const Tensor& sf) {
  o_norm_mxfp8_impl(x, g, weight, eps, q, sf, false);
}

void launch_k64_tc(void const* kdq, void const* kr, void const* inv, void const* mqk, float const* gt, void const* beta, void const* v,
                   float const* init_state, float* final_state, void* out,
                   int const* cu_seqlens, int T_total, int H, int N, int C_total,
                   int flags, long long* dbg, cudaStream_t stream);

void k64_rec(const Tensor& kdq, const Tensor& kr, const Tensor& inv,
             const Tensor& mqk, const Tensor& gt, const Tensor& beta, const Tensor& v,
             const Tensor& init_state, const Tensor& final_state, const Tensor& out,
             const Tensor& cu_seqlens, int64_t flags, const Tensor& dbg) {
  STD_TORCH_CHECK(out.dim() == 3 && out.size(2) == 128 && out.is_contiguous() &&
                      out.scalar_type() == ScalarType::BFloat16,
                  "k64_rec: out must be contiguous bf16 (T, H, 128)");
  const int64_t T = out.size(0), H = out.size(1);
  const int64_t N = cu_seqlens.numel() - 1;
  STD_TORCH_CHECK(gt.dim() == 3 && gt.size(0) == H && gt.size(2) == 128 &&
                      gt.scalar_type() == ScalarType::Float && gt.is_contiguous(),
                  "k64_rec: g_total must be fp32 (H, C, 128)");
  const int64_t C = gt.size(1);
  STD_TORCH_CHECK(kdq.is_contiguous() && kdq.numel() == H * C * 128 * 128 &&
                      kdq.scalar_type() == ScalarType::BFloat16,
                  "k64_rec: kd|qd image must be bf16 (H, C, 128 x 128)");
  STD_TORCH_CHECK(v.is_contiguous() && v.numel() == H * C * 64 * 128 &&
                      v.scalar_type() == ScalarType::BFloat16,
                  "k64_rec: v image must be bf16 (H, C, 64 x 128)");
  STD_TORCH_CHECK(kr.is_contiguous() && kr.numel() == H * C * 64 * 128 &&
                      kr.scalar_type() == ScalarType::BFloat16,
                  "k64_rec: kr image must be bf16 (H, C, 64 x 128)");
  for (const Tensor* t : {&inv, &mqk}) {
    STD_TORCH_CHECK(t->is_contiguous() && t->numel() == H * C * 64 * 64 &&
                        t->scalar_type() == ScalarType::BFloat16,
                    "k64_rec: INV / Mqk images must be bf16 (H, C, 64 x 64)");
  }
  STD_TORCH_CHECK(beta.is_contiguous() && beta.numel() == H * T &&
                      beta.scalar_type() == ScalarType::BFloat16,
                  "k64_rec: beta must be bf16 (H, T)");
  STD_TORCH_CHECK(init_state.numel() == N * H * 128 * 128 &&
                      final_state.numel() == N * H * 128 * 128 &&
                      init_state.scalar_type() == ScalarType::Float &&
                      final_state.scalar_type() == ScalarType::Float,
                  "k64_rec: states must be fp32 (N, H, 128, 128)");
  STD_TORCH_CHECK(out.numel() == T * H * 128 && out.scalar_type() == ScalarType::BFloat16,
                  "k64_rec: out must be bf16 (T, H, 128)");
  STD_TORCH_CHECK(cu_seqlens.scalar_type() == ScalarType::Int, "k64_rec: int32 cu_seqlens");
  void* stream_ptr = nullptr;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_current_cuda_stream(v.get_device_index(), &stream_ptr));
  launch_k64_tc(kdq.const_data_ptr(), kr.const_data_ptr(),
                inv.const_data_ptr(), mqk.const_data_ptr(),
                reinterpret_cast<const float*>(gt.const_data_ptr()), beta.const_data_ptr(),
                v.const_data_ptr(), reinterpret_cast<const float*>(init_state.const_data_ptr()),
                reinterpret_cast<float*>(final_state.mutable_data_ptr()), out.mutable_data_ptr(),
                reinterpret_cast<const int*>(cu_seqlens.const_data_ptr()), int(T), int(H),
                int(N), int(C), int(flags), reinterpret_cast<long long*>(dbg.mutable_data_ptr()),
                static_cast<cudaStream_t>(stream_ptr));
}

void launch_prep64(void const* q, void const* k, void const* g, void const* beta,
                   int64_t row_stride_qk, int64_t row_stride_g, int64_t row_stride_beta,
                   float const* a_log, float const* dt_bias, float lower_bound, float scale,
                   int const* cu_seqlens, int N, int H, int T_total, int C_total,
                   void* kdq, void* kr, void* inv, void* mqk, float* gt, void* beta_out,
                   cudaStream_t stream);

// q, k: (T, H, 128) bf16 with a row stride (post-conv); g: (T, H, 128) bf16 raw
// gate with a row stride; beta: (T, H) bf16 raw logits with a row stride.
void prep64(const Tensor& q, const Tensor& k, const Tensor& g, const Tensor& beta,
            const Tensor& a_log, const Tensor& dt_bias, double lower_bound, double scale,
            const Tensor& cu_seqlens, const Tensor& kdq, const Tensor& kr, const Tensor& inv,
            const Tensor& mqk, const Tensor& gt, const Tensor& beta_out) {
  STD_TORCH_CHECK(q.dim() == 3 && q.size(2) == 128 && q.stride(2) == 1 && q.stride(1) == 128 &&
                      k.stride(0) == q.stride(0) && k.stride(1) == 128 && k.stride(2) == 1 &&
                      q.scalar_type() == ScalarType::BFloat16 &&
                      k.scalar_type() == ScalarType::BFloat16,
                  "prep64: q / k must be bf16 (T, H, 128) with dense heads and one row stride");
  STD_TORCH_CHECK(g.dim() == 3 && g.stride(1) == 128 && g.stride(2) == 1 &&
                      g.scalar_type() == ScalarType::BFloat16 && beta.dim() == 2 &&
                      beta.stride(1) == 1 && beta.scalar_type() == ScalarType::BFloat16,
                  "prep64: g (T, H, 128) / beta (T, H) bf16 with dense heads");
  const int64_t T = q.size(0), H = q.size(1), N = cu_seqlens.numel() - 1;
  const int64_t C = gt.size(1);
  STD_TORCH_CHECK(gt.dim() == 3 && gt.size(0) == H && gt.is_contiguous() &&
                      kdq.numel() == H * C * 128 * 128 && kr.numel() == H * C * 64 * 128 &&
                      inv.numel() == H * C * 64 * 64 && mqk.numel() == H * C * 64 * 64 &&
                      beta_out.numel() == H * T,
                  "prep64: image shapes");
  void* stream_ptr = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(q.get_device_index(), &stream_ptr));
  launch_prep64(q.const_data_ptr(), k.const_data_ptr(), g.const_data_ptr(), beta.const_data_ptr(),
                q.stride(0), g.stride(0), beta.stride(0),
                reinterpret_cast<const float*>(a_log.const_data_ptr()),
                reinterpret_cast<const float*>(dt_bias.const_data_ptr()), float(lower_bound),
                float(scale), reinterpret_cast<const int*>(cu_seqlens.const_data_ptr()), int(N),
                int(H), int(T), int(C), kdq.mutable_data_ptr(), kr.mutable_data_ptr(),
                inv.mutable_data_ptr(), mqk.mutable_data_ptr(),
                reinterpret_cast<float*>(gt.mutable_data_ptr()), beta_out.mutable_data_ptr(),
                static_cast<cudaStream_t>(stream_ptr));
}

void launch_prep64_tc(void const* x, int64_t ldx, float const* conv_w, float* conv_state,
                      int64_t st_s0, int64_t st_s1, int64_t st_s2, int32_t const* cache_indices,
                      int64_t ci_s0, bool const* has_init, int null_block_id, void const* g,
                      int64_t ldg, void const* beta, int64_t ldb, float const* a_log,
                      float const* dt_bias, float lower_bound, float scale, int const* cu_seqlens,
                      int N, int H, int T, int C_total, void* kdq, void* kr, void* inv, void* mqk,
                      float* gt, void* v_img, void* beta_out, long long* dbg, cudaStream_t stream);

// Chunk-64 prepare on the raw in_proj rows with the short conv fused.
// x: (T, 3 H 128) bf16 raw q|k|v rows; conv_w (3 H 128, 4) FP32; conv_state
// (lines, 3 H 128, >= 3) FP32 (updated in place for each sequence);
// g: (T, H, 128) bf16 raw gate; beta: (T, H) bf16 raw logits (row strides).
void prep64_tc(const Tensor& x, const Tensor& conv_w, const Tensor& conv_state,
               const Tensor& cache_indices, const Tensor& has_init, int64_t null_block_id,
               const Tensor& g, const Tensor& beta, const Tensor& a_log, const Tensor& dt_bias,
               double lower_bound, double scale, const Tensor& cu_seqlens, const Tensor& kdq,
               const Tensor& kr, const Tensor& inv, const Tensor& mqk, const Tensor& gt,
               const Tensor& v_img, const Tensor& beta_out, const Tensor& dbg) {
  const int64_t T = x.size(0), H = g.size(1), N = cu_seqlens.numel() - 1;
  STD_TORCH_CHECK(x.dim() == 2 && x.size(1) == 3 * H * 128 && x.stride(1) == 1 &&
                      x.stride(0) % 8 == 0 && x.scalar_type() == ScalarType::BFloat16,
                  "prep64_tc: x must be bf16 (T, 3 H 128) with 16 B aligned rows");
  STD_TORCH_CHECK(conv_w.scalar_type() == ScalarType::Float && conv_w.is_contiguous() &&
                      conv_w.size(0) == 3 * H * 128 && conv_w.size(1) == 4,
                  "prep64_tc: conv weight (3 H 128, 4) FP32");
  STD_TORCH_CHECK(conv_state.scalar_type() == ScalarType::Float && conv_state.dim() == 3 &&
                      conv_state.size(1) == 3 * H * 128 && conv_state.size(2) >= 3,
                  "prep64_tc: conv state (lines, 3 H 128, >= 3) FP32");
  STD_TORCH_CHECK(g.dim() == 3 && g.size(0) == T && g.stride(1) == 128 && g.stride(2) == 1 &&
                      g.scalar_type() == ScalarType::BFloat16 && beta.dim() == 2 &&
                      beta.stride(1) == 1 && beta.scalar_type() == ScalarType::BFloat16,
                  "prep64_tc: g (T, H, 128) / beta (T, H) bf16 with dense heads");
  const int64_t C = gt.size(1);
  STD_TORCH_CHECK(gt.dim() == 3 && gt.size(0) == H && gt.is_contiguous() &&
                      kdq.numel() == H * C * 128 * 128 && kr.numel() == H * C * 64 * 128 &&
                      inv.numel() == H * C * 64 * 64 && mqk.numel() == H * C * 64 * 64 &&
                      v_img.numel() == H * C * 64 * 128 && beta_out.numel() == H * T,
                  "prep64_tc: image shapes");
  void* stream_ptr = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(x.get_device_index(), &stream_ptr));
  launch_prep64_tc(
      x.const_data_ptr(), x.stride(0), reinterpret_cast<const float*>(conv_w.const_data_ptr()),
      reinterpret_cast<float*>(conv_state.mutable_data_ptr()), conv_state.stride(0),
      conv_state.stride(1), conv_state.stride(2),
      reinterpret_cast<const int32_t*>(cache_indices.const_data_ptr()), cache_indices.stride(0),
      reinterpret_cast<const bool*>(has_init.const_data_ptr()), int(null_block_id),
      g.const_data_ptr(), g.stride(0), beta.const_data_ptr(), beta.stride(0),
      reinterpret_cast<const float*>(a_log.const_data_ptr()),
      reinterpret_cast<const float*>(dt_bias.const_data_ptr()), float(lower_bound), float(scale),
      reinterpret_cast<const int*>(cu_seqlens.const_data_ptr()), int(N), int(H), int(T), int(C),
      kdq.mutable_data_ptr(), kr.mutable_data_ptr(), inv.mutable_data_ptr(),
      mqk.mutable_data_ptr(), reinterpret_cast<float*>(gt.mutable_data_ptr()),
      v_img.mutable_data_ptr(), beta_out.mutable_data_ptr(),
      dbg.numel() >= 2 * C * 16 ? reinterpret_cast<long long*>(dbg.mutable_data_ptr()) : nullptr,
      static_cast<cudaStream_t>(stream_ptr));
}
