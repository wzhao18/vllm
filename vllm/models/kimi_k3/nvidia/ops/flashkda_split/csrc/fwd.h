// Kimi-K3: FlashKDA csrc/fwd.h @ vllm-project/FlashKDA 17a037d98da5 (MIT, see
// LICENSE.FlashKDA), with a `mode` argument (bit 0: K1 prepare, bit 1: K2
// recurrence, bit 2: exact fp32 state I/O for segment partials).
#pragma once
#include <cuda_runtime.h>

#include <cutlass/bfloat16.h>

template <int D, bool HasStateIn = true, bool HasStateOut = true,
          bool StateFP32 = false, bool HasCheckpoint = false,
          bool IsVarlen = true, typename SeqlenT = int64_t>
void launch_fwd(
    cutlass::bfloat16_t const* q_ptr, cutlass::bfloat16_t const* k_ptr,
    cutlass::bfloat16_t const* v_ptr, cutlass::bfloat16_t const* g_bf16_ptr,
    cutlass::bfloat16_t const* beta_ptr, void const* initial_state_ptr,
    float scale, void* final_state_ptr, void* checkpoint_state_ptr,
    SeqlenT const* checkpoint_offsets_ptr, cutlass::bfloat16_t* out_ptr,
    void* workspace_ptr, int total_tiles, int T_total, int H, int N,
    SeqlenT const* cu_seqlens_ptr, float const* A_log_ptr,
    float const* dt_bias_ptr, float gate_scale, int num_sms,
    cudaStream_t stream, int mode);
