# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark an FP8 block-scaled Kimi-K3 shared expert."""

from __future__ import annotations

import argparse
import statistics

import torch

from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    deepgemm_post_process_fp8_weight_block,
    per_token_group_quant_fp8_packed_for_deepgemm,
    situ_mul_quant_fp8_packed_triton,
)
from vllm.utils.deep_gemm import fp8_gemm_nt, per_block_cast_to_fp8


def event_times(fn, warmup: int, iterations: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    return [start.elapsed_time(end) for start, end in zip(starts, ends, strict=True)]


def quantize_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    quantized, scale = per_block_cast_to_fp8(
        weight, block_size=[128, 128], use_ue8m0=True
    )
    return deepgemm_post_process_fp8_weight_block(
        quantized,
        scale,
        quant_block_shape=(128, 128),
        use_e8m0=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[16, 20, 32, 64, 160, 256]
    )
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--intermediate", type=int, default=12288)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    torch.manual_seed(0)
    gate_up_weight = torch.randn(
        2 * args.intermediate,
        args.hidden,
        dtype=torch.bfloat16,
        device="cuda",
    ) / 64
    down_weight = torch.randn(
        args.hidden,
        args.intermediate,
        dtype=torch.bfloat16,
        device="cuda",
    ) / 64
    gate_up_q, gate_up_scale = quantize_weight(gate_up_weight)
    down_q, down_scale = quantize_weight(down_weight)

    for tokens in args.tokens:
        hidden_states = torch.randn(
            tokens, args.hidden, dtype=torch.bfloat16, device="cuda"
        )
        gate_up_output = torch.empty(
            tokens, 2 * args.intermediate, dtype=torch.bfloat16, device="cuda"
        )
        output = torch.empty(
            tokens, args.hidden, dtype=torch.bfloat16, device="cuda"
        )

        def fp8_shared() -> torch.Tensor:
            hidden_q, hidden_scale = per_token_group_quant_fp8_packed_for_deepgemm(
                hidden_states, group_size=128, use_ue8m0=True
            )
            fp8_gemm_nt(
                (hidden_q, hidden_scale),
                (gate_up_q, gate_up_scale),
                gate_up_output,
                is_deep_gemm_e8m0_used=True,
            )
            activated_q, activated_scale = situ_mul_quant_fp8_packed_triton(
                gate_up_output,
                group_size=128,
                beta=4.0,
                linear_beta=25.0,
            )
            fp8_gemm_nt(
                (activated_q, activated_scale),
                (down_q, down_scale),
                output,
                is_deep_gemm_e8m0_used=True,
            )
            return output

        def dense_shared() -> torch.Tensor:
            gate_up = torch.nn.functional.linear(hidden_states, gate_up_weight)
            gate, up = gate_up.chunk(2, dim=-1)
            gate = 4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)
            up = 25.0 * torch.tanh(up / 25.0)
            return torch.nn.functional.linear(gate * up, down_weight)

        actual = fp8_shared().clone()
        expected = dense_shared()
        relative_l2 = (
            (actual.float() - expected.float()).norm() / expected.float().norm()
        ).item()
        cosine = torch.nn.functional.cosine_similarity(
            actual.float().flatten(), expected.float().flatten(), dim=0
        ).item()
        fp8_times = event_times(fp8_shared, args.warmup, args.iterations)
        dense_times = event_times(dense_shared, args.warmup, args.iterations)
        fp8_p50 = statistics.median(fp8_times)
        dense_p50 = statistics.median(dense_times)
        print(
            f"tokens={tokens} fp8_p50_ms={fp8_p50:.6f} "
            f"dense_p50_ms={dense_p50:.6f} speedup={dense_p50 / fp8_p50:.3f} "
            f"relative_l2={relative_l2:.6f} cosine={cosine:.8f}"
        )


if __name__ == "__main__":
    main()
