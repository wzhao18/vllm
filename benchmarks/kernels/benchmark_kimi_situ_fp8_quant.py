# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse

import torch

from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8_packed_for_deepgemm,
    situ_mul_quant_fp8_packed_triton,
)


def measure(fn, warmup: int, repetitions: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repetitions):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repetitions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 32, 160, 256])
    parser.add_argument("--intermediate-size", type=int, default=6144)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=100)
    args = parser.parse_args()

    torch.manual_seed(42)
    for num_tokens in args.tokens:
        gate_up = torch.randn(
            (num_tokens, 2 * args.intermediate_size),
            dtype=torch.bfloat16,
            device="cuda",
        )

        def separate(gate_up=gate_up, num_tokens=num_tokens):
            activated = torch.empty(
                (num_tokens, args.intermediate_size),
                dtype=gate_up.dtype,
                device=gate_up.device,
            )
            torch.ops._C.situ_and_mul(activated, gate_up, 4.0, 25.0)
            return per_token_group_quant_fp8_packed_for_deepgemm(
                activated,
                group_size=128,
                use_ue8m0=True,
            )

        def fused(gate_up=gate_up):
            return situ_mul_quant_fp8_packed_triton(
                gate_up,
                group_size=128,
                beta=4.0,
                linear_beta=25.0,
            )

        expected_q, expected_scale = separate()
        actual_q, actual_scale = fused()
        max_diff = (actual_q.float() - expected_q.float()).abs().max().item()
        scale_match = torch.equal(actual_scale, expected_scale)
        separate_ms = measure(separate, args.warmup, args.repetitions)
        fused_ms = measure(fused, args.warmup, args.repetitions)
        print(
            f"tokens={num_tokens:4d} separate={separate_ms:.6f} ms "
            f"fused={fused_ms:.6f} ms speedup={separate_ms / fused_ms:.3f}x "
            f"max_q_diff={max_diff:.6f} scale_match={scale_match}"
        )


if __name__ == "__main__":
    main()
