# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark Kimi-K3's fused dense MoE input and output projections."""

from __future__ import annotations

import argparse
import statistics

import torch


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


def p50(fn, warmup: int, iterations: int) -> float:
    return statistics.median(event_times(fn, warmup, iterations))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[16, 20, 32, 64, 160, 256]
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    hidden = 7168
    routed = 3584
    shared = 6144
    torch.manual_seed(0)
    routed_down = torch.randn(
        routed, hidden, dtype=torch.bfloat16, device="cuda"
    ) / 64
    shared_gate_up = torch.randn(
        2 * shared, hidden, dtype=torch.bfloat16, device="cuda"
    ) / 64
    fused_input_weight = torch.cat((routed_down, shared_gate_up), dim=0)
    routed_up = torch.randn(
        hidden, routed, dtype=torch.bfloat16, device="cuda"
    ) / 64
    shared_down = torch.randn(
        hidden, shared, dtype=torch.bfloat16, device="cuda"
    ) / 64
    fused_output_weight = torch.cat((shared_down, routed_up), dim=1)

    for tokens in args.tokens:
        hidden_states = torch.randn(
            tokens, hidden, dtype=torch.bfloat16, device="cuda"
        )
        routed_states = torch.randn(
            tokens, routed, dtype=torch.bfloat16, device="cuda"
        )
        shared_states = torch.randn(
            tokens, shared, dtype=torch.bfloat16, device="cuda"
        )

        def separate_input():
            return (
                torch.nn.functional.linear(hidden_states, routed_down),
                torch.nn.functional.linear(hidden_states, shared_gate_up),
            )

        def fused_input():
            return torch.nn.functional.linear(hidden_states, fused_input_weight)

        def separate_output():
            shared_output = torch.nn.functional.linear(shared_states, shared_down)
            return shared_output.addmm_(routed_states, routed_up.t())

        def fused_output():
            states = torch.cat((shared_states, routed_states), dim=-1)
            return torch.nn.functional.linear(states, fused_output_weight)

        separate_routed, separate_shared = separate_input()
        fused_routed, fused_shared = fused_input().split((routed, 2 * shared), dim=-1)
        separate_tail = separate_output()
        fused_tail = fused_output()
        input_routed_diff = (separate_routed - fused_routed).abs().max().item()
        input_shared_diff = (separate_shared - fused_shared).abs().max().item()
        output_diff = (separate_tail - fused_tail).abs().max().item()

        input_separate = p50(separate_input, args.warmup, args.iterations)
        input_fused = p50(fused_input, args.warmup, args.iterations)
        output_separate = p50(separate_output, args.warmup, args.iterations)
        output_fused = p50(fused_output, args.warmup, args.iterations)
        print(
            f"tokens={tokens} input_separate_ms={input_separate:.6f} "
            f"input_fused_ms={input_fused:.6f} "
            f"input_speedup={input_separate / input_fused:.3f} "
            f"output_separate_ms={output_separate:.6f} "
            f"output_fused_ms={output_fused:.6f} "
            f"output_speedup={output_separate / output_fused:.3f} "
            f"input_routed_max_diff={input_routed_diff:.6f} "
            f"input_shared_max_diff={input_shared_diff:.6f} "
            f"output_max_diff={output_diff:.6f} "
            "total_saved_ms="
            f"{input_separate + output_separate - input_fused - output_fused:.6f}"
        )


if __name__ == "__main__":
    main()
