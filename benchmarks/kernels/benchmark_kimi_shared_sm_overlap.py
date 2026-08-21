# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import statistics

import torch

from vllm.models.kimi_k3.nvidia.model import _KimiDeepGemmSharedSession


def measure(function, repetitions: int) -> list[float]:
    for _ in range(10):
        function()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        function()
        end.record()
    torch.cuda.synchronize()
    return [start.elapsed_time(end) for start, end in zip(starts, ends, strict=True)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=20)
    parser.add_argument("--shared-sms", type=int, default=28)
    parser.add_argument("--routed-sms", type=int, default=120)
    parser.add_argument("--routed-repeats", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    hidden = 7168
    expert_intermediate = 3072
    shared_experts = 2
    x = torch.randn(args.tokens, hidden, dtype=torch.bfloat16, device="cuda")
    gate_up = torch.randn(
        2 * expert_intermediate * shared_experts,
        hidden,
        dtype=torch.bfloat16,
        device="cuda",
    )
    down = torch.randn(
        hidden,
        expert_intermediate * shared_experts,
        dtype=torch.bfloat16,
        device="cuda",
    )
    shared = _KimiDeepGemmSharedSession(
        num_tokens=args.tokens,
        hidden_size=hidden,
        expert_intermediate_size=expert_intermediate,
        num_shared_experts=shared_experts,
        num_sms=args.shared_sms,
    )
    routed = _KimiDeepGemmSharedSession(
        num_tokens=args.tokens,
        hidden_size=hidden,
        expert_intermediate_size=expert_intermediate,
        num_shared_experts=shared_experts,
        num_sms=args.routed_sms,
    )
    stream = torch.cuda.Stream()
    start_event = torch.cuda.Event()
    done_event = torch.cuda.Event()

    def run_shared() -> None:
        shared.run(
            x,
            gate_up,
            down,
            situ_beta=4.0,
            situ_linear_beta=25.0,
        )

    def run_routed() -> None:
        for _ in range(args.routed_repeats):
            routed.run(
                x,
                gate_up,
                down,
                situ_beta=4.0,
                situ_linear_beta=25.0,
            )

    def sequential() -> None:
        run_shared()
        run_routed()

    def shared_first() -> None:
        start_event.record()
        with torch.cuda.stream(stream):
            start_event.wait()
            run_shared()
            done_event.record()
        run_routed()
        done_event.wait()

    def routed_first() -> None:
        start_event.record()
        run_routed()
        with torch.cuda.stream(stream):
            start_event.wait()
            run_shared()
            done_event.record()
        done_event.wait()

    sequential_times = measure(sequential, args.iterations)
    shared_first_times = measure(shared_first, args.iterations)
    routed_first_times = measure(routed_first, args.iterations)
    sequential_p50 = statistics.median(sequential_times)
    shared_first_p50 = statistics.median(shared_first_times)
    routed_first_p50 = statistics.median(routed_first_times)
    print(
        f"tokens={args.tokens} shared_sms={args.shared_sms} "
        f"routed_sms={args.routed_sms} routed_repeats={args.routed_repeats} "
        f"sequential_p50_ms={sequential_p50:.6f} "
        f"shared_first_p50_ms={shared_first_p50:.6f} "
        f"shared_first_speedup={sequential_p50 / shared_first_p50:.3f}x "
        f"routed_first_p50_ms={routed_first_p50:.6f} "
        f"routed_first_speedup={sequential_p50 / routed_first_p50:.3f}x"
    )


if __name__ == "__main__":
    main()
