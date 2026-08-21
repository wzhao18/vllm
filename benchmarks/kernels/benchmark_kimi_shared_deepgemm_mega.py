# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import os

import torch
import torch.distributed as dist

from vllm.utils.deep_gemm import _import_deep_gemm


def measure(function, repetitions: int) -> float:
    for _ in range(10):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repetitions):
        function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repetitions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=[32, 160, 256])
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--situ", action="store_true")
    parser.add_argument("--native-shared", action="store_true")
    parser.add_argument("--shared-partitions", type=int, default=2)
    parser.add_argument("--compare-direct-shared-input", action="store_true")
    parser.add_argument("--num-sms", type=int)
    args = parser.parse_args()
    if args.compare_direct_shared_input and not args.native_shared:
        parser.error("--compare-direct-shared-input requires --native-shared")

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29581")
    dist.init_process_group("nccl", rank=0, world_size=1)
    deep_gemm = _import_deep_gemm()

    hidden = 7168
    shared_intermediate = 6144
    virtual_experts = args.shared_partitions
    if shared_intermediate % virtual_experts:
        raise ValueError("shared-partitions must divide 6144")
    intermediate = shared_intermediate // virtual_experts
    experts = 1 if args.native_shared else virtual_experts
    topk = 1 if args.native_shared else virtual_experts
    capacity = max(args.tokens)
    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        dist.group.WORLD,
        experts,
        capacity,
        topk,
        hidden,
        intermediate,
        mma_type="bf16xbf16",
        activation="situ" if args.situ else "swiglu",
        num_shared_experts=virtual_experts if args.native_shared else 0,
    )
    grouped_l1 = torch.randn(
        virtual_experts,
        2 * intermediate,
        hidden,
        device="cuda",
        dtype=torch.bfloat16,
    ) / 64
    grouped_l2 = torch.randn(
        virtual_experts,
        hidden,
        intermediate,
        device="cuda",
        dtype=torch.bfloat16,
    ) / 64
    gate = torch.cat(tuple(grouped_l1[:, :intermediate]), dim=0)
    up = torch.cat(tuple(grouped_l1[:, intermediate:]), dim=0)
    combined_l1 = torch.cat((gate, up), dim=0)
    combined_l2 = torch.cat(tuple(grouped_l2), dim=1)
    if args.native_shared:
        l1 = torch.zeros(
            1,
            2 * intermediate,
            hidden,
            device="cuda",
            dtype=torch.bfloat16,
        )
        l2 = torch.zeros(
            1,
            hidden,
            intermediate,
            device="cuda",
            dtype=torch.bfloat16,
        )
    else:
        l1, l2 = grouped_l1, grouped_l2
    l1, l2 = deep_gemm.transform_weights_for_mega_moe(
        l1,
        l2,
        activation="situ" if args.situ else "swiglu",
    )
    shared_l1 = shared_l2 = None
    if args.native_shared:
        shared_l1, shared_l2 = deep_gemm.transform_weights_for_mega_moe(
            combined_l1,
            combined_l2,
            activation="situ" if args.situ else "swiglu",
        )

    for tokens in args.tokens:
        hidden_states = torch.randn(
            tokens, hidden, device="cuda", dtype=torch.bfloat16
        )
        if args.native_shared:
            topk_ids = torch.full((tokens, 1), -1, device="cuda", dtype=torch.int64)
            topk_weights = torch.zeros(tokens, 1, device="cuda")
        else:
            topk_ids = torch.arange(experts, device="cuda").expand(tokens, experts)
            topk_weights = torch.ones(tokens, experts, device="cuda")
        output = torch.empty_like(hidden_states)
        buffer.topk_idx[:tokens].copy_(topk_ids)
        buffer.topk_weights[:tokens].copy_(topk_weights)

        def run_buffered(
            tokens=tokens,
            hidden_states=hidden_states,
            output=output,
        ) -> None:
            buffer.x[:tokens].copy_(hidden_states)
            deep_gemm.bf16_mega_moe(
                output,
                l1,
                l2,
                buffer,
                shared_l1_weights=shared_l1,
                shared_l2_weights=shared_l2,
                activation="situ" if args.situ else "swiglu",
                situ_beta=4.0 if args.situ else None,
                situ_linear_beta=25.0 if args.situ else None,
                fast_math=True,
                num_sms=args.num_sms,
            )

        buffered_elapsed = measure(run_buffered, args.repetitions)
        direct_elapsed = None
        if args.compare_direct_shared_input:

            def run_direct(
                hidden_states=hidden_states,
                output=output,
            ) -> None:
                deep_gemm.bf16_mega_moe(
                    output,
                    l1,
                    l2,
                    buffer,
                    shared_l1_weights=shared_l1,
                    shared_l2_weights=shared_l2,
                    shared_l1_acts=hidden_states,
                    activation="situ" if args.situ else "swiglu",
                    situ_beta=4.0 if args.situ else None,
                    situ_linear_beta=25.0 if args.situ else None,
                    fast_math=True,
                    num_sms=args.num_sms,
                )

            direct_elapsed = measure(run_direct, args.repetitions)

        def run_dense(hidden_states=hidden_states) -> torch.Tensor:
            gate_up = torch.nn.functional.linear(hidden_states, combined_l1)
            dense_gate, dense_up = gate_up.chunk(2, dim=-1)
            if args.situ:
                dense_gate = (
                    4.0
                    * torch.tanh(dense_gate / 4.0)
                    * torch.sigmoid(dense_gate)
                )
                dense_up = 25.0 * torch.tanh(dense_up / 25.0)
                activated = dense_gate * dense_up
            else:
                activated = torch.nn.functional.silu(dense_gate) * dense_up
            return torch.nn.functional.linear(activated, combined_l2)

        dense_elapsed = measure(run_dense, args.repetitions)
        if args.compare_direct_shared_input:
            run_direct()
        else:
            run_buffered()
        expected = run_dense()
        relative_l2 = (
            (output.float() - expected.float()).norm() / expected.float().norm()
        ).item()
        cosine = torch.nn.functional.cosine_similarity(
            output.float().flatten(), expected.float().flatten(), dim=0
        ).item()
        norm_ratio = (output.float().norm() / expected.float().norm()).item()
        direct_metrics = ""
        if direct_elapsed is not None:
            direct_metrics = (
                f"direct_shared_input={direct_elapsed:.6f} ms "
                f"direct_speedup={buffered_elapsed / direct_elapsed:.3f}x "
            )
        print(
            f"native_shared={args.native_shared} tokens={tokens} "
            f"deepgemm_bf16_mega={buffered_elapsed:.6f} ms "
            f"{direct_metrics}"
            f"dense={dense_elapsed:.6f} ms "
            f"speedup={dense_elapsed / (direct_elapsed or buffered_elapsed):.3f}x "
            f"relative_l2={relative_l2:.6f} cosine={cosine:.6f} "
            f"norm_ratio={norm_ratio:.6f}"
        )

    buffer.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
