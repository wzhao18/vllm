# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark Kimi-K3 speculative recurrent KDA decode.

This benchmark uses the Kimi-K3 TP8 shape: 12 heads, head dimension 128,
five tokens per sequence, and BF16 recurrent state. Each sample measures a
CUDA graph containing independent layer calls with CUPTI and a cold L2 cache.

Example:
    .venv/bin/python benchmarks/kernels/benchmark_kimi_k3_kda_recurrent.py \
        --batches 1 8 32 64 128 --output results.json
"""

import argparse
import functools
import json
import statistics
from collections.abc import Callable
from pathlib import Path

import torch
from flashinfer.testing import bench_gpu_time

from vllm.models.kimi_k3.nvidia.ops.third_party.kda import (
    fused_recurrent_kda,
    fused_recurrent_kda_fwd,
)
from vllm.triton_utils import triton

HEADS = 12
HEAD_DIM = 128
TOKENS_PER_SEQUENCE = 5
GATE_LOWER_BOUND = -5.0


class Inputs:
    def __init__(self, batch: int) -> None:
        total_tokens = batch * TOKENS_PER_SEQUENCE
        shape = (1, total_tokens, HEADS, HEAD_DIM)
        self.q, self.k, self.v, self.raw_g = (
            torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(4)
        )
        self.raw_beta = torch.randn(
            1, total_tokens, HEADS, device="cuda", dtype=torch.bfloat16
        )
        self.a_log = torch.randn(HEADS, device="cuda", dtype=torch.float32)
        self.dt_bias = torch.randn(HEADS * HEAD_DIM, device="cuda", dtype=torch.float32)
        self.state = (
            0.01
            * torch.randn(
                total_tokens + 1,
                HEADS,
                HEAD_DIM,
                HEAD_DIM,
                device="cuda",
                dtype=torch.float32,
            )
        ).to(torch.bfloat16)
        self.state_indices = torch.arange(
            1, total_tokens + 1, device="cuda", dtype=torch.int32
        ).view(batch, TOKENS_PER_SEQUENCE)
        self.cu_seqlens = torch.arange(
            0,
            total_tokens + 1,
            TOKENS_PER_SEQUENCE,
            device="cuda",
            dtype=torch.int32,
        )
        self.num_accepted_tokens = torch.ones(batch, device="cuda", dtype=torch.int32)
        self.out = torch.empty_like(self.q)


def run(inputs: Inputs) -> None:
    fused_recurrent_kda(
        q=inputs.q,
        k=inputs.k,
        v=inputs.v,
        raw_g=inputs.raw_g,
        raw_beta=inputs.raw_beta,
        A_log=inputs.a_log,
        dt_bias=inputs.dt_bias,
        lower_bound=GATE_LOWER_BOUND,
        initial_state=inputs.state,
        cu_seqlens=inputs.cu_seqlens,
        ssm_state_indices=inputs.state_indices,
        num_accepted_tokens=inputs.num_accepted_tokens,
        out=inputs.out,
        fuse_gate=True,
    )


def check_correctness(batch: int) -> None:
    torch.manual_seed(1234)
    expected = Inputs(batch)
    torch.manual_seed(1234)
    actual = Inputs(batch)
    fused_recurrent_kda_fwd(
        q=expected.q,
        k=expected.k,
        v=expected.v,
        g=expected.raw_g,
        beta=expected.raw_beta,
        scale=HEAD_DIM**-0.5,
        initial_state=expected.state,
        inplace_final_state=True,
        cu_seqlens=expected.cu_seqlens,
        ssm_state_indices=expected.state_indices,
        num_accepted_tokens=expected.num_accepted_tokens,
        use_qk_l2norm_in_kernel=True,
        A_log=expected.a_log,
        dt_bias=expected.dt_bias,
        lower_bound=GATE_LOWER_BOUND,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        out=expected.out,
    )
    run(actual)
    torch.accelerator.synchronize()
    torch.testing.assert_close(actual.out, expected.out, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(
        actual.state.float(), expected.state.float(), rtol=2e-3, atol=2e-3
    )
    del expected, actual
    torch.accelerator.empty_cache()


def bench_graph_layers(
    calls: list[Callable[[], None]], dry_runs: int, repeats: int
) -> float:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for call in calls[:3]:
            call()
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for call in calls:
            call()
    samples = bench_gpu_time(
        graph.replay,
        dry_run_iters=dry_runs,
        repeat_iters=repeats,
        enable_cupti=True,
        use_cuda_graph=False,
        cold_l2_cache=True,
    )
    elapsed_us = statistics.median(float(sample) for sample in samples) * 1e3
    return elapsed_us / len(calls)


def measure(batch: int, layers: int, dry_runs: int, repeats: int) -> float:
    inputs = [Inputs(batch) for _ in range(layers)]
    elapsed_us = bench_graph_layers(
        [functools.partial(run, item) for item in inputs], dry_runs, repeats
    )
    del inputs
    torch.accelerator.synchronize()
    torch.accelerator.empty_cache()
    return elapsed_us


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32, 64, 128])
    parser.add_argument("--layers", type=int, default=69)
    parser.add_argument("--dry-runs", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(1)
    props = torch.cuda.get_device_properties(0)
    results = {
        "metadata": {
            "device": props.name,
            "compute_capability": [props.major, props.minor],
            "torch": torch.__version__,
            "triton": triton.__version__,
            "heads": HEADS,
            "head_dim": HEAD_DIM,
            "tokens_per_sequence": TOKENS_PER_SEQUENCE,
            "state_dtype": "bfloat16",
            "layers": args.layers,
            "dry_runs": args.dry_runs,
            "repeats": args.repeats,
            "timing": "median CUPTI CUDA-graph GPU span, cold L2",
        },
        "measurements": [],
    }
    for batch in args.batches:
        check_correctness(batch)
        elapsed_us = measure(batch, args.layers, args.dry_runs, args.repeats)
        results["measurements"].append({"batch": batch, "us_per_layer": elapsed_us})
        print(f"batch={batch:>3} {elapsed_us:8.3f} us")
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
