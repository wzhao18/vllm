# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbenchmark Kimi K3's KDA prefill backends on NVIDIA Blackwell.

The default cases use Kimi K3's TP8 shape of 12 heads by 128 dimensions.
FlashInfer is measured through vLLM's adapter, and Triton is measured through
the fallback used by the model. CUPTI reports the GPU span from the first to
last kernel in each backend call, with an L2 flush before every iteration.

Example:
    FLASHINFER_WORKSPACE_BASE=/tmp .venv/bin/python \
        benchmarks/kernels/benchmark_kimi_k3_kda_prefill.py
"""

import argparse
import json
import statistics
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch
from flashinfer.kda_prefill import RecurrentKDAPrefillWorkspace
from flashinfer.testing import bench_gpu_time
from flashinfer.utils import get_compute_capability

from vllm.models.kimi_k3.nvidia.kda import _flashinfer_kda_prefill
from vllm.models.kimi_k3.nvidia.ops.third_party.kda import (
    chunk_kda_with_fused_gate,
)

HEAD_DIM = 128
LOWER_BOUND = -5.0
SUPPORTED_CAPABILITIES = {(10, 0), (10, 3)}


@dataclass(frozen=True)
class Case:
    name: str
    seq_lens: tuple[int, ...]
    seed: int


CASES = (
    Case("fixed128", (128,), 27001),
    Case("fixed512", (512,), 27002),
    Case("fixed2048", (2048,), 27003),
    Case("fixed8192", (8192,), 27004),
    Case("mixed", (128, 256, 512, 1024, 2048, 4096), 27005),
    Case("uniform", (1024,) * 8, 27006),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=[case.name for case in CASES],
        default=[case.name for case in CASES],
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=("flashinfer", "triton"),
        default=("flashinfer", "triton"),
    )
    parser.add_argument(
        "--heads",
        type=int,
        default=12,
        help="KDA heads per rank; Kimi K3 has 96 heads globally.",
    )
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--benchmark-iters", type=int, default=50)
    parser.add_argument(
        "--warm-l2",
        action="store_true",
        help="Disable the default L2 flush between iterations.",
    )
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.heads <= 0:
        parser.error("--heads must be positive")
    if args.warmup_iters <= 0 or args.benchmark_iters <= 0:
        parser.error("iteration counts must be positive")
    return args


def require_cupti() -> str:
    try:
        cupti_version = version("cupti-python")
    except PackageNotFoundError as error:
        raise RuntimeError(
            "cupti-python >= 13 is required; install it with "
            "`uv pip install 'cupti-python>=13'`"
        ) from error
    if int(cupti_version.split(".", 1)[0]) < 13:
        raise RuntimeError(f"cupti-python >= 13 is required, found {cupti_version}")
    return cupti_version


def make_inputs(
    case: Case,
    heads: int,
    state_rotations: int,
) -> dict[str, torch.Tensor]:
    total_tokens = sum(case.seq_lens)
    shape = (1, total_tokens, heads, HEAD_DIM)
    generator = torch.Generator(device="cuda").manual_seed(case.seed)
    q, k, v, raw_g = (
        torch.randn(
            shape,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        for _ in range(4)
    )
    raw_beta = torch.randn(
        (1, total_tokens, heads),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    A_log = torch.rand(
        heads,
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )
    dt_bias = torch.rand(
        (heads, HEAD_DIM),
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )
    initial_state = (
        0.25
        * torch.randn(
            (len(case.seq_lens), heads, HEAD_DIM, HEAD_DIM),
            generator=generator,
            device="cuda",
            dtype=torch.float32,
        )
    ).to(torch.bfloat16)
    offsets = [0]
    for seq_len in case.seq_lens:
        offsets.append(offsets[-1] + seq_len)
    cu_seqlens = torch.tensor(offsets, device="cuda", dtype=torch.int64)
    seq_order = torch.argsort(cu_seqlens.diff(), descending=True).to(torch.int32)
    state_pool = initial_state.unsqueeze(0).expand(
        state_rotations, *initial_state.shape
    )
    return {
        "q": q,
        "k": k,
        "v": v,
        "raw_g": raw_g,
        "raw_beta": raw_beta,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "initial_state": initial_state,
        "state_pool": state_pool.clone(),
        "cu_seqlens": cu_seqlens,
        "seq_order": seq_order,
        "flashinfer_output": torch.empty_like(q),
    }


def measure_backend(
    run,
    *,
    warmup_iters: int,
    benchmark_iters: int,
    cold_l2_cache: bool,
) -> tuple[float, list[float]]:
    samples = bench_gpu_time(
        run,
        dry_run_iters=warmup_iters,
        repeat_iters=benchmark_iters,
        enable_cupti=True,
        use_cuda_graph=False,
        cold_l2_cache=cold_l2_cache,
    )
    samples_us = [float(sample) * 1000 for sample in samples]
    return statistics.median(samples_us), samples_us


def benchmark_case(
    case: Case,
    args: argparse.Namespace,
) -> dict[str, Any]:
    # CUPTI performs one initialization call and five estimation calls before
    # the requested warmup and measured iterations.
    state_rotations = args.warmup_iters + args.benchmark_iters + 8
    inputs = make_inputs(case, args.heads, state_rotations)
    result: dict[str, Any] = {
        "case": case.name,
        "heads": args.heads,
        "head_dim": HEAD_DIM,
        "seq_lens": list(case.seq_lens),
        "total_tokens": sum(case.seq_lens),
        "state_rotations": state_rotations,
    }

    flashinfer_state_pool = inputs["state_pool"]
    triton_state_pool = inputs["state_pool"].clone()
    cursors = {"flashinfer": 0, "triton": 0}
    triton_result: list[torch.Tensor | None] = [None, None]
    flashinfer_workspace = RecurrentKDAPrefillWorkspace(inputs["q"].device)

    def run_flashinfer() -> None:
        index = cursors["flashinfer"]
        cursors["flashinfer"] += 1
        _flashinfer_kda_prefill(
            q=inputs["q"],
            k=inputs["k"],
            v=inputs["v"],
            raw_g=inputs["raw_g"],
            raw_beta=inputs["raw_beta"],
            A_log=inputs["A_log"],
            dt_bias=inputs["dt_bias"],
            lower_bound=LOWER_BOUND,
            initial_state=flashinfer_state_pool[index],
            cu_seqlens=inputs["cu_seqlens"],
            out=inputs["flashinfer_output"],
            seq_order=inputs["seq_order"],
            prefill_workspace=flashinfer_workspace,
        )

    def run_triton() -> None:
        index = cursors["triton"]
        cursors["triton"] += 1
        output, final_state = chunk_kda_with_fused_gate(
            q=inputs["q"],
            k=inputs["k"],
            v=inputs["v"],
            raw_g=inputs["raw_g"],
            raw_beta=inputs["raw_beta"],
            A_log=inputs["A_log"],
            g_bias=inputs["dt_bias"],
            lower_bound=LOWER_BOUND,
            initial_state=triton_state_pool[index],
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=inputs["cu_seqlens"],
        )
        triton_result[:] = output, final_state

    if set(args.backends) == {"flashinfer", "triton"}:
        run_flashinfer()
        run_triton()
        torch.accelerator.synchronize()
        triton_output, triton_state = triton_result
        assert triton_output is not None and triton_state is not None
        output_delta = inputs["flashinfer_output"].float() - triton_output.float()
        state_delta = flashinfer_state_pool[0].float() - triton_state.float()
        result["triton_comparison"] = {
            "output_max_abs": output_delta.abs().max().item(),
            "output_relative_l2": (
                output_delta.norm() / triton_output.float().norm()
            ).item(),
            "state_max_abs": state_delta.abs().max().item(),
            "state_relative_l2": (
                state_delta.norm() / triton_state.float().norm()
            ).item(),
            "note": (
                "Diagnostic only: FlashInfer rounds the recurrent state to BF16 "
                "after each token, while Triton accumulates within a chunk."
            ),
        }

    timings: dict[str, Any] = {}
    for backend, run, state_pool in (
        ("flashinfer", run_flashinfer, flashinfer_state_pool),
        ("triton", run_triton, triton_state_pool),
    ):
        if backend not in args.backends:
            continue
        state_pool.copy_(inputs["initial_state"].unsqueeze(0))
        cursors[backend] = 0
        median_us, samples_us = measure_backend(
            run,
            warmup_iters=args.warmup_iters,
            benchmark_iters=args.benchmark_iters,
            cold_l2_cache=not args.warm_l2,
        )
        timings[backend] = {
            "median_us": median_us,
            "samples_us": samples_us,
        }
    if "flashinfer" in timings and "triton" in timings:
        timings["speedup"] = (
            timings["triton"]["median_us"] / timings["flashinfer"]["median_us"]
        )
    result["timings"] = timings
    return result


def main() -> None:
    args = parse_args()
    if not torch.accelerator.is_available():
        raise RuntimeError("CUDA is required")
    capability = get_compute_capability(torch.device("cuda"))
    if capability not in SUPPORTED_CAPABILITIES:
        raise RuntimeError(
            "FlashInfer recurrent KDA prefill requires SM100 or SM103, "
            f"found SM{capability[0]}{capability[1]}."
        )
    cupti_version = require_cupti()
    properties = torch.cuda.get_device_properties(0)
    metadata = {
        "device": properties.name,
        "compute_capability": list(capability),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "flashinfer_python": version("flashinfer-python"),
        "flashinfer_cubin": version("flashinfer-cubin"),
        "flashinfer_jit_cache": version("flashinfer-jit-cache"),
        "cupti_python": cupti_version,
        "cold_l2_cache": not args.warm_l2,
        "warmup_iters": args.warmup_iters,
        "benchmark_iters": args.benchmark_iters,
    }
    print(
        f"device: {properties.name}  cc: {capability[0]}.{capability[1]}  "
        f"heads: {args.heads}"
    )
    print(
        f"{'case':>12} {'tokens':>8} {'seqs':>6} {'FlashInfer us':>14} "
        f"{'Triton us':>11} {'speedup':>9}"
    )

    results = []
    selected_cases = [case for case in CASES if case.name in args.cases]
    for case in selected_cases:
        result = benchmark_case(case, args)
        results.append(result)
        timings = result["timings"]
        flashinfer_us = timings.get("flashinfer", {}).get("median_us")
        triton_us = timings.get("triton", {}).get("median_us")
        speedup = timings.get("speedup")
        print(
            f"{case.name:>12} {sum(case.seq_lens):8d} {len(case.seq_lens):6d} "
            f"{flashinfer_us if flashinfer_us is not None else float('nan'):14.3f} "
            f"{triton_us if triton_us is not None else float('nan'):11.3f} "
            f"{speedup if speedup is not None else float('nan'):8.2f}x"
        )

    if args.json is not None:
        args.json.write_text(
            json.dumps({"metadata": metadata, "results": results}, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
