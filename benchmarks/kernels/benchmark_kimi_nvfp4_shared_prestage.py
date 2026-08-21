# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Smoke and benchmark the staged NVFP4 Kimi shared-expert wrapper."""

from __future__ import annotations

import argparse
import time

import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.models.kimi_k3.nvidia.model import KimiFusedSharedExpert


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--intermediate", type=int, default=6144)
    parser.add_argument("--cudagraph-replays", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(0)
    started = time.perf_counter()
    with set_current_vllm_config(VllmConfig()):
        module = KimiFusedSharedExpert(
            args.hidden,
            args.intermediate,
            args.tokens,
            4.0,
            25.0,
            28,
            use_nvfp4=True,
        ).cuda()
    gate = torch.randn(
        args.intermediate,
        args.hidden,
        dtype=torch.bfloat16,
        device="cuda",
    ) / 64
    up = torch.randn_like(gate) / 64
    down = torch.randn(
        args.hidden,
        args.intermediate,
        dtype=torch.bfloat16,
        device="cuda",
    ) / 64
    module._load_gate_up_weight(module.gate_up_proj.weight, gate, 0)
    module._load_gate_up_weight(module.gate_up_proj.weight, up, 1)
    module.down_proj.weight.data.copy_(down)
    hidden = torch.randn(
        args.tokens, args.hidden, dtype=torch.bfloat16, device="cuda"
    )
    torch.cuda.synchronize()
    weights_ready = time.perf_counter()

    module.finalize_weights()
    torch.cuda.synchronize()
    weights_transformed = time.perf_counter()

    module.stage_nvfp4(hidden)
    actual = module.forward_staged_nvfp4(hidden).clone()
    if args.cudagraph_replays:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            module.stage_nvfp4(hidden)
            graph_output = module.forward_staged_nvfp4(hidden)
        for _ in range(args.cudagraph_replays):
            graph.replay()
        torch.cuda.synchronize()
        actual = graph_output.clone()
    gate_output = hidden @ gate.t()
    up_output = hidden @ up.t()
    gate_output = 4.0 * torch.tanh(gate_output / 4.0) * torch.sigmoid(gate_output)
    up_output = 25.0 * torch.tanh(up_output / 25.0)
    expected = (gate_output * up_output).to(torch.bfloat16) @ down.t()
    relative_l2 = (
        (actual.float() - expected.float()).norm() / expected.float().norm()
    ).item()
    cosine = torch.nn.functional.cosine_similarity(
        actual.float().flatten(), expected.float().flatten(), dim=0
    ).item()
    print(
        f"setup_seconds={weights_ready - started:.3f} "
        f"transform_seconds={weights_transformed - weights_ready:.3f}"
    )
    print(f"relative_l2={relative_l2:.6f} cosine={cosine:.8f}")


if __name__ == "__main__":
    main()
