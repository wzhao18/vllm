# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
from pathlib import Path

import torch
from safetensors import safe_open

from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    ref_nvfp4_quant_dequant,
)
from vllm.utils.deep_gemm import per_block_cast_to_fp8


def load_weight(checkpoint: Path, name: str) -> torch.Tensor:
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def quantize_dequantize(weight: torch.Tensor) -> torch.Tensor:
    global_scale = ((448.0 * 6.0) / weight.float().abs().max()).reshape(1)
    return ref_nvfp4_quant_dequant(weight.cuda(), global_scale.cuda(), 16)


def fp8_quantize_dequantize(weight: torch.Tensor) -> torch.Tensor:
    quantized, scale = per_block_cast_to_fp8(
        weight.cuda(), [128, 128], use_ue8m0=True
    )
    expanded = scale.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    return (quantized.float() * expanded[: weight.shape[0], : weight.shape[1]]).to(
        weight.dtype
    )


def fp8_activation_quantize_dequantize(value: torch.Tensor) -> torch.Tensor:
    quantized, scale = per_token_group_quant_fp8(
        value.contiguous(), 128, use_ue8m0=True
    )
    expanded = scale.repeat_interleave(128, dim=-1)
    return (quantized.float() * expanded).to(value.dtype)


def relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual.float() - expected.float()).norm() / expected.float().norm())


def cosine_similarity(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float(
        torch.nn.functional.cosine_similarity(
            actual.float().flatten(), expected.float().flatten(), dim=0
        )
    )


def situ(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    gate = 4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)
    up = 25.0 * torch.tanh(up / 25.0)
    return gate * up


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--tokens", type=int, nargs="+", default=[32, 160, 256])
    args = parser.parse_args()

    prefix = f"language_model.model.layers.{args.layer}.block_sparse_moe"
    gate = load_weight(
        args.checkpoint, f"{prefix}.shared_experts.gate_proj.weight"
    ).cuda()
    up = load_weight(args.checkpoint, f"{prefix}.shared_experts.up_proj.weight").cuda()
    down = load_weight(
        args.checkpoint, f"{prefix}.shared_experts.down_proj.weight"
    ).cuda()
    gate_qdq = quantize_dequantize(gate)
    up_qdq = quantize_dequantize(up)
    down_qdq = quantize_dequantize(down)
    gate_fp8 = fp8_quantize_dequantize(gate)
    up_fp8 = fp8_quantize_dequantize(up)
    down_fp8 = fp8_quantize_dequantize(down)

    for name, weight, quantized in (
        ("gate", gate, gate_qdq),
        ("up", up, up_qdq),
        ("down", down, down_qdq),
    ):
        print(
            f"weight={name} rel_l2={relative_l2(quantized, weight):.6f} "
            f"cosine={cosine_similarity(quantized, weight):.8f}"
        )

    for name, weight, quantized in (
        ("gate", gate, gate_fp8),
        ("up", up, up_fp8),
        ("down", down, down_fp8),
    ):
        print(
            f"weight_fp8={name} rel_l2={relative_l2(quantized, weight):.6f} "
            f"cosine={cosine_similarity(quantized, weight):.8f}"
        )

    torch.manual_seed(42)
    hidden_size = gate.shape[1]
    for tokens in args.tokens:
        hidden = torch.randn(
            tokens, hidden_size, device="cuda", dtype=torch.bfloat16
        )
        expected = situ(
            torch.nn.functional.linear(hidden, gate),
            torch.nn.functional.linear(hidden, up),
        )
        expected = torch.nn.functional.linear(expected, down)
        actual = situ(
            torch.nn.functional.linear(hidden, gate_qdq),
            torch.nn.functional.linear(hidden, up_qdq),
        )
        actual = torch.nn.functional.linear(actual, down_qdq)
        actual_fp8 = situ(
            torch.nn.functional.linear(hidden, gate_fp8),
            torch.nn.functional.linear(hidden, up_fp8),
        )
        actual_fp8 = torch.nn.functional.linear(actual_fp8, down_fp8)
        hidden_fp8 = fp8_activation_quantize_dequantize(hidden)
        actual_fp8_full = situ(
            torch.nn.functional.linear(hidden_fp8, gate_fp8),
            torch.nn.functional.linear(hidden_fp8, up_fp8),
        )
        actual_fp8_full = fp8_activation_quantize_dequantize(actual_fp8_full)
        actual_fp8_full = torch.nn.functional.linear(actual_fp8_full, down_fp8)
        print(
            f"tokens={tokens} output_rel_l2={relative_l2(actual, expected):.6f} "
            f"output_cosine={cosine_similarity(actual, expected):.8f} "
            f"fp8_output_rel_l2={relative_l2(actual_fp8, expected):.6f} "
            f"fp8_output_cosine={cosine_similarity(actual_fp8, expected):.8f} "
            f"full_fp8_output_rel_l2={relative_l2(actual_fp8_full, expected):.6f} "
            f"full_fp8_output_cosine="
            f"{cosine_similarity(actual_fp8_full, expected):.8f}"
        )


if __name__ == "__main__":
    main()
