# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbenchmark Kimi K3's KDA decode backends under CUDA graph replay.

Compares FlashInfer fused decode with BF16 recurrent state, vLLM's native fused
decode with FP32 recurrent state, and vLLM's Triton fallback with BF16 state.
The default shape is Kimi K3 TP8: 12 heads x 128, convolution width 4, and gate
lower bound -5.0. By default, each graph contains independently allocated state
for all 69 KDA layers in the model.

Example:
    .venv/bin/python benchmarks/kernels/benchmark_kimi_k3_kda_decode.py \
        --tokens 1 8 32 64 128 --heads 12 --layers 69
"""

import argparse
import functools

import torch

from vllm.triton_utils import triton

HEAD_DIM = 128
CONV_WIDTH = 4
GATE_LOWER_BOUND = -5.0
NORM_EPS = 1e-5
NUM_KDA_LAYERS = 69
DTYPE = torch.bfloat16


def _bench_graph_layers(calls: list) -> float:
    """Per-layer milliseconds for a graph holding one call per KDA layer."""
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
    total = triton.testing.do_bench(
        graph.replay, warmup=50, rep=300, return_mode="median"
    )
    return total / len(calls)


class Inputs:
    def __init__(
        self,
        num_tokens: int,
        num_heads: int,
        state_dtype: torch.dtype,
    ) -> None:
        torch.manual_seed(0)
        device = "cuda"
        dim = num_heads * HEAD_DIM
        num_slots = num_tokens + 8
        self.num_heads = num_heads
        self.num_tokens = num_tokens
        self.mixed_qkv = torch.randn(num_tokens, 3 * dim, device=device, dtype=DTYPE)
        self.conv_weights = torch.randn(
            3 * dim, CONV_WIDTH, device=device, dtype=torch.float32
        )
        self.decode_conv1d_weight = torch.stack(
            [
                self.conv_weights[i * dim : (i + 1) * dim].transpose(0, 1).contiguous()
                for i in range(3)
            ]
        )
        self.conv_state = torch.randn(
            num_slots, CONV_WIDTH - 1, 3 * dim, device=device, dtype=DTYPE
        )
        self.recurrent_state = torch.randn(
            num_slots,
            num_heads,
            HEAD_DIM,
            HEAD_DIM,
            device=device,
            dtype=state_dtype,
        )
        self.g1 = torch.randn(
            1, num_tokens, num_heads, HEAD_DIM, device=device, dtype=DTYPE
        )
        self.g2 = torch.randn(
            num_tokens, num_heads, HEAD_DIM, device=device, dtype=DTYPE
        )
        self.beta = torch.randn(1, num_tokens, num_heads, device=device, dtype=DTYPE)
        self.A_log = torch.randn(num_heads, device=device, dtype=torch.float32)
        self.dt_bias = torch.randn(dim, device=device, dtype=torch.float32)
        self.norm_weight_bf16 = torch.ones(HEAD_DIM, device=device, dtype=DTYPE)
        self.decode_norm_weight = self.norm_weight_bf16.float()
        # Slots start at 1: slot 0 is NULL_BLOCK_ID, which the fused kernel
        # treats as a padded row and skips, so timing it measures nothing.
        self.state_indices = torch.arange(
            1, num_tokens + 1, device=device, dtype=torch.int32
        )
        self.conv_state_t = self.conv_state.transpose(-1, -2)
        self.out = torch.empty(
            1, num_tokens, num_heads, HEAD_DIM, device=device, dtype=DTYPE
        )
        self.conv_out = torch.empty_like(self.mixed_qkv)


def _triton_gated_norm(inp: Inputs, core_attn_out: torch.Tensor) -> torch.Tensor:
    from vllm.third_party.flash_linear_attention.ops.kda import rms_norm_gated

    return rms_norm_gated(
        core_attn_out,
        inp.g2,
        inp.norm_weight_bf16,
        None,
        activation="sigmoid",
        eps=NORM_EPS,
    )


def vllm_fallback_bf16(inp: Inputs) -> None:
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_update,
    )
    from vllm.models.kimi_k3.nvidia.ops.third_party.kda import (
        fused_recurrent_kda_packed_decode,
    )

    causal_conv1d_update(
        inp.mixed_qkv,
        inp.conv_state_t,
        inp.conv_weights,
        None,
        activation="silu",
        conv_state_indices=inp.state_indices,
        validate_data=False,
        out=inp.conv_out,
    )
    core_attn_out, _ = fused_recurrent_kda_packed_decode(
        mixed_qkv=inp.conv_out,
        raw_g=inp.g1,
        raw_beta=inp.beta,
        A_log=inp.A_log,
        dt_bias=inp.dt_bias,
        lower_bound=GATE_LOWER_BOUND,
        initial_state=inp.recurrent_state,
        state_indices=inp.state_indices,
    )
    inp.out.copy_(_triton_gated_norm(inp, core_attn_out))


def native_fused_fp32(inp: Inputs) -> None:
    from vllm import _custom_ops as ops

    ops.fused_kda_decode(
        x=inp.mixed_qkv,
        weight=inp.decode_conv1d_weight,
        bias=None,
        conv_state=inp.conv_state_t,
        raw_g=inp.g1,
        raw_beta=inp.beta,
        A_log=inp.A_log,
        dt_bias=inp.dt_bias,
        state_indices=inp.state_indices,
        state=inp.recurrent_state,
        out=inp.out,
        lower_bound=GATE_LOWER_BOUND,
        output_gate=inp.g2,
        norm_weight=inp.decode_norm_weight,
        norm_eps=NORM_EPS,
    )


def flashinfer_fused_bf16(inp: Inputs) -> None:
    from vllm.models.kimi_k3.nvidia.kda import _flashinfer_fused_kda_decode

    _flashinfer_fused_kda_decode(
        x=inp.mixed_qkv,
        weight=inp.decode_conv1d_weight,
        conv_state=inp.conv_state_t,
        raw_g=inp.g1,
        raw_beta=inp.beta,
        A_log=inp.A_log,
        dt_bias=inp.dt_bias,
        state_indices=inp.state_indices,
        state=inp.recurrent_state,
        output_gate=inp.g2,
        norm_weight=inp.decode_norm_weight,
        lower_bound=GATE_LOWER_BOUND,
        norm_eps=NORM_EPS,
        out=inp.out,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 32, 64, 128])
    parser.add_argument(
        "--layers",
        type=int,
        default=NUM_KDA_LAYERS,
        help=(
            "give each of N layers its own state buffers so the recurrent "
            "state cannot remain resident in L2; defaults to Kimi K3's "
            f"{NUM_KDA_LAYERS} KDA layers"
        ),
    )
    parser.add_argument(
        "--heads",
        type=int,
        nargs="+",
        default=[12],
        help="KDA heads per rank (96 total / TP size)",
    )
    args = parser.parse_args()
    if args.layers <= 0:
        parser.error("--layers must be positive")

    if not hasattr(torch.ops._C, "fused_kda_decode"):
        raise SystemExit("vLLM was built without the fused KDA decode kernel")

    props = torch.cuda.get_device_properties(0)
    print(f"device: {props.name}  timing: cuda-graph replay  layers: {args.layers}")
    print(
        f"{'heads':>6} {'tokens':>7} {'FI BF16 us':>11} "
        f"{'native FP32 us':>14} {'fallback BF16 us':>16} "
        f"{'FI/native':>10} {'FI/fallback':>12}"
    )
    for num_heads in args.heads:
        for num_tokens in args.tokens:
            layers = [
                Inputs(num_tokens, num_heads, torch.bfloat16)
                for _ in range(args.layers)
            ]
            flashinfer_ms = _bench_graph_layers(
                [functools.partial(flashinfer_fused_bf16, inp) for inp in layers]
            )
            del layers
            torch.accelerator.empty_cache()

            layers = [
                Inputs(num_tokens, num_heads, torch.float32) for _ in range(args.layers)
            ]
            native_ms = _bench_graph_layers(
                [functools.partial(native_fused_fp32, inp) for inp in layers]
            )
            del layers
            torch.accelerator.empty_cache()

            layers = [
                Inputs(num_tokens, num_heads, torch.bfloat16)
                for _ in range(args.layers)
            ]
            fallback_ms = _bench_graph_layers(
                [functools.partial(vllm_fallback_bf16, inp) for inp in layers]
            )
            del layers
            torch.accelerator.empty_cache()

            print(
                f"{num_heads:>6} {num_tokens:>7} "
                f"{flashinfer_ms * 1e3:>11.2f} {native_ms * 1e3:>14.2f} "
                f"{fallback_ms * 1e3:>16.2f} "
                f"{native_ms / flashinfer_ms:>9.2f}x "
                f"{fallback_ms / flashinfer_ms:>11.2f}x"
            )
    print(
        "\nFI/native and FI/fallback are speedups: values above 1 mean "
        "FlashInfer BF16 is faster. Every measurement replays one CUDA graph "
        "containing one call for each independently allocated KDA layer."
    )


if __name__ == "__main__":
    main()
