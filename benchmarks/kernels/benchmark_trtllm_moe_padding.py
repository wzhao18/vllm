# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check graph padding and time full TRT-LLM MoE, including modular routing.

Requires padding-mask support in FlashInfer and vLLM grouped top-k. ``--kimi``
uses Kimi-K3 TP8 per-rank dimensions. Packed weights have constant scales;
correctness compares masked and unmasked execution with identical weights.
Timing uses CUDA graphs and CUPTI with cold L2. Run on an otherwise idle GPU.
"""

import argparse
import json
import statistics
from functools import partial

import torch
from flashinfer.fused_moe import (
    trtllm_fp4_block_scale_moe,
    trtllm_fp4_block_scale_routed_moe,
)
from flashinfer.testing import bench_gpu_time_with_cupti
from flashinfer.tllm_enums import ActivationType, RoutingMethodType

import vllm._custom_ops as ops


def make_calls(common, logits, bias):
    k = common["top_k"]

    def mono(padding, finalize=True):
        return trtllm_fp4_block_scale_moe(
            **common,
            routing_logits=logits,
            routing_bias=bias,
            is_padding=padding,
            do_finalize=finalize,
        )

    def modular(padding):
        probs, ids = ops.grouped_topk(
            logits, 1, 1, k, True, 1.0, bias, 1, is_padding=padding
        )
        return trtllm_fp4_block_scale_routed_moe(
            **common,
            topk_ids=(ids, probs),
            routing_bias=None,
        )

    return mono, modular


def main(args):
    torch.manual_seed(17)
    e, h, i, k = (896, 3584, 384, 16) if args.kimi else (32, 256, 256, 8)
    print(
        json.dumps(
            dict(
                gpu=torch.cuda.get_device_name(),
                experts=e,
                hidden=h,
                intermediate=i,
                topk=k,
                padding_logits=args.padding_logits,
            )
        ),
        flush=True,
    )
    weights = dict(
        gemm1_weights=torch.randint(
            0, 256, (e, 2 * i, h // 2), dtype=torch.uint8, device="cuda"
        ),
        gemm2_weights=torch.randint(
            0, 256, (e, h, i // 2), dtype=torch.uint8, device="cuda"
        ),
        gemm1_weights_scale=torch.full(
            (e, 2 * i, h // 32), 121, dtype=torch.uint8, device="cuda"
        ).view(torch.float8_e4m3fn),
        gemm2_weights_scale=torch.full(
            (e, h, i // 32), 121, dtype=torch.uint8, device="cuda"
        ).view(torch.float8_e4m3fn),
        gemm1_bias=None,
        gemm2_bias=None,
        gemm1_alpha=torch.full((e,), 4.0, device="cuda"),
        gemm1_beta=torch.full((e,), 25.0, device="cuda"),
        gemm1_clamp_limit=None,
        output1_scale_scalar=None,
        output1_scale_gate_scalar=None,
        output2_scale_scalar=None,
        num_experts=e,
        top_k=k,
        n_group=1,
        topk_group=1,
        intermediate_size=i,
        local_expert_offset=0,
        local_num_experts=e,
        routed_scaling_factor=1.0,
        routing_method_type=RoutingMethodType.DeepSeekV3,
        activation_type=ActivationType.Situ.value,
        enable_pdl=True,
    )
    bias = torch.rand(e, dtype=torch.float32, device="cuda") * 0.05
    for b, n in [
        (16, 9),
        (24, 17),
        (40, 33),
        (72, 65),
        (128, 121),
        (144, 137),
        (272, 257),
    ]:
        x = torch.randn((b, h), device="cuda").to(torch.float8_e4m3fn)
        scale = torch.full((b, h // 32), 127, dtype=torch.uint8, device="cuda").view(
            torch.float8_e4m3fn
        )
        logits = torch.randn((b, e), dtype=torch.float32, device="cuda")
        if args.padding_logits == "zero":
            logits[n:] = 0
        mask = torch.arange(b, device="cuda") >= n
        common = dict(weights, hidden_states=x, hidden_states_scale=scale)
        mono, modular = make_calls(common, logits, bias)

        for mode, fn in [("monolithic", mono), ("modular", modular)]:
            baseline = fn(None)[0].clone()
            fn(mask)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = fn(mask)[0]
            for pattern in ("prefix", "empty", "full", "gaps", "prefix"):
                mask.fill_(pattern == "empty")
                if pattern == "prefix":
                    mask[n:] = True
                elif pattern == "gaps":
                    mask[1::2] = True
                graph.replay()
                torch.accelerator.synchronize()
                torch.testing.assert_close(
                    output[~mask], baseline[~mask], atol=0.002, rtol=0.02
                )
                assert torch.isfinite(output).all()
                assert (output[mask] == 0).all()
            print(
                json.dumps(dict(mode=mode, capacity=b, valid=n, correctness="passed")),
                flush=True,
            )
            if args.bench:
                for skip in (False, True):
                    timing = bench_gpu_time_with_cupti(
                        partial(fn, mask if skip else None),
                        dry_run_iters=10,
                        repeat_iters=50,
                        use_cuda_graph=True,
                        cold_l2_cache=True,
                    )
                    print(
                        json.dumps(
                            dict(
                                mode=mode,
                                capacity=b,
                                valid=n,
                                skip=skip,
                                median_us=statistics.median(timing) * 1000,
                                useful_tflops=(6 * n * k * h * i)
                                / (statistics.median(timing) * 1e9),
                            )
                        ),
                        flush=True,
                    )
        # Check the unfinalized mapping consumed by Kimi's fused tail.
        mask[:] = torch.arange(b, device="cuda") >= n
        base, base_w, base_idx = mono(None, False)
        base_result = (base[base_idx.view(b, k)].float() * base_w.view(b, k, 1)).sum(1)
        out, w, idx = mono(mask, False)
        idx = idx.view(b, k)
        assert (idx[mask] == -1).all()
        result = (out[idx[:n]].float() * w.view(b, k, 1)[:n]).sum(1)
        torch.testing.assert_close(result, base_result[:n], atol=0.002, rtol=0.02)
        print(
            json.dumps(
                dict(mode="deferred", capacity=b, valid=n, correctness="passed")
            ),
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kimi", action="store_true")
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--padding-logits", choices=["zero", "random"], default="zero")
    main(parser.parse_args())
