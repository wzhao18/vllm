# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 KDA prefill chain microbench at the deployed per-GPU shape.

Default shape: DEP16 (P3-16) F16K, 3 sequences x ~5.4K fresh tokens at TP1,
96 heads, head_dim 128, FP32 recurrent state, in_proj row stride 49376.
Times every kernel of the chain with torch.profiler (CUPTI): the fused Q|K|V
conv, FlashKDA (stock build: prepare K1 + recurrence K2 + beta transpose),
the gated o_norm with the MXFP8 o_proj-input write, and the state glue.

usage: python bench_kda_prefill_chain.py [--tokens 16384] [--seqs 3]
       [--heads 96] [--iters 10] [--impl stock|split|onorm]
"""

import argparse
import json
from collections import defaultdict

import torch

H_DEFAULT = 96
D = 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=16384)
    ap.add_argument("--seqs", type=int, default=3)
    ap.add_argument("--heads", type=int, default=H_DEFAULT)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--row-stride", type=int, default=49376)
    ap.add_argument("--json", default="")
    ap.add_argument("--impl", default="stock", choices=["stock", "split", "onorm", "tc", "fla"])
    args = ap.parse_args()

    import vllm._flashkda_C  # noqa: F401

    from vllm.models.kimi_k3.nvidia.ops import kda_conv, kda_norm
    from vllm.models.kimi_k3.nvidia.ops.mxfp8_act import Mxfp8Buffer

    dev = torch.device("cuda")
    torch.manual_seed(0)
    T, N, H = args.tokens, args.seqs, args.heads
    P = H * D
    lens = [T // N] * N
    lens[-1] += T - sum(lens)
    cu = torch.tensor(
        [0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device=dev
    )
    bf = torch.bfloat16
    proj = torch.randn(T, args.row_stride, dtype=bf, device=dev) * 0.5
    mixed_qkv = proj[:, : 3 * P]
    g2 = proj[:, 3 * P : 4 * P].view(T, H, D)
    beta = proj[:, 4 * P + D : 4 * P + D + H].unsqueeze(0)  # [1, T, H] strided
    g1 = torch.randn(1, T, H, D, dtype=bf, device=dev) * 0.5
    conv_w = torch.randn(3 * P, 4, dtype=torch.float32, device=dev) * 0.3
    lines = N + 2
    conv_state = torch.zeros(lines, 3 * P, 3, dtype=bf, device=dev)
    sidx = torch.arange(1, N + 1, dtype=torch.int32, device=dev)
    has_init = torch.zeros(N, dtype=torch.bool, device=dev)
    A_log = torch.randn(H, dtype=torch.float32, device=dev) * 0.5
    dt_bias = torch.randn(H * D, dtype=torch.float32, device=dev) * 0.1
    lower_bound = -5.0
    init_state = torch.zeros(N, H, D, D, dtype=torch.float32, device=dev)
    final_state = torch.empty_like(init_state)
    out = torch.empty(1, T, H, D, dtype=bf, device=dev)
    ws = torch.empty(
        torch.ops._flashkda_C.get_workspace_size(T, H, N), dtype=torch.uint8, device=dev
    )
    norm_w = torch.ones(D, dtype=torch.float32, device=dev)
    mx = Mxfp8Buffer(T, P, dev, bf)

    from vllm.models.kimi_k3.nvidia.ops import kda_segscan
    from vllm.models.kimi_k3.nvidia.specialized_prefill.kda import o_norm_mxfp8

    kda_segscan.load_extension()

    def step_split():
        q, k, v = kda_conv.qkv_conv(
            mixed_qkv, conv_w, conv_state, cu, sidx, has_init, P
        ).unbind(0)
        q, k, v = (x.view(1, T, H, D) for x in (q, k, v))
        torch.ops.k3_flashkda_split.fwd(
            q,
            k,
            v,
            g1,
            beta,
            D**-0.5,
            out,
            ws,
            A_log,
            dt_bias.view(H, D),
            lower_bound,
            init_state,
            final_state,
            cu,
            None,
            None,
            3,
        )
        kda_norm.rmsnorm_gated_mxfp8(out, g2, norm_w, 1e-6, out=mx)

    def step_stock():
        q, k, v = kda_conv.qkv_conv(
            mixed_qkv, conv_w, conv_state, cu, sidx, has_init, P
        ).unbind(0)
        q, k, v = (x.view(1, T, H, D) for x in (q, k, v))
        torch.ops._flashkda_C.fwd(
            q,
            k,
            v,
            g1,
            beta,
            D**-0.5,
            out,
            ws,
            A_log,
            dt_bias.view(H, D),
            lower_bound,
            init_state,
            final_state,
            cu,
            None,
            None,
        )
        kda_norm.rmsnorm_gated_mxfp8(out, g2, norm_w, 1e-6, out=mx)

    def step_onorm():
        q, k, v = kda_conv.qkv_conv(
            mixed_qkv, conv_w, conv_state, cu, sidx, has_init, P
        ).unbind(0)
        q, k, v = (x.view(1, T, H, D) for x in (q, k, v))
        torch.ops._flashkda_C.fwd(
            q,
            k,
            v,
            g1,
            beta,
            D**-0.5,
            out,
            ws,
            A_log,
            dt_bias.view(H, D),
            lower_bound,
            init_state,
            final_state,
            cu,
            None,
            None,
        )
        o_norm_mxfp8(out, proj[:, 3 * P : 4 * P], norm_w, 1e-6, mx)

    def step_tc():
        q, k, v = kda_conv.qkv_conv(
            mixed_qkv, conv_w, conv_state, cu, sidx, has_init, P
        ).unbind(0)
        q, k, v = (x.view(1, T, H, D) for x in (q, k, v))
        torch.ops.k3_flashkda_split.fwd(
            q,
            k,
            v,
            g1,
            beta,
            D**-0.5,
            out,
            ws,
            A_log,
            dt_bias.view(H, D),
            lower_bound,
            init_state,
            final_state,
            cu,
            None,
            None,
            3 | 32,
        )
        o_norm_mxfp8(out, proj[:, 3 * P : 4 * P], norm_w, 1e-6, mx)

    def step_fla():
        # vLLM's FLA chunk_kda (chunk 64, sub-chunk 16 intra, Triton): the
        # chunk-64 recurrence as a measured reference for the chain latency.
        from vllm.models.kimi_k3.nvidia.ops.third_party.kda.chunk import (
            chunk_kda_with_fused_gate,
        )

        q, k, v = kda_conv.qkv_conv(
            mixed_qkv, conv_w, conv_state, cu, sidx, has_init, P
        ).unbind(0)
        q, k, v = (x.view(1, T, H, D) for x in (q, k, v))
        chunk_kda_with_fused_gate(
            q, k, v, g1, beta.contiguous(), A_log, dt_bias, D**-0.5,
            init_state, True, lower_bound, True, cu, out,
        )
        o_norm_mxfp8(out, proj[:, 3 * P : 4 * P], norm_w, 1e-6, mx)

    step = {
        "fla": step_fla,
        "tc": step_tc,
        "stock": step_stock,
        "split": step_split,
        "onorm": step_onorm,
    }[args.impl]
    for _ in range(3):
        step()
    torch.cuda.synchronize()
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(args.iters):
            step()
        torch.cuda.synchronize()
    acc = defaultdict(float)
    cnt = defaultdict(int)
    for e in prof.events():
        if e.device_type.name == "CUDA" and e.device_time_total > 0:
            acc[e.name] += e.device_time_total
            cnt[e.name] += 1
    rows = sorted(
        ((v / args.iters, cnt[k] // args.iters, k) for k, v in acc.items()),
        reverse=True,
    )
    total = sum(r[0] for r in rows)
    print(f"impl={args.impl} shape T={T} N={N} H={H}: total {total:.1f} us/layer")
    for us, c, name in rows:
        print(f"  {us:9.1f} us  x{c}  {name[:150]}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {
                    "impl": args.impl,
                    "T": T,
                    "N": N,
                    "H": H,
                    "total_us": total,
                    "kernels": [[n, us, c] for us, c, n in rows],
                },
                f,
                indent=1,
            )


if __name__ == "__main__":
    main()
