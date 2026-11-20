# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P-GEMM R7: MegaMoE dedup top-k reduce with the latent RMSNorm fused (E2a-dedup) vs
the deployed pair (dedup reduce, then the stock latent rms_norm), one GPU, at the
prefill sub-call shape (8192 rows x latent 3584, world 4 / 16, top-16 of 896 routing
masks). Warm (back-to-back) and THRASH (512 MB L2 flush before every call) arms;
correctness: fused vs reduce + stock rms_norm <= 1 BF16 ulp.

    python3 pgemm_r7_lnorm.py [--rows 8192] [--hidden 3584] [--iters 50]
"""

import argparse
import json

import torch

from vllm import _custom_ops as ops


def compile_dedup(hidden, world, **kw):
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as cutlass_torch

    from vllm.models.kimi_k3.nvidia.megamoe_rubin.src_gp.sources.kernel_src.rubin.inference.mega.topk_reduce_dedup import (  # noqa: E501
        TopkReduceDedup,
    )
    from vllm.models.kimi_k3.nvidia.megamoe_rubin.src_gp.sources.quant_def import (
        CombineFormat,
    )

    def conv(v):
        if isinstance(v, torch.Tensor):
            return cutlass_torch.from_dlpack(v, assumed_align=16).mark_layout_dynamic(
                leading_dim=v.dim() - 1
            )
        return cutlass.Float32(v) if isinstance(v, float) else v

    op = TopkReduceDedup(hidden, world, CombineFormat.parse("bf16"))
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    fn = cute.compile(op, stream=stream, **{k: conv(v) for k, v in kw.items()})
    conv_kw = {k: conv(v) for k, v in kw.items()}
    return lambda: fn(stream=stream, **conv_kw)


def routing_mask(rows, world, experts=896, topk=16, gen=None):
    scores = torch.rand((rows, experts), device="cuda", generator=gen)
    ids = scores.topk(topk, dim=1).indices
    ranks = ids // (experts // world)
    mask = torch.zeros(rows, dtype=torch.int64, device="cuda")
    for k in range(topk):
        mask |= 1 << ranks[:, k]
    return mask.to(torch.int32)


def timed(fn, iters, flush=None):
    start = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    for i in range(iters):
        if flush is not None:
            flush.zero_()
        start[i].record()
        fn()
        end[i].record()
    torch.cuda.synchronize()
    t = sorted(s.elapsed_time(e) * 1e3 for s, e in zip(start, end))
    return t[len(t) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=8192)
    ap.add_argument("--hidden", type=int, default=3584)
    ap.add_argument("--worlds", default="4,16")
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()
    torch.cuda.set_device(0)
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(7)
    flush = torch.empty(512 << 20, dtype=torch.uint8, device=dev)
    n, hidden = args.rows, args.hidden
    weight = (1 + 0.1 * torch.randn(hidden, device=dev, generator=g)).to(torch.bfloat16)
    eps = 1e-6
    for world in (int(w) for w in args.worlds.split(",")):
        slots = (torch.randn((n, world, hidden), device=dev, generator=g) * 0.5).to(
            torch.bfloat16
        )
        mask = routing_mask(n, world, gen=g)
        reduced = torch.empty((n, hidden), dtype=torch.bfloat16, device=dev)
        normed = torch.empty_like(reduced)
        fused_out = torch.empty_like(reduced)
        plain = compile_dedup(
            hidden, world, rank_slots=slots, valid_rank_mask=mask, reduced_output=reduced
        )
        fused = compile_dedup(
            hidden,
            world,
            rank_slots=slots,
            valid_rank_mask=mask,
            reduced_output=fused_out,
            rmsnorm_weight=weight,
            rmsnorm_eps=eps,
        )

        def norm_only():
            ops.rms_norm(normed, reduced, weight, eps)

        def pair():
            plain()
            norm_only()

        pair()
        fused()
        torch.cuda.synchronize()
        ulp = (fused_out.view(torch.int16).int() - normed.view(torch.int16).int()).abs()
        res = {
            "world": world,
            "rows": n,
            "hidden": hidden,
            "ranks_per_token": float(
                ((mask[:, None] >> torch.arange(world, device=dev)) & 1).sum(1).float().mean()
            ),
            "max_ulp": int(ulp.max()),
            "frac_ulp": float((ulp > 0).float().mean()),
        }
        for arm, fn in (("reduce", plain), ("norm", norm_only), ("pair", pair), ("fused", fused)):
            res[f"{arm}_warm_us"] = round(timed(fn, args.iters), 2)
            res[f"{arm}_thrash_us"] = round(timed(fn, args.iters, flush), 2)
        res["saved_warm_us"] = round(res["pair_warm_us"] - res["fused_warm_us"], 2)
        res["saved_thrash_us"] = round(res["pair_thrash_us"] - res["fused_thrash_us"], 2)
        print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
