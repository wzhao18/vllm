#!/usr/bin/env python3
"""KDA prefill in_proj on a prefix-cache checkpoint step: int53 path vs fused.

One KDA layer's in_proj + short conv + checkpoint conv-state export on an
F16K / F8K prefix-hit step (3 sequences with initial state, one checkpoint
per sequence at the last 256-aligned offset of its chunk), deployed K3.1 sizes
(P = 12288 planes, K = 7168, g|f_a|beta = 12512 columns), real-valued MXFP8
operands:

  stock  cuBLASLt in_proj (q|k|v|g|f_a|beta) + ops/kda_conv.qkv_conv + exporter
         reading the in_proj output (int53 on checkpoint steps)
  fused  KdaConvGemm with the checkpoint halo + cuBLASLt g|f_a|beta + exporter
         reading the halo (this change)
  fused0 KdaConvGemm without checkpoint + cuBLASLt g|f_a|beta (steps without
         checkpoints, both versions)

Checks the stored conv state (sequence lines + checkpoint slots) and the conv
output bitwise between stock and fused, then times interleaved A B B A blocks
(back-to-back calls per block, CUDA events) and prints B - A mean and 95% CI.
"""

import argparse
import json
import statistics

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", default="5461,2731")
    ap.add_argument("--p", type=int, default=12288)
    ap.add_argument("--k", type=int, default=7168)
    ap.add_argument("--rest", type=int, default=12512)
    ap.add_argument("--align", type=int, default=256)
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--calls", type=int, default=10)
    ap.add_argument("--pair", default="stock,fused")
    args = ap.parse_args()

    from flashinfer import mxfp8_quantize

    from vllm.model_executor.layers.mamba.checkpoint import (
        MambaPrefillCheckpointExporter,
        MambaPrefillCheckpointMetadata,
    )
    from vllm.models.kimi_k3.nvidia.ops import kda_conv
    from vllm.models.kimi_k3.nvidia.specialized_prefill import kda_conv_in_proj as kcp
    from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm.kda_conv_gemm import (  # noqa: E501
        KdaConvGemm,
    )
    from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

    dev = "cuda"
    p, k = args.p, args.k
    nq = 3 * p
    torch.manual_seed(0)
    w = torch.randn(nq + args.rest, k, device=dev, dtype=torch.bfloat16) * 0.05
    wq, ws = mxfp8_quantize(w, is_sf_swizzled_layout=True)
    del w
    ws = ws.view(torch.uint8).reshape(-1)
    cut = nq * (k // 32)
    cw = (torch.randn(nq, 4, device=dev) * 0.5).contiguous()
    gemm = KdaConvGemm(wq[:nq], ws[:cut], cw, p, tactic=kcp.TACTIC)
    w_rest, s_rest = wq[nq:], ws[cut:]
    exporter = MambaPrefillCheckpointExporter(state_len=3)

    def mm(xq, xs, wt, st):
        return torch._scaled_mm(
            xq, wt.t(),
            scale_a=xs.view(torch.float8_e8m0fnu), scale_b=st.view(torch.float8_e8m0fnu),
            out_dtype=torch.bfloat16,
        )  # fmt: skip

    a_arm, b_arm = args.pair.split(",")
    for sl in [int(v) for v in args.seqs.split(",")]:
        lens = [sl, sl, sl + 1]
        m = sum(lens)
        x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
        xq, xs = mxfp8_quantize(x, is_sf_swizzled_layout=True)
        xs = xs.view(torch.uint8).reshape(-1)
        qsl = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()],
                           dtype=torch.int32, device=dev)  # fmt: skip
        lines = torch.tensor([1, 2, 3], dtype=torch.int32, device=dev)
        has_init = torch.ones(3, dtype=torch.bool, device=dev)
        offs = [(n // args.align) * args.align for n in lens]
        ckpt = MambaPrefillCheckpointMetadata(
            torch.tensor(offs, dtype=torch.int32, device=dev),
            torch.tensor([4, 5, 6], dtype=torch.int32, device=dev),
            offsets=offs,
        )
        assert NULL_BLOCK_ID not in (1, 2, 3, 4, 5, 6)
        state0 = torch.randn(8, nq, 4, device=dev).to(torch.bfloat16)[:, :, :3]
        st = {a: state0.clone() for a in ("stock", "fused", "fused0")}
        halo = torch.empty(3, 3, nq, dtype=torch.bfloat16, device=dev)

        def stock():
            proj = mm(xq, xs, wq, ws)
            y = kda_conv.qkv_conv(proj[:, :nq], cw, st["stock"], qsl, lines, has_init, p)
            exporter.export(ckpt, raw_qkv=proj[:, :nq], conv_state=st["stock"],
                            cu_seqlens=qsl)  # fmt: skip
            return y

        def fused():
            y, _ = gemm(xq, xs, st["fused"], qsl, lines, has_init,
                        null_block_id=NULL_BLOCK_ID,
                        checkpoint_offsets=ckpt.checkpoint_offsets, halo=halo)  # fmt: skip
            mm(xq, xs, w_rest, s_rest)
            exporter.export(ckpt, raw_qkv=halo.view(9, nq), conv_state=st["fused"],
                            cu_seqlens=kcp._halo_qsl(ckpt))  # fmt: skip
            return y

        def fused0():
            y, _ = gemm(xq, xs, st["fused0"], qsl, lines, has_init,
                        null_block_id=NULL_BLOCK_ID)  # fmt: skip
            mm(xq, xs, w_rest, s_rest)
            return y

        arms = {"stock": stock, "fused": fused, "fused0": fused0}
        y_ref = stock()
        y_f = fused()
        torch.cuda.synchronize()
        res = {
            "m": m,
            "offsets": offs,
            "bitwise_out": bool(torch.equal(y_ref.view(torch.int16), y_f.view(torch.int16))),
            "bitwise_state": bool(
                torch.equal(st["stock"].view(torch.int16), st["fused"].view(torch.int16))
            ),
        }

        def block(fn):
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(args.calls):
                fn()
            e1.record()
            torch.cuda.synchronize()
            return e0.elapsed_time(e1) * 1000.0 / args.calls

        for fn in arms.values():
            block(fn)
        ta, tb, d = [], [], []
        for _ in range(args.rounds):
            a1 = block(arms[a_arm])
            b1 = block(arms[b_arm])
            b2 = block(arms[b_arm])
            a2 = block(arms[a_arm])
            ta += [a1, a2]
            tb += [b1, b2]
            d.append((b1 + b2 - a1 - a2) / 2)
        mean = statistics.fmean(d)
        ci = 2.201 * statistics.stdev(d) / len(d) ** 0.5 if len(d) > 1 else 0.0
        res.update(
            {
                a_arm + "_us": round(statistics.median(ta), 1),
                b_arm + "_us": round(statistics.median(tb), 1),
                "b_minus_a_us": round(mean, 1),
                "ci95_us": round(ci, 1),
                "rounds": len(d),
            }
        )
        print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
