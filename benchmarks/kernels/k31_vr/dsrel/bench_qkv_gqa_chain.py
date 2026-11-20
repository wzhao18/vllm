# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""qkv GEMM -> drafter GQA attention chain microbench (TP4 q3, one Rubin GPU, graphs).

Question: how much of GqaWindowUmma's fixed per-launch time (CTA start spread, setup,
first ring fill) hides behind its predecessor, the qkv projection, when the
predecessor lets the attention CTAs land early (griddepcontrol.launch_dependents at
its start) and the attention issues its first ring fill before griddepcontrol.wait.

Arms per batch (us per layer, median of 5 graph replays, cold KV and weights):
  gqa        attention alone (back-to-back calls)
  q:torch    qkv alone, cuBLAS (torch.mm, as F.linear in the drafter)
  q:pw       qkv alone, prewait_gemm (trigger after the accumulator)
  q:pws      qkv alone, prewait_gemm with launch_dependents at kernel start
  c:<q>      the chain qkv -> attention; exposed attention = c:<q> - q:<q>
``--fills`` rebuilds the attention with each first-fill placement (GqaWindowUmma
``fill``: 4 production, 2 the whole first ring before the PDL wait).

  python3 bench_qkv_gqa_chain.py --batches 32 --fills 4,2
"""

import argparse
import functools
import hashlib
import importlib.util
import os
import sys
from types import SimpleNamespace

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "bench_gqa_int", os.path.join(HERE, "bench_gqa_attention_int.py")
)
bg = importlib.util.module_from_spec(_spec)
sys.modules["bench_gqa_int"] = bg
_spec.loader.exec_module(bg)


def start_trigger_module():
    """prewait_gemm with griddepcontrol.launch_dependents at the kernel start."""
    from torch.utils.cpp_extension import load_inline

    import vllm.envs as envs
    from vllm.models.kimi_k3.nvidia.specialized_decode import prewait_gemm as pw

    src = pw._CUDA_SRC
    head = src.index("wsg_k(")
    brace = src.index("{", src.index("const P p)", head))
    src = (
        src[: brace + 1]
        + '\n  if (threadIdx.x == 0) asm volatile("griddepcontrol.launch_dependents;");'
        + src[brace + 1 :]
    )
    tag = hashlib.sha256((src + pw._CPP_SRC).encode()).hexdigest()[:16]
    build = os.path.join(envs.VLLM_CACHE_ROOT, "k3_prewait_gemm_start", tag)
    os.makedirs(build, exist_ok=True)
    return load_inline(
        name=f"k3_prewait_gemm_start_{tag}", cpp_sources=pw._CPP_SRC,
        cuda_sources=src, functions=["prewait_gemm"],
        extra_cuda_cflags=["-O3", "-U__CUDA_NO_HALF_OPERATORS__",
                           "-U__CUDA_NO_HALF_CONVERSIONS__",
                           "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-gencode",
                           "arch=compute_107a,code=sm_107a"],
        build_directory=build, verbose=False,
    )  # fmt: skip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", default="32")
    parser.add_argument("--fills", default="4,2")
    parser.add_argument("--ctx", type=int, default=150000)
    parser.add_argument("--qkv", default="torch,pw,pws")
    args = parser.parse_args()

    from vllm.forward_context import ForwardContext, override_forward_context
    from vllm.models.kimi_k3.nvidia.specialized_decode import prewait_gemm as pw
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.draft import (
        gqa_attention as ga,
    )
    from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    H, q, G, HEAD = 4, 3, bg.G, bg.HEAD
    width = (G * H + 2 * H) * HEAD
    hidden = 7168
    pws = start_trigger_module() if "pws" in args.qkv else None
    table = bg.rope_table(device)
    orig = ga.compile_window_umma
    L = args.ctx + q
    page = 4096
    layer_page = H * page * 2 * HEAD
    stride = bg.LAYERS * layer_page + 128
    lo = L - q - (bg.WINDOW - 1)
    pages = list(range(lo // page, (L - 1) // page + 1))
    for B in [int(b) for b in args.batches.split(",")]:
        rows = B * q
        per_call = B * H * (bg.WINDOW - 1 + q) * 256
        copies = max(2, min(64, (600 << 20) // per_call))
        blocks = copies * B * len(pages) + 2
        buffer = torch.empty(blocks * stride, dtype=torch.bfloat16, device=device)
        buffer.normal_()
        views = [
            buffer.as_strided((blocks, H, page, 2 * HEAD),
                              (stride, page * 2 * HEAD, 2 * HEAD, 1), i * layer_page)
            for i in range(bg.LAYERS)
        ]  # fmt: skip
        perm = torch.randperm(blocks - 1, device=device)[: copies * B * len(pages)] + 1
        perm = perm.view(copies, B, len(pages)).int()
        seq = torch.full((B,), L, dtype=torch.int32, device=device)
        positions = torch.arange(L - q, L, device=device).repeat(B)
        x = torch.randn((rows, hidden), device=device).bfloat16() * 0.05
        ws = [torch.randn((width, hidden), device=device).bfloat16() * 0.02
              for _ in range(copies)]  # fmt: skip
        packed = [pw.pack_weight(w) for w in ws] if pw.supported(rows, hidden) else None
        cfg = pw.CONFIGS.get(rows)
        cases = []
        for c in range(copies):
            tbl = torch.zeros((B, (L - 1) // page + 2), dtype=torch.int32, device=device)
            tbl[:, pages[0] : pages[-1] + 1] = perm[c]
            pos = positions.cpu()
            slots = (tbl.cpu()[torch.arange(B).repeat_interleave(q), pos // page].long()
                     * page + pos % page).to(device)  # fmt: skip
            md = FlashAttentionMetadata(
                num_actual_tokens=rows, max_query_len=q,
                query_start_loc=torch.arange(0, (B + 1) * q, q, dtype=torch.int32,
                                             device=device),
                max_seq_len=L, seq_lens=seq, block_table=tbl, slot_mapping=slots,
                causal=True,
            )  # fmt: skip
            cases.append((md, slots, torch.empty((rows, width), device=device,
                                                 dtype=torch.bfloat16)))  # fmt: skip

        def qkv_fn(kind, c):
            out = cases[c][2]
            if kind == "torch":
                return lambda: torch.mm(x, ws[c].t(), out=out)
            if kind == "pw":
                return lambda: pw.gemm(out, x, packed[c], width)
            return lambda: pws.prewait_gemm(out, x, packed[c], width, *cfg)

        res = {}
        kinds = [k for k in args.qkv.split(",") if k == "torch" or packed is not None]
        for kind in kinds:
            res[f"q:{kind}"] = bg.timed([qkv_fn(kind, c) for c in range(copies)], 20)
        for fill in [int(f) for f in args.fills.split(",")]:
            ga.compile_window_umma = functools.partial(orig, fill=fill)
            layers = bg.make_layers(H, device, table)
            config = SimpleNamespace(
                speculative_config=SimpleNamespace(num_speculative_tokens=q),
                model_config=SimpleNamespace(max_model_len=bg.MAX_POS),
            )
            state = ga.install_gqa_attention(layers, config, [rows], [B * (q + 1)], True)
            for i, layer in enumerate(layers):
                layer.attn.bind_kv_cache(views[i])
            names = [layer.attn.layer_name for layer in layers]

            def attn_fn(c):
                md, slots, out = cases[c]
                fc = ForwardContext(no_compile_layers={},
                                    attn_metadata={n: md for n in names},
                                    slot_mapping={n: slots for n in names})  # fmt: skip

                def run():
                    with override_forward_context(fc):
                        assert state.forward(0, out, positions) is not None

                return run

            loops = max(1, 200 // copies)
            res[f"gqa f{fill}"] = bg.timed([attn_fn(c) for c in range(copies)], loops)
            for kind in kinds:
                chain = []
                for c in range(copies):
                    chain += [qkv_fn(kind, c), attn_fn(c)]
                # timed() divides by len(calls): x2 = us per (qkv + attention) pair
                res[f"c:{kind} f{fill}"] = 2 * bg.timed(chain, loops)
        ga.compile_window_umma = orig
        print(f"B{B} rows {rows} shape {ga.window_shape(H, q, B)} copies {copies} "
              f"prewait cfg {cfg}")  # fmt: skip
        for k, v in res.items():
            extra = ""
            if k.startswith("c:"):
                kind = k.split()[0][2:]
                extra = f"   exposed attention {v - res['q:' + kind]:6.2f}"
            print(f"  {k:14s} {v:7.2f}{extra}", flush=True)
        del buffer, views, cases, ws, packed


if __name__ == "__main__":
    main()
