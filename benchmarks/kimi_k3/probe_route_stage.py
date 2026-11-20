# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Probe: route_stage latency breakdown by source variants (timing only; the
variants do not produce correct outputs). Same timing as bench_route_stage."""

import argparse
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from bench_route_stage import E, HIDDEN, bufs, timeit  # noqa: E402

VARIANTS = {
    "full": [],
    "noquant": [("  if (quant) {\n    if (NVFP4) {", "  if (false) {\n    if (NVFP4) {")],
    "notopk": [("  route_block(a, row, lg, s_bias, s_unb, s_cand);\n}", "}")],
    "pass1": [("  if (wid != 0) return;", "  return;")],
    "empty": [
        ("  if (quant) {\n    if (NVFP4) {", "  if (false) {\n    if (NVFP4) {"),
        ("  route_block(a, row, lg, s_bias, s_unb, s_cand);\n}", "}"),
    ],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", default="32,96")
    p.add_argument("--variants", default=",".join(VARIANTS))
    args = p.parse_args()
    import vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe.route_stage as rs

    base = rs._CUDA_SRC
    dev = torch.device("cuda")
    for name in args.variants.split(","):
        src = base
        for old, new in VARIANTS[name]:
            assert old in src, (name, old)
            src = src.replace(old, new)
        rs._CUDA_SRC, rs._mod = src, None
        mod = rs._module()
        line = []
        for n in [int(r) for r in args.rows.split(",")]:
            torch.manual_seed(n)
            logits = torch.randn(n, E, device=dev) * 3
            bias = torch.randn(E, device=dev) * 0.1
            latent = (torch.randn(n, HIDDEN, device=dev) * 4).to(torch.bfloat16)
            o = bufs(384, True, dev)

            def f():
                mod.route_stage(logits, E, bias, o[0], o[1], o[2], o[3], o[4], None,
                                E, True, 2.827, True, 2.5, latent)

            line.append(f"n{n}={timeit(f, 24, 50, 7):.2f}")
        print(f"VARIANT {name:8s} " + " ".join(line), flush=True)


if __name__ == "__main__":
    main()
