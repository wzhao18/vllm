# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which (request, KV head) units GqaWindowKols gets wrong at one batch (vs the FP64
rows of the attention test), with each unit's pieces (CTA, piece index, stages).

  python3 diag_kols.py 4      (run from the vLLM tree)
"""

import os
import sys

import torch

sys.path.insert(0, os.getcwd())

import tests.models.kimi_k3.test_dspark_gqa_attention_int as T  # noqa: E402

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.draft import (  # noqa: E402
    gqa_attention as ga,
)


def pieces(stages, sms, hkv=4):
    pre = [0]
    for s in stages:
        pre.append(pre[-1] + s)
    total = hkv * pre[-1]
    where = {}
    for cta in range(sms):
        s0, s1 = cta * total // sms, (cta + 1) * total // sms
        s, k = s0, 0
        while s < s1:
            r = max(i for i in range(len(stages)) if hkv * pre[i] <= s)
            rel = s - hkv * pre[r]
            h, a = divmod(rel, max(stages[r], 1))
            n = min(stages[r] - a, s1 - s)
            where.setdefault((r, h), []).append((cta, k, a, n, s - s0))
            s, k = s + n, k + 1
    return where


def main():
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    ga.GQA_WINDOW[(4, 3)] = ((0, (1, -3, 6, 0)),)
    device = T._device()
    gen = torch.Generator(device=device).manual_seed(57)
    layers = T.make_layers(4, gen)
    state = T.install(layers, 3, (B,))
    ours, stock, exact, lengths, differ = T.run_case(
        state, layers, "rows", B, 3, gen, seed=5700 + B
    )
    lens = [int(x) for x in (lengths.tolist() if hasattr(lengths, "tolist") else lengths)]
    stages = []
    for ln in lens:
        lo = max(ln - 3 - 2047, 0)
        stages.append(0 if ln <= 0 else ((ln - 1) // 64 - lo // 64 + 2) // 2)
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    where = pieces(stages, sms)
    print(f"B{B} lengths {lens[:8]}... stages {stages[:8]}... cache differ {differ}")
    for i, (o, e) in enumerate(zip(ours, exact)):
        cols = o.shape[-1]
        print(f"layer {i}: out {tuple(o.shape)} exact {tuple(e.shape)}")
        err = (o.double() - e.double()).abs().view(B, -1, cols // 512, 512).amax(dim=(1, 3))
        bad = (err > 0.05 * e.abs().max().item()).nonzero().tolist()
        print(f"layer {i}: {len(bad)} bad units of {B * 4}")
        for r, h in bad[:12]:
            print(f"  unit ({r},{h}) err {err[r, h].item():.3f} pieces (cta, k, a, n, js): "
                  f"{where.get((r, h))}")
        if i == 0:
            good = [u for u in where if u not in {tuple(b) for b in bad}][:4]
            for u in good:
                print(f"  good {u}: {where[u]}")


if __name__ == "__main__":
    main()
