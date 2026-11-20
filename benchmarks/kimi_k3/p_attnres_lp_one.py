# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One attn_res_lp call (for ncu / quick timing): --nb blocks, delta, BF16 +
MXFP8 output (the post-attention MoE case) or --variant q (MXFP8 only)."""
import argparse

import torch

H = 7168

p = argparse.ArgumentParser()
p.add_argument("--rows", type=int, default=16384)
p.add_argument("--nb", type=int, default=5)
p.add_argument("--variant", default="bq")
p.add_argument("--iters", type=int, default=3)
p.add_argument("--depth", type=int, default=3)
p.add_argument("--probes", default="0")
p.add_argument("--vec", type=int, default=8)
p.add_argument("--groups", type=int, default=3)
p.add_argument("--dielocal", type=int, default=0)
args = p.parse_args()
from vllm.models.kimi_k3.nvidia.specialized_prefill import attn_res_lp

T, NB = args.rows, 8
attn_res_lp.RING_DEPTH = args.depth
ncalls = 187
w = [((1 + 0.1 * torch.randn(H, device="cuda")).bfloat16(),
      (torch.randn(H, device="cuda") * H**-0.5).bfloat16()) for _ in range(ncalls)]
plan = attn_res_lp.Plan(w, [1e-6] * ncalls, 12, NB)
blocks = torch.randn(T, NB, H, device="cuda").bfloat16()
if args.dielocal:
    from vllm.models.kimi_k3.nvidia.specialized_prefill import attn_res_dielocal
    dl = attn_res_dielocal.block_residual(blocks[:, 0], NB)
    assert dl is not None
    dl.copy_(blocks)
    blocks = dl
attn_res_lp.begin(plan, blocks)
st = attn_res_lp._state[0]
st.rstd.uniform_(0.5, 1.5)
for b in range(NB):
    st.logits[b] = torch.randn(2 * plan.calls_after(b), T, device="cuda")
prefix = torch.randn(T, H, device="cuda").bfloat16()
delta = torch.randn(T, H, device="cuda").bfloat16()
onw = (1 + 0.1 * torch.randn(H, device="cuda")).bfloat16()
call = 24 * (args.nb - 1) + 5
ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
stamps = torch.zeros(1024 * 16 * 8, dtype=torch.int64, device="cuda")
for probe in [int(x) for x in args.probes.split(",")]:
  for i in range(args.iters):
    ev[0].record()
    attn_res_lp.attn_res(prefix, delta, blocks, call, args.nb, -1, onw, 1e-6,
                         args.variant == "bq", True, _probe=probe, _vec=args.vec, _groups=args.groups,
                         _stamps=stamps if probe & 16 else None)
    ev[1].record()
    torch.cuda.synchronize()
    us = ev[0].elapsed_time(ev[1]) * 1e3
    rows = args.nb + 2 + 1 + (1 if args.variant == "bq" else 0) + 0.5
    if i == args.iters - 1:
        print(f"g{args.groups} v{args.vec} nb{args.nb} {args.variant} d{args.depth} dl{args.dielocal} probe{probe}: {us:.1f} us  model-bytes {rows * T * H * 2 / us / 1e6:.2f} TB/s  reads {(args.nb + 2) * T * H * 2 / us / 1e6:.2f} TB/s")

    if probe & 16:
        st_ = stamps.view(-1, 16, 8).cpu().double()
        st_ = st_[st_[:, 0, 0] != 0]
        d = (st_[:, 2:15, 1:7] - st_[:, 2:15, 0:6]).reshape(-1, 6)
        tot = (st_[:, 3:15, 0] - st_[:, 2:14, 0]).reshape(-1)
        print("stamps (cycles, median over CTAs x tokens 2..14): "
              "logits->prefix landed, ->delta+u, ->stats/stores, ->blocks done, "
              "->group barrier, ->epilogue end:",
              [round(x) for x in d.median(0).values.tolist()],
              "token period", round(tot.median().item()))
