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
p.add_argument("--depth", type=int, default=2)
p.add_argument("--probes", default="0")
p.add_argument("--vec", type=int, default=8)
p.add_argument("--groups", type=int, default=4)
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
st = attn_res_lp._states[blocks.data_ptr()]
st.rstd.uniform_(0.5, 1.5)
for b in range(NB):
    st.logits[b] = torch.randn(2 * plan.calls_after(b), T, device="cuda")
prefix = torch.randn(T, H, device="cuda").bfloat16()
if args.dielocal:  # the model's prefix lives in the die-local prefix buffer
    pb = attn_res_dielocal.prefix_out(prefix, blocks)
    pb.copy_(prefix)
    prefix = pb
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
                         _stamps=stamps if probe & 112 else None)
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
              "logits->prefix landed, ->delta+u, ->stats/stores, ->stats barrier, "
              "->blocks done, ->epilogue end:",
              [round(x) for x in d.median(0).values.tolist()],
              "token period", round(tot.median().item()))

    if probe & 64 and i == args.iters - 1:
        cta = stamps.view(-1, 4)[:1024].cpu().double()
        t0 = cta[cta[:, 0] != 0][:, 0].min() if probe & 32 else None
        gs = stamps[4096:4096 + 1024 * args.groups * 4].view(-1, 4).cpu().double()
        gs = gs[gs[:, 3] != 0]
        for d in (0, 1):
            x = gs[gs[:, 0] == d]
            if len(x) == 0:
                continue
            ex = x[:, 3] - (t0 if t0 is not None else gs[:, 3].min())
            qs = torch.quantile(ex, torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0], dtype=torch.float64))
            print(f"die {d}: groups {len(x)} own {int(x[:, 1].sum())} other {int(x[:, 2].sum())}"
                  f" tok/group {x[:, 1:3].sum(1).mean():.2f} exit us q0/10/50/90/100"
                  f" {[round(v / 1e3, 1) for v in qs.tolist()]}"
                  f" exit(other>0) {round(ex[x[:, 2] > 0].mean().item() / 1e3, 1) if (x[:, 2] > 0).any() else '-'}")

    if probe & 32:
        st_ = stamps.view(-1, 4)[:1024].cpu().double()
        st_ = st_[st_[:, 0] != 0]
        t0 = st_[:, 0].min()
        q = lambda x: [round(v / 1e3, 1) for v in torch.quantile(x, torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)).tolist()]
        print("cta us (min/med/max): entry", q(st_[:, 0] - t0), "loop start", q(st_[:, 1] - t0),
              "group0 done", q(st_[:, 2] - t0), "cta done", q(st_[:, 3] - t0))
