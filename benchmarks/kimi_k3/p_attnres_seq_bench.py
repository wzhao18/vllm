# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P-ATTNRES: the full 93-layer prefill AttnRes call sequence of one DEP16 rank
(F16K rows) with the deployed per-layer arguments (num_blocks grows with depth,
block writes every 12 layers, MXFP8 pre on KDA layers, BF16+MXFP8 post on MoE
layers, native kernel on DSA pre / dense post / final). CUPTI kernel times per
call, bytes moved per call and the effective TB/s; the sum is comparable with
the p16 profile's ATTNRES family (187 calls)."""

import argparse
import json
import math
import statistics

import torch
from torch.profiler import ProfilerActivity, profile

H = 7168
EPS = 1e-6
NL = 93
BS = 12


def build_calls(nl: int):
    calls = []
    nb_total = math.ceil(nl / BS)
    for l in range(nl):
        write = l % BS == 0
        widx = l // BS if write else -1
        prev = math.ceil(l / BS)
        dsa = l % 4 == 0
        calls.append(dict(l=l, kind="pre", impl="native" if dsa else "mx_q",
                          nb=prev, widx=widx, delta=l > 0, out_norm=True))
        calls.append(dict(l=l, kind="post", impl="native" if l == 0 else "mx_bq",
                          nb=prev + int(write), widx=-1, delta=not write,
                          out_norm=True))
    calls.append(dict(l=nl, kind="final", impl="native", nb=nb_total, widx=-1,
                      delta=True, out_norm=False))
    return calls, nb_total


def call_bytes(c, T):
    row = T * H * 2
    rd = (c["nb"] + 1 + int(c["delta"])) * row
    wr = 0
    if c["delta"] or c["widx"] >= 0:
        wr += row  # prefix (+delta) write-back
    if c["widx"] >= 0:
        wr += row
    if c["impl"] in ("native", "mx_bq"):
        wr += row
    if c["impl"] in ("mx_q", "mx_bq"):
        wr += T * H + T * H // 32
    return rd, wr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=16384)
    p.add_argument("--layers", type=int, default=NL)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--depth", type=int, default=0,
                   help="0 = production kernels; 2/3 = the previous one-token-"
                        "per-SM kernel at that ring depth + native calls")
    p.add_argument("--dielocal", type=int, default=0,
                   help="1 = die-local block buffer (production prefill)")
    args = p.parse_args()
    from vllm import _custom_ops as ops
    from vllm.models.kimi_k3.nvidia.specialized_prefill import attn_res_dielocal
    from vllm.models.kimi_k3.nvidia.specialized_prefill.attn_res_mxfp8 import (
        attn_res_bf16,
        attn_res_mxfp8,
    )

    T = args.rows
    calls, nbt = build_calls(args.layers)
    torch.manual_seed(0)
    blocks = torch.randn(T, nbt, H, device="cuda").bfloat16()
    if args.dielocal:
        dl = attn_res_dielocal.block_residual(blocks[:, 0], nbt)
        assert dl is not None, "single-die device"
        dl.copy_(blocks)
        blocks = dl
    prefix = torch.randn(T, H, device="cuda").bfloat16()
    delta = torch.randn(T, H, device="cuda").bfloat16()
    w = (1 + 0.1 * torch.randn(H, device="cuda")).bfloat16()
    qk = (torch.randn(H, device="cuda") * H**-0.5).bfloat16()
    depth = args.depth or None

    attn_out = torch.randn(T, H, device="cuda").bfloat16()
    cur = [prefix]

    def run_one(c):
        # As the model: a write layer's post AttnRes takes the attention
        # output as its prefix (no delta); with --dielocal the kernels move the
        # prefix into the die-local prefix buffer, which the model adopts.
        d = delta if c["delta"] else None
        p_in = attn_out if (c["kind"] == "post" and not c["delta"]) else cur[0]
        po = attn_res_dielocal.prefix_out(p_in, blocks) if depth is None else None
        if c["impl"] == "native" and depth is None:
            attn_res_bf16(p_in, d, blocks, w, qk, w if c["out_norm"] else None,
                          c["nb"], c["widx"], EPS, EPS, prefix_out=po)
        elif c["impl"] == "native":
            ops.kimi_k3_attn_res(p_in, d, blocks, w, qk,
                                 w if c["out_norm"] else None, c["nb"],
                                 c["widx"], EPS, EPS)
        else:
            attn_res_mxfp8(p_in, d, blocks, w, qk, w, c["nb"], c["widx"],
                           EPS, EPS, bf16_out=c["impl"] == "mx_bq",
                           chunk_depth=depth, prefix_out=po)
        cur[0] = p_in if po is None else po
        if c["kind"] == "final":
            cur[0] = prefix

    def seq():
        for c in calls:
            run_one(c)

    for _ in range(2):
        seq()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(args.iters):
            seq()
            torch.cuda.synchronize()
    evs = [e for e in prof.events()
           if e.device_type == torch.autograd.DeviceType.CUDA
           and "Memset" not in e.name and "Memcpy" not in e.name]
    evs.sort(key=lambda e: e.time_range.start)
    n = len(calls)
    assert len(evs) == n * args.iters, (len(evs), n, args.iters)
    per = [statistics.median(evs[i * n + j].time_range.elapsed_us()
                             for i in range(args.iters)) for j in range(n)]
    groups = {}
    tot_us = tot_b = 0.0
    for c, us in zip(calls, per):
        rd, wr = call_bytes(c, T)
        key = f"{c['kind']}.{c['impl']}.nb{c['nb']}.d{int(c['delta'])}.w{int(c['widx'] >= 0)}"
        g = groups.setdefault(key, [0, 0.0, rd + wr])
        g[0] += 1
        g[1] += us
        tot_us += us
        tot_b += rd + wr
    for k, (cnt, us, b) in sorted(groups.items()):
        print(json.dumps({"call": k, "n": cnt, "us": round(us / cnt, 1),
                          "MB": round(b / 1e6, 1),
                          "TBps": round(b / (us / cnt) / 1e6, 2)}))
    print(json.dumps({"rows": T, "layers": args.layers, "calls": n,
                      "depth": args.depth, "dielocal": args.dielocal, "total_ms": round(tot_us / 1e3, 3),
                      "GB": round(tot_b / 1e9, 2),
                      "TBps": round(tot_b / tot_us / 1e6, 2),
                      "ms_at_13.9TBps": round(tot_b / 13.9e12 * 1e3, 3)}))


if __name__ == "__main__":
    main()
