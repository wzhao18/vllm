# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Debug: topk_v2 vs topk_merge on one test_k31_dcp_exchange case (torchrun, 4 GPUs)."""

import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../tests/models/kimi_k3"))
import test_k31_dcp_exchange as t  # noqa: E402

from vllm.models.kimi_k3.nvidia.dsa_kernels.dcp_exchange.ops import (  # noqa: E402
    DcpExchange,
    local_len_of,
)


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    ex = DcpExchange(dist.group.WORLD, dev, cap=1536, cap1=256, one_round_window=256,
                     interleave=int(os.environ.get("IL", "1")))
    W = ex.W
    case, B, q, ctx, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), \
        int(sys.argv[5])
    K = t.K
    T = B * q
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    ctxs = torch.randint(max(1, ctx // 4), ctx + 1, (B,), generator=gen, device=dev)
    if case == "short":
        ctxs = torch.randint(1, K + 1, (B,), generator=gen, device=dev)
    ctxs[0] = ctx
    il = int(os.environ.get("IL", "1"))
    case = "normal" if case == "short" else case
    G = (ctxs[:, None] - q + 1 + torch.arange(q, device=dev)[None, :]).clamp(min=1).reshape(-1)
    G = G.to(torch.int32)
    t.INTERLEAVE_CUR[0], t.WORLD[0] = il, W
    full = t.make_logits(case, T, G, gen, dev)
    Gmax = full.shape[1]
    Lr = local_len_of(G.to(torch.int64), rank, W, il).to(torch.int32)
    Lmax_all = int(local_len_of(torch.tensor([Gmax]), 0, W, il))
    pos = torch.arange(Lmax_all, device=dev)
    gid = t.local_to_global(pos, rank, W, il)
    stride = (Lmax_all + 7) // 8 * 8 + 8
    local = torch.full((T, stride), float("nan"), dtype=torch.bfloat16, device=dev)
    ok = gid < Gmax
    local[:, :Lmax_all][:, ok] = full[:, gid[ok]]
    out = torch.full((T, K + 64), -7, dtype=torch.int32, device=dev)
    cnt = torch.full((T,), -7, dtype=torch.int32, device=dev)
    ex.topk_merge(local, Lr, G, out, cnt, max_local_len=int(Lr.max()), interleave=il)
    o2 = torch.full((T, K + 64), -7, dtype=torch.int32, device=dev)
    c2 = torch.full((T,), -7, dtype=torch.int32, device=dev)
    dbg = torch.zeros(T * 64, dtype=torch.int64, device=dev)
    ex.topk_v2(local, Lr, G, o2, c2, dbg=dbg, interleave=il)
    torch.cuda.synchronize()
    ex.check()
    keys = t.keys_of(local)
    d = dbg.view(T, 64).cpu()
    for r in range(T):
        a = set(out[r, : int(cnt[r])].tolist())
        b = set(o2[r, : int(c2[r])].tolist())
        same_order = torch.equal(out[r, :K], o2[r, :K])
        if a != b or int(cnt[r]) != int(c2[r]) or not same_order:
            tier = "radix" if d[r, 7] else "tie" if d[r, 6] else "r1"
            only_a = sorted(a - b)[:6]
            only_b = sorted(b - a)[:6]
            print(f"[rank {rank}] row {r} G {int(G[r])} L {int(Lr[r])} cnt {int(cnt[r])}/{int(c2[r])} "
                  f"tier {tier} sets_equal {a == b} order_equal {same_order} "
                  f"merge-only {[(p, int(keys[r, p]), int(gid[p])) for p in only_a]} "
                  f"v2-only {[(p, int(keys[r, p]), int(gid[p])) for p in only_b]}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
