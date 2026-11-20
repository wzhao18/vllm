# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The SP o-projection "pre_push" GEMM (the vendored sm_107 MXFP8 pre GEMM with the
reduce-scatter push epilogue) against the same GEMM writing its local output, on one
GPU: the four row owners' receive slots are local buffers, so this measures the GEMM
and its epilogue without NVLink. Bitwise check of the pushed words against the
sanitized local output, then CUDA-graph timing with rotating cold weight copies.

  python bench_kimi_k3_sdec_oproj_pre_push.py --rows 128,256,384
"""

import argparse

import torch

WIDTH, K, TP = 7168, 3072, 4
SENTINEL = -0x80000000


def mx(x):
    from vllm.models.kimi_k3.nvidia.ops import mxfp8_act

    buf = mxfp8_act.Mxfp8Buffer(x.shape[0], x.shape[1], x.device)
    mxfp8_act.quantize(x, out=buf)
    return buf


def graph_time(fn, reps=16, warm=3):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(warm):
            fn(i)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(reps):
            fn(i)
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(9):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1000 / reps)
    return sorted(ts)[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="128,256,384")
    ap.add_argument("--copies", type=int, default=8)
    ap.add_argument("--ks", action="store_true", help="also OprojKsPush")
    ap.add_argument("--ks-dbg", default="0", help="OprojKsPush probe masks (ks_push.py)")
    ap.add_argument("--no-pre-push", action="store_true")
    args = ap.parse_args()
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.mxfp8 import pre_tactic
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.sm107.prewait import (
        Mxfp8PreGemm,
    )

    dev = torch.device("cuda")
    torch.manual_seed(0)
    w = mx(torch.randn(WIDTH, K, device=dev).bfloat16() * K**-0.5)
    copies = [(w.data.clone(), w.scale.clone()) for _ in range(args.copies)]
    for rows in [int(r) for r in args.rows.split(",")]:
        n = rows // TP
        cap = n
        tactic, prewait, stages = pre_tactic(WIDTH, K, rows)
        x = mx(torch.randn(rows, K, device=dev).bfloat16())
        local = Mxfp8PreGemm(WIDTH, K, rows, tactic, prewait=prewait, max_stages=stages)
        push = Mxfp8PreGemm(
            WIDTH, K, rows, tactic, prewait=prewait, max_stages=stages, push=True
        )
        out = torch.empty(rows, WIDTH, dtype=torch.bfloat16, device=dev)
        recv = torch.empty(TP, cap, WIDTH, dtype=torch.bfloat16, device=dev)
        recv.view(torch.int32).fill_(SENTINEL)
        peers = torch.tensor(
            [recv[o].data_ptr() for o in range(TP)], dtype=torch.int64, device=dev
        )
        # this rank = 0: owner o's slot [0] holds rows o n .. o n + n - 1
        pargs = Mxfp8PreGemm.push_args(peers, 0, n, cap, rows)
        local(x.data, x.scale, w.data, w.scale, out)
        torch.cuda.synchronize()
        words = out.view(torch.int32).clone()
        words[words == SENTINEL] = 0
        t_local = graph_time(
            lambda i: local(x.data, x.scale, *copies[i % len(copies)], out)
        )
        line = f"R{rows} tactic {tactic}: pre {t_local:.2f} us"
        if not args.no_pre_push:
            push(x.data, x.scale, w.data, w.scale, out, pargs)
            torch.cuda.synchronize()
            got = recv.view(torch.int32).reshape(TP * cap, WIDTH // 2)[:rows]
            same = (got == words).float().mean().item()
            t_push = graph_time(
                lambda i: push(x.data, x.scale, *copies[i % len(copies)], out, pargs)
            )
            line += f"  pre_push {t_push:.2f} us/call (bitwise {same * 100:.3f}%)"
        print(line, flush=True)
        if args.ks:
            for dbg in (int(d) for d in args.ks_dbg.split(",")):
                res = ks_push(rows, n, cap, x, copies, recv, peers, words, dbg)
                print(f"R{rows} dbg {dbg} " + res, flush=True)


def ks_push(rows, n, cap, x, copies, recv, peers, words, dbg=0):
    """OprojKsPush (K split 3, block by block) into the same local slots: bits vs the
    pre GEMM's sanitized output and us/call."""
    import cutlass
    import cutlass.cute as cute
    from cutlass import Int32
    from cutlass.cute.runtime import make_ptr

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        stream,
        to_cute,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.oproj import ks_push
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.oproj import sp

    dev = x.data.device
    blocks = -(-rows // ks_push.TB)
    code = sp.compile_ks_push(blocks, K, TP, cap, dev.index, dbg)
    scratch, counters = sp.ks_scratch(dev)
    order = ks_push.block_order(rows, n, 0)
    order = torch.tensor(order + [0] * (ks_push.MAX_BLOCKS - len(order)),
                         dtype=torch.int32, device=dev)  # fmt: skip
    views = (to_cute(peers), to_cute(scratch), to_cute(counters), to_cute(order))
    g, f8, e8 = cute.AddressSpace.gmem, cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU

    def call(i):
        w, ws = copies[i % len(copies)]
        ptrs = tuple(
            make_ptr(t, a.data_ptr(), g, assumed_align=16)
            for t, a in ((f8, w), (e8, ws), (f8, x.data), (e8, x.scale))
        )
        code(*ptrs, *views, Int32(0), Int32(rows), stream(dev))

    recv.view(torch.int32).fill_(SENTINEL)
    call(0)
    torch.cuda.synchronize()
    got = recv.view(torch.int32).reshape(TP * cap, WIDTH // 2)[:rows]
    bits = (got == words).float().mean().item()
    ref = words.view(torch.bfloat16).float()
    err = (got.view(torch.bfloat16).float() - ref).abs().max().item()
    t = graph_time(call)
    line = f"ks_push {t:.2f} us/call (bits {bits * 100:.2f}%, max err {err:.3g})"
    if dbg & 32:  # one more call, then its stamps (us after the earliest CTA start)
        call(0)
        torch.cuda.synchronize()
        st = scratch[-ks_push.STAMP_FLOATS :].view(torch.int64)
        st = st.view(-1, ks_push.STAMPS)[: 2 * 28 * ks_push.KS].double()
        t0 = st[:, 0].min()
        names = ("start", "A", "B0", "B1", "B2", "M0", "M1", "M2", "E0", "E1", "E2",
                 "P0", "P1", "P2", "end", "Xw")  # fmt: skip
        cols = [i for i in range(ks_push.STAMPS) if (st[:, i] > 0).any()]
        med = {names[i]: ((st[:, i][st[:, i] > 0] - t0).median() / 1e3).item() for i in cols}
        line += " | stamps us: " + " ".join(f"{k} {v:.2f}" for k, v in med.items())
    return line


if __name__ == "__main__":
    main()
