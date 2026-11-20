# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OprojPushMx (MXFP8 SP o-projection fused with the reduce-scatter push) on one GPU:
the four row owners' receive slots are local buffers, so this measures the GEMM and
its push epilogue without NVLink. Correctness against torch._scaled_mm on the same
MXFP8 buffers (Lamport-sanitized), then CUDA-graph timing against scaled_mm with
rotating cold weight copies.

  python bench_kimi_k3_sdec_oproj_push_mx.py --rows 128,256,384 --variants s6i1x1,s4i2x1
"""

import argparse
import re

import torch

WIDTH, K, TP, CAP = 7168, 3072, 4, 128


def mx(x):
    from vllm.models.kimi_k3.nvidia.ops import mxfp8_act

    buf = mxfp8_act.Mxfp8Buffer(x.shape[0], x.shape[1], x.device)
    mxfp8_act.quantize(x, out=buf)
    return buf


def graph_time(fn, reps=20, warm=3):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warm):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(7):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1000 / reps)
    return sorted(ts)[len(ts) // 2]


def sanitize(t):
    w = t.contiguous().view(torch.int32).clone()
    w[w == -0x80000000] = 0
    return w.view(torch.bfloat16).view(t.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="128,256,384")
    ap.add_argument("--variants", default="s4i1x1")
    ap.add_argument("--copies", type=int, default=12, help="weight copies (> L2)")
    ap.add_argument("--rank", type=int, default=1)
    ap.add_argument("--bf16", action="store_true", help="OprojPushBf16 vs torch.mm")
    args = ap.parse_args()
    if args.bf16:
        return main_bf16(args)
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_ptr

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        stream,
        to_cute,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.oproj.push_mx import (
        OprojPushMx,
    )

    dev = torch.device("cuda", 0)
    torch.manual_seed(0)
    ws = [mx(torch.randn(WIDTH, K, device=dev, dtype=torch.bfloat16) * 0.05)
          for _ in range(args.copies)]  # fmt: skip
    g8 = cute.AddressSpace.gmem
    f8, e8 = cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU
    # four owners' recv buffers [TP senders, CAP rows, WIDTH], Lamport-armed
    recv = torch.empty(TP, TP, CAP, WIDTH, device=dev, dtype=torch.bfloat16)
    peers = torch.tensor([recv[o].data_ptr() for o in range(TP)], dtype=torch.int64,
                         device=dev)  # fmt: skip
    peers_c = to_cute(peers)
    rank = args.rank
    print(f"VR200 {torch.cuda.get_device_name(0)} sms "
          f"{torch.cuda.get_device_properties(0).multi_processor_count}", flush=True)  # fmt: skip
    for rows in [int(r) for r in args.rows.split(",")]:
        n = rows // TP
        x = mx(torch.randn(rows, K, device=dev, dtype=torch.bfloat16))
        ref = torch._scaled_mm(
            x.data, ws[0].data.t(), scale_a=x.scale.view(torch.float8_e8m0fnu),
            scale_b=ws[0].scale.view(torch.float8_e8m0fnu), out_dtype=torch.bfloat16,
        )  # fmt: skip
        outs = [torch.empty(rows, WIDTH, device=dev, dtype=torch.bfloat16) for _ in ws]

        def stock(i=[0]):
            j = i[0] % len(ws)
            i[0] += 1
            torch._scaled_mm(
                x.data, ws[j].data.t(), scale_a=x.scale.view(torch.float8_e8m0fnu),
                scale_b=ws[j].scale.view(torch.float8_e8m0fnu), out_dtype=torch.bfloat16,
                out=outs[j],
            )  # fmt: skip

        t_stock = graph_time(stock)
        print(f"R{rows} scaled_mm {t_stock:.2f} us/call ({len(ws)} weight copies)",
              flush=True)  # fmt: skip
        for var in args.variants.split(","):
            m = re.fullmatch(r"s(\d+)i(\d+)x(\d+)(t\d+)?", var)
            stages, iw, ix = (int(g) for g in m.groups()[:3])
            kt = int(m.group(4)[1:]) if m.group(4) else 256
            try:
                kern = OprojPushMx(K, rows, TP, CAP, kt=kt, stages=stages, iw=iw, ix=ix)

                def ptrs(w):
                    return (
                        make_ptr(f8, w.data.data_ptr(), g8, assumed_align=16),
                        make_ptr(e8, w.scale.data_ptr(), g8, assumed_align=16),
                        make_ptr(f8, x.data.data_ptr(), g8, assumed_align=16),
                        make_ptr(e8, x.scale.data_ptr(), g8, assumed_align=16),
                    )

                ri = cutlass.Int32(rows)
                fn = cute.compile(kern, *ptrs(ws[0]), peers_c, cutlass.Int32(rank), ri,
                                  stream(dev))  # fmt: skip
                recv.view(torch.int32).fill_(-0x80000000)
                fn(*ptrs(ws[0]), peers_c, cutlass.Int32(rank), ri, stream(dev))
                torch.cuda.synchronize()
                got = torch.stack([recv[o, rank, :n] for o in range(TP)]).view(rows, WIDTH)
                want = sanitize(ref)
                same = (got.view(torch.int16) == want.view(torch.int16)).float().mean().item()
                err = ((got.float() - ref.float()).abs().max()
                       / ref.float().abs().max()).item()  # fmt: skip
                armed = (recv.view(torch.int32) == -0x80000000).sum().item()
                expect_armed = TP * TP * CAP * WIDTH // 2 - rows * WIDTH // 2
                print(f"  {var}: bitwise-equal {same * 100:.3f}% max rel err {err:.2e} "
                      f"untouched words {armed} (expect >= {expect_armed - 64})",
                      flush=True)  # fmt: skip
                cargs = [ptrs(w) for w in ws]

                def mine(i=[0]):
                    j = i[0] % len(ws)
                    i[0] += 1
                    fn(*cargs[j], peers_c, cutlass.Int32(rank), ri, stream(dev))

                t = graph_time(mine)
                print(f"  {var}: {t:.2f} us/call (scaled_mm {t_stock:.2f})", flush=True)
            except Exception as e:  # report and go on with the next variant
                print(f"  {var}: FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)


def main_bf16(args):
    """OprojPushBf16 (the BF16 DSA o-projection) vs torch.mm (stock's cuBLAS)."""
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_ptr

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        stream,
        to_cute,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.oproj.push_mx import (
        OprojPushBf16,
    )

    dev = torch.device("cuda", 0)
    torch.manual_seed(0)
    ws = [torch.randn(WIDTH, K, device=dev, dtype=torch.bfloat16) * 0.05
          for _ in range(max(1, args.copies // 2))]  # fmt: skip
    g = cute.AddressSpace.gmem
    bf = cutlass.BFloat16
    recv = torch.empty(TP, TP, CAP, WIDTH, device=dev, dtype=torch.bfloat16)
    peers = torch.tensor([recv[o].data_ptr() for o in range(TP)], dtype=torch.int64,
                         device=dev)  # fmt: skip
    peers_c = to_cute(peers)
    rank = args.rank
    for rows in [int(r) for r in args.rows.split(",")]:
        n = rows // TP
        x = torch.randn(rows, K, device=dev, dtype=torch.bfloat16)
        ref = torch.mm(x, ws[0].t())
        outs = [torch.empty(rows, WIDTH, device=dev, dtype=torch.bfloat16) for _ in ws]

        def stock(i=[0]):
            j = i[0] % len(ws)
            i[0] += 1
            torch.mm(x, ws[j].t(), out=outs[j])

        t_stock = graph_time(stock)
        print(f"BF16 R{rows} torch.mm {t_stock:.2f} us/call ({len(ws)} weight copies)",
              flush=True)  # fmt: skip
        for var in args.variants.split(","):
            m = re.fullmatch(r"s(\d+)i(\d+)x(\d+)(t\d+)?", var)
            stages, iw, ix = (int(g) for g in m.groups()[:3])
            kt = int(m.group(4)[1:]) if m.group(4) else 128
            try:
                kern = OprojPushBf16(K, rows, TP, CAP, kt=kt, stages=stages, iw=iw, ix=ix)
                ri = cutlass.Int32(rows)

                def ptrs(w):
                    return (make_ptr(bf, w.data_ptr(), g, assumed_align=16),
                            make_ptr(bf, x.data_ptr(), g, assumed_align=16))  # fmt: skip

                fn = cute.compile(kern, *ptrs(ws[0]), peers_c, cutlass.Int32(rank), ri,
                                  stream(dev))  # fmt: skip
                recv.view(torch.int32).fill_(-0x80000000)
                fn(*ptrs(ws[0]), peers_c, cutlass.Int32(rank), ri, stream(dev))
                torch.cuda.synchronize()
                got = torch.stack([recv[o, rank, :n] for o in range(TP)]).view(rows, WIDTH)
                want = sanitize(ref)
                same = (got.view(torch.int16) == want.view(torch.int16)).float().mean().item()
                err = ((got.float() - ref.float()).abs().max()
                       / ref.float().abs().max()).item()  # fmt: skip
                print(f"  bf16 {var}: bitwise-equal {same * 100:.3f}% max rel err "
                      f"{err:.2e}", flush=True)  # fmt: skip
                cargs = [ptrs(w) for w in ws]

                def mine(i=[0]):
                    j = i[0] % len(ws)
                    i[0] += 1
                    fn(*cargs[j], peers_c, cutlass.Int32(rank), ri, stream(dev))

                t = graph_time(mine)
                print(f"  bf16 {var}: {t:.2f} us/call (torch.mm {t_stock:.2f})", flush=True)
            except Exception as e:  # report and go on with the next variant
                print(f"  bf16 {var}: FAILED {type(e).__name__}: {str(e)[:300]}",
                      flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
