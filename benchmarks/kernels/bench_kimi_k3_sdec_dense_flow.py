# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DenseFlow (one-grid die-local dense MLP: gate_up + SiTU + down) vs the stock
cuBLAS gate_up + SituAndMul + cuBLAS down, at the layer-0 dense MLP shape
(H 7168, I 33792) and decode shard token counts. Correctness vs stock (h and y),
determinism, graph-replay time; ``--debug 3`` = loads-only skeleton (no MMA, no
epilogue), ``--debug 4`` adds per-SM stream spans."""

import argparse
import sys
import time
import traceback

import torch
import torch.nn.functional as F


def graph_time(fn, reps=10, iters=5):
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        for _ in range(reps):
            fn()
    torch.cuda.synchronize()
    best = 1e9
    for _ in range(iters):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        best = min(best, e0.elapsed_time(e1) * 1e3 / reps)
    return best


def place_rows(w, die_of_row):
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common import dies

    row_bytes = w.shape[1] * w.element_size()
    nbytes = w.numel() * w.element_size()
    chunks = -(-nbytes // dies.CHUNK)
    domains = [die_of_row(min(w.shape[0] - 1, (k * dies.CHUNK + dies.CHUNK // 2) // row_bytes))
               for k in range(chunks)]  # fmt: skip
    raw = dies.localized_empty(nbytes, domains, w.device)
    out = raw.view(w.dtype).view(w.shape)
    out.copy_(w)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="+", default=[32, 64, 96])
    ap.add_argument("--h", type=int, default=7168)
    ap.add_argument("--i", type=int, default=33792)
    ap.add_argument("--beta", type=float, default=4.0)
    ap.add_argument("--linear-beta", type=float, default=25.0)
    ap.add_argument("--debug", type=int, nargs="*", default=[])
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--no-stock", action="store_true")
    ap.add_argument("--one-die", action="store_true")
    args = ap.parse_args()

    import cutlass

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common import dies
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import stream
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm import dense_flow as dfl
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm import dense_stream as dsm

    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    torch.manual_seed(0)
    H, I = args.h, args.i
    wgu = torch.randn(2 * I, H, device=dev, dtype=torch.bfloat16) * 0.02
    wd = torch.randn(H, I, device=dev, dtype=torch.bfloat16) * 0.02
    doms = tuple(dies.sm_domains(dev.index))
    if args.one_die:
        doms = tuple(0 if d >= 0 else -1 for d in doms)
    s0, s1 = doms.count(0), doms.count(1)
    nbA0 = dsm.split_blocks(I // 64, s0, s1)
    nbB0 = dsm.split_blocks(H // 128, s0, s1)
    print(f"SMs {len(doms)}: die0 {s0} die1 {s1}; die-0 blocks A {nbA0} of {I // 64}, "
          f"B {nbB0} of {H // 128}", flush=True)  # fmt: skip
    if s1:
        wgu_l = place_rows(wgu, lambda row: int((row % I) // 64 >= nbA0))
        wd_l = place_rows(wd, lambda row: int(row // 128 >= nbB0))
    else:
        wgu_l, wd_l = wgu, wd
    gbytes = (wgu.numel() + wd.numel()) * 2
    beta, lin = args.beta, args.linear_beta
    for T in args.rows:
        x = torch.randn(T, H, device=dev, dtype=torch.bfloat16)
        gu = torch.empty(T, 2 * I, device=dev, dtype=torch.bfloat16)
        h_ref = torch.empty(T, I, device=dev, dtype=torch.bfloat16)

        def stock():
            torch.mm(x, wgu.t(), out=gu)
            torch.ops._C.situ_and_mul(h_ref, gu, beta, lin)
            return F.linear(h_ref, wd)

        y_ref = stock()
        torch.cuda.synchronize()
        res = {}
        try:
            for debug in [0] + list(args.debug):
                op = dfl.DenseFlow(T, H, I, beta, lin, nbA0=nbA0, nbB0=nbB0,
                                   two_dies=s1 > 0, debug=debug)  # fmt: skip
                tab = dfl.flow_tables(op, doms, None, dev)
                sync = torch.zeros(dfl.SYNC_WORDS, dtype=torch.int32, device=dev)
                sync_ref = sync
                ws = torch.zeros(op.ws_bytes(s0 + s1), dtype=torch.uint8, device=dev)
                h = torch.zeros(T, I, device=dev, dtype=torch.bfloat16)
                y = torch.zeros(T, H, device=dev, dtype=torch.bfloat16)
                t0 = time.time()
                fn, fa = dfl.compile_dense_flow(op, wgu_l, x, wd_l, h, y, ws, sync, *tab, s0,
                                                s1, stream(dev))  # fmt: skip
                cc = (cutlass.Int32(s0), cutlass.Int32(s1), cutlass.Int32(T))

                def run(fn=fn, fa=fa, cc=cc):
                    fn(*fa, *cc, stream(dev))

                run()
                torch.cuda.synchronize()
                print(f"T{T} d{debug}: stages {op.stages}, compiled+ran in "
                      f"{time.time() - t0:.1f} s", flush=True)  # fmt: skip
                if not debug:
                    h1, y1 = h.clone(), y.clone()
                    run()
                    torch.cuda.synchronize()
                    det = bool(torch.equal(h, h1) and torch.equal(y, y1))
                    eh = ((h.float() - h_ref.float()).abs().max() /
                          h_ref.float().abs().max()).item()  # fmt: skip
                    ey = ((y.float() - y_ref.float()).abs().max() /
                          y_ref.float().abs().max()).item()  # fmt: skip
                    neq = (h != h_ref).float().mean().item()
                    print(f"T{T} flow: h rel {eh:.2e} (bits differ {neq:.4f}) y rel {ey:.2e} "
                          f"deterministic {det}", flush=True)  # fmt: skip
                t = dfl.flow_times(sync, s0 + s1)
                span = (t[:, 1] - t[:, 0]).double() / 1e3
                start = (t[:, 0] - t[:, 0].min()).double() / 1e3
                print(f"T{T} d{debug}: per-SM stream us min/med/max {span.min():.2f}/"
                      f"{span.median():.2f}/{span.max():.2f}; first stage spread "
                      f"{start.max():.2f}; last end {((t[:, 1] - t[:, 0].min()).max() / 1e3):.2f}",
                      flush=True)  # fmt: skip
                if debug & dsm.STAMPS:
                    base = dfl.STAMP_BASE
                    n = s0 + s1
                    stp = sync[base: base + n * dfl.STAMP_WORDS].view(torch.int64).view(n, -1)
                    stp = stp.double().cpu()
                    t0s = stp[:, 7].min()
                    names = {0: "claim", 2: "A_end", 3: "A_epi", 5: "watch_done", 1: "mma_end",
                             6: "end"}  # fmt: skip
                    txt = []
                    for i, nm in names.items():
                        v = (stp[:, i] - t0s) / 1e3
                        txt.append(f"{nm} {v.min():.1f}/{v.median():.1f}/{v.max():.1f}")
                    for i, nm in ((4, "watch_polls_ns"), (9, "mma_wait_B"), (10, "mma_wait_A"),
                                  (11, "A_finish_sum"), (12, "fin_tmem"), (13, "fin_part"),
                                  (14, "fin_situ_store")):
                        v = stp[:, i] / 1e3
                        txt.append(f"{nm} {v.min():.1f}/{v.median():.1f}/{v.max():.1f}")
                    print(f"T{T} d{debug} stamps min/med/max us: " + "  ".join(txt), flush=True)
                res[f"flow d{debug}"] = graph_time(run, args.reps)
        except Exception:
            traceback.print_exc()
            sys.stdout.flush()
        if not args.no_stock:
            res["stock gu+situ+down"] = graph_time(stock, args.reps)
        for k, v in res.items():
            print(f"T{T} {k:22s} {v:8.2f} us/call  {gbytes / 1e6 / v:6.2f} TB/s", flush=True)
        print(f"T{T} weights {gbytes / 1e6:.0f} MB: SOL at 18.47 TB/s "
              f"{gbytes / 18.47e6:.1f} us", flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
