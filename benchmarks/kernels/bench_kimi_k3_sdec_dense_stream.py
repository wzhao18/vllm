# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DenseStream microbench: the layer-0 dense MLP of the SP decode shard (gate_up
2 I x H with SiTU, down H x I; H 7168, I 33792, BF16) at T tokens per GPU.

Arms (us/call, CUDA graph of back-to-back calls, weights >> L2):
  stock      F.linear (cuBLAS) + SituAndMul + F.linear, default (striped) weights
  ds-str     DenseStream gate_up+SiTU, down on the default (striped) weights
  ds-loc     DenseStream on die-local weights (block b of die b >= nb / 2)
  skel       ds-loc with debug bits (--debug: 1 no MMA, 2 no epilogue)
Correctness: DenseStream h / y vs the stock path (BF16 rounding of a reordered FP32
sum), and run-to-run bitwise determinism.
"""

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
    """A copy of the contiguous 2-D ``w`` whose 2 MiB chunks are homed on the die of
    their middle row (``die_of_row(row) -> 0 | 1``)."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common import dies

    row_bytes = w.shape[1] * w.element_size()
    nbytes = w.numel() * w.element_size()
    chunks = -(-nbytes // dies.CHUNK)
    domains = []
    for k in range(chunks):
        mid = min(w.shape[0] - 1, (k * dies.CHUNK + dies.CHUNK // 2) // row_bytes)
        domains.append(die_of_row(mid))
    raw = dies.localized_empty(nbytes, domains, w.device)
    out = raw.view(w.dtype).view(w.shape)
    out.copy_(w)
    return out


def print_stamps(label, sync, slots, dsm):
    """Per-CTA globaltimer stamps (us from the earliest start): quantiles."""
    base = 16 + dsm.MAX_SLOTS
    st = sync[base: base + slots * dsm.STAMP_WORDS].view(torch.int64).view(slots, 8)
    st = st.double().cpu()
    t0 = st[:, 0].min()
    rel = (st - t0) / 1e3
    names = ["start", "mma_end", "lastseg", "acc_rdy", "flags", "finish", "end"]
    out = []
    for i, n in enumerate(names):
        col = rel[:, i]
        out.append(f"{n} {col.median():.2f}/{col.max():.2f}")
    print(f"{label} stamps med/max us: " + "  ".join(out), flush=True)
    d = rel[:, 6] - rel[:, 1]
    print(f"{label} end-mma_end med {d.median():.2f} max {d.max():.2f}; "
          f"acc_rdy-lastseg med {(rel[:, 3] - rel[:, 2]).median():.2f}; "
          f"flags-acc_rdy med {(rel[:, 4] - rel[:, 3]).median():.2f} max "
          f"{(rel[:, 4] - rel[:, 3]).max():.2f}; finish-flags med "
          f"{(rel[:, 5] - rel[:, 4]).median():.2f}", flush=True)  # fmt: skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="+", default=[32, 64, 96])
    ap.add_argument("--h", type=int, default=7168)
    ap.add_argument("--i", type=int, default=33792)
    ap.add_argument("--beta", type=float, default=4.0)
    ap.add_argument("--linear-beta", type=float, default=25.0)
    ap.add_argument("--debug", type=int, nargs="*", default=[])
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--only", default="gu,down")
    ap.add_argument("--no-stock", action="store_true")
    args = ap.parse_args()

    import cutlass

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common import dies
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import stream
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm import dense_stream as dsm

    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    torch.manual_seed(0)
    H, I = args.h, args.i
    wgu = torch.randn(2 * I, H, device=dev, dtype=torch.bfloat16) * 0.02
    wd = torch.randn(H, I, device=dev, dtype=torch.bfloat16) * 0.02
    doms = dies.sm_domains(dev.index)
    s0, s1 = doms.count(0), doms.count(1)
    table = torch.zeros(1024, dtype=torch.int32, device=dev)
    table[: len(doms)] = torch.tensor(doms, dtype=torch.int32, device=dev)
    print(f"SMs {len(doms)}: die0 {s0} die1 {s1}", flush=True)
    nb0_gu = dsm.split_blocks(I // 128, s0, s1)
    nb0_d = dsm.split_blocks(H // 128, s0, s1)
    print(f"die-0 blocks: gate_up {nb0_gu} of {I // 128}, down {nb0_d} of {H // 128}",
          flush=True)  # fmt: skip
    wgu_loc = place_rows(
        wgu, lambda row: int((row % I) >= nb0_gu * 128)
    )  # gate and up halves split alike
    wd_loc = place_rows(wd, lambda row: int(row >= nb0_d * 128))
    sync = torch.zeros(dsm.SYNC_WORDS, dtype=torch.int32, device=dev)
    gbytes = (wgu.numel() + wd.numel()) * 2
    print(f"weights gate_up {wgu.numel() * 2 / 1e6:.1f} MB down {wd.numel() * 2 / 1e6:.1f} MB",
          flush=True)  # fmt: skip

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
                op_gu = dsm.DenseStream(T, I, H, situ_mode=True, beta=beta,
                                        linear_beta=lin, debug=debug, nb0=nb0_gu)  # fmt: skip
                op_d = dsm.DenseStream(T, H, I, debug=debug, nb0=nb0_d)
                slots = s0 + s1
                ws = torch.zeros(max(op_gu.ws_bytes(slots), op_d.ws_bytes(slots)),
                                 dtype=torch.uint8, device=dev)  # fmt: skip
                if not debug:
                    print(f"T{T}: gu stages {op_gu.stages} staged {op_gu.nstage_p} x "
                          f"{op_gu.part} B; down stages {op_d.stages} staged "
                          f"{op_d.nstage_p} x {op_d.part} B", flush=True)  # fmt: skip
                h = torch.empty(T, I, device=dev, dtype=torch.bfloat16)
                y = torch.empty(T, H, device=dev, dtype=torch.bfloat16)
                sync_d = torch.zeros_like(sync)
                for tag, wg, wdd in (("str", wgu, wd), ("loc", wgu_loc, wd_loc)):
                    if debug and tag == "str":
                        continue
                    if debug & dsm.CONTIG_W:  # skeleton: each weight box 16 KB contiguous
                        wg, wdd = wg.view(-1, 64), wdd.view(-1, 64)
                    fg, ag = dsm.compile_dense_stream(op_gu, wg, x, h, ws, sync, table, s0, s1,
                                                      stream(dev))  # fmt: skip
                    fd, ad = None, None
                    if "down" in args.only:
                        fd, ad = dsm.compile_dense_stream(op_d, wdd, h, y, ws, sync_d,
                                                          table, s0, s1, stream(dev))  # fmt: skip
                    s0c, s1c, tc = cutlass.Int32(s0), cutlass.Int32(s1), cutlass.Int32(T)

                    def run_gu(fg=fg, ag=ag):
                        fg(*ag, s0c, s1c, tc, stream(dev))

                    def run_d(fd=fd, ad=ad):
                        if fd is not None:
                            fd(*ad, s0c, s1c, tc, stream(dev))

                    t0 = time.time()
                    run_gu()
                    run_d()
                    torch.cuda.synchronize()
                    print(f"T{T} {tag} d{debug}: compiled+ran in {time.time() - t0:.1f} s",
                          flush=True)  # fmt: skip
                    if not debug:
                        h1, y1 = h.clone(), y.clone()
                        run_gu()
                        run_d()
                        torch.cuda.synchronize()
                        det = bool(torch.equal(h, h1) and torch.equal(y, y1))
                        eh = ((h.float() - h_ref.float()).abs().max() /
                              h_ref.float().abs().max()).item()  # fmt: skip
                        ey = ((y.float() - y_ref.float()).abs().max() /
                              y_ref.float().abs().max()).item()  # fmt: skip
                        neq = (h != h_ref).float().mean().item()
                        print(f"T{T} {tag}: h rel {eh:.2e} (bits differ {neq:.4f}) y rel "
                              f"{ey:.2e} deterministic {det}", flush=True)  # fmt: skip
                    if debug & dsm.STAMPS:
                        for nm, sy, fn in (("gu", sync, run_gu), ("down", sync_d, run_d)):
                            fn()
                            torch.cuda.synchronize()
                            print_stamps(f"T{T} {tag} d{debug} {nm}", sy, s0 + s1, dsm)
                    name = f"ds-{tag}" + (f"-d{debug}" if debug else "")
                    if "gu" in args.only:
                        res[name + " gu"] = graph_time(run_gu, args.reps)
                    if "down" in args.only:
                        res[name + " down"] = graph_time(run_d, args.reps)
            if not args.no_stock:
                if "gu" in args.only:
                    res["stock gu+situ"] = graph_time(
                        lambda: (torch.mm(x, wgu.t(), out=gu),
                                 torch.ops._C.situ_and_mul(h_ref, gu, beta, lin)),
                        args.reps,
                    )  # fmt: skip
                if "down" in args.only:
                    res["stock down"] = graph_time(lambda: F.linear(h_ref, wd), args.reps)
        except Exception:
            traceback.print_exc()
            sys.stdout.flush()
        for k, v in res.items():
            mb = (wgu.numel() if " gu" in k else wd.numel()) * 2 / 1e6
            print(f"T{T} {k:22s} {v:8.2f} us/call  {mb / v:6.2f} TB/s", flush=True)
        print(f"T{T} weights {gbytes / 1e6:.0f} MB: SOL at 18.47 TB/s "
              f"{gbytes / 18.47e6:.1f} us", flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
