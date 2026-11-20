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
    """Per-CTA globaltimer stamps (us from the earliest kernel entry): min/med/max."""
    base = dsm.STAMP_BASE
    st = sync[base: base + slots * dsm.STAMP_WORDS].view(torch.int64).view(slots, -1)
    st = st[:, :8].double().cpu()
    t0 = st[:, 7].min() if bool((st[:, 7] > 0).all()) else st[:, 0].min()
    rel = (st - t0) / 1e3
    names = ["claim", "mma_end", "lastseg", "accrdy|first", "flags", "finish", "end",
             "entry"]  # fmt: skip
    out = [f"{n} {rel[:, i].min():.2f}/{rel[:, i].median():.2f}/{rel[:, i].max():.2f}"
           for i, n in enumerate(names) if bool((st[:, i] > 0).any())]  # fmt: skip
    print(f"{label} stamps min/med/max us: " + "  ".join(out), flush=True)


def sm_rates(label, sync, slots, dsm, fn, runs):
    """Per-SM stream time (first stage landed -> last stage consumed) over ``runs``
    calls of a NO_EPILOGUE|STAMPS kernel: spread, run-to-run correlation, and the
    span if every SM's range were sized by its measured rate."""
    base = dsm.STAMP_BASE
    per = []
    for _ in range(runs):
        fn()
        torch.cuda.synchronize()
        st = sync[base: base + slots * dsm.STAMP_WORDS].view(torch.int64).view(slots, -1)
        st = st.cpu()
        sm = st[:, 8].long()
        t = (st[:, 1] - st[:, 3]).double() / 1e3
        v = torch.zeros(1024, dtype=torch.float64)
        v[sm] = t
        per.append((sm, v))
    sm = per[0][0]
    m = torch.stack([v[sm] for _, v in per])
    qs = torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0], dtype=torch.float64)
    q = torch.quantile(m[0], qs)
    cs = [float(torch.corrcoef(torch.stack([m[0], m[i]]))[0, 1]) for i in range(1, runs)]
    mean = m.mean(0)
    bal = 1.0 / (1.0 / mean).mean()
    qtxt = "/".join(f"{x:.2f}" for x in q.tolist())
    ctxt = " ".join(f"{c:.2f}" for c in cs)
    span = float(m.max(1).values.mean())
    print(f"{label} per-SM stream us q0/10/50/90/100 {qtxt} corr {ctxt} "
          f"rate-sized {bal:.2f} vs max {span:.2f}", flush=True)
    order = torch.argsort(mean)
    print(f"{label} fastest smids {sm[order[:12]].tolist()} "
          f"slowest {sm[order[-12:]].tolist()}", flush=True)
    srt = sm.tolist()
    by = sorted(zip(srt, mean.tolist()))
    print(f"{label} by smid: " + " ".join(f"{a}:{b:.1f}" for a, b in by), flush=True)
    print(f"{label} by slot: " + " ".join(f"{i}:{a}:{b:.2f}" for i, (a, b) in
                                          enumerate(zip(srt, mean.tolist()))), flush=True)


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
    ap.add_argument("--rates", type=int, default=0)
    ap.add_argument("--calib", action="store_true")
    ap.add_argument("--calib-trace", type=int, default=0)
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

    def tabs(op):  # per-SM ranges: equal, or by the calibrated stream rates (--calib)
        return dsm.slot_tables(op, doms, None, dev)

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
                    tg = tabs(op_gu)
                    fg, ag = dsm.compile_dense_stream(op_gu, wg, x, h, ws, sync, *tg, s0, s1,
                                                      stream(dev))  # fmt: skip
                    td = tabs(op_d)
                    fd, ad = None, None
                    if "down" in args.only:
                        fd, ad = dsm.compile_dense_stream(op_d, wdd, h, y, ws, sync_d,
                                                          *td, s0, s1, stream(dev))  # fmt: skip
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
                    if args.calib and tag == "loc":
                        ks = [(op_gu, sync, *tg)] + ([(op_d, sync_d, *td)] if fd else [])
                        if args.calib_trace and debug & dsm.STAMPS:
                            for rnd in range(args.calib_trace):
                                dsm.calibrate(lambda: (run_gu(), run_d()), ks, doms, rounds=1)
                                for nm, sy, fn in (("gu", sync, run_gu), ("down", sync_d, run_d)):
                                    sm_rates(f"T{T} d{debug} round{rnd} {nm}", sy, s0 + s1, dsm,
                                             fn, 2)  # fmt: skip
                                print(f"T{T} d{debug} round{rnd} gu "
                                      f"{graph_time(run_gu, args.reps):.2f} down "
                                      f"{graph_time(run_d, args.reps):.2f} us", flush=True)
                        else:
                            dsm.calibrate(lambda: (run_gu(), run_d()), ks, doms)
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
                            if args.rates and debug & dsm.NO_EPILOGUE:
                                sm_rates(f"T{T} {tag} d{debug} {nm}", sy, s0 + s1, dsm, fn,
                                         args.rates)  # fmt: skip
                    name = f"ds-{tag}" + (f"-d{debug}" if debug else "")
                    if "gu" in args.only:
                        res[name + " gu"] = graph_time(run_gu, args.reps)
                    if "down" in args.only:
                        res[name + " down"] = graph_time(run_d, args.reps)
            if "plain" in args.only:  # unfused: DenseStream gate_up (2 I outputs) + stock SiTU
                op_p = dsm.DenseStream(T, 2 * I, H, nb0=dsm.split_blocks(2 * I // 128, s0, s1))
                wgu_p = place_rows(wgu, lambda row: int(row >= op_p.nb0 * 128))
                ws_p = torch.zeros(op_p.ws_bytes(s0 + s1) if hasattr(op_p, "ws_bytes") else 1,
                                   dtype=torch.uint8, device=dev)  # fmt: skip
                sync_p = torch.zeros(dsm.SYNC_WORDS, dtype=torch.int32, device=dev)
                gu_p = torch.empty(T, 2 * I, device=dev, dtype=torch.bfloat16)
                h_p = torch.empty(T, I, device=dev, dtype=torch.bfloat16)
                tp = tabs(op_p)
                fp, ap = dsm.compile_dense_stream(op_p, wgu_p, x, gu_p, ws_p, sync_p, *tp,
                                                  s0, s1, stream(dev))  # fmt: skip
                c0, c1, ct = cutlass.Int32(s0), cutlass.Int32(s1), cutlass.Int32(T)

                def run_p():
                    fp(*ap, c0, c1, ct, stream(dev))
                    torch.ops._C.situ_and_mul(h_p, gu_p, beta, lin)

                run_p()
                torch.cuda.synchronize()
                if args.calib:
                    dsm.calibrate(run_p, [(op_p, sync_p, *tp)], doms)
                e = ((h_p.float() - h_ref.float()).abs().max() / h_ref.float().abs().max()).item()
                print(f"T{T} plain: h rel {e:.2e}", flush=True)
                res["ds-plain gu+stock situ"] = graph_time(run_p, args.reps)
                res["ds-plain gu only"] = graph_time(
                    lambda: fp(*ap, c0, c1, ct, stream(dev)), args.reps)
            if "sp" in args.only:  # the production SpDenseMlp (fused <= 32 rows, unfused above)
                import types

                from vllm.models.kimi_k3.nvidia.specialized_decode.runtime.sp_dense import (
                    SpDenseMlp,
                )

                class _Situ:
                    def __init__(self):
                        self.beta, self.linear_beta = beta, lin

                    def __call__(self, v):
                        o = v.new_empty(v.shape[0], v.shape[1] // 2)
                        torch.ops._C.situ_and_mul(o, v, beta, lin)
                        return o

                lin_ = lambda w: types.SimpleNamespace(weight=torch.nn.Parameter(w.clone(), False),
                                                       bias=None, tp_size=1)  # fmt: skip
                mlp = types.SimpleNamespace(gate_up_proj=lin_(wgu), down_proj=lin_(wd),
                                            act_fn=_Situ(), shard_sequence_parallel=False,
                                            gemm_rs_ar=None)  # fmt: skip
                dense = SpDenseMlp(mlp, (T,), dev, 0)
                ysp = dense.run(x)
                torch.cuda.synchronize()
                e = ((ysp.float() - y_ref.float()).abs().max() / y_ref.float().abs().max()).item()
                print(f"T{T} sp: y rel {e:.2e}", flush=True)
                res["sp dense run"] = graph_time(lambda: dense.run(x), args.reps)
                del dense, mlp
                torch.cuda.empty_cache()
            if not args.no_stock:
                res["stock gu+situ+down"] = graph_time(stock, args.reps)
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
