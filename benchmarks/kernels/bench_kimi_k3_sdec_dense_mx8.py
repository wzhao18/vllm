# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DenseMx8 (specialized decode MXFP8 projection) vs the stock scaled_mm path.

Correctness against torch._scaled_mm on the same MXFP8 buffers, then timing in a
CUDA graph: back to back, and behind a predecessor that holds the SMs like the SP
gather (``--pred-us``: 672 128-thread CTAs that trigger PDL at entry and spin).

  python bench_kimi_k3_sdec_dense_mx8.py --rows 384 --n 12448 --k 7168
"""

import argparse
import re
import sys
import time
import traceback
import types

import torch


def mx(x):
    from vllm.models.kimi_k3.nvidia.ops import mxfp8_act

    buf = mxfp8_act.Mxfp8Buffer(x.shape[0], x.shape[1], x.device)
    mxfp8_act.quantize(x, out=buf)
    return buf


def ptrs(w, x, out, part, tick):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_ptr

    g = cute.AddressSpace.gmem
    f8, e8 = cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU
    return (
        make_ptr(f8, w.data.data_ptr(), g, assumed_align=16),
        make_ptr(e8, w.scale.data_ptr(), g, assumed_align=16),
        make_ptr(f8, x.data.data_ptr(), g, assumed_align=16),
        make_ptr(e8, x.scale.data_ptr(), g, assumed_align=16),
        make_ptr(cutlass.BFloat16, out.data_ptr(), g, assumed_align=16),
        make_ptr(cutlass.Float32, part.data_ptr(), g, assumed_align=16),
        make_ptr(cutlass.Int32, tick.data_ptr(), g, assumed_align=16),
    )


def die_copies(w, dev):
    """W (FP8 rows + swizzled scales) as [2] copies, copy d homed on die d (2 MiB
    chunks; a straddling chunk goes to its majority die)."""
    from vllm.models.kimi_k3.nvidia.megamoe_rubin.die_local import chunk_domains
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common import dies

    out = []
    for t in (w.data, w.scale):
        nb = t.numel() * t.element_size()
        raw = dies.localized_empty(2 * nb, chunk_domains(2 * nb, nb), dev)
        two = raw.view(t.dtype).view(2, *t.shape)
        two[0].copy_(t)
        two[1].copy_(t)
        out.append(two)
    return types.SimpleNamespace(data=out[0], scale=out[1])


def make_pred(ctas, dev):
    """A co-residable PDL predecessor: ``ctas`` 128-thread CTAs without shared memory
    that wait for their own predecessor, trigger their dependents and spin ``ns``.
    Behind it a GEMM whose weight prologue runs before ``griddepcontrol.wait``
    overlaps that prologue with the spin (the in-grid op's run-ahead)."""
    import cutlass
    import cutlass.cute as cute

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import stream
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.mla.qvalid import spin_ns

    class Pred:
        @cute.jit
        def __call__(self, ns: cutlass.Int64, s):
            self.kernel(ns).launch(grid=[ctas, 1, 1], block=[128, 1, 1], stream=s,
                                   use_pdl=True)  # fmt: skip

        @cute.kernel
        def kernel(self, ns: cutlass.Int64):
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()
            spin_ns(ns)

    p = Pred()
    fn = cute.compile(p, cutlass.Int64(1000), stream(dev))
    return lambda ns: fn(cutlass.Int64(ns), stream(dev))


def graph_time(fn, reps=50, iters=5):
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        for _ in range(reps):
            fn()
    torch.cuda.synchronize()
    best = 1e9
    for _ in range(iters):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        best = min(best, e0.elapsed_time(e1) * 1e3 / reps)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="+", default=[384])
    ap.add_argument("--n", type=int, default=12448)
    ap.add_argument("--k", type=int, default=7168)
    ap.add_argument("--variants", default="c1s4k2,c0s4k2,c1s5k2,c1s4k1")
    ap.add_argument("--copies", type=int, default=4, help="weight copies (> L2)")
    ap.add_argument("--pred-us", type=float, nargs="*", default=[],
                    help="also time each kernel behind a co-resident PDL spin of these us; "
                         "exposed = chain - spin-only chain")  # fmt: skip
    args = ap.parse_args()
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_ptr  # noqa: F401

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import stream
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.dense_mx8 import (
        DenseMx8,
    )

    dev = torch.device("cuda", 0)
    torch.manual_seed(0)
    n, k = args.n, args.k
    ws = [mx(torch.randn(n, k, device=dev, dtype=torch.bfloat16) * 0.05)
          for _ in range(args.copies)]  # fmt: skip
    print(f"VR200 {torch.cuda.get_device_name(0)} sms "
          f"{torch.cuda.get_device_properties(0).multi_processor_count}", flush=True)  # fmt: skip
    wcopies = None  # die-local double buffers of ws (variant key l1), built once
    for rows in args.rows:
        x = mx(torch.randn(rows, k, device=dev, dtype=torch.bfloat16))
        ref = torch._scaled_mm(
            x.data, ws[0].data.t(), scale_a=x.scale.view(torch.float8_e8m0fnu),
            scale_b=ws[0].scale.view(torch.float8_e8m0fnu), out_dtype=torch.bfloat16,
        )  # fmt: skip
        outs = [torch.empty(rows, n, device=dev, dtype=torch.bfloat16) for _ in ws]

        def stock(i=[0]):
            j = i[0] % len(ws)
            i[0] += 1
            torch._scaled_mm(
                x.data, ws[j].data.t(), scale_a=x.scale.view(torch.float8_e8m0fnu),
                scale_b=ws[j].scale.view(torch.float8_e8m0fnu), out_dtype=torch.bfloat16,
                out=outs[j],
            )  # fmt: skip

        t_stock = graph_time(stock)
        print(f"R{rows} scaled_mm {t_stock:.2f} us/call (graph, {len(ws)} weight copies)",
              flush=True)  # fmt: skip
        pred = make_pred(torch.cuda.get_device_properties(0).multi_processor_count, dev)
        spin = {}

        def behind(name, fn):
            for us in args.pred_us:
                ns = int(us * 1000)
                if us not in spin:
                    spin[us] = graph_time(lambda: pred(ns))
                t_ch = graph_time(lambda: (pred(ns), fn()))
                print(f"  {name} behind spin {us:g} us: exposed {t_ch - spin[us]:.2f} us "
                      f"(chain {t_ch:.2f}, spin-only {spin[us]:.2f})", flush=True)  # fmt: skip

        behind("scaled_mm", stock)
        for var in args.variants.split(","):
            f = dict(re.findall(r"([a-z])(\d+)", var))  # c<col>s<stages>k<ksplit>[m<mc>][d<debug>][t<kt>]
            col, stg, ks = int(f["c"]), int(f["s"]), int(f["k"])
            mc, dbg = int(f.get("m", 1)), int(f.get("d", 0))
            kt = int(f.get("t", 128))
            try:
                wdie = ()
                if int(f.get("l", 0)):  # die-local W: each CTA streams its die's copy
                    from vllm.models.kimi_k3.nvidia.megamoe_rubin.die_local import (
                        die_sm_mask32,
                    )

                    wdie = die_sm_mask32(0)
                    assert wdie, "no trusted SM -> die map"
                    if wcopies is None:
                        wcopies = [die_copies(w, dev) for w in ws]
                g = DenseMx8(n, k, rows, n, ksplit=ks, stages=stg, collector=bool(col), mc=mc,
                             kred=bool(int(f.get("r", 0))), pair=bool(int(f.get("p", 1))),
                             iw=int(f.get("w", 1)), ix=int(f.get("x", 1)),
                             debug=dbg, kt=kt, pre_ns=int(f.get("e", 0)) * 1000,
                             xcopies=int(f.get("y", 1)), wdie=wdie,
                             sym=bool(int(f.get("g", 0))))
                pw, tw = g.scratch_words()
                part = torch.zeros(pw, device=dev, dtype=torch.float32)
                tick = torch.zeros(tw, device=dev, dtype=torch.int32)
                t0 = time.time()
                xv = x
                if int(f.get("y", 1)) > 1:  # private activation copy per row block
                    yc = int(f["y"])
                    xv = types.SimpleNamespace(data=x.data.repeat(yc, 1).contiguous(),
                                               scale=x.scale.reshape(-1).repeat(yc))
                wv = wcopies if wdie else ws
                args0 = ptrs(wv[0], xv, outs[0], part, tick)
                fn = cute.compile(g, *args0, stream(dev))
                print(f"  {var}: compiled in {time.time() - t0:.1f}s grid {g.grid} "
                      f"sfa/sfb cols {g.sfa_cols}/{g.sfb_cols}", flush=True)  # fmt: skip
                all_args = [ptrs(w, xv, o, part, tick) for w, o in zip(wv, outs)]
                fn(*all_args[0], stream(dev))
                torch.cuda.synchronize()
                d = (outs[0].float() - ref.float())
                rel = (d.norm() / ref.float().norm()).item()
                mism = (d.abs() > 1e-2 * ref.float().abs().max()).sum().item()
                bad_rows = (d.abs().amax(1) > 1e-2 * ref.float().abs().max()).nonzero()
                print(f"  {var}: rel {rel:.3e} maxabs {d.abs().max().item():.3e} "
                      f"mismatch {mism} rows {bad_rows[:6].flatten().tolist()}", flush=True)  # fmt: skip
                if mism and not dbg:
                    bc = (d.abs().amax(0) > 1e-2 * ref.float().abs().max()).nonzero().flatten()
                    print(f"     bad cols {bc.numel()} first {bc[:8].tolist()}", flush=True)
                    continue

                def ours(i=[0]):
                    j = i[0] % len(ws)
                    i[0] += 1
                    fn(*all_args[j], stream(dev))

                t = graph_time(ours)
                print(f"  {var}: {t:.2f} us/call  ({t_stock / t:.2f}x scaled_mm)", flush=True)
                if dbg & 16384:  # globaltimer stamps of the last call (DenseMx8 STAMP)
                    base = pw - 16 * g.grid
                    st = part[base:].view(torch.int64).view(g.grid, 8).double().cpu()
                    t0 = st[:, 0].min()
                    rel = (st - t0) / 1e3
                    names = ["entry", "accfull", "publish", "peer", "landed", "end",
                             "xwait", "lastmma"]  # fmt: skip
                    line = " ".join(
                        f"{nm} {rel[:, i].min():.2f}/{rel[:, i].median():.2f}/"
                        f"{rel[:, i].max():.2f}" for i, nm in enumerate(names)
                    )
                    print(f"  {var}: stamps min/med/max us: {line}", flush=True)
                    d = {
                        "acc-xwait": rel[:, 1] - rel[:, 6], "pub": rel[:, 2] - rel[:, 1],
                        "wait": rel[:, 3] - rel[:, 2], "copy": rel[:, 4] - rel[:, 3],
                        "sum+st": rel[:, 5] - rel[:, 4], "epi": rel[:, 5] - rel[:, 1],
                    }  # fmt: skip
                    print(f"  {var}: phases med/max us: " + " ".join(
                        f"{k_} {v_.median():.2f}/{v_.max():.2f}" for k_, v_ in d.items()
                    ), flush=True)  # fmt: skip
                if int(f.get("e", 0)):
                    print(f"  {var}: in-kernel pre-wait spin {f['e']} us: exposed "
                          f"{t - int(f['e']):.2f} us", flush=True)  # fmt: skip
                behind(var, ours)
            except Exception as e:  # report and go on with the next variant
                tb = traceback.format_exc().splitlines()
                print(f"  {var}: FAILED {type(e).__name__}: {str(e)[:1500]}", flush=True)
                for line in [l for l in tb if "dense_mx8" in l or "Error" in l][-8:]:
                    print("     " + line.strip()[:300], flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
