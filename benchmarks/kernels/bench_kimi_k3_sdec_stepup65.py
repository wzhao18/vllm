# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 specialized decode: per-rows tactic candidates for the GEMMs that step up
above 64 shard rows (DDSA step-up, 2026-11-22) on one GPU, cold weights (a CUDA graph
chain with one weight copy per call), us per call, plus the max |d| vs the current
path's output.

  * router | latent down BF16 [4480, 7168] -> FP32 (main stream): split-K tactics
    (M tile, token tile, split, stages) of ``PreparedProjection``; prewait T64 / T96
    at their exact token counts.
  * shared expert MXFP8 gate_up [12288, 7168] and down [7168, 6144] (side stream):
    fi_cute / scaled_mm / the vendored sm_107 kernel ("pre") on fixed tactics.

python benchmarks/kernels/bench_kimi_k3_sdec_stepup65.py [--rows 64,66,72,80,96]
    [--parts bf16,gu,down]
"""

# ruff: noqa: B023  (every lambda runs inside its loop iteration)
import argparse
import statistics
import traceback

import torch

SPLITK = (
    (64, 64, 2, 6),
    (64, 80, 2, 5),
    (64, 96, 2, 5),
    (64, 80, 2, 4),
    (64, 96, 2, 4),
    (64, 80, 4, 4),
    (64, 96, 4, 3),
    (64, 72, 2, 5),
    (64, 80, 1, 5),
    (128, 80, 2, 3),
    (64, 128, 2, 3),
)
# (tile_mn, cluster_mn, swap_ab, inst_m), prewait k-blocks
PRE = (
    (((128, 64), (1, 1), False, 128), 4),
    (((128, 128), (1, 1), False, 128), 4),
    (((256, 64), (2, 1), True, 256), 4),
    (((256, 128), (2, 1), True, 256), 4),
    (((256, 128), (2, 1), True, 256), 8),
    (((128, 128), (1, 1), True, 128), 4),
    (((128, 64), (1, 1), True, 128), 4),
)
MX = {"gu": (12288, 7168), "down": (7168, 6144), "kdain": (12448, 7168)}
# kda_in runs on R = 4 n verify rows: its current table changes tactic at R 257
PRE_KDA = (
    (((256, 128), (2, 1), False, 256), 4),
    (((256, 192), (2, 1), True, 256), 4),
    (((256, 128), (2, 1), True, 256), 4),
    (((256, 256), (2, 1), False, 256), 4),
    (((256, 192), (2, 1), False, 256), 4),
    (((128, 128), (1, 1), False, 128), 4),
)
BF16 = (4480, 7168)


def time_graph(fn, ncopy, reps=10, rounds=7):
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for i in range(ncopy):
            fn(i)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        for i in range(ncopy):
            fn(i)
    torch.cuda.synchronize()
    ts = []
    for _ in range(rounds):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(reps):
            g.replay()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1e3 / reps / ncopy)
    del g
    return statistics.median(ts)


def ncopies(nbytes, l2=256 << 20):
    return max(4, -(-3 * l2 // nbytes))


def report(part, rows, arm, us, sol, err):
    print(f"{part} R{rows} {arm:<34} {us:7.2f} us  SOL {sol:5.1f}  x{us / sol:4.2f}  "
          f"maxd {err:.3g}", flush=True)  # fmt: skip


def bench_bf16(rows_list, dev, arm=None):
    from vllm.models.kimi_k3.nvidia.specialized_decode import prewait_gemm
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.projection import (
        PreparedProjection,
    )

    n, k = BF16
    nc = ncopies(n * k * 2)
    ws = [torch.randn(n, k, device=dev, dtype=torch.bfloat16) * 0.02 for _ in range(nc)]
    sol = n * k * 2 / 8e12 * 1e6
    packed = None
    for rows in rows_list:
        x = torch.randn(rows, k, device=dev, dtype=torch.bfloat16)
        ref = x.float() @ ws[0].float().t()
        scale = ref.abs().max().item()
        outs = [
            torch.empty(rows, n, device=dev, dtype=torch.float32) for _ in range(nc)
        ]
        for tac in SPLITK if arm is None else SPLITK[arm : arm + 1]:
            name = "splitk " + ".".join(map(str, tac))
            try:
                projs = [PreparedProjection(w, {rows: tac}, output_dtype=torch.float32)
                         for w in ws]  # fmt: skip
                for p in projs:
                    p.prepare()
                projs[0].run(x, outs[0])
                err = (outs[0] - ref).abs().max().item() / scale
                us = time_graph(lambda i: projs[i].run(x, outs[i]), nc)
                report("bf16", rows, name, us, sol, err)
            except Exception as e:  # noqa: BLE001
                print(f"bf16 R{rows} {name} FAILED {type(e).__name__}: {str(e)[:160]}",
                      flush=True)  # fmt: skip
        if arm in (None, len(SPLITK)) and prewait_gemm.supported(rows, k):
            if packed is None:
                packed = [prewait_gemm.pack_weight(w) for w in ws]
            prewait_gemm.gemm(outs[0], x, packed[0], n)
            err = (outs[0] - ref).abs().max().item() / scale
            us = time_graph(lambda i: prewait_gemm.gemm(outs[i], x, packed[i], n), nc)
            report("bf16", rows, f"prewait T{rows}", us, sol, err)
        torch.cuda.empty_cache()


def bench_mx(part, rows_list, dev, arm=None):
    import flashinfer

    from vllm.models.kimi_k3.nvidia.ops import mxfp8_act
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.sm107.prewait import (
        Mxfp8PreGemm,
    )

    n, k = MX[part]

    def mx(t):
        buf = mxfp8_act.Mxfp8Buffer(t.shape[0], t.shape[1], t.device)
        mxfp8_act.quantize(t, out=buf)
        return buf

    nc = ncopies(n * k)
    ws = [
        mx(torch.randn(n, k, device=dev, dtype=torch.bfloat16) * 0.05)
        for _ in range(nc)
    ]
    sol = (n * k + n * k // 32) / 8e12 * 1e6
    for rows in rows_list:
        x = mx(torch.randn(rows, k, device=dev, dtype=torch.bfloat16))
        outs = [
            torch.empty(rows, n, device=dev, dtype=torch.bfloat16) for _ in range(nc)
        ]

        def kind(name):
            def run(i):
                w = ws[i]
                if name == "scaled_mm":
                    return torch._scaled_mm(
                        x.data, w.data.t(), scale_a=x.scale.view(torch.float8_e8m0fnu),
                        scale_b=w.scale.view(torch.float8_e8m0fnu),
                        out_dtype=torch.bfloat16, out=outs[i],
                    )  # fmt: skip
                return flashinfer.mm_mxfp8(
                    x.data, w.data.t(), x.scale, w.scale, out=outs[i],
                    out_dtype=torch.bfloat16,
                    backend="cute-dsl" if name == "fi_cute" else "cutlass",
                )  # fmt: skip

            return run

        kind("fi_cute")(0)
        ref = outs[0].float().clone()  # fi_cute (the current path at <= 80 shard rows)
        scale = ref.abs().max().item()
        arms = [(name, kind(name)) for name in ("fi_cute", "scaled_mm")]
        if arm is not None and arm < 2:
            arms = arms[arm : arm + 1]
        elif arm is not None:
            arms = []
        pre = PRE_KDA if part == "kdain" else PRE
        if (
            arm is not None
        ):  # one arm per process: an illegal tactic poisons the context
            pre = pre[arm - 2 : arm - 1] if arm >= 2 else ()
        for tactic, prewait in pre:
            try:
                g = Mxfp8PreGemm(n, k, rows, tactic, use_pdl=True, prewait=prewait,
                                 device=dev)  # fmt: skip
            except Exception as e:  # noqa: BLE001
                print(f"{part} R{rows} pre {tactic} FAILED {type(e).__name__}: "
                      f"{str(e)[:160]}", flush=True)  # fmt: skip
                continue
            label = (
                f"pre {tactic[0]}c{tactic[1][0]}{'s' if tactic[2] else ''} pw{prewait}"
            )
            arms.append((label, lambda i, g=g: g(x.data, x.scale, ws[i].data, ws[i].scale,
                                                 outs[i])))  # fmt: skip
        for name, fn in arms:
            try:
                outs[0].zero_()
                fn(0)
                torch.cuda.synchronize()
                err = (outs[0].float() - ref).abs().max().item() / scale
                us = time_graph(fn, nc)
                report(part, rows, name, us, sol, err)
            except Exception as e:  # noqa: BLE001
                print(f"{part} R{rows} {name} FAILED {type(e).__name__}: {str(e)[:160]}",
                      flush=True)  # fmt: skip
                traceback.print_exc(limit=2)
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="64,65,66,68,72,76,80,88,96")
    ap.add_argument(
        "--arm",
        type=int,
        default=None,
        help="run one arm (index: bf16 SPLITK..., then prewait; MXFP8 fi_cute, "
        "scaled_mm, then the pre tactics) in this process",
    )
    ap.add_argument(
        "--narms", action="store_true", help="print the arm count of --parts"
    )
    ap.add_argument("--parts", default="bf16,gu,down,kdain")
    ap.add_argument("--kda-rows", default="256,260,264,272,288")  # R = 4 n
    ap.add_argument(
        "--pre",
        default=None,
        help="override the pre tactic list: a Python literal of "
        "((tile_mn, cluster_mn, swap_ab, inst_m), prewait) pairs",
    )
    args = ap.parse_args()
    dev = torch.device("cuda", 0)
    torch.manual_seed(0)
    rows = [int(r) for r in args.rows.split(",")]
    props = torch.cuda.get_device_properties(0)
    print(f"{props.name} sms {props.multi_processor_count}", flush=True)
    if args.pre:
        import ast

        global PRE, PRE_KDA
        PRE = PRE_KDA = tuple(ast.literal_eval(args.pre))
    if args.narms:
        part = args.parts
        print(
            len(SPLITK) + 1
            if part == "bf16"
            else 2 + len(PRE_KDA if part == "kdain" else PRE)
        )
        return
    for part in args.parts.split(","):
        if part == "bf16":
            bench_bf16(rows, dev, args.arm)
        else:
            kr = [int(r) for r in args.kda_rows.split(",")]
            bench_mx(part, kr if part == "kdain" else rows, dev, args.arm)


if __name__ == "__main__":
    main()
