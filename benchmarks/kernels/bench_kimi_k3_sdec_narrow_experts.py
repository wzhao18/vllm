# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 specialized decode: per-call time of the small-batch routed experts
(TP16 per-rank shapes, M8 ``RubinExpertsKernel`` / M16-64 ``RubinGroupedExpertsKernel`` /
M65+ the wide engines through ``WideExperts``)
on one Rubin GPU, cold weights (L2 flushed before every call), verify-like routes.

Builds the production weights and launches exactly as
``tests/models/kimi_k3/test_specialized_decode_wide_experts_gpu.py`` (``run_narrow``)
does, then times ``PersistentExperts.run`` with CUDA events. Prints, per row count:
distinct experts, median / p10 / p90 us per call, the active-expert bytes rate, and a
digest of the expanded output rows (bitwise comparison across builds).

    python benchmarks/kernels/bench_kimi_k3_sdec_narrow_experts.py \
        --test tests/models/kimi_k3/test_specialized_decode_wide_experts_gpu.py \
        --rows 8 16 32 64
"""

import argparse
import hashlib
import importlib.util

# Bytes one active expert's packed weights and scales occupy per TP16 rank (the SOL's
# dense count; the padded layout moves more).
EXPERT_BYTES = 1_096_704


def load(path):
    spec = importlib.util.spec_from_file_location("wide_experts_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--q", type=int, default=4)
    parser.add_argument("--iters", type=int, default=60)
    parser.add_argument(
        "--no-dies", action="store_true",
        help="die-oblivious kernels on the placed weights (die-matching check)",
    )  # fmt: skip
    parser.add_argument(
        "--width", type=int, default=0,
        help="wide rows: stream width (8 / 16 / 32) instead of experts_runtime.WIDE's",
    )  # fmt: skip
    parser.add_argument(
        "--trace", action="store_true",
        help="grouped kernel phase stamps (us from the first CTA's entry, median / max "
        "over CTAs) of the last timed call",
    )  # fmt: skip
    args = parser.parse_args()

    import torch

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        stream,
        to_cute,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe import (
        experts_runtime as er,
    )

    if args.width:
        er.WIDE[16] = f"65-:{args.width}"
    t = load(args.test)
    dev = torch.device("cuda", 0)
    torch.accelerator.set_device_index(dev)
    trace = None
    if args.trace:
        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe import (
            experts_rubin64 as r64,
        )

        trace = torch.zeros(256 * 512 + 512, dtype=torch.uint8, device=dev)
        init = r64.RubinGroupedExpertsKernel.__init__

        def traced(self, *a, **k):
            if type(self) is r64.RubinGroupedExpertsKernel:  # (subclasses pass trace)
                k.setdefault("trace", trace.data_ptr())
            init(self, *a, **k)

        r64.RubinGroupedExpertsKernel.__init__ = traced
    built = t.production_experts(t.logical_weights(dev), dev)
    flush = torch.empty(1 << 29, dtype=torch.uint8, device=dev)
    for rows in args.rows:
        ids, topk_w, latent = t.verify_routes(rows, dev, q=args.q)
        latent, ids = latent.contiguous(), ids.contiguous()
        buffers = er.scratch(16, dev, rows=max(rows, 64))
        experts = er.PersistentExperts(
            *built["weights"], beta=t.BETA, linear_beta=t.LINEAR_BETA,
            dies=() if args.no_dies else built["dies"], q=4,
        )  # fmt: skip
        out = torch.empty(rows * t.TOPK, t.H_LATENT, dtype=torch.bfloat16, device=dev)
        wide, rubin = None, False
        if rows > 64:  # the wide engines (``WideExperts``, plan written in front)
            partial = torch.empty(rows, t.H_LATENT, dtype=torch.bfloat16, device=dev)
            wide = er.WideExperts(experts, buffers, {rows: partial}, planned=True)
            writer = er.compile_wide_plan(rows, dev.index)
            width = experts.wide[rows]
            kernel = f"RubinVWExpertsKernel w{width}" if width > 8 else "RubinWide w8"
        else:
            experts.prepare(rows, buffers, out, (), True, False, False)
            rubin = rows in experts.rubin
        if wide is not None:
            pass
        elif rubin:
            writer = er.compile_image_plan(rows, dev.index)
            kernel = "RubinExpertsKernel"
        else:
            assert experts.grouped64(rows), rows
            writer = er.compile_plan(rows, dev.index)
            kernel = "RubinGroupedExpertsKernel"
        mailbox = rubin and er.polled(16, rows, dev)
        if wide is not None:
            out = partial
        if mailbox:  # the routing kernel's ids and copy-CTA count
            box = torch.full((er.ROUTE_WORDS,), -1, dtype=torch.int32, device=dev)
            box[: rows * t.TOPK] = ids.reshape(-1)
            box[rows * t.TOPK] = 2 * rows - 1

        def call():
            writer(to_cute(latent, 16), to_cute(buffers["plan"], 16), stream(dev))
            if mailbox:
                er.route_mailbox(buffers["workspace"]).copy_(box)
            flush.zero_()
            start.record()
            if wide is not None:
                wide.run(latent, topk_w, ids)
            else:
                experts.run(latent, ids)
            end.record()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(5):
            call()
        torch.cuda.synchronize()
        digest = hashlib.sha1(out.view(torch.uint8).cpu().numpy().tobytes())
        times = []
        for _ in range(args.iters):
            call()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) * 1000.0)
        times.sort()
        n = len(times)
        med, p10, p90 = times[n // 2], times[n // 10], times[9 * n // 10]
        distinct = int(torch.unique(ids).numel())
        rate = distinct * EXPERT_BYTES / (med * 1e-6) / 1e12
        print(
            f"M{rows} {kernel} distinct {distinct} us med {med:.2f} p10 {p10:.2f} "
            f"p90 {p90:.2f} sol-bytes rate {rate:.2f} TB/s "
            f"out {digest.hexdigest()[:16]}",
            flush=True,
        )
        if trace is not None and not rubin and wide is None:
            trace.zero_()
            call()
            torch.cuda.synchronize()
            phases(trace, rows)


# Stamp slots (``RubinGroupedExpertsKernel.body`` / ``stream`` / ``consume``).
PHASES = (
    (0, "entry"), (1, "dep wait done"), (2, "grouped"), (11, "partitioned"),
    (3, "1st issue"), (6, "1st MMA"), (4, "FC1 issued"), (7, "1st h wait"),
    (8, "1st h ready"), (5, "end issued"), (14, "gather done"), (9, "MMAs done"),
    (12, "h loader done"), (10, "epilogue done"), (13, "CTA end"),
)  # fmt: skip


def phases(trace, rows):
    import torch

    words = trace[: 256 * 512].view(torch.int64).view(256, 32, 2).cpu()
    stamps = words[:, :16, 0]
    used = stamps[:, 13] > 0
    stamps = stamps[used]
    t0 = stamps[:, 0].min()
    cells = []
    for slot, name in PHASES:
        col = stamps[:, slot]
        col = (col[col > 0] - t0).double() / 1000.0
        if col.numel():
            cells.append(f"{name} {col.median():.1f}/{col.max():.1f}")
    head = f"  M{rows} CTAs {int(used.sum())} phases us (median/max): "
    print(head + "; ".join(cells))


if __name__ == "__main__":
    main()
