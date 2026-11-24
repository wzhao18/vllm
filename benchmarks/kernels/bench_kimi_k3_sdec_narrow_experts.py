# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 specialized decode: per-call time of the small-batch routed experts
(TP16 per-rank shapes, M8 ``RubinExpertsKernel`` / M16-64 ``RubinGroupedExpertsKernel``)
on one Rubin GPU, cold weights (L2 flushed before every call), q8 verify-like routes.

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
    parser.add_argument("--q", type=int, default=8)
    parser.add_argument("--iters", type=int, default=60)
    args = parser.parse_args()

    import torch

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        stream,
        to_cute,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe import (
        experts_runtime as er,
    )

    t = load(args.test)
    dev = torch.device("cuda", 0)
    torch.accelerator.set_device_index(dev)
    built = t.production_experts(t.logical_weights(dev), dev)
    flush = torch.empty(1 << 29, dtype=torch.uint8, device=dev)
    for rows in args.rows:
        ids, _, latent = t.verify_routes(rows, dev, q=args.q)
        latent, ids = latent.contiguous(), ids.contiguous()
        buffers = er.scratch(16, dev)
        experts = er.PersistentExperts(
            *built["weights"], beta=t.BETA, linear_beta=t.LINEAR_BETA,
            dies=built["dies"], q=4,
        )  # fmt: skip
        out = torch.empty(rows * t.TOPK, t.H_LATENT, dtype=torch.bfloat16, device=dev)
        experts.prepare(rows, buffers, out, (), True, False, False)
        rubin = rows in experts.rubin
        if rubin:
            writer = er.compile_image_plan(rows, dev.index)
            kernel = "RubinExpertsKernel"
        else:
            assert experts.grouped64(rows), rows
            writer = er.compile_plan(rows, dev.index)
            kernel = "RubinGroupedExpertsKernel"
        mailbox = rubin and er.polled(16, rows, dev)
        box = torch.full((er.ROUTE_WORDS,), -1, dtype=torch.int32, device=dev)
        box[: rows * t.TOPK] = ids.reshape(-1)
        box[rows * t.TOPK] = 2 * rows - 1

        def call():
            writer(to_cute(latent, 16), to_cute(buffers["plan"], 16), stream(dev))
            if mailbox:
                er.route_mailbox(buffers["workspace"]).copy_(box)
            flush.zero_()
            start.record()
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


if __name__ == "__main__":
    main()
