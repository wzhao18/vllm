# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SP decode KDA core microbench at the deployed D3-32 per-GPU shape (24 KDA heads
per rank, FP32 state, q4, deferred RecoverSSM with the pending commit folded).

One CUDA graph of LAYERS core calls, each on its own layer (weights, records) and its
own disjoint checkpoint slots (state working set far above L2), replayed; prints us per
layer. Every request has a pending commit of 1..4 tokens (steady state), so folding
cores read + write the state; non-folding cores (ws) are timed without their flush.

usage: python kda_sp_bench.py --n 32 64 96 128 --kinds wide ws [--layers 69]
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.getcwd())
os.environ.setdefault("VLLM_SSM_CONV_STATE_LAYOUT", "DS")

from tests.models.kimi_k3 import test_specialized_decode_sp_kda_gpu as kda  # noqa: E402
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.kda import sp  # noqa: E402

Q, HEADS, DIM, MAX_REQS = kda.Q, kda.HEADS, kda.DIM, kda.MAX_REQS


def bench_flush(n, layers, iters, seed=0):
    """The deferred flush KdaWS needs before its verify (production n >= 71): the
    FP32 commit of every layer's pending replay (KDADeferredCommitContext.flush) on
    one cache for all layers (the same pages per layer: as many bytes as the
    deployed per-layer caches)."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.recoverssm.context import (
        KDADeferredCommitContext,
    )

    g = torch.Generator().manual_seed(seed)
    kv = list(kda.caches(n + 2))
    owners = []
    for i in range(layers):
        attention = kda.layer(seed=i)
        attention.kv_cache = kv
        owners.append(attention)
    context = KDADeferredCommitContext.create(
        owners, spec_query_len=Q, max_num_reqs=MAX_REQS
    )
    slots = (torch.randperm(n + 1, generator=g)[:n] + 1).to(torch.int32)
    state_indices = slots.view(n, 1).to(kda.DEVICE)
    record_rows = torch.randperm(MAX_REQS, generator=g)[:n].to(torch.int32)
    record_rows = record_rows.to(kda.DEVICE)
    pending = torch.randint(1, Q + 1, (MAX_REQS,), generator=g).to(torch.int32)
    pending = pending.to(kda.DEVICE)

    def run():
        context.flush(state_indices, record_rows, pending)

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1e3 / layers)
    times.sort()
    del graph
    return times[len(times) // 2], times[0]


def bench(n, kind, layers, iters, seed=0):
    if kind == "flush":
        return bench_flush(n, layers, iters, seed)
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.kda import flow

    stages = None
    if kind[-1].isdigit():  # flow1 / flow2: the state ring depth
        kind, stages = kind[:-1], int(kind[-1])
    real_stages = flow.max_stages
    if stages:
        flow.max_stages = lambda *args: stages
    sp.executable.cache_clear()
    rows = Q * n
    slots = layers * n + 2
    kv = list(kda.caches(slots))
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(slots - 1, generator=g) + 1
    projected = torch.randn(rows, kda.PITCH, generator=torch.Generator().manual_seed(1))
    projected = projected.to(torch.bfloat16).to(kda.DEVICE)
    out = torch.empty(rows, DIM, dtype=torch.bfloat16, device=kda.DEVICE)
    saved = dict(sp.RECORD_KINDS)
    sp.RECORD_KINDS[HEADS] = f"1-:{kind}"
    calls = []
    try:
        for i in range(layers):
            attention = kda.layer(seed=i)
            attention.deferred_recoverssm = True
            checkpoints = order[i * n : (i + 1) * n].to(torch.int32).view(n, 1)
            record_rows = torch.randperm(MAX_REQS, generator=g)[:n].to(torch.int32)
            pending = torch.randint(1, Q + 1, (MAX_REQS,), generator=g).to(torch.int32)
            metadata = kda.Module(
                spec_state_indices_tensor=checkpoints.to(kda.DEVICE),
                num_accepted_tokens=torch.ones(n, dtype=torch.int32, device=kda.DEVICE),
                num_spec_decodes=n,
                recoverssm_deferred=kda.Module(
                    record_rows=record_rows.to(kda.DEVICE),
                    pending=pending.to(kda.DEVICE),
                    flush=lambda: None,
                ),
            )
            core = sp.SpKdaCore(attention, (rows,), records="deferred", ctx=kda.context())
            core.prepare_caches(kv)
            calls.append((core, metadata))
    finally:
        sp.RECORD_KINDS.clear()
        sp.RECORD_KINDS.update(saved)
        flow.max_stages = real_stages

    def run():
        for core, metadata in calls:
            core.run(projected, out, caches=kv, metadata=metadata)

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1e3 / layers)
    times.sort()
    del graph
    return times[len(times) // 2], times[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, nargs="+", default=[32, 64, 96, 128])
    p.add_argument("--kinds", nargs="+", default=["wide", "ws"])
    p.add_argument("--layers", type=int, default=69)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--json", default="")
    p.add_argument("--trace", action="store_true", help="KdaFlow cycle trace")
    p.add_argument("--skip", default="", help="KdaFlow diagnostics: fold,store,verify,prep")
    p.add_argument("--hang", action="store_true", help="KdaFlow stuck-wait report")
    p.add_argument("--scalar", action="store_true", help="KdaFlow scalar FP32 pairs")
    p.add_argument("--fence", type=int, default=0, help="KdaFlow fold fence interval")
    p.add_argument("--noprefetch", action="store_true", help="KdaFlow no prefetch")
    p.add_argument("--slots", type=int, default=0, help="KdaFlow page slots")
    p.add_argument("--page-ops", type=int, default=0, help="KdaFlow bulk copies per page")
    p.add_argument("--unfused", action="store_true", help="KdaFlow two verify passes")
    a = p.parse_args()
    if a.trace:
        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.kda import flow

        flow.TRACE = True
    if a.hang:
        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.kda import flow

        flow.HANG = True
    if a.unfused:
        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.kda import flow

        flow.FUSED = False
    if a.scalar or a.fence or a.noprefetch:
        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.kda import flow

        flow.PACKED = not a.scalar
        flow.FOLD_FENCE = a.fence or flow.FOLD_FENCE
        flow.PREFETCH = not a.noprefetch
    if a.slots:
        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.kda import flow

        flow.SLOTS = a.slots
    if a.page_ops:
        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.kda import flow

        flow.PAGE_OPS = a.page_ops
    if a.skip:
        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.kda import flow

        flow.SKIP = tuple(a.skip.split(","))
    props = torch.cuda.get_device_properties(0)
    print(f"device {props.name} sms {props.multi_processor_count}", flush=True)
    results = []
    for n in a.n:
        for kind in a.kinds:
            try:
                med, best = bench(n, kind, a.layers, a.iters)
            except Exception as exc:  # noqa: BLE001
                print(f"n{n} {kind}: FAILED {type(exc).__name__}: {exc}", flush=True)
                continue
            items = HEADS * n
            mb = items * (2 * 128 * 128 * 4) / 1e6
            print(
                f"n{n:4d} r{n / 4:5.1f} {kind:9s} {med:7.2f} us/layer (best {best:7.2f})"
                f"  state R+W {mb:6.1f} MB -> {mb / med:5.2f} TB/s"
                f"  x69 {med * 69 / 1e3:5.3f} ms",
                flush=True,
            )
            results.append(dict(n=n, kind=kind, us=med, best=best))
            torch.cuda.empty_cache()
    if a.json:
        with open(a.json, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
