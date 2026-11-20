# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The SP DSA DcpValue ("proj": W_UV on tcgen05, push, merge, gate) of a Kimi K3.1
decode slice at DCP4 / 24 heads per rank (one DP rank of D3-32), on 4 GPUs.

Per row count: CUDA-graph time per call of ``--layers`` back-to-back calls (the two
mailbox parities alternating, as consecutive DSA layers), rank max; with
``--stamps`` the last call's per-CTA phase stamps (dcp_value.STAMP_ADDR):
  0 entry, 1 after griddepcontrol_wait + TMEM alloc, 2 / 4 unit 0 / 1 MMA done,
  3 / 5 unit 0 / 1 push issued, 8 pushes drained, 9 LSE polls done, 10 end.

  python benchmarks/kernels/k31_vr/bench_dattn_dcpv.py [--rows 128,256,384] [--stamps]
"""

import argparse
import os
import socket
import statistics

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

HEADS, Q, CAP = 24, 4, 512
GATE = HEADS * 128
PITCH = 2112 + GATE + 128 + 32
SLOTS = 16


def _port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _stats(values):
    values = sorted(values)
    return "%6.2f/%6.2f/%6.2f" % (
        values[0], statistics.median(values), values[-1]
    )


def _worker(rank, size, port, rows, layers, iters, stamps, knobs=0):
    torch.accelerator.set_device_index(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank,
        world_size=size, device_id=dev,
    )  # fmt: skip
    group = dist.group.WORLD
    try:
        from torch.distributed._symmetric_memory import enable_symm_mem_for_group

        enable_symm_mem_for_group(group.group_name)
    except (ImportError, AttributeError):
        pass
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        layout,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa import (
        dcp_value,
        sp,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.dcp_stage import (
        compile_dcp_value,
        publish_args,
        run_value,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.r1 import DsaBox

    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    stamp = torch.zeros(sms * SLOTS, dtype=torch.int64, device=dev)
    if stamps:
        dcp_value.STAMP_ADDR = stamp.data_ptr()
    dcp_value.KNOBS = knobs
    box = sp.sp_dsa_box(group, dev, rank, heads=HEADS, cap=CAP, interleave=1)
    gen = torch.Generator(device=dev).manual_seed(11 + rank)
    w_uv = (torch.randn(HEADS, 128, 512, generator=gen, device=dev) / 16).bfloat16()
    w_uv = w_uv.transpose(1, 2)
    for batch in rows:
        reqs = batch // Q
        latent = torch.randn(batch, box.q_heads, 512, generator=gen, device=dev)
        latent = latent.bfloat16()
        lse = torch.randn(batch, box.q_heads, generator=gen, device=dev) * 3
        valid = torch.randint(400, 640, (batch,), generator=gen, device=dev).int()
        totals = torch.ones(reqs, device=dev, dtype=torch.int32)
        projected = torch.randn(batch, PITCH, generator=gen, device=dev).bfloat16()
        out = torch.zeros(batch, GATE, device=dev, dtype=torch.bfloat16)
        dsa_box = DsaBox(box, (batch,), "proj", fp8=True, verify_q=Q)
        w_g = dsa_box.gather_heads(w_uv)
        value = compile_dcp_value(
            HEADS, size, batch, layout(w_uv), PITCH, dev.index, sms, Q,
            box.q_heads, True, False, "proj", box.cap, False,
        )  # fmt: skip
        grid = dcp_value.DcpValue(
            HEADS, size, batch, sms, q=Q, q_heads=box.q_heads, per_row=True,
            partials="proj", rearm=False,
        ).grid  # fmt: skip

        def call(parity):
            publish = publish_args(box, parity)
            run_value(value, dsa_box, parity, publish, HEADS,
                      (latent, lse, valid), totals, None, projected, out, w_g)  # fmt: skip

        for i in range(4):
            call(i % 2)
        torch.accelerator.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        side = torch.cuda.Stream(dev)
        side.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(side), torch.cuda.graph(
            graph, stream=side, capture_error_mode="thread_local"
        ):
            for i in range(layers):
                call(i % 2)
        torch.accelerator.synchronize()
        dist.barrier()
        graph.replay()
        torch.accelerator.synchronize()
        dist.barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            graph.replay()
        end.record()
        torch.accelerator.synchronize()
        per = torch.tensor(
            [start.elapsed_time(end) * 1000 / (iters * layers)], device=dev
        )
        dist.all_reduce(per, op=dist.ReduceOp.MAX)
        out_dir = os.environ.get("DSK_OUT")
        if out_dir and os.environ.get("DSK_TAG"):
            torch.save(out.cpu(), "%s/out_%s_r%d_rank%d.pt" % (
                out_dir, os.environ["DSK_TAG"], batch, rank))
        line = "knobs %d rows %3d grid %s units %s: %.2f us/call (rank max, %d x %d)" % (
            knobs, batch, grid, HEADS * size * -(-batch // 128), per.item(), iters,
            layers,
        )
        if stamps:
            ctas = grid[0] * grid[1]
            s = stamp.view(sms, SLOTS)[:ctas].double().cpu()
            t0 = s[:, 0].min()
            rel = (s - t0) / 1000.0
            have = s != 0
            parts = []
            for slot, name in ((0, "entry"), (1, "alloc"), (6, "land0"), (7, "land7"),
                               (2, "mma0"), (3, "push0"),
                               (4, "mma1"), (5, "push1"), (8, "drain"),
                               (9, "lsepoll"), (11, "val1"), (12, "val2"),
                               (10, "end")):  # fmt: skip
                vals = rel[have[:, slot], slot].tolist()
                if vals:
                    parts.append("%s %s" % (name, _stats(vals)))
            line += "\n  r%d stamps us (min/med/max vs first entry): " % rank
            line += "; ".join(parts)
            # per-CTA phase spans (median / max over CTAs)
            spans = []
            for a_, b_, name in ((1, 3, "alloc>push0"), (1, 8, "alloc>drain"),
                                 (1, 9, "alloc>lse"), (9, 11, "lse>val1"),
                                 (11, 12, "val1>val2"), (9, 10, "lse>end"),
                                 (1, 10, "alloc>end")):  # fmt: skip
                ok = have[:, a_] & have[:, b_]
                if ok.any():
                    d = (rel[ok, b_] - rel[ok, a_]).tolist()
                    spans.append("%s %.2f/%.2f" % (name, statistics.median(d), max(d)))
            line += "\n  r%d spans us (med/max): " % rank + "; ".join(spans)
            out_dir = os.environ.get("DSK_OUT")
            if out_dir:
                torch.save(s, "%s/st_k%d_r%d_rank%d.pt" % (out_dir, knobs, batch, rank))
        dist.barrier()
        for r in range(size):
            if r == rank and (r == 0 or stamps):
                print(line, flush=True)
            dist.barrier()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", default="128,256,384")
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--stamps", action="store_true")
    # dcp_value.KNOBS ablations, one process group each (0 = production code)
    parser.add_argument("--knobs", default="0")
    a = parser.parse_args()
    rows = [int(r) for r in a.rows.split(",")]
    for knobs in [int(k) for k in a.knobs.split(",")]:
        args = (4, _port(), rows, a.layers, a.iters, a.stamps, knobs)
        mp.start_processes(
            _worker, args=args, nprocs=4, join=True, start_method="spawn"
        )


if __name__ == "__main__":
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    main()
