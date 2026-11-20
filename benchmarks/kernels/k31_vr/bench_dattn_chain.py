# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The SP DSA sparse core (sparse_mla_ts) -> DcpValue ("proj") chain of a Kimi K3.1
decode slice at DCP4 / 24 heads per rank (one DP rank of D3-32), on 4 GPUs: what the
DcpValue exposes after the core, static kernel_tc vs kernel_ov (overlapped: the core
publishes its rows per wave block, DcpValue CTAs take the SMs it vacates).

Per row count: CUDA-graph time per layer of ``--layers`` chained layers (mailbox parities
alternating), rank max: the core alone, then core + DcpValue per mode; the outputs of
the modes are compared bitwise (same inputs).

  python benchmarks/kernels/k31_vr/bench_dattn_chain.py [--rows 128,256,384]
"""

import argparse
import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

HEADS, Q, CAP = 24, 4, 512
GATE = HEADS * 128
PITCH = 2112 + GATE + 128 + 32


def _port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _case(rows, local, jitter, ctx, gen, dev, width=1024):
    """A row's local keys: ``local`` +- ``jitter`` distinct rows of its request."""
    slots = torch.full((rows, width), -1, dtype=torch.int32, device=dev)
    valid = torch.empty(rows, dtype=torch.int32, device=dev)
    n = torch.randint(local - jitter, local + jitter + 1, (rows,), generator=gen,
                      device=dev).clamp(0, width)
    for r in range(rows):
        k = int(n[r])
        idx = torch.randperm(ctx, generator=gen, device=dev)[:k]
        slots[r, :k] = (idx + (r // Q) * ctx).int()
        valid[r] = k
    return slots, valid


def _graph_time(fn, layers, iters, dev):
    for i in range(3):
        fn(i % 2)
    torch.accelerator.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    side = torch.cuda.Stream(dev)
    side.wait_stream(torch.cuda.current_stream(dev))
    with torch.cuda.stream(side), torch.cuda.graph(
        graph, stream=side, capture_error_mode="thread_local"
    ):
        for i in range(layers):
            fn(i % 2)
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
    per = torch.tensor([start.elapsed_time(end) * 1000 / (iters * layers)], device=dev)
    dist.all_reduce(per, op=dist.ReduceOp.MAX)
    return per.item(), graph


def _worker(rank, size, port, rows, layers, iters, modes, local, jitter, stamps):
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
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa import (
        sparse_mla_ts,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.sparse_mla_ts import (  # noqa: E501
        sparse_mla_decode_ts,
    )

    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    stamp = torch.zeros(sms * dcp_value.STAMP_SLOTS, dtype=torch.int64, device=dev)
    ts_stamp = torch.zeros(sms * 8, dtype=torch.int64, device=dev)
    if stamps:
        dcp_value.STAMP_ADDR = stamp.data_ptr()
        sparse_mla_ts.TS_STAMP_ADDR = ts_stamp.data_ptr()
    box = sp.sp_dsa_box(group, dev, rank, heads=HEADS, cap=CAP, interleave=1)
    gen = torch.Generator(device=dev).manual_seed(11 + rank)
    w_uv = (torch.randn(HEADS, 128, 512, generator=gen, device=dev) / 16).bfloat16()
    w_uv = w_uv.transpose(1, 2)
    ctx = 135000 // 4
    reqs_max = -(-max(rows) // Q)
    kv = (torch.randn(reqs_max * ctx, 576, device=dev, generator=gen) * 0.5).to(
        torch.float8_e4m3fn).view(torch.uint8)
    bmm1, bmm2 = 0.07, 0.8
    for batch in rows:
        reqs = batch // Q
        query = (torch.randn(batch, box.q_heads, 576, device=dev, generator=gen) * 0.5)
        query[:, 96:] = 0
        query = query.to(torch.float8_e4m3fn).view(torch.uint8)
        slots, valid = _case(batch, local, jitter, ctx, gen, dev)
        latent = torch.zeros(batch, box.q_heads, 512, dtype=torch.bfloat16, device=dev)
        lse = torch.zeros(batch, box.q_heads, dtype=torch.float32, device=dev)
        totals = torch.ones(reqs, device=dev, dtype=torch.int32)
        projected = torch.randn(batch, PITCH, generator=gen, device=dev).bfloat16()
        dsa_box = DsaBox(box, (batch,), "proj", fp8=True, verify_q=Q)
        w_g = dsa_box.gather_heads(w_uv)
        mail = dsa_box.mailbox(batch)
        kernels = {}
        def ov_mode(mode):
            return mode.startswith("ov")

        for mode in modes:
            # ovK<n>: kernel_ov with dcp_value.KNOBS n (ablation ladder); tcc / ovc:
            # KNOBS 8192 (latent rows read 1 KiB apart: the head-major access pattern)
            knobs = int(mode[3:]) if mode.startswith("ovK") else 0
            # ovs: kernel_ov reading a static copy of the latent (not the core's
            # output of this layer): is the core-written latent the slow part?
            if mode in ("tcc", "ovc"):
                knobs = 8192
            if mode == "ovp":  # static latent at a 129-head row pitch (camping probe)
                knobs = 131072
            if mode == "ovh":  # static head-major latent [heads, rows, 512]
                knobs = 262144
            dcp_value.KNOBS = knobs
            kernels[mode] = compile_dcp_value(
                HEADS, size, batch, layout(w_uv), PITCH, dev.index, sms, Q,
                box.q_heads, True, False, "proj", box.cap, False,
                sms if ov_mode(mode) else 0,
            )  # fmt: skip
        dcp_value.KNOBS = 0

        def core(parity, ready=False):
            sync = mail.sync[parity, dcp_value.SYNC_READY:] if ready else None
            sparse_mla_decode_ts(query, kv, slots, valid, latent, lse, bmm1, bmm2,
                                 ready=sync)

        line = "rows %3d core %.2f us" % (batch, _graph_time(core, layers, iters, dev)[0])
        if "ov" in modes:  # the core publishing its rows, nobody consuming them
            pub = _graph_time(lambda p: core(p, True), layers, iters, dev)[0]
            mail.sync.zero_()
            line += " (publishing %.2f us)" % pub
        outs = {}
        for mode in modes:
            out = torch.zeros(batch, GATE, device=dev, dtype=torch.bfloat16)

            static = latent.clone() if mode == "ovs" else latent
            if mode == "ovp":
                pad = torch.zeros(batch, box.q_heads + 1, 512, dtype=latent.dtype, device=dev)
                pad[:, : box.q_heads] = latent
                static = pad
            if mode == "ovh":
                static = latent.transpose(0, 1).contiguous()  # [heads, rows, 512]

            def chain(parity, mode=mode, out=out, static=static):
                core(parity, ov_mode(mode))
                publish = publish_args(box, parity)
                run_value(kernels[mode], dsa_box, parity, publish, HEADS,
                          (static, lse, valid), totals, None, projected, out, w_g)

            t, graph = _graph_time(chain, layers, iters, dev)
            outs[mode] = out.clone()
            line += "; %s chain %.2f us" % (mode, t)
            if stamps and ov_mode(mode):
                # one more replay of the timed graph: the last layer's stamps (slot 0 =
                # entry; a slot older than the CTA's entry is from an earlier layer)
                stamp.zero_()
                ts_stamp.zero_()
                torch.accelerator.synchronize()
                dist.barrier()
                graph.replay()
                torch.accelerator.synchronize()
                s = stamp.view(sms, dcp_value.STAMP_SLOTS).double().cpu()
                have = s[:, 0] != 0
                e0 = s[have, 0].min()
                ts = ts_stamp.view(sms, 8).double().cpu()
                # the last layer's core: stamps within 60 us before its DcpValue's end
                last = s[have, 10].max()
                tparts = []
                for slot, name in ((0, "tsrow0"), (1, "tsrow1"), (7, "tsexit")):
                    ok = (ts[:, slot] != 0) & (ts[:, slot] > last - 60000)
                    v = ((ts[ok, slot] - e0) / 1000.0).sort().values
                    if len(v):
                        tparts.append("%s %.1f/%.1f/%.1f" % (
                            name, v[0], v[len(v) // 2], v[-1]))
                units = [int(x) for x in s[have, 11].tolist()]
                parts = ["ctas %d units/cta hist %s" % (
                    int(have.sum()), {k: units.count(k) for k in sorted(set(units))})]
                for slot, name in (((0, "entry"), (1, "alloc"), (4, "claim0"),
                                   (5, "ready0"), (6, "tma0"), (2, "acc0"),
                                   (3, "push0"), (7, "readyN"), (8, "drain"),
                                   (9, "merged"), (10, "end"))) + tuple(
                                   (base + k, "%s%d" % (nm, k)) for base, nm in (
                                       (12, "ld"), (28, "dr"), (16, "mm"), (20, "ac"),
                                       (24, "pu"))
                                   for k in range(4)):
                    ok = have & (s[:, slot] >= s[:, 0]) & (s[:, slot] != 0)
                    v = ((s[ok, slot] - e0) / 1000.0).sort().values
                    if len(v):
                        parts.append("%s %.1f/%.1f/%.1f" % (
                            name, v[0], v[len(v) // 2], v[-1]))
                line += "\n  r%d %s stamps us (min/med/max): %s" % (rank, mode, "; ".join(tparts + parts))
        if len(modes) > 1:
            a, b = (outs[m] for m in modes[:2])
            same = torch.equal(a.view(torch.int16), b.view(torch.int16))
            line += "; %s==%s bitwise %s" % (modes[0], modes[1], same)
            ok = torch.tensor([1 if same else 0], device=dev)
            dist.all_reduce(ok, op=dist.ReduceOp.MIN)
            line += " (all ranks %s)" % bool(ok.item())
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
    parser.add_argument("--modes", default="tc,ov")
    parser.add_argument("--local", type=int, default=512)
    parser.add_argument("--jitter", type=int, default=40)
    parser.add_argument("--stamps", action="store_true")
    a = parser.parse_args()
    rows = [int(r) for r in a.rows.split(",")]
    modes = a.modes.split(",")
    args = (4, _port(), rows, a.layers, a.iters, modes, a.local, a.jitter, a.stamps)
    mp.start_processes(_worker, args=args, nprocs=4, join=True, start_method="spawn")


if __name__ == "__main__":
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    main()
