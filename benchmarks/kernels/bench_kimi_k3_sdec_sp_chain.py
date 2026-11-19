# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 SP decode (D3-32: TP4 = DCP4 = SP4) layer-boundary chain microbench.

Times, in one CUDA graph of the 93 decoder layers, the per-layer small ops that sit
between the attention core and the MLP GEMMs on this rank's SP shard (n = R / 4 rows):

  pre AttnRes (stock op) -> SP all-gather -> [in-proj .. core .. o-proj: not run]
  -> SP reduce-scatter (push this rank's partial + owner sum) -> post AttnRes (stock op)
  -> [MLP: not run; its output is a fixed delta]

with the model's AttnRes block pattern (block size 12, 8 blocks). Variants:
``full`` (the chain), ``comm`` (AG + RS only), ``attnres`` (pre + post only), and the
fused variants of this workstream when present. Run with torchrun on 4 GPUs.

usage: torchrun --nproc-per-node 4 bench_kimi_k3_sdec_sp_chain.py [--rows 16,128,384]
       [--variants full,comm,attnres] [--layers 93] [--profile DIR]
"""

import argparse
import json
import os
import statistics

import torch
import torch.distributed as dist

WIDTH = 7168
CAP = 512
BLOCK = 12
SENTINEL = -0x80000000


def cdiv(a, b):
    return -(-a // b)


def sanitized_randn(shape, gen, device, scale=1.0):
    x = (torch.randn(shape, generator=gen) * scale).bfloat16()
    words = x.view(torch.int32)
    words[words == SENTINEL] = 0
    return x.to(device)


class Chain:
    def __init__(self, comm, rank, world, rows, layers, device):
        self.comm, self.rank, self.world = comm, rank, world
        self.rows, self.n, self.layers = rows, rows // world, layers
        n = self.n
        gen = torch.Generator().manual_seed(1234 + rank)
        nblocks = cdiv(layers, BLOCK)
        self.nblocks = nblocks
        self.blocks = sanitized_randn((n, nblocks, WIDTH), gen, device)
        self.prefix0 = sanitized_randn((n, WIDTH), gen, device)
        self.prefix = self.prefix0.clone()
        self.delta = sanitized_randn((n, WIDTH), gen, device, 0.1)
        self.partial = sanitized_randn((rows, WIDTH), gen, device, 0.05)
        self.rs_out = [torch.empty((n, WIDTH), dtype=torch.bfloat16, device=device)
                       for _ in range(2)]
        # per-layer weights (norm ~1, query small, output norm ~1)
        self.w = []
        for _ in range(layers):
            ws = []
            for _ in range(2):
                norm = (1 + 0.1 * torch.randn(WIDTH, generator=gen)).bfloat16()
                query = (0.05 * torch.randn(WIDTH, generator=gen)).bfloat16()
                out = (1 + 0.1 * torch.randn(WIDTH, generator=gen)).bfloat16()
                ws.append(tuple(t.to(device) for t in (norm, query, out)))
            self.w.append(ws)

    def boundaries(self, mx=False):
        """``mx``: the MXFP8 forms (MXFP8 all-gather of every layer, the post-AttnRes
        also writing an MXFP8 input scratch): the NVFP4 checkpoint's KDA layers."""
        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.attn_res import (
            sp_fused,
        )
        from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm import mxfp8

        key = "_bmx" if mx else "_b"
        if getattr(self, key, None) is None:
            bounds = []
            scratch = mxfp8.Mxfp8Input(self.n, WIDTH, self.comm.device) if mx else None
            for layer in range(self.layers):
                prev = cdiv(layer, BLOCK)
                write = layer // BLOCK if layer % BLOCK == 0 else -1
                eps = (1e-6, 1e-6)
                bounds.append(sp_fused.SpBoundary(
                    self.comm, (self.w[layer][0], eps), (self.w[layer][1], eps),
                    prev, write, self.n, {self.rows: True}, layer == 0,
                    gather_mx=mx, out_mx=scratch,
                ))
            setattr(self, key, bounds)
        return getattr(self, key)

    def run_fused(self, mx=False):
        bounds = self.boundaries(mx)
        hidden = None
        for layer in range(self.layers):
            parity = layer % 2
            b = bounds[layer]
            b.gather(hidden, self.prefix, self.blocks, parity)
            b.reduce(self.prefix, self.blocks, self.rs_out[parity], parity,
                     self.partial)
            hidden = self.delta

    def check(self, layers):
        """Fused vs stock on single layers (eager): gathered rows, post output,
        prefix."""
        from vllm.models.kimi_k3.nvidia.ops.attn_res import attn_res

        comm, n, out = self.comm, self.n, {}
        bounds = self.boundaries()
        for layer in layers:
            parity = layer % 2
            prev = cdiv(layer, BLOCK)
            write = layer % BLOCK == 0
            delta = None if layer == 0 else self.delta
            res = {}
            for mode in ("stock", "fused"):
                prefix = self.prefix0.clone()
                blocks = self.blocks.clone()
                torch.distributed.barrier()
                if mode == "stock":
                    norm, query, out_w = self.w[layer][0]
                    x = attn_res(prefix, delta, blocks, norm, query, out_w,
                                 num_blocks=prev,
                                 block_write_idx=(layer // BLOCK if write else -1),
                                 eps=1e-6, output_norm_eps=1e-6)
                    g = comm.all_gather(x, parity).clone()
                    h = comm.reduce_scatter(self.rs_out[parity], parity, self.rows,
                                            self.partial)
                    norm, query, out_w = self.w[layer][1]
                    if write:
                        prefix = h.clone()
                        d = None
                    else:
                        d = h
                    y = attn_res(prefix, d, blocks, norm, query, out_w,
                                 num_blocks=prev + int(write), block_write_idx=-1,
                                 eps=1e-6, output_norm_eps=1e-6)
                else:
                    b = bounds[layer]
                    g = b.gather(delta, prefix, blocks, parity).clone()
                    y = b.reduce(prefix, blocks, self.rs_out[parity], parity,
                                 self.partial).clone()
                torch.accelerator.synchronize(comm.device)
                res[mode] = (g.float(), y.float(), prefix.float(), blocks.float())
            names = ("gather", "post_out", "prefix", "blocks")
            for name, a, b in zip(names, res["stock"], res["fused"]):
                err = (a - b).abs().max().item()
                ref = a.abs().max().item()
                out[f"L{layer}.{name}"] = (err, ref)
            # armed?
            g = comm.gather[parity, : self.rows].view(torch.int32)
            r = comm.recv[parity].view(torch.int32)
            out[f"L{layer}.armed"] = (float(not bool((g == SENTINEL).all() and
                                                     (r == SENTINEL).all())), 0.0)
        return out

    def run(self, variant):
        from vllm.models.kimi_k3.nvidia.ops.attn_res import attn_res

        if variant in ("fused", "fused_mx"):
            return self.run_fused(variant == "fused_mx")

        comm, n = self.comm, self.n
        prefix = self.prefix
        hidden = None
        do_attn = variant in ("full", "attnres")
        do_comm = variant in ("full", "comm")
        if variant == "rs":  # the reduce-scatter alone (comm - rs = the all-gather)
            for layer in range(self.layers):
                comm.reduce_scatter(self.rs_out[layer % 2], layer % 2, self.rows,
                                    self.partial)
            return
        for layer in range(self.layers):
            parity = layer % 2
            prev = cdiv(layer, BLOCK)
            write = layer % BLOCK == 0
            norm, query, out_w = self.w[layer][0]
            if do_attn:
                x = attn_res(
                    prefix, hidden, self.blocks, norm, query, out_w,
                    num_blocks=prev, block_write_idx=(layer // BLOCK if write else -1),
                    eps=1e-6, output_norm_eps=1e-6,
                )  # fmt: skip
            else:
                x = self.delta
            if do_comm:
                comm.all_gather(x, parity)
                h = comm.reduce_scatter(self.rs_out[parity], parity, self.rows,
                                        self.partial)
            else:
                h = self.rs_out[parity]
            if do_attn:
                norm, query, out_w = self.w[layer][1]
                if write:
                    prefix = h
                    d = None
                else:
                    d = h
                hidden = attn_res(
                    prefix, d, self.blocks, norm, query, out_w,
                    num_blocks=prev + int(write), block_write_idx=-1,
                    eps=1e-6, output_norm_eps=1e-6,
                )  # fmt: skip
            hidden = self.delta  # (the MLP output: fixed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", default="16,128,384")
    p.add_argument("--variants", default="full,comm,attnres")
    p.add_argument("--layers", type=int, default=93)
    p.add_argument("--replays", type=int, default=30)
    p.add_argument("--profile", default="")
    p.add_argument("--check", action="store_true")
    p.add_argument("--out", default="")
    args = p.parse_args()
    local = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(local)
    device = torch.device(f"cuda:{local}")
    dist.init_process_group(backend="nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    group = dist.group.WORLD
    try:
        from torch.distributed._symmetric_memory import enable_symm_mem_for_group

        enable_symm_mem_for_group(group.group_name)
    except (ImportError, AttributeError):
        pass
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common import sp_comm

    comm = sp_comm.SpComm(group, device, world, rank, rows_cap=CAP)
    results = {}
    for rows in [int(r) for r in args.rows.split(",")]:
        chain = Chain(comm, rank, world, rows, args.layers, device)
        if args.check:
            res = chain.check([0, 1, 5, 12, 13, 50, 92])
            from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.attn_res import (
                sp_fused,
            )

            if rank == 0:
                print("residency", sp_fused.RESIDENCY, flush=True)
            if rank == 0:
                for k, (err, ref) in res.items():
                    print(f"R{rows} check {k:>16}: max|diff| {err:.3e} (max|ref| "
                          f"{ref:.3e})", flush=True)
        for variant in args.variants.split(","):
            chain.prefix.copy_(chain.prefix0)
            dist.barrier()
            chain.run(variant)  # warm (compiles)
            torch.accelerator.synchronize(device)
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                chain.run(variant)
            torch.accelerator.synchronize(device)
            times = []
            for _ in range(args.replays):
                dist.barrier()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                times.append(start.elapsed_time(end) * 1e3)
            med = statistics.median(times[3:])
            t = torch.tensor([med], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            key = f"R{rows}_{variant}"
            results[key] = dict(
                rows=rows, n=rows // world, variant=variant, step_us=t.item(),
                per_layer_us=t.item() / args.layers,
            )
            if rank == 0:
                print(f"{key:>20}: graph {t.item():9.1f} us  "
                      f"{t.item() / args.layers:7.2f} us/layer", flush=True)
            if args.profile and variant in ("full", "fused", "fused_mx"):
                from torch.profiler import ProfilerActivity, profile

                dist.barrier()
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    for _ in range(3):
                        graph.replay()
                    torch.accelerator.synchronize(device)
                if rank == 0:
                    os.makedirs(args.profile, exist_ok=True)
                    stats = {}
                    for ev in prof.events():
                        if ev.device_type.name != "CUDA":
                            continue
                        s = stats.setdefault(ev.name[:90], [])
                        s.append(ev.device_time_total if hasattr(ev, "device_time_total") else ev.cuda_time_total)
                    lines = []
                    for name, ts in sorted(stats.items(), key=lambda kv: -sum(kv[1])):
                        lines.append(f"{sum(ts) / 3:9.1f} us/step  {len(ts) // 3:4d} calls  "
                                     f"{statistics.median(ts):7.2f} med  {name}")
                    with open(f"{args.profile}/R{rows}_{variant}_kernels.txt", "w") as f:
                        f.write("\n".join(lines) + "\n")
                    print("\n".join(lines[:12]), flush=True)
            del graph
    if rank == 0 and args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
