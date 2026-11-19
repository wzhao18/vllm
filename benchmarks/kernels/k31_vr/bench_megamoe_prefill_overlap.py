#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""DEP16 prefill MoE block on VR200: Rubin MegaMoE (NVFP4 W4A4) in token-capacity
sub-calls, alone and with the MXFP8 shared expert on a side stream.

Per-GPU shape of P3-16 (TP1 x DP16, EP16): T tokens per rank (16384 at F16K) go
through ``ceil(T / C)`` MegaMoE launches of capacity C (8192: the sub-call size),
and the shared expert (gate_up [12288, 7168] MXFP8, SiTU-and-mul, down
[7168, 6144] MXFP8; cuBLASLt ``_scaled_mm`` as KimiK3Mxfp8Kernel above 256 rows)
reads the same T rows. Modes (each a CUDA graph, max over ranks per sample):

  moe      MegaMoE sub-calls only
  shared   shared expert only (full GPU)
  serial   MegaMoE then shared on one stream (the stock DEP16 order)
  overlap  shared on a side stream forked before the first MegaMoE launch and
           issued after it, so the persistent grid takes its SMs first;
           ``--grid-caps`` sweeps the MegaMoE persistent-grid cap (SMs left to
           the side stream)

Run on one node with ``--num-experts 224`` (EP4 standing in for the EP16
per-rank shape: 56 local experts, top-16).
"""

import os

os.environ.setdefault("CUTE_DSL_ARCH", "sm_107a")

import argparse  # noqa: E402
import json  # noqa: E402
import statistics  # noqa: E402

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402


def init_distributed():
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local % torch.cuda.device_count())
    if world > 1:
        dist.init_process_group(
            "nccl", rank=rank, world_size=world,
            device_id=torch.device("cuda", torch.cuda.current_device()),
        )
    return rank, world


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--num-experts", type=int, default=896)
    p.add_argument("--tokens", type=int, default=16384, help="tokens per rank")
    p.add_argument("--capacity", type=int, default=8192, help="sub-call capacity")
    p.add_argument("--knobs", required=True, help="bare knob dict (JSON)")
    p.add_argument(
        "--grid-caps", default="0",
        help="comma list of persistent-grid caps in SMs; 0 = uncapped, "
        "negative = SM count minus that many",
    )
    p.add_argument("--modes", default="moe,shared,serial,overlap")
    p.add_argument(
        "--variant", choices=("inference", "inference_dedup"), default="inference"
    )
    p.add_argument(
        "--routing", choices=("random", "local"), default="random",
        help="local: every route to this rank's own experts (no NVLink payload; "
        "the compute-only floor of the layer)",
    )
    p.add_argument("--hidden", type=int, default=7168)
    p.add_argument("--shared-inter", type=int, default=6144)
    p.add_argument("--samples", type=int, default=10)
    p.add_argument("--replays", type=int, default=3)
    p.add_argument("--profile-dir", default=None)
    p.add_argument("--json", default=None)
    return p.parse_args()


def measure(fn, samples, replays, group):
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    vals = []
    for _ in range(samples):
        if group is not None:
            dist.barrier(group)
        torch.cuda._sleep(2_000_000)
        start.record()
        for _ in range(replays):
            fn()
        end.record()
        end.synchronize()
        vals.append(start.elapsed_time(end) * 1000.0 / replays)
    t = torch.tensor(vals, dtype=torch.float64, device="cuda")
    if group is not None:
        dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group)
    v = t.tolist()
    return {"p50_us": statistics.median(v), "min_us": min(v), "max_us": max(v)}


def mxfp8_weight(n, k, gen):
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
    )

    w = (torch.randn((n, k), generator=gen, device="cuda") * k**-0.5).to(
        torch.bfloat16
    )
    q, s = mxfp8_e4m3_quantize(w, is_sf_swizzled_layout=True)
    return q.t(), s.view(torch.float8_e8m0fnu)  # [K, N] column-major, as the layer


def main():
    args = parse_args()
    rank, world = init_distributed()
    group = dist.group.WORLD if world > 1 else None
    dev = torch.device("cuda", torch.cuda.current_device())
    sms = torch.cuda.get_device_properties(dev).multi_processor_count

    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
    )
    from vllm.models.kimi_k3.nvidia.megamoe_rubin import weights as W
    from vllm.models.kimi_k3.nvidia.megamoe_rubin.config import (
        RubinBucket,
        RubinKnobs,
        RubinMoeProblem,
    )
    from vllm.models.kimi_k3.nvidia.megamoe_rubin.runtime import RubinMegaMoE

    problem = RubinMoeProblem(num_experts=args.num_experts, quant_kind="nvfp4")
    knobs = RubinKnobs.from_dict(json.loads(args.knobs))
    T, C = args.tokens, args.capacity
    nsub = -(-T // C)
    gen = torch.Generator(device=dev).manual_seed(1234 + rank)

    # routed-expert weights (canonical MXFP4 -> exact NVFP4), deterministic per expert
    e_local = problem.num_experts // world
    hdim, inter = problem.hidden, problem.intermediate
    u8 = dict(dtype=torch.uint8, device=dev)
    w13 = torch.empty((e_local, 2 * inter, hdim // 2), **u8)
    s13 = torch.empty((e_local, 2 * inter, hdim // 32), **u8)
    w2 = torch.empty((e_local, hdim, inter // 2), **u8)
    s2 = torch.empty((e_local, hdim, inter // 32), **u8)
    for e in range(e_local):
        g = torch.Generator(device=dev).manual_seed(7 + rank * e_local + e)
        a = torch.randn((2 * inter, hdim), generator=g, device=dev) * hdim**-0.5
        w13[e], s13[e] = W.quantize_mxfp4_block32(a)
        b = torch.randn((hdim, inter), generator=g, device=dev) * inter**-0.5
        w2[e], s2[e] = W.quantize_mxfp4_block32(b)

    # inputs: latent rows for the MegaMoE, hidden rows for the shared expert
    lat = torch.randn((T, hdim), generator=gen, device=dev).to(torch.bfloat16)
    e_loc = problem.num_experts // world
    if args.routing == "local":
        scores = torch.rand((T, e_loc), generator=gen, device=dev)
        ids = scores.topk(problem.top_k, dim=1).indices + rank * e_loc
    else:
        scores = torch.rand((T, problem.num_experts), generator=gen, device=dev)
        ids = scores.topk(problem.top_k, dim=1).indices
    ids = ids.to(torch.int32).contiguous()
    wts = torch.rand((T, problem.top_k), generator=gen, device=dev) + 0.05
    wts = (wts / wts.sum(1, keepdim=True)).float().contiguous()
    out = torch.empty((T, hdim), dtype=torch.bfloat16, device=dev)
    x = torch.randn((T, args.hidden), generator=gen, device=dev).to(torch.bfloat16)
    sw13, sw13_s = mxfp8_weight(2 * args.shared_inter, args.hidden, gen)
    sw2, sw2_s = mxfp8_weight(args.hidden, args.shared_inter, gen)
    from vllm.models.kimi_k3.nvidia.ops import mxfp8_act

    def shared():
        q, s = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
        gu = torch._scaled_mm(
            q, sw13, scale_a=s.view(torch.float8_e8m0fnu), scale_b=sw13_s,
            out_dtype=torch.bfloat16,
        )
        # KimiMLP._act_mxfp8: one fused SiTU-and-mul + MXFP8 quant (down_proj input)
        qa = mxfp8_act.situ_and_mul_quantize(
            gu, problem.situ_beta, problem.situ_linear_beta
        )
        return torch._scaled_mm(
            qa.data, sw2, scale_a=qa.scale.view(torch.float8_e8m0fnu), scale_b=sw2_s,
            out_dtype=torch.bfloat16,
        )

    side = torch.cuda.Stream()
    caps = []
    for c in args.grid_caps.split(","):
        c = int(c)
        caps.append(None if c == 0 else (sms + c if c < 0 else c))
    modes = args.modes.split(",")
    rows = []
    for cap in caps:
        bucket = RubinBucket(
            args.variant, C,
            knobs if cap is None else RubinKnobs.from_dict(
                {**knobs.as_dict(), "fallback_cluster_shape_mn": None}
            ),
            max_sm_count=cap,
        )
        moe = RubinMegaMoE(problem, group, [bucket])
        weights = moe.prepare_weights(w13, s13, w2, s2, layer_id=0)
        moe.warmup(weights, ids_dtypes=(torch.int32,), weight_dtypes=(torch.float32,))

        def moe_calls():
            for i in range(nsub):
                lo, hi = i * C, min(T, (i + 1) * C)
                moe.forward(
                    lat[lo:hi], ids[lo:hi], wts[lo:hi], weights,
                    bucket=bucket, out=out[lo:hi], ep_tokens=hi - lo,
                )

        def serial():
            moe_calls()
            shared()

        def overlap():
            main = torch.cuda.current_stream()
            side.wait_stream(main)
            moe_calls()  # host issues the persistent grids first
            with torch.cuda.stream(side):
                shared()
            main.wait_stream(side)

        fns = {"moe": moe_calls, "shared": shared, "serial": serial, "overlap": overlap}
        ref = fp = None
        for mode in modes:
            if cap is not None and mode == "shared":
                continue
            fn = fns[mode]
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            if group is not None:
                dist.barrier(group)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fn()
            graph.replay()
            torch.cuda.synchronize()
            if mode in ("moe", "serial", "overlap"):
                fp = out.float().abs().sum().item()
                ref = fp if ref is None else ref
            res = measure(graph.replay, args.samples, args.replays, group)
            prof = None
            if args.profile_dir and mode in ("moe", "overlap"):
                # every rank replays (the MegaMoE is collective); rank 0 records
                from torch.profiler import ProfilerActivity, profile

                if group is not None:
                    dist.barrier(group)
                with profile(activities=[ProfilerActivity.CUDA]) as pr:
                    graph.replay()
                    torch.cuda.synchronize()
                if rank == 0:
                    os.makedirs(args.profile_dir, exist_ok=True)
                    pr.export_chrome_trace(
                        os.path.join(args.profile_dir, f"{mode}_cap{cap or 0}.json")
                    )
                    tot = {}
                    for ev in pr.events():
                        if ev.device_type == torch.autograd.DeviceType.CUDA:
                            k = ev.name[:90]
                            tot[k] = tot.get(k, 0.0) + ev.device_time
                    prof = dict(sorted(tot.items(), key=lambda kv: -kv[1])[:12])
            if group is not None:
                dist.barrier(group)
            row = {
                "mode": mode, "variant": args.variant, "grid_cap": cap, "sms": sms,
                "tokens": T, "capacity": C,
                "subcalls": nsub, "world": world, "routing": args.routing,
                "num_experts": problem.num_experts,
                "out_abs_sum": fp if mode != "shared" else None, **res,
                "kernels_us": prof,
            }
            rows.append(row)
            if rank == 0:
                print(json.dumps(row), flush=True)
                if args.json:
                    with open(args.json, "a") as f:
                        f.write(json.dumps(row) + "\n")
            del graph
        torch.cuda.synchronize()
        moe.destroy()
        del moe, weights
        torch.cuda.empty_cache()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
