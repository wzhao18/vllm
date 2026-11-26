# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""8-rank microbench of the Kimi K3 BF16 SP all-gather transports
(``vllm/models/kimi_k3/nvidia/ops/sp_all_gather.py``), one task per GPU
(torchrun), no vLLM init: a symmetric two-slot buffer of world x rows x 7168
BF16 like the production one, then per mode the gather of each rank's shard
(rows x 7168 BF16):

  tri16  Triton push on 16-bit elements + symm barrier (the stock DSA gather)
  tri32  the same Triton push on 32-bit words + symm barrier
  tma    TMA bulk push with counted completion (no barrier)

Time = CUDA events around the gather after a symm barrier (ranks leave it
within ~1-2 us), median / max over iterations, rank median and rank max.
--skew-us delays the last rank's gather by a GPU sleep. Every mode's gathered
slot is checked bitwise against NCCL all_gather_into_tensor. Writes one JSON
line per mode to --out (rank 0)."""

import argparse
import json
import statistics

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

H = 7168


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=2048)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--modes", default="tri16,tri32,tma")
    ap.add_argument("--skew-us", type=float, default=0.0)
    ap.add_argument("--graph", action="store_true", help="also time the tma gather replayed from a CUDA graph")
    ap.add_argument("--out", default="")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    import os

    dist.init_process_group("nccl")
    W, me = dist.get_world_size(), dist.get_rank()
    lr = int(os.environ.get("LOCAL_RANK", me % torch.cuda.device_count()))
    torch.cuda.set_device(lr)
    dev = torch.device("cuda", lr)
    from vllm.models.kimi_k3.nvidia.ops import sp_all_gather as m

    R = args.rows
    slot_numel = W * R * H
    storage = symm_mem.empty(2 * slot_numel, dtype=torch.bfloat16, device=dev)
    handle = symm_mem.rendezvous(storage, dist.group.WORLD.group_name)
    bases = [handle.get_buffer(p, [2 * slot_numel], torch.bfloat16, 0).data_ptr() for p in range(W)]
    peer_ptrs = torch.tensor([[b + s * slot_numel * 2 for b in bases] for s in range(2)], dtype=torch.int64, device=dev)
    pads, pad_size = handle.signal_pad_ptrs, handle.signal_pad_size
    ctr_ptrs = torch.tensor([[p + pad_size - 4 * (2 - s) for p in pads] for s in range(2)], dtype=torch.int64, device=dev)
    nsm = torch.cuda.get_device_properties(dev).multi_processor_count
    ext = m._tma_module()
    ext.sp_ag_tma_push(storage[:0], peer_ptrs[0], 0, ctr_ptrs[0], me, W, m._TMA_CHUNK, 1, m._TMA_SPIN_NS)
    torch.cuda.synchronize()
    handle.barrier()

    gen = torch.Generator(device="cpu").manual_seed(1234 + me)
    x = torch.randn(R, H, generator=gen).bfloat16().to(dev)
    ref = torch.empty(W * R, H, dtype=torch.bfloat16, device=dev)
    dist.all_gather_into_tensor(ref, x)
    nbytes = R * H * 2
    sleep_cycles = int(args.skew_us * 1.9e3) if me == W - 1 else 0  # ~1.9 GHz

    def gather(mode, slot):
        if mode == "tma":
            ext.sp_ag_tma_push(x, peer_ptrs[slot], me * nbytes, ctr_ptrs[slot], me, W, m._TMA_CHUNK,
                               min(nsm, -(-nbytes // m._TMA_CHUNK) * W), m._TMA_SPIN_NS)
            return
        if mode == "tri16":
            src, blk = x.reshape(-1), m._BLOCK
        else:
            src, blk = x.view(torch.int32).reshape(-1), m._WORD_BLOCK
        n = src.numel()
        m._push_shard_kernel[(min(nsm, -(-n // blk)),)](
            src, peer_ptrs[slot], me * n, n, me, WORLD=W, BLOCK=blk, num_warps=m._NUM_WARPS)
        handle.barrier()

    out = {}
    for mode in args.modes.split(","):
        times = []
        for it in range(args.iters + 3):
            slot = it % 2
            storage[slot * slot_numel : (slot + 1) * slot_numel].zero_()
            torch.cuda.synchronize()
            handle.barrier()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            if sleep_cycles:
                torch.cuda._sleep(sleep_cycles)
            e0.record()
            gather(mode, slot)
            e1.record()
            torch.cuda.synchronize()
            if it >= 3:
                times.append(e0.elapsed_time(e1) * 1e3)
            got = storage[slot * slot_numel : (slot + 1) * slot_numel].view(W * R, H)
            ok = torch.equal(got.view(torch.int16), ref.view(torch.int16))
            okt = torch.tensor([int(ok)], device=dev)
            dist.all_reduce(okt, op=dist.ReduceOp.MIN)
            assert okt.item() == 1, f"{mode}: gathered slot differs from NCCL (rank {me} ok={ok})"
        med = statistics.median(times)
        allm = [None] * W
        dist.all_gather_object(allm, (med, max(times)))
        out[mode] = allm
        if me == 0:
            meds = [a[0] for a in allm]
            ingress = (W - 1) * nbytes / 1e6
            rec = dict(tag=args.tag, rows=R, mode=mode, skew_us=args.skew_us, rank_median_us=statistics.median(meds),
                       rank_max_us=max(meds), worst_iter_us=max(a[1] for a in allm), per_rank_med=meds,
                       ingress_mb=ingress, tbps=ingress / statistics.median(meds))
            print(f"PR3B {args.tag} rows {R} {mode}: rank med {rec['rank_median_us']:.1f} max {rec['rank_max_us']:.1f} "
                  f"worst {rec['worst_iter_us']:.1f} us  {ingress / rec['rank_median_us']:.3f} TB/s ingress  bitwise ok", flush=True)
            if args.out:
                with open(args.out, "a") as f:
                    f.write(json.dumps(rec) + "\n")
    if args.graph and "tma" in args.modes:
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        handle.barrier()
        with torch.cuda.stream(s):
            gather("tma", 0)  # warm on the side stream
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        handle.barrier()
        with torch.cuda.graph(g):
            gather("tma", 0)
            gather("tma", 1)
        times = []
        for it in range(args.iters):
            x.copy_(torch.randn(R, H, generator=gen).bfloat16().to(dev))
            dist.all_gather_into_tensor(ref, x)
            torch.cuda.synchronize()
            handle.barrier()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            g.replay()
            e1.record()
            torch.cuda.synchronize()
            times.append(e0.elapsed_time(e1) * 1e3 / 2)
            for slot in (0, 1):
                got = storage[slot * slot_numel : (slot + 1) * slot_numel].view(W * R, H)
                assert torch.equal(got.view(torch.int16), ref.view(torch.int16)), f"graph replay slot {slot} differs"
        if me == 0:
            print(f"PR3B {args.tag} rows {R} tma-graph: {statistics.median(times):.1f} us per gather, "
                  f"{args.iters} replays bitwise ok", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
