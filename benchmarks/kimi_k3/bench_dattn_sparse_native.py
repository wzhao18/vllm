# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""D-ATTN: native sparse MLA decode (dsa.sparse_mla) vs FlashInfer trtllm-gen at the
D3-32 per-GPU shapes (R rows, 128-head q mailbox rows with 96 real heads, ~512 local
keys per row of a DCP4 shard, per-tensor FP8 KV). Checks both against an FP32
reference on a subset of rows (latent and LSE), then times both (cold L2).

  python3 benchmarks/kimi_k3/bench_dattn_sparse_native.py --rows 128,256,384,512
"""

import argparse
import json
import math

import torch


def timed(fn, iters=30, flush=None):
    times = []
    for _ in range(iters + 3):
        if flush is not None:
            flush.zero_()
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) * 1e3)
    times = sorted(times[3:])
    return times[len(times) // 2]


def make_case(rows, local, jitter, ctx_rows, gen, width=1024, empty=0):
    dev = "cuda"
    q = 4
    reqs = -(-rows // q)
    slots = torch.full((rows, width), -1, dtype=torch.int32, device=dev)
    valid = torch.empty(rows, dtype=torch.int32, device=dev)
    for r in range(rows):
        n = local
        if jitter:
            n = int(torch.randint(local - jitter, local + jitter + 1, (1,), generator=gen,
                                  device=dev).item())
        if r < empty:
            n = 0
        n = max(0, min(n, width))
        idx = torch.randperm(ctx_rows, generator=gen, device=dev)[:n]
        slots[r, :n] = (idx + (r // q) * ctx_rows).int()
        valid[r] = n
    return slots, valid


def reference(query, kv, slots, valid, rows_check, bmm1, bmm2, real=96):
    """FP32 exact attention of ``rows_check`` rows -> latent [k, 96, 512], lse2."""
    outs, lses = [], []
    for r in rows_check:
        n = int(valid[r])
        k = kv[slots[r, :n].long()].float()  # [n, 576]
        qf = query[r, :real].float()  # [96, 576]
        s = (qf @ k.T) * bmm1  # natural-log logits
        lse = torch.logsumexp(s, dim=-1) * (1 / math.log(2))
        p = torch.softmax(s, dim=-1)
        outs.append((p @ k[:, :512]) * bmm2)
        lses.append(lse)
    return torch.stack(outs), torch.stack(lses)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", default="128,256,384,512")
    p.add_argument("--local", type=int, default=512)
    p.add_argument("--jitter", type=int, default=40)
    p.add_argument("--ctx", type=int, default=135000 // 4)
    p.add_argument("--stages", type=int, default=3)
    p.add_argument("--clusters", type=int, default=0)
    p.add_argument("--stamps", action="store_true")
    p.add_argument("--loaders", type=int, default=6)
    p.add_argument("--no-mc", action="store_true")
    p.add_argument("--v1", action="store_true", help="the V-split pair kernel")
    p.add_argument("--check", type=int, default=24)
    p.add_argument("--no-trtllm", action="store_true")
    p.add_argument("--out", default="")
    a = p.parse_args()
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.sparse_mla import (
        sparse_mla_decode,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.sparse_mla_km import (
        sparse_mla_decode_km,
    )

    def native(query, kv, slots, valid, lat, lse, bmm1, bmm2, stamps=None):
        if a.v1:
            return sparse_mla_decode(query, kv, slots, valid, lat, lse, bmm1, bmm2,
                                     stages=a.stages, clusters=a.clusters or None,
                                     loaders=a.loaders, multicast=not a.no_mc,
                                     stamps=stamps)
        return sparse_mla_decode_km(query, kv, slots, valid, lat, lse, bmm1, bmm2,
                                    grid=a.clusters or None, stages=a.stages,
                                    loaders=a.loaders, stamps=stamps)
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import _kv_counters

    dev, heads = "cuda", 128
    gen = torch.Generator(device=dev).manual_seed(0)
    flush = torch.empty(1 << 28, device=dev, dtype=torch.float32)
    rows_list = [int(r) for r in a.rows.split(",")]
    reqs_max = -(-max(rows_list) // 4)
    kv = (torch.randn(reqs_max * a.ctx, 576, device=dev, generator=gen) * 0.5).to(
        torch.float8_e4m3fn)
    pages = kv.view(-1, 64, 576)
    workspace = torch.zeros(256 << 20, dtype=torch.uint8, device=dev)
    bmm1, bmm2 = 0.07, 0.8
    res = {}
    for rows in rows_list:
        query = (torch.randn(rows, heads, 576, device=dev, generator=gen) * 0.5).to(
            torch.float8_e4m3fn)
        query[:, 96:] = 0
        slots, valid = make_case(rows, a.local, a.jitter, a.ctx, gen, empty=2)
        lat_t = torch.zeros(rows, heads, 512, dtype=torch.bfloat16, device=dev)
        lse_t = torch.zeros(rows, heads, dtype=torch.float32, device=dev)
        lat_n = torch.zeros_like(lat_t)
        lse_n = torch.zeros_like(lse_t)
        extra = _kv_counters(rows, heads, query.device)
        extra["lse"] = lse_t

        def trt():
            trtllm_batch_decode_with_kv_cache_mla(
                query=query.unsqueeze(1), kv_cache=pages.unsqueeze(1),
                workspace_buffer=workspace, qk_nope_head_dim=128, kv_lora_rank=512,
                qk_rope_head_dim=64, block_tables=slots.unsqueeze(1), seq_lens=valid,
                max_seq_len=slots.shape[1], out=lat_t.unsqueeze(1), bmm1_scale=bmm1,
                bmm2_scale=bmm2, sparse_mla_top_k=slots.shape[1], return_lse=True,
                **extra)

        def nat():
            native(query, kv.view(torch.uint8), slots, valid, lat_n, lse_n, bmm1, bmm2)

        nat()
        torch.cuda.synchronize()
        out = {}
        if not a.no_trtllm:
            trt()
            torch.cuda.synchronize()
        check = [r for r in range(min(rows, a.check)) if int(valid[r]) > 0]
        ref_o, ref_l = reference(query, kv, slots, valid, check, bmm1, bmm2)
        for name, lat, lse in (("native", lat_n, lse_n), ("trtllm", lat_t, lse_t)):
            if name == "trtllm" and a.no_trtllm:
                continue
            o = lat[check, :96].float()
            err = (o - ref_o).abs()
            out[name] = dict(
                max_abs=round(err.max().item(), 5),
                mean_abs=round(err.mean().item(), 6),
                ref_mean_abs=round(ref_o.abs().mean().item(), 5),
                lse_max_abs=round((lse[check, :96] - ref_l).abs().max().item(), 5),
                nan=bool(torch.isnan(lat[:, :96]).any().item()),
            )
        if not a.no_trtllm:
            nz = valid > 0
            d = (lat_n[nz, :96].float() - lat_t[nz, :96].float()).abs()
            out["native_vs_trtllm"] = dict(
                max_abs=round(d.max().item(), 5), mean_abs=round(d.mean().item(), 6),
                lse_max_abs=round((lse_n[nz, :96] - lse_t[nz, :96]).abs().max().item(),
                                  5),
            )
        out["native_us"] = round(timed(nat, flush=flush), 2)
        if not a.v1:
            from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.sparse_mla_km import (  # noqa: E501
                kernel_info,
            )

            grid = a.clusters or min(torch.cuda.get_device_properties(0).multi_processor_count, rows)
            out["regs_local"] = kernel_info(0, grid, a.stages, a.loaders)
        if a.stamps:
            st = torch.zeros(424 * 64, dtype=torch.int64, device=dev)
            flush.zero_()
            native(query, kv.view(torch.uint8), slots, valid, lat_n, lse_n, bmm1, bmm2,
                   stamps=st)
            torch.cuda.synchronize()
            st = st.view(424, 64)
            used = st[:, 0] > 0
            st = st[used]
            t0 = st[:, 0].min()
            rel = (st - t0).float() / 1e3
            rel[st == 0] = float("nan")
            med = rel.nanmedian(dim=0).values
            mx = torch.where(torch.isnan(rel), torch.full_like(rel, -1), rel).max(0).values
            names = {0: "start", 1: "setup", 40: "mma_q0", 41: "qk0_issued", 31: "t1_a_ld", 32: "t1_b_redux", 33: "t1_c_bar1", 34: "t1_d_bar2", 35: "t1_e_exp", 36: "t1_f_pvwait", 37: "t1_g_resc",
                     44: "mma_q1", 45: "qk1_issued"}
            for ri in range(2):
                names[2 + 16 * ri] = f"r{ri}_qfull"
                for t in range(10):
                    names[3 + 16 * ri + t] = f"r{ri}_S{t}"
                names[13 + 16 * ri] = f"r{ri}_Odone"
                names[14 + 16 * ri] = f"r{ri}_epi"
            for t in range(8):
                names[48 + t] = f"r0_kv{t}"
                names[56 + t] = f"r0_p{t}"
            out["stamps_us_median_max"] = {
                names[i]: (round(med[i].item(), 2), round(mx[i].item(), 2))
                for i in sorted(names) if not torch.isnan(med[i])}
        if not a.no_trtllm:
            out["trtllm_us"] = round(timed(trt, flush=flush), 2)
        out["gathered_MB"] = round(int(valid.sum()) * 576 / 1e6, 1)
        res[f"R{rows}"] = out
        print(f"R{rows}", json.dumps(out), flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
