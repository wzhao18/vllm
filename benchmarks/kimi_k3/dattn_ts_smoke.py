# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""D-ATTN: smoke run of dsa.sparse_mla_ts (progress printed per step: compile, launch,
sync) at a few row counts; with --knobs for diagnostics."""

import argparse
import sys
import time

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", default="4,128")
    p.add_argument("--local", type=int, default=512)
    p.add_argument("--knobs", default="")
    p.add_argument("--reps", type=int, default=50)
    a = p.parse_args()
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.sparse_mla_ts import (
        _compiled,
        sparse_mla_decode_ts,
    )

    knobs = tuple(int(v) for v in a.knobs.split(",") if v)
    dev = "cuda"
    t0 = time.time()
    _compiled(torch.cuda.get_device_properties(0).multi_processor_count, knobs)
    print(f"compiled {time.time() - t0:.1f}s", flush=True)
    gen = torch.Generator(device=dev).manual_seed(0)
    ctx = 8192
    for rows in [int(r) for r in a.rows.split(",")]:
        reqs = -(-rows // 4)
        kv = (torch.randn(reqs * ctx, 576, device=dev, generator=gen) * 0.5).to(
            torch.float8_e4m3fn)
        q = (torch.randn(rows, 128, 576, device=dev, generator=gen) * 0.5).to(
            torch.float8_e4m3fn)
        q[:, 96:] = 0
        slots = torch.full((rows, 2048), -1, dtype=torch.int32, device=dev)
        valid = torch.zeros(rows, dtype=torch.int32, device=dev)
        for r in range(rows):
            n = a.local if r % 5 else a.local // 3
            if r == rows - 1 and rows > 4:
                n = 0
            slots[r, :n] = (torch.randperm(ctx, generator=gen, device=dev)[:n]
                            + (r // 4) * ctx).int()
            valid[r] = n
        lat = torch.zeros(rows, 128, 512, dtype=torch.bfloat16, device=dev)
        lse = torch.zeros(rows, 128, dtype=torch.float32, device=dev)
        torch.cuda.synchronize()
        print(f"R{rows} launch", flush=True)
        sparse_mla_decode_ts(q.view(torch.uint8), kv.view(torch.uint8), slots, valid, lat,
                             lse, 0.07, 0.8, knobs=knobs)
        print(f"R{rows} launched", flush=True)
        torch.cuda.synchronize()
        print(f"R{rows} done: lse finite {bool(torch.isfinite(lse[:-1, :96]).all())}, "
              f"latent nan {bool(torch.isnan(lat).any())}", flush=True)
        for it in range(a.reps):
            sparse_mla_decode_ts(q.view(torch.uint8), kv.view(torch.uint8), slots, valid,
                                 lat, lse, 0.07, 0.8, knobs=knobs)
        torch.cuda.synchronize()
        print(f"R{rows} {a.reps} reps done", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
