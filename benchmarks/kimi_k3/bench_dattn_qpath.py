# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""D-ATTN q-path microbench (1 GPU, SP decode shapes: TP4 = DCP4, 24 heads / rank,
q4, R = 4 x requests per DP rank rows).

Arms (us per layer, cold L2: a 1 GiB write between iterations):
  mlaquery      today's 24-head MlaQuery (fork mode, no publish: a lower bound of the
                in-step kernel, which also multicasts 24 heads to 4 ranks)
  qb24/qb96     cuBLAS q_c [R,1536] x W_qb^T (+ the indexer wq_b 4096 columns) for
                24 or all 96 heads
  uk24/uk96     torch.bmm of q_nope with W_UK_T for 24 / 96 heads
  pack96        BF16 latent + rope -> FP8 [R, 128, 576] (torch ops)
"""

import argparse
import json

import torch

HEADS, PITCH, UNIT, FP8_ROW = 24, 5344, 64, 576


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


def mlaquery_arm(rows, flush):
    import cutlass

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        fake,
        layout,
        stream,
        to_cute,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.query_stage import (  # noqa: E501
        compile_query,
    )

    bf, dev = torch.bfloat16, "cuda"
    projected = (torch.randn(rows, PITCH, device=dev) * 3).to(bf)
    weights = (
        (torch.rand(1536, device=dev) + 0.5).to(bf),
        (torch.rand(512, device=dev) + 0.5).to(bf),
        (torch.randn(HEADS * 192, 1536, device=dev) / 40).to(bf),
        (torch.randn(HEADS, 128, 512, device=dev) / 12).to(bf),
    )
    blocks = -(-rows // UNIT) + 2
    slots = torch.randperm(blocks * UNIT, device=dev)[:rows]
    slots[torch.arange(rows, device=dev) % 4 != 0] = -1  # DCP4: 1/4 owned
    mla = torch.zeros(blocks, UNIT, FP8_ROW, dtype=torch.uint8, device=dev)
    specs = tuple(layout(t) for t in (*weights, mla))
    fakes = tuple(
        fake(shape, stride, dtype, 4 if dtype == torch.float32 else 16)
        for shape, stride, dtype in specs
    )
    kernel = compile_query(
        rows, HEADS, PITCH, fakes, torch.cuda.current_device(), fp8=True,
        overlap="fork",
    )  # fmt: skip
    query = torch.zeros(rows, HEADS, 576, device=dev, dtype=torch.uint8)
    q_c = torch.zeros((rows, 1536), device=dev, dtype=bf)
    one = torch.ones(1, device=dev)
    args = (to_cute(projected), *(to_cute(w) for w in weights), to_cute(mla))
    args += (to_cute(slots, 8), to_cute(one, 4), to_cute(one, 4))
    args += (cutlass.Float32(1e-6), to_cute(query), to_cute(q_c), stream(dev))
    return timed(lambda: kernel(*args), flush=flush)


def torch_arms(rows, flush):
    bf, dev = torch.bfloat16, "cuda"
    out = {}
    q_c = torch.randn(rows, 1536, device=dev).to(bf)
    for h in (24, 96):
        w = (torch.randn(h * 192 + 4096, 1536, device=dev) / 40).to(bf)
        y = torch.empty(rows, h * 192 + 4096, device=dev, dtype=bf)
        out[f"qb{h}"] = timed(lambda: torch.mm(q_c, w.t(), out=y), flush=flush)
        q = y[:, : h * 192].view(rows, h, 192)
        uk = (torch.randn(h, 128, 512, device=dev) / 12).to(bf)
        lat = torch.empty(h, rows, 512, device=dev, dtype=bf)
        nope = q[..., :128].transpose(0, 1)
        out[f"uk{h}"] = timed(lambda: torch.bmm(nope, uk, out=lat), flush=flush)
        if h == 96:
            qbuf = torch.zeros(rows, 128, 576, device=dev, dtype=torch.float8_e4m3fn)
            inv = torch.full((1,), 2.0, device=dev)

            def pack():
                qbuf[:, :96, :512].copy_(
                    (lat.transpose(0, 1).float() * inv).to(torch.float8_e4m3fn)
                )
                qbuf[:, :96, 512:].copy_(
                    (q[..., 128:].float() * inv).to(torch.float8_e4m3fn)
                )

            out["pack96_torch"] = timed(pack, flush=flush)
    return out


def local_arm(rows_list, flush):
    """dsa.local_query.LocalQuery.run end to end (96 heads, one GPU, no gathers)."""
    import sys
    from pathlib import Path

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.local_query import (  # noqa: E501
        LocalQuery,
    )

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
    from models.kimi_k3.test_specialized_decode_sp_local_query_gpu import (
        context,
        stub_attention,
    )

    dev, bf = "cuda", torch.bfloat16
    gen = torch.Generator(device=dev).manual_seed(0)
    local = LocalQuery(stub_attention(gen), None, 1, rows_list, context(), 128)
    out = {}
    for rows in rows_list:
        projected = (torch.randn(rows, PITCH, device=dev) * 3).to(bf)
        q_c = torch.randn(rows, 1536, device=dev).to(bf)
        blocks = -(-rows // UNIT) + 2
        slots = torch.randperm(blocks * UNIT, device=dev)[:rows]
        slots[torch.arange(rows, device=dev) % 4 != 0] = -1
        cache = torch.zeros(blocks, UNIT, FP8_ROW, dtype=torch.uint8, device=dev)
        out[rows] = (
            timed(lambda: local.run(q_c, projected, cache, slots), flush=flush),
            local.projection.kinds[rows],
        )
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", default="128,256,384,512")
    p.add_argument("--out", default="")
    a = p.parse_args()
    flush = torch.empty(1 << 28, device="cuda", dtype=torch.float32)
    res = {}
    rows_list = [int(r) for r in a.rows.split(",")]
    local = local_arm(rows_list, flush)
    for rows in rows_list:
        r = {"mlaquery": mlaquery_arm(rows, flush)}
        r.update(torch_arms(rows, flush))
        r["local96"], r["local96_tactic"] = local[rows]
        res[rows] = r
        print(rows, json.dumps({k: round(v, 2) if isinstance(v, float) else v for k, v in r.items()}), flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
