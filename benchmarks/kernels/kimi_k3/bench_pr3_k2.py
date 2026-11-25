# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PR-3 K2 microbench: KDA in_proj chain at the TP8 p2k / p4k shapes, stock
(MXFP8 gather + fused Q|K|V conv GEMM) vs overlapped (the gather pushed from the
GEMM's idle SMs while it runs), the production code paths, every rank.

torchrun --nnodes 2 --nproc-per-node 4 ... bench_pr3_k2.py --rows 2048
Prints per-rank medians (us) and the rank max; checks the overlapped q|k|v
planes and conv state bitwise against the stock ones every run.
"""

import argparse
import json
import os
import statistics

import torch

from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)

HIDDEN = 7168
P = 1536


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=2048)
    ap.add_argument("--nseq", type=int, default=8)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--skew-us", type=float, default=0.0)
    ap.add_argument("--form", default="conv", choices=["conv", "graph"],
                    help="conv: fused Q|K|V+conv GEMM (eager); graph: ag q|k|v GEMM + cuBLAS "
                    "g|f_a|beta vs one cuBLAS in_proj (the CUDA-graph path), captured")
    ap.add_argument("--out", default="")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    world = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    lr = int(os.environ["LOCAL_RANK"])
    config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=world))
    with set_current_vllm_config(config):
        dev = torch.device("cuda", lr)
        torch.accelerator.set_device_index(dev)
        init_distributed_environment()
        initialize_model_parallel(tensor_model_parallel_size=world)
        import torch.distributed as dist

        from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
        from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
            mxfp8_e4m3_quantize,
        )
        from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic
        from vllm.models.kimi_k3.nvidia.ops import sp_all_gather as sp_ag
        from vllm.models.kimi_k3.nvidia.specialized_prefill.kda_conv_in_proj import (
            _LocalAg,
            overlap_idle_sms,
            tactic_for,
        )
        from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm.kda_conv_gemm import (  # noqa: E501
            KdaConvGemm,
            ag_tactic,
            cluster_rows,
        )
        from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

        R, M = a.rows, a.rows * world
        assert sp_ag.init_sp_all_gather_buffer(R, HIDDEN, torch.bfloat16)
        gen = torch.Generator(device="cpu").manual_seed(11)
        w = (torch.randn(3 * P, HIDDEN, generator=gen) * 0.02).bfloat16().to(dev)
        wq, ws = mxfp8_e4m3_quantize(w, is_sf_swizzled_layout=True)
        conv_w = (torch.randn(3 * P, 4, generator=gen) * 0.5).to(dev)
        gemm = KdaConvGemm(wq, ws.view(torch.uint8).reshape(-1), conv_w, P,
                           tactic=tactic_for(3 * P))  # fmt: skip
        idle = overlap_idle_sms(dev, gemm.tactic)
        cl = cluster_rows(ag_tactic(gemm.tactic))
        g2 = torch.Generator(device="cpu").manual_seed(100 + rank)
        x = (torch.randn(R, HIDDEN, generator=g2)).bfloat16().to(dev)
        d, sf = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
        qa = QuantizedActivation(data=d, scale=sf, orig_dtype=torch.bfloat16,
                                 orig_shape=x.shape, quant_key=kMxfp8Dynamic)  # fmt: skip
        qsl = torch.tensor([i * M // a.nseq for i in range(a.nseq + 1)], dtype=torch.int32, device=dev)
        lines = torch.arange(a.nseq, dtype=torch.int32, device=dev)
        init = torch.zeros(a.nseq, dtype=torch.bool, device=dev)
        st0 = torch.zeros(a.nseq, 3 * P, 3, dtype=torch.bfloat16, device=dev)
        # compile both forms before any timing (the overlapped one with all
        # tiles local: no waits, no pushers)
        zero = torch.zeros(64, dtype=torch.int32, device=dev)
        sx0 = torch.full((M * HIDDEN // 32,), 127, dtype=torch.uint8, device=dev)
        x0 = torch.zeros(M, HIDDEN, dtype=torch.float8_e4m3fn, device=dev)
        gemm(x0, sx0, st0.clone(), qsl, lines, init, null_block_id=NULL_BLOCK_ID)
        gemm(x0, sx0, st0.clone(), qsl, lines, init, null_block_id=NULL_BLOCK_ID,
             ag=_LocalAg(zero.data_ptr(), M // cl))  # fmt: skip
        torch.accelerator.synchronize()
        sp_ag.enable_overlap(idle, cl)
        sp_ag.begin_forward(1 << 30)
        skew = int(a.skew_us * 2000) if rank == world - 1 else 0
        res = {"stock": [], "overlap": []}
        if a.form == "graph":
            from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm.kda_conv_gemm import (  # noqa: E501
                AgPlainGemm,
            )

            REST = 1680
            wr = (torch.randn(REST, HIDDEN, generator=gen) * 0.02).bfloat16().to(dev)
            wfull, sfull = mxfp8_e4m3_quantize(torch.cat([w, wr]), is_sf_swizzled_layout=True)
            wrq, wrs = mxfp8_e4m3_quantize(wr, is_sf_swizzled_layout=True)
            agp = AgPlainGemm(wq, ws.view(torch.uint8).reshape(-1), gemm.tactic)
            qkv0 = torch.empty(M, 3 * P, dtype=torch.bfloat16, device=dev)
            agp(x0, sx0, qkv0, _LocalAg(zero.data_ptr(), M // cl))
            e8 = torch.float8_e8m0fnu

            def stock_g():
                g = sp_ag.kimi_sp_all_gather_mxfp8(qa, M)
                return torch._scaled_mm(g.data, wfull.t(), scale_a=g.scale.view(e8),
                                        scale_b=sfull.view(e8), out_dtype=torch.bfloat16)  # fmt: skip

            def over_g():
                g = sp_ag.kimi_sp_all_gather_mxfp8_overlapped(qa, M, cl)
                qkv = torch.empty(M, 3 * P, dtype=torch.bfloat16, device=dev)
                agp(g.data, g.scale.view(torch.uint8).reshape(-1), qkv, sp_ag.take_pending(g))
                rest = torch._scaled_mm(g.data, wrq.t(), scale_a=g.scale.view(e8),
                                        scale_b=wrs.view(e8), out_dtype=torch.bfloat16)  # fmt: skip
                return torch.cat([qkv, rest], dim=-1)

            graphs = {}
            for name, fn in (("stock", stock_g), ("overlap", over_g)):
                for _ in range(2):
                    fn()
                torch.accelerator.synchronize()
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr):
                    outg = fn()
                graphs[name] = (gr, outg)
            torch.accelerator.synchronize()

        def one(mode):
            if a.form == "graph":
                gr, outg = graphs[mode]
                dist.barrier()
                torch.accelerator.synchronize()
                torch.cuda._sleep(4_000_000)
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e0.record()
                if skew:
                    torch.cuda._sleep(skew)
                gr.replay()
                e1.record()
                torch.accelerator.synchronize()
                return e0.elapsed_time(e1) * 1e3, outg.clone(), outg[:1].clone()

            st = st0.clone()
            dist.barrier()
            torch.accelerator.synchronize()
            torch.cuda._sleep(4_000_000)  # hold the GPU while the host enqueues
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            if skew:
                torch.cuda._sleep(skew)
            if mode == "stock":
                g = sp_ag.kimi_sp_all_gather_mxfp8(qa, M)
                out, _ = gemm(g.data, g.scale.view(torch.uint8).reshape(-1), st, qsl, lines,
                              init, null_block_id=NULL_BLOCK_ID)  # fmt: skip
            else:
                g = sp_ag.kimi_sp_all_gather_mxfp8_overlapped(qa, M, cl)
                assert g is not None
                out, _ = gemm(g.data, g.scale.view(torch.uint8).reshape(-1), st, qsl, lines,
                              init, null_block_id=NULL_BLOCK_ID,
                              ag=sp_ag.take_pending(g))  # fmt: skip
            e1.record()
            torch.accelerator.synchronize()
            return e0.elapsed_time(e1) * 1e3, out, st

        for _ in range(3):
            one("stock"), one("overlap")
        for i in range(a.iters):
            order = ("stock", "overlap") if i % 2 == 0 else ("overlap", "stock")
            outs = {}
            for m in order:
                t, out, st = one(m)
                res[m].append(t)
                outs[m] = (out, st)
            assert torch.equal(outs["stock"][0], outs["overlap"][0]), "planes differ"
            assert torch.equal(outs["stock"][1], outs["overlap"][1]), "conv state differs"
        med = {m: statistics.median(v) for m, v in res.items()}
        allm = [None] * world
        dist.all_gather_object(allm, med)
        if rank == 0:
            s = {m: [r[m] for r in allm] for m in med}
            line = (
                f"PR3K2 {a.tag} {a.form} W{world} R{R} M{M} idle{idle} cl{cl} skew{a.skew_us}: "
                f"stock med/max {statistics.median(s['stock']):.1f}/{max(s['stock']):.1f} us, "
                f"overlap {statistics.median(s['overlap']):.1f}/{max(s['overlap']):.1f} us, "
                f"delta(max) {max(s['overlap']) - max(s['stock']):+.1f} us; bitwise ok x{a.iters}"
            )
            print(line, flush=True)
            if a.out:
                with open(a.out, "a") as f:
                    f.write(json.dumps({"tag": a.tag, "W": world, "R": R, "skew_us": a.skew_us,
                                        "per_rank": allm, "line": line}) + "\n")  # fmt: skip
        dist.barrier()


if __name__ == "__main__":
    main()
