"""D-GEMM microbench: the decode dense GEMM family at D3-32 per-GPU shapes on one VR200 GPU.

Every point: a CUDA graph of C back-to-back calls, each on its own cold weight copy (C copies > 1.5 GB, i.e. well
past L2), replayed; us per call = replay time / C. Same activation for every call (L2-resident, as in-step).
BF16 impls: table (sdec DecodeProjection with sp_policy's tactic), cublas (torch.mm), lt (torch.mm, cuBLASLt pref).
MXFP8 impls (pre-quantized input): fi_cute / fi_cutlass / fi_trtllm (flashinfer.mm_mxfp8 backends), scaled_mm
(cuBLASLt torch._scaled_mm), plus the activation quant alone: q_k3 (ops/mxfp8_act.quantize), q_fi (mxfp8_quantize).
usage: python gemm_bench.py out.json [family,...]"""
import json
import math
import sys
import traceback

import torch

DEV = torch.device("cuda", 0)
torch.cuda.set_device(DEV)

# family: (N, K, rows kind, dtypes)
FAMS = {
    "kda_in": (12448, 7168, "R", ("bf16", "mxfp8")),
    "kda_o": (7168, 3072, "R", ("bf16", "mxfp8")),
    "f_b": (3072, 128, "R", ("bf16",)),
    "dsa_in": (5344, 7168, "R", ("bf16",)),
    "q_b": (4608, 1536, "R", ("bf16",)),
    "wq_b": (4096, 1536, "R", ("bf16",)),
    "dsa_o": (7168, 3072, "R", ("bf16",)),
    "router": (896, 7168, "n", ("bf16",)),
    "latent_down": (3584, 7168, "n", ("bf16",)),
    "latent_up": (7168, 3584, "n", ("bf16",)),
    "shared_gu": (12288, 7168, "n", ("bf16", "mxfp8")),
    "shared_down": (7168, 6144, "n", ("bf16", "mxfp8")),
}
TABLE_FAMILY = {"kda_in": "kda_in", "kda_o": "o_proj", "dsa_in": "dsa_in", "q_b": "q_b", "wq_b": "wq_b",
                "dsa_o": "o_proj", "latent_down": "latent_down", "shared_gu": "shared_gu",
                "shared_down": "shared_down"}
R_ROWS = (16, 128, 256, 384)
N_ROWS = (4, 32, 64, 96)


def ctx():
    from vllm.models.kimi_k3.nvidia.specialized_decode.shapes import Context
    sms = torch.cuda.get_device_properties(DEV).multi_processor_count
    return Context(arch="rubin", tp=4, rank=0, dcp=4, sms=sms, q=4, max_rows=512, max_context=131072,
                   bf16_state=False, dp=8, ep=32, sequence_parallel=True, heads_per_rank=24)


def timed_chain(calls, reps=20):
    """calls: list of zero-arg callables (one per cold weight copy). Returns us per call."""
    for c in calls[:2]:
        c()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for c in calls:
            c()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        for c in calls:
            c()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(5):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(reps):
            g.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1e3 / (reps * len(calls)))
    del g
    return sorted(ts)[len(ts) // 2]


def copies(wbytes):
    return max(8, min(64, math.ceil(1.5e9 / wbytes)))


def bench_bf16(fam, n, k, rows, res):
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.decode import (DecodeProjection, blas_library)
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.sp_policy import sp_projection_tactics
    c = copies(n * k * 2)
    ws = [(torch.randn(n, k, device=DEV) * 0.02).bfloat16() for _ in range(c)]
    for r in rows:
        x = torch.randn(r, k, device=DEV).bfloat16()
        out = torch.empty(r, n, device=DEV, dtype=torch.bfloat16)
        resid = torch.randn(r, n, device=DEV).bfloat16()
        impls = {}
        if fam == "latent_up":  # stock beta-add GEMM
            impls["cublas"] = [lambda w=w: resid.addmm_(x, w.t()) for w in ws]

            def lt_add(w):
                with blas_library("cublaslt"):
                    resid.addmm_(x, w.t())
            impls["lt"] = [lambda w=w: lt_add(w) for w in ws]
        else:
            impls["cublas"] = [lambda w=w: torch.mm(x, w.t(), out=out) for w in ws]

            def lt(w):
                with blas_library("cublaslt"):
                    torch.mm(x, w.t(), out=out)
            impls["lt"] = [lambda w=w: lt(w) for w in ws]
        tf = TABLE_FAMILY.get(fam)
        if tf is not None:
            try:
                tac = sp_projection_tactics(tf, n, k, (r,), ctx())
                projs = [DecodeProjection(w, (r,), tac, family=tf) for w in ws]
                impls["table"] = [lambda p=p: p.run(x, out) for p in projs]
                res.append(dict(fam=fam, dtype="bf16", rows=r, impl="table_tactic", note=str(tac[r])))
            except Exception as e:  # noqa: BLE001
                res.append(dict(fam=fam, dtype="bf16", rows=r, impl="table", err=repr(e)[:300]))
        for name, calls in impls.items():
            try:
                us = timed_chain(calls)
                res.append(dict(fam=fam, dtype="bf16", rows=r, impl=name, us=us, n=n, k=k, copies=c,
                                tbps=n * k * 2 / us / 1e6))
                print(f"{fam:12s} bf16  r={r:4d} {name:10s} {us:8.2f} us {n*k*2/us/1e6:6.2f} TB/s", flush=True)
            except Exception as e:  # noqa: BLE001
                res.append(dict(fam=fam, dtype="bf16", rows=r, impl=name, err=repr(e)[:300]))
                print(f"{fam} bf16 r={r} {name} ERR {e!r}"[:300], flush=True)
    del ws
    torch.cuda.empty_cache()


def bench_mxfp8(fam, n, k, rows, res):
    import flashinfer
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (mxfp8_e4m3_quantize,
                                                                          swizzle_mxfp8_scale)
    from vllm.models.kimi_k3.nvidia.ops import mxfp8_act
    c = copies(n * k)
    wq = []
    for _ in range(c):
        w = (torch.randn(n, k, device=DEV) * 0.02).bfloat16()
        wf, ws = mxfp8_e4m3_quantize(w, is_sf_swizzled_layout=False)
        ws = ws.view(torch.uint8).reshape(n, k // 32)
        sw = swizzle_mxfp8_scale(ws, M=n, K=k).contiguous()
        item = dict(nk=wf.contiguous(), sw=sw)
        try:
            pn = -(-n // 128) * 128
            wp, sp = wf, ws
            if pn != n:
                wp = wf.new_zeros((pn, k)); wp[:n] = wf
                sp = ws.new_zeros((pn, k // 32)); sp[:n] = ws
            item["tw"] = flashinfer.shuffle_matrix_a(wp.contiguous(), 128).reshape(pn, k)
            item["ts"] = flashinfer.shuffle_matrix_sf_a(sp.contiguous(), 128, num_elts_per_sf=32).reshape(-1)
        except Exception as e:  # noqa: BLE001
            item["terr"] = repr(e)[:200]
        wq.append(item)
        del w
    for r in rows:
        x = torch.randn(r, k, device=DEV).bfloat16()
        xq, xs = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
        impls = {}
        impls["fi_cute"] = [lambda it=it: flashinfer.mm_mxfp8(xq, it["nk"].t(), xs, it["sw"],
                                                              out_dtype=torch.bfloat16, backend="cute-dsl")
                            for it in wq]
        impls["fi_cutlass"] = [lambda it=it: flashinfer.mm_mxfp8(xq, it["nk"].t(), xs, it["sw"],
                                                                 out_dtype=torch.bfloat16, backend="cutlass")
                               for it in wq]
        impls["scaled_mm"] = [lambda it=it: torch._scaled_mm(xq, it["nk"].t(), scale_a=xs.view(torch.float8_e8m0fnu),
                                                             scale_b=it["sw"].view(torch.float8_e8m0fnu),
                                                             out_dtype=torch.bfloat16) for it in wq]
        if "tw" in wq[0]:
            try:
                from vllm.utils.flashinfer import flashinfer_mxfp8_quantize_8x4
                xq8, xs8 = flashinfer_mxfp8_quantize_8x4(x)
                impls["fi_trtllm"] = [lambda it=it: flashinfer.mm_mxfp8(xq8, it["tw"].t(), xs8, it["ts"],
                                                                        out_dtype=torch.bfloat16, backend="trtllm",
                                                                        use_8x4_sf_layout=True) for it in wq]
            except Exception as e:  # noqa: BLE001
                res.append(dict(fam=fam, dtype="mxfp8", rows=r, impl="fi_trtllm", err=repr(e)[:300]))
        impls["q_k3"] = [lambda: mxfp8_act.quantize(x)] * 16
        impls["q_fi"] = [lambda: mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)] * 16
        ref = None
        for name, calls in impls.items():
            try:
                if not name.startswith("q_"):
                    o = calls[0]()
                    if ref is None:
                        ref = o.float()
                    err = (o.float() - ref).abs().max().item() / (ref.abs().max().item() + 1e-9)
                else:
                    err = 0.0
                us = timed_chain(calls)
                wb = 0 if name.startswith("q_") else n * k + n * k // 32
                res.append(dict(fam=fam, dtype="mxfp8", rows=r, impl=name, us=us, n=n, k=k, copies=c,
                                tbps=wb / us / 1e6, relerr_vs_first=err))
                print(f"{fam:12s} mxfp8 r={r:4d} {name:10s} {us:8.2f} us {wb/us/1e6:6.2f} TB/s relerr {err:.3g}",
                      flush=True)
            except Exception as e:  # noqa: BLE001
                res.append(dict(fam=fam, dtype="mxfp8", rows=r, impl=name, err=repr(e)[:300]))
                print(f"{fam} mxfp8 r={r} {name} ERR {e!r}"[:300], flush=True)
                traceback.print_exc(limit=2)
    del wq
    torch.cuda.empty_cache()


def main():
    out = sys.argv[1]
    only = sys.argv[2].split(",") if len(sys.argv) > 2 else list(FAMS)
    res = []
    p = torch.cuda.get_device_properties(DEV)
    res.append(dict(device=p.name, sms=p.multi_processor_count, l2=getattr(p, "L2_cache_size", None)))
    print(res[0], flush=True)
    for fam in only:
        n, k, kind, dts = FAMS[fam]
        rows = R_ROWS if kind == "R" else N_ROWS
        if "bf16" in dts:
            bench_bf16(fam, n, k, rows, res)
        if "mxfp8" in dts:
            bench_mxfp8(fam, n, k, rows, res)
        json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
