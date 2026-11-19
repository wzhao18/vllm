"""D-GEMM stream GEMM microbench: correctness + chained cold-weight timing at the D3-32
per-GPU decode shapes (VR200). usage: python3 sg_bench.py <out.json> [quick]"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

HERE = Path(__file__).resolve().parent
SRC = HERE / "stream_gemm.cu"


def build():
    digest = hashlib.sha1(SRC.read_bytes()).hexdigest()[:10]
    bdir = HERE / "build" / digest
    bdir.mkdir(parents=True, exist_ok=True)
    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}a"
    t = time.time()
    ext = load(name=f"k31sg_{digest}", sources=[str(SRC)],
               extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}",
                                  "-std=c++20", "-lineinfo"],
               extra_cflags=["-O3", "-std=c++20"], extra_ldflags=["-lcuda"],
               build_directory=str(bdir), verbose=False)
    print(f"build {time.time() - t:.1f}s arch {arch}", flush=True)
    return ext


SMEM_MAX = 232448


def stages_for(ext, bn, limit=SMEM_MAX):
    s = 16
    while s > 2 and ext.smem_bytes(bn, s) > limit:
        s -= 1
    return s


# (name, N, K, rows list) per GPU at D3-32 (R families: R = 128/256; n families: n = 32/64/96)
SHAPES = [
    ("latent_down", 3584, 7168, [32, 64, 96]),
    ("router+latent_down", 4480, 7168, [32, 64, 96]),
    ("router", 896, 7168, [32, 64, 96]),
    ("latent_up", 7168, 3584, [32, 64, 96]),
    ("shared_down_bf16", 7168, 6144, [32, 64, 96]),
    ("shared_gu_bf16", 12288, 7168, [32, 64, 96]),
    ("dsa_o_proj", 7168, 3072, [128, 256]),
    ("dsa_qkv_a", 2272, 7168, [128, 256]),
    ("dsa_g_proj", 3072, 7168, [128, 256]),
    ("kda_in_bf16", 12448, 7168, [128, 256]),
]


def chain_time(fn_list, iters=20):
    """Median us per call of a CUDA graph running the calls back to back."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for f in fn_list:
            f()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for f in fn_list:
            f()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1000 / len(fn_list))
    ts.sort()
    return ts[len(ts) // 2]


def main():
    out_path = sys.argv[1]
    quick = len(sys.argv) > 2 and sys.argv[2] == "quick"
    torch.manual_seed(0)
    dev = torch.device("cuda")
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    ext = build()
    res = {"sms": sms, "rows": []}
    global NOST
    NOST = torch.empty(0, dtype=torch.int64, device=dev)
    ws = torch.empty(sms * 2 * 256 * 128, dtype=torch.float32, device=dev)
    shapes = [SHAPES[0], SHAPES[1], SHAPES[3], SHAPES[5], SHAPES[6]] if quick else SHAPES
    for name, n, k, rows in shapes:
        wbytes = n * k * 2
        copies = max(4, min(24, (400 << 20) // wbytes))
        ws_list = [torch.randn(n, k, device=dev, dtype=torch.bfloat16) * 0.05
                   for _ in range(copies)]
        for r in rows:
            x = torch.randn(r, k, device=dev, dtype=torch.bfloat16)
            bn = (r + 15) // 16 * 16
            st = stages_for(ext, bn)
            cnts = [torch.zeros((n + 127) // 128, dtype=torch.int32, device=dev)
                    for _ in range(copies)]
            outs = [torch.empty(r, n, device=dev, dtype=torch.bfloat16) for _ in range(copies)]
            row = dict(name=name, n=n, k=k, r=r, stages=st, wMB=wbytes / 1e6)
            try:
                for _ in range(2):
                    ext.run(ws_list[0], x, outs[0], ws, cnts[0], sms, st, NOST, 0)
                torch.cuda.synchronize()
                ref = (x.float() @ ws_list[0].float().t())
                err = ((outs[0].float() - ref).abs().max() / ref.abs().max()).item()
                bf = (x @ ws_list[0].t()).float()
                err_bf = ((bf - ref).abs().max() / ref.abs().max()).item()
                row.update(err=err, err_torch=err_bf)
                t_mm = chain_time([lambda i=i: torch.mm(x, ws_list[i].t(), out=outs[i])
                                   for i in range(copies)])
                row.update(us_torch=t_mm, model_us=wbytes / 13.9e6)
                st2 = stages_for(ext, bn, 113 * 1024)
                for tag, sv_, pf in (("s1", st, 0), ("s2", st2, 12), ("s2p24", st2, 24)):
                    ext.run(ws_list[0], x, outs[0], ws, cnts[0], sms, sv_, NOST, pf)
                    torch.cuda.synchronize()
                    e2 = ((outs[0].float() - ref).abs().max() / ref.abs().max()).item()
                    t_sg = chain_time([lambda i=i: ext.run(ws_list[i], x, outs[i], ws, cnts[i],
                                                           sms, sv_, NOST, pf)
                                       for i in range(copies)])
                    stamps = torch.zeros(sms * 8, dtype=torch.int64, device=dev)
                    for i in (1, 2):
                        ext.run(ws_list[i], x, outs[i], ws, cnts[i], sms, sv_, NOST, pf)
                    ext.run(ws_list[3], x, outs[3], ws, cnts[3], sms, sv_, stamps, pf)
                    torch.cuda.synchronize()
                    sv = stamps.view(sms, 8).cpu()
                    sv = sv[sv[:, 0] > 0]
                    t0 = sv[:, 0].min().item()
                    prof = {}
                    for j, nm in enumerate(["entry", "pdl_wait", "acc_full", "part_stored",
                                            "cnt_passed", "done"]):
                        col = sv[:, j]
                        col = col[col > 0]
                        if len(col):
                            rel = (col - t0).float() / 1000
                            prof[nm] = [round(rel.median().item(), 2), round(rel.max().item(), 2)]
                    smid = sv[:, 6].tolist()
                    prof["max_cta_per_sm"] = max(smid.count(v) for v in set(smid))
                    row[tag] = dict(stages=sv_, pf=pf, us=round(t_sg, 2), err=e2,
                                    tbs=round(wbytes / t_sg / 1e6, 2), prof=prof)
            except Exception as e:  # noqa: BLE001
                row["exc"] = repr(e)[:400]
            print(json.dumps(row), flush=True)
            res["rows"].append(row)
            json.dump(res, open(out_path, "w"), indent=1)
        del ws_list
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
