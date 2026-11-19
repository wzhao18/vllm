"""G-NSPLIT microbench (VR200): N-partitioned weight-stationary GEMM vs cuBLAS at the
D3-32 per-GPU decode shapes, chained cold weights in a CUDA graph (each call's
predecessor is the previous GEMM, PDL edges). usage: python3 ns_bench.py <out.json> [names]"""
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

HERE = Path(__file__).resolve().parent
SRC = HERE / "nsplit_gemm.cu"
SMEM_MAX = 334848


def build():
    digest = hashlib.sha1(SRC.read_bytes()).hexdigest()[:10]
    bdir = HERE / "build" / f"ns-{digest}"
    bdir.mkdir(parents=True, exist_ok=True)
    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}a"
    t = time.time()
    ext = load(name=f"k31ns_{digest}", sources=[str(SRC)],
               extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}",
                                  "-std=c++20", "-lineinfo"],
               extra_cflags=["-O3", "-std=c++20"], extra_ldflags=["-lcuda"],
               build_directory=str(bdir), verbose=False)
    print(f"build {time.time() - t:.1f}s arch {arch}", flush=True)
    return ext


# name, dtype, N, K, rows (R <= 128 only: the M=128 token tile)
SHAPES = [
    ("kda_o", "fp8", 7168, 3072, [128]),
    ("shared_down", "fp8", 7168, 6144, [32, 64, 96]),
    ("kda_in", "fp8", 12448, 7168, [128]),
    ("shared_gu", "fp8", 12288, 7168, [32, 96]),
    ("latent_up", "bf16", 7168, 3584, [32, 64, 96]),
    ("latent_down", "bf16", 3584, 7168, [32, 96]),
    ("dsa_o", "bf16", 7168, 3072, [128]),
]


def chain_time(fns, iters=20):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for f in fns:
            f()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for f in fns:
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
        ts.append(a.elapsed_time(b) * 1000 / len(fns))
    ts.sort()
    return ts[len(ts) // 2]


def variants(ext, sms, n, kbytes):
    kb = kbytes // 128
    out = []
    for nc in (32, 48, 64, 96, 128):
        tiles = -(-n // nc)
        if tiles > sms * 1.1 or tiles < sms * 0.33:
            continue
        for xg, xst, nxp in ((1, 8, 4), (2, 4, 4), (1, 4, 4)):
            wg = max(1, -(-32768 // (nc * 128)))
            nws = -(-kb // wg)
            wst = min(64, nws)
            while wst > 2 and ext.operand_bytes(nc, wg, wst, xg, xst) > 262144:
                wst -= 1
            if ext.operand_bytes(nc, wg, wst, xg, xst) > 262144:
                continue
            lcm = wg * xg // __import__("math").gcd(wg, xg)
            for rotq in (0, lcm):
                if rotq and kb % rotq:
                    continue
                out.append(dict(nc=nc, tiles=tiles, wg=wg, wst=wst, xg=xg, xst=xst, nxp=nxp,
                                na=1, rotq=rotq, prestage=round(min(1.0, wst / nws), 2)))
    return out


def phases(sv, grid):
    sv = sv.view(grid, 8).cpu().double()
    t0 = sv[:, 0].min()
    pr = {}
    for j, nm in ((0, "entry"), (1, "wait_done"), (4, "first_mma"), (5, "half_mma"), (7, "last_mma"), (2, "acc_full"),
                  (3, "stored")):
        rel = (sv[:, j] - t0) / 1000
        pr[nm] = [round(rel.median().item(), 2), round(rel.max().item(), 2)]
    return pr


def main():
    out_path = sys.argv[1]
    sel = sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[2] else None
    torch.manual_seed(0)
    dev = torch.device("cuda", torch.cuda.current_device())
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    ext = build()
    res = {"sms": sms, "rows": []}
    nost = torch.empty(0, dtype=torch.int64, device=dev)
    for name, dt, n, k, rows in SHAPES:
        if sel and name not in sel:
            continue
        esz = 1 if dt == "fp8" else 2
        wbytes = n * k * esz
        copies = max(4, min(16, (400 << 20) // wbytes))
        if dt == "fp8":
            ws = [(torch.randn(n, k, device=dev) * 0.5).to(torch.float8_e4m3fn)
                  for _ in range(copies)]
        else:
            ws = [torch.randn(n, k, device=dev, dtype=torch.bfloat16) * 0.05
                  for _ in range(copies)]
        one = torch.ones((), device=dev)
        for r in rows:
            if dt == "fp8":
                x = (torch.randn(r, k, device=dev) * 0.5).to(torch.float8_e4m3fn)
            else:
                x = torch.randn(r, k, device=dev, dtype=torch.bfloat16)
            outs = [torch.empty(r, n, device=dev, dtype=torch.bfloat16) for _ in range(copies)]
            ref = x.float() @ ws[0].float().t()
            row = dict(name=name, dtype=dt, n=n, k=k, r=r, wMB=round(wbytes / 1e6, 2),
                       model13_9=round(wbytes / 13.9e6, 2))
            if dt == "fp8":
                def lib(i, w=None):
                    torch._scaled_mm(x, (w if w is not None else ws[i]).t(), one, one,
                                     out_dtype=torch.bfloat16, out=outs[i])
            else:
                def lib(i, w=None):
                    torch.mm(x, (w if w is not None else ws[i]).t(), out=outs[i])
            lib(0)
            torch.cuda.synchronize()
            row["err_lib"] = ((outs[0].float() - ref).abs().max() / ref.abs().max()).item()
            row["us_lib"] = round(chain_time([lambda i=i: lib(i) for i in range(copies)]), 2)
            row["us_lib_warm"] = round(chain_time([lambda i=i: lib(i, ws[0])
                                                   for i in range(copies)]), 2)
            print(json.dumps(row), flush=True)
            vs = []
            for v in variants(ext, sms, n, k * esz):
                tag = "nc{nc}.t{tiles}.wg{wg}.wst{wst}.x{xg}x{xst}p{nxp}.rot{rotq}".format(**v)
                try:
                    def call(i, st=nost, w=None):
                        ext.run(w if w is not None else ws[i], x, outs[i], v["nc"], v["wg"],
                                v["wst"], v["xg"], v["xst"], v["na"], 3, v["nxp"], v["rotq"], 0, 0, st)
                    outs[0].zero_()
                    call(0)
                    torch.cuda.synchronize()
                    err = ((outs[0].float() - ref).abs().max() / ref.abs().max()).item()
                    us = chain_time([lambda i=i: call(i) for i in range(copies)])
                    usw = chain_time([lambda i=i: call(i, nost, ws[0]) for i in range(copies)])
                    grid = v["tiles"]
                    stamps = torch.zeros(grid * 8, dtype=torch.int64, device=dev)
                    call(1)
                    call(2)
                    call(3, stamps)
                    torch.cuda.synchronize()
                    d = dict(tag=tag, **v, us=round(us, 2), us_warm=round(usw, 2), err=err,
                             tbs=round(wbytes / us / 1e6, 2), prof=phases(stamps, grid))
                except Exception as e:  # noqa: BLE001
                    d = dict(tag=tag, exc=repr(e)[:300])
                    torch.cuda.synchronize()
                vs.append(d)
                print(json.dumps(d), flush=True)
            row["variants"] = vs
            ok = [v for v in vs if "us" in v and v["err"] < 2 * row["err_lib"] + 1e-3]
            if ok:
                best = min(ok, key=lambda v: v["us"])
                row["best"] = best["tag"]
                row["best_us"] = best["us"]
                row["best_vs_lib"] = round(best["us"] / row["us_lib"], 2)
            print(json.dumps({k_: row[k_] for k_ in row if k_ != "variants"}), flush=True)
            res["rows"].append(row)
            json.dump(res, open(out_path, "w"), indent=1)
        del ws
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
