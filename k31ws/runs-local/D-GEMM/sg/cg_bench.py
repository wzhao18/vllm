"""D-GEMM cluster split-K GEMM microbench (VR200): correctness + chained cold-weight
timing at the D3-32 per-GPU decode shapes, striped vs die-local weights, against
cuBLAS (torch.mm). usage: python3 cg_bench.py <out.json> [quick|<names,comma>]"""
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

HERE = Path(__file__).resolve().parent
SRC = HERE / "cluster_gemm.cu"
CHUNK = 2 << 20


def build():
    digest = hashlib.sha1(SRC.read_bytes()).hexdigest()[:10]
    bdir = HERE / "build" / f"cg-{digest}"
    bdir.mkdir(parents=True, exist_ok=True)
    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}a"
    t = time.time()
    ext = load(name=f"k31cg_{digest}", sources=[str(SRC)],
               extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}",
                                  "-std=c++20", "-lineinfo"],
               extra_cflags=["-O3", "-std=c++20"], extra_ldflags=["-lcuda"],
               build_directory=str(bdir), verbose=False)
    print(f"build {time.time() - t:.1f}s arch {arch}", flush=True)
    return ext


SHAPES = [
    ("latent_down", 3584, 7168, [32, 64, 96]),
    ("router+latent_down", 4480, 7168, [32, 64, 96]),
    ("latent_up", 7168, 3584, [32, 64, 96]),
    ("router", 896, 7168, [96]),
    ("shared_down_bf16", 7168, 6144, [96]),
    ("shared_gu_bf16", 12288, 7168, [96]),
    ("dsa_o_proj", 7168, 3072, [128, 256]),
    ("dsa_in", 5344, 7168, [128, 256]),
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


def placed(w, t0, dies):
    """(w0, w1) copies of w: rows [0, 128 t0) homed on die 0, the rest on die 1."""
    n, k = w.shape
    b0 = t0 * 128 * k * 2
    b1 = (n - t0 * 128) * k * 2
    c0, c1 = -(-b0 // CHUNK), -(-b1 // CHUNK)
    raw = dies.localized_empty((c0 + c1) * CHUNK, [0] * c0 + [1] * c1, w.device)
    w0 = raw[:b0].view(torch.bfloat16).view(t0 * 128, k)
    w1 = raw[c0 * CHUNK:c0 * CHUNK + b1].view(torch.bfloat16).view(n - t0 * 128, k)
    w0.copy_(w[:t0 * 128])
    w1.copy_(w[t0 * 128:])
    return w0, w1, raw


def variants(ext, sms, n, k, r):
    tiles = (n + 127) // 128
    kb = k // 64
    bn = (r + 15) // 16 * 16
    out = []
    for per_sm, budget in ((1, 227 * 1024), (2, 113 * 1024)):
        for mult in (1, 2):
            c = max(1, min(16, kb, round(mult * sms / tiles)))
            if c > r:
                continue
            for skb in (1, 2):
                st = 16
                while st > 2 and ext.smem_bytes(bn, st, skb) > budget:
                    st -= 1
                if ext.smem_bytes(bn, st, skb) > budget:
                    continue
                for nprod in (3,):
                    v = (c, st, skb, nprod, per_sm)
                    if v not in out:
                        out.append(v)
    return out


def phases(sv, grid):
    sv = sv.view(grid, 8).cpu()
    t0 = sv[:, 0][sv[:, 0] > 0].min().item()
    prof = {}
    for j, nm in ((1, "pdl_wait"), (2, "acc_full"), (3, "part_synced"), (5, "done"),
                  (7, "last_full")):
        col = sv[:, j]
        col = col[col > 0]
        if len(col):
            rel = (col - t0).float() / 1000
            prof[nm] = [round(rel.median().item(), 2), round(rel.max().item(), 2)]
    dt = sv[:, 6]
    lead = dt[dt != 0] if (dt != 0).any() else dt
    die = (lead >> 32)
    tile = lead & 0xFFFFFFFF
    prof["clusters"] = int(len(lead))
    return prof


def main():
    global dies
    out_path = sys.argv[1]
    sel = sys.argv[2] if len(sys.argv) > 2 else ""
    torch.manual_seed(0)
    dev = torch.device("cuda", torch.cuda.current_device())
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    ext = build()
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common import dies as _d
    dies = _d
    sm_die = dies.sm_domain_table(dev)
    res = {"sms": sms, "die_sms": [int((sm_die == 0).sum()), int((sm_die == 1).sum())],
           "rows": []}
    print(json.dumps(res), flush=True)
    empty = torch.empty(0, dtype=torch.int32, device=dev)
    nost = torch.empty(0, dtype=torch.int64, device=dev)
    if sel == "quick":
        shapes = [SHAPES[0], SHAPES[2]]
    elif sel:
        names = sel.split(",")
        shapes = [s for s in SHAPES if s[0] in names]
    else:
        shapes = SHAPES
    for name, n, k, rows in shapes:
        wbytes = n * k * 2
        tiles = (n + 127) // 128
        t0 = tiles // 2
        copies = max(4, min(16, (400 << 20) // wbytes))
        ws = [torch.randn(n, k, device=dev, dtype=torch.bfloat16) * 0.05
              for _ in range(copies)]
        loc = [placed(w, t0, dies) for w in ws]
        for r in rows:
            x = torch.randn(r, k, device=dev, dtype=torch.bfloat16)
            outs = [torch.empty(r, n, device=dev, dtype=torch.bfloat16)
                    for _ in range(copies)]
            ref = x.float() @ ws[0].float().t()
            row = dict(name=name, n=n, k=k, r=r, wMB=round(wbytes / 1e6, 2),
                       model13_9=round(wbytes / 13.9e6, 2))
            bf = (x @ ws[0].t()).float()
            row["err_torch"] = ((bf - ref).abs().max() / ref.abs().max()).item()
            row["us_torch"] = round(chain_time(
                [lambda i=i: torch.mm(x, ws[i].t(), out=outs[i]) for i in range(copies)]), 2)
            row["us_torch_warm"] = round(chain_time(
                [lambda i=i: torch.mm(x, ws[0].t(), out=outs[i]) for i in range(copies)]), 2)
            vs = []
            for c, st, skb, nprod, per_sm in variants(ext, sms, n, k, r):
                smem = ext.smem_bytes((r + 15) // 16 * 16, st, skb)
                occ = ext.max_clusters(c, smem)
                for place in ("striped", "die"):
                    tag = f"c{c}.s{st}.k{skb}.p{nprod}.{per_sm}x.{place}"
                    try:
                        cnts = [torch.zeros(4, dtype=torch.int32, device=dev)
                                for _ in range(copies)]
                        if place == "die":
                            args = [(loc[i][0], loc[i][1], t0, sm_die) for i in range(copies)]
                        else:
                            args = [(ws[i], ws[i], tiles, empty) for i in range(copies)]

                        def call(i, stamps=nost):
                            a0, a1, tt, sd = args[i]
                            ext.run(a0, a1, tt, x, outs[i], cnts[i], sd, c, st, skb,
                                    nprod, stamps)
                        outs[0].zero_()
                        call(0)
                        call(0)
                        torch.cuda.synchronize()
                        err = ((outs[0].float() - ref).abs().max() / ref.abs().max()).item()
                        us = chain_time([lambda i=i: call(i) for i in range(copies)])

                        def callw(i):
                            a0, a1, tt, sd = args[0]
                            ext.run(a0, a1, tt, x, outs[i], cnts[i], sd, c, st, skb,
                                    nprod, nost)
                        usw = chain_time([lambda i=i: callw(i) for i in range(copies)])
                        grid = tiles * c
                        stamps = torch.zeros(grid * 8, dtype=torch.int64, device=dev)
                        call(1)
                        call(2)
                        call(3, stamps)
                        torch.cuda.synchronize()
                        vs.append(dict(tag=tag, occ_clusters=occ, us=round(us, 2), us_warm=round(usw, 2), err=err,
                                       tbs=round(wbytes / us / 1e6, 2),
                                       prof=phases(stamps, grid)))
                    except Exception as e:  # noqa: BLE001
                        vs.append(dict(tag=tag, occ_clusters=occ, exc=repr(e)[:300]))
                        torch.cuda.synchronize()
                    print(json.dumps(vs[-1]), flush=True)
            row["variants"] = vs
            ok = [v for v in vs if "us" in v and v["err"] < 2 * row["err_torch"] + 1e-3]
            if ok:
                best = min(ok, key=lambda v: v["us"])
                row["best"] = best["tag"]
                row["best_us"] = best["us"]
            print(json.dumps({k_: row[k_] for k_ in row if k_ != "variants"}), flush=True)
            res["rows"].append(row)
            json.dump(res, open(out_path, "w"), indent=1)
        del ws, loc
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
