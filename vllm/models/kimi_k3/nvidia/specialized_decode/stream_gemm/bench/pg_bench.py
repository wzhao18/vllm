"""G-PERSIST microbench: persistent multi-phase GEMM chain vs the same body launched per
GEMM (PDL) vs cuBLAS (+ an activation kernel), chained cold weights in a CUDA graph, at
the D3-32 per-GPU decode shapes (VR200). usage: python3 pg_bench.py <out.json> [rows]"""
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

HERE = Path(__file__).resolve().parent
SRC = HERE / "persist_gemm.cu"


def build():
    digest = hashlib.sha1(SRC.read_bytes()).hexdigest()[:10]
    bdir = HERE / "build" / digest
    bdir.mkdir(parents=True, exist_ok=True)
    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}a"
    t = time.time()
    ext = load(name=f"k31pg_{digest}", sources=[str(SRC)],
               extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}",
                                  "-std=c++20", "-lineinfo"],
               extra_cflags=["-O3", "-std=c++20"], extra_ldflags=["-lcuda"],
               build_directory=str(bdir), verbose=False)
    print(f"build {time.time() - t:.1f}s arch {arch}", flush=True)
    return ext


SMEM_MAX = 232448


def stages_for(ext, bn, limit=SMEM_MAX):
    s = 16
    while s > 2 and ext.psmem_bytes(bn, s) > limit:
        s -= 1
    return s


def graph_us(fn_list, iters=20):
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


def interleave_gu(gate, up):
    """[I, K] gate + [I, K] up -> [2I, K]: per 128-row tile, 64 gate rows then the
    same 64 up rows (the persistent act phase's layout)."""
    i, k = gate.shape
    return torch.stack([gate.view(i // 64, 64, k), up.view(i // 64, 64, k)], 1).reshape(2 * i, k)


# chains: list of (name, [(N, K, act)]) ; act phases: N = 2 * I (gate + up)
CHAINS = [
    ("shared_bf16", [(12288, 7168, 1), (7168, 6144, 0)]),
    ("latent_down_up", [(3584, 7168, 0), (7168, 3584, 0)]),
]


def main():
    out_path = sys.argv[1]
    rows = [int(v) for v in sys.argv[2].split(",")] if len(sys.argv) > 2 else [32, 64, 96]
    torch.manual_seed(0)
    dev = torch.device("cuda")
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    ext = build()
    res = {"sms": sms, "rows": []}
    ws = torch.empty(sms * 2 * 128 * 128, dtype=torch.float32, device=dev)
    no = torch.empty(0, dtype=torch.int64, device=dev)
    for name, phases in CHAINS:
        wbytes = sum(n * k * 2 for n, k, _ in phases)
        copies = max(4, min(16, (600 << 20) // wbytes))
        W = [[torch.randn(n, k, device=dev, dtype=torch.bfloat16) * 0.03 for n, k, _ in phases]
             for _ in range(copies)]
        # torch layout of an act phase: [gate; up] -> persistent layout interleaved
        Wp = [[interleave_gu(w[: n // 2], w[n // 2:]) if a else w
               for w, (n, k, a) in zip(ws_, phases)] for ws_ in W]
        for r in rows:
            row = dict(chain=name, r=r, wMB=round(wbytes / 1e6, 1), copies=copies)
            try:
                x = torch.randn(r, phases[0][1], device=dev, dtype=torch.bfloat16)
                bn = (r + 15) // 16 * 16
                st = stages_for(ext, bn)
                row["stages"] = st
                outs = [[torch.empty(r, n // 2 if a else n, device=dev, dtype=torch.bfloat16)
                         for n, k, a in phases] for _ in range(copies)]
                tout = [[torch.empty(r, n, device=dev, dtype=torch.bfloat16)
                         for n, k, a in phases] for _ in range(copies)]
                cnts = [[torch.zeros(n // 128, dtype=torch.int32, device=dev)
                         for n, k, a in phases] for _ in range(copies)]
                bars = [torch.zeros(2, dtype=torch.int32, device=dev) for _ in range(copies)]
                acts = [a for _, _, a in phases]

                def torch_chain(c):
                    h = x
                    for p, (n, k, a) in enumerate(phases):
                        torch.mm(h, W[c][p].t(), out=tout[c][p])
                        h = tout[c][p]
                        if a:
                            h = F.silu(h[:, : n // 2]) * h[:, n // 2:]
                    return h

                def persist(c, pf=0, stamps=no):
                    ext.prun(Wp[c], x, outs[c], acts, ws, cnts[c], bars[c], sms, st, pf, stamps)
                    return outs[c][-1]

                def sep(c, pf=0):
                    h = x
                    for p in range(len(phases)):
                        ext.prun([Wp[c][p]], h, [outs[c][p]], [acts[p]], ws, [cnts[c][p]],
                                 bars[c], sms, st, pf, no)
                        h = outs[c][p]
                    return h

                ref = torch_chain(0).float()
                got = persist(0).float()
                torch.cuda.synchronize()
                row["err"] = ((got - ref).abs().max() / ref.abs().max()).item()
                got2 = sep(1).float()
                ref1 = torch_chain(1).float()
                row["err_sep"] = ((got2 - ref1).abs().max() / ref1.abs().max()).item()
                # bitwise determinism of the persistent chain across replays
                a1 = persist(2).clone()
                a2 = persist(2).clone()
                row["deterministic"] = bool(torch.equal(a1, a2))
                row["us_torch"] = round(graph_us([lambda c=c: torch_chain(c) for c in range(copies)]), 2)
                # cuBLAS GEMMs alone (no act kernel): the boundary-free share
                row["us_torch_gemms_only"] = round(graph_us(
                    [lambda c=c, p=p: torch.mm(x if p == 0 else x.new_empty(0), W[c][p].t(), out=tout[c][p])
                     for c in range(copies) for p in range(1)]), 2) if False else None
                row["us_sep"] = round(graph_us([lambda c=c: sep(c) for c in range(copies)]), 2)
                row["us_persist"] = round(graph_us([lambda c=c: persist(c) for c in range(copies)]), 2)
                row["us_persist_pf12"] = round(graph_us([lambda c=c: persist(c, 12) for c in range(copies)]), 2)
                # per-phase single-GEMM times (same body, PDL chain of that phase only)
                for p in range(len(phases)):
                    hin = x if p == 0 else outs[0][p - 1]
                    row[f"us_phase{p}_alone"] = round(graph_us(
                        [lambda c=c: ext.prun([Wp[c][p]], hin, [outs[c][p]], [acts[p]], ws,
                                              [cnts[c][p]], bars[c], sms, st, 0, no)
                         for c in range(copies)]), 2)
                    hin_t = x if p == 0 else (F.silu(tout[0][0][:, :phases[0][0] // 2]) *
                                              tout[0][0][:, phases[0][0] // 2:]
                                              if phases[0][2] else tout[0][0]).contiguous()
                    row[f"us_phase{p}_cublas"] = round(graph_us(
                        [lambda c=c: torch.mm(hin_t, W[c][p].t(), out=tout[c][p])
                         for c in range(copies)]), 2)
                row["model_us"] = round(wbytes / 13.9e6, 2)
                stamps = torch.zeros(sms * 8, dtype=torch.int64, device=dev)
                persist(1)
                persist(2)
                persist(3, 0, stamps)
                torch.cuda.synchronize()
                sv = stamps.view(sms, 8).cpu()
                t0 = sv[:, 0].min().item()
                prof = {}
                for j, nm in ((0, "entry"), (1, "last_phase_go"), (7, "last_full"), (5, "done")):
                    col = sv[:, j]
                    col = col[col > 0]
                    if len(col):
                        rel = (col - t0).float() / 1000
                        prof[nm] = [round(rel.median().item(), 2), round(rel.max().item(), 2)]
                row["prof"] = prof
            except Exception as e:  # noqa: BLE001
                row["exc"] = repr(e)[:400]
            row.pop("us_torch_gemms_only", None)
            print(json.dumps(row), flush=True)
            res["rows"].append(row)
            json.dump(res, open(out_path, "w"), indent=1)
        del W, Wp
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
