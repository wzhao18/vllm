"""G-PERSIST boundary microbench (VR200). usage: python3 boundary.py <out.json>"""
import json
import sys
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

HERE = Path(__file__).resolve().parent


def graph_us(fn, iters=30):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(5):
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
        ts.append(a.elapsed_time(b) * 1000)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}a"
    ext = load(name="k31_gp_boundary", sources=[str(HERE / "boundary.cu")],
               extra_cuda_cflags=["-O3", "-std=c++20", f"-gencode=arch=compute_{arch},code=sm_{arch}"], build_directory=str(HERE / "bbuild"),
               verbose=False)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    dev = "cuda"
    sink = torch.zeros(4 * sms, dtype=torch.int32, device=dev)
    ctr = torch.zeros(1, dtype=torch.int32, device=dev)
    no = torch.empty(0, dtype=torch.int64, device=dev)
    smem = 200 * 1024
    res = {"sms": sms, "rows": []}
    K = 20
    for ns in (0, 2000, 5000):
        row = {"ns": ns, "K": K}
        for pdl in (0, 1):
            t = graph_us(lambda: [ext.spin(sms, smem, ns, pdl, sink, no) for _ in range(K)])
            row[f"chain_pdl{pdl}_us"] = round(t, 2)
            row[f"per_boundary_pdl{pdl}_us"] = round((t - K * ns / 1000) / K, 3)
        t1 = graph_us(lambda: ext.spin(sms, smem, ns, 1, sink, no))
        row["single_us"] = round(t1, 2)

        def pers():
            ctr.zero_()
            ext.persist(sms, smem, ns, K, sink, ctr, 0)
        t = graph_us(pers)
        row["persist_us"] = round(t, 2)
        row["per_barrier_us"] = round((t - K * ns / 1000) / K, 3)
        flags = torch.zeros(sms, dtype=torch.int32, device=dev)

        def pflags():
            flags.zero_()
            ext.persist_flags(sms, smem, ns, K, sink, flags, 0)
        t = graph_us(pflags)
        row["persist_flags_us"] = round(t, 2)
        row["per_flag_barrier_us"] = round((t - K * ns / 1000) / K, 3)

        def pflags1():
            flags.zero_()
            ext.persist_flags(sms, smem, ns, 1, sink, flags, 0)
        row["persist_flags_1phase_us"] = round(graph_us(pflags1), 2)
        print(json.dumps(row), flush=True)
        res["rows"].append(row)
    json.dump(res, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
