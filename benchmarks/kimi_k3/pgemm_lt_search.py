#!/usr/bin/env python3
"""P-GEMM: cuBLASLt algorithm search for the prefill MXFP8 GEMMs (the shapes torch._scaled_mm serves in step).

Per shape and row count: every algorithm cuBLASLt's heuristic offers (lt_mxfp8.heuristic), bitwise vs
torch._scaled_mm, burst us (back-to-back, median), then a sustained power-capped loop (NVML clock/power, J/GEMM)
for the default pick and the fastest burst candidates. One JSON line per result."""
import json
import sys
import threading
import time

import torch
from flashinfer import mxfp8_quantize

sys.path.insert(0, "/k31ov/benchmarks/kimi_k3")
from pgemm_clk_probe import nvml_handle, sampler  # noqa: E402

from vllm.models.kimi_k3.nvidia.specialized_prefill import lt_mxfp8  # noqa: E402

SHAPES = {
    "kda_in_proj": (49376, 7168),
    "dense_gate_up": (67584, 7168),
    "dense_down": (7168, 33792),
    "o_proj": (7168, 12288),
    "g_proj|shared_gate_up": (12288, 7168),
    "shared_down": (7168, 6144),
    "latent_down": (3584, 7168),
    "latent_up": (7168, 3584),
    "dsa_qkv_a": (2272, 7168),
    "dsa_q_b": (22528, 1536),
    "pack_n": (15872, 7168),
    "pack_k": (7168, 9728),
}


def burst(f, reps=5, it=10):
    for _ in range(3):
        f()
    ts = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(it):
            f()
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e) * 1000 / it)
    return sorted(ts)[len(ts) // 2]


def sustained(f, secs, lib, h):
    time.sleep(3)
    smp, stop = [], threading.Event()
    th = threading.Thread(target=sampler, args=(lib, h, smp, stop), daemon=True)
    th.start()
    t0, lat = time.time(), []
    while time.time() - t0 < secs:
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(10):
            f()
        e.record()
        e.synchronize()
        lat.append((time.time() - t0, s.elapsed_time(e) * 100))
    stop.set()
    th.join()
    ls = sorted(x[1] for x in lat if x[0] >= secs / 2)
    ss = [x for x in smp if x[0] - t0 >= secs / 2]
    us = ls[len(ls) // 2]
    mhz = sorted(x[1] for x in ss)[len(ss) // 2]
    pw = sorted(x[2] for x in ss)[len(ss) // 2]
    return {"us_sust": round(us, 1), "mhz": mhz, "power_w": round(pw), "j": round(pw * us * 1e-6, 3)}


def main():
    rows_list = [int(r) for r in (sys.argv[1] if len(sys.argv) > 1 else "16384,8192").split(",")]
    secs = float(sys.argv[2]) if len(sys.argv) > 2 else 6
    only = sys.argv[3].split(",") if len(sys.argv) > 3 else None
    lib, h, bus, rc, lim = nvml_handle()
    print(json.dumps({"lt_version": lt_mxfp8.version(), "limit_w": lim}), flush=True)
    ws = torch.empty(64 << 20, dtype=torch.uint8, device="cuda")
    torch.manual_seed(0)
    for name, (n, k) in SHAPES.items():
        if only and name not in only:
            continue
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.05
        wq, wsf = mxfp8_quantize(w, is_sf_swizzled_layout=True)
        del w
        for m in rows_list:
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            xq, xsf = mxfp8_quantize(x, is_sf_swizzled_layout=True)
            del x
            e8 = torch.float8_e8m0fnu

            def ref_fn():
                return torch._scaled_mm(xq, wq.t(), scale_a=xsf.view(e8), scale_b=wsf.view(e8),
                                        out_dtype=torch.bfloat16)

            ref = ref_fn()
            t_ref = burst(ref_fn)
            out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
            algos = lt_mxfp8.heuristic(xq, wq, xsf, wsf, 64)
            res = []
            for i, a in enumerate(algos):
                need = lt_mxfp8.check(xq, wq, xsf, wsf, a)
                if need < 0 or need > ws.numel():
                    continue
                try:
                    out.zero_()
                    lt_mxfp8.gemm(out, xq, wq, xsf, wsf, a, ws)
                    torch.cuda.synchronize()
                    bit = bool(torch.equal(out, ref))
                    md = float((out.float() - ref.float()).abs().max())
                    t = burst(lambda a=a: lt_mxfp8.gemm(out, xq, wq, xsf, wsf, a, ws))
                except Exception as ex:  # noqa: BLE001
                    print(json.dumps({"shape": name, "m": m, "i": i, "error": str(ex)[:160]}), flush=True)
                    continue
                res.append((t, i, a, bit, md))
            res.sort(key=lambda r: r[0])
            d0 = next((r for r in res if r[1] == 0), None)
            print(json.dumps({"shape": name, "m": m, "n": n, "k": k, "torch_us": round(t_ref, 1), "n_algos": len(algos),
                              "default_us": round(d0[0], 1) if d0 else None,
                              "best": [{"i": r[1], "us": round(r[0], 1), "bitwise": r[3], "maxdiff": r[4],
                                        **lt_mxfp8.attrs(r[2])} for r in res[:5]]}), flush=True)
            if secs > 0 and m == rows_list[0]:
                cand = [r for r in res[:3]] + ([d0] if d0 and d0 not in res[:3] else [])
                for t, i, a, bit, md in cand:
                    s = sustained(lambda a=a: lt_mxfp8.gemm(out, xq, wq, xsf, wsf, a, ws), secs, lib, h)
                    print(json.dumps({"shape": name, "m": m, "sust_i": i, "us_burst": round(t, 1), **s,
                                      "hex": lt_mxfp8.algo_to_hex(a)}), flush=True)
            del xq, xsf, out, ref
        del wq, wsf
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
