#!/usr/bin/env python3
"""P-GEMM HW-CLK probe: sustained cuBLAS GEMM loops (MXFP8 KDA in_proj, BF16 MoE latent_down, BF16 latent_up) with an
in-process NVML sampler (10 ms, device by PCI bus id): SM clock, power, enforced limit, clock-event reasons.
Per 1 s window: GEMM us (events), NVML MHz / W / reasons. Also a 70%-duty variant (GEMM bursts + idle) of in_proj."""
import ctypes
import json
import sys
import threading
import time

import torch


def nvml_handle():
    lib = ctypes.CDLL("libnvidia-ml.so.1")
    lib.nvmlInit_v2()
    p = torch.cuda.get_device_properties(0)
    bus = "%08x:%02x:%02x.0" % (p.pci_domain_id, p.pci_bus_id, p.pci_device_id)
    h = ctypes.c_void_p()
    rc = lib.nvmlDeviceGetHandleByPciBusId_v2(bus.encode(), ctypes.byref(h))
    lim = ctypes.c_uint()
    lib.nvmlDeviceGetEnforcedPowerLimit(h, ctypes.byref(lim))
    return lib, h, bus, rc, lim.value / 1000


def sampler(lib, h, out, stop):
    rs_fn = getattr(lib, "nvmlDeviceGetCurrentClocksEventReasons", None) or lib.nvmlDeviceGetCurrentClocksThrottleReasons
    c, p, r = ctypes.c_uint(), ctypes.c_uint(), ctypes.c_ulonglong()
    while not stop.is_set():
        lib.nvmlDeviceGetClockInfo(h, 1, ctypes.byref(c))
        lib.nvmlDeviceGetPowerUsage(h, ctypes.byref(p))
        rs_fn(h, ctypes.byref(r))
        out.append((time.time(), c.value, p.value / 1000, r.value))
        time.sleep(0.01)


def run(name, fn, secs, lib, h, lim, duty=1.0):
    smp, stop = [], threading.Event()
    th = threading.Thread(target=sampler, args=(lib, h, smp, stop), daemon=True)
    th.start()
    t0 = time.time()
    lat = []
    while time.time() - t0 < secs:
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(5):
            fn()
        e.record()
        e.synchronize()
        ms = s.elapsed_time(e)
        lat.append((time.time() - t0, ms * 1000 / 5))
        if duty < 1.0:
            time.sleep(ms / 1000 * (1 - duty) / duty)
    stop.set()
    th.join()
    for w in range(int(secs)):
        ls = sorted(x[1] for x in lat if w <= x[0] < w + 1)
        ss = [x for x in smp if w <= x[0] - t0 < w + 1]
        if not ls or not ss:
            continue
        mh = sorted(x[1] for x in ss)
        pw = sorted(x[2] for x in ss)
        rs = 0
        for x in ss:
            rs |= x[3]
        print(json.dumps({"probe": name, "duty": duty, "t": w, "gemm_us": round(ls[len(ls) // 2], 1),
                          "nvml_mhz": mh[len(mh) // 2], "power_w": round(pw[len(pw) // 2]), "limit_w": lim,
                          "reasons": hex(rs)}), flush=True)


def energy(secs):
    """M2: sustained (power-capped) us, MHz, W and J/GEMM per tactic of the sm_107 skeleton vs cuBLASLt."""
    from flashinfer import mxfp8_quantize

    from vllm.models.kimi_k3.nvidia.specialized_prefill.sm107_gemm.kda_conv_gemm import PlainGemm

    lib, h, bus, rc, lim = nvml_handle()
    torch.manual_seed(0)
    m, n, k = 16384, 36864, 7168
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.05
    xq, xs = mxfp8_quantize(x, is_sf_swizzled_layout=True)
    wq, ws = mxfp8_quantize(w, is_sf_swizzled_layout=True)
    del w
    out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    fns = {"cublaslt": lambda: torch._scaled_mm(xq, wq.t(), scale_a=xs.view(torch.float8_e8m0fnu),
                                                scale_b=ws.view(torch.float8_e8m0fnu), out_dtype=torch.bfloat16)}
    for name, tac in [("512x256_c2x1", ((512, 256), (2, 1), 256, 8, "n")),
                      ("512x256_c4x1", ((512, 256), (4, 1), 256, 8, "n")),
                      ("512x256_c2x4_m4", ((512, 256), (2, 4), 256, 4, "m", 0, 0, "")),
                      ("256x256_c2x1", ((256, 256), (2, 1), 256, 8, "n")),
                      ("256x256_c2x4", ((256, 256), (2, 4), 256, 8, "n"))]:
        try:
            g = PlainGemm(wq, ws, tac)
            g(xq, xs, out)
            fns[name] = (lambda g=g: g(xq, xs, out))
        except Exception as e:  # noqa: BLE001
            print(json.dumps({"tactic": name, "error": str(e)[:200]}), flush=True)
    torch.cuda.synchronize()
    for name, f in fns.items():
        time.sleep(4)
        smp, stop = [], threading.Event()
        th = threading.Thread(target=sampler, args=(lib, h, smp, stop), daemon=True)
        th.start()
        t0 = time.time()
        lat = []
        while time.time() - t0 < secs:
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(5):
                f()
            e.record()
            e.synchronize()
            lat.append((time.time() - t0, s.elapsed_time(e) * 200))
        stop.set()
        th.join()
        ls = sorted(x[1] for x in lat if x[0] >= secs / 2)
        ss = [x for x in smp if x[0] - t0 >= secs / 2]
        first = sorted(x[1] for x in lat[:3])
        us, mhz, pw = ls[len(ls) // 2], sorted(x[1] for x in ss)[len(ss) // 2], sorted(x[2] for x in ss)[len(ss) // 2]
        print(json.dumps({"tactic": name, "us_burst": round(first[0], 1), "us_sust": round(us, 1), "mhz": mhz,
                          "power_w": round(pw), "j_per_gemm": round(pw * us * 1e-6, 3),
                          "clk_per_gemm_M": round(mhz * us * 1e-6, 3)}), flush=True)


def main():
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 10
    if len(sys.argv) > 2 and sys.argv[2] == "energy":
        return energy(secs)
    from flashinfer import mxfp8_quantize

    lib, h, bus, rc, lim = nvml_handle()
    print(json.dumps({"bus": bus, "nvml_rc": rc, "limit_w": lim}), flush=True)
    torch.manual_seed(0)
    m = 16384
    x = torch.randn(m, 7168, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(36864, 7168, device="cuda", dtype=torch.bfloat16) * 0.05
    xq, xs = mxfp8_quantize(x, is_sf_swizzled_layout=True)
    wq, ws = mxfp8_quantize(w, is_sf_swizzled_layout=True)
    del w
    wd = torch.randn(3584, 7168, device="cuda", dtype=torch.bfloat16) * 0.02
    lat = torch.randn(m, 3584, device="cuda", dtype=torch.bfloat16)
    wu = torch.randn(7168, 3584, device="cuda", dtype=torch.bfloat16) * 0.02
    res = torch.randn(m, 7168, device="cuda", dtype=torch.bfloat16)

    def inproj():
        return torch._scaled_mm(xq, wq.t(), scale_a=xs.view(torch.float8_e8m0fnu),
                                scale_b=ws.view(torch.float8_e8m0fnu), out_dtype=torch.bfloat16)

    fns = {"cublaslt_mxfp8_inproj": inproj, "cublas_bf16_latent_down": lambda: torch.mm(x, wd.T),
           "cublas_bf16_latent_up": lambda: res.addmm_(lat, wu.T)}
    for k, f in fns.items():
        f()
    torch.cuda.synchronize()
    time.sleep(3)  # cool-down so each probe starts from the idle clock
    for k, f in fns.items():
        run(k, f, secs, lib, h, lim)
        time.sleep(3)
    run("cublaslt_mxfp8_inproj", inproj, secs, lib, h, lim, duty=0.7)


if __name__ == "__main__":
    main()
