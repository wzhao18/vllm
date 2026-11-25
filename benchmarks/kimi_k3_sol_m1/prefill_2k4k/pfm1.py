# pfm1.py -- M1 probes for designs/prefill-2k4k.md (EXPERIMENT ONLY, not production code).
# One node (4 VR200). Probes:
#   E  : energy/clock of prefill-shape GEMMs under the power cap (NVML energy counter + 5 ms clock/power samples);
#        compares measured dynamic J/FLOP and clock with the perf-model power_solve, with and without operand
#        re-delivery energy (critic B1).
#   S  : expert weight stream at prefill tile shapes: (a) dense NVFP4 GEMM [T tokens] x [112 experts x FC1 cols]
#        (flashinfer cutlass) and (b) the Rubin MegaMoE N256 (cluster 2x1, 2-CTA) kernel at real p2k/p4k counts.
#   C  : context push: copy engine vs SM store push, 4 ranks all-to-all, alone and concurrent with a tensor-bound
#        GEMM (link-idle window proxy); reports push rate and GEMM slowdown.
import json, os, sys, time, threading, math
import torch

OUT = sys.argv[1]
WHICH = sys.argv[2] if len(sys.argv) > 2 else "E,S,C"
res = []
def emit(d):
    d["t"] = time.strftime("%H:%M:%S"); print("PFM1 " + json.dumps(d), flush=True); res.append(d)
    json.dump(res, open(OUT, "w"), indent=1)

import pynvml as N
N.nvmlInit(); NG = torch.cuda.device_count()
H = [N.nvmlDeviceGetHandleByIndex(i) for i in range(N.nvmlDeviceGetCount())]
# map torch index -> nvml handle via uuid
def nv_of(i):
    u = "GPU-" + str(torch.cuda.get_device_properties(i).uuid)
    for h in H:
        if N.nvmlDeviceGetUUID(h) == u: return h
    return H[i]
HT = [nv_of(i) for i in range(NG)]
emit(dict(probe="env", ngpu=NG, torch=torch.__version__, limits=[N.nvmlDeviceGetPowerManagementLimit(h) / 1e3 for h in HT],
          name=N.nvmlDeviceGetName(HT[0]), host=os.uname().nodename))
rs = getattr(N, "nvmlDeviceGetCurrentClocksEventReasons", None) or N.nvmlDeviceGetCurrentClocksThrottleReasons
SAMP = []; _stop = [False]
def _poll():
    while not _stop[0]:
        t = time.perf_counter()
        for i, h in enumerate(HT):
            try: SAMP.append((t, i, N.nvmlDeviceGetClockInfo(h, N.NVML_CLOCK_SM), N.nvmlDeviceGetPowerUsage(h) / 1e3, rs(h)))
            except Exception: pass
        time.sleep(0.005)
threading.Thread(target=_poll, daemon=True).start()
def energy(i): return N.nvmlDeviceGetTotalEnergyConsumption(HT[i]) / 1e3  # J
def win(t0, t1, gpus):
    out = {}
    for g in gpus:
        s = [x for x in SAMP if t0 <= x[0] <= t1 and x[1] == g]
        if not s: continue
        c = sorted(x[2] for x in s); p = sorted(x[3] for x in s); r = 0
        for x in s: r |= x[4]
        out[g] = dict(clk_p10=c[len(c) // 10], clk_p50=c[len(c) // 2], clk_p90=c[9 * len(c) // 10], p_p50=round(p[len(p) // 2]),
                      p_mean=round(sum(p) / len(p)), reasons=hex(r), n=len(s))
    return out

def sustained(fn, gpus, secs=3.0, inner=5):
    """warm 0.6 s (past the 30-50 ms cap response), then time `secs`; energy counter per GPU over the window."""
    for g in gpus: torch.cuda.synchronize(g)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 0.6:
        fn()
        for g in gpus: torch.cuda.synchronize(g)
    e0 = [energy(g) for g in gpus]; t0 = time.perf_counter(); it = 0
    while time.perf_counter() - t0 < secs:
        for _ in range(inner): fn()
        it += inner
        for g in gpus: torch.cuda.synchronize(g)
    t1 = time.perf_counter(); e1 = [energy(g) for g in gpus]
    return dict(iters=it, wall_s=t1 - t0, per_iter_us=(t1 - t0) / it * 1e6, energy_J=[b - a for a, b in zip(e0, e1)],
                power_W=[(b - a) / (t1 - t0) for a, b in zip(e0, e1)], nvml=win(t0, t1, gpus))

def idle_power(gpus, secs=1.5):
    for g in gpus: torch.cuda.synchronize(g)
    time.sleep(0.3); e0 = [energy(g) for g in gpus]; t0 = time.perf_counter(); time.sleep(secs); t1 = time.perf_counter()
    return [(energy(g) - a) / (t1 - t0) for g, a in zip(gpus, e0)], win(t0, t1, gpus)

# ---------------------------------------------------------------- E: GEMM energy at prefill shapes
def probe_E():
    dev = torch.device("cuda", 0); torch.cuda.set_device(0)
    ip, iw = idle_power([0]); emit(dict(probe="E_idle", idle_W=ip, nvml=iw))
    one = torch.ones((), device=dev)
    shapes = [  # name, M, K, N, prec
        ("kda_in_proj_p2k", 16384, 7168, 6176, "fp8"),
        ("kda_in_proj_p4k", 32768, 7168, 6176, "fp8"),
        ("dsa_qkv_a_p2k", 16384, 7168, 2272, "bf16"),
        ("ref_8192cube", 8192, 8192, 8192, "fp8"),
        ("ref_8192cube", 8192, 8192, 8192, "bf16"),
    ]
    for name, M, K, Nn, prec in shapes:
        g = torch.Generator(device=dev).manual_seed(1)
        a = torch.randn(M, K, generator=g, device=dev); b = torch.randn(Nn, K, generator=g, device=dev) * 0.05
        if prec == "bf16":
            A, B = a.bfloat16(), b.bfloat16(); fn = lambda: A @ B.t()
        else:
            A = (a / 4).to(torch.float8_e4m3fn); B = (b * 20).to(torch.float8_e4m3fn)
            fn = lambda: torch._scaled_mm(A, B.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16)
        del a, b
        r = sustained(fn, [0]); fl = 2.0 * M * K * Nn
        r.update(probe="E_gemm", name=name, prec=prec, M=M, K=K, N=Nn, flop=fl, tflops=fl / r["per_iter_us"] / 1e6,
                 J_per_iter=r["energy_J"][0] / r["iters"], idle_W=ip[0])
        emit(r)
        del A, B; torch.cuda.empty_cache()
    # MXFP8 block-scaled form (stock in_proj kind), if this torch supports it
    try:
        M, K, Nn = 16384, 7168, 6172
        Np = (Nn + 127) // 128 * 128
        A = (torch.randn(M, K, device=dev) / 4).to(torch.float8_e4m3fn); B = (torch.randn(Np, K, device=dev)).to(torch.float8_e4m3fn)
        def blk(rows):  # swizzled e8m0 scale layout = 128x4 blocks, unit scale
            return torch.full((((rows + 127) // 128) * 128 * ((K // 32 + 3) // 4) * 4,), 127, dtype=torch.uint8, device=dev).view(torch.float8_e8m0fnu)
        sa, sb = blk(M), blk(Np)
        fn = lambda: torch._scaled_mm(A, B.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
        fn(); torch.cuda.synchronize()
        r = sustained(fn, [0]); fl = 2.0 * M * K * Np
        r.update(probe="E_gemm", name="kda_in_proj_p2k_mxfp8", prec="fp8", M=M, K=K, N=Np, flop=fl, tflops=fl / r["per_iter_us"] / 1e6,
                 J_per_iter=r["energy_J"][0] / r["iters"], idle_W=ip[0])
        emit(r)
    except Exception as e:
        emit(dict(probe="E_gemm", name="kda_in_proj_p2k_mxfp8", error=repr(e)[:300]))
    torch.cuda.empty_cache()

# ---------------------------------------------------------------- S: expert stream at prefill tile shapes
def probe_S_dense():
    dev = torch.device("cuda", 0); torch.cuda.set_device(0)
    try:
        import flashinfer
        from flashinfer import fp4_quantize, mm_fp4
    except Exception as e:
        emit(dict(probe="S_dense", error="flashinfer import " + repr(e)[:200])); return
    E, LAT, I = 112, 3584, 3072
    Nw = E * 2 * I  # FC1 gate+up of all local experts
    for T, swap in ((320, True), (608, True), (256, True), (512, True)):
        try:
            a = torch.randn(T, LAT, device=dev, dtype=torch.bfloat16)
            w = (torch.randn(Nw, LAT, device=dev) * 0.05).bfloat16()
            ga = (448 * 6) / a.float().abs().amax(); gb = (448 * 6) / w.float().abs().amax()
            aq, asf = fp4_quantize(a, ga, 16, False, True)
            wq, wsf = fp4_quantize(w, gb, 16, False, True)
            del w; torch.cuda.empty_cache()
            alpha = (1.0 / (ga * gb)).float().reshape(1)
            out = torch.empty(T, Nw, device=dev, dtype=torch.bfloat16)
            if swap:   # weights as the M operand (MegaMoE swap-AB form: M = weight rows, N = tokens)
                out = torch.empty(Nw, T, device=dev, dtype=torch.bfloat16)
                fn = lambda: mm_fp4(wq, aq.T, wsf, asf.T, alpha, torch.bfloat16, out, block_size=16, use_8x4_sf_layout=False, backend="cutlass")
            else:
                fn = lambda: mm_fp4(aq, wq.T, asf, wsf.T, alpha, torch.bfloat16, out, block_size=16, use_8x4_sf_layout=False, backend="cutlass")
            fn(); torch.cuda.synchronize()
            r = sustained(fn, [0]); fl = 2.0 * T * Nw * LAT; wb = Nw * LAT * (0.5 + 1 / 16)
            r.update(probe="S_dense", T=T, swap_ab=swap, N=Nw, K=LAT, flop=fl, weight_bytes=wb, tflops=fl / r["per_iter_us"] / 1e6,
                     stream_TBps=wb / r["per_iter_us"] / 1e6, J_per_iter=r["energy_J"][0] / r["iters"])
            emit(r)
            del aq, wq, out; torch.cuda.empty_cache()
        except Exception as e:
            emit(dict(probe="S_dense", T=T, error=repr(e)[:400]))

def probe_S_megamoe():
    sys.path.insert(0, os.environ["N256_DIR"])
    import m1bench as MB
    from vllm.models.kimi_k3.nvidia.megamoe_rubin import config as C
    from vllm.models.kimi_k3.nvidia.megamoe_rubin import experts as X
    from vllm.models.kimi_k3.nvidia.megamoe_rubin import weights as W
    from vllm.models.kimi_k3.nvidia.megamoe_rubin.runtime import RubinMegaMoE
    dev = torch.device("cuda", 0); torch.cuda.set_device(0)
    entries = C.load_knob_source(C.SHIPPED_KNOBS)["entries"]
    problem = C.RubinMoeProblem(num_experts=MB.EXPERTS, quant_kind="nvfp4")
    gen = torch.Generator(device=dev).manual_seed(3)
    full = W.prepare_nvfp4_weights(*MB.mx_layer(gen, MB.EXPERTS, problem.hidden, problem.intermediate, dev),
                                   hidden=problem.hidden, intermediate=problem.intermediate)
    spec = json.load(open(os.path.join(os.environ["N256_DIR"], "counts_n256.json")))
    wb_exp = 3 * problem.hidden * problem.intermediate * (0.5 + 1 / 16)
    for bspec in os.environ.get("PFM1_BUCKETS", "256:16384:0:2048").split(","):
        tile_n, cap, sms, rec_tokens = (int(x) for x in bspec.split(":"))
        try:
            bucket, rec = MB.make_bucket(C, X, entries, tile_n, cap, sms or None, rec_tokens)
            moe = RubinMegaMoE(problem, None, [bucket]); moe.warmup(full, ids_dtypes=(torch.int32,))
        except Exception as e:
            emit(dict(probe="S_megamoe", bucket=bspec, error=repr(e)[:400])); continue
        emit(dict(probe="S_megamoe_bucket", bucket=bspec, knobs=str(rec.get("knobs"))[:400], kernel=moe.sessions[bucket].name))
        for case in ("anchor_start_p2k", "p4k_x2"):
            cnt = spec[case]["layers"][0]
            ids, wts = MB.count_routes(cnt, dev, seed=0); n = ids.shape[0]
            if n > cap: continue
            h = torch.randn(n, problem.hidden, device=dev).to(torch.bfloat16); out = torch.empty_like(h)
            fn = lambda: moe.forward(h, ids, wts, full, bucket=bucket, out=out)
            for _ in range(5): fn()
            torch.cuda.synchronize()
            from torch.profiler import ProfilerActivity, profile
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for _ in range(20): fn()
                torch.cuda.synchronize()
            per = {}
            for e in prof.events():
                if e.device_type.name == "CUDA" and e.device_time_total > 0: per.setdefault(e.name, []).append(e.device_time_total)
            mk, mt = max(per.items(), key=lambda kv: sum(kv[1])); mt = sorted(mt); main_us = mt[len(mt) // 2]
            r = sustained(fn, [0])
            act = sum(1 for c in cnt if c > 0); wb = act * wb_exp
            fl = 2.0 * sum(cnt) * 3 * problem.hidden * problem.intermediate
            r.update(probe="S_megamoe", bucket=bspec, case=case, rows=n, routes=int(sum(cnt)), max_c=int(max(cnt)),
                     mean_c=sum(cnt) / len(cnt), main_us=main_us, main_name=mk[:60], weight_bytes=wb, flop_useful=fl,
                     stream_TBps_main=wb / main_us / 1e6, tflops_main=fl / main_us / 1e6, J_per_iter=r["energy_J"][0] / r["iters"])
            emit(r)
        moe.destroy(); del moe; torch.cuda.synchronize(); torch.cuda.empty_cache()

# ---------------------------------------------------------------- C: context push CE vs SM push
def probe_C():
    import triton, triton.language as tl
    @triton.jit
    def push_k(src, dst, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0); nprog = tl.num_programs(0)
        for off in range(pid * BLOCK, n, nprog * BLOCK):
            o = off + tl.arange(0, BLOCK); m = o < n
            tl.store(dst + o, tl.load(src + o, mask=m), mask=m)
    G = list(range(NG))
    for i in G:
        for j in G:
            if i != j:
                try: torch.cuda.set_device(i); torch.empty(1, device=f"cuda:{i}").copy_(torch.empty(1, device=f"cuda:{j}"))
                except Exception: pass
    # per rank: push PB bytes split over 3 peers (p2k per-DSA-layer push 452 MB over 7 peers; here 3 peers in-node)
    PB = 452 * 2**20; per = PB // (NG - 1) // 16 * 16
    src = [torch.randint(0, 2**15, (per // 2,), dtype=torch.int16, device=f"cuda:{i}") for i in G]
    dst = {(i, j): torch.empty(per // 2, dtype=torch.int16, device=f"cuda:{j}") for i in G for j in G if i != j}
    import ctypes
    import glob
    rt = None
    for cand in ["libcudart.so", "libcudart.so.13", "libcudart.so.12"] + glob.glob("/usr/local/cuda*/lib64/libcudart.so*") + glob.glob(os.path.dirname(torch.__file__) + "/../nvidia/cu*/lib/libcudart.so*"):
        try: rt = ctypes.CDLL(cand); break
        except OSError: pass
    emit(dict(probe="C_cudart", lib=str(rt)))
    pst = {(i, k): torch.cuda.Stream(device=i) for i in G for k in range(1, NG)}   # one stream per (rank, peer)
    streams = {i: pst[(i, 1)] for i in G}
    gstreams = {i: torch.cuda.Stream(device=i) for i in G}
    CH = int(os.environ.get("PFM1_CE_CHUNK_MB", "7")) * 2**20
    def ce_push(chunk, conc=True):
        for i in G:
            torch.cuda.set_device(i)
            for k in range(1, NG):
                j = (i + k) % NG; st = pst[(i, k if conc else 1)]
                s_ = src[i].data_ptr(); d_ = dst[(i, j)].data_ptr(); n = per
                step = chunk or n
                for o in range(0, n, step):
                    rt.cudaMemcpyAsync(ctypes.c_void_p(d_ + o), ctypes.c_void_p(s_ + o), ctypes.c_size_t(min(step, n - o)), 4, ctypes.c_void_p(st.cuda_stream))
    def sm_push(ctas, conc=True):
        for i in G:
            for k in range(1, NG):
                j = (i + k) % NG
                with torch.cuda.device(i), torch.cuda.stream(pst[(i, k if conc else 1)]):
                    push_k[(ctas,)](src[i].view(torch.int64), dst[(i, j)].view(torch.int64), per // 8, BLOCK=2048, num_warps=8)
    # tensor-bound GEMM per GPU (indexer-scorer-like window): fp8 8192 x 7168 x 8192
    one = {i: torch.ones((), device=f"cuda:{i}") for i in G}
    GA = {i: (torch.randn(8192, 7168, device=f"cuda:{i}") / 4).to(torch.float8_e4m3fn) for i in G}
    GB = {i: (torch.randn(8192, 7168, device=f"cuda:{i}")).to(torch.float8_e4m3fn) for i in G}
    def gemm_all(reps):
        for i in G:
            with torch.cuda.stream(gstreams[i]):
                for _ in range(reps): torch._scaled_mm(GA[i], GB[i].t(), scale_a=one[i], scale_b=one[i], out_dtype=torch.bfloat16)
    def timed(push=None, gemm_reps=0, reps=5):
        # returns (push wall per round us, gemm per call us) measured with events on each device
        res_push, res_gemm = [], []
        for r in range(reps + 1):
            for i in G: torch.cuda.synchronize(i)
            ev = {}
            for i in G:
                s0 = torch.cuda.Event(enable_timing=True); s1 = torch.cuda.Event(enable_timing=True)
                g0 = torch.cuda.Event(enable_timing=True); g1 = torch.cuda.Event(enable_timing=True)
                s0.record(gstreams[i]); g0.record(gstreams[i]); ev[i] = (s0, s1, g0, g1)
                for k in range(1, NG): pst[(i, k)].wait_event(s0)
            if gemm_reps: gemm_all(gemm_reps)
            if push: push()
            for i in G:
                ev[i][3].record(gstreams[i])
                for k in range(1, NG): gstreams[i].wait_stream(pst[(i, k)]) if False else None
                ee = []
                for k in range(1, NG):
                    e_ = torch.cuda.Event(enable_timing=True); e_.record(pst[(i, k)]); ee.append(e_)
                ev[i] = (ev[i][0], ee, ev[i][2], ev[i][3])
            for i in G: torch.cuda.synchronize(i)
            if r == 0: continue
            if push: res_push.append(max(ev[i][0].elapsed_time(e_) for i in G for e_ in ev[i][1]) * 1e3)
            if gemm_reps: res_gemm.append(max(ev[i][2].elapsed_time(ev[i][3]) for i in G) * 1e3 / gemm_reps)
        md = lambda v: sorted(v)[len(v) // 2] if v else None
        return md(res_push), md(res_gemm), (min(res_push) if res_push else None, max(res_push) if res_push else None)
    tot = per * (NG - 1)  # bytes sent per rank per round
    _, g_alone, _ = timed(None, gemm_reps=20)
    emit(dict(probe="C_gemm_alone", gemm_us=g_alone))
    variants = [("CE_serial_peers", lambda: ce_push(0, False)), ("CE_3peer_streams", lambda: ce_push(0)),
                ("CE_3peer_streams_7MB_chunks", lambda: ce_push(CH))]
    for c in (11, 22, 44):
        variants.append((f"SM_push_3peer_streams_{c}cta_each", (lambda c=c: sm_push(c))))
    if os.environ.get("PFM1_C_RATE", "0") == "1":
        variants = []
        CH32 = 32 * 2**20
        def ce_rate(rate_tbps):
            # per rank: 3 peer streams, 32 MB copies separated by a GPU sleep so the rank pushes ~rate_tbps
            t_copy = CH32 / (0.70e12 / 3); t_gap = max(0.0, 3 * CH32 / (rate_tbps * 1e12) - t_copy)
            cyc = int(t_gap * 1.5e9)
            for i in G:
                torch.cuda.set_device(i)
                for k in range(1, NG):
                    j = (i + k) % NG; st = pst[(i, k)]
                    s_ = src[i].data_ptr(); d_ = dst[(i, j)].data_ptr()
                    for o in range(0, per, CH32):
                        rt.cudaMemcpyAsync(ctypes.c_void_p(d_ + o), ctypes.c_void_p(s_ + o), ctypes.c_size_t(min(CH32, per - o)), 4, ctypes.c_void_p(st.cuda_stream))
                        if cyc:
                            with torch.cuda.stream(st): torch.cuda._sleep(cyc)
        for rt_ in (0.06, 0.12, 0.2, 0.3, 0.45):
            variants.append((f"CE_rate_{rt_}", (lambda r=rt_: ce_rate(r))))
    if os.environ.get("PFM1_C2", "0") == "1":
        # warm (cap engaged) ABAB: GEMM chain alone vs GEMM chain + push, same reps, 3 pairs per variant
        def warm(ms=400):
            t0 = time.perf_counter()
            while (time.perf_counter() - t0) * 1e3 < ms:
                gemm_all(10)
                for i in G: torch.cuda.synchronize(i)
        CH32 = 32 * 2**20
        def ce_rate2(rate_tbps):
            t_copy = CH32 / (0.70e12 / 3); t_gap = max(0.0, 3 * CH32 / (rate_tbps * 1e12) - t_copy); cyc = int(t_gap * 1.5e9)
            for i in G:
                torch.cuda.set_device(i)
                for k in range(1, NG):
                    j = (i + k) % NG; st = pst[(i, k)]; s_ = src[i].data_ptr(); d_ = dst[(i, j)].data_ptr()
                    for o in range(0, per, CH32):
                        rt.cudaMemcpyAsync(ctypes.c_void_p(d_ + o), ctypes.c_void_p(s_ + o), ctypes.c_size_t(min(CH32, per - o)), 4, ctypes.c_void_p(st.cuda_stream))
                        if cyc:
                            with torch.cuda.stream(st): torch.cuda._sleep(cyc)
        for name, fn, reps in (("CE_full_3streams", lambda: ce_push(0), 30), ("CE_rate_0.15", lambda: ce_rate2(0.15), 40),
                               ("SM_push_44cta_x3", lambda: sm_push(44), 30)):
            for rep in range(3):
                warm(); tw0 = time.perf_counter(); _, ga, _ = timed(None, gemm_reps=reps, reps=2); tw1 = time.perf_counter()
                warm(); tw2 = time.perf_counter(); pc, gc, _ = timed(fn, gemm_reps=reps, reps=2); tw3 = time.perf_counter()
                emit(dict(probe="C2", variant=name, rep=rep, gemm_reps=reps, gemm_alone_us=ga, gemm_conc_us=gc, gemm_slowdown=gc / ga - 1,
                          push_conc_us=pc, push_conc_TBps=tot / pc / 1e6, chain_alone_us=ga * reps, conc_total_us=max(pc, gc * reps),
                          nv_alone={g: (v["clk_p50"], v["p_mean"]) for g, v in win(tw0, tw1, G).items()},
                          nv_conc={g: (v["clk_p50"], v["p_mean"]) for g, v in win(tw2, tw3, G).items()}))
        variants = []
    for name, fn in variants:
        try:
            p_alone, _, sp = timed(fn)
            emit(dict(probe="C_push_alone", variant=name, bytes_per_rank=tot, push_us=p_alone, TBps_per_rank=tot / p_alone / 1e6, spread=sp))
            # concurrent with ~enough GEMM to cover the push
            reps = max(4, int(p_alone / g_alone * 1.5) + 1)
            tw0 = time.perf_counter(); _, g_a2, _ = timed(None, gemm_reps=reps); tw1 = time.perf_counter()
            p_c, g_c, sp = timed(fn, gemm_reps=reps); tw2 = time.perf_counter()
            emit(dict(probe="C_nvml", variant=name, gemm_alone_us=g_a2, nvml_gemm_alone=win(tw0, tw1, G), nvml_concurrent=win(tw1, tw2, G)))
            serial = reps * g_alone + p_alone; conc = max(p_c, reps * g_c)
            emit(dict(probe="C_push_with_gemm", variant=name, gemm_reps=reps, push_us=p_c, TBps_per_rank=tot / p_c / 1e6,
                      gemm_us=g_c, gemm_slowdown=g_c / g_alone - 1, serial_us=serial, concurrent_us=conc,
                      push_hidden_frac=(serial - conc) / p_alone, spread=sp))
        except Exception as e:
            emit(dict(probe="C_push", variant=name, error=repr(e)[:400]))

for w in WHICH.split(","):
    t0 = time.time()
    try:
        {"E": probe_E, "SD": probe_S_dense, "SM": probe_S_megamoe, "C": probe_C}[w]()
    except Exception as e:
        import traceback; emit(dict(probe=w, error=traceback.format_exc()[-1500:]))
    emit(dict(probe="done", which=w, secs=round(time.time() - t0, 1)))
_stop[0] = True
