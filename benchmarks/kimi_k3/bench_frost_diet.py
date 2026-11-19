# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kill/keep experiments for the FROST KDA prefill core at DEP16 F16K.

ref : FROST as deployed (safe gate, q/k L2 norm and beta sigmoid in kernel).
A   : diet ceiling: q/k L2-normalized and the safe gate evaluated in torch
      (FP32), FROST with use_qk_l2norm_in_kernel=False and a log-domain gate.
B   : identity-augmented sequence cut: sequence s is cut at token c through
      cu_seqlens (the second piece starts from h0=0, no copies), a second call
      runs the second piece with V=0, h0=I (outputs R, M), then
      O_p2 += R @ X^T and S = H + X @ M with X the first piece's final state.
C   : head-sliced calls on two streams (high-priority tail heads).

usage: python bench_frost_diet.py [--tokens 16384] [--seqs 3] [--cuts 2:0.5]
"""

import argparse
import time
from collections import defaultdict

import torch

D = 128


def rel_rms(a, b):
    a, b = a.float(), b.float()
    return ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt().clamp_min(1e-30)).item()


def kernel_rows(fn, iters):
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    acc, cnt = defaultdict(float), defaultdict(int)
    for e in prof.events():
        if e.device_type.name == "CUDA" and e.device_time_total > 0:
            acc[e.name] += e.device_time_total
            cnt[e.name] += 1
    return sorted(((t / iters, cnt[n] // iters, n) for n, t in acc.items()), reverse=True)


def timeit(name, fn, iters):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    walls = []
    for _ in range(3):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(iters):
            fn()
        e1.record()
        torch.cuda.synchronize()
        walls.append(e0.elapsed_time(e1) * 1e3 / iters)
    rows = kernel_rows(fn, iters)
    tot = sum(r[0] for r in rows)
    print(f"TIME {name}: wall {min(walls):.1f} us (reps {' '.join(f'{w:.1f}' for w in walls)}), kernels {tot:.1f} us")
    for us, c, n in rows[:6]:
        print(f"  {us:8.1f} us x{c} {n[:100]}")
    return min(walls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=16384)
    ap.add_argument("--seqs", type=int, default=3)
    ap.add_argument("--heads", type=int, default=96)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--cuts", default="2:0.5;2:0.6;0:0.5,1:0.5,2:0.5")
    ap.add_argument("--exps", default="ref,A,B,C,D,E,stock")
    ap.add_argument("--split-heads", type=int, default=70)
    args = ap.parse_args()

    from cudnn.linear_attention.ops import kimi_delta_attention as kda

    dev = torch.device("cuda")
    torch.manual_seed(0)
    T, N, H = args.tokens, args.seqs, args.heads
    lens = [T // N] * N
    lens[-1] += T - sum(lens)
    starts = [0] + list(torch.tensor(lens).cumsum(0).tolist())
    cu = torch.tensor(starts, dtype=torch.int32, device=dev)
    bf, f32 = torch.bfloat16, torch.float32
    q = torch.randn(T, H, D, dtype=bf, device=dev)
    k = torch.randn(T, H, D, dtype=bf, device=dev)
    v = torch.randn(T, H, D, dtype=bf, device=dev) * 0.5
    g = torch.randn(T, H, D, dtype=bf, device=dev) * 0.5
    beta = torch.randn(T, H, dtype=bf, device=dev)
    # spread of per-head amplitudes: steep and slow decay heads
    A_log = torch.linspace(-3.0, 2.0, H, device=dev).to(f32)
    dt_bias = torch.randn(H, D, dtype=f32, device=dev) * 0.1
    lb = -5.0
    h0 = torch.randn(N, H, D, D, dtype=f32, device=dev) * 0.05
    sc = D**-0.5
    exps = args.exps.split(",")
    res = {}

    def frost(qq, kk, vv, gg, bb, cus, init, safe=True, l2=True, **kw):
        extra = dict(safe_gate=True, gate_lower_bound=lb, a_log=A_log, dt_bias=dt_bias) if safe else {}
        return kda(qq, kk, vv, gg, bb, cus, scale=sc, initial_state=init, output_final_state=True,
                   use_qk_l2norm_in_kernel=l2, use_beta_sigmoid_in_kernel=True,
                   plan_name="kda_frost", **extra, **kw)[:2]

    def run_ref():
        res["ref"] = frost(q, k, v, g, beta, cu, h0)

    t0 = time.time()
    run_ref()
    torch.cuda.synchronize()
    print(f"ref first call {time.time() - t0:.1f} s", flush=True)
    o_ref, s_ref = res["ref"]
    timeit("ref", run_ref, args.iters)

    if "stock" in exps:
        try:
            import vllm._flashkda_C  # noqa: F401

            out = torch.empty(1, T, H, D, dtype=bf, device=dev)
            fs = torch.empty_like(h0)
            ws = torch.empty(torch.ops._flashkda_C.get_workspace_size(T, H, N), dtype=torch.uint8, device=dev)
            torch.ops._flashkda_C.fwd(q[None], k[None], v[None], g[None], beta[None], sc, out, ws, A_log,
                                      dt_bias, lb, h0, fs, cu, None, None)
            torch.cuda.synchronize()
            print(f"CHECK ref vs flashkda: out {rel_rms(o_ref, out.view(T, H, D)):.3e} state {rel_rms(s_ref, fs):.3e}")
            res["stock"] = (out.view(T, H, D), fs)
        except Exception as e:  # noqa: BLE001
            print(f"stock: {type(e).__name__}: {str(e)[:200]}")

    def check(name, o, s):
        line = f"CHECK {name} vs ref: out {rel_rms(o, o_ref):.3e} state {rel_rms(s, s_ref):.3e}"
        if "stock" in res:
            line += f" | vs flashkda: out {rel_rms(o, res['stock'][0]):.3e} state {rel_rms(s, res['stock'][1]):.3e}"
        print(line + f" nan {torch.isnan(o).any().item()}", flush=True)

    if "A" in exps:
        def l2n(x):
            xf = x.float()
            return (xf * torch.rsqrt(xf.pow(2).sum(-1, keepdim=True) + 1e-6)).to(bf)

        qn, kn = l2n(q), l2n(k)
        glog = lb * torch.sigmoid(A_log.exp()[None, :, None] * (g.float() + dt_bias[None]))
        for gname, gg in (("fp32", glog), ("bf16", glog.to(bf))):
            try:
                def run_a(gg=gg):
                    res["A"] = frost(qn, kn, v, gg, beta, cu, h0, safe=False, l2=False, gate_domain="log")

                t0 = time.time()
                run_a()
                torch.cuda.synchronize()
                print(f"A[{gname}] first call {time.time() - t0:.1f} s", flush=True)
                check(f"A[{gname}]", *res["A"])
                timeit(f"A[{gname}]", run_a, args.iters)
            except Exception as e:  # noqa: BLE001
                print(f"A[{gname}]: {type(e).__name__}: {str(e)[:300]}", flush=True)

    if "B" in exps:
        eye = torch.eye(D, dtype=f32, device=dev)
        vzero = torch.zeros_like(v)
        s_aux = torch.cuda.Stream(priority=-1)
        for spec in args.cuts.split(";"):
            cuts = {int(a): float(b) for a, b in (p.split(":") for p in spec.split(","))}
            bounds, inits, piece_of = [0], [], []
            for i in range(N):
                if i in cuts:
                    c = starts[i] + int(round(lens[i] * cuts[i] / 64)) * 64
                    bounds += [c, starts[i + 1]]
                    inits += [h0[i], torch.zeros_like(h0[i])]
                    piece_of.append((len(bounds) - 3, c, starts[i + 1]))
                else:
                    bounds.append(starts[i + 1])
                    inits.append(h0[i])
            cu_b = torch.tensor(bounds, dtype=torch.int32, device=dev)
            init_b = torch.stack(inits)
            # identity runs: the second pieces as their own sequences (token slices are views)
            segs = [(c, e, torch.tensor([0, e - c], dtype=torch.int32, device=dev)) for _, c, e in piece_of]
            eye_h = eye.expand(1, H, D, D).contiguous()

            def run_b(concurrent):
                o, s = frost(q, k, v, g, beta, cu_b, init_b)
                ids = []
                strm = s_aux if concurrent else torch.cuda.current_stream()
                if concurrent:
                    s_aux.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(strm):
                    for c, e, cu_i in segs:
                        ids.append(frost(q[c:e], k[c:e], vzero[c:e], g[c:e], beta[c:e], cu_i, eye_h))
                if concurrent:
                    torch.cuda.current_stream().wait_stream(s_aux)
                fin = []
                for (p1, c, e), (r, m) in zip(piece_of, ids):
                    x = s[p1]  # (H, V, K): true state at the cut
                    # O[t, h, v] += sum_k R[t, h, k] X[h, v, k]
                    corr = torch.bmm(r.float().transpose(0, 1), x.transpose(1, 2)).transpose(0, 1)
                    o[c:e] = (o[c:e].float() + corr).to(bf)
                    fin.append(s[p1 + 1] + torch.bmm(x, m[0]))
                res["B"] = (o, s, fin)

            # the corrected O includes the identity-run correction; the uncorrected
            # (V, 0) piece output is overwritten in place, so rerun per timing call
            name = f"B[{spec}]"
            try:
                t0 = time.time()
                run_b(False)
                torch.cuda.synchronize()
                print(f"{name} first call {time.time() - t0:.1f} s", flush=True)
                o, s, fin = res["B"]
                states = []
                j = 0
                for i in range(N):
                    if i in cuts:
                        states.append(fin[j])
                        j += 1
                    else:
                        states.append(s[[b for b in range(len(bounds) - 1) if bounds[b] == starts[i]][0]])
                check(name, o, torch.stack(states))
                for (p1, c, e) in piece_of:
                    print(f"  piece [{c},{e}) out rel-rms vs ref {rel_rms(o[c:e], o_ref[c:e]):.3e}")
                timeit(name + " serial", lambda: run_b(False), args.iters)
                timeit(name + " 2-stream", lambda: run_b(True), args.iters)
            except Exception as e:  # noqa: BLE001
                print(f"{name}: {type(e).__name__}: {str(e)[:300]}", flush=True)

    hs = args.split_heads

    def head_parts(view):
        parts = []
        for lo, hi in ((0, hs), (hs, H)):
            cp = (lambda x: x) if view else (lambda x: x.contiguous())
            parts.append(tuple(cp(x[:, lo:hi]) for x in (q, k, v, g)) + (cp(beta[:, lo:hi]),
                         h0[:, lo:hi].contiguous(), A_log[lo:hi].contiguous(), dt_bias[lo:hi].contiguous()))
        return parts

    if "C" in exps or "D" in exps:
        parts = head_parts(False)
        s_hi = torch.cuda.Stream(priority=-1)

        def call(p):
            qq, kk, vv, gg, bb, ii, al, dt = p
            return kda(qq, kk, vv, gg, bb, cu, scale=sc, initial_state=ii, output_final_state=True,
                       use_qk_l2norm_in_kernel=True, use_beta_sigmoid_in_kernel=True, safe_gate=True,
                       gate_lower_bound=lb, a_log=al, dt_bias=dt, plan_name="kda_frost")[:2]

        def run_c(concurrent):
            if concurrent:
                s_hi.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s_hi):
                    b = call(parts[1])
                a = call(parts[0])
                torch.cuda.current_stream().wait_stream(s_hi)
            else:
                a = call(parts[0])
                b = call(parts[1])
            res["C"] = (a, b)

        try:
            run_c(False)
            torch.cuda.synchronize()
            (oa, sa), (ob, sb) = res["C"]
            check("C", torch.cat([oa, ob], 1), torch.cat([sa, sb], 1))
            timeit("C serial", lambda: run_c(False), args.iters)
            timeit("C 2-stream", lambda: run_c(True), args.iters)
        except Exception as e:  # noqa: BLE001
            print(f"C: {type(e).__name__}: {str(e)[:300]}", flush=True)
        try:
            vp = head_parts(True)
            (oa, sa) = call(vp[0])
            (ob, sb) = call(vp[1])
            torch.cuda.synchronize()
            check("C-views", torch.cat([oa, ob], 1), torch.cat([sa, sb], 1))
            parts = vp
            print("C-views: strided head views ACCEPTED (no copies)")
        except Exception as e:  # noqa: BLE001
            print(f"C-views: {type(e).__name__}: {str(e)[:300]}", flush=True)

    if "D" in exps:
        # tail overlap: o_norm of the main heads runs while the tail heads' FROST
        # call (its prep + d_v-half launches) leaves SMs idle
        from vllm.models.kimi_k3.nvidia.ops import kda_segscan
        from vllm.models.kimi_k3.nvidia.ops.mxfp8_act import Mxfp8Buffer

        kda_segscan.load_extension()
        onorm = torch.ops.k3_flashkda_split.o_norm_mxfp8
        P = H * D
        proj = (torch.randn(T, 4 * P + 128, device=dev) * 2).to(bf)
        g2 = proj[:, 3 * P : 4 * P]
        wn = (torch.rand(D, device=dev) + 0.5).contiguous()
        out_all = Mxfp8Buffer(T, P, dev, bf)
        out_a = Mxfp8Buffer(T, hs * D, dev, bf)
        out_b = Mxfp8Buffer(T, (H - hs) * D, dev, bf)
        s_b = torch.cuda.Stream(priority=-1)

        def serial():
            o, _ = frost(q, k, v, g, beta, cu, h0)
            onorm(o[None], g2, wn, 1e-6, out_all.data, out_all.scale)

        def overlapped():
            oa, _ = call(parts[0])
            s_b.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s_b):
                ob, _ = call(parts[1])
            onorm(oa[None], g2[:, : hs * D], wn, 1e-6, out_a.data, out_a.scale)
            with torch.cuda.stream(s_b):
                onorm(ob[None], g2[:, hs * D :], wn, 1e-6, out_b.data, out_b.scale)
            torch.cuda.current_stream().wait_stream(s_b)

        try:
            timeit("D serial frost+onorm", serial, args.iters)
            timeit(f"D split {hs}/{H - hs} overlapped", overlapped, args.iters)
        except Exception as e:  # noqa: BLE001
            print(f"D: {type(e).__name__}: {str(e)[:300]}", flush=True)

    if "E" in exps:
        # head-split calls in place on head views of the recurrent state pool
        pool0 = torch.randn(N + 5, H, D, D, dtype=f32, device=dev) * 0.05
        idx = torch.tensor([2, N + 3, 0][:N] + list(range(5, 5 + max(0, N - 3))), dtype=torch.int32, device=dev)

        def inplace(pool, lo, hi):
            return kda(q[:, lo:hi], k[:, lo:hi], v[:, lo:hi], g[:, lo:hi], beta[:, lo:hi], cu, scale=sc,
                       initial_state=pool[:, lo:hi], state_indices=idx, output_final_state=True,
                       use_qk_l2norm_in_kernel=True, use_beta_sigmoid_in_kernel=True, safe_gate=True,
                       gate_lower_bound=lb, a_log=A_log[lo:hi].contiguous(), dt_bias=dt_bias[lo:hi].contiguous(),
                       plan_name="kda_frost")[0]

        try:
            pr = pool0.clone()
            o_full = inplace(pr, 0, H)
            ps = pool0.clone()
            oa = inplace(ps, 0, hs)
            ob = inplace(ps, hs, H)
            torch.cuda.synchronize()
            o_split = torch.cat([oa, ob], 1)
            print(f"CHECK E split-in-place vs full-in-place: out {rel_rms(o_split, o_full):.3e} pool {rel_rms(ps, pr):.3e}"
                  f" heads<{hs} out bitwise {torch.equal(oa, o_full[:, :hs])} pool bitwise {torch.equal(ps[:, :hs], pr[:, :hs])}"
                  f" untouched rows kept {torch.equal(ps[1], pool0[1])}", flush=True)
            timeit("E full in place", lambda: inplace(pr, 0, H), args.iters)
            timeit("E split in place serial", lambda: (inplace(ps, 0, hs), inplace(ps, hs, H)), args.iters)
        except Exception as e:  # noqa: BLE001
            print(f"E: {type(e).__name__}: {str(e)[:300]}", flush=True)

    if "F" in exps:
        from vllm.models.kimi_k3.nvidia.ops import kda_segscan
        from vllm.models.kimi_k3.nvidia.ops.mxfp8_act import Mxfp8Buffer

        kda_segscan.load_extension()
        onorm = torch.ops.k3_flashkda_split.o_norm_mxfp8
        P = H * D
        x = (torch.randn(1, T, H, D, device=dev) * 3).to(bf)
        g2 = (torch.randn(T, P + 64, device=dev) * 2).to(bf)[:, :P]
        wn = (torch.rand(D, device=dev) + 0.5).contiguous()
        ref_b, out_b = Mxfp8Buffer(T, P, dev, bf), Mxfp8Buffer(T, P, dev, bf)
        onorm(x, g2, wn, 1e-6, ref_b.data, ref_b.scale)
        xa, xb = x[:, :, :hs].contiguous(), x[:, :, hs:].contiguous()
        try:
            onorm(xa, g2, wn, 1e-6, out_b.data, out_b.scale, 0)
            onorm(xb, g2, wn, 1e-6, out_b.data, out_b.scale, hs)
            torch.cuda.synchronize()
            print(f"CHECK F o_norm head windows bitwise: data {torch.equal(out_b.data.view(torch.uint8), ref_b.data.view(torch.uint8))}"
                  f" scale {torch.equal(out_b.scale[: T * P // 32], ref_b.scale[: T * P // 32])}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"F: {type(e).__name__}: {str(e)[:300]}", flush=True)


if __name__ == "__main__":
    main()
