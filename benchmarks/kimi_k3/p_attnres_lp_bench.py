# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P-ATTNRES: precomputed-logit AttnRes (specialized_prefill attn_res_lp) on
the full 93-layer prefill call sequence of one DEP16 rank (F16K rows).

--check: every call vs the stock kimi_k3_attn_res (output max |diff| in BF16
ulps, prefix / block writes bit-exact, MXFP8 = mxfp8_quantize(BF16 output)).
Timing: CUPTI per-kernel totals of the LP sequence (AttnRes kernels + logit
GEMMs) and of the previous production sequence (old kernel ring depth 3 +
native), interleaved A B B A in one process."""

import argparse
import json
import math

import torch
from torch.profiler import ProfilerActivity, profile

H = 7168
EPS = 1e-6
NL = 93
BS = 12


def build_calls(nl):
    calls = []
    for l in range(nl):
        write = l % BS == 0
        prev = math.ceil(l / BS)
        dsa = l % 4 == 0
        calls.append(dict(c=2 * l, kind="pre", impl="native" if dsa else "mx_q",
                          nb=prev, widx=l // BS if write else -1, delta=l > 0,
                          out_norm=True))
        calls.append(dict(c=2 * l + 1, kind="post",
                          impl="native" if l == 0 else "mx_bq",
                          nb=prev + int(write), widx=-1, delta=not write,
                          out_norm=True))
    calls.append(dict(c=2 * nl, kind="final", impl="native",
                      nb=math.ceil(nl / BS), widx=-1, delta=True, out_norm=False))
    return calls, math.ceil(nl / BS)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=16384)
    p.add_argument("--layers", type=int, default=NL)
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--check", type=int, default=1)
    p.add_argument("--old", type=int, default=1)
    p.add_argument("--depths", default="2")
    p.add_argument("--percall", type=int, default=1)
    p.add_argument("--dielocal", type=int, default=0)
    p.add_argument("--groups", default="4")
    p.add_argument("--vecs", default="8")
    p.add_argument("--qa", default="2", help="claim run-ahead builds")
    p.add_argument("--pdl", default="1", help="PDL builds (1 production, 0 plain)")
    p.add_argument("--qdeep", default="3", help="ring depth of MXFP8-only calls with >= 5 blocks")
    p.add_argument("--resid", default="a", help="residual-stream modes a (today), b (die-local delta), "
                   "c (pre-summed die-local prefix: no delta read, no write-back)")
    p.add_argument("--probe", default="0", help="kernel probe builds (128: data-movement skeleton)")
    p.add_argument("--producer", type=int, default=0, help="time mm vs addmm (+prefix) at [rows, 7168]")
    args = p.parse_args()
    from vllm import _custom_ops as ops
    from vllm.models.kimi_k3.nvidia.ops import mxfp8_act
    from vllm.models.kimi_k3.nvidia.specialized_prefill import attn_res_lp
    from vllm.models.kimi_k3.nvidia.specialized_prefill.attn_res_mxfp8 import (
        attn_res_mxfp8,
    )

    T = args.rows
    calls, nbt = build_calls(args.layers)
    torch.manual_seed(0)
    nw = [(1 + 0.1 * torch.randn(H, device="cuda")).bfloat16() for _ in calls]
    qk = [(torch.randn(H, device="cuda") * H**-0.5).bfloat16() for _ in calls]
    onw = (1 + 0.1 * torch.randn(H, device="cuda")).bfloat16()
    plan = attn_res_lp.Plan(list(zip(nw, qk)), [EPS] * len(calls), BS, nbt)
    prefix0 = torch.randn(T, H, device="cuda").bfloat16()
    delta = torch.randn(T, H, device="cuda").bfloat16()
    attn_out = torch.randn(T, H, device="cuda").bfloat16()

    from vllm.models.kimi_k3.nvidia.specialized_prefill import attn_res_dielocal

    class State:
        def __init__(self, dl=False):
            self.blocks = torch.zeros(T, nbt, H, device="cuda").bfloat16()
            if dl:
                b = attn_res_dielocal.block_residual(self.blocks[:, 0], nbt)
                assert b is not None, "single-die device"
                b.zero_()
                self.blocks = b
            self.dl = dl
            self.prefix = prefix0.clone()
            self.prefix_src = -1

    def p_in(st, c):
        # A write layer's post AttnRes takes the attention output as its
        # prefix (the model's prefix_sum = hidden_states).
        if c["c"] == 0:
            st.prefix_src = -1
        if c["kind"] == "post" and not c["delta"] and st.prefix_src != c["c"]:
            st.prefix = attn_out.clone()
            st.prefix_src = c["c"]
        return st.prefix

    def run_ref(st, c, quant=False):
        pi = p_in(st, c)
        d = delta if c["delta"] else None
        if c["impl"] == "native" or quant is None:
            out = ops.kimi_k3_attn_res(pi, d, st.blocks, nw[c["c"]], qk[c["c"]],
                                       onw if c["out_norm"] else None, c["nb"],
                                       c["widx"], EPS, EPS)
            return out, None
        out, qa = attn_res_mxfp8(pi, d, st.blocks, nw[c["c"]], qk[c["c"]], onw,
                                 c["nb"], c["widx"], EPS, EPS,
                                 bf16_out=c["impl"] == "mx_bq", chunk_depth=3)
        return out, qa

    # Residual-stream modes (SOL audit 05:32 item 3): a = today; b = the delta /
    # attention-output rows homed per die like the block rows; c = the producer
    # writes prefix + delta into the die-local prefix buffer (no delta read, no
    # prefix write-back here). probe: kernel probe build (128 = skeleton).
    mode = {"resid": "a", "probe": 0}
    dl_rows = {}

    def dielocal_copy(x, key):
        if key not in dl_rows:
            from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common import dies
            sup = attn_res_dielocal.SUPER_TOKENS
            tok = -(-T // sup) * sup
            per = sup * H * 2 // attn_res_dielocal.CHUNK
            nbytes = tok * H * 2
            raw = dies.localized_empty(nbytes, [(c // per) & 1 for c in range(nbytes // attn_res_dielocal.CHUNK)],
                                       torch.cuda.current_device())
            dl_rows[key] = raw[:nbytes].view(torch.bfloat16).view(tok, H)[:T]
            dl_rows[key].copy_(x)
        return dl_rows[key]

    def run_lp(st, c, force_bf16=False):
        if c["c"] == 0:
            attn_res_lp.begin(plan, st.blocks)
        pi = p_in(st, c)
        d = delta if c["delta"] else None
        bf = c["impl"] != "mx_q" or force_bf16
        qu = c["impl"] != "native"
        po = attn_res_dielocal.prefix_out(pi, st.blocks) if st.dl else None
        if mode["resid"] == "b" and st.dl:
            if d is not None:
                d = dielocal_copy(delta, "delta")
            if c["kind"] == "post" and not c["delta"]:  # the attention output as prefix
                pi = dielocal_copy(attn_out, "attn_out")
        if mode["resid"] == "c" and st.dl:
            if getattr(st, "c_prefix", None) is None:
                st.c_prefix = attn_res_dielocal.prefix_out(prefix0, st.blocks)
                st.c_prefix.copy_(prefix0)
            pi, d, po = st.c_prefix, None, None
        r = attn_res_lp.attn_res(pi, d, st.blocks, c["c"], c["nb"], c["widx"],
                                 onw if c["out_norm"] else None, EPS, bf, qu,
                                 prefix_out=po, _probe=mode["probe"])
        if po is not None and c["kind"] != "final":
            st.prefix = po
        return r

    if args.check:
        sr, sn = State(), State(bool(args.dielocal))
        worst = 0.0
        bad = []
        def ref64(st, c):
            pi = p_in(st, c).double()
            if c["delta"]:
                pi = (p_in(st, c).float() + delta.float()).bfloat16().double()
            w = nw[c["c"]].double() * qk[c["c"]].double()
            src = [st.blocks[:, i].double() for i in range(c["nb"])] + [pi]
            acc = None
            lg = []
            for v in src:
                lg.append((v @ w) * torch.rsqrt((v * v).mean(-1) + EPS))
            pw = torch.softmax(torch.stack(lg, -1), -1)
            for j, v in enumerate(src):
                acc = pw[:, j:j + 1] * v if acc is None else acc + pw[:, j:j + 1] * v
            if c["out_norm"]:
                acc = acc * torch.rsqrt((acc * acc).mean(-1, keepdim=True) + EPS) \
                    * onw.double()
            return acc

        def err(o, r):
            rms = torch.sqrt((r * r).mean(-1, keepdim=True))
            d = (o.double() - r).abs() / rms
            return d.max().item(), d.mean().item()

        for c in calls:
            if c["c"] in (3, 51, 101, 151, 186):
                r64 = ref64(sr, c)
            else:
                r64 = None
            ref, _ = run_ref(sr, c, quant=None)
            out, qa = run_lp(sn, c, force_bf16=True)
            ok_p = torch.equal(sr.prefix, sn.prefix)
            ok_b = torch.equal(sr.blocks, sn.blocks)
            diff = (out.float() - ref.float()).abs()
            # error in units of the BF16 ulp of the reference
            ulp = torch.clamp(ref.float().abs(), min=1e-30) * 2.0**-7
            rel = (diff / ulp).max().item()
            mism = (diff > 0).float().mean().item()
            worst = max(worst, rel)
            okq = True
            if qa is not None:
                rq = mxfp8_act.quantize(out)
                okq = torch.equal(rq.data.view(torch.uint8), qa.data.view(torch.uint8)) \
                    and rq.scale.numel() == qa.scale.numel() and \
                    torch.equal(rq.scale.view(torch.uint8), qa.scale.view(torch.uint8))
            if not (ok_p and ok_b and okq):
                bad.append(dict(c=c["c"], nb=c["nb"], p=ok_p, b=ok_b, q=okq,
                                ulp=round(rel, 2)))
            if r64 is not None:
                print(json.dumps({"fp64": c["c"], "nb": c["nb"],
                                  "stock_max_mean": [f"{x:.3e}" for x in err(ref, r64)],
                                  "lp_max_mean": [f"{x:.3e}" for x in err(out, r64)]}))
                del r64
            if c["c"] in (1, 2, 50, 100, 186):
                print(json.dumps({"check": c["c"], "nb": c["nb"], "max_ulp": round(rel, 3),
                                  "frac_diff": round(mism, 4), "prefix_eq": ok_p,
                                  "blocks_eq": ok_b, "q_eq": okq}))
        print(json.dumps({"check_worst_ulp": round(worst, 3), "bad": bad[:8],
                          "n_bad": len(bad)}))

    def timed(fn, dl=False):
        st = State(dl)
        for _ in range(2):
            for c in calls:
                fn(st, c)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(args.iters):
                for c in calls:
                    fn(st, c)
                torch.cuda.synchronize()
        tot = {}
        # PDL-clipped totals (as the in-step parser): a kernel's time starts
        # when its stream predecessor ends (one stream here).
        evs = sorted((e for e in prof.events()
                      if e.device_type == torch.autograd.DeviceType.CUDA
                      and "Memset" not in e.name and "Memcpy" not in e.name),
                     key=lambda e: e.time_range.start)
        last_end = None
        for e in evs:
            k = "lp" if "attn_res_lp" in e.name else (
                "attnres_old" if "attn_res" in e.name else "other:" + e.name[:40])
            s0, s1 = e.time_range.start, e.time_range.end
            c0 = s0 if last_end is None else max(s0, min(last_end, s1))
            last_end = s1 if last_end is None else max(last_end, s1)
            tot[k] = tot.get(k, 0.0) + (s1 - s0) / args.iters
            tot["clip:" + k] = tot.get("clip:" + k, 0.0) + (s1 - c0) / args.iters
        return {k: round(v / 1e3, 3) for k, v in sorted(tot.items())}

    for pd in [int(x) for x in args.depths.split(",")] if args.percall else []:
        attn_res_lp.RING_DEPTH = pd
        attn_res_lp.RING_DEPTH_Q_DEEP = pd
        st = State(bool(args.dielocal))
        for c in calls:
            run_lp(st, c)
        torch.cuda.synchronize()
        groups = {}
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            st = State(bool(args.dielocal))
            for c in calls:
                run_lp(st, c)
            torch.cuda.synchronize()
        evs = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
               and "attn_res_lp" in e.name]
        evs.sort(key=lambda e: e.time_range.start)
        for c, e in zip(calls, evs):
            rows = c["nb"] + 1 + int(c["delta"]) + 1 + (c["widx"] >= 0) \
                + (1 if c["impl"] != "mx_q" else 0) + (0.5 if c["impl"] != "native" else 0)
            k = f"{c['kind']}.{c['impl']}.nb{c['nb']}.d{int(c['delta'])}"
            g = groups.setdefault(k, [0, 0.0, rows])
            g[0] += 1
            g[1] += e.time_range.elapsed_us()
        for k, (n, us, rows) in sorted(groups.items()):
            print(json.dumps({"depth": pd, "call": k, "n": n, "us": round(us / n, 1),
                              "TBps": round(rows * T * H * 2 / (us / n) / 1e6, 2)}))

    def lp_arm(qa, pdl, qdeep, resid="a", probe=0):
        def fn(st, c):
            attn_res_lp.CLAIM_AHEAD = qa
            attn_res_lp.PDL = pdl
            attn_res_lp.RING_DEPTH_Q_DEEP = qdeep
            mode["resid"], mode["probe"] = resid, probe
            return run_lp(st, c)
        return fn

    res = []
    for d, gr, vv in [(int(x), int(y), int(v)) for v in args.vecs.split(",")
                      for y in args.groups.split(",") for x in args.depths.split(",")]:
        attn_res_lp.RING_DEPTH = d
        attn_res_lp.GROUPS = gr
        attn_res_lp.VEC = vv
        arms = [("old", lambda st, c: run_ref(st, c))] if args.old else []
        for qa in [int(x) for x in args.qa.split(",")]:
            for pdl in [int(x) for x in args.pdl.split(",")]:
                for qd in [int(x) for x in args.qdeep.split(",")]:
                    for rs in args.resid.split(","):
                        for pb in [int(x) for x in args.probe.split(",")]:
                            arms.append((f"lp_v{vv}g{gr}d{d}qa{qa}p{pdl}qd{qd}r{rs}pb{pb}",
                                         lp_arm(qa, pdl, qd, rs, pb)))
        order = arms + arms[::-1]
        for name, fn in order:
            r = timed(fn, bool(args.dielocal) and name != "old")
            r["arm"] = name

            def fam(pre):
                tot = 0.0
                for k, v in r.items():
                    if not isinstance(v, float) or not k.startswith(pre):
                        continue
                    b = k[len(pre):]
                    if b.startswith("clip:"):
                        continue
                    if b in ("lp", "attnres_old") or "nvjet" in b or "gemm" in b.lower():
                        tot += v
                return round(tot, 3)
            r["total_ms"] = fam("")
            r["clip_ms"] = fam("clip:")
            r = {k: v for k, v in r.items()
                 if k in ("arm", "total_ms", "clip_ms", "lp", "attnres_old", "clip:lp")}
            print(json.dumps(r))
            res.append(r)

    if args.producer:
        # Producer price of the fused residual (item 3): the o_proj-shaped GEMM
        # writing y vs reading prefix and writing prefix + y (cuBLAS beta = 1).
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        for kdim in (4096, 8192):
            x = torch.randn(T, kdim, device="cuda").bfloat16()
            w = torch.randn(kdim, H, device="cuda").bfloat16() * kdim**-0.5
            pre = torch.randn(T, H, device="cuda").bfloat16()
            out = {}
            for name in ("mm", "addmm", "addmm", "mm"):
                fn = (lambda: torch.mm(x, w)) if name == "mm" else (lambda: torch.addmm(pre, x, w))
                for _ in range(3):
                    fn()
                ev[0].record()
                for _ in range(20):
                    fn()
                ev[1].record()
                torch.cuda.synchronize()
                out.setdefault(name, []).append(ev[0].elapsed_time(ev[1]) * 1e3 / 20)
            print(json.dumps({"producer_k": kdim, "rows": T,
                              "mm_us": round(sum(out["mm"]) / 2, 1),
                              "addmm_us": round(sum(out["addmm"]) / 2, 1)}))


if __name__ == "__main__":
    main()
