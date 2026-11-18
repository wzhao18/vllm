"""Run the K3.1 SOL model end to end and write out/*.json.

    python3 k31vr/solmodel/run.py            # ~1 min, stdlib only
    python3 k31vr/solmodel/render.py         # -> k31vr/sol.html
"""
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cost  # noqa: E402
import decode as Dec  # noqa: E402
import hw as HWm  # noqa: E402
import kv  # noqa: E402
import model as Mdl  # noqa: E402
import prefill as Pre  # noqa: E402
from topologies import ALL, DECODE, PREFILL  # noqa: E402

OUT = os.path.join(HERE, "out")
COLS = HWm.COLUMNS
CKS = Mdl.CKPT_ORDER


def rnd(x, n=4):
    if isinstance(x, float):
        if x == 0 or not math.isfinite(x):
            return 0.0 if not math.isfinite(x) else x
        return float("%.*g" % (n, x))
    if isinstance(x, dict):
        return {k: rnd(v, n) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [rnd(v, n) for v in x]
    return x


# ---------------------------------------------------------------------------------------------- prefill
def run_prefill():
    out = {"F_scan": Pre.F_SCAN, "L_scan": Pre.L_SCAN, "L_ref": Pre.L_REF, "F_ref": Pre.F_REF, "f_req": Pre.F_REQ,
           "variants": list(Pre.VARIANTS), "res": {}}
    for ck in CKS:
        for col in COLS:
            for var in Pre.VARIANTS:
                key = "%s|%s|%s" % (ck, col, var)
                per = {}
                for t in PREFILL:
                    grid = []
                    for L in Pre.L_SCAN + [Pre.L_REF]:
                        row = []
                        for F in Pre.F_SCAN:
                            p = Pre.run_point(t, ck, col, F, L, variant=var)
                            row.append([p["tok_s_gpu"], p["t_ms"], 1 if p["feasible"] else 0])
                        grid.append(row)
                    ref_row = grid[-1]
                    feas = [i for i, c in enumerate(ref_row) if c[2]] or list(range(len(ref_row)))
                    bi = max(feas, key=lambda i: ref_row[i][0])
                    bestF = Pre.F_SCAN[bi]
                    best = Pre.run_point(t, ck, col, bestF, Pre.L_REF, variant=var)
                    ref = Pre.run_point(t, ck, col, Pre.F_REF, Pre.L_REF, variant=var)
                    for p in (best, ref):
                        p.pop("ops", None)
                    per[t.id] = dict(grid=grid, best=best, bestF=bestF, ref=ref)
                out["res"][key] = per
    # TTFT and P/D transfer (base variant)
    tt = {}
    for ck in CKS:
        for col in COLS:
            for t in PREFILL:
                Fb = out["res"]["%s|%s|base" % (ck, col)][t.id]["bestF"]
                a = Pre.ttft(t, ck, col, 6000, 72900, Fb)
                b = Pre.ttft(t, ck, col, 74000, 0, Fb)
                tt["%s|%s|%s" % (ck, col, t.id)] = dict(ttft_6k_ms=a, ttft_74k_ms=b, F=Fb)
            # P8: long-request context parallel over G=8 from a DEP16 pool (TTFT-only term)
            base = ALL["P3-16"]
            G = 8
            r = Pre.run_point(base, ck, col, 74000 / G, 74000, f_req=74000 / G, detail=False)
            kda_core = r["fam"].get("kda_core", 0.0)
            hw = HWm.HWS[col]
            scan = 69 * math.log2(G) * (hw.hop_s + 6.29e6 / hw.nvl) * 1e3
            tt["%s|%s|P8" % (ck, col)] = dict(ttft_74k_ms=r["t_ms"] + kda_core + scan, G=G,
                                              note="74K zero-hit request over CP8: one 9.25K-row slice per GPU + KDA "
                                                   "fix-up (+1x KDA core) + log2(8)-round transition exchange")
    out["ttft"] = tt
    out["nixl"] = {str(L): Pre.nixl_bytes(L) for L in (78400, 135000, 145000)}
    out["fan_in"] = {t.id: Pre.fan_in(t) for t in PREFILL}
    return out


# ---------------------------------------------------------------------------------------------- decode
def run_decode():
    out = {"k_list": Dec.K_LIST, "AL": Mdl.AL_GOLDEN, "L": Dec.L_DEC, "variants": list(Dec.VARIANTS), "res": {},
           "cap": {}, "best": {}, "pareto": {}}
    for ck in CKS:
        for k in Dec.K_LIST:
            for t in DECODE:
                cap = Dec.afd_capacity(t, ck, k, Dec.L_DEC) if t.afd else kv.capacity(t, ck, k, Dec.L_DEC)
                cap189 = (Dec.afd_capacity(t, ck, k, Mdl.AGENTX["decode_ctx_osl_weighted"]) if t.afd
                          else kv.capacity(t, ck, k, Mdl.AGENTX["decode_ctx_osl_weighted"]))
                cap["rmax_189K"] = cap189["rmax"]
                out["cap"]["%s|%d|%s" % (ck, k, t.id)] = cap
            for col in COLS:
                for var in Dec.VARIANTS:
                    key = "%s|%d|%s|%s" % (ck, k, col, var)
                    per = {}
                    allpts = []
                    for t in DECODE:
                        _, pts = Dec.curve(t, ck, col, k, var)
                        per[t.id] = pts
                        for p in pts:
                            allpts.append((t.id, p))
                    out["res"][key] = per
                    # best topology per r (grid points only)
                    best = []
                    for r in Dec.R_GRID:
                        cands = [(p["tps_gpu"], tid, p["t_ms"]) for tid, p in allpts if abs(p["r"] - r) < 1e-9]
                        if cands:
                            c = max(cands)
                            second = sorted(cands)[-2] if len(cands) > 1 else None
                            best.append(dict(r=r, topo=c[1], tps_gpu=c[0], t_ms=c[2],
                                             runner=second[1] if second else None,
                                             margin=(c[0] / second[0] - 1) if second else None))
                    out["best"][key] = best
                    # pareto envelope across topologies (max tok/s/GPU at each interactivity)
                    pts = sorted(allpts, key=lambda x: -x[1]["tps_user"])
                    env, mx = [], -1
                    for tid, p in pts:
                        if p["tps_gpu"] > mx:
                            mx = p["tps_gpu"]
                            env.append(dict(topo=tid, r=p["r"], tps_user=p["tps_user"], tps_gpu=p["tps_gpu"], t_ms=p["t_ms"]))
                    out["pareto"][key] = env
    return out


# ---------------------------------------------------------------------------------------------- op tables
def merge_ops(a, b):
    rows = []
    for x, y in zip(a, b):
        rows.append([x["id"], x["fam"], x["n"], x["flops"], x["w"], x["a"], x["kv"], x["st"], x["dt"], x["t"], y["t"],
                     x["bound"][0], y["bound"][0]])
    return rows


def merge_comm(a, b):
    return [[x["layer"], x["prim"], x["name"], x["in_B"], x["out_B"], x["hops"], x["t"], y["t"], 1 if x["async_"] else 0]
            for x, y in zip(a, b)]


DEC_REGIMES = [1, 8, 32]


def run_optables():
    out = {"cols": ["id", "fam", "count", "flops", "w_B", "act_B", "kv_B", "state_B", "dtype", "t_sol_s", "t_ach_s",
                    "bound_sol", "bound_ach"],
           "comm_cols": ["layer", "prim", "name", "in_B", "out_B", "hops", "t_sol_s", "t_ach_s", "async"],
           "regimes": {}, "tables": {}}
    for ck in CKS:
        for t in PREFILL:
            reg = "prefill F%d L%.0fK" % (Pre.F_REF, Pre.L_REF / 1e3)
            a = Pre.run_point(t, ck, "sol", Pre.F_REF, Pre.L_REF, detail=True)
            b = Pre.run_point(t, ck, "achievable", Pre.F_REF, Pre.L_REF, detail=True)
            out["tables"]["%s|%s|prefill" % (ck, t.id)] = dict(
                regime=reg, ops=merge_ops(a["ops"], b["ops"]), comm=merge_comm(a["comm_rows"], b["comm_rows"]),
                t_sol_ms=a["t_ms"], t_ach_ms=b["t_ms"], units_sol=a["units"], units_ach=b["units"],
                per_layer_sol=a["per_layer_ms"], per_layer_ach=b["per_layer_ms"], fam_sol=a["fam"], fam_ach=b["fam"],
                comm_sol=a["comm_raw"], comm_ach=b["comm_raw"])
        for t in DECODE:
            if t.afd:
                continue
            cap = kv.capacity(t, ck, 3, Dec.L_DEC)
            for r in DEC_REGIMES:
                if r > cap["rmax"] or r * t.N < 1:
                    continue
                a = Dec.run_point(t, ck, "sol", r, 3, detail=True)
                b = Dec.run_point(t, ck, "achievable", r, 3, detail=True)
                out["tables"]["%s|%s|decode r%d" % (ck, t.id, r)] = dict(
                    regime="decode r=%d k=3 L=135K" % r, ops=merge_ops(a["ops"], b["ops"]),
                    comm=merge_comm(a["comm_rows"], b["comm_rows"]), t_sol_ms=a["t_ms"], t_ach_ms=b["t_ms"],
                    units_sol=a["units"], units_ach=b["units"], fam_sol=a["fam"], fam_ach=b["fam"],
                    comm_sol=a["comm_raw"], comm_ach=b["comm_raw"], per_layer_sol=a["per_layer_ms"],
                    per_layer_ach=b["per_layer_ms"])
    return out


# ---------------------------------------------------------------------------------------------- meta
ASSUMPTIONS = [
    "SOL column = public VR200 spec peaks (BF16 4 / FP8 17.5 / FP4 35 PF dense, HBM4 22 TB/s, NVLink6 1.8 TB/s per "
    "direction), hop = one CUDA-graph node (0.78 us), zero per-kernel gap. Achievable column = our Hecate measurements "
    "(3.65 / 10.9 / 21.1 PF, 13.9 TB/s, 1.25 TB/s ingress, 2.86 us hop, 0.78 us per kernel). No fitted derating.",
    "FP32 SIMT rate (KDA recurrence, conv) = 212 SMs x 128 lanes x 2 x 2.424 GHz = 131.6 TF in both columns (not measured).",
    "Per-op roofline t = max(FLOPs/peak[dtype], bytes/HBM); ops are additive (no cross-op overlap) except two-microbatch O1.",
    "Ops flagged fused_in_sol (conv, gate/beta norm, o_norm, act, gate_mul, top-k of MoE, latent_norm, ...) carry no "
    "activation HBM traffic. Op-level SOL fusions (O7) are always on: fused indexer score+select (no logits scratch), "
    "final AttnRes mix and drafter context rows only for consumed rows (last token / last 2048 per request in prefill, "
    "accepted rows in decode), Markov vocab shard reused from L2 across the k positions when it fits (<= 64 MiB).",
    "FP8 MLA KV is the standard 576-B row (512 latent + 64 rope FP8); MXFP4 indexer key 68 B; DSA KV = 15,456 B/token.",
    "KDA state per request = one FP32 recurrent state (434 MB) + FP32 conv (3+k columns) + RecoverSSM records "
    "(6.78 MB x q); the deployed allocator reserves far more (workload.md) - that gap is an implementation item.",
    "mxfp4_release routed experts run W4A8 at the FP8 MMA rate with MXFP8 dispatch (3696 B/route); O4 (NVFP4 "
    "activations, FP4 rate, 2016 B/route) is not in the headline because the mission fixes MXFP8 activations.",
    "nvfp4_modelopt: experts NVFP4 W4A4 at FP4 rate (2016 B/route), KDA projections + shared experts MXFP8 at FP8 rate; "
    "MLA, router, latent, dense L0, lm_head BF16. KV/state precision identical to mxfp4_release.",
    "Routing is uniform: experts touched per GPU = (896/E) x (1 - (1 - 16/896)^(rows in EP domain)); EP imbalance 1.0 "
    "in both columns (O5 is a sensitivity knob, not applied).",
    "All-reduce / reduce-scatter take the cheaper of the implemented primitives (NVLS 2 hops vs one-shot / push 1 hop) "
    "per column; NVLS per-GPU bytes ~M in, M(1+1/N) out (comm.json).",
    "Async comm (ShardQ context push on the copy engine) hides under the attention compute of the same layer; all other "
    "comm is exposed unless O1 (two microbatches, t_layer = max(compute, comm), weights streamed twice) is on.",
    "DcpExchange top-k merge bytes 1536 B per row per peer (inferred); sharded verify 100 B per row per peer.",
    "Capacity: HBM budget 272.48 GiB (measured free at vLLM start) - resident weights - 16 GiB reserve (activations, "
    "graphs, symmetric buffers, MegaMoE arena). r_max is per GPU (a group of N holds N x r requests).",
    "Prefill chunks are packed from AgentX requests with 5,456 fresh tokens each over a 72.9K prefix (fresh-weighted "
    "mean); L_ref = 78.4K. Headline = best chunk size per engine in {4K..64K} that fits HBM (0.5 MB per row of live "
    "activations assumed). KDA state checkpoint export for the prefix cache is counted (CKPT=1).",
    "Decode reference context 135K (mission); r_max also reported at the OSL-weighted 189K. AL = golden 3.08/3.81/4.16 "
    "at k=3/5/7. Draft = one parallel 5-layer pass on k query rows + k sequential Markov steps.",
    "DSA-DP owner balance and DP lock-step padding are taken as perfect (SOL); D10 split indexer therefore only adds "
    "comm here - its value is under owner skew, which the SOL model does not price.",
    "PP (P7) stages are balanced by a DP partition over the per-layer times; the boundary push carries the residual + "
    "AttnRes block state (1 + ceil(l/12) rows of 14,336 B per token). Fill/drain bubbles ignored (many chunks in flight).",
    "AFD (D11): two microbatches ping-pong; per MoE layer max(A layer, F experts, A<->F transfer) per half batch; best "
    "of the A:F splits 56:16 / 48:24 / 40:32 per r.",
]
OPEN_Q = [
    "MoE activations for mxfp4_release prefill: MXFP8 (headline) or NVFP4 (the Rubin backend above 512 tokens/rank)?",
    "Dispatch dedup with per-rank FP32 pre-combine (O2) changes the BF16 rounding order of the routed sum - acceptable as 'order may differ'?",
    "KDA core precision: FlashKDA stages the FP32 state into BF16 MMAs in prefill; FP32/TF32 math would ~2x the prefill KDA core.",
    "DcpExchange per-row bytes (X_topk) and the drafter forward row count per verify outcome are inferred, not read from kernels.",
    "Rack-wide EP (D7/D8/P6) imbalance factor: SOL uses 1.0; which measured EPLB max/mean should the achievable column use?",
    "KDA segment-scan context parallel for long prompts (P8) reorders the FP32 recurrence across ranks.",
    "The FP32 SIMT rate and the MXFP4 GEMM rate are not measured on VR200; die-local 17.1 TB/s comes from a code docstring.",
    "Spec values are NVIDIA's preliminary public figures and may change.",
]
ANCHORS = [
    dict(name="C512 hybrid decode step (D3, k=3 assumed)", topo="D3", kind="decode",
         measured={"4": 21.4, "16": 25.1, "32": 25.2, "50": 30.5, "64": 31.7},
         note="measured per-rank batch B -> step ms (release C512 step bench); model r = B requests per GPU"),
    dict(name="C512 hybrid prefill (P2)", topo="P2", kind="prefill", measured_tok_s_gpu=8300.0, F=15600,
         note="2.1 steps/s x ~15.6K tokens per DP4 engine (4 GPUs) = ~8.3K fresh tok/s/GPU (engine logs); "
              "lower bound step 165 ms + 11 ms per 1K tokens"),
]


def run_meta(dec):
    anchors = []
    for a in ANCHORS:
        t = ALL[a["topo"]]
        if a["kind"] == "decode":
            rows = []
            for B, ms in a["measured"].items():
                r = float(B)
                s = Dec.run_point(t, "mxfp4_release", "sol", r, 3)
                h = Dec.run_point(t, "mxfp4_release", "achievable", r, 3)
                rows.append(dict(r=r, measured_ms=ms, sol_ms=s["t_ms"], ach_ms=h["t_ms"]))
            anchors.append(dict(a, rows=rows))
        else:
            s = Pre.run_point(t, "mxfp4_release", "sol", a["F"], Pre.L_REF)
            h = Pre.run_point(t, "mxfp4_release", "achievable", a["F"], Pre.L_REF)
            anchors.append(dict(a, sol_tok_s_gpu=s["tok_s_gpu"], ach_tok_s_gpu=h["tok_s_gpu"], sol_ms=s["t_ms"],
                                ach_ms=h["t_ms"]))
    return dict(
        generated=time.strftime("%Y-%m-%d %H:%M"),
        hw=HWm.constants_table(), hw_cols={c: HWm.HWS[c].as_dict() for c in COLS},
        model=Mdl.model_constants_table(), checkpoints=Mdl.checkpoint_table(),
        ckpt_labels={c: Mdl.CHECKPOINTS[c]["label"] for c in CKS},
        topologies=[t.as_dict() for t in PREFILL + DECODE],
        families={f: Mdl.FAMILY_LABEL[f] for f in Mdl.FAMILY_ORDER}, family_order=Mdl.FAMILY_ORDER,
        group_order=Mdl.GROUP_ORDER, assumptions=ASSUMPTIONS, open_questions=OPEN_Q, anchors=anchors,
        op_info={o.id: dict(name=o.name, shapes=o.shapes, layer=o.layer_type, fused=o.fused,
                            cite=o.cite[:2]) for o in Mdl.OPS},
        reserve_GiB=kv.RESERVE_GIB, agentx=Mdl.AGENTX,
    )


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    pre = run_prefill()
    print("prefill done %.1fs" % (time.time() - t0))
    dec = run_decode()
    print("decode done %.1fs" % (time.time() - t0))
    opt = run_optables()
    print("op tables done %.1fs" % (time.time() - t0))
    meta = run_meta(dec)
    for name, obj in (("prefill", pre), ("decode", dec), ("optables", opt), ("meta", meta)):
        with open(os.path.join(OUT, name + ".json"), "w") as f:
            json.dump(rnd(obj), f, separators=(",", ":"))
        print("wrote out/%s.json (%.0f KB)" % (name, os.path.getsize(os.path.join(OUT, name + ".json")) / 1e3))
    summary(pre, dec)


def summary(pre, dec):
    for ck in CKS:
        print("\n== %s prefill (L_ref %.0fK, best chunk per engine)" % (ck, Pre.L_REF / 1e3))
        for var in ("base", "O1+O2"):
            for col in COLS:
                per = pre["res"]["%s|%s|%s" % (ck, col, var)]
                top = sorted(per.items(), key=lambda kv_: -kv_[1]["best"]["tok_s_gpu"])[:4]
                print("  %-6s %-10s " % (var, col) + "  ".join("%s %.1fK@F%dK" % (tid, v["best"]["tok_s_gpu"] / 1e3, v["bestF"] // 1024) for tid, v in top))
        print("== %s decode best per r (k=3, achievable, base / O1+O2+O3)" % ck)
        for var in ("base", "O1+O2+O3"):
            b = dec["best"]["%s|3|achievable|%s" % (ck, var)]
            print("  %-9s " % var + " ".join("r%g:%s %.0f" % (x["r"], x["topo"], x["tps_gpu"]) for x in b))


if __name__ == "__main__":
    main()
