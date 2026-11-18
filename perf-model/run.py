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

import calibrate as Cal  # noqa: E402
import cost  # noqa: E402
import decode as Dec  # noqa: E402
import hw as HWm  # noqa: E402
import kv  # noqa: E402
import model as Mdl  # noqa: E402
import prefill as Pre  # noqa: E402
from topologies import ALL, DECODE, EP_MAX, PREFILL, SCOPE  # noqa: E402

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
    out["scope"] = [t.id for t in PREFILL if t.id in SCOPE]
    out["rank_scope"] = {k: sorted(((tid, v["best"]["tok_s_gpu"], v["bestF"]) for tid, v in per.items() if tid in SCOPE),
                                   key=lambda x: -x[1]) for k, per in out["res"].items()}
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
                cap = kv.capacity(t, ck, k, Dec.L_DEC)
                cap189 = kv.capacity(t, ck, k, Mdl.AGENTX["decode_ctx_osl_weighted"])
                cap["rmax_189K"] = cap189["rmax"]
                if not t.afd:      # achievable = packed block pool (per-request block floor); calibrated = x measured
                    ca = kv.capacity(t, ck, k, Dec.L_DEC, col="achievable")
                    cap["rmax_ach"], cap["per_req_ach_GB"], cap["blocks_ach"] = ca["rmax"], ca["per_req_GB"], ca["blocks"]
                    cap["rmax_ach_189K"] = kv.capacity(t, ck, k, Mdl.AGENTX["decode_ctx_osl_weighted"], col="achievable")["rmax"]
                    cap["rmax_cal"], cap["rmax_cal_method"] = Cal.decode_rmax(t, ck, k, ca["rmax"])
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
                    out["pareto"][key] = _envelope(allpts)
                    # <= 32-GPU scope (2026-11-18)
                    sp = [(tid, p) for tid, p in allpts if tid in SCOPE]
                    out.setdefault("best_scope", {})[key] = _best_per_r(sp)
                    out.setdefault("pareto_scope", {})[key] = _envelope(sp)
    out["scope"] = [t.id for t in DECODE if t.id in SCOPE]
    return out


def _best_per_r(allpts):
    best = []
    for r in Dec.R_GRID:
        cands = sorted((p["tps_gpu"], tid, p["t_ms"], p["tps_user"]) for tid, p in allpts if abs(p["r"] - r) < 1e-9)
        if cands:
            c, s = cands[-1], (cands[-2] if len(cands) > 1 else None)
            best.append(dict(r=r, topo=c[1], tps_gpu=c[0], t_ms=c[2], tps_user=c[3], runner=s[1] if s else None,
                             margin=(c[0] / s[0] - 1) if s else None))
    return best


def _envelope(allpts):
    """Pareto envelope across topologies (max tok/s/GPU at each interactivity)."""
    pts = sorted(allpts, key=lambda x: -x[1]["tps_user"])
    env, mx = [], -1
    for tid, p in pts:
        if p["tps_gpu"] > mx:
            mx = p["tps_gpu"]
            env.append(dict(topo=tid, r=p["r"], tps_user=p["tps_user"], tps_gpu=p["tps_gpu"], t_ms=p["t_ms"]))
    return env


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
    "direction), zero per-kernel gap; there is no public NVLink latency figure, so a dependent hop uses our measured "
    "one-way flag (2.86 us) and is never set below a measured hand-off. Achievable column = our Hecate "
    "measurements (3.65 / 10.9 / 21.1 PF, 13.9 TB/s, 2.86 us push hop + 0.035 us per polled peer, 0.78 us graph node) "
    "plus the wave-1 microbench fixes (2026-11-18): push 1.06 TB/s (measured ceiling), NVLS AR alpha 12.6-12.9 us + "
    "algbw 0.64/0.77/0.84/0.88 TB/s (N = 2/4/8/16, multimem kernel), NVLS RS alpha + M / 1.06 TB/s (0.88 at N = 2), "
    "standalone exchange floor = fastest measured exchange per N (3.86/4.22/4.49/4.89 us; fused sdec TP-path exchanges "
    "pay the hop only), 2 us floor on the grid-dependent kernels (KDA recurrence, indexer, sparse MLA, MegaMoE), "
    "MegaMoE expert weights streamed die-local (17.1 TB/s). Tests keep the column at or below every comm / MegaMoE / "
    "step row and e2e anchor. No fitted derating; the calibrated column (calibrate.py) is the only derated one.",
    "FP32 SIMT (131.6 TF) and MUFU exp (8.2 T/s) rates are derived from measured attributes (212 SMs x 128 lanes x 2, "
    "x 16 MUFU, x 2.424 GHz) in both columns (not measured).",
    "Per-op roofline t = max(FLOPs/peak[dtype], bytes/HBM, SIMT/peak_simt + SFU/peak_mufu); ops are additive (no "
    "cross-op overlap) except two-microbatch O1. A launched (non-fused) kernel costs max(roofline, per-kernel gap): "
    "the gap overlaps the previous kernel under PDL; comm events pay their hop latency only.",
    "Sparse MLA runs FP8 Q x FP8 KV MMAs (the 576-B per-tensor FP8 row; a 656-B fp8_ds_mla row would be BF16), "
    "co-bound with one MUFU exp per (row, head, key). The indexer is max(FP4 MMA, FP32 epilogue relu + head-weight FMA "
    "over 32 heads at the SIMT rate, key bytes).",
    "Ops flagged fused_in_sol (conv, gate/beta norm, o_norm, act, gate_mul, top-k of MoE, latent_norm, ...) carry no "
    "activation HBM traffic, but their second input is read by the consumer (KDA / DSA o_proj read the output gate). "
    "The MoE fc1->fc2 intermediate stays in L2 up to 64 MiB per GPU (decode) and round-trips HBM above (prefill). "
    "O7 SOL fusions are always on: exact top-k without a full logits pass when cheaper (min of BF16 scratch write+read "
    "and a two-pass recompute; the scratch wins everywhere today), final AttnRes mix on the R logit rows and aux "
    "captures / drafter context rows only for consumed rows in prefill, Markov vocab shard reused from L2 (<= 64 MiB).",
    "FP8 MLA KV is the standard 576-B row (512 latent + 64 rope FP8); MXFP4 indexer key 68 B; DSA KV = 15,456 B/token. "
    "Sparse-MLA KV reads use one SOL rule in both phases: the union bound min(Lu, Ltk) (decode: q x 2048 rows per "
    "request at 135K; the true union of overlapping verify rows is smaller, worth ~2% of a high-r DEP step).",
    "KDA state per request = one FP32 recurrent state (434 MB) + FP32 conv (3+k columns) + RecoverSSM records "
    "(1024 B per row per head = 6.78 MB x q, written for q rows and read for AL rows per step; one constant drives "
    "traffic and capacity); the deployed allocator reserves far more (workload.md) - that gap is an implementation item.",
    "mxfp4_release routed experts run W4A8 at the FP8 MMA rate with MXFP8 dispatch (3696 B/route); O4 (NVFP4 "
    "activations, FP4 rate, 2016 B/route) is not in the headline because the mission fixes MXFP8 activations. The "
    "deployed Rubin MegaMoE switches to NVFP4 above 512 tokens/rank: prefill microbenches must pin MXFP8 or compare to O4.",
    "nvfp4_modelopt: experts NVFP4 W4A4 at FP4 rate (2016 B/route), KDA projections + shared experts MXFP8 at FP8 rate; "
    "MLA, router, latent, dense L0, lm_head BF16. KV/state precision identical to mxfp4_release.",
    "Routing is uniform: experts touched per GPU = ceil(896/E) x (1 - (1 - 16/896)^(rows in EP domain)); EP sizes that "
    "do not divide 896 (72, 24) hold ceil(896/E) expert slots per GPU with redundant replicas carrying the load evenly "
    "(weights charged, imbalance 1.0 in both columns; O5 is a sensitivity knob, not applied).",
    "All-reduce / reduce-scatter take the cheapest implemented primitive per column: NVLS multimem, one-shot Lamport push "
    "(1 hop, (N-1)M in) or two-shot push (2 hops). sdec-style Lamport pushes are priced at the flag hop (16-rank "
    "two-shot owner reduce measured 3.39 us after the last push); the torch one_shot (pull + barrier) bench does not "
    "apply to them.",
    "TP-sharded MoE-dense (D7, D8, D10, D11): router/latent_down column shards are transposed to row ownership for "
    "top-k + dispatch, and the combined latent is transposed back for a K-split latent_up whose partials ride the "
    "shared-expert RS / AR (2 transposes per MoE layer; no replicated latent_up weights).",
    "Async comm (ShardQ context push on the copy engine) hides under the attention compute of the same layer; all other "
    "comm is exposed unless O1 (two microbatches, t_layer = max(compute, comm), weights streamed twice) is on. ShardQ "
    "prefill uses slice-local q_b / W_UK / g with all-head weights (no q scatter) and an out gather before the head-TP "
    "o_proj.",
    "DcpExchange top-k merge bytes 1536 B per row per peer (inferred); sharded verify 100 B per row per peer. DSA-DP "
    "decode push_in/out move 0 bytes (SP shards hold whole requests, the holder is the owner); prefill pays (N-1)/N.",
    "Capacity: HBM budget 272.48 GiB (measured free at vLLM start) - resident weights - 16 GiB reserve (activations, "
    "graphs, symmetric buffers, MegaMoE arena). r_max is per GPU (a group of N holds N x r requests). SOL = byte-exact "
    "state; achievable = the packed hybrid block pool at the best pinnable block size (multiples of 64): whole "
    "bs-token rows of all 24 DSA layers per request (DSA rows ceil((L+1+k)/(bs x DCP)), G >= 3 KDA state groups x "
    "(1 + k) spec slots, drafter SWA ring; deployed D1 24 / 36 and D2 36 / 52 rows reproduced); the SP path is "
    "charged the byte-exact state (its measured 13 rows are not G x (1 + k) state rows).",
    "MegaMoE (EP) is one persistent kernel: layer = max(expert ops, dispatch + combine bytes / NVLink) + combine drain "
    "hop + one expert's worth of the shorter phase (moe lane: the fused kernel hides the link under the GEMMs).",
    "P1-type TP prefill: o_proj reduce-scatters in the GEMM, AttnRes runs on the 1/N token shard, a push AG rebuilds the "
    "rows (RS + AG per sublayer instead of an AR); TP decode (sdec) keeps the fused AR and all-row AttnRes.",
    "Prefill chunks are packed from AgentX requests with 5,456 fresh tokens each over a 72.9K prefix (fresh-weighted "
    "mean); L_ref = 78.4K. Headline = best chunk size per engine in {4K..64K} that fits HBM (0.5 MB per row of live "
    "activations assumed). KDA state checkpoint export for the prefix cache is counted (CKPT=1). KDA core prefill has a "
    "chunk-serial latency floor (FlashKDA K2: ~0.49 us per 16-token chunk, kda_segscan.py) relieved by segment-parallel "
    "scan (3x K2 work + scan) - it matters for TTFT and few-sequence chunks only.",
    "Decode reference context 135K (mission); r_max also reported at the OSL-weighted 189K. AL = golden 3.08/3.81/4.16 "
    "at k=3/5/7. Draft = one parallel 5-layer pass on k query rows + k sequential Markov steps.",
    "DSA-DP owner balance and DP lock-step padding are taken as perfect (SOL); D10 split indexer therefore only adds "
    "comm here - its value is under owner skew, which the SOL model does not price.",
    "PP (P7) stages are balanced by a DP partition over the per-layer times; the boundary push carries the residual + "
    "AttnRes block state (1 + ceil(l/12) rows of 14,336 B per token). Fill/drain bubbles ignored (many chunks in flight).",
    "AFD (D11): two microbatches ping-pong; per MoE layer max(A layer, F experts, A<->F transfer) per half batch, "
    "transfer = max(egress, ingress) per side (full duplex); best of the A:F splits 56:16 / 48:24 / 40:32 per r; one "
    "capacity function (kv.afd_capacity) reports r_max per split.",
    "Anchors are sanity checks only (never fitted): D1 e2e (21 runs), C512 D3 decode (r = B/4 per DP4 rank), C512 P2 "
    "prefill (computed prompt tokens) and PR #20 (24-layer layout ratios); the page flags any anchor where the "
    "achievable column is slower than measured.",
]
OPEN_Q = [
    "MoE activations for mxfp4_release prefill: MXFP8 (headline) or NVFP4 (the Rubin backend above 512 tokens/rank)?",
    "C512 P2 prefill slope: the model's marginal cost (achievable ~16.7 ms per 1K chunk tokens, SOL ~13.5) is steeper "
    "than the quoted 11 ms/1K lower-bound fit; no per-step prefill logs exist to rebuild the measured slope.",
    "SOL per-hop latency: no public NVLink latency figure; both columns use our measured one-way flag (2.86 us).",
    "Dispatch dedup with per-rank FP32 pre-combine (O2) changes the BF16 rounding order of the routed sum - acceptable as 'order may differ'?",
    "KDA core precision: FlashKDA stages the FP32 state into BF16 MMAs in prefill; FP32/TF32 math would ~2x the prefill KDA core.",
    "DcpExchange per-row bytes (X_topk) and the drafter forward row count per verify outcome are inferred, not read from kernels.",
    "Rack-wide EP (D7/D8/P6) imbalance factor: SOL uses 1.0; which measured EPLB max/mean should the achievable column use?",
    "KDA segment-scan context parallel for long prompts (P8) reorders the FP32 recurrence across ranks.",
    "The FP32 SIMT rate and the MXFP4 GEMM rate are not measured on VR200; die-local 17.1 TB/s comes from a code docstring.",
    "Spec values are NVIDIA's preliminary public figures and may change.",
]
ANCHORS_D1 = os.path.join(HERE, "inputs", "anchors_d1.json")   # inputs/anchors_src/extract_d1.py
C512_DECODE = {"4": 21.4, "16": 25.1, "32": 25.2, "50": 30.5, "64": 31.7}   # release C512 step bench, B per DP rank
PR20 = [  # Inferact/nvidia-vllm-agentx #20: 24-layer real-weight model, AgentX c3072, fresh tok/s/GPU
    ("DEP12", 50.0e3, dict(N=1, residual="dep", dsa="local", moe_dense="rep", E=12)),
    ("TP4xDP3 DSA-DP", 46.1e3, dict(N=4, residual="sp", dsa="dsadp", moe_dense="rep", E=12)),
    ("DP4xTP4/DCP4", 41.4e3, dict(N=4, residual="sp", dsa="dcp", moe_dense="rep", E=16)),
    ("TP8+DCP8", 32.2e3, dict(N=8, residual="tp", dsa="dcp", moe_dense="tokenshard", E=8)),
]


def _row(label, unit, measured, sol, ach, faster_is="lower", feasible=True, note=""):
    """ratio = model-achievable speed / measured speed (>= 1 expected: the model is an upper bound)."""
    if faster_is == "lower":
        ra, rs = measured / ach, measured / sol
    else:
        ra, rs = ach / measured, sol / measured
    return dict(label=label, unit=unit, measured=measured, sol=sol, ach=ach, ratio_ach=ra, ratio_sol=rs,
                feasible=feasible, warn=bool(ra < 1.0), note=note)


def run_anchors():
    from topologies import Topo
    out = []
    # (1) C512 hybrid decode step bench: B is requests per DP rank = one TP4 group of 4 GPUs -> r = B / 4
    t = ALL["D3"]
    cap = kv.capacity(t, "mxfp4_release", 3, Dec.L_DEC)["rmax"]
    rows = []
    for B, ms in C512_DECODE.items():
        r = float(B) / t.N
        s = Dec.run_point(t, "mxfp4_release", "sol", r, 3)
        h = Dec.run_point(t, "mxfp4_release", "achievable", r, 3)
        rows.append(_row("B%s per DP4 rank (r=%g)" % (B, r), "ms/step", ms, s["t_ms"], h["t_ms"],
                         feasible=r <= cap + 1e-9, note="" if r <= cap + 1e-9 else "r > r_max %.1f" % cap))
    out.append(dict(name="C512 hybrid decode step (D3, k=3, L=135K)", rows=rows,
                    note="release C512 step bench; B = requests per DP rank (a TP4 group of 4 GPUs), so r = B/4"))
    # (2) D1 e2e (TP16+DCP16 sdec): step = aiperf mean ITL x engine-logged mean AL at the logged mean running batch
    with open(ANCHORS_D1) as f:
        d1 = json.load(f)
    t = ALL["D1"]
    rows = []
    for a in d1:
        k = int(a["k"])
        s = Dec.run_point(t, "mxfp4_release", "sol", a["r"], k)
        h = Dec.run_point(t, "mxfp4_release", "achievable", a["r"], k)
        rows.append(_row("%s %s (B %.1f, k%d, AL %.2f)" % (a["sweep"].split("-")[1], a["label"], a["B"], k, a["AL"]),
                         "ms/step", a["step_ms"], s["t_ms"], h["t_ms"], note=a["run"]))
    out.append(dict(name="D1 e2e decode (TP16+DCP16 sdec, mxfp4, L=135K model)", rows=rows,
                    note="21 frozen/release AgentX runs; measured step includes host/scheduler time and real contexts "
                         "(not 135K); a ratio < 1 means the model column is slower than shipped code (flagged)"))
    # (3) C512 hybrid prefill (P2): computed prompt tokens (incl. Mooncake-miss recompute), not fresh tokens
    t = ALL["P2"]
    rows = []
    for lab, act in (("MXFP8 experts (model headline)", None), ("NVFP4 act (as deployed >512 tok/rank, O4)", "nvfp4")):
        s = Pre.run_point(t, "mxfp4_release", "sol", 15600, Pre.L_REF, moe_act=act)
        h = Pre.run_point(t, "mxfp4_release", "achievable", 15600, Pre.L_REF, moe_act=act)
        rows.append(_row("prompt tok/s/GPU, F=15.6K, " + lab, "tok/s/GPU", 8300.0, s["tok_s_gpu"], h["tok_s_gpu"],
                         faster_is="higher"))
    sl = {}
    for col in COLS:
        a8 = Pre.run_point(t, "mxfp4_release", col, 8192, Pre.L_REF)["t_ms"]
        a16 = Pre.run_point(t, "mxfp4_release", col, 16384, Pre.L_REF)["t_ms"]
        sl[col] = (a16 - a8) / 8.192
    rows.append(_row("marginal step ms per 1K chunk tokens (8K->16K)", "ms/1K", 11.0, sl["sol"], sl["achievable"],
                     note="measured slope is a lower-bound fit (165 ms + 11 ms/1K), not rebuilt from per-step logs "
                          "(none recorded); ratio < 1 = model steeper than the fit (open)"))
    out.append(dict(name="C512 hybrid prefill (P2 DP4 x TP4/DCP4, mxfp4)", rows=rows,
                    note="engine logs ~32.6K computed prompt tok/s per DP4 engine = ~8.3K/GPU incl. recompute of "
                         "Mooncake misses; the fresh-token rate is 4.34 req/s x 5,456 / 4 = ~5.9K/GPU"))
    # (4) PR #20 (24-layer model): first 24 layers + embed/final/sync, 16K tokens per GPU per step
    rows = []
    base = None
    res = []
    for lab, meas, kw in PR20:
        tt = Topo("PR20-" + lab, lab, "prefill", existing=True, **kw)
        r_ = {}
        for col in COLS:
            F = 16384 * tt.N
            p = Pre.run_point(tt, "mxfp4_release", col, F, Pre.L_REF, detail=True)
            ms = sum(p["per_layer_ms"][:24]) + p["glob_ms"]["embed"] + p["glob_ms"]["final"] + p["glob_ms"]["step"]
            r_[col] = F / (tt.N * ms * 1e-3)
        res.append((lab, meas, r_))
    m0, a0 = res[0][1], res[0][2]["achievable"]
    for lab, meas, r_ in res:
        rows.append(_row(lab + " (rel. to DEP12: meas %.2f, model %.2f)" % (meas / m0, r_["achievable"] / a0),
                         "tok/s/GPU", meas, r_["sol"], r_["achievable"], faster_is="higher"))
    out.append(dict(name="PR #20 prefill, 24-layer model (mxfp4, 16K tokens per GPU per step)", rows=rows,
                    note="only layout-comparative measurement; validates the prefill ranking (model drafter excluded)"))
    return out


def run_meta(dec):
    anchors = run_anchors()
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
        reserve_GiB=kv.RESERVE_GIB, agentx=Mdl.AGENTX, scope=SCOPE, scope_rule="<= %d GPUs per instance" % EP_MAX,
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
    cal = run_calibrated(pre, dec)
    print("calibrated + ranking done %.1fs" % (time.time() - t0))
    for name, obj in (("prefill", pre), ("decode", dec), ("optables", opt), ("meta", meta), ("calibrated", cal)):
        with open(os.path.join(OUT, name + ".json"), "w") as f:
            json.dump(rnd(obj), f, separators=(",", ":"))
        print("wrote out/%s.json (%.0f KB)" % (name, os.path.getsize(os.path.join(OUT, name + ".json")) / 1e3))
    summary(pre, dec)
    summary_ranking(cal)
    for a in meta["anchors"]:
        for r in a["rows"]:
            if r["warn"]:
                print("WARNING anchor %s / %s: model achievable is slower than measured (ratio %.2f)"
                      % (a["name"], r["label"], r["ratio_ach"]))


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
        print("== %s <=%d-GPU scope: prefill ranking (O1+O2, achievable) / decode best per r (k=3, achievable, base)"
              % (ck, EP_MAX))
        print("  " + "  ".join("%s %.1fK" % (tid, v / 1e3) for tid, v, _ in pre["rank_scope"]["%s|achievable|O1+O2" % ck][:6]))
        b = dec["best_scope"]["%s|3|achievable|base" % ck]
        print("  " + " ".join("r%g:%s %.0f" % (x["r"], x["topo"], x["tps_gpu"]) for x in b))


# ---------------------------------------------------------------------------------------------- calibrated + ranking
BANDS = [("<=0.5", 0.0, 0.5), ("1-2", 1.0, 2.0), ("3-8", 3.0, 8.0), ("12-24", 12.0, 24.0), ("32-64", 32.0, 64.0)]
COLS3 = ("sol", "achievable", "calibrated")
EP16_GPUS = 16


def _pt(p):
    return None if p is None else dict(t_ms=p["t_ms"], tps_gpu=p["tps_gpu"], tps_user=p["tps_user"])


def _decode_cal(ck, dec, k=3):
    """Per decode topology in scope: grid points with sol / achievable / calibrated (base variant, k=3)."""
    res_s = {tid: {round(p["r"], 6): p for p in pts} for tid, pts in dec["res"]["%s|%d|sol|base" % (ck, k)].items()}
    res_a = {tid: {round(p["r"], 6): p for p in pts} for tid, pts in dec["res"]["%s|%d|achievable|base" % (ck, k)].items()}
    out = {}
    for t in DECODE:
        if t.id not in SCOPE:
            continue
        cap_a = kv.capacity(t, ck, k, Dec.L_DEC, col="achievable")["rmax"]
        rcal, how = Cal.decode_rmax(t, ck, k, cap_a)
        rows = []
        for r in Dec.R_GRID:
            key = round(r, 6)
            ps, pa = res_s[t.id].get(key), res_a[t.id].get(key)
            if ps is None and pa is None:
                continue
            pc = None
            if pa is not None and r <= rcal + 1e-9:
                pc = Cal.decode_calibrated(t, ck, pa)
            rows.append(dict(r=r, sol=_pt(ps), achievable=_pt(pa), calibrated=pc))
        out[t.id] = dict(gpus=t.gpus, rmax=dict(sol=kv.capacity(t, ck, k, Dec.L_DEC)["rmax"], achievable=cap_a,
                                                 calibrated=rcal), rmax_cal_method=how,
                         front=Cal.FRONT.get(t.id), note=Cal.NOTES.get(t.id, ""), points=rows)
    return out


def _band_rank(per, col, ids, tags=None):
    """Per band: the best topology at every r grid point of the band (by tok/s/GPU; equal r = equal requests per
    GPU, so this is also the best interactivity). Band winner = the topology that wins the most r points of the band;
    ties go to the higher mean of tps_gpu / best tps_gpu over the band's points (0 where over capacity).
    tags: for the calibrated column, only points whose tag is in tags compete ('extrapolated' never does)."""
    out = {}
    for name, lo, hi in BANDS:
        rs = [r for r in Dec.R_GRID if lo - 1e-9 <= r <= hi + 1e-9]
        by_r, score, wins = [], {}, {}
        for r in rs:
            c = sorted(((x[col]["tps_gpu"], tid, x[col].get("tag")) for tid in ids for x in per[tid]["points"]
                        if abs(x["r"] - r) < 1e-9 and x[col] is not None
                        and (tags is None or x[col].get("tag") in tags)), reverse=True)
            if not c:
                continue
            by_r.append(dict(r=r, topo=c[0][1], tag=c[0][2], tps_gpu=c[0][0],
                             tps_user=c[0][0] / r, runner=c[1][1] if len(c) > 1 else None,
                             margin=(c[0][0] / c[1][0] - 1) if len(c) > 1 else None))
            wins[c[0][1]] = wins.get(c[0][1], 0) + 1
            for v, tid, _ in c:
                score[tid] = score.get(tid, 0.0) + v / c[0][0]
        if not by_r:
            out[name] = None
            continue
        n = len(by_r)
        ranked = sorted(score, key=lambda tid: (-wins.get(tid, 0), -score[tid]))
        out[name] = dict(winner=ranked[0], wins=wins.get(ranked[0], 0), n_r=n, score=score[ranked[0]] / n,
                         runner=ranked[1] if len(ranked) > 1 else None,
                         runner_score=score[ranked[1]] / n if len(ranked) > 1 else None, by_r=by_r)
    return out


def run_calibrated(pre, dec):
    out = dict(rule="calibrated = achievable x per-component microbench derates (calibrate.py); scope <= %d GPUs; "
                    "base variant (no O1/O2/O3: the measured code has none); decode k=3 L=135K; prefill at L=78.4K, "
                    "chunk capped at the largest measured chunk of the path (64K reported as extrapolated); "
                    "'extrapolated' points never win a band; measured_only = D1/D2/D3/D3-32 measured points only; "
                    "nvfp4_modelopt calibrated = unmeasured transfer" % EP_MAX,
               derates=Cal.summary(), bands=[b[0] for b in BANDS], decode={}, prefill={}, rank={})
    for ck in CKS:
        per = _decode_cal(ck, dec)
        out["decode"][ck] = per
        ids32 = list(per)
        ids16 = [t for t in ids32 if ALL[t].gpus <= EP16_GPUS]
        rk = {}
        for scope, ids in (("scope32", ids32), ("scope16", ids16)):
            rk["decode|sol|" + scope] = _band_rank(per, "sol", ids)
            rk["decode|achievable|" + scope] = _band_rank(per, "achievable", ids)
            rk["decode|calibrated|" + scope] = _band_rank(per, "calibrated", ids, tags=("measured", "projected"))
        rk["decode|calibrated|measured_only"] = _band_rank(per, "calibrated", ids32, tags=("measured",))
        pp = {}
        for t in PREFILL:
            if t.id not in SCOPE:
                continue
            ra = pre["res"]["%s|achievable|base" % ck][t.id]
            rs = pre["res"]["%s|sol|base" % ck][t.id]
            path = Cal.prefill_path(t.id)
            Fcap = min(ra["bestF"], Cal.prefill_table(path, ck == "nvfp4_modelopt")["Fmax"])
            c = Cal.prefill_calibrated(t, ck, Fcap, Pre.L_REF)
            c64 = Cal.prefill_calibrated(t, ck, ra["bestF"], Pre.L_REF)
            pp[t.id] = dict(gpus=t.gpus, bestF=ra["bestF"], sol=dict(tok_s_gpu=rs["best"]["tok_s_gpu"], bestF=rs["bestF"]),
                            achievable=dict(tok_s_gpu=ra["best"]["tok_s_gpu"], t_ms=ra["best"]["t_ms"]),
                            achievable_O1O2=pre["res"]["%s|achievable|O1+O2" % ck][t.id]["best"]["tok_s_gpu"],
                            calibrated=c, calibrated_bestF_extrapolated=c64, path=path)
        out["prefill"][ck] = pp
        for col in COLS3:
            def val(v):
                return v[col]["tok_s_gpu"] if v[col] else 0.0
            r32 = sorted(((tid, val(v)) for tid, v in pp.items()), key=lambda x: -x[1])
            rk["prefill|%s|scope32" % col] = r32
            rk["prefill|%s|scope16" % col] = [x for x in r32 if ALL[x[0]].gpus <= EP16_GPUS]
        rk["prefill|calibrated|measured_only"] = [x for x in rk["prefill|calibrated|scope32"] if x[0] in Cal.PRE_MEASURED]
        out["rank"][ck] = rk
    return out


def summary_ranking(cal):
    for ck in CKS:
        rk = cal["rank"][ck]
        print("\n== %s ranking, <=32 GPUs (EP16-only winner in brackets when different)%s"
              % (ck, "  [calibrated = unmeasured transfer]" if ck != "mxfp4_release" else ""))
        for col in ("achievable", "calibrated"):
            a, b = rk["prefill|%s|scope32" % col], rk["prefill|%s|scope16" % col]
            print("  prefill %-10s " % col + "  ".join("%s %.1fK" % (t, v / 1e3) for t, v in a[:5])
                  + ("" if a[0][0] == b[0][0] else "   [EP16: %s %.1fK]" % (b[0][0], b[0][1] / 1e3)))
            x, y = rk["decode|%s|scope32" % col], rk["decode|%s|scope16" % col]
            print("  decode  %-10s " % col + "  ".join(
                "%s:%s%s" % (n, x[n]["winner"] if x[n] else "-",
                             "" if not x[n] or not y[n] or x[n]["winner"] == y[n]["winner"] else "[%s]" % y[n]["winner"])
                for n in cal["bands"]))
        m = rk["decode|calibrated|measured_only"]
        print("  decode  measured-only " + "  ".join("%s:%s" % (n, m[n]["winner"] if m[n] else "-") for n in cal["bands"]))


if __name__ == "__main__":
    main()
