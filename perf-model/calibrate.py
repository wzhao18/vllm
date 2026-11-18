"""Calibrated projection column: achievable x per-component microbench derates (inputs/derating.json, lane rows).

Decode (per point t, r):
    calibrated = experts_ach x d_experts  +  rest_ach x d_rest(front path, r)
  * experts = the routed-expert component (cost.py: whole MegaMoE layer for EP layouts, routed-expert ops for MoE-TP).
    d_experts: EP -> the MegaMoE family derate (moe lane rows: whole fused layer measured / this model's layer, by EP,
    tokens per rank and quant; interpolated in log T) x 1.2 for the vendored in-step kernel of the release tree
    (gap512); MoE-TP (D1) -> the sdec MoE-TP16 kernel family (1.19).
  * rest = everything else; d_rest is recomputed from the measured step of the front path at that r:
        d_rest = (measured - experts_ach x d_experts) / rest_ach         (log-r interpolation between measured r)
    front paths: TP + DCP sdec front (D1) for D1 / D8-x; SP path D3 (EP16) for D3, D5, D6, D7-x-16, D4-16, D9-16;
    SP path D3-32 (EP32, wave 2) for D3-32, D7-x-32, D4-32, D9-32.
    So the calibrated column reproduces every measured D1 / D3 step exactly, and an unmeasured layout takes the
    measured expert-kernel family for its own experts (EP MegaMoE) and the measured front derate for the rest.
  * D2: the measured run is MoE-TP8, the model D2 is EP8 -> whole-path derate (what D2 runs today).
  * Tags: 'measured' (D1 / D2 / D3 / D3-32 inside the measured r range), 'projected' (component transfer to an unmeasured
    layout, or MegaMoE family transferred from EP16 to EP32), 'extrapolated' (r outside the front path's measured
    range or tokens per rank outside the measured MegaMoE range: derates clamped). Band winners skip 'extrapolated'
    points; run.py also reports measured-path-only winners.
Prefill: calibrated step = achievable step x the measured step derate of the path at chunk F (log-F interpolation
  between the measured F8K / F16K / F32K rows at L78K; P4 at its measured 4 x 4096 request shape). The ranking uses the
  chunk capped at the largest measured F of the path; 64K is reported separately as extrapolated. Analogs:
      P6-16, P6-32 -> P4-2    P6-4-16 -> P4-4    P9, P9-x, P3-32 -> P3-16   (P2b / P5-4 / P5-8 measured in wave 2;
      P3-32 refused at load on the release tree: no EP32 MegaMoE knob record)
  P6-x are projected WITHOUT dispatch dedup (not implemented; measured dedup was slower).
Capacity: D1 / D2 / D3 / D3-32 take their measured r_max. TP-front and DEP layouts get an itemised transfer of D1's non-KV
  memory (workload.json hbm: gpu-memory-utilization 0.90, non-torch 23.97 GiB, activations 7.25 GiB, vision tower
  0.84 GiB, replicated latent/router where the model shards them, MoE-TP intermediate padding only for MoE-TP layouts)
  over the deployed (unpinned) packed layout; SP-path layouts take D3's (EP16) or D3-32's (EP32) measured /
  achievable ratio (projected).
nvfp4_modelopt: no step was measured on it. Prefill uses the measured NVFP4-activation MoE rows of the release
  checkpoint (vs this model with O4 NVFP4 activations), decode uses the nvfp4 MegaMoE family rows and the mxfp4 front
  derates; the whole column is labelled 'unmeasured transfer'.
No e2e data, no scheduling model.
"""
import copy
import json
import math
import os
import re

import cost
import decode as Dec
import kv
import model as Mdl
import prefill as Pre
from hw import GiB
from topologies import ALL

HERE = os.path.dirname(os.path.abspath(__file__))
CK = "mxfp4_release"          # the checkpoint every wave-1 step bench ran
K = 3

# decode lane README "Capacity finding": measured max r at 135K (bench KV pool, production gpu-memory-utilization)
MEASURED_RMAX = {("D1", 3): 15.6, ("D1", 7): 9.7, ("D2", 3): 9.9, ("D2", 7): 6.7, ("D3", 3): 33.0,
                 ("D3-32", 3): 44.8}          # wave 2: 2329 blocks/rank at gmu 0.93, 13 blocks/request
# measured per-request block rows (decode lane) and their block size: (rows per GPU, bs)
MEASURED_ROWS = {("D1", 3): (24, 1024), ("D1", 7): (36, 1024), ("D2", 3): (36, 1024), ("D2", 7): (52, 1024),
                 ("D3", 3): (13, 4096), ("D3-32", 3): (13, 4096)}
# workload.json hbm (D1 decode TP16 release role), GiB per GPU
HBM = dict(device=278.54, gmu=0.90, weights_measured=131.18, non_torch=23.97, activations=7.25, vision=0.84)
SDEC_MOE_TP_DERATE = 1.19      # derating.json by_family decode 'experts (MoE-TP16 sdec rubin, D1)'
# The release tree runs the vLLM-vendored MegaMoE (megamoe_rubin / deep_gemm situ) in step, the moe-lane rows are the
# newest FlashInfer build (6e438852b). gap512: in-step vendored 109 / 115 / 122 us vs B10 91-95 us at 32 / 64 / 80
# tokens per rank; decode prof3 r1 ~92 us vs B10 77.8 us at 4 -> vendored / B10 ~1.2 in decode. Calibrated decode =
# release-tree kernels, so the EP expert family is the moe-lane derate x 1.2 (this also keeps the vendored-kernel
# excess out of the D3 front derate).
VENDORED_MEGAMOE_DECODE = 1.2

# front path per layout: the measured step whose non-expert derate it takes. EP32 layouts on SP / DSA-DP / DEP
# take the measured EP32 SP path D3-32 (wave 2), EP <= 16 ones D3; TP-front layouts take D1.
FRONT = {"D1": "D1", "D8-16": "D1", "D8-32": "D1", "D3": "D3", "D3-32": "D3-32", "D5": "D3", "D6": "D3",
         "D4-16": "D3", "D4-32": "D3-32", "D9-16": "D3", "D9-32": "D3-32", "D7-4-16": "D3", "D7-2-16": "D3",
         "D7-8-16": "D3", "D7-4-32": "D3-32", "D7-2-32": "D3-32", "D7-8-32": "D3-32"}
DEC_MEASURED = ("D1", "D2", "D3", "D3-32")
PRE_ANALOG = {"P6-16": "P4-2", "P6-32": "P4-2", "P6-4-16": "P4-4",
              "P9": "P3-16", "P9-16": "P3-16", "P9-32": "P3-16", "P3-32": "P3-16"}
PRE_MEASURED = ("P1", "P2", "P3-16", "P4-4", "P4-2", "P2b", "P5-4", "P5-8")    # P2b / P5-x: wave 2
NOTES = {"D2": "measured path is MoE-TP8 (sdec refuses EP8); the model D2 is EP8 - calibrated = what D2 runs today",
         "D1": "k=3 derates; the k7 r8 MoE path cliff (2.1x) is not in the k=3 curve",
         "EP32": "MegaMoE family transferred from EP16 rows (no EP32 measurement)"}


def _load(p):
    with open(os.path.join(HERE, p)) as f:
        return json.load(f)


DER = _load("inputs/derating.json")
MOE_ROWS = _load("microbench/results/moe/rows.json")


def _lerp_log(pts, x):
    """pts = [(x, y)] sorted; log-x interpolation, clamped; returns (y, clamped)."""
    if x <= pts[0][0]:
        return pts[0][1], x < pts[0][0] * (1 - 1e-9)
    if x >= pts[-1][0]:
        return pts[-1][1], x > pts[-1][0] * (1 + 1e-9)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            w = (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0))
            return y0 + w * (y1 - y0), False
    return pts[-1][1], True


# ---------------------------------------------------------------------------------------------- MegaMoE family
_MM = {}


def _row_layer(x):
    """This model's achievable MegaMoE layer (s) at a moe-lane row's shape."""
    ck, tid, reg = x["topology"].split("|")
    t = ALL[tid]
    if reg.startswith("decode"):
        r = float(reg.split("r")[1])
        env, ph = Mdl.env_decode(r * t.N, K, Dec.L_DEC), "decode"
    else:
        env, ph = Mdl.env_prefill(16384, Pre.L_REF, f_req=4096 if tid.startswith("P4") else Pre.F_REQ), "prefill"
    return cost.evaluate(t, ph, ck, "achievable", env)["megamoe"]["layer"], ck


def megamoe_table(quant):
    """{EP: [(tokens per rank, measured / model layer)]} from the moe-lane rows (no-dedup layouts only)."""
    if quant not in _MM:
        tab = {}
        for x in MOE_ROWS:
            if x["quant_kind"] != quant or x["topology"].split("|")[1].startswith("P6"):
                continue
            lay, _ = _row_layer(x)
            tab.setdefault(x["ep"], {})[x["tokens_per_rank"]] = x["measured_us"] * 1e-6 / lay
        _MM[quant] = {ep: sorted(v.items()) for ep, v in tab.items()}
    return _MM[quant]


def megamoe_derate(ck, E, T):
    """(derate, tag) of the MegaMoE family at EP E and T tokens per rank."""
    quant = "nvfp4w_nvfp4a" if ck == "nvfp4_modelopt" else "mxfp4w_mxfp8a"
    tab = megamoe_table(quant)
    ep = E if E in tab else min(tab, key=lambda e: abs(e - E))
    d, clamped = _lerp_log(tab[ep], T)
    tag = "extrapolated" if clamped else ("projected" if ep != E else "measured")
    return d, tag


def experts_derate(t, ck, T):
    if t.experts == "tp":
        return SDEC_MOE_TP_DERATE, "measured"
    d, tag = megamoe_derate(ck, t.E, T)
    return d * VENDORED_MEGAMOE_DECODE, tag


# ---------------------------------------------------------------------------------------------- decode fronts
def _measured_points(tid, k=K):
    pts = {}
    for p in DER["by_topology"][tid]["points"]:
        if p["k"] == k and p.get("moe_quant", "mxfp4_mxfp8") == "mxfp4_mxfp8":
            pts.setdefault(p["r_per_gpu"], []).append(p["measured_ms"])
    return [(r, sorted(v)[len(v) // 2]) for r, v in sorted(pts.items())]


_FRONT = {}


def front_table(path):
    """[(r, d_rest)] of a measured front path (D1 / D3): rest derate with the expert family taken out."""
    if path not in _FRONT:
        out = []
        for r, ms in _measured_points(path):
            p = Dec.run_point(ALL[path], CK, "achievable", r, K)
            de, _ = experts_derate(ALL[path], CK, p["rows_per_rank"])
            rest = p["t_ms"] - p["experts_ms"]
            out.append((r, (ms - p["experts_ms"] * de) / rest))
        _FRONT[path] = out
    return _FRONT[path]


_D2 = {}


def d2_table():
    if not _D2:
        _D2["t"] = [(r, ms / Dec.run_point(ALL["D2"], CK, "achievable", r, K)["t_ms"]) for r, ms in _measured_points("D2")]
    return _D2["t"]


def decode_calibrated(t, ck, p):
    """Calibrated point from an achievable decode point p (Dec.run_point output). Returns dict or None."""
    r = p["r"]
    if t.id == "D2":
        d, cl = _lerp_log(d2_table(), r)
        tc = p["t_ms"] * d
        tag = "extrapolated" if cl else "measured"
        parts = dict(path="D2 (MoE-TP8 run)", d_step=d)
    else:
        path = FRONT.get(t.id)
        if path is None:
            return None
        dr, cl = _lerp_log(front_table(path), r)
        de, etag = experts_derate(t, ck, p["rows_per_rank"])
        tc = p["experts_ms"] * de + (p["t_ms"] - p["experts_ms"]) * dr
        if cl:
            tag = "extrapolated"
        elif t.id == path:
            tag = "measured"                      # the whole step is measured at this r (interpolated between r)
        elif etag == "extrapolated":
            tag = "extrapolated"
        else:
            tag = "projected"
        parts = dict(path=path, d_rest=dr, d_experts=de, experts_tag=etag, experts_ach_ms=p["experts_ms"])
    if ck != CK and tag == "measured":
        tag = "projected"                         # nothing was measured on this checkpoint
    AL = Mdl.AL_GOLDEN[K]
    out = dict(t_ms=tc, tps_gpu=r * AL / (tc * 1e-3), tps_user=AL / (tc * 1e-3), tag=tag, **parts)
    if ck != CK:
        out["transfer"] = "unmeasured transfer (no nvfp4_modelopt step measured)"
    return out


# ---------------------------------------------------------------------------------------------- capacity
def _extra_weights_GiB(t, ck):
    """Non-KV memory the deployed code holds beyond this model's resident weights (itemised from D1)."""
    C = Mdl.C
    extra = HBM["vision"]
    if t.experts == "tp" and t.N > 1:            # MoE-TP pads intermediate per partition to >= 256 (model.py)
        per = C["I_E"] / t.N
        if per < 256:
            _, by = cost.resident_weights(t, "decode", ck)
            extra += by["experts"] * (256.0 / per - 1) / GiB
    if t.moe_dense in ("tp", "tpfront") and t.N > 1:  # latent up/down + router stay replicated in the code
        ckc = Mdl.ckpt_cfg(ck)
        env = dict(Mdl.env_decode(1, K, 1000), T=1.0, B=1.0, nseq=1.0, R=1.0, Tc=1.0, Tq=1.0)
        full = sum(Mdl.evaluate(Mdl.OPS_BY_ID[o], env, ckc)["w"] * Mdl.count(Mdl.OPS_BY_ID[o], env)
                   for o in ("MOE.latent_down", "MOE.latent_up", "MOE.router"))
        extra += full * (1 - 1.0 / t.N) / GiB
    return extra


def decode_rmax(t, ck, k, rmax_ach):
    """(calibrated r_max, method)."""
    if (t.id, k) in MEASURED_RMAX:
        return MEASURED_RMAX[(t.id, k)], "measured"
    if t.id in ("D2",):
        return MEASURED_RMAX[("D2", 3)] * rmax_ach / kv.capacity(ALL["D2"], CK, 3, Dec.L_DEC, col="achievable")["rmax"], "measured k3 ratio"
    if t.residual == "sp":
        ref = "D3-32" if t.E >= 32 else "D3"
        f = MEASURED_RMAX[(ref, 3)] / kv.capacity(ALL[ref], CK, 3, Dec.L_DEC, col="achievable")["rmax"]
        return rmax_ach * f, "projected (%s ratio)" % ref
    w, _ = cost.resident_weights(t, "decode", ck)
    pool = HBM["device"] * HBM["gmu"] - (w / GiB + _extra_weights_GiB(t, ck)) - HBM["non_torch"] - HBM["activations"]
    b, _ = kv.packed_request_group_bytes(t, k, Dec.L_DEC)
    return max(pool, 0.0) * GiB / b, "projected (itemised D1 non-KV memory)"


# ---------------------------------------------------------------------------------------------- prefill
def _prefill_points(tid, nvfp4_act=False):
    """Measured L78K steps of a prefill path: [(F, f_req, measured ms)]."""
    out = []
    for p in DER["by_topology"][tid]["points"]:
        nv = p.get("moe_quant", "").startswith("nvfp4")
        if nv != nvfp4_act or "gap512" in p["case"] or "L78k" not in p["case"]:
            continue
        m = re.search(r"model \S+ F(\d+)(?: f(\d+))?", p["case"])
        if not m:
            continue
        out.append((int(m.group(1)), int(m.group(2)) if m.group(2) else Pre.F_REQ, p["measured_ms"]))
    return sorted(out)


_PRE = {}


def prefill_table(path, nvfp4_act=False):
    """[(F, derate)], f_req of the measured shape, max measured F."""
    key = (path, nvfp4_act)
    if key not in _PRE:
        pts = _prefill_points(path, nvfp4_act)
        if not pts:                              # no NVFP4-act row for this path: MXFP8-act derates (transfer)
            _PRE[key] = dict(prefill_table(path, False), fallback="MXFP8-act derates (no NVFP4-act row)")
            return _PRE[key]
        tab = []
        for F, f, ms in pts:
            a = Pre.run_point(ALL[path], CK, "achievable", F, Pre.L_REF, f_req=f,
                              moe_act="nvfp4" if nvfp4_act else None)["t_ms"]
            tab.append((F, ms / a))
        _PRE[key] = dict(tab=tab, f=pts[0][1], Fmax=max(F for F, _, _ in pts), pts=pts)
    return _PRE[key]


def no_dedup(t):
    """The shipped variant of a dedup layout (MegaMoE has no dispatch dedup)."""
    if not t.dedup:
        return t
    u = copy.copy(t)
    u.dedup = False
    return u


def prefill_path(tid):
    return tid if tid in PRE_MEASURED else PRE_ANALOG.get(tid)


def prefill_calibrated(t, ck, F, L, f_req=None):
    """Calibrated fresh tok/s/GPU at chunk F (base variant). F above the path's largest measured chunk is tagged
    'extrapolated' (derate clamped)."""
    path = prefill_path(t.id)
    if path is None:
        return None
    nv = ck == "nvfp4_modelopt"
    tb = prefill_table(path, nvfp4_act=nv)
    d, cl = _lerp_log(tb["tab"], F)
    f = f_req or (tb["f"] if t.id == path else Pre.F_REQ)
    u = no_dedup(t)
    a = Pre.run_point(u, ck, "achievable", F, L, f_req=min(f, F))
    step = a["t_ms"] * d
    tag = "extrapolated" if cl else ("measured" if t.id == path and any(F == x[0] for x in tb["tab"]) and not nv
                                     else "projected")
    out = dict(path=path, derate=d, F=F, f_req=f, t_ms=step, tok_s_gpu=F / (t.N * step * 1e-3),
               ach_shipped_t_ms=a["t_ms"], dedup_removed=u is not t, tag=tag, Fmax_measured=tb["Fmax"])
    if nv:
        out["transfer"] = "unmeasured transfer: NVFP4-act MoE rows of the release checkpoint"
    return out


def summary():
    out = {"decode_front": {p: [dict(r=r, d_rest=d) for r, d in front_table(p)] for p in ("D1", "D3", "D3-32")},
           "decode_D2": [dict(r=r, d_step=d) for r, d in d2_table()],
           "megamoe": {q: {str(ep): [dict(T=T, d=d) for T, d in v] for ep, v in megamoe_table(q).items()}
                       for q in ("mxfp4w_mxfp8a", "nvfp4w_nvfp4a")},
           "prefill": {p: [dict(F=F, d=d) for F, d in prefill_table(p)["tab"]] for p in PRE_MEASURED},
           "prefill_nvfp4_act": {p: [dict(F=F, d=d) for F, d in prefill_table(p, True)["tab"]] for p in PRE_MEASURED},
           "measured_rmax": {"%s|k%d" % k: v for k, v in MEASURED_RMAX.items()},
           "front": FRONT, "prefill_analog": PRE_ANALOG, "notes": NOTES, "hbm_items_GiB": HBM}
    return out
