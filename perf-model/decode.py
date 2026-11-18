"""Decode: step time vs requests per GPU r (each request = 1 + k verify rows + DSpark draft of k tokens).

    B_group = r * N requests on one TP group (N = 1 for DEP / DSA-CP); rows = B_group * (1 + k)
    step    = sum over layers (op roofline + comm) + drafter + verify + launch gaps   [cost.step]
    ITL     = step / AL          interactivity (tok/s/user) = AL / step          tok/s/GPU = r * AL / step
Capacity: r <= r_max (kv.capacity).  Variants: base | O2+O3 (dedup + die-local streaming) | O1+O2+O3 (+two-batch
overlap, evaluated at half batch twice).  D11 AFD picks the best A:F pool split per r.
"""
import cost
import model as Mdl
from kv import afd_capacity, capacity
from topologies import ALL

R_GRID = [0.0625, 0.125, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
K_LIST = [3, 5, 7]
L_DEC = Mdl.AGENTX["decode_ctx"]
VARIANTS = {"base": dict(o1=False, o2=False, o3=False), "O2+O3": dict(o1=False, o2=True, o3=True),
            "O1+O2+O3": dict(o1=True, o2=True, o3=True)}


def r_points(t, rmax):
    N = t.N
    pts = []
    for r in R_GRID:
        B = r * N
        if B < 1 - 1e-9 or abs(B - round(B)) > 1e-9 or r > rmax + 1e-9:
            continue
        pts.append(r)
    # last point exactly at capacity (integer group batch)
    Bcap = int(rmax * N)
    if Bcap >= 1:
        rc = Bcap / float(N)
        if not pts or rc > pts[-1] * 1.05:
            pts.append(rc)
    return pts


def run_point(t, ck, col, r, k, L=L_DEC, variant="base", detail=False):
    if t.afd:
        return run_afd(t, ck, col, r, k, L, variant)
    v = VARIANTS[variant]
    B = r * t.N
    if v["o1"] and B >= 2:
        a = _run(t, ck, col, r, k, L, True, v, detail)
        b = _run(t, ck, col, r, k, L, False, v, detail)
        return a if a["t_ms"] <= b["t_ms"] else b
    return _run(t, ck, col, r, k, L, False, v, detail)


def _run(t, ck, col, r, k, L, o1, v, detail):
    B = r * t.N
    env = Mdl.env_decode(B / 2.0 if o1 else B, k, L)
    res = cost.evaluate(t, "decode", ck, col, env, o2=v["o2"], o3=v["o3"], ops_detail=detail)
    st = cost.step(res, o1=o1, reps=2 if o1 else 1)
    AL = Mdl.AL_GOLDEN[k]
    out = dict(r=r, o1=o1, t_ms=st["t"] * 1e3, tps_gpu=r * AL / st["t"], tps_user=AL / st["t"],
               groups={g: x * 1e3 for g, x in st["groups"].items()}, hops=st["hops"], n_comm=st["n_comm"])
    if detail:
        out["fam"] = {f: x * 1e3 for f, x in st["fam"].items()}
        out["comm_by"] = {p: x * 1e3 for p, x in st["comm_by"].items()}
        out["comm_raw"] = {p: x * 1e3 for p, x in st["comm_raw"].items()}
        out["units"] = {kk: dict(comp=x["comp"] * 1e3, comm=x["comm"] * 1e3) for kk, x in cost.unit_summary(res).items()}
        out["per_layer_ms"] = [x * 1e3 for x in st["per_layer"]]
        out["ops"] = res["rows"]
        out["comm_rows"] = res["comm_rows"]
        out["comp_ms"], out["comm_ms"], out["kern_ms"] = st["comp"] * 1e3, st["comm"] * 1e3, st["kern"] * 1e3
    return out


def run_afd(t, ck, col, r, k, L, variant):
    """Attention-FFN disaggregation, two microbatches ping-pong.

    A-pool (nA GPUs, TP4 groups as D7): every op except routed experts.  F-pool (nF GPUs): experts only.
    Per MoE layer: t = 2 * max(A layer half, F experts half, A<->F transfer half); other layers 2 * A half."""
    v = VARIANTS[variant]
    AL = Mdl.AL_GOLDEN[k]
    ckc = Mdl.ckpt_cfg(ck)
    best = None
    q = 1 + k
    hw = cost.HWS[col]
    capA = afd_capacity(t, ck, k, L)["rmax_A"]
    C = Mdl.C
    for (nA, nF) in t.afd["choices"]:
        rA = r * (nA + nF) / nA
        B = rA * t.N
        if B < 2 or rA > capA + 1e-9:
            continue
        envh = Mdl.env_decode(B / 2.0, k, L)
        res = cost.evaluate(t, "decode", ck, col, envh, o3=v["o3"], exclude=cost.EXPERT_OPS, no_ep=True)
        st = cost.step(res)
        rows_tot = rA * nA * q / 2.0                           # half-batch rows of the whole A pool
        TF = rows_tot / nF                                     # rows per F GPU (16 routes each)
        e_act = cost.experts_per_gpu(nF) * Mdl.touched_frac(rows_tot)
        inter = TF * C["TOPK_E"] * C["I_E"] * ckc["exp_aB"]
        envF = dict(Mdl.env_decode(1, k, L), T=TF, E_ACT=e_act, MOE_SPILL=1 if inter > cost.L2_SPILL_B else 0)
        hbm = hw.hbm_die_local if v["o3"] else hw.hbm
        tf = 0.0
        for oid in cost.EXPERT_OPS:
            op = Mdl.OPS_BY_ID[oid]
            full = Mdl.evaluate(op, envF, ckc)
            dt = Mdl.op_dtype(op, ckc)
            b = full["w"] + full["a_in"] + full["a_out"]
            tf += max(full["flops"] / hw.peak[dt], b / hbm)
        m = B / 2.0 * q / t.N                                  # rows per A GPU per half batch
        f = 1.0
        if v["o2"]:
            f = nF * (1 - (1 - 1.0 / nF) ** 16) / 16.0
        # NVLink is full duplex: each side's time is max(egress, ingress) per direction
        xa = max(m * 16 * ckc["dispatch_row"] * f, m * 16 * 7168 * f)       # A: dispatch out, combine in
        xf = max(TF * 16 * ckc["dispatch_row"] * f, TF * 16 * 7168 * f)     # F: dispatch in, combine out
        x = 3 * hw.hop_s + max(xa, xf) / hw.nvl
        tot = 0.0
        wait = 0.0
        for (l, att, mlp), tl in zip(cost.LAYERS, st["per_layer"]):
            if mlp == "moe":
                lay = max(tl, tf, x)
                wait += lay - tl
            else:
                lay = tl
            tot += 2 * lay
        g = st["glob"]
        tot += 2 * (g["embed"] + g["final"] + g["drafter"] + g["step"])
        if best is None or tot < best["t"]:
            groups = {kk: 2 * vv for kk, vv in st["groups"].items()}
            groups["routed experts"] = groups.get("routed experts", 0.0) + 2 * wait
            best = dict(t=tot, split=(nA, nF), groups=groups)
    if best is None:
        return None
    return dict(r=r, t_ms=best["t"] * 1e3, tps_gpu=r * AL / best["t"], tps_user=AL / best["t"],
                groups={kk: vv * 1e3 for kk, vv in best["groups"].items()}, split=best["split"])


def curve(t, ck, col, k, variant, L=L_DEC):
    cap = capacity(t, ck, k, L)
    pts = []
    for r in r_points(t, cap["rmax"]):
        p = run_point(t, ck, col, r, k, L, variant)
        if p:
            pts.append(p)
    return cap, pts
