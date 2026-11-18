"""Cost engine: per-op roofline + per-primitive communication for one (topology, phase, checkpoint, hw column, step).

Per op, per GPU (explicit roofline, no fitted derating):
    t_op = max( FLOPs_gpu / peak[dtype],  bytes_gpu / HBM )
    bytes_gpu = W/div_w + (A_in + A_out)/div_row + KV/div_kv + State/div_st      (A = 0 for ops fused in SOL)
Per communication event (bytes per GPU over NVLink):
    t_comm = hops * hop_latency + max(bytes_in, bytes_out) / NVLink_per_direction
    (an all-reduce / reduce-scatter takes the cheaper of its implemented primitives for that column)
Per layer:  t_layer = sum(t_op) + sync comm + max(0, async comm - layer compute) + kernels * per_kernel_gap
O1 (two-microbatch overlap): the step is evaluated at half the rows and t_layer = max(compute, comm) per half,
counted twice (weights are streamed twice: that cost is real and kept).
"""
import math

import model as Mdl
from hw import HWS
from topologies import comm_events, op_map

LAYER_OF = {"global": "embed", "all": "attnres", "kda": "kda", "dsa": "dsa", "moe": "moe", "dense_mlp": "dense",
            "final": "final", "dspark": "drafter"}
ROW_B = 14336
COMM_OPS = ("L.sp_gather", "F.sp_gather_out")
EXPERT_OPS = ("MOE.fc1", "MOE.act", "MOE.fc2")
_EVAL_CACHE = {}


def _full(op, env, ck_key, ck):
    key = (op.id, ck_key, env["T"], env["B"], env["L"], env["k"], env["Lsum"], env["Tc"], env["E_ACT"], env["PREFILL"])
    r = _EVAL_CACHE.get(key)
    if r is None:
        r = Mdl.evaluate(op, env, ck)
        if len(_EVAL_CACHE) > 400000:
            _EVAL_CACHE.clear()
        _EVAL_CACHE[key] = r
    return r


def comm_time(e, hw):
    alts = e["alts"] if "alts" in e else [e]
    best = None
    for a in alts:
        t = a["hops"] * hw.hop_s + max(a["in_B"], a["out_B"]) / hw.nvl
        if best is None or t < best[0]:
            best = (t, a)
    return best


def expert_env(env, t, phase):
    """E_ACT for the per-GPU expert weight stream: experts this GPU holds that the step touches."""
    E = Mdl.C["E"]
    N = t.N
    if t.experts == "tp":
        rows = env["T"]
        return E * Mdl.touched_frac(rows) / N      # each GPU holds 1/N of every expert
    rows_domain = env["T"] / N * t.E            # rows across the EP domain
    return (E / t.E) * Mdl.touched_frac(rows_domain)


def evaluate(t, phase, ck_name, col, env, o2=False, o3=False, moe_act=None, imb=1.0, ops_detail=False,
             exclude=(), no_ep=False):
    """One step on one GPU of topology t. Returns dict with per-op rows, per-layer-type unit times, families."""
    hw = HWS[col]
    ck = Mdl.ckpt_cfg(ck_name, moe_act)
    ck_key = (ck_name, moe_act)
    hbm = hw.hbm_die_local if o3 else hw.hbm
    e_act = expert_env(env, t, phase)
    rows = []
    unit = {k: dict(comp=0.0, kernels=0.0) for k in LAYER_OF.values()}
    fam = {}
    tot_flops = tot_bytes = 0.0
    for op in Mdl.OPS:
        if not op.in_phase(phase) or op.id in COMM_OPS or op.id in exclude:
            continue
        cnt = Mdl.count(op, env)
        if cnt == 0:
            continue
        m = op_map(op.id, t, phase)
        if op.id in ("MOE.fc1", "MOE.fc2"):
            env2 = dict(env, E_ACT=e_act)
            full = Mdl.evaluate(op, env2, ck)
            w = full["w"]
        else:
            full = _full(op, env, ck_key, ck)
            w = full["w"] / m["w"]
        mult = imb if op.id in EXPERT_OPS else 1.0
        flops = full["flops"] / m["row"] * m["fl"] * mult
        a = (full["a_in"] + full["a_out"]) / m["row"] * mult
        kv = (full["kv_r"] + full["kv_w"]) / m["kv"]
        st = (full["st_r"] + full["st_w"]) / m["st"]
        if op.id == "DS.markov" and w <= 64 * 2 ** 20 and cnt > 1:
            w = w / cnt          # O7: the vocab shard of Markov w2/w1 stays in L2 across the k positions
        b = w + a + kv + st
        dt = Mdl.op_dtype(op, ck)
        tc = flops / hw.peak[dt]
        tm = b / hbm
        tt = max(tc, tm)
        lt = LAYER_OF.get(op.layer_type, "attnres")
        per = cnt if lt in ("embed", "final", "drafter") else 1.0   # per-layer types store one instance
        unit[lt]["comp"] += tt * per
        if not op.fused:
            unit[lt]["kernels"] += per
        f = Mdl.FAMILY[op.id]
        fam[f] = fam.get(f, 0.0) + tt * cnt
        tot_flops += flops * cnt
        tot_bytes += b * cnt
        if ops_detail:
            rows.append(dict(id=op.id, fam=f, n=cnt, flops=flops, w=w, a=a, kv=kv, st=st, dt=dt, t=tt,
                             bound="compute" if tc >= tm else "memory"))
    # communication
    ev_all = comm_events(t, phase, env, ck, imb=imb, o2=o2)
    if no_ep:
        ev_all["moe"] = [e for e in ev_all["moe"] if not (("prim" in e) and e["prim"].startswith(("P11", "P12")))]
        ev_all["step"] = []
    comm_unit = {}
    comm_rows = []
    for lt, evs in ev_all.items():
        sync = asyn = 0.0
        by = {}
        for e in evs:
            tcm, a = comm_time(e, hw)
            if a.get("async_"):
                asyn += tcm
            else:
                sync += tcm
            by[a["prim"]] = by.get(a["prim"], 0.0) + tcm
            if ops_detail:
                comm_rows.append(dict(layer=lt, prim=a["prim"], name=a["name"], in_B=a["in_B"], out_B=a["out_B"],
                                      hops=a["hops"], t=tcm, async_=bool(a.get("async_"))))
        comm_unit[lt] = dict(sync=sync, asyn=asyn, by=by, n=len(evs))
    return dict(unit=unit, comm=comm_unit, fam=fam, rows=rows, comm_rows=comm_rows, hw=hw,
                flops=tot_flops, bytes=tot_bytes)


LAYER_COUNTS = {"embed": 1, "attnres": None, "kda": 69, "dsa": 24, "moe": 92, "dense": 1, "final": 1, "drafter": 1}


def layer_list():
    """0-based layers: (idx, attention type, mlp type)."""
    dsa = set(range(3, 92, 4)) | {92}
    out = []
    for l in range(93):
        out.append((l, "dsa" if l in dsa else "kda", "dense" if l == 0 else "moe"))
    return out


LAYERS = layer_list()
assert sum(1 for x in LAYERS if x[1] == "dsa") == 24


def layer_time(r, att, mlp, o1=False):
    """Time of one transformer layer from unit results r (per instance)."""
    u, cm, k = r["unit"], r["comm"], r["hw"].kernel_s
    comp = u["attnres"]["comp"] / 1.0 + u[att]["comp"] + u[mlp]["comp"]
    # attnres ops are per layer (count NL): unit comp is one instance (pre + post)
    kern = (u["attnres"]["kernels"] + u[att]["kernels"] + u[mlp]["kernels"] + cm[att]["n"] + cm[mlp]["n"]) * k
    sync = cm[att]["sync"] + cm[mlp]["sync"]
    asyn = cm[att]["asyn"] + cm[mlp]["asyn"]
    if o1:
        return max(comp, sync + asyn) + kern, comp, max(0.0, max(comp, sync + asyn) - comp), kern
    exp_async = max(0.0, asyn - u[att]["comp"])
    return comp + sync + exp_async + kern, comp, sync + exp_async, kern


def global_time(r, lt, o1=False):
    u, cm, k = r["unit"], r["comm"], r["hw"].kernel_s
    comp = u[lt]["comp"]
    c = cm[lt]["sync"] + cm[lt]["asyn"]
    kern = (u[lt]["kernels"] + cm[lt]["n"]) * k
    if o1:
        return max(comp, c) + kern, comp, max(0.0, max(comp, c) - comp), kern
    return comp + c + kern, comp, c, kern


def step(r, o1=False, reps=1):
    """Assemble the step from unit results. Returns totals + per-layer list + family/group breakdown (seconds)."""
    per_layer = []
    tot = comp = comm = kern = 0.0
    for (l, att, mlp) in LAYERS:
        t, c, x, kk = layer_time(r, att, mlp, o1)
        per_layer.append(t * reps)
        tot += t
        comp += c
        comm += x
        kern += kk
    glob = {}
    for lt in ("embed", "final", "drafter", "step"):
        if lt == "step":
            c = r["comm"]["step"]["sync"]
            glob[lt] = c * reps
            tot += c
            comm += c
            continue
        t, c, x, kk = global_time(r, lt, o1)
        glob[lt] = t * reps
        tot += t
        comp += c
        comm += x
        kern += kk
    # family breakdown (compute), comm by primitive scaled to the exposed comm
    fam = {f: v * reps for f, v in r["fam"].items()}
    fam["launch"] = kern * reps
    raw_comm = {}
    for lt, cu in r["comm"].items():
        n = 92 if lt == "moe" else 69 if lt == "kda" else 24 if lt == "dsa" else 1
        for p, v in cu["by"].items():
            raw_comm[p] = raw_comm.get(p, 0.0) + v * n
    raw_tot = sum(raw_comm.values())
    comm_by = {p: (v / raw_tot * comm * reps if raw_tot > 0 else 0.0) for p, v in raw_comm.items()}
    groups = {}
    for f, v in fam.items():
        g = Mdl.GROUP_OF.get(f, "other")
        groups[g] = groups.get(g, 0.0) + v
    groups["comm"] = comm * reps
    return dict(t=tot * reps, comp=comp * reps, comm=comm * reps, kern=kern * reps, per_layer=per_layer, glob=glob,
                fam=fam, comm_by=comm_by, comm_raw={p: v * reps for p, v in raw_comm.items()}, groups=groups)


def unit_summary(r):
    """Per layer-type unit times (one instance), compute and comm, for the per-layer table."""
    out = {}
    for lt in ("attnres", "kda", "dsa", "moe", "dense", "embed", "final", "drafter"):
        out[lt] = dict(comp=r["unit"][lt]["comp"], comm=r["comm"].get(lt, {}).get("sync", 0.0) + r["comm"].get(lt, {}).get("asyn", 0.0))
    return out


def resident_weights(t, phase, ck_name, moe_act=None):
    """Weight bytes resident per GPU (each parameter counted once)."""
    ck = Mdl.ckpt_cfg(ck_name, moe_act)
    env = Mdl.env_decode(1, 3, 1000) if phase == "decode" else Mdl.env_prefill(1, 1000)
    env = dict(env, T=1.0, B=1.0, nseq=1.0, R=1.0, Tc=1.0, Tq=1.0)
    C = Mdl.C
    tot = 0.0
    by = {}
    skip = set(COMM_OPS) | {"G.embed", "DS.embed", "DS.lm_head", "DS.markov", "MOE.fc1", "MOE.fc2", "MOE.dispatch",
                            "MOE.combine"}
    for op in Mdl.OPS:
        if not op.in_phase(phase) or op.id in skip:
            continue
        cnt = Mdl.count(op, env)
        full = Mdl.evaluate(op, env, ck)
        if full["w"] == 0:
            continue
        m = op_map(op.id, t, phase)
        b = full["w"] / m["w"] * cnt
        f = Mdl.FAMILY[op.id]
        by[f] = by.get(f, 0.0) + b
        tot += b
    N = t.N
    emb = C["V"] * C["H"] * 2 / N
    by["embed"] = by.get("embed", 0.0) + emb
    tot += emb
    if phase == "decode":       # Markov w1 [V,512] + w2 [512,V], vocab-sharded with the drafter
        mk = 2 * C["V"] * C["D_R"] * 2 / t.ND
        by["drafter"] = by.get("drafter", 0.0) + mk
        tot += mk
    eb = Mdl.expert_bytes(ck) * C["E"] * C["N_MOE"]
    if t.afd:
        ex = 0.0
    elif t.experts == "tp":
        ex = eb / N
    else:
        ex = eb / t.E
    ex /= max(t.S, 1)
    by["experts"] = ex
    tot += ex
    if t.S > 1:            # pipeline stage holds 1/S of the non-expert layers too
        nonexp = tot - ex
        tot = nonexp / t.S + ex
        by = {k: (v if k == "experts" else v / t.S) for k, v in by.items()}
    return tot, by
