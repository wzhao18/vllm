"""mk_model_rows.py (Mac, read-only use of perf-model): model rows for the k31ws 1-node slices, 24-layer view.

  python3 mk_model_rows.py            -> model_rows_L24_<ckpt>.json (decode D3-32) and pf_model_rows_L24.json (prefill P3-16)

Decode: D3-32 per DP rank B = 32/64/96 (r8/r16/r24), L = 135K, k=3, sol + achievable per op and comm row
(gap_parse.py format, same as perf-model/microbench/rhea1n/model_rows.py), for ckpt mxfp4_release and nvfp4_modelopt.
Prefill: P3-16 F16K / F8K at L_REF (78K), f_req 5456, per op (prefill.run_point detail), both ckpts.
Both are rescaled to the bench's 24-layer view (18 KDA + 6 DSA + 23 MoE + 1 dense): KDA ops x 18/69, DSA x 6/24,
MOE x 23/92, per-layer AttnRes x 24/93, comm by layer kind; whole-step terms unchanged. The 93-layer totals are kept
(t_ach_ms_93) for the full-model view.
"""
import json
import os
import sys

PM = "/Users/weizha/k31vr-dev/perf-model"
sys.path.insert(0, PM)
os.chdir(PM)
import decode as Dec  # noqa: E402
import prefill as Pre  # noqa: E402
from topologies import ALL, PREFILL  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SCALE = {"KDA": 18 / 69, "DSA": 6 / 24, "MOE": 23 / 92, "L": 24 / 93}
COMM_SCALE = {"kda": 18 / 69, "dsa": 6 / 24, "moe": 23 / 92}
LAY = {"kda": 69, "dsa": 24, "moe": 92, "embed": 1, "final": 1, "drafter": 1, "step": 1, "dense": 1}
CKPTS = ("mxfp4_release", "nvfp4_modelopt")


def norm_op(r):
    if isinstance(r, dict):
        return r
    return dict(zip(["id", "fam", "count", "flops", "w_B", "act_B", "kv_B", "state_B", "dtype", "t_s", "bound"], r))


def norm_comm(x):
    return x if isinstance(x, dict) else dict(zip(["layer", "prim", "name", "in_B", "out_B", "hops", "t_s"], x))


def merge(a, b):
    ops = {}
    for x, y in zip(map(norm_op, a["ops"]), map(norm_op, b["ops"])):
        assert x["id"] == y["id"]
        c = x.get("count", x.get("n"))
        tx, ty = x.get("t_s", x.get("t")), y.get("t_s", y.get("t"))
        f = SCALE.get(x["id"].split(".")[0], 1.0)
        ops[x["id"]] = dict(fam=x["fam"], count=c * f, t_sol_us=tx * c * f * 1e6, t_ach_us=ty * c * f * 1e6,
                            t_ach_us_93=ty * c * 1e6, bound_ach=y.get("bound"), dtype=y.get("dt", y.get("dtype")))
    comm = {}
    for x, y in zip(map(norm_comm, a["comm_rows"]), map(norm_comm, b["comm_rows"])):
        n = LAY.get(x["layer"], 1) * COMM_SCALE.get(x["layer"], 1.0)
        key = "COMM." + x["prim"].split()[0] + "." + x["layer"]
        e = comm.setdefault(key, dict(prim=x["prim"], layer=x["layer"], t_sol_us=0.0, t_ach_us=0.0, events=0))
        e["t_sol_us"] += x.get("t_s", x.get("t")) * n * 1e6
        e["t_ach_us"] += y.get("t_s", y.get("t")) * n * 1e6
        e["events"] += n
    sum_ach = (sum(o["t_ach_us"] for o in ops.values()) + sum(c["t_ach_us"] for c in comm.values())) / 1e3
    sum_sol = (sum(o["t_sol_us"] for o in ops.values()) + sum(c["t_sol_us"] for c in comm.values())) / 1e3
    return dict(t_sol_ms=sum_sol, t_ach_ms=sum_ach, t_sol_ms_93=a["t_ms"], t_ach_ms_93=b["t_ms"], ops=ops, comm=comm,
                fam_ach=b.get("fam"), fam_sol=a.get("fam"))


def main():
    t = ALL["D3-32"]
    for ck in CKPTS:
        out = {"decode": {}, "_ckpt": ck, "_view": "L24 (18 KDA + 6 DSA + 23 MoE + 1 dense)"}
        for B in (32, 64, 96):
            r = B / 4.0
            a = Dec.run_point(t, ck, "sol", r, 3, L=135000, detail=True)
            b = Dec.run_point(t, ck, "achievable", r, 3, L=135000, detail=True)
            out["decode"]["B%d_L135K" % B] = dict(B=B, r=r, L=135000, **merge(a, b))
            m = out["decode"]["B%d_L135K" % B]
            print(ck, "decode B%d r%g: 93L ach %.2f ms; L24 additive ach %.2f sol %.2f ms" % (
                B, r, m["t_ach_ms_93"], m["t_ach_ms"], m["t_sol_ms"]))
        name = "model_rows_L24.json" if ck == "mxfp4_release" else "model_rows_L24_%s.json" % ck
        json.dump(out, open(os.path.join(HERE, name), "w"), indent=1)
    T = {x.id: x for x in PREFILL}
    pf = {"prefill": {}, "_view": "L24"}
    for ck in CKPTS:
        for F in (16384, 8192):
            a = Pre.run_point(T["P3-16"], ck, "sol", F, Pre.L_REF, f_req=5456, detail=True)
            b = Pre.run_point(T["P3-16"], ck, "achievable", F, Pre.L_REF, f_req=5456, detail=True)
            k = "%s|F%d" % (ck, F)
            pf["prefill"][k] = dict(ckpt=ck, F=F, L=Pre.L_REF, tok_s_gpu_ach_93=b["tok_s_gpu"], **merge(a, b))
            m = pf["prefill"][k]
            print(ck, "prefill P3-16 F%d: 93L ach %.1f ms (%.0f tok/s/GPU); L24 additive ach %.1f ms" % (
                F, m["t_ach_ms_93"], b["tok_s_gpu"], m["t_ach_ms"]))
    json.dump(pf, open(os.path.join(HERE, "pf_model_rows_L24.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
