"""gap512: perf-model op rows (per step, us) for the C512 decode (D3) and prefill (P2) bench cases.
usage: python3 model_rows.py <out.json>"""
import json
import sys

PM = "/Users/weizha/k31vr-dev/perf-model"
sys.path.insert(0, PM)
import decode as Dec  # noqa: E402
import prefill as Pre  # noqa: E402
from topologies import ALL  # noqa: E402

CK = "mxfp4_release"
LAYER_COUNT = None


def table(res_s, res_a):
    ops = {}
    cnt = {}
    for rs, ra in zip(res_s["ops"], res_a["ops"]):
        oid = rs[0]
        assert oid == ra[0]
        c = rs[2]
        cnt[rs[1]] = cnt.get(rs[1], 0)
        ops[oid] = dict(fam=rs[1], count=c, t_sol_us=rs[9] * c * 1e6 if isinstance(rs[9], float) else None,
                        t_ach_us=ra[10] * c * 1e6 if isinstance(ra[10], float) else None)
    return ops


def oprow(r):
    # res rows: dicts or lists? normalise via run.merge_ops ordering (id, fam, count, flops, w, act, kv, state, dtype, t)
    return r


def norm_ops(rows):
    out = []
    for r in rows:
        if isinstance(r, dict):
            out.append(r)
        else:
            out.append(dict(zip(["id", "fam", "count", "flops", "w_B", "act_B", "kv_B", "state_B", "dtype", "t_s",
                                 "bound"], r)))
    return out


def merge(a, b, lay_counts):
    A, B = norm_ops(a["ops"]), norm_ops(b["ops"])
    ops = {}
    for x, y in zip(A, B):
        assert x["id"] == y["id"]
        c = x.get("count", x.get("n"))
        ops[x["id"]] = dict(fam=x["fam"], count=c, t_sol_us=x.get("t_s", x.get("t")) * c * 1e6, t_ach_us=y.get("t_s", y.get("t")) * c * 1e6,
                            bound_ach=y.get("bound"))
    comm = {}
    for x, y in zip(a["comm_rows"], b["comm_rows"]):
        xs = x if isinstance(x, dict) else dict(zip(["layer", "prim", "name", "in_B", "out_B", "hops", "t_s"], x))
        ys = y if isinstance(y, dict) else dict(zip(["layer", "prim", "name", "in_B", "out_B", "hops", "t_s"], y))
        n = lay_counts.get(xs["layer"], 1)
        key = "COMM." + xs["prim"].split()[0] + "." + xs["layer"]
        e = comm.setdefault(key, dict(prim=xs["prim"], layer=xs["layer"], t_sol_us=0.0, t_ach_us=0.0, events=0))
        e["t_sol_us"] += xs.get("t_s", xs.get("t")) * n * 1e6
        e["t_ach_us"] += ys.get("t_s", ys.get("t")) * n * 1e6
        e["events"] += n
    return dict(t_sol_ms=a["t_ms"], t_ach_ms=b["t_ms"], ops=ops, comm=comm, fam_ach=b.get("fam"),
                fam_sol=a.get("fam"), comm_ach=b.get("comm_raw"))


LAY = {"kda": 69, "dsa": 24, "moe": 92, "embed": 1, "final": 1, "drafter": 1, "step": 1, "dense": 1}


def main():
    out = {"decode": {}, "prefill": {}}
    t = ALL["D3"]
    for B in (16, 32, 50, 64, 80, 92):
        r = B / 4.0
        for L in (130000, 135000, 190000):
            a = Dec.run_point(t, CK, "sol", r, 3, L=L, detail=True)
            b = Dec.run_point(t, CK, "achievable", r, 3, L=L, detail=True)
            out["decode"]["B%d_L%dK" % (B, L // 1000)] = dict(B=B, r=r, L=L, **merge(a, b, LAY))
    t = ALL["P2"]
    for name, F, L, freq in (("pf_q12k_L78k", 12288, 78336, 12288), ("pf_q2x6k_L78k", 12288, 78336, 6144),
                             ("pf_q8k_L78k", 8192, 78336, 8192), ("pf_q16k_L78k", 16384, 78336, 16384),
                             ("pf_q12k_fresh", 12288, 12288, 12288)):
        a = Pre.run_point(t, CK, "sol", F, L, f_req=freq, detail=True)
        b = Pre.run_point(t, CK, "achievable", F, L, f_req=freq, detail=True)
        out["prefill"][name] = dict(F=F, L=L, f_req=freq, o1=b["o1"], **merge(a, b, LAY))
    with open(sys.argv[1], "w") as f:
        json.dump(out, f, indent=1)
    for k, v in out["decode"].items():
        print(k, "sol %.2f ach %.2f ms" % (v["t_sol_ms"], v["t_ach_ms"]))
    for k, v in out["prefill"].items():
        print(k, "sol %.2f ach %.2f ms o1 %s" % (v["t_sol_ms"], v["t_ach_ms"], v["o1"]))


main()
