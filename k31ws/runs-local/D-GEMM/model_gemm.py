"""D-GEMM: perf-model achievable rows of the decode dense GEMM family on D3-32 (nvfp4_modelopt and
mxfp4_release) at r = 1, 8, 16, 24, 32. Read-only use of the perf model.
usage: python3 model_gemm.py [out.json]"""
import json
import sys

PM = "/Users/weizha/k31vr-dev/perf-model"
sys.path.insert(0, PM)
import decode as Dec  # noqa: E402
from topologies import ALL  # noqa: E402

FAMS = ("kda_proj", "mla_proj", "router", "latent", "shared")
COLS = ["id", "fam", "count", "flops", "w_B", "act_B", "kv_B", "state_B", "dtype", "t_s", "bound"]
topo = ALL["D3-32"] if isinstance(ALL, dict) else [t for t in ALL if getattr(t, "id", t) == "D3-32"][0]
out = {}
for ck in ("nvfp4_modelopt", "mxfp4_release"):
    for r in (1, 8, 16, 24, 32):
        b = Dec.run_point(topo, ck, "achievable", r, 3, detail=True)
        rows = []
        tot = 0.0
        for x in b["ops"]:
            x = x if isinstance(x, dict) else dict(zip(COLS, x))
            if x["fam"] not in FAMS:
                continue
            t = x.get("t_s", x.get("t"))
            c = x.get("count", x.get("n"))
            rows.append(dict(id=x["id"], fam=x["fam"], count=c, w_B=x.get("w_B", x.get("w")), act_B=x.get("act_B", x.get("a")), dtype=x.get("dtype", x.get("dt")),
                             t_ach_us=t * 1e6, step_ms=t * c * 1e3))
            tot += t * c * 1e3
        out["%s|r%d" % (ck, r)] = dict(step_ms=b["t_ms"], family_ach_ms=tot, rows=rows)
        print("%s r%-2d step %.3f ms  GEMM family ach %.3f ms" % (ck, r, b["t_ms"], tot))
        for x in rows:
            print("   %-18s %-6s x%-3d w=%7.2fMB ach %6.2f us  %.3f ms" % (x["id"], x["dtype"], x["count"], x["w_B"] / 1e6,
                                                                       x["t_ach_us"], x["step_ms"]))
if len(sys.argv) > 1:
    json.dump(out, open(sys.argv[1], "w"), indent=1)
