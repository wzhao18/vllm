"""Model op tables (optables.json format) at the exact measured prefill shapes.
usage: cd perf-model && python3 gen_tables.py <out.json>"""
import json
import sys

sys.path.insert(0, ".")
import prefill as Pre  # noqa: E402
import run as Run  # noqa: E402
from topologies import PREFILL  # noqa: E402

CASES = [  # (key, topo id, F, f_req)
    ("P1 F16384", "P1", 16384, 5456), ("P2 F16384", "P2", 16384, 5456), ("P2 F32768", "P2", 32768, 5456),
    ("P3-16 F16384", "P3-16", 16384, 5456), ("P3-16 F8192", "P3-16", 8192, 5456),
    ("P4-4 F16384 f4096", "P4-4", 16384, 4096), ("P4-2 F16384 f4096", "P4-2", 16384, 4096),
]
T = {t.id: t for t in PREFILL}
opt = json.load(open("out/optables.json"))
out = {"cols": opt["cols"], "comm_cols": opt["comm_cols"], "tables": {}}
for key, tid, F, f in CASES:
    t = T[tid]
    a = Pre.run_point(t, "mxfp4_release", "sol", F, Pre.L_REF, f_req=f, detail=True)
    b = Pre.run_point(t, "mxfp4_release", "achievable", F, Pre.L_REF, f_req=f, detail=True)
    out["tables"][key] = dict(topo=tid, F=F, f_req=f, L=Pre.L_REF, ops=Run.merge_ops(a["ops"], b["ops"]),
                              comm=Run.merge_comm(a["comm_rows"], b["comm_rows"]), t_sol_ms=a["t_ms"], t_ach_ms=b["t_ms"],
                              tok_s_gpu_ach=b["tok_s_gpu"], tok_s_gpu_sol=a["tok_s_gpu"],
                              per_layer_ach=b["per_layer_ms"], per_layer_sol=a["per_layer_ms"])
    print(key, "t_ach %.1f ms t_sol %.1f ms ach %.0f tok/s/GPU" % (b["t_ms"], a["t_ms"], b["tok_s_gpu"]))
json.dump(out, open(sys.argv[1], "w"))
