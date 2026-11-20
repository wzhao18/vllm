"""summarize.py <parsed.json> <rows.json> [tag]: per-case step + family tables (rhea1n D3-32 slice) -> rows.json + text.

Reads gap_parse.py output (per op-group measured busy / exposed vs model achievable) and fam.py's family grouping.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fam as F  # noqa: E402

MEAS_MULTINODE_MS = {32: 20.92, 64: 26.98, 96: 30.48}  # wave2 D3-32 EP32 step (8 nodes, agentx ctx, drafter incl.)


def main():
    d = json.load(open(sys.argv[1]))
    tag = sys.argv[3] if len(sys.argv) > 3 else ""
    rows = []
    for case, c in sorted(d.items(), key=lambda kv: kv[1]["B"]):
        B = c["B"]
        ach = (c["model_t_ach_us"] - c["model_not_in_bench_ach_us"]) / 1e3
        step = (c["step_median_us"] or 0) / 1e3
        print("== %s %s r%g: step %.2f ms vs model ach (bench scope) %.2f ms = x%.2f; gap %.2f ms; 8-node D3-32 %.2f ms"
              % (tag, case, B / 4, step, ach, step / ach if ach else 0, step - ach, MEAS_MULTINODE_MS.get(B, 0)))
        rows.append(dict(kind="step", tag=tag, case=case, B=B, r=B / 4, step_ms=step, model_ach_ms=ach,
                         model_sol_ms=c["model_t_sol_us"] / 1e3, derate=step / ach if ach else None,
                         multinode_d3x32_ms=MEAS_MULTINODE_MS.get(B)))
        for r in F.fam_table(c):
            print("  %-68s busy %6.2f expo %6.2f ach %6.2f  x%-6s gap %6.2f ms" % (
                r["family"], r["busy_us"] / 1e3, r["exposed_us"] / 1e3, r["t_ach_us"] / 1e3,
                "%.2f" % r["derate"] if r["derate"] else "-", r["exposed_gap_us"] / 1e3))
            rows.append(dict(kind="family", tag=tag, case=case, B=B, r=B / 4, family=r["family"],
                             busy_ms=r["busy_us"] / 1e3, exposed_ms=r["exposed_us"] / 1e3, ach_ms=r["t_ach_us"] / 1e3,
                             derate=r["derate"], gap_ms=r["exposed_gap_us"] / 1e3))
        for g in c["rows"]:
            top = sorted(g["kernels"].items(), key=lambda kv: -kv[1])[:4]
            rows.append(dict(kind="group", tag=tag, case=case, B=B, group=g["group"], busy_ms=g["measured_us"] / 1e3,
                             exposed_ms=g["exposed_us"] / 1e3, ach_ms=g["t_ach_us"] / 1e3, derate=g["derate"],
                             calls=g["calls"], top_kernels_us=dict(top)))
    old = json.load(open(sys.argv[2])) if os.path.exists(sys.argv[2]) else []
    old = [x for x in old if x.get("tag") != tag]
    json.dump(old + rows, open(sys.argv[2], "w"), indent=1)


if __name__ == "__main__":
    main()
