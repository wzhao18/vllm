"""rescale_rows.py <model_rows.json> <out.json>: D3-32 model rows (93 layers) -> the 24-layer bench view.

The Rhea 1-node bench runs a 24-layer view of K3.1 (18 KDA + 6 DSA layers, 23 MoE + 1 dense, AttnRes over 24 layers),
so per-layer op time is compared on the same layer counts: KDA ops x 18/69, DSA x 6/24, MOE x 23/92, per-layer AttnRes
(L.*) x 24/93, per-layer comm by its layer kind. Whole-step terms (embed, dense, final, drafter, step) are unchanged.
"""
import json
import sys

SCALE = {"KDA": 18 / 69, "DSA": 6 / 24, "MOE": 23 / 92, "L": 24 / 93}
COMM_SCALE = {"kda": 18 / 69, "dsa": 6 / 24, "moe": 23 / 92}


def main():
    d = json.load(open(sys.argv[1]))
    out = {"decode": {}}
    for key, row in d["decode"].items():
        r = json.loads(json.dumps(row))
        for oid, o in r["ops"].items():
            f = SCALE.get(oid.split(".")[0], 1.0)
            o["t_sol_us"] *= f
            o["t_ach_us"] *= f
            o["count"] = o["count"] * f
        for cid, c in r["comm"].items():
            f = COMM_SCALE.get(c["layer"], 1.0)
            c["t_sol_us"] *= f
            c["t_ach_us"] *= f
            c["events"] = c["events"] * f
        r["t_ach_ms_93"] = row["t_ach_ms"]
        r["t_sol_ms_93"] = row["t_sol_ms"]
        r["t_ach_ms"] = (sum(o["t_ach_us"] for o in r["ops"].values())
                         + sum(c["t_ach_us"] for c in r["comm"].values())) / 1e3
        r["t_sol_ms"] = (sum(o["t_sol_us"] for o in r["ops"].values())
                         + sum(c["t_sol_us"] for c in r["comm"].values())) / 1e3
        # 93-layer additive sum, for the ratio check against the model's step (overlap/launch terms differ)
        r["sum_ach_ms_93"] = (sum(o["t_ach_us"] for o in row["ops"].values())
                              + sum(c["t_ach_us"] for c in row["comm"].values())) / 1e3
        out["decode"][key] = r
        print(key, "93L step ach %.2f (additive %.2f) -> 24L additive ach %.2f ms"
              % (row["t_ach_ms"], r["sum_ach_ms_93"], r["t_ach_ms"]))
    json.dump(out, open(sys.argv[2], "w"), indent=1)


if __name__ == "__main__":
    main()
