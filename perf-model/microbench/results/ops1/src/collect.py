"""Collect ops1 jsonl rows (pulled from Hecate) into rows.json + a markdown summary on stdout.
usage: collect.py <dir with result subdirs> <rows.json>"""
import collections
import glob
import json
import sys

rows, env, errors = {}, {}, {}
for f in sorted(glob.glob(sys.argv[1] + "/*/rows.g*.jsonl")):
    for line in open(f):
        r = json.loads(line)
        if r.get("kind") == "env":
            env[f.split("/")[-2]] = r
        elif r.get("kind") == "case":
            if "error" in r:
                errors[r["case"]] = r
            else:
                rows[r["case"]] = r  # later runs override
                errors.pop(r["case"], None)

out = []
for cid, r in sorted(rows.items()):
    m, ach = r["measured_us"], r["t_ach_us"]
    d = m / ach if (ach and m) else None
    notes = r.get("notes", "")
    if r.get("backends"):
        notes = f"backends {r['backends']} best={r.get('best')}; " + notes
    if r.get("parts"):
        notes = f"parts {r['parts']}; " + notes
    out.append({"case": cid, "topology": r.get("topologies"), "ckpts": r.get("ckpts"), "op_ids": r.get("ops"),
                "shape": r.get("shape"), "dtype": r.get("dtype"), "measured_us": m, "t_sol_us": r["t_sol_us"],
                "t_ach_us": ach, "derate": round(d, 3) if d else None, "method": r.get("method"),
                "kernel_debug": bool(d and d > 2.0), "notes": notes})
for cid, r in sorted(errors.items()):
    out.append({"case": cid, "shape": r.get("shape"), "measured_us": None, "error": r["error"]})
json.dump(out, open(sys.argv[2], "w"), indent=1)

fam = collections.defaultdict(list)
for o in out:
    if o.get("derate"):
        fam[o["case"].rsplit("-", 1)[0]].append(o)
print(f"{len(rows)} measured, {len(errors)} errors; envs: " + "; ".join(
    f"{k}: {v.get('gpu')} torch {v.get('torch')} fi {str(v.get('flashinfer'))[:30]}" for k, v in env.items()))
for f, os_ in fam.items():
    ds = sorted(o["derate"] for o in os_)
    print(f"{f}: n={len(ds)} derate median {ds[len(ds) // 2]:.2f} min {ds[0]:.2f} max {ds[-1]:.2f}, "
          f">2.0: {sum(d > 2 for d in ds)}")
