"""Inline out/*.json (+ calibration and microbench evidence) into sol_template.html -> k31vr/sol.html (self-contained)."""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
MB = os.path.join(HERE, "microbench")


def _dst():
    """k31vr/sol.html. solmodel/ may be a symlink (k31vr/solmodel -> perf-model), so resolve via the repo layout."""
    up = os.path.dirname(HERE)
    if os.path.basename(up) == "k31vr":
        return os.path.join(up, "sol.html")
    if os.path.isdir(os.path.join(up, "k31vr")):
        return os.path.join(up, "k31vr", "sol.html")
    return os.path.join(up, "sol.html")


DST = os.environ.get("SOL_HTML") or _dst()

# D1 decode role HBM itemisation (GiB per GPU, release TP16=DCP16, k3, gpu-memory-utilization 0.90; decode lane
# capacity finding + calibrate.HBM). Weights include the MoE-TP16 intermediate padding (I_E/16 < 256 padded to 256).
D1_HBM = {
    "device_GiB": 278.5,
    "items": [
        ["weights (model resident)", 103.1, "w"],
        ["MoE-TP16 intermediate padding (in weights)", 28.1, "pad"],
        ["non-torch (CUDA ctx, NCCL/NVSHMEM, symm buffers)", 24.0, "nt"],
        ["gpu-memory-utilization headroom (0.90)", 21.8, "head"],
        ["activations / graphs", 7.3, "act"],
        ["used before startup profiling", 6.1, "start"],
        ["KV pool", 88.3, "kv"],
    ],
    "weights_total_GiB": 131.2,
}

SCOPE_NOTE = ("Scope (2026-11-18): at most 32 GPUs per instance; silicon measurements reach EP32 at most (D3-32). "
              "EP64/72, PP4 and AFD stay modelled but are not ranked. Calibrated and measured columns are "
              "mxfp4_release (MXFP8 MoE activations, BF16 linears, FP8 MLA KV, MXFP4 indexer KV, FP32 KDA state, exact "
              "top-k); nvfp4_modelopt calibrated is an unmeasured transfer.")

BLOCKERS = [
    ["P3-32 (DEP32 prefill)", "Refused at load on the release tree: the kimi_k3_mega_moe knob table has records only "
     "for world_size 4/8/16 (KnobLookupError at megamoe_rubin/config.py resolve_knobs). Needs a tuned EP32 knob record; "
     "until then P3-32 / P9-32 / P6-32 calibrated values are projected from P3-16 / P4-2."],
    ["D3-32 r >= 48 (SP decode)", "Specialized decode refuses sequence parallelism above 512 verify rows "
     "(max_num_seqs x q): r48 (B192, 768 rows) raises ValueError. r32 (B128) OOMs in the bench graph capture "
     "(bench artifact). The 32-64 band therefore has no calibrated or measured winner."],
    ["D5, D7-x, D8-x not implemented", "The model's mid/high-band winners (D7-8-32, D7-4-32, D8-32) and D5 have no "
     "code path; their calibrated values are analog transfers from D1 (TP front) or D3 / D3-32 (SP front), tagged "
     "projected."],
    ["In-step MegaMoE is the vendored kernel", "All in-step rows use vLLM-vendored megamoe_rubin; FlashInfer "
     "6e438852b was measured standalone only (projected -1.6..-2.5 ms per D3 step, about -21 ms per P1 step)."],
]


def _load(path):
    with open(path) as f:
        return json.load(f)


def kernel_debug():
    """Ranked KERNEL-DEBUG table from microbench/RESULTS.md section 3."""
    txt = open(os.path.join(MB, "RESULTS.md")).read()
    sec = txt.split("## 3. KERNEL-DEBUG", 1)[1].split("**Model fixes", 1)[0]
    out = []
    for line in sec.splitlines():
        if not line.startswith("| "):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 6 or not re.match(r"^(\d+|-)$", cells[0]):
            continue
        item = cells[1]
        m = re.match(r"^\*\*(.+?)\*\*\s*(.*)$", item)
        title, rest = (m.group(1), m.group(2)) if m else (item, "")
        clean = lambda s: s.replace("**", "")
        out.append({"n": cells[0], "title": clean(title), "detail": clean(rest), "derate": clean(cells[2]),
                    "excess": clean(cells[3]), "hyp": clean(cells[4]), "src": clean(cells[5])})
    return out


def op_derates():
    """Per-family measured vs model-achievable rows for every profiled topology (wave-1 model tables)."""
    cases = []

    def row(name, r):
        return {"name": name, "meas": r["measured_us"], "ach": r.get("t_ach_us") or 0.0, "d": r.get("derate")}

    dec = _load(os.path.join(MB, "results", "decode", "rows.json"))
    steps = {}
    for r in dec:
        if r["kind"] == "step" and r["k"] == 3:
            steps.setdefault((r["topology"], r["r_per_gpu"]), r)
    groups = {}
    for r in dec:
        if r["kind"] == "step":
            continue
        key, name = r["case"].split(": ", 1) if ": " in r["case"] else (r["case"], r["case"])
        groups.setdefault((r["topology"], key, r["r_per_gpu"]), []).append(row(name, r))
    for (topo, key, rr), rows in sorted(groups.items(), key=lambda x: (x[0][0], x[0][2])):
        st = steps.get((topo, rr))
        cases.append({"topo": topo, "phase": "decode", "label": "%s decode r=%g, %s" % (topo, rr, key.split()[0]),
                      "src": "microbench/results/decode " + key, "rows": rows,
                      "step": {"meas": st["measured_us"], "ach": st["t_ach_us"]} if st else None})

    gap = _load(os.path.join(MB, "results", "gap512", "rows.json"))
    bycase = {}
    for r in gap:
        if r["kind"] == "decode" and (r["group"].startswith("FAMILY") or r["group"] == "STEP"):
            bycase.setdefault(r["case"], []).append(r)
    for case, rs in bycase.items():
        step = [r for r in rs if r["group"] == "STEP"]
        B = int(re.search(r"_B(\d+)_", case).group(1))
        fam = [row(r["group"].replace("FAMILY: ", ""), r) for r in rs if r["group"] != "STEP"]
        cases.append({"topo": "D3", "phase": "decode", "label": "D3 decode r=%g, C512 profile (B%d)" % (B / 4.0, B),
                      "src": "microbench/results/gap512 " + case, "rows": fam,
                      "step": {"meas": step[0]["measured_us"], "ach": step[0]["t_ach_us"]} if step else None})

    pre = _load(os.path.join(MB, "results", "prefill", "rows.json"))
    full = {r["topology"]: r for r in pre if r.get("row") == "full step" and "release" not in r["case"]
            and "fresh" not in r["case"] and "pf32k" not in r["case"] and "no profile" not in r["case"]}
    pcases = {}
    for r in pre:
        if "(step median" in r["case"]:
            pcases.setdefault((r["topology"], r["case"]), []).append(row(r["row"], r))
    for (topo, case), rows in pcases.items():
        st = full.get(topo)
        cases.append({"topo": topo, "phase": "prefill", "label": "%s prefill %s" % (topo, case.split(" [")[0].split(" ", 1)[1]),
                      "src": "microbench/results/prefill " + case, "rows": rows,
                      "step": {"meas": st["measured_us"], "ach": st["t_ach_us"]} if st else None})
    return cases


def main():
    data = {}
    for name in ("meta", "prefill", "decode", "optables", "calibrated"):
        data[name] = _load(os.path.join(HERE, "out", name + ".json"))
    der = _load(os.path.join(HERE, "inputs", "derating.json"))
    data["derating"] = {k: der[k] for k in ("generated", "scope", "definition", "versions", "by_family",
                                            "by_topology", "review_caveats")}
    data["opd"] = op_derates()
    data["kdbg"] = kernel_debug()
    data["static"] = {"d1_hbm": D1_HBM, "scope_note": SCOPE_NOTE, "blockers": BLOCKERS,
                      "active_lane": "KERNEL-DEBUG #1: SP-path BF16 dense GEMMs (in progress)"}
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    with open(os.path.join(HERE, "sol_template.html")) as f:
        tpl = f.read()
    assert "/*__DATA__*/" in tpl
    html = tpl.replace("/*__DATA__*/", "var DATA = " + blob + ";", 1)
    with open(DST, "w") as f:
        f.write(html)
    print("wrote %s (%.1f MB)" % (DST, os.path.getsize(DST) / 1e6))


if __name__ == "__main__":
    main()
