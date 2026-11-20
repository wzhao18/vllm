"""Render k31vr/topology-tables.html: prefill + decode topology tables (model achievable vs measured) for both
checkpoints, from out/calibrated.json (model) and the calibrated 'measured' tags (silicon step benches)."""
import html
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "k31vr", "topology-tables.html")
CAL = json.load(open(os.path.join(HERE, "out", "calibrated.json")))
DER = json.load(open(os.path.join(HERE, "inputs", "derating.json")))["by_topology"]


def measured_prefill(k, F):
    """Exact measured tok/s/GPU of the MXFP8-activation L78K step at chunk F (None if not measured)."""
    for p in DER.get(k, {}).get("points", []):
        if p.get("moe_quant") == "mxfp4_mxfp8" and "L78k" in p["case"].replace(" ", "") and \
                ("pf%dk" % (F // 1024)) in p["case"] and "fresh" not in p["case"]:
            return p["measured_tok_s_gpu"]
    return None

NAMES = {
    # prefill
    "P1": "DCP8, today's production (TP8 + DCP8 with ShardQ, experts EP8 token-sharded)",
    "P2": "Hybrid DCP4 × DP4, EP16 (sequence-parallel TP4 = DCP4, ShardQ)",
    "P2b": "Hybrid DCP8 × DP2, EP16 (sequence-parallel TP8 = DCP8, ShardQ)",
    "P3-16": "DEP16 (attention and KDA data-parallel per GPU, experts EP16)",
    "P3-32": "DEP32 (attention and KDA data-parallel per GPU, experts EP32)",
    "P4-4": "Attention-DP with TP4 linears (PR #20), 4 DP groups, EP16",
    "P4-2": "Attention-DP with TP2 linears (PR #20), 8 DP groups, EP16",
    "P5-4": "TP4 with replicated DSA, EP16",
    "P5-8": "TP8 with replicated DSA, experts EP8 token-sharded",
    "P6-16": "Attention-DP with TP2 linears + expert-dispatch dedup, EP16",
    "P6-4-16": "Attention-DP with TP4 linears + expert-dispatch dedup, EP16",
    "P6-32": "Attention-DP with TP2 linears + expert-dispatch dedup, EP32",
    "P9": "DEP16 + DSA KV in DCP8 layout (ShardQ gather on DSA layers)",
    "P9-16": "DEP16 + DSA KV in DCP16 layout (ShardQ gather on DSA layers)",
    "P9-32": "DEP32 + DSA KV in DCP16 layout (ShardQ gather on DSA layers)",
    # decode
    "D1": "TP16 + DCP16, MoE TP16 (today's low-latency decode)",
    "D2": "TP8 + DCP8, experts EP8 token-sharded",
    "D3": "Hybrid DCP4 × DP4, EP16 (sequence-parallel TP4 = DCP4)",
    "D3-32": "Hybrid DCP4 × DP8, EP32 (sequence-parallel TP4 = DCP4)",
    "D4-16": "DEP16 (everything data-parallel per GPU, experts EP16)",
    "D4-32": "DEP32 (everything data-parallel per GPU, experts EP32)",
    "D5": "TP4 + DSA data-parallel (owner-local), replicated dense, EP16",
    "D6": "TP4 with replicated DSA, EP16",
    "D7-2-16": "Attention-DP with TP2 linears (KDA, router, latent, shared TP2), EP16",
    "D7-4-16": "Attention-DP with TP4 linears (KDA, router, latent, shared TP4), EP16",
    "D7-8-16": "Attention-DP with TP8 linears (KDA, router, latent, shared TP8), EP16",
    "D7-2-32": "Attention-DP with TP2 linears (KDA, router, latent, shared TP2), EP32",
    "D7-4-32": "Attention-DP with TP4 linears (KDA, router, latent, shared TP4), EP32",
    "D7-8-32": "Attention-DP with TP8 linears (KDA, router, latent, shared TP8), EP32",
    "D8-16": "TP16 + DCP16 attention, TP16 dense linears, experts EP16",
    "D8-32": "TP16 + DCP16 attention, TP16 dense linears, experts EP32",
    "D9-16": "DEP16 + DSA in DCP16 layout on DSA layers",
    "D9-32": "DEP32 + DSA in DCP16 layout on DSA layers",
}
CKPTS = [("mxfp4_release", "MXFP4 checkpoint",
          "Routed experts MXFP4 weights with MXFP8 activations (W4A8). Shared experts, KDA projections, MLA, router, "
          "latent and lm_head in BF16. FP8 MLA KV, MXFP4 indexer KV, FP32 KDA state."),
         ("nvfp4_modelopt", "NVFP4 checkpoint",
          "Routed experts NVFP4 weights and activations (W4A4). Shared experts and KDA in/out projections MXFP8 W8A8. "
          "MLA, router, latent and lm_head in BF16. Same KV and state precision as the MXFP4 checkpoint.")]
R_COLS = [1, 4, 8, 16, 24]


def fmt(x):
    return "" if x is None else f"{x:,.0f}"


def prefill_rows(ck):
    rows = []
    for k, v in CAL["prefill"][ck].items():
        cal = v["calibrated"]
        ach = cal["tok_s_gpu"] * cal["t_ms"] / cal["ach_shipped_t_ms"] if cal.get("ach_shipped_t_ms") else None
        meas = measured_prefill(k, cal["F"]) if (cal.get("tag") == "measured" and ck == "mxfp4_release") else None
        note = []
        if cal.get("F") and cal["F"] != 16384:
            note.append(f"{cal['F'] // 1024}K chunk")
        if cal.get("dedup_removed"):
            note.append("without dedup")
        rows.append(dict(key=k, name=NAMES.get(k, k), gpus=v["gpus"], ach=ach, meas=meas, note=", ".join(note)))
    rows.sort(key=lambda r: -(r["ach"] or 0))
    return rows


def decode_rows(ck):
    rows = []
    for k, v in CAL["decode"][ck].items():
        pts = {p["r"]: p for p in v["points"]}
        ach, meas = {}, {}
        for r in R_COLS:
            p = pts.get(r)
            a = (p or {}).get("achievable")
            ach[r] = a["tps_gpu"] if a else None
            c = (p or {}).get("calibrated")
            meas[r] = c["tps_gpu"] if (c and c.get("tag") == "measured" and ck == "mxfp4_release"
                                       and v.get("rmax_cal_method") == "measured" and r <= v["rmax"]["calibrated"]) else None
        mcap = v["rmax"]["calibrated"] if (v.get("rmax_cal_method") == "measured" and ck == "mxfp4_release") else None
        rows.append(dict(key=k, name=NAMES.get(k, k), gpus=v["gpus"], ach=ach, meas=meas,
                         cap_a=v["rmax"]["achievable"], cap_m=mcap))
    rows.sort(key=lambda r: -(r["ach"][8] or 0))
    return rows


def cell(x, best, cls="num"):
    s = fmt(x)
    if x is not None and best is not None and abs(x - best) < 0.5:
        s = f"<strong>{s}</strong>"
    return f'<td class="{cls}">{s}</td>'


def prefill_table(ck):
    rows = prefill_rows(ck)
    ba = max((r["ach"] for r in rows if r["ach"]), default=None)
    bm = max((r["meas"] for r in rows if r["meas"]), default=None)
    out = ['<div class="scroll"><table><thead><tr><th class="topo">Topology</th><th>GPUs</th>'
           '<th>Model, achievable</th><th>Measured</th></tr></thead><tbody>']
    for r in rows:
        note = f'<span class="note">{html.escape(r["note"])}</span>' if r["note"] else ""
        out.append(f'<tr><td class="topo">{html.escape(r["name"])}{note}</td><td class="num">{r["gpus"]}</td>'
                   + cell(r["ach"], ba) + cell(r["meas"], bm) + "</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def decode_table(ck):
    rows = decode_rows(ck)
    ba = {r: max((x["ach"][r] for x in rows if x["ach"][r]), default=None) for r in R_COLS}
    bm = {r: max((x["meas"][r] for x in rows if x["meas"][r]), default=None) for r in R_COLS}
    head1 = "".join(f'<th colspan="2" class="grp">r = {r}</th>' for r in R_COLS)
    head2 = "".join('<th class="sub">Model</th><th class="sub">Meas.</th>' for _ in R_COLS)
    out = ['<div class="scroll"><table class="dec"><thead>',
           f'<tr><th rowspan="2" class="topo">Topology</th><th rowspan="2">GPUs</th>'
           f'<th colspan="2" class="grp">Max requests / GPU</th>{head1}</tr>',
           f'<tr><th class="sub">Model</th><th class="sub">Meas.</th>{head2}</tr></thead><tbody>']
    for x in rows:
        cm = "" if x["cap_m"] is None else "%.1f" % x["cap_m"]
        tds = [f'<td class="topo">{html.escape(x["name"])}</td><td class="num">{x["gpus"]}</td>',
               f'<td class="num">{x["cap_a"]:.0f}</td><td class="num">{cm}</td>']
        for r in R_COLS:
            a = x["ach"][r]
            tds.append(cell(a, ba[r]) if a is not None else '<td class="num dash">—</td>')
            tds.append(cell(x["meas"][r], bm[r], "num meas"))
        out.append("<tr>" + "".join(tds) + "</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


CSS = """
:root{--bg:#f6f7f9;--panel:#ffffff;--fg:#1b2230;--muted:#5d6878;--line:#dfe3ea;--accent:#2f6f5e;--meas:#eef4f1;
--sans:'Inter',system-ui,-apple-system,'Segoe UI',sans-serif;--head:'Space Grotesk','Inter',system-ui,sans-serif;
--mono:'JetBrains Mono',ui-monospace,Menlo,monospace}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#11151b;--panel:#171c24;--fg:#e4e8ee;
--muted:#9aa5b4;--line:#2a313c;--accent:#7cc4ad;--meas:#18241f;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#11151b;--panel:#171c24;--fg:#e4e8ee;--muted:#9aa5b4;--line:#2a313c;--accent:#7cc4ad;
--meas:#18241f;color-scheme:dark}
body{background:var(--bg);color:var(--fg);font-family:var(--sans);line-height:1.5;
padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));padding-block:28px 48px}
main{max-width:1280px;margin:0 auto;display:grid;gap:36px}
h1,h2,h3{font-family:var(--head);text-wrap:balance;margin:0}
h1{font-size:1.75rem}h2{font-size:1.3rem;border-bottom:2px solid var(--accent);padding-bottom:6px}
h3{font-size:1.05rem;color:var(--muted);font-weight:600}
p{margin:0;max-width:75ch;color:var(--muted)}
section{display:grid;gap:14px;min-width:0}
.ck{display:grid;gap:10px;min-width:0}
.scroll{overflow-x:auto;background:var(--panel);border:1px solid var(--line);border-radius:6px}
table{border-collapse:collapse;width:100%;font-size:0.86rem;font-variant-numeric:tabular-nums}
th,td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th{font-family:var(--head);font-weight:600;font-size:0.78rem;letter-spacing:0.03em;color:var(--muted);vertical-align:bottom}
th.grp{text-align:center;border-left:1px solid var(--line)}
th.sub{font-size:0.72rem}
td.topo,th.topo{text-align:left;white-space:normal;min-width:300px;max-width:440px}
td.meas{background:var(--meas)}
td.dash{color:var(--muted)}
strong{color:var(--accent);font-weight:700}
.note{display:block;font-size:0.75rem;color:var(--muted)}
tbody tr:last-child td{border-bottom:none}
.legend{font-size:0.82rem;color:var(--muted);display:grid;gap:4px}
code{font-family:var(--mono);font-size:0.85em}
"""


def page():
    parts = ['<title>K3.1 VR200 Topology Tables</title>',
             '<meta name="claude:viewport-fit-sides" content="cover">',
             '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&'
             'family=Space+Grotesk:wght@500;600&family=JetBrains+Mono&display=swap">',
             f"<style>{CSS}</style><main>",
             "<header style=\"display:grid;gap:8px\"><h1>Kimi K3.1 on VR200: prefill and decode topologies</h1>"
             "<p>Speed-of-light model (achievable column: physics with measured hardware ceilings, a strict upper bound) "
             "against silicon step benchmarks. Instances of at most 32 GPUs. Bold marks the best value in each column.</p>"
             "</header>"]
    parts.append('<section><h2>Prefill</h2><div class="legend"><span>Output: throughput in tok/s per GPU for a 16K-token '
                 'fresh chunk on 78K cached context (the measured shape). Sorted by model achievable.</span>'
                 '<span>"Without dedup": the model has dispatch dedup in its design, removed here because MegaMoE does not '
                 'implement it.</span></div>')
    for ck, title, desc in CKPTS:
        parts.append(f'<div class="ck"><h3>{title}</h3><p>{desc}</p>{prefill_table(ck)}</div>')
    parts.append("</section>")
    parts.append('<section><h2>Decode</h2><div class="legend"><span>Output tokens per second per GPU at r requests per '
                 'GPU (k = 3 speculative decoding, 135K context). Sorted by model achievable at r = 8.</span>'
                 '<span>"—": over the model\'s KV capacity. Blank measured cell: not measured, or over the measured KV '
                 'capacity.</span></div>')
    for ck, title, desc in CKPTS:
        parts.append(f'<div class="ck"><h3>{title}</h3><p>{desc}</p>{decode_table(ck)}</div>')
    parts.append("</section>")
    parts.append('<section><p style="font-size:0.82rem">No step benchmark has run on the NVFP4 checkpoint yet, so its '
                 'measured columns are empty. Source: <code>perf-model/out/calibrated.json</code>, regenerated by '
                 '<code>perf-model/render_tables.py</code>.</p></section></main>')
    return "\n".join(parts)


if __name__ == "__main__":
    open(OUT, "w").write(page())
    print(OUT)
