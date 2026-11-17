#!/usr/bin/env python3
"""pareto_html.py: write the 4-way per-MW pareto charts into k31vr/index.html (between <!-- PARETO --> markers).

Series:
  - VR200 release-draft sweep (results/k31vr-release-draft-sweep-20261117/points.json), ongoing
  - VR200 frozen preview-draft sweep (results/k31vr-frozen-sweep-20261114/points.json), all measured points
  - GB300 dynamo-vLLM and GB300 dynamo-SGLang, latest InferenceX curves (k31vr/data/gb300_inferencex.json,
    from https://inferencex.semianalysis.com/api/v1/benchmarks?model=Kimi-K3.1-Pro)
Per MW = tok/s/GPU / all-in kW per GPU (VR200 3.3 kW, GB300 2.12 kW), shown as M tok/s per MW.
Each series draws its own pareto front as a line; every measured point is a marker with its label and a hover tooltip.
"""
import html, json, math, os, re

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
KW = {"vr": 3.3, "gb": 2.12}
PANELS = [("p90_intvty", "P90 interactivity (tok/s/user) = 1000 / P90 ITL", True),
          ("p90_e2e_norm_intvty", "P90 E2E-normalized interactivity (tok/s/user)", True),
          ("p90_e2el_s", "P90 end-to-end latency (s)", False),
          ("p90_ttft_s", "P90 TTFT (s)", False)]


def load_series():
    rel = json.load(open(f"{ROOT}/results/k31vr-release-draft-sweep-20261117/points.json"))["points"]
    frz = json.load(open(f"{ROOT}/results/k31vr-frozen-sweep-20261114/points.json"))["points"]
    gb = json.load(open(f"{ROOT}/k31vr/data/gb300_inferencex.json"))["series"]
    def vr(p, run_dir):
        return dict(label=p["label"], y=p["tput_per_gpu"] / KW["vr"] / 1000, tps=p["tput_per_gpu"],
                    tip=f"{p['label']} ({p['topology']}, {p['gpus']} GPUs), run {run_dir}/{p['run']}", **{k: p[k] for k, _, _ in PANELS})
    def g(p, fw, date):
        return dict(label=p["label"], y=p["tput_per_gpu"] / KW["gb"] / 1000, tps=p["tput_per_gpu"],
                    tip=f"GB300 {fw} {p['label']} ({p['topo']}), InferenceX id {p['id']}, {date}", **{k: p[k] for k, _, _ in PANELS})
    return [
        dict(key="rel", name="VR200 release draft (ongoing, 2026-11-17)", cls="s-rel", marker="circle", dash="",
             pts=[vr(p, "k31vr-release-draft-sweep-20261117/runs") for p in rel]),
        dict(key="frz", name="VR200 frozen sweep, preview draft (2026-11-14)", cls="s-frz", marker="square", dash="5 4",
             pts=[vr(p, "k31vr-frozen-sweep-20261114/runs") for p in frz]),
        dict(key="gbv", name=f"GB300 dynamo-vLLM (InferenceX {gb['dynamo-vllm']['curve_date']})", cls="s-gbv", marker="triangle",
             dash="2 3", pts=[g(p, "dynamo-vLLM", gb["dynamo-vllm"]["curve_date"]) for p in gb["dynamo-vllm"]["points"]]),
        dict(key="gbs", name=f"GB300 dynamo-SGLang (InferenceX {gb['dynamo-sglang']['curve_date']})", cls="s-gbs", marker="diamond",
             dash="8 3 2 3", pts=[g(p, "dynamo-SGLang", gb["dynamo-sglang"]["curve_date"]) for p in gb["dynamo-sglang"]["points"]]),
    ]


def front(pts, key, hb):
    out, best = [], -1.0
    for p in sorted(pts, key=lambda p: -p[key] if hb else p[key]):
        if p["y"] > best:
            out.append(p); best = p["y"]
    return sorted(out, key=lambda p: p[key])


def nice_ticks(lo, hi, n=5):
    span = hi - lo or 1
    step = 10 ** math.floor(math.log10(span / n))
    for m in (1, 2, 2.5, 5, 10):
        if span / (step * m) <= n:
            step *= m; break
    t, out = math.floor(lo / step) * step, []
    while True:
        out.append(round(t, 10))
        if t >= hi - 1e-9:
            return out
        t += step


def marker(kind, x, y, r=4.2):
    if kind == "circle":
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}"/>'
    if kind == "square":
        return f'<rect x="{x - r:.1f}" y="{y - r:.1f}" width="{2 * r}" height="{2 * r}"/>'
    if kind == "triangle":
        return f'<path d="M{x:.1f},{y - r - 1:.1f} L{x + r + 0.5:.1f},{y + r:.1f} L{x - r - 0.5:.1f},{y + r:.1f} Z"/>'
    return f'<path d="M{x:.1f},{y - r - 1:.1f} L{x + r + 1:.1f},{y:.1f} L{x:.1f},{y + r + 1:.1f} L{x - r - 1:.1f},{y:.1f} Z"/>'


def fmt(v):
    return f"{v:,.0f}" if v >= 100 else (f"{v:,.1f}" if v >= 10 else f"{v:,.2f}".rstrip("0").rstrip("."))


def panel(series, key, label, hb):
    W, H, L, R, T, B = 560, 380, 52, 16, 14, 46
    xs = [p[key] for s in series for p in s["pts"]]
    ys = [p["y"] for s in series for p in s["pts"]]
    xt, yt = nice_ticks(0, max(xs) * 1.04), nice_ticks(0, max(ys) * 1.06)
    x1, y1 = xt[-1], yt[-1]
    sx = lambda v: L + (W - L - R) * v / x1
    sy = lambda v: H - B - (H - T - B) * v / y1
    o = [f'<svg class="pchart" viewBox="0 0 {W} {H}" role="img" aria-label="Throughput per MW vs {html.escape(label)}">']
    for t in yt:
        o.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{sy(t):.1f}" y2="{sy(t):.1f}"/>'
                 f'<text class="tick" x="{L - 6}" y="{sy(t) + 4:.1f}" text-anchor="end">{fmt(t)}</text>')
    for t in xt:
        o.append(f'<line class="grid" x1="{sx(t):.1f}" x2="{sx(t):.1f}" y1="{T}" y2="{H - B}"/>'
                 f'<text class="tick" x="{sx(t):.1f}" y="{H - B + 16}" text-anchor="middle">{fmt(t)}</text>')
    o.append(f'<line class="axis" x1="{L}" x2="{W - R}" y1="{H - B}" y2="{H - B}"/><line class="axis" x1="{L}" x2="{L}" y1="{T}" y2="{H - B}"/>')
    o.append(f'<text class="axlab" x="{(L + W - R) / 2:.0f}" y="{H - 8}" text-anchor="middle">{html.escape(label)} — {"higher" if hb else "lower"} is better</text>')
    o.append(f'<text class="axlab" transform="translate(13,{(T + H - B) / 2:.0f}) rotate(-90)" text-anchor="middle">M tok/s per MW (input + output)</text>')
    placed = []
    for s in reversed(series):  # draw the release series last (on top)
        f = front(s["pts"], key, hb)
        if len(f) > 1:
            d = " ".join(f'{"M" if i == 0 else "L"}{sx(p[key]):.1f},{sy(p["y"]):.1f}' for i, p in enumerate(f))
            o.append(f'<path class="fr {s["cls"]}" d="{d}"' + (f' stroke-dasharray="{s["dash"]}"' if s["dash"] else "") + "/>")
        for p in s["pts"]:
            x, y = sx(p[key]), sy(p["y"])
            tip = (f'{p["tip"]}: {p["tps"]:,.0f} tok/s/GPU = {p["y"]:.2f} M tok/s/MW; P90 intvty {p["p90_intvty"]:.1f}, '
                   f'E2E-norm {p["p90_e2e_norm_intvty"]:.1f}, E2E {p["p90_e2el_s"]:.1f} s, TTFT {p["p90_ttft_s"]:.2f} s')
            o.append(f'<g class="pt {s["cls"]}"><title>{html.escape(tip)}</title>{marker(s["marker"], x, y)}'
                     f'<circle class="hit" cx="{x:.1f}" cy="{y:.1f}" r="9"/></g>')
            lx, ly, wl = x + 6, y - 6, 5.6 * len(p["label"])
            for dy in (0, 11, -11, 22, -22, 33):
                box = (lx, ly + dy - 9, lx + wl, ly + dy + 2)
                if all(box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3] for b in placed):
                    ly += dy; break
            placed.append((lx, ly - 9, lx + wl, ly + 2))
            o.append(f'<text class="plab {s["cls"]}" x="{lx:.1f}" y="{ly:.1f}">{html.escape(p["label"])}</text>')
    o.append("</svg>")
    return "\n".join(o)


def legend(series):
    items = []
    for s in series:
        sw = (f'<svg width="34" height="14" aria-hidden="true"><line class="fr {s["cls"]}" x1="2" x2="32" y1="7" y2="7"'
              + (f' stroke-dasharray="{s["dash"]}"' if s["dash"] else "") + f'/><g class="pt {s["cls"]}">{marker(s["marker"], 17, 7, 3.6)}</g></svg>')
        items.append(f'<span class="lg">{sw}{html.escape(s["name"])} <span class="muted">({len(s["pts"])} points)</span></span>')
    return '<div class="legend">' + "".join(items) + "</div>"


def main():
    series = load_series()
    body = [legend(series), '<div class="pgrid">']
    for key, label, hb in PANELS:
        body.append(f'<figure class="pfig">{panel(series, key, label, hb)}</figure>')
    body.append("</div>")
    p = f"{ROOT}/k31vr/index.html"
    t = open(p).read()
    new = "<!-- PARETO -->\n" + "\n".join(body) + "\n<!-- /PARETO -->"
    assert t.count("<!-- PARETO -->") == 1 and t.count("<!-- /PARETO -->") == 1
    t = re.sub(r"<!-- PARETO -->.*<!-- /PARETO -->", lambda m: new, t, flags=re.S)
    open(p, "w").write(t)
    print("pareto: " + ", ".join(f'{s["key"]} {len(s["pts"])}' for s in series))


if __name__ == "__main__":
    main()
