#!/usr/bin/env python3
"""plot_pareto4.py: 4-way pareto of the VR200 release-draft sweep (DSpark e4e6e626, real K3.1-Pro d9e99b07, frozen tree rel-1767cf7)
vs the frozen preview-draft sweep (../k31vr-frozen-sweep-20261114) and the GB300 SemiAnalysis baseline, per GPU and per MW.
points.json: {"points": [{label, run, chosen: bool, tput_per_gpu, p90_intvty, p90_e2e_norm_intvty, p90_e2el_s, p90_ttft_s, ...}]}
Chosen points are blue (line joins the chosen points in x order); points not used for the pareto are grey. Linear axes."""
import json, os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
H = os.path.dirname(os.path.abspath(__file__))
vr = json.load(open(f"{H}/points.json"))["points"]
old = [p for p in json.load(open(f"{H}/../k31vr-frozen-sweep-20261114/points.json"))["points"] if p.get("chosen")]
gb = json.load(open(f"{H}/gb300_sa_k31.json"))["points"]
panels = [("p90_intvty", "P90 interactivity (tok/s/user) = 1000 / P90 ITL", True),
          ("p90_e2e_norm_intvty", "P90 E2E-normalized interactivity (tok/s/user)", True),
          ("p90_e2el_s", "P90 end-to-end latency (s)", False), ("p90_ttft_s", "P90 TTFT (s)", False)]
BLUE, ORANGE, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#b4b2ab", "#52514e", "#898781"
def front(pts, hb):
    f, best = [], -1
    for p in sorted(pts, key=lambda p: -p[0] if hb else p[0]):
        if p[1] > best: f.append(p); best = p[1]
    return sorted(f)
def fig4(per_mw, out):
    sy = (lambda t, kw: t / kw / 1000) if per_mw else (lambda t, kw: t / 1000)
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    for ax, (k, lab, hb) in zip(axs.flat, panels):
        g = [(p[k], sy(p["tput_per_gpu"], 2.12), p["label"]) for p in gb]
        ch = [(p[k], sy(p["tput_per_gpu"], 3.3), p["label"]) for p in vr if p.get("chosen")]
        un = [(p[k], sy(p["tput_per_gpu"], 3.3), p["label"]) for p in vr if not p.get("chosen")]
        gf = front(g, hb)
        ax.plot([p[0] for p in gf], [p[1] for p in gf], ":^", color=ORANGE, lw=2, ms=7, mec="white", mew=1.2, label="GB300 (SemiAnalysis InferenceX K3.1, dynamo-vllm, 2026-11-10)")
        ax.scatter([p[0] for p in g], [p[1] for p in g], s=30, color=ORANGE, edgecolor="white", lw=1, zorder=3)
        for x, y, c in g: ax.annotate(c, (x, y), xytext=(5, -10), textcoords="offset points", fontsize=6.5, color=ORANGE)
        if un:
            ax.scatter([p[0] for p in un], [p[1] for p in un], s=36, color=GREY, edgecolor="white", lw=1, zorder=2, label="VR200 runs not used for the pareto")
            for x, y, c in un: ax.annotate(c, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=6.5, color=MUTED)
        oc = sorted([(p[k], sy(p["tput_per_gpu"], 3.3), p["label"]) for p in old], key=lambda p: p[1])
        ax.plot([p[0] for p in oc], [p[1] for p in oc], "--s", color=GREY, lw=1.5, ms=5, mec="white", mew=1, zorder=3, label="VR200 frozen sweep, preview draft (2026-11-14)")
        if ch:
            cs = sorted(ch, key=lambda p: p[1])
            ax.plot([p[0] for p in cs], [p[1] for p in cs], "-o", color=BLUE, lw=2, ms=7, mec="white", mew=1.2, zorder=4, label="VR200 release draft (rel-1767cf7, real K3.1-Pro, release golden AL)")
            for x, y, c in ch: ax.annotate(c, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=7.5, color=INK)
        ax.set_xlabel(lab + ("  (higher is better)" if hb else "  (lower is better)"), fontsize=9)
        ax.set_ylabel("M tok/s per all-in utility MW (input + output)" if per_mw else "Throughput per GPU (K tok/s/GPU, input + output)", fontsize=9)
        ax.set_ylim(0, None); ax.set_xlim(0, None); ax.grid(color="#e1e0d9", lw=0.8); ax.set_axisbelow(True)
        for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    h, l = [], []
    for ax in axs.flat:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l: h.append(hh); l.append(ll)
    fig.legend(h, l, frameon=False, loc="upper center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("Kimi K3.1 AgentX 4-way pareto " + ("per MW (VR200 3.3 kW, GB300 2.12 kW all-in per GPU)" if per_mw else "per GPU") + ": VR200 release DSpark draft vs preview draft vs GB300", fontsize=13)
    fig.text(0.01, 0.005, "VR200 release draft: Inferact/Kimi-K3.1-Pro-DSpark e4e6e626 on real Kimi-K3.1-Pro d9e99b07, frozen tree rel-1767cf7 (wzhao/k31-dspark-release), "
             "60-min AgentX profiling, release golden AL (k3 3.08, k5 3.81, k7 4.16), gsm8k (real rejection) on the same allocation.\n"
             "Grey: frozen sweep 2026-11-14 (preview draft, test weights, golden AL k3 2.88 / k7 3.69), chosen points. "
             "GB300: SemiAnalysis InferenceX Kimi-K3.1-Pro GB300 dynamo-vllm (7 runs, 2026-11-10).", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.025, 1, 0.93)); fig.savefig(out)
fig4(False, f"{H}/pareto4_per_gpu.png"); fig4(True, f"{H}/pareto4_per_mw.png")
