#!/usr/bin/env python3
"""plot_pareto4.py: 4-way pareto of the current-code VR200 points (int64.2 / f19, 8 nodes) vs the 2026-11-17 release-draft
sweep (../k31vr-release-draft-sweep-20261117, chosen points, tree rel-1767cf7) and the InferenceX VR200 dynamo-sglang
baseline (vr200_sglang_infx.json), per GPU and per MW (3.3 kW all-in per VR200 GPU for every series).
Adapted from ../k31vr-release-draft-sweep-20261117/plot_pareto4.py.
points.json: {"points": [{label, run, tput_per_gpu, p90_intvty, p90_e2e_norm_intvty, p90_e2el_s, p90_ttft_s, win10_60: {...}}]}
Blue filled = full profiling phase (frz_metrics.py, the same convention as the other two series); blue hollow = minutes
10-60 (win_metrics.py 'end', the campaign ledger's convention), joined to its full-phase twin by a thin line. Linear axes."""
import json, os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
H = os.path.dirname(os.path.abspath(__file__))
cur = json.load(open(f"{H}/points.json"))["points"]
rel = [p for p in json.load(open(f"{H}/../k31vr-release-draft-sweep-20261117/points.json"))["points"] if p.get("chosen")]
sg = json.load(open(f"{H}/vr200_sglang_infx.json"))["points"]
KW = 3.3
panels = [("p90_intvty", "P90 interactivity (tok/s/user) = 1000 / P90 ITL", True),
          ("p90_e2e_norm_intvty", "P90 E2E-normalized interactivity (tok/s/user)", True),
          ("p90_e2el_s", "P90 end-to-end latency (s)", False), ("p90_ttft_s", "P90 TTFT (s)", False)]
BLUE, ORANGE, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#a3a19a", "#52514e", "#898781"
def front(pts, hb):
    f, best = [], -1
    for p in sorted(pts, key=lambda p: -p[0] if hb else p[0]):
        if p[1] > best: f.append(p); best = p[1]
    return sorted(f)
def fig4(per_mw, out):
    sy = (lambda t: t / KW / 1000) if per_mw else (lambda t: t / 1000)
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    for ax, (k, lab, hb) in zip(axs.flat, panels):
        g = [(p[k], sy(p["tput_per_gpu"]), p["label"]) for p in sg]
        gf = front(g, hb)
        ax.plot([p[0] for p in gf], [p[1] for p in gf], ":^", color=ORANGE, lw=2, ms=7, mec="white", mew=1.2, zorder=3,
                label="InferenceX VR200 dynamo-sglang (SemiAnalysis, 2026-11-20), pareto front")
        ax.scatter([p[0] for p in g], [p[1] for p in g], s=30, color=ORANGE, edgecolor="white", lw=1, zorder=3, label="InferenceX VR200 dynamo-sglang, all points")
        for x, y, c in g: ax.annotate(c, (x, y), xytext=(5, -10), textcoords="offset points", fontsize=6.5, color=ORANGE)
        rc = sorted([(p[k], sy(p["tput_per_gpu"]), p["label"]) for p in rel], key=lambda p: p[1])
        ax.plot([p[0] for p in rc], [p[1] for p in rc], "--s", color=GREY, lw=1.5, ms=5, mec="white", mew=1, zorder=2,
                label="VR200 vLLM release-draft sweep 2026-11-17 (tree rel-1767cf7; not re-run on current code)")
        for x, y, c in rc: ax.annotate(c, (x, y), xytext=(5, 4), textcoords="offset points", fontsize=6.5, color=MUTED)
        cf = sorted([(p[k], sy(p["tput_per_gpu"]), p["label"]) for p in cur], key=lambda p: p[1])
        cw = [(p["win10_60"][k], sy(p["win10_60"]["tput_per_gpu"])) for p in sorted(cur, key=lambda p: p["tput_per_gpu"])]
        for (x0, y0, _), (x1, y1) in zip(cf, cw): ax.plot([x0, x1], [y0, y1], "-", color=BLUE, lw=0.8, alpha=0.6, zorder=4)
        ax.plot([p[0] for p in cf], [p[1] for p in cf], "-o", color=BLUE, lw=2, ms=7, mec="white", mew=1.2, zorder=5,
                label="VR200 vLLM current code (int64.2 / f19 15c281dc32, 8 nodes), full profiling phase")
        ax.scatter([p[0] for p in cw], [p[1] for p in cw], s=46, facecolor="white", edgecolor=BLUE, lw=1.5, zorder=5,
                   label="same runs, minutes 10-60 (ledger convention)")
        for i, (x, y, c) in enumerate(cf):
            ax.annotate(c, (x, y), xytext=(-8, -13 if i == 0 else 7), textcoords="offset points", fontsize=7.5, color=INK, ha="right")
        ax.set_xlabel(lab + ("  (higher is better)" if hb else "  (lower is better)"), fontsize=9)
        ax.set_ylabel("M tok/s per all-in utility MW (input + output)" if per_mw else "Throughput per GPU (K tok/s/GPU, input + output)", fontsize=9)
        ax.set_ylim(0, None); ax.set_xlim(0, None); ax.grid(color="#e1e0d9", lw=0.8); ax.set_axisbelow(True)
        for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    h, l = [], []
    for ax in axs.flat:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l: h.append(hh); l.append(ll)
    fig.legend(h, l, frameon=False, loc="upper center", ncol=2, fontsize=8.5, bbox_to_anchor=(0.5, 0.96))
    fig.suptitle("Kimi K3.1 AgentX 4-way pareto on VR200 " + ("per MW (3.3 kW all-in per GPU)" if per_mw else "per GPU")
                 + ": vLLM current code vs release-draft sweep vs InferenceX sglang", fontsize=13)
    fig.text(0.01, 0.005, "Current code: int64.2 (f19, commit 15c281dc32, an ancestor of Inferact wzhao/k31-agentx-dev), 8n = P DEP16 + D DP4 x (TP4+DCP4) EP16, 32 GPUs, "
             "60-min AgentX profiling, golden AL k3; gsm8k (real rejection) strict 0.9795 on the same build.\n"
             "Grey: release-draft sweep chosen points (2026-11-17, tree rel-1767cf7). Orange: InferenceX API, Kimi-K3.1-Pro vr200 dynamo-sglang agentic, 11 points (2026-11-20).\n"
             "Workload caveat: InferenceX mean input per request varies 132K-315K by point vs ~149K here (see README).",
             fontsize=7.2, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92)); fig.savefig(out)
fig4(False, f"{H}/pareto4_per_gpu.png"); fig4(True, f"{H}/pareto4_per_mw.png")
