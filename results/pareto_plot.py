import csv, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
VR_KW, GB_KW = 3.3, 2.12
# K3.1 VR freeze sweep (official 60-min AgentX, live_metrics.py): (point, tok/s/GPU, interactivity = 1000/p90 ITL, run)
vr = [("C1", 1347, 805, "a1057686"), ("C8", 2960, 553, "a1057685"), ("C48 (1P1D)", 15593, 305, "a1055047"),
      ("C64", 21002, 250, "a1055046"), ("C160", 31174, 164, "a1047927"), ("C224", 42908, 119, "a1052665"),
      ("C512", 76544, 87, "a1050536")]
# GB300 Dynamo vLLM baseline, M tok/s/MW at 2.12 kW/GPU: (point, M tok/s/MW, interactivity)
gb = [("c1", 0.72, 451), ("c20", 2.2, 263), ("c32", 3.6, 211), ("c56", 5.8, 159), ("c192", 9.7, 94), ("c384", 14.4, 72), ("c352", 17.1, 52)]
vr_pts = sorted([(x, t / VR_KW / 1000, c, t) for c, t, x, _ in vr])
gb_pts = sorted([(x, m, c, m * 1e6 * GB_KW / 1e6 * 1000 / 1000) for c, m, x in gb])
fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
ax.plot([p[0] for p in vr_pts], [p[1] for p in vr_pts], "-o", color="#2a78d6", lw=2, ms=7, mec="white", mew=1.5, label="K3.1 VR200 (final freeze sweep: C160, C512 freeze d; others freeze e)")
ax.plot([p[0] for p in gb_pts], [p[1] for p in gb_pts], ":^", color="#eb6834", lw=2, ms=8, mec="white", mew=1.5, label="GB300 Dynamo vLLM baseline")
for x, y, c, _ in vr_pts: ax.annotate(f"{c} {y:.2f}M", (x, y), xytext=(6, 6), textcoords="offset points", fontsize=8, color="#52514e")
for x, y, c, _ in gb_pts: ax.annotate(f"{c} {y:.1f}M", (x, y), xytext=(7, -4), textcoords="offset points", fontsize=8, color="#52514e", ha="left")
ax.set_xlabel("P90 interactivity (tok/s/user) = 1000 / P90 ITL (ms)")
ax.set_ylabel("Token throughput per all-in utility MW (M tok/s/MW)")
ax.set_title("Kimi K3.1 AgentX pareto: VR200 vs GB300 baseline", fontsize=12)
ax.set_xlim(0, 860); ax.set_ylim(0, 25)
ax.grid(color="#e1e0d9", lw=0.8); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(frameon=False, loc="upper right")
fig.text(0.01, 0.005, "y = total (input+output) tok/s per GPU x 1e6 / all-in utility W per GPU (VR200 3.3 kW, GB300 2.12 kW). VR points: 60-min AgentX runs on Hecate, 2026-11-12/13.", fontsize=7, color="#898781")
fig.tight_layout(rect=(0, 0.02, 1, 1))
fig.savefig(sys.argv[1])
with open(sys.argv[2], "w", newline="") as f:
    w = csv.writer(f); w.writerow(["series", "point", "interactivity_p90", "tok_s_per_gpu", "M_tok_s_per_MW", "run"])
    for c, t, x, r in vr: w.writerow(["K3.1 VR200 final freeze sweep", c, x, t, round(t / VR_KW / 1000, 3), r])
    for c, m, x in gb: w.writerow(["GB300 Dynamo vLLM baseline", c, x, round(m * GB_KW * 1000), m, ""])
