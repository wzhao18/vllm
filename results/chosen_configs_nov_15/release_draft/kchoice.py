#!/usr/bin/env python3
"""kchoice.py: throughput-optimal draft length k per chosen point for the release draft. This is a recommendation only: the
renders keep the frozen k. Standard library only; prints the tables quoted in README.md.

Question: given the release golden AL curve (k1..k7 = 1.86 2.56 3.08 3.46 3.81 4.00 4.16) and the release drafter cost, which
k gives the most tokens per second per request at each point's load?
  - On TP16=DCP16 decode, k <= 7 (KdaSplit verify, the speculative-config cap).
  - C512's sequence-parallel decode takes k = 3 only (runtime/__init__.py sp_unsupported; SP_MAX_ROWS 512 = 128 seqs x q4),
    so C512 stays at k3.
At a fixed concurrency the gain shows up mostly as interactivity x. AgentX users spend about 16 s per cycle idle (think/tool
time G) against about 3-6 s in decode, so y at the same C moves about 5x less. Along the pareto front, though, the x gain buys y:
the chosen C48 -> C64 segment has d ln y / d ln x = -1.2.

Model (per decode instance, calibrated on the 15 TP16=DCP16 frozen-sweep AgentX runs; inputs in RUNS):
  step(B, q) = T(B, q) + Dr(B, k)
    q = k + 1 verify rows per request.
    B = requests in decode. The step is evaluated at max(B, 1), the batch while decoding; the mean B includes idle time.
  T_A(B, q): the target verify step, fitted on every run's measured server step minus its preview drafter (Dr_prev, the
    per-shape drafter time of the TP16=DCP16 step bench):  T_A = a + b E_d(qB) + c qB + d B.
    E_d(M) = 896 (1 - (1 - 16/896)^M) distinct experts (MoE weight reads); qB rows (attention, indexer, KDA, GEMMs); B
    per-request work.
  T(B, q) = T_A(B, 8) - kappa (T_A(B, 8) - T_A(B, q)). T_A's q-sensitivity is scaled by kappa, fitted to the one clean
    measured q contrast: the C96 1P2D A/B (k7 a1093473 vs k3 a1097000, same deployment, B 7.06 vs 7.15). The k7 vs k3
    server step curves at B 25.5 (xpyd/model.out) give kappa 0.76, a cross-check. kappa = 1 is reported as the
    "uncalibrated" bound. Per point, T is scaled to hit that run's measured step at its own (B, q).
  Dr_rel(B, k) = 0.186 + 0.00127 B k ms: the release drafter with the ported fast path (GQA window attention, no MLA; the
    study's estimates 0.18-0.21 ms at B1 k7 ... 0.33 ms at B38 k3). Sensitivity: the stock release drafter
    Dr_stock = 0.55 + 0.0012 B k ms.
  Closed loop at the point's concurrency C (the xpyd/model.py form, with per-point factors eta, chi, tau from its run):
    D = eta OSL step / AL / 1e3 s
    lambda = C / (G + TTFT + D), with G = G0 - 0.242 (E2E - E2E0)
    B = chi lambda D / ny
    x = 1000 / (tau step / AL)
    y = lambda tok/req / GPUs
    TTFT is held: k does not change the prefill load.
Break-even: at equal B, k beats k0 iff step(k) / step(k0) < AL(k) / AL(k0).
"""

from __future__ import annotations

import math

RELEASE_AL = {1: 1.86, 2: 2.56, 3: 3.08, 4: 3.46, 5: 3.81, 6: 4.00, 7: 4.16}
PREVIEW_AL = {1: 1.82, 2: 2.44, 3: 2.88, 4: 3.19, 5: 3.41, 6: 3.57, 7: 3.69}

# Per-run measured inputs: hti/lowlat/midcurve/xpyd/extract.py over each run's profiling window (client profile_export.jsonl,
# decode server 10-s intervals; caches in hti/lowlat/midcurve/xpyd/cache and dsrel/cfg/kchoice/cache on Hecate).
# job: (C, k, AL, decode instances, B/inst, step ms, lambda/s, G s, TTFT mean s, D s, OSL, tok/req, p90 ITL ms, GPUs, client
#       decode concurrency)
RUNS = {
    "a1093473": (96, 7, 3.69, 2, 7.063, 10.460, 4.6916, 16.670, 0.827, 2.965, 1056.7, 146066, 3.4450, 40, 13.912),
    "a1093679": (160, 3, 2.88, 1, 35.304, 14.985, 6.3577, 16.206, 3.455, 5.506, 1062.6, 145595, 5.8696, 24, 35.003),
    "a1093680": (64, 7, 3.69, 1, 12.277, 13.081, 3.3124, 15.161, 0.484, 3.676, 1041.6, 150545, 4.2965, 24, 12.176),
    "a1093681": (48, 7, 3.69, 1, 6.698, 10.154, 2.3001, 17.553, 0.409, 2.906, 1055.8, 162196, 3.3771, 24, 6.685),
    "a1093682": (8, 7, 3.69, 1, 0.733, 5.699, 0.4057, 17.732, 0.237, 1.750, 1165.8, 170973, 1.8880, 24, 0.710),
    "a1093683": (1, 7, 3.69, 1, 0.245, 4.552, 0.1122, 6.397, 0.285, 2.230, 1753.1, 287119, 1.2844, 24, 0.250),
    "a1093685": (224, 3, 2.88, 1, 69.584, 21.672, 8.9720, 16.132, 1.219, 7.615, 1022.2, 148340, 8.8401, 32, 68.326),
    "a1094590": (160, 3, 2.88, 1, 38.854, 15.690, 6.6813, 17.214, 0.963, 5.770, 1063.0, 148709, 6.2012, 32, 38.554),
    "a1094730": (112, 3, 2.88, 1, 24.110, 13.089, 5.0820, 15.928, 1.442, 4.669, 1040.5, 141030, 5.3670, 24, 23.727),
    "a1095076": (256, 3, 2.88, 1, 88.160, 25.089, 10.0184, 15.325, 1.567, 8.661, 1004.2, 143542, 10.0056, 32, 86.775),
    "a1096270": (128, 3, 2.88, 1, 27.634, 13.476, 5.5608, 16.413, 1.684, 4.921, 1060.5, 138949, 5.4158, 24, 27.363),
    "a1096427": (160, 3, 2.88, 1, 37.672, 15.628, 6.5055, 16.698, 2.176, 5.720, 1062.8, 146584, 6.1889, 24, 37.214),
    "a1096999": (16, 7, 3.69, 2, 0.550, 4.898, 0.7747, 18.977, 0.264, 1.412, 1028.6, 160294, 1.6651, 40, 1.094),
    "a1097000": (96, 3, 2.88, 2, 7.154, 8.358, 4.6671, 16.492, 0.984, 3.093, 1054.2, 145384, 3.4503, 40, 14.434),
    "a1098025": (72, 7, 3.69, 3, 2.831, 7.495, 3.9019, 15.650, 0.594, 2.208, 1067.9, 146686, 2.5158, 56, 8.616),
}
# the chosen TP16=DCP16 points (C512's SP decode is k3-only)
POINTS = (("C1", "a1093683"), ("C16 1P2D", "a1096999"), ("C72 1P3D", "a1098025"), ("C48 1P1D", "a1093681"),
          ("C64", "a1093680"), ("C128 1P1D", "a1096270"), ("C160 1P1D llh", "a1096427"))
AB = ("a1093473", "a1097000")  # C96 1P2D, k7 vs k3, same deployment
# xpyd/model.out server step curves at B 25.5: k7 interval 17.60 ms; k3 anchors (24.11, 13.09)-(29.13, 14.02) -> 13.35 ms
CROSS_B, CROSS_K7, CROSS_K3 = 25.5, 17.60, 13.09 + (14.02 - 13.09) * (25.5 - 24.11) / (29.13 - 24.11)
# preview drafter per decode step, TP16=DCP16 step bench (hti/lowlat/results/dec-c48-bal-1064153-1113224545-1331131,
# dec-c160-bal-1064153-1113225937-1632543: "target X us, drafter Y us"), B -> ms
DR_PREV = {7: [(1, 0.322), (2, 0.380), (3, 0.399), (6, 0.491), (12, 0.738), (22, 1.022)],
           3: [(8, 0.397), (16, 0.498), (32, 0.719), (40, 0.892), (55, 1.097), (64, 1.154)]}
G_SLOPE = -0.242  # xpyd model: G = 17.85 - 0.242 * E2E_mean


def Ed(M: float) -> float:
    return 896 * (1 - (1 - 16 / 896) ** M)


def interp(tab, x):
    if x <= tab[0][0]:
        (x0, y0), (x1, y1) = tab[0], tab[1]
    elif x >= tab[-1][0]:
        (x0, y0), (x1, y1) = tab[-2], tab[-1]
    else:
        i = next(i for i in range(1, len(tab)) if x <= tab[i][0])
        (x0, y0), (x1, y1) = tab[i - 1], tab[i]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def dr_prev(B, k):
    return max(0.30, interp(DR_PREV[k], B))


def dr_rel(B, k):
    return 0.186 + 0.00127 * B * k


def dr_stock(B, k):
    return 0.55 + 0.0012 * B * k


def lstsq(A, b):
    n = len(A[0]); M = [[sum(A[r][i] * A[r][j] for r in range(len(A))) for j in range(n)] for i in range(n)]
    v = [sum(A[r][i] * b[r] for r in range(len(A))) for i in range(n)]
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(M[r][i])); M[i], M[p] = M[p], M[i]; v[i], v[p] = v[p], v[i]
        for r in range(n):
            if r != i and M[i][i]:
                f = M[r][i] / M[i][i]; M[r] = [a - f * c for a, c in zip(M[r], M[i])]; v[r] -= f * v[i]
    return [v[i] / M[i][i] for i in range(n)]


def feats(B, q):
    B = max(B, 1.0)
    return [1.0, Ed(q * B), q * B, B]


class Target:
    def __init__(self, kappa=None, exclude=()):
        rows = [(r[4], r[1] + 1, r[5] - dr_prev(r[4], r[1])) for j, r in RUNS.items() if j not in exclude]
        self.c = lstsq([feats(B, q) for B, q, _ in rows], [t for _, _, t in rows])
        self.rows = rows
        if kappa is None:  # fit to the A/B's measured target ratio q4/q8
            (_, k7, _, _, b7, s7, *_), (_, k3, _, _, b3, s3, *_) = RUNS[AB[0]], RUNS[AB[1]]
            meas = (s3 - dr_prev(b3, k3)) / (s7 - dr_prev(b7, k7))
            b = (b3 + b7) / 2
            model = self.ta(b, 4) / self.ta(b, 8)
            kappa = (1 - meas) / (1 - model)
        self.kappa = kappa

    def ta(self, B, q):
        return sum(a * b for a, b in zip(self.c, feats(B, q)))

    def __call__(self, B, q):
        return self.ta(B, 8) - self.kappa * (self.ta(B, 8) - self.ta(B, q))


def solve(job, k, T, al_table, drafter):
    """Closed-loop projection of run `job`'s deployment at draft length k (AL from al_table, drafter model `drafter`)."""
    C, k0, al0, ny, B0, st0, lam0, G0, tt0, D0, osl, tok, itl90, gpus, _ = RUNS[job]
    scale = (st0 - dr_prev(B0, k0)) / T(B0, k0 + 1)  # per-point calibration at the measured (B, q)
    eta = D0 * al0 / (osl * st0 / 1e3)
    chi = B0 * ny / (lam0 * D0)
    tau = itl90 * al0 / st0
    al = al_table[k]

    def step(B):
        return scale * T(B, k + 1) + drafter(max(B, 1.0), k)

    B, D, lam = B0, D0, lam0
    for _ in range(1000):
        D = eta * osl * step(B) / al / 1e3
        G = G0 + G_SLOPE * (D - D0)
        lam = C / (G + tt0 + D)
        Bn = chi * lam * D / ny
        if abs(Bn - B) < 1e-10:
            break
        B = 0.5 * B + 0.5 * Bn
    s = step(B)
    return dict(B=B, step=s, x=1000 / (tau * s / al), y=lam * tok / gpus)


def measured(job):
    C, k0, al0, ny, B0, st0, lam0, G0, tt0, D0, osl, tok, itl90, gpus, _ = RUNS[job]
    return dict(B=B0, step=st0, x=1000 / itl90, y=lam0 * tok / gpus)


def main():
    Tc = Target()
    Tu = Target(kappa=1.0)
    c = Tc.c
    print("## Target verify step T_A (15 TP16=DCP16 full-hour runs: server step minus preview drafter), step at max(B, 1)")
    print(f"T_A(B, q) = {c[0]:.3f} + {c[1]*1e3:.3f} us E_d(qB) + {c[2]*1e3:.2f} us qB + {c[3]*1e3:.1f} us B   [ms]")
    print("| run | C | k | B | measured step | preview drafter | target | T_A | error |")
    print("|---|---|---|---|---|---|---|---|---|")
    errs = []
    for j, r in sorted(RUNS.items(), key=lambda kv: (kv[1][1], kv[1][4])):
        t = r[5] - dr_prev(r[4], r[1]); f = Tc.ta(r[4], r[1] + 1); errs.append(f / t - 1)
        print(f"| {j} | {r[0]} | {r[1]} | {r[4]:.2f} | {r[5]:.2f} | {dr_prev(r[4], r[1]):.2f} | {t:.2f} | {f:.2f} | {100*(f/t-1):+.1f}% |")
    print(f"rms {100*math.sqrt(sum(e*e for e in errs)/len(errs)):.1f}%")
    cb = CROSS_B
    model_x = Tc.ta(cb, 4) / Tc.ta(cb, 8)
    meas_x = (CROSS_K3 - dr_prev(cb, 3)) / (CROSS_K7 - dr_prev(cb, 7))
    print(f"\nkappa (q-sensitivity scale of T_A) fitted to the C96 1P2D A/B (target q4/q8 measured "
          f"{(RUNS[AB[1]][5]-dr_prev(RUNS[AB[1]][4],3))/(RUNS[AB[0]][5]-dr_prev(RUNS[AB[0]][4],7)):.3f} vs T_A "
          f"{Tc.ta(7.1, 4)/Tc.ta(7.1, 8):.3f} at B 7.1): kappa = {Tc.kappa:.3f}; cross-check, server step curves at B {cb}: "
          f"target q4/q8 {meas_x:.3f} vs T_A {model_x:.3f} -> kappa {(1-meas_x)/(1-model_x):.3f}")

    print("\n## Validation: C96 1P2D A/B, each run projected from the other (preview AL, preview drafter)")
    for T, nm in ((Tc, f"kappa {Tc.kappa:.2f}"), (Tu, "kappa 1 (uncalibrated)")):
        for src, dst in (AB, AB[::-1]):
            p = solve(src, RUNS[dst][1], T, PREVIEW_AL, dr_prev)
            m = measured(dst)
            print(f"{nm}: {src} (k{RUNS[src][1]}) -> k{RUNS[dst][1]}: projected x {p['x']:.1f} y {p['y']:.0f} B {p['B']:.2f} step "
                  f"{p['step']:.2f} | measured {dst}: x {m['x']:.1f} y {m['y']:.0f} B {m['B']:.2f} step {m['step']:.2f} | "
                  f"error x {100*(p['x']/m['x']-1):+.1f}% y {100*(p['y']/m['y']-1):+.1f}%")

    print("\n## Break-even at equal B: k beats k0 iff step(k)/step(k0) < AL(k)/AL(k0)")
    for k0 in (7, 3):
        print(f"k0 = {k0}: " + ", ".join(f"k{k} {RELEASE_AL[k]/RELEASE_AL[k0]:.3f}" for k in range(3, 8) if k != k0)
              + f"   (preview table: " + ", ".join(f"k{k} {PREVIEW_AL[k]/PREVIEW_AL[k0]:.3f}" for k in range(3, 8) if k != k0) + ")")

    for dname, drafter in (("fast-path release drafter (Dr_rel)", dr_rel), ("stock release drafter (Dr_stock)", dr_stock)):
        print(f"\n## Release draft, {dname}: change of x / y vs the frozen k at the same concurrency "
              f"(calibrated kappa {Tc.kappa:.2f}; [uncalibrated kappa 1])")
        print("| point | frozen k | B | measured (preview) x / y | release at frozen k: x / y | k3 | k4 | k5 | k6 | k7 | best k |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for label, job in POINTS:
            k0 = RUNS[job][1]
            m = measured(job)
            base = solve(job, k0, Tc, RELEASE_AL, drafter)
            base_u = solve(job, k0, Tu, RELEASE_AL, drafter)
            cells, best = [], (k0, 1.0, 1.0)
            for k in range(3, 8):
                p = solve(job, k, Tc, RELEASE_AL, drafter)
                pu = solve(job, k, Tu, RELEASE_AL, drafter)
                rx, ry, rxu = p["x"] / base["x"], p["y"] / base["y"], pu["x"] / base_u["x"]
                cells.append(f"{100*(rx-1):+.1f}% / {100*(ry-1):+.1f}% [{100*(rxu-1):+.1f}%]")
                if rx > best[2] * 1.0000001:
                    best = (k, ry, rx)
            print(f"| {label} | {k0} | {m['B']:.2f} | {m['x']:.0f} / {m['y']:.0f} | {base['x']:.0f} ({100*(base['x']/m['x']-1):+.1f}%) / "
                  f"{base['y']:.0f} ({100*(base['y']/m['y']-1):+.1f}%) | " + " | ".join(cells) +
                  f" | k{best[0]} ({100*(best[2]-1):+.1f}% x, {100*(best[1]-1):+.1f}% y) |")


if __name__ == "__main__":
    main()
