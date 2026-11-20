"""project93.py <parsed.json> <model_rows.json (93 layers)> <out.json> [tag]: 24-layer Rhea bench -> 93-layer D3-32 step.

Per op group, the measured busy / exposed time of the 24-layer view is scaled to the 93-layer model by its layer kind
(KDA x 69/18, DSA x 24/6, MoE sublayer x 92/23, AttnRes x 187/49, SP comm x 93/24; dense / embed / other x 1) and
compared with the 93-layer model achievable per group. Two terms are SUBSTITUTED, not measured here:
  MOE.megamoe: the 1-node run is EP4 with 896 experts (224 per GPU, 8x the EP32 expert weights per GPU), so its MegaMoE
    is not the D3-32 kernel. Substituted: FlashInfer SM107 EP32 MX standalone (moe_ep32 rows, T = B tokens per rank)
    x 92 layers, range x1.0 (FI) .. x1.2 (vendored in-step, calibrate.py).
  TAIL.drafter+final: the bench ran the stock DSpark preview drafter (5 MLA layers); production runs the release GQA
    drafter. Substituted: gap512 measured tail at the nearest B (D3 EP16, release drafter).
"""
import json
import sys

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import fam as F  # noqa: E402
import gap_parse as G  # noqa: E402

KIND_SCALE = {"KDA": 69 / 18, "DSA": 24 / 6, "MOE": 92 / 23}
GROUP_SCALE = {"ATTNRES": 187 / 49, "COMM.SP_AG": 93 / 24, "COMM.SP_RS": 93 / 24}
# FlashInfer EP32 MX standalone per layer (us), moe_ep32 rows (bench mode, random routing): T tokens/rank = B
FI_EP32_MX = {32: 66.64, 64: 70.31, 96: (70.31 + 77.51) / 2}
# gap512 measured TAIL (release drafter + final) exposed us at B32 / B64 / B92 (D3 EP16, L190K)
GAP512_TAIL = {32: 525.0, 64: 697.0, 96: 891.0}
MULTINODE = {32: 20.92, 64: 26.98, 96: 30.48}


def scale(group):
    if group in GROUP_SCALE:
        return GROUP_SCALE[group]
    return KIND_SCALE.get(group.split(".")[0], 1.0)


def main():
    d = json.load(open(sys.argv[1]))
    model = json.load(open(sys.argv[2]))
    tag = sys.argv[4] if len(sys.argv) > 4 else ""
    out = {}
    for case, c in sorted(d.items(), key=lambda kv: kv[1]["B"]):
        B = c["B"]
        mrow = model["decode"]["B%d_L135K" % B]
        rows = []
        for g in c["rows"]:
            name = g["group"]
            ids = G.GROUPS.get(name, [])
            s, a = G.model_sum(mrow, ids)
            f = scale(name)
            busy, expo = g["measured_us"] * f, g["exposed_us"] * f
            sub = ""
            if name == "MOE.megamoe":
                fi = FI_EP32_MX[B] * 92
                busy = expo = fi * 1.1
                sub = "substituted: FI EP32 MX standalone x92 x1.1 (range x1.0-1.2); 1-node EP4 measured %.0f us/layer" % (
                    g["measured_us"] / 23)
            elif name == "TAIL.drafter+final":
                busy = expo = GAP512_TAIL[B]
                sub = "substituted: gap512 release-drafter tail; 1-node stock preview drafter measured %.0f us" % g["measured_us"]
            elif name == "WAIT":
                continue
            rows.append(dict(group=name, measured_us=busy, exposed_us=expo, t_ach_us=a, t_sol_us=s,
                             derate=busy / a if a else None, exposed_gap_us=expo - a, calls=g["calls"],
                             kernels=g["kernels"], scale=f, substituted=sub,
                             per_layer_us=g["measured_us"] / max(1, round(g["calls"] / max(1, c["rows"][0]["calls"])))))
        cc = dict(rows=rows)
        fams = F.fam_table(cc)
        step = sum(r["exposed_us"] for r in rows)
        nib = G.model_sum(mrow, G.NOT_IN_BENCH)[1]
        ach = mrow["t_ach_ms"] * 1e3 - nib
        out[case] = dict(B=B, r=B / 4, bench24_step_us=c["step_median_us"], projected93_step_us=step,
                         model_ach_bench_scope_us=ach, model_sol_us=mrow["t_sol_ms"] * 1e3,
                         multinode_measured_ms=MULTINODE[B], rows=rows, families=fams)
        print("== %s r%g: 24L bench step %.2f ms | projected 93L step %.2f ms (MoE+tail substituted) | model ach %.2f ms "
              "(x%.2f) | 8-node D3-32 measured %.2f ms"
              % (tag, B / 4, c["step_median_us"] / 1e3, step / 1e3, ach / 1e3, step / ach, MULTINODE[B]))
        for r in fams:
            print("   %-68s busy %6.2f expo %6.2f ach %6.2f x%-5s gap %5.2f"
                  % (r["family"][:68], r["busy_us"] / 1e3, r["exposed_us"] / 1e3, r["t_ach_us"] / 1e3,
                     "%.2f" % r["derate"] if r["derate"] else "-", r["exposed_gap_us"] / 1e3))
    json.dump(out, open(sys.argv[3], "w"), indent=1)


if __name__ == "__main__":
    main()
