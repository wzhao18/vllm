#!/usr/bin/env python3
"""EP16 / EP28 / EP32 MegaMoE rows (FlashInfer SM107 EP32 build a368c5f66, lane r3-ep 89fafd77e) vs the current model.

    python3 perf-model/microbench/results/moe_ep32/mkrows.py     # raw/ep32_rows.jsonl -> rows.json

Source: Hecate megamoe-bench/opt/results.jsonl rows labelled ep32-* (copied to raw/). Kept: bench mode (ep32-bn,
capacity = T, decode T 1-512) and prefill bench mode (ep32-pf / pf16k / pf16g, capacity = T, T 4096-16384); median
over repeats per (quant, EP, T). Not kept: vLLM mode (within 1-3% of bench), C (EP16 regression), X (cross-EP).
t_ach_us = this model's achievable MegaMoE layer at the same EP and tokens per rank (calibrate.megamoe_model_layer).
"""
import collections
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PM = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, PM)
import calibrate as Cal  # noqa: E402

QK = {"mxfp4_mxfp8": "mxfp4w_mxfp8a", "nvfp4": "nvfp4w_nvfp4a"}
CK = {"mxfp4_mxfp8": "mxfp4_release", "nvfp4": "nvfp4_modelopt"}


def main():
    raw = [json.loads(l) for l in open(os.path.join(HERE, "raw", "ep32_rows.jsonl"))]
    grp = collections.defaultdict(list)
    for x in raw:
        lab = x["label"]
        if not lab.startswith(("ep32-bn.", "ep32-pf.", "ep32-pf16k.", "ep32-pf16g.")):
            continue
        if x["status"] != "ok" or x["check"] != "pass":
            continue
        grp[(x["quant"], x["ep"], x["T"])].append(x)
    rows = []
    for (q, ep, T), xs in sorted(grp.items()):
        meas = statistics.median(x["median_us"] for x in xs)
        sol = Cal.megamoe_model_layer(CK[q], ep, T, col="sol") * 1e6
        ach = Cal.megamoe_model_layer(CK[q], ep, T) * 1e6
        rows.append(dict(case="megamoe-ep32-%s-ep%d-T%d" % (q, ep, T), ep=ep, tokens_per_rank=T, quant_kind=QK[q],
                         ckpt=CK[q], mode="prefill bench" if T >= 4096 else "bench",
                         measured_us=round(meas, 2), measured_kernel_us=round(meas, 2), n_repeats=len(xs),
                         t_sol_us=round(sol, 2), t_ach_us=round(ach, 2), derate=round(meas / ach, 3),
                         kernel="FlashInfer SM107 MegaMoE a368c5f66 (lane r3-ep 89fafd77e)",
                         labels=sorted({x["label"] for x in xs}),
                         notes=("EP32 prefill cliff: +21-30% vs EP16 at T16384 (kernel issue, not modelled)"
                                if ep == 32 and T == 16384 else "")))
    with open(os.path.join(HERE, "rows.json"), "w") as f:
        json.dump(rows, f, indent=1)
    for r in rows:
        print("%-14s EP%-2d T%-5d meas %8.1f  ach %8.1f  derate %.2f %s" % (r["quant_kind"], r["ep"], r["tokens_per_rank"],
              r["measured_us"], r["t_ach_us"], r["derate"], "VIOL" if r["t_ach_us"] > r["measured_us"] else ""))


if __name__ == "__main__":
    main()
