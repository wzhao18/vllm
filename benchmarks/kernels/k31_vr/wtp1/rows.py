"""wep-tp1-attn ledger rows from `vllm bench forward --profile-dir` CUPTI kernel sums of the DEP4 8-layer bench.

Rows (ledger.py CK names, same kernel families as CURRENT_SPLIT["dep16_rest"]):
  dsa_indexer = indexer logits kernel(s); dsa_topk = top-k kernels (+ logits upcast); dsa_core = sparse MLA core
  (+ its reduction); drafter = DSpark drafter dense MLA (tokenspeed split + reduction). dsa_glue (reported, not a
  ledger row today) = decode-query absorb/cat/quant + index remap + indexer q quant/prologue.
Per DSA layer = kernel sum / DSA layers in the bench (2 of 8); full-model step = per layer x 24 DSA layers;
drafter is per step (the drafter's layers do not depend on the target truncation).
usage: python3 rows.py <prof dir> [--dsa-layers 2] [--json out]"""
import argparse, glob, json, os, re, statistics
RULES = [
    ("dsa_indexer", r"paged_mqa_logits|mqa_logits"),
    ("dsa_topk", r"topk_cs|cooperative_topk|persistent_topk|top_k_per_row|_upcast_logits|deep_select|topk_per_row|"
                 r"filter_and_convert|stable_topk|balanced|\bsample_kernel|\bfilter_kernel|\bselect_kernel|wtp1_topk|"
                 r"fused_logits_topk"),
    ("dsa_core", r"prims_ts|MlaDecodeTs|fmhaSm10\d.*Sparse|fmhaReductionKernel|sparse_mla_attn|SparseMla"),
    ("drafter", r"tokenspeed_mla|KimiK3MLADecodeQConcat"),
    ("dsa_glue", r"sparse_mla_index_remap|_dsa_fused_prologue|_indexer_q_quant|_dsa_decode_q_absorb|_pad_query_heads|"
                 r"scaled_fp8_quant_kernel_strided"),
]
FULL_DSA_LAYERS = 24
def classify(name):
    for comp, pat in RULES:
        if re.search(pat, name):
            return comp
    return None
def load(path):
    d = json.load(open(path)); out = {c: 0.0 for c, _ in RULES}; names = {}
    for k in d["kernels"]:
        n = k.get("example") or k["name"]; c = classify(n)
        if c:
            out[c] += k["us"]; names.setdefault(c, []).append((round(k["us"], 1), k["calls"], n[:80]))
    return d["span_us"], out, names
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("prof_dir"); ap.add_argument("--dsa-layers", type=int, default=2)
    ap.add_argument("--json"); ap.add_argument("-v", action="store_true"); a = ap.parse_args()
    res = {}
    cases = sorted({os.path.basename(f).split(".rank")[0] for f in glob.glob(os.path.join(a.prof_dir, "*.rank*.kernels.json"))})
    for case in cases:
        per = []
        for f in sorted(glob.glob(os.path.join(a.prof_dir, f"{case}.rank*.kernels.json"))):
            span, rows, names = load(f); per.append((span, rows))
            if a.v and f.endswith("rank0.kernels.json"):
                for c, ls in names.items():
                    for x in sorted(ls, reverse=True):
                        print(f"   {case} {c:12s} {x}")
        r = {"span_us_mean": statistics.mean(p[0] for p in per), "ranks": len(per), "layer_us": {}, "step_ms_full": {},
             "layer_us_max_rank": {}}
        for c, _ in RULES:
            vals = [p[1][c] for p in per]
            if c == "drafter":
                r["step_ms_full"][c] = statistics.mean(vals) / 1e3
                r["layer_us"][c] = statistics.mean(vals); r["layer_us_max_rank"][c] = max(vals)
            else:
                lay = [v / a.dsa_layers for v in vals]
                r["layer_us"][c] = statistics.mean(lay); r["layer_us_max_rank"][c] = max(lay)
                r["step_ms_full"][c] = statistics.mean(lay) * FULL_DSA_LAYERS / 1e3
        res[case] = r
        print(f"{case}: span {r['span_us_mean']:.0f} us | per DSA layer us (mean/max rank): " +
              ", ".join(f"{c} {r['layer_us'][c]:.1f}/{r['layer_us_max_rank'][c]:.1f}" for c, _ in RULES if c != "drafter") +
              f" | drafter/step {r['layer_us']['drafter']:.0f}/{r['layer_us_max_rank']['drafter']:.0f} us | full-step ms: " +
              ", ".join(f"{c} {v:.3f}" for c, v in r["step_ms_full"].items()))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
if __name__ == "__main__":
    main()
