"""d4x4 per-component breakdown of `vllm bench forward --profile-dir` kernel summaries (CUPTI kernel sums; streams overlap).
usage: python3 components.py <prof dir> [--rank N] [--json out]"""
import argparse, glob, json, os, re
RULES = [
    ("dcpx", r"dcp_exchange|dcpx|k31_dcp"),
    ("comm", r"nccl|[Aa]ll[Rr]educe|allreduce|all_gather|AllGather|ReduceScatter|reduce_scatter|alltoall|a2a|lamport|oneshot|twoshot|cross_device|custom_ar|symm|dcp_.*(gather|scatter|combine)|lse_combine|correct_attn"),
    ("idx_logits", r"paged_mqa_logits|mqa_logits"),
    ("idx_topk", r"topk_cs|cooperative_topk|persistent_topk|top_k_per_row|_upcast_logits|deep_select|topk_per_row|filter_and_convert|stable_topk|pack_dcp_topk"),
    ("sparse", r"fmhaSm10\d.*Sparse|fmhaReductionKernel|prims|sparse_mla|SparseMla"),
    ("draft_mla", r"tokenspeed_mla|KimiK3MLADecodeQConcat"),
    ("kda", r"[Kk]da|recoverssm|gated_delta|causal_conv|fused_recurrent"),
    ("moe", r"mega_moe|megamoe|single_group_topk|grouped_topk"),
    ("gemm", r"nvjet|cublas|splitKreduce|gemm|Gemm|GEMM|LLBf16|mxfp8|cutlass|sm100_|Sm100"),
]
ORDER = [r[0] for r in RULES] + ["other"]
def classify(name):
    for comp, pat in RULES:
        if re.search(pat, name):
            return comp
    return "other"
def load(path):
    d = json.load(open(path)); comps = {}; top = {}
    for k in d["kernels"]:
        n = k.get("example") or k["name"]; c = classify(n)
        comps[c] = comps.get(c, 0.0) + k["us"]
        if c == "other":
            top[n[:90]] = top.get(n[:90], 0.0) + k["us"]
    return d["span_us"], comps, top
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("prof_dir"); ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--json"); ap.add_argument("--other", action="store_true"); a = ap.parse_args()
    out = {}
    print("| case | span us | " + " | ".join(ORDER) + " |"); print("|---|---|" + "---|" * len(ORDER))
    for f in sorted(glob.glob(os.path.join(a.prof_dir, f"*.rank{a.rank}.kernels.json"))):
        case = os.path.basename(f).split(".rank")[0]
        span, comps, top = load(f); out[case] = dict(span_us=span, **comps)
        print(f"| {case} | {span:.0f} | " + " | ".join(f"{comps.get(c, 0.0):.0f}" for c in ORDER) + " |")
        if a.other:
            for n, us in sorted(top.items(), key=lambda x: -x[1])[:12]:
                print(f"    other {us:8.0f}  {n}")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
if __name__ == "__main__":
    main()
