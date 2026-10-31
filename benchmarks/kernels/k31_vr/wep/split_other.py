# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Split a DEP step's CUPTI kernel sums (`vllm bench forward --profile-dir`
*.kernels.json) into the gap-ledger rows that wep-proj-gemm-p2 owns, with the
ledger's `other` row (norms, AttnRes, router, embed, epilogues) split further.

usage: python split_other.py <kernels.json> [...] [--layers 8|full] [--json out]

`--layers 8`: the layer-reduced bench (K31W0_LAYERS=8: 2 DSA, 6 KDA, 7 MoE
+ 1 dense, all 8 with AttnRes / norms); per-layer kernels are scaled to the
full K3.1 stack (24 DSA, 69 KDA, 92 MoE + 1 dense = 93 layers) by the layer
family that launches them; per-step kernels (embedding, sampling-side,
drafter) are not scaled. The scale factors are per category below.
"""

import argparse
import json
import re

# (category, ledger row, regex on the kernel name/example, scale family)
# families: dsa (24/2), kda (69/6), moe (92/7), layer (93/8), step (1)
RULES = [
    # attention-side rows (not owned here; listed so they leave `other`)
    ("idx_logits", "dsa_indexer", r"paged_mqa_logits|mqa_logits", "dsa"),
    ("idx_topk", "dsa_topk",
     r"topk_cs|cooperative_topk|persistent_topk|top_k_per_row|_upcast_logits|"
     r"deep_select|topk_per_row|balanced_topk|sampled_topk", "dsa"),
    ("sparse", "dsa_core", r"fmhaSm10\d.*Sparse|fmhaReductionKernel|sparse_mla", "dsa"),
    ("dsa_prep", "dsa_core",
     r"concat_and_cache_mla|_indexer_k_norm_quant|_indexer_q_quant|_pad_query_heads|"
     r"dsa_fused_prologue|_dsa_|indexer|scaled_fp8_quant", "dsa"),
    ("draft_mla", "drafter", r"tokenspeed_mla|KimiK3MLADecodeQConcat|dflash|dspark", "step"),
    ("kda", "kda", r"[Kk]da(?!_rmsnorm_gated)|recoverssm|gated_delta|causal_conv|"
     r"fused_recurrent|segscan|flashkda", "kda"),
    ("moe_routed", "moe_routed", r"mega_moe|megamoe", "moe"),
    # owned rows
    ("gemm", "projections+moe_shared", r"nvjet|cublas|splitKreduce|gemm|Gemm|GEMM|"
     r"LLBf16|cutlass.*[Bb]lockscaled|Sm100BlockScaled|sm107.*mma|dense_blockscaled", "layer"),
    ("act_quant", "projections+moe_shared",
     r"mxfp8_quantize|mxfp8|Mxfp8|act_quant_kernel|quantize_kernel|cvt_fp16_to_fp4|"
     r"fp4_quant|scaled_fp4", "layer"),
    ("kda_out_norm", "other", r"_kda_rmsnorm_gated", "kda"),
    # the ledger books the router in moe_shared ("MoE shared/router/latent")
    ("router", "moe_shared", r"single_group_topk|grouped_topk|topk_softmax|_router", "moe"),
    ("attn_res", "other", r"attn_res", "layer"),
    ("rms_norm", "other", r"rms_norm|rmsnorm|RMSNorm|layernorm", "layer"),
    ("act_fn", "other", r"situ_and_mul|silu_and_mul|act_and_mul|gelu", "moe"),
    ("gate_mul", "other", r"sigmoid|_gate", "dsa"),
    ("rotary", "other", r"rotary", "dsa"),
    ("copy_cat", "other", r"CatArrayBatchedCopy|Memcpy|copy_|indexSelect|gather|scatter|"
     r"index_elementwise", "layer"),
    ("elementwise", "other", r"elementwise_kernel|vectorized_elementwise|reduce_kernel", "layer"),
    ("sampling", "other", r"_gumbel_sample|_kernel_slots|sample|argmax|softmax", "step"),
]
SCALE = {"dsa": 24 / 2, "kda": 69 / 6, "moe": 92 / 7, "layer": 93 / 8, "step": 1.0}


def classify(name):
    for cat, row, pat, fam in RULES:
        if re.search(pat, name):
            return cat, row, fam
    return "unclassified", "other", "layer"


def split(path, layers):
    d = json.load(open(path))
    cats = {}
    names = {}
    for k in d["kernels"]:
        name = k.get("example") or k["name"]
        cat, row, fam = classify(name)
        s = SCALE[fam] if layers == "8" else 1.0
        us = k["us"] * s
        c = cats.setdefault(cat, dict(row=row, us=0.0, calls=0))
        c["us"] += us
        c["calls"] += k["calls"]
        names.setdefault(cat, []).append((round(us, 1), k["calls"], k["name"][:90]))
    return d, cats, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--layers", default="8")
    ap.add_argument("--json")
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    res = {}
    for f in a.files:
        d, cats, names = split(f, a.layers)
        rows = {}
        for cat, c in cats.items():
            rows[c["row"]] = rows.get(c["row"], 0.0) + c["us"]
        res[f] = dict(span_us=d["span_us"], cats={k: round(v["us"], 1) for k, v in cats.items()},
                      rows={k: round(v, 1) for k, v in rows.items()})
        print(f"== {f}  span {d['span_us']:.0f} us (as measured, not scaled)")
        for cat, c in sorted(cats.items(), key=lambda kv: -kv[1]["us"]):
            print(f"  {cat:14s} {c['row']:24s} {c['us']:9.1f} us  calls {c['calls']}")
            if a.v or cat == "unclassified":
                for n in sorted(names[cat], reverse=True)[:8]:
                    print(f"      {n}")
        print("  rows: " + ", ".join(f"{k} {v / 1e3:.3f} ms" for k, v in sorted(rows.items())))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
