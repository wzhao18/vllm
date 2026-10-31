"""Per-DSA-layer kernel breakdown of a `vllm bench forward --profile-dir` trace (K3.1, TP1 / DEP).

usage: python3 dsa_layer.py <trace.json.gz> [<trace.json.gz> ...] [--json out.json] [--seq]

A DSA attention block is cut from the CUPTI kernel trace as the kernels between the
AttnRes (pre-attention residual + norm) that precedes a DeepGEMM paged-MQA-logits launch and
the next AttnRes (the pre-MoE one), on any stream. Kernels are grouped by role:

  proj      projection GEMMs (fused_qkv_a / wkv_index / q_b / wq_index / g / o, MXFP8 or BF16,
            incl. split-K reduces and MXFP8 activation quant)
  absorb    W_UK / W_UV batched GEMMs (cuBLAS NN/TN BMM on [96, rows, 128|512])
  prologue  q_a / kv_a norms, MLA + indexer key insert (fused prologue or stock pieces)
  idx_q     indexer query quant (+ weight fold)
  logits    DeepGEMM paged MQA logits
  topk      indexer top-k (balanced sample/filter/select, cooperative, upcast)
  qprep     sparse-core query prep (ql_nope|q_pe concat, FP8 quant, head pad, index remap)
  core      sparse MLA core (trtllm-gen fmha / prims_ts) + its reduction
  gate      output gate (sigmoid * mul) and other elementwise glue
Every layer's sum and its wall span (first start to last end) are reported; the median over
all DSA layers in all replays is printed per kernel name.
"""
import argparse
import gzip
import json
import re
import statistics
from collections import defaultdict

ROLES = [
    ("logits", r"mqa_logits"),
    ("topk", r"topk|_upcast_logits|balanced|sample_kernel|filter_kernel|select_kernel|deep_select"),
    ("core", r"fmhaSm10\d|fmhaReduction|prims|sparse_mla_fwd|SparseMla|sparse_decode|flash_fwd_splitkv_mla|tilelang"),
    ("qprep", r"CatArray|scaled_fp8_quant|_pad_query_heads|index_remap|decode_q_quant|qprep|static_scaled_fp8"),
    ("idx_q", r"indexer_q_quant|q_quant_mxfp4|fold_weights"),
    ("prologue", r"rms_norm|dsa_fused_prologue|_indexer_k_norm|concat_and_cache_mla|prologue|k_norm"),
    ("absorb", r"_NNT$|_NN$|bmm|Bmm"),
    ("proj", r"nvjet|cublas|splitKreduce|gemm|Gemm|GEMM|LLBf16|mxfp8|Mxfp8|quantize|cutlass|cute"),
    ("gate", r"sigmoid|BinaryFunctor|elementwise|gate"),
]


def role(name):
    for r, pat in ROLES:
        if re.search(pat, name):
            return r
    return "other"


def kernels(path):
    t = json.load(gzip.open(path) if path.endswith(".gz") else open(path))
    ev = t["traceEvents"] if isinstance(t, dict) else t
    ks = [e for e in ev if e.get("cat") == "kernel"]
    ks.sort(key=lambda e: e["ts"])
    return ks


def dsa_blocks(ks):
    anchors = [i for i, e in enumerate(ks) if "mqa_logits" in e["name"]]
    res = [i for i, e in enumerate(ks) if "attn_res" in e["name"]]
    blocks = []
    for a in anchors:
        before = [i for i in res if i < a]
        after = [i for i in res if i > a]
        if not before or not after:
            continue
        lo, hi = before[-1] + 1, after[0]
        if hi - lo > 200:
            continue
        blocks.append(ks[lo:hi])
    return blocks


def short(name):
    name = re.sub(r"^void ", "", name)
    name = re.sub(r"<.*", "", name) if not name.startswith("nvjet") else name
    return name[:90]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--json")
    ap.add_argument("--seq", action="store_true", help="print one block's kernel sequence")
    a = ap.parse_args()
    out = {}
    for path in a.traces:
        blocks = dsa_blocks(kernels(path))
        if not blocks:
            print(path, "no DSA blocks")
            continue
        sums = [sum(e["dur"] for e in b) for b in blocks]
        spans = [b[-1]["ts"] + b[-1]["dur"] - b[0]["ts"] for b in blocks]
        by_role = defaultdict(list)
        by_name = defaultdict(list)
        for b in blocks:
            r = defaultdict(float)
            n = defaultdict(float)
            for e in b:
                r[role(e["name"])] += e["dur"]
                n[(role(e["name"]), short(e["name"]))] += e["dur"]
            for k, v in r.items():
                by_role[k].append(v)
            for k, v in n.items():
                by_name[k].append(v)
        med = {k: statistics.median(v + [0.0] * (len(blocks) - len(v))) for k, v in by_role.items()}
        case = path.split("/")[-1].split(".rank")[0]
        print(f"\n## {case}: {len(blocks)} DSA blocks, median sum {statistics.median(sums):.1f} us, "
              f"median span {statistics.median(spans):.1f} us")
        order = ["proj", "absorb", "prologue", "idx_q", "logits", "topk", "qprep", "core", "gate", "other"]
        print("| " + " | ".join(order) + " |")
        print("|" + "---|" * len(order))
        print("| " + " | ".join(f"{med.get(k, 0.0):.1f}" for k in order) + " |")
        for (r, n), v in sorted(by_name.items(), key=lambda kv: (order.index(kv[0][0]), -statistics.median(kv[1]))):
            print(f"  {r:8s} {statistics.median(v):7.1f}  x{len(v):3d}  {n}")
        if a.seq:
            b = blocks[len(blocks) // 2]
            t0 = b[0]["ts"]
            for e in b:
                print(f"    {e['ts'] - t0:8.1f} {e['dur']:7.1f} s{e['args'].get('stream')} {role(e['name']):8s} {short(e['name'])}")
        out[case] = dict(blocks=len(blocks), sum_us=statistics.median(sums), span_us=statistics.median(spans), **med)
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
