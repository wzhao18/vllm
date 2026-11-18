"""Add KERNEL-DEBUG hypotheses + priority to rows.json and print summary tables (markdown)."""
import collections
import json
import statistics as st
import sys

base = sys.argv[1]
rows = json.load(open(base + "/rows.json"))
cases = {c["id"]: c for c in json.load(open(base + "/src/cases_ops1.json"))}


def hyp(r):
    c = r["case"]
    s = r.get("shape") or {}
    m, a = r["measured_us"], r["t_ach_us"]
    if m - a < 6 and a < 3:
        return ("fixed per-kernel floor: launch + cold-DRAM latency + one partial wave (measured ~4-12 us at "
                "any size); model t_ach has no per-kernel floor (only the 0.78 us gap in scoring)")
    if c.startswith("gemm"):
        if s.get("batch", 1) > 1:
            return "torch.bmm (cuBLAS batched) at tiny K=128/N=128 per head: poor tile fill; fused per-head kernel needed"
        if s["M"] <= 1024:
            return ("weight-streaming GEMM below HBM roofline: cuBLAS/cuBLASLt split-K tiles at small M do not "
                    "saturate HBM (low_latency CuTe table did not apply to these shapes in this tree)")
        return "large-M GEMM below MMA roofline (tile quantization / wave tail at this N)"
    if c.startswith("sparse_mla"):
        if s["rows"] >= 2048:
            return ("trtllm-gen sparse MLA runs ~3.7-4.1 PFLOPS on 128 padded heads (2.9 PF useful), cold == warm: "
                    "not HBM bound; head padding 96->128 costs 1.33x, the rest is the per-token decode-kernel "
                    "formulation (one request per token, no KV reuse across rows; softmax/PV pipeline) vs the "
                    "model's FP8 MMA rate")
        return "small-row trtllm-gen launch: few CTAs, fixed ~10-12 us floor (cold == warm)"
    if c.startswith("dsa_indexer"):
        if "requests_in_chunk" in s:
            return ("logits at ~6.9 PF FP4 are close to model; DeepSelect top-2048 over 78K BF16 columns adds "
                    "~75% (2.4 TB/s over the logits) that the model treats as near-free")
        return ("stock paged DeepGEMM logits run next_n=1 per row, so the q=4 rows of a request each stream the "
                "keys (4x the model's 68 B/key scan); DeepSelect top-k latency floor ~10-37 us for few long rows")
    if c.startswith("kda"):
        if s.get("phase") == "prefill":
            return ("FlashKDA chunked recurrence: sequential over 5.4K-token sequences with only nseq*heads "
                    "CTAs; 96 heads falls off the v-split build (stock path) -> 2.8x")
        return ("sdec PreparedKda verify writes every token's FP32 checkpoint state (1 read + 4 writes of "
                "64 KiB/head/request) = ~5x the model's state bytes; production RecoverSSM (records, not states) "
                "is not covered by this bench; c512 step profile shows 33 us/layer at B50")
    if c.startswith("attnres"):
        return "Triton/CUDA AttnRes mix: per-row CTA with up to 9 sources; small rows latency-bound"
    if c.startswith("drafter"):
        if s.get("op") == "DS.attn":
            return ("FlashInfer paged prefill for q3 x 2048-window GQA d64: ~22 us floor and ~90 us at 268 MB KV "
                    "(3 TB/s): plan/split-kv + merge kernels; needs a decode-style GQA kernel")
        return "cuBLAS skinny GEMM (K=512) below HBM roofline / fixed floor"
    return ""


for r in rows:
    if r.get("measured_us") is None:
        continue
    c = cases.get(r["case"], {})
    r["priority"] = c.get("priority")
    r["count_per_step"] = c.get("count_per_step")
    r["excess_us_per_step"] = round((r["measured_us"] - r["t_ach_us"]) * (c.get("count_per_step") or 0), 1)
    r["kernel_debug"] = bool(r["derate"] and r["derate"] > 2.0)
    if r["kernel_debug"] and not r.get("hypothesis"):
        r["hypothesis"] = hyp(r)
json.dump(rows, open(base + "/rows.json", "w"), indent=1)

fam = collections.defaultdict(list)
for r in rows:
    if r.get("derate"):
        f = r["case"].rsplit("-", 1)[0]
        if f == "gemm":
            s = r["shape"]
            f = f"gemm {r['dtype']} {'bmm' if s['batch'] > 1 else ''}{'M<=64' if s['M'] <= 64 else 'M<=1024' if s['M'] <= 1024 else 'M>1024'}"
        if f == "drafter_verify":
            f = "drafter " + r["shape"]["op"]
        if f == "kda":
            f = "kda " + r["shape"]["phase"]
        if f == "dsa_indexer_topk":
            f = "dsa idx+topk " + ("prefill" if "requests_in_chunk" in r["shape"] else "decode")
        if f == "sparse_mla":
            f = "sparse_mla " + ("prefill" if r["shape"]["rows"] >= 2048 and "DCP" not in r["shape"]["mode"] else "decode")
        fam[f].append(r)
print("| family | n | median derate | time-weighted (sum meas / sum ach) | min | max | >2.0 |")
print("|---|---|---|---|---|---|---|")
for f, rs in sorted(fam.items()):
    d = [r["derate"] for r in rs]
    tw = sum(r["measured_us"] for r in rs) / sum(r["t_ach_us"] for r in rs)
    print(f"| {f} | {len(rs)} | {st.median(d):.2f} | {tw:.2f} | {min(d):.2f} | {max(d):.2f} | {sum(x > 2 for x in d)} |")
print()
print("| case | prio | ops | topologies | shape | measured us | t_ach us | t_sol us | derate |")
print("|---|---|---|---|---|---|---|---|---|")
for r in sorted([r for r in rows if r.get("priority") == "P0"], key=lambda r: r["case"]):
    s = r["shape"]
    sh = ",".join(f"{k}={v}" for k, v in s.items() if k in ("batch", "M", "N", "K", "rows", "B", "heads", "mode", "phase", "tokens", "mean_keys_per_row") and not (k == "batch" and v == 1))
    print(f"| {r['case']} | P0 | {'/'.join(r['op_ids'])} | {','.join(r['topology'])} | {sh} {r.get('dtype') or ''} | "
          f"{r['measured_us']:.1f} | {r['t_ach_us']:.1f} | {r['t_sol_us']:.1f} | {r['derate']:.2f}{' KD' if r['kernel_debug'] else ''} |")
print()
print("top excess us/step (measured - t_ach) x count:")
for r in sorted([r for r in rows if r.get("excess_us_per_step") is not None], key=lambda r: -r["excess_us_per_step"])[:25]:
    print(f"  {r['case']} {'/'.join(r['op_ids'])} {','.join(r['topology'])} excess {r['excess_us_per_step']:.0f} us/step derate {r['derate']:.2f}")
