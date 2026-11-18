"""assemble.py <out dir>: lane gap512 rows.json from the parsed decode/prefill tables + MegaMoE arm rows."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fam  # noqa: E402

OUT = sys.argv[1]
H = os.path.dirname(os.path.abspath(__file__))
W = "/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/gap512"
VER_D = ("release tree overlay dsrel/ov/rel-1767cf7 (rel-1767cf7 sweep tree staged by mkov.sh; SOURCE_REV base 6a14e2fad), "
         "image vllm-openai-cu134-nightly-3f1bcfd82b, FlashInfer 0.7.3.dev20261031, MegaMoE = vendored deep_gemm "
         "sm100_fp8_fp4_mega_moe_situ_impl (image deep_gemm + vLLM sm_107 situ patch; NOT FI 6e438852b), release DSpark "
         "draft kimi-k3.1-pro-dspark-e4e6e626 on FLASH_ATTN, decode_backend latency_specialized, dummy weights, "
         "routing balanced, FP8 MLA KV, MXFP4 indexer, FP32 KDA state, MXFP8 MoE act (kernel fp8_fp4)")
VER_P = VER_D.replace("decode_backend latency_specialized", "prefill role of nq-c512-xdr (decode_backend auto, "
                                                            "ShardQ, KDA segscan, VEC_SITU)")
rows = []


def add_case(path, topo, ver, kind, hosts):
    d = json.load(open(path))
    for case, c in d.items():
        ach = c["model_t_ach_us"] - (c.get("model_not_in_bench_ach_us") or 0)
        rows.append(dict(case=case, kind=kind, topology=topo, group="STEP", op_ids=["STEP"],
                         measured_us=round(c["step_median_us"], 1), t_sol_us=round(c["model_t_sol_us"], 1),
                         t_ach_us=round(ach, 1), derate=round(c["step_median_us"] / ach, 3),
                         model_key=c.get("model_key"),
                         notes="whole forward: CUDA-graph replay median (rank-max, 20x5); model minus target LM head/"
                               "rejection/DP sync (not in the bench graph). hosts %s. %s" % (hosts, ver)))
        if "B" in c:
            for f in fam.fam_table(c):
                rows.append(dict(case=case, kind=kind, topology=topo, group="FAMILY: " + f["family"],
                                 op_ids=f["groups"], measured_us=round(f["busy_us"], 1),
                                 exposed_us=round(f["exposed_us"], 1), t_sol_us=round(f["t_sol_us"], 1),
                                 t_ach_us=round(f["t_ach_us"], 1),
                                 derate=round(f["derate"], 2) if f["derate"] else None,
                                 exposed_gap_us=round(f["exposed_gap_us"], 1), notes="family sum of the groups below"))
        for r in c["rows"]:
            der = r.get("derate")
            flag = "KERNEL-DEBUG" if der and der > 2.0 and r["measured_us"] > 100 else ""
            rows.append(dict(case=case, kind=kind, topology=topo, group=r["group"], op_ids=r["op_ids"],
                             measured_us=round(r["measured_us"], 1), measured_max_rank_us=round(r["measured_max_us"], 1),
                             exposed_us=round(r["exposed_us"], 1), t_sol_us=round(r["t_sol_us"], 1),
                             t_ach_us=round(r["t_ach_us"], 1), derate=round(der, 2) if der else None,
                             exposed_gap_us=round(r["exposed_gap_us"], 1), flag=flag, calls=r["calls"],
                             kernels={k: round(v, 1) for k, v in r["kernels"].items()},
                             notes="us per step per GPU; measured = summed kernel time (same-stream PDL overlap "
                                   "clipped), median of 3 profiled replays then median over 16 ranks; exposed = "
                                   "additive share of the busy timeline (cross-stream overlap split evenly, spin-wait "
                                   "kernels get only idle instants)"))


add_case(os.path.join(H, "dec1/parsed.json"), "D3 DP4x(TP4+DCP4) SP, EP16, k3 (C512 decode)", VER_D, "decode",
         "hecate0370-0373 job 1137218")
add_case(os.path.join(H, "dec_b80/parsed.json"), "D3 DP4x(TP4+DCP4) SP, EP16, k3 (C512 decode)", VER_D, "decode",
         "hecate0046-0049 job 1137216")
add_case(os.path.join(H, "dec_b92cap/parsed.json"), "D3 DP4x(TP4+DCP4) SP, EP16, k3 (C512 decode)",
         VER_D + "; contexts capped to 12.0M tokens per DP rank (scale 0.686, mean 130K) after the uncapped B92 OOMed "
                 "in the bench state snapshot; model at L=130K", "decode", "hecate0046-0049 job 1137216")
add_case(os.path.join(H, "pre1/parsed.json"), "P2 DP4x(TP4=DCP4) SP ShardQ, EP16 (C512 prefill)", VER_P, "prefill",
         "hecate0077-0080 job 1137222")

# MegaMoE: in-step vendored deep_gemm vs standalone rows (existing megamoe-bench, EP16 mxfp4w_mxfp8a, random routing)
B7 = {16: 108.9, 32: 108.7, 64: 114.9, 128: 119.1, 2048: 596.6, 4096: 1111.7}
B10 = {16: 90.2, 32: 91.4, 64: 93.8, 128: 97.1, 2048: 465.0, 4096: 900.7}


def interp(tab, x):
    ks = sorted(tab)
    for a, b in zip(ks, ks[1:]):
        if a <= x <= b:
            return tab[a] + (tab[b] - tab[a]) * (x - a) / (b - a)
    return None


model = json.load(open(os.path.join(H, "model_rows.json")))
for path, key in (("dec1/parsed.json", None), ("dec_b80/parsed.json", None), ("dec_b92cap/parsed.json", None),
                  ("pre1/parsed.json", "pre")):
    d = json.load(open(os.path.join(H, path)))
    for case, c in d.items():
        r = next(x for x in c["rows"] if x["group"] == "MOE.megamoe")
        if key == "pre":
            tok = c["F"] * 4 // 16
        else:
            tok = c["B"] * 4 * 4 // 16
        per = r["measured_us"] / 92.0
        ach = r["t_ach_us"] / 92.0
        b7, b10 = interp(B7, tok), interp(B10, tok)
        rows.append(dict(case=case, kind="megamoe-arm", topology="EP16 MegaMoE, %d tokens/rank" % tok,
                         group="MOE.megamoe per layer", op_ids=["MOE.dispatch", "MOE.fc1", "MOE.act", "MOE.fc2",
                                                                "MOE.combine", "COMM.P11.moe", "COMM.P12.moe"],
                         measured_us=round(per, 1), t_ach_us=round(ach, 1), derate=round(per / ach, 2),
                         standalone_vendored_B7_us=round(b7, 1) if b7 else None,
                         standalone_newest_B10_us=round(b10, 1) if b10 else None,
                         derate_newest_B10=round(b10 / ach, 2) if b10 else None,
                         projected_step_saving_us=round((per - b10) * 92, 0) if b10 else None,
                         notes="in-step = vendored deep_gemm situ kernel (this run). Standalone rows reused from "
                               "megamoe-bench results/rows.jsonl (B7 deepgemm_vllm_sm107_situ_patch = same kernel; "
                               "B10 = FlashInfer 6e438852b tag wzhao/kimi-k3-megamoe-sm107-final-20261117), EP16, "
                               "mxfp4w_mxfp8a, random routing, linear interpolation in tokens/rank. Not an in-step "
                               "measurement of B10 (no vLLM wiring of the FI build in rel-1767cf7)."))

os.makedirs(OUT, exist_ok=True)
json.dump(rows, open(os.path.join(OUT, "rows.json"), "w"), indent=1)
print(len(rows), "rows")
for r in rows:
    if r["kind"] == "megamoe-arm":
        print(r["case"], r["topology"], r["measured_us"], r["t_ach_us"], r["standalone_vendored_B7_us"],
              r["standalone_newest_B10_us"], r["projected_step_saving_us"])
