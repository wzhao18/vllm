#!/usr/bin/env python3
"""seed.py: initial campaign.json and status/*.json (2026-11-18). Owners overwrite their status file from here on.
Baselines: decode = D3-32 r24 (MXFP4, measured 8-node step 30.48 ms), per-family split transferred from the gap512 EP16
in-step profile (B92, r23) with MegaMoE moved to EP32; prefill = DEP16 F16K L78K NVFP4 projection (perf-model/plan
ledgers, reviewed). The Rhea 1-node profile (perf-model/microbench/results/rhea1n_d3x32) replaces the decode split."""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T = "2026-11-18T12:40-08:00"
camp = {
  "goal": "Close the gap between the implemented kernels and the perf model's achievable time for Kimi K3.1 on VR200: "
          "NVFP4 checkpoint (NVFP4 W4A4 routed experts, MXFP8 shared experts + KDA projections), DEP16 prefill, "
          "DCP4 x DP8 EP32 decode. Every kernel family gets an owner, a measured gap analysis and a target of at most "
          "1.1x model achievable.",
  "steps": {
    "Prefill DEP16 (tok/s/GPU)": {"now": "13.9K MXFP4 · 15.4K NVFP4-act", "ach": "30.7K NVFP4", "note": "F16K, L78K"},
    "Decode r8 (tok/s/GPU)": {"now": "1,178 MXFP4", "ach": "2,855 NVFP4", "note": "D3-32, 20.9 ms step"},
    "Decode r24 (tok/s/GPU)": {"now": "2,425 MXFP4", "ach": "6,280 NVFP4", "note": "D3-32, 30.5 ms step"},
  },
  "findings": [
    {"t": "11-18", "text": "Topology is not the lever: best unbuilt decode layouts project only +2-5% over DCP4 x DP8 EP32; the step is 2.1-2.3x model achievable, all of it kernel time."},
    {"t": "11-18", "text": "NVFP4 near-term (enablement only) projects prefill 19.4-20.5K and decode r8 1.2-1.45K; the rest of the gap to 30.7K / 2.9K is kernel efficiency."},
    {"t": "11-18", "text": "Largest decode gaps (r24, EP16-transferred): BF16 GEMMs 4.2 ms, MegaMoE ~3.4, DCP top-k 2.0, DSA attention 1.7, KDA 1.6, small ops 1.2."},
  ],
}
D = dict(side="decode", step_ms=30.5, shape="D3-32 r24: 96 requests / DP rank (384 verify rows), L≈135K, k=3",
         basis="gap512 EP16 in-step profile B92 (r23), MegaMoE moved to EP32 from the 8-node step delta", state="queued",
         owner="unassigned", updated=T, work=[], evidence=["perf-model/microbench/results/gap512/parsed/decode_families.txt"])
P = dict(side="prefill", step_ms=825, shape="DEP16 F16K (3 x 5,456 tok) at L78K, NVFP4", state="queued", owner="unassigned",
         basis="PROJECTED (reviewed ledger, perf-model/plan/nvfp4_ledgers_summary.txt)", updated=T, work=[],
         evidence=["perf-model/plan/PLAN_NVFP4.md"])
I = dict(side="infra", state="queued", owner="unassigned", updated=T, work=[], basis="-", shape="-")
ws = [
 dict(I, id="HW", title="Vera Rubin hardware reference database", gap_analysis="Every kernel workstream needs the VR200-specific features (tcgen05 block-scaled MMA shapes, TMEM, 2-CTA, TMA multicast/.COUNTED, CLC, PDL, locality domains, NVLink6 multimem) and their measured costs in one place.", next="Build the private hwdb and the per-workstream feature briefs."),
 dict(I, id="INFRA", title="Node pool, 1-node emulations, profiling and integration branch", gap_analysis="Kernel work must iterate on 1-4 GPUs while 8-node EP32 holds queue: 1-node DP-rank slice of D3-32 and a 1-node DEP16 slice, per-family profile parser vs model at the same shapes, integration branch for verified wins.", next="Take over the Rhea 1-node D3-32 profile, build the DEP16 1-node slice."),
 dict(I, id="EN", title="NVFP4 checkpoint enablement", gap_analysis="No NVFP4 number exists: MegaMoE format hard-coded to MXFP4 (model.py:816), Rubin MegaMoE refuses NVFP4 scales (experts.py:289/442), no NVFP4 EP32 decode backend, specialized decode raises on MXFP8 projections.", next="Native ModelOpt NVFP4 load into Rubin MegaMoE + dummy-safe NVFP4 view."),
 dict(D, id="D-GEMM", title="Decode: low-latency dense GEMM family (BF16 + MXFP8)", baseline_ms=8.9, current_ms=8.9, achievable_ms=4.7, target_ms=5.2,
      gap_analysis="Weight-streaming GEMMs at M=32-384 reach 1.8-8.7 TB/s of 13.9 achievable; cuBLAS picks 2-CTA 64xN / 160x64 tiles with 156-448 CTAs; split-K reduce launches ~1.2 ms/step; a CuTe split-K kernel collapses above 200 rows.",
      inefficiencies=["Too few bytes in flight per SM for 30-180 MB weight streams", "Separate split-K reduce kernels", "SplitKGemm tile policy collapse above 200 rows", "MXFP8 activation quant as separate kernels (NVFP4 ckpt)"]),
 dict(D, id="D-MOE", title="Decode: MegaMoE NVFP4 EP32 + shared-expert overlap", baseline_ms=8.3, current_ms=8.3, achievable_ms=4.9, target_ms=5.4,
      gap_analysis="Vendored deep_gemm fp8_fp4 MegaMoE runs 109-122 us/layer vs FI 91-95 us standalone; ~50 us fixed per call; shared experts run serially after MegaMoE; no NVFP4 EP32 decode backend exists yet.",
      inefficiencies=["Old vendored kernel", "Fixed per-call chain (~50 us/layer)", "Shared experts not overlapped", "NVFP4 W4A4 path missing at EP32"]),
 dict(D, id="D-TOPK", title="Decode: DSA exact top-k + DCP merge", baseline_ms=2.3, current_ms=2.3, achievable_ms=0.22, target_ms=0.25,
      gap_analysis="topk_merge_kernel<16> 52-77 us/layer vs 6-7 modelled; pp hist/window/win passes 25-45 us vs 1.4; merge grid [2, rows] walks 4 ranks x 2048 candidates serially with remote NVLink reads and flag polls; logits re-read three times; on the critical stream.",
      inefficiencies=["2 CTAs per row serial merge over remote reads", "Three passes over logits", "PDL off on the merge", "Critical-stream placement"]),
 dict(D, id="D-ATTN", title="Decode: DSA attention path (q path, sparse MLA, DCP value, indexer)", baseline_ms=3.3, current_ms=3.3, achievable_ms=1.3, target_ms=1.45,
      gap_analysis="MlaQuery q path 28-50 us/layer vs 5-10; sparse MLA fmha 18.7 us vs 5 at B50 (96->128 head padding, per-row 2048-row gather); indexer re-streams keys for each of a request's 4 verify rows.",
      inefficiencies=["96->128 head padding", "Per-row KV re-gather", "Unfused q path", "Indexer 4-row key re-streaming"]),
 dict(D, id="D-KDA", title="Decode: KDA core (FP32 state)", baseline_ms=3.4, current_ms=3.4, achievable_ms=1.7, target_ms=1.9,
      gap_analysis="KdaWide 30-49 us/layer vs 14-25 modelled: ~35% of HBM on 166 MB FP32 state per layer; per-(head-group, request) CTA grid latency-bound on the q=4 serial update; KdaWS above B80 is a separate kernel.",
      inefficiencies=["Latency-bound CTA grid", "35% HBM utilization on state stream", "Two kernels across batch sizes"]),
 dict(D, id="D-FUSE", title="Decode: small-op fusion, SP collectives, drafter tail", baseline_ms=3.9, current_ms=3.9, achievable_ms=1.5, target_ms=1.65,
      gap_analysis="AttnRes mix 5.8 us x 187 calls vs <1 us; unfused SiLU, latent RMSNorm, router top-k, MegaMoE prep ~1.2 ms; SP RS/AG at the Lamport floor plus separate local-sum kernels; drafter tail 2-3x.",
      inefficiencies=["Standalone elementwise kernels the model fuses", "Separate exchange + reduce kernels", "Drafter tail kernels"]),
 dict(P, id="P-MOE", title="Prefill: MegaMoE FI NVFP4 (2x8K) + shared-expert side stream", baseline_ms=320, current_ms=320, achievable_ms=198, target_ms=218,
      gap_analysis="Vendored megamoe_rubin NVFP4 in-step inferred 281-358 ms vs FI 230.7 ms standalone (2x8K) vs model 197.9; separate router/reduce staging kernels; link-bound layer (1.97 ms link floor) exposes ~0.68 ms/layer of GEMM; shared experts lose their overlap when staging is removed.",
      inefficiencies=["Vendored kernel staging", "Partial GEMM/link overlap", "Shared-expert side stream starved of SMs"]),
 dict(P, id="P-KDA", title="Prefill: KDA chain (core + conv + gated o_norm)", baseline_ms=78, current_ms=78, achievable_ms=16, target_ms=18,
      gap_analysis="96 heads at TP1 fall off the v-split FlashKDA build: 288 CTAs walk 5.4K tokens chunk-serially; prepare is a separate full pass; conv and gated RMSNorm are standalone HBM passes.",
      inefficiencies=["No 96-head v-split build", "Separate prepare pass", "Unfused conv and o_norm"]),
 dict(P, id="P-MLA", title="Prefill: sparse MLA + prologue", baseline_ms=67, current_ms=67, achievable_ms=16, target_ms=17.5,
      gap_analysis="Generation-phase trtllm-gen kernel on 3072 ShardQ rows at ~30% of FP8 MMA rate; 96->128 head padding; V512 split possibly recomputes QK (hypothesis); separate quant/pad/insert prologue kernels plus an unidentified 1 ms CuTe kernel.",
      inefficiencies=["Decode-form kernel used for prefill", "Head padding", "Unfused prologue"]),
 dict(P, id="P-IDX", title="Prefill: indexer logits + exact top-k", baseline_ms=61.5, current_ms=61.5, achievable_ms=33.5, target_ms=37,
      gap_analysis="topk_select_bf16 2 x ~120 us/layer (7x model); indexer logits re-read.", inefficiencies=["Multi-pass top-k", "Logit re-reads"]),
 dict(P, id="P-GEMM", title="Prefill: MXFP8 / BF16 GEMMs, router/latent, AttnRes, small ops", baseline_ms=290, current_ms=290, achievable_ms=245, target_ms=262,
      gap_analysis="MXFP8 KDA in_proj (unaligned N=49376) 1.16-1.40x, shared MXFP8 + SiLU-mul + quant, router/top-k/latent BF16 GEMMs 1.4x, AttnRes 6.6 ms, small BF16 GEMMs + drafter 14 ms excess.",
      inefficiencies=["Unaligned N tactics", "Separate activation quant", "Unfused router/latent epilogues"]),
]
os.makedirs(f"{ROOT}/status", exist_ok=True)
for w in ws:
    p = f"{ROOT}/status/{w['id']}.json"
    if not os.path.exists(p):
        json.dump(w, open(p, "w"), indent=1)
if not os.path.exists(f"{ROOT}/campaign.json"):
    json.dump(camp, open(f"{ROOT}/campaign.json", "w"), indent=1)
print("seeded", len(ws))
