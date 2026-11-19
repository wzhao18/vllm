# k31ws step benches (INFRA)

All run on the Rhea login node through the shared pool (`wsgpu.sh`), with an overlay staged by `mkov.sh`.
Results land in `/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/<NAME>/step_{d,p,multi}/<TAG>/`.

| script | what | GPUs |
|---|---|---|
| `tools/step_d.sh [-n ID] [-t TAG] [-e 32] [-c u132k] [-x] <ov> <rows> [prof]` | D3-32 decode, one DP rank (TP4=DCP4, specialized decode, k=3, drafter) | `-g 4` |
| `tools/step_p.sh [-n ID] [-t TAG] [-e 16] [-c cases] [-x] <ov> [prof]` | P3-16 prefill, DEP4 on one node standing in for DEP16 | `-g 4` |
| `tools/step_multi.sh [-n ID] d8 <ov> <rows> [prof]` | full D3-32 (DP8 EP32, 93 layers) | `-g 8n` |
| `tools/step_multi.sh [-n ID] p16 <ov> - [prof]` | full P3-16 DEP16 (4 of the 8 nodes, 93 layers) | `-g 8n` |

Rows (decode) = verify rows per DP rank = 4 x requests per DP rank: r8/r16/r24 per GPU = 128,256,384.
Long runs: `setsid nohup bash .../step_d.sh ... > log 2>&1 < /dev/null &` and poll `status.txt` / the log.

## 1-node slice vs deployment

- **24-layer view** (`models/k31-view-L24`: 18 KDA + 6 DSA + 23 MoE + 1 dense, same 3:1 pattern, AttnRes block 12):
  896 experts at EP4 do not fit at 93 layers. Model rows are rescaled to the same layer counts
  (`mk_model_rows.py` -> `model_rows_L24*.json`, `pf_model_rows_L24.json`), so per-family ratios are like for like;
  multiply per-layer numbers by the 93-layer counts for the full step.
- **Expert-count emulation** (`nq_step.py --emulate-ep N`, default 32 decode / 16 prefill): bench-only. After the dummy
  weights are drawn, the router selection bias of experts outside an active set is set to -100 (sigmoid scores are in
  (0,1), so top-16 always picks active experts; routing weights come from the scores, not the bias). Active set:
  `e % (E/ep) < E/N`, i.e. E/N experts on every EP rank (28 at decode, 56 at prefill) under linear or round-robin
  placement. Token-expert pairs per GPU equal the deployed ones (rows x 16 / 4 per GPU here = rows x 8 x 16 / 32 at
  EP32), so each GPU runs the deployed expert count at the deployed tokens per expert. Needed because the specialized
  decode fuses the router GEMM into its MoE front (reads `moe.gate.weight` directly), so `--routing balanced`
  never reaches its graph; the bench's balanced fixture is made emulation-aware too (stock paths), and its per-case
  bias reset is undone. Without it the slice touches all 224 local experts per GPU (8x the deployed expert weight
  traffic at decode).
- **Die-split active set + EP-size records** (INFRA 2026-11-18 17:35, from D-MOE): the E/N active experts per rank are
  split over the two die halves of the rank's local experts (local ids [0, a/2) and [L/2, L/2 + a/2)), as the deployed
  die-local placement streams them, and decode MegaMoE selects the emulated EP size's buckets/records (megamoe_rubin
  select_rubin_buckets knob_world_size hook; EP32 128-token record at decode). References measured before this change
  are not comparable to runs after it.
- **MegaMoE dispatch/combine** spans 4 GPUs of one node instead of 32 (decode) / 16 (prefill) GPUs.
- **Drafter**: DSpark preview (`dspark-preview-L24`: aux layer ids remapped to the 24-layer target), included by
  default (`-x` excludes it). Decode: TOKENSPEED_MLA backend per the release config (also at prefill: the preview drafter is MLA with FP8 KV).
- Context: decode `u132k` (uniform 132K, matches the model's L135K rows) or `agentx`; prefill F16K / F8K at
  KV length 78,360 (prefix hit, 3 requests per GPU) and a fresh 16K.
- Timing: one-shot CUDA graph per case, rank-max, median of iters x samples (`vllm bench forward`); no LM head,
  rejection or CPU step overhead.

## Model rows: fused MegaMoE (INFRA 2026-11-18 17:50)

perf-model prices a MegaMoE layer as one fused kernel (cost.megamoe_time: the NVLink dispatch/combine stream runs under
the expert GEMMs). The rows carry that layer as `megamoe` (per view) and `t_{sol,ach}_ms_fused` (the additive step with
the fused MegaMoE in place of MOE.dispatch..combine + COMM.P11/P12); gap_parse/pf_fam use them for the MegaMoE family
and the step header. With it the per-family sum equals the model step (prefill F16K 532.8 ms; decode r24 11.71 vs 11.77
ms, 93 layers). Regenerate: `python3 mk_model_rows.py` (L24) and `python3 mk_model_rows.py --L93`.
Per-family tables for k31ws/measured/: `mk_measured.py <gap.json> <out.json> decode|prefill <slice> <overlay> <run>`.

## Parsing

- decode: `gap_parse.py decode <run dir> model_rows_L24.json gap.json --L 135` then `fam.py gap.json`
  (step_d.sh does this when profiled).
- prefill: `pf_fam.py <prof dir> pf_model_rows_L24.json` (kernel-name families vs model achievable).
