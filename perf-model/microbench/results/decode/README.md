# Lane decode: decode step time vs requests per GPU (2026-11-18)

## What ran

Whole decode step, measured with the nq step bench (`vllm bench forward`, one CUDA graph per case). The step is the target
verify over 93 layers plus the DSpark k-step propose. The bench leaves out the LM head/logits, rejection and CPU
overhead, so the derate against the model's `t_ach` is a lower bound of about 0.1 ms. Only the specialized decode (ls) arm ran.

- Code: overlay `$W/dsrel/ov/rel-1767cf7`, the release draft sweep tree. Its base is dsrel-base-6a14e2f (`SOURCE_REV`
  6a14e2fada), with the rel-1767cf7 files shipped over it. Image `vllm-openai-cu134-nightly-3f1bcfd82b-aarch64.sqsh`.
  FlashInfer 0.7.3.dev20261031 (nq env). Drafter: release DSpark `jh/noquant/cfg2/deps/kimi-k3.1-pro-dspark-e4e6e626`
  (GQA, FLASH_ATTN draft backend).
- MoE: D1 and D2 use the specialized decode's own CuTe Rubin MoE (MoE-TP). D3 EP16 uses the vLLM-vendored MegaMoE
  (`megamoe_rubin`: `deep_gemm::sm100_fp8_fp4_mega_moe_situ_impl`). The newest FlashInfer MegaMoE 6e438852b was NOT
  used: the step bench has no hook for it.
- Precision (logged in the cfg.sh files): fp8 MLA KV, mxfp4 indexer KV, fp32 KDA state, BF16 linears, MoE activation
  mxfp8 (`quantization-config {"moe":{"activation":"mxfp8"}}`), and exact top-k (deep_select / topk_select). Weights
  are dummy (timing only). Model config view: `e2e/model-view/kimi-k3.1-pro-k3type`.
- Contexts: `u132k`, every request at 135,168 tokens, which matches the model's L = 135K. Five `reuse_int5` rows come
  from the existing `dsrel/results/int5` relfa runs with AgentX contexts (mean 139-196K) and were used as a cross-check.
  They agree within 1-4%.
- Harness: a lane copy of the dsrel tools in `$W/mb/decode/tools`. The copy sets `cfg2bench.py` WORLD and the
  `nq_run1.sh` node count from `NQ_WORLD_REQ` (8 = 2 nodes), and adds `dec_common.sh` (drun/dprof) and `mkcfg.py`.
  New point configs are in `$W/mb/decode/cfg/*.json`, built from the nq snapshots:
  - `d2k7tp` / `d2k3tp`: TP8 = DCP8 with MoE-TP8, from c64 / c256, with role env `VLLM_PACKED_KV_STATE_GROUPS=4`.
  - `d2k7` / `d2k3`: EP8.
  - `d4k3`: DEP16.
- Profiled runs: torch profiler, parsed with `hti/kb/tools/kb_parse.py` into `breakdown.txt` and `kb_breakdown.json`
  under `results/prof*/<tag>/`.

Queue tasks are in `$W/mb/decode/tasks/`. Results are in `$W/mb/decode/results/<task>/` (`status.txt`, `<tag>.log`,
`.cfg.sh`, `.shapes.txt`; profiles under `prof*/`).

| task | runner job / hosts | content |
|---|---|---|
| 20-nq4-mbdecode-d1k3 | 1137221 hecate0073-0076 | D1 k3 B1-B128 (d1k3lo). The hi run (B256/B512) is over the bench KV capacity |
| 20-nq4-mbdecode-d1k7 | 1137223 hecate0221,0222,0224,0225 | D1 k7 B1-B16 (d1k7lo). The hi run is over KV capacity |
| 19-nq4-mbdecode-d1hi2 | 1137216 hecate0046-0049 | D1 k7 B64/B128. D1 k3 B192/B240 and D1 k7 B144 OOM on the bench state snapshot |
| 20-nq4-mbdecode-d3 | 1137216 hecate0046-0049 | D3 k3 B1-B32. B64+ OOM with the pool sized for B128. D3 k7 is refused |
| 19-nq4-mbdecode-d3hi | 1137218 hecate0370-0373 | D3 k3 B48/B64/B96, one process each. B128 OOMs (snapshot) |
| 21-nq4-mbdecode-d2 / d2tp / 19-d2tpb / 19-d2c | 1137240, 1137217 | D2 EP8 is refused. MoE-TP8 needs PACKED_KV_STATE_GROUPS=4. Final D2 rows: d2k3c, d2k7tplo, d2k7c40/c48 |
| 22-nq4-mbdecode-prof1 / 21-prof1b / 22-prof3 | 1137223, 1137217, 1137222 | profiles: D1 k7 B1/B16/B128, D1 k3 B16/B128, D3 k3 B4/B32/B64 |

## Step time vs model (`out/decode.json`, mxfp4_release, base, achievable)

Here r = requests per GPU (B per DP rank / TP), and derate = measured / t_ach.

| topo | k | r | measured ms | t_ach ms | t_sol ms | derate |
|---|---|---|---|---|---|---|
| D1 TP16=DCP16 | 3 | 0.0625 / 0.125 / 0.25 | 5.18 / 5.11 / 6.83 | 3.61 / 4.21 / 5.32 | 2.21 / 2.60 / 3.35 | 1.44 / 1.21 / 1.28 |
| D1 | 3 | 1 / 4 / 8 | 11.08 / 18.28 / 25.09 | 8.95 / 13.08 / 16.50 | 5.83 / 8.43 / 10.61 | 1.24 / 1.40 / 1.52 |
| D1 | 7 | 0.0625 / 0.125 / 0.25 / 1 | 4.87 / 6.66 / 8.75 / 15.02 | 4.22 / 5.34 / 7.00 / 10.97 | 2.61 / 3.36 / 4.52 / 7.11 | 1.15 / 1.25 / 1.25 / 1.37 |
| D1 | 7 | 4 / 8 | 24.77 / **51.17** | 16.22 / 22.86 | 10.50 / 14.64 | 1.53 / **2.24 KERNEL-DEBUG** |
| D2 TP8=DCP8 (MoE-TP8, not the model's EP8) | 3 | 0.125 / 0.25 / 0.5 / 1 | 7.32 / 10.76 / 11.10 / 15.48 | 5.09 / 6.01 / 7.68 / 10.45 | 3.44 / 4.04 / 5.13 / 6.94 | 1.44 / 1.79 / 1.45 / 1.48 |
| D2 | 3 | 4 / 8 / 9 | 33.30 / 38.86 / 42.66 | 17.78 / 20.54 / 21.06 | 11.75 / 13.67 / 14.04 | 1.87 / 1.89 / **2.03** |
| D2 | 7 | 0.125 / 0.25 / 0.5 / 1 | 9.37 / 10.94 / 15.45 / 20.96 | 6.03 / 7.70 / 10.45 / 14.16 | 4.05 / 5.14 / 6.94 / 9.46 | 1.55 / 1.42 / 1.48 / 1.48 |
| D2 | 7 | 4 / 5 / 6 | 38.51 / 47.12 / 49.50 | 20.25 / 21.21 / 22.18 | 13.55 / 14.26 / 14.97 | 1.90 / **2.22 / 2.23** |
| D3 DP4x(TP4=DCP4) EP16 | 3 | 0.25 / 0.5 / 1 / 2 | 15.57 / 17.57 / 21.11 / 22.33 | 8.16 / 9.41 / 11.09 / 12.71 | 5.37 / 6.18 / 7.27 / 8.32 | 1.91 / 1.87 / 1.90 / 1.76 |
| D3 | 3 | 4 / 8 / 12 / 16 / 24 | 24.88 / 24.16 / 27.22 / 30.30 / 33.34 | 13.74 / 14.70 / 15.60 / 16.51 / 18.84 | 9.01 / 9.69 / 10.34 / 11.28 / 13.18 | 1.81 / 1.64 / 1.75 / 1.84 / 1.77 |

Model `best` ranking restricted to the measured D1-D3. The measured order equals the model order at every r measured:
- r <= 4: D1 wins. At r = 4 the measured order is D1 18.3 < D3 24.9 < D2 33.3 ms; the model gives 13.1 < 13.7 < 17.8.
- r = 8: D3 wins narrowly (24.2 vs D1 25.1 ms; the model gives 14.7 vs 16.5).
- r >= 12: only D3 is measurable here.
- Every gap is about 1.2-1.9x larger in absolute ms.
- The model's global best (D7/D8 at EP64, and D4-72) is out of scope: EP <= 16. It was not re-ranked here.

## Per-op derating (profiled; `rows.json` kind=op)

| family | D1 r8 (k3) | D3 r1 / r8 (k3) | first hypothesis |
|---|---|---|---|
| routed experts (+dispatch/combine) | 1.19 (sdec MoE kernel only) | 1.56 / 1.31 (MegaMoE situ) | near model; vendored MegaMoE, not the 6e438852b build |
| BF16 projections (KDA/MLA/shared/latent/dense) | sdec CuTe; o-proj+AR 1.26, attn_res 2.02 | **2.65 / 2.35** | D3 SP path runs cuBLAS nvjet split-K (128x8/128x40 tiles, M = 16-128) at about 40% of the weight-stream roofline. The specialized CuTe GEMMs D1 uses are not wired for SP |
| KDA core | **3.44** | **9.7 / 2.67** | fixed per-layer latency: 69 launches of about 10-25 us, not state bytes. The model has 76 us total at r1 |
| sparse MLA (MlaQuery + FMHA + index) | **5.1 (MlaQuery) / 10.3 (FMHA)** | **114 / 23** | the model's dsa_attn (10-80 us per step) has no per-layer kernel floor. MlaQuery (W_UK + q path) is 40-80 us per layer. On D3, DsaOverlap QueryWait adds 526 us of waiting |
| indexer + top-k | **4.6** | **5.5** (r8) | paged_mqa_logits at 16-20 us per layer, latency-floored |
| DCP exchange | 1.25 | **4.0 / 3.9** | `k31dcpx::topk_merge_kernel<16>` takes 1185 us at D3 r8, against P8 = 152 us in the model. 49 us per layer for a 4-way merge |
| SP AG/RS | - | **2.3 / 1.59** | sdec SpAllGather/SpReduceScatter at about 6.5 us per call x 186 |

D1 kernels overlap across streams (busy sum 35.4 ms against a 25.1 ms span at r8; at r1 the MoE kernel spans 98% of
the step). D1 per-kernel times therefore include waits inside fused comm kernels and are upper bounds. D3 overlap is
small (busy sum / span about 1.23).

### KERNEL-DEBUG flags (derate > 2)
1. **D1 k7 r8 step (2.24): a step cliff.** B64 takes 24.8 ms and B128 takes 51.2 ms. At 1024 verify rows the MoE leaves
   the sdec Rubin path for stock trtllm-gen `bmm_*MxE2m1MxE4m3*` (9.6 + 8.2 ms, plus finalize 1.5 ms and the
   tail/latent/push 7.9 ms). The likely cause is the sdec MoE wide-row table cap (`check_wide`); check the sdec
   coverage line at M=1024. k3 at 512 rows stays on the sdec path (25.1 ms).
2. **D2 r >= 5-9 (2.0-2.2).** D2 runs MoE-TP8 (the specialized decode refuses EP8), while the model assumes EP8
   token-shard. The model's routed-experts term (13 ms) does not represent a TP8 expert shard. Re-model D2 as MoE-TP8,
   or get EP8 supported.
3. **D3 BF16 dense GEMMs (2.35-2.65)**: as in the table above.
4. **D3 / D1 DSA family** (MlaQuery, FMHA, indexer, topk_merge) and **KDA core**: per-layer fixed-latency floors that the
   model omits. These are kernel/latency fixes, plus a model fix: add about 10-40 us per kernel per layer.
5. **D3 overall (1.64-1.91) at every r**: about 6-7 ms of the gap is the cuBLAS GEMMs. The rest is DSA/DCP/KDA latency.

## Capacity finding (affects the model's rmax)

The bench refuses cases above the server's KV capacity: production gpu-memory-utilization 0.9 (0.93 for c512),
1024-token blocks. Blocks needed per request at 135K:

| topology | blocks per request | capacity | max r at 135K | model rmax |
|---|---|---|---|---|
| D1 k3 | 24 | 6061-6414 | about 15.6 | 66.4 |
| D1 k7 | 36 | 5588 | about 9.7 | 64.8 |
| D2 k3 | 36 | 2857 | about 9.9 | 26.1 |
| D2 k7 | 52 | 2779 | about 6.7 | 25.4 |
| D3 k3 | 13 per DP rank | 1738 | about 33 | 47.8 |

Scaling the contexts down barely reduces the per-request block count (D1 k7 B256: 9217 blocks at full context, 8193 at
about 0.6x). Most of the per-request cost is a fixed floor, likely state/packed groups, SWA, and spec slots in the
1024-token block layout, not context. The model's KV capacity (per_req 2.65 GB/16) is 3-7x optimistic for D1/D2 in
this vLLM build.

Separately, the bench-only state snapshot OOMs near capacity: D1 k3 B192, D1 k7 B144, and D3 B128. That limit is a
bench artifact, not production.

## Failed / unrun

- **D4-16 (DEP16)**: not run. The specialized decode refuses EP outside TP4 SP ("does not support expert parallelism
  (except TP4 = DCP4 sequence parallelism over MegaMoE)"). The only alternative is the stock decode backend, which the
  standing rule forbids on VR. The config is ready at `$W/mb/decode/cfg/d4k3.json`.
- **D5 (TP4 DSA-DP, PR #23 `juntianl/k31vr-dsadp-pd`)**: not run. It is a stock-backend DSA-DP decode, under the same
  rule, and the specialized decode refuses EP16 outside SP. The PR's own `vllm bench forward` (TP4, 64K, k7) reports
  DSA-DP vs replicated DSA at +1.1 / +0.1 / -2.5 / -3.7 / -5.0% step for r = 2 / 4 / 8 / 16 / 32. That puts D5 near the
  D3-class TP4 step, with D5's benefit in KV capacity.
- **D2 EP8**: refused by the specialized decode (same EP error). D2 was measured as MoE-TP8.
- **D3 k7**: refused ("sequence parallelism on speculation other than DSpark with 3 draft tokens").
- **D1 r >= 12 (k3) and r >= 9 (k7), D3 r 32**: over the bench KV capacity or OOM on the snapshot (above).


## Review fixes (adversarial review, 2026-11-18)

No row values changed. Checked: D1 r1/r8, D2 r1/r8 and D3 r1/r8 `t_ach_us` / `t_sol_us` equal the current
`out/optables.json` step totals (8.951 / 16.50, 10.45 / 20.54, 11.09 / 14.70 ms); medians match the raw
`[forward]` lines in `$W/mb/decode/results/*/*.log`; D1 logs show the `FLASHINFER_TRTLLM_MXFP4_MXFP8` MoE backend
(MXFP8 activations) and fp8 KV. Caveats to carry with these rows:

- **No numeric check in these runs.** The harness's `graph vs eager rel rms` is 1.4 (D1, D2 at low B) or 0.0 (D2 at
  high B, all D3). `nq_step.py` documents it as saturating with dummy weights, "not a correctness signal". The rows
  are timing only.
- **D2 rows are not a model derate.** They ran MoE-TP8; the model's D2 is EP8. Do not use D2 derates (1.44-2.23) to
  fit or judge the D2 model term.
- **D3 uses the vendored MegaMoE.** The newest FlashInfer 6e438852b build is about 15-20% faster per MoE layer at
  32-128 tokens/rank (moe lane), worth about 1.6-2.5 ms per D3 step (gap512 projection). D3 derates include that.
- **D1 k7 r8 (B128, 1024 rows) cliff** is a MoE path switch (specialized MoE to stock trtllm-gen bmm), not a model
  term; keep it out of trend fits until the cause is confirmed.
