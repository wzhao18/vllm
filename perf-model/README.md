# solmodel: Kimi K3.1-Pro on VR200, first-principles speed-of-light model

Op-level and layer-level SOL model of every K3.1-Pro op (DSpark draft included) under each prefill and decode sharding
strategy, including the communication each strategy adds. Prefill and decode are modelled separately and can be paired
freely through NIXL. Both serving checkpoints (`mxfp4_release`, `nvfp4_modelopt`) are evaluated everywhere. The page is
`k31vr/sol.html`.

```
python3 k31vr/solmodel/run.py      # writes k31vr/solmodel/out/{prefill,decode,optables,meta,calibrated}.json (~15 s, stdlib only)
python3 k31vr/solmodel/render.py   # inlines out/*.json into sol_template.html -> k31vr/sol.html
```

Python 3.9+, no third-party packages. `python3 k31vr/solmodel/test_solmodel.py` runs the consistency checks.

## Files

| file | role |
|---|---|
| `inputs/ops.json` | op inventory, 62 ops: FLOPs and byte formulas of the full op (`gen_ops.py`, cites into the serving code) |
| `inputs/hw.json` | VR200 constants: public spec plus our Hecate measurements (`hw_src/build_hw.py`) |
| `inputs/comm.json`, `inputs/topologies.md` | communication primitives P1-P21 and per-topology mechanics |
| `inputs/workload.json` | AgentX workload, KV and state bytes, weight bytes by family |
| `inputs/checkpoints.json` | the two serving checkpoints |
| `inputs/anchors_d1.json` | D1 e2e decode anchors (21 AgentX runs: running batch, k, logged AL, aiperf ITL), built by `inputs/anchors_src/extract_d1.py` from `results/` |
| `test_solmodel.py` | stdlib consistency checks (KV-row/dtype pairing, record bytes vs capacity, anchor mapping, AFD capacity, expert partition, TP-MoE transposes) |
| `proposals.md` | strategy list P1-P9 and D1-D11 with overlays O1-O7 (the model implements these) |
| `hw.py` | two hardware columns: `sol` (spec peaks) and `achievable` (measured peaks) |
| `model.py` | compiles the ops.json formulas; checkpoint dtype and bytes per op family; step symbol environments |
| `topologies.py` | each strategy as modes; generic per-op sharding divisors (`op_map`); per-layer comm events (`comm_events`) |
| `cost.py` | per-op roofline, comm pricing, layer and step assembly, resident weights |
| `kv.py` | KV and state capacity, r<sub>max</sub> per GPU |
| `prefill.py` | chunk time, fresh tok/s/GPU, F x L scans, TTFT, NIXL bytes |
| `decode.py` | step time vs requests per GPU for k = 3/5/7, the AFD pool model, pareto curves |
| `calibrate.py` | calibrated column: achievable x per-path microbench derate (`inputs/derating.json`, analog rules for unmeasured paths) |
| `run.py` | runs everything and writes `out/*.json`; `out/calibrated.json` = sol / achievable / calibrated per topology, per r-band, rankings (<= 32 and <= 16 GPUs) |
| `render.py`, `sol_template.html` | build the self-contained page |
| `screening/` | the earlier screening scripts behind `proposals.md` §4. These are not the SOL model. |

## Equations

**Symbols.** One step runs on one engine:
- an engine is a TP group of N GPUs; N = 1 for DEP and DSA-CP;
- prefill: T = F chunk tokens, packed from requests with f = 5,456 fresh tokens over a prefix P = L - f;
- decode: B = r x N requests and T = B(1+k) verify rows.

`model.env_prefill` and `model.env_decode` build every symbol of ops.json from these: `Lsum`, `Ltk`, `Lu`, `R`, `Tc`, `Tq`, `nseq`, `AL` and the switches.

**Per op, per GPU** (`cost.evaluate`):

```
full      = ops.json formulas (unsharded), with the checkpoint applied
FLOPs_gpu = full.flops / div_row * pad
bytes_gpu = full.w / div_w + (full.a_in + full.a_out) / div_row + full.kv / div_kv + full.state / div_st
            (a_in = a_out = 0 for ops flagged fused_in_sol)
t_op      = max(FLOPs_gpu / peak[dtype(op, checkpoint)], bytes_gpu / HBM, SIMT_gpu / peak_simt + SFU_gpu / peak_mufu)
t_kernel  = max(t_op, per-kernel gap)        (non-fused ops; the gap overlaps the previous kernel under PDL)
```

`simt_flops` / `sfu_ops` are the FP32 SIMT and MUFU work that runs beside the MMA: the indexer epilogue (relu +
head-weight FMA over 32 heads per (row, key)) and the sparse-MLA softmax (one exp per (row, head, key)). Sparse MLA uses
the FP8 MMA rate because the 576-B per-tensor FP8 KV row implies FP8 Q x FP8 KV (`model.sparse_mla_dtype`; a 656-B
fp8_ds_mla row would be BF16).

The divisors come from `topologies.op_map`. A head- or column-sharded op is (N, N). A token-sharded op with replicated
weights is (N, 1). Replicated compute is (1, 1). Mixed ops are weighted by the share that is sharded:
- the packed q_b|wq_index GEMM is 82% head-sharded, 18% replicated;
- KDA in_proj keeps f_a replicated.

Three special cases:
- **Padded heads.** Sparse MLA on replicated DSA pays `pad = max(96/N, 64)/96 * N`.
- **Routed experts.** Each GPU streams the experts it holds that the step touches:
  `E_ACT = ceil(896/E) * (1 - (1 - 16/896)^(rows in the EP domain))` (EP sizes that do not divide 896 hold
  ceil(896/E) slots with redundant replicas). With TP experts (D1) every GPU holds 1/N of every
  expert. Expert FLOPs per GPU are the group FLOPs / N, multiplied by the imbalance factor (1.0).
- **Markov head.** The Markov w1/w2 vocabulary shard is read once per step when it fits in L2 (64 MiB or less).

**Per communication event** (`topologies.comm_events`, `cost.comm_time`):

```
push (Lamport mailbox / symmetric-memory push):  t = hops * push_hop(N) + max(bytes_in_per_GPU, bytes_out_per_GPU) / NVLink
NVLS (multimem):                                   t = alpha(N) + M / algbw(N)     (SOL: 2 hops + M(1+1/N) / NVLink)
```

All-reduce and reduce-scatter take the cheapest of their implemented primitives:
- NVLS multimem (P1/P3);
- one-shot Lamport push (P1c, 1 hop) or two-shot push (P1b, 2 hops); push RS (P3b, 1 hop).

Comm events pay their hop latency only (no extra per-kernel gap). Each step reports its dependent hop count and number of
comm events (`hops`, `n_comm` in out/decode.json and on the page) so the latency floor can be rescaled.

The events below are built per layer type, with bytes from comm.json:

| layer / step part | events |
|---|---|
| attention, TP | AR [T,h] |
| attention, SP | AG + RS |
| DSA-DP | push_in + push_out |
| DCP decode | q publish P6, top-k merge P8, value push P7b |
| ShardQ prefill | context push P9 (copy engine, async), out gather P5 (q_b / W_UK / g slice-local, no q scatter) |
| MoE | P14 front AG + AR (D1 TP experts), AR (token shard), TP-sharded MoE dense: AG (SP) + router/latent col->row transpose + latent row->col transpose + RS/AR, EP dispatch P11 + combine P12 |
| drafter | 11 ArNorm sites, k Markov exchanges, ctx AG, aux AG (SP decode) |
| final / verify | sharded verify P16 |
| per step | DP sync P19 |

Overlay O2 (dedup) scales EP bytes by U/16 with `U = E(1 - (1 - 1/E)^16)`.

**Layer and step** (`cost.step`):

```
t_layer = sum t_kernel(attnres + attention + mlp) + sync comm + max(0, async comm - attention compute)
O1: evaluate at half rows, t_layer = max(compute, comm) + gaps, counted twice   (applied only where it helps)
t_step  = sum over the 93 layers (0-based DSA layers 3,7,...,91,92; layer 0 dense MLP) + embed + final + drafter + sync
```

**Prefill.**
- Throughput: `tok/s/GPU = F / (N * t_step)`; for PP, `F / (S * max stage)`, where stages are a balanced DP partition of the per-layer times plus a boundary push.
- Requests: `req/s/GPU = tok/s/GPU / 5,456`.
- Headline: the best chunk size per engine in 4K-64K at L_ref = 78.4K that fits HBM.
- TTFT: a lower bound, the request runs alone on its engine.
- P8: a TTFT-only term.

**Decode.** `ITL = t_step / AL` and `tok/s/GPU = r * AL / t_step`, for r up to r<sub>max</sub>.
- D11 (AFD): per MoE layer, `2 * max(A half-batch layer, F experts, A<->F transfer)`, using the best of three pool splits.
- Best topology per r: the highest tok/s/GPU among the feasible points.
- Pareto: the envelope of max tok/s/GPU over decreasing interactivity.

**Capacity** (`kv.capacity`):

```
r_max = (272.48 GiB - resident weights - 16 GiB reserve) / bytes_group
bytes_group = DSA KV (15,456 B/token x L x replication) + 434 MB FP32 KDA state + FP32 conv + RecoverSSM + drafter SWA
```

r<sub>max</sub> is per GPU: a group of N holds N x r requests.

## Hardware columns

| | SOL (public spec) | achievable (our measurements) |
|---|---|---|
| BF16 / FP8 / FP4 dense | 4.0 / 17.5 / 35 PF | 3.65 / 10.9 / 21.1 PF |
| FP32 SIMT | 131.6 TF (212 SMs x 128 x 2 x 2.424 GHz, measured attributes) | same |
| HBM | 22 TB/s | 13.9 TB/s (17.1 die-local under O3) |
| NVLink per direction | 1.8 TB/s | 1.06 TB/s (comm lane maximum: push 1.04, multicast AG 1.05, NVLS RS 1.06 TB/s) |
| FP32 SIMT / MUFU exp | 131.6 TF / 8.2 Tops (derived) | same |
| push hop | 2.86 us (no public latency figure; never below our measured one-way flag) | 2.86 us + 0.035 us per polled peer |
| NVLS all-reduce | 2 hops + M(1+1/N) at NVLink | alpha 12.6 / 12.9 / 12.9 us (symmbench), algbw 0.64 / 0.77 / 0.84 / 0.88 TB/s (N = 2 / 4 / 8 / 16; comm lane multimem kernel fit) |
| NVLS reduce-scatter | 2 hops + M at NVLink | alpha + M / 1.06 TB/s (N >= 4), M / 0.88 TB/s at N = 2 (comm lane) |
| launched exchange | hops only | max(hops x hop + bytes / NVLink, floor(N)), floor = fastest measured standalone exchange 3.86 / 4.22 / 4.49 / 4.89 us (N = 2/4/8/16); exchanges fused into compute kernels (sdec TP path) pay the hop only |
| per-kernel gap | 0 | 0.78 us graph node, overlapped under PDL (a kernel costs max(roofline, gap)) |
| per-kernel floor | 0 | 2 us on the grid-dependent kernels (KDA recurrence, indexer, sparse MLA, one per MegaMoE layer): ~4 us measured launched kernel minus the PDL-overlapped launch; bounded by the D1 e2e anchors |
| MegaMoE expert weights | HBM | die-local 17.1 TB/s (megamoe_rubin placement; EP8 T=32 measures > 13.9 TB/s) |

Neither column is fitted to e2e or step measurements. The anchors in `out/meta.json` and on the page are sanity checks
only: D1 e2e decode (21 runs), C512 D3 decode step bench (r = B/4), C512 P2 prefill (computed prompt tokens, plus the
O4 variant and the marginal slope), and PR #20 (24-layer layout ratios). `run.py` prints a WARNING, and the page marks
in red, every anchor where the achievable column is slower than the measurement.

## 2026-11-18 update: <= 32-GPU scope, wave-1 model fixes, calibrated column

- **Scope** (`topologies.SCOPE`): at most 32 GPUs per instance (EP <= 32). EP32 variants D3-32 (DP8 x TP4 SP), D4-32,
  D7-{2,4,8}-32, D8-32, D9-32, P3-32, P6-32, P9-32 join the -16 variants; EP64/72, PP4 and AFD stay modelled but unranked.
- **Model fixes** (`microbench/RESULTS.md` "Model fixes" + adversarial review; physics, not fitted):
  MegaMoE = max(ops, dispatch + combine bytes) + combine drain hop + one expert of the shorter phase (`cost.megamoe_time`,
  a SOL overlap assumption checked against every moe row); P1-type TP prefill runs AttnRes on the 1/N RS shard and
  replaces each sublayer AR by RS (in the o_proj GEMM) + AG (`topologies.tp_reduce`); the hardware rows above; and the
  packed KV block pool in the achievable capacity at the best pinnable block size (`kv.py`).
- **Upper-bound tests**: achievable <= every comm row, MegaMoE row, decode step row and shipped e2e anchor (the C512
  prefill slope fit is allow-listed); achievable KV bytes per request <= the measured deployments.
- **Calibrated column** (`calibrate.py`): decode = experts x expert-kernel family derate (MegaMoE rows for EP, sdec
  MoE-TP16 for D1) + rest x the measured front derate (D1 TP front, D3 SP front); prefill = step derate of the path,
  chunk capped at the largest measured chunk. Points are tagged measured / projected / extrapolated; extrapolated
  points never win a band, and measured-path-only winners are reported alongside. nvfp4_modelopt is an unmeasured
  transfer. `run.py` prints the per-band rankings for <= 32 and <= 16 GPUs.

## Main assumptions

See `run.py` (`ASSUMPTIONS`) or section (f) of the page. The most consequential:
- **State.** SOL minimal state: one FP32 KDA state per request and FP32 conv. The deployed allocator reserves 4-22x more.
- **KV row.** The FP8 MLA KV row is 576 B (per-tensor FP8), so sparse MLA runs FP8 Q x FP8 KV MMAs.
- **Latency.** No public NVLink latency exists: both columns use our measured 2.86 us one-way flag per push hop; NVLS
  uses measured alpha/algbw in the achievable column. Hop counts per step are published for rescaling.
- **Expert activations.**
  - mxfp4 experts run W4A8 at the FP8 rate. O4 (NVFP4 activations) is not in the headline.
  - nvfp4 experts run at the FP4 rate.
  - nvfp4 KDA projections and shared experts run at the FP8 rate.
- **Balance.** Routing is uniform with imbalance 1.0. DSA-DP owners and DP lock-step are perfectly balanced, so D10
  (split indexer) only adds comm in this model.

## Matching to layer microbenchmarks (next stage)

Every number on the page traces to an op row (`out/optables.json`: per-GPU FLOPs, weight, activation, KV and state
bytes, dtype, t_SOL and t_achievable, bound) or a comm row (primitive, bytes in/out, hops). The plan:
1. Benchmark each op family at its exact per-GPU shard shape.
2. Record measured / t_achievable as the per-op derating.
3. Then compare the per-layer timeline and the full step against `vllm bench forward`.

| model ops | model term | microbenchmark |
|---|---|---|
| `KDA.in_proj/f_b_proj/o_proj`, `DSA.*` projections, `MOE.router/latent_*/shared_*`, `DENSE.*`, `F.lm_head`, `DS.qkv/o_proj/gate_up/down/fc/ctx_kv_proj` | GEMM roofline at (M = rows per GPU, N, K, dtype) | kernel-microbenchmark: cuBLAS / CuTe GEMM at the sharded shapes; FP8 (MXFP8) for nvfp4 KDA and shared experts |
| `KDA.conv/gate_beta_norm/core_prefill/core_decode/o_norm` | FP32 SIMT + BF16 MMA, FP32 state bytes / N | FlashKDA chunked prefill bench; sdec KDA recurrence bench (B, q, 96/N heads) |
| `DSA.prologue/idx_q_quant/idx_score/topk` | max(MXFP4 MMA, FP32 epilogue, 68 B/token key scan); exact top-2048 via BF16 scratch or two-pass recompute (cheaper) | DeepGEMM `fp4_mqa_logits` (MMA + epilogue together) + top-k bench per mode (DCP key shard, DSA-DP owner, ShardQ rows) |
| `DSA.w_uk/sparse_mla/w_uv/gate_mul` | FP8 Q x FP8 KV MMA over 2048 rows x 576 B, co-bound with MUFU exp | FlashMLA sparse / CuTe sparse MLA (FP8 query path) at heads per GPU (and padded 64) |
| `MOE.dispatch/fc1/act/fc2/combine` | touched-expert weight stream (ceil(896/EP) slots) + FP8/FP4 MMA + fc1->fc2 intermediate (HBM above 64 MiB/GPU) + P11/P12 | newest packaged MegaMoE (the other session's build) at EP size and rows per rank; time dispatch, GEMMs and combine separately. **mxfp4_release prefill: pin MXFP8 activations** (the packaged Rubin MegaMoE switches to NVFP4 W4A4 above 512 tokens/rank, `megamoe_rubin/experts.py:87`) and assert the logged quant kind, or compare against the O4 row |
| `L.attnres_pre/post`, `F.attnres_final`, `DS.aux_capture` | memory-bound mixes of up to 9 BF16 sources | AttnRes mix kernel bench (rows x sources) |
| `DS.attn/markov/lm_head/confidence`, `F.rejection` | SWA GQA 2048 window; rank-512 Markov per position over vocab/N | DSpark draft step bench (sdec) at k and B |
| comm P1/P1b/P1c/P2/P3/P3b/P5/P6/P7b/P8/P9/P13/P14/P16/P18/P19 | push: hops x push hop(N) + bytes / NVLink; NVLS: alpha(N) + M / algbw(N) | symmbench / ncclbench at the message sizes in the op-table comm rows; large-message (64-256 MiB) multimem AR at N = 4/8/16 is still unmeasured |
| one layer of each type; the full step | per-layer timeline (`per_layer_sol/ach` in optables) | `vllm bench forward` per layer type and full step (model-microbenchmark) |

Existing harnesses to reuse first: the JHB k31-ws sol/sol2/opbreak/breakdown/pr130bench benches and the release decode
step bench (C512 hybrid: B4 21.4 ms ... B64 31.7 ms).

## Review log (2026-11-17)

Three reviews (op-formula, topology/comm, anchors/page). Issue -> fix; numbers are mxfp4_release unless noted.

| # | sev | issue | fix |
|---|---|---|---|
| R1-1 | high | Sparse MLA priced at the BF16 rate although the 576-B per-tensor FP8 row runs FP8 Q x FP8 KV | `DSA.sparse_mla` compute = FP8 MMA (`model.sparse_mla_dtype`: 576 -> fp8, 656 -> bf16, tested); absorbed 2*Nh*(576+512) FLOPs kept; added the MUFU exp term (one exp per (row, head, key), co-bound with FP8 MMA) and the softmax SIMT term |
| R1-2 | med | Indexer per-(row,key) epilogue charged at the FP4 MMA rate | `DSA.idx_score` split: `flops` = FP4 MMA only, `simt_flops` = 4*Ni*Lsum (relu + head-weight FMA, FP32) at the SIMT rate; t = max(mma, epilogue, bytes). Packed BF16x2 noted as lossy, not used |
| R1-3 | low | RecoverSSM record traffic (512 B/row/head) disagreed with capacity (1024 B) | one constant `REC_B` = 1024 B (FP32 correction + BF16 key_gate[2dk]); write q rows, read AL rows; capacity uses the same constant (tested) |
| R1-4 | low | `DS.aux_capture` priced as a full AttnRes mix; `F.attnres_final` on all T rows despite the O7 claim | aux capture = release `prefix_only` add, on Tc rows in prefill; final mix on R rows in prefill under O7 (new `O7` env switch) |
| R1-5 | low | Sparse-MLA KV reuse rule differed between phases | one rule (union bound `min(Lu, Ltk)`) in both phases; the decode union of overlapping verify rows being smaller is documented (~2% of a high-r DEP step) |
| R1-6 | low | Fused binary elementwise ops dropped their second input; MoE intermediate never touched HBM | KDA o_proj reads the o_norm gate, DSA o_proj reads the gate_mul gate; MoE fc1->fc2 intermediate round-trips HBM when > 64 MiB per GPU (`MOE_SPILL`; L2-resident in decode, spilled in prefill) |
| R1-7 | low | Fused exact top-k with no logits scratch assumed free | O7 select = cheaper of BF16 scratch write+read and a two-pass recompute (2x indexer MMA + key reads); the scratch wins at every evaluated point (recorded as `topk_select`) |
| R1-8 | low | MegaMoE switches mxfp4 prefill to NVFP4 activations above 512 tokens/rank | microbench mapping (README, page, assumptions): pin MXFP8 or compare to O4; the P2 anchor shows both |
| R1-9 | low | KDA chunk-serial latency missing (TTFT, few-sequence chunks) | `KDA.core_prefill` = max(roofline, min(serial walk ceil(f/16) x 0.49 us, segscan 3x roofline + walk/S)) per layer |
| R2-1 | high | Achievable comm latency flat 2.86 us for every primitive and N | per-primitive pricing: NVLS AR/RS use measured multimem alpha(N) and NCCL-NVLS algbw(N); push primitives use the flag hop + 0.035 us per polled peer (16-rank Lamport reduce 3.39 us, `twoshot.py`); two-shot push added as an AR alternative. The torch one_shot pull bench (49 us at n16) does not describe sdec's Lamport push, and the D1 e2e anchors bound it: 459 comm events per D1 step at 12.7 us each would exceed the measured 4.5 ms C1 step. Hop counts are published per step |
| R2-2 | high | TP-sharded MoE-dense missed the col->row transpose and the latent return | two transposes per MoE layer for D7/D8/D10/D11 (router+latent col->row, combined latent row->col for a K-split latent_up riding the existing RS/AR); D8's full front AG replaced by the transpose |
| R2-3 | med | SOL hop = one graph node (0.78 us), below any measured hand-off | SOL hop = 2.86 us measured flag (no public latency figure); hop counts published so readers can rescale |
| R2-4 | med | NVLS AR bandwidth at 1.1 TB/s unmeasured | achievable NVLS uses the measured NCCL NVLS 256 MiB algbw (0.29-0.46 TB/s); two-shot push (measured push rate) is the alternative; P1 achievable falls to 9.8K (was 10.8K); large-message multimem remains to be measured (open) |
| R2-5 | low | ShardQ: slice-local q_b with a q scatter (double count) | SOL variant: slice-local q_b / W_UK / g, no P4 scatter; P5 out gather kept for the head-TP o_proj |
| R2-6 | low | Drafter aux AG missing under SP decode | P18 aux AG (Tc x 5 x 14,336 B x (N-1)/N) added when the residual is SP |
| R2-7 | low | DSA-DP decode push charged (N-1)/N rows | decode push bytes = 0 (SP shards hold whole requests); hand-off latency kept; prefill unchanged |
| R2-8 | low | AFD transfer summed both directions; two AFD capacity functions disagreed | per-side max(egress, ingress); one `kv.afd_capacity` (r_max per split, curve uses the largest), tested |
| R2-9 | low | 896 experts over 72 (and 24) GPUs fractional | ceil(896/EP) expert slots per GPU (13 at EP72, 40 redundant replicas carrying load evenly), charged in resident weights and the touched-expert stream |
| R3-1 | high | C512 decode anchor mapped r = B (B is per DP4 rank) | r = B/4; rows beyond r_max flagged infeasible; measured/achievable 1.7-1.9x, measured/SOL 2.6-2.9x |
| R3-2 | high | Achievable column slower than shipped D1 at low r (launch gaps stacked on comm events) | kernel gap overlapped under PDL (max(roofline, gap)), comm events pay hop latency only; 21 D1 e2e anchors added (`inputs/anchors_d1.json`) with a warning when the model is slower: achievable is now 1.06-1.67x faster than measured at every D1 point (C1 4.22 ms vs 4.55 frozen / 4.48 release) |
| R3-3 | med | P2 prefill anchor mislabelled and slope mismatch | relabelled as computed prompt tok/s/GPU (8.3K; fresh ~5.9K), O4 variant added (15.9K achievable), marginal-slope row added (model 16.7 / 13.5 ms per 1K vs the 11 ms/1K lower-bound fit, flagged). Not rebuilt from per-step logs: the C512 prefill logs only have 10-s throughput lines (open) |
| R3-4 | low | PR #20 not used | added as an anchor (24 layers + embed/final/sync, 16K tokens per GPU, PR layouts incl. DEP12 / TP4xDP3 DSA-DP): model/measured 1.18-1.41x; ratios to DEP12 model 0.91 / 0.88 / 0.57 vs measured 0.92 / 0.83 / 0.64 |
| R3-5 | low | Anchor table could not show the D1 agreement | anchor table rebuilt: per-row measured, SOL, achievable, speed ratios, red when achievable < measured |
