# solmodel: Kimi K3.1-Pro on VR200, first-principles speed-of-light model

Op-level and layer-level SOL model of every K3.1-Pro op (DSpark draft included) under each prefill and decode sharding
strategy, including the communication each strategy adds. Prefill and decode are modelled separately and can be paired
freely through NIXL. Both serving checkpoints (`mxfp4_release`, `nvfp4_modelopt`) are evaluated everywhere. The page is
`k31vr/sol.html`.

```
python3 k31vr/solmodel/run.py      # writes k31vr/solmodel/out/{prefill,decode,optables,meta}.json (~5 s, stdlib only)
python3 k31vr/solmodel/render.py   # inlines out/*.json into sol_template.html -> k31vr/sol.html
```

Python 3.9+, no third-party packages.

## Files

| file | role |
|---|---|
| `inputs/ops.json` | op inventory, 62 ops: FLOPs and byte formulas of the full op (`gen_ops.py`, cites into the serving code) |
| `inputs/hw.json` | VR200 constants: public spec plus our Hecate measurements (`hw_src/build_hw.py`) |
| `inputs/comm.json`, `inputs/topologies.md` | communication primitives P1-P21 and per-topology mechanics |
| `inputs/workload.json` | AgentX workload, KV and state bytes, weight bytes by family |
| `inputs/checkpoints.json` | the two serving checkpoints |
| `proposals.md` | strategy list P1-P9 and D1-D11 with overlays O1-O7 (the model implements these) |
| `hw.py` | two hardware columns: `sol` (spec peaks) and `achievable` (measured peaks) |
| `model.py` | compiles the ops.json formulas; checkpoint dtype and bytes per op family; step symbol environments |
| `topologies.py` | each strategy as modes; generic per-op sharding divisors (`op_map`); per-layer comm events (`comm_events`) |
| `cost.py` | per-op roofline, comm pricing, layer and step assembly, resident weights |
| `kv.py` | KV and state capacity, r<sub>max</sub> per GPU |
| `prefill.py` | chunk time, fresh tok/s/GPU, F x L scans, TTFT, NIXL bytes |
| `decode.py` | step time vs requests per GPU for k = 3/5/7, the AFD pool model, pareto curves |
| `run.py` | runs everything and writes `out/*.json` |
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
t_op      = max(FLOPs_gpu / peak[dtype(op, checkpoint)], bytes_gpu / HBM)       bound = whichever term wins
```

The divisors come from `topologies.op_map`. A head- or column-sharded op is (N, N). A token-sharded op with replicated
weights is (N, 1). Replicated compute is (1, 1). Mixed ops are weighted by the share that is sharded:
- the packed q_b|wq_index GEMM is 82% head-sharded, 18% replicated;
- KDA in_proj keeps f_a replicated.

Three special cases:
- **Padded heads.** Sparse MLA on replicated DSA pays `pad = max(96/N, 64)/96 * N`.
- **Routed experts.** Each GPU streams the experts it holds that the step touches:
  `E_ACT = (896/E) * (1 - (1 - 16/896)^(rows in the EP domain))`. With TP experts (D1) every GPU holds 1/N of every
  expert. Expert FLOPs per GPU are the group FLOPs / N, multiplied by the imbalance factor (1.0).
- **Markov head.** The Markov w1/w2 vocabulary shard is read once per step when it fits in L2 (64 MiB or less).

**Per communication event** (`topologies.comm_events`, `cost.comm_time`):

```
t = hops * hop_latency + max(bytes_in_per_GPU, bytes_out_per_GPU) / NVLink_per_direction
```

All-reduce and reduce-scatter take the cheaper of their implemented primitives:
- NVLS (P1/P3): 2 hops;
- one-shot or push (P1c/P3b): 1 hop.

The events below are built per layer type, with bytes from comm.json:

| layer / step part | events |
|---|---|
| attention, TP | AR [T,h] |
| attention, SP | AG + RS |
| DSA-DP | push_in + push_out |
| DCP decode | q publish P6, top-k merge P8, value push P7b |
| ShardQ prefill | context push P9 (copy engine, async), q scatter P4, out gather P5 |
| MoE | P14/P15 (TP-MoE), AR (token shard), AG+RS (TP-sharded MoE dense), EP dispatch P11 + combine P12 |
| drafter | 11 ArNorm sites, k Markov exchanges, ctx AG |
| final / verify | sharded verify P16 |
| per step | DP sync P19 |

Overlay O2 (dedup) scales EP bytes by U/16 with `U = E(1 - (1 - 1/E)^16)`.

**Layer and step** (`cost.step`):

```
t_layer = sum t_op(attnres + attention + mlp) + sync comm + max(0, async comm - attention compute) + kernels * gap
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
| NVLink per direction | 1.8 TB/s | 1.25 TB/s ingress |
| hop / per-kernel gap | 0.78 us / 0 | 2.86 us / 0.78 us |

Neither column is fitted to e2e or step measurements. The anchors in `out/meta.json` and on the page are sanity checks
only.

## Main assumptions

See `run.py` (`ASSUMPTIONS`) or section (f) of the page. The most consequential:
- **State.** SOL minimal state: one FP32 KDA state per request and FP32 conv. The deployed allocator reserves 4-22x more.
- **KV row.** The FP8 MLA KV row is 576 B.
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
| `DSA.prologue/idx_q_quant/idx_score/topk` | MXFP4 MMA + 68 B/token key scan; exact top-2048 | indexer score + top-k bench per mode (DCP key shard, DSA-DP owner, ShardQ rows) |
| `DSA.w_uk/sparse_mla/w_uv/gate_mul` | BF16 MMA over 2048 FP8 rows x 576 B | FlashMLA sparse / CuTe sparse MLA at heads per GPU (and padded 64) |
| `MOE.dispatch/fc1/act/fc2/combine` | touched-expert weight stream + FP8/FP4 MMA + P11/P12 | newest packaged MegaMoE (the other session's build) at EP size and rows per rank; time dispatch, GEMMs and combine separately |
| `L.attnres_pre/post`, `F.attnres_final`, `DS.aux_capture` | memory-bound mixes of up to 9 BF16 sources | AttnRes mix kernel bench (rows x sources) |
| `DS.attn/markov/lm_head/confidence`, `F.rejection` | SWA GQA 2048 window; rank-512 Markov per position over vocab/N | DSpark draft step bench (sdec) at k and B |
| comm P1/P1c/P2/P3/P3b/P4/P5/P6/P7b/P8/P9/P13/P14/P16/P19 | hops x hop + bytes / NVLink | symmbench / ncclbench at the message sizes in the op-table comm rows |
| one layer of each type; the full step | per-layer timeline (`per_layer_sol/ach` in optables) | `vllm bench forward` per layer type and full step (model-microbenchmark) |

Existing harnesses to reuse first: the JHB k31-ws sol/sol2/opbreak/breakdown/pr130bench benches and the release decode
step bench (C512 hybrid: B4 21.4 ms ... B64 31.7 ms).
