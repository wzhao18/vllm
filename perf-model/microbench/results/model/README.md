# Lane `model`: SOL model re-scoped to EP <= 16 (2026-11-18)

No GPU work in this lane. The model was re-run with new EP16 topologies; `targets.json` holds the per-op model
times that the measuring lanes divide by (`derate = measured_us / t_ach_us`).

Regenerate (stdlib only, ~12 s):

```
python3 perf-model/run.py && python3 perf-model/render.py && python3 perf-model/test_solmodel.py
python3 perf-model/microbench/results/model/make_targets.py      # -> targets.json
```

Tests: 10/10 pass (new `test_ep16_scope`: scope membership, <= 16 GPUs, two MoE-dense transposes + EP dispatch on
D7-4-16 / D8-16, 56 expert slots per GPU at EP16). One anchor warning was already there and is unchanged: the C512 P2
prefill marginal slope against an 11 ms/1K lower-bound fit.

## What changed (perf-model/topologies.py, run.py, test_solmodel.py)

* Scope rule `topologies.in_scope`: EP <= 16 **and** <= 16 GPUs per engine or EP domain (`EP_MAX = 16`, 4 VR200 nodes).
  D1 (MoE-TP16, no EP) is in scope. Dropped: P3-32/72, P6 (EP64), P7 (PP4 = 64 GPUs), D4-32/72, D7-x (EP64), D8 (EP64),
  D9 (EP64), D10, D11 (AFD, 72 GPUs). These stay in the model and on the page, but nothing below ranks them.
* New decode topologies:
  * `D7-4-16`, `D7-2-16`, `D7-8-16`: layer-typed wide-EP with EP16. TP_a attention is TP2, TP4 or TP8 with SP
    residual and DSA-DP. MoE-dense is TP-sharded (two transposes and an AG/RS per MoE layer). Experts are EP16, so
    16/TP_a attention groups share one EP domain, with 56 expert slots per GPU.
  * `D8-16`: TP16+DCP16 attention, TP16 MoE-dense (transposes + AR), and EP16 over the same 16 GPUs.
  * `D9-16`: DEP16 + DCP16 on the 24 DSA layers. This is an extra variant and it never wins.
* New prefill topologies: `P6-16` (TP2 KDA + DSA-DP, EP16 + O2 dedup) and `P6-4-16` (the same at TP4). `P9` (DEP16 +
  DCP8 on DSA) was already EP16. `P9-16` (DCP16 on DSA) was added.
* New output keys: `decode.json` has `best_ep16`, `pareto_ep16` and `scope_ep16`. `prefill.json` has `rank_ep16`
  (per `ckpt|col|variant`: [topo, tok/s/GPU, best F]) and `scope_ep16`. `meta.json` has `scope_ep16`. `run.py` prints
  an EP<=16 summary block. `sol_template.html` was not edited (out of lane), so the page shows the new rows in its
  generic tables but its featured lists still name the EP64 ids.

## Decode: best topology per band under EP <= 16 (k=3, L=135K, achievable column)

Bands are by requests per GPU r, with interactivity in tok/s/user. "Loss" compares against the best layout when EP64/72
is allowed.

### mxfp4_release, base (no overlap)

| band | r | best (EP<=16) | tok/s/GPU | tok/s/user | runner-up (margin) | best with EP64/72 | loss |
|---|---|---|---|---|---|---|---|
| min latency | 0.0625-0.25 | D1 | 53-145 | 854-579 | D8-16 / D2 (+13-25%) | D1 | 0% |
| low | 0.5-2 | D1 | 221-558 | 441-279 | D8-16 / D7-8-16 (+4-10%) | D8 / D7-8 | -3 to -30% |
| mid | 3-8 | D7-8-16 | 757-1781 | 252-223 | D7-4-16 / D1 (+0-4%) | D7-8 | -32 to -34% |
| mid-high | 12-16 | D7-4-16 | 2459-3093 | 205-193 | D7-8-16 / D5 (+2%) | D7-4 | -27 to -29% |
| high | 24-48 | D5 / D7-2-16 / D7-4-16 (tie, <1%) | 4035-5554 | 168-116 | | D7-4 / D7-2 | -21 to -23% |
| capacity end | 55 | D7-4-16 (r_max 55.3) | 5846 | 106 | | | |

### mxfp4_release, O1+O2+O3 (two-microbatch overlap + dispatch dedup + die-local streaming)

| band | r | best | tok/s/GPU | tok/s/user | loss vs EP64 |
|---|---|---|---|---|---|
| <= 0.5 | | D1 | 56-252 | 893-505 | 0% |
| 1-2 | | D1 (D8-16 / D7-8-16 within 2-6%) | 398-630 | 398-315 | -13 to -28% |
| 3-6 | | D7-8-16 | 872-1607 | 291-268 | -30 to -31% |
| 8-12 | | D7-4-16 | 2040-2883 | 255-240 | -28% |
| 16-32 | | D5 (D7-4-16 / D7-2-16 within 0.5-1.5%) | 3599-5680 | 225-178 | -30 to -33% |
| 48-60 | | D7-4-16 to D7-8-16 (capacity end r 60.3) | 6442-7239 | 134-120 | -35% |

### nvfp4_modelopt

* Base: D1 for r <= 2, D7-8-16 for r 3-4, D3 for r 6-8, and D5 for r 12-48 (D3, D7-2-16 within 1-3%). At the capacity
  end D7-2-16 reaches 6824 tok/s/GPU at r 50.5.
* O1+O2+O3: D1 for r <= 1.5, D6 for r 2-4, and D5 for r 6-48 (8060 tok/s/GPU at r 48, 168 tok/s/user).

The r=64 grid point appears only for D1 / D8-16 (r_max 66). It sits below the r=48 point and is not on the frontier.

**Why EP16 costs about 30% at r >= 1.** At EP16 every GPU holds 56 expert slots, and at r >= 1 almost all of them are
touched each step. The routed-expert weight stream therefore dominates: at r=8 it is 6.59 ms of the step for every
EP16 layout (D7-8-16, D5, D1, D8-16), against 1.72 ms for D7-8 at EP64. The layouts differ only by attention and
communication terms of about 1-3 ms. As a result the mid and high bands are near-ties (often < 2% margin). Pick among
them on measured attention and communication derating, not on the model.

Capacity at L=135K (r_max per GPU, mxfp4 / nvfp4):

| D1 | D8-16 | D7-8-16 | D7-4-16 | D7-2-16 | D3 | D5 | D4-16 | D9-16 | D2 | D6 |
|---|---|---|---|---|---|---|---|---|---|---|
| 66.4/65.4 | 66.4/65.4 | 60.3/60.3 | 55.3/57.2 | 45.2/51.0 | 47.8/53.1 | 45.0/50.2 | 24.9/38.5 | 24.9/38.5 | 26.1/24.0 | 14.3/15.8 |

## Prefill ranking under EP <= 16 (L_ref 78.4K, best chunk per engine; every winner picks F = 64K)

Fresh tok/s/GPU, achievable column (SOL in parentheses):

| rank | mxfp4 base | mxfp4 O1+O2 | nvfp4 base | nvfp4 O1+O2 |
|---|---|---|---|---|
| 1 | P6-16 17.8K (22.0K) | P3-16 = P9 = P9-16 21.5K (25.8K) | P6-16 28.7K (39.6K) | P3-16 = P9 = P9-16 37.8K (51.6K) |
| 2 | P3-16 = P9 = P9-16 17.1K | P4-2 = P6-16 21.5K | P3-16 = P9 = P9-16 27.3K | P4-2 = P6-16 37.7K |
| 3 | P6-4-16 16.8K | P4-4 = P6-4-16 21.1K | P6-4-16 26.1K | P4-4 = P6-4-16 36.6K |
| 4 | P4-2 16.6K | P2 20.7K | P4-2 26.0K | P2 35.4K |
| 5 | P4-4 15.6K | P2b 19.5K | P4-4 23.8K | P2b 32.3K |
| 6 | P2 15.0K | P5-4 17.8K | P2 22.5K | P5-4 27.8K |
| last | P1 9.8K, P5-8 8.3K | P1 16.2K, P5-8 12.5K | P1 12.5K, P5-8 10.2K | P1 23.7K, P5-8 16.5K |

Reading:

* Without overlap, P6-16 (the EP16 version of the EP64 headline layout) is the best EP<=16 prefill. It leads
  P3-16 / P9 by 4% and P4-2 by 8%. The gain over P4-2 comes from O2 dedup on the EP16 dispatch/combine (step comm
  80 ms vs 115 ms per 16K chunk on a TP2 engine).
* With O1+O2, every DEP-like or DSA-DP layout (P3-16, P9, P9-16, P4-2, P6-16) converges to within 0.3% of 21.5K
  (mxfp4) / 37.8K (nvfp4). Communication hides under compute, and the step is compute-bound on KDA projections,
  experts and shared experts. The decision then rests on TTFT, KV fan-in to the decode layout, and measured derating.
* P9 and P9-16 tie P3-16 exactly. DSA-CP's value is balance of long-context DSA work across GPUs, which the SOL model
  (perfect balance, async context gather hidden) does not price. This is a model limitation, not a finding that
  DSA-CP is useless.
* P1 (TP8+DCP8, the deployed layout) is last but one in every column. Per-layer AR on 16K rows is the cost.

## targets.json (for the measuring lanes)

`targets.json` is keyed `ckpt|topology|band`, with bands `prefill_F16K_L78K`, `decode_r1`, `decode_r8` and
`decode_r32`. It covers all 23 in-scope topologies x 2 checkpoints (84 tables; r above r_max is absent, for example
D4-16, D9-16, D2 and D6 at r32). Each table has:

* `step_sol_ms` and `step_ach_ms`.
* `ops`: one entry per op, with `op_id` (the model op id from `inputs/ops.json`), `fam`, `count` (instances per
  step), `dtype`, `t_sol_us`, `t_ach_us` (**per instance**), `bound_ach`, and the per-GPU flops / bytes.
* `comm`: one entry per event, with `layer`, `prim`, `name`, bytes in/out, `hops`, `t_sol_us`, `t_ach_us` and
  `async_`.

Shapes follow `topologies.op_map`. Decode r means requests per GPU, so a group of N holds r x N requests with (1+k)
rows each. Prefill F=16384 is per engine (N GPUs). The source is `out/optables.json` from this run; the same numbers
appear on the page.

`rows.json` is an empty list because this lane measured nothing.

Versions: perf-model at k31vr-dev `wzhao/k31-agentx-dev` working tree (uncommitted edits to topologies.py, run.py and
test_solmodel.py). Inputs (`inputs/*.json`) are unchanged. HW columns are public VR200 spec (SOL) and Hecate
measurements (achievable), with no fitted derating.
