# Lane comm: NVLink communication primitives at the op-table sizes (EP <= 16)

Measured 2026-11-18 on Hecate VR200 (NVL72 rack, 4 nodes = 16 GPUs per task) through the queue/runner2 system.
Final data: run7. Runs 1-6 were harness iterations, listed below.

## What ran

- **Cases.** All 173 in-scope comm cases of `microbench/matrix.json` (family `comm`). Topologies P1, P2, P3-16,
  P4-2, P6, P9, D1, D3, D4-16, D5, D7-2/4/8, D8, for both checkpoints. N <= 16. DEP32/72 and EP64-72 rows were dropped.
  These collapse to 169 distinct (kind, N, bytes, hops) shapes. On top of them there is an alpha/bandwidth grid at
  N = 2/4/8/16: 16 KiB-256 MiB for AR, RS, AG and A2A, plus barrier-only runs.
- **Group layout.** The 16 ranks are split into 16/N groups of N consecutive ranks. All groups run at the same time,
  as DP x TP deployments do. N=4 fits in one node. N=8 and N=16 span nodes (cross-node NVLink in the rack).
- **Timing.** Each time is one CUDA graph of 3-50 calls. The median of 5 replays is divided by the call count, then
  the max is taken over all ranks. NCCL runs eagerly because graph-replayed NCCL a2a hung (see "harness iterations").
- **Kernels.** All are standalone in `src/commbench.py` (torch symmetric memory plus Triton). No vLLM or MegaMoE code
  was imported. Correctness checks pass on every rank at every N:
  - NVLS AR/RS: `multimem.ld_reduce ... acc::f32` (+ `multimem.st`)
  - push AG
  - multicast AG
  - Lamport exchange

| kind | variants (measured = fastest *implementable* one; NCCL / CE are references) |
|---|---|
| AR (M = full tensor) | `nvls_tri` (multimem ld_reduce acc-f32 + multimem.st on own 1/N, flag barrier before/after = gemm_rs_ar epilogue algorithm), `lamport_oneshot` (one-kernel Lamport AG of M + local fp32 sum), `lamport_twoshot` (Lamport scatter + sum + Lamport AG), `torch_multimem/twoshot/oneshot`, `nccl` |
| RS (M = full tensor) | `nvls_tri` (ld_reduce own 1/N), `lamport_sum` (Lamport scatter + local sum), `push_sum` (push + flag barrier + torch.sum), `nccl`, plus `dbg_push_only` / `dbg_sum_only` decomposition |
| AG (bytes = shard) | `push` (= SymmSpAllGather kernel: persistent 16-B stores to every peer, serial peer loop), `push2d` (one program column per peer), `mc` (multimem.st of the shard to the multicast address), `lamport`, `ce`, `nccl` |
| A2A (bytes = per-peer chunk) | `push`, `push2d`, `pull` (remote loads), `lamport`, `ce`, `nccl` |
| barrier | Triton flag barrier (release/acquire sys), torch symm `barrier()` |

**Mapping from model rows to kinds** (`src/post.py`, `rows.json`):

| model primitives | kind | size parameter |
|---|---|---|
| AR rows | AR | `msg_B` |
| RS rows | RS | `msg_B` |
| P2 / P6 / P14-front / P18 (all-gather pattern) | AG | shard = in_B / (N-1) |
| P5 / P7b / P8 / P11 / P12 / P13 / P14-transpose / P16 | A2A | chunk = max(in, out) / (N-1) |
| P9 | `a2a_ce` (copy engine) | |
| P19 | barrier | |

Hops > 1 add (hops - 1) dependent hand-offs: barriers, or Lamport rounds.

Limits of this mapping:
- EP dispatch and combine (P11/P12) are measured as uniform A2A at the exact model bytes. There is no routing
  imbalance and no per-route metadata. The real MegaMoE kernels are the moe lane's job.

`rows.json` fields:
- case, topology, op_ids, measured_us, t_sol_us, t_ach_us
- derate = measured / t_ach
- variant, all_variants_us, notes

`fits.json` holds the per (kind, N, variant) grid: alpha = best time at <= 64 KiB, and bw = fitted slope over the
16-256 MiB points.

## Versions / provenance

- **Image:** `/lustre/fsw/coreai_comparch_inferencex/weizha/k3vr/images/vllm-openai-cu134-nightly-3f1bcfd82b-aarch64.sqsh`
- **Software:** torch 2.16.0.dev20261009+cu134, Triton 3.8.0, NCCL 2.30.7 (torch's), driver 620.70.06
- **Device:** "NVIDIA VR NVL72", 212 SMs
- **NCCL env:** `NCCL_MNNVL_ENABLE=1 NCCL_CUMEM_ENABLE=1 NCCL_NVLS_ENABLE=1`
- **Code under test:** none from the release tree (`rel-1767cf7`) or MegaMoE. Only the standalone kernels above.

**Tasks** (`$W/queue/done/`, logs `<task>.log`). `$W` = `/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr`:

| task | job / hosts | note |
|---|---|---|
| `15-nq4-mbcomm-run1-m50.sh` | 1118808 / hecate0344,0345,0350,0351 | torch barrier; graph-NCCL hang at N16/N8 |
| `14-nq4-mbcomm-run2-m50.sh` | 1137216 / hecate0046-0049 | flag barrier, NCCL eager |
| `14-nq4-mbcomm-run3-m20.sh` | int32 views, push2d | |
| `14-nq4-mbcomm-run4-m20.sh` | 1137240 / hecate0217,0220,0223,0226 | Lamport compile error |
| `14-nq4-mbcomm-run5-m20.sh` | 1137240 | Lamport exchange |
| `14-nq4-mbcomm-run6-m20.sh` | 1137118 / hecate0218,0230,0231,0232 | Lamport AR/RS |
| `14-nq4-mbcomm-run7-m20.sh` | 1137118 / hecate0218,0230,0231,0232 | **final**: per-case Lamport re-arm, RS decomposition |

- **Raw data:** `$W/mbcomm/run7/{rows_N16,rows_N8,rows_N4,rows_N2}.jsonl`, the logs `N*.log`, and `preflight.txt`.
  The preflight confirms that no keepalive or other compute app was on the GPUs at start.
  Local copies are in `raw/`.
- **Pipeline:** `src/post.py run7 cases.json rows.json` -> `rows.json`, `fits.json`.

## Results vs model

Per family (weighted = sum(measured x count) / sum(t_ach x count) over the family's cases):

| primitive | cases | weighted meas/ach | median derate | range | meas/SOL |
|---|---|---|---|---|---|
| AR | 19 | 0.99 | 1.74 | 0.88-2.42 | 2.09 |
| RS | 11 | 2.02 | 2.33 | 1.64-2.70 | 2.72 |
| P2 SP-AG | 13 | 1.33 | 1.63 | 1.21-1.83 | 1.79 |
| P2 drafter ctx AG | 15 | 1.35 | 1.54 | 1.26-2.04 | 1.82 |
| P18 drafter aux AG | 9 | 1.41 | 1.61 | 1.30-1.70 | 1.85 |
| P5 ShardQ out gather | 2 | 1.30 | 1.31 | 1.29-1.31 | 1.83 |
| P6 DCP q publish | 6 | 1.25 | 1.49 | 1.18-1.61 | 1.73 |
| P7b DCP value push | 6 | 1.41 | 1.55 | 1.29-1.80 | 1.88 |
| P8 DCP top-k merge | 6 | 1.43 | 1.63 | 1.29-1.73 | 1.82 |
| P9 ShardQ ctx push (CE) | 2 | 2.22 | 2.52 | 1.86-2.52 | 3.16 |
| P11 EP-dispatch | 14 | 1.25 | 1.37 | 1.22-1.62 | 1.78 |
| P12 EP-combine | 7 | 1.22 | 1.24 | 1.21-1.67 | 1.76 |
| P13 DSA-DP push in/out | 8 | 1.32 | 1.44 | 1.30-1.46 | 1.76 |
| P14 TP-MoE front AG | 3 | 1.34 | 1.45 | 1.25-1.57 | 1.80 |
| P14 TP-MoE transpose | 24 | 1.71 | 1.72 | 1.54-1.80 | 1.87 |
| P16 Markov exchange | 12 | 1.69 | 1.64 | 1.46-2.68 | 1.81 |
| P16 sharded verify | 15 | 1.64 | 1.64 | 1.53-1.74 | 1.76 |
| P19 DP sync | 1 | 1.70 | 1.70 | - | 1.99 |

**By topology.** Comm-only meas/ach, all regimes and checkpoints pooled:

- **Prefill:** P1 1.05, P2 1.32, P3-16 1.22, P4-2 1.30, P6 1.86, P9 1.23
- **Decode:** D1 1.32, D3 1.64, D4-16 1.55, D5 1.66, D7-2 1.95, D7-4 1.80, D7-8 1.70, D8 1.33

Decode is small-message dominated, so its derate is the latency floor, not bandwidth.

### Large-message multimem (NVLS) all-reduce, N = 4/8/16 (was unmeasured)

`nvls_tri` (fastest at >= 16 MiB everywhere). Times in us, with algbw = M / t:

| N | 64 MiB | 128 MiB | 256 MiB | algbw @256 MiB | model t_ach @256 MiB | SOL @256 MiB | NCCL @256 MiB | torch multimem_all_reduce_ @256 MiB |
|---|---|---|---|---|---|---|---|---|
| 2 | 120.6 | 224.1 | 434.6 | 0.62 TB/s | 836 | 229 | 641 | 3453 |
| 4 | 103.3 | 191.8 | 365.6 | 0.73 TB/s | 836 | 192 | 831 | 1775 |
| 8 | 97.6 | 177.3 | 336.0 | 0.80 TB/s | 930 | 174 | 922 | 927 |
| 16 | 101.9 | 177.4 | 328.3 | 0.82 TB/s | 595 | 164 | 593 | 494 |

- **Fitted algbw (16-256 MiB slope):** 0.64 / 0.77 / 0.84 / 0.88 TB/s at N = 2/4/8/16. The intercept is 15-23 us,
  which includes two flag-barrier kernels.
- **Against the model:**
  - The model's achievable NVLS column (NCCL-derived algbw 0.33 / 0.29 / 0.46 TB/s) is 1.8-2.8x too slow.
  - The measured kernel is about 2.0x the NVLS SOL (2 hops + M(1+1/N) / 1.8 TB/s).
- **Real case.** comm-011 (P1, AR N=8, 235 MB) measures 296 us against t_ach 335 and SOL 153.
- **torch `multimem_all_reduce_`.** It launches few CTAs and reaches only 0.08-0.56 TB/s. Do not use it for large
  messages.

### NVLS reduce-scatter and SM push / multicast bandwidth (per-GPU egress at 128-256 MiB)

- **NVLS RS (ld_reduce, N >= 4).** M / t = 1.00-1.06 TB/s. Each GPU's whole tensor goes into the switch, so RS
  costs about 0.73-0.77x the AR, not 0.5x. At N=2, push RS moves half the bytes of NVLS RS; with a fused sum it
  should win.
- **SM push** (push/push2d, AG and A2A, N = 2-16): 0.91-1.04 TB/s.
- **Multicast AG** (`mc`): 0.98 TB/s at N=8, 1.05 TB/s at N=16 (ingress-bound like push). It is only 0.52 at N=2
  and 0.82 at N=4.
- **Remote-load pull:** 0.75-0.84 TB/s.
- **Copy engine** (one memcpy per peer): 0.2-0.8 TB/s.
- **NCCL:** 0.20-0.44 TB/s.
- **Summary.** The 1.25 TB/s ingress used as `nvl` in the achievable column is not reached by any variant. The
  measured ceiling is about 1.0-1.05 TB/s.

### Latency floor (small messages)

| mechanism | N=2 | N=4 | N=8 | N=16 |
|---|---|---|---|---|
| one-kernel Lamport exchange (+1 counter kernel; a2a / AG) | 5.0 / 4.9 | 5.2 / 5.2 | 5.2 / 4.9 | 5.7 / 5.4 us |
| push2d + flag barrier (2 kernels) | 8.0 | 8.3 | 8.2 | 8.1 us |
| production-style serial-peer push + barrier (a2a) | 8.2 | 9.7 | 12.8 | 17.7 us |
| flag barrier alone / torch symm barrier alone | 5.4 / 3.5 | 5.4 / 3.6 | 5.6 / 3.8 | 5.7 / 4.0 us |
| NVLS AR with 2 barriers (`nvls_tri`) | 15.3 | 15.5 | 15.9 | 15.6 us |
| NCCL a2a / AG (eager) | 36 / 31 | 38 / 34 | 40 / 26 | 43 / 48 us |

The model charges hops x 2.86 us (+0.035 us per polled peer) for a push. A standalone exchange costs 5-6 us, so the
median decode-family derate is about 1.6-1.7. In-kernel fused exchanges (sdec Lamport) should sit between the two.

## KERNEL-DEBUG flags (derate > 2.0): 16 cases

1. **RS small: 8 cases.** comm-119/123/164/197/203/210 (D3/D5/D7-x) and comm-137 (D7-8), plus AR small, 5 cases
   (comm-393/467/552/556/561, D7-2/4/8). Measured 7-12 us against t_ach 2.9-5.6. The time is the Lamport exchange
   floor (about 5 us) plus a separate local-sum kernel (2-3.4 us).
   - **Hypothesis:** harness floor, not wire time. A fused exchange+reduce kernel (sdec twoshot / SpReduceScatter
     style) removes one kernel node; the model's single 2.9-us hop is still about 2x optimistic for any standalone
     launch.
2. **comm-112: RS N=2, 235 MB (P4-2, P6).** Measured 262 us against 97.
   - **Decomposition:** push 124 us (0.95 TB/s, fine) plus an unfused `torch.sum` over [2, 58.7M] bf16 of 138 us,
     about 5x the HBM roofline for a column reduction.
   - **Hypothesis:** the separate reduce kernel. A fused owner-sum should give about 150 us (derate about 1.55).
     NVLS ld_reduce (267 us) is no help at N=2.
3. **comm-230: P9 CE context push, N=8, 18.9 MB per peer.** Measured 275 us against 109.
   - Copy engine: one memcpy node per peer, with low CE peer-write bandwidth (0.27-0.75 TB/s) and 4-16 us per copy
     node. SM push takes 147 us.
   - **Hypothesis:** CE issue serialization and CE bandwidth. It runs async in production, so check overlap before
     treating it as critical-path time. comm-158 (N=4) is 1.86.
4. **comm-583 (P16 Markov, N=4, 64 B) and comm-800 (P2 drafter ctx AG, N=16, 44 KB).** Derate 2.0-2.7 at the
   push+barrier floor (about 8 us). In these cases the bench Lamport kernel hits a recurring about-63-us stall, which
   is a bench-kernel issue. Same floor explanation as item 1.

Not flagged but worth a production fix:
- The SymmSpAllGather design (one CTA loops over peers, loading each peer address in turn) costs +10 us at N=16 small
  messages. It also loses about 23% bandwidth at N=16 large (0.77 vs 1.0 TB/s for one program column per peer).
- At N=4 (where it ships today) it is on par.

## Suggested updates to inputs/comm.json / hw.json (achievable column only)

1. **NVLS AR.** Replace the NCCL-derived `nvls_algbw` with the multimem kernel fit: {2: 0.64, 4: 0.77, 8: 0.84,
   16: 0.88} TB/s. Keep alpha about 13 us for fused production kernels (15.6-16.3 us measured with two separate
   barrier kernels).
2. **NVLS RS.** Change `rs` from 0.5 x M / algbw to M / 1.06 TB/s for N >= 4 (measured 1.00-1.06). Use 0.88 TB/s
   at N = 2.
3. **Push bandwidth.** Change `nvl` (achievable per-GPU push ingress/egress) from 1.25 to 1.0 TB/s. All SM push,
   push2d and multicast AG variants top out at 0.91-1.05 TB/s.
4. **Push latency.** Keep 2.86 us per dependent hop inside a fused kernel. Add a standalone-collective floor of
   about 5 us per launched exchange kernel (Lamport, measured 5.0-5.7 us N=2-16), or about 8 us if it needs a
   separate barrier kernel.
5. **P9 (CE context push).** Use about 0.45 TB/s (N=4-8 measured 0.44-0.80) plus about 4-16 us per peer copy, or
   re-price it as SM push (0.95 TB/s).
6. **P11/P12** (uniform A2A at model bytes). Derate 1.22-1.37 at prefill sizes, consistent with item 3. Routing
   imbalance and metadata are not included (moe lane).
7. **Multicast AG.** Model it as an option for N >= 8 (`mc`: 7.8-8 us floor, 1.0-1.05 TB/s). It writes the shard
   once and cuts egress (N-1)x; useful for P2/P6/P18 when egress, not ingress, binds.

## Failed / unrun

- **All 173 in-scope cases were measured.** No case was dropped.
- **Production vLLM kernels were not run:** `gemm_rs_ar` CuTe NVLS, sdec twoshot / SpReduceScatter, and
  `mnnvl_multimem_reduce_scatter`.
  - `benchmarks/kernels/benchmark_multimem_reduce_scatter.py` hard-gates on SM100/103 and TP <= 8.
  - `benchmark_gemm_rs_ar.py` needs a model config.
  - The standalone kernels here implement the same algorithms (ld_reduce acc-f32 + multimem.st; Lamport sentinel
    exchange).
- **Not done:** routing-imbalanced EP dispatch, P10, P20/P21 (P->D, out of scope), and the N=32/72 rows (out of
  scope).
- **Harness iterations:**
  - run1 hung twice in graph-captured NCCL a2a at N=16/8 (20-min step timeouts, no rows). NCCL has run eagerly
    since then.
  - The torch symm barrier was replaced by a flag barrier. Its alone time is 1.8 us higher, but it is graph-safe
    across interleaved kernels.
  - The multimem kernels first read bf16 through int32 indexing (fixed in run3; checks pass).
  - The Lamport kernel first failed to compile on a non-constexpr global (fixed in run5).
  - Lamport slots are re-armed per case since run7.


## Review fixes (adversarial review, 2026-11-18)

No row values changed. Spot checks passed: derate = measured / t_ach on every row, and `t_ach_us` / `t_sol_us` match
the current `out/optables.json` (the model lane's re-scope did not change any per-op value used here).
- comm-011 (AR N=8, 235 MB) matches `raw/rows_N8.jsonl` (nvls_tri 295.4 us, NCCL 796.4 us); comm-112 / comm-230 / comm-001 byte counts are the model's per-GPU egress (in_B) split per peer where noted, and t_ach equals the optables row.
- Caution: P11/P12 EP dispatch/combine rows are standalone uniform all-to-alls. In production they are fused inside MegaMoE and overlap fc1/fc2 (moe lane derate 0.85-1.02 for the whole fused layer at prefill), so do not add the P11/P12 derate on top of the moe lane's numbers.
