# K3.1-Pro on VR200: topology and communication mechanics (SOL model input)

Scope: how each parallelism shards every op of Kimi K3.1-Pro and its DSpark draft, which
communication it adds per layer, and how KV/state capacity per GPU scales. It ends with new
strategies that are not implemented. The machine-readable companion is `comm.json`: one
primitive catalog plus per-topology per-layer comm lists with size formulas.

All statements are taken from the code of `wzhao/k31-agentx-dev` in `/Users/weizha/k31vr-dev`
(paths relative to `vllm/`), from PRs Inferact/nvidia-vllm-agentx #20 and #23 (diffs read in
full), and from the run configs and logs under `results/`. Section 10 lists what is inferred
rather than read.

Hardware constants (NVLink rate, hop latency, NVLS) are in `hw.md` / `hw.json`. This file
prices communication only in bytes and dependent hops. Recommended cost model (`hw.md` §4):
`t = hops x 2.9 us + max(in_B, out_B) / 1.25 TB/s` per GPU, with SOL at 1.8 TB/s.

---

## 0. Notation

| symbol | meaning |
|---|---|
| `h` | hidden size, 7168. A BF16 residual row is `2h` = 14,336 B. |
| `N` | TP group size (heads / columns sharded over it). |
| `W` | DCP size. In every shipped DCP config `W = N`: DCP reuses the TP ranks. |
| `D` | DP size (number of attention groups). |
| `E` | EP size. It is `D x N` whenever MegaMoE runs (all ranks of the instance). |
| `T` | rows of a step on one TP group. Prefill: fresh tokens. Decode: `R = B x q`. |
| `B`, `q` | requests in the group; `q = 1 + k` rows per request with DSpark k. |
| `n` | rows per rank under sequence parallelism, `ceil(T / N)`. |
| `C_r` | context length of request r (prefix + new). `ΣC` is the sum over the step's requests. |
| `o` | the DSA owner card of a request (DSA-DP). `T_o` is the rows owned by card o. |
| `in_B` / `out_B` | bytes received / sent per GPU over NVLink in one collective. |
| `hops` | dependent NVLink handoffs (flag waits) in the collective. |

**Model constants used below.**
- Config: `k31_config.json` and `transformers_utils/configs/kimi_linear.py`.
- Layer indices are 0-based. `is_kda_layer(i)` tests `(i+1) in kda_layers`, so:
  - layer 0 is KDA with the dense MLP (intermediate 33792);
  - DSA layers are 3, 7, ..., 91 and 92 (24 layers);
  - the other 69 layers are KDA;
  - layers 1-92 are MoE (92 layers).
- DSA (`nvidia/dsa.py`):
  - 96 heads, `q_lora` 1536, `kv_lora` 512, nope 128, rope 64, `v_head` 128;
  - output gate `g_proj` (full rank);
  - indexer: 32 heads x 128, top-k 2048.
- KDA (`nvidia/kda.py`): 96 heads x 128, conv 4, full-rank gate (`f_a` 7168->128, `f_b` 128->12288).
- MoE (`nvidia/model.py` `KimiMoE`):
  - 896 experts, top-16;
  - latent 3584, expert intermediate 3072;
  - 2 shared experts, so a shared intermediate of 6144;
  - BF16 router;
  - BF16 latent down (7168->3584) and latent up (3584->7168).
- DSpark draft (Hecate `kimi-k3.1-pro-dspark-e4e6e626/config.json`, `nvidia/dspark_gqa.py`):
  - 5 Qwen3 SWA layers, 96 q / 16 kv heads x 64, window 2048;
  - MLP 14336 (SiLU);
  - `fc` 35840 -> 7168, reading 5 aux layers (76, 80, 84, 88, 92);
  - Markov rank 512, block 7;
  - shares the target's embedding and LM head (vocab 163840).
- KV formats (`v1/attention/ops/dcp_shardq/rank_major.py` docstring: "68 of the 644 bytes per token"):
  - MLA FP8 per-tensor `kv-cache-dtype: fp8` = 576 B per token per DSA layer (512 latent + 64 rope);
  - MXFP4 indexer = 68 B per token per DSA layer (64 B E2M1 + 4 B UE8M0);
  - so 644 B per token per DSA layer, and 15,456 B per token over the 24 layers.

---

## 1. Op inventory by sharding class

Which dimension each op shards on, per layer type. "col" is column-parallel (output sharded),
"row" is row-parallel (input sharded, output partial), "rep" is replicated weight. Sources are
code lines.

### 1.1 DSA layer (`nvidia/dsa.py` `KimiK3DSAAttention`, `KimiK3DSAIndexer`)

| op | shape (full) | TP class | cite |
|---|---|---|---|
| `fused_qkv_a_proj` | 7168 -> 1536 + 576 | **rep** (`disable_tp=True`) | dsa.py:533 |
| `q_a_layernorm`, `kv_a_layernorm` | per token | rep (all rows on every rank) | |
| `q_b_proj` | 1536 -> 96 x 192 | col (heads / N) | dsa.py:544 |
| `kv_b_proj` (absorbed into W_UK / W_UV) | 512 -> 96 x 256 | col (heads / N) | dsa.py:569 |
| `g_proj` (output gate) | 7168 -> 96 x 128 | col | dsa.py:630 |
| `o_proj` | 96 x 128 -> 7168 | row, partial -> AR/RS | dsa.py:576 |
| indexer `wq_index` | 1536 -> 32 x 128 | **rep** | dsa.py:179 |
| indexer `wkv_index` (k 128 + head weights 32) | 7168 -> 160 | **rep** (`disable_tp`) | dsa.py:189 |
| MLA KV insert (FP8 576 B/token) | per token | every rank computes it; written where the slot lives | |
| indexer K insert (MXFP4 68 B/token) | per token | same | |
| indexer scoring | `T x C x 32 heads x 128` | rep under plain TP | |
| top-k 2048 | per row | rep under plain TP | |
| sparse MQA (absorbed, 576-dim q, 512-dim latent out) | `T x 96 x 2048` keys | heads / N | |

- Packed forward (`pack_projections`, dsa.py:773; only when DCP = 1): it packs
  `[q_a | kv_a | wk | weights_proj]` into one GEMM and `[q_b | wq_index]` into another.
- Plain TP therefore replicates per DSA layer:
  - the indexer projections;
  - the indexer scoring;
  - the `fused_qkv_a` GEMM.
- The sparse-MLA kernel also pads few local heads: 24 heads at TP4 and 12 at TP8 are padded to the
  kernel tile of 64/128 heads. See the `shardq.py` docstring: "the sparse-MLA kernel pads its few
  local heads". ShardQ, DCP and DSA-DP all run all 96 heads (padded to 128) on one card instead.

### 1.2 KDA layer (`nvidia/kda.py` `KimiK3DeltaAttention`)

| op | shape | TP class | cite |
|---|---|---|---|
| `in_proj_qkvgfab` | 7168 -> 4 x 12288 + 128 (f_a) + 96 (beta) (+pad) | col on q,k,v,g,beta; **f_a rep** (`replicated_shard_id=4`) | kda.py:684 |
| `f_b_proj` | 128 -> 12288 | col | kda.py:695 |
| `conv1d` (FP32 weights, kernel 4) | 3 x 12288 channels | col (channels / N) | kda.py:709 |
| recurrence (FP32 state 96 x 128 x 128) | per head | heads / N, no comm | |
| `o_norm` (gated RMSNorm), `o_proj` | 12288 -> 7168 | row -> AR/RS | kda.py:932 |

- The TP4 local width is 12448 = 4 x 3072 + 128 + 24 + 8 pad. This matches the `kda_in=...
  cublas 12448x7168` lines in the C512 decode log.
- **KDA is naturally head-parallel.** The state, conv and recurrence shard by head with no
  intra-layer communication.

### 1.3 MoE layer (`nvidia/model.py` `KimiMoE`)

| op | shape | class | cite |
|---|---|---|---|
| router `gate` | 7168 -> 896 (FP32 out) | rep | model.py:685 |
| `routed_expert_down_proj` (latent) | 7168 -> 3584 | **rep** (`ReplicatedLinear`) | model.py:747 |
| `routed_expert_norm` + `routed_expert_up_proj` | 3584 -> 7168 | **rep** | model.py:766 |
| shared experts (2) | 7168 -> 2 x 6144 -> 7168 | col/row under TP; **rep** under SP (`disable_tp=replicate`) | model.py:419 |
| routed experts (MegaMoE) | 896 x (3584 -> 2 x 3072 -> 3584), MXFP4 | EP over E ranks (896/E experts each) | model.py:798 |
| routed experts (TP MoE, no EP) | intermediate 3072 / N | TP (all experts on every rank) | FusedMoE + LatentMoERunner |

- MegaMoE requires EP (model.py:654).
- The Rubin backend picks the activation format per server (`megamoe_rubin/experts.py:145`):
  - MXFP8 activations x MXFP4 weights up to 512 tokens per rank;
  - above that, an exact MXFP4 -> NVFP4 weight conversion with NVFP4 activations.
- The prefill logs show `nvfp4` for both the P/D TP8 prefill (EP8, bucket 2048) and the C512
  prefill (EP16, bucket 8192). Under the "MXFP8 activations" reading of the rules, the SOL model
  should treat the dispatch payload as a parameter (both are in `comm.json`).

### 1.4 Other ops

| op | class | cite |
|---|---|---|
| Embedding | `VocabParallelEmbedding` 163840 / N. Variants: AR of the partial lookup; RS under SP for >= 2048 tokens (`sp_embed`); owner-rank publish in the TP16 specialized decode. | model.py:233, 1548; `specialized_decode/cutedsl/common/embedding.py` |
| AttnRes (`attn_res`) | per-token RMS + dot. Every layer reads the prefix sum of `ceil(layer/12)` block states. | model.py `_pre_attn_norm` / `_post_attn_norm` |
| AttnRes comm | none; under SP it runs on the shard | |
| Final norm + LM head | `ParallelLMHead` vocab / N. Logits are all-gathered (stock) or kept sharded (specialized-decode verify and Markov exchanges). | model.py:2069; `runtime/verify.py`, `runtime/sampled_verify.py` |
| Layer-0 dense MLP | col/row under TP; replicated under SP (`KimiMLP`, `can_shard_sequence_parallel=True` only with the env opt-in) | model.py:379 |

---

## 2. Communication primitives in the code

Every primitive the serving code issues, with per-GPU bytes. `M` is the full tensor size in bytes.

| id | primitive | implementation (cite) | in_B per GPU | out_B per GPU | hops |
|---|---|---|---|---|---|
| P1 | All-reduce, NVLS, fused into the GEMM (GEMM-AR) | `model_executor/kernels/linear/cute_dsl/gemm_rs_ar.py`: `multimem.ld_reduce` (FP32 in switch) + `multimem.st` | ~M | ~M(1 + 1/N) | 2 |
| P1b | All-reduce two-shot Lamport | `specialized_decode/cutedsl/common/twoshot.py` (TP16 o_proj, MoE tail) | 2M(N-1)/N | 2M(N-1)/N | 2 |
| P1c | All-reduce one-shot Lamport (multicast push) | `oproj/prepared.py` B<=8, `draft/allreduce.py` `ArNorm` | (N-1)M | M (multicast) | 1 |
| P2 | SP all-gather (push into peers' symmetric buffer + barrier) | `nvidia/ops/sp_all_gather.py`. 0.98 TB/s at 17.4K rows TP4. Custom IPC AG below its size, NCCL if no symm mem. | (N-1) x n x 2h | (N-1) x n x 2h (or n x 2h with multicast) | 1 + barrier |
| P2b | SP all-gather, decode (multicast Lamport) | `cutedsl/common/sp_comm.py` `SpAllGather` | (N-1) x n x 2h | n x 2h | 1 |
| P3 | SP reduce-scatter (GEMM-RS fused NVLS) | `gemm_rs_ar.py` RS mode, bound in `maybe_init_gemm_rs_ar` (model.py:296) | ~T x 2h / N (NVLS ld_reduce of own rows) | ~T x 2h (N-1)/N | 1-2 |
| P3b | SP reduce-scatter, decode (push to row owner, owner sums FP32) | `sp_comm.py` `SpReduceScatter`, `oproj/sp.py` | (N-1) x n x 2h | (N-1) x n x 2h | 1 |
| P4 | ShardQ query scatter `shard_q` (heads -> token slice) | `v1/attention/ops/shardq.py:349`, push rows + 2 barriers | (T/N)(N-1)(H/N) x d x b | T (N-1)/N (H/N) x d x b | 1 + 2 barriers |
| P5 | ShardQ output gather `unshard_out` | shardq.py:418 | same with `d_v` | same | 1 + 2 barriers |
| P6 | DCP query gather (heads all-gather), direct symm | `v1/attention/ops/dcp.py` `DirectDCPQGatherWorkspace` (1310); sdec `MlaQuery` publish into every rank's q mailbox (`cutedsl/mla/dcp.py`, `dsa/dcp_stage.py`) | R (W-1) (H/W) x 576 x 1 B (FP8) | R (H/W) x 576 (multicast) | 1 |
| P7 | DCP output combine, LSE A2A | `dcp.py:945` `dcp_a2a_lse_reduce` (direct symm `DirectDCPA2AWorkspace`); sdec `DcpValue` | R (W-1)(H/W)(d_out x 2 + lse) | same | 1 |
| P7b | DCP output combine, "proj" | `dsa/dcp_value.py`: W_UV on all gathered heads first, then push `d_v = 128` instead of 512 | R (W-1)(H/W)(128 x 2 + 4) | same | 1 |
| P8 | DCP decode top-k merge, `DcpExchange` (histogram threshold, exact) | `nvidia/dsa_kernels/dcp_exchange/ops.py`, `decode.py` | ~R (W-1) x X B, X ~ 1-2 KB per row (1 round; 2 above the one-round row cap; +tie round) | same | 1-3 |
| P8b | DCP prefill top-k merge, histogram | `dsa_kernels/dcp_prefill_topk_merge/hist_merge.py`: "(W-1) x (3 x 512 + 4 x tie_words) bytes per row" | T (W-1)(1536 + 4 x tie_words) | same | 4-5 |
| P8c | DCP top-k merge, generic (stock) | all-gather of `(score, id)` candidates, "W x K x 8 bytes per row, ~1.9 GB per rank for a 16K-token chunk" (hist_merge docstring) | T x W x 2048 x 8 | T x 2048 x 8 x (W-1) | 1 |
| P9 | DCP ShardQ gathered-context push (prefill) | `v1/attention/ops/dcp_shardq/context.py`, `gather.py` (SM push), `rank_major.py` (copy engine + flag) | ΣC (W-1)/W x 644 + T (W-1)/W x 68 | same | 1 (CE, off critical path) |
| P10 | DCP chunked-context KV gather (dense-MHA / legacy prefill) | `dcp.py` `DirectDCPKVGatherWorkspace` (1466) | C (W-1)/W x 576 | same | 1 |
| P11 | MegaMoE dispatch (`token_in` pull, **per route**) | `megamoe_rubin/src/sources/communication/nvlink_domain/token_comm.py`: "a source sends at most `max_tokens_per_rank x topk` routes to one destination" | routes_in x a, with a = 3696 B (MXFP8: 3584 + 112 SF) or 2016 B (NVFP4: 1792 + 224) | m x 16 x a (E-1)/E | 2 (sizes round + data) |
| P12 | MegaMoE combine (`token_back` push, BF16 per route, then `TopkReduce`) | same; `runtime.py`, `session.py` | m x 16 x 7168 (E-1)/E | routes_in x 7168 | 1 + barrier |
| P13 | DSA-DP `push_in` / `push_out` (prefill, SP) | PR #20 `nvidia/ops/dsa_dp.py`, `shardq._push_rows` + 2 barriers each | owner: T_o x 2h x (non-local fraction) | shard: n x 2h x (non-local fraction) | 1 + 2 barriers |
| P13b | DSA-DP Lamport exchange (decode, FULL graphs) | PR #23 `dsa_dp.py` `_lamport_exchange` (bf16 -0.0 sentinel, no barrier) | segment rows not local x 2h | same | 1 |
| P14 | TP MoE front all-gather (router logits + latent shards, multicast) | `cutedsl/moe/front.py` | R (N-1)/N x (896 x 4 + 3584 x 2) | R/N x 10752 (multicast) | 1 |
| P15 | TP MoE tail (latent AR + RMS + K-split up-proj + shared RS + output multicast) | `cutedsl/moe/tail.py`; stock `latent_moe_runner.py` (2 ARs: latent [T, 3584] + shared [T, 7168]) | ~AR(7168 R) + AR(14336 R) | same | 2-3 |
| P16 | Vocab-sharded verify / Markov exchange | `runtime/sampled_verify.py` (2 rounds of small per-row entries); `cutedsl/draft/markov.py` (u64 keys per position x rank x CTA) | ~R (N-1) x O(100 B) per round | same | 2 (verify); k (Markov, one per draft position) |
| P17 | Logits all-gather (stock) | `ParallelLMHead` + `tensor_model_parallel_gather` | R (N-1)/N x 163840 x 2 | same | 1 |
| P18 | Final SP all-gather with drafter aux packed | model.py:1865: `[hidden \| 5 aux]` = 6h wide (stock path, not the push buffer) | (N-1) x n x 6 x 2h | same | 1 |
| P19 | DP sync (num tokens across DP, MegaMoE bucket) | DP coordinator + `ep_mean_tokens` (`session.py:64`) | tiny | tiny | 1 (CPU/GPU) |
| P20 | NIXL P->D KV (RDMA, not NVLink) | `distributed/kv_transfer/kv_connector/v1/nixl/*` | per request, §7 | | |
| P21 | Mooncake store load/save (RDMA, host/remote) | `.../mooncake/store/worker.py` | per request, §7 | | |

**Notes on P1/P3 bytes.**
- NVLS reduces in the switch, so a rank's traffic is ~M in and ~M out per all-reduce, against
  2M(N-1)/N for a ring or two-shot all-reduce.
- An SP reduce-scatter is half of an all-reduce, and the SP all-gather is the other half. **SP
  never moves fewer bytes than TP's AR; it only moves the MLP / MoE work onto token shards.**

---

## 3. Per-topology mechanics (per layer)

Each subsection states:
- the residual layout;
- per layer type: who computes what and the collectives, with `comm.json` ids;
- what is replicated;
- where the topology runs today.

### 3.1 TP-N, replicated residual (no SP)

Used today:
- by the P/D prefill instance, TP8 + DCP8, DP1, EP8 with MoE token shard (`use_moe_token_shard`,
  model.py:205);
- by the TP16 + DCP16 decode (MoE `auto`, no EP), with the specialized decode.

`use_sequence_parallel` (model.py:1240) is False when DP = 1 without MegaMoE, or when the MoE
token shard is on.

**Per layer:**

| layer | work | comm |
|---|---|---|
| KDA (69) | in_proj col, conv + recurrence on 96/N heads, o_proj row | **AR [T, h]** (P1 GEMM-AR from the prefill sizes; P1b/P1c in the specialized decode) |
| DSA (24) | `fused_qkv_a` rep, q_b/g/kv_b col, o_proj row. Attention depends on DCP / ShardQ (§3.3, §3.4). Without either: KV replicated, indexer replicated, local heads padded. | **AR [T, h]** + DCP / ShardQ comm |
| MoE (92), token shard (P/D prefill, EP = N) | Routed path on this rank's T/N rows: router, latent down, MegaMoE dispatch/combine, latent up. Shared experts TP-sharded on all T rows; the routed rows are added into the shared partial (GEMM-AR epilogue, from 288 tokens: `TOKEN_SHARD_GEMM_AR_MIN_TOKENS`). | **AR [T, h]** (one per MoE layer, model.py:1051) + P11 / P12 on T/N rows |
| MoE (92), plain TP (TP16 decode) | Routed experts sharded on the intermediate (3072/16 = 192 per rank, all 896 experts on every rank). Router + latent down sharded in the specialized front, then gathered. | P14 front AG + P15 tail (≈ AR(7168 R) + AR(14336 R)) |
| layer-0 MLP | col/row | AR [T, h] |
| embedding | vocab / N | AR [T, h] (or owner publish) |
| LM head | vocab / N | logits AG (P17) or sharded verify (P16) |

- **Per-layer sum (MoE layers):** two ARs of `T x 2h` bytes (attention o_proj + MoE), plus
  dispatch/combine on T/N rows.
- **Replicated:**
  - `fused_qkv_a`;
  - indexer projections;
  - KDA `f_a`;
  - the token-shard routed path (none, it is sharded).

### 3.2 TP-N with sequence parallelism (SP)

- **Condition:** EP, TP > 1, PP = 1, and (MegaMoE or DP > 1), and not token shard (model.py:1240).
- **Used by:** the C512 hybrid prefill and decode (DP4 x TP4, EP16); the DSA-DP layouts of PR #20/#23.
- **Residual:** row-sharded `[n, h]` per rank, `n = ceil(T/N)` (`sp_shard` pads to a multiple of N).

**Per KDA / DSA layer** (model.py:1451-1480):
1. AttnRes and pre-norm run on the shard.
2. **AG** `[n, h]` -> `[T, h]` (P2: `kimi_sp_all_gather`, two-slot symmetric buffer).
   - It allocates 896 MiB per GPU at 32K tokens on TP4. In C512 prefill: "symmetric-memory push,
     8192 rows x 7168 per rank, 896 MiB".
3. Attention on all T rows, heads / N.
4. o_proj partial -> **RS** (P3 GEMM-RS, or `sp_reduce_scatter`).

**MoE on the shard:**
- Router, latent down/up and the shared experts all use **replicated weights** on n rows, with no
  TP collective. The weights are replicated: `KimiMLP(disable_tp=replicate)`.
- The shared expert overlaps the MegaMoE launch on a side stream (`_shared_overlap`, model.py:852).
- `VLLM_KIMI_K3_SHARD_SP_SHARED_EXPERT` (default off) instead TP-shards the shared expert with its
  own AG + RS.
- Routed: MegaMoE EP over D x N ranks, on the n shard rows (P11/P12).

**Other ops:**
- **Embedding:** `sp_embed`. For T >= 2048 it is the RS of the partial lookup (half an AR);
  otherwise AR + shard.
- **Final:** AG of `[n, h]` with the 5 aux rows packed (6h wide, P18), or SP-local aux (PR #20,
  `ops/sp_local_aux.py`, eager steps >= 4096 rows). Then the drafter projects its own shard and
  all-gathers `[n, h]` of projected rows.

**Per-layer sum:** AG + RS = one AR-equivalent of `T x 2h` per attention layer. The MoE layer adds
no TP collective.

**Replicated:**
- Weight streaming for the MoE dense parts. Per GPU per MoE layer: router 12.8 MB, latent 102.8 MB
  and shared 264.2 MB, all BF16.
- No duplicated FLOPs.

**The specialized decode SP step** (`runtime/sp_attention.py`) replaces AG -> in_proj -> core ->
o_proj -> RS by `SpComm` Lamport kernels:
- multicast AG (P2b);
- `SpOproj` fused GEMM + push to row owners (P3b), owner FP32 sum.

The capture sizes are `R = 4n`, n = 1..128 requests per DP group (0.25..32 per GPU).

### 3.3 DCP, decode (W = N, KV sharded by token, interleave 1)

**KV layout:**
- Global token t lives on rank `t mod W` at local position `t div W` (`dcp_shardq/gather.py`
  docstring).
- MLA (576 B) and indexer (68 B) rows are both sharded. Each rank holds `C/W` of every context.

**Per DSA layer, specialized decode** (`specialized_decode/cutedsl/dsa/{prepared,dcp_stage,
dcp_value,indexer_stage,wide,sp}.py`, `cutedsl/mla/dcp.py`):
1. **MlaQuery.** Absorbed query of this rank's H/W heads, FP8 per tensor `[R, H/W, 576]`. It is
   multicast into every rank's q mailbox `[R, H, 576]` (P6), with KV insert on the owner slot only.
2. **Indexer.**
   - The indexer query is computed for all R rows (`wq_index` rep, MXFP4 indexer q).
   - Each rank scores its own `C/W` keys, giving bf16 logits `[R, C/W]`.
   - **exact global top-k** via `DcpExchange.topk_merge` (P8), or the #63 one-round merge.
   - The result is filtered to this rank's slots: about `2048/W` keys per row.
3. **Sparse core.** All `H` gathered heads over the local selected keys, giving a BF16 latent
   partial + LSE (trtllm-gen).
4. **DcpValue "proj".**
   - W_UV over **all gathered heads** (replicated W x: each rank projects all 96 heads of its
     partial).
   - The value rows `[R, H/W, 128]` BF16 + LSE are pushed to each head owner (P7b).
   - The owner LSE-merges its H/W heads over W sources (FP32, one BF16 rounding).
   - Then the gate.
   - Non-proj variant: push the 512-wide latent (P7, 4x bytes).
5. o_proj (row) + AR (TP16: P1b two-shot / P1c) or RS (SP, P3b).

**Stock DCP decode** (`MLADCPManager`, dcp.py:1582):
- Steps: direct q-gather (P6), local attention, A2A LSE combine (P7) of `[R, H, 512 + lse]`, then
  head-scattered. Top-k uses the generic merge (P8c) or the hooks.
- **Env:** `VLLM_USE_DIRECT_DCP_A2A`, `VLLM_USE_DIRECT_DCP_Q_GATHER` and
  `VLLM_USE_DIRECT_DCP_KV_GATHER` (all `1` in every shipped config) select the symmetric-memory
  workspaces over NCCL (`envs.py:2465-2472`).
- The drafter / large local steps fall back to NCCL above `get_dcp_workspace_max_num_tokens`.

**Per-row bytes per DSA layer, W = 16, H = 96:**
- q in = 15 x 6 x 576 = 51.8 KB;
- proj out/in = 15 x 6 x 260 = 23.4 KB;
- top-k ~ 15 x 1-2 KB.
- Compare the o_proj AR: two-shot 2 x 14,336 x 15/16 = 26.9 KB.

**Replicated under DCP:**
- the indexer query projection and `fused_qkv_a` (all rows on all ranks);
- W_UV over all heads ("proj");
- the KV row compute: every rank computes the MLA/indexer row of every new token, and only the
  owner stores it.

### 3.4 DCP prefill = DCP ShardQ (gathered context), and TP ShardQ

**DCP ShardQ:**
- **Enable:** `dcp_shardq_configured` (dcp.py:1086) requires `sparse_mla_shard_q`, `1 < DCP == TP`,
  interleave 1, no PCP / HiSparse / ubatching, and a non-consumer instance. The logs show ~100% of
  prefill tokens on ShardQ for TP8 + DCP8 and DP4 x (TP4 + DCP4).
- **Per DSA layer** (`dcp_shardq/context.py` docstring):
  1. **Context push (P9).** Every rank pushes its owned context rows of the step's requests (MLA
     576 B + indexer 68 B per token, plus indexer rows of new tokens) into **every** rank's
     symmetric workspace.
     - `in_B = out_B ≈ ΣC x 644 x (W-1)/W + T x 68 x (W-1)/W`.
     - Rank-major MLA rows move by the **copy engine** on a side stream while the indexer runs;
       a flag (`cuStreamWaitValue32`) gates the sparse attention (`rank_major.py`). No SM time.
  2. Every rank computes and writes **all** new tokens' MLA rows into its own workspace
     (`kv_insert.py`). The kv_a GEMM is replicated.
  3. **Q scatter (P4), proj variant** (`forward_mqa_shardq_proj`,
     `sparse_mla_attention.py:1327`): q before W_UK, BF16 `[T, H/W, 192]` -> `[T/W, H, 192]`. Then
     W_UK, FP8 quant + head pad, sparse MQA over the gathered context with all heads, and W_UV.
  4. **Output gather (P5):** value `[T/W, H, 128]` BF16 -> `[T, H/W, 128]`.
     - Non-proj variant: q is the MQA query `[T, H/W, 576]`, and the latent out is
       `[T/W, H, 512]` BF16.
  5. The indexer scores only this rank's ShardQ rows against the whole gathered context. **No
     top-k merge and no LSE combine.**
  6. Row split (`fold_split`): each request is folded into 2W pieces, and rank r owns pieces r and
     2W-1-r, which balances the causal indexer.
  7. o_proj + AR (TP, P1) or RS (SP, P3).
- **Per-layer bytes, T = 16K fresh tokens, ΣC ≈ 130K (one request), W = 8:**
  - context ≈ 130K x 644 x 7/8 = 73 MB;
  - q = 16K x 12 x 192 x 2 x 7/8 = 64.5 MB;
  - out = 16K x 12 x 128 x 2 x 7/8 = 43 MB;
  - o_proj AR = 16K x 14,336 = 235 MB (NVLS: ~in and out each).
  - **The AR dominates.**
- **Workspace:** ~0.7 GB per 1M tokens of gathered context per rank (fp8; `context.py`
  `workspace_max_tokens`). At least `max_model_len` = 1M tokens.

**TP ShardQ without DCP** (`VLLM_K3_TP_SHARDQ=1`, PR #20 commit 2):
- KV is replicated, so there is no context push.
- Q scatter and out gather are as above; the indexer scores T/N rows.

**Indexer split** (PR #20, default when ShardQ and DSA-DP are off, `layers/sparse_attn_indexer.py`):
each TP rank scores T/N rows, and the ranks all-gather the top-k `[T, 2048]` int32. `in_B = T x
2048 x 4 x (N-1)/N`.

**Legacy DCP prefill (not ShardQ):**
- q gather;
- generic candidate merge P8c (1.9 GB per rank per 16K chunk);
- LSE A2A of the 512-wide latent;
- **attention repeated W times** (`context.py`: "moves gigabytes and repeats the attention work W
  times over").
- Kept only as a fallback.

### 3.5 Hybrid DP x (TP + DCP), EP across DP (C512: DP4 x (TP4 = DCP4 = SP4), EP16)

- **Config:** `results/k31vr-frozen-sweep-20261114/runs/C512-a1093684/agentx/config.yaml`.
  - TP 4, DCP 4, DP 4, interleave 1, `dcp-comm-backend a2a`, `kimi_k3_mega_moe`;
  - P max-num-batched-tokens 32768, D max-num-batched-tokens 8192 with max-num-seqs 128 per rank;
  - fp8 KV + mxfp4 indexer.
- **Layout:** SP inside each TP4 group (§3.2), DCP4 = the TP4 group (§3.3 / §3.4), MegaMoE EP over
  all 16 GPUs.
- **Per layer, one DP group:**
  - KDA: AG + RS on TP4.
  - DSA: AG, DCP4 q-publish + top-k + DcpValue (decode) or ShardQ + context push (prefill), RS.
  - MoE: EP16 dispatch/combine on n rows.
- **Cross-DP:**
  - only MegaMoE plus the DP token-count sync;
  - all EP ranks must launch the same bucket in lock step, padded to the DP-max (`runtime.py`:
    "Lock-step contract").
  - The decode MegaMoE has two buckets: a decode bucket (<= 4096 routes per rank) and the capacity
    bucket.
- **Measured anchors (sanity only):**
  - decode step B4 21.4 -> B64 31.7 ms;
  - prefill ~13K-token steps, step >= 165 ms + 11 ms per 1K tokens;
  - 4.34 req/s per DP4 engine.

### 3.6 DEP (DP attention + EP MoE, TP = 1)

- **Residual:** all rows local.
- **Attention:** the whole layer (all 96 heads, all KDA heads, all projections) on one GPU per
  request. No TP comm.
- **MoE:** MegaMoE EP over all D ranks (P11/P12). Each GPU sends its own m rows.
- **DP padding:** every rank must run the same MegaMoE bucket, so ranks with fewer tokens still
  launch (padding rows masked with `topk_idx = -1`, `staging.py`). Decode needs equal graph
  sizes across DP.
- **Per-GPU weights:**
  - all dense BF16 weights are replicated (~113 GB at TP1);
  - routed experts are 1/E (1446 GB / E);
  - §6.
- **Decode cost character:** every GPU streams the full dense weights every step (no 1/N), so
  small B is weight-bound.
- **Measured (PR #20):** DEP12 prefill 50.0K tok/s/GPU on the 24-layer model, the best measured.

### 3.7 DSA-DP (TP for GEMMs/KDA/MoE, DSA per request on one owner card)

PRs #20 (prefill) and #23 (decode + P/D). Enabled with `VLLM_K3_TP_DSA_DP=1`, for TP > 1 and DCP = 1.

**DSA layer** (PR #20 `model.py` `_run_dsa_dp` / `_dsa_scatter`; `dsa.py` `local_full_heads`):
- q_b, kv_b, g_proj and o_proj use `disable_tp`, so **the DSA weights are replicated on every card**
  (480 MB / layer BF16).
- No reduce on o_proj.
- The owner card computes its requests' rows with all 96 heads:
  - indexer and top-k;
  - sparse MQA;
  - gate;
  - o_proj.
- Other cards skip the request.

**Under SP:**
- `push_in` (P13): each card pushes the rows of its sequence shard to their owner, into the owner's
  `dsadp_in` buffer at the owner's position. Rows already on the owner stay.
- `push_out` (P13): the owner pushes outputs back to the shard owners. Rows nobody owns are zeroed.
- These replace the AG and the RS of the DSA layer. Each row crosses NVLink at most once: "a
  quarter of the all-gather's traffic at TP4".
- **Bytes:**
  - owner in ≈ `T_o x 2h x (N-1)/N`;
  - shard out ≈ `n x 2h x (N-1)/N` (balanced: both ≈ `T/N x 2h x (N-1)/N`);
  - the same again for push_out.
- Each prefill push has 2 symm-mem barriers.

**Without SP:** zeros for non-owned rows + `tensor_model_parallel_all_reduce` `[T, h]`. That is the
same bytes as TP, but the DSA work is de-duplicated.

**Decode** (PR #23):
- FULL CUDA graphs need fixed shapes, so the batch is put in **DEP order**:
  - card r takes the r-th `1/N` segment;
  - each owner's segment is padded to the busiest owner (`dep_order`);
  - graph sizes are multiples of `N x q`.
- **Padding overhead:** `N x max_o(B_o) / B`.
- The exchange is a Lamport push (P13b) of rows not already on their card, with no barrier. It
  works out to ~`(n_local_misplaced) x 2h` each way.
- Per-owner decode cap: `max_num_seqs / N`.

**KDA, MoE, dense:** unchanged TP/SP/EP. Per layer comm becomes:
- KDA: AG + RS (SP);
- DSA: push_in + push_out;
- MoE: EP.

**KV** (PR #20 `v1/core/dsa_dp_regions.py`):
- Global block ids `[0, S)` are a shared pool (KDA state, drafter SWA).
- Region c is `[S + c x D, S + (c+1) x D)` and maps to card c's local pages.
- A request's DSA + indexer KV comes only from its owner's region: **one copy per TP group**.
- Region sizes are static per card, so owner skew wastes capacity.

**Scheduler:**
- per-owner DSA cost budget, chunk cost `n x (1 + (ctx + n/2) / C0)`, C0 = 180,000 (fitted:
  2.1e-3 ms/token + 1.18e-8 ms/token/ctx for 6 DSA layers);
- prompt chunk 4096 (`VLLM_K3_DSADP_CHUNK`);
- owners are chosen by load (PR #23 commit 3).

**Drafter:** runs on every card (TP), with its KV in the shared pool (`gqa_draft.py` `_dsa_dp_kv`).

**P/D** (PR #23):
- P exports `remote_dsa_owner = o`.
- D pins the owner `o x tp_D / tp_P`. The D NIXL worker reads the DSA groups only on the owner card,
  from P rank o. Other groups are as before.
- Supported pairs: TP4 -> TP4, and TP4 -> TP1 (DEP).

**Measured** (24-layer model, AgentX c3072, prefill tok/s/GPU):

| layout | tok/s/GPU |
|---|---|
| TP4 x DP3 replicated DSA | 39.9K |
| TP4 x DP3 DSA-DP | 42.8K |
| TP4 x DP3 DSA-DP + SP-local aux | 46.1K |
| TP2 x DP6 | 45.3K |
| DP4 x TP4 / DCP4 | 41.4K |
| TP8 + DCP8 | 32.2K |
| DEP12 | 50.0K |

- Remaining gap to DEP12 per PR #20: KDA SP all-gather (~12 ms/step), drafter, MoE, prep/sample.
- DSA KV per TP group: 11.6M -> 20-21M tokens.
- Decode vs replicated DSA at 2 / 4 / 8 / 16 / 32 req/GPU: +1.1 / +0.1 / -2.5 / -3.7 / -5.0 %
  step time.
- Long-context single-node: DSA-DP is 7-12% slower (one chunk's DSA lands on one card).

### 3.8 MoE: EP (MegaMoE) vs TP

**MegaMoE EP** (`megamoe_rubin/*`):
- One fused launch per layer: router metadata + dispatch (`token_in`, **one pull per route**) +
  FC1 + SiTU + FC2 + BF16 `token_back` per route, then a separate BF16 `TopkReduce`.
- **Bytes per routed token per MoE layer:**
  - dispatch 16 x 3696 = 59.1 KB (MXFP8) or 16 x 2016 = 32.3 KB (NVFP4);
  - combine 16 x 7168 = 114.7 KB.
  - The remote fraction is (E-1)/E.
- That is **~12x a BF16 hidden row (14.3 KB) per token per layer**. EP comm volume is per token and
  independent of E. The per-GPU load is `m x 173.8 KB x (E-1)/E` with m rows per rank, times an
  imbalance factor (max/mean routes per rank, after EPLB).
- Die-local expert streaming (`die_local.py`): homes experts on their die. It is a memory-locality
  change (17.1 vs 13.9 TB/s), with no comm change.

**TP MoE (no EP, TP16 decode):**
- Every rank holds 1/N of every expert. No dispatch.
- Comm per MoE layer ≈ front AG `R x 10.75 KB x (N-1)/N` + latent AR `R x 7168` + shared/out AR
  `R x 14336` (P14 + P15).
- Weight bytes read per step: every touched expert's 1/N slice. At large R this is the same total
  as EP.

### 3.9 Drafter (DSpark GQA) under each topology

**Per draft step** (`dspark_gqa.py`, `runtime/draft.py`, `cutedsl/draft/*`):

| part | TP-N | SP (C512) | DEP | DSA-DP |
|---|---|---|---|---|
| qkv / o_proj | col (96/N q, 16/N kv heads; N <= 16) / row + AR fused with RMSNorm (`fused_allreduce_rms_norm`, P1c `ArNorm`) | TP4 as TP | local, full | TP (every card) |
| MLP 14336 | col/row + AR | same | local | TP |
| AR sites per draft forward | 11 (embedding, two per layer, final norm; `draft/allreduce.py`), each `[R_d, h]` BF16 | same | 0 | same as TP |
| context step `fc` 35840 -> 7168 | rep. For >= 256 rows per rank: row-shard + AG `[rows, h]` (`combine_aux_rows`) | SP-local aux: project own shard, AG projected rows (PR #20) | local | TP |
| SWA K/V insert (5 layers) | kv heads / N | same | local | shared pool, kv heads / N |
| LM head + Markov | vocab / N + Markov key exchange per position (P16, `markov.py`); sampled: per-rank draft-logits columns | same | full vocab local | TP |
| aux input from the target | replicated (TP) | final AG `[hidden \| 5 aux]` (6h wide, P18) or SP-local | local | TP / SP |

`R_d` = rows of the draft forward (requests x draft block).
The stock fused AR + RMSNorm costs ~13 us per site at TP16 B1.

---

## 4. Compute and weight duplication (what is NOT divided by the group)

For the SOL model these are terms that do not scale with 1/N.

| topology | replicated FLOPs per rank | replicated weight streaming per rank |
|---|---|---|
| TP-N (no SP) | `fused_qkv_a` (7168 x 2112), indexer q (1536 x 4096) + k/w (7168 x 160), KDA f_a (7168 x 128): all T rows. Without ShardQ / DCP / DSA-DP also: indexer scoring `T x C x 32 x 128` and top-k; KV insert. Plain-TP stock MoE: router + latent down/up on all rows (the specialized front shards them). | Those weights (45 MB / DSA layer, 1.8 MB / KDA layer) |
| TP-N + SP | same attention replication (all T rows after AG) | MoE dense: router + latent + shared = 380 MB / MoE layer BF16 (x 92 = 35 GB per step), layer-0 MLP 1.45 GB |
| DCP (decode) | + W_UV over all 96 heads ("proj") on every rank; indexer q projection for all rows | as TP |
| DCP ShardQ (prefill) | W_UK / W_UV on the rank's T/W slice with all heads (no dup); new-token MLA rows computed on every rank | as TP |
| DSA-DP | none for DSA (owner only) | DSA weights 480 MB / layer x 24 = 11.5 GB per card |
| DEP | none | all dense weights (~113 GB per GPU, see §6) |

**Local-head padding** (TP without ShardQ / DCP / DSA-DP): the sparse MLA computes
`ceil_to_tile(96/N) / (96/N)` x the needed heads. For example, 24 -> 64 is 2.67x at TP4.

---

## 5. KV / state capacity per GPU

**Per-request bytes (exact formulas; `solmodel_work/cap.py` evaluates them):**

| state | bytes per request (full, TP1) | how it shards |
|---|---|---|
| MLA FP8 KV | `24 x 576 x C` = 13,824 B/token | replicated over TP (no DCP); /W under DCP; owner-only under DSA-DP; full under DEP |
| Indexer MXFP4 KV | `24 x 68 x C` = 1,632 B/token | same as MLA |
| KDA recurrent FP32 | `69 x 96 x 128 x 128 x 4` = 434.1 MB | /N (heads) under TP / SP / DCP / DSA-DP; full under DEP |
| KDA conv FP32 | `69 x 3 x 12288 x (3 + num_spec) x 4` = 30.5 MB (no spec), 61.0 (k3), 101.7 (k7) | /N |
| RecoverSSM records (`--use-replayssm`) | `69 x q x 96 x (128 x 4 + 256 x 2)` = 6.78 MB x q | /N |
| Drafter SWA KV (BF16) | `5 x 2 x 16 x 64 x 2` = 20,480 B/token x min(C, 2048 + tile/block slack) ≈ 42-47 MB | /N by kv heads (N <= 16); not DCP-sharded; full under DEP |
| KDA prefix checkpoints (prefix cache, Mooncake) | per checkpoint = KDA recurrent + conv | /N |

**At C = 135K, k3, per request:**
- MLA + indexer 2.09 GB;
- KDA 0.495 GB;
- SWA 0.047 GB;
- RecoverSSM 0.027 GB.

**Per GPU per request:**

| topology | bytes / GPU / request at C = 135K |
|---|---|
| TP-N, no DCP (replicated DSA) | 2.09 + (0.495 + 0.047 + 0.027)/N GB. The DSA KV does not shrink: TP4 = 2.23 GB. |
| DCP W = N | (2.09 + 0.495 + 0.047 + 0.027)/N. TP4 = 0.66 GB, TP16 = 0.166 GB. |
| DSA-DP (N) | mean (2.09 + ...)/N as DCP, but the DSA part sits on the owner's static region. Capacity = min over cards of region / owner load. |
| DEP | 2.66 GB on one GPU |
| Hybrid DP x (TP = DCP) | as DCP with N = 4 |

- **Weights per GPU** (`cap.py`, BF16 dense + MXFP4 routed 4.25 bits; HBM 286.5 GB):

  | layout | weights / GPU |
  |---|---|
  | TP16 (MoE TP) | 108.6 GB |
  | TP8 EP8 token shard | 205.4 GB |
  | DP4 x TP4 SP EP16 | 147.1 GB |
  | TP4 x DP3 DSA-DP EP12 | 185.0 GB |
  | DEP12 | 234.4 GB |
  | DEP16 | 204.2 GB |
  | DEP32 | 159.0 GB |
  | DEP72 | 133.9 GB |

  The routed experts are 1446 GB in total, so **EP >= 8 is needed just to fit**.
- **KV free per GPU** = `gmu x HBM - weights - reserve`, where the reserve covers:
  - activations;
  - CUDA graphs;
  - the MegaMoE arena + workspaces (C512 prefill: 0.94 + 7.5 GB per rank; decode TP8 P: 0.23 +
    0.96 GB);
  - the SP AG buffer (896 MiB);
  - the DCP ShardQ workspace (~0.7 GB per 1M tokens);
  - drafter weights.
- **The hybrid KV manager packs every group into one page pool.** The KDA state pages, the attention
  pages and the SWA slots share a page size, so a capacity model must follow `v1/core/kv_cache_utils`
  page packing, not raw bytes.
- **Logged anchors:**

  | instance | "GPU KV cache size" |
  |---|---|
  | TP16 + DCP16 decode (C64) | 56.5M tokens per engine |
  | TP8 + DCP8 prefill | 10.8M |
  | C512 DP4 x (TP4 + DCP4) decode | 19.3M per DP engine |
  | C512 prefill | 16.9M |

---

## 6. Weight bytes per GPU by layout (formula)

```
W_gpu = 69 x [(KDA_col + f_b + o_proj)/N + f_a] x 2
      + 24 x [DSA_rep + (DSA_col + o_proj)/N_dsa] x 2      (N_dsa = 1 under DSA-DP / DEP)
      + 92 x [MoE_rep + shared / N_sh] x 2                   (N_sh = 1 under SP / DEP, N under TP)
      + dense0 / N_sh x 2 + (emb + head) / N x 2
      + 92 x 896 x 3 x 3584 x 3072 x 4.25/8 / E              (routed, MXFP4)
```

**Parameter counts:**

| term | params |
|---|---|
| KDA_col | 7168 x (4 x 12288 + 96) = 353.0M |
| f_b | 1.57M |
| KDA o_proj | 88.1M |
| f_a | 0.92M |
| DSA_rep | 7168 x 2112 + 1536 x 4096 + 7168 x 160 = 22.6M |
| DSA_col | 1536 x 18432 + 512 x 24576 + 7168 x 12288 = 128.9M |
| DSA o_proj | 88.1M |
| MoE_rep | 7168 x 896 + 2 x 7168 x 3584 = 57.8M |
| shared | 3 x 7168 x 6144 = 132.1M |
| dense0 | 3 x 7168 x 33792 = 726.7M |
| emb, head | 1174.4M each |

The drafter adds ~5 GB BF16: 5 layers x 411M + fc 257M + Markov. It shares the embedding and LM head.

---

## 7. Non-NVLink transfer per request (P/D and prefix store)

**NIXL P -> D per request** (RDMA; D reads):
- target KV of the tokens D lacks: `ΔC x 15,456 B` (DSA groups);
- KDA state 434 MB + conv 30.5-102 MB (k-dependent; the latest checkpoint);
- drafter SWA window ~42-47 MB.
- **Total at ΔC = 135K: ~2.6 GB per request.**
- Homogeneous DCP (P and D same W, a rule): each rank moves 1/W.
- DSA-DP: the DSA groups move only between P rank o and D owner `o x tp_D / tp_P` (PR #23 step 5).

**Mooncake store** (prefill prefix hits, ~95-97%):
- **Load** per request: the hit prefix's DSA KV (`C_hit x 15,456`) + KDA checkpoint (~0.46-0.53 GB) +
  SWA window.
- **Save:** new blocks + checkpoints.
- Per prefill GPU it is sharded like the KV.
- At ~5.5-6.6K fresh tokens per request, that is ~2 GB loaded per ~6K computed tokens.
- These bytes are not NVLink. They load the host/RDMA path and the HBM copy engine. Memory note:
  "Mooncake not the prefill limit".

---

## 8. New strategies (not implemented)

Each entry gives the mechanism, the comm per layer (formula), the capacity effect and the main risk.
Exactness: every entry keeps stock math. Only reduction order changes, which the rules allow.

**S1. DSA-DP + split indexer.**
- **Mechanism:**
  - The owner keeps MLA KV + sparse attention, but every card of the TP group holds 1/N of the
    owner's indexer keys (68 B/token, sharded by token mod N).
  - The indexer query (32 x 128 MXFP4 ≈ 2.2 KB per row) is broadcast.
  - Each card scores its key shard, followed by an exact histogram top-k merge (P8 machinery).
  - The selected global ids (2048 x 4 B) go to the owner.
- **Comm per row:** `(N-1) x (2.2 KB + ~1.5 KB)` + 8 KB to the owner.
- **Effect:** removes the owner hotspot of the O(R x C) indexer at 135K contexts; MLA KV stays one
  copy.
- **Fit:** decode at long context; prefill chunks (the PR #23 "one chunk on one card" problem).

**S2. DCP only for DSA layers, DEP / TP for the rest ("DSA-CP").**
- **Mechanism:**
  - KDA head-parallel (TP-N_k) or DEP;
  - MoE EP;
  - DSA layers DCP over a group G that can differ from the TP group (e.g. G = 8 or 16 under DEP).
- **Comm per DSA layer per row:**
  - q publish `(G-1) x 96 x 576 B` (FP8, all heads, because each DEP rank owns all heads);
  - proj combine `(G-1) x 96 x 260 B`;
  - top-k `(G-1) x ~1.5 KB`;
  - total ≈ `(G-1) x 82 KB`.
- **Capacity:** DSA KV / G, with KDA full per request (DEP) or /N_k.
- **Trade:** this per-row cost is much higher than DSA-DP, so it pays only when the DSA KV, not
  compute, binds (high concurrency, decode).

**S3. Context-parallel prefill over the cached prefix: query shipping vs context gather.**
- **Mechanism:** keep the prefix KV sharded and ship queries to keys instead of gathering the
  context.
- **Bytes per DSA layer per request:**

  | method | bytes |
  |---|---|
  | DCP ShardQ (context gather) | `C x 644 x (W-1)/W` |
  | query shipping | `T x (W-1) x (2.2 KB idx q + 55 KB MLA q + 25 KB proj out + 1.5 KB top-k)` ≈ `T x (W-1) x 84 KB` |

- **Crossover:** `T / C ≈ 644 / (84 KB x W)`, about 1/1000 at W = 8. Prefill chunks (T/C ≈ 0.05)
  should keep context gathering. Decode (T = q rows) should ship queries, which is what DCP decode
  does.
- **Use:** it also prices PCP and ring-attention variants.

**S4. Pipeline / layer split (PP) over the NVL72 domain.**
- **Mechanism:** S stages of contiguous layers.
- **Boundary transfer:** `[T, h]` residual + the AttnRes state: the block prefix sum and the live
  block outputs, about `1 + ceil(layer/12)` rows of h. That is ≤ 9 x 14,336 B per token, once per
  boundary.
- **Capacity:**
  - weights per GPU / S;
  - KV per GPU only for its layers;
  - no DSA KV duplication with TP = 1 per stage.
- **Cost:** pipeline bubbles. Decode latency is S x the stage time; it needs many microbatches.
- **Fit:** a candidate for the prefill topology (the fresh chunks are big) combined with DEP inside
  each stage. It cuts the per-GPU weights for DEP (DEP12 at 234 GB leaves little KV room).

**S5. Dispatch dedup per destination rank, with in-rank pre-combine.**
- **Mechanism:** send a token once per destination rank instead of once per route; reduce the
  routes of one rank before sending back.
- **Bytes per token:** `U(E) x (3696 + 7168 or FP32 partial)`, with
  `U(E) = E x (1 - (1 - 1/E)^16)` = 10.3 (EP16), 12.4 (EP32), 14.5 (EP72).
- **Savings:** -36% / -23% / -9% of MoE comm at EP16 / 32 / 72.
- **Combine precision:** an FP32 per-rank partial then one BF16 rounding. That is only a reorder of
  stock's BF16-per-route + `TopkReduce`, but it **does not reproduce stock's per-route BF16
  rounding**, so it needs sign-off as "order may differ".

**S6. Mixed parallel degree per layer type.**
- **Mechanism:**
  - residual SP-sharded over a large group G (e.g. 16);
  - KDA layers head-parallel over subgroups of N_k (AG/RS inside the subgroup, n = T_sub / N_k);
  - DSA layers DSA-DP (or DCP) inside the subgroup;
  - MoE EP over G or the whole rack;
  - dense layer-0 / LM head TP-G.
- **Comm:** as §3.2 / §3.7 with N = N_k. No extra resharding, because SP shards are already
  per-subgroup.
- **Exploits:**
  - KDA is the bulk of the attention weights (69 layers x 882 MB), so TP-N_k amortizes it at small
    batch;
  - the MoE is per-token and best as wide EP.
- The present C512 layout is the special case G = 16, N_k = 4, DCP for DSA.

**S7. Attention-DP with GEMM-TP for decode (ADP-GTP).**
- **Mechanism:** DSA-DP extended to KDA. The KDA core and state move to the request owner, while the
  in/out projections stay TP.
- **Needs:** an A2A of the projected `[R, 4 x 12288 + ...]` BF16 rows (98 KB/row) to the owner and
  `[R, 12288]` (24.6 KB/row) back.
- **Verdict:** **worse than head-parallel KDA**, whose state already shards by head with no exchange.
  Recorded to close the option. The KDA state per GPU would also become per-owner (/1).

**S8. Sequence-parallel KDA (chunks of one long fresh prefill across ranks).**
- **Mechanism:**
  - each rank runs the chunked WY recurrence on its token slice from a zero state;
  - it returns the per-head transition (128 x 128 FP32 per head);
  - a prefix scan over ranks: `log2(N)` rounds of `96/N_h x 128 x 128 x 4 B` (= 6.29 MB / N_h per
    layer per request);
  - then a fix-up pass.
- **Use:** single-request TTFT (C1). AgentX fresh chunks (~6K) do not need it.
- **Exactness:** FP32 reorder of the recurrence. It needs sign-off: the stock chunk order changes.

**S9. Two-microbatch overlap (DBO-style) of all SP / DSA-DP / EP comm.**
- **Mechanism:** split a step into µb A and µb B, and run µb A's AG / RS / dispatch / combine under
  µb B's GEMMs.
- **Model:** `t_layer = max(compute, comm) + switch overhead` instead of the sum.
- **Today:**
  - GEMM-RS / AR fuses the reduction into the GEMM;
  - MegaMoE fuses dispatch with FC1;
  - the DCP ShardQ context push rides the copy engine under the indexer.
  - The SP AG and DSA-DP pushes are still exposed: KDA AG ~12 ms / step in PR #20.
- **Constraint:** `use_ubatching` excludes DCP ShardQ (dcp.py:1086), and the MegaMoE lock-step
  needs both microbatches on every EP rank.

**S10. Quantize-at-source for comm the consumer quantizes anyway.**
- **Already done:** the DCP q publish in FP8 (576 B vs 1152) and the MegaMoE dispatch in MXFP8.
- **Open:**
  - ShardQ proj variant: q could be shipped after W_UK + FP8 quant (`[T, H, 576]` FP8 = 576 B/head)
    vs BF16 192-dim (384 B/head). BF16 pre-W_UK is already smaller, so keep it.
  - SP AG of the KDA in_proj input could ship MXFP8 **only if** in_proj is an MXFP8 GEMM (online
    MXFP8 linears exist: `mxfp8.py`). Under the BF16-linears rule it is not allowed.
- **Rule check:** no lossy compression of BF16 activations.

**S11. Expert placement: die-local, rack-wide EP, redundant hot experts.**
- **Mechanism:**
  - EP72 with attention DP gives 20 GB of experts per GPU and per-token dispatch volume constant in
    E;
  - EPLB redundancy cuts the max/mean routes (the imbalance factor multiplies MoE comm and compute
    time);
  - die-local placement (already on for decode) treats each die as a half-rank for HBM.
- **Model input:** an imbalance factor per E, and the die-local bandwidth 17.1 vs 13.9 TB/s.

**S12. Drafter placement.**
- **Options:**
  - drafter TP inside one node (TP4) even when the target is TP16 (11 AR sites on 4 ranks instead
    of 16);
  - drafter DP per owner: full drafter weights (5 GB) streamed per step, 0 ARs.
- **Crossover:** `5 GB / 13.9 TB/s = 0.36 ms` of weight streaming vs `11 x AR(R_d x 14 KB, N)` per
  draft step. At B1 TP16 the ARs are ~11 x 13 us = 0.14 ms, so TP wins. Drafter-DP wins only at
  large R_d on DEP decode.

**S13. Indexer KV in its own parallel group ("indexer DCP").**
- **Mechanism:** shard only the indexer keys (68 B/token, 10% of the DSA KV) over a wide group;
  replicate or own the MLA KV.
- **Comm:** as S1.
- **Effect:** the O(C) indexer scan, the dominant DSA prefill FLOPs at long context, scales with the
  group, while the MLA traffic is unchanged.

---

## 9. Anchors for sanity (not for fitting)

| what | value | source |
|---|---|---|
| SP push AG 17.4K rows TP4 | 191 us (0.98 TB/s); NCCL ring 463 us | `ops/sp_all_gather.py` |
| GEMM-RS vs GEMM + NCCL RS, 40-layer TP4 SP prefill | -4.4% at 17.4K tokens | model.py:296 docstring |
| Token-shard GEMM-AR threshold | 288 tokens; -1 ms / 384-token step on TP8 | model.py:202 |
| DCP ShardQ kv_insert | ~110 us per DSA layer at 17K tokens (unfused) | `dcp_shardq/kv_insert.py` |
| MegaMoE EP16 8192-token bucket (prefill, NVFP4) | p50 1282 us | C512 prefill log |
| MegaMoE EP8 2048 bucket (NVFP4) | p50 ~374 us | C64 prefill log |
| MegaMoE genphase EP16 1024 capacity | NVFP4 346.6 us, MX 381.8 us | `megamoe_rubin/experts.py:89` |
| W_UV in-kernel vs batched GEMM (DCP8 GB300) | 47-51 us at 320 rows | `dsa/dcp_stage.py` |
| PR #20 / #23 prefill and decode | §3.7 | PR bodies |

---

## 10. Inferred, not read (verify before relying on them)

1. **P8 `DcpExchange` per-row bytes.** I priced them by analogy with the prefill histogram merge
   (`(W-1) x (1536 + 4 x tie_words)`). The decode kernel's mailbox layout is in
   `dcp_exchange/csrc/dcp_exchange.cu`, which I did not parse.
2. **NVLS AR / RS per-GPU bytes.** Taken as ≈ M in and out. This is the switch-reduction
   accounting, not measured per link.
3. **MegaMoE token metadata bytes per route** (TokenSrcMetadata + score, tens of bytes) are ignored
   next to the 3.7-7.2 KB payloads.
4. **Drafter draft-forward rows `R_d`.** DSpark drafts a block of k positions per request, but I did
   not trace the exact row count of the draft forward and the context step per verify outcome.
5. **The hybrid KV manager's page packing** is noted (§5) but not modeled. The logged "KV cache size
   tokens" cannot be converted to requests with the raw bytes of §5.
6. **The PR #20/#23 code is not on `wzhao/k31-agentx-dev`.** It is from the PR diffs: base
   `wzhao/agentx-vr-nov-13`, head `juntianl/k31vr-tp-dsadp` / `juntianl/k31vr-dsadp-pd`.
