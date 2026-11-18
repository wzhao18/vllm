| family | n | median derate | time-weighted (sum meas / sum ach) | min | max | >2.0 |
|---|---|---|---|---|---|---|
| attnres | 27 | 5.19 | 1.46 | 1.17 | 482.91 | 18 |
| drafter DS.attn | 15 | 8.29 | 6.01 | 4.38 | 37.92 | 15 |
| drafter DS.markov | 15 | 4.21 | 2.25 | 1.29 | 11.43 | 10 |
| dsa idx+topk decode | 12 | 7.61 | 6.33 | 5.00 | 53.73 | 12 |
| dsa idx+topk prefill | 5 | 1.85 | 1.85 | 1.84 | 1.90 | 0 |
| gemm bf16 M<=1024 | 129 | 2.84 | 1.65 | 1.03 | 50.69 | 88 |
| gemm bf16 M<=64 | 138 | 2.23 | 1.37 | 1.05 | 109.71 | 82 |
| gemm bf16 M>1024 | 75 | 1.21 | 1.19 | 0.95 | 18.45 | 17 |
| gemm bf16 bmmM<=1024 | 6 | 3.13 | 2.70 | 1.79 | 8.09 | 5 |
| gemm bf16 bmmM<=64 | 8 | 4.09 | 4.32 | 3.35 | 26.27 | 8 |
| gemm bf16 bmmM>1024 | 9 | 1.23 | 1.22 | 1.02 | 3.81 | 1 |
| gemm fp8 M<=1024 | 28 | 4.05 | 2.48 | 1.19 | 16.34 | 24 |
| gemm fp8 M<=64 | 28 | 2.63 | 1.95 | 1.19 | 18.52 | 20 |
| gemm fp8 M>1024 | 19 | 1.26 | 1.25 | 1.06 | 7.15 | 5 |
| kda decode | 15 | 4.30 | 3.72 | 3.08 | 10.00 | 15 |
| kda prefill | 4 | 1.71 | 1.98 | 1.37 | 2.77 | 1 |
| sparse_mla decode | 12 | 6.86 | 5.09 | 3.18 | 36.58 | 12 |
| sparse_mla prefill | 5 | 3.60 | 3.70 | 3.25 | 3.86 | 5 |

| case | prio | ops | topologies | shape | measured us | t_ach us | t_sol us | derate |
|---|---|---|---|---|---|---|---|---|
| dsa_indexer_topk-042 | P0 | DSA.idx_q_quant/DSA.idx_score/DSA.topk | P4-2,P6 | rows=8192,mean_keys_per_row=75628.64,heads=32,mode=DSA-DP owner (all rows of owned requests)  | 1293.1 | 697.8 | 663.2 | 1.85 |
| dsa_indexer_topk-087 | P0 | DSA.idx_q_quant/DSA.idx_score/DSA.topk | D5,D7-2,D7-4,D7-8 | rows=128,mean_keys_per_row=135000,heads=32,mode=DSA-DP owner (all rows of owned requests)  | 131.8 | 26.2 | 18.4 | 5.03 KD |
| gemm-003 | P0 | KDA.in_proj | P3-16,P9 | M=16384,N=49376,K=7168 bf16 | 3903.0 | 3177.0 | 2899.0 | 1.23 |
| gemm-004 | P0 | KDA.in_proj | P2 | M=16384,N=12440,K=7168 bf16 | 858.5 | 800.5 | 730.5 | 1.07 |
| gemm-005 | P0 | KDA.in_proj | P4-2,P6 | M=16384,N=24752,K=7168 bf16 | 1684.6 | 1593.0 | 1453.0 | 1.06 |
| gemm-021 | P0 | DSA.g_proj/MOE.shared_gate_up | P3-16,P9 | M=16384,N=12288,K=7168 bf16 | 815.1 | 790.7 | 721.6 | 1.03 |
| gemm-022 | P0 | KDA.in_proj | P3-16,P9 | M=16384,N=49376,K=7168 fp8 | 1489.7 | 1064.0 | 662.7 | 1.40 |
| gemm-023 | P0 | DSA.g_proj/MOE.shared_gate_up | P2 | M=4096,N=12288,K=7168 bf16 | 211.9 | 197.7 | 180.4 | 1.07 |
| gemm-024 | P0 | DSA.g_proj/MOE.shared_gate_up | P4-2,P6 | M=8192,N=12288,K=7168 bf16 | 389.8 | 395.4 | 360.8 | 0.99 |
| gemm-026 | P0 | DSA.g_proj/MOE.shared_gate_up | D3,D4-16,D4-32,D5,D7-2,D7-4,D7-8 | M=4,N=12288,K=7168 bf16 | 20.4 | 12.7 | 8.0 | 1.60 |
| gemm-027 | P0 | KDA.in_proj | P4-2,P6 | M=16384,N=24752,K=7168 fp8 | 566.1 | 533.4 | 332.2 | 1.06 |
| gemm-029 | P0 | DSA.o_proj/KDA.o_proj | P3-16,P9 | M=16384,N=7168,K=12288 bf16 | 927.8 | 790.7 | 721.6 | 1.17 |
| gemm-030 | P0 | KDA.in_proj | P2 | M=16384,N=12440,K=7168 fp8 | 314.0 | 268.1 | 167.0 | 1.17 |
| gemm-032 | P0 | DSA.o_proj/KDA.o_proj | P2 | M=16384,N=7168,K=3072 bf16 | 219.4 | 197.7 | 180.4 | 1.11 |
| gemm-035 | P0 | DSA.g_proj/MOE.shared_gate_up | D3,D4-16,D4-32,D5,D7-2,D7-4,D7-8 | M=32,N=12288,K=7168 bf16 | 22.3 | 12.8 | 8.1 | 1.75 |
| gemm-037 | P0 | KDA.o_proj/MOE.shared_down | P3-16,P4-2,P6,P9 | M=16384,N=7168,K=6144 bf16 | 386.0 | 395.4 | 360.8 | 0.98 |
| gemm-038 | P0 | KDA.in_proj | D4-16,D4-32 | M=4,N=49376,K=7168 bf16 | 55.9 | 51.0 | 32.2 | 1.10 |
| gemm-046 | P0 | DSA.o_proj/KDA.o_proj | D4-16,D4-32,D5,D7-2,D7-4,D7-8 | M=4,N=7168,K=12288 bf16 | 20.3 | 12.7 | 8.0 | 1.60 |
| gemm-048 | P0 | KDA.in_proj | D4-16,D4-32 | M=32,N=49376,K=7168 bf16 | 57.5 | 51.2 | 32.3 | 1.12 |
| gemm-055 | P0 | DSA.o_proj/KDA.o_proj | D4-16,D4-32,D5,D7-2,D7-4,D7-8 | M=32,N=7168,K=12288 bf16 | 21.3 | 12.8 | 8.1 | 1.66 |
| gemm-066 | P0 | DSA.g_proj/MOE.shared_gate_up | D3,D4-16,D4-32,D5,D7-2,D7-4,D7-8 | M=128,N=12288,K=7168 bf16 | 18.6 | 13.0 | 8.2 | 1.43 |
| gemm-079 | P0 | KDA.in_proj | D4-16,D4-32 | M=4,N=49376,K=7168 fp8 | 31.2 | 26.3 | 16.6 | 1.19 |
| gemm-096 | P0 | KDA.in_proj | D3,D5,D7-4 | M=16,N=12440,K=7168 bf16 | 17.1 | 12.9 | 8.1 | 1.33 |
| gemm-097 | P0 | KDA.in_proj | D3,D5,D7-4 | M=512,N=12440,K=7168 bf16 | 36.1 | 25.0 | 22.8 | 1.44 |
| gemm-098 | P0 | KDA.in_proj | D4-16,D4-32 | M=32,N=49376,K=7168 fp8 | 32.4 | 26.5 | 16.8 | 1.22 |
| gemm-103 | P0 | MOE.latent_down | D3,D4-16,D4-32,D5 | M=4,N=3584,K=7168 bf16 | 9.9 | 3.7 | 2.3 | 2.66 KD |
| gemm-104 | P0 | MOE.latent_up | D3,D4-16,D4-32,D5 | M=4,N=7168,K=3584 bf16 | 9.4 | 3.7 | 2.3 | 2.53 KD |
| gemm-107 | P0 | DSA.o_proj/KDA.o_proj | D4-16,D4-32,D5,D7-2,D7-4,D7-8 | M=128,N=7168,K=12288 bf16 | 20.6 | 13.3 | 8.4 | 1.55 |
| gemm-113 | P0 | MOE.shared_gate_up | D3,D4-16,D4-32,D5 | M=4,N=12288,K=7168 fp8 | 11.4 | 6.5 | 4.1 | 1.74 |
| kda-020 | P0 | KDA.conv/KDA.core_prefill/KDA.gate_beta_norm/KDA.o_norm | P2 | phase=prefill,tokens=16384,heads=24  | 292.8 | 188.4 | 188.4 | 1.55 |
| kda-025 | P0 | KDA.conv/KDA.core_decode/KDA.gate_beta_norm/KDA.o_norm | D3,D5,D7-4 | phase=decode,B=128,heads=24  | 107.8 | 35.0 | 22.2 | 3.08 KD |
| kda-064 | P0 | KDA.conv/KDA.core_decode/KDA.gate_beta_norm/KDA.o_norm | D4-16,D4-32 | phase=decode,B=32,heads=96  | 124.8 | 35.0 | 22.2 | 3.56 KD |
| kda-082 | P0 | KDA.conv/KDA.core_decode/KDA.gate_beta_norm/KDA.o_norm | D7-2 | phase=decode,B=64,heads=48  | 108.6 | 35.0 | 22.2 | 3.10 KD |
| kda-090 | P0 | KDA.conv/KDA.core_decode/KDA.gate_beta_norm/KDA.o_norm | D3,D5,D7-4 | phase=decode,B=32,heads=24  | 35.1 | 8.8 | 5.6 | 4.01 KD |
| kda-099 | P0 | KDA.conv/KDA.core_decode/KDA.gate_beta_norm/KDA.o_norm | D7-8 | phase=decode,B=256,heads=12  | 150.6 | 35.0 | 22.2 | 4.30 KD |

top excess us/step (measured - t_ach) x count:
  gemm-109 KDA.in_proj P1 excess 149178 us/step derate 6.35
  gemm-003 KDA.in_proj P3-16,P9 excess 50093 us/step derate 1.23
  sparse_mla-162 DSA.gate_mul/DSA.prologue/DSA.sparse_mla P9 excess 45422 us/step derate 3.86
  sparse_mla-161 DSA.gate_mul/DSA.prologue/DSA.sparse_mla P3-16 excess 43643 us/step derate 3.75
  kda-077 KDA.conv/KDA.core_prefill/KDA.gate_beta_norm/KDA.o_norm P3-16,P9 excess 30822 us/step derate 2.77
  gemm-022 KDA.in_proj P3-16,P9 excess 29374 us/step derate 1.40
  dsa_indexer_topk-080 DSA.idx_q_quant/DSA.idx_score/DSA.topk P3-16 excess 28040 us/step derate 1.84
  dsa_indexer_topk-081 DSA.idx_q_quant/DSA.idx_score/DSA.topk P9 excess 28032 us/step derate 1.84
  sparse_mla-094 DSA.gate_mul/DSA.prologue/DSA.sparse_mla P4-2,P6 excess 20636 us/step derate 3.60
  dsa_indexer_topk-042 DSA.idx_q_quant/DSA.idx_score/DSA.topk P4-2,P6 excess 14288 us/step derate 1.85
  gemm-029 DSA.o_proj/KDA.o_proj P3-16,P9 excess 12750 us/step derate 1.17
  kda-051 KDA.conv/KDA.core_prefill/KDA.gate_beta_norm/KDA.o_norm P4-2,P6 excess 12432 us/step derate 1.86
  sparse_mla-089 DSA.gate_mul/DSA.prologue/DSA.sparse_mla P2 excess 9603 us/step derate 3.31
  gemm-212 KDA.in_proj D7-8 excess 9116 us/step derate 6.23
  gemm-252 KDA.in_proj D1,D8 excess 9108 us/step derate 6.12
  kda-099 KDA.conv/KDA.core_decode/KDA.gate_beta_norm/KDA.o_norm D7-8 excess 7976 us/step derate 4.30
  dsa_indexer_topk-043 DSA.idx_q_quant/DSA.idx_score/DSA.topk P2 excess 7368 us/step derate 1.88
  kda-020 KDA.conv/KDA.core_prefill/KDA.gate_beta_norm/KDA.o_norm P2 excess 7202 us/step derate 1.55
  gemm-005 KDA.in_proj P4-2,P6 excess 6319 us/step derate 1.06
  kda-064 KDA.conv/KDA.core_decode/KDA.gate_beta_norm/KDA.o_norm D4-16,D4-32 excess 6196 us/step derate 3.56
  kda-082 KDA.conv/KDA.core_decode/KDA.gate_beta_norm/KDA.o_norm D7-2 excess 5078 us/step derate 3.10
  kda-146 KDA.conv/KDA.core_decode/KDA.gate_beta_norm/KDA.o_norm D1,D8 excess 5057 us/step derate 3.09
  kda-025 KDA.conv/KDA.core_decode/KDA.gate_beta_norm/KDA.o_norm D3,D5,D7-4 excess 5021 us/step derate 3.08
  sparse_mla-310 DSA.gate_mul/DSA.prologue/DSA.sparse_mla P1 excess 4954 us/step derate 3.25
  gemm-091 MOE.shared_gate_up P3-16,P9 excess 4546 us/step derate 1.19
