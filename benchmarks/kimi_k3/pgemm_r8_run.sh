#!/bin/bash
# pgemm_r8_run.sh <outdir> [ledger args...]: P-GEMM R8 on one 4-GPU grant, three independent GPUs:
#   GPU0  deployed-kernel energy ledger (pgemm_r8_ledger.py, M 16384 + 8192)
#   GPU1  ncu (full set, base units) of one call per BF16 target / control row, then a summary
#   GPU2  cuBLASLt tactic J/call search (ltpow) for the BF16 rows R7 never searched
O=$1; shift
K=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws
mkdir -p $O/ncu
cd /k31ov/benchmarks/kimi_k3 || exit 1
nvidia-smi --query-gpu=index,name,power.limit,clocks.sm,temperature.gpu --format=csv > $O/gpus.txt
(CUDA_VISIBLE_DEVICES=0 python3 pgemm_r8_ledger.py --out $O/ledger.json "$@" > $O/ledger.log 2>&1) &
(
  for r in moe_latent_down moe_latent_down_pin dsa_o_proj dsa_o_proj_heur dense_gate_up \
           dense_gate_up_heur dense_down dense_down_heur moe_router dsa_qkv_a moe_latent_up \
           dsa_g_proj dsa_qb_wq kda_f_b; do
    CUDA_VISIBLE_DEVICES=1 $K/tools/nsight/bin/ncu --profile-from-start off --set full \
      --print-units base --csv --page raw python3 pgemm_r8_ncu.py --row $r \
      > $O/ncu/$r.csv 2> $O/ncu/$r.err
  done
  python3 pgemm_r8_ncusum.py $O/ncu $O/ncu.jsonl > $O/ncu.log 2>&1
) &
(
  G=$K/runs/P-GEMM/lt
  BIN=/tmp/ltpow_r8_$$
  /usr/local/cuda/bin/nvcc -gencode arch=compute_100,code=compute_100 -O2 -std=c++17 \
    -o $BIN $G/ltpow.cu -lcublasLt -ldl > $O/ltpow.log 2>&1 || exit 1
  export LT_NEXP=0 LT_NFIN=14 LT_NMEAS=3 LT_POW=2.5 LT_POWR=2
  for L in "router bf16 16384 896 7168 0 1" "qkv_a bf16 16384 2272 7168 0 0" \
           "qb_wq bf16 16384 22528 1536 0 0" "kda_fb bf16 16384 12288 128 0 0"; do
    echo "$L" > /tmp/ltpow_r8_one.$$
    CUDA_VISIBLE_DEVICES=2 timeout 1200 $BIN /tmp/ltpow_r8_one.$$ $O/ltpow.jsonl 300 >> $O/ltpow.log 2>&1 \
      || echo "SHAPE FAILED rc=$?: $L" >> $O/ltpow.log
  done
  rm -f $BIN /tmp/ltpow_r8_one.$$
) &
wait
echo "R8 DONE $(date +%T)"
tail -n 3 $O/ledger.log
grep -c '"row"' $O/ledger.log
wc -l < $O/ncu.jsonl
grep -c '"name"' $O/ltpow.jsonl 2>/dev/null
