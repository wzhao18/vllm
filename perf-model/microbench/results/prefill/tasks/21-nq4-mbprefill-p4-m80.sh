#!/bin/bash
# mb lane prefill: P4 (TP x DP with DSA-DP, PR #20 juntianl/k31vr-tp-dsadp 83cd6d8e12 staged over vllm-jh-242fb49cd4 as
# mb/prefill/ov/pr20-83cd6d8), EP16, 4 nodes, dummy weights, MXFP8-pinned MegaMoE.
# VLLM_K3_TP_DSA_DP=1 (owner = batch index mod TP: the bench has no scheduler), VLLM_K3_TP_DSA_DP_KV=0 (replicated DSA
# KV: the bench allocates blocks without the per-owner region manager; the owner computes the same rows either way).
# P4-4 = TP4 x DP4 (+ torch profile), then P4-2 = TP2 x DP8. 4 requests x 4096 fresh at L 78K per TP group (F 16384).
source /lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/mb/prefill/tools/pf_common.sh
R=$L/results/p4; mkdir -p $R
pst "p4 start job $SLURM_JOB_ID hosts $NQ_HOSTS deadline $NQ_DEADLINE"
guard 30 || exit 0
export PF_TREE=$L/ov/pr20-83cd6d8 VLLM_K3_TP_DSA_DP=1 VLLM_K3_TP_DSA_DP_KV=0 K31P1_PROFILE_RUNS=2
P4=("${BASE[@]}" --enable-ep-weight-filter --max-num-batched-tokens 16384 --max-num-seqs 32 --speculative-config "$(spec 3 3.08)")
C4=(--case pf16k_L78k=4*4096@78360 --case pf16k_fresh=4*4096)
PF_TIMEOUT=2700 PF_MOE_QUANT=mxfp8 PF_CNAME=k31pf-p4 bash $T/pf_run.sh $NQ_HOSTS 4 $R/p4_4_mx "${P4[@]}" -tp 4 -dp 4 "${C4[@]}" \
  --output-json $R/p4_4_mx.json --profile-dir $R/p4_4_mx.prof > $R/p4_4_mx.out 2>&1
pst "p4-4: $(grep -a '^PFEND' $R/p4_4_mx.log) $(grep -a '^\[forward\].*median' $R/p4_4_mx.log | sed 's/, sample spread.*//' | tr '\n' ';')"
for f in $R/p4_4_mx.prof/*.rank0.trace.json.gz; do
  [ -f "$f" ] && python3 $T/p1_attr.py $f --json ${f%.trace.json.gz}.attr.json > ${f%.trace.json.gz}.attr.txt 2>&1
done
if guard 25; then
  PF_TIMEOUT=1800 PF_MOE_QUANT=mxfp8 PF_CNAME=k31pf-p4 bash $T/pf_run.sh $NQ_HOSTS 4 $R/p4_2_mx "${P4[@]}" -tp 2 -dp 8 "${C4[@]}" \
    --output-json $R/p4_2_mx.json > $R/p4_2_mx.out 2>&1
  pst "p4-2: $(grep -a '^PFEND' $R/p4_2_mx.log) $(grep -a '^\[forward\].*median' $R/p4_2_mx.log | sed 's/, sample spread.*//' | tr '\n' ';')"
fi
pst "p4 done"
