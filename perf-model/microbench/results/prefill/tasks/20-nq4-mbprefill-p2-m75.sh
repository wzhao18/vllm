#!/bin/bash
# mb lane prefill: P2 (DP4 x TP4=DCP4 ShardQ, EP16, 4 nodes) prefill step, release tree rel-1767cf7, dummy weights.
# Run 1: MXFP8-pinned MegaMoE + torch profile, F16K (model ref) and F32K (production MNBT) per DP group at L 78K.
# Run 2 (time permitting): release MegaMoE policy (NVFP4 above 512 tok/rank), timing only.
source /lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/mb/prefill/tools/pf_common.sh
R=$L/results/p2; mkdir -p $R
pst "p2 start job $SLURM_JOB_ID hosts $NQ_HOSTS deadline $NQ_DEADLINE"
guard 30 || exit 0
export PF_TIMEOUT=3000 K31P1_PROFILE_RUNS=2
PF_MOE_QUANT=mxfp8 PF_CNAME=k31pf-a bash $T/pf_run.sh $NQ_HOSTS 4 $R/p2_mx "${P2[@]}" "${C16[@]}" "${C32[@]}" \
  --output-json $R/p2_mx.json --profile-dir $R/p2_mx.prof > $R/p2_mx.out 2>&1
pst "p2 A: $(grep -a '^PFEND' $R/p2_mx.log) $(grep -a '^\[forward\].*median' $R/p2_mx.log | sed 's/, sample spread.*//' | tr '\n' ';')"
for f in $R/p2_mx.prof/*.rank0.trace.json.gz; do
  [ -f "$f" ] && python3 $T/p1_attr.py $f --json ${f%.trace.json.gz}.attr.json > ${f%.trace.json.gz}.attr.txt 2>&1
done
if guard 25; then
  PF_TIMEOUT=1800 PF_MOE_QUANT=release PF_CNAME=k31pf-a bash $T/pf_run.sh $NQ_HOSTS 4 $R/p2_rel "${P2[@]}" "${C16[@]}" "${C32[@]}" \
    --output-json $R/p2_rel.json > $R/p2_rel.out 2>&1
  pst "p2 B: $(grep -a '^PFEND' $R/p2_rel.log) $(grep -a '^\[forward\].*median' $R/p2_rel.log | sed 's/, sample spread.*//' | tr '\n' ';')"
fi
pst "p2 done"
