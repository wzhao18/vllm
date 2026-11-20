#!/bin/bash
# mb lane prefill: P1 (TP8+DCP8 ShardQ, EP8, 2 nodes) prefill step, release tree rel-1767cf7, dummy weights.
# Two P1 instances in parallel on the task's 4 nodes: A = MXFP8-pinned MegaMoE (model precision) + torch profile
# (per-layer / per-module attribution), B = release MegaMoE policy (NVFP4 above 512 tok/rank), timing only.
source /lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/mb/prefill/tools/pf_common.sh
H=(${NQ_HOSTS//,/ }); HA=${H[0]},${H[1]}; HB=${H[2]},${H[3]}; R=$L/results/p1
mkdir -p $R
pst "p1 start job $SLURM_JOB_ID hosts $NQ_HOSTS deadline $NQ_DEADLINE"
guard 30 || exit 0
export PF_PACKED=1 PF_TIMEOUT=3000 K31P1_PROFILE_RUNS=2
PF_MOE_QUANT=mxfp8 PF_CNAME=k31pf-a bash $T/pf_run.sh $HA 2 $R/p1_mx "${P1[@]}" "${C16[@]}" \
  --output-json $R/p1_mx.json --profile-dir $R/p1_mx.prof > $R/p1_mx.out 2>&1 &
PF_MOE_QUANT=release PF_CNAME=k31pf-b bash $T/pf_run.sh $HB 2 $R/p1_rel "${P1[@]}" "${C16[@]}" \
  --output-json $R/p1_rel.json > $R/p1_rel.out 2>&1 &
wait
pst "p1 A: $(grep -a '^PFEND' $R/p1_mx.log) $(grep -a '^\[forward\].*median' $R/p1_mx.log | sed 's/, sample spread.*//' | tr '\n' ';')"
pst "p1 B: $(grep -a '^PFEND' $R/p1_rel.log) $(grep -a '^\[forward\].*median' $R/p1_rel.log | sed 's/, sample spread.*//' | tr '\n' ';')"
for f in $R/p1_mx.prof/*.rank0.trace.json.gz; do
  [ -f "$f" ] && python3 $T/p1_attr.py $f --json ${f%.trace.json.gz}.attr.json > ${f%.trace.json.gz}.attr.txt 2>&1
done
pst "p1 done"
