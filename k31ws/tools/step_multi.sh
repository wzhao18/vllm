#!/bin/bash
# step_multi.sh [-n NAME] [-t TAG] [-m MIN] [-c CTX|CASES] d8|p16 <ovtag> <rows (d8) | - (p16)> [prof 0|1]
# (Rhea login node, under setsid nohup) Full-model multi-node step benches on the 8-node pool hold (wsgpu.sh -g 8n):
#  d8   D3-32 decode at the deployed topology: TP4=DCP4 x DP8, EP32 MegaMoE, specialized decode, k=3, 93-layer K3.1
#       view, DSpark preview drafter; rows = verify rows per DP rank (128,256,384 = r8,r16,r24); ctx u132k (default).
#  p16  P3-16 prefill on 4 of the 8 nodes: DEP16 (TP1 x DP16, EP16), stock prefill, F8K at L78K (+ F16K, which OOMed
#       at TP1 with MXFP4 weights; kept to re-check with NVFP4) ; -c case list (default pf8k_L78k,pf16k_L78k).
# No expert-count emulation (real EP). Output $K/runs/<NAME>/step_multi/<TAG>/ (x.log, x.json, prof-x/).
K=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws; S=$K/tools/stepbench
NAME=${WS_NAME:-anon}; TAG=""; MIN=90; CTX=""
while getopts n:t:m:c: o; do
  case $o in n) NAME=$OPTARG;; t) TAG=$OPTARG;; m) MIN=$OPTARG;; c) CTX=$OPTARG;; *) exit 2;; esac
done
shift $((OPTIND-1))
MODE=$1; OV=$2; ROWS=$3; PROF=${4:-0}
[ -n "$MODE" ] && [ -n "$OV" ] || { sed -n 2,10p $0; exit 2; }
TAG=${TAG:-$MODE-$OV-$(date +%m%d%H%M)}
R=$K/runs/$NAME/step_multi/$TAG; mkdir -p $R
O=$R/x
[ -e $O.log ] && { echo "step_multi: $R exists"; exit 3; }
TARGET=${MULTI_TARGET:-$K/models/k31-view}; DRAFT=${MULTI_DRAFT:-$K/models/dspark-preview}
case $MODE in
  d8) NN=8; export NQ_WORLD=32
      NQ_TARGET=$TARGET python3 $S/cfg2bench.py $S/rhea_d3x32_dp8.json --arm ls --drafter $DRAFT --rows $ROWS > $O.cfg.sh || exit 2
      python3 $S/nq_shapes.py 4 $O.shapes.json --rows $ROWS --ctx ${CTX:-u132k} > $O.shapes.txt || exit 2 ;;
  p16) NN=4; export NQ_WORLD=16
      NQ_TARGET=$TARGET python3 $S/cfg2bench.py $S/rhea_p3x16_dep16.json --arm stock --drafter $DRAFT --full-capture > $O.cfg.sh || exit 2
      python3 -c "
import json
d = json.load(open('$S/pf_shapes_p3.json')); want = '${CTX:-pf8k_L78k,pf16k_L78k}'.split(',')
json.dump({'cases': [c for c in d['cases'] if c['name'] in want]}, open('$O.shapes.json', 'w'), indent=1)
print(want)" > $O.shapes.txt || exit 2 ;;
  *) echo "mode d8|p16"; exit 2;;
esac
EXTRA=(); [ "$PROF" = 1 ] && EXTRA+=(--profile-dir $R/prof-x)
{ echo "STEP_MULTI mode=$MODE ov=$OV rows=$ROWS ctx=$CTX prof=$PROF target=$TARGET draft=$DRAFT name=$NAME start=$(date +%FT%T)"
  cat $O.shapes.txt; grep '^#' $O.cfg.sh; } > $O.log
RUN=$R/run.sh
printf '%s\n' '#!/bin/bash' \
  "# runs on the login node under wsgpu.sh -g 8n (WS_JOB / WS_NODES / WS_OV / WS_IMG set)" \
  "NL=\$(scontrol show hostnames \$WS_NODES | head -$NN | paste -sd,)" \
  "echo \"step_multi: job \$WS_JOB nodes \$NL\"" \
  "unset SLURM_NTASKS SLURM_NPROCS SLURM_NTASKS_PER_NODE SLURM_TASKS_PER_NODE SLURM_NNODES SLURM_JOB_NUM_NODES SLURM_NODELIST" \
  "rm -f $O.rdv" \
  "timeout ${MULTI_TIMEOUT:-5400} srun --overlap --jobid \$WS_JOB -N$NN -w \$NL --ntasks-per-node=4 --kill-on-bad-exit=1 \\" \
  "  --container-image=\$WS_IMG --container-mounts=/lustre:/lustre --no-container-mount-home --container-workdir=/tmp \\" \
  "  bash $K/tools/wsenv.sh bash $S/step_rank.sh $O ${EXTRA[*]}" > $RUN
echo "$(date +%T) start $TAG" > $R/status.txt
bash $K/tools/wsgpu.sh -g 8n -w ${MULTI_WAIT:-240} -m $MIN -o $OV -n $NAME -- bash $RUN >> $O.log 2>&1
rc=$?
echo "STEP_MULTI end rc=$rc $(date +%FT%T)" >> $O.log
echo "$(date +%T) end rc=$rc" >> $R/status.txt
grep -a "^\[forward\].*median\|Traceback\|Error" $O.log | sed 's/, sample spread.*//' | head -20
echo "step_multi: results in $R"
exit $rc
