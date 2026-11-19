#!/bin/bash
# step_d.sh [-n NAME] [-t TAG] [-m MIN] [-e EMUL_EP] [-c CTX] [-x] <ovtag> <rows> [prof 0|1]
# (run on the Rhea login node; long runs under setsid nohup)
# 1-node D3-32 decode step slice: one DP rank of the deployed hybrid DCP4 x DP8 EP32 (TP4 = DCP4 attention with
# ShardQ DSA, SP dense path, MegaMoE, specialized decode, spec decode k=3, DSpark preview drafter), on 4 pool GPUs
# (wsgpu.sh -g 4) with overlay <ovtag>. Rows = verify rows per DP rank (B x 4): 32 / 64 / 96 requests per DP rank =
# r8 / r16 / r24 per GPU -> rows 128,256,384.
# Slice vs deployment (see k31ws/tools/stepbench/README.md):
#  * 24-layer view (18 KDA + 6 DSA + 23 MoE + 1 dense; 896 experts x 93 layers do not fit at EP4);
#  * expert-count emulation (default EP32, -e 0 = off): routing restricted to 28 experts per GPU, the deployed
#    per-GPU expert count at the deployed token-expert pairs per GPU (nq_step.py patch_expert_emulation);
#  * MegaMoE dispatch/combine over 4 GPUs of one node instead of 32 over the rack fabric;
#  * DSpark preview drafter (24-layer target view) included.
# Context: -c u132k (default; uniform 132K = the model's L135K rows) or agentx.
# Output: $K/runs/<NAME>/step_d/<TAG>/ (agentx.log = rank 0 log, agentx.json, prof-agentx/, gap.json + fam.txt when
# profiled: per-family busy/exposed vs model achievable at the same shape (24-layer model rows)).
# -x: --exclude-drafter (target only). -q nvfp4: NVFP4 checkpoint view (k31-nvfp4-view-L24, EN) + NVFP4 model rows.
K=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws; S=$K/tools/stepbench
NAME=${WS_NAME:-anon}; TAG=""; MIN=75; EMU=32; CTX=u132k; XD=""
Q=mxfp4
while getopts n:t:m:e:c:q:x o; do
  case $o in n) NAME=$OPTARG;; t) TAG=$OPTARG;; m) MIN=$OPTARG;; e) EMU=$OPTARG;; c) CTX=$OPTARG;; q) Q=$OPTARG;; x) XD=1;; *) exit 2;; esac
done
shift $((OPTIND-1))
OV=$1; ROWS=$2; PROF=${3:-0}
[ -n "$OV" ] && [ -n "$ROWS" ] || { sed -n 2,20p $0; exit 2; }
TAG=${TAG:-$OV-r${ROWS//,/_}-$(date +%m%d%H%M)}
R=$K/runs/$NAME/step_d/$TAG; mkdir -p $R
[ -e $R/agentx.log ] && { echo "step_d: $R exists"; exit 3; }
DRAFT=$K/models/dspark-preview-L24; TARGET=$K/models/k31-view-L24; CS=""
case $Q in mxfp4) ;; nvfp4) TARGET=$K/models/k31-nvfp4-view-L24; CS=_nvfp4;; *) echo "-q mxfp4|nvfp4"; exit 2;; esac
NQ_TARGET=$TARGET python3 $S/cfg2bench.py $S/rhea_d3x32_dp1$CS.json --arm ls --drafter $DRAFT --rows $ROWS > $R/agentx.cfg.sh || exit 2
grep -q "NQ_ARGS=(--model $TARGET" $R/agentx.cfg.sh || { echo "step_d: target rewrite failed"; exit 2; }
python3 $S/nq_shapes.py 4 $R/agentx.shapes.json --rows $ROWS --ctx $CTX > $R/agentx.shapes.txt || exit 2
EXTRA=(); [ "$PROF" = 1 ] && EXTRA+=(--profile-dir $R/prof-agentx); [ -n "$XD" ] && EXTRA+=(--exclude-drafter)
{ echo "STEP_D ov=$OV rows=$ROWS ctx=$CTX prof=$PROF ckpt=$Q emulate_ep=$EMU exclude_drafter=${XD:-0} name=$NAME start=$(date +%FT%T)"
  cat $R/agentx.shapes.txt; grep '^#' $R/agentx.cfg.sh; } > $R/agentx.log
echo "$(date +%T) start $TAG" > $R/status.txt
NQ_EMULATE_EP=$EMU bash $K/tools/wsgpu.sh -g 4 -m $MIN -o $OV -n $NAME -P strict:$R/agentx.cfg.sh -- \
  env NQ_EMULATE_EP=$EMU bash $S/step_inner.sh $R/agentx "${EXTRA[@]}" >> $R/agentx.log 2>&1
rc=$?
echo "STEP_D end rc=$rc $(date +%FT%T)" >> $R/agentx.log
echo "$(date +%T) end rc=$rc" >> $R/status.txt
grep -a "^\[forward\].*median\|^NQ: expert emulation\|Traceback\|Error" $R/agentx.log | sed 's/, sample spread.*//' | head -20
if [ "$PROF" = 1 ] && [ $rc = 0 ]; then
  ML=$S/model_rows_L24.json; [ $Q = nvfp4 ] && ML=$S/model_rows_L24_nvfp4_modelopt.json; [ "$CTX" = u132k ] || echo "step_d: model rows are L135K; ctx $CTX differs"
  python3 $S/gap_parse.py decode $R $ML $R/gap.json --L 135 > $R/gap.txt 2>&1
  python3 $S/fam.py $R/gap.json > $R/fam.txt 2>&1
  cat $R/fam.txt
fi
echo "step_d: results in $R"
exit $rc
