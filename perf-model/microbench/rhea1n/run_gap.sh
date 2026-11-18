#!/bin/bash
# run_gap.sh <jobid> <tag> <ov> <rows> [profile 0|1] [ctx]: Rhea 1-node D3-32 per-DP-rank slice (TP4=DCP4, DP1, EP4 with
# 112 experts = 28/GPU as at EP32), latency_specialized decode step bench with the nq harness (Rhea variant), optional
# torch profile. Runs on the Rhea login node (srun --overlap into the hold). Outputs in $W/runs/<tag>/.
W=/lustre/fsw/coreai_comparch_inferencex/weizha/k31gap; T=$W/tools
J=$1; TAG=$2; OV=$3; ROWS=$4; PROF=${5:-1}; CTX=${6:-u132k}
NL=$(squeue -j $J -h -o %N)
R=$W/runs/$TAG; mkdir -p $R
export NQ_OV=$W/ov/$OV NQ_BUILD=$W/cache/build/$OV NQ_TIMEOUT=${NQ_TIMEOUT:-5400}
# 24-layer view (18 KDA + 6 DSA + 23 MoE + 1 dense, same 3:1 pattern, AttnRes block 12): 896 experts at EP4 do not fit
# at 93 layers and the specialized decode hard-codes the 896-expert MoE front (no expert-count emulation).
export NQ_HF_OVERRIDES=${NQ_HF_OVERRIDES-}
export NQ_TARGET=${NQ_TARGET:-$W/models/k31-view-L24}
DRAFT=${NQ_DRAFT:-$W/models/dspark-preview-L24}
EXTRA=(); [ "$PROF" = 1 ] && EXTRA=(-- --profile-dir $R/prof-agentx)
echo "$(date +%T) start $TAG job $J node $NL ov $OV rows $ROWS ctx $CTX prof $PROF target $NQ_TARGET draft $DRAFT" | tee -a $R/status.txt
bash $T/nq_run1.sh $J $NL $R/agentx $T/rhea_d3x32_dp1.json ls $DRAFT $ROWS $CTX "${EXTRA[@]}" > $R/agentx.out 2>&1
echo "$(date +%T) end rc=$? $(grep -a '^\[forward\].*median' $R/agentx.log | sed 's/, sample spread.*//; s/^\[forward\] //' | tr '\n' ';' | cut -c1-600)" | tee -a $R/status.txt
