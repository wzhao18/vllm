#!/bin/bash
# Review fixes, final policy (short batches keep the uniform split): q4 + q8 serving-size cases, W0 stock vs M1+M5.
W=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr; T=$W/w6/tools; R=$W/w6/results; J=${1:-$SLURM_JOB_ID}
while pgrep -f q_rf3.sh >/dev/null; do sleep 20; done
bash $T/abba.sh $J 1 4 stock m5rf2 $W/w6/shapes_rf_q4.json $R/n1_rf4_q4 --max-num-seqs 192 > $R/n1_rf4_q4.out 2>&1
bash $T/abba.sh $J 1 8 stock m5rf2 $W/w6/shapes_rf_q8.json $R/n1_rf4_q8 --max-num-seqs 96 > $R/n1_rf4_q8.out 2>&1
