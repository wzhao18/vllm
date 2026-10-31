#!/bin/bash
# Review fixes: serving-size batches (q4 128/192 requests, q8 64/96 per rank) on the n1 8-layer DEP4
# harness, W0 stock (split16) vs M1+M5 with the Int32 split cap. Before the fix M5 raised at q4 >= 128 / q8 >= 64.
W=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr; T=$W/w6/tools; R=$W/w6/results; J=${1:-$SLURM_JOB_ID}
bash $T/abba.sh $J 1 4 stock m5rf $W/w6/shapes_rf_q4.json $R/n1_rf2_q4 --max-num-seqs 192 > $R/n1_rf2_q4.out 2>&1
bash $T/abba.sh $J 1 8 stock m5rf $W/w6/shapes_rf_q8.json $R/n1_rf2_q8 --max-num-seqs 96 > $R/n1_rf2_q8.out 2>&1
