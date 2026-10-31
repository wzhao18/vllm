#!/bin/bash
# Queue task body (runner on a 4-node block): full K3.1 DEP16 ABBA, stock top-k (auto) vs balanced,
# q4 then q8, W0 shapes. Results $W/w6/results/n4_q{4,8}_auto_bal.
W=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr
export GMU=0.95
bash $W/w6/tools/abba.sh $SLURM_JOB_ID 4 4 auto bal $W/w0dep/shapes/w0_q4.json $W/w6/results/n4_q4_auto_bal > $W/w6/results/n4_q4_auto_bal.out 2>&1
bash $W/w6/tools/abba.sh $SLURM_JOB_ID 4 8 auto bal $W/w0dep/shapes/w0_q8.json $W/w6/results/n4_q8_auto_bal > $W/w6/results/n4_q8_auto_bal.out 2>&1
