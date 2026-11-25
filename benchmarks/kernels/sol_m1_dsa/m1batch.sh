#!/bin/bash
# M1 batch 2: select variants, gather+QK at B1-B4 rows, prestage modes 0-3, Lamport rounds + contention (4 GPUs)
D=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/sol16/M1DSA
PROBES=sel ARGS_EXTRA="--rows 8,64,512 --agg 2,0" bash $D/m1run.sh
PROBES=gqk ARGS_EXTRA="--rows 8,16,32 --splits 1,2,4,8" bash $D/m1run.sh
PROBES=pre bash $D/m1run.sh
PROBES=lam bash $D/m1run.sh
