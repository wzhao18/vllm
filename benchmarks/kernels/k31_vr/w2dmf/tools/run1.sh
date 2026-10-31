#!/bin/bash
# run1.sh <job> <gpus> <log> <cmd...> : one process on GPU(s) <gpus> of a held node, inside the image
J=$1; G=$2; L=$3; shift 3
W=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr
CUDA_VISIBLE_DEVICES=$G bash $W/env/run_in_job.sh $J k31w2dmf $L "$@"
