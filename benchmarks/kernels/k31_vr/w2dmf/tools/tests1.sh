#!/bin/bash
# tests1.sh <tag> <pytest args...>: pytest inside the image (conftest replaced by k31w_fixtures)
W=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr
export K31VR_VLLM_TAG=w2dmf-$1; shift
source $W/env/k31vr_env.sh
export VLLM_CACHE_ROOT=$W/cache/jit/vllm-w2dmf PYTHONDONTWRITEBYTECODE=1 VLLM_SSM_CONV_STATE_LAYOUT=DS
cd $W/w2dmf/src-${K31VR_VLLM_TAG#w2dmf-}; export PYTHONPATH=$PYTHONPATH:$W/wide/tools
python3 -m pytest --noconftest -p no:cacheprovider -p k31w_fixtures -q "$@" 2>&1 | tail -${TAILN:-60}
