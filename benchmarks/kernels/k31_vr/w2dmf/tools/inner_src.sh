#!/bin/bash
# inside container: inner_src.sh <env-tag> <cmd...>   (cwd = $W/w2dmf/src-<tag>)
export K31VR_VLLM_TAG=w2dmf-$1; shift
source /lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/env/k31vr_env.sh
export VLLM_CACHE_ROOT=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/cache/jit/vllm-w2dmf
cd /lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/w2dmf/src-${K31VR_VLLM_TAG#w2dmf-}
"$@"
