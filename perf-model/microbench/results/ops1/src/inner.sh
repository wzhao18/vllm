#!/bin/bash
# inner.sh <outdir> <script.py> <args...>: inside the e2e image on one VR node; runs the script once per GPU 0..NG-1
# (NG=${OPS1_NG:-4}) with shard i/NG as the last arg, in the release tree env (rel-1767cf7 + FI 0.7.3.dev20261031).
W=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr; L=$W/mb-ops1
OUT=$1; SCRIPT=$2; shift 2
TREE=${OPS1_TREE:-$W/dsrel/trees/rel-1767cf7}
export PYTHONPATH=$L/src:$TREE:$W/env/fi-0.7.3.dev20261031:$W/e2e/env/pydeps${OPS1_EXTRA_PP:+:$OPS1_EXTRA_PP}
export HOME=$W/home XDG_CACHE_HOME=$W/cache/xdg XDG_CONFIG_HOME=$W/cache/xdg-config XDG_DATA_HOME=$W/cache/xdg-data
export HF_HOME=$W/cache/hf PIP_CACHE_DIR=$W/cache/pip UV_CACHE_DIR=$W/cache/uv
export CUTE_DSL_CACHE_DIR=$W/cache/jit/cutedsl TORCHINDUCTOR_CACHE_DIR=$W/cache/jit/inductor VLLM_CACHE_ROOT=$W/cache/jit/vllm
export DG_JIT_CACHE_DIR=$W/cache/jit/deepgemm TRITON_CACHE_DIR=/tmp/ops1-jit/triton CUDA_CACHE_PATH=/tmp/ops1-jit/nv
export K31_DSAC_BUILD_DIR=$L/build/dsac NQ_BUILD=$L/build
export VLLM_SSM_CONV_STATE_LAYOUT=DS PYTHONUNBUFFERED=1
mkdir -p $OUT $TRITON_CACHE_DIR $CUDA_CACHE_PATH $L/build/dsac
cd /tmp
NG=${OPS1_NG:-4}; G0=${OPS1_G0:-0}; NT=${OPS1_NT:-$NG}
echo "INNER $(hostname) $(date +%T) tree=$TREE rev=$(head -1 $TREE/SOURCE_REV) script=$SCRIPT args=$*"
nvidia-smi --query-gpu=index,name,memory.used,clocks.sm,clocks.mem --format=csv,noheader
for ((g = 0; g < NG; g++)); do
  CUDA_VISIBLE_DEVICES=$g timeout ${OPS1_TIMEOUT:-3000} python3 $L/src/$SCRIPT "$@" $OUT/rows.g$((g+G0)).jsonl $((g+G0))/$NT > $OUT/g$((g+G0)).log 2>&1 &
done
wait
for ((g = G0; g < G0 + NG; g++)); do echo "== g$g: $(grep -c '^ROW' $OUT/g$g.log) rows, $(grep -c '"error"' $OUT/rows.g$g.jsonl 2>/dev/null) errors; tail -2 $OUT/g$g.log"; done
echo "INNER done $(date +%T)"
