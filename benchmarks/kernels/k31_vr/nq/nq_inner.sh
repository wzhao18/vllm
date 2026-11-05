#!/bin/bash
# nq_inner.sh <out prefix> [extra bench args]: one rank (srun task) of the 16-rank nq step bench, inside the e2e image.
# Reads <out>.cfg.sh (cfg2bench.py output: production decode env + engine args) and <out>.shapes.json (nq_shapes.py).
# env: NQ_OV (code-under-test overlay: a whole vllm tree from stage_overlay.sh; empty = the base env), NQ_BUILD.
# PYTHONPATH = the nq e2e decode role's (mk_nq.py PP["decode"]) minus dynamo/pinhook (serving only): base vLLM env
# vllm-jh-242fb49cd4 (jiahanc/k31-vr-agentx-local 242fb49cd4), FlashInfer 0.7.3.dev20261031, e2e pydeps.
W=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr
D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=$1; shift
# NQ_OV (stage_overlay.sh) is a whole vllm tree: the base env with the checkout's changed vllm/ files shipped over it,
# so it replaces the base env on PYTHONPATH (nq_step.py checks vllm / kimi_k3 resolve inside it)
VLLM_ENV=${NQ_OV:-${NQ_VLLM_ENV:-$W/env/vllm-jh-242fb49cd4}}
export PYTHONPATH=$VLLM_ENV:$W/env/fi-0.7.3.dev20261031:$W/e2e/env/pydeps
# caches: the merge-gate / oracle bench layout (env/k31vr_env.sh): Lustre JIT caches shared across runs (CuTe AOT objects
# under VLLM_CACHE_ROOT are content-addressed), Triton / CUDA caches node-local
export HOME=$W/home XDG_CACHE_HOME=$W/cache/xdg XDG_CONFIG_HOME=$W/cache/xdg-config XDG_DATA_HOME=$W/cache/xdg-data
export HF_HOME=$W/cache/hf PIP_CACHE_DIR=$W/cache/pip UV_CACHE_DIR=$W/cache/uv
export CUTE_DSL_CACHE_DIR=$W/cache/jit/cutedsl TORCHINDUCTOR_CACHE_DIR=$W/cache/jit/inductor VLLM_CACHE_ROOT=$W/cache/jit/vllm
export DG_JIT_CACHE_DIR=$W/cache/jit/deepgemm
export TRITON_CACHE_DIR=/tmp/k31nq-jit/triton CUDA_CACHE_PATH=/tmp/k31nq-jit/nvcache
# torch-extension builds (DCP exchange, DSA core) per code tree: their sources live in kimi_k3, so two overlays must
# never share a build dir (mutual rebuilds under one lock); the three slots of a block share one (same tree)
export NQ_BUILD=${NQ_BUILD:?NQ_BUILD}
source $OUT.cfg.sh  # export VLLM_*/NCCL_*/K31_* + FLASHINFER_CUBIN_DIR/JIT_DIR, TVM_FFI_CACHE_DIR (nq e2e Lustre dirs)
[ -n "$FLASHINFER_JIT_DIR" ] && export FLASHINFER_WORKSPACE_BASE=$(dirname $FLASHINFER_JIT_DIR)
mkdir -p $HOME $XDG_CACHE_HOME $CUTE_DSL_CACHE_DIR $TORCHINDUCTOR_CACHE_DIR $VLLM_CACHE_ROOT $DG_JIT_CACHE_DIR \
  $TRITON_CACHE_DIR $CUDA_CACHE_PATH $NQ_BUILD/dcpx $NQ_BUILD/dsac 2>/dev/null
# bench-only: 1M max-model-len check, unbuffered logs, the node's 4 GPUs for the 4 local ranks
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1,2,3
if [ "$SLURM_PROCID" = 0 ]; then
  echo "NQ rank0 $(hostname) ntasks $SLURM_NTASKS overlay ${NQ_OV:-none} build $NQ_BUILD"
  echo "NQ env $(env | grep -E '^(VLLM_|NCCL_|K31_|K3_)' | sort | tr '\n' ' ')"
  echo "NQ args ${NQ_ARGS[*]}"
fi
if [ "$SLURM_LOCALID" = 0 ]; then
  echo "NQ node $(hostname) gpus: $(nvidia-smi --query-gpu=index,memory.used,clocks.mem --format=csv,noheader | tr '\n' ' ')"
fi
cd /tmp
exec python3 $D/nq_step.py --init-file $OUT.rdv --capture-sizes "$NQ_CAP" --max-capture $NQ_MAXCAP --overlay "$NQ_OV" -- \
  "${NQ_ARGS[@]}" --shapes $OUT.shapes.json --kv-len-jitter 0 --routing balanced --seed 0 \
  --iters ${NQ_ITERS:-20} --samples ${NQ_SAMPLES:-5} --output-json $OUT.json "$@"
