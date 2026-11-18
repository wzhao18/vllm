#!/bin/bash
# wsenv.sh CMD...: container-side env for k31ws runs (called by wsgpu.sh inside the Rhea image).
# PYTHONPATH: the overlay vllm tree $K/ov/$WS_OV first, then the image's FlashInfer 0.7.3.dev20261031 and pydeps under
# /opt/k3 (as the rhea1n step bench); WS_PYTHONPATH (optional) is prepended, e.g. a FlashInfer checkout on Lustre.
# Caches on Lustre (never /home); Triton / CUDA caches node-local. Torch-extension builds per overlay (two overlays must
# never share a build dir).
K=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws
OVD=$K/ov/${WS_OV:?WS_OV}
export PYTHONPATH=${WS_PYTHONPATH:+$WS_PYTHONPATH:}$OVD:/opt/k3/fi-0.7.3.dev20261031:/opt/k3/pydeps
export HOME=$K/home XDG_CACHE_HOME=$K/cache/xdg XDG_CONFIG_HOME=$K/cache/xdg-config XDG_DATA_HOME=$K/cache/xdg-data
export HF_HOME=$K/cache/hf PIP_CACHE_DIR=$K/cache/pip UV_CACHE_DIR=$K/cache/uv
export CUTE_DSL_CACHE_DIR=$K/cache/jit/cutedsl TORCHINDUCTOR_CACHE_DIR=$K/cache/jit/inductor VLLM_CACHE_ROOT=$K/cache/jit/vllm
export DG_JIT_CACHE_DIR=$K/cache/jit/deepgemm FLASHINFER_WORKSPACE_BASE=$K/cache/fi
export TRITON_CACHE_DIR=/tmp/k31ws-jit/triton CUDA_CACHE_PATH=/tmp/k31ws-jit/nvcache
export NQ_BUILD=$K/cache/build/$WS_OV TORCH_EXTENSIONS_DIR=$K/cache/build/$WS_OV/torch_ext
export CUDA_VISIBLE_DEVICES=${WS_GPUS:-0,1,2,3} PYTHONUNBUFFERED=1 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
mkdir -p $HOME $XDG_CACHE_HOME $CUTE_DSL_CACHE_DIR $TORCHINDUCTOR_CACHE_DIR $VLLM_CACHE_ROOT $DG_JIT_CACHE_DIR \
  $FLASHINFER_WORKSPACE_BASE $TRITON_CACHE_DIR $CUDA_CACHE_PATH $NQ_BUILD/dcpx $NQ_BUILD/dsac $TORCH_EXTENSIONS_DIR 2>/dev/null
cd ${WS_CWD:-$OVD}
exec "$@"
