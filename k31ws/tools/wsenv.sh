#!/bin/bash
# wsenv.sh CMD...: container-side env for k31ws runs (called by wsgpu.sh inside the Rhea image).
# PYTHONPATH: the overlay vllm tree $K/ov/$WS_OV first, then the image's FlashInfer 0.7.3.dev20261031 and pydeps under
# /opt/k3 (as the rhea1n step bench); WS_PYTHONPATH (optional) is prepended, e.g. a FlashInfer checkout on Lustre.
# Caches (FAST 2026-11-18): every JIT cache is persistent on Lustre and shared where its key is content-addressed.
#  * The container /tmp is a fresh tmpfs per srun step, so the old node-local Triton / CUDA caches were cold on EVERY
#    run (all Triton kernels recompiled each time). Triton keys = source + compiler + options hash, written by atomic
#    rename: one shared dir is correct across overlays. CUDA driver JIT cache: per node on Lustre.
#  * Torch-extension builds of the Kimi kernels (dcp_exchange / dsa_core) are named by the digest of their source, so
#    one shared build root serves every overlay (was per overlay: 34 s rebuild on each new overlay's first run).
#    TORCH_EXTENSIONS_DIR (extensions not named by digest) stays per overlay.
#  * FlashInfer JIT: the image's FI shares one cache; a WS_PYTHONPATH checkout gets its own cache (same version string,
#    different sources would rebuild each other under one lock).
K=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws
OVD=$K/ov/${WS_OV:?WS_OV}
export PYTHONPATH=${WS_PYTHONPATH:+$WS_PYTHONPATH:}$OVD:/opt/k3/fi-0.7.3.dev20261031:/opt/k3/pydeps
export HOME=$K/home XDG_CACHE_HOME=$K/cache/xdg XDG_CONFIG_HOME=$K/cache/xdg-config XDG_DATA_HOME=$K/cache/xdg-data
export HF_HOME=$K/cache/hf PIP_CACHE_DIR=$K/cache/pip UV_CACHE_DIR=$K/cache/uv
export CUTE_DSL_CACHE_DIR=$K/cache/jit/cutedsl TORCHINDUCTOR_CACHE_DIR=$K/cache/jit/inductor VLLM_CACHE_ROOT=$K/cache/jit/vllm
export DG_JIT_CACHE_DIR=$K/cache/jit/deepgemm FLASHINFER_WORKSPACE_BASE=$K/cache/fi
if [ -n "$WS_PYTHONPATH" ]; then
  export FLASHINFER_WORKSPACE_BASE=$K/cache/fi-ws/$(echo "$WS_PYTHONPATH" | md5sum | cut -c1-12)
fi
export TRITON_CACHE_DIR=$K/cache/jit/triton CUDA_CACHE_PATH=$K/cache/jit/nvcache/$(hostname -s)
export NQ_BUILD=$K/cache/build/shared TORCH_EXTENSIONS_DIR=$K/cache/build/$WS_OV/torch_ext
export K31_DCPX_BUILD_DIR=${K31_DCPX_BUILD_DIR:-$NQ_BUILD/dcpx} K31_DSAC_BUILD_DIR=${K31_DSAC_BUILD_DIR:-$NQ_BUILD/dsac}
export CUDA_VISIBLE_DEVICES=${WS_GPUS:-0,1,2,3} PYTHONUNBUFFERED=1 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
mkdir -p $HOME $XDG_CACHE_HOME $CUTE_DSL_CACHE_DIR $TORCHINDUCTOR_CACHE_DIR $VLLM_CACHE_ROOT $DG_JIT_CACHE_DIR \
  $FLASHINFER_WORKSPACE_BASE $TRITON_CACHE_DIR $CUDA_CACHE_PATH $NQ_BUILD/dcpx $NQ_BUILD/dsac $TORCH_EXTENSIONS_DIR 2>/dev/null
cd ${WS_CWD:-$OVD}
exec "$@"
