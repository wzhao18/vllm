#!/bin/bash
# mb lane prefill inner: one rank (srun task) inside the release kmc image (k31vr/kmc:kmr-sweep-freeze-r1).
# usage: inner.sh <rdv file> <bench forward args...>
# env: PF_TREE (vllm tree, default the release draft sweep tree rel-1767cf7), PF_PACKED (1: P1 packed block env),
#      PF_MOE_QUANT (mxfp8|release), K31W0_LAYERS (truncate target), K31P1_* (eager timer / profile knobs)
W=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr; L=$W/mb/prefill
RDV=$1; shift
TREE=${PF_TREE:-$W/dsrel/trees/rel-1767cf7}
export PYTHONPATH=$TREE:/opt/k3/fi-0.7.3.dev20261031:/opt/k3/pydeps
export FLASHINFER_CUBIN_DIR=/opt/k3/fi-cubins-0.7.3.dev20261031
# release e2e caches (dsrel-* renders): FlashInfer JIT / TVM-FFI on Lustre, CuTe pretrace of the kmc image
J=$W/cache/jit-nq/integ-af10ccc2c4
export FLASHINFER_JIT_DIR=$J/flashinfer-0.7.3.dev20261031/jit FLASHINFER_WORKSPACE_BASE=$J/flashinfer-0.7.3.dev20261031
export TVM_FFI_CACHE_DIR=$J/tvmffi
export VLLM_KIMI_K3_CUTE_CACHE_DIR=$W/cache/cute-pretrace/k31vr-kmc-kmr-sweep-freeze-r1
# lane-private build / JIT dirs (never share torch-extension build dirs across trees)
export VLLM_CACHE_ROOT=$L/cache/vllm DG_JIT_CACHE_DIR=$L/cache/deepgemm TORCHINDUCTOR_CACHE_DIR=$L/cache/inductor
export K31_DCPX_BUILD_DIR=$L/cache/build/dcpx K31_DSAC_BUILD_DIR=$L/cache/build/dsac
export CUTE_DSL_CACHE_DIR=$W/cache/jit/cutedsl
export HOME=$W/home XDG_CACHE_HOME=$W/cache/xdg XDG_CONFIG_HOME=$W/cache/xdg-config XDG_DATA_HOME=$W/cache/xdg-data
export HF_HOME=$W/cache/hf PIP_CACHE_DIR=$W/cache/pip UV_CACHE_DIR=$W/cache/uv HF_HUB_OFFLINE=1
export TRITON_CACHE_DIR=/tmp/k31pf-jit/triton CUDA_CACHE_PATH=/tmp/k31pf-jit/nvcache
mkdir -p $VLLM_CACHE_ROOT $DG_JIT_CACHE_DIR $TORCHINDUCTOR_CACHE_DIR $K31_DCPX_BUILD_DIR $K31_DSAC_BUILD_DIR \
  $TRITON_CACHE_DIR $CUDA_CACHE_PATH $HOME 2>/dev/null
# release prefill-role environment (e2e/renders/dsrel-*/render/config.yaml `environment`, engine keys only)
export NCCL_CUMEM_ENABLE=1 NCCL_MNNVL_ENABLE=1 NCCL_NET_PLUGIN=none NCCL_NVLS_ENABLE=1 NCCL_P2P_LEVEL=NVL
export VLLM_ALLREDUCE_USE_FLASHINFER=1 VLLM_SSM_CONV_STATE_LAYOUT=DS VLLM_USE_DIRECT_DCP_A2A=1 \
  VLLM_USE_DIRECT_DCP_KV_GATHER=1 VLLM_USE_DIRECT_DCP_Q_GATHER=1 VLLM_USE_NCCL_SYMM_MEM=0 VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_RPC_TIMEOUT=600000 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 PYTHONUNBUFFERED=1
if [ "${PF_PACKED:-0}" = 1 ]; then export VLLM_KIMI_K3_PACKED_BLOCK_SIZE=1024 VLLM_PACKED_KV_STATE_GROUPS=4; fi
export CUDA_VISIBLE_DEVICES=0,1,2,3
if [ "$SLURM_PROCID" = 0 ]; then
  echo "PF rank0 $(hostname) ntasks $SLURM_NTASKS tree $TREE ($(head -c 80 $TREE/SOURCE_REV)) layers ${K31W0_LAYERS:-all} moe ${PF_MOE_QUANT:-mxfp8}"
  echo "PF env $(env | grep -E '^(VLLM_|NCCL_|K31|PF_)' | sort | tr '\n' ' ')"
fi
if [ "$SLURM_LOCALID" = 0 ]; then
  echo "PF node $(hostname) gpus: $(nvidia-smi --query-gpu=index,memory.used,clocks.max.sm,clocks.mem --format=csv,noheader | tr '\n' ' ')"
fi
cd /tmp
exec python3 $L/tools/pf_bench.py --init-file $RDV -- "$@"
