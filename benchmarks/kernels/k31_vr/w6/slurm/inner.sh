#!/bin/bash
# W6 inner: one rank inside the container (K31VR_VLLM_TAG selects the env overlay).
W=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr
RDV=$1; shift
source $W/env/k31vr_env.sh
export NCCL_MNNVL_ENABLE=1 NCCL_CUMEM_ENABLE=1 NCCL_NVLS_ENABLE=1 NCCL_P2P_LEVEL=NVL NCCL_NET_PLUGIN=none
export VLLM_USE_NCCL_SYMM_MEM=0 VLLM_ALLREDUCE_USE_FLASHINFER=1 VLLM_SSM_CONV_STATE_LAYOUT=DS VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_RPC_TIMEOUT=600000 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
if [ "$SLURM_LOCALID" = 0 ]; then
  echo "[w6] $(hostname) node $SLURM_NODEID rank $SLURM_PROCID/$SLURM_NTASKS tag $K31VR_VLLM_TAG layers ${K31W0_LAYERS:-all}"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
fi
cd $W/w0dep/tools
exec python3 bench_dep_step.py --init-file $RDV -- "$@"
