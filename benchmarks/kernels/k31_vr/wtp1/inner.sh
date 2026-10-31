#!/bin/bash
# wtp1 inner (copy of d4x4): one rank (srun task) inside the K3 VR image container.
# usage: inner.sh <rdv file> <bench forward args...>   (env: TAG, D4_PTS=1 for prims_ts on SM107, DCPX=0|1|auto)
W=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr
RDV=$1; shift
export K31VR_VLLM_TAG=${TAG:?TAG}
if [ "${D4_PTS:-0}" = 1 ]; then export K31VR_EXTRA_PYTHONPATH=$W/w2/site:$W/env K31VR_FORCE_PRIMS_TS_SM107=1; fi
source $W/env/k31vr_env.sh
export NCCL_MNNVL_ENABLE=1 NCCL_CUMEM_ENABLE=1 NCCL_NVLS_ENABLE=1 NCCL_P2P_LEVEL=NVL NCCL_NET_PLUGIN=none
export VLLM_USE_DIRECT_DCP_A2A=1 VLLM_USE_DIRECT_DCP_Q_GATHER=1 VLLM_USE_DIRECT_DCP_KV_GATHER=1 VLLM_USE_NCCL_SYMM_MEM=0 VLLM_ALLREDUCE_USE_FLASHINFER=1 VLLM_SSM_CONV_STATE_LAYOUT=DS VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_K31_DCP_EXCHANGE=${DCPX:-0} VLLM_K31_DCP_EXCHANGE_GRAPH_ONLY=${VLLM_K31_DCP_EXCHANGE_GRAPH_ONLY:-0} K31_DCPX_BUILD_DIR=$W/cache/jit/dcpx
export VLLM_RPC_TIMEOUT=600000 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
if [ "$SLURM_LOCALID" = 0 ]; then
  echo "[wtp1] $(hostname) node $SLURM_NODEID rank $SLURM_PROCID/$SLURM_NTASKS tag $K31VR_VLLM_TAG pts ${D4_PTS:-0} dcpx $VLLM_K31_DCP_EXCHANGE layers ${K31W0_LAYERS:-all}"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; echo
fi
cd $W/wtp1/tools
exec python3 bench_dep_step.py --init-file $RDV -- "$@"
