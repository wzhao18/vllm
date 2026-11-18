#!/bin/bash
#SBATCH --job-name=coreai_comparch_inferencex-k31vr.feprof
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --account=coreai_comparch_inferencex
#SBATCH --partition=batch-spx,batch-xdr
#SBATCH --time=01:30:00
#SBATCH --mem=0
#SBATCH --output=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/fe/logs/%x-%j.out
#SBATCH --export=ALL,HOME=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/e2e/home,XDG_CACHE_HOME=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/e2e/cache/xdg-host,ENROOT_ROOTFS_WRITABLE=1
# CPU-only offline replay of the frontend preprocessing path (no GPU use). Args: <tag> [extra prof_fe.py args]
TAG=${1:?tag}; shift
F=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/fe
IMG=/lustre/fsw/coreai_comparch_inferencex/weizha/k3vr/images/k31vr-kmc_kmr-sweep-freeze_p242fb49cd4-mr4b8c04e28d7_d3c468ef795-kmm7989055e66_fi0.7.3.dev20261031_mc0.3.14_base3f1bcfd82b_r1-aarch64.sqsh
srun --container-image=$IMG --no-container-mount-home \
  --container-mounts=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/e2e/model-view/kimi-k3.1-pro-k3type:/model:ro,/lustre/fsw/coreai_comparch_inferencex/weizha/models/kimi-k3.1-pro-release-d9e99b07:/lustre/fsw/coreai_comparch_inferencex/weizha/models/kimi-k3.1-pro-release-d9e99b07:ro,/lustre/fsw/coreai_comparch_inferencex/weizha/k3vr/aiperf-cache:/aiperf_mmap_cache:ro,$F:$F \
  --export=ALL,CUDA_VISIBLE_DEVICES=,PYTHONPATH=/opt/k3/ov-fe:/opt/k3/fi-0.7.3.dev20261031:/opt/k3/pydeps:/opt/k3/dynamo,HF_HUB_OFFLINE=1,VLLM_K3_FAST_SPLIT=1,VLLM_K3_TOKCACHE_BYTES=512M,VLLM_K3_SKIP_ID_VALIDATION=1,VLLM_K3_BYTE_DETOK=1,VLLM_USE_RUST_FRONTEND=1 \
  python3 $F/prof_fe.py --cache /aiperf_mmap_cache/ae5de539bb1b8a5e5e8d10bd8adb18fb --model /model --out $F/prof-$TAG-$SLURM_JOB_ID.json "$@"
