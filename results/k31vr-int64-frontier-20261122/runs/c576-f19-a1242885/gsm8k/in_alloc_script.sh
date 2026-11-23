#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=coreai_cserve_dev-k31e2e.agx-r8-k3-c512-f19-ev
#SBATCH --output=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/e2e/outputs/%j/logs/sweep_%j.log

#SBATCH --nodes=8

#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1



#SBATCH --segment=8


#SBATCH --exclusive

#SBATCH --account=coreai_cserve_dev
#SBATCH --time=04:00:00
#SBATCH --partition=batch


#SBATCH --mem=0



#SBATCH --cpus-per-task=352



#SBATCH --export=ALL,HOME=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/e2e/home,UV_CACHE_DIR=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/e2e/cache/uv-host,XDG_CACHE_HOME=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/e2e/cache/xdg-host,ENROOT_ROOTFS_WRITABLE=1




# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Orchestrator runs on HOST (needs srun access), spawns containerized workers
# Config: /lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/e2e/renders/agx-r8-k3-c512-f19-ev/hybrid.yaml
# Generated: 20261110_182801

set -e

# Setup directories
SRTCTL_SOURCE="/lustre/fsw/coreai_comparch_inferencex/weizha/k3vr/srt-slurm"
OUTPUT_BASE="/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/e2e/outputs"
OUTPUT_DIR="/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/e2e/outputs/a110790-agx-r8-k3-c512-f19-ev-1122082418"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# Rendered by `srtctl render`: nothing submitted this script, so nothing copied the
# recipe into OUTPUT_DIR after sbatch handed back the job id. Stage it here instead.
cp "/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/e2e/renders/agx-r8-k3-c512-f19-ev/render"/*.yaml "${OUTPUT_DIR}/"


# Export for Python orchestrator to use the same paths
export SRTCTL_OUTPUT_DIR="${OUTPUT_DIR}"

# Redirect stderr to stdout (SLURM captures both via --output)
exec 2>&1

echo "=========================================="
echo "🚀 srtctl Sweep Orchestrator"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Config: /lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/e2e/renders/agx-r8-k3-c512-f19-ev/hybrid.yaml"
echo "Nodes: ${SLURM_JOB_NUM_NODES}"
echo "CPUs per node: ${SLURM_JOB_CPUS_PER_NODE:-unknown}"
echo "CPUs on orchestrator node: ${SLURM_CPUS_ON_NODE:-unknown}"
echo "GPUs per node: ${SLURM_GPUS_PER_NODE:-unknown}"
echo "Container: /lustre/fsw/coreai_comparch_inferencex/weizha/k3vr/images/k31vr-kmc_kmr-sweep-freeze_p242fb49cd4-mr4b8c04e28d7_d3c468ef795-kmm7989055e66_fi0.7.3.dev20261031_mc0.3.14_base3f1bcfd82b_r1-aarch64.sqsh"
echo "Start: $(date)"
echo "=========================================="
echo ""

# submit.py always copies the original submission config to OUTPUT_DIR/config.yaml.
# For override jobs it also writes the resolved runtime config below.
echo "Runtime config: ${OUTPUT_DIR}/config.yaml"

# Set source directory for container mounts (/configs)
export SRTCTL_SOURCE_DIR="${SRTCTL_SOURCE}"


export W1_MC_OVERLAY=/opt/k3/mooncake-0.3.14
export UCX_GPU_NIC_ASSIGNMENT_MODE=off
export DYN_TCP_REQUEST_TIMEOUT=120
export DYN_REQUEST_PLANE=tcp
export ETCD_LEASE_TTL=600
export VLLM_USE_DIRECT_DCP_A2A=1
export VLLM_USE_DIRECT_DCP_Q_GATHER=1
export VLLM_USE_DIRECT_DCP_KV_GATHER=1
export VLLM_ALLREDUCE_USE_FLASHINFER=1
export VLLM_RPC_TIMEOUT=600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_USE_RUST_FRONTEND=1
export VLLM_SERVER_DEV_MODE=1
export VLLM_ENGINE_READY_TIMEOUT_S=5400
export VLLM_MOONCAKE_LOAD_RECV_THREADS=4
export MC_SLICE_SIZE=1048576
export MC_MAX_MR_SIZE=4294967296
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export WITH_NVIDIA_PEERMEM=0
export NCCL_MNNVL_ENABLE=1
export NCCL_CUMEM_ENABLE=1
export NCCL_NVLS_ENABLE=1
export NCCL_P2P_LEVEL=NVL
export NCCL_NET_PLUGIN=none
export VLLM_USE_NCCL_SYMM_MEM=0
export UCX_TLS=rc,cuda_copy
export UCX_NET_DEVICES=roce_vf0_r1:1,roce_vf0_r2:1,roce_vf0_r3:1,roce_vf0_r4:1,roce_vf1_r1:1,roce_vf1_r2:1,roce_vf1_r3:1,roce_vf1_r4:1
export UCX_MEMTYPE_CACHE=n
export UCX_MEMTYPE_REG_WHOLE=n
export UCX_RCACHE_MAX_UNRELEASED=1024
export UCX_TCP_AF_PRIO=inet
export VLLM_SSM_CONV_STATE_LAYOUT=DS
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TRITON_CACHE_DIR=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/cache/jit/triton
export CUDA_CACHE_PATH=/raid/scratch/k31ws-agx/nvcache
export CUDA_CACHE_MAXSIZE=4294967296
export VLLM_CACHE_ROOT=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/cache/jit/vllm
export CUTE_DSL_CACHE_DIR=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/cache/jit/cutedsl
export TORCHINDUCTOR_CACHE_DIR=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/cache/jit/inductor
export FLASHINFER_WORKSPACE_BASE=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/cache/fi
export DG_JIT_CACHE_DIR=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/cache/jit/deepgemm
export FLASHINFER_CUBIN_DIR=/opt/k3/fi-cubins-0.7.3.dev20261031
export TVM_FFI_CACHE_DIR=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/cache/jit/tvmffi
export VLLM_ENABLE_CUDA_COMPATIBILITY=0
export TORCH_EXTENSIONS_DIR=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/cache/build/g-15c281dc32/torch_ext
export NQ_BUILD=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/cache/build/shared
export K31_DCPX_BUILD_DIR=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/cache/build/shared/dcpx
export K31_DSAC_BUILD_DIR=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/cache/build/shared/dsac
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export UCX_IB_GID_INDEX=3
export UCX_IB_TRAFFIC_CLASS=96
export UCX_IB_SL=1
export UCX_IB_ROCE_LOCAL_SUBNET=y
export UCX_IB_ROCE_SUBNET_PREFIX_LEN=11
export UCX_MAX_RNDV_RAILS=2















































echo ""
echo "Preparing srtctl environment..."

# Use compute-arch uv from srt-slurm/bin (installed by make setup ARCH=<compute_arch>)
# This avoids arch mismatch when login node (x86_64) != compute node (aarch64)
export PATH="${SRTCTL_SOURCE}/bin:$HOME/.local/bin:$PATH"

# Use a compute-specific venv so we don't pick up the login node's .venv/
# (which may be built for a different arch, e.g. x86_64 login vs aarch64 compute)
export UV_PROJECT_ENVIRONMENT="${SRTCTL_SOURCE}/.venv-compute"

if ! command -v uv &> /dev/null; then
    echo "ERROR: uv not found. Run 'make setup ARCH=<compute_arch>' first."
    echo "  e.g., make setup ARCH=aarch64  (for GB200/Grace compute nodes)"
    exit 1
fi

echo "Using uv $(uv --version)..."

if [ -n "${DYNAMO_WHEEL_NAME:-}" ] || [ "${SRTCTL_PREFETCH_AI_DYNAMO:-0}" = "1" ]; then
    if [ -f "${SRTCTL_SOURCE}/src/srtctl/runtime_scripts/dynamo_wheels.py" ]; then
        uv run --no-project --python "${DYNAMO_PYTHON_VERSION:-3.12}" --with pip \
            python "${SRTCTL_SOURCE}/src/srtctl/runtime_scripts/dynamo_wheels.py" prefetch
    else
        echo "WARNING: ${SRTCTL_SOURCE}/src/srtctl/runtime_scripts/dynamo_wheels.py not found"
    fi
fi



echo "Running orchestrator on host (with srun access)..."
echo ""

# Run orchestrator on the HOST (not in container) so it has access to srun
# The orchestrator will spawn containerized workers
# Output goes to SLURM's --output (sweep_%j.log)
set +e  # Temporarily disable exit-on-error to capture exit code
cd "${SRTCTL_SOURCE}"
# Lock-protected venv sync to avoid concurrent .venv-compute teardown/recreate
# races across simultaneous SLURM jobs sharing this directory. uv may decide
# the existing venv is invalid (broken python interpreter symlink) and tear
# it down + rebuild; if N jobs do that at once they stomp each other with
# "Directory not empty" / "Failed to spawn 'python -m'" / missing CACHEDIR.TAG
# errors. The flock serializes only the venv init; the lock is released
# before the actual sweep so jobs still run in parallel.
flock -x "${SRTCTL_SOURCE}/.venv-compute.lock" uv sync --python 3.12 --no-dev
uv run --python 3.12 --no-dev --no-sync -m srtctl.cli.do_sweep "${OUTPUT_DIR}/config.yaml"
EXIT_CODE=$?
set -e  # Re-enable exit-on-error

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Sweep completed successfully"
else
    echo "✗ Sweep failed (exit code: $EXIT_CODE)"
fi
echo "End: $(date)"
echo "=========================================="

exit ${EXIT_CODE}