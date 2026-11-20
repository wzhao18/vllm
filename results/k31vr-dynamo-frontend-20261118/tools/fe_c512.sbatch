#!/bin/bash
#SBATCH --job-name=coreai_comparch_inferencex-k31vr-fe8
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --segment=8
#SBATCH --exclusive
#SBATCH --account=coreai_comparch_inferencex
#SBATCH --partition=batch-xdr
#SBATCH --time=04:30:00
#SBATCH --mem=0
#SBATCH --cpus-per-task=352
#SBATCH --output=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/fe/logs/%x-%j.out
#SBATCH --export=ALL,HOME=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/e2e/home,UV_CACHE_DIR=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/e2e/cache/uv-host,XDG_CACHE_HOME=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/e2e/cache/xdg-host,ENROOT_ROOTFS_WRITABLE=1
# fe_c512.sbatch <render name>: the Dynamo-frontend workstream's own allocation (NOT a k31vr-r2hd hold: the dsrel dispatcher
# never sees it). Runs the render with run_in_alloc.sh plus a host-side frontend CPU monitor on every node.
RN=${1:?render}; W=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr; F=$W/fe
M=$F/mon/$RN-$SLURM_JOB_ID; mkdir -p $M
srun --overlap --jobid=$SLURM_JOB_ID -N $SLURM_NNODES --ntasks-per-node=1 -J femon bash $F/fe_mon.sh $M > $M/femon.out 2>&1 &
K31_MIN_LEFT_S=7000 bash $W/e2e/k31r/run_in_alloc.sh $W/e2e/renders/$RN $RN; rc=$?
echo "run_in_alloc rc=$rc"
O=$(ls -td $W/e2e/outputs/a$SLURM_JOB_ID-$RN-* 2>/dev/null | head -n 1); echo "output $O"; echo "$O" > $M/output_dir
exit $rc
