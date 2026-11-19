#!/bin/bash
#SBATCH --exclusive
# holdka.sh: batch script of a k31ws pool hold on Hecate (XC 2026-11-18; submitted by holdd-ws.sh). Hecate's site
# reaper cancels allocations whose GPUs look idle, so every node of the hold runs one persistent wska.py step
# (an idle guard that never uses CUDA itself; it starts a GEMM burner only while no wsgpu.sh run uses the node and
# kills it the moment one does). Steps that die are restarted. The script never exits: the hold ends at its time
# limit (or by Slurm preemption on backfill). Never edit this file in place (running jobs read it): write + mv.
K=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws
IMG=/lustre/fsw/coreai_comparch_inferencex/weizha/images/kimi-k31-wzhao-nov-11-aarch64.sqsh
J=$SLURM_JOB_ID; ALL=($(scontrol show hostnames $SLURM_JOB_NODELIST)); S=$K/pool/ka/$J; mkdir -p $S
echo "holdka $J $SLURM_JOB_NAME ${ALL[*]} start $(date +%FT%T)"
while :; do
  for h in "${ALL[@]}"; do
    [ -f $S/$h.pid ] && kill -0 $(cat $S/$h.pid) 2>/dev/null && continue
    srun --overlap --jobid=$J -N1 -n1 -w $h -J wska --container-image=$IMG --container-mounts=/lustre:/lustre \
      --no-container-mount-home --export=ALL,HOME=$K/home python3 $K/tools/wska.py >> $S/$h.log 2>&1 < /dev/null &
    echo $! > $S/$h.pid
  done
  sleep 60
done
