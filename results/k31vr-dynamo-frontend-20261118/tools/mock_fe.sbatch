#!/bin/bash
#SBATCH --job-name=coreai_comparch_inferencex-k31vr.femock
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --account=coreai_comparch_inferencex
#SBATCH --partition=batch-spx,batch-xdr
#SBATCH --time=02:00:00
#SBATCH --mem=0
#SBATCH --output=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/fe/logs/%x-%j.out
#SBATCH --export=ALL,HOME=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/e2e/home,XDG_CACHE_HOME=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/e2e/cache/xdg-host,ENROOT_ROOTFS_WRITABLE=1
# mock_fe.sbatch <tag> <n_frontends> <conc> <duration_s> <speedup> [pyspy]: frontend capacity harness (CPU only; GPUs unused).
TAG=${1:?}; NF=${2:?}; CONC=${3:?}; DUR=${4:?}; SPD=${5:?}; PYSPY=${6:-1}
F=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/fe; O=$F/mock/$TAG-$SLURM_JOB_ID; mkdir -p $O
IMG=/lustre/fsw/coreai_comparch_inferencex/weizha/k3vr/images/k31vr-kmc_kmr-sweep-freeze_p242fb49cd4-mr4b8c04e28d7_d3c468ef795-kmm7989055e66_fi0.7.3.dev20261031_mc0.3.14_base3f1bcfd82b_r1-aarch64.sqsh
MNT=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/e2e/model-view/kimi-k3.1-pro-k3type:/model:ro,/lustre/fsw/coreai_comparch_inferencex/weizha/models/kimi-k3.1-pro-release-d9e99b07:/lustre/fsw/coreai_comparch_inferencex/weizha/models/kimi-k3.1-pro-release-d9e99b07:ro,/lustre/fsw/coreai_comparch_inferencex/weizha/k3vr/aiperf-cache:/aiperf_mmap_cache,/lustre/fsw/coreai_comparch_inferencex/weizha/k3vr/hf-hub:/hf_hub_cache,/lustre/fsw/coreai_comparch_inferencex/weizha/k3vr/InferenceX-9a132463c/inferencex-e2e:/infmax-workspace,/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr:/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr
if (( NF > 1 )); then
  awk -v n=$NF '/__SERVERS__/{for(i=0;i<n;i++) printf "        server 127.0.0.1:%d;\n", 8100+i; next} {print}' $F/nginx_mock.conf.tmpl > $O/nginx.conf
  srun --overlap -N1 -n1 -J nginx --container-image=$F/images/nginx_1.27.4-aarch64.sqsh --container-mounts=$O:/ngx --no-container-mount-home \
    nginx -c /ngx/nginx.conf -g "daemon off;" > $O/nginx.out 2>&1 &
fi
srun --overlap -N1 -n1 -J mfe-mon bash $F/fe_mon.sh $O > /dev/null 2>&1 &
srun -N1 -n1 -J femock --container-image=$IMG --container-mounts=$MNT --no-container-mount-home \
  --export=ALL,PYSPY_BIN=$F/tools/py-spy,AGX=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/jh/noquant/cfg2/agentx-runtime/nq4-c512/venv/bin/agentx \
  bash $F/mock_inner.sh $O $NF $CONC $DUR $SPD $PYSPY
echo "done $O"
