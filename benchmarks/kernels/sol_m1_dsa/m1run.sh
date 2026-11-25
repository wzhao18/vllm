#!/bin/bash
# SOL M1 DSA probes, 1 node (run inside a 1-node pool hold via sol16-n1run.sh). PROBES="sel topn gqk pre lam"
set -a; . /lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/sol16/prod/SOL16_MEASURE.env; set +a  # FAST rule 6
K=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws
IMG=/lustre/fsw/coreai_comparch_inferencex/weizha/images/kimi-k31-wzhao-nov-11-aarch64.sqsh
D=$K/sol16/M1DSA; OV=${OV:-g-8dde40ae0a}
export M1_BUILD=$D/build
run() { WS_OV=$OV srun -N1 -n1 --container-image=$IMG --container-mounts=/lustre:/lustre,$K/ov/$OV:/k31ov \
  --no-container-mount-home --container-workdir=$D --export=ALL,WS_OV=$OV,M1_BUILD=$M1_BUILD \
  bash $K/sol16/tools/sol16-wsenv.sh bash -c "cd $D; $*"; }
echo "M1DSA node $(hostname) ov $OV $(date +%T) probes ${PROBES:=sel topn gqk pre lam}"
for p in $PROBES; do
  echo "== $p $(date +%T)"
  case $p in
    lam) run "torchrun --nproc-per-node 4 m1lam.py";;
    *) run "python3 m1dsa.py $p ${ARGS_EXTRA}";;
  esac
done
echo "=== done $(date +%T)"
