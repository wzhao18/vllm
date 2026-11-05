#!/bin/bash
# push_tools.sh: copy this directory to $NB/tools on Hecate (the block tasks run the copy) and refresh the JSON snapshot
# of the nq decode configs ($NB/cfg; batch hosts need no PyYAML; cfg2bench refuses a snapshot whose yaml changed).
set -e
NB=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/nqbench
D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ssh hecate "mkdir -p $NB/tools $NB/cfg $NB/results $NB/locks $NB/ov $NB/tasks"
rsync -a --exclude __pycache__ $D/ hecate:$NB/tools/
ssh hecate "python3 $NB/tools/cfg2bench.py --snapshot $NB/cfg"
