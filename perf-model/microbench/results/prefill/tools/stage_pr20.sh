#!/bin/bash
# stage the PR #20 checkout (juntianl/k31vr-tp-dsadp 83cd6d8e12) as a whole vllm tree on Hecate:
# base env vllm-jh-242fb49cd4 hard-linked + every vllm/ file differing from 242fb49cd4 shipped over it (as stage_overlay.sh)
set -eo pipefail
SRC=$1; OVN=$2
W=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr; BASE=$W/env/vllm-jh-242fb49cd4; OV=$W/mb/prefill/ov/$OVN
TMP=$(mktemp -d)
git -C $SRC diff --name-status --no-renames 242fb49cd4 -- vllm | awk '{print ($1 == "T" ? "M" : $1), $2}' > $TMP/st
awk '$1 != "D" {print $2}' $TMP/st > $TMP/ship.txt
awk '$1 == "D" {print $2}' $TMP/st > $TMP/del.txt
{ echo "$(git -C $SRC rev-parse --short=10 HEAD) juntianl/k31vr-tp-dsadp (PR #20) + $(wc -l < $TMP/st | tr -d ' ') vllm/ files differing from 242fb49cd4 (base $BASE), staged $(date +%FT%T)"; cat $TMP/st; } > $TMP/SOURCE_REV
ssh -o BatchMode=yes hecate "mkdir -p $W/mb/prefill/ov && [ ! -e $OV ] && rsync -a --link-dest=$BASE/ --exclude __pycache__ $BASE/ $OV/"
rsync -a --checksum --files-from=$TMP/ship.txt $SRC/ hecate:$OV/
[ -s $TMP/del.txt ] && ssh -o BatchMode=yes hecate "cd $OV && xargs -r rm -f" < $TMP/del.txt
scp -q $TMP/SOURCE_REV hecate:$OV/SOURCE_REV
echo "staged $OV: $(wc -l < $TMP/ship.txt) shipped, $(wc -l < $TMP/del.txt) deleted"
rm -rf $TMP
