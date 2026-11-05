#!/bin/bash
# stage_overlay.sh <checkout> <ovtag>: (run where the checkout is, with ssh hecate) stage the checkout as the code under
# test in $NB/ov/<ovtag>: a whole vllm tree = the base env $BASE (vllm-jh-242fb49cd4, hard-linked: no copy, the base is
# never written) with EVERY file that differs from 242fb49cd4 under vllm/ shipped over it (modified, added, untracked;
# deleted files removed), plus the checkout's tests/ and pyproject.toml (the GPU correctness stage, nq_gputest.sh).
# SOURCE_REV records the revision and the list ('M|A|D <path>' under vllm/, 'T <path>' changed tests, 'NOT-SHIPPED
# <path>' changes outside vllm/ and tests/ that need a build, e.g. csrc/). An existing <ovtag> is refused (a block may
# be running on it: stage a new tag). Then renders the queue task templates for it into $NB/tasks/ (staging a task into
# $W/queue/pending is the user's step: cp $NB/tasks/<task>.sh $W/queue/pending/).
set -eo pipefail
SRC=$(cd "$1" && pwd); TAG=$2
[ -n "$TAG" ] || { echo "usage: stage_overlay.sh <checkout> <ovtag>"; exit 2; }
[[ $TAG =~ ^[A-Za-z0-9._-]+$ ]] || { echo "bad ovtag $TAG"; exit 2; }
W=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr; NB=$W/nqbench
BASE=$W/env/vllm-jh-242fb49cd4; BASEREV=242fb49cd4
D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OV=$NB/ov/$TAG; P=$NB/ov/.$TAG.partial
git -C $SRC cat-file -e $BASEREV^{commit}
ssh hecate "[ ! -e $OV ]" || { echo "overlay $OV exists: refusing to overwrite (a block may run on it); use a new tag"; exit 3; }
ssh hecate "head -c 10 $BASE/SOURCE_REV" | grep -qx $BASEREV || { echo "base env $BASE is not $BASEREV"; exit 3; }
TMP=$(mktemp -d); trap 'rm -rf $TMP' EXIT
# every path that differs from the base revision (committed since, staged, unstaged, untracked; no rename pairing)
git -C $SRC diff --name-status --no-renames $BASEREV -- vllm | awk '{print ($1 == "T" ? "M" : $1), $2}' > $TMP/vllm.st
git -C $SRC ls-files --others --exclude-standard -- vllm | sed 's/^/A /' >> $TMP/vllm.st
sort -u -k2 $TMP/vllm.st -o $TMP/vllm.st
awk '$1 != "D" {print $2}' $TMP/vllm.st > $TMP/ship.txt
awk '$1 == "D" {print $2}' $TMP/vllm.st > $TMP/del.txt
{ git -C $SRC diff --name-only --no-renames $BASEREV -- tests; git -C $SRC ls-files --others --exclude-standard -- tests; } \
  | sort -u > $TMP/tests.txt
{ git -C $SRC diff --name-only --no-renames $BASEREV -- . ':!vllm' ':!tests' ':!benchmarks/kernels/k31_vr'
  git -C $SRC ls-files --others --exclude-standard -- . ':!vllm' ':!tests' ':!benchmarks/kernels/k31_vr'; } \
  | sort -u > $TMP/notshipped.txt
REV="$(git -C $SRC rev-parse --short=10 HEAD) $(git -C $SRC rev-parse --abbrev-ref HEAD)"
{
  echo "$REV + $(wc -l < $TMP/vllm.st | tr -d ' ') vllm/ files differing from $BASEREV (base $BASE), staged $(date +%FT%T)"
  cat $TMP/vllm.st
  sed 's/^/T /' $TMP/tests.txt
  sed 's/^/NOT-SHIPPED /' $TMP/notshipped.txt
} > $TMP/SOURCE_REV
if [ -s $TMP/notshipped.txt ]; then
  echo "WARNING: changes outside vllm/ and tests/ are NOT in the overlay (need a build):"; sed 's/^/  /' $TMP/notshipped.txt
fi
# base tree, hard-linked (rsync replaces a shipped file by a new inode: temp file + rename, never --inplace)
ssh hecate "rm -rf $P && mkdir -p $NB/ov $NB/tasks && rsync -a --link-dest=$BASE/ --exclude __pycache__ --exclude SOURCE_REV $BASE/ $P/"
[ -s $TMP/ship.txt ] && rsync -a --checksum --files-from=$TMP/ship.txt $SRC/ hecate:$P/
[ -s $TMP/del.txt ] && ssh hecate "cd $P && xargs -r rm -f" < $TMP/del.txt
rsync -a --delete --exclude __pycache__ --exclude '*.pyc' $SRC/tests/ hecate:$P/tests/
rsync -a $SRC/pyproject.toml hecate:$P/pyproject.toml
cat $TMP/SOURCE_REV | ssh hecate "cat > $P/SOURCE_REV"
# verify: every shipped file matches the checkout, every deleted file is gone, the base is untouched
MD5="md5sum"; command -v md5sum > /dev/null || MD5="md5 -r"
(cd $SRC && xargs $MD5 < $TMP/ship.txt) | awk '{print $1, $2}' | sort -k2 > $TMP/local.md5
ssh hecate "cd $P && xargs -r md5sum" < $TMP/ship.txt | awk '{print $1, $2}' | sort -k2 > $TMP/remote.md5
cmp -s $TMP/local.md5 $TMP/remote.md5 || { echo "shipped files differ from the checkout:"; diff $TMP/local.md5 $TMP/remote.md5; exit 4; }
if [ -s $TMP/del.txt ]; then
  ssh hecate "cd $P && for f in \$(cat); do [ -e \$f ] && echo \"still present: \$f\"; done; true" < $TMP/del.txt | tee $TMP/left.txt
  [ -s $TMP/left.txt ] && exit 4
fi
ssh hecate "[ ! -e $OV ] && mv -T $P $OV" || { echo "overlay $OV appeared while staging: left $P"; exit 3; }
echo "staged $OV: $(wc -l < $TMP/ship.txt | tr -d ' ') shipped, $(wc -l < $TMP/del.txt | tr -d ' ') deleted, tests/ $(wc -l < $TMP/tests.txt | tr -d ' ') changed"
for t in $D/tasks/*.sh; do
  n=$(basename $t .sh)-$TAG.sh
  sed "s/__OVTAG__/$TAG/g" $t | ssh hecate "cat > $NB/tasks/$n"
  echo "rendered $NB/tasks/$n"
done
ssh hecate "cat $OV/SOURCE_REV"
