#!/bin/bash
# mkov.sh <worktree> <tag>  (run on the Mac): stage a checkout as overlay $K/ov/<tag> on BOTH clusters (Rhea and
# Hecate; same path, separate filesystems) for wsgpu.sh / wsrun.sh -o <tag>.
# mkov.sh -c rhea|hecate <worktree> <tag>: stage on one cluster only.
# mkov.sh --copy <tag> [rhea|hecate]: mirror an EXISTING overlay from Rhea to Hecate (default; or the reverse when the
#   argument is rhea), e.g. an INFRA trial overlay you want to A/B on Hecate. Only files that differ from base-4fbaa15
#   travel (rsync quick check against a hard-linked base on both ends, relayed through the Mac).
# The overlay = base-4fbaa15 (the image's vllm tree with wzhao/k31-sp-gemm 4fbaa15189 shipped over it, build artifacts
# kept), hard-linked (cp -al: no copy, the base is never written), plus every vllm/ file of the checkout that differs
# from 4fbaa15189 (committed, staged, unstaged, untracked; deletions removed) and the checkout's tests/ and benchmarks/.
# rsync replaces shipped files by new inodes (temp + rename), so the hard-linked base is untouched.
# Tags are immutable: a cluster that already has the tag is skipped (stage a new tag, e.g. <ws>-<shortrev>-<n>).
# Hecate has no route to Rhea: every byte is relayed by the Mac (XC 2026-11-18).
set -eo pipefail
K=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws; BASE=$K/ov/base-4fbaa15; BASEREV=4fbaa15189
SSH="ssh -o BatchMode=yes"
q() { grep -v -E "post-quantum|store now|openssh.com/pq" >&2 || true; }  # drop the login nodes' kex warning
CLUSTERS="rhea hecate"
if [ "$1" = --copy ]; then
  TAG=$2; TO=${3:-hecate}; FROM=rhea; [ $TO = rhea ] && FROM=hecate
  [[ $TAG =~ ^[A-Za-z0-9._-]+$ ]] || { echo "usage: mkov.sh --copy <tag> [rhea|hecate]"; exit 2; }
  $SSH $FROM "[ -d $K/ov/$TAG ]" 2> >(q) || { echo "no overlay $TAG on $FROM"; exit 3; }
  $SSH $TO "[ ! -e $K/ov/$TAG ]" 2> >(q) || { echo "overlay $TAG already on $TO"; exit 0; }
  LB=${MKOV_LOCAL_BASE:-/Users/weizha/k31ws-xc/base-4fbaa15}   # Mac mirror of base-4fbaa15 (rsync -a from Rhea)
  [ -d $LB/vllm ] || { echo "no local base mirror $LB (rsync -a rhea:$BASE $(dirname $LB)/)"; exit 2; }
  TMP=$(mktemp -d); trap 'rm -rf $TMP' EXIT
  # pull: files equal to the base (same size + mtime: cp -al keeps them) are hard-linked from the local base mirror
  rsync -a --delete --link-dest=$LB $FROM:$K/ov/$TAG/ $TMP/ov/ 2> >(q)
  P=$K/ov/.$TAG.partial
  $SSH $TO "rm -rf $P && cp -al $BASE $P" 2> >(q)
  # push: unchanged files are skipped by the quick check; changed ones replace the hard links with new inodes
  rsync -a --delete $TMP/ov/ $TO:$P/ 2> >(q)
  $SSH $TO "[ ! -e $K/ov/$TAG ] && mv -T $P $K/ov/$TAG" 2> >(q)
  echo "copied overlay $TAG $FROM -> $TO"; exit 0
fi
if [ "$1" = -c ]; then CLUSTERS=$2; shift 2; fi
SRC=$(cd "$1" && pwd); TAG=$2
[[ $TAG =~ ^[A-Za-z0-9._-]+$ ]] || { echo "usage: mkov.sh [-c rhea|hecate] <worktree> <tag> | --copy <tag> [rhea|hecate]"; exit 2; }
git -C $SRC cat-file -e $BASEREV^{commit}
TMP=$(mktemp -d); trap 'rm -rf $TMP' EXIT
git -C $SRC diff --name-status --no-renames $BASEREV -- vllm | awk '{print ($1 == "T" ? "M" : $1), $2}' > $TMP/st
git -C $SRC ls-files --others --exclude-standard -- vllm | sed 's/^/A /' >> $TMP/st
sort -u -k2 $TMP/st -o $TMP/st
awk '$1 != "D" {print $2}' $TMP/st > $TMP/ship; awk '$1 == "D" {print $2}' $TMP/st > $TMP/del
{ echo "$(git -C $SRC rev-parse --short=10 HEAD) $(git -C $SRC rev-parse --abbrev-ref HEAD) + $(wc -l < $TMP/st | tr -d ' ') vllm/ files vs $BASEREV over base-4fbaa15, staged $(date +%FT%T)"
  cat $TMP/st; } > $TMP/SOURCE_REV
stage() {  # stage <cluster host>
  local H=$1 P=$K/ov/.$TAG.partial
  $SSH $H "[ ! -e $K/ov/$TAG ]" 2> >(q) || { echo "$H: overlay $TAG exists, skipped"; return 0; }
  $SSH $H "rm -rf $P && cp -al $BASE $P" 2> >(q)
  [ -s $TMP/ship ] && rsync -a --checksum --files-from=$TMP/ship $SRC/ $H:$P/ 2> >(q)
  [ -s $TMP/del ] && $SSH $H "cd $P && xargs -r rm -f" < $TMP/del 2> >(q)
  rsync -a --delete --exclude __pycache__ --exclude '*.pyc' $SRC/tests/ $H:$P/tests/ 2> >(q)
  rsync -a --delete --exclude __pycache__ --exclude '*.pyc' $SRC/benchmarks/ $H:$P/benchmarks/ 2> >(q)
  rsync -a $TMP/SOURCE_REV $H:$P/SOURCE_REV.ws 2> >(q)
  $SSH $H "[ ! -e $K/ov/$TAG ] && mv -T $P $K/ov/$TAG" 2> >(q)
  echo "$H: staged $K/ov/$TAG: $(wc -l < $TMP/ship | tr -d ' ') shipped, $(wc -l < $TMP/del | tr -d ' ') deleted"
}
rc=0
for H in $CLUSTERS; do stage $H || { echo "$H: staging FAILED"; rc=1; }; done
exit $rc
