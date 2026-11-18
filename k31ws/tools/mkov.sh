#!/bin/bash
# mkov.sh <worktree> <tag>  (run on the Mac): stage a checkout as overlay $K/ov/<tag> on Rhea for wsgpu.sh -o <tag>.
# The overlay = base-4fbaa15 (the image's vllm tree with wzhao/k31-sp-gemm 4fbaa15189 shipped over it, build artifacts
# kept), hard-linked (cp -al: no copy, the base is never written), plus every vllm/ file of the checkout that differs
# from 4fbaa15189 (committed, staged, unstaged, untracked; deletions removed) and the checkout's tests/ and benchmarks/.
# rsync replaces shipped files by new inodes (temp + rename), so the hard-linked base is untouched.
# Tags are immutable: an existing tag is refused (stage a new one, e.g. <ws>-<shortrev>-<n>).
set -eo pipefail
SRC=$(cd "$1" && pwd); TAG=$2; BASEREV=4fbaa15189
K=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws; BASE=$K/ov/base-4fbaa15
[[ $TAG =~ ^[A-Za-z0-9._-]+$ ]] || { echo "usage: mkov.sh <worktree> <tag>"; exit 2; }
git -C $SRC cat-file -e $BASEREV^{commit}
ssh -o BatchMode=yes rhea "[ ! -e $K/ov/$TAG ]" || { echo "overlay $TAG exists; use a new tag"; exit 3; }
TMP=$(mktemp -d); trap 'rm -rf $TMP' EXIT
git -C $SRC diff --name-status --no-renames $BASEREV -- vllm | awk '{print ($1 == "T" ? "M" : $1), $2}' > $TMP/st
git -C $SRC ls-files --others --exclude-standard -- vllm | sed 's/^/A /' >> $TMP/st
sort -u -k2 $TMP/st -o $TMP/st
awk '$1 != "D" {print $2}' $TMP/st > $TMP/ship; awk '$1 == "D" {print $2}' $TMP/st > $TMP/del
P=$K/ov/.$TAG.partial
ssh -o BatchMode=yes rhea "rm -rf $P && cp -al $BASE $P"
[ -s $TMP/ship ] && rsync -a --checksum --files-from=$TMP/ship $SRC/ rhea:$P/
[ -s $TMP/del ] && ssh -o BatchMode=yes rhea "cd $P && xargs -r rm -f" < $TMP/del
rsync -a --delete --exclude __pycache__ --exclude '*.pyc' $SRC/tests/ rhea:$P/tests/
rsync -a --delete --exclude __pycache__ --exclude '*.pyc' $SRC/benchmarks/ rhea:$P/benchmarks/
{ echo "$(git -C $SRC rev-parse --short=10 HEAD) $(git -C $SRC rev-parse --abbrev-ref HEAD) + $(wc -l < $TMP/st | tr -d ' ') vllm/ files vs $BASEREV over base-4fbaa15, staged $(date +%FT%T)"
  cat $TMP/st; } > $TMP/SOURCE_REV
rsync -a $TMP/SOURCE_REV rhea:$P/SOURCE_REV.ws
ssh -o BatchMode=yes rhea "[ ! -e $K/ov/$TAG ] && mv -T $P $K/ov/$TAG"
echo "staged $K/ov/$TAG: $(wc -l < $TMP/ship | tr -d ' ') shipped, $(wc -l < $TMP/del | tr -d ' ') deleted"
