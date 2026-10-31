#!/bin/bash
# usage: push_env.sh <tag> -- self-contained env $W/env/vllm-w2dmf-<tag> (no dir symlinks; compiled files hard-linked from
# vllm-98c712e55; every git-tracked + untracked vllm/ file from this worktree). Pushes tests/benchmarks to $W/w2dmf/src-<tag>, tools to $W/w2dmf/.
set -e
TAG=$1; WT=/Users/weizha/vllm/rainhollow-k31vr-w2-dep-moe-floor
W=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr
SP=/private/tmp/claude-501/-Users-weizha/8fd11ba1-2bcc-4054-91f7-43fb556afe02/scratchpad
HS=$SP/hs; TL=$SP/k31vr/ledger/w2-dep-moe-floor/tools
cd $WT
CS=$(git diff --name-only 98c712e55 -- csrc/ CMakeLists.txt cmake/ setup.py)
[ -z "$CS" ] || { echo "csrc changed since 98c712e55: rebuild needed: $CS"; exit 1; }
PYF=$( (git ls-files vllm; git ls-files --others --exclude-standard vllm) | sort -u)
SHA=$(git rev-parse --short=9 HEAD); DIRTY=$(git status --porcelain vllm | wc -l | tr -d ' ')
COPYFILE_DISABLE=1 tar --no-xattrs -czf - $PYF tests/__init__.py tests/conftest.py tests/models/__init__.py tests/utils.py tests/models/kimi_k3 tests/kernels/quantization benchmarks/kernels/k31_vr 2>/dev/null | \
$HS "set -e; R=$W/env/vllm-98c712e55; E=$W/env/vllm-w2dmf-$TAG; S=$W/w2dmf/src-$TAG; mkdir -p $W/w2dmf/results $W/w2dmf/logs; rm -rf \$S; mkdir -p \$S; cd \$S; tar xzf - 2>/dev/null
 if [ -e \$E ]; then rm -rf \$E.old; mv \$E \$E.old; fi; mkdir -p \$E
 ( cd \$R/vllm && find -L . -type d -name __pycache__ -prune -o -type d -print ) | (cd \$E && mkdir -p vllm && cd vllm && xargs mkdir -p)
 ( cd \$R/vllm && find -L . -name __pycache__ -prune -o -type f -print ) | while read f; do ln \$(readlink -f \$R/vllm/\$f) \$E/vllm/\$f; done
 cd \$S && find vllm -type f | tar cf - --no-recursion -T - | (cd \$E && tar xf - --unlink-first)
 [ -e \$E/vllm/_version.py ] || cp \$R/vllm/_version.py \$E/vllm/_version.py
 ln -s \$(readlink -f \$R/vllm-0.32.0rc2.dev0+g98c712e55.k31vr.dist-info) \$E/
 find \$E/vllm -type l -xtype d | head -1 | grep -q . && { echo 'dir symlink found'; exit 1; }
 rsync -a --exclude=__pycache__ /lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/env/vllm-w6-rf/tokenspeed_mla/ \$E/tokenspeed_mla/  # W6 M5 var-split TokenSpeed patch
 rm -rf \$E.old \$S/vllm
 echo 'self-contained: 98c712e55 compiled files (hard links) + Python of wzhao/k31-vr-w2-dep-moe-floor@$SHA (+$DIRTY dirty) $TAG' > \$E/K31VR_MERGE_OVERLAY.txt; echo env \$E ok \$(find \$E -name '*.py' | wc -l) py"
rsync -a $TL/ hecate:$W/w2dmf/ 2>&1 | grep -v "^\*\*\|post-quantum\|store now\|server may" || true
