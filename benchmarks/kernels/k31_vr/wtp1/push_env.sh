#!/bin/bash
# usage: push_env.sh <tag>  -- builds $W/env/vllm-wtp1-<tag> on Hecate: every vllm/ source file of this worktree
# (real copies, never links) + symlinks to the compiled *.so of the original SM107 build (vllm-98c712e55); csrc
# must be unchanged since a050f29477. Real copies keep the overlay immune to other tracks writing through
# hard/symlinked base envs (the mwide-a050f29 base was overwritten by such a write on 2026-10-31 09:57).
# Also pushes tests/ + benchmarks/ to $W/wtp1/src-<tag> and the wtp1 tools to $W/wtp1/tools.
set -e
TAG=$1; WT=/Users/weizha/vllm/rainhollow-k31vr-wep-tp1-attn
W=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr
HS=/private/tmp/claude-501/-Users-weizha/8fd11ba1-2bcc-4054-91f7-43fb556afe02/scratchpad/hs
cd $WT
CS=$(git diff --name-only a050f29477 -- csrc/ CMakeLists.txt)
[ -z "$CS" ] || { echo "csrc changed: rebuild needed"; exit 1; }
FILES=$( (git ls-files vllm; git ls-files --others --exclude-standard vllm) | grep -v "\.so$")
COPYFILE_DISABLE=1 tar --no-xattrs -czf /tmp/wtp1_env_$TAG.tgz $FILES tests/__init__.py tests/conftest.py tests/utils.py \
  tests/kernels/attention tests/models/__init__.py tests/models/kimi_k3 benchmarks/kernels/k31_vr 2>/dev/null
for i in 1 2 3 4 5 6; do
  ssh hecate "set -e; E=$W/env/vllm-wtp1-$TAG; S=$W/wtp1/src-$TAG; B=$W/env/vllm-98c712e55; rm -rf \$S \$E; mkdir -p \$S \$E $W/wtp1/tools $W/wtp1/results
   cd \$S; tar xzf -; cp -r \$S/benchmarks/kernels/k31_vr/wtp1/. $W/wtp1/tools/
   cp -n $W/d4x4/tools/decode_ctx_vectors.json $W/wtp1/tools/ 2>/dev/null || true
   cp -a \$S/vllm \$E/vllm
   cd \$B; for f in \$(find vllm \( -type f -o -type l \) ! -path '*__pycache__*'); do case \$f in *.so) ;; *) [ -e \$E/\$f ] && continue;; esac; mkdir -p \$E/\$(dirname \$f); ln -sf \$(readlink -f \$B/\$f) \$E/\$f; done
   cp -a $W/env/vllm-w6-rf/tokenspeed_mla \$E/tokenspeed_mla; rm -rf \$E/tokenspeed_mla/__pycache__  # W6 M5 var-split tokenspeed
   ln -s \$(readlink -f \$B/vllm-0.32.0rc2.dev0+g98c712e55.k31vr.dist-info) \$E/
   echo 'full source copy of wzhao/k31-vr-wep-tp1-attn $TAG + .so of vllm-98c712e55' > \$E/K31VR_MERGE_OVERLAY.txt
   echo env \$E ok \$(find \$E/vllm -name '*.so' | wc -l) so \$(find \$E/vllm -type l | wc -l) links" < /tmp/wtp1_env_$TAG.tgz 2>&1 | grep -v "post-quantum\|store now\|server may" && exit 0
  sleep 15
done; exit 1
