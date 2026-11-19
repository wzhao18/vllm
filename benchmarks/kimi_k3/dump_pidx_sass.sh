#!/bin/bash
# Compile the sm_107 prefill indexer score kernel with PTX/CUBIN kept and summarize the
# SASS (spills, instruction mix). Usage: dump_pidx_sass.sh <dump_dir>
D=$1; mkdir -p $D
export CUTE_DSL_KEEP_PTX=1 CUTE_DSL_KEEP_CUBIN=1 CUTE_DSL_DUMP_DIR=$D CUTE_DSL_CACHE_DIR=$D/cache
python3 -c "
import torch
from vllm.models.kimi_k3.nvidia.specialized_prefill import indexer as p
p._score_kernel(212, 3)
print('compiled')
" 2>&1 | grep -v "flashinfer\|deprecated" | tail -3
ls -la $D | head -20
for f in $(find $D -name "*.cubin" | head -3); do
  echo "== $f"
  /usr/local/cuda/bin/cuobjdump -res-usage $f 2>&1 | grep -i "REG\|STACK\|LOCAL" | head -5
  /usr/local/cuda/bin/nvdisasm -c $f > $f.sass 2>/dev/null
  echo "lines $(wc -l < $f.sass) local: $(grep -c 'LDL\|STL' $f.sass)"
  grep -o "^\s*/\*[0-9a-f]*\*/\s*[A-Z0-9_.]*" $f.sass | awk '{print $2}' | sed 's/\..*//' | sort | uniq -c | sort -rn | head -25
done
for f in $(find $D -name "*.ptx" | head -2); do
  echo "== $f local: $(grep -c 'ld.local\|st.local' $f)"
done
