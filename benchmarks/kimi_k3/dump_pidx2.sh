#!/bin/bash
# Compile the sm_107 prefill indexer kernels with PTX + CUBIN kept; print full resource usage,
# PTX local-memory traffic and look for an nvdisasm in the image. Usage: dump_pidx2.sh <dir>
D=$1; mkdir -p $D
export CUTE_DSL_KEEP_PTX=1 CUTE_DSL_KEEP_CUBIN=1 CUTE_DSL_DUMP_DIR=$D CUTE_DSL_CACHE_DIR=$D/cache
python3 -c "
import torch
from vllm.models.kimi_k3.nvidia.specialized_prefill import indexer as p
p._score_kernel(212, 0)
p._select_kernel(212, 2048)
print('compiled')
" 2>&1 | grep -v "flashinfer\|deprecated" | tail -3
for f in $(find $D -name "*.cubin"); do
  echo "== $(basename $f | cut -c1-80)"; /usr/local/cuda/bin/cuobjdump -res-usage $f 2>&1 | grep -v "^$" | cut -c1-200
done
for f in $(find $D -name "*.ptx"); do
  echo "== $(basename $f | cut -c1-80) ld.local $(grep -c 'ld.local' $f) st.local $(grep -c 'st.local' $f) tcgen05.ld $(grep -c 'tcgen05.ld' $f) wait::ld $(grep -c 'tcgen05.wait::ld' $f) cvt.rn.relu $(grep -c 'cvt.rn.relu' $f) fma.rn.bf16x2 $(grep -c 'fma.rn.bf16x2' $f)"
done
find / -xdev -name "nvdisasm*" 2>/dev/null | head -5
python3 -c "import nvidia, os; print(os.listdir(os.path.dirname(nvidia.__path__[0])+'/nvidia'))" 2>/dev/null
