#!/bin/bash
# run_pidx2.sh + SASS of the kept cubins (nvdisasm copied out of the image's triton wheel).
O=$1; shift
bash benchmarks/kimi_k3/run_pidx2.sh $O "$@"
ND=/usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/bin/nvdisasm
mkdir -p $(dirname $O)/bin && cp -n $ND $(dirname $O)/bin/ 2>/dev/null
for f in $(find $O.dump -name "*.cubin"); do
  $ND -c $f > $f.sass 2>$f.sass.err || { echo "nvdisasm failed: $(head -2 $f.sass.err)"; continue; }
  echo "== $(basename $f | cut -c1-70): $(wc -l < $f.sass) SASS lines"
done
