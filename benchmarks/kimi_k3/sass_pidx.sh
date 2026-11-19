#!/bin/bash
# SASS of the kept cubins (image nvdisasm from the triton wheel). Usage: sass_pidx.sh <dump_dir>
D=$1; ND=/usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/bin/nvdisasm
$ND --version | tail -1
for f in $(find $D -name "*.cubin"); do
  $ND -c $f > $f.sass 2>$f.sass.err || { echo "nvdisasm failed: $(head -3 $f.sass.err)"; continue; }
  echo "== $(basename $f | cut -c1-80): $(wc -l < $f.sass) lines"
  grep -oE "^\s+/\*[0-9a-f]{4,}\*/\s+[@!A-Z0-9_.P ]*[A-Z][A-Z0-9_.]+" $f.sass | awk '{print $NF}' | sed 's/\..*//' | sort | uniq -c | sort -rn | head -30 | tr '\n' ' '; echo
done
