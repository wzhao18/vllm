#!/bin/bash
# Register / shared-memory / spill usage of the stock FlashKDA build and the
# in-tree split build's K1 / K2 kernels (cuobjdump -res-usage).
CUOBJ=$(command -v cuobjdump || echo /usr/local/cuda/bin/cuobjdump)
STOCK=$(python3 -c "import vllm._flashkda_C as m; print(m.__file__)" 2>/dev/null)
SPLIT=$(ls -t ${VLLM_CACHE_ROOT:-$HOME/.cache/vllm}/k3_flashkda_split/*/k3_flashkda_split.so 2>/dev/null | head -1)
for f in "$STOCK" "$SPLIT"; do
  echo "== $f"
  $CUOBJ -res-usage "$f" 2>/dev/null | grep -A1 -E "Function .*(recurrence|prepare)" | \
    grep -E "Function|REG" | sed -E 's/Function (_Z[0-9]+_flash_kda_fwd_[a-z]+).*/\1/' | \
    paste - - | sort | uniq -c | sort -rn | head -40
done
