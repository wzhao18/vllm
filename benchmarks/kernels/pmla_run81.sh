#!/bin/bash
R=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws/runs/P-MLA/run81; mkdir -p $R; export PATH=$PATH:/usr/local/cuda/bin
for f in benchmarks/kernels/pmla_ptx/*.ptx; do for V in 9.4 9.5; do sed "s/VER/$V/" $f > $R/t.ptx; if ptxas -arch=sm_107a $R/t.ptx -o $R/t.cubin > $R/$(basename $f).$V.log 2>&1; then echo "$(basename $f) $V OK" >> $R/summary.txt; cuobjdump -sass $R/t.cubin | grep -E "LDTM|RED|FMNMX" | head -5 >> $R/summary.txt; break; else echo "$(basename $f) $V FAIL $(head -c 200 $R/$(basename $f).$V.log)" >> $R/summary.txt; fi; done; done
ptxas --version | tail -1 >> $R/summary.txt; echo DONE > $R/DONE
