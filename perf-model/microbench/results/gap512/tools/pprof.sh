#!/bin/bash
# mb lane gap512: C512 hybrid PREFILL (P2: DP4 x TP4=DCP4 ShardQ, EP16, MXFP8 MoE act) forward on the release tree
# overlay rel-1767cf7, chunks of 8K/12K/16K tokens per DP rank at a 78K context (chunked continuation), 2x6K, and a fresh
# 12K; torch-profiler per-kernel breakdown (vllm bench forward --profile-dir). gap512/pf_run1.sh + pf_cfg.py.
W=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr; G=$W/gap512
export NQ_OV=$W/dsrel/ov/rel-1767cf7 NQ_BUILD=$W/nqbench/cache/build/rel-1767cf7
J=$SLURM_JOB_ID; NL=$NQ_HOSTS
P=$G/${GAP_TAG:-pre-rel1767}; mkdir -p $P
pst() { echo "$(date +%T) $*" | tee -a $P/status.txt; }
pst "gap512 pprof on $NL job $J deadline $NQ_DEADLINE ov $NQ_OV"
R=$P/c512p; mkdir -p $R; rm -rf $R/prof $R/pf.*
NQ_TIMEOUT=4500 bash $G/pf_run1.sh $J $NL $R/pf $G/${GAP_SHAPES:-pf_shapes.json} -- --profile-dir $R/prof > $R/pf.out 2>&1
pst "end $(grep -a '^NQEND' $R/pf.log | head -1): $(grep -a '^\[forward\].*median' $R/pf.log | sed 's/, sample spread.*//; s/^\[forward\] //' | tr '\n' ';' | cut -c1-700)"
grep -a "Error\b\|Traceback" $R/pf.log | sed 's/^\[rank[0-9]*\]: //' | sort | uniq -c | sort -rn | head -3 | sed 's/^/  ERR /' | tee -a $P/status.txt
grep -ai "mxfp8\|nvfp4\|quant kind\|activation" $R/pf.log | sort | uniq -c | sort -rn | head -8 | sed 's/^/  QUANT /' | tee -a $P/status.txt
pst "gap512 pprof done"
