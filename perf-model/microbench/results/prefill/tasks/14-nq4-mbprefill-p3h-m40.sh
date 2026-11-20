#!/bin/bash
# mb lane prefill: P3-16 (DEP16) F16K per rank at L 78K with MXFP8-pinned MegaMoE, timing only (no profile).
# p3e's MXFP8 arm OOMed by ~2 GiB (sparse-MLA output alloc) with a 16 GiB KV pool while the release (NVFP4) arm fit:
# KV 14 GiB (42 blocks, warmup needs 40) + expandable segments (2.7 GiB was reserved-unallocated).
source /lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/mb/prefill/tools/pf_common.sh
R=$L/results/p3h; mkdir -p $R
pst "p3h start job $SLURM_JOB_ID hosts $NQ_HOSTS deadline $NQ_DEADLINE"
guard 20 || exit 0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PF_TIMEOUT=1500 PF_MOE_QUANT=mxfp8 PF_CNAME=k31pf-a bash $T/pf_run.sh $NQ_HOSTS 4 $R/p3_mx16 "${P3[@]}" --max-model-len 131072 \
  --kv-cache-memory-bytes 15032385536 --max-num-seqs 4 --case pf16k_L78k=2*5461@78360,5462@78360 \
  --output-json $R/p3_mx16.json > $R/p3_mx16.out 2>&1
pst "p3h: $(grep -a '^PFEND' $R/p3_mx16.log) $(grep -a '^\[forward\].*median' $R/p3_mx16.log | sed 's/, sample spread.*//' | tr '\n' ';')"
pst "p3h done"
