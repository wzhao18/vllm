#!/bin/bash
# mb lane prefill: P3-16 (DEP16: TP1 x DP16, EP16, 4 nodes) prefill step, release tree rel-1767cf7, dummy weights.
# retry 6: KV 16 GiB (8 GiB = 24 blocks, warmup needs 40); (F 8192 per rank: F16K at TP1 OOMs, 225 GiB weights + KV + ~30 GiB 16K activations; model grid P3-16 F8K L78K 486.5 ms) - earlier: --kv-cache-memory-bytes 16 GiB + --max-num-seqs 4 (retry 3: warmup needs 322 blocks with 30 decode requests, 24 fit) (retry 2 OOMed allocating the 2 GiB sparse-MLA output at TP1: 225 GiB weights + 0.92 KV pool + bench snapshot); --max-model-len 131072 --kv-cache-memory-bytes 17179869184 --max-num-seqs 4 (TP1 KV pool cannot hold one 1M request). F16K tokens per rank at L 78K (model ref) + a fresh 16K chunk. MXFP8-pinned MegaMoE + torch profile, then the
# release MegaMoE policy (NVFP4 above 512 tok/rank) timing only.
source /lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/mb/prefill/tools/pf_common.sh
R=$L/results/p3g; mkdir -p $R
pst "p3g start job $SLURM_JOB_ID hosts $NQ_HOSTS deadline $NQ_DEADLINE"
guard 30 || exit 0
export PF_TIMEOUT=2400 K31P1_PROFILE_RUNS=2
PF_MOE_QUANT=mxfp8 PF_CNAME=k31pf-a bash $T/pf_run.sh $NQ_HOSTS 4 $R/p3_mx "${P3[@]}" --max-model-len 131072 --kv-cache-memory-bytes 17179869184 --max-num-seqs 4 --max-num-batched-tokens 8192 --case pf8k_L78k=2*2731@78360,2730@78360 --case pf8k_fresh=8192 \
  --output-json $R/p3_mx.json --profile-dir $R/p3_mx.prof > $R/p3_mx.out 2>&1
pst "p3g A: $(grep -a '^PFEND' $R/p3_mx.log) $(grep -a '^\[forward\].*median' $R/p3_mx.log | sed 's/, sample spread.*//' | tr '\n' ';')"
for f in $R/p3_mx.prof/*.rank0.trace.json.gz; do
  [ -f "$f" ] && python3 $T/p1_attr.py $f --json ${f%.trace.json.gz}.attr.json > ${f%.trace.json.gz}.attr.txt 2>&1
done
if guard 25; then
  PF_TIMEOUT=1800 PF_MOE_QUANT=release PF_CNAME=k31pf-a bash $T/pf_run.sh $NQ_HOSTS 4 $R/p3_rel "${P3[@]}" --max-model-len 131072 --kv-cache-memory-bytes 17179869184 --max-num-seqs 4 --max-num-batched-tokens 8192 --case pf8k_L78k=2*2731@78360,2730@78360 --case pf8k_fresh=8192 \
    --output-json $R/p3_rel.json > $R/p3_rel.out 2>&1
  pst "p3g B: $(grep -a '^PFEND' $R/p3_rel.log) $(grep -a '^\[forward\].*median' $R/p3_rel.log | sed 's/, sample spread.*//' | tr '\n' ';')"
fi
pst "p3g done"
