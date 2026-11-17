#!/bin/bash
# dsrel k5 vs k3 at the c160 decode config (nq-c160-xdr: TP16 = DCP16, k3). NQ_K=5 overrides num_speculative_tokens
# (NOT the config's k) in the k5 runs. Overlay int1 (model track's GQA drafter fast path + bench-only stock-attention shim).
# Same 4 nodes, same AgentX context vectors (q4 group at both q), no context cap (as ksweep-6a14e2f):
#   relfa = release draft (FLASH_ATTN draft backend), tgt = target only (--exclude-drafter);
#   k5: q6 rows 96,168,228,288 / k3: q4 rows 64,112,152,192 (B16 / B28 / B38 / B48).
export OVTAG=int1 RUN_GUARD=900
source /lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/dsrel/tools/tasks/step_common.sh
R=$W/dsrel/results/k5ab2-int1; mkdir -p $R   # own results dir (run() reads $R at call time); attempt 2 (attempt 1, k5ab-int1, lost to the preemption of backfill hold 1119830)
unset NQ_MAX_KV_TOKENS
echo "$(date +%T) k5ab c160 on $NQ_HOSTS job $J (ov $NQ_OV)" | tee -a $R/status.txt
NQ_K=5 run c160k5-relfa c160 $REL 96,168,228,288
run c160k3-relfa c160 $REL 64,112,152,192
NQ_K=5 run c160k5-tgt c160 new 96,168,228,288 -- --exclude-drafter
run c160k3-tgt c160 new 64,112,152,192 -- --exclude-drafter
echo "$(date +%T) k5ab c160 done" | tee -a $R/status.txt
