#!/bin/bash
# dsrel k5 coverage: the C128 / C160 1P1D llh decode role at k5 with its PRODUCTION capture list (release_draft/k5 render:
# 123 sizes = 96 verify rows + 90 drafter batches, cap 960 = 160 x q6, max-num-seqs 160), all captured (NQ_FULL_CAPTURE).
# Bench-only config dsrel/k5cfg/nq-c160k5p-xdr.json = nq-c160-xdr.json with that decode role. Overlay int1. Target only
# first (NQ-NOFALLBACK: an uncovered q6 shape refuses at build / capture), then the release drafter (int1's drafter is
# non-strict: an uncovered piece logs "keeps stock (...)" instead). Rows B16 / B28 / B38 / B48 (q6).
export OVTAG=int1 RUN_GUARD=1500 NQ_FULL_CAPTURE=1 NQ_TIMEOUT=3600
source /lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/dsrel/tools/tasks/step_common.sh
R=$W/dsrel/results/k5full-int1; mkdir -p $R   # own results dir (run() reads $R at call time)
P=$W/dsrel/k5cfg/nq-c160k5p-xdr.json
unset NQ_MAX_KV_TOKENS NQ_K
echo "$(date +%T) k5full c160 on $NQ_HOSTS job $J (ov $NQ_OV, config $P)" | tee -a $R/status.txt
run c160k5-tgt $P new 96,168,228,288 -- --exclude-drafter
run c160k5-relfa $P $REL 96,168,228,288
echo "$(date +%T) k5full c160 done" | tee -a $R/status.txt
