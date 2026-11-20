#!/bin/bash
# mb lane gap512: torch-profiler breakdown of the C512 hybrid decode step (D3: DP4 x TP4+DCP4, EP16, k3) on the release
# tree overlay rel-1767cf7 at B 32/50/64/92 per DP rank (rows = B*4: 128,200,256,368), release DSpark draft (FLASH_ATTN),
# vendored MegaMoE (deep_gemm mega_moe of the image/overlay). kb_parse.py breakdown + gap512 per-op parser afterwards.
# Same harness as dsrel prof_common.sh (nq_run1.sh -> nq_step.py, vllm bench forward --profile-dir), own result dir,
# longer timeout (4 cases).
W=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr; T=$W/dsrel/tools
REL=$W/jh/noquant/cfg2/deps/kimi-k3.1-pro-dspark-e4e6e626
export NQ_OV=$W/dsrel/ov/rel-1767cf7 NQ_BUILD=$W/nqbench/cache/build/rel-1767cf7
J=$SLURM_JOB_ID; NL=$NQ_HOSTS
P=$W/gap512/${GAP_TAG:-dec-rel1767}; mkdir -p $P
ROWS=${GAP_ROWS:-128,200,256,368}
pst() { echo "$(date +%T) $*" | tee -a $P/status.txt; }
pst "gap512 dprof on $NL job $J deadline $NQ_DEADLINE ov $NQ_OV rows $ROWS"
unset NQ_MAX_KV_TOKENS
for i in 1 2 3; do
  R=$P/c512-relfa; mkdir -p $R
  if [ -n "$NQ_DEADLINE" ] && [ $(( $(date +%s) + 1500 )) -gt $NQ_DEADLINE ]; then pst "skip (time)"; break; fi
  rm -rf $R/prof-agentx $R/agentx.*
  pst "start c512-relfa rows $ROWS (cap ${NQ_MAX_KV_TOKENS:-none})"
  NQ_DRAFT_BACKEND=FLASH_ATTN NQ_TIMEOUT=4800 bash $T/nq_run1.sh $J $NL $R/agentx c512 ls $REL $ROWS agentx -- --profile-dir $R/prof-agentx > $R/agentx.out 2>&1
  pst "end $(grep -a '^NQEND' $R/agentx.log | head -1): $(grep -a '^\[forward\].*median' $R/agentx.log | sed 's/, sample spread.*//; s/^\[forward\] //' | tr '\n' ';' | cut -c1-600)"
  python3 $W/hti/kb/tools/kb_parse.py $R > $R/breakdown.txt 2>&1
  pst "parsed: $(grep -a 'target .* us, drafter .* us' $R/breakdown.txt | sed 's/.*| target/target/' | tr '\n' ';' | cut -c1-400)"
  grep -aq "cases need more KV blocks" $R/agentx.log 2>/dev/null || break
  cap=$(grep -aoE "~[0-9]+ fits every case" $R/agentx.log | head -1 | tr -dc 0-9)
  [ -z "$cap" ] && { pst "KV short and no suggested cap; stop"; break; }
  export NQ_MAX_KV_TOKENS=$(( cap * 97 / 100 )); pst "KV short -> NQ_MAX_KV_TOKENS=$NQ_MAX_KV_TOKENS"
done
pst "gap512 dprof done"
