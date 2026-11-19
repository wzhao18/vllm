#!/bin/bash
# step_p.sh [-n NAME] [-t TAG] [-m MIN] [-e EMUL_EP] [-c CASES] [-x] <ovtag> [prof 0|1]
# (run on the Rhea login node; long runs under setsid nohup)
# 1-node P3-16 prefill slice: DEP4 on one node standing in for DEP16 (TP1 attention per GPU, DP4, MegaMoE EP4 with
# expert-count emulation EP16 = 56 experts per GPU at the deployed token-expert pairs per GPU), stock prefill forward
# (kimi_k3_mega_moe, trtllm_gvr2 exact top-k, MXFP8 MoE activations, FP8 MLA KV, MXFP4 indexer keys, FP32 KDA state),
# DSpark preview drafter (k=3, TOKENSPEED_MLA) included unless -x. Per-GPU shapes of P3-16 (stepbench/pf_shapes_p3.json):
#   pf16k_L78k  F = 16K fresh tokens per GPU (3 requests) at KV length 78K (prefix hit), the model's reference regime
#   pf8k_L78k   F = 8K at 78K;  pf16k_fresh  one fresh 16K request
# -q nvfp4: NVFP4 checkpoint view (k31-nvfp4-view-L24, EN). -c: comma list of case names (default pf16k_L78k,pf8k_L78k). 24-layer view (as step_d.sh).
# Output: $K/runs/<NAME>/step_p/<TAG>/ (pf.log, pf.json, prof-pf/, fam.txt when profiled).
K=/lustre/fsw/coreai_comparch_inferencex/weizha/k31ws; S=$K/tools/stepbench
NAME=${WS_NAME:-anon}; TAG=""; MIN=75; EMU=16; CASES=pf16k_L78k,pf8k_L78k; XD=""
Q=mxfp4
while getopts n:t:m:e:c:q:x o; do
  case $o in n) NAME=$OPTARG;; t) TAG=$OPTARG;; m) MIN=$OPTARG;; e) EMU=$OPTARG;; c) CASES=$OPTARG;; q) Q=$OPTARG;; x) XD=1;; *) exit 2;; esac
done
shift $((OPTIND-1))
OV=$1; PROF=${2:-0}
[ -n "$OV" ] || { sed -n 2,12p $0; exit 2; }
TAG=${TAG:-$OV-$(date +%m%d%H%M)}
R=$K/runs/$NAME/step_p/$TAG; mkdir -p $R
[ -e $R/pf.log ] && { echo "step_p: $R exists"; exit 3; }
DRAFT=$K/models/dspark-preview-L24; TARGET=$K/models/k31-view-L24; CS=""
case $Q in mxfp4) ;; nvfp4) TARGET=$K/models/k31-nvfp4-view-L24; CS=_nvfp4;; *) echo "-q mxfp4|nvfp4"; exit 2;; esac
NQ_TARGET=$TARGET python3 $S/cfg2bench.py $S/rhea_p3x16_dep4$CS.json --arm stock --drafter $DRAFT --full-capture > $R/pf.cfg.sh || exit 2
python3 -c "
import json, sys
d = json.load(open('$S/pf_shapes_p3.json')); want = '$CASES'.split(',')
cs = [c for c in d['cases'] if c['name'] in want]
assert len(cs) == len(want), ('unknown case', want)
json.dump({'cases': cs}, open('$R/pf.shapes.json', 'w'), indent=1)
print(' '.join(c['name'] + ':' + '+'.join(str(r['query_len']) + '@' + str(r['kv_len']) for r in c['requests']) for c in cs))
" > $R/pf.shapes.txt || exit 2
EXTRA=(); [ "$PROF" = 1 ] && EXTRA+=(--profile-dir $R/prof-pf); [ -n "$XD" ] && EXTRA+=(--exclude-drafter)
{ echo "STEP_P ov=$OV cases=$CASES prof=$PROF ckpt=$Q emulate_ep=$EMU exclude_drafter=${XD:-0} name=$NAME start=$(date +%FT%T)"
  cat $R/pf.shapes.txt; grep '^#' $R/pf.cfg.sh; } > $R/pf.log
echo "$(date +%T) start $TAG" > $R/status.txt
NQ_ITERS=${NQ_ITERS:-3} NQ_SAMPLES=${NQ_SAMPLES:-3} NQ_EMULATE_EP=$EMU bash $K/tools/wsgpu.sh -g 4 -m $MIN -o $OV -n $NAME -P strict:$R/pf.cfg.sh -- \
  bash $S/step_inner.sh $R/pf "${EXTRA[@]}" >> $R/pf.log 2>&1
rc=$?
echo "STEP_P end rc=$rc $(date +%FT%T)" >> $R/pf.log
echo "$(date +%T) end rc=$rc" >> $R/status.txt
grep -a "^\[forward\].*median\|^NQ: expert emulation\|Traceback\|Error" $R/pf.log | sed 's/, sample spread.*//' | head -20
if [ "$PROF" = 1 ] && [ $rc = 0 ] && [ -f $S/pf_fam.py ]; then
  python3 $S/pf_fam.py $R/prof-pf $S/pf_model_rows_L24.json $Q > $R/fam.txt 2>&1; cat $R/fam.txt
fi
echo "step_p: results in $R"
exit $rc
