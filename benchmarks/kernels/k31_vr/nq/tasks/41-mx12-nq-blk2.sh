#!/bin/bash
# queue task for holds/runner.sbatch POOL=mx12 (12-node --segment=12 block, 3:30 h): nq step A/B block 2, stock
# (decode_backend auto) vs latency_specialized at TP16=DCP16, code under test = overlay __OVTAG__ (stage_overlay.sh).
# Splits the block into 3 concurrent 4-node slots (plans/block2.plan), summarizes, then scancels the block.
# Results: /lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/nqbench/results/blk2-__OVTAG__/{SUMMARY.txt,A,B,C}
export NQ_TASK=41-mx12-nq-blk2-__OVTAG__ NQ_TAG=blk2-__OVTAG__
export NQ_OV=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/nqbench/ov/__OVTAG__
exec bash /lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/nqbench/tools/nq_block.sh \
  /lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/nqbench/tools/plans/block2.plan
