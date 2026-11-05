#!/bin/bash
# nq_gputest.sh <jobid> <host A> <host B> <out dir>: the GPU correctness stage of an nq block (plan line '<slot> gputest'),
# on the code under test NQ_OV (stage_overlay.sh: the overlay tree carries the checkout's tests/), inside the nq e2e
# image, concurrently:
#   host A, 4 GPUs: pytest per file on tests/models/kimi_k3/test_specialized_decode_*gpu*.py, *_draft_nq.py,
#     *_wide_nq.py, *_plan.py, *_dsa*.py, then each of those files that is also a torchrun program ('torchrun' +
#     __main__) under torchrun (its documented --nproc-per-node, at most 4);
#   host B, 2 GPUs (the 2-GPU CI layout): pytest per file on the tests/distributed files of the specialized decode
#     (names *specialized_decode* / *kimi_k3* / *k31* / *dcp_exchange*, or changed vs 242fb49cd4: SOURCE_REV 'T' lines),
#     and their torchrun programs at 2 processes.
# Each file: <out>/gpu<N>/<name>.{xml,log}, a status line in <out>/gpu<N>/status.txt; then <out>/GPUTEST.txt (nq_junit.py:
# per file pass/fail/error/skip counts, failing tests, skip reasons, FAIL / TIMEOUT / NO-TESTS / ALL-SKIPPED flags).
# env: NQ_OV (required), NQ_IMG, NQ_CNAME, NQ_BUILD (default $NB/cache/build/<ovtag>: shared with the A/B processes of
# the same tree), NQ_TEST_TIMEOUT (s for the whole stage, default 4500), NQ_TEST_FILE_TIMEOUT (s per file, default 1800).
W=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr; K=/lustre/fsw/coreai_comparch_inferencex/weizha/k3vr
M=/lustre/fsw/coreai_comparch_inferencex/weizha/models; NB=$W/nqbench
D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
J=$1; HA=$2; HB=$3; OUT=$4
mkdir -p $OUT
if [ -z "$NQ_OV" ] || [ ! -d "$NQ_OV/tests" ]; then
  echo "GPUTEST NOT RUN: no overlay with tests/ (NQ_OV=${NQ_OV:-unset})" | tee $OUT/GPUTEST.txt; exit 2
fi
export NQ_OV NQ_BUILD=${NQ_BUILD:-$NB/cache/build/$(basename $NQ_OV)} ENROOT_CACHE_PATH=$K/enroot-cache
IMG=${NQ_IMG:-$K/images/vllm-openai-cu134-nightly-3f1bcfd82b-aarch64.sqsh}
unset SLURM_NTASKS SLURM_NPROCS SLURM_NTASKS_PER_NODE SLURM_TASKS_PER_NODE SLURM_NNODES SLURM_JOB_NUM_NODES SLURM_NODELIST
CARGS=(--container-image=$IMG --container-name=${NQ_CNAME:-k31nq-step} --no-container-mount-home
       --container-mounts=$W:$W,$K:$K,$M:$M)
echo "$(date +%T) gputest start: 4-GPU on $HA, 2-GPU on $HB, overlay $NQ_OV ($(head -1 $NQ_OV/SOURCE_REV 2>/dev/null))"
for spec in "4 $HA" "2 $HB"; do
  set -- $spec
  timeout ${NQ_TEST_TIMEOUT:-4500} srun --overlap --jobid $J -N1 -n1 -w $2 "${CARGS[@]}" \
    bash $D/nq_gputest_inner.sh $1 $OUT/gpu$1 > $OUT/gpu$1.log 2>&1 &
  pids[$1]=$!
done
# Each stage's exit: 124 = the stage timeout cut it (nq_junit: TIMEOUT, and its unfinished planned runs NOT RUN).
for n in 4 2; do
  wait ${pids[$n]}; rc=$?; mkdir -p $OUT/gpu$n; echo "NQTEST stage gpu$n rc=$rc" >> $OUT/gpu$n/status.txt
done
python3 $D/nq_junit.py $OUT/gpu4 $OUT/gpu2 > $OUT/GPUTEST.txt 2>&1
cat $OUT/GPUTEST.txt
echo "$(date +%T) gputest end"
