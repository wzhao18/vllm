#!/bin/bash
# nq_gputest_inner.sh <ngpu 4|2> <out dir>: inside the nq e2e image on one node, the GPU correctness tests of the
# overlay NQ_OV (see nq_gputest.sh). Environment = the merge-gate test harness (env/k31vr_env.sh for the base env
# vllm-jh-242fb49cd4, i.e. FlashInfer 0.7.3.dev20261031 and its JIT dirs; merge/tools/tests1.v2.sh: --noconftest with the
# k31m_fixtures plugin, VLLM_SSM_CONV_STATE_LAYOUT=DS) with the overlay tree first on PYTHONPATH in place of the base env.
W=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr
N=$1; OUT=$2
export K31VR_VLLM_TAG=jh-242fb49cd4
source $W/env/k31vr_env.sh
export PYTHONPATH=$NQ_OV:${PYTHONPATH#*:}:$W/e2e/env/pydeps:$W/wide/tools:$W/merge/tools/shim
export VLLM_SSM_CONV_STATE_LAYOUT=DS PYTHONUNBUFFERED=1
export K31_DCPX_BUILD_DIR=${NQ_BUILD:?NQ_BUILD}/dcpx K31_DSAC_BUILD_DIR=$NQ_BUILD/dsac
if [ $N = 2 ]; then export CUDA_VISIBLE_DEVICES=0,1; else export CUDA_VISIBLE_DEVICES=0,1,2,3; fi
mkdir -p $OUT $K31_DCPX_BUILD_DIR $K31_DSAC_BUILD_DIR; : > $OUT/status.txt
cd $NQ_OV
echo "NQTEST host $(hostname) gpus $CUDA_VISIBLE_DEVICES: $(nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader | tr '\n' ' ')"
python3 -c "import vllm, os; print('NQTEST vllm from', os.path.dirname(vllm.__file__))"
if [ $N = 4 ]; then
  T=tests/models/kimi_k3
  FILES=$(ls $T/test_specialized_decode_*gpu*.py $T/test_specialized_decode_draft_nq.py $T/test_specialized_decode_wide_nq.py \
    $T/test_specialized_decode_plan.py $T/test_specialized_decode_dsa*.py 2>/dev/null | sort -u)
else
  FILES=$( { ls tests/distributed/test_*specialized_decode*.py tests/distributed/test_*kimi_k3*.py \
      tests/distributed/test_*k31*.py tests/distributed/test_*dcp_exchange*.py 2>/dev/null
    sed -n 's/^T \(tests\/distributed\/.*test_[^/]*\.py\)$/\1/p' SOURCE_REV | while read -r f; do [ -f $f ] && echo $f; done
  } | sort -u)
fi
[ -n "$FILES" ] || { echo "NQTEST none - rc=NONE secs=0 (no test files of this stage in the overlay)" | tee -a $OUT/status.txt; exit 0; }
# The planned runs, before any starts: a run with a plan line and no result line did not finish (nq_junit: NOT RUN).
TR=""
for f in $FILES; do
  echo "NQTEST plan pytest $f" >> $OUT/status.txt
  # (a file whose pytest already wraps torchrun is not run under torchrun again)
  if grep -q "torchrun" $f && grep -q "__main__" $f && ! grep -q "def test_gpu_torchrun" $f; then
    TR="$TR $f"; echo "NQTEST plan torchrun $f" >> $OUT/status.txt
  fi
done
for f in $FILES; do
  n=$(basename $f .py); t0=$(date +%s)
  timeout ${NQ_TEST_FILE_TIMEOUT:-1800} python3 -m pytest --noconftest -p no:cacheprovider -p k31m_fixtures -q -rfEs \
    --junitxml=$OUT/$n.xml $f > $OUT/$n.log 2>&1
  rc=$?; echo "NQTEST pytest $f rc=$rc secs=$(( $(date +%s) - t0 ))" | tee -a $OUT/status.txt
done
for f in $TR; do
  np=$(grep -o "nproc-per-node[ =][0-9]*" $f | head -1 | grep -o "[0-9]*$"); np=${np:-$N}; [ $np -gt $N ] && np=$N
  n=$(basename $f .py); t0=$(date +%s)
  timeout ${NQ_TEST_FILE_TIMEOUT:-1800} python3 -m torch.distributed.run --nproc-per-node $np $f > $OUT/$n.torchrun$np.log 2>&1
  rc=$?; echo "NQTEST torchrun $f rc=$rc np=$np secs=$(( $(date +%s) - t0 ))" | tee -a $OUT/status.txt
done
