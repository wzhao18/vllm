#!/bin/bash
# mock_inner.sh <outdir> <n_frontends> <conc> <duration_s> <speedup> [pyspy:0|1]  -- runs INSIDE the k31vr image container.
# Real Dynamo frontend(s) with the production C512 args/env + dynamo.mocker prefill/decode workers (no GPU) + the production
# AgentX client (same venv, dataset, scenario) -> frontend capacity / pre-route delay under the real request stream.
O=${1:?}; NF=${2:?}; CONC=${3:?}; DUR=${4:?}; SPD=${5:?}; PYSPY=${6:-1}
mkdir -p $O; cd $O
export DYN_DISCOVERY_BACKEND=file DYN_FILE_KV=/tmp/dynkv-$SLURM_JOB_ID-$$ DYN_REQUEST_PLANE=tcp DYN_EVENT_PLANE=zmq
export DYN_LOG="info,dynamo_runtime::pipeline::network::ingress::push_handler=warn" DYN_TCP_REQUEST_TIMEOUT=120
export HF_HUB_CACHE=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/fe/hfhub
export PYTHONPATH=/opt/k3/ov-fe:/opt/k3/fi-0.7.3.dev20261031:/opt/k3/pydeps:/opt/k3/dynamo HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=
rm -rf $DYN_FILE_KV; mkdir -p $DYN_FILE_KV
MOCK="python3 -m dynamo.mocker --model-path /model --model-name moonshotai/Kimi-K3 --max-model-len 1048576 --block-size 64 --speedup-ratio $SPD --num-workers 4 --max-num-seqs 256 --max-num-batched-tokens 65536 --num-gpu-blocks-override 400000"
$MOCK --disaggregation-mode prefill > mock_prefill.log 2>&1 &
$MOCK --disaggregation-mode decode > mock_decode.log 2>&1 &
FE_ENV="FE_INLINE_RENDER=${FE_INLINE_RENDER:-} FE_SWITCH_INTERVAL=${FE_SWITCH_INTERVAL:-} FE_PROBE_DIR=$O PYTHONPATH=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/fe/probe:$PYTHONPATH DYN_LIFECYCLE_TRACE_ENABLED=1 DYN_FORWARD_SESSION_ID=1 DYN_METADATA_HEADER=x-dynamo- DYN_ROUTER_ACTIVE_REQUEST_EXPIRY_SECS=3600 DYN_TOKENIZER_CACHE_BYTES=8589934592 VLLM_K3_FAST_SPLIT=1 VLLM_K3_TOKCACHE_BYTES=512M VLLM_K3_SKIP_ID_VALIDATION=1 VLLM_K3_BYTE_DETOK=1 VLLM_USE_RUST_FRONTEND=1"
for i in $(seq 0 $((NF-1))); do
  port=$((8100+i)); [[ $NF == 1 ]] && port=8000
  cmd="python3 -m dynamo.frontend --http-port=$port --router-mode least-loaded --router-session-affinity-ttl-secs 900 --dyn-chat-processor vllm --trust-remote-code"
  if [[ $PYSPY == 1 ]]; then cmd="$PYSPY_BIN record -r ${PYSPY_RATE:-40} --threads --nonblocking ${PYSPY_FLAGS:-} --format raw -o $O/pyspy_fe$i.txt -- $cmd"; fi
  env $FE_ENV $cmd > fe_$i.out 2>&1 & FEPIDS+=($!)
done
# wait for the model on the public port (nginx started by the caller when NF > 1)
for t in $(seq 1 180); do curl -sf localhost:8000/v1/models 2>/dev/null | grep -q Kimi && break; sleep 5; done
curl -s localhost:8000/v1/models | head -c 300; echo
RT=$O/agentx-rt; UC=/lustre/fsw/coreai_comparch_inferencex/weizha/k31vr/fe/agentx-rt/uv-cache; if ! $RT/venv/bin/python -c 1 2>/dev/null; then rm -rf $RT/venv; UV_CACHE_DIR=$UC /usr/local/bin/uv venv --python 3.11 $RT/venv && UV_CACHE_DIR=$UC /usr/local/bin/uv pip install --python $RT/venv/bin/python --no-config -r /infmax-workspace/infx/bench/agentic/requirements.txt > $O/uv.log 2>&1; fi; AGX=$RT/venv/bin/agentx
echo "$(date +%T) client start" > client.marker
cd /infmax-workspace 2>/dev/null || true
env PYTHONPATH= HF_HUB_CACHE=/hf_hub_cache AIPERF_DATASET_MMAP_CACHE_DIR=/aiperf_mmap_cache AIPERF_USE_DYNAMO_CONV_AWARE_ROUTING=0 \
  AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=true AIPERF_DYNAMO_SESSION_TIMEOUT_SECONDS=14400 \
  AIPERF_DATASET_CONFIGURATION_TIMEOUT=5400 AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=5400 HOME=$O/home \
  $AGX profile --url http://127.0.0.1:8000 --model moonshotai/Kimi-K3 --tokenizer /model --concurrency $CONC \
  --benchmark-duration $DUR --tokenizer-trust-remote-code --output-artifact-dir $O/aiperf_artifacts \
  --public-dataset semianalysis_cc_traces_weka_062126 > $O/client.log 2>&1
echo "client rc=$? $(date +%T)" >> $O/client.marker
kill -INT ${FEPIDS[@]} 2>/dev/null; sleep 30
