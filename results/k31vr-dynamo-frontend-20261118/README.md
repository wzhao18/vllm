# Dynamo frontend bottleneck on Hecate (K3.1 VR200 AgentX), 2026-11-18

Goal: the Dynamo frontend must never bound any K3.1 VR200 AgentX point, up to and beyond C512 (~15 req/s), with >= 2x headroom.

## TL;DR

- **Root cause.** The frontend runs `--dyn-chat-processor vllm`: per request, Python validates the request, renders the K3 XTML
  chat (`encoding_k3.build_chat_segments`) and tokenizes it (checkpoint `TikTokenTokenizer` + our exact K3 token cache), and it
  also runs the whole output path (vLLM `OutputProcessor` detokenize + `StreamingPostProcessor`) for every streamed chunk. All of
  it shares **one CPython GIL per frontend process**. vLLM's renderer hops render+tokenize to a thread pool with
  `renderer_num_workers = 1`. That thread releases the GIL around every tiktoken call and then waits to get it back while the
  event loop streams outputs. Under load, requests queue in that thread for tens of seconds while **no thread is CPU-saturated**.
  `--dyn-preprocess-workers` (process pool) is rejected for the vllm processor, so one frontend process is a hard ceiling.
- **Not the cause:** preprocessing CPU (7.7 ms/request on a Vera core, i.e. ~130 req/s per process), the release checkpoint's
  chat assets (`encoding_k3.py` differs from the frozen one by one line, `_VALID_THINKING_EFFORTS`; `tokenization_kimi.py` is
  byte-identical), the router (selection 18 ms), the asyncio event loop (lag ~0.1-1 ms).
- **Fix (landed in configs):** `frontend.enable_multiple_frontends: true` with N Dynamo frontends behind nginx and
  `nginx_session_affinity: true` (consistent hash of `X-Dynamo-Session-ID`), so a conversation always reaches the same frontend:
  its K3 token cache and the router's session affinity are preserved. Tool: `tools/mk_mfe.py <src render> <dst render> <N>`.
- Evidence: harness A/B (below) and the production C512 rerun (section "Production C512 rerun").

## What was measured

### 1. Offline replay of the preprocessing path (CPU only)

`tools/prof_fe.py` (job `tools/prof_fe.sbatch`): 600 cc-traces requests rebuilt from the AgentX mmap dataset cache, replayed
turn by turn over 64 interleaved sessions (as with session affinity) through Dynamo's own `preprocess_chat_request` +
`InputProcessor.process_inputs`, release model view, production frontend env (`VLLM_K3_FAST_SPLIT`, 512M token cache).

| stage | mean ms | p50 | p90 | share |
|---|---|---|---|---|
| json.loads (body 0.34 MB mean) | 0.46 | 0.26 | 0.80 | 6 % |
| pydantic validate | 0.22 | 0.19 | 0.31 | 3 % |
| render + tokenize (K3 token cache) | 5.86 | 2.27 | 11.35 | 76 % |
| whole preprocess_chat_request | 7.14 | 3.63 | 13.62 | 93 % |
| build engine inputs + process_inputs | 0.10 | 0.07 | 0.16 | 1 % |
| **total** | **7.69** | 4.12 | 14.43 | 100 % |

Mean prompt 84K tokens. cProfile: 52 % of the time in tiktoken `CoreBPE.encode` (only the new messages; the K3 token cache hits
the rest), the XTML render itself ~3 %. **One process can preprocess ~130 req/s**: preprocessing CPU is ~10x below the limit
seen at C512.

### 2. Mocker harness: the real frontend under the real AgentX stream (no GPUs)

`tools/mock_fe.sbatch <tag> <n_frontends> <conc> <duration> <speedup>`: the production frontend command and env
(`--router-mode least-loaded --router-session-affinity-ttl-secs 900 --dyn-chat-processor vllm`, release model view,
K3 tokenizer speedups) + `dynamo.mocker` (4 prefill + 4 decode workers, disaggregated, `--speedup-ratio 1000`, i.e. effectively
instant engines) + the production AgentX client (same pinned `agentx-harness`, scenario, dataset), all on one Vera node; N > 1
puts nginx (same consistent-hash session affinity as srtctl's template) in front. A passive probe (`probe/sitecustomize.py`,
env-gated) records, per request: Rust receive -> Python coroutine start (handoff), coroutine start -> engine call (`pre` =
validate + render + tokenize + process_inputs, including any wait for the tokenizer thread), engine call -> first chunk, plus
asyncio loop lag. `tools/fe_mon.sh` samples per-thread CPU of every frontend process.

| run | frontends | conc | TTFT p50 | TTFT p95 | input tokens served | `pre` p50 (steady state) | busiest frontend thread |
|---|---|---|---|---|---|---|---|
| m2-f1 | 1 | 512 | 29.7 s | 80.7 s | 0.52 B | 30-34 s (m3) | 0.35 cores |
| m2-f4 | 4 + nginx | 512 | 17 ms | 74 ms | 2.74 B (5.3x) | | 0.10-0.12 cores each |
| m4-f1 switch interval 200 us | 1 | 512 | 503 ms | 68 s | 0.12 B | 0.5 s, spikes 82 s | process 34.9 cores (tokio threads spin on GIL handoffs) |
| m6-f1 inline render | 1 | 512 | 273 ms | 4.9 s | 0.41 B | 16-30 ms | 0.36 (peak 0.95); loop lag max 4 s |
| m5-f4 | 4 + nginx | 1536 | 109 ms | 11.7 s | 2.98 B | 1.8-3.5 s | 0.30 each |
| **m7-f1** (clean A/B, probe on) | 1 | 512 | **31.3 s** | 48.9 s | 0.87 B | **32-47 s** | 0.31 |
| **m7-f4** (clean A/B, probe on) | 4 + nginx | 512 | **104 ms** | 713 ms | 1.69 B | **70 ms** (p90 0.16-0.23 s) | 0.27-0.29 each |

Reading:
- One frontend collapses at C512 (requests wait 30 s in `pre`) although its busiest thread uses 0.35 cores and the loop lag
  is ~0.1 ms: a GIL handoff queue on the single renderer thread, not CPU.
- Shrinking the GIL switch interval confirms the GIL (behaviour changes completely) but is no fix: 35 cores of spinning.
- Rendering inline on the event loop removes the thread hop (`pre` 30 s -> 16-30 ms) but then large cache-miss conversations
  block the loop for up to 4 s, and throughput is ~2.6x below 4 frontends. One Python process stays the ceiling.
- Four frontends behind nginx take the same C512 stream at TTFT p50 17 ms and 5.3x the tokens.
- At C1536 (3x the sessions) 4 frontends reach their knee in the harness (~17-24 req/s total, `pre` p50 2-3.5 s), i.e.
  ~5 req/s per frontend under harness output load.
- The harness is harsher than production on the output side: the mocker streams 1 token per chunk at near-infinite decode
  speed (production: ~3.25 tokens per speculative chunk at ~12 ms ITL), so harness per-frontend capacity is a lower bound.

### 3. Production C512 rerun (multi-frontend)

Render `e2e/renders/dsfe-c512-mfe7-spx` = `dsrel-c512-spx` + `enable_multiple_frontends: true`, 7 frontends (one per non-head
node; nginx on the head), `nginx_session_affinity: true`, the passive probe (`FE_PROBE_DIR=/logs`, frozen copy `fe/probe-v1`).
Every other key unchanged (asserted by `mk_mfe.py`). Own spx allocation (`tools/fe_c512.sbatch`; all xdr nodes are reserved
from 2026-11-18 06:00 PST by `co_exclude_maint`).

PENDING.

## Why the release run crossed the cliff and the frozen one did not

The frozen C512 (spx, 14.9 req/s) already queued 0.64 s mean / 1.82 s p90 before routing; the release C512 (xdr, 13.2 req/s)
queued 8.06 s / 12.6 s. Same frontend code and image; chat assets equivalent (above). One frontend sits on a cliff at ~13-15
req/s: its capacity depends on how much GIL time the output stream takes, so small differences in host or decode output
cadence move it across. It needs headroom, not tuning.

## Files

- `tools/prof_fe.py`, `tools/prof_fe.sbatch`: offline preprocessing replay.
- `tools/mock_fe.sbatch`, `tools/mock_inner.sh`, `tools/nginx_mock.conf.tmpl`: mocker harness.
- `probe/sitecustomize.py`: passive frontend probe (`FE_PROBE_DIR`); experiment knobs `FE_SWITCH_INTERVAL`, `FE_INLINE_RENDER`
  (harness only, never in production runs).
- `tools/fe_mon.sh`, `tools/femon_summary.py`: per-thread CPU of frontend/nginx processes.
- `tools/probe_join.py`: join probe records with the frontend log (`request received`) per 2-minute window.
- `tools/preroute_mfe.py`: pre-route delay (`request received` -> `Selected worker`) per frontend, single or multi.
- `tools/mk_mfe.py`: make a multi-frontend copy of a render (parsed-YAML diff asserted).
- `tools/fe_c512.sbatch`: the C512 rerun job (own allocation, `run_in_alloc.sh` + `fe_mon.sh`).
