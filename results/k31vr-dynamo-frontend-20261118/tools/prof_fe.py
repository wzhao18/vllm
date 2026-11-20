"""prof_fe.py: offline replay of AgentX cc-traces requests through the Dynamo vLLM chat-processor preprocessing path
(the exact code the frontend runs per request with --dyn-chat-processor vllm), one process, no GPU.

Per request: body = json.dumps(request) (what aiperf sends); stages timed:
  loads     json.loads(body)                      (stand-in for Rust serde -> Python dict conversion)
  validate  prepost._validate_chat_completion_request (pydantic ChatCompletionRequest)
  render    encoding_k3.build_chat_segments (XTML render, Python)
  tokenize  tokenizer._encode_chat_segments (tiktoken + K3 token cache)
  pre       whole preprocess_chat_request (validate + render + tokenize + rest)
  engine    _build_engine_inputs + InputProcessor.process_inputs
Conversations are replayed turn by turn, interleaved round-robin over --sessions sessions (as with nginx session affinity:
each conversation's turns reach the same process in order), so the K3 token cache sees the serving hit pattern.
Usage: python prof_fe.py --cache <aiperf mmap cache dir> --model /model --sessions 64 --max-requests 600 --out out.json
"""
import argparse, asyncio, cProfile, io, json, os, pstats, random, statistics, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True); ap.add_argument("--model", default="/model")
ap.add_argument("--sessions", type=int, default=64); ap.add_argument("--max-requests", type=int, default=600)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
ap.add_argument("--cprofile", type=int, default=150, help="profile this many requests (after warm) with cProfile")
a = ap.parse_args()

idx = json.load(open(os.path.join(a.cache, "index.dat")))
ids = list(idx["conversation_ids"]); random.Random(a.seed).shuffle(ids)
fds = open(os.path.join(a.cache, "dataset.dat"), "rb")
def load_conv(cid):
    o = idx["offsets"][cid]; fds.seek(o["offset"]); return json.loads(fds.read(o["size"]))

def requests_of(c):
    msgs = []
    if c.get("system_message"): msgs.append({"role": "system", "content": c["system_message"]})
    if c.get("user_context_message"): msgs.append({"role": "user", "content": c["user_context_message"]})
    for t in c["turns"]:
        if t.get("reset_context"):
            msgs = msgs[:1] if msgs and msgs[0]["role"] == "system" else []
        msgs = msgs + [dict(m) for m in (t.get("raw_messages") or [])]
        if not msgs or msgs[-1]["role"] != "user":
            continue  # the recorded assistant reply comes with the next turn's delta
        yield {"model": "moonshotai/Kimi-K3", "messages": list(msgs), "max_tokens": int(t.get("max_tokens") or 64),
               "stream": True, "stream_options": {"include_usage": True}}

# ---- build the processor exactly as dynamo.frontend.vllm_processor.EngineFactory.chat_engine_factory ----
t0 = time.time()
from vllm.config import ModelConfig, VllmConfig, LoadConfig, CacheConfig
from vllm.v1.engine.input_processor import InputProcessor
from vllm.sampling_params import SamplingParams, RequestOutputKind
from dynamo.frontend import prepost
from dynamo.frontend.vllm_processor import _build_engine_inputs, _ensure_chat_template
try:
    from dynamo.frontend.vllm_processor import GENERATION_TASKS
except ImportError:
    GENERATION_TASKS = ("generate",)
os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
mc = ModelConfig(model=a.model, tokenizer_mode="auto", config_format="auto", trust_remote_code=True, max_model_len=1048576)
vc = VllmConfig(model_config=mc, load_config=LoadConfig(load_format="dummy"), cache_config=CacheConfig())
os.environ.setdefault("VLLM_MEDIA_CONNECTOR", "dynamo")
import dynamo.common.multimodal.media_connector  # noqa: F401
ip = InputProcessor(vc); tok = ip.get_tokenizer(); _ensure_chat_template(tok, a.model, None)
print(f"setup {time.time()-t0:.1f}s tokenizer {type(tok).__name__} mro {[c.__name__ for c in type(tok).__mro__][:4]}", flush=True)

# ---- stage timers on the tokenizer's render/encode entry points ----
STAGE = {}
def wrap_mod_fn(mod, name, key):
    f = getattr(mod, name)
    def w(*args, **kw):
        s = time.perf_counter(); r = f(*args, **kw); STAGE[key] = STAGE.get(key, 0.0) + time.perf_counter() - s; return r
    setattr(mod, name, w)
inner = getattr(tok, "tokenizer", tok)  # vLLM wrapper -> checkpoint TikTokenTokenizer
tk_mod = sys.modules[type(inner).__module__] if type(inner).__module__ in sys.modules else None
targets = []
for obj in (inner, tok):
    m = sys.modules.get(type(obj).__module__)
    if m is not None and hasattr(m, "build_chat_segments"):
        wrap_mod_fn(m, "build_chat_segments", "render"); targets.append(m.__name__); break
enc = getattr(inner, "_encode_chat_segments", None)
if enc is not None:
    def enc_w(segments, _e=enc):
        s = time.perf_counter(); r = _e(segments); STAGE["tokenize"] = STAGE.get("tokenize", 0.0) + time.perf_counter() - s; return r
    inner._encode_chat_segments = enc_w; targets.append("_encode_chat_segments")
print("stage hooks:", targets, flush=True)

async def one(req):
    STAGE.clear(); r = {}
    body = json.dumps(req).encode(); r["bytes"] = len(body)
    s = time.perf_counter(); d = json.loads(body); r["loads"] = time.perf_counter() - s
    s = time.perf_counter(); prepost._validate_chat_completion_request(dict(d)); r["validate"] = time.perf_counter() - s
    s = time.perf_counter()
    pre = await prepost.preprocess_chat_request(d, tokenizer=tok, renderer=ip.renderer, tool_parser_class=None,
                                                reasoning_parser_class=None, model_config=ip.model_config)
    r["pre"] = time.perf_counter() - s
    r["render"] = STAGE.get("render", 0.0); r["tokenize"] = STAGE.get("tokenize", 0.0)
    r["tokens"] = len(pre.prompt_token_ids)
    s = time.perf_counter()
    ei = await _build_engine_inputs(ip.renderer, pre.engine_prompt, pre.prompt_token_ids, cache_salt=None,
                                    mm_processor_kwargs=None, defer_multimodal_processing=False)
    sp = SamplingParams(output_kind=RequestOutputKind.DELTA, max_tokens=req["max_tokens"])
    ip.process_inputs(f"r{random.random()}", ei, sp, GENERATION_TASKS)
    r["engine"] = time.perf_counter() - s
    r["total"] = r["loads"] + r["pre"] + r["engine"]
    return r

async def main():
    gens = []; it = iter(ids)
    def refill():
        for cid in it:
            g = requests_of(load_conv(cid)); return g
        return None
    for _ in range(a.sessions):
        g = refill(); gens.append(g)
    rows = []; prof = cProfile.Profile(); n = 0
    while len(rows) < a.max_requests and any(gens):
        for i, g in enumerate(gens):
            if g is None: continue
            req = next(g, None)
            if req is None:
                gens[i] = refill(); continue
            on = a.max_requests // 3 <= n < a.max_requests // 3 + a.cprofile
            if on: prof.enable()
            r = await one(req)
            if on: prof.disable()
            r["session"] = i; rows.append(r); n += 1
            if n % 50 == 0:
                print(f"{n} reqs; last {r['tokens']} tok {r['total']*1e3:.1f} ms", flush=True)
            if len(rows) >= a.max_requests: break
    s = io.StringIO(); pstats.Stats(prof, stream=s).sort_stats("tottime").print_stats(35)
    s2 = io.StringIO(); pstats.Stats(prof, stream=s2).sort_stats("cumulative").print_stats(45)
    return rows, s.getvalue(), s2.getvalue()

rows, ptot, pcum = asyncio.run(main())
def summ(k):
    v = [r[k] for r in rows]; v.sort()
    return {"mean": statistics.fmean(v), "p50": v[len(v)//2], "p90": v[int(len(v)*0.9)], "sum": sum(v)}
keys = ["loads", "validate", "render", "tokenize", "pre", "engine", "total"]
S = {k: summ(k) for k in keys}
tok_mean = statistics.fmean(r["tokens"] for r in rows); by_mean = statistics.fmean(r["bytes"] for r in rows)
print(f"\n{len(rows)} requests, mean prompt {tok_mean:.0f} tokens, mean body {by_mean/1e6:.2f} MB")
for k in keys:
    print(f"{k:9s} mean {S[k]['mean']*1e3:8.2f} ms  p50 {S[k]['p50']*1e3:8.2f}  p90 {S[k]['p90']*1e3:8.2f}  share {S[k]['sum']/S['total']['sum']*100:5.1f}%")
print(f"single-process capacity (1/mean total) = {1/S['total']['mean']:.1f} req/s")
json.dump({"summary": S, "mean_tokens": tok_mean, "mean_bytes": by_mean, "rows": rows,
           "cprofile_tottime": ptot, "cprofile_cumulative": pcum}, open(a.out, "w"))
print(ptot[:6000]); print(pcum[:7000])
