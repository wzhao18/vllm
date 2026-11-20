"""Passive Dynamo-frontend probe (observation only; no behaviour change). Active only when FE_PROBE_DIR is set.

Patches dynamo.frontend.vllm_processor.VllmProcessor after it is imported:
  generator()            -> t_enter  : the request's Python coroutine first runs on the asyncio loop
  _generate_and_stream() -> t_pre    : preprocessing (validate + render + tokenize + process_inputs) done, engine call next
  first yielded item     -> t_first  : first output chunk handed back to Rust
and runs an event-loop lag sampler (sleep 50 ms, record the overshoot). Writes JSON lines to
$FE_PROBE_DIR/probe-<host>-<pid>.jsonl: {"rid","t_enter","t_pre","t_first","n_msgs","chars"} per request and
{"lag": [p50, p99, max], "t", "inflight"} every 10 s. rid = the Rust request context id (= request_id in the
frontend log's `request received` line), so t_enter - received is the Rust->Python handoff wait.
"""
import os

if os.environ.get("FE_SWITCH_INTERVAL"):  # experiment knob: CPython GIL switch interval (default 0.005 s)
    import sys as _s
    _s.setswitchinterval(float(os.environ["FE_SWITCH_INTERVAL"]))

import importlib.abc, importlib.machinery, sys
_PATCHES = {}

if os.environ.get("FE_PROBE_DIR"):
    def _patch(mod):
        import asyncio, json, socket, time
        P = mod.VllmProcessor
        out = open(os.path.join(os.environ["FE_PROBE_DIR"], f"probe-{socket.gethostname().split('.')[0]}-{os.getpid()}.jsonl"), "a", buffering=1)
        state = {"inflight": 0, "lag": [], "started": False}
        cur = {}

        async def _lagger():
            while True:
                t = time.perf_counter(); await asyncio.sleep(0.05)
                state["lag"].append(time.perf_counter() - t - 0.05)
                if len(state["lag"]) >= 200:
                    v = sorted(state["lag"]); state["lag"] = []
                    out.write(json.dumps({"t": time.time(), "lag": [v[len(v)//2], v[int(len(v)*.99)], v[-1]], "inflight": state["inflight"]}) + "\n")

        orig_gen, orig_gs = P.generator, P._generate_and_stream

        async def generator(self, request, context=None):
            t_enter = time.time()
            if not state["started"]:
                state["started"] = True; asyncio.get_running_loop().create_task(_lagger())
            rid = None
            try:
                rid = context.id() if context is not None and callable(getattr(context, "id", None)) else None
            except Exception:
                rid = None
            msgs = request.get("messages") or []
            rec = {"rid": rid, "t_enter": t_enter, "t_pre": None, "t_first": None, "n_msgs": len(msgs),
                   "chars": sum(len(m.get("content") or "") if isinstance(m.get("content"), str) else 0 for m in msgs if isinstance(m, dict))}
            task = asyncio.current_task(); cur[id(task)] = rec; state["inflight"] += 1
            try:
                async for item in orig_gen(self, request, context=context):
                    if rec["t_first"] is None:
                        rec["t_first"] = time.time()
                    yield item
            finally:
                state["inflight"] -= 1; cur.pop(id(task), None)
                out.write(json.dumps(rec) + "\n")

        async def _generate_and_stream(self, *a, **kw):
            rec = cur.get(id(asyncio.current_task()))
            if rec is not None and rec["t_pre"] is None:
                rec["t_pre"] = time.time()
            async for item in orig_gs(self, *a, **kw):
                yield item

        P.generator, P._generate_and_stream = generator, _generate_and_stream

    _PATCHES["dynamo.frontend.vllm_processor"] = _patch

if os.environ.get("FE_INLINE_RENDER") == "1":
    # Experiment: run the K3 renderer's render+tokenize inline on the asyncio loop instead of hopping to the
    # renderer's single-worker thread pool (same function, same result; only the thread it runs on changes).
    def _patch_k3(mod):
        cls = mod.KimiK3Renderer
        orig_init = cls.__init__
        def __init__(self, *a, **kw):
            orig_init(self, *a, **kw)
            sync = self._apply_chat_template
            async def _inline(conversation, params):
                return sync(conversation, params)
            self._apply_chat_template_async = _inline
        cls.__init__ = __init__
    _PATCHES["vllm.renderers.kimi_k3"] = _patch_k3


class _Loader(importlib.abc.Loader):
    def __init__(self, inner, fn): self.inner, self.fn = inner, fn
    def create_module(self, spec): return self.inner.create_module(spec)
    def exec_module(self, module):
        self.inner.exec_module(module)
        try:
            self.fn(module)
        except Exception as e:  # never break the frontend
            sys.stderr.write(f"fe_probe: patch of {module.__name__} failed: {e!r}\n")


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        fn = _PATCHES.get(name)
        if fn is None:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is not None and spec.loader is not None:
            spec.loader = _Loader(spec.loader, fn)
        return spec


if _PATCHES:
    sys.meta_path.insert(0, _Finder())

# Chain to the sitecustomize this file shadows (e.g. the distro's /usr/lib/python3.X/sitecustomize.py).
def _chain():
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    for d in sys.path:
        p = os.path.join(d, "sitecustomize.py")
        if d and os.path.abspath(d) != here and os.path.isfile(p):
            try:
                exec(compile(open(p).read(), p, "exec"), {"__name__": "sitecustomize", "__file__": p})
            except Exception:
                pass
            return
_chain()
