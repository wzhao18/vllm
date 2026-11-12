# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persistent compile cache of the Kimi K3 CuTe DSL kernels, keyed before tracing.

``cute.compile`` keeps its result in memory only (a compile-only call forces the
DSL's ``no_cache``), and the DSL's own key is the *traced* IR, so even a cached
compile pays the Python tracing. A Kimi K3.1 decode start issues ~10K compiles
per rank (specialized decode providers, KV-layout cores, drafter, KDA head): 53 of
its 64 minutes, GPUs idle.

``install()`` wraps ``cute.compile`` / ``cute.experimental.compile`` once per
process. Compiles of kernels defined under ``vllm/models/kimi_k3`` go through
``_compile``; every other caller gets the DSL call unchanged.

- Key (taken before tracing): the digest of every ``.py`` file of the ``vllm``
  package (any source edit is a new key space), the CuTe DSL / torch / Python
  versions, the GPU (capability, SM count), the DSL and Kimi K3 environment, the
  compile options, and a deep fingerprint of the kernel callable (the kernel
  object's state, function bytecode, closures and defaults) and of every
  argument (fake tensor layouts, scalars, real tensors with their address). An
  argument the fingerprint cannot describe makes that compile uncached: it is
  compiled in memory, counted and named in the cache log, never silently served.
- Entry: the AOT object of the compile (``export_to_c`` for TVM-FFI launchers,
  ``dump_to_object`` otherwise) and a JSON record (launcher kind, DSL module hash
  of the traced IR, compile seconds), written by atomic rename: ranks, nodes and
  processes share one directory without locks. A hit loads the object
  (``cute.runtime.load_module``): the cubin the compile produced, so kernels are
  bitwise the compiled ones, and neither tracing nor the MLIR pipeline runs. A
  miss compiles, exports and returns the loaded object, so cold and warm starts
  run the same launcher. An entry that does not load is logged, recompiled and
  replaced.
- ``VERIFY`` (tests only): every hit is also traced and its DSL module hash must
  equal the stored one, which checks that the pre-trace key covers every trace
  input of the kernels a start compiles.
"""

import ctypes
import dataclasses
import enum
import functools
import hashlib
import json
import marshal
import os
import shutil
import socket
import sys
import tempfile
import threading
import time
import types
from dataclasses import dataclass, field
from inspect import Parameter
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# Bumped when the key or the entry layout changes.
FORMAT = 1
# Tests: re-trace every hit and compare the DSL module hash with the stored one.
VERIFY = False

_MAX_NODES = 200_000  # fingerprint object-graph size limit (uncached above)
# Per-run environment (paths, hosts, ports, timeouts): never read by a trace.
_RUN_ENV_SUFFIXES = (
    "_DIR", "_PATH", "_ROOT", "_HOST", "_PORT", "_ADDR", "_FILE", "_TIMEOUT",
    "_TIMEOUT_S", "_TIMEOUT_SECONDS", "_URL", "_IP", "_ID", "_NAME",
)  # fmt: skip
_ENV_PREFIXES = ("CUTE_DSL_", "CUTLASS_", "VLLM_", "K3", "K31", "KIMI")
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))  # fmt: skip  (the ``vllm`` package directory)
_K3_ROOT = os.path.join(_PKG_ROOT, "models", "kimi_k3") + os.sep


class _Uncacheable(Exception):
    """The fingerprint cannot describe an input: compile without the cache."""


@dataclass
class Stats:
    loaded: int = 0  # hits
    compiled: int = 0  # misses: traced, compiled, exported
    uncached: int = 0  # compiled in memory only
    key_s: float = 0.0  # fingerprinting
    load_s: float = 0.0
    compile_s: float = 0.0  # trace + MLIR + export of misses and uncached
    # what made compiles uncached -> count
    reasons: dict = field(default_factory=dict)
    # kernel -> [hits, misses, uncached, compile seconds]
    kernels: dict = field(default_factory=dict)


stats = Stats()
_lock = threading.Lock()
_events = 0
_started = time.perf_counter()


# --------------------------------------------------------------------------- key


def _qual(obj: Any) -> str:
    return f"{getattr(obj, '__module__', '?')}.{getattr(obj, '__qualname__', '?')}"


@functools.cache
def _foreign_module(name: str) -> str:
    """A module outside the vllm tree: its file's identity (path, size, mtime)."""
    module = sys.modules.get(name)
    path = getattr(module, "__file__", None)
    if not path:
        return f"{name}:builtin"
    if os.path.abspath(path).startswith(_PKG_ROOT + os.sep):
        return f"{name}:tree"
    try:
        st = os.stat(path)
    except OSError:
        return f"{name}:{path}:?"
    return f"{name}:{path}:{st.st_size}:{st.st_mtime_ns}"


class _Fingerprint:
    """Deterministic description of an object graph (cycles by back reference)."""

    _SCALARS = (type(None), bool, int, float, complex, str, bytes)

    def __init__(self):
        self.parts: list[str] = []
        self.memo: dict[int, int] = {}
        self.keep: list[Any] = []  # memoized objects stay alive: ids stay unique
        self.modules: set[str] = set()
        self.nodes = 0

    def digest(self) -> str:
        foreign = sorted(_foreign_module(m) for m in self.modules)
        text = "\x1f".join(self.parts) + "\x1e" + "\x1f".join(foreign)
        return hashlib.sha256(text.encode()).hexdigest()

    def _code(self, code: types.CodeType) -> str:
        return hashlib.sha256(marshal.dumps(code)).hexdigest()[:32]

    def _classes(self, cls: type) -> None:
        """A class's bases (their code runs in the trace too: e.g. a Kimi K3 subclass
        of a tokenspeed kernel) by name and module."""
        for k in cls.__mro__:
            self.parts.append(f"mro:{_qual(k)}")
            self.modules.add(getattr(k, "__module__", "?") or "?")

    def add(self, o: Any) -> None:
        self.nodes += 1
        if self.nodes > _MAX_NODES:
            raise _Uncacheable("object graph too large")
        w, t = self.parts.append, type(o)
        if t in self._SCALARS:
            w(f"{t.__name__}:{o!r}")
            return
        oid = id(o)
        ref = self.memo.get(oid)
        if ref is not None:
            w(f"@{ref}")
            return
        self.memo[oid] = len(self.memo)
        self.keep.append(o)
        module = getattr(t, "__module__", None)
        if module:
            self.modules.add(module)
        if isinstance(o, enum.Enum):
            w(f"E:{_qual(t)}:{o.name}")
            self.add(o.value)
        elif isinstance(o, self._SCALARS):  # subclasses (IntEnum handled above)
            w(f"S:{_qual(t)}:{o!r}")
        elif isinstance(o, (tuple, list)):
            w(f"L:{_qual(t)}:{len(o)}")
            for item in o:
                self.add(item)
        elif isinstance(o, dict):
            w(f"D:{_qual(t)}:{len(o)}")
            for k, v in o.items():
                self.add(k)
                self.add(v)
        elif isinstance(o, (set, frozenset)):
            items = []
            for item in o:
                sub = _Fingerprint()
                sub.add(item)
                self.modules |= sub.modules
                items.append(sub.digest())
            w(f"Set:{_qual(t)}:{','.join(sorted(items))}")
        elif isinstance(o, type):
            w(f"T:{_qual(o)}")
            self._classes(o)
        elif isinstance(o, types.FunctionType):
            w(f"F:{_qual(o)}:{self._code(o.__code__)}")
            self.modules.add(o.__module__ or "?")
            self.add(o.__defaults__)
            self.add(o.__kwdefaults__)
            for cell in o.__closure__ or ():
                try:
                    self.add(cell.cell_contents)
                except ValueError:  # empty cell
                    w("cell:empty")
        elif isinstance(o, types.MethodType):
            w("M")
            self.add(o.__func__)
            self.add(o.__self__)
        elif isinstance(o, (types.BuiltinFunctionType, types.BuiltinMethodType)):
            w(f"B:{_qual(o)}")
            owner = getattr(o, "__self__", None)
            if owner is not None and not isinstance(owner, types.ModuleType):
                self.add(owner)
        elif isinstance(o, types.ModuleType):
            w(f"Mod:{o.__name__}")
            self.modules.add(o.__name__)
        elif isinstance(o, functools.partial):
            w("P")
            self.add(o.func)
            self.add(o.args)
            self.add(o.keywords)
        elif isinstance(o, (staticmethod, classmethod)):
            w(f"SM:{t.__name__}")
            self.add(o.__func__)
        elif isinstance(o, property):
            w("Prop")
            self.add((o.fget, o.fset, o.fdel))
        elif isinstance(o, torch.Tensor):
            ptr = o.data_ptr() if o.device.type != "meta" else 0
            w(
                f"Tensor:{o.dtype}:{tuple(o.shape)}:{o.stride()}:"
                f"{o.storage_offset()}:{o.device}:{ptr}"
            )
        elif isinstance(
            o, (torch.dtype, torch.device, torch.Size, torch.layout, torch.memory_format)
        ):
            w(f"t:{o!r}")
        elif isinstance(o, ctypes._SimpleCData):
            w(f"C:{t.__name__}:{o.value!r}")
        else:
            state = getattr(o, "__dict__", None)
            slots = [
                s
                for k in t.__mro__
                for s in getattr(k, "__slots__", ())
                if s not in ("__dict__", "__weakref__")
            ]
            if not isinstance(state, dict) and not slots:
                raise _Uncacheable(_qual(t))
            w(f"O:{_qual(t)}")
            self._classes(t)
            if isinstance(state, dict):
                self.add(state)
            for name in slots:
                w(f".{name}")
                try:
                    self.add(getattr(o, name))
                except AttributeError:
                    w("unset")


@functools.cache
def _source_digest() -> str:
    """Every ``.py`` of the vllm package, by relative path and content."""
    h = hashlib.sha256()
    for dirpath, dirnames, filenames in os.walk(_PKG_ROOT):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for name in sorted(filenames):
            if name.endswith(".py"):
                path = os.path.join(dirpath, name)
                h.update(os.path.relpath(path, _PKG_ROOT).encode() + b"\0")
                with open(path, "rb") as f:
                    h.update(hashlib.sha256(f.read()).digest())
    return h.hexdigest()


# Packages outside the vllm tree whose Python code Kimi K3 kernels run while tracing
# (base classes, helpers): each enters the key by version and install location.
_FOREIGN_PACKAGES = ("cutlass", "tvm_ffi", "flashinfer", "tokenspeed_mla", "cuda", "triton")


@functools.cache
def _package(name: str) -> str:
    import importlib.metadata
    import importlib.util

    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        spec = None
    if spec is None or not spec.origin:
        return f"{name} absent"
    try:
        version = getattr(sys.modules.get(name), "__version__", None) or next(
            iter(importlib.metadata.packages_distributions().get(name, ())), "?"
        )
    except Exception:
        version = "?"
    st = os.stat(spec.origin)
    return f"{name} {version} {spec.origin} {st.st_size} {st.st_mtime_ns}"


def _versions() -> str:
    import cutlass

    parts = [f"f{FORMAT}", f"cutlass-dsl {cutlass.__version__}"]
    parts += [f"torch {torch.__version__}", f"python {sys.version}"]
    parts += [_package(name) for name in _FOREIGN_PACKAGES]
    return "|".join(parts)


def _environment() -> str:
    """The GPU and the trace-relevant environment (read per compile: launchers set
    CUTE_DSL_ARCH during startup)."""
    parts = []
    if torch.cuda.is_initialized():
        index = torch.cuda.current_device()
        p = torch.cuda.get_device_properties(index)
        parts.append(f"gpu {p.name} {p.major}.{p.minor} {p.multi_processor_count}")
    else:
        parts.append("gpu none")
    for k in sorted(os.environ):
        if k.startswith(_ENV_PREFIXES) and not k.endswith(_RUN_ENV_SUFFIXES):
            parts.append(f"{k}={os.environ[k]}")
    return "\n".join(parts)


def key_of(compile_fn: Any, fn: Any, args: tuple, kwargs: dict) -> str:
    """The pre-trace key of ``compile_fn(fn, *args, **kwargs)`` (raises
    ``_Uncacheable`` when an input cannot be described)."""
    fp = _Fingerprint()
    fp.add(getattr(compile_fn, "_compile_options", None))
    fp.add(fn)
    fp.add(args)
    fp.add(kwargs)
    head = f"{_versions()}\n{_source_digest()}\n{_environment()}\n"
    return hashlib.sha256((head + fp.digest()).encode()).hexdigest()


# ------------------------------------------------------------------------- store


def cache_dir() -> str:
    from vllm import envs

    import cutlass

    version = f"cutlass-dsl-{cutlass.__version__}-f{FORMAT}"
    return os.path.join(envs.VLLM_CACHE_ROOT, "kimi_k3", "cutedsl_pretrace", version)


def _paths(key: str) -> tuple[str, str]:
    base = os.path.join(cache_dir(), key[:2], key)
    return base + ".o", base + ".json"


def _prefix(key: str) -> str:
    return f"k3c_{key[:32]}"


def _write_atomic(path: str, write) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    work = tempfile.mkdtemp(prefix=".w-", dir=os.path.dirname(path))
    try:
        tmp = os.path.join(work, "entry")
        write(tmp)
        os.replace(tmp, path)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _cacheable(compiled: Any, tvm_ffi: bool) -> bool:
    """Whether the loaded object calls exactly like ``compiled``: a TVM-FFI object
    loads as the plain ``tvm_ffi.Function``, which is ``compiled``'s call when every
    argument is positional without a default and none is a raw pointer address."""
    if not tvm_ffi:
        return True
    arguments = compiled.execution_args
    positional = (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
    return not arguments.has_pointer_address_arg_specs and all(
        p.kind in positional
        and p.default is Parameter.empty
        and not dataclasses.is_dataclass(p.annotation)
        for p in arguments.signature.parameters.values()
    )


def _load(key: str, tvm_ffi: bool) -> Any:
    from cutlass.cute.runtime import load_module

    module = load_module(_paths(key)[0], enable_tvm_ffi=tvm_ffi)
    return module[_prefix(key)]


def _read_record(key: str) -> dict | None:
    obj, meta = _paths(key)
    if not os.path.exists(obj):
        return None
    with open(meta) as f:  # written before the object: present with it
        return json.load(f)


# ----------------------------------------------------------------------- compile


def _kernel_name(fn: Any) -> str:
    target = getattr(fn, "__func__", fn)
    if not isinstance(target, (types.FunctionType, type)):
        target = type(target)
    return getattr(target, "__qualname__", "?")


def _record(name: str, slot: int | None, seconds: float = 0.0, **times) -> None:
    """Add ``times`` to the totals and, with a ``slot`` (0 hit, 1 miss, 2 uncached),
    one event of kernel ``name``."""
    global _events
    with _lock:
        for k, v in times.items():
            setattr(stats, k, getattr(stats, k) + v)
        if slot is None:
            return
        entry = stats.kernels.setdefault(name, [0, 0, 0, 0.0])
        entry[slot] += 1
        entry[3] += seconds
        _events += 1
        report = _events % 500 == 0
    if report:
        log_summary("progress")


def _trace_hook(found: dict):
    def hook(owner, module, function_name):
        found["ir"] = owner.get_module_hash(module, function_name)
        options, settings = owner.compile_options, owner.envar
        found["tvm_ffi"] = bool(options.enable_tvm_ffi or settings.enable_tvm_ffi)

    return hook


class _TraceDone(Exception):
    """Raised by the VERIFY hook: the module hash is taken, skip the MLIR pipeline."""


def _trace_only(inner, fn, args, kwargs) -> dict:
    """Trace ``fn`` and return its DSL module hash without compiling it."""
    found: dict = {}
    record = _trace_hook(found)

    def hook(owner, module, function_name):
        record(owner, module, function_name)
        raise _TraceDone

    try:
        inner(fn, *args, trace_finalize_hooks=hook, **kwargs)
    except Exception as error:  # the DSL wraps the hook's exception
        seen = set()
        while error is not None and id(error) not in seen:
            if isinstance(error, _TraceDone):
                return found
            seen.add(id(error))
            error = error.__cause__ or error.__context__
        raise
    return found


def _compile_and_store(inner, fn, args, kwargs, key: str | None, name: str):
    found: dict = {}
    start = time.perf_counter()
    compiled = inner(fn, *args, trace_finalize_hooks=_trace_hook(found), **kwargs)
    tvm_ffi = found.get("tvm_ffi", False)
    if key is None or "ir" not in found or not _cacheable(compiled, tvm_ffi):
        if key is not None:
            why = "no module hash" if "ir" not in found else "loaded launcher differs"
            with _lock:
                reason = f"{name}: {why}"
                stats.reasons[reason] = stats.reasons.get(reason, 0) + 1
        _record(name, 2, time.perf_counter() - start, uncached=1,
                compile_s=time.perf_counter() - start)  # fmt: skip
        return compiled
    obj, meta = _paths(key)
    record = {
        "kernel": name,
        "tvm_ffi": tvm_ffi,
        "ir": found["ir"],
        "compile_s": round(time.perf_counter() - start, 3),
        "host": socket.gethostname(),
    }

    def write_obj(path):
        if tvm_ffi:
            compiled.export_to_c(path, _prefix(key))
        else:  # the object only: no C header (it rejects *args launchers)
            with open(path, "wb") as f:
                f.write(compiled.dump_to_object(_prefix(key)))

    def write_meta(path):
        with open(path, "w") as f:
            json.dump(record, f)

    try:
        _write_atomic(meta, write_meta)
        _write_atomic(obj, write_obj)
        loaded = _load(key, tvm_ffi)
    except Exception as error:
        logger.warning(
            "Kimi K3 CuTe cache: %s compiled but its object cannot be stored or "
            "loaded (%s: %s); this compile stays in memory",
            name, type(error).__name__, str(error)[:300],
        )  # fmt: skip
        with _lock:
            reason = f"{name}: export {type(error).__name__}"
            stats.reasons[reason] = stats.reasons.get(reason, 0) + 1
        seconds = time.perf_counter() - start
        _record(name, 2, seconds, uncached=1, compile_s=seconds)
        return compiled
    seconds = time.perf_counter() - start
    _record(name, 1, seconds, compiled=1, compile_s=seconds)
    return loaded


def _compile(inner, fn, args: tuple, kwargs: dict):
    name = _kernel_name(fn)
    start = time.perf_counter()
    try:
        key = key_of(inner, fn, args, kwargs)
    except _Uncacheable as why:
        with _lock:
            reason = f"{name}: {why}"
            stats.reasons[reason] = stats.reasons.get(reason, 0) + 1
        key = None
    _record(name, None, key_s=time.perf_counter() - start)
    if key is None:
        return _compile_and_store(inner, fn, args, kwargs, None, name)
    record = _read_record(key)
    if record is None:
        return _compile_and_store(inner, fn, args, kwargs, key, name)
    loaded_at = time.perf_counter()
    try:
        loaded = _load(key, record["tvm_ffi"])
    except Exception as error:
        logger.warning(
            "Kimi K3 CuTe cache: entry %s (%s) does not load (%s: %s); recompiling "
            "and replacing it", key[:16], name, type(error).__name__, str(error)[:300],
        )  # fmt: skip
        return _compile_and_store(inner, fn, args, kwargs, key, name)
    if VERIFY:
        found = _trace_only(inner, fn, args, kwargs)
        if found.get("ir") != record["ir"]:
            raise RuntimeError(
                f"Kimi K3 CuTe cache: key {key[:16]} of {name} hit an entry traced "
                f"to a different module (stored {record['ir'][:16]}, now "
                f"{str(found.get('ir'))[:16]}): the pre-trace key misses an input"
            )
    _record(name, 0, 0.0, loaded=1, load_s=time.perf_counter() - loaded_at)
    return loaded


class _CachedCompile:
    """``cute.compile`` (or ``cute.experimental.compile``) with the cache in front
    for Kimi K3 kernels; ``[options]`` and the other attributes pass through."""

    _k3_cached = True

    def __init__(self, inner):
        self._inner = inner

    def __getitem__(self, options):
        return _CachedCompile(self._inner[options])

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __call__(self, fn, *args, **kwargs):
        if not _is_k3_kernel(fn) or "trace_finalize_hooks" in kwargs:
            return self._inner(fn, *args, **kwargs)
        return _compile(self._inner, fn, args, kwargs)


def _is_k3_kernel(fn: Any) -> bool:
    target = getattr(fn, "__func__", fn)
    if not isinstance(target, (types.FunctionType, type)):
        target = type(target)
    module = sys.modules.get(getattr(target, "__module__", "") or "")
    path = getattr(module, "__file__", None)
    return bool(path) and os.path.abspath(path).startswith(_K3_ROOT)


def install() -> None:
    """Put the cache in front of ``cute.compile`` and ``cute.experimental.compile``
    (once per process; a no-op without the CuTe DSL)."""
    try:
        import cutlass.cute as cute
        import cutlass.cute.experimental as experimental
        import cutlass.cute.export  # noqa: F401  (sets the AOT load provider)
    except ImportError:
        return
    for owner in (cute, experimental):
        if not getattr(owner.compile, "_k3_cached", False):
            owner.compile = _CachedCompile(owner.compile)


def log_summary(stage: str) -> None:
    with _lock:
        s = Stats(**{k: v for k, v in vars(stats).items()
                     if k not in ("kernels", "reasons")})  # fmt: skip
        kernels = sorted(stats.kernels.items(), key=lambda kv: -kv[1][3])[:8]
        reasons = sorted(stats.reasons.items(), key=lambda kv: -kv[1])[:5]
    logger.info(
        "Kimi K3 CuTe cache (%s, %.0f s since import): %d loaded, %d compiled, %d "
        "uncached; keys %.0f s, loads %.0f s, compiles %.0f s (%s); slowest: %s%s",
        stage, time.perf_counter() - _started, s.loaded, s.compiled, s.uncached,
        s.key_s, s.load_s, s.compile_s, cache_dir(),
        ", ".join(f"{n} {e[1] + e[2]}x {e[3]:.0f} s" for n, e in kernels),
        "; uncached: " + ", ".join(f"{r} {c}x" for r, c in reasons) if reasons else "",
    )  # fmt: skip
