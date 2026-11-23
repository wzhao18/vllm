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

- Key (taken before tracing): the CuTe DSL / torch / Python versions and the
  versions of the foreign packages kernels run while tracing, the GPU (capability,
  SM count), the DSL and Kimi K3 environment, the target GPU architecture and the
  full resolved compile option set (``_compile_target``), and a deep
  fingerprint of the kernel callable (the kernel object's state, function bytecode,
  closures and defaults) and of every argument (fake tensor layouts, scalars, real
  tensors with their address, host tensors also with their contents). An argument
  the fingerprint cannot describe makes that compile uncached: it is compiled in
  memory, counted and named in the cache log, never silently served.
- Sources: an entry records the vllm files its trace depends on (``_sources``: the
  modules of the kernel and its arguments, every file whose code ran during the
  trace, and their direct imports) with their content digests, and is a hit only
  while all of them are unchanged. An edit elsewhere in the tree keeps the entry;
  an edit to one of its files adds a new entry for the key next to the old one.
- Entry: the AOT object of the compile (``export_to_c`` for TVM-FFI launchers,
  ``dump_to_object`` otherwise) and a JSON record (launcher kind, DSL module hash
  of the traced IR, compile seconds, sources), written by atomic rename. A hit
  loads the object (``cute.runtime.load_module``): the cubin the compile produced,
  so kernels are bitwise the compiled ones, and neither tracing nor the MLIR
  pipeline runs. A miss compiles, exports and returns the loaded object, so cold
  and warm starts run the same launcher. An entry that does not load is logged,
  recompiled and replaced.
- Directory: ``<base>/<DSL version, FORMAT>/<install identity>``, base =
  ``VLLM_KIMI_K3_CUTE_CACHE_DIR`` (a location only; not part of any key) or
  ``<VLLM_CACHE_ROOT>/kimi_k3/cutedsl_pretrace``. The install identity
  (``install_identity``) hashes the installed distributions and the files of the
  foreign packages kernels trace with, so a new image, an overlay or a patched
  package file starts a new directory instead of hitting entries of the old code.
- Shared cold starts: ranks, nodes and processes share the directory (put
  ``VLLM_KIMI_K3_CUTE_CACHE_DIR`` on a shared file system for cross-node
  sharing). A miss takes the key's claim file; a process that finds it taken gets
  a ``_Deferred`` kernel and goes on, so the ranks of a deployment compile disjoint
  parts of the key set in parallel and load the rest (see "cross-process claims").
- ``VERIFY`` (tests and validation starts only; a module attribute, no
  environment switch): every hit is also traced and its DSL module hash must equal
  the stored one, which checks that the pre-trace key covers every trace input of
  the kernels a start compiles. A mismatch raises ``VerifyMismatch`` (a
  BaseException: kernel compile fallbacks cannot swallow it).
"""

import ast
import contextlib
import ctypes
import dataclasses
import enum
import functools
import hashlib
import json
import marshal
import math
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
FORMAT = 4
# Tests: re-trace every hit and compare the DSL module hash with the stored one.
VERIFY = False

_MAX_NODES = 200_000  # fingerprint object-graph size limit (uncached above)
# Per-run environment (paths, hosts, ports, timeouts, the launcher's container name,
# node list and time budget: K31_CNAME / K31_NODES / K31_MIN_LEFT_S): never read by a
# trace. K31_CNAME carries the launch timestamp, so keying on it missed every entry on
# each restart (CUTECACHE keydiff of two decode starts on one hold, 2026-11-22).
_RUN_ENV_SUFFIXES = (
    "_DIR", "_PATH", "_ROOT", "_HOST", "_PORT", "_ADDR", "_FILE", "_TIMEOUT",
    "_TIMEOUT_S", "_TIMEOUT_SECONDS", "_URL", "_IP", "_ID", "_NAME", "_CNAME",
    "_NODES", "_NODELIST", "_LEFT_S",
)  # fmt: skip
_ENV_PREFIXES = ("CUTE_DSL_", "CUTLASS_", "VLLM_", "K3", "K31", "KIMI")
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))  # fmt: skip  (the ``vllm`` package directory)
_K3_ROOT = os.path.join(_PKG_ROOT, "models", "kimi_k3") + os.sep


class _Uncacheable(Exception):
    """The fingerprint cannot describe an input: compile without the cache."""


_IR_VALUES = ("OpResult", "BlockArgument", "Value")
# Runtime arguments over real memory (from_dlpack, make_ptr). Not _FakeTensor: its DSL
# key holds id()s of its SymInts, and its attributes fingerprint as they are.
_DSL_RUNTIME_ARGS = ("cutlass.cute.runtime._Tensor", "cutlass.cute.runtime._Pointer")


def _dsl_runtime_arg(t: type) -> bool:
    return _qual(t) in _DSL_RUNTIME_ARGS and isinstance(
        getattr(t, "__cache_key__", None), property
    )


@dataclass
class Stats:
    loaded: int = 0  # hits
    compiled: int = 0  # misses: traced, compiled, exported
    uncached: int = 0  # compiled in memory only
    key_s: float = 0.0  # fingerprinting
    load_s: float = 0.0
    compile_s: float = 0.0  # trace + MLIR + export of misses and uncached
    deferred: int = 0  # claimed by another process when first asked for
    wait_s: float = 0.0  # resolving deferred kernels (waiting for peers)
    verified: int = 0  # VERIFY: hits re-traced to the stored module hash
    mismatched: int = 0  # VERIFY: hits that did not (raised VerifyMismatch)
    # what made compiles uncached -> count
    reasons: dict = field(default_factory=dict)
    # kernel -> [hits, misses, uncached, compile seconds, deferred]
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
        self.namespaces: set[int] = set()  # local classes whose namespace was added
        self.nodes = 0

    def digest(self) -> str:
        foreign = sorted(_foreign_module(m) for m in self.modules)
        text = "\x1f".join(self.parts) + "\x1e" + "\x1f".join(foreign)
        return hashlib.sha256(text.encode()).hexdigest()

    def _code(self, code: types.CodeType) -> str:
        return hashlib.sha256(marshal.dumps(code)).hexdigest()[:32]

    # Class namespace entries that are not trace inputs.
    _CLASS_SKIP = frozenset((
        "__dict__", "__weakref__", "__doc__", "__module__", "__qualname__",
        "__firstlineno__", "__static_attributes__",
    ))  # fmt: skip

    def _classes(self, cls: type) -> None:
        """A class's bases (their code runs in the trace too: e.g. a Kimi K3 subclass
        of a tokenspeed kernel) by name and module. A class created inside a function
        (``<locals>`` in its qualified name, e.g. ``_without_reducer(polled_core(
        batch, ...))``) is also described by its namespace: its methods' code,
        defaults and closures carry the factory's arguments, which neither the name
        nor the source files tell apart."""
        local = []
        for k in cls.__mro__:
            self.parts.append(f"mro:{_qual(k)}")
            self.modules.add(getattr(k, "__module__", "?") or "?")
            if "<locals>" in getattr(k, "__qualname__", ""):
                local.append(k)
        for k in local:
            if id(k) in self.namespaces:  # once per fingerprint (cycles via __class__)
                self.parts.append(f"ns@{_qual(k)}")
                continue
            self.namespaces.add(id(k))
            self.keep.append(k)
            names = [n for n in vars(k) if n not in self._CLASS_SKIP]
            self.parts.append(f"ns:{_qual(k)}:{len(names)}")
            for name in names:
                self.parts.append(f".{name}")
                self.add(vars(k)[name])

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
            # The DSL preprocessor replaces a jit function's __code__ on its first
            # compile and keeps the source's in _original_code: hash that one, so a
            # key does not depend on which kernels this process compiled before.
            code = getattr(o, "_original_code", None) or o.__code__
            w(f"F:{_qual(o)}:{self._code(code)}")
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
            if o.layout != torch.strided or o.is_quantized:
                raise _Uncacheable(f"tensor layout {o.layout}")
            ptr = o.data_ptr() if o.device.type != "meta" else 0
            w(
                f"Tensor:{o.dtype}:{tuple(o.shape)}:{o.stride()}:"
                f"{o.storage_offset()}:{o.device}:{ptr}"
            )
            if o.device.type == "cpu":  # a trace may read host values
                w(f"data:{_cpu_tensor_digest(o)}")
        elif isinstance(
            o, (torch.dtype, torch.device, torch.Size, torch.layout, torch.memory_format)
        ):
            w(f"t:{o!r}")
        elif _qual(t) == "cuda.bindings.driver.CUstream":
            # A launch argument, not traced into the kernel: its handle differs per
            # process (it made QueryUK and SparseMlaDecodeKM recompile every start).
            w("CUstream")
        elif _dsl_runtime_arg(t):
            # A CuTe DSL runtime argument (from_dlpack tensor, make_ptr pointer): the
            # DSL's own JIT cache key (dtype, layout marks, address space, assumed
            # alignment) is what a trace reads of it; the DLPack capsule and the
            # address are launch values (BlockScaledSwapAb*MoeKernel compiled
            # outside the cache at every start because of the capsule).
            ck = repr(o.__cache_key__)
            if " at 0x" in ck:  # an object without a value repr: not a stable description
                raise _Uncacheable(f"{_qual(t)} cache key {ck[:160]}")
            w(f"DSLArg:{_qual(t)}:{ck}")
        elif callable(getattr(t, "__cute_cache_key__", None)):
            # A launch argument that declares its trace-relevant part (e.g. the
            # MegaMoE SymmetricBufferHost: addresses are runtime fields).
            w(f"CCK:{_qual(t)}")
            self._classes(t)
            self.add(o.__cute_cache_key__())
        elif _qual(t).startswith("cutlass._mlir.") and t.__name__ in _IR_VALUES:
            # An IR value an earlier trace left on a kernel object: it belongs to
            # that trace's finished module and a new trace recomputes it.
            w(f"IRValue:{t.__name__}")
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
                raise _Uncacheable(f"{_qual(t)} at {' '.join(self.parts[-6:])[-240:]}")
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


def _cpu_tensor_digest(t: torch.Tensor) -> str:
    """Content digest of a host tensor (its logical elements, as bytes)."""
    flat = t.detach().contiguous().reshape(-1)
    if flat.numel() == 0:
        return "empty"
    return hashlib.sha256(flat.view(torch.uint8).numpy().tobytes()).hexdigest()[:32]


def _tree_file(path: str) -> str | None:
    """``path`` relative to the vllm package when it is one of its ``.py`` files."""
    if not path or not path.endswith(".py"):
        return None
    path = os.path.abspath(path)
    if not path.startswith(_PKG_ROOT + os.sep):
        return None
    return os.path.relpath(path, _PKG_ROOT)


@functools.cache
def _file_digest(rel: str) -> str:
    """Content digest of a vllm ``.py`` file (read once per process)."""
    try:
        with open(os.path.join(_PKG_ROOT, rel), "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:32]
    except OSError:
        return "missing"


def _module_file(name: str) -> str | None:
    """The vllm file of dotted module ``name`` (module or package), if any."""
    if name != "vllm" and not name.startswith("vllm."):
        return None
    base = os.path.join(_PKG_ROOT, *name.split(".")[1:])
    for path in (base + ".py", os.path.join(base, "__init__.py")):
        if os.path.isfile(path):
            return os.path.relpath(path, _PKG_ROOT)
    return None


@functools.cache
def _direct_imports(rel: str) -> tuple[str, ...]:
    """The vllm files a vllm file imports (any import statement, any scope)."""
    try:
        with open(os.path.join(_PKG_ROOT, rel), "rb") as f:
            tree = ast.parse(f.read())
    except (OSError, SyntaxError, ValueError):
        return ()
    parts = ["vllm", *rel[:-3].split(os.sep)]
    package = parts if parts[-1] == "__init__" else parts[:-1]
    if package[-1] == "__init__":
        package = package[:-1]
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                anchor = package[: len(package) - (node.level - 1)]
                base = ".".join(anchor + ([node.module] if node.module else []))
            else:
                base = node.module or ""
            names.append(base)
            names += [f"{base}.{a.name}" for a in node.names]
    found = {f for f in map(_module_file, names) if f is not None}
    found.discard(rel)
    return tuple(sorted(found))


def _sources(modules: set[str], executed: set[str]) -> dict[str, str]:
    """The vllm source files a kernel's trace depends on, with their digests: the
    modules of the kernel callable and its arguments (classes, functions), every
    file whose code ran during the trace, and the files those import directly (a
    value a traced function takes from a cached helper in an imported module, whose
    code then did not run again). Files outside the vllm package are covered by the
    package versions in the key."""
    seeds = {f for f in map(_module_file, modules) if f is not None}
    seeds |= {f for f in map(_tree_file, executed) if f is not None}
    files = set(seeds)
    for rel in seeds:
        files.update(_direct_imports(rel))
    return {rel: _file_digest(rel) for rel in sorted(files)}


class _Executed:
    """The Python files whose code runs inside the ``with`` block (each code object
    reported once per compile through ``sys.monitoring``; all threads)."""

    _tool: int | None = None

    def __init__(self):
        self.files: set[str] = set()

    def _start(self, code, offset):
        self.files.add(code.co_filename)
        return sys.monitoring.DISABLE

    def __enter__(self):
        mon = sys.monitoring
        if _Executed._tool is None:
            for tool in (4, 3, 5, 1):
                if mon.get_tool(tool) is None:
                    mon.use_tool_id(tool, "kimi_k3_cute_cache")
                    _Executed._tool = tool
                    break
            else:
                raise RuntimeError("Kimi K3 CuTe cache: no free sys.monitoring tool")
        tool = _Executed._tool
        mon.register_callback(tool, mon.events.PY_START, self._start)
        mon.set_events(tool, mon.events.PY_START)
        mon.restart_events()  # code reported (disabled) by an earlier compile
        return self

    def __exit__(self, *exc):
        mon = sys.monitoring
        mon.set_events(_Executed._tool, 0)
        mon.register_callback(_Executed._tool, mon.events.PY_START, None)
        return False


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


def _jit_function(fn: Any) -> Any:
    """The ``@cute.jit`` function ``cute.compile(fn, ...)`` compiles (as the DSL
    resolves it: bound method, callable instance, decorator wrappers)."""
    target = fn
    if isinstance(target, types.MethodType):
        target = target.__func__
    elif not isinstance(target, types.FunctionType):
        call = getattr(type(target), "__call__", None)
        target = getattr(call, "__func__", call)
    while hasattr(target, "__wrapped__"):
        target = target.__wrapped__
    return target


def _fallback_arch() -> str:
    """The DSL's own default when no DSL object is reachable: ``CUTE_DSL_ARCH``,
    else the current device (``sm_<major><minor>a`` from Hopper on)."""
    arch = os.environ.get("CUTE_DSL_ARCH")
    if arch:
        return arch
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        return f"sm_{major}{minor}{'a' if major >= 9 else ''}"
    return "sm_100a"


def _dsl_arch(fn: Any) -> str:
    """The GPU architecture the DSL compiles ``fn`` for without a ``--gpu-arch``
    option (its environment manager's arch: ``CUTE_DSL_ARCH`` read at DSL start,
    else the detected device)."""
    target = _jit_function(fn)
    dsl = getattr(target, "_dsl_object", None)
    if dsl is None and hasattr(target, "_dsl_cls"):
        dsl = target._dsl_cls._get_dsl()  # the singleton the compile materializes
    envar = getattr(dsl, "envar", None)
    arch = getattr(envar, "arch", None) if envar is not None else None
    return str(arch) if arch else _fallback_arch()


def _serialized(resolved: Any) -> tuple[tuple[str, ...], str]:
    """(every option of a ``CompileOptions`` by class and value, its ``--gpu-arch``).
    By class, not ``serialize()``: options outside the MLIR pipeline string (e.g.
    ``EnableTVMFFI``) serialize to nothing but change the compiled launcher."""
    if resolved is None:
        return (), ""
    flags = sorted(
        f"{type(o).__name__}={getattr(o, 'value', o)!r}"
        for o in resolved.options.values()
    )
    return tuple(flags), resolved.gpu_arch


@functools.lru_cache(maxsize=256)
def _parsed_options(raw: str) -> tuple[tuple[str, ...], str]:
    """An ``options=`` string resolved by the DSL's own parser (once per string)."""
    from cutlass.base_dsl.compiler import _parse_compile_options_from_str

    return _serialized(_parse_compile_options_from_str(raw))


def _compile_target(compile_fn: Any, fn: Any, kwargs: dict) -> str:
    """The GPU architecture and the full compile option set of this compile, as the
    DSL resolves them (``options=`` string, else the ``cute.compile[...]`` options):
    every option serialized, so a default the DSL changes or an architecture taken
    from the environment is explicit in the key."""
    raw = kwargs.get("options")
    try:
        if isinstance(raw, str) and raw.strip():
            flags, arch = _parsed_options(raw)
        else:
            flags, arch = _serialized(getattr(compile_fn, "_compile_options", None))
        arch = arch or _dsl_arch(fn)
    except Exception as error:  # an option string the DSL itself would reject
        raise _Uncacheable(
            f"compile options not resolvable ({type(error).__name__}: {error})"
        ) from error
    return f"arch {arch}\noptions {' '.join(flags)}"


_RUNTIME = "__k3_cute_runtime_argument__"


def _runtime_marker(annotation: Any, value: Any) -> tuple[str, str, str]:
    """Key stand-in for a DSL numeric value (``cutlass.Int32(n)``, ``Int64``,
    ``Float32``, ...) passed to a ``@cute.jit`` parameter annotated with a DSL numeric
    type: the DSL makes it a runtime argument of the compiled function, so the trace
    never reads the value (measured: equal DSL module hashes across values). A plain
    Python number passed there is NOT: the DSL folds it into the trace as a constant
    (a different module per value), so it stays in the key by value. Only the
    annotation and the value's type are trace inputs."""
    return (_RUNTIME, _qual(annotation), _qual(type(value)))


def _numeric_annotation(annotation: Any) -> bool:
    try:
        from cutlass.cutlass_dsl import Numeric
    except ImportError:
        return False
    return isinstance(annotation, type) and issubclass(annotation, Numeric)


def _runtime_values(fn: Any, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
    """``args`` / ``kwargs`` with the values bound to DSL-numeric parameters of the
    compiled ``@cute.jit`` function replaced by ``_runtime_marker`` (anything the binding
    cannot place keeps its value: a conservative key, never a wrong hit)."""
    import inspect
    import typing

    target = _jit_function(fn)
    try:
        sig = inspect.signature(target)
        hints = typing.get_type_hints(target)
    except Exception:
        return args, kwargs
    params = list(sig.parameters.values())
    if params and params[0].name == "self" and not isinstance(fn, types.FunctionType):
        params = params[1:]  # a bound method or a callable kernel object
    try:
        bound = inspect.Signature(params).bind_partial(*args, **kwargs)
    except TypeError:
        return args, kwargs
    runtime = {
        name: _runtime_marker(hints[name], value)
        for name, value in bound.arguments.items()
        if _numeric_annotation(hints.get(name))
        and isinstance(value, _dsl_numeric_types())
    }
    if not runtime:
        return args, kwargs
    names = [p.name for p in params]
    new_args = tuple(
        runtime.get(names[i], a) if i < len(names) else a for i, a in enumerate(args)
    )
    new_kwargs = {k: runtime.get(k, v) for k, v in kwargs.items()}
    return new_args, new_kwargs


def _dsl_numeric_types() -> tuple[type, ...]:
    try:
        from cutlass.cutlass_dsl import Numeric
    except ImportError:
        return ()
    return (Numeric,)


def _key_and_modules(compile_fn: Any, fn: Any, args: tuple, kwargs: dict):
    fp = _Fingerprint()
    fp.add(getattr(compile_fn, "_compile_options", None))
    fp.add(fn)
    args, kwargs = _runtime_values(fn, args, kwargs)
    fp.add(args)
    fp.add(kwargs)
    target = _compile_target(compile_fn, fn, kwargs)
    head = f"{_versions()}\n{_environment()}\n{target}\n"
    return hashlib.sha256((head + fp.digest()).encode()).hexdigest(), fp.modules


def key_of(compile_fn: Any, fn: Any, args: tuple, kwargs: dict) -> str:
    """The pre-trace key of ``compile_fn(fn, *args, **kwargs)`` (raises
    ``_Uncacheable`` when an input cannot be described). The kernel's vllm source
    files are not in it: each entry records the files its trace depends on
    (``_sources``) and is a hit only while they are unchanged."""
    return _key_and_modules(compile_fn, fn, args, kwargs)[0]


# ------------------------------------------------------------------------- store


# Installed code hashed into the cache directory name (``install_identity``): the
# foreign packages the kernels run while tracing, and the CuTe DSL distribution
# (its compiler libraries live next to, not inside, the ``cutlass`` package).
_IDENTITY_PACKAGES = (*_FOREIGN_PACKAGES, "nvidia_cutlass_dsl")
# Compiled files: identified by size and modification time (Python files by content).
_BINARY_SUFFIXES = (".so", ".pyd", ".cubin", ".fatbin", ".ptx", ".bc")


def _package_roots(name: str) -> list[str]:
    import importlib.util

    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return []
    if spec is None:
        return []
    if spec.submodule_search_locations:
        return sorted(os.path.realpath(p) for p in spec.submodule_search_locations)
    return [os.path.realpath(spec.origin)] if spec.origin else []


def _hash_tree(h: Any, root: str) -> int:
    """Add every Python file under ``root`` (by content) and every compiled file
    (by size and modification time) to ``h``; return the number of files."""
    if os.path.isfile(root):
        walk = [(os.path.dirname(root), [], [os.path.basename(root)])]
    else:
        walk = os.walk(root)
    count = 0
    for folder, dirs, files in walk:
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for name in sorted(files):
            path = os.path.join(folder, name)
            rel = os.path.relpath(path, root)
            if name.endswith(".py"):
                try:
                    with open(path, "rb") as f:
                        digest = hashlib.sha256(f.read()).hexdigest()
                except OSError:
                    digest = "?"
                h.update(f"py\0{rel}\0{digest}\n".encode())
            elif name.endswith(_BINARY_SUFFIXES) or ".so." in name:
                try:
                    st = os.stat(path)
                    stamp = f"{st.st_size}:{st.st_mtime_ns}"
                except OSError:
                    stamp = "?"
                h.update(f"bin\0{rel}\0{stamp}\n".encode())
            else:
                continue
            count += 1
    return count


@functools.cache
def install_identity() -> str:
    """Digest of the installed code a compile depends on outside the vllm tree: the
    Python version, every installed distribution (name and version) and the files
    of the foreign packages the kernels trace with (``_IDENTITY_PACKAGES``: Python
    files by content, compiled files by size and modification time). It names the
    cache directory, so another image, an overlay on the package path or a patched
    file of one of those packages never reads an entry of the old one. (The vllm
    files a trace runs are matched per entry, ``_sources``.)"""
    import importlib.metadata

    start = time.perf_counter()
    h = hashlib.sha256(f"f{FORMAT}\n{sys.version}\n".encode())
    dists = set()
    for dist in importlib.metadata.distributions():
        try:
            dists.add(f"{dist.metadata['Name']}=={dist.version}")
        except Exception:  # a broken metadata directory
            dists.add(f"?{getattr(dist, '_path', '?')}")
    h.update("\n".join(sorted(dists)).encode())
    files = 0
    for name in _IDENTITY_PACKAGES:
        roots = _package_roots(name)
        h.update(f"\npackage {name}: {roots}\n".encode())
        for root in roots:
            files += _hash_tree(h, root)
    identity = h.hexdigest()[:20]
    logger.info(
        "Kimi K3 CuTe cache: install identity %s (%d distributions, %d package files, "
        "%.1f s)", identity, len(dists), files, time.perf_counter() - start,
    )  # fmt: skip
    return identity


# Location only (no behavior): a directory shared by every rank and node of a
# deployment, e.g. on Lustre. Unset: ``<VLLM_CACHE_ROOT>/kimi_k3/cutedsl_pretrace``.
LOCATION_ENV = "VLLM_KIMI_K3_CUTE_CACHE_DIR"


def cache_dir() -> str:
    import cutlass

    base = os.environ.get(LOCATION_ENV)
    if not base:
        from vllm import envs

        base = os.path.join(envs.VLLM_CACHE_ROOT, "kimi_k3", "cutedsl_pretrace")
    version = f"cutlass-dsl-{cutlass.__version__}-f{FORMAT}"
    return os.path.join(base, version, install_identity())


def _entry_dir(key: str) -> str:
    """A key's entries: one per source version, ``<variant>.json`` + ``<variant>.o``."""
    return os.path.join(cache_dir(), key[:2], key)


def _paths(key: str, variant: str) -> tuple[str, str]:
    base = os.path.join(_entry_dir(key), variant)
    return base + ".o", base + ".json"


def _variant(sources: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()[:24]


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


_NOT_CACHEABLE = object()


def _json_value(value: Any) -> bool:
    """A default that survives a JSON round trip unchanged (type and value)."""
    if type(value) is float:
        return math.isfinite(value)
    return type(value) in (type(None), bool, int, str)


def _launcher(compiled: Any, tvm_ffi: bool) -> Any:
    """How the loaded object is made to call exactly like ``compiled``: None (the
    loaded object as is), a kwargs-wrapper spec (dict), or _NOT_CACHEABLE.

    A TVM-FFI object loads as the plain positional ``tvm_ffi.Function``. That is
    ``compiled``'s call when every argument is positional without a default. With
    keyword-only arguments or defaults, ``compiled`` calls through
    ``tvm_ffi.utils.kwargs_wrapper.make_kwargs_wrapper`` with the DSL's
    ``KwargsWrapperSpec`` (recorded at its construction, ``install``): the loaded
    object gets the same wrapper from the stored spec. Raw pointer-address
    arguments and dataclass arguments (unpacked by the wrapper) are not cached."""
    if not tvm_ffi:
        return None
    arguments = compiled.execution_args
    if arguments.has_pointer_address_arg_specs:
        return _NOT_CACHEABLE
    params = list(arguments.signature.parameters.values())
    if any(dataclasses.is_dataclass(p.annotation) for p in params):
        return _NOT_CACHEABLE
    positional = (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
    if all(p.kind in positional and p.default is Parameter.empty for p in params):
        return None
    recorded = getattr(compiled, "_k3_kwargs_wrapper", None)
    if recorded is None:
        return _NOT_CACHEABLE
    spec, dataclass_args = recorded
    defaults = (*spec.arg_defaults, *spec.kwonly_defaults.values())
    if dataclass_args or not all(map(_json_value, defaults)):
        return _NOT_CACHEABLE
    return {
        "arg_names": list(spec.arg_names),
        "arg_defaults": list(spec.arg_defaults),
        "kwonly_names": list(spec.kwonly_names),
        "kwonly_defaults": dict(spec.kwonly_defaults),
    }


def _load(key: str, variant: str, tvm_ffi: bool, kwargs: dict | None = None) -> Any:
    from cutlass.cute.runtime import load_module

    module = load_module(_paths(key, variant)[0], enable_tvm_ffi=tvm_ffi)
    loaded = module[_prefix(key)]
    if kwargs is None:
        return loaded
    from tvm_ffi.utils.kwargs_wrapper import make_kwargs_wrapper

    return make_kwargs_wrapper(
        loaded,
        arg_names=kwargs["arg_names"],
        arg_defaults=tuple(kwargs["arg_defaults"]),
        kwonly_names=kwargs["kwonly_names"],
        kwonly_defaults=kwargs["kwonly_defaults"],
    )


def _find(key: str) -> tuple[str, dict] | None:
    """The entry of ``key`` whose recorded source files equal the current ones."""
    folder = _entry_dir(key)
    try:
        names = os.listdir(folder)
    except FileNotFoundError:
        return None
    for name in sorted(names):
        if not name.endswith(".json"):
            continue
        variant = name[:-5]
        obj, meta = _paths(key, variant)
        if not os.path.exists(obj):  # the record is written first
            continue
        try:
            with open(meta) as f:
                record = json.load(f)
        except (OSError, ValueError):
            continue
        sources = record.get("sources", {})
        if all(_file_digest(rel) == d for rel, d in sources.items()):
            return variant, record
    return None


# ------------------------------------------------------------- cross-process claims
#
# Every rank of a deployment (and every node sharing the cache directory) compiles
# the same keys in the same order. A miss first takes the key's claim file
# (O_CREAT | O_EXCL, atomic on local file systems and Lustre); a process that finds
# the claim taken does not wait but returns a ``_Deferred`` for the key and goes on
# to its next compile, so the ranks spread the misses over themselves instead of all
# compiling all of them. A deferred kernel resolves (loads the peer's entry) on its
# first use. Claims are refreshed by a heartbeat; a claim not refreshed for
# CLAIM_STALE_S (its process died) is taken over.

CLAIM_STALE_S = 300.0
_HEARTBEAT_S = 20.0
_held: set[str] = set()
_heartbeat: threading.Thread | None = None
_pending: list["_Deferred"] = []


def _claim_path(key: str) -> str:
    return _entry_dir(key) + ".claim"


def _beat() -> None:
    while True:
        time.sleep(_HEARTBEAT_S)
        with _lock:
            held = list(_held)
        for path in held:
            try:
                os.utime(path)
            except OSError:
                pass


def _claimer_dead(path: str) -> bool:
    """A claim by a process of this host that no longer exists."""
    try:
        with open(path) as f:
            host, pid = f.read().split()
        if host != socket.gethostname() or int(pid) == os.getpid():
            return False
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return True
    except (OSError, ValueError):
        return False
    return False


def _claim(key: str) -> bool:
    global _heartbeat
    path = _claim_path(key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for _ in range(3):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            try:
                age = time.time() - os.stat(path).st_mtime
            except FileNotFoundError:
                continue  # released meanwhile: try again
            if age < CLAIM_STALE_S and not _claimer_dead(path):
                return False
            stale = f"{path}.stale-{socket.gethostname()}-{os.getpid()}"
            try:  # one taker wins the rename
                os.rename(path, stale)
            except FileNotFoundError:
                return False
            logger.warning(
                "Kimi K3 CuTe cache: claim %s not refreshed for %.0f s or its process "
                "is gone; taking it over", os.path.basename(path), age,
            )  # fmt: skip
            with contextlib.suppress(OSError):
                os.unlink(stale)
            continue
        with os.fdopen(fd, "w") as f:
            f.write(f"{socket.gethostname()} {os.getpid()}\n")
        with _lock:
            _held.add(path)
            if _heartbeat is None:
                _heartbeat = threading.Thread(
                    target=_beat, name="k3-cute-cache-claims", daemon=True
                )
                _heartbeat.start()
        return True
    return False


def _release(key: str) -> None:
    path = _claim_path(key)
    with _lock:
        _held.discard(path)
    with contextlib.suppress(OSError):
        os.unlink(path)


class _Deferred:
    """A kernel another process is compiling: resolves to the loaded entry (or
    compiles it, when the claim frees without an entry this process can use) on the
    first call or attribute access, or in ``settle()``."""

    __slots__ = ("_job", "_obj")

    def __init__(self, job: tuple):
        self._job, self._obj = job, None

    def _resolve(self) -> Any:
        if self._obj is None:
            self._obj = _await(*self._job)
            self._job = None
        return self._obj

    def __call__(self, *args, **kwargs):
        return self._resolve()(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._resolve(), name)


_phase_depth = 0


@contextlib.contextmanager
def shared_phase(name: str):
    """A build phase whose misses claimed by peers are deferred (the rank goes on to
    its next compile) and all resolved when the phase ends. Outside a phase, a miss
    another process has claimed waits for that entry: no deferred kernel outlives
    its phase, so none is first resolved while serving."""
    global _phase_depth
    _phase_depth += 1
    try:
        yield
    finally:
        _phase_depth -= 1
    if _phase_depth == 0:
        settle()
        log_summary(name, table=True)


def settle() -> None:
    """Resolve every deferred kernel of this process (end of a build phase)."""
    while True:
        with _lock:
            pending = list(_pending)
            _pending.clear()
        if not pending:
            return
        for deferred in pending:
            deferred._resolve()


# ----------------------------------------------------------------------- compile


def _kernel_name(fn: Any) -> str:
    target = getattr(fn, "__func__", fn)
    if not isinstance(target, (types.FunctionType, type)):
        target = type(target)
    return getattr(target, "__qualname__", "?")


def _record(name: str, slot: int | None, seconds: float = 0.0, **times) -> None:
    """Add ``times`` to the totals and, with a ``slot`` (0 hit, 1 miss, 2 uncached,
    3 deferred to another process), one event of kernel ``name``."""
    global _events
    with _lock:
        for k, v in times.items():
            setattr(stats, k, getattr(stats, k) + v)
        if slot is None:
            return
        entry = stats.kernels.setdefault(name, [0, 0, 0, 0.0, 0])
        entry[slot if slot < 3 else 4] += 1
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


def _uncached_reason(name: str, why: str) -> None:
    """Count a compile that bypasses the cache; the first one of each kernel and
    reason is logged by name (the kernel is compiled normally, in memory)."""
    with _lock:
        reason = f"{name}: {why}"
        first = reason not in stats.reasons
        stats.reasons[reason] = stats.reasons.get(reason, 0) + 1
    if first:
        logger.info(
            "Kimi K3 CuTe cache: %s compiled without the cache (%s)", name, why
        )


def _compile_and_store(inner, fn, args, kwargs, key, modules, name):
    found: dict = {}
    start = time.perf_counter()
    if key is None:
        compiled = inner(fn, *args, trace_finalize_hooks=_trace_hook(found), **kwargs)
        seconds = time.perf_counter() - start
        _record(name, 2, seconds, uncached=1, compile_s=seconds)
        return compiled
    with _Executed() as executed:
        compiled = inner(fn, *args, trace_finalize_hooks=_trace_hook(found), **kwargs)
    tvm_ffi = found.get("tvm_ffi", False)
    launcher = _launcher(compiled, tvm_ffi) if "ir" in found else _NOT_CACHEABLE
    if launcher is _NOT_CACHEABLE:
        why = "no module hash" if "ir" not in found else "loaded launcher differs"
        _uncached_reason(name, why)
        seconds = time.perf_counter() - start
        _record(name, 2, seconds, uncached=1, compile_s=seconds)
        return compiled
    sources = _sources(modules, executed.files)
    variant = _variant(sources)
    obj, meta = _paths(key, variant)
    record = {
        "kernel": name,
        "tvm_ffi": tvm_ffi,
        "ir": found["ir"],
        "compile_s": round(time.perf_counter() - start, 3),
        "host": socket.gethostname(),
        "sources": sources,
        "kwargs": launcher,
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
        loaded = _load(key, variant, tvm_ffi, launcher)
    except Exception as error:
        logger.warning(
            "Kimi K3 CuTe cache: %s compiled but its object cannot be stored or "
            "loaded (%s: %s); this compile stays in memory",
            name, type(error).__name__, str(error)[:300],
        )  # fmt: skip
        _uncached_reason(name, f"export {type(error).__name__}")
        seconds = time.perf_counter() - start
        _record(name, 2, seconds, uncached=1, compile_s=seconds)
        return compiled
    seconds = time.perf_counter() - start
    _record(name, 1, seconds, compiled=1, compile_s=seconds)
    return loaded


class VerifyMismatch(BaseException):
    """VERIFY: a cache hit traced to another DSL module than the stored one (the
    pre-trace key misses a trace input), or its re-trace failed. A BaseException,
    so no kernel's compile-failure fallback (``except Exception``) can turn it into
    a different path: the start fails."""


def _verify(inner, fn, args, kwargs, key, variant, name, stored) -> None:
    """Trace a hit again and require its DSL module hash to equal the stored one."""
    try:
        traced = _trace_only(inner, fn, args, kwargs).get("ir")
    except Exception as error:
        _record(name, None, mismatched=1)
        raise VerifyMismatch(
            f"Kimi K3 CuTe cache VERIFY: re-trace of the hit {key[:16]}/{variant} "
            f"({name}) failed: {type(error).__name__}: {error}"
        ) from error
    if traced != stored:
        _record(name, None, mismatched=1)
        message = (
            f"Kimi K3 CuTe cache VERIFY MISMATCH: key {key[:16]}/{variant} of {name} "
            f"hit an entry traced to a different module (stored {stored[:16]}, now "
            f"{str(traced)[:16]}): the pre-trace key misses an input"
        )
        logger.error(message)
        raise VerifyMismatch(message)
    _record(name, None, verified=1)


def _load_entry(inner, fn, args, kwargs, key, modules, name, found):
    variant, record = found
    loaded_at = time.perf_counter()
    try:
        loaded = _load(key, variant, record["tvm_ffi"], record.get("kwargs"))
    except Exception as error:
        logger.warning(
            "Kimi K3 CuTe cache: entry %s/%s (%s) does not load (%s: %s); recompiling "
            "and replacing it",
            key[:16], variant, name, type(error).__name__, str(error)[:300],
        )  # fmt: skip
        return _compile_and_store(inner, fn, args, kwargs, key, modules, name)
    if VERIFY:
        _verify(inner, fn, args, kwargs, key, variant, name, record["ir"])
    _record(name, 0, 0.0, loaded=1, load_s=time.perf_counter() - loaded_at)
    return loaded


def _compile_claimed(inner, fn, args, kwargs, key, modules, name):
    """Under this process's claim of ``key``: load an entry a peer finished in the
    meantime, else compile and store."""
    try:
        found = _find(key)
        if found is not None:
            return _load_entry(inner, fn, args, kwargs, key, modules, name, found)
        return _compile_and_store(inner, fn, args, kwargs, key, modules, name)
    finally:
        _release(key)


def _await(inner, fn, args, kwargs, key, modules, name):
    """Resolve a deferred kernel: wait for the claim holder's entry (or take the
    claim when it frees without an entry for these sources, or goes stale)."""
    start, delay, warned = time.perf_counter(), 0.02, False
    while True:
        found = _find(key)
        if found is not None:
            loaded = _load_entry(inner, fn, args, kwargs, key, modules, name, found)
            _record(name, None, wait_s=time.perf_counter() - start)
            return loaded
        if _claim(key):
            compiled = _compile_claimed(inner, fn, args, kwargs, key, modules, name)
            _record(name, None, wait_s=time.perf_counter() - start)
            return compiled
        waited = time.perf_counter() - start
        if waited > 120 and not warned:
            warned = True
            logger.info(
                "Kimi K3 CuTe cache: %s waiting %.0f s for a peer's compile of %s",
                name, waited, key[:16],
            )  # fmt: skip
        time.sleep(delay)
        delay = min(delay * 1.5, 0.5)


def _compile(inner, fn, args: tuple, kwargs: dict):
    name = _kernel_name(fn)
    start = time.perf_counter()
    for value in (*args, *kwargs.values()):  # a deferred kernel as an argument
        if isinstance(value, _Deferred):
            value._resolve()
    args = tuple(a._obj if isinstance(a, _Deferred) else a for a in args)
    kwargs = {k: v._obj if isinstance(v, _Deferred) else v for k, v in kwargs.items()}
    try:
        key, modules = _key_and_modules(inner, fn, args, kwargs)
        if not hasattr(sys, "monitoring"):  # the sources need sys.monitoring
            raise _Uncacheable("Python without sys.monitoring")
    except _Uncacheable as why:
        _uncached_reason(name, str(why))
        key, modules = None, set()
    _record(name, None, key_s=time.perf_counter() - start)
    if key is None:
        return _compile_and_store(inner, fn, args, kwargs, None, modules, name)
    found = _find(key)
    if found is not None:
        return _load_entry(inner, fn, args, kwargs, key, modules, name, found)
    if _claim(key):
        return _compile_claimed(inner, fn, args, kwargs, key, modules, name)
    if _phase_depth == 0:  # outside a build phase: wait for the peer's entry
        return _await(inner, fn, args, kwargs, key, modules, name)
    deferred = _Deferred((inner, fn, args, kwargs, key, modules, name))
    with _lock:
        _pending.append(deferred)
    _record(name, 3, deferred=1)
    return deferred


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
    _record_kwargs_wrappers()


def _record_kwargs_wrappers() -> None:
    """Keep the DSL's kwargs-wrapper spec on each TVM-FFI launcher built with one,
    so its stored entry loads with the same wrapper (``_launcher``)."""
    try:
        from cutlass.cutlass_dsl.tvm_ffi_provider import (
            TVMFFIJitCompiledFunctionWithKwargs as launcher,
        )
    except ImportError:
        return
    init = launcher.__init__
    if getattr(init, "_k3_recorded", False):
        return

    @functools.wraps(init)
    def recording_init(self, *args, **kwargs):
        spec = kwargs.get("kwargs_wrapper_spec")
        dataclass_args = list(kwargs.get("map_dataclass_to_tuple") or ())
        init(self, *args, **kwargs)
        if spec is not None:
            self._k3_kwargs_wrapper = (spec, dataclass_args)

    recording_init._k3_recorded = True
    launcher.__init__ = recording_init


def log_summary(stage: str, table: bool = False) -> None:
    """One line of totals; with ``table``, one line per kernel (calls, compiles,
    seconds) sorted by compile seconds."""
    with _lock:
        s = Stats(**{k: v for k, v in vars(stats).items()
                     if k not in ("kernels", "reasons")})  # fmt: skip
        kernels = sorted(stats.kernels.items(), key=lambda kv: -kv[1][3])
        reasons = sorted(stats.reasons.items(), key=lambda kv: -kv[1])[:5]
    logger.info(
        "Kimi K3 CuTe cache (%s, %.0f s since import): %d loaded, %d compiled, %d "
        "uncached, %d deferred to peers (waited %.0f s); keys %.0f s, loads %.0f s, "
        "compiles %.0f s (%s)%s; slowest: %s%s",
        stage, time.perf_counter() - _started, s.loaded, s.compiled, s.uncached,
        s.deferred, s.wait_s, s.key_s, s.load_s, s.compile_s, cache_dir(),
        f"; VERIFY {s.verified} hits re-traced equal, {s.mismatched} mismatched"
        if VERIFY else "",
        ", ".join(f"{n} {e[1] + e[2]}x {e[3]:.0f} s" for n, e in kernels[:8]),
        "; uncached: " + ", ".join(f"{r} {c}x" for r, c in reasons) if reasons else "",
    )  # fmt: skip
    if table:
        for n, e in kernels:
            logger.info(
                "Kimi K3 CuTe cache kernel %s: calls %d, loaded %d, compiled %d, "
                "uncached %d, deferred %d, compile %.1f s",
                n, sum(e[:3]) + e[4], e[0], e[1], e[2], e[4], e[3],
            )  # fmt: skip
