# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 pre-trace CuTe DSL compile cache (``vllm/models/kimi_k3/nvidia/cute_cache.py``).

- Fingerprint/key unit cases (no GPU): equal inputs give equal keys; a change of a
  kernel attribute, an argument layout, a compile option, the GPU architecture
  environment or the source digest gives a new key; objects the fingerprint cannot
  describe are reported, not guessed.
- Real kernels (Blackwell+): each start runs in a fresh process sharing one
  ``VLLM_CACHE_ROOT``. The cold start compiles and stores, the warm start loads
  every kernel without tracing (and, in VERIFY mode, re-traces each hit to check
  the stored DSL module hash), the outputs are bitwise those of the plain
  ``cute.compile``; a corrupt entry is logged, recompiled and replaced. Kernels:
  a test elementwise kernel (CuTe launcher) and the specialized decode's LM-head
  logits GEMM and embedding gather (TVM-FFI launchers), compiled through the
  installed ``cute.compile`` exactly as the decode backend does.
"""

import json
import os
import subprocess
import sys
import threading
import time

import pytest
import torch

from vllm.platforms import current_platform

ROWS, VOCAB, HIDDEN = 5, 1024, 7168


# ----------------------------------------------------------------- fingerprints


class _Kernel:
    def __init__(self, scale, tile=(4, 8)):
        self.scale, self.tile = scale, tile


def _key(fn, *args, compile_fn=None, **kwargs):
    from vllm.models.kimi_k3.nvidia import cute_cache

    return cute_cache.key_of(compile_fn, fn, args, kwargs)


class _DecodeKernel:
    """Stand-in for a specialized decode kernel object: the codegen-relevant launch
    configuration lives in its attributes (as on the real kernel classes)."""

    def __init__(self, world=16, ep=16, dcp=8, tp=4, rows=128, knobs=None):
        self.world, self.ep, self.dcp, self.tp, self.rows = world, ep, dcp, tp, rows
        self.knobs = knobs if knobs is not None else {
            "mma_tiler_mnk": (256, 128, 256), "cluster_shape_mn": (2, 1),
            "epi_flag_batches": (1, 4), "schedule_policy": ("decode", None),
        }


def test_key_separates_every_codegen_relevant_setting(monkeypatch, tmp_path):
    """Safety: a key that is too coarse would load a kernel compiled for another
    shape (wrong output, no error). Starts that differ in any codegen-relevant
    setting get different keys; starts that differ only in the launcher's
    per-start identity get the same key."""
    for name in ("K31_CNAME", "K31_NODES", "K31_MIN_LEFT_S"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CUTE_DSL_ARCH", "sm_107a")
    x = torch.empty(128, 3584, dtype=torch.bfloat16)
    knob_file = tmp_path / "knobs.json"
    knob_file.write_text(json.dumps({"epi_flag_batches": [1, 4]}))

    def key(kernel=None, arg=x, **kw):
        return _key(kernel or _DecodeKernel(**kw), arg, 3, options="--opt-level 3")

    base = key()
    variants = {
        "world size": key(world=32),
        "EP size": key(ep=32),
        "DCP layout": key(dcp=16),
        "TP layout": key(tp=8),
        "row bucket": key(rows=176),
        "knob record": key(knobs={**_DecodeKernel().knobs, "epi_flag_batches": (2, 4)}),
        "knob file": key(knobs=json.loads(knob_file.read_text())),
        "argument rows": key(arg=torch.empty(176, 3584, dtype=torch.bfloat16)),
        "argument dtype": key(arg=torch.empty(128, 3584, dtype=torch.float16)),
        "argument width": key(arg=torch.empty(128, 7168, dtype=torch.bfloat16)),
    }
    monkeypatch.setenv("CUTE_DSL_ARCH", "sm_100a")
    variants["arch"] = key()
    monkeypatch.setenv("CUTE_DSL_ARCH", "sm_107a")
    assert key() == base
    for what, k in variants.items():
        assert k != base, f"{what} does not change the key"
    assert len(set(variants.values())) == len(variants), "two settings share a key"
    # the launcher's per-start identity is not a trace input
    monkeypatch.setenv("K31_CNAME", "k31e2e-105921-1121222010")
    monkeypatch.setenv("K31_NODES", "rhea0073,rhea0074")
    monkeypatch.setenv("K31_MIN_LEFT_S", "2700")
    first = key()
    monkeypatch.setenv("K31_CNAME", "k31e2e-107336-1122001106")
    monkeypatch.setenv("K31_NODES", "rhea0109,rhea0113")
    monkeypatch.setenv("K31_MIN_LEFT_S", "1200")
    assert key() == first == base


def test_key_is_deterministic_and_input_sensitive(monkeypatch):
    from vllm.models.kimi_k3.nvidia import cute_cache

    a = _key(_Kernel(2.0), torch.empty(0), 3, options="--opt-level 3")
    assert a == _key(_Kernel(2.0), torch.empty(0), 3, options="--opt-level 3")
    assert a != _key(_Kernel(2.5), torch.empty(0), 3, options="--opt-level 3")
    assert a != _key(_Kernel(2.0, (4, 16)), torch.empty(0), 3, options="--opt-level 3")
    assert a != _key(_Kernel(2.0), torch.empty(1), 3, options="--opt-level 3")
    assert a != _key(_Kernel(2.0), torch.empty(0), 4, options="--opt-level 3")
    assert a != _key(_Kernel(2.0), torch.empty(0), 3, options="--opt-level 2")
    assert a != _key(_Kernel(2.0), torch.empty(0, dtype=torch.int8), 3,
                     options="--opt-level 3")  # fmt: skip
    monkeypatch.setenv("CUTE_DSL_ARCH", "sm_107a")
    b = _key(_Kernel(2.0), torch.empty(0), 3, options="--opt-level 3")
    monkeypatch.setenv("CUTE_DSL_ARCH", "sm_100a")
    assert b != _key(_Kernel(2.0), torch.empty(0), 3, options="--opt-level 3")
    # per-run paths and hosts are not trace inputs
    monkeypatch.setenv("VLLM_ITERLOG_DIR", "/tmp/a")
    c = _key(_Kernel(2.0), torch.empty(0), 3, options="--opt-level 3")
    monkeypatch.setenv("VLLM_ITERLOG_DIR", "/tmp/b")
    assert c == _key(_Kernel(2.0), torch.empty(0), 3, options="--opt-level 3")
    # the launcher's per-start identity (container name with its timestamp, node list,
    # time budget) is not a trace input either
    for name, a1, a2 in (("K31_CNAME", "k31e2e-105921-1121222010", "k31e2e-105921-1121232827"),
                         ("K31_NODES", "rhea0073,rhea0074", "rhea0055,rhea0056"),
                         ("K31_MIN_LEFT_S", "2700", "900")):
        monkeypatch.setenv(name, a1)
        d = _key(_Kernel(2.0), torch.empty(0), 3, options="--opt-level 3")
        monkeypatch.setenv(name, a2)
        assert d == _key(_Kernel(2.0), torch.empty(0), 3, options="--opt-level 3"), name
    # sources are not in the key (each entry records its own files)
    assert not hasattr(cute_cache, "_source_digest")


def test_key_has_explicit_arch_and_resolved_options(monkeypatch):
    """The target architecture and the resolved option set are in the key: an
    architecture the DSL takes from the device (no --gpu-arch, no environment
    change) and each option value give another key."""
    from vllm.models.kimi_k3.nvidia import cute_cache

    k = _Kernel(1.0)
    target = cute_cache._compile_target(None, k, {"options": "--gpu-arch=sm_107a"})
    assert target.startswith("arch sm_107a\n"), target
    assert "sm_107a" in target.split("\n")[1]  # also among the serialized options
    o2 = cute_cache._compile_target(None, k, {"options": "--opt-level 2"})
    o3 = cute_cache._compile_target(None, k, {"options": "--opt-level 3"})
    assert o2 != o3
    ffi = cute_cache._compile_target(None, k, {"options": "--enable-tvm-ffi"})
    assert "EnableTVMFFI=True" in ffi, ffi  # not part of the pipeline string
    assert "EnableTVMFFI=False" in o2, o2  # every option, defaults included
    assert ffi != cute_cache._compile_target(None, k, {})
    # no --gpu-arch: the DSL's architecture, explicitly
    monkeypatch.setattr(cute_cache, "_dsl_arch", lambda fn: "sm_100a")
    a = _key(k, torch.empty(0), options="--opt-level 2")
    assert cute_cache._compile_target(None, k, {}).startswith("arch sm_100a\n")
    monkeypatch.setattr(cute_cache, "_dsl_arch", lambda fn: "sm_107a")
    assert _key(k, torch.empty(0), options="--opt-level 2") != a
    # an option string the DSL rejects: uncached, not guessed
    with pytest.raises(cute_cache._Uncacheable, match="compile options"):
        _key(k, options="--no-such-option 1")


def test_dsl_arch_is_the_dsl_environment_arch():
    import cutlass.cute as cute

    from vllm.models.kimi_k3.nvidia import cute_cache

    @cute.jit
    def jitted(x: cute.Tensor):
        pass

    arch = cute_cache._dsl_arch(jitted)
    assert arch.startswith("sm_"), arch
    if os.environ.get("CUTE_DSL_ARCH"):
        assert arch == os.environ["CUTE_DSL_ARCH"]


def test_sources_cover_kernel_executed_files_and_their_imports():
    from vllm.models.kimi_k3.nvidia import cute_cache
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.oproj import producer

    module = producer.__name__
    executed = {producer.__file__, "/usr/lib/python3/foreign.py", "<string>"}
    sources = cute_cache._sources({module, "torch"}, executed)
    rel = os.path.relpath(producer.__file__, cute_cache._PKG_ROOT)
    assert rel in sources  # the kernel's own module
    assert all(not os.path.isabs(f) for f in sources)  # foreign files: versions
    # its direct imports (helpers whose cached values a trace may read)
    assert set(cute_cache._direct_imports(rel)) <= set(sources)
    assert any("common" in f for f in cute_cache._direct_imports(rel))
    # an unrelated file is not a dependency
    assert "vllm/entrypoints/cli/main.py" not in sources
    assert "entrypoints/cli/main.py" not in sources


def test_find_matches_only_unchanged_sources(tmp_path, monkeypatch):
    from vllm.models.kimi_k3.nvidia import cute_cache

    monkeypatch.setattr(cute_cache, "cache_dir", lambda: str(tmp_path))
    digests = {"a.py": "1", "b.py": "2", "c.py": "9"}
    monkeypatch.setattr(cute_cache, "_file_digest", lambda rel: digests[rel])
    key = "ab" * 32
    for sources in ({"a.py": "1", "b.py": "2"}, {"a.py": "0"}):
        variant = cute_cache._variant(sources)
        obj, meta = cute_cache._paths(key, variant)
        os.makedirs(os.path.dirname(obj), exist_ok=True)
        with open(meta, "w") as f:
            json.dump({"sources": sources, "tvm_ffi": False, "ir": "x"}, f)
        open(obj, "wb").close()
    found = cute_cache._find(key)
    assert found is not None and found[1]["sources"] == {"a.py": "1", "b.py": "2"}
    digests["c.py"] = "edited"  # a file outside the entry's sources: still a hit
    assert cute_cache._find(key) is not None
    digests["b.py"] = "edited"  # one of its sources: a miss
    assert cute_cache._find(key) is None
    digests["a.py"] = "0"  # the other entry's sources
    assert cute_cache._find(key)[1]["sources"] == {"a.py": "0"}


def test_claims_exclusive_stale_and_dead_owner(tmp_path, monkeypatch):
    from vllm.models.kimi_k3.nvidia import cute_cache

    monkeypatch.setattr(cute_cache, "cache_dir", lambda: str(tmp_path))
    key = "cd" * 32
    assert cute_cache._claim(key)
    path = cute_cache._claim_path(key)
    assert os.path.exists(path)
    with open(path, "w") as f:  # held by a live process of this host
        f.write(f"{cute_cache.socket.gethostname()} {os.getppid()}\n")
    assert not cute_cache._claim(key)
    old = time.time() - cute_cache.CLAIM_STALE_S - 10  # not refreshed: taken over
    os.utime(path, (old, old))
    assert cute_cache._claim(key)
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    with open(path, "w") as f:  # fresh, but its process is gone: taken over
        f.write(f"{cute_cache.socket.gethostname()} {dead.pid}\n")
    assert cute_cache._claim(key)
    with open(path, "w") as f:  # another host's fresh claim: kept
        f.write(f"other-host {dead.pid}\n")
    assert not cute_cache._claim(key)
    cute_cache._release(key)
    assert not os.path.exists(path)


def test_key_covers_cpu_tensor_contents():
    """Host tensors (arguments or kernel state) enter the key by contents, not only
    layout and address: the same buffer with other values is another key."""
    host = torch.arange(16, dtype=torch.int32)
    k = _Kernel(1.0)
    k.table = host
    a, a_arg = _key(k), _key(_Kernel(1.0), host)
    assert a == _key(k) and a_arg == _key(_Kernel(1.0), host)
    host[3] = 100  # in place: same address, layout and dtype
    assert _key(k) != a
    assert _key(_Kernel(1.0), host) != a_arg
    # a strided view and a bfloat16 (no numpy dtype) tensor are hashed too
    view = torch.arange(32, dtype=torch.float32).view(4, 8)[:, ::2]
    b = _key(_Kernel(1.0), view)
    view[0, 0] = -1.0
    assert _key(_Kernel(1.0), view) != b
    half = torch.ones(8, dtype=torch.bfloat16)
    c = _key(_Kernel(1.0), half)
    half[0] = 2.0
    assert _key(_Kernel(1.0), half) != c
    assert _key(_Kernel(1.0), torch.empty(0)) == _key(_Kernel(1.0), torch.empty(0))


def test_cuda_stream_argument_is_keyed_by_type():
    """A CUstream launch argument is described by its type: two processes (other
    stream handles) share the cache entry instead of compiling without it."""
    driver = pytest.importorskip("cuda.bindings.driver")
    a, b = driver.CUstream(0), driver.CUstream(12345)
    assert _key(_Kernel(1.0), a) == _key(_Kernel(1.0), b)
    assert _key(_Kernel(1.0), a) != _key(_Kernel(2.0), a)


def _capsule(address):
    import ctypes

    new = ctypes.pythonapi.PyCapsule_New
    new.restype, new.argtypes = ctypes.py_object, [ctypes.c_void_p, ctypes.c_char_p]
    new.argtypes += [ctypes.c_void_p]
    return new(address, b"dltensor", None)


class _DslTensor:
    """Stand-in for cute.runtime._Tensor: a DLPack capsule plus the DSL cache key."""

    __module__, __qualname__ = "cutlass.cute.runtime", "_Tensor"

    def __init__(self, address, layout):
        self._dlpack_data, self._layout = _capsule(address), layout

    @property
    def __cache_key__(self):
        return ("Float8E4M3FN", 16, self._layout)


def test_dsl_runtime_argument_is_keyed_by_its_dsl_cache_key():
    """A from_dlpack argument is described by the DSL's own cache key: the capsule
    and address differ per process (the MegaMoE kernels compiled outside the cache
    at every start), the layout marks do not."""
    a, b = _DslTensor(0x1000, (2, 1)), _DslTensor(0x2000, (2, 1))
    assert _key(_Kernel(1.0), a) == _key(_Kernel(1.0), b)
    assert _key(_Kernel(1.0), a) != _key(_Kernel(1.0), _DslTensor(0x1000, (1, 2)))


def test_stale_ir_value_on_a_kernel_is_keyed_by_type():
    """An IR value a finished trace left on the kernel object does not make the
    next compile of that object uncacheable."""
    ir_value = type("OpResult", (), {"__slots__": (), "__module__": "cutlass._mlir.ir"})
    k, j = _Kernel(1.0), _Kernel(1.0)
    k.tma, j.tma = ir_value(), ir_value()
    assert _key(k) == _key(j)
    assert _key(k) != _key(_Kernel(1.0))


class _PeerMapper:
    """A launch argument with runtime addresses that declares its trace key."""

    def __init__(self, base, ranks):
        self.base_address, self.offsets, self.max_ranks = base, tuple(range(ranks)), ranks

    def __cute_cache_key__(self):
        return self.max_ranks, len(self.offsets)


def test_declared_cute_cache_key_replaces_the_attributes():
    """The MegaMoE peer mapper's workspace address differs per process: keyed by
    what it declares, two processes share the entry; the rank count still keys."""
    four = _key(_Kernel(1.0), _PeerMapper(0x7F00, 4))
    assert four == _key(_Kernel(1.0), _PeerMapper(0x9F00, 4))
    assert four != _key(_Kernel(1.0), _PeerMapper(0x7F00, 8))


def test_fingerprint_functions_closures_cycles():
    def make(n):
        def inner():
            return n

        return inner

    k = _Kernel(1.0)
    k.self_ref, k.fn = k, make(1)
    k2 = _Kernel(1.0)
    k2.self_ref, k2.fn = k2, make(1)
    assert _key(k) == _key(k2)  # equal graphs (with a cycle), different objects
    k2.fn = make(2)  # same code, other closure value
    assert _key(k) != _key(k2)
    k2.fn = lambda: 1  # other code
    assert _key(k) != _key(k2)
    assert _key(k, {1, 2}) == _key(k, {2, 1})
    assert _key(k, (1, 2)) != _key(k, (2, 1))


def test_key_covers_base_classes_and_foreign_packages():
    from vllm.models.kimi_k3.nvidia import cute_cache

    class A:
        pass

    class B:
        pass

    one = type("Kernel", (A,), {"__module__": __name__})
    two = type("Kernel", (B,), {"__module__": __name__})
    assert _key(one()) != _key(two())  # same name, other base class code
    versions = cute_cache._versions()
    for name in ("cutlass", "flashinfer", "tokenspeed_mla"):
        assert name in versions


def test_install_identity_namespaces_the_directory(tmp_path, monkeypatch):
    """A patched Python file or a replaced compiled file of a traced package (an
    overlay, a new image) gives a new cache directory."""
    from vllm.models.kimi_k3.nvidia import cute_cache

    pkg = tmp_path / "k3cc_fake_pkg"
    (pkg / "sub").mkdir(parents=True)
    (pkg / "__init__.py").write_text("X = 1\n")
    (pkg / "sub" / "mod.py").write_text("def f():\n    return 1\n")
    (pkg / "sub" / "_ext.so").write_bytes(b"\0" * 64)
    (pkg / "__pycache__").mkdir()
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(cute_cache, "_IDENTITY_PACKAGES", ("k3cc_fake_pkg",))
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path / "cache"))

    def identity():
        cute_cache.install_identity.cache_clear()
        return cute_cache.install_identity()

    try:
        base = identity()
        assert base == identity()  # stable
        assert cute_cache.cache_dir().endswith(os.sep + base)
        (pkg / "__pycache__" / "mod.cpython-312.pyc").write_bytes(b"x")
        assert identity() == base  # bytecode caches are not code
        (pkg / "sub" / "mod.py").write_text("def f():\n    return 2\n")
        patched = identity()
        assert patched != base  # a patched submodule
        (pkg / "sub" / "_ext.so").write_bytes(b"\0" * 65)
        replaced = identity()
        assert replaced not in (base, patched)  # a replaced extension
        (pkg / "sub" / "extra.py").write_text("")
        assert identity() not in (base, patched, replaced)  # an added file
    finally:
        cute_cache.install_identity.cache_clear()


def test_verify_mismatch_raises_through_fallbacks(tmp_path, monkeypatch):
    """VERIFY: a hit whose re-trace gives another module hash (or whose re-trace
    fails) raises VerifyMismatch, which an ``except Exception`` compile fallback
    does not catch; an equal hash is counted as verified."""
    from vllm.models.kimi_k3.nvidia import cute_cache

    loaded = object()
    monkeypatch.setattr(cute_cache, "VERIFY", True)
    monkeypatch.setattr(cute_cache, "_load", lambda *a: loaded)
    record = {"ir": "a" * 64, "tvm_ffi": False}
    entry = ("v1", record)
    traced = {"ir": "a" * 64}

    def trace_only(*a):
        if isinstance(traced, Exception):
            raise traced
        return traced

    monkeypatch.setattr(cute_cache, "_trace_only", trace_only)
    before = cute_cache.stats.verified
    hit = cute_cache._load_entry(None, None, (), {}, "k" * 64, set(), "K", entry)
    assert hit is loaded and cute_cache.stats.verified == before + 1

    traced = {"ir": "b" * 64}
    with pytest.raises(cute_cache.VerifyMismatch, match="MISMATCH"):
        try:  # a kernel's compile-failure fallback
            cute_cache._load_entry(None, None, (), {}, "k" * 64, set(), "K", entry)
        except Exception:
            pytest.fail("VerifyMismatch was caught by an `except Exception` fallback")
    traced = {}  # the re-trace produced no module hash
    with pytest.raises(cute_cache.VerifyMismatch):
        cute_cache._load_entry(None, None, (), {}, "k" * 64, set(), "K", entry)
    traced = ValueError("trace failed")
    with pytest.raises(cute_cache.VerifyMismatch, match="trace failed"):
        cute_cache._load_entry(None, None, (), {}, "k" * 64, set(), "K", entry)
    assert issubclass(cute_cache.VerifyMismatch, BaseException)
    assert not issubclass(cute_cache.VerifyMismatch, Exception)


def test_location_setting_moves_the_directory_not_the_key(tmp_path, monkeypatch):
    from vllm.models.kimi_k3.nvidia import cute_cache

    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path / "root"))
    monkeypatch.delenv(cute_cache.LOCATION_ENV, raising=False)
    default = cute_cache.cache_dir()
    assert default.startswith(str(tmp_path / "root" / "kimi_k3" / "cutedsl_pretrace"))
    key = _key(_Kernel(1.0), torch.empty(0))
    monkeypatch.setenv(cute_cache.LOCATION_ENV, str(tmp_path / "shared"))
    moved = cute_cache.cache_dir()
    assert moved.startswith(str(tmp_path / "shared") + os.sep)
    assert os.path.relpath(moved, tmp_path / "shared") == os.path.relpath(
        default, tmp_path / "root" / "kimi_k3" / "cutedsl_pretrace"
    )  # same <DSL version>/<install identity> below the base
    assert _key(_Kernel(1.0), torch.empty(0)) == key  # a location, not an input


def test_key_covers_classes_built_by_factories():
    """Kernel classes created inside a function (mla/dcp.py ``_without_reducer(
    polled_core(batch, ...))``) share their qualified name across factory
    arguments; their methods' closures and class attributes enter the key."""

    def factory(batch, balanced):
        class Core(_Kernel):
            BALANCED = balanced

            def tiles(self):
                return batch * 2

        def without_reducer(base):
            class SplitPartials(base):
                def reduction_kernel(self, *args):
                    return super().reduction_kernel(*args)

            return SplitPartials

        return without_reducer(Core)

    def preprocessed():  # the DSL swaps __code__ on first compile
        return 1

    k = _Kernel(1.0)
    k.fn = preprocessed
    before = _key(k)
    preprocessed._original_code = preprocessed.__code__
    preprocessed.__code__ = (lambda: 2).__code__
    assert _key(k) == before

    a = _key(factory(8, False)(1.0))
    assert a == _key(factory(8, False)(1.0))  # same arguments, new class objects
    assert a != _key(factory(16, False)(1.0))  # a closure value of a base method
    assert a != _key(factory(8, True)(1.0))  # a class attribute of a base
    # a module-level class stays described by name (its code: the source digests)
    assert _key(_Kernel(1.0)) == _key(_Kernel(1.0))


def test_uncacheable_inputs_are_reported():
    from vllm.models.kimi_k3.nvidia import cute_cache

    k = _Kernel(1.0)
    k.lock = threading.Lock()
    with pytest.raises(cute_cache._Uncacheable, match="lock"):
        _key(k)


def test_uncached_kernels_are_compiled_and_logged_by_name(monkeypatch):
    """A kernel whose inputs cannot be fingerprinted is compiled normally (the plain
    DSL compile, its result returned as is) and named in the log once per reason."""
    from vllm.models.kimi_k3.nvidia import cute_cache

    messages = []
    monkeypatch.setattr(
        cute_cache.logger, "info", lambda msg, *a: messages.append(msg % a)
    )
    compiled = object()
    calls = []

    def inner(fn, *args, trace_finalize_hooks=None, **kwargs):
        calls.append(fn)
        return compiled

    class LockedKernel:
        def __init__(self):
            self.lock = threading.Lock()

    before = cute_cache.stats.uncached
    for _ in range(2):
        assert cute_cache._compile(inner, LockedKernel(), (), {}) is compiled
    assert len(calls) == 2 and cute_cache.stats.uncached == before + 2
    named = [m for m in messages if "LockedKernel compiled without the cache" in m]
    assert len(named) == 1, messages
    assert "lock" in named[0]


def test_only_kimi_k3_kernels_are_cached():
    from vllm.models.kimi_k3.nvidia import cute_cache
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.verify import lmhead

    assert cute_cache._is_k3_kernel(lmhead.compile_lmhead_logits.__wrapped__)
    assert not cute_cache._is_k3_kernel(_Kernel(1.0))
    assert not cute_cache._is_k3_kernel(torch.add)


# ------------------------------------------------------------------ real kernels

_needs_blackwell = pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.has_device_capability(100),
    reason="CuTe DSL kernels need Blackwell or newer",
)


def _scale_kernel_class():
    import cutlass
    import cutlass.cute as cute

    class Scale:
        def __init__(self, factor):
            self.factor = factor

        @cute.jit
        def __call__(self, x: cute.Tensor, y: cute.Tensor):
            self.kernel(x, y).launch(grid=[x.shape[0] // 128, 1, 1], block=[128, 1, 1])

        @cute.kernel
        def kernel(self, x: cute.Tensor, y: cute.Tensor):
            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()
            i = bidx * 128 + tidx
            y[i] = x[i] * cutlass.Float32(self.factor)

    return Scale


def _run_kernels(cached: bool):
    """Compile (installed cache, or the plain DSL compile) and run the kernels on
    fixed inputs; return their outputs."""
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack, make_fake_compact_tensor

    from vllm.models.kimi_k3.nvidia import cute_cache
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common import (
        embedding,
        host,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.verify import lmhead

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    gen = torch.Generator(device=device).manual_seed(0)
    randn = lambda *s: torch.randn(  # noqa: E731
        *s, generator=gen, device=device, dtype=torch.bfloat16
    )
    plain = cute.compile._inner if hasattr(cute.compile, "_inner") else cute.compile
    if not cached:  # the DSL compile without the cache
        saved = cute.compile
        cute.compile = plain
    outputs = {}
    try:
        arch = host.architecture(device)
        weight, hidden = randn(VOCAB, HIDDEN), randn(ROWS, HIDDEN)
        _, logits = lmhead.compile_lmhead_logits.__wrapped__(VOCAB, HIDDEN, arch)
        out = torch.empty(ROWS, VOCAB, device=device, dtype=torch.bfloat16)
        logits(weight, hidden, out, ROWS)
        outputs["lmhead_logits"] = out

        table = randn(VOCAB, HIDDEN)
        ids = torch.tensor([3, 1000, 7, 1023, 1500], device=device, dtype=torch.int64)
        gather = embedding.compile_gather.__wrapped__(torch.int64, VOCAB, arch)
        out = torch.empty(ids.shape[0], HIDDEN, device=device, dtype=torch.bfloat16)
        gather(ids, table, out, 0, 2 * VOCAB)
        outputs["embedding"] = out

        # a CuTe launcher (not TVM-FFI), compiled through the cache entry point
        x = randn(4096).float()
        y = torch.empty_like(x)
        fx = make_fake_compact_tensor(cutlass.Float32, (4096,), assumed_align=16)
        kernel = _scale_kernel_class()(3.0)
        if cached:
            run = cute_cache._compile(plain, kernel, (fx, fx), {})
        else:
            run = plain(kernel, fx, fx)
        run(from_dlpack(x, assumed_align=16), from_dlpack(y, assumed_align=16))
        outputs["scale"] = y
        torch.cuda.synchronize()
    finally:
        if not cached:
            cute.compile = saved
    return {k: v.cpu() for k, v in outputs.items()}


def _worker(path: str, verify: bool):
    from vllm.models.kimi_k3.nvidia import cute_cache

    cute_cache.VERIFY = verify
    cached = _run_kernels(True)
    stats = {k: v for k, v in vars(cute_cache.stats).items() if k != "kernels"}
    reference = _run_kernels(False)
    torch.save({"cached": cached, "reference": reference}, path + ".pt")
    with open(path + ".json", "w") as f:
        json.dump({"stats": stats, "dir": cute_cache.cache_dir()}, f)


def _start(tmp_path, name, verify=False):
    env = dict(os.environ, VLLM_CACHE_ROOT=str(tmp_path / "cache"))
    env.pop("VLLM_KIMI_K3_CUTE_CACHE_DIR", None)  # the test directory
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))  # fmt: skip
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    out = str(tmp_path / name)
    cmd = [sys.executable, os.path.abspath(__file__), out, "1" if verify else "0"]
    subprocess.run(cmd, env=env, check=True, timeout=900)
    with open(out + ".json") as f:
        meta = json.load(f)
    return meta, torch.load(out + ".pt")


def _bitwise(a, b):
    return a.dtype == b.dtype and torch.equal(a.view(torch.uint8), b.view(torch.uint8))


@_needs_blackwell
def test_warm_start_loads_every_kernel_bitwise(tmp_path):
    cold, cold_out = _start(tmp_path, "cold")
    s = cold["stats"]
    assert (s["compiled"], s["loaded"], s["uncached"]) == (3, 0, 0), s
    objects = [
        os.path.join(r, f) for r, _, fs in os.walk(cold["dir"]) for f in fs
        if f.endswith(".o")
    ]  # fmt: skip
    assert len(objects) == 3, objects

    warm, warm_out = _start(tmp_path, "warm", verify=True)
    s = warm["stats"]
    assert (s["compiled"], s["loaded"], s["uncached"]) == (0, 3, 0), s
    for name, ref in cold_out["reference"].items():
        assert ref.abs().sum() > 0, name  # the kernels did real work
        assert _bitwise(cold_out["cached"][name], ref), f"cold {name}"
        assert _bitwise(warm_out["cached"][name], ref), f"warm {name}"

    for path in objects[:1]:  # a corrupt entry: logged, recompiled, replaced
        with open(path, "wb") as f:
            f.write(b"truncated")
    broken, broken_out = _start(tmp_path, "broken")
    s = broken["stats"]
    assert (s["compiled"], s["loaded"]) == (1, 2), s
    for name, ref in cold_out["reference"].items():
        assert _bitwise(broken_out["cached"][name], ref), f"rebuilt {name}"


def _scale_runtime_kernel_class():
    """A kernel object whose ``factor`` is a runtime ``Float32`` argument and whose
    ``rows`` is a ``Constexpr`` (traced)."""
    import cutlass
    import cutlass.cute as cute

    class ScaleRt:
        @cute.jit
        def __call__(self, x: cute.Tensor, y: cute.Tensor, factor: cutlass.Float32,
                     base: cutlass.Int64, rows: cutlass.Constexpr = 128):  # fmt: skip
            self.kernel(x, y, factor).launch(
                grid=[x.shape[0] // rows, 1, 1], block=[rows, 1, 1]
            )

        @cute.kernel
        def kernel(self, x: cute.Tensor, y: cute.Tensor, factor: cutlass.Float32):
            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()
            i = bidx * 128 + tidx
            y[i] = x[i] * factor

    return ScaleRt


def test_runtime_numeric_argument_values_are_not_in_the_key():
    """DSL numeric values (cutlass.Float32(v), Int64(a)) bound to DSL-numeric
    parameters are runtime arguments: not in the key. Plain Python numbers there are
    folded into the trace as constants, so their values stay in the key, as do every
    Constexpr value and the kernel object's state."""
    import cutlass
    from cutlass.cute.runtime import make_fake_compact_tensor

    from vllm.models.kimi_k3.nvidia import cute_cache

    f32, i64 = cutlass.Float32, cutlass.Int64
    fx = make_fake_compact_tensor(cutlass.Float32, (4096,), assumed_align=16)
    k = _scale_runtime_kernel_class()()
    a = _key(k, fx, fx, f32(2.0), i64(1 << 40))
    assert a == _key(k, fx, fx, f32(3.5), i64((1 << 40) + 4096))  # launch values
    assert a != _key(k, fx, fx, 2.0, i64(1 << 40))  # a Python float: a constant
    assert _key(k, fx, fx, 2.0, 7) != _key(k, fx, fx, 3.0, 7)
    assert _key(k, fx, fx, 2.0, 7) != _key(k, fx, fx, 2.0, 8)
    assert a != _key(k, fx, fx, f32(2.0), i64(1 << 40), 256)  # a Constexpr value
    k.tag = "x"  # kernel object state
    assert a != _key(k, fx, fx, f32(2.0), i64(1 << 40))
    # keyword binding follows the same rule
    assert _key(k, fx, fx, factor=f32(1.0), base=i64(3)) == _key(
        k, fx, fx, factor=f32(9.0), base=i64(5)
    )

    def f(n: cutlass.Int32, m: cutlass.Int32, c: cutlass.Constexpr):
        pass

    assert cute_cache._runtime_values(f, (cutlass.Int32(5), 6, 7), {})[0] == (
        cute_cache._runtime_marker(cutlass.Int32, cutlass.Int32(5)), 6, 7
    )


@_needs_blackwell
def test_runtime_argument_hit_is_the_same_kernel(tmp_path, monkeypatch):
    """Two compiles that differ only in the values of DSL numeric runtime arguments:
    one compile, one hit; VERIFY re-traces the hit to the stored DSL module hash
    (exact), and the hit runs with the launch values it is given."""
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack, make_fake_compact_tensor

    from vllm.models.kimi_k3.nvidia import cute_cache

    f32, i64 = cutlass.Float32, cutlass.Int64
    monkeypatch.setenv("VLLM_KIMI_K3_CUTE_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(cute_cache, "VERIFY", True)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    inner = cute.compile._inner if hasattr(cute.compile, "_inner") else cute.compile
    fx = make_fake_compact_tensor(cutlass.Float32, (4096,), assumed_align=16)
    k = _scale_runtime_kernel_class()()
    before = dict(vars(cute_cache.stats))
    cute_cache._compile(inner, k, (fx, fx, f32(2.0), i64(1 << 40)), {})
    hit = cute_cache._compile(inner, k, (fx, fx, f32(3.0), i64((1 << 40) + 64)), {})
    s = vars(cute_cache.stats)
    assert s["compiled"] - before["compiled"] == 1
    assert s["loaded"] - before["loaded"] == 1
    assert s["verified"] - before["verified"] == 1
    assert s["mismatched"] == before["mismatched"]
    x = torch.arange(4096, device=device, dtype=torch.float32)
    y = torch.empty_like(x)
    hit(from_dlpack(x, assumed_align=16), from_dlpack(y, assumed_align=16), 5.0, 0)
    torch.cuda.synchronize()
    assert torch.equal(y, x * 5.0)


@_needs_blackwell
def test_verify_flags_a_too_coarse_key(tmp_path, monkeypatch):
    """Negative control for the restart check: with a deliberately too-coarse key
    (the 13ed4cd676 bug: Python numbers keyed as runtime arguments although the DSL
    folds them into the trace as constants), a compile with another constant hits
    the first entry, and VERIFY's re-trace flags the module-hash mismatch instead
    of returning the wrong kernel."""
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_compact_tensor

    from vllm.models.kimi_k3.nvidia import cute_cache

    monkeypatch.setenv("VLLM_KIMI_K3_CUTE_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(cute_cache, "VERIFY", True)
    torch.cuda.set_device(0)

    def too_coarse(fn, args, kwargs):  # every numeric-annotated value as a marker
        marked = tuple(
            cute_cache._runtime_marker(cutlass.Float32, cutlass.Float32(0.0))
            if isinstance(a, float) else a for a in args
        )
        return marked, kwargs

    monkeypatch.setattr(cute_cache, "_runtime_values", too_coarse)
    inner = cute.compile._inner if hasattr(cute.compile, "_inner") else cute.compile
    fx = make_fake_compact_tensor(cutlass.Float32, (4096,), assumed_align=16)
    k = _scale_runtime_kernel_class()()
    before = dict(vars(cute_cache.stats))
    cute_cache._compile(inner, k, (fx, fx, 2.0, cutlass.Int64(0)), {})
    assert vars(cute_cache.stats)["compiled"] - before["compiled"] == 1
    with pytest.raises(cute_cache.VerifyMismatch):
        cute_cache._compile(inner, k, (fx, fx, 3.0, cutlass.Int64(0)), {})
    assert vars(cute_cache.stats)["mismatched"] - before["mismatched"] == 1


def _scale_kwargs_kernel_class():
    """A TVM-FFI launcher with a defaulted runtime argument: compiled, it calls
    through the DSL's kwargs wrapper."""
    import cutlass
    import cutlass.cute as cute

    class ScaleKw:
        @cute.jit
        def __call__(
            self, x: cute.Tensor, y: cute.Tensor, stream, factor: cutlass.Float32 = 2.0
        ):
            self.kernel(x, y, factor).launch(
                grid=[x.shape[0] // 128, 1, 1], block=[128, 1, 1], stream=stream
            )

        @cute.kernel
        def kernel(self, x: cute.Tensor, y: cute.Tensor, factor: cutlass.Float32):
            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()
            i = bidx * 128 + tidx
            y[i] = x[i] * factor

    return ScaleKw


def _kwargs_worker(path: str):
    """Compile the kwargs launcher through the cache (cold or warm) and the plain
    DSL compile; call each with the default, positionally and by keyword."""
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

    from vllm.models.kimi_k3.nvidia import cute_cache

    cute_cache.VERIFY = True
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    inner = cute.compile._inner if hasattr(cute.compile, "_inner") else cute.compile
    fx = make_fake_compact_tensor(cutlass.Float32, (4096,), assumed_align=16)
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    kernel = _scale_kwargs_kernel_class()()
    options = "--enable-tvm-ffi"
    cached = cute_cache._compile(inner, kernel, (fx, fx, stream), {"options": options})
    plain = inner(kernel, fx, fx, stream, options=options)
    x = torch.arange(4096, device=device, dtype=torch.float32)
    outs = {}
    for name, run in (("cached", cached), ("plain", plain)):
        calls = []
        for args, kwargs in (((), {}), ((3.0,), {}), ((), {"factor": 4.0})):
            y = torch.empty_like(x)
            run(x, y, *args, **kwargs)
            calls.append(y.cpu())
        outs[name] = calls
    torch.cuda.synchronize()
    stats = {k: v for k, v in vars(cute_cache.stats).items() if k != "kernels"}
    torch.save(outs, path + ".pt")
    with open(path + ".json", "w") as f:
        json.dump({"stats": stats}, f)


@_needs_blackwell
def test_kwargs_launcher_is_cached_and_calls_alike(tmp_path):
    env = dict(os.environ, VLLM_CACHE_ROOT=str(tmp_path / "cache"))
    env.pop("VLLM_KIMI_K3_CUTE_CACHE_DIR", None)  # the test directory
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))  # fmt: skip
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    results = []
    for arm in ("cold", "warm"):
        out = str(tmp_path / arm)
        cmd = [sys.executable, os.path.abspath(__file__), "kwargs", out]
        subprocess.run(cmd, env=env, check=True, timeout=900)
        with open(out + ".json") as f:
            results.append((json.load(f)["stats"], torch.load(out + ".pt")))
    (cold, cold_out), (warm, warm_out) = results
    assert (cold["compiled"], cold["uncached"]) == (1, 0), cold
    assert (warm["loaded"], warm["compiled"], warm["uncached"]) == (1, 0, 0), warm
    x = torch.arange(4096, dtype=torch.float32)
    expected = [x * 2.0, x * 3.0, x * 4.0]  # default, positional, keyword
    for outs in (cold_out, warm_out):
        for got, ref, want in zip(outs["cached"], outs["plain"], expected):
            assert torch.equal(got, ref) and torch.equal(got, want)


SHARED_KERNELS = 12


def _shared_worker(path: str, start_file: str):
    """One of two processes compiling the same SHARED_KERNELS kernels in a shared
    phase over one cache directory; writes its stats and outputs."""
    import cutlass
    from cutlass.cute.runtime import from_dlpack, make_fake_compact_tensor

    from vllm.models.kimi_k3.nvidia import cute_cache

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    import cutlass.cute as cute

    inner = cute.compile._inner if hasattr(cute.compile, "_inner") else cute.compile
    fx = make_fake_compact_tensor(cutlass.Float32, (4096,), assumed_align=16)
    Scale = _scale_kernel_class()
    while not os.path.exists(start_file):  # both processes start together
        time.sleep(0.01)
    with cute_cache.shared_phase("test"):
        runs = [
            cute_cache._compile(inner, Scale(float(i + 1)), (fx, fx), {})
            for i in range(SHARED_KERNELS)
        ]
        deferred = sum(isinstance(r, cute_cache._Deferred) for r in runs)
    assert not cute_cache._pending  # the phase resolved every deferred kernel
    x = torch.arange(4096, device=device, dtype=torch.float32)
    outs = []
    for run in runs:
        y = torch.empty_like(x)
        run(from_dlpack(x, assumed_align=16), from_dlpack(y, assumed_align=16))
        outs.append(y.cpu())
    stats = {k: v for k, v in vars(cute_cache.stats).items() if k != "kernels"}
    torch.save(outs, path + ".pt")
    with open(path + ".json", "w") as f:
        json.dump({"stats": stats, "deferred_handles": deferred}, f)


@_needs_blackwell
def test_ranks_split_cold_compiles(tmp_path):
    env = dict(os.environ, VLLM_CACHE_ROOT=str(tmp_path / "cache"))
    env.pop("VLLM_KIMI_K3_CUTE_CACHE_DIR", None)  # the test directory
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))  # fmt: skip
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    start = str(tmp_path / "go")
    procs = [
        subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "shared",
             str(tmp_path / f"p{i}"), start],
            env=env,
        )  # fmt: skip
        for i in range(2)
    ]
    time.sleep(20)  # both imported and waiting
    open(start, "w").close()
    for p in procs:
        assert p.wait(timeout=900) == 0
    metas = []
    for i in range(2):
        with open(tmp_path / f"p{i}.json") as f:
            metas.append(json.load(f))
        outs = torch.load(tmp_path / f"p{i}.pt")
        x = torch.arange(4096, dtype=torch.float32)
        for k, y in enumerate(outs):
            assert torch.equal(y, x * float(k + 1)), (i, k)
    compiled = [m["stats"]["compiled"] for m in metas]
    loaded = [m["stats"]["loaded"] for m in metas]
    # every kernel compiled exactly once over both processes, the rest loaded
    assert sum(compiled) == SHARED_KERNELS, metas
    assert [c + l for c, l in zip(compiled, loaded)] == [SHARED_KERNELS] * 2, metas
    assert all(m["stats"]["uncached"] == 0 for m in metas), metas
    # both did part of the work (the claims split the misses)
    assert min(compiled) > 0, metas
    claims = [
        f for r, _, fs in os.walk(tmp_path / "cache") for f in fs if f.endswith(".claim")
    ]
    assert not claims, claims  # every claim released


if __name__ == "__main__":
    if sys.argv[1] == "shared":
        _shared_worker(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "kwargs":
        _kwargs_worker(sys.argv[2])
    else:
        _worker(sys.argv[1], sys.argv[2] == "1")
