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
    # sources are not in the key (each entry records its own files)
    assert not hasattr(cute_cache, "_source_digest")


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


def test_uncacheable_inputs_are_reported():
    from vllm.models.kimi_k3.nvidia import cute_cache

    k = _Kernel(1.0)
    k.lock = threading.Lock()
    with pytest.raises(cute_cache._Uncacheable, match="lock"):
        _key(k)


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
