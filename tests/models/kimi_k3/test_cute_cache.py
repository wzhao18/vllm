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
    # any source edit is a new key space
    monkeypatch.setattr(cute_cache, "_source_digest", lambda: "edited tree")
    assert c != _key(_Kernel(2.0), torch.empty(0), 3, options="--opt-level 3")


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


if __name__ == "__main__":
    _worker(sys.argv[1], sys.argv[2] == "1")
