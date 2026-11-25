# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compile durations and the startup summary of the JIT monitor (logging
only). Backends are mocked, so no GPU."""

import sys
from types import ModuleType, SimpleNamespace
from typing import Any, cast
from unittest import mock

import pytest

from vllm.utils import jit_monitor

from .test_hooks import _make_fake_knobs, _patch_jit_modules, _triton_hook_kwargs

pytestmark = pytest.mark.cpu_test

# Captured at import: conftest replaces it with a no-op for the other tests.
_REAL_CPP_HOOK = jit_monitor._setup_cpp_extension_hook


def _messages(m: mock.MagicMock) -> list[str]:
    return [c[0][0] % c[0][1:] for c in m.call_args_list]


class _FakeCache:
    """A compile cache with the K3 cache's shape: ``_k3_cached``, ``_inner``;
    ``hits`` serve without calling the inner compile, ``deferred`` return a
    ``_Deferred`` (handed to a peer process)."""

    _k3_cached = True

    def __init__(self, inner, hits=(), deferred=()):
        self._inner = inner
        self.hits = set(hits)
        self.deferred = set(deferred)

    def __getitem__(self, options):
        return self

    def __call__(self, fn, *args, **kwargs):
        if fn.__name__ in self.hits:
            return "loaded"
        if fn.__name__ in self.deferred:
            return type("_Deferred", (), {})()
        return self._inner(fn, *args, **kwargs)


def _kernel(name):
    def fn():
        return None

    fn.__name__ = name
    return fn


def _fake_cute(compile_fn, experimental_compile=None):
    cute = cast(Any, ModuleType("cutlass.cute"))
    cute.__path__ = []
    cute.compile = compile_fn
    parent = cast(Any, ModuleType("cutlass"))
    parent.__path__ = []
    parent.cute = cute
    modules = {"cutlass": parent, "cutlass.cute": cute}
    if experimental_compile is not None:
        experimental = cast(Any, ModuleType("cutlass.cute.experimental"))
        experimental.compile = experimental_compile
        cute.experimental = experimental
        modules["cutlass.cute.experimental"] = experimental
    return modules


def _patch(cute_modules):
    return mock.patch.dict(
        sys.modules,
        {
            **cute_modules,
            "triton": None,
            "flashinfer.jit.core": None,
            "flashinfer.jit.cubin_loader": None,
            "tilelang.jit.kernel": None,
        },
    )


def test_startup_cute_compiles_log_duration_and_summary():
    inner = mock.MagicMock(return_value="compiled")
    with _patch(_fake_cute(inner)), mock.patch.object(jit_monitor, "HAS_TRITON", False):
        import cutlass.cute as cute

        jit_monitor.install_timing_hooks()
        with mock.patch.object(jit_monitor.logger, "info") as info:
            assert cute.compile(_kernel("k_a")) == "compiled"
            assert cute.compile(_kernel("k_b")) == "compiled"
        lines = _messages(info)
        assert len(lines) == 2
        assert all(x.startswith("JIT_COMPILE CuTeDSL") for x in lines)
        assert all(x.endswith(" s]") for x in lines)
        assert jit_monitor.compile_stats("startup")["CuTeDSL"][0] == 2
        with mock.patch.object(jit_monitor.logger, "info") as info:
            jit_monitor.activate()
        summary = [x for x in _messages(info) if x.startswith("JIT_COMPILE_SUMMARY")]
        assert len(summary) == 1
        assert "startup: 2 compiles" in summary[0] and "CuTeDSL 2x" in summary[0]


@pytest.mark.parametrize("monitor_first", [True, False])
def test_cache_in_between_labels_compile_load_deferred(monitor_first):
    """Either install order: a miss is a compile, a hit a cache load, a
    deferred kernel 'deferred to peer'; after ready a hit still reports (a
    cold cache would compile it there)."""
    inner = mock.MagicMock(return_value="compiled")
    with _patch(_fake_cute(inner)), mock.patch.object(jit_monitor, "HAS_TRITON", False):
        import cutlass.cute as cute

        if monitor_first:
            jit_monitor.install_timing_hooks()
            cute.compile = _FakeCache(cute.compile, hits={"hit"}, deferred={"peer"})
        else:
            cute.compile = _FakeCache(cute.compile, hits={"hit"}, deferred={"peer"})
            jit_monitor.install_timing_hooks()
        cute.compile(_kernel("miss"))
        cute.compile(_kernel("hit"))
        cute.compile(_kernel("peer"))
        stats = jit_monitor.compile_stats("startup")
        assert stats["CuTeDSL"][0] == 1  # the miss compiled
        if monitor_first:  # the cache wraps the monitor: it sees misses only
            assert set(stats) == {"CuTeDSL"}
        else:
            assert stats["CuTeDSL (cache load)"][0] == 1
            assert stats["CuTeDSL (deferred to peer)"][0] == 1
        inner.assert_called_once()
        jit_monitor.activate()  # the monitor is outermost again
        with mock.patch.object(jit_monitor.logger, "error") as err:
            cute.compile(_kernel("hit"))
        (line,) = _messages(err)
        assert line.startswith("JIT_AFTER_WARMUP CuTeDSL (cache load)")
        assert "hit" in line and line.split("[")[-1].startswith("0.")


def test_experimental_compile_is_logging_only():
    exp = mock.MagicMock(return_value="compiled")
    modules = _fake_cute(mock.MagicMock(return_value="c"), experimental_compile=exp)
    with _patch(modules), mock.patch.object(jit_monitor, "HAS_TRITON", False):
        import cutlass.cute as cute

        with jit_monitor.record_jit_events() as events:
            cute.experimental.compile(_kernel("sdec_kernel"), options="x")
        assert events == []  # logging only: never fails the verification
        jit_monitor.activate(mode="error")
        with mock.patch.object(jit_monitor.logger, "error") as err:
            assert cute.experimental.compile(_kernel("late")) == "compiled"
        (line,) = _messages(err)
        assert line.startswith("JIT_AFTER_WARMUP CuTeDSL experimental")
        assert jit_monitor.compile_stats("serving")["CuTeDSL experimental"][0] == 1


def test_monitor_passes_attributes_through():
    compile_fn = mock.MagicMock(return_value="c")
    compile_fn._compile_options = ("opt",)
    with (
        _patch(_fake_cute(compile_fn)),
        mock.patch.object(jit_monitor, "HAS_TRITON", False),
    ):
        import cutlass.cute as cute

        jit_monitor.install_timing_hooks()
        assert isinstance(cute.compile, jit_monitor._MonitoredCuteCompile)
        assert cute.compile._compile_options == ("opt",)
        # A trace (a cache computing its key) is not a compile.
        cute.compile(_kernel("k"), trace_finalize_hooks=object())
        assert jit_monitor.compile_stats("startup") == {}


def test_triton_compile_duration_from_cache_hook():
    existing = mock.MagicMock(return_value=None)
    fake = _make_fake_knobs()
    fake.runtime.jit_cache_hook = existing
    with _patch_jit_modules(fake):
        jit_monitor.install_timing_hooks()
        kwargs = _triton_hook_kwargs("tk")
        with mock.patch.object(
            jit_monitor.time, "perf_counter", side_effect=[1.0, 3.5]
        ):
            assert fake.runtime.jit_cache_hook(**kwargs) is None
            with mock.patch.object(jit_monitor.logger, "info") as info:
                fake.runtime.jit_post_compile_hook(**kwargs)
    existing.assert_called_once()
    (line,) = _messages(info)
    assert line.startswith("JIT_COMPILE Triton") and "[2.500 s]" in line
    assert jit_monitor.compile_stats("startup")["Triton"] == (1, 2.5)


def test_triton_cache_hook_keeps_existing_skip():
    fake = _make_fake_knobs()
    fake.runtime.jit_cache_hook = mock.MagicMock(return_value=True)
    with _patch_jit_modules(fake):
        jit_monitor.install_timing_hooks()
        assert fake.runtime.jit_cache_hook(**_triton_hook_kwargs("tk")) is True


def test_triton_autotune_listener_logs_misses_only_never_raises():
    fake = _make_fake_knobs()
    fake.autotuning.listener = None
    with _patch_jit_modules(fake):
        jit_monitor.activate(mode="error")
        listener = fake.autotuning.listener
        fn = SimpleNamespace(__name__="tuned")
        with mock.patch.object(jit_monitor.logger, "error") as err:
            listener(fn=fn, key=(8,), duration=1.25, cache_hit=True)
            err.assert_not_called()
            listener(fn=fn, key=(8,), duration=1.25, cache_hit=False)
        (line,) = _messages(err)
        assert "Triton autotune" in line and "[1.250 s]" in line


def test_report_compile_time_adds_seconds_not_compiles():
    jit_monitor.report_compile("torch.compile", "graph for compile range [1, 8]")
    with mock.patch.object(jit_monitor.logger, "info") as info:
        jit_monitor.report_compile_time(
            "torch.compile", "graph for compile range [1, 8]", 4.0
        )
    assert jit_monitor.compile_stats("startup")["torch.compile"] == (1, 4.0)
    (line,) = _messages(info)
    assert line.startswith("JIT_COMPILE_TIME torch.compile") and "[4.000 s]" in line


def test_flashinfer_build_and_cubin_download_logged(tmp_path):
    class JitSpec:
        def __init__(self, name):
            self.name = name

        def _timed_build(self):
            return None

    core = cast(Any, ModuleType("flashinfer.jit.core"))
    core.JitSpec = JitSpec
    cubin = cast(Any, ModuleType("flashinfer.jit.cubin_loader"))
    cubin.FLASHINFER_CUBIN_DIR = tmp_path
    (tmp_path / "have.cubin").write_bytes(b"x")

    def get_artifact(file_name, sha256):
        return b"x"

    cubin.get_artifact = get_artifact
    cubin.get_cubin = get_artifact
    with (
        mock.patch.dict(
            sys.modules,
            {
                "flashinfer.jit.core": core,
                "flashinfer.jit.cubin_loader": cubin,
                "cutlass.cute": None,
                "tilelang.jit.kernel": None,
            },
        ),
        mock.patch.object(jit_monitor, "HAS_TRITON", False),
    ):
        jit_monitor.install_timing_hooks()
        assert cubin.get_cubin is cubin.get_artifact is not get_artifact
        with jit_monitor.record_jit_events() as events:
            JitSpec("fused_moe")._timed_build()
            assert cubin.get_artifact("have.cubin", "s") == b"x"
            assert cubin.get_artifact("new/Bmm_E2m1.cubin", "s") == b"x"
        assert events == []  # logging only
    stats = jit_monitor.compile_stats("startup")
    assert stats["FlashInfer"][0] == 1
    assert stats["FlashInfer cubin"][0] == 2


def test_summary_orders_backends_by_seconds():
    jit_monitor._account("CuTeDSL", "a", 1.0)
    jit_monitor._account("Triton", "b", 5.0)
    with mock.patch.object(jit_monitor.logger, "info") as info:
        jit_monitor.log_compile_summary("startup")
    (line,) = _messages(info)
    assert line.startswith("JIT_COMPILE_SUMMARY startup: 2 compiles, 6.0 s")
    assert line.index("Triton 1x 5.0 s") < line.index("CuTeDSL 1x 1.0 s")
    assert "(slowest b 5.0 s)" in line


def test_deepgemm_cache_watch_logs_builds_and_loads(tmp_path):
    import os

    watch = jit_monitor._DeepGemmCacheWatch(str(tmp_path), interval=3600)
    try:
        entry = tmp_path / "sm100_paged_mqa_logits_metadata.abc"
        entry.mkdir()
        (entry / "kernel.cu").write_text("x")
        os.utime(entry / "kernel.cu", (100.0, 100.0))
        (entry / ".committed").write_text("")
        os.utime(entry / ".committed", (112.5, 112.5))
        with mock.patch.object(jit_monitor.logger, "info") as info:
            jit_monitor._timing_installed = True
            watch.poll()
            watch.poll()  # unchanged: nothing new
            os.utime(entry / ".committed", (200.0, 200.0))
            watch.poll()  # DeepGEMM touched it: a load
        lines = _messages(info)
        assert len(lines) == 2
        assert lines[0].startswith("JIT_COMPILE DeepGEMM kernel build")
        assert "[12.500 s]" in lines[0]
        assert lines[1].startswith("JIT_COMPILE DeepGEMM kernel load")
        assert jit_monitor.compile_stats("startup")["DeepGEMM"][0] == 2
    finally:
        watch.stop()


def test_deepgemm_watch_after_ready_is_logging_only(tmp_path):
    watch = jit_monitor._DeepGemmCacheWatch(str(tmp_path / "missing"), interval=3600)
    try:
        watch.poll()  # a missing directory scans empty
        jit_monitor.activate(mode="error")
        (tmp_path / "missing" / "k.1").mkdir(parents=True)
        (tmp_path / "missing" / "k.1" / ".committed").write_text("")
        with mock.patch.object(jit_monitor.logger, "error") as err:
            watch.poll()
        (line,) = _messages(err)
        assert line.startswith("JIT_AFTER_WARMUP DeepGEMM kernel build")
    finally:
        watch.stop()


def test_deepgemm_load_before_activate_is_a_startup_event(tmp_path):
    """A kernel that warmup loaded, not yet polled by the watch thread when
    warmup ends, is a startup event: ``activate()`` reports the pending
    events first. (Reported at the thread's next poll, up to its interval
    later, it read as a load inside the first step: the DCP ShardQ layer
    test's JIT check then failed on the ranks whose poll fell in its step.)"""
    watch = jit_monitor._DeepGemmCacheWatch(str(tmp_path), interval=3600)
    jit_monitor._deepgemm_watch = watch  # conftest's reset stops it
    entry = tmp_path / "sm100_mqa_logits.abc"
    entry.mkdir()
    (entry / ".committed").write_text("")
    with (
        mock.patch.object(jit_monitor.logger, "error") as err,
        mock.patch.object(jit_monitor.logger, "info"),
    ):
        jit_monitor.activate(mode="error")
        watch.poll()  # the thread's next poll: nothing left to report
    assert _messages(err) == []
    assert jit_monitor.compile_stats("startup")["DeepGEMM"][0] == 1
    assert "DeepGEMM" not in jit_monitor.compile_stats("serving")


def test_poll_cache_watches_reports_pending_events_once(tmp_path):
    import os
    import threading

    watch = jit_monitor._DeepGemmCacheWatch(str(tmp_path), interval=3600)
    jit_monitor._deepgemm_watch = watch
    jit_monitor._timing_installed = True
    entry = tmp_path / "sm100_mqa_logits.abc"
    entry.mkdir()
    (entry / ".committed").write_text("")
    with mock.patch.object(jit_monitor.logger, "info") as info:
        jit_monitor.poll_cache_watches()  # now, not at the thread's next poll
        assert len(_messages(info)) == 1
        os.utime(entry / ".committed", (4e9, 4e9))  # a later load
        threads = [
            threading.Thread(target=jit_monitor.poll_cache_watches) for _ in range(8)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    lines = _messages(info)
    assert len(lines) == 2, lines
    assert lines[1].startswith("JIT_COMPILE DeepGEMM kernel load")


def test_poll_cache_watches_without_watch_is_a_no_op():
    assert jit_monitor._deepgemm_watch is None
    jit_monitor.poll_cache_watches()


def test_cpp_extension_load_timed():
    cpp = cast(Any, ModuleType("torch.utils.cpp_extension"))
    cpp.load = mock.MagicMock(return_value="mod")
    cpp.load_inline = mock.MagicMock(return_value="mod_inline")
    with mock.patch.dict(sys.modules, {"torch.utils.cpp_extension": cpp}):
        _REAL_CPP_HOOK()  # conftest stubs the module attribute
        jit_monitor._timing_installed = True
        with mock.patch.object(jit_monitor.logger, "info") as info:
            assert cpp.load(name="kda_conv_ext", sources=[]) == "mod"
            assert cpp.load_inline("situ_ext", cpp_sources="") == "mod_inline"
    lines = _messages(info)
    assert [x.split(":")[0] for x in lines] == [
        "JIT_COMPILE torch cpp_extension load (build or cached load)",
        "JIT_COMPILE torch cpp_extension load_inline (build or cached load)",
    ]
    assert "kda_conv_ext" in lines[0] and "situ_ext" in lines[1]
