# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""record_jit_events(): the startup warmup verification's JIT recorder.
Backends are mocked, so no GPU."""

from unittest import mock

import pytest

from vllm.utils import jit_monitor

from .test_hooks import _make_fake_knobs, _patch_jit_modules, _triton_hook_kwargs

pytestmark = pytest.mark.cpu_test


def test_record_collects_and_does_not_warn():
    fake = _make_fake_knobs()
    with (
        _patch_jit_modules(fake),
        mock.patch.object(jit_monitor, "_triton_jit_functions", return_value=[]),
        mock.patch.object(jit_monitor.logger, "error") as warn,
        jit_monitor.record_jit_events() as events,
    ):
        fake.runtime.jit_post_compile_hook(**_triton_hook_kwargs("late_kernel"))
    warn.assert_not_called()
    assert len(events) == 1 and events[0].startswith("Triton: late_kernel")


def test_record_is_silent_before_activate_and_warns_after():
    fake = _make_fake_knobs()
    with (
        _patch_jit_modules(fake),
        mock.patch.object(jit_monitor, "_triton_jit_functions", return_value=[]),
    ):
        with jit_monitor.record_jit_events():
            pass
        hook = fake.runtime.jit_post_compile_hook
        with mock.patch.object(jit_monitor.logger, "error") as warn:
            # Hooks installed by the recorder stay quiet until activate().
            hook(**_triton_hook_kwargs("startup_kernel"))
            warn.assert_not_called()
            jit_monitor.activate()
            # activate() must not chain a second copy of the hook.
            assert fake.runtime.jit_post_compile_hook is hook
            hook(**_triton_hook_kwargs("serving_kernel"))
            warn.assert_called_once()


def test_record_reports_cache_growth_without_hook():
    """A compile that bypasses jit_post_compile_hook still shows up."""

    class FakeJit:
        __name__ = "silent_kernel"

        def __init__(self):
            self.device_caches = {0: ({}, None)}

    fn = FakeJit()
    fake = _make_fake_knobs()
    with (
        _patch_jit_modules(fake),
        mock.patch.object(jit_monitor, "_triton_jit_functions", return_value=[fn]),
        jit_monitor.record_jit_events() as events,
    ):
        fn.device_caches[0][0]["k"] = object()
    assert events == ["Triton (cache): silent_kernel +1"]


def test_record_does_not_nest():
    with (
        mock.patch.object(jit_monitor, "_install_jit_hooks"),
        mock.patch.object(jit_monitor, "_triton_jit_functions", return_value=[]),
        jit_monitor.record_jit_events(),
        pytest.raises(RuntimeError, match="does not nest"),
        jit_monitor.record_jit_events(),
    ):
        pass


def test_after_activate_compile_is_tagged_error():
    """e2e health checks grep JIT_AFTER_WARMUP; every compile logs one."""
    fake = _make_fake_knobs()
    with _patch_jit_modules(fake):
        jit_monitor.activate()
    with mock.patch.object(jit_monitor.logger, "error") as err:
        fake.runtime.jit_post_compile_hook(**_triton_hook_kwargs("k1"))
        fake.runtime.jit_post_compile_hook(**_triton_hook_kwargs("k1"))
        jit_monitor.report_compile("torch.compile", "graph for compile range [1, 8]")
    assert err.call_count == 3
    for call in err.call_args_list:
        assert (call[0][0] % call[0][1:]).startswith("JIT_AFTER_WARMUP ")


def test_report_compile_ignored_before_activate():
    with mock.patch.object(jit_monitor.logger, "error") as err:
        jit_monitor.report_compile("torch.compile", "graph")
    err.assert_not_called()
