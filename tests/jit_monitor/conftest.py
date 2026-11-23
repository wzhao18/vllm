# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading

import pytest

from vllm.utils import jit_monitor


@pytest.fixture(autouse=True)
def _reset_monitor(monkeypatch):
    """Reset global monitor state between tests.

    ``activate()`` installs process-global hooks and flips module globals, so
    without this every test would inherit the previous test's monitor state.
    """

    def reset():
        jit_monitor._active = False
        jit_monitor._mode = "warn"
        jit_monitor._verbose = False
        jit_monitor._triton_hook_installed = False
        jit_monitor._recorded_events = None
        jit_monitor._cutedsl_hook_installed = False
        jit_monitor._tilelang_hook_installed = False
        jit_monitor._tilelang_jitimpl_compile_depth = 0
        jit_monitor._flashinfer_hook_installed = False
        jit_monitor._timing_installed = False
        jit_monitor._triton_autotune_listener_installed = False
        jit_monitor._compile_stats = {"startup": {}, "serving": {}}
        jit_monitor._tls = threading.local()
        jit_monitor._cpp_extension_hook_installed = False
        if jit_monitor._deepgemm_watch is not None:
            jit_monitor._deepgemm_watch.stop()
        jit_monitor._deepgemm_watch = None

    # The DeepGEMM cache watch thread and the cpp_extension hook are tested
    # directly (test_timing.py); other tests must not start or patch them.
    monkeypatch.setattr(jit_monitor, "_setup_deepgemm_watch", lambda: None)
    monkeypatch.setattr(jit_monitor, "_setup_cpp_extension_hook", lambda: None)
    reset()
    yield
    reset()
