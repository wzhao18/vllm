# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Monitor unexpected kernel JIT compilation during inference.

After server warmup completes, any kernel JIT compilation or autotuning event
indicates a cache miss or unexpected input shape that causes a latency spike.
This module registers hooks in supported runtimes to detect such events so
they can be investigated.

Set ``--jit-monitor-mode=error`` to fail fast on unexpected runtime
compilation. Set ``--jit-monitor-verbose`` to log every JIT compile with
additional runtime details. Verbose logging is intentionally opt-in because it
can emit many logs and add overhead.

Currently monitors:
- CuTeDSL ``cute.compile`` and ``cute.experimental.compile`` calls (a call
  that a compile cache in between serves from disk is tagged "cache load")
- Triton ``@triton.autotune`` cache misses (via ``knobs.autotuning.print``
  and, where Triton has it, ``knobs.autotuning.listener``)
- Triton ``@triton.jit`` first-time compilations
  (via ``knobs.runtime.jit_post_compile_hook``)
- TileLang ``@tilelang.jit`` first-time compilations
- FlashInfer JIT module builds and cubin loads (downloads)
- torch.compile graphs (``report_compile`` / ``report_compile_time``)

Every monitored compile logs its wall-clock duration: one ``JIT_COMPILE``
INFO line during startup (once ``install_timing_hooks`` ran), the
``JIT_AFTER_WARMUP`` line after ready. ``activate()`` logs a per-worker
``JIT_COMPILE_SUMMARY`` of the startup compiles (count and seconds by
backend). Timing is logging only: it never changes what compiles or raises.
"""

import contextlib
import functools
import importlib
import os
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import suppress
from typing import Any, Literal, cast

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils.importing import HAS_TRITON

logger = init_logger(__name__)
JitMonitorMode = Literal["warn", "error"]

_active: bool = False
_mode: JitMonitorMode = "warn"
_verbose: bool = False
_triton_hook_installed: bool = False
_cutedsl_hook_installed: bool = False
_tilelang_hook_installed: bool = False
# While set, monitored JIT events are appended here instead of warned/raised
# (see ``record_jit_events``).
_recorded_events: list[str] | None = None
_tilelang_jitimpl_compile_depth: int = 0
_flashinfer_hook_installed: bool = False
# ``install_timing_hooks()`` ran: startup compiles log their duration.
_timing_installed: bool = False
_triton_autotune_listener_installed: bool = False
_cpp_extension_hook_installed: bool = False
_deepgemm_watch: "_DeepGemmCacheWatch | None" = None

# Compile time by phase ("startup" before ``activate()``, "serving" after)
# and backend: [compiles, seconds, slowest (seconds, name)].
_stats_lock = threading.Lock()
_compile_stats: dict[str, dict[str, list[Any]]] = {"startup": {}, "serving": {}}
# Per thread: open CuTe compile frames (nested wrappers) and Triton compile
# start times (``jit_cache_hook`` -> ``jit_post_compile_hook``).
_tls = threading.local()


def is_active() -> bool:
    """Return whether the JIT compilation monitor is currently active."""
    return _active


def activate(*, mode: JitMonitorMode = "warn", verbose: bool = False) -> None:
    """Enable JIT compilation monitoring after warmup.

    Call once per worker process at the end of
    :func:`compile_or_warm_up_model`. After activation every monitored kernel
    compilation or autotuning benchmark that happens during inference will be
    logged as a warning or raised as an error, depending on ``mode``.

    Safe to call multiple times; subsequent calls are no-ops.

    If the user has explicitly set ``TRITON_PRINT_AUTOTUNING=0`` in
    their environment, autotuning printing is left disabled; the JIT
    compilation hook is still registered regardless.
    """
    global _active, _mode, _verbose
    if _active:
        return
    if mode not in ("warn", "error"):
        raise ValueError(f"Unsupported JIT monitor mode: {mode!r}")
    _setup_triton_autotuning_print()
    _install_jit_hooks()
    # Warmup's cache-directory events not yet polled are startup events: the
    # watch thread would report them after _active flips, as post-warmup ones.
    poll_cache_watches()
    # Before _active flips: the startup totals are final here.
    log_compile_summary("startup")
    _active = True
    _mode = mode
    _verbose = verbose

    logger.info(
        "Kernel JIT monitor activated; monitored JIT compilations during "
        "inference will use mode=%s.",
        mode,
    )


def install_timing_hooks() -> None:
    """Install the compile hooks at worker start, so startup compiles (model
    build, warmup, CUDA graph capture) log their duration and add to the
    ``JIT_COMPILE_SUMMARY`` that ``activate()`` logs. Logging only."""
    global _timing_installed
    _timing_installed = True
    _install_jit_hooks()


def _install_jit_hooks() -> None:
    """Install the compile hooks (idempotent; they report via _handle_jit_event)."""
    _setup_triton_jit_hook()
    _setup_triton_autotune_listener()
    _setup_cutedsl_jit_hook()
    _setup_flashinfer_jit_hook()
    _setup_cpp_extension_hook()
    _setup_deepgemm_watch()

    # Refer to #51159. tilelang ships broken symbols
    # on rocm.
    # TODO: Remove the guard once tilelang upstream is fixed.
    if not current_platform.is_rocm():
        _setup_tilelang_jit_hook()


def _triton_jit_functions() -> list[Any]:
    """Every live Triton ``JITFunction`` (gc scan; startup verification only)."""
    if not HAS_TRITON:
        return []
    import gc

    from triton.runtime.jit import JITFunction

    return [obj for obj in gc.get_objects() if isinstance(obj, JITFunction)]


def _triton_compiled_variants(fns: list[Any]) -> dict[int, int]:
    """Number of compiled specializations held by each JITFunction (by id)."""
    counts: dict[int, int] = {}
    for fn in fns:
        total = 0
        for entry in getattr(fn, "device_caches", {}).values():
            kernel_cache = entry[0] if isinstance(entry, tuple) else entry
            with suppress(TypeError):
                total += len(kernel_cache)
        counts[id(fn)] = total
    return counts


@contextlib.contextmanager
def record_jit_events() -> Iterator[list[str]]:
    """Collect every monitored JIT compilation in this block.

    Yields a list that receives one ``"<backend>: <kernel>"`` entry per
    compile event (Triton, CuTeDSL, TileLang hooks), plus one entry per Triton
    ``JITFunction`` whose in-memory compiled-variant count grew in the block
    (catches compiles that bypass ``jit_post_compile_hook``). Events are
    recorded instead of warned or raised. Used by the startup warmup
    verification, which turns any entry into a startup error.
    """
    global _recorded_events
    if _recorded_events is not None:
        raise RuntimeError("record_jit_events() does not nest.")
    _install_jit_hooks()
    fns = _triton_jit_functions()
    before = _triton_compiled_variants(fns)
    events: list[str] = []
    _recorded_events = events
    try:
        yield events
    finally:
        _recorded_events = None
        after = _triton_compiled_variants(fns)
        hooked = {event.split(": ", 1)[-1].split(" ", 1)[0] for event in events}
        for fn in fns:
            grew = after.get(id(fn), 0) - before.get(id(fn), 0)
            name = getattr(fn, "__name__", None) or getattr(fn, "name", "<unknown>")
            if grew > 0 and name not in hooked:
                events.append(f"Triton (cache): {name} +{grew}")


# ------------------------------------------------------------------
# Triton autotuning print
# ------------------------------------------------------------------


def _setup_triton_autotuning_print() -> None:
    """Enable ``TRITON_PRINT_AUTOTUNING`` unless the user opted out."""
    if not HAS_TRITON:
        return
    from triton import knobs  # type: ignore[import-untyped]

    user_val = os.environ.get("TRITON_PRINT_AUTOTUNING")
    if user_val == "0":
        logger.debug(
            "TRITON_PRINT_AUTOTUNING=0 set by user; "
            "autotuning messages will stay suppressed."
        )
        return

    knobs.autotuning.print = True


# ------------------------------------------------------------------
# Triton JIT compilation hook
# ------------------------------------------------------------------


def _account(backend: str, fn_name: str, seconds: float | None, count: int = 1):
    """Add one compile (``count``) of ``seconds`` to this phase's totals."""
    phase = "serving" if _active else "startup"
    with _stats_lock:
        entry = _compile_stats[phase].setdefault(backend, [0, 0.0, (0.0, "")])
        entry[0] += count
        entry[1] += seconds or 0.0
        if seconds is not None and seconds > entry[2][0]:
            entry[2] = (seconds, fn_name)


def compile_stats(phase: str = "startup") -> dict[str, tuple[int, float]]:
    """{backend: (compiles, seconds)} of ``phase`` ("startup" or "serving")."""
    with _stats_lock:
        return {b: (e[0], e[1]) for b, e in _compile_stats[phase].items()}


def log_compile_summary(phase: str = "startup") -> None:
    """One INFO line: compiles and seconds of ``phase`` in total and by backend
    (with each backend's slowest compile)."""
    with _stats_lock:
        rows = sorted(_compile_stats[phase].items(), key=lambda kv: -kv[1][1])
        rows = [(b, e[0], e[1], e[2]) for b, e in rows]
    count = sum(r[1] for r in rows)
    seconds = sum(r[2] for r in rows)
    by_backend = ", ".join(
        f"{b} {n}x {t:.1f} s" + (f" (slowest {s[1]} {s[0]:.1f} s)" if s[1] else "")
        for b, n, t, s in rows
    )
    logger.info(
        "JIT_COMPILE_SUMMARY %s: %d compiles, %.1f s%s",
        phase,
        count,
        seconds,
        f"; {by_backend}" if by_backend else "",
    )


def _seconds_suffix(seconds: float | None) -> str:
    return "" if seconds is None else f" [{seconds:.3f} s]"


def _gates_now() -> bool:
    """Whether a gated event raised before its compile would: error mode."""
    return _active and _mode == "error" and _recorded_events is None


def _handle_jit_event(
    *,
    backend: str,
    event: str,
    fn_name: str,
    detail: str | None = None,
    seconds: float | None = None,
    gate: bool = True,
    count: int = 1,
) -> None:
    """Report one monitored compile. ``seconds``: its wall-clock duration
    (None when the hook cannot time it). ``gate=False``: logging only (a
    backend whose events neither fail the startup verification nor raise in
    error mode)."""
    _account(backend, fn_name, seconds, count)
    if count == 0:  # the duration of a compile reported before it ran
        logger.info(
            "JIT_COMPILE_TIME %s: %s%s", backend, fn_name, _seconds_suffix(seconds)
        )
        return
    if _recorded_events is not None and gate:
        _recorded_events.append(
            f"{backend}: {fn_name}"
            + (f" ({detail})" if detail else "")
            + _seconds_suffix(seconds)
        )
        return
    if not _active:
        # Startup compiles (warmup, CUDA graph capture) are expected: log
        # their duration only (hooks run from ``install_timing_hooks``).
        # Per-kernel lines for compiles only: a build loads thousands of
        # cached kernels (counted in the summary).
        if _timing_installed and not backend.endswith(
            (" (cache load)", " (deferred to peer)")
        ):
            logger.info(
                "JIT_COMPILE %s %s: %s%s",
                backend,
                event,
                fn_name,
                (f" ({detail})" if detail else "") + _seconds_suffix(seconds),
            )
        return
    message = (
        "%s %s during inference: %s%s. "
        "This causes a latency spike; consider extending warmup "
        "to cover this shape/config."
    )
    detail_suffix = (f" ({detail})" if detail else "") + _seconds_suffix(seconds)
    args = (backend, event, fn_name, detail_suffix)

    if _mode == "error" and gate:
        raise RuntimeError(message % args)

    # Every post-warmup compile is an ERROR line with a fixed tag that the e2e
    # health checks grep for (JIT_AFTER_WARMUP); a run that has one did not
    # measure the steady state.
    logger.error("JIT_AFTER_WARMUP " + message, *args)


def report_compile(backend: str, fn_name: str, detail: str | None = None) -> None:
    """Report a compile from a runtime without a hook here (e.g. torch.compile
    graphs): recorded under ``record_jit_events``, reported after
    ``activate()``, ignored during startup."""
    _handle_jit_event(
        backend=backend, event="compilation", fn_name=fn_name, detail=detail
    )


def report_compile_time(backend: str, fn_name: str, seconds: float) -> None:
    """Log the duration of a compile already counted by ``report_compile``
    (logging only: adds seconds, not a compile, and never raises)."""
    _handle_jit_event(
        backend=backend,
        event="compilation time",
        fn_name=fn_name,
        seconds=seconds,
        gate=False,
        count=0,
    )


def _safe_repr(value: object, *, max_len: int = 120) -> str:
    try:
        text = repr(value)
    except Exception:
        text = f"<{type(value).__name__}>"
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def _get_compile_info(kwargs: Mapping[str, object]) -> dict:
    compile_info = kwargs.get("compile")
    if isinstance(compile_info, dict):
        return compile_info
    return {}


def _constant_name(fn: object, path: object) -> str:
    jit_function = getattr(fn, "jit_function", None)
    params = getattr(jit_function, "params", ())
    if isinstance(path, tuple) and path and isinstance(path[0], int):
        idx = path[0]
        if idx < len(params):
            param_name = getattr(params[idx], "name", None)
            if param_name is not None:
                if len(path) == 1:
                    return param_name
                suffix = "".join(f"[{part!r}]" for part in path[1:])
                return f"{param_name}{suffix}"
    return str(path)


def _format_constants(fn: object, compile_info: Mapping[str, object]) -> str:
    constants = compile_info.get("constants")
    if not isinstance(constants, Mapping) or not constants:
        return "{}"

    items = sorted(
        (
            (_constant_name(fn, path), _safe_repr(value))
            for path, value in constants.items()
        ),
        key=lambda item: item[0],
    )
    return "{" + ", ".join(f"{name}={value}" for name, value in items) + "}"


def _format_signature(compile_info: Mapping[str, object]) -> str:
    signature = compile_info.get("signature")
    if not isinstance(signature, Mapping) or not signature:
        return "{}"
    items = sorted((str(k), _safe_repr(v)) for k, v in signature.items())
    return "{" + ", ".join(f"{name}={value}" for name, value in items) + "}"


def _format_extra_compile_info(compile_info: Mapping[str, object]) -> str:
    skip_keys = frozenset(
        {
            "constants",
            "signature",
            "key",
            "fn",
            "name",
        }
    )
    items = [
        f"{name}={_safe_repr(value)}"
        for name, value in sorted(compile_info.items())
        if name not in skip_keys
    ]
    return "{" + ", ".join(items) + "}"


def _format_verbose_triton_compile_details(kwargs: Mapping[str, object]) -> str:
    compile_info = _get_compile_info(kwargs)
    fn = kwargs.get("fn")
    key = compile_info.get("key") or kwargs.get("key")
    return (
        f"constexprs={_format_constants(fn, compile_info)}; "
        f"signature={_format_signature(compile_info)}; "
        f"extra_compile_info={_format_extra_compile_info(compile_info)}; "
        f"key={_safe_repr(key)}"
    )


def _log_triton_jit_compile(fn_name: str, kwargs, seconds: float | None = None) -> None:
    detail = (
        _format_verbose_triton_compile_details(kwargs)
        if _verbose or _recorded_events is not None
        else None
    )
    _handle_jit_event(
        backend="Triton",
        event="kernel JIT compilation",
        fn_name=fn_name,
        detail=detail,
        seconds=seconds,
    )


def _triton_compile_starts() -> dict[tuple[int, str], float]:
    starts = getattr(_tls, "triton_starts", None)
    if starts is None:
        starts = _tls.triton_starts = {}
    return starts


def _triton_compile_id(kwargs: Mapping[str, object]) -> tuple[int, str]:
    return id(kwargs.get("fn")), _safe_repr(kwargs.get("key"), max_len=4096)


def _setup_triton_jit_hook() -> None:
    """Register a jit_post_compile_hook that warns on compilation."""
    global _triton_hook_installed
    if not HAS_TRITON or _triton_hook_installed:
        return
    from triton import knobs
    from triton.runtime import jit as _triton_jit

    # kernels pass non-JSON-serializable constexprs
    # make that serialization non-fatal
    _orig = _triton_jit.serialize_specialization_data
    if not getattr(_orig, "_vllm_guarded", False):

        @functools.wraps(_orig)
        def _guarded(*args, **kwargs):
            try:
                return _orig(*args, **kwargs)
            except (TypeError, ValueError):
                return None  # best-effort metadata; monitor ignores it

        cast(Any, _guarded)._vllm_guarded = True
        _triton_jit.serialize_specialization_data = _guarded

    existing_hook = knobs.runtime.jit_post_compile_hook

    def _on_jit_compile(**kwargs):
        # `jit_post_compile_hook` is Triton internal API and its
        # signature has changed across releases (kwargs added/renamed).
        # Accept **kwargs so an upstream change cannot crash this hook
        # with TypeError, and forward the full kwarg set to any
        # pre-existing hook unchanged.
        fn = kwargs.get("fn")
        fn_name = getattr(fn, "name", "<unknown>")
        started = _triton_compile_starts().pop(_triton_compile_id(kwargs), None)
        seconds = None if started is None else time.perf_counter() - started
        _log_triton_jit_compile(fn_name, kwargs, seconds)
        if existing_hook is not None:
            return existing_hook(**kwargs)
        return None

    knobs.runtime.jit_post_compile_hook = _on_jit_compile
    # `jit_cache_hook` runs before each compile (same kwargs): it only stamps
    # the start for the duration and returns what a pre-existing hook returns
    # (a truthy value skips the compile, as before).
    if hasattr(knobs.runtime, "jit_cache_hook"):
        existing_cache_hook = knobs.runtime.jit_cache_hook

        def _on_jit_compile_start(**kwargs):
            _triton_compile_starts()[_triton_compile_id(kwargs)] = time.perf_counter()
            if existing_cache_hook is not None:
                return existing_cache_hook(**kwargs)
            return None

        knobs.runtime.jit_cache_hook = _on_jit_compile_start
    _triton_hook_installed = True


def _setup_triton_autotune_listener() -> None:
    """Log each Triton autotune benchmark with its duration (where Triton
    has ``knobs.autotuning.listener``). Logging only: an autotune also
    compiles its configs, which the JIT hook gates."""
    global _triton_autotune_listener_installed
    if not HAS_TRITON or _triton_autotune_listener_installed:
        return
    from triton import knobs

    autotuning = getattr(knobs, "autotuning", None)
    if autotuning is None or not hasattr(autotuning, "listener"):
        return
    existing = autotuning.listener

    def _on_autotune(**kwargs):
        if not kwargs.get("cache_hit"):
            fn = kwargs.get("fn")
            _handle_jit_event(
                backend="Triton autotune",
                event="autotuning",
                fn_name=getattr(fn, "__name__", None)
                or getattr(fn, "name", "<unknown>"),
                detail=f"key={_safe_repr(kwargs.get('key'))}",
                seconds=kwargs.get("duration"),
                gate=False,
            )
        if existing is not None:
            return existing(**kwargs)
        return None

    autotuning.listener = _on_autotune
    _triton_autotune_listener_installed = True


# ------------------------------------------------------------------
# CuTeDSL JIT compilation hook
# ------------------------------------------------------------------


class _MonitoredCuteCompile:
    """Times and reports CuTe DSL compiles; a plain function would break
    ``cute.compile[opts]``. Other attributes pass through to the wrapped
    compile (e.g. a compile cache's markers and options).

    A compile cache can sit on either side: wrapped by this (every call is a
    new kernel variant; one the cache serves from disk is reported as a
    "cache load") or wrapping this (only its misses reach this). Nested
    monitored calls report once, from the outermost.
    """

    def __init__(self, inner, backend: str = "CuTeDSL", gate: bool = True):
        self._inner = inner
        self._backend = backend
        self._gate = gate

    def __getitem__(self, options) -> "_MonitoredCuteCompile":
        return _MonitoredCuteCompile(self._inner[options], self._backend, self._gate)

    def __getattr__(self, name):
        return getattr(self.__dict__["_inner"], name)

    def __call__(self, *args, **kwargs):
        if "trace_finalize_hooks" in kwargs:  # a trace (cache key), not a compile
            return self._inner(*args, **kwargs)
        kernel = args[0] if args else kwargs.get("function")
        kernel_name = getattr(kernel, "__name__", None)
        if kernel_name is None:
            kernel_name = (
                kernel.__class__.__name__ if kernel is not None else "<unknown>"
            )
        frames = getattr(_tls, "cute_frames", None)
        if frames is None:
            frames = _tls.cute_frames = []
        if frames:  # inside a monitored call: it compiles, the outer reports
            frames[-1]["compiled"] = True
            frames.append({"compiled": True})
            try:
                return self._inner(*args, **kwargs)
            finally:
                frames.pop()
        if self._gate and _gates_now():
            _handle_jit_event(
                backend=self._backend, event="JIT compilation", fn_name=kernel_name
            )
        frame = {"compiled": False}
        frames.append(frame)
        start = time.perf_counter()
        result = None
        try:
            result = self._inner(*args, **kwargs)
            return result
        finally:
            frames.pop()
            seconds = time.perf_counter() - start
            # A cache in between that served it without compiling (no
            # monitored inner call) took a load, or handed it to a peer
            # process (it resolves at the end of the build phase); with no
            # cache in between (or one that cannot tell) it is a compile.
            kind = ""
            cached = getattr(self._inner, "_k3_cached", False) is True
            if cached and not frame["compiled"]:
                deferred = type(result).__name__ == "_Deferred"
                kind = " (deferred to peer)" if deferred else " (cache load)"
            _handle_jit_event(
                backend=self._backend + kind,
                event="JIT compilation",
                fn_name=kernel_name,
                seconds=seconds,
                gate=self._gate,
            )


def _wrap_compile(owner: Any, backend: str, gate: bool) -> None:
    """Make ``owner.compile`` a monitored compile if its outermost layer is
    not one (a compile cache may have wrapped an earlier monitor)."""
    compile_fn = getattr(owner, "compile", None)
    if compile_fn is None:
        return
    if not isinstance(compile_fn, _MonitoredCuteCompile):
        compile_fn = owner.compile = _MonitoredCuteCompile(compile_fn, backend, gate)
    # A compile cache under the monitor: monitor what it calls on a miss too,
    # so a miss reports "compiled" and a hit "cache load".
    cache = compile_fn._inner
    inner = getattr(cache, "_inner", None)
    if (
        getattr(cache, "_k3_cached", False) is True
        and inner is not None
        and not isinstance(inner, _MonitoredCuteCompile)
    ):
        cache._inner = _MonitoredCuteCompile(inner, backend, gate)


def _setup_cutedsl_jit_hook() -> None:
    """Wrap ``cutlass.cute.compile`` (gated, as before) and
    ``cutlass.cute.experimental.compile`` (logging only) to report compiles.
    Re-wraps when a compile cache installed itself over an earlier monitor,
    so the monitor stays outermost and sees every new kernel variant."""
    global _cutedsl_hook_installed

    try:
        import cutlass.cute as cute
    except Exception:
        logger.debug("CuTeDSL is not available; skipping CuTeDSL JIT monitor.")
        return

    _wrap_compile(cute, "CuTeDSL", gate=True)
    try:
        experimental = importlib.import_module("cutlass.cute.experimental")
    except Exception:
        experimental = None
    if experimental is not None:
        _wrap_compile(experimental, "CuTeDSL experimental", gate=False)
    _cutedsl_hook_installed = True


# ------------------------------------------------------------------
# TileLang JIT compilation hook
# ------------------------------------------------------------------


def _tilelang_arg(
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
    index: int,
    name: str,
    default: object = None,
) -> object:
    if len(args) > index:
        return args[index]
    return kwargs.get(name, default)


def _tilelang_kernel_name(func: object) -> str:
    attrs = getattr(func, "attrs", None)
    global_symbol = None
    get_attr = getattr(attrs, "get", None)
    if callable(get_attr):
        try:
            global_symbol = get_attr("global_symbol")
        except Exception:
            global_symbol = None
    elif isinstance(attrs, Mapping):
        global_symbol = attrs.get("global_symbol")

    if global_symbol is not None:
        return str(global_symbol)

    name = getattr(func, "__name__", None)
    if name is not None:
        return str(name)
    return func.__class__.__name__ if func is not None else "<unknown>"


def _tilelang_call_kwargs(kwargs: Mapping[str, object]) -> dict[str, object]:
    call_kwargs = dict(kwargs)
    tune_params = call_kwargs.pop("__tune_params", {})
    if isinstance(tune_params, Mapping):
        call_kwargs.update(tune_params)
    return call_kwargs


def _tilelang_cache_miss_key(
    jit_impl: object,
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
) -> object | None:
    if kwargs.get("__return_compile_arguments", False):
        return None

    call_kwargs = _tilelang_call_kwargs(kwargs)
    func = getattr(jit_impl, "func", None)
    parse_args = getattr(func, "parse_args", None)
    if not callable(parse_args):
        return None

    try:
        if getattr(jit_impl, "mode", None) == "auto":
            impl = cast(Any, jit_impl)
            mode = impl._infer_jit_mode(*args, **call_kwargs)
            impl.mode = mode
            if func is not None:
                func.set_mode(mode)
        key, _ = parse_args(*args, **call_kwargs)
    except Exception:
        return None

    cache = getattr(jit_impl, "_kernel_cache", {})
    if isinstance(cache, Mapping):
        return key if key not in cache else None

    get = getattr(cache, "get", None)
    if not callable(get):
        return key
    try:
        return key if get(key) is None else None
    except Exception:
        return key


def _format_tilelang_runtime_shapes(
    jit_impl: object,
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
) -> str:
    signature = getattr(jit_impl, "signature", None)
    bind_partial = getattr(signature, "bind_partial", None)
    if not callable(bind_partial):
        return "{}"

    try:
        bound = bind_partial(*args, **_tilelang_call_kwargs(kwargs))
        bound.apply_defaults()
    except Exception:
        return "{}"

    items = []
    for name, value in bound.arguments.items():
        if str(name).startswith("__"):
            continue
        if not (hasattr(value, "shape") and hasattr(value, "dtype")):
            continue
        shape = getattr(value, "shape", None)
        if shape is not None:
            with suppress(TypeError):
                shape = tuple(shape)
        items.append(f"{name}={_safe_repr(shape, max_len=80)}")
    return "{" + ", ".join(items) + "}"


def _format_verbose_tilelang_compile_details(
    jit_impl: object,
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
    cache_key: object,
) -> str:
    return (
        f"cache_key={_safe_repr(cache_key, max_len=400)}; "
        f"runtime_shapes={_format_tilelang_runtime_shapes(jit_impl, args, kwargs)}"
    )


def _log_tilelang_jit_compile(
    fn_name: str,
    detail: str | None = None,
    seconds: float | None = None,
) -> None:
    _handle_jit_event(
        backend="TileLang",
        event="JIT compilation",
        fn_name=fn_name,
        detail=detail,
        seconds=seconds,
    )


def _setup_tilelang_jit_hook() -> None:
    """Wrap TileLang JIT entry points to warn on compilation."""
    global _tilelang_hook_installed
    if _tilelang_hook_installed:
        return

    try:
        tilelang_kernel = importlib.import_module("tilelang.jit.kernel")
    except Exception:
        logger.debug("TileLang is not available; skipping TileLang JIT monitor.")
        return

    jit_kernel_cls = getattr(tilelang_kernel, "JITKernel", None)
    if jit_kernel_cls is None:
        logger.debug(
            "TileLang JITKernel is unavailable; skipping TileLang JIT monitor."
        )
        return

    try:
        tilelang_jit = importlib.import_module("tilelang.jit")
    except Exception:
        tilelang_jit = None
    jit_impl_cls = getattr(tilelang_jit, "JITImpl", None)
    original_init = jit_kernel_cls.__init__

    @functools.wraps(original_init)
    def _init_with_monitor(self, *args, **kwargs):
        from_database = bool(_tilelang_arg(args, kwargs, 7, "from_database", False))
        if from_database or _tilelang_jitimpl_compile_depth != 0:
            return original_init(self, *args, **kwargs)
        name = _tilelang_kernel_name(_tilelang_arg(args, kwargs, 0, "func"))
        if _gates_now():
            _log_tilelang_jit_compile(name)
        start = time.perf_counter()
        try:
            return original_init(self, *args, **kwargs)
        finally:
            _log_tilelang_jit_compile(name, seconds=time.perf_counter() - start)

    jit_kernel_cls.__init__ = _init_with_monitor

    if jit_impl_cls is not None:
        # JITImpl.__call__ only calls compile() on a kernel-cache miss, so
        # hooking compile() keeps the per-call hot path untouched.
        original_compile = jit_impl_cls.compile

        @functools.wraps(original_compile)
        def _compile_with_monitor(self, *args, **kwargs):
            global _tilelang_jitimpl_compile_depth
            _tilelang_jitimpl_compile_depth += 1
            try:
                detail = None
                if _verbose:
                    detail = _format_verbose_tilelang_compile_details(
                        self, args, kwargs, _tilelang_cache_miss_key(self, args, kwargs)
                    )
                func = getattr(self, "func", None)
                orig_func = getattr(func, "orig_func", None)
                name = _tilelang_kernel_name(orig_func or func)
                if _gates_now():
                    _log_tilelang_jit_compile(name, detail)
                start = time.perf_counter()
                try:
                    return original_compile(self, *args, **kwargs)
                finally:
                    _log_tilelang_jit_compile(
                        name, detail, seconds=time.perf_counter() - start
                    )
            finally:
                _tilelang_jitimpl_compile_depth -= 1

        jit_impl_cls.compile = _compile_with_monitor

    _tilelang_hook_installed = True


# ------------------------------------------------------------------
# FlashInfer JIT build / cubin load hook (logging only)
# ------------------------------------------------------------------


def _setup_flashinfer_jit_hook() -> None:
    """Time FlashInfer JIT module builds (``JitSpec._timed_build``, else
    ``JitSpec.build``) and cubin loads (``cubin_loader.get_artifact``, which
    downloads a cubin missing from the local cubin directory). Logging only:
    these events neither fail the startup verification nor raise."""
    global _flashinfer_hook_installed
    if _flashinfer_hook_installed:
        return
    try:
        fi_core = importlib.import_module("flashinfer.jit.core")
        fi_cubin = importlib.import_module("flashinfer.jit.cubin_loader")
    except Exception:
        logger.debug("FlashInfer JIT is not available; skipping its monitor.")
        return

    spec_cls = getattr(fi_core, "JitSpec", None)
    method = next(
        (m for m in ("_timed_build", "build") if spec_cls and hasattr(spec_cls, m)),
        None,
    )
    if method is not None and not getattr(
        getattr(spec_cls, method), "_vllm_monitored", False
    ):
        original_build = getattr(spec_cls, method)

        @functools.wraps(original_build)
        def _build_with_monitor(self, *args, **kwargs):
            start = time.perf_counter()
            try:
                return original_build(self, *args, **kwargs)
            finally:
                _handle_jit_event(
                    backend="FlashInfer",
                    event="JIT module build",
                    fn_name=str(getattr(self, "name", "<unknown>")),
                    seconds=time.perf_counter() - start,
                    gate=False,
                )

        cast(Any, _build_with_monitor)._vllm_monitored = True
        setattr(spec_cls, method, _build_with_monitor)

    original_get = getattr(fi_cubin, "get_artifact", None)
    if original_get is not None and not getattr(original_get, "_vllm_monitored", False):
        cubin_dir = getattr(fi_cubin, "FLASHINFER_CUBIN_DIR", None)

        @functools.wraps(original_get)
        def _get_artifact_with_monitor(file_name, *args, **kwargs):
            local = None
            with suppress(Exception):
                local = os.path.exists(os.path.join(str(cubin_dir), file_name))
            start = time.perf_counter()
            try:
                return original_get(file_name, *args, **kwargs)
            finally:
                _handle_jit_event(
                    backend="FlashInfer cubin",
                    event="load" if local else "download",
                    fn_name=os.path.basename(str(file_name)),
                    seconds=time.perf_counter() - start,
                    gate=False,
                )

        cast(Any, _get_artifact_with_monitor)._vllm_monitored = True
        fi_cubin.get_artifact = _get_artifact_with_monitor
        if getattr(fi_cubin, "get_cubin", None) is original_get:
            fi_cubin.get_cubin = _get_artifact_with_monitor
    _flashinfer_hook_installed = True


# ------------------------------------------------------------------
# torch.utils.cpp_extension build/load hook (logging only)
# ------------------------------------------------------------------


def _setup_cpp_extension_hook() -> None:
    """Time ``torch.utils.cpp_extension.load`` / ``load_inline`` (in-tree
    .cu extensions built on first use; a cached build still checks and
    imports). Callers that import the function at call time see the wrapper.
    Logging only."""
    global _cpp_extension_hook_installed
    if _cpp_extension_hook_installed:
        return
    try:
        cpp_extension = importlib.import_module("torch.utils.cpp_extension")
    except Exception:
        return
    for attr in ("load", "load_inline"):
        original = getattr(cpp_extension, attr, None)
        if original is None or getattr(original, "_vllm_monitored", False) is True:
            continue

        def _make(original=original, attr=attr):
            @functools.wraps(original)
            def _load_with_monitor(*args, **kwargs):
                name = kwargs.get("name", args[0] if args else "<unknown>")
                start = time.perf_counter()
                try:
                    return original(*args, **kwargs)
                finally:
                    _handle_jit_event(
                        backend="torch cpp_extension",
                        event=f"{attr} (build or cached load)",
                        fn_name=str(name),
                        seconds=time.perf_counter() - start,
                        gate=False,
                    )

            cast(Any, _load_with_monitor)._vllm_monitored = True
            return _load_with_monitor

        setattr(cpp_extension, attr, _make())
    _cpp_extension_hook_installed = True


# ------------------------------------------------------------------
# DeepGEMM JIT cache watch (logging only)
# ------------------------------------------------------------------


class _DeepGemmCacheWatch:
    """DeepGEMM compiles and caches its kernels in C++ (no Python hook): watch
    its cache directory (``DG_JIT_CACHE_DIR``/cache, else
    ``VLLM_CACHE_ROOT``/deep_gemm/cache, as vllm.utils.deep_gemm sets it)
    instead. A new ``<kernel>.<hash>/.committed`` is a build (its duration:
    from the entry's oldest file to the commit); a newer ``.committed`` mtime
    on an existing entry is a load (DeepGEMM touches it). The directory is
    shared by the processes that use it, so an event may come from another
    process; with a per-run cache directory they are this run's.

    The thread reports an event at its next poll, up to ``interval`` after it
    happened; ``poll_cache_watches()`` reports the pending ones at once, at a
    phase boundary (``activate()``), so each lands in the phase it happened
    in. Polls are serialized: an event is reported once."""

    def __init__(self, root: str, interval: float = 2.0):
        self.root = root
        self.interval = interval
        self._lock = threading.Lock()
        self.seen = self._scan()
        self._stop = threading.Event()
        self.thread = threading.Thread(
            target=self._run, name="jit-monitor-deepgemm", daemon=True
        )
        self.thread.start()

    def _scan(self) -> dict[str, float]:
        seen: dict[str, float] = {}
        with suppress(OSError):
            for entry in os.scandir(self.root):
                with suppress(OSError):
                    seen[entry.name] = os.stat(
                        os.path.join(entry.path, ".committed")
                    ).st_mtime
        return seen

    def poll(self) -> None:
        with self._lock:
            self._poll()

    def _poll(self) -> None:
        now = self._scan()
        for name, mtime in now.items():
            before = self.seen.get(name)
            if before is not None and mtime <= before:
                continue
            seconds = None
            if before is None:
                with suppress(OSError, ValueError):
                    entry = os.path.join(self.root, name)
                    first = min(
                        os.stat(os.path.join(entry, f)).st_mtime
                        for f in os.listdir(entry)
                        if f != ".committed"
                    )
                    seconds = max(0.0, mtime - first)
            _handle_jit_event(
                backend="DeepGEMM",
                event="kernel build" if before is None else "kernel load",
                fn_name=name,
                detail="cache dir watch",
                seconds=seconds,
                gate=False,
            )
        self.seen = now

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            with suppress(Exception):
                self.poll()

    def stop(self) -> None:
        self._stop.set()


def poll_cache_watches() -> None:
    """Report the cache-directory watches' pending events now, on this thread.

    A watch thread reports an event (a DeepGEMM kernel build or load) at its
    next poll, up to its interval after the event. Call at a phase boundary
    (``activate()`` does) so the events before it are reported before it."""
    if _deepgemm_watch is not None:
        _deepgemm_watch.poll()


def _setup_deepgemm_watch() -> None:
    global _deepgemm_watch
    if _deepgemm_watch is not None:
        return
    from vllm import envs

    # The directory vllm.utils.deep_gemm points DeepGEMM at when unset.
    base = os.environ.get("DG_JIT_CACHE_DIR") or os.path.join(
        envs.VLLM_CACHE_ROOT, "deep_gemm"
    )
    # Watched even before DeepGEMM creates it (a scan of a missing dir is empty).
    _deepgemm_watch = _DeepGemmCacheWatch(os.path.join(base, "cache"))


@contextlib.contextmanager
def numba_workqueue_threading_layer() -> Iterator[None]:
    """Force numba's fork-safe `workqueue` threading layer for this block.

    GNU OpenMP (numba's default `omp` threading layer) aborts the process
    if a forked child re-enters an OpenMP-active runtime. vLLM forks the
    EngineCore subprocess from a process that may already have launched
    numba's parallel accelerator, so the first call to any
    `@njit(parallel=True)` function must happen under `workqueue` instead.
    The threading layer choice is sticky for the life of the process once
    launched, so restoring the config on exit does not undo the effect.
    """
    import numba

    key = "NUMBA_THREADING_LAYER"
    previous_env = os.environ.get(key)
    previous_config = numba.config.THREADING_LAYER
    os.environ[key] = "workqueue"
    numba.config.THREADING_LAYER = "workqueue"
    try:
        yield
    finally:
        if previous_env is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous_env
        numba.config.THREADING_LAYER = previous_config
