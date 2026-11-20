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
- CuTeDSL cute.compile calls
- Triton ``@triton.autotune`` cache misses (via ``knobs.autotuning.print``)
- Triton ``@triton.jit`` first-time compilations
  (via ``knobs.runtime.jit_post_compile_hook``)
- TileLang ``@tilelang.jit`` first-time compilations
"""

import contextlib
import functools
import importlib
import os
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
    _active = True
    _mode = mode
    _verbose = verbose

    _setup_triton_autotuning_print()
    _install_jit_hooks()

    logger.info(
        "Kernel JIT monitor activated; monitored JIT compilations during "
        "inference will use mode=%s.",
        mode,
    )


def _install_jit_hooks() -> None:
    """Install the compile hooks (idempotent; they report via _handle_jit_event)."""
    _setup_triton_jit_hook()
    _setup_cutedsl_jit_hook()

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


def _handle_jit_event(
    *,
    backend: str,
    event: str,
    fn_name: str,
    detail: str | None = None,
) -> None:
    if _recorded_events is not None:
        _recorded_events.append(
            f"{backend}: {fn_name}" + (f" ({detail})" if detail else "")
        )
        return
    if not _active:
        # Hooks installed by record_jit_events() before activate(): startup
        # compiles (warmup, CUDA graph capture) are expected.
        return
    message = (
        "%s %s during inference: %s%s. "
        "This causes a latency spike; consider extending warmup "
        "to cover this shape/config."
    )
    detail_suffix = f" ({detail})" if detail else ""
    args = (backend, event, fn_name, detail_suffix)
    # DIAGNOSTIC ONLY (INFRA-D jitdiag, never lands): where the post-warmup compile came from.
    import traceback

    frames = [
        f
        for f in traceback.extract_stack()[:-1]
        if "/vllm/" in f.filename and "jit_monitor" not in f.filename
    ]
    logger.error(
        "JIT_STACK %s %s%s :: %s",
        backend,
        fn_name,
        detail_suffix,
        " <- ".join(
            f"{f.filename.split('/vllm/')[-1]}:{f.lineno}:{f.name}"
            for f in reversed(frames[-8:])
        ),
    )

    if _mode == "error":
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


def _log_triton_jit_compile(fn_name: str, kwargs) -> None:
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
    )


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
        _log_triton_jit_compile(fn_name, kwargs)
        if existing_hook is not None:
            return existing_hook(**kwargs)
        return None

    knobs.runtime.jit_post_compile_hook = _on_jit_compile
    _triton_hook_installed = True


# ------------------------------------------------------------------
# CuTeDSL JIT compilation hook
# ------------------------------------------------------------------


def _log_cutedsl_jit_compile(fn_name: str) -> None:
    _handle_jit_event(
        backend="CuTeDSL",
        event="JIT compilation",
        fn_name=fn_name,
    )


class _MonitoredCuteCompile:
    """Logs JIT compilations; a plain function would break ``cute.compile[opts]``."""

    def __init__(self, inner):
        self._inner = inner

    def __getitem__(self, options) -> "_MonitoredCuteCompile":
        return _MonitoredCuteCompile(self._inner[options])

    def __call__(self, *args, **kwargs):
        kernel = args[0] if args else kwargs.get("function")
        kernel_name = getattr(kernel, "__name__", None)
        if kernel_name is None:
            kernel_name = (
                kernel.__class__.__name__ if kernel is not None else "<unknown>"
            )
        # DIAGNOSTIC ONLY (INFRA-D jitdiag): argument shapes / types of the compile.
        shapes = []
        for a in args[1:10]:
            try:
                s = getattr(a, "shape", None)
                shapes.append(_safe_repr(s if s is not None else type(a).__name__, max_len=48))
            except Exception:
                shapes.append("?")
        _handle_jit_event(
            backend="CuTeDSL",
            event="JIT compilation",
            fn_name=kernel_name,
            detail="args " + " | ".join(shapes),
        )
        return self._inner(*args, **kwargs)


def _setup_cutedsl_jit_hook() -> None:
    """Wrap ``cutlass.cute.compile`` to warn on compilation."""
    global _cutedsl_hook_installed
    if _cutedsl_hook_installed:
        return

    try:
        import cutlass.cute as cute
    except Exception:
        logger.debug("CuTeDSL is not available; skipping CuTeDSL JIT monitor.")
        return

    cute.compile = _MonitoredCuteCompile(cute.compile)
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
) -> None:
    _handle_jit_event(
        backend="TileLang",
        event="JIT compilation",
        fn_name=fn_name,
        detail=detail,
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
        if not from_database and _tilelang_jitimpl_compile_depth == 0:
            func = _tilelang_arg(args, kwargs, 0, "func")
            _log_tilelang_jit_compile(_tilelang_kernel_name(func))
        return original_init(self, *args, **kwargs)

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
                _log_tilelang_jit_compile(
                    _tilelang_kernel_name(orig_func or func), detail
                )
                return original_compile(self, *args, **kwargs)
            finally:
                _tilelang_jitimpl_compile_depth -= 1

        jit_impl_cls.compile = _compile_with_monitor

    _tilelang_hook_installed = True


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
