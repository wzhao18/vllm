# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer's persistent dense tcgen05 GEMM with its epilogue op on the accumulator.

FlashInfer's ``PersistentDenseGemmKernel`` (flashinfer/gemm/kernels/
bmm_fp8_blackwell.py) applies the epilogue op after converting the Float32
accumulator to the output type. The context pack needs it before: it rounds the
accumulator to BF16 (the stock ``kv_b_proj`` output) and only then casts to FP8, so
an FP8 output matches the stock path bit for bit.

That is a one-line change in two places of the kernel's epilogue
(flashinfer/gemm/kernels/epilogue_utils.py). Instead of vendoring both modules,
this derives them from the installed FlashInfer: the patched sources are written
once under ``VLLM_CACHE_ROOT`` (CuTe DSL compiles from source files) and imported
from there. The patch points are asserted, so a FlashInfer that no longer matches
fails here, at import, rather than computing anything differently.
"""

import hashlib
import importlib.util
import inspect
import os
import sys
from pathlib import Path

from flashinfer.gemm.kernels import bmm_fp8_blackwell, epilogue_utils

from vllm import envs

_EPILOGUE_OP = (
    "acc_vec = epilogue_op(acc_vec.to(gemm_kernel.c_dtype))",
    "acc_vec = epilogue_op(acc_vec).to(gemm_kernel.c_dtype)",
)
_EPILOGUE_IMPORT = "from .epilogue_utils import ("


def _replace(source: str, old: str, new: str, count: int) -> str:
    found = source.count(old)
    if found != count:
        raise ImportError(
            f"FlashInfer GEMM source changed: expected {count} x {old!r}, found {found}"
        )
    return source.replace(old, new)


def _load(name: str, source: str):
    """Import ``source`` as a module backed by a file (CuTe DSL needs the source)."""
    digest = hashlib.sha256(source.encode()).hexdigest()[:16]
    module_name = f"vllm_tep_prefill_{name}_{digest}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    root = Path(envs.VLLM_CACHE_ROOT) / "tep_prefill"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{module_name}.py"
    if not path.exists():
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(source)
        os.replace(tmp, path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _derive():
    epilogue = _load(
        "epilogue",
        _replace(inspect.getsource(epilogue_utils), *_EPILOGUE_OP, count=2),
    )
    gemm = _load(
        "gemm",
        _replace(
            inspect.getsource(bmm_fp8_blackwell),
            _EPILOGUE_IMPORT,
            f"from {epilogue.__name__} import (",
            count=1,
        ),
    )
    return gemm.PersistentDenseGemmKernel


PersistentDenseGemmKernel = _derive()
