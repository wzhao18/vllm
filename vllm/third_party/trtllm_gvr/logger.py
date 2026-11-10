# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stand-in for ``tensorrt_llm.logger`` used by the vendored kernels."""

from vllm.logger import init_logger

_logger = init_logger("vllm.third_party.trtllm_gvr")


class _Logger:
    """Forwards ``logger.<level>(msg)`` and ``logger.<level>_once(msg, key=...)``."""

    def __getattr__(self, name: str):
        once = name.endswith("_once")
        level = name[: -len("_once")] if once else name
        if level == "warn":
            level = "warning"
        if once:
            method = getattr(_logger, f"{level}_once", _logger.info_once)
            return lambda msg, *args, key=None, **kwargs: method(msg)
        method = getattr(_logger, level, _logger.info)
        return lambda msg, *args, **kwargs: method(msg)


logger = _Logger()
