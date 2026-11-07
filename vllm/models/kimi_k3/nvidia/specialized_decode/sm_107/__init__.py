# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 specialized decode for SM107 (Rubin / VR200), selected by
``additional_config={"decode_backend": "latency_specialized"}`` on a compute
capability 10.7 GPU.

Every other device (SM100 / SM103) and ``decode_backend=auto`` keep the package's
own specialized decode (``..runtime``) unchanged: ``specialized_decode_config``
returns its config. Import-light: nothing here imports CuTe DSL.
"""


def specialized_decode_config(vllm_config):
    """The deployment's specialized decode config, None for the stock decode.

    ``latency_specialized`` on SM107: the SM107 decode's ``Sm107DecodeConfig``
    (raises for a deployment it does not serve; there is no fallback to the other
    specialized decode or to stock). Otherwise the package's own
    ``runtime.specialized_decode_config``."""
    extra = vllm_config.additional_config
    backend = extra.get("decode_backend", "auto") if isinstance(extra, dict) else "auto"
    if backend == "latency_specialized" and _is_sm107():
        from .integration import sm107_decode_config

        return sm107_decode_config(vllm_config)
    from ..runtime import specialized_decode_config as package_config

    return package_config(vllm_config)


def _is_sm107() -> bool:
    from vllm.platforms import current_platform

    return current_platform.is_cuda() and current_platform.is_device_capability(
        (10, 7)
    )
