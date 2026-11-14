# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from .cute_cache import install as _install_cute_cache

# Kimi K3 CuTe DSL kernels: persistent pre-trace compile cache (cute_cache.py).
_install_cute_cache()
