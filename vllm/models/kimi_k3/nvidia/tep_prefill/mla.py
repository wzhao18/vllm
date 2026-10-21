# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The TEP prefill's MLA paths of one layer, built on first use (after the weights
are loaded and processed)."""


class TepMla:
    def __init__(self):
        self._context_kv = None

    def context_kv(self, module):
        """The fused context pack (``context_kv.ContextKv``) of ``module``."""
        if self._context_kv is None:
            from .context_kv import ContextKv

            self._context_kv = ContextKv(module)
        return self._context_kv

    @staticmethod
    def absorbed(module):
        """The absorbed continuation prefill (shared by every MLA layer)."""
        from .absorbed import AbsorbedPrefill

        return AbsorbedPrefill.get(module)
