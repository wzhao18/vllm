# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized decode's RecoverSSM (``--use-replayssm`` KDA verify records).

Ported from the upstream specialized decode: ``context`` (the ``fp32_deferred``
commit context), ``kernels`` (the Triton plans and the record-slot verify) and
``state`` (how the deferred commit installs on the stock layers, runner state and
KDA metadata builders).

``prepare_recoverssm`` is the SP decode's entry point (C512, ``runtime.sp_attention``):
it decides, once and before any provider is built, how the specialized KDA core
(``cutedsl.kda.sp.SpKdaCore``) writes its verify records and who folds them:

==========  ==================================================================
mode        records / commit
==========  ==================================================================
"deferred"  per request slot (``install_layers``); the commit after sampling
            only stages each slot's accepted count (``pending``), the next
            verify folds it (``KDARecoverSSMDeferred.flush``, or a replaying
            core). Needs the runner-state side, ``bind_model_state``, which a
            stock hook (``KimiK3ModelState``) must call.
"triton"    per step row (the stock layout); the stock Triton commit after
            sampling folds them, unchanged (``defer=False``).
"none"      the KDA layers keep no records (no ``--use-replayssm``): plain
            verify with q state slots.
==========  ==================================================================

The stock CuTe RecoverSSM backend ("cute": SM10x with a BF16 SSM state) is not served:
the SP decode runs an FP32 KDA state (its records come from the Triton verify math).
"""

from typing import Any

MODES = ("deferred", "triton", "none")


def _kda_attention(layer) -> Any | None:
    """The KDA attention module of a decoder layer (or the module itself)."""
    attention = getattr(layer, "self_attn", layer)
    return attention if hasattr(attention, "in_proj_qkvgfab") else None


def prepare_recoverssm(layers, ctx, *, defer: bool = True) -> str:
    """The RecoverSSM records mode of the SP decode's KDA layers (module docstring).

    ``layers``: the rank's decoder layers (KDA attention modules are accepted too);
    ``ctx``: the ``shapes.Context``. ``defer`` (default) installs the deferred commit
    on the KDA layers (``state.install_layers``) and returns "deferred"; False keeps
    the stock layout and commit ("triton"). Raises ``shapes.Unsupported`` for a
    deployment the specialized KDA core cannot serve with stock precision."""
    from ..shapes import Unsupported

    attentions = [a for a in map(_kda_attention, layers) if a is not None]
    if not attentions:
        raise Unsupported("kda", None, "no KDA layer on this rank")
    with_records = [
        getattr(a, "recoverssm_records", None) is not None for a in attentions
    ]
    if not any(with_records):
        return "none"
    if not all(with_records):
        raise Unsupported(
            "kda", None, "RecoverSSM records on some KDA layers only (mixed verify)"
        )
    backends = {getattr(a, "recoverssm_backend", None) for a in attentions}
    if backends != {"triton"}:
        raise Unsupported(
            "kda",
            None,
            f"RecoverSSM backend {sorted(map(str, backends))}: the SP KDA core serves "
            "the FP32-state Triton RecoverSSM only (the stock CuTe backend runs a "
            "BF16 state)",
        )
    if ctx.bf16_state:
        raise Unsupported("kda", None, "RecoverSSM records need an FP32 SSM state")
    if any(a.gate_lower_bound is None for a in attentions):
        raise Unsupported(
            "kda", None, "RecoverSSM cores need a bounded KDA gate (gate_lower_bound)"
        )
    if any(a.conv1d.bias is not None for a in attentions):
        raise Unsupported("kda", None, "RecoverSSM cores do not apply a conv bias")
    if not defer:
        return "triton"
    if ctx.arch not in ("rubin", "gb300"):
        raise Unsupported(
            "kda", None, f"the deferred FP32 commit needs SM10x (arch {ctx.arch})"
        )
    from .state import install_layers

    install_layers(layers)
    missing = [a.prefix for a in attentions if not getattr(a, "deferred_recoverssm", 0)]
    if missing:
        raise Unsupported(
            "kda", None, f"deferred RecoverSSM not installed on {missing[:4]}"
        )
    return "deferred"


def bind_model_state(state) -> None:
    """The runner-state side of the "deferred" mode, called once on the
    ``KimiK3ModelState`` after the specialized decode is built: per-request pending
    replay counts and the KDA metadata builders' deferred build
    (``state.install_model_state``). Without it every deferred verify raises
    (``SpKdaCore.run``): the stock commit would fold records the deferred layout
    does not hold per step row."""
    from .state import install_model_state

    install_model_state(state)


__all__ = ["MODES", "bind_model_state", "prepare_recoverssm"]
