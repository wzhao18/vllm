# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Spin-guard ordering: every non-MegaMoE bounded wait must time out before MegaMoE's.

A rank stuck outside MegaMoE leaves its peers waiting inside MegaMoE. If the peers' MegaMoE
guard fired first, the healthy ranks would trap and tear the job down while the stuck rank
never reports. Each other guard family must therefore have a strictly shorter bound, so the
stuck rank names its own wait first. CPU only: the bounds are read from the sources."""

import ast
import os
import re

import pytest

_NV = os.path.join(os.path.dirname(__file__), "..", "..", "..", "vllm", "models", "kimi_k3", "nvidia")


def _read(rel: str) -> str:
    with open(os.path.join(_NV, rel)) as f:
        return f.read()


def _py_const(rel: str, name: str) -> float:
    for node in ast.parse(_read(rel)).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return float(eval(compile(ast.Expression(node.value), rel, "eval"), {}))  # literal arithmetic only
    raise AssertionError(f"{name} not found in {rel}")


def _megamoe_ns() -> list[float]:
    return [
        _py_const("megamoe_rubin/src/sources/helpers/spin_guard.py", "SPIN_GUARD_NS"),
        _py_const("megamoe_rubin/src_gp/common/spin_guard.py", "SPIN_GUARD_NS"),
    ]


def _cuh_ns() -> float:
    m = re.search(r"#define K31_SPIN_GUARD_NS (\d+)ull", _read("k31_spin_guard.cuh"))
    assert m, "K31_SPIN_GUARD_NS default not found"
    return float(m.group(1))


def _dsa_core_default_ns() -> float:
    m = re.search(r"spin_s: float = ([\d.]+)", _read("dsa_kernels/dsa_core/ops.py"))
    assert m, "dsa_core spin_s default not found"
    return float(m.group(1)) * 1e9


def _dcp_exchange_default_ns() -> float:
    m = re.search(r"kSpinNsDefault = ([\d\sLl*]+);", _read("dsa_kernels/dcp_exchange/csrc/dcp_exchange.cu"))
    assert m, "kSpinNsDefault not found"
    return float(eval(m.group(1).replace("LL", "").replace("ll", "")))


_OTHER = {
    "k31_spin_guard.py (CuTe DSL waits)": lambda: _py_const("k31_spin_guard.py", "SPIN_GUARD_NS"),
    "k31_spin_guard.cuh (CUDA waits: FlashKDA, PIDX, KDA)": _cuh_ns,
    "dcp_exchange SERVING_SPIN_S": lambda: _py_const("dsa_kernels/dcp_exchange/decode.py", "SERVING_SPIN_S") * 1e9,
    "dcp_exchange startup default": _dcp_exchange_default_ns,
    "dsa_core SERVING_SPIN_S": lambda: _py_const("dsa_kernels/dsa_core/decode.py", "SERVING_SPIN_S") * 1e9,
    "dsa_core startup default": _dsa_core_default_ns,
}


def test_megamoe_guard_copies_agree():
    a, b = _megamoe_ns()
    assert a == b


@pytest.mark.parametrize("family", sorted(_OTHER))
def test_guard_fires_before_megamoe(family):
    bound = _OTHER[family]()
    assert 0 < bound < min(_megamoe_ns()), f"{family}: {bound / 1e9:g} s is not below MegaMoE's guard"
