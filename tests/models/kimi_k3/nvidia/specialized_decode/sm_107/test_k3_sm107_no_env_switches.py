# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Ported from the source branch's tests/models/kimi_k3/test_k31_no_env_switches.py for
# the SM107 specialized decode (vllm/models/kimi_k3/nvidia/specialized_decode/sm_107).
"""The SM107 specialized decode selects its paths in code, not through environment
switches: no source under specialized_decode/sm_107 reads an environment variable
(``envs.X``, ``os.environ`` / ``os.getenv`` / C ``getenv``) other than the
infrastructure ones (extension build directory, CUDA location, vLLM's cache root) and
stock (upstream) vLLM settings it honours with their stock meaning (CPU only)."""

import re
from pathlib import Path

ROOT = (
    Path(__file__).resolve().parents[6]
    / "vllm"
    / "models"
    / "kimi_k3"
    / "nvidia"
    / "specialized_decode"
    / "sm_107"
)
READ = re.compile(
    r"os\.environ\.get\(\s*['\"]([A-Za-z0-9_]+)|os\.environ\[\s*['\"]([A-Za-z0-9_]+)['\"]\s*\]"
    r"(?!\s*=[^=])|getenv\(\s*['\"]([A-Za-z0-9_]+)|\benvs\.([A-Z][A-Z0-9_]+)"
    r"|getattr\(\s*envs\s*,\s*['\"]([A-Za-z0-9_]+)"
)
ALLOWED = {
    # infrastructure: where the DSA core's extension build goes, where CUDA is
    "K31_DSAC_BUILD_DIR",
    "CUDA_HOME",
    "CUDA_PATH",
    # stock vLLM settings, honoured with their stock meaning (not K3 feature switches)
    "VLLM_CACHE_ROOT",
    "VLLM_MOE_SKIP_PADDING",
    "VLLM_ENABLE_GEMM_RS",
    "VLLM_USE_FUSED_MOE_GROUPED_TOPK",
    # upstream vLLM Kimi K3 option (stock code: #50487), off by default and set by no
    # serving recipe
    "VLLM_KIMI_K3_AUX_ATTN_RES_STREAM",
}


def test_no_env_switches():
    assert ROOT.is_dir(), ROOT
    reads = {}
    for path in sorted(ROOT.rglob("*")):
        if path.suffix not in (".py", ".cu", ".cuh", ".h", ".cpp"):
            continue
        for match in READ.finditer(path.read_text()):
            name = next(g for g in match.groups() if g)
            reads.setdefault(name, set()).add(str(path.relative_to(ROOT)))
    switches = {k: sorted(v) for k, v in reads.items() if k not in ALLOWED}
    assert not switches, switches


def test_removed_switches_are_not_read(monkeypatch):
    """The source branch's removed switches change nothing here: the DSA sparse
    core's policy and tables are constants (dsa_core.decode), and its extension has
    no fault or debug build switches."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.sm_107.dsa_kernels.dsa_core import (  # noqa: E501
        decode,
        ops,
    )

    for name in (
        "VLLM_K31_DSA_CORE",
        "VLLM_K31_DSA_CORE_ROWS",
        "VLLM_K31_DSA_CORE_VARIANT",
        "VLLM_K31_DSA_CORE_SKIP_LAYERS",
        "VLLM_K31_DSA_CORE_SPREAD",
        "VLLM_K31_DSA_CORE_SPIN_S",
        "VLLM_K31_DSA_CORE_STATS",
        "K31_DSAC_FAULT",
    ):
        monkeypatch.setenv(name, "0")
    for name in ("ENV", "ROWS_ENV", "VARIANT_ENV", "SKIP_ENV", "SPREAD_ENV"):
        assert not hasattr(decode, name), name
    for name in ("STATS_ENV", "SPIN_ENV", "CountStats", "count_stats", "wanted"):
        assert not hasattr(decode, name), name
    assert not hasattr(ops, "FAULT")
    assert (decode.ROWS, decode.VARIANT, decode.SERVING_SPIN_S) == (
        16,
        {8: "1-16:2x16s2"},
        600.0,
    )
