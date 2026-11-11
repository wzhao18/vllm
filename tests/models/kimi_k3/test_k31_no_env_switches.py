# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Kimi K3 NVIDIA code selects its paths in code, not through environment
switches: no source under vllm/models/kimi_k3/nvidia reads an environment variable
(``envs.X``, ``os.environ`` / ``os.getenv`` / C ``getenv``) other than the
infrastructure ones (extension build directories, CUDA_HOME) and stock (upstream)
vLLM settings it honours with their stock meaning; and the role policies that replaced
the recipes' per-role switches (CPU only)."""

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3] / "vllm" / "models" / "kimi_k3" / "nvidia"
READ = re.compile(
    r"os\.environ\.get\(\s*['\"]([A-Za-z0-9_]+)|os\.environ\[\s*['\"]([A-Za-z0-9_]+)['\"]\s*\]"
    r"(?!\s*=[^=])|getenv\(\s*['\"]([A-Za-z0-9_]+)|\benvs\.([A-Z][A-Z0-9_]+)"
    r"|getattr\(\s*envs\s*,\s*['\"]([A-Za-z0-9_]+)"
)
ALLOWED = {
    # infrastructure: where extension builds go, where CUDA is
    "K31_DCPX_BUILD_DIR",
    "K31_DSAC_BUILD_DIR",
    "CUDA_HOME",
    # stock vLLM settings, honoured with their stock meaning (not K3 feature switches)
    "VLLM_CACHE_ROOT",
    "VLLM_BATCH_INVARIANT",
    "VLLM_MOE_SKIP_PADDING",
    "VLLM_USE_FUSED_MOE_GROUPED_TOPK",
    "VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD",
    "VLLM_DISABLE_SHARED_EXPERTS_STREAM",
    # stock vllm/envs.py opt-in (default off) of the stock sparse_attn_indexer's
    # histogram DCP prefill top-k merge, whose kernels live under dsa_kernels/
    # dcp_prefill_topk_merge; the K3 indexer mirrors it (stock behaviour), no recipe
    # sets it
    "VLLM_DSA_DCP_PREFILL_HIST_MERGE",
    # upstream vLLM Kimi K3 options (stock code: #50656, #50487), off by default and
    # set by no serving recipe
    "VLLM_KIMI_K3_SHARD_SP_SHARED_EXPERT",
    "VLLM_KIMI_K3_AUX_ATTN_RES_STREAM",
}
# Byte-identical vendored kernel sources (megamoe_rubin/VENDOR.md); their diagnostic
# variables (MEGA_USE_NCU, MEGA_F4CVT_USE_MANUAL) are never set by vLLM.
VENDORED = ROOT / "megamoe_rubin" / "src"


def test_no_env_switches():
    reads = {}
    scanned = 0
    for path in sorted(ROOT.rglob("*")):
        if path.suffix not in (".py", ".cu", ".cuh", ".h", ".cpp"):
            continue
        if path.is_relative_to(VENDORED):
            continue
        # macOS AppleDouble metadata ("._x.py" next to x.py on copied trees) is not
        # source; any other file is scanned even when not valid UTF-8 (the patterns
        # are ASCII), so one stray byte cannot turn the lint off.
        if path.name.startswith("._") or not path.is_file():
            continue
        scanned += 1
        text = path.read_bytes().decode("utf-8", errors="replace")
        for match in READ.finditer(text):
            name = next(g for g in match.groups() if g)
            reads.setdefault(name, set()).add(str(path.relative_to(ROOT)))
    assert scanned > 100, f"only {scanned} sources under {ROOT}"
    switches = {k: sorted(v) for k, v in reads.items() if k not in ALLOWED}
    assert not switches, switches


def _config(kv_role=None, moe_backend="deep_gemm_mega_moe", tp=8, dp=1):
    kv_transfer = None
    if kv_role is not None:
        kv_transfer = SimpleNamespace(
            is_kv_producer=kv_role in ("kv_producer", "kv_both"),
            is_kv_consumer=kv_role in ("kv_consumer", "kv_both"),
        )
    return SimpleNamespace(
        kv_transfer_config=kv_transfer,
        kernel_config=SimpleNamespace(moe_backend=moe_backend),
        parallel_config=SimpleNamespace(
            enable_expert_parallel=True,
            pipeline_parallel_size=1,
            data_parallel_size=dp,
            tensor_parallel_size=tp,
        ),
    )


@pytest.mark.parametrize(
    "kv_role,expected",
    [
        ("kv_both", True),  # the P/D prefill workers of the recipes
        ("kv_producer", True),
        ("kv_consumer", False),  # decode keeps plain TP / SP (C512's SP topology)
        (None, False),  # unified serving: as before (off)
    ],
)
def test_moe_token_shard_role(kv_role, expected):
    from vllm.models.kimi_k3.nvidia.model import use_moe_token_shard

    assert use_moe_token_shard(_config(kv_role)) is expected
    assert use_moe_token_shard(_config(kv_role, tp=1)) is False
    assert use_moe_token_shard(_config(kv_role, dp=4)) is False
    assert use_moe_token_shard(_config(kv_role, moe_backend="auto")) is False


@pytest.mark.parametrize(
    "kv_role,expected",
    [
        ("kv_both", True),
        ("kv_producer", True),
        ("kv_consumer", False),  # a decode-only instance prefills only to recompute
        (None, True),
    ],
)
def test_kda_segscan_role(kv_role, expected):
    from vllm.models.kimi_k3.nvidia.ops import kda_segscan

    assert kda_segscan.wanted(_config(kv_role)) is expected
