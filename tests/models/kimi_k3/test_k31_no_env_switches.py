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


@pytest.mark.parametrize("kv_role", ["kv_both", "kv_producer", "kv_consumer", None])
@pytest.mark.parametrize("moe_backend", ["deep_gemm_mega_moe", "kimi_k3_mega_moe"])
def test_sequence_parallel_every_role(kv_role, moe_backend):
    """Sequence parallelism is a property of the layout, not of the role: the P/D
    prefill workers (kv_both / kv_producer: TP8+DCP8 EP8 MegaMoE) run SP + EP like
    decode and unified serving (no MoE token shard)."""
    from vllm.models.kimi_k3.nvidia.model import use_sequence_parallel

    assert use_sequence_parallel(_config(kv_role, moe_backend=moe_backend))
    assert use_sequence_parallel(_config(kv_role, moe_backend=moe_backend, dp=4))
    assert not use_sequence_parallel(_config(kv_role, moe_backend=moe_backend, tp=1))
    # Non-MegaMoE EP: SP only with DP > 1.
    assert not use_sequence_parallel(_config(kv_role, moe_backend="auto"))
    assert use_sequence_parallel(_config(kv_role, moe_backend="auto", dp=4))


def test_no_moe_token_shard():
    """The MoE token shard (GEMM-AR over a replicated stream) is gone: no role
    policy, no env switch."""
    from vllm import envs
    from vllm.models.kimi_k3.nvidia import model

    assert not hasattr(model, "use_moe_token_shard")
    assert not any("MOE_TOKEN_SHARD" in name for name in envs.environment_variables)


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


@pytest.mark.parametrize(
    "kv_role,expected",
    [
        ("kv_both", True),  # P/D prefill: full grid + stream-priority overlap
        ("kv_producer", True),
        ("kv_consumer", False),  # decode: SM-capped side-stream overlap (unchanged)
        (None, False),  # unified serving: unchanged
    ],
)
def test_prefill_full_grid_shared_overlap_by_role(kv_role, expected):
    """The shared-expert overlap form of the sequence-parallel MegaMoE layers is decided
    by role (no switch): P/D prefill workers keep the full MegaMoE grid
    (KimiMoE._overlapped_forward's priority form), every other role the 148-SM cap."""
    from vllm.models.kimi_k3.nvidia.megamoe_rubin import experts
    from vllm.models.kimi_k3.nvidia.model import prefill_full_grid_shared_overlap

    cfg = _config(kv_role)
    cfg.additional_config = {}
    assert prefill_full_grid_shared_overlap(cfg) is expected
    assert experts.RUBIN_SHARED_OVERLAP_MAX_SMS == 148
