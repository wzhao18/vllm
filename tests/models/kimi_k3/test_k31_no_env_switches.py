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
    # the CuTe DSL's own target-architecture variable (the DSL reads it at start):
    # cute_cache reads it only so the compile-cache key names the arch the DSL
    # compiles for; it selects no K3 path
    "CUTE_DSL_ARCH",
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
    # Mi Sun Park 1408a90317: K31_SPIN_GUARD=0 traces the DSL poll loops without
    # the bounded-spin hang detector (same polls, order and values; part of the
    # cute_cache key). Adopted by the user on 2026-11-24 for every K3.1 decode
    # measurement and checkpoint. It selects no compute path, only whether a hang
    # traps. Open: replace it with a code policy (e.g. by role) and drop this entry.
    "K31_SPIN_GUARD",
    # Mi Sun Park, 214b7620ce; owner decides code policy; prefill does not use it.
    # K31CH_FRONT_TMEM appends row ranges to the TP16 MoeFrontTmem64 decode front
    # table (specialized_decode/cutedsl/moe/front.py); unset keeps the stock table.
    "K31CH_FRONT_TMEM",
    # Mi Sun Park; owner decides code policy; prefill does not use it.
    # K31EX_W2_DENSE=1 gives the TP16 routed experts kernels dense w2 rows
    # (specialized_decode/cutedsl/moe/experts_runtime.py, dense_w2); unset keeps
    # the padded parameter and the stock path.
    "K31EX_W2_DENSE",
    # Mi Sun Park; owner decides code policy; prefill does not use it.
    # K31PL_FOLD_ROWS appends row ranges to the TP16 KDA_FOLD_PAD decode table
    # (specialized_decode/runtime/providers.py): fewer than 8 requests run the
    # 8-request KDA fold with padding requests; unset keeps the in-projection ->
    # KdaSplit pair.
    "K31PL_FOLD_ROWS",
    # wzhao/k31-ep16-fused-decode; owner decides code policy. The EP16 fused
    # MegaMoE + shared-expert decode bucket is on by default;
    # VLLM_K31_EP_FUSED_DECODE=0 restores the unfused path (megamoe_rubin/experts.py).
    "VLLM_K31_EP_FUSED_DECODE",
    # wzhao/k31-pf-p3-expand-kdaconv; owner decides code policy; decode does not use
    # them. K31_COMPACT_SCALES_AUTO_PREFILL=1 restores the earlier auto (compact NVFP4
    # scales on every P worker; megamoe_rubin/scale_staging.py). K31_KDA_CONV_DEFER=0
    # captures the stock in_proj instead of deferring the fused KDA conv in_proj to
    # the eager break; K31_KDA_CONV_ONE_SHOT_CAPTURE=1 is measurement only (vllm
    # bench forward) (specialized_prefill/kda_conv_in_proj.py).
    "K31_COMPACT_SCALES_AUTO_PREFILL",
    "K31_KDA_CONV_DEFER",
    "K31_KDA_CONV_ONE_SHOT_CAPTURE",
    # wzhao/k31-dsa-chain-forks; owner decides code policy; prefill does not use them;
    # unset keeps the stock tables and paths. K31_SP_LOCAL_QUERY replaces the SP
    # local-query row table (specialized_decode/cutedsl/dsa/sp.py);
    # K31_DSA_EARLY_FORK=1 forks the query beside the index chain on layers without
    # the #63 merge; K31_DSA_LOCAL_LATE is a rows table launching the local query
    # after wq_b + DsaIndex (specialized_decode/cutedsl/dsa/prepared.py).
    "K31_SP_LOCAL_QUERY",
    "K31_DSA_EARLY_FORK",
    "K31_DSA_LOCAL_LATE",
    # Mi Sun Park; owner decides code policy; prefill does not use it.
    # K31PL_LOOP4 appends request rows to the TP16 q8 FP32_LOOP verify table
    # (specialized_decode/cutedsl/kda/prepared.py): those requests run FP32_WIDE
    # (the record protocol's LoopDefer4); unset keeps the table's rows.
    "K31PL_LOOP4",
    # Mi Sun Park; owner decides code policy; prefill does not use it.
    # K31PL_SHARED_EARLY appends rows to SHARED_EARLY_ROWS
    # (specialized_decode/cutedsl/moe/front_rows.py): e.g. "97-128:3" pushes those
    # rows' shared RS right behind the early down rows; unset keeps the table.
    "K31PL_SHARED_EARLY",
    # Mi Sun Park; owner decides code policy; prefill does not use it.
    # K31DSC_CORE_SPREAD appends rows to CORE_SPREAD
    # (specialized_decode/cutedsl/dsa/prepared.py): those DSA sparse core rows push
    # into the spread DcpValue merge (LSE box [CAP, H, DCP]); unset keeps the table.
    "K31DSC_CORE_SPREAD",
}
# Vendored FlashInfer MegaMoE kernel sources (megamoe_rubin/VENDOR.md): ``src``
# (a368c5f66) and ``src_gp`` (FlashInfer main 855eece4e9). Their env reads are
# upstream's, byte for byte (diagnostics MEGA_USE_NCU, MEGA_F4CVT_USE_MANUAL,
# MEGA_DIAG_SKIP_TOKEN_BACK, MEGA_DEDUP_WAIT_SLEEP[_MAX]_NS and the DSL's
# CUTE_DSL_ARCH) and are never set by vLLM.
VENDORED = (ROOT / "megamoe_rubin" / "src", ROOT / "megamoe_rubin" / "src_gp")


def test_no_env_switches():
    reads = {}
    scanned = 0
    for path in sorted(ROOT.rglob("*")):
        if path.suffix not in (".py", ".cu", ".cuh", ".h", ".cpp"):
            continue
        if any(path.is_relative_to(v) for v in VENDORED):
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
