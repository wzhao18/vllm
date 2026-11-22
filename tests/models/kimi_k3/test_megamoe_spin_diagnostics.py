# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rubin MegaMoE hang diagnostics (MEGAMOEFIX 2026-11-22, VENDOR.md S9). CPU only.

1. The missing-rank verdict of ``spin_guard_ranks`` (host never issued / GPU never started / inside the call),
   mirrored in Python and checked on the three ways a rank can fail to publish.
2. Source structure of the dedup router: the entry stamp precedes every wait, every arrival is OR-ed into the
   peer's mask before it is counted, and the sizes wait reports missing ranks.
3. Inventory: every load-spin on the dedup kernel path (router, token back, schedulers, generic spin_wait, the
   polled reduce) is bounded by a spin guard.
4. The host stamp skips CUDA graph capture.
"""

import inspect
import pathlib
import re

import pytest

PKG = pathlib.Path(__file__).resolve().parents[3] / "vllm/models/kimi_k3/nvidia/megamoe_rubin"


def _verdict(issued: int, entered: int, expected: int) -> str:
    # mirror of common/spin_guard.spin_guard_ranks
    if issued < expected:
        return "H1_host_never_issued"
    if entered < expected:
        return "H2_gpu_stuck_before_call"
    return "IN_call_not_published"


@pytest.mark.parametrize(
    "issued,entered,expected,verdict",
    [
        (6, 6, 7, "H1_host_never_issued"),  # host stalled before issuing call 7 (or skipped a launch)
        (7, 6, 7, "H2_gpu_stuck_before_call"),  # issued, but its stream is stuck in an earlier kernel
        (7, 7, 7, "IN_call_not_published"),  # entered, stuck inside the router before publishing
        (8, 7, 7, "IN_call_not_published"),  # host already ahead: still inside call 7
    ],
)
def test_missing_rank_verdict(issued, entered, expected, verdict):
    assert _verdict(issued, entered, expected) == verdict


def test_verdict_strings_match_kernel_source():
    src = (PKG / "src_gp/common/spin_guard.py").read_text()
    for v in ("H1_host_never_issued", "H2_gpu_stuck_before_call", "IN_call_not_published"):
        assert f"verdict={v}" in src
    assert (PKG / "src/sources/helpers/spin_guard.py").read_text() == src  # the two trees stay identical


cutlass = pytest.importorskip("cutlass")


def _dedup():
    from vllm.models.kimi_k3.nvidia.megamoe_rubin.src_gp.sources.communication.nvlink_domain import (
        token_comm_dedup as D,
    )

    return D


def test_router_stamps_entry_before_any_wait():
    D = _dedup()
    src = inspect.getsource(D._DedupMetadataPushRouter._router_kernel)
    stamp = src.index("progress_region")
    assert stamp < src.index("wait_previous_done")
    for call in ("_router_single_cta(", "_router_multiple_ctas(", "_router_push_metadata("):
        assert stamp < src.index(call)


def test_arrivals_are_masked_before_counted():
    D = _dedup()
    sizes = inspect.getsource(D._DedupMetadataPushRouter._broadcast_sizes_to_peers)
    assert sizes.index("sizes_mask_region") < sizes.index("red_or_relaxed_sys_b32") < sizes.index("red_add_relaxed_sys_s32")
    meta = inspect.getsource(D._DedupMetadataPushRouter._router_push_metadata)
    assert meta.index("red_or_relaxed_sys_b32") < meta.index("red_add_relaxed_sys_s32")
    assert "spin_guard_ranks(" in inspect.getsource(D._DedupMetadataPushRouter)


def test_diagnostic_regions_registered_and_masks_reset_per_call():
    D = _dedup()
    src = inspect.getsource(D.DedupCallTail.register_device_workspace)
    assert re.search(r"progress_region.*zero_on_first_allocate", src, re.S)
    assert src.count('reset="tail_reset"') == 2  # the two arrival masks are per call


_SPIN_FILES = [
    "src_gp/sources/communication/nvlink_domain/token_comm_dedup.py",
    "src_gp/sources/helpers/dsl_helpers.py",
    "src_gp/sources/helpers/topk_reduce.py",
    "src_gp/sources/kernel_src/schedulers/work_id_claim.py",
    "src_gp/sources/helpers/software_sync.py",
    "src_gp/sources/kernel_src/rubin/inference/mega/genphase_tail.py",
]


@pytest.mark.parametrize("rel", _SPIN_FILES)
def test_every_load_spin_is_guarded(rel):
    lines = (PKG / rel).read_text().split("\n")
    unguarded = []
    for i, line in enumerate(lines):
        if not re.match(r"\s*while\b", line):
            continue
        indent = len(line) - len(line.lstrip())
        body, j = [], i + 1
        while j < len(lines) and (not lines[j].strip() or len(lines[j]) - len(lines[j].lstrip()) > indent):
            body.append(lines[j])
            j += 1
        text = line + "\n".join(body)
        waits = "nanosleep(" in text or re.search(r"load\(", line)
        if waits and "spin_guard" not in text:
            unguarded.append(f"{rel}:{i + 1}: {line.strip()}")
    assert not unguarded, unguarded


def test_host_stamp_skips_capture():
    from vllm.models.kimi_k3.nvidia.megamoe_rubin import session as S

    src = inspect.getsource(S.RubinMegaMoeSession._stamp_issued)
    assert src.index("is_current_stream_capturing") < src.index("copy_(")
    launch = inspect.getsource(S.RubinMegaMoeSession.launch)
    assert launch.index("_stamp_issued()") < launch.index("self.compiled[")
