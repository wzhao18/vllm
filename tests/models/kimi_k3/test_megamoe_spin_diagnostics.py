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
        j = i
        while not lines[j].rstrip().endswith(":"):  # a header that spans lines ends with its own ':'
            j += 1
        body, j = [], j + 1
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


def test_one_printer_per_rank_and_site():
    """Trap lines from every spinning CTA flooded the printf FIFO and dropped the MISSING lines of one rank
    (injected-hang check, 2026-11-22): every dedup guard elects one printer through the per-rank flag word."""
    src = (PKG / "src_gp/sources/communication/nvlink_domain/token_comm_dedup.py").read_text()
    sites = re.findall(r"spin_guard(?:_value|_ranks)?\(\s*spin_start_ns, (2\d),", src)
    assert sorted(set(sites)) == [str(s) for s in range(20, 28)]
    calls = re.findall(r"spin_guard(?:_value|_ranks)?\(\s*spin_start_ns, 2\d,", src)
    assert len(calls) == src.count("print_flag=") == 8
    guard = (PKG / "src_gp/common/spin_guard.py").read_text()
    assert guard.count("if _elect_printer(site, print_flag):") == 3


# -- host-visible spin records (shared helper vllm/models/kimi_k3/nvidia/spin_record.py) ----------------------------
def test_spin_record_device_and_host_layout_agree():
    from vllm.models.kimi_k3.nvidia import spin_record as SR

    dev = (PKG / "src_gp/common/spin_guard.py").read_text()
    assert f"RECORD_WORDS = {SR.RECORD_WORDS}" in dev
    assert f"RECORD_DONE = 0x{SR.DONE:X}" in dev
    assert f"RECORD_SOURCE_MEGAMOE = {SR.SOURCE_MEGAMOE}" in dev
    assert "Int64(40)" in dev and "Int64(16)" in dev  # entries at word 10, 4 words each


def test_spin_record_format_names_missing_ranks():
    from vllm.models.kimi_k3.nvidia import spin_record as SR

    words = [0] * SR.RECORD_WORDS
    words[:10] = [SR.DONE, SR.SOURCE_MEGAMOE, 20, 0, 7, 3, 4, 188, 13, 2]
    words[10:18] = [1, 7, 6, 2, 5, 6, 6, 1]
    line = SR.format_record(words)
    assert line.startswith("K31_SPIN_RECORD source=megamoe_rubin site=20 rank=0 call=7 observed=3 target=4")
    assert "rank 1 issued 7 entered 6 H2_gpu_stuck_before_call" in line
    assert "rank 5 issued 6 entered 6 H1_host_never_issued" in line


def test_every_dedup_guard_writes_a_record():
    src = (PKG / "src_gp/sources/communication/nvlink_domain/token_comm_dedup.py").read_text()
    assert src.count("record=") == src.count("print_flag=") == 8
    session = (PKG / "session.py").read_text()
    assert "_write_spin_record_address()" in session


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="pinned memory needs CUDA")
def test_spin_record_watcher_logs_a_completed_record():
    from vllm.models.kimi_k3.nvidia import spin_record as SR

    rec = SR.SpinRecords(rank=3, poll_s=0.05)
    w = rec.buffer[: SR.RECORD_WORDS]
    w[1:10] = __import__("torch").tensor([SR.SOURCE_MEGAMOE, 21, 3, 9, 15, 16, 7, 0, 0], dtype=w.dtype)
    w[0] = SR.DONE
    import time

    for _ in range(100):
        if rec._logged[0]:
            break
        time.sleep(0.05)
    assert rec._logged[0]


def test_guards_delay_the_trap():
    """A trap from any thread kills the grid at once: an immediate trap of a non-elected thread could cut the elected
    printer's record writes short (PIDX 2026-11-22). Every guard traps through the delayed _trap."""
    src = (PKG / "src_gp/common/spin_guard.py").read_text()
    body = src[src.index("def _trap() -> None:") :]
    assert body.index("TRAP_DELAY_NS") < body.index("_trap_now()")
    assert src.count("_trap_now()") == 1  # only the delayed _trap() traps


def test_spin_record_format_pidx_record():
    """PIDX raw CUDA guards (k31_spin_guard.cuh) write record 1 with SOURCE_PIDX and the kernel name in words 48..63."""
    from vllm.models.kimi_k3.nvidia import spin_record as SR

    words = [0] * SR.RECORD_WORDS
    words[:10] = [SR.DONE, SR.SOURCE_PIDX, 438, 5, 1, 0, 1, 77, 1500, 0]
    name = b"attn_res_lp\0"
    name += b"\0" * (-len(name) % 4)
    for i in range(len(name) // 4):
        words[SR.NAME_WORD + i] = int.from_bytes(name[4 * i : 4 * i + 4], "little")
    assert SR.format_record(words) == (
        "K31_SPIN_RECORD source=pidx kernel=attn_res_lp site=438 rank=5 call=1 size=1500 "
        "observed=0 target=1 smid=77"
    )
