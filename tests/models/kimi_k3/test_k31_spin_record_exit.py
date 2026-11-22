# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""spin_record.py: the watcher reads records without torch and is stopped at exit.

A daemon watcher inside a torch call during interpreter teardown aborted processes ("terminate called without an
active exception"); every MegaMoE process creates this watcher (session carve)."""

import signal
import subprocess
import sys
import textwrap
import time

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU (pinned host memory)")


@cuda
def test_completed_record_is_read_through_ctypes_and_logged_once():
    from vllm.models.kimi_k3.nvidia import spin_record as SR

    rec = SR.SpinRecords(rank=3, poll_s=60.0)  # its own instance; the watcher stays asleep
    try:
        words = [0] * SR.RECORD_WORDS
        words[:9] = [SR.DONE, SR.SOURCE_MEGAMOE, 20, 3, 77, 15, 16, 188, 0xFFFB]
        rec.buffer[: SR.RECORD_WORDS] = torch.tensor(words, dtype=torch.int32)
        assert rec._words(0) == rec.buffer[: SR.RECORD_WORDS].tolist()
        lines = rec.check()
        assert len(lines) == 1 and lines[0] == SR.format_record(words)
        assert rec.check() == []  # logged once
    finally:
        rec.close()


def _child(mode: str) -> str:
    return textwrap.dedent(
        f"""
        import signal, sys, time, torch
        from vllm.models.kimi_k3.nvidia import spin_record as SR
        signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))
        SR.get(0)
        torch.cuda.synchronize()
        print("READY", flush=True)
        time.sleep({30 if mode == "sigterm" else 0.4})
        """
    )


@cuda
@pytest.mark.parametrize("mode", ["exit", "sigterm"])
def test_process_exits_cleanly_with_records(mode):
    p = subprocess.Popen([sys.executable, "-c", _child(mode)], stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True)
    seen = []
    for line in p.stdout:  # warnings may come first (stderr is merged)
        seen.append(line)
        if line.strip().endswith("READY"):
            break
    if mode == "sigterm":
        time.sleep(1.1)  # several 0.25 s watcher polls in flight
        p.send_signal(signal.SIGTERM)
    rest, _ = p.communicate(timeout=60)
    text = "".join(seen) + rest
    assert p.returncode == 0, (p.returncode, text[-2000:])
    assert "terminate called" not in text
