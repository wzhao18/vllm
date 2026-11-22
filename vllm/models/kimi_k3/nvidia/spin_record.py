# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-visible spin-guard records (shared by the MegaMoE and the PIDX spin guards).

Device ``printf`` from a kernel that traps is only flushed if the host later synchronizes the
context; a process that dies first (or a kernel that never completes) loses it. A guard that
times out therefore also writes one record into this process's pinned host memory (device
accessible through UVA), fences at system scope and only then traps. A host watcher thread logs
every completed record; pinned host memory survives the CUDA context's loss.

Layout: ``NUM_RECORDS`` records of ``RECORD_WORDS`` int32 words.

======  ===========================================================================
word    meaning
======  ===========================================================================
0       state: 0 free, 1 claimed by a device thread (filling), ``DONE`` complete
1       source (``SOURCE_*``: which guard family wrote it)
2       site (guard site id)
3       rank (set by the host at creation)
4       call id (or expected call id)
5       observed value
6       target value
7       SM id
8       aux (MegaMoE: arrival mask over ranks; PIDX: the launch's size)
9       number of entries
10+4i   entry i: rank, issued, entered, verdict (``VERDICTS``), at most 13 entries
48..63  PIDX (no entries): kernel name, NUL-terminated (k31_spin_guard.cuh)
======  ===========================================================================

Record 0 is reserved for the MegaMoE missing-rank report; record 1 takes the first other guard.
"""

from __future__ import annotations

import atexit
import sys
import threading

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

RECORD_WORDS = 64
NUM_RECORDS = 2
DONE = 0x5350494E  # 'SPIN'
SOURCE_MEGAMOE = 1
SOURCE_PIDX = 2
MAX_ENTRIES = (RECORD_WORDS - 10) // 4
NAME_WORD = 48  # PIDX records: kernel name in words NAME_WORD..RECORD_WORDS-1
VERDICTS = {1: "H1_host_never_issued", 2: "H2_gpu_stuck_before_call", 3: "IN_call_not_published"}
_SOURCES = {SOURCE_MEGAMOE: "megamoe_rubin", SOURCE_PIDX: "pidx"}

_lock = threading.Lock()
_instance: SpinRecords | None = None


def _name(words: list[int]) -> str:
    raw = b"".join((w & 0xFFFFFFFF).to_bytes(4, "little") for w in words[NAME_WORD:])
    return raw.split(b"\0", 1)[0].decode(errors="replace")


def format_record(words: list[int]) -> str:
    if words[1] == SOURCE_PIDX:
        return (
            f"K31_SPIN_RECORD source=pidx kernel={_name(words)} site={words[2]} "
            f"rank={words[3]} call={words[4]} size={words[8]} observed={words[5]} "
            f"target={words[6]} smid={words[7]}"
        )
    n = max(0, min(words[9], MAX_ENTRIES))
    entries = "; ".join(
        f"rank {words[10 + 4 * i]} issued {words[11 + 4 * i]} entered {words[12 + 4 * i]} "
        f"{VERDICTS.get(words[13 + 4 * i], words[13 + 4 * i])}"
        for i in range(n)
    )
    return (
        f"K31_SPIN_RECORD source={_SOURCES.get(words[1], words[1])} site={words[2]} "
        f"rank={words[3]} call={words[4]} observed={words[5]} target={words[6]} smid={words[7]} "
        f"aux=0x{words[8] & 0xFFFFFFFF:x}" + (f" missing=[{entries}]" if n else "")
    )


class SpinRecords:
    """This process's record buffer and its watcher thread."""

    def __init__(self, rank: int, poll_s: float = 0.25):
        self.buffer = torch.zeros(NUM_RECORDS * RECORD_WORDS, dtype=torch.int32, pin_memory=True)
        for r in range(NUM_RECORDS):
            self.buffer[r * RECORD_WORDS + 3] = rank
        self._logged = [False] * NUM_RECORDS
        self._poll_s = poll_s
        self._thread = threading.Thread(target=self._watch, name="k31-spin-records", daemon=True)
        self._thread.start()
        atexit.register(self.check)

    @property
    def address(self) -> int:
        """Device-accessible address of record 0 (UVA: the pinned host pointer)."""
        return self.buffer.data_ptr()

    def check(self) -> list[str]:
        """Log every newly completed record once; returns the new lines."""
        lines = []
        for r in range(NUM_RECORDS):
            if self._logged[r]:
                continue
            words = self.buffer[r * RECORD_WORDS : (r + 1) * RECORD_WORDS].tolist()
            if words[0] != DONE:
                continue
            self._logged[r] = True
            line = format_record(words)
            lines.append(line)
            logger.error("%s", line)
            print(line, file=sys.stderr, flush=True)
        return lines

    def _watch(self) -> None:
        import time

        while not all(self._logged):
            time.sleep(self._poll_s)
            self.check()


def get(rank: int) -> SpinRecords:
    """The process-wide record buffer (created on first use, before any guarded launch)."""
    global _instance
    with _lock:
        if _instance is None:
            _instance = SpinRecords(rank)
        return _instance
