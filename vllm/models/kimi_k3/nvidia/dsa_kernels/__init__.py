# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CuTe DSL kernels for Kimi K3.1 DSA (sparse attention + indexer)."""

import os
from pathlib import Path


def stable_sources(build: str, sources: list[Path]) -> list[str]:
    """``sources`` and the .cu / .cuh files beside them, copied once into ``build/src``.

    ``build`` must be named by the digest of exactly those files (it is shared by
    every checkout of the same sources); torch's ninja build compares source and
    object mtimes, so a checkout whose unchanged copy was newer than the last build
    rebuilt it (~40 s under the build lock). The copy is written once, so its mtime
    never changes.
    """
    src = Path(build) / "src"
    if not src.is_dir():
        tmp = Path(build) / f"src.tmp.{os.getpid()}"
        tmp.mkdir(parents=True, exist_ok=True)
        for f in {s.parent for s in sources}:
            for header in f.iterdir():
                if header.suffix in (".cu", ".cuh"):
                    (tmp / header.name).write_bytes(header.read_bytes())
        try:
            tmp.rename(src)
        except OSError:  # another process created it first
            for f in tmp.iterdir():
                f.unlink()
            tmp.rmdir()
    return [str(src / s.name) for s in sources]
