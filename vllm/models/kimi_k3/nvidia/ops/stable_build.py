# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sources of the Kimi K3 JIT extensions, copied where their mtimes never change."""

import os
import shutil

SPIN_GUARD = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "k31_spin_guard.cuh")


def spin_guard_bytes() -> bytes:
    """The shared bounded-wait header (k31_spin_guard.cuh): fold it into the build
    digest of every extension that includes it."""
    with open(SPIN_GUARD, "rb") as fh:
        return fh.read()


def spin_rank() -> int:
    """The rank printed by k31_spin_guard.cuh timeout lines (-1: unknown)."""
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:  # noqa: BLE001 - diagnostics only
        pass
    for key in ("RANK", "LOCAL_RANK"):
        if os.environ.get(key, "").lstrip("-").isdigit():
            return int(os.environ[key])
    return -1


def spin_record_ptr() -> int:
    """Device address of the record the raw CUDA guards fill on a timeout (record 1 of
    this process's host-mapped spin_record buffer; 0: none); see k31_spin_guard.cuh."""
    try:
        from vllm.models.kimi_k3.nvidia import spin_record

        rec = spin_record.get(spin_rank())
        return rec.address + spin_record.RECORD_WORDS * rec.buffer.element_size()
    except Exception:  # noqa: BLE001 - diagnostics only
        return 0


def stable_copy(build: str, root: str, files: list[str] | None = None,
                extra: list[str] | None = None) -> str:
    """Return ``build/src``: a copy of ``files`` (relative to ``root``; every file
    below ``root`` when None), written once.

    ``extra``: absolute paths copied flat next to the sources (shared headers such as
    k31_spin_guard.cuh). ``build`` must be named by a digest of exactly the copied
    files, so it is shared by every checkout of the same sources. torch's ninja build compares source and
    object mtimes: built straight from a checkout whose unchanged copy is newer than
    the last build, the shared object is rebuilt under its build lock (FlashKDA:
    ~75 s at every new checkout). The copy appears by one atomic rename, so its
    mtimes never change.
    """
    src = os.path.join(build, "src")
    if not os.path.isdir(src):
        tmp = os.path.join(build, f"src.tmp.{os.getpid()}")
        shutil.rmtree(tmp, ignore_errors=True)
        if files is None:
            shutil.copytree(
                root,
                tmp,
                copy_function=shutil.copyfile,
                ignore=shutil.ignore_patterns("__pycache__"),
            )
        else:
            for f in files:
                dst = os.path.join(tmp, f)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copyfile(os.path.join(root, f), dst)
        for path in extra or ():  # files from elsewhere, next to the sources
            shutil.copyfile(path, os.path.join(tmp, os.path.basename(path)))
        try:
            os.rename(tmp, src)
        except OSError:  # another process created it first
            shutil.rmtree(tmp, ignore_errors=True)
    return src
