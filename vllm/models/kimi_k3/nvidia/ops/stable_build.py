# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sources of the Kimi K3 JIT extensions, copied where their mtimes never change."""

import os
import shutil


def stable_copy(build: str, root: str, files: list[str] | None = None) -> str:
    """Return ``build/src``: a copy of ``files`` (relative to ``root``; every file
    below ``root`` when None), written once.

    ``build`` must be named by a digest of exactly the copied files, so it is shared
    by every checkout of the same sources. torch's ninja build compares source and
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
        try:
            os.rename(tmp, src)
        except OSError:  # another process created it first
            shutil.rmtree(tmp, ignore_errors=True)
    return src
