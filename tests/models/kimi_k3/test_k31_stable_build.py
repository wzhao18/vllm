# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Kimi K3 JIT extensions compile a copy of their sources written once, so a
checkout whose unchanged sources are newer than the last build never rebuilds the
shared digest-named object."""

import os

from vllm.models.kimi_k3.nvidia.ops.stable_build import stable_copy


def _checkout(root):
    (root / "csrc" / "smxx").mkdir(parents=True)
    (root / "csrc" / "binding.cpp").write_text('#include "smxx/k.cuh"\n')
    (root / "csrc" / "smxx" / "k.cuh").write_text("// header\n")
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "x.pyc").write_bytes(b"\0")
    return root


def _newer(path):
    t = os.stat(path).st_mtime_ns + 10**12
    os.utime(path, ns=(t, t))


def test_tree_copied_once(tmp_path):
    checkout = _checkout(tmp_path / "checkout")
    build = tmp_path / "build" / "0123456789abcdef-s"
    build.mkdir(parents=True)

    src = stable_copy(str(build), str(checkout))
    assert src == str(build / "src")
    assert (build / "src" / "csrc" / "smxx" / "k.cuh").read_text() == "// header\n"
    assert not (build / "src" / "__pycache__").exists()
    copied = build / "src" / "csrc" / "binding.cpp"
    mtime = os.stat(copied).st_mtime_ns

    _newer(checkout / "csrc" / "binding.cpp")  # another checkout, same content
    assert stable_copy(str(build), str(checkout)) == src
    assert os.stat(copied).st_mtime_ns == mtime
    assert [p.name for p in build.iterdir()] == ["src"]  # no temp dirs left


def test_listed_files_only(tmp_path):
    checkout = _checkout(tmp_path / "checkout")
    build = tmp_path / "build" / "fedcba9876543210-s"
    build.mkdir(parents=True)

    src = stable_copy(str(build), str(checkout / "csrc"), ["binding.cpp"])
    assert sorted(os.listdir(src)) == ["binding.cpp"]


def test_copy_made_by_another_process_wins(tmp_path, monkeypatch):
    """Two processes copy at once: the first rename wins, the loser's copy goes."""
    import shutil

    checkout = _checkout(tmp_path / "checkout")
    build = tmp_path / "build" / "00112233aabbccdd-s"
    build.mkdir(parents=True)
    copytree, raced = shutil.copytree, []

    def racing(*args, **kw):  # the other process renames its copy meanwhile
        if not raced:  # (copytree recurses through the module name)
            raced.append(1)
            (build / "src").mkdir()
            (build / "src" / "first.txt").write_text("first copy\n")
        return copytree(*args, **kw)

    monkeypatch.setattr(shutil, "copytree", racing)
    assert stable_copy(str(build), str(checkout)) == str(build / "src")
    assert os.listdir(build / "src") == ["first.txt"]
    assert [p.name for p in build.iterdir()] == ["src"]
