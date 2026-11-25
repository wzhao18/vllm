# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Row-sliced bulk reads (``_SafetensorsBulkReader`` with ``read_rows``): the
declared rows hold the file's bytes, other pages are not read and fault."""

import mmap
import subprocess
import sys
import textwrap

import pytest
import torch
from safetensors.torch import save_file

from vllm.model_executor.model_loader.weight_utils import _SafetensorsBulkReader

TP = 4


def _tensors() -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    out = {}
    for e in range(3):
        for shard in ("w1", "w3"):
            out[f"layers.0.experts.{e}.{shard}.weight_packed"] = torch.randint(
                0, 256, (3072, 1792), generator=g, dtype=torch.uint8
            )
            out[f"layers.0.experts.{e}.{shard}.weight_scale"] = torch.randint(
                0, 256, (3072, 112), generator=g, dtype=torch.uint8
            )
        out[f"layers.0.experts.{e}.w2.weight_packed"] = torch.randint(
            0, 256, (3584, 1536), generator=g, dtype=torch.uint8
        )
    out["layers.0.norm.weight"] = torch.randn(7168, generator=g).bfloat16()
    out["layers.0.q_proj.weight"] = torch.randn(96, 7168, generator=g).bfloat16()
    return out


def _rows(rank: int):
    def read_rows(name: str, shape: list[int], dtype: torch.dtype):
        if ".w1." in name or ".w3." in name:
            per = shape[0] // TP
            return per * rank, per
        return None

    return read_rows


def _read(
    path: str, read_rows=None, coalesce: int | None = None
) -> tuple[dict, _SafetensorsBulkReader]:
    reader = _SafetensorsBulkReader([path], None, read_rows)
    if coalesce is not None:
        reader.COALESCE_BYTES = coalesce
    try:
        return dict(reader.tensors(path)), reader
    finally:
        reader.close()


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory) -> tuple[str, dict[str, torch.Tensor]]:
    tensors = _tensors()
    path = str(tmp_path_factory.mktemp("ckpt") / "model-00001-of-00001.safetensors")
    save_file(tensors, path)
    return path, tensors


@pytest.mark.parametrize("coalesce", [0, 1 << 20])
@pytest.mark.parametrize("rank", range(TP))
def test_declared_rows_are_the_file_bytes(checkpoint, rank, coalesce):
    path, ref = checkpoint
    full, _ = _read(path)
    sliced, reader = _read(path, _rows(rank), coalesce)
    assert sliced.keys() == ref.keys()
    for name, t in ref.items():
        assert sliced[name].shape == t.shape and sliced[name].dtype == t.dtype
        assert torch.equal(full[name], t)
        rows = _rows(rank)(name, list(t.shape), t.dtype)
        if rows is None:
            assert torch.equal(sliced[name], t), name
        else:
            assert torch.equal(sliced[name].narrow(0, *rows), t.narrow(0, *rows))
    sliced_bytes = sum(
        t.nbytes for n, t in ref.items() if ".w1." in n or ".w3." in n
    )
    assert reader.sliced_tensors == 12
    if coalesce == 0:
        # Only the whole pages around each rank's rows are read.
        widened = 12 * 2 * mmap.PAGESIZE
        assert reader.bytes_skipped >= sliced_bytes * (TP - 1) // TP - widened


def test_rows_outside_the_declared_range_fault(checkpoint):
    path, _ = checkpoint
    code = textwrap.dedent(
        f"""
        import torch
        from vllm.model_executor.model_loader.weight_utils import (
            _SafetensorsBulkReader,
        )
        r = _SafetensorsBulkReader(
            [{path!r}], None, lambda n, s, d: (768, 768) if ".w1." in n else None
        )
        t = dict(r.tensors({path!r}))["layers.0.experts.0.w1.weight_packed"]
        print("inside", int(t[768:1536].sum()), flush=True)
        print("outside", int(t[100].sum()), flush=True)
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
    )
    assert "inside" in proc.stdout
    assert "outside" not in proc.stdout
    assert proc.returncode != 0


def test_empty_rows_read_nothing(checkpoint):
    path, ref = checkpoint
    got, reader = _read(path, lambda n, s, d: (0, 0) if ".w3." in n else None, 0)
    assert reader.sliced_tensors == 6
    w3_bytes = sum(t.nbytes for n, t in ref.items() if ".w3." in n)
    assert reader.bytes_skipped >= w3_bytes - 6 * 2 * mmap.PAGESIZE
    for name, t in ref.items():
        if ".w3." not in name:
            assert torch.equal(got[name], t), name


def test_rows_outside_the_tensor_raise(checkpoint):
    path, _ = checkpoint
    with pytest.raises(ValueError, match="outside the tensor"):
        _read(path, lambda n, s, d: (s[0] - 1, 2) if ".w1." in n else None)
