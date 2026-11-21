# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The bulk-read safetensors path yields exactly the tensors of the mmap path."""

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from vllm.model_executor.model_loader import weight_utils
from vllm.model_executor.model_loader.ep_weight_filter import should_skip_weight
from vllm.model_executor.model_loader.weight_utils import _SafetensorsBulkReader


def _shard(i: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(i)
    tensors = {
        f"model.layers.{i}.mlp.experts.{e}.{w}.weight": torch.randint(
            0, 255, (48, 40 + e), dtype=torch.uint8, generator=g
        )
        for e in range(6)
        for w in ("w1", "w2", "w3")
    }
    tensors.update(
        {
            f"model.layers.{i}.mlp.experts.{e}.w2.weight_scale": torch.randn(
                48, 5, generator=g
            ).to(torch.float8_e4m3fn)
            for e in range(6)
        }
    )
    tensors[f"model.layers.{i}.mlp.experts.0.w1.input_scale"] = torch.tensor(0.5)
    tensors[f"model.layers.{i}.self_attn.o_proj.weight"] = torch.randn(
        33, 17, generator=g
    ).to(torch.bfloat16)
    tensors[f"model.layers.{i}.norm.weight"] = torch.randn(7, generator=g)
    tensors[f"model.layers.{i}.big"] = torch.randn(3000, generator=g).double()
    tensors[f"model.layers.{i}.odd"] = torch.arange(3, dtype=torch.int8)
    tensors[f"model.layers.{i}.ids"] = torch.arange(5, dtype=torch.int64)
    return tensors


def _bytes(t: torch.Tensor) -> torch.Tensor:
    return t.reshape(-1).view(torch.uint8)


def _mmap_tensors(path, local_expert_ids):
    with safe_open(path, framework="pt") as f:
        return {
            k: f.get_tensor(k)
            for k in f.keys()  # noqa: SIM118
            if not should_skip_weight(k, local_expert_ids)
        }


@pytest.mark.parametrize("local_expert_ids", [None, {1, 4}])
@pytest.mark.parametrize(
    "sizes",
    [
        dict(),  # defaults: one group per file
        dict(GROUP_BYTES=4096, GAP_BYTES=0, PIECE_BYTES=1000, WINDOW_BYTES=1),
        dict(GROUP_BYTES=20000, GAP_BYTES=64, PIECE_BYTES=7, WINDOW_BYTES=50000),
    ],
)
def test_bulk_reader_matches_mmap(tmp_path, monkeypatch, local_expert_ids, sizes):
    for k, v in sizes.items():
        monkeypatch.setattr(_SafetensorsBulkReader, k, v)
    files = []
    for i in range(3):
        path = str(tmp_path / f"model-{i}.safetensors")
        save_file(_shard(i), path)
        files.append(path)
    empty = str(tmp_path / "model-3.safetensors")
    save_file({"model.layers.3.mlp.experts.0.w1.weight": torch.zeros(2, 2)}, empty)
    files.append(empty)

    reader = _SafetensorsBulkReader(files, local_expert_ids)
    try:
        for path in files:
            got = dict(reader.tensors(path))
            want = _mmap_tensors(path, local_expert_ids)
            assert set(got) == set(want)
            for k, w in want.items():
                g = got[k]
                assert g.dtype == w.dtype and g.shape == w.shape, k
                assert torch.equal(_bytes(g), _bytes(w)), k
    finally:
        reader.close()


def test_iterator_uses_bulk_reads_on_network_fs(tmp_path, monkeypatch):
    path = str(tmp_path / "model-0.safetensors")
    save_file(_shard(0), path)
    monkeypatch.setattr(weight_utils, "_get_fs_type", lambda files: "lustre")
    used = []
    orig = _SafetensorsBulkReader.tensors

    def tensors(self, p):
        used.append(p)
        yield from orig(self, p)

    monkeypatch.setattr(_SafetensorsBulkReader, "tensors", tensors)
    got = dict(weight_utils.safetensors_weights_iterator([path], False))
    assert used == [path]
    want = _mmap_tensors(path, None)
    assert set(got) == set(want)
    for k, w in want.items():
        assert torch.equal(_bytes(got[k]), _bytes(w)), k
