# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""sdec DenseFlow (the layer-0 dense MLP on the SP shard, gate_up + SiTU + down in one
grid) against the stock path (cuBLAS + situ_and_mul + cuBLAS) within the BF16
rounding of a reordered FP32 sum, bitwise deterministic, at shapes whose stream-K
ranges cut blocks into many segments (several partials a block); and the SP wiring
(die-local re-homing keeps the stock module's values)."""

import types

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="tcgen05 (Blackwell / Rubin)",
)

BETA, LINEAR = 4.0, 25.0


def _stock(x, wgu, wd):
    gu = x @ wgu.t()
    h = torch.empty(x.shape[0], wd.shape[1], dtype=x.dtype, device=x.device)
    torch.ops._C.situ_and_mul(h, gu, BETA, LINEAR)
    return h, F.linear(h, wd)


def _close(out, ref):
    err = (out.float() - ref.float()).abs().max().item()
    scale = ref.float().abs().max().item()
    assert err <= 2e-2 * scale, (err, scale)


class _Situ:
    """The stock SituAndMul op (the KimiMLP act_fn)."""

    beta, linear_beta = BETA, LINEAR

    def __call__(self, x):
        out = x.new_empty(x.shape[0], x.shape[1] // 2)
        torch.ops._C.situ_and_mul(out, x, BETA, LINEAR)
        return out


def _mlp(wgu, wd):
    act = _Situ()
    lin = lambda w: types.SimpleNamespace(weight=torch.nn.Parameter(w, False),
                                          bias=None, tp_size=1)  # fmt: skip
    return types.SimpleNamespace(
        gate_up_proj=lin(wgu), down_proj=lin(wd), act_fn=act,
        shard_sequence_parallel=False, gemm_rs_ar=None,
    )  # fmt: skip


@pytest.mark.parametrize("rows", [1, 20, 40, 96, 128, 144, 160, 176, 192, 256])
def test_dense_stream_matches_stock(rows):
    from vllm.models.kimi_k3.nvidia.specialized_decode.runtime.sp_dense import (
        SpDenseMlp,
    )

    dev = torch.device("cuda")
    torch.manual_seed(rows)
    hidden, inter = 1024, 4096  # ~2-3 K steps a range: every block in many segments
    wgu = torch.randn(2 * inter, hidden, device=dev, dtype=torch.bfloat16) * 0.03
    wd = torch.randn(hidden, inter, device=dev, dtype=torch.bfloat16) * 0.02
    x = torch.randn(rows, hidden, device=dev, dtype=torch.bfloat16)
    h_ref, y_ref = _stock(x, wgu, wd)
    mlp = _mlp(wgu.clone(), wd.clone())
    dense = SpDenseMlp(mlp, (rows,), dev, 0)
    y = dense.run(x).clone()
    _close(dense.h[:rows], h_ref)
    _close(y, y_ref)
    # The re-homed stock weights keep their values; reruns are bitwise.
    assert torch.equal(mlp.gate_up_proj.weight, wgu)
    assert torch.equal(mlp.down_proj.weight, wd)
    for _ in range(3):
        assert torch.equal(dense.run(x), y)


def test_dense_stream_layer0_shape():
    from vllm.models.kimi_k3.nvidia.specialized_decode.runtime.sp_dense import (
        SpDenseMlp,
    )

    dev = torch.device("cuda")
    torch.manual_seed(0)
    hidden, inter, rows = 7168, 33792, 32
    wgu = torch.randn(2 * inter, hidden, device=dev, dtype=torch.bfloat16) * 0.02
    wd = torch.randn(hidden, inter, device=dev, dtype=torch.bfloat16) * 0.02
    x = torch.randn(rows, hidden, device=dev, dtype=torch.bfloat16)
    h_ref, y_ref = _stock(x, wgu, wd)
    dense = SpDenseMlp(_mlp(wgu, wd), (rows,), dev, 0)
    y = dense.run(x)
    _close(dense.h[:rows], h_ref)
    _close(y, y_ref)
    # SiTU on the BF16-rounded gate / up as the stock op: only GEMM-order flips.
    assert (dense.h[:rows] != h_ref).float().mean().item() < 0.02


def test_dense_flow_rows_above_128_are_row_exact():
    """A row's result does not depend on how many rows share the launch (the
    stream-K ranges split K, not rows): the 176- and 256-row specializations give the
    first 128 rows bitwise as the 128-row one, and every row as a 1-row run."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.runtime.sp_dense import (
        MAX_ROWS,
        SpDenseMlp,
    )

    assert MAX_ROWS == 256
    dev = torch.device("cuda")
    torch.manual_seed(7)
    hidden, inter = 1024, 4096
    wgu = torch.randn(2 * inter, hidden, device=dev, dtype=torch.bfloat16) * 0.03
    wd = torch.randn(hidden, inter, device=dev, dtype=torch.bfloat16) * 0.02
    x = torch.randn(256, hidden, device=dev, dtype=torch.bfloat16)
    dense = SpDenseMlp(_mlp(wgu, wd), (1, 128, 176, 256), dev, 0)
    y128 = dense.run(x[:128]).clone()
    y176 = dense.run(x[:176]).clone()
    y256 = dense.run(x).clone()
    assert torch.equal(y176[:128], y128)
    assert torch.equal(y256[:176], y176)
    for r in (0, 127, 128, 175, 255):
        assert torch.equal(dense.run(x[r : r + 1]), y256[r : r + 1]), r


def test_dense_rows_above_the_cap_raise_at_build():
    from vllm.models.kimi_k3.nvidia.specialized_decode.runtime.sp_dense import (
        SpDenseMlp,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.shapes import Unsupported

    dev = torch.device("cuda")
    wgu = torch.zeros(2 * 4096, 1024, device=dev, dtype=torch.bfloat16)
    wd = torch.zeros(1024, 4096, device=dev, dtype=torch.bfloat16)
    with pytest.raises(Unsupported, match="shard rows"):
        SpDenseMlp(_mlp(wgu, wd), (128, 272), dev, 0)


def test_sp_block_dense_layers_run_dense_flow_or_raise():
    """Every captured shard size of the SP decode's dense layer runs DenseFlow (up
    to sp_dense.MAX_ROWS, 129-256 included) or the build raises: no stock dense
    module on a captured shape."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.runtime.sp_attention import (
        dense_mlps,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.shapes import Unsupported

    dev = torch.device("cuda")
    wgu = torch.randn(2 * 4096, 1024, device=dev, dtype=torch.bfloat16) * 0.03
    wd = torch.randn(1024, 4096, device=dev, dtype=torch.bfloat16) * 0.02
    dense = types.SimpleNamespace(is_moe_layer=False, mlp=_mlp(wgu, wd), layer_idx=0)
    moe = types.SimpleNamespace(is_moe_layer=True, mlp=None, layer_idx=1)
    built = dense_mlps([dense, moe], (16, 128, 144, 176, 256), dev)
    assert set(built) == {0}
    assert all(built[0].serves(n) for n in (16, 128, 144, 176, 256))
    with pytest.raises(Unsupported, match="shard rows"):
        dense_mlps([dense], (128, 272), dev)
