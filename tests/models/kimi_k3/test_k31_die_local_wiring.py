# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The SP decode's die-local GEMM sites on a two-die part (VR200): the policy picks
the die-local GEMM only for a weight placed on the dies, ``DecodeProjection`` runs it
within one BF16 rounding of an FP32 reference and refuses an unplaced weight, and a
stock linear given ``attach_die_local_method`` runs it on the rows it serves and the
stock GEMM (bitwise) on the others."""

import pytest
import torch

pytest.importorskip("cutlass")

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
    pytest.skip("needs a Rubin GPU (compute capability 10.7)", allow_module_level=True)

from vllm.model_executor.layers.linear import UnquantizedLinearMethod  # noqa: E402
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm import (  # noqa: E402
    die_local,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.decode import (  # noqa: E402
    DecodeProjection,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.sp_policy import (  # noqa: E402
    DIE_LOCAL_KINDS,
    die_local_tactics,
    sp_projection_tactics,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.shapes import (  # noqa: E402
    Context,
    Unsupported,
)

DEVICE = torch.device("cuda", 0)
if not die_local.two_dies(DEVICE.index):
    pytest.skip("needs a two-die part", allow_module_level=True)


def context():
    torch.cuda.set_device(DEVICE)
    sms = torch.cuda.get_device_properties(DEVICE).multi_processor_count
    return Context(
        arch="rubin", tp=4, rank=0, dcp=1, sms=sms, q=6, max_rows=192,
        max_context=131072, bf16_state=False, dp=4, ep=16, sequence_parallel=True,
        heads_per_rank=24,
    )  # fmt: skip


def operands(n, k, rows, seed):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    x = torch.randn(rows, k, device=DEVICE, generator=g).bfloat16()
    w = (torch.randn(n, k, device=DEVICE, generator=g) * 0.02).bfloat16()
    return x, w


def check_close(out, x, w):
    ref = (x.float() @ w.float().t()).bfloat16().float()
    err = (out.float() - ref).abs()
    assert bool((err <= ref.abs() * 2**-7 + 2**-12).all()), err.max().item()


SHAPES = {"kda_in": (12448, 7168), "shared_gu_side": (12288, 7168),
          "latent_down": (3584, 7168)}  # fmt: skip


@pytest.mark.parametrize("kind", list(SHAPES))
def test_policy_picks_die_local_for_placed_weights(kind):
    rows = (6, 48, 192, 288)
    stock = sp_projection_tactics(kind, *SHAPES[kind], rows, context())
    assert all(t[0] != "die_local" for t in stock.values())
    weight = torch.nn.Parameter(torch.empty(0), requires_grad=False)
    assert die_local_tactics(kind, weight, stock) == stock  # not placed
    weight.on_dies = True
    placed = die_local_tactics(kind, weight, stock)
    for r in rows:
        if kind in DIE_LOCAL_KINDS and r <= die_local.MAX_ROWS:
            assert placed[r] == ("die_local", None), r
        else:  # past MAX_ROWS, or a family the die-local GEMM does not serve
            assert placed[r] == stock[r], r


def test_projection_runs_die_local_on_placed_weight():
    n, k, rows = 6144, 7168, (6, 48, 192)
    x, w = operands(n, k, max(rows), seed=1)
    tactics = dict.fromkeys(rows, ("die_local", None))
    offset = torch.empty(n + 8, k, device=DEVICE, dtype=torch.bfloat16)[8:]
    with pytest.raises(Unsupported, match="placed"):  # not 2 MiB aligned
        DecodeProjection(offset.copy_(w), rows, tactics, family="kda_in")
    placed = die_local.relocate_halves(w)
    projection = DecodeProjection(placed, rows, tactics, family="kda_in")
    out = torch.empty(max(rows), n, device=DEVICE, dtype=torch.bfloat16)
    projection.check_operands(x, out)
    for r in rows:
        out.fill_(float("nan"))
        projection.run(x[:r], out[:r])
        check_close(out[:r], x[:r], w)


def test_linear_method_serves_decode_rows_only(monkeypatch):
    n, k, cap = 4096, 7168, 48
    x, w = operands(n, k, 300, seed=2)
    linear = torch.nn.Module()
    linear.weight = torch.nn.Parameter(w.clone(), requires_grad=False)
    linear.quant_method = UnquantizedLinearMethod()
    assert not die_local.attach_die_local_method(linear, cap)  # not placed
    die_local.relocate_parameter_(linear.weight)
    assert die_local.attach_die_local_method(linear, cap)
    method = linear.quant_method
    assert isinstance(method, die_local.DieLocalLinearMethod)
    calls = []
    run = die_local.DieLocalEngine.__call__

    def counted(engine, a, out):
        calls.append(a.shape[0])
        return run(engine, a, out)

    monkeypatch.setattr(die_local.DieLocalEngine, "__call__", counted)
    check_close(method.apply(linear, x[:cap]), x[:cap], w)
    assert calls == [cap]
    stock = method.apply(linear, x)  # past the engine's rows: the stock GEMM
    assert calls == [cap]
    assert torch.equal(stock, torch.nn.functional.linear(x, w))
