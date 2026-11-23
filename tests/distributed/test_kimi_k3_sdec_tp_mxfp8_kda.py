# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 TP specialized decode (non-SP, TP = DCP, MoE without EP) on the NVFP4
checkpoint's online-MXFP8 KDA projections, against the stock layers.

On 2 or 4 Rubin ranks (the deployment is TP16; every provider is generic in TP), the
real stock linears with the export's quantization config (``modelopt_mixed``, MXFP8
W8A8 with 32-wide E8M0 blocks, ``KimiK3Mxfp8Kernel`` dispatch as the K3 model installs
it), checkpoint-format tensors through their weight loaders, 96 KDA heads:

* ``in_proj_qkvgfab`` (``_KimiGDNMergedColumnParallelLinear``, full-rank gate, f_a
  replicated, TP padding): the TP provider's in-projection (``Mxfp8Projection`` as
  ``providers.Attention`` builds it, every decode row count of the policy seams)
  equals the stock layer's forward bit for bit where it runs stock's kernel (every
  row of the TP16 [3216, 7168] shard), else (a measured ``MX_TUNED`` kind) within one
  BF16 rounding; ``check_bf16_projections`` accepts the
  MXFP8 in / o projections of a KDA layer and still refuses any other MXFP8 linear;
* ``o_proj`` (``RowParallelLinear``, reduce_results): ``oproj.mxfp8.Mxfp8Oproj`` (the
  MXFP8 GEMM, its partial pushed into the BF16 path's mailboxes, the BF16 path's
  consumers with their AttnRes) vs the stock ``o_proj`` (MXFP8 GEMM + vLLM all-reduce)
  and the stock AttnRes op (``seams.StockAttnRes.for_post``) on the same inputs: each
  rank's partial is stock's bit for bit; the updated prefix is within one BF16 rounding
  of the FP32 rank-order sum of the stock partials plus the old prefix; the AttnRes
  output within reduction noise of stock's (FP32 sums in another order); for a
  has-delta and a block-write layer on alternating parities, over one-shot,
  two-shot (column owners reduce), wide two-shot and row-sharded (owners) batches;
  every mailbox re-armed after each step; a captured graph replays the eager bits.

Needs 2+ Rubin (SM107) GPUs with NVLink multicast."""

import os
import time
import zlib
from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp

from vllm.utils.network_utils import get_open_port
from vllm.utils.system_utils import update_environment_variables

HIDDEN, HEADS, DIM = 7168, 96, 128
LAYER = "language_model.model.layers.1.self_attn"
SENTINEL = -0x80000000
# in-projection rows: the B1 / skinny / CuTe-DSL / cuBLASLt seams and wide rows
IN_ROWS = (1, 3, 8, 16, 17, 64, 128, 384, 512)
# o-projection rows: one-shot 1-16, two-shot 17-64 (patched in at this TP), wide
# two-shot 80 / 128, row-sharded 256
TWOSHOT_TABLE = "17-64:1"
OUT_ROWS = (1, 4, 8, 16, 24, 40, 64, 80, 128, 256)
SHARDED = frozenset({256})
GRAPH_ROWS = (8, 40, 128)


def _has_multicast_support() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from torch._C._autograd import DeviceType
        from torch._C._distributed_c10d import _SymmetricMemory

        return _SymmetricMemory.has_multicast_support(DeviceType.CUDA, 0)
    except Exception:
        return False


def _rubin(world) -> bool:
    if not torch.cuda.is_available() or torch.cuda.device_count() < world:
        return False
    capabilities = {torch.cuda.get_device_capability(i) for i in range(world)}
    return capabilities == {(10, 7)} and _has_multicast_support()


def _generator(*key, device="cuda"):
    gen = torch.Generator(device=device)
    gen.manual_seed(zlib.crc32(repr(key).encode()))
    return gen


def _export_config() -> dict:
    names = ("q_proj", "k_proj", "v_proj", "g_proj", "f_a_proj", "b_proj")
    names += ("in_proj_qkvgfab", "o_proj")
    return {
        "quant_method": "modelopt_mixed",
        "quant_algo": "MIXED_PRECISION",
        "group_size": 16,
        "ignore": [],
        "quantized_layers": {f"{LAYER}.{n}": {"quant_algo": "MXFP8"} for n in names},
    }


def _ckpt(n, k, *key, scale=0.02):
    """Checkpoint tensors of one MXFP8 linear: E4M3 [n, k], UE8M0 bytes [n, k/32]."""
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
    )

    g = _generator("w", n, k, *key)
    w = (torch.randn(n, k, device="cuda", generator=g) * scale).bfloat16()
    wq, ws = mxfp8_e4m3_quantize(w, is_sf_swizzled_layout=False)
    return wq.contiguous(), ws.view(torch.uint8).reshape(n, k // 32).contiguous()


def _load(param, data, shard=None):
    if shard is None:
        param.weight_loader(param, data)
    else:
        param.weight_loader(param, data, shard)


def _rows_error(got, want):
    got, want = got.double(), want.double()
    return (got - want).norm(dim=-1) / want.norm(dim=-1).clamp_min(1e-30)


def _check_rows(name, got, want, median=1e-2, worst=4e-2):
    assert torch.isfinite(got).all(), f"{name}: non-finite"
    errors = _rows_error(got, want)
    assert errors.median() <= median and errors.max() <= worst, (
        f"{name}: row errors median {errors.median():.3g} max {errors.max():.3g}"
    )


def _within_rounding(got, delta, old):
    """``got`` = bf16(bf16(sum) [+ old]) within the BF16 roundings of ``delta`` (the
    FP32 rank-order sum of the stock partials) and of the result: the two FP32 sums
    differ in order only, then each path rounds the sum and the prefix add once."""
    want = delta.bfloat16().float() + (0 if old is None else old.float())
    tol = delta.abs() * 2.0**-7 + want.abs() * 2.0**-8
    tol = tol + torch.finfo(torch.bfloat16).tiny
    return bool(((got.float() - want).abs() <= tol).all())


def _say(rank, *what):
    """Progress on stderr (a hung collective otherwise shows nothing)."""
    import sys

    print(
        f"[{time.strftime('%H:%M:%S')} rank {rank}]", *what, file=sys.stderr, flush=True
    )


def _armed(*tensors):
    return all(bool((t.view(torch.int32) == SENTINEL).all()) for t in tensors)


def _in_projection(rank, world, qcfg):
    from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import (
        _KimiGDNMergedColumnParallelLinear,
    )

    p = HEADS * DIM
    local = 4 * p // world + DIM + HEADS // world
    pad = -local % 16
    sizes = [p] * 4 + [DIM, HEADS] + ([pad * world] if pad else [])
    layer = _KimiGDNMergedColumnParallelLinear(
        HIDDEN, sizes, replicated_shard_id=4, tp_size=world, bias=False,
        quant_config=qcfg, params_dtype=torch.bfloat16,
        prefix=f"{LAYER}.in_proj_qkvgfab",
    )  # fmt: skip
    for shard, n in enumerate(sizes):
        if shard == len(sizes) - 1 and pad:
            wq = torch.zeros(n, HIDDEN, device="cuda").to(torch.float8_e4m3fn)
            ws = torch.full((n, HIDDEN // 32), 127, device="cuda", dtype=torch.uint8)
        else:
            wq, ws = _ckpt(n, HIDDEN, "in", shard)
        _load(layer.weight, wq, shard)
        _load(layer.weight_scale, ws, shard)
    layer.quant_method.process_weights_after_loading(layer)
    return layer, local + pad


def _o_projection(world, qcfg):
    from vllm.model_executor.layers.linear import RowParallelLinear

    layer = RowParallelLinear(
        HEADS * DIM, HIDDEN, bias=False, quant_config=qcfg,
        params_dtype=torch.bfloat16, prefix=f"{LAYER}.o_proj",
    )  # fmt: skip
    wq, ws = _ckpt(HIDDEN, HEADS * DIM, "out")
    _load(layer.weight, wq)
    _load(layer.weight_scale, ws)
    layer.quant_method.process_weights_after_loading(layer)
    return layer


def _decoder(index, write, o_proj, device):
    """The decoder attributes ``Mxfp8Oproj`` / ``StockAttnRes.for_post`` read."""
    g = _generator("decoder", index, device=device)

    def norm(eps=1e-6):
        weight = 1 + 0.1 * torch.randn(HIDDEN, device=device, generator=g)
        return SimpleNamespace(weight=weight.bfloat16(), variance_epsilon=eps)

    query = 0.05 * torch.randn(1, HIDDEN, device=device, generator=g)
    return SimpleNamespace(
        layer_idx=index, prev_valid_blocks=3, is_block_write_layer=write,
        block_write_idx=3 if write else -1, mlp_res_norm=norm(),
        mlp_res_proj=SimpleNamespace(weight=query.bfloat16()),
        post_attention_layernorm=norm(), self_attn=SimpleNamespace(o_proj=o_proj),
    )  # fmt: skip


def _check_formats(in_proj, o_proj, qcfg, world):
    """``check_bf16_projections``: a KDA layer's MXFP8 in / o projections pass;
    an MXFP8 linear anywhere else in the layer is refused at build."""
    from torch import nn

    from vllm.model_executor.layers.linear import ColumnParallelLinear
    from vllm.models.kimi_k3.nvidia.specialized_decode.runtime import providers

    attention = nn.Module()
    attention.in_proj_qkvgfab, attention.o_proj = in_proj, o_proj
    attention.f_b_proj = ColumnParallelLinear(
        DIM, HEADS * DIM, bias=False, params_dtype=torch.bfloat16,
        prefix=f"{LAYER}.f_b_proj",
    )  # fmt: skip
    decoder = nn.Module()
    decoder.self_attn = attention
    providers.check_bf16_projections(decoder)
    assert providers.mxfp8_kda_linear(decoder, "self_attn.o_proj", o_proj)
    # an MXFP8 linear the TP providers read as BF16 (e.g. a shared expert; a
    # separate module: named_modules visits each module once)
    decoder.shared = _o_projection(world, qcfg)
    with pytest.raises(ValueError, match="needs BF16 projections"):
        providers.check_bf16_projections(decoder)


def _worker(env: dict[str, str]) -> None:
    update_environment_variables(env)
    rank, world = int(env["RANK"]), int(env["WORLD_SIZE"])
    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(rank)
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptMixedPrecisionConfig,
    )
    from vllm.models.kimi_k3.nvidia import model as kimi_model
    from vllm.models.kimi_k3.nvidia.mxfp8 import enable_kimi_k3_mxfp8_dispatch
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.attn_res.sharded import (  # noqa: E501
        ShardBox,
        owned,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.twoshot import (
        TwoShot,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.mxfp8 import (
        Mxfp8Projection,
        mxfp8_tactics,
        stock_kind,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.oproj import (
        mxfp8 as oproj_mxfp8,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.oproj import (
        prepared,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.runtime.seams import (
        StockAttnRes,
    )

    with set_current_vllm_config(VllmConfig()), torch.device(device):
        init_distributed_environment(
            world_size=world, rank=rank, local_rank=rank,
            distributed_init_method=f"tcp://localhost:{env['MASTER_PORT']}",
            backend="nccl",
        )  # fmt: skip
        initialize_model_parallel(world, 1)
        try:
            group = get_tp_group().device_group
            cpu_group = get_tp_group().cpu_group
            try:  # (older torch: symmetric memory is enabled per group)
                from torch.distributed._symmetric_memory import (
                    enable_symm_mem_for_group,
                )

                enable_symm_mem_for_group(group.group_name)
            except (ImportError, AttributeError):
                pass
            qcfg = ModelOptMixedPrecisionConfig.from_config(_export_config())
            qcfg.packed_modules_mapping = dict(
                kimi_model.KimiLinearModel.packed_modules_mapping
            )
            in_proj, pitch = _in_projection(rank, world, qcfg)
            o_proj = _o_projection(world, qcfg)
            holder = torch.nn.ModuleList([in_proj, o_proj])
            assert enable_kimi_k3_mxfp8_dispatch(holder) == 2
            _check_formats(in_proj, o_proj, qcfg, world)
            _say(rank, "layers built, formats checked")

            # W1: the TP in-projection vs the stock layer, bit for bit
            projection = Mxfp8Projection(
                in_proj, IN_ROWS, family="kda_in", cap=IN_ROWS[-1]
            )
            assert projection.n == pitch, (projection.n, pitch)
            for r in IN_ROWS:
                g = _generator("x", r, device=device)  # the same x on every rank
                x = torch.randn(r, HIDDEN, device=device, generator=g).bfloat16()
                want = in_proj(x)[0]
                got = torch.empty(r, pitch, device=device, dtype=torch.bfloat16)
                projection.run(x, got)
                if projection.engine(r) == stock_kind(pitch, HIDDEN, r):
                    assert torch.equal(got, want), f"in-projection R{r}"
                else:  # a measured MX_TUNED kind (TP4's [12448, 7168] rows)
                    tol = want.float().abs().amax(dim=1, keepdim=True) * 2.0**-7
                    diff = (got.float() - want.float()).abs()
                    assert (diff <= tol + 1e-3).all(), f"in-projection R{r}"
            # the TP16 shards (in [3216, 7168], o [7168, 768]) run stock's kernel,
            # except the in-projection's measured 17-128 rows (MX_TUNED "pre": the
            # prewait fi_cute, test_specialized_decode_tp16nv_gpu.py)
            for n, kk in ((3216, HIDDEN), (HIDDEN, 768)):
                rows = (*IN_ROWS, 24, 1024, 1536, 2048)
                tactics = mxfp8_tactics(n, kk, rows)
                for r in rows:
                    pre = n == 3216 and 17 <= r <= 128
                    want = "pre" if pre else stock_kind(n, kk, r)
                    assert tactics[r] == want, (n, kk, r, tactics[r])
            _say(rank, "W1 in-projection vs stock ok at", IN_ROWS)

            # W2: the MXFP8 o-projection + exchange + AttnRes vs stock
            prepared.TWOSHOT[world] = TWOSHOT_TABLE
            cap = max(OUT_ROWS)
            k = HEADS * DIM // world
            mailbox = prepared.OprojMailbox(group, device, world, rank)
            twoshot = TwoShot(group, device, world, rank)
            shard = ShardBox(group, device, world, rank, cap)
            local = torch.empty(cap, k, device=device, dtype=torch.bfloat16)
            partial = torch.empty(cap, HIDDEN, device=device, dtype=torch.bfloat16)
            decoders = [_decoder(i, i == 1, o_proj, device) for i in range(2)]
            boxes_args = (mailbox, OUT_ROWS, None, twoshot, local, (SHARDED, shard))
            providers = [
                oproj_mxfp8.Mxfp8Oproj(d, *boxes_args, partial=partial)
                for d in decoders
            ]
            stock = [StockAttnRes.for_post(d) for d in decoders]
            _say(rank, "W2 providers compiled")
            boxes = [mailbox.buffer, twoshot.recv, twoshot.gather]
            boxes += [twoshot.rows_box(cap).recv, twoshot.rows_box(cap).gather]
            boxes += [shard.owners.buffer, shard.gathers[0].buffer]

            def inputs(r, i):
                g = _generator("o", r, i, rank, device=device)
                x = torch.randn(r, k, device=device, generator=g).bfloat16()
                g = _generator("res", r, i, device=device)  # shared by the ranks
                prefix = torch.randn(r, HIDDEN, device=device, generator=g).bfloat16()
                blocks = torch.randn(
                    r, 4, HIDDEN, device=device, generator=g
                ).bfloat16()
                return x, prefix, blocks

            for r in OUT_ROWS:
                for i, (provider, ref) in enumerate(zip(providers, stock)):
                    x, prefix, blocks = inputs(r, i)
                    local[:r].copy_(x)
                    ours_prefix, ours_blocks = prefix.clone(), blocks.clone()
                    out = torch.empty(r, HIDDEN, device=device, dtype=torch.bfloat16)
                    _say(rank, f"W2 run R{r} layer {i}")
                    provider.run(local[:r], ours_prefix, ours_blocks, out)
                    torch.accelerator.synchronize()
                    case = f"o-projection R{r} layer {i} rank {rank}"
                    # this rank's partial: stock's o_proj GEMM bit for bit
                    want = o_proj.quant_method.apply(o_proj, x)
                    assert torch.equal(partial[:r], want), case
                    parts = [torch.empty_like(want) for _ in range(world)]
                    torch.distributed.all_gather(parts, want, group=group)
                    total = parts[0].float()
                    for part in parts[1:]:
                        total = total + part.float()
                    delta = o_proj(x)[0]  # stock: GEMM + vLLM all-reduce
                    ref_prefix, ref_blocks = prefix.clone(), blocks.clone()
                    ref_out = ref.run(delta, ref_prefix, ref_blocks)
                    rows = slice(None)
                    if r in SHARDED:  # the owners update only their rows' prefix
                        first, count = owned(r, world, rank)
                        rows = slice(first, first + count)
                    old = prefix[rows] if i == 0 else None  # (block write: no delta)
                    assert _within_rounding(ours_prefix[rows], total[rows], old), case
                    _check_rows(f"{case} prefix", ours_prefix[rows], ref_prefix[rows])
                    _check_rows(f"{case} out", out, ref_out)
                    assert torch.equal(ours_blocks, blocks), case
                    assert _armed(*boxes), f"{case}: a mailbox left unarmed"
                    _say(rank, "W2 ok", case)
                    # no rank pushes the next case before every rank looked at its
                    # mailboxes (the next push lands in the peers' memory)
                    torch.distributed.barrier(group=cpu_group)

            # a captured two-layer step replays its eager bits
            for r in GRAPH_ROWS:
                state = [inputs(r, i) for i in range(2)]
                outs = [torch.empty(r, HIDDEN, device=device).bfloat16()]
                outs.append(torch.empty_like(outs[0]))
                prefixes = [s[1].clone() for s in state]

                def step(state=state, outs=outs, prefixes=prefixes, r=r):
                    for i, provider in enumerate(providers):
                        local[:r].copy_(state[i][0])
                        provider.run(local[:r], prefixes[i], state[i][2], outs[i])

                step()
                eager = [o.clone() for o in outs]
                eager_prefix = [p.clone() for p in prefixes]
                graph = torch.cuda.CUDAGraph()
                for p, s in zip(prefixes, state):
                    p.copy_(s[1])
                with torch.cuda.graph(graph):
                    step()
                for _ in range(2):
                    for p, s in zip(prefixes, state):
                        p.copy_(s[1])
                    outs[0].zero_()
                    outs[1].zero_()
                    graph.replay()
                    torch.accelerator.synchronize()
                    for a, b in zip(outs + prefixes, eager + eager_prefix):
                        assert torch.equal(a, b), f"graph R{r}"
                    assert _armed(*boxes), f"graph R{r}: a mailbox left unarmed"
                    torch.distributed.barrier(group=cpu_group)
            _say(rank, "graph replays ok")
            torch.distributed.barrier(group=group)
        except BaseException:
            import traceback

            # (no collective teardown: the other ranks may wait in a mailbox)
            _say(rank, "FAILED\n" + traceback.format_exc())
            os._exit(1)
        cleanup_dist_env_and_memory()


def _distributed_run(fn, world_size: int) -> None:
    port = str(get_open_port())
    ctx = mp.get_context("spawn")
    processes = []
    for rank in range(world_size):
        env = {
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": port,
        }
        process = ctx.Process(target=fn, args=(env,))
        processes.append(process)
        process.start()
    deadline = time.monotonic() + int(os.environ.get("SDEC_TEST_TIMEOUT", 900))
    for process in processes:
        process.join(timeout=max(0.0, deadline - time.monotonic()))
    for process in processes:
        if process.is_alive():
            process.kill()
            process.join()
    assert [process.exitcode for process in processes] == [0] * world_size


@pytest.mark.parametrize("world", [2, 4])
def test_tp_mxfp8_kda_projections_match_stock(world):
    """W1 + W2 on ``world`` Rubin ranks (module docstring)."""
    if not _rubin(world):
        pytest.skip(f"needs {world} Rubin (SM107) GPUs with NVLink multicast")
    pytest.importorskip("cutlass")
    _distributed_run(_worker, world_size=world)
