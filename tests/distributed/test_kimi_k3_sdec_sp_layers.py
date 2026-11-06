# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 SP specialized decode, per layer, against the stock layer on one node
(4 Rubin GPUs: TP4 = DCP4 = the SP group, DP1; C512's decode attention at full
size: hidden 7168, 96 heads (24 per rank), q4 DSpark verify rows R = 4 n, n = 1..128
requests per DP group, 0.25..32 per GPU).

Every case builds a real stock layer (``KimiK3DSAAttention`` on FLASHINFER_MLA_SPARSE
trtllm-gen, per-tensor FP8 MLA KV, MXFP4 or FP8 indexer; ``KimiK3DeltaAttention``
with an FP32 state and RecoverSSM records, ``--use-replayssm``), random weights
(each rank its own shard), and runs one decode step through

  stock   sp_all_gather(x_shard)[:R] -> layer forward (o_proj not reducing, as the
          SP decoder sets it) -> sp_reduce_scatter      (the stock decoder's SP seam)
  ours    runtime.sp_attention.SpAttentionBlock.run(x_shard): SpComm all-gather ->
          DecodeProjection -> SpDsaCore / SpKdaCore -> SpOproj with its 4-rank
          reduce-scatter

on the same input rows, the same metadata and each on its own copy of the caches:

* ``test_sp_dsa_block_matches_stock_layer``: the gated attention rows entering
  o_proj and this rank's output shard (rows whose global length fits the 2048-key
  top-k select every key on both paths: within BF16 reduction order; all rows: the
  error distribution, which also admits the two paths' exact top-k near-ties), the
  new MLA rows (FP8 codes one step apart where the RMSNorm rounds differently, the
  rope part exact), the new indexer keys (few codes different) and every other cache
  byte as it was; then the block replayed in a CUDA graph bit for bit as its eager
  run. Contexts per request cycle through short (all keys) and long ones, one of
  128K tokens. The top-k runs the #63 merge or, as C512 deploys it (VLLM_K31_DCP_
  EXCHANGE=1, parts topk, one round to 256 rows), DcpExchange on both paths.
* ``test_sp_kda_block_matches_stock_layer`` (two steps) and
  ``test_sp_kda_recoverssm_multistep`` (``CHAIN_STEPS`` steps, the specialized block
  replayed in one CUDA graph): RecoverSSM over consecutive steps of the same
  requests with partial acceptance (random 1..4 accepted per request and step, the
  stock Triton commit after each step on each path): per step the records (keys,
  raw gates, corrections), the conv window, the o_norm-gated rows entering o_proj and
  the shard; after each commit the committed FP32 state of every request. The two
  paths' in-projections round differently (BF16 GEMM order), so everything is
  compared within BF16 rounding bounds; the committed state's error must not grow
  past its bound over the steps.
* every case checks the startup coverage of its blocks (each R a specialized
  choice) and that refused configurations raise ``shapes.Unsupported`` with the
  cause: 192 gathered DSA heads, a DCP2 context on the DCP4 box, a BF16 KDA state,
  a non-SP o-projection, a row past the mailboxes.

Rows: ``*_ROWS`` (a subset spanning every policy seam) per test; the ``slow_test``
variants sweep all 128 captured R. Tolerances are BF16 / FP32 reduction-order bounds
(the stock reduce-scatter sums BF16 partials over NCCL; ours in FP32, rounded once).
Needs 4 Rubin (SM107) GPUs with NVLink multicast; CuTe compiles dominate a cold
cache."""

import os
import time
import zlib
from types import SimpleNamespace

import pytest
import torch

from vllm.utils.network_utils import get_open_port

WORLD, Q, HEADS, WIDTH = 4, 4, 24, 7168
CAP = 512  # max_num_seqs 128 x q4
ALL_ROWS = tuple(range(4, CAP + 1, 4))
# DSA: front / fork query overlap (16 | 20), the 64-row seam, DcpExchange's one /
# two rounds (256 | 260), the cap.
DSA_ROWS = (4, 16, 20, 64, 68, 128, 256, 260, 512)
# KDA record cores by requests n (cutedsl.kda.sp.RECORD_KINDS at 24 heads): KdaSplit
# n1, KdaWS n8, KdaWide n16, KdaStream n32 / n35, KdaWS n81 / n128.
KDA_ROWS = (4, 32, 64, 128, 140, 324, 512)
CHAIN_ROWS, CHAIN_STEPS = (4, 64, 140, 512), 8
TOPK, UNIT, EPS = 2048, 64, 1e-6
FP8_ROW, INDEX_ROWS = 576, {"fp8": 132, "mxfp4": 68}
MAX_LEN = 1 << 17
LONG = MAX_LEN - 64  # the u128k request
SHORT = (600, 1500, 2050, 4321, 3000, 9000)
# C512's decode environment (its stock DSA top-k exchange) and conv layout.
DEPLOYED_ENV = {"VLLM_SSM_CONV_STATE_LAYOUT": "DS"}
EXCHANGE_ENV = {
    "VLLM_K31_DCP_EXCHANGE": "1",
    "VLLM_K31_DCP_EXCHANGE_PARTS": "topk",
    "VLLM_K31_DCP_EXCHANGE_ONE_ROUND_ROWS": "256",
    "K31_DCPX_PDL": "0",
}
TIMEOUT = 3600  # one launch, every rank (CuTe compiles on a cold cache)


def _has_multicast_support() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from torch._C._autograd import DeviceType
        from torch._C._distributed_c10d import _SymmetricMemory

        return _SymmetricMemory.has_multicast_support(DeviceType.CUDA, 0)
    except Exception:
        return False


def _rubin_node() -> bool:
    if not torch.cuda.is_available() or torch.cuda.device_count() < WORLD:
        return False
    capabilities = {torch.cuda.get_device_capability(i) for i in range(WORLD)}
    return capabilities == {(10, 7)} and _has_multicast_support()


pytestmark = [
    pytest.mark.distributed(num_gpus=WORLD),
    pytest.mark.skipif(
        not _rubin_node(),
        reason="needs 4 Rubin (SM107) GPUs with NVLink multicast (C512's TP4 group)",
    ),
]


def _generator(*key, device="cuda"):
    """The same stream on every rank for ``key`` (inputs shared by the TP group)."""
    gen = torch.Generator(device=device)
    gen.manual_seed(zlib.crc32(repr(key).encode()))
    return gen


def _rows_error(got, want):
    """Per-row relative L2 error of two [rows, width] tensors."""
    got, want = got.double(), want.double()
    return (got - want).norm(dim=-1) / want.norm(dim=-1).clamp_min(1e-30)


def _msg(prefix):
    """An ``assert_close`` message naming the case."""
    return lambda message: f"{prefix}: {message}"


def _keep_input(kept):
    """A forward pre-hook keeping a copy of the module's input (the rows entering
    the stock o_proj)."""

    def hook(module, args):
        kept.append(args[0].clone())

    return hook


def _check_rows(name, got, want, exact=None, median=1e-2, tail=3e-2, worst=0.15):
    """``exact`` rows (both paths over the same keys and inputs) within BF16 order;
    all rows: the median and 98th percentile of the per-row error, and no row far
    off (a near-tie top-k swap moves a row by one key's weight)."""
    errors = _rows_error(got, want)
    if exact is None:
        exact = torch.zeros_like(errors, dtype=torch.bool)
    if exact.any():
        torch.testing.assert_close(
            got[exact].float(), want[exact].float(), atol=2e-2, rtol=2e-2,
            msg=_msg(f"{name} (rows over the same keys)"),
        )  # fmt: skip
        assert float(errors[exact].max()) <= 2e-2, (name, errors[exact].max())
    quantiles = torch.quantile(errors.float(), torch.tensor([0.5, 0.98]).cuda())
    print(
        f"{name}: per-row error median {quantiles[0]:.2e} p98 {quantiles[1]:.2e} max "
        f"{errors.max():.2e} ({int(exact.sum())} of {exact.numel()} exact)"
    )
    assert quantiles[0] <= median and quantiles[1] <= tail, (name, quantiles)
    assert float(errors.max()) <= worst, (name, errors.max())


# --------------------------------------------------------------------------- launch


def _launch(case: str, *args) -> None:
    """``case`` on 4 spawned ranks (one per GPU); every rank must exit cleanly within
    one deadline (a hung mailbox fails the test, never CI)."""
    import torch.multiprocessing as mp

    port = get_open_port()
    context = mp.get_context("spawn")
    processes = [
        context.Process(target=_worker, args=(rank, port, case, args))
        for rank in range(WORLD)
    ]
    for process in processes:
        process.start()
    deadline = time.monotonic() + TIMEOUT
    for process in processes:
        process.join(timeout=max(0.0, deadline - time.monotonic()))
    hung = [p.pid for p in processes if p.is_alive()]
    for process in processes:
        if process.is_alive():
            process.kill()
            process.join()
    assert not hung, f"{case}: ranks {hung} did not finish in {TIMEOUT} s (a hang?)"
    assert [p.exitcode for p in processes] == [0] * WORLD, case


def _worker(rank: int, port: int, case: str, args: tuple) -> None:
    """One TP4 = DCP4 rank: the vLLM groups (and their symmetric memory), the
    workspace and the DCP top-k merge's buffers, then ``case``."""
    env = dict(DEPLOYED_ENV)
    if case == "dsa" and args[1] == "exchange":
        env.update(EXCHANGE_ENV)
    os.environ.update(env)
    torch.accelerator.set_device_index(rank)
    device = torch.device("cuda", rank)

    from vllm.config import set_current_vllm_config
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.platforms import current_platform
    from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager

    current_platform.import_kernels()
    import vllm.kernels.vllm_c  # noqa: F401  (the vllm_c RMSNorm provider)

    config = _dsa_config(args[0]) if case == "dsa" else _kda_config()
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=WORLD,
            rank=rank,
            local_rank=rank,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
        )
        initialize_model_parallel(
            tensor_model_parallel_size=WORLD, decode_context_model_parallel_size=WORLD
        )
        group = get_tp_group().device_group
        try:  # (older torch: symmetric memory is enabled per group)
            from torch.distributed._symmetric_memory import enable_symm_mem_for_group

            enable_symm_mem_for_group(group.group_name)
        except (ImportError, AttributeError):
            pass
        init_workspace_manager(device)
        try:
            if case == "dsa":
                from vllm.v1.attention.ops.dcp_topk_merge import init_dcp_topk_merge

                assert init_dcp_topk_merge(config, TOPK)
                _dsa_case(config, rank, device, *args)
            else:
                _kda_case(config, rank, device, *args)
        finally:
            torch.accelerator.synchronize()
            reset_workspace_manager()
    cleanup_dist_env_and_memory()


def _context(rank, device, **overrides):
    from vllm.models.kimi_k3.nvidia.specialized_decode.shapes import Context

    fields = dict(
        arch="rubin", tp=WORLD, rank=rank, dcp=WORLD,
        sms=torch.cuda.get_device_properties(device).multi_processor_count, q=Q,
        max_rows=CAP, max_context=MAX_LEN, bf16_state=False, dp=1, ep=WORLD,
        sequence_parallel=True, heads_per_rank=HEADS,
    )  # fmt: skip
    fields.update(overrides)
    return Context(**fields)


class _Step:
    """The parts of ``sp_attention.SpStep`` an ``SpAttentionBlock`` reads at build:
    the context, rows, mailboxes, DSA box / scratch, records mode and the shared
    scratch buffers (``SpStep.scratch``)."""

    def __init__(self, ctx, rows, device, comm, records="none", box=None, dsa=None):
        self.ctx, self.rows, self.device, self.comm = ctx, rows, device, comm
        self.q, self.records = ctx.q, records
        self.dsa_box, self.dsa_scratch, self.dsa_layers = box, dsa, 0
        self.buffers: dict = {}
        self.shard_cap = rows[-1] // ctx.tp

    def scratch(self, name, shape):
        from vllm.models.kimi_k3.nvidia.specialized_decode.runtime import (
            sp_attention,
        )

        return sp_attention.SpStep.scratch(self, name, shape)


def _sp_block(layer_idx, attention, step):
    """The SP block of one stock layer (ordinal 0: parity 0), and the shared
    coverage check: every row of the step a specialized (not stock) choice."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.runtime import sp_attention
    from vllm.models.kimi_k3.nvidia.specialized_decode.shapes import Path

    decoder = SimpleNamespace(layer_idx=layer_idx, self_attn=attention)
    block = sp_attention.SpAttentionBlock(decoder, step, 0)
    for r in step.rows:
        choices = {"ag": step.comm.choice(r), **block.choices(r)}
        assert set(choices) == {"ag", f"{block.kind}_in", block.kind, "oproj_rs"}
        stock = [
            f for f, c in choices.items() if c.path not in (Path.GENERIC, Path.TUNED)
        ]
        assert not stock, (r, stock)
    return block


def _barrier():
    """All ranks done with the previous kernels (a TP collective between two SP
    blocks of the same mailbox parity, as the model's final sp_all_gather is)."""
    import torch.distributed as dist

    from vllm.distributed import get_tp_group

    torch.accelerator.synchronize()
    dist.barrier(group=get_tp_group().cpu_group)


# ------------------------------------------------------------------------------ DSA


def _dsa_config(indexer):
    """The DSA layer's vLLM config at TP4 = DCP4 (interleave 1), FLASHINFER_MLA_
    SPARSE trtllm-gen on per-tensor FP8 KV, the ``indexer`` key cache, q4 verify
    rows; the metadata builders read DeepSeek-V2-Lite's MLA sizes (kv_lora 512, rope
    64) as test_specialized_decode_dsa.py's layer harness does."""
    from tests.models.kimi_k3 import test_specialized_decode_dsa as dsa
    from tests.v1.attention.utils import create_vllm_config
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    config = create_vllm_config(
        model_name=dsa.LAYER_MODEL,
        max_model_len=MAX_LEN,
        block_size=UNIT,
        max_num_seqs=CAP // Q,
        max_num_batched_tokens=8192,
        hf_config_override={"index_topk": TOPK},
    )
    parallel = config.parallel_config
    parallel.tensor_parallel_size = parallel.world_size = WORLD
    parallel.decode_context_parallel_size = WORLD
    parallel.cp_kv_cache_interleave_size = 1
    config.cache_config.cache_dtype = "fp8"
    config.attention_config.backend = AttentionBackendEnum.FLASHINFER_MLA_SPARSE
    config.attention_config.indexer_kv_dtype = indexer
    config.kernel_config.flashinfer_sparse_mla_kernel = "trtllm"
    config.speculative_config = SimpleNamespace(
        num_speculative_tokens=Q - 1,
        enable_adaptive_verification=False,
        parallel_drafting=False,
    )
    return config


def _dsa_attention(config, rank, topk_buffer):
    """A K3.1 DSA layer at C512's sizes (96 heads over TP4, hidden 7168, 32 index
    heads of 128), random weights (this rank's shards), MLA weights processed, the
    SP decoder's non-reducing o_proj."""
    from vllm.models.kimi_k3.nvidia.dsa import KimiK3DSAAttention
    from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig

    text = KimiLinearConfig(
        hidden_size=WIDTH, num_attention_heads=WORLD * HEADS, q_lora_rank=1536,
        kv_lora_rank=512, qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128,
        mla_use_nope=True, mla_use_output_gate=True, rms_norm_eps=EPS,
        dsa_config={
            "dsa_layers": [4], "index_topk": TOPK, "index_head_dim": 128,
            "index_num_heads": 32, "index_rope_dim": 0,
            "index_k_norm_type": "layernorm",
        },
    )  # fmt: skip
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("cuda"):
            attn = KimiK3DSAAttention(
                text, hidden_size=WIDTH, num_heads=WORLD * HEADS, qk_nope_head_dim=128,
                qk_rope_head_dim=64, v_head_dim=128, q_lora_rank=1536,
                kv_lora_rank=512, cache_config=config.cache_config,
                prefix="model.layers.3.self_attn", topk_indices_buffer=topk_buffer,
            )  # fmt: skip
    finally:
        torch.set_default_dtype(torch.float32)
    torch.manual_seed(5 + rank)
    with torch.no_grad():
        for name, param in attn.named_parameters():
            if name.endswith("bias"):
                param.copy_(torch.randn_like(param) / 10)
            elif "norm" in name:
                param.copy_(torch.rand_like(param) + 0.5)
            else:
                param.copy_(torch.randn_like(param) * param.shape[-1] ** -0.5)
    attn.mla_attn.mla_attn.process_weights_after_loading(torch.bfloat16)
    attn.o_proj.reduce_results = False  # (KimiDecoderLayer under SP)
    return attn


def _page_table(lengths, blocks, gen):
    """Each request's pages over a fixed ``blocks``-page cache, in random order."""
    pages = [max(-(-n // UNIT), 1) for n in lengths]
    assert sum(pages) <= blocks, (sum(pages), blocks)
    order = torch.randperm(blocks, generator=gen, device="cuda").int()
    table = torch.zeros(len(lengths), max(pages), dtype=torch.int32, device="cuda")
    start = 0
    for r, n in enumerate(pages):
        table[r, :n] = order[start : start + n]
        start += n
    return table


def _contexts(reqs):
    """Tokens before the step per request: short ones (all keys fit the top-k),
    long ones and one of 128K (from two requests on)."""
    return [LONG if r == 1 else SHORT[r % len(SHORT)] + 37 * r for r in range(reqs)]


def _local(contexts, rank):
    return [(c + Q - rank + WORLD - 1) // WORLD for c in contexts]


def _pages(cache, row):
    """Each packed indexer page's bytes [blocks, UNIT x row]."""
    return torch.as_strided(
        cache, (cache.shape[0], cache.shape[1] * row), (cache.stride(0), 1)
    )


def _index_token_bytes(cache, slots, row, key_bytes):
    """(page, byte offset) of every byte the tokens at ``slots`` own in a packed
    indexer cache: their key codes, then their 4 scale bytes."""
    page, entry = (slots // UNIT).long(), (slots % UNIT).long()
    keys = key_bytes * entry.view(-1, 1) + torch.arange(key_bytes, device="cuda")
    scales = UNIT * key_bytes + 4 * entry.view(-1, 1) + torch.arange(4, device="cuda")
    return page, torch.cat([keys, scales], 1)


def _dsa_case(config, rank, device, indexer, topk_mode, rows, refusals):
    from tests.models.kimi_k3 import test_specialized_decode_dsa as dsa
    from vllm import _custom_ops as ops
    from vllm import ir
    from vllm.distributed import get_tp_group
    from vllm.distributed.parallel_state import graph_capture
    from vllm.forward_context import create_forward_context, override_forward_context
    from vllm.models.common.ops.sequence_parallel import (
        sp_all_gather,
        sp_reduce_scatter,
    )
    from vllm.models.kimi_k3.nvidia.ops.dsa_indexer import indexer_k_norm_quant_insert
    from vllm.models.kimi_k3.nvidia.ops.dsa_indexer_mxfp4 import (
        mxfp4_k_norm_quant_insert,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.sp_comm import (
        SpComm,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.sp import (
        sp_dsa_box,
        sp_dsa_scratch,
    )
    from vllm.v1.attention.backends.mla.indexer import (
        DeepseekV32IndexerMetadataBuilder,
    )
    from vllm.v1.worker.block_table import get_block_table_width

    bf = torch.bfloat16
    index_row = INDEX_ROWS[indexer]
    key_bytes = index_row - 4
    insert = mxfp4_k_norm_quant_insert if indexer == "mxfp4" else (
        indexer_k_norm_quant_insert
    )  # fmt: skip
    topk_buffer = torch.full((8192, TOPK), -1, dtype=torch.int32, device=device)
    attn = _dsa_attention(config, rank, topk_buffer)
    layer, idx = attn.mla_attn.mla_attn, attn.dsa_indexer
    if topk_mode == "exchange":  # C512's stock hook, attached as the model does
        from vllm.models.kimi_k3.nvidia.dsa_kernels.dcp_exchange.decode import (
            DcpxDecode,
        )

        dcpx = DcpxDecode(config, device, WORLD * HEADS, idx.topk_tokens)
        assert "topk" in dcpx.parts, dcpx.parts
        dcpx.attach(attn)
        dcpx.warm()
        dcpx.serve()

    # One cache per path, sized for the widest case (the kernels compile for it).
    widest = max(rows) // Q
    blocks = sum(max(-(-n // UNIT), 1) for n in _local(_contexts(widest), rank)) + 2
    caches = {
        path: (
            dsa.packed_cache(blocks, UNIT, FP8_ROW, UNIT),
            dsa.packed_cache(blocks, UNIT, index_row, dsa.GAP),
        )
        for path in ("stock", "ours")
    }
    spec = layer.get_kv_cache_spec(config)
    builder = layer.attn_backend.get_builder_cls()(
        spec, [layer.layer_name], config, device
    )
    index_spec = idx.k_cache.get_kv_cache_spec(config)
    index_builder = DeepseekV32IndexerMetadataBuilder(
        index_spec, [idx.k_cache.prefix], config, device,
        block_table_width=get_block_table_width(
            index_spec.max_num_blocks_per_req(config, MAX_LEN), UNIT
        ),
    )  # fmt: skip
    names = layer.layer_name, idx.k_cache.prefix

    def bind(path):
        layer.kv_cache, idx.k_cache.kv_cache = caches[path]

    # The SP block, built (collectively) once for every case's rows.
    ctx = _context(rank, device)
    group = get_tp_group().device_group
    comm = SpComm(group, device, WORLD, rank, width=WIDTH, rows_cap=CAP)
    box = sp_dsa_box(group, device, rank, heads=HEADS, cap=CAP, interleave=1)
    step = _Step(
        ctx, tuple(rows), device, comm, box=box, dsa=sp_dsa_scratch(device, CAP)
    )
    bind("ours")
    block = _sp_block(3, attn, step)
    block.core.prepare_cache(caches["ours"][0])

    for r in rows:
        reqs, n = r // Q, r // WORLD
        gen = _generator("dsa", r, rank)  # this rank's pages and old keys
        shared = _generator("dsa-step", r)  # the step's inputs (every rank)
        contexts = _contexts(reqs)
        local = _local(contexts, rank)
        mla_table = _page_table(local, blocks, gen)
        index_table = _page_table(local, blocks, gen)
        old = [range(c) for c in contexts]
        old_mla = dsa.owned_slots(mla_table, old, WORLD, rank)
        old_index = dsa.owned_slots(index_table, old, WORLD, rank)
        stock_mla, stock_index = caches["stock"]
        stock_mla.zero_()
        stock_index.zero_()
        count = old_mla.numel()
        unit = torch.ones(1, dtype=torch.float32, device=device)
        kv_c = torch.randn(count, 512, device=device, dtype=bf, generator=gen)
        k_pe = torch.randn(count, 64, device=device, dtype=bf, generator=gen)
        ops.concat_and_cache_mla(kv_c, k_pe, stock_mla, old_mla, "fp8", unit)
        norm = torch.ones(128, dtype=bf, device=device)
        chunk = 16384
        scratch = torch.empty(chunk, TOPK, dtype=torch.int32, device=device)
        for start in range(0, count, chunk):
            keys = torch.randn(
                min(chunk, count - start), 128, device=device, dtype=bf, generator=gen
            )
            rows_here = keys.shape[0]
            insert(
                keys, norm, torch.zeros_like(norm), EPS,
                old_index[start : start + rows_here], stock_index,
                scratch[:rows_here], rows_here,
            )  # fmt: skip
        for ours, theirs in zip(caches["ours"], caches["stock"]):
            ours.copy_(theirs)
        new = [range(c, c + Q) for c in contexts]
        new_mla = dsa.owned_slots(mla_table, new, WORLD, rank)
        new_index = dsa.owned_slots(index_table, new, WORLD, rank)
        md = builder.build(
            0, dsa.common_metadata(contexts, Q, mla_table, new_mla, WORLD, rank)
        )
        index_md = index_builder.build(
            0, dsa.common_metadata(contexts, Q, index_table, new_index, WORLD, rank)
        )
        forward = create_forward_context(
            dict(zip(names, (md, index_md))),
            config,
            slot_mapping=dict(zip(names, (new_mla, index_md.slot_mapping))),
        )
        x = torch.randn(r, WIDTH, device=device, dtype=bf, generator=shared)
        x_shard = x[rank * n : (rank + 1) * n].contiguous()
        positions = torch.tensor([p for pos in new for p in pos], device=device)
        lengths = torch.tensor(
            [c + j + 1 for c in contexts for j in range(Q)], device=device
        )
        exact = lengths <= TOPK

        # Stock: the SP decoder's seam around the layer forward.
        entering: list = []
        hook = attn.o_proj.register_forward_pre_hook(_keep_input(entering))
        bind("stock")
        # (its RMSNorms as a worker without inductor runs them: vllm_c, one rounding)
        norms = ir.ops.rms_norm.set_priority(["vllm_c", "native"])
        with override_forward_context(forward), norms:
            gathered = sp_all_gather(x_shard)[:r]
            want = sp_reduce_scatter(attn(positions=positions, hidden_states=gathered))
        hook.remove()

        # Ours: the SP block on the same rows, metadata and its own caches.
        bind("ours")
        run = SimpleNamespace(slot_mapping={layer.layer_name: new_mla}, q=Q)
        with override_forward_context(forward):
            got = block.run(x_shard, [caches["ours"][0]], md, run)
        torch.accelerator.synchronize()
        where = f"DSA R{r} ({indexer}, {topk_mode}) rank {rank}"
        _check_rows(f"{where} o_proj input", block.attn_out[:r], entering[0], exact)
        _check_rows(f"{where} shard", got, want, exact[rank * n : (rank + 1) * n])

        # The step's cache writes: this rank's new MLA rows and indexer keys.
        (ours_mla, ours_index), (stock_mla, stock_index) = (
            caches["ours"],
            caches["stock"],
        )
        written = new_mla[new_mla >= 0]
        at = written // UNIT, written % UNIT
        a, b = ours_mla[at], stock_mla[at]
        assert torch.equal(a[:, 512:], b[:, 512:]), f"{where}: rope part"
        assert dsa.fp8_codes_adjacent(a[:, :512], b[:, :512]), where
        assert (a[:, :512] != b[:, :512]).sum() <= 0.05 * a[:, :512].numel(), where
        ours_mla[at], stock_mla[at] = 0, 0
        assert torch.equal(ours_mla, stock_mla), f"{where}: other MLA bytes"
        keyed = new_index[new_index >= 0]
        page, offsets = _index_token_bytes(ours_index, keyed, index_row, key_bytes)
        pages = _pages(ours_index, index_row), _pages(stock_index, index_row)
        a, b = (p[page.view(-1, 1), offsets] for p in pages)
        assert (a != b).sum() <= 0.05 * a.numel(), f"{where}: new indexer keys"
        for p in pages:
            p[page.view(-1, 1), offsets] = 0
        assert torch.equal(*pages), f"{where}: other indexer bytes"
        for ours, theirs in zip(caches["ours"], caches["stock"]):
            ours.copy_(theirs)  # (the new rows as stock wrote them, for the replay)

        # The block captured and replayed: the eager run's shard bit for bit.
        eager = got.clone()
        _barrier()
        graph = torch.cuda.CUDAGraph()
        with (
            override_forward_context(forward),
            graph_capture(device) as capture,
            torch.cuda.graph(graph, stream=capture.stream),
        ):
            replayed = block.run(x_shard, [caches["ours"][0]], md, run)
        _barrier()
        graph.replay()
        torch.accelerator.synchronize()
        assert torch.equal(replayed, eager), (where, (replayed - eager).abs().max())
        _barrier()

    if refusals:
        _dsa_refusals(attn, block, step, rank, device)


def _dsa_refusals(attn, block, step, rank, device):
    """Configurations the SP DSA path refuses at build, naming the cause."""
    from vllm.distributed import get_tp_group
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.dsa.sp import (
        SpDsaCore,
        sp_dsa_box,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.shapes import Unsupported

    group = get_tp_group().device_group
    with pytest.raises(Unsupported, match="192 gathered query heads"):
        sp_dsa_box(group, device, rank, heads=48, cap=CAP, interleave=1)
    dcp2 = _context(rank, device, dcp=2)
    with pytest.raises(Unsupported, match=r"DCP4 box under TP4 / DCP2"):
        SpDsaCore(
            attn, step.rows, step.dsa_scratch, box=step.dsa_box, parity=1, ctx=dcp2
        )
    with pytest.raises(Unsupported, match="above the DcpBox's 512 rows"):
        SpDsaCore(
            attn, (*step.rows, 516), step.dsa_scratch, box=step.dsa_box, parity=1,
            ctx=_context(rank, device),
        )  # fmt: skip
    with pytest.raises(Unsupported, match="not a prepared row count"):
        block.core.choice(step.rows[0] + 2)


@pytest.mark.parametrize("topk", ["merge", "exchange"])
@pytest.mark.parametrize("indexer", ["mxfp4", "fp8"])
def test_sp_dsa_block_matches_stock_layer(indexer, topk):
    """SpAttentionBlock (DSA) vs the stock KimiK3DSAAttention under the SP seam at
    ``DSA_ROWS`` (module docstring); C512 deploys the MXFP4 indexer with DcpExchange."""
    pytest.importorskip("cutlass")
    _launch("dsa", indexer, topk, DSA_ROWS, True)


@pytest.mark.slow_test
def test_sp_dsa_block_matches_stock_layer_all_rows():
    """The DSA A/B at every captured R (4..512), as C512 deploys it."""
    pytest.importorskip("cutlass")
    _launch("dsa", "mxfp4", "exchange", ALL_ROWS, False)


# ------------------------------------------------------------------------------ KDA


def _kda_config():
    """The KDA layer's vLLM config: TP4 = DCP4, a BF16 conv / FP32 recurrent state,
    RecoverSSM records (--use-replayssm with q4 DSpark), 128 requests; the stock
    layer's Triton prefill (decode is what is compared)."""
    from tests.models.kimi_k3 import test_specialized_decode_dsa as dsa
    from tests.v1.attention.utils import create_vllm_config

    config = create_vllm_config(
        model_name=dsa.LAYER_MODEL,
        max_model_len=MAX_LEN,
        block_size=UNIT,
        max_num_seqs=CAP // Q,
        max_num_batched_tokens=8192,
    )
    parallel = config.parallel_config
    parallel.tensor_parallel_size = parallel.world_size = WORLD
    parallel.decode_context_parallel_size = WORLD
    parallel.cp_kv_cache_interleave_size = 1
    cache = config.cache_config
    cache.mamba_ssm_cache_dtype = "float32"
    cache.use_replayssm = True
    cache.use_kda_recoverssm = True
    config.additional_config = {"kda_prefill_backend": "triton"}
    config.speculative_config = SimpleNamespace(
        num_speculative_tokens=Q - 1,
        enable_adaptive_verification=False,
        parallel_drafting=False,
    )
    return config


def _kda_attention(config, rank):
    """A K3.1 KDA layer at C512's sizes (96 heads over TP4, hidden 7168, width-4
    conv, bounded gate), random weights (this rank's shards; the in-projection's
    pad rows zero, as the loader leaves them), the SP decoder's o_proj."""
    from vllm.models.kimi_k3.nvidia.kda import KimiK3DeltaAttention
    from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig

    text = KimiLinearConfig(
        hidden_size=WIDTH, rms_norm_eps=1e-5, hidden_act="silu",
        linear_attn_config={
            "kda_layers": [2], "full_attn_layers": [4], "head_dim": 128,
            "num_heads": WORLD * HEADS, "short_conv_kernel_size": 4,
            "use_full_rank_gate": True, "gate_lower_bound": -5.0,
        },
    )  # fmt: skip
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("cuda"):
            attn = KimiK3DeltaAttention(text, config, prefix="model.layers.1.self_attn")
    finally:
        torch.set_default_dtype(torch.float32)
    assert attn.use_recoverssm and attn.recoverssm_backend == "triton"
    g = torch.Generator(device="cuda").manual_seed(11 + rank)

    def randn(t, scale):
        return torch.randn(t.shape, generator=g, device="cuda") * scale

    dim = HEADS * 128
    with torch.no_grad():
        w = attn.in_proj_qkvgfab.weight
        w.copy_(randn(w, WIDTH**-0.5))
        if attn.in_proj_padding:
            w[-attn.in_proj_padding :] = 0
        attn.f_b_proj.weight.copy_(randn(attn.f_b_proj.weight, 128**-0.5))
        attn.conv1d.weight.copy_(randn(attn.conv1d.weight, 0.5))
        attn.dt_bias.copy_(randn(attn.dt_bias, 0.5))
        attn.A_log.copy_(randn(attn.A_log, 0.5))
        attn.o_norm.weight.copy_(1 + randn(attn.o_norm.weight, 0.1))
        attn.o_proj.weight.copy_(randn(attn.o_proj.weight, dim**-0.5))
        if attn.decode_conv1d_weight is not None:  # (the loader's decode copies)
            conv = attn.conv1d.weight.squeeze(1)
            for i in range(3):
                attn.decode_conv1d_weight[i].copy_(conv[i * dim : (i + 1) * dim].T)
        if attn.decode_norm_weight is not None:
            attn.decode_norm_weight.copy_(attn.o_norm.weight)
    attn.o_proj.reduce_results = False  # (KimiDecoderLayer under SP)
    return attn


def _kda_caches(slots, gen):
    """The DS conv window [slot, 3 dim, q + 2] (BF16) and the FP32 state."""
    dim = HEADS * 128
    conv = torch.randn(slots, 3 * dim, Q + 2, generator=gen, device="cuda")
    state = torch.randn(slots, HEADS, 128, 128, generator=gen, device="cuda") * 0.1
    return [conv.to(torch.bfloat16), state]


def _kda_case(config, rank, device, rows, steps, graph, refusals):
    from vllm.distributed import get_tp_group
    from vllm.distributed.parallel_state import graph_capture
    from vllm.forward_context import create_forward_context, override_forward_context
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
    from vllm.models.common.ops.sequence_parallel import (
        sp_all_gather,
        sp_reduce_scatter,
    )
    from vllm.models.kimi_k3.nvidia.kda_metadata import KimiK3KDAMetadata
    from vllm.models.kimi_k3.nvidia.ops.recoverssm import KDARecoverSSMCommitContext
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.sp_comm import (
        SpComm,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.recoverssm import (
        prepare_recoverssm,
    )

    assert is_conv_state_dim_first()  # C512's DS conv layout
    attn = _kda_attention(config, rank)
    ctx = _context(rank, device)
    decoder = SimpleNamespace(layer_idx=1, self_attn=attn)
    records = prepare_recoverssm([decoder], ctx, defer=False)
    assert records == "triton", records
    widest = max(rows) // Q
    slots = 2 * widest + 2
    gen = _generator("kda-caches", rank)
    caches = {"stock": _kda_caches(slots, gen)}
    caches["ours"] = [t.clone() for t in caches["stock"]]
    group = get_tp_group().device_group
    comm = SpComm(group, device, WORLD, rank, width=WIDTH, rows_cap=CAP)
    step = _Step(ctx, tuple(rows), device, comm, records=records)
    block = _sp_block(1, attn, step)
    block.core.prepare_caches(caches["ours"])

    def metadata(reqs, checkpoints):
        r = reqs * Q
        return KimiK3KDAMetadata(
            num_prefills=0, num_prefill_tokens=0, num_decodes=0, num_decode_tokens=0,
            num_spec_decodes=reqs, num_spec_decode_tokens=r, num_actual_tokens=r,
            spec_query_start_loc=torch.arange(
                0, r + 1, Q, dtype=torch.int32, device=device
            ),
            spec_state_indices_tensor=checkpoints.view(reqs, 1),
            num_accepted_tokens=torch.ones(reqs, dtype=torch.int32, device=device),
            uniform_spec_sequence_length=Q,
        )  # fmt: skip

    def commit(path, accepted, checkpoints, starts):
        attn.kv_cache = caches[path]
        KDARecoverSSMCommitContext.create(
            [attn], spec_query_len=Q, max_num_reqs=CAP // Q, backend="triton"
        ).commit(accepted, checkpoints, starts)

    rec = attn.recoverssm_records
    for r in rows:
        reqs, n = r // Q, r // WORLD
        # Both paths start every case from the same states (the previous case's
        # commits left them within its bounds, not equal).
        for ours, theirs in zip(caches["ours"], caches["stock"]):
            ours.copy_(theirs)
        order = torch.randperm(
            slots - 1, generator=_generator("slots", r, device="cpu")
        )
        checkpoints = (order[:reqs] + 1).to(torch.int32).to(device)
        md = metadata(reqs, checkpoints)
        starts = md.spec_query_start_loc
        forward = create_forward_context({attn.prefix: md}, config)
        x_shard = torch.empty(n, WIDTH, dtype=torch.bfloat16, device=device)
        run = SimpleNamespace(slot_mapping={}, q=Q)
        replay = replayed = None
        if graph:  # the specialized block captured once, replayed every step
            _barrier()
            replay = torch.cuda.CUDAGraph()
            with (
                graph_capture(device) as capture,
                torch.cuda.graph(replay, stream=capture.stream),
            ):
                replayed = block.run(x_shard, caches["ours"], md, run)
            _barrier()
        state_errors = []
        for t in range(steps):
            where = f"KDA R{r} step {t} rank {rank}"
            x = torch.randn(
                r, WIDTH, dtype=torch.bfloat16, device=device,
                generator=_generator("kda-x", r, t),
            )  # fmt: skip
            x_shard.copy_(x[rank * n : (rank + 1) * n])
            # Stock: the SP decoder's seam around the layer forward (its records).
            entering: list = []
            hook = attn.o_proj.register_forward_pre_hook(_keep_input(entering))
            attn.kv_cache = caches["stock"]
            with override_forward_context(forward):
                gathered = sp_all_gather(x_shard)[:r]
                positions = torch.zeros(r, dtype=torch.long, device=device)
                want = sp_reduce_scatter(
                    attn(hidden_states=gathered, positions=positions)
                )
            hook.remove()
            stock_rec = (rec.correction[:reqs].clone(), rec.key_gate[:reqs].clone())
            # Ours: the SP block (its records into the layer's record rows).
            if replay is not None:
                replay.replay()
                got = replayed
            else:
                with override_forward_context(forward):
                    got = block.run(x_shard, caches["ours"], md, run)
            torch.accelerator.synchronize()
            # (no top-k: every row is compared, within the in-projections' BF16
            # rounding difference carried through the conv and the recurrence)
            bounds = dict(median=1e-2, tail=2e-2, worst=5e-2)
            _check_rows(
                f"{where} o_proj input", block.attn_out[:r], entering[0], **bounds
            )
            _check_rows(f"{where} shard", got, want, **bounds)
            # Records and conv windows: the two in-projections round differently
            # (BF16 GEMM order), so within BF16 bounds, at the stock row layout.
            correction, key_gate = rec.correction[:reqs], rec.key_gate[:reqs]
            torch.testing.assert_close(
                key_gate.float(), stock_rec[1].float(), rtol=2**-6, atol=2e-2,
                msg=_msg(f"{where} keys / raw gates"),
            )  # fmt: skip
            error = float((correction - stock_rec[0]).norm() / stock_rec[0].norm())
            assert error <= 2e-2, (where, "corrections", error)
            live = checkpoints.long()
            a, b = caches["ours"][0][live].float(), caches["stock"][0][live].float()
            torch.testing.assert_close(
                a, b, rtol=2**-7, atol=1e-2, msg=_msg(f"{where} conv window")
            )
            # Both commit their own records (the same accepted counts).
            accepted = torch.randint(
                1, Q + 1, (reqs,), generator=_generator("accepted", r, t, device="cpu")
            )
            accepted[0] = Q
            if reqs > 1:
                accepted[1] = 1
            accepted = accepted.to(torch.int32).to(device)
            commit("ours", accepted, checkpoints, starts)
            rec.correction[:reqs].copy_(stock_rec[0])
            rec.key_gate[:reqs].copy_(stock_rec[1])
            commit("stock", accepted, checkpoints, starts)
            torch.accelerator.synchronize()
            ours_state, stock_state = caches["ours"][1][live], caches["stock"][1][live]
            error = float(
                (ours_state.double() - stock_state.double()).norm()
                / stock_state.double().norm()
            )
            state_errors.append(error)
            assert error <= 2e-2, (where, "committed state", error)
            others = torch.ones(slots, dtype=torch.bool, device=device)
            others[live] = False
            for i in range(2):  # slots of no request untouched on both paths
                assert torch.equal(
                    caches["ours"][i][others], caches["stock"][i][others]
                )
            _barrier()
        print(f"KDA R{r} rank {rank}: committed state error per step {state_errors}")
        # BF16 input rounding enters every step; the recurrence must not amplify it.
        assert max(state_errors) <= 4 * state_errors[0] + 1e-3, state_errors

    if refusals:
        _kda_refusals(attn, block, step, rank, device)


def _kda_refusals(attn, block, step, rank, device):
    """Configurations the SP KDA path and the SP o-projection refuse at build."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.kda.sp import SpKdaCore
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.oproj.sp import SpOproj
    from vllm.models.kimi_k3.nvidia.specialized_decode.shapes import Unsupported

    bf16 = _context(rank, device, bf16_state=True)
    with pytest.raises(Unsupported, match="FP32 SSM state"):
        SpKdaCore(attn, step.rows, records="triton", ctx=bf16)
    with pytest.raises(Unsupported, match="not whole requests"):
        SpKdaCore(attn, (6,), records="triton", ctx=step.ctx)
    off = _context(rank, device, sequence_parallel=False)
    with pytest.raises(Unsupported, match="SP o-projection at TP4"):
        SpOproj(attn.o_proj.weight, step.comm, step.rows, off)
    with pytest.raises(Unsupported, match=r"not a multiple of TP4 in 1\.\.512"):
        SpOproj(attn.o_proj.weight, step.comm, (516,), step.ctx)
    missing = next(r for r in ALL_ROWS if r not in step.rows)
    with pytest.raises(Unsupported, match="not a served row count"):
        block.core.choice(missing)


def test_sp_kda_block_matches_stock_layer():
    """SpAttentionBlock (KDA, RecoverSSM records) vs the stock KimiK3DeltaAttention
    under the SP seam at ``KDA_ROWS``, two steps with partial acceptance."""
    pytest.importorskip("cutlass")
    _launch("kda", KDA_ROWS, 2, False, True)


def test_sp_kda_recoverssm_multistep():
    """``CHAIN_STEPS`` RecoverSSM steps with partial acceptance, the specialized
    block replayed from one CUDA graph: records, conv window and committed FP32
    state against the stock layer and commit at every step."""
    pytest.importorskip("cutlass")
    _launch("kda", CHAIN_ROWS, CHAIN_STEPS, True, False)


@pytest.mark.slow_test
def test_sp_kda_block_matches_stock_layer_all_rows():
    """The KDA A/B at every captured R (4..512)."""
    pytest.importorskip("cutlass")
    _launch("kda", ALL_ROWS, 2, False, False)
