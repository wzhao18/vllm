# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 specialized decode on CPU: the routing of its wide verify steps (above
the tuned rows) between the specialized providers and the stock seams, K3.1's DSA
seam (S2-DSA) included."""

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

pytest.importorskip("cutlass")
pytest.importorskip("flashinfer")

from vllm.model_executor.layers.linear import UnquantizedLinearMethod  # noqa: E402
from vllm.model_executor.layers.mla import (  # noqa: E402
    MultiHeadLatentAttentionWrapper,
)
from vllm.models.kimi_k3.nvidia.dsa import KimiK3DSAIndexer  # noqa: E402
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe.dense import (  # noqa: E402
    PreparedDenseMLP,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe.prepared import (  # noqa: E402
    PreparedMoE,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.runtime import (  # noqa: E402
    DecodeConfig,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.runtime import (  # noqa: E402
    decode as decode_module,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.runtime import (  # noqa: E402
    providers as providers_module,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.runtime.decode import (  # noqa: E402
    SpecializedDecode,
    Step,
    wide_gate,
    wide_rows,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.runtime.providers import (  # noqa: E402
    BATCHES,
    Attention,
    dsa_seam_columns,
    pack_dsa_projection,
    projection_policies,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.shapes import (  # noqa: E402
    Choice,
    Context,
    Path,
    Shape,
    captured_rows,
)

HIDDEN = 16  # hidden width (7168 in the model): no test below depends on it
# K3.1 attention at TP16 (6 of 96 heads per rank): packed projection rows of
# [q_c 1536 | kv_c 512 + k_pe 64 | gate 768 | wk 128 | wv 32] = 3040 columns.
Q_LORA, KV_LORA, ROPE, NOPE, V_DIM, HEADS = 1536, 512, 64, 128, 128, 6
INDEX_DIM, INDEX_HEADS, PITCH = 128, 32, 3040
COLUMNS = (Q_LORA, KV_LORA + ROPE, HEADS * V_DIM, INDEX_DIM, INDEX_HEADS)
# Decode capture sizes of the AgentX recipes (disagg-p8-d16-variants-pr1524-k31):
# DL7 (DSpark q8, max-num-seqs 32) and DL3 cap-128 (q4, max-num-seqs 128).
DL7 = (7, 8, 14, 16, 21, 24, 28, 32, 35, 40, 42, 48, 49, 56, 63, 64, 70, 72, 77, 80,
       84, 88, 91, 96, 98, 104, 105, 112, 119, 120, 126, 128, 133, 136, 140, 144, 147,
       152, 154, 160, 161, 168, 175, 176, 182, 184, 189, 192, 196, 200, 203, 208, 210,
       216, 217, 224, 232, 240, 248, 256)  # fmt: skip
DL3 = (6, 8, 15, 16, 18, 20, 21, 24, 30, 32, 39, 40, 45, 48, 54, 56, 63, 64, 66, 68,
       69, 72, 75, 76, 78, 80, 81, 84, 87, 88, 90, 92, 93, 96, 100, 102, 104, 108, 112,
       114, 116, 120, 124, 126, 128, 132, 136, 138, 144, 150, 152, 156, 160, 162, 168,
       174, 176, 180, 184, 186, 192, 200, 204, 208, 216, 224, 228, 232, 240, 248, 252,
       256, 264, 272, 276, 288, 300, 304, 312, 320, 324, 336, 348, 352, 360, 368, 372,
       384, 400, 416, 432, 448, 464, 480, 496, 512)  # fmt: skip
# C48's decode role (override_2p1d_dl3_cap64_mc): q4 on the base tree, max-num-seqs 64.
DL3_CAP64 = tuple(m for m in DL3 if m <= 256 and m not in (204, 228, 252))


@pytest.mark.parametrize(
    "dsa,tp,dcp,q,rubin,seams,why",
    [
        (True, 16, 16, 8, True, True, None),  # K3.1's decode role
        (True, 16, 8, 8, True, True, "DSA wide steps need TP16 = DCP16"),
        (True, 16, 2, 8, True, True, "DSA wide steps need TP16 = DCP16"),
        (True, 16, 1, 8, True, True, "DSA wide steps need TP16 = DCP16"),
        (True, 8, 8, 8, True, True, "DSA wide steps need TP16 = DCP16"),
        (True, 8, 1, 8, True, True, "DSA wide steps need TP16 = DCP16"),
        (True, 16, 16, 1, True, True, "no prepared providers above 64 rows"),
        (True, 16, 16, 8, False, True, "no prepared providers above 64 rows"),
        (True, 16, 16, 8, True, False, "sequence parallelism"),
        (False, 16, 16, 8, True, True, None),  # K3 (dense MLA): as before
        (False, 16, 1, 8, True, True, None),
        (False, 8, 8, 8, True, True, None),
        (False, 16, 16, 1, True, True, "no prepared providers above 64 rows"),
        (False, 16, 16, 8, False, True, "no prepared providers above 64 rows"),
        (False, 16, 16, 8, True, False, "sequence parallelism"),
    ],
)
def test_wide_gate(dsa, tp, dcp, q, rubin, seams, why):
    """Verify graphs above 64 rows run the specialized decode on Rubin without
    sequence parallelism; with K3.1's DSA layers only at TP16 = DCP16. Otherwise the
    gate names the reason the coverage log prints."""
    assert wide_gate(q, rubin=rubin, seams=seams, dsa=dsa, tp=tp, dcp=dcp) == why


def test_wide_rows_of_the_recipes():
    """K3.1's decode role takes every captured verify graph above 64 rows: DL7's 24
    (72-256 rows), C48's 38 (q4 on the base tree, 68-256) and DL3 cap-128's 65
    (68-512 rows, 24 of them above 256)."""
    assert wide_rows(captured_rows(DL7, 8, 256), 8) == tuple(range(72, 257, 8))
    c48 = wide_rows(captured_rows(DL3_CAP64, 4, 256), 4)
    assert (len(c48), c48[0], c48[-1]) == (38, 68, 256)
    dl3 = wide_rows(captured_rows(DL3, 4, 512), 4)
    assert (len(dl3), dl3[0], dl3[-1], sum(m > 256 for m in dl3)) == (65, 68, 512, 24)


class Built(Exception):
    """Stops a constructor under test once the part the test checks has run."""


@pytest.mark.parametrize(
    "cc,sp,dsa,tp,dcp,wide",
    [
        ((10, 7), False, True, 16, 16, True),  # K3.1's decode role
        ((10, 7), False, True, 16, 1, False),
        ((10, 7), False, True, 8, 8, False),
        ((10, 3), False, True, 16, 16, False),  # GB300
        ((10, 7), True, True, 16, 16, False),
        ((10, 7), False, False, 8, 8, True),  # K3
    ],
)
def test_decode_reads_the_gate_from_the_deployment(
    monkeypatch, cc, sp, dsa, tp, dcp, wide
):
    """SpecializedDecode gates its wide rows on the device, the backbone's sequence
    parallelism and DSA layers, the TP group and the deployment's DCP; every decode
    buffer then holds the largest wide row (64 without)."""
    group = SimpleNamespace(world_size=tp, rank_in_group=0, device_group=None)
    platform = SimpleNamespace(current_device=lambda: "cuda:0")
    device = SimpleNamespace(major=cc[0], minor=cc[1], multi_processor_count=212)
    monkeypatch.setattr(decode_module, "get_tp_group", lambda: group)
    monkeypatch.setattr(decode_module, "current_platform", platform)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: device)
    monkeypatch.setattr(decode_module, "Context", MagicMock(side_effect=Built))
    text = SimpleNamespace(dsa_config=SimpleNamespace() if dsa else None)
    backbone = SimpleNamespace(config=text, use_sequence_parallel=sp)
    config = DecodeConfig(
        max_context=1 << 20, mla_core="auto", dsa_core="auto", dcp=dcp, q=8,
        capture_sizes=DL7, captured=captured_rows(DL7, 8, 256),
    )  # fmt: skip
    decode: Any = object.__new__(SpecializedDecode)
    with pytest.raises(Built):
        SpecializedDecode.__init__(decode, backbone, config)
    assert decode.wide == (tuple(range(72, 257, 8)) if wide else ())
    assert (decode.wide_off is None) == wide
    assert decode.capacity == (256 if wide else 64)


@pytest.mark.parametrize("dsa", [True, False], ids=["k31-tp8", "k3-table-miss"])
def test_coverage_names_why_wide_rows_run_stock(dsa):
    """Captured rows above 64 that run the stock forward name the cause: K3.1's TP16 =
    DCP16 gate, or a WIDE_ROWS miss."""
    decode: Any = object.__new__(SpecializedDecode)
    decode.q, decode.captured = 8, (64, 72, 256)
    decode.batches, decode.wide = (64,), ()
    decode.choices = lambda b, q: {}
    decode.wide_off = wide_gate(8, rubin=True, seams=True, dsa=dsa, tp=8, dcp=8)
    why = "DSA wide steps need TP16 = DCP16" if dsa else "not in WIDE_ROWS"
    assert decode.plan_coverage().reasons == {72: why, 256: why}


class Linear(torch.nn.Module):
    """A stock linear's call (``(x @ weight.T, None)``), BF16 and unquantized like
    the model's (``providers.bf16_weight`` reads its quant method), on small-integer
    weights: the input projections' outputs are exact, so the packed GEMM and the
    stock ones give the same columns."""

    def __init__(self, rows, cols=HIDDEN, trace=None, name=""):
        super().__init__()
        weight = torch.randint(-1, 2, (rows, cols), dtype=torch.bfloat16)
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.quant_method = UnquantizedLinearMethod()
        self.trace, self.name = trace, name

    def forward(self, x):
        if self.trace is not None:
            self.trace.append((self.name, x))
        return x @ self.weight.T, None


class SparseMla:
    """The sparse MLA attention layer (cache insert, DCP exchanges, core, W_UV)."""

    def __init__(self, trace):
        self.trace = trace
        ready = lambda: trace.append(("topk_ready",))  # noqa: E731
        self.impl = SimpleNamespace(record_logical_topk_ready=ready)

    def __call__(self, q, kv_c, k_pe, output_shape=None, q_dcp_replicated=None):
        assert q_dcp_replicated is None
        self.trace.append(("mla_attn", q, kv_c, k_pe, output_shape))
        return q.flatten(1)[:, : output_shape[1]] + kv_c[:, :1] + k_pe[:, 0, :1]


class Indexer(SimpleNamespace):
    __call__ = KimiK3DSAIndexer.forward


def dsa_layer(fused_insert):
    """A K3.1 DSA attention module whose ops after its input GEMMs record their calls
    in ``trace``; its stock forward is the real wrapper and indexer forward."""
    trace: list = []

    def op(name, result=lambda *args: None):
        def call(*args):
            trace.append((name, *args))
            return result(*args)

        return call

    indexer = Indexer(
        index_head_dim=INDEX_DIM,
        index_num_heads=INDEX_HEADS,
        index_rope_dim=0,
        use_fused_k_insert=fused_insert,
        wq_index=Linear(INDEX_HEADS * INDEX_DIM, Q_LORA, trace, "wq_index"),
        wkv_index=Linear(INDEX_DIM + INDEX_HEADS),
        _fused_k_insert=op("k_insert"),
        k_index_norm=op("k_norm", lambda k: k + 3),
        score=op("score"),
    )
    wrapper = SimpleNamespace(
        q_lora_rank=Q_LORA,
        kv_lora_rank=KV_LORA,
        qk_rope_head_dim=ROPE,
        qk_nope_head_dim=NOPE,
        qk_head_dim=NOPE + ROPE,
        v_head_dim=V_DIM,
        num_heads=HEADS,
        fuse_qkv_rmsnorm=False,
        dcp_q_replicate=False,
        skip_topk=False,
        is_sparse=True,
        rotary_emb=None,
        indexer=indexer,
        indexer_rope_emb=None,
        fused_qkv_a_proj=Linear(Q_LORA + KV_LORA + ROPE),
        q_a_layernorm=op("q_a_norm", lambda q_c: q_c + 1),
        kv_a_layernorm=op("kv_a_norm", lambda kv_c: kv_c + 2),
        q_b_proj=Linear(HEADS * (NOPE + ROPE), Q_LORA, trace, "q_b"),
        mla_attn=SparseMla(trace),
        g_proj=Linear(HEADS * V_DIM),
        o_proj=lambda rows: (rows, None),
    )
    attention = SimpleNamespace(
        mla_attn=wrapper,
        dsa_indexer=indexer,
        fused_qkv_a_proj=wrapper.fused_qkv_a_proj,
        g_proj=wrapper.g_proj,
    )
    return attention, trace


def same(a, b):
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return torch.equal(a, b)
    return type(a) is type(b) and a == b


@pytest.mark.parametrize("fused_insert", [True, False], ids=["fused-insert", "k-norm"])
def test_stock_dsa_is_the_stock_layer(fused_insert):
    """Seam S2-DSA on the packed in-projection rows runs the stock DSA layer's ops
    after its input GEMMs, in its order, on the same operands (each column block is
    one stock linear's output), and writes the stock gated output."""
    torch.manual_seed(0)
    attention, trace = dsa_layer(fused_insert)
    packed = pack_dsa_projection(attention)  # the stock linears become its views
    x = torch.randint(-2, 3, (72, HIDDEN), dtype=torch.bfloat16)  # 9 requests x q8
    want = MultiHeadLatentAttentionWrapper.forward(attention.mla_attn, None, x)
    stock = list(trace)
    insert = ["k_insert"] if fused_insert else ["k_norm"]
    ops = ["q_a_norm", "kv_a_norm", "q_b", "wq_index", *insert, "score"]
    assert [op[0] for op in stock] == [*ops, "topk_ready", "mla_attn"]
    trace.clear()
    columns = dsa_seam_columns(attention, packed.shape[0])
    assert packed.shape[0] == PITCH and columns == COLUMNS
    local = torch.full((72, HEADS * V_DIM), torch.nan, dtype=torch.bfloat16)
    Attention.stock_dsa(attention, x, x @ packed.T, local, columns)
    assert [op[0] for op in trace] == [op[0] for op in stock]
    for seam_op, stock_op in zip(trace, stock):
        assert len(seam_op) == len(stock_op)
        assert all(map(same, seam_op, stock_op)), seam_op[0]
    assert torch.equal(local, want)


@pytest.mark.parametrize(
    "target,name,value,pitch,reason",
    [
        ("mla_attn", "q_lora_rank", None, PITCH, "without q-LoRA"),
        ("mla_attn", "fuse_qkv_rmsnorm", True, PITCH, "fused q / kv RMSNorms"),
        ("mla_attn", "rotary_emb", torch.nn.Identity(), PITCH, "MLA RoPE"),
        ("dsa_indexer", "index_rope_dim", 64, PITCH, "indexer RoPE"),
        ("mla_attn", "g_proj", None, PITCH, "without the output gate"),
        ("mla_attn", "dcp_q_replicate", True, PITCH, "DCP-replicated"),
        ("mla_attn", "skip_topk", True, PITCH, "top-k shared"),
        (None, None, None, 2880, "2880-column rows"),
    ],
)
def test_dsa_seam_refuses_other_layers(target, name, value, pitch, reason):
    """A DSA layer whose stock forward the seam does not reproduce, or whose packed
    rows it would split wrongly, fails at startup instead of at a wide step."""
    attention, _ = dsa_layer(fused_insert=True)
    if target is not None:
        setattr(getattr(attention, target), name, value)
    with pytest.raises(ValueError, match=reason):
        dsa_seam_columns(attention, pitch)


@pytest.mark.parametrize("wide", [(72, 256), ()], ids=["wide", "tuned-only"])
@pytest.mark.parametrize("skip_topk", [False, True], ids=["k31", "shared-topk"])
def test_attention_checks_the_dsa_seam_at_startup(monkeypatch, wide, skip_topk):
    """A DSA layer's provider takes seam S2-DSA's column split when it is built with
    wide rows, so a layer the seam does not reproduce fails at startup; without wide
    rows (K3.1 off TP16 = DCP16) it needs no seam and builds as before."""
    attention, _ = dsa_layer(fused_insert=True)
    attention.mla_attn.skip_topk = skip_topk
    device = SimpleNamespace(multi_processor_count=212)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: device)
    for name in ("PreparedProjection", "SkinnyProjection", "dsa_scratch"):
        monkeypatch.setattr(providers_module, name, MagicMock())
    monkeypatch.setattr(providers_module, "PreparedDsa", MagicMock(side_effect=Built))
    runtime = SimpleNamespace(
        tp=16, dcp=1, q=8, tuned=BATCHES, batches=(*BATCHES, *wide), wide=wide,
        queries=(1, 8), dsa_core="auto", dsa_layers=0,
        context=Context(
            arch="rubin", tp=16, rank=0, dcp=1, sms=212, q=8, max_rows=256,
            max_context=1 << 20, bf16_state=True,
        ),
    )  # fmt: skip
    provider: Any = object.__new__(Attention)
    decoder = SimpleNamespace(self_attn=attention)
    if wide and skip_topk:
        with pytest.raises(ValueError, match="top-k shared"):
            Attention.__init__(provider, decoder, runtime, None, None, None)
        return
    with pytest.raises(Built):  # past the DSA branch's checks, at PreparedDsa
        Attention.__init__(provider, decoder, runtime, None, None, None)
    assert provider.dsa_columns == (COLUMNS if wide else ())


def test_dsa_rows_split_between_two_cores(monkeypatch):
    """K3.1 at TP16 = DCP16: a DSA layer builds #108's PreparedDsa on the tuned rows
    (up to 64) and WideDsa on every wide row (DSA_WIDE's automatic, open-ended policy:
    no row cap and no environment switch), on one shared scratch (sized for the
    widest), DcpBox and parity; ``run_dsa`` runs each step on the core of its rows."""
    from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadata

    attention, _ = dsa_layer(fused_insert=True)
    attention.num_local_heads = HEADS
    device = SimpleNamespace(multi_processor_count=212)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: device)
    for name in ("PreparedProjection", "SkinnyProjection"):
        monkeypatch.setattr(providers_module, name, MagicMock())
    scratch, box = MagicMock(name="scratch"), MagicMock(name="box")
    monkeypatch.setattr(
        providers_module, "dsa_scratch", MagicMock(return_value=scratch)
    )
    monkeypatch.setattr(providers_module, "dcp_box", MagicMock(return_value=box))
    monkeypatch.setattr(providers_module, "dsa_on_flashinfer", lambda attention: True)
    tuned = MagicMock(name="PreparedDsa")
    monkeypatch.setattr(providers_module, "PreparedDsa", tuned)
    monkeypatch.setattr(providers_module, "WideDsa", MagicMock(side_effect=Built))
    wide = (72, 80, 88, 96, 104, 128, 256)
    runtime = SimpleNamespace(
        tp=16, dcp=16, q=8, tuned=BATCHES, batches=(*BATCHES, *wide), wide=wide,
        queries=(1, 8), dsa_core="auto", dsa_layers=0,
        context=Context(
            arch="rubin", tp=16, rank=0, dcp=16, sms=212, q=8, max_rows=256,
            max_context=1 << 20, bf16_state=True,
        ),
    )  # fmt: skip
    provider: Any = object.__new__(Attention)
    with pytest.raises(Built):  # at WideDsa, after PreparedDsa
        Attention.__init__(
            provider, SimpleNamespace(self_attn=attention), runtime, None, None, None
        )
    rows = set(wide)  # 72-256: all admitted, above WIDE_MAX_ROWS (96) too
    assert provider.dsa_wide == rows
    providers_module.dsa_scratch.assert_called_once_with(runtime, max(rows))
    policy = dict(queries=(1, 8), core="auto", dcp=box, parity=0)
    tuned.assert_called_once_with(
        attention, dict.fromkeys(BATCHES, PITCH), scratch, **policy
    )
    providers_module.WideDsa.assert_called_once_with(
        attention, dict.fromkeys(sorted(rows), PITCH), scratch, **policy
    )

    core, core_wide = provider.core, provider.core_wide = MagicMock(), MagicMock()
    indexer = MagicMock(spec=DeepseekV32IndexerMetadata)
    indexer.slot_mapping = torch.zeros(96, dtype=torch.long)
    attention.mla_attn.mla_attn.layer_name = "attn"
    attention.dsa_indexer.k_cache = SimpleNamespace(prefix="index")
    context = SimpleNamespace(attn_metadata={"index": indexer})
    monkeypatch.setattr("vllm.forward_context.get_forward_context", lambda: context)
    provider.layer = SimpleNamespace(self_attn=attention, layer_idx=3)
    provider.prefetch = MagicMock()
    for b, ran, idle in ((72, core_wide, core), (64, core, core_wide)):
        ran.reset_mock()
        idle.reset_mock()
        projected = torch.zeros(b, PITCH)
        provider.run_dsa(projected, None, [None], None, {"attn": None}, 8)
        assert ran.run.call_count == 1 and ran.run.call_args.args[0] is projected
        idle.run.assert_not_called()


@pytest.mark.parametrize("prepared", [False, True], ids=["seam", "prepared"])
@pytest.mark.parametrize("fused_oproj", [True, False], ids=["two-shot", "stock-oproj"])
def test_run_wide_runs_dsa_layers_through_the_seam(monkeypatch, fused_oproj, prepared):
    """A DSA layer's wide step: the in-projection into the packed rows, seam S2-DSA
    into the o-projection's input rows, then the fused o-projection, or the stock one
    (seam S3: DSA layers have no GEMM-AR) and the unfused AttnRes."""
    b = 72
    seam = MagicMock()
    monkeypatch.setattr(Attention, "stock_dsa", seam)
    module = SimpleNamespace(o_proj=lambda rows: (rows + 1, None))
    provider: Any = object.__new__(Attention)
    projected, local = torch.zeros(b, PITCH), torch.zeros(b, HEADS * V_DIM)
    provider.kind, provider.layer = "dsa", SimpleNamespace(self_attn=module)
    provider.runtime = SimpleNamespace(stock_attn_res=frozenset())
    provider.storage = {b: (projected, local)}
    provider.projection = MagicMock()
    provider.projection.fallback.kernels = {b: None}
    provider.projection.run.side_effect = lambda x, out: out
    provider.kda_stock = provider.mla_wide = frozenset()
    provider.dsa_columns = COLUMNS
    provider.dsa_wide = frozenset({b} if prepared else ())
    provider.run_dsa = MagicMock()
    provider.oproj_wide = frozenset({b} if fused_oproj else ())
    provider.output, provider.post = MagicMock(), MagicMock()
    x, out = torch.zeros(b, HIDDEN), torch.zeros(b, HIDDEN)
    rows = provider.run_wide(x, out, None, None, [], None, {}, 8, None)
    if prepared:  # WideDsa on the widened box (dabb1639b6)
        seam.assert_not_called()
        provider.run_dsa.assert_called_once_with(projected, local, [], None, {}, 8)
    else:
        provider.run_dsa.assert_not_called()
        seam.assert_called_once_with(module, x, projected, local, COLUMNS)
    if fused_oproj:
        provider.output.run.assert_called_once_with(local, None, None, out)
        assert rows is provider.output.run.return_value[0]
    else:
        assert rows is provider.post.run.return_value
        assert torch.equal(provider.post.run.call_args.args[0], local + 1)


def test_wide_choices_name_the_dsa_seam():
    """The coverage log of a K3.1 wide step names the DSA seam and its in-projection:
    split-K up to 128 rows, cuBLAS above, as ``run_wide`` runs it (by the rows the
    projection prepares)."""
    rows = (72, 128, 136, 256, 512)
    decode: Any = object.__new__(SpecializedDecode)
    decode.attention = {
        0: SimpleNamespace(kind="kda", oproj_wide=frozenset(rows)),
        1: SimpleNamespace(
            kind="dsa", oproj_wide=frozenset(rows), dsa_wide=frozenset({72})
        ),
    }
    decode.stock_attn_res = decode.sharded = frozenset()
    decode.feed_forward = {}
    decode.context = Context(
        arch="rubin", tp=16, rank=0, dcp=16, sms=212, q=8, max_rows=512,
        max_context=1 << 20, bf16_state=True,
    )  # fmt: skip
    split_k = projection_policies(16, PITCH, rows, sms=212, q=8)
    assert set(split_k) == {72, 128}
    for b in rows:
        choices = decode.wide_choices(b, 8)
        gemm = "splitk" if b in split_k else "cublas"
        if b == 72:  # WideDsa (DSA_WIDE, dabb1639b6)
            assert choices["dsa"] == Choice(Path.GENERIC, "wide", "prepared")
        else:
            assert choices["dsa"] == Choice(Path.STOCK, note=f"{gemm} + S2")
        assert "mla" not in choices


# Rubin TP16 in-projection tactics by row bin (WIDE_B8, WIDE_B16_32, WIDE_B64): the
# DSA rows' and MLA's, and the split-K ones the DSA rows ran before.
DSA = {
    8: (128, 8, 4, 11, 15, None, 4),
    16: (128, 16, 4, 8, 15, None, 4),
    32: (64, 32, 2, 8),
    64: (128, 64, 6, 14, 14),
}
MLA = {
    8: (128, 8, 7, 12, 15, None, 4),
    16: (128, 16, 7, 10, 15, None, 4),
    32: (128, 32, 7, 6, 15, None, 4),
    64: (128, 64, 7, 16, 14),
}
SPLIT_K = {
    8: (64, 8, 4, 8),
    16: (64, 16, 2, 10),
    32: (64, 32, 2, 8),
    64: (64, 64, 2, 6),
}
DL7_TUNED = (*range(1, 9), 16, 24, 32, 40, 48, 56, 64)  # DL7's tuned decode rows


def row_bin(rows):
    return next(b for b in DSA if rows <= b)


def test_dsa_in_projection_takes_the_tensor_memory_tactics():
    """Rubin TP16: the DSA in-projection (3040 rows, 24 M tiles) stages its weights in
    tensor memory at up to 16 rows (split 4) and 33-64 rows (split 6), and keeps split-K
    at 17-32 rows, above 64 (F5) and on GB300; the KDA and MLA rows keep their
    tactics."""
    dsa = projection_policies(16, PITCH, DL7_TUNED, sms=212, q=8)
    assert dsa == {m: DSA[row_bin(m)] for m in DL7_TUNED}
    assert dsa[24] == dsa[32] == SPLIT_K[32]
    mla = projection_policies(16, 2880, DL7_TUNED, sms=212, q=8)
    polled = {8: (1,)}  # canonical NaNs where MlaQuery polls x (MLA only)
    assert mla == {m: MLA[row_bin(m)] + polled.get(m, ()) for m in DL7_TUNED}
    kda = projection_policies(16, 3216, DL7_TUNED, sms=212, q=8)
    wide_kda = (128, 64, 6, 14, 14)
    assert kda == {m: (64, 8, 2, 12) if m <= 8 else wide_kda for m in DL7_TUNED}
    wide = projection_policies(16, PITCH, (72, 96, 128), sms=212, q=8)
    assert wide == dict.fromkeys((72, 96, 128), (64, 64, 2, 6))
    gb300 = projection_policies(16, PITCH, DL7_TUNED, sms=152, q=8)
    five = {64: (64, 64, 2, 5)}  # GB300's B64 stages
    assert gb300 == {m: five.get(row_bin(m), SPLIT_K[row_bin(m)]) for m in DL7_TUNED}


@pytest.mark.parametrize("sms", [212, 152], ids=["rubin", "gb300"])
def test_dsa_layer_preloads_beside_its_attn_res(monkeypatch, sms):
    """A DSA layer whose in-projection stages its weights in tensor memory asks the
    AttnRes before it for carve-out 100 and lists those batches (``tmem_batches``:
    the previous MoE tail releases that AttnRes at entry, no next-attention prefetch),
    as an MLA layer does; on the decode ring's codes only (``before_ring_only``)."""
    attention, _ = dsa_layer(fused_insert=True)
    device = SimpleNamespace(multi_processor_count=sms)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: device)
    for name in ("PreparedProjection", "SkinnyProjection", "dsa_scratch"):
        monkeypatch.setattr(providers_module, name, MagicMock())
    monkeypatch.setattr(providers_module, "PreparedDsa", MagicMock(side_effect=Built))
    runtime = SimpleNamespace(
        tp=16, dcp=1, q=8, tuned=DL7_TUNED, batches=DL7_TUNED, wide=(),
        queries=(1, 8), dsa_core="auto", dsa_layers=0,
    )  # fmt: skip
    provider: Any = object.__new__(Attention)
    decoder = SimpleNamespace(self_attn=attention)
    with pytest.raises(Built):  # past the carve-out, at PreparedDsa
        Attention.__init__(provider, decoder, runtime, None, None, None)
    rubin = sms > 152
    tmem = tuple(m for m in DL7_TUNED if not 16 < m <= 32)
    assert provider.kind == "dsa"
    assert provider.before_carveout == (100 if rubin else None)
    assert provider.before_ring_only == rubin
    assert provider.tmem_batches == (tmem if rubin else ())


@pytest.mark.parametrize("ring_only", [False, True])
def test_attn_res_carveout_on_the_decode_ring_only(monkeypatch, ring_only):
    """``ring_only`` (the AttnRes before a DSA layer): the decode ring's codes (with
    and without aux) take the carve-out; the plain-input, wide-ring and token-batched
    codes keep the default, as before D1. Without it every code takes the carve-out
    (KDA / MLA)."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.attn_res import (
        prepared,
    )

    device = SimpleNamespace(multi_processor_count=212)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: device)
    monkeypatch.setattr(prepared, "architecture", lambda device: "sm_107a")
    monkeypatch.setattr(prepared, "to_cute", lambda tensor, *args: tensor)
    monkeypatch.setattr(prepared, "kernel_registers", lambda code, index: 0)
    monkeypatch.setattr(prepared, "compile_consumer", lambda *key: key)
    carved = lambda code, index, value: ("carved", value, code)  # noqa: E731
    monkeypatch.setattr(prepared, "prepare_shared_carveout", carved)
    weights = [torch.zeros(4)] * 3
    attn_res = prepared.PreparedAttnRes(weights, (2, True, -1, False), (1e-6,) * 2,
                                        carveout=100, ring_only=ring_only)  # fmt: skip
    attn_res.bind_ring(torch.zeros(1, 64, 4))
    attn_res.bind_ring_rows(torch.zeros(1, 256, 4))
    attn_res.bind_aux()
    ring = (attn_res.ring[0], attn_res.aux[1])
    rest = (attn_res.code, attn_res.ring_rows[0], attn_res.aux[0], attn_res.aux[2])
    rest += tuple(c for codes in attn_res.rows_codes.values() for c in codes)
    for code in ring:
        assert code[:2] == ("carved", 100)
    for code in rest:
        if ring_only:
            assert code[0] != "carved"
        else:
            assert code[:2] == ("carved", 100)


def bf16_ulp(values):
    exponent = torch.floor(torch.log2(values.abs().clamp_min(2.0**-126)))
    return torch.exp2(exponent - 7)


def rank_order(weight, x, split):
    """The split's partials summed in rank order and rounded once: rank r's K slice
    (64-wide tiles, the first ``tiles % split`` ranks one more) as one in-order MMA
    chain in FP32 (an unsplit SplitKTmemGemm on that slice)."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.projection import (
        PreparedProjection,
    )

    m, total, first = x.shape[0], None, 0
    tiles, extra = divmod(x.shape[1] // 64, split)
    for rank in range(split):
        count = tiles + (rank < extra)
        k = slice(64 * first, 64 * (first + count))
        first += count
        policy = {m: (128, 32, 1, 4, min(count, 15))}
        part = PreparedProjection(weight[:, k], policy, output_dtype=torch.float32)
        out = torch.empty(m, weight.shape[0], device=x.device)
        part.prepare().run(x[:, k], out)
        total = out if total is None else total + out
    return total.bfloat16()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="Rubin (SM107) tensor-memory GEMMs",
)
def test_dsa_in_projection_tensor_memory_numerics():
    """The DSA in-projection on its tensor-memory tactics writes every output element
    (the 96-row last M tile too, nothing past its 3040 columns) as the rank-order sum of
    its split's K-slice partials, bit for bit (and as the plain SplitKGemm of the same
    split where that divides K); against an FP64 reference its max and mean errors stay
    within one BF16 ulp of the split-K tactics it replaces (a stated FP32 accumulation
    reorder)."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.gemm.projection import (
        PreparedProjection,
    )

    torch.manual_seed(0)
    weight = (0.02 * torch.randn(PITCH, 7168, device="cuda")).bfloat16()
    x = torch.randn(64, 7168, device="cuda").bfloat16()
    tmem = projection_policies(16, PITCH, DL7_TUNED, sms=212, q=8)

    def run(tactic, m, pitch=PITCH):
        projection = PreparedProjection(weight, {m: tactic}).prepare()
        out = torch.full((m, pitch), -1, dtype=torch.int16, device="cuda")  # 0xFFFF
        projection.run(x[:m], out.view(torch.bfloat16)[:, :PITCH])
        torch.cuda.synchronize()
        return out

    for m in (m for m in DL7_TUNED if len(tmem[m]) > 4):
        padded = run(tmem[m], m, pitch=PITCH + 64)
        assert not (padded[:, :PITCH] == -1).any(), m
        assert (padded[:, PITCH:] == -1).all(), m
        new, split = padded[:, :PITCH], tmem[m][2]
        assert torch.equal(new, rank_order(weight, x[:m], split).view(torch.int16)), m
        if 7168 % (128 * split) == 0:
            plain = (128, min(tmem[m][1], 32), split, 4)
            assert torch.equal(new, run(plain, m)), m
        old = run(SPLIT_K[row_bin(m)], m)
        ref = x[:m].double() @ weight.double().T
        err = [(t.view(torch.bfloat16).double() - ref).abs() for t in (new, old)]
        ulp = bf16_ulp(ref)
        assert err[0].max() <= err[1].max() + ulp.max(), m
        assert err[0].mean() <= err[1].mean() + ulp.mean(), m
        print(
            f"M{m} split {split}: max err {err[0].max():.3e} (split-K "
            f"{err[1].max():.3e}), mean {err[0].mean():.3e} ({err[1].mean():.3e}), "
            f"differs from split-K {(new != old).float().mean().item():.2%}"
        )


def wide_step(b, *, wide_moe):
    """A three-layer wide step (a dense layer, then two MoE layers) whose AttnRes
    sites run the stock op (``stock_attn_res``), on mocked providers. ``wide_moe``:
    the MoE providers serve the rows (each tail's ring consumer runs the next
    AttnRes site), else the stock MoE layer does (seam S4)."""
    decode: Any = object.__new__(SpecializedDecode)
    layers = [
        SimpleNamespace(layer_idx=i, self_attn=SimpleNamespace(), mlp=MagicMock())
        for i in range(3)
    ]
    for layer in layers:
        layer.mlp.side_effect = lambda rows: torch.zeros_like(rows)
    decode.layers = tuple(layers)
    decode.verify_rows = lambda step, b: None
    decode.head_batches = decode.aux_layers = ()
    decode.dense_rings = {}
    decode.hidden = tuple(torch.zeros(b, HIDDEN) for _ in range(2))
    decode.prefix = torch.zeros(b, HIDDEN)
    decode.blocks = torch.zeros(b, 2, HIDDEN)
    decode.wide = decode.stock_attn_res = frozenset({b})
    decode.moe_batches = (b,) if wide_moe else ()
    decode.before = {i: MagicMock() for i in range(3)}
    decode.before_stock = {i: MagicMock() for i in range(3)}
    decode.final, decode.final_stock = MagicMock(), MagicMock()
    for stock in (*decode.before_stock.values(), decode.final_stock):
        stock.run.side_effect = lambda delta, *args: torch.zeros_like(delta)
    decode.attention = {i: MagicMock() for i in range(3)}
    for attention in decode.attention.values():
        attention.run.side_effect = lambda rows, *args, **kwargs: torch.zeros_like(rows)
    decode.feed_forward = {0: MagicMock(spec=PreparedDenseMLP)}
    decode.feed_forward |= {i: MagicMock(spec=PreparedMoE) for i in (1, 2)}
    for moe in decode.feed_forward.values():
        moe.run.side_effect = lambda rows, target, **kwargs: target
    decode.fused_before = {1: decode.before[2], 2: decode.final}
    decode.prefetch, decode.kda_side = MagicMock(), MagicMock()
    decode.backbone = SimpleNamespace(norm=lambda rows, residual: rows)
    step = Step(q=8, metadata=[None] * 3, slot_mapping={}, positions=None)
    return decode, step


@pytest.mark.parametrize("wide_moe", [True, False], ids=["moe-tail-ring", "stock-moe"])
def test_stock_attn_res_sites_run_once(wide_moe):
    """Above 256 rows (base STOCK_ROWS) a site that the previous MoE tail's ring
    consumer already ran is skipped: the stock op on its output would apply the
    AttnRes a second time. Sites after a stock MoE layer still run the stock op."""
    b = 264
    decode, step = wide_step(b, wide_moe=wide_moe)
    out = decode.run_path(step, None, torch.zeros(b, HIDDEN))
    ran = [decode.before_stock[i].run.call_count for i in range(3)]
    assert not any(before.run.called for before in decode.before.values())
    if wide_moe:
        # Layer 2's site and the final one are the MoE tails' ring consumers.
        assert ran == [1, 1, 0] and not decode.final_stock.run.called
        tail_rows = decode.feed_forward[1].run.call_args.args[1]
        assert decode.attention[2].run.call_args.args[0] is tail_rows
        assert out is decode.feed_forward[2].run.call_args.args[1]
    else:
        assert ran == [1, 1, 1] and decode.final_stock.run.call_count == 1


# FP32 KDA recurrent state (--mamba-ssm-cache-dtype float32). The tests below build
# the KDA kernel objects without a GPU (their constructors' shape / dtype /
# shared-memory asserts, which ``prepare_kv`` hits at startup; no compile; they need
# cutlass and flashinfer only). The compiles and the numerics (head and wide verify
# vs KdaSplit, bitwise) are test_specialized_decode_kda_fp32_gpu.py's (Rubin).
RUBIN = (10, 7)
KDA_HEADS = 6  # TP16 (96 KDA heads)


def kda_kernel(requests, q, bf16_state, gate_mode=0, heads=KDA_HEADS):
    """The plain verify kernel ``kda.prepared.executable`` builds (Rubin, whole state
    rows), constructed but not compiled."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.kda import prepared

    schedule = prepared.schedule_of(heads, requests, gate_mode, q, RUBIN, bf16_state)
    options = (heads, requests, gate_mode, False, schedule, q, bf16_state)
    return prepared.make_kernel(options, RUBIN)


def kda_context(bf16_state, q, tp=16):
    return Context(
        arch="rubin", tp=tp, rank=0, dcp=tp, sms=212, q=q, max_rows=512 * q,
        max_context=1 << 20, bf16_state=bf16_state,
    )  # fmt: skip


@pytest.mark.parametrize("bf16_state", [False, True], ids=["fp32", "bf16"])
@pytest.mark.parametrize("q,requests", [(4, 512), (8, 128)], ids=["q4", "q8"])
def test_kda_verify_kernels_take_the_state_dtype(q, requests, bf16_state):
    """Every request count up to 512 (q4) / 128 (q8) builds its verify kernel with
    either state dtype: the BF16-only chunked verify (Stream3, which asserts a BF16
    state) never serves an FP32 state, and the coverage note names the kernel that
    runs."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.kda import prepared

    names = {
        prepared.KdaVerifyWide: "wide",
        prepared.KdaVerifyHalves: "halves",
        prepared.KdaVerifyStream: "stream",
        prepared.KdaVerifyStream3: "stream3",
    }
    for b in range(1, requests + 1):
        kernel = kda_kernel(b, q, bf16_state)  # raises on an unsupported dtype
        assert kernel.bf16_state == bf16_state
        name = names.get(type(kernel), "kdasplit")
        note = prepared.choose(Shape(b * q, q), kda_context(bf16_state, q)).note
        assert note.split()[0] == name, (b, note)
        assert prepared.verify_kernel(b, q, RUBIN, bf16_state) == name
        if not bf16_state:
            assert name not in ("stream", "stream3"), b


def test_kda_q4_wide_policy():
    """q4 rows (cap-128 overlay): Halves at 35-48 and Wide at 49-64 with either state;
    Stream3 from 65 requests with a BF16 state, open-ended (no request cap); with an
    FP32 state those rows run KdaVerifyWide, as do q8 rows above VERIFY_WIDE's 64."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.kda import prepared

    kind = prepared.verify_kernel
    for state in (False, True):
        assert kind(12, 4, RUBIN, state) == "wide"
        assert [kind(b, 4, RUBIN, state) for b in (33, 34)] == ["wide"] * 2
        assert {kind(b, 4, RUBIN, state) for b in range(35, 49)} == {"halves"}
        assert {kind(b, 4, RUBIN, state) for b in range(49, 65)} == {"wide"}
        assert {kind(b, 8, RUBIN, state) for b in range(65, 1025)} == {"wide"}
    bf16 = {kind(b, 4, RUBIN, True) for b in range(65, 1025)}
    fp32 = {kind(b, 4, RUBIN, False) for b in range(65, 1025)}
    assert bf16 == {"stream3"} and fp32 == {"wide"}


@pytest.mark.parametrize("bf16_state", [False, True], ids=["fp32", "bf16"])
def test_kda_tp8_verify_keeps_kdasplit(bf16_state):
    """TP8 verify layers (12 heads, gate mode 1, fast math) are outside the wide
    kernels' asserts: they keep KdaSplit at every request count, and say so."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.kda import prepared

    for b in (16, 36, 64, 65, 256):
        kernel = kda_kernel(b, 4, bf16_state, gate_mode=1, heads=12)
        assert type(kernel) is prepared.KdaSplit
        note = prepared.choose(Shape(4 * b, 4), kda_context(bf16_state, 4, 8)).note
        assert note.startswith("kdasplit")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_kda_heads_take_an_fp32_state(dtype):
    """KdaHead (q1 B8), KdaHeadVerify (B1 x q8) and KdaHeadMulti (2 x q8, 2 x q4)
    keep the state in registers through KdaSplit's load_state / store_state: an FP32
    state fits the same shared-memory budget, so the head serves it too; other
    dtypes and shapes without a kernel keep the in-projection -> KdaSplit pair."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.kda.head import (
        head_supported,
    )

    for batch, q in ((8, 1), (8, 8), (16, 8), (8, 4)):
        assert head_supported(KDA_HEADS, dtype, False, 7168, batch, q), (batch, q)
    assert not head_supported(KDA_HEADS, dtype, False, 7168, 32, 8)  # 4 requests
    assert not head_supported(KDA_HEADS, dtype, False, 7168, 4, 4)  # B1 x q4
    assert not head_supported(KDA_HEADS, torch.float16, False, 7168, 8, 8)


@pytest.fixture
def head_probe(monkeypatch):
    """``kda.head`` with CUDA reported present, its compile functions recorded (and
    failing when ``fail`` holds their name), and its admission caches cleared."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.kda import head

    calls: list = []
    fail: set = set()

    def compiler(name):
        def compile_(*args, **kwargs):
            calls.append((name, args, kwargs))
            if name in fail:
                raise RuntimeError("ptxas: out of registers")

        return compile_

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(head, "architecture", lambda device: "sm_107a")
    monkeypatch.setattr(head, "polled_out_batches", lambda tp, device: (8,))
    monkeypatch.setattr(head, "compile_head", compiler("head"))
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.kda import (
        head_multi,
        head_verify,
    )

    monkeypatch.setattr(head_verify, "compile_head_verify", compiler("verify"))
    monkeypatch.setattr(head_multi, "compile_head_multi", compiler("multi"))
    monkeypatch.setattr(torch.accelerator, "device_index", lambda index: nullcontext())
    for cached in (head.head_constructs, head.head_compiles):
        cached.cache_clear()
    yield head, calls, fail
    for cached in (head.head_constructs, head.head_compiles):
        cached.cache_clear()


def kda_core_stub(conv_dim_first=True):
    """The ``PreparedKda`` fields a KdaHead's tensors read (TP16: 6 heads)."""
    meta = dict(device="meta")
    weights = (
        torch.empty((3, 4, 768), dtype=torch.float32, **meta),  # conv (decode layout)
        torch.empty(768, dtype=torch.float32, **meta),  # bias (dt: no conv bias)
        torch.empty((768, 128), dtype=torch.bfloat16, **meta),  # F_B
        torch.empty(6, dtype=torch.float32, **meta),  # A_log
        torch.empty(768, dtype=torch.float32, **meta),  # dt bias
        torch.empty(128, dtype=torch.float32, **meta),  # o_norm
    )
    return SimpleNamespace(
        heads=KDA_HEADS, dim=768, has_bias=False, weights=weights,
        conv_dim_first=conv_dim_first, device=torch.device("cuda", 0),
    )  # fmt: skip


@pytest.mark.parametrize(
    "batch,q,name",
    [(8, 1, "head"), (8, 8, "verify"), (16, 8, "multi"), (8, 4, "multi")],
)
def test_kda_fp32_head_compile_probe(head_probe, batch, q, name):
    """An FP32 state admits a KdaHead shape only once its kernels compile on the
    device (the polled variant too where POLLED_OUT puts one), on meta caches of the
    deployment's layout; a failed compile keeps the pair (no startup failure), and
    the BF16 state (the measured serving dtype) is not probed."""
    head, calls, fail = head_probe
    core = kda_core_stub()
    probe = dict(core=core, fa_width=144, cache_tokens=q if q > 1 else 8, tp=16)
    args = (KDA_HEADS, torch.float32, False, 7168, batch, q)
    assert head.head_supported(*args, **probe)
    polled = batch == 8  # POLLED_OUT's batch (stubbed: 8)
    assert [c[0] for c in calls] == [name] * (1 + polled)
    specs = calls[0][1][4 if name == "head" else 3]
    conv, state = specs[2], specs[7]
    assert conv[0] == (16, 3 * 768, probe["cache_tokens"] + 2)
    assert conv[2] == torch.bfloat16
    assert state[0] == (16, KDA_HEADS, 128, 128) and state[2] == torch.float32
    assert specs[3][0] == (batch, 144) and specs[8][0] == (batch, 768)
    assert specs[6][0] == ((batch,) if q == 1 else (batch // q, q))
    assert len(specs) == (11 if q == 1 else 12)

    calls.clear()  # cached: one probe per shape and layout, not per layer
    assert head.head_supported(*args, **probe)
    assert not calls

    head.head_compiles.cache_clear()
    fail.add(name)
    assert not head.head_supported(*args, **probe)

    calls.clear()
    bf16 = (KDA_HEADS, torch.bfloat16, False, 7168, batch, q)
    assert head.head_supported(*bf16, **probe) and not calls


@pytest.mark.parametrize("compiles", [True, False], ids=["compiles", "fails"])
def test_kda_fp32_head_probe_decides_in_the_provider(monkeypatch, head_probe, compiles):
    """A Rubin TP16 KDA layer (q8) with an FP32 state builds its KdaHead (q1 B8, B1 x
    q8, 2 x q8) only when the kernels compile; when they fail it builds none, before
    anything is set up for the head (no KdaHead index or F_A buffer, which the MoE
    tail's early release and the AttnRes carve-out follow), and keeps the pair."""
    head, calls, fail = head_probe
    if not compiles:
        fail.update({"head", "verify", "multi"})
    batches = (*range(1, 9), 16)
    weight = torch.empty((3216, 7168), dtype=torch.bfloat16, device="meta")
    monkeypatch.setattr(providers_module, "bf16_weight", lambda linear: weight)
    monkeypatch.setattr(providers_module, "head_supported", head.head_supported)
    stub = kda_core_stub()
    monkeypatch.setattr(providers_module, "PreparedKda", lambda *a, **k: stub)
    monkeypatch.setattr(providers_module, "is_conv_state_dim_first", lambda: True)
    monkeypatch.setattr(providers_module, "polled_out_batches", lambda tp, d: (8,))
    for name in ("PreparedProjection", "SkinnyProjection"):
        monkeypatch.setattr(providers_module, name, MagicMock())
    built = MagicMock(name="PreparedKdaHead")
    monkeypatch.setattr(providers_module, "PreparedKdaHead", built)
    monkeypatch.setattr(providers_module, "PreparedOproj", MagicMock(side_effect=Built))
    device = SimpleNamespace(multi_processor_count=212)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: device)
    runtime = SimpleNamespace(
        tp=16, q=8, tuned=batches, batches=batches, wide=(), bf16_state=False,
        buffers={}, device="cpu", kda_heads=0, kda_side=None, capacity=16,
        sharded=frozenset(), shard_box=None, context=kda_context(False, 8),
    )  # fmt: skip
    decoder = SimpleNamespace(
        self_attn=SimpleNamespace(in_proj_qkvgfab=None), layer_idx=0,
        is_moe_layer=False,
    )  # fmt: skip
    provider: Any = object.__new__(Attention)
    with pytest.raises(Built):  # past the KDA branch, at the o-projection
        Attention.__init__(provider, decoder, runtime, None, None, None)
    assert provider.kind == "kda"
    if compiles:
        assert sorted({c[0] for c in calls}) == ["head", "multi", "verify"]
        built.assert_called_once()
        args, kwargs = built.call_args
        assert args[2] == [8] and kwargs["verify"] == [(8, 8), (16, 8)]
        assert kwargs["polled"] == [8] and runtime.kda_heads == 1
        assert provider.head is built.return_value
        assert provider.head_state == torch.float32
        assert runtime.buffers["kda.head.fa"].shape == (128, 64, 144)
    else:  # the verify head is only probed for an admitted q1 batch
        assert sorted({c[0] for c in calls}) == ["head", "multi"]
        built.assert_not_called()
        assert provider.head is None and provider.head_state is None
        assert runtime.kda_heads == 0 and "kda.head.fa" not in runtime.buffers
