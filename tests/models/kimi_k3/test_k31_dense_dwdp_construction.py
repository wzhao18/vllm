# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Non-expert DWDP (dense_dwdp): every weight storage of a decoder layer, whatever
its dtype or quantization, moves into a blob right after the layer is built, so a
member holds only its 1/N share before loading. Post-load processing runs per layer
in a load window on the whole layer (fetched like a forward fetches it), and its
results are sharded again; after the forward fetch every member must hold exactly
what processing the unsharded layer gives.

The fabric primitives are faked on CPU: a VA range is one flat buffer whose home
shard is the member's fabric memory, an imported shard views the other member's
buffer, the copies are tensor copies; members run as threads."""

import threading
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.multiprocessing.reductions import StorageWeakRef

from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.models.kimi_k3.nvidia import dense_dwdp

N = 4  # members
_SYNC = torch.cuda.synchronize
CHUNK = 4096
_TLS = threading.local()


def _group():
    return SimpleNamespace(world_size=N, rank_in_group=_TLS.me)


class _Mem:
    """One fabric allocation: bound to the range region that maps it first."""

    def __init__(self, nbytes):
        self.nbytes = nbytes
        self.view = None


class _Fabric:
    """edwdp's create / export / VaRange / Copies and the member store on CPU."""

    def __init__(self):
        self.device = "cpu"
        self.created = self.released = 0
        self.handles: dict[bytes, _Mem] = {}
        self.lock = threading.Lock()
        self.barrier = threading.Barrier(N)
        self.posted: dict = {}

    def create(self, nbytes, device, die=None):
        assert nbytes % CHUNK == 0
        with self.lock:
            self.created += nbytes
        return _Mem(nbytes)

    def export(self, mem):
        key = b"h%d" % id(mem)
        with self.lock:
            self.handles[key] = mem
        return key

    def release(self, mem):
        self.released += 1

    def range(self, nbytes, device):
        fabric = self
        buf = torch.full((nbytes,), 0xEE, dtype=torch.uint8, device=fabric.device)
        state = {"imported": None}
        calls = []

        def map_(offset, size, mem):
            if mem.view is None:
                mem.view = buf[offset : offset + size]

        def map_imported(offset, size, key):
            calls.append(("import", offset, key))
            if offset == 0 and key in fabric.handles:  # a whole peer shard
                state["imported"] = fabric.handles[key].view

        return SimpleNamespace(
            map=map_,
            unmap=lambda offset, size: calls.append(("unmap", offset)),
            map_imported=map_imported,
            tensor=lambda: buf if state["imported"] is None else state["imported"],
            calls=calls,
        )

    def store(self, group, members=N):
        fabric, me = self, group.rank_in_group

        def exchange(mine, tag="handles"):
            with fabric.lock:
                fabric.posted[(tag, me)] = dict(mine)  # a gather copies
            fabric.barrier.wait()
            return [fabric.posted[(tag, m)] for m in range(members)]

        return SimpleNamespace(
            exchange=exchange, barrier=lambda tag: fabric.barrier.wait()
        )


class _Copies:
    def __init__(self, pairs, n_streams):
        self.pairs = pairs

    def fetch(self, streams, handles):
        for dst, src in self.pairs:
            dst.copy_(src)

    def wait(self):
        pass


@pytest.fixture
def fabric(monkeypatch):
    f = _Fabric()
    edwdp = dense_dwdp.edwdp
    monkeypatch.setattr(edwdp, "create", f.create)
    monkeypatch.setattr(edwdp, "export", f.export)
    monkeypatch.setattr(edwdp, "release", f.release)
    monkeypatch.setattr(edwdp, "VaRange", f.range)
    monkeypatch.setattr(edwdp, "Copies", _Copies)
    monkeypatch.setattr(edwdp, "GroupStore", f.store)
    monkeypatch.setattr(edwdp, "copy_streams", lambda: ([None] * 4, [None] * 4))
    monkeypatch.setattr(edwdp, "_dies", lambda: SimpleNamespace(CHUNK=CHUNK))
    monkeypatch.setattr(edwdp, "get", lambda: None)
    monkeypatch.setattr(edwdp, "dense_enabled", lambda: True)
    monkeypatch.setattr(dense_dwdp, "MIN_BYTES", 1024)
    monkeypatch.setattr(dense_dwdp, "DEVICE_TYPE", "cpu")
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (0, 0))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device=None: 0)
    monkeypatch.setattr("vllm.distributed.get_ep_group", _group)
    return f


class _Method(QuantizeMethodBase):
    """A linear method whose post-load processing is ``fn(layer)``."""

    def __init__(self, fn=None):
        self.fn = fn

    def create_weights(self, *args, **kwargs):
        raise NotImplementedError

    def apply(self, layer, *args, **kwargs):
        raise NotImplementedError

    def process_weights_after_loading(self, layer):
        if self.fn is not None:
            self.fn(layer)


class _Linear(LinearBase):
    """A LinearBase without the TP setup: a weight (and an MXFP8-style per-32
    scale) of the given dtype, post-load processing ``fn``."""

    def __init__(self, rows, cols, dtype=torch.bfloat16, fn=None, scale=False):
        nn.Module.__init__(self)
        self.quant_method = _Method(fn)
        self.weight = nn.Parameter(torch.zeros(rows, cols, dtype=dtype), False)
        if scale:
            self.weight_scale = nn.Parameter(
                torch.zeros(rows, cols // 32, dtype=torch.uint8), False
            )


def _transpose(layer):  # a kernel layout that re-views the weight (CuTe MXFP8)
    layer.weight.data = layer.weight.data.t()


def _copy(layer):  # a same-size replacement (e.g. a die-localized copy)
    layer.weight.data = layer.weight.data.clone() + 1


def _in_place(layer):  # rewrites the loaded bytes in place
    layer.weight.data.view(torch.uint8).bitwise_xor_(0x01)


def _mxfp8(layer):
    """The FlashInfer CuTe DSL MXFP8 kernel's own processing: the weight becomes
    its [K, N] transpose view, the scales are swizzled to F8_128x4 (in place when
    the rows are a multiple of 128, else a new padded storage)."""
    from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
        FlashInferCutedslMxfp8LinearKernel,
    )

    kernel = object.__new__(FlashInferCutedslMxfp8LinearKernel)
    kernel.process_weights_after_loading(layer)


class _Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.kv_b_proj = _Linear(64, 32)  # read whole by an absorption
        self.o_proj = _Linear(32, 96, fn=_transpose)

    def dwdp_resident_linears(self):
        return [self.kv_b_proj]


class _Gemm:  # a fused GEMM built at construction that keeps a weight view
    def __init__(self, w):
        self.w = w


_Gemm.__module__ = "vllm.models.kimi_k3.nvidia.test_stub"  # the walk follows it


class _Layer(nn.Module):
    """Weights of several dtypes / quantizations and post-load transforms of
    every kind: none, re-view, same-size replacement, in-place rewrite, scale
    swizzle in place and to a bigger storage, packing (``_prepare``) and an
    absorption of a resident linear (``_prepare``)."""

    def __init__(self):
        super().__init__()
        self.attn = _Attn()
        self.gate_up = _Linear(128, 32)
        self.down = _Linear(32, 64, fn=_copy)
        self.fp8 = _Linear(64, 64, dtype=torch.float8_e4m3fn, fn=_in_place)
        self.mx = _Linear(256, 256, dtype=torch.float8_e4m3fn, fn=_mxfp8, scale=True)
        self.mx_pad = _Linear(
            200, 256, dtype=torch.float8_e4m3fn, fn=_mxfp8, scale=True
        )
        self.a = _Linear(32, 32)
        self.b = _Linear(48, 32)
        self.norm = nn.Parameter(torch.ones(32), requires_grad=False)
        self._gemm = _Gemm(self.gate_up.weight.data[64:])


def _prepare(layer):
    """The model's per-layer transforms: pack a | b into one new weight, absorb
    the resident kv_b_proj into a new parameter."""
    packed = torch.cat([layer.a.weight.data, layer.b.weight.data])
    layer.packed = nn.Parameter(packed, requires_grad=False)
    layer.a.weight.data = layer.packed.data[:32]
    layer.b.weight.data = layer.packed.data[32:]
    w = layer.attn.kv_b_proj.weight.data
    layer.attn.w_uk = nn.Parameter(
        (w.float() * 0.5).to(w.dtype).t().contiguous(), False
    )


def _weights(layer_idx, cls=None):
    """The checkpoint tensors of a layer (the same for every member)."""
    g = torch.Generator().manual_seed(layer_idx)
    out = {}
    for name, p in (cls or _Layer)().named_parameters():
        if p.dtype == torch.uint8:
            out[name] = torch.randint(100, 140, p.shape, generator=g, dtype=p.dtype)
        else:
            out[name] = torch.randn(p.shape, generator=g).to(p.dtype)
    return out


def _load(layers, first=10):
    """The weight loader: write each checkpoint tensor through its parameter."""
    for i, layer in enumerate(layers):
        weights = _weights(first + i, type(layer))
        for name, p in layer.named_parameters():
            p.data.copy_(weights[name])


def _build(me, num_layers=3, first=10, cls=None):
    """Member ``me``'s layers, each sharded as soon as it is built."""
    dense_dwdp._B = None
    _TLS.me = me
    layers = []
    for i in range(num_layers):
        layer = (cls or _Layer)()
        dense_dwdp.shard_layer(layer, first + i)
        layers.append(layer)
    built = dense_dwdp._B
    dense_dwdp._B = None
    return layers, built


class _Model(nn.Module):
    """vLLM's process_weights_after_loading runs the generic per-module
    processing, then this hook: the model's per-layer transforms, under DWDP in
    each layer's load window."""

    def __init__(self, layers, built=None, first=10, prepare=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.built, self.first, self.dwdp = built, first, None
        self.prepare = prepare or (lambda layer: _prepare(layer))

    def process_weights_after_loading(self):
        if self.built is None:
            for layer in self.layers:
                self.prepare(layer)
        else:
            self.dwdp = dense_dwdp.DenseDwdp(
                list(self.layers), self.first, None, self.built, self.prepare,
                torch.bfloat16,
            )  # fmt: skip


def _post_load(model):
    from vllm.model_executor.model_loader.utils import process_weights_after_loading

    cfg = SimpleNamespace(
        word_embeddings_untied_by_checkpoint=False,
        dtype=torch.bfloat16,
        quantization=None,
    )
    process_weights_after_loading(model, cfg, torch.device("cpu"))


def _tensors(layer):
    """Every parameter and fused-GEMM view the forward reads, by name."""
    out = dict(layer.named_parameters())
    out["_gemm.w"] = layer._gemm.w
    return out


def _bits(t):
    return t.reshape(-1).contiguous().view(torch.uint8)


def test_construction_shards_every_weight_storage(fabric, monkeypatch):
    """After construction a member holds, per sharded layer, one 1/N home shard
    of every weight storage of at least MIN_BYTES (any dtype or quantization,
    scales included), plus staging slots created once per slot; the full
    storages are freed, the views built with the layer follow, and only the
    dwdp_resident_linears and small parameters stay whole."""
    old = []
    real_shard = dense_dwdp._Built.shard

    def shard(self, layer, layer_idx, stop):
        old.extend(
            StorageWeakRef(p.untyped_storage())
            for p in (layer.gate_up.weight, layer.fp8.weight, layer.mx.weight_scale)
        )
        real_shard(self, layer, layer_idx, stop)

    monkeypatch.setattr(dense_dwdp._Built, "shard", shard)
    layers, built = _build(me=1)
    assert built.start == 10 and sorted(built.layers) == [11, 12]
    assert all(ref.expired() for ref in old)
    layer = layers[1]
    sharded = [
        p
        for name, p in layer.named_parameters()
        if p.nbytes >= 1024 and not name.startswith("attn.kv_b_proj")
    ]
    eligible = sum(dense_dwdp.aligned(p.nbytes) for p in sharded)
    blob, shard_bytes, regions = built.layers[11]
    assert shard_bytes == dense_dwdp.aligned(-(-eligible // N), CHUNK)
    assert len(regions) == len(sharded)
    target = blob.untyped_storage().data_ptr()
    for p in sharded:
        assert p.untyped_storage().data_ptr() == target
    assert layer._gemm.w.untyped_storage().data_ptr() == target
    for t in (layer.attn.kv_b_proj.weight, layer.norm):
        assert t.untyped_storage().data_ptr() != target
    assert built.resident == 2 * layer.attn.kv_b_proj.weight.nbytes
    # Home shards of 2 layers, staging slots for N - 1 members in 2 slots.
    assert fabric.created == shard_bytes * (2 + 2 * (N - 1))
    # Their processing reads a moved storage: vLLM's generic pass skips them.
    assert layer.mx.defer_weight_processing and layer.gate_up.defer_weight_processing
    assert not getattr(layer.attn.kv_b_proj, "defer_weight_processing", False)


def test_disabled_without_dwdp(fabric, monkeypatch):
    monkeypatch.setattr(dense_dwdp.edwdp, "dense_enabled", lambda: False)
    layers, built = _build(me=0)
    assert built is None and fabric.created == 0


def test_vocabulary_table_keeps_share_and_reads_peers_in_place(fabric):
    """A vocabulary table keeps its 1/N home shard; until install the other
    members' rows map one shared stand-in (the loader's writes land there), then
    they map those members' memory and the stand-in is released. A tied table is
    sharded once."""
    dense_dwdp._B = None
    _TLS.me = me = 1
    weight = torch.randn(1000, 64).to(torch.bfloat16)
    table = nn.Parameter(weight.clone(), requires_grad=False)
    old = StorageWeakRef(table.untyped_storage())
    dense_dwdp.shard_table(table)
    dense_dwdp.shard_table(table)  # tied (e.g. LM head = embedding)
    built = dense_dwdp._B
    dense_dwdp._B = None
    assert old.expired() and len(built.tables) == 1
    key, rng, shard, param, _ = built.tables[0]
    assert param is table
    assert shard == dense_dwdp.aligned(-(-table.nbytes // N), CHUNK)
    assert fabric.created == 2 * shard  # home + the shared stand-in
    # This member's rows kept what construction wrote.
    rows = shard // (64 * 2)
    assert torch.equal(
        table[me * rows : (me + 1) * rows], weight[me * rows : (me + 1) * rows]
    )
    built.map_tables([{key: b"m%d" % m} for m in range(N)])
    peers = [m for m in range(N) if m != me]
    assert rng.calls == [
        call
        for m in peers
        for call in (("unmap", m * shard), ("import", m * shard, b"m%d" % m))
    ]
    assert fabric.released == 1 and not built.stand_ins
    table.data = torch.zeros_like(table)
    with pytest.raises(RuntimeError, match="replaced after loading"):
        built.map_tables([{key: b""} for _ in range(N)])


def test_layer_with_nothing_left_gets_no_load_time_blob():
    """A layer whose every weight was sharded at construction has no load-time
    blob (aligned() rounds 0 up to a chunk: an empty plan used to get one)."""
    assert dense_dwdp._shard_bytes([], 0, N, CHUNK) == 0
    assert dense_dwdp._shard_bytes([object()], 5000, N, CHUNK) == 4096


def test_generic_post_load_skips_deferred_modules():
    """The generic process_weights_after_loading leaves a module flagged
    defer_weight_processing to the model and processes every other one."""
    calls = []
    model = nn.Module()
    model.a = _Linear(4, 32, fn=lambda layer: calls.append("a"))
    model.b = _Linear(4, 32, fn=lambda layer: calls.append("b"))
    model.b.defer_weight_processing = True
    _post_load(model)
    assert calls == ["a"]


def test_loaded_poisons_the_staging_parts(fabric):
    """After loading, the other members' rows of a construction blob read as
    NaN bytes: a transform that reads a sharded weight outside its load window
    fails loudly."""
    layers, built = _build(me=2)
    _load(layers)
    built.poison()
    blob, shard, _ = built.layers[11]
    assert bool((blob[:shard] == 0xFF).all()) and bool(
        (blob[3 * shard :] == 0xFF).all()
    )
    assert not bool((blob[2 * shard : 3 * shard] == 0xFF).all())


def _run_members(fabric, num_layers, check, cls=None, prepare=None):
    """Build, load and post-load ``N`` members (threads) and an unsharded
    reference; ``check(member_model, reference_model)`` per member."""
    ref = _Model([(cls or _Layer)() for _ in range(num_layers)], prepare=prepare)
    _load(ref.layers)
    _post_load(ref)
    members = [_build(me, num_layers, cls=cls) for me in range(N)]
    for layers, built in members:
        _load(layers)
        built.poison()
    models = [_Model(layers, built, prepare=prepare) for layers, built in members]
    errors = []

    def run(me):
        _TLS.me = me
        try:
            _post_load(models[me])
        except threading.BrokenBarrierError:
            pass  # another member failed
        except BaseException as e:  # surface in the main thread
            errors.append(e)
            fabric.barrier.abort()

    threads = [threading.Thread(target=run, args=(me,), daemon=True) for me in range(N)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    if errors:
        raise errors[0]
    for model in models:
        check(model, ref)
    return models


def test_load_windows_match_unsharded(fabric):
    """Every member processes each layer in its load window; after a layer's
    forward fetch every parameter and fused-GEMM view holds exactly what the
    unsharded processing gives (re-views, replacements moved back, in-place
    rewrites, scales swizzled in place or to a padded storage, packing and an
    absorption sharded at load), and nothing stays deferred."""

    def check(model, ref):
        dwdp = model.dwdp
        for i in range(1, len(model.layers)):
            dwdp.copies[i].fetch(None, None)
            got, want = _tensors(model.layers[i]), _tensors(ref.layers[i])
            assert sorted(got) == sorted(want)
            for name in want:
                g, w = got[name], want[name]
                assert g.shape == w.shape and g.stride() == w.stride(), name
                assert torch.equal(_bits(g), _bits(w)), (i, name)
        for layer in model.layers:
            for m in layer.modules():
                assert not getattr(m, "defer_weight_processing", False)

    models = _run_members(fabric, num_layers=5, check=check)
    # The die-localized-style copy went back into its construction shard; the
    # padded scales and the packed / absorbed weights went to load-time blobs.
    layer = models[0].layers[2]
    blob = models[0].built.layers[12][0].untyped_storage().data_ptr()
    assert layer.down.weight.untyped_storage().data_ptr() == blob
    assert layer.mx.weight_scale.untyped_storage().data_ptr() == blob
    for t in (layer.mx_pad.weight_scale, layer.packed, layer.attn.w_uk):
        assert t.untyped_storage().data_ptr() != blob


class _PlainLayer(nn.Module):
    """Weights without post-load transforms: all sharded at construction."""

    def __init__(self):
        super().__init__()
        self.up = _Linear(64, 32)
        self.down = _Linear(32, 64, dtype=torch.float8_e4m3fn)


def test_install_without_transforms_makes_no_load_time_blob(fabric):
    """When the load windows change nothing, install creates no load-time blob:
    each layer copies only its construction shards from every other member, and
    the forward fetch still gives the checkpoint."""
    created = {}

    def check(model, ref):
        assert len(model.dwdp.copies) == 3 and not model.dwdp.copies[0].pairs
        for i in (1, 2):
            blob, shard, _ = model.built.layers[10 + i]
            me = model.built.me
            dsts = [dst.data_ptr() for dst, _ in model.dwdp.copies[i].pairs]
            assert dsts == [blob[m * shard :].data_ptr() for m in range(N) if m != me]
            model.dwdp.copies[i].fetch(None, None)
            for (name, g), (_, w) in zip(
                model.layers[i].named_parameters(), ref.layers[i].named_parameters()
            ):
                assert torch.equal(_bits(g), _bits(w)), (i, name)

    real = dense_dwdp._Built.poison

    def poison(self):
        created[self.me] = fabric.created  # all members built before any install
        real(self)

    dense_dwdp._Built.poison = poison
    try:
        _run_members(fabric, 3, check, cls=_PlainLayer, prepare=lambda layer: None)
    finally:
        dense_dwdp._Built.poison = real
    assert fabric.created == max(created.values())  # nothing created at install


def test_window_transform_keeping_old_storage_raises(fabric, monkeypatch):
    """A transform whose result something outside the layer still references
    cannot be sharded silently: install raises."""
    held = []
    real = _prepare

    def prepare(layer):
        real(layer)
        held.append(layer.packed.data)  # outlives the move

    monkeypatch.setitem(globals(), "_prepare", prepare)
    with pytest.raises(RuntimeError, match="still referenced"):
        _run_members(fabric, num_layers=2, check=lambda model, ref: None)


# ---- GPU: the same equivalence over vLLM's real linear methods ----

_METHODS = ("bf16", "fp8_block", "mxfp8", "nvfp4")


def _quant_config(method):
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptMxFp8Config,
        ModelOptNvFp4Config,
    )

    if method == "bf16":
        return None
    if method == "fp8_block":
        return Fp8Config(
            is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128]
        )
    if method == "mxfp8":
        return ModelOptMxFp8Config(True, None, [])
    return ModelOptNvFp4Config(
        is_checkpoint_nvfp4_serialized=True,
        kv_cache_quant_algo=None,
        exclude_modules=[],
        group_size=16,
    )


class _RealLayer(nn.Module):
    """Two linears of one vLLM linear method (their own post-load processing:
    kernel layouts, scale swizzles, global-scale fusion), plus a resident BF16
    one."""

    def __init__(self, method, idx):
        from vllm.model_executor.layers.linear import ReplicatedLinear

        super().__init__()
        qc = _quant_config(method)
        self.up = ReplicatedLinear(
            512, 768, bias=False, params_dtype=torch.bfloat16, quant_config=qc,
            prefix=f"layers.{idx}.up", return_bias=False,
        )  # fmt: skip
        self.down = ReplicatedLinear(
            768, 512, bias=False, params_dtype=torch.bfloat16, quant_config=qc,
            prefix=f"layers.{idx}.down", return_bias=False,
        )  # fmt: skip


def _real_checkpoint(layer, seed):
    """A valid checkpoint tensor for every parameter (finite scales)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    out = {}
    for name, p in layer.named_parameters():
        shape, dt = tuple(p.shape), p.dtype
        if dt == torch.uint8 and "scale" in name:  # E8M0
            t = torch.randint(120, 130, shape, generator=g, device="cuda", dtype=dt)
        elif dt == torch.uint8:  # packed FP4
            t = torch.randint(0, 256, shape, generator=g, device="cuda", dtype=dt)
        elif dt == torch.float32:
            t = torch.rand(shape, generator=g, device="cuda") * 0.01 + 0.001
        else:
            t = (torch.randn(shape, generator=g, device="cuda") * 0.05).to(dt)
            if "scale" in name:
                t = t.float().abs().add(0.01).to(dt)
        out[name] = t
    return out


def _real_load(layers, first):
    for i, layer in enumerate(layers):
        weights = _real_checkpoint(layer, first + i)
        for name, p in layer.named_parameters():
            loader = getattr(p, "weight_loader", None)
            if loader is None:
                p.data.copy_(weights[name])
            else:
                loader(p, weights[name])


@pytest.fixture(scope="module")
def tp1():
    """A one-process TP / EP world for vLLM's linear layers."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.utils.network_utils import get_open_port

    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
            local_rank=0,
        )
        initialize_model_parallel(1, 1)
        yield
    cleanup_dist_env_and_memory()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real kernels need CUDA")
@pytest.mark.parametrize("method", _METHODS)
def test_load_windows_match_unsharded_real_methods(tp1, fabric, monkeypatch, method):
    """Per vLLM linear method: after each layer's forward fetch every member's
    parameters and the layer's output on one input equal the unsharded model's
    bitwise."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.utils.torch_utils import set_default_torch_dtype

    vllm_config = VllmConfig()
    # the FP8 linear method reads the model dtype
    object.__setattr__(
        vllm_config,
        "model_config",
        SimpleNamespace(dtype=torch.bfloat16, hf_text_config=None),
    )
    monkeypatch.setattr(dense_dwdp, "DEVICE_TYPE", "cuda")
    monkeypatch.setattr(torch.cuda, "synchronize", _SYNC)
    fabric.device = "cuda"
    num_layers, first = 4, 10
    with (
        set_current_vllm_config(vllm_config),
        set_default_torch_dtype(torch.bfloat16),
        torch.device("cuda"),
    ):
        ref = _Model([_RealLayer(method, first + i) for i in range(num_layers)])
        _real_load(ref.layers, first)
        members = []
        for me in range(N):
            dense_dwdp._B = None
            _TLS.me = me
            layers = []
            for i in range(num_layers):
                layer = _RealLayer(method, first + i)
                dense_dwdp.shard_layer(layer, first + i)
                layers.append(layer)
            members.append((layers, dense_dwdp._B))
            dense_dwdp._B = None
        for layers, built in members:
            _real_load(layers, first)
            built.poison()

    def no_prepare(layer):
        pass

    monkeypatch.setitem(globals(), "_prepare", no_prepare)
    _post_load(ref)
    models = [_Model(layers, built) for layers, built in members]
    errors = []

    def run(me):
        _TLS.me = me
        try:
            _post_load(models[me])
        except threading.BrokenBarrierError:
            pass
        except BaseException as e:
            errors.append(e)
            fabric.barrier.abort()

    threads = [threading.Thread(target=run, args=(me,), daemon=True) for me in range(N)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    if errors:
        raise errors[0]
    x = torch.randn(64, 512, device="cuda", dtype=torch.bfloat16)
    for model in models:
        for i in range(1, num_layers):
            model.dwdp.copies[i].fetch(None, None)
            got, want = model.layers[i], ref.layers[i]
            for (name, g), (_, w) in zip(
                got.named_parameters(), want.named_parameters()
            ):
                assert torch.equal(_bits(g), _bits(w)), (method, i, name)
            torch.cuda.synchronize()
            assert torch.equal(
                _bits(got.down(got.up(x))), _bits(want.down(want.up(x)))
            ), (method, i)


# ---- expert DWDP (two engines, EP x DWDP2): format-agnostic halves ----


def _expert_engine(fabric, index):
    """An ExpertDwdp of engine ``index`` on the fake fabric (no pairing dir)."""
    from vllm.models.kimi_k3.nvidia.megamoe_rubin import dwdp as edwdp

    x = object.__new__(edwdp.ExpertDwdp)
    x.index, x.device, x.die_placed, x.ep_rank = index, 0, False, 0
    x.streams, x.stream_handles = [None] * 4, [None] * 4
    member = fabric.store(SimpleNamespace(rank_in_group=index), members=2)
    x.store = SimpleNamespace(
        exchange=member.exchange, barrier=lambda tag: fabric.pair.wait()
    )
    x.order, x.slots, x.exports = [], {}, {}
    return x


def _full_experts(num_experts, layer, kind):
    """One layer's kernel tensors of every local expert (experts on dim 0) in
    several storage formats: packed FP4 bytes behind a permuted K-major view, E4M3
    and E8M0 scale planes, BF16."""
    g = torch.Generator().manual_seed(layer)
    w1 = torch.randint(0, 256, (num_experts, 64, 32), generator=g, dtype=torch.uint8)
    w2 = torch.randint(0, 256, (num_experts, 32, 32), generator=g, dtype=torch.uint8)
    if kind == "nvfp4":
        s1 = torch.randn(num_experts, 512, generator=g).to(torch.float8_e4m3fn)
        s2 = torch.randn(num_experts, 256, generator=g).to(torch.float8_e4m3fn)
    elif kind == "mxfp4":
        s1 = torch.randint(110, 140, (num_experts, 256), generator=g, dtype=torch.uint8)
        s2 = torch.randint(110, 140, (num_experts, 128), generator=g, dtype=torch.uint8)
    else:  # bf16
        s1 = torch.randn(num_experts, 512, generator=g).to(torch.bfloat16)
        s2 = torch.randn(num_experts, 256, generator=g).to(torch.bfloat16)
    return (w1.permute(0, 2, 1), s1), (w2.permute(0, 2, 1), s2)


@pytest.mark.parametrize("kind", ["nvfp4", "mxfp4", "bf16"])
def test_expert_halves_match_whole_experts(fabric, monkeypatch, kind):
    """Each engine holds the home half of every MoE layer after the first (they
    alternate per layer); after the partner's half is fetched every assembled
    kernel tensor equals the whole layer's tensor bitwise, with its view's shape
    and strides, whatever the storage format."""
    from vllm.models.kimi_k3.nvidia.megamoe_rubin import dwdp as edwdp
    from vllm.models.kimi_k3.nvidia.megamoe_rubin.weights import RubinMoeWeights

    for name in ("create", "export", "VaRange", "Copies"):
        monkeypatch.setattr(edwdp, name, getattr(dense_dwdp.edwdp, name))
    # finalize registers an exit barrier with the partner (a second process)
    monkeypatch.setattr(edwdp.atexit, "register", lambda fn: None)
    fabric.pair = threading.Barrier(2)
    experts, layers = 8, 4
    engines = [_expert_engine(fabric, i) for i in range(2)]
    out = [[], []]
    for e, x in enumerate(engines):
        for layer in range(layers):
            module = SimpleNamespace()
            module._dwdp_layer, lo, hi = x.claim(experts)
            fc1, fc2 = _full_experts(experts, layer, kind)
            weights = RubinMoeWeights(
                kind,
                tuple(t[lo:hi] for t in fc1),
                tuple(t[lo:hi] for t in fc2),
                fc1_alpha=torch.ones(experts),
            )
            out[e].append(x.register(module, weights))
    fabric.barrier = threading.Barrier(2)
    threads = [threading.Thread(target=x.finalize, daemon=True) for x in engines]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    for e, x in enumerate(engines):
        for layer in range(layers):
            x._fetch(x.order[layer]) if layer else None
            fc1, fc2 = _full_experts(experts, layer, kind)
            got = out[e][layer]
            for g_t, w_t in zip((*got.fc1, *got.fc2), (*fc1, *fc2)):
                assert g_t.shape == w_t.shape and g_t.stride() == w_t.stride()
                assert torch.equal(_bits(g_t.contiguous()), _bits(w_t.contiguous()))
            assert torch.equal(got.fc1_alpha, torch.ones(experts))


# ---- shared-expert selector (TP > 1 SP prefill worker): replicated shared experts ----


class _RLinear(_Linear):
    """A replicated linear (``disable_tp``: tp_size 1)."""

    tp_size = 1


class _TpLinear(_Linear):
    """A TP-sharded linear: each member holds a different slice."""

    tp_size = 8


class _SharedExpert(nn.Module):
    """The SP shared expert: replicated MXFP8 gate_up / down (weight + E8M0 scale,
    scales swizzled in place by the kernel's processing)."""

    def __init__(self, linear=_RLinear):
        super().__init__()
        self.gate_up_proj = linear(256, 256, torch.float8_e4m3fn, _mxfp8, True)
        self.down_proj = linear(128, 256, torch.float8_e4m3fn, _mxfp8, True)


class _Moe(nn.Module):
    def __init__(self, linear=_RLinear):
        super().__init__()
        self.shared_experts = _SharedExpert(linear)
        self.gate = _RLinear(64, 64)  # replicated router: not selected
        self.routed_expert_down_proj = _RLinear(64, 64)  # replicated: not selected


class _SpLayer(nn.Module):
    """A TP8 SP decoder layer: TP-sharded attention, a MoE with a replicated shared
    expert, router and latent projection."""

    def __init__(self, linear=_RLinear):
        super().__init__()
        self.attn = _TpLinear(128, 64, fn=_transpose)
        self.mlp = _Moe(linear)


def _select(layer):
    return getattr(getattr(layer, "mlp", None), "shared_experts", None)


@pytest.fixture
def shared_fabric(fabric, monkeypatch):
    """The fake fabric with the shared-expert selector on (no dwdp mode)."""
    monkeypatch.setattr(dense_dwdp.edwdp, "dense_enabled", lambda: False)
    monkeypatch.setattr(dense_dwdp, "_SELECT", _select)
    return fabric


def _shared_params(layer):
    return {f"mlp.shared_experts.{n}": p
            for n, p in layer.mlp.shared_experts.named_parameters()}  # fmt: skip


def test_shared_selector_shards_only_the_shared_expert(shared_fabric):
    """Under the shared-expert selector a member's construction blob holds exactly
    the shared expert's storages (weights and scales); the TP-sharded attention and
    the other replicated weights (router, latent projection) keep their own storage,
    and no vocabulary table is sharded (they are TP-sharded under TP > 1)."""
    layers, built = _build(me=1, cls=_SpLayer)
    assert built.start == 10 and sorted(built.layers) == [11, 12]
    blob, shard_bytes, regions = built.layers[11]
    target = blob.untyped_storage().data_ptr()
    shared = _shared_params(layers[1])
    assert len(shared) == 4 and len(regions) == 4
    for name, p in layers[1].named_parameters():
        moved = p.untyped_storage().data_ptr() == target
        assert moved == (name in shared), name
    eligible = sum(dense_dwdp.aligned(p.nbytes) for p in shared.values())
    assert shard_bytes == dense_dwdp.aligned(-(-eligible // N), CHUNK)
    # the first layer stays resident
    assert all(
        p.untyped_storage().data_ptr() not in {b.untyped_storage().data_ptr()
                                               for b, _, _ in built.layers.values()}
        for p in layers[0].parameters()
    )  # fmt: skip
    table = nn.Parameter(torch.zeros(1000, 64, dtype=torch.bfloat16), False)
    ptr = table.data_ptr()
    dense_dwdp.shard_table(table)
    assert table.data_ptr() == ptr and dense_dwdp._B is None
    dense_dwdp._B = None


def test_shared_selector_never_takes_tp_sharded_weights(shared_fabric):
    """A selected module that owns a TP-sharded (non-identical) weight, or a
    parameter outside a linear, raises at construction: never sharded silently."""
    with pytest.raises(RuntimeError, match="not a replicated linear"):
        _build(me=0, cls=lambda: _SpLayer(linear=_TpLinear))
    dense_dwdp._B = None

    class _Odd(_SpLayer):
        def __init__(self):
            super().__init__()
            self.mlp.shared_experts.norm = nn.Parameter(torch.ones(1 << 20), False)

    with pytest.raises(RuntimeError, match="not a replicated linear"):
        _build(me=0, cls=_Odd)
    dense_dwdp._B = None


def test_shared_selector_exclusive_with_dwdp_modes(monkeypatch):
    monkeypatch.setattr(dense_dwdp.edwdp, "dense_enabled", lambda: True)
    with pytest.raises(ValueError, match="exclusive"):
        dense_dwdp.configure_shared(_select)
    dense_dwdp.configure_shared(None)
    assert dense_dwdp._SELECT is None


def test_shared_selector_load_windows_match_unsharded(shared_fabric):
    """Every member processes each layer in its load window (scale swizzles on the
    fetched whole shared expert); after a layer's fetch every parameter equals the
    unsharded processing bitwise, the non-selected ones never moved, and the copies
    start at the MegaMoE launch."""

    def check(model, ref):
        dwdp = model.dwdp
        assert dwdp.issue_at_moe
        blobs = {
            b.untyped_storage().data_ptr() for b, _, _ in model.built.layers.values()
        }
        for i in range(1, len(model.layers)):
            dwdp.copies[i].fetch(None, None)
            got = dict(model.layers[i].named_parameters())
            want = dict(ref.layers[i].named_parameters())
            assert sorted(got) == sorted(want)
            for name in want:
                g, w = got[name], want[name]
                assert g.shape == w.shape and g.stride() == w.stride(), name
                assert torch.equal(_bits(g), _bits(w)), (i, name)
                moved = g.untyped_storage().data_ptr() in blobs
                assert moved == name.startswith("mlp.shared_experts."), (i, name)

    _run_members(shared_fabric, 4, check, cls=_SpLayer, prepare=lambda layer: None)


class _Sched:
    """Copies that record when they are fetched / waited."""

    log: list = []

    def __init__(self, i):
        self.i, self.in_flight = i, False

    def fetch(self, streams, handles):
        if not self.in_flight:
            self.in_flight = True
            _Sched.log.append(("fetch", self.i))

    def wait(self):
        if self.in_flight:
            self.in_flight = False
            _Sched.log.append(("wait", self.i))


def _sched(issue_at_moe, layers=6):
    d = object.__new__(dense_dwdp.DenseDwdp)
    d.start, d.streams, d.handles = 0, None, None
    d.copies = [_Sched(i) for i in range(layers)]
    d.copies[0].fetch = lambda *a: None  # resident first layer: no copies
    d.issue_at_moe, d._ordinal = issue_at_moe, None
    return d


@pytest.mark.parametrize("moe_layers", ["all", "none", "odd"])
def test_shared_selector_issue_point(moe_layers):
    """Shared-expert selector: layer i + 2's copies start at layer i's MegaMoE
    launch (not at a layer start, where the SP all-gather runs), every layer waits
    for its own before it starts, a slot (i % 3) is refilled only after the layer
    that last read it (i - 3) has finished, and a layer whose predecessor launched
    no MegaMoE still gets its copies (at its predecessor's start). Repeated
    forwards and MegaMoE sub-calls fetch each layer once."""
    for _ in range(2):  # a second forward starts from the same state
        _Sched.log = []
        d = _sched(issue_at_moe=True)
        d.moe_start()  # outside a forward: nothing
        assert _Sched.log == []
        for i in range(6):
            d.layer_start(i)
            if moe_layers == "all" or (moe_layers == "odd" and i % 2):
                _Sched.log.append(("moe", i))
                d.moe_start()
                d.moe_start()  # a second sub-call: no second copy
            _Sched.log.append(("layer", i))
        log = _Sched.log
        assert sorted(i for kind, i in log if kind == "fetch") == [1, 2, 3, 4, 5]
        for i in range(1, 6):
            f, w = log.index(("fetch", i)), log.index(("wait", i))
            assert f < w < log.index(("layer", i))
            if i >= 3:
                assert f > log.index(("layer", i - 3)), (i, log)  # slot reuse
            if i >= 3 and ("moe", i - 2) in log:
                assert log[f - 1] == ("moe", i - 2), (i, log)  # at the MegaMoE
            elif i >= 3:  # no MegaMoE in layer i - 2: at layer i - 1's start
                assert log[f - 1] == ("wait", i - 1), (i, log)


def test_whole_layer_dwdp_issue_point_unchanged():
    """dwdp="dense": copies start at the layer start, depth - 1 ahead; the MegaMoE
    launch hook does nothing."""
    _Sched.log = []
    d = _sched(issue_at_moe=False)
    for i in range(6):
        d.layer_start(i)
        d.moe_start()
        _Sched.log.append(("layer", i))
    assert _Sched.log[:3] == [("fetch", 1), ("fetch", 2), ("layer", 0)]
    assert _Sched.log[3:6] == [("wait", 1), ("fetch", 3), ("layer", 1)]
