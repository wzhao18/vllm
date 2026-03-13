# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Common utilities for attention benchmarking."""

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from batch_spec import get_batch_type, parse_batch_spec
from rich.console import Console
from rich.table import Table


def batch_spec_sort_key(spec: str) -> tuple[int, int, int]:
    """
    Extract sorting key from batch spec: (batch_size, max_q_len, max_kv_len).

    This ensures results are sorted by batch size first, then query length,
    then sequence length, rather than alphabetically.
    """
    try:
        requests = parse_batch_spec(spec)
        batch_size = len(requests)
        max_q_len = max(r.q_len for r in requests) if requests else 0
        max_kv_len = max(r.kv_len for r in requests) if requests else 0
        return (batch_size, max_q_len, max_kv_len)
    except Exception:
        # Fallback for unparsable specs
        return (0, 0, 0)


# Mock classes for vLLM attention infrastructure


class MockHfConfig:
    """Mock HuggingFace config that satisfies vLLM's requirements."""

    def __init__(self, mla_dims: dict, index_topk: int | None = None):
        self.num_attention_heads = mla_dims["num_q_heads"]
        self.num_key_value_heads = mla_dims["num_kv_heads"]
        self.hidden_size = mla_dims["head_dim"] * mla_dims["num_q_heads"]
        self.model_type = "deepseek_v2"
        self.is_encoder_decoder = False
        self.kv_lora_rank = mla_dims["kv_lora_rank"]
        self.qk_nope_head_dim = mla_dims["qk_nope_head_dim"]
        self.qk_rope_head_dim = mla_dims["qk_rope_head_dim"]
        self.v_head_dim = mla_dims["v_head_dim"]
        self.qk_head_dim = mla_dims["qk_nope_head_dim"] + mla_dims["qk_rope_head_dim"]
        if index_topk is not None:
            self.index_topk = index_topk

    def get_text_config(self):
        return self


# Import AttentionLayerBase at module level to avoid circular dependencies
try:
    from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
except ImportError:
    AttentionLayerBase = object  # Fallback


class MockKVBProj:
    """Mock KV projection layer for MLA prefill mode.

    Mimics ColumnParallelLinear behavior for kv_b_proj in MLA backends.
    Projects kv_c_normed ([num_tokens, kv_lora_rank]) to
    ([num_tokens, num_heads * (qk_nope_head_dim + v_head_dim)]).

    Supports two modes controlled by self._use_real_weight:
    - False (default, benchmarking): __call__ returns random output with no
      matmul, so forward_mha benchmark timing is unaffected.
    - True (correctness check): __call__ does the real float32 matmul so
      forward_mha produces numerically correct output for comparison.

    A deterministic weight (seed=0) is always stored in self.weight so the
    SDPA reference can reproduce the projection independently.
    """

    def __init__(
        self,
        num_heads: int,
        qk_nope_head_dim: int,
        v_head_dim: int,
        kv_lora_rank: int,
        device: torch.device | str = "cpu",
    ):
        self.num_heads = num_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.v_head_dim = v_head_dim
        self.out_dim = qk_nope_head_dim + v_head_dim
        self.quant_method = None
        # Toggle to switch between real projection (correctness) and random
        # output (benchmark).  Python's special-method dispatch always goes
        # through the class, so toggling a flag inside __call__ is the only
        # reliable way to change behavior without monkey-patching the class.
        self._use_real_weight = False
        out_features = num_heads * self.out_dim
        # Deterministic weight — fork_rng isolates from global random state.
        with torch.random.fork_rng():
            torch.manual_seed(0)
            self.weight = torch.randn(
                out_features,
                kv_lora_rank,
                dtype=torch.bfloat16,
                device=device,
            ) / (kv_lora_rank**0.5)

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Project x or return random output depending on _use_real_weight.

        Args:
            x: Input tensor [num_tokens, kv_lora_rank]

        Returns:
            (output, None) matching the ColumnParallelLinear API,
            where output has shape [num_tokens, num_heads * out_dim].
        """
        if self._use_real_weight:
            # Float32 matmul for numerical accuracy during correctness checks.
            return (
                (x.float() @ self.weight.T.float()).to(x.dtype),
                None,
            )
        return (
            torch.randn(
                x.shape[0],
                self.num_heads * self.out_dim,
                device=x.device,
                dtype=x.dtype,
            ),
            None,
        )


class MockIndexer:
    """Mock Indexer for sparse MLA backends.

    Provides topk_indices_buffer that sparse MLA backends use to determine
    which KV cache slots to attend to for each token.
    """

    def __init__(
        self,
        max_num_tokens: int,
        topk_tokens: int,
        device: torch.device,
    ):
        self.topk_tokens = topk_tokens
        self.topk_indices_buffer = torch.zeros(
            (max_num_tokens, topk_tokens),
            dtype=torch.int32,
            device=device,
        )

    def fill_random_indices(self, num_tokens: int, max_kv_len: int):
        """Fill topk_indices_buffer with random valid indices for benchmarking."""
        indices = torch.randint(
            0,
            max_kv_len,
            (num_tokens, self.topk_tokens),
            dtype=torch.int32,
            device=self.topk_indices_buffer.device,
        )
        self.topk_indices_buffer[:num_tokens] = indices


class MockLayer(AttentionLayerBase):
    """Mock attention layer with scale parameters and impl.

    Inherits from AttentionLayerBase so it passes isinstance checks
    in get_layers_from_vllm_config when FlashInfer prefill is enabled.
    """

    def __init__(self, device: torch.device, impl=None, kv_cache_spec=None):
        # Don't call super().__init__() as AttentionLayerBase doesn't have __init__
        self._k_scale = torch.tensor(1.0, device=device)
        self._v_scale = torch.tensor(1.0, device=device)
        self._q_scale = torch.tensor(1.0, device=device)
        # Scalar floats for kernels that need them
        self._k_scale_float = float(self._k_scale.item())
        self._v_scale_float = float(self._v_scale.item())
        self._q_scale_float = float(self._q_scale.item())
        # AttentionImpl for metadata builders to query
        self.impl = impl
        # KV cache spec for get_kv_cache_spec
        self._kv_cache_spec = kv_cache_spec

    def get_attn_backend(self):
        """Get the attention backend class (required by AttentionLayerBase)."""
        # Return None as this is just a mock layer for benchmarking
        return None

    def get_kv_cache_spec(self):
        """Get the KV cache spec (required by AttentionLayerBase)."""
        return self._kv_cache_spec


@dataclass
class ParameterSweep:
    """Configuration for sweeping a backend parameter."""

    param_name: str  # Name of the backend parameter to sweep
    values: list[Any]  # List of values to test
    include_auto: bool = False  # Also test with param unset (auto mode)
    label_format: str = "{backend}_{param_name}_{value}"  # Result label template

    def get_label(self, backend: str, value: Any) -> str:
        """Generate a label for a specific parameter value."""
        return self.label_format.format(
            backend=backend, param_name=self.param_name, value=value
        )


@dataclass
class ModelParameterSweep:
    """Configuration for sweeping a model configuration parameter."""

    param_name: str  # Name of the model config parameter to sweep (e.g., "num_q_heads")
    values: list[Any]  # List of values to test
    label_format: str = "{backend}_{param_name}_{value}"  # Result label template

    def get_label(self, backend: str, value: Any) -> str:
        """Generate a label for a specific parameter value."""
        return self.label_format.format(
            backend=backend, param_name=self.param_name, value=value
        )


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""

    backend: str
    batch_spec: str
    num_layers: int
    head_dim: int
    num_q_heads: int
    num_kv_heads: int
    block_size: int
    device: str
    dtype: torch.dtype = torch.float16
    repeats: int = 1
    warmup_iters: int = 3
    profile_memory: bool = False
    use_cuda_graphs: bool = False

    # MLA-specific
    prefill_backend: str | None = None
    kv_lora_rank: int | None = None
    qk_nope_head_dim: int | None = None
    qk_rope_head_dim: int | None = None
    v_head_dim: int | None = None

    # "bfloat16" or "fp8"
    kv_cache_dtype: str = "bfloat16"

    # Backend-specific tuning
    num_kv_splits: int | None = None  # CUTLASS MLA
    reorder_batch_threshold: int | None = None  # FlashAttn MLA, FlashMLA


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    config: BenchmarkConfig
    mean_time: float  # seconds
    std_time: float  # seconds
    min_time: float  # seconds
    max_time: float  # seconds
    throughput_tokens_per_sec: float | None = None
    memory_allocated_mb: float | None = None
    memory_reserved_mb: float | None = None
    error: str | None = None

    @property
    def success(self) -> bool:
        """Whether benchmark completed successfully."""
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "config": asdict(self.config),
            "mean_time": self.mean_time,
            "std_time": self.std_time,
            "min_time": self.min_time,
            "max_time": self.max_time,
            "throughput_tokens_per_sec": self.throughput_tokens_per_sec,
            "memory_allocated_mb": self.memory_allocated_mb,
            "memory_reserved_mb": self.memory_reserved_mb,
            "error": self.error,
        }


class ResultsFormatter:
    """Format and display benchmark results."""

    def __init__(self, console: Console | None = None):
        self.console = console or Console()

    def print_table(
        self,
        results: list[BenchmarkResult],
        backends: list[str],
        compare_to_fastest: bool = True,
    ):
        """
        Print results as a rich table.

        Args:
            results: List of BenchmarkResult
            backends: List of backend names being compared
            compare_to_fastest: Show percentage comparison to fastest
        """
        # Group by batch spec, preserving first-occurrence order
        by_spec = {}
        specs_order = []
        for r in results:
            spec = r.config.batch_spec
            if spec not in by_spec:
                by_spec[spec] = {}
                specs_order.append(spec)
            by_spec[spec][r.config.backend] = r

        # Sort specs by (batch_size, q_len, kv_len) instead of alphabetically
        specs_order = sorted(by_spec.keys(), key=batch_spec_sort_key)

        # Create shortened backend names for display
        def shorten_backend_name(name: str) -> str:
            """Shorten long backend names for table display."""
            # Remove common prefixes
            name = name.replace("flashattn_mla", "famla")
            name = name.replace("flashinfer_mla", "fimla")
            name = name.replace("flashmla", "fmla")
            name = name.replace("cutlass_mla", "cmla")
            name = name.replace("numsplits", "ns")
            return name

        table = Table(title="Attention Benchmark Results")
        table.add_column("Batch\nSpec", no_wrap=True)
        table.add_column("Type", no_wrap=True)
        table.add_column("Batch\nSize", justify="right", no_wrap=True)

        multi = len(backends) > 1
        for backend in backends:
            short_name = shorten_backend_name(backend)
            # Time column
            col_time = f"{short_name}\nTime (s)"
            table.add_column(col_time, justify="right", no_wrap=False)
            if multi and compare_to_fastest:
                # Relative performance column
                col_rel = f"{short_name}\nvs Best"
                table.add_column(col_rel, justify="right", no_wrap=False)

        # Add rows
        for spec in specs_order:
            spec_results = by_spec[spec]
            times = {b: r.mean_time for b, r in spec_results.items() if r.success}
            best_time = min(times.values()) if times else 0.0

            batch_type = get_batch_type(spec)
            batch_size = len(parse_batch_spec(spec))
            row = [spec, batch_type, str(batch_size)]
            for backend in backends:
                if backend in spec_results:
                    r = spec_results[backend]
                    if r.success:
                        row.append(f"{r.mean_time:.6f}")
                        if multi and compare_to_fastest:
                            pct = (
                                (r.mean_time / best_time * 100) if best_time > 0 else 0
                            )
                            pct_str = f"{pct:.1f}%"
                            if r.mean_time == best_time:
                                pct_str = f"[bold green]{pct_str}[/]"
                            row.append(pct_str)
                    else:
                        row.append("[red]ERROR[/]")
                        if multi and compare_to_fastest:
                            row.append("-")
                else:
                    row.append("-")
                    if multi and compare_to_fastest:
                        row.append("-")

            table.add_row(*row)

        self.console.print(table)

    def save_csv(self, results: list[BenchmarkResult], path: str):
        """Save results to CSV file."""
        if not results:
            return

        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "backend",
                    "batch_spec",
                    "num_layers",
                    "mean_time",
                    "std_time",
                    "throughput",
                    "memory_mb",
                ],
            )
            writer.writeheader()
            for r in results:
                writer.writerow(
                    {
                        "backend": r.config.backend,
                        "batch_spec": r.config.batch_spec,
                        "num_layers": r.config.num_layers,
                        "mean_time": r.mean_time,
                        "std_time": r.std_time,
                        "throughput": r.throughput_tokens_per_sec or 0,
                        "memory_mb": r.memory_allocated_mb or 0,
                    }
                )

        self.console.print(f"[green]Saved CSV results to {path}[/]")

    def save_json(self, results: list[BenchmarkResult], path: str):
        """Save results to JSON file."""
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        data = [r.to_dict() for r in results]
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        self.console.print(f"[green]Saved JSON results to {path}[/]")


def setup_mla_dims(model_name: str = "deepseek-v3") -> dict:
    """
    Get MLA dimensions for known models.

    Args:
        model_name: Model identifier

    Returns:
        Dict with MLA dimension configuration
    """
    configs = {
        "deepseek-v2": {
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
            "num_q_heads": 128,
            "num_kv_heads": 1,
            "head_dim": 576,
        },
        "deepseek-v3": {
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
            "num_q_heads": 128,
            "num_kv_heads": 1,
            "head_dim": 576,
        },
        "deepseek-v2-lite": {
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
            "num_q_heads": 16,
            "num_kv_heads": 1,
            "head_dim": 576,
        },
    }

    if model_name not in configs:
        raise ValueError(
            f"Unknown model '{model_name}'. Known models: {list(configs.keys())}"
        )

    return configs[model_name]


def get_attention_scale(head_dim: int) -> float:
    """Compute attention scale factor (1/sqrt(d))."""
    return 1.0 / math.sqrt(head_dim)


def is_mla_backend(backend: str) -> bool:
    """
    Check if backend is an MLA backend using the AttentionBackendEnum.

    Args:
        backend: Backend name matching AttentionBackendEnum exactly
        (e.g., "FLASHMLA_SPARSE")

    Returns:
        True if the backend is an MLA backend, False otherwise
    """
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    try:
        backend_enum = AttentionBackendEnum[backend]
        backend_class = backend_enum.get_class()
        return backend_class.is_mla()
    except (KeyError, ValueError, ImportError, AttributeError):
        return False


# ============================================================================
# KV cache population and decoding helpers
# ============================================================================


def populate_mla_kv_cache(
    kv_cache: torch.Tensor,
    prefill_inputs: dict,
    metadata,
    block_size: int,
    kv_lora_rank: int,
    kv_cache_dtype: str,
) -> None:
    """Write new-token KV data into kv_cache at slot positions.

    Handles bfloat16, fp8, and fp8_ds_mla (DeepSeek 656-byte packed format).
    """
    from vllm.platforms import current_platform

    _sm = metadata.slot_mapping
    _blocks = (_sm // block_size).long()
    _offsets = (_sm % block_size).long()
    k_c = prefill_inputs["k_c_normed"]  # [total_tokens, kv_lora_rank]
    k_p = prefill_inputs["k_pe"][:, 0, :]  # [total_tokens, qk_rope_head_dim]

    if kv_cache_dtype == "fp8_ds_mla":
        fp8_dtype = current_platform.fp8_dtype()
        fp8_max = torch.finfo(fp8_dtype).max
        k_c_f = k_c.float()
        num_groups = kv_lora_rank // 128
        fp8_parts: list[torch.Tensor] = []
        scale_parts: list[torch.Tensor] = []
        for j in range(num_groups):
            chunk = k_c_f[:, j * 128 : (j + 1) * 128]
            scale = chunk.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / fp8_max
            scale_parts.append(scale.to(torch.float32))  # [N, 1] f32
            fp8_parts.append((chunk / scale).to(fp8_dtype).view(torch.uint8))
        fp8_bytes = torch.cat(fp8_parts, dim=-1)  # [N, kv_lora_rank] u8
        scale_bytes = torch.cat(scale_parts, dim=-1).view(torch.uint8)  # [N, num_groups*4] u8
        rope_bytes = k_p.to(torch.bfloat16).view(torch.uint8)  # [N, rope_dim*2] u8
        token_data = torch.cat([fp8_bytes, scale_bytes, rope_bytes], dim=-1)
        kv_cache[_blocks, _offsets] = token_data
    elif kv_cache_dtype == "fp8":
        fp8_dtype = current_platform.fp8_dtype()
        kv_cache[_blocks, _offsets, :kv_lora_rank] = k_c.to(fp8_dtype)
        kv_cache[_blocks, _offsets, kv_lora_rank:] = k_p.to(fp8_dtype)
    else:  # bfloat16
        kv_cache[_blocks, _offsets, :kv_lora_rank] = k_c.to(kv_cache.dtype)
        kv_cache[_blocks, _offsets, kv_lora_rank:] = k_p.to(kv_cache.dtype)
    torch.accelerator.synchronize()


def mla_kv_cache_to_bf16(
    kv_cache: torch.Tensor,
    kv_cache_dtype: str,
    mla_dims: dict,
) -> torch.Tensor:
    """Return a bfloat16 [num_blocks, block_size, kv_lora_rank+qk_rope_head_dim]
    tensor suitable for passing to compute_mqa_reference / compute_mha_reference.

    For bfloat16 caches the original tensor is returned unchanged.
    For fp8, standard dequantization via .float() is used.
    For fp8_ds_mla the 656-byte packed format is decoded: 512 fp8 values
    (with per-128 float32 scales) followed by 64 unquantized bfloat16 rope values.
    """
    if kv_cache_dtype not in ("fp8", "fp8_ds_mla"):
        return kv_cache

    from vllm.platforms import current_platform

    kv_lora_rank = mla_dims["kv_lora_rank"]
    qk_rope_head_dim = mla_dims["qk_rope_head_dim"]
    num_blocks, block_size = kv_cache.shape[:2]

    if kv_cache_dtype == "fp8":
        return kv_cache.float().bfloat16()

    # fp8_ds_mla: [num_blocks, block_size, 656] uint8
    fp8_dtype = current_platform.fp8_dtype()
    num_groups = kv_lora_rank // 128
    flat = kv_cache.reshape(num_blocks * block_size, 656)  # [N, 656] u8

    # First kv_lora_rank bytes are fp8 values
    fp8_vals = flat[:, :kv_lora_rank].view(fp8_dtype).float()  # [N, 512]
    # Next num_groups*4 bytes are float32 per-128 scales
    scale_start = kv_lora_rank
    scale_end = scale_start + num_groups * 4
    scales = flat[:, scale_start:scale_end].view(torch.float32)  # [N, num_groups]
    for j in range(num_groups):
        fp8_vals[:, j * 128 : (j + 1) * 128] *= scales[:, j : j + 1]
    # Remaining bytes are bfloat16 rope values
    k_pe_vals = flat[:, scale_end:].view(torch.bfloat16).float()  # [N, 64]

    result = torch.cat([fp8_vals, k_pe_vals], dim=-1).bfloat16()
    return result.reshape(num_blocks, block_size, kv_lora_rank + qk_rope_head_dim)


# ============================================================================
# Correctness helpers
# ============================================================================


def sdpa_for_request(
    q: torch.Tensor,  # [q_len, N, D_q]
    k: torch.Tensor,  # [kv_len, N, D_q]
    v: torch.Tensor,  # [kv_len, N, D_v]
    ctx_len: int,
    scale: float,
    device: torch.device,
) -> torch.Tensor:
    """Causal per-request SDPA. Returns [q_len, N, D_v]."""
    q_len = q.shape[0]
    kv_len = k.shape[0]

    attn_mask = torch.zeros(q_len, kv_len, dtype=torch.bool, device=device)
    attn_mask[:, :ctx_len] = True
    causal = torch.tril(torch.ones(q_len, q_len, dtype=torch.bool, device=device))
    attn_mask[:, ctx_len:] = causal

    out = torch.nn.functional.scaled_dot_product_attention(
        q.unsqueeze(0).transpose(1, 2),
        k.unsqueeze(0).transpose(1, 2),
        v.unsqueeze(0).transpose(1, 2),
        attn_mask=attn_mask,
        scale=scale,
    )
    return out.squeeze(0).transpose(0, 1)  # [q_len, N, D_v]


# ---- MLA correctness helpers ----


def compute_mqa_reference(
    decode_inputs,  # tuple (q_nope [T,N,L], q_pe [T,N,R]) or concat [T,N,L+R]
    kv_cache: torch.Tensor,  # [num_blocks, block_size, kv_lora_rank + qk_rope_head_dim]
    metadata,
    mla_dims: dict,
    is_sparse: bool = False,
) -> torch.Tensor:
    """SDPA reference for forward_mqa. Called after the forward pass.

    Reads K/V directly from kv_cache (context=zeros, new token written by forward).
    Returns [total_decode_tokens, num_q_heads, kv_lora_rank].
    """
    device = kv_cache.device
    kv_lora_rank = mla_dims["kv_lora_rank"]
    qk_rope_head_dim = mla_dims["qk_rope_head_dim"]
    qk_nope_head_dim = mla_dims["qk_nope_head_dim"]
    num_q_heads = mla_dims["num_q_heads"]
    scale = 1.0 / math.sqrt(qk_nope_head_dim + qk_rope_head_dim)
    block_size = kv_cache.shape[1]

    if isinstance(decode_inputs, tuple):
        q = torch.cat([decode_inputs[0], decode_inputs[1]], dim=-1).float()
    else:
        q = decode_inputs.float()

    if is_sparse:
        block_table = metadata.block_table
        seq_lens = metadata.seq_lens
        query_start_loc = metadata.query_start_loc
    else:
        block_table = metadata.decode.block_table
        seq_lens = metadata.decode.seq_lens
        batch_size = int(seq_lens.shape[0])
        query_start_loc = torch.arange(batch_size + 1, device=device)

    out_parts: list[torch.Tensor] = []
    for i in range(int(seq_lens.shape[0])):
        q_start = int(query_start_loc[i].item())
        q_end = int(query_start_loc[i + 1].item())
        kv_len = int(seq_lens[i].item())
        ctx_len = kv_len - (q_end - q_start)

        t = torch.arange(kv_len, device=device)
        blk = block_table[i, t // block_size].long()
        off = (t % block_size).long()
        entries = kv_cache[blk, off].float()  # [kv_len, L+R]
        kv_c = entries[:, :kv_lora_rank]
        k_pe = entries[:, kv_lora_rank:]

        k = torch.cat([kv_c, k_pe], dim=-1).unsqueeze(1).expand(-1, num_q_heads, -1)
        v = kv_c.unsqueeze(1).expand(-1, num_q_heads, -1)
        out_parts.append(
            sdpa_for_request(q[q_start:q_end], k, v, ctx_len, scale, device)
        )

    return torch.cat(out_parts, dim=0)


def compute_mha_reference(
    q: torch.Tensor,  # [total_prefill_tokens, N, qk_nope_head_dim + qk_rope_head_dim]
    kv_cache: torch.Tensor,  # [num_blocks, block_size, kv_lora_rank + qk_rope_head_dim]
    metadata,
    kv_b_proj_weight: torch.Tensor,  # [N*(qk_nope_head_dim + v_head_dim), kv_lora_rank]
    mla_dims: dict,
) -> torch.Tensor:
    """SDPA reference for forward_mha. Called after the forward pass.

    Reads K/V directly from kv_cache (context=zeros, new tokens written by forward).
    Returns [total_prefill_tokens, N * v_head_dim].
    """
    device = kv_cache.device
    kv_lora_rank = mla_dims["kv_lora_rank"]
    qk_nope_head_dim = mla_dims["qk_nope_head_dim"]
    qk_rope_head_dim = mla_dims["qk_rope_head_dim"]
    v_head_dim = mla_dims["v_head_dim"]
    num_q_heads = mla_dims["num_q_heads"]
    scale = 1.0 / math.sqrt(qk_nope_head_dim + qk_rope_head_dim)
    block_size = kv_cache.shape[1]

    prefill_meta = metadata.prefill
    block_table = prefill_meta.block_table
    query_start_loc = prefill_meta.query_start_loc
    chunked_context = prefill_meta.chunked_context

    w = kv_b_proj_weight.float()
    out_parts: list[torch.Tensor] = []
    for i in range(block_table.shape[0]):
        q_start = int(query_start_loc[i].item())
        q_end = int(query_start_loc[i + 1].item())
        q_len = q_end - q_start
        if chunked_context is not None:
            # seq_lens is 2D [num_chunks, num_prefills]; sum over chunks
            ctx_len = int(chunked_context.seq_lens[:, i].sum().item())
        else:
            ctx_len = 0
        kv_len = q_len + ctx_len

        t = torch.arange(kv_len, device=device)
        blk = block_table[i, t // block_size].long()
        off = (t % block_size).long()
        entries = kv_cache[blk, off].float()  # [kv_len, L+R]
        kv_c_seq = entries[:, :kv_lora_rank]
        k_pe_seq = entries[:, kv_lora_rank:]

        kv_exp = (kv_c_seq @ w.T).view(
            kv_len, num_q_heads, qk_nope_head_dim + v_head_dim
        )
        k = torch.cat(
            [
                kv_exp[:, :, :qk_nope_head_dim],
                k_pe_seq.unsqueeze(1).expand(-1, num_q_heads, -1),
            ],
            dim=-1,
        )
        v = kv_exp[:, :, qk_nope_head_dim:]
        out_parts.append(
            sdpa_for_request(
                q[q_start:q_end].float(), k, v, ctx_len, scale, device
            ).flatten(start_dim=-2)
        )

    return torch.cat(out_parts, dim=0)


# ---- Standard attention correctness helpers ----


def _extract_seq_info(
    attn_metadata,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract (seq_lens, query_start_loc) from metadata.

    Handles multiple metadata types:
    - FlashAttentionMetadata / TritonAttentionMetadata: seq_lens at top level
    - FlashInferMetadata: seq_lens nested in decode/prefill sub-metadata
    """
    if hasattr(attn_metadata, "seq_lens") and attn_metadata.seq_lens is not None:
        return attn_metadata.seq_lens, attn_metadata.query_start_loc

    # FlashInfer: seq_lens in decode/prefill sub-metadata.
    # Decode requests are reordered first by reorder_for_flashinfer.
    seq_lens_parts: list[torch.Tensor] = []
    q_start_list: list[int] = [0]
    dc = getattr(attn_metadata, "decode", None)
    pf = getattr(attn_metadata, "prefill", None)
    if dc is not None:
        seq_lens_parts.append(dc.seq_lens)
        for _ in range(int(dc.seq_lens.shape[0])):
            q_start_list.append(q_start_list[-1] + 1)  # decode: q_len = 1
    if pf is not None:
        seq_lens_parts.append(pf.seq_lens)
        deltas = (pf.cum_seq_lens_q[1:] - pf.cum_seq_lens_q[:-1]).tolist()
        for dq in deltas:
            q_start_list.append(q_start_list[-1] + int(dq))
    if not seq_lens_parts:
        raise AttributeError(f"Cannot find seq_lens in {type(attn_metadata).__name__}")
    seq_lens = torch.cat(seq_lens_parts)
    query_start_loc = torch.tensor(q_start_list, dtype=torch.int32, device=device)
    return seq_lens, query_start_loc


def compute_std_attn_reference(
    q: torch.Tensor,  # [total_q, num_q_heads, head_size]
    k_new: torch.Tensor,  # [total_q, num_kv_heads, head_size]
    v_new: torch.Tensor,  # [total_q, num_kv_heads, head_size]
    cache: torch.Tensor,  # kv_cache (unused; context is zero-initialized)
    attn_metadata,
) -> torch.Tensor:
    """Per-request causal SDPA reference. Returns [total_q, num_q_heads, head_size].

    Takes the same inputs as impl.forward (minus layer/output). Context
    positions are zero-initialized (cache starts at zero, context is never
    written), so the reference constructs context K/V as zeros directly —
    no cache reading required, which avoids backend-specific cache layouts.
    New token K/V come from k_new/v_new (same values written to cache by
    impl.do_kv_cache_update before impl.forward is called).
    """
    device = q.device
    scale = 1.0 / math.sqrt(q.shape[-1])
    num_kv_heads = k_new.shape[1]
    head_size = k_new.shape[2]
    seq_lens, query_start_loc = _extract_seq_info(attn_metadata, device)
    batch_size = int(seq_lens.shape[0])
    head_rep = q.shape[1] // num_kv_heads
    out_parts: list[torch.Tensor] = []
    for i in range(batch_size):
        q_start = int(query_start_loc[i].item())
        q_end = int(query_start_loc[i + 1].item())
        kv_len = int(seq_lens[i].item())
        ctx_len = kv_len - (q_end - q_start)
        qi = q[q_start:q_end]
        ki_new = k_new[q_start:q_end]
        vi_new = v_new[q_start:q_end]
        if ctx_len > 0:
            zeros_k = torch.zeros(
                ctx_len, num_kv_heads, head_size, dtype=k_new.dtype, device=device
            )
            zeros_v = torch.zeros(
                ctx_len, num_kv_heads, head_size, dtype=v_new.dtype, device=device
            )
            ki = torch.cat([zeros_k, ki_new], dim=0)
            vi = torch.cat([zeros_v, vi_new], dim=0)
        else:
            ki, vi = ki_new, vi_new
        if head_rep > 1:
            ki = ki.repeat_interleave(head_rep, dim=1)
            vi = vi.repeat_interleave(head_rep, dim=1)
        out_parts.append(sdpa_for_request(qi, ki, vi, ctx_len, scale, device))
    return torch.cat(out_parts, dim=0)
