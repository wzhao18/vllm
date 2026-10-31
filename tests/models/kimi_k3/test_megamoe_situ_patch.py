# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 MegaMoE header patch (vllm/models/kimi_k3/nvidia/megamoe).

CPU: the patch applies to the installed DeepGEMM header and the overlay mirrors
the include tree. GPU (>= 2 SM100-family GPUs, DeepGEMM): the patched kernel is
bitwise identical to the stock kernel and matches a PyTorch reference of the
K3.1 routed experts (MXFP8 x MXFP4, SiTU alpha 4 / beta 25, FP8 L1 output) on
EP-sharded problems with random top-16 routing.
"""

import importlib
import os
import socket
from pathlib import Path

import pytest
import torch

from vllm.models.kimi_k3.nvidia.megamoe.overlay import overlay_root
from vllm.models.kimi_k3.nvidia.megamoe.situ_patch import (
    FLAGS,
    MARKER,
    STOCK_HEADER,
    PatchError,
    patch_situ_header,
)


def _deep_gemm():
    for name in ("deep_gemm", "vllm.third_party.deep_gemm"):
        try:
            return importlib.import_module(name)
        except Exception:  # noqa: BLE001
            continue
    return None


def _stock_header() -> Path | None:
    dg = _deep_gemm()
    if dg is None:
        return None
    p = Path(dg.__file__).resolve().parent / "include" / STOCK_HEADER
    return p if p.is_file() else None


def test_patch_applies_to_installed_header():
    stock = _stock_header()
    if stock is None:
        pytest.skip("DeepGEMM with the SiTU MegaMoE header is not installed")
    text = stock.read_text()
    out = patch_situ_header(text)
    assert MARKER in out
    for flag in FLAGS:
        assert f"#define {flag} 1" in out
    # Every stock path is kept under #else
    assert out.count("ptx::atomic_add_sys(") == text.count("ptx::atomic_add_sys(")
    assert "kBeforeDispatchPullBarrierTag" in out
    assert "// Iterate over all token chunks" in out
    off = patch_situ_header(text, {f: 0 for f in FLAGS})
    assert "#define K31_BAR_ALL_CTAS 0" in off
    with pytest.raises(PatchError):
        patch_situ_header(out)
    with pytest.raises(PatchError):
        patch_situ_header(text.replace("// Count experts' tokens", "// counted"))


def test_overlay_mirrors_and_replaces(tmp_path):
    stock = tmp_path / "stock"
    (stock / "include/deep_gemm/impls").mkdir(parents=True)
    (stock / "include/deep_gemm/common").mkdir(parents=True)
    (stock / "include/deep_gemm/impls/a.cuh").write_text("stock a")
    (stock / "include/deep_gemm/impls/b.cuh").write_text("stock b")
    (stock / "include/deep_gemm/common/c.cuh").write_text("stock c")
    new = tmp_path / "a_new.cuh"
    new.write_text("patched a")
    root = overlay_root(stock, tmp_path / "cache", {"deep_gemm/impls/a.cuh": new})
    inc = root / "include/deep_gemm"
    assert (inc / "impls/a.cuh").read_text() == "patched a"
    assert (inc / "impls/b.cuh").read_text() == "stock b"
    assert (inc / "common/c.cuh").read_text() == "stock c"
    assert (inc / "common").is_symlink()  # untouched directories are linked whole
    assert overlay_root(stock, tmp_path / "cache", {"deep_gemm/impls/a.cuh": new}) == root
    with pytest.raises(FileNotFoundError):
        overlay_root(stock, tmp_path / "cache2", {"deep_gemm/impls/x.cuh": new})


# ---------------------------------------------------------------------------- GPU
HIDDEN, INTER, TOPK = 3584, 3072, 16
ALPHA, BETA = 4.0, 25.0
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _gpu_ok() -> bool:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        return False
    if torch.cuda.get_device_capability()[0] != 10:
        return False
    return _stock_header() is not None


def _dequant_fp4(packed: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    """[n, k/2] int8 (low nibble first) x [n, k/32] float scales -> [n, k] float."""
    lut = torch.tensor(E2M1, device=packed.device)
    p = packed.view(torch.uint8).to(torch.int32)
    codes = torch.stack((p & 0xF, p >> 4), dim=-1).flatten(-2)
    vals = lut[codes & 7] * (1 - 2 * (codes >> 3)).float()
    return vals * sf.repeat_interleave(32, dim=-1)


def _quant_fp8_ue8m0(x: torch.Tensor) -> torch.Tensor:
    """Per-32 MXFP8 (E4M3, power-of-two scale) quantize-dequantize."""
    v = x.float().view(x.shape[0], -1, 32)
    amax = v.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    sf = torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0)))
    return ((v / sf).to(torch.float8_e4m3fn).float() * sf).view_as(x)


def _reference(xq, idx, tw, w1, w2):
    """y[t] = sum_j tw[t, j] * expert_{idx[t, j]}(x[t]) with the kernel's roundings."""
    y = torch.zeros(xq.shape[0], HIDDEN, device=xq.device)
    for t in range(xq.shape[0]):
        for j in range(TOPK):
            e = int(idx[t, j])
            h = w1[e] @ xq[t]
            gate = h[:INTER].bfloat16().float()
            up = h[INTER:].bfloat16().float()
            g = ALPHA * torch.tanh(gate / ALPHA) * torch.sigmoid(gate)
            act = g * BETA * torch.tanh(up / BETA) * tw[t, j]
            a = _quant_fp8_ue8m0(act[None])[0]
            y[t] += (w2[e] @ a).bfloat16().float()
    return y


def _worker(rank: int, nranks: int, port: int, cache: str, out: dict):
    import torch.distributed as dist

    from vllm.models.kimi_k3.nvidia.megamoe import install_patched_header

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=nranks,
        device_id=torch.device(f"cuda:{rank}"),
    )
    group = dist.new_group(list(range(nranks)))
    dg = _deep_gemm()
    utils = importlib.import_module(dg.__name__ + ".utils")
    epr = 4
    num_experts = epr * nranks
    g = torch.Generator(device="cuda").manual_seed(1234)  # same weights on every rank

    def cast(n, k):
        bf = 0.05 * torch.randn((num_experts, n, k), device="cuda", generator=g)
        w = torch.empty((num_experts, n, k // 2), device="cuda", dtype=torch.int8)
        sf = torch.empty((num_experts, n, k // 32), device="cuda", dtype=torch.float)
        for e in range(num_experts):
            w[e], sf[e] = utils.per_token_cast_to_fp4(bf[e].bfloat16(), use_ue8m0=True, gran_k=32)
        deq = torch.stack([_dequant_fp4(w[e], sf[e]) for e in range(num_experts)])
        lo = rank * epr
        local = (w[lo:lo + epr].contiguous(),
                 dg.transform_sf_into_required_layout(sf[lo:lo + epr].contiguous(), n, k, (1, 32), epr))
        return local, deq

    l1, w1_ref = cast(2 * INTER, HIDDEN)
    l2, w2_ref = cast(HIDDEN, INTER)
    l1, l2 = dg.transform_weights_for_mega_moe(l1, l2, activation="situ")

    cases = (1, 7, 64)
    buf = dg.get_symm_buffer_for_mega_moe(group, num_experts, max(cases), TOPK, HIDDEN, INTER,
                                          mma_type="fp8xfp4", activation="situ")
    data = {}
    for t in cases:
        gr = torch.Generator(device="cuda").manual_seed(100 * t + rank)
        x = torch.randn((t, HIDDEN), device="cuda", generator=gr).bfloat16()
        idx = torch.randn((t, num_experts), device="cuda", generator=gr).topk(TOPK, -1).indices
        tw = torch.rand((t, TOPK), device="cuda", generator=gr) + 0.1
        tw = tw / tw.sum(-1, keepdim=True)
        data[t] = (x, idx, tw)

    def run(t):
        x, idx, tw = data[t]
        xf, xs = utils.per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
        buf.x[:t].copy_(xf)
        buf.x_sf[:t].copy_(xs)
        buf.topk_idx[:t].copy_(idx)
        buf.topk_weights[:t].copy_(tw)
        y = torch.empty((t, HIDDEN), device="cuda", dtype=torch.bfloat16)
        dg.fp8_fp4_mega_moe(y, l1, l2, buf, activation="situ", activation_clamp=None,
                            fast_math=True, activation_alpha=ALPHA, activation_beta=BETA)
        torch.cuda.synchronize()
        dist.barrier(group)
        return y

    stock = {t: run(t) for t in cases}
    assert install_patched_header(dg, Path(cache)) is not None
    for _ in range(3):  # repeated launches exercise the barrier phase / counters across launches
        patched = {t: run(t) for t in cases}
        for t in cases:
            assert torch.equal(patched[t], stock[t]), (rank, t)
    for t in cases:
        x, idx, tw = data[t]
        xq = utils.per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=32)
        xq = xq[0].float() * xq[1].repeat_interleave(32, dim=-1)
        ref = _reference(xq, idx, tw, w1_ref, w2_ref)
        err = (patched[t].float() - ref).norm() / ref.norm()
        assert err < 2e-2, (rank, t, float(err))
        out[(rank, t)] = float(err)
    buf.destroy()
    dist.destroy_process_group()


@pytest.mark.skipif(not _gpu_ok(), reason="needs >= 2 SM100-family GPUs and DeepGEMM")
def test_patched_kernel_bitwise_and_reference(tmp_path):
    nranks = min(4, torch.cuda.device_count())
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    os.environ.setdefault("VLLM_CACHE_ROOT", str(tmp_path / "vllm_cache"))
    mgr = torch.multiprocessing.Manager()
    out = mgr.dict()
    torch.multiprocessing.spawn(_worker, args=(nranks, port, str(tmp_path / "cache"), out),
                                nprocs=nranks)
    assert len(out) == nranks * 3
