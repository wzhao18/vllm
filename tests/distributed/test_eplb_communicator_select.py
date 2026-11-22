# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for the automatic EPLB communicator selection."""

import pytest

from vllm.config.parallel import EPLBConfig, ParallelConfig
from vllm.distributed.eplb.eplb_communicator import select_eplb_communicator


@pytest.mark.parametrize("nixl_available", [True, False])
@pytest.mark.parametrize("use_async", [True, False])
def test_nvlink_domain_selects_symm_mem(nixl_available, use_async):
    # One NVLink domain (incl. MNNVL) always pushes over symm_mem, even when
    # NIXL is installed.
    assert (
        select_eplb_communicator(
            in_nvlink_domain=True,
            nixl_available=nixl_available,
            use_async=use_async,
        )
        == "symm_mem"
    )


@pytest.mark.parametrize("use_async", [True, False])
def test_multi_domain_prefers_nixl(use_async):
    assert (
        select_eplb_communicator(
            in_nvlink_domain=False, nixl_available=True, use_async=use_async
        )
        == "nixl"
    )


def test_multi_domain_sync_without_nixl_uses_gloo():
    assert (
        select_eplb_communicator(
            in_nvlink_domain=False, nixl_available=False, use_async=False
        )
        == "torch_gloo"
    )


def test_no_working_communicator_raises():
    # Async gloo never lands under load: raise instead of picking it.
    with pytest.raises(ValueError, match="No working EPLB communicator"):
        select_eplb_communicator(
            in_nvlink_domain=False, nixl_available=False, use_async=True
        )


@pytest.mark.parametrize("nixl_available", [True, False])
@pytest.mark.parametrize("use_async", [True, False])
def test_vr_multi_domain_raises_instead_of_nixl(nixl_available, use_async):
    with pytest.raises(ValueError, match="NIXL EPLB transfers crash"):
        select_eplb_communicator(
            in_nvlink_domain=False,
            nixl_available=nixl_available,
            use_async=use_async,
            nixl_eplb_broken=True,
        )


def test_vr_single_domain_uses_symm_mem():
    assert (
        select_eplb_communicator(
            in_nvlink_domain=True,
            nixl_available=True,
            use_async=True,
            nixl_eplb_broken=True,
        )
        == "symm_mem"
    )


def _config(monkeypatch, *, cuda, xpu=False, elastic=False, nixl=True, comm=None):
    import vllm.config.parallel as parallel_mod
    import vllm.distributed.nixl_utils as nixl_utils

    monkeypatch.setattr(parallel_mod.current_platform, "is_cuda", lambda: cuda)
    monkeypatch.setattr(parallel_mod.current_platform, "is_xpu", lambda: xpu)
    monkeypatch.setattr(nixl_utils, "is_nixl_available", lambda: nixl)
    return ParallelConfig(
        data_parallel_size=2,
        enable_expert_parallel=True,
        enable_eplb=True,
        enable_elastic_ep=elastic,
        eplb_config=EPLBConfig(communicator=comm),
    )


def test_config_leaves_cuda_static_ep_unresolved(monkeypatch):
    # Resolved by EplbState once the EP group exists (needs the NVLink probe),
    # so ParallelConfig must not pin nixl on CUDA any more.
    assert _config(monkeypatch, cuda=True).eplb_config.communicator is None


def test_config_explicit_communicator_kept(monkeypatch):
    cfg = _config(monkeypatch, cuda=True, comm="nixl")
    assert cfg.eplb_config.communicator == "nixl"


def test_config_xpu_and_non_cuda(monkeypatch):
    assert _config(monkeypatch, cuda=False, xpu=True).eplb_config.communicator == (
        "torch_xccl"
    )
    assert _config(monkeypatch, cuda=False).eplb_config.communicator == "nixl"
    assert (
        _config(monkeypatch, cuda=False, nixl=False).eplb_config.communicator
        == "torch_gloo"
    )
