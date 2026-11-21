# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The async worker that transfers experts in the background."""

import threading
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.distributed.parallel_state import get_eplb_group
from vllm.logger import init_logger

from .eplb_utils import CpuGpuEvent, device_stream
from .migration_scheduler import schedule_migration_batches_for_layers
from .rebalance_execute import (
    AsyncEplbLayerResult,
    buffer_row_maps,
    buffer_rows_needed,
    transfer_layer,
)

if TYPE_CHECKING:
    from .eplb_state import EplbModelState, EplbState

logger = init_logger(__name__)


def plan_layer_groups(
    old_map: np.ndarray, new_map: np.ndarray, num_local_experts: int
) -> list[tuple[list[int], list[list[np.ndarray]]]]:
    """Greedily group consecutive layers whose staged rows fit in one buffer of
    `num_local_experts` rows on every rank. Returns, per group, the layers and
    each layer's per-rank buffer row maps. Deterministic, so every rank plans
    the same groups."""
    groups: list[tuple[list[int], list[list[np.ndarray]]]] = []
    layers: list[int] = []
    maps: list[list[np.ndarray]] = []
    used = None
    for layer in range(old_map.shape[0]):
        need = buffer_rows_needed(old_map[layer], new_map[layer], num_local_experts)
        counts = np.array([n.shape[0] for n in need])
        if used is None or (used + counts > num_local_experts).any():
            if layers:
                groups.append((layers, maps))
            layers, maps, used = [], [], np.zeros_like(counts)
        maps.append(buffer_row_maps(need, used.tolist(), num_local_experts))
        layers.append(layer)
        used = used + counts
    if layers:
        groups.append((layers, maps))
    return groups


def start_async_worker(
    state: "EplbState",
    is_profile: bool = False,
) -> threading.Thread:
    rank = get_eplb_group().device_group.rank()
    device_index = state.device_index
    assert state.is_async

    def thread_target() -> None:
        assert device_index is not None
        torch.accelerator.set_device_index(device_index)
        stream = torch.Stream(device=torch.accelerator.current_accelerator())
        try:
            transfer_run_periodically(
                state=state,
                stream=stream,
                is_profile=is_profile,
            )
        except Exception as exc:  # pragma: no cover - diagnostic path
            logger.exception("async loop error (Rank %d): %s", rank, str(exc))

    thread = threading.Thread(target=thread_target, daemon=True)
    thread.start()
    return thread


def run_rebalance_experts(
    model_state: "EplbModelState",
    eplb_state: "EplbState",
    physical_to_logical_map_cpu: torch.Tensor,
    stream: torch.Stream,
) -> torch.Tensor:
    assert model_state.eplb_stats is not None
    eplb_stats = model_state.eplb_stats

    # Move the global expert load window to CPU for computation.
    with device_stream(stream):
        global_expert_load_window = eplb_stats.global_expert_load_window.cpu()
    # Compute new expert mappings for the model
    new_physical_to_logical_map = eplb_state.policy.rebalance_experts(
        global_expert_load_window,
        eplb_stats.num_replicas,
        eplb_stats.num_groups,
        eplb_stats.num_nodes,
        eplb_stats.num_gpus,
        physical_to_logical_map_cpu,
        move_budget=eplb_state.parallel_config.eplb_config.resolved_move_budget(
            eplb_stats.num_gpus
        ),
    )
    assert new_physical_to_logical_map.device == torch.device("cpu")

    return new_physical_to_logical_map


def transfer_run_periodically(
    state: "EplbState",
    stream: torch.Stream,
    is_profile: bool = False,
) -> None:
    while True:
        state.rearrange_event.wait(stream=stream)

        eplb_group = get_eplb_group().device_group
        eplb_cpu_group = get_eplb_group().cpu_group
        ep_rank = eplb_group.rank()

        assert state.is_async
        for model_state in state.model_states.values():
            layer_idx = 0
            # Set the async worker's CUDA stream on the communicator
            model_state.communicator.set_stream(stream)
            num_layers = model_state.model.num_moe_layers

            # Snapshot the physical_to_logical_map (synchronized with
            # rearrange_event) and copy it to CPU
            with device_stream(stream):
                physical_to_logical_map_cpu = model_state.physical_to_logical_map.cpu()

            new_physical_to_logical_map = run_rebalance_experts(
                model_state, state, physical_to_logical_map_cpu, stream
            )

            migration_batches_by_layer = None
            if state.parallel_config.eplb_config.migration_batching_enabled:
                profile_name = "eplb: schedule migration batches"
                num_local_experts = model_state.model.expert_weights[0][0].shape[0]
                # The placement snapshots stay fixed for this rebalance cycle.
                # Precompute all layer schedules once, but execute transfers one
                # layer at a time below to preserve the migration lifecycle. This
                # profiling range therefore covers scheduling for every MoE layer.
                with torch.profiler.record_function(profile_name):
                    migration_batches_by_layer = schedule_migration_batches_for_layers(
                        num_local_experts,
                        physical_to_logical_map_cpu.numpy(),
                        new_physical_to_logical_map.numpy(),
                    )

            if (
                state.parallel_config.eplb_config.layer_group_transfers
                and not is_profile
                and migration_batches_by_layer is None
                and model_state.communicator.needs_recv_rows
            ):
                _transfer_layer_groups(
                    model_state,
                    physical_to_logical_map_cpu,
                    new_physical_to_logical_map,
                    eplb_group,
                    eplb_cpu_group,
                    stream,
                )
                continue

            # Execute one EPLB layer transfer per model forward pass. Each iteration
            # of this loop will copy the new set of expert weights into
            # model_state.expert_buffer, which will be consumed by the main thread in
            # move_to_workspace.
            # We sync the rebalanced flag across ranks before each iteration so
            # all ranks make a coordinated decision to continue or stop.
            while layer_idx < num_layers:
                flag = torch.tensor(
                    [int(model_state.rebalanced)],
                    dtype=torch.int32,
                    device="cpu",
                )
                torch.distributed.all_reduce(flag, group=eplb_cpu_group)
                if int(flag.item()) != eplb_cpu_group.size():
                    logger.warning(
                        "async worker (rank=%d): layer %d coordinated stop "
                        "(flag_sum=%d, group_size=%d)",
                        ep_rank,
                        layer_idx,
                        int(flag.item()),
                        eplb_cpu_group.size(),
                    )
                    model_state.rebalanced = False
                    break

                transfer_metadata = transfer_layer(
                    old_layer_indices=physical_to_logical_map_cpu[layer_idx],
                    new_layer_indices=new_physical_to_logical_map[layer_idx],
                    expert_weights=model_state.model.expert_weights[layer_idx],
                    expert_weights_buffer=model_state.expert_buffer,
                    communicator=model_state.communicator,
                    ep_group=eplb_group,
                    is_profile=is_profile,
                    stream=stream,
                    layer_idx=layer_idx,
                    enable_migration_batching=(
                        state.parallel_config.eplb_config.migration_batching_enabled
                    ),
                    migration_batches=(
                        # This is only the current layer's schedule; it does not
                        # combine transfers from different layers.
                        migration_batches_by_layer[layer_idx]
                        if migration_batches_by_layer is not None
                        else None
                    ),
                )

                # Wait until all writes to expert_buffer have finished before making the
                # AsyncEplbLayerResult visible to the main thread.
                stream.synchronize()

                # This event guarantees that expert_buffer will not be overwritten by
                # subsequent iterations of this loop until the main thread has consumed
                # it. Record is called by the main thread after move_from_buffer().
                consumed_event = CpuGpuEvent()

                model_state.pending_result = AsyncEplbLayerResult(
                    layer_idx=layer_idx,
                    new_physical_to_logical_map=new_physical_to_logical_map[layer_idx],
                    transfer_metadata=transfer_metadata,
                    consumed_event=consumed_event,
                )

                # Block this thread until the main thread and main stream
                # finish copying model_state.expert_buffer into
                # model_state.model.expert_weights[layer_idx]
                consumed_event.wait(stream=stream)
                assert model_state.pending_result is None
                layer_idx += 1


def _transfer_layer_groups(
    model_state: "EplbModelState",
    old_map: torch.Tensor,
    new_map: torch.Tensor,
    eplb_group,
    eplb_cpu_group,
    stream: torch.Stream,
) -> None:
    """Transfer layer groups that fit in the receive buffer together: one
    communicator execute (barrier pair) per group, and the main thread applies
    the whole group in one step."""
    num_local_experts = model_state.model.expert_weights[0][0].shape[0]
    groups = plan_layer_groups(old_map.numpy(), new_map.numpy(), num_local_experts)
    for layers, maps in groups:
        flag = torch.tensor([int(model_state.rebalanced)], dtype=torch.int32)
        torch.distributed.all_reduce(flag, group=eplb_cpu_group)
        if int(flag.item()) != eplb_cpu_group.size():
            logger.warning(
                "async worker (rank=%d): layer %d coordinated stop",
                eplb_group.rank(),
                layers[0],
            )
            model_state.rebalanced = False
            return
        metas = [
            transfer_layer(
                old_layer_indices=old_map[layer],
                new_layer_indices=new_map[layer],
                expert_weights=model_state.model.expert_weights[layer],
                expert_weights_buffer=model_state.expert_buffer,
                communicator=model_state.communicator,
                ep_group=eplb_group,
                stream=stream,
                layer_idx=layer,
                buffer_maps=layer_maps,
                defer_execute=True,
            )
            for layer, layer_maps in zip(layers, maps)
        ]
        model_state.communicator.execute()
        stream.synchronize()
        consumed_event = CpuGpuEvent()
        model_state.pending_result = [
            AsyncEplbLayerResult(
                layer_idx=layer,
                new_physical_to_logical_map=new_map[layer],
                transfer_metadata=meta,
                consumed_event=consumed_event,
            )
            for layer, meta in zip(layers, metas)
        ]
        consumed_event.wait(stream=stream)
        assert model_state.pending_result is None
