# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared configuration and setup for EC and KV Mooncake Store clients."""

import json
import os
from dataclasses import dataclass
from typing import Any, Literal

import regex as re

from vllm.logger import init_logger

logger = init_logger(__name__)

DEFAULT_GLOBAL_SEGMENT_SIZE = 4 * 1024 * 1024 * 1024  # 4 GiB
DEFAULT_LOCAL_BUFFER_SIZE = 4 * 1024 * 1024 * 1024  # 4 GiB
DEFAULT_TENANT_ID = "default"


MooncakeMode = Literal["embedded", "standalone-store"]


@dataclass
class MooncakeStoreConfig:
    """Configuration for MooncakeDistributedStore.

    ``mode`` selects the topology: ``embedded`` (each rank contributes
    ``global_segment_size`` in-process) or ``standalone-store`` (rank
    contributes 0; an external ``mooncake_client`` process owns the pool
    and the SSD tier).
    """

    metadata_server: str
    master_server_address: str
    protocol: str
    device_name: str
    mode: MooncakeMode = "embedded"
    global_segment_size: int = DEFAULT_GLOBAL_SEGMENT_SIZE
    local_buffer_size: int = DEFAULT_LOCAL_BUFFER_SIZE
    enable_offload: bool = False
    tenant_id: str = DEFAULT_TENANT_ID

    def __post_init__(self) -> None:
        if self.mode not in ("embedded", "standalone-store"):
            raise ValueError(f"unknown Mooncake mode: {self.mode!r}")
        if self.local_buffer_size <= 0:
            raise ValueError("local_buffer_size must be > 0")
        if self.mode == "embedded" and self.global_segment_size == 0:
            raise ValueError("embedded mode requires global_segment_size > 0")
        if self.mode == "standalone-store" and self.global_segment_size != 0:
            raise ValueError("standalone-store mode requires global_segment_size == 0")

    @staticmethod
    def from_file(file_path: str) -> "MooncakeStoreConfig":
        with open(file_path) as file:
            config = json.load(file)
        return MooncakeStoreConfig(
            metadata_server=config.get("metadata_server", ""),
            master_server_address=config.get("master_server_address", ""),
            protocol=config.get("protocol", "rdma"),
            device_name=config.get("device_name", ""),
            mode=config.get("mode", "embedded"),
            global_segment_size=_parse_size(
                config.get("global_segment_size", DEFAULT_GLOBAL_SEGMENT_SIZE)
            ),
            local_buffer_size=_parse_size(
                config.get("local_buffer_size", DEFAULT_LOCAL_BUFFER_SIZE)
            ),
            enable_offload=bool(config.get("enable_offload", False)),
            tenant_id=_normalize_tenant_id(config.get("tenant_id", DEFAULT_TENANT_ID)),
        )

    @staticmethod
    def load_from_config() -> "MooncakeStoreConfig":
        config_path = os.getenv("MOONCAKE_CONFIG_PATH")
        if not config_path:
            raise ValueError(
                "The environment variable 'MOONCAKE_CONFIG_PATH' is not set."
            )
        return MooncakeStoreConfig.from_file(config_path)


def _normalize_tenant_id(value: Any) -> str:
    if value is None:
        return DEFAULT_TENANT_ID
    if not isinstance(value, str):
        raise TypeError(
            f"tenant_id must be a string or null, got {type(value).__name__}: {value!r}"
        )
    tenant_id = value.strip()
    return tenant_id if tenant_id else DEFAULT_TENANT_ID


def _parse_size(value: Any) -> int:
    """Parse storage size strings with units: GB, MB, KB, B."""
    if isinstance(value, int):
        return value
    if not isinstance(value, str):
        try:
            return int(value)
        except (TypeError, ValueError) as e:
            raise TypeError(f"Unsupported type for size: {type(value)}") from e

    cleaned = value.strip().lower()
    if not cleaned:
        raise ValueError("Size cannot be empty.")

    unit_multipliers = {
        "gb": 1024**3,
        "mb": 1024**2,
        "kb": 1024,
        "b": 1,
    }
    match = re.match(r"^\s*([\d.]+)\s*(gb|mb|kb|b)?\s*$", cleaned)
    if not match:
        raise ValueError(f"Invalid format: '{value}'")

    number_str = match.group(1)
    unit = match.group(2) or "b"
    multiplier = unit_multipliers[unit]

    try:
        numeric_value = float(number_str)
    except ValueError as exc:
        raise ValueError(f"Invalid numeric value '{number_str}' in: '{value}'") from exc
    return int(numeric_value * multiplier)


NUMA_DEVICE_NAME_PREFIX = "numa:"


def parse_numa_device_names(device_name: str) -> dict[int, str] | None:
    """Parse ``numa:<node>=<devices>;<node>=<devices>`` into a node table.

    Returns None when ``device_name`` is a plain device list.
    """
    spec = device_name.strip()
    if not spec.startswith(NUMA_DEVICE_NAME_PREFIX):
        return None
    table: dict[int, str] = {}
    for entry in spec[len(NUMA_DEVICE_NAME_PREFIX) :].split(";"):
        entry = entry.strip()
        if not entry:
            continue
        node, sep, devices = entry.partition("=")
        if not sep or not node.strip().isdigit() or not devices.strip():
            raise ValueError(
                f"Invalid NUMA device_name entry {entry!r}; expected "
                "'numa:<node>=<dev>[,<dev>...];<node>=...'"
            )
        table[int(node)] = devices.strip()
    if not table:
        raise ValueError(f"Empty NUMA device_name {device_name!r}")
    return table


def resolve_device_name(device_name: str, numa_node: int | None = None) -> str:
    """Return the RDMA device list for this process.

    A ``numa:`` table selects the list for the NUMA node of the current CUDA
    device. On hosts with several NUMA nodes and NICs per node (e.g. Vera
    Rubin), listing every NIC lets the transfer engine pair a GPU with NICs
    on the other socket, which can cut GPU <-> host-segment bandwidth several
    times; the per-node list keeps each rank on its own socket's NICs.
    """
    table = parse_numa_device_names(device_name)
    if table is None:
        return device_name
    if numa_node is None:
        import torch

        from vllm.platforms import current_platform

        nodes = current_platform.get_all_device_numa_nodes()
        if not nodes:
            raise RuntimeError(
                "Mooncake NUMA device_name needs the GPU NUMA node, which "
                "could not be detected"
            )
        numa_node = nodes[torch.accelerator.current_device_index()]
    if numa_node not in table:
        raise ValueError(
            f"Mooncake NUMA device_name has no entry for NUMA node {numa_node}: "
            f"{device_name!r}"
        )
    logger.info(
        "Mooncake RDMA devices for NUMA node %d: %s", numa_node, table[numa_node]
    )
    return table[numa_node]


def setup_mooncake_store(
    store: Any, config: MooncakeStoreConfig, local_hostname: str
) -> None:
    """Initialize a native Store, preserving explicit tenant configuration."""
    setup_kwargs: dict[str, str] = {}
    if config.tenant_id != DEFAULT_TENANT_ID:
        setup_kwargs["tenant_id"] = config.tenant_id
    ret = store.setup(
        local_hostname,
        config.metadata_server,
        config.global_segment_size,
        config.local_buffer_size,
        config.protocol,
        resolve_device_name(config.device_name),
        config.master_server_address,
        **setup_kwargs,
    )
    if ret != 0:
        raise RuntimeError("Initialize MooncakeDistributedStore failed.")
