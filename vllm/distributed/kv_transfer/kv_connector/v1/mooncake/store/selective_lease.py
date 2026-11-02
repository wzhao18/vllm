# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Selective read leases for Mooncake store prefix lookups.

``batch_is_exist`` leases every key it finds, so a lookup renews every stored
key of the prompt, including keys a load of the hit skips, such as blocks
outside a sliding window or states at earlier block boundaries. With
``VLLM_MOONCAKE_STORE_SELECTIVE_LEASE`` the lookup probes lease-free
(``batch_probe_key``) and leases only the keys the hit will load.
"""

import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.bounded import (
    StoreCallGuard,
    StoreCallTimeout,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.coordinator import (  # noqa: E501
    MooncakeStoreCoordinator,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (  # noqa: E501
    ChunkedTokenDatabase,
    MooncakeLookupResult,
    PoolKey,
)
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_cache_interface import KVCacheGroupSpec

logger = init_logger(__name__)

# Lease rounds before giving up on a hit whose keys keep being evicted between
# probe and lease. Only a hit whose lease found every key is returned.
_LEASE_ROUNDS = 2
# Bound on one lookup's probe and leases together, below the core scheduler's
# 10 s pending-lookup fallback and LookupKeyClient's 10 s lookup_timeout_s.
DEFAULT_LOOKUP_STORE_TIMEOUT_S = 5.0
# Key probed at startup to check that the master serves batch_probe_key.
_SENTINEL_KEY = "vllm-selective-lease-probe"


class _LeaseFailedError(Exception):
    """The lease RPC raised or returned per-key error results."""


class _RecordOperation(Protocol):
    def __call__(
        self,
        operation: str,
        duration_seconds: float,
        num_keys: int,
        *,
        num_bytes: int = 0,
        status: str = "ok",
        num_failed_keys: int = 0,
    ) -> None: ...


def lookup_store_timeout(extra_config: Mapping[str, Any]) -> float:
    """``lookup_store_timeout_s`` of the connector's extra config.

    Raises unless it ends a lookup's store calls before the lookup is a miss
    anyway: past the core scheduler's pending-lookup fallback or
    LookupKeyClient's ``lookup_timeout_s``.
    """
    from vllm.v1.core.sched.scheduler import KV_CONNECTOR_LOOKUP_TIMEOUT_S

    timeout_s = float(
        extra_config.get("lookup_store_timeout_s", DEFAULT_LOOKUP_STORE_TIMEOUT_S)
    )
    limit_s = min(
        KV_CONNECTOR_LOOKUP_TIMEOUT_S,
        float(extra_config.get("lookup_timeout_s", 10.0)),
    )
    if not 0 < timeout_s < limit_s:
        raise ValueError(
            f"lookup_store_timeout_s={timeout_s} must be above 0 and below "
            f"{limit_s} s, the core's pending-lookup fallback "
            "(VLLM_KV_CONNECTOR_LOOKUP_TIMEOUT_S) and lookup_timeout_s"
        )
    return timeout_s


class SelectiveLookupLease:
    """Leases only the keys a prefix-lookup hit will load.

    The lookup probes its candidates with :meth:`probe`, computes the hit as
    usual, and passes it through :meth:`lease_hit`.
    """

    def __init__(
        self,
        store: Any,
        coord: MooncakeStoreCoordinator,
        token_dbs: Sequence[ChunkedTokenDatabase],
        kv_cache_groups: Sequence[KVCacheGroupSpec],
        lookup_key_prefixes: Sequence[Sequence[str]],
        record_operation: _RecordOperation,
        timeout_s: float | None = None,
    ):
        self._check_supported(store)
        self.store = store
        self.coord = coord
        self.token_dbs = token_dbs
        self.kv_cache_groups = kv_cache_groups
        self.lookup_key_prefixes = lookup_key_prefixes
        self.record_operation = record_operation
        # A lookup's probe and leases share timeout_s, from the probe on: a
        # lookup whose calls do not return within it is a miss, and while a call
        # that timed out is still out, later ones fail fast instead of queueing
        # behind it on the lookup server's single thread.
        self._guard = (
            None if timeout_s is None else StoreCallGuard("MooncakeLookup", timeout_s)
        )
        self._deadline = 0.0

    @staticmethod
    def _check_supported(store: Any) -> None:
        if not callable(getattr(store, "batch_probe_key", None)):
            raise RuntimeError(
                "VLLM_MOONCAKE_STORE_SELECTIVE_LEASE requires "
                "MooncakeDistributedStore.batch_probe_key, which the installed "
                "Mooncake version does not provide."
            )

    def check_master(self) -> None:
        """Raise unless the master answers a lease-free probe.

        A master older than the client answers each key with an error code
        instead of raising, which would turn every lookup into a silent miss.
        """
        try:
            res = self.probe([_SENTINEL_KEY])
        except Exception as e:
            raise RuntimeError(
                "VLLM_MOONCAKE_STORE_SELECTIVE_LEASE: the Mooncake master did not "
                f"answer batch_probe_key: {e}"
            ) from e
        if len(res) != 1 or res[0] not in (0, 1):
            raise RuntimeError(
                "VLLM_MOONCAKE_STORE_SELECTIVE_LEASE: the Mooncake master rejected "
                f"batch_probe_key (result {list(res)}); the master and every "
                "client must run Mooncake 0.3.14 or newer"
            )

    def probe(self, keys: list[str]) -> list[int]:
        """Existence of ``keys``, without granting a lease.

        Starts the lookup's store-call budget; its leases get what is left.
        """
        if self._guard is not None:
            self._deadline = time.monotonic() + self._guard.timeout_s
        return self._call("lookup_probe", self.store.batch_probe_key, keys)

    def lease_hit(
        self,
        block_hashes: Sequence[BlockHash],
        hit: MooncakeLookupResult,
        exists_set: set[tuple[int, bytes]],
        longest_hit: Callable[[set[tuple[int, bytes]]], MooncakeLookupResult],
        mask_num: int = 0,
    ) -> MooncakeLookupResult:
        """Lease the keys a load of ``hit`` reads and return the hit to load.

        ``exists_set`` holds the probed ``(group, hash)`` pairs and
        ``longest_hit`` recomputes the hit from them. The lease covers only the
        keys this hit loads, e.g. the state at its tail boundary but not states
        at earlier boundaries, nor the first ``mask_num`` tokens, which the load
        skips (the local hit the lookup trusted). If a key was evicted since the
        probe, the shorter hit can end at one of those unleased keys, so it is
        leased in another round. Returns an empty hit if the lease fails or the
        hit keeps shrinking.
        """
        try:
            for _ in range(_LEASE_ROUNDS):
                if hit.hit_length <= 0:
                    return hit
                evicted = self._lease(block_hashes, hit, mask_num)
                if not evicted:
                    return hit
                exists_set = exists_set - evicted
                hit = longest_hit(exists_set)
        except _LeaseFailedError as e:
            logger.error("Failed to lease a Mooncake lookup hit: %s", e)
        except Exception:
            # The lookup server has no handler of its own; report a miss.
            logger.exception("Failed to lease a Mooncake lookup hit")
        # The lease failed, or keys kept being evicted until rounds ran out.
        return MooncakeLookupResult(0)

    def loaded_keys(
        self,
        block_hashes: Sequence[BlockHash],
        hit: MooncakeLookupResult,
        mask_num: int = 0,
    ) -> tuple[list[str], list[tuple[int, bytes]]]:
        """Keys a load of ``hit`` reads, in every rank namespace, and their
        ``(group, hash)`` pairs.

        Mirrors ``KVCacheStoreRecvingThread._handle_load``: chunks from
        ``mask_num``, each group's load mask, with the tail chunk keyed by the
        hit's tail boundary.
        """
        hit_length = hit.hit_length
        load_masks = self.coord.load_mask(block_hashes, hit_length)
        tail_tokens = {b.group_id: b.num_tokens for b in hit.tail_key_boundaries}
        keys: list[str] = []
        key_meta: list[tuple[int, bytes]] = []
        for g_idx, db in enumerate(self.token_dbs):
            if not self.kv_cache_groups[g_idx].kv_cache_spec.prefix_cacheable:
                continue
            mask = load_masks[g_idx]
            key_prefixes = self.lookup_key_prefixes[g_idx]
            for start, end, block_hash in db.process_tokens(
                hit_length, block_hashes, mask_num
            ):
                chunk_idx = start // db.block_size
                if chunk_idx >= len(mask) or not mask[chunk_idx]:
                    continue
                if end == hit_length and g_idx in tail_tokens:
                    block_hash = block_hashes[
                        tail_tokens[g_idx] // db.hash_block_size - 1
                    ]
                keys.extend(
                    PoolKey.build_key_string(key_prefix, block_hash.hex())
                    for key_prefix in key_prefixes
                )
                key_meta.extend([(g_idx, bytes(block_hash))] * len(key_prefixes))
        return keys, key_meta

    def _lease(
        self,
        block_hashes: Sequence[BlockHash],
        hit: MooncakeLookupResult,
        mask_num: int,
    ) -> set[tuple[int, bytes]]:
        """Lease the keys a load of ``hit`` reads.

        Returns the ``(group, hash)`` pairs no longer in the store. Raises
        ``_LeaseFailedError`` when the lease RPC fails, including per-key error
        results.
        """
        keys, key_meta = self.loaded_keys(block_hashes, hit, mask_num)
        if not keys:
            return set()
        start = time.perf_counter()
        try:
            res = self._call("lookup_lease", self.store.batch_is_exist, keys)
        except Exception as e:
            self._record_lease_error(start, len(keys), len(keys))
            raise _LeaseFailedError(f"lease RPC raised: {e}") from e
        num_errors = sum(1 for r in res if r < 0)
        if num_errors:
            self._record_lease_error(start, len(keys), num_errors)
            raise _LeaseFailedError(
                f"lease RPC returned errors for {num_errors} of {len(keys)} keys"
            )
        evicted = [meta for meta, r in zip(key_meta, res, strict=True) if r != 1]
        self.record_operation(
            "lookup_lease",
            time.perf_counter() - start,
            len(keys),
            status="partial_failure" if evicted else "ok",
            num_failed_keys=len(evicted),
        )
        return set(evicted)

    def _call(
        self, op: str, fn: Callable[[list[str]], list[int]], keys: list[str]
    ) -> list[int]:
        if self._guard is None:
            return fn(keys)
        remaining_s = self._deadline - time.monotonic()
        if remaining_s <= 0:
            raise StoreCallTimeout(
                f"MooncakeLookup: {op} skipped, the lookup's "
                f"{self._guard.timeout_s:.1f} s ran out"
            )
        return self._guard.call(op, fn, keys, keys=keys, timeout_s=remaining_s)

    def close(self) -> None:
        if self._guard is not None:
            self._guard.shutdown()

    def _record_lease_error(self, start: float, num_keys: int, num_failed: int) -> None:
        self.record_operation(
            "lookup_lease",
            time.perf_counter() - start,
            num_keys,
            status="error",
            num_failed_keys=num_failed,
        )
