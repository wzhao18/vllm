# MooncakeStoreConnector Write-Back Design

This note explains the write-back changes in this branch for
`MooncakeStoreConnector`. It is intended to help reviewers understand the
control flow, the scheduler and worker responsibilities, and the current
limitations of the first implementation.

## Summary

The existing Mooncake store path behaves as a write-through cache. Once a
request computes KV blocks, the connector stores those blocks to Mooncake while
the request finishes. This is simple, but it has two drawbacks:

- Store CPU work can contend with the engine even though it runs on a background
  Python thread.
- If the Mooncake CPU/DRAM segment is smaller than the GPU prefix-cache
  capacity, write-through can duplicate most recently completed requests in CPU
  memory even while the same blocks are still cached on GPU.

This branch adds a `store_policy="write_back"` mode. In write-back mode,
MooncakeStoreConnector does not store a request just because the request
finished. Instead, it stores hashed GPU prefix-cache blocks when the allocator
is about to reuse those free GPU blocks. The allocation is deferred while the
selected blocks are asynchronously written back. After all workers report the
write-back event complete, the scheduler releases those pinned blocks back to
the front of the free queue so allocation can retry.

The default behavior remains unchanged:

```json
{
  "kv_connector": "MooncakeStoreConnector",
  "kv_role": "kv_both"
}
```

Write-back is opt-in:

```json
{
  "kv_connector": "MooncakeStoreConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "store_policy": "write_back"
  }
}
```

## Configuration

`store_policy` selects the producer-side store policy:

- `write_through`: existing behavior. Request KV is stored as request lifecycle
  metadata is emitted. Finished request blocks may be held until async stores
  complete.
- `write_back`: new behavior. Request-finish stores are skipped. Evictable GPU
  prefix-cache blocks are stored when allocation is about to reuse them.

`write_back_max_blocks_per_step` is kept as a compatibility knob. Setting it to
`0` disables the write-back pre-allocation path. Positive values no longer cap
the batch size; a write-back event covers the hashed blocks in the free-queue
prefix that the allocator is about to reuse.

## High-Level Flow

### Write-through

In `write_through` mode, the connector keeps the old behavior:

1. A request computes prompt KV on GPU.
2. Scheduler metadata describes the newly saved request range.
3. The worker store thread writes those KV chunks to Mooncake.
4. If the request is finishing, the scheduler can delay freeing blocks until
   async store completion is observed.

### Write-back

In `write_back` mode, the request-completion path is intentionally quiet:

1. A request computes prompt KV on GPU.
2. When the request finishes, its hashed blocks enter the GPU prefix-cache free
   queue.
3. The connector does not immediately store those blocks to Mooncake.
4. Later, if a scheduler step needs to allocate GPU blocks, the allocator calls
   a pre-allocation hook before committing the allocation.
5. The Mooncake scheduler peeks at the candidate free blocks that allocation
   would reuse.
6. Hashed blocks that are not already written back are pinned and packaged into
   `WriteBackStoreMeta`.
7. The allocation returns `None` for that step, so the request remains
   schedulable and retries in a later scheduler step.
8. Worker store threads asynchronously write the selected GPU blocks to
   Mooncake.
9. Worker metadata reports completed write-back event IDs back to the
   scheduler.
10. Once all expected workers report an event, the scheduler marks those
    `(block_id, block_hash)` pairs as completed and releases the pinned blocks
    back to the front of the free queue.

The key property is that write-back happens just before reuse, not at request
finish.

## Scheduler Changes

Most of the policy is implemented in
`vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py`.

### Policy State

The scheduler parses:

- `store_policy`
- `write_back_max_blocks_per_step` (only `0` has special behavior)

It also tracks write-back state:

- `_gpu_block_pool`: the GPU `BlockPool` bound through the connector.
- `_write_back_events`: event ID to `(block_ids, block_hashes)`.
- `_write_back_pending_counts`: worker completion counts per event.
- `_write_back_inflight_block_ids`: physical GPU block IDs currently pinned for
  write-back.
- `_write_back_completed_blocks`: block hashes already written back for a
  physical block ID.
- `_pending_write_back_stores`: store metadata to attach to the next connector
  metadata object.

### Request-Finish Behavior

When `store_policy == "write_back"`, request-finish stores are skipped:

- `build_connector_meta()` uses `skip_save` for normal request-save metadata.
- `request_finished()` returns `(False, None)` in write-back mode, so the normal
  request-finish store path does not hold blocks for per-request saving.

This means completed requests leave their hashed blocks in the GPU prefix cache
until those blocks are actually needed for reuse.

### Allocation Deferral

The core scheduler now passes a pre-allocation callback to
`KVCacheManager.allocate_slots()`. The callback is invoked after the allocator
knows how many blocks it would allocate, but before the allocation mutates block
ownership.

For Mooncake write-back, the callback delegates to:

```python
MooncakeStoreScheduler.write_back_blocks_before_allocate(num_blocks)
```

If this returns `True`, `allocate_slots()` returns `None` and the scheduler does
not allocate blocks in that step. In the regular running-request allocation
loop, this also avoids treating the miss as normal memory pressure that would
preempt another request.

### Candidate Block Selection

`write_back_blocks_before_allocate()`:

1. Exits immediately unless the connector is a producer in `write_back` mode.
2. If a prior write-back event is pending or in flight, returns `True` so the
   allocation retries after completion instead of issuing another event.
3. Peeks at the first `num_blocks` entries in the GPU free queue using
   `BlockPool.peek_free_blocks()`.
4. For each candidate block, collects its current `block_hash` and any hash
   aliases in `cached_block_hashes_by_block`.
5. Skips null, un-hashed, in-flight, and already-completed `(block_id, hash)`
   pairs.
6. Pins the selected blocks by calling `BlockPool.touch()`.
7. Adds one `WriteBackStoreMeta` to `_pending_write_back_stores`.
8. Returns `True` to defer allocation.

If a block has multiple hash aliases, the same block ID can appear multiple
times in `WriteBackStoreMeta.block_ids`, paired with different hashes. It is
still pinned once as one distinct physical block.

### Completion Handling

Workers report completed write-back events through `MooncakeStoreWorkerMetadata`.
The scheduler aggregates counts across workers and waits for
`_expected_worker_count` reports before processing an event.

When an event completes:

1. The scheduler removes the event from `_write_back_events`.
2. It clears the event's physical block IDs from
   `_write_back_inflight_block_ids`.
3. It records each completed `(block_id, block_hash)` in
   `_write_back_completed_blocks`.
4. It releases the pinned GPU blocks with `BlockPool.free_blocks_to_front()`.

Returning blocks to the front preserves allocator reuse order for the blocks
that triggered the deferred allocation.

## Worker Changes

Worker-side changes are in
`vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py` and
`data.py`.

### Metadata

The branch adds:

- `WriteBackStoreMeta`: scheduler-to-worker metadata for write-back stores.
- `MooncakeStoreWorkerMetadata`: worker-to-scheduler metadata for completed
  write-back events.

`MooncakeStoreConnectorMetadata` now carries both request metadata and
write-back store metadata.

### Write-Back Store Path

The worker enqueue path records a CUDA event before submitting write-back store
work. The store thread synchronizes that event before reading GPU memory, so
the write-back copy is ordered after the model work that produced the KV data.

`KVCacheStoreSendingThread._handle_write_back_request()` handles
`WriteBackStoreMeta`:

1. Converts each `(block_id, block_hash_with_group)` to a Mooncake store key.
2. Uses `get_group_id()` to select the correct token database.
3. Uses the cached group hash directly, matching the normal write-through
   `process_tokens()`/`key_for()` lookup key path.
4. Uses `prepare_block_value()` to compute addresses and sizes for the full GPU
   KV block.
5. Applies existing `put_step` striping so each TP rank only stores its assigned
   subset of keys.
6. Calls `batch_is_exist()` and writes only missing keys.
7. Records `write_back_exists` and `write_back_put` operation telemetry.
8. Reports the write-back event complete in a `finally` block.

If a write-back put fails, the worker logs and records the failure but still
reports event completion so the scheduler does not pin GPU blocks indefinitely.
The consequence is a possible future external-cache miss, not an incorrect
cache hit.

## Lookup Union of GPU and External Cache

Write-back changes lookup semantics. With write-through, the external store was
the main source of truth for external prefix hits. With write-back, part of a
prefix can still be in the local GPU prefix cache while another part has already
been written back to Mooncake.

The branch therefore changes lookup to evaluate the union of:

- local GPU prefix-cache hits already known by vLLM, and
- Mooncake external-store hits.

The scheduler passes `local_hit_tokens=num_computed_tokens` into
`LookupKeyClient.lookup()`. The lookup protocol adds a `local_hit_tokens` frame,
with backward-compatible decoding for the old frame layout.

On the worker side, `MooncakeStoreWorker.lookup()`:

1. Marks chunks covered by `local_hit_tokens` as present in `exists_set`.
2. Queries Mooncake for the remaining candidate chunks.
3. Adds externally-present chunks to the same `exists_set`.
4. Calls `MooncakeStoreCoordinator.find_longest_cache_hit()` over that union.

This is required for cases where neither GPU nor Mooncake alone has a long
enough contiguous prefix, but the combination does.

## Allocator and Block-Pool Hooks

The branch adds two small block-pool helpers:

- `peek_free_blocks(num_blocks)`: inspect the next free blocks without removing
  them.
- `free_blocks_to_front(blocks)`: release pinned blocks back to the front of the
  free queue after write-back completion.

`KVCacheManager.allocate_slots()` now accepts an optional `pre_allocate_blocks`
callback. The callback can be invoked both when allocation would otherwise fail
for lack of free blocks and immediately before a successful allocation would be
committed.

The scheduler passes the Mooncake write-back callback into both relevant
allocation paths.

## Metrics and Observability

The write-back path records Mooncake store operation telemetry with these
operation names:

- `write_back_exists`
- `write_back_put`

Reduced logger stats derive fields such as:

- `write_back_exists_count`
- `write_back_exists_total_keys`
- `write_back_put_count`
- `write_back_put_total_bytes`
- `write_back_put_failed_keys`

Prometheus exports them through the existing Mooncake store metrics with the
`operation` label:

- `vllm:mooncake_store_operation_total`
- `vllm:mooncake_store_operation_keys_total`
- `vllm:mooncake_store_operation_bytes_total`
- `vllm:mooncake_store_operation_failed_keys_total`
- `vllm:mooncake_store_operation_time_seconds`

For example, `write_back_exists_count` in reduced logs is the number of
existence-check batches, while `write_back_exists_total_keys` is the number of
keys checked across those batches.

## Tests Added or Updated

Mooncake tests cover:

- Connector delegation for write-back allocation and worker metadata.
- Store-policy validation and write-through default behavior.
- Write-back skipping request-finish saves.
- No free-queue scan during metadata build.
- Pin and release of GPU blocks around write-back events.
- Preserving allocator reuse order after completion.
- Hash alias handling for reused physical blocks.
- Avoiding duplicate write-back for already-completed `(block_id, hash)` pairs.
- Rewriting when a physical block is later reused with a new hash.
- Worker-side write-back store key construction.
- Reusing the cached group hash directly when `hash_block_size != block_size`.
- `put_step` striping for write-back.
- Visibility of written-back blocks to lookup.
- Multi-group write-back lookup.
- Lookup protocol carrying `local_hit_tokens`.

Simple KV offload tests in this branch are test/support coverage only. They add
or adjust tests for:

- DMA copy grouping by KV-cache group.
- Store-copy stream ordering.
- CUDA memcpy argument lifetime and bounds checking.
- Worker handling of shared storage so the same tensor is not registered twice.

## End-to-End Sanity Result

The Llama 8B write-back sanity run used:

- model: `meta-llama/Llama-3.1-8B-Instruct`
- `store_policy`: `write_back`
- `write_back_max_blocks_per_step`: positive values do not cap the event size;
  use `0` only to disable the path.
- Mooncake segment: `8GB`
- GPU KV capacity: about `8628` blocks
- probe evictor count: `18`

The latest local result:

```json
{
  "passed": true,
  "failed_requests": 0,
  "probe_prompt_tokens": 43782,
  "probe_external_prefix_cache_hits": 36160.0,
  "probe_external_prefix_cache_queries": 43462.0,
  "fill_probe_exact_match_rate": 1.0
}
```

This confirms that blocks written back from GPU become visible to Mooncake
lookup and can be loaded during the probe phase.

## Current Limitations

This is a first implementation with intentionally limited scope:

- Write-back completion is event-level, not per-block or per-chunk. A later
  refinement could issue larger write-back sets in smaller independent chunks
  and release completed chunks earlier.
- The scheduler defers the whole allocation attempt when write-back is pending.
  It does not yet skip pending blocks and allocate from already-clean later
  blocks in the free queue.
- The write-back cap controls burst size. Very small values can create multiple
  deferred scheduler steps before a large allocation can proceed.
- A failed write-back releases the GPU block after recording/logging failure.
  This favors scheduler liveness over guaranteed external-cache residency.
- Only hashed GPU prefix-cache blocks are written back. Null blocks, un-hashed
  blocks, and actively referenced blocks are ignored.
- Mooncake capacity and eviction policy still determine whether written-back
  keys remain available long enough for a later hit.

## Files Touched

Main production files:

- `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/connector.py`
- `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/data.py`
- `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/protocol.py`
- `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py`
- `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py`
- `vllm/v1/core/block_pool.py`
- `vllm/v1/core/kv_cache_manager.py`
- `vllm/v1/core/sched/scheduler.py`

Documentation and tests:

- `docs/features/mooncake_store_connector_usage.md`
- `tests/v1/kv_connector/unit/test_mooncake_store_connector.py`
- `tests/v1/kv_connector/unit/test_mooncake_store_scheduler.py`
- `tests/v1/kv_connector/unit/test_mooncake_store_worker.py`
- `tests/v1/simple_kv_offload/test_copy_backend.py`
- `tests/v1/simple_kv_offload/test_cuda_mem_ops.py`
- `tests/v1/simple_kv_offload/test_worker.py`
