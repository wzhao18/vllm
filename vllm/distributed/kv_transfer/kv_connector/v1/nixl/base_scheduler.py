# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Base scheduler-side logic for the NIXL connector."""

import contextlib
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import msgspec
import zmq

from vllm import envs
from vllm.distributed.kv_transfer.kv_connector.utils import (
    BlockIds,
    EngineId,
    clip_ssm_state_blocks,
    yield_req_data,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    GET_META_MSG,
    LEASE_HEARTBEAT_MSG,
    LEASE_QUERY_MSG,
    HeartbeatInfo,
    NixlConnectorMetadata,
    NixlHandshakePayload,
    ReqId,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils import (
    get_representative_spec_type,
    zmq_ctx,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.utils.network_utils import make_zmq_path
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    FullAttentionSpec,
    MambaSpec,
    SlidingWindowSpec,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.dcp_pair_exchange import (
        PairUnit,
    )
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request

logger = init_logger(__name__)


_SIDE_HB_TIMEOUT_S = 2.0


def send_lease_heartbeat(
    host: str, port: int, req_ids: list[str], timeout_s: float
) -> list[bool]:
    """Renew P-side leases of ``req_ids`` over P's side channel."""
    path = make_zmq_path("tcp", host, port)
    with zmq_ctx(zmq.REQ, path) as sock:
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, max(1, int(timeout_s * 1000)))
        sock.send(msgspec.msgpack.encode((LEASE_HEARTBEAT_MSG, req_ids)))
        try:
            reply = sock.recv()
        except zmq.Again as e:
            raise TimeoutError(f"no heartbeat answer from {path}") from e
    return [r is True for r in msgspec.msgpack.decode(reply)]


class NixlBaseConnectorScheduler:
    """Base implementation of Scheduler side methods shared by pull and push."""

    # Emitted in kv_transfer_params so an external router can distinguish a
    # pull (READ) producer from a push (WRITE) one. Overridden by the push
    # scheduler.
    _TRANSFER_MODE: str = "pull"

    # DCP pair mode (decoder, pull; see dcp_pair_exchange.py). The class default
    # keeps schedulers built without __init__ (unit tests) off it.
    _pair_mode: bool = False

    def __init__(
        self,
        vllm_config: "VllmConfig",
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ):
        self.vllm_config = vllm_config
        parallel_config = vllm_config.parallel_config
        # TP1 PCP+DCP exposes its DCP shards as transfer ranks.
        self.transfer_tp_size = max(
            parallel_config.tensor_parallel_size,
            parallel_config.decode_context_parallel_size,
        )
        self.block_size = vllm_config.cache_config.block_size
        self.engine_id: EngineId = engine_id
        self.kv_cache_config = kv_cache_config
        self.side_channel_host = envs.VLLM_NIXL_SIDE_CHANNEL_HOST
        self.side_channel_port = (
            envs.VLLM_NIXL_SIDE_CHANNEL_PORT
            + vllm_config.parallel_config.data_parallel_index
        )
        assert vllm_config.kv_transfer_config is not None
        self._kv_lease_duration: int = (
            vllm_config.kv_transfer_config.get_from_extra_config(
                "kv_lease_duration", 30
            )
        )
        # NOTE (NickLucche): For now we use a hardcoded value for a simpler interface.
        self._heartbeat_interval = self._kv_lease_duration // 6
        if current_platform.device_type == "cpu":
            self.use_host_buffer = False
        else:
            self.use_host_buffer = (
                vllm_config.kv_transfer_config.kv_buffer_device == "cpu"
            )
        self._is_hma_required = (
            not vllm_config.scheduler_config.disable_hybrid_kv_cache_manager
            # Also handle unlikely SW-only model case instead of checking num_groups>1.
            and any(
                not isinstance(g.kv_cache_spec, FullAttentionSpec)
                for g in kv_cache_config.transfer_groups
            )
        )
        self._has_mamba = kv_cache_config.has_mamba_layers
        self._bounded_replay = any(
            g.kv_cache_spec.prefix_replay_tokens > 0
            for g in kv_cache_config.kv_cache_groups
        )

        logger.info("Initializing NIXL Scheduler %s", engine_id)
        if vllm_config.scheduler_config.disable_hybrid_kv_cache_manager:
            logger.info("Hybrid Memory Allocator is enabled with NIXL")

        # Background thread for handling new handshake requests.
        self._nixl_handshake_listener_t: threading.Thread | None = None
        self._stop_event = threading.Event()
        # Payloads the listener serves; a republish after a wake-up updates them.
        self._handshake_data: dict[tuple[int, int], bytes] = {}
        self._registration_generation = 0

        # Requests that need to start recv/send.
        # New requests are added by update_state_after_alloc in
        # the scheduler. Used to make metadata passed to Worker.
        self._reqs_need_recv: dict[
            ReqId, tuple[Request, BlockIds, tuple[int, ...], bool]
        ] = {}
        self._reqs_need_save: dict[ReqId, Request] = {}
        # Reqs to send and their expiration time
        self._reqs_need_send: dict[ReqId, float] = {}
        self._reqs_in_batch: set[ReqId] = set()
        # P side, pull mode: requests whose blocks are held for a remote READ.
        # Dropped when the workers report the send finished, before the
        # blocks are freed; read by the side-channel listener thread.
        self._held_send_leases: set[ReqId] = set()
        self._held_send_leases_lock = threading.Lock()
        # P side: renewals received on the side channel, for the next step.
        self._pending_lease_renewals: set[ReqId] = set()
        # Reqs to remove from processed set because they're not to send after
        # remote prefill or aborted.
        self._reqs_not_processed: set[ReqId] = set()

        # DCP pair mode (decoder, pull): partner DCP ranks swap half-blocks after
        # each pull at interleave 1. Only a JSON true enables it, as in the
        # worker, whose startup checks reject any other truthy value.
        self._pair_mode = (
            self._TRANSFER_MODE == "pull"
            and vllm_config.kv_transfer_config.get_from_extra_config(
                "dcp_pair_exchange", False
            )
            is True
        )
        # The exchanged groups: the DCP-sharded attention groups, whose blocks
        # the pull matches by DCP position. Transfer-group indices, as in the
        # worker's read path.
        self._pair_groups = (
            tuple(
                i
                for i, g in enumerate(kv_cache_config.transfer_groups)
                if g.kv_cache_spec.dcp_sharded
                and issubclass(
                    get_representative_spec_type(g.kv_cache_spec), AttentionSpec
                )
            )
            if self._pair_mode
            else ()
        )
        # Exchange plans, built at allocation. Ready once every rank finished
        # the pull, then emitted with the next step's connector metadata.
        self._pair_plans: dict[ReqId, list[PairUnit]] = {}
        self._pair_ready: dict[ReqId, list[PairUnit]] = {}
        if self._pair_mode:
            logger.info(
                "NIXL DCP pair mode: partner ranks exchange the blocks of "
                "transfer groups %s after each pull from a prefiller with half "
                "this DCP size.",
                self._pair_groups,
            )

        # Heartbeat tracking: requests needing periodic lease-renewal heartbeats to
        # remote P-side, stored as ready-to-send HeartbeatInfo grouped by remote engine
        self._heartbeat_by_engine: dict[EngineId, HeartbeatInfo] = {}
        # Reverse lookup: local req_id -> (engine_id, remote_req_id) for O(1) removal
        self._heartbeat_req_engine: dict[ReqId, tuple[EngineId, ReqId]] = {}
        self._last_heartbeat_time: float = 0.0
        # D side: remote request ids whose P answers side-channel heartbeats
        # (P advertised remote_lease_check). Guarded with the heartbeat maps
        # by _heartbeat_lock: the side-channel heartbeat thread reads them.
        self._side_hb_req_ids: set[ReqId] = set()
        self._heartbeat_lock = threading.Lock()
        self._side_hb_thread: threading.Thread | None = None
        self._side_hb_enabled = bool(
            vllm_config.kv_transfer_config.get_from_extra_config(
                "kv_side_channel_heartbeat", True
            )
        )

        # Gather Sliding Window sizes for each kv cache group (if any) in number of
        # blocks per KV cache group. This is used to clip the local attention window.
        sw_sizes_tokens: list[tuple[int, int]] = [
            (g.kv_cache_spec.sliding_window, g.kv_cache_spec.block_size)
            if isinstance(g.kv_cache_spec, SlidingWindowSpec)
            else (0, self.block_size)
            for g in kv_cache_config.transfer_groups
        ]
        # cdiv(n_tokens, block_size) gives blocks/window; add 1 to conservatively
        # account for boundary overlap eg window isn't fully aligned with blocks.
        self.blocks_per_sw = [
            cdiv(n_tokens, block_size) + 1 if n_tokens else 0
            for n_tokens, block_size in sw_sizes_tokens
        ]

        # Trailing scratch slots that mamba managers co-allocate per request
        # for speculative decoding; None for non-SSM groups.
        self._ssm_spec_blocks = [
            g.kv_cache_spec.num_speculative_blocks
            if isinstance(g.kv_cache_spec, MambaSpec)
            else None
            for g in kv_cache_config.transfer_groups
        ]

        # Threshold to decide whether to compute kv cache locally
        # or pull from a remote node: minimum number of remote
        # tokens to amortize the xfer latencies
        self.kv_recompute_threshold: int = int(
            vllm_config.kv_transfer_config.get_from_extra_config(
                "kv_recompute_threshold", 64
            )
        )

        # Bi-directional KV transfer feature supports KV block
        # transfers from D node to P node
        self.is_bidirectional_kv_xfer_enabled = (
            vllm_config.kv_transfer_config.get_from_extra_config(
                "bidirectional_kv_xfer", False
            )
        )
        self.decoder_kv_blocks_ttl = (
            vllm_config.kv_transfer_config.get_from_extra_config(
                "decoder_kv_blocks_ttl", 480
            )
        )

        if self.is_bidirectional_kv_xfer_enabled and self.kv_recompute_threshold > 0:
            logger.info(
                "Bidirectional KV transfer is enabled and the kv "
                "recompute threshold is set to %d tokens."
                "KV blocks on D are released after a TTL of %d seconds.",
                self.kv_recompute_threshold,
                self.decoder_kv_blocks_ttl,
            )

    def shutdown(self):
        self._stop_event.set()
        if self._nixl_handshake_listener_t is not None:
            self._nixl_handshake_listener_t.join()
            self._nixl_handshake_listener_t = None

    def on_new_request(self, request: "Request") -> None:
        """Track a request that may need heartbeats."""
        params = request.kv_transfer_params
        if params is not None and params.get("do_remote_decode"):
            self._truncate_request_for_prefill(request)

        # NOTE (NickLucche) This excludes request meant for P, ie heartbeats are
        # effectively disabled for Bidirectional KV transfer.
        if params is None or not params.get("do_remote_prefill"):
            return
        # Only track if all required remote fields are present.
        remote_engine_id = params.get("remote_engine_id")
        remote_request_id = params.get("remote_request_id")
        host = params.get("remote_host")
        port = params.get("remote_port")
        tp_size = params.get("tp_size")
        dcp_size = params.get("dcp_size", 1)
        pp_size = params.get("pp_size", 1)
        if (
            remote_engine_id is None
            or remote_request_id is None
            or host is None
            or port is None
            or tp_size is None
        ):
            return
        with self._heartbeat_lock_ctx():
            if remote_engine_id not in self._heartbeat_by_engine:
                self._heartbeat_by_engine[remote_engine_id] = HeartbeatInfo(
                    req_ids=set(),
                    host=host,
                    port=port,
                    tp_size=tp_size,
                    dcp_size=dcp_size,
                    pp_size=pp_size,
                )
            self._heartbeat_by_engine[remote_engine_id].req_ids.add(remote_request_id)
            self._heartbeat_req_engine[request.request_id] = (
                remote_engine_id,
                remote_request_id,
            )
            if params.get("remote_lease_check") is True and getattr(
                self, "_side_hb_enabled", False
            ):
                self._side_hb_req_ids.add(remote_request_id)
                self._ensure_side_heartbeat_thread()

    def _heartbeat_lock_ctx(self):
        lock = getattr(self, "_heartbeat_lock", None)
        return lock if lock is not None else contextlib.nullcontext()

    def _stop_heartbeat(self, req_id: ReqId) -> None:
        """Remove *req_id* from heartbeat tracking (if tracked)."""
        with self._heartbeat_lock_ctx():
            if key := self._heartbeat_req_engine.pop(req_id, None):
                engine_id, remote_id = key
                if hasattr(self, "_side_hb_req_ids"):
                    self._side_hb_req_ids.discard(remote_id)
                if info := self._heartbeat_by_engine.get(engine_id):
                    info.req_ids.discard(remote_id)
                    if not info.req_ids:
                        # Clean up empty engines so we don't leak a key when
                        # remote dies.
                        del self._heartbeat_by_engine[engine_id]

    def _ensure_side_heartbeat_thread(self) -> None:
        if self._side_hb_thread is not None:
            return
        self._side_hb_thread = threading.Thread(
            target=self._side_heartbeat_loop,
            daemon=True,
            name="nixl_lease_heartbeat",
        )
        self._side_hb_thread.start()

    def _side_heartbeat_targets(self) -> list[tuple[str, int, list[ReqId]]]:
        targets = []
        with self._heartbeat_lock:
            for info in self._heartbeat_by_engine.values():
                ids = [r for r in info.req_ids if r in self._side_hb_req_ids]
                if ids:
                    targets.append((info.host, info.port, ids))
        return targets

    def _send_side_heartbeats(self) -> None:
        """One round of lease heartbeats over P's side channel."""
        for host, port, ids in self._side_heartbeat_targets():
            try:
                send_lease_heartbeat(host, port, ids, _SIDE_HB_TIMEOUT_S)
            except Exception as e:
                logger.debug(
                    "Side-channel lease heartbeat to %s:%s failed: %r",
                    host,
                    port,
                    e,
                )

    def _side_heartbeat_loop(self) -> None:
        """Renew P leases while requests wait for their KV, off the step loop.

        The NIXL heartbeat is built in D's step loop and rides the NIXL/UCX
        endpoint of the reads it protects, so a stalled step loop (DP
        lockstep) or a stalled endpoint lets leases expire mid-pull.
        """
        interval = max(1.0, min(float(self._heartbeat_interval), 5.0))
        while not self._stop_event.wait(interval):
            self._send_side_heartbeats()

    def _queue_lease_renewals(self, req_ids: list[str]) -> list[bool]:
        """P side (listener thread): renew leases still held; the workers
        apply it with the next step's metadata."""
        with self._held_send_leases_lock:
            held = [r in self._held_send_leases for r in req_ids]
            self._pending_lease_renewals.update(r for r, h in zip(req_ids, held) if h)
        return held

    def get_exchange_clipped_blocks(
        self,
        block_ids: BlockIds,
        clip_ssm: bool = True,
        num_tokens: int | None = None,
    ) -> BlockIds:
        """Clip a request's block lists down to the transferable blocks.

        Sliding-window groups keep only the in-window tail: the KV cache
        manager allocates blocks for the entire sequence length and cleans up
        out-of-window blocks only prior to the `request_finished_all_groups`
        hook.

        SSM groups keep only their state-bearing slot: the trailing
        speculative scratch slots and everything before the running state
        (null placeholders and the previous step's superseded state) go.

        Use this at every block-id exchange point. Pass ``clip_ssm=False``
        for per-step partial lists (host-buffer save), where the SSM strip
        does not apply. Pass the computed ``num_tokens`` to drop speculative
        lookahead blocks first, so a window ends at the last computed token
        on both sides.
        """
        if len(block_ids) == 0:
            # No blocks to clip, e.g. a full prefix cache hit.
            return block_ids
        block_ids = self.kv_cache_config.select_transfer_block_ids(block_ids)
        if not self._is_hma_required:
            return block_ids
        # NOTE (NickLucche) This logic is currently handled at the connector level
        # because offloading connectors might want to receive the whole sequence even
        # for SWA groups. We will abstract this logic once the interface is more stable
        assert len(block_ids) == len(self.blocks_per_sw), (
            "Number of KV cache groups must match"
        )
        clipped = []
        for i, blocks in enumerate(block_ids):
            if n_sw := self.blocks_per_sw[i]:
                if num_tokens is not None:
                    block_size = self.kv_cache_config.transfer_groups[
                        i
                    ].kv_cache_spec.block_size
                    blocks = blocks[: cdiv(num_tokens, block_size)]
                blocks = blocks[-n_sw:]
            elif (
                clip_ssm
                and blocks
                and (n_spec_blocks := self._ssm_spec_blocks[i]) is not None
            ):
                blocks = clip_ssm_state_blocks(blocks, n_spec_blocks)
            clipped.append(blocks)
        return tuple(clipped)

    def set_xfer_handshake_metadata(
        self, metadata: dict[tuple[int, int], KVConnectorHandshakeMetadata]
    ) -> None:
        """Set the KV connector handshake metadata for this connector.

        Args:
            metadata (dict): the handshake metadata to set.

        """
        encoder = msgspec.msgpack.Encoder()
        for (pp_rank, tp_rank), rank_metadata in metadata.items():
            if not isinstance(rank_metadata, NixlHandshakePayload):
                raise ValueError(
                    "NixlConnectorScheduler expects NixlHandshakePayload for "
                    "handshake metadata."
                )
            encoded = encoder.encode(rank_metadata)
            if self._handshake_data.get((pp_rank, tp_rank)) != encoded:
                # Peers must handshake again to load the changed registration.
                self._registration_generation += 1
            self._handshake_data[(pp_rank, tp_rank)] = encoded
            logger.debug(
                "PP rank %d, TP rank %d: encoded NixlHandshakePayload size: %s bytes",
                pp_rank,
                tp_rank,
                str(len(self._handshake_data[(pp_rank, tp_rank)])),
            )

        # Only start the listener when we have metadata to serve.
        if self._nixl_handshake_listener_t is None:
            ready_event = threading.Event()
            self._nixl_handshake_listener_t = threading.Thread(
                target=self._nixl_handshake_listener,
                args=(
                    self._handshake_data,
                    ready_event,
                    self._stop_event,
                    self.side_channel_host,
                    self.side_channel_port,
                ),
                kwargs={
                    "lease_query": self._query_send_leases,
                    "lease_renew": self._queue_lease_renewals,
                },
                daemon=True,
                name="nixl_handshake_listener",
            )
            self._nixl_handshake_listener_t.start()
            ready_event.wait()  # Wait for listener ZMQ socket to be ready.

    @staticmethod
    def _nixl_handshake_listener(
        encoded_data: dict[tuple[int, int], Any],
        ready_event: threading.Event,
        stop_event: threading.Event,
        host: str,
        port: int,
        lease_query: Callable[[list[str]], list[bool]] | None = None,
        lease_renew: Callable[[list[str]], list[bool]] | None = None,
    ):
        """Background thread for getting new NIXL handshakes."""
        # NOTE(rob): this is a simple implementation. We will move
        # to a better approach via HTTP endpoint soon.

        # Listen for new requests for metadata.
        path = make_zmq_path("tcp", host, port)
        logger.debug("Starting listening on path: %s", path)
        with zmq_ctx(zmq.ROUTER, path) as sock:
            sock.setsockopt(zmq.RCVTIMEO, 1000)
            ready_event.set()
            while True:
                try:
                    identity, _, msg = sock.recv_multipart()
                except zmq.Again:
                    if stop_event.is_set():
                        break
                    continue
                decoded = msgspec.msgpack.decode(msg)
                if decoded and decoded[0] == LEASE_HEARTBEAT_MSG:
                    # (LEASE_HEARTBEAT_MSG, [req_ids]) -> [renewed].
                    req_ids = list(decoded[1])
                    renewed = (
                        lease_renew(req_ids)
                        if lease_renew is not None
                        else [False] * len(req_ids)
                    )
                    sock.send_multipart(
                        (identity, b"", msgspec.msgpack.encode(renewed))
                    )
                    continue
                if decoded and decoded[0] == LEASE_QUERY_MSG:
                    # (LEASE_QUERY_MSG, [req_ids]) -> [held].
                    req_ids = list(decoded[1])
                    held = (
                        lease_query(req_ids)
                        if lease_query is not None
                        else [False] * len(req_ids)
                    )
                    sock.send_multipart((identity, b"", msgspec.msgpack.encode(held)))
                    continue
                # Decode (GET_META_MSG, pp_rank, tp_rank).
                msg, target_pp_rank, target_tp_rank = decoded
                logger.debug(
                    "Received message for pp rank %s, tp rank %s",
                    target_pp_rank,
                    target_tp_rank,
                )
                if msg != GET_META_MSG:
                    logger.warning("Connection listener got unexpected message %s", msg)
                # Echo our perf_counter so P can estimate the clock offset.
                # perf_counter is only comparable within a process, so this
                # listener must run in the same process that stamps the block
                # expiry deadline (`_reqs_need_send`).
                ts = msgspec.msgpack.encode(time.perf_counter())
                sock.send_multipart(
                    (identity, b"", encoded_data[(target_pp_rank, target_tp_rank)], ts)
                )

    def _prefill_backoff(self) -> int:
        """Trailing prompt tokens the prefiller must not compute; the decoder
        recomputes them locally.

        Mamba needs h(N-1) so the decoder can derive h(N) itself. DSV41's sliding
        window under bounded replay is not replayed after a P/D load, so it must
        be laid out for the token the decoder recomputes. Multi-module
        MTP needs to keep its whole lookahead window off of the prefiller, which
        would otherwise embed the unverified drafts in the MTP layer's KV cache. The
        decoder would never rebuild them, because the update is sized by the rejection
        count, which is zero for the first decode.
        """
        return max(
            1 if self._has_mamba or self._bounded_replay else 0,
            self.vllm_config.num_prefill_lookahead_tokens - 1,
        )

    def _get_remote_prefill_token_count(self, num_prompt_tokens: int) -> int:
        """D-side only. The number of prompt tokens to load from the prefiller.
        Stops short of the trailing ``_prefill_backoff()`` tokens that the decoder
        will recompute locally."""
        backoff = self._prefill_backoff()
        if backoff and num_prompt_tokens > backoff:
            return num_prompt_tokens - backoff
        return num_prompt_tokens

    def _truncate_request_for_prefill(self, request: "Request") -> None:
        """P-side only: drop the trailing ``_prefill_backoff()`` prompt tokens
        so the prefiller stops short of what the decoder recomputes locally.
        For Mamba that is the single token needed to yield h(N-1); for
        multi-module MTP it is the drafter's whole lookahead window.

        Guarded by ``_p_side_truncated`` to avoid repeated truncation if the
        request is preempted and rescheduled."""
        backoff = self._prefill_backoff()
        params = request.kv_transfer_params
        if (
            backoff
            and params is not None
            # Guard against repeated truncation after preemption/reschedule.
            and not params.get("_p_side_truncated")
            and request.num_prompt_tokens > backoff
            # A request that skips reading external KV (prompt logprobs)
            # loads nothing on the decoder, which recomputes past its local hit.
            and not request.skip_reading_external_kv
        ):
            if request.prompt_token_ids is not None:
                del request.prompt_token_ids[-backoff:]
            elif request.prompt_embeds is not None:
                request.prompt_embeds = request.prompt_embeds[:-backoff]
            else:
                return

            del request._all_token_ids[-backoff:]
            request.num_prompt_tokens -= backoff
            request.max_tokens = 1
            params["_p_side_truncated"] = True

    def _build_save_meta(
        self,
        meta: NixlConnectorMetadata,
        scheduler_output: SchedulerOutput,
    ) -> None:
        # only called when use_host_buffer is True to build the save metadata

        # NOTE: For the prefill side, there might be a chance that an early added
        # request is a chunked prefill, so we need to check if new blocks are added
        for req_id, new_block_id_groups, _ in yield_req_data(scheduler_output):
            req_to_save = self._reqs_need_save.get(req_id)
            if req_to_save is None or new_block_id_groups is None:
                continue
            req = req_to_save

            assert req.kv_transfer_params is not None
            clipped_block_id_groups = self.get_exchange_clipped_blocks(
                new_block_id_groups, clip_ssm=False
            )
            meta.add_new_req_to_save(
                request_id=req_id,
                local_block_ids=clipped_block_id_groups,
                kv_transfer_params=req.kv_transfer_params,
            )
            assert scheduler_output.num_scheduled_tokens is not None
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
            is_partial = (
                req.num_computed_tokens + num_scheduled_tokens
            ) < req.num_prompt_tokens
            if not is_partial:
                # For non-partial prefills, once new req_meta is scheduled, it
                # can be removed from _reqs_need_save.
                # For partial prefill case, we will retain the request in
                # _reqs_need_save until all blocks are scheduled with req_meta.
                # Therefore, only pop if `not is_partial`.
                self._reqs_need_save.pop(req_id)

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = NixlConnectorMetadata()

        # Loop through scheduled reqs and convert to ReqMeta.
        for req_id, (
            req,
            block_ids,
            cached,
            awaiting_kvs,
        ) in self._reqs_need_recv.items():
            assert req.kv_transfer_params is not None
            meta.add_new_req_to_recv(
                request_id=req_id,
                local_block_ids=block_ids,
                kv_transfer_params=req.kv_transfer_params,
                local_num_computed_blocks=cached,
                awaiting_kvs=awaiting_kvs,
            )

        if self.use_host_buffer:
            self._build_save_meta(meta, scheduler_output)

        meta.reqs_to_send = self._reqs_need_send
        # Clock reference for reqs_to_send: deadlines above are in this
        # process's perf_counter domain; workers (possibly on other nodes,
        # where perf_counter has a different epoch) rebase against this.
        meta.scheduler_clock = time.perf_counter()
        meta.reqs_in_batch = self._reqs_in_batch
        meta.reqs_not_processed = self._reqs_not_processed

        if getattr(self, "_pending_lease_renewals", None):
            with self._held_send_leases_lock:
                meta.lease_renewals = self._pending_lease_renewals
                self._pending_lease_renewals = set()

        if self._pair_mode and self._pair_ready:
            # This step is the first that can promote these requests, so every
            # rank exchanges their blocks before any forward reads them.
            meta.reqs_to_pair_exchange, self._pair_ready = self._pair_ready, {}

        # Package heartbeats, throttled by heartbeat_interval.
        if self._heartbeat_by_engine:
            now = time.perf_counter()
            if now - self._last_heartbeat_time >= self._heartbeat_interval:
                self._last_heartbeat_time = now
                meta.heartbeat_by_engine = self._heartbeat_by_engine

        # Clear the list once workers start the transfers
        self._reqs_need_recv.clear()
        self._reqs_in_batch = set()
        self._reqs_not_processed = set()
        self._reqs_need_send = {}

        return meta

    def _hold_send_lease(self, req_id: ReqId) -> None:
        with self._held_send_leases_lock:
            self._held_send_leases.add(req_id)

    def _query_send_leases(self, req_ids: list[str]) -> list[bool]:
        """Whether each request's blocks are still held for a remote READ.

        Runs on the side-channel listener thread. A request leaves the set
        (update_connector_output) before the scheduler frees its blocks, so
        True means none of them has been freed, let alone reallocated.
        """
        with self._held_send_leases_lock:
            return [r in self._held_send_leases for r in req_ids]

    def update_connector_output(self, connector_output: "KVConnectorOutput") -> None:
        """Stop heartbeating for requests whose KV transfer completed."""
        for req_id in connector_output.finished_recving or ():
            self._stop_heartbeat(req_id)
            # A pull completes once, on every rank (a retry after a remote
            # disconnect reports nothing until it ends). A failed pull drops its
            # plan: the request recomputes from token 0 or fails.
            if (
                self._pair_mode
                and (units := self._pair_plans.pop(req_id, None))
                and req_id not in connector_output.failed_recving
            ):
                self._pair_ready[req_id] = units
        # Called before the scheduler frees finished_sending blocks.
        lock = getattr(self, "_held_send_leases_lock", None)
        if connector_output.finished_sending and lock is not None:
            with lock:
                self._held_send_leases.difference_update(
                    connector_output.finished_sending
                )

    def has_pending_push_work(self) -> bool:
        return False

    ############################################################
    # Abstract methods that subclasses must implement
    ############################################################

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int, bool]:
        raise NotImplementedError

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        raise NotImplementedError

    def request_finished(
        self,
        request: "Request",
        block_ids: BlockIds,
    ) -> tuple[bool, dict[str, Any] | None]:
        raise NotImplementedError
