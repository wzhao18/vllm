# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pull-specific scheduler-side logic for the NIXL connector."""

import time
from typing import TYPE_CHECKING, Any

from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_scheduler import (
    NixlBaseConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.dcp_pair_exchange import (
    NixlPairModeError,
    check_pair_coverage,
    pair_units,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    KVFingerprintReq,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


class NixlPullConnectorScheduler(NixlBaseConnectorScheduler):
    """Pull-specific scheduler logic (READ-based KV transfer)."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, engine_id, kv_cache_config)

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int, bool]:
        """For remote prefill, pull all prompt blocks from remote
        asynchronously relative to engine execution.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request
        Returns:
            * the number of tokens that can be loaded from the
              external KV cache beyond what is already computed.
            * true if the external KV cache tokens will be loaded
              asynchronously (between scheduler steps).

        """
        params = request.kv_transfer_params
        logger.debug(
            "NIXLConnector get_num_new_matched_tokens: "
            "num_computed_tokens=%s, kv_transfer_params=%s",
            num_computed_tokens,
            params,
        )

        if params is not None and params.get("do_remote_prefill"):
            # Remote prefill: get all prompt blocks from remote.
            token_ids = request.prompt_token_ids or []
            actual = self._get_remote_prefill_token_count(len(token_ids))
            count = actual - num_computed_tokens
            if count > 0:
                return count, True

        if (
            params is not None
            and params.get("do_remote_decode")
            and params.get("remote_block_ids")
            and all(
                p in params
                for p in (
                    "remote_engine_id",
                    "remote_request_id",
                    "remote_host",
                    "remote_port",
                )
            )
        ):
            # Decode node has kv blocks for part of prefill request, so, provide them
            # as an external token count to scheduler.
            # The tokens will be loaded if not already present
            # in the prefill node local cache
            remote_num_tokens = params.get("remote_num_tokens") or 0
            count = (
                min(remote_num_tokens, request.num_prompt_tokens) - num_computed_tokens
            )
            if count > 0:
                # Check kv_recompute_threshold: skip pull if
                # remote tokens are below the threshold.
                if (
                    self.kv_recompute_threshold > 0
                    and count < self.kv_recompute_threshold
                ):
                    logger.debug(
                        "Skipping remote pull for %s: %d remote tokens < threshold %d",
                        request.request_id,
                        count,
                        self.kv_recompute_threshold,
                    )
                    return 0, False
                return count, True

        # No remote prefill for this request.
        return 0, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        params = request.kv_transfer_params
        logger.debug(
            "NIXLConnector update_state_after_alloc: "
            "num_external_tokens=%s, kv_transfer_params=%s",
            num_external_tokens,
            params,
        )

        if not params:
            return

        if params.get("do_remote_decode") or (
            params.get("do_remote_prefill") and self.is_bidirectional_kv_xfer_enabled
        ):
            self._reqs_in_batch.add(request.request_id)
        if self.use_host_buffer and params.get("do_remote_decode"):
            # NOTE: when accelerator is not directly supported by Nixl,
            # prefilled blocks need to be saved to host memory before transfer.
            self._reqs_need_save[request.request_id] = request
        elif params.get("do_remote_prefill") or (
            params.get("do_remote_decode")
            and self.is_bidirectional_kv_xfer_enabled
            and not params.get("_remote_blocks_processed")
        ):
            if params.get("remote_block_ids"):
                if all(
                    p in params
                    for p in (
                        "remote_engine_id",
                        "remote_request_id",
                        "remote_host",
                        "remote_port",
                    )
                ):
                    # If remote_blocks and num_external_tokens = 0, we have
                    # a full prefix cache hit on the local node. We need to call
                    # send_notif in _read_blocks to free the memory on the remote node.

                    unhashed_local_block_ids: BlockIds = (
                        blocks.get_unhashed_block_ids_all_groups()
                        if num_external_tokens > 0
                        else ()
                    )
                    local_block_ids = self.get_exchange_clipped_blocks(
                        unhashed_local_block_ids
                    )
                    # Blocks covered by the local prefix cache, per KV cache group.
                    # Each count fixes where that group's DCP slice starts, which the
                    # worker needs to line up with the remote's slice.
                    local_num_computed_blocks = tuple(
                        sum(
                            block.block_hash is not None and not block.is_null
                            for block in group
                        )
                        for group in blocks.blocks
                    )
                    if (
                        self._pair_mode
                        and params.get("do_remote_prefill")
                        and num_external_tokens > 0
                    ):
                        self._plan_pair_exchange(
                            request,
                            blocks,
                            local_block_ids,
                            local_num_computed_blocks,
                            num_external_tokens,
                        )

                    # Get unhashed blocks to pull from remote. Mind that a full prefix
                    # cache hit is indicated with an empty list.
                    self._reqs_need_recv[request.request_id] = (
                        request,
                        local_block_ids,
                        local_num_computed_blocks,
                        # Parked in WAITING_FOR_REMOTE_KVS only when there is
                        # something to pull; a full local hit stays RUNNING.
                        num_external_tokens > 0,
                    )

                else:
                    logger.warning(
                        "Got invalid KVTransferParams: %s. This "
                        "request will not utilize KVTransfer",
                        params,
                    )
            else:
                assert num_external_tokens == 0
            # Only trigger 1 KV transfer per request.
            params["do_remote_prefill"] = False
            params["_remote_blocks_processed"] = True

    def _plan_pair_exchange(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        local_block_ids: BlockIds,
        local_num_computed_blocks: tuple[int, ...],
        num_external_tokens: int,
    ) -> None:
        """DCP pair mode: record the blocks partner ranks exchange once every
        rank has pulled ``request`` from a prefiller with half this DCP size.

        Raises:
            NixlPairModeError: The prefiller did not advertise its DCP size, or
                computed fewer tokens or blocks than this decoder loads. That
                is P/D configuration skew, so it stops the engine instead of
                leaving rows unexchanged or zero-filling the tail.

        """
        params = request.kv_transfer_params
        assert params is not None
        remote_dcp_size = params.get("dcp_size")
        if remote_dcp_size is None:
            # The workers pair by the handshake's DCP size; without this one,
            # a pair-mode pull would go unexchanged.
            raise NixlPairModeError(
                "DCP pair exchange: the prefiller of request "
                f"{request.request_id} did not advertise its DCP size"
            )
        dcp_size = self.vllm_config.parallel_config.decode_context_parallel_size
        if remote_dcp_size * 2 != dcp_size:
            # An equal DCP size needs no exchange; the handshake rejects others.
            return
        remote_block_ids = params["remote_block_ids"]
        num_tokens = self._get_remote_prefill_token_count(
            len(request.prompt_token_ids or [])
        )
        check_pair_coverage(
            num_tokens,
            params.get("remote_num_tokens"),
            remote_block_ids,
            self._pair_groups,
            remote_block_tokens=self.block_size * remote_dcp_size,
        )
        if units := pair_units(
            local_block_ids,
            local_num_computed_blocks,
            remote_block_ids,
            self._pair_groups,
        ):
            self._pair_plans[request.request_id] = units
        if self._fingerprint:
            # The request's own blocks, and its hit from the token counts: a
            # plan with a wrong 2m offset must not shift the logged tokens too.
            self._fingerprint_pulls[request.request_id] = KVFingerprintReq(
                req_id=params["remote_request_id"],
                block_ids=blocks.get_block_ids(),
                num_tokens=num_tokens,
                first_block=(num_tokens - num_external_tokens)
                // (self.block_size * dcp_size),
            )

    def request_finished(
        self,
        request: "Request",
        block_ids: "BlockIds",
    ) -> tuple[bool, dict[str, Any] | None]:
        """Once a request is finished, determine whether request blocks
        should be freed now or will be sent asynchronously and freed later.
        """
        from vllm.v1.request import RequestStatus

        if self._pair_mode:
            # Before any early return: a finished request's blocks are freed
            # when its pull ends, or at once, so its exchange must not run.
            self._pair_plans.pop(request.request_id, None)
            self._pair_ready.pop(request.request_id, None)
            if self._fingerprint:
                self._fingerprint_pulls.pop(request.request_id, None)
                self._fingerprint_pulls_ready.pop(request.request_id, None)

        params = request.kv_transfer_params
        logger.debug(
            "NIXLConnector request_finished(%s), request_status=%s, "
            "kv_transfer_params=%s",
            request.request_id,
            request.status,
            params,
        )
        if not params:
            return False, None

        is_p_node = bool(params.get("do_remote_decode"))
        is_d_node = not is_p_node

        # Stop heartbeating for aborted requests that never reached finished_recving:
        # normal path cleans up in update_connector_output.
        self._stop_heartbeat(request.request_id)

        if params.get("do_remote_prefill"):
            # If do_remote_prefill is still True when the request is finished,
            # update_state_after_alloc must not have been called (the request
            # must have been aborted before it was scheduled, e.g. via the
            # abort_immediately path used to clean up KV-transfer requests
            # rejected at the D-side serving layer).
            # To avoid stranding the prefill blocks in the prefill instance,
            # we must add empty block_ids to _reqs_need_recv so that our
            # worker side will notify and free blocks in the prefill instance.
            self._reqs_need_recv[request.request_id] = (request, [], (), False)
            params["do_remote_prefill"] = False
            return False, None

        if is_d_node and not self.is_bidirectional_kv_xfer_enabled:
            return False, None

        if request.status not in (
            RequestStatus.FINISHED_LENGTH_CAPPED,
            RequestStatus.FINISHED_STOPPED,
        ):
            # Also include the case of a P/D Prefill request with immediate
            # block free (eg abort). Stop tracking this request.
            self._reqs_not_processed.add(request.request_id)
            # Clear _reqs_need_save if a request is aborted as partial prefill.
            self._reqs_need_save.pop(request.request_id, None)
            return False, None

        # TODO: check whether block_ids actually ever be 0. If not we could
        # remove the conditional below
        delay_free_blocks = any(len(group) > 0 for group in block_ids)
        remote_num_tokens = 0
        blocks_expiry_time = None
        lease_deadline = None
        if delay_free_blocks:
            # Prefill request on remote. It will be read from D upon completion
            request_kv_blocks_ttl = self._kv_lease_duration
            if is_d_node:
                # For blocks pinned on D, use a simpler timeout for now instead of a
                # lease mechanism as turn2 request is client-driven.
                request_kv_blocks_ttl = self.decoder_kv_blocks_ttl
            logger.debug(
                "NIXLConnector request_finished(%s) waiting for %d seconds "
                "before releasing blocks",
                request.request_id,
                request_kv_blocks_ttl,
            )
            self._reqs_need_send[request.request_id] = (
                time.perf_counter() + request_kv_blocks_ttl
            )
            if is_d_node:
                # D blocks expiry time exported for the turn-2 readback.
                blocks_expiry_time = self._reqs_need_send[request.request_id]
            else:
                # P->D pull: export the initial lease deadline (D's no-RTT
                # fast path) and answer lease queries until the blocks are
                # released.
                lease_deadline = self._reqs_need_send[request.request_id]
                self._hold_send_lease(request.request_id)
                if self._fingerprint:
                    # Sent with reqs_to_send; the workers log it before they
                    # report it sent.
                    self._fingerprint_sends[request.request_id] = KVFingerprintReq(
                        req_id=request.request_id,
                        block_ids=tuple(list(group) for group in block_ids),
                        num_tokens=request.num_computed_tokens,
                    )
            # NOTE HMA will "mark" empty/null blocks in groups with 0s (eg SWA ones),
            # trimming down after allocating for the whole sequence length. Empty
            # blocks are always at the start of the list.
            # Here we "unpad" blocks to send the actual remote blocks to be read.
            remote_num_tokens = request.num_computed_tokens
            block_ids = self.get_exchange_clipped_blocks(
                block_ids, num_tokens=remote_num_tokens
            )

        return delay_free_blocks, dict(
            do_remote_prefill=is_p_node,
            do_remote_decode=is_d_node,
            remote_block_ids=block_ids,
            remote_engine_id=self.engine_id,
            remote_request_id=request.request_id,
            remote_host=self.side_channel_host,
            remote_port=self.side_channel_port,
            tp_size=self.transfer_tp_size,
            dcp_size=self.vllm_config.parallel_config.decode_context_parallel_size,
            pp_size=self.vllm_config.parallel_config.pipeline_parallel_size,
            remote_num_tokens=remote_num_tokens,
            remote_blocks_expiry_time=blocks_expiry_time,
            remote_registration_generation=self._registration_generation,
            remote_lease_deadline=lease_deadline,
            remote_lease_check=lease_deadline is not None,
            transfer_mode=self._TRANSFER_MODE,
        )
