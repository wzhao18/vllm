# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The engine and an external store must name the same replay boundary.

``MooncakeStoreCoordinator.get_replay_boundary`` used to be a hand-copied
mirror of ``KVCacheCoordinator.get_replay_boundary`` -- same policy, written
twice, in two files, against differently-named block-size attributes
(``scheduler_block_size`` vs ``lcm_block_size``). Review asked for one
function so the two cannot drift.

They now both delegate to ``kv_cache_utils.replay_boundary``. This pins the
property that matters: a store retaining at a position the engine does not
resume at keeps state where nothing can reach it.
"""

from types import SimpleNamespace

import pytest
import torch

from tests.v1.core.test_prefix_caching import make_kv_cache_manager
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.coordinator import (
    MooncakeStoreCoordinator,
)
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import init_none_hash, replay_boundary
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)


def _groups(block_size: int) -> list[KVCacheGroupSpec]:
    return [
        KVCacheGroupSpec(
            ["full"],
            FullAttentionSpec(
                block_size=block_size, num_kv_heads=1, head_size=1, dtype=torch.float32
            ),
        ),
        KVCacheGroupSpec(
            ["mamba"],
            MambaSpec(
                block_size=block_size,
                shapes=((1, 1),),
                dtypes=(torch.float32,),
                mamba_cache_mode="align",
            ),
        ),
    ]


def _engine_coordinator(block_size: int, hash_block_size: int, use_eagle: bool):
    init_none_hash(sha256)
    config = KVCacheConfig(
        num_blocks=1024,
        kv_cache_tensors=[],
        kv_cache_groups=_groups(block_size),
    )
    return make_kv_cache_manager(
        kv_cache_config=config,
        max_model_len=1 << 20,
        enable_caching=True,
        hash_block_size=hash_block_size,
        use_eagle=use_eagle,
    ).coordinator


BLOCK_SIZES = [(8, 2), (16, 4), (64, 16), (128, 8), (512, 32), (1536, 128)]
PROMPT_LENS = [1, 2, 7, 8, 9, 15, 63, 64, 65, 127, 512, 513, 1984, 2020, 4096, 12289]


@pytest.mark.parametrize("block_size,hash_block_size", BLOCK_SIZES)
@pytest.mark.parametrize("use_eagle", [True, False])
def test_store_and_engine_agree_on_the_replay_boundary(
    block_size: int, hash_block_size: int, use_eagle: bool
):
    """Same policy, two call sites, no drift."""
    engine = _engine_coordinator(block_size, hash_block_size, use_eagle)
    store = MooncakeStoreCoordinator(
        kv_cache_groups=_groups(block_size),
        scheduler_block_size=block_size,
        hash_block_size=hash_block_size,
        use_eagle=use_eagle,
    )
    assert bool(engine.eagle_group_ids) == bool(store.eagle_group_ids), (
        "the two coordinators disagree on whether eagle is on, so comparing "
        "their boundaries would not test the shared policy"
    )

    for num_prompt_tokens in PROMPT_LENS:
        request = SimpleNamespace(num_prompt_tokens=num_prompt_tokens)
        assert engine.get_replay_boundary(request) == store.get_replay_boundary(
            num_prompt_tokens
        ), (
            f"engine and store disagree at num_prompt_tokens="
            f"{num_prompt_tokens} (block_size={block_size}, eagle={use_eagle})"
        )


@pytest.mark.parametrize("block_size,hash_block_size", BLOCK_SIZES)
def test_replay_boundary_leaves_room_for_the_eagle_block(
    block_size: int, hash_block_size: int
):
    """Under eagle the block above the boundary must fit inside the prompt.

    That is the whole reason the boundary sits one alignment unit low; if it
    did not, the position the sibling matches and drops back from would run
    past the prompt.
    """
    for num_prompt_tokens in PROMPT_LENS:
        boundary = replay_boundary(num_prompt_tokens, block_size, use_eagle=True)
        assert boundary % block_size == 0, (
            f"boundary {boundary} is not on the block grid"
        )
        assert boundary >= 0
        if boundary:
            assert boundary + block_size <= num_prompt_tokens, (
                f"boundary {boundary} leaves no room for the eagle block "
                f"within {num_prompt_tokens} prompt tokens"
            )


def test_no_second_implementation_of_the_boundary_policy():
    """Guard the dedup: neither call site may recompute the policy inline.

    Review's concern was drift, not a live bug -- the two copies agreed at the
    time because ``lcm_block_size`` is assigned ``scheduler_block_size``. This
    fails if someone re-inlines the arithmetic instead of calling the helper.
    """
    import inspect

    from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator

    for fn in (
        KVCacheCoordinator.get_replay_boundary,
        MooncakeStoreCoordinator.get_replay_boundary,
    ):
        src = inspect.getsource(fn)
        assert "replay_boundary(" in src, f"{fn.__qualname__} does not delegate"
        assert "// self." not in src, (
            f"{fn.__qualname__} recomputes the boundary inline instead of "
            f"delegating to kv_cache_utils.replay_boundary"
        )
