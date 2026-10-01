# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay user actions with an independent minimum-hit oracle.

Infinite store capacity, retention zero, DSpark and one FlashKDA checkpoint.
The oracle tracks reusable prompt checkpoints, not vLLM's cache tables. Extra
intermediate-state hits are allowed, but losing a promised checkpoint is not.
"""

import pytest

from tests.v1.kv_connector.unit.hybrid_mooncake_replay import (
    NUM_BLOCKS,
    REPLAY_SETUP,
    Replay,
    Sequence,
)

TEST_SEQUENCES = (
    # Store-side attention proof, including a partial and an aligned tail.
    Sequence("pr56615-a", (7_449, "reset", 7_449)),
    Sequence("pr56615-b", (23_196, "reset", 23_196)),
    Sequence("pr56615-c", (41_300, "reset", 41_300)),
    Sequence("pr56615-aligned", (7_297, "reset", 7_297)),
    # Lookup-side EAGLE proof must not drop two units for an exact resend.
    Sequence("pr57180-local", (7_040, 7_040)),
    Sequence("agentx-84224", (83_992, 84_154, 84_380, 84_455, "reset", 84_455), 6),
    Sequence("agentx-92288", (91_679, 92_033, 92_451, 92_526, "reset", 92_526), 6),
    Sequence("multiple-resets", (7_449, "reset", 7_449, "reset", 7_705)),
    Sequence("repeat-after-reset", (7_449, "reset", 7_449, 7_449)),
    Sequence("rewind-and-extend", (7_449, 7_040, 7_449, 7_705, "reset", 7_705)),
)

BOUNDARY_SEQUENCES = tuple(
    Sequence(
        f"boundary-{length}", (length, length, length + 256, "reset", length + 256)
    )
    for boundary in (128, 896, 7_168, 8_192)
    for length in (boundary - 1, boundary, boundary + 1)
)


@pytest.mark.parametrize(
    "sequence", TEST_SEQUENCES + BOUNDARY_SEQUENCES, ids=lambda sequence: sequence.name
)
@pytest.mark.parametrize("cache_backend", ["gpu", "mooncake"])
def test_hybrid_mooncake_replay(sequence: Sequence, cache_backend: str):
    results = Replay(NUM_BLOCKS, use_mooncake=cache_backend == "mooncake").run_sequence(
        sequence
    )
    local_history = []
    store_history = []
    unit = REPLAY_SETUP.pmu

    def minimum_hit(history, length, suffix):
        candidates = [0]
        for previous_length, previous_suffix in history:
            checkpoint = max(0, (previous_length // unit - 1) * unit)
            proof_end = checkpoint + unit
            common_prefix = min(previous_length - previous_suffix, length - suffix)
            if proof_end <= common_prefix and checkpoint < length:
                candidates.append(checkpoint)
        return max(candidates)

    for turn, result in enumerate(results):
        if result["operation"] == "reset":
            local_history.clear()
            continue
        length = result["length"]
        suffix = result["branch_suffix_tokens"]
        local_expected = minimum_hit(local_history, length, suffix)
        store_expected = minimum_hit(store_history, length, suffix)
        observations = (
            ("GPU lookup", result["initial_gpu_local_hit"], local_expected),
            ("connector bookkeeping", result["initial_connector_hit"], local_expected),
            ("Mooncake lookup/store", result["initial_mooncake_hit"], store_expected),
        )
        for path, actual, expected in (
            observations[:2] if cache_backend == "gpu" else observations
        ):
            assert expected <= actual < length and actual % unit == 0, (
                f"{sequence.name}, action {turn}, {path}: "
                f"expected hit in [{expected}, {length - 1}], actual={actual}; "
                f"PMU={unit}, steps={sequence.steps}"
            )
        local_history.append((length, suffix))
        store_history.append((length, suffix))
