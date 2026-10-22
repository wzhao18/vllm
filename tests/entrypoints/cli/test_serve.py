# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the ``vllm serve`` CLI subcommand."""

import argparse
from unittest.mock import MagicMock, patch

import pytest

from vllm.entrypoints.cli.serve import ServeSubcommand, run_headless


class _StopAfterEngineConfig(Exception):
    pass


def test_headless_imports_reasoning_parser_plugin_before_engine_config():
    plugin_path = "/tmp/custom_reasoning_parser.py"
    args = argparse.Namespace(
        api_server_count=0,
        reasoning_parser_plugin=plugin_path,
    )
    engine_args = MagicMock()

    with (
        patch(
            "vllm.entrypoints.cli.serve.vllm.AsyncEngineArgs.from_cli_args",
            return_value=engine_args,
        ),
        patch(
            "vllm.reasoning.ReasoningParserManager.import_reasoning_parser"
        ) as import_plugin,
    ):

        def create_engine_config(**kwargs):
            import_plugin.assert_called_once_with(plugin_path)
            raise _StopAfterEngineConfig

        engine_args.create_engine_config.side_effect = create_engine_config

        with pytest.raises(_StopAfterEngineConfig):
            run_headless(args)


def _serve_args(**overrides) -> argparse.Namespace:
    args = argparse.Namespace(
        model_tag=None,
        headless=False,
        api_server_count=None,
        data_parallel_size=4,
        data_parallel_size_local=None,
        data_parallel_rank=None,
        data_parallel_start_rank=None,
        data_parallel_external_lb=False,
        data_parallel_hybrid_lb=False,
        data_parallel_multi_port_external_lb=False,
        data_parallel_lb_policy="context_tokens",
    )
    for name, value in overrides.items():
        setattr(args, name, value)
    return args


@pytest.mark.parametrize("hybrid_lb", [False, True])
def test_context_tokens_lb_policy_defaults_to_one_api_server(monkeypatch, hybrid_lb):
    """The policy needs every request to pass through one API server, so it
    overrides the default of one API server per (local) DP rank."""
    args = _serve_args(
        data_parallel_hybrid_lb=hybrid_lb,
        data_parallel_size_local=2 if hybrid_lb else None,
        data_parallel_start_rank=0 if hybrid_lb else None,
    )
    monkeypatch.delenv("VLLM_USE_RUST_FRONTEND", raising=False)
    with (
        patch("vllm.entrypoints.cli.serve.run_multi_api_server") as multi_server,
        patch("vllm.entrypoints.cli.serve.uvloop.run") as single_server,
        patch("vllm.entrypoints.cli.serve.run_server", MagicMock()),
    ):
        ServeSubcommand.cmd(args)

    multi_server.assert_not_called()
    single_server.assert_called_once()


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"data_parallel_external_lb": True}, "external DP load balancing"),
        ({"data_parallel_rank": 0}, "external DP load balancing"),
    ],
)
def test_context_tokens_lb_policy_rejects_external_lb(monkeypatch, overrides, match):
    monkeypatch.delenv("VLLM_USE_RUST_FRONTEND", raising=False)
    with pytest.raises(ValueError, match=match):
        ServeSubcommand.cmd(_serve_args(**overrides))


def test_context_tokens_lb_policy_rejects_rust_frontend(monkeypatch):
    """The Rust frontend routes between DP ranks itself."""
    monkeypatch.setenv("VLLM_USE_RUST_FRONTEND", "1")
    monkeypatch.setenv("VLLM_RUST_FRONTEND_PATH", "/bin/true")
    with pytest.raises(ValueError, match="Rust frontend"):
        ServeSubcommand.cmd(_serve_args())
