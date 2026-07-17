from __future__ import annotations

import json
from pathlib import Path

import pytest

from notebooklm_graph_pipe.paths import LOCAL_MODEL_DIR
import notebooklm_graph_pipe.runtime.llm_routing as routing
from src.shared import common_fn


def _write_config(tmp_path: Path, payload: dict) -> str:
    path = tmp_path / "llm-routing.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_resolve_graph_build_embedding_uses_role_config(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        {
            "embeddings": {
                "graph_build": {
                    "client": "openrouter",
                    "model": "text-embedding-3-large",
                    "dimension": 1024,
                }
            }
        },
    )

    resolved = routing.resolve_graph_build_embedding(
        config_path=config_path,
        embedding_provider=None,
        embedding_model=None,
        default_provider="sentence-transformer",
        default_model="all-MiniLM-L6-v2",
    )

    assert resolved.client == "openrouter"
    assert resolved.model == "text-embedding-3-large"
    assert resolved.dimension == 1024


def test_resolve_graph_build_embedding_merges_partial_cli_override_and_clears_dimension(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        {
            "embeddings": {
                "graph_build": {
                    "client": "openai",
                    "model": "text-embedding-3-small",
                    "dimension": 1536,
                }
            }
        },
    )

    resolved = routing.resolve_graph_build_embedding(
        config_path=config_path,
        embedding_provider="openrouter",
        embedding_model=None,
        default_provider="sentence-transformer",
        default_model="all-MiniLM-L6-v2",
    )

    assert resolved.client == "openrouter"
    assert resolved.model == "text-embedding-3-small"
    assert resolved.dimension is None


def test_resolve_graph_build_embedding_accepts_legacy_default_provider_without_config() -> None:
    resolved = routing.resolve_graph_build_embedding(
        config_path=None,
        embedding_provider=None,
        embedding_model=None,
        default_provider="sentence-transformer",
        default_model="all-MiniLM-L6-v2",
    )

    assert resolved.client == "sentence-transformer"
    assert resolved.model == "all-MiniLM-L6-v2"
    assert resolved.dimension is None


def test_resolve_graph_build_embedding_applies_cli_override_without_validating_legacy_default() -> None:
    resolved = routing.resolve_graph_build_embedding(
        config_path=None,
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        default_provider="sentence-transformer",
        default_model="all-MiniLM-L6-v2",
    )

    assert resolved.client == "openai"
    assert resolved.model == "text-embedding-3-small"
    assert resolved.dimension is None


def test_resolve_prompt_role_rejects_unknown_client(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        {
            "single_prompt": {
                "tier2_primary": {
                    "client": "anthropic",
                    "model": "bad",
                }
            }
        },
    )

    with pytest.raises(ValueError, match="unsupported client"):
        routing.resolve_prompt_role(
            config_path,
            routing.TIER2_PRIMARY_ROLE,
            default_client="genai",
            default_model="gemini-3.1-flash-lite-preview",
        )


def test_resolve_prompt_role_accepts_subscription_client_effort(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        {
            "single_prompt": {
                "tier3_judge_secondary": {
                    "client": "codex",
                    "model": "gpt-5.6-luna",
                    "reasoning_effort": "medium",
                }
            }
        },
    )

    resolved = routing.resolve_prompt_role(
        config_path,
        routing.TIER3_JUDGE_SECONDARY_ROLE,
        default_client="genai",
        default_model="gemini-3-flash-preview",
    )

    assert resolved == routing.PromptRoleConfig(
        client="codex",
        model="gpt-5.6-luna",
        reasoning_effort="medium",
    )


def test_resolve_prompt_role_rejects_effort_for_api_client(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        {
            "single_prompt": {
                "tier2_primary": {
                    "client": "openrouter",
                    "model": "minimax/minimax-m3",
                    "reasoning_effort": "low",
                }
            }
        },
    )

    with pytest.raises(ValueError, match="subscription CLI"):
        routing.resolve_prompt_role(
            config_path,
            routing.TIER2_PRIMARY_ROLE,
            default_client="genai",
            default_model="gemini-3.1-flash-lite-preview",
        )


def test_load_embedding_model_openrouter_honors_dimension_override(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeEmbeddings:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    monkeypatch.setattr(common_fn, "OpenAIEmbeddings", FakeEmbeddings)
    monkeypatch.setattr(common_fn, "_detect_embedding_dimension", lambda embeddings: 9999)

    embeddings, dimension = common_fn.load_embedding_model(
        "openrouter",
        "text-embedding-3-large",
        embedding_dimension_override=1024,
    )

    assert isinstance(embeddings, FakeEmbeddings)
    assert captured["model"] == "text-embedding-3-large"
    assert captured["dimensions"] == 1024
    assert captured["base_url"] == "https://openrouter.ai/api/v1"
    assert dimension == 1024


def test_sentence_transformer_model_path_is_repo_anchored(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        common_fn,
        "_ensure_sentence_transformer_model_downloaded",
        lambda model_name, model_path: captured.update({"model_name": model_name, "model_path": model_path}),
    )
    monkeypatch.setattr(common_fn, "HuggingFaceEmbeddings", lambda model_name: {"model_name": model_name})
    common_fn._embedding_instances.clear()
    common_fn._embedding_locks.clear()

    embeddings = common_fn._get_sentence_transformer_embedding("all-MiniLM-L6-v2")

    assert embeddings == {"model_name": str(LOCAL_MODEL_DIR)}
    assert captured["model_path"] == str(LOCAL_MODEL_DIR)


def test_resolve_agent_role_accepts_args_and_env(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        {
            "agents": {
                "review": {
                    "client": "claude",
                    "model": "claude-haiku-4-5",
                    "executable": "claude",
                    "args": ["--mcp-config", "C:\\temp\\mcp.json", "--strict-mcp-config"],
                    "env": {"XDG_CONFIG_HOME": "C:\\temp\\xdg"},
                }
            }
        },
    )

    resolved = routing.resolve_agent_role(
        config_path,
        routing.AGENT_REVIEW_ROLE,
        default_client="codex",
        default_model=None,
        default_executable="codex",
    )

    assert resolved.client == "claude"
    assert resolved.model == "claude-haiku-4-5"
    assert resolved.executable == "claude"
    assert resolved.args == ("--mcp-config", "C:\\temp\\mcp.json", "--strict-mcp-config")
    assert resolved.env == {"XDG_CONFIG_HOME": "C:\\temp\\xdg"}
