from __future__ import annotations

import json
from pathlib import Path

import pytest

import llm_routing as routing
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
