from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

AGENT_CLIENTS = {"codex", "claude", "opencode"}
SINGLE_PROMPT_CLIENTS = {"genai", "openai", "openrouter", "codex", "claude"}
EMBEDDING_CLIENTS = {"genai", "openai", "openrouter"}

AGENT_REVIEW_ROLE = "agents.review"
AGENT_TAXONOMY_TAIL_ROLE = "agents.taxonomy_tail"

TIER2_PRIMARY_ROLE = "single_prompt.tier2_primary"
TIER2_SECONDARY_ROLE = "single_prompt.tier2_secondary"
TAXONOMY_PRIMARY_ROLE = "single_prompt.taxonomy_primary"
TAXONOMY_SECONDARY_ROLE = "single_prompt.taxonomy_secondary"
ANSWER_ROLE = "single_prompt.answer"
COMMUNITY_REPORT_ROLE = "single_prompt.community_report"
GLOBAL_MAP_ROLE = "single_prompt.global_map"
GLOBAL_REDUCE_ROLE = "single_prompt.global_reduce"
DRIFT_PLANNER_ROLE = "single_prompt.drift_planner"
PROMPT_TUNING_ROLE = "single_prompt.prompt_tuning"
CLAIM_EXTRACTION_ROLE = "single_prompt.claim_extraction"
GRAPH_EXTRACTION_ROLE = "single_prompt.graph_extraction"
EVALUATION_JUDGE_ROLE = "single_prompt.evaluation_judge"
EVALUATION_QUESTION_ROLE = "single_prompt.evaluation_question_generation"
TIER3_JUDGE_PRIMARY_ROLE = "single_prompt.tier3_judge_primary"
TIER3_JUDGE_SECONDARY_ROLE = "single_prompt.tier3_judge_secondary"

TIER3_EMBEDDING_ROLE = "embeddings.tier3"
GRAPH_BUILD_EMBEDDING_ROLE = "embeddings.graph_build"


@dataclass(frozen=True)
class AgentRoleConfig:
    client: str
    model: str | None = None
    executable: str | None = None
    args: tuple[str, ...] = ()
    env: dict[str, str] | None = None


@dataclass(frozen=True)
class PromptRoleConfig:
    client: str
    model: str
    reasoning_effort: str | None = None


@dataclass(frozen=True)
class EmbeddingRoleConfig:
    client: str
    model: str
    dimension: int | None = None


def default_agent_executable(client: str) -> str:
    return {
        "codex": "codex",
        "claude": "claude",
        "opencode": "opencode",
    }[client]


def api_key_env_var_for_client(client: str) -> str | None:
    return {
        "genai": "GOOGLE_API_KEY",
        "openai": "OPENAI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }.get(client)


@lru_cache(maxsize=16)
def _load_config(path: str) -> dict[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("LLM routing config must be a JSON object.")
    return payload


def load_routing_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    return _load_config(str(Path(path).resolve()))


def _lookup_role(config: dict[str, Any], role: str) -> dict[str, Any] | None:
    current: Any = config
    for part in role.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    if current is None:
        return None
    if not isinstance(current, dict):
        raise ValueError(f"Role '{role}' must be configured as a JSON object.")
    return current


def _normalize_client(value: Any, *, allowed: set[str], role: str) -> str:
    client = str(value or "").strip().lower()
    if client not in allowed:
        raise ValueError(f"Role '{role}' has unsupported client '{value}'.")
    return client


def resolve_agent_role(
    config_path: str | None,
    role: str,
    *,
    default_client: str,
    default_model: str | None = None,
    default_executable: str | None = None,
) -> AgentRoleConfig:
    config = load_routing_config(config_path)
    raw = _lookup_role(config, role) or {}
    client = _normalize_client(raw.get("client", default_client), allowed=AGENT_CLIENTS, role=role)
    model = raw.get("model", default_model)
    if model is not None:
        model = str(model).strip() or None
    executable = raw.get("executable", default_executable or default_agent_executable(client))
    executable = str(executable).strip() if executable is not None else None
    raw_args = raw.get("args") or []
    if not isinstance(raw_args, list) or not all(isinstance(item, (str, int, float)) for item in raw_args):
        raise ValueError(f"Role '{role}' args must be a JSON array of scalars.")
    args = tuple(str(item).strip() for item in raw_args if str(item).strip())
    raw_env = raw.get("env") or {}
    if not isinstance(raw_env, dict) or not all(isinstance(key, str) for key in raw_env):
        raise ValueError(f"Role '{role}' env must be a JSON object.")
    env = {key: str(value) for key, value in raw_env.items()}
    return AgentRoleConfig(
        client=client,
        model=model,
        executable=executable or None,
        args=args,
        env=env or None,
    )


def resolve_prompt_role(
    config_path: str | None,
    role: str,
    *,
    default_client: str,
    default_model: str,
    default_reasoning_effort: str | None = None,
) -> PromptRoleConfig:
    config = load_routing_config(config_path)
    raw = _lookup_role(config, role) or {}
    client = _normalize_client(raw.get("client", default_client), allowed=SINGLE_PROMPT_CLIENTS, role=role)
    model = str(raw.get("model", default_model)).strip()
    if not model:
        raise ValueError(f"Role '{role}' must define a non-empty model.")
    inherited_effort = default_reasoning_effort if client == default_client else None
    reasoning_effort_raw = raw.get("reasoning_effort", inherited_effort)
    reasoning_effort = str(reasoning_effort_raw).strip().lower() if reasoning_effort_raw is not None else None
    if reasoning_effort not in {None, "low", "medium", "high", "xhigh"}:
        raise ValueError(f"Role '{role}' has invalid reasoning_effort '{reasoning_effort_raw}'.")
    if reasoning_effort and client not in {"codex", "claude"}:
        raise ValueError(f"Role '{role}' can only set reasoning_effort for a subscription CLI client.")
    if client == "claude" and reasoning_effort == "xhigh":
        raise ValueError(f"Role '{role}' cannot use xhigh reasoning_effort with Claude CLI.")
    return PromptRoleConfig(client=client, model=model, reasoning_effort=reasoning_effort)


def resolve_embedding_role(
    config_path: str | None,
    role: str,
    *,
    default_client: str,
    default_model: str,
    default_dimension: int | None = None,
) -> EmbeddingRoleConfig:
    config = load_routing_config(config_path)
    raw = _lookup_role(config, role) or {}
    client = _normalize_client(raw.get("client", default_client), allowed=EMBEDDING_CLIENTS, role=role)
    model = str(raw.get("model", default_model)).strip()
    if not model:
        raise ValueError(f"Role '{role}' must define a non-empty model.")
    dimension_raw = raw.get("dimension", default_dimension)
    dimension = None
    if dimension_raw is not None:
        try:
            dimension = int(dimension_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Role '{role}' has invalid dimension '{dimension_raw}'.") from exc
        if dimension <= 0:
            raise ValueError(f"Role '{role}' must define a positive dimension.")
    return EmbeddingRoleConfig(client=client, model=model, dimension=dimension)


def resolve_graph_build_embedding(
    *,
    config_path: str | None,
    embedding_provider: str | None,
    embedding_model: str | None,
    default_provider: str,
    default_model: str,
) -> EmbeddingRoleConfig:
    config = load_routing_config(config_path)
    raw = _lookup_role(config, GRAPH_BUILD_EMBEDDING_ROLE) or {}
    explicit_provider = str(embedding_provider or "").strip()
    explicit_model = str(embedding_model or "").strip()
    client = explicit_provider or str(raw.get("client", default_provider)).strip()
    model = explicit_model or str(raw.get("model", default_model)).strip()
    if not client:
        raise ValueError(f"Role '{GRAPH_BUILD_EMBEDDING_ROLE}' must define a non-empty client.")
    if not model:
        raise ValueError(f"Role '{GRAPH_BUILD_EMBEDDING_ROLE}' must define a non-empty model.")
    if explicit_provider or explicit_model:
        return EmbeddingRoleConfig(client=client, model=model, dimension=None)
    dimension_raw = raw.get("dimension")
    dimension = None
    if dimension_raw is not None:
        try:
            dimension = int(dimension_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Role '{GRAPH_BUILD_EMBEDDING_ROLE}' has invalid dimension '{dimension_raw}'.") from exc
        if dimension <= 0:
            raise ValueError(f"Role '{GRAPH_BUILD_EMBEDDING_ROLE}' must define a positive dimension.")
    return EmbeddingRoleConfig(
        client=client,
        model=model,
        dimension=dimension,
    )


def missing_required_env_vars(*clients: str) -> list[str]:
    missing: list[str] = []
    seen: set[str] = set()
    for client in clients:
        env_var = api_key_env_var_for_client(client)
        if not env_var or env_var in seen:
            continue
        seen.add(env_var)
        if not os.environ.get(env_var):
            missing.append(env_var)
    return missing
