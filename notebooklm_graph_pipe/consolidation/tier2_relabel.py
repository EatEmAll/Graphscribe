"""
Tier 2 - LLM Label Normalization
================================
Reclassifies nodes labelled only as `Concept` into more specific types
(Metric, Strategy, Algorithm, Method, Model, etc.) using Gemini.

Usage:
    python scripts/consolidation/consolidate_tier2_relabel.py [--dry-run] [--batch-size 50]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from google import genai
from neo4j import GraphDatabase

from notebooklm_graph_pipe.paths import CONSOLIDATION_CACHE_DIR
from notebooklm_graph_pipe.runtime.graph_text_utils import coerce_text, sorted_unique_texts
from notebooklm_graph_pipe.runtime.llm_json_utils import (
    JsonDiskCache,
    build_single_prompt_clients,
    generate_json_payload,
    is_transient_model_error,
    make_cache_key,
)
from notebooklm_graph_pipe.runtime.llm_routing import TIER2_PRIMARY_ROLE, TIER2_SECONDARY_ROLE, PromptRoleConfig, resolve_prompt_role

DEFAULT_NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
DEFAULT_NEO4J_USER = os.environ.get("NEO4J_USERNAME", "neo4j")
DEFAULT_NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password123")
DEFAULT_NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")
PRIMARY_CLIENT = os.environ.get("TIER2_PRIMARY_CLIENT", "openrouter")
MODEL_NAME = os.environ.get("TIER2_MODEL_NAME", "minimax/minimax-m3")
SECOND_STAGE_CLIENT = os.environ.get("TIER2_SECOND_STAGE_CLIENT", "codex")
SECOND_STAGE_MODEL_NAME = os.environ.get("TIER2_SECOND_STAGE_MODEL_NAME", "gpt-5.6-luna")
SECOND_STAGE_REASONING_EFFORT = os.environ.get("TIER2_SECOND_STAGE_REASONING_EFFORT", "low")
LOW_CONFIDENCE_THRESHOLD = float(os.environ.get("TIER2_LOW_CONFIDENCE_THRESHOLD", "0.65"))
MAX_RETRIES = int(os.environ.get("TIER2_MAX_RETRIES", "3"))
INITIAL_RETRY_DELAY_SECONDS = float(os.environ.get("TIER2_INITIAL_RETRY_DELAY_SECONDS", "1.0"))
DEFAULT_CACHE_FILE = os.environ.get(
    "TIER2_CACHE_FILE",
    str(CONSOLIDATION_CACHE_DIR / "tier2_classification_cache.json"),
)
DEFAULT_LABELS = [
    "Concept",
    "Metric",
    "Strategy",
    "Algorithm",
    "Method",
    "Model",
    "Technology",
    "Process",
    "Field",
    "Event",
    "Asset",
    "Variable",
    "Risk",
    "Signal",
    "Rule",
    "Indicator",
    "Financial Metric",
    "Trading Concept",
    "Market Feature",
    "Math",
    "Data Structure",
    "Trading System",
    "Account",
    "Condition",
]
DEFAULT_LABEL_CATALOG = {
    "labels": list(DEFAULT_LABELS),
    "preferred_examples": {
        "Financial Metric": "Measurable financial quantities such as P&L, Sharpe ratio, drawdown, fee, or return.",
        "Trading Concept": "Trading ideas such as liquidity, leverage, slippage, execution edge, or market regime.",
        "Market Feature": "Market microstructure or price-action features such as order book depth, spread, or volume profile.",
        "Data Structure": "Programming or mathematical data structures such as tensor, queue, tree, or array.",
        "Trading System": "Named end-to-end trading systems, playbooks, or execution frameworks.",
    },
    "fallback_label": "Concept",
}

DEFAULT_LABEL_ALIASES = {
    "financial": "Financial Metric",
    "finance": "Financial Metric",
    "financial metric": "Financial Metric",
    "financial metrics": "Financial Metric",
    "trading": "Trading Concept",
    "trading concept": "Trading Concept",
    "trading concepts": "Trading Concept",
    "market concept": "Trading Concept",
    "market concepts": "Trading Concept",
    "state": "Condition",
    "state like": "Condition",
    "state-like": "Condition",
    "condition": "Condition",
    "conditions": "Condition",
    "workflow": "Process",
    "process": "Process",
    "method": "Method",
    "methods": "Method",
    "mathematical": "Math",
    "mathematics": "Math",
    "math": "Math",
    "structure": "Data Structure",
    "data structure": "Data Structure",
    "data structures": "Data Structure",
}


def load_label_catalog(labels_json: str | None) -> dict[str, Any]:
    if labels_json:
        payload = json.loads(Path(labels_json).read_text(encoding="utf-8"))
    else:
        payload = DEFAULT_LABEL_CATALOG

    labels = [str(label).strip() for label in payload.get("labels", []) if str(label).strip()]
    if not labels:
        raise RuntimeError("Tier 2 label catalog must contain at least one label.")
    if "Concept" not in labels:
        raise RuntimeError("Tier 2 label catalog must include 'Concept'.")
    if any("`" in label for label in labels):
        raise RuntimeError("Tier 2 label catalog labels cannot contain backticks.")
    if len({label.casefold() for label in labels}) != len(labels):
        raise RuntimeError("Tier 2 label catalog labels must be unique.")

    preferred_examples = {
        str(label).strip(): str(text).strip()
        for label, text in dict(payload.get("preferred_examples", {})).items()
        if str(label).strip() in labels and str(text).strip()
    }
    return {
        "labels": labels,
        "preferred_examples": preferred_examples,
        "fallback_label": "Concept",
    }


def build_system_prompt(label_catalog: dict[str, Any]) -> str:
    labels = label_catalog["labels"]
    examples = label_catalog.get("preferred_examples", {})
    example_lines = "\n".join(
        f"- {label}: {examples.get(label, 'Use only when it is the most specific available fit.')}"
        for label in labels
    )
    examples_block = f"\nLabel guidance:\n{example_lines}\n"
    return (
        "You are a knowledge graph schema expert in quantitative finance and algorithmic trading.\n\n"
        "Given an entity and local graph context, assign the SINGLE most specific label from:\n"
        f"{labels}\n"
        f"{examples_block}"
        "Rules:\n"
        '- Return strict JSON only: {"label":"...", "confidence":0.0, "reason":"..."}.\n'
        '- Use "Concept" only as the last-resort fallback when no existing label fits clearly.\n'
        "- Prefer the most specific label available in the provided catalog.\n"
        "- Use graph neighborhood signals when they disambiguate the entity.\n"
        "- Do not invent new labels.\n"
    )


def _normalize_alias_key(value: str) -> str:
    lowered = "".join(ch.lower() if ch.isalnum() else " " for ch in value)
    return " ".join(lowered.split())


def build_label_alias_map(label_catalog: dict[str, Any]) -> dict[str, str]:
    alias_map: dict[str, str] = {}
    for label in label_catalog["labels"]:
        alias_map[_normalize_alias_key(label)] = label
    for alias, label in DEFAULT_LABEL_ALIASES.items():
        if label in label_catalog["labels"]:
            alias_map[_normalize_alias_key(alias)] = label
    return alias_map


def normalize_label(raw_label: str, label_catalog: dict[str, Any]) -> str | None:
    alias_map = build_label_alias_map(label_catalog)
    normalized = _normalize_alias_key(raw_label.strip().strip('"').strip("'"))
    if not normalized:
        return None
    mapped = alias_map.get(normalized)
    if mapped:
        return mapped
    for alias, label in alias_map.items():
        if normalized == alias or normalized.startswith(alias) or alias.startswith(normalized):
            return label
    return None


def _is_transient_error(exc: Exception) -> bool:
    return is_transient_model_error(exc)


def _normalize_node_for_prompt(node: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": _coerce_text(node.get("name")),
        "description": _coerce_text(node.get("description")) or "No description.",
        "labels": sorted_unique_texts(node.get("labels", [])) or ["Concept"],
        "degree": int(node.get("degree", 0) or 0),
        "neighbor_labels": sorted_unique_texts(node.get("neighbor_labels", [])),
        "relation_types": sorted_unique_texts(node.get("relation_types", [])),
        "neighbor_ids": sorted_unique_texts(node.get("neighbor_ids", [])),
    }


def _build_prompt(node: dict[str, Any]) -> str:
    return "\n".join(
        [
            f'Entity: "{node["name"]}"',
            f'Description: "{node["description"]}"',
            f'Current labels: {", ".join(node["labels"])}',
            f'Degree: {node["degree"]}',
            f'Neighbor labels: {", ".join(node["neighbor_labels"]) if node["neighbor_labels"] else "None"}',
            f'Relation types: {", ".join(node["relation_types"]) if node["relation_types"] else "None"}',
            f'Neighbor ids: {", ".join(node["neighbor_ids"]) if node["neighbor_ids"] else "None"}',
            'Return JSON: {"label":"...", "confidence":0.0, "reason":"..."}',
        ]
    )


def _build_classification_cache_key(
    *,
    client_name: str,
    model_name: str,
    reasoning_effort: str | None,
    normalized_node: dict[str, Any],
    system_instruction: str,
) -> str:
    return make_cache_key(
        namespace="tier2_classification_v1",
        payload={
            "client_name": client_name,
            "model_name": model_name,
            "reasoning_effort": reasoning_effort,
            "system_instruction": system_instruction,
            "node": normalized_node,
            "temperature": 0.0,
            "max_output_tokens": 120,
        },
    )


def _classify_once(
    clients: dict[str, Any] | Any,
    *,
    role_config: PromptRoleConfig | None = None,
    model_name: str | None = None,
    node: dict[str, Any],
    label_catalog: dict[str, Any],
    stage: str = "primary",
    cache: JsonDiskCache | None = None,
) -> dict[str, Any]:
    if role_config is None:
        if not model_name:
            raise ValueError("role_config or model_name is required.")
        role_config = PromptRoleConfig(client="genai", model=model_name)
    normalized_node = _normalize_node_for_prompt(node)
    system_instruction = build_system_prompt(label_catalog)
    cache_key = _build_classification_cache_key(
        client_name=role_config.client,
        model_name=role_config.model,
        reasoning_effort=role_config.reasoning_effort,
        normalized_node=normalized_node,
        system_instruction=system_instruction,
    )
    if cache is not None:
        cached_result = cache.get(cache_key)
        if isinstance(cached_result, dict):
            return dict(cached_result)

    payload, error_message = generate_json_payload(
        _resolve_client(clients, role_config.client),
        client_name=role_config.client,
        model_name=role_config.model,
        reasoning_effort=role_config.reasoning_effort,
        prompt=_build_prompt(normalized_node),
        system_instruction=system_instruction,
        max_output_tokens=120,
        temperature=0.0,
        max_attempts=1,
    )
    if payload is None:
        return {
            "status": "unresolved",
            "label": None,
            "confidence": 0.0,
            "reason": error_message or "Invalid or empty JSON response",
            "client_name": role_config.client,
            "model_name": role_config.model,
            "stage": stage,
        }

    raw_label = str(payload.get("label", "")).strip()
    mapped_label = normalize_label(raw_label, label_catalog)
    confidence = payload.get("confidence", 0.0)
    try:
        confidence_value = max(0.0, min(float(confidence), 1.0))
    except (TypeError, ValueError):
        confidence_value = 0.0

    result = {
        "status": "classified" if mapped_label else "unresolved",
        "label": mapped_label,
        "raw_label": raw_label,
        "confidence": confidence_value,
        "reason": str(payload.get("reason", "")).strip(),
        "client_name": role_config.client,
        "model_name": role_config.model,
        "stage": stage,
    }
    if cache is not None and result["status"] == "classified":
        cache.set(cache_key, result)
    return result


def classify_entity(
    clients: dict[str, Any] | Any,
    node: dict[str, Any],
    label_catalog: dict[str, Any],
    primary_role_config: PromptRoleConfig | None = None,
    secondary_role_config: PromptRoleConfig | None = None,
    cache: JsonDiskCache | None = None,
) -> dict[str, Any]:
    primary_role = primary_role_config or resolve_prompt_role(
        None,
        TIER2_PRIMARY_ROLE,
        default_client="genai",
        default_model=MODEL_NAME,
    )
    secondary_role = secondary_role_config or resolve_prompt_role(
        None,
        TIER2_SECONDARY_ROLE,
        default_client="genai",
        default_model=SECOND_STAGE_MODEL_NAME,
    )
    delay = INITIAL_RETRY_DELAY_SECONDS
    last_error: str | None = None
    for attempt in range(MAX_RETRIES):
        primary = _classify_once(
            clients,
            role_config=primary_role,
            node=node,
            label_catalog=label_catalog,
            stage="primary",
            cache=cache,
        )
        if primary["status"] == "classified" and primary["confidence"] >= LOW_CONFIDENCE_THRESHOLD:
            return {
                **primary,
                "used_second_stage": False,
                "attempted_second_stage": False,
            }

        if primary["status"] == "unresolved":
            last_error = primary["reason"]
            if _is_transient_error(RuntimeError(primary["reason"])):
                if attempt < MAX_RETRIES - 1:
                    time.sleep(delay)
                    delay *= 2
                    continue
                break
            else:
                break

        secondary = _classify_once(
            clients,
            role_config=secondary_role,
            node=node,
            label_catalog=label_catalog,
            stage="second_stage",
            cache=cache,
        )
        if secondary["status"] == "classified":
            return {
                **secondary,
                "used_second_stage": True,
                "attempted_second_stage": True,
            }
        if primary["status"] == "classified":
            return {
                **primary,
                "used_second_stage": False,
                "attempted_second_stage": True,
            }

        last_error = secondary["reason"] or primary["reason"] or "Low confidence classification"
        if attempt == MAX_RETRIES - 1 or not _is_transient_error(RuntimeError(last_error)):
            break
        time.sleep(delay)
        delay *= 2

    message = last_error if last_error else "Unknown classification failure"
    return {
        "status": "unresolved",
        "label": None,
        "confidence": 0.0,
        "reason": message,
        "client_name": primary_role.client,
        "model_name": primary_role.model,
        "stage": "primary",
        "used_second_stage": False,
        "attempted_second_stage": False,
    }


def _resolve_client(clients: dict[str, Any] | Any, client_name: str) -> Any:
    if isinstance(clients, dict):
        return clients[client_name]
    return clients


def fetch_concept_only_nodes(
    session,
    max_nodes: int | None,
    scope_revision_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    query = """
        MATCH (n:__Entity__:Concept)
        WHERE NOT n:CorpusSource
          AND NOT EXISTS { (n)-[:HAS_SOURCE|MATERIALIZED_AS|LEGACY_EVIDENCE]-() }
          AND ALL(l IN labels(n) WHERE l IN ['__Entity__', 'Concept'])
          AND ($scope_revision_ids IS NULL OR EXISTS {
              MATCH (n)<-[:HAS_ENTITY]-(:ParentChunk)<-[:HAS_PARENT]-(revision:DocumentRevision)
              WHERE revision.id IN $scope_revision_ids
          })
        OPTIONAL MATCH (n)-[r]-(m)
        OPTIONAL MATCH (n)-[tax:SUBCLASS_OF|INSTANCE_OF|TYPE_OF|IS_A]->()
        WITH n,
             count(DISTINCT r) AS degree,
             collect(DISTINCT type(r))[0..6] AS relation_types,
             collect(DISTINCT coalesce(toString(m.id), elementId(m)))[0..6] AS neighbor_ids,
             collect(DISTINCT [l IN labels(m) WHERE l <> '__Entity__'])[0..6] AS neighbor_label_sets,
             count(DISTINCT tax) AS outgoing_taxonomy_count
        RETURN
            elementId(n) AS eid,
            n.id AS name,
            n.description AS description,
            [l IN labels(n) WHERE l <> '__Entity__'] AS labels,
            degree,
            relation_types,
            neighbor_ids,
            neighbor_label_sets,
            outgoing_taxonomy_count
        ORDER BY toLower(toString(n.id)) ASC
    """
    if max_nodes is not None:
        query += "\nLIMIT $max_nodes"
        result = session.run(query, max_nodes=max_nodes, scope_revision_ids=scope_revision_ids)
    else:
        result = session.run(query, scope_revision_ids=scope_revision_ids)
    return [
        {
            "eid": row["eid"],
            "name": row["name"],
            "description": row["description"],
            "labels": [label for label in row["labels"] if label],
            "degree": int(row["degree"] or 0),
            "relation_types": [value for value in row["relation_types"] if value],
            "neighbor_ids": [value for value in row["neighbor_ids"] if value],
            "neighbor_labels": [
                label
                for labels in row["neighbor_label_sets"]
                if labels
                for label in labels
                if label and label != "__Entity__"
            ][:6],
            "outgoing_taxonomy_count": int(row["outgoing_taxonomy_count"] or 0),
        }
        for row in result
    ]


def apply_label(session, eid: str, new_label: str) -> None:
    safe_label = new_label.replace("`", "")
    session.run(
        f"""
        MATCH (n:__Entity__)
        WHERE elementId(n) = $eid AND NOT n:CorpusSource
          AND NOT EXISTS {{ (n)-[:HAS_SOURCE|MATERIALIZED_AS|LEGACY_EVIDENCE]-() }}
        SET n:`{safe_label}`
        """,
        eid=eid,
    )


_coerce_text = coerce_text


def _write_summary(summary_json: str | None, summary: dict[str, Any]) -> None:
    if not summary_json:
        return
    path = Path(summary_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _append_decision_jsonl(path: Path | None, record: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def run(
    *,
    dry_run: bool,
    batch_size: int,
    sleep_seconds: float,
    max_nodes: int | None,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    neo4j_database: str,
    labels_json: str | None,
    decisions_jsonl: str | None,
    cache_file: str = DEFAULT_CACHE_FILE,
    summary_json: str | None = None,
    llm_routing_config: str | None = None,
    scope_revision_ids: list[str] | None = None,
) -> dict[str, Any]:
    primary_role_config = resolve_prompt_role(
        llm_routing_config,
        TIER2_PRIMARY_ROLE,
        default_client=PRIMARY_CLIENT,
        default_model=MODEL_NAME,
    )
    secondary_role_config = resolve_prompt_role(
        llm_routing_config,
        TIER2_SECONDARY_ROLE,
        default_client=SECOND_STAGE_CLIENT,
        default_model=SECOND_STAGE_MODEL_NAME,
        default_reasoning_effort=SECOND_STAGE_REASONING_EFFORT,
    )
    clients = build_single_prompt_clients(primary_role_config.client, secondary_role_config.client)
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    label_catalog = load_label_catalog(labels_json)
    classification_cache = JsonDiskCache(cache_file)
    decisions_path = Path(decisions_jsonl) if decisions_jsonl else None
    if decisions_path is not None and decisions_path.exists():
        decisions_path.unlink()
    summary: dict[str, Any]
    started = time.time()

    try:
        with driver.session(database=neo4j_database) as session:
            print("Fetching Concept-only nodes...")
            nodes = (
                fetch_concept_only_nodes(
                    session,
                    max_nodes=max_nodes,
                    scope_revision_ids=scope_revision_ids,
                )
                if scope_revision_ids is not None
                else fetch_concept_only_nodes(session, max_nodes=max_nodes)
            )
            print(f"  Found {len(nodes)} Concept-only nodes")

            reclassified = 0
            kept_concept = 0
            unresolved = 0
            low_confidence_relabels = 0
            second_stage_relabels = 0
            relabeled_without_taxonomy_support = 0
            label_counts: dict[str, int] = {}

            for i, node in enumerate(nodes):
                name = _coerce_text(node["name"])
                node["name"] = name
                node["description"] = _coerce_text(node.get("description"))
                result = classify_entity(
                    clients,
                    node,
                    label_catalog,
                    primary_role_config=primary_role_config,
                    secondary_role_config=secondary_role_config,
                    cache=classification_cache,
                )
                new_label = result["label"] or "Concept"
                if result["status"] == "unresolved":
                    label_counts["Unresolved"] = label_counts.get("Unresolved", 0) + 1
                    marker = "?"
                    print(
                        f"  [{i + 1}/{len(nodes)}] {marker} '{name}' -> unresolved "
                        f"({result['model_name']}: {result['reason']})"
                    )
                    unresolved += 1
                else:
                    label_counts[new_label] = label_counts.get(new_label, 0) + 1
                    marker = "->" if new_label != "Concept" else "="
                    confidence = f"{result['confidence']:.2f}"
                    print(
                        f"  [{i + 1}/{len(nodes)}] {marker} '{name}' -> {new_label} "
                        f"(confidence={confidence}, model={result['model_name']})"
                    )

                if result["status"] != "unresolved" and new_label != "Concept":
                    if not dry_run:
                        apply_label(session, node["eid"], new_label)
                    reclassified += 1
                    if result["confidence"] < LOW_CONFIDENCE_THRESHOLD:
                        low_confidence_relabels += 1
                    if result.get("used_second_stage"):
                        second_stage_relabels += 1
                    if int(node.get("outgoing_taxonomy_count", 0)) == 0:
                        relabeled_without_taxonomy_support += 1
                elif result["status"] != "unresolved":
                    kept_concept += 1

                _append_decision_jsonl(
                    decisions_path,
                    {
                        "eid": node["eid"],
                        "name": name,
                        "old_label": "Concept",
                        "new_label": new_label if result["status"] != "unresolved" else None,
                        "status": result["status"],
                        "confidence": round(float(result["confidence"]), 4),
                        "reason": result["reason"],
                        "client_name": result.get("client_name", primary_role_config.client),
                        "model_name": result["model_name"],
                        "stage": result.get("stage", "primary"),
                        "used_second_stage": bool(result.get("used_second_stage", False)),
                        "attempted_second_stage": bool(result.get("attempted_second_stage", False)),
                        "unresolved": result["status"] == "unresolved",
                        "labels": list(node.get("labels", [])),
                        "degree": int(node.get("degree", 0)),
                        "relation_types": list(node.get("relation_types", [])),
                        "neighbor_ids": list(node.get("neighbor_ids", [])),
                        "neighbor_labels": list(node.get("neighbor_labels", [])),
                        "outgoing_taxonomy_count": int(node.get("outgoing_taxonomy_count", 0)),
                    },
                )

                if (i + 1) % batch_size == 0 and sleep_seconds > 0:
                    time.sleep(sleep_seconds)

            sorted_counts = dict(
                sorted(label_counts.items(), key=lambda item: (-item[1], item[0]))
            )
            summary = {
                "tier": 2,
                "dry_run": dry_run,
                "client_name": primary_role_config.client,
                "model_name": primary_role_config.model,
                "second_stage_client_name": secondary_role_config.client,
                "second_stage_model_name": secondary_role_config.model,
                "params": {
                    "batch_size": batch_size,
                    "sleep_seconds": sleep_seconds,
                    "max_nodes": max_nodes,
                    "neo4j_uri": neo4j_uri,
                    "neo4j_database": neo4j_database,
                    "labels_json": labels_json,
                    "decisions_jsonl": decisions_jsonl,
                    "cache_file": cache_file,
                },
                "label_catalog": label_catalog,
                "processed_nodes": len(nodes),
                "reclassified": reclassified,
                "kept_concept": kept_concept,
                "unresolved": unresolved,
                "low_confidence_relabels": low_confidence_relabels,
                "second_stage_relabels": second_stage_relabels,
                "relabeled_without_taxonomy_support": relabeled_without_taxonomy_support,
                "label_distribution": sorted_counts,
                "cache_hits": classification_cache.hits,
                "cache_entries": len(classification_cache.records),
                "duration_seconds": round(time.time() - started, 3),
            }
    finally:
        classification_cache.save()
        driver.close()

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Summary:")
    print(f"  Reclassified: {summary['reclassified']}")
    print(f"  Kept as Concept: {summary['kept_concept']}")
    print(f"  Unresolved: {summary['unresolved']}")
    print(f"  Cache hits: {summary['cache_hits']}")
    print("\n  Label distribution:")
    for label, count in summary["label_distribution"].items():
        print(f"    {label}: {count}")
    if dry_run:
        print("\n[DRY RUN] No changes written.")
    else:
        print("\nTier 2 complete.")

    _write_summary(summary_json=summary_json, summary=summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Tier 2: LLM label normalization")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=1.0,
        help="Sleep interval between processed batches.",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=None,
        help="Max number of Concept-only nodes to process in this pass.",
    )
    parser.add_argument("--summary-json", type=str, default=None)
    parser.add_argument("--labels-json", type=str, default=None)
    parser.add_argument("--decisions-jsonl", type=str, default=None)
    parser.add_argument("--cache-file", type=str, default=DEFAULT_CACHE_FILE)
    parser.add_argument("--neo4j-uri", type=str, default=DEFAULT_NEO4J_URI)
    parser.add_argument("--neo4j-user", type=str, default=DEFAULT_NEO4J_USER)
    parser.add_argument("--neo4j-password", type=str, default=DEFAULT_NEO4J_PASSWORD)
    parser.add_argument("--neo4j-database", type=str, default=DEFAULT_NEO4J_DATABASE)
    parser.add_argument("--llm-routing-config", type=str, default=None)
    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0")
    if args.sleep_seconds < 0:
        parser.error("--sleep-seconds must be >= 0")
    if args.max_nodes is not None and args.max_nodes <= 0:
        parser.error("--max-nodes must be > 0")

    run(
        dry_run=args.dry_run,
        batch_size=args.batch_size,
        sleep_seconds=args.sleep_seconds,
        max_nodes=args.max_nodes,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        neo4j_database=args.neo4j_database,
        labels_json=args.labels_json,
        decisions_jsonl=args.decisions_jsonl,
        cache_file=args.cache_file,
        summary_json=args.summary_json,
        llm_routing_config=args.llm_routing_config,
    )


if __name__ == "__main__":
    main()
