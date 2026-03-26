"""
Self-improving node consolidation loop for Tier 1-3.

Workflow:
1. Run Tier 1, Tier 2, Tier 3 once.
2. Review quality via `codex exec` + Neo4j MCP.
3. If not consolidated enough, adapt Tier 2/3 params and the Tier 2 label catalog.
4. Repeat until stop criteria are met, taxonomy debt plateaus, or max iterations are reached.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase

from graph_text_utils import normalize_name, token_set
from llm_routing import (
    AGENT_REVIEW_ROLE,
    AGENT_TAXONOMY_TAIL_ROLE,
    AgentRoleConfig,
    resolve_agent_role,
)

REPO_ROOT = Path(__file__).resolve().parent

TIER2_BOUNDS = {
    "batch_size": (10, 200),
    "sleep_seconds": (0.0, 3.0),
    "max_nodes": (50, 5000),
}
TIER3_BOUNDS = {
    "threshold": (0.83, 0.95),
    "max_candidates": (100, 2000),
    "max_merges": (10, 400),
}
MAX_TIER2_LABELS = 40
VALID_DIAGNOSES = {"balanced", "duplicate_debt", "taxonomy_debt", "mixed_debt"}
CODEX_REVIEW_MAX_ATTEMPTS = int(os.environ.get("CODEX_REVIEW_MAX_ATTEMPTS", "3"))
AGENT_COMMAND_TIMEOUT_SECONDS = int(os.environ.get("AGENT_COMMAND_TIMEOUT_SECONDS", "300"))
WINDOWS_ARGV_PROMPT_SOFT_LIMIT = 7800
WITHOUT_TAXONOMY_PLATEAU_MIN_DELTA = 0.05
COMBINED_STOP_REASON = "combined_stop_gate_met_consecutively"
TIER2_SKIP_REASON = "concept_ratio_within_target"
TAXONOMY_SKIP_REASON = "taxonomy_kpis_within_target"
CODEX_TAIL_SKIP_REASON = "taxonomy_tail_not_needed"
CODEX_TAIL_ACTIONS = {"relabel", "add_relation", "deprioritize", "blocked"}
TAXONOMY_OUTPUT_RELATIONS = {"SUBCLASS_OF", "INSTANCE_OF", "TYPE_OF"}
STRUCTURAL_RELATION_SKIP_REASONS = {
    "Reverse taxonomy relation already exists",
    "Would introduce taxonomy cycle",
    "Existing SUBCLASS_OF target already present",
    "Existing INSTANCE_OF target already present",
    "Existing TYPE_OF target already present",
}

DEFAULT_TIER2_LABELS = [
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
DEFAULT_TIER2_GUIDANCE = {
    "Financial Metric": "Measurable financial quantities such as P&L, Sharpe ratio, drawdown, fee, or return.",
    "Trading Concept": "Trading ideas such as liquidity, leverage, slippage, execution edge, or market regime.",
    "Market Feature": "Market microstructure or price-action features such as order book depth, spread, or volume profile.",
    "Data Structure": "Programming or mathematical data structures such as tensor, queue, tree, or array.",
    "Trading System": "Named end-to-end trading systems, playbooks, or execution frameworks.",
}


@dataclass
class Tier2LabelCatalog:
    labels: list[str]
    preferred_examples: dict[str, str]
    fallback_label: str = "Concept"


@dataclass
class Tier2Params:
    batch_size: int = 50
    sleep_seconds: float = 1.0
    max_nodes: int = 1000
    cache_file: str = "tier2_classification_cache.json"
    neo4j_uri: str = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.environ.get("NEO4J_USERNAME", "neo4j")
    neo4j_password: str = os.environ.get("NEO4J_PASSWORD", "password123")
    neo4j_database: str = os.environ.get("NEO4J_DATABASE", "neo4j")


@dataclass
class Tier3Params:
    threshold: float = 0.85
    max_candidates: int = 600
    max_merges: int = 200
    sleep_seconds: float = 0.0
    cache_file: str = "embeddings_cache.pkl"
    judge_cache_file: str = "tier3_judge_cache.json"
    neo4j_uri: str = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.environ.get("NEO4J_USERNAME", "neo4j")
    neo4j_password: str = os.environ.get("NEO4J_PASSWORD", "password123")
    neo4j_database: str = os.environ.get("NEO4J_DATABASE", "neo4j")


@dataclass
class TaxonomyParams:
    max_nodes: int = 500
    candidate_limit: int = 8
    embedding_threshold: float = 0.84
    neo4j_uri: str = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.environ.get("NEO4J_USERNAME", "neo4j")
    neo4j_password: str = os.environ.get("NEO4J_PASSWORD", "password123")
    neo4j_database: str = os.environ.get("NEO4J_DATABASE", "neo4j")


@dataclass
class OrchestratorConfig:
    max_iterations: int = 5
    target_concept_ratio: float = 0.05
    target_duplicate_rate: float = 0.015
    target_concept_without_taxonomy_ratio: float = 0.60
    required_consecutive_passes: int = 2
    taxonomy_plateau_reviews: int = 2
    taxonomy_plateau_min_delta: float = 0.002
    run_dir: str | None = None
    codex_bin: str = "codex"
    llm_routing_config: str | None = None
    dry_run: bool = False
    resume: bool = False


def _default_tier2_catalog() -> Tier2LabelCatalog:
    return Tier2LabelCatalog(
        labels=list(DEFAULT_TIER2_LABELS),
        preferred_examples=dict(DEFAULT_TIER2_GUIDANCE),
        fallback_label="Concept",
    )


def _write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        text = raw_line.strip()
        if not text:
            continue
        rows.append(json.loads(text))
    return rows


def _get_or_create_iteration_record(state: dict[str, Any], iteration: int, phase: str) -> dict[str, Any]:
    for record in state["iterations"]:
        if int(record.get("iteration", -1)) == iteration:
            record.setdefault("phase", phase)
            return record
    record = {"iteration": iteration, "phase": phase}
    state["iterations"].append(record)
    return record


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _clamp_int(value: Any, lower: int, upper: int, fallback: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    return max(lower, min(upper, parsed))


def _clamp_float(value: Any, lower: float, upper: float, fallback: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    return max(lower, min(upper, parsed))


def _normalize_label(label: Any) -> str:
    if label is None:
        return ""
    return str(label).strip()


def _ensure_safe_label(label: str) -> None:
    if not label:
        raise ValueError("Label names must be non-empty.")
    if "`" in label:
        raise ValueError(f"Label '{label}' contains an unsafe backtick.")


def _serialize_catalog(catalog: Tier2LabelCatalog) -> dict[str, Any]:
    return {
        "labels": list(catalog.labels),
        "preferred_examples": dict(catalog.preferred_examples),
        "fallback_label": catalog.fallback_label,
    }


def _deserialize_catalog(payload: dict[str, Any] | None) -> Tier2LabelCatalog:
    if not payload:
        return _default_tier2_catalog()
    labels = [_normalize_label(label) for label in payload.get("labels", [])]
    preferred_examples = {
        _normalize_label(key): str(value).strip()
        for key, value in dict(payload.get("preferred_examples", {})).items()
        if _normalize_label(key)
    }
    fallback_label = _normalize_label(payload.get("fallback_label", "Concept")) or "Concept"
    catalog = Tier2LabelCatalog(labels=labels, preferred_examples=preferred_examples, fallback_label=fallback_label)
    _validate_catalog(catalog)
    return catalog


def _validate_catalog(catalog: Tier2LabelCatalog) -> None:
    labels = [_normalize_label(label) for label in catalog.labels]
    if not labels:
        raise ValueError("Tier 2 label catalog cannot be empty.")
    if len(labels) > MAX_TIER2_LABELS:
        raise ValueError(f"Tier 2 label catalog cannot exceed {MAX_TIER2_LABELS} labels.")

    seen: set[str] = set()
    for label in labels:
        _ensure_safe_label(label)
        folded = label.casefold()
        if folded in seen:
            raise ValueError(f"Tier 2 label catalog contains duplicate label '{label}'.")
        seen.add(folded)

    if catalog.fallback_label != "Concept":
        raise ValueError("Tier 2 fallback_label must remain 'Concept'.")
    if "Concept" not in labels:
        raise ValueError("Tier 2 label catalog must include 'Concept'.")

    for key in catalog.preferred_examples:
        if key not in labels:
            raise ValueError(f"Tier 2 guidance references unknown label '{key}'.")

    catalog.labels = labels
    catalog.preferred_examples = {
        key: str(value).strip()
        for key, value in catalog.preferred_examples.items()
        if str(value).strip()
    }


def _concept_only_without_taxonomy_ratio(review: dict[str, Any]) -> float:
    concept_only_count = int(review["kpis"]["concept_only_count"])
    if concept_only_count <= 0:
        return 0.0
    return float(review["taxonomy_kpis"]["concept_only_without_taxonomy_count"]) / concept_only_count


def passes_consolidation_gate(
    kpis: dict[str, Any],
    target_concept_ratio: float,
    target_duplicate_rate: float,
) -> bool:
    concept_ratio = float(kpis["concept_only_ratio"])
    duplicate_rate = float(kpis["duplicate_candidate_rate"])
    return concept_ratio <= target_concept_ratio and duplicate_rate <= target_duplicate_rate


def passes_semantic_gate(
    review: dict[str, Any],
    *,
    target_concept_ratio: float,
    target_duplicate_rate: float,
    target_concept_without_taxonomy_ratio: float,
) -> bool:
    if not passes_consolidation_gate(
        review["kpis"],
        target_concept_ratio=target_concept_ratio,
        target_duplicate_rate=target_duplicate_rate,
    ):
        return False
    if int(review["taxonomy_kpis"]["concept_only_with_similarity_or_alias_count"]) != 0:
        return False
    return _concept_only_without_taxonomy_ratio(review) <= target_concept_without_taxonomy_ratio


def passes_stop_gate(
    review: dict[str, Any],
    *,
    target_concept_ratio: float,
    target_duplicate_rate: float,
    target_concept_without_taxonomy_ratio: float,
) -> bool:
    return passes_semantic_gate(
        review,
        target_concept_ratio=target_concept_ratio,
        target_duplicate_rate=target_duplicate_rate,
        target_concept_without_taxonomy_ratio=target_concept_without_taxonomy_ratio,
    )


def update_consecutive_passes(current_consecutive: int, gate_passed: bool) -> int:
    return current_consecutive + 1 if gate_passed else 0


def parse_codex_review_output(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if not text:
        raise ValueError("Empty codex review output.")

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise ValueError("Codex output did not contain a valid JSON object.") from None
        payload = json.loads(match.group(0))

    if isinstance(payload, list):
        if len(payload) != 1 or not isinstance(payload[0], dict):
            raise ValueError("Codex output list must contain a single JSON object.")
        payload = payload[0]

    if not isinstance(payload, dict):
        raise ValueError("Codex output must be a JSON object.")
    return _normalize_json_object_keys(payload)


def parse_codex_taxonomy_tail_output(raw_text: str) -> dict[str, Any]:
    return parse_codex_review_output(raw_text)


def _normalize_json_object_keys(value: Any) -> Any:
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            normalized[str(key).strip()] = _normalize_json_object_keys(item)
        return normalized
    if isinstance(value, list):
        return [_normalize_json_object_keys(item) for item in value]
    return value


def validate_review_payload(payload: dict[str, Any]) -> dict[str, Any]:
    required_top = [
        "is_consolidated",
        "diagnosis",
        "kpis",
        "taxonomy_kpis",
        "focus_examples",
        "proposed_tier2",
        "proposed_tier3",
        "proposed_tier2_catalog",
        "rationale",
        "confidence",
    ]
    for key in required_top:
        if key not in payload:
            raise ValueError(f"Review payload missing required key '{key}'.")

    diagnosis = str(payload["diagnosis"]).strip()
    if diagnosis not in VALID_DIAGNOSES:
        raise ValueError(f"diagnosis must be one of {sorted(VALID_DIAGNOSES)}.")

    required_kpis = [
        "entity_count",
        "concept_only_count",
        "concept_only_ratio",
        "duplicate_anchor_count",
        "duplicate_candidate_rate",
        "subclass_rel_count",
    ]
    for key in required_kpis:
        if key not in payload["kpis"]:
            raise ValueError(f"Review payload missing KPI '{key}'.")

    required_taxonomy_kpis = [
        "concept_only_without_taxonomy_count",
        "concept_only_degree_le_2_count",
        "concept_only_degree_le_3_count",
        "concept_only_with_similarity_or_alias_count",
    ]
    for key in required_taxonomy_kpis:
        if key not in payload["taxonomy_kpis"]:
            raise ValueError(f"Review payload missing taxonomy KPI '{key}'.")

    required_focus_examples = ["high_degree_concept_only", "low_degree_concept_only"]
    for key in required_focus_examples:
        if key not in payload["focus_examples"]:
            raise ValueError(f"Review payload missing focus_examples key '{key}'.")

    required_tier2 = ["batch_size", "sleep_seconds", "max_nodes"]
    for key in required_tier2:
        if key not in payload["proposed_tier2"]:
            raise ValueError(f"Review payload missing proposed_tier2 key '{key}'.")

    required_tier3 = ["threshold", "max_candidates", "max_merges"]
    for key in required_tier3:
        if key not in payload["proposed_tier3"]:
            raise ValueError(f"Review payload missing proposed_tier3 key '{key}'.")

    catalog_payload = payload["proposed_tier2_catalog"]
    required_catalog = ["labels", "add", "remove", "rename_map", "guidance", "rationale"]
    for key in required_catalog:
        if key not in catalog_payload:
            raise ValueError(f"Review payload missing proposed_tier2_catalog key '{key}'.")

    normalized_catalog = {
        "labels": [_normalize_label(label) for label in catalog_payload["labels"]],
        "add": [_normalize_label(label) for label in catalog_payload["add"]],
        "remove": [_normalize_label(label) for label in catalog_payload["remove"]],
        "rename_map": {
            _normalize_label(old): _normalize_label(new)
            for old, new in dict(catalog_payload["rename_map"]).items()
        },
        "guidance": {
            _normalize_label(label): str(text).strip()
            for label, text in dict(catalog_payload["guidance"]).items()
            if _normalize_label(label)
        },
        "rationale": str(catalog_payload["rationale"]),
    }

    normalized = {
        "is_consolidated": bool(payload["is_consolidated"]),
        "diagnosis": diagnosis,
        "kpis": {
            "entity_count": int(payload["kpis"]["entity_count"]),
            "concept_only_count": int(payload["kpis"]["concept_only_count"]),
            "concept_only_ratio": float(payload["kpis"]["concept_only_ratio"]),
            "duplicate_anchor_count": int(payload["kpis"]["duplicate_anchor_count"]),
            "duplicate_candidate_rate": float(payload["kpis"]["duplicate_candidate_rate"]),
            "subclass_rel_count": int(payload["kpis"]["subclass_rel_count"]),
        },
        "taxonomy_kpis": {
            "concept_only_without_taxonomy_count": int(payload["taxonomy_kpis"]["concept_only_without_taxonomy_count"]),
            "concept_only_degree_le_2_count": int(payload["taxonomy_kpis"]["concept_only_degree_le_2_count"]),
            "concept_only_degree_le_3_count": int(payload["taxonomy_kpis"]["concept_only_degree_le_3_count"]),
            "concept_only_with_similarity_or_alias_count": int(
                payload["taxonomy_kpis"]["concept_only_with_similarity_or_alias_count"]
            ),
        },
        "focus_examples": {
            "high_degree_concept_only": [str(item) for item in payload["focus_examples"]["high_degree_concept_only"]],
            "low_degree_concept_only": [str(item) for item in payload["focus_examples"]["low_degree_concept_only"]],
        },
        "proposed_tier2": {
            "batch_size": int(payload["proposed_tier2"]["batch_size"]),
            "sleep_seconds": float(payload["proposed_tier2"]["sleep_seconds"]),
            "max_nodes": int(payload["proposed_tier2"]["max_nodes"]),
        },
        "proposed_tier3": {
            "threshold": float(payload["proposed_tier3"]["threshold"]),
            "max_candidates": int(payload["proposed_tier3"]["max_candidates"]),
            "max_merges": int(payload["proposed_tier3"]["max_merges"]),
        },
        "proposed_tier2_catalog": normalized_catalog,
        "rationale": str(payload["rationale"]),
        "confidence": float(payload["confidence"]),
    }
    if normalized["confidence"] < 0 or normalized["confidence"] > 1:
        raise ValueError("confidence must be in [0,1].")
    return normalized


def validate_codex_taxonomy_tail_payload(payload: dict[str, Any]) -> dict[str, Any]:
    for key in ("summary", "decisions"):
        if key not in payload:
            raise ValueError(f"Taxonomy tail payload missing required key '{key}'.")
    summary = payload["summary"]
    decisions = payload["decisions"]
    for key in (
        "queue_count",
        "processed_count",
        "recommended_relabels",
        "recommended_relations",
        "deprioritized",
        "blocked",
        "rationale",
        "confidence",
    ):
        if key not in summary:
            raise ValueError(f"Taxonomy tail summary missing required key '{key}'.")
    if not isinstance(decisions, list):
        raise ValueError("Taxonomy tail decisions must be a list.")

    normalized_decisions: list[dict[str, Any]] = []
    for row in decisions:
        if not isinstance(row, dict):
            raise ValueError("Each taxonomy tail decision must be a JSON object.")
        for key in ("eid", "name", "action", "label", "relation", "target_eid", "reason", "priority", "confidence"):
            if key not in row:
                raise ValueError(f"Taxonomy tail decision missing required key '{key}'.")
        action = str(row["action"]).strip()
        if action not in CODEX_TAIL_ACTIONS:
            raise ValueError(f"Unsupported taxonomy tail action '{action}'.")
        relation = str(row["relation"]).strip().upper()
        if relation == "IS_A":
            relation = "TYPE_OF"
        if relation not in {*TAXONOMY_OUTPUT_RELATIONS, "NONE"}:
            raise ValueError(f"Unsupported taxonomy tail relation '{relation}'.")
        confidence = float(row["confidence"])
        if confidence < 0 or confidence > 1:
            raise ValueError("Taxonomy tail confidence must be in [0,1].")
        normalized_decisions.append(
            {
                "eid": str(row["eid"]).strip(),
                "name": str(row["name"]).strip(),
                "action": action,
                "label": _normalize_label(row["label"]) or None,
                "relation": relation,
                "target_eid": str(row["target_eid"]).strip() if row["target_eid"] is not None else None,
                "reason": str(row["reason"]).strip(),
                "priority": int(row["priority"]),
                "confidence": confidence,
            }
        )

    normalized = {
        "summary": {
            "queue_count": int(summary["queue_count"]),
            "processed_count": int(summary["processed_count"]),
            "recommended_relabels": int(summary["recommended_relabels"]),
            "recommended_relations": int(summary["recommended_relations"]),
            "deprioritized": int(summary["deprioritized"]),
            "blocked": int(summary["blocked"]),
            "rationale": str(summary["rationale"]).strip(),
            "confidence": float(summary["confidence"]),
        },
        "decisions": normalized_decisions,
    }
    if normalized["summary"]["confidence"] < 0 or normalized["summary"]["confidence"] > 1:
        raise ValueError("Taxonomy tail summary confidence must be in [0,1].")
    return normalized


def enforce_precision_first_threshold(
    *,
    proposed_threshold: float,
    current_threshold: float,
    duplicate_candidate_rate: float,
    target_duplicate_rate: float,
    last_tier3_summary: dict[str, Any] | None,
) -> float:
    if proposed_threshold >= current_threshold:
        return proposed_threshold

    if last_tier3_summary is None:
        return current_threshold

    alias_acceptance_rate = float(last_tier3_summary.get("alias_acceptance_rate", 0.0))
    judged_pairs = int(last_tier3_summary.get("judged_pairs", 0))
    materially_above_target = duplicate_candidate_rate > (target_duplicate_rate * 1.2)
    strong_alias_quality = judged_pairs >= 20 and alias_acceptance_rate >= 0.2
    if materially_above_target and strong_alias_quality:
        return proposed_threshold
    return current_threshold


def classify_failure_mode(
    *,
    review: dict[str, Any],
    target_concept_ratio: float,
    target_duplicate_rate: float,
    target_concept_without_taxonomy_ratio: float,
) -> str:
    concept_ratio = float(review["kpis"]["concept_only_ratio"])
    duplicate_rate = float(review["kpis"]["duplicate_candidate_rate"])
    concept_only_count = int(review["kpis"]["concept_only_count"])
    taxonomy_kpis = review["taxonomy_kpis"]
    without_taxonomy_ratio = _concept_only_without_taxonomy_ratio(review)
    taxonomy_pressure = (
        concept_ratio > target_concept_ratio
        or without_taxonomy_ratio > target_concept_without_taxonomy_ratio
    )
    duplicate_pressure = duplicate_rate > target_duplicate_rate

    if concept_only_count > 0:
        low_degree_ratio = int(taxonomy_kpis["concept_only_degree_le_3_count"]) / concept_only_count
        similarity_ratio = (
            int(taxonomy_kpis["concept_only_with_similarity_or_alias_count"]) / concept_only_count
        )
    else:
        without_taxonomy_ratio = 0.0
        low_degree_ratio = 0.0
        similarity_ratio = 0.0

    if taxonomy_pressure and not duplicate_pressure and similarity_ratio >= 0.15:
        duplicate_pressure = True

    if taxonomy_pressure and duplicate_pressure:
        return "mixed_debt"
    if duplicate_pressure:
        return "duplicate_debt"
    if taxonomy_pressure:
        return "taxonomy_debt"
    return "balanced"


def should_run_tier2(
    review: dict[str, Any],
    *,
    target_concept_ratio: float,
) -> bool:
    return float(review["kpis"]["concept_only_ratio"]) > target_concept_ratio


def should_run_taxonomy(
    review: dict[str, Any],
    *,
    target_concept_ratio: float,
    target_concept_without_taxonomy_ratio: float,
) -> bool:
    return should_run_tier2(review, target_concept_ratio=target_concept_ratio) or (
        _concept_only_without_taxonomy_ratio(review) > target_concept_without_taxonomy_ratio
    )


def should_run_tier3(
    review: dict[str, Any],
    last_tier3_summary: dict[str, Any] | None,
    *,
    target_concept_ratio: float,
    target_duplicate_rate: float,
    target_concept_without_taxonomy_ratio: float,
) -> bool:
    failure_mode = classify_failure_mode(
        review=review,
        target_concept_ratio=target_concept_ratio,
        target_duplicate_rate=target_duplicate_rate,
        target_concept_without_taxonomy_ratio=target_concept_without_taxonomy_ratio,
    )
    if failure_mode in {"duplicate_debt", "mixed_debt"}:
        return True
    if failure_mode != "taxonomy_debt":
        return False

    concept_only_count = int(review["kpis"]["concept_only_count"])
    similarity_count = int(review["taxonomy_kpis"]["concept_only_with_similarity_or_alias_count"])
    if concept_only_count > 0 and similarity_count / concept_only_count >= 0.2:
        return True

    if last_tier3_summary is None:
        return False

    judged_pairs = int(last_tier3_summary.get("judged_pairs", 0))
    alias_acceptance_rate = float(last_tier3_summary.get("alias_acceptance_rate", 0.0))
    return judged_pairs >= 50 and alias_acceptance_rate >= 0.1


def apply_guardrails(
    *,
    review: dict[str, Any],
    current_tier2: Tier2Params,
    current_tier3: Tier3Params,
    target_concept_ratio: float,
    target_duplicate_rate: float,
    target_concept_without_taxonomy_ratio: float,
    last_tier3_summary: dict[str, Any] | None,
) -> tuple[Tier2Params, Tier3Params]:
    proposed_t2 = review["proposed_tier2"]
    proposed_t3 = review["proposed_tier3"]

    next_tier2 = Tier2Params(
        batch_size=_clamp_int(
            proposed_t2["batch_size"],
            TIER2_BOUNDS["batch_size"][0],
            TIER2_BOUNDS["batch_size"][1],
            current_tier2.batch_size,
        ),
        sleep_seconds=_clamp_float(
            proposed_t2["sleep_seconds"],
            TIER2_BOUNDS["sleep_seconds"][0],
            TIER2_BOUNDS["sleep_seconds"][1],
            current_tier2.sleep_seconds,
        ),
        max_nodes=_clamp_int(
            proposed_t2["max_nodes"],
            TIER2_BOUNDS["max_nodes"][0],
            TIER2_BOUNDS["max_nodes"][1],
            current_tier2.max_nodes,
        ),
        cache_file=current_tier2.cache_file,
        neo4j_uri=current_tier2.neo4j_uri,
        neo4j_user=current_tier2.neo4j_user,
        neo4j_password=current_tier2.neo4j_password,
        neo4j_database=current_tier2.neo4j_database,
    )

    if not should_run_tier3(
        review,
        last_tier3_summary,
        target_concept_ratio=target_concept_ratio,
        target_duplicate_rate=target_duplicate_rate,
        target_concept_without_taxonomy_ratio=target_concept_without_taxonomy_ratio,
    ):
        return next_tier2, current_tier3

    proposed_threshold = _clamp_float(
        proposed_t3["threshold"],
        TIER3_BOUNDS["threshold"][0],
        TIER3_BOUNDS["threshold"][1],
        current_tier3.threshold,
    )
    guarded_threshold = enforce_precision_first_threshold(
        proposed_threshold=proposed_threshold,
        current_threshold=current_tier3.threshold,
        duplicate_candidate_rate=review["kpis"]["duplicate_candidate_rate"],
        target_duplicate_rate=target_duplicate_rate,
        last_tier3_summary=last_tier3_summary,
    )

    next_tier3 = Tier3Params(
        threshold=guarded_threshold,
        max_candidates=_clamp_int(
            proposed_t3["max_candidates"],
            TIER3_BOUNDS["max_candidates"][0],
            TIER3_BOUNDS["max_candidates"][1],
            current_tier3.max_candidates,
        ),
        max_merges=_clamp_int(
            proposed_t3["max_merges"],
            TIER3_BOUNDS["max_merges"][0],
            TIER3_BOUNDS["max_merges"][1],
            current_tier3.max_merges,
        ),
        sleep_seconds=current_tier3.sleep_seconds,
        cache_file=current_tier3.cache_file,
        judge_cache_file=current_tier3.judge_cache_file,
        neo4j_uri=current_tier3.neo4j_uri,
        neo4j_user=current_tier3.neo4j_user,
        neo4j_password=current_tier3.neo4j_password,
        neo4j_database=current_tier3.neo4j_database,
    )
    return next_tier2, next_tier3


def apply_catalog_guardrails(
    *,
    review: dict[str, Any],
    current_catalog: Tier2LabelCatalog,
    live_labels: set[str] | None = None,
) -> tuple[Tier2LabelCatalog, dict[str, Any], dict[str, Any]]:
    proposal = review["proposed_tier2_catalog"]
    normalized_proposal = {
        "labels": [_normalize_label(label) for label in proposal["labels"]],
        "add": [_normalize_label(label) for label in proposal["add"]],
        "remove": [_normalize_label(label) for label in proposal["remove"]],
        "rename_map": {
            _normalize_label(old): _normalize_label(new)
            for old, new in proposal["rename_map"].items()
        },
        "guidance": {
            _normalize_label(label): str(text).strip()
            for label, text in proposal["guidance"].items()
            if _normalize_label(label) and str(text).strip()
        },
        "rationale": str(proposal["rationale"]),
    }

    notes: list[str] = []
    guidance = normalized_proposal["guidance"]
    live_labels = live_labels or set()
    accepted_additions: list[str] = []
    rejected_additions: list[str] = []

    for label in normalized_proposal["add"]:
        if not label or label in current_catalog.labels:
            continue
        if label not in live_labels:
            rejected_additions.append(label)
            continue
        accepted_additions.append(label)

    if (
        normalized_proposal["labels"] != current_catalog.labels
        or any(normalized_proposal[key] for key in ("add", "remove"))
        or normalized_proposal["rename_map"]
    ):
        if accepted_additions:
            notes.append("Applied graph-native catalog additions only; removals and renames remain frozen.")
        else:
            notes.append("Catalog structural changes are operationally frozen except for graph-native additive labels.")
    if rejected_additions:
        notes.append(f"Ignored non-graph-native label additions: {', '.join(rejected_additions)}.")

    next_catalog = Tier2LabelCatalog(
        labels=[*current_catalog.labels, *accepted_additions],
        preferred_examples={
            **current_catalog.preferred_examples,
            **{
                label: text
                for label, text in guidance.items()
                if label in current_catalog.labels or label in accepted_additions
            },
            **{
                label: "Use only when it is the most specific existing fit."
                for label in accepted_additions
                if label not in guidance
            },
        },
        fallback_label="Concept",
    )
    _validate_catalog(next_catalog)

    proposal_artifact = {
        **normalized_proposal,
        "accepted_additions": accepted_additions,
        "rejected_additions": rejected_additions,
        "notes": notes,
        "is_valid": True,
        "structural_changes_applied": bool(accepted_additions),
    }
    applied_artifact = {
        **_serialize_catalog(next_catalog),
        "accepted_additions": accepted_additions,
        "rejected_additions": rejected_additions,
        "notes": notes,
        "is_fallback": False,
        "structural_changes_applied": bool(accepted_additions),
    }
    return next_catalog, proposal_artifact, applied_artifact


def _fetch_live_entity_labels(params: TaxonomyParams) -> set[str]:
    driver = GraphDatabase.driver(params.neo4j_uri, auth=(params.neo4j_user, params.neo4j_password))
    try:
        with driver.session(database=params.neo4j_database) as session:
            rows = session.run(
                """
                MATCH (n:__Entity__)
                UNWIND [label IN labels(n) WHERE label <> '__Entity__'] AS label
                RETURN DISTINCT label
                """
            )
            return {str(row["label"]).strip() for row in rows if str(row["label"]).strip()}
    finally:
        driver.close()


def update_plateau_streak(
    *,
    previous_review: dict[str, Any] | None,
    current_review: dict[str, Any],
    current_streak: int,
    min_delta: float,
) -> int:
    if current_review["diagnosis"] != "taxonomy_debt":
        return 0
    if previous_review is None or previous_review.get("diagnosis") != "taxonomy_debt":
        return 0
    prev_ratio = float(previous_review["kpis"]["concept_only_ratio"])
    curr_ratio = float(current_review["kpis"]["concept_only_ratio"])
    ratio_improvement = prev_ratio - curr_ratio
    without_taxonomy_improvement = (
        _concept_only_without_taxonomy_ratio(previous_review) - _concept_only_without_taxonomy_ratio(current_review)
    )
    return (
        current_streak + 1
        if ratio_improvement < min_delta and without_taxonomy_improvement < WITHOUT_TAXONOMY_PLATEAU_MIN_DELTA
        else 0
    )


def _run_command(command: list[str], log_file: Path) -> None:
    result = subprocess.run(
        command,
        text=True,
        input=None,
        capture_output=True,
        cwd=REPO_ROOT,
        check=False,
    )
    log_file.write_text(
        f"COMMAND: {' '.join(command)}\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}",
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(command)}. "
            f"See log: {log_file}"
        )


def _run_command_with_input(command: list[str], log_file: Path, stdin_text: str) -> None:
    result = subprocess.run(
        command,
        text=True,
        input=stdin_text,
        capture_output=True,
        cwd=REPO_ROOT,
        check=False,
    )
    log_file.write_text(
        f"COMMAND: {' '.join(command)}\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}",
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(command)}. "
            f"See log: {log_file}"
        )


def _run_agent_command(
    command: list[str],
    log_file: Path,
    raw_output_file: Path,
    *,
    stdin_text: str | None = None,
    env_overrides: dict[str, str] | None = None,
) -> None:
    timeout_seconds = AGENT_COMMAND_TIMEOUT_SECONDS if AGENT_COMMAND_TIMEOUT_SECONDS > 0 else None
    try:
        result = subprocess.run(
            command,
            text=True,
            input=stdin_text,
            capture_output=True,
            cwd=REPO_ROOT,
            env={**os.environ, **(env_overrides or {})},
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        log_file.write_text(
            f"COMMAND: {' '.join(command)}\n\nTIMEOUT_SECONDS: {timeout_seconds}\n\nSTDOUT:\n{stdout}\n\nSTDERR:\n{stderr}",
            encoding="utf-8",
        )
        raise RuntimeError(
            f"Command timed out after {timeout_seconds}s: {' '.join(command)}. "
            f"See log: {log_file}"
        ) from exc
    log_file.write_text(
        f"COMMAND: {' '.join(command)}\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}",
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(command)}. "
            f"See log: {log_file}"
        )
    raw_output_file.write_text(result.stdout, encoding="utf-8")


def _resolve_cli_executable(executable: str) -> str:
    candidate = Path(executable)
    if candidate.exists():
        return str(candidate.resolve())
    if os.name == "nt" and candidate.parent == Path("."):
        lookup = candidate.stem or executable
        try:
            result = subprocess.run(
                ["where.exe", lookup],
                text=True,
                capture_output=True,
                check=False,
                cwd=REPO_ROOT,
            )
        except Exception:
            result = None
        if result and result.returncode == 0:
            candidates = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            preferred_suffixes = (
                f"\\{lookup.lower()}.exe",
                f"\\{lookup.lower()}.cmd",
                f"\\{lookup.lower()}.bat",
                f"\\{lookup.lower()}",
            )
            for suffix in preferred_suffixes:
                for match in candidates:
                    if match.lower().endswith(suffix):
                        return match
    resolved = shutil.which(executable)
    return resolved or executable


def _review_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": [
            "is_consolidated",
            "diagnosis",
            "kpis",
            "taxonomy_kpis",
            "focus_examples",
            "proposed_tier2",
            "proposed_tier3",
            "proposed_tier2_catalog",
            "rationale",
            "confidence",
        ],
    }


def _taxonomy_tail_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["summary", "decisions"],
    }


def _build_agent_command(
    *,
    role_config: AgentRoleConfig,
    raw_output_file: Path,
    schema_file: Path | None,
    prompt: str,
) -> tuple[list[str], str | None]:
    executable = _resolve_cli_executable(role_config.executable or "codex")
    model_args = ["--model", role_config.model] if role_config.model else []
    if role_config.client == "codex":
        if role_config.model:
            model_args = ["-m", role_config.model]
        command = [executable, "exec", *role_config.args, "--ephemeral", *model_args, "-o", str(raw_output_file), "-"]
        if os.name == "nt" and executable.lower().endswith(".cmd"):
            command = ["cmd", "/c", *command]
        return command, prompt
    if role_config.client == "claude":
        command = [executable, *role_config.args, "-p", "--permission-mode", "bypassPermissions", *model_args]
        command.extend(["--output-format", "json"])
        return command, prompt
    if role_config.client == "opencode":
        command = [executable, "run", *role_config.args, *model_args, prompt]
        return command, None
    raise ValueError(f"Unsupported agent client '{role_config.client}'.")


def _run_agent_prompt(
    *,
    role_config: AgentRoleConfig,
    prompt: str,
    raw_output_file: Path,
    stdout_log_file: Path,
    schema_payload: dict[str, Any] | None = None,
) -> str:
    schema_file = None
    if schema_payload is not None and role_config.client not in {"claude"}:
        schema_file = raw_output_file.with_suffix(".schema.json")
        schema_file.write_text(json.dumps(schema_payload, indent=2), encoding="utf-8")
    command, stdin_text = _build_agent_command(
        role_config=role_config,
        raw_output_file=raw_output_file,
        schema_file=schema_file,
        prompt=prompt,
    )
    if role_config.client == "codex":
        _run_command_with_input(command, stdout_log_file, stdin_text or "")
    else:
        _run_agent_command(
            command,
            stdout_log_file,
            raw_output_file,
            stdin_text=stdin_text,
            env_overrides=role_config.env,
        )
    raw_text = raw_output_file.read_text(encoding="utf-8")
    if role_config.client == "claude":
        normalized = _extract_claude_text(raw_text)
        raw_output_file.write_text(normalized, encoding="utf-8")
        return normalized
    if role_config.client == "opencode":
        normalized = _extract_opencode_text(raw_text)
        raw_output_file.write_text(normalized, encoding="utf-8")
        return normalized
    return raw_text


def _extract_claude_text(raw_text: str) -> str:
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        return raw_text
    if not isinstance(payload, dict):
        return raw_text
    result = payload.get("result")
    if not isinstance(result, str):
        return raw_text
    stripped = result.strip()
    if stripped.startswith("```"):
        fence_lines = stripped.splitlines()
        if len(fence_lines) >= 3 and fence_lines[0].startswith("```") and fence_lines[-1] == "```":
            return "\n".join(fence_lines[1:-1]).strip()
    return stripped


def _extract_opencode_text(raw_text: str) -> str:
    text_parts: list[str] = []
    for line in raw_text.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            return raw_text
        if payload.get("type") != "text":
            continue
        part = payload.get("part") or {}
        text = part.get("text")
        if isinstance(text, str):
            text_parts.append(text)
    return "\n".join(text_parts) if text_parts else raw_text


def run_tier1(*, params: Tier2Params, dry_run: bool, iteration_dir: Path) -> None:
    command = [
        sys.executable,
        "consolidate_tier1_lemmatize.py",
        "--neo4j-uri",
        params.neo4j_uri,
        "--neo4j-user",
        params.neo4j_user,
        "--neo4j-password",
        params.neo4j_password,
        "--neo4j-database",
        params.neo4j_database,
    ]
    if dry_run:
        command.append("--dry-run")
    _run_command(command, iteration_dir / "tier1.log")


def run_tier2(
    *,
    params: Tier2Params,
    catalog: Tier2LabelCatalog,
    dry_run: bool,
    iteration_dir: Path,
    llm_routing_config: str | None = None,
) -> dict[str, Any]:
    summary_path = iteration_dir / "tier2_summary.json"
    catalog_path = iteration_dir / "tier2_catalog.json"
    decisions_path = iteration_dir / "tier2_decisions.jsonl"
    _write_json(catalog_path, _serialize_catalog(catalog))
    command = [
        sys.executable,
        "consolidate_tier2_relabel.py",
        "--batch-size",
        str(params.batch_size),
        "--sleep-seconds",
        str(params.sleep_seconds),
        "--max-nodes",
        str(params.max_nodes),
        "--cache-file",
        params.cache_file,
        "--neo4j-uri",
        params.neo4j_uri,
        "--neo4j-user",
        params.neo4j_user,
        "--neo4j-password",
        params.neo4j_password,
        "--neo4j-database",
        params.neo4j_database,
        "--labels-json",
        str(catalog_path),
        "--decisions-jsonl",
        str(decisions_path),
        "--summary-json",
        str(summary_path),
    ]
    if llm_routing_config:
        command.extend(["--llm-routing-config", llm_routing_config])
    if dry_run:
        command.append("--dry-run")
    _run_command(command, iteration_dir / "tier2.log")
    return _read_json(summary_path)


def run_taxonomy(
    *,
    params: TaxonomyParams,
    catalog: Tier2LabelCatalog,
    dry_run: bool,
    iteration_dir: Path,
    prior_taxonomy_decisions_jsonl: str | None = None,
    prior_review_json: str | None = None,
    llm_routing_config: str | None = None,
) -> dict[str, Any]:
    summary_path = iteration_dir / "taxonomy_summary.json"
    decisions_path = iteration_dir / "taxonomy_decisions.jsonl"
    catalog_path = iteration_dir / "tier2_catalog.json"
    tier2_decisions_path = iteration_dir / "tier2_decisions.jsonl"
    if not catalog_path.exists():
        _write_json(catalog_path, _serialize_catalog(catalog))
    command = [
        sys.executable,
        "consolidate_taxonomy_cleanup.py",
        "--max-nodes",
        str(params.max_nodes),
        "--candidate-limit",
        str(params.candidate_limit),
        "--embedding-threshold",
        str(params.embedding_threshold),
        "--neo4j-uri",
        params.neo4j_uri,
        "--neo4j-user",
        params.neo4j_user,
        "--neo4j-password",
        params.neo4j_password,
        "--neo4j-database",
        params.neo4j_database,
        "--labels-json",
        str(catalog_path),
        "--tier2-decisions-jsonl",
        str(tier2_decisions_path),
        "--decisions-jsonl",
        str(decisions_path),
        "--summary-json",
        str(summary_path),
    ]
    if llm_routing_config:
        command.extend(["--llm-routing-config", llm_routing_config])
    if prior_taxonomy_decisions_jsonl:
        command.extend(["--prior-taxonomy-decisions-jsonl", prior_taxonomy_decisions_jsonl])
    if prior_review_json:
        command.extend(["--prior-review-json", prior_review_json])
    if dry_run:
        command.append("--dry-run")
    _run_command(command, iteration_dir / "taxonomy.log")
    return _read_json(summary_path)


def _taxonomy_codex_queue_path(iteration_dir: Path) -> Path:
    return iteration_dir / "taxonomy_codex_queue.jsonl"


def _codex_taxonomy_tail_raw_path(iteration_dir: Path) -> Path:
    return iteration_dir / "codex_taxonomy_tail_raw.txt"


def _codex_taxonomy_tail_output_path(iteration_dir: Path) -> Path:
    return iteration_dir / "codex_taxonomy_tail.json"


def _applied_codex_taxonomy_tail_output_path(iteration_dir: Path) -> Path:
    return iteration_dir / "applied_codex_taxonomy_tail.json"


def _normalize_name_text(value: str | None) -> str:
    return normalize_name(value or "")


def _token_set_text(value: str | None) -> set[str]:
    return token_set(value or "")


def _relation_direction_is_plausible_for_tail(
    *,
    source_name: str,
    target_name: str,
    relation: str,
) -> tuple[bool, str]:
    if relation not in TAXONOMY_OUTPUT_RELATIONS:
        return False, "Unsupported relation"
    normalized_source = _normalize_name_text(source_name)
    normalized_target = _normalize_name_text(target_name)
    source_tokens = _token_set_text(source_name)
    target_tokens = _token_set_text(target_name)
    if normalized_source and normalized_target and normalized_source == normalized_target:
        return False, "Target name matches source name"
    if source_tokens and target_tokens and source_tokens < target_tokens:
        return False, "Target appears lexically narrower than source"
    if normalized_source and normalized_target and normalized_target.startswith(f"{normalized_source} "):
        return False, "Target appears to specialize the source"
    return True, ""


def _can_apply_taxonomy_relation(
    session: Any,
    *,
    source_eid: str,
    relation: str,
    target_eid: str,
) -> tuple[bool, str]:
    if relation not in TAXONOMY_OUTPUT_RELATIONS:
        return False, "Unsupported relation"
    if source_eid == target_eid:
        return False, "Self-loop"
    reverse = session.run(
        """
        MATCH (source)-[rel:SUBCLASS_OF|INSTANCE_OF|TYPE_OF|IS_A]->(target)
        WHERE elementId(source) = $target_eid AND elementId(target) = $source_eid
        RETURN count(rel) AS reverse_count
        """,
        source_eid=source_eid,
        target_eid=target_eid,
    ).single()
    if int(reverse["reverse_count"]) > 0:
        return False, "Reverse taxonomy relation already exists"
    cycle = session.run(
        """
        MATCH (source) WHERE elementId(source) = $source_eid
        MATCH (target) WHERE elementId(target) = $target_eid
        RETURN EXISTS {
            MATCH (target)-[:SUBCLASS_OF|INSTANCE_OF|TYPE_OF|IS_A*1..6]->(source)
        } AS would_cycle
        """,
        source_eid=source_eid,
        target_eid=target_eid,
    ).single()
    if bool(cycle["would_cycle"]):
        return False, "Would introduce taxonomy cycle"
    conflicting = session.run(
        f"""
        MATCH (source)-[rel:{relation}]->(other)
        WHERE elementId(source) = $source_eid AND elementId(other) <> $target_eid
        RETURN count(rel) AS conflicting_count
        """,
        source_eid=source_eid,
        target_eid=target_eid,
    ).single()
    if int(conflicting["conflicting_count"]) > 0:
        return False, f"Existing {relation} target already present"
    return True, ""


def _apply_taxonomy_label(session: Any, *, source_eid: str, old_labels: list[str], new_label: str) -> None:
    safe_new_label = new_label.replace("`", "")
    removal_clauses = []
    for old_label in old_labels:
        if old_label in {"__Entity__", safe_new_label}:
            continue
        removal_clauses.append(f"REMOVE n:`{old_label.replace('`', '')}`")
    removal_block = "\n        ".join(removal_clauses)
    session.run(
        f"""
        MATCH (n) WHERE elementId(n) = $source_eid
        SET n:`{safe_new_label}`
        {removal_block}
        """,
        source_eid=source_eid,
    )


def _add_taxonomy_relation(session: Any, *, source_eid: str, relation: str, target_eid: str) -> None:
    session.run(
        f"""
        MATCH (source) WHERE elementId(source) = $source_eid
        MATCH (target) WHERE elementId(target) = $target_eid
        MERGE (source)-[:{relation}]->(target)
        """,
        source_eid=source_eid,
        target_eid=target_eid,
    )


def _build_codex_taxonomy_tail_prompt(
    *,
    queue_rows: list[dict[str, Any]],
    current_catalog: Tier2LabelCatalog,
    target_concept_without_taxonomy_ratio: float,
    iteration_dir: Path,
    prior_review_path: Path | None,
) -> str:
    queue_payload = json.dumps(queue_rows, indent=2)
    prior_review_text = str(prior_review_path) if prior_review_path is not None else "None"
    return f"""
Use Neo4j MCP in read-only mode only.
Review only the queued residual taxonomy nodes below. Prefer concrete executable repairs over generic commentary.

Current target:
- concept_only_without_taxonomy_ratio <= {target_concept_without_taxonomy_ratio}

Current Tier 2 catalog:
{json.dumps(_serialize_catalog(current_catalog), indent=2)}

Execution artifacts for this iteration:
- taxonomy_decisions.jsonl: {iteration_dir / "taxonomy_decisions.jsonl"}
- taxonomy_summary.json: {iteration_dir / "taxonomy_summary.json"}
- prior codex_review.json: {prior_review_text}

Queued residual taxonomy nodes:
{queue_payload}

Instructions:
- Verify each queued node is still a live concept-only residual without outgoing taxonomy support before recommending a repair.
- Inspect local graph context through Neo4j MCP as needed.
- Choose exactly one action per node: relabel, add_relation, deprioritize, or blocked.
- For relabel, choose one label.
- For add_relation, choose exactly one relation and one provided target candidate.
- Use only the provided target candidates for add_relation.
- Do not invent labels unless they already exist in the live graph and are clearly the best fit.
- If the node is semantically meaningful but structurally blocked, use blocked, not deprioritize.
- If the node is placeholder/noise residue, use deprioritize.
- Rank nodes by fix priority using smaller integers for higher priority.

Return strict JSON only with this schema:
{{
  "summary": {{
    "queue_count": int,
    "processed_count": int,
    "recommended_relabels": int,
    "recommended_relations": int,
    "deprioritized": int,
    "blocked": int,
    "rationale": "short text",
    "confidence": float
  }},
  "decisions": [
    {{
      "eid": "string",
      "name": "string",
      "action": "relabel|add_relation|deprioritize|blocked",
      "label": "string or null",
      "relation": "SUBCLASS_OF|INSTANCE_OF|TYPE_OF|NONE",
      "target_eid": "string or null",
      "reason": "short text",
      "priority": 1,
      "confidence": 0.0
    }}
  ]
}}
No markdown, no prose outside JSON.
""".strip()


def run_codex_taxonomy_tail(
    *,
    codex_bin: str,
    llm_routing_config: str | None = None,
    current_catalog: Tier2LabelCatalog,
    target_concept_without_taxonomy_ratio: float,
    iteration_dir: Path,
    prior_review_path: Path | None,
) -> dict[str, Any]:
    queue_rows = _read_jsonl(_taxonomy_codex_queue_path(iteration_dir))
    raw_output_file = _codex_taxonomy_tail_raw_path(iteration_dir)
    stdout_log_file = iteration_dir / "codex_taxonomy_tail_exec.log"
    prompt = _build_codex_taxonomy_tail_prompt(
        queue_rows=queue_rows,
        current_catalog=current_catalog,
        target_concept_without_taxonomy_ratio=target_concept_without_taxonomy_ratio,
        iteration_dir=iteration_dir,
        prior_review_path=prior_review_path,
    )
    agent_role = resolve_agent_role(
        llm_routing_config,
        AGENT_TAXONOMY_TAIL_ROLE,
        default_client="codex",
        default_model=None,
        default_executable=codex_bin,
    )
    last_error: Exception | None = None
    for attempt in range(1, max(CODEX_REVIEW_MAX_ATTEMPTS, 1) + 1):
        raw_text = _run_agent_prompt(
            role_config=agent_role,
            prompt=prompt,
            raw_output_file=raw_output_file,
            stdout_log_file=stdout_log_file,
            schema_payload=_taxonomy_tail_json_schema(),
        )
        try:
            parsed = parse_codex_taxonomy_tail_output(raw_text)
            validated = validate_codex_taxonomy_tail_payload(parsed)
            _write_json(_codex_taxonomy_tail_output_path(iteration_dir), validated)
            return validated
        except (json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
            last_error = exc
            with stdout_log_file.open("a", encoding="utf-8") as handle:
                handle.write(f"\n\nTAIL_PARSE_ATTEMPT_{attempt}_ERROR:\n{type(exc).__name__}: {exc}\n")
            if attempt >= max(CODEX_REVIEW_MAX_ATTEMPTS, 1):
                break
    assert last_error is not None
    raise last_error


def apply_codex_taxonomy_tail(
    *,
    params: TaxonomyParams,
    current_catalog: Tier2LabelCatalog,
    iteration_dir: Path,
    dry_run: bool,
) -> dict[str, Any]:
    queue_rows = _read_jsonl(_taxonomy_codex_queue_path(iteration_dir))
    queue_by_eid = {str(row.get("eid")): row for row in queue_rows if str(row.get("eid", "")).strip()}
    decisions_payload = _read_json(_codex_taxonomy_tail_output_path(iteration_dir))
    allowed_labels = set(current_catalog.labels)
    applied_rows: list[dict[str, Any]] = []
    summary = {
        "queue_count": len(queue_rows),
        "processed_count": 0,
        "applied_relabels": 0,
        "applied_relations": 0,
        "deprioritized": 0,
        "blocked": 0,
        "skipped_invalid": 0,
    }
    driver = GraphDatabase.driver(params.neo4j_uri, auth=(params.neo4j_user, params.neo4j_password))
    try:
        with driver.session(database=params.neo4j_database) as session:
            for decision in decisions_payload["decisions"]:
                result = {
                    **decision,
                    "applied": False,
                    "skipped_reason": "",
                }
                queue_row = queue_by_eid.get(decision["eid"])
                summary["processed_count"] += 1
                if queue_row is None:
                    result["skipped_reason"] = "Decision eid not present in taxonomy Codex queue"
                    summary["skipped_invalid"] += 1
                    applied_rows.append(result)
                    continue
                if decision["action"] == "relabel":
                    if not decision["label"] or decision["label"] not in allowed_labels:
                        result["skipped_reason"] = "Proposed label not in active catalog"
                        summary["skipped_invalid"] += 1
                    else:
                        if not dry_run:
                            _apply_taxonomy_label(
                                session,
                                source_eid=decision["eid"],
                                old_labels=list(queue_row.get("current_labels", [])),
                                new_label=decision["label"],
                            )
                        result["applied"] = True
                        summary["applied_relabels"] += 1
                elif decision["action"] == "add_relation":
                    candidate_by_eid = {
                        str(candidate.get("eid")): candidate
                        for candidate in queue_row.get("candidate_targets", [])
                        if str(candidate.get("eid", "")).strip()
                    }
                    candidate = candidate_by_eid.get(decision["target_eid"] or "")
                    if decision["relation"] not in TAXONOMY_OUTPUT_RELATIONS:
                        result["skipped_reason"] = "Invalid relation"
                        summary["skipped_invalid"] += 1
                    elif candidate is None:
                        result["skipped_reason"] = "Target candidate not in provided list"
                        summary["skipped_invalid"] += 1
                    else:
                        plausible, skipped_reason = _relation_direction_is_plausible_for_tail(
                            source_name=decision["name"],
                            target_name=str(candidate.get("name", "")),
                            relation=decision["relation"],
                        )
                        if not plausible:
                            result["skipped_reason"] = skipped_reason
                        else:
                            allowed, skipped_reason = _can_apply_taxonomy_relation(
                                session,
                                source_eid=decision["eid"],
                                relation=decision["relation"],
                                target_eid=decision["target_eid"] or "",
                            )
                            if not allowed:
                                result["skipped_reason"] = skipped_reason
                            else:
                                if not dry_run:
                                    _add_taxonomy_relation(
                                        session,
                                        source_eid=decision["eid"],
                                        relation=decision["relation"],
                                        target_eid=decision["target_eid"] or "",
                                    )
                                result["applied"] = True
                                summary["applied_relations"] += 1
                elif decision["action"] == "deprioritize":
                    summary["deprioritized"] += 1
                elif decision["action"] == "blocked":
                    summary["blocked"] += 1
                applied_rows.append(result)
    finally:
        driver.close()
    payload = {"summary": summary, "results": applied_rows}
    _write_json(_applied_codex_taxonomy_tail_output_path(iteration_dir), payload)
    return payload


def _merge_codex_tail_into_taxonomy_summary(iteration_dir: Path, taxonomy_summary: dict[str, Any], tail_applied: dict[str, Any]) -> dict[str, Any]:
    merged = dict(taxonomy_summary)
    merged["codex_queue_count"] = int(merged.get("codex_queue_count", 0) or 0)
    merged["codex_tail_applied_relabels"] = int(tail_applied["summary"].get("applied_relabels", 0) or 0)
    merged["codex_tail_applied_relations"] = int(tail_applied["summary"].get("applied_relations", 0) or 0)
    merged["codex_tail_deprioritized"] = int(tail_applied["summary"].get("deprioritized", 0) or 0)
    merged["codex_tail_blocked"] = int(tail_applied["summary"].get("blocked", 0) or 0)
    _write_json(iteration_dir / "taxonomy_summary.json", merged)
    return merged


def _compute_live_concept_only_without_taxonomy_ratio(params: TaxonomyParams) -> float:
    driver = GraphDatabase.driver(params.neo4j_uri, auth=(params.neo4j_user, params.neo4j_password))
    try:
        with driver.session(database=params.neo4j_database) as session:
            row = session.run(
                """
                MATCH (n:__Entity__:Concept)
                WHERE ALL(l IN labels(n) WHERE l IN ['__Entity__', 'Concept'])
                WITH collect(n) AS concept_nodes
                WITH
                    size(concept_nodes) AS concept_only_count,
                    size([
                        node IN concept_nodes
                        WHERE NOT EXISTS { (node)-[:SUBCLASS_OF|INSTANCE_OF|IS_A|TYPE_OF]->() }
                    ]) AS concept_only_without_taxonomy_count
                RETURN concept_only_count, concept_only_without_taxonomy_count
                """
            ).single()
            concept_only_count = int(row["concept_only_count"] or 0)
            if concept_only_count <= 0:
                return 0.0
            return float(row["concept_only_without_taxonomy_count"] or 0) / concept_only_count
    finally:
        driver.close()


def _maybe_run_codex_taxonomy_tail(
    *,
    config: OrchestratorConfig,
    state: dict[str, Any],
    state_path: Path,
    run_dir: Path,
    iteration_dir: Path,
    iteration_record: dict[str, Any],
    iteration_index: int,
    taxonomy_params: TaxonomyParams,
    current_catalog: Tier2LabelCatalog,
) -> None:
    if "codex_taxonomy_tail_summary" in iteration_record or iteration_record.get("codex_taxonomy_tail_skipped", False):
        return
    queue_path = _taxonomy_codex_queue_path(iteration_dir)
    queue_rows = _read_jsonl(queue_path)
    live_ratio = _compute_live_concept_only_without_taxonomy_ratio(taxonomy_params)
    should_run_codex_tail = (
        "taxonomy_summary" in iteration_record
        and not iteration_record.get("taxonomy_skipped", False)
        and live_ratio > config.target_concept_without_taxonomy_ratio
        and bool(queue_rows)
    )
    if should_run_codex_tail:
        prior_review_path = run_dir / f"iteration_{iteration_index - 1}" / "codex_review.json"
        if iteration_index <= 0 or not prior_review_path.exists():
            prior_review_path = None
        tail_output_path = _codex_taxonomy_tail_output_path(iteration_dir)
        tail_applied_path = _applied_codex_taxonomy_tail_output_path(iteration_dir)
        if tail_output_path.exists():
            tail_review = _read_json(tail_output_path)
        else:
            state["active_step"] = f"iteration_{iteration_index}_codex_taxonomy_tail"
            _write_json(state_path, state)
            tail_kwargs = {
                "codex_bin": config.codex_bin,
                "current_catalog": current_catalog,
                "target_concept_without_taxonomy_ratio": config.target_concept_without_taxonomy_ratio,
                "iteration_dir": iteration_dir,
                "prior_review_path": prior_review_path,
            }
            if config.llm_routing_config:
                tail_kwargs["llm_routing_config"] = config.llm_routing_config
            tail_review = run_codex_taxonomy_tail(**tail_kwargs)
        if tail_applied_path.exists():
            tail_applied = _read_json(tail_applied_path)
        else:
            state["active_step"] = f"iteration_{iteration_index}_codex_taxonomy_tail_apply"
            _write_json(state_path, state)
            tail_applied = apply_codex_taxonomy_tail(
                params=taxonomy_params,
                current_catalog=current_catalog,
                iteration_dir=iteration_dir,
                dry_run=config.dry_run,
            )
        iteration_record["codex_taxonomy_tail_skipped"] = False
        iteration_record.pop("codex_taxonomy_tail_skip_reason", None)
        iteration_record["codex_taxonomy_tail_summary"] = {
            "live_concept_only_without_taxonomy_ratio": live_ratio,
            "review": tail_review["summary"],
            "application": tail_applied["summary"],
        }
        if "taxonomy_summary" in iteration_record:
            iteration_record["taxonomy_summary"] = _merge_codex_tail_into_taxonomy_summary(
                iteration_dir,
                iteration_record["taxonomy_summary"],
                tail_applied,
            )
        _write_json(state_path, state)
        return
    iteration_record["codex_taxonomy_tail_skipped"] = True
    iteration_record["codex_taxonomy_tail_skip_reason"] = CODEX_TAIL_SKIP_REASON
    _write_json(state_path, state)


def run_tier3(
    *,
    params: Tier3Params,
    dry_run: bool,
    iteration_dir: Path,
    llm_routing_config: str | None = None,
) -> dict[str, Any]:
    summary_path = iteration_dir / "tier3_summary.json"
    command = [
        sys.executable,
        "consolidate_tier3_semantic.py",
        "--threshold",
        str(params.threshold),
        "--max-candidates",
        str(params.max_candidates),
        "--max-merges",
        str(params.max_merges),
        "--sleep-seconds",
        str(params.sleep_seconds),
        "--cache-file",
        params.cache_file,
        "--judge-cache-file",
        params.judge_cache_file,
        "--neo4j-uri",
        params.neo4j_uri,
        "--neo4j-user",
        params.neo4j_user,
        "--neo4j-password",
        params.neo4j_password,
        "--neo4j-database",
        params.neo4j_database,
        "--summary-json",
        str(summary_path),
    ]
    if llm_routing_config:
        command.extend(["--llm-routing-config", llm_routing_config])
    if dry_run:
        command.append("--dry-run")
    _run_command(command, iteration_dir / "tier3.log")
    return _read_json(summary_path)


def _write_review_diagnostics(
    *,
    iteration_dir: Path,
    review: dict[str, Any],
    effective_diagnosis: str | None = None,
    tier2_summary: dict[str, Any] | None = None,
    taxonomy_summary: dict[str, Any] | None = None,
    evidence_iteration_dir: Path | None = None,
    consolidation_gate_pass: bool | None = None,
    semantic_gate_pass: bool | None = None,
) -> None:
    suspicious_examples: list[dict[str, Any]] = []
    evidence_dir = evidence_iteration_dir or iteration_dir
    execution_artifact_mode = "unknown"
    for summary in (taxonomy_summary, tier2_summary):
        if isinstance(summary, dict) and "dry_run" in summary:
            execution_artifact_mode = "dry_run" if bool(summary["dry_run"]) else "live_run"
            break
    tier2_decisions_path = evidence_dir / "tier2_decisions.jsonl"
    taxonomy_decisions_path = evidence_dir / "taxonomy_decisions.jsonl"
    for candidate_path in (taxonomy_decisions_path, tier2_decisions_path):
        if not candidate_path.exists():
            continue
        for raw_line in candidate_path.read_text(encoding="utf-8").splitlines():
            text = raw_line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if payload.get("status") == "unresolved" or payload.get("skipped_reason") or payload.get("suspicious"):
                suspicious_examples.append(
                    {
                        "name": payload.get("name"),
                        "action": payload.get("action"),
                        "reason": payload.get("suspicious_reason") or payload.get("skipped_reason") or payload.get("reason"),
                        "confidence": payload.get("confidence"),
                    }
                )
            if len(suspicious_examples) >= 5:
                break
        if len(suspicious_examples) >= 5:
            break

    _write_json(
        iteration_dir / "review_diagnostics.json",
        {
            "raw_diagnosis": review["diagnosis"],
            "effective_diagnosis": effective_diagnosis or review["diagnosis"],
            "kpi_source": "live_graph_review",
            "execution_artifact_mode": execution_artifact_mode,
            "evidence_iteration_dir": str(evidence_dir),
            "kpis": review["kpis"],
            "taxonomy_kpis": review["taxonomy_kpis"],
            "concept_only_without_taxonomy_ratio": _concept_only_without_taxonomy_ratio(review),
            "consolidation_gate_pass": consolidation_gate_pass,
            "semantic_gate_pass": semantic_gate_pass,
            "focus_examples": review["focus_examples"],
            "tier2_summary": tier2_summary,
            "taxonomy_summary": taxonomy_summary,
            "taxonomy_relation_counts_by_type": (taxonomy_summary or {}).get("relations_added", {}),
            "suspicious_relabel_count": (taxonomy_summary or {}).get("suspicious_relabels", 0),
            "suspicious_keep_label_concept_count": (taxonomy_summary or {}).get("suspicious_keep_label_concepts", 0),
            "relabeled_nodes_missing_taxonomy_support": (
                (taxonomy_summary or {}).get("relabeled_without_taxonomy_support")
                or (tier2_summary or {}).get("relabeled_without_taxonomy_support", 0)
            ),
            "residual_seed_count": (taxonomy_summary or {}).get("residual_seed_count", 0),
            "carry_forward_seed_count": (taxonomy_summary or {}).get("carry_forward_seed_count", 0),
            "review_focus_seed_count": (taxonomy_summary or {}).get("review_focus_seed_count", 0),
            "post_relabel_relation_attempts": (taxonomy_summary or {}).get("post_relabel_relation_attempts", 0),
            "post_relabel_relations_added": (taxonomy_summary or {}).get("post_relabel_relations_added", 0),
            "suspicious_relabel_examples": suspicious_examples,
            "catalog_rationale": review["proposed_tier2_catalog"]["rationale"],
        },
    )


def _build_codex_prompt(
    *,
    review_client: str | None,
    target_concept_ratio: float,
    target_duplicate_rate: float,
    target_concept_without_taxonomy_ratio: float,
    current_tier2: Tier2Params,
    current_tier3: Tier3Params,
    current_catalog: Tier2LabelCatalog,
) -> str:
    output_contract = [
        "Structured-output contract:",
        "- Machine-consumed response: output exactly one JSON object.",
        '- First character must be "{" and last character must be "}".',
        "- No markdown fences, headings, bullets, prose, or lead-in/outro text.",
        "- Include every required key from the schema below exactly once; use 0, [], or {} when needed.",
        "- If no catalog changes are proposed, keep the full labels list and use add/remove=[], rename_map/guidance={}.",
        "- Before sending, internally verify that json.loads(final_response) would succeed.",
    ]
    if review_client == "opencode":
        output_contract.extend(
            [
                "- You are not writing a human-readable report. You are filling an API response object.",
                '- Do not say "I\'ll analyze", "Here is", or similar lead-in text.',
            ]
        )
    return f"""
  Use the Neo4j MCP tool in read-only mode to review graph consolidation quality.
  The execution pipeline now includes a precision-first taxonomy cleanup pass between Tier 2 and Tier 3.
  Prioritize semantic cleanliness under the current label set before recommending structural label catalog changes.
  {"\n  ".join(output_contract)}
  Run Cypher queries to compute these KPIs:
  1) entity_count: MATCH (n:__Entity__) RETURN count(n) AS entity_count
  2) concept_only_count:
     MATCH (n:__Entity__:Concept)
     WHERE ALL(l IN labels(n) WHERE l IN ['__Entity__', 'Concept'])
   RETURN count(n) AS concept_only_count
3) subclass_rel_count: MATCH ()-[r:SUBCLASS_OF]->() RETURN count(r) AS subclass_rel_count
4) duplicate_anchor_count using this duplicate-anchor logic:
   MATCH (n:!Chunk&!Session&!Document&!`__Community__`) WITH n
   WHERE n.embedding IS NOT NULL AND n.id IS NOT NULL
   WITH n ORDER BY count {{ (n)--() }} DESC, size(toString(n.id)) DESC
   WITH collect(n) AS nodes
   UNWIND nodes AS n
   WITH n, [other IN nodes
      WHERE elementId(n) < elementId(other) AND labels(n) = labels(other)
      AND (
          (size(toString(other.id)) > 2 AND toLower(toString(n.id)) CONTAINS toLower(toString(other.id)))
          OR (size(toString(n.id)) > 2 AND toLower(toString(other.id)) CONTAINS toLower(toString(n.id)))
          OR (size(toString(n.id)) > 5 AND apoc.text.distance(toLower(toString(n.id)), toLower(toString(other.id))) < 3)
          OR vector.similarity.cosine(other.embedding, n.embedding) > 0.97
      )
   ] AS similar
   WHERE size(similar) > 0
   RETURN count(DISTINCT n) AS duplicate_anchor_count

Compute:
- concept_only_ratio = concept_only_count / entity_count (0 if entity_count == 0)
- duplicate_candidate_rate = duplicate_anchor_count / entity_count (0 if entity_count == 0)

Run these additional taxonomy-focused queries:
5) concept_only_without_taxonomy_count:
   MATCH (n:__Entity__:Concept)
   WHERE ALL(l IN labels(n) WHERE l IN ['__Entity__', 'Concept'])
   AND NOT EXISTS {{ (n)-[:SUBCLASS_OF|INSTANCE_OF|IS_A|TYPE_OF]->() }}
   RETURN count(n) AS concept_only_without_taxonomy_count
6) concept_only_degree_le_2_count:
   MATCH (n:__Entity__:Concept)
   WHERE ALL(l IN labels(n) WHERE l IN ['__Entity__', 'Concept'])
   WITH n, count {{ (n)--() }} AS degree
   WHERE degree <= 2
   RETURN count(n) AS concept_only_degree_le_2_count
7) concept_only_degree_le_3_count:
   MATCH (n:__Entity__:Concept)
   WHERE ALL(l IN labels(n) WHERE l IN ['__Entity__', 'Concept'])
   WITH n, count {{ (n)--() }} AS degree
   WHERE degree <= 3
   RETURN count(n) AS concept_only_degree_le_3_count
8) concept_only_with_similarity_or_alias_count:
   MATCH (n:__Entity__:Concept)
   WHERE ALL(l IN labels(n) WHERE l IN ['__Entity__', 'Concept'])
   AND EXISTS {{ (n)-[:SIMILAR_TO|ALIAS_OF|RELATED_TO]-() }}
   RETURN count(n) AS concept_only_with_similarity_or_alias_count
9) focus_examples.high_degree_concept_only:
   MATCH (n:__Entity__:Concept)
   WHERE ALL(l IN labels(n) WHERE l IN ['__Entity__', 'Concept'])
   RETURN n.id AS id, count {{ (n)--() }} AS degree
   ORDER BY degree DESC, id ASC
   LIMIT 10
10) focus_examples.low_degree_concept_only:
    MATCH (n:__Entity__:Concept)
    WHERE ALL(l IN labels(n) WHERE l IN ['__Entity__', 'Concept'])
    RETURN n.id AS id, count {{ (n)--() }} AS degree
    ORDER BY degree ASC, id ASC
    LIMIT 10

Assess consolidation against:
- concept_only_ratio <= {target_concept_ratio}
- duplicate_candidate_rate <= {target_duplicate_rate}
- concept_only_without_taxonomy_ratio <= {target_concept_without_taxonomy_ratio}
- concept_only_with_similarity_or_alias_count == 0

Diagnose the main remaining problem:
- taxonomy_debt: duplicate rate is already acceptable, but concept_only_ratio is still too high or too many remaining concept-only nodes still lack taxonomy edges
- duplicate_debt: duplicate_candidate_rate is the dominant remaining problem
- mixed_debt: both problems are materially above target
- balanced: graph satisfies the combined completion gate or is close enough that neither debt type dominates

Current params:
- tier2: batch_size={current_tier2.batch_size}, sleep_seconds={current_tier2.sleep_seconds}, max_nodes={current_tier2.max_nodes}
- tier3: threshold={current_tier3.threshold}, max_candidates={current_tier3.max_candidates}, max_merges={current_tier3.max_merges}

Current Tier 2 catalog:
{json.dumps(_serialize_catalog(current_catalog), indent=2)}

For proposed_tier2_catalog:
- prefer reusing existing labels where possible
- add labels only when a substantial cluster of concept-only nodes is poorly served by the current catalog
- remove labels only when they are redundant or no longer useful
- use rename_map to converge old labels into better names
- keep the catalog conservative and bounded
- do not add labels just because Tier 2 is under-applying obvious existing labels like Trading Concept, Financial Metric, Condition, Process, Method, Math, or Data Structure
- in the current cleanup phase, focus on guidance updates first because structural catalog changes are ignored while semantic cleanliness is being stabilized

Return ONLY strict JSON with this exact schema:
{{
  "is_consolidated": true|false,
  "diagnosis": "balanced"|"duplicate_debt"|"taxonomy_debt"|"mixed_debt",
  "kpis": {{
    "entity_count": int,
    "concept_only_count": int,
    "concept_only_ratio": float,
    "duplicate_anchor_count": int,
    "duplicate_candidate_rate": float,
    "subclass_rel_count": int
  }},
  "taxonomy_kpis": {{
    "concept_only_without_taxonomy_count": int,
    "concept_only_degree_le_2_count": int,
    "concept_only_degree_le_3_count": int,
    "concept_only_with_similarity_or_alias_count": int
  }},
  "focus_examples": {{
    "high_degree_concept_only": ["id", "..."],
    "low_degree_concept_only": ["id", "..."]
  }},
  "proposed_tier2": {{
    "batch_size": int,
    "sleep_seconds": float,
    "max_nodes": int
  }},
  "proposed_tier3": {{
    "threshold": float,
    "max_candidates": int,
    "max_merges": int
  }},
  "proposed_tier2_catalog": {{
    "labels": ["full", "ordered", "catalog"],
    "add": ["new labels only"],
    "remove": ["dropped labels not covered by rename_map"],
    "rename_map": {{"Old Label": "New Label"}},
    "guidance": {{"Label": "short prompt hint or example"}},
    "rationale": "short text"
  }},
  "rationale": "short text",
  "confidence": float
}}
Do not include markdown fences or any extra text.
""".strip()


def run_codex_review(
    *,
    codex_bin: str,
    llm_routing_config: str | None = None,
    target_concept_ratio: float,
    target_duplicate_rate: float,
    target_concept_without_taxonomy_ratio: float,
    current_tier2: Tier2Params,
    current_tier3: Tier3Params,
    current_catalog: Tier2LabelCatalog,
    iteration_dir: Path,
) -> dict[str, Any]:
    raw_output_file = iteration_dir / "codex_review_raw.txt"
    stdout_log_file = iteration_dir / "codex_review_exec.log"
    agent_role = resolve_agent_role(
        llm_routing_config,
        AGENT_REVIEW_ROLE,
        default_client="codex",
        default_model=None,
        default_executable=codex_bin,
    )
    base_prompt = _build_codex_prompt(
        review_client=agent_role.client,
        target_concept_ratio=target_concept_ratio,
        target_duplicate_rate=target_duplicate_rate,
        target_concept_without_taxonomy_ratio=target_concept_without_taxonomy_ratio,
        current_tier2=current_tier2,
        current_tier3=current_tier3,
        current_catalog=current_catalog,
    )
    last_error: Exception | None = None
    for attempt in range(1, max(CODEX_REVIEW_MAX_ATTEMPTS, 1) + 1):
        prompt = base_prompt
        if last_error is not None:
            retry_suffix = '\n\nIMPORTANT: Previous reply was invalid. Retry with JSON only. Begin with "{" and end with "}".'
            if agent_role.client == "opencode" and len(base_prompt) + len(retry_suffix) > WINDOWS_ARGV_PROMPT_SOFT_LIMIT:
                prompt = base_prompt
            else:
                prompt = f"{base_prompt}{retry_suffix}"
        raw_text = _run_agent_prompt(
            role_config=agent_role,
            prompt=prompt,
            raw_output_file=raw_output_file,
            stdout_log_file=stdout_log_file,
            schema_payload=_review_json_schema(),
        )
        try:
            parsed = parse_codex_review_output(raw_text)
            validated = validate_review_payload(parsed)
            _write_json(iteration_dir / "codex_review.json", validated)
            _write_review_diagnostics(
                iteration_dir=iteration_dir,
                review=validated,
                consolidation_gate_pass=passes_consolidation_gate(
                    validated["kpis"],
                    target_concept_ratio=target_concept_ratio,
                    target_duplicate_rate=target_duplicate_rate,
                ),
                semantic_gate_pass=passes_semantic_gate(
                    validated,
                    target_concept_ratio=target_concept_ratio,
                    target_duplicate_rate=target_duplicate_rate,
                    target_concept_without_taxonomy_ratio=target_concept_without_taxonomy_ratio,
                ),
            )
            return validated
        except (json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
            last_error = exc
            with stdout_log_file.open("a", encoding="utf-8") as handle:
                handle.write(f"\n\nREVIEW_PARSE_ATTEMPT_{attempt}_ERROR:\n{type(exc).__name__}: {exc}\n")
            if attempt >= max(CODEX_REVIEW_MAX_ATTEMPTS, 1):
                break

    assert last_error is not None
    raise last_error


def _initialize_run_state(config: OrchestratorConfig, run_dir: Path) -> dict[str, Any]:
    default_catalog = _default_tier2_catalog()
    return {
        "status": "running",
        "started_at_utc": _utc_now_iso(),
        "config": asdict(config),
        "next_iteration": 0,
        "consecutive_passes": 0,
        "plateau_streak": 0,
        "active_step": "initializing",
        "last_error": None,
        "current_params": {
            "tier2": asdict(Tier2Params()),
            "taxonomy": asdict(TaxonomyParams()),
            "tier3": asdict(Tier3Params()),
        },
        "current_tier2_catalog": _serialize_catalog(default_catalog),
        "last_tier2_catalog_proposal": None,
        "last_tier2_summary": None,
        "last_taxonomy_summary": None,
        "last_tier3_summary": None,
        "last_review": None,
        "last_diagnosis": None,
        "consolidation_gate_pass": None,
        "semantic_gate_pass": None,
        "concept_only_without_taxonomy_ratio": None,
        "iterations": [],
        "run_dir": str(run_dir),
    }


def _prepare_state(state: dict[str, Any], state_path: Path) -> dict[str, Any]:
    state.setdefault("iterations", [])
    state.setdefault(
        "current_params",
        {"tier2": asdict(Tier2Params()), "taxonomy": asdict(TaxonomyParams()), "tier3": asdict(Tier3Params())},
    )
    state["current_params"].setdefault("taxonomy", asdict(TaxonomyParams()))
    state.setdefault("current_tier2_catalog", _serialize_catalog(_default_tier2_catalog()))
    state.setdefault("consecutive_passes", 0)
    state.setdefault("plateau_streak", 0)
    state.setdefault("next_iteration", 0)
    state.setdefault("active_step", "idle")
    state.setdefault("last_error", None)
    state.setdefault("last_review", None)
    state.setdefault("last_diagnosis", None)
    state.setdefault("consolidation_gate_pass", None)
    state.setdefault("semantic_gate_pass", None)
    state.setdefault("concept_only_without_taxonomy_ratio", None)
    state.setdefault("last_tier2_summary", None)
    state.setdefault("last_taxonomy_summary", None)
    state.setdefault("last_tier3_summary", None)
    state["status"] = "running"
    state["last_error"] = None
    state.pop("failed_at_utc", None)
    _write_json(state_path, state)
    return state


def _restore_runtime_state(
    state: dict[str, Any],
) -> tuple[Tier2Params, TaxonomyParams, Tier3Params, Tier2LabelCatalog, int, int, int, dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    tier2 = Tier2Params(**state["current_params"]["tier2"])
    taxonomy = TaxonomyParams(**state["current_params"]["taxonomy"])
    tier3 = Tier3Params(**state["current_params"]["tier3"])
    tier2_catalog = _deserialize_catalog(state.get("current_tier2_catalog"))
    consecutive_passes = int(state.get("consecutive_passes", 0))
    plateau_streak = int(state.get("plateau_streak", 0))
    next_iteration = int(state.get("next_iteration", 0))
    last_tier2_summary = state.get("last_tier2_summary")
    last_taxonomy_summary = state.get("last_taxonomy_summary")
    last_tier3_summary = state.get("last_tier3_summary")
    last_review = state.get("last_review")
    if last_review and state.get("last_diagnosis"):
        last_review = {**last_review, "diagnosis": state["last_diagnosis"]}
    return (
        tier2,
        taxonomy,
        tier3,
        tier2_catalog,
        consecutive_passes,
        plateau_streak,
        next_iteration,
        last_tier2_summary,
        last_taxonomy_summary,
        last_tier3_summary,
        last_review,
    )


def _set_terminal_state(
    state: dict[str, Any],
    *,
    state_path: Path,
    status: str,
    stop_reason: str,
    next_iteration: int,
    consecutive_passes: int,
    plateau_streak: int,
    review: dict[str, Any],
    diagnosis: str,
) -> dict[str, Any]:
    state["status"] = status
    state["completed_at_utc"] = _utc_now_iso()
    state["stop_reason"] = stop_reason
    state["consecutive_passes"] = consecutive_passes
    state["plateau_streak"] = plateau_streak
    state["next_iteration"] = next_iteration
    state["active_step"] = "idle"
    state["last_error"] = None
    state["last_review"] = {**review, "diagnosis": diagnosis}
    state["last_diagnosis"] = diagnosis
    state["consolidation_gate_pass"] = passes_consolidation_gate(
        review["kpis"],
        target_concept_ratio=float(state["config"]["target_concept_ratio"]),
        target_duplicate_rate=float(state["config"]["target_duplicate_rate"]),
    )
    state["semantic_gate_pass"] = passes_semantic_gate(
        review,
        target_concept_ratio=float(state["config"]["target_concept_ratio"]),
        target_duplicate_rate=float(state["config"]["target_duplicate_rate"]),
        target_concept_without_taxonomy_ratio=float(state["config"].get("target_concept_without_taxonomy_ratio", 0.60)),
    )
    state["concept_only_without_taxonomy_ratio"] = _concept_only_without_taxonomy_ratio(review)
    _write_json(state_path, state)
    return state


def _run_initial_iteration(
    *,
    config: OrchestratorConfig,
    run_dir: Path,
    state: dict[str, Any],
    state_path: Path,
    tier2: Tier2Params,
    taxonomy: TaxonomyParams,
    tier3: Tier3Params,
    tier2_catalog: Tier2LabelCatalog,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    iteration_dir = run_dir / "iteration_0"
    iteration_dir.mkdir(parents=True, exist_ok=True)
    iter0 = _get_or_create_iteration_record(state, 0, "initial_tiers")

    if not iter0.get("tier1_completed", False):
        state["active_step"] = "iteration_0_tier1"
        _write_json(state_path, state)
        run_tier1(params=tier2, dry_run=config.dry_run, iteration_dir=iteration_dir)
        iter0["tier1_completed"] = True
        _write_json(state_path, state)

    if "tier2_summary" not in iter0:
        state["active_step"] = "iteration_0_tier2"
        _write_json(state_path, state)
        tier2_kwargs = {
            "params": tier2,
            "catalog": tier2_catalog,
            "dry_run": config.dry_run,
            "iteration_dir": iteration_dir,
        }
        if config.llm_routing_config:
            tier2_kwargs["llm_routing_config"] = config.llm_routing_config
        iter0["tier2_summary"] = run_tier2(**tier2_kwargs)
        iter0["tier2_catalog"] = _serialize_catalog(tier2_catalog)
        _write_json(state_path, state)

    if "taxonomy_summary" not in iter0:
        state["active_step"] = "iteration_0_taxonomy"
        _write_json(state_path, state)
        taxonomy_kwargs = {
            "params": taxonomy,
            "catalog": tier2_catalog,
            "dry_run": config.dry_run,
            "iteration_dir": iteration_dir,
            "prior_taxonomy_decisions_jsonl": None,
            "prior_review_json": None,
        }
        if config.llm_routing_config:
            taxonomy_kwargs["llm_routing_config"] = config.llm_routing_config
        iter0["taxonomy_summary"] = run_taxonomy(**taxonomy_kwargs)
        _write_json(state_path, state)

    _maybe_run_codex_taxonomy_tail(
        config=config,
        state=state,
        state_path=state_path,
        run_dir=run_dir,
        iteration_dir=iteration_dir,
        iteration_record=iter0,
        iteration_index=0,
        taxonomy_params=taxonomy,
        current_catalog=tier2_catalog,
    )

    if "tier3_summary" not in iter0:
        state["active_step"] = "iteration_0_tier3"
        _write_json(state_path, state)
        tier3_kwargs = {
            "params": tier3,
            "dry_run": config.dry_run,
            "iteration_dir": iteration_dir,
        }
        if config.llm_routing_config:
            tier3_kwargs["llm_routing_config"] = config.llm_routing_config
        iter0["tier3_summary"] = run_tier3(**tier3_kwargs)
        _write_json(state_path, state)

    iter0["applied_params"] = {"tier2": asdict(tier2), "taxonomy": asdict(taxonomy), "tier3": asdict(tier3)}
    state["next_iteration"] = 1
    state["current_params"] = {"tier2": asdict(tier2), "taxonomy": asdict(taxonomy), "tier3": asdict(tier3)}
    state["current_tier2_catalog"] = _serialize_catalog(tier2_catalog)
    state["last_tier2_summary"] = iter0["tier2_summary"]
    state["last_taxonomy_summary"] = iter0["taxonomy_summary"]
    state["last_tier3_summary"] = iter0["tier3_summary"]
    state["active_step"] = "idle"
    state["last_error"] = None
    _write_json(state_path, state)
    return iter0["tier2_summary"], iter0["taxonomy_summary"], iter0["tier3_summary"]


def run_self_improving(config: OrchestratorConfig) -> dict[str, Any]:
    run_dir = (
        Path(config.run_dir).expanduser().resolve()
        if config.run_dir
        else (REPO_ROOT / "runs" / f"consolidate_{_timestamp()}").resolve()
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "run_state.json"

    if config.resume and state_path.exists():
        state = _read_json(state_path)
    else:
        state = _initialize_run_state(config=config, run_dir=run_dir)

    if state.get("status") in {"completed", "max_iterations_reached", "plateau_detected"}:
        return state

    state = _prepare_state(state, state_path)
    (
        tier2,
        taxonomy,
        tier3,
        tier2_catalog,
        consecutive_passes,
        plateau_streak,
        next_iteration,
        last_tier2_summary,
        last_taxonomy_summary,
        last_tier3_summary,
        last_review,
    ) = _restore_runtime_state(state)

    try:
        if next_iteration == 0:
            next_iteration = 1
            last_tier2_summary, last_taxonomy_summary, last_tier3_summary = _run_initial_iteration(
                config=config,
                run_dir=run_dir,
                state=state,
                state_path=state_path,
                tier2=tier2,
                taxonomy=taxonomy,
                tier3=tier3,
                tier2_catalog=tier2_catalog,
            )

        while next_iteration <= config.max_iterations:
            iteration_dir = run_dir / f"iteration_{next_iteration}"
            iteration_dir.mkdir(parents=True, exist_ok=True)
            iteration_record = _get_or_create_iteration_record(state, next_iteration, "review")

            if "review" not in iteration_record:
                state["active_step"] = f"iteration_{next_iteration}_review"
                _write_json(state_path, state)
                review_kwargs = {
                    "codex_bin": config.codex_bin,
                    "target_concept_ratio": config.target_concept_ratio,
                    "target_duplicate_rate": config.target_duplicate_rate,
                    "target_concept_without_taxonomy_ratio": config.target_concept_without_taxonomy_ratio,
                    "current_tier2": tier2,
                    "current_tier3": tier3,
                    "current_catalog": tier2_catalog,
                    "iteration_dir": iteration_dir,
                }
                if config.llm_routing_config:
                    review_kwargs["llm_routing_config"] = config.llm_routing_config
                iteration_record["review"] = run_codex_review(**review_kwargs)
                _write_json(state_path, state)

            review = iteration_record["review"]
            effective_diagnosis = classify_failure_mode(
                review=review,
                target_concept_ratio=config.target_concept_ratio,
                target_duplicate_rate=config.target_duplicate_rate,
                target_concept_without_taxonomy_ratio=config.target_concept_without_taxonomy_ratio,
            )
            iteration_record["raw_diagnosis"] = review["diagnosis"]
            iteration_record["diagnosis"] = effective_diagnosis
            consolidation_gate_pass = passes_consolidation_gate(
                review["kpis"],
                target_concept_ratio=config.target_concept_ratio,
                target_duplicate_rate=config.target_duplicate_rate,
            )
            semantic_gate_pass = passes_semantic_gate(
                review,
                target_concept_ratio=config.target_concept_ratio,
                target_duplicate_rate=config.target_duplicate_rate,
                target_concept_without_taxonomy_ratio=config.target_concept_without_taxonomy_ratio,
            )
            concept_only_without_taxonomy_ratio = _concept_only_without_taxonomy_ratio(review)
            _write_review_diagnostics(
                iteration_dir=iteration_dir,
                review=review,
                effective_diagnosis=effective_diagnosis,
                tier2_summary=last_tier2_summary,
                taxonomy_summary=last_taxonomy_summary,
                evidence_iteration_dir=run_dir / f"iteration_{max(next_iteration - 1, 0)}",
                consolidation_gate_pass=consolidation_gate_pass,
                semantic_gate_pass=semantic_gate_pass,
            )

            if {
                "consecutive_passes",
                "gate_pass",
                "consolidation_gate_pass",
                "semantic_gate_pass",
                "concept_only_without_taxonomy_ratio",
            }.issubset(iteration_record):
                consecutive_for_iteration = int(iteration_record["consecutive_passes"])
                gate_pass = bool(iteration_record.get("gate_pass", False))
            else:
                gate_pass = passes_stop_gate(
                    review,
                    target_concept_ratio=config.target_concept_ratio,
                    target_duplicate_rate=config.target_duplicate_rate,
                    target_concept_without_taxonomy_ratio=config.target_concept_without_taxonomy_ratio,
                )
                consecutive_for_iteration = update_consecutive_passes(consecutive_passes, gate_pass)

            plateau_for_iteration = update_plateau_streak(
                previous_review=last_review,
                current_review={**review, "diagnosis": effective_diagnosis},
                current_streak=plateau_streak,
                min_delta=config.taxonomy_plateau_min_delta,
            )
            is_consolidated = gate_pass and consecutive_for_iteration >= config.required_consecutive_passes
            iteration_record["consolidation_gate_pass"] = consolidation_gate_pass
            iteration_record["semantic_gate_pass"] = semantic_gate_pass
            iteration_record["concept_only_without_taxonomy_ratio"] = concept_only_without_taxonomy_ratio
            iteration_record["gate_pass"] = gate_pass
            iteration_record["consecutive_passes"] = consecutive_for_iteration
            iteration_record["plateau_streak"] = plateau_for_iteration
            iteration_record["is_consolidated"] = is_consolidated
            _write_json(state_path, state)

            consecutive_passes = consecutive_for_iteration
            plateau_streak = plateau_for_iteration

            if is_consolidated:
                return _set_terminal_state(
                    state,
                    state_path=state_path,
                    status="completed",
                    stop_reason=COMBINED_STOP_REASON,
                    next_iteration=next_iteration + 1,
                    consecutive_passes=consecutive_passes,
                    plateau_streak=plateau_streak,
                    review=review,
                    diagnosis=effective_diagnosis,
                )

            if effective_diagnosis == "taxonomy_debt" and plateau_streak >= config.taxonomy_plateau_reviews:
                return _set_terminal_state(
                    state,
                    state_path=state_path,
                    status="plateau_detected",
                    stop_reason="taxonomy_debt_plateau",
                    next_iteration=next_iteration + 1,
                    consecutive_passes=consecutive_passes,
                    plateau_streak=plateau_streak,
                    review=review,
                    diagnosis=effective_diagnosis,
                )

            if next_iteration >= config.max_iterations:
                return _set_terminal_state(
                    state,
                    state_path=state_path,
                    status="max_iterations_reached",
                    stop_reason="max_iterations_reached_before_gate",
                    next_iteration=next_iteration + 1,
                    consecutive_passes=consecutive_passes,
                    plateau_streak=plateau_streak,
                    review=review,
                    diagnosis=effective_diagnosis,
                )

            if "applied_params" in iteration_record:
                next_tier2 = Tier2Params(**iteration_record["applied_params"]["tier2"])
                next_taxonomy = TaxonomyParams(**iteration_record["applied_params"].get("taxonomy", asdict(taxonomy)))
                next_tier3 = Tier3Params(**iteration_record["applied_params"]["tier3"])
            else:
                next_tier2, next_tier3 = apply_guardrails(
                    review=review,
                    current_tier2=tier2,
                    current_tier3=tier3,
                    target_concept_ratio=config.target_concept_ratio,
                    target_duplicate_rate=config.target_duplicate_rate,
                    target_concept_without_taxonomy_ratio=config.target_concept_without_taxonomy_ratio,
                    last_tier3_summary=last_tier3_summary,
                )
                next_taxonomy = taxonomy
                iteration_record["applied_params"] = {
                    "tier2": asdict(next_tier2),
                    "taxonomy": asdict(next_taxonomy),
                    "tier3": asdict(next_tier3),
                }
                _write_json(iteration_dir / "applied_params.json", iteration_record["applied_params"])
                _write_json(state_path, state)

            if "tier2_catalog_applied" in iteration_record:
                next_catalog = _deserialize_catalog(iteration_record["tier2_catalog_applied"])
            else:
                live_labels = _fetch_live_entity_labels(next_taxonomy)
                next_catalog, proposal_artifact, applied_artifact = apply_catalog_guardrails(
                    review=review,
                    current_catalog=tier2_catalog,
                    live_labels=live_labels,
                )
                iteration_record["tier2_catalog_proposal"] = proposal_artifact
                iteration_record["tier2_catalog_applied"] = applied_artifact
                _write_json(iteration_dir / "tier2_catalog_proposal.json", proposal_artifact)
                _write_json(iteration_dir / "tier2_catalog_applied.json", applied_artifact)
                _write_json(state_path, state)

            if "tier2_summary" not in iteration_record and not iteration_record.get("tier2_skipped", False):
                if should_run_tier2(
                    review,
                    target_concept_ratio=config.target_concept_ratio,
                ):
                    iteration_record["tier2_skipped"] = False
                    iteration_record.pop("tier2_skip_reason", None)
                    state["current_params"] = {
                        "tier2": asdict(next_tier2),
                        "taxonomy": asdict(next_taxonomy),
                        "tier3": asdict(next_tier3),
                    }
                    state["current_tier2_catalog"] = _serialize_catalog(next_catalog)
                    state["last_tier2_catalog_proposal"] = iteration_record.get("tier2_catalog_proposal")
                    state["active_step"] = f"iteration_{next_iteration}_tier2"
                    _write_json(state_path, state)
                    tier2_kwargs = {
                        "params": next_tier2,
                        "catalog": next_catalog,
                        "dry_run": config.dry_run,
                        "iteration_dir": iteration_dir,
                    }
                    if config.llm_routing_config:
                        tier2_kwargs["llm_routing_config"] = config.llm_routing_config
                    iteration_record["tier2_summary"] = run_tier2(**tier2_kwargs)
                    iteration_record["tier2_catalog"] = _serialize_catalog(next_catalog)
                    _write_json(state_path, state)
                else:
                    iteration_record["tier2_skipped"] = True
                    iteration_record["tier2_skip_reason"] = TIER2_SKIP_REASON
                    iteration_record.pop("tier2_summary", None)
                    iteration_record.pop("tier2_catalog", None)
                    _write_json(state_path, state)

            if "taxonomy_summary" not in iteration_record and not iteration_record.get("taxonomy_skipped", False):
                if should_run_taxonomy(
                    review,
                    target_concept_ratio=config.target_concept_ratio,
                    target_concept_without_taxonomy_ratio=config.target_concept_without_taxonomy_ratio,
                ):
                    iteration_record["taxonomy_skipped"] = False
                    iteration_record.pop("taxonomy_skip_reason", None)
                    state["active_step"] = f"iteration_{next_iteration}_taxonomy"
                    _write_json(state_path, state)
                    taxonomy_kwargs = {
                        "params": next_taxonomy,
                        "catalog": next_catalog,
                        "dry_run": config.dry_run,
                        "iteration_dir": iteration_dir,
                        "prior_taxonomy_decisions_jsonl": str(run_dir / f"iteration_{next_iteration - 1}" / "taxonomy_decisions.jsonl")
                        if (run_dir / f"iteration_{next_iteration - 1}" / "taxonomy_decisions.jsonl").exists()
                        else None,
                        "prior_review_json": str(run_dir / f"iteration_{next_iteration - 1}" / "codex_review.json")
                        if (run_dir / f"iteration_{next_iteration - 1}" / "codex_review.json").exists()
                        else None,
                    }
                    if config.llm_routing_config:
                        taxonomy_kwargs["llm_routing_config"] = config.llm_routing_config
                    iteration_record["taxonomy_summary"] = run_taxonomy(**taxonomy_kwargs)
                    _write_json(state_path, state)
                else:
                    iteration_record["taxonomy_skipped"] = True
                    iteration_record["taxonomy_skip_reason"] = TAXONOMY_SKIP_REASON
                    iteration_record.pop("taxonomy_summary", None)
                    _write_json(state_path, state)

            _maybe_run_codex_taxonomy_tail(
                config=config,
                state=state,
                state_path=state_path,
                run_dir=run_dir,
                iteration_dir=iteration_dir,
                iteration_record=iteration_record,
                iteration_index=next_iteration,
                taxonomy_params=next_taxonomy,
                current_catalog=next_catalog,
            )

            if "tier3_summary" not in iteration_record and not iteration_record.get("tier3_skipped", False):
                if should_run_tier3(
                    {**review, "diagnosis": effective_diagnosis},
                    last_tier3_summary,
                    target_concept_ratio=config.target_concept_ratio,
                    target_duplicate_rate=config.target_duplicate_rate,
                    target_concept_without_taxonomy_ratio=config.target_concept_without_taxonomy_ratio,
                ):
                    iteration_record["tier3_skipped"] = False
                    iteration_record.pop("tier3_skip_reason", None)
                    state["active_step"] = f"iteration_{next_iteration}_tier3"
                    _write_json(state_path, state)
                    tier3_kwargs = {
                        "params": next_tier3,
                        "dry_run": config.dry_run,
                        "iteration_dir": iteration_dir,
                    }
                    if config.llm_routing_config:
                        tier3_kwargs["llm_routing_config"] = config.llm_routing_config
                    iteration_record["tier3_summary"] = run_tier3(**tier3_kwargs)
                    _write_json(state_path, state)
                else:
                    iteration_record["tier3_skipped"] = True
                    iteration_record["tier3_skip_reason"] = "taxonomy_debt"
                    iteration_record.pop("tier3_summary", None)

            iteration_record["phase"] = "review_and_rerun"
            tier2 = next_tier2
            taxonomy = next_taxonomy
            tier3 = next_tier3
            tier2_catalog = next_catalog
            last_tier2_summary = iteration_record.get("tier2_summary", last_tier2_summary)
            last_taxonomy_summary = iteration_record.get("taxonomy_summary", last_taxonomy_summary)
            last_tier3_summary = iteration_record.get("tier3_summary", last_tier3_summary)
            last_review = {**review, "diagnosis": effective_diagnosis}
            next_iteration += 1

            state["current_params"] = {
                "tier2": asdict(tier2),
                "taxonomy": asdict(taxonomy),
                "tier3": asdict(tier3),
            }
            state["current_tier2_catalog"] = _serialize_catalog(tier2_catalog)
            state["last_tier2_catalog_proposal"] = iteration_record.get("tier2_catalog_proposal")
            state["last_tier2_summary"] = last_tier2_summary
            state["last_taxonomy_summary"] = last_taxonomy_summary
            state["last_tier3_summary"] = last_tier3_summary
            state["last_review"] = last_review
            state["last_diagnosis"] = effective_diagnosis
            state["consolidation_gate_pass"] = consolidation_gate_pass
            state["semantic_gate_pass"] = semantic_gate_pass
            state["concept_only_without_taxonomy_ratio"] = concept_only_without_taxonomy_ratio
            state["consecutive_passes"] = consecutive_passes
            state["plateau_streak"] = plateau_streak
            state["next_iteration"] = next_iteration
            state["active_step"] = "idle"
            state["last_error"] = None
            _write_json(state_path, state)

        state["status"] = "max_iterations_reached"
        state["completed_at_utc"] = _utc_now_iso()
        state["stop_reason"] = "loop_exit"
        state["active_step"] = "idle"
        state["last_error"] = None
        _write_json(state_path, state)
        return state
    except Exception as exc:
        state["status"] = "failed"
        state["failed_at_utc"] = _utc_now_iso()
        state["active_step"] = "failed"
        state["last_error"] = {
            "type": exc.__class__.__name__,
            "message": str(exc),
            "iteration": next_iteration,
            "at_utc": _utc_now_iso(),
        }
        _write_json(state_path, state)
        raise


def parse_args() -> OrchestratorConfig:
    parser = argparse.ArgumentParser(description="Self-improving consolidation loop orchestrator.")
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--target-concept-ratio", type=float, default=0.05)
    parser.add_argument("--target-duplicate-rate", type=float, default=0.015)
    parser.add_argument("--target-concept-without-taxonomy-ratio", type=float, default=0.60)
    parser.add_argument("--required-consecutive-passes", type=int, default=2)
    parser.add_argument("--taxonomy-plateau-reviews", type=int, default=2)
    parser.add_argument("--taxonomy-plateau-min-delta", type=float, default=0.002)
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--codex-bin", type=str, default="codex")
    parser.add_argument("--llm-routing-config", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.max_iterations <= 0:
        parser.error("--max-iterations must be > 0")
    if not 0 <= args.target_concept_ratio <= 1:
        parser.error("--target-concept-ratio must be in [0,1]")
    if not 0 <= args.target_duplicate_rate <= 1:
        parser.error("--target-duplicate-rate must be in [0,1]")
    if not 0 <= args.target_concept_without_taxonomy_ratio <= 1:
        parser.error("--target-concept-without-taxonomy-ratio must be in [0,1]")
    if args.required_consecutive_passes <= 0:
        parser.error("--required-consecutive-passes must be > 0")
    if args.taxonomy_plateau_reviews <= 0:
        parser.error("--taxonomy-plateau-reviews must be > 0")
    if args.taxonomy_plateau_min_delta < 0:
        parser.error("--taxonomy-plateau-min-delta must be >= 0")

    return OrchestratorConfig(
        max_iterations=args.max_iterations,
        target_concept_ratio=args.target_concept_ratio,
        target_duplicate_rate=args.target_duplicate_rate,
        target_concept_without_taxonomy_ratio=args.target_concept_without_taxonomy_ratio,
        required_consecutive_passes=args.required_consecutive_passes,
        taxonomy_plateau_reviews=args.taxonomy_plateau_reviews,
        taxonomy_plateau_min_delta=args.taxonomy_plateau_min_delta,
        run_dir=args.run_dir,
        codex_bin=args.codex_bin,
        llm_routing_config=args.llm_routing_config,
        dry_run=args.dry_run,
        resume=args.resume,
    )


def main() -> None:
    config = parse_args()
    state = run_self_improving(config=config)
    print(json.dumps({"status": state["status"], "run_dir": state["run_dir"]}, indent=2))


if __name__ == "__main__":
    main()
