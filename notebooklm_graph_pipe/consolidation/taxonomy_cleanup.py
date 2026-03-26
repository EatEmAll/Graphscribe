"""
Precision-first taxonomy cleanup after Tier 2 relabeling.

This pass audits remaining concept-only nodes and recent Tier 2 relabels, then
applies high-confidence relabel corrections or taxonomy edges to existing nodes
only. It never creates synthetic ontology nodes.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from google import genai
from neo4j import GraphDatabase

from notebooklm_graph_pipe.consolidation.tier2_relabel import load_label_catalog
from notebooklm_graph_pipe.runtime.graph_text_utils import coerce_text, normalize_name, token_set
from notebooklm_graph_pipe.runtime.llm_json_utils import build_single_prompt_clients, generate_json_payload
from notebooklm_graph_pipe.runtime.llm_routing import TAXONOMY_PRIMARY_ROLE, TAXONOMY_SECONDARY_ROLE, PromptRoleConfig, resolve_prompt_role

DEFAULT_NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
DEFAULT_NEO4J_USER = os.environ.get("NEO4J_USERNAME", "neo4j")
DEFAULT_NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password123")
DEFAULT_NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

MODEL_NAME = os.environ.get("TAXONOMY_MODEL_NAME", "gemini-3.1-flash-lite-preview")
SECOND_STAGE_MODEL_NAME = os.environ.get("TAXONOMY_SECOND_STAGE_MODEL_NAME", "gemini-3.1-pro-preview")
LOW_CONFIDENCE_THRESHOLD = float(os.environ.get("TAXONOMY_LOW_CONFIDENCE_THRESHOLD", "0.7"))
APPLY_CONFIDENCE_THRESHOLD = float(os.environ.get("TAXONOMY_APPLY_CONFIDENCE_THRESHOLD", "0.85"))
MODEL_MAX_ATTEMPTS = int(os.environ.get("TAXONOMY_MODEL_MAX_ATTEMPTS", "3"))
MODEL_RETRY_SLEEP_SECONDS = float(os.environ.get("TAXONOMY_MODEL_RETRY_SLEEP_SECONDS", "1.0"))
DEFAULT_MAX_NODES = int(os.environ.get("TAXONOMY_MAX_NODES", "500"))
DEFAULT_CANDIDATE_LIMIT = int(os.environ.get("TAXONOMY_CANDIDATE_LIMIT", "8"))
DEFAULT_EMBEDDING_THRESHOLD = float(os.environ.get("TAXONOMY_EMBEDDING_THRESHOLD", "0.84"))
ALLOWED_OUTPUT_RELATIONS = {"SUBCLASS_OF", "INSTANCE_OF", "TYPE_OF", "NONE"}
ALLOWED_ACTIONS = {"keep_label", "relabel", "add_relation", "none", "blocked", "unresolved"}
STRUCTURAL_BLOCK_REASONS = {
    "Reverse taxonomy relation already exists",
    "Would introduce taxonomy cycle",
    "Existing SUBCLASS_OF target already present",
    "Existing INSTANCE_OF target already present",
    "Existing TYPE_OF target already present",
}


_coerce_text = coerce_text
_normalize_name = normalize_name
_token_set = token_set


def _write_json(path: str | None, payload: dict[str, Any]) -> None:
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _append_jsonl(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _parse_jsonl(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return []
    source = Path(path)
    if not source.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        rows.append(json.loads(text))
    return rows


def _parse_json(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    source = Path(path)
    if not source.exists():
        return {}
    return json.loads(source.read_text(encoding="utf-8"))


def _derive_codex_queue_path(decisions_path: Path | None, summary_json: str | None) -> Path | None:
    if decisions_path is not None:
        return decisions_path.with_name("taxonomy_codex_queue.jsonl")
    if summary_json:
        return Path(summary_json).with_name("taxonomy_codex_queue.jsonl")
    return None


def _normalize_relation(value: str | None) -> str:
    relation = _coerce_text(value).upper()
    if relation == "IS_A":
        return "TYPE_OF"
    return relation if relation in ALLOWED_OUTPUT_RELATIONS else "NONE"


def _current_primary_label(node: dict[str, Any]) -> str:
    for label in node.get("labels", []):
        if label and label != "__Entity__":
            return label
    return "Concept"


def _cosine_similarity(left: list[float] | None, right: list[float] | None) -> float:
    if not left or not right:
        return 0.0
    vec_left = np.asarray(left, dtype=float)
    vec_right = np.asarray(right, dtype=float)
    denom = float(np.linalg.norm(vec_left) * np.linalg.norm(vec_right))
    if denom == 0.0:
        return 0.0
    return float(np.dot(vec_left, vec_right) / denom)


def _is_concept_only(node: dict[str, Any]) -> bool:
    labels = [label for label in node.get("labels", []) if label and label != "__Entity__"]
    return not labels or labels == ["Concept"]


def _taxonomy_count(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _normalize_decision(
    payload: dict[str, Any],
    *,
    node: dict[str, Any],
    label_catalog: dict[str, Any],
    relation_only: bool = False,
) -> dict[str, Any]:
    action = _coerce_text(payload.get("action")).lower()
    if action not in ALLOWED_ACTIONS:
        action = "unresolved"

    current_label = _current_primary_label(node)
    label = _coerce_text(payload.get("label")) or None
    relation = _normalize_relation(payload.get("relation"))
    target_eid = _coerce_text(payload.get("target_eid")) or None

    try:
        confidence = max(0.0, min(float(payload.get("confidence", 0.0)), 1.0))
    except (TypeError, ValueError):
        confidence = 0.0

    if action == "unresolved":
        label = None
        relation = "NONE"
        target_eid = None
    elif action in {"keep_label", "none", "blocked"}:
        if relation_only:
            action = "none"
        label = current_label if action in {"keep_label", "blocked"} else label
        relation = "NONE"
        target_eid = None
    elif action == "relabel":
        if relation_only:
            action = "unresolved"
            label = None
            relation = "NONE"
            target_eid = None
        else:
            relation = "NONE"
            target_eid = None
            if not label:
                label = current_label
    elif action == "add_relation":
        label = current_label
        if relation == "NONE" or not target_eid:
            action = "unresolved"
            label = None
            relation = "NONE"
            target_eid = None

    if label and label not in label_catalog["labels"] and action != "relabel":
        label = current_label

    return {
        "status": "classified" if action != "unresolved" else "unresolved",
        "action": action,
        "label": label,
        "relation": relation,
        "target_eid": target_eid,
        "confidence": confidence,
        "reason": _coerce_text(payload.get("reason")),
    }


def _should_escalate_to_second_stage(
    primary: dict[str, Any],
    *,
    node: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> bool:
    _ = node
    if primary["status"] == "unresolved":
        return bool(candidates)
    if primary["action"] in {"add_relation", "relabel"}:
        return primary["confidence"] < APPLY_CONFIDENCE_THRESHOLD
    return False


def fetch_taxonomy_candidates(
    session,
    *,
    seed_eids: list[str],
    max_nodes: int,
) -> list[dict[str, Any]]:
    query = """
        MATCH (n:__Entity__)
        WHERE
            (
                n:Concept
                AND ALL(label IN labels(n) WHERE label IN ['__Entity__', 'Concept'])
            )
            OR elementId(n) IN $seed_eids
        OPTIONAL MATCH (n)-[r]-(m)
        OPTIONAL MATCH (n)-[tax:SUBCLASS_OF|INSTANCE_OF|TYPE_OF|IS_A]->(tax_target)
        WITH n,
             elementId(n) IN $seed_eids AS is_seed,
             ALL(label IN labels(n) WHERE label IN ['__Entity__', 'Concept']) AS is_concept_only,
             count(DISTINCT r) AS degree,
             count(DISTINCT tax) AS outgoing_taxonomy_count,
             collect(DISTINCT type(r))[0..8] AS relation_types,
             collect(DISTINCT coalesce(toString(m.id), elementId(m)))[0..8] AS neighbor_ids,
             collect(DISTINCT [label IN labels(m) WHERE label <> '__Entity__'])[0..8] AS neighbor_label_sets,
             collect(DISTINCT coalesce(toString(tax_target.id), elementId(tax_target)))[0..4] AS taxonomy_targets
        RETURN DISTINCT
            elementId(n) AS eid,
            n.id AS name,
            n.description AS description,
            [label IN labels(n) WHERE label <> '__Entity__'] AS labels,
            degree,
            outgoing_taxonomy_count,
            relation_types,
            neighbor_ids,
            neighbor_label_sets,
            taxonomy_targets,
            is_concept_only,
            is_seed,
            n.embedding AS embedding
        ORDER BY
            CASE WHEN is_seed THEN 0 WHEN is_concept_only AND outgoing_taxonomy_count = 0 THEN 1 WHEN is_concept_only THEN 2 ELSE 3 END,
            degree DESC,
            toLower(toString(n.id)) ASC
    """
    result = session.run(query, seed_eids=seed_eids)
    return [
        {
            "eid": row["eid"],
            "name": _coerce_text(row["name"]),
            "description": _coerce_text(row["description"]),
            "labels": [label for label in row["labels"] if label],
            "degree": int(row["degree"] or 0),
            "outgoing_taxonomy_count": int(row["outgoing_taxonomy_count"] or 0),
            "relation_types": [value for value in row["relation_types"] if value],
            "neighbor_ids": [value for value in row["neighbor_ids"] if value],
            "neighbor_labels": [
                label
                for labels in row["neighbor_label_sets"]
                if labels
                for label in labels
                if label and label != "__Entity__"
            ][:8],
            "taxonomy_targets": [value for value in row["taxonomy_targets"] if value],
            "is_concept_only": bool(row.get("is_concept_only", False)),
            "is_seed": bool(row.get("is_seed", False)),
            "embedding": row.get("embedding"),
        }
        for row in result
    ]


def fetch_candidate_pool(session) -> list[dict[str, Any]]:
    result = session.run(
        """
        MATCH (n:__Entity__)
        OPTIONAL MATCH (n)-[r]-(m)
        RETURN
            elementId(n) AS eid,
            n.id AS name,
            n.description AS description,
            [label IN labels(n) WHERE label <> '__Entity__'] AS labels,
            collect(DISTINCT coalesce(toString(m.id), elementId(m)))[0..12] AS neighbor_ids,
            collect(DISTINCT [label IN labels(m) WHERE label <> '__Entity__'])[0..8] AS neighbor_label_sets,
            count(DISTINCT r) AS degree,
            count { (n)-[:SUBCLASS_OF|INSTANCE_OF|TYPE_OF|IS_A]->() } AS outgoing_taxonomy_count,
            count { ()-[:SUBCLASS_OF|INSTANCE_OF|TYPE_OF|IS_A]->(n) } AS incoming_taxonomy_target_count,
            n.embedding AS embedding
        """
    )
    return [
        {
            "eid": row["eid"],
            "name": _coerce_text(row["name"]),
            "description": _coerce_text(row["description"]),
            "labels": [label for label in row["labels"] if label],
            "neighbor_ids": [value for value in row["neighbor_ids"] if value],
            "neighbor_labels": [
                label
                for labels in row["neighbor_label_sets"]
                if labels
                for label in labels
                if label and label != "__Entity__"
            ][:8],
            "degree": int(row["degree"] or 0),
            "outgoing_taxonomy_count": int(row["outgoing_taxonomy_count"] or 0),
            "incoming_taxonomy_target_count": int(row["incoming_taxonomy_target_count"] or 0),
            "embedding": row.get("embedding"),
        }
        for row in result
    ]


def fetch_parent_candidates(
    session,
    *,
    node: dict[str, Any],
    candidate_pool: list[dict[str, Any]],
    compatible_labels: list[str],
    candidate_limit: int,
    embedding_threshold: float,
) -> list[dict[str, Any]]:
    _ = session
    source_name = _normalize_name(node.get("name", ""))
    source_tokens = _token_set(node.get("name", ""))
    source_neighbors = set(node.get("neighbor_ids", []))
    compatible = set(compatible_labels)
    shortlist: list[dict[str, Any]] = []

    for candidate in candidate_pool:
        if candidate["eid"] == node["eid"]:
            continue
        lexical_overlap = 0
        candidate_name = _normalize_name(candidate.get("name", ""))
        candidate_tokens = _token_set(candidate.get("name", ""))
        if source_name and candidate_name and (
            source_name == candidate_name
            or source_name in candidate_name
            or candidate_name in source_name
            or (source_tokens and candidate_tokens and source_tokens <= candidate_tokens)
            or (source_tokens and candidate_tokens and candidate_tokens <= source_tokens)
        ):
            lexical_overlap = 1
        shared_neighbors = len(source_neighbors & set(candidate.get("neighbor_ids", [])))
        compatible_label_count = len(set(candidate.get("labels", [])) & compatible)
        embedding_similarity = _cosine_similarity(node.get("embedding"), candidate.get("embedding"))
        if (
            lexical_overlap == 0
            and shared_neighbors == 0
            and compatible_label_count == 0
            and embedding_similarity < embedding_threshold
        ):
            continue
        shortlist.append(
            {
                "eid": candidate["eid"],
                "name": candidate["name"],
                "description": candidate["description"],
                "labels": list(candidate.get("labels", [])),
                "lexical_overlap": lexical_overlap,
                "shared_neighbors": shared_neighbors,
                "compatible_label_count": compatible_label_count,
                "candidate_has_outgoing_taxonomy": int(candidate.get("outgoing_taxonomy_count", 0) > 0),
                "candidate_is_taxonomy_target": int(candidate.get("incoming_taxonomy_target_count", 0) > 0),
                "embedding_similarity": round(embedding_similarity, 4),
                "candidate_degree": int(candidate.get("degree", 0) or 0),
            }
        )

    shortlist.sort(
        key=lambda row: (
            -int(row["lexical_overlap"]),
            -int(row["shared_neighbors"]),
            -int(row["compatible_label_count"]),
            -int(row["candidate_has_outgoing_taxonomy"]),
            -int(row["candidate_is_taxonomy_target"]),
            -float(row["embedding_similarity"]),
            -int(row["candidate_degree"]),
            row["name"].casefold(),
        )
    )
    return shortlist[:candidate_limit]


def _seed_eids_from_prior_taxonomy(decisions: list[dict[str, Any]]) -> list[str]:
    seed_eids: list[str] = []
    for row in decisions:
        eid = _coerce_text(row.get("eid"))
        if not eid:
            continue
        taxonomy_count_after = _taxonomy_count(
            row.get("outgoing_taxonomy_count_after", row.get("outgoing_taxonomy_count", 0))
        )
        if (
            row.get("status") == "unresolved"
            or row.get("suspicious")
            or _coerce_text(row.get("skipped_reason"))
            or (bool(row.get("applied")) and taxonomy_count_after == 0)
        ):
            seed_eids.append(eid)
    seen: set[str] = set()
    ordered: list[str] = []
    for eid in seed_eids:
        if eid in seen:
            continue
        seen.add(eid)
        ordered.append(eid)
    return ordered


def _review_focus_name_set(review_payload: dict[str, Any]) -> set[str]:
    if not review_payload:
        return set()
    focus_examples = review_payload.get("focus_examples", {})
    names = [
        _normalize_name(_coerce_text(name))
        for bucket in ("high_degree_concept_only", "low_degree_concept_only")
        for name in focus_examples.get(bucket, [])
    ]
    return {name for name in names if name}


def _prior_carry_forward_counts(decisions: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in decisions:
        eid = _coerce_text(row.get("eid"))
        if not eid:
            continue
        if not (
            row.get("queue_reason_code")
            or row.get("action") == "blocked"
            or _taxonomy_count(row.get("outgoing_taxonomy_count_after", row.get("outgoing_taxonomy_count", 0))) == 0
        ):
            continue
        try:
            counts[eid] = max(counts.get(eid, 0), int(row.get("carry_forward_count", 0) or 0))
        except (TypeError, ValueError):
            counts[eid] = max(counts.get(eid, 0), 0)
    return counts


def _candidate_snapshot(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "eid": candidate["eid"],
            "name": candidate["name"],
            "labels": list(candidate.get("labels", [])),
            "description": candidate.get("description"),
            "lexical_overlap": int(candidate.get("lexical_overlap", 0) or 0),
            "shared_neighbors": int(candidate.get("shared_neighbors", 0) or 0),
            "compatible_label_count": int(candidate.get("compatible_label_count", 0) or 0),
            "candidate_has_outgoing_taxonomy": int(candidate.get("candidate_has_outgoing_taxonomy", 0) or 0),
            "candidate_is_taxonomy_target": int(candidate.get("candidate_is_taxonomy_target", 0) or 0),
            "embedding_similarity": float(candidate.get("embedding_similarity", 0.0) or 0.0),
            "candidate_degree": int(candidate.get("candidate_degree", 0) or 0),
        }
        for candidate in candidates
    ]


def _queue_reason_from_decision(
    *,
    final_action: str,
    skipped_reason: str,
    node: dict[str, Any],
    outgoing_taxonomy_count_after: int,
    applied: bool,
    follow_up_applied: bool,
) -> tuple[str | None, str | None]:
    if final_action == "blocked":
        return "blocked", skipped_reason or "Taxonomy action is structurally blocked."
    if final_action == "none" and _is_concept_only(node) and outgoing_taxonomy_count_after == 0:
        return "none_without_taxonomy", "Concept-only residual remains without taxonomy support."
    if skipped_reason in STRUCTURAL_BLOCK_REASONS:
        return "structural_block", skipped_reason
    if applied and not follow_up_applied and outgoing_taxonomy_count_after == 0:
        return "post_relabel_without_taxonomy", "Relabel applied but taxonomy attachment is still missing."
    return None, None


def _prioritize_taxonomy_nodes(
    nodes: list[dict[str, Any]],
    *,
    tier2_seed_eids: set[str],
    carry_forward_seed_eids: set[str],
    review_focus_names: set[str],
    max_nodes: int,
) -> tuple[list[dict[str, Any]], int, int, int]:
    prioritized: list[dict[str, Any]] = []
    for node in nodes:
        is_concept_only = _is_concept_only(node)
        no_taxonomy = int(node.get("outgoing_taxonomy_count", 0)) == 0
        normalized_name = _normalize_name(node.get("name", ""))
        is_carry_forward_seed = node["eid"] in carry_forward_seed_eids
        is_tier2_seed = node["eid"] in tier2_seed_eids
        is_review_focus_seed = is_concept_only and normalized_name in review_focus_names
        if is_concept_only and not no_taxonomy and not (
            is_carry_forward_seed or is_tier2_seed or is_review_focus_seed
        ):
            continue
        if is_concept_only and no_taxonomy:
            bucket = 0
        elif is_carry_forward_seed:
            bucket = 1
        elif is_tier2_seed:
            bucket = 2
        elif is_review_focus_seed:
            bucket = 3
        else:
            bucket = 4
        prioritized.append(
            {
                **node,
                "is_tier2_seed": is_tier2_seed,
                "is_carry_forward_seed": is_carry_forward_seed,
                "is_review_focus_seed": is_review_focus_seed,
                "residual_priority_bucket": bucket,
            }
        )
    prioritized.sort(
        key=lambda node: (
            int(node["residual_priority_bucket"]),
            -int(node.get("degree", 0) or 0),
            _normalize_name(node.get("name", "")),
        )
    )
    limited = prioritized[:max_nodes]
    residual_seed_count = sum(
        1
        for node in limited
        if node.get("is_tier2_seed") or node.get("is_carry_forward_seed") or node.get("is_review_focus_seed")
    )
    carry_forward_seed_count = sum(1 for node in limited if node.get("is_carry_forward_seed"))
    review_focus_seed_count = sum(1 for node in limited if node.get("is_review_focus_seed"))
    return limited, residual_seed_count, carry_forward_seed_count, review_focus_seed_count


def build_system_prompt(label_catalog: dict[str, Any], *, relation_only: bool = False) -> str:
    labels = label_catalog["labels"]
    guidance = label_catalog.get("preferred_examples", {})
    guidance_lines = "\n".join(
        f"- {label}: {guidance.get(label, 'Use only when it is the most specific existing fit.')}"
        for label in labels
    )
    actions_block = (
        "Choose exactly one action:\n"
        "- add_relation: keep current label and add exactly one taxonomy relation to one provided target candidate\n"
        "- blocked: no safe write should be attempted because the case is structurally or semantically blocked for this pass\n"
        "- none: do nothing because evidence is weak\n"
        "- unresolved: response cannot be trusted or the candidates are insufficient\n\n"
        if relation_only
        else
        "Choose exactly one action:\n"
        "- keep_label: current label already looks correct; do not add a taxonomy edge\n"
        "- relabel: current label looks wrong; choose a better existing label from the catalog\n"
        "- add_relation: keep current label and add exactly one taxonomy relation to one provided target candidate\n"
        "- blocked: semantically meaningful but not safe to write automatically in this pass\n"
        "- none: do nothing because evidence is weak\n"
        "- unresolved: response cannot be trusted or the candidates are insufficient\n\n"
    )
    relation_only_rules = (
        "- The label has already been corrected for this node in a previous step of the same pass.\n"
        "- Do not relabel again; only decide whether to add one taxonomy relation.\n"
        "- If no broader parent is clearly defensible, prefer none.\n"
        if relation_only
        else ""
    )
    return (
        "You are a precision-first knowledge graph taxonomy expert.\n"
        "Your job is to improve semantic cleanliness without inventing ontology nodes.\n\n"
        "Allowed labels:\n"
        f"{labels}\n\n"
        "Label guidance:\n"
        f"{guidance_lines}\n\n"
        f"{actions_block}"
        "Relation semantics:\n"
        "- SUBCLASS_OF: class/concept to broader class/concept\n"
        "- INSTANCE_OF: concrete named instance to class/concept\n"
        "- TYPE_OF: fallback type-membership only when subclass vs instance is unclear\n\n"
        "Rules:\n"
        '- Return strict JSON only: {"action":"...","label":"...","relation":"...","target_eid":"...","confidence":0.0,"reason":"..."}\n'
        f"{relation_only_rules}"
        "- Do not invent labels.\n"
        "- Use only the provided candidate target_eid values.\n"
        "- Prefer no write over a questionable write.\n"
        "- For keep_label, none, or blocked, relation must be NONE and target_eid must be null.\n"
        "- For add_relation, relation must be SUBCLASS_OF, INSTANCE_OF, or TYPE_OF and target_eid must be one of the provided candidates.\n"
        "- For SUBCLASS_OF, INSTANCE_OF, and TYPE_OF, the target must be broader than the source, never a narrower specialization of it.\n"
        "- Do not use keep_label as a default escape hatch.\n"
        "- If the current label is Concept and the node has no taxonomy edge, choose keep_label only when the node is genuinely too broad for a better label and none of the candidates is a defensible broader parent.\n"
        "- If the candidates are weak or ambiguous, prefer none or unresolved over an overconfident keep_label.\n"
        "- Reserve confidence >= 0.95 for cases with explicit lexical or definitional evidence.\n"
        "- No markdown, no code fences, no commentary, no extra keys.\n"
    )


def _build_prompt(node: dict[str, Any], candidates: list[dict[str, Any]], *, relation_only: bool = False) -> str:
    candidate_lines = []
    for candidate in candidates:
        candidate_lines.append(
            (
                f'- eid={candidate["eid"]}; id="{candidate["name"]}"; '
                f'labels={candidate["labels"] or ["Concept"]}; '
                f'lexical_overlap={candidate.get("lexical_overlap", 0)}; '
                f'shared_neighbors={candidate.get("shared_neighbors", 0)}; '
                f'candidate_has_outgoing_taxonomy={candidate.get("candidate_has_outgoing_taxonomy", 0)}; '
                f'candidate_is_taxonomy_target={candidate.get("candidate_is_taxonomy_target", 0)}; '
                f'embedding_similarity={candidate.get("embedding_similarity", 0.0)}; '
                f'candidate_degree={candidate.get("candidate_degree", 0)}; '
                f'description="{candidate["description"] or "No description."}"'
            )
        )
    return "\n".join(
        [
            f'Entity eid: {node["eid"]}',
            f'Entity id: "{node["name"]}"',
            f'Entity description: "{node["description"] or "No description."}"',
            f'Current labels: {", ".join(node["labels"]) if node["labels"] else "Concept"}',
            f'Concept-only node: {"yes" if _is_concept_only(node) else "no"}',
            f'Degree: {node["degree"]}',
            f'Neighbor labels: {", ".join(node["neighbor_labels"]) if node["neighbor_labels"] else "None"}',
            f'Relation types: {", ".join(node["relation_types"]) if node["relation_types"] else "None"}',
            f'Outgoing taxonomy targets: {", ".join(node["taxonomy_targets"]) if node["taxonomy_targets"] else "None"}',
            f'Follow-up after relabel: {"yes" if relation_only else "no"}',
            "Candidate taxonomy targets:",
            *candidate_lines,
            "Decision guidance:",
            "- Use add_relation when one candidate is clearly the broader parent or type.",
            *(
                []
                if relation_only
                else [
                    "- Use keep_label only when the current label is already the best catalog label and no candidate is a defensible parent.",
                    "- Use blocked when the node looks meaningful but this pass should defer a safe write.",
                ]
            ),
            *(
                [
                    "- If the entity is still just Concept and the evidence is weak, prefer none or unresolved instead of high-confidence keep_label.",
                ]
                if not relation_only
                else [
                    "- The label has already been corrected; only add a relation when one broader parent is clearly defensible.",
                ]
            ),
            'Return JSON: {"action":"...","label":"...","relation":"...","target_eid":"...","confidence":0.0,"reason":"..."}',
        ]
    )


def _classify_once(
    clients: dict[str, Any] | Any,
    *,
    role_config: PromptRoleConfig | None = None,
    model_name: str | None = None,
    label_catalog: dict[str, Any],
    node: dict[str, Any],
    candidates: list[dict[str, Any]],
    stage: str = "primary",
    relation_only: bool = False,
) -> dict[str, Any]:
    if role_config is None:
        if not model_name:
            raise ValueError("role_config or model_name is required.")
        role_config = PromptRoleConfig(client="genai", model=model_name)
    prompt = _build_prompt(node, candidates, relation_only=relation_only)
    payload, error_message = generate_json_payload(
        _resolve_client(clients, role_config.client),
        client_name=role_config.client,
        model_name=role_config.model,
        prompt=prompt,
        system_instruction=build_system_prompt(label_catalog, relation_only=relation_only),
        max_output_tokens=220,
        temperature=0.0,
        max_attempts=max(MODEL_MAX_ATTEMPTS, 1),
        retry_sleep_seconds=MODEL_RETRY_SLEEP_SECONDS,
    )
    if payload is None:
        return {
            "status": "unresolved",
            "action": "unresolved",
            "label": None,
            "relation": "NONE",
            "target_eid": None,
            "confidence": 0.0,
            "reason": error_message or "Invalid or empty JSON response",
            "client_name": role_config.client,
            "model_name": role_config.model,
            "stage": stage,
        }

    normalized = _normalize_decision(
        payload,
        node=node,
        label_catalog=label_catalog,
        relation_only=relation_only,
    )
    return {
        **normalized,
        "client_name": role_config.client,
        "model_name": role_config.model,
        "stage": stage,
    }


def classify_taxonomy_action(
    clients: dict[str, Any] | Any,
    *,
    label_catalog: dict[str, Any],
    node: dict[str, Any],
    candidates: list[dict[str, Any]],
    primary_role_config: PromptRoleConfig | None = None,
    secondary_role_config: PromptRoleConfig | None = None,
    relation_only: bool = False,
) -> dict[str, Any]:
    primary_role = primary_role_config or resolve_prompt_role(
        None,
        TAXONOMY_PRIMARY_ROLE,
        default_client="genai",
        default_model=MODEL_NAME,
    )
    secondary_role = secondary_role_config or resolve_prompt_role(
        None,
        TAXONOMY_SECONDARY_ROLE,
        default_client="genai",
        default_model=SECOND_STAGE_MODEL_NAME,
    )
    primary = _classify_once(
        clients,
        role_config=primary_role,
        label_catalog=label_catalog,
        node=node,
        candidates=candidates,
        stage="primary",
        relation_only=relation_only,
    )
    needs_second_stage = _should_escalate_to_second_stage(primary, node=node, candidates=candidates)
    if not needs_second_stage:
        return {
            **primary,
            "used_second_stage": False,
            "attempted_second_stage": False,
        }

    secondary = _classify_once(
        clients,
        role_config=secondary_role,
        label_catalog=label_catalog,
        node=node,
        candidates=candidates,
        stage="second_stage",
        relation_only=relation_only,
    )
    if secondary["status"] == "classified":
        return {
            **secondary,
            "used_second_stage": True,
            "attempted_second_stage": True,
        }
    if primary["status"] == "classified" and primary["action"] in {"keep_label", "none"}:
        return {
            **primary,
            "used_second_stage": False,
            "attempted_second_stage": True,
        }
    return {
        **primary,
        "status": "unresolved",
        "action": "unresolved",
        "label": None,
        "relation": "NONE",
        "target_eid": None,
        "confidence": max(primary["confidence"], secondary["confidence"]),
        "reason": secondary["reason"] or primary["reason"] or "Low confidence taxonomy decision",
        "client_name": secondary["client_name"],
        "model_name": secondary["model_name"],
        "stage": "second_stage",
        "used_second_stage": False,
        "attempted_second_stage": True,
    }


def _call_classify_taxonomy_action(
    clients: dict[str, Any] | Any,
    *,
    label_catalog: dict[str, Any],
    node: dict[str, Any],
    candidates: list[dict[str, Any]],
    primary_role_config: PromptRoleConfig,
    secondary_role_config: PromptRoleConfig,
    relation_only: bool = False,
) -> dict[str, Any]:
    signature = inspect.signature(classify_taxonomy_action)
    supports_kwargs = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())
    supported_names = set(signature.parameters)
    kwargs: dict[str, Any] = {
        "label_catalog": label_catalog,
        "node": node,
        "candidates": candidates,
    }
    if supports_kwargs or "primary_role_config" in supported_names:
        kwargs["primary_role_config"] = primary_role_config
    if supports_kwargs or "secondary_role_config" in supported_names:
        kwargs["secondary_role_config"] = secondary_role_config
    if relation_only and (supports_kwargs or "relation_only" in supported_names):
        kwargs["relation_only"] = relation_only
    return classify_taxonomy_action(clients, **kwargs)


def _resolve_client(clients: dict[str, Any] | Any, client_name: str) -> Any:
    if isinstance(clients, dict):
        return clients[client_name]
    return clients


def _label_is_allowed(label: str | None, label_catalog: dict[str, Any]) -> bool:
    return bool(label) and label in label_catalog["labels"]


def _candidate_lookup(candidates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {candidate["eid"]: candidate for candidate in candidates}


def _relation_direction_is_plausible(
    *,
    node: dict[str, Any],
    candidate: dict[str, Any],
    relation: str,
) -> tuple[bool, str]:
    if relation not in {"SUBCLASS_OF", "INSTANCE_OF", "TYPE_OF"}:
        return False, "Unsupported relation"

    source_name = _normalize_name(node.get("name", ""))
    target_name = _normalize_name(candidate.get("name", ""))
    source_tokens = _token_set(node.get("name", ""))
    target_tokens = _token_set(candidate.get("name", ""))

    if source_name and target_name and source_name == target_name:
        return False, "Target name matches source name"
    if source_tokens and target_tokens and source_tokens < target_tokens:
        return False, "Target appears lexically narrower than source"
    if source_name and target_name and target_name.startswith(f"{source_name} "):
        return False, "Target appears to specialize the source"

    return True, ""


def can_apply_relation(session, *, source_eid: str, relation: str, target_eid: str) -> tuple[bool, str]:
    if relation not in {"SUBCLASS_OF", "INSTANCE_OF", "TYPE_OF"}:
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


def apply_label(session, *, source_eid: str, old_labels: list[str], new_label: str) -> None:
    safe_new_label = new_label.replace("`", "")
    removal_clauses = []
    for old_label in old_labels:
        if old_label in {"__Entity__", safe_new_label}:
            continue
        safe_old_label = old_label.replace("`", "")
        removal_clauses.append(f"REMOVE n:`{safe_old_label}`")
    removal_block = "\n        ".join(removal_clauses)
    session.run(
        f"""
        MATCH (n) WHERE elementId(n) = $source_eid
        SET n:`{safe_new_label}`
        {removal_block}
        """,
        source_eid=source_eid,
    )


def add_relation(session, *, source_eid: str, relation: str, target_eid: str) -> None:
    session.run(
        f"""
        MATCH (source) WHERE elementId(source) = $source_eid
        MATCH (target) WHERE elementId(target) = $target_eid
        MERGE (source)-[:{relation}]->(target)
        """,
        source_eid=source_eid,
        target_eid=target_eid,
    )


def _apply_relation_decision(
    session,
    *,
    node: dict[str, Any],
    relation: str | None,
    target_eid: str | None,
    confidence: float,
    candidate_by_eid: dict[str, dict[str, Any]],
    dry_run: bool,
) -> dict[str, Any]:
    normalized_relation = _normalize_relation(relation)
    normalized_target_eid = _coerce_text(target_eid) or None
    result = {
        "applied": False,
        "skipped_reason": "",
        "relation": normalized_relation,
        "target_eid": normalized_target_eid,
    }
    if normalized_relation not in {"SUBCLASS_OF", "INSTANCE_OF", "TYPE_OF"}:
        result["skipped_reason"] = "Invalid relation"
        return result
    if normalized_target_eid not in candidate_by_eid:
        result["skipped_reason"] = "Target candidate not in provided list"
        return result
    if confidence < APPLY_CONFIDENCE_THRESHOLD:
        result["skipped_reason"] = "Relation confidence below apply threshold"
        return result

    target_candidate = candidate_by_eid[normalized_target_eid]
    plausible, skipped_reason = _relation_direction_is_plausible(
        node=node,
        candidate=target_candidate,
        relation=normalized_relation,
    )
    if not plausible:
        result["skipped_reason"] = skipped_reason
        return result

    allowed, skipped_reason = can_apply_relation(
        session,
        source_eid=node["eid"],
        relation=normalized_relation,
        target_eid=normalized_target_eid,
    )
    if not allowed:
        result["skipped_reason"] = skipped_reason
        return result

    if not dry_run:
        add_relation(
            session,
            source_eid=node["eid"],
            relation=normalized_relation,
            target_eid=normalized_target_eid,
        )
    result["applied"] = True
    return result


def _seed_eids_from_tier2(decisions: list[dict[str, Any]]) -> list[str]:
    seed_eids: list[str] = []
    for row in decisions:
        if not row.get("eid"):
            continue
        if row.get("unresolved"):
            seed_eids.append(str(row["eid"]))
            continue
        new_label = _coerce_text(row.get("new_label"))
        old_label = _coerce_text(row.get("old_label"))
        confidence = float(row.get("confidence", 0.0) or 0.0)
        if new_label and new_label != old_label:
            seed_eids.append(str(row["eid"]))
            continue
        if confidence < LOW_CONFIDENCE_THRESHOLD or row.get("used_second_stage"):
            seed_eids.append(str(row["eid"]))
    seen: set[str] = set()
    deduped: list[str] = []
    for eid in seed_eids:
        if eid in seen:
            continue
        seen.add(eid)
        deduped.append(eid)
    return deduped


def _compatible_labels(node: dict[str, Any]) -> list[str]:
    return sorted(set(node.get("labels", []) + node.get("neighbor_labels", [])))


def _node_after_relabel(node: dict[str, Any], new_label: str) -> dict[str, Any]:
    updated = dict(node)
    updated["labels"] = [new_label]
    return updated


def run(
    *,
    dry_run: bool,
    max_nodes: int,
    candidate_limit: int,
    embedding_threshold: float,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    neo4j_database: str,
    labels_json: str | None,
    tier2_decisions_jsonl: str | None,
    prior_taxonomy_decisions_jsonl: str | None,
    prior_review_json: str | None,
    summary_json: str | None,
    decisions_jsonl: str | None,
    llm_routing_config: str | None = None,
) -> dict[str, Any]:
    primary_role_config = resolve_prompt_role(
        llm_routing_config,
        TAXONOMY_PRIMARY_ROLE,
        default_client="genai",
        default_model=MODEL_NAME,
    )
    secondary_role_config = resolve_prompt_role(
        llm_routing_config,
        TAXONOMY_SECONDARY_ROLE,
        default_client="genai",
        default_model=SECOND_STAGE_MODEL_NAME,
    )
    label_catalog = load_label_catalog(labels_json)
    tier2_decisions = _parse_jsonl(tier2_decisions_jsonl)
    prior_taxonomy_decisions = _parse_jsonl(prior_taxonomy_decisions_jsonl)
    prior_review = _parse_json(prior_review_json)
    decisions_path = Path(decisions_jsonl) if decisions_jsonl else None
    codex_queue_path = _derive_codex_queue_path(decisions_path, summary_json)
    if decisions_path is not None and decisions_path.exists():
        decisions_path.unlink()
    if codex_queue_path is not None and codex_queue_path.exists():
        codex_queue_path.unlink()

    clients = build_single_prompt_clients(primary_role_config.client, secondary_role_config.client)
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    started = time.time()
    summary: dict[str, Any]

    try:
        with driver.session(database=neo4j_database) as session:
            tier2_seed_eids = _seed_eids_from_tier2(tier2_decisions)
            carry_forward_seed_eids = _seed_eids_from_prior_taxonomy(prior_taxonomy_decisions)
            review_focus_names = _review_focus_name_set(prior_review)
            raw_nodes = fetch_taxonomy_candidates(
                session,
                seed_eids=list(dict.fromkeys(tier2_seed_eids + carry_forward_seed_eids)),
                max_nodes=max_nodes,
            )
            nodes, residual_seed_count, carry_forward_seed_count, review_focus_seed_count = _prioritize_taxonomy_nodes(
                raw_nodes,
                tier2_seed_eids=set(tier2_seed_eids),
                carry_forward_seed_eids=set(carry_forward_seed_eids),
                review_focus_names=review_focus_names,
                max_nodes=max_nodes,
            )
            candidate_pool = fetch_candidate_pool(session)
            prior_carry_forward_counts = _prior_carry_forward_counts(prior_taxonomy_decisions)
            print(f"Fetched {len(nodes)} taxonomy candidates")

            processed_nodes = 0
            relabels_applied = 0
            unresolved = 0
            second_stage_decisions = 0
            second_stage_attempts = 0
            suspicious_relabels = 0
            suspicious_keep_label_concepts = 0
            relabeled_without_taxonomy_support = 0
            post_relabel_relation_attempts = 0
            post_relabel_relations_added = 0
            actions = {"keep_label": 0, "relabel": 0, "add_relation": 0, "none": 0, "blocked": 0, "unresolved": 0}
            relations_added = {"SUBCLASS_OF": 0, "INSTANCE_OF": 0, "TYPE_OF": 0}
            codex_queue_count = 0

            for node in nodes:
                compatible_labels = _compatible_labels(node)
                candidates = fetch_parent_candidates(
                    session,
                    node=node,
                    candidate_pool=candidate_pool,
                    compatible_labels=compatible_labels,
                    candidate_limit=candidate_limit,
                    embedding_threshold=embedding_threshold,
                )
                decision = _call_classify_taxonomy_action(
                    clients,
                    label_catalog=label_catalog,
                    node=node,
                    candidates=candidates,
                    primary_role_config=primary_role_config,
                    secondary_role_config=secondary_role_config,
                )
                processed_nodes += 1
                if decision.get("attempted_second_stage"):
                    second_stage_attempts += 1
                if decision.get("used_second_stage"):
                    second_stage_decisions += 1

                candidate_by_eid = _candidate_lookup(candidates)
                applied = False
                skipped_reason = ""
                decision_relation = decision["relation"]
                decision_target_eid = decision["target_eid"]
                follow_up_action = None
                follow_up_relation = "NONE"
                follow_up_target_eid = None
                follow_up_confidence = 0.0
                follow_up_reason = ""
                follow_up_applied = False
                follow_up_candidates: list[dict[str, Any]] = []
                final_action = decision["action"] if decision["status"] == "classified" else "unresolved"
                queue_node = node
                queue_candidates = candidates

                if decision["status"] == "unresolved":
                    unresolved += 1
                elif (
                    decision["action"] == "keep_label"
                    and _is_concept_only(node)
                    and int(node["outgoing_taxonomy_count"]) == 0
                    and decision["confidence"] >= 0.95
                ):
                    suspicious_keep_label_concepts += 1
                elif decision["action"] == "relabel":
                    if not _label_is_allowed(decision["label"], label_catalog):
                        skipped_reason = "Proposed label not in active catalog"
                        suspicious_relabels += 1
                    elif decision["label"] == "Concept":
                        skipped_reason = "Relabeling to Concept is not useful in taxonomy cleanup"
                        suspicious_relabels += 1
                    elif decision["confidence"] < APPLY_CONFIDENCE_THRESHOLD:
                        skipped_reason = "Relabel confidence below apply threshold"
                        suspicious_relabels += 1
                    else:
                        if not dry_run:
                            apply_label(
                                session,
                                source_eid=node["eid"],
                                old_labels=list(node.get("labels", [])),
                                new_label=decision["label"],
                            )
                        relabels_applied += 1
                        applied = True
                        if int(node["outgoing_taxonomy_count"]) == 0:
                            post_relabel_relation_attempts += 1
                            updated_node = _node_after_relabel(node, decision["label"])
                            queue_node = updated_node
                            follow_up_candidates = fetch_parent_candidates(
                                session,
                                node=updated_node,
                                candidate_pool=candidate_pool,
                                compatible_labels=_compatible_labels(updated_node),
                                candidate_limit=candidate_limit,
                                embedding_threshold=embedding_threshold,
                            )
                            follow_up = _call_classify_taxonomy_action(
                                clients,
                                label_catalog=label_catalog,
                                node=updated_node,
                                candidates=follow_up_candidates,
                                primary_role_config=primary_role_config,
                                secondary_role_config=secondary_role_config,
                                relation_only=True,
                            )
                            follow_up_action = follow_up["action"]
                            follow_up_confidence = round(float(follow_up["confidence"]), 4)
                            follow_up_reason = follow_up["reason"]
                            if follow_up.get("attempted_second_stage"):
                                second_stage_attempts += 1
                            if follow_up.get("used_second_stage"):
                                second_stage_decisions += 1
                            follow_up_candidate_by_eid = _candidate_lookup(follow_up_candidates)
                            if follow_up["status"] == "classified" and follow_up["action"] == "add_relation":
                                follow_up_result = _apply_relation_decision(
                                    session,
                                    node=updated_node,
                                    relation=follow_up["relation"],
                                    target_eid=follow_up["target_eid"],
                                    confidence=float(follow_up["confidence"]),
                                    candidate_by_eid=follow_up_candidate_by_eid,
                                    dry_run=dry_run,
                                )
                                follow_up_relation = follow_up_result["relation"]
                                follow_up_target_eid = follow_up_result["target_eid"]
                                follow_up_applied = bool(follow_up_result["applied"])
                                if follow_up_applied:
                                    relations_added[follow_up_relation] += 1
                                    post_relabel_relations_added += 1
                                elif not follow_up_reason:
                                    follow_up_reason = follow_up_result["skipped_reason"]
                            if not follow_up_applied:
                                relabeled_without_taxonomy_support += 1
                            queue_candidates = follow_up_candidates
                elif decision["action"] == "add_relation":
                    relation_result = _apply_relation_decision(
                        session,
                        node=node,
                        relation=decision["relation"],
                        target_eid=decision["target_eid"],
                        confidence=float(decision["confidence"]),
                        candidate_by_eid=candidate_by_eid,
                        dry_run=dry_run,
                    )
                    decision_relation = relation_result["relation"]
                    decision_target_eid = relation_result["target_eid"]
                    skipped_reason = relation_result["skipped_reason"]
                    applied = bool(relation_result["applied"])
                    if applied:
                        relations_added[decision_relation] += 1
                    elif skipped_reason in STRUCTURAL_BLOCK_REASONS:
                        final_action = "blocked"
                elif decision["action"] == "blocked":
                    skipped_reason = decision["reason"] or "Taxonomy action deferred as blocked."

                actions[final_action] += 1

                suspicious = bool(
                    skipped_reason
                    or decision["status"] == "unresolved"
                    or (
                        decision["action"] == "keep_label"
                        and _is_concept_only(node)
                        and int(node["outgoing_taxonomy_count"]) == 0
                        and decision["confidence"] >= 0.95
                    )
                )
                suspicious_reason = (
                    skipped_reason
                    or (
                        "High-confidence keep_label on concept-only node without taxonomy support"
                        if decision["action"] == "keep_label"
                        and _is_concept_only(node)
                        and int(node["outgoing_taxonomy_count"]) == 0
                        and decision["confidence"] >= 0.95
                        else ""
                    )
                )
                outgoing_taxonomy_count_after = int(node["outgoing_taxonomy_count"])
                if applied and final_action == "add_relation":
                    outgoing_taxonomy_count_after = max(outgoing_taxonomy_count_after, 1)
                if follow_up_applied:
                    outgoing_taxonomy_count_after = max(outgoing_taxonomy_count_after, 1)

                queue_reason_code, queue_reason = _queue_reason_from_decision(
                    final_action=final_action,
                    skipped_reason=follow_up_reason or skipped_reason,
                    node=queue_node,
                    outgoing_taxonomy_count_after=outgoing_taxonomy_count_after,
                    applied=applied,
                    follow_up_applied=follow_up_applied,
                )
                carry_forward_count = prior_carry_forward_counts.get(node["eid"], 0) + (1 if queue_reason_code else 0)

                _append_jsonl(
                    decisions_path,
                    {
                        "eid": node["eid"],
                        "name": node["name"],
                        "labels": node["labels"],
                        "status": decision["status"],
                        "action": final_action,
                        "original_action": decision["action"],
                        "label": decision["label"],
                        "relation": decision_relation,
                        "target_eid": decision_target_eid,
                        "confidence": round(float(decision["confidence"]), 4),
                        "reason": decision["reason"],
                        "client_name": decision.get("client_name", primary_role_config.client),
                        "model_name": decision["model_name"],
                        "stage": decision["stage"],
                        "used_second_stage": bool(decision.get("used_second_stage", False)),
                        "attempted_second_stage": bool(decision.get("attempted_second_stage", False)),
                        "applied": applied,
                        "skipped_reason": skipped_reason,
                        "suspicious": suspicious,
                        "suspicious_reason": suspicious_reason,
                        "outgoing_taxonomy_count": node["outgoing_taxonomy_count"],
                        "outgoing_taxonomy_count_after": outgoing_taxonomy_count_after,
                        "is_tier2_seed": bool(node.get("is_tier2_seed", False)),
                        "is_carry_forward_seed": bool(node.get("is_carry_forward_seed", False)),
                        "is_review_focus_seed": bool(node.get("is_review_focus_seed", False)),
                        "residual_priority_bucket": int(node.get("residual_priority_bucket", 0)),
                        "candidate_count": len(candidates),
                        "candidate_ids": [candidate["eid"] for candidate in candidates],
                        "follow_up_action": follow_up_action,
                        "follow_up_relation": follow_up_relation,
                        "follow_up_target_eid": follow_up_target_eid,
                        "follow_up_confidence": follow_up_confidence,
                        "follow_up_reason": follow_up_reason,
                        "follow_up_applied": follow_up_applied,
                        "carry_forward_count": carry_forward_count,
                        "queue_reason_code": queue_reason_code,
                        "queue_reason": queue_reason,
                        "candidate_targets": _candidate_snapshot(queue_candidates if queue_reason_code else candidates),
                    },
                )
                if queue_reason_code:
                    codex_queue_count += 1
                    queue_label = decision["label"] if final_action == "relabel" and applied else _current_primary_label(queue_node)
                    _append_jsonl(
                        codex_queue_path,
                        {
                            "eid": node["eid"],
                            "name": node["name"],
                            "description": queue_node.get("description"),
                            "current_labels": list(queue_node.get("labels", [])),
                            "suggested_label": queue_label,
                            "decision_action": final_action,
                            "decision_reason": decision["reason"],
                            "decision_confidence": round(float(decision["confidence"]), 4),
                            "follow_up_action": follow_up_action,
                            "follow_up_reason": follow_up_reason,
                            "carry_forward_count": carry_forward_count,
                            "queue_reason_code": queue_reason_code,
                            "queue_reason": queue_reason,
                            "degree": int(queue_node.get("degree", 0) or 0),
                            "relation_types": list(queue_node.get("relation_types", [])),
                            "neighbor_ids": list(queue_node.get("neighbor_ids", [])),
                            "neighbor_labels": list(queue_node.get("neighbor_labels", [])),
                            "taxonomy_targets": list(queue_node.get("taxonomy_targets", [])),
                            "candidate_targets": _candidate_snapshot(queue_candidates),
                        },
                    )

            summary = {
                "tier": "taxonomy",
                "dry_run": dry_run,
                "client_name": primary_role_config.client,
                "model_name": primary_role_config.model,
                "second_stage_client_name": secondary_role_config.client,
                "second_stage_model_name": secondary_role_config.model,
                "params": {
                    "max_nodes": max_nodes,
                    "candidate_limit": candidate_limit,
                    "embedding_threshold": embedding_threshold,
                    "neo4j_uri": neo4j_uri,
                    "neo4j_database": neo4j_database,
                    "labels_json": labels_json,
                    "tier2_decisions_jsonl": tier2_decisions_jsonl,
                    "prior_taxonomy_decisions_jsonl": prior_taxonomy_decisions_jsonl,
                    "prior_review_json": prior_review_json,
                    "decisions_jsonl": decisions_jsonl,
                },
                "processed_nodes": processed_nodes,
                "seed_eid_count": len(tier2_seed_eids),
                "residual_seed_count": residual_seed_count,
                "carry_forward_seed_count": carry_forward_seed_count,
                "review_focus_seed_count": review_focus_seed_count,
                "actions": actions,
                "unresolved": unresolved,
                "relabels_applied": relabels_applied,
                "relations_added": relations_added,
                "second_stage_decisions": second_stage_decisions,
                "second_stage_attempts": second_stage_attempts,
                "suspicious_relabels": suspicious_relabels,
                "suspicious_keep_label_concepts": suspicious_keep_label_concepts,
                "relabeled_without_taxonomy_support": relabeled_without_taxonomy_support,
                "post_relabel_relation_attempts": post_relabel_relation_attempts,
                "post_relabel_relations_added": post_relabel_relations_added,
                "codex_queue_count": codex_queue_count,
                "duration_seconds": round(time.time() - started, 3),
            }
    finally:
        driver.close()

    _write_json(summary_json, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Precision-first taxonomy cleanup after Tier 2.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-nodes", type=int, default=DEFAULT_MAX_NODES)
    parser.add_argument("--candidate-limit", type=int, default=DEFAULT_CANDIDATE_LIMIT)
    parser.add_argument("--embedding-threshold", type=float, default=DEFAULT_EMBEDDING_THRESHOLD)
    parser.add_argument("--labels-json", type=str, default=None)
    parser.add_argument("--tier2-decisions-jsonl", type=str, default=None)
    parser.add_argument("--prior-taxonomy-decisions-jsonl", type=str, default=None)
    parser.add_argument("--prior-review-json", type=str, default=None)
    parser.add_argument("--summary-json", type=str, default=None)
    parser.add_argument("--decisions-jsonl", type=str, default=None)
    parser.add_argument("--neo4j-uri", type=str, default=DEFAULT_NEO4J_URI)
    parser.add_argument("--neo4j-user", type=str, default=DEFAULT_NEO4J_USER)
    parser.add_argument("--neo4j-password", type=str, default=DEFAULT_NEO4J_PASSWORD)
    parser.add_argument("--neo4j-database", type=str, default=DEFAULT_NEO4J_DATABASE)
    parser.add_argument("--llm-routing-config", type=str, default=None)
    args = parser.parse_args()

    if args.max_nodes <= 0:
        parser.error("--max-nodes must be > 0")
    if args.candidate_limit <= 0:
        parser.error("--candidate-limit must be > 0")
    if not 0.0 <= args.embedding_threshold <= 1.0:
        parser.error("--embedding-threshold must be in [0.0, 1.0]")

    summary = run(
        dry_run=args.dry_run,
        max_nodes=args.max_nodes,
        candidate_limit=args.candidate_limit,
        embedding_threshold=args.embedding_threshold,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        neo4j_database=args.neo4j_database,
        labels_json=args.labels_json,
        tier2_decisions_jsonl=args.tier2_decisions_jsonl,
        prior_taxonomy_decisions_jsonl=args.prior_taxonomy_decisions_jsonl,
        prior_review_json=args.prior_review_json,
        summary_json=args.summary_json,
        decisions_jsonl=args.decisions_jsonl,
        llm_routing_config=args.llm_routing_config,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
