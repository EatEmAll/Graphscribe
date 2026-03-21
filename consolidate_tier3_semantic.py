"""
Tier 3 - Precision-first semantic deduplication.

This pass stays focused on alias cleanup. Taxonomy creation happens elsewhere.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
from google import genai
from google.genai import types
from neo4j import GraphDatabase

from graph_text_utils import coerce_text, normalize_name, sorted_unique_texts, token_set
from llm_json_utils import JsonDiskCache, generate_json_payload, make_cache_key

DEFAULT_NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
DEFAULT_NEO4J_USER = os.environ.get("NEO4J_USERNAME", "neo4j")
DEFAULT_NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password123")
DEFAULT_NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
EMBED_MODEL = os.environ.get("TIER3_EMBED_MODEL", "gemini-embedding-001")
PRIMARY_JUDGE_MODEL = os.environ.get(
    "TIER3_JUDGE_MODEL_PRIMARY",
    os.environ.get("TIER3_JUDGE_MODEL", "gemini-3.1-flash-lite-preview"),
)
SECOND_STAGE_JUDGE_MODEL = os.environ.get("TIER3_JUDGE_MODEL_SECOND_STAGE", "gemini-3-flash-preview")
LOW_CONFIDENCE_THRESHOLD = float(os.environ.get("TIER3_LOW_CONFIDENCE_THRESHOLD", "0.72"))
MODEL_MAX_ATTEMPTS = int(os.environ.get("TIER3_MODEL_MAX_ATTEMPTS", "3"))
MODEL_RETRY_SLEEP_SECONDS = float(os.environ.get("TIER3_MODEL_RETRY_SLEEP_SECONDS", "1.0"))
NEIGHBORS_PER_ENTITY = int(os.environ.get("TIER3_NEIGHBORS_PER_ENTITY", "12"))

DEFAULT_THRESHOLD = 0.85
DEFAULT_EMBED_PROGRESS_EVERY = 100
DEFAULT_MAX_CANDIDATES = 600
DEFAULT_MAX_MERGES = 200
DEFAULT_SLEEP_SECONDS = 0.0
DEFAULT_CACHE_FILE = "embeddings_cache.pkl"
DEFAULT_JUDGE_CACHE_FILE = os.environ.get("TIER3_JUDGE_CACHE_FILE", "tier3_judge_cache.json")
LABEL_COMPATIBILITY_GROUPS = [
    {"Metric", "Financial Metric"},
    {"Strategy", "Trading System", "Trading Concept"},
    {"Signal", "Indicator", "Market Feature"},
    {"Method", "Algorithm", "Model"},
]

JUDGE_SYSTEM = (
    "You are a high-precision knowledge graph deduplication expert. "
    "Only return ALIAS when both entities clearly refer to the same concept. "
    "If unsure, return DIFFERENT."
)


_coerce_text = coerce_text
_normalize_name = normalize_name
_token_set = token_set
_sorted_unique_texts = sorted_unique_texts


def _canonical_name(name_a: str, name_b: str) -> str:
    if not name_a and not name_b:
        return ""
    if not name_a:
        return name_b
    if not name_b:
        return name_a
    return name_a if len(name_a) <= len(name_b) else name_b


def _looks_alias_like(name_a: str, name_b: str) -> bool:
    normalized_a = _normalize_name(name_a)
    normalized_b = _normalize_name(name_b)
    if not normalized_a or not normalized_b:
        return False
    if normalized_a == normalized_b:
        return True
    tokens_a = _token_set(name_a)
    tokens_b = _token_set(name_b)
    if tokens_a and tokens_a == tokens_b:
        return True
    if normalized_a in normalized_b or normalized_b in normalized_a:
        return True
    acronym_a = "".join(token[0] for token in tokens_a if token)
    acronym_b = "".join(token[0] for token in tokens_b if token)
    return bool(acronym_a and acronym_b and {acronym_a, acronym_b} == {normalized_a.replace(" ", ""), normalized_b.replace(" ", "")})


def _labels_are_clearly_incompatible(labels_a: list[str], labels_b: list[str]) -> bool:
    set_a = {label for label in labels_a if label not in {"__Entity__", "Concept"}}
    set_b = {label for label in labels_b if label not in {"__Entity__", "Concept"}}
    if not set_a or not set_b:
        return False
    if set_a & set_b:
        return False
    for group in LABEL_COMPATIBILITY_GROUPS:
        if set_a <= group and set_b <= group:
            return False
    return True


def _normalize_entity_for_prompt(entity: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": _coerce_text(entity.get("name")),
        "description": _coerce_text(entity.get("description")),
        "labels": _sorted_unique_texts(entity.get("labels", [])) or ["Concept"],
        "degree": int(entity.get("degree", 0) or 0),
        "relation_types": _sorted_unique_texts(entity.get("relation_types", [])),
        "neighbor_labels": _sorted_unique_texts(entity.get("neighbor_labels", [])),
    }


def _entity_prompt_sort_key(entity: dict[str, Any]) -> tuple[Any, ...]:
    return (
        entity["name"].casefold(),
        entity["description"].casefold(),
        tuple(entity["labels"]),
        entity["degree"],
        tuple(entity["relation_types"]),
        tuple(entity["neighbor_labels"]),
    )


def _normalize_entity_pair(entity_a: dict[str, Any], entity_b: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    normalized_a = _normalize_entity_for_prompt(entity_a)
    normalized_b = _normalize_entity_for_prompt(entity_b)
    ordered = sorted((normalized_a, normalized_b), key=_entity_prompt_sort_key)
    return ordered[0], ordered[1]


def embed_batch(
    client: genai.Client,
    texts: list[str],
    cache_file: str,
    progress_every: int = DEFAULT_EMBED_PROGRESS_EVERY,
) -> list[np.ndarray]:
    all_embeddings: list[np.ndarray] = []
    embed_dim: int | None = None
    cache: dict[str, np.ndarray] = {}

    cache_path = Path(cache_file)
    if cache_path.exists():
        try:
            with cache_path.open("rb") as handle:
                cache = pickle.load(handle)
            if cache:
                embed_dim = next(iter(cache.values())).shape[0]
        except Exception as exc:
            print(f"  Warning: failed to load cache '{cache_file}': {exc}")

    new_embeddings_computed = False
    for index, text in enumerate(texts):
        if text in cache:
            vec = cache[text]
            if embed_dim is None:
                embed_dim = vec.shape[0]
            all_embeddings.append(vec)
            continue
        try:
            result = client.models.embed_content(
                model=EMBED_MODEL,
                contents=text,
                config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
            )
            vec = np.array(result.embeddings[0].values)
            if embed_dim is None:
                embed_dim = vec.shape[0]
                print(f"  Embedding dim detected: {embed_dim}")
            cache[text] = vec
            all_embeddings.append(vec)
            new_embeddings_computed = True
        except Exception as exc:
            print(f"  Embed failed for item {index}: {exc}")
            dim = embed_dim if embed_dim is not None else 1
            all_embeddings.append(np.zeros(dim))

        if progress_every > 0 and (index + 1) % progress_every == 0:
            print(f"  Embedded {index + 1}/{len(texts)}...")

    if embed_dim and embed_dim > 1:
        for index, vec in enumerate(all_embeddings):
            if vec.shape[0] == 1:
                all_embeddings[index] = np.zeros(embed_dim)

    if new_embeddings_computed:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("wb") as handle:
                pickle.dump(cache, handle)
            print(f"  Saved {len(cache)} embeddings to {cache_file}")
        except Exception as exc:
            print(f"  Warning: failed to save cache '{cache_file}': {exc}")

    return all_embeddings


def _build_prompt(entity_a: dict[str, Any], entity_b: dict[str, Any]) -> str:
    return "\n".join(
        [
            "Two graph entities:",
            (
                f'A: id="{entity_a["name"]}", labels={entity_a["labels"] or ["Concept"]}, '
                f'degree={entity_a["degree"]}, relation_types={entity_a["relation_types"]}, '
                f'neighbor_labels={entity_a["neighbor_labels"]}, description="{entity_a["description"] or ""}"'
            ),
            (
                f'B: id="{entity_b["name"]}", labels={entity_b["labels"] or ["Concept"]}, '
                f'degree={entity_b["degree"]}, relation_types={entity_b["relation_types"]}, '
                f'neighbor_labels={entity_b["neighbor_labels"]}, description="{entity_b["description"] or ""}"'
            ),
            "Rules:",
            "- Return JSON only with keys verdict, confidence, reason.",
            "- Use ALIAS only when both nodes clearly refer to the same entity or concept.",
            "- If labels or neighborhood evidence materially conflict, return DIFFERENT.",
            'Return strict JSON only: {"verdict":"ALIAS|DIFFERENT","confidence":0.0,"reason":"..."}',
        ]
    )


def _build_judge_cache_key(
    *,
    model_name: str,
    entity_a: dict[str, Any],
    entity_b: dict[str, Any],
) -> str:
    return make_cache_key(
        namespace="tier3_judge_v1",
        payload={
            "model_name": model_name,
            "system_instruction": JUDGE_SYSTEM,
            "entity_a": entity_a,
            "entity_b": entity_b,
            "temperature": 0.0,
            "max_output_tokens": 120,
        },
    )


def _judge_once(
    client: genai.Client,
    *,
    model_name: str,
    entity_a: dict[str, Any],
    entity_b: dict[str, Any],
    cache: JsonDiskCache | None = None,
) -> dict[str, Any]:
    normalized_a, normalized_b = _normalize_entity_pair(entity_a, entity_b)
    cache_key = _build_judge_cache_key(
        model_name=model_name,
        entity_a=normalized_a,
        entity_b=normalized_b,
    )
    if cache is not None:
        cached_result = cache.get(cache_key)
        if isinstance(cached_result, dict):
            return dict(cached_result)

    payload, error_message = generate_json_payload(
        client,
        model_name=model_name,
        prompt=_build_prompt(normalized_a, normalized_b),
        system_instruction=JUDGE_SYSTEM,
        max_output_tokens=120,
        temperature=0.0,
        max_attempts=max(MODEL_MAX_ATTEMPTS, 1),
        retry_sleep_seconds=MODEL_RETRY_SLEEP_SECONDS,
    )
    if payload is None:
        return {
            "status": "unresolved",
            "verdict": "DIFFERENT",
            "confidence": 0.0,
            "reason": error_message or "Invalid or empty JSON response",
            "model_name": model_name,
            "stage": "primary" if model_name == PRIMARY_JUDGE_MODEL else "second_stage",
        }

    verdict = _coerce_text(payload.get("verdict")).upper()
    if verdict not in {"ALIAS", "DIFFERENT"}:
        verdict = "DIFFERENT"
    try:
        confidence = max(0.0, min(float(payload.get("confidence", 0.0)), 1.0))
    except (TypeError, ValueError):
        confidence = 0.0
    result = {
        "status": "classified",
        "verdict": verdict,
        "confidence": confidence,
        "reason": _coerce_text(payload.get("reason")),
        "model_name": model_name,
        "stage": "primary" if model_name == PRIMARY_JUDGE_MODEL else "second_stage",
    }
    if cache is not None:
        cache.set(cache_key, result)
    return result


def judge_pair(
    client: genai.Client,
    entity_a: dict[str, Any],
    entity_b: dict[str, Any],
    cache: JsonDiskCache | None = None,
) -> dict[str, Any]:
    primary = _judge_once(
        client,
        model_name=PRIMARY_JUDGE_MODEL,
        entity_a=entity_a,
        entity_b=entity_b,
        cache=cache,
    )
    needs_second_stage = (
        primary["status"] == "unresolved"
        or primary["confidence"] < LOW_CONFIDENCE_THRESHOLD
    )
    if not needs_second_stage:
        return {
            **primary,
            "used_second_stage": False,
            "attempted_second_stage": False,
        }

    secondary = _judge_once(
        client,
        model_name=SECOND_STAGE_JUDGE_MODEL,
        entity_a=entity_a,
        entity_b=entity_b,
        cache=cache,
    )
    if secondary["status"] == "classified":
        return {
            **secondary,
            "used_second_stage": True,
            "attempted_second_stage": True,
        }
    return {
        **primary,
        "verdict": "DIFFERENT",
        "used_second_stage": False,
        "attempted_second_stage": True,
    }


def fetch_entities(session) -> list[dict[str, Any]]:
    result = session.run(
        """
        MATCH (n:__Entity__)
        OPTIONAL MATCH (n)-[r]-(m)
        OPTIONAL MATCH (n)-[tax:SUBCLASS_OF|INSTANCE_OF|TYPE_OF|IS_A]-(tax_neighbor)
        WITH n,
             count(DISTINCT r) AS degree,
             collect(DISTINCT type(r))[0..8] AS relation_types,
             collect(DISTINCT [label IN labels(m) WHERE label <> '__Entity__'])[0..8] AS neighbor_label_sets,
             collect(DISTINCT elementId(tax_neighbor))[0..12] AS taxonomy_neighbor_eids
        RETURN
            elementId(n) AS eid,
            n.id AS name,
            n.description AS description,
            [label IN labels(n) WHERE label <> '__Entity__'] AS labels,
            degree,
            relation_types,
            neighbor_label_sets,
            taxonomy_neighbor_eids
        """
    )
    return [
        {
            "eid": row["eid"],
            "name": _coerce_text(row["name"]),
            "description": _coerce_text(row["description"]),
            "labels": [label for label in row["labels"] if label],
            "degree": int(row["degree"] or 0),
            "relation_types": [value for value in row["relation_types"] if value],
            "neighbor_labels": [
                label
                for labels in row["neighbor_label_sets"]
                if labels
                for label in labels
                if label and label != "__Entity__"
            ][:8],
            "taxonomy_neighbor_eids": [value for value in row["taxonomy_neighbor_eids"] if value],
        }
        for row in result
    ]


def merge_pair(session, eid_a: str, eid_b: str, canonical_name: str) -> None:
    session.run(
        """
        MATCH (a) WHERE elementId(a) = $eid_a
        MATCH (b) WHERE elementId(b) = $eid_b
        WITH [a, b] AS nodes
        CALL apoc.refactor.mergeNodes(nodes, {properties: 'combine', mergeRels: true})
        YIELD node
        SET node.id = $canonical_name
        RETURN node
        """,
        eid_a=eid_a,
        eid_b=eid_b,
        canonical_name=canonical_name,
    )


def _write_summary(summary_json: str | None, summary: dict[str, Any]) -> None:
    if not summary_json:
        return
    path = Path(summary_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _alias_priority(entity_a: dict[str, Any], entity_b: dict[str, Any]) -> int:
    score = 0
    if _looks_alias_like(entity_a["name"], entity_b["name"]):
        score += 2
    labels_a = {label for label in entity_a["labels"] if label != "Concept"}
    labels_b = {label for label in entity_b["labels"] if label != "Concept"}
    if labels_a and labels_b and labels_a == labels_b:
        score += 1
    return score


def _should_skip_pair(entity_a: dict[str, Any], entity_b: dict[str, Any]) -> bool:
    if not entity_a["name"] or not entity_b["name"]:
        return True
    if entity_a["name"] == entity_b["name"]:
        return True
    if entity_b["eid"] in entity_a["taxonomy_neighbor_eids"] or entity_a["eid"] in entity_b["taxonomy_neighbor_eids"]:
        return True
    if _labels_are_clearly_incompatible(entity_a["labels"], entity_b["labels"]) and not _looks_alias_like(
        entity_a["name"],
        entity_b["name"],
    ):
        return True
    return False


def run(
    *,
    dry_run: bool,
    threshold: float,
    max_candidates: int,
    max_merges: int,
    sleep_seconds: float,
    cache_file: str = DEFAULT_CACHE_FILE,
    judge_cache_file: str = DEFAULT_JUDGE_CACHE_FILE,
    neo4j_uri: str = DEFAULT_NEO4J_URI,
    neo4j_user: str = DEFAULT_NEO4J_USER,
    neo4j_password: str = DEFAULT_NEO4J_PASSWORD,
    neo4j_database: str = DEFAULT_NEO4J_DATABASE,
    summary_json: str | None = None,
) -> dict[str, Any]:
    if not GOOGLE_API_KEY:
        raise RuntimeError("Set GOOGLE_API_KEY environment variable.")

    started = time.time()
    client = genai.Client(api_key=GOOGLE_API_KEY)
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    judge_cache = JsonDiskCache(judge_cache_file)
    summary: dict[str, Any]

    try:
        with driver.session(database=neo4j_database) as session:
            print("Fetching all __Entity__ nodes...")
            entities = fetch_entities(session)
            print(f"  Total: {len(entities)}")

            if not entities:
                summary = {
                    "tier": 3,
                    "dry_run": dry_run,
                    "embed_model": EMBED_MODEL,
                    "judge_model_primary": PRIMARY_JUDGE_MODEL,
                    "judge_model_second_stage": SECOND_STAGE_JUDGE_MODEL,
                    "params": {
                        "threshold": threshold,
                        "max_candidates": max_candidates,
                        "max_merges": max_merges,
                        "sleep_seconds": sleep_seconds,
                        "neighbors_per_entity": max(1, min(NEIGHBORS_PER_ENTITY, max_candidates)),
                        "cache_file": cache_file,
                        "judge_cache_file": judge_cache_file,
                        "neo4j_uri": neo4j_uri,
                        "neo4j_database": neo4j_database,
                    },
                    "entity_count": 0,
                    "candidate_count_total": 0,
                    "candidate_count_limited": 0,
                    "candidate_count_filtered": 0,
                    "judged_pairs": 0,
                    "confirmed_merges": 0,
                    "relations_added": 0,
                    "kept_separate": 0,
                    "judge_counts": {"ALIAS": 0, "DIFFERENT": 0},
                    "second_stage_judgements": 0,
                    "second_stage_attempts": 0,
                    "merge_cap_hit": False,
                    "alias_acceptance_rate": 0.0,
                    "judge_cache_hits": judge_cache.hits,
                    "judge_cache_entries": len(judge_cache.records),
                    "duration_seconds": round(time.time() - started, 3),
                }
                _write_summary(summary_json, summary)
                return summary

            texts = [f"{item['name'] or ''}: {item['description'] or ''}" for item in entities]
            print(f"Embedding {len(texts)} entities...")
            embeddings = embed_batch(client, texts=texts, cache_file=cache_file)
            if not embeddings:
                raise RuntimeError("No embeddings were returned.")
            print(f"  Done. Embedding dim: {len(embeddings[0])}")

            emb_matrix = np.array(embeddings)
            norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
            emb_norm = emb_matrix / (norms + 1e-10)

            print(f"Scanning pairs with cosine similarity >= {threshold}...")
            candidates_all: list[tuple[int, int, float, int]] = []
            filtered_out = 0
            total = len(entities)
            row_candidate_limit = max(1, min(NEIGHBORS_PER_ENTITY, max_candidates))
            for left_index in range(total):
                similarities = emb_norm[left_index] @ emb_norm[left_index + 1 :].T
                if similarities.size == 0:
                    continue
                candidate_offsets = np.flatnonzero(similarities >= threshold)
                if candidate_offsets.size == 0:
                    continue
                if candidate_offsets.size > row_candidate_limit:
                    top_local = np.argpartition(similarities[candidate_offsets], -row_candidate_limit)[-row_candidate_limit:]
                    candidate_offsets = candidate_offsets[top_local]
                ordered_offsets = candidate_offsets[np.argsort(similarities[candidate_offsets])[::-1]]
                for offset in ordered_offsets:
                    similarity = float(similarities[offset])
                    right_index = left_index + 1 + offset
                    entity_a = entities[left_index]
                    entity_b = entities[right_index]
                    if _should_skip_pair(entity_a, entity_b):
                        filtered_out += 1
                        continue
                    candidates_all.append(
                        (
                            left_index,
                            right_index,
                            float(similarity),
                            _alias_priority(entity_a, entity_b),
                        )
                    )

            candidates_sorted = sorted(candidates_all, key=lambda item: (-item[3], -item[2]))
            candidates = candidates_sorted[:max_candidates]
            print(f"  Found {len(candidates_all)} candidate pairs")
            print(f"  Filtered out {filtered_out} clearly bad pairs before LLM judgement")
            print(f"  Limiting to top {len(candidates)} pairs for LLM judgement...")

            merged_count = 0
            kept_count = 0
            judged_pairs = 0
            second_stage_judgements = 0
            second_stage_attempts = 0
            merge_cap_hit = False
            merged_eids: set[str] = set()
            judge_counts = {"ALIAS": 0, "DIFFERENT": 0}

            for left_index, right_index, similarity, _priority in candidates:
                entity_a = entities[left_index]
                entity_b = entities[right_index]
                if entity_a["eid"] in merged_eids or entity_b["eid"] in merged_eids:
                    continue

                verdict = judge_pair(client, entity_a, entity_b, cache=judge_cache)
                judged_pairs += 1
                if verdict.get("attempted_second_stage"):
                    second_stage_attempts += 1
                if verdict.get("used_second_stage"):
                    second_stage_judgements += 1
                judge_counts[verdict["verdict"]] = judge_counts.get(verdict["verdict"], 0) + 1

                if verdict["verdict"] == "ALIAS":
                    if merged_count >= max_merges:
                        print(
                            f"  [{similarity:.3f}] '{entity_a['name']}' vs '{entity_b['name']}' -> ALIAS "
                            "(SKIP, merge cap reached)"
                        )
                        merge_cap_hit = True
                        kept_count += 1
                    else:
                        print(
                            f"  [{similarity:.3f}] '{entity_a['name']}' vs '{entity_b['name']}' -> ALIAS "
                            f"(confidence={verdict['confidence']:.2f}, model={verdict['model_name']})"
                        )
                        if not dry_run:
                            canonical_name = _canonical_name(entity_a["name"], entity_b["name"])
                            merge_pair(session, entity_a["eid"], entity_b["eid"], canonical_name)
                        merged_count += 1
                        merged_eids.add(entity_a["eid"])
                        merged_eids.add(entity_b["eid"])
                else:
                    print(
                        f"  [{similarity:.3f}] '{entity_a['name']}' vs '{entity_b['name']}' -> DIFFERENT "
                        f"(confidence={verdict['confidence']:.2f}, model={verdict['model_name']})"
                    )
                    kept_count += 1

                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)

            alias_acceptance_rate = round((judge_counts["ALIAS"] / judged_pairs), 4) if judged_pairs else 0.0
            summary = {
                "tier": 3,
                "dry_run": dry_run,
                "embed_model": EMBED_MODEL,
                "judge_model_primary": PRIMARY_JUDGE_MODEL,
                "judge_model_second_stage": SECOND_STAGE_JUDGE_MODEL,
                "params": {
                    "threshold": threshold,
                    "max_candidates": max_candidates,
                    "max_merges": max_merges,
                    "sleep_seconds": sleep_seconds,
                    "neighbors_per_entity": row_candidate_limit,
                    "cache_file": cache_file,
                    "judge_cache_file": judge_cache_file,
                    "neo4j_uri": neo4j_uri,
                    "neo4j_database": neo4j_database,
                },
                "entity_count": len(entities),
                "candidate_count_total": len(candidates_all),
                "candidate_count_limited": len(candidates),
                "candidate_count_filtered": filtered_out,
                "judged_pairs": judged_pairs,
                "confirmed_merges": merged_count,
                "relations_added": 0,
                "kept_separate": kept_count,
                "judge_counts": judge_counts,
                "second_stage_judgements": second_stage_judgements,
                "second_stage_attempts": second_stage_attempts,
                "merge_cap_hit": merge_cap_hit,
                "alias_acceptance_rate": alias_acceptance_rate,
                "judge_cache_hits": judge_cache.hits,
                "judge_cache_entries": len(judge_cache.records),
                "duration_seconds": round(time.time() - started, 3),
            }
    finally:
        judge_cache.save()
        driver.close()

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Summary:")
    print(f"  Confirmed merges (ALIAS): {summary['confirmed_merges']}")
    print(f"  Relations added (taxonomy): {summary['relations_added']}")
    print(f"  Kept separate: {summary['kept_separate']}")
    print(f"  Judge cache hits: {summary['judge_cache_hits']}")
    if dry_run:
        print("\n[DRY RUN] No changes written.")
    else:
        print("\nTier 3 complete.")

    _write_summary(summary_json, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Tier 3: Precision-first semantic dedup")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES)
    parser.add_argument("--max-merges", type=int, default=DEFAULT_MAX_MERGES)
    parser.add_argument("--sleep-seconds", type=float, default=DEFAULT_SLEEP_SECONDS)
    parser.add_argument("--cache-file", type=str, default=DEFAULT_CACHE_FILE)
    parser.add_argument("--judge-cache-file", type=str, default=DEFAULT_JUDGE_CACHE_FILE)
    parser.add_argument("--summary-json", type=str, default=None)
    parser.add_argument("--neo4j-uri", type=str, default=DEFAULT_NEO4J_URI)
    parser.add_argument("--neo4j-user", type=str, default=DEFAULT_NEO4J_USER)
    parser.add_argument("--neo4j-password", type=str, default=DEFAULT_NEO4J_PASSWORD)
    parser.add_argument("--neo4j-database", type=str, default=DEFAULT_NEO4J_DATABASE)
    args = parser.parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be in [0.0, 1.0]")
    if args.max_candidates <= 0:
        parser.error("--max-candidates must be > 0")
    if args.max_merges <= 0:
        parser.error("--max-merges must be > 0")
    if args.sleep_seconds < 0:
        parser.error("--sleep-seconds must be >= 0")

    run(
        dry_run=args.dry_run,
        threshold=args.threshold,
        max_candidates=args.max_candidates,
        max_merges=args.max_merges,
        sleep_seconds=args.sleep_seconds,
        cache_file=args.cache_file,
        judge_cache_file=args.judge_cache_file,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        neo4j_database=args.neo4j_database,
        summary_json=args.summary_json,
    )


if __name__ == "__main__":
    main()
