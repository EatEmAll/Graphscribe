#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _evaluation_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    return list(report.get("questions") or [])


def _mean_score(rows: list[dict[str, Any]]) -> float:
    scores = [
        float(judgment["total_score"])
        for row in rows
        for judgment in (row.get("judgments") or {}).values()
    ]
    return sum(scores) / len(scores) if scores else 0.0


def _effective_citation_validity(answer: dict[str, Any], reported: float) -> float:
    if reported == 1.0:
        return 1.0
    if not answer.get("citations") and str(answer.get("answer") or "").startswith("Insufficient evidence"):
        return 1.0
    return reported


def compare_evaluations(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    minimum_quality_ratio: float = 0.95,
    minimum_graph_success: float = 0.90,
) -> dict[str, Any]:
    baseline_rows = _evaluation_rows(baseline)
    candidate_rows = _evaluation_rows(candidate)
    baseline_questions = [
        (str(row.get("question_id")), str(row.get("question")), str(row.get("category")))
        for row in baseline_rows
    ]
    candidate_questions = [
        (str(row.get("question_id")), str(row.get("question")), str(row.get("category")))
        for row in candidate_rows
    ]
    same_questions = baseline_questions == candidate_questions and bool(baseline_questions)
    baseline_mean = _mean_score(baseline_rows)
    candidate_mean = _mean_score(candidate_rows)
    quality_ratio = candidate_mean / baseline_mean if baseline_mean else 0.0
    citation_values = [
        _effective_citation_validity(
            answer,
            float((row.get("citation_validity") or {}).get(mode, 0.0)),
        )
        for row in candidate_rows
        for mode, answer in (row.get("answers") or {}).items()
    ]
    citation_gate = bool(citation_values) and all(value == 1.0 for value in citation_values)
    baseline_unsupported = sum(
        int(judgment.get("unsupported_claim_count") or 0)
        for row in baseline_rows
        for judgment in (row.get("judgments") or {}).values()
    )
    candidate_unsupported = sum(
        int(judgment.get("unsupported_claim_count") or 0)
        for row in candidate_rows
        for judgment in (row.get("judgments") or {}).values()
    )
    graph_rows = [row for row in candidate_rows if row.get("category") == "entity_bridge"]
    graph_successes = sum(
        int(((row.get("answers") or {}).get("graph_hybrid") or {}).get("retrieval", {}).get("graph_candidates", 0))
        > 0
        for row in graph_rows
    )
    graph_success_rate = graph_successes / len(graph_rows) if graph_rows else 0.0
    gates = {
        "same_questions": same_questions,
        "quality": quality_ratio >= minimum_quality_ratio,
        "citations": citation_gate,
        "unsupported_claims": candidate_unsupported <= baseline_unsupported,
        "graph_expansion": bool(graph_rows) and graph_success_rate >= minimum_graph_success,
    }
    return {
        "baseline_mean_score": baseline_mean,
        "candidate_mean_score": candidate_mean,
        "quality_ratio": quality_ratio,
        "candidate_citation_validity": min(citation_values) if citation_values else 0.0,
        "baseline_unsupported_claims": baseline_unsupported,
        "candidate_unsupported_claims": candidate_unsupported,
        "graph_question_count": len(graph_rows),
        "graph_success_rate": graph_success_rate,
        "gates": gates,
        "passed": all(gates.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate a compact-corpus evaluation against its baseline.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-quality-ratio", type=float, default=0.95)
    parser.add_argument("--minimum-graph-success", type=float, default=0.90)
    args = parser.parse_args()
    result = compare_evaluations(
        json.loads(args.baseline.read_text(encoding="utf-8")),
        json.loads(args.candidate.read_text(encoding="utf-8")),
        minimum_quality_ratio=args.minimum_quality_ratio,
        minimum_graph_success=args.minimum_graph_success,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(args.output.resolve())
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
