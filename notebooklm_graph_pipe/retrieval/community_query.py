from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from notebooklm_graph_pipe.runtime.llm_routing import (
    DRIFT_PLANNER_ROLE,
    GLOBAL_MAP_ROLE,
    GLOBAL_REDUCE_ROLE,
)
from notebooklm_graph_pipe.runtime.model_executor import ModelExecutor, ModelRequest

from .answering import GroundedAnswerer


MAP_SCHEMA: dict[str, object] = {
    "type": "object",
    "required": ["points"],
    "properties": {
        "points": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["text", "score", "source_parent_ids"],
                "properties": {
                    "text": {"type": "string"},
                    "score": {"type": "number"},
                    "source_parent_ids": {"type": "array", "items": {"type": "string"}},
                },
            },
        }
    },
}
REDUCE_SCHEMA: dict[str, object] = {
    "type": "object",
    "required": ["answer", "citation_parent_ids"],
    "properties": {
        "answer": {"type": "string"},
        "citation_parent_ids": {"type": "array", "items": {"type": "string"}},
    },
}
PLANNER_SCHEMA: dict[str, object] = {
    "type": "object",
    "required": ["subqueries"],
    "properties": {
        "subqueries": {
            "type": "array",
            "minItems": 1,
            "maxItems": 5,
            "items": {"type": "string"},
        },
        "rationale": {"type": "string"},
    },
}


@dataclass
class CommunityQueryEngine:
    backend: Any
    embedder: Any
    executor: ModelExecutor
    local_answerer: GroundedAnswerer
    map_batch_size: int = 8

    def global_answer(self, question: str) -> dict[str, Any]:
        reports = self.backend.community_reports()
        if not reports:
            return self._insufficient("No active community reports are available.", "global")
        requests = []
        for start in range(0, len(reports), self.map_batch_size):
            batch = reports[start : start + self.map_batch_size]
            requests.append(
                ModelRequest(
                    role=GLOBAL_MAP_ROLE,
                    prompt=(
                        "Extract relevant answer points for the question from these grounded community reports. "
                        "Every point must retain only source_parent_ids present in the supplied findings.\n\n"
                        f"Question: {question}\n\nReports:\n{json.dumps(batch, ensure_ascii=False, sort_keys=True)}"
                    ),
                    system_instruction="Return source-grounded map points only.",
                    response_schema=MAP_SCHEMA,
                    max_output_tokens=2048,
                    cache_namespace="global-map",
                )
            )
        mapped = asyncio.run(self.executor.amap_json(requests))
        allowed = self._report_parent_ids(reports)
        points = []
        invalid = 0
        for result in mapped:
            for point in (result.payload or {}).get("points") or []:
                parent_ids = list(dict.fromkeys(str(value) for value in point.get("source_parent_ids") or []))
                if not parent_ids or not set(parent_ids).issubset(allowed):
                    invalid += 1
                    continue
                points.append({**point, "source_parent_ids": parent_ids})
        if not points:
            return self._insufficient("Community reports did not yield grounded answer points.", "global")
        return self._reduce(question, points, "global", {"reports": len(reports), "map_batches": len(requests), "invalid_points": invalid})

    def drift_answer(self, question: str, *, graph_hops: int = 1) -> dict[str, Any]:
        primers = self.backend.search_community_reports(question, self.embedder.embed_query(question))
        if not primers:
            return self._insufficient("No active community primer was found.", "drift")
        request = ModelRequest(
            role=DRIFT_PLANNER_ROLE,
            prompt=(
                "Plan up to five focused evidence questions that together answer the user's question.\n\n"
                f"Question: {question}\n\nCommunity primers:\n"
                + json.dumps(primers, ensure_ascii=False, sort_keys=True)
            ),
            system_instruction="Return a bounded research plan, not an answer.",
            response_schema=PLANNER_SCHEMA,
            max_output_tokens=1024,
            cache_namespace="drift-plan",
        )
        plan = self.executor.execute_json(request).payload or {}
        subqueries = [str(value).strip() for value in plan.get("subqueries") or [] if str(value).strip()][:5]
        evidence = []
        warnings = []
        for subquery in subqueries:
            result = self.local_answerer.answer(subquery, mode="graph_hybrid", graph_hops=graph_hops)
            warnings.extend(result.get("warnings") or [])
            parent_ids = [str(item["parent_id"]) for item in result.get("citations") or [] if item.get("parent_id")]
            if parent_ids:
                evidence.append(
                    {
                        "text": f"Question: {subquery}\nAnswer: {result['answer']}",
                        "score": 1.0,
                        "source_parent_ids": parent_ids,
                    }
                )
        if not evidence:
            return self._insufficient("DRIFT follow-ups did not retrieve grounded evidence.", "drift")
        answer = self._reduce(
            question,
            evidence,
            "drift",
            {"primer_reports": len(primers), "subqueries": subqueries},
        )
        answer["warnings"].extend(dict.fromkeys(warnings))
        return answer

    def _reduce(
        self,
        question: str,
        points: list[dict[str, Any]],
        mode: str,
        diagnostics: dict[str, Any],
    ) -> dict[str, Any]:
        parent_ids = list(dict.fromkeys(parent for point in points for parent in point["source_parent_ids"]))
        parents = self.backend.parent_contexts(parent_ids)
        valid_ids = set(parents)
        points = [
            {**point, "source_parent_ids": [parent for parent in point["source_parent_ids"] if parent in valid_ids]}
            for point in points
        ]
        points = [point for point in points if point["source_parent_ids"]]
        request = ModelRequest(
            role=GLOBAL_REDUCE_ROLE,
            prompt=(
                "Synthesize a direct answer using only these grounded points. Return the parent IDs supporting "
                "the final answer in citation_parent_ids.\n\n"
                f"Question: {question}\n\nPoints:\n{json.dumps(points, ensure_ascii=False, sort_keys=True)}"
            ),
            system_instruction="Do not introduce claims beyond the supplied grounded points.",
            response_schema=REDUCE_SCHEMA,
            max_output_tokens=4096,
            cache_namespace=f"{mode}-reduce",
        )
        payload = self.executor.execute_json(request).payload or {}
        cited = [
            value
            for value in dict.fromkeys(str(item) for item in payload.get("citation_parent_ids") or [])
            if value in valid_ids
        ]
        if not cited:
            return self._insufficient("The synthesized answer did not retain valid source evidence.", mode)
        citations = []
        for index, parent_id in enumerate(cited, 1):
            parent = parents[parent_id]
            citations.append(
                {
                    "id": f"S{index}",
                    "parent_id": parent_id,
                    "document_id": str(parent["document_id"]),
                    "title": str(parent.get("title") or ""),
                    "source_uri": str(parent.get("source_uri") or ""),
                    "page_start": parent.get("page_start"),
                    "page_end": parent.get("page_end"),
                    "timestamp_start_ms": parent.get("timestamp_start_ms"),
                    "timestamp_end_ms": parent.get("timestamp_end_ms"),
                    "section_path": list(parent.get("section_path") or ()),
                    "quote_preview": str(parent.get("text") or "")[:280],
                }
            )
        return {
            "answer": str(payload.get("answer") or "").strip(),
            "citations": citations,
            "retrieval": {"mode": mode, **diagnostics, "parents_used": len(citations)},
            "warnings": [],
        }

    @staticmethod
    def _report_parent_ids(reports: list[dict[str, Any]]) -> set[str]:
        return {
            str(finding["parent_id"])
            for report in reports
            for finding in report.get("findings") or []
            if finding.get("parent_id")
        }

    @staticmethod
    def _insufficient(reason: str, mode: str) -> dict[str, Any]:
        return {
            "answer": "Insufficient evidence was found in the active corpus.",
            "citations": [],
            "retrieval": {"mode": mode},
            "warnings": [reason],
        }
