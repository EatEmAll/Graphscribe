#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

REPO_ROOT_PATH = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))

from notebooklm_graph_pipe.paths import REPO_ROOT
from notebooklm_graph_pipe.runtime.llm_json_utils import build_single_prompt_clients, generate_json_payload
from notebooklm_graph_pipe.runtime.llm_routing import (
    EVALUATION_JUDGE_ROLE,
    EVALUATION_QUESTION_ROLE,
    PromptRoleConfig,
    resolve_prompt_role,
)
from notebooklm_graph_pipe.service.core import CorpusService
from notebooklm_graph_pipe.service.jobs import CorpusJobManager
from notebooklm_graph_pipe.service.registry import CorpusRegistry
from notebooklm_graph_pipe.service.runtime import RuntimeFactory


TEXT_HYBRID = "hybrid"
GRAPH_HYBRID = "graph_hybrid"


@dataclass(frozen=True)
class EvaluationQuestion:
    id: str
    text: str
    reserve: bool = False


def load_questions(path: Path) -> list[EvaluationQuestion]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("questions") if isinstance(payload, dict) else payload
    if not isinstance(raw, list) or not raw:
        raise ValueError("Question file must contain a non-empty questions list.")
    questions = [
        EvaluationQuestion(
            id=str(item.get("question_id") or item.get("id") or f"Q{index:02d}"),
            text=str(item["text"]).strip(),
            reserve=bool(item.get("reserve", False)),
        )
        for index, item in enumerate(raw, 1)
    ]
    if any(not question.text for question in questions):
        raise ValueError("Evaluation questions cannot be empty.")
    return questions


def citation_validity(answer: dict[str, Any]) -> float:
    citations = answer.get("citations") or []
    if not citations:
        return 0.0
    valid = sum(
        1
        for citation in citations
        if citation.get("id") and citation.get("document_id") and citation.get("source_uri")
    )
    return valid / len(citations)


class EvaluationModel:
    def __init__(
        self,
        question_role: PromptRoleConfig,
        judge_role: PromptRoleConfig,
        clients: dict[str, Any],
        generator=generate_json_payload,
    ):
        self.question_role = question_role
        self.judge_role = judge_role
        self.clients = clients
        self.generator = generator

    @classmethod
    def from_routing_config(cls, config_path: str | None) -> "EvaluationModel":
        question_role = resolve_prompt_role(
            config_path,
            EVALUATION_QUESTION_ROLE,
            default_client="genai",
            default_model="gemini-2.5-flash",
        )
        judge_role = resolve_prompt_role(
            config_path,
            EVALUATION_JUDGE_ROLE,
            default_client="genai",
            default_model="gemini-2.5-flash",
        )
        clients = build_single_prompt_clients(question_role.client, judge_role.client)
        return cls(question_role, judge_role, clients)

    def generate_questions(self, service: CorpusService, corpus_key: str) -> list[EvaluationQuestion]:
        discovery_queries = [
            "What are the principal themes and source-backed conclusions in this corpus?",
            "Which entities, relationships, disagreements, and cross-document bridges are most important?",
        ]
        context_parts: list[str] = []
        for query in discovery_queries:
            result = service.search(
                corpus_key,
                {
                    "query": query,
                    "mode": GRAPH_HYBRID,
                    "top_k": 12,
                    "graph_hops": 1,
                },
            )
            for context in result.get("contexts") or []:
                context_parts.append(f"SOURCE: {context.get('title')}\n{context.get('text')}")
        if not context_parts:
            raise RuntimeError("Cannot generate evaluation questions without retrieved source context.")
        prompt = (
            "Create exactly 10 corpus-grounded evaluation questions. The first 8 are primary and the last 2 are reserves. "
            "Mix fact lookup, paraphrase, rare keyword, cross-document synthesis, entity bridge, disagreement, and unanswerable/coverage questions. "
            "Return JSON with a questions array; each item has question_id, text, and reserve.\n\n"
            + "\n\n".join(context_parts[:16])
        )
        payload, error = self.generator(
            self.clients[self.question_role.client],
            client_name=self.question_role.client,
            model_name=self.question_role.model,
            prompt=prompt,
            system_instruction="Design difficult but answerable source-grounded retrieval evaluations.",
            max_output_tokens=4096,
            temperature=0.0,
            max_attempts=2,
        )
        if payload is None:
            raise RuntimeError(f"Question generation failed: {error}")
        raw = payload.get("questions")
        if not isinstance(raw, list) or len(raw) != 10:
            raise RuntimeError("Question generation must return exactly 10 questions.")
        questions = [
            EvaluationQuestion(
                id=f"Q{index:02d}",
                text=str(item.get("text") or "").strip(),
                reserve=index > 8,
            )
            for index, item in enumerate(raw, 1)
        ]
        if any(not question.text for question in questions):
            raise RuntimeError("Question generation returned an empty question.")
        return questions

    def judge(self, question: str, mode: str, answer: dict[str, Any]) -> dict[str, Any]:
        prompt = (
            "Score this source-grounded answer from 1 to 5 on correctness, completeness, evidence_quality, and cross_document_synthesis. "
            "Count claims that are not supported by the supplied citation records as unsupported_claim_count. "
            "Return JSON with those fields, total_score, rationale, and weakness_tags.\n\n"
            f"Question: {question}\nMode: {mode}\nAnswer payload:\n{json.dumps(answer, ensure_ascii=False)}"
        )
        payload, error = self.generator(
            self.clients[self.judge_role.client],
            client_name=self.judge_role.client,
            model_name=self.judge_role.model,
            prompt=prompt,
            system_instruction="You are a strict retrieval evaluation judge. Use no outside knowledge.",
            max_output_tokens=2048,
            temperature=0.0,
            max_attempts=2,
        )
        if payload is None:
            raise RuntimeError(f"Evaluation judge failed: {error}")
        scores = {
            name: int(payload.get(name) or 0)
            for name in ("correctness", "completeness", "evidence_quality", "cross_document_synthesis")
        }
        if any(value < 1 or value > 5 for value in scores.values()):
            raise RuntimeError(f"Evaluation judge returned invalid factor scores: {scores}")
        total = sum(scores.values())
        return {
            **scores,
            "total_score": total,
            "normalized_score": total / 2,
            "citation_validity": citation_validity(answer),
            "unsupported_claim_count": max(0, int(payload.get("unsupported_claim_count") or 0)),
            "rationale": str(payload.get("rationale") or ""),
            "weakness_tags": list(payload.get("weakness_tags") or []),
        }


def evaluate(
    service: CorpusService,
    corpus_key: str,
    questions: list[EvaluationQuestion],
    judge: Callable[[str, str, dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for question in questions:
        answers: dict[str, dict[str, Any]] = {}
        for mode in (TEXT_HYBRID, GRAPH_HYBRID):
            answer = service.answer(corpus_key, {"question": question.text, "mode": mode, "graph_hops": 1})
            answers[mode] = answer
        row: dict[str, Any] = {
            "question_id": question.id,
            "question": question.text,
            "reserve": question.reserve,
            "answers": answers,
            "citation_validity": {
                mode: citation_validity(answer) for mode, answer in answers.items()
            },
        }
        if judge:
            row["judgments"] = {
                mode: judge(question.text, mode, answer) for mode, answer in answers.items()
            }
        rows.append(row)
    wins = {TEXT_HYBRID: 0, GRAPH_HYBRID: 0, "tie": 0}
    if judge:
        for row in rows:
            left = row["judgments"][TEXT_HYBRID]["total_score"]
            right = row["judgments"][GRAPH_HYBRID]["total_score"]
            winner = "tie" if abs(left - right) <= 1 else GRAPH_HYBRID if right > left else TEXT_HYBRID
            row["winner"] = winner
            wins[winner] += 1
    return {
        "corpus_key": corpus_key,
        "conditions": [TEXT_HYBRID, GRAPH_HYBRID],
        "questions": rows,
        "summary": {
            "question_count": len(rows),
            "mean_citation_validity": {
                mode: sum(row["citation_validity"][mode] for row in rows) / len(rows)
                for mode in (TEXT_HYBRID, GRAPH_HYBRID)
            },
            "wins": wins,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare text-hybrid and graph-hybrid retrieval on a v3 corpus.")
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--questions-file")
    parser.add_argument("--output")
    parser.add_argument("--llm-routing-config")
    args = parser.parse_args()
    manifest_path = Path(args.manifest_path).resolve()
    registry = CorpusRegistry(manifest_path.parent.parent)
    entry = next(
        (entry for entry in registry.entries().values() if entry.manifest_path == manifest_path),
        None,
    )
    if entry is None:
        parser.error("Manifest must live under a registered corpus directory.")
    service = CorpusService(registry, RuntimeFactory(args.llm_routing_config), CorpusJobManager(registry, REPO_ROOT))
    try:
        model = EvaluationModel.from_routing_config(args.llm_routing_config)
        questions = load_questions(Path(args.questions_file)) if args.questions_file else model.generate_questions(service, entry.key)
        report = evaluate(service, entry.key, questions, model.judge)
    finally:
        service.close()
    output = Path(args.output).resolve() if args.output else manifest_path.parent / "evaluation.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
