from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

from notebooklm_graph_pipe.runtime.llm_json_utils import (
    build_single_prompt_clients,
    generate_json_payload,
)
from notebooklm_graph_pipe.runtime.llm_routing import (
    ANSWER_ROLE,
    PromptRoleConfig,
    resolve_prompt_role,
)

from .hybrid import HybridRetriever, SearchRequest


@dataclass
class GroundedAnswerer:
    retriever: HybridRetriever
    role: PromptRoleConfig
    client: Any
    generator: Callable[..., tuple[dict[str, Any] | None, str]] = generate_json_payload

    @classmethod
    def from_routing_config(cls, retriever: HybridRetriever, config_path: str | None) -> "GroundedAnswerer":
        role = resolve_prompt_role(
            config_path,
            ANSWER_ROLE,
            default_client="genai",
            default_model="gemini-2.5-flash",
        )
        client = build_single_prompt_clients(role.client)[role.client]
        return cls(retriever, role, client)

    def answer(self, question: str, *, mode: str = "graph_hybrid", graph_hops: int = 1) -> dict[str, Any]:
        result = self.retriever.search(
            SearchRequest(question, mode=mode, graph_hops=graph_hops, include_diagnostics=True)
        )
        if not result.contexts:
            return {
                "answer": "Insufficient evidence was found in the active corpus.",
                "citations": [],
                "retrieval": result.diagnostics,
                "warnings": ["No active source context was retrieved."],
            }
        context_text = "\n\n".join(
            f"[{item.citation_id}] {item.title}\n{item.text}" for item in result.contexts
        )
        prompt = (
            "Answer the question using only the supplied source context. "
            "Cite supporting claims with the exact citation IDs. If the context is insufficient, say so. "
            "Return JSON with keys answer and citation_ids.\n\n"
            f"Question: {question}\n\nContext:\n{context_text}"
        )
        payload, error = self.generator(
            self.client,
            client_name=self.role.client,
            model_name=self.role.model,
            prompt=prompt,
            system_instruction="You are a source-grounded research assistant. Never invent citations.",
            max_output_tokens=4096,
            temperature=0.0,
            max_attempts=2,
        )
        warnings: list[str] = []
        if payload is None:
            return {
                "answer": "Answer generation failed after retrieval.",
                "citations": [],
                "retrieval": result.diagnostics,
                "warnings": [error],
            }
        valid = {item.citation_id: item for item in result.contexts}
        requested_ids = [str(value) for value in payload.get("citation_ids") or []]
        cited_in_text = re.findall(r"\[(S\d+)\]", str(payload.get("answer") or ""))
        citation_ids = list(dict.fromkeys([*requested_ids, *cited_in_text]))
        invalid = [citation for citation in citation_ids if citation not in valid]
        if invalid:
            warnings.append(f"Removed invalid model citations: {', '.join(invalid)}")
        citation_ids = [citation for citation in citation_ids if citation in valid]
        answer = str(payload.get("answer") or "").strip()
        for invalid_id in invalid:
            answer = answer.replace(f"[{invalid_id}]", "")
        if not answer or not citation_ids:
            answer = "Insufficient evidence was found in the active corpus."
            citation_ids = []
            if not invalid:
                warnings.append("The generated answer did not contain a valid source citation.")
        return {
            "answer": answer,
            "citations": [valid[citation].citation_payload() for citation in citation_ids],
            "retrieval": result.diagnostics,
            "warnings": warnings,
        }
