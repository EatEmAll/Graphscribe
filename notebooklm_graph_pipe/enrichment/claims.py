from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

from notebooklm_graph_pipe.community.models import stable_hash
from notebooklm_graph_pipe.runtime.llm_routing import CLAIM_EXTRACTION_ROLE
from notebooklm_graph_pipe.runtime.model_executor import ModelExecutor, ModelRequest


CLAIM_SCHEMA: dict[str, object] = {
    "type": "object",
    "required": ["claims"],
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["subject", "predicate", "object", "stance", "extraction_confidence"],
                "properties": {
                    "subject": {"type": "string"},
                    "predicate": {"type": "string"},
                    "object": {"type": "string"},
                    "stance": {"type": "string", "enum": ["SUPPORTS", "CONTRADICTS"]},
                    "valid_from": {"type": ["string", "null"]},
                    "valid_to": {"type": ["string", "null"]},
                    "extraction_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
        }
    },
}


@dataclass(frozen=True)
class Claim:
    id: str
    subject: str
    predicate: str
    object: str
    stance: str
    valid_from: str | None
    valid_to: str | None
    extraction_confidence: float
    source_parent_id: str


class ClaimExtractor:
    def __init__(self, executor: ModelExecutor):
        self.executor = executor

    @property
    def fingerprint(self) -> str:
        return self.executor.model_fingerprint(
            CLAIM_EXTRACTION_ROLE,
            {"schema": CLAIM_SCHEMA, "prompt_version": 1},
        )

    def extract(self, parent_id: str, text: str) -> tuple[Claim, ...]:
        request = ModelRequest(
            role=CLAIM_EXTRACTION_ROLE,
            prompt=(
                "Extract only explicit, independently meaningful claims from this source. Use ISO dates when "
                "the source provides temporal bounds. Mark a claim CONTRADICTS only when the text explicitly "
                f"rejects that proposition.\n\nParent ID: {parent_id}\n\n{text}"
            ),
            system_instruction="Return source-grounded claims only.",
            response_schema=CLAIM_SCHEMA,
            max_output_tokens=4096,
            cache_namespace="claim-extraction",
            idempotency_key=parent_id,
        )
        payload = self.executor.execute_json(request).payload or {}
        claims = []
        for raw in payload.get("claims") or []:
            valid_from = self._date(raw.get("valid_from"))
            valid_to = self._date(raw.get("valid_to"))
            if valid_from and valid_to and valid_from > valid_to:
                raise ValueError("Claim valid_from cannot be after valid_to.")
            semantics = {
                "subject": str(raw["subject"]).strip(),
                "predicate": str(raw["predicate"]).strip(),
                "object": str(raw["object"]).strip(),
                "valid_from": valid_from,
                "valid_to": valid_to,
            }
            if not all((semantics["subject"], semantics["predicate"], semantics["object"])):
                raise ValueError("Claim subject, predicate, and object must be non-empty.")
            claims.append(
                Claim(
                    stable_hash(semantics),
                    **semantics,
                    stance=str(raw["stance"]),
                    extraction_confidence=float(raw["extraction_confidence"]),
                    source_parent_id=parent_id,
                )
            )
        return tuple(claims)

    @staticmethod
    def _date(value: Any) -> str | None:
        if value in {None, ""}:
            return None
        return date.fromisoformat(str(value)).isoformat()


def claim_rows(claims: tuple[Claim, ...]) -> list[dict[str, Any]]:
    return [asdict(claim) for claim in claims]
