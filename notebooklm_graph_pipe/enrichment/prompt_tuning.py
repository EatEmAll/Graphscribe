from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from notebooklm_graph_pipe.community.models import stable_hash
from notebooklm_graph_pipe.ingestion.manifest import CorpusManifest, save_manifest
from notebooklm_graph_pipe.runtime.llm_routing import PROMPT_TUNING_ROLE
from notebooklm_graph_pipe.runtime.model_executor import ModelExecutor, ModelRequest


TUNING_SCHEMA: dict[str, object] = {
    "type": "object",
    "required": ["domain", "entity_types", "relationship_guidance", "extraction_instructions"],
    "properties": {
        "domain": {"type": "string"},
        "entity_types": {"type": "array", "items": {"type": "string"}},
        "relationship_guidance": {"type": "array", "items": {"type": "string"}},
        "extraction_instructions": {"type": "string"},
    },
}


@dataclass(frozen=True)
class PromptProposal:
    id: str
    corpus_id: str
    sample_parent_ids: tuple[str, ...]
    current_catalog_hash: str
    domain: str
    entity_types: tuple[str, ...]
    relationship_guidance: tuple[str, ...]
    extraction_instructions: str
    status: str = "REVIEW_REQUIRED"


class PromptTuner:
    def __init__(self, executor: ModelExecutor):
        self.executor = executor

    def propose(
        self,
        corpus_id: str,
        parents: Sequence[dict[str, Any]],
        current_entity_types: Sequence[str],
    ) -> PromptProposal:
        samples = sorted(parents, key=lambda item: str(item["parent_id"]))
        request = ModelRequest(
            role=PROMPT_TUNING_ROLE,
            prompt=(
                "Infer a conservative domain-specific graph extraction ontology. Prefer the current catalog "
                "when it fits and propose only types grounded in multiple samples.\n\n"
                f"Current entity types: {json.dumps(sorted(set(current_entity_types)))}\n\n"
                f"Samples: {json.dumps(samples, ensure_ascii=False, sort_keys=True)}"
            ),
            system_instruction="Return a prompt proposal for human review; do not activate it.",
            response_schema=TUNING_SCHEMA,
            max_output_tokens=4096,
            cache_namespace="prompt-tuning",
        )
        payload = self.executor.execute_json(request).payload or {}
        core = {
            "corpus_id": corpus_id,
            "sample_parent_ids": tuple(str(item["parent_id"]) for item in samples),
            "current_catalog_hash": stable_hash(sorted(set(current_entity_types))),
            "domain": str(payload.get("domain") or "").strip(),
            "entity_types": tuple(sorted(set(str(value).strip() for value in payload.get("entity_types") or [] if str(value).strip()))),
            "relationship_guidance": tuple(str(value).strip() for value in payload.get("relationship_guidance") or [] if str(value).strip()),
            "extraction_instructions": str(payload.get("extraction_instructions") or "").strip(),
        }
        return PromptProposal(stable_hash(core), **core)


def save_proposal(path: Path, proposal: PromptProposal) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(asdict(proposal), indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def activate_proposal(
    proposal_path: Path,
    manifest_path: Path,
    manifest: CorpusManifest,
    *,
    expected_proposal_id: str,
) -> PromptProposal:
    payload = json.loads(proposal_path.read_text(encoding="utf-8"))
    proposal = PromptProposal(
        **{
            **payload,
            "sample_parent_ids": tuple(payload["sample_parent_ids"]),
            "entity_types": tuple(payload["entity_types"]),
            "relationship_guidance": tuple(payload["relationship_guidance"]),
        }
    )
    if proposal.id != expected_proposal_id:
        raise ValueError("Prompt proposal confirmation does not match the reviewed artifact.")
    if proposal.corpus_id != manifest.corpus_id or proposal.status != "REVIEW_REQUIRED":
        raise ValueError("Prompt proposal is not eligible for this corpus.")
    manifest.graph["active_prompt_hash"] = proposal.id
    manifest.graph["entity_types"] = list(proposal.entity_types)
    manifest.graph["relationship_guidance"] = list(proposal.relationship_guidance)
    save_manifest(manifest_path, manifest)
    active = PromptProposal(**{**asdict(proposal), "status": "ACTIVE"})
    save_proposal(proposal_path, active)
    return active
