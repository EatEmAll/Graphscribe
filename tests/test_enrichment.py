from __future__ import annotations

from pathlib import Path

import pytest

from notebooklm_graph_pipe.enrichment.claims import ClaimExtractor, claim_rows
from notebooklm_graph_pipe.enrichment.prompt_tuning import PromptTuner, activate_proposal, save_proposal
from notebooklm_graph_pipe.enrichment.provisional import ProvisionalConfig, ProvisionalGraphExtractor
from notebooklm_graph_pipe.ingestion.manifest import CorpusManifest, load_manifest, save_manifest
from notebooklm_graph_pipe.runtime.llm_routing import CLAIM_EXTRACTION_ROLE, PROMPT_TUNING_ROLE
from notebooklm_graph_pipe.runtime.model_executor import ModelExecutor, ModelUsage


class Adapter:
    provider = "test"
    model = "test"

    def execute(self, request):
        if request.role == PROMPT_TUNING_ROLE:
            payload = {
                "domain": "software architecture",
                "entity_types": ["Component", "Decision", "Component"],
                "relationship_guidance": ["Connect decisions to affected components."],
                "extraction_instructions": "Extract explicit design decisions.",
            }
        else:
            payload = {
                "claims": [
                    {
                        "subject": "GraphScribe",
                        "predicate": "uses",
                        "object": "Neo4j",
                        "stance": "SUPPORTS",
                        "valid_from": "2025-01-01",
                        "valid_to": None,
                        "extraction_confidence": 0.9,
                    }
                ]
            }
        return "", payload, ModelUsage()


def executor() -> ModelExecutor:
    return ModelExecutor(
        {"test": Adapter()},
        {PROMPT_TUNING_ROLE: "test", CLAIM_EXTRACTION_ROLE: "test"},
    )


def test_prompt_proposal_requires_exact_reviewed_id_before_activation(tmp_path: Path) -> None:
    proposal = PromptTuner(executor()).propose(
        "corpus",
        [{"parent_id": "p1", "text": "GraphScribe uses Neo4j."}],
        ["System"],
    )
    proposal_path = tmp_path / "proposal.json"
    manifest_path = tmp_path / "manifest.json"
    manifest = CorpusManifest("corpus", "demo", "Demo", {"uri": "bolt://test"})
    save_proposal(proposal_path, proposal)
    save_manifest(manifest_path, manifest)

    with pytest.raises(ValueError, match="confirmation"):
        activate_proposal(proposal_path, manifest_path, manifest, expected_proposal_id="wrong")

    active = activate_proposal(
        proposal_path,
        manifest_path,
        manifest,
        expected_proposal_id=proposal.id,
    )
    assert active.status == "ACTIVE"
    assert load_manifest(manifest_path).graph["active_prompt_hash"] == proposal.id
    assert '"status": "ACTIVE"' in proposal_path.read_text(encoding="utf-8")


def test_provisional_extraction_prunes_cross_document_hubs() -> None:
    extractor = ProvisionalGraphExtractor(ProvisionalConfig(max_document_frequency_ratio=0.5))
    result = extractor.extract_batch(
        [
            {"parent_id": "p1", "text": "GraphScribe connects Neo4j. GraphScribe architecture architecture."},
            {"parent_id": "p2", "text": "GraphScribe exports Parquet. GraphScribe analytics analytics."},
        ]
    )

    assert all(node.id != "graphscribe" for graph in result.values() for node in graph.nodes)
    assert any(node.id == "architecture" for node in result["p1"].nodes)
    assert all(relationship.type == "CO_OCCURS" for graph in result.values() for relationship in graph.relationships)


def test_claim_extraction_is_temporal_and_parent_grounded() -> None:
    claims = ClaimExtractor(executor()).extract("parent-1", "GraphScribe uses Neo4j.")

    assert claims[0].valid_from == "2025-01-01"
    assert claims[0].source_parent_id == "parent-1"
    assert claim_rows(claims)[0]["stance"] == "SUPPORTS"
