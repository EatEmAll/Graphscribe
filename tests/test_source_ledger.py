from dataclasses import replace

import pytest

from notebooklm_graph_pipe.ingestion.ids import ledger_source_id
from notebooklm_graph_pipe.ingestion.models import CanonicalBlock, CanonicalDocument
from notebooklm_graph_pipe.ingestion.source_ledger import canonical_uri, identity_from_document


CORPUS_ID = "66b7034e-5c48-584b-98e9-d78091e74445"


def document(**metadata):
    return CanonicalDocument(
        corpus_id=CORPUS_ID,
        document_id="old-document",
        revision_id="revision",
        source_type="markdown",
        source_uri="moved/source.md",
        title="Title",
        source_checksum="abc",
        extractor="markdown",
        extractor_version="1",
        blocks=(CanonicalBlock("b", 0, "paragraph", "text"),),
        metadata=metadata,
    )


def test_ledger_id_is_deterministic_and_provider_scoped():
    assert ledger_source_id(CORPUS_ID, "YouTube", "YUgCUKfZq4E") == ledger_source_id(
        CORPUS_ID, "youtube", "YUgCUKfZq4E"
    )
    assert ledger_source_id(CORPUS_ID, "youtube", "YUgCUKfZq4E") != ledger_source_id(
        CORPUS_ID, "notebooklm", "YUgCUKfZq4E"
    )


def test_youtube_uri_normalization():
    assert canonical_uri("https://youtu.be/YUgCUKfZq4E?t=2") == "https://www.youtube.com/watch?v=YUgCUKfZq4E"


def test_notebooklm_identity_takes_precedence_over_path():
    first = identity_from_document(document(notebooklm={"source_id": "nlm-1", "notebook_ids": ["nb"]}))
    moved = identity_from_document(replace(document(notebooklm={"source_id": "nlm-1"}), source_uri="other.md"))
    assert first.id == moved.id
    assert first.provider == "notebooklm"


def test_empty_provider_identity_is_rejected():
    with pytest.raises(ValueError):
        ledger_source_id(CORPUS_ID, "", "x")
