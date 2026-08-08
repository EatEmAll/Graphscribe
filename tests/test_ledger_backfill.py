import json

import pytest

from notebooklm_graph_pipe.ingestion.ledger_backfill import backfill_run_id, reconcile_inventory


CORPUS_ID = "66b7034e-5c48-584b-98e9-d78091e74445"
NOTEBOOK_ID = "dfae4958-fe79-4b26-bb93-dd50b6561adb"


def inventory(tmp_path, rows):
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps({"sources": rows}), encoding="utf-8")
    return path


def rows(count=244):
    return [
        {
            "source_id": f"source-{index:03}", "title": f"Title {index}",
            "source_type": "youtube", "url": None,
            "relative_path": f"documents/source-{index:03}.md", "sha256": f"checksum-{index}",
            "acquisition": "notebooklm_text_fallback",
        }
        for index in range(count)
    ]


def test_reconciliation_requires_exact_244_ids(tmp_path):
    old = rows()
    live = [{"id": row["source_id"], "title": row["title"], "type": row["source_type"]} for row in old[:-1]]
    with pytest.raises(ValueError, match="source IDs differ"):
        reconcile_inventory(corpus_id=CORPUS_ID, notebook_id=NOTEBOOK_ID,
                            inventory_path=inventory(tmp_path, old), live_sources=live,
                            canonical_documents=[], legacy_documents=[])


def test_reconciliation_builds_active_and_legacy_only_without_guessing_urls(tmp_path):
    old = rows()
    for index in range(237, 244):
        old[index]["relative_path"] = None
        old[index]["sha256"] = None
        old[index]["url"] = f"https://youtu.be/{index:011d}"
    live = [{"id": row["source_id"], "title": row["title"], "type": row["source_type"]} for row in old]
    canonical = [
        {"source_id": row["source_id"], "document_id": f"doc-{index}", "title": row["title"],
         "checksum": row["sha256"], "vector_ready": True, "graph_ready": True}
        for index, row in enumerate(old[:237])
    ]
    legacy = [{"source_id": row["source_id"], "document_id": f"legacy-{index}"} for index, row in enumerate(old)]
    reconciled = reconcile_inventory(corpus_id=CORPUS_ID, notebook_id=NOTEBOOK_ID,
                                     inventory_path=inventory(tmp_path, old), live_sources=live,
                                     canonical_documents=canonical, legacy_documents=legacy)
    assert sum(row["retrieval_status"] == "ACTIVE" for row in reconciled) == 237
    assert sum(row["retrieval_status"] == "LEGACY_ONLY" for row in reconciled) == 7
    assert all(row["canonical_uri"] is None for row in reconciled[:237])
    assert len({row["id"] for row in reconciled}) == 244
    assert backfill_run_id(reconciled) == backfill_run_id(list(reversed(reconciled)))
