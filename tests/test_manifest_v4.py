from __future__ import annotations

import json

from notebooklm_graph_pipe.ingestion.manifest import CorpusManifest, load_manifest, save_manifest


def test_v3_manifest_is_normalized_to_v6_without_enabling_features(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "version": 3,
                "corpus": {"id": "c1", "key": "demo", "title": "Demo"},
                "neo4j": {"uri": "bolt://test"},
                "embedding": {
                    "provider": "sentence-transformer",
                    "model": "all-MiniLM-L6-v2",
                    "dimension": 384,
                    "normalized": True,
                },
                "retrieval": {"unit": "chunk"},
                "sources": {},
            }
        ),
        encoding="utf-8",
    )

    manifest = load_manifest(path)

    assert manifest is not None
    assert manifest.version == 6
    assert manifest.community["enabled"] is False
    assert manifest.graph["claims_enabled"] is False
    assert manifest.graph["extraction_mode"] == "llm"
    assert manifest.retrieval_vector_provider == "neo4j"
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == 3


def test_v4_manifest_round_trip_preserves_new_configuration(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    manifest = CorpusManifest(
        "c1",
        "demo",
        "Demo",
        {"uri": "bolt://test"},
        retrieval_vector_provider="lancedb",
        retrieval_vector_location="vectors/demo",
        community={"enabled": True, "algorithm": "hierarchical_leiden"},
    )

    assert manifest.community["seed"] == 42

    save_manifest(path, manifest)
    loaded = load_manifest(path)

    assert loaded is not None
    assert loaded.retrieval_vector_provider == "lancedb"
    assert loaded.retrieval_vector_location == "vectors/demo"
    assert loaded.community["enabled"] is True
    assert loaded.community["seed"] == 42
