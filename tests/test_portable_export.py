from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from notebooklm_graph_pipe.exports.portable import export_portable_projection


class Session:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def run(self, query, **parameters):
        if "RETURN entity.id AS entity_id" in query:
            return [
                {
                    "entity_id": "entity-a",
                    "entity_type": "System",
                    "description": "A",
                    "source_parent_ids": ["parent"],
                    "extraction_states": ["VERIFIED"],
                },
                {
                    "entity_id": "entity-b",
                    "entity_type": "System",
                    "description": "B",
                    "source_parent_ids": ["parent"],
                    "extraction_states": ["VERIFIED"],
                },
            ]
        if "RETURN source.id AS source_id" in query:
            return [
                {
                    "source_id": "entity-a",
                    "target_id": "entity-b",
                    "relationship_type": "USES",
                    "description": "uses",
                    "source_parent_ids": ["parent"],
                    "provisional_parent_ids": [],
                }
            ]
        return []


class Driver:
    def session(self, *, database):
        return Session()


def test_portable_export_writes_versioned_parquet_manifest_and_graphml(tmp_path: Path) -> None:
    result = export_portable_projection(Driver(), "neo4j", "corpus", tmp_path, graphml=True)
    target = Path(result["path"])
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["authority"] == "neo4j"
    assert manifest["export_schema_version"] == 1
    assert manifest["files"]["entities.parquet"]["rows"] == 2
    assert manifest["files"]["relationships.parquet"]["rows"] == 1
    assert (target / "graph.graphml").is_file()
    assert pq.read_table(target / "entities.parquet").num_rows == 2
