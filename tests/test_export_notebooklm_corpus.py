from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from scripts import export_notebooklm_corpus as exporter


def completed(payload: object, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], returncode, json.dumps(payload), "failure" if returncode else "")


def test_nlm_client_parses_list_and_wrapped_content() -> None:
    responses = iter(
        [
            completed([{"id": "n1", "title": "Notebook"}]),
            completed([{"id": "s1", "title": "Source", "type": "pdf", "url": None}]),
            completed({"value": {"content": "body", "title": "Source", "source_type": "pdf"}}),
        ]
    )
    client = exporter.NlmClient(runner=lambda *args, **kwargs: next(responses))

    assert client.notebooks() == [exporter.Notebook("n1", "Notebook")]
    source = client.sources("n1")[0]
    assert source == exporter.Source("s1", "Source", "pdf")
    assert client.source_content(source)["content"] == "body"


def test_nlm_client_rejects_command_failure() -> None:
    client = exporter.NlmClient(runner=lambda *args, **kwargs: completed({}, returncode=1))
    with pytest.raises(exporter.ExportError, match="NotebookLM command failed"):
        client.notebooks()


def test_extract_source_ids_ignores_non_uuid_names() -> None:
    expected = "cef5b79e-b1dd-475a-adc6-69dc280c3d1c"
    assert exporter.extract_source_ids([f"source_4_{expected}.txt", "unrelated.txt"]) == {expected}


class FakeClient:
    def __init__(self, details: dict[str, dict[str, object]]):
        self.details = details

    def source_content(self, source: exporter.Source) -> dict[str, object]:
        return self.details[source.id]


def test_export_deduplicates_same_source_id_and_preserves_provenance(tmp_path: Path) -> None:
    notebooks = [exporter.Notebook("n1", "One"), exporter.Notebook("n2", "Two")]
    shared = exporter.Source("cef5b79e-b1dd-475a-adc6-69dc280c3d1c", "A: title", "pdf")
    current = exporter.Source("258af73f-18f2-4d6c-af9d-044e0273b341", "New", "web")
    source_map = {"n1": [shared], "n2": [shared, current]}
    details = {
        shared.id: {"content": "shared body", "title": shared.title, "source_type": "pdf"},
        current.id: {"content": "new body", "title": current.title, "source_type": "web"},
    }

    inventory = exporter.export_sources(
        FakeClient(details), notebooks, source_map, {shared.id}, tmp_path / "out"
    )

    assert inventory["summary"] == {
        "legacy_source_ids": 1,
        "current_sources": 2,
        "legacy_matched": 1,
        "current_new": 1,
        "legacy_missing": 0,
    }
    record = next(row for row in inventory["sources"] if row["source_id"] == shared.id)
    assert [item["id"] for item in record["notebooks"]] == ["n1", "n2"]
    markdown = (tmp_path / "out" / record["relative_path"]).read_text(encoding="utf-8")
    assert "notebooklm_text_fallback" in markdown
    assert "shared body" in markdown


def test_export_prefers_unique_original_file(tmp_path: Path) -> None:
    originals = tmp_path / "originals"
    originals.mkdir()
    (originals / "paper.pdf").write_bytes(b"pdf")
    source = exporter.Source("258af73f-18f2-4d6c-af9d-044e0273b341", "paper.pdf", "pdf")

    inventory = exporter.export_sources(
        FakeClient({source.id: {"content": "fallback", "title": "paper.pdf", "source_type": "pdf"}}),
        [exporter.Notebook("n1", "One")],
        {"n1": [source]},
        {source.id},
        tmp_path / "out",
        originals_root=originals,
    )

    record = inventory["sources"][0]
    assert record["acquisition"] == "original_file"
    assert record["relative_path"].startswith("documents/paper__")
    assert (tmp_path / "out" / record["relative_path"]).read_bytes() == b"pdf"


def test_export_writes_youtube_declaration_when_url_is_available(tmp_path: Path) -> None:
    source = exporter.Source("cef5b79e-b1dd-475a-adc6-69dc280c3d1c", "Video", "youtube")
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    inventory = exporter.export_sources(
        FakeClient({source.id: {"content": "transcript", "title": "Video", "source_type": "youtube", "url": url}}),
        [exporter.Notebook("n1", "One")],
        {"n1": [source]},
        {source.id},
        tmp_path / "out",
    )

    assert inventory["sources"][0]["acquisition"] == "youtube_url"
    payload = yaml.safe_load((tmp_path / "out" / "sources.yaml").read_text(encoding="utf-8"))
    assert payload["youtube"][0]["url"] == url


def test_export_recovers_youtube_url_from_title(tmp_path: Path) -> None:
    url = "https://youtu.be/dQw4w9WgXcQ"
    source = exporter.Source("cef5b79e-b1dd-475a-adc6-69dc280c3d1c", url, "youtube")

    inventory = exporter.export_sources(
        FakeClient({source.id: {"content": "", "title": url, "source_type": "youtube", "url": None}}),
        [exporter.Notebook("n1", "One")],
        {"n1": [source]},
        {source.id},
        tmp_path / "out",
    )

    payload = yaml.safe_load((tmp_path / "out" / "sources.yaml").read_text(encoding="utf-8"))
    assert inventory["sources"][0]["acquisition"] == "youtube_url"
    assert payload["youtube"][0]["url"] == url


def test_export_rejects_empty_content(tmp_path: Path) -> None:
    source = exporter.Source("cef5b79e-b1dd-475a-adc6-69dc280c3d1c", "Empty", "pdf")
    with pytest.raises(exporter.ExportError, match="no recoverable content"):
        exporter.export_sources(
            FakeClient({source.id: {"content": "", "title": "Empty", "source_type": "pdf"}}),
            [exporter.Notebook("n1", "One")],
            {"n1": [source]},
            {source.id},
            tmp_path / "out",
        )


def test_duplicate_youtube_urls_preserve_distinct_source_ids_as_text(tmp_path: Path) -> None:
    first = exporter.Source("cef5b79e-b1dd-475a-adc6-69dc280c3d1c", "First", "youtube")
    second = exporter.Source("258af73f-18f2-4d6c-af9d-044e0273b341", "Second", "youtube")
    url = "https://youtu.be/dQw4w9WgXcQ"
    inventory = exporter.export_sources(
        FakeClient(
            {
                first.id: {"content": "first", "title": "First", "source_type": "youtube", "url": url},
                second.id: {"content": "second", "title": "Second", "source_type": "youtube", "url": url},
            }
        ),
        [exporter.Notebook("n1", "One")],
        {"n1": [first, second]},
        {first.id, second.id},
        tmp_path / "out",
    )

    assert {row["acquisition"] for row in inventory["sources"]} == {"notebooklm_text_fallback"}
    assert not (tmp_path / "out" / "sources.yaml").exists()


def test_overwrite_requires_export_marker(tmp_path: Path) -> None:
    output = tmp_path / "out"
    output.mkdir()
    (output / "user-file.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(exporter.ExportError, match="unmarked"):
        exporter.prepare_output(output, overwrite=True)
    assert (output / "user-file.txt").read_text(encoding="utf-8") == "keep"


def test_overwrite_rejects_forged_export_marker(tmp_path: Path) -> None:
    output = tmp_path / "out"
    output.mkdir()
    (output / exporter.MARKER_NAME).write_text("{}", encoding="utf-8")
    (output / "user-file.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(exporter.ExportError, match="unmarked"):
        exporter.prepare_output(output, overwrite=True)
    assert (output / "user-file.txt").read_text(encoding="utf-8") == "keep"


def test_inventory_contains_no_neo4j_credentials(tmp_path: Path) -> None:
    source = exporter.Source("cef5b79e-b1dd-475a-adc6-69dc280c3d1c", "Source", "pdf")
    exporter.export_sources(
        FakeClient({source.id: {"content": "body", "title": "Source", "source_type": "pdf"}}),
        [exporter.Notebook("n1", "One")],
        {"n1": [source]},
        {source.id},
        tmp_path / "out",
    )

    serialized = (tmp_path / "out" / "source_inventory.json").read_text(encoding="utf-8")
    assert "NEO4J_PASSWORD" not in serialized
    assert "password" not in serialized.casefold()
