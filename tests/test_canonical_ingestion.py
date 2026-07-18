from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from notebooklm_graph_pipe.ingestion.adapters import (
    ExtractionContext,
    MarkdownAdapter,
    TextAdapter,
    docling_blocks,
    YoutubeAdapter,
    YoutubeSource,
    _yt_dlp_transcript,
    youtube_video_id,
)
from notebooklm_graph_pipe.ingestion.chunking import ChunkingConfig, HierarchicalChunker
from notebooklm_graph_pipe.ingestion.ids import corpus_id
from notebooklm_graph_pipe.ingestion.manifest import CorpusManifest, SourceManifestEntry, load_manifest, save_manifest


class WordTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return list(range(len(text.split())))

    def decode(self, token_ids, *, skip_special_tokens: bool = True) -> str:
        return " ".join(f"word{token}" for token in token_ids)


def context(root: Path) -> ExtractionContext:
    return ExtractionContext(corpus_id("demo"), root)


def test_text_adapter_is_stable_and_document_scoped(tmp_path: Path) -> None:
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_text("Repeated paragraph.\n\nSecond paragraph.", encoding="utf-8")
    second.write_text("Repeated paragraph.\n\nSecond paragraph.", encoding="utf-8")
    adapter = TextAdapter()

    first_doc = adapter.extract(first, context(tmp_path))
    repeated = adapter.extract(first, context(tmp_path))
    second_doc = adapter.extract(second, context(tmp_path))

    assert first_doc.document_id == repeated.document_id
    assert first_doc.revision_id == repeated.revision_id
    assert first_doc.document_id != second_doc.document_id
    assert first_doc.blocks[0].block_id != second_doc.blocks[0].block_id


def test_markdown_adapter_preserves_structure_and_frontmatter(tmp_path: Path) -> None:
    source = tmp_path / "notes.md"
    source.write_text(
        "---\ntitle: Research Notes\ntag: test\n---\n# Heading\n\nParagraph text.\n\n```python\nprint('x')\n```\n",
        encoding="utf-8",
    )
    document = MarkdownAdapter().extract(source, context(tmp_path))

    assert document.title == "Research Notes"
    assert [block.block_type for block in document.blocks] == ["heading", "paragraph", "code"]
    assert document.blocks[1].section_path == ("Heading",)
    assert "tag" in document.metadata
    assert "title: Research Notes" not in document.text


def test_hierarchical_chunker_enforces_child_limit(tmp_path: Path) -> None:
    source = tmp_path / "large.txt"
    source.write_text(" ".join(f"token-{index}" for index in range(100)), encoding="utf-8")
    document = TextAdapter().extract(source, context(tmp_path))
    chunker = HierarchicalChunker(
        WordTokenizer(),
        ChunkingConfig(
            child_target_tokens=20,
            child_max_tokens=24,
            child_overlap_tokens=4,
            child_min_tokens=4,
            parent_target_tokens=50,
            parent_max_tokens=120,
        ),
    )
    result = chunker.chunk(document)

    assert result.parents
    assert result.children
    assert all(chunk.token_count <= 24 for chunk in result.children)
    assert all(chunk.parent_id in {parent.id for parent in result.parents} for chunk in result.children)
    assert len({chunk.id for chunk in result.children}) == len(result.children)
    assert all(parent.token_count <= 120 for parent in result.parents)


def test_table_chunks_repeat_header(tmp_path: Path) -> None:
    source = tmp_path / "table.md"
    source.write_text(
        "# Data\n\n| Name | Value |\n| --- | --- |\n" + "\n".join(f"| row-{i} | value-{i} |" for i in range(20)),
        encoding="utf-8",
    )
    document = MarkdownAdapter().extract(source, context(tmp_path))
    result = HierarchicalChunker(
        WordTokenizer(),
        ChunkingConfig(child_target_tokens=20, child_max_tokens=24, child_overlap_tokens=4, child_min_tokens=4),
    ).chunk(document)
    table_children = [chunk for chunk in result.children if "| Name | Value |" in chunk.text]
    assert len(table_children) > 1
    assert all("| --- | --- |" in chunk.text for chunk in table_children)


def test_docling_projection_preserves_page_and_heading() -> None:
    class Label:
        def __init__(self, value):
            self.value = value

    class Prov:
        def __init__(self, page_no):
            self.page_no = page_no

    class Item:
        def __init__(self, ref, text, label, page, level=1):
            self.self_ref = ref
            self.text = text
            self.label = Label(label)
            self.prov = [Prov(page)]
            self.level = level

    class Document:
        texts = [Item("#/texts/0", "Methods", "section_heading", 2), Item("#/texts/1", "Details", "paragraph", 2)]
        tables = []

    blocks = docling_blocks(Document())
    assert blocks[0]["block_type"] == "heading"
    assert blocks[1]["section_path"] == ("Methods",)
    assert blocks[1]["page_number"] == 2


def test_youtube_adapter_uses_fallback_when_api_fails(tmp_path: Path) -> None:
    class Api:
        def list(self, video_id):
            raise RuntimeError("blocked")

    def fallback(video_id, languages):
        return ([{"text": "fallback", "start": 2.0, "duration": 1.0}], "en")

    document = YoutubeAdapter(api_factory=Api, subtitle_fallback=fallback).extract(
        YoutubeSource("https://youtu.be/abcdefghijk"),
        context(tmp_path),
    )
    assert document.blocks[0].text == "fallback"
    assert document.blocks[0].timestamp_start_ms == 2000


def test_youtube_fallback_uses_active_python_environment(monkeypatch) -> None:
    captured = []

    def run(command, **kwargs):
        captured.append(command)
        return SimpleNamespace(returncode=1, stderr="expected failure")

    monkeypatch.setattr("notebooklm_graph_pipe.ingestion.adapters.subprocess.run", run)
    with pytest.raises(RuntimeError, match="expected failure"):
        _yt_dlp_transcript("abcdefghijk", ("en",))

    assert captured[0][:3] == [sys.executable, "-m", "yt_dlp"]


def test_manifest_v3_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    manifest = CorpusManifest(
        corpus_id=corpus_id("demo"),
        corpus_key="demo",
        title="Demo",
        neo4j={"uri": "bolt://127.0.0.1:7687"},
        sources={
            "a.txt": SourceManifestEntry(
                document_id="doc",
                active_revision_id="revision",
                checksum="sum",
                extractor="text",
                extractor_version="1",
            )
        },
    )
    save_manifest(path, manifest)

    loaded = load_manifest(path)

    assert loaded is not None
    assert loaded.to_dict() == manifest.to_dict()
    assert loaded.retrieval_unit == "chunk"
    assert loaded.retrieval_vector_index == "chunk_embedding_v1"


def test_manifest_v3_parent_retrieval_profile_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    manifest = CorpusManifest(
        corpus_id=corpus_id("compact"),
        corpus_key="compact",
        title="Compact",
        neo4j={"uri": "bolt://127.0.0.1:7687"},
        retrieval_unit="parent",
        retrieval_vector_index="parent_embedding_v1",
        retrieval_keyword_index="parent_keyword_v1",
    )
    save_manifest(path, manifest)

    loaded = load_manifest(path)

    assert loaded is not None
    assert loaded.retrieval_unit == "parent"
    assert loaded.to_dict()["retrieval"] == {
        "unit": "parent",
        "vector_index": "parent_embedding_v1",
        "keyword_index": "parent_keyword_v1",
    }


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://www.youtube.com/watch?v=abcdefghijk", "abcdefghijk"),
        ("https://youtu.be/abcdefghijk", "abcdefghijk"),
        ("https://www.youtube.com/shorts/abcdefghijk", "abcdefghijk"),
    ],
)
def test_youtube_url_normalization(url: str, expected: str) -> None:
    assert youtube_video_id(url) == expected


def test_youtube_adapter_prefers_manual_transcript(tmp_path: Path) -> None:
    class Transcript:
        language_code = "en"
        is_generated = False

        def fetch(self):
            return self

        def to_raw_data(self):
            return [{"text": "hello world", "start": 1.5, "duration": 2.0}]

    class TranscriptList:
        def find_manually_created_transcript(self, languages):
            return Transcript()

        def __iter__(self):
            yield Transcript()

    class Api:
        def list(self, video_id):
            assert video_id == "abcdefghijk"
            return TranscriptList()

    document = YoutubeAdapter(api_factory=Api).extract(
        YoutubeSource("https://youtu.be/abcdefghijk"),
        context(tmp_path),
    )

    assert document.source_uri == "https://www.youtube.com/watch?v=abcdefghijk"
    assert document.blocks[0].timestamp_start_ms == 1500
    assert document.blocks[0].timestamp_end_ms == 3500
