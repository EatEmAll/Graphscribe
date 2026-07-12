from __future__ import annotations

import hashlib
import re
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import parse_qs, urlparse

from .ids import (
    block_id,
    canonical_file_identity,
    canonical_youtube_identity,
    document_id,
    revision_id,
)
from .models import CanonicalBlock, CanonicalDocument


def file_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ExtractionContext:
    corpus_id: str
    corpus_root: Path
    preferred_languages: tuple[str, ...] = ("en",)


class SourceAdapter(ABC):
    name = "base"
    version = "1"

    @abstractmethod
    def supports(self, source: object) -> bool: ...

    @abstractmethod
    def extract(self, source: object, context: ExtractionContext) -> CanonicalDocument: ...


def _decode_text(path: Path) -> tuple[str, tuple[str, ...]]:
    from charset_normalizer import from_bytes

    raw = path.read_bytes()
    if not raw:
        raise ValueError(f"Source file is empty: {path}")
    try:
        return raw.decode("utf-8"), ()
    except UnicodeDecodeError:
        best = from_bytes(raw).best()
        if best is None:
            return raw.decode("utf-8", errors="replace"), ("Encoding detection failed; invalid bytes were replaced.",)
        encoding = best.encoding or "utf-8"
        return str(best), (f"Decoded source using detected encoding {encoding}.",)


def _document(
    *,
    context: ExtractionContext,
    source_type: str,
    source_uri: str,
    relative_path: str | None,
    title: str,
    checksum: str,
    extractor: str,
    extractor_version: str,
    raw_blocks: Iterable[dict[str, Any]],
    language: str | None = None,
    metadata: dict[str, Any] | None = None,
    warnings: Iterable[str] = (),
) -> CanonicalDocument:
    doc_id = document_id(context.corpus_id, source_uri)
    rev_id = revision_id(doc_id, checksum, f"{extractor}:{extractor_version}")
    blocks: list[CanonicalBlock] = []
    for ordinal, raw in enumerate(raw_blocks):
        text = str(raw.get("text") or "").strip()
        block_type = str(raw.get("block_type") or "paragraph")
        if not text and block_type != "page_break":
            continue
        blocks.append(
            CanonicalBlock(
                block_id=block_id(rev_id, len(blocks), text),
                ordinal=len(blocks),
                block_type=block_type,
                text=text,
                section_path=tuple(raw.get("section_path") or ()),
                page_number=raw.get("page_number"),
                timestamp_start_ms=raw.get("timestamp_start_ms"),
                timestamp_end_ms=raw.get("timestamp_end_ms"),
                source_offset_start=raw.get("source_offset_start"),
                source_offset_end=raw.get("source_offset_end"),
                metadata=dict(raw.get("metadata") or {}),
            )
        )
    return CanonicalDocument(
        corpus_id=context.corpus_id,
        document_id=doc_id,
        revision_id=rev_id,
        source_type=source_type,
        source_uri=source_uri,
        relative_path=relative_path,
        title=title,
        language=language,
        source_checksum=checksum,
        extractor=extractor,
        extractor_version=extractor_version,
        extracted_at=datetime.now(timezone.utc).isoformat(),
        blocks=tuple(blocks),
        metadata=dict(metadata or {}),
        warnings=tuple(warnings),
    )


class TextAdapter(SourceAdapter):
    name = "text"

    def supports(self, source: object) -> bool:
        return isinstance(source, Path) and source.suffix.lower() == ".txt"

    def extract(self, source: object, context: ExtractionContext) -> CanonicalDocument:
        path = Path(source)
        text, warnings = _decode_text(path)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        blocks: list[dict[str, Any]] = []
        for match in re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", text, re.DOTALL):
            blocks.append(
                {
                    "block_type": "paragraph",
                    "text": match.group(0),
                    "source_offset_start": match.start(),
                    "source_offset_end": match.end(),
                }
            )
        relative = canonical_file_identity(path, context.corpus_root)
        return _document(
            context=context,
            source_type="text",
            source_uri=relative,
            relative_path=relative,
            title=path.stem,
            checksum=file_checksum(path),
            extractor=self.name,
            extractor_version=self.version,
            raw_blocks=blocks,
            warnings=warnings,
        )


class MarkdownAdapter(SourceAdapter):
    name = "markdown"

    def supports(self, source: object) -> bool:
        return isinstance(source, Path) and source.suffix.lower() in {".md", ".markdown"}

    def extract(self, source: object, context: ExtractionContext) -> CanonicalDocument:
        path = Path(source)
        text, warnings = _decode_text(path)
        blocks, metadata = parse_markdown_blocks(text)
        relative = canonical_file_identity(path, context.corpus_root)
        return _document(
            context=context,
            source_type="markdown",
            source_uri=relative,
            relative_path=relative,
            title=metadata.get("title") or path.stem,
            checksum=file_checksum(path),
            extractor=self.name,
            extractor_version=self.version,
            raw_blocks=blocks,
            metadata=metadata,
            warnings=warnings,
        )


def parse_markdown_blocks(text: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    metadata: dict[str, Any] = {}
    if normalized.startswith("---\n"):
        end = normalized.find("\n---\n", 4)
        if end != -1:
            try:
                import yaml

                parsed = yaml.safe_load(normalized[4:end]) or {}
                if isinstance(parsed, dict):
                    metadata = parsed
            except Exception:
                metadata["frontmatter_warning"] = "Frontmatter could not be parsed."
            normalized = normalized[end + 5 :]

    blocks: list[dict[str, Any]] = []
    headings: list[str] = []
    lines = normalized.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1
            continue
        heading = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading:
            level = len(heading.group(1))
            title = heading.group(2).strip()
            headings = headings[: level - 1] + [title]
            blocks.append({"block_type": "heading", "text": title, "section_path": tuple(headings)})
            index += 1
            continue
        if line.startswith("```"):
            code_lines = [line]
            index += 1
            while index < len(lines):
                code_lines.append(lines[index])
                closing = lines[index].startswith("```")
                index += 1
                if closing:
                    break
            blocks.append({"block_type": "code", "text": "\n".join(code_lines), "section_path": tuple(headings)})
            continue
        if "|" in line and index + 1 < len(lines) and re.match(r"^\s*\|?\s*:?-+", lines[index + 1]):
            table_lines = [line, lines[index + 1]]
            index += 2
            while index < len(lines) and "|" in lines[index] and lines[index].strip():
                table_lines.append(lines[index])
                index += 1
            blocks.append({"block_type": "table", "text": "\n".join(table_lines), "section_path": tuple(headings)})
            continue
        block_type = "list" if re.match(r"^\s*(?:[-*+] |\d+\. )", line) else "quote" if line.lstrip().startswith(">") else "paragraph"
        paragraph = [line]
        index += 1
        while index < len(lines) and lines[index].strip():
            if re.match(r"^(#{1,6})\s+", lines[index]) or lines[index].startswith("```"):
                break
            paragraph.append(lines[index])
            index += 1
        blocks.append({"block_type": block_type, "text": "\n".join(paragraph), "section_path": tuple(headings)})
    return blocks, metadata


class PdfAdapter(SourceAdapter):
    name = "docling"

    def __init__(self, converter_factory: Callable[[], Any] | None = None):
        self.converter_factory = converter_factory

    def supports(self, source: object) -> bool:
        return isinstance(source, Path) and source.suffix.lower() == ".pdf"

    def _converter(self) -> Any:
        if self.converter_factory:
            return self.converter_factory()
        from docling.document_converter import DocumentConverter

        return DocumentConverter()

    def extract(self, source: object, context: ExtractionContext) -> CanonicalDocument:
        path = Path(source)
        warnings: list[str] = []
        try:
            result = self._converter().convert(path)
            blocks = docling_blocks(result.document)
            if not blocks:
                markdown = result.document.export_to_markdown()
                blocks, _ = parse_markdown_blocks(markdown)
                warnings.append("Docling element provenance was unavailable; Markdown projection used.")
            metadata: dict[str, Any] = {}
            page_count = getattr(result.document, "num_pages", None)
            if page_count is not None:
                metadata["page_count"] = page_count
        except Exception as exc:
            warnings.append(f"Docling extraction failed; PyMuPDF fallback used: {exc}")
            from langchain_community.document_loaders import PyMuPDFLoader

            pages = PyMuPDFLoader(str(path)).load()
            blocks = [
                {"block_type": "paragraph", "text": page.page_content, "page_number": index + 1}
                for index, page in enumerate(pages)
                if page.page_content.strip()
            ]
            metadata = {"fallback": "PyMuPDF"}
        relative = canonical_file_identity(path, context.corpus_root)
        return _document(
            context=context,
            source_type="pdf",
            source_uri=relative,
            relative_path=relative,
            title=path.stem,
            checksum=file_checksum(path),
            extractor=self.name,
            extractor_version=self.version,
            raw_blocks=blocks,
            metadata=metadata,
            warnings=warnings,
        )


def _docling_position(item: Any) -> int:
    reference = str(getattr(item, "self_ref", ""))
    match = re.search(r"/(\d+)$", reference)
    return int(match.group(1)) if match else 10**9


def _docling_page(item: Any) -> int | None:
    provenance = list(getattr(item, "prov", None) or [])
    if not provenance:
        return None
    page = getattr(provenance[0], "page_no", None)
    return int(page) if page is not None else None


def docling_blocks(document: Any) -> list[dict[str, Any]]:
    """Project Docling items into ordered canonical blocks with page provenance."""
    items: list[tuple[int, dict[str, Any]]] = []
    headings: list[str] = []
    source_items = [*list(getattr(document, "texts", None) or []), *list(getattr(document, "tables", None) or [])]
    source_items.sort(key=_docling_position)
    for item in source_items:
        label_obj = getattr(item, "label", "paragraph")
        label = str(getattr(label_obj, "value", label_obj)).lower()
        if "table" in label or item in list(getattr(document, "tables", None) or []):
            exporter = getattr(item, "export_to_markdown", None)
            text = exporter(doc=document) if callable(exporter) else str(getattr(item, "text", ""))
            block_type = "table"
        else:
            text = str(getattr(item, "text", ""))
            block_type = "heading" if "heading" in label or "title" in label else "paragraph"
        text = text.strip()
        if not text:
            continue
        if block_type == "heading":
            level = int(getattr(item, "level", 1) or 1)
            headings = headings[: max(0, level - 1)] + [text]
        items.append(
            (
                _docling_position(item),
                {
                    "block_type": block_type,
                    "text": text,
                    "section_path": tuple(headings),
                    "page_number": _docling_page(item),
                },
            )
        )
    return [payload for _, payload in sorted(items, key=lambda row: row[0])]


@dataclass(frozen=True)
class YoutubeSource:
    url: str
    title: str | None = None
    preferred_languages: tuple[str, ...] = ("en",)


def youtube_video_id(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc.lower() in {"youtu.be", "www.youtu.be"}:
        value = parsed.path.strip("/").split("/")[0]
    elif "youtube.com" in parsed.netloc.lower():
        if parsed.path.startswith("/shorts/"):
            value = parsed.path.split("/")[2]
        else:
            value = parse_qs(parsed.query).get("v", [""])[0]
    else:
        value = ""
    if not re.fullmatch(r"[0-9A-Za-z_-]{11}", value):
        raise ValueError(f"Invalid YouTube URL: {url}")
    return value


class YoutubeAdapter(SourceAdapter):
    name = "youtube-transcript-api"

    def __init__(
        self,
        api_factory: Callable[[], Any] | None = None,
        subtitle_fallback: Callable[[str, tuple[str, ...]], tuple[list[dict[str, Any]], str | None]] | None = None,
    ):
        self.api_factory = api_factory
        self.subtitle_fallback = subtitle_fallback or _yt_dlp_transcript

    def supports(self, source: object) -> bool:
        return isinstance(source, YoutubeSource)

    def extract(self, source: object, context: ExtractionContext) -> CanonicalDocument:
        item = source if isinstance(source, YoutubeSource) else YoutubeSource(str(source))
        video_id = youtube_video_id(item.url)
        if self.api_factory:
            api = self.api_factory()
        else:
            from youtube_transcript_api import YouTubeTranscriptApi

            api = YouTubeTranscriptApi()
        languages = item.preferred_languages or context.preferred_languages
        try:
            transcript_list = api.list(video_id)
            try:
                transcript = transcript_list.find_manually_created_transcript(list(languages))
            except Exception:
                try:
                    transcript = transcript_list.find_generated_transcript(list(languages))
                except Exception:
                    transcript = next(iter(transcript_list))
            segments = transcript.fetch().to_raw_data()
            language = getattr(transcript, "language_code", None)
            generated = bool(getattr(transcript, "is_generated", False))
        except Exception:
            segments, language = self.subtitle_fallback(video_id, tuple(languages))
            generated = True
        raw_blocks = [
            {
                "block_type": "transcript",
                "text": segment["text"],
                "timestamp_start_ms": round(float(segment["start"]) * 1000),
                "timestamp_end_ms": round((float(segment["start"]) + float(segment.get("duration", 0))) * 1000),
            }
            for segment in segments
            if str(segment.get("text") or "").strip()
        ]
        identity = canonical_youtube_identity(video_id)
        checksum = hashlib.sha256(
            "\n".join(f"{item['timestamp_start_ms']}:{item['text']}" for item in raw_blocks).encode("utf-8")
        ).hexdigest()
        return _document(
            context=context,
            source_type="youtube",
            source_uri=identity,
            relative_path=None,
            title=item.title or video_id,
            checksum=checksum,
            extractor=self.name,
            extractor_version=self.version,
            raw_blocks=raw_blocks,
            language=language,
            metadata={
                "video_id": video_id,
                "is_generated": generated,
            },
        )


def _parse_vtt_timestamp(value: str) -> int:
    parts = value.replace(",", ".").split(":")
    seconds = float(parts[-1]) + (int(parts[-2]) * 60 if len(parts) > 1 else 0) + (int(parts[-3]) * 3600 if len(parts) > 2 else 0)
    return round(seconds * 1000)


def _yt_dlp_transcript(video_id: str, languages: tuple[str, ...]) -> tuple[list[dict[str, Any]], str | None]:
    with tempfile.TemporaryDirectory(prefix="corpus-youtube-") as temp_dir:
        output_template = str(Path(temp_dir) / "transcript.%(ext)s")
        command = [
            sys.executable,
            "-m",
            "yt_dlp",
            "--skip-download",
            "--write-subs",
            "--write-auto-subs",
            "--sub-format",
            "vtt",
            "--sub-langs",
            ",".join(languages) if languages else "all,-live_chat",
            "--output",
            output_template,
            canonical_youtube_identity(video_id),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "yt-dlp subtitle extraction failed")
        candidates = sorted(Path(temp_dir).glob("*.vtt"))
        if not candidates:
            raise RuntimeError("yt-dlp did not produce a subtitle file")
        subtitle_path = candidates[0]
        language_match = re.search(r"\.([A-Za-z-]+)\.vtt$", subtitle_path.name)
        language = language_match.group(1) if language_match else None
        segments: list[dict[str, Any]] = []
        lines = subtitle_path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
        index = 0
        while index < len(lines):
            timing = re.match(r"([^ ]+)\s+-->\s+([^ ]+)", lines[index])
            if not timing:
                index += 1
                continue
            start_ms = _parse_vtt_timestamp(timing.group(1))
            end_ms = _parse_vtt_timestamp(timing.group(2))
            index += 1
            text_lines: list[str] = []
            while index < len(lines) and lines[index].strip():
                text_lines.append(re.sub(r"<[^>]+>", "", lines[index]).strip())
                index += 1
            text = " ".join(part for part in text_lines if part)
            if text:
                segments.append({"text": text, "start": start_ms / 1000, "duration": (end_ms - start_ms) / 1000})
        if not segments:
            raise RuntimeError("yt-dlp subtitle file contained no transcript cues")
        return segments, language


DEFAULT_ADAPTERS: tuple[SourceAdapter, ...] = (
    TextAdapter(),
    MarkdownAdapter(),
    PdfAdapter(),
    YoutubeAdapter(),
)


def adapter_for(source: object, adapters: Iterable[SourceAdapter] = DEFAULT_ADAPTERS) -> SourceAdapter:
    for adapter in adapters:
        if adapter.supports(source):
            return adapter
    raise ValueError(f"No source adapter supports {source!r}")
