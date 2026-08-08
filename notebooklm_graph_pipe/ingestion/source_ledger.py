from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .ids import ledger_source_id
from .models import CanonicalDocument


LEDGER_VERSION = 1


def canonical_uri(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip()
    parsed = urlparse(value)
    if parsed.scheme in {"http", "https"}:
        host = parsed.netloc.lower().removeprefix("www.")
        path = parsed.path.rstrip("/") or "/"
        if host in {"youtube.com", "m.youtube.com", "youtu.be"}:
            video_id = path.strip("/") if host == "youtu.be" else parse_qs(parsed.query).get("v", [""])[0]
            if re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
                return f"https://www.youtube.com/watch?v={video_id}"
        return parsed._replace(scheme=parsed.scheme.lower(), netloc=host, path=path, fragment="").geturl()
    return Path(value).as_posix().casefold()


@dataclass(frozen=True)
class SourceIdentity:
    corpus_id: str
    provider: str
    provider_source_id: str
    title: str
    source_type: str
    canonical_uri: str | None
    content_checksum: str
    acquisition_method: str | None = None
    notebook_ids: tuple[str, ...] = ()

    @property
    def id(self) -> str:
        return ledger_source_id(self.corpus_id, self.provider, self.provider_source_id)

    def to_row(self) -> dict[str, Any]:
        return {**asdict(self), "id": self.id, "notebook_ids": list(self.notebook_ids)}


def identity_from_document(document: CanonicalDocument) -> SourceIdentity:
    notebook = document.metadata.get("notebooklm") if isinstance(document.metadata, dict) else None
    if isinstance(notebook, dict) and notebook.get("source_id"):
        provider = "notebooklm"
        provider_source_id = str(notebook["source_id"])
        uri = notebook.get("original_url") or document.source_uri
        acquisition = notebook.get("acquisition")
        notebook_ids = tuple(str(value) for value in notebook.get("notebook_ids") or ())
        source_type = str(notebook.get("source_type") or document.source_type)
    else:
        uri = document.source_uri
        normalized = canonical_uri(uri)
        if normalized and "youtube.com/watch?v=" in normalized:
            provider = "youtube"
            provider_source_id = normalized.rsplit("=", 1)[-1]
        else:
            provider = "file"
            provider_source_id = normalized or uri
        acquisition = document.extractor
        notebook_ids = ()
        source_type = document.source_type
    return SourceIdentity(
        corpus_id=document.corpus_id,
        provider=provider,
        provider_source_id=provider_source_id,
        title=document.title,
        source_type=source_type,
        canonical_uri=canonical_uri(uri),
        content_checksum=document.source_checksum,
        acquisition_method=str(acquisition) if acquisition else None,
        notebook_ids=notebook_ids,
    )
