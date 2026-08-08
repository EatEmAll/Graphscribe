from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .ids import ledger_source_id
from .models import CanonicalDocument


LEDGER_VERSION = 2


def content_fingerprint(text: str) -> str:
    normalized = " ".join(text.replace("\r\n", "\n").replace("\r", "\n").split()).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


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
    notebooklm_source_id: str | None = None
    ledger_id: str | None = None

    @property
    def id(self) -> str:
        return self.ledger_id or ledger_source_id(self.corpus_id, self.provider, self.provider_source_id)

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row.pop("ledger_id", None)
        return {**row, "id": self.id, "notebook_ids": list(self.notebook_ids)}


def identity_from_document(document: CanonicalDocument) -> SourceIdentity:
    fingerprint = content_fingerprint(document.text)
    notebook = document.metadata.get("notebooklm") if isinstance(document.metadata, dict) else None
    if isinstance(notebook, dict) and notebook.get("source_id"):
        uri = notebook.get("original_url") or document.source_uri
        acquisition = notebook.get("acquisition")
        notebook_ids = tuple(str(value) for value in notebook.get("notebook_ids") or ())
        source_type = str(notebook.get("source_type") or document.source_type)
        normalized = canonical_uri(uri)
        if source_type == "youtube":
            provider = "youtube"
            provider_source_id = (
                normalized.rsplit("=", 1)[-1]
                if normalized and "youtube.com/watch?v=" in normalized
                else f"content:{fingerprint}:notebooklm:{notebook['source_id']}"
            )
        else:
            provider = "document"
            provider_source_id = f"content:{fingerprint}:notebooklm:{notebook['source_id']}"
        notebooklm_source_id = str(notebook["source_id"])
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
        notebooklm_source_id = None
    return SourceIdentity(
        corpus_id=document.corpus_id,
        provider=provider,
        provider_source_id=provider_source_id,
        title=document.title,
        source_type=source_type,
        canonical_uri=canonical_uri(uri),
        content_checksum=fingerprint,
        acquisition_method=str(acquisition) if acquisition else None,
        notebook_ids=notebook_ids,
        notebooklm_source_id=notebooklm_source_id,
    )
