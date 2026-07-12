from __future__ import annotations

import secrets
from pathlib import Path


def load_or_create_token(path: Path) -> str:
    if path.exists():
        value = path.read_text(encoding="utf-8").strip()
        if len(value) < 32:
            raise ValueError(f"API token is too short: {path}")
        return value
    path.parent.mkdir(parents=True, exist_ok=True)
    value = secrets.token_urlsafe(48)
    path.write_text(value + "\n", encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return value
