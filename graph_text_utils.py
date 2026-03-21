from __future__ import annotations

from typing import Any


def coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [str(item).strip() for item in value if str(item).strip()]
        return " | ".join(parts)
    return str(value).strip()


def sorted_unique_texts(values: list[Any]) -> list[str]:
    cleaned = {str(value).strip() for value in values if str(value).strip()}
    return sorted(cleaned, key=str.casefold)


def normalize_name(value: str) -> str:
    lowered = "".join(ch.lower() if ch.isalnum() else " " for ch in value)
    return " ".join(lowered.split())


def token_set(value: str) -> set[str]:
    normalized = normalize_name(value)
    return {token for token in normalized.split() if token}
