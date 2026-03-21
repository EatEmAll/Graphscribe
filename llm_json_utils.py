from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Callable

from google.genai import types


def extract_response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    return text.strip() if isinstance(text, str) else ""


def parse_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            payload = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return None
    return payload if isinstance(payload, dict) else None


def stable_json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def make_cache_key(*, namespace: str, payload: Any) -> str:
    digest = hashlib.sha256()
    digest.update(namespace.encode("utf-8"))
    digest.update(b":")
    digest.update(stable_json_dumps(payload).encode("utf-8"))
    return digest.hexdigest()


class JsonDiskCache:
    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path else None
        self.records: dict[str, Any] = {}
        self.hits = 0
        self._dirty = False
        self._load()

    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"  Warning: failed to load cache '{self.path}': {exc}")
            return

        if isinstance(payload, dict) and isinstance(payload.get("records"), dict):
            self.records = payload["records"]
            return
        if isinstance(payload, dict):
            self.records = payload
            return
        print(f"  Warning: cache '{self.path}' did not contain a JSON object.")

    def get(self, key: str) -> Any | None:
        value = self.records.get(key)
        if value is not None:
            self.hits += 1
        return value

    def set(self, key: str, value: Any) -> None:
        if self.records.get(key) == value:
            return
        self.records[key] = value
        self._dirty = True

    def save(self) -> None:
        if self.path is None or not self._dirty:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps({"records": self.records}, ensure_ascii=True, sort_keys=True, indent=2),
                encoding="utf-8",
            )
            self._dirty = False
        except Exception as exc:
            print(f"  Warning: failed to save cache '{self.path}': {exc}")


def is_transient_model_error(exc: Exception) -> bool:
    message = str(exc).lower()
    transient_markers = (
        "getaddrinfo failed",
        "temporarily unavailable",
        "connection reset",
        "connection aborted",
        "connection refused",
        "deadline",
        "timed out",
        "timeout",
        "429",
        "503",
        "rate limit",
        "service unavailable",
        "unavailable",
        "socket",
    )
    return any(marker in message for marker in transient_markers)


def generate_json_payload(
    client: Any,
    *,
    model_name: str,
    prompt: str,
    system_instruction: str,
    max_output_tokens: int,
    temperature: float = 0.0,
    max_attempts: int = 1,
    retry_sleep_seconds: float = 0.0,
    retry_on_exception: Callable[[Exception], bool] | None = None,
) -> tuple[dict[str, Any] | None, str]:
    last_error = "Invalid or empty JSON response"
    attempts = max(max_attempts, 1)
    for attempt in range(1, attempts + 1):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                    response_mime_type="application/json",
                ),
            )
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < attempts and (retry_on_exception is None or retry_on_exception(exc)):
                if retry_sleep_seconds > 0:
                    time.sleep(retry_sleep_seconds * attempt)
                continue
            return None, f"Model request failed after retries: {last_error}"

        payload = parse_json_object(extract_response_text(response))
        if payload is not None:
            return payload, ""

        if attempt < attempts and retry_sleep_seconds > 0:
            time.sleep(retry_sleep_seconds * attempt)

    return None, last_error
