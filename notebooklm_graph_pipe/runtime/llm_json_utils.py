from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from google import genai
from google.genai import types
from openai import OpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
CLI_JSON_SCHEMA: dict[str, Any] = {"type": "object", "additionalProperties": True}
CODEX_CLI_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"payload": {"type": "string"}},
    "required": ["payload"],
    "additionalProperties": False,
}
DEFAULT_CLI_TIMEOUT_SECONDS = 120.0


@dataclass(frozen=True)
class SubscriptionCliClient:
    name: str
    executable: str
    timeout_seconds: float = DEFAULT_CLI_TIMEOUT_SECONDS


@dataclass(frozen=True)
class CliResponse:
    output_text: str


def extract_response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str):
        return output_text.strip()
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
    client_name: str = "genai",
    model_name: str,
    prompt: str,
    system_instruction: str,
    max_output_tokens: int,
    temperature: float = 0.0,
    reasoning_effort: str | None = None,
    max_attempts: int = 1,
    retry_sleep_seconds: float = 0.0,
    retry_on_exception: Callable[[Exception], bool] | None = None,
) -> tuple[dict[str, Any] | None, str]:
    last_error = "Invalid or empty JSON response"
    attempts = max(max_attempts, 1)
    for attempt in range(1, attempts + 1):
        try:
            response = _generate_structured_response(
                client,
                client_name=client_name,
                model_name=model_name,
                prompt=prompt,
                system_instruction=system_instruction,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
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


def build_openai_compatible_client(*, api_key: str, base_url: str | None = None) -> OpenAI:
    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def build_single_prompt_clients(*client_names: str) -> dict[str, Any]:
    clients: dict[str, Any] = {}
    for raw_client_name in client_names:
        client_name = raw_client_name.strip().lower()
        if not client_name or client_name in clients:
            continue
        if client_name == "genai":
            api_key = os.environ.get("GOOGLE_API_KEY", "")
            if not api_key:
                raise RuntimeError("Set GOOGLE_API_KEY environment variable.")
            clients[client_name] = genai.Client(api_key=api_key)
            continue
        if client_name == "openai":
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                raise RuntimeError("Set OPENAI_API_KEY environment variable.")
            clients[client_name] = build_openai_compatible_client(api_key=api_key)
            continue
        if client_name == "openrouter":
            api_key = os.environ.get("OPENROUTER_API_KEY", "")
            if not api_key:
                raise RuntimeError("Set OPENROUTER_API_KEY environment variable.")
            clients[client_name] = build_openai_compatible_client(api_key=api_key, base_url=OPENROUTER_BASE_URL)
            continue
        if client_name in {"codex", "claude"}:
            executable = shutil.which(client_name)
            if not executable:
                raise RuntimeError(f"Install the {client_name} CLI and ensure it is available on PATH.")
            timeout_raw = os.environ.get("LLM_CLI_TIMEOUT_SECONDS", str(DEFAULT_CLI_TIMEOUT_SECONDS))
            try:
                timeout_seconds = float(timeout_raw)
            except ValueError as exc:
                raise RuntimeError("LLM_CLI_TIMEOUT_SECONDS must be a number.") from exc
            if timeout_seconds <= 0:
                raise RuntimeError("LLM_CLI_TIMEOUT_SECONDS must be positive.")
            clients[client_name] = SubscriptionCliClient(
                name=client_name,
                executable=executable,
                timeout_seconds=timeout_seconds,
            )
            continue
        raise ValueError(f"Unsupported single-prompt client '{raw_client_name}'.")
    return clients


def _generate_structured_response(
    client: Any,
    *,
    client_name: str,
    model_name: str,
    prompt: str,
    system_instruction: str,
    max_output_tokens: int,
    temperature: float,
    reasoning_effort: str | None = None,
) -> Any:
    normalized_client = client_name.strip().lower()
    if normalized_client == "genai":
        return client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                response_mime_type="application/json",
            ),
        )
    if normalized_client in {"openai", "openrouter"}:
        return client.responses.create(
            model=model_name,
            input=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt},
            ],
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
    if normalized_client in {"codex", "claude"}:
        if not isinstance(client, SubscriptionCliClient) or client.name != normalized_client:
            raise TypeError(f"Expected a {normalized_client} subscription CLI client.")
        return _generate_cli_response(
            client,
            model_name=model_name,
            prompt=prompt,
            system_instruction=system_instruction,
            reasoning_effort=reasoning_effort,
        )
    raise ValueError(f"Unsupported single-prompt client '{client_name}'.")


def _generate_cli_response(
    client: SubscriptionCliClient,
    *,
    model_name: str,
    prompt: str,
    system_instruction: str,
    reasoning_effort: str | None,
) -> CliResponse:
    if client.name == "codex":
        response_instruction = (
            "Return only one object matching the supplied schema. Serialize the requested JSON object "
            "as a JSON string in the payload field."
        )
        schema = CODEX_CLI_JSON_SCHEMA
    else:
        response_instruction = "Return only one JSON object that satisfies the supplied JSON schema."
        schema = CLI_JSON_SCHEMA
    combined_prompt = f"{system_instruction.strip()}\n\n{response_instruction}\n\n{prompt.strip()}"
    schema_json = json.dumps(schema, separators=(",", ":"))

    with tempfile.TemporaryDirectory(prefix=f"llm-{client.name}-") as temp_dir:
        temp_path = Path(temp_dir)
        if client.name == "codex":
            schema_path = temp_path / "output-schema.json"
            output_path = temp_path / "response.json"
            schema_path.write_text(schema_json, encoding="utf-8")
            args = [
                client.executable,
                "--ask-for-approval",
                "never",
                "--sandbox",
                "read-only",
                "exec",
                "--ephemeral",
                "--skip-git-repo-check",
                "--color",
                "never",
                "--output-schema",
                str(schema_path),
                "--output-last-message",
                str(output_path),
                "--model",
                model_name,
            ]
            if reasoning_effort:
                args.extend(["--config", f'model_reasoning_effort="{reasoning_effort}"'])
            args.append("-")
            _run_cli(args, prompt=combined_prompt, cwd=temp_path, timeout_seconds=client.timeout_seconds)
            if not output_path.exists():
                raise RuntimeError("Codex CLI did not write its final response.")
            output_text = output_path.read_text(encoding="utf-8").strip()
            try:
                envelope = json.loads(output_text)
            except json.JSONDecodeError as exc:
                raise RuntimeError("Codex CLI returned an invalid JSON envelope.") from exc
            payload = envelope.get("payload") if isinstance(envelope, dict) else None
            if not isinstance(payload, str):
                raise RuntimeError("Codex CLI response did not contain a JSON payload string.")
            return CliResponse(output_text=payload.strip())

        args = [
            client.executable,
            "--print",
            "--no-session-persistence",
            "--permission-mode",
            "dontAsk",
            "--tools",
            "",
            "--output-format",
            "json",
            "--json-schema",
            schema_json,
            "--model",
            model_name,
        ]
        if reasoning_effort:
            args.extend(["--effort", reasoning_effort])
        completed = _run_cli(args, prompt=combined_prompt, cwd=temp_path, timeout_seconds=client.timeout_seconds)
        try:
            envelope = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Claude CLI returned an invalid JSON envelope.") from exc
        structured_output = envelope.get("structured_output") if isinstance(envelope, dict) else None
        if isinstance(structured_output, dict):
            return CliResponse(output_text=json.dumps(structured_output, ensure_ascii=True))
        result = envelope.get("result") if isinstance(envelope, dict) else None
        if isinstance(result, str):
            return CliResponse(output_text=result.strip())
        raise RuntimeError("Claude CLI response did not contain structured output.")


def _run_cli(
    args: list[str],
    *,
    prompt: str,
    cwd: Path,
    timeout_seconds: float,
) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            args,
            input=prompt,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"{Path(args[0]).stem} CLI timed out after {timeout_seconds:g} seconds.") from exc
    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout).strip()
        if len(details) > 500:
            details = details[-500:]
        suffix = f": {details}" if details else ""
        raise RuntimeError(f"{Path(args[0]).stem} CLI exited with code {completed.returncode}{suffix}")
    return completed
