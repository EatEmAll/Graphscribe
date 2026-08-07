from __future__ import annotations

import asyncio
import hashlib
import json
import random
import sqlite3
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from jsonschema import ValidationError, validate


@dataclass(frozen=True)
class ModelRequest:
    role: str
    prompt: str
    system_instruction: str
    response_schema: dict[str, object] | None
    max_output_tokens: int
    temperature: float = 0.0
    cache_namespace: str = "default"
    idempotency_key: str | None = None


@dataclass(frozen=True)
class ModelUsage:
    requests: int = 1
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    latency_ms: int = 0


@dataclass(frozen=True)
class ModelResult:
    text: str
    payload: dict[str, object] | None
    provider: str
    model: str
    attempts: int
    cache_hit: bool
    usage: ModelUsage


@dataclass(frozen=True)
class ExecutionPolicy:
    max_concurrency: int = 4
    max_attempts: int = 2
    timeout_seconds: float = 120.0
    requests_per_minute: int | None = None
    tokens_per_minute: int | None = None
    cache_enabled: bool = True

    def __post_init__(self) -> None:
        if self.max_concurrency <= 0 or self.max_attempts <= 0 or self.timeout_seconds <= 0:
            raise ValueError("Model execution limits must be positive.")
        if self.requests_per_minute is not None and self.requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive when configured.")
        if self.tokens_per_minute is not None and self.tokens_per_minute <= 0:
            raise ValueError("tokens_per_minute must be positive when configured.")


class ModelAdapter(Protocol):
    provider: str
    model: str

    def execute(self, request: ModelRequest) -> tuple[str, dict[str, object] | None, ModelUsage]: ...


class SQLiteModelCache:
    def __init__(self, path: str | Path | None):
        self.path = Path(path).resolve() if path else None
        self._lock = threading.Lock()
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as connection:
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS model_cache ("
                    "cache_key TEXT PRIMARY KEY, result_json TEXT NOT NULL, created_at REAL NOT NULL)"
                )

    def _connect(self) -> sqlite3.Connection:
        if self.path is None:
            raise RuntimeError("Model cache is disabled.")
        return sqlite3.connect(self.path, timeout=30)

    def get(self, cache_key: str) -> ModelResult | None:
        if self.path is None:
            return None
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT result_json FROM model_cache WHERE cache_key = ?", (cache_key,)
            ).fetchone()
        if row is None:
            return None
        payload = json.loads(row[0])
        payload["usage"] = ModelUsage(**payload["usage"])
        payload["cache_hit"] = True
        return ModelResult(**payload)

    def set(self, cache_key: str, result: ModelResult) -> None:
        if self.path is None or result.payload is None:
            return
        stored = asdict(result)
        stored["cache_hit"] = False
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO model_cache(cache_key, result_json, created_at) VALUES (?, ?, ?)",
                (cache_key, json.dumps(stored, ensure_ascii=True, sort_keys=True), time.time()),
            )


class JsonlMetricsSink:
    def __init__(self, path: str | Path | None):
        self.path = Path(path).resolve() if path else None
        self._lock = threading.Lock()

    def record(self, request: ModelRequest, result: ModelResult) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": time.time(),
            "role": request.role,
            "provider": result.provider,
            "model": result.model,
            "attempts": result.attempts,
            "cache_hit": result.cache_hit,
            **asdict(result.usage),
        }
        with self._lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


class ModelExecutor:
    def __init__(
        self,
        adapters: dict[str, ModelAdapter],
        role_adapters: dict[str, str],
        *,
        policies: dict[str, ExecutionPolicy] | None = None,
        cache_path: str | Path | None = None,
        metrics_path: str | Path | None = None,
        transient_error: Callable[[Exception], bool] | None = None,
    ):
        self.adapters = dict(adapters)
        self.role_adapters = dict(role_adapters)
        self.policies = dict(policies or {})
        self.cache = SQLiteModelCache(cache_path)
        self.metrics = JsonlMetricsSink(metrics_path)
        self.transient_error = transient_error or (lambda exc: True)
        self._semaphores: dict[tuple[str, int], asyncio.Semaphore] = {}
        self._rate_lock = threading.Lock()
        self._request_windows: dict[str, deque[float]] = defaultdict(deque)
        self._token_windows: dict[str, deque[tuple[float, int]]] = defaultdict(deque)

    @staticmethod
    def cache_key(request: ModelRequest, adapter: ModelAdapter) -> str:
        canonical = json.dumps(
            {
                "provider": adapter.provider,
                "model": adapter.model,
                **asdict(request),
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _resolve(self, role: str) -> tuple[ModelAdapter, ExecutionPolicy]:
        try:
            adapter = self.adapters[self.role_adapters[role]]
        except KeyError as exc:
            raise ValueError(f"No model adapter is configured for role '{role}'.") from exc
        return adapter, self.policies.get(role, ExecutionPolicy())

    def model_fingerprint(self, role: str, contract: object | None = None) -> str:
        adapter, _ = self._resolve(role)
        canonical = json.dumps(
            {"provider": adapter.provider, "model": adapter.model, "contract": contract},
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def execute_json(self, request: ModelRequest) -> ModelResult:
        adapter, policy = self._resolve(request.role)
        key = self.cache_key(request, adapter)
        if policy.cache_enabled:
            cached = self.cache.get(key)
            if cached is not None:
                self.metrics.record(request, cached)
                return cached
        started = time.monotonic()
        last_error: Exception | None = None
        for attempt in range(1, policy.max_attempts + 1):
            try:
                self._wait_for_capacity(request.role, request, policy)
                text, payload, usage = adapter.execute(request)
                if request.response_schema is not None and not isinstance(payload, dict):
                    raise ValueError("Model response did not contain a JSON object.")
                if request.response_schema is not None:
                    try:
                        validate(payload, request.response_schema)
                    except ValidationError as exc:
                        raise ValueError(f"Model response failed schema validation: {exc.message}") from exc
                elapsed = int((time.monotonic() - started) * 1000)
                result = ModelResult(
                    text=text,
                    payload=payload,
                    provider=adapter.provider,
                    model=adapter.model,
                    attempts=attempt,
                    cache_hit=False,
                    usage=ModelUsage(
                        requests=usage.requests,
                        input_tokens=usage.input_tokens,
                        output_tokens=usage.output_tokens,
                        cost_usd=usage.cost_usd,
                        latency_ms=elapsed,
                    ),
                )
                if policy.cache_enabled:
                    self.cache.set(key, result)
                self.metrics.record(request, result)
                return result
            except Exception as exc:
                last_error = exc
                if attempt >= policy.max_attempts or not self.transient_error(exc):
                    raise
                delay = min(5.0, 0.25 * (2 ** (attempt - 1))) + random.random() * 0.1
                time.sleep(delay)
        raise RuntimeError("Model execution failed without an error.") from last_error

    def _wait_for_capacity(
        self,
        role: str,
        request: ModelRequest,
        policy: ExecutionPolicy,
    ) -> None:
        if policy.requests_per_minute is None and policy.tokens_per_minute is None:
            return
        estimated_tokens = max(1, (len(request.prompt) + len(request.system_instruction)) // 4)
        if policy.tokens_per_minute is not None and estimated_tokens > policy.tokens_per_minute:
            raise ValueError(
                f"One request for role '{role}' exceeds its tokens_per_minute policy."
            )
        while True:
            now = time.monotonic()
            wait_seconds = 0.0
            with self._rate_lock:
                request_window = self._request_windows[role]
                token_window = self._token_windows[role]
                while request_window and now - request_window[0] >= 60.0:
                    request_window.popleft()
                while token_window and now - token_window[0][0] >= 60.0:
                    token_window.popleft()
                if policy.requests_per_minute is not None and len(request_window) >= policy.requests_per_minute:
                    wait_seconds = max(wait_seconds, 60.0 - (now - request_window[0]))
                token_total = sum(value for _, value in token_window)
                if policy.tokens_per_minute is not None and token_total + estimated_tokens > policy.tokens_per_minute:
                    wait_seconds = max(wait_seconds, 60.0 - (now - token_window[0][0]))
                if wait_seconds <= 0:
                    request_window.append(now)
                    token_window.append((now, estimated_tokens))
                    return
            time.sleep(min(wait_seconds, 1.0))

    async def aexecute_json(self, request: ModelRequest) -> ModelResult:
        _, policy = self._resolve(request.role)
        loop_key = (request.role, id(asyncio.get_running_loop()))
        semaphore = self._semaphores.setdefault(loop_key, asyncio.Semaphore(policy.max_concurrency))
        async with semaphore:
            return await asyncio.wait_for(
                asyncio.to_thread(self.execute_json, request), timeout=policy.timeout_seconds
            )

    async def amap_json(
        self,
        requests: Sequence[ModelRequest],
        *,
        max_concurrency: int | None = None,
    ) -> list[ModelResult]:
        if not requests:
            return []
        if max_concurrency is not None and max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive.")
        local = asyncio.Semaphore(max_concurrency or len(requests))

        async def execute(request: ModelRequest) -> ModelResult:
            async with local:
                return await self.aexecute_json(request)

        return list(await asyncio.gather(*(execute(request) for request in requests)))
