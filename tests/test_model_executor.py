from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from notebooklm_graph_pipe.runtime.model_executor import (
    ExecutionPolicy,
    ModelExecutor,
    ModelRequest,
    ModelUsage,
)


@dataclass
class Adapter:
    provider: str = "fake"
    model: str = "fake-1"
    calls: int = 0

    def execute(self, request):
        self.calls += 1
        return '{"ok":true}', {"ok": True}, ModelUsage()


def _request(value: str = "hello") -> ModelRequest:
    return ModelRequest("answer", value, "system", {"type": "object"}, 100)


def test_executor_caches_only_successful_structured_results(tmp_path) -> None:
    adapter = Adapter()
    executor = ModelExecutor(
        {"fake": adapter},
        {"answer": "fake"},
        cache_path=tmp_path / "cache.sqlite3",
        metrics_path=tmp_path / "metrics.jsonl",
    )

    first = executor.execute_json(_request())
    second = executor.execute_json(_request())

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert adapter.calls == 1
    assert len((tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_executor_async_map_preserves_request_order(tmp_path) -> None:
    adapter = Adapter()
    executor = ModelExecutor(
        {"fake": adapter},
        {"answer": "fake"},
        policies={"answer": ExecutionPolicy(max_concurrency=2)},
        cache_path=tmp_path / "cache.sqlite3",
    )

    results = asyncio.run(executor.amap_json([_request("one"), _request("two")]))

    assert [result.payload for result in results] == [{"ok": True}, {"ok": True}]
    assert adapter.calls == 2


def test_executor_rejects_payload_that_does_not_match_schema(tmp_path) -> None:
    adapter = Adapter()
    executor = ModelExecutor(
        {"fake": adapter},
        {"answer": "fake"},
        policies={"answer": ExecutionPolicy(max_attempts=1)},
        cache_path=tmp_path / "cache.sqlite3",
    )
    request = ModelRequest(
        "answer",
        "hello",
        "system",
        {"type": "object", "required": ["answer"]},
        100,
    )

    with pytest.raises(ValueError, match="schema validation"):
        executor.execute_json(request)


def test_execution_policy_rejects_nonpositive_rate_limits() -> None:
    with pytest.raises(ValueError, match="requests_per_minute"):
        ExecutionPolicy(requests_per_minute=0)
    with pytest.raises(ValueError, match="tokens_per_minute"):
        ExecutionPolicy(tokens_per_minute=0)


def test_token_policy_rejects_single_request_larger_than_budget() -> None:
    executor = ModelExecutor(
        {"fake": Adapter()},
        {"answer": "fake"},
        policies={"answer": ExecutionPolicy(tokens_per_minute=1)},
    )

    with pytest.raises(ValueError, match="exceeds"):
        executor.execute_json(_request())


def test_model_fingerprint_includes_contract() -> None:
    executor = ModelExecutor({"fake": Adapter()}, {"answer": "fake"})

    assert executor.model_fingerprint("answer", {"version": 1}) == executor.model_fingerprint(
        "answer", {"version": 1}
    )
    assert executor.model_fingerprint("answer", {"version": 1}) != executor.model_fingerprint(
        "answer", {"version": 2}
    )
