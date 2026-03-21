from __future__ import annotations

import asyncio
from types import SimpleNamespace

import src.adaptive_retry as adaptive_retry
from src.shared.llm_graph_builder_exception import LLMGraphBuilderException


def test_is_output_limit_error_detects_known_markers() -> None:
    assert adaptive_retry.is_output_limit_error(
        "Graph transformation failed: Could not parse response content as the length limit was reached"
    )
    assert adaptive_retry.is_output_limit_error("finish_reason:length")
    assert not adaptive_retry.is_output_limit_error("temporary network timeout")


def test_build_adaptive_retry_plan_uses_expected_ladder() -> None:
    params = SimpleNamespace(token_chunk_size=4000, chunk_overlap=400, chunks_to_combine=2)

    plan = adaptive_retry.build_adaptive_retry_plan(params)

    assert [attempt["name"] for attempt in plan] == [
        "original",
        "low-verbosity-same-chunks",
        "rechunk-2000",
        "rechunk-1000",
        "rechunk-500",
    ]
    assert plan[0]["token_chunk_size"] == 4000
    assert plan[0]["chunk_overlap"] == 400
    assert plan[0]["chunks_to_combine"] == 2
    assert plan[1]["chunks_to_combine"] == 1
    assert plan[1]["low_verbosity"] is True
    assert plan[2]["rebuild_chunks"] is True
    assert plan[2]["token_chunk_size"] == 2000
    assert plan[3]["token_chunk_size"] == 1000
    assert plan[4]["token_chunk_size"] == 500


def test_resolve_graph_transformer_settings_low_verbosity_disables_descriptions() -> None:
    settings = adaptive_retry.resolve_graph_transformer_settings(
        supports_structured_output=True,
        is_groq=False,
        low_verbosity=True,
    )

    assert settings["mode"] == "low_verbosity"
    assert settings["node_properties"] is False
    assert settings["relationship_properties"] is False
    assert settings["ignore_tool_usage"] is True


def test_run_with_adaptive_output_limit_fallback_retries_then_succeeds() -> None:
    params = SimpleNamespace(
        file_name="demo.txt",
        token_chunk_size=4000,
        chunk_overlap=400,
        chunks_to_combine=2,
    )
    attempts: list[str] = []
    cleanup_calls: list[str] = []

    async def runner(attempt: dict):
        attempts.append(attempt["name"])
        if len(attempts) < 3:
            raise LLMGraphBuilderException(
                "Graph transformation failed: Could not parse response content as the length limit was reached"
            )
        return "ok"

    def cleanup() -> None:
        cleanup_calls.append("cleanup")

    result = asyncio.run(adaptive_retry.run_with_adaptive_output_limit_fallback(params, runner, cleanup))

    assert result == "ok"
    assert attempts == ["original", "low-verbosity-same-chunks", "rechunk-2000"]
    assert cleanup_calls == ["cleanup", "cleanup"]


def test_run_with_adaptive_output_limit_fallback_does_not_retry_non_output_errors() -> None:
    params = SimpleNamespace(
        file_name="demo.txt",
        token_chunk_size=4000,
        chunk_overlap=400,
        chunks_to_combine=2,
    )
    cleanup_calls: list[str] = []

    async def runner(_attempt: dict):
        raise LLMGraphBuilderException("temporary network timeout")

    def cleanup() -> None:
        cleanup_calls.append("cleanup")

    try:
        asyncio.run(adaptive_retry.run_with_adaptive_output_limit_fallback(params, runner, cleanup))
    except LLMGraphBuilderException as exc:
        assert str(exc) == "temporary network timeout"
    else:
        raise AssertionError("Expected LLMGraphBuilderException")

    assert cleanup_calls == []

def test_run_chunk_batch_with_output_limit_fallback_retries_current_batch() -> None:
    params = SimpleNamespace(
        file_name="demo.txt",
        token_chunk_size=4000,
        chunk_overlap=400,
        chunks_to_combine=2,
    )
    attempts: list[tuple[int, bool]] = []

    async def runner(attempt: dict):
        attempts.append((attempt["chunks_to_combine"], attempt["low_verbosity"]))
        if len(attempts) == 1:
            raise LLMGraphBuilderException(
                "Graph transformation failed: Could not parse response content as the length limit was reached"
            )
        return "ok"

    result = asyncio.run(
        adaptive_retry.run_chunk_batch_with_output_limit_fallback(
            params,
            adaptive_retry.build_adaptive_retry_plan(params)[0],
            runner,
            file_name="demo.txt",
            chunk_start_index=0,
            chunk_end_index=20,
        )
    )

    assert result == "ok"
    assert attempts == [(2, False), (1, True)]


def test_run_with_adaptive_output_limit_fallback_can_resume_at_rechunk_stage() -> None:
    params = SimpleNamespace(
        file_name="demo.txt",
        token_chunk_size=4000,
        chunk_overlap=400,
        chunks_to_combine=2,
    )
    attempts: list[str] = []
    cleanup_calls: list[str] = []

    async def runner(attempt: dict):
        attempts.append(attempt["name"])
        if len(attempts) == 1:
            raise LLMGraphBuilderException(
                "Graph transformation failed: Could not parse response content as the length limit was reached"
            )
        return "ok"

    def cleanup() -> None:
        cleanup_calls.append("cleanup")

    result = asyncio.run(
        adaptive_retry.run_with_adaptive_output_limit_fallback(
            params,
            runner,
            cleanup,
            start_index=2,
            initial_error_message="Graph transformation failed: Could not parse response content as the length limit was reached",
        )
    )

    assert result == "ok"
    assert attempts == ["rechunk-2000", "rechunk-1000"]
    assert cleanup_calls == ["cleanup"]
