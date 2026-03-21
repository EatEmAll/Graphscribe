from __future__ import annotations

import logging

from src.shared.llm_graph_builder_exception import LLMGraphBuilderException

OUTPUT_LIMIT_ERROR_MARKERS = (
    "length limit was reached",
    "could not parse response content",
    "maximum context length",
    "maximum output tokens",
    "max output tokens",
    "response was truncated",
    "finish_reason:length",
)

ADAPTIVE_RETRY_FALLBACKS = (
    {"name": "original", "low_verbosity": False, "rebuild_chunks": False, "chunks_to_combine": None},
    {"name": "low-verbosity-same-chunks", "low_verbosity": True, "rebuild_chunks": False, "chunks_to_combine": 1},
    {"name": "rechunk-2000", "low_verbosity": True, "rebuild_chunks": True, "token_chunk_size": 2000, "chunk_overlap": 200, "chunks_to_combine": 1},
    {"name": "rechunk-1000", "low_verbosity": True, "rebuild_chunks": True, "token_chunk_size": 1000, "chunk_overlap": 100, "chunks_to_combine": 1},
    {"name": "rechunk-500", "low_verbosity": True, "rebuild_chunks": True, "token_chunk_size": 500, "chunk_overlap": 50, "chunks_to_combine": 1},
)


def is_output_limit_error(message: str | None) -> bool:
    """Return True when an LLM error indicates the model hit an output-size limit."""
    normalized = (message or "").strip().lower()
    return any(marker in normalized for marker in OUTPUT_LIMIT_ERROR_MARKERS)


def resolve_graph_transformer_settings(
    *,
    supports_structured_output: bool,
    is_groq: bool,
    low_verbosity: bool,
) -> dict[str, object]:
    if low_verbosity:
        return {
            "mode": "low_verbosity",
            "node_properties": False,
            "relationship_properties": False,
            "ignore_tool_usage": True,
        }
    if supports_structured_output and not is_groq:
        return {
            "mode": "structured",
            "node_properties": ["description"],
            "relationship_properties": ["description"],
            "ignore_tool_usage": False,
        }
    return {
        "mode": "unstructured",
        "node_properties": False,
        "relationship_properties": False,
        "ignore_tool_usage": True,
    }


def build_adaptive_retry_plan(params) -> list[dict]:
    default_token_chunk_size = params.token_chunk_size or 2000
    default_chunk_overlap = params.chunk_overlap or 200
    default_chunks_to_combine = params.chunks_to_combine or 1
    plan: list[dict] = []
    for fallback in ADAPTIVE_RETRY_FALLBACKS:
        plan.append(
            {
                "name": fallback["name"],
                "token_chunk_size": fallback.get("token_chunk_size", default_token_chunk_size),
                "chunk_overlap": fallback.get("chunk_overlap", default_chunk_overlap),
                "chunks_to_combine": fallback.get("chunks_to_combine") or default_chunks_to_combine,
                "low_verbosity": fallback["low_verbosity"],
                "rebuild_chunks": fallback["rebuild_chunks"],
            }
        )
    return plan


def describe_adaptive_retry_attempt(attempt: dict) -> str:
    return (
        f"{attempt['name']} "
        f"(token_chunk_size={attempt['token_chunk_size']}, "
        f"chunk_overlap={attempt['chunk_overlap']}, "
        f"chunks_to_combine={attempt['chunks_to_combine']}, "
        f"low_verbosity={attempt['low_verbosity']}, "
        f"rebuild_chunks={attempt['rebuild_chunks']})"
    )


async def run_chunk_batch_with_output_limit_fallback(
    params,
    attempt: dict,
    chunk_runner,
    *,
    file_name: str,
    chunk_start_index: int,
    chunk_end_index: int,
):
    try:
        return await chunk_runner(attempt)
    except LLMGraphBuilderException as exc:
        if attempt["low_verbosity"] or not is_output_limit_error(str(exc)):
            raise

        fallback_attempt = build_adaptive_retry_plan(params)[1]
        logging.warning(
            "Adaptive extraction fallback for %s chunk batch %s-%s after output-limit failure. "
            "reason=%s next_attempt=%s",
            file_name,
            chunk_start_index + 1,
            chunk_end_index,
            str(exc),
            describe_adaptive_retry_attempt(fallback_attempt),
        )
        return await chunk_runner(fallback_attempt)


async def run_with_adaptive_output_limit_fallback(
    params,
    attempt_runner,
    cleanup_runner,
    *,
    start_index: int = 0,
    initial_error_message: str | None = None,
):
    plan = build_adaptive_retry_plan(params)
    if start_index < 0 or start_index >= len(plan):
        raise ValueError(f"Invalid adaptive retry start index: {start_index}")

    first_error_message: str | None = initial_error_message
    for attempt_index in range(start_index, len(plan)):
        attempt = plan[attempt_index]
        try:
            result = await attempt_runner(attempt)
            if attempt_index > 0:
                logging.info(
                    "Adaptive extraction retry succeeded for %s using %s",
                    params.file_name,
                    describe_adaptive_retry_attempt(attempt),
                )
            return result
        except LLMGraphBuilderException as exc:
            message = str(exc)
            if not is_output_limit_error(message) or attempt_index == len(plan) - 1:
                raise
            first_error_message = first_error_message or message
            logging.warning(
                "Adaptive extraction fallback for %s after output-limit failure. "
                "reason=%s next_attempt=%s",
                params.file_name,
                first_error_message,
                describe_adaptive_retry_attempt(plan[attempt_index + 1]),
            )
            cleanup_runner()
