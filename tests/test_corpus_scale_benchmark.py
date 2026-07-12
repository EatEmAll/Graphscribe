from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_corpus_scale.py"
SPEC = importlib.util.spec_from_file_location("benchmark_corpus_scale", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_percentile_interpolates() -> None:
    assert MODULE.percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    assert MODULE.percentile([1.0, 2.0, 3.0, 4.0], 95) == 3.85


def test_load_queries_accepts_evaluation_question_shape(tmp_path: Path) -> None:
    path = tmp_path / "questions.json"
    path.write_text('{"questions":[{"text":"First?"},{"text":"Second?"}]}', encoding="utf-8")
    assert MODULE.load_queries(path) == ["First?", "Second?"]


def test_run_benchmark_reports_latency_distribution(monkeypatch) -> None:
    clock = iter([0.0, 0.0, 0.01, 0.01, 0.03, 0.03])
    monkeypatch.setattr(MODULE.time, "perf_counter", lambda: next(clock))

    class Retriever:
        def search(self, request):
            return object()

    runtime = type("Runtime", (), {"retriever": Retriever()})()
    result = MODULE.run_benchmark(runtime, ["q"], modes=["hybrid"], warmups=0, iterations=2, top_k=12)

    assert result["hybrid"]["iterations"] == 2
    assert result["hybrid"]["latency_ms"]["p50"] == pytest.approx(15.0)
