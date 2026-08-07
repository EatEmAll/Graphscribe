from __future__ import annotations

import json

import pytest

from notebooklm_graph_pipe.workflows import (
    StepResult,
    WorkflowContext,
    WorkflowDefinition,
    WorkflowEngine,
)


class Step:
    version = "1"

    def __init__(self, name, calls):
        self.name = name
        self.calls = calls

    def fingerprint(self, context):
        return f"{self.name}:{context.configuration['value']}"

    def run(self, context):
        self.calls.append(self.name)
        return StepResult(outputs={"name": self.name})


def test_workflow_resume_reuses_matching_steps(tmp_path) -> None:
    calls = []
    definition = WorkflowDefinition("demo", "1", (Step("one", calls), Step("two", calls)))
    engine = WorkflowEngine(tmp_path)
    context = WorkflowContext("corpus", tmp_path / "unused", {"value": 1})

    first = engine.run(definition, context, run_id="run-1")
    resumed = engine.run(
        definition,
        WorkflowContext("corpus", tmp_path / "unused", {"value": 1}),
        run_id="run-1",
        resume=True,
    )

    assert first.status == resumed.status == "completed"
    assert calls == ["one", "two"]
    assert json.loads((tmp_path / "run-1" / "run.json").read_text(encoding="utf-8"))["status"] == "completed"


def test_workflow_resume_rejects_changed_inputs(tmp_path) -> None:
    definition = WorkflowDefinition("demo", "1", (Step("one", []),))
    engine = WorkflowEngine(tmp_path)
    engine.run(
        definition,
        WorkflowContext("corpus", tmp_path / "unused", {"value": 1}),
        run_id="run-1",
    )

    with pytest.raises(ValueError, match="do not match"):
        engine.run(
            definition,
            WorkflowContext("corpus", tmp_path / "unused", {"value": 2}),
            run_id="run-1",
            resume=True,
        )
