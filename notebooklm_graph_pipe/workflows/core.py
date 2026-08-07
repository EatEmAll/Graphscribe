from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fingerprint(value: Any) -> str:
    canonical = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True)
class StepResult:
    outputs: dict[str, Any] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    @property
    def output_fingerprint(self) -> str:
        return _fingerprint({"outputs": self.outputs, "counters": self.counters})


@dataclass
class WorkflowContext:
    corpus_id: str | None
    run_dir: Path
    configuration: dict[str, Any]
    services: dict[str, Any] = field(default_factory=dict)
    inputs: dict[str, Any] = field(default_factory=dict)


class WorkflowStep(Protocol):
    name: str
    version: str

    def fingerprint(self, context: WorkflowContext) -> str: ...

    def run(self, context: WorkflowContext) -> StepResult: ...


@dataclass(frozen=True)
class WorkflowDefinition:
    name: str
    version: str
    steps: tuple[WorkflowStep, ...]


@dataclass
class WorkflowRun:
    run_id: str
    workflow_name: str
    workflow_version: str
    corpus_id: str | None
    status: str
    current_step: str | None
    input_fingerprint: str
    created_at: str
    updated_at: str
    error: str | None = None


class WorkflowEngine:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    def run(
        self,
        definition: WorkflowDefinition,
        context: WorkflowContext,
        *,
        run_id: str,
        resume: bool = False,
    ) -> WorkflowRun:
        run_dir = self.root / run_id
        run_path = run_dir / "run.json"
        input_fingerprint = _fingerprint(
            {"configuration": context.configuration, "inputs": context.inputs}
        )
        if resume:
            if not run_path.exists():
                raise ValueError(f"Workflow run does not exist: {run_id}")
            run = WorkflowRun(**json.loads(run_path.read_text(encoding="utf-8")))
            if (
                run.workflow_name != definition.name
                or run.workflow_version != definition.version
                or run.corpus_id != context.corpus_id
                or run.input_fingerprint != input_fingerprint
            ):
                raise ValueError("Workflow resume inputs do not match the original run.")
            run.status = "running"
            run.error = None
            run.updated_at = _now()
            _write_json(run_path, asdict(run))
        else:
            if run_path.exists():
                raise ValueError(f"Workflow run already exists: {run_id}")
            created = _now()
            run = WorkflowRun(
                run_id=run_id,
                workflow_name=definition.name,
                workflow_version=definition.version,
                corpus_id=context.corpus_id,
                status="running",
                current_step=None,
                input_fingerprint=input_fingerprint,
                created_at=created,
                updated_at=created,
            )
            _write_json(run_path, asdict(run))

        context.run_dir = run_dir
        previous_changed = False
        try:
            for step in definition.steps:
                step_path = run_dir / "steps" / f"{step.name}.json"
                step_fingerprint = step.fingerprint(context)
                prior = json.loads(step_path.read_text(encoding="utf-8")) if step_path.exists() else None
                reusable = bool(
                    resume
                    and not previous_changed
                    and prior
                    and prior.get("status") == "completed"
                    and prior.get("version") == step.version
                    and prior.get("input_fingerprint") == step_fingerprint
                )
                if reusable:
                    context.inputs[step.name] = prior.get("result") or {}
                    continue
                previous_changed = True
                run.current_step = step.name
                run.updated_at = _now()
                _write_json(run_path, asdict(run))
                result = step.run(context)
                context.inputs[step.name] = result.outputs
                _write_json(
                    step_path,
                    {
                        "name": step.name,
                        "version": step.version,
                        "status": "completed",
                        "input_fingerprint": step_fingerprint,
                        "output_fingerprint": result.output_fingerprint,
                        "result": result.outputs,
                        "counters": result.counters,
                        "warnings": list(result.warnings),
                        "completed_at": _now(),
                    },
                )
            run.status = "completed"
            run.current_step = None
            run.error = None
        except Exception as exc:
            run.status = "failed"
            run.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            run.updated_at = _now()
            _write_json(run_path, asdict(run))
        return run
